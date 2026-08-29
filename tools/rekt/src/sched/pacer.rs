use std::time::Duration;

/// open-loop arrival pacing. `due(now)` reports how many arrivals have come due
/// since the last call. the contract that matters: the schedule is a function
/// of absolute time, so a late poll (slow target) yields a catch-up rather than
/// a slipped grid. that is what keeps offered rate honest under load.
pub trait Pacer {
    fn due(&mut self, now: Duration) -> u64;
}

/// arrivals sit on the absolute grid `k / rate`; emitted-to-date is
/// `floor(now / interval)`, so the long-run offered rate holds no matter how
/// far behind a poll arrives.
#[derive(Debug, Clone)]
pub struct GridPacer {
    interval_nanos: u64,
    emitted: u64,
}

impl GridPacer {
    #[must_use]
    pub fn new(rate_per_sec: f64) -> Self {
        debug_assert!(!rate_per_sec.is_nan(), "rate_per_sec must not be NaN");
        // 0 means no load; clamp interval to >=1ns so the divide can never trap
        let interval_nanos = if rate_per_sec > 0.0 { ((1e9 / rate_per_sec) as u64).max(1) } else { 0 };
        Self { interval_nanos, emitted: 0 }
    }
}

impl Pacer for GridPacer {
    fn due(&mut self, now: Duration) -> u64 {
        if self.interval_nanos == 0 {
            return 0;
        }
        let now_nanos = now.as_nanos().min(u128::from(u64::MAX)) as u64;
        let target = now_nanos / self.interval_nanos;
        let due = target.saturating_sub(self.emitted);
        // max keeps the count monotonic if a caller ever passes a backward `now`
        self.emitted = self.emitted.max(target);
        due
    }
}

/// the naive "sleep one interval between fires" pacer. it sets the next deadline
/// from the actual poll time, so a late poll pushes every later arrival further
/// out and the offered rate quietly decays under load. kept as the benched
/// baseline [`GridPacer`] has to beat — its decay is coordinated omission.
#[derive(Debug, Clone)]
pub struct IntervalPacer {
    interval: Duration,
    next: Duration,
    started: bool,
}

impl IntervalPacer {
    #[must_use]
    pub fn new(rate_per_sec: f64) -> Self {
        debug_assert!(!rate_per_sec.is_nan(), "rate_per_sec must not be NaN");
        let interval = if rate_per_sec > 0.0 {
            let secs = 1.0 / rate_per_sec;
            if secs.is_finite() && secs < 1e12 { Duration::from_secs_f64(secs) } else { Duration::MAX }
        } else {
            Duration::MAX
        };
        Self {
            interval,
            next: Duration::ZERO,
            started: false,
        }
    }
}

impl Pacer for IntervalPacer {
    fn due(&mut self, now: Duration) -> u64 {
        if !self.started {
            self.started = true;
            self.next = now.saturating_add(self.interval);
            return 1;
        }
        if now >= self.next {
            self.next = now.saturating_add(self.interval);
            1
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    // tests assert on known values; unwrap/expect are the clearer failure here
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    fn offered_on_time(pacer: &mut dyn Pacer, window: Duration, step: Duration) -> u64 {
        let mut now = Duration::ZERO;
        let mut total = 0;
        while now <= window {
            total += pacer.due(now);
            now += step;
        }
        total
    }

    fn offered_with_stall(pacer: &mut dyn Pacer, window: Duration, step: Duration, stall: Duration) -> u64 {
        let mut now = Duration::ZERO;
        let mut total = 0;
        let half = window / 2;
        let mut stalled = false;
        while now <= window {
            total += pacer.due(now);
            if !stalled && now >= half {
                stalled = true;
                now += stall;
            } else {
                now += step;
            }
        }
        total
    }

    #[proxima::test]
    async fn grid_holds_rate_on_time() {
        let mut pacer = GridPacer::new(1000.0);
        let total = offered_on_time(&mut pacer, Duration::from_secs(2), Duration::from_micros(500));
        assert_eq!(total, 2000);
    }

    #[proxima::test]
    async fn grid_holds_rate_under_stall() {
        let mut pacer = GridPacer::new(1000.0);
        let total = offered_with_stall(&mut pacer, Duration::from_secs(2), Duration::from_micros(500), Duration::from_millis(800));
        assert_eq!(total, 2000);
    }

    #[proxima::test]
    async fn grid_ignores_backward_time() {
        let mut pacer = GridPacer::new(1000.0);
        assert_eq!(pacer.due(Duration::from_secs(1)), 1000);
        assert_eq!(pacer.due(Duration::from_millis(500)), 0);
        assert_eq!(pacer.due(Duration::from_secs(1)), 0);
        assert_eq!(pacer.due(Duration::from_millis(1500)), 500);
    }

    #[proxima::test]
    async fn interval_drifts_under_stall() {
        let mut grid = GridPacer::new(1000.0);
        let mut naive = IntervalPacer::new(1000.0);
        let grid_total = offered_with_stall(&mut grid, Duration::from_secs(2), Duration::from_micros(500), Duration::from_millis(800));
        let naive_total = offered_with_stall(&mut naive, Duration::from_secs(2), Duration::from_micros(500), Duration::from_millis(800));
        assert!(naive_total < grid_total, "naive {naive_total} should trail grid {grid_total}");
    }

    #[proxima::test]
    async fn zero_rate_offers_nothing() {
        let mut pacer = GridPacer::new(0.0);
        let total = offered_on_time(&mut pacer, Duration::from_secs(5), Duration::from_millis(1));
        assert_eq!(total, 0);
    }

    #[proxima::test]
    async fn negative_rate_is_clamped() {
        let mut pacer = GridPacer::new(-100.0);
        assert_eq!(pacer.due(Duration::from_secs(1)), 0);
    }
}
