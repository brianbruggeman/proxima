//! live run metrics: the time-series a frontend plots while a drive is in
//! flight. the run records via proxima telemetry [`Counter`](proxima_telemetry::metric::Counter)
//! instruments — this module owns only the *display-side* shaping: turning
//! successive cumulative counts into per-second rates ([`Series`]) and a
//! human-readable magnitude ([`human_count`]). both are pure and dep-free so
//! they unit-test without a terminal.
//!
//! rekt is a throughput-*measurement* instrument, so recording must not perturb
//! the hot loop: the metered worker keeps its local `u64` tallies and folds a
//! delta into its registered counter once every [`FLUSH_EVERY`] completions (one
//! relaxed atomic add, off the send). the telemetry drainer snapshots those
//! counters into metric samples on its own thread; nothing here touches the loop.

use std::collections::VecDeque;

/// how many completions a worker buffers locally before folding the delta into
/// its telemetry counter. big enough that the counter add is off the hot path,
/// small enough that a ~100ms drain still sees fresh motion at high rps.
pub const FLUSH_EVERY: u64 = 256;

/// one sampler reading: cumulative totals plus the instantaneous rate since the
/// previous reading.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    pub elapsed: f64,
    pub hits: u64,
    pub errors: u64,
    pub hits_per_sec: f64,
    pub errors_per_sec: f64,
}

/// the bounded time series a chart plots. `push` turns successive cumulative
/// snapshots into per-second rates by differencing against the prior reading.
#[derive(Debug)]
pub struct Series {
    samples: VecDeque<Sample>,
    capacity: usize,
    prev: Option<(f64, u64, u64)>,
    peak_rps: f64,
}

impl Series {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(capacity),
            capacity: capacity.max(1),
            prev: None,
            peak_rps: 0.0,
        }
    }

    /// record a cumulative snapshot. the first one only seeds the baseline (no
    /// prior reading to difference against); every later one yields a rate.
    pub fn push(&mut self, elapsed: f64, hits: u64, errors: u64) {
        if let Some((prev_elapsed, prev_hits, prev_errors)) = self.prev {
            let dt = elapsed - prev_elapsed;
            if dt > 0.0 {
                let hits_per_sec = hits.saturating_sub(prev_hits) as f64 / dt;
                let errors_per_sec = errors.saturating_sub(prev_errors) as f64 / dt;
                self.peak_rps = self.peak_rps.max(hits_per_sec);
                self.samples.push_back(Sample {
                    elapsed,
                    hits,
                    errors,
                    hits_per_sec,
                    errors_per_sec,
                });
                while self.samples.len() > self.capacity {
                    self.samples.pop_front();
                }
            }
        }
        self.prev = Some((elapsed, hits, errors));
    }

    #[must_use]
    pub fn latest(&self) -> Option<&Sample> {
        self.samples.back()
    }

    #[must_use]
    pub fn peak_rps(&self) -> f64 {
        self.peak_rps
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// `(elapsed, rate)` points for the two chart series, over the whole window.
    #[must_use]
    pub fn hits_points(&self) -> Vec<(f64, f64)> {
        self.samples
            .iter()
            .map(|s| (s.elapsed, s.hits_per_sec))
            .collect()
    }

    #[must_use]
    pub fn errors_points(&self) -> Vec<(f64, f64)> {
        self.samples
            .iter()
            .map(|s| (s.elapsed, s.errors_per_sec))
            .collect()
    }

    /// `[min, max]` elapsed spanned by the retained samples.
    #[must_use]
    pub fn x_bounds(&self) -> [f64; 2] {
        match (self.samples.front(), self.samples.back()) {
            (Some(first), Some(last)) if last.elapsed > first.elapsed => [first.elapsed, last.elapsed],
            (Some(first), _) => [first.elapsed, first.elapsed + 1.0],
            _ => [0.0, 1.0],
        }
    }
}

/// a rate or count rendered for a human: `10`, `1.2k`, `382.4k`, `1.24M`. this
/// is the "10rps -> 400k rps" readout the dashboard leads with.
#[must_use]
pub fn human_count(value: f64) -> String {
    let magnitude = value.abs();
    if magnitude >= 1e9 {
        format!("{:.2}G", value / 1e9)
    } else if magnitude >= 1e6 {
        format!("{:.2}M", value / 1e6)
    } else if magnitude >= 1e3 {
        format!("{:.1}k", value / 1e3)
    } else {
        format!("{value:.0}")
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn first_push_only_seeds_no_rate_yet() {
        let mut series = Series::new(16);
        series.push(0.0, 0, 0);
        assert!(series.is_empty());
        assert!(series.latest().is_none());
    }

    #[test]
    fn rate_is_delta_over_dt() {
        let mut series = Series::new(16);
        series.push(1.0, 0, 0);
        series.push(2.0, 1000, 5); // +1000 hits, +5 errors over 1.0s
        let sample = series.latest().expect("a sample");
        assert!((sample.hits_per_sec - 1000.0).abs() < f64::EPSILON);
        assert!((sample.errors_per_sec - 5.0).abs() < f64::EPSILON);
        assert!((series.peak_rps() - 1000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn capacity_evicts_oldest() {
        let mut series = Series::new(2);
        series.push(0.0, 0, 0);
        series.push(1.0, 100, 0);
        series.push(2.0, 200, 0);
        series.push(3.0, 300, 0);
        assert_eq!(series.hits_points().len(), 2);
        assert!((series.x_bounds()[0] - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn non_monotonic_time_is_ignored() {
        let mut series = Series::new(16);
        series.push(1.0, 0, 0);
        series.push(1.0, 500, 0); // same timestamp: no rate, no panic
        assert!(series.is_empty());
    }

    #[test]
    fn human_count_scales_by_magnitude() {
        assert_eq!(human_count(10.0), "10");
        assert_eq!(human_count(1200.0), "1.2k");
        assert_eq!(human_count(382_400.0), "382.4k");
        assert_eq!(human_count(1_240_000.0), "1.24M");
    }
}
