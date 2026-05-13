//! RTT estimation per [RFC 9002 §5].
//!
//! [RFC 9002 §5]: https://www.rfc-editor.org/rfc/rfc9002#section-5

use crate::quic::time::Duration;

use super::constants::K_INITIAL_RTT_MICROS;

/// Estimator state per RFC 9002 §5.
#[derive(Debug, Clone, Copy)]
pub struct RttEstimator {
    pub smoothed_rtt: Option<Duration>,
    pub rttvar: Option<Duration>,
    pub min_rtt: Option<Duration>,
    pub latest_rtt: Option<Duration>,
    pub first_sample_taken: bool,
}

impl Default for RttEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl RttEstimator {
    /// Construct an unsampled estimator. `smoothed_rtt` / `rttvar` /
    /// `min_rtt` / `latest_rtt` are all `None` until the first
    /// `on_sample` call.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            smoothed_rtt: None,
            rttvar: None,
            min_rtt: None,
            latest_rtt: None,
            first_sample_taken: false,
        }
    }

    /// Return the smoothed RTT or [`Self::initial_rtt`] if no sample
    /// has been taken yet.
    #[must_use]
    pub fn smoothed_rtt_or_initial(&self) -> Duration {
        self.smoothed_rtt.unwrap_or(Self::initial_rtt())
    }

    /// Return the rttvar or `initial_rtt / 2` if no sample taken.
    #[must_use]
    pub fn rttvar_or_initial(&self) -> Duration {
        self.rttvar
            .unwrap_or(Duration::from_micros(K_INITIAL_RTT_MICROS / 2))
    }

    /// The kInitialRtt constant per RFC 9002 §6.2.2 — 333 ms.
    #[must_use]
    pub const fn initial_rtt() -> Duration {
        Duration::from_micros(K_INITIAL_RTT_MICROS)
    }

    /// Apply a new RTT sample.
    ///
    /// Algorithm per RFC 9002 §5.3:
    /// 1. `min_rtt = min(min_rtt, latest_rtt)`.
    /// 2. First sample: `smoothed_rtt = latest_rtt`, `rttvar = latest_rtt / 2`.
    /// 3. Subsequent samples:
    ///    - `adjusted_rtt = latest_rtt - ack_delay` if `min_rtt + ack_delay
    ///      <= latest_rtt`, else `latest_rtt`.
    ///    - `rttvar = 3/4 * rttvar + 1/4 * |smoothed_rtt - adjusted_rtt|`.
    ///    - `smoothed_rtt = 7/8 * smoothed_rtt + 1/8 * adjusted_rtt`.
    pub fn on_sample(&mut self, latest_rtt: Duration, ack_delay: Duration) {
        self.latest_rtt = Some(latest_rtt);
        self.min_rtt = match self.min_rtt {
            Some(current) => Some(current.min(latest_rtt)),
            None => Some(latest_rtt),
        };

        if !self.first_sample_taken {
            self.smoothed_rtt = Some(latest_rtt);
            self.rttvar = Some(Duration::from_micros(latest_rtt.as_micros() / 2));
            self.first_sample_taken = true;
            return;
        }

        let min_rtt = self.min_rtt.unwrap_or(latest_rtt);
        let adjusted_rtt = if min_rtt.as_micros() + ack_delay.as_micros() <= latest_rtt.as_micros()
        {
            latest_rtt - ack_delay
        } else {
            latest_rtt
        };

        let smoothed = self.smoothed_rtt.unwrap_or(latest_rtt);
        let rttvar = self
            .rttvar
            .unwrap_or(Duration::from_micros(latest_rtt.as_micros() / 2));
        let rttvar_sample = abs_diff(smoothed, adjusted_rtt);

        // rttvar = (3/4) * rttvar + (1/4) * rttvar_sample
        let new_rttvar =
            Duration::from_micros((3 * rttvar.as_micros() + rttvar_sample.as_micros()) / 4);
        // smoothed_rtt = (7/8) * smoothed_rtt + (1/8) * adjusted_rtt
        let new_smoothed =
            Duration::from_micros((7 * smoothed.as_micros() + adjusted_rtt.as_micros()) / 8);
        self.rttvar = Some(new_rttvar);
        self.smoothed_rtt = Some(new_smoothed);
    }
}

fn abs_diff(left: Duration, right: Duration) -> Duration {
    if left.as_micros() >= right.as_micros() {
        Duration::from_micros(left.as_micros() - right.as_micros())
    } else {
        Duration::from_micros(right.as_micros() - left.as_micros())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn ms(value: u64) -> Duration {
        Duration::from_millis(value)
    }

    #[test]
    fn new_estimator_is_empty() {
        let estimator = RttEstimator::new();
        assert!(!estimator.first_sample_taken);
        assert_eq!(estimator.smoothed_rtt, None);
        assert_eq!(estimator.rttvar, None);
        assert_eq!(estimator.min_rtt, None);
        assert_eq!(estimator.latest_rtt, None);
    }

    #[test]
    fn first_sample_seeds_smoothed_and_rttvar() {
        let mut estimator = RttEstimator::new();
        estimator.on_sample(ms(100), ms(5));
        assert_eq!(estimator.smoothed_rtt, Some(ms(100)));
        assert_eq!(estimator.rttvar, Some(ms(50)));
        assert_eq!(estimator.min_rtt, Some(ms(100)));
        assert_eq!(estimator.latest_rtt, Some(ms(100)));
        assert!(estimator.first_sample_taken);
    }

    #[test]
    fn worked_example_from_design_doc() {
        // From docs/proxima-quic/c14-loss-detection-design.md — RTT walked example.
        // Sample 1: latest_rtt=100 ms, ack_delay=5 ms
        // Sample 2: latest_rtt=90 ms, ack_delay=2 ms
        // Expected after sample 2:
        //   min_rtt = 90 ms; adjusted_rtt = 90 ms (guard fails);
        //   rttvar = 40 ms; smoothed_rtt = 98.75 ms
        let mut estimator = RttEstimator::new();
        estimator.on_sample(ms(100), ms(5));
        estimator.on_sample(ms(90), ms(2));
        assert_eq!(estimator.min_rtt, Some(ms(90)));
        // 98.75 ms = 98750 µs
        assert_eq!(estimator.smoothed_rtt, Some(Duration::from_micros(98_750)));
        assert_eq!(estimator.rttvar, Some(ms(40)));
    }

    #[test]
    fn ack_delay_adjustment_applies_when_guard_passes() {
        // First sample to seed.
        let mut estimator = RttEstimator::new();
        estimator.on_sample(ms(100), ms(0));
        // Second sample: latest_rtt=120, ack_delay=10.
        // min_rtt=100, min_rtt+ack_delay=110, 110 <= 120 TRUE → adjusted=110.
        // rttvar_sample = |100 - 110| = 10
        // rttvar = (3/4) * 50 + (1/4) * 10 = 37 + 2 = 40 (integer-arithmetic: 150/4 = 37, 10/4 = 2, total 40)
        // Actually: (3 * 50_000 + 10_000) / 4 = 160_000 / 4 = 40_000 µs = 40 ms ✓
        // smoothed = (7 * 100_000 + 110_000) / 8 = 810_000 / 8 = 101_250 µs = 101.25 ms
        estimator.on_sample(ms(120), ms(10));
        assert_eq!(estimator.smoothed_rtt, Some(Duration::from_micros(101_250)));
        assert_eq!(estimator.rttvar, Some(ms(40)));
    }

    #[test]
    fn min_rtt_tracks_smallest_sample() {
        let mut estimator = RttEstimator::new();
        estimator.on_sample(ms(100), ms(0));
        estimator.on_sample(ms(50), ms(0));
        estimator.on_sample(ms(80), ms(0));
        assert_eq!(estimator.min_rtt, Some(ms(50)));
    }
}
