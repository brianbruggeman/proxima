//! RFC 6298 retransmission-timeout estimation with Karn's algorithm.
//!
//! Jacobson/Karels smoothing in integer microseconds (the RFC's `alpha = 1/8`,
//! `beta = 1/4` become `>>3` and `>>2` shifts, exactly as Linux computes them):
//!
//! - first sample R:  `SRTT = R`, `RTTVAR = R/2`
//! - later sample R': `RTTVAR = 3/4·RTTVAR + 1/4·|SRTT-R'|`, then
//!   `SRTT = 7/8·SRTT + 1/8·R'`
//! - `RTO = SRTT + max(G, 4·RTTVAR)`, clamped to `[1s, 60s]`
//!
//! Karn's algorithm (RFC 6298 §3, rule 6): a sample drawn from a retransmitted
//! segment is ambiguous and MUST NOT update the estimator. On timeout the RTO
//! is doubled (§5.5) and that backed-off value holds until the next clean
//! sample recomputes from SRTT/RTTVAR.

use super::time::Duration;

/// RFC 6298 RTO estimator. Microsecond integer arithmetic, no float, no_std.
#[derive(Debug, Clone, Copy)]
pub struct RtoEstimator {
    srtt_us: Option<u64>,
    rttvar_us: u64,
    rto_us: u64,
}

impl RtoEstimator {
    /// RFC 6298 §2.4: a computed RTO below 1 second is rounded up to 1 second.
    pub const RTO_MIN: Duration = Duration::from_secs(1);
    /// Upper clamp (RFC 6298 §2.5 recommends a max of at least 60 seconds).
    pub const RTO_MAX: Duration = Duration::from_secs(60);
    /// Clock granularity G used in `max(G, 4·RTTVAR)`; 1 ms here.
    pub const GRANULARITY: Duration = Duration::from_millis(1);

    const K: u64 = 4;

    /// A fresh estimator before any sample: RFC 6298 §2.1 initial RTO of 1s.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            srtt_us: None,
            rttvar_us: 0,
            rto_us: Self::RTO_MIN.as_micros(),
        }
    }

    /// Feed a round-trip sample. Karn's algorithm: retransmit-derived samples
    /// are ambiguous and ignored.
    pub fn on_sample(&mut self, sample: Duration, from_retransmit: bool) {
        if from_retransmit {
            return;
        }
        let measured = sample.as_micros();
        match self.srtt_us {
            None => {
                self.srtt_us = Some(measured);
                self.rttvar_us = measured / 2;
            }
            Some(srtt) => {
                self.rttvar_us =
                    self.rttvar_us - (self.rttvar_us >> 2) + (srtt.abs_diff(measured) >> 2);
                self.srtt_us = Some(srtt - (srtt >> 3) + (measured >> 3));
            }
        }
        self.recompute();
    }

    fn recompute(&mut self) {
        let srtt = self.srtt_us.unwrap_or(0);
        let variance_term = Self::K
            .saturating_mul(self.rttvar_us)
            .max(Self::GRANULARITY.as_micros());
        let raw = srtt.saturating_add(variance_term);
        self.rto_us = raw.clamp(Self::RTO_MIN.as_micros(), Self::RTO_MAX.as_micros());
    }

    /// Double the RTO on a timeout (RFC 6298 §5.5), capped at [`Self::RTO_MAX`].
    pub fn backoff(&mut self) {
        self.rto_us = self.rto_us.saturating_mul(2).min(Self::RTO_MAX.as_micros());
    }

    #[must_use]
    pub fn rto(&self) -> Duration {
        Duration::from_micros(self.rto_us)
    }

    #[must_use]
    pub fn smoothed_rtt(&self) -> Option<Duration> {
        self.srtt_us.map(Duration::from_micros)
    }

    #[must_use]
    pub fn rtt_variation(&self) -> Duration {
        Duration::from_micros(self.rttvar_us)
    }
}

impl Default for RtoEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// After any sequence of RTT samples, RTO stays clamped to [RTO_MIN, RTO_MAX].
        #[test]
        fn rto_stays_within_rfc6298_bounds_after_arbitrary_samples(
            samples in prop::collection::vec((0_u64..=10_000_000, any::<bool>()), 0..32),
        ) {
            let mut estimator = RtoEstimator::new();
            for (micros, from_retransmit) in samples {
                estimator.on_sample(Duration::from_micros(micros), from_retransmit);
                let rto = estimator.rto();
                prop_assert!(
                    rto >= RtoEstimator::RTO_MIN,
                    "RTO {rto:?} fell below RTO_MIN {:?}", RtoEstimator::RTO_MIN
                );
                prop_assert!(
                    rto <= RtoEstimator::RTO_MAX,
                    "RTO {rto:?} exceeded RTO_MAX {:?}", RtoEstimator::RTO_MAX
                );
            }
        }

        /// After any number of `backoff` calls RTO must not exceed RTO_MAX.
        #[test]
        fn rto_backoff_caps_at_rto_max(backoffs in 0_usize..=64) {
            let mut estimator = RtoEstimator::new();
            estimator.on_sample(Duration::from_millis(500), false);
            for _ in 0..backoffs {
                estimator.backoff();
            }
            prop_assert!(
                estimator.rto() <= RtoEstimator::RTO_MAX,
                "RTO exceeded RTO_MAX after {backoffs} backoffs"
            );
        }

        /// `on_sample` never panics for any microsecond value.
        #[test]
        fn on_sample_never_panics(micros in any::<u64>(), from_retransmit in any::<bool>()) {
            let mut estimator = RtoEstimator::new();
            estimator.on_sample(Duration::from_micros(micros), from_retransmit);
        }
    }

    #[test]
    fn initial_rto_is_one_second() {
        assert_eq!(RtoEstimator::new().rto(), Duration::from_secs(1));
    }

    // First sample: SRTT=R, RTTVAR=R/2, RTO=SRTT+4·RTTVAR (RFC 6298 §2.2).
    // R=500ms -> SRTT=500ms, RTTVAR=250ms, RTO=500+1000=1500ms (exceeds the min).
    #[test]
    fn first_sample_sets_srtt_and_rttvar() {
        let mut estimator = RtoEstimator::new();
        estimator.on_sample(Duration::from_millis(500), false);
        assert_eq!(estimator.smoothed_rtt(), Some(Duration::from_millis(500)));
        assert_eq!(estimator.rtt_variation(), Duration::from_millis(250));
        assert_eq!(estimator.rto(), Duration::from_millis(1500));
    }

    // Second sample, integer Jacobson/Karels (discipline-log worked example):
    // sample1=100ms -> SRTT=100000, RTTVAR=50000
    // sample2=120ms -> RTTVAR=3/4·50000+1/4·20000=42500; SRTT=7/8·100000+1/8·120000=102500
    #[test]
    fn second_sample_smooths_per_rfc6298() {
        let mut estimator = RtoEstimator::new();
        estimator.on_sample(Duration::from_millis(100), false);
        estimator.on_sample(Duration::from_millis(120), false);
        assert_eq!(
            estimator.smoothed_rtt(),
            Some(Duration::from_micros(102_500))
        );
        assert_eq!(estimator.rtt_variation(), Duration::from_micros(42_500));
        // RTO = 102500 + 4·42500 = 272500us < 1s -> clamped up to 1s.
        assert_eq!(estimator.rto(), Duration::from_secs(1));
    }

    #[test]
    fn karn_suppresses_retransmit_sample() {
        let mut estimator = RtoEstimator::new();
        estimator.on_sample(Duration::from_millis(500), false);
        let before = estimator.rto();
        estimator.on_sample(Duration::from_millis(50), true);
        assert_eq!(estimator.rto(), before);
        assert_eq!(estimator.smoothed_rtt(), Some(Duration::from_millis(500)));
    }

    #[test]
    fn backoff_doubles_then_caps_at_max() {
        let mut estimator = RtoEstimator::new();
        estimator.on_sample(Duration::from_millis(500), false);
        assert_eq!(estimator.rto(), Duration::from_millis(1500));
        estimator.backoff();
        assert_eq!(estimator.rto(), Duration::from_millis(3000));
        for _ in 0..10 {
            estimator.backoff();
        }
        assert_eq!(estimator.rto(), RtoEstimator::RTO_MAX);
    }
}
