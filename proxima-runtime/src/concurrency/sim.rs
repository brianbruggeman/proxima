//! A deterministic closed-loop workload with a known crest — the proof harness
//! for the controller. No IO, no sleeps, no RNG: convergence is a property the
//! `#[test]` and the bench both assert against the same model.
//!
//! The model follows Little's Law. Below the crest, added in-flight work is pure
//! gain (latency at the floor, throughput rises linearly). Above it, the executor
//! / poll-set overhead makes round-trips grow quadratically, so throughput falls
//! — a real crest at `peak`, not a plateau.

use core::time::Duration;

use super::Sample;

/// A workload whose throughput crests at `peak` concurrency.
#[derive(Debug, Clone, Copy)]
pub struct CrestModel {
    /// The concurrency at which throughput peaks (the answer the controller must
    /// find).
    pub peak: usize,
    /// The latency floor — round-trip with zero queueing.
    pub floor: Duration,
    /// Deterministic ripple amplitude on throughput (fraction), so the CoV gate
    /// has sub-threshold noise to reject. Phase advances by `concurrency` each
    /// read, giving a repeatable but non-trivial wobble.
    pub ripple: f64,
    /// The coefficient of variation reported in every sample (the noise floor the
    /// gate divides by). Set at or above `ripple` so genuine wobble is gated out.
    pub cov: f64,
}

impl CrestModel {
    /// A crest at `peak` with a 1 ms floor, 1% ripple, 2% reported CoV — the
    /// ripple sits under the gate's noise floor (`coefficient_of_variation_threshold·CoV`) so a genuine
    /// per-step signal change is distinguishable from wobble.
    #[must_use]
    pub fn new(peak: usize) -> Self {
        Self {
            peak,
            floor: Duration::from_millis(1),
            ripple: 0.01,
            cov: 0.02,
        }
    }

    /// The round-trip at `concurrency`: the floor up to the crest, then growing
    /// quadratically past it (`floor · (c/peak)²`).
    #[must_use]
    pub fn rtt(&self, concurrency: usize) -> Duration {
        let peak = self.peak.max(1) as f64;
        let c = concurrency.max(1) as f64;
        if c <= peak {
            self.floor
        } else {
            let stretch = (c / peak) * (c / peak);
            self.floor.mul_f64(stretch)
        }
    }

    /// Throughput at `concurrency` = `c / rtt(c)`. Rises to `peak/floor` at the
    /// crest, then falls. A deterministic ripple wobbles it within the noise
    /// floor so the gate has something to reject.
    #[must_use]
    pub fn throughput(&self, concurrency: usize) -> f64 {
        let rtt_secs = self.rtt(concurrency).as_secs_f64();
        let base = concurrency as f64 / rtt_secs;
        // a repeatable wobble in [-ripple, +ripple], no RNG.
        let phase = ((concurrency.wrapping_mul(2_654_435_761) >> 8) & 0xff) as f64 / 255.0;
        let wobble = (phase - 0.5) * 2.0 * self.ripple;
        base * (1.0 + wobble)
    }

    /// The full sample the controller reads at this concurrency.
    #[must_use]
    pub fn sample(&self, concurrency: usize) -> Sample {
        let rtt = self.rtt(concurrency);
        Sample {
            concurrency,
            throughput: self.throughput(concurrency),
            cov: self.cov,
            rtt_min: self.floor,
            rtt_p50: rtt,
            rtt_p99: rtt,
            util: (concurrency as f64 / self.peak.max(1) as f64).min(1.0),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn throughput_crests_at_peak() {
        let model = CrestModel::new(32);
        let at_peak = model.throughput(32);
        // strictly above the crest throughput must fall
        assert!(
            model.throughput(64) < at_peak,
            "throughput must fall past the crest"
        );
        assert!(
            model.throughput(128) < model.throughput(64),
            "monotone decline past crest"
        );
        // below the crest, throughput rises with concurrency
        assert!(
            model.throughput(16) < at_peak,
            "throughput rises to the crest"
        );
    }

    #[test]
    fn rtt_is_floor_below_crest_and_grows_above() {
        let model = CrestModel::new(32);
        assert_eq!(model.rtt(16), model.floor);
        assert_eq!(model.rtt(32), model.floor);
        assert!(model.rtt(64) > model.floor);
    }
}
