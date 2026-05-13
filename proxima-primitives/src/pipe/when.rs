use serde::{Deserialize, Serialize};

// ── seeded stochastic decision — the reusable keystone ───────────────────────
//
// `When` is a config-expressible, seeded, DETERMINISTIC coin flip. It is the
// shared primitive behind chaos injection, fuzz-gating, sampling, and canary
// routing — none of which are baked in here. The contract is narrow on purpose:
// `fires(call_index)` is a pure function of `(seed, prob, call_index)`. No
// global RNG, no wall-clock, no `Instant` — the same `(seed, prob)` always
// produces the same fire sequence, so a config replays bit-for-bit.

/// A seeded, deterministic stochastic gate.
///
/// `fires(call_index)` returns `true` with probability `prob`, derived purely
/// from `(seed, call_index)` — reproducible across runs, machines, and threads.
/// Wire it as a `Filter` predicate to drop/pass a fraction of calls, or reuse
/// it directly for sampling/canary/chaos decisions.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct When {
    /// Fire probability in `[0, 1]`. `0.0` never fires; `1.0` always fires.
    pub prob: f64,
    /// Seed pinning the fire sequence. Same `(seed, prob)` => same sequence.
    pub seed: u64,
}

impl When {
    /// Start a builder from the fire probability (clamped into `[0, 1]`).
    #[must_use]
    pub fn prob(probability: f64) -> Self {
        Self {
            prob: probability.clamp(0.0, 1.0),
            seed: 0,
        }
    }

    /// Pin the seed for the fire sequence.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Deterministically decide whether call `call_index` fires. Pure in
    /// `(seed, prob, call_index)`: the rng is seeded per call so the result
    /// never depends on prior calls, ordering, or any global state.
    #[must_use]
    pub fn fires(&self, call_index: u64) -> bool {
        if self.prob <= 0.0 {
            return false;
        }
        if self.prob >= 1.0 {
            return true;
        }
        let mut rng = fastrand::Rng::with_seed(self.seed.wrapping_add(call_index));
        rng.f64() < self.prob
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn fire_sequence_is_deterministic_across_runs() {
        let gate = When::prob(0.5).seed(0xABCD_1234);
        let first: Vec<bool> = (0..512).map(|index| gate.fires(index)).collect();
        let second: Vec<bool> = (0..512).map(|index| gate.fires(index)).collect();
        assert_eq!(
            first, second,
            "same (seed, prob) yields an identical fire sequence"
        );
    }

    #[test]
    fn prob_zero_never_fires_and_prob_one_always_fires() {
        let never = When::prob(0.0).seed(7);
        let always = When::prob(1.0).seed(7);
        for index in 0..1_000 {
            assert!(!never.fires(index), "prob 0.0 must never fire");
            assert!(always.fires(index), "prob 1.0 must always fire");
        }
    }

    #[test]
    fn prob_is_clamped_into_unit_interval() {
        assert_eq!(
            When::prob(-3.0).prob,
            0.0,
            "negative probability clamps to 0"
        );
        assert_eq!(
            When::prob(42.0).prob,
            1.0,
            "above-one probability clamps to 1"
        );
    }

    #[test]
    fn distinct_seeds_diverge() {
        let left = When::prob(0.5).seed(1);
        let right = When::prob(0.5).seed(2);
        let diverged = (0..512).any(|index| left.fires(index) != right.fires(index));
        assert!(diverged, "different seeds must produce different sequences");
    }

    #[test]
    fn empirical_fire_rate_tracks_probability() {
        let gate = When::prob(0.25).seed(99);
        let fired = (0..10_000_u64).filter(|&index| gate.fires(index)).count();
        let rate = fired as f64 / 10_000.0;
        assert!(
            (rate - 0.25).abs() < 0.05,
            "observed rate {rate} should track prob 0.25"
        );
    }
}
