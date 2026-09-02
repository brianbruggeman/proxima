//! Shared deterministic test/example fixtures — gated behind `test-support`
//! (default-off) so nothing here reaches a normal build. This crate and
//! `omega` both pull it in via a `dev-dependencies` edge on themselves/each
//! other with the feature enabled, so `cargo test`/`cargo run --example`
//! see it without any extra `--features` flag, and a plain `cargo build`
//! never does.

/// Deterministic pseudo-random source for reproducible float inputs across
/// tests, examples, and benches — not cryptographic, never seeded from
/// entropy. Was copy-pasted verbatim into 7 separate files before this
/// module existed.
pub struct Lcg(pub u64);

impl Lcg {
    /// Uniform in `[-1, 1)`. Uses the top 32 bits of the 64-bit LCG state
    /// (an LCG's low bits have short periods; the high bits do not) divided
    /// by [`u32::MAX`] -- shifting by 33 (a 31-bit remainder) while
    /// dividing by a 32-bit max was this function's own prior bug: `bits`
    /// could never exceed roughly half of `u32::MAX`, so every caller of
    /// this "uniform in `[-1, 1)`" function was actually drawing from
    /// `[-1, 0)`, silently halving both the mean and the variance every
    /// caller of this function assumed.
    pub fn next_unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let bits = (self.0 >> 32) as u32;
        (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}
