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

/// One named backend's flat row-major output for a parity comparison --
/// `label` names the backend (e.g. `"cpu"`, `"metal_production"`,
/// `"metal_wide"`) for the printed table and any failure message.
pub struct ParityRun<'a> {
    pub label: &'a str,
    pub values: &'a [f32],
}

/// The shared row-by-row parity loop every CPU/Metal (or width/width)
/// differential test in this workspace was hand-rolling: each candidate's
/// row is compared against the matching baseline row, relative to that
/// row's own L2 norm (immune to a row's absolute scale varying case to
/// case, per guiding-principle 9 -- never all-zero/all-one filler). Prints
/// one line per (row, candidate) so a red run names the exact numbers that
/// disagree, and returns the formatted failures whose relative error
/// exceeds `tolerance` for the caller to assert on.
#[cfg(feature = "std")]
#[must_use]
pub fn compare_rows_relative_to_norm(
    case_label: &str,
    rows: usize,
    tolerance: f32,
    baseline: &ParityRun<'_>,
    candidates: &[ParityRun<'_>],
) -> Vec<String> {
    let row_length = baseline.values.len() / rows.max(1);
    let mut failures = Vec::new();
    for row in 0..rows {
        let expected_row = &baseline.values[row * row_length..(row + 1) * row_length];
        let row_norm = expected_row
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt()
            .max(1e-6);
        for candidate in candidates {
            let actual_row = &candidate.values[row * row_length..(row + 1) * row_length];
            let relative_error = expected_row
                .iter()
                .zip(actual_row.iter())
                .map(|(expected, actual)| (actual - expected).abs())
                .fold(0.0_f32, f32::max)
                / row_norm;
            std::println!(
                "{case_label} row={row} backend={} vs {} relative_error={relative_error:e}",
                candidate.label, baseline.label
            );
            if relative_error > tolerance {
                failures.push(std::format!(
                    "{case_label} row={row} backend={} vs {} exceeds tolerance {tolerance:e}: relative_error={relative_error:e}",
                    candidate.label, baseline.label
                ));
            }
        }
    }
    failures
}

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
