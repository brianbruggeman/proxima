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
                candidate.label,
                baseline.label
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

/// The node ids [`widened_sum_of_squares`] built, named by their role.
pub struct WidenedSumOfSquares {
    pub program: alloc::vec::Vec<crate::op::Op>,
    pub input: crate::op::NodeId,
    pub squares: crate::op::NodeId,
    pub widened: crate::op::NodeId,
    pub sum: crate::op::NodeId,
    pub mean: crate::op::NodeId,
    pub narrowed: crate::op::NodeId,
}

/// `(ggml_float)(x*x)` summed, divided by `n` in double, narrowed to `f32`:
/// llama.cpp's `rms_norm_f32` accumulation (`ggml-cpu/ops.cpp:3957-3964`)
/// spelled in the IR's own dtypes, over a `[rows, width]` input.
#[must_use]
pub fn widened_sum_of_squares(rows: u32, width: u32) -> WidenedSumOfSquares {
    use crate::dtype::DType;
    use crate::map::{self, IndexMap};
    use crate::op::{Extent, Keep, Op, Reduce, ReduceInit, ScalarOp, append};

    let mut program = alloc::vec::Vec::new();
    let full = || IndexMap::Affine(map::projection(2, &[0, 1]));
    let row = || IndexMap::Affine(map::projection(1, &[0]));
    let scalar = || IndexMap::Affine(map::projection(1, &[]));
    let input = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: alloc::vec![Extent::Static(rows), Extent::Static(width)],
            name: None,
        },
    );
    let squares = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![(input, full()), (input, full())],
            name: None,
        },
    );
    let widened = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT64,
            body: ScalarOp::Identity,
            operands: alloc::vec![(squares, full())],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT64,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: widened,
            in_map: full(),
            out_map: IndexMap::Affine(map::projection(2, &[0])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    let count = append(
        &mut program,
        Op::Constant {
            dtype: DType::FLOAT64,
            shape: alloc::vec::Vec::new(),
            value: width as f32,
        },
    );
    let mean = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT64,
            body: ScalarOp::Divide,
            operands: alloc::vec![(sum, row()), (count, scalar())],
            name: None,
        },
    );
    let narrowed = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Identity,
            operands: alloc::vec![(mean, row())],
            name: None,
        },
    );
    WidenedSumOfSquares {
        program,
        input,
        squares,
        widened,
        sum,
        mean,
        narrowed,
    }
}

/// How many norms in `program` widen their squares to f64 -- each
/// [`crate::spec::mean_of_squares`] island adds exactly two nodes over the old
/// f32 form (the widen and the narrow), so a graph golden recorded before the
/// islands differs by twice this.
#[must_use]
pub fn widened_norms(program: &[crate::op::Op]) -> usize {
    program
        .iter()
        .filter(|expr| {
            matches!(expr, crate::op::Op::Elementwise { dtype, body: crate::ScalarOp::Identity, operands, .. }
                if *dtype == crate::DType::FLOAT64
                    && operands.len() == 1
                    && program[operands[0].0.0 as usize].dtype() == crate::DType::FLOAT32)
        })
        .count()
}

/// `crate::cpu::evaluate_named` with the f32 online-softmax cached attention: a reference for
/// tests that compare a fused op against an unfused f32 program, which cannot state llama's f16
/// flash arithmetic.
///
/// # Errors
/// What [`crate::cpu::evaluate_quantized_named_with_config`] raises.
pub fn evaluate_named_online_f32(
    program: &[crate::op::Op],
    symbols: &[u64],
    named: &[(&str, &[f32])],
    outputs: &[crate::op::NodeId],
) -> Result<crate::Evaluated, crate::TensorError> {
    let blocks: alloc::vec::Vec<(&str, crate::QuantizedBlock<'_>)> = named
        .iter()
        .map(|(name, data)| (*name, crate::QuantizedBlock::Float32(data)))
        .collect();
    let config = crate::config::TensorExecutionConfig::builder()
        .numeric(crate::NumericPolicy::bit_exact().with_flash_narrowing(true))
        .build();
    let mut pools = crate::cpu::BufferPools::default();
    let mut validated = None;
    crate::cpu::evaluate_quantized_named_with_config(
        program,
        symbols,
        &blocks,
        outputs,
        (&mut pools, &mut validated),
        None,
        &config,
    )
}
