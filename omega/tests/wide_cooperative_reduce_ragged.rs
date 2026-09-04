//! GPU-vs-CPU parity gate for `metal-wide-cooperative-reduce`'s ragged
//! cases -- the guard `msl::push_cooperative_reduce_tail`'s two-level fold
//! exists for (`docs/discipline.md`'s row for this initiative): a reduction
//! extent that is not a whole multiple of the chosen threadgroup width, and
//! one smaller than a single simdgroup (`SIMD_WIDTH`, 32). Every test here
//! requires a real Metal device and this feature on -- none of them skip.

#![cfg(all(
    feature = "metal",
    feature = "metal-wide-cooperative-reduce",
    target_os = "macos"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp, append,
    evaluate, infer, projection,
};

/// Peak-magnitude-normalized relative error, never per-row -- a per-row
/// relative error explodes at zero crossings (a near-zero reference makes an
/// ordinary rounding-noise diff read as a huge relative error) without
/// saying anything about correctness. Normalizing by the batch's own peak
/// magnitude instead keeps the denominator meaningful across the whole
/// comparison.
fn peak_normalized_max_relative_error(cpu: &[f32], metal: &[f32]) -> f32 {
    let peak = cpu.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
    if peak == 0.0 {
        return cpu
            .iter()
            .zip(metal.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
    }
    cpu.iter()
        .zip(metal.iter())
        .map(|(left, right)| (left - right).abs() / peak)
        .fold(0.0f32, f32::max)
}

/// Looser than `omega/tests/metal_parity.rs`'s own `assert_parity` default
/// (`1e-6`) -- the two-level fold changes summation ORDER
/// (simdgroup-then-threadgroup instead of one flat `simd_sum`), so bit
/// equality is not expected, and at the deepest chain this sweep drives
/// (4095/4096/4097 terms, the real RMS-norm width) that reorder measured a
/// worst-case abs diff of `5.7220459e-5` (`add`, cols=4095) -- the same
/// shape `metal_parity.rs::attention_block_spec_parity_matches_within_
/// epsilon`'s own doc widens its bound for ("a longer chain of GPU-vs-CPU
/// reduction legitimately accumulates more float error than the 1e-6
/// default"). `1e-4` carries ~1.7x headroom over that measured worst case.
const ABSOLUTE_EPSILON: f32 = 1e-4;

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

/// `next_unit` scaled into `[0.98, 1.02)` -- see
/// `metal_parity.rs::random_vec_near_one`'s own doc for why a `Multiply`
/// reduce needs this instead of raw `[-1, 1)` values (a repeated product
/// over magnitudes below 1 collapses to 0 within a few dozen terms and
/// makes a broken lane combination indistinguishable from a correct one).
fn random_vec_near_one(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| 1.0 + lcg.next_unit() * 0.02).collect()
}

/// One `Reduce` over a `(1, cols)` input's last axis, output `(1,)` --
/// `axis_reduce_program`'s shape in `metal_parity.rs`, with `rows` pinned to
/// 1 so `cols` alone controls the reduction extent `msl::
/// cooperative_reduce_width` sizes the dispatch from.
fn single_row_reduce_program(cols: u32, body: ScalarOp, init: ReduceInit) -> Vec<Op> {
    let mut program = Vec::new();
    let input = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1), Extent::Static(cols)],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body,
            init,
            operand: input,
            in_map: IndexMap::Affine(projection(2, &[0, 1])),
            out_map: IndexMap::Affine(projection(2, &[0])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    program
}

/// `cols` values swept across both ragged guards this feature's tail fold
/// depends on: a width `cooperative_reduce_width` would round UP to (so the
/// per-lane loop leaves some lanes idle on the last stride) and a width the
/// chosen threadgroup width does NOT evenly divide, alongside the real
/// RMS-norm width (4096, `docs/discipline.md`'s measured defect) and its
/// immediate ragged neighbors (4095, 4097).
///
/// - `1`: single element -- every lane past 0 idle for the whole reduce.
/// - `17`, `31`: below `SIMD_WIDTH` (32) -- exercises `reduction_total < 32`.
/// - `33`, `63`: just above one simdgroup, not a multiple of 32.
/// - `100`: `cooperative_reduce_width` picks `width = max(32, next_mul_32(25))
///   = 32` here (`100 / 4 = 25` rounds up to 32) -- NOT a multiple of 100
///   itself, the ragged per-lane-loop case at the smallest scaled width.
/// - `385`: the real op count from the sealed decode-step measurement this
///   initiative closes (`gpu_ns_per_op` row), scaled down to an extent.
/// - `4095`, `4096`, `4097`: the real RMS-norm width and its immediate
///   ragged neighbors.
const RAGGED_COLS: [u32; 9] = [1, 17, 31, 33, 63, 100, 385, 4095, 4097];

/// `4096` alone, kept separate from [`RAGGED_COLS`] only so a failure
/// reads as "the exact measured shape broke", not lost in a sweep line.
const RMS_NORM_COLS: u32 = 4096;

fn assert_wide_reduce_parity(case: &str, cols: u32, body: ScalarOp, init: ReduceInit, input: &[f32]) {
    let program = single_row_reduce_program(cols, body, init);
    infer(&program, &[]).unwrap_or_else(|error| panic!("{case} cols={cols}: infers: {error}"));
    let cpu = evaluate(&program, &[], &[input], &[])
        .unwrap_or_else(|error| panic!("{case} cols={cols}: cpu evaluates: {error}"));
    let metal = omega::execute(&program, &[], &[QuantizedBlock::Float32(input)], &[])
        .unwrap_or_else(|error| panic!("{case} cols={cols}: metal executes on a real device: {error}"));

    assert_eq!(
        cpu.root().len(),
        metal.root().len(),
        "{case} cols={cols}: element count mismatch"
    );
    let max_abs_diff = cpu
        .root()
        .iter()
        .zip(metal.root().iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    let relative = peak_normalized_max_relative_error(cpu.root(), metal.root());
    assert!(
        max_abs_diff <= ABSOLUTE_EPSILON,
        "{case} cols={cols}: max abs diff {max_abs_diff:e} exceeds {ABSOLUTE_EPSILON:e} \
         (peak-normalized relative {relative:e})"
    );
    println!(
        "{case} cols={cols}: max abs diff = {max_abs_diff:e}, peak-normalized relative = {relative:e}"
    );
}

#[test]
fn wide_cooperative_reduce_add_holds_parity_on_ragged_extents() {
    for cols in RAGGED_COLS.iter().copied().chain([RMS_NORM_COLS]) {
        let input = random_vec(0x1000 + u64::from(cols), cols as usize);
        assert_wide_reduce_parity("add", cols, ScalarOp::Add, ReduceInit::Zero, &input);
    }
}

#[test]
fn wide_cooperative_reduce_maximum_holds_parity_on_ragged_extents() {
    for cols in RAGGED_COLS.iter().copied().chain([RMS_NORM_COLS]) {
        let input = random_vec(0x2000 + u64::from(cols), cols as usize);
        assert_wide_reduce_parity(
            "maximum",
            cols,
            ScalarOp::Maximum,
            ReduceInit::NegativeInfinity,
            &input,
        );
    }
}

#[test]
fn wide_cooperative_reduce_minimum_holds_parity_on_ragged_extents() {
    for cols in RAGGED_COLS.iter().copied().chain([RMS_NORM_COLS]) {
        let input = random_vec(0x3000 + u64::from(cols), cols as usize);
        assert_wide_reduce_parity(
            "minimum",
            cols,
            ScalarOp::Minimum,
            ReduceInit::PositiveInfinity,
            &input,
        );
    }
}

#[test]
fn wide_cooperative_reduce_multiply_holds_parity_on_ragged_extents() {
    for cols in RAGGED_COLS.iter().copied().chain([RMS_NORM_COLS]) {
        let input = random_vec_near_one(0x4000 + u64::from(cols), cols as usize);
        assert_wide_reduce_parity("multiply", cols, ScalarOp::Multiply, ReduceInit::One, &input);
    }
}
