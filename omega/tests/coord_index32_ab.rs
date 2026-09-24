//! `PROXIMA_COORD_INDEX32=1` narrows the generic cooperative-reduce
//! coordinate-decomposition loop's divisors to `uint`; every other
//! expression (strides, bases, pointers, reduction/epilogue arithmetic, the
//! `Uniforms` ABI, dispatch geometry) is untouched. This gate proves the
//! narrowing changes no arithmetic: for every shape below,
//! `PROXIMA_COORD_INDEX32=1` must produce bit-exact output against the
//! unset-env default on the same operand bytes.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce, ReduceInit,
    ScalarOp, append, projection,
};

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

/// `dims` elementwise-multiplied then reduced over `reduce_axis` -- a plain
/// weighted sum, no epilogue, no packed operand: the SAME shape family
/// `ADMISSION.md`'s own captured attention kernels take (weights x values,
/// reduced over the context axis), generalized to any rank/axis so the same
/// builder covers the attention shapes, the boundary case, and the plain
/// generic-reduce case below.
fn weighted_sum_program(dims: &[u32], reduce_axis: u16) -> (Vec<Op>, NodeId) {
    let rank = dims.len() as u16;
    let shape: Vec<Extent> = dims.iter().map(|&d| Extent::Static(d)).collect();
    let mut program = Vec::new();
    let weights = append(
        &mut program,
        Op::Input { dtype: DType::Float32, shape: shape.clone(), name: None },
    );
    let values = append(
        &mut program,
        Op::Input { dtype: DType::Float32, shape, name: None },
    );
    let identity_axes: Vec<u16> = (0..rank).collect();
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weights, IndexMap::Affine(projection(rank, &identity_axes))),
                (values, IndexMap::Affine(projection(rank, &identity_axes))),
            ],
            name: None,
        },
    );
    let output_axes: Vec<u16> = (0..rank).filter(|&axis| axis != reduce_axis).collect();
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(projection(rank, &identity_axes)),
            out_map: IndexMap::Affine(projection(rank, &output_axes)),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    (program, sum)
}

fn total_elements(dims: &[u32]) -> usize {
    dims.iter().map(|&d| d as usize).product()
}

fn output_elements(dims: &[u32], reduce_axis: usize) -> usize {
    dims.iter()
        .enumerate()
        .filter(|&(axis, _)| axis != reduce_axis)
        .map(|(_, &d)| d as usize)
        .product()
}

/// Runs the weighted-sum reduce once under the given
/// `PROXIMA_COORD_INDEX32` env value (`None` = unset, today's default emit)
/// and returns the raw output bit patterns -- `to_bits()`, not the floats,
/// for exact-equality comparison.
fn run_bits(dims: &[u32], reduce_axis: usize, seed: u64, coord_index32_env: Option<&str>) -> Vec<u32> {
    let total = total_elements(dims);
    let weights = random_vec(seed, total);
    let values = random_vec(seed + 9973, total);
    let (program, sum) = weighted_sum_program(dims, reduce_axis as u16);
    let blocks = [QuantizedBlock::Float32(&weights), QuantizedBlock::Float32(&values)];

    let output = temp_env::with_var("PROXIMA_COORD_INDEX32", coord_index32_env, || {
        let plan = omega::plan(&program, &[], &blocks, &[sum], NumericPolicy::llama_relaxed())
            .expect("metal plans the weighted-sum reduce");
        omega::execute_plan(&plan, &blocks).expect("metal runs the weighted-sum reduce on a real device")
    });
    output.root().iter().map(|value| value.to_bits()).collect()
}

/// One `(dims, reduce_axis)` case: `PROXIMA_COORD_INDEX32=1` must reproduce
/// the unset-env default bit-for-bit.
fn assert_coord_index32_bit_exact(dims: &[u32], reduce_axis: usize, seed: u64) {
    let baseline = run_bits(dims, reduce_axis, seed, None);
    let shared = run_bits(dims, reduce_axis, seed, Some("1"));
    assert_eq!(
        baseline.len(),
        output_elements(dims, reduce_axis),
        "degenerate gate: dims={dims:?} reduce_axis={reduce_axis} baseline produced the wrong \
         element count"
    );
    assert_eq!(
        baseline, shared,
        "dims={dims:?} reduce_axis={reduce_axis}: PROXIMA_COORD_INDEX32=1 produced different bits \
         than the unset-env default -- the coordinate-decomposition narrowing must not reorder or \
         alter any output's value"
    );
}

/// `ADMISSION.md`'s own captured shape family, width 160 (`f57f93ff7f161e04`
/// analog): `[600, 608, 1, 8, 128]`, reduced over the context axis (1).
#[test]
fn attention_shape_608x8x128_bit_exact() {
    assert_coord_index32_bit_exact(&[600, 608, 1, 8, 128], 1, 4001);
}

/// `ADMISSION.md`'s own second captured shape (`3ca4119c197e7eda` analog,
/// width 64): `[600, 608, 1, 8, 256]`.
#[test]
fn attention_shape_608x8x256_bit_exact() {
    assert_coord_index32_bit_exact(&[600, 608, 1, 8, 256], 1, 4013);
}

/// A plain, softmax/elementwise-free generic reduce -- rank 3, single
/// reduce axis, no epilogue, exercising the SAME `push_cooperative_reduce_
/// body` seam at a shape unrelated to attention. `output_total = 32 * 32 =
/// 1024`, an exact multiple of the cooperative width -- the boundary case
/// below is this test's own non-multiple contrast.
#[test]
fn plain_generic_reduce_bit_exact() {
    assert_coord_index32_bit_exact(&[32, 4096, 32], 1, 4021);
}

/// Boundary case: `output_total` is deliberately NOT a multiple of the
/// cooperative width (`SIMD_WIDTH = 32` at minimum) -- `37 * 11 = 407`,
/// `407 % 32 != 0`, so the LAST cooperative-reduce threadgroup's
/// `output_index >= u.output_total` guard actually masks a partial group.
#[test]
fn boundary_output_total_not_multiple_of_width_bit_exact() {
    assert_coord_index32_bit_exact(&[37, 4096, 11], 1, 4033);
}
