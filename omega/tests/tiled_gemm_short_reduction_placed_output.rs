//! A dense-batched GEMM whose reduction is shorter than `TILED_GEMM_BLOCK_K`
//! and whose output count reaches `COOPERATIVE_SERIAL_MIN_OUTPUTS` must write
//! every output element, whatever the output buffer held before the dispatch.
//!
//! The shape is the 512-row prefill projection of the gemma4 serving plan
//! (`[token=512, batch=1, feature=256, reduce=16]`, 131072 outputs): the dense
//! tile grid (`tiled_gemm_grid2d_spec`) admits it, so the launch is a 16x4
//! grid of tile threadgroups, and the kernel body has to be the tile body that
//! covers the whole grid. A body that indexes by `thread_position_in_grid`
//! alone fills only the first 512 outputs of the 131072 and leaves the rest as
//! whatever the arena slot held, which differs between a fresh slot (zeros) and
//! a reused one (the previous request's bytes).

#![cfg(all(feature = "metal", feature = "metal-tiled-gemm", feature = "metal-output-placement", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use omega::PlacedBuffer;
use proxima_numeric::{DType, NumericPolicy};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp, append, projection};

const TOKEN: u32 = 512;
const REDUCE_LEN: u32 = 16;
const BATCH: u32 = 1;
const FEATURE: u32 = 256;
const OUTPUT_COUNT: usize = (TOKEN * BATCH * FEATURE) as usize;
const OUTPUT_BYTES: usize = OUTPUT_COUNT * size_of::<f32>();

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit() * 2.0 - 1.0).collect()
}

/// Iteration dims: 0=token, 1=reduce, 2=batch, 3=feature; `out_map` lists
/// (token, batch, feature), so the output is `[token][batch][feature]` row-major.
fn short_reduction_projection_program() -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(FEATURE), Extent::Static(BATCH), Extent::Static(REDUCE_LEN)],
            name: None,
        },
    );
    let other = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(TOKEN), Extent::Static(BATCH), Extent::Static(REDUCE_LEN)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(4, &[3, 2, 1]))),
                (other, IndexMap::Affine(projection(4, &[0, 2, 1]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(projection(4, &[0, 1, 2, 3])),
            out_map: IndexMap::Affine(projection(4, &[0, 2, 3])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    (program, sum)
}

fn reference_output(weight: &[f32], other: &[f32]) -> Vec<f32> {
    let reduce_len = REDUCE_LEN as usize;
    let feature = FEATURE as usize;
    (0..TOKEN as usize)
        .flat_map(|token| (0..feature).map(move |column| (token, column)))
        .map(|(token, column)| {
            let weight_row = &weight[column * reduce_len..(column + 1) * reduce_len];
            let other_row = &other[token * reduce_len..(token + 1) * reduce_len];
            weight_row
                .iter()
                .zip(other_row)
                .map(|(left, right)| f64::from(*left) * f64::from(*right))
                .sum::<f64>() as f32
        })
        .collect()
}

fn run_on_prefilled_output(
    plan: &omega::Plan,
    sum: NodeId,
    blocks: &[QuantizedBlock<'_>],
    prefill_bits: u32,
) -> Vec<f32> {
    let buffer: PlacedBuffer = omega::allocate_placed_buffer(OUTPUT_BYTES).expect("allocates the caller-owned output slot");
    omega::write_placed_buffer_f32(&buffer, 0, &vec![f32::from_bits(prefill_bits); OUTPUT_COUNT]);
    omega::execute_plan_with_placements(plan, blocks, &[], &[(sum, &buffer, 0)], &mut Vec::new())
        .expect("executes the short-reduction projection into the prefilled slot");
    omega::read_placed_buffer_f32(&buffer, 0, OUTPUT_COUNT)
}

fn first_difference(left: &[f32], right: &[f32]) -> Option<(usize, usize, f32, f32)> {
    left.iter()
        .zip(right)
        .position(|(left_value, right_value)| left_value.to_bits() != right_value.to_bits())
        .map(|index| (index / FEATURE as usize, index % FEATURE as usize, left[index], right[index]))
}

#[test]
fn tiled_gemm_short_reduction_output_is_independent_of_the_slots_prior_bytes() {
    let (program, sum) = short_reduction_projection_program();
    let weight = random_vec(7001, (FEATURE * BATCH * REDUCE_LEN) as usize);
    let other = random_vec(8002, (TOKEN * BATCH * REDUCE_LEN) as usize);
    let blocks = [QuantizedBlock::Float32(&weight), QuantizedBlock::Float32(&other)];
    let plan = omega::plan(&program, &[], &blocks, &[sum], NumericPolicy::llama_relaxed())
        .expect("plans the short-reduction projection once, reused by every run below");

    let zero_filled = run_on_prefilled_output(&plan, sum, &blocks, 0);
    let poisoned = run_on_prefilled_output(&plan, sum, &blocks, 0xFFFF_FFFF);
    let reference = reference_output(&weight, &other);

    let differing = zero_filled
        .iter()
        .zip(&poisoned)
        .filter(|(zero_value, poison_value)| zero_value.to_bits() != poison_value.to_bits())
        .count();
    assert_eq!(
        differing,
        0,
        "{differing}/{OUTPUT_COUNT} outputs depend on the slot's prior bytes; first (token, feature, zero-filled, 0xFF-filled) = {:?}",
        first_difference(&zero_filled, &poisoned)
    );

    let worst = zero_filled
        .iter()
        .zip(&reference)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst < 1e-3,
        "the projection must match the f64 reference, worst absolute error {worst} over {OUTPUT_COUNT} outputs"
    );
}
