//! `push_packed_row_multi_row_q4_0_pair_lane_body` (`omega/src/msl/
//! elementwise_reduce_core.rs`) splits each `Q4_0` block across eight lanes of
//! a simdgroup (four elements per lane) and folds every token of a group
//! against one decode of those elements. It sums in a different order than the
//! lane-per-element arm it replaces (`PROXIMA_Q4_0_MULTI_ROW_PAIR_LANE=0`), so
//! the gate here is the one the other `Q4_0` multi-row arms hold against the
//! f32 CPU oracle: relative error below 1e-5 over real-quantized weights, for
//! every token row, across token counts that pad the activation cap and
//! reduction lengths from two to forty-eight four-block groups.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_gguf::quant::q4_0::{BLOCK_BYTES, QK4_0, dequantize, quantize};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    AxisTerm, DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce,
    ReduceInit, ScalarOp, affine, append, projection,
};

const RELATIVE_TOLERANCE: f32 = 1e-5;

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

/// `[out_dim, in_dim] x [tokens, in_dim * stride] -> [tokens, out_dim]`,
/// reduced over `in_dim`; with `fused_scale` the sum feeds a per-feature
/// multiply, which `bind` fuses into the reduce as its epilogue.
fn matmul_program(
    tokens: u32,
    in_dim: u32,
    out_dim: u32,
    stride: i32,
    fused_scale: bool,
) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::UInt8,
            shape: vec![Extent::Static(out_dim), Extent::Static(in_dim)],
            name: None,
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(tokens), Extent::Static(in_dim * stride as u32)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(3, &[1, 2]))),
                (
                    activation,
                    IndexMap::Affine(affine(
                        3,
                        &[
                            (&[AxisTerm::scaled(0, 1)], 0),
                            (&[AxisTerm::scaled(2, stride)], 0),
                        ],
                    )),
                ),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    if !fused_scale {
        return (program, sum);
    }
    let scale = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(out_dim)],
            name: None,
        },
    );
    let scaled = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (sum, IndexMap::Affine(projection(2, &[0, 1]))),
                (scale, IndexMap::Affine(projection(2, &[1]))),
            ],
            name: None,
        },
    );
    (program, scaled)
}

struct Fixture {
    packed: Vec<u8>,
    dequantized: Vec<f32>,
    dense: Vec<f32>,
    scale: Vec<f32>,
}

fn fixture(tokens: usize, in_dim: usize, out_dim: usize, seed: u64) -> Fixture {
    let row_bytes = in_dim / QK4_0 * BLOCK_BYTES;
    let mut packed = vec![0u8; out_dim * row_bytes];
    for (row, chunk) in packed.chunks_exact_mut(row_bytes).enumerate() {
        quantize(&random_vec(seed + row as u64, in_dim), chunk).unwrap();
    }
    let mut dequantized = vec![0.0f32; out_dim * in_dim];
    dequantize(&packed, &mut dequantized).unwrap();
    Fixture {
        packed,
        dequantized,
        dense: random_vec(seed + 9973, tokens * in_dim),
        scale: random_vec(seed + 31, out_dim),
    }
}

fn spread_activation(dense: &[f32], tokens: usize, in_dim: usize, stride: usize) -> Vec<f32> {
    let mut spread = vec![f32::NAN; tokens * in_dim * stride];
    for token in 0..tokens {
        for position in 0..in_dim {
            spread[token * in_dim * stride + position * stride] = dense[token * in_dim + position];
        }
    }
    spread
}

struct Run {
    output: Vec<f32>,
    kernel_keys: Vec<String>,
}

#[derive(Clone, Copy)]
struct Case {
    tokens: usize,
    in_dim: usize,
    out_dim: usize,
    stride: usize,
    fused_scale: bool,
    seed: u64,
}

fn run(case: Case, fixture: &Fixture, pair_lane: &str) -> Run {
    let Case { tokens, in_dim, out_dim, stride, fused_scale, .. } = case;
    let (program, root) =
        matmul_program(tokens as u32, in_dim as u32, out_dim as u32, stride as i32, fused_scale);
    let activation = spread_activation(&fixture.dense, tokens, in_dim, stride);
    let mut blocks = vec![
        QuantizedBlock::Packed { codec: omega::Codec::Q4_0, bytes: &fixture.packed },
        QuantizedBlock::Float32(&activation),
    ];
    if fused_scale {
        blocks.push(QuantizedBlock::Float32(&fixture.scale));
    }
    temp_env::with_vars(
        [
            ("PROXIMA_TILED_GEMM_Q4_0", Some("0")),
            ("PROXIMA_Q4_0_MULTI_ROW_PAIR_LANE", Some(pair_lane)),
        ],
        || {
            let plan = omega::plan(&program, &[], &blocks, &[root], NumericPolicy::default())
                .expect("metal plans the Q4_0 matmul");
            let kernel_keys = plan.kernel_keys().expect("plan exposes its kernel keys");
            let output = omega::execute_plan(&plan, &blocks).expect("metal runs the matmul");
            Run { output: output.root().to_vec(), kernel_keys }
        },
    )
}

fn oracle(case: Case, fixture: &Fixture) -> Vec<f32> {
    let Case { tokens, in_dim, out_dim, fused_scale, .. } = case;
    let mut expected = vec![0.0f32; tokens * out_dim];
    for token in 0..tokens {
        for row in 0..out_dim {
            let sum: f64 = (0..in_dim)
                .map(|position| {
                    f64::from(fixture.dequantized[row * in_dim + position])
                        * f64::from(fixture.dense[token * in_dim + position])
                })
                .sum();
            let scale = if fused_scale { f64::from(fixture.scale[row]) } else { 1.0 };
            expected[token * out_dim + row] = (sum * scale) as f32;
        }
    }
    expected
}

fn relative_error(actual: &[f32], expected: &[f32]) -> f32 {
    let max_diff = actual.iter().zip(expected).map(|(left, right)| (left - right).abs()).fold(0.0, f32::max);
    let max_magnitude = expected.iter().map(|value| value.abs()).fold(0.0, f32::max);
    max_diff / max_magnitude
}

fn assert_matches_the_cpu_oracle(case: Case, expect_pair_lane: bool) {
    let fixture = fixture(case.tokens, case.in_dim, case.out_dim, case.seed);
    let multi = run(case, &fixture, "1");
    assert_eq!(
        multi.output.len(),
        case.tokens * case.out_dim,
        "degenerate gate: wrong element count"
    );
    let admitted = multi.kernel_keys.iter().any(|key| key.contains("_q0l"));
    assert_eq!(
        admitted, expect_pair_lane,
        "tokens={} K={} stride={}: pair-lane admission is {admitted}, expected {expect_pair_lane}; keys={:?}",
        case.tokens, case.in_dim, case.stride, multi.kernel_keys
    );
    if case.fused_scale {
        assert!(
            multi.kernel_keys.iter().any(|key| key.contains("epi")),
            "fused-scale case never fused an epilogue, so it does not exercise the lane write-back; keys={:?}",
            multi.kernel_keys
        );
    }
    let error = relative_error(&multi.output, &oracle(case, &fixture));
    assert!(
        error < RELATIVE_TOLERANCE,
        "tokens={} K={} rows={} stride={} fused={}: relative error {error} against the f32 CPU oracle",
        case.tokens, case.in_dim, case.out_dim, case.stride, case.fused_scale
    );
}

const fn case(tokens: usize, in_dim: usize, out_dim: usize, seed: u64) -> Case {
    Case { tokens, in_dim, out_dim, stride: 1, fused_scale: false, seed }
}

#[test]
fn tokens2_k1536_rows4096_match_the_oracle() {
    assert_matches_the_cpu_oracle(case(2, 1536, 4096, 11), true);
}

#[test]
fn tokens3_k1536_rows2048_match_the_oracle_through_a_padded_cap() {
    assert_matches_the_cpu_oracle(case(3, 1536, 2048, 23), true);
}

#[test]
fn tokens4_k1536_rows2048_match_the_oracle() {
    assert_matches_the_cpu_oracle(case(4, 1536, 2048, 37), true);
}

#[test]
fn tokens8_k1536_rows2048_match_the_oracle() {
    assert_matches_the_cpu_oracle(case(8, 1536, 2048, 41), true);
}

#[test]
fn tokens9_k1536_rows1024_match_the_oracle_across_a_partial_second_group() {
    assert_matches_the_cpu_oracle(case(9, 1536, 1024, 53), true);
}

#[test]
fn tokens2_k6144_rows1536_match_the_oracle_over_forty_eight_groups() {
    assert_matches_the_cpu_oracle(case(2, 6144, 1536, 59), true);
}

#[test]
fn tokens2_k256_rows64_match_the_oracle_with_two_groups() {
    assert_matches_the_cpu_oracle(case(2, 256, 64, 61), true);
}

#[test]
fn tokens2_rows6_fewer_rows_than_two_groups_match_the_oracle() {
    assert_matches_the_cpu_oracle(case(2, 1536, 6, 83), true);
}

#[test]
fn tokens4_k1536_rows1536_fused_epilogue_matches_the_oracle() {
    assert_matches_the_cpu_oracle(
        Case { fused_scale: true, ..case(4, 1536, 1536, 89) },
        true,
    );
}

#[test]
fn tokens8_k1536_rows512_fused_epilogue_matches_the_oracle_with_thirty_two_pairs() {
    assert_matches_the_cpu_oracle(
        Case { fused_scale: true, ..case(8, 1536, 512, 97) },
        true,
    );
}

#[test]
fn tokens3_k1536_rows512_strided_activation_stays_on_the_generic_arm_and_matches_the_oracle() {
    assert_matches_the_cpu_oracle(Case { stride: 2, ..case(3, 1536, 512, 101) }, false);
}

#[test]
fn the_generic_arm_is_reachable_and_sums_in_a_different_order() {
    let case = case(2, 1536, 256, 103);
    let fixture = fixture(case.tokens, case.in_dim, case.out_dim, case.seed);
    let pair_lane = run(case, &fixture, "1");
    let generic = run(case, &fixture, "0");

    assert!(
        !generic.kernel_keys.iter().any(|key| key.contains("_q0l")),
        "PROXIMA_Q4_0_MULTI_ROW_PAIR_LANE=0 still selected the pair-lane arm: {:?}",
        generic.kernel_keys
    );
    let differing = pair_lane
        .output
        .iter()
        .zip(&generic.output)
        .filter(|(left, right)| left.to_bits() != right.to_bits())
        .count();
    assert!(
        differing > 0,
        "control: two arms with different summation orders produced identical bits over {} outputs, \
         so the A/B switch is not selecting two kernels",
        pair_lane.output.len()
    );
    let expected = oracle(case, &fixture);
    assert!(relative_error(&generic.output, &expected) < RELATIVE_TOLERANCE);
}

#[test]
fn distinct_activation_rows_produce_distinct_output_rows() {
    let case = case(2, 1536, 256, 107);
    let fixture = fixture(case.tokens, case.in_dim, case.out_dim, case.seed);
    let multi = run(case, &fixture, "1").output;

    assert_ne!(
        &multi[..case.out_dim],
        &multi[case.out_dim..],
        "control: two different activation rows must not produce the same output row"
    );
}
