//! `push_packed_row_blocked_body`'s single-token (decode) write-back finishes
//! each of a simdgroup's output rows on its own lane when a reduce epilogue is
//! fused, and a fused body that mentions one tensor several times loads it
//! once. This gate fuses `sum * (gate * gate + gate)` -- the gate vector is
//! mentioned three times, the shape a fused `gelu` or `silu` takes -- into a
//! one-token packed matvec and holds every output against the f64 CPU oracle:
//! a full row count, a count that leaves a partial row group (the lanes past
//! the last row must stay masked), counts below the rows one simdgroup folds,
//! and `Q6_K`, whose simdgroup folds a single row.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_gguf::quant::{q4_0, q6_k};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce, ReduceInit,
    ScalarOp, append, projection,
};

const RELATIVE_TOLERANCE: f32 = 1e-5;

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

fn elementwise(program: &mut Vec<Op>, body: ScalarOp, operands: Vec<(NodeId, IndexMap)>) -> NodeId {
    append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body,
            operands,
            name: None,
        },
    )
}

/// `[out_dim, in_dim] x [1, in_dim]` reduced over `in_dim`, then
/// `sum * (gate * gate + gate)` over a `[out_dim]` gate input.
fn gated_matvec_program(in_dim: u32, out_dim: u32) -> (Vec<Op>, NodeId) {
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
            shape: vec![Extent::Static(1), Extent::Static(in_dim)],
            name: None,
        },
    );
    let product = elementwise(
        &mut program,
        ScalarOp::Multiply,
        vec![
            (weight, IndexMap::Affine(projection(3, &[1, 2]))),
            (activation, IndexMap::Affine(projection(3, &[0, 2]))),
        ],
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
    let gate = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(out_dim)],
            name: None,
        },
    );
    let gate_read = || (gate, IndexMap::Affine(projection(1, &[0])));
    let squared = elementwise(&mut program, ScalarOp::Multiply, vec![gate_read(), gate_read()]);
    let shaped = elementwise(
        &mut program,
        ScalarOp::Add,
        vec![(squared, IndexMap::Affine(projection(1, &[0]))), gate_read()],
    );
    let gated = elementwise(
        &mut program,
        ScalarOp::Multiply,
        vec![
            (sum, IndexMap::Affine(projection(2, &[0, 1]))),
            (shaped, IndexMap::Affine(projection(2, &[1]))),
        ],
    );
    (program, gated)
}

struct Weights {
    codec: omega::Codec,
    packed: Vec<u8>,
    dequantized: Vec<f32>,
}

fn q4_0_weights(in_dim: usize, out_dim: usize, seed: u64) -> Weights {
    let row_bytes = in_dim / q4_0::QK4_0 * q4_0::BLOCK_BYTES;
    let mut packed = vec![0u8; out_dim * row_bytes];
    for (row, chunk) in packed.chunks_exact_mut(row_bytes).enumerate() {
        q4_0::quantize(&random_vec(seed + row as u64, in_dim), chunk).unwrap();
    }
    let mut dequantized = vec![0.0f32; out_dim * in_dim];
    q4_0::dequantize(&packed, &mut dequantized).unwrap();
    Weights { codec: omega::Codec::Q4_0, packed, dequantized }
}

fn q6k_weights(in_dim: usize, out_dim: usize, seed: u64) -> Weights {
    let row_bytes = in_dim / q6_k::QK_K * q6_k::BLOCK_BYTES;
    let mut packed = vec![0u8; out_dim * row_bytes];
    for (row, chunk) in packed.chunks_exact_mut(row_bytes).enumerate() {
        q6_k::quantize(&random_vec(seed + row as u64, in_dim), chunk).unwrap();
    }
    let mut dequantized = vec![0.0f32; out_dim * in_dim];
    q6_k::dequantize(&packed, &mut dequantized).unwrap();
    Weights { codec: omega::Codec::Q6K, packed, dequantized }
}

fn assert_gated_matvec_matches_the_cpu_oracle(
    weights: &Weights,
    in_dim: usize,
    out_dim: usize,
    seed: u64,
) {
    let activation = random_vec(seed + 9973, in_dim);
    let gate = random_vec(seed + 31, out_dim);
    let (program, root) = gated_matvec_program(in_dim as u32, out_dim as u32);
    let blocks = [
        QuantizedBlock::Packed { codec: weights.codec, bytes: &weights.packed },
        QuantizedBlock::Float32(&activation),
        QuantizedBlock::Float32(&gate),
    ];
    let plan = omega::plan(&program, &[], &blocks, &[root], NumericPolicy::default())
        .expect("metal plans the gated packed matvec");
    let keys = plan.kernel_keys().expect("plan exposes its kernel keys");
    let output = omega::execute_plan(&plan, &blocks).expect("metal runs the matvec");

    assert!(
        keys.iter().any(|key| key.contains("epi") && key.contains("_al_e_0_0_0")),
        "the three mentions of the gate never fused into one epilogue with a repeated operand: {keys:?}"
    );
    assert_eq!(output.root().len(), out_dim, "degenerate gate: wrong element count");
    let expected: Vec<f32> = (0..out_dim)
        .map(|row| {
            let sum: f64 = (0..in_dim)
                .map(|position| {
                    f64::from(weights.dequantized[row * in_dim + position])
                        * f64::from(activation[position])
                })
                .sum();
            let value = f64::from(gate[row]);
            (sum * (value * value + value)) as f32
        })
        .collect();
    assert!(
        output.root().iter().all(|value| value.is_finite()),
        "K={in_dim} rows={out_dim}: a lane wrote nothing or garbage"
    );
    let max_diff = output
        .root()
        .iter()
        .zip(&expected)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max);
    let max_magnitude = expected.iter().map(|value| value.abs()).fold(0.0, f32::max);
    assert!(
        max_diff / max_magnitude < RELATIVE_TOLERANCE,
        "K={in_dim} rows={out_dim} codec={:?}: relative error {} against the oracle",
        weights.codec,
        max_diff / max_magnitude
    );
}

#[test]
fn q4_0_one_token_k1536_rows1536_every_lane_writes_its_own_row() {
    let weights = q4_0_weights(1536, 1536, 11);
    assert_gated_matvec_matches_the_cpu_oracle(&weights, 1536, 1536, 11);
}

#[test]
fn q4_0_one_token_k1536_rows6_masks_the_lanes_past_the_last_row() {
    let weights = q4_0_weights(1536, 6, 23);
    assert_gated_matvec_matches_the_cpu_oracle(&weights, 1536, 6, 23);
}

#[test]
fn q4_0_one_token_k256_rows5_leaves_a_partial_group_of_one_row() {
    let weights = q4_0_weights(256, 5, 37);
    assert_gated_matvec_matches_the_cpu_oracle(&weights, 256, 5, 37);
}

#[test]
fn q4_0_one_token_k256_rows3_fills_fewer_rows_than_one_group() {
    let weights = q4_0_weights(256, 3, 41);
    assert_gated_matvec_matches_the_cpu_oracle(&weights, 256, 3, 41);
}

#[test]
fn q6_k_one_token_k1024_rows7_folds_one_row_per_simdgroup() {
    let weights = q6k_weights(1024, 7, 53);
    assert_gated_matvec_matches_the_cpu_oracle(&weights, 1024, 7, 53);
}
