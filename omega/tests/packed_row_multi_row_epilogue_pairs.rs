//! `push_packed_row_multi_row_body` writes a `cap`-token by `rows`-row
//! accumulator group back with one lane per `(token, row)` pair, each lane
//! running the fused reduce epilogue for its own output (`push_multi_row_lane_
//! epilogue`, `omega/src/msl/elementwise_reduce_core.rs`). This gate fuses a
//! per-feature scale into a multi-token packed matmul and holds every output
//! against the f32 CPU oracle at the tolerance the packed multi-row arms use:
//! the pair count spans two to thirty-two (fewer pairs than lanes, exactly one
//! round of lanes), a row count that leaves a partial feature group masks the
//! tail pairs, and `Q6_K` (one row per simdgroup) exercises a pair index that
//! is the token slot alone.

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

/// `[out_dim, in_dim] x [tokens, in_dim] -> [tokens, out_dim]`, reduced over
/// `in_dim`, then multiplied by a per-feature scale that `bind` fuses into the
/// reduce as its epilogue.
fn scaled_matmul_program(tokens: u32, in_dim: u32, out_dim: u32) -> (Vec<Op>, NodeId) {
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
            shape: vec![Extent::Static(tokens), Extent::Static(in_dim)],
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
                (activation, IndexMap::Affine(projection(3, &[0, 2]))),
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

fn assert_fused_epilogue_matches_the_cpu_oracle(
    weights: &Weights,
    tokens: usize,
    in_dim: usize,
    out_dim: usize,
    seed: u64,
) {
    let activation = random_vec(seed + 9973, tokens * in_dim);
    let scale = random_vec(seed + 31, out_dim);
    let (program, root) = scaled_matmul_program(tokens as u32, in_dim as u32, out_dim as u32);
    let blocks = [
        QuantizedBlock::Packed { codec: weights.codec, bytes: &weights.packed },
        QuantizedBlock::Float32(&activation),
        QuantizedBlock::Float32(&scale),
    ];
    let (output, kernel_keys) = temp_env::with_var("PROXIMA_TILED_GEMM_Q4_0", Some("0"), || {
        let plan = omega::plan(&program, &[], &blocks, &[root], NumericPolicy::default())
            .expect("metal plans the scaled packed matmul");
        let keys = plan.kernel_keys().expect("plan exposes its kernel keys");
        let output = omega::execute_plan(&plan, &blocks).expect("metal runs the matmul");
        (output.root().to_vec(), keys)
    });

    assert!(
        kernel_keys.iter().any(|key| key.contains("epi")),
        "the scale never fused into the reduce, so no epilogue write-back ran: {kernel_keys:?}"
    );
    assert_eq!(output.len(), tokens * out_dim, "degenerate gate: wrong element count");
    let expected: Vec<f32> = (0..tokens * out_dim)
        .map(|index| {
            let (token, row) = (index / out_dim, index % out_dim);
            let sum: f64 = (0..in_dim)
                .map(|position| {
                    f64::from(weights.dequantized[row * in_dim + position])
                        * f64::from(activation[token * in_dim + position])
                })
                .sum();
            (sum * f64::from(scale[row])) as f32
        })
        .collect();
    let max_diff = output.iter().zip(&expected).map(|(left, right)| (left - right).abs()).fold(0.0, f32::max);
    let max_magnitude = expected.iter().map(|value| value.abs()).fold(0.0, f32::max);
    assert!(
        max_diff / max_magnitude < RELATIVE_TOLERANCE,
        "tokens={tokens} K={in_dim} rows={out_dim} codec={:?}: relative error {} against the oracle",
        weights.codec,
        max_diff / max_magnitude
    );
}

#[test]
fn q4_0_tokens2_k1536_rows1536_two_pairs_per_row_group() {
    let weights = q4_0_weights(1536, 1536, 11);
    assert_fused_epilogue_matches_the_cpu_oracle(&weights, 2, 1536, 1536, 11);
}

#[test]
fn q4_0_tokens4_k1536_rows1536_sixteen_pairs() {
    let weights = q4_0_weights(1536, 1536, 23);
    assert_fused_epilogue_matches_the_cpu_oracle(&weights, 4, 1536, 1536, 23);
}

#[test]
fn q4_0_tokens8_k1536_rows512_thirty_two_pairs_fill_one_round_of_lanes() {
    let weights = q4_0_weights(1536, 512, 37);
    assert_fused_epilogue_matches_the_cpu_oracle(&weights, 8, 1536, 512, 37);
}

#[test]
fn q4_0_tokens3_k1536_rows6_masks_the_pairs_past_the_last_row() {
    let weights = q4_0_weights(1536, 6, 41);
    assert_fused_epilogue_matches_the_cpu_oracle(&weights, 3, 1536, 6, 41);
}

#[test]
fn q4_0_tokens9_k1536_rows256_second_token_group_is_partial() {
    let weights = q4_0_weights(1536, 256, 47);
    assert_fused_epilogue_matches_the_cpu_oracle(&weights, 9, 1536, 256, 47);
}

#[test]
fn q6k_tokens4_k1536_rows512_one_row_per_simdgroup() {
    let weights = q6k_weights(1536, 512, 53);
    assert_fused_epilogue_matches_the_cpu_oracle(&weights, 4, 1536, 512, 53);
}

#[test]
fn q6k_tokens8_k1536_rows64_pair_index_is_the_token_slot() {
    let weights = q6k_weights(1536, 64, 59);
    assert_fused_epilogue_matches_the_cpu_oracle(&weights, 8, 1536, 64, 59);
}
