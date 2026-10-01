//! `push_packed_row_multi_row_q6k_body` (`omega/src/msl/elementwise_reduce_core.rs`)
//! folds a token group against ONE decode of each `Q6_K` super-block, on the
//! same lane map and per-lane accumulation order as the single-row ggml port
//! (`push_q6k_ggml_port_body`). This gate proves that equivalence on bytes:
//! every row of a `tokens`-wide matmul must equal, bit for bit (`to_bits()`),
//! the same matmul dispatched with a single token -- the single-row body --
//! on that token's activation row, over the same real-quantized `Q6_K` weights.
//! The tied gemma4-E2B LM head (`token_embd.weight`, `[1536, 262144]`) is the
//! production shape: K = 1536 reduces over six super-blocks per row.

#![cfg(all(
    feature = "metal",
    feature = "metal-q4k-ggml-port",
    not(feature = "metal-q4k-split-k"),
    target_os = "macos"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_gguf::quant::q6_k::{BLOCK_BYTES, QK_K, quantize};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    AxisTerm, DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce,
    ReduceInit, ScalarOp, affine, append, projection,
};

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

/// `[out_dim, in_dim] x [tokens, in_dim * stride] -> [tokens, out_dim]`,
/// reduced over `in_dim`, the activation read at `stride` elements per
/// reduction step. `stride == 1` is the contiguous read the production head
/// takes; larger values exercise the stride-multiplying address arm.
fn matmul_program(tokens: u32, in_dim: u32, out_dim: u32, stride: i32) -> (Vec<Op>, NodeId) {
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
            shape: vec![
                Extent::Static(tokens),
                Extent::Static(in_dim * stride as u32),
            ],
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
    (program, sum)
}

fn pack_rows(rows: &[Vec<f32>], in_dim: usize) -> Vec<u8> {
    let row_bytes = in_dim / QK_K * BLOCK_BYTES;
    let mut packed = vec![0u8; rows.len() * row_bytes];
    for (row, row_packed) in rows.iter().zip(packed.chunks_exact_mut(row_bytes)) {
        quantize(row, row_packed).unwrap();
    }
    packed
}

/// Spreads a dense `[tokens, in_dim]` activation to `[tokens, in_dim *
/// stride]`, poisoning every slot the stride skips with `NaN` so a read at the
/// wrong stride lands on a poisoned value instead of a plausible one.
fn spread_activation(dense: &[f32], tokens: usize, in_dim: usize, stride: usize) -> Vec<f32> {
    let mut spread = vec![f32::NAN; tokens * in_dim * stride];
    for token in 0..tokens {
        for position in 0..in_dim {
            spread[token * in_dim * stride + position * stride] = dense[token * in_dim + position];
        }
    }
    spread
}

fn run_bits(
    packed: &[u8],
    activation: &[f32],
    tokens: usize,
    in_dim: usize,
    out_dim: usize,
    stride: usize,
) -> Vec<u32> {
    let (program, sum) =
        matmul_program(tokens as u32, in_dim as u32, out_dim as u32, stride as i32);
    let blocks = [
        QuantizedBlock::Packed {
            codec: omega::Codec::Q6K,
            bytes: packed,
        },
        QuantizedBlock::Float32(activation),
    ];
    let plan = omega::plan(&program, &[], &blocks, &[sum], NumericPolicy::default())
        .expect("metal plans the Q6_K matmul");
    let output =
        omega::execute_plan(&plan, &blocks).expect("metal runs the matmul on a real device");
    output.root().iter().map(|value| value.to_bits()).collect()
}

struct Case {
    tokens: usize,
    in_dim: usize,
    out_dim: usize,
    stride: usize,
    seed: u64,
}

fn assert_each_token_row_matches_the_single_row_body(case: &Case) {
    let Case {
        tokens,
        in_dim,
        out_dim,
        stride,
        seed,
    } = *case;
    let rows: Vec<Vec<f32>> = (0..out_dim)
        .map(|row| random_vec(seed + row as u64, in_dim))
        .collect();
    let packed = pack_rows(&rows, in_dim);
    let dense = random_vec(seed + 9973, tokens * in_dim);

    let multi = run_bits(
        &packed,
        &spread_activation(&dense, tokens, in_dim, stride),
        tokens,
        in_dim,
        out_dim,
        stride,
    );
    assert_eq!(
        multi.len(),
        tokens * out_dim,
        "degenerate gate: tokens={tokens} K={in_dim} rows={out_dim} produced the wrong element count"
    );

    for token in 0..tokens {
        let single_activation = spread_activation(
            &dense[token * in_dim..(token + 1) * in_dim],
            1,
            in_dim,
            stride,
        );
        let single = run_bits(&packed, &single_activation, 1, in_dim, out_dim, stride);
        assert_eq!(
            &multi[token * out_dim..(token + 1) * out_dim],
            single.as_slice(),
            "tokens={tokens} K={in_dim} rows={out_dim} stride={stride}: token {token} of the \
             multi-row dispatch differs from the single-row body on the same activation row"
        );
    }
}

#[test]
fn tokens2_k1536_rows4096_match_single_row() {
    assert_each_token_row_matches_the_single_row_body(&Case {
        tokens: 2,
        in_dim: 1536,
        out_dim: 4096,
        stride: 1,
        seed: 11,
    });
}

#[test]
fn tokens3_k1536_rows4096_match_single_row_through_a_padded_cap() {
    assert_each_token_row_matches_the_single_row_body(&Case {
        tokens: 3,
        in_dim: 1536,
        out_dim: 4096,
        stride: 1,
        seed: 23,
    });
}

#[test]
fn tokens4_k1536_rows2048_match_single_row() {
    assert_each_token_row_matches_the_single_row_body(&Case {
        tokens: 4,
        in_dim: 1536,
        out_dim: 2048,
        stride: 1,
        seed: 37,
    });
}

#[test]
fn tokens8_k1536_rows2048_match_single_row() {
    assert_each_token_row_matches_the_single_row_body(&Case {
        tokens: 8,
        in_dim: 1536,
        out_dim: 2048,
        stride: 1,
        seed: 41,
    });
}

#[test]
fn tokens9_k1536_rows1024_match_single_row_across_a_partial_second_group() {
    assert_each_token_row_matches_the_single_row_body(&Case {
        tokens: 9,
        in_dim: 1536,
        out_dim: 1024,
        stride: 1,
        seed: 53,
    });
}

#[test]
fn tokens2_k6144_rows1536_match_single_row_over_twenty_four_super_blocks() {
    assert_each_token_row_matches_the_single_row_body(&Case {
        tokens: 2,
        in_dim: 6144,
        out_dim: 1536,
        stride: 1,
        seed: 59,
    });
}

#[test]
fn tokens2_k256_rows64_match_single_row_with_one_super_block_per_row() {
    assert_each_token_row_matches_the_single_row_body(&Case {
        tokens: 2,
        in_dim: 256,
        out_dim: 64,
        stride: 1,
        seed: 61,
    });
}

#[test]
fn tokens3_k1536_rows512_strided_activation_match_single_row() {
    assert_each_token_row_matches_the_single_row_body(&Case {
        tokens: 3,
        in_dim: 1536,
        out_dim: 512,
        stride: 2,
        seed: 67,
    });
}

#[test]
fn distinct_activation_rows_produce_distinct_output_rows() {
    let (tokens, in_dim, out_dim) = (2, 1536, 256);
    let rows: Vec<Vec<f32>> = (0..out_dim)
        .map(|row| random_vec(71 + row as u64, in_dim))
        .collect();
    let packed = pack_rows(&rows, in_dim);
    let dense = random_vec(71 + 9973, tokens * in_dim);

    let multi = run_bits(&packed, &dense, tokens, in_dim, out_dim, 1);

    assert_ne!(
        &multi[..out_dim],
        &multi[out_dim..],
        "control: two different activation rows must not produce the same output row, \
         or the per-token comparison above cannot tell tokens apart"
    );
}
