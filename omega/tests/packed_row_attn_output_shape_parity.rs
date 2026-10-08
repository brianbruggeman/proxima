//! The decode attention-output projection of a decoder with per-layer inputs is a
//! Q4_0 matvec whose reduction folds two contiguous axes (kv heads x head
//! dim, `[1, 1, 8, 256, 1536]` in a decode census) and whose output carries
//! two unit axes. It takes the single-token packed-row body and, with no
//! fused epilogue, the lane-parallel plain write-back: lane `q` of a
//! simdgroup stores output row `q`. This gate holds every output of that
//! shape, and of a row count that leaves a partial row group, against an f64
//! dequantize-and-dot oracle.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_gguf::quant::q4_0;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce, ReduceInit,
    ScalarOp, append, projection,
};

const RELATIVE_TOLERANCE: f32 = 1e-5;
const KV_HEADS: u32 = 8;
const HEAD_DIM: u32 = 256;

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

/// `[rows, heads, head_dim] x [1, 1, heads, head_dim]` reduced over `heads`
/// and `head_dim`, iteration axes `(token, unit, heads, head_dim, rows)`.
fn attention_output_program(rows: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::UInt8,
            shape: vec![
                Extent::Static(rows),
                Extent::Static(KV_HEADS),
                Extent::Static(HEAD_DIM),
            ],
            name: None,
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![
                Extent::Static(1),
                Extent::Static(1),
                Extent::Static(KV_HEADS),
                Extent::Static(HEAD_DIM),
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
                (weight, IndexMap::Affine(projection(5, &[4, 2, 3]))),
                (activation, IndexMap::Affine(projection(5, &[0, 1, 2, 3]))),
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
            in_map: IndexMap::Affine(projection(5, &[0, 1, 2, 3, 4])),
            out_map: IndexMap::Affine(projection(5, &[0, 1, 4])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    (program, sum)
}

fn assert_attention_output_matches_the_cpu_oracle(rows: usize, seed: u64) {
    let in_dim = (KV_HEADS * HEAD_DIM) as usize;
    let row_bytes = in_dim / q4_0::QK4_0 * q4_0::BLOCK_BYTES;
    let mut packed = vec![0u8; rows * row_bytes];
    for (row, chunk) in packed.chunks_exact_mut(row_bytes).enumerate() {
        q4_0::quantize(&random_vec(seed + row as u64, in_dim), chunk).unwrap();
    }
    let mut dequantized = vec![0.0f32; rows * in_dim];
    q4_0::dequantize(&packed, &mut dequantized).unwrap();
    let activation = random_vec(seed + 9973, in_dim);

    let (program, root) = attention_output_program(rows as u32);
    let blocks = [
        QuantizedBlock::Packed { codec: omega::Codec::Q4_0, bytes: &packed },
        QuantizedBlock::Float32(&activation),
    ];
    let plan = omega::plan(&program, &[], &blocks, &[root], NumericPolicy::default())
        .expect("metal plans the attention-output matvec");
    let output = omega::execute_plan(&plan, &blocks).expect("metal runs the matvec");

    assert_eq!(output.root().len(), rows, "wrong element count for rows={rows}");
    assert!(
        output.root().iter().all(|value| value.is_finite()),
        "rows={rows}: a lane wrote nothing or garbage"
    );
    let expected: Vec<f32> = (0..rows)
        .map(|row| {
            (0..in_dim)
                .map(|position| {
                    f64::from(dequantized[row * in_dim + position]) * f64::from(activation[position])
                })
                .sum::<f64>() as f32
        })
        .collect();
    let max_diff = output
        .root()
        .iter()
        .zip(&expected)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max);
    let max_magnitude = expected.iter().map(|value| value.abs()).fold(0.0, f32::max);
    assert!(
        max_diff / max_magnitude < RELATIVE_TOLERANCE,
        "rows={rows}: relative error {} against the oracle",
        max_diff / max_magnitude
    );
}

#[test]
fn q4_0_attention_output_k2048_rows1536_every_lane_writes_its_own_row() {
    assert_attention_output_matches_the_cpu_oracle(1536, 101);
}

#[test]
fn q4_0_attention_output_k2048_rows6_masks_the_lanes_past_the_last_row() {
    assert_attention_output_matches_the_cpu_oracle(6, 211);
}

#[test]
fn q4_0_attention_output_k2048_rows4_fills_exactly_one_simdgroup() {
    assert_attention_output_matches_the_cpu_oracle(4, 307);
}
