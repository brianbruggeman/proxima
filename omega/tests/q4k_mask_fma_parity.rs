//! Correctness gate for `metal-q4k-mask-fma`, the mask-without-shift/
//! branch-free rewrite of the row-blocked packed path's Q4_K SCALE-DEFERRED
//! matvec body (`omega/src/msl.rs`'s `push_q4k_header_decode`/
//! `push_q4k_product_reduce_body`, `docs/discipline.md` row assigned
//! centrally). This is the "cover sub_blocks 0..7 explicitly" test the
//! landing's own gate requires -- `q4k_real_checkpoint_parity.rs` already
//! covers this rewrite incidentally at production scale (64 rows x 16
//! super-blocks/row), but ONE super-block here means every sub_block 0..7
//! and both nibble halves (elements 0..127 low, 128..255 high) fire on
//! EVERY row, deterministically, with no dependency on a checkpoint file
//! being present on the host.
//!
//! Only compiles under the feature it gates -- there is nothing to test
//! here without it, and the unpatched path already has this same coverage
//! shape via `q4k_row_blocked_matmul_defers_scale_to_once_per_sub_block`
//! (a codegen-source-text check, not a device-execution one) plus every
//! `q4k_real_checkpoint_parity.rs`/`metal_parity.rs` run today.

#![cfg(all(feature = "metal-q4k-mask-fma", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, dequantize, quantize};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp,
    append, evaluate, map,
};

fn random_vec(seed: u64, count: usize, spread: f32) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count)
        .map(|_| (lcg.next_unit() * 2.0 - 1.0) * spread)
        .collect()
}

/// `[rows, k] x [k, 1] -> [rows, 1]`, `weight_dtype` distinguishing "packed
/// bytes" (`UInt8`) from the dequantized `f32` oracle -- the same shape
/// `q4k_real_checkpoint_parity.rs`'s own `matmul_program` builds, restated
/// here since this is a standalone integration test binary.
fn matmul_program(rows: u32, k: u32, weight_dtype: DType) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: weight_dtype,
            shape: vec![Extent::Static(rows), Extent::Static(k)],
            name: None,
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(k), Extent::Static(1)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (activation, IndexMap::Affine(map::projection(3, &[2, 1]))),
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
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("q4k_mask_fma_matmul".into()),
        }),
    );
    (program, sum)
}

/// Exactly `QK_K` (256) -- one Q4_K super-block per row, so every one of the
/// codec's 8 `sub_block` values and both nibble halves fire on every row of
/// this test, not just somewhere across a large sweep.
const IN_DIM: usize = QK_K;
/// `2 * PACKED_ROWS_PER_GROUP` (`msl.rs`) -- two full row-blocked SIMD
/// groups, so the fixture also exercises the group boundary, not only a
/// single partial group.
const OUT_DIM: usize = 8;

#[test]
fn masked_fma_q4k_matvec_matches_the_dequantized_f32_cpu_path_across_every_sub_block() {
    let rows: Vec<Vec<f32>> = (0..OUT_DIM)
        .map(|row| random_vec(4_100 + row as u64, IN_DIM, 6.0))
        .collect();

    // a runtime binding, not the `const IN_DIM` directly: clippy's
    // `chunks_exact_to_as_chunks` fires on a compile-time-constant chunk
    // size and suggests `as_chunks_mut::<IN_DIM>()`, which would pin this
    // fixture's row width into the slice's TYPE -- the same posture
    // `q4k_real_checkpoint_parity.rs`'s own `in_dim: usize` parameter
    // already takes for exactly this reason.
    let in_dim = IN_DIM;
    let blocks_per_row = in_dim / QK_K;
    assert_eq!(
        blocks_per_row, 1,
        "fixture must be exactly one super-block per row"
    );
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    let mut packed = vec![0u8; OUT_DIM * row_bytes];
    for (row, row_packed) in rows.iter().zip(packed.chunks_exact_mut(row_bytes)) {
        quantize(row, row_packed).expect("IN_DIM is a whole multiple of QK_K");
    }

    let mut dequantized = vec![0.0f32; OUT_DIM * in_dim];
    for (row_packed, row_f32) in packed
        .chunks_exact(row_bytes)
        .zip(dequantized.chunks_exact_mut(in_dim))
    {
        dequantize(row_packed, row_f32).expect("packed row dequantizes");
    }

    let activation = random_vec(97, in_dim, 3.0);

    let (packed_program, packed_sum) = matmul_program(OUT_DIM as u32, IN_DIM as u32, DType::UInt8);
    let metal = omega::execute(
        &packed_program,
        &[],
        &[
            QuantizedBlock::Q4K(&packed),
            QuantizedBlock::Float32(&activation),
        ],
        &[packed_sum],
    )
    .expect("metal executes the masked-fma q4_k matmul");

    let (f32_program, f32_sum) = matmul_program(OUT_DIM as u32, IN_DIM as u32, DType::Float32);
    let cpu = evaluate(&f32_program, &[], &[&dequantized, &activation], &[f32_sum])
        .expect("dequantized f32 cpu matmul evaluates");

    let actual = metal.root();
    let expected = cpu.root();
    assert_eq!(
        actual.len(),
        OUT_DIM,
        "degenerate gate: no outputs compared"
    );
    assert_eq!(actual.len(), expected.len());

    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(
            got.is_finite(),
            "masked-fma kernel produced a non-finite value: {got}"
        );
        max_diff = max_diff.max((got - want).abs());
    }
    // batch-peak normalization, not per-row relative error: a per-row
    // denominator explodes at zero crossings and has produced a false
    // "872%" bug report on this codebase before
    // (`feedback_per_row_relative_error_explodes_at_zero_crossings`).
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let relative = max_diff / max_magnitude;
    eprintln!(
        "masked-fma q4_k matvec ({OUT_DIM} rows, one super-block/row, k={IN_DIM}) vs dequantized-f32 cpu: \
         max_diff={max_diff} max_magnitude={max_magnitude} relative={relative}"
    );
    assert!(
        relative < 1e-5,
        "mask-without-shift/branch-free rewrite disagrees with the dequantized reference: \
         relative={relative} max_diff={max_diff}"
    );
}
