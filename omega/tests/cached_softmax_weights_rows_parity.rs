//! The K-row `CachedSoftmaxWeights` lowering against the unfused chain it
//! replaces, on the device. Under `bit_exact` the recognizer binds the
//! gemma4-shaped two-range program's attention as `CachedSoftmaxWeights` at K
//! new rows (`metal-attn-split-rows`); the same program planned with the
//! cached-attention fusion off runs the literal 13-op chain per layer.
//!
//! What is asserted byte for byte is what the K-row kernel owns, on the first
//! layer where both plans feed the op identical inputs: the primary
//! `[query row, key, row]` weights, `new_weight_sum` and `new_attended`. The
//! kernel folds the new keys left to right, the order of the chain's serial
//! reduce under `bit_exact`. That order is the chain's only while its weighted
//! value reduce is not admitted as a tiled GEMM: from `TILED_GEMM_MIN_TOKENS`
//! rows on the chain accumulates in tiles, and a left fold can differ from it
//! in the last bits, so those cells bound `new_attended` to
//! `NEW_ATTENDED_TOLERANCE` instead. `cached_weight_sum` is the cached fold the
//! K=1 kernel already ships (a lane tree against the chain's left fold), and
//! the device does not reproduce the chain's bytes there at K=1 either (the
//! `rows 1` cells are that control), so it is bounded to `CACHED_SUM_TOLERANCE`.
//! The logits of the whole two-layer program are bounded to `LOGITS_TOLERANCE`.
//!
//! The CPU half (the K-row oracle against the CPU chain) is
//! `proxima-tensor`'s `the_k_row_softmax_weights_oracle_equals_the_unfused_chain_exactly`.
//! Needs a Metal device.

#![cfg(all(
    feature = "metal",
    feature = "metal-attn-split-rows",
    target_os = "macos"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::{BoundOpKind, NodeId, NumericPolicy, bind_with_fusion, infer};

mod gemma4_rows_support;
mod support;
use gemma4_rows_support::{VOCAB, bucket_for, fixture};
use support::as_named_blocks;

const VERIFY_ROWS: [usize; 9] = [1, 2, 3, 4, 5, 8, 9, 17, 49];
const CACHED_LENGTHS: [usize; 3] = [33, 511, 1099];
const CACHED_SUM_TOLERANCE: f32 = 1e-5;
const NEW_ATTENDED_TOLERANCE: f32 = 1e-5;
const LOGITS_TOLERANCE: f32 = 1e-4;

#[cfg(feature = "metal-tiled-gemm")]
const CHAIN_TILES_FROM_ROWS: u64 = omega::sized::TILED_GEMM_MIN_TOKENS;
#[cfg(not(feature = "metal-tiled-gemm"))]
const CHAIN_TILES_FROM_ROWS: u64 = u64::MAX;

struct SoftmaxOutputs {
    weights: NodeId,
    cached_weight_sum: NodeId,
    new_weight_sum: NodeId,
    new_attended: NodeId,
}

fn first_softmax_outputs(resolved: &[proxima_tensor::BoundOp]) -> (usize, SoftmaxOutputs) {
    let fused: Vec<&proxima_tensor::BoundOp> = resolved
        .iter()
        .filter(|bound| matches!(bound.kind, BoundOpKind::CachedSoftmaxWeights { .. }))
        .collect();
    let first = fused.first().expect("the program binds a softmax weights op");
    let BoundOpKind::CachedSoftmaxWeights {
        cached_weight_sum,
        new_weight_sum,
        new_attended,
        ..
    } = &first.kind
    else {
        unreachable!("the filter keeps only CachedSoftmaxWeights");
    };
    (
        fused.len(),
        SoftmaxOutputs {
            weights: first.node,
            cached_weight_sum: *cached_weight_sum,
            new_weight_sum: *new_weight_sum,
            new_attended: *new_attended,
        },
    )
}

fn differing_bits(left: &[f32], right: &[f32]) -> usize {
    assert_eq!(left.len(), right.len());
    assert!(!left.is_empty(), "an empty output compares nothing");
    left.iter()
        .zip(right)
        .filter(|(left_value, right_value)| left_value.to_bits() != right_value.to_bits())
        .count()
}

fn relative_difference(expected: &[f32], actual: &[f32]) -> f32 {
    assert_eq!(expected.len(), actual.len());
    assert!(!expected.is_empty(), "an empty output compares nothing");
    let magnitude = expected.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
    let difference = expected
        .iter()
        .zip(actual)
        .map(|(want, got)| (want - got).abs())
        .fold(0.0f32, f32::max);
    difference / magnitude.max(f32::MIN_POSITIVE)
}

#[test]
fn the_fused_k_row_softmax_weights_return_the_unfused_chains_bytes() {
    let policy = NumericPolicy::bit_exact();
    let mut cells = 0_usize;

    for rows in VERIFY_ROWS {
        for cached_len in CACHED_LENGTHS {
            let fixture = fixture(rows, cached_len);
            let named = as_named_blocks(&fixture.named);
            let cell = format!(
                "rows {rows} cached_len {cached_len} bucket {}",
                bucket_for(cached_len)
            );

            let shapes = infer(&fixture.program, &fixture.symbols).expect("the program infers");
            let resolved =
                bind_with_fusion(&fixture.program, &shapes, &[fixture.logits], true, policy)
                    .expect("the program binds");
            let (fused_ops, outputs) = first_softmax_outputs(&resolved);
            assert_eq!(
                fused_ops, 2,
                "{cell}: both layers must bind the K-row softmax weights"
            );

            let roots = [
                fixture.logits,
                outputs.weights,
                outputs.cached_weight_sum,
                outputs.new_weight_sum,
                outputs.new_attended,
            ];
            let fused = omega::plan_named_with_placed_inputs(
                &fixture.program,
                &fixture.symbols,
                &named,
                &roots,
                policy,
                &[],
                true,
            )
            .expect("metal plans the fused program");
            let unfused = omega::plan_named_with_placed_inputs(
                &fixture.program,
                &fixture.symbols,
                &named,
                &roots,
                policy,
                &[],
                false,
            )
            .expect("metal plans the unfused program");
            let fused_result =
                omega::execute_plan_named(&fused, &named).expect("metal runs the fused program");
            let unfused_result = omega::execute_plan_named(&unfused, &named)
                .expect("metal runs the unfused program");
            let output = |result: &proxima_tensor::Evaluated, node: NodeId| -> Vec<f32> {
                result
                    .get(node)
                    .expect("the requested root was computed")
                    .0
                    .to_vec()
            };

            assert_eq!(
                fused_result.root().len(),
                rows * VOCAB as usize,
                "{cell}: one logits row per new token"
            );
            let chain_tiles = rows as u64 >= CHAIN_TILES_FROM_ROWS;
            let weights_bits = differing_bits(
                &output(&fused_result, outputs.weights),
                &output(&unfused_result, outputs.weights),
            );
            let new_sum_bits = differing_bits(
                &output(&fused_result, outputs.new_weight_sum),
                &output(&unfused_result, outputs.new_weight_sum),
            );
            let attended_bits = differing_bits(
                &output(&fused_result, outputs.new_attended),
                &output(&unfused_result, outputs.new_attended),
            );
            let attended_relative = relative_difference(
                &output(&unfused_result, outputs.new_attended),
                &output(&fused_result, outputs.new_attended),
            );
            let cached_sum_relative = relative_difference(
                &output(&unfused_result, outputs.cached_weight_sum),
                &output(&fused_result, outputs.cached_weight_sum),
            );
            let logits_relative =
                relative_difference(unfused_result.root(), fused_result.root());
            eprintln!(
                "softmax_weights rows parity: {cell} weights_differing={weights_bits} \
                 new_sum_differing={new_sum_bits} new_attended_differing={attended_bits} \
                 new_attended_relative={attended_relative:e} chain_tiles={chain_tiles} \
                 cached_sum_relative={cached_sum_relative:e} logits_relative={logits_relative:e}"
            );

            assert_eq!(weights_bits, 0, "{cell}: the weights differ in bytes");
            assert_eq!(new_sum_bits, 0, "{cell}: new_weight_sum differs in bytes");
            if chain_tiles {
                assert!(
                    attended_relative < NEW_ATTENDED_TOLERANCE,
                    "{cell}: new_attended relative difference {attended_relative}"
                );
            } else {
                assert_eq!(attended_bits, 0, "{cell}: new_attended differs in bytes");
            }
            assert!(
                cached_sum_relative < CACHED_SUM_TOLERANCE,
                "{cell}: cached_weight_sum relative difference {cached_sum_relative}"
            );
            assert!(
                logits_relative < LOGITS_TOLERANCE,
                "{cell}: logits relative difference {logits_relative}"
            );
            cells += 1;
        }
    }
    assert_eq!(cells, 27, "9 row counts x 3 cached lengths");
}
