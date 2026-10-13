//! A CUDA refit across KV lengths re-emits source only for ops whose bound op
//! changed. Builds the real gemma4-shaped two-range decode program (the same
//! dual-RoPE schedule `gemma4_program_metal_cpu_parity.rs` documents), binds it
//! at two KV lengths, and checks the reuse path against a fresh emit for every op.

#![cfg(feature = "cuda")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use omega::{CudaKernel, PackedOperands, emit_cuda_reusing, emit_cuda_with_policy};
use proxima_tensor::spec::{
    Activation, AttentionScoreScale, EmbeddingScale, ExpertGatingFunc, FfnCombination,
    KeySourceKind, LayerAttentionConfig, LayerFfnConfig, LayerKind, LayerSchedule,
    ParallelDenseMoeConfig, RopePairing, RopeTableSel, ValueSourceKind,
    scheduled_two_range_cached_forward_program_with_experts,
};
use proxima_tensor::{BoundOp, NodeId, NumericPolicy, Op, bind_with_fusion, infer, prune_dead};

const VOCAB: u32 = 6;
const EMBEDDING: u32 = 16;
const FEED_FORWARD: u32 = 8;
const EXPERT_FEED_FORWARD: u32 = 8;
const QUERY_HEADS: u32 = 2;
const KV_HEADS: u32 = 1;
const EXPERT_COUNT: u32 = 4;
const EXPERT_USED_COUNT: u32 = 2;
const HEAD_DIM_FULL: u32 = 512;
const HEAD_DIM_SWA: u32 = 256;
const SLIDING_WINDOW: u32 = 2;
const SLIDING_PATTERN: [bool; 4] = [true, false, true, false];

fn gemma4_schedule(sliding_pattern: &[bool]) -> Vec<LayerSchedule> {
    let ffn = LayerFfnConfig {
        post_attention_norm: true,
        combination: FfnCombination::ParallelDenseMoe(ParallelDenseMoeConfig {
            dense_post_norm: true,
            routed_post_norm: true,
            combined_post_norm: true,
            routed_pre_norm: true,
            router_scale: true,
            expert_output_scale: true,
        }),
        output_scale: true,
        routed_gating: ExpertGatingFunc::Softmax,
        routed_expert_bias: false,
        dense_feed_forward: None,
        exclusive_dense_post_norm: false,
        activation: Activation::GeluTanh,
        ple: false,
    };
    sliding_pattern
        .iter()
        .map(|&is_sliding| {
            let attention = if is_sliding {
                LayerAttentionConfig {
                    head_dim: HEAD_DIM_SWA,
                    kv_heads: KV_HEADS,
                    mask_window: Some(SLIDING_WINDOW),
                    value_source_kind: ValueSourceKind::ProjectedV,
                    key_source_kind: KeySourceKind::ProjectedK,
                    rope_table: RopeTableSel {
                        cos_name: "rope_cos_swa".into(),
                        sin_name: "rope_sin_swa".into(),
                    },
                    rope_pairing: RopePairing::SplitHalf {
                        pairs: HEAD_DIM_SWA / 2,
                    },
                    score_scale: AttentionScoreScale::Unscaled,
                    value_norm: true,
                    qk_norm: true,
                }
            } else {
                LayerAttentionConfig {
                    head_dim: HEAD_DIM_FULL,
                    kv_heads: KV_HEADS,
                    mask_window: None,
                    value_source_kind: ValueSourceKind::SharedWithKey,
                    key_source_kind: KeySourceKind::ProjectedK,
                    rope_table: RopeTableSel {
                        cos_name: "rope_cos".into(),
                        sin_name: "rope_sin".into(),
                    },
                    rope_pairing: RopePairing::SplitHalf {
                        pairs: HEAD_DIM_FULL / 2,
                    },
                    score_scale: AttentionScoreScale::Unscaled,
                    value_norm: true,
                    qk_norm: true,
                }
            };
            LayerSchedule {
                kind: LayerKind::Attention,
                attention,
                ffn,
            }
        })
        .collect()
}

fn decode_program() -> (Vec<Op>, NodeId) {
    let schedule = gemma4_schedule(&SLIDING_PATTERN);
    let (program, logits, _layer_roots, _moe_sites, _head_repeats) =
        scheduled_two_range_cached_forward_program_with_experts(
            VOCAB,
            EMBEDDING,
            FEED_FORWARD,
            EXPERT_FEED_FORWARD,
            QUERY_HEADS,
            SLIDING_PATTERN.len() as u32,
            EXPERT_COUNT,
            EXPERT_USED_COUNT,
            0,
            &schedule,
            Some(EmbeddingScale::Sqrt),
            None,
            true,
            None,
            false,
        )
        .expect("the gemma4-shaped two-range decode program lowers at the synthetic width");
    (program, logits)
}

fn bound_at(program: &[Op], logits: NodeId, kv_length: u64) -> Vec<BoundOp> {
    let shapes = infer(program, &[1, kv_length]).expect("decode program infers at the kv length");
    let bound = bind_with_fusion(program, &shapes, &[logits], false, NumericPolicy::default())
        .expect("decode program binds at the kv length");
    prune_dead(bound, &[logits])
}

fn fresh_kernels(bound: &[BoundOp], packed: &PackedOperands) -> Vec<CudaKernel> {
    bound
        .iter()
        .map(|op| emit_cuda_with_policy(op, packed, NumericPolicy::default()))
        .collect::<Result<_, _>>()
        .expect("every bound op of the decode program emits cuda source")
}

#[test]
fn reused_cuda_source_equals_a_fresh_emit_for_every_op_across_kv_lengths() {
    let (program, logits) = decode_program();
    let packed = PackedOperands::new();
    let before = bound_at(&program, logits, 6);
    let after = bound_at(&program, logits, 7);
    let previous_kernels = fresh_kernels(&before, &packed);

    let (reused_kernels, reused_count) = emit_cuda_reusing(
        &before,
        previous_kernels,
        &after,
        &packed,
        NumericPolicy::default(),
    )
    .expect("the reuse path emits the decode program");

    let fresh = fresh_kernels(&after, &packed);
    assert_eq!(reused_kernels.len(), fresh.len(), "one kernel per bound op");
    assert_eq!(
        reused_kernels, fresh,
        "reused source differs from a fresh emit"
    );
    assert!(
        reused_count > 0,
        "no op was skipped across kv lengths 6 -> 7"
    );
    assert!(
        reused_count < after.len(),
        "every op was skipped although kv-sized ops changed extents"
    );
    println!("ops={} reused={reused_count}", after.len());
}

#[test]
fn nothing_is_reused_when_the_previous_binding_is_empty() {
    let (program, logits) = decode_program();
    let packed = PackedOperands::new();
    let after = bound_at(&program, logits, 7);

    let (kernels, reused_count) =
        emit_cuda_reusing(&[], Vec::new(), &after, &packed, NumericPolicy::default())
            .expect("a cold emit succeeds");

    assert_eq!(reused_count, 0);
    assert_eq!(kernels, fresh_kernels(&after, &packed));
}
