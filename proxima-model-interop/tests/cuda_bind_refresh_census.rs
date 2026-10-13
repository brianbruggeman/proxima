//! Measures how much of a KV step an incremental CUDA re-bind could reuse.
//! Binds the real gemma4-shaped two-range decode program at KV 6 and KV 7 with
//! the CUDA driver's bind configuration (`fuse_cached_attention = false`), marks
//! the bound ops whose inputs changed shape with `touches_changed`, and runs
//! `refresh_bound_ops` on each of them. Prints the per-variant refusal counts.

#![cfg(feature = "cuda")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, BTreeSet};

use proxima_tensor::bind::{RefreshRefusal, refresh_bound_ops, touches_changed};
use proxima_tensor::spec::{
    Activation, AttentionScoreScale, EmbeddingScale, ExpertGatingFunc, FfnCombination,
    KeySourceKind, LayerAttentionConfig, LayerFfnConfig, LayerKind, LayerSchedule,
    ParallelDenseMoeConfig, RopePairing, RopeTableSel, ValueSourceKind,
    scheduled_two_range_cached_forward_program_with_experts,
};
use proxima_tensor::{
    BoundOp, NodeId, NumericPolicy, Op, Shapes, bind_with_fusion, infer, prune_dead,
};

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

fn shapes_at(program: &[Op], kv_length: u64) -> Shapes {
    infer(program, &[1, kv_length]).expect("decode program infers at the kv length")
}

fn bind_resolved(program: &[Op], shapes: &Shapes, logits: NodeId) -> Vec<BoundOp> {
    let bound = bind_with_fusion(program, shapes, &[logits], false, NumericPolicy::default())
        .expect("decode program binds at the kv length");
    prune_dead(bound, &[logits])
}

fn changed_nodes(previous: &Shapes, next: &Shapes, program_len: usize) -> BTreeSet<NodeId> {
    (0..program_len)
        .map(|position| NodeId(u32::try_from(position).expect("program fits in u32 nodes")))
        .filter(|node| previous.of(*node) != next.of(*node))
        .collect()
}

fn kind_name(bound: &BoundOp) -> String {
    format!("{:?}", bound.kind)
        .chars()
        .take_while(|character| character.is_alphanumeric())
        .collect()
}

fn refusal_name(refusal: &RefreshRefusal) -> &'static str {
    match refusal {
        RefreshRefusal::PositionOutOfRange { .. } => "PositionOutOfRange",
        RefreshRefusal::MatcherFusedKind { .. } => "MatcherFusedKind",
        RefreshRefusal::EpilogueFusionNotLocal { .. } => "EpilogueFusionNotLocal",
        RefreshRefusal::OpKindMismatch { .. } => "OpKindMismatch",
        RefreshRefusal::RecomposedBodyMismatch { .. } => "RecomposedBodyMismatch",
        RefreshRefusal::RecomposedOperandsDiffer { .. } => "RecomposedOperandsDiffer",
        RefreshRefusal::ShapeError { .. } => "ShapeError",
    }
}

#[derive(Debug, Default)]
struct Census {
    touched_kinds: BTreeMap<String, usize>,
    refreshed: usize,
    refusals: BTreeMap<&'static str, (usize, String)>,
}

fn take_census(
    resolved: &[BoundOp],
    program: &[Op],
    outputs: &[NodeId],
    next: &Shapes,
    touched: &[usize],
) -> Census {
    let mut census = Census::default();
    for &position in touched {
        let mut masks = vec![0_u64; resolved.len()];
        masks[position] = 1;
        match refresh_bound_ops(
            resolved,
            program,
            outputs,
            next,
            &masks,
            1,
            NumericPolicy::default(),
        ) {
            Ok(_) => census.refreshed += 1,
            Err(refusal) => {
                let entry = census
                    .refusals
                    .entry(refusal_name(&refusal))
                    .or_insert((0, refusal.to_string()));
                entry.0 += 1;
            }
        }
        *census
            .touched_kinds
            .entry(kind_name(&resolved[position]))
            .or_default() += 1;
    }
    census
}

#[test]
fn cuda_bind_refresh_census_across_kv_6_to_7() {
    let (program, logits) = decode_program();
    let previous = shapes_at(&program, 6);
    let next = shapes_at(&program, 7);
    let resolved = bind_resolved(&program, &previous, logits);
    let changed = changed_nodes(&previous, &next, program.len());
    let touched: Vec<usize> = resolved
        .iter()
        .enumerate()
        .filter(|(_, bound)| touches_changed(bound, &program, &changed))
        .map(|(position, _)| position)
        .collect();

    let census = take_census(&resolved, &program, &[logits], &next, &touched);

    let refused: usize = census.refusals.values().map(|(count, _)| count).sum();
    println!(
        "census kv 6->7: total={} touched={} refreshed={} refused={}",
        resolved.len(),
        touched.len(),
        census.refreshed,
        refused
    );
    for (kind, count) in &census.touched_kinds {
        println!("touched kind {kind}: {count}");
    }
    for (name, (count, first)) in &census.refusals {
        println!("refusal {name}: {count} (first: {first})");
    }
    assert!(
        !touched.is_empty(),
        "no op touched by a kv 6->7 shape change"
    );
    assert!(
        touched.len() < resolved.len(),
        "every op touched although only kv-sized ops changed shape"
    );
    assert_eq!(
        census.refreshed + refused,
        touched.len(),
        "every touched op is either refreshed or refused"
    );
}
