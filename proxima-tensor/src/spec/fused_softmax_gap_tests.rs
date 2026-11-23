//! A logits-only two-layer cached model binds the fused `CachedSoftmaxWeights`; asking for every
//! node binds the unfused chain. Both are the same f32 softmax, so they must agree to ulps.

use super::*;
use crate::test_support::{Lcg, evaluate_named_online_f32};

const VOCAB: u32 = 5;
const EMBEDDING: u32 = 8;
const FEED_FORWARD: u32 = 6;
const KV_HEADS: u32 = 2;
const GROUP: u32 = 2;
const HEAD_DIM: u32 = 4;
const NEW_TOKENS: u32 = 2;
const CACHED_TOKENS: u32 = 3;
const LAYERS: u32 = 2;
const THETA: f32 = 10000.0;

fn rope_tables(first_position: f32) -> (Vec<f32>, Vec<f32>) {
    let positions: Vec<f32> = (0..NEW_TOKENS)
        .map(|row| first_position + row as f32)
        .collect();
    let (cos, sin) = multi_axis_rope_tables(&positions, &[HEAD_DIM], &[THETA], DType::FLOAT32)
        .expect("rope tables build from real positions and theta");
    let decode = |bytes: &[u8]| -> Vec<f32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect()
    };
    (decode(&cos), decode(&sin))
}

fn model_schedule(pairing: RopePairing) -> Vec<LayerSchedule> {
    let attention = LayerAttentionConfig {
        head_dim: HEAD_DIM,
        kv_heads: KV_HEADS,
        mask_window: None,
        value_source_kind: ValueSourceKind::ProjectedV,
        key_source_kind: KeySourceKind::ProjectedK,
        rope_table: RopeTableSel {
            cos_name: "rope_cos".into(),
            sin_name: "rope_sin".into(),
        },
        rope_pairing: pairing,
        score_scale: AttentionScoreScale::Unscaled,
        value_norm: false,
        qk_norm: false,
        attention_sinks: false,
        rope: true,
    };
    (0..LAYERS)
        .map(|_| LayerSchedule {
            kind: LayerKind::Attention,
            attention: attention.clone(),
            ffn: LayerFfnConfig::exclusive(),
        })
        .collect()
}

fn model_program() -> (Vec<Op>, NodeId) {
    let (program, logits, ..) = scheduled_two_range_cached_forward_program_with_layer_roots(
        VOCAB,
        EMBEDDING,
        FEED_FORWARD,
        FEED_FORWARD,
        KV_HEADS * GROUP,
        LAYERS,
        0,
        0,
        LAYERS,
        &model_schedule(RopePairing::Interleaved),
        None,
        None,
        false,
        false,
        None,
        false,
        0,
        1,
        None,
        LoopCdConfig::default(),
        SharedExpertWidth::NONE,
        DType::FLOAT32,
        KeyCacheLayout::Planes,
    None,)
    .expect("a two-layer cached model builds");
    (program, logits)
}

fn extent_value(extent: &Extent) -> usize {
    match extent {
        Extent::Static(value) => *value as usize,
        Extent::Symbolic(0) => NEW_TOKENS as usize,
        Extent::Symbolic(_) => CACHED_TOKENS as usize,
    }
}

fn model_values(name: &str, shape: &[Extent]) -> Vec<f32> {
    let count: usize = shape.iter().map(extent_value).product();
    match name {
        "ids" => (0..count).map(|row| (row + 1) as f32).collect(),
        "cached_len" => alloc::vec![CACHED_TOKENS as f32],
        "eps" => alloc::vec![1e-6; count],
        "rope_cos" => rope_tables(CACHED_TOKENS as f32).0,
        "rope_sin" => rope_tables(CACHED_TOKENS as f32).1,
        _ => {
            let seed = name.bytes().fold(0x6262_0001u64, |acc, byte| {
                acc.wrapping_mul(31).wrapping_add(u64::from(byte))
            });
            let mut lcg = Lcg(seed);
            (0..count).map(|_| lcg.next_unit()).collect()
        }
    }
}

fn evaluate(program: &[Op], outputs: &[NodeId]) -> crate::Evaluated {
    let owned: Vec<(String, Vec<f32>)> = program
        .iter()
        .filter_map(|op| match op {
            Op::Input {
                shape,
                name: Some(name),
                ..
            } => Some((name.clone(), model_values(name, shape))),
            _ => None,
        })
        .collect();
    let named: Vec<(&str, &[f32])> = owned
        .iter()
        .map(|(name, values)| (name.as_str(), values.as_slice()))
        .collect();
    let symbols = [u64::from(NEW_TOKENS), u64::from(CACHED_TOKENS)];
    evaluate_named_online_f32(program, &symbols, &named, outputs).expect("the model evaluates")
}

fn all_float32_nodes(program: &[Op]) -> Vec<NodeId> {
    (0..program.len() as u32)
        .map(NodeId)
        .filter(|node| program[node.0 as usize].dtype() == DType::FLOAT32)
        .collect()
}

#[proxima::test]
async fn two_kv_head_logits_only_plan_matches_the_unfused_plan_within_rounding() {
    let (program, logits) = model_program();

    let logits_only = evaluate(&program, &[logits])
        .get(logits)
        .expect("logits requested")
        .0
        .to_vec();
    let every_node = evaluate(&program, &all_float32_nodes(&program))
        .get(logits)
        .expect("logits requested")
        .0
        .to_vec();

    assert_eq!(logits_only.len(), (NEW_TOKENS * VOCAB) as usize);
    let worst = logits_only
        .iter()
        .zip(&every_node)
        .map(|(fused, unfused)| (fused - unfused).abs() / unfused.abs().max(1e-3))
        .fold(0.0f32, f32::max);
    assert!(
        worst < 1e-4,
        "worst relative gap {worst}: logits-only {logits_only:?} every-node {every_node:?}"
    );
}
