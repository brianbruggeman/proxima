//! `append_two_range_cached_attention` over a one-row key cache against the same
//! keys held in two planes: every score is the same products summed in the same
//! `e0,o0,e1,o1` order, so the layer outputs must agree bit for bit.

use super::*;
use crate::bind::{BoundOp, BoundOpKind, Executes, Form, bind_with_fusion_for};
use crate::shape::infer;
use crate::test_support::{Lcg, evaluate_named_online_f32};
use proxima_numeric::NumericPolicy;

const EMBEDDING: u32 = 8;
const KV_HEADS: u32 = 2;
const GROUP: u32 = 2;
const HEAD_DIM: u32 = 4;
const PAIRS: u32 = HEAD_DIM / 2;
const NEW_TOKENS: u32 = 2;
const CACHED_TOKENS: u32 = 3;
const LAYERS: u32 = 2;
const THETA: f32 = 10000.0;

struct Fixture {
    program: Vec<Op>,
    inputs: Vec<(String, Vec<f32>)>,
}

impl Fixture {
    fn leaf(&mut self, name: &str, shape: &[u32], values: Vec<f32>) -> NodeId {
        let extents = shape.iter().map(|&extent| Extent::Static(extent)).collect();
        self.inputs.push((String::from(name), values));
        input_leaf(&mut self.program, DType::FLOAT32, extents, name)
    }

    fn random_leaf(&mut self, name: &str, shape: &[u32]) -> NodeId {
        let seed = name.bytes().fold(0x6212_0001u64, |acc, byte| {
            acc.wrapping_mul(31).wrapping_add(u64::from(byte))
        });
        let mut lcg = Lcg(seed);
        let count = shape.iter().product::<u32>() as usize;
        let values = (0..count).map(|_| lcg.next_unit()).collect();
        self.leaf(name, shape, values)
    }
}

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

fn cached_keys(layer: u32) -> Vec<f32> {
    let mut lcg = Lcg(0x6212_1000 + u64::from(layer));
    (0..CACHED_TOKENS * KV_HEADS * 2 * PAIRS)
        .map(|_| lcg.next_unit())
        .collect()
}

fn half_of_rows(rows: &[f32], half: usize) -> Vec<f32> {
    let pairs = PAIRS as usize;
    rows.chunks(2 * pairs)
        .flat_map(|row| row[half * pairs..(half + 1) * pairs].to_vec())
        .collect()
}

fn key_leaves(fixture: &mut Fixture, layer: u32, layout: KeyCacheLayout) -> (NodeId, NodeId) {
    let keys = cached_keys(layer);
    let rows = [CACHED_TOKENS, KV_HEADS];
    match layout {
        KeyCacheLayout::Planes => {
            let shape = [rows[0], rows[1], PAIRS];
            let even = fixture.leaf(
                &alloc::format!("kv_cache.{layer}.k_even"),
                &shape,
                half_of_rows(&keys, 0),
            );
            let odd = fixture.leaf(
                &alloc::format!("kv_cache.{layer}.k_odd"),
                &shape,
                half_of_rows(&keys, 1),
            );
            (even, odd)
        }
        KeyCacheLayout::Rows => {
            let shape = [rows[0], rows[1], 2 * PAIRS];
            let both = fixture.leaf(&alloc::format!("kv_cache.{layer}.k"), &shape, keys);
            (both, both)
        }
    }
}

fn build(layout: KeyCacheLayout, pairing: RopePairing) -> (Fixture, NodeId, NodeId) {
    let mut fixture = Fixture {
        program: Vec::new(),
        inputs: Vec::new(),
    };
    let heads = KV_HEADS * GROUP;
    let (cos_values, sin_values) = rope_tables(CACHED_TOKENS as f32);
    let cos = fixture.leaf("rope_cos", &[NEW_TOKENS, PAIRS], cos_values);
    let sin = fixture.leaf("rope_sin", &[NEW_TOKENS, PAIRS], sin_values);
    let group_ones = fixture.leaf(
        "group_ones",
        &[KV_HEADS, GROUP],
        alloc::vec![1.0; (KV_HEADS * GROUP) as usize],
    );
    let local_future = (0..NEW_TOKENS * NEW_TOKENS)
        .map(|cell| f32::from(u8::from(cell % NEW_TOKENS > cell / NEW_TOKENS)))
        .collect();
    let is_future_local = fixture.leaf("is_future_local", &[NEW_TOKENS, NEW_TOKENS], local_future);
    let is_future_cached = fixture.leaf(
        "is_future_cached",
        &[NEW_TOKENS, CACHED_TOKENS],
        alloc::vec![0.0; (NEW_TOKENS * CACHED_TOKENS) as usize],
    );
    let inv_dim = count_constant(&mut fixture.program, EMBEDDING);
    let eps = scalar_constant(&mut fixture.program, 1e-6);
    let inv_sqrt = scalar_constant(&mut fixture.program, 1.0 / (HEAD_DIM as f32).sqrt());
    let inv_head = count_constant(&mut fixture.program, HEAD_DIM);
    let neg_infinity = scalar_constant(&mut fixture.program, f32::NEG_INFINITY);
    let mut x = fixture.random_leaf("x", &[NEW_TOKENS, EMBEDDING]);
    let mut first_layer_output = x;
    for layer in 0..LAYERS {
        let norm = fixture.random_leaf(
            &alloc::format!("blk.{layer}.attn_norm.weight"),
            &[EMBEDDING],
        );
        let wq = fixture.random_leaf(
            &alloc::format!("blk.{layer}.attn_q.weight"),
            &[EMBEDDING, heads, HEAD_DIM],
        );
        let wk = fixture.random_leaf(
            &alloc::format!("blk.{layer}.attn_k.weight"),
            &[EMBEDDING, KV_HEADS, HEAD_DIM],
        );
        let wv = fixture.random_leaf(
            &alloc::format!("blk.{layer}.attn_v.weight"),
            &[EMBEDDING, KV_HEADS, HEAD_DIM],
        );
        let wo = fixture.random_leaf(
            &alloc::format!("blk.{layer}.attn_output.weight"),
            &[KV_HEADS, GROUP, HEAD_DIM, EMBEDDING],
        );
        let v_cache = fixture.random_leaf(
            &alloc::format!("kv_cache.{layer}.v"),
            &[CACHED_TOKENS, KV_HEADS, HEAD_DIM],
        );
        let (k_even, k_odd) = key_leaves(&mut fixture, layer, layout);
        let (out, _) = append_two_range_cached_attention(
            &mut fixture.program,
            x,
            inv_dim,
            eps,
            inv_sqrt,
            inv_head,
            Some((cos, sin)),
            PAIRS,
            group_ones,
            is_future_local,
            neg_infinity,
            is_future_cached,
            neg_infinity,
            GROUP,
            None,
            norm,
            None,
            KeySource::Projected {
                wk,
                k_norm_weight: None,
            },
            wq,
            ValueSource::Projected(wv),
            wo,
            pairing,
            None,
            false,
            k_even,
            k_odd,
            v_cache,
            layout,
        )
        .expect("one cached attention layer builds");
        if layer == 0 {
            first_layer_output = out;
        }
        x = out;
    }
    (fixture, first_layer_output, x)
}

fn run(layout: KeyCacheLayout, pairing: RopePairing) -> (Vec<f32>, Vec<f32>) {
    let (fixture, first, last) = build(layout, pairing);
    let named: Vec<(&str, &[f32])> = fixture
        .inputs
        .iter()
        .map(|(name, values)| (name.as_str(), values.as_slice()))
        .collect();
    let evaluated = evaluate_named_online_f32(&fixture.program, &[], &named, &[first, last])
        .expect("the cached attention stack evaluates");
    let read = |node| evaluated.get(node).expect("output requested").0.to_vec();
    (read(first), read(last))
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[proxima::test]
#[case::split_half(RopePairing::SplitHalf { pairs: PAIRS })]
#[case::adjacent_pairs(RopePairing::Interleaved)]
async fn one_row_key_cache_gives_the_two_plane_outputs_bit_for_bit(#[case] pairing: RopePairing) {
    let (planes_first, planes_last) = run(KeyCacheLayout::Planes, pairing);
    let (rows_first, rows_last) = run(KeyCacheLayout::Rows, pairing);

    assert_eq!(planes_last.len(), (NEW_TOKENS * EMBEDDING) as usize);
    assert!(planes_last.iter().all(|value| value.is_finite()));
    assert_eq!(bits(&rows_first), bits(&planes_first), "layer 0 output");
    assert_eq!(bits(&rows_last), bits(&planes_last), "layer 1 output");
    std::println!(
        "{pairing:?} rows bits {:?} planes bits {:?}",
        &bits(&rows_last)[..4],
        &bits(&planes_last)[..4]
    );
}

const VOCAB: u32 = 5;
const FEED_FORWARD: u32 = 6;

fn model_schedule(kv_heads: u32, pairing: RopePairing) -> Vec<LayerSchedule> {
    let attention = LayerAttentionConfig {
        head_dim: HEAD_DIM,
        kv_heads,
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

fn model_program(kv_heads: u32, layout: KeyCacheLayout, pairing: RopePairing) -> (Vec<Op>, NodeId) {
    let (program, logits, _) = model_program_with_roots(kv_heads, layout, pairing);
    (program, logits)
}

fn model_program_with_roots(
    kv_heads: u32,
    layout: KeyCacheLayout,
    pairing: RopePairing,
) -> (Vec<Op>, NodeId, Vec<CachedLayerRoots>) {
    let (program, logits, roots, ..) = scheduled_two_range_cached_forward_program_with_layer_roots(
        VOCAB,
        EMBEDDING,
        FEED_FORWARD,
        FEED_FORWARD,
        kv_heads * GROUP,
        LAYERS,
        0,
        0,
        LAYERS,
        &model_schedule(kv_heads, pairing),
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
        layout,
    )
    .expect("a two-layer cached model builds");
    let attention = roots
        .into_iter()
        .filter_map(|root| match root {
            LayerCacheRoots::Attention(roots) => Some(roots),
            _ => None,
        })
        .collect();
    (program, logits, attention)
}

fn declared_inputs(program: &[Op]) -> Vec<(String, Vec<Extent>)> {
    program
        .iter()
        .filter_map(|op| match op {
            Op::Input { shape, name, .. } => name.clone().map(|name| (name, shape.clone())),
            _ => None,
        })
        .collect()
}

fn extent_value(extent: &Extent) -> usize {
    match extent {
        Extent::Static(value) => *value as usize,
        Extent::Symbolic(0) => NEW_TOKENS as usize,
        Extent::Symbolic(_) => CACHED_TOKENS as usize,
    }
}

fn model_values(name: &str, shape: &[Extent], layout: KeyCacheLayout) -> Vec<f32> {
    let count: usize = shape.iter().map(extent_value).product();
    let key_layer = name
        .strip_prefix("kv_cache.")
        .and_then(|rest| rest.split_once('.'))
        .filter(|(_, leaf)| leaf.starts_with('k'))
        .and_then(|(layer, leaf)| Some((layer.parse::<u32>().ok()?, leaf)));
    match (name, key_layer, layout) {
        ("ids", ..) => (0..count).map(|row| (row + 1) as f32).collect(),
        ("cached_len", ..) => alloc::vec![CACHED_TOKENS as f32],
        ("eps", ..) => alloc::vec![1e-6; count],
        ("rope_cos", ..) => rope_tables(CACHED_TOKENS as f32).0,
        ("rope_sin", ..) => rope_tables(CACHED_TOKENS as f32).1,
        (_, Some((layer, "k")), _) => cached_keys(layer),
        (_, Some((layer, "k_even")), _) => half_of_rows(&cached_keys(layer), 0),
        (_, Some((layer, _)), _) => half_of_rows(&cached_keys(layer), 1),
        _ => {
            let seed = name.bytes().fold(0x6212_0002u64, |acc, byte| {
                acc.wrapping_mul(31).wrapping_add(u64::from(byte))
            });
            let mut lcg = Lcg(seed);
            (0..count).map(|_| lcg.next_unit()).collect()
        }
    }
}

fn model_logits(layout: KeyCacheLayout, pairing: RopePairing) -> Vec<f32> {
    model_logits_with_outputs(layout, pairing, false)
}

/// The logits, with only the logits requested or with every float32 node requested: the second
/// keeps every intermediate alive, so no op that absorbs one can bind.
fn model_logits_with_outputs(
    layout: KeyCacheLayout,
    pairing: RopePairing,
    every_node: bool,
) -> Vec<f32> {
    let (program, logits) = model_program(KV_HEADS, layout, pairing);
    let outputs: Vec<NodeId> = if every_node {
        (0..program.len() as u32)
            .map(NodeId)
            .filter(|node| program[node.0 as usize].dtype() == DType::FLOAT32)
            .collect()
    } else {
        alloc::vec![logits]
    };
    let owned: Vec<(String, Vec<f32>)> = declared_inputs(&program)
        .into_iter()
        .map(|(name, shape)| {
            let values = model_values(&name, &shape, layout);
            (name, values)
        })
        .collect();
    let named: Vec<(&str, &[f32])> = owned
        .iter()
        .map(|(name, values)| (name.as_str(), values.as_slice()))
        .collect();
    let symbols = [u64::from(NEW_TOKENS), u64::from(CACHED_TOKENS)];
    let evaluated = evaluate_named_online_f32(&program, &symbols, &named, &outputs)
        .expect("the two-layer cached model evaluates");
    evaluated.get(logits).expect("logits requested").0.to_vec()
}

#[proxima::test]
async fn with_layer_roots_declares_one_key_leaf_per_layer_under_rows_and_two_planes_otherwise() {
    let pairing = RopePairing::Interleaved;
    let rows = declared_inputs(&model_program(KV_HEADS, KeyCacheLayout::Rows, pairing).0);
    let planes = declared_inputs(&model_program(KV_HEADS, KeyCacheLayout::Planes, pairing).0);
    let key_leaves = |inputs: &[(String, Vec<Extent>)]| -> Vec<String> {
        inputs
            .iter()
            .filter(|(name, _)| name.starts_with("kv_cache.") && !name.ends_with(".v"))
            .map(|(name, _)| name.clone())
            .collect()
    };

    assert_eq!(
        key_leaves(&rows),
        ["kv_cache.0.k", "kv_cache.1.k"],
        "one row leaf per layer, no k_even or k_odd"
    );
    assert_eq!(
        key_leaves(&planes),
        [
            "kv_cache.0.k_even",
            "kv_cache.0.k_odd",
            "kv_cache.1.k_even",
            "kv_cache.1.k_odd"
        ]
    );
    let row_shape = &rows
        .iter()
        .find(|(name, _)| name == "kv_cache.0.k")
        .expect("row leaf")
        .1;
    assert_eq!(
        row_shape,
        &[
            Extent::Symbolic(1),
            Extent::Static(KV_HEADS),
            Extent::Static(2 * PAIRS)
        ]
    );
}

#[proxima::test]
#[case::split_half(RopePairing::SplitHalf { pairs: PAIRS })]
#[case::adjacent_pairs(RopePairing::Interleaved)]
async fn with_layer_roots_logits_match_bit_for_bit_between_key_cache_layouts(
    #[case] pairing: RopePairing,
) {
    let planes = model_logits(KeyCacheLayout::Planes, pairing);
    let rows = model_logits(KeyCacheLayout::Rows, pairing);

    assert_eq!(planes.len(), (NEW_TOKENS * VOCAB) as usize);
    assert!(planes.iter().all(|value| value.is_finite()), "{planes:?}");
    assert_eq!(bits(&rows), bits(&planes), "logits of the two-layer model");
}

fn bind_model(kv_heads: u32, layout: KeyCacheLayout, executes: &Executes) -> Vec<BoundOp> {
    let (program, logits) = model_program(kv_heads, layout, RopePairing::Interleaved);
    let symbols = [u64::from(NEW_TOKENS), u64::from(CACHED_TOKENS)];
    let shapes = infer(&program, &symbols).expect("the two-layer cached model infers its shapes");
    let numeric = NumericPolicy::bit_exact().with_flash_narrowing(true);
    bind_with_fusion_for(&program, &shapes, &[logits], true, numeric, executes)
        .expect("the two-layer cached model binds")
}

fn count_bound_ops(bound: &[BoundOp], is_kind: fn(&BoundOpKind) -> bool) -> usize {
    bound.iter().filter(|op| is_kind(&op.kind)).count()
}

fn is_cached_attention(kind: &BoundOpKind) -> bool {
    matches!(kind, BoundOpKind::CachedAttention { .. })
}

fn is_cached_softmax_weights(kind: &BoundOpKind) -> bool {
    matches!(kind, BoundOpKind::CachedSoftmaxWeights { .. })
}

#[proxima::test]
#[case::rows(KeyCacheLayout::Rows)]
#[case::planes(KeyCacheLayout::Planes)]
async fn one_kv_head_logits_only_plan_fuses_one_cached_softmax_weights_per_layer(
    #[case] layout: KeyCacheLayout,
) {
    let bound = bind_model(1, layout, &Executes::CPU);

    assert_eq!(
        count_bound_ops(&bound, is_cached_softmax_weights),
        LAYERS as usize,
        "one fused softmax per layer under {layout:?}"
    );
}

#[proxima::test]
#[case::rows(KeyCacheLayout::Rows)]
#[case::planes(KeyCacheLayout::Planes)]
async fn two_kv_head_logits_only_plan_fuses_one_cached_attention_per_layer_and_no_softmax_weights(
    #[case] layout: KeyCacheLayout,
) {
    let bound = bind_model(KV_HEADS, layout, &Executes::CPU);

    assert_eq!(
        count_bound_ops(&bound, is_cached_attention),
        LAYERS as usize,
        "one fused cached attention per layer under {layout:?}"
    );
    assert_eq!(
        count_bound_ops(&bound, is_cached_softmax_weights),
        0,
        "the softmax-weights kernel is exact for one kv head only"
    );
}

fn executes_without_cached_softmax_weights() -> Executes {
    Executes::CPU.without(Form::CachedSoftmaxWeights)
}

#[proxima::test]
async fn rows_cached_attention_reads_the_odd_key_half_at_base_pairs_and_the_even_half_at_base_zero()
{
    let bound = bind_model(
        KV_HEADS,
        KeyCacheLayout::Rows,
        &executes_without_cached_softmax_weights(),
    );

    let key_bases: Vec<[i64; 4]> = bound
        .iter()
        .filter_map(|op| match &op.kind {
            BoundOpKind::CachedAttention { operands, .. } => {
                Some([2, 3, 4, 5].map(|slot| operands[slot].1.base))
            }
            _ => None,
        })
        .collect();

    assert_eq!(
        key_bases,
        vec![[0, i64::from(PAIRS), 0, i64::from(PAIRS)]; LAYERS as usize],
        "even, odd, even, odd key slots of each layer's fused cached attention"
    );
}

#[proxima::test]
#[case::rows(KeyCacheLayout::Rows)]
#[case::planes(KeyCacheLayout::Planes)]
async fn logits_only_plan_matches_the_every_node_plan_within_rounding(
    #[case] layout: KeyCacheLayout,
) {
    let pairing = RopePairing::Interleaved;

    let fused = model_logits(layout, pairing);
    let unfused = model_logits_with_outputs(layout, pairing, true);

    let worst = fused
        .iter()
        .zip(&unfused)
        .map(|(fused, unfused)| (fused - unfused).abs() / unfused.abs().max(1e-3))
        .fold(0.0f32, f32::max);
    assert!(
        worst < 1e-4,
        "{layout:?} worst relative gap {worst}: logits-only {fused:?} every-node {unfused:?}"
    );
}
