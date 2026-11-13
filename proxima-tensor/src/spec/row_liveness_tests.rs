//! PROXNUM-262d: after the last cache-writing layer a `last_row_only` prefill
//! keeps only the output row (`.claude/.../kv-epic/PROXNUM-262-row-liveness.md`,
//! worked example 1).

use super::*;

const VOCAB: u32 = 5;
const EMBEDDING: u32 = 8;
const FEED_FORWARD: u32 = 6;
const QUERY_HEADS: u32 = 2;
const KV_HEADS: u32 = 1;
const HEAD_DIM: u32 = 4;
const LAYERS: usize = 35;
const WINDOW: u32 = 512;
const NEW_TOKENS: u64 = 4;
const CACHED_TOKENS: u64 = 3;

fn layer_attention(
    layer: usize,
    first_shared: Option<usize>,
    sliding_donor: usize,
    global_donor: usize,
) -> LayerAttentionConfig {
    let global = layer % 5 == 4;
    let donor = if global { global_donor } else { sliding_donor };
    let shared = first_shared.is_some_and(|first| layer >= first);
    let (key_source_kind, value_source_kind) = match (shared, global) {
        (true, _) => (
            KeySourceKind::SharedFromLayer(donor as u32),
            ValueSourceKind::SharedFromLayer(donor as u32),
        ),
        (false, true) => (KeySourceKind::ProjectedK, ValueSourceKind::SharedWithKey),
        (false, false) => (KeySourceKind::ProjectedK, ValueSourceKind::ProjectedV),
    };
    let (cos_name, sin_name) = if global {
        ("rope_cos", "rope_sin")
    } else {
        ("rope_cos_swa", "rope_sin_swa")
    };
    LayerAttentionConfig {
        head_dim: HEAD_DIM,
        kv_heads: KV_HEADS,
        mask_window: (!global).then_some(WINDOW),
        value_source_kind,
        key_source_kind,
        rope_table: RopeTableSel {
            cos_name: cos_name.into(),
            sin_name: sin_name.into(),
        },
        rope_pairing: RopePairing::SplitHalf {
            pairs: HEAD_DIM / 2,
        },
        score_scale: AttentionScoreScale::Unscaled,
        value_norm: !shared,
        qk_norm: true,
        attention_sinks: false,
        rope: true,
    }
}

fn gemma4_shaped_schedule(first_shared: Option<usize>) -> Vec<LayerSchedule> {
    (0..LAYERS)
        .map(|layer| LayerSchedule {
            kind: LayerKind::Attention,
            attention: layer_attention(layer, first_shared, 13, 14),
            ffn: LayerFfnConfig::exclusive(),
        })
        .collect()
}

fn build(schedule: &[LayerSchedule], last_row_only: bool, row_cut: bool) -> Vec<Op> {
    let (program, ..) = scheduled_two_range_cached_forward_program_with_layer_roots(
        VOCAB,
        EMBEDDING,
        FEED_FORWARD,
        FEED_FORWARD,
        QUERY_HEADS,
        LAYERS as u32,
        0,
        0,
        LAYERS as u32,
        schedule,
        None,
        None,
        last_row_only,
        row_cut,
        None,
        false,
        0,
        1,
        None,
        LoopCdConfig::default(),
        SharedExpertWidth::NONE,
        DType::FLOAT32,
    )
    .expect("the gemma4-shaped two-range cached program lowers");
    program
}

fn row_gathers_before_the_output_norm(program: &[Op]) -> usize {
    let output_norm = leaf_index(program, "output_norm.weight").expect("the output norm leaf");
    let Some(row_leaf) = leaf_index(program, "lm_head_row") else {
        return 0;
    };
    program[..output_norm]
        .iter()
        .filter(|expr| {
            matches!(expr, Op::Elementwise { operands, .. }
            if operands.iter().any(|(_, map)| matches!(
                map,
                crate::map::IndexMap::Computed { indices, .. } if indices.0 as usize == row_leaf
            )))
        })
        .count()
}

fn leaf_index(program: &[Op], wanted: &str) -> Option<usize> {
    program
        .iter()
        .position(|expr| matches!(expr, Op::Input { name: Some(name), .. } if name == wanted))
}

fn leaf_count(program: &[Op], wanted: &str) -> usize {
    program
        .iter()
        .filter(|expr| matches!(expr, Op::Input { name: Some(name), .. } if name == wanted))
        .count()
}

fn is_row_gather(expr: &Op) -> bool {
    matches!(expr, Op::Elementwise { operands, .. }
        if operands.iter().any(|(_, map)| matches!(map, crate::map::IndexMap::Computed { .. })))
}

fn token_rows_per_layer(program: &[Op]) -> Vec<Vec<u64>> {
    let symbols = [NEW_TOKENS, CACHED_TOKENS];
    let shapes = crate::shape::infer(program, &symbols).expect("the program infers");
    let extents = shapes.extents();
    let mut bounds: Vec<usize> = (0..LAYERS)
        .map(|layer| {
            leaf_index(program, &alloc::format!("blk.{layer}.attn_norm.weight"))
                .expect("every layer declares its attn_norm leaf")
        })
        .collect();
    bounds.push(leaf_index(program, "output_norm.weight").expect("the output norm leaf exists"));
    bounds
        .windows(2)
        .map(|range| {
            (range[0]..range[1])
                .filter(|&node| {
                    !matches!(program[node], Op::Input { .. }) && !is_row_gather(&program[node])
                })
                .filter_map(|node| match extents[node].as_slice() {
                    [rows, width]
                        if *width == u64::from(EMBEDDING) || *width == u64::from(FEED_FORWARD) =>
                    {
                        Some(*rows)
                    }
                    _ => None,
                })
                .collect()
        })
        .collect()
}

#[test]
fn gemma4_shared_kv_prefill_runs_one_row_after_the_last_owning_layer() {
    let schedule = gemma4_shaped_schedule(Some(15));

    let rows = token_rows_per_layer(&build(&schedule, true, true));

    assert_eq!(rows.len(), LAYERS);
    for (layer, layer_rows) in rows.iter().enumerate() {
        let expected = if layer <= 14 { NEW_TOKENS } else { 1 };
        assert!(
            !layer_rows.is_empty(),
            "layer {layer} produced no row-shaped ops to count"
        );
        assert!(
            layer_rows.iter().all(|&count| count == expected),
            "layer {layer} must carry {expected} rows on every hidden-width op, got {layer_rows:?}"
        );
    }
}

#[test]
fn cut_declares_the_lm_head_row_leaf_exactly_once() {
    let program = build(&gemma4_shaped_schedule(Some(15)), true, true);

    assert_eq!(leaf_count(&program, "lm_head_row"), 1);
}

#[test]
fn the_cut_layer_is_the_last_cache_writing_layer_only_for_last_row_programs_without_recurrence() {
    let shared = gemma4_shaped_schedule(Some(15));
    let unshared = gemma4_shaped_schedule(None);

    assert_eq!(row_cut_after(&shared, true, false), Some(14));
    assert_eq!(row_cut_after(&shared, false, false), None);
    assert_eq!(row_cut_after(&shared, true, true), None);
    assert_eq!(row_cut_after(&unshared, true, false), None);
}

#[test]
fn decode_shaped_program_has_no_row_gather_before_the_final_norm() {
    let schedule = gemma4_shaped_schedule(Some(15));

    let decode = build(&schedule, true, false);
    let prefill = build(&schedule, true, true);

    assert_eq!(leaf_count(&decode, "lm_head_row"), 1);
    assert_eq!(row_gathers_before_the_output_norm(&decode), 0);
    assert!(
        row_gathers_before_the_output_norm(&prefill) > 0,
        "the prefill program must gather the residual before the shared-KV tail"
    );
}

#[test]
fn decode_shaped_program_equals_the_program_built_without_the_cut_parameter() {
    let schedule = gemma4_shaped_schedule(Some(15));
    let (legacy, ..) = scheduled_two_range_cached_forward_program_with_experts(
        VOCAB,
        EMBEDDING,
        FEED_FORWARD,
        FEED_FORWARD,
        QUERY_HEADS,
        LAYERS as u32,
        0,
        0,
        LAYERS as u32,
        &schedule,
        None,
        None,
        true,
        None,
        false,
        SharedExpertWidth::NONE,
    )
    .expect("the two-range cached program lowers");

    assert!(legacy == build(&schedule, true, false));
}

#[test]
fn without_last_row_only_every_layer_keeps_every_row() {
    let program = build(&gemma4_shaped_schedule(Some(15)), false, true);

    let rows = token_rows_per_layer(&program);

    assert_eq!(leaf_count(&program, "lm_head_row"), 0);
    assert!(
        rows.iter().all(|layer_rows| !layer_rows.is_empty()
            && layer_rows.iter().all(|&count| count == NEW_TOKENS)),
        "a non-last_row_only program must keep {NEW_TOKENS} rows through every layer, got {rows:?}"
    );
}

#[test]
fn without_shared_layers_last_row_only_keeps_every_row_until_the_head_gather() {
    let program = build(&gemma4_shaped_schedule(None), true, true);

    let rows = token_rows_per_layer(&program);

    assert_eq!(leaf_count(&program, "lm_head_row"), 1);
    assert!(
        rows.iter().all(|layer_rows| !layer_rows.is_empty()
            && layer_rows.iter().all(|&count| count == NEW_TOKENS)),
        "a schedule with no shared layers must not cut, got {rows:?}"
    );
}

#[test]
fn layers_before_the_last_owning_layer_are_identical_with_and_without_the_cut() {
    let schedule = gemma4_shaped_schedule(Some(15));
    let uncut = build(&schedule, false, false);
    let cut = build(&schedule, true, true);

    let last_owner_leaf = "blk.14.attn_norm.weight";
    let uncut_head = leaf_index(&uncut, last_owner_leaf).expect("uncut layer 14 leaf");
    let cut_head = leaf_index(&cut, last_owner_leaf).expect("cut layer 14 leaf");

    assert!(uncut_head > 0, "layers 0..13 must contribute ops");
    assert!(uncut[..uncut_head] == cut[..cut_head]);
}
