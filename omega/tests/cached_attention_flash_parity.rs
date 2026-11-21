//! Flash-vs-legacy bit parity for the cached-attention flash decomposition
//! (`AttentionDecomposition::Flash`, `omega/src/msl/cached_attention_flash_tiled.rs`) against the
//! row-tiled legacy kernel it replaces, on flint's full-layer shape: 32 query heads on 4 kv heads
//! (eight per kv head), head_dim 128, 256 or 512, an attention sink, a binary16 cache. Each cell plans one
//! two-layer prefill twice over the same packed cache, once per decomposition, and asserts the
//! logits are bit-identical, so every difference between the two kernels shows up as a nonzero
//! `max_abs_diff`. The legacy kernel is the oracle: a difference here is a flash defect.
//!
//! Needs a Metal device.

#![cfg(all(
    feature = "metal",
    feature = "metal-attn-split-rows",
    feature = "metal-attn-variants",
    target_os = "macos"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use half::f16;
use proxima_numeric::float_layout::FloatLayout;
use proxima_tensor::cpu::QuantizedBlock;
use proxima_tensor::spec::{
    SLIDING_CACHED_LEN_INPUT,
    AttentionScoreScale, KeySourceKind, LayerAttentionConfig, LayerFfnConfig, LayerKind,
    LayerSchedule, RopePairing, RopeTableSel, SharedExpertWidth, ValueSourceKind,
    scheduled_two_range_cached_forward_program_with_experts,
};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{NodeId, Op, infer};

mod support;
use support::production_numeric_policy;

const VOCAB: u32 = 64;
const EMBEDDING: u32 = 256;
const FEED_FORWARD: u32 = 128;
const QUERY_HEADS: u32 = 32;
const KV_HEADS: u32 = 4;
const LAYERS: u32 = 2;
const FLINT_WINDOW: u32 = 513;

struct Fixture {
    program: Vec<Op>,
    symbols: Vec<u64>,
    logits: NodeId,
    named: Vec<(String, Vec<f32>)>,
}

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

fn schedule(window: Option<u32>, head_dim: u32) -> Vec<LayerSchedule> {
    let attention = LayerAttentionConfig {
        head_dim,
        kv_heads: KV_HEADS,
        mask_window: window,
        value_source_kind: ValueSourceKind::ProjectedV,
        key_source_kind: KeySourceKind::ProjectedK,
        rope_table: RopeTableSel {
            cos_name: "rope_cos".into(),
            sin_name: "rope_sin".into(),
        },
        rope_pairing: RopePairing::Interleaved,
        score_scale: AttentionScoreScale::InverseSqrtQueryPreAttnScalar(head_dim),
        value_norm: false,
        qk_norm: true,
        attention_sinks: true,
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

fn fixture(rows: u64, cached_len: u64, window: Option<u32>, head_dim: u32) -> Fixture {
    fixture_with_query_heads(rows, cached_len, window, head_dim, QUERY_HEADS)
}

fn fixture_with_query_heads(rows: u64, cached_len: u64, window: Option<u32>, head_dim: u32, query_heads: u32) -> Fixture {
    fixture_with_ring(rows, cached_len, window, head_dim, query_heads, false)
}

/// `ring` lays every windowed layer out as flint's serving path does: its cache leaves are bounded by the
/// sliding symbol at the whole window (`proxima-model-interop/src/generate/kv_ring.rs:76`, bound at
/// `decode.rs:2324`), a 513-row extent whatever the live rows, and its mask reads `cached_len_swa`.
fn fixture_with_ring(rows: u64, cached_len: u64, window: Option<u32>, head_dim: u32, query_heads: u32, ring: bool) -> Fixture {
    let (program, logits, _cache_roots, _moe_sites, _head_repeats) =
        scheduled_two_range_cached_forward_program_with_experts(
            VOCAB,
            EMBEDDING,
            FEED_FORWARD,
            0,
            query_heads,
            LAYERS,
            0,
            0,
            LAYERS,
            &schedule(window, head_dim),
            None,
            None,
            false,
            None,
            ring,
            SharedExpertWidth::NONE,
        )
        .expect("the flint-shaped flash prefill program builds");
    let ring_extent = u64::from(window.unwrap_or(0));
    let symbols = if ring { vec![rows, cached_len, ring_extent] } else { vec![rows, cached_len] };
    let shapes = infer(&program, &symbols).expect("the prefill program infers");
    let mut named = Vec::new();
    for (position, op) in program.iter().enumerate() {
        let Op::Input { name, .. } = op else { continue };
        let count: usize = shapes
            .of(NodeId(position as u32))
            .iter()
            .map(|extent| *extent as usize)
            .product();
        let name = name.clone().expect("every block input is named");
        let data = match name.as_str() {
            "ids" => (0..count).map(|index| (index % VOCAB as usize) as f32).collect(),
            "eps" => vec![1e-5f32; count],
            "cached_len" => vec![cached_len as f32],
            SLIDING_CACHED_LEN_INPUT => vec![cached_len.min(ring_extent) as f32],
            _ if name.ends_with(".attn_sinks.weight") => random_vec(position as u64 + 1, count)
                .into_iter()
                .map(|value| value * 4.0)
                .collect(),
            _ => random_vec(position as u64 + 1, count),
        };
        named.push((name, data));
    }
    Fixture {
        program,
        symbols,
        logits,
        named,
    }
}

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
        .collect()
}

fn logits_under(fixture: &Fixture, decomposition: omega::AttentionDecomposition) -> (Vec<u32>, Vec<String>) {
    logits_with_qk_operand(fixture, decomposition, Some(FloatLayout::BF32))
}

/// `qk_operand` is the flash QK MMA operand layout: `None` is the default (the K cache's own layout, half), `Some(BF32)` is the binary32 operand that matches legacy's dot bit for bit.
fn logits_with_qk_operand(
    fixture: &Fixture,
    decomposition: omega::AttentionDecomposition,
    qk_operand: Option<FloatLayout>,
) -> (Vec<u32>, Vec<String>) {
    let policy = production_numeric_policy();
    let packed: Vec<Option<Vec<u8>>> = fixture
        .named
        .iter()
        .map(|(name, values)| name.starts_with("kv_cache.").then(|| f16_bytes(values)))
        .collect();
    let blocks: Vec<(&str, QuantizedBlock<'_>)> = fixture
        .named
        .iter()
        .zip(&packed)
        .map(|((name, values), bytes)| match bytes {
            Some(bytes) => (
                name.as_str(),
                QuantizedBlock::Packed {
                    codec: omega::Codec::Float16,
                    bytes,
                },
            ),
            None => (name.as_str(), QuantizedBlock::Float32(values)),
        })
        .collect();
    let roots = [fixture.logits];
    let mut plan = omega::plan_named(&fixture.program, &fixture.symbols, &blocks, &roots, policy)
        .expect("metal plans the flint-shaped prefill");
    plan.set_flash_qk_operand(qk_operand);
    plan.set_attention_variant(omega::AttentionVariant {
        kv_storage: omega::AttentionKvStorage::F16,
        decomposition,
        ..omega::AttentionVariant::default()
    })
    .expect("the decomposition is admitted for the shape");
    let keys = plan.kernel_keys().expect("pipeline identities are collected");
    let output = omega::execute_plan_named(&plan, &blocks).expect("metal runs the prefill");
    (output.root().iter().map(|value| value.to_bits()).collect(), keys)
}

fn flash_vs_legacy(head_dim: u32, label: &str, rows: u64, cached_len: u64, window: Option<u32>, ring: bool) -> Result<(), String> {
    let cell = format!("d{head_dim} {label}: rows {rows} cached_len {cached_len} window {window:?} ring {ring}");
    let fixture = fixture_with_ring(rows, cached_len, window, head_dim, QUERY_HEADS, ring);
    let (legacy, legacy_keys) = logits_under(&fixture, omega::AttentionDecomposition::Legacy);
    let (flash, flash_keys) = logits_under(&fixture, omega::AttentionDecomposition::Flash);
    let (flash_again, _) = logits_under(&fixture, omega::AttentionDecomposition::Flash);
    let (legacy_again, _) = logits_under(&fixture, omega::AttentionDecomposition::Legacy);
    let diag = format!(
        "DIAG flash_vs_flash={} legacy_vs_legacy={} flash_nan={}",
        flash.iter().zip(&flash_again).filter(|(l, r)| l != r).count(),
        legacy.iter().zip(&legacy_again).filter(|(l, r)| l != r).count(),
        flash.iter().filter(|bits| f32::from_bits(**bits).is_nan()).count()
    );
    assert!(!legacy.is_empty(), "{cell}: an empty root compares nothing");
    assert!(
        flash_keys.iter().any(|key| key.contains("_flash")),
        "{cell}: the flash plan must run the flash kernel, keys {flash_keys:?}"
    );
    assert!(
        legacy_keys.iter().all(|key| !key.contains("_flash")),
        "{cell}: the legacy plan must not run the flash kernel"
    );
    let differing = legacy.iter().zip(&flash).filter(|(left, right)| left != right).count();
    let first = legacy.iter().zip(&flash).position(|(left, right)| left != right);
    let max_abs_diff = legacy
        .iter()
        .zip(&flash)
        .map(|(left, right)| (f32::from_bits(*left) - f32::from_bits(*right)).abs())
        .fold(0.0f32, f32::max);
    let sample = first.map(|index| (f32::from_bits(legacy[index]), f32::from_bits(flash[index])));
    let legacy_attention: Vec<&String> = legacy_keys.iter().filter(|key| key.contains("cached_attention")).collect();
    eprintln!(
        "flash parity: {cell} elements={} differing={differing} max_abs_diff={max_abs_diff:e} first={first:?} sample={sample:?} legacy_kernels={legacy_attention:?}",
        legacy.len()
    );
    if differing == 0 && flash == flash_again {
        return Ok(());
    }
    Err(format!(
        "{cell}: {differing} of {} elements differ, first {first:?} max_abs_diff={max_abs_diff:e} sample (legacy, flash)={sample:?} {diag}",
        legacy.len()
    ))
}

fn assert_all_cells_match(head_dim: u32, cells: &[Cell]) {
    assert_cells_match(head_dim, cells, false);
}

fn assert_cells_match(head_dim: u32, cells: &[Cell], ring: bool) {
    let failures: Vec<String> = cells
        .iter()
        .filter_map(|(label, rows, cached_len, window)| flash_vs_legacy(head_dim, label, *rows, *cached_len, *window, ring).err())
        .collect();
    eprintln!("flash parity SUMMARY: cells={} failing={}", cells.len(), failures.len());
    assert!(failures.is_empty(), "{} of {} cells differ:\n{}", failures.len(), cells.len(), failures.join("\n"));
}

type Cell = (&'static str, u64, u64, Option<u32>);

const WHOLE_AND_TAIL_CELLS: [Cell; 7] = [
    ("whole blocks", 512, 2048, None),
    ("whole small", 64, 64, None),
    ("tail q485 over c2048", 485, 2048, None),
    ("tail q487 flint last chunk", 487, 3584, None),
    ("tail q84 over c520", 84, 520, None),
    ("tail q130 over c1000", 130, 1000, None),
    ("tail q71 over c72", 71, 72, None),
];

const SLIDING_RING_CELLS: [Cell; 6] = [
    ("ring window 513 middle chunk q512 over c2048", 512, 2048, Some(FLINT_WINDOW)),
    ("ring window 513 last chunk q485 over c3584", 485, 3584, Some(FLINT_WINDOW)),
    ("ring window 513 chunk q512 over c1536", 512, 1536, Some(FLINT_WINDOW)),
    ("ring window 513 second chunk q512 over c512", 512, 512, Some(FLINT_WINDOW)),
    ("ring window 513 tail q485 over c2048", 485, 2048, Some(FLINT_WINDOW)),
    ("ring window 513 small q64 over c64", 64, 64, Some(FLINT_WINDOW)),
];

const SLIDING_CELLS: [Cell; 6] = [
    ("window 513 whole blocks", 512, 2048, Some(FLINT_WINDOW)),
    ("window 513 tail q485 over c2048", 485, 2048, Some(FLINT_WINDOW)),
    ("window 513 flint last chunk", 487, 3584, Some(FLINT_WINDOW)),
    ("window 513 small q64 over c64", 64, 64, Some(FLINT_WINDOW)),
    ("window 100 unaligned tail q130 over c1000", 130, 1000, Some(100)),
    ("window 33 tail q71 over c72", 71, 72, Some(33)),
];

#[test]
fn flash_holds_bit_parity_with_legacy_on_whole_blocks_and_tail_rows() {
    assert_all_cells_match(128, &WHOLE_AND_TAIL_CELLS);
}

#[test]
fn flash_holds_bit_parity_with_legacy_at_head_dim_256_on_whole_blocks_and_tail_rows() {
    assert_all_cells_match(256, &WHOLE_AND_TAIL_CELLS);
}

#[test]
fn flash_holds_bit_parity_with_legacy_at_head_dim_256_under_sliding_windows() {
    assert_all_cells_match(256, &SLIDING_CELLS);
}

#[test]
fn flash_holds_bit_parity_with_legacy_under_the_flint_sliding_window() {
    assert_all_cells_match(128, &SLIDING_CELLS);
}

#[test]
fn flash_holds_bit_parity_with_legacy_at_head_dim_512_on_whole_blocks_and_tail_rows() {
    assert_all_cells_match(512, &WHOLE_AND_TAIL_CELLS);
}

#[test]
fn flash_holds_bit_parity_with_legacy_on_flints_ring_cache_sliding_prefill_chunks() {
    assert_cells_match(128, &SLIDING_RING_CELLS, true);
}

#[test]
fn flash_holds_bit_parity_with_legacy_at_head_dim_512_under_sliding_windows() {
    assert_all_cells_match(512, &SLIDING_CELLS);
}

fn changed_elements(left: &[u32], right: &[u32]) -> usize {
    left.iter().zip(right).filter(|(a, b)| a != b).count() + left.len().abs_diff(right.len())
}

fn where_admitted_vs_legacy(
    head_dim: u32,
    label: &str,
    rows: u64,
    cached_len: u64,
    window: Option<u32>,
) -> Result<(), String> {
    let cell = format!("d{head_dim} {label}: rows {rows} cached_len {cached_len} window {window:?}");
    let fixture = fixture(rows, cached_len, window, head_dim);
    let (legacy, _) = logits_under(&fixture, omega::AttentionDecomposition::Legacy);
    let (_, flash_keys) = logits_under(&fixture, omega::AttentionDecomposition::Flash);
    let (admitted, admitted_keys) = logits_under(&fixture, omega::AttentionDecomposition::FlashWhereAdmitted);
    assert!(!legacy.is_empty(), "{cell}: an empty root compares nothing");
    assert!(
        flash_keys.iter().any(|key| key.contains("_flash")),
        "{cell}: the flash oracle must run the flash kernel"
    );
    let differing = changed_elements(&legacy, &admitted);
    let flash_attention: Vec<&String> = flash_keys.iter().filter(|key| key.contains("cached_attention")).collect();
    let admitted_attention: Vec<&String> = admitted_keys.iter().filter(|key| key.contains("cached_attention")).collect();
    eprintln!(
        "where_admitted parity: {cell} elements={} differing={differing} flash_kernels={flash_attention:?} admitted_kernels={admitted_attention:?}",
        legacy.len()
    );
    if admitted_keys != flash_keys {
        return Err(format!(
            "{cell}: flash_where_admitted ran different kernels than Flash, differing={differing}, flash {flash_attention:?} admitted {admitted_attention:?}"
        ));
    }
    if admitted_keys.iter().any(|key| key.contains("_flash_if_admitted")) {
        return Err(format!("{cell}: an unresolved decomposition reached a kernel identity"));
    }
    if differing == 0 {
        return Ok(());
    }
    Err(format!("{cell}: {differing} of {} elements differ from legacy", legacy.len()))
}

fn assert_where_admitted_cells_match(head_dim: u32, cells: &[Cell]) {
    let failures: Vec<String> = cells
        .iter()
        .filter_map(|(label, rows, cached_len, window)| {
            where_admitted_vs_legacy(head_dim, label, *rows, *cached_len, *window).err()
        })
        .collect();
    eprintln!("where_admitted parity SUMMARY: cells={} failing={}", cells.len(), failures.len());
    assert!(failures.is_empty(), "{} of {} cells differ:\n{}", failures.len(), cells.len(), failures.join("\n"));
}

#[test]
fn flash_where_admitted_runs_flash_with_legacy_parity_at_head_dim_128_on_whole_blocks_and_tail_rows() {
    assert_where_admitted_cells_match(128, &WHOLE_AND_TAIL_CELLS);
}

#[test]
fn flash_where_admitted_runs_flash_with_legacy_parity_at_head_dim_128_under_sliding_windows() {
    assert_where_admitted_cells_match(128, &SLIDING_CELLS);
}

#[test]
fn flash_where_admitted_keeps_legacy_for_an_op_flash_refuses() {
    let query_heads = 64;
    let fixture = fixture_with_query_heads(64, 64, None, 128, query_heads);
    let (legacy, legacy_keys) = logits_under(&fixture, omega::AttentionDecomposition::Legacy);
    let (admitted, admitted_keys) = logits_under(&fixture, omega::AttentionDecomposition::FlashWhereAdmitted);
    assert!(!legacy.is_empty(), "an empty root compares nothing");
    assert!(
        admitted_keys.iter().all(|key| !key.contains("_flash")),
        "a refused op must run the legacy kernel, keys {admitted_keys:?}"
    );
    assert_eq!(admitted_keys, legacy_keys);
    assert_eq!(changed_elements(&legacy, &admitted), 0);
}

fn where_admitted_runs_legacy(head_dim: u32, rows: u64, cached_len: u64) {
    let cell = format!("d{head_dim}: rows {rows} cached_len {cached_len}");
    let fixture = fixture(rows, cached_len, None, head_dim);
    let (legacy, legacy_keys) = logits_under(&fixture, omega::AttentionDecomposition::Legacy);
    let (admitted, admitted_keys) = logits_under(&fixture, omega::AttentionDecomposition::FlashWhereAdmitted);
    assert!(!legacy.is_empty(), "{cell}: an empty root compares nothing");
    let differing = changed_elements(&legacy, &admitted);
    eprintln!("where_admitted split parity: {cell} elements={} differing={differing} kernels={admitted_keys:?}", legacy.len());
    assert!(
        admitted_keys.iter().all(|key| !key.contains("flash")),
        "{cell}: an op Legacy splits must run Legacy, keys {admitted_keys:?}"
    );
    assert_eq!(admitted_keys, legacy_keys, "{cell}");
    assert_eq!(differing, 0, "{cell}");
}

#[test]
fn flash_where_admitted_runs_legacy_for_ops_legacy_splits() {
    where_admitted_runs_legacy(128, 8, 2048);
    where_admitted_runs_legacy(128, 16, 4096);
}

struct Divergence {
    max_abs: f64,
    max_rel: f64,
    scaled: f64,
    rms: f64,
}

fn divergence(legacy: &[u32], flash: &[u32]) -> Divergence {
    let pairs = legacy.iter().zip(flash).map(|(left, right)| (f64::from(f32::from_bits(*left)), f64::from(f32::from_bits(*right))));
    let (mut max_abs, mut max_rel, mut scale, mut square) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (left, right) in pairs {
        let gap = (left - right).abs();
        max_abs = max_abs.max(gap);
        max_rel = max_rel.max(gap / left.abs().max(1e-6));
        scale = scale.max(left.abs());
        square += gap * gap;
    }
    Divergence { max_abs, max_rel, scaled: max_abs / scale.max(1e-6), rms: (square / legacy.len() as f64).sqrt() }
}

/// legacy dots in binary32, so the default flash (half QK operands, binary32 accumulate) differs from it by the half operand
/// rounding alone. bound = max|flash - legacy| / max|legacy| over the logits; measured worst 1.50e-4 (d128 tail q71 over c72,
/// 2026-11-21), bound is 2x that rounded up to one figure. per-element relative error is not used: near-zero logits make it reach 3.9.
const HALF_QK_MAX_SCALED_DIVERGENCE: f64 = 3e-4;

fn default_flash_stays_within_the_half_operand_bound(head_dim: u32) {
    let mut lines = Vec::new();
    let mut worst = 0.0f64;
    for (label, rows, cached_len, window) in WHOLE_AND_TAIL_CELLS {
        let fixture = fixture(rows, cached_len, window, head_dim);
        let (legacy, _) = logits_with_qk_operand(&fixture, omega::AttentionDecomposition::Legacy, None);
        let (flash, keys) = logits_with_qk_operand(&fixture, omega::AttentionDecomposition::Flash, None);
        assert!(keys.iter().any(|key| key.contains("_flash")), "{label}: the default plan must run the flash kernel");
        let gap = divergence(&legacy, &flash);
        assert!(gap.max_abs > 0.0, "{label}: half operands must differ from the binary32 dot, or the default is not the cache layout");
        worst = worst.max(gap.scaled);
        lines.push(format!("half qk vs legacy: d{head_dim} {label} max_abs={:e} max_rel={:e} scaled={:e} rms={:e}", gap.max_abs, gap.max_rel, gap.scaled, gap.rms));
    }
    assert!(worst <= HALF_QK_MAX_SCALED_DIVERGENCE, "default flash left the half-operand bound {HALF_QK_MAX_SCALED_DIVERGENCE:e}:\n{}", lines.join("\n"));
}

#[test]
fn default_flash_with_half_qk_operands_stays_within_the_bound_of_legacy_at_head_dim_128() {
    default_flash_stays_within_the_half_operand_bound(128);
}

#[test]
fn default_flash_with_half_qk_operands_stays_within_the_bound_of_legacy_at_head_dim_256() {
    default_flash_stays_within_the_half_operand_bound(256);
}
