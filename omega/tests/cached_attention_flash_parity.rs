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
use proxima_tensor::cpu::QuantizedBlock;
use proxima_tensor::spec::{
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
            false,
            SharedExpertWidth::NONE,
        )
        .expect("the flint-shaped flash prefill program builds");
    let symbols = vec![rows, cached_len];
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

fn flash_vs_legacy(head_dim: u32, label: &str, rows: u64, cached_len: u64, window: Option<u32>) -> Result<(), String> {
    let cell = format!("d{head_dim} {label}: rows {rows} cached_len {cached_len} window {window:?}");
    let fixture = fixture(rows, cached_len, window, head_dim);
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
    let failures: Vec<String> = cells
        .iter()
        .filter_map(|(label, rows, cached_len, window)| flash_vs_legacy(head_dim, label, *rows, *cached_len, *window).err())
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
