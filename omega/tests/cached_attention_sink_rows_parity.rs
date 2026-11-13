//! Metal-vs-CPU parity for the attention-sink term of the row-tiled form
//! (`BoundOpKind::CachedAttention::attention_sink` with `K >= 2` query rows,
//! drawn by `omega/src/msl/cached_attention_row_tiled.rs`: the sink is the
//! online softmax's initial state, `max = sink`, `sum = 1`). Internal
//! consistency only: the CPU evaluator is this repo's own oracle.
//!
//! The program is the scheduled two-range cached builder with
//! `attention_sinks` on, at flint's head ratio (32 query heads on 4 kv heads,
//! head_dim 64), swept over prompt widths 84, 512 and 2024, one full and one
//! sliding layer schedule. Each cell asserts the bound ops carry the sink
//! operand and render the `_rt_sk` kernel before comparing; a control feeds the
//! CPU negated sinks and must disagree, and a sink-less program must keep the
//! pre-sink row-tiled text.

#![cfg(all(
    feature = "metal",
    feature = "metal-attn-split-rows",
    target_os = "macos"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::spec::{
    AttentionScoreScale, KeySourceKind, LayerAttentionConfig, LayerFfnConfig, LayerKind,
    LayerSchedule, RopePairing, RopeTableSel, SharedExpertWidth, ValueSourceKind,
    scheduled_two_range_cached_forward_program_with_experts,
};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{BoundOpKind, NodeId, Op, bind, infer};

mod support;
use support::{as_named_blocks, evaluate_gpu_oracle, production_numeric_policy};

const VOCAB: u32 = 64;
const EMBEDDING: u32 = 256;
const FEED_FORWARD: u32 = 128;
const QUERY_HEADS: u32 = 32;
const KV_HEADS: u32 = 4;
const HEAD_DIM: u32 = 64;
const LAYERS: u32 = 2;
const CACHED_LEN: u64 = 64;
const PROMPT_ROWS: [u64; 3] = [84, 512, 2024];
const WINDOWS: [Option<u32>; 2] = [None, Some(128)];

struct Fixture {
    program: Vec<Op>,
    symbols: Vec<u64>,
    roots: Vec<NodeId>,
    named: Vec<(String, Vec<f32>)>,
}

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

fn schedule(attention_sinks: bool, window: Option<u32>) -> Vec<LayerSchedule> {
    let attention = LayerAttentionConfig {
        head_dim: HEAD_DIM,
        kv_heads: KV_HEADS,
        mask_window: window,
        value_source_kind: ValueSourceKind::ProjectedV,
        key_source_kind: KeySourceKind::ProjectedK,
        rope_table: RopeTableSel {
            cos_name: "rope_cos".into(),
            sin_name: "rope_sin".into(),
        },
        rope_pairing: RopePairing::Interleaved,
        score_scale: if attention_sinks {
            AttentionScoreScale::InverseSqrtQueryPreAttnScalar(HEAD_DIM)
        } else {
            AttentionScoreScale::Unscaled
        },
        value_norm: false,
        qk_norm: true,
        attention_sinks,
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

fn fixture(rows: u64, window: Option<u32>, attention_sinks: bool) -> Fixture {
    fixture_with_cache(rows, CACHED_LEN, window, attention_sinks)
}

fn fixture_with_cache(rows: u64, cached_len: u64, window: Option<u32>, attention_sinks: bool) -> Fixture {
    let (program, logits, cache_roots, _moe_sites, _head_repeats) =
        scheduled_two_range_cached_forward_program_with_experts(
            VOCAB,
            EMBEDDING,
            FEED_FORWARD,
            0,
            QUERY_HEADS,
            LAYERS,
            0,
            0,
            LAYERS,
            &schedule(attention_sinks, window),
            None,
            None,
            false,
            None,
            false,
            SharedExpertWidth::NONE,
        )
        .expect("the flint-shaped sink prefill program builds");
    let mut roots = vec![logits];
    for (even, odd, value) in &cache_roots {
        roots.extend([*even, *odd, *value]);
    }
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
        roots,
        named,
    }
}

fn max_relative(expected: &[f32], actual: &[f32]) -> f32 {
    let magnitude = expected.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
    let difference = expected
        .iter()
        .zip(actual)
        .map(|(want, got)| (want - got).abs())
        .fold(0.0f32, f32::max);
    difference / magnitude.max(f32::MIN_POSITIVE)
}

fn assert_finite(values: &[f32], what: &str) {
    assert!(!values.is_empty(), "{what}: an empty root compares nothing");
    assert!(values.iter().all(|value| value.is_finite()), "{what}: non-finite output");
}

fn assert_sink_kernels(fixture: &Fixture, cell: &str) {
    let policy = production_numeric_policy();
    let shapes = infer(&fixture.program, &fixture.symbols).expect("the prefill program infers");
    let resolved = bind(&fixture.program, &shapes, &fixture.roots[..1], policy)
        .expect("the prefill program binds");
    let attention: Vec<_> = resolved
        .iter()
        .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .collect();
    assert_eq!(attention.len(), LAYERS as usize, "{cell}: one fused op per layer");
    for bound in &attention {
        let kernel = omega::emit(bound, &omega::PackedOperands::new(), policy)
            .expect("the bound sink attention op emits");
        assert!(
            kernel.entry.ends_with("_rt_sk"),
            "{cell}: the op must take the sink row-tiled form, got {}",
            kernel.entry
        );
        assert!(
            kernel.source.contains("in9 [[buffer(9)]]"),
            "{cell}: the sink binds at buffer 9 after the cached_len operand"
        );
    }
}

fn metal_and_cpu(fixture: &Fixture, cpu_named: &[(String, Vec<f32>)]) -> (Vec<f32>, Vec<f32>) {
    let policy = production_numeric_policy();
    let roots = [fixture.roots[0]];
    let metal_named = as_named_blocks(&fixture.named);
    let cpu_blocks = as_named_blocks(cpu_named);
    let cpu = evaluate_gpu_oracle(&fixture.program, &fixture.symbols, &cpu_blocks, &roots);
    let plan = omega::plan_named(&fixture.program, &fixture.symbols, &metal_named, &roots, policy)
        .expect("metal plans the sink prefill program");
    let metal = omega::execute_plan_named(&plan, &metal_named).expect("metal runs the sink prefill program");
    (cpu.root().to_vec(), metal.root().to_vec())
}

#[test]
fn the_sink_row_tiled_form_holds_parity_with_the_cpu_evaluator_across_prompt_widths() {
    let mut cells = 0_usize;
    for window in WINDOWS {
        for rows in PROMPT_ROWS {
            let cell = format!("rows {rows} window {window:?}");
            let fixture = fixture(rows, window, true);
            assert_sink_kernels(&fixture, &cell);
            let (expected, actual) = metal_and_cpu(&fixture, &fixture.named);
            assert_finite(&expected, "cpu root");
            assert_finite(&actual, "metal root");
            assert_eq!(actual.len(), expected.len());
            let relative = max_relative(&expected, &actual);
            eprintln!("sink row_tiled parity: {cell} relative={relative}");
            assert!(relative < 1e-4, "{cell}: metal disagrees with cpu: relative={relative}");
            cells += 1;
        }
    }
    assert_eq!(cells, 6, "3 prompt widths x 2 window schedules");
}

#[test]
fn the_sink_row_tiled_form_masks_a_partial_last_key_tile_when_the_cache_is_not_a_multiple_of_eight() {
    let cells: [(u64, u64, Option<u32>); 5] = [
        (84, 513, None),
        (84, 9, None),
        (84, 7, None),
        (84, 1, None),
        (84, 513, Some(128)),
    ];
    for (rows, cached_len, window) in cells {
        let cell = format!("rows {rows} cached_len {cached_len} window {window:?}");
        let fixture = fixture_with_cache(rows, cached_len, window, true);
        assert_sink_kernels(&fixture, &cell);
        let (expected, actual) = metal_and_cpu(&fixture, &fixture.named);
        assert_finite(&expected, "cpu root");
        assert_finite(&actual, "metal root");
        assert_eq!(actual.len(), expected.len());
        let relative = max_relative(&expected, &actual);
        eprintln!("sink row_tiled partial tile parity: {cell} relative={relative}");
        assert!(relative < 1e-4, "{cell}: metal disagrees with cpu: relative={relative}");
    }
}

#[test]
fn a_cpu_oracle_fed_negated_sinks_departs_from_the_metal_row_tiled_sink_kernel() {
    let fixture = fixture(84, None, true);
    let other_sinks: Vec<(String, Vec<f32>)> = fixture
        .named
        .iter()
        .map(|(name, data)| {
            if name.ends_with(".attn_sinks.weight") {
                (name.clone(), data.iter().map(|value| -value).collect())
            } else {
                (name.clone(), data.clone())
            }
        })
        .collect();
    let (expected, actual) = metal_and_cpu(&fixture, &other_sinks);
    let relative = max_relative(&expected, &actual);
    eprintln!("sink rows control: relative={relative}");
    assert!(
        relative > 1e-3,
        "the oracle given negated sinks must disagree with the kernel; relative={relative}"
    );
}

#[test]
fn a_sinkless_program_keeps_the_pre_sink_row_tiled_kernel() {
    let fixture = fixture(84, None, false);
    let policy = production_numeric_policy();
    let shapes = infer(&fixture.program, &fixture.symbols).expect("the prefill program infers");
    let resolved = bind(&fixture.program, &shapes, &fixture.roots[..1], policy)
        .expect("the prefill program binds");
    let mut checked = 0_usize;
    for bound in resolved
        .iter()
        .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
    {
        let kernel = omega::emit(bound, &omega::PackedOperands::new(), policy)
            .expect("the bound attention op emits");
        assert!(kernel.entry.ends_with("_rt"), "got {}", kernel.entry);
        assert!(!kernel.source.contains("in9"), "a sink-less op renders no sink term");
        assert!(
            kernel.source.contains("device float* out [[buffer(9)]], constant Uniforms& u [[buffer(10)]]"),
            "a sink-less op keeps out at buffer 9 and the uniforms at 10"
        );
        checked += 1;
    }
    assert_eq!(checked, LAYERS as usize, "every layer's op was checked");
}
