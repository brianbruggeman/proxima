//! Metal-vs-CPU parity for the attention-sink term of the split-KV decode form
//! (`BoundOpKind::CachedAttention::attention_sink`, drawn by
//! `omega/src/msl/cached_attention_decode_split.rs`). Internal consistency
//! only: the CPU evaluator is this repo's own oracle; the unfused chain's
//! agreement with the CPU fused op is the `proxima-tensor` k_rows tests.
//!
//! The program is the scheduled two-range cached decode builder with
//! `attention_sinks` on, at flint's head ratio (32 query heads on 4 kv heads,
//! head_dim 64), swept over cached lengths that land on one, two, seven and
//! thirty-two splits. Each cell asserts the bound ops carry the sink operand
//! and render the `_ds_sk` kernel before comparing, and a control feeds the
//! CPU a different sink vector to prove the comparison can fail.

#![cfg(all(
    feature = "metal",
    feature = "metal-attn-split-decode",
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
const CACHED_LENGTHS: [u64; 4] = [31, 33, 199, 1099];

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

fn schedule(attention_sinks: bool) -> Vec<LayerSchedule> {
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
        rope_pairing: RopePairing::Interleaved,
        score_scale: AttentionScoreScale::Unscaled,
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

fn fixture(cached_len: u64, attention_sinks: bool) -> Fixture {
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
            &schedule(attention_sinks),
            None,
            None,
            false,
            None,
            false,
            SharedExpertWidth::NONE,
        )
        .expect("the flint-shaped sink decode program builds");
    let mut roots = vec![logits];
    for (even, odd, value) in &cache_roots {
        roots.extend([*even, *odd, *value]);
    }
    let symbols = vec![1, cached_len];
    let shapes = infer(&program, &symbols).expect("the decode program infers");
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
            "ids" => vec![3.0f32; count],
            "eps" => vec![1e-5f32; count],
            "cached_len" => vec![cached_len as f32],
            // spread wide enough that some heads sit above their row maximum and some far below it
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

/// Binds `fixture` under the production policy and returns how many bound
/// attention ops carry a sink operand, asserting each renders the sink kernel.
fn assert_sink_kernels(fixture: &Fixture, cached_len: u64) -> usize {
    let policy = production_numeric_policy();
    let shapes = infer(&fixture.program, &fixture.symbols).expect("the decode program infers");
    let resolved = bind(&fixture.program, &shapes, &fixture.roots[..1], policy)
        .expect("the decode program binds");
    let attention: Vec<_> = resolved
        .iter()
        .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .collect();
    assert_eq!(attention.len(), LAYERS as usize, "cached_len {cached_len}: one fused op per layer");
    for bound in &attention {
        let kernel = omega::emit(bound, &omega::PackedOperands::new(), policy)
            .expect("the bound sink attention op emits");
        assert!(
            kernel.entry.ends_with("_ds_sk"),
            "cached_len {cached_len}: the op must take the sink decode split, got {}",
            kernel.entry
        );
        assert!(
            kernel.source.contains("in9 [[buffer(9)]]"),
            "cached_len {cached_len}: the sink binds at buffer 9 after the cached_len operand"
        );
    }
    attention.len()
}

fn metal_and_cpu(fixture: &Fixture, cpu_named: &[(String, Vec<f32>)]) -> (Vec<f32>, Vec<f32>) {
    let policy = production_numeric_policy();
    let roots = [fixture.roots[0]];
    let metal_named = as_named_blocks(&fixture.named);
    let cpu_blocks = as_named_blocks(cpu_named);
    let cpu = evaluate_gpu_oracle(&fixture.program, &fixture.symbols, &cpu_blocks, &roots);
    let plan = omega::plan_named(&fixture.program, &fixture.symbols, &metal_named, &roots, policy)
        .expect("metal plans the sink decode program");
    let metal = omega::execute_plan_named(&plan, &metal_named).expect("metal runs the sink decode program");
    (cpu.root().to_vec(), metal.root().to_vec())
}

#[test]
fn the_sink_decode_split_holds_parity_with_the_cpu_evaluator_across_split_counts() {
    let mut cells = 0_usize;
    for cached_len in CACHED_LENGTHS {
        let fixture = fixture(cached_len, true);
        assert_eq!(assert_sink_kernels(&fixture, cached_len), LAYERS as usize);
        let (expected, actual) = metal_and_cpu(&fixture, &fixture.named);
        assert_finite(&expected, "cpu root");
        assert_finite(&actual, "metal root");
        assert_eq!(actual.len(), expected.len());
        let relative = max_relative(&expected, &actual);
        eprintln!("sink decode_split parity: cached_len={cached_len} relative={relative}");
        assert!(
            relative < 1e-4,
            "cached_len {cached_len}: metal disagrees with cpu on the sink decode root: relative={relative}"
        );
        cells += 1;
    }
    assert_eq!(cells, 4, "4 cached lengths");
}

#[test]
fn a_cpu_oracle_fed_different_sinks_departs_from_the_metal_sink_kernel() {
    let cached_len = 199;
    let fixture = fixture(cached_len, true);
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
    eprintln!("sink control: cached_len={cached_len} relative={relative}");
    assert!(
        relative > 1e-3,
        "the oracle given negated sinks must disagree with the kernel; relative={relative} means the sink does not reach the output"
    );
}

#[test]
fn a_sinkless_program_keeps_the_pre_sink_decode_split_kernel() {
    let fixture = fixture(199, false);
    let policy = production_numeric_policy();
    let shapes = infer(&fixture.program, &fixture.symbols).expect("the decode program infers");
    let resolved = bind(&fixture.program, &shapes, &fixture.roots[..1], policy)
        .expect("the decode program binds");
    let mut checked = 0_usize;
    for bound in resolved
        .iter()
        .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
    {
        let kernel = omega::emit(bound, &omega::PackedOperands::new(), policy)
            .expect("the bound attention op emits");
        assert!(kernel.entry.ends_with("_ds"), "got {}", kernel.entry);
        assert!(!kernel.source.contains("in9"), "a sink-less op renders no sink term");
        assert!(
            kernel.source.contains("device float* out [[buffer(9)]], constant Uniforms& u [[buffer(10)]]"),
            "a sink-less op keeps out at buffer 9 and the uniforms at 10"
        );
        checked += 1;
    }
    assert_eq!(checked, LAYERS as usize, "every layer's op was checked");
}
