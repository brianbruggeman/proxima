//! ROW 375: is the cached-attention kernel slow PER DISPATCH compared with
//! llama's flash-attention kernel at the same shape, or is the ROW 358
//! ablation's 1.094ms/32-dispatch in-program delta mostly bytes/structure?
//!
//! Builds the REAL openchat decode shape (32 query heads, 8 kv heads,
//! head_dim 128, GQA group 4) through the production single-range merged-KV
//! builder, [`mistral_single_range_cached_forward_program`] -- the exact
//! function `omega/tests/support/mod.rs`'s `real_single_range_forward_
//! fixture_with_padding` already wraps at toy dimensions for parity tests;
//! this file calls the SAME production function at real dimensions instead
//! of adding a second copy of its shape logic. `feed_forward`/`vocab` are
//! kept small (256/64) because this row measures ONLY the `cached_attention`
//! kind's own GPU time -- the FFN/embedding/head matvecs the forward pass
//! also emits are dead weight for this question, and shrinking them keeps
//! compile+bind+plan fast without changing the attention kernel's own shape,
//! inputs, or dispatch count.
//!
//! `block_count = 32` gives 32 independent per-layer `cached_attention`
//! dispatches sharing one `plan`/`execute_plan_named_op_timed` call -- the
//! "32 instances per command buffer" shape ROW 368/372's rmsnorm harness
//! established for amortizing per-dispatch fixed cost, reused here through
//! the production emit path rather than a synthetic INSTANCES loop, since a
//! real 32-layer forward already produces exactly that many attention
//! dispatches.
//!
//! `NumericPolicy::llama_relaxed()` is the compiled production default
//! (`proxima-tensor/src/numeric.rs`'s own doc). Merged-KV form throughout
//! (ROW 366: `cached_key_rows: 0`, a single contiguous KV range) --
//! `mistral_single_range_cached_forward_program` is the fused-single-range
//! builder that produces exactly that shape, never the two-range kind
//! `mistral_cached_forward_program` emits.

#![cfg(all(
    feature = "metal",
    feature = "cached-attention-streaming",
    target_os = "macos"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Instant;

use proxima_tensor::spec::{DuplicateHeadPosition, mistral_single_range_cached_forward_program};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{BoundOpKind, NodeId, NumericPolicy, Op, bind, block_node_ids, infer};

mod support;
use support::as_named_blocks;

const REPEATS: usize = 7;
const QUERY_HEADS: u32 = 32;
const KV_HEADS: u32 = 8;
const HEAD_DIM: u32 = 128;
const EMBEDDING: u32 = QUERY_HEADS * HEAD_DIM;
const FEED_FORWARD: u32 = 256;
const VOCAB: u32 = 64;
const LAYERS: u32 = 32;

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

type Row375Fixture = (Vec<Op>, Vec<u64>, Vec<NodeId>, Vec<(String, Vec<f32>)>);

/// Builds the real-shape openchat single-range forward at one context
/// length. `context_length` is `cached_len + 1` (one new query row, the
/// decode shape) -- this row never exercises prefill (`new_count > 1`).
fn build_fixture(context_length: u64) -> Row375Fixture {
    let cached_len = context_length - 1;
    let new_count = 1u64;

    let (program, logits_root, cache_roots, _) = mistral_single_range_cached_forward_program(
        VOCAB,
        EMBEDDING,
        FEED_FORWARD,
        QUERY_HEADS,
        KV_HEADS,
        HEAD_DIM,
        LAYERS,
        false,
        DuplicateHeadPosition::None,
    )
    .expect("the real-shape single-range forward program builds");

    let mut roots = vec![logits_root];
    for (even, odd, value) in &cache_roots {
        roots.push(*even);
        roots.push(*odd);
        roots.push(*value);
    }

    let symbols = vec![new_count, context_length];
    let shapes = infer(&program, &symbols).expect("the real-shape single-range forward infers");

    let mut named: Vec<(String, Vec<f32>)> = Vec::new();
    for node in block_node_ids(&program) {
        let Op::Input { name, .. } = &program[node.0 as usize] else {
            unreachable!("block_node_ids only ever returns Op::Input nodes")
        };
        let name = name.clone().expect("every block input in this program is named");
        let count: usize = shapes.of(node).iter().map(|extent| *extent as usize).product();
        let data = if name == "ids" {
            vec![3.0f32; count]
        } else if name == "eps" {
            vec![1e-5f32; count]
        } else if name == "cached_len" {
            vec![cached_len as f32]
        } else {
            random_vec(node.0 as u64 + 1, count)
        };
        named.push((name, data));
    }

    (program, symbols, roots, named)
}

struct Stats {
    mean_us: f64,
    median_us: f64,
    min_us: f64,
    max_us: f64,
    cov_pct: f64,
}

fn stats(samples: &[f64]) -> Stats {
    let len = samples.len();
    let mut sorted = samples.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).expect("no NaN sample"));
    let mean = samples.iter().sum::<f64>() / len as f64;
    let median = if len.is_multiple_of(2) {
        (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
    } else {
        sorted[len / 2]
    };
    let variance = samples.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / len as f64;
    let cov_pct = if mean.abs() > f64::MIN_POSITIVE {
        100.0 * variance.sqrt() / mean
    } else {
        0.0
    };
    Stats {
        mean_us: mean,
        median_us: median,
        min_us: sorted[0],
        max_us: sorted[len - 1],
        cov_pct,
    }
}

/// Runs one context-length cell: builds the real-shape fixture, binds it
/// once (`NumericPolicy::llama_relaxed()`, production's compiled default) to
/// find which resolved positions are the [`BoundOpKind::CachedAttention`]
/// dispatches, then times [`REPEATS`] warmed-up
/// `omega::metal::execute_plan_named_op_timed` runs and sums ONLY those
/// positions' own `gpu_ns` -- the same per-op GPU-time attribution
/// `rmsnorm_fused_epilogue_cost.rs`'s `ROW 372` cell uses, filtered to one
/// kind instead of summed over the whole plan.
fn run_cell(label: &str, context_length: u64) {
    let (program, symbols, roots, named) = build_fixture(context_length);
    let named_blocks = as_named_blocks(&named);

    let shapes = infer(&program, &symbols).expect("real-shape fixture infers");
    let resolved = bind(&program, &shapes, &roots, NumericPolicy::llama_relaxed())
        .expect("real-shape fixture binds");
    let attention_positions: Vec<usize> = resolved
        .iter()
        .enumerate()
        .filter(|(_, bound)| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .map(|(position, _)| position)
        .collect();
    assert_eq!(
        attention_positions.len(),
        LAYERS as usize,
        "{label}: expected one cached_attention dispatch per layer"
    );

    let plan = omega::plan_named(&program, &symbols, &named_blocks, &roots, NumericPolicy::llama_relaxed())
        .expect("metal plans the real-shape fixture");

    let (_evaluated, warm_timings) =
        omega::metal::execute_plan_named_op_timed(&plan, &named_blocks).expect("warm-up gpu-timed run");
    drop(warm_timings);

    let mut us_per_dispatch_samples = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let started = Instant::now();
        let (_evaluated, timings) =
            omega::metal::execute_plan_named_op_timed(&plan, &named_blocks).expect("timed run");
        let wall_us = started.elapsed().as_secs_f64() * 1e6;
        let attention_gpu_ns: u64 = attention_positions
            .iter()
            .map(|position| timings[*position].gpu_ns)
            .sum();
        let us_per_dispatch = attention_gpu_ns as f64 / LAYERS as f64 / 1e3;
        us_per_dispatch_samples.push(us_per_dispatch);
        println!("ROW 375 {label} raw wall_us={wall_us:.2} attention_gpu_ns_total={attention_gpu_ns}");
    }
    let stat = stats(&us_per_dispatch_samples);
    let kv_bytes_per_dispatch = context_length * KV_HEADS as u64 * HEAD_DIM as u64 * 2 * 4;
    let effective_gbps = kv_bytes_per_dispatch as f64 / (stat.median_us * 1e-6) / 1e9;
    println!(
        "ROW 375 {label} context_length={context_length} dispatches={LAYERS} \
         gpu_us_per_dispatch: mean={:.3} median={:.3} min={:.3} max={:.3} cov={:.2}% \
         kv_bytes_per_dispatch={kv_bytes_per_dispatch} effective_gbps={effective_gbps:.2}",
        stat.mean_us, stat.median_us, stat.min_us, stat.max_us, stat.cov_pct
    );
}

#[test]
#[ignore = "timed GPU cell, run explicitly per ROW 375's re-prove command"]
fn cached_attention_bare_context_40() {
    run_cell("a_context40", 40);
}

#[test]
#[ignore = "timed GPU cell, run explicitly per ROW 375's re-prove command"]
fn cached_attention_bare_context_512() {
    run_cell("b_context512", 512);
}
