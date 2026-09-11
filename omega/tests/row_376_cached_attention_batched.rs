//! ROW 376: replaces ROW 375's harness. ROW 375 called its OWN [`bind`] to
//! find "attention positions" and then indexed
//! [`omega::metal::execute_plan_named_op_timed`]'s returned timing vector
//! with those positions -- but the driver's internal `prepare()` (`omega::
//! metal::prepare`, `prune_dead`/`promote_output_placed_nodes`) runs its own
//! bind and may prune or reorder nodes relative to a caller's separate bind
//! call, so a position from one bind is not provably a position in the
//! other's timing vector. This file never does that: every summed record is
//! selected by [`omega::metal::OpGpuTiming::kind`] -- the SAME field the
//! execution call itself derives from the resolved op it just dispatched --
//! and the record count is asserted against the expected dispatch count
//! (`LAYERS`) on every repeat, not assumed from a separate structure. ROW
//! 375's per-dispatch numbers are WITHDRAWN (see the addendum in
//! `proxima-tensor/docs/discipline.md`'s ROW 375 entry) because the records
//! they summed were never shown to be attention records at all.
//!
//! This file also replaces ROW 375's one-command-buffer-PER-OP instrument
//! (`execute_plan_named_op_timed`, `omega/src/metal.rs`'s own doc: one
//! `MTLCommandBuffer` per `BoundOp`) with [`omega::metal::
//! execute_plan_named_timed`], a one-command-buffer-for-the-WHOLE-PLAN
//! variant added alongside it -- the same "N instances, one command buffer"
//! shape `rmsnorm_fused_epilogue_cost.rs`'s ROW 368/372 batched harness
//! already uses through `execute_plan_named`, with the command buffer's own
//! `GPUStartTime`/`GPUEndTime` read back once instead of per op. The
//! per-op-timed call is kept as a VALIDATION cross-check (kind-filtered,
//! count-asserted), never as the reported per-dispatch number.
//!
//! Builds the REAL openchat decode shape (32 query heads, 8 kv heads,
//! head_dim 128, GQA group 4) through the production single-range merged-KV
//! builder, [`mistral_single_range_cached_forward_program`] -- the exact
//! function `omega/tests/support/mod.rs`'s `real_single_range_forward_
//! fixture_with_padding` already wraps at toy dimensions for parity tests;
//! this file calls the SAME production function at real dimensions instead
//! of adding a second copy of its shape logic. `feed_forward`/`vocab` are
//! kept small (256/64) because this row measures the `cached_attention`
//! kind's own GPU time against the WHOLE plan's total -- the FFN/embedding/
//! head matvecs the forward pass also emits are dead weight for this
//! question, and shrinking them keeps their share of the one shared command
//! buffer's total time small without changing the attention kernel's own
//! shape, inputs, or dispatch count. Their non-zero share is exactly why
//! this file reports the per-op-timed, kind-filtered sum ALONGSIDE the
//! batched total: the difference between the two is the FFN/embedding
//! contamination in the batched number, made visible rather than hidden.
//!
//! `block_count = 32` gives 32 independent per-layer `cached_attention`
//! dispatches sharing one `plan`/`execute_plan_named_timed` call -- a real
//! 32-layer forward already produces exactly that many attention dispatches.
//!
//! `NumericPolicy::llama_relaxed()` is the compiled production default
//! (`proxima-tensor/src/numeric.rs`'s own doc). Merged-KV form throughout
//! (ROW 366: `cached_key_rows: 0`, a single contiguous KV range) --
//! `mistral_single_range_cached_forward_program` is the fused-single-range
//! builder that produces exactly that shape, never the two-range kind
//! `mistral_cached_forward_program` emits.
//!
//! The `context_length=4096` cell's KV cache roots are ~1 GiB of f32
//! (`LAYERS * 2 * 4096 * KV_HEADS * HEAD_DIM * 4` bytes) -- `build_fixture`
//! allocates each named input's data ONCE (outside the repeat loop) via the
//! same [`Lcg`] synthesizer every other cell already uses, so this is not a
//! new allocation shape, only a larger one.

#![cfg(all(
    feature = "metal",
    feature = "cached-attention-streaming",
    feature = "instrument",
    target_os = "macos"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Instant;

use proxima_tensor::spec::{DuplicateHeadPosition, mistral_single_range_cached_forward_program};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{NodeId, NumericPolicy, Op, block_node_ids, infer};

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
const ATTENTION_KIND: &str = "cached_attention";

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

type Row376Fixture = (Vec<Op>, Vec<u64>, Vec<NodeId>, Vec<(String, Vec<f32>)>);

/// Builds the real-shape openchat single-range forward at one context
/// length. `context_length` is `cached_len + 1` (one new query row, the
/// decode shape) -- this row never exercises prefill (`new_count > 1`).
fn build_fixture(context_length: u64) -> Row376Fixture {
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
        false,
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
        let name = name
            .clone()
            .expect("every block input in this program is named");
        let count: usize = shapes
            .of(node)
            .iter()
            .map(|extent| *extent as usize)
            .product();
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
    let variance = samples
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / len as f64;
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

/// Runs one context-length cell. Two independent timed loops share the
/// SAME `plan`:
///
/// - the BATCHED loop calls [`omega::metal::execute_plan_named_timed`]
///   (one shared command buffer for the whole 32-layer plan) and divides
///   its total `gpu_ns` by `LAYERS` -- this is the reported per-dispatch
///   number, and the KV bytes/GB/s below are derived from its median.
/// - the VALIDATION loop calls `execute_plan_named_op_timed` (one command
///   buffer per op) on every repeat, filters the returned records to
///   `kind == "cached_attention"` and asserts there are exactly `LAYERS`
///   of them -- every summed record is identified by its OWN kind field
///   from THIS SAME execution, never by a position borrowed from a
///   different bind.
fn run_cell(label: &str, context_length: u64) {
    let (program, symbols, roots, named) = build_fixture(context_length);
    let named_blocks = as_named_blocks(&named);

    let plan = omega::plan_named(
        &program,
        &symbols,
        &named_blocks,
        &roots,
        NumericPolicy::llama_relaxed(),
    )
    .expect("metal plans the real-shape fixture");

    omega::metal::execute_plan_named(&plan, &named_blocks).expect("plain warm-up run");

    omega::metal::execute_plan_named_timed(&plan, &named_blocks).expect("batched warm-up run");
    let mut batched_us_per_dispatch = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let started = Instant::now();
        let (_evaluated, total_gpu_ns) =
            omega::metal::execute_plan_named_timed(&plan, &named_blocks)
                .expect("batched timed run");
        let wall_us = started.elapsed().as_secs_f64() * 1e6;
        batched_us_per_dispatch.push(total_gpu_ns as f64 / LAYERS as f64 / 1e3);
        println!("ROW 376 {label} batched raw wall_us={wall_us:.2} total_gpu_ns={total_gpu_ns}");
    }
    let batched_stat = stats(&batched_us_per_dispatch);

    let (_evaluated, warm_op_timings) =
        omega::metal::execute_plan_named_op_timed(&plan, &named_blocks, None)
            .expect("per-op warm-up run");
    drop(warm_op_timings);
    let mut per_op_us_per_dispatch = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let (_evaluated, timings) =
            omega::metal::execute_plan_named_op_timed(&plan, &named_blocks, None)
                .expect("per-op validation run");
        let attention: Vec<&omega::metal::OpGpuTiming> = timings
            .iter()
            .filter(|timing| timing.kind == ATTENTION_KIND)
            .collect();
        assert_eq!(
            attention.len(),
            LAYERS as usize,
            "{label}: expected exactly {LAYERS} records with kind==\"{ATTENTION_KIND}\", found {}",
            attention.len()
        );
        let attention_gpu_ns: u64 = attention.iter().map(|timing| timing.gpu_ns).sum();
        per_op_us_per_dispatch.push(attention_gpu_ns as f64 / LAYERS as f64 / 1e3);
    }
    let per_op_stat = stats(&per_op_us_per_dispatch);

    let kv_bytes_per_dispatch = context_length * KV_HEADS as u64 * HEAD_DIM as u64 * 2 * 4;
    let effective_gbps = kv_bytes_per_dispatch as f64 / (batched_stat.median_us * 1e-6) / 1e9;
    println!(
        "ROW 376 {label} context_length={context_length} dispatches={LAYERS} \
         batched_us_per_dispatch: mean={:.3} median={:.3} min={:.3} max={:.3} cov={:.2}% \
         per_op_timed_kind_validated_us_per_dispatch: mean={:.3} median={:.3} min={:.3} max={:.3} cov={:.2}% \
         kv_bytes_per_dispatch={kv_bytes_per_dispatch} effective_gbps_from_batched={effective_gbps:.2}",
        batched_stat.mean_us,
        batched_stat.median_us,
        batched_stat.min_us,
        batched_stat.max_us,
        batched_stat.cov_pct,
        per_op_stat.mean_us,
        per_op_stat.median_us,
        per_op_stat.min_us,
        per_op_stat.max_us,
        per_op_stat.cov_pct,
    );
}

/// The block-staged Q·K/softmax/V rewrite (`omega::msl::block_width_for`,
/// gated by `NumericRewrite::TreeReduce`) reassociates the per-key
/// online-softmax fold into one combine per 32-lane sub-block. `bit_exact`
/// renders width `1` (today's strictly-sequential body, unchanged);
/// `llama_relaxed` renders the block-staged body this row's own timed cells
/// exercise but never checked against a reference. This function is the
/// functional parity check the timed cells above are missing: same fixture,
/// same plan shape, two numeric policies, one un-timed `execute_plan_named`
/// call each.
fn assert_block_staging_parity(context_length: u64) {
    let (program, symbols, roots, named) = build_fixture(context_length);
    let named_blocks = as_named_blocks(&named);

    let sequential_plan = omega::plan_named(
        &program,
        &symbols,
        &named_blocks,
        &roots,
        NumericPolicy::bit_exact(),
    )
    .expect("bit-exact (block_width=1) plan builds");
    let sequential = omega::metal::execute_plan_named(&sequential_plan, &named_blocks)
        .expect("bit-exact (block_width=1) run succeeds");

    let block_staged_plan = omega::plan_named(
        &program,
        &symbols,
        &named_blocks,
        &roots,
        NumericPolicy::llama_relaxed(),
    )
    .expect("llama_relaxed (block-staged) plan builds");
    let block_staged = omega::metal::execute_plan_named(&block_staged_plan, &named_blocks)
        .expect("llama_relaxed (block-staged) run succeeds");

    let expected = sequential.root();
    let actual = block_staged.root();
    assert_eq!(
        actual.len(),
        expected.len(),
        "context_length={context_length}"
    );

    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let max_diff = expected
        .iter()
        .zip(actual.iter())
        .map(|(want, got)| (want - got).abs())
        .fold(0.0f32, f32::max);
    let relative = max_diff / max_magnitude.max(f32::MIN_POSITIVE);
    println!(
        "ROW 376 block-staging parity context_length={context_length} \
         max_diff={max_diff} max_magnitude={max_magnitude} relative={relative}"
    );
    assert!(
        relative < 1e-4,
        "context_length={context_length}: block-staged (width>1) disagrees with the \
         sequential (width=1) kernel: relative={relative} max_diff={max_diff}"
    );
}

#[test]
fn block_staged_attention_holds_parity_at_context_40() {
    assert_block_staging_parity(40);
}

#[test]
fn block_staged_attention_holds_parity_at_context_512() {
    assert_block_staging_parity(512);
}

#[test]
fn block_staged_attention_holds_parity_at_context_4096() {
    assert_block_staging_parity(4096);
}

#[test]
#[ignore = "timed GPU cell, run explicitly per ROW 376's re-prove command"]
fn cached_attention_batched_context_40() {
    run_cell("a_context40", 40);
}

#[test]
#[ignore = "timed GPU cell, run explicitly per ROW 376's re-prove command"]
fn cached_attention_batched_context_512() {
    run_cell("b_context512", 512);
}

#[test]
#[ignore = "timed GPU cell, run explicitly per ROW 376's re-prove command"]
fn cached_attention_batched_context_4096() {
    run_cell("c_context4096", 4096);
}
