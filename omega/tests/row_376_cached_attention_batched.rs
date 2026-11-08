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
//! builder, [`gqa_single_range_cached_forward_program`] -- the exact
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
//! `gqa_single_range_cached_forward_program` is the fused-single-range
//! builder that produces exactly that shape, never the two-range kind
//! `gqa_cached_forward_program` emits.
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

#[path = "support/row_376_fixture.rs"]
mod fixture;
mod support;

use fixture::build_fixture;
use proxima_numeric::NumericPolicy;
use support::as_named_blocks;

/// The block-staged Q·K/softmax/V rewrite (`omega::msl::block_width_for`,
/// gated by `NumericRewrite::TreeReduce`) reassociates the per-key
/// online-softmax fold into one combine per 32-lane sub-block. `bit_exact`
/// renders width `1` (today's strictly-sequential body, unchanged);
/// `llama_relaxed` renders the block-staged body this row's own timed cells
/// exercise but never checked against a reference. This function is the
/// functional parity check the timed cells (the `row_376_cached_attention_timed_cells` example) are missing: same fixture,
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
