//! CPU-vs-Metal parity for the two-range `BoundOpKind::CachedAttention`
//! kernel on the real Qwen3-1.7B shape (GQA plus per-head QK-norm), the
//! shape `metal_real_forward.rs`'s own two-range fixture never exercises
//! (that one is Mistral-shaped: GQA but no QK-norm). Named per
//! `generate.rs:6620-6630`'s own doc: the next diagnostic step for the
//! "direct decode of a real Qwen3-1.7B checkpoint is garbage on Metal" defect
//! is comparing the fused kernel against the CPU oracle on identical bytes
//! before suspecting anything else in the two-range step.
//!
//! Swept over `cached_len` in `{0, 5, 33}` and `new_count` (query_rows) in
//! `{1, 4}` -- six cells, each its own bind + CPU run + Metal run on
//! byte-identical named blocks (same RNG seed per position, so `q`/`k`/`v`/
//! cache data is identical between the two evaluators; only the evaluator
//! differs).

#![cfg(all(
    feature = "metal",
    feature = "cached-attention-streaming",
    feature = "instrument",
    target_os = "macos"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::cpu::evaluate_quantized_named_with_scratch;
use proxima_tensor::instrument::{path_totals, reset_path};
use proxima_tensor::{BoundOpKind, NumericPolicy, bind, infer};

mod support;
use support::{as_named_blocks, production_numeric_policy, qwen3_gqa_qk_norm_forward_fixture};

/// One `(cached_len, new_count, policy)` cell: binds the qwen3 GQA+QK-norm
/// fixture, asserts it actually fuses into `BoundOpKind::CachedAttention` (a
/// non-fusing cell would make a parity pass meaningless -- it would be
/// comparing two copies of the SAME unfused elementwise/reduce chain, never
/// touching the kernel this test exists to check), runs the CPU oracle and
/// the real Metal device on byte-identical named blocks, and returns the
/// logits root's max-abs diff.
fn parity_cell(cached_len: u64, new_count: u64, policy: NumericPolicy) -> f32 {
    let (program, symbols, roots, owned) = qwen3_gqa_qk_norm_forward_fixture(new_count, cached_len);
    let output_roots = [roots[0]];
    let named = as_named_blocks(&owned);

    let shapes = infer(&program, &symbols).expect("qwen3 gqa+qk_norm fixture infers");
    let resolved =
        bind(&program, &shapes, &output_roots, policy).expect("qwen3 gqa+qk_norm fixture binds");
    assert!(
        resolved
            .iter()
            .any(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. })),
        "cached_len={cached_len} new_count={new_count}: fixture must fuse into \
         BoundOpKind::CachedAttention or this parity check exercises the wrong kernel"
    );

    let mut free_buffers = Vec::new();
    let mut validated = None;
    let cpu = evaluate_quantized_named_with_scratch(
        &program,
        &symbols,
        &named,
        &output_roots,
        &mut free_buffers,
        &mut validated,
    )
    .expect("cpu runs the qwen3 gqa+qk_norm fixture");

    reset_path();
    let plan = omega::plan_named(&program, &symbols, &named, &output_roots, policy)
        .expect("metal plans the qwen3 gqa+qk_norm fixture");
    let metal = omega::execute_plan_named(&plan, &named)
        .expect("metal runs the qwen3 gqa+qk_norm fixture on a real device");
    assert!(
        path_totals().op_kind_cached_attention >= 1,
        "cached_len={cached_len} new_count={new_count}: metal execution must record at \
         least one CachedAttention op kind now that omega::metal calls record_op_kind"
    );

    let max_diff = cpu
        .root()
        .iter()
        .zip(metal.root())
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "qwen3_gqa_qk_norm_two_range_parity cached_len={cached_len} new_count={new_count} max_diff={max_diff}"
    );
    max_diff
}

#[test]
fn the_two_range_cached_attention_kernel_holds_parity_on_the_qwen3_gqa_qk_norm_shape() {
    let mut worst: Option<(u64, u64, f32)> = None;
    for cached_len in [0u64, 5, 33] {
        for new_count in [1u64, 4] {
            let max_diff = parity_cell(cached_len, new_count, NumericPolicy::default());
            if worst.is_none_or(|(_, _, previous)| max_diff > previous) {
                worst = Some((cached_len, new_count, max_diff));
            }
            assert!(
                max_diff < 1e-3,
                "cached_len={cached_len} new_count={new_count}: metal disagrees with cpu on \
                 the qwen3 gqa+qk_norm two-range fused root: max_diff={max_diff}"
            );
        }
    }
    let (cached_len, new_count, max_diff) = worst.expect("six cells always run");
    eprintln!(
        "qwen3_gqa_qk_norm_two_range_parity worst cell: cached_len={cached_len} \
         new_count={new_count} max_diff={max_diff}"
    );
}

/// The regression this file exists to pin: `omega::msl::cached_attention_
/// merge_needed` (`omega/src/msl.rs:3336`) used to answer purely from
/// `cached_key_rows + new_key_rows >= ATTENTION_SPLIT_KEYS_PER_SPLIT_AT_SCALE`
/// (128) under `NumericPolicy::llama_relaxed()` -- true for BOTH the
/// single-range nine-operand kind (`cached_key_rows == 0`, which actually
/// implements the `ContextSplitMerge` split/merge protocol) and this
/// two-range kind (`cached_key_rows != 0`), which does not: `render_cached_
/// attention`'s two-range branches always write the complete, correctly
/// normalized result straight to `out[]` in one pass. `emit`/
/// `kernel_dispatch_shape` then swapped the op's bindings to `split_bindings_
/// with_scratch` and `emit_cached_attention_merge` dispatched a companion
/// merge kernel that reinterpreted the already-correct `out[]` bytes as
/// `(max, sum, weighted[head_dim])` triples and overwrote them with garbage.
///
/// MEASURED before the fix (`omega/src/msl.rs`'s `cached_attention_merge_
/// needed` gaining the `single_range_dynamic` gate): `cached_len=200,
/// new_count=1` max_diff=12.626673; `cached_len=700, new_count=1`
/// max_diff=10.351054 (`NumericPolicy::llama_relaxed()`, this exact fixture).
/// `cached_len=5` (below the 128-key split-at-scale knee, `splits_for`
/// returns `1`, no merge dispatch) was noise-floor both before and after --
/// included here as the negative control. All four cells are noise-floor
/// (`< 1e-3`) after the fix.
#[test]
fn the_two_range_cached_attention_kernel_holds_parity_under_production_policy_past_the_split_knee()
{
    let mut worst: Option<(u64, u64, f32)> = None;
    for cached_len in [5u64, 200, 700] {
        for new_count in [1u64, 4] {
            let max_diff = parity_cell(cached_len, new_count, production_numeric_policy());
            if worst.is_none_or(|(_, _, previous)| max_diff > previous) {
                worst = Some((cached_len, new_count, max_diff));
            }
            assert!(
                max_diff < 1e-3,
                "cached_len={cached_len} new_count={new_count}: metal disagrees with cpu on \
                 the qwen3 gqa+qk_norm two-range fused root under production_numeric_policy(): \
                 max_diff={max_diff} -- this is the ContextSplitMerge/two-range regression \
                 (omega/src/msl.rs:3336's cached_attention_merge_needed) if it comes back"
            );
        }
    }
    let (cached_len, new_count, max_diff) = worst.expect("six cells always run");
    eprintln!(
        "qwen3_gqa_qk_norm_two_range_parity_production_policy worst cell: cached_len={cached_len} \
         new_count={new_count} max_diff={max_diff}"
    );
}
