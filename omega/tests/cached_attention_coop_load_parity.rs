//! Metal-vs-CPU parity for the cooperative-load rewrite of
//! `render_cached_attention` (`omega/src/msl.rs`'s doc), on the SINGLE-RANGE
//! fused kind specifically -- the nine-operand dynamic-`cached_len` shape
//! (`BoundOpKind::CachedAttention`'s own doc) `metal_real_forward.rs`'s
//! `fused_cached_attention_root_agrees_between_cpu_and_metal` never reaches,
//! since that fixture's `mistral_cached_forward_program` only ever produces
//! the eight-operand two-range kind.
//!
//! Run at three `kv-capacity-bucket` paddings (0, 1, 5 rows past the merged
//! `cached_len + new_count` length) because the cooperative load's stride is
//! driven by `head_dim`/`query_groups`, both baked at bind time from the
//! REAL band, never from the padded buffer shape -- a kernel that confused
//! the two would only diverge once the padding is nonzero, which is exactly
//! why `metal_agrees_with_cpu_on_a_nonempty_kv_cache` (padding-free) cannot
//! catch this class of bug on its own.

#![cfg(all(
    feature = "metal",
    feature = "cached-attention-streaming",
    target_os = "macos"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::cpu::evaluate_quantized_named_with_scratch;
use proxima_tensor::{BoundOpKind, NumericPolicy, bind};

mod support;
use support::{as_named_blocks, production_numeric_policy, real_single_range_forward_fixture_with_padding};

/// One padding value's worth of the parity check, shared by every case in
/// [`the_single_range_fused_kernel_holds_parity_at_every_kv_capacity_bucket_padding`]
/// so a failure names the exact padding it happened at without three copies
/// of the same body. `NumericPolicy::default()` (bit-exact) never admits
/// `ContextSplitMerge`, so `splits_for` is pinned at `1` here and the merge
/// path this file's `m_greater_than_one_prefill_holds_parity_past_the_split_knee`
/// exercises never engages -- this arm is purely the cooperative-load body.
fn assert_parity_at_padding(padding: u64) {
    assert_parity_at_shape(5, 1, padding, NumericPolicy::default());
}

/// [`assert_parity_at_padding`] generalized over `cached_len`/`new_count`/
/// `policy` so the M > 1 (prefill) case below can share the same body --
/// production (2026-09-07): a chat's second turn (~100 new rows atop ~600
/// cached keys, merged context 664 past the
/// `ATTENTION_SPLIT_KEYS_PER_SPLIT_AT_SCALE` (128) knee, under
/// `NumericPolicy::llama_relaxed()` -- the policy `ServingConfig::default()`
/// actually runs, which is what admits `ContextSplitMerge` and engages the
/// split+merge kernels at all) produced 1024 tokens of `!!!!` (token id 0),
/// while turn one (prefill from empty) was fine.
fn assert_parity_at_shape(cached_len: u64, new_count: u64, padding: u64, policy: NumericPolicy) {
    let (program, symbols, roots, owned) =
        real_single_range_forward_fixture_with_padding(cached_len, new_count, padding);
    let output_roots = [roots[0]];
    let named = as_named_blocks(&owned);

    let shapes = proxima_tensor::infer(&program, &symbols)
        .expect("single-range padded fixture infers");
    let resolved =
        bind(&program, &shapes, &output_roots, policy).expect("single-range padded fixture binds");
    let fused = resolved
        .iter()
        .find(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .expect("the single-range program fuses into CachedAttention");
    assert_eq!(
        fused.operands().len(),
        9,
        "cached_len={cached_len} new_count={new_count} padding={padding}: the single-range \
         candidate must carry the dynamic ninth `cached_len` operand, not the static \
         eight-operand two-range shape"
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
    .expect("cpu runs the padded single-range program");

    let plan = omega::plan_named(&program, &symbols, &named, &output_roots, policy)
        .expect("metal plans the padded single-range program");
    let metal = omega::execute_plan_named(&plan, &named)
        .expect("metal runs the padded single-range program on a real device");

    let expected = cpu.root();
    let actual = metal.root();
    assert_eq!(
        actual.len(),
        expected.len(),
        "cached_len={cached_len} new_count={new_count} padding={padding}"
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
    eprintln!(
        "single-range coop-load parity: cached_len={cached_len} new_count={new_count} \
         padding={padding} max_diff={max_diff} max_magnitude={max_magnitude} relative={relative}"
    );
    assert!(
        relative < 1e-4,
        "cached_len={cached_len} new_count={new_count} padding={padding}: metal disagrees with \
         cpu on the single-range fused root: relative={relative} max_diff={max_diff}"
    );
}

#[test]
fn the_single_range_fused_kernel_holds_parity_at_every_kv_capacity_bucket_padding() {
    for padding in [0u64, 1, 5] {
        assert_parity_at_padding(padding);
    }
}

/// ROW: a prefill dispatch (`query_rows > 1`) whose merged context (664)
/// sits past the split-at-scale knee (128) must still agree with the CPU
/// reference. `cached_attention_scratch_len` (`omega/src/metal.rs`) sizes
/// the scratch buffer by the LIVE `splits_for(664, ..) == 6`, but
/// `render_cached_attention`'s split kernel and `render_cached_attention_merge`
/// both address that buffer with the COMPILED `ATTENTION_SPLIT_MAX == 32` as
/// the per-row stride -- every `query_index > 0` writes past its allocated
/// 6-split slot, corrupting neighboring rows and, once `query_index` is a
/// handful of rows in, running past the buffer entirely.
#[test]
fn m_greater_than_one_prefill_holds_parity_past_the_split_knee() {
    assert_parity_at_shape(600, 64, 0, production_numeric_policy());
}
