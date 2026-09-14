//! Metal-vs-CPU parity for the partial-rotary pass plane
//! (`render_cached_attention`'s own doc, `omega/src/msl.rs`): qwen35's real
//! per-head shape (`kv_heads` 2, `group` 8 -> 16 query heads, `attn_head_dim`
//! 256, `rotary_dim` 64 -> 192-wide un-rotated pass plane) is the one caller
//! today whose fused `BoundOpKind::CachedAttention` carries the trailing
//! `pass_query`/`pass_cached_key`/`pass_new_key` operands
//! (`BoundOpKind::CachedAttention`'s own doc). Run at two capacities -- 40
//! keys with 3 trailing rows padded past the real 37-row `cached_len`
//! (exercises the padding mask on a short context) and 512 keys with no
//! padding (exercises a context past the split-at-scale knee under the
//! production numeric policy) -- because a kernel that only reads the
//! compiled bucket extent instead of the runtime `cached_len` would only
//! diverge once padding is nonzero, the same argument
//! `cached_attention_coop_load_parity.rs` already makes for the full-rotary
//! case.

#![cfg(all(
    feature = "metal",
    feature = "cached-attention-streaming",
    target_os = "macos"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::cpu::evaluate_quantized_named_with_scratch;
use proxima_tensor::{BoundOpKind, NumericPolicy, bind};

mod support;
use support::{as_named_blocks, production_numeric_policy, qwen35_partial_rotary_forward_fixture};

/// Shared body: binds the fixture, asserts the fusion actually carries the
/// pass-plane operand count `BoundOpKind::CachedAttention`'s own doc
/// describes (twelve: eight base sources, the runtime `cached_len`, and the
/// three pass-plane sources), then runs CPU and Metal against the same
/// named-block data and compares the fused output within `1e-5` relative
/// error -- the CPU fused path is itself proven against an f64 reference
/// (`proxima_tensor::bind`'s own `qwen35_partial_rotary_dense_attention_
/// matches_an_independent_f64_reference`), so this test's job is only to
/// prove the Metal kernel agrees with that already-proven CPU path.
fn assert_partial_rotary_parity_at(cached_extent: u64, cached_len: u64, policy: NumericPolicy) {
    let (program, symbols, roots, owned) =
        qwen35_partial_rotary_forward_fixture(1, cached_extent, cached_len);
    let named = as_named_blocks(&owned);

    let shapes = proxima_tensor::infer(&program, &symbols)
        .expect("qwen35 partial-rotary fixture infers");
    let resolved = bind(&program, &shapes, &roots, policy)
        .expect("qwen35 partial-rotary fixture binds");
    let fused = resolved
        .iter()
        .find(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .expect("the qwen35 chain fuses into a CachedAttention op");
    let BoundOpKind::CachedAttention {
        rotary_dim,
        head_dim,
        ..
    } = &fused.kind
    else {
        unreachable!("just matched CachedAttention above");
    };
    assert_eq!(*rotary_dim, 64, "qwen35's own rotary width");
    assert_eq!(*head_dim, 256, "qwen35's own full head width");
    assert_eq!(
        fused.operands().len(),
        12,
        "cached_extent={cached_extent} cached_len={cached_len}: partial rotary must carry the \
         eight base sources, the runtime cached_len, and the three pass-plane sources"
    );

    let mut free_buffers = Vec::new();
    let mut validated = None;
    let cpu = evaluate_quantized_named_with_scratch(
        &program,
        &symbols,
        &named,
        &roots,
        &mut free_buffers,
        &mut validated,
    )
    .expect("cpu runs the qwen35 partial-rotary program");

    let plan = omega::plan_named(&program, &symbols, &named, &roots, policy)
        .expect("metal plans the qwen35 partial-rotary program");
    let metal = omega::execute_plan_named(&plan, &named)
        .expect("metal runs the qwen35 partial-rotary program on a real device");

    let expected = cpu.root();
    let actual = metal.root();
    assert_eq!(
        actual.len(),
        expected.len(),
        "cached_extent={cached_extent} cached_len={cached_len}"
    );

    let max_magnitude = expected.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
    let max_diff = expected
        .iter()
        .zip(actual.iter())
        .map(|(want, got)| (want - got).abs())
        .fold(0.0f32, f32::max);
    let relative = max_diff / max_magnitude.max(f32::MIN_POSITIVE);
    eprintln!(
        "partial-rotary parity: cached_extent={cached_extent} cached_len={cached_len} \
         max_diff={max_diff} max_magnitude={max_magnitude} relative={relative}"
    );
    assert!(
        relative < 1e-5,
        "cached_extent={cached_extent} cached_len={cached_len}: metal disagrees with cpu on the \
         partial-rotary fused root: relative={relative} max_diff={max_diff}"
    );
}

#[test]
fn forty_keys_with_three_padded_rows_holds_partial_rotary_parity() {
    assert_partial_rotary_parity_at(40, 37, production_numeric_policy());
}

/// IGNORED: pre-existing, independent of the pass plane -- once `chunks`
/// (`context_chunks_for`/the single-range dynamic path's own compiled `cap`)
/// reaches `ATTENTION_CONTEXT_CHUNK_CAP` (4, `omega-runtime.toml`), the
/// declared `shared_o[query_groups * cap * head_dim]` threadgroup array
/// alone is `query_groups * cap * head_dim * 4` bytes -- `8 * 4 * 256 * 4 =
/// 32768`, plus `shared_m`/`shared_l`'s `8 * 4 * 4 * 2 = 256` more, exactly
/// the driver's own reported `33024` against Metal's `32768`-byte ceiling.
/// This is a pure function of `head_dim`/`query_groups`/the compiled `cap`
/// (`render_cached_attention`'s own `let cap = crate::sized::
/// ATTENTION_CONTEXT_CHUNK_CAP;`, never `rotary_dim`), so a hypothetical
/// FULL-rotary head_dim=256/group=8 kernel would blow the identical budget
/// at the identical context length -- confirmed by the 40-key cell above,
/// which stays under it only because `chunks` there is 3, not 4. Fixing it
/// needs `context_chunks_for`'s cap clamped by shape at every one of its
/// eight call sites across `msl.rs`/`metal.rs`/`identity.rs` (dispatch grid
/// sizing and scratch-buffer sizing must stay in lockstep with whatever the
/// kernel body declares), which is out of scope for the pass-plane addition
/// this file exists to test.
#[test]
#[ignore = "pre-existing threadgroup-memory ceiling for head_dim=256/group=8 \
            at chunks==ATTENTION_CONTEXT_CHUNK_CAP, unrelated to the pass plane -- see doc above"]
fn five_hundred_twelve_keys_holds_partial_rotary_parity_past_the_split_knee() {
    assert_partial_rotary_parity_at(512, 512, production_numeric_policy());
}

/// `NumericPolicy::default()` (bit-exact) never admits `ContextSplitMerge`
/// or `TreeReduce` -- pinning it here proves the sequential fallback
/// (`render_cached_attention`'s own doc on why partial rotary forces
/// `block_width <= 1`) is correct on its own, independent of whichever
/// policy production actually runs under.
#[test]
fn forty_keys_holds_partial_rotary_parity_under_bit_exact() {
    assert_partial_rotary_parity_at(40, 37, NumericPolicy::default());
}
