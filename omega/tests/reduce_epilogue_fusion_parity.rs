//! Metal-vs-CPU parity for a fused [`proxima_tensor::BoundOpKind::Reduce::
//! epilogue_body`] on the real single-range forward program
//! (`support::real_single_range_forward_fixture_with_padding`, the same
//! fixture `cached_attention_coop_load_parity.rs` uses), run at the same
//! three `kv-capacity-bucket` paddings that test sweeps: 0, 1, and 5 rows
//! past the merged `cached_len + new_count` length.
//!
//! With `reduce-epilogue-fusion` compiled into both `proxima-tensor` (via
//! this crate's own passthrough feature) and `omega`, `bind`'s post-pass
//! fuses at least one `Reduce`'s elementwise consumer into its epilogue on
//! this program (`docs/discipline.md`'s own census: 4/layer -- `global_max`,
//! `residual1`, `ffn_hidden`, `x_next`), so this test also asserts that
//! actually happened rather than silently passing on an unfused program.

#![cfg(all(
    feature = "metal",
    feature = "reduce-epilogue-fusion",
    target_os = "macos"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::cpu::evaluate_quantized_named_with_scratch;
use proxima_tensor::{BoundOpKind, bind};

mod support;
use support::{as_named_blocks, real_single_range_forward_fixture_with_padding};

fn epilogued_reduce_count(resolved: &[proxima_tensor::BoundOp]) -> usize {
    resolved
        .iter()
        .filter(|bound| {
            matches!(
                &bound.kind,
                BoundOpKind::Reduce {
                    epilogue_operands, ..
                } if !epilogue_operands.is_empty()
            )
        })
        .count()
}

/// One padding value's worth of the parity check -- same structure as
/// `cached_attention_coop_load_parity.rs`'s own `assert_parity_at_padding`,
/// naming the exact padding a failure happened at.
fn assert_parity_at_padding(padding: u64) {
    const CACHED_LEN: u64 = 5;
    const NEW_COUNT: u64 = 1;

    let (program, symbols, roots, owned) =
        real_single_range_forward_fixture_with_padding(CACHED_LEN, NEW_COUNT, padding);
    let output_roots = [roots[0]];
    let named = as_named_blocks(&owned);

    let shapes =
        proxima_tensor::infer(&program, &symbols).expect("single-range padded fixture infers");
    let resolved =
        bind(&program, &shapes, &output_roots).expect("single-range padded fixture binds");
    assert!(
        epilogued_reduce_count(&resolved) > 0,
        "padding={padding}: reduce-epilogue-fusion is compiled in but fused nothing on \
         the real single-range program -- either the fixture no longer carries a fusable \
         reduce+elementwise pair or the fusion pass regressed"
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

    let plan = omega::plan_named(&program, &symbols, &named, &output_roots)
        .expect("metal plans the padded single-range program");
    let metal = omega::execute_plan_named(&plan, &named)
        .expect("metal runs the padded single-range program on a real device");

    let expected = cpu.root();
    let actual = metal.root();
    assert_eq!(actual.len(), expected.len(), "padding={padding}");

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
        "reduce-epilogue-fusion parity: padding={padding} max_diff={max_diff} \
         max_magnitude={max_magnitude} relative={relative}"
    );
    assert!(
        relative < 1e-4,
        "padding={padding}: metal disagrees with cpu on the fused-epilogue root: \
         relative={relative} max_diff={max_diff}"
    );
}

/// Ignored while the broadcast-reduce epilogue's Metal renderer is
/// unwritten: `bind::reduce_epilogue_candidates` now ALSO fuses this real
/// program's own RMSNorm `x * inv_rms` tail (`bind.rs`'s
/// `BoundOpKind::Reduce::epilogue_broadcast_axes`, the composed-body-drop
/// fix this doc's own `epilogued_reduce_count` census predates), which
/// `omega::msl::render_reduce` correctly rejects with `EmitError::
/// EpilogueNotSupported` rather than emit a wrong kernel
/// (`render_reduce_rejects_a_broadcast_reduce_epilogue`, `msl.rs`, proves
/// the rejection itself). Re-enable once that renderer lands.
#[ignore = "broadcast-reduce epilogue Metal renderer not yet landed"]
#[test]
fn the_fused_epilogue_holds_parity_at_every_kv_capacity_bucket_padding() {
    for padding in [0u64, 1, 5] {
        assert_parity_at_padding(padding);
    }
}

/// Same shape as `metal_parity.rs`'s own determinism sweeps: the fused
/// epilogue kernel must produce the SAME bytes every dispatch, not merely
/// numbers within tolerance of each other -- a race in the epilogue's own
/// buffer indexing (a wrong `epi{index}` slot, a stale uniform) would show up
/// as run-to-run jitter before it ever failed the CPU-parity check above.
/// Ignored for the same reason as `the_fused_epilogue_holds_parity_at_every_
/// kv_capacity_bucket_padding` above: this program now includes a
/// broadcast-reduce epilogue site Metal has no renderer for yet.
#[ignore = "broadcast-reduce epilogue Metal renderer not yet landed"]
#[test]
fn the_fused_epilogue_is_byte_identical_across_twenty_dispatches() {
    const CACHED_LEN: u64 = 5;
    const NEW_COUNT: u64 = 1;

    let (program, symbols, roots, owned) =
        real_single_range_forward_fixture_with_padding(CACHED_LEN, NEW_COUNT, 0);
    let output_roots = [roots[0]];
    let named = as_named_blocks(&owned);

    let plan = omega::plan_named(&program, &symbols, &named, &output_roots)
        .expect("metal plans the single-range program");
    let first = omega::execute_plan_named(&plan, &named)
        .expect("metal runs the single-range program on a real device")
        .root()
        .to_vec();

    for run in 1..20 {
        let repeat = omega::execute_plan_named(&plan, &named)
            .expect("metal re-runs the single-range program on a real device");
        assert_eq!(
            repeat.root(),
            first.as_slice(),
            "run {run}: the fused-epilogue kernel produced different bytes on a repeat dispatch"
        );
    }
}
