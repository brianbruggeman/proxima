//! wgpu/WGSL-vs-CPU parity for a fused [`proxima_tensor::BoundOpKind::Reduce::
//! epilogue_body`] on the real single-range forward program -- the WGSL
//! counterpart of `reduce_epilogue_fusion_parity.rs`'s Metal gate, run
//! through the same `omega::backend::{plan_named, execute_plan_named}`
//! wrapper `wgpu_parity.rs` uses (see `crate::wgsl::push_reduce_epilogue_write`,
//! ROW 294: the wgsl emitter renders this fused epilogue now instead of
//! rejecting it).
//!
//! With `reduce-epilogue-fusion` compiled into both `proxima-tensor` (via
//! this crate's own passthrough feature) and `omega`, `bind`'s post-pass
//! fuses at least one `Reduce`'s elementwise consumer into its epilogue on
//! this program -- so this test also asserts that actually happened rather
//! than silently passing on an unfused program.

#![cfg(all(feature = "cpu", feature = "wgpu-backend", feature = "reduce-epilogue-fusion"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use omega::backend::{Engine, GpuDriver, execute_plan_named, plan_named};
use proxima_tensor::{BoundOpKind, NumericPolicy, bind};

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

/// Ignored: this program's own RMSNorm `x * inv_rms` tail fuses into a
/// broadcast-reduce epilogue (`bind::BoundOpKind::Reduce::epilogue_
/// broadcast_axes`) now that `bind`'s fusion pass matches it, and
/// `crate::wgsl::render_reduce` correctly rejects that shape with
/// `EmitError::EpilogueNotSupported` rather than silently emit a kernel
/// that reads/writes the wrong element count (this renderer never widened
/// its `epilogue_operand_strides`/output write past `output_rank`, unlike
/// Metal's `push_broadcast_epilogue_write`). Re-enable once WGSL grows the
/// same widened write.
#[ignore = "broadcast-reduce epilogue WGSL renderer not yet landed"]
#[test]
fn the_fused_epilogue_holds_parity_on_wgpu_against_cpu() {
    const CACHED_LEN: u64 = 5;
    const NEW_COUNT: u64 = 1;

    let (program, symbols, roots, owned) =
        real_single_range_forward_fixture_with_padding(CACHED_LEN, NEW_COUNT, 0);
    let output_roots = [roots[0]];
    let named = as_named_blocks(&owned);

    let shapes = proxima_tensor::infer(&program, &symbols).expect("single-range fixture infers");
    let resolved = bind(&program, &shapes, &output_roots, NumericPolicy::default())
        .expect("single-range fixture binds");
    assert!(
        epilogued_reduce_count(&resolved) > 0,
        "reduce-epilogue-fusion is compiled in but fused nothing on the real \
         single-range program -- either the fixture no longer carries a fusable \
         reduce+elementwise pair or the fusion pass regressed"
    );

    let mut cpu_plan = plan_named(
        Engine::Cpu,
        None,
        &program,
        &symbols,
        &named,
        &output_roots,
        NumericPolicy::default(),
    )
    .expect("cpu plans the single-range program");
    let cpu = execute_plan_named(&mut cpu_plan, &named).expect("cpu runs the single-range program");

    let mut wgpu_plan = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Wgpu),
        &program,
        &symbols,
        &named,
        &output_roots,
        NumericPolicy::default(),
    )
    .expect("wgpu plans the single-range program");
    let wgpu = execute_plan_named(&mut wgpu_plan, &named)
        .expect("wgpu runs the single-range program on a real device");

    let expected = cpu.root();
    let actual = wgpu.root();
    assert_eq!(actual.len(), expected.len());

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
        "wgsl reduce-epilogue-fusion parity: max_diff={max_diff} max_magnitude={max_magnitude} \
         relative={relative}"
    );
    assert!(
        relative < 1e-4,
        "wgpu disagrees with cpu on the fused-epilogue root: relative={relative} \
         max_diff={max_diff}"
    );
}
