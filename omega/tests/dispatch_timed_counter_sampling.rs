//! Proves [`omega::execute_plan_with_placements_dispatch_timed`] measures
//! something real, inside the SAME single command buffer
//! [`omega::execute_plan_with_placements`] submits, rather than degrading
//! silently to zeroed timings: every position gets an [`omega::metal::OpGpuTiming`]
//! entry, `sampling_mode` names the counter-sampling point the device
//! actually reported support for (never a guess), and -- when the mode is
//! not `"unsupported"` -- at least one entry's `gpu_ns` is nonzero (the
//! degenerate-empty-profile check this workspace's N==0-is-RED discipline
//! asks for).

#![cfg(all(feature = "metal-output-placement", feature = "instrument", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::{
    DType, Extent, IndexMap, NumericPolicy, Op, QuantizedBlock, ScalarOp, append, projection,
};

/// `Input(extent) -> body_0(x) -> body_1(...) -> ... -> body_n(...)`, the
/// same shape [`decode_step_telemetry_budget`]'s `chained_unary_program`
/// builds: each stage a DISTINCT [`ScalarOp`], so each has its own
/// `kernel_cache_key` and none fuse into a shared kernel -- a per-position
/// table with more than one real row to disagree about.
///
/// Note: `prepare`/`emit` may fuse a linear elementwise chain like this one
/// into fewer resolved dispatches than `bodies.len()` -- this test asserts
/// the counter-sampling mechanism runs and reports real time, not a
/// specific fusion outcome.
fn chained_unary_program(extent: u32, bodies: &[ScalarOp]) -> (Vec<Op>, proxima_tensor::NodeId) {
    let mut program = Vec::new();
    let mut current = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(extent)],
            name: None,
        },
    );
    for body in bodies {
        current = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: *body,
                operands: vec![(current, IndexMap::Affine(projection(1, &[0])))],
                name: None,
            },
        );
    }
    (program, current)
}

#[test]
fn every_dispatch_gets_a_timing_and_a_named_sampling_mode() {
    const EXTENT: u32 = 8;
    let bodies = [ScalarOp::Identity, ScalarOp::Negate];
    let (program, output) = chained_unary_program(EXTENT, &bodies);
    let plan = omega::plan(&program, &[], &[QuantizedBlock::Float32(&[0.0; EXTENT as usize])], &[output], NumericPolicy::default())
        .expect("plans the two-dispatch program");

    let input = vec![1.0f32; EXTENT as usize];
    let (_evaluated, timings, sampling_mode, encoder_split_ns) =
        omega::execute_plan_with_placements_dispatch_timed(
            &plan,
            &[QuantizedBlock::Float32(&input)],
            &[],
            &[],
        )
        .expect("dispatch-timed execution runs on a real Metal device");
    assert!(
        encoder_split_ns.is_none(),
        "a plan with no `set_encoder_split_at` call must report `None`, never a fabricated split"
    );

    assert!(
        !timings.is_empty(),
        "a two-op program must produce at least one OpGpuTiming entry (the emitted plan may \
         fuse the chain into fewer dispatches than ops -- this asserts the mechanism runs, not \
         a specific fusion outcome)"
    );
    assert!(
        matches!(
            sampling_mode,
            "dispatch-boundary" | "stage-boundary" | "unsupported"
        ),
        "sampling_mode must name one of the three measured device states, got {sampling_mode:?}"
    );

    if sampling_mode != "unsupported" {
        let total_gpu_ns: u64 = timings.iter().map(|timing| timing.gpu_ns).sum();
        assert!(
            total_gpu_ns > 0,
            "device reported counter-sampling support ({sampling_mode}) but every dispatch's \
             gpu_ns was 0 -- a degenerate empty profile reads as RED, not quiet"
        );
    }
}

/// ROW 329: [`omega::metal::Plan::set_encoder_split_at`] ends the compute
/// encoder before the named position and opens a second one, instead of
/// ROW 309's original one-encoder-per-position stage-boundary fallback.
/// On this device (`AtStageBoundary`, not `AtDispatchBoundary`) that means
/// exactly two GPU-side stage-boundary samples instead of one per
/// position: this test asserts `encoder_split_ns` comes back `Some` with
/// both halves nonzero when the mode is not `"unsupported"`, and that
/// every per-position `OpGpuTiming::gpu_ns` reads `0` -- the split
/// samples describe encoder-level spans, not per-op ones, and this
/// function never fabricates a per-op number it did not measure.
#[test]
fn encoder_split_reports_two_nonzero_encoder_spans_and_zeroes_per_op_gpu_ns() {
    const EXTENT: u32 = 8;
    let bodies = [ScalarOp::Identity, ScalarOp::Negate, ScalarOp::Identity, ScalarOp::Negate];
    let (program, output) = chained_unary_program(EXTENT, &bodies);
    let mut plan = omega::plan(&program, &[], &[QuantizedBlock::Float32(&[0.0; EXTENT as usize])], &[output], NumericPolicy::default())
        .expect("plans the four-dispatch program");

    let input = vec![1.0f32; EXTENT as usize];
    plan.set_encoder_split_at(Some(1));
    let (_evaluated, timings, sampling_mode, encoder_split_ns) =
        omega::execute_plan_with_placements_dispatch_timed(
            &plan,
            &[QuantizedBlock::Float32(&input)],
            &[],
            &[],
        )
        .expect("split dispatch-timed execution runs on a real Metal device");

    if sampling_mode == "unsupported" {
        assert!(
            encoder_split_ns.is_none(),
            "an unsupported device must report no split timing, never a fabricated one"
        );
        return;
    }

    let (encoder_one_ns, encoder_two_ns) = encoder_split_ns.expect(
        "a split position on a device with real counter-sampling support must report both \
         encoder spans",
    );
    assert!(
        encoder_one_ns > 0 && encoder_two_ns > 0,
        "both encoder spans must carry real GPU time on a supported device, got \
         encoder_one_ns={encoder_one_ns} encoder_two_ns={encoder_two_ns}"
    );
    assert!(
        timings.iter().all(|timing| timing.gpu_ns == 0),
        "split mode's three encoder-level samples must never be misread as per-position ones"
    );
}
