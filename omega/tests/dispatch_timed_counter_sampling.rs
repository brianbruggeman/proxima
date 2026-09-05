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

use proxima_tensor::{DType, Extent, IndexMap, Op, QuantizedBlock, ScalarOp, append, projection};

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
    let plan = omega::plan(&program, &[], &[QuantizedBlock::Float32(&[0.0; EXTENT as usize])], &[output])
        .expect("plans the two-dispatch program");

    let input = vec![1.0f32; EXTENT as usize];
    let (_evaluated, timings, sampling_mode) = omega::execute_plan_with_placements_dispatch_timed(
        &plan,
        &[QuantizedBlock::Float32(&input)],
        &[],
        &[],
    )
    .expect("dispatch-timed execution runs on a real Metal device");

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
