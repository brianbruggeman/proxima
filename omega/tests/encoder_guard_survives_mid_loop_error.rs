//! Reproduces the crash behind `-[_MTLCommandEncoder dealloc]:134: failed
//! assertion 'Command encoder released without endEncoding'` and proves
//! `omega::metal`'s `EncoderGuard` (`omega/src/metal.rs`) closes it.
//!
//! Every `execute*` entry point used to open ONE compute encoder and call
//! `endEncoding()` exactly once, textually after its per-`BoundOp` loop. A
//! `?` from inside that loop skipped the call, and the `Retained` encoder
//! handle's `dealloc` at the next autorelease-pool drain hit the ObjC
//! runtime's own assertion — a `SIGTRAP` that killed the process and
//! swallowed the real Rust `Err`. This file builds a plan whose FIRST
//! dispatched op is small and succeeds, and whose LAST op requests an
//! output buffer no real device will hand out (a two-axis broadcast product,
//! `EACH_AXIS_ELEMENTS^2` elements — see that constant's own doc for why a
//! single huge axis alone was not enough), so `encode_op`'s `?` returns from
//! `execute_plan` after the small op has already been encoded into the
//! guarded encoder — the exact "mid-loop" shape the crash needed. If the
//! guard is doing its job, this test process is still alive to observe
//! `Err` at all.
//!
//! An integration test (not `#[cfg(test)]`) so it always runs a real
//! process the ObjC assertion could actually kill, rather than living
//! inside the library's own nextest process where a `SIGTRAP` reads as one
//! more failed test instead of the harness itself dying.

#![cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::{DType, Extent, IndexMap, Op, QuantizedBlock, ScalarOp, append, map};

/// A single-axis Iota at `u32::MAX` elements (a ~17.2 GiB single-buffer
/// request) was NOT enough to force `newBufferWithLength_options` to
/// return `None` on real hardware — Apple Silicon's unified memory backs a
/// `StorageModeShared` buffer lazily, so a 17 GiB *request* alone does not
/// commit 17 GiB of physical pages. Two independent `u32`-width axes
/// broadcast together (an outer product, the same "different projection
/// per operand" shape `fused_matmul_kernel`/`embedding_matmul_kernel` in
/// `metal_compile_gate.rs` already use) reach a genuinely impossible
/// element count without needing either axis's OWN data larger than
/// `EACH_AXIS_ELEMENTS` (each Iota's own buffer is cheap: `EACH_AXIS_ELEMENTS`
/// elements, not `EACH_AXIS_ELEMENTS^2`).
const EACH_AXIS_ELEMENTS: u32 = 300_000;

#[test]
fn execute_plan_survives_an_error_encoded_after_the_first_op() {
    let mut program = Vec::new();
    let small_input = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: None,
        },
    );
    let small_output = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: vec![(small_input, IndexMap::Affine(map::projection(1, &[0])))],
            name: None,
        },
    );
    let axis_a = append(
        &mut program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(EACH_AXIS_ELEMENTS),
        },
    );
    let axis_b = append(
        &mut program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(EACH_AXIS_ELEMENTS),
        },
    );
    let impossible_output = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (axis_a, IndexMap::Affine(map::projection(2, &[0]))),
                (axis_b, IndexMap::Affine(map::projection(2, &[1]))),
            ],
            name: None,
        },
    );

    let symbols: Vec<u64> = Vec::new();
    let outputs = [small_output, impossible_output];
    let blocks = [QuantizedBlock::Float32(&[1.0, 2.0, 3.0, 4.0])];

    let plan = omega::plan(
        &program,
        &symbols,
        &blocks,
        &outputs,
        proxima_tensor::NumericPolicy::default(),
    )
    .expect("planning is pure CPU-side shape/symbol work — it never touches the device");

    let result = omega::execute_plan(&plan, &blocks);

    assert!(
        result.is_err(),
        "the impossible-size Iota output must fail inside encode_op, not silently succeed"
    );
    let error = result.expect_err("checked above");
    println!("execute_plan_survives_an_error_encoded_after_the_first_op: got {error}");

    // Reaching here at all is the assertion that matters: on the pre-guard
    // code, the `Retained` encoder handle's autorelease-pool drain would
    // have raised `-[_MTLCommandEncoder dealloc]` before this line ever ran.
    let second_result = omega::execute_plan(&plan, &blocks);
    assert!(
        second_result.is_err(),
        "the guard must not have poisoned the plan/device for a second call"
    );
}
