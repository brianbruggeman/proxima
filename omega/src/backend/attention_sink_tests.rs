//! The attention sink's Metal parity tests, in their own module because
//! `backend`'s `programmable_tests` is gated on the `cpu` feature, which is not
//! what AC8 and AC20 are about. A `--no-default-features --features metal` build
//! now compiles and runs them (`cargo val -p proxima-omega --no-default-features
//! --features metal --lib`, 623 tests discovered), which needed the
//! `Plan::Cpu`-absent cfg gaps in `backend.rs` closed first.

use super::*;
use crate::config::ResourceConstraints;
use proxima_tensor::QuantizedBlock;
use proxima_tensor::spec::attention_sink_fixture as sink;

/// Engine pinned to the Metal driver in `Configured` mode, so the plan is
/// `Plan::Metal` or an error -- never a silent CPU run.
fn sink_metal_config() -> RuntimeConfig {
    RuntimeConfig::builder()
        .with_engine("gpu")
        .with_gpu_driver("metal")
        .with_constraints(ResourceConstraints::default().with_max_host_tensor_bytes(Some(4096)))
        .build()
        .expect("the sink fixture config must validate")
}

/// AC8: the sink term's MSL emission and dispatch, against the SAME
/// hand-worked literal the CPU test pins
/// (`proxima_tensor::spec::attention_sink_fixture`) -- the two backends
/// validate one derivation, never each other (SPEC
/// `serve-flint-through-omega`'s own oracle rule).
///
/// The shape is the shared two-block combine (`append_local_block_and_combine`)
/// with sink presence on: two key heads with two query heads each, so the
/// sink's own address `group*u + g` is observable (with one key head `u` is
/// always 0 and a dropped coefficient is invisible), head 0's sink dominating
/// its scores and head 3's dominating by more so both sides of the incumbent's
/// `MAX(max, sink)` run.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn attention_sink_metal_parity_literal_matches_the_hand_worked_softmax_with_sink() {
    let (program, attended) = sink::sink_combine_program(true).expect("the sink combine lowers");
    let owned = sink::sink_combine_inputs(true);
    let values: Vec<(&str, QuantizedBlock<'_>)> = owned
        .iter()
        .map(|(name, data)| (*name, QuantizedBlock::Float32(data.as_slice())))
        .collect();
    let (mut plan, report) =
        plan_named_with_config(&sink_metal_config(), &program, &[], &values, &[attended])
            .expect("the sink combine must plan for Metal");
    assert_eq!(
        report[0].driver,
        Some(GpuDriver::Metal),
        "the sink term must be dispatched on Metal, not quietly run on the CPU"
    );
    // the report's driver is the REQUESTED one; the plan variant is what
    // decides dispatch, and the Metal driver has no CPU arm
    assert!(
        matches!(plan, Plan::Metal(_)),
        "the sink combine must plan as a Metal plan, not fall back to the CPU evaluator"
    );
    let executed =
        execute_plan_named(&mut plan, &values).expect("Metal must evaluate the sink combine");
    let (observed, shape) = executed
        .get(attended)
        .expect("the attended values are a requested output");
    assert_eq!(
        shape,
        [1, 2, 2, 2],
        "one query row x 2 key heads x 2 query heads x value width 2"
    );

    let expected = sink::hand_worked_attended(Some(sink::SINKS));
    assert!(
        sink::attended_agrees(observed, &expected),
        "Metal {observed:?} must equal the hand-worked softmax-with-sink {expected:?}"
    );
    // a program that bound the leaf and added nothing reproduces the sink-less
    // derivation instead; asserting it does NOT is what makes the run above
    // evidence the term is read rather than evidence the two agree by luck
    let without_sink = sink::hand_worked_attended(None);
    assert!(
        !sink::attended_agrees(observed, &without_sink),
        "Metal must not reproduce the sink-less literal {without_sink:?}"
    );
}

/// AC20: the same comparison the parity test runs rejects a perturbed literal,
/// on the same Metal residual -- so AC8 is not a comparison that can never
/// fail.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn control_attention_sink_metal_parity_rejects_a_perturbed_literal() {
    let (program, attended) = sink::sink_combine_program(true).expect("the sink combine lowers");
    let owned = sink::sink_combine_inputs(true);
    let values: Vec<(&str, QuantizedBlock<'_>)> = owned
        .iter()
        .map(|(name, data)| (*name, QuantizedBlock::Float32(data.as_slice())))
        .collect();
    let (mut plan, _) =
        plan_named_with_config(&sink_metal_config(), &program, &[], &values, &[attended])
            .expect("the sink combine must plan for Metal");
    let executed =
        execute_plan_named(&mut plan, &values).expect("Metal must evaluate the sink combine");
    let (observed, _) = executed
        .get(attended)
        .expect("the attended values are a requested output");

    let literal = sink::hand_worked_attended(Some(sink::SINKS));
    assert!(
        sink::attended_agrees(observed, &literal),
        "the literal the control perturbs must be the one Metal reproduces"
    );
    for position in 0..literal.len() {
        let mut perturbed = literal.clone();
        perturbed[position] += 1.0e-2;
        assert!(
            !sink::attended_agrees(observed, &perturbed),
            "a 1e-2 change at element {position} must be rejected"
        );
    }
}
