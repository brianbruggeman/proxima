//! `omega::backend`'s whole reason to exist: the SAME program and SAME
//! named blocks, run through the SAME wrapper on both compiled backends,
//! never naming `proxima_tensor::cpu` or `omega::metal` at the call site.
//!
//! Reuses `metal_real_forward.rs`'s real forward graph construction (see
//! `support::real_forward_fixture`) rather than a second copy — the two
//! gates would otherwise drift on which named block gets which random seed.

#![cfg(all(feature = "cpu", feature = "metal", target_os = "macos"))]
// every expect below runs against data this test just built or a real
// device call; a failure there IS the test failing, not a case to recover.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use omega::backend::{Backend, execute_plan_named, plan_named};

mod support;
use support::{as_named_blocks, real_forward_fixture};

#[test]
fn the_wrapper_agrees_with_itself_across_cpu_and_metal() {
    const VOCAB: usize = 64;

    let (program, symbols, roots, owned) = real_forward_fixture();
    let named = as_named_blocks(&owned);

    let mut cpu_plan = plan_named(Backend::Cpu, &program, &symbols, &named, &roots)
        .expect("omega::backend plans the real forward on cpu");
    let cpu = execute_plan_named(&mut cpu_plan, &named)
        .expect("omega::backend runs the real forward on cpu");

    let mut metal_plan = plan_named(Backend::Metal, &program, &symbols, &named, &roots)
        .expect("omega::backend plans the real forward on metal");
    let metal = execute_plan_named(&mut metal_plan, &named)
        .expect("omega::backend runs the real forward on a real device");

    let expected = cpu.root();
    let actual = metal.root();
    assert_eq!(
        actual.len(),
        VOCAB,
        "degenerate gate: logits must be one row of the vocabulary"
    );
    assert_eq!(actual.len(), expected.len());

    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(got.is_finite(), "metal, via the wrapper, produced a non-finite logit: {got}");
        max_diff = max_diff.max((got - want).abs());
    }
    let max_magnitude = expected.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
    let relative = max_diff / max_magnitude.max(f32::MIN_POSITIVE);
    eprintln!(
        "backend wrapper real forward: max_diff={max_diff} max_magnitude={max_magnitude} relative={relative}"
    );
    assert!(
        relative < 1e-4,
        "omega::backend's cpu and metal arms disagree on the real forward: relative={relative} max_diff={max_diff}"
    );
}

#[test]
fn the_same_process_runs_one_plan_on_cpu_and_the_next_on_metal() {
    // proves selection is per-call, not a process-wide "current backend":
    // both calls happen in the same test process, back to back, with no env
    // var touched in between.
    let (program, symbols, roots, owned) = real_forward_fixture();
    let named = as_named_blocks(&owned);

    let mut cpu_plan = plan_named(Backend::Cpu, &program, &symbols, &named, &roots)
        .expect("cpu plans first");
    let _cpu = execute_plan_named(&mut cpu_plan, &named).expect("cpu executes first");

    let mut metal_plan = plan_named(Backend::Metal, &program, &symbols, &named, &roots)
        .expect("metal plans immediately after, same process");
    let _metal = execute_plan_named(&mut metal_plan, &named).expect("metal executes immediately after");
}
