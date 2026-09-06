//! Proves ROW ### (`proxima-tensor/docs/discipline.md`): once a [`omega::Plan`]
//! has resolved its per-position pipelines, a LATER `execute_plan_with_placements`
//! call against the SAME plan never rebuilds `kernel_cache_key`'s `String`,
//! `kernel_dispatch_shape`'s `Vec<Binding>`, or `pipeline_for`'s own `format!`
//! cache-key lookup -- see `metal.rs`'s `resolve_steps`/`ResolvedStep` for the
//! mechanism. `#[global_allocator]` is process-wide, so this file is the only
//! consumer of `proxima_test::alloc_count` in this crate's test suite --
//! `cargo nextest` gives each test binary its own process, so no other test
//! shares this counter.

#![cfg(all(feature = "alloc-count", feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_test::alloc_count::{CountingAllocator, allocations, recorded_sizes, reset};
use proxima_tensor::{DType, Extent, IndexMap, Op, QuantizedBlock, ScalarOp, append, projection};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// `Input -> Identity -> Identity` -- two [`Op::Elementwise`] nodes, so
/// `Plan::prepared.resolved` holds two positions and this test can name a
/// genuine SECOND step distinct from the first.
fn two_step_identity_chain(extent: u32) -> (Vec<Op>, proxima_tensor::NodeId) {
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(extent)],
            name: None,
        },
    );
    let first = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: vec![(source, IndexMap::Affine(projection(1, &[0])))],
            name: None,
        },
    );
    let second = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: vec![(first, IndexMap::Affine(projection(1, &[0])))],
            name: None,
        },
    );
    (program, second)
}

/// The claim: on a plan-cache HIT (every call after the first against the
/// same [`omega::Plan`]), `execute_plan_with_placements` performs the SAME
/// number of allocations call after call, steady-state -- the per-step
/// `kernel_cache_key`/`kernel_dispatch_shape`/`pipeline_for` cost this test's
/// own doc names is either present on EVERY call (unbounded, old shape) or
/// absent from every call after the first (this landing's shape). Two
/// consecutive warm calls agreeing exactly is the observable difference: a
/// per-call allocation floor unrelated to op count cannot grow between them
/// either way, but a per-OP cost that scaled with `prepared.resolved.len()`
/// before this landing would still show up identically on every call too --
/// this test's second assertion (the absolute count against the two-op vs.
/// one-op plan) is what actually isolates the per-step marginal cost.
#[test]
fn a_warm_plan_hit_allocates_the_same_amount_every_call() {
    const EXTENT: u32 = 4;
    let (program, _root) = two_step_identity_chain(EXTENT);
    let block = [1.0f32, 2.0, 3.0, 4.0];
    let plan = omega::plan(&program, &[], &[QuantizedBlock::Float32(&block)], &[])
        .expect("plans the two-step identity chain");

    // Cold: builds `resolved_steps` from empty, compiles both kernels'
    // pipelines for the first time -- allocation-heavy by construction, not
    // part of this test's own claim.
    omega::execute_plan_with_placements(&plan, &[QuantizedBlock::Float32(&block)], &[], &[])
        .expect("first (cold) call warms the plan's pipeline cache");

    let before_second = allocations();
    omega::execute_plan_with_placements(&plan, &[QuantizedBlock::Float32(&block)], &[], &[])
        .expect("second (warm) call");
    let second_call_allocations = allocations() - before_second;

    let before_third = allocations();
    omega::execute_plan_with_placements(&plan, &[QuantizedBlock::Float32(&block)], &[], &[])
        .expect("third (warm) call");
    let third_call_allocations = allocations() - before_third;

    eprintln!(
        "second_call_allocations={second_call_allocations} third_call_allocations={third_call_allocations}"
    );
    assert_eq!(
        second_call_allocations, third_call_allocations,
        "two consecutive warm calls against the same unchanged plan must allocate \
         identically -- a per-step String/Vec cost that varied with anything but the \
         call's own fixed setup work would show up as drift here"
    );
}

/// Isolates the per-step marginal cost directly: a one-op plan and a
/// two-op plan differ ONLY in `prepared.resolved.len()` (one extra
/// `Elementwise` identity node) -- every other per-call cost
/// (`device_buffers`'s `BTreeMap`, `pending_faults`'s `Vec`, block upload)
/// is the same shape for both. `resolve_steps`/`encode_op`'s hit path reads
/// the already-built `Vec<ResolvedStep>` by position instead of rebuilding
/// `kernel_cache_key`/`kernel_dispatch_shape`/`pipeline_for`'s cache key per
/// op, so the SECOND (warm) call's allocation count must be identical for
/// both plans despite the extra op -- the per-step marginal is the number
/// this test reports, and it is 0 after this landing.
#[test]
fn a_warm_call_s_allocation_count_does_not_grow_with_extra_steps() {
    const EXTENT: u32 = 4;
    let block = [1.0f32, 2.0, 3.0, 4.0];

    let mut one_op_program = Vec::new();
    let source = append(
        &mut one_op_program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(EXTENT)],
            name: None,
        },
    );
    append(
        &mut one_op_program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: vec![(source, IndexMap::Affine(projection(1, &[0])))],
            name: None,
        },
    );
    let one_op_plan = omega::plan(
        &one_op_program,
        &[],
        &[QuantizedBlock::Float32(&block)],
        &[],
    )
    .expect("plans the one-op chain");

    let (two_op_program, _root) = two_step_identity_chain(EXTENT);
    let two_op_plan = omega::plan(
        &two_op_program,
        &[],
        &[QuantizedBlock::Float32(&block)],
        &[],
    )
    .expect("plans the two-op chain");

    // warm both plans' pipeline caches before measuring either.
    omega::execute_plan_with_placements(&one_op_plan, &[QuantizedBlock::Float32(&block)], &[], &[])
        .expect("warms the one-op plan");
    omega::execute_plan_with_placements(&two_op_plan, &[QuantizedBlock::Float32(&block)], &[], &[])
        .expect("warms the two-op plan");

    let before_one_op = allocations();
    omega::execute_plan_with_placements(&one_op_plan, &[QuantizedBlock::Float32(&block)], &[], &[])
        .expect("warm one-op call");
    let one_op_allocations = allocations() - before_one_op;

    let before_two_op = allocations();
    omega::execute_plan_with_placements(&two_op_plan, &[QuantizedBlock::Float32(&block)], &[], &[])
        .expect("warm two-op call");
    let two_op_allocations = allocations() - before_two_op;

    eprintln!("one_op_allocations={one_op_allocations} two_op_allocations={two_op_allocations}");
    assert_eq!(
        one_op_allocations, two_op_allocations,
        "a warm call's allocation count must not grow with an extra plan position -- \
         report BEFORE/AFTER: one_op={one_op_allocations} two_op={two_op_allocations}"
    );
}

/// Names ROW 303's residual: not just the COUNT of a warm step's allocations
/// but the SIZE of each one, in order, so the fix (or its owner-documented
/// exceptions) can be pinned to a concrete list rather than a bare integer.
/// `reset` before the warm call isolates this call's ring window from the
/// cold call and any earlier test in this binary.
#[test]
fn a_warm_plan_hit_s_allocations_are_named_by_size() {
    const EXTENT: u32 = 4;
    let (program, _root) = two_step_identity_chain(EXTENT);
    let block = [1.0f32, 2.0, 3.0, 4.0];
    let plan = omega::plan(&program, &[], &[QuantizedBlock::Float32(&block)], &[])
        .expect("plans the two-step identity chain");

    omega::execute_plan_with_placements(&plan, &[QuantizedBlock::Float32(&block)], &[], &[])
        .expect("first (cold) call warms the plan's pipeline cache");

    reset();
    omega::execute_plan_with_placements(&plan, &[QuantizedBlock::Float32(&block)], &[], &[])
        .expect("second (warm) call, the one under proof");
    let warm_call_allocations = allocations();
    let warm_call_sizes = recorded_sizes();

    reset();
    omega::execute_plan_with_placements(&plan, &[QuantizedBlock::Float32(&block)], &[], &[])
        .expect("third (warm) call, confirms the second call's count is steady-state");
    let third_call_allocations = allocations();
    let third_call_sizes = recorded_sizes();

    eprintln!(
        "warm_call_allocations={warm_call_allocations} warm_call_sizes={warm_call_sizes:?}"
    );
    eprintln!(
        "third_call_allocations={third_call_allocations} third_call_sizes={third_call_sizes:?}"
    );
    assert_eq!(
        warm_call_allocations,
        warm_call_sizes.len(),
        "the ring must have captured every allocation this call made"
    );
    assert_eq!(
        warm_call_sizes, third_call_sizes,
        "two consecutive warm calls must allocate the identical sizes in the identical \
         order -- a residual that grew or shrank between them would mean it is not yet \
         steady-state"
    );
    // ROW 303's residual: 18 allocations before this landing, reduced to 11 by
    // making the hazard-input list and the uniform-byte packing plan-owned
    // and reused (`Plan::hazard_state`, `Plan::uniform_scratch`) instead of
    // rebuilt every call -- see this landing's own commit for the removed
    // sizes. The remaining 11 are `device_buffers`, a fresh `BTreeMap` built
    // from scratch every call by this function's own design (its VALUES --
    // the block upload's device buffer -- can change call to call, unlike
    // the hazard/uniform scratch this landing reused) plus `finish`'s
    // read-back `Vec<f32>`, the actual output payload the caller receives,
    // not scratch the plan could own instead.
    assert_eq!(
        warm_call_allocations, 11,
        "a warm plan-hit step's allocation count moved -- update this assertion \
         alongside whatever fix (or regression) changed it: sizes in order: \
         {warm_call_sizes:?}"
    );
}
