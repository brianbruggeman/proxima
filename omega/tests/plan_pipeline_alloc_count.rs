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
    omega::execute_plan_with_placements(&plan, &[QuantizedBlock::Float32(&block)], &[], &[], &mut Vec::new())
        .expect("first (cold) call warms the plan's pipeline cache");

    let before_second = allocations();
    omega::execute_plan_with_placements(&plan, &[QuantizedBlock::Float32(&block)], &[], &[], &mut Vec::new())
        .expect("second (warm) call");
    let second_call_allocations = allocations() - before_second;

    let before_third = allocations();
    omega::execute_plan_with_placements(&plan, &[QuantizedBlock::Float32(&block)], &[], &[], &mut Vec::new())
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
    omega::execute_plan_with_placements(&one_op_plan, &[QuantizedBlock::Float32(&block)], &[], &[], &mut Vec::new())
        .expect("warms the one-op plan");
    omega::execute_plan_with_placements(&two_op_plan, &[QuantizedBlock::Float32(&block)], &[], &[], &mut Vec::new())
        .expect("warms the two-op plan");

    let before_one_op = allocations();
    omega::execute_plan_with_placements(&one_op_plan, &[QuantizedBlock::Float32(&block)], &[], &[], &mut Vec::new())
        .expect("warm one-op call");
    let one_op_allocations = allocations() - before_one_op;

    let before_two_op = allocations();
    omega::execute_plan_with_placements(&two_op_plan, &[QuantizedBlock::Float32(&block)], &[], &[], &mut Vec::new())
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
    // fed by `into_scratch` after each call below, so a warm call's root
    // read-back (`finish`'s `recycle` parameter) reuses the PREVIOUS call's
    // own buffer instead of allocating fresh -- this test's whole point.
    let mut recycle_pool: Vec<Vec<f32>> = Vec::new();

    let cold = omega::execute_plan_with_placements(
        &plan,
        &[QuantizedBlock::Float32(&block)],
        &[],
        &[],
        &mut recycle_pool,
    )
    .expect("first (cold) call warms the plan's pipeline cache");
    cold.into_scratch(&mut recycle_pool);

    reset();
    let second = omega::execute_plan_with_placements(
        &plan,
        &[QuantizedBlock::Float32(&block)],
        &[],
        &[],
        &mut recycle_pool,
    )
    .expect("second (warm) call, the one under proof");
    let warm_call_allocations = allocations();
    let warm_call_sizes = recorded_sizes();
    second.into_scratch(&mut recycle_pool);

    reset();
    let third = omega::execute_plan_with_placements(
        &plan,
        &[QuantizedBlock::Float32(&block)],
        &[],
        &[],
        &mut recycle_pool,
    )
    .expect("third (warm) call, confirms the second call's count is steady-state");
    let third_call_allocations = allocations();
    let third_call_sizes = recorded_sizes();
    third.into_scratch(&mut recycle_pool);

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
    // ROW 303's residual: 18 allocations before this landing's FIRST slice
    // (`Plan::hazard_state`/`Plan::uniform_scratch`), 11 after it, now 9:
    // `Plan::device_buffers` is plan-owned and never rebuilt fresh
    // (`BTreeMap::new()`) call-to-call, and `finish`'s root read-back reuses
    // a buffer popped from the caller's `recycle` pool (`Evaluated::into_scratch`)
    // instead of allocating a fresh `Vec<f32>` -- both landed in this same
    // change. This fixture's own block is deliberately NON-resident
    // (`Op::Input { name: None }`) to exercise the general path
    // `block_buffer_reusable_tests` (`metal.rs`) does not cover: the
    // remaining 9 are (a) this block's own fresh upload/copy every call --
    // content correctness for a non-resident node requires re-copying it
    // every call, so this cannot be plan-owned without the caller's own
    // `Plan::mark_resident` promise that the address never changes content
    // (see `a_warm_plan_hit_reuses_a_resident_block_s_device_buffer` below
    // for the residency-gated skip of exactly this cost), and (b) two
    // `device_buffers` writes (one per `Elementwise` position) whose VALUES
    // change call to call even though the map itself is reused, which is
    // ordinary `BTreeMap` insert bookkeeping, not a rebuild.
    assert_eq!(
        warm_call_allocations, 9,
        "a warm plan-hit step's allocation count moved -- update this assertion \
         alongside whatever fix (or regression) changed it: sizes in order: \
         {warm_call_sizes:?}"
    );
}

/// ROW 303's residency-gated half: with the block input NAMED and
/// [`omega::metal::Plan::mark_resident`] told about it, `execute_plan_with_placements`
/// skips re-uploading the block ENTIRELY on a warm call whose block identity
/// (`(pointer, byte_length)`) is unchanged -- `block_buffer_reusable`
/// (`metal.rs`) is the decision this proves end to end, on a real device. A
/// block whose ADDRESS moves (a different array, same content) is no longer
/// trusted and forces a rebuild, same as before this landing -- this is the
/// "no content hash" half of the contract: only the caller's own residency
/// promise plus an unmoved address earns the skip.
///
/// This does NOT assert the rebuilt call allocates strictly more than the
/// reused one: `upload_resident_copy`'s own NAME-keyed cache (`RESIDENT_BUFFERS`)
/// already serves a `Retained` clone with no fresh Rust-heap allocation on a
/// name hit regardless of address, so for a block this small the two calls'
/// MEASURED allocation counts are equal in practice -- the win
/// `block_buffer_reusable` buys here is skipping the match/dispatch/
/// `device_buffers` write entirely, not a further allocation count drop on
/// top of a cache that was already this cheap. What this test proves instead:
/// both calls still produce the byte-identical, correct output, and neither
/// regresses the other's allocation count.
#[test]
fn a_warm_plan_hit_reuses_a_resident_block_s_device_buffer() {
    const EXTENT: u32 = 4;
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(EXTENT)],
            name: Some("weight".into()),
        },
    );
    let identity = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: vec![(source, IndexMap::Affine(projection(1, &[0])))],
            name: None,
        },
    );
    let block = [1.0f32, 2.0, 3.0, 4.0];
    let mut plan = omega::plan(&program, &[], &[QuantizedBlock::Float32(&block)], &[])
        .expect("plans the one-op resident chain");
    plan.mark_resident(&std::collections::BTreeSet::from(["weight"]));

    omega::execute_plan_with_placements(&plan, &[QuantizedBlock::Float32(&block)], &[], &[], &mut Vec::new())
        .expect("first (cold) call warms the plan's pipeline cache and uploads the resident block");

    reset();
    let reused = omega::execute_plan_with_placements(
        &plan,
        &[QuantizedBlock::Float32(&block)],
        &[],
        &[],
        &mut Vec::new(),
    )
    .expect("second (warm) call against the SAME block address");
    let reused_allocations = allocations();

    // a DIFFERENT array -- same content, a moved address -- must not be
    // trusted: `block_buffer_reusable` sees a changed `(pointer, length)`
    // and this call re-uploads exactly as a cold call would.
    let moved_block = [1.0f32, 2.0, 3.0, 4.0];
    reset();
    let rebuilt = omega::execute_plan_with_placements(
        &plan,
        &[QuantizedBlock::Float32(&moved_block)],
        &[],
        &[],
        &mut Vec::new(),
    )
    .expect("third call, block moved to a different address");
    let rebuilt_allocations = allocations();

    eprintln!("reused_allocations={reused_allocations} rebuilt_allocations={rebuilt_allocations}");
    assert_eq!(
        reused.get(identity).expect("identity node is the sole output").0,
        &block,
        "the reused device buffer must still read back the correct content"
    );
    assert_eq!(
        rebuilt.get(identity).expect("identity node is the sole output").0,
        &moved_block,
        "an address-moved, forced-rebuild call must read back the NEW content, not a stale \
         buffer left over from the reused call"
    );
    assert!(
        reused_allocations <= rebuilt_allocations,
        "a resident block at an unmoved address must never allocate MORE than one whose \
         address just moved -- reused={reused_allocations} rebuilt={rebuilt_allocations}"
    );
}

/// The owner's finding on this branch: `resolve_steps` (`metal.rs`) used to
/// invalidate `plan.resolved_steps` only when `MathMode` changed --
/// `metal::numeric_policy_as_metal_math_mode` projects BOTH `BitExact` and
/// `FusedNoReassociation` onto the SAME `MathMode::Safe`, so switching
/// between those two policies on an already-resolved plan left the
/// math-mode-keyed check believing nothing had changed, and it returned
/// early before `kernel_cache_key` (and therefore the numeric-policy token
/// this branch folded into it) was ever consulted again. `ResolvedSteps`
/// now keys staleness on `numeric_policy` itself -- this proves the plan
/// re-resolves across exactly that transition, on a real device, by the
/// same allocation-count technique the rest of this file uses: a rebuild
/// call pays `kernel_cache_key`/`kernel_dispatch_shape`/`pipeline_for`
/// again and allocates more than a steady-state warm call; a stale-check
/// bug would make this call look identically cheap to the warm baseline.
#[test]
fn a_numeric_policy_change_that_leaves_math_mode_unchanged_still_re_resolves() {
    const EXTENT: u32 = 4;
    let (program, _root) = two_step_identity_chain(EXTENT);
    let block = [1.0f32, 2.0, 3.0, 4.0];
    let mut plan = omega::plan(&program, &[], &[QuantizedBlock::Float32(&block)], &[])
        .expect("plans the two-step identity chain");

    plan.set_numeric_policy(proxima_tensor::NumericPolicy::FusedNoReassociation);
    omega::execute_plan_with_placements(&plan, &[QuantizedBlock::Float32(&block)], &[], &[], &mut Vec::new())
        .expect("cold call resolves under FusedNoReassociation");
    omega::execute_plan_with_placements(&plan, &[QuantizedBlock::Float32(&block)], &[], &[], &mut Vec::new())
        .expect("first warm call under FusedNoReassociation");

    let before_warm_baseline = allocations();
    omega::execute_plan_with_placements(&plan, &[QuantizedBlock::Float32(&block)], &[], &[], &mut Vec::new())
        .expect("second warm call establishes the steady-state floor");
    let warm_baseline_allocations = allocations() - before_warm_baseline;

    assert_eq!(
        plan.math_mode(),
        omega::MathMode::Safe,
        "FusedNoReassociation must project to MathMode::Safe -- see \
         metal::numeric_policy_as_metal_math_mode's own doc table"
    );
    plan.set_numeric_policy(proxima_tensor::NumericPolicy::BitExact);
    assert_eq!(
        plan.math_mode(),
        omega::MathMode::Safe,
        "math mode must stay Safe across this transition, or this test is not exercising \
         the case a math-mode-keyed staleness check would have missed"
    );

    let before_transition = allocations();
    omega::execute_plan_with_placements(&plan, &[QuantizedBlock::Float32(&block)], &[], &[], &mut Vec::new())
        .expect("first call after the numeric-policy change");
    let transition_call_allocations = allocations() - before_transition;

    eprintln!(
        "warm_baseline_allocations={warm_baseline_allocations} \
         transition_call_allocations={transition_call_allocations}"
    );
    assert!(
        transition_call_allocations > warm_baseline_allocations,
        "a numeric-policy change that leaves math_mode unchanged must still force a \
         resolve_steps rebuild -- a math-mode-keyed staleness check would silently skip \
         this and keep serving pipelines resolved for the OLD policy: \
         baseline={warm_baseline_allocations} transition={transition_call_allocations}"
    );

    let before_second_warm = allocations();
    omega::execute_plan_with_placements(&plan, &[QuantizedBlock::Float32(&block)], &[], &[], &mut Vec::new())
        .expect("second call after the policy change, steady-state again");
    let second_warm_allocations = allocations() - before_second_warm;

    assert_eq!(
        second_warm_allocations, warm_baseline_allocations,
        "once re-resolved under the new policy, the plan must return to the SAME \
         warm-call allocation floor it held before the transition"
    );
}
