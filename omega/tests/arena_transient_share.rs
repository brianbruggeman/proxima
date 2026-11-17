//! Plans of different row counts that run one after another on one thread lease one
//! transient arena buffer, and sharing it changes no output byte.
//!
//! A prefill of 4069 tokens runs 512-row chunks and a 485-row tail; each plan's packed
//! arena slot holds only outputs that die inside one call, so the tail plan reuses the
//! larger plan's buffer instead of holding a second one resident.

#![cfg(all(feature = "metal-output-placement", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use omega::metal::OUTPUT_BUFFER_ALLOCATIONS;
use proxima_numeric::{DType, NumericPolicy};
use proxima_tensor::{Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp, append, projection};

const LARGE_ROWS: u32 = 512 * 256;
const SMALL_ROWS: u32 = 485 * 256;

/// `Input -> Identity -> {Reduce(sum), Reduce(max)}`: the `rows`-long Identity output has two
/// readers, so it is not fused away, and it dies inside the call, so the packed layout puts it in the transient slot; the scalar
/// scalar roots keep slots of their own.
fn chain_program(rows: u32) -> (Vec<Op>, [NodeId; 2]) {
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(rows)],
            name: None,
        },
    );
    let copy = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Identity,
            operands: vec![(source, IndexMap::Affine(projection(1, &[0])))],
            name: None,
        },
    );
    let reduce = |program: &mut Vec<Op>, body: ScalarOp, init: ReduceInit| {
        append(
            program,
            Op::Reduce(Reduce {
                dtype: DType::FLOAT32,
                body,
                init,
                operand: copy,
                in_map: IndexMap::Affine(projection(1, &[0])),
                out_map: IndexMap::Affine(projection(1, &[])),
                keep: Keep::Reduce,
                name: None,
            }),
        )
    };
    let total = reduce(&mut program, ScalarOp::Add, ReduceInit::Zero);
    let peak = reduce(&mut program, ScalarOp::Maximum, ReduceInit::NegativeInfinity);
    (program, [total, peak])
}

fn plan_for(rows: u32) -> (omega::metal::Plan, [NodeId; 2], Vec<f32>) {
    let (program, roots) = chain_program(rows);
    let input: Vec<f32> = (0..rows).map(|index| (index % 7) as f32 - 3.0).collect();
    let plan = omega::plan(
        &program,
        &[],
        &[QuantizedBlock::Float32(&input)],
        &roots,
        NumericPolicy::default(),
    )
    .expect("plans the chain program");
    (plan, roots, input)
}

fn run(plan: &omega::metal::Plan, roots: [NodeId; 2], input: &[f32]) -> Vec<f32> {
    let evaluated = omega::execute_plan_with_placements(
        plan,
        &[QuantizedBlock::Float32(input)],
        &[],
        &[],
        &mut Vec::new(),
    )
    .expect("executes the chain program");
    roots
        .iter()
        .flat_map(|root| evaluated.get(*root).expect("root is an output").0.to_vec())
        .collect()
}

#[test]
fn a_smaller_plan_reuses_the_larger_plans_transient_arena_buffer() {
    let (large, large_roots, large_input) = plan_for(LARGE_ROWS);
    let (small, small_roots, small_input) = plan_for(SMALL_ROWS);

    let before = OUTPUT_BUFFER_ALLOCATIONS.get();
    let large_out = run(&large, large_roots, &large_input);
    let after_large = OUTPUT_BUFFER_ALLOCATIONS.get();
    let small_out = run(&small, small_roots, &small_input);
    let after_small = OUTPUT_BUFFER_ALLOCATIONS.get();

    let large_id = large.arena_transient_buffer_id().expect("large plan leases its transient slot");
    let small_id = small.arena_transient_buffer_id().expect("small plan leases its transient slot");
    assert_eq!(large_id, small_id, "both plans must hold one arena allocation");
    assert_eq!(
        (after_large - before) - (after_small - after_large),
        1,
        "the small plan must allocate exactly one device buffer fewer than the large plan \
         (large={}, small={})",
        after_large - before,
        after_small - after_large,
    );
    assert_eq!(large_out, vec![large_input.iter().sum::<f32>(), 3.0]);
    assert_eq!(small_out.len(), 2);
}

#[test]
fn shared_and_separate_arenas_produce_identical_outputs_when_plans_alternate() {
    let (large, large_roots, large_input) = plan_for(LARGE_ROWS);
    let (small, small_roots, small_input) = plan_for(SMALL_ROWS);

    let mut shared = Vec::new();
    for _ in 0..3 {
        shared.push((run(&large, large_roots, &large_input), run(&small, small_roots, &small_input)));
    }

    // a fresh thread has its own pool, so this plan owns an arena no other plan touches
    let separate = std::thread::spawn(move || {
        let (alone, alone_roots, alone_input) = plan_for(SMALL_ROWS);
        let out = run(&alone, alone_roots, &alone_input);
        assert!(alone.arena_transient_buffer_id().is_some());
        out
    })
    .join()
    .expect("separate-arena run completes");

    for (large_out, small_out) in &shared {
        assert_eq!(large_out.len(), 2);
        assert_eq!(large_out, &shared[0].0);
        assert_eq!(small_out, &separate, "shared arena must match a private arena byte for byte");
    }
    assert_eq!(large.arena_transient_buffer_id(), small.arena_transient_buffer_id());
}

#[test]
fn a_larger_plan_built_after_a_smaller_one_gets_its_own_sufficient_buffer() {
    let (small, small_roots, small_input) = plan_for(SMALL_ROWS);
    let small_out = run(&small, small_roots, &small_input);
    let (large, large_roots, large_input) = plan_for(LARGE_ROWS);
    let large_out = run(&large, large_roots, &large_input);

    assert_ne!(small.arena_transient_buffer_id(), large.arena_transient_buffer_id());
    assert_eq!(small_out.len(), 2);
    assert_eq!(large_out.len(), 2);
    assert!(small.arena_allocated_bytes() < large.arena_allocated_bytes());
}
