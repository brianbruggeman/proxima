//! Proves the `metal-output-placement` capability: an op's output can land
//! directly in a buffer the CALLER owns, at a byte offset the caller
//! chooses, and that buffer survives across separate
//! [`omega::execute_plan_with_placements`] calls.
//!
//! Two runs of the SAME plan write into the SAME caller-owned
//! [`omega::PlacedBuffer`] at two different offsets — the first at `0`, the
//! second at a NON-ZERO offset — and this test reads the buffer back once,
//! after both runs, and asserts it holds both runs' data side by side. That
//! is the two things `execute_plan`'s per-call fresh `allocate_buffer` could
//! never do: persist across calls, and land somewhere other than its own
//! byte `0`.

#![cfg(all(feature = "metal-output-placement", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::{DType, Extent, IndexMap, Op, QuantizedBlock, ScalarOp, append, projection};

/// `Input(extent) -> Elementwise(Identity)` — the smallest program whose
/// output is byte-exact-equal to its input, so parity between what was
/// uploaded and what a placed read reports back is a single float
/// comparison, not a numerical-tolerance one.
fn identity_program(extent: u32) -> (Vec<Op>, proxima_tensor::NodeId) {
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(extent)],
            name: None,
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
    (program, identity)
}

#[test]
fn a_placed_buffer_holds_both_runs_data_at_their_own_offsets() {
    const EXTENT: u32 = 4;
    const ELEMENT_BYTES: usize = size_of::<f32>();
    const RUN_BYTES: usize = EXTENT as usize * ELEMENT_BYTES;
    const SECOND_RUN_OFFSET: usize = RUN_BYTES;

    let (program, identity_node) = identity_program(EXTENT);
    let plan = omega::plan(&program, &[], &[QuantizedBlock::Float32(&[0.0; EXTENT as usize])], &[])
        .expect("plans the identity program once, reused by both runs below");

    let buffer = omega::allocate_placed_buffer(RUN_BYTES * 2)
        .expect("allocates one caller-owned buffer sized for both runs' outputs");

    let first_run = [1.0f32, 2.0, 3.0, 4.0];
    omega::execute_plan_with_placements(
        &plan,
        &[QuantizedBlock::Float32(&first_run)],
        &[],
        &[(identity_node, &buffer, 0)],
    )
    .expect("first run writes into the placed buffer at offset 0");

    let second_run = [10.0f32, 20.0, 30.0, 40.0];
    omega::execute_plan_with_placements(
        &plan,
        &[QuantizedBlock::Float32(&second_run)],
        &[],
        &[(identity_node, &buffer, SECOND_RUN_OFFSET)],
    )
    .expect("second run, against the SAME plan and buffer, writes at a non-zero offset");

    let at_offset_zero = omega::read_placed_buffer_f32(&buffer, 0, EXTENT as usize);
    let at_second_offset =
        omega::read_placed_buffer_f32(&buffer, SECOND_RUN_OFFSET, EXTENT as usize);

    assert_eq!(
        at_offset_zero, first_run,
        "the first run's data must still be at offset 0 after the second run executed"
    );
    assert_eq!(
        at_second_offset, second_run,
        "the second run's data must be readable at its own non-zero offset"
    );
    assert_ne!(
        at_offset_zero, at_second_offset,
        "degenerate gate: two different inputs must not have produced identical bytes"
    );
}

/// The KV-cache shape this whole feature exists for, collapsed to the
/// smallest program that still exercises it: within ONE
/// [`omega::execute_plan_with_placements`] call, op A writes this token's
/// fresh row into a placed buffer at a NON-ZERO offset, and a LATER op B —
/// same call, same serial `MTLComputeCommandEncoder` — reads the buffer's
/// FULL range from offset `0`, covering both the pre-existing prefix (from
/// an earlier call) and A's just-written suffix. If sequential encode order
/// plus Metal's default hazard tracking did not cover a write and a read at
/// DIFFERENT offsets of the SAME resource, op B would read stale
/// (pre-write) bytes for the suffix instead of what A wrote; it does not.
#[test]
fn a_program_reads_a_placed_write_from_a_later_op_in_the_same_call() {
    const PREFIX_LEN: u32 = 4;
    const ROW_LEN: u32 = 4;
    const TOTAL_LEN: u32 = PREFIX_LEN + ROW_LEN;
    const ELEMENT_BYTES: usize = size_of::<f32>();
    const ROW_OFFSET: usize = PREFIX_LEN as usize * ELEMENT_BYTES;

    let buffer = omega::allocate_placed_buffer(TOTAL_LEN as usize * ELEMENT_BYTES)
        .expect("allocates one buffer sized for the prefix plus one new row");

    // Seed the "pre-existing prefix" the way a real caller would have: an
    // EARLIER `execute_plan_with_placements` call, proven safe by the test
    // above this one.
    let (seed_program, seed_node) = identity_program(PREFIX_LEN);
    let seed_plan = omega::plan(
        &seed_program,
        &[],
        &[QuantizedBlock::Float32(&[0.0; PREFIX_LEN as usize])],
        &[],
    )
    .expect("plans the prefix-seeding program");
    let prefix = [100.0f32, 200.0, 300.0, 400.0];
    omega::execute_plan_with_placements(
        &seed_plan,
        &[QuantizedBlock::Float32(&prefix)],
        &[],
        &[(seed_node, &buffer, 0)],
    )
    .expect("seeds the buffer's prefix in its own prior call");

    // The real program: op A (row_out) writes at the non-zero row offset;
    // op B (cache_out), later in program order, reads the cache node's
    // FULL range starting at offset 0 -- both encoded into the SAME call's
    // SAME serial encoder.
    let mut program = Vec::new();
    let row_in = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(ROW_LEN)],
            name: None,
        },
    );
    let row_out = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: vec![(row_in, IndexMap::Affine(projection(1, &[0])))],
            name: None,
        },
    );
    let cache_in = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(TOTAL_LEN)],
            name: None,
        },
    );
    let cache_out = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: vec![(cache_in, IndexMap::Affine(projection(1, &[0])))],
            name: None,
        },
    );
    assert!(
        row_out.0 < cache_out.0,
        "degenerate gate: op A (row_out) must be encoded before op B (cache_out) for this test \
         to prove anything about within-call write-then-read ordering"
    );

    // `row_out` must be a declared output too, not just a
    // `output_placements` entry: `omega::metal::prepare`'s `prune_dead`
    // pass drops any node unreachable from the plan's own declared
    // `outputs`, and a placed output has no graph edge to `cache_out` at
    // all (the two are only aliased through the shared `PlacedBuffer`,
    // invisible to the tensor graph) -- see
    // `execute_plan_with_placements`'s own doc.
    let plan = omega::plan(
        &program,
        &[],
        &[
            QuantizedBlock::Float32(&[0.0; ROW_LEN as usize]),
            QuantizedBlock::Float32(&[0.0; TOTAL_LEN as usize]),
        ],
        &[row_out, cache_out],
    )
    .expect("plans the read-after-placed-write program");

    let new_row = [11.0f32, 22.0, 33.0, 44.0];
    let unused_cache_block = [0.0f32; TOTAL_LEN as usize];
    let evaluated = omega::execute_plan_with_placements(
        &plan,
        &[
            QuantizedBlock::Float32(&new_row),
            // ignored: `cache_in` is input-placed below, so this position's
            // block is never uploaded -- present only to keep
            // `block_nodes.iter().zip(blocks.iter())` aligned.
            QuantizedBlock::Float32(&unused_cache_block),
        ],
        &[(cache_in, &buffer, 0)],
        &[(row_out, &buffer, ROW_OFFSET)],
    )
    .expect("op A's write and op B's read of the same buffer execute in one call");

    let (actual, _shape) = evaluated
        .get(cache_out)
        .expect("cache_out was requested as this plan's output");
    let mut expected = prefix.to_vec();
    expected.extend_from_slice(&new_row);
    assert_eq!(
        actual, expected,
        "op B must see the pre-existing prefix AND op A's fresh write, in the same call"
    );
    assert!(
        evaluated.get(row_out).is_none(),
        "row_out is a placed output that is NOT this plan's root (cache_out, the last \
         program node, is) -- its bytes already live in the caller's own PlacedBuffer, so \
         `finish` must not copy them into `Evaluated` too; a `Some` here would mean the \
         skip-placed-readback optimization regressed"
    );
    let placed_bytes = omega::read_placed_buffer_f32(&buffer, ROW_OFFSET, ROW_LEN as usize);
    assert_eq!(
        placed_bytes, new_row,
        "the placed write itself must still have happened -- only the redundant \
         Evaluated copy of row_out is skipped, not the GPU dispatch that produced it"
    );
}
