//! The missing oracle `proxima-tensor/docs/discipline.md`'s horizontal-
//! packed-merge design note (ROW 566/567) named but never built: a synthetic
//! multi-dispatch [`omega::Plan`] that actually runs `execute_plan_with_placements`
//! on a real device, so gate (2)'s claim ("N independent packed-row matvecs
//! collapse into one `grid.z` dispatch, byte-identical output, every hazard
//! output still recorded") is measured, not argued.
//!
//! Eight independent Q4_K `[2048 x 512]` matvecs share ONE weight buffer (a
//! distinct static byte offset per round) and ONE activation buffer, writing
//! into ONE output buffer at eight distinct offsets -- exactly
//! [`omega::execute_plan_with_placements`]'s existing `input_placements`/
//! `output_placements` capability (`metal_output_placement.rs`), not a new
//! mechanism. `metal-horizontal-merge` OFF is this file's baseline: 8 real
//! GPU dispatches, 8 hazard-written outputs, output bit-exact vs the
//! independent dequantize+dot reference every packed-row test in this crate
//! already holds itself to (`q4k_matmul_layout.rs`'s own convention).

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use core::mem::size_of;

use objc2_metal::MTLBuffer;
use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, dequantize, quantize};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, Reduce, ReduceInit, ScalarOp, append,
    projection,
};

const ROUNDS: usize = 8;
const IN_DIM: usize = 2048;
const OUT_DIM: usize = 512;

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

/// `Multiply`-then-`Add`, weight declared `[in_dim, out_dim]` -- the same
/// reduction-axis-first convention `q4k_matmul_layout.rs`'s own
/// `matmul_program` uses, so this fixture's byte layout is proven correct by
/// a test already in this suite rather than a fresh, unverified convention.
fn append_matvec(program: &mut Vec<Op>, weight: NodeId, activation: NodeId) -> NodeId {
    let product = append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(2, &[1, 0]))),
                (activation, IndexMap::Affine(projection(2, &[1]))),
            ],
            name: None,
        },
    );
    append(
        program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(projection(2, &[0, 1])),
            out_map: IndexMap::Affine(projection(2, &[0])),
            keep: Keep::Reduce,
            name: None,
        }),
    )
}

fn pack_rows(rows: &[Vec<f32>]) -> Vec<u8> {
    let blocks_per_row = IN_DIM / QK_K;
    let mut packed = vec![0u8; rows.len() * blocks_per_row * BLOCK_BYTES];
    for (row, row_packed) in rows
        .iter()
        .zip(packed.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
    {
        quantize(row, row_packed).expect("IN_DIM is a whole multiple of QK_K");
    }
    packed
}

fn expected_output(packed: &[u8], activation: &[f32]) -> Vec<f32> {
    let blocks_per_row = IN_DIM / QK_K;
    let mut expected = Vec::with_capacity(OUT_DIM);
    for row_packed in packed.chunks_exact(blocks_per_row * BLOCK_BYTES) {
        let mut row = vec![0.0f32; IN_DIM];
        dequantize(row_packed, &mut row).expect("packed row dequantizes");
        expected.push(
            row.iter()
                .zip(activation.iter())
                .map(|(weight, value)| weight * value)
                .sum(),
        );
    }
    expected
}

unsafe fn write_placed_buffer(buffer: &omega::PlacedBuffer, offset: usize, bytes: &[u8]) {
    let pointer = buffer.contents();
    // SAFETY: `buffer` is `storageModeShared` (`allocate_placed_buffer`'s own
    // contract) and the caller sized it to hold `offset + bytes.len()`.
    unsafe {
        core::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            pointer.as_ptr().cast::<u8>().add(offset),
            bytes.len(),
        );
    }
}

/// The whole synthetic fixture: `ROUNDS` independent matvecs, all inputs and
/// outputs caller-placed so this proves the SAME shape a real merge would
/// see (weight/activation/output buffer identity) rather than one this
/// harness's own defaults happen to produce. Returns the plan, every
/// placement, and the per-round expected (independent-reference) output so
/// both the merged and unmerged path reuse one fixture.
struct Fixture {
    plan: omega::Plan,
    weight_buffer: omega::PlacedBuffer,
    activation_buffer: omega::PlacedBuffer,
    output_buffer: omega::PlacedBuffer,
    activation_node: NodeId,
    weight_nodes: Vec<NodeId>,
    weight_names: Vec<String>,
    output_nodes: Vec<NodeId>,
    expected: Vec<Vec<f32>>,
}

fn build_fixture() -> Fixture {
    let mut program = Vec::new();
    // weight nodes declared BEFORE the activation node -- `q4k_matmul_layout.rs`'s
    // own `matmul_program` convention this fixture's doc already claims to
    // follow. `build_merged_dispatch` hardcodes `bound.operands()[0]` as the
    // weight and `[1]` as the activation (`omega/src/metal.rs:1096-1097`);
    // declaring activation first gave every weight a HIGHER `NodeId` than
    // the activation, which silently swapped that positional assumption and
    // made every merge candidate fail `packed_operands.contains_key` --
    // `metal-horizontal-merge` never merged anything, always falling
    // through to 8 ordinary dispatches regardless of the feature.
    let weight_nodes: Vec<NodeId> = (0..ROUNDS)
        .map(|round| {
            append(
                &mut program,
                Op::Input {
                    dtype: DType::UInt8,
                    shape: vec![Extent::Static(IN_DIM as u32), Extent::Static(OUT_DIM as u32)],
                    name: Some(format!("weight_{round}")),
                },
            )
        })
        .collect();
    let activation_node = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(IN_DIM as u32)],
            name: Some("activation".into()),
        },
    );
    let output_nodes: Vec<NodeId> = weight_nodes
        .iter()
        .map(|&weight| append_matvec(&mut program, weight, activation_node))
        .collect();

    let activation = random_vec(97, IN_DIM);
    let mut placed_input_nodes = vec![activation_node];
    placed_input_nodes.extend(weight_nodes.iter().copied());

    // A placed node's own `QuantizedBlock` content is never read (the real
    // bytes come from the placement below) -- naming it here is the ONLY way
    // to mark a placed weight `Q4K`-packed rather than the placement path's
    // own `Float32` default (`resolve_named_blocks_with_placed_nodes`'s own
    // doc): `prepare`'s dtype gate and `packed_operands_of` both key off
    // this codec tag, not off the placement. `execute_plan_named_with_placements`
    // needs the SAME pairing again on every call (its own `blocks` array is
    // rebuilt fresh, not cached on the `Plan`), so `weight_names` travels in
    // the fixture and `Fixture::named` rebuilds this exact list.
    let weight_names: Vec<String> = (0..ROUNDS).map(|round| format!("weight_{round}")).collect();
    let mut named: Vec<(&str, proxima_tensor::QuantizedBlock<'_>)> =
        vec![("activation", proxima_tensor::QuantizedBlock::Float32(&[]))];
    named.extend(
        weight_names
            .iter()
            .map(|name| (name.as_str(), proxima_tensor::QuantizedBlock::Q4K(&[]))),
    );

    let plan = omega::plan_named_with_placed_inputs(
        &program,
        &[],
        &named,
        &output_nodes,
        NumericPolicy::default(),
        &placed_input_nodes,
    )
    .expect("plans the eight independent matvecs");

    let weight_bytes_per_round = (IN_DIM / QK_K) * BLOCK_BYTES * OUT_DIM;
    let weight_buffer = omega::allocate_placed_buffer(weight_bytes_per_round * ROUNDS)
        .expect("allocates one shared weight buffer for all eight rounds");
    let activation_buffer =
        omega::allocate_placed_buffer(IN_DIM * size_of::<f32>()).expect("allocates the activation buffer");
    let output_buffer = omega::allocate_placed_buffer(OUT_DIM * size_of::<f32>() * ROUNDS)
        .expect("allocates one shared output buffer for all eight rounds");

    unsafe {
        write_placed_buffer(
            &activation_buffer,
            0,
            core::slice::from_raw_parts(activation.as_ptr().cast::<u8>(), core::mem::size_of_val(activation.as_slice())),
        );
    }

    let mut expected = Vec::with_capacity(ROUNDS);
    for round in 0..ROUNDS {
        let rows: Vec<Vec<f32>> = (0..OUT_DIM)
            .map(|row| random_vec(1_000_000 + round as u64 * 10_000 + row as u64, IN_DIM))
            .collect();
        let packed = pack_rows(&rows);
        assert_eq!(packed.len(), weight_bytes_per_round, "fixture's own byte-length arithmetic");
        unsafe {
            write_placed_buffer(&weight_buffer, round * weight_bytes_per_round, &packed);
        }
        // proves the WRITE side of this fixture, independent of the GPU
        // read this test's own assertion exercises below -- a host-side
        // readback of the exact bytes just written, at this round's own
        // offset.
        unsafe {
            let readback = core::slice::from_raw_parts(
                weight_buffer.contents().as_ptr().cast::<u8>().add(round * weight_bytes_per_round),
                packed.len(),
            );
            assert_eq!(readback, packed.as_slice(), "fixture wrote the wrong bytes at round {round}'s own offset");
        }
        expected.push(expected_output(&packed, &activation));
    }

    Fixture {
        plan,
        weight_buffer,
        activation_buffer,
        output_buffer,
        activation_node,
        weight_nodes,
        weight_names,
        output_nodes,
        expected,
    }
}

impl Fixture {
    /// The same `(name, QuantizedBlock)` pairing `build_fixture` planned
    /// with -- `execute_plan_named_with_placements` needs it EVERY call, not
    /// just at plan time, to size its own per-call `blocks` array
    /// (`resolve_named_blocks_with_placed_inputs`'s own zip against
    /// `block_nodes`); the content is still never read for a placed node.
    fn named(&self) -> Vec<(&str, proxima_tensor::QuantizedBlock<'_>)> {
        let mut named: Vec<(&str, proxima_tensor::QuantizedBlock<'_>)> =
            vec![("activation", proxima_tensor::QuantizedBlock::Float32(&[]))];
        named.extend(
            self.weight_names
                .iter()
                .map(|name| (name.as_str(), proxima_tensor::QuantizedBlock::Q4K(&[]))),
        );
        named
    }

    fn input_placements(&self) -> Vec<(NodeId, &omega::PlacedBuffer, usize)> {
        let weight_bytes_per_round = (IN_DIM / QK_K) * BLOCK_BYTES * OUT_DIM;
        let mut placements = vec![(self.activation_node, &self.activation_buffer, 0)];
        placements.extend(
            self.weight_nodes
                .iter()
                .enumerate()
                .map(|(round, &node)| (node, &self.weight_buffer, round * weight_bytes_per_round)),
        );
        placements
    }

    fn output_placements(&self) -> Vec<(NodeId, &omega::PlacedBuffer, usize)> {
        self.output_nodes
            .iter()
            .enumerate()
            .map(|(round, &node)| (node, &self.output_buffer, round * OUT_DIM * size_of::<f32>()))
            .collect()
    }

    fn assert_outputs_match_reference(&self, relative_tolerance: f32) {
        for round in 0..ROUNDS {
            // `read_placed_buffer_f32` takes a byte offset, not an element
            // count -- round 0 (offset 0) hid this by coincidence.
            let actual = omega::read_placed_buffer_f32(
                &self.output_buffer,
                round * OUT_DIM * size_of::<f32>(),
                OUT_DIM,
            );
            for (index, (&value, &reference)) in actual.iter().zip(self.expected[round].iter()).enumerate() {
                let scale = reference.abs().max(f32::MIN_POSITIVE);
                let relative = (value - reference).abs() / scale;
                assert!(
                    relative < relative_tolerance,
                    "round {round} row {index}: got={value} reference={reference} relative={relative}"
                );
            }
        }
    }
}

/// The baseline this whole design note measures against: `metal-horizontal-merge`
/// OFF (this crate's own default), eight real dispatches, output bit-exact vs
/// the independent dequantize+dot reference.
#[test]
fn eight_independent_matvecs_run_unmerged_today() {
    let fixture = build_fixture();
    let named = fixture.named();
    let input_placements = fixture.input_placements();
    let output_placements = fixture.output_placements();
    omega::execute_plan_named_with_placements(
        &fixture.plan,
        &named,
        &input_placements,
        &output_placements,
    )
    .expect("eight independent matvecs execute");
    fixture.assert_outputs_match_reference(1e-2);
}

/// Gate (2)'s two remaining, unmeasured claims from this file's own module
/// doc: every one of the 8 positions still gets its own hazard bookkeeping
/// (`HAZARD_STEP_CALLS`), REGARDLESS of how many real dispatches that
/// collapses to (`ENCODE_DISPATCH_CALLS`: 8 with `metal-horizontal-merge`
/// off, 1 on). `PROXIMA_ROW570_DUMP`, when set, writes every round's raw
/// `f32` output bytes to that path so a caller can `cmp` two runs of this
/// same test built under different feature sets -- the only way to compare
/// a compile-time feature's on/off output from outside a single process.
#[test]
fn dispatch_and_hazard_counts_match_the_design_note() {
    let fixture = build_fixture();
    let named = fixture.named();
    let input_placements = fixture.input_placements();
    let output_placements = fixture.output_placements();
    #[cfg(feature = "instrument")]
    {
        let _ = omega::metal::ENCODE_DISPATCH_CALLS.snapshot_and_reset();
        let _ = omega::metal::HAZARD_STEP_CALLS.snapshot_and_reset();
    }
    omega::execute_plan_named_with_placements(
        &fixture.plan,
        &named,
        &input_placements,
        &output_placements,
    )
    .expect("eight independent matvecs execute");
    fixture.assert_outputs_match_reference(1e-2);
    #[cfg(feature = "instrument")]
    {
        let expected_dispatches: u64 = if cfg!(feature = "metal-horizontal-merge") { 1 } else { 8 };
        assert_eq!(
            omega::metal::ENCODE_DISPATCH_CALLS.snapshot_and_reset(),
            expected_dispatches,
            "metal-horizontal-merge={}: dispatch count",
            cfg!(feature = "metal-horizontal-merge")
        );
        assert_eq!(
            omega::metal::HAZARD_STEP_CALLS.snapshot_and_reset(),
            ROUNDS as u64,
            "every one of the 8 positions must still run its own hazard bookkeeping"
        );
    }
    if let Ok(path) = std::env::var("PROXIMA_ROW570_DUMP") {
        let mut bytes = Vec::with_capacity(ROUNDS * OUT_DIM * size_of::<f32>());
        for round in 0..ROUNDS {
            let values = omega::read_placed_buffer_f32(
                &fixture.output_buffer,
                round * OUT_DIM * size_of::<f32>(),
                OUT_DIM,
            );
            bytes.extend(values.iter().flat_map(|value| value.to_le_bytes()));
        }
        std::fs::write(&path, &bytes).expect("writes the row570 on/off comparison dump");
    }
}

/// The mixed-buffer refusal `split_by_shared_buffers`'s own doc names: round
/// 7's weight is copied into its OWN, separate buffer instead of the shared
/// one every other round uses. `metal-horizontal-merge` must still merge the
/// other 7 (one dispatch) and fall the mismatched one through to its own
/// ordinary dispatch -- 2 dispatches total, never 1 (would silently read the
/// wrong weight) and never 8 (would defeat the merge for 7 rounds that had
/// no reason to split).
#[test]
fn mixed_buffer_member_does_not_merge() {
    let fixture = build_fixture();
    let weight_bytes_per_round = (IN_DIM / QK_K) * BLOCK_BYTES * OUT_DIM;
    let odd_one_out = ROUNDS - 1;
    let separate_weight_buffer =
        omega::allocate_placed_buffer(weight_bytes_per_round).expect("allocates the odd-one-out's own weight buffer");
    unsafe {
        let source = core::slice::from_raw_parts(
            fixture
                .weight_buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(odd_one_out * weight_bytes_per_round),
            weight_bytes_per_round,
        );
        write_placed_buffer(&separate_weight_buffer, 0, source);
    }
    let named = fixture.named();
    let mut input_placements = fixture.input_placements();
    input_placements[1 + odd_one_out] = (fixture.weight_nodes[odd_one_out], &separate_weight_buffer, 0);
    let output_placements = fixture.output_placements();
    #[cfg(feature = "instrument")]
    let _ = omega::metal::ENCODE_DISPATCH_CALLS.snapshot_and_reset();
    omega::execute_plan_named_with_placements(&fixture.plan, &named, &input_placements, &output_placements)
        .expect("seven merged plus one ordinary dispatch execute");
    fixture.assert_outputs_match_reference(1e-2);
    #[cfg(feature = "instrument")]
    {
        let expected_dispatches: u64 = if cfg!(feature = "metal-horizontal-merge") { 2 } else { 8 };
        assert_eq!(
            omega::metal::ENCODE_DISPATCH_CALLS.snapshot_and_reset(),
            expected_dispatches,
            "metal-horizontal-merge={}: the mismatched weight buffer must not join the merge group",
            cfg!(feature = "metal-horizontal-merge")
        );
    }
}

const RAW_SPLIT_SIZE: usize = QK_K;
const RAW_SPLIT_ROUNDS: usize = 4;

/// A genuine dataflow edge inside an otherwise-mergeable bucket: round 3's
/// own "activation" operand is round 2's OUTPUT node, not the shared root
/// activation every other round reads. `group_mergeable_positions`'s own
/// pure unit test (`a_raw_edge_between_two_members_excludes_the_reader_from_the_group`)
/// already proves the algorithm excludes the reader; this is that same
/// shape run end to end on a real device, through `ensure_merged_dispatches`,
/// so the claim is measured against a real dispatch count, not just the
/// pure grouping function. Expected: round 3 is excluded from the group of
/// 3 -- 2 real dispatches with `metal-horizontal-merge` on (the group of 3,
/// plus round 3 alone), 4 with it off.
#[test]
fn raw_split_member_forces_its_own_dispatch() {
    let mut program = Vec::new();
    // weight nodes before the activation node -- see `build_fixture`'s own
    // comment on why declaration order (not just operand-vec position)
    // decides which slot `build_merged_dispatch` reads as the weight.
    let weight_nodes: Vec<NodeId> = (0..RAW_SPLIT_ROUNDS)
        .map(|round| {
            append(
                &mut program,
                Op::Input {
                    dtype: DType::UInt8,
                    shape: vec![
                        Extent::Static(RAW_SPLIT_SIZE as u32),
                        Extent::Static(RAW_SPLIT_SIZE as u32),
                    ],
                    name: Some(format!("raw_weight_{round}")),
                },
            )
        })
        .collect();
    let activation_node = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(RAW_SPLIT_SIZE as u32)],
            name: Some("activation".into()),
        },
    );
    let mut output_nodes = Vec::with_capacity(RAW_SPLIT_ROUNDS);
    for (round, &weight_node) in weight_nodes.iter().enumerate() {
        let this_round_activation = if round == 3 { output_nodes[2] } else { activation_node };
        output_nodes.push(append_matvec(&mut program, weight_node, this_round_activation));
    }

    let activation = random_vec(11, RAW_SPLIT_SIZE);
    let mut placed_input_nodes = vec![activation_node];
    placed_input_nodes.extend(weight_nodes.iter().copied());
    let weight_names: Vec<String> = (0..RAW_SPLIT_ROUNDS).map(|round| format!("raw_weight_{round}")).collect();
    let mut named: Vec<(&str, proxima_tensor::QuantizedBlock<'_>)> =
        vec![("activation", proxima_tensor::QuantizedBlock::Float32(&[]))];
    named.extend(weight_names.iter().map(|name| (name.as_str(), proxima_tensor::QuantizedBlock::Q4K(&[]))));

    let plan = omega::plan_named_with_placed_inputs(
        &program,
        &[],
        &named,
        &output_nodes,
        NumericPolicy::default(),
        &placed_input_nodes,
    )
    .expect("plans the four raw-split matvecs");

    let weight_bytes_per_round = (RAW_SPLIT_SIZE / QK_K) * BLOCK_BYTES * RAW_SPLIT_SIZE;
    let weight_buffer = omega::allocate_placed_buffer(weight_bytes_per_round * RAW_SPLIT_ROUNDS)
        .expect("allocates the shared raw-split weight buffer");
    let activation_buffer =
        omega::allocate_placed_buffer(RAW_SPLIT_SIZE * size_of::<f32>()).expect("allocates the activation buffer");
    let output_buffer = omega::allocate_placed_buffer(RAW_SPLIT_SIZE * size_of::<f32>() * RAW_SPLIT_ROUNDS)
        .expect("allocates the shared output buffer");

    unsafe {
        write_placed_buffer(
            &activation_buffer,
            0,
            core::slice::from_raw_parts(activation.as_ptr().cast::<u8>(), core::mem::size_of_val(activation.as_slice())),
        );
    }

    let mut packed_weights = Vec::with_capacity(RAW_SPLIT_ROUNDS);
    for round in 0..RAW_SPLIT_ROUNDS {
        let rows: Vec<Vec<f32>> = (0..RAW_SPLIT_SIZE)
            .map(|row| random_vec(2_000_000 + round as u64 * 10_000 + row as u64, RAW_SPLIT_SIZE))
            .collect();
        let mut packed = vec![0u8; RAW_SPLIT_SIZE / QK_K * BLOCK_BYTES];
        for (row, row_packed) in rows.iter().zip(packed.as_chunks_mut::<BLOCK_BYTES>().0) {
            quantize(row, row_packed).expect("RAW_SPLIT_SIZE is a whole multiple of QK_K");
        }
        unsafe {
            write_placed_buffer(&weight_buffer, round * weight_bytes_per_round, &packed);
        }
        packed_weights.push(packed);
    }

    let dequant_dot = |packed: &[u8], input: &[f32]| -> Vec<f32> {
        let mut expected = Vec::with_capacity(RAW_SPLIT_SIZE);
        for row_packed in packed.as_chunks::<BLOCK_BYTES>().0 {
            let mut row = vec![0.0f32; RAW_SPLIT_SIZE];
            dequantize(row_packed, &mut row).expect("packed row dequantizes");
            expected.push(row.iter().zip(input.iter()).map(|(weight, value)| weight * value).sum());
        }
        expected
    };
    let expected0 = dequant_dot(&packed_weights[0], &activation);
    let expected1 = dequant_dot(&packed_weights[1], &activation);
    let expected2 = dequant_dot(&packed_weights[2], &activation);
    let expected3 = dequant_dot(&packed_weights[3], &expected2);
    let expected = [expected0, expected1, expected2, expected3];

    let output_placements: Vec<(NodeId, &omega::PlacedBuffer, usize)> = output_nodes
        .iter()
        .enumerate()
        .map(|(round, &node)| (node, &output_buffer, round * RAW_SPLIT_SIZE * size_of::<f32>()))
        .collect();
    let mut input_placements = vec![(activation_node, &activation_buffer, 0usize)];
    input_placements.extend(
        weight_nodes
            .iter()
            .enumerate()
            .map(|(round, &node)| (node, &weight_buffer, round * weight_bytes_per_round)),
    );

    #[cfg(feature = "instrument")]
    let _ = omega::metal::ENCODE_DISPATCH_CALLS.snapshot_and_reset();
    omega::execute_plan_named_with_placements(&plan, &named, &input_placements, &output_placements)
        .expect("the raw-split program executes");
    #[cfg(feature = "instrument")]
    {
        let expected_dispatches: u64 = if cfg!(feature = "metal-horizontal-merge") { 2 } else { 4 };
        assert_eq!(
            omega::metal::ENCODE_DISPATCH_CALLS.snapshot_and_reset(),
            expected_dispatches,
            "metal-horizontal-merge={}: round 3's read of round 2's output must exclude it from the merge group",
            cfg!(feature = "metal-horizontal-merge")
        );
    }

    for (round, round_expected) in expected.iter().enumerate() {
        let actual = omega::read_placed_buffer_f32(
            &output_buffer,
            round * RAW_SPLIT_SIZE * size_of::<f32>(),
            RAW_SPLIT_SIZE,
        );
        for (index, (&value, &reference)) in actual.iter().zip(round_expected.iter()).enumerate() {
            let scale = reference.abs().max(f32::MIN_POSITIVE);
            let relative = (value - reference).abs() / scale;
            assert!(
                relative < 1e-2,
                "round {round} row {index}: got={value} reference={reference} relative={relative}"
            );
        }
    }
}

/// ROW 570's ladder: wall-clock time around one call to
/// `execute_plan_named_with_placements` for the whole 8-round fixture, on
/// the SAME 0.59 MB (`weight_bytes_per_round`) Q4_K slab shape every other
/// case in this file uses. `execute_plan_with_placements` always
/// `waitUntilCompleted`s before returning (its own doc), so this wall-clock
/// window covers exactly one command buffer's worth of encode+dispatch+GPU
/// execution -- the CPU-side encode overhead is real but IDENTICAL in
/// shape between the two feature builds (8 ordinary `encode_op` calls vs 1
/// `handle_merged_position` call), so it does not favor either arm. Which
/// of "8 separate dispatches" or "1 depth-8 dispatch" this measures is
/// entirely decided by which feature set the binary was BUILT with
/// (`metal-horizontal-merge` on or off) -- run this test under both to get
/// both arms. Two untimed warmup calls pay the pipeline-cache-miss cost
/// once, outside the 7 recorded samples, matching how the crate itself
/// only ever pays that cost on the plan's first execution.
#[test]
#[ignore = "perf ladder -- run explicitly, once per metal-horizontal-merge feature setting"]
fn ladder_eight_dispatches_vs_one_merged_dispatch_gpu_time() {
    let fixture = build_fixture();
    let named = fixture.named();
    let input_placements = fixture.input_placements();
    let output_placements = fixture.output_placements();
    for _ in 0..2 {
        omega::execute_plan_named_with_placements(&fixture.plan, &named, &input_placements, &output_placements)
            .expect("warmup run executes");
    }
    let mut samples_us: Vec<f64> = Vec::with_capacity(7);
    for _ in 0..7 {
        let started = std::time::Instant::now();
        omega::execute_plan_named_with_placements(&fixture.plan, &named, &input_placements, &output_placements)
            .expect("timed run executes");
        samples_us.push(started.elapsed().as_secs_f64() * 1_000_000.0);
    }
    fixture.assert_outputs_match_reference(1e-2);
    let mut sorted = samples_us.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    eprintln!(
        "row570_ladder metal-horizontal-merge={} samples_us={samples_us:?} median_us={median}",
        cfg!(feature = "metal-horizontal-merge")
    );
}

const LAZY_SIZE: usize = QK_K;
const LAZY_ROUNDS: usize = 4;

/// ROW 572's own missing case: neither the shared activation NOR any
/// member's output is a caller placement here -- the activation is an
/// intermediate (`raw_activation + raw_activation`, computed by an earlier
/// position, never an `Op::Input` leaf) and every output is a plain plan
/// root, read back through `Evaluated::get` the way a real qwen35moe decode
/// step's own MoE projection is. This exercises BOTH halves of ROW 572's fix
/// at once: `split_by_shared_buffers` admitting on the activation's NODE
/// identity alone (its buffer does not exist until the intermediate's own
/// position executes, one position before the group's leader), and
/// `ensure_merged_group_resolved`'s fresh-allocation branch (no member is
/// caller-placed, so the group's shared output is allocated at the leader's
/// own first encode, not read from `output_placed`).
#[test]
fn lazy_activation_and_output_exercise_the_fresh_allocation_path() {
    let mut program = Vec::new();
    let weight_nodes: Vec<NodeId> = (0..LAZY_ROUNDS)
        .map(|round| {
            append(
                &mut program,
                Op::Input {
                    dtype: DType::UInt8,
                    shape: vec![Extent::Static(LAZY_SIZE as u32), Extent::Static(LAZY_SIZE as u32)],
                    name: Some(format!("lazy_weight_{round}")),
                },
            )
        })
        .collect();
    let raw_activation_node = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(LAZY_SIZE as u32)],
            name: Some("raw_activation".into()),
        },
    );
    // A genuine intermediate: multiple reduces below read it, so `bind`'s
    // own last-use fusion rule cannot absorb it into any one of them --
    // `append_matvec`'s own doc names this exact rule.
    let activation_node = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: vec![
                (raw_activation_node, IndexMap::Affine(projection(1, &[0]))),
                (raw_activation_node, IndexMap::Affine(projection(1, &[0]))),
            ],
            name: None,
        },
    );
    let output_nodes: Vec<NodeId> = weight_nodes
        .iter()
        .map(|&weight| append_matvec(&mut program, weight, activation_node))
        .collect();

    let raw_activation = random_vec(23, LAZY_SIZE);
    let activation: Vec<f32> = raw_activation.iter().map(|value| value * 2.0).collect();
    let mut placed_input_nodes = vec![raw_activation_node];
    placed_input_nodes.extend(weight_nodes.iter().copied());
    let weight_names: Vec<String> = (0..LAZY_ROUNDS).map(|round| format!("lazy_weight_{round}")).collect();
    let mut named: Vec<(&str, proxima_tensor::QuantizedBlock<'_>)> =
        vec![("raw_activation", proxima_tensor::QuantizedBlock::Float32(&[]))];
    named.extend(weight_names.iter().map(|name| (name.as_str(), proxima_tensor::QuantizedBlock::Q4K(&[]))));

    let plan = omega::plan_named_with_placed_inputs(
        &program,
        &[],
        &named,
        &output_nodes,
        NumericPolicy::default(),
        &placed_input_nodes,
    )
    .expect("plans the lazy-activation matvecs");

    let weight_bytes_per_round = (LAZY_SIZE / QK_K) * BLOCK_BYTES * LAZY_SIZE;
    let weight_buffer = omega::allocate_placed_buffer(weight_bytes_per_round * LAZY_ROUNDS)
        .expect("allocates the shared lazy weight buffer");
    let raw_activation_buffer =
        omega::allocate_placed_buffer(LAZY_SIZE * size_of::<f32>()).expect("allocates the raw activation buffer");

    unsafe {
        write_placed_buffer(
            &raw_activation_buffer,
            0,
            core::slice::from_raw_parts(
                raw_activation.as_ptr().cast::<u8>(),
                core::mem::size_of_val(raw_activation.as_slice()),
            ),
        );
    }

    let mut expected = Vec::with_capacity(LAZY_ROUNDS);
    for round in 0..LAZY_ROUNDS {
        let rows: Vec<Vec<f32>> = (0..LAZY_SIZE)
            .map(|row| random_vec(3_000_000 + round as u64 * 10_000 + row as u64, LAZY_SIZE))
            .collect();
        let mut packed = vec![0u8; LAZY_SIZE / QK_K * BLOCK_BYTES];
        for (row, row_packed) in rows.iter().zip(packed.as_chunks_mut::<BLOCK_BYTES>().0) {
            quantize(row, row_packed).expect("LAZY_SIZE is a whole multiple of QK_K");
        }
        unsafe {
            write_placed_buffer(&weight_buffer, round * weight_bytes_per_round, &packed);
        }
        let mut round_expected: Vec<f32> = Vec::with_capacity(LAZY_SIZE);
        for row_packed in packed.as_chunks::<BLOCK_BYTES>().0 {
            let mut row = vec![0.0f32; LAZY_SIZE];
            dequantize(row_packed, &mut row).expect("packed row dequantizes");
            round_expected.push(row.iter().zip(activation.iter()).map(|(weight, value)| weight * value).sum());
        }
        expected.push(round_expected);
    }

    let input_placements: Vec<(NodeId, &omega::PlacedBuffer, usize)> = {
        let mut placements = vec![(raw_activation_node, &raw_activation_buffer, 0usize)];
        placements.extend(
            weight_nodes
                .iter()
                .enumerate()
                .map(|(round, &node)| (node, &weight_buffer, round * weight_bytes_per_round)),
        );
        placements
    };
    // No `output_placements` at all -- every output is a plain plan root,
    // read back through `Evaluated::get` exactly like a real decode step's
    // own MoE projection, never a caller-placed buffer.
    #[cfg(feature = "instrument")]
    let _ = omega::metal::ENCODE_DISPATCH_CALLS.snapshot_and_reset();
    let evaluated = omega::execute_plan_named_with_placements(&plan, &named, &input_placements, &[])
        .expect("the lazy-activation program executes");
    #[cfg(feature = "instrument")]
    {
        // +1 for the intermediate activation's own dispatch
        // (`raw_activation + raw_activation`), materialized because
        // multiple reduces below read it.
        let expected_dispatches: u64 = if cfg!(feature = "metal-horizontal-merge") { 2 } else { 5 };
        assert_eq!(
            omega::metal::ENCODE_DISPATCH_CALLS.snapshot_and_reset(),
            expected_dispatches,
            "metal-horizontal-merge={}: an intermediate activation and unplaced outputs must still merge",
            cfg!(feature = "metal-horizontal-merge")
        );
    }

    for (round, &node) in output_nodes.iter().enumerate() {
        let (actual, _shape) = evaluated.get(node).expect("this round's output is a plan root");
        for (index, (&value, &reference)) in actual.iter().zip(expected[round].iter()).enumerate() {
            let scale = reference.abs().max(f32::MIN_POSITIVE);
            let relative = (value - reference).abs() / scale;
            assert!(
                relative < 1e-2,
                "round {round} row {index}: got={value} reference={reference} relative={relative}"
            );
        }
    }
}
