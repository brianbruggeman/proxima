//! Dimension coalescing against an independent scalar walk. The reference
//! (`reference_elementwise`, `reference_reduce`) evaluates a bound op one full
//! coordinate at a time through `Layout::offset_of` and `apply_body`, so it
//! shares no loop nest, odometer or width kernel with the executor. Every
//! layout in the generated ops is drawn from the cases a coalescing pass has
//! to get right: contiguous runs, broadcast (stride 0) runs, reversed
//! (negative stride) runs, padded strides that must NOT merge, and size-1
//! axes.

use super::*;
use crate::bind::{BodyStep, Layout, Lookup};
use crate::test_support::Lcg;
use proptest::prelude::*;
use smallvec::{SmallVec, smallvec};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AxisKind {
    Dense,
    Broadcast,
    Reversed,
    Padded,
}

fn pick_axis_kind(lcg: &mut Lcg) -> AxisKind {
    match (lcg.next_unit() * 20.0) as u32 {
        0..=9 => AxisKind::Dense,
        10..=14 => AxisKind::Broadcast,
        15..=16 => AxisKind::Reversed,
        _ => AxisKind::Padded,
    }
}

fn pick_extent(lcg: &mut Lcg) -> u64 {
    match (lcg.next_unit() * 10.0) as u64 {
        0..=1 => 1,
        value => value.min(5),
    }
}

/// A layout over `extents` built innermost-out, and the buffer length it
/// addresses. `Reversed` walks its axis backwards from the far end, so its
/// stride is negative and `base` moves to the last element of that axis.
fn build_layout(extents: &[u64], kinds: &[AxisKind]) -> (Layout, usize) {
    let mut strides: SmallVec<[i64; 8]> = smallvec![0; extents.len()];
    let mut base = 0i64;
    let mut running = 1i64;
    for axis in (0..extents.len()).rev() {
        let extent = extents[axis] as i64;
        match kinds[axis] {
            AxisKind::Broadcast => {}
            AxisKind::Dense => {
                strides[axis] = running;
                running *= extent;
            }
            AxisKind::Reversed => {
                strides[axis] = -running;
                base += (extent - 1) * running;
                running *= extent;
            }
            AxisKind::Padded => {
                running *= 2;
                strides[axis] = running;
                running *= extent;
            }
        }
    }
    (
        Layout {
            base,
            strides: strides.into_iter().collect(),
        },
        running as usize,
    )
}

fn operand_values(lcg: &mut Lcg, len: usize, integers: bool) -> Vec<f32> {
    (0..len)
        .map(|_| {
            let unit = lcg.next_unit();
            if integers {
                (unit * 7.0).floor() - 3.0
            } else {
                unit * 2.0 - 1.0
            }
        })
        .collect()
}

type OperandList = Vec<(NodeId, Layout, Option<Lookup>)>;
type BufferTable = Vec<Option<Vec<f32>>>;

struct Case {
    op: BoundOp,
    buffers: BufferTable,
    output_len: usize,
}

fn generic_body() -> ComposedBody {
    ComposedBody {
        steps: vec![
            BodyStep {
                op: ScalarOp::Multiply,
                args: vec![StepArg::Operand(0), StepArg::Operand(1)],
            },
            BodyStep {
                op: ScalarOp::Add,
                args: vec![StepArg::Step(0), StepArg::Operand(2)],
            },
        ],
    }
}

fn multiply_body() -> ComposedBody {
    ComposedBody::leaf(ScalarOp::Multiply)
}

fn operand_set(
    lcg: &mut Lcg,
    extents: &[u64],
    count: usize,
    integers: bool,
) -> (OperandList, BufferTable) {
    let mut operands = Vec::new();
    let mut buffers = Vec::new();
    for index in 0..count {
        let kinds: Vec<AxisKind> = extents.iter().map(|_| pick_axis_kind(lcg)).collect();
        let (layout, len) = build_layout(extents, &kinds);
        operands.push((NodeId(index as u32), layout, None));
        buffers.push(Some(operand_values(lcg, len, integers)));
    }
    (operands, buffers)
}

fn elementwise_case(seed: u64, integers: bool) -> Case {
    let mut lcg = Lcg(seed);
    let rank = 2 + (lcg.next_unit() * 3.0) as usize;
    let extents: Vec<u64> = (0..rank).map(|_| pick_extent(&mut lcg)).collect();
    let generic = lcg.next_unit() < 0.5;
    let (body, operand_count) = if generic {
        (generic_body(), 3)
    } else {
        (multiply_body(), 2)
    };
    let (operands, buffers) = operand_set(&mut lcg, &extents, operand_count, integers);
    let output_len = extents.iter().product::<u64>() as usize;
    Case {
        op: BoundOp {
            node: NodeId(operand_count as u32),
            dtype: DType::FLOAT32,
            extents,
            kind: BoundOpKind::Elementwise { body, operands },
        },
        buffers,
        output_len,
    }
}

/// Kept axes chosen at random, in ascending order unless the draw swaps
/// them; at least one axis is kept and at least one is folded away.
fn pick_output_axes(lcg: &mut Lcg, rank: usize) -> Vec<u16> {
    let mut kept: Vec<u16> = (0..rank as u16).filter(|_| lcg.next_unit() < 0.6).collect();
    if kept.is_empty() {
        kept.push(rank as u16 - 1);
    }
    if kept.len() == rank {
        kept.pop();
    }
    if lcg.next_unit() < 0.2 {
        kept.reverse();
    }
    kept
}

fn output_layout(extents: &[u64], output_axes: &[u16]) -> (Layout, usize) {
    let mut strides: SmallVec<[i64; 8]> = smallvec![0; extents.len()];
    let mut running = 1i64;
    for &axis in output_axes.iter().rev() {
        strides[axis as usize] = running;
        running *= extents[axis as usize] as i64;
    }
    (
        Layout {
            base: 0,
            strides: strides.into_iter().collect(),
        },
        running as usize,
    )
}

/// The folds a width-stepped reduce has to combine correctly: sum, max and
/// min from their identity seeds, and a sum seeded from the first element.
fn pick_fold(lcg: &mut Lcg) -> (ScalarOp, ReduceInit) {
    match (lcg.next_unit() * 4.0) as u32 {
        0 => (ScalarOp::Add, ReduceInit::Zero),
        1 => (ScalarOp::Maximum, ReduceInit::NegativeInfinity),
        2 => (ScalarOp::Minimum, ReduceInit::PositiveInfinity),
        _ => (ScalarOp::Add, ReduceInit::FirstElement),
    }
}

fn reduce_case(seed: u64, integers: bool) -> Case {
    reduce_case_with(seed, integers, None)
}

/// `generic` forces the fused multi-step body (`Some(true)`) or the bare
/// multiply (`Some(false)`); `None` lets the seed choose.
fn reduce_case_with(seed: u64, integers: bool, generic: Option<bool>) -> Case {
    let mut lcg = Lcg(seed);
    let rank = 2 + (lcg.next_unit() * 3.0) as usize;
    let extents: Vec<u64> = (0..rank).map(|_| pick_extent(&mut lcg)).collect();
    let output_axes = pick_output_axes(&mut lcg, rank);
    let (reduce_op, init) = pick_fold(&mut lcg);
    let drawn_generic = lcg.next_unit() < 0.5;
    let generic = generic.unwrap_or(drawn_generic);
    let (body, operand_count) = if generic {
        (generic_body(), 3)
    } else {
        (multiply_body(), 2)
    };
    let (operands, buffers) = operand_set(&mut lcg, &extents, operand_count, integers);
    let (out_layout, output_len) = output_layout(&extents, &output_axes);
    Case {
        op: BoundOp {
            node: NodeId(operand_count as u32),
            dtype: DType::FLOAT32,
            extents,
            kind: BoundOpKind::Reduce {
                element_body: body,
                reduce_op,
                init,
                keep: Keep::Reduce,
                operands,
                output_axes: output_axes.into_iter().collect(),
                out_layout,
                out_scatter: None,
                epilogue_body: ComposedBody::leaf(ScalarOp::Identity),
                epilogue_operands: Vec::new(),
                epilogue_broadcast_axes: smallvec![],
                accumulator: DType::FLOAT32,
            },
        },
        buffers,
        output_len,
    }
}

fn odometer_step(coordinate: &mut [u64], extents: &[u64]) -> bool {
    for axis in (0..extents.len()).rev() {
        coordinate[axis] += 1;
        if coordinate[axis] < extents[axis] {
            return true;
        }
        coordinate[axis] = 0;
    }
    false
}

fn body_at(case: &Case, body: &ComposedBody, coordinate: &[u64]) -> f32 {
    let values: Vec<f32> = case
        .op
        .operands()
        .iter()
        .map(|(node, layout, _)| {
            let buffer = case.buffers[node.0 as usize]
                .as_ref()
                .expect("every operand node has a buffer");
            buffer[layout.offset_of(coordinate) as usize]
        })
        .collect();
    let mut step_values = vec![0.0f32; body.steps.len()];
    apply_body(body, &values, &mut step_values)
}

fn reference_elementwise(case: &Case) -> Vec<f32> {
    let extents = &case.op.extents;
    let mut coordinate = vec![0u64; extents.len()];
    let mut output = Vec::with_capacity(case.output_len);
    loop {
        output.push(body_at(case, case.op.element_body(), &coordinate));
        if !odometer_step(&mut coordinate, extents) {
            return output;
        }
    }
}

fn reference_reduce(case: &Case) -> Vec<f32> {
    let BoundOpKind::Reduce {
        out_layout,
        reduce_op,
        init,
        ..
    } = &case.op.kind
    else {
        unreachable!("reference_reduce is only called on a reduce case")
    };
    let extents = &case.op.extents;
    let mut output: Vec<Option<f32>> = vec![initial_value(*init); case.output_len];
    let mut coordinate = vec![0u64; extents.len()];
    loop {
        let value = body_at(case, case.op.element_body(), &coordinate);
        let slot = out_layout.offset_of(&coordinate) as usize;
        output[slot] = Some(match output[slot] {
            Some(accumulator) => apply_scalar_op(*reduce_op, &[accumulator, value]),
            None => value,
        });
        if !odometer_step(&mut coordinate, extents) {
            break;
        }
    }
    output
        .into_iter()
        .map(|slot| slot.expect("every output position receives at least one reduction step"))
        .collect()
}

fn run_executor(case: &Case) -> Vec<f32> {
    let mut output = vec![0.0f32; case.output_len];
    match &case.op.kind {
        BoundOpKind::Elementwise { .. } => {
            run_elementwise_dispatch(&case.op, &case.buffers, None, &mut output)
                .expect("generated elementwise op runs");
        }
        _ => {
            run_reduce(&case.op, &case.buffers, &mut output, None, UNPLANNED_KERNEL_POLICY)
                .expect("generated reduce op runs");
        }
    }
    output
}

fn run_executor_uncoalesced(case: &Case) -> Vec<f32> {
    let mut output = vec![0.0f32; case.output_len];
    match &case.op.kind {
        BoundOpKind::Elementwise { .. } => {
            run_elementwise_dispatch_planned(&case.op, &case.buffers, None, &mut output)
                .expect("generated elementwise op runs uncoalesced");
        }
        _ => {
            run_reduce_planned(&case.op, &case.buffers, &mut output, None, UNPLANNED_KERNEL_POLICY)
                .expect("generated reduce op runs uncoalesced");
        }
    }
    output
}

fn assert_same(actual: &[f32], expected: &[f32], integers: bool, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: output length");
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        let matches = if integers {
            got.to_bits() == want.to_bits()
        } else {
            (got - want).abs() <= 1e-5 * (1.0 + want.abs())
        };
        assert!(matches, "{label}: position {index} got {got} want {want}");
    }
}

proptest! {
    #[test]
    fn coalesced_elementwise_matches_the_scalar_reference_bit_exact_on_integers(seed in any::<u64>()) {
        let case = elementwise_case(seed, true);
        let expected = reference_elementwise(&case);
        assert_same(&run_executor(&case), &expected, true, "coalesced integer elementwise");
    }

    #[test]
    fn coalesced_elementwise_matches_the_scalar_reference_within_tolerance_on_floats(seed in any::<u64>()) {
        let case = elementwise_case(seed, false);
        let expected = reference_elementwise(&case);
        assert_same(&run_executor(&case), &expected, false, "coalesced float elementwise");
    }

    #[test]
    fn coalesced_reduce_matches_the_scalar_reference_bit_exact_on_integers(seed in any::<u64>()) {
        let case = reduce_case(seed, true);
        let expected = reference_reduce(&case);
        assert_same(&run_executor(&case), &expected, true, "coalesced integer reduce");
    }

    #[test]
    fn coalesced_reduce_matches_the_scalar_reference_within_tolerance_on_floats(seed in any::<u64>()) {
        let case = reduce_case(seed, false);
        let expected = reference_reduce(&case);
        assert_same(&run_executor(&case), &expected, false, "coalesced float reduce");
    }

    #[test]
    fn uncoalesced_elementwise_still_matches_the_scalar_reference(seed in any::<u64>()) {
        let case = elementwise_case(seed, true);
        let expected = reference_elementwise(&case);
        let actual = run_executor_uncoalesced(&case);
        assert_same(&actual, &expected, true, "uncoalesced integer elementwise");
    }

    #[test]
    fn uncoalesced_reduce_still_matches_the_scalar_reference(seed in any::<u64>()) {
        let case = reduce_case(seed, true);
        let expected = reference_reduce(&case);
        let actual = run_executor_uncoalesced(&case);
        assert_same(&actual, &expected, true, "uncoalesced integer reduce");
    }

    #[test]
    fn coalesced_op_covers_the_same_iteration_space_as_the_original(seed in any::<u64>()) {
        let case = reduce_case(seed, true);
        if let Some(merged) = coalesce_dims(&case.op) {
            prop_assert_eq!(
                merged.extents.iter().product::<u64>(),
                case.op.extents.iter().product::<u64>()
            );
            prop_assert!(merged.extents.len() < case.op.extents.len());
        }
    }
}

fn layout(base: i64, strides: &[i64]) -> Layout {
    Layout {
        base,
        strides: strides.iter().copied().collect(),
    }
}

fn plain_elementwise(extents: Vec<u64>, strides: &[&[i64]]) -> BoundOp {
    let operands = strides
        .iter()
        .enumerate()
        .map(|(index, axis_strides)| (NodeId(index as u32), layout(0, axis_strides), None))
        .collect();
    BoundOp {
        node: NodeId(strides.len() as u32),
        dtype: DType::FLOAT32,
        extents,
        kind: BoundOpKind::Elementwise {
            body: multiply_body(),
            operands,
        },
    }
}

fn operand_strides(op: &BoundOp) -> Vec<Vec<i64>> {
    op.operands()
        .iter()
        .map(|(_, layout, _)| layout.strides.to_vec())
        .collect()
}

#[test]
fn a_fully_contiguous_elementwise_op_collapses_to_one_axis() {
    let op = plain_elementwise(vec![4, 3, 2], &[&[6, 2, 1], &[6, 2, 1]]);

    let merged = coalesce_dims(&op).expect("a dense [4,3,2] op merges");

    assert_eq!(merged.extents, vec![24]);
    assert_eq!(operand_strides(&merged), vec![vec![1], vec![1]]);
}

#[test]
fn a_broadcast_pair_merges_alongside_a_dense_pair() {
    let op = plain_elementwise(vec![4, 8], &[&[8, 1], &[0, 0]]);

    let merged = coalesce_dims(&op).expect("dense and all-broadcast operands share a merge");

    assert_eq!(merged.extents, vec![32]);
    assert_eq!(operand_strides(&merged), vec![vec![1], vec![0]]);
}

#[test]
fn a_per_row_broadcast_blocks_the_merge() {
    let op = plain_elementwise(vec![4, 8], &[&[8, 1], &[1, 0]]);

    assert!(coalesce_dims(&op).is_none());
}

#[test]
fn a_transposed_layout_blocks_the_merge() {
    let op = plain_elementwise(vec![3, 4], &[&[1, 3], &[4, 1]]);

    assert!(coalesce_dims(&op).is_none());
}

#[test]
fn a_padded_row_stride_blocks_the_merge() {
    let op = plain_elementwise(vec![3, 4], &[&[8, 1], &[4, 1]]);

    assert!(coalesce_dims(&op).is_none());
}

#[test]
fn a_reversed_contiguous_block_merges_with_a_negative_stride() {
    let mut op = plain_elementwise(vec![3, 4], &[&[-4, -1]]);
    if let BoundOpKind::Elementwise { operands, .. } = &mut op.kind {
        operands[0].1.base = 11;
    }

    let merged = coalesce_dims(&op).expect("a fully reversed dense block is one reversed run");

    assert_eq!(merged.extents, vec![12]);
    assert_eq!(operand_strides(&merged), vec![vec![-1]]);
    assert_eq!(merged.operands()[0].1.base, 11);
}

#[test]
fn size_one_axes_merge_with_any_neighbour_whatever_their_stride() {
    let op = plain_elementwise(vec![1, 5, 1], &[&[99, 1, 77]]);

    let merged = coalesce_dims(&op).expect("size-1 axes contribute no addressing");

    assert_eq!(merged.extents, vec![5]);
    assert_eq!(operand_strides(&merged), vec![vec![1]]);
}

#[test]
fn a_gathered_operand_is_never_coalesced() {
    let mut op = plain_elementwise(vec![4, 8], &[&[8, 1], &[8, 1]]);
    if let BoundOpKind::Elementwise { operands, .. } = &mut op.kind {
        operands[1].2 = Some(Lookup {
            indices: NodeId(9),
            index_layout: layout(0, &[1, 0]),
            element_stride: 8,
            extent: 4,
        });
    }

    assert!(coalesce_dims(&op).is_none());
}

/// `[m, n, k]` matmul fold (`out[m, n] += a[m, k] * b[k, n]`), the iteration
/// order `neon_tile_full_output`'s `matmul_program` binds.
fn mnk_matmul_fold(m: u64, n: u64, k: u64) -> BoundOp {
    let (k_i, n_i) = (k as i64, n as i64);
    BoundOp {
        node: NodeId(2),
        dtype: DType::FLOAT32,
        extents: vec![m, n, k],
        kind: BoundOpKind::Reduce {
            element_body: multiply_body(),
            reduce_op: ScalarOp::Add,
            init: ReduceInit::Zero,
            keep: Keep::Reduce,
            operands: vec![
                (NodeId(0), layout(0, &[k_i, 0, 1]), None),
                (NodeId(1), layout(0, &[0, 1, n_i]), None),
            ],
            output_axes: smallvec![0, 1],
            out_layout: layout(0, &[n_i, 1, 0]),
            out_scatter: None,
            epilogue_body: ComposedBody::leaf(ScalarOp::Identity),
            epilogue_operands: Vec::new(),
            epilogue_broadcast_axes: smallvec![],
            accumulator: DType::FLOAT32,
        },
    }
}

#[test]
fn a_unit_row_axis_of_a_matmul_fold_is_not_absorbed_into_the_width_axis() {
    let op = mnk_matmul_fold(1, 384, 64);

    assert!(
        coalesce_dims(&op).is_none(),
        "absorbing m=1 into n leaves a fold with no leading axis, which both GEMM tiles decline"
    );
}

#[test]
fn a_unit_row_axis_before_a_leading_batch_axis_keeps_its_own_group() {
    let mut op = mnk_matmul_fold(1, 384, 64);
    op.extents = vec![1, 3, 384, 64];
    if let BoundOpKind::Reduce {
        operands,
        output_axes,
        out_layout,
        ..
    } = &mut op.kind
    {
        operands[0].1 = layout(0, &[0, 64, 0, 1]);
        operands[1].1 = layout(0, &[0, 0, 1, 384]);
        *output_axes = smallvec![0, 1, 2];
        *out_layout = layout(0, &[0, 384, 1, 0]);
    }

    assert!(coalesce_dims(&op).is_none());
}

#[test]
fn a_trailing_unit_kept_axis_still_merges_into_the_row() {
    let mut op = mnk_matmul_fold(3, 1, 4);
    if let BoundOpKind::Reduce {
        operands,
        out_layout,
        ..
    } = &mut op.kind
    {
        operands[0].1 = layout(0, &[4, 0, 1]);
        operands[1].1 = layout(0, &[0, 1, 1]);
        *out_layout = layout(0, &[1, 1, 0]);
    }

    let merged =
        coalesce_dims(&op).expect("a kept extent-1 axis behind a wider one widens the row");

    assert_eq!(merged.extents, vec![3, 4]);
}

#[test]
fn a_unit_row_matmul_fold_runs_identically_coalesced_and_not() {
    let mut lcg = Lcg(0x6d31_0001);
    let case = Case {
        op: mnk_matmul_fold(1, 24, 8),
        buffers: vec![
            Some(operand_values(&mut lcg, 8, true)),
            Some(operand_values(&mut lcg, 8 * 24, true)),
        ],
        output_len: 24,
    };
    let expected = reference_reduce(&case);

    assert_same(
        &run_executor(&case),
        &expected,
        true,
        "coalesced unit-row fold",
    );
    assert_same(
        &run_executor_uncoalesced(&case),
        &expected,
        true,
        "uncoalesced unit-row fold",
    );
}

fn delta_net_state_fold() -> BoundOp {
    let state = layout(0, &[64, 16, 2, 1]);
    let key = layout(0, &[32, 0, 2, 1]);
    let gate = layout(0, &[0, 0, 2, 1]);
    BoundOp {
        node: NodeId(3),
        dtype: DType::FLOAT32,
        extents: vec![4, 4, 8, 2],
        kind: BoundOpKind::Reduce {
            element_body: generic_body(),
            reduce_op: ScalarOp::Add,
            init: ReduceInit::Zero,
            keep: Keep::Reduce,
            operands: vec![
                (NodeId(0), state, None),
                (NodeId(1), key, None),
                (NodeId(2), gate, None),
            ],
            output_axes: smallvec![1, 2, 3],
            out_layout: layout(0, &[0, 16, 2, 1]),
            out_scatter: None,
            epilogue_body: ComposedBody::leaf(ScalarOp::Identity),
            epilogue_operands: Vec::new(),
            epilogue_broadcast_axes: smallvec![],
            accumulator: DType::FLOAT32,
        },
    }
}

#[test]
fn the_delta_net_state_fold_widens_its_group_two_row_to_kv_heads_times_group() {
    let merged = coalesce_dims(&delta_net_state_fold()).expect("kv_heads and group are contiguous");

    let BoundOpKind::Reduce {
        output_axes,
        out_layout,
        ..
    } = &merged.kind
    else {
        unreachable!("a reduce stays a reduce")
    };
    assert_eq!(merged.extents, vec![4, 4, 16]);
    assert_eq!(output_axes.as_slice(), &[1, 2]);
    assert_eq!(out_layout.strides.as_slice(), &[0, 16, 1]);
    assert_eq!(
        operand_strides(&merged),
        vec![vec![64, 16, 1], vec![32, 0, 1], vec![0, 0, 1]]
    );
}

#[test]
fn kept_axes_written_in_swapped_order_do_not_merge() {
    let mut op = delta_net_state_fold();
    if let BoundOpKind::Reduce {
        output_axes,
        out_layout,
        ..
    } = &mut op.kind
    {
        *output_axes = smallvec![1, 3, 2];
        *out_layout = layout(0, &[0, 16, 1, 8]);
    }

    assert!(
        coalesce_dims(&op).is_none(),
        "the (2,3) pair is written as 3 then 2, so nothing in this fold merges"
    );
}

/// One output buffer, named through the table type the cases already use.
type Column = <<BufferTable as IntoIterator>::Item as IntoIterator>::Item;

fn bias_add_body() -> ComposedBody {
    ComposedBody {
        steps: vec![BodyStep {
            op: ScalarOp::Add,
            args: vec![StepArg::Operand(0), StepArg::Operand(1)],
        }],
    }
}

/// Fuses a bias-add epilogue onto a fold. `bias` is addressed by position in
/// the fold's `output_axes`, the space `apply_plain_reduce_epilogue` walks.
fn with_bias_epilogue(op: &mut BoundOp, bias: Layout) {
    let bias_node = NodeId(op.node.0 + 1);
    if let BoundOpKind::Reduce {
        epilogue_body,
        epilogue_operands,
        ..
    } = &mut op.kind
    {
        *epilogue_body = bias_add_body();
        epilogue_operands.clear();
        epilogue_operands.push((bias_node, bias, None));
    }
}

fn epilogue_strides(op: &BoundOp) -> Vec<Vec<i64>> {
    let BoundOpKind::Reduce {
        epilogue_operands, ..
    } = &op.kind
    else {
        unreachable!("only a reduce carries an epilogue")
    };
    epilogue_operands
        .iter()
        .map(|(_, layout, _)| layout.strides.to_vec())
        .collect()
}

/// Fold then fused epilogue on `op` exactly as given: the executor's own
/// `run_reduce_planned` followed by `apply_reduce_epilogue`.
fn run_fold_with_epilogue(op: &BoundOp, buffers: &BufferTable, output_len: usize) -> Column {
    let mut output: Column = vec![Default::default(); output_len];
    run_reduce_planned(op, buffers, &mut output, None, UNPLANNED_KERNEL_POLICY).expect("fold runs");
    apply_reduce_epilogue(op, buffers, &mut output).expect("epilogue runs");
    output
}

/// The same op run uncoalesced and through `coalesce_dims`; the last field
/// is whether the pass merged anything.
fn run_both_ways(case: &Case) -> (Column, Column, bool) {
    let uncoalesced = run_fold_with_epilogue(&case.op, &case.buffers, case.output_len);
    match coalesce_dims(&case.op) {
        Some(merged) => {
            let coalesced = run_fold_with_epilogue(&merged, &case.buffers, case.output_len);
            (uncoalesced, coalesced, true)
        }
        None => (uncoalesced.clone(), uncoalesced, false),
    }
}

fn reduce_case_with_bias(seed: u64, integers: bool) -> Case {
    let mut case = reduce_case_with(seed, integers, Some(true));
    let mut lcg = Lcg(seed ^ 0xb1a5_0001);
    let BoundOpKind::Reduce { output_axes, .. } = &case.op.kind else {
        unreachable!("reduce_case_with builds a reduce")
    };
    let local_extents: Vec<u64> = output_axes
        .iter()
        .map(|&axis| case.op.extents[axis as usize])
        .collect();
    let kinds: Vec<AxisKind> = local_extents
        .iter()
        .map(|_| pick_axis_kind(&mut lcg))
        .collect();
    let (bias, bias_len) = build_layout(&local_extents, &kinds);
    with_bias_epilogue(&mut case.op, bias);
    case.buffers.push(None);
    case.buffers
        .push(Some(operand_values(&mut lcg, bias_len, integers)));
    case
}

#[test]
fn a_bias_epilogue_that_is_one_run_over_a_kept_pair_lets_it_merge() {
    let mut op = delta_net_state_fold();
    with_bias_epilogue(&mut op, layout(0, &[16, 2, 1]));

    let merged = coalesce_dims(&op).expect("kv_heads and group are contiguous in the bias too");

    assert_eq!(merged.extents, vec![4, 4, 16]);
    assert_eq!(epilogue_strides(&merged), vec![vec![16, 1]]);
}

#[test]
fn a_bias_epilogue_with_a_padded_kept_axis_blocks_that_merge() {
    let mut op = delta_net_state_fold();
    with_bias_epilogue(&mut op, layout(0, &[16, 3, 1]));

    assert!(
        coalesce_dims(&op).is_none(),
        "the bias steps 3 over a group of 2, so (kv_heads, group) is not one run there"
    );
}

#[test]
fn a_bias_epilogue_that_is_broadcast_over_a_kept_pair_still_merges() {
    let mut op = delta_net_state_fold();
    with_bias_epilogue(&mut op, layout(0, &[16, 0, 0]));

    let merged = coalesce_dims(&op).expect("a stride-0 pair is one affine run");

    assert_eq!(merged.extents, vec![4, 4, 16]);
    assert_eq!(epilogue_strides(&merged), vec![vec![16, 0]]);
}

#[test]
fn a_gathered_epilogue_operand_is_never_coalesced() {
    let mut op = delta_net_state_fold();
    with_bias_epilogue(&mut op, layout(0, &[16, 2, 1]));
    if let BoundOpKind::Reduce {
        epilogue_operands, ..
    } = &mut op.kind
    {
        epilogue_operands[0].2 = Some(Lookup {
            indices: NodeId(9),
            index_layout: layout(0, &[0, 0, 0]),
            element_stride: 1,
            extent: 4,
        });
    }

    assert!(coalesce_dims(&op).is_none());
}

#[test]
fn the_delta_net_fold_with_a_bias_runs_identically_coalesced_and_not() {
    let mut lcg = Lcg(0xde17_b1a5);
    let mut op = delta_net_state_fold();
    with_bias_epilogue(&mut op, layout(0, &[16, 2, 1]));
    let case = Case {
        op,
        buffers: vec![
            Some(operand_values(&mut lcg, 256, true)),
            Some(operand_values(&mut lcg, 128, true)),
            Some(operand_values(&mut lcg, 16, true)),
            None,
            Some(operand_values(&mut lcg, 4 * 8 * 2, true)),
        ],
        output_len: 4 * 8 * 2,
    };

    let (uncoalesced, coalesced, merged) = run_both_ways(&case);

    assert!(merged, "the bias is one run over (kv_heads, group)");
    assert_same(&coalesced, &uncoalesced, true, "bias fold coalesced");
}

proptest! {
    #[test]
    fn a_coalesced_fold_with_a_bias_epilogue_is_bit_exact_on_integers(seed in any::<u64>()) {
        let case = reduce_case_with_bias(seed, true);
        let (uncoalesced, coalesced, _) = run_both_ways(&case);
        assert_same(&coalesced, &uncoalesced, true, "coalesced bias fold, integers");
    }

    #[test]
    fn a_coalesced_fold_with_a_bias_epilogue_matches_within_tolerance_on_floats(seed in any::<u64>()) {
        let case = reduce_case_with_bias(seed, false);
        let (uncoalesced, coalesced, _) = run_both_ways(&case);
        assert_same(&coalesced, &uncoalesced, false, "coalesced bias fold, floats");
    }
}

/// The FLUX VAE conv as the bound op carries it: output `(r, g, c)` =
/// `(rows, 32, 16)`, contraction `(t, i, j)` = `(9, 32, 16)`. The staged
/// activation is read at `9*r + t` rows of 512, so `r` and `t` carry strides
/// 4608 and 512 into one buffer; the weight is `[32, 16, 9, 32, 16]`; the
/// bias `[32, 16]` is fused and addressed over `(r, g, c)`.
fn vae_conv_fold(rows: u64) -> BoundOp {
    let mut op = delta_net_state_fold();
    op.extents = vec![rows, 32, 16, 9, 32, 16];
    if let BoundOpKind::Reduce {
        element_body,
        operands,
        output_axes,
        out_layout,
        ..
    } = &mut op.kind
    {
        *element_body = multiply_body();
        *operands = vec![
            (NodeId(0), layout(0, &[4608, 0, 0, 512, 16, 1]), None),
            (NodeId(1), layout(0, &[0, 73728, 4608, 512, 16, 1]), None),
        ];
        *output_axes = smallvec![0, 1, 2];
        *out_layout = layout(0, &[512, 16, 1, 0, 0, 0]);
    }
    op.node = NodeId(2);
    with_bias_epilogue(&mut op, layout(0, &[0, 16, 1]));
    op
}

#[test]
fn the_vae_conv_fold_coalesces_to_rank_three_and_runs_bit_identically() {
    let rows = 4u64;
    let mut lcg = Lcg(0x0aec_0001);
    let case = Case {
        op: vae_conv_fold(rows),
        buffers: vec![
            Some(operand_values(&mut lcg, (rows * 4608) as usize, true)),
            Some(operand_values(&mut lcg, 512 * 4608, true)),
            None,
            Some(operand_values(&mut lcg, 512, true)),
        ],
        output_len: (rows * 512) as usize,
    };

    let merged = coalesce_dims(&case.op).expect("every axis group is contiguous");
    let (uncoalesced, coalesced, _) = run_both_ways(&case);

    assert_eq!(merged.extents, vec![rows, 512, 4608]);
    assert_eq!(epilogue_strides(&merged), vec![vec![0, 1]]);
    assert_eq!(
        operand_strides(&merged),
        vec![vec![4608, 0, 1], vec![0, 4608, 1]]
    );
    assert_same(&coalesced, &uncoalesced, true, "vae conv fold coalesced");
}

#[test]
fn the_delta_net_state_fold_runs_identically_coalesced_and_not() {
    let mut lcg = Lcg(0xde17_a001);
    let case = Case {
        op: delta_net_state_fold(),
        buffers: vec![
            Some(operand_values(&mut lcg, 256, true)),
            Some(operand_values(&mut lcg, 128, true)),
            Some(operand_values(&mut lcg, 16, true)),
        ],
        output_len: 4 * 8 * 2,
    };
    let expected = reference_reduce(&case);

    let coalesced = run_executor(&case);
    let uncoalesced = run_executor_uncoalesced(&case);

    assert_same(&coalesced, &expected, true, "coalesced state fold");
    assert_same(&uncoalesced, &expected, true, "uncoalesced state fold");
}

proptest! {
    #[test]
    fn generic_body_reduce_matches_the_scalar_reference_bit_exact_on_integers(seed in any::<u64>()) {
        let case = reduce_case_with(seed, true, Some(true));
        let expected = reference_reduce(&case);
        let uncoalesced = run_executor_uncoalesced(&case);
        assert_same(&uncoalesced, &expected, true, "generic reduce without coalescing");
        assert_same(&run_executor(&case), &expected, true, "generic reduce with coalescing");
    }

    #[test]
    fn generic_body_reduce_matches_the_scalar_reference_within_tolerance_on_floats(seed in any::<u64>()) {
        let case = reduce_case_with(seed, false, Some(true));
        let expected = reference_reduce(&case);
        let uncoalesced = run_executor_uncoalesced(&case);
        assert_same(&uncoalesced, &expected, false, "generic reduce without coalescing");
        assert_same(&run_executor(&case), &expected, false, "generic reduce with coalescing");
    }
}

fn contiguous_generic_reduce(reduce_op: ScalarOp, init: ReduceInit) -> Case {
    let mut lcg = Lcg(0x6e17_0003);
    let operands = vec![
        (NodeId(0), layout(0, &[32, 1]), None),
        (NodeId(1), layout(0, &[0, 1]), None),
        (NodeId(2), layout(0, &[32, 1]), None),
    ];
    let op = BoundOp {
        node: NodeId(3),
        dtype: DType::FLOAT32,
        extents: vec![6, 32],
        kind: BoundOpKind::Reduce {
            element_body: generic_body(),
            reduce_op,
            init,
            keep: Keep::Reduce,
            operands,
            output_axes: smallvec![1],
            out_layout: layout(0, &[0, 1]),
            out_scatter: None,
            epilogue_body: ComposedBody::leaf(ScalarOp::Identity),
            epilogue_operands: Vec::new(),
            epilogue_broadcast_axes: smallvec![],
            accumulator: DType::FLOAT32,
        },
    };
    Case {
        op,
        buffers: vec![
            Some(operand_values(&mut lcg, 192, true)),
            Some(operand_values(&mut lcg, 32, true)),
            Some(operand_values(&mut lcg, 192, true)),
        ],
        output_len: 32,
    }
}

#[test]
fn a_wide_contiguous_generic_reduce_matches_the_reference_for_every_fold() {
    for (reduce_op, init) in [
        (ScalarOp::Add, ReduceInit::Zero),
        (ScalarOp::Add, ReduceInit::FirstElement),
        (ScalarOp::Maximum, ReduceInit::NegativeInfinity),
        (ScalarOp::Minimum, ReduceInit::PositiveInfinity),
    ] {
        let case = contiguous_generic_reduce(reduce_op, init);
        let expected = reference_reduce(&case);

        assert_same(
            &run_executor(&case),
            &expected,
            true,
            &format!("{reduce_op:?} from {init:?}"),
        );
    }
}

#[cfg(feature = "instrument")]
#[test]
fn a_wide_contiguous_generic_reduce_takes_the_width_path_not_the_per_element_loop() {
    let case = contiguous_generic_reduce(ScalarOp::Add, ReduceInit::Zero);
    let width_fast_before = crate::instrument::PATH_WIDTH_FAST.get();
    let generic_before = crate::instrument::PATH_GENERIC.get();

    let _ = run_executor(&case);

    assert_eq!(
        crate::instrument::PATH_WIDTH_FAST.get() - width_fast_before,
        1,
        "the fused body is stepped a whole row at a time"
    );
    assert_eq!(
        crate::instrument::PATH_GENERIC.get() - generic_before,
        0,
        "no per-element fallback runs for an affine generic body"
    );
}
