use super::*;
use crate::{BodyStep, Layout};

/// What [`step`] did with one bound op: nothing (the f32 path runs it), a
/// FLOAT64 node now held in the wide buffers, an f32 narrowing of a wide
/// operand ready for the f32 buffers, or a FLOAT64 constant held wide that
/// also leaves its exact f32 copy for the f32 buffers.
pub(super) enum IslandStep {
    NotIsland,
    Wide,
    Narrowed(Vec<f32>),
    WideWithCopy(Vec<f32>),
}

impl IslandStep {
    /// How many nodes this step made live, for an evaluator's peak count: a
    /// constant's wide buffer and its f32 copy are one node.
    pub(super) fn live_buffers(&self) -> usize {
        match self {
            Self::NotIsland => 0,
            Self::Wide | Self::Narrowed(_) | Self::WideWithCopy(_) => 1,
        }
    }

    /// The f32 buffer this step leaves for the node, if any.
    pub(super) fn into_f32(self) -> Option<Vec<f32>> {
        match self {
            Self::Narrowed(values) | Self::WideWithCopy(values) => Some(values),
            Self::NotIsland | Self::Wide => None,
        }
    }
}

/// Runs a FLOAT64 bound op in f64, or narrows an f32 identity reading one into
/// a buffer from `narrow_pool`, leaving every other op to the f32 kernels.
/// `wide` grows to the program length on the first FLOAT64 node, so an
/// f32-only program never allocates it; a FLOAT64 node's buffer comes from
/// `wide_pool`, where [`retire`] returns it, so a steady-state call that keeps
/// its pools allocates nothing here.
pub(super) fn step<B: Deref<Target = [f32]>>(
    computed: &BoundOp,
    buffers: &[Option<B>],
    wide: &mut Vec<Option<Cow<'static, [f64]>>>,
    wide_pool: &mut Vec<Vec<f64>>,
    narrow_pool: &mut Vec<Vec<f32>>,
    program_len: usize,
) -> Result<IslandStep, TensorError> {
    if computed.dtype == DType::FLOAT64 {
        if wide.is_empty() {
            wide.resize(program_len, None);
        }
        let mut output = take_or_allocate(wide_pool, node_output_len(computed));
        output.fill(0.0);
        run_wide(computed, buffers, wide, &mut output)?;
        // an f64 constant's literal is an f32, so its f32 copy is exact: a
        // fused f32 kernel reading the count a norm divides by gets it there
        let constant_copy = matches!(computed.kind, BoundOpKind::Constant { .. }).then(|| {
            let mut narrowed = take_or_allocate(narrow_pool, output.len());
            for (slot, &value) in narrowed.iter_mut().zip(&output) {
                *slot = value as f32;
            }
            narrowed
        });
        wide[computed.node.0 as usize] = Some(Cow::Owned(output));
        return Ok(constant_copy.map_or(IslandStep::Wide, IslandStep::WideWithCopy));
    }
    if wide.is_empty() || computed.dtype != DType::FLOAT32 {
        return Ok(IslandStep::NotIsland);
    }
    match single_wide_operand(computed, wide) {
        Some(source) => {
            let mut narrowed = take_or_allocate(narrow_pool, source.len());
            for (slot, &value) in narrowed.iter_mut().zip(source) {
                *slot = value as f32;
            }
            Ok(IslandStep::Narrowed(narrowed))
        }
        None => Ok(IslandStep::NotIsland),
    }
}

fn single_wide_operand<'wide>(
    computed: &BoundOp,
    wide: &'wide [Option<Cow<'static, [f64]>>],
) -> Option<&'wide [f64]> {
    let BoundOpKind::Elementwise { operands, .. } = &computed.kind else {
        return None;
    };
    let [(source, _, None)] = operands.as_slice() else {
        return None;
    };
    let data = wide.get(source.0 as usize)?.as_deref()?;
    (data.len() == node_output_len(computed)).then_some(data)
}

fn run_wide<B: Deref<Target = [f32]>>(
    computed: &BoundOp,
    buffers: &[Option<B>],
    wide: &[Option<Cow<'static, [f64]>>],
    output: &mut [f64],
) -> Result<(), TensorError> {
    let not_lowerable = |reason| TensorError::NotLowerable {
        node: computed.node,
        reason,
    };
    match &computed.kind {
        BoundOpKind::Constant { value } => {
            output.fill(f64::from(*value));
            Ok(())
        }
        BoundOpKind::Elementwise { operands, .. } => match operands.as_slice() {
            [(source, _, None)] if wide.get(source.0 as usize).is_some_and(Option::is_none) => {
                let narrow = buffers
                    .get(source.0 as usize)
                    .and_then(Option::as_deref)
                    .ok_or(not_lowerable(
                        "a widening convert's f32 operand has no buffer",
                    ))?;
                if narrow.len() != output.len() {
                    return Err(not_lowerable(
                        "a widening convert must read its operand one to one",
                    ));
                }
                for (slot, &value) in output.iter_mut().zip(narrow) {
                    *slot = f64::from(value);
                }
                Ok(())
            }
            _ => run_elementwise_typed::<f64>(computed, wide, &[], output),
        },
        BoundOpKind::Reduce {
            accumulator,
            element_body,
            epilogue_body,
            epilogue_operands,
            operands,
            ..
        } if reduce_epilogue_is_identity(epilogue_body, epilogue_operands) => {
            let reads_f32 = operands
                .iter()
                .any(|(node, _, _)| wide.get(node.0 as usize).is_none_or(Option::is_none));
            if *accumulator == DType::FLOAT64 && reads_f32 {
                accumulate_f32_elements_in_f64(computed, element_body, buffers, output)
            } else {
                run_reduce_typed::<f64>(computed, wide, &[], output)
            }
        }
        BoundOpKind::Reduce {
            accumulator: DType::FLOAT64,
            ..
        } => run_widened_fold(computed, buffers, |target, widened, _| {
            output[target] = widened;
        }),
        _ => Err(not_lowerable(
            "only elementwise, reduce and constant ops run in f64",
        )),
    }
}

/// Returns the f64 buffer of every node retiring at this position to
/// `wide_pool`, the f64 counterpart of each loop's own f32 retirement; the
/// count is how many live nodes it ended, leaving a node that also holds an
/// f32 copy to the f32 retirement.
pub(super) fn retire(
    wide: &mut [Option<Cow<'static, [f64]>>],
    wide_pool: &mut Vec<Vec<f64>>,
    retiring: &[NodeId],
    counted_as_f32: impl Fn(NodeId) -> bool,
) -> usize {
    let mut dropped = 0;
    for node in retiring {
        let Some(slot) = wide.get_mut(node.0 as usize) else {
            continue;
        };
        match slot.take() {
            Some(Cow::Owned(buffer)) => {
                wide_pool.push(buffer);
                dropped += usize::from(!counted_as_f32(*node));
            }
            Some(Cow::Borrowed(_)) => dropped += usize::from(!counted_as_f32(*node)),
            None => {}
        }
    }
    dropped
}

/// A fold whose element body runs in f32 over f32 operands and whose sum
/// accumulates in f64: `(ggml_float)(x * x)` summed, ggml's `rms_norm_f32`
/// (`ggml-cpu/ops.cpp:3957-3960`). Walks the fold's whole iteration space
/// once, row-major, evaluating the element body per point and adding it into
/// the f64 output slot its output axes address.
fn accumulate_f32_elements_in_f64<B: Deref<Target = [f32]>>(
    computed: &BoundOp,
    element_body: &ComposedBody,
    buffers: &[Option<B>],
    output: &mut [f64],
) -> Result<(), TensorError> {
    let not_lowerable = |reason| TensorError::NotLowerable {
        node: computed.node,
        reason,
    };
    let BoundOpKind::Reduce {
        operands,
        out_layout,
        ..
    } = &computed.kind
    else {
        return Err(not_lowerable(
            "only a reduce accumulates wider than its elements",
        ));
    };
    let sources: Vec<(&[f32], &Layout)> = operands
        .iter()
        .map(|(node, layout, lookup)| {
            if lookup.is_some() {
                return Err(not_lowerable("a widened accumulation does not gather"));
            }
            buffers
                .get(node.0 as usize)
                .and_then(Option::as_deref)
                .map(|data| (data, layout))
                .ok_or(not_lowerable("operand buffer missing at evaluation time"))
        })
        .collect::<Result<_, _>>()?;
    output.fill(0.0);
    if let Some(row_squares) = squared_rows(computed, element_body, &sources) {
        sum_squared_rows_in_f64(row_squares, &computed.extents, out_layout, output);
        return Ok(());
    }
    let mut coordinate = vec![0u64; computed.extents.len()];
    let mut operand_values = vec![0.0f32; sources.len()];
    let mut step_values = vec![0.0f32; element_body.steps.len()];
    let total: u64 = computed.extents.iter().product();
    for _ in 0..total {
        for (value, (data, layout)) in operand_values.iter_mut().zip(&sources) {
            *value = data[layout.offset_of(&coordinate) as usize];
        }
        let element = apply_body(element_body, &operand_values, &mut step_values);
        output[out_layout.offset_of(&coordinate) as usize] += f64::from(element);
        for (axis, extent) in coordinate.iter_mut().zip(&computed.extents).rev() {
            *axis += 1;
            if *axis < *extent {
                break;
            }
            *axis = 0;
        }
    }
    Ok(())
}

/// The squared operand and its single reduced axis when the fold is a sum of
/// squares -- the shape every norm binds to, which
/// [`sum_squared_rows_in_f64`] sums without the generic body interpreter.
fn squared_rows<'data>(
    computed: &BoundOp,
    element_body: &ComposedBody,
    sources: &[(&'data [f32], &'data Layout)],
) -> Option<(&'data [f32], &'data Layout, usize)> {
    let BoundOpKind::Reduce {
        operands,
        output_axes,
        ..
    } = &computed.kind
    else {
        return None;
    };
    let squares = matches!(
        element_body.steps.as_slice(),
        [BodyStep { op: ScalarOp::Multiply, args }, rest @ ..]
            if args.as_slice() == [StepArg::Operand(0), StepArg::Operand(1)]
                && rest.iter().all(|step| step.op == ScalarOp::Identity)
    );
    let same_operand = matches!(operands.as_slice(), [(left, left_layout, None), (right, right_layout, None)]
        if left == right && left_layout == right_layout);
    let reduced: Vec<usize> = (0..computed.extents.len())
        .filter(|axis| !output_axes.contains(&(*axis as u16)))
        .collect();
    let (data, layout) = *sources.first()?;
    (squares && same_operand && reduced.len() == 1).then_some((data, layout, reduced[0]))
}

// each row is summed strictly in order, one f64 add per element, the order
// ggml's scalar `sum += (ggml_float)(x*x)` uses, so every row matches it bit
// for bit; eight rows advance together only so their chains overlap
fn sum_squared_rows_in_f64(
    (data, layout, reduced_axis): (&[f32], &Layout, usize),
    extents: &[u64],
    out_layout: &Layout,
    output: &mut [f64],
) {
    let width = extents[reduced_axis];
    let stride = layout.strides[reduced_axis];
    let row_count = extents.iter().product::<u64>() / width.max(1);
    let mut coordinate = vec![0u64; extents.len()];
    let mut rows: Vec<(i64, usize)> = Vec::with_capacity(row_count as usize);
    for _ in 0..row_count {
        rows.push((
            layout.offset_of(&coordinate),
            out_layout.offset_of(&coordinate) as usize,
        ));
        for axis in (0..extents.len())
            .rev()
            .filter(|axis| *axis != reduced_axis)
        {
            coordinate[axis] += 1;
            if coordinate[axis] < extents[axis] {
                break;
            }
            coordinate[axis] = 0;
        }
    }
    if stride == 1 {
        let row = |start: i64| &data[start as usize..start as usize + width as usize];
        let (groups, remainder) = rows.as_chunks::<ROW_LANES>();
        for group in groups {
            let sums = sum_interleaved_rows(core::array::from_fn(|lane| row(group[lane].0)));
            for (sum, (_, target)) in sums.iter().zip(group) {
                output[*target] = *sum;
            }
        }
        for (start, target) in remainder {
            output[*target] = row(*start)
                .iter()
                .fold(0.0f64, |sum, value| sum + f64::from(value * value));
        }
        return;
    }
    for (start, target) in &rows {
        let mut sum = 0.0f64;
        for step in 0..width as i64 {
            let value = data[(start + step * stride) as usize];
            sum += f64::from(value * value);
        }
        output[*target] = sum;
    }
}

fn sum_interleaved_rows(lanes: [&[f32]; ROW_LANES]) -> [f64; ROW_LANES] {
    let width = lanes[0].len();
    let lanes = lanes.map(|lane| &lane[..width]);
    let mut sums = [0.0f64; ROW_LANES];
    for index in 0..width {
        for (sum, lane) in sums.iter_mut().zip(&lanes) {
            let value = lane[index];
            *sum += f64::from(value * value);
        }
    }
    sums
}

const ROW_LANES: usize = 8;

/// A fused reduce that accumulates in f64 while its result is f32: the norm
/// whose reduce absorbed the widen and whose epilogue carries the f64 divide,
/// the narrowing identity and the f32 tail.
pub(super) fn run_widened_fold_into<B: Deref<Target = [f32]>>(
    computed: &BoundOp,
    buffers: &[Option<B>],
    output: &mut [f32],
) -> Result<(), TensorError> {
    run_widened_fold(computed, buffers, |target, _, narrowed| {
        output[target] = narrowed;
    })
}

/// The fold sums f32 elements in f64 (in ggml's order for a plain sum of
/// squares) and the epilogue runs in f64 up to its narrow -- the first
/// identity of an earlier step -- and in f32 after it, so every value crosses
/// to f32 exactly where the program says. `store` gets each result's offset
/// with its f64 and f32 values; an epilogue with no narrow yet is all f64.
fn run_widened_fold<B: Deref<Target = [f32]>>(
    computed: &BoundOp,
    buffers: &[Option<B>],
    store: impl FnMut(usize, f64, f32),
) -> Result<(), TensorError> {
    let not_lowerable = |reason| TensorError::NotLowerable {
        node: computed.node,
        reason,
    };
    let BoundOpKind::Reduce {
        element_body,
        output_axes,
        epilogue_body,
        epilogue_operands,
        epilogue_broadcast_axes,
        ..
    } = &computed.kind
    else {
        return Err(not_lowerable(
            "only a reduce accumulates wider than its elements",
        ));
    };
    let fold_extents: Vec<u64> = output_axes
        .iter()
        .map(|&axis| computed.extents[axis as usize])
        .collect();
    let fold_len = if epilogue_broadcast_axes.is_empty() {
        node_output_len(computed)
    } else {
        odometer_len(&fold_extents) as usize
    };
    let mut fold = vec![0.0f64; fold_len];
    accumulate_f32_elements_in_f64(computed, element_body, buffers, &mut fold)?;
    let operand_buffers: Vec<&[f32]> = epilogue_operands
        .iter()
        .map(|(node, _, lookup)| {
            if lookup.is_some() {
                return Err(not_lowerable(
                    "reduce epilogue does not support a gathered operand",
                ));
            }
            buffer_of(buffers, *node)
        })
        .collect::<Result<_, _>>()?;
    let narrow = epilogue_body.steps.iter().position(|step| {
        step.op == ScalarOp::Identity && matches!(step.args.as_slice(), [StepArg::Step(_)])
    });
    if epilogue_broadcast_axes.is_empty() {
        plain_widened_epilogue(computed, narrow, &fold, &operand_buffers, store)
    } else {
        broadcast_widened_epilogue(computed, narrow, &fold, &operand_buffers, store)
    }
}

fn plain_widened_epilogue(
    computed: &BoundOp,
    narrow: Option<usize>,
    fold: &[f64],
    operand_buffers: &[&[f32]],
    mut store: impl FnMut(usize, f64, f32),
) -> Result<(), TensorError> {
    let BoundOpKind::Reduce {
        output_axes,
        out_layout,
        epilogue_body,
        epilogue_operands,
        ..
    } = &computed.kind
    else {
        return Ok(());
    };
    let fold_extents: Vec<u64> = output_axes
        .iter()
        .map(|&axis| computed.extents[axis as usize])
        .collect();
    let every_step: Vec<usize> = (0..epilogue_body.steps.len()).collect();
    let mut coordinate = vec![0u64; fold_extents.len()];
    let mut full_coordinate = vec![0u64; computed.extents.len()];
    let mut steps = step_scratch(epilogue_body);
    let mut operand_values = vec![0.0f64; epilogue_operands.len() + 1];
    for flat in 0..odometer_len(&fold_extents) {
        unflatten_into(flat, &fold_extents, &mut coordinate);
        full_coordinate.fill(0);
        for (index, &axis) in output_axes.iter().enumerate() {
            full_coordinate[axis as usize] = coordinate[index];
        }
        let fold_offset = out_layout.offset_of(&full_coordinate) as usize;
        for (slot, (_, layout, _)) in epilogue_operands.iter().enumerate() {
            operand_values[slot] =
                f64::from(operand_buffers[slot][layout.offset_of(&coordinate) as usize]);
        }
        operand_values[epilogue_operands.len()] = fold[fold_offset];
        let mixed = (computed.node, epilogue_body, narrow);
        evaluate_mixed(mixed, &every_step, &operand_values, &mut steps)?;
        let (wide, narrowed) = last_step(&steps);
        store(fold_offset, wide, narrowed);
    }
    Ok(())
}

/// The norm's shape: one fold value per row, re-broadcast over the rest of
/// the full extents. Every step that reads only the fold and operands that
/// do not vary along a broadcast axis -- a norm's divide, narrow, `+ eps`,
/// root and reciprocal -- runs once per row; only the rest, the `x * inv * g`
/// multiplies, runs per element, walking the innermost broadcast axis by
/// stride. Hoisting changes where a step runs, never its inputs, so the
/// values are the same ones the per-element form computes.
fn broadcast_widened_epilogue(
    computed: &BoundOp,
    narrow: Option<usize>,
    fold: &[f64],
    operand_buffers: &[&[f32]],
    mut store: impl FnMut(usize, f64, f32),
) -> Result<(), TensorError> {
    let BoundOpKind::Reduce {
        output_axes,
        out_layout,
        epilogue_body,
        epilogue_operands,
        epilogue_broadcast_axes,
        ..
    } = &computed.kind
    else {
        return Ok(());
    };
    let Some((&inner, outer_axes)) = epilogue_broadcast_axes.split_last() else {
        return Ok(());
    };
    let mixed = (computed.node, epilogue_body, narrow);
    let fold_slot = epilogue_operands.len();
    let row_invariant_operand: Vec<bool> = epilogue_operands
        .iter()
        .map(|(_, layout, _)| {
            epilogue_broadcast_axes
                .iter()
                .all(|&axis| layout.stride(axis) == 0)
        })
        .chain(core::iter::once(true))
        .collect();
    let (per_row, per_element) = split_by_row(epilogue_body, &row_invariant_operand);
    let element_operands: Vec<(usize, i64)> = (0..fold_slot)
        .filter(|slot| !row_invariant_operand[*slot])
        .map(|slot| (slot, epilogue_operands[slot].1.stride(inner)))
        .collect();
    let row_major = row_major_strides(&computed.extents);
    let fold_extents: Vec<u64> = output_axes
        .iter()
        .map(|&axis| computed.extents[axis as usize])
        .collect();
    let outer_extents: Vec<u64> = outer_axes
        .iter()
        .map(|&axis| computed.extents[axis as usize])
        .collect();
    let inner_extent = computed.extents[inner as usize];
    let mut fold_coordinate = vec![0u64; fold_extents.len()];
    let mut outer_coordinate = vec![0u64; outer_extents.len()];
    let mut full_coordinate = vec![0u64; computed.extents.len()];
    let mut element_bases = vec![0i64; element_operands.len()];
    let mut steps = step_scratch(epilogue_body);
    let mut operand_values = vec![0.0f64; fold_slot + 1];
    for row in 0..odometer_len(&fold_extents) {
        unflatten_into(row, &fold_extents, &mut fold_coordinate);
        full_coordinate.fill(0);
        for (index, &axis) in output_axes.iter().enumerate() {
            full_coordinate[axis as usize] = fold_coordinate[index];
        }
        for (slot, (_, layout, _)) in epilogue_operands.iter().enumerate() {
            if row_invariant_operand[slot] {
                operand_values[slot] =
                    f64::from(operand_buffers[slot][layout.offset_of(&full_coordinate) as usize]);
            }
        }
        operand_values[fold_slot] = fold[out_layout.offset_of(&full_coordinate) as usize];
        evaluate_mixed(mixed, &per_row, &operand_values, &mut steps)?;
        for outer in 0..odometer_len(&outer_extents) {
            unflatten_into(outer, &outer_extents, &mut outer_coordinate);
            for (index, &axis) in outer_axes.iter().enumerate() {
                full_coordinate[axis as usize] = outer_coordinate[index];
            }
            full_coordinate[inner as usize] = 0;
            for (base, (slot, _)) in element_bases.iter_mut().zip(&element_operands) {
                *base = epilogue_operands[*slot].1.offset_of(&full_coordinate);
            }
            let output_base: u64 = full_coordinate
                .iter()
                .zip(&row_major)
                .map(|(coordinate, stride)| coordinate * stride)
                .sum();
            for position in 0..inner_extent {
                for (base, (slot, stride)) in element_bases.iter().zip(&element_operands) {
                    let offset = base + position as i64 * stride;
                    operand_values[*slot] = f64::from(operand_buffers[*slot][offset as usize]);
                }
                evaluate_mixed(mixed, &per_element, &operand_values, &mut steps)?;
                let (wide, narrowed) = last_step(&steps);
                let target = output_base + position * row_major[inner as usize];
                store(target as usize, wide, narrowed);
            }
        }
    }
    Ok(())
}

fn row_major_strides(extents: &[u64]) -> Vec<u64> {
    let mut strides = vec![1u64; extents.len()];
    for axis in (0..extents.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * extents[axis + 1];
    }
    strides
}

fn step_scratch(body: &ComposedBody) -> (Vec<f64>, Vec<f32>) {
    (
        vec![0.0f64; body.steps.len()],
        vec![0.0f32; body.steps.len()],
    )
}

fn last_step(steps: &(Vec<f64>, Vec<f32>)) -> (f64, f32) {
    let last = steps.0.len() - 1;
    (steps.0[last], steps.1[last])
}

/// The steps whose inputs are fixed for a whole row, then the rest, each in
/// body order.
fn split_by_row(body: &ComposedBody, row_invariant_operand: &[bool]) -> (Vec<usize>, Vec<usize>) {
    let mut row_invariant = vec![false; body.steps.len()];
    for (index, step) in body.steps.iter().enumerate() {
        row_invariant[index] = step.args.iter().all(|arg| match arg {
            StepArg::Operand(slot) => row_invariant_operand[*slot as usize],
            StepArg::Step(earlier) => row_invariant[*earlier as usize],
        });
    }
    (0..body.steps.len()).partition(|index| row_invariant[*index])
}

/// Runs `indices` of the body: a step before the narrow in f64, the narrow
/// and every step after it in f32.
fn evaluate_mixed(
    (node, body, narrow): (NodeId, &ComposedBody, Option<usize>),
    indices: &[usize],
    operands: &[f64],
    steps: &mut (Vec<f64>, Vec<f32>),
) -> Result<(), TensorError> {
    let (wide_steps, narrow_steps) = steps;
    let wide_until = narrow.unwrap_or(body.steps.len());
    for &index in indices {
        let step = &body.steps[index];
        if index < wide_until {
            let mut args = [0.0f64; 3];
            for (slot, arg) in step.args.iter().enumerate() {
                args[slot] = match arg {
                    StepArg::Operand(operand) => operands[*operand as usize],
                    StepArg::Step(earlier) => wide_steps[*earlier as usize],
                };
            }
            wide_steps[index] = <f64 as Element>::apply(node, step.op, &args[..step.args.len()])?;
            narrow_steps[index] = wide_steps[index] as f32;
        } else {
            let mut args = [0.0f32; 3];
            for (slot, arg) in step.args.iter().enumerate() {
                args[slot] = match arg {
                    StepArg::Operand(operand) => operands[*operand as usize] as f32,
                    StepArg::Step(earlier) => narrow_steps[*earlier as usize],
                };
            }
            narrow_steps[index] = apply_scalar_op(step.op, &args[..step.args.len()]);
            wide_steps[index] = f64::from(narrow_steps[index]);
        }
    }
    Ok(())
}
