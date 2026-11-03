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

// summed strictly in order, one f64 add per element, the order ggml's scalar
// `sum += (ggml_float)(x*x)` uses, so the f64 result matches it bit for bit
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
    for _ in 0..row_count {
        let start = layout.offset_of(&coordinate);
        let mut sum = 0.0f64;
        for step in 0..width as i64 {
            let value = data[(start + step * stride) as usize];
            sum += f64::from(value * value);
        }
        output[out_layout.offset_of(&coordinate) as usize] = sum;
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
}

/// A fused reduce that accumulates in f64 while its result is f32: the norm
/// whose reduce absorbed the widen and whose epilogue carries the f64 divide,
/// the narrowing identity and the f32 tail. The fold sums in f64 (in ggml's
/// order for a plain sum of squares), the epilogue runs in f64 up to and
/// including its first identity of an earlier step -- the narrow -- and in
/// f32 after it, so every value crosses to f32 exactly where the program says.
pub(super) fn run_widened_fold_into<B: Deref<Target = [f32]>>(
    computed: &BoundOp,
    buffers: &[Option<B>],
    output: &mut [f32],
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
        out_layout,
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
    let mut fold = vec![0.0f64; node_output_len(computed)];
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
    let walk: &[u64] = if epilogue_broadcast_axes.is_empty() {
        &fold_extents
    } else {
        &computed.extents
    };
    let mut coordinate = vec![0u64; walk.len()];
    let mut full_coordinate = vec![0u64; computed.extents.len()];
    let mut wide_steps = vec![0.0f64; epilogue_body.steps.len()];
    let mut narrow_steps = vec![0.0f32; epilogue_body.steps.len()];
    let mut operand_values = vec![0.0f64; epilogue_operands.len() + 1];
    for flat in 0..odometer_len(walk) {
        unflatten_into(flat, walk, &mut coordinate);
        if epilogue_broadcast_axes.is_empty() {
            full_coordinate.fill(0);
            for (index, &axis) in output_axes.iter().enumerate() {
                full_coordinate[axis as usize] = coordinate[index];
            }
        } else {
            full_coordinate.copy_from_slice(&coordinate);
        }
        let fold_offset = out_layout.offset_of(&full_coordinate) as usize;
        let operand_coordinate = if epilogue_broadcast_axes.is_empty() {
            &coordinate
        } else {
            &full_coordinate
        };
        for (slot, (_, layout, _)) in epilogue_operands.iter().enumerate() {
            operand_values[slot] =
                f64::from(operand_buffers[slot][layout.offset_of(operand_coordinate) as usize]);
        }
        operand_values[epilogue_operands.len()] = fold[fold_offset];
        let value = mixed_epilogue(
            computed.node,
            epilogue_body,
            narrow,
            &operand_values,
            &mut wide_steps,
            &mut narrow_steps,
        )?;
        let target = if epilogue_broadcast_axes.is_empty() {
            fold_offset
        } else {
            flat as usize
        };
        output[target] = value;
    }
    Ok(())
}

fn mixed_epilogue(
    node: NodeId,
    body: &ComposedBody,
    narrow: Option<usize>,
    operands: &[f64],
    wide_steps: &mut [f64],
    narrow_steps: &mut [f32],
) -> Result<f32, TensorError> {
    let wide_until = narrow.unwrap_or(body.steps.len());
    for (index, step) in body.steps.iter().enumerate() {
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
        }
    }
    Ok(narrow_steps[body.steps.len() - 1])
}
