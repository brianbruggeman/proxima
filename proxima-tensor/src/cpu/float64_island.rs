use super::*;

/// What [`step`] did with one bound op: nothing (the f32 path runs it), a
/// FLOAT64 node now held in the wide buffers, or an f32 narrowing of a wide
/// operand ready for the f32 buffers.
pub(super) enum IslandStep {
    NotIsland,
    Wide,
    Narrowed(Vec<f32>),
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
        wide[computed.node.0 as usize] = Some(Cow::Owned(output));
        return Ok(IslandStep::Wide);
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
            epilogue_body,
            epilogue_operands,
            ..
        } if reduce_epilogue_is_identity(epilogue_body, epilogue_operands) => {
            run_reduce_typed::<f64>(computed, wide, &[], output)
        }
        _ => Err(not_lowerable(
            "only elementwise, reduce and constant ops run in f64",
        )),
    }
}

/// Returns the f64 buffer of every node retiring at this position to
/// `wide_pool`, the f64 counterpart of each loop's own f32 retirement; the
/// count is how many live buffers it ended.
pub(super) fn retire(
    wide: &mut [Option<Cow<'static, [f64]>>],
    wide_pool: &mut Vec<Vec<f64>>,
    retiring: &[NodeId],
) -> usize {
    let mut dropped = 0;
    for node in retiring {
        let Some(slot) = wide.get_mut(node.0 as usize) else {
            continue;
        };
        match slot.take() {
            Some(Cow::Owned(buffer)) => {
                wide_pool.push(buffer);
                dropped += 1;
            }
            Some(Cow::Borrowed(_)) => dropped += 1,
            None => {}
        }
    }
    dropped
}
