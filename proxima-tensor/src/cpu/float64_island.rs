use super::*;

/// What [`step`] did with one bound op: nothing (the f32 path runs it), a
/// FLOAT64 node now held in the wide buffers, or an f32 narrowing of a wide
/// operand ready for the f32 buffers.
pub(super) enum IslandStep {
    NotIsland,
    Wide,
    Narrowed(Vec<f32>),
}

/// Runs a FLOAT64 bound op in f64, or narrows an f32 identity reading one,
/// leaving every other op to the f32 kernels. `wide` grows to the program
/// length on the first FLOAT64 node, so an f32-only program never allocates it.
pub(super) fn step<B: Deref<Target = [f32]>>(
    computed: &BoundOp,
    buffers: &[Option<B>],
    wide: &mut Vec<Option<Cow<'static, [f64]>>>,
    program_len: usize,
) -> Result<IslandStep, TensorError> {
    if computed.dtype == DType::FLOAT64 {
        if wide.is_empty() {
            wide.resize(program_len, None);
        }
        let mut output = vec![0.0f64; node_output_len(computed)];
        run_wide(computed, buffers, wide, &mut output)?;
        wide[computed.node.0 as usize] = Some(Cow::Owned(output));
        return Ok(IslandStep::Wide);
    }
    if wide.is_empty() || computed.dtype != DType::FLOAT32 {
        return Ok(IslandStep::NotIsland);
    }
    match single_wide_operand(computed, wide) {
        Some(source) => Ok(IslandStep::Narrowed(source.iter().map(|&value| value as f32).collect())),
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
    let not_lowerable = |reason| TensorError::NotLowerable { node: computed.node, reason };
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
                    .ok_or(not_lowerable("a widening convert's f32 operand has no buffer"))?;
                if narrow.len() != output.len() {
                    return Err(not_lowerable("a widening convert must read its operand one to one"));
                }
                for (slot, &value) in output.iter_mut().zip(narrow) {
                    *slot = f64::from(value);
                }
                Ok(())
            }
            _ => run_elementwise_typed::<f64>(computed, wide, &[], output),
        },
        BoundOpKind::Reduce { .. } => run_reduce_typed::<f64>(computed, wide, &[], output),
        _ => Err(not_lowerable("only elementwise, reduce and constant ops run in f64")),
    }
}

/// Drops the f64 buffer of every node retiring at this position, the f64
/// counterpart of each loop's own f32 retirement.
pub(super) fn retire(wide: &mut [Option<Cow<'static, [f64]>>], retiring: &[NodeId]) {
    for node in retiring {
        if let Some(slot) = wide.get_mut(node.0 as usize) {
            *slot = None;
        }
    }
}
