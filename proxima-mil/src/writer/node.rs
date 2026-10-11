//! One program node to its MIL operations, in the block's order.

use half::f16;
use proxima_tensor::{DType, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp};

use super::message::{
    Immediate, Kind, Name, attribute_blob, attribute_immediate, begin_operation, input_immediate,
    input_name, operation_output,
};
use super::views::{Site, Stack, broadcast_operand, not_emitted};
use super::{Context, MilError};
use crate::census::scalar_form;
use crate::weights::BlobWriter;
use crate::wire::Encoder;

/// Emits the operations of `node`; a node the contraction below absorbs emits none.
///
/// # Errors
/// [`MilError`] for a node the writer cannot express or a weight it cannot find.
pub fn emit_node(
    context: &Context<'_>,
    node: NodeId,
    encoder: &mut Encoder<'_>,
    blob: &mut BlobWriter<'_>,
) -> Result<(), MilError> {
    let op = context
        .program
        .get(node.0 as usize)
        .ok_or(MilError::UnknownNode { node })?;
    match op {
        Op::Input { dtype, name, .. } => {
            input(context, node, *dtype, name.as_deref(), encoder, blob)
        }
        Op::Constant { dtype, value, .. } => constant(context, node, *dtype, *value, encoder),
        Op::Iota { dtype, .. } => iota(context, node, *dtype, encoder),
        Op::Elementwise {
            dtype,
            body,
            operands,
            ..
        } => elementwise(context, node, *dtype, *body, operands, encoder),
        Op::Reduce(reduce) => reduction(context, node, reduce, encoder),
    }
}

pub(super) fn kind_of(node: NodeId, dtype: DType) -> Result<Kind, MilError> {
    match dtype {
        DType::Float16 | DType::Float32 | DType::BFloat16 => Ok(Kind::Float16),
        DType::Bool => Ok(Kind::Bool),
        DType::Int8 | DType::Int16 | DType::Int32 | DType::UInt8 | DType::UInt16 => Ok(Kind::Int32),
        other => Err(MilError::UnsupportedDType { node, dtype: other }),
    }
}

fn input(
    context: &Context<'_>,
    node: NodeId,
    dtype: DType,
    name: Option<&str>,
    encoder: &mut Encoder<'_>,
    blob: &mut BlobWriter<'_>,
) -> Result<(), MilError> {
    let kind = kind_of(node, dtype)?;
    let dims = context.shapes.of(node);
    match name.and_then(|wanted| context.weight(wanted)) {
        Some(weight) => weight_constant(node, weight, dims, encoder, blob),
        None => {
            let source = function_input_name(node, name)?;
            cast(encoder, &source, &Name::node(node.0), kind, dims);
            Ok(())
        }
    }
}

pub(super) fn function_input_name(node: NodeId, name: Option<&str>) -> Result<Name, MilError> {
    match name {
        Some(text) => Name::identifier(text).ok_or_else(|| MilError::InvalidName {
            node,
            name: text.into(),
        }),
        None => Ok(Name::step(node.0, "in", 0)),
    }
}

fn weight_constant(
    node: NodeId,
    weight: &super::WeightView<'_>,
    dims: &[u64],
    encoder: &mut Encoder<'_>,
    blob: &mut BlobWriter<'_>,
) -> Result<(), MilError> {
    let offset = blob.write_weight(weight.bytes, weight.dtype)?;
    let name = Name::node(node.0);
    let operation = begin_operation(encoder, "const");
    operation_output(encoder, &name, Kind::Float16, dims);
    attribute_immediate(encoder, "name", Immediate::Text(name_text(&name)));
    attribute_blob(encoder, "val", dims, offset);
    encoder.close(operation);
    Ok(())
}

fn name_text(name: &Name) -> &str {
    core::str::from_utf8(name.as_bytes()).unwrap_or_default()
}

fn constant(
    context: &Context<'_>,
    node: NodeId,
    dtype: DType,
    value: f32,
    encoder: &mut Encoder<'_>,
) -> Result<(), MilError> {
    let kind = kind_of(node, dtype)?;
    let dims = context.shapes.of(node);
    let count = dims.iter().product::<u64>();
    let name = Name::node(node.0);
    let immediate = match kind {
        Kind::Bool => Immediate::Flag(value != 0.0),
        Kind::Int32 => Immediate::Int(value as i64),
        _ if dims.is_empty() => Immediate::Half(f16::from_f32(value)),
        _ => Immediate::HalfFill {
            value: f16::from_f32(value),
            count,
        },
    };
    let operation = begin_operation(encoder, "const");
    operation_output(encoder, &name, kind, dims);
    attribute_immediate(encoder, "name", Immediate::Text(name_text(&name)));
    attribute_immediate(encoder, "val", immediate);
    encoder.close(operation);
    Ok(())
}

fn iota(
    context: &Context<'_>,
    node: NodeId,
    dtype: DType,
    encoder: &mut Encoder<'_>,
) -> Result<(), MilError> {
    let kind = kind_of(node, dtype)?;
    let dims = context.shapes.of(node);
    let extent = i64::try_from(dims.first().copied().unwrap_or(0)).unwrap_or(0);
    let counted = if kind == Kind::Int32 {
        Name::node(node.0)
    } else {
        Name::step(node.0, "r", 0)
    };
    let operation = begin_operation(encoder, "range_1d");
    input_immediate(encoder, "end", Immediate::Int(extent));
    input_immediate(encoder, "start", Immediate::Int(0));
    input_immediate(encoder, "step", Immediate::Int(1));
    operation_output(encoder, &counted, Kind::Int32, dims);
    encoder.close(operation);
    if kind != Kind::Int32 {
        cast(encoder, &counted, &Name::node(node.0), kind, dims);
    }
    Ok(())
}

pub(super) fn cast(
    encoder: &mut Encoder<'_>,
    source: &Name,
    target: &Name,
    kind: Kind,
    dims: &[u64],
) {
    let operation = begin_operation(encoder, "cast");
    input_name(encoder, "x", source);
    input_immediate(encoder, "dtype", Immediate::Text(kind.cast_name()));
    operation_output(encoder, target, kind, dims);
    encoder.close(operation);
}

fn elementwise(
    context: &Context<'_>,
    node: NodeId,
    dtype: DType,
    body: ScalarOp,
    operands: &[(NodeId, proxima_tensor::IndexMap)],
    encoder: &mut Encoder<'_>,
) -> Result<(), MilError> {
    if absorbed_by_contraction(context, node) {
        return Ok(());
    }
    let kind = kind_of(node, dtype)?;
    let extents = context.shapes.of(node);
    let mut inputs = Stack::<Name>::new();
    for (position, (operand, map)) in operands.iter().enumerate() {
        let site = Site {
            host: node,
            position,
            kind: kind_of(*operand, context.dtype_of(*operand)?)?,
        };
        let aligned = broadcast_operand(context, encoder, &site, *operand, map, extents)?;
        inputs.push(node, aligned.name)?;
    }
    let operand_kind = kind_of(operands[0].0, context.dtype_of(operands[0].0)?)?;
    scalar_operation(
        node,
        body,
        inputs.as_slice(),
        (kind, operand_kind),
        extents,
        encoder,
    )
}

fn scalar_operation(
    node: NodeId,
    body: ScalarOp,
    inputs: &[Name],
    (kind, operand_kind): (Kind, Kind),
    dims: &[u64],
    encoder: &mut Encoder<'_>,
) -> Result<(), MilError> {
    if body == ScalarOp::Identity && kind != operand_kind {
        cast(encoder, &inputs[0], &Name::node(node.0), kind, dims);
        return Ok(());
    }
    let produced = match body {
        ScalarOp::Greater | ScalarOp::Equal => Kind::Bool,
        _ => kind,
    };
    let result = if produced == kind {
        Name::node(node.0)
    } else {
        Name::step(node.0, "b", 0)
    };
    let operator = scalar_form(body)
        .operators()
        .first()
        .copied()
        .unwrap_or("identity");
    let operation = begin_operation(encoder, operator);
    scalar_inputs(encoder, body, inputs);
    operation_output(encoder, &result, produced, dims);
    encoder.close(operation);
    if produced != kind {
        cast(encoder, &result, &Name::node(node.0), kind, dims);
    }
    Ok(())
}

fn scalar_inputs(encoder: &mut Encoder<'_>, body: ScalarOp, inputs: &[Name]) {
    let keys: &[&str] = match body {
        ScalarOp::Select => &["cond", "a", "b"],
        _ if body.arity() == 2 => &["x", "y"],
        _ => &["x"],
    };
    keys.iter()
        .zip(inputs)
        .for_each(|(key, name)| input_name(encoder, key, name));
    match body {
        ScalarOp::Negate => input_immediate(encoder, "y", Immediate::Half(f16::from_f32(-1.0))),
        ScalarOp::Reciprocal | ScalarOp::Logarithm => {
            input_immediate(encoder, "epsilon", Immediate::Half(f16::from_f32(0.0)));
        }
        _ => {}
    }
}

fn absorbed_by_contraction(context: &Context<'_>, node: NodeId) -> bool {
    let is_product = matches!(
        context.program.get(node.0 as usize),
        Some(Op::Elementwise {
            body: ScalarOp::Multiply,
            ..
        })
    );
    let mut readers = context
        .program
        .iter()
        .enumerate()
        .skip(node.0 as usize + 1)
        .filter(|(_, op)| op.dependencies().contains(&node))
        .peekable();
    is_product
        && readers.peek().is_some()
        && readers.all(|(index, _)| context.contraction_of(NodeId(index as u32)).is_some())
}

fn reduction(
    context: &Context<'_>,
    node: NodeId,
    reduce: &Reduce,
    encoder: &mut Encoder<'_>,
) -> Result<(), MilError> {
    if reduce.keep == Keep::Scan || reduce.init == ReduceInit::FirstElement {
        return Err(not_emitted(node, "scan or argmax reduction"));
    }
    match context.contraction_of(node) {
        Some(contraction) => super::contraction::emit(context, node, &contraction, encoder),
        None => super::reduce::emit(context, node, reduce, encoder),
    }
}
