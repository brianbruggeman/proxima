//! `LayerNormalization`, `Gelu`, `Squeeze` and `Expand`: the four
//! standard-domain ops BERT-family exporters emit that the first 48
//! [`lower_node`](super) arms lack. Each is a composition of the same
//! closed vocabulary the parent module teaches -- `Elementwise`, `Reduce`,
//! `Constant` over [`ScalarOp`] bodies and [`IndexMap`] addressing -- so no
//! new `Op` form or `ScalarOp` is introduced (see [`crate`]'s scope note).
//!
//! The pattern each follows is one an existing arm already uses:
//! `LayerNormalization` is [`super::lower_softmax`]'s reduce-then-broadcast
//! with a mean in place of a max; `Gelu` is [`super::lower_sigmoid`]'s
//! chain of single-purpose elementwise steps; `Squeeze` is
//! [`super::lower_reshape`]'s layout change, materialized when a real axis
//! disappears; `Expand` is [`super::lower_binary`]'s numpy broadcast, pinned
//! to the target shape by a constant operand.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::f32::consts::{FRAC_1_SQRT_2, FRAC_2_SQRT_PI};

use proxima_tensor::{
    AxisIndex, DType, Extent, IndexMap, IndexPattern, NodeId, Op, ReduceInit, ScalarOp, append,
    projection,
};

use super::{
    FoldState, LowerError, Value, attr_float, attr_int, attr_ints, attr_str, bind_output,
    broadcast_shapes, build_elementwise, build_reduce, constant_scalar, identity_pattern, lookup,
    operand_pattern, optional_input, resolve_constant_array, scalar_broadcast_pattern,
    single_term_axis,
};
use crate::messages::NodeProto;

const GELU_CUBIC_COEFFICIENT: f32 = 0.044_715;
const SQRT_TWO_OVER_PI: f32 = FRAC_2_SQRT_PI * FRAC_1_SQRT_2;
const DEFAULT_EPSILON: f32 = 1e-5;

fn reject(node: &NodeProto<'_>, reason: String) -> LowerError {
    LowerError::UnsupportedShape {
        name: node.name.to_string(),
        op_type: node.op_type.to_string(),
        reason,
    }
}

fn real_rank(value: &Value) -> usize {
    value
        .view
        .as_ref()
        .map_or(value.shape.len(), |view| view.iter().flatten().count())
}

fn bind_value(
    values: &mut alloc::collections::BTreeMap<String, Value>,
    node: &NodeProto<'_>,
    value: Value,
) {
    if let Some(output_name) = node.output.first() {
        values.insert((*output_name).to_string(), value);
    }
}

fn pointwise(program: &mut Vec<Op>, body: ScalarOp, source: NodeId, rank: usize) -> NodeId {
    let operands = alloc::vec![(source, IndexMap::Affine(identity_pattern(rank)))];
    build_elementwise(program, body, operands)
}

fn pointwise_pair(
    program: &mut Vec<Op>,
    body: ScalarOp,
    lhs: NodeId,
    rhs: NodeId,
    rank: usize,
) -> NodeId {
    let operands = alloc::vec![
        (lhs, IndexMap::Affine(identity_pattern(rank))),
        (rhs, IndexMap::Affine(identity_pattern(rank))),
    ];
    build_elementwise(program, body, operands)
}

fn pointwise_constant(
    program: &mut Vec<Op>,
    body: ScalarOp,
    source: NodeId,
    constant: f32,
    rank: usize,
) -> NodeId {
    let constant_id = constant_scalar(program, constant);
    let operands = alloc::vec![
        (source, IndexMap::Affine(identity_pattern(rank))),
        (
            constant_id,
            IndexMap::Affine(scalar_broadcast_pattern(rank))
        ),
    ];
    build_elementwise(program, body, operands)
}

fn normalized_axis(node: &NodeProto<'_>, axis: i64, rank: usize) -> Result<usize, LowerError> {
    let resolved = if axis < 0 { axis + rank as i64 } else { axis };
    usize::try_from(resolved)
        .ok()
        .filter(|&candidate| candidate < rank)
        .ok_or_else(|| {
            reject(
                node,
                alloc::format!("axis {axis} is out of range for rank {rank}"),
            )
        })
}

fn kept_projection(rank: usize, axis: usize) -> IndexPattern {
    let kept: Vec<u16> = (0..axis as u16).collect();
    projection(rank as u16, &kept)
}

fn mean_over_tail(program: &mut Vec<Op>, source: NodeId, shape: &[u64], axis: usize) -> NodeId {
    let rank = shape.len();
    let sum = build_reduce(
        program,
        ScalarOp::Add,
        ReduceInit::Zero,
        source,
        identity_pattern(rank),
        kept_projection(rank, axis),
        None,
    );
    let count: u64 = shape[axis..].iter().product();
    pointwise_constant(program, ScalarOp::Multiply, sum, 1.0 / count as f32, axis)
}

fn read_with_statistic(
    program: &mut Vec<Op>,
    body: ScalarOp,
    wide: NodeId,
    statistic: NodeId,
    shape: &[u64],
    axis: usize,
) -> NodeId {
    let rank = shape.len();
    let operands = alloc::vec![
        (wide, IndexMap::Affine(identity_pattern(rank))),
        (statistic, IndexMap::Affine(kept_projection(rank, axis))),
    ];
    build_elementwise(program, body, operands)
}

fn standardize(
    program: &mut Vec<Op>,
    source: NodeId,
    shape: &[u64],
    axis: usize,
    epsilon: f32,
) -> NodeId {
    let mean = mean_over_tail(program, source, shape, axis);
    let centered = read_with_statistic(program, ScalarOp::Subtract, source, mean, shape, axis);
    let squared = pointwise_pair(program, ScalarOp::Multiply, centered, centered, shape.len());
    let variance = mean_over_tail(program, squared, shape, axis);
    let stabilized = pointwise_constant(program, ScalarOp::Add, variance, epsilon, axis);
    let deviation = pointwise(program, ScalarOp::SquareRoot, stabilized, axis);
    read_with_statistic(program, ScalarOp::Divide, centered, deviation, shape, axis)
}

fn apply_parameter(
    program: &mut Vec<Op>,
    node: &NodeProto<'_>,
    body: ScalarOp,
    normalized: NodeId,
    parameter: &Value,
    shape: &[u64],
) -> Result<NodeId, LowerError> {
    if broadcast_shapes(node, shape, &parameter.shape)? != shape {
        return Err(reject(
            node,
            alloc::format!(
                "parameter shape {:?} does not broadcast unidirectionally to input shape {shape:?}",
                parameter.shape
            ),
        ));
    }
    let operands = alloc::vec![
        (normalized, IndexMap::Affine(identity_pattern(shape.len()))),
        (
            parameter.node,
            IndexMap::Affine(operand_pattern(parameter, shape))
        ),
    ];
    Ok(build_elementwise(program, body, operands))
}

/// `LayerNormalization(X, Scale, B?; axis, epsilon)`: `(x - mean) /
/// sqrt(variance + epsilon) * scale + bias`, with mean and variance taken
/// over axes `[axis, rank)`. Two `Reduce(+)` steps and eight elementwise
/// steps over bodies the algebra already has; the statistics are read back
/// at full rank through [`kept_projection`], exactly how
/// [`super::lower_softmax`] reads its row maximum. The optional `Mean` and
/// `InvStdDev` outputs are a typed error rather than a silent drop.
pub(super) fn lower_layer_normalization(
    program: &mut Vec<Op>,
    values: &mut alloc::collections::BTreeMap<String, Value>,
    node: &NodeProto<'_>,
) -> Result<(), LowerError> {
    let input = lookup(values, node, 0)?.clone();
    let scale = lookup(values, node, 1)?.clone();
    if node.output.iter().skip(1).any(|name| !name.is_empty()) {
        return Err(reject(
            node,
            "LayerNormalization lowering produces Y only, not Mean or InvStdDev".to_string(),
        ));
    }
    if input.view.is_some() {
        return Err(reject(node, "LayerNormalization of a value carrying a virtual (Unsqueeze-inserted) axis view is not lowered".to_string()));
    }
    let axis = normalized_axis(
        node,
        attr_int(node, "axis").unwrap_or(-1),
        input.shape.len(),
    )?;
    let epsilon = attr_float(node, "epsilon").unwrap_or(DEFAULT_EPSILON);
    let standardized = standardize(program, input.node, &input.shape, axis, epsilon);
    let mut result = apply_parameter(
        program,
        node,
        ScalarOp::Multiply,
        standardized,
        &scale,
        &input.shape,
    )?;
    if optional_input(node, 2).is_some() {
        let bias = lookup(values, node, 2)?.clone();
        result = apply_parameter(program, node, ScalarOp::Add, result, &bias, &input.shape)?;
    }
    bind_output(values, node, 0, result, input.shape);
    Ok(())
}

fn erf_gate(program: &mut Vec<Op>, source: NodeId, rank: usize) -> NodeId {
    let scaled = pointwise_constant(program, ScalarOp::Multiply, source, FRAC_1_SQRT_2, rank);
    let erf = pointwise(program, ScalarOp::Erf, scaled, rank);
    pointwise_constant(program, ScalarOp::Add, erf, 1.0, rank)
}

fn tanh_gate(program: &mut Vec<Op>, source: NodeId, rank: usize) -> NodeId {
    let squared = pointwise_pair(program, ScalarOp::Multiply, source, source, rank);
    let cubed = pointwise_pair(program, ScalarOp::Multiply, squared, source, rank);
    let cubic = pointwise_constant(
        program,
        ScalarOp::Multiply,
        cubed,
        GELU_CUBIC_COEFFICIENT,
        rank,
    );
    let inner = pointwise_pair(program, ScalarOp::Add, source, cubic, rank);
    let scaled = pointwise_constant(program, ScalarOp::Multiply, inner, SQRT_TWO_OVER_PI, rank);
    let tanh = pointwise(program, ScalarOp::Tanh, scaled, rank);
    pointwise_constant(program, ScalarOp::Add, tanh, 1.0, rank)
}

/// `Gelu(X; approximate)`: `0.5 * x * (1 + erf(x / sqrt(2)))` for
/// `approximate = "none"` (the default), `0.5 * x * (1 + tanh(sqrt(2 / pi) *
/// (x + 0.044715 * x^3)))` for `"tanh"`. Single-operand steps read the input
/// at its real rank and pass `shape` and `view` through, the way
/// [`super::lower_unary`] does.
pub(super) fn lower_gelu(
    program: &mut Vec<Op>,
    values: &mut alloc::collections::BTreeMap<String, Value>,
    node: &NodeProto<'_>,
) -> Result<(), LowerError> {
    let input = lookup(values, node, 0)?.clone();
    let rank = real_rank(&input);
    let gate = match attr_str(node, "approximate").unwrap_or(b"none") {
        b"none" | b"" => erf_gate(program, input.node, rank),
        b"tanh" => tanh_gate(program, input.node, rank),
        other => {
            return Err(reject(
                node,
                alloc::format!(
                    "Gelu approximate {:?} is neither \"none\" nor \"tanh\"",
                    String::from_utf8_lossy(other)
                ),
            ));
        }
    };
    let half_input = pointwise_constant(program, ScalarOp::Multiply, input.node, 0.5, rank);
    let result = pointwise_pair(program, ScalarOp::Multiply, half_input, gate, rank);
    bind_value(
        values,
        node,
        Value {
            node: result,
            shape: input.shape,
            view: input.view,
            flatten_source: None,
        },
    );
    Ok(())
}

fn constant_operand(
    node: &NodeProto<'_>,
    fold: &FoldState,
    index: usize,
) -> Result<Vec<i64>, LowerError> {
    let name = node
        .input
        .get(index)
        .copied()
        .ok_or_else(|| LowerError::MissingInput {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            index,
        })?;
    resolve_constant_array(
        name,
        &fold.initializer_data,
        &fold.constant_values,
        &fold.constant_arrays,
    )
    .map(|array| array.iter().map(|&element| element as i64).collect())
    .ok_or_else(|| {
        reject(
            node,
            alloc::format!(
                "{} input {index} must be a lower-time constant",
                node.op_type
            ),
        )
    })
}

fn squeezed_axes(
    node: &NodeProto<'_>,
    fold: &FoldState,
    shape: &[u64],
) -> Result<Vec<usize>, LowerError> {
    let declared = match (optional_input(node, 1), attr_ints(node, "axes")) {
        (Some(_), _) => Some(constant_operand(node, fold, 1)?),
        (None, Some(axes)) => Some(axes.to_vec()),
        (None, None) => None,
    };
    let mut axes: Vec<usize> = match declared {
        Some(declared) => declared
            .iter()
            .map(|&axis| normalized_axis(node, axis, shape.len()))
            .collect::<Result<_, _>>()?,
        None => (0..shape.len()).filter(|&axis| shape[axis] == 1).collect(),
    };
    axes.sort_unstable();
    axes.dedup();
    match axes.iter().find(|&&axis| shape[axis] != 1) {
        Some(&axis) => Err(reject(
            node,
            alloc::format!("Squeeze axis {axis} has extent {}, not 1", shape[axis]),
        )),
        None => Ok(axes),
    }
}

fn drop_virtual_axes(
    input: &Value,
    view: &[Option<u16>],
    kept: &[usize],
    out_shape: Vec<u64>,
) -> Value {
    let kept_view: Vec<Option<u16>> = kept.iter().map(|&axis| view[axis]).collect();
    let still_virtual = kept_view.iter().any(Option::is_none);
    Value {
        node: input.node,
        shape: out_shape,
        view: still_virtual.then_some(kept_view),
        flatten_source: None,
    }
}

fn materialize_squeeze(
    program: &mut Vec<Op>,
    node: &NodeProto<'_>,
    input: &Value,
    kept: &[usize],
    out_shape: Vec<u64>,
) -> Result<Value, LowerError> {
    let real_of = |axis: usize| {
        input
            .view
            .as_ref()
            .map_or(Some(axis as u16), |view| view[axis])
    };
    if kept.iter().any(|&axis| real_of(axis).is_none()) {
        return Err(reject(node, "Squeeze of a real axis beside a remaining virtual (Unsqueeze-inserted) axis is not lowered".to_string()));
    }
    let axes: Vec<AxisIndex> = (0..input.shape.len())
        .filter(|&axis| real_of(axis).is_some())
        .map(|axis| {
            kept.iter()
                .position(|&candidate| candidate == axis)
                .map_or_else(AxisIndex::default, |position| {
                    single_term_axis(position as u16)
                })
        })
        .collect();
    let pattern = IndexPattern {
        iter_rank: kept.len() as u16,
        axes,
    };
    let operands = alloc::vec![(input.node, IndexMap::Affine(pattern))];
    let id = build_elementwise(program, ScalarOp::Identity, operands);
    Ok(Value {
        node: id,
        shape: out_shape,
        view: None,
        flatten_source: None,
    })
}

/// `Squeeze(data, axes?)`: removes size-1 axes, `axes` read from the second
/// input (opset 13) or the attribute (earlier), every unit axis when absent.
/// Dropping only virtual axes edits [`Value::view`] and emits no `Op`; once
/// a real axis disappears the view cannot name it, so the result is one
/// `Identity` elementwise whose operand pattern reads the dropped axis at
/// index zero -- the degenerate-axis spelling [`super::broadcast_pattern`]
/// uses for a broadcast unit dimension.
pub(super) fn lower_squeeze(
    program: &mut Vec<Op>,
    values: &mut alloc::collections::BTreeMap<String, Value>,
    fold: &FoldState,
    node: &NodeProto<'_>,
) -> Result<(), LowerError> {
    let input = lookup(values, node, 0)?.clone();
    let squeezed = squeezed_axes(node, fold, &input.shape)?;
    let kept: Vec<usize> = (0..input.shape.len())
        .filter(|axis| !squeezed.contains(axis))
        .collect();
    let out_shape: Vec<u64> = kept.iter().map(|&axis| input.shape[axis]).collect();
    let value = match &input.view {
        _ if squeezed.is_empty() => input.clone(),
        Some(view) if squeezed.iter().all(|&axis| view[axis].is_none()) => {
            drop_virtual_axes(&input, view, &kept, out_shape)
        }
        _ => materialize_squeeze(program, node, &input, &kept, out_shape)?,
    };
    bind_value(values, node, value);
    Ok(())
}

fn target_extents(node: &NodeProto<'_>, fold: &FoldState) -> Result<Vec<u64>, LowerError> {
    constant_operand(node, fold, 1)?
        .iter()
        .map(|&extent| u64::try_from(extent).ok().filter(|&positive| positive > 0))
        .collect::<Option<_>>()
        .ok_or_else(|| reject(node, "Expand shape entries must be positive".to_string()))
}

/// `Expand(input, shape)`: numpy bidirectional broadcast of `input` to
/// `shape`, `shape` a lower-time constant. The broadcast read is
/// [`operand_pattern`] -- the one [`super::lower_binary`] builds -- but a
/// lone broadcast operand leaves the widened axes with nothing pinning their
/// extent (`shape::infer` rejects it), so the read is multiplied by an
/// all-ones [`Op::Constant`] of the target shape. `x * 1.0` is bit-exact for
/// every finite `x` and for infinities, and the constant supplies the
/// extents.
pub(super) fn lower_expand(
    program: &mut Vec<Op>,
    values: &mut alloc::collections::BTreeMap<String, Value>,
    fold: &FoldState,
    node: &NodeProto<'_>,
) -> Result<(), LowerError> {
    let input = lookup(values, node, 0)?.clone();
    let out_shape = broadcast_shapes(node, &input.shape, &target_extents(node, fold)?)?;
    if out_shape == input.shape {
        bind_value(values, node, input);
        return Ok(());
    }
    let ones = append(
        program,
        Op::Constant {
            dtype: DType::FLOAT32,
            shape: out_shape
                .iter()
                .map(|&extent| Extent::Static(extent as u32))
                .collect(),
            value: 1.0,
        },
    );
    let operands = alloc::vec![
        (
            input.node,
            IndexMap::Affine(operand_pattern(&input, &out_shape))
        ),
        (ones, IndexMap::Affine(identity_pattern(out_shape.len()))),
    ];
    let id = build_elementwise(program, ScalarOp::Multiply, operands);
    bind_output(values, node, 0, id, out_shape);
    Ok(())
}
