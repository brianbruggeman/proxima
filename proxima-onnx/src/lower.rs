//! `NodeProto` -> [`proxima_tensor::Op`] lowering: the graph-composition
//! layer this crate's own module doc (see [`crate`]) says the parser never
//! performs. Every function here composes the closed op algebra
//! ([`Op::Input`]/[`Op::Elementwise`]/[`Op::Reduce`]/[`Op::Iota`]/
//! [`Op::Constant`], 17 [`ScalarOp`] bodies, [`IndexMap`] addressing) --
//! see `proxima-tensor/src/op.rs` and `proxima-tensor/src/map.rs` for that
//! vocabulary, and `proxima-tensor/src/spec.rs` (`matmul`/`softmax`/
//! `rmsnorm`) for the composition pattern this module follows. No new `Op`
//! form, `ScalarOp`, or `IndexMap` variant is added here -- an operator this
//! module cannot compose is a documented gap ([`LowerError`]), never a
//! reason to extend the target ISA.
//!
//! # Shape
//!
//! [`lower_graph`] is a pure function over an already-parsed [`GraphProto`]:
//! no IO, no partial progress, one call in, one [`Lowered`] program out --
//! it does not need to be a [`Pipe`](proxima_primitives::pipe::Pipe) of its
//! own, since `In -> Result<Out, Err>` is already exactly this function's
//! shape and a one-shot transform gains nothing from the trait's
//! composition machinery. It walks `graph.node` once, in the order ONNX's
//! own spec requires (every node's inputs are produced by an earlier node,
//! a graph input, or an initializer) -- the same backwards-reference
//! discipline [`Op`]'s own doc describes for the program it builds, so no
//! separate topological sort is needed.
//!
//! # Value environment
//!
//! [`Value`] tracks, per ONNX value name, the [`NodeId`] that produces it
//! and its shape (concrete `u64` extents -- symbolic/dynamic ONNX
//! dimensions are out of scope for this pass, see [`LowerError::UnsupportedShape`]).
//! Initializers become named [`Op::Input`] leaves up front (matching
//! `proxima-tensor`'s own convention -- see `spec.rs`'s `symbolic_leaf`/
//! `input_leaf` -- of binding weights by name via
//! [`proxima_tensor::evaluate_named`] rather than folding literal tensor
//! data into the program); [`Op::Constant`] is reserved for the
//! rank-0 broadcast scalars this module's own compositions need (Relu's
//! `0`, Sigmoid's `1`, Gemm's `alpha`/`beta`), exactly the role its own doc
//! describes.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use proxima_tensor::{
    AxisIndex, AxisTerm, DType, Extent, IndexMap, IndexPattern, Keep, NodeId, Op, Reduce,
    ReduceInit, ScalarOp, append, projection,
};
use thiserror::Error;

use crate::messages::{AttributeProto, DimensionValue, GraphProto, NodeProto, TensorProto, TypeValue, ValueInfoProto};

/// Everything that can go wrong lowering a parsed ONNX graph into a
/// [`proxima_tensor::Op`] program. Distinct from
/// [`crate::error::OnnxError`], which is about malformed protobuf bytes --
/// every error here is raised against an already-*valid* parse that this
/// pass cannot, or does not yet, compose.
#[derive(Debug, Error)]
pub enum LowerError {
    /// `op_type` has no lowering in this module -- either a genuine
    /// RISC-sufficiency gap (the current op algebra cannot express it) or a
    /// deferred one (expressible, not yet implemented). See this crate's
    /// `lower` module doc for the running list.
    #[error("node {name:?}: op_type {op_type:?} has no lowering to proxima_tensor::Op")]
    UnsupportedOp { name: String, op_type: String },

    #[error("node {name:?} (op_type {op_type:?}) references unknown value {value:?}")]
    UnknownValue { name: String, op_type: String, value: String },

    #[error("node {name:?} (op_type {op_type:?}) is missing required input at position {index}")]
    MissingInput { name: String, op_type: String, index: usize },

    #[error("node {name:?} (op_type {op_type:?}) has an unsupported shape: {reason}")]
    UnsupportedShape { name: String, op_type: String, reason: String },

    #[error("initializer {name:?} declares data_type {data_type} with no float_data/raw_data payload this lowering can decode")]
    UndecodableInitializer { name: String, data_type: i32 },

    #[error("graph {name:?} has no nodes")]
    EmptyGraph { name: String },

    /// The RISC-sufficiency boundary this module's control-flow lowering
    /// hits and does not cross: a pure-dataflow `Vec<Op>` with
    /// backwards-only references (`op.rs:9-11`) has no runtime branch and
    /// no back-edge, so an iteration count (`Loop`'s trip count or its
    /// per-iteration `cond`) that is only known from *computed data* has no
    /// representation in this ISA. Distinct from [`Self::UnsupportedShape`]:
    /// that variant names a gap this pass could close by composing more of
    /// the existing algebra; this one names a gap the algebra itself cannot
    /// close without becoming a different kind of machine (a VM with jumps),
    /// which is out of scope for a dataflow program. See this module's own
    /// doc for the `If`/`Scan`/`Loop` classification this error is raised
    /// from.
    #[error("node {name:?} (op_type {op_type:?}) needs control flow this dataflow ISA cannot express: {reason}")]
    DataDependentControlFlow { name: String, op_type: String, reason: String },
}

/// One ONNX value's lowering state: which [`NodeId`] produces it and its
/// concrete shape.
///
/// `view` is `None` for every physically produced value (an `Op` whose own
/// iteration space equals `shape`). [`lower_unsqueeze`] is the one producer
/// that sets it: `Unsqueeze` only inserts size-1 axes, so binding its output
/// to a *new* `Op::Elementwise` would leave that inserted axis with no
/// operand pinning its extent -- `shape::infer` requires every iteration
/// axis be covered by at least one operand, which a single-operand reshape
/// can never satisfy for a genuinely new axis. Instead `view` aliases
/// straight to the pre-Unsqueeze `node`, recording which of *its* real axes
/// (`Some`) or which are purely virtual padding (`None`) each logical axis
/// of `shape` is, so [`operand_pattern`] can build the consuming op's
/// `IndexPattern` directly against the real node -- exactly the
/// `projection(3, &[0, 2])`-shaped pattern `lower::matmul2d` already builds
/// by hand for this same broadcast shape.
#[derive(Debug, Clone)]
struct Value {
    node: NodeId,
    shape: Vec<u64>,
    view: Option<Vec<Option<u16>>>,
}

/// The result of [`lower_graph`]: a self-contained `Op` program, the
/// initializer data it expects bound by name (via
/// [`proxima_tensor::evaluate_named`]), the graph's own declared inputs
/// (also bound by name), and its declared outputs as `(name, NodeId)` pairs
/// -- exactly [`proxima_tensor::evaluate_named`]'s `outputs` argument shape.
#[derive(Debug)]
pub struct Lowered {
    pub program: Vec<Op>,
    /// `(name, data)` for every initializer this graph declared, decoded to
    /// f32 -- see [`decode_f32_tensor`] for the two payload shapes
    /// (`float_data` or little-endian `raw_data`) this pass understands.
    pub initializers: Vec<(String, Vec<f32>)>,
    pub graph_inputs: Vec<String>,
    pub graph_outputs: Vec<(String, NodeId)>,
}

/// Lower a parsed ONNX [`GraphProto`] into a [`proxima_tensor::Op`] program.
///
/// Composes, never invents: every branch below builds an `Op::Elementwise`/
/// `Op::Reduce`/`Op::Constant` composition over what
/// `proxima-tensor/src/spec.rs` already establishes as this ISA's
/// vocabulary (`matmul` is `Reduce(+)` over `Elementwise(*)`; `softmax` is
/// two reduces and three elementwise ops -- this module's [`lower_softmax`]
/// is that same shape). See this module's own doc for the value-tracking
/// discipline and the crate doc for the coverage table.
pub fn lower_graph(graph: &GraphProto<'_>) -> Result<Lowered, LowerError> {
    let mut program: Vec<Op> = Vec::new();
    let mut values: BTreeMap<String, Value> = BTreeMap::new();
    let mut initializers: Vec<(String, Vec<f32>)> = Vec::new();

    // `Reshape`'s `shape` operand is consumed only as *values* this pass
    // reads directly (see `lower_reshape`) -- its `NodeId` is never named by
    // any `Op` in the program. A shape-only initializer that appears
    // *nowhere else* therefore does not need a live `Op::Input` leaf: one
    // would sit dead in `program` forever, and a non-`Float32` dead leaf
    // (real ONNX `Reshape` shape tensors are `int64`) trips
    // `cpu::reject_non_float32`'s whole-program scan, which does not prune
    // unreachable nodes. Skipping the leaf for names used ONLY this way
    // keeps every genuinely consumed initializer (weights, `Gather`
    // indices, ...) on the same live-leaf path as before.
    let shape_only_names: alloc::collections::BTreeSet<&str> = graph
        .node
        .iter()
        .filter(|node| node.op_type == "Reshape")
        .filter_map(|node| node.input.get(1).copied())
        .filter(|&name| {
            let used_elsewhere = graph.node.iter().any(|node| {
                node.input.iter().enumerate().any(|(index, &input_name)| input_name == name && !(node.op_type == "Reshape" && index == 1))
            });
            let is_graph_output = graph.output.iter().any(|output| output.name == name);
            !used_elsewhere && !is_graph_output
        })
        .collect();

    let mut initializer_data: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    for tensor in &graph.initializer {
        let shape = tensor_shape(tensor);
        let data = decode_numeric_tensor(tensor)?;
        initializer_data.insert(tensor.name.to_string(), data.clone());
        if shape_only_names.contains(tensor.name) {
            continue;
        }
        let node = append(
            &mut program,
            Op::Input {
                dtype: onnx_dtype_to_op_dtype(tensor.data_type),
                shape: shape.iter().map(|&extent| Extent::Static(extent as u32)).collect(),
                name: Some(tensor.name.to_string()),
            },
        );
        values.insert(tensor.name.to_string(), Value { node, shape, view: None });
        initializers.push((tensor.name.to_string(), data));
    }

    let mut graph_inputs = Vec::new();
    for input in &graph.input {
        if values.contains_key(input.name) {
            continue;
        }
        let shape = value_info_shape(input)?;
        let node = append(
            &mut program,
            Op::Input {
                dtype: value_info_dtype(input),
                shape: shape.iter().map(|&extent| Extent::Static(extent as u32)).collect(),
                name: Some(input.name.to_string()),
            },
        );
        values.insert(input.name.to_string(), Value { node, shape, view: None });
        graph_inputs.push(input.name.to_string());
    }

    if graph.node.is_empty() {
        return Err(LowerError::EmptyGraph { name: graph.name.to_string() });
    }

    let mut constant_values: BTreeMap<String, f32> = BTreeMap::new();
    for node in &graph.node {
        lower_node(&mut program, &mut values, &initializer_data, &mut constant_values, node)?;
    }

    let mut graph_outputs = Vec::new();
    for output in &graph.output {
        let value = lookup_by_name(&values, output.name, "graph_output", graph.name)?;
        graph_outputs.push((output.name.to_string(), value.node));
    }

    Ok(Lowered { program, initializers, graph_inputs, graph_outputs })
}

fn lower_node(
    program: &mut Vec<Op>,
    values: &mut BTreeMap<String, Value>,
    initializer_data: &BTreeMap<String, Vec<f32>>,
    constant_values: &mut BTreeMap<String, f32>,
    node: &NodeProto<'_>,
) -> Result<(), LowerError> {
    match node.op_type {
        "Add" => lower_binary(program, values, node, ScalarOp::Add),
        "Sub" => lower_binary(program, values, node, ScalarOp::Subtract),
        "Mul" => lower_binary(program, values, node, ScalarOp::Multiply),
        "Div" => lower_binary(program, values, node, ScalarOp::Divide),
        "Relu" => lower_relu(program, values, node),
        "Sigmoid" => lower_sigmoid(program, values, node),
        "Tanh" => lower_unary(program, values, node, ScalarOp::Tanh),
        "Exp" => lower_unary(program, values, node, ScalarOp::Exponential),
        "Log" => lower_unary(program, values, node, ScalarOp::Logarithm),
        "Sqrt" => lower_unary(program, values, node, ScalarOp::SquareRoot),
        "Neg" => lower_unary(program, values, node, ScalarOp::Negate),
        "Reciprocal" => lower_unary(program, values, node, ScalarOp::Reciprocal),
        "Identity" => lower_unary(program, values, node, ScalarOp::Identity),
        "Erf" => lower_unary(program, values, node, ScalarOp::Erf),
        "Max" => lower_binary(program, values, node, ScalarOp::Maximum),
        "Min" => lower_binary(program, values, node, ScalarOp::Minimum),
        "Greater" => lower_binary(program, values, node, ScalarOp::Greater),
        "Equal" => lower_binary(program, values, node, ScalarOp::Equal),
        "MatMul" => lower_matmul(program, values, node),
        "Gemm" => lower_gemm(program, values, node),
        "Softmax" => lower_softmax(program, values, node),
        "Transpose" => lower_transpose(program, values, node),
        "Gather" => lower_gather(program, values, node),
        "Unsqueeze" => lower_unsqueeze(program, values, node),
        "Constant" => lower_constant(program, values, constant_values, node),
        "Where" => lower_where(program, values, node),
        "If" => lower_if(program, values, initializer_data, constant_values, node),
        "Scan" => lower_scan(program, values, initializer_data, constant_values, node),
        "Loop" => lower_loop(program, values, initializer_data, constant_values, node),
        "ReduceSum" => lower_reduce(program, values, node, ScalarOp::Add, ReduceInit::Zero),
        "ReduceMax" => lower_reduce(program, values, node, ScalarOp::Maximum, ReduceInit::NegativeInfinity),
        "ReduceMin" => lower_reduce(program, values, node, ScalarOp::Minimum, ReduceInit::PositiveInfinity),
        "ReduceProd" => lower_reduce(program, values, node, ScalarOp::Multiply, ReduceInit::One),
        "Reshape" => lower_reshape(values, initializer_data, node),
        "Flatten" => lower_flatten(values, node),
        "Concat" => lower_concat(program, values, node),
        "Conv" => lower_conv(program, values, node),
        "ConvTranspose" => lower_convtranspose(program, values, node),
        "MaxPool" => lower_maxpool(program, values, node),
        "AveragePool" => lower_averagepool(program, values, node),
        other => Err(LowerError::UnsupportedOp { name: node.name.to_string(), op_type: other.to_string() }),
    }
}

fn lookup<'value>(
    values: &'value BTreeMap<String, Value>,
    node: &NodeProto<'_>,
    index: usize,
) -> Result<&'value Value, LowerError> {
    let name = node.input.get(index).ok_or_else(|| LowerError::MissingInput {
        name: node.name.to_string(),
        op_type: node.op_type.to_string(),
        index,
    })?;
    lookup_by_name(values, name, node.op_type, node.name)
}

fn lookup_by_name<'value>(
    values: &'value BTreeMap<String, Value>,
    name: &str,
    op_type: &str,
    node_name: &str,
) -> Result<&'value Value, LowerError> {
    values.get(name).ok_or_else(|| LowerError::UnknownValue {
        name: node_name.to_string(),
        op_type: op_type.to_string(),
        value: name.to_string(),
    })
}

fn bind_output(values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>, index: usize, id: NodeId, shape: Vec<u64>) {
    if let Some(name) = node.output.get(index) {
        values.insert((*name).to_string(), Value { node: id, shape, view: None });
    }
}

fn find_attr<'node>(node: &'node NodeProto<'_>, name: &str) -> Option<&'node AttributeProto<'node>> {
    node.attribute.iter().find(|attribute| attribute.name == name)
}

fn attr_int(node: &NodeProto<'_>, name: &str) -> Option<i64> {
    find_attr(node, name).map(|attribute| attribute.i)
}

fn attr_float(node: &NodeProto<'_>, name: &str) -> Option<f32> {
    find_attr(node, name).map(|attribute| attribute.f)
}

fn attr_ints<'node>(node: &'node NodeProto<'_>, name: &str) -> Option<&'node [i64]> {
    find_attr(node, name).map(|attribute| attribute.ints.as_slice())
}

fn identity_pattern(rank: usize) -> IndexPattern {
    let axes: Vec<u16> = (0..rank as u16).collect();
    projection(rank as u16, &axes)
}

/// A rank-0 operand's pattern against an `out_rank`-dimensional iteration
/// space -- the spelling [`proxima_tensor::spec`] calls `"->s"`/`"->sd"`:
/// no axes, so the same scalar reads at every iteration position.
fn scalar_broadcast_pattern(out_rank: usize) -> IndexPattern {
    projection(out_rank as u16, &[])
}

fn constant_scalar(program: &mut Vec<Op>, value: f32) -> NodeId {
    append(program, Op::Constant { dtype: DType::Float32, shape: Vec::new(), value })
}

fn build_elementwise(program: &mut Vec<Op>, body: ScalarOp, operands: Vec<(NodeId, IndexMap)>) -> NodeId {
    append(program, Op::Elementwise { dtype: DType::Float32, body, operands, name: None })
}

fn build_reduce(
    program: &mut Vec<Op>,
    body: ScalarOp,
    init: ReduceInit,
    operand: NodeId,
    in_map: IndexPattern,
    out_map: IndexPattern,
    name: Option<String>,
) -> NodeId {
    append(
        program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body,
            init,
            operand,
            in_map: IndexMap::Affine(in_map),
            out_map: IndexMap::Affine(out_map),
            keep: Keep::Reduce,
            name,
        }),
    )
}

/// Numpy-style broadcast of two shapes, right-aligned -- the rule every
/// ONNX binary elementwise op (`Add`/`Sub`/`Mul`/`Div`) and `Gemm`'s bias
/// addition follow.
fn broadcast_shapes(node: &NodeProto<'_>, lhs: &[u64], rhs: &[u64]) -> Result<Vec<u64>, LowerError> {
    let rank = lhs.len().max(rhs.len());
    let mut reversed = Vec::with_capacity(rank);
    for axis_from_right in 0..rank {
        let left = extent_from_right(lhs, axis_from_right);
        let right = extent_from_right(rhs, axis_from_right);
        let resolved = match (left, right) {
            (a, b) if a == b => a,
            (1, b) => b,
            (a, 1) => a,
            (a, b) => {
                return Err(LowerError::UnsupportedShape {
                    name: node.name.to_string(),
                    op_type: node.op_type.to_string(),
                    reason: format!("incompatible broadcast extents {a} and {b} at axis -{}", axis_from_right + 1),
                });
            }
        };
        reversed.push(resolved);
    }
    reversed.reverse();
    Ok(reversed)
}

fn extent_from_right(shape: &[u64], axis_from_right: usize) -> u64 {
    if axis_from_right >= shape.len() { 1 } else { shape[shape.len() - 1 - axis_from_right] }
}

/// An operand's [`IndexPattern`] against a broadcast iteration space of
/// `out_shape`: right-aligned, with a degenerate (empty-terms) axis for
/// every size-1 operand dimension the output widens -- the general
/// numpy-broadcast case, not just the leading-axis-drop case `projection`
/// alone covers.
fn broadcast_pattern(operand_shape: &[u64], out_shape: &[u64]) -> IndexPattern {
    let out_rank = out_shape.len() as u16;
    let leading = out_shape.len() - operand_shape.len();
    let axes = operand_shape
        .iter()
        .enumerate()
        .map(|(index, &extent)| {
            let iter_axis = (leading + index) as u16;
            if extent == 1 && out_shape[leading + index] != 1 {
                AxisIndex::default()
            } else {
                AxisIndex { terms: core::iter::once(AxisTerm::projection(iter_axis)).collect(), offset: 0 }
            }
        })
        .collect();
    IndexPattern { iter_rank: out_rank, axes }
}

/// [`broadcast_pattern`] against `value`'s *logical* shape, then composed
/// with `value.view` (see [`Value`]'s own doc) so the resulting
/// [`IndexPattern`] addresses `value.node` at its real rank -- the general
/// form of the by-hand `projection(3, &[0, 2])` [`matmul2d`] already builds
/// for this exact broadcast shape.
fn operand_pattern(value: &Value, out_shape: &[u64]) -> IndexPattern {
    let out_rank = out_shape.len() as u16;
    let logical = broadcast_pattern(&value.shape, out_shape);
    let Some(view) = &value.view else { return logical };

    let mut by_real_axis: alloc::collections::BTreeMap<u16, u16> = alloc::collections::BTreeMap::new();
    for (logical_axis, axis_index) in logical.axes.iter().enumerate() {
        let Some(real_axis) = view[logical_axis] else { continue };
        if let [term] = axis_index.terms.as_slice() {
            by_real_axis.insert(real_axis, term.axis);
        }
    }
    let axes = (0..by_real_axis.len() as u16)
        .map(|real_axis| AxisIndex { terms: core::iter::once(AxisTerm::projection(by_real_axis[&real_axis])).collect(), offset: 0 })
        .collect();
    IndexPattern { iter_rank: out_rank, axes }
}

fn lower_binary(program: &mut Vec<Op>, values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>, body: ScalarOp) -> Result<(), LowerError> {
    let lhs = lookup(values, node, 0)?.clone();
    let rhs = lookup(values, node, 1)?.clone();
    let out_shape = broadcast_shapes(node, &lhs.shape, &rhs.shape)?;
    let operands = alloc::vec![
        (lhs.node, IndexMap::Affine(operand_pattern(&lhs, &out_shape))),
        (rhs.node, IndexMap::Affine(operand_pattern(&rhs, &out_shape))),
    ];
    let id = build_elementwise(program, body, operands);
    bind_output(values, node, 0, id, out_shape);
    Ok(())
}

fn lower_unary(program: &mut Vec<Op>, values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>, body: ScalarOp) -> Result<(), LowerError> {
    let input = lookup(values, node, 0)?.clone();
    let pattern = identity_pattern(input.shape.len());
    let id = build_elementwise(program, body, alloc::vec![(input.node, IndexMap::Affine(pattern))]);
    bind_output(values, node, 0, id, input.shape);
    Ok(())
}

/// `Relu(x) = max(x, 0)` -- one [`ScalarOp::Maximum`] against a rank-0
/// [`Op::Constant`], the decomposition the ISA's own doc names for exactly
/// this kind of clamp.
fn lower_relu(program: &mut Vec<Op>, values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>) -> Result<(), LowerError> {
    let input = lookup(values, node, 0)?.clone();
    let rank = input.shape.len();
    let zero = constant_scalar(program, 0.0);
    let operands = alloc::vec![
        (input.node, IndexMap::Affine(identity_pattern(rank))),
        (zero, IndexMap::Affine(scalar_broadcast_pattern(rank))),
    ];
    let id = build_elementwise(program, ScalarOp::Maximum, operands);
    bind_output(values, node, 0, id, input.shape);
    Ok(())
}

/// `Sigmoid(x) = 1 / (1 + exp(-x))` -- four elementwise ops over the
/// existing `Negate`/`Exponential`/`Add`/`Reciprocal` bodies; no dedicated
/// sigmoid primitive exists in the ISA on purpose (this crate's own doc:
/// "composite activations desugar into several expressions").
fn lower_sigmoid(program: &mut Vec<Op>, values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>) -> Result<(), LowerError> {
    let input = lookup(values, node, 0)?.clone();
    let rank = input.shape.len();
    let negated = build_elementwise(program, ScalarOp::Negate, alloc::vec![(input.node, IndexMap::Affine(identity_pattern(rank)))]);
    let exponentiated = build_elementwise(program, ScalarOp::Exponential, alloc::vec![(negated, IndexMap::Affine(identity_pattern(rank)))]);
    let one = constant_scalar(program, 1.0);
    let denominator = build_elementwise(
        program,
        ScalarOp::Add,
        alloc::vec![
            (exponentiated, IndexMap::Affine(identity_pattern(rank))),
            (one, IndexMap::Affine(scalar_broadcast_pattern(rank))),
        ],
    );
    let id = build_elementwise(program, ScalarOp::Reciprocal, alloc::vec![(denominator, IndexMap::Affine(identity_pattern(rank)))]);
    bind_output(values, node, 0, id, input.shape);
    Ok(())
}

/// `MatMul` for rank-2 or batched (rank >= 2) operands: the contracted
/// axis (`K`) is carried as one more non-projected axis in the shared
/// iteration space alongside `M`/`N`, and leading batch axes ride the same
/// way -- `Reduce(+)` over `Elementwise(*)` generalizes without a new `Op`
/// form because [`Reduce::in_map`]/`out_map` are already addressed against
/// an arbitrary-rank iteration space (see `proxima-tensor/src/op.rs`'s own
/// `Reduce` doc). Batch dims broadcast numpy-style via [`broadcast_shapes`]/
/// [`broadcast_pattern`] -- the same machinery [`lower_binary`] already
/// uses -- so `[B, M, K] x [K, N]` (rhs has no batch dims at all) is just the
/// degenerate case where `extent_from_right` treats the missing leading axes
/// as broadcastable.
fn lower_matmul(program: &mut Vec<Op>, values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>) -> Result<(), LowerError> {
    let lhs = lookup(values, node, 0)?.clone();
    let rhs = lookup(values, node, 1)?.clone();
    if lhs.shape.len() < 2 || rhs.shape.len() < 2 {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: "MatMul lowering requires both operands to be at least rank 2".to_string(),
        });
    }
    let lhs_batch = &lhs.shape[..lhs.shape.len() - 2];
    let rhs_batch = &rhs.shape[..rhs.shape.len() - 2];
    let out_batch = broadcast_shapes(node, lhs_batch, rhs_batch)?;
    let batch_rank = out_batch.len();

    let m = lhs.shape[lhs.shape.len() - 2];
    let k = lhs.shape[lhs.shape.len() - 1];
    let k_rhs = rhs.shape[rhs.shape.len() - 2];
    let n = rhs.shape[rhs.shape.len() - 1];
    if k != k_rhs {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: format!("contracted dim mismatch: lhs contributes {k}, rhs contributes {k_rhs}"),
        });
    }

    let iter_rank = (batch_rank + 3) as u16;
    let m_axis = batch_rank as u16;
    let k_axis = batch_rank as u16 + 1;
    let n_axis = batch_rank as u16 + 2;

    let lhs_pattern = batched_operand_pattern(lhs_batch, &out_batch, iter_rank, &[m_axis, k_axis]);
    let rhs_pattern = batched_operand_pattern(rhs_batch, &out_batch, iter_rank, &[k_axis, n_axis]);

    let out_shape: Vec<u64> = out_batch.iter().copied().chain([m, n]).collect();
    let out_kept: Vec<u16> = (0..batch_rank as u16).chain([m_axis, n_axis]).collect();

    let product = build_elementwise(program, ScalarOp::Multiply, alloc::vec![(lhs.node, IndexMap::Affine(lhs_pattern)), (rhs.node, IndexMap::Affine(rhs_pattern))]);
    let id = build_reduce(program, ScalarOp::Add, ReduceInit::Zero, product, identity_pattern(iter_rank as usize), projection(iter_rank, &out_kept), Some("matmul".to_string()));
    bind_output(values, node, 0, id, out_shape);
    Ok(())
}

/// An operand's pattern into a matmul's shared `(batch..., trailing_axes)`
/// iteration space: the operand's own batch dims broadcast against
/// `out_batch` exactly like [`broadcast_pattern`] (missing leading batch
/// axes included -- the `[B, M, K] x [K, N]` case), then `trailing_axes`
/// (its own `M`/`K` or `K`/`N` pair) are appended as plain projections.
fn batched_operand_pattern(operand_batch: &[u64], out_batch: &[u64], iter_rank: u16, trailing_axes: &[u16]) -> IndexPattern {
    let mut axes = broadcast_pattern(operand_batch, out_batch).axes;
    axes.extend(trailing_axes.iter().map(|&axis| AxisIndex { terms: core::iter::once(AxisTerm::projection(axis)).collect(), offset: 0 }));
    IndexPattern { iter_rank, axes }
}

/// The shared matmul core [`lower_matmul`] and [`lower_gemm`] both build:
/// an iteration space `(i, j, k)` with `k` contracted, `lhs`/`rhs` each
/// addressed by their own (possibly transposed) pattern into that space.
fn matmul2d(program: &mut Vec<Op>, lhs: NodeId, lhs_pattern: IndexPattern, rhs: NodeId, rhs_pattern: IndexPattern, name: Option<String>) -> NodeId {
    let product = build_elementwise(
        program,
        ScalarOp::Multiply,
        alloc::vec![(lhs, IndexMap::Affine(lhs_pattern)), (rhs, IndexMap::Affine(rhs_pattern))],
    );
    build_reduce(program, ScalarOp::Add, ReduceInit::Zero, product, projection(3, &[0, 1, 2]), projection(3, &[0, 1]), name)
}

/// `Gemm(A, B, C) = alpha * (A' @ B') + beta * C`, `A'`/`B'` optionally
/// transposed by swapping which iteration axis each operand's pattern
/// projects (transpose is a permuted [`IndexMap`], never a copy -- see
/// `proxima-tensor/src/map.rs`'s own doc table). `alpha`/`beta` are always
/// emitted as elementwise multiplies against a rank-0 [`Op::Constant`],
/// even at their 1.0 defaults -- one fewer special case than skipping the
/// no-op multiply, and no different a numeric result.
fn lower_gemm(program: &mut Vec<Op>, values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>) -> Result<(), LowerError> {
    let a = lookup(values, node, 0)?.clone();
    let b = lookup(values, node, 1)?.clone();
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: "Gemm lowering supports rank-2 A and B only".to_string(),
        });
    }
    let trans_a = attr_int(node, "transA").unwrap_or(0) != 0;
    let trans_b = attr_int(node, "transB").unwrap_or(0) != 0;
    let alpha = attr_float(node, "alpha").unwrap_or(1.0);
    let beta = attr_float(node, "beta").unwrap_or(1.0);

    let (m, k, a_pattern) =
        if trans_a { (a.shape[1], a.shape[0], projection(3, &[2, 0])) } else { (a.shape[0], a.shape[1], projection(3, &[0, 2])) };
    let (k2, n, b_pattern) =
        if trans_b { (b.shape[1], b.shape[0], projection(3, &[1, 2])) } else { (b.shape[0], b.shape[1], projection(3, &[2, 1])) };
    if k != k2 {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: format!("contracted dim mismatch: A contributes {k}, B contributes {k2}"),
        });
    }
    let out_shape = alloc::vec![m, n];

    let matmul = matmul2d(program, a.node, a_pattern, b.node, b_pattern, Some("gemm_matmul".to_string()));
    let alpha_node = constant_scalar(program, alpha);
    let scaled = build_elementwise(
        program,
        ScalarOp::Multiply,
        alloc::vec![(matmul, IndexMap::Affine(identity_pattern(2))), (alpha_node, IndexMap::Affine(scalar_broadcast_pattern(2)))],
    );

    let result = match node.input.get(2) {
        Some(c_name) => {
            let c_value = lookup_by_name(values, c_name, node.op_type, node.name)?.clone();
            let beta_node = constant_scalar(program, beta);
            // Scale `c` by beta at its OWN rank first (fully resolved by
            // `c`'s own pure projection), then broadcast the *result* into
            // `out_shape` in the same op that also carries `scaled`'s full
            // rank -- an isolated `c * beta` op addressed only by `c`'s
            // narrower pattern would leave the axes `c` broadcasts over
            // unconstrained (no operand in that op mentions them), the same
            // way `rmsnorm`'s `gamma` multiply in `spec.rs` always shares
            // its op with a full-rank sibling operand.
            let c_rank = c_value.shape.len();
            let c_scaled = build_elementwise(
                program,
                ScalarOp::Multiply,
                alloc::vec![
                    (c_value.node, IndexMap::Affine(identity_pattern(c_rank))),
                    (beta_node, IndexMap::Affine(scalar_broadcast_pattern(c_rank))),
                ],
            );
            let c_scaled_pattern = broadcast_pattern(&c_value.shape, &out_shape);
            build_elementwise(
                program,
                ScalarOp::Add,
                alloc::vec![(scaled, IndexMap::Affine(identity_pattern(2))), (c_scaled, IndexMap::Affine(c_scaled_pattern))],
            )
        }
        None => scaled,
    };
    bind_output(values, node, 0, result, out_shape);
    Ok(())
}

/// `Softmax` along any axis: max-shift, `exp`, sum-reduce, divide -- the
/// same four-step composition `proxima-tensor/src/spec.rs`'s attention
/// blocks build inline (see this crate's own module doc pointer to
/// `spec.rs`). The reduced axis generalizes to any `axis` attribute by
/// pointing both `Reduce`s' `out_map` (and the broadcast back for the
/// subtract/divide) at `kept` -- every axis but the reduced one -- via
/// [`projection`], reused both as the reduce's `out_map` and, unchanged, as
/// the later elementwise operand pattern reading that reduced result back at
/// full rank: no transpose round-trip needed, since [`Op::Reduce`]'s own
/// `in_map`/`out_map` already address an arbitrary-rank iteration space.
fn lower_softmax(program: &mut Vec<Op>, values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>) -> Result<(), LowerError> {
    let input = lookup(values, node, 0)?.clone();
    let rank = input.shape.len();
    if rank == 0 {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: "Softmax requires input of rank >= 1".to_string(),
        });
    }
    let axis = attr_int(node, "axis").unwrap_or(-1);
    let normalized_axis = if axis < 0 { axis + rank as i64 } else { axis };
    if normalized_axis < 0 || normalized_axis as usize >= rank {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: format!("Softmax axis {axis} is out of range for rank {rank}"),
        });
    }
    let reduced_axis = normalized_axis as u16;
    let kept: Vec<u16> = (0..rank as u16).filter(|&candidate| candidate != reduced_axis).collect();
    let out_map = projection(rank as u16, &kept);

    let row_max = build_reduce(program, ScalarOp::Maximum, ReduceInit::NegativeInfinity, input.node, identity_pattern(rank), out_map.clone(), None);
    let shifted = build_elementwise(
        program,
        ScalarOp::Subtract,
        alloc::vec![(input.node, IndexMap::Affine(identity_pattern(rank))), (row_max, IndexMap::Affine(out_map.clone()))],
    );
    let exponentiated = build_elementwise(program, ScalarOp::Exponential, alloc::vec![(shifted, IndexMap::Affine(identity_pattern(rank)))]);
    let row_sum = build_reduce(program, ScalarOp::Add, ReduceInit::Zero, exponentiated, identity_pattern(rank), out_map.clone(), None);
    let id = build_elementwise(
        program,
        ScalarOp::Divide,
        alloc::vec![(exponentiated, IndexMap::Affine(identity_pattern(rank))), (row_sum, IndexMap::Affine(out_map))],
    );
    bind_output(values, node, 0, id, input.shape);
    Ok(())
}

/// `Transpose(perm)`: a permuted [`IndexMap`] over an
/// [`ScalarOp::Identity`] elementwise op -- no data movement in the
/// algebra, only in whichever backend materializes the result, exactly the
/// "transpose | permute the projected iteration axes" row in
/// `proxima-tensor/src/map.rs`'s own doc table.
fn lower_transpose(program: &mut Vec<Op>, values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>) -> Result<(), LowerError> {
    let input = lookup(values, node, 0)?.clone();
    let rank = input.shape.len();
    let perm: Vec<u16> = match find_attr(node, "perm") {
        Some(attribute) if !attribute.ints.is_empty() => attribute.ints.iter().map(|&value| value as u16).collect(),
        _ => (0..rank as u16).rev().collect(),
    };
    if perm.len() != rank {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: format!("perm has {} entries, input has rank {rank}", perm.len()),
        });
    }
    let out_shape: Vec<u64> = perm.iter().map(|&axis| input.shape[axis as usize]).collect();

    // operand axis k (input axis k) is read from iteration axis i where
    // perm[i] == k, i.e. the inverse permutation.
    let mut inverse = alloc::vec![0u16; rank];
    for (destination_axis, &source_axis) in perm.iter().enumerate() {
        inverse[source_axis as usize] = destination_axis as u16;
    }
    let pattern = projection(rank as u16, &inverse);
    let id = build_elementwise(program, ScalarOp::Identity, alloc::vec![(input.node, IndexMap::Affine(pattern))]);
    bind_output(values, node, 0, id, out_shape);
    Ok(())
}

/// `Gather(data, indices, axis)`: [`IndexMap::Computed`] over
/// [`ScalarOp::Identity`], the same "gather (read-side)" row
/// `proxima-tensor/src/map.rs`'s doc table names and
/// `proxima-tensor/src/spec.rs`'s `embedding_lookup` builds for the
/// `axis=0`, rank-2-table case -- generalized here to any `data` rank, any
/// `indices` rank, and any gather `axis`: the iteration space orders `data`'s
/// leading (pre-axis) axes, then `indices`' own axes, then `data`'s trailing
/// (post-axis) axes -- exactly ONNX's own output shape
/// `data.shape[:axis] + indices.shape + data.shape[axis+1:]` -- so `base`'s
/// entries just shift which iteration axis each non-gathered `data` axis
/// reads from by `indices_rank` once past `axis`.
fn lower_gather(program: &mut Vec<Op>, values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>) -> Result<(), LowerError> {
    let data = lookup(values, node, 0)?.clone();
    let indices = lookup(values, node, 1)?.clone();
    if data.shape.is_empty() {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: "Gather requires data of rank >= 1".to_string(),
        });
    }
    let data_rank = data.shape.len();
    let axis = attr_int(node, "axis").unwrap_or(0);
    let normalized_axis = if axis < 0 { axis + data_rank as i64 } else { axis };
    if normalized_axis < 0 || normalized_axis as usize >= data_rank {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: format!("Gather axis {axis} is out of range for data rank {data_rank}"),
        });
    }
    let axis = normalized_axis as usize;

    let indices_rank = indices.shape.len();
    let iter_rank = (data_rank - 1 + indices_rank) as u16;
    let index_map = projection(iter_rank, &(0..indices_rank as u16).map(|offset| axis as u16 + offset).collect::<Vec<_>>());

    let mut base_axes: Vec<AxisIndex> = Vec::with_capacity(data_rank);
    for data_axis in 0..data_rank {
        if data_axis == axis {
            base_axes.push(AxisIndex::default());
        } else if data_axis < axis {
            base_axes.push(AxisIndex { terms: core::iter::once(AxisTerm::projection(data_axis as u16)).collect(), offset: 0 });
        } else {
            let iter_axis = (data_axis - 1 + indices_rank) as u16;
            base_axes.push(AxisIndex { terms: core::iter::once(AxisTerm::projection(iter_axis)).collect(), offset: 0 });
        }
    }
    let base = IndexPattern { iter_rank, axes: base_axes };
    let gathered_map = IndexMap::Computed { indices: indices.node, index_map, base, gathered_dim: axis as u16 };

    let id = append(program, Op::Elementwise { dtype: DType::Float32, body: ScalarOp::Identity, operands: alloc::vec![(data.node, gathered_map)], name: None });
    let mut out_shape = data.shape[..axis].to_vec();
    out_shape.extend_from_slice(&indices.shape);
    out_shape.extend_from_slice(&data.shape[axis + 1..]);
    bind_output(values, node, 0, id, out_shape);
    Ok(())
}

/// `Unsqueeze(axes)`: inserts a size-1 dimension at each position in `axes`
/// (sorted ascending, matching this crate's own [`crate::lift`] emission).
///
/// Binds a [`Value`] view straight onto the pre-Unsqueeze `node` rather than
/// appending a new `Op` -- see [`Value`]'s own doc for why a standalone
/// single-operand reshape that introduces a genuinely new axis cannot be an
/// `Op::Elementwise` of its own (`shape::infer` requires every iteration
/// axis be pinned by some operand, and a fresh axis has none). This is the
/// inverse of [`crate::lift::lift_graph`]'s `Unsqueeze` prelude for a
/// broadcast operand.
fn lower_unsqueeze(_program: &mut Vec<Op>, values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>) -> Result<(), LowerError> {
    let input = lookup(values, node, 0)?.clone();
    let mut axes: Vec<u16> = attr_ints(node, "axes").unwrap_or(&[]).iter().map(|&value| value as u16).collect();
    axes.sort_unstable();
    let out_rank = input.shape.len() + axes.len();
    let mut out_shape = Vec::with_capacity(out_rank);
    let mut view: Vec<Option<u16>> = Vec::with_capacity(out_rank);
    let mut input_axis = 0usize;
    for position in 0..out_rank as u16 {
        if axes.contains(&position) {
            out_shape.push(1u64);
            view.push(None);
        } else {
            out_shape.push(input.shape[input_axis]);
            let real_axis = match &input.view {
                Some(existing) => existing[input_axis],
                None => Some(input_axis as u16),
            };
            view.push(real_axis);
            input_axis += 1;
        }
    }
    if let Some(output_name) = node.output.first() {
        values.insert((*output_name).to_string(), Value { node: input.node, shape: out_shape, view: Some(view) });
    }
    Ok(())
}

/// `Constant(value: TensorProto)`: this pass only supports a *uniform*
/// tensor value (every element equal), which is exactly the shape
/// [`crate::lift::lift_graph`]'s own `Op::Constant` -> `Constant` node
/// emission produces -- [`Op::Constant`] itself carries a single scalar
/// broadcast across its declared shape, never per-element data.
fn lower_constant(
    program: &mut Vec<Op>,
    values: &mut BTreeMap<String, Value>,
    constant_values: &mut BTreeMap<String, f32>,
    node: &NodeProto<'_>,
) -> Result<(), LowerError> {
    let tensor = find_attr(node, "value").and_then(|attribute| attribute.t.as_ref()).ok_or_else(|| LowerError::UnsupportedShape {
        name: node.name.to_string(),
        op_type: node.op_type.to_string(),
        reason: "Constant node has no \"value\" tensor attribute".to_string(),
    })?;
    let shape = tensor_shape(tensor);
    let decoded = decode_numeric_tensor(tensor).map_err(|_| LowerError::UndecodableInitializer { name: node.name.to_string(), data_type: tensor.data_type })?;
    let value = decoded.first().copied().unwrap_or(0.0);
    if decoded.iter().any(|&element| element != value) {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: "Constant lowering supports a uniform tensor value only".to_string(),
        });
    }
    let id = append(program, Op::Constant { dtype: DType::Float32, shape: shape.iter().map(|&extent| Extent::Static(extent as u32)).collect(), value });
    if let Some(output_name) = node.output.first() {
        // recorded unconditionally (uniform value, whatever the shape) so a
        // downstream `If`/`Loop` condition or trip count sourced from a
        // `Constant` node -- not only a graph initializer -- is still
        // foldable at lower time; see `constant_scalar_value`.
        constant_values.insert((*output_name).to_string(), value);
    }
    bind_output(values, node, 0, id, shape);
    Ok(())
}

/// `Where(condition, X, Y)`: elementwise select, numpy-broadcast across all
/// three operands -- the reverse of [`crate::lift::scalar_op_type`]'s own
/// `ScalarOp::Select -> "Where"` emission. Pure dataflow, no subgraph: the
/// same [`ScalarOp::Select`] three-operand composition [`concat_pair`] and
/// [`pad_axis`] already build for their own clamp-and-select shapes.
fn lower_where(program: &mut Vec<Op>, values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>) -> Result<(), LowerError> {
    let cond = lookup(values, node, 0)?.clone();
    let lhs = lookup(values, node, 1)?.clone();
    let rhs = lookup(values, node, 2)?.clone();
    let out_shape = broadcast_shapes(node, &broadcast_shapes(node, &cond.shape, &lhs.shape)?, &rhs.shape)?;
    let operands = alloc::vec![
        (cond.node, IndexMap::Affine(operand_pattern(&cond, &out_shape))),
        (lhs.node, IndexMap::Affine(operand_pattern(&lhs, &out_shape))),
        (rhs.node, IndexMap::Affine(operand_pattern(&rhs, &out_shape))),
    ];
    let id = build_elementwise(program, ScalarOp::Select, operands);
    bind_output(values, node, 0, id, out_shape);
    Ok(())
}

/// An ONNX optional input: `node.input[index]` is either absent (the list
/// is shorter) or present as an empty string -- both spellings mean "not
/// provided" per the spec (`Loop`'s `M`/`cond` are the running example).
fn optional_input<'node>(node: &'node NodeProto<'_>, index: usize) -> Option<&'node str> {
    match node.input.get(index) {
        Some(&name) if !name.is_empty() => Some(name),
        _ => None,
    }
}

/// A build-time-constant scalar for `name`, if this pass can fold one --
/// either a `Constant` node's recorded uniform value ([`lower_constant`]) or
/// a decoded initializer whose every element agrees (the same "uniform
/// tensor" test [`lower_constant`] itself applies). This is the single test
/// [`lower_if`]/[`lower_loop`] use to decide whether a condition or trip
/// count is a lower-time constant (unrollable) or only known from computed
/// data (the RISC-sufficiency boundary, [`LowerError::DataDependentControlFlow`]).
fn constant_scalar_value(name: &str, initializer_data: &BTreeMap<String, Vec<f32>>, constant_values: &BTreeMap<String, f32>) -> Option<f32> {
    if let Some(&value) = constant_values.get(name) {
        return Some(value);
    }
    let data = initializer_data.get(name)?;
    let first = *data.first()?;
    data.iter().all(|&element| element == first).then_some(first)
}

/// Lowers `subgraph`'s own node list directly into the caller's shared
/// `program`/`values` -- `If`'s `then_branch`/`else_branch` and `Scan`'s/
/// `Loop`'s `body` all carry an ordinary [`GraphProto`] with no outer
/// framing of its own, so recursively driving [`lower_node`] over it is the
/// whole mechanism: every subgraph node that reads an outer-scope value
/// (ONNX's own implicit-capture rule for `If`/`Loop`/`Scan` bodies) finds it
/// already bound in `values`, exactly as if it were one more node in the
/// parent graph. Subgraph-local `initializer`s are out of scope (deferred:
/// a name a subgraph declares as its own initializer, never referenced by
/// an outer node, surfaces as an ordinary [`LowerError::UnknownValue`] the
/// first subgraph node that reads it -- If/Loop/Scan bodies in practice
/// only ever capture outer names or compute fresh ones via `Constant`).
fn lower_subgraph_nodes(
    program: &mut Vec<Op>,
    values: &mut BTreeMap<String, Value>,
    initializer_data: &BTreeMap<String, Vec<f32>>,
    constant_values: &mut BTreeMap<String, f32>,
    subgraph: &GraphProto<'_>,
) -> Result<(), LowerError> {
    for node in &subgraph.node {
        lower_node(program, values, initializer_data, constant_values, node)?;
    }
    Ok(())
}

fn required_graph_attr<'node>(node: &'node NodeProto<'_>, name: &str) -> Result<&'node GraphProto<'node>, LowerError> {
    find_attr(node, name).and_then(|attribute| attribute.g.as_ref()).ok_or_else(|| LowerError::UnsupportedShape {
        name: node.name.to_string(),
        op_type: node.op_type.to_string(),
        reason: format!("{} requires a {name:?} graph attribute", node.op_type),
    })
}

/// `If(cond, then_branch, else_branch)`.
///
/// A build-time-constant `cond` (a graph initializer or a folded
/// `Constant`) picks the branch AT LOWER TIME: only the chosen subgraph is
/// recursively lowered ([`lower_subgraph_nodes`]) and appended to `program`
/// -- the other branch contributes nothing, exactly a compile-time `if`.
///
/// A data-dependent `cond` cannot pick a branch at lower time, but *can*
/// still lower to pure dataflow (Option A from this module's doc): both
/// subgraphs are lowered unconditionally (ONNX subgraphs are pure and
/// side-effect-free, so evaluating both wastes compute, never correctness),
/// and each output position becomes one [`ScalarOp::Select`] over the two
/// branch results. This only works when each output pair is shape-
/// conformable -- if `then_branch` and `else_branch` produce a differently-
/// shaped tensor at the same output position, [`ScalarOp::Select`] has no
/// broadcast that reconciles them, and that is named as
/// [`LowerError::UnsupportedShape`] rather than silently picking one side.
fn lower_if(
    program: &mut Vec<Op>,
    values: &mut BTreeMap<String, Value>,
    initializer_data: &BTreeMap<String, Vec<f32>>,
    constant_values: &mut BTreeMap<String, f32>,
    node: &NodeProto<'_>,
) -> Result<(), LowerError> {
    let cond_name = node.input.first().ok_or_else(|| LowerError::MissingInput { name: node.name.to_string(), op_type: node.op_type.to_string(), index: 0 })?;
    let then_branch = required_graph_attr(node, "then_branch")?;
    let else_branch = required_graph_attr(node, "else_branch")?;

    if let Some(cond_value) = constant_scalar_value(cond_name, initializer_data, constant_values) {
        let chosen = if cond_value != 0.0 { then_branch } else { else_branch };
        lower_subgraph_nodes(program, values, initializer_data, constant_values, chosen)?;
        for (index, output_name) in node.output.iter().enumerate() {
            let branch_output = chosen.output.get(index).ok_or_else(|| LowerError::UnsupportedShape {
                name: node.name.to_string(),
                op_type: node.op_type.to_string(),
                reason: format!("chosen If branch declares fewer outputs than the If node at position {index}"),
            })?;
            let value = lookup_by_name(values, branch_output.name, node.op_type, node.name)?.clone();
            values.insert((*output_name).to_string(), value);
        }
        return Ok(());
    }

    let cond_value = lookup_by_name(values, cond_name, node.op_type, node.name)?.clone();
    lower_subgraph_nodes(program, values, initializer_data, constant_values, then_branch)?;
    lower_subgraph_nodes(program, values, initializer_data, constant_values, else_branch)?;
    for index in 0..node.output.len() {
        let then_output = then_branch.output.get(index).ok_or_else(|| LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: format!("then_branch declares fewer outputs than the If node at position {index}"),
        })?;
        let else_output = else_branch.output.get(index).ok_or_else(|| LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: format!("else_branch declares fewer outputs than the If node at position {index}"),
        })?;
        let then_value = lookup_by_name(values, then_output.name, node.op_type, node.name)?.clone();
        let else_value = lookup_by_name(values, else_output.name, node.op_type, node.name)?.clone();
        if then_value.shape != else_value.shape {
            return Err(LowerError::UnsupportedShape {
                name: node.name.to_string(),
                op_type: node.op_type.to_string(),
                reason: format!(
                    "data-dependent If cannot Select between differently-shaped branch outputs at position {index}: then={:?} else={:?}",
                    then_value.shape, else_value.shape
                ),
            });
        }
        let out_shape = then_value.shape.clone();
        let rank = out_shape.len();
        let operands = alloc::vec![
            (cond_value.node, IndexMap::Affine(operand_pattern(&cond_value, &out_shape))),
            (then_value.node, IndexMap::Affine(identity_pattern(rank))),
            (else_value.node, IndexMap::Affine(identity_pattern(rank))),
        ];
        let id = build_elementwise(program, ScalarOp::Select, operands);
        bind_output(values, node, index, id, out_shape);
    }
    Ok(())
}

/// A fixed (lower-time-known) slice of `value`'s leading axis at `index`:
/// operand axis 0 reads a plain constant `offset` -- no `terms`, so it is
/// not the iteration space's "pure projection" case
/// (`shape::infer::unify_iteration_space`, `proxima-tensor/src/shape.rs:195`)
/// and instead goes through `bounds_check`
/// (`proxima-tensor/src/shape.rs:411`), which validates `0 <= index <
/// value.shape[0]` directly against `value`'s own already-resolved extent.
/// Every trailing axis is a plain projection onto the one-narrower iteration
/// space. This is [`Scan`](lower_scan)'s per-iteration `scan_input` slice --
/// deliberately not [`IndexMap::Computed`] (that machinery is for a
/// data-dependent index; `index` here is a lower-time constant, the loop
/// counter of an unrolled `for`).
fn slice_axis0(program: &mut Vec<Op>, value: &Value, index: u64) -> Value {
    let rank = value.shape.len();
    let mut axes = Vec::with_capacity(rank);
    axes.push(AxisIndex { terms: Default::default(), offset: index as i32 });
    for axis in 1..rank {
        axes.push(AxisIndex { terms: core::iter::once(AxisTerm::projection((axis - 1) as u16)).collect(), offset: 0 });
    }
    let pattern = IndexPattern { iter_rank: (rank - 1) as u16, axes };
    let id = build_elementwise(program, ScalarOp::Identity, alloc::vec![(value.node, IndexMap::Affine(pattern))]);
    Value { node: id, shape: value.shape[1..].to_vec(), view: None }
}

/// `Scan(initial_state..., scan_input...)`: `Scan`'s own trip count is
/// never data-dependent in this crate's model -- it is the leading extent of
/// its `scan_input` tensors (`axis=0`, the only case this lowering
/// supports), and every extent this pass tracks is already
/// [`proxima_tensor::Extent::Static`] (see this module's own doc on
/// [`Value`]). So unlike `Loop`, `Scan` needs no constant-folding test at
/// all: it is unconditionally unrollable here, `body` appended once per
/// iteration with each `scan_input` read through [`slice_axis0`] and each
/// state variable rebound to the previous iteration's body output --
/// ordinary backwards-referencing dataflow, no new `Op` form.
///
/// Deferred, named gaps: `scan_output` (per-iteration outputs stacked back
/// into a leading axis needs a concat-shaped composition this pass does not
/// yet build), more than one `scan_input`/state variable, and any
/// `scan_input_axes`/`scan_output_axes`/`scan_input_directions` attribute
/// other than the all-default case.
fn lower_scan(
    program: &mut Vec<Op>,
    values: &mut BTreeMap<String, Value>,
    initializer_data: &BTreeMap<String, Vec<f32>>,
    constant_values: &mut BTreeMap<String, f32>,
    node: &NodeProto<'_>,
) -> Result<(), LowerError> {
    let name = node.name.to_string();
    let op_type = node.op_type.to_string();
    let num_scan_inputs = attr_int(node, "num_scan_inputs").ok_or_else(|| LowerError::UnsupportedShape {
        name: name.clone(),
        op_type: op_type.clone(),
        reason: "Scan requires a num_scan_inputs attribute".to_string(),
    })?;
    if num_scan_inputs != 1 || node.input.len() != 2 {
        return Err(LowerError::UnsupportedShape {
            name,
            op_type,
            reason: "Scan lowering supports exactly one state variable and one scan_input".to_string(),
        });
    }
    let body = required_graph_attr(node, "body")?;
    if body.input.len() != 2 || body.output.len() != 1 {
        return Err(LowerError::UnsupportedShape {
            name,
            op_type,
            reason: "Scan lowering requires a body with exactly one state input/output and one scan_input slice (no scan_output)".to_string(),
        });
    }

    let mut state = lookup(values, node, 0)?.clone();
    let scan_input = lookup(values, node, 1)?.clone();
    if scan_input.shape.is_empty() {
        return Err(LowerError::UnsupportedShape { name, op_type, reason: "Scan scan_input must be rank >= 1".to_string() });
    }
    let trip_count = scan_input.shape[0];

    let state_name = body.input[0].name;
    let slice_name = body.input[1].name;
    for iteration in 0..trip_count {
        let slice = slice_axis0(program, &scan_input, iteration);
        values.insert(state_name.to_string(), state.clone());
        values.insert(slice_name.to_string(), slice);
        lower_subgraph_nodes(program, values, initializer_data, constant_values, body)?;
        state = lookup_by_name(values, body.output[0].name, "Scan", node.name)?.clone();
    }
    bind_output(values, node, 0, state.node, state.shape);
    Ok(())
}

/// `Loop(M, cond, v_initial...)`: unrolls exactly `M` iterations when `M` is
/// a lower-time constant ([`constant_scalar_value`]) -- the static-trip-count
/// case, appending `body` once per iteration exactly like [`lower_scan`],
/// each loop-carried dependency rebound to the previous iteration's body
/// output. `body`'s own `cond_out` (its first declared output) is read but
/// never gated on: this lowering's documented, narrow assumption is that a
/// caller supplying a constant `M` intends exactly `M` iterations (ONNX's
/// own spec allows discarding every `cond_out` reference once `max_trip_count`
/// alone determines the loop, the case this lowering restricts to by
/// additionally requiring `cond` be absent or a lower-time-constant `true`).
///
/// # The RISC-sufficiency boundary
///
/// When `M` is absent, or present but not a lower-time constant, or `cond`
/// is present and not a lower-time-constant `true`, the number of iterations
/// depends on values only known once the program *runs* -- there is no
/// unroll-at-lower-time answer, honest or otherwise. A pure-dataflow
/// `Vec<Op>` with backwards-only references (`proxima-tensor/src/op.rs:9-11`)
/// has no runtime branch and no back-edge to fall back to either: every
/// `NodeId` a node reads must already be fully computed earlier in the same
/// fixed-length program, so "run this subprogram again, conditioned on data
/// this subprogram itself produced" has no expression in the algebra. This
/// is named [`LowerError::DataDependentControlFlow`], never a fabricated
/// primitive and never a silently-truncated unroll.
fn lower_loop(
    program: &mut Vec<Op>,
    values: &mut BTreeMap<String, Value>,
    initializer_data: &BTreeMap<String, Vec<f32>>,
    constant_values: &mut BTreeMap<String, f32>,
    node: &NodeProto<'_>,
) -> Result<(), LowerError> {
    let name = node.name.to_string();
    let op_type = node.op_type.to_string();
    let boundary = |reason: &str| LowerError::DataDependentControlFlow { name: name.clone(), op_type: op_type.clone(), reason: reason.to_string() };

    let trip_count = match optional_input(node, 0) {
        Some(trip_name) => match constant_scalar_value(trip_name, initializer_data, constant_values) {
            Some(value) => value as u64,
            None => return Err(boundary("trip count M is only known from computed data, not a lower-time constant")),
        },
        None => return Err(boundary("no trip count M was given, so iteration count depends only on a runtime cond")),
    };
    if let Some(cond_name) = optional_input(node, 1) {
        match constant_scalar_value(cond_name, initializer_data, constant_values) {
            Some(value) if value != 0.0 => {}
            Some(_) => return Err(boundary("initial cond is a lower-time-constant false, zero iterations is not modeled")),
            None => return Err(boundary("cond is only known from computed data, so iterations may terminate early at runtime")),
        }
    }

    let body = required_graph_attr(node, "body")?;
    let num_state = node.input.len().saturating_sub(2);
    if body.input.len() != num_state + 2 || body.output.len() != num_state + 1 {
        return Err(LowerError::UnsupportedShape {
            name,
            op_type,
            reason: "Loop lowering requires body inputs [iter_num, cond, state...] and outputs [cond_out, state...] with matching state counts".to_string(),
        });
    }

    let mut state: Vec<Value> = (0..num_state).map(|index| lookup(values, node, index + 2).cloned()).collect::<Result<_, _>>()?;

    for iteration in 0..trip_count {
        let iter_node = constant_scalar(program, iteration as f32);
        values.insert(body.input[0].name.to_string(), Value { node: iter_node, shape: Vec::new(), view: None });
        for (state_index, state_value) in state.iter().enumerate() {
            values.insert(body.input[state_index + 2].name.to_string(), state_value.clone());
        }
        lower_subgraph_nodes(program, values, initializer_data, constant_values, body)?;
        state = (0..num_state)
            .map(|index| lookup_by_name(values, body.output[index + 1].name, "Loop", node.name).cloned())
            .collect::<Result<_, _>>()?;
    }

    for (index, state_value) in state.iter().enumerate() {
        bind_output(values, node, index, state_value.node, state_value.shape.clone());
    }
    Ok(())
}

/// `ReduceSum`/`ReduceMax`/`ReduceMin`/`ReduceProd` with an `axes` attribute
/// (opset<13 attribute form, matching [`crate::lift::lift_graph`]'s own
/// emission) and `keepdims == 0` -- the only form [`Op::Reduce`] can
/// express, since it drops the reduced axes rather than keeping a size-1
/// placeholder.
fn lower_reduce(program: &mut Vec<Op>, values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>, body: ScalarOp, init: ReduceInit) -> Result<(), LowerError> {
    let input = lookup(values, node, 0)?.clone();
    let rank = input.shape.len();
    let keepdims = attr_int(node, "keepdims").unwrap_or(0);
    if keepdims != 0 {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: "reduce lowering supports keepdims=0 only".to_string(),
        });
    }
    let axes: Vec<usize> = attr_ints(node, "axes").unwrap_or(&[]).iter().map(|&value| if value < 0 { value + rank as i64 } else { value } as usize).collect();
    let kept: Vec<u16> = (0..rank as u16).filter(|axis| !axes.contains(&(*axis as usize))).collect();
    let out_shape: Vec<u64> = kept.iter().map(|&axis| input.shape[axis as usize]).collect();
    let id = build_reduce(program, body, init, input.node, identity_pattern(rank), projection(rank as u16, &kept), None);
    bind_output(values, node, 0, id, out_shape);
    Ok(())
}

/// `Reshape(data, shape)`: a contiguous reshape -- merge or split of axes --
/// is pure layout, never compute (same element order, reinterpreted
/// extents), so this binds a [`Value`] view straight onto `data`'s node,
/// exactly [`lower_unsqueeze`]'s alias pattern (see [`Value`]'s own doc),
/// rather than an `Op` that would need floor-div/mod to express the
/// merge direction affinely. `shape` must be a decoded initializer (ONNX's
/// own convention for a constant-folded target shape); `0` copies the
/// source extent (unless `allowzero`) and at most one `-1` is inferred from
/// the total element count, both per the ONNX `Reshape` spec.
fn lower_reshape(values: &mut BTreeMap<String, Value>, initializer_data: &BTreeMap<String, Vec<f32>>, node: &NodeProto<'_>) -> Result<(), LowerError> {
    let input = lookup(values, node, 0)?.clone();
    let shape_name = node.input.get(1).ok_or_else(|| LowerError::MissingInput { name: node.name.to_string(), op_type: node.op_type.to_string(), index: 1 })?;
    let shape_data = initializer_data.get(*shape_name).ok_or_else(|| LowerError::UnsupportedShape {
        name: node.name.to_string(),
        op_type: node.op_type.to_string(),
        reason: "Reshape lowering requires the shape input to be a decoded initializer".to_string(),
    })?;

    let allowzero = attr_int(node, "allowzero").unwrap_or(0) != 0;
    let mut target: Vec<i64> = shape_data.iter().map(|&value| value as i64).collect();
    for (axis, value) in target.iter_mut().enumerate() {
        if *value == 0 && !allowzero {
            *value = *input.shape.get(axis).ok_or_else(|| LowerError::UnsupportedShape {
                name: node.name.to_string(),
                op_type: node.op_type.to_string(),
                reason: format!("shape entry 0 at axis {axis} has no matching source axis to copy"),
            })? as i64;
        }
    }
    let negative_slots = target.iter().filter(|&&value| value == -1).count();
    if negative_slots > 1 {
        return Err(LowerError::UnsupportedShape { name: node.name.to_string(), op_type: node.op_type.to_string(), reason: "Reshape shape has more than one -1 entry".to_string() });
    }
    let total: i64 = input.shape.iter().product::<u64>() as i64;
    if negative_slots == 1 {
        let known_product: i64 = target.iter().filter(|&&value| value != -1).product();
        let inferred = if known_product == 0 { 0 } else { total / known_product };
        for value in &mut target {
            if *value == -1 {
                *value = inferred;
            }
        }
    }
    let out_shape: Vec<u64> = target.iter().map(|&value| value as u64).collect();
    if out_shape.iter().product::<u64>() as i64 != total {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: format!("Reshape target element count does not match source ({total} elements)"),
        });
    }

    if let Some(output_name) = node.output.first() {
        values.insert((*output_name).to_string(), Value { node: input.node, shape: out_shape, view: None });
    }
    Ok(())
}

/// `Flatten(axis)`: the two-axis special case of a contiguous reshape --
/// `[prod(dims[:axis]), prod(dims[axis:])]` -- computed straight from the
/// already-known source shape, no `shape` input to decode. Same view-alias
/// treatment as [`lower_reshape`]: layout, not compute.
fn lower_flatten(values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>) -> Result<(), LowerError> {
    let input = lookup(values, node, 0)?.clone();
    let rank = input.shape.len();
    let axis = attr_int(node, "axis").unwrap_or(1);
    let normalized_axis = if axis < 0 { axis + rank as i64 } else { axis };
    if normalized_axis < 0 || normalized_axis as usize > rank {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: format!("Flatten axis {axis} is out of range for rank {rank}"),
        });
    }
    let split = normalized_axis as usize;
    let leading: u64 = input.shape[..split].iter().product();
    let trailing: u64 = input.shape[split..].iter().product();
    let out_shape = alloc::vec![leading, trailing];

    if let Some(output_name) = node.output.first() {
        values.insert((*output_name).to_string(), Value { node: input.node, shape: out_shape, view: None });
    }
    Ok(())
}

/// `Concat(inputs..., axis)`: folded pairwise (`concat(a, b, c) =
/// concat(concat(a, b), c)`), each pairwise step a "clamped select" -- the
/// deferred doc's own sketch. Neither operand can be addressed by a plain
/// affine pattern along the concat axis (`lhs`'s valid range ends before
/// `rhs`'s begins), so each side reads through [`IndexMap::Computed`] with a
/// *clamped* index (`Minimum`/`Maximum` composing the position [`Op::Iota`]
/// down into that operand's own valid range -- never out of bounds), and
/// [`ScalarOp::Select`] picks the correct side by comparing the raw
/// (unclamped) position against `lhs`'s extent. This is the same `Computed`
/// mechanism [`lower_gather`] uses, generalized from an externally supplied
/// index to a locally computed one.
fn lower_concat(program: &mut Vec<Op>, values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>) -> Result<(), LowerError> {
    if node.input.len() < 2 {
        return Err(LowerError::MissingInput { name: node.name.to_string(), op_type: node.op_type.to_string(), index: 1 });
    }
    let first = lookup(values, node, 0)?.clone();
    let rank = first.shape.len();
    let axis = attr_int(node, "axis").ok_or_else(|| LowerError::UnsupportedShape {
        name: node.name.to_string(),
        op_type: node.op_type.to_string(),
        reason: "Concat requires an axis attribute".to_string(),
    })?;
    let normalized_axis = if axis < 0 { axis + rank as i64 } else { axis };
    if normalized_axis < 0 || normalized_axis as usize >= rank {
        return Err(LowerError::UnsupportedShape { name: node.name.to_string(), op_type: node.op_type.to_string(), reason: format!("Concat axis {axis} is out of range for rank {rank}") });
    }
    let axis = normalized_axis as usize;

    let mut accumulator = first;
    for index in 1..node.input.len() {
        let next = lookup(values, node, index)?.clone();
        accumulator = concat_pair(program, node, &accumulator, &next, axis)?;
    }
    bind_output(values, node, 0, accumulator.node, accumulator.shape);
    Ok(())
}

fn build_elementwise_dtype(program: &mut Vec<Op>, dtype: DType, body: ScalarOp, operands: Vec<(NodeId, IndexMap)>) -> NodeId {
    append(program, Op::Elementwise { dtype, body, operands, name: None })
}

fn concat_pair(program: &mut Vec<Op>, node: &NodeProto<'_>, lhs: &Value, rhs: &Value, axis: usize) -> Result<Value, LowerError> {
    let rank = lhs.shape.len();
    if rhs.shape.len() != rank {
        return Err(LowerError::UnsupportedShape { name: node.name.to_string(), op_type: node.op_type.to_string(), reason: "Concat inputs must share rank".to_string() });
    }
    let axis_u16 = axis as u16;
    let out_rank = rank as u16;
    let lhs_extent = lhs.shape[axis];
    let rhs_extent = rhs.shape[axis];
    let out_extent = lhs_extent + rhs_extent;
    let mut out_shape = lhs.shape.clone();
    out_shape[axis] = out_extent;

    // Only the two nodes bound directly as a `Computed::indices` reference
    // below (`lhs_index`/`rhs_index`) need an integer `DType` tag --
    // `cpu::reject_non_float32` only exempts nodes reachable that way, so
    // every node feeding *into* them (still exact-integer-valued f32
    // arithmetic under the hood, per this crate's own "every buffer is f32"
    // convention) stays tagged `Float32`.
    let position = append(program, Op::Iota { dtype: DType::Float32, extent: Extent::Static(out_extent as u32) });
    let lhs_extent_const = constant_scalar(program, lhs_extent as f32);
    let zero_const = constant_scalar(program, 0.0);
    let lhs_extent_minus1_const = constant_scalar(program, lhs_extent as f32 - 1.0);
    let rhs_extent_minus1_const = constant_scalar(program, rhs_extent as f32 - 1.0);

    let lhs_index = build_elementwise_dtype(
        program,
        DType::Int32,
        ScalarOp::Minimum,
        alloc::vec![(position, IndexMap::Affine(identity_pattern(1))), (lhs_extent_minus1_const, IndexMap::Affine(scalar_broadcast_pattern(1)))],
    );
    let position_minus_lhs = build_elementwise(
        program,
        ScalarOp::Subtract,
        alloc::vec![(position, IndexMap::Affine(identity_pattern(1))), (lhs_extent_const, IndexMap::Affine(scalar_broadcast_pattern(1)))],
    );
    let position_minus_lhs_floored = build_elementwise(
        program,
        ScalarOp::Maximum,
        alloc::vec![(position_minus_lhs, IndexMap::Affine(identity_pattern(1))), (zero_const, IndexMap::Affine(scalar_broadcast_pattern(1)))],
    );
    let rhs_index = build_elementwise_dtype(
        program,
        DType::Int32,
        ScalarOp::Minimum,
        alloc::vec![(position_minus_lhs_floored, IndexMap::Affine(identity_pattern(1))), (rhs_extent_minus1_const, IndexMap::Affine(scalar_broadcast_pattern(1)))],
    );
    let cond = build_elementwise(
        program,
        ScalarOp::Greater,
        alloc::vec![(lhs_extent_const, IndexMap::Affine(scalar_broadcast_pattern(1))), (position, IndexMap::Affine(identity_pattern(1)))],
    );

    let index_map_pattern = projection(out_rank, &[axis_u16]);
    let base = concat_base_pattern(out_rank, rank, axis);

    let lhs_gathered_map = IndexMap::Computed { indices: lhs_index, index_map: index_map_pattern.clone(), base: base.clone(), gathered_dim: axis_u16 };
    let lhs_gathered = append(program, Op::Elementwise { dtype: DType::Float32, body: ScalarOp::Identity, operands: alloc::vec![(lhs.node, lhs_gathered_map)], name: None });

    let rhs_gathered_map = IndexMap::Computed { indices: rhs_index, index_map: index_map_pattern, base, gathered_dim: axis_u16 };
    let rhs_gathered = append(program, Op::Elementwise { dtype: DType::Float32, body: ScalarOp::Identity, operands: alloc::vec![(rhs.node, rhs_gathered_map)], name: None });

    let id = build_elementwise(
        program,
        ScalarOp::Select,
        alloc::vec![
            (cond, IndexMap::Affine(projection(out_rank, &[axis_u16]))),
            (lhs_gathered, IndexMap::Affine(identity_pattern(rank))),
            (rhs_gathered, IndexMap::Affine(identity_pattern(rank))),
        ],
    );
    Ok(Value { node: id, shape: out_shape, view: None })
}

/// A `Computed` gather's `base` pattern for a concat operand: every axis but
/// `axis` reads its own iteration axis directly (the two concat operands
/// share every non-concat extent), `axis` itself is unused (the fetched,
/// clamped index supplies it instead) -- the same shape [`lower_gather`]'s
/// own `base_axes` construction builds.
fn concat_base_pattern(iter_rank: u16, rank: usize, skip_axis: usize) -> IndexPattern {
    let axes = (0..rank)
        .map(|data_axis| {
            if data_axis == skip_axis {
                AxisIndex::default()
            } else {
                AxisIndex { terms: core::iter::once(AxisTerm::projection(data_axis as u16)).collect(), offset: 0 }
            }
        })
        .collect();
    IndexPattern { iter_rank, axes }
}

fn attr_str<'node>(node: &'node NodeProto<'_>, name: &str) -> Option<&'node [u8]> {
    find_attr(node, name).map(|attribute| attribute.s)
}

fn attr_ints_or(node: &NodeProto<'_>, name: &str, default: &[i64]) -> Vec<i64> {
    attr_ints(node, name).map(<[i64]>::to_vec).unwrap_or_else(|| default.to_vec())
}

/// Zero- (or `fill`-) pads one axis of `value` by `(before, after)` --
/// `Conv`/`MaxPool`/`AveragePool`'s `pads` attribute. This is
/// [`concat_pair`]'s clamp-and-select shape with one real operand instead of
/// two: an [`Op::Iota`] position, shifted and clamped into the source's
/// valid range for an [`IndexMap::Computed`] gather, and a `Greater`-built
/// validity mask that [`ScalarOp::Select`] routes between the gathered value
/// and `fill`.
///
/// A negative-offset [`AxisIndex`] cannot spell this directly:
/// `shape::infer`'s `unify_iteration_space` rejects an out-of-bounds read at
/// the iteration space's own zero origin (`specs/conv2d.toml`'s header
/// documents this empirically), so the padded region must be a real,
/// zero-origined operand before any window reads it.
fn pad_axis(program: &mut Vec<Op>, value: &Value, axis: usize, before: u64, after: u64, fill: f32) -> Value {
    if before == 0 && after == 0 {
        return value.clone();
    }
    let rank = value.shape.len();
    let rank_u16 = rank as u16;
    let extent = value.shape[axis];
    let out_extent = extent + before + after;
    let mut out_shape = value.shape.clone();
    out_shape[axis] = out_extent;

    let position = append(program, Op::Iota { dtype: DType::Float32, extent: Extent::Static(out_extent as u32) });
    let before_const = constant_scalar(program, before as f32);
    let zero_const = constant_scalar(program, 0.0);
    let minus_one_const = constant_scalar(program, -1.0);
    let extent_const = constant_scalar(program, extent as f32);
    let extent_minus1_const = constant_scalar(program, extent as f32 - 1.0);

    let shifted = build_elementwise(
        program,
        ScalarOp::Subtract,
        alloc::vec![(position, IndexMap::Affine(identity_pattern(1))), (before_const, IndexMap::Affine(scalar_broadcast_pattern(1)))],
    );
    let clamped_low = build_elementwise(
        program,
        ScalarOp::Maximum,
        alloc::vec![(shifted, IndexMap::Affine(identity_pattern(1))), (zero_const, IndexMap::Affine(scalar_broadcast_pattern(1)))],
    );
    let clamped_index = build_elementwise_dtype(
        program,
        DType::Int32,
        ScalarOp::Minimum,
        alloc::vec![(clamped_low, IndexMap::Affine(identity_pattern(1))), (extent_minus1_const, IndexMap::Affine(scalar_broadcast_pattern(1)))],
    );
    let valid_low = build_elementwise(
        program,
        ScalarOp::Greater,
        alloc::vec![(shifted, IndexMap::Affine(identity_pattern(1))), (minus_one_const, IndexMap::Affine(scalar_broadcast_pattern(1)))],
    );
    let valid_high = build_elementwise(
        program,
        ScalarOp::Greater,
        alloc::vec![(extent_const, IndexMap::Affine(scalar_broadcast_pattern(1))), (shifted, IndexMap::Affine(identity_pattern(1)))],
    );
    let mask = build_elementwise(
        program,
        ScalarOp::Multiply,
        alloc::vec![(valid_low, IndexMap::Affine(identity_pattern(1))), (valid_high, IndexMap::Affine(identity_pattern(1)))],
    );

    let index_map_pattern = projection(rank_u16, &[axis as u16]);
    let base = concat_base_pattern(rank_u16, rank, axis);
    let gathered_map = IndexMap::Computed { indices: clamped_index, index_map: index_map_pattern, base, gathered_dim: axis as u16 };
    let gathered = build_elementwise(program, ScalarOp::Identity, alloc::vec![(value.node, gathered_map)]);

    let fill_const = constant_scalar(program, fill);
    let output = build_elementwise(
        program,
        ScalarOp::Select,
        alloc::vec![
            (mask, IndexMap::Affine(projection(rank_u16, &[axis as u16]))),
            (gathered, IndexMap::Affine(identity_pattern(rank))),
            (fill_const, IndexMap::Affine(scalar_broadcast_pattern(rank))),
        ],
    );
    Value { node: output, shape: out_shape, view: None }
}

/// A static, in-bounds contiguous slice of `value` along `axis`:
/// `[start, start + length)`, every other axis read straight through -- the
/// static-extent, no-div/no-mod half of grouped convolution's channel split
/// ([`lower_conv`]'s `group` handling), and general enough for any other
/// static-range slice this module later needs.
///
/// [`IndexMap::Computed`] over an `Iota + start` index, the same "gather
/// (read-side)" shape [`lower_gather`] and [`pad_axis`] both build (see
/// [`pad_axis`]'s own doc for why an offset [`AxisIndex`] cannot spell this
/// directly: `shape::infer`'s `unify_iteration_space` would try to pin the
/// sliced axis's extent from `value`'s own (larger) axis whenever `start`
/// happens to be `0`, exactly the axis-0 group case, so only a `Computed`
/// gather -- whose `base` operand is *skipped* by `unify_iteration_space`,
/// its extent instead supplied by the `indices` node's own `length`-sized
/// shape -- pins the output extent correctly for every `start`, `0`
/// included). No mask or clamp needed here, unlike [`pad_axis`]: every index
/// this builds is already in `value`'s bounds by construction (`start +
/// length <= value.shape[axis]`, [`lower_conv`]'s own group-size check).
fn slice_axis_range(program: &mut Vec<Op>, value: &Value, axis: usize, start: u64, length: u64) -> Value {
    let rank = value.shape.len();
    let rank_u16 = rank as u16;
    let mut out_shape = value.shape.clone();
    out_shape[axis] = length;

    let position = append(program, Op::Iota { dtype: DType::Float32, extent: Extent::Static(length as u32) });
    let start_const = constant_scalar(program, start as f32);
    let shifted = build_elementwise(
        program,
        ScalarOp::Add,
        alloc::vec![(position, IndexMap::Affine(identity_pattern(1))), (start_const, IndexMap::Affine(scalar_broadcast_pattern(1)))],
    );
    let index = build_elementwise_dtype(program, DType::Int32, ScalarOp::Identity, alloc::vec![(shifted, IndexMap::Affine(identity_pattern(1)))]);

    let index_map_pattern = projection(rank_u16, &[axis as u16]);
    let base = concat_base_pattern(rank_u16, rank, axis);
    let gathered_map = IndexMap::Computed { indices: index, index_map: index_map_pattern, base, gathered_dim: axis as u16 };
    let id = build_elementwise(program, ScalarOp::Identity, alloc::vec![(value.node, gathered_map)]);
    Value { node: id, shape: out_shape, view: None }
}

/// A `Value`-level permutation, the same composition [`lower_transpose`]
/// builds against `values`/`node` directly -- factored out so
/// [`lower_convtranspose`] can permute a weight tensor's channel axes
/// without a synthetic [`NodeProto`] to drive it through.
fn permute_value(program: &mut Vec<Op>, value: &Value, perm: &[u16]) -> Value {
    let rank = value.shape.len();
    let out_shape: Vec<u64> = perm.iter().map(|&axis| value.shape[axis as usize]).collect();
    let mut inverse = alloc::vec![0u16; rank];
    for (destination_axis, &source_axis) in perm.iter().enumerate() {
        inverse[source_axis as usize] = destination_axis as u16;
    }
    let pattern = projection(rank as u16, &inverse);
    let id = build_elementwise(program, ScalarOp::Identity, alloc::vec![(value.node, IndexMap::Affine(pattern))]);
    Value { node: id, shape: out_shape, view: None }
}

/// Reverses `value` along `axis`: index `i` reads source index `extent - 1 -
/// i` -- the same [`IndexMap::Computed`] shape [`slice_axis_range`] builds
/// (see that function's own doc), with `index = (extent - 1) - Iota`
/// instead of `index = start + Iota`. [`lower_convtranspose`]'s stride-1
/// equivalence to an ordinary [`conv2d_core`] needs the kernel spatially
/// flipped; this is that flip, one spatial axis at a time.
fn reverse_axis(program: &mut Vec<Op>, value: &Value, axis: usize) -> Value {
    let rank = value.shape.len();
    let rank_u16 = rank as u16;
    let extent = value.shape[axis];
    let out_shape = value.shape.clone();

    let position = append(program, Op::Iota { dtype: DType::Float32, extent: Extent::Static(extent as u32) });
    let extent_minus1_const = constant_scalar(program, extent as f32 - 1.0);
    let reversed = build_elementwise(
        program,
        ScalarOp::Subtract,
        alloc::vec![(extent_minus1_const, IndexMap::Affine(scalar_broadcast_pattern(1))), (position, IndexMap::Affine(identity_pattern(1)))],
    );
    let index = build_elementwise_dtype(program, DType::Int32, ScalarOp::Identity, alloc::vec![(reversed, IndexMap::Affine(identity_pattern(1)))]);

    let index_map_pattern = projection(rank_u16, &[axis as u16]);
    let base = concat_base_pattern(rank_u16, rank, axis);
    let gathered_map = IndexMap::Computed { indices: index, index_map: index_map_pattern, base, gathered_dim: axis as u16 };
    let id = build_elementwise(program, ScalarOp::Identity, alloc::vec![(value.node, gathered_map)]);
    Value { node: id, shape: out_shape, view: None }
}

/// `ConvTranspose`, rank-4, `group = 1`, any `stride`/`dilation`.
///
/// # Two compositions, chosen by `stride`/`dilation`
///
/// `ConvTranspose`'s forward relation is `out_pos = in_pos*stride +
/// ky*dilation - pad` -- the exact two-term shape [`window_axis`] already
/// builds for ordinary `Conv`, but read in the *scatter* direction: for
/// `ConvTranspose`, `out_pos` is the value this affine combination
/// *produces*, not an input it reads. [`Op::Reduce`]'s `out_map` is
/// restricted to a *pure* single-axis projection in v1
/// (`proxima-tensor/src/shape.rs:449-455`, `"reduce output maps must be
/// pure projections in v1"`), so this two-term combination cannot be
/// `out_map` directly.
///
/// At `stride = 1, dilation = 1` the scatter direction collapses to an
/// *ordinary* convolution: `ConvTranspose(x, w, stride=1, pad=p) ==
/// Conv(pad(x, k-1-p), flip_spatial(transpose_channels(w)))` -- weight
/// channels swapped `[ci, co, kh, kw] -> [co, ci, kh, kw]` ([`permute_value`],
/// the same composition [`lower_transpose`] itself builds) and the kernel
/// spatially reversed ([`reverse_axis`], twice), then handed straight to
/// [`conv2d_core`] with adjusted padding `k - 1 - p` per side -- exactly
/// [`lower_conv`]'s `group = 1` composition, no new machinery at all.
///
/// At any other `stride`/`dilation`, [`convtranspose2d_scatter`] composes
/// the general relation directly via the scatter-add-as-masked-reduce idiom
/// `proxima-tensor/src/cpu.rs:16801-16860`
/// (`scatter_add_into_a_known_destination_via_mask_composition`) proves: keep
/// `oy`/`ox` as their own free (pure-projection) iteration axes, add
/// `ci`/`iy`/`ix`/`ky`/`kx` as reduced axes, and replace the affine `out_map`
/// with two `Equal` masks (`oy == iy*stride_h + ky*dilation_h - pad_top`,
/// `ox` likewise, each itself ordinary `Multiply`/`Subtract`/`Equal`
/// `ScalarOp`s over `Iota`-sourced positions -- see [`scatter_mask_axis`])
/// multiplied into the reduced operand -- no div/mod, no new `Op`/
/// `ScalarOp`/`IndexMap`.
fn lower_convtranspose(program: &mut Vec<Op>, values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>) -> Result<(), LowerError> {
    let image = lookup(values, node, 0)?.clone();
    let weight = lookup(values, node, 1)?.clone();
    let name = node.name.to_string();
    let op_type = node.op_type.to_string();

    if image.shape.len() != 4 || weight.shape.len() != 4 {
        return Err(LowerError::UnsupportedShape { name, op_type, reason: "ConvTranspose lowering supports 2D (rank-4 NCHW image, rank-4 CiCoKhKw weight) only".to_string() });
    }
    if let Some(auto_pad) = attr_str(node, "auto_pad")
        && auto_pad != b"NOTSET"
    {
        return Err(LowerError::UnsupportedShape { name, op_type, reason: "ConvTranspose lowering supports auto_pad=NOTSET only".to_string() });
    }
    let group = attr_int(node, "group").unwrap_or(1);
    if group != 1 {
        return Err(LowerError::UnsupportedShape { name, op_type, reason: "ConvTranspose lowering supports group=1 only".to_string() });
    }
    if attr_ints(node, "output_padding").is_some_and(|values| values.iter().any(|&value| value != 0)) {
        return Err(LowerError::UnsupportedShape { name, op_type, reason: "ConvTranspose lowering does not support a nonzero output_padding".to_string() });
    }
    let strides = attr_ints_or(node, "strides", &[1, 1]);
    let dilations = attr_ints_or(node, "dilations", &[1, 1]);
    let pads = attr_ints_or(node, "pads", &[0, 0, 0, 0]);
    if strides.len() != 2 || dilations.len() != 2 || pads.len() != 4 {
        return Err(LowerError::UnsupportedShape { name, op_type, reason: "ConvTranspose lowering supports 2D strides/dilations/pads only".to_string() });
    }
    let (stride_h, stride_w) = (strides[0], strides[1]);
    let (dilation_h, dilation_w) = (dilations[0], dilations[1]);
    let (pad_top, pad_left, pad_bottom, pad_right) = (pads[0], pads[1], pads[2], pads[3]);

    let in_channels = image.shape[1];
    let weight_in_channels = weight.shape[0];
    let out_channels = weight.shape[1];
    let kernel_h = weight.shape[2];
    let kernel_w = weight.shape[3];
    if weight_in_channels != in_channels {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: format!("ConvTranspose weight in-channels {weight_in_channels} does not match image channels {in_channels}"),
        });
    }

    let bias = match node.input.get(2) {
        Some(bias_name) => Some(lookup_by_name(values, bias_name, node.op_type, node.name)?.clone()),
        None => None,
    };
    if let Some(bias) = &bias
        && bias.shape != alloc::vec![out_channels]
    {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: "ConvTranspose bias must be a rank-1 tensor sized to out_channels".to_string(),
        });
    }

    if stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1 {
        let new_pad_top = kernel_h as i64 - 1 - pad_top;
        let new_pad_bottom = kernel_h as i64 - 1 - pad_bottom;
        let new_pad_left = kernel_w as i64 - 1 - pad_left;
        let new_pad_right = kernel_w as i64 - 1 - pad_right;
        if new_pad_top < 0 || new_pad_bottom < 0 || new_pad_left < 0 || new_pad_right < 0 {
            return Err(LowerError::UnsupportedShape {
                name: node.name.to_string(),
                op_type: node.op_type.to_string(),
                reason: "ConvTranspose lowering requires pads <= kernel_size - 1 on every side at stride=1".to_string(),
            });
        }

        let transposed_weight = permute_value(program, &weight, &[1, 0, 2, 3]);
        let flipped_h = reverse_axis(program, &transposed_weight, 2);
        let flipped_weight = reverse_axis(program, &flipped_h, 3);

        let attrs = Conv2dAttrs {
            stride_h: 1,
            stride_w: 1,
            dilation_h: 1,
            dilation_w: 1,
            pad_top: new_pad_top,
            pad_left: new_pad_left,
            pad_bottom: new_pad_bottom,
            pad_right: new_pad_right,
        };
        let result = conv2d_core(program, node, &image, &flipped_weight, bias.as_ref(), attrs, Some("convtranspose2d".to_string()))?;
        bind_output(values, node, 0, result.node, result.shape);
        return Ok(());
    }

    let result = convtranspose2d_scatter(program, &image, &weight, bias.as_ref(), kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w, pad_top, pad_left, pad_bottom, pad_right)?;
    bind_output(values, node, 0, result.node, result.shape);
    Ok(())
}

/// One position axis of the general `ConvTranspose` scatter relation
/// `oy == iy*stride + ky*dilation - pad`, materialized as a rank-3
/// `[out_extent, in_extent, kernel_extent]` `0.0`/`1.0` mask -- the affine
/// combination [`window_axis`] builds for ordinary `Conv` (reading *from* an
/// operand at a computed position), read instead as a *destination* equality
/// test, exactly the `Iota` + `Equal` scatter idiom
/// `proxima-tensor/src/cpu.rs:16801` (`scatter_add_into_a_known_destination_via_mask_composition`)
/// proves generalizes past a single Iota: two scaled `Iota`s summed and
/// shifted give the many-body affine combination that idiom's single
/// `indices` operand does not need. No div/mod, no new `Op`/`ScalarOp`/
/// `IndexMap`.
fn scatter_mask_axis(program: &mut Vec<Op>, out_extent: u64, in_extent: u64, kernel_extent: u64, stride: i64, dilation: i64, pad: i64) -> NodeId {
    let out_pos = append(program, Op::Iota { dtype: DType::Float32, extent: Extent::Static(out_extent as u32) });
    let in_pos = append(program, Op::Iota { dtype: DType::Float32, extent: Extent::Static(in_extent as u32) });
    let kernel_pos = append(program, Op::Iota { dtype: DType::Float32, extent: Extent::Static(kernel_extent as u32) });

    let stride_c = constant_scalar(program, stride as f32);
    let dilation_c = constant_scalar(program, dilation as f32);
    let pad_c = constant_scalar(program, pad as f32);

    let scaled_in = build_elementwise(program, ScalarOp::Multiply, alloc::vec![(in_pos, IndexMap::Affine(identity_pattern(1))), (stride_c, IndexMap::Affine(scalar_broadcast_pattern(1)))]);
    let scaled_kernel = build_elementwise(
        program,
        ScalarOp::Multiply,
        alloc::vec![(kernel_pos, IndexMap::Affine(identity_pattern(1))), (dilation_c, IndexMap::Affine(scalar_broadcast_pattern(1)))],
    );
    let summed = build_elementwise(
        program,
        ScalarOp::Add,
        alloc::vec![(scaled_in, IndexMap::Affine(projection(2, &[0]))), (scaled_kernel, IndexMap::Affine(projection(2, &[1])))],
    );
    let source_pos = build_elementwise(program, ScalarOp::Subtract, alloc::vec![(summed, IndexMap::Affine(identity_pattern(2))), (pad_c, IndexMap::Affine(scalar_broadcast_pattern(2)))]);

    build_elementwise(
        program,
        ScalarOp::Equal,
        alloc::vec![(out_pos, IndexMap::Affine(projection(3, &[0]))), (source_pos, IndexMap::Affine(projection(3, &[1, 2])))],
    )
}

/// The general (any `stride`/`dilation`) `ConvTranspose`, `group = 1`: a
/// masked-reduce scatter that widens its iteration space one axis at a time
/// -- `n`/`ci`/`iy`/`ix` (image) times `ci`/`co`/`ky`/`kx` (weight) first
/// (rank 7, `ci` reduced immediately by being shared), then the `oy` mask
/// folds in (rank 8), then the `ox` mask (rank 9), each step's own two
/// operands covering that step's whole iteration space -- `shape::infer`
/// resolves an axis's extent from a single node's operands, never across a
/// chain, which is why this cannot be one 9-operand product. `n`/`co`/`oy`/
/// `ox` stay free (pure projections) into the final [`build_reduce`];
/// `ci`/`iy`/`ix`/`ky`/`kx` reduce away. The two spatial `Equal` masks from
/// [`scatter_mask_axis`] gate which `(iy, ky)`/`(ix, kx)` pairs actually
/// contribute to each `(oy, ox)`. See [`lower_convtranspose`]'s own doc for
/// why `Op::Reduce`'s pure-projection `out_map` rules out a fused `IndexMap`
/// here, and this function's cost is `O(out_h*out_w*in_h*in_w*kh*kw)` per
/// the same doc's `O(out_h*in_h*kh)`-per-axis accounting.
#[allow(clippy::too_many_arguments)]
fn convtranspose2d_scatter(
    program: &mut Vec<Op>,
    image: &Value,
    weight: &Value,
    bias: Option<&Value>,
    kernel_h: u64,
    kernel_w: u64,
    stride_h: i64,
    stride_w: i64,
    dilation_h: i64,
    dilation_w: i64,
    pad_top: i64,
    pad_left: i64,
    pad_bottom: i64,
    pad_right: i64,
) -> Result<Value, LowerError> {
    let batch = image.shape[0];
    let in_h = image.shape[2];
    let in_w = image.shape[3];
    let out_channels = weight.shape[1];

    let out_h = ((in_h as i64 - 1) * stride_h - pad_top - pad_bottom + dilation_h * (kernel_h as i64 - 1) + 1).max(0) as u64;
    let out_w = ((in_w as i64 - 1) * stride_w - pad_left - pad_right + dilation_w * (kernel_w as i64 - 1) + 1).max(0) as u64;

    let mask_h = scatter_mask_axis(program, out_h, in_h, kernel_h, stride_h, dilation_h, pad_top);
    let mask_w = scatter_mask_axis(program, out_w, in_w, kernel_w, stride_w, dilation_w, pad_left);

    // Each of the three products below must, on its own, cover every axis
    // of its own iteration space from its two operands (`shape::infer`
    // resolves an iteration axis's extent from a single node's operands,
    // never across the chain) -- so `oy`/`ox` are folded in one at a time,
    // widening the iteration space only once a mask supplies that axis's
    // pure projection. Final axis order: 0=n 1=co 2=ci 3=iy 4=ix 5=ky 6=kx
    // 7=oy 8=ox.
    let reduce_space = projection(7, &[0, 2, 3, 4]); // image: n, ci, iy, ix
    let weight_pattern = projection(7, &[2, 1, 5, 6]); // weight: ci, co, kh, kw
    let image_times_weight = build_elementwise(program, ScalarOp::Multiply, alloc::vec![(image.node, IndexMap::Affine(reduce_space)), (weight.node, IndexMap::Affine(weight_pattern))]);

    let step1_pattern = projection(8, &[0, 1, 2, 3, 4, 5, 6]);
    let mask_h_pattern = projection(8, &[7, 3, 5]); // mask_h: oy, iy, ky
    let masked_h = build_elementwise(program, ScalarOp::Multiply, alloc::vec![(image_times_weight, IndexMap::Affine(step1_pattern)), (mask_h, IndexMap::Affine(mask_h_pattern))]);

    let step2_pattern = projection(9, &[0, 1, 2, 3, 4, 5, 6, 7]);
    let mask_w_pattern = projection(9, &[8, 4, 6]); // mask_w: ox, ix, kx
    let masked = build_elementwise(program, ScalarOp::Multiply, alloc::vec![(masked_h, IndexMap::Affine(step2_pattern)), (mask_w, IndexMap::Affine(mask_w_pattern))]);

    let out_shape = alloc::vec![batch, out_channels, out_h, out_w];
    let reduced = build_reduce(program, ScalarOp::Add, ReduceInit::Zero, masked, identity_pattern(9), projection(9, &[0, 1, 7, 8]), Some("convtranspose2d_scatter".to_string()));

    let result = match bias {
        Some(bias) => build_elementwise(
            program,
            ScalarOp::Add,
            alloc::vec![(reduced, IndexMap::Affine(identity_pattern(4))), (bias.node, IndexMap::Affine(projection(4, &[1])))],
        ),
        None => reduced,
    };
    Ok(Value { node: result, shape: out_shape, view: None })
}

/// One `stride*out + dilation*kernel` window term -- `map.rs`'s
/// `a_convolution_axis_is_two_terms` unit test is the whole reason
/// [`AxisIndex`] sums terms instead of holding one.
fn window_axis(out_axis: u16, kernel_axis: u16, stride: i64, dilation: i64) -> AxisIndex {
    AxisIndex {
        terms: alloc::vec![AxisTerm::scaled(out_axis, stride as i32), AxisTerm::scaled(kernel_axis, dilation as i32)].into_iter().collect(),
        offset: 0,
    }
}

/// `Conv`'s output spatial extent for one axis, ONNX's own formula:
/// `floor((padded - dilation*(kernel-1) - 1) / stride) + 1`. `None` when the
/// kernel's dilated span does not fit the padded axis at all.
fn conv_output_extent(padded_extent: u64, kernel_extent: u64, stride: i64, dilation: i64) -> Option<u64> {
    let span = (dilation as u64).checked_mul(kernel_extent.checked_sub(1)?)?.checked_add(1)?;
    if span > padded_extent {
        return None;
    }
    Some((padded_extent - span) / stride as u64 + 1)
}

/// The stride/dilation/output-extent parameters [`window_materialize`] and
/// [`conv_output_extent`] share -- one field group traveling together
/// (`Reduce`'s own doc names this idiom), not ten positional arguments.
#[derive(Debug, Clone, Copy)]
struct WindowSpec {
    out_h: u64,
    out_w: u64,
    kernel_h: u64,
    kernel_w: u64,
    stride_h: i64,
    stride_w: i64,
    dilation_h: i64,
    dilation_w: i64,
}

/// Materializes `image`'s (already-padded, rank-4 `[n, c, h, w]`) sliding
/// window as a concrete rank-6 `[n, c, out_h, out_w, kh, kw]` tensor: the
/// two-term window axis multiplied against an all-ones [`Op::Constant`]
/// stamp shaped `[out_h, out_w, kh, kw]`.
///
/// The stamp is not decoration -- [`shape::infer`](proxima_tensor::shape)'s
/// `unify_iteration_space` (`proxima-tensor/src/shape.rs:195-213`) resolves
/// an iteration axis's extent only from a *pure* single-term, zero-offset
/// projection, never from a windowed axis, and [`Op::Reduce`] unifies over
/// its one operand alone. Nothing else in this op purely projects
/// `oy`/`ox`/`ky`/`kx`, so without the stamp those four axes stay
/// unconstrained and inference fails. This is `specs/conv2d.toml`'s own
/// kernel-replication trick (its header explains the same requirement),
/// spelled with a cheap all-ones marker instead of duplicated real weight
/// data -- both pay im2col's inherent `O(out_h*out_w*kh*kw)` materialization
/// cost; neither is a new ISA primitive.
fn window_materialize(program: &mut Vec<Op>, image: &Value, spec: WindowSpec) -> Value {
    let image_pattern = IndexPattern {
        iter_rank: 6,
        axes: alloc::vec![
            AxisIndex { terms: core::iter::once(AxisTerm::projection(0)).collect(), offset: 0 },
            AxisIndex { terms: core::iter::once(AxisTerm::projection(1)).collect(), offset: 0 },
            window_axis(2, 4, spec.stride_h, spec.dilation_h),
            window_axis(3, 5, spec.stride_w, spec.dilation_w),
        ],
    };
    let stamp = append(
        program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![spec.out_h, spec.out_w, spec.kernel_h, spec.kernel_w].iter().map(|&extent| Extent::Static(extent as u32)).collect(),
            value: 1.0,
        },
    );
    let stamp_pattern = projection(6, &[2, 3, 4, 5]);
    let windowed = build_elementwise(
        program,
        ScalarOp::Multiply,
        alloc::vec![(image.node, IndexMap::Affine(image_pattern)), (stamp, IndexMap::Affine(stamp_pattern))],
    );
    let shape = alloc::vec![image.shape[0], image.shape[1], spec.out_h, spec.out_w, spec.kernel_h, spec.kernel_w];
    Value { node: windowed, shape, view: None }
}

/// Rank-3 (`[n, c, w]`) analogue of [`window_materialize`]: one spatial axis
/// instead of two, materializing a rank-4 `[n, c, out_w, kw]` tensor via the
/// same two-term `window_axis` + all-ones stamp technique. `Conv1d`'s whole
/// reason for existing as its own function rather than a generalized-rank
/// [`window_materialize`]: the iteration-space axis positions (`2,3` here vs
/// `2,3,4,5` for 2D) are fixed per rank, not data, so genericizing over rank
/// would trade a second small function for a rank-indexed axis-layout table
/// -- no simpler, and this crate's own convention (`lower_gather` aside,
/// which *is* genuinely rank-generic) is one function per fixed spatial rank.
fn window_materialize1d(program: &mut Vec<Op>, image: &Value, out_w: u64, kernel_w: u64, stride_w: i64, dilation_w: i64) -> Value {
    let image_pattern = IndexPattern {
        iter_rank: 4,
        axes: alloc::vec![
            AxisIndex { terms: core::iter::once(AxisTerm::projection(0)).collect(), offset: 0 },
            AxisIndex { terms: core::iter::once(AxisTerm::projection(1)).collect(), offset: 0 },
            window_axis(2, 3, stride_w, dilation_w),
        ],
    };
    let stamp = append(
        program,
        Op::Constant { dtype: DType::Float32, shape: alloc::vec![out_w, kernel_w].iter().map(|&extent| Extent::Static(extent as u32)).collect(), value: 1.0 },
    );
    let stamp_pattern = projection(4, &[2, 3]);
    let windowed = build_elementwise(
        program,
        ScalarOp::Multiply,
        alloc::vec![(image.node, IndexMap::Affine(image_pattern)), (stamp, IndexMap::Affine(stamp_pattern))],
    );
    let shape = alloc::vec![image.shape[0], image.shape[1], out_w, kernel_w];
    Value { node: windowed, shape, view: None }
}

/// Rank-5 (`[n, c, d, h, w]`) analogue of [`window_materialize`]: three
/// spatial axes instead of two, materializing a rank-8 `[n, c, od, oh, ow,
/// kd, kh, kw]` tensor via the same two-term `window_axis` + all-ones stamp
/// technique -- the window machinery is axis-generic, this is the third
/// (and, past rank 5, the pattern this crate's own convention would repeat
/// again) fixed-spatial-rank instantiation alongside [`window_materialize1d`]
/// and [`window_materialize`].
#[allow(clippy::too_many_arguments)]
fn window_materialize3d(
    program: &mut Vec<Op>,
    image: &Value,
    out_d: u64,
    out_h: u64,
    out_w: u64,
    kernel_d: u64,
    kernel_h: u64,
    kernel_w: u64,
    stride_d: i64,
    stride_h: i64,
    stride_w: i64,
    dilation_d: i64,
    dilation_h: i64,
    dilation_w: i64,
) -> Value {
    let image_pattern = IndexPattern {
        iter_rank: 8,
        axes: alloc::vec![
            AxisIndex { terms: core::iter::once(AxisTerm::projection(0)).collect(), offset: 0 },
            AxisIndex { terms: core::iter::once(AxisTerm::projection(1)).collect(), offset: 0 },
            window_axis(2, 5, stride_d, dilation_d),
            window_axis(3, 6, stride_h, dilation_h),
            window_axis(4, 7, stride_w, dilation_w),
        ],
    };
    let stamp = append(
        program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![out_d, out_h, out_w, kernel_d, kernel_h, kernel_w].iter().map(|&extent| Extent::Static(extent as u32)).collect(),
            value: 1.0,
        },
    );
    let stamp_pattern = projection(8, &[2, 3, 4, 5, 6, 7]);
    let windowed = build_elementwise(
        program,
        ScalarOp::Multiply,
        alloc::vec![(image.node, IndexMap::Affine(image_pattern)), (stamp, IndexMap::Affine(stamp_pattern))],
    );
    let shape = alloc::vec![image.shape[0], image.shape[1], out_d, out_h, out_w, kernel_d, kernel_h, kernel_w];
    Value { node: windowed, shape, view: None }
}

/// `Conv`, rank-3 `[n, ci, w]` image and rank-3 `[co, ci, kw]` weight
/// (`group=1` only -- see [`lower_conv`]'s own doc for why the 2D path
/// supports `group`; the same static channel-slice + [`concat_pair`]
/// decomposition applies here unchanged and is deferred only for lack of
/// time, not a boundary). Otherwise the exact rank-3 mirror of
/// [`conv2d_core`]: [`pad_axis`] on the one spatial axis,
/// [`window_materialize1d`], `Elementwise(Multiply)` against the weight,
/// `Reduce(Add)` over `(ci, kw)`.
fn conv1d_core(program: &mut Vec<Op>, node: &NodeProto<'_>, image: &Value, weight: &Value, bias: Option<&Value>, attrs: Conv2dAttrs) -> Result<Value, LowerError> {
    let name = node.name.to_string();
    let op_type = node.op_type.to_string();
    let batch = image.shape[0];
    let out_channels = weight.shape[0];
    let kernel_w = weight.shape[2];

    let padded = pad_axis(program, image, 2, attrs.pad_left as u64, attrs.pad_right as u64, 0.0);
    let out_w = conv_output_extent(padded.shape[2], kernel_w, attrs.stride_w, attrs.dilation_w)
        .ok_or_else(|| LowerError::UnsupportedShape { name: name.clone(), op_type: op_type.clone(), reason: "Conv1d kernel does not fit the padded input width".to_string() })?;

    let windowed = window_materialize1d(program, &padded, out_w, kernel_w, attrs.stride_w, attrs.dilation_w);

    // shared iteration space: 0=n 1=co 2=ow 3=ci 4=kw
    let windowed_pattern = projection(5, &[0, 3, 2, 4]);
    let weight_pattern = projection(5, &[1, 3, 4]);
    let product = build_elementwise(
        program,
        ScalarOp::Multiply,
        alloc::vec![(windowed.node, IndexMap::Affine(windowed_pattern)), (weight.node, IndexMap::Affine(weight_pattern))],
    );

    let out_shape = alloc::vec![batch, out_channels, out_w];
    let reduced = build_reduce(program, ScalarOp::Add, ReduceInit::Zero, product, identity_pattern(5), projection(5, &[0, 1, 2]), Some("conv1d".to_string()));

    let result = match bias {
        Some(bias) => {
            if bias.shape != alloc::vec![out_channels] {
                return Err(LowerError::UnsupportedShape { name, op_type, reason: "Conv bias must be a rank-1 tensor sized to out_channels".to_string() });
            }
            build_elementwise(
                program,
                ScalarOp::Add,
                alloc::vec![(reduced, IndexMap::Affine(identity_pattern(3))), (bias.node, IndexMap::Affine(projection(3, &[1])))],
            )
        }
        None => reduced,
    };
    Ok(Value { node: result, shape: out_shape, view: None })
}

/// The `strides`/`dilations`/`pads` 2D `Conv` attribute triple, parsed once
/// and shared by [`conv2d_core`] and every group slice [`lower_conv`] builds.
#[derive(Debug, Clone, Copy)]
struct Conv2dAttrs {
    stride_h: i64,
    stride_w: i64,
    dilation_h: i64,
    dilation_w: i64,
    pad_top: i64,
    pad_left: i64,
    pad_bottom: i64,
    pad_right: i64,
}

/// The `SAME_UPPER`/`SAME_LOWER` pad split for one spatial axis, ONNX's own
/// formula: enough total padding that `out = ceil(in / stride)`, split so
/// the larger half lands after the axis for `SAME_UPPER` and before it for
/// `SAME_LOWER`. Feeds straight into the same `pad_axis`/`conv_output_extent`
/// pair the explicit `pads` attribute already drives -- no new padding
/// machinery, only where the two numbers come from.
fn same_pad_axis(input_extent: u64, kernel_extent: u64, stride: i64, dilation: i64, lower: bool) -> (i64, i64) {
    let stride = stride as u64;
    let output_extent = input_extent.div_ceil(stride).max(1);
    let span = (dilation as u64) * kernel_extent.saturating_sub(1) + 1;
    let needed = ((output_extent - 1) * stride + span).saturating_sub(input_extent);
    let small = needed / 2;
    let large = needed - small;
    if lower { (large as i64, small as i64) } else { (small as i64, large as i64) }
}

/// Resolves `strides`/`dilations`/`pads` for a 2D `Conv`/`ConvTranspose`
/// window, honoring `auto_pad` (`NOTSET` reads the explicit `pads`
/// attribute unchanged; `VALID` means zero padding; `SAME_UPPER`/
/// `SAME_LOWER` compute it via [`same_pad_axis`] from the image and kernel
/// spatial extents) -- the one place every 2D window op resolves padding, so
/// `auto_pad` support lands once for `Conv`, `MaxPool`, and `AveragePool`.
fn parse_conv2d_attrs(node: &NodeProto<'_>, image_h: u64, image_w: u64, kernel_h: u64, kernel_w: u64) -> Result<Conv2dAttrs, LowerError> {
    let name = node.name.to_string();
    let op_type = node.op_type.to_string();
    let strides = attr_ints_or(node, "strides", &[1, 1]);
    let dilations = attr_ints_or(node, "dilations", &[1, 1]);
    if strides.len() != 2 || dilations.len() != 2 {
        return Err(LowerError::UnsupportedShape { name, op_type, reason: "Conv lowering supports 2D strides/dilations only".to_string() });
    }
    let (stride_h, stride_w) = (strides[0], strides[1]);
    let (dilation_h, dilation_w) = (dilations[0], dilations[1]);

    let auto_pad = attr_str(node, "auto_pad").unwrap_or(b"NOTSET");
    let (pad_top, pad_bottom, pad_left, pad_right) = match auto_pad {
        b"NOTSET" => {
            let pads = attr_ints_or(node, "pads", &[0, 0, 0, 0]);
            if pads.len() != 4 {
                return Err(LowerError::UnsupportedShape { name, op_type, reason: "Conv lowering supports 2D pads only".to_string() });
            }
            (pads[0], pads[2], pads[1], pads[3])
        }
        b"VALID" => (0, 0, 0, 0),
        b"SAME_UPPER" | b"SAME_LOWER" => {
            let lower = auto_pad == b"SAME_LOWER";
            let (top, bottom) = same_pad_axis(image_h, kernel_h, stride_h, dilation_h, lower);
            let (left, right) = same_pad_axis(image_w, kernel_w, stride_w, dilation_w, lower);
            (top, bottom, left, right)
        }
        _ => return Err(LowerError::UnsupportedShape { name, op_type, reason: "Conv lowering supports auto_pad NOTSET/VALID/SAME_UPPER/SAME_LOWER only".to_string() }),
    };
    Ok(Conv2dAttrs { stride_h, stride_w, dilation_h, dilation_w, pad_top, pad_left, pad_bottom, pad_right })
}

/// [`Conv1dAttrs::stride_w`]/etc. parsed from a `Conv1d` node's
/// `strides`/`dilations`/`pads`, reusing [`Conv2dAttrs`]'s field names so
/// [`conv1d_core`] shares its parameter shape with [`conv2d_core`] -- the
/// unused `_h` fields are never read on the rank-3 path.
fn parse_conv1d_attrs(node: &NodeProto<'_>, image_w: u64, kernel_w: u64) -> Result<Conv2dAttrs, LowerError> {
    let name = node.name.to_string();
    let op_type = node.op_type.to_string();
    let strides = attr_ints_or(node, "strides", &[1]);
    let dilations = attr_ints_or(node, "dilations", &[1]);
    if strides.len() != 1 || dilations.len() != 1 {
        return Err(LowerError::UnsupportedShape { name, op_type, reason: "Conv1d lowering supports 1D strides/dilations only".to_string() });
    }
    let (stride_w, dilation_w) = (strides[0], dilations[0]);

    let auto_pad = attr_str(node, "auto_pad").unwrap_or(b"NOTSET");
    let (pad_left, pad_right) = match auto_pad {
        b"NOTSET" => {
            let pads = attr_ints_or(node, "pads", &[0, 0]);
            if pads.len() != 2 {
                return Err(LowerError::UnsupportedShape { name, op_type, reason: "Conv1d lowering supports 1D pads only".to_string() });
            }
            (pads[0], pads[1])
        }
        b"VALID" => (0, 0),
        b"SAME_UPPER" | b"SAME_LOWER" => same_pad_axis(image_w, kernel_w, stride_w, dilation_w, auto_pad == b"SAME_LOWER"),
        _ => return Err(LowerError::UnsupportedShape { name, op_type, reason: "Conv1d lowering supports auto_pad NOTSET/VALID/SAME_UPPER/SAME_LOWER only".to_string() }),
    };
    Ok(Conv2dAttrs { stride_h: 1, stride_w, dilation_h: 1, dilation_w, pad_top: 0, pad_left, pad_bottom: 0, pad_right })
}

/// `Conv`, rank-3 (`group >= 1`): parses attrs, validates channels, and
/// calls [`conv1d_core`] directly for `group=1`, or once per group over a
/// static channel slice ([`slice_axis_range`]) for `group > 1`, concatenating
/// the per-group outputs back along the output-channel axis
/// ([`concat_pair`]) -- the exact rank-3 mirror of [`lower_conv`]'s
/// `group != 1` decomposition (see that function's own doc for why this is
/// the RISC-correct resolution rather than a fused `IndexMap`).
fn lower_conv1d(program: &mut Vec<Op>, values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>) -> Result<(), LowerError> {
    let image = lookup(values, node, 0)?.clone();
    let weight = lookup(values, node, 1)?.clone();
    let name = node.name.to_string();
    let op_type = node.op_type.to_string();

    let group = attr_int(node, "group").unwrap_or(1);
    if group < 1 {
        return Err(LowerError::UnsupportedShape { name, op_type, reason: "Conv1d group attribute must be >= 1".to_string() });
    }
    let group = group as u64;

    let attrs = parse_conv1d_attrs(node, image.shape[2], weight.shape[2])?;
    let bias = match node.input.get(2) {
        Some(bias_name) => Some(lookup_by_name(values, bias_name, node.op_type, node.name)?.clone()),
        None => None,
    };

    if group == 1 {
        let in_channels = image.shape[1];
        if weight.shape[1] != in_channels {
            return Err(LowerError::UnsupportedShape {
                name: node.name.to_string(),
                op_type: node.op_type.to_string(),
                reason: format!("Conv1d weight in-channels {} does not match image channels {in_channels}", weight.shape[1]),
            });
        }
        let result = conv1d_core(program, node, &image, &weight, bias.as_ref(), attrs)?;
        bind_output(values, node, 0, result.node, result.shape);
        return Ok(());
    }

    let in_channels = image.shape[1];
    let total_out_channels = weight.shape[0];
    let weight_in_channels = weight.shape[1];
    if in_channels % group != 0 || total_out_channels % group != 0 {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: format!("Conv1d image channels {in_channels} and weight output channels {total_out_channels} must both be evenly divisible by group {group}"),
        });
    }
    let in_channels_per_group = in_channels / group;
    let out_channels_per_group = total_out_channels / group;
    if weight_in_channels != in_channels_per_group {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: format!("Conv1d weight in-channels {weight_in_channels} does not match image channels {in_channels} / group {group}"),
        });
    }
    if let Some(bias) = &bias
        && bias.shape != alloc::vec![total_out_channels]
    {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: "Conv1d bias must be a rank-1 tensor sized to the total (grouped) output channels".to_string(),
        });
    }

    let mut accumulator: Option<Value> = None;
    for group_index in 0..group {
        let image_slice = slice_axis_range(program, &image, 1, group_index * in_channels_per_group, in_channels_per_group);
        let weight_slice = slice_axis_range(program, &weight, 0, group_index * out_channels_per_group, out_channels_per_group);
        let bias_slice = bias.as_ref().map(|bias| slice_axis_range(program, bias, 0, group_index * out_channels_per_group, out_channels_per_group));
        let group_result = conv1d_core(program, node, &image_slice, &weight_slice, bias_slice.as_ref(), attrs)?;
        accumulator = Some(match accumulator {
            None => group_result,
            Some(previous) => concat_pair(program, node, &previous, &group_result, 1)?,
        });
    }
    let Some(result) = accumulator else {
        return Err(LowerError::UnsupportedShape { name: node.name.to_string(), op_type: node.op_type.to_string(), reason: "Conv1d group must be >= 1".to_string() });
    };
    bind_output(values, node, 0, result.node, result.shape);
    Ok(())
}

/// The `group=1` `Conv` core: `Reduce(Add)` over `Elementwise(Multiply)` of
/// the materialized window against the kernel weights -- `specs/conv2d.toml`'s
/// composition (see that file's own doc), with the weight tensor left at its
/// natural ONNX `[co, ci, kh, kw]` shape (a pure per-axis projection, no
/// broadcast-to-every-output-position replication) since [`window_materialize`]
/// already supplies the pure `oy`/`ox` projections `shape::infer` needs.
/// [`lower_conv`] calls this once directly for `group=1`, or once per group
/// (each group's slice already matching this shape) for `group != 1`.
fn conv2d_core(
    program: &mut Vec<Op>,
    node: &NodeProto<'_>,
    image: &Value,
    weight: &Value,
    bias: Option<&Value>,
    attrs: Conv2dAttrs,
    op_name: Option<String>,
) -> Result<Value, LowerError> {
    let name = node.name.to_string();
    let op_type = node.op_type.to_string();
    let batch = image.shape[0];
    let out_channels = weight.shape[0];
    let kernel_h = weight.shape[2];
    let kernel_w = weight.shape[3];

    let padded_w = pad_axis(program, image, 3, attrs.pad_left as u64, attrs.pad_right as u64, 0.0);
    let padded = pad_axis(program, &padded_w, 2, attrs.pad_top as u64, attrs.pad_bottom as u64, 0.0);

    let out_h = conv_output_extent(padded.shape[2], kernel_h, attrs.stride_h, attrs.dilation_h)
        .ok_or_else(|| LowerError::UnsupportedShape { name: name.clone(), op_type: op_type.clone(), reason: "Conv kernel does not fit the padded image height".to_string() })?;
    let out_w = conv_output_extent(padded.shape[3], kernel_w, attrs.stride_w, attrs.dilation_w)
        .ok_or_else(|| LowerError::UnsupportedShape { name: name.clone(), op_type: op_type.clone(), reason: "Conv kernel does not fit the padded image width".to_string() })?;

    let windowed = window_materialize(
        program,
        &padded,
        WindowSpec { out_h, out_w, kernel_h, kernel_w, stride_h: attrs.stride_h, stride_w: attrs.stride_w, dilation_h: attrs.dilation_h, dilation_w: attrs.dilation_w },
    );

    // shared iteration space: 0=n 1=co 2=oy 3=ox 4=ci 5=ky 6=kx
    let windowed_pattern = projection(7, &[0, 4, 2, 3, 5, 6]);
    let weight_pattern = projection(7, &[1, 4, 5, 6]);
    let product = build_elementwise(
        program,
        ScalarOp::Multiply,
        alloc::vec![(windowed.node, IndexMap::Affine(windowed_pattern)), (weight.node, IndexMap::Affine(weight_pattern))],
    );

    let out_shape = alloc::vec![batch, out_channels, out_h, out_w];
    let reduced = build_reduce(program, ScalarOp::Add, ReduceInit::Zero, product, identity_pattern(7), projection(7, &[0, 1, 2, 3]), op_name);

    let result = match bias {
        Some(bias) => {
            if bias.shape != alloc::vec![out_channels] {
                return Err(LowerError::UnsupportedShape { name, op_type, reason: "Conv bias must be a rank-1 tensor sized to out_channels".to_string() });
            }
            build_elementwise(
                program,
                ScalarOp::Add,
                alloc::vec![(reduced, IndexMap::Affine(identity_pattern(4))), (bias.node, IndexMap::Affine(projection(4, &[1])))],
            )
        }
        None => reduced,
    };
    Ok(Value { node: result, shape: out_shape, view: None })
}

/// The `strides`/`dilations`/`pads` 3D `Conv`/pooling attribute sextuple --
/// [`Conv2dAttrs`]'s three-spatial-axis mirror.
#[derive(Debug, Clone, Copy)]
struct Conv3dAttrs {
    stride_d: i64,
    stride_h: i64,
    stride_w: i64,
    dilation_d: i64,
    dilation_h: i64,
    dilation_w: i64,
    pad_d0: i64,
    pad_h0: i64,
    pad_w0: i64,
    pad_d1: i64,
    pad_h1: i64,
    pad_w1: i64,
}

/// [`parse_conv2d_attrs`]'s rank-5 mirror: same `auto_pad`
/// (`NOTSET`/`VALID`/`SAME_UPPER`/`SAME_LOWER`) resolution via
/// [`same_pad_axis`], three spatial axes instead of two. ONNX's `pads`
/// attribute for a 3D op is `[d0, h0, w0, d1, h1, w1]` (all "begin" axes,
/// then all "end" axes).
fn parse_conv3d_attrs(node: &NodeProto<'_>, image_d: u64, image_h: u64, image_w: u64, kernel_d: u64, kernel_h: u64, kernel_w: u64) -> Result<Conv3dAttrs, LowerError> {
    let name = node.name.to_string();
    let op_type = node.op_type.to_string();
    let strides = attr_ints_or(node, "strides", &[1, 1, 1]);
    let dilations = attr_ints_or(node, "dilations", &[1, 1, 1]);
    if strides.len() != 3 || dilations.len() != 3 {
        return Err(LowerError::UnsupportedShape { name, op_type, reason: "Conv lowering supports 3D strides/dilations only".to_string() });
    }
    let (stride_d, stride_h, stride_w) = (strides[0], strides[1], strides[2]);
    let (dilation_d, dilation_h, dilation_w) = (dilations[0], dilations[1], dilations[2]);

    let auto_pad = attr_str(node, "auto_pad").unwrap_or(b"NOTSET");
    let (pad_d0, pad_d1, pad_h0, pad_h1, pad_w0, pad_w1) = match auto_pad {
        b"NOTSET" => {
            let pads = attr_ints_or(node, "pads", &[0, 0, 0, 0, 0, 0]);
            if pads.len() != 6 {
                return Err(LowerError::UnsupportedShape { name, op_type, reason: "Conv lowering supports 3D pads only".to_string() });
            }
            (pads[0], pads[3], pads[1], pads[4], pads[2], pads[5])
        }
        b"VALID" => (0, 0, 0, 0, 0, 0),
        b"SAME_UPPER" | b"SAME_LOWER" => {
            let lower = auto_pad == b"SAME_LOWER";
            let (d0, d1) = same_pad_axis(image_d, kernel_d, stride_d, dilation_d, lower);
            let (h0, h1) = same_pad_axis(image_h, kernel_h, stride_h, dilation_h, lower);
            let (w0, w1) = same_pad_axis(image_w, kernel_w, stride_w, dilation_w, lower);
            (d0, d1, h0, h1, w0, w1)
        }
        _ => return Err(LowerError::UnsupportedShape { name, op_type, reason: "Conv lowering supports auto_pad NOTSET/VALID/SAME_UPPER/SAME_LOWER only".to_string() }),
    };
    Ok(Conv3dAttrs { stride_d, stride_h, stride_w, dilation_d, dilation_h, dilation_w, pad_d0, pad_h0, pad_w0, pad_d1, pad_h1, pad_w1 })
}

/// `Conv`, rank-5 `[n, ci, d, h, w]` image and rank-5 `[co, ci, kd, kh, kw]`
/// weight (`group=1` only -- the same static channel-slice + [`concat_pair`]
/// decomposition [`lower_conv`] uses for 2D applies here unchanged and is
/// deferred only for lack of time, not a boundary). The exact rank-5 mirror
/// of [`conv2d_core`]: [`pad_axis`] on each of the three spatial axes,
/// [`window_materialize3d`], `Elementwise(Multiply)` against the weight,
/// `Reduce(Add)` over `(ci, kd, kh, kw)`.
fn conv3d_core(program: &mut Vec<Op>, node: &NodeProto<'_>, image: &Value, weight: &Value, bias: Option<&Value>, attrs: Conv3dAttrs) -> Result<Value, LowerError> {
    let name = node.name.to_string();
    let op_type = node.op_type.to_string();
    let batch = image.shape[0];
    let out_channels = weight.shape[0];
    let (kernel_d, kernel_h, kernel_w) = (weight.shape[2], weight.shape[3], weight.shape[4]);

    let padded_w = pad_axis(program, image, 4, attrs.pad_w0 as u64, attrs.pad_w1 as u64, 0.0);
    let padded_h = pad_axis(program, &padded_w, 3, attrs.pad_h0 as u64, attrs.pad_h1 as u64, 0.0);
    let padded = pad_axis(program, &padded_h, 2, attrs.pad_d0 as u64, attrs.pad_d1 as u64, 0.0);

    let out_d = conv_output_extent(padded.shape[2], kernel_d, attrs.stride_d, attrs.dilation_d)
        .ok_or_else(|| LowerError::UnsupportedShape { name: name.clone(), op_type: op_type.clone(), reason: "Conv kernel does not fit the padded image depth".to_string() })?;
    let out_h = conv_output_extent(padded.shape[3], kernel_h, attrs.stride_h, attrs.dilation_h)
        .ok_or_else(|| LowerError::UnsupportedShape { name: name.clone(), op_type: op_type.clone(), reason: "Conv kernel does not fit the padded image height".to_string() })?;
    let out_w = conv_output_extent(padded.shape[4], kernel_w, attrs.stride_w, attrs.dilation_w)
        .ok_or_else(|| LowerError::UnsupportedShape { name: name.clone(), op_type: op_type.clone(), reason: "Conv kernel does not fit the padded image width".to_string() })?;

    let windowed = window_materialize3d(
        program,
        &padded,
        out_d,
        out_h,
        out_w,
        kernel_d,
        kernel_h,
        kernel_w,
        attrs.stride_d,
        attrs.stride_h,
        attrs.stride_w,
        attrs.dilation_d,
        attrs.dilation_h,
        attrs.dilation_w,
    );

    // shared iteration space: 0=n 1=co 2=od 3=oh 4=ow 5=ci 6=kd 7=kh 8=kw
    let windowed_pattern = projection(9, &[0, 5, 2, 3, 4, 6, 7, 8]);
    let weight_pattern = projection(9, &[1, 5, 6, 7, 8]);
    let product = build_elementwise(
        program,
        ScalarOp::Multiply,
        alloc::vec![(windowed.node, IndexMap::Affine(windowed_pattern)), (weight.node, IndexMap::Affine(weight_pattern))],
    );

    let out_shape = alloc::vec![batch, out_channels, out_d, out_h, out_w];
    let reduced = build_reduce(program, ScalarOp::Add, ReduceInit::Zero, product, identity_pattern(9), projection(9, &[0, 1, 2, 3, 4]), Some("conv3d".to_string()));

    let result = match bias {
        Some(bias) => {
            if bias.shape != alloc::vec![out_channels] {
                return Err(LowerError::UnsupportedShape { name, op_type, reason: "Conv bias must be a rank-1 tensor sized to out_channels".to_string() });
            }
            build_elementwise(
                program,
                ScalarOp::Add,
                alloc::vec![(reduced, IndexMap::Affine(identity_pattern(5))), (bias.node, IndexMap::Affine(projection(5, &[1])))],
            )
        }
        None => reduced,
    };
    Ok(Value { node: result, shape: out_shape, view: None })
}

/// `Conv`, rank-5 (`group=1` only): parses attrs, validates channels, and
/// calls [`conv3d_core`] -- [`lower_conv1d`]'s rank-5 sibling, called from
/// [`lower_conv`]'s rank dispatch.
fn lower_conv3d(program: &mut Vec<Op>, values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>) -> Result<(), LowerError> {
    let image = lookup(values, node, 0)?.clone();
    let weight = lookup(values, node, 1)?.clone();
    let name = node.name.to_string();
    let op_type = node.op_type.to_string();

    let group = attr_int(node, "group").unwrap_or(1);
    if group != 1 {
        return Err(LowerError::UnsupportedShape { name, op_type, reason: "Conv3d lowering supports group=1 only (grouped Conv3d deferred)".to_string() });
    }

    let in_channels = image.shape[1];
    if weight.shape[1] != in_channels {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: format!("Conv3d weight in-channels {} does not match image channels {in_channels}", weight.shape[1]),
        });
    }

    let attrs = parse_conv3d_attrs(node, image.shape[2], image.shape[3], image.shape[4], weight.shape[2], weight.shape[3], weight.shape[4])?;
    let bias = match node.input.get(2) {
        Some(bias_name) => Some(lookup_by_name(values, bias_name, node.op_type, node.name)?.clone()),
        None => None,
    };
    let result = conv3d_core(program, node, &image, &weight, bias.as_ref(), attrs)?;
    bind_output(values, node, 0, result.node, result.shape);
    Ok(())
}

/// `Conv`, `group >= 1`: `group=1` calls [`conv2d_core`] directly; `group > 1`
/// decomposes into `group` independent [`conv2d_core`] calls, each over a
/// static, in-bounds channel *slice* ([`slice_axis_range`]) of the image
/// (input channels `[g*(Ci/G), (g+1)*(Ci/G))`) and the weight/bias (output
/// channels `[g*(Co/G), (g+1)*(Co/G))`), then [`concat_pair`]s the `group`
/// per-group outputs back along the output-channel axis.
///
/// This is the RISC-correct resolution, not a workaround: a single fused
/// `IndexMap` for grouped conv would need `co / (Co/G)` to pick each output
/// channel's input-channel range, and floor-division has no [`AxisTerm`]
/// (affine sums coefficients, never divides) -- the same limit that keeps
/// reshape-merge a layout op rather than an `IndexMap`. `group` is a
/// *static* extent, so the loop below is a lower-time unroll (`group` more
/// `Op`s in `program`), never a runtime branch -- no div/mod anywhere, no new
/// `Op`/`ScalarOp`/`IndexMap` variant.
fn lower_conv(program: &mut Vec<Op>, values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>) -> Result<(), LowerError> {
    let image = lookup(values, node, 0)?.clone();
    let weight = lookup(values, node, 1)?.clone();
    let name = node.name.to_string();
    let op_type = node.op_type.to_string();

    if image.shape.len() == 3 && weight.shape.len() == 3 {
        return lower_conv1d(program, values, node);
    }
    if image.shape.len() == 5 && weight.shape.len() == 5 {
        return lower_conv3d(program, values, node);
    }

    if image.shape.len() != 4 || weight.shape.len() != 4 {
        return Err(LowerError::UnsupportedShape {
            name,
            op_type,
            reason: "Conv lowering supports 1D convolution (rank-3 NCW image/CoCiKw weight), 2D convolution (rank-4 NCHW image, rank-4 CoCiKhKw weight), or 3D convolution (rank-5 NCDHW image, rank-5 CoCiKdKhKw weight) only".to_string(),
        });
    }
    let group = attr_int(node, "group").unwrap_or(1);
    if group < 1 {
        return Err(LowerError::UnsupportedShape { name, op_type, reason: "Conv group attribute must be >= 1".to_string() });
    }
    let group = group as u64;

    let attrs = parse_conv2d_attrs(node, image.shape[2], image.shape[3], weight.shape[2], weight.shape[3])?;
    let batch = image.shape[0];
    let in_channels = image.shape[1];
    let total_out_channels = weight.shape[0];
    let weight_in_channels = weight.shape[1];

    let bias = match node.input.get(2) {
        Some(bias_name) => Some(lookup_by_name(values, bias_name, node.op_type, node.name)?.clone()),
        None => None,
    };

    if group == 1 {
        if weight_in_channels != in_channels {
            return Err(LowerError::UnsupportedShape {
                name: node.name.to_string(),
                op_type: node.op_type.to_string(),
                reason: format!("Conv weight in-channels {weight_in_channels} does not match image channels {in_channels}"),
            });
        }
        let result = conv2d_core(program, node, &image, &weight, bias.as_ref(), attrs, Some("conv2d".to_string()))?;
        bind_output(values, node, 0, result.node, result.shape);
        return Ok(());
    }

    if in_channels % group != 0 || total_out_channels % group != 0 || batch == 0 {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: format!("Conv image channels {in_channels} and weight output channels {total_out_channels} must both be evenly divisible by group {group}"),
        });
    }
    let in_channels_per_group = in_channels / group;
    let out_channels_per_group = total_out_channels / group;
    if weight_in_channels != in_channels_per_group {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: format!("Conv weight in-channels {weight_in_channels} does not match image channels {in_channels} / group {group}"),
        });
    }
    if let Some(bias) = &bias
        && bias.shape != alloc::vec![total_out_channels]
    {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: "Conv bias must be a rank-1 tensor sized to the total (grouped) output channels".to_string(),
        });
    }

    let mut accumulator: Option<Value> = None;
    for group_index in 0..group {
        let image_slice = slice_axis_range(program, &image, 1, group_index * in_channels_per_group, in_channels_per_group);
        let weight_slice = slice_axis_range(program, &weight, 0, group_index * out_channels_per_group, out_channels_per_group);
        let bias_slice = bias.as_ref().map(|bias| slice_axis_range(program, bias, 0, group_index * out_channels_per_group, out_channels_per_group));
        let group_result = conv2d_core(program, node, &image_slice, &weight_slice, bias_slice.as_ref(), attrs, None)?;
        accumulator = Some(match accumulator {
            None => group_result,
            Some(previous) => concat_pair(program, node, &previous, &group_result, 1)?,
        });
    }
    let Some(result) = accumulator else {
        return Err(LowerError::UnsupportedShape { name: node.name.to_string(), op_type: node.op_type.to_string(), reason: "Conv group must be >= 1".to_string() });
    };
    bind_output(values, node, 0, result.node, result.shape);
    Ok(())
}

/// The `Conv`/`MaxPool`/`AveragePool`-shared plan: parse `kernel_shape`/
/// `strides`/`dilations`/`pads`, pad the input, and derive the pooled output
/// shape -- the same [`pad_axis`]/[`conv_output_extent`] steps [`lower_conv`]
/// runs, minus the weight operand pooling has none of.
struct PoolPlan {
    padded: Value,
    kernel_h: u64,
    kernel_w: u64,
    stride_h: i64,
    stride_w: i64,
    dilation_h: i64,
    dilation_w: i64,
    out_shape: Vec<u64>,
}

fn plan_pool(program: &mut Vec<Op>, node: &NodeProto<'_>, image: &Value, fill: f32) -> Result<PoolPlan, LowerError> {
    let name = node.name.to_string();
    let op_type = node.op_type.to_string();
    if image.shape.len() != 4 {
        return Err(LowerError::UnsupportedShape { name, op_type, reason: format!("{} lowering supports rank-4 NCHW input only", node.op_type) });
    }
    let kernel_shape = attr_ints(node, "kernel_shape").ok_or_else(|| LowerError::UnsupportedShape {
        name: node.name.to_string(),
        op_type: node.op_type.to_string(),
        reason: format!("{} requires a kernel_shape attribute", node.op_type),
    })?;
    if kernel_shape.len() != 2 {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: format!("{} lowering supports 2D pooling only", node.op_type),
        });
    }
    let (kernel_h, kernel_w) = (kernel_shape[0] as u64, kernel_shape[1] as u64);
    let strides = attr_ints_or(node, "strides", &[1, 1]);
    let dilations = attr_ints_or(node, "dilations", &[1, 1]);
    if strides.len() != 2 || dilations.len() != 2 {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: format!("{} lowering supports 2D strides/dilations only", node.op_type),
        });
    }
    let (stride_h, stride_w) = (strides[0], strides[1]);
    let (dilation_h, dilation_w) = (dilations[0], dilations[1]);

    let auto_pad = attr_str(node, "auto_pad").unwrap_or(b"NOTSET");
    let (pad_top, pad_bottom, pad_left, pad_right) = match auto_pad {
        b"NOTSET" => {
            let pads = attr_ints_or(node, "pads", &[0, 0, 0, 0]);
            if pads.len() != 4 {
                return Err(LowerError::UnsupportedShape { name, op_type, reason: format!("{} lowering supports 2D pads only", node.op_type) });
            }
            (pads[0], pads[2], pads[1], pads[3])
        }
        b"VALID" => (0, 0, 0, 0),
        b"SAME_UPPER" | b"SAME_LOWER" => {
            let lower = auto_pad == b"SAME_LOWER";
            let (top, bottom) = same_pad_axis(image.shape[2], kernel_h, stride_h, dilation_h, lower);
            let (left, right) = same_pad_axis(image.shape[3], kernel_w, stride_w, dilation_w, lower);
            (top, bottom, left, right)
        }
        _ => {
            return Err(LowerError::UnsupportedShape {
                name,
                op_type,
                reason: format!("{} lowering supports auto_pad NOTSET/VALID/SAME_UPPER/SAME_LOWER only", node.op_type),
            });
        }
    };

    let padded_w = pad_axis(program, image, 3, pad_left as u64, pad_right as u64, fill);
    let padded = pad_axis(program, &padded_w, 2, pad_top as u64, pad_bottom as u64, fill);

    let out_h = conv_output_extent(padded.shape[2], kernel_h, stride_h, dilation_h).ok_or_else(|| LowerError::UnsupportedShape {
        name: node.name.to_string(),
        op_type: node.op_type.to_string(),
        reason: format!("{} kernel does not fit the padded input height", node.op_type),
    })?;
    let out_w = conv_output_extent(padded.shape[3], kernel_w, stride_w, dilation_w).ok_or_else(|| LowerError::UnsupportedShape {
        name: node.name.to_string(),
        op_type: node.op_type.to_string(),
        reason: format!("{} kernel does not fit the padded input width", node.op_type),
    })?;

    let out_shape = alloc::vec![image.shape[0], image.shape[1], out_h, out_w];
    Ok(PoolPlan { padded, kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w, out_shape })
}

/// `MaxPool`: [`window_materialize`] (padded with `-inf` so a padded cell
/// never wins the max), then `Reduce(Maximum)` over the window's trailing
/// `kh`/`kw` axes -- the same sliding-window shape as [`lower_conv`], with no
/// weight operand and no channel reduction (`ScalarOp::Maximum` is not
/// `ScalarOp::Add`, so `-inf * 1.0` from the stamp multiply staying `-inf`,
/// never `NaN`, is what makes the padding value safe to carry through
/// [`window_materialize`]'s multiply).
///
/// Deferred: rank other than 4 or 5 (1D `MaxPool`), `storage_order`,
/// `ceil_mode`, indices output (`Y` only, never the optional `Indices`).
fn lower_maxpool(program: &mut Vec<Op>, values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>) -> Result<(), LowerError> {
    let image = lookup(values, node, 0)?.clone();
    if image.shape.len() == 5 {
        return lower_maxpool3d(program, values, node);
    }
    let plan = plan_pool(program, node, &image, f32::NEG_INFINITY)?;
    let windowed = window_materialize(
        program,
        &plan.padded,
        WindowSpec {
            out_h: plan.out_shape[2],
            out_w: plan.out_shape[3],
            kernel_h: plan.kernel_h,
            kernel_w: plan.kernel_w,
            stride_h: plan.stride_h,
            stride_w: plan.stride_w,
            dilation_h: plan.dilation_h,
            dilation_w: plan.dilation_w,
        },
    );
    let result = build_reduce(
        program,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        windowed.node,
        identity_pattern(6),
        projection(6, &[0, 1, 2, 3]),
        Some("maxpool2d".to_string()),
    );
    bind_output(values, node, 0, result, plan.out_shape);
    Ok(())
}

/// `AveragePool`: [`window_materialize`] (zero-padded), `Reduce(Add)` over
/// `kh`/`kw`, then `Multiply` by the constant `1/(kh*kw)` -- correct for
/// `count_include_pad=1` and for any window with no padding, where the fixed
/// divisor and ONNX's own default (`count_include_pad=0`, a per-position
/// valid-cell divisor) agree. A padded window with the default
/// `count_include_pad=0` is a named, deferred gap: it needs a per-output-
/// position valid-count divisor this composition does not yet build, not a
/// silently wrong average.
fn lower_averagepool(program: &mut Vec<Op>, values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>) -> Result<(), LowerError> {
    let image = lookup(values, node, 0)?.clone();
    if image.shape.len() == 5 {
        return lower_averagepool3d(program, values, node);
    }
    let count_include_pad = attr_int(node, "count_include_pad").unwrap_or(0);
    let has_nonzero_pad = attr_ints(node, "pads").is_some_and(|pads| pads.iter().any(|&value| value != 0));
    if has_nonzero_pad && count_include_pad == 0 {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: "AveragePool lowering supports nonzero pads only with count_include_pad=1 (the ONNX default excludes padding from the divisor, which needs a per-position valid-count divisor not yet composed)".to_string(),
        });
    }
    let plan = plan_pool(program, node, &image, 0.0)?;
    let windowed = window_materialize(
        program,
        &plan.padded,
        WindowSpec {
            out_h: plan.out_shape[2],
            out_w: plan.out_shape[3],
            kernel_h: plan.kernel_h,
            kernel_w: plan.kernel_w,
            stride_h: plan.stride_h,
            stride_w: plan.stride_w,
            dilation_h: plan.dilation_h,
            dilation_w: plan.dilation_w,
        },
    );
    let summed = build_reduce(
        program,
        ScalarOp::Add,
        ReduceInit::Zero,
        windowed.node,
        identity_pattern(6),
        projection(6, &[0, 1, 2, 3]),
        Some("averagepool2d".to_string()),
    );
    let window_size = (plan.kernel_h * plan.kernel_w) as f32;
    let inverse = constant_scalar(program, 1.0 / window_size);
    let result = build_elementwise(
        program,
        ScalarOp::Multiply,
        alloc::vec![(summed, IndexMap::Affine(identity_pattern(4))), (inverse, IndexMap::Affine(scalar_broadcast_pattern(4)))],
    );
    bind_output(values, node, 0, result, plan.out_shape);
    Ok(())
}

/// [`PoolPlan`]'s rank-5 mirror -- three spatial axes instead of two.
struct PoolPlan3d {
    padded: Value,
    kernel_d: u64,
    kernel_h: u64,
    kernel_w: u64,
    stride_d: i64,
    stride_h: i64,
    stride_w: i64,
    dilation_d: i64,
    dilation_h: i64,
    dilation_w: i64,
    out_shape: Vec<u64>,
}

/// [`plan_pool`]'s rank-5 mirror: same `kernel_shape`/`strides`/
/// `dilations`/`pads`/`auto_pad` resolution (via [`parse_conv3d_attrs`]'s
/// `pads`-ordering convention), three spatial axes instead of two.
fn plan_pool3d(program: &mut Vec<Op>, node: &NodeProto<'_>, image: &Value, fill: f32) -> Result<PoolPlan3d, LowerError> {
    let kernel_shape = attr_ints(node, "kernel_shape").ok_or_else(|| LowerError::UnsupportedShape {
        name: node.name.to_string(),
        op_type: node.op_type.to_string(),
        reason: format!("{} requires a kernel_shape attribute", node.op_type),
    })?;
    if kernel_shape.len() != 3 {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: format!("{} lowering supports 2D or 3D pooling only", node.op_type),
        });
    }
    let (kernel_d, kernel_h, kernel_w) = (kernel_shape[0] as u64, kernel_shape[1] as u64, kernel_shape[2] as u64);

    let attrs = parse_conv3d_attrs(node, image.shape[2], image.shape[3], image.shape[4], kernel_d, kernel_h, kernel_w)?;

    let padded_w = pad_axis(program, image, 4, attrs.pad_w0 as u64, attrs.pad_w1 as u64, fill);
    let padded_h = pad_axis(program, &padded_w, 3, attrs.pad_h0 as u64, attrs.pad_h1 as u64, fill);
    let padded = pad_axis(program, &padded_h, 2, attrs.pad_d0 as u64, attrs.pad_d1 as u64, fill);

    let out_d = conv_output_extent(padded.shape[2], kernel_d, attrs.stride_d, attrs.dilation_d).ok_or_else(|| LowerError::UnsupportedShape {
        name: node.name.to_string(),
        op_type: node.op_type.to_string(),
        reason: format!("{} kernel does not fit the padded input depth", node.op_type),
    })?;
    let out_h = conv_output_extent(padded.shape[3], kernel_h, attrs.stride_h, attrs.dilation_h).ok_or_else(|| LowerError::UnsupportedShape {
        name: node.name.to_string(),
        op_type: node.op_type.to_string(),
        reason: format!("{} kernel does not fit the padded input height", node.op_type),
    })?;
    let out_w = conv_output_extent(padded.shape[4], kernel_w, attrs.stride_w, attrs.dilation_w).ok_or_else(|| LowerError::UnsupportedShape {
        name: node.name.to_string(),
        op_type: node.op_type.to_string(),
        reason: format!("{} kernel does not fit the padded input width", node.op_type),
    })?;

    let out_shape = alloc::vec![image.shape[0], image.shape[1], out_d, out_h, out_w];
    Ok(PoolPlan3d {
        padded,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d: attrs.stride_d,
        stride_h: attrs.stride_h,
        stride_w: attrs.stride_w,
        dilation_d: attrs.dilation_d,
        dilation_h: attrs.dilation_h,
        dilation_w: attrs.dilation_w,
        out_shape,
    })
}

/// `MaxPool`, rank-5 (3D): [`lower_maxpool`]'s rank-5 mirror, called from
/// its rank dispatch.
fn lower_maxpool3d(program: &mut Vec<Op>, values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>) -> Result<(), LowerError> {
    let image = lookup(values, node, 0)?.clone();
    let plan = plan_pool3d(program, node, &image, f32::NEG_INFINITY)?;
    let windowed = window_materialize3d(
        program,
        &plan.padded,
        plan.out_shape[2],
        plan.out_shape[3],
        plan.out_shape[4],
        plan.kernel_d,
        plan.kernel_h,
        plan.kernel_w,
        plan.stride_d,
        plan.stride_h,
        plan.stride_w,
        plan.dilation_d,
        plan.dilation_h,
        plan.dilation_w,
    );
    let result = build_reduce(
        program,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        windowed.node,
        identity_pattern(8),
        projection(8, &[0, 1, 2, 3, 4]),
        Some("maxpool3d".to_string()),
    );
    bind_output(values, node, 0, result, plan.out_shape);
    Ok(())
}

/// `AveragePool`, rank-5 (3D): [`lower_averagepool`]'s rank-5 mirror, called
/// from its rank dispatch. Same `count_include_pad`/nonzero-`pads`
/// restriction as the 2D path -- see [`lower_averagepool`]'s own doc.
fn lower_averagepool3d(program: &mut Vec<Op>, values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>) -> Result<(), LowerError> {
    let count_include_pad = attr_int(node, "count_include_pad").unwrap_or(0);
    let has_nonzero_pad = attr_ints(node, "pads").is_some_and(|pads| pads.iter().any(|&value| value != 0));
    if has_nonzero_pad && count_include_pad == 0 {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: "AveragePool lowering supports nonzero pads only with count_include_pad=1 (the ONNX default excludes padding from the divisor, which needs a per-position valid-count divisor not yet composed)".to_string(),
        });
    }
    let image = lookup(values, node, 0)?.clone();
    let plan = plan_pool3d(program, node, &image, 0.0)?;
    let windowed = window_materialize3d(
        program,
        &plan.padded,
        plan.out_shape[2],
        plan.out_shape[3],
        plan.out_shape[4],
        plan.kernel_d,
        plan.kernel_h,
        plan.kernel_w,
        plan.stride_d,
        plan.stride_h,
        plan.stride_w,
        plan.dilation_d,
        plan.dilation_h,
        plan.dilation_w,
    );
    let summed = build_reduce(
        program,
        ScalarOp::Add,
        ReduceInit::Zero,
        windowed.node,
        identity_pattern(8),
        projection(8, &[0, 1, 2, 3, 4]),
        Some("averagepool3d".to_string()),
    );
    let window_size = (plan.kernel_d * plan.kernel_h * plan.kernel_w) as f32;
    let inverse = constant_scalar(program, 1.0 / window_size);
    let result = build_elementwise(
        program,
        ScalarOp::Multiply,
        alloc::vec![(summed, IndexMap::Affine(identity_pattern(5))), (inverse, IndexMap::Affine(scalar_broadcast_pattern(5)))],
    );
    bind_output(values, node, 0, result, plan.out_shape);
    Ok(())
}

fn tensor_shape(tensor: &TensorProto<'_>) -> Vec<u64> {
    tensor.dims.iter().map(|&value| value as u64).collect()
}

/// This crate's `Op::Input::dtype` for a wire `TensorProto.DataType`/
/// `TypeProto.Tensor.elem_type` value -- `Int32`/`Int64` become
/// [`DType::Int32`] (the dtype [`crate::map::IndexMap::Computed`]'s own doc
/// requires for a gather's `indices` operand), everything this pass
/// understands numerically otherwise becomes [`DType::Float32`]: every
/// backend this crate's evaluator ships carries every buffer as f32
/// regardless of logical dtype (see [`decode_numeric_tensor`]), so `dtype`
/// here is purely the shape-inference-time integer/float distinction, never
/// a storage format choice.
fn onnx_dtype_to_op_dtype(data_type: i32) -> DType {
    match data_type {
        6 | 7 => DType::Int32,
        _ => DType::Float32,
    }
}

/// Decode a [`TensorProto`]'s payload to `f32` -- `float_data`/`int64_data`/
/// `int32_data` if present, otherwise little-endian `raw_data`
/// reinterpreted per [`TensorProto::data_type`]. Integer payloads decode to
/// their numeric `f32` value (`value as f32`), never a bit-pattern
/// reinterpretation: every buffer this crate's evaluator carries is f32
/// regardless of logical dtype (see [`onnx_dtype_to_op_dtype`]'s doc), so an
/// integer initializer's *value* is what a gather's `indices` operand needs
/// to read back, not its raw bit representation.
fn decode_numeric_tensor(tensor: &TensorProto<'_>) -> Result<Vec<f32>, LowerError> {
    if !tensor.float_data.is_empty() {
        return Ok(tensor.float_data.clone());
    }
    if !tensor.int64_data.is_empty() {
        return Ok(tensor.int64_data.iter().map(|&value| value as f32).collect());
    }
    if !tensor.int32_data.is_empty() {
        return Ok(tensor.int32_data.iter().map(|&value| value as f32).collect());
    }
    if let Some(raw) = tensor.raw_data {
        match tensor.data_type {
            7 if raw.len() % 8 == 0 => {
                return Ok(raw.as_chunks::<8>().0.iter().map(|&chunk| i64::from_le_bytes(chunk) as f32).collect());
            }
            6 if raw.len() % 4 == 0 => {
                return Ok(raw.as_chunks::<4>().0.iter().map(|&chunk| i32::from_le_bytes(chunk) as f32).collect());
            }
            _ if raw.len() % 4 == 0 => {
                return Ok(raw.as_chunks::<4>().0.iter().map(|&chunk| f32::from_le_bytes(chunk)).collect());
            }
            _ => {}
        }
    }
    if tensor.dims.iter().product::<i64>() == 0 {
        return Ok(Vec::new());
    }
    Err(LowerError::UndecodableInitializer { name: tensor.name.to_string(), data_type: tensor.data_type })
}

/// A graph input's declared dtype, via [`onnx_dtype_to_op_dtype`] --
/// defaults to [`DType::Float32`] if the type is missing or not a plain
/// tensor, matching [`value_info_shape`]'s own error path shape (a caller
/// that needs the strict form gets [`LowerError::UnsupportedShape`] from
/// that function regardless).
fn value_info_dtype(value_info: &ValueInfoProto<'_>) -> DType {
    let Some(type_proto) = value_info.r#type.as_ref() else { return DType::Float32 };
    let Some(TypeValue::Tensor(tensor_type)) = type_proto.value.as_ref() else { return DType::Float32 };
    onnx_dtype_to_op_dtype(tensor_type.elem_type)
}

/// A graph input's declared shape, rejecting symbolic (named, not valued)
/// dimensions -- this pass lowers to concrete [`Extent::Static`] leaves
/// only, matching the concrete test fixture this crate ships.
fn value_info_shape(value_info: &ValueInfoProto<'_>) -> Result<Vec<u64>, LowerError> {
    let unsupported = |reason: &str| LowerError::UnsupportedShape {
        name: value_info.name.to_string(),
        op_type: "graph_input".to_string(),
        reason: reason.to_string(),
    };
    let type_proto = value_info.r#type.as_ref().ok_or_else(|| unsupported("missing type"))?;
    let Some(TypeValue::Tensor(tensor_type)) = type_proto.value.as_ref() else {
        return Err(unsupported("only Tensor-typed graph inputs are supported"));
    };
    let shape = tensor_type.shape.as_ref().ok_or_else(|| unsupported("missing shape"))?;
    shape
        .dim
        .iter()
        .map(|dimension| match &dimension.value {
            Some(DimensionValue::Value(value)) => Ok(*value as u64),
            _ => Err(unsupported("symbolic dimensions are not supported by this lowering")),
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec;

    use proxima_tensor::cpu::evaluate_named;

    use super::*;
    use crate::messages::{Dimension, TensorShapeProto, TypeProto, TypeProtoTensor};

    fn input_value_info(name: &'static str, dims: &[i64]) -> ValueInfoProto<'static> {
        let shape = TensorShapeProto {
            dim: dims
                .iter()
                .map(|&value| Dimension { value: Some(DimensionValue::Value(value)), denotation: "" })
                .collect(),
        };
        ValueInfoProto {
            name,
            r#type: Some(TypeProto {
                value: Some(TypeValue::Tensor(TypeProtoTensor { elem_type: 1, shape: Some(shape) })),
                denotation: "",
            }),
            doc_string: "",
        }
    }

    fn f32_initializer(name: &'static str, dims: &[i64], data: &[f32]) -> TensorProto<'static> {
        TensorProto { dims: dims.to_vec(), data_type: 1, float_data: data.to_vec(), name, ..TensorProto::default() }
    }

    /// `Add`'s broadcast path: a `[3]` bias vector added into a `[2, 3]`
    /// matrix, the same "broadcast bias add" shape `proxima-tensor/src/shape.rs`'s
    /// own unit test (`a_broadcast_bias_add_resolves_the_wider_shape`)
    /// exercises for the underlying `Op::Elementwise`.
    #[test]
    fn add_broadcasts_a_bias_vector_into_a_matrix() {
        let x_initializer = f32_initializer("x", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let bias_initializer = f32_initializer("bias", &[3], &[10.0, 20.0, 30.0]);
        let node = NodeProto { input: vec!["x", "bias"], output: vec!["y"], op_type: "Add", name: "add", ..NodeProto::default() };
        let graph = GraphProto {
            node: vec![node],
            name: "add_graph",
            initializer: vec![x_initializer, bias_initializer],
            input: Vec::new(),
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower Add");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate Add");
        let (data, shape) = evaluated.get(output).expect("y present");

        assert_eq!(shape, &[2, 3]);
        assert_eq!(data, &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    /// `Transpose(perm = [1, 0])` on a `[2, 3]` matrix: a permuted
    /// [`IndexMap`], never a copy in the algebra.
    #[test]
    fn transpose_permutes_a_matrix() {
        let x_initializer = f32_initializer("x", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let perm_attribute = AttributeProto { name: "perm", ints: vec![1, 0], ..AttributeProto::default() };
        let node =
            NodeProto { input: vec!["x"], output: vec!["y"], op_type: "Transpose", name: "transpose", attribute: vec![perm_attribute], ..NodeProto::default() };
        let graph = GraphProto {
            node: vec![node],
            name: "transpose_graph",
            initializer: vec![x_initializer],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower Transpose");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate Transpose");
        let (data, shape) = evaluated.get(output).expect("y present");

        assert_eq!(shape, &[3, 2]);
        assert_eq!(data, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    /// `Gather(data, indices, axis=0)` on a `[3, 2]` embedding-style table
    /// with `indices = [2, 0]`: `IndexMap::Computed`, the same composition
    /// `proxima-tensor/src/spec.rs`'s `embedding_lookup` builds, generalized
    /// through this module's [`lower_gather`].
    #[test]
    fn gather_selects_rows_by_index() {
        let table_initializer = f32_initializer("table", &[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let indices_initializer = TensorProto { dims: vec![2], data_type: 7, int64_data: vec![2, 0], name: "ids", ..TensorProto::default() };
        let node = NodeProto { input: vec!["table", "ids"], output: vec!["y"], op_type: "Gather", name: "gather", ..NodeProto::default() };
        let graph = GraphProto {
            node: vec![node],
            name: "gather_graph",
            initializer: vec![table_initializer, indices_initializer],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower Gather");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate Gather");
        let (data, shape) = evaluated.get(output).expect("y present");

        assert_eq!(shape, &[2, 2]);
        assert_eq!(data, &[5.0, 6.0, 1.0, 2.0]);
    }

    /// An `op_type` this module never composes -- [`LowerError::UnsupportedOp`]
    /// carries the name so the RISC-sufficiency gaps list has something to
    /// point at, never a silent fallback.
    #[test]
    fn unsupported_op_type_is_a_typed_error_not_a_panic() {
        let x_initializer = f32_initializer("x", &[2], &[1.0, 2.0]);
        let node = NodeProto { input: vec!["x"], output: vec!["y"], op_type: "LSTM", name: "lstm", ..NodeProto::default() };
        let graph = GraphProto {
            node: vec![node],
            name: "unsupported_graph",
            initializer: vec![x_initializer],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let error = lower_graph(&graph).expect_err("LSTM has no lowering");
        assert!(matches!(error, LowerError::UnsupportedOp { .. }), "expected UnsupportedOp, got {error:?}");
    }

    /// `Reshape` merging `[2, 3, 4]` into `[6, 4]`: pure layout, no `Op`
    /// appended -- the program is exactly as long after lowering as before
    /// the `Reshape` node, proving this is a [`Value`] view alias
    /// ([`lower_unsqueeze`]'s own pattern), never a new compute `Op`.
    #[test]
    fn reshape_merges_axes_without_appending_an_op() {
        let x_initializer = f32_initializer("x", &[2, 3, 4], &(0..24).map(|value| value as f32).collect::<Vec<_>>());
        let shape_initializer = TensorProto { dims: vec![2], data_type: 7, int64_data: vec![6, 4], name: "shape", ..TensorProto::default() };
        let node = NodeProto { input: vec!["x", "shape"], output: vec!["y"], op_type: "Reshape", name: "reshape", ..NodeProto::default() };
        let graph = GraphProto {
            node: vec![node],
            name: "reshape_merge_graph",
            initializer: vec![x_initializer, shape_initializer],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower Reshape merge");
        assert_eq!(lowered.program.len(), 1, "a contiguous reshape appends no new Op (the shape tensor is value-only, never a live leaf)");

        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate Reshape merge");
        let (data, _) = evaluated.get(output).expect("y present");
        let expected: Vec<f32> = (0..24).map(|value| value as f32).collect();
        assert_eq!(data, expected.as_slice(), "merge preserves element order (flattened)");
    }

    /// `Reshape` merging all the way down to a flat `[24]` -- the same
    /// view-alias mechanism, one more axis collapsed.
    #[test]
    fn reshape_flattens_to_rank_one_without_appending_an_op() {
        let x_initializer = f32_initializer("x", &[2, 3, 4], &(0..24).map(|value| value as f32).collect::<Vec<_>>());
        let shape_initializer = TensorProto { dims: vec![1], data_type: 7, int64_data: vec![24], name: "shape", ..TensorProto::default() };
        let node = NodeProto { input: vec!["x", "shape"], output: vec!["y"], op_type: "Reshape", name: "reshape", ..NodeProto::default() };
        let graph = GraphProto {
            node: vec![node],
            name: "reshape_flatten_graph",
            initializer: vec![x_initializer, shape_initializer],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower Reshape flatten");
        assert_eq!(lowered.program.len(), 1, "a contiguous reshape appends no new Op (the shape tensor is value-only, never a live leaf)");

        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate Reshape flatten");
        let (data, _) = evaluated.get(output).expect("y present");
        let expected: Vec<f32> = (0..24).map(|value| value as f32).collect();
        assert_eq!(data, expected.as_slice());
    }

    /// `Reshape` splitting `[24]` into `[2, 3, 4]`: the reverse direction,
    /// also a pure view alias with no new `Op`.
    #[test]
    fn reshape_splits_a_flat_axis_without_appending_an_op() {
        let x_initializer = f32_initializer("x", &[24], &(0..24).map(|value| value as f32).collect::<Vec<_>>());
        let shape_initializer = TensorProto { dims: vec![3], data_type: 7, int64_data: vec![2, 3, 4], name: "shape", ..TensorProto::default() };
        let node = NodeProto { input: vec!["x", "shape"], output: vec!["y"], op_type: "Reshape", name: "reshape", ..NodeProto::default() };
        let graph = GraphProto {
            node: vec![node],
            name: "reshape_split_graph",
            initializer: vec![x_initializer, shape_initializer],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower Reshape split");
        assert_eq!(lowered.program.len(), 1, "a contiguous reshape appends no new Op (the shape tensor is value-only, never a live leaf)");

        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate Reshape split");
        let (data, _) = evaluated.get(output).expect("y present");
        let expected: Vec<f32> = (0..24).map(|value| value as f32).collect();
        assert_eq!(data, expected.as_slice(), "split preserves element order (still flattened)");
    }

    /// `Concat` along `axis=0` of two `[2, 2]` matrices.
    #[test]
    fn concat_joins_two_matrices_along_axis_zero() {
        let a_initializer = f32_initializer("a", &[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b_initializer = f32_initializer("b", &[2, 2], &[5.0, 6.0, 7.0, 8.0]);
        let axis_attribute = AttributeProto { name: "axis", i: 0, ..AttributeProto::default() };
        let node =
            NodeProto { input: vec!["a", "b"], output: vec!["y"], op_type: "Concat", name: "concat", attribute: vec![axis_attribute], ..NodeProto::default() };
        let graph = GraphProto {
            node: vec![node],
            name: "concat_axis0_graph",
            initializer: vec![a_initializer, b_initializer],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower Concat axis=0");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate Concat axis=0");
        let (data, shape) = evaluated.get(output).expect("y present");

        assert_eq!(shape, &[4, 2]);
        assert_eq!(data, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    /// `Concat` along a general (`axis=1`) axis, three inputs folded
    /// pairwise.
    #[test]
    fn concat_joins_three_matrices_along_a_general_axis() {
        let a_initializer = f32_initializer("a", &[2, 1], &[1.0, 4.0]);
        let b_initializer = f32_initializer("b", &[2, 2], &[2.0, 3.0, 5.0, 6.0]);
        let c_initializer = f32_initializer("c", &[2, 1], &[7.0, 8.0]);
        let axis_attribute = AttributeProto { name: "axis", i: 1, ..AttributeProto::default() };
        let node = NodeProto {
            input: vec!["a", "b", "c"],
            output: vec!["y"],
            op_type: "Concat",
            name: "concat",
            attribute: vec![axis_attribute],
            ..NodeProto::default()
        };
        let graph = GraphProto {
            node: vec![node],
            name: "concat_axis1_graph",
            initializer: vec![a_initializer, b_initializer, c_initializer],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower Concat axis=1");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate Concat axis=1");
        let (data, shape) = evaluated.get(output).expect("y present");

        assert_eq!(shape, &[2, 4]);
        assert_eq!(data, &[1.0, 2.0, 3.0, 7.0, 4.0, 5.0, 6.0, 8.0]);
    }

    /// Batched `MatMul`: `[2, 2, 3] x [2, 3, 2]`, independently computed per
    /// batch.
    #[test]
    fn matmul_contracts_a_batch_of_matrices() {
        let a_initializer = f32_initializer("a", &[2, 2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let b_initializer = f32_initializer("b", &[2, 3, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 2.0, 1.0, 1.0]);
        let node = NodeProto { input: vec!["a", "b"], output: vec!["y"], op_type: "MatMul", name: "matmul", ..NodeProto::default() };
        let graph = GraphProto {
            node: vec![node],
            name: "batched_matmul_graph",
            initializer: vec![a_initializer, b_initializer],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower batched MatMul");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate batched MatMul");
        let (data, shape) = evaluated.get(output).expect("y present");

        assert_eq!(shape, &[2, 2, 2]);
        // batch 0: [[1,2,3],[4,5,6]] @ [[1,0],[0,1],[1,1]] = [[4,5],[10,11]]
        // batch 1: [[1,0,0],[0,1,0]] @ [[2,0],[0,2],[1,1]] = [[2,0],[0,2]]
        assert_eq!(data, &[4.0, 5.0, 10.0, 11.0, 2.0, 0.0, 0.0, 2.0]);
    }

    /// Batched `MatMul` where the right-hand operand carries no batch
    /// dimension at all -- `[2, 2, 3] x [3, 2]` -- and broadcasts across the
    /// batch the way [`broadcast_shapes`] already treats a missing leading
    /// axis as extent 1.
    #[test]
    fn matmul_broadcasts_a_shared_right_hand_matrix_across_the_batch() {
        let a_initializer = f32_initializer("a", &[2, 2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let b_initializer = f32_initializer("b", &[3, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let node = NodeProto { input: vec!["a", "b"], output: vec!["y"], op_type: "MatMul", name: "matmul", ..NodeProto::default() };
        let graph = GraphProto {
            node: vec![node],
            name: "broadcast_matmul_graph",
            initializer: vec![a_initializer, b_initializer],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower broadcast MatMul");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate broadcast MatMul");
        let (data, shape) = evaluated.get(output).expect("y present");

        assert_eq!(shape, &[2, 2, 2]);
        // batch 0: [[1,2,3],[4,5,6]] @ [[1,0],[0,1],[1,1]] = [[4,5],[10,11]]
        // batch 1: [[1,0,0],[0,1,0]] @ same b = [[1,0],[0,1]]
        assert_eq!(data, &[4.0, 5.0, 10.0, 11.0, 1.0, 0.0, 0.0, 1.0]);
    }

    /// `Softmax(axis=0)` on a `[3, 2]` input -- the reduced axis is the
    /// leading one, not the last, proving [`lower_softmax`]'s generalization
    /// beyond the previous "last axis only" restriction.
    #[test]
    fn softmax_reduces_the_leading_axis() {
        let x_initializer = f32_initializer("x", &[3, 2], &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let axis_attribute = AttributeProto { name: "axis", i: 0, ..AttributeProto::default() };
        let node = NodeProto { input: vec!["x"], output: vec!["y"], op_type: "Softmax", name: "softmax", attribute: vec![axis_attribute], ..NodeProto::default() };
        let graph = GraphProto {
            node: vec![node],
            name: "softmax_axis0_graph",
            initializer: vec![x_initializer],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower Softmax axis=0");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate Softmax axis=0");
        let (data, shape) = evaluated.get(output).expect("y present");

        assert_eq!(shape, &[3, 2]);
        // every column is identical ([0,0,0] or [1,1,1]), so softmax over the
        // 3-element column is uniform: 1/3 each.
        for &value in data {
            assert!((value - 1.0 / 3.0).abs() < 1e-6, "expected uniform 1/3, got {value}");
        }
    }

    /// `Softmax(axis=1)` on a rank-3 `[2, 3, 2]` input -- a middle axis,
    /// neither leading nor trailing.
    #[test]
    fn softmax_reduces_a_middle_axis_of_a_rank_three_input() {
        let x_initializer = f32_initializer("x", &[2, 3, 2], &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let axis_attribute = AttributeProto { name: "axis", i: 1, ..AttributeProto::default() };
        let node = NodeProto { input: vec!["x"], output: vec!["y"], op_type: "Softmax", name: "softmax", attribute: vec![axis_attribute], ..NodeProto::default() };
        let graph = GraphProto {
            node: vec![node],
            name: "softmax_axis1_graph",
            initializer: vec![x_initializer],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower Softmax axis=1");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate Softmax axis=1");
        let (data, shape) = evaluated.get(output).expect("y present");

        assert_eq!(shape, &[2, 3, 2]);
        // batch 0 is all zero along the reduced axis -> uniform 1/3.
        for &value in &data[0..6] {
            assert!((value - 1.0 / 3.0).abs() < 1e-6, "expected uniform 1/3, got {value}");
        }
        // batch 1, each column sums to 1.
        let column0_sum = data[6] + data[8] + data[10];
        let column1_sum = data[7] + data[9] + data[11];
        assert!((column0_sum - 1.0).abs() < 1e-5);
        assert!((column1_sum - 1.0).abs() < 1e-5);
        assert!(data[10] > data[8] && data[8] > data[6], "softmax preserves ordering within the reduced axis");
    }

    /// `Gather(data, indices, axis=1)` on a `[2, 3]` table with
    /// `indices = [2, 0]`, proving [`lower_gather`]'s generalization beyond
    /// `axis=0`.
    #[test]
    fn gather_selects_columns_by_index_at_a_general_axis() {
        let table_initializer = f32_initializer("table", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let indices_initializer = TensorProto { dims: vec![2], data_type: 7, int64_data: vec![2, 0], name: "ids", ..TensorProto::default() };
        let axis_attribute = AttributeProto { name: "axis", i: 1, ..AttributeProto::default() };
        let node =
            NodeProto { input: vec!["table", "ids"], output: vec!["y"], op_type: "Gather", name: "gather", attribute: vec![axis_attribute], ..NodeProto::default() };
        let graph = GraphProto {
            node: vec![node],
            name: "gather_axis1_graph",
            initializer: vec![table_initializer, indices_initializer],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower Gather axis=1");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate Gather axis=1");
        let (data, shape) = evaluated.get(output).expect("y present");

        assert_eq!(shape, &[2, 2]);
        // row 0: [1,2,3] -> columns [2,0] -> [3,1]; row 1: [4,5,6] -> [6,4]
        assert_eq!(data, &[3.0, 1.0, 6.0, 4.0]);
    }

    /// Sanity check on [`input_value_info`]'s own helper, exercised
    /// indirectly by every graph-input-driven test elsewhere in this crate
    /// (`tests.rs`'s MLP fixture) -- kept here so a change to
    /// [`value_info_shape`]'s symbolic-dimension rejection has a direct
    /// unit test next to the function it covers.
    #[test]
    fn symbolic_graph_input_dimension_is_rejected() {
        let mut value_info = input_value_info("x", &[2]);
        if let Some(TypeProto { value: Some(TypeValue::Tensor(tensor)), .. }) = value_info.r#type.as_mut() {
            tensor.shape = Some(TensorShapeProto { dim: vec![Dimension { value: Some(DimensionValue::Param("batch")), denotation: "" }] });
        }
        let error = value_info_shape(&value_info).expect_err("symbolic dim rejected");
        assert!(matches!(error, LowerError::UnsupportedShape { .. }));
    }

    fn ints_attribute(name: &'static str, ints: Vec<i64>) -> AttributeProto<'static> {
        AttributeProto { name, ints, ..AttributeProto::default() }
    }

    /// `Conv`, stride 1, no padding: a 3x3 all-ones kernel over a `4x4`
    /// image is a 3x3 sliding-window sum -- `Reduce(Add)` over
    /// `Elementwise(Multiply)` against the [`window_materialize`]d input,
    /// [`lower_conv`]'s whole composition, hand-verified against an
    /// independently summed reference.
    #[test]
    fn conv_stride1_no_pad_sums_each_3x3_window() {
        let image = f32_initializer("image", &[1, 1, 4, 4], &(1..=16).map(|value| value as f32).collect::<Vec<_>>());
        let weight = f32_initializer("weight", &[1, 1, 3, 3], &[1.0; 9]);
        let node = NodeProto { input: vec!["image", "weight"], output: vec!["y"], op_type: "Conv", name: "conv", ..NodeProto::default() };
        let graph = GraphProto {
            node: vec![node],
            name: "conv_stride1_graph",
            initializer: vec![image, weight],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower Conv");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate Conv");
        let (data, shape) = evaluated.get(output).expect("y present");

        assert_eq!(shape, &[1, 1, 2, 2]);
        assert_eq!(data, &[54.0, 63.0, 90.0, 99.0], "hand-summed 3x3 windows over the 4x4 image");
    }

    /// `Conv`, stride 2 and `pads = [1, 1, 1, 1]`: exercises [`pad_axis`]'s
    /// clamp-and-select zero padding together with the two-term
    /// `stride*out + dilation*kernel` window axis in the same op, hand-
    /// verified against a manually zero-padded 6x6 reference.
    #[test]
    fn conv_stride2_with_padding_sums_each_padded_window() {
        let image = f32_initializer("image", &[1, 1, 4, 4], &(1..=16).map(|value| value as f32).collect::<Vec<_>>());
        let weight = f32_initializer("weight", &[1, 1, 3, 3], &[1.0; 9]);
        let node = NodeProto {
            input: vec!["image", "weight"],
            output: vec!["y"],
            op_type: "Conv",
            name: "conv",
            attribute: vec![ints_attribute("strides", vec![2, 2]), ints_attribute("pads", vec![1, 1, 1, 1])],
            ..NodeProto::default()
        };
        let graph = GraphProto {
            node: vec![node],
            name: "conv_stride2_pad1_graph",
            initializer: vec![image, weight],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower Conv stride2 pad1");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate Conv stride2 pad1");
        let (data, shape) = evaluated.get(output).expect("y present");

        assert_eq!(shape, &[1, 1, 2, 2]);
        assert_eq!(data, &[14.0, 30.0, 57.0, 99.0], "hand-summed windows over the zero-padded 6x6 image");
    }

    /// `Conv` with a bias operand: the trailing broadcast `Add` [`lower_conv`]
    /// appends over the `co` axis, on top of the stride-1 no-pad case above.
    #[test]
    fn conv_adds_a_per_output_channel_bias() {
        let image = f32_initializer("image", &[1, 1, 4, 4], &(1..=16).map(|value| value as f32).collect::<Vec<_>>());
        let weight = f32_initializer("weight", &[1, 1, 3, 3], &[1.0; 9]);
        let bias = f32_initializer("bias", &[1], &[100.0]);
        let node = NodeProto { input: vec!["image", "weight", "bias"], output: vec!["y"], op_type: "Conv", name: "conv", ..NodeProto::default() };
        let graph = GraphProto {
            node: vec![node],
            name: "conv_bias_graph",
            initializer: vec![image, weight, bias],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower Conv with bias");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate Conv with bias");
        let (data, shape) = evaluated.get(output).expect("y present");

        assert_eq!(shape, &[1, 1, 2, 2]);
        assert_eq!(data, &[154.0, 163.0, 190.0, 199.0], "bias 100 added to every windowed sum");
    }

    /// `Conv` with `group=2` on an all-zero image lowers cleanly to an
    /// all-zero output -- [`grouped_conv_depthwise_slices_channels_independently`]
    /// below is the sharper, non-degenerate check on the same path; this one
    /// keeps a direct regression on the shape [`lower_conv`]'s grouped
    /// decomposition ([`slice_axis_range`] + [`conv2d_core`] + [`concat_pair`])
    /// produces, no longer a deferred [`LowerError::UnsupportedShape`] gap.
    #[test]
    fn conv_grouped_convolution_lowers_to_the_expected_output_shape() {
        let image = f32_initializer("image", &[1, 2, 4, 4], &[0.0; 32]);
        let weight = f32_initializer("weight", &[2, 1, 3, 3], &[1.0; 18]);
        let node = NodeProto {
            input: vec!["image", "weight"],
            output: vec!["y"],
            op_type: "Conv",
            name: "conv",
            attribute: vec![AttributeProto { name: "group", i: 2, ..AttributeProto::default() }],
            ..NodeProto::default()
        };
        let graph = GraphProto {
            node: vec![node],
            name: "conv_grouped_graph",
            initializer: vec![image, weight],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower grouped Conv");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate grouped Conv");
        let (data, shape) = evaluated.get(output).expect("y present");

        assert_eq!(shape, &[1, 2, 2, 2]);
        assert!(data.iter().all(|&value| value == 0.0), "an all-zero image convolves to an all-zero output");
    }

    /// `MaxPool`, 2x2 kernel and stride: [`window_materialize`] padded with
    /// `-inf` (never contending for the max), `Reduce(Maximum)` over the
    /// window -- hand-verified against the per-block max of a 4x4 image.
    #[test]
    fn maxpool_2x2_takes_the_max_of_each_block() {
        let image = f32_initializer("image", &[1, 1, 4, 4], &(1..=16).map(|value| value as f32).collect::<Vec<_>>());
        let node = NodeProto {
            input: vec!["image"],
            output: vec!["y"],
            op_type: "MaxPool",
            name: "maxpool",
            attribute: vec![ints_attribute("kernel_shape", vec![2, 2]), ints_attribute("strides", vec![2, 2])],
            ..NodeProto::default()
        };
        let graph = GraphProto {
            node: vec![node],
            name: "maxpool_graph",
            initializer: vec![image],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower MaxPool");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate MaxPool");
        let (data, shape) = evaluated.get(output).expect("y present");

        assert_eq!(shape, &[1, 1, 2, 2]);
        assert_eq!(data, &[6.0, 8.0, 14.0, 16.0], "max of each 2x2 block of the 4x4 image");
    }

    /// `AveragePool`, 2x2 kernel and stride: [`window_materialize`] padded
    /// with zero (unused here, no padding attribute), `Reduce(Add)` then a
    /// `1/4` scale -- hand-verified against the per-block mean.
    #[test]
    fn averagepool_2x2_takes_the_mean_of_each_block() {
        let image = f32_initializer("image", &[1, 1, 4, 4], &(1..=16).map(|value| value as f32).collect::<Vec<_>>());
        let node = NodeProto {
            input: vec!["image"],
            output: vec!["y"],
            op_type: "AveragePool",
            name: "averagepool",
            attribute: vec![ints_attribute("kernel_shape", vec![2, 2]), ints_attribute("strides", vec![2, 2])],
            ..NodeProto::default()
        };
        let graph = GraphProto {
            node: vec![node],
            name: "averagepool_graph",
            initializer: vec![image],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower AveragePool");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate AveragePool");
        let (data, shape) = evaluated.get(output).expect("y present");

        assert_eq!(shape, &[1, 1, 2, 2]);
        assert_eq!(data, &[3.5, 5.5, 11.5, 13.5], "mean of each 2x2 block of the 4x4 image");
    }

    /// `AveragePool` with nonzero `pads` and ONNX's default `count_include_pad=0`:
    /// a deferred, named gap -- the fixed `1/(kh*kw)` divisor this
    /// composition builds is only correct when padding is excluded from the
    /// count (`count_include_pad=1`) or absent entirely, never silently
    /// applied to a boundary-diluted average.
    #[test]
    fn averagepool_padded_default_count_is_a_named_unsupported_shape_not_a_silent_average() {
        let image = f32_initializer("image", &[1, 1, 4, 4], &(1..=16).map(|value| value as f32).collect::<Vec<_>>());
        let node = NodeProto {
            input: vec!["image"],
            output: vec!["y"],
            op_type: "AveragePool",
            name: "averagepool",
            attribute: vec![ints_attribute("kernel_shape", vec![2, 2]), ints_attribute("pads", vec![1, 1, 1, 1])],
            ..NodeProto::default()
        };
        let graph = GraphProto {
            node: vec![node],
            name: "averagepool_padded_graph",
            initializer: vec![image],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let error = lower_graph(&graph).expect_err("padded AveragePool with count_include_pad=0 is a deferred gap");
        assert!(matches!(error, LowerError::UnsupportedShape { .. }), "expected UnsupportedShape, got {error:?}");
    }

    fn graph_attribute(name: &'static str, subgraph: GraphProto<'static>) -> AttributeProto<'static> {
        AttributeProto { name, g: Some(subgraph), ..AttributeProto::default() }
    }

    /// `Where(cond, x, y)`: pure dataflow, [`ScalarOp::Select`] over the
    /// three operands, no subgraph -- the reverse of `Select -> "Where"`
    /// lift emission.
    #[test]
    fn where_selects_elementwise_between_two_tensors() {
        let cond_initializer = f32_initializer("cond", &[2], &[1.0, 0.0]);
        let x_initializer = f32_initializer("x", &[2], &[10.0, 20.0]);
        let y_initializer = f32_initializer("y", &[2], &[30.0, 40.0]);
        let node = NodeProto { input: vec!["cond", "x", "y"], output: vec!["z"], op_type: "Where", name: "where", ..NodeProto::default() };
        let graph = GraphProto {
            node: vec![node],
            name: "where_graph",
            initializer: vec![cond_initializer, x_initializer, y_initializer],
            output: vec![ValueInfoProto { name: "z", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower Where");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate Where");
        let (data, shape) = evaluated.get(output).expect("z present");

        assert_eq!(shape, &[2]);
        assert_eq!(data, &[10.0, 40.0], "cond[0]=1 picks x[0]=10, cond[1]=0 picks y[1]=40");
    }

    /// `If` whose `cond` is a graph initializer (a lower-time constant):
    /// only `then_branch` is lowered and inlined, `else_branch` contributes
    /// nothing to the program.
    #[test]
    fn if_with_constant_true_condition_inlines_only_then_branch() {
        let x_initializer = f32_initializer("x", &[1], &[5.0]);
        let cond_initializer = f32_initializer("cond", &[], &[1.0]);
        let then_branch = GraphProto {
            node: vec![NodeProto { input: vec!["x"], output: vec!["then_out"], op_type: "Identity", name: "then_identity", ..NodeProto::default() }],
            name: "then",
            output: vec![ValueInfoProto { name: "then_out", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };
        let else_branch = GraphProto {
            node: vec![NodeProto { input: vec!["x"], output: vec!["else_out"], op_type: "Neg", name: "else_neg", ..NodeProto::default() }],
            name: "else",
            output: vec![ValueInfoProto { name: "else_out", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };
        let node = NodeProto {
            input: vec!["cond"],
            output: vec!["y"],
            op_type: "If",
            name: "if",
            attribute: vec![graph_attribute("then_branch", then_branch), graph_attribute("else_branch", else_branch)],
            ..NodeProto::default()
        };
        let graph = GraphProto {
            node: vec![node],
            name: "if_const_graph",
            initializer: vec![x_initializer, cond_initializer],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower If with constant true cond");
        assert_eq!(
            lowered.program.len(),
            3,
            "x's and cond's Input leaves plus then_branch's Identity are appended -- else_branch is never lowered"
        );

        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate If constant true");
        let (data, _) = evaluated.get(output).expect("y present");
        assert_eq!(data, &[5.0], "constant-true cond picks then_branch (Identity(x))");
    }

    /// `If` whose `cond` is only known from computed data (`Greater`, not an
    /// initializer or folded `Constant`): both branches are lowered
    /// unconditionally and the output is a per-position `Select` (Option A
    /// from this module's own doc), valid here because both branches
    /// produce the same `[1]` shape.
    #[test]
    fn if_with_data_dependent_condition_selects_between_both_lowered_branches() {
        let x_initializer = f32_initializer("x", &[1], &[3.0]);
        let zero_initializer = f32_initializer("zero", &[1], &[0.0]);
        let cond_node = NodeProto { input: vec!["x", "zero"], output: vec!["cond"], op_type: "Greater", name: "greater", ..NodeProto::default() };
        let then_branch = GraphProto {
            node: vec![NodeProto { input: vec!["x"], output: vec!["then_out"], op_type: "Identity", name: "then_identity", ..NodeProto::default() }],
            name: "then",
            output: vec![ValueInfoProto { name: "then_out", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };
        let else_branch = GraphProto {
            node: vec![NodeProto { input: vec!["x"], output: vec!["else_out"], op_type: "Neg", name: "else_neg", ..NodeProto::default() }],
            name: "else",
            output: vec![ValueInfoProto { name: "else_out", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };
        let if_node = NodeProto {
            input: vec!["cond"],
            output: vec!["y"],
            op_type: "If",
            name: "if",
            attribute: vec![graph_attribute("then_branch", then_branch), graph_attribute("else_branch", else_branch)],
            ..NodeProto::default()
        };
        let graph = GraphProto {
            node: vec![cond_node, if_node],
            name: "if_data_dependent_graph",
            initializer: vec![x_initializer, zero_initializer],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower If with data-dependent cond");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate If data-dependent");
        let (data, _) = evaluated.get(output).expect("y present");
        assert_eq!(data, &[3.0], "x=3 > 0, cond true, Select picks then_branch's Identity(x)=3");
    }

    /// The boundary Option A itself names: a data-dependent `If` whose two
    /// branches produce differently-shaped outputs has no `Select` that
    /// reconciles them -- [`LowerError::UnsupportedShape`], never a silent
    /// pick of one side.
    #[test]
    fn if_data_dependent_with_shape_mismatched_branches_is_a_named_unsupported_shape() {
        let x_initializer = f32_initializer("x", &[1], &[3.0]);
        let zero_initializer = f32_initializer("zero", &[1], &[0.0]);
        let cond_node = NodeProto { input: vec!["x", "zero"], output: vec!["cond"], op_type: "Greater", name: "greater", ..NodeProto::default() };
        let then_branch = GraphProto {
            node: vec![NodeProto { input: vec!["x"], output: vec!["then_out"], op_type: "Identity", name: "then_identity", ..NodeProto::default() }],
            name: "then",
            output: vec![ValueInfoProto { name: "then_out", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };
        let else_branch = GraphProto {
            node: vec![NodeProto {
                input: vec!["x", "x"],
                output: vec!["else_out"],
                op_type: "Concat",
                name: "else_concat",
                attribute: vec![AttributeProto { name: "axis", i: 0, ..AttributeProto::default() }],
                ..NodeProto::default()
            }],
            name: "else",
            output: vec![ValueInfoProto { name: "else_out", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };
        let if_node = NodeProto {
            input: vec!["cond"],
            output: vec!["y"],
            op_type: "If",
            name: "if",
            attribute: vec![graph_attribute("then_branch", then_branch), graph_attribute("else_branch", else_branch)],
            ..NodeProto::default()
        };
        let graph = GraphProto {
            node: vec![cond_node, if_node],
            name: "if_shape_mismatch_graph",
            initializer: vec![x_initializer, zero_initializer],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let error = lower_graph(&graph).expect_err("differently-shaped branches cannot Select");
        assert!(matches!(error, LowerError::UnsupportedShape { .. }), "expected UnsupportedShape, got {error:?}");
    }

    /// `Scan` with one state variable and one `scan_input`: unrolled `trip
    /// count = scan_input.shape[0]` times, each iteration reading its slice
    /// via [`slice_axis0`] and folding into the running state -- a plain sum
    /// over `[1, 2, 3, 4]`.
    #[test]
    fn scan_sums_a_sequence_by_unrolling_the_body() {
        let state_initializer = f32_initializer("state0", &[], &[0.0]);
        let sequence_initializer = f32_initializer("seq", &[4], &[1.0, 2.0, 3.0, 4.0]);
        let body = GraphProto {
            node: vec![NodeProto { input: vec!["state_in", "slice"], output: vec!["state_out"], op_type: "Add", name: "body_add", ..NodeProto::default() }],
            name: "scan_body",
            input: vec![ValueInfoProto { name: "state_in", ..ValueInfoProto::default() }, ValueInfoProto { name: "slice", ..ValueInfoProto::default() }],
            output: vec![ValueInfoProto { name: "state_out", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };
        let node = NodeProto {
            input: vec!["state0", "seq"],
            output: vec!["y"],
            op_type: "Scan",
            name: "scan",
            attribute: vec![AttributeProto { name: "num_scan_inputs", i: 1, ..AttributeProto::default() }, graph_attribute("body", body)],
            ..NodeProto::default()
        };
        let graph = GraphProto {
            node: vec![node],
            name: "scan_sum_graph",
            initializer: vec![state_initializer, sequence_initializer],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower Scan");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate Scan");
        let (data, _) = evaluated.get(output).expect("y present");
        assert_eq!(data, &[10.0], "1 + 2 + 3 + 4 unrolled across four appended body copies");
    }

    /// `Loop` with a lower-time-constant `M` (no `cond` input): unrolled
    /// exactly `M` times, each iteration's `state_out` feeding the next
    /// iteration's `state_in` -- three `+1` steps starting from `0`.
    #[test]
    fn loop_with_static_trip_count_unrolls_the_body_exactly_m_times() {
        let trip_initializer = f32_initializer("trip", &[], &[3.0]);
        let state_initializer = f32_initializer("state0", &[], &[0.0]);
        let one_initializer = f32_initializer("one", &[], &[1.0]);
        let body = GraphProto {
            node: vec![NodeProto { input: vec!["state_in", "one"], output: vec!["state_out"], op_type: "Add", name: "body_add", ..NodeProto::default() }],
            name: "loop_body",
            input: vec![
                ValueInfoProto { name: "iter_num", ..ValueInfoProto::default() },
                ValueInfoProto { name: "cond_in", ..ValueInfoProto::default() },
                ValueInfoProto { name: "state_in", ..ValueInfoProto::default() },
            ],
            output: vec![ValueInfoProto { name: "cond_out_unused", ..ValueInfoProto::default() }, ValueInfoProto { name: "state_out", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };
        let node = NodeProto {
            input: vec!["trip", "", "state0"],
            output: vec!["y"],
            op_type: "Loop",
            name: "loop",
            attribute: vec![graph_attribute("body", body)],
            ..NodeProto::default()
        };
        let graph = GraphProto {
            node: vec![node],
            name: "loop_static_graph",
            initializer: vec![trip_initializer, state_initializer, one_initializer],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower Loop with static trip count");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate Loop static");
        let (data, _) = evaluated.get(output).expect("y present");
        assert_eq!(data, &[3.0], "0 + 1 + 1 + 1 across three unrolled body copies");
    }

    /// The RISC-sufficiency boundary this crate's control-flow lowering
    /// does not cross: `Loop`'s trip count `M` is only known from computed
    /// data (a graph input, never a constant), so there is no lower-time
    /// answer to "how many times does the body append" --
    /// [`LowerError::DataDependentControlFlow`], never a fabricated
    /// primitive and never a silently truncated unroll.
    #[test]
    fn loop_with_runtime_trip_count_is_a_named_data_dependent_control_flow_boundary() {
        let state_initializer = f32_initializer("state0", &[], &[0.0]);
        let one_initializer = f32_initializer("one", &[], &[1.0]);
        let body = GraphProto {
            node: vec![NodeProto { input: vec!["state_in", "one"], output: vec!["state_out"], op_type: "Add", name: "body_add", ..NodeProto::default() }],
            name: "loop_body",
            input: vec![
                ValueInfoProto { name: "iter_num", ..ValueInfoProto::default() },
                ValueInfoProto { name: "cond_in", ..ValueInfoProto::default() },
                ValueInfoProto { name: "state_in", ..ValueInfoProto::default() },
            ],
            output: vec![ValueInfoProto { name: "cond_out_unused", ..ValueInfoProto::default() }, ValueInfoProto { name: "state_out", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };
        let node = NodeProto {
            input: vec!["trip", "", "state0"],
            output: vec!["y"],
            op_type: "Loop",
            name: "loop",
            attribute: vec![graph_attribute("body", body)],
            ..NodeProto::default()
        };
        let graph = GraphProto {
            node: vec![node],
            name: "loop_runtime_trip_graph",
            input: vec![input_value_info("trip", &[])],
            initializer: vec![state_initializer, one_initializer],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let error = lower_graph(&graph).expect_err("a graph-input trip count is not a lower-time constant");
        assert!(
            matches!(error, LowerError::DataDependentControlFlow { .. }),
            "expected DataDependentControlFlow, got {error:?}"
        );
    }

    /// Depthwise `Conv`, `group = in_channels = 2`: [`lower_conv`]'s grouped
    /// path (`group != 1`) slices each of the 2 input channels via
    /// [`slice_axis_range`], runs [`conv2d_core`] once per group with that
    /// group's own `[1, 1, kh, kw]` weight slice, and [`concat_pair`]s the two
    /// single-channel results back along the output-channel axis -- static
    /// group-slicing plus `Concat`, no div/mod, no new `Op`/`IndexMap`.
    /// Each channel's 3x3 all-ones kernel over its own 4x4 plane sums to the
    /// same values [`conv_stride1_no_pad_sums_each_3x3_window`] already
    /// hand-verifies for a single channel; channel 1 is doubled so the two
    /// groups are independently distinguishable in the output.
    #[test]
    fn grouped_conv_depthwise_slices_channels_independently() {
        let channel0: Vec<f32> = (1..=16).map(|value| value as f32).collect();
        let channel1: Vec<f32> = channel0.iter().map(|&value| value * 2.0).collect();
        let mut image_data = channel0.clone();
        image_data.extend(channel1.iter().copied());
        let image = f32_initializer("image", &[1, 2, 4, 4], &image_data);
        let weight = f32_initializer("weight", &[2, 1, 3, 3], &[1.0; 18]);
        let group_attribute = AttributeProto { name: "group", i: 2, ..AttributeProto::default() };
        let node = NodeProto {
            input: vec!["image", "weight"],
            output: vec!["y"],
            op_type: "Conv",
            name: "conv",
            attribute: vec![group_attribute],
            ..NodeProto::default()
        };
        let graph = GraphProto {
            node: vec![node],
            name: "grouped_conv_depthwise_graph",
            initializer: vec![image, weight],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower grouped Conv");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate grouped Conv");
        let (data, shape) = evaluated.get(output).expect("y present");

        assert_eq!(shape, &[1, 2, 2, 2]);
        // channel 0 group: same 3x3 window sums as the single-channel case.
        assert_eq!(&data[0..4], &[54.0, 63.0, 90.0, 99.0], "group 0 matches the un-grouped single-channel sums");
        // channel 1 group: every input value doubled -> every window sum doubled.
        assert_eq!(&data[4..8], &[108.0, 126.0, 180.0, 198.0], "group 1 sees only its own (doubled) channel");
    }

    /// `group` not dividing the image's channel count is a named
    /// [`LowerError::UnsupportedShape`], never a silent truncation.
    #[test]
    fn grouped_conv_rejects_a_group_that_does_not_divide_channels() {
        let image = f32_initializer("image", &[1, 3, 4, 4], &[1.0; 48]);
        let weight = f32_initializer("weight", &[2, 1, 3, 3], &[1.0; 18]);
        let group_attribute = AttributeProto { name: "group", i: 2, ..AttributeProto::default() };
        let node = NodeProto {
            input: vec!["image", "weight"],
            output: vec!["y"],
            op_type: "Conv",
            name: "conv",
            attribute: vec![group_attribute],
            ..NodeProto::default()
        };
        let graph = GraphProto {
            node: vec![node],
            name: "grouped_conv_bad_group_graph",
            initializer: vec![image, weight],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let error = lower_graph(&graph).expect_err("group=2 does not evenly divide 3 image channels");
        assert!(matches!(error, LowerError::UnsupportedShape { .. }));
    }

    /// `Conv1d`, stride 1, no padding: [`lower_conv1d`]'s rank-3 mirror of
    /// [`conv2d_core`] -- a length-3 all-ones kernel over a length-5 signal
    /// is a 3-wide sliding-window sum, hand-verified.
    #[test]
    fn conv1d_stride1_no_pad_sums_each_3_wide_window() {
        let image = f32_initializer("image", &[1, 1, 5], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let weight = f32_initializer("weight", &[1, 1, 3], &[1.0; 3]);
        let node = NodeProto { input: vec!["image", "weight"], output: vec!["y"], op_type: "Conv", name: "conv1d", ..NodeProto::default() };
        let graph = GraphProto {
            node: vec![node],
            name: "conv1d_stride1_graph",
            initializer: vec![image, weight],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower Conv1d");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate Conv1d");
        let (data, shape) = evaluated.get(output).expect("y present");

        assert_eq!(shape, &[1, 1, 3]);
        assert_eq!(data, &[6.0, 9.0, 12.0], "hand-summed 3-wide windows over [1,2,3,4,5]");
    }

    /// `Conv1d`, stride 2 and `pads = [1, 1]`: exercises [`pad_axis`]'s
    /// zero padding together with the two-term window axis on the one
    /// spatial dimension, hand-verified against a manually zero-padded
    /// length-7 reference.
    #[test]
    fn conv1d_stride2_with_padding_sums_each_padded_window() {
        let image = f32_initializer("image", &[1, 1, 5], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let weight = f32_initializer("weight", &[1, 1, 3], &[1.0; 3]);
        let node = NodeProto {
            input: vec!["image", "weight"],
            output: vec!["y"],
            op_type: "Conv",
            name: "conv1d",
            attribute: vec![ints_attribute("strides", vec![2]), ints_attribute("pads", vec![1, 1])],
            ..NodeProto::default()
        };
        let graph = GraphProto {
            node: vec![node],
            name: "conv1d_stride2_pad1_graph",
            initializer: vec![image, weight],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower Conv1d stride2 pad1");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate Conv1d stride2 pad1");
        let (data, shape) = evaluated.get(output).expect("y present");

        // padded: [0, 1, 2, 3, 4, 5, 0] -- windows at offsets 0,2,4: [0,1,2]=3, [2,3,4]=9, [4,5,0]=9
        assert_eq!(shape, &[1, 1, 3]);
        assert_eq!(data, &[3.0, 9.0, 9.0], "hand-summed windows over the zero-padded length-7 signal");
    }

    /// `ConvTranspose`, stride 1, no padding, single channel: a `2x2` image
    /// against a `2x2` kernel produces a `3x3` "full" output --
    /// [`lower_convtranspose`]'s stride-1 equivalence to
    /// `Conv(pad(x, k-1), flip(w))`, hand-verified against the direct
    /// `out[oy,ox] = sum_{iy+ky=oy, ix+kx=ox} x[iy,ix]*w[ky,kx]` definition.
    #[test]
    fn convtranspose_stride1_no_pad_produces_the_full_output() {
        let image = f32_initializer("image", &[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let weight = f32_initializer("weight", &[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let node = NodeProto { input: vec!["image", "weight"], output: vec!["y"], op_type: "ConvTranspose", name: "convtranspose", ..NodeProto::default() };
        let graph = GraphProto {
            node: vec![node],
            name: "convtranspose_stride1_graph",
            initializer: vec![image, weight],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower ConvTranspose");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate ConvTranspose");
        let (data, shape) = evaluated.get(output).expect("y present");

        assert_eq!(shape, &[1, 1, 3, 3]);
        assert_eq!(data, &[1.0, 4.0, 4.0, 6.0, 20.0, 16.0, 9.0, 24.0, 16.0], "hand-derived full-output overlap-add");
    }

    /// `ConvTranspose` with `strides = [2, 2]`: the general masked-reduce
    /// scatter [`convtranspose2d_scatter`] composes -- a `2x2` image against
    /// a `2x2` kernel produces the "spread apart, overlap-add" `4x4` output
    /// the direct `out[iy*stride+ky, ix*stride+kx] += x[iy,ix]*w[ky,kx]`
    /// scatter definition gives, hand-verified per output cell.
    #[test]
    fn convtranspose_stride_two_scatters_each_input_into_a_spread_output() {
        let image = f32_initializer("image", &[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let weight = f32_initializer("weight", &[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let node = NodeProto {
            input: vec!["image", "weight"],
            output: vec!["y"],
            op_type: "ConvTranspose",
            name: "convtranspose",
            attribute: vec![ints_attribute("strides", vec![2, 2])],
            ..NodeProto::default()
        };
        let graph = GraphProto {
            node: vec![node],
            name: "convtranspose_stride2_graph",
            initializer: vec![image, weight],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower ConvTranspose stride=2");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate ConvTranspose stride=2");
        let (data, shape) = evaluated.get(output).expect("y present");

        assert_eq!(shape, &[1, 1, 4, 4]);
        assert_eq!(
            data,
            &[1.0, 2.0, 2.0, 4.0, 3.0, 4.0, 6.0, 8.0, 3.0, 6.0, 4.0, 8.0, 9.0, 12.0, 12.0, 16.0],
            "each input cell scattered into its own non-overlapping 2x2 kernel block"
        );
    }

    /// `Conv` with `auto_pad = SAME_UPPER`: [`same_pad_axis`]'s formula
    /// computes `pads = [1, 1]` for a length-4 signal, 3-wide kernel,
    /// stride 1 (`needed = 2`, split evenly `1/1`) -- the output stays
    /// length-4, hand-verified against the zero-padded length-6 reference.
    #[test]
    fn conv1d_auto_pad_same_upper_keeps_the_input_length() {
        let image = f32_initializer("image", &[1, 1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let weight = f32_initializer("weight", &[1, 1, 3], &[1.0; 3]);
        let node = NodeProto {
            input: vec!["image", "weight"],
            output: vec!["y"],
            op_type: "Conv",
            name: "conv1d_same",
            attribute: vec![AttributeProto { name: "auto_pad", s: b"SAME_UPPER", ..AttributeProto::default() }],
            ..NodeProto::default()
        };
        let graph = GraphProto {
            node: vec![node],
            name: "conv1d_same_upper_graph",
            initializer: vec![image, weight],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower Conv1d SAME_UPPER");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate Conv1d SAME_UPPER");
        let (data, shape) = evaluated.get(output).expect("y present");

        // padded (pad=[1,1]): [0,1,2,3,4,0] -- windows: [0,1,2]=3 [1,2,3]=6 [2,3,4]=9 [3,4,0]=7
        assert_eq!(shape, &[1, 1, 4]);
        assert_eq!(data, &[3.0, 6.0, 9.0, 7.0], "SAME_UPPER's even split pads both edges by one");
    }

    /// `Conv1d`, `group = 2` (depthwise): each of 2 input channels convolves
    /// against its own single-channel kernel, output channels concatenated
    /// -- the rank-3 mirror of the grouped `Conv2d` test above, hand-verified
    /// per channel.
    #[test]
    fn conv1d_grouped_convolves_each_channel_independently() {
        let image = f32_initializer("image", &[1, 2, 4], &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]);
        let weight = f32_initializer("weight", &[2, 1, 2], &[1.0, 1.0, 2.0, 2.0]);
        let node = NodeProto {
            input: vec!["image", "weight"],
            output: vec!["y"],
            op_type: "Conv",
            name: "conv1d_grouped",
            attribute: vec![AttributeProto { name: "group", i: 2, ..AttributeProto::default() }],
            ..NodeProto::default()
        };
        let graph = GraphProto {
            node: vec![node],
            name: "conv1d_grouped_graph",
            initializer: vec![image, weight],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower grouped Conv1d");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate grouped Conv1d");
        let (data, shape) = evaluated.get(output).expect("y present");

        // channel 0: [1,2,3,4] * kernel [1,1] -> [3,5,7]; channel 1: [10,20,30,40] * kernel [2,2] -> [60,100,140]
        assert_eq!(shape, &[1, 2, 3]);
        assert_eq!(data, &[3.0, 5.0, 7.0, 60.0, 100.0, 140.0], "each group convolves its own channel with its own kernel");
    }

    /// `Conv`, rank-5 (3D), stride 1, no padding: [`lower_conv3d`]'s rank-5
    /// mirror of [`conv2d_core`] -- a `2`-deep all-ones kernel sliding over a
    /// `3`-deep, otherwise unit, signal is a 2-wide sliding-window sum along
    /// the depth axis, hand-verified.
    #[test]
    fn conv3d_stride1_no_pad_sums_each_2_deep_window() {
        let image = f32_initializer("image", &[1, 1, 3, 1, 1], &[1.0, 2.0, 3.0]);
        let weight = f32_initializer("weight", &[1, 1, 2, 1, 1], &[1.0, 1.0]);
        let node = NodeProto { input: vec!["image", "weight"], output: vec!["y"], op_type: "Conv", name: "conv3d", ..NodeProto::default() };
        let graph = GraphProto {
            node: vec![node],
            name: "conv3d_stride1_graph",
            initializer: vec![image, weight],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower Conv3d");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate Conv3d");
        let (data, shape) = evaluated.get(output).expect("y present");

        assert_eq!(shape, &[1, 1, 2, 1, 1]);
        assert_eq!(data, &[3.0, 5.0], "hand-summed 2-deep windows over the depth axis [1,2,3]");
    }

    /// `MaxPool`, rank-5 (3D): [`lower_maxpool3d`]'s rank-5 mirror of
    /// [`lower_maxpool`] -- a 2-deep window sliding over a 3-deep signal
    /// takes the max of each adjacent pair, hand-verified.
    #[test]
    fn maxpool3d_takes_the_max_of_each_2_deep_window() {
        let image = f32_initializer("image", &[1, 1, 3, 1, 1], &[1.0, 3.0, 2.0]);
        let node = NodeProto {
            input: vec!["image"],
            output: vec!["y"],
            op_type: "MaxPool",
            name: "maxpool3d",
            attribute: vec![ints_attribute("kernel_shape", vec![2, 1, 1])],
            ..NodeProto::default()
        };
        let graph = GraphProto {
            node: vec![node],
            name: "maxpool3d_graph",
            initializer: vec![image],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower MaxPool3d");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate MaxPool3d");
        let (data, shape) = evaluated.get(output).expect("y present");

        assert_eq!(shape, &[1, 1, 2, 1, 1]);
        assert_eq!(data, &[3.0, 3.0], "max of each 2-deep window over [1,3,2]: max(1,3)=3, max(3,2)=3");
    }

    /// `AveragePool`, rank-5 (3D): [`lower_averagepool3d`]'s rank-5 mirror
    /// of [`lower_averagepool`] -- a 2-deep window sliding over a 3-deep
    /// signal takes the mean of each adjacent pair, hand-verified.
    #[test]
    fn averagepool3d_takes_the_mean_of_each_2_deep_window() {
        let image = f32_initializer("image", &[1, 1, 3, 1, 1], &[1.0, 3.0, 5.0]);
        let node = NodeProto {
            input: vec!["image"],
            output: vec!["y"],
            op_type: "AveragePool",
            name: "averagepool3d",
            attribute: vec![ints_attribute("kernel_shape", vec![2, 1, 1])],
            ..NodeProto::default()
        };
        let graph = GraphProto {
            node: vec![node],
            name: "averagepool3d_graph",
            initializer: vec![image],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower AveragePool3d");
        let named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let output = lowered.graph_outputs[0].1;
        let evaluated = evaluate_named(&lowered.program, &[], &named, &[output]).expect("evaluate AveragePool3d");
        let (data, shape) = evaluated.get(output).expect("y present");

        assert_eq!(shape, &[1, 1, 2, 1, 1]);
        assert_eq!(data, &[2.0, 4.0], "mean of each 2-deep window over [1,3,5]: (1+3)/2=2, (3+5)/2=4");
    }
}
