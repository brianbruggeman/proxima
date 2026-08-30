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

    for node in &graph.node {
        lower_node(&mut program, &mut values, &initializer_data, node)?;
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
        "Constant" => lower_constant(program, values, node),
        "ReduceSum" => lower_reduce(program, values, node, ScalarOp::Add, ReduceInit::Zero),
        "ReduceMax" => lower_reduce(program, values, node, ScalarOp::Maximum, ReduceInit::NegativeInfinity),
        "ReduceMin" => lower_reduce(program, values, node, ScalarOp::Minimum, ReduceInit::PositiveInfinity),
        "ReduceProd" => lower_reduce(program, values, node, ScalarOp::Multiply, ReduceInit::One),
        "Reshape" => lower_reshape(values, initializer_data, node),
        "Flatten" => lower_flatten(values, node),
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
fn lower_constant(program: &mut Vec<Op>, values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>) -> Result<(), LowerError> {
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
    bind_output(values, node, 0, id, shape);
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
        let node = NodeProto { input: vec!["x"], output: vec!["y"], op_type: "Concat", name: "concat", ..NodeProto::default() };
        let graph = GraphProto {
            node: vec![node],
            name: "unsupported_graph",
            initializer: vec![x_initializer],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let error = lower_graph(&graph).expect_err("Concat has no lowering");
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
}
