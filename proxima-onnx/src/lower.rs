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
#[derive(Debug, Clone)]
struct Value {
    node: NodeId,
    shape: Vec<u64>,
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

    for tensor in &graph.initializer {
        let shape = tensor_shape(tensor);
        let data = decode_numeric_tensor(tensor)?;
        let node = append(
            &mut program,
            Op::Input {
                dtype: onnx_dtype_to_op_dtype(tensor.data_type),
                shape: shape.iter().map(|&extent| Extent::Static(extent as u32)).collect(),
                name: Some(tensor.name.to_string()),
            },
        );
        values.insert(tensor.name.to_string(), Value { node, shape });
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
        values.insert(input.name.to_string(), Value { node, shape });
        graph_inputs.push(input.name.to_string());
    }

    if graph.node.is_empty() {
        return Err(LowerError::EmptyGraph { name: graph.name.to_string() });
    }

    for node in &graph.node {
        lower_node(&mut program, &mut values, node)?;
    }

    let mut graph_outputs = Vec::new();
    for output in &graph.output {
        let value = lookup_by_name(&values, output.name, "graph_output", graph.name)?;
        graph_outputs.push((output.name.to_string(), value.node));
    }

    Ok(Lowered { program, initializers, graph_inputs, graph_outputs })
}

fn lower_node(program: &mut Vec<Op>, values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>) -> Result<(), LowerError> {
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
        "MatMul" => lower_matmul(program, values, node),
        "Gemm" => lower_gemm(program, values, node),
        "Softmax" => lower_softmax(program, values, node),
        "Transpose" => lower_transpose(program, values, node),
        "Gather" => lower_gather(program, values, node),
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
        values.insert((*name).to_string(), Value { node: id, shape });
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

fn lower_binary(program: &mut Vec<Op>, values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>, body: ScalarOp) -> Result<(), LowerError> {
    let lhs = lookup(values, node, 0)?.clone();
    let rhs = lookup(values, node, 1)?.clone();
    let out_shape = broadcast_shapes(node, &lhs.shape, &rhs.shape)?;
    let operands = alloc::vec![
        (lhs.node, IndexMap::Affine(broadcast_pattern(&lhs.shape, &out_shape))),
        (rhs.node, IndexMap::Affine(broadcast_pattern(&rhs.shape, &out_shape))),
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

/// `MatMul` for rank-2 operands: `Reduce(+)` over `Elementwise(*)`,
/// node-for-node the composition `proxima-tensor/src/lib.rs`'s own module
/// doc builds and evaluates. Batched (rank > 2) `MatMul` is out of scope --
/// see this crate's own doc for why (deferred, not a sufficiency gap).
fn lower_matmul(program: &mut Vec<Op>, values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>) -> Result<(), LowerError> {
    let lhs = lookup(values, node, 0)?.clone();
    let rhs = lookup(values, node, 1)?.clone();
    if lhs.shape.len() != 2 || rhs.shape.len() != 2 {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: "MatMul lowering supports rank-2 operands only (no batch dimensions)".to_string(),
        });
    }
    if lhs.shape[1] != rhs.shape[0] {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: format!("contracted dim mismatch: lhs is [{}, {}], rhs is [{}, {}]", lhs.shape[0], lhs.shape[1], rhs.shape[0], rhs.shape[1]),
        });
    }
    let out_shape = alloc::vec![lhs.shape[0], rhs.shape[1]];
    let id = matmul2d(program, lhs.node, projection(3, &[0, 2]), rhs.node, projection(3, &[2, 1]), Some("matmul".to_string()));
    bind_output(values, node, 0, id, out_shape);
    Ok(())
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

/// `Softmax` along the last axis of a rank-2 input: max-shift, `exp`,
/// sum-reduce, divide -- the same four-step composition
/// `proxima-tensor/src/spec.rs`'s attention blocks build inline (see this
/// crate's own module doc pointer to `spec.rs`). Any axis but the last, or
/// any rank but 2, is out of scope -- deferred, not a sufficiency gap (the
/// same max/exp/sum/div composition works over any reduced axis; only the
/// `IndexPattern`s generalizing to it were not written in this pass).
fn lower_softmax(program: &mut Vec<Op>, values: &mut BTreeMap<String, Value>, node: &NodeProto<'_>) -> Result<(), LowerError> {
    let input = lookup(values, node, 0)?.clone();
    if input.shape.len() != 2 {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: "Softmax lowering supports rank-2 (batch, class) input only".to_string(),
        });
    }
    let rank = 2usize;
    let axis = attr_int(node, "axis").unwrap_or(-1);
    let normalized_axis = if axis < 0 { axis + rank as i64 } else { axis };
    if normalized_axis != 1 {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: "Softmax lowering supports reducing the last axis only".to_string(),
        });
    }

    let row_max = build_reduce(program, ScalarOp::Maximum, ReduceInit::NegativeInfinity, input.node, projection(2, &[0, 1]), projection(2, &[0]), None);
    let shifted = build_elementwise(
        program,
        ScalarOp::Subtract,
        alloc::vec![(input.node, IndexMap::Affine(identity_pattern(rank))), (row_max, IndexMap::Affine(projection(2, &[0])))],
    );
    let exponentiated = build_elementwise(program, ScalarOp::Exponential, alloc::vec![(shifted, IndexMap::Affine(identity_pattern(rank)))]);
    let row_sum = build_reduce(program, ScalarOp::Add, ReduceInit::Zero, exponentiated, projection(2, &[0, 1]), projection(2, &[0]), None);
    let id = build_elementwise(
        program,
        ScalarOp::Divide,
        alloc::vec![(exponentiated, IndexMap::Affine(identity_pattern(rank))), (row_sum, IndexMap::Affine(projection(2, &[0])))],
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

/// `Gather(data, indices, axis=0)`: [`IndexMap::Computed`] over
/// [`ScalarOp::Identity`], the same "gather (read-side)" row
/// `proxima-tensor/src/map.rs`'s doc table names and
/// `proxima-tensor/src/spec.rs`'s `embedding_lookup` builds for the
/// `axis=0`, rank-2-table case -- generalized here to any `data` rank and
/// any `indices` rank. `axis != 0` is deferred (the same `Computed`
/// mechanism reaches it with a permuted `base` pattern; not implemented in
/// this pass).
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
    let axis = attr_int(node, "axis").unwrap_or(0);
    let normalized_axis = if axis < 0 { axis + data.shape.len() as i64 } else { axis };
    if normalized_axis != 0 {
        return Err(LowerError::UnsupportedShape {
            name: node.name.to_string(),
            op_type: node.op_type.to_string(),
            reason: "Gather lowering supports axis=0 only".to_string(),
        });
    }

    let indices_rank = indices.shape.len();
    let data_rank = data.shape.len();
    let iter_rank = (indices_rank + data_rank - 1) as u16;
    let index_map = projection(iter_rank, &(0..indices_rank as u16).collect::<Vec<_>>());

    let mut base_axes: Vec<AxisIndex> = alloc::vec![AxisIndex::default()];
    for (position, _) in (1..data_rank).enumerate() {
        let iter_axis = (indices_rank + position) as u16;
        base_axes.push(AxisIndex { terms: core::iter::once(AxisTerm::projection(iter_axis)).collect(), offset: 0 });
    }
    let base = IndexPattern { iter_rank, axes: base_axes };
    let gathered_map = IndexMap::Computed { indices: indices.node, index_map, base, gathered_dim: 0 };

    let id = append(program, Op::Elementwise { dtype: DType::Float32, body: ScalarOp::Identity, operands: alloc::vec![(data.node, gathered_map)], name: None });
    let mut out_shape = indices.shape.clone();
    out_shape.extend_from_slice(&data.shape[1..]);
    bind_output(values, node, 0, id, out_shape);
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
