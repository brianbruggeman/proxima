//! `proxima_tensor::Op` program -> ONNX wire bytes: the mirror image of
//! [`crate::lower`], and the reason this half returns bytes rather than an
//! owned [`crate::messages::GraphProto`].
//!
//! [`crate::messages`]'s structs are zero-copy *parse* targets by design --
//! `&'a str`/`&'a [u8]` borrowing straight from the wire buffer (see that
//! module's own doc). Lift has no wire buffer yet; it is synthesizing new
//! names and tensor bytes that do not exist anywhere before this call runs.
//! Building a `GraphProto<'a>` around that data would need either an owned
//! sibling of every message type (doubling this crate's surface for a
//! write-only convenience) or leaking the owned strings to manufacture a
//! `'static` borrow (a hidden allocation-that-never-frees, ruled out by the
//! no-`unsafe`-to-fake-a-lifetime discipline this crate already holds).
//! [`crate::writer`]'s `push_*` primitives build wire bytes directly from
//! borrowed `&str`/`&[i64]`/... without ever owning a message struct, so
//! this module reuses exactly those (marked `pub(crate)` for this purpose)
//! to assemble `NodeProto`/`TensorProto`/`GraphProto` fragments the same
//! way this crate's own `tests.rs` fixture builders already do by hand.
//! [`lift_graph`] returns the serialized `GraphProto` field payload;
//! [`lift_model`] wraps that in a minimal `ModelProto` (`ir_version` plus
//! that `graph` field) -- together they are "lift, then serialize" fused
//! into the only shape this crate's types make sound.
//!
//! # Faithful, primitive-to-primitive
//!
//! Every [`Op`] form lowers to the same primitive ONNX operators
//! [`crate::lower`] already reads back (`Add`/`Mul`/`Max`/... for
//! [`ScalarOp`], `ReduceSum`/`ReduceMax`/`ReduceMin`/`ReduceProd` for
//! [`Reduce`], `Transpose`/`Unsqueeze`/`Gather` for [`IndexMap`]
//! addressing). A contracted `IndexMap` (the shape `lower::matmul2d`
//! builds) lifts as `Mul` + `ReduceSum`, never `MatMul` -- raising
//! primitives back into a fused op is deferred pattern-raising, out of
//! scope here (see this crate's own crate-level doc).
//!
//! # Coverage and gaps
//!
//! All 17 [`ScalarOp`] bodies map to a primitive ONNX op (see
//! [`scalar_op_type`]). [`Reduce`] bodies restricted to the associative set
//! ONNX ships a primitive reducer for (`Add`/`Multiply`/`Maximum`/
//! `Minimum`); anything else, and any `Keep::Scan` body but `Add`
//! (`CumSum`), is a documented [`LiftError`], never a fabricated op.
//! [`IndexMap::Affine`] axes restricted to unit-coefficient, zero-offset
//! projections (transpose/broadcast/identity -- everything
//! [`crate::lower`]'s own compositions produce); a convolution-shaped axis
//! (nonunit coefficient, the one case [`crate::map`]'s own doc table names
//! that this crate's `lower` module never emits) is a documented gap, not a
//! silent truncation. [`IndexMap::Computed`] is supported only in the exact
//! axis-0 gather shape [`crate::lower::lower_gather`] itself produces.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use proxima_tensor::{AxisIndex, Extent, IndexMap, IndexPattern, Keep, NodeId, Op, ScalarOp};
use thiserror::Error;

use crate::writer::{push_i32, push_i64, push_len, push_packed_f32, push_packed_i64, push_str};

/// Everything a faithful, primitive-to-primitive lift cannot express for a
/// given [`Op`] program -- see this module's doc for the scope each variant
/// marks as deferred versus a genuine ONNX expressiveness gap (there are
/// none of the latter for the 17 [`ScalarOp`] bodies; see [`scalar_op_type`]).
#[derive(Debug, Error)]
pub enum LiftError {
    #[error("node {node}: symbolic extent has no faithful ONNX concrete shape")]
    SymbolicExtent { node: NodeId },
    #[error("node {node}: index pattern axis has a non-unit coefficient or nonzero offset (convolution-shaped) -- no single faithful ONNX op expresses this")]
    NonAffineAxis { node: NodeId },
    #[error("node {node}: index pattern has a broadcast (zero-term) axis on a nonzero-rank operand -- deferred, not attempted by this pass")]
    DegenerateBroadcastAxis { node: NodeId },
    #[error("node {node}: reduce/scan out_map is not order-preserving -- deferred, not attempted by this pass")]
    PermutedOutMap { node: NodeId },
    #[error("node {node}: reduce body {body:?} has no faithful primitive ONNX reduce op")]
    UnsupportedReduceBody { node: NodeId, body: ScalarOp },
    #[error("node {node}: Keep::Scan reduce body {body:?} has no faithful ONNX cumulative op, or spans more than one axis")]
    UnsupportedScan { node: NodeId, body: ScalarOp },
    #[error("node {node}: data-dependent (Computed) index map is out of scope except the axis-0 gather shape lower_gather produces")]
    UnsupportedComputedMap { node: NodeId },
    #[error("input node {node} ({name:?}) is neither a declared graph input nor a declared initializer")]
    UnboundInput { node: NodeId, name: String },
    #[error("initializer {name:?} has no data among the initializers this lift was given")]
    MissingInitializerData { name: String },
}

/// Mirrors [`crate::lower::Lowered`] in reverse: everything [`lift_graph`]
/// needs to turn a program back into a graph -- which names are graph
/// inputs, which are initializers (with their data), and which [`NodeId`]s
/// are the declared outputs.
#[derive(Debug)]
pub struct LiftInput<'a> {
    pub program: &'a [Op],
    pub initializers: &'a [(String, Vec<f32>)],
    pub graph_inputs: &'a [String],
    pub graph_outputs: &'a [(String, NodeId)],
    pub graph_name: &'a str,
}

/// The 17 [`ScalarOp`] bodies, each mapped to the primitive ONNX op with
/// identical semantics. Total -- every arm of the closed `ScalarOp` enum is
/// covered, so this can never itself be a [`LiftError`] source.
fn scalar_op_type(body: ScalarOp) -> &'static str {
    match body {
        ScalarOp::Identity => "Identity",
        ScalarOp::Add => "Add",
        ScalarOp::Subtract => "Sub",
        ScalarOp::Multiply => "Mul",
        ScalarOp::Divide => "Div",
        ScalarOp::Maximum => "Max",
        ScalarOp::Minimum => "Min",
        ScalarOp::Negate => "Neg",
        ScalarOp::Reciprocal => "Reciprocal",
        ScalarOp::Exponential => "Exp",
        ScalarOp::Logarithm => "Log",
        ScalarOp::SquareRoot => "Sqrt",
        ScalarOp::Tanh => "Tanh",
        ScalarOp::Erf => "Erf",
        ScalarOp::Greater => "Greater",
        ScalarOp::Equal => "Equal",
        ScalarOp::Select => "Where",
    }
}

/// The associative bodies ONNX ships a primitive reducer for.
fn reduce_op_type(body: ScalarOp) -> Option<&'static str> {
    match body {
        ScalarOp::Add => Some("ReduceSum"),
        ScalarOp::Multiply => Some("ReduceProd"),
        ScalarOp::Maximum => Some("ReduceMax"),
        ScalarOp::Minimum => Some("ReduceMin"),
        _ => None,
    }
}

fn extents_to_dims(node: NodeId, shape: &[Extent]) -> Result<Vec<i64>, LiftError> {
    shape
        .iter()
        .map(|extent| match extent {
            Extent::Static(value) => Ok(i64::from(*value)),
            Extent::Symbolic(_) => Err(LiftError::SymbolicExtent { node }),
        })
        .collect()
}

fn attr_ints(name: &str, values: &[i64]) -> Vec<u8> {
    let mut buf = Vec::new();
    push_str(1, name, &mut buf);
    push_i32(20, 7, &mut buf); // AttributeType::Ints
    push_packed_i64(8, values, &mut buf);
    buf
}

fn attr_int(name: &str, value: i64) -> Vec<u8> {
    let mut buf = Vec::new();
    push_str(1, name, &mut buf);
    push_i32(20, 2, &mut buf); // AttributeType::Int
    push_i64(3, value, &mut buf);
    buf
}

fn attr_tensor(name: &str, tensor_bytes: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    push_str(1, name, &mut buf);
    push_i32(20, 4, &mut buf); // AttributeType::Tensor
    push_len(5, tensor_bytes, &mut buf);
    buf
}

/// `TensorProto` bytes for a float32 tensor materialized in full (`dims`,
/// `data_type = 1`, `float_data`, `name`) -- the shape every initializer and
/// every [`Op::Constant`] this pass emits takes.
fn float_tensor_bytes(dims: &[i64], name: &str, values: &[f32]) -> Vec<u8> {
    let mut buf = Vec::new();
    push_packed_i64(1, dims, &mut buf);
    push_i32(2, 1, &mut buf);
    push_packed_f32(4, values, &mut buf);
    push_str(8, name, &mut buf);
    buf
}

/// `ValueInfoProto` bytes for a float32 tensor of concrete rank -- the
/// shape every graph input this pass emits takes.
fn value_info_bytes(name: &str, dims: &[i64]) -> Vec<u8> {
    let mut shape = Vec::new();
    for dim in dims {
        let mut dimension = Vec::new();
        push_i64(1, *dim, &mut dimension);
        push_len(1, &dimension, &mut shape);
    }
    let mut tensor_type = Vec::new();
    push_i32(1, 1, &mut tensor_type);
    push_len(2, &shape, &mut tensor_type);
    let mut type_proto = Vec::new();
    push_len(1, &tensor_type, &mut type_proto);
    let mut buf = Vec::new();
    push_str(1, name, &mut buf);
    push_len(2, &type_proto, &mut buf);
    buf
}

/// One assembled `NodeProto`, pushed as a length-delimited `node` field
/// (field 1) directly into the caller's graph buffer.
fn emit_node(buf: &mut Vec<u8>, inputs: &[&str], outputs: &[&str], name: &str, op_type: &str, attributes: &[Vec<u8>]) {
    let mut node = Vec::new();
    for input in inputs {
        push_str(1, input, &mut node);
    }
    for output in outputs {
        push_str(2, output, &mut node);
    }
    push_str(3, name, &mut node);
    push_str(4, op_type, &mut node);
    for attribute in attributes {
        push_len(5, attribute, &mut node);
    }
    push_len(1, &node, buf);
}

/// One operand axis's affine reading, decomposed for [`resolve_affine`]:
/// which iteration axis it reads (`None` for a zero-term broadcast axis).
fn single_term_axis(axis_index: &AxisIndex) -> Option<u16> {
    match axis_index.terms.as_slice() {
        [term] if term.coeff == 1 && axis_index.offset == 0 => Some(term.axis),
        _ => None,
    }
}

/// Resolves one elementwise/reduce operand's [`IndexPattern`] against a
/// source value already bound to `source_name`, inserting whatever
/// `Transpose`/`Unsqueeze` prelude nodes are needed and returning the name
/// the consuming node should read.
///
/// Every operand axis must carry exactly one unit-coefficient, zero-offset
/// term (a plain projection) -- see [`single_term_axis`] -- covering
/// `pattern.axes.len()` of the `pattern.iter_rank` iteration axes. Sorting
/// those covered axes ascending gives the operand-axis order a `Transpose`
/// must first produce (identity if already ascending); the iteration axes
/// left uncovered are exactly where `Unsqueeze` inserts a size-1 dimension,
/// which is sound because ONNX's own numpy-broadcast rule then fills every
/// remaining axis the way [`proxima_tensor`]'s own `broadcast_pattern`
/// (`lower.rs`) already assumes.
fn resolve_affine(node: NodeId, buf: &mut Vec<u8>, fresh: &mut u32, source_name: &str, pattern: &IndexPattern) -> Result<String, LiftError> {
    let mut covered: Vec<(u16, u16)> = Vec::with_capacity(pattern.axes.len());
    for (operand_axis, axis_index) in pattern.axes.iter().enumerate() {
        match single_term_axis(axis_index) {
            Some(iter_axis) => covered.push((iter_axis, operand_axis as u16)),
            None if axis_index.terms.is_empty() => {
                return Err(LiftError::DegenerateBroadcastAxis { node });
            }
            None => return Err(LiftError::NonAffineAxis { node }),
        }
    }
    covered.sort_by_key(|&(iter_axis, _)| iter_axis);
    let order: Vec<u16> = covered.iter().map(|&(_, operand_axis)| operand_axis).collect();
    let missing: Vec<i64> = (0..pattern.iter_rank).filter(|iter_axis| !covered.iter().any(|&(candidate, _)| candidate == *iter_axis)).map(i64::from).collect();

    let is_identity_order = order.iter().enumerate().all(|(index, &axis)| axis as usize == index);
    let mut current = source_name.to_string();

    if !is_identity_order {
        *fresh += 1;
        let transposed = format!("lift_transpose_{}", *fresh);
        let perm: Vec<i64> = order.iter().map(|&axis| i64::from(axis)).collect();
        emit_node(buf, &[current.as_str()], &[transposed.as_str()], &transposed, "Transpose", &[attr_ints("perm", &perm)]);
        current = transposed;
    }

    if !missing.is_empty() {
        *fresh += 1;
        let unsqueezed = format!("lift_unsqueeze_{}", *fresh);
        emit_node(buf, &[current.as_str()], &[unsqueezed.as_str()], &unsqueezed, "Unsqueeze", &[attr_ints("axes", &missing)]);
        current = unsqueezed;
    }

    Ok(current)
}

fn reduced_axes(node: NodeId, in_map: &IndexPattern, out_map: &IndexPattern) -> Result<Vec<i64>, LiftError> {
    let mut out_covered: Vec<u16> = Vec::with_capacity(out_map.axes.len());
    for axis_index in &out_map.axes {
        match single_term_axis(axis_index) {
            Some(iter_axis) => out_covered.push(iter_axis),
            None => return Err(LiftError::PermutedOutMap { node }),
        }
    }
    let ascending = out_covered.windows(2).all(|pair| pair[0] < pair[1]);
    if !ascending {
        return Err(LiftError::PermutedOutMap { node });
    }
    Ok((0..in_map.iter_rank).filter(|axis| !out_covered.contains(axis)).map(i64::from).collect())
}

/// Serialize an [`Op`] program back into ONNX `GraphProto` wire bytes (the
/// `graph` field's payload, unwrapped -- see this module's doc for why
/// bytes, not an owned [`crate::messages::GraphProto`]).
///
/// # Errors
///
/// A [`LiftError`] naming the first [`Op`] this faithful, primitive-only
/// pass cannot express -- see this module's doc for the coverage table and
/// what is a genuine gap versus deferred scope.
pub fn lift_graph(input: LiftInput<'_>) -> Result<Vec<u8>, LiftError> {
    let mut names: Vec<String> = Vec::with_capacity(input.program.len());
    let mut nodes = Vec::new();
    let mut initializers = Vec::new();
    let mut graph_inputs = Vec::new();
    let mut fresh: u32 = 0;

    let mut output_names_by_node: alloc::collections::BTreeMap<u32, Vec<String>> = alloc::collections::BTreeMap::new();
    for (output_name, node_id) in input.graph_outputs {
        output_names_by_node.entry(node_id.0).or_default().push(output_name.clone());
    }

    for (index, op) in input.program.iter().enumerate() {
        let node_id = NodeId(index as u32);
        let primary_name = match output_names_by_node.get(&node_id.0).and_then(|list| list.first()) {
            Some(declared) => declared.clone(),
            None => match op {
                Op::Input { name: Some(bound_name), .. } => bound_name.clone(),
                _ => format!("v{index}"),
            },
        };

        match op {
            Op::Input { shape, name, .. } => {
                let bound_name = name.clone().ok_or_else(|| LiftError::UnboundInput { node: node_id, name: primary_name.clone() })?;
                let dims = extents_to_dims(node_id, shape)?;
                if input.graph_inputs.iter().any(|candidate| candidate == &bound_name) {
                    graph_inputs.push(value_info_bytes(&bound_name, &dims));
                } else if let Some((_, data)) = input.initializers.iter().find(|(candidate, _)| candidate == &bound_name) {
                    initializers.push(float_tensor_bytes(&dims, &bound_name, data));
                } else {
                    return Err(LiftError::MissingInitializerData { name: bound_name });
                }
            }
            Op::Constant { shape, value, .. } => {
                let dims = extents_to_dims(node_id, shape)?;
                let element_count: usize = dims.iter().map(|dim| *dim as usize).product::<usize>().max(1);
                let values: Vec<f32> = alloc::vec![*value; element_count];
                let tensor = float_tensor_bytes(&dims, &primary_name, &values);
                emit_node(&mut nodes, &[], &[primary_name.as_str()], &primary_name, "Constant", &[attr_tensor("value", &tensor)]);
            }
            Op::Iota { extent, .. } => {
                let Extent::Static(count) = extent else {
                    return Err(LiftError::SymbolicExtent { node: node_id });
                };
                let values: Vec<f32> = (0..*count).map(|value| value as f32).collect();
                initializers.push(float_tensor_bytes(&[i64::from(*count)], &primary_name, &values));
            }
            Op::Elementwise { body, operands, .. } => {
                let mut operand_names: Vec<String> = Vec::with_capacity(operands.len());
                for (operand_id, map) in operands {
                    let source_name = names[operand_id.0 as usize].clone();
                    let resolved = match map {
                        IndexMap::Affine(pattern) => resolve_affine(node_id, &mut nodes, &mut fresh, &source_name, pattern)?,
                        IndexMap::Computed { indices, index_map, base, gathered_dim } => {
                            resolve_gather(node_id, &mut nodes, &mut fresh, &names, &source_name, *indices, index_map, base, *gathered_dim)?
                        }
                    };
                    operand_names.push(resolved);
                }
                let operand_refs: Vec<&str> = operand_names.iter().map(String::as_str).collect();
                emit_node(&mut nodes, &operand_refs, &[primary_name.as_str()], &primary_name, scalar_op_type(*body), &[]);
            }
            Op::Reduce(reduce) => {
                let in_pattern = match &reduce.in_map {
                    IndexMap::Affine(pattern) => pattern,
                    IndexMap::Computed { .. } => return Err(LiftError::UnsupportedComputedMap { node: node_id }),
                };
                let out_pattern = match &reduce.out_map {
                    IndexMap::Affine(pattern) => pattern,
                    IndexMap::Computed { .. } => return Err(LiftError::UnsupportedComputedMap { node: node_id }),
                };
                let source_name = names[reduce.operand.0 as usize].clone();
                let resolved = resolve_affine(node_id, &mut nodes, &mut fresh, &source_name, in_pattern)?;
                let axes = reduced_axes(node_id, in_pattern, out_pattern)?;

                match reduce.keep {
                    Keep::Reduce => {
                        let op_type = reduce_op_type(reduce.body).ok_or(LiftError::UnsupportedReduceBody { node: node_id, body: reduce.body })?;
                        emit_node(&mut nodes, &[resolved.as_str()], &[primary_name.as_str()], &primary_name, op_type, &[attr_ints("axes", &axes), attr_int("keepdims", 0)]);
                    }
                    Keep::Scan => {
                        if reduce.body != ScalarOp::Add || axes.len() != 1 {
                            return Err(LiftError::UnsupportedScan { node: node_id, body: reduce.body });
                        }
                        fresh += 1;
                        let axis_name = format!("lift_cumsum_axis_{fresh}");
                        initializers.push({
                            let mut buf = Vec::new();
                            push_packed_i64(1, &[], &mut buf);
                            push_i32(2, 7, &mut buf);
                            push_packed_i64(7, &axes, &mut buf);
                            push_str(8, &axis_name, &mut buf);
                            buf
                        });
                        emit_node(&mut nodes, &[resolved.as_str(), axis_name.as_str()], &[primary_name.as_str()], &primary_name, "CumSum", &[]);
                    }
                }
            }
        }

        if let Some(extra_output_names) = output_names_by_node.get(&node_id.0) {
            for extra in extra_output_names.iter().skip(1) {
                emit_node(&mut nodes, &[primary_name.as_str()], &[extra.as_str()], extra, "Identity", &[]);
            }
        }

        names.push(primary_name);
    }

    let mut graph = Vec::new();
    graph.extend_from_slice(&nodes);
    push_str(2, input.graph_name, &mut graph);
    for initializer in &initializers {
        push_len(5, initializer, &mut graph);
    }
    for graph_input in &graph_inputs {
        push_len(11, graph_input, &mut graph);
    }
    for (output_name, _) in input.graph_outputs {
        let mut value_info = Vec::new();
        push_str(1, output_name, &mut value_info);
        push_len(12, &value_info, &mut graph);
    }
    Ok(graph)
}

/// The axis-0 gather shape [`crate::lower::lower_gather`] produces: `base`'s
/// entry at `gathered_dim` is empty by construction (see [`IndexMap`]'s own
/// doc), so only `gathered_dim == 0` with an otherwise order-preserving
/// `base` is attempted; anything else is [`LiftError::UnsupportedComputedMap`].
#[allow(clippy::too_many_arguments)]
fn resolve_gather(
    node: NodeId,
    buf: &mut Vec<u8>,
    fresh: &mut u32,
    names: &[String],
    data_name: &str,
    indices: NodeId,
    index_map: &IndexPattern,
    base: &IndexPattern,
    gathered_dim: u16,
) -> Result<String, LiftError> {
    if gathered_dim != 0 {
        return Err(LiftError::UnsupportedComputedMap { node });
    }
    let indices_rank = index_map.iter_rank;
    for (position, axis_index) in base.axes.iter().enumerate().skip(1) {
        let expected = indices_rank + (position as u16 - 1);
        if single_term_axis(axis_index) != Some(expected) {
            return Err(LiftError::UnsupportedComputedMap { node });
        }
    }
    let indices_name = names[indices.0 as usize].clone();
    *fresh += 1;
    let output_name = format!("lift_gather_{}", *fresh);
    emit_node(buf, &[data_name, indices_name.as_str()], &[output_name.as_str()], &output_name, "Gather", &[attr_int("axis", 0)]);
    Ok(output_name)
}

/// Wraps [`lift_graph`]'s output in a minimal `ModelProto` (`ir_version = 8`,
/// no opset/producer metadata) -- the full "lift, then serialize" round trip
/// [`crate::tests`]'s writable round-trip test drives end to end.
///
/// # Errors
///
/// Whatever [`lift_graph`] returns.
pub fn lift_model(input: LiftInput<'_>) -> Result<Vec<u8>, LiftError> {
    let graph = lift_graph(input)?;
    let mut model = Vec::new();
    push_i64(1, 8, &mut model);
    push_len(7, &graph, &mut model);
    Ok(model)
}
