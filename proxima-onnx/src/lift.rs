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
//! # Faithful, primitive-to-primitive -- with one named pattern raised back
//!
//! Every [`Op`] form lowers to the same primitive ONNX operators
//! [`crate::lower`] already reads back (`Add`/`Mul`/`Max`/... for
//! [`ScalarOp`], `ReduceSum`/`ReduceMax`/`ReduceMin`/`ReduceProd` for
//! [`Reduce`], `Transpose`/`Unsqueeze`/`Gather` for [`IndexMap`]
//! addressing). This is the default and the fallback for everything this
//! module does not specifically recognize.
//!
//! [`try_matmul_shape`] is the one exception: a `Reduce(Add)`-over-
//! `Elementwise(Multiply)` in the exact rank-2, untransposed, unbatched
//! shape `lower::lower_matmul` and `lower::matmul2d` (via
//! `lower::lower_gemm`, untransposed) both build lifts as a single ONNX
//! `MatMul` node, not `Mul` + `ReduceSum`. Batched or transposed
//! contractions, and the materialized-window `Reduce` [`crate::lower::conv2d_core`]
//! builds (which would raise to a named `Conv`), still fall through to the
//! faithful primitive lift -- further, genuinely expressible pattern-raising
//! deferred here for lack of time, not because the shape resists it.
//!
//! # Coverage and gaps
//!
//! All 17 [`ScalarOp`] bodies map to a primitive ONNX op (see
//! [`scalar_op_type`]). [`Reduce`] bodies restricted to the associative set
//! ONNX ships a primitive reducer for (`Add`/`Multiply`/`Maximum`/
//! `Minimum`); anything else, and any `Keep::Scan` body but `Add`
//! (`CumSum`), is a documented [`LiftError`], never a fabricated op -- ONNX
//! has no `CumProd`/`CumMax`/`CumMin` primitive, so those stay a genuine,
//! named gap rather than a fabricated composition.
//!
//! [`IndexMap::Affine`] axes: a plain unit-coefficient, zero-offset
//! projection (transpose/broadcast/identity) resolves directly.
//! A **two-term** axis (`coeff_a * iter[a] + coeff_b * iter[b] + offset`,
//! the convolution/pooling window shape [`crate::map`]'s own doc table
//! names) resolves via [`resolve_affine`]'s window case: a `Constant`
//! integer index table plus a `Gather`, reading off the two iteration
//! axes' extents from a *sibling* operand of the same node that exposes
//! them as a plain projection against a leaf ([`Op::Input`]/
//! [`Op::Constant`]) with a statically known shape -- exactly the
//! `unify_iteration_space` technique `crate::lower`'s `window_materialize`
//! doc describes for shape inference, reused here in reverse. A
//! **degenerate** (zero-term) axis -- a size-1 operand dimension ONNX's own
//! numpy-broadcast rule already handles -- resolves by keeping that
//! dimension in place (no `Gather` needed) at the right-aligned iteration
//! axis `crate::lower`'s `broadcast_pattern` would have assigned it.
//! Anything wider (three or more terms, or a two-term axis whose iteration
//! extents cannot be recovered from a sibling operand) is
//! [`LiftError::NonAffineAxis`], a genuine residual: no single ONNX op
//! expresses an arbitrary affine combination without more context than the
//! node carries.
//!
//! A [`Reduce`]'s `out_map` need not be axis-ascending: [`reduced_axes`]
//! reads off the surviving axes' order and, when it differs from the
//! `ReduceSum`/... op's own ascending output order, lifts a trailing
//! `Transpose` to restore it -- the inverse of the identity `reduced_axes`
//! already relied on shape inference to prove sound on the read side.
//!
//! [`IndexMap::Computed`] is supported for the order-preserving gather shape
//! [`crate::lower::lower_gather`] produces at any axis (see [`resolve_gather`])
//! -- output order `data.shape[:axis] + indices.shape + data.shape[axis+1:]`,
//! lifted as a single `Gather(data, indices, axis)`. This shape also covers
//! [`crate::lower::pad_axis`]'s zero-padded reads and [`crate::lower::concat_pair`]'s
//! per-leg reads (both a rank-1-indices specialization of the same pattern),
//! so a padded `Conv`/`Pool`/`Concat` round-trips as `Gather` + `Where`. Scatter
//! (a data-dependent *output* map) is unsupported upstream in
//! `proxima_tensor` itself and so never reaches this module.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use proxima_tensor::{AxisIndex, DType, Extent, IndexMap, IndexPattern, Keep, NodeId, Op, ScalarOp};
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
    #[error("node {node}: index pattern axis has three or more terms, or a two-term (window) axis whose iteration extents no sibling operand exposes -- no single faithful ONNX op expresses this")]
    NonAffineAxis { node: NodeId },
    #[error("node {node}: reduce/scan out_map axis is not a plain projection, or repeats an iteration axis -- no single faithful ONNX op expresses this")]
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

/// `TensorProto` bytes for an int64 tensor materialized in full (`dims`,
/// `data_type = 7`, `int64_data`, `name`) -- the shape a [`resolve_affine`]
/// window axis's `Gather` index table takes (same layout the `CumSum` axis
/// initializer already uses further down this file).
fn int64_tensor_bytes(dims: &[i64], name: &str, values: &[i64]) -> Vec<u8> {
    let mut buf = Vec::new();
    push_packed_i64(1, dims, &mut buf);
    push_i32(2, 7, &mut buf);
    push_packed_i64(7, values, &mut buf);
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

/// A recognized `Reduce(+)`-over-`Elementwise(*)` matmul shape (see this
/// module's own doc, "pattern-raising"): the two rank-2 operands feeding a
/// plain-`MatMul`-shaped contraction, and the `Elementwise` node they are
/// consumed through -- never separately emitted, since [`lift_graph`] folds
/// it directly into the `MatMul` this shape names.
struct MatmulShape {
    elementwise_node: NodeId,
    lhs: NodeId,
    rhs: NodeId,
}

/// Recognizes the exact `Reduce(Add)`-over-`Elementwise(Multiply)` shape
/// `lower::lower_matmul` (batch-free) and `lower::matmul2d`
/// (`lower::lower_gemm`, untransposed) both build: an iteration space `(M,
/// K, N)` in any axis order, `lhs` reading `(M, K)` in that order, `rhs`
/// reading `(K, N)` in that order, the reduce contracting `K` and outputting
/// `(M, N)` in that order -- each a pure, unit-coefficient, zero-offset
/// projection over a rank-2 operand (no batch, no transpose, no broadcast).
/// Anything else (batched, transposed, broadcast, a non-`Add`/`Multiply`
/// body) returns `None`, and [`lift_graph`] falls through to the faithful
/// primitive `Mul` + `ReduceSum` lift this module's doc names as the
/// default. Raising a transposed or batched contraction to `MatMul`/`Gemm`
/// is a further, genuinely expressible refinement of this same idea,
/// deferred here for lack of time, not because the shape resists it.
fn try_matmul_shape(program: &[Op], reduce: &proxima_tensor::Reduce) -> Option<MatmulShape> {
    if reduce.body != ScalarOp::Add || reduce.keep != Keep::Reduce {
        return None;
    }
    let IndexMap::Affine(in_pattern) = &reduce.in_map else { return None };
    let IndexMap::Affine(out_pattern) = &reduce.out_map else { return None };
    if in_pattern.iter_rank != 3 || in_pattern.axes.len() != 3 {
        return None;
    }
    for (index, axis) in in_pattern.axes.iter().enumerate() {
        if single_term_axis(axis) != Some(index as u16) {
            return None;
        }
    }

    let elementwise_node = reduce.operand;
    let Some(Op::Elementwise { body: ScalarOp::Multiply, operands, .. }) = program.get(elementwise_node.0 as usize) else {
        return None;
    };
    let [(lhs, IndexMap::Affine(lhs_pattern)), (rhs, IndexMap::Affine(rhs_pattern))] = operands.as_slice() else {
        return None;
    };
    if lhs_pattern.iter_rank != 3 || lhs_pattern.axes.len() != 2 || rhs_pattern.iter_rank != 3 || rhs_pattern.axes.len() != 2 {
        return None;
    }
    let lhs_m = single_term_axis(&lhs_pattern.axes[0])?;
    let lhs_k = single_term_axis(&lhs_pattern.axes[1])?;
    let rhs_k = single_term_axis(&rhs_pattern.axes[0])?;
    let rhs_n = single_term_axis(&rhs_pattern.axes[1])?;
    if lhs_k != rhs_k {
        return None;
    }
    let mut axes = [lhs_m, lhs_k, rhs_n];
    axes.sort_unstable();
    if axes != [0, 1, 2] {
        return None;
    }

    if out_pattern.iter_rank != 3 || out_pattern.axes.len() != 2 {
        return None;
    }
    let out_first = single_term_axis(&out_pattern.axes[0])?;
    let out_second = single_term_axis(&out_pattern.axes[1])?;
    if out_first != lhs_m || out_second != rhs_n {
        return None;
    }

    Some(MatmulShape { elementwise_node, lhs: *lhs, rhs: *rhs })
}

/// How many times each [`NodeId`] is read as an operand anywhere in
/// `program`, or named as a declared graph output -- [`lift_graph`] only
/// folds a matched [`MatmulShape::elementwise_node`] into its `MatMul`
/// (skipping that node's own primitive emission) when this count is
/// exactly one, so a multiply reused by a second consumer (or itself a
/// declared output) still gets its faithful, independently-referenceable
/// `Mul` node.
fn count_consumers(program: &[Op], graph_outputs: &[(String, NodeId)]) -> BTreeMap<u32, u32> {
    let mut counts: BTreeMap<u32, u32> = BTreeMap::new();
    let bump = |counts: &mut BTreeMap<u32, u32>, node: NodeId| *counts.entry(node.0).or_insert(0) += 1;
    let bump_map = |counts: &mut BTreeMap<u32, u32>, map: &IndexMap| {
        if let IndexMap::Computed { indices, .. } = map {
            bump(counts, *indices);
        }
    };
    for op in program {
        match op {
            Op::Elementwise { operands, .. } => {
                for (operand, map) in operands {
                    bump(&mut counts, *operand);
                    bump_map(&mut counts, map);
                }
            }
            Op::Reduce(reduce) => {
                bump(&mut counts, reduce.operand);
                bump_map(&mut counts, &reduce.in_map);
                bump_map(&mut counts, &reduce.out_map);
            }
            Op::Input { .. } | Op::Constant { .. } | Op::Iota { .. } => {}
        }
    }
    for (_, node) in graph_outputs {
        bump(&mut counts, *node);
    }
    counts
}

/// A source node's statically known shape, when it is a leaf that carries
/// one directly ([`Op::Input`]/[`Op::Constant`] both hold `Vec<Extent>`) --
/// the only two node kinds [`known_iter_extents`] can read an extent off of
/// without running full shape inference.
fn leaf_shape(program: &[Op], node: NodeId) -> Option<&[Extent]> {
    match program.get(node.0 as usize)? {
        Op::Input { shape, .. } | Op::Constant { shape, .. } => Some(shape),
        _ => None,
    }
}
/// For every iteration axis a *sibling* operand exposes as a plain
/// projection against a leaf of known shape, its extent -- exactly the
/// technique `crate::lower`'s `window_materialize` doc names
/// (`unify_iteration_space` resolving an axis's extent only from a pure
/// projection), reused here so a two-term window axis on another operand of
/// the same node can size its `Gather` index table.
fn known_iter_extents(operands: &[(NodeId, IndexMap)], program: &[Op]) -> BTreeMap<u16, u64> {
    let mut extents = BTreeMap::new();
    for (operand_id, map) in operands {
        let IndexMap::Affine(pattern) = map else { continue };
        let Some(shape) = leaf_shape(program, *operand_id) else { continue };
        for (operand_axis, axis_index) in pattern.axes.iter().enumerate() {
            let Some(iter_axis) = single_term_axis(axis_index) else { continue };
            if let Some(Extent::Static(extent)) = shape.get(operand_axis) {
                extents.entry(iter_axis).or_insert(u64::from(*extent));
            }
        }
    }
    extents
}

/// Resolves one elementwise/reduce operand's [`IndexPattern`] against a
/// source value already bound to `source_name`, inserting whatever
/// `Gather`/`Transpose`/`Unsqueeze` prelude nodes (plus, for a window axis,
/// one fresh graph-level initializer) are needed and returning the name the
/// consuming node should read.
///
/// Three operand-axis shapes are handled, matched left to right against
/// `pattern.axes` (see this module's doc for the coverage table):
///
/// - a plain unit-coefficient, zero-offset term ([`single_term_axis`]) --
///   contributes one real axis, read directly;
/// - a zero-term (degenerate broadcast) axis -- contributes one real axis at
///   ONNX's own right-aligned broadcast position, read directly (its extent
///   is 1, so no `Gather` is needed; the axis is left in place for the
///   consuming op's numpy-broadcast rule to fill);
/// - a two-term window axis (`coeff_a * iter[a] + coeff_b * iter[b] +
///   offset`) -- resolved via a fresh int64 index-table initializer plus a
///   `Gather` along that axis, sized from [`known_iter_extents`]; a
///   [`LiftError::NonAffineAxis`] if either iteration axis's extent cannot
///   be recovered from a sibling operand.
///
/// Once every operand axis has a known real position and target iteration
/// axis, the same tail as before applies: sort by iteration axis ascending
/// to find the `Transpose` perm (identity if already ascending), then
/// `Unsqueeze` in whichever iteration axes no operand axis touched at all.
fn resolve_affine(
    node: NodeId,
    buf: &mut Vec<u8>,
    initializers: &mut Vec<Vec<u8>>,
    fresh: &mut u32,
    source_name: &str,
    pattern: &IndexPattern,
    known_extents: &BTreeMap<u16, u64>,
) -> Result<String, LiftError> {
    let mut current = source_name.to_string();
    let mut covered: Vec<(u16, u16)> = Vec::with_capacity(pattern.axes.len());
    let mut shift: i64 = 0;

    for (operand_axis, axis_index) in pattern.axes.iter().enumerate() {
        let real_axis = (operand_axis as i64 + shift) as u16;
        match axis_index.terms.as_slice() {
            [term] if term.coeff == 1 && axis_index.offset == 0 => {
                covered.push((term.axis, real_axis));
            }
            [] => {
                // right-aligned broadcast position `crate::lower`'s `broadcast_pattern` assigns.
                let leading = pattern.iter_rank as i64 - pattern.axes.len() as i64;
                let virtual_axis = (leading + operand_axis as i64) as u16;
                covered.push((virtual_axis, real_axis));
            }
            [term_a, term_b] => {
                let extent_a = *known_extents.get(&term_a.axis).ok_or(LiftError::NonAffineAxis { node })?;
                let extent_b = *known_extents.get(&term_b.axis).ok_or(LiftError::NonAffineAxis { node })?;
                let mut indices = Vec::with_capacity((extent_a * extent_b) as usize);
                for value_a in 0..extent_a as i64 {
                    for value_b in 0..extent_b as i64 {
                        indices.push(value_a * i64::from(term_a.coeff) + value_b * i64::from(term_b.coeff) + i64::from(axis_index.offset));
                    }
                }
                // a graph-level initializer, not a `Constant` node -- `crate::lower`'s
                // `Constant` lowering only reads back a *uniform* value (`Op::Constant`
                // carries one `f32`, never an array), so a real index table must arrive
                // the same way the `CumSum` axis initializer further down this file does.
                *fresh += 1;
                let indices_name = format!("lift_window_indices_{}", *fresh);
                initializers.push(int64_tensor_bytes(&[extent_a as i64, extent_b as i64], &indices_name, &indices));

                *fresh += 1;
                let gathered = format!("lift_window_gather_{}", *fresh);
                emit_node(buf, &[current.as_str(), indices_name.as_str()], &[gathered.as_str()], &gathered, "Gather", &[attr_int("axis", i64::from(real_axis))]);
                current = gathered;

                covered.push((term_a.axis, real_axis));
                covered.push((term_b.axis, real_axis + 1));
                shift += 1;
            }
            _ => return Err(LiftError::NonAffineAxis { node }),
        }
    }

    covered.sort_by_key(|&(iter_axis, _)| iter_axis);
    let order: Vec<u16> = covered.iter().map(|&(_, real_axis)| real_axis).collect();
    let missing: Vec<i64> = (0..pattern.iter_rank).filter(|iter_axis| !covered.iter().any(|&(candidate, _)| candidate == *iter_axis)).map(i64::from).collect();

    let is_identity_order = order.iter().enumerate().all(|(index, &axis)| axis as usize == index);

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

/// The reduced axes and, when the surviving axes are not already
/// ascending, the `Transpose` perm needed to restore [`Reduce::out_map`]'s
/// actual order -- `ReduceSum`/... always emit their surviving axes in
/// ascending order with `keepdims=0`, so a permuted `out_map` (a genuine
/// axis reorder, not a malformed one) is a faithful `Transpose` away rather
/// than a [`LiftError`].
struct ReducedAxes {
    axes: Vec<i64>,
    perm: Option<Vec<i64>>,
}

fn reduced_axes(node: NodeId, in_map: &IndexPattern, out_map: &IndexPattern) -> Result<ReducedAxes, LiftError> {
    let mut out_covered: Vec<u16> = Vec::with_capacity(out_map.axes.len());
    for axis_index in &out_map.axes {
        match single_term_axis(axis_index) {
            Some(iter_axis) => out_covered.push(iter_axis),
            None => return Err(LiftError::PermutedOutMap { node }),
        }
    }
    let mut ascending_order = out_covered.clone();
    ascending_order.sort_unstable();
    if ascending_order.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(LiftError::PermutedOutMap { node });
    }
    let axes = (0..in_map.iter_rank).filter(|axis| !out_covered.contains(axis)).map(i64::from).collect();
    let is_ascending = out_covered.windows(2).all(|pair| pair[0] < pair[1]);
    let perm = if is_ascending {
        None
    } else {
        let mapped: Vec<i64> = out_covered
            .iter()
            .map(|axis| ascending_order.iter().position(|candidate| candidate == axis).map(|position| position as i64))
            .collect::<Option<Vec<i64>>>()
            .ok_or(LiftError::PermutedOutMap { node })?;
        Some(mapped)
    };
    Ok(ReducedAxes { axes, perm })
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

    // Pattern-raising (this module's own doc): a matched `MatmulShape`'s
    // `elementwise_node` folds directly into the `MatMul` its consuming
    // `Reduce` emits, so it is never separately lifted as a primitive `Mul`
    // -- guarded by `consumer_counts` so a multiply reused by a second
    // consumer, or itself a declared output, still gets its own faithful
    // node.
    let consumer_counts = count_consumers(input.program, input.graph_outputs);
    let mut matmul_shapes: BTreeMap<u32, MatmulShape> = BTreeMap::new();
    let mut subsumed_elementwise: alloc::collections::BTreeSet<u32> = alloc::collections::BTreeSet::new();
    for (index, op) in input.program.iter().enumerate() {
        let Op::Reduce(reduce) = op else { continue };
        let Some(shape) = try_matmul_shape(input.program, reduce) else { continue };
        if consumer_counts.get(&shape.elementwise_node.0).copied().unwrap_or(0) != 1 {
            continue;
        }
        subsumed_elementwise.insert(shape.elementwise_node.0);
        matmul_shapes.insert(index as u32, shape);
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
            Op::Input { dtype, shape, name } => {
                let bound_name = name.clone().ok_or_else(|| LiftError::UnboundInput { node: node_id, name: primary_name.clone() })?;
                let dims = extents_to_dims(node_id, shape)?;
                if input.graph_inputs.iter().any(|candidate| candidate == &bound_name) {
                    graph_inputs.push(value_info_bytes(&bound_name, &dims));
                } else if let Some((_, data)) = input.initializers.iter().find(|(candidate, _)| candidate == &bound_name) {
                    // an `Op::Input` tagged `Int32` is a `Computed`-gather
                    // `indices` leaf (see `onnx_dtype_to_op_dtype`'s own doc);
                    // this crate stores every buffer as f32 regardless, so
                    // round-tripping it through `float_tensor_bytes` here
                    // would re-parse back to `DType::Float32` and trip
                    // `shape::infer`'s `check_indices_dtype` -- an int64
                    // tensor is the faithful ONNX encoding that survives.
                    if *dtype == DType::Float32 {
                        initializers.push(float_tensor_bytes(&dims, &bound_name, data));
                    } else {
                        // an integer-tagged `Op::Input` carries an exact-integer
                        // f32 value by construction (this module's own doc,
                        // `fold_constant_indices`'s doc) -- a direct cast, not
                        // `round()`, both keeps this tier-1-clean (`round()`
                        // needs `libm`/`std`, neither available here) and
                        // matches that invariant rather than papering over a
                        // violation of it.
                        let exact: Vec<i64> = data.iter().map(|&value| value as i64).collect();
                        initializers.push(int64_tensor_bytes(&dims, &bound_name, &exact));
                    }
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
            Op::Elementwise { body, operands, .. } if subsumed_elementwise.contains(&node_id.0) => {
                // Folded directly into the `MatMul` its consuming `Reduce`
                // emits below (see `try_matmul_shape`) -- `primary_name` is
                // still pushed to `names` past this match so index
                // alignment with `program` holds, but nothing references it.
                let _ = (body, operands);
            }
            Op::Elementwise { body, operands, .. } => {
                let known_extents = known_iter_extents(operands, input.program);
                let mut operand_names: Vec<String> = Vec::with_capacity(operands.len());
                for (operand_id, map) in operands {
                    let source_name = names[operand_id.0 as usize].clone();
                    let resolved = match map {
                        IndexMap::Affine(pattern) => resolve_affine(node_id, &mut nodes, &mut initializers, &mut fresh, &source_name, pattern, &known_extents)?,
                        IndexMap::Computed { indices, index_map, base, gathered_dim } => resolve_gather(
                            node_id,
                            &mut nodes,
                            &mut initializers,
                            &mut fresh,
                            input.program,
                            &names,
                            &source_name,
                            *indices,
                            index_map,
                            base,
                            *gathered_dim,
                        )?,
                    };
                    operand_names.push(resolved);
                }
                let operand_refs: Vec<&str> = operand_names.iter().map(String::as_str).collect();
                emit_node(&mut nodes, &operand_refs, &[primary_name.as_str()], &primary_name, scalar_op_type(*body), &[]);
            }
            Op::Reduce(reduce) if matmul_shapes.contains_key(&node_id.0) => {
                // Pattern-raised (this module's own doc): the shape
                // `try_matmul_shape` matched at `index` is a plain `MatMul`,
                // not `Mul` + `ReduceSum` -- `lhs`/`rhs` are each read by a
                // pure, unmodified projection (see that function's own doc),
                // so their already-emitted names are the faithful `MatMul`
                // operands directly, no `resolve_affine` transform needed.
                let shape = &matmul_shapes[&node_id.0];
                let lhs_name = names[shape.lhs.0 as usize].clone();
                let rhs_name = names[shape.rhs.0 as usize].clone();
                emit_node(&mut nodes, &[lhs_name.as_str(), rhs_name.as_str()], &[primary_name.as_str()], &primary_name, "MatMul", &[]);
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
                let resolved = resolve_affine(node_id, &mut nodes, &mut initializers, &mut fresh, &source_name, in_pattern, &BTreeMap::new())?;
                let reduce_axes = reduced_axes(node_id, in_pattern, out_pattern)?;
                let final_name = primary_name.clone();
                let staged_name = if reduce_axes.perm.is_some() {
                    fresh += 1;
                    format!("lift_reduce_staged_{fresh}")
                } else {
                    final_name.clone()
                };

                match reduce.keep {
                    Keep::Reduce => {
                        let op_type = reduce_op_type(reduce.body).ok_or(LiftError::UnsupportedReduceBody { node: node_id, body: reduce.body })?;
                        emit_node(&mut nodes, &[resolved.as_str()], &[staged_name.as_str()], &staged_name, op_type, &[attr_ints("axes", &reduce_axes.axes), attr_int("keepdims", 0)]);
                    }
                    Keep::Scan => {
                        if reduce.body != ScalarOp::Add || reduce_axes.axes.len() != 1 {
                            return Err(LiftError::UnsupportedScan { node: node_id, body: reduce.body });
                        }
                        fresh += 1;
                        let axis_name = format!("lift_cumsum_axis_{fresh}");
                        initializers.push({
                            let mut buf = Vec::new();
                            push_packed_i64(1, &[], &mut buf);
                            push_i32(2, 7, &mut buf);
                            push_packed_i64(7, &reduce_axes.axes, &mut buf);
                            push_str(8, &axis_name, &mut buf);
                            buf
                        });
                        emit_node(&mut nodes, &[resolved.as_str(), axis_name.as_str()], &[staged_name.as_str()], &staged_name, "CumSum", &[]);
                    }
                }

                if let Some(perm) = &reduce_axes.perm {
                    emit_node(&mut nodes, &[staged_name.as_str()], &[final_name.as_str()], &final_name, "Transpose", &[attr_ints("perm", perm)]);
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

/// One scalar step of [`fold_constant_indices`]'s tiny lift-time
/// interpreter -- only the bodies [`crate::lower::pad_axis`]'s and
/// [`crate::lower::concat_pair`]'s own clamp-index chains actually build
/// (`Subtract`/`Maximum`/`Minimum`, plus `Identity`/`Add`/`Multiply` for
/// generality); anything else falls through to `None`, propagating a real,
/// data-dependent operand out of the fold rather than fabricating a value.
fn eval_scalar_op(body: ScalarOp, operands: &[f32]) -> Option<f32> {
    match (body, operands) {
        (ScalarOp::Identity, [value]) => Some(*value),
        (ScalarOp::Add, [left, right]) => Some(left + right),
        (ScalarOp::Subtract, [left, right]) => Some(left - right),
        (ScalarOp::Multiply, [left, right]) => Some(left * right),
        (ScalarOp::Maximum, [left, right]) => Some(left.max(*right)),
        (ScalarOp::Minimum, [left, right]) => Some(left.min(*right)),
        _ => None,
    }
}

/// Evaluates `node`'s value at lift time when its whole upstream subgraph is
/// provably input-free -- `Op::Constant`/`Op::Iota` leaves and rank-<=1
/// `Op::Elementwise` arithmetic over them, exactly the shape
/// [`crate::lower::pad_axis`]'s clamped-index chain (`Iota` position,
/// `Subtract`/`Maximum`/`Minimum` against `Constant` scalars) and
/// [`crate::lower::concat_pair`]'s `lhs_index`/`rhs_index` chains build. This
/// crate stores every buffer as f32 regardless of logical dtype (this
/// module's own doc), so re-lowering the *unfolded* chain (plain ONNX
/// `Sub`/`Max`/`Min` nodes) would lose the `DType::Int32` tag
/// `shape::infer`'s `check_indices_dtype` requires on a `Computed`'s
/// `indices` -- folding to a concrete int64 initializer at lift time sides
/// that requirement rather than needing a `Cast` op this crate does not lift.
/// `None` propagates a genuine data-dependent `indices` operand (a real
/// embedding-lookup `Op::Input`) straight through to [`resolve_gather`]'s
/// ordinary named-reference path.
fn fold_constant_indices(program: &[Op], node: NodeId) -> Option<Vec<f32>> {
    match program.get(node.0 as usize)? {
        Op::Constant { shape, value, .. } => {
            let mut element_count: u64 = 1;
            for extent in shape {
                let Extent::Static(count) = extent else { return None };
                element_count = element_count.checked_mul(u64::from(*count))?;
            }
            Some(alloc::vec![*value; element_count as usize])
        }
        Op::Iota { extent: Extent::Static(count), .. } => Some((0..*count).map(|value| value as f32).collect()),
        Op::Elementwise { body, operands, .. } => {
            let mut folded_operands: Vec<Vec<f32>> = Vec::with_capacity(operands.len());
            for (operand_id, map) in operands {
                let IndexMap::Affine(pattern) = map else { return None };
                if pattern.iter_rank > 1 || pattern.axes.len() > 1 {
                    return None;
                }
                folded_operands.push(fold_constant_indices(program, *operand_id)?);
            }
            let length = folded_operands.iter().map(Vec::len).max()?;
            let mut result = Vec::with_capacity(length);
            for position in 0..length {
                let values: Vec<f32> = folded_operands.iter().map(|values| if values.len() == 1 { values[0] } else { values[position] }).collect();
                result.push(eval_scalar_op(*body, &values)?);
            }
            Some(result)
        }
        Op::Iota { .. } | Op::Input { .. } | Op::Reduce(_) => None,
    }
}

/// The exact shape [`crate::lower::lower_gather`] produces at any axis --
/// output order `data.shape[:axis] + indices.shape + data.shape[axis+1:]` --
/// which is also the shape [`crate::lower::pad_axis`]'s and
/// [`crate::lower::concat_pair`]'s own `Computed` gathers collapse to for a
/// rank-1 `indices` (their `base` is [`crate::lower::concat_base_pattern`],
/// identity everywhere but `gathered_dim`, which is exactly what this
/// function's `base` check reduces to when `indices_rank == 1`). So a single
/// faithful ONNX `Gather(data, indices, axis=gathered_dim)` covers a real
/// embedding-style lookup, a padded `Conv`/`Pool` read, and a `Concat` leg
/// alike -- the validity `Select`/`Where` (`pad_axis`'s clamp-fill, `Concat`'s
/// side-pick) that wraps the gather is a *separate* `Op::Elementwise`, lifted
/// by the ordinary `ScalarOp::Select -> "Where"` path already in
/// [`scalar_op_type`]. Any other `index_map`/`base` shape --
/// non-order-preserving indices axes, a `base` that reads a data axis out of
/// order -- has no single faithful ONNX op and stays
/// [`LiftError::UnsupportedComputedMap`].
#[allow(clippy::too_many_arguments)]
fn resolve_gather(
    node: NodeId,
    buf: &mut Vec<u8>,
    initializers: &mut Vec<Vec<u8>>,
    fresh: &mut u32,
    program: &[Op],
    names: &[String],
    data_name: &str,
    indices: NodeId,
    index_map: &IndexPattern,
    base: &IndexPattern,
    gathered_dim: u16,
) -> Result<String, LiftError> {
    let indices_rank = index_map.axes.len() as u16;
    for (position, axis_index) in index_map.axes.iter().enumerate() {
        let expected = gathered_dim + position as u16;
        if single_term_axis(axis_index) != Some(expected) {
            return Err(LiftError::UnsupportedComputedMap { node });
        }
    }
    for (data_axis, axis_index) in base.axes.iter().enumerate() {
        let data_axis = data_axis as u16;
        if data_axis == gathered_dim {
            continue;
        }
        let expected = if data_axis < gathered_dim { data_axis } else { data_axis - 1 + indices_rank };
        if single_term_axis(axis_index) != Some(expected) {
            return Err(LiftError::UnsupportedComputedMap { node });
        }
    }

    let indices_name = if indices_rank == 1 {
        match fold_constant_indices(program, indices) {
            Some(values) => {
                // exact-integer-valued by construction (`fold_constant_indices`'s
                // own doc) -- a direct cast, not `round()`, keeps this tier-1
                // clean (`round()` needs `libm`/`std`, neither available here).
                let exact: Vec<i64> = values.iter().map(|&value| value as i64).collect();
                *fresh += 1;
                let folded_name = format!("lift_gather_indices_{}", *fresh);
                initializers.push(int64_tensor_bytes(&[exact.len() as i64], &folded_name, &exact));
                folded_name
            }
            None => names[indices.0 as usize].clone(),
        }
    } else {
        names[indices.0 as usize].clone()
    };

    *fresh += 1;
    let output_name = format!("lift_gather_{}", *fresh);
    emit_node(buf, &[data_name, indices_name.as_str()], &[output_name.as_str()], &output_name, "Gather", &[attr_int("axis", i64::from(gathered_dim))]);
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec;

    use proxima_tensor::{ReduceInit, append, cpu::evaluate_named};

    use super::*;
    use crate::lower::lower_graph;
    use crate::messages::{AttributeProto, GraphProto, NodeProto, TensorProto, ValueInfoProto};
    use crate::pipe::parse_complete;

    fn f32_initializer(name: &'static str, dims: &[i64], data: &[f32]) -> TensorProto<'static> {
        TensorProto { dims: dims.to_vec(), data_type: 1, float_data: data.to_vec(), name, ..TensorProto::default() }
    }

    fn ints_attribute(name: &'static str, ints: Vec<i64>) -> AttributeProto<'static> {
        AttributeProto { name, ints, ..AttributeProto::default() }
    }

    /// Drives the full write-side loop this crate's ONNX support closes,
    /// mirroring `crate::tests`'s own `op_program_lifts_to_onnx_bytes_and_lowers_back_to_an_equivalent_program`:
    /// `onnx graph -> lower -> Op` (baseline evaluation), then
    /// `Op -> lift -> onnx bytes -> lower -> Op` (round-tripped evaluation),
    /// asserting the two agree to `1e-4`.
    fn assert_graph_round_trips_through_lift(graph: &GraphProto<'_>, lifted_name: &'static str) {
        let original = lower_graph(graph).expect("lower the fixture graph");
        let original_named: Vec<(&str, &[f32])> = original.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let original_output = original.graph_outputs[0].1;
        let baseline = evaluate_named(&original.program, &[], &original_named, &[original_output]).expect("evaluate the original program");
        let (baseline_data, baseline_shape) = baseline.get(original_output).expect("baseline output present");

        let lift_input = LiftInput {
            program: &original.program,
            initializers: &original.initializers,
            graph_inputs: &original.graph_inputs,
            graph_outputs: &original.graph_outputs,
            graph_name: lifted_name,
        };
        let lifted_bytes = lift_model(lift_input).expect("lift the program to onnx bytes");

        let reparsed_model = parse_complete(&lifted_bytes).expect("lifted bytes parse back to a ModelProto");
        let reparsed_graph = reparsed_model.graph.as_ref().expect("lifted graph present");
        assert!(!reparsed_graph.node.is_empty(), "lifted graph carries its primitive-op nodes");

        let reloaded = lower_graph(reparsed_graph).expect("lower the lifted graph back to Op");
        let reloaded_named: Vec<(&str, &[f32])> = reloaded.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        let reloaded_output = reloaded.graph_outputs[0].1;
        let evaluated = evaluate_named(&reloaded.program, &[], &reloaded_named, &[reloaded_output]).expect("evaluate the round-tripped program");
        let (data, shape) = evaluated.get(reloaded_output).expect("round-tripped output present");

        assert_eq!(shape, baseline_shape, "lift round trip changed the output shape");
        for (actual, expected) in data.iter().zip(baseline_data.iter()) {
            assert!((actual - expected).abs() < 1e-4, "round-tripped {actual} does not match baseline {expected}");
        }
    }

    /// The convolution-shaped gap this module's doc names: `Conv`'s
    /// `window_materialize`d `Elementwise` operand carries a two-term
    /// `stride*out + dilation*kernel` axis on `image` -- [`resolve_affine`]'s
    /// `Constant` index table plus `Gather` is the only new machinery this
    /// closes, so a bare stride-1, no-padding `Conv` (no `pad_axis` gather
    /// in the way -- see this test's own doc for why padding stays out of
    /// scope) is the direct proof it lifts and lowers back to the same
    /// 3x3-window sums `crate::lower`'s own `conv_stride1_no_pad_sums_each_3x3_window`
    /// hand-verifies.
    #[test]
    fn conv_stride1_window_axis_round_trips_through_lift() {
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

        assert_graph_round_trips_through_lift(&graph, "conv_stride1_lifted");
    }

    /// `MaxPool`, 2x2 kernel and stride: the same window-axis machinery as
    /// `Conv`, but feeding a `Keep::Reduce` `Maximum` body instead of `Add`
    /// -- proves [`resolve_affine`]'s window case is shared, not
    /// `Conv`-specific.
    #[test]
    fn maxpool_2x2_window_axis_round_trips_through_lift() {
        let image = f32_initializer("image", &[1, 1, 4, 4], &[1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0, 9.0, 11.0, 10.0, 12.0, 13.0, 15.0, 14.0, 16.0]);
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

        assert_graph_round_trips_through_lift(&graph, "maxpool_lifted");
    }

    /// The degenerate-broadcast gap: `Add`'s `lower_binary` builds a
    /// zero-term axis for any operand dimension that is `1` where the
    /// broadcast output is wider (`crate::lower`'s `broadcast_pattern`) --
    /// here the right-hand `[1, 3]` operand against a `[2, 3]` left-hand
    /// one. [`resolve_affine`]'s degenerate case keeps that dimension in
    /// place rather than erroring, letting ONNX's own numpy-broadcast rule
    /// on the emitted `Add` fill it back in.
    #[test]
    fn degenerate_broadcast_axis_round_trips_through_lift() {
        let lhs = f32_initializer("lhs", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rhs = f32_initializer("rhs", &[1, 3], &[10.0, 20.0, 30.0]);
        let node = NodeProto { input: vec!["lhs", "rhs"], output: vec!["y"], op_type: "Add", name: "add", ..NodeProto::default() };
        let graph = GraphProto {
            node: vec![node],
            name: "broadcast_add_graph",
            initializer: vec![lhs, rhs],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        assert_graph_round_trips_through_lift(&graph, "broadcast_add_lifted");
    }

    /// The permuted-`out_map` gap: a hand-built [`proxima_tensor::Op`]
    /// program (no ONNX op composition `crate::lower` builds ever produces
    /// this shape, so it is constructed directly rather than through
    /// `lower_graph`) whose `Reduce` keeps iteration axes `{0, 2}` of a
    /// rank-3 input but lists them `out_map`-order `[2, 0]` -- the surviving
    /// axes transposed relative to `ReduceSum`'s own ascending output order.
    /// [`reduced_axes`]'s `perm` plus a trailing `Transpose` is the only new
    /// machinery this closes.
    #[test]
    fn permuted_reduce_out_map_round_trips_through_lift() {
        let mut program = Vec::new();
        let x = append(
            &mut program,
            Op::Input { dtype: proxima_tensor::DType::Float32, shape: vec![Extent::Static(2), Extent::Static(3), Extent::Static(4)], name: Some("x".to_string()) },
        );
        let in_map = IndexMap::Affine(IndexPattern {
            iter_rank: 3,
            axes: vec![
                AxisIndex { terms: core::iter::once(proxima_tensor::AxisTerm::projection(0)).collect(), offset: 0 },
                AxisIndex { terms: core::iter::once(proxima_tensor::AxisTerm::projection(1)).collect(), offset: 0 },
                AxisIndex { terms: core::iter::once(proxima_tensor::AxisTerm::projection(2)).collect(), offset: 0 },
            ],
        });
        let out_map = IndexMap::Affine(IndexPattern {
            iter_rank: 3,
            axes: vec![
                AxisIndex { terms: core::iter::once(proxima_tensor::AxisTerm::projection(2)).collect(), offset: 0 },
                AxisIndex { terms: core::iter::once(proxima_tensor::AxisTerm::projection(0)).collect(), offset: 0 },
            ],
        });
        let reduced = append(
            &mut program,
            Op::Reduce(proxima_tensor::Reduce {
                dtype: proxima_tensor::DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: x,
                in_map,
                out_map,
                keep: Keep::Reduce,
                name: None,
            }),
        );

        let x_data: [f32; 24] = core::array::from_fn(|index| index as f32);
        let named: [(&str, &[f32]); 1] = [("x", &x_data)];
        let baseline = evaluate_named(&program, &[], &named, &[reduced]).expect("evaluate the permuted-out_map fixture");
        let (baseline_data, baseline_shape) = baseline.get(reduced).expect("baseline output present");
        assert_eq!(baseline_shape, &[4, 2], "out_map [2, 0] transposes the naturally-ascending [2, 4] reduce shape");

        let lift_input = LiftInput {
            program: &program,
            initializers: &[],
            graph_inputs: &["x".to_string()],
            graph_outputs: &[("y".to_string(), reduced)],
            graph_name: "permuted_reduce_lifted",
        };
        let lifted_bytes = lift_model(lift_input).expect("lift the permuted-out_map program to onnx bytes");

        let reparsed_model = parse_complete(&lifted_bytes).expect("lifted bytes parse back to a ModelProto");
        let reparsed_graph = reparsed_model.graph.as_ref().expect("lifted graph present");
        let reloaded = lower_graph(reparsed_graph).expect("lower the lifted permuted-reduce graph back to Op");
        let reloaded_named: [(&str, &[f32]); 1] = [("x", &x_data)];
        let reloaded_output = reloaded.graph_outputs[0].1;
        let evaluated = evaluate_named(&reloaded.program, &[], &reloaded_named, &[reloaded_output]).expect("evaluate the round-tripped permuted-reduce program");
        let (data, shape) = evaluated.get(reloaded_output).expect("round-tripped output present");

        assert_eq!(shape, baseline_shape);
        for (actual, expected) in data.iter().zip(baseline_data.iter()) {
            assert!((actual - expected).abs() < 1e-4, "round-tripped {actual} does not match baseline {expected}");
        }
    }

    /// A three-term axis has no faithful ONNX composition
    /// [`resolve_affine`] attempts -- this stays [`LiftError::NonAffineAxis`],
    /// never a silent truncation to the first two terms.
    #[test]
    fn three_term_axis_is_a_named_lift_error_not_a_silent_truncation() {
        let x = append(
            &mut Vec::new(),
            Op::Input { dtype: proxima_tensor::DType::Float32, shape: vec![], name: None },
        );
        let three_terms = AxisIndex {
            terms: vec![proxima_tensor::AxisTerm::scaled(0, 1), proxima_tensor::AxisTerm::scaled(1, 1), proxima_tensor::AxisTerm::scaled(2, 1)].into_iter().collect(),
            offset: 0,
        };
        let pattern = IndexPattern { iter_rank: 3, axes: vec![three_terms] };
        let mut buf = Vec::new();
        let mut fresh = 0u32;
        let mut initializers = Vec::new();
        let error = resolve_affine(x, &mut buf, &mut initializers, &mut fresh, "source", &pattern, &BTreeMap::new()).expect_err("three-term axis is unsupported");
        assert!(matches!(error, LiftError::NonAffineAxis { .. }));
    }

    /// `Gather(data, indices, axis=1)` on a `[2, 3]` table -- the mirror of
    /// `crate::lower`'s own `gather_selects_columns_by_index_at_a_general_axis`
    /// fixture, proving [`resolve_gather`]'s generalization beyond
    /// `gathered_dim == 0` closes the write side too.
    #[test]
    fn gather_at_a_general_axis_round_trips_through_lift() {
        let table = f32_initializer("table", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let indices = TensorProto { dims: vec![2], data_type: 7, int64_data: vec![2, 0], name: "ids", ..TensorProto::default() };
        let axis_attribute = AttributeProto { name: "axis", i: 1, ..AttributeProto::default() };
        let node =
            NodeProto { input: vec!["table", "ids"], output: vec!["y"], op_type: "Gather", name: "gather", attribute: vec![axis_attribute], ..NodeProto::default() };
        let graph = GraphProto {
            node: vec![node],
            name: "gather_axis1_graph",
            initializer: vec![table, indices],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        assert_graph_round_trips_through_lift(&graph, "gather_axis1_lifted");
    }

    /// A padded `Conv` -- `pads = [1, 1, 1, 1]`, stride 2 -- exercises
    /// [`crate::lower::pad_axis`]'s zero-fill `Computed` gather (rank-1
    /// indices, `gathered_dim` at each spatial axis) wrapped in a validity
    /// `Select`, on top of `resolve_affine`'s window-axis `Gather` this
    /// module already closed. This is the closed write-side residual: before
    /// [`resolve_gather`]'s generalization, `pad_axis`'s non-zero-axis
    /// `Computed` gather was [`LiftError::UnsupportedComputedMap`], so a
    /// padded `Conv`/`Pool` could not be lifted back to ONNX at all.
    #[test]
    fn padded_conv_round_trips_through_lift() {
        let image = f32_initializer("image", &[1, 1, 4, 4], &(1..=16).map(|value| value as f32).collect::<Vec<_>>());
        let weight = f32_initializer("weight", &[1, 1, 3, 3], &[1.0; 9]);
        let node = NodeProto {
            input: vec!["image", "weight"],
            output: vec!["y"],
            op_type: "Conv",
            name: "conv",
            attribute: vec![ints_attribute("pads", vec![1, 1, 1, 1]), ints_attribute("strides", vec![2, 2])],
            ..NodeProto::default()
        };
        let graph = GraphProto {
            node: vec![node],
            name: "padded_conv_graph",
            initializer: vec![image, weight],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        assert_graph_round_trips_through_lift(&graph, "padded_conv_lifted");
    }

    /// `Keep::Scan` with a `Multiply` body has no ONNX cumulative-product
    /// primitive (`CumSum` is the only one ONNX ships) -- stays a named
    /// [`LiftError::UnsupportedScan`], never fabricated as `CumSum`.
    #[test]
    fn non_add_scan_body_is_a_named_lift_error() {
        let mut program = Vec::new();
        let x = append(&mut program, Op::Input { dtype: proxima_tensor::DType::Float32, shape: vec![Extent::Static(4)], name: Some("x".to_string()) });
        let identity = IndexMap::Affine(IndexPattern {
            iter_rank: 1,
            axes: vec![AxisIndex { terms: core::iter::once(proxima_tensor::AxisTerm::projection(0)).collect(), offset: 0 }],
        });
        let scanned = append(
            &mut program,
            Op::Reduce(proxima_tensor::Reduce {
                dtype: proxima_tensor::DType::Float32,
                body: ScalarOp::Multiply,
                init: ReduceInit::One,
                operand: x,
                in_map: identity.clone(),
                out_map: identity,
                keep: Keep::Scan,
                name: None,
            }),
        );

        let lift_input = LiftInput {
            program: &program,
            initializers: &[],
            graph_inputs: &["x".to_string()],
            graph_outputs: &[("y".to_string(), scanned)],
            graph_name: "non_add_scan",
        };
        let error = lift_model(lift_input).expect_err("Multiply scan body has no faithful ONNX cumulative op");
        assert!(matches!(error, LiftError::UnsupportedScan { .. }));
    }

    /// Pattern-raising (this module's own doc, "Faithful, primitive-to-
    /// primitive" and [`try_matmul_shape`]): a lowered `MatMul` lifts back
    /// to a single ONNX `MatMul` node, never the primitive `Mul` +
    /// `ReduceSum` this module's default composes -- and the intermediate
    /// `Elementwise(Multiply)` [`try_matmul_shape`] folds into it is never
    /// separately emitted either. Round-trip correctness reuses
    /// [`assert_graph_round_trips_through_lift`]; this test additionally
    /// inspects the lifted node list directly.
    #[test]
    fn matmul_lifts_to_a_matmul_node_not_mul_reducesum() {
        let lhs = f32_initializer("lhs", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rhs = f32_initializer("rhs", &[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let node = NodeProto { input: vec!["lhs", "rhs"], output: vec!["y"], op_type: "MatMul", name: "matmul", ..NodeProto::default() };
        let graph = GraphProto {
            node: vec![node],
            name: "matmul_graph",
            initializer: vec![lhs, rhs],
            output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
            ..GraphProto::default()
        };

        let lowered = lower_graph(&graph).expect("lower MatMul");
        let lift_input = LiftInput {
            program: &lowered.program,
            initializers: &lowered.initializers,
            graph_inputs: &lowered.graph_inputs,
            graph_outputs: &lowered.graph_outputs,
            graph_name: "matmul_lifted",
        };
        let lifted_bytes = lift_model(lift_input).expect("lift MatMul to onnx bytes");
        let reparsed_model = parse_complete(&lifted_bytes).expect("lifted bytes parse back to a ModelProto");
        let reparsed_graph = reparsed_model.graph.as_ref().expect("lifted graph present");

        let op_types: Vec<&str> = reparsed_graph.node.iter().map(|node| node.op_type).collect();
        assert_eq!(op_types, ["MatMul"], "lifted graph is a single named MatMul node, not Mul+ReduceSum");

        assert_graph_round_trips_through_lift(&graph, "matmul_lifted_roundtrip");
    }
}
