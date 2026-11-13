//! Split a program at a named cut edge — see `lib.rs`'s "Distribution
//! stance" for the design this implements: a partition is a projection of
//! `&[Op]`, a cut edge becomes a named [`Op::Input`] on the consuming side,
//! and the wire payload is `dtype + shape + bytes`, exactly what
//! [`cpu::Evaluated::get`](crate::cpu::Evaluated::get) already hands back.
//!
//! No new type: [`partition_at`] is a pure function over the same
//! `Vec<Op>` / [`NodeId`] vocabulary every other pass in this crate uses.
//! `symbols` is required (not just `program`) because a cut node's shape can
//! carry a symbolic extent (sequence length, batch size), and the wire
//! payload crossing the cut must be concrete — the same reason
//! [`shape::infer`] itself takes `symbols`.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::error::TensorError;
use crate::map::IndexMap;
use crate::op::{Extent, NodeId, Op, Reduce, ScalarOp};
use crate::shape;

/// Every `NodeId` an op at or after `boundary` reaches backwards across it,
/// in original numbering.
fn producer_side_refs(op: &Op, boundary: u32) -> Vec<NodeId> {
    match op {
        Op::Input { .. } | Op::Iota { .. } | Op::Constant { .. } => Vec::new(),
        Op::Elementwise { operands, .. } => {
            let mut references = Vec::with_capacity(operands.len() * 2);
            for (node, index_map) in operands {
                if node.0 <= boundary {
                    references.push(*node);
                }
                if let IndexMap::Computed { indices, .. } = index_map
                    && indices.0 <= boundary
                {
                    references.push(*indices);
                }
            }
            references
        }
        Op::Reduce(reduce) => {
            if reduce.operand.0 <= boundary {
                alloc::vec![reduce.operand]
            } else {
                Vec::new()
            }
        }
    }
}

/// Rewrite every `NodeId` an op names through `table`, leaving everything
/// else (dtype, body, maps, names) untouched. `table` is total over every
/// reference `op` can make by construction of [`partition_at`]'s caller
/// loop, so a miss means the caller built `table` wrong — reported as the
/// same "reads a node that is not there yet" error [`shape::infer`] would
/// raise for the same defect in a hand-written program.
pub(crate) fn remap(
    op: &Op,
    table: &BTreeMap<NodeId, NodeId>,
    self_id: NodeId,
) -> Result<Op, TensorError> {
    let translate = |node: NodeId| {
        table
            .get(&node)
            .copied()
            .ok_or(TensorError::NodeOutOfRange(self_id, node))
    };
    match op {
        Op::Input { dtype, shape, name } => Ok(Op::Input {
            dtype: *dtype,
            shape: shape.clone(),
            name: name.clone(),
        }),
        Op::Iota { dtype, extent } => Ok(Op::Iota {
            dtype: *dtype,
            extent: *extent,
        }),
        Op::Constant {
            dtype,
            shape,
            value,
        } => Ok(Op::Constant {
            dtype: *dtype,
            shape: shape.clone(),
            value: *value,
        }),
        Op::Elementwise {
            dtype,
            body,
            operands,
            name,
        } => {
            let mut translated = Vec::with_capacity(operands.len());
            for (node, index_map) in operands {
                let translated_map = match index_map {
                    IndexMap::Affine(pattern) => IndexMap::Affine(pattern.clone()),
                    IndexMap::Computed {
                        indices,
                        index_map,
                        base,
                        gathered_dim,
                    } => IndexMap::Computed {
                        indices: translate(*indices)?,
                        index_map: index_map.clone(),
                        base: base.clone(),
                        gathered_dim: *gathered_dim,
                    },
                };
                translated.push((translate(*node)?, translated_map));
            }
            Ok(Op::Elementwise {
                dtype: *dtype,
                body: *body,
                operands: translated,
                name: name.clone(),
            })
        }
        Op::Reduce(reduce) => {
            let remap_index_map = |index_map: &IndexMap| -> Result<IndexMap, TensorError> {
                match index_map {
                    IndexMap::Affine(pattern) => Ok(IndexMap::Affine(pattern.clone())),
                    IndexMap::Computed {
                        indices,
                        index_map,
                        base,
                        gathered_dim,
                    } => Ok(IndexMap::Computed {
                        indices: translate(*indices)?,
                        index_map: index_map.clone(),
                        base: base.clone(),
                        gathered_dim: *gathered_dim,
                    }),
                }
            };
            Ok(Op::Reduce(Reduce {
                dtype: reduce.dtype,
                body: reduce.body,
                init: reduce.init,
                operand: translate(reduce.operand)?,
                in_map: remap_index_map(&reduce.in_map)?,
                out_map: remap_index_map(&reduce.out_map)?,
                keep: reduce.keep,
                name: reduce.name.clone(),
            }))
        }
    }
}

// a type alias, not a new type: names the tuple `partition_at` already
// returns so clippy's `type_complexity` lint reads it once instead of
// flagging the signature; every element keeps its own doc below.
type Partitioned = (Vec<Op>, Vec<(NodeId, String)>, Vec<Op>);

// same reasoning as `Partitioned` above, one alias per distinct tuple shape
// the `partition_between*` family returns.
type SegmentAndCuts = (Vec<Op>, Vec<(NodeId, String)>);
type SegmentCutsEnd = (Vec<Op>, Vec<(NodeId, String)>, NodeId);
type SegmentCutsMapping = (Vec<Op>, Vec<(NodeId, String)>, BTreeMap<NodeId, NodeId>);

/// Split `program` into a producer sub-program (everything at or before
/// `boundary`, unchanged — positions are backward-only, so nothing later
/// can be referenced without crossing the cut) and a consumer sub-program
/// (everything after `boundary`, renumbered).
///
/// Every producer-side node a consumer op reaches across `boundary` becomes
/// a leading, named `Op::Input` in the returned consumer program — its
/// `Op::name()` if it has one, else a synthesized `"__cut_{id}"`, both
/// concrete-shaped via `symbols` (see this module's doc). The second
/// element of the return is exactly the `(NodeId, name)` pairs needed to
/// bind the two halves back together: evaluate the producer with these
/// `NodeId`s as `outputs`, pair each result with its `name`, and feed that
/// straight into [`cpu::evaluate_named`](crate::cpu::evaluate_named) for
/// the consumer — see `cpu.rs`'s round-trip test.
///
/// `boundary` must be a valid position in `program`; a `boundary` at the
/// program's last index yields an empty consumer.
pub fn partition_at(
    program: &[Op],
    symbols: &[u64],
    boundary: NodeId,
) -> Result<Partitioned, TensorError> {
    let boundary_index = boundary.0 as usize;
    if boundary_index >= program.len() {
        return Err(TensorError::UnknownOutput(boundary));
    }

    let producer = program[..=boundary_index].to_vec();
    let shapes = shape::infer(&producer, symbols)?;

    let mut crossing: BTreeMap<NodeId, ()> = BTreeMap::new();
    for op in &program[boundary_index + 1..] {
        for node in producer_side_refs(op, boundary.0) {
            crossing.insert(node, ());
        }
    }

    let mut table: BTreeMap<NodeId, NodeId> = BTreeMap::new();
    let mut cut_inputs = Vec::with_capacity(crossing.len());
    let mut consumer = Vec::with_capacity(crossing.len() + program.len() - boundary_index - 1);
    for (position, node) in crossing.keys().enumerate() {
        let extents: Vec<Extent> = shapes
            .of(*node)
            .iter()
            .map(|extent| Extent::Static(*extent as u32))
            .collect();
        let name = producer[node.0 as usize]
            .name()
            .map(String::from)
            .unwrap_or_else(|| format!("__cut_{}", node.0));
        let new_id = NodeId(position as u32);
        table.insert(*node, new_id);
        cut_inputs.push((*node, name.clone()));
        consumer.push(Op::Input {
            dtype: producer[node.0 as usize].dtype(),
            shape: extents,
            name: Some(name),
        });
    }

    for (offset, op) in program[boundary_index + 1..].iter().enumerate() {
        let self_id = NodeId((boundary_index + 1 + offset) as u32);
        let new_id = NodeId(consumer.len() as u32);
        table.insert(self_id, new_id);
        consumer.push(remap(op, &table, self_id)?);
    }

    Ok((producer, cut_inputs, consumer))
}

/// Extracts the contiguous operation range `(start_exclusive, end_inclusive]`
/// as an evaluable program. References to nodes at or before `start_exclusive`
/// become named inputs; nodes inside the range are renumbered densely. The
/// returned cut list uses original node ids, so a caller can carry values
/// between sequential layer segments without relying on positional ids.
pub fn partition_between(
    program: &[Op],
    symbols: &[u64],
    start_exclusive: Option<NodeId>,
    end_inclusive: NodeId,
) -> Result<SegmentAndCuts, TensorError> {
    let (segment, cuts, _) =
        partition_between_with_output(program, symbols, start_exclusive, end_inclusive)?;
    Ok((segment, cuts))
}

/// Like [`partition_between`], also returns the dense id corresponding to the
/// requested end node. Graph builders may store operations out of dependency
/// order, so the extracted body is stably topologically ordered before ids are
/// assigned.
pub fn partition_between_with_output(
    program: &[Op],
    symbols: &[u64],
    start_exclusive: Option<NodeId>,
    end_inclusive: NodeId,
) -> Result<SegmentCutsEnd, TensorError> {
    let (segment, cuts, mapping) =
        partition_between_with_mapping(program, symbols, start_exclusive, end_inclusive)?;
    let mapped_end = mapping
        .get(&end_inclusive)
        .copied()
        .ok_or(TensorError::UnknownOutput(end_inclusive))?;
    Ok((segment, cuts, mapped_end))
}

/// Like [`partition_between_with_output`], retaining the original-to-dense
/// node map so a caller can request cache and diagnostic roots produced inside
/// the same segment without relying on the source operation order.
pub fn partition_between_with_mapping(
    program: &[Op],
    symbols: &[u64],
    start_exclusive: Option<NodeId>,
    end_inclusive: NodeId,
) -> Result<SegmentCutsMapping, TensorError> {
    let end = end_inclusive.0 as usize;
    if end >= program.len() {
        return Err(TensorError::UnknownOutput(end_inclusive));
    }
    let start = start_exclusive.map_or(usize::MAX, |node| node.0 as usize);
    if start != usize::MAX && start >= end {
        return Err(TensorError::UnknownOutput(end_inclusive));
    }
    let first = if start == usize::MAX { 0 } else { start + 1 };
    let shapes = shape::infer(&program[..=end], symbols)?;
    let mut crossing = BTreeMap::new();
    for op in &program[first..=end] {
        let references = match op {
            Op::Input { .. } | Op::Iota { .. } | Op::Constant { .. } => Vec::new(),
            Op::Elementwise { operands, .. } => {
                let mut references = Vec::with_capacity(operands.len() * 2);
                for (node, index_map) in operands {
                    references.push(*node);
                    if let IndexMap::Computed { indices, .. } = index_map {
                        references.push(*indices);
                    }
                }
                references
            }
            Op::Reduce(reduce) => {
                let mut references = alloc::vec![reduce.operand];
                if let IndexMap::Computed { indices, .. } = &reduce.in_map {
                    references.push(*indices);
                }
                if let IndexMap::Computed { indices, .. } = &reduce.out_map {
                    references.push(*indices);
                }
                references
            }
        };
        for node in references {
            if (node.0 as usize) < first {
                crossing.insert(node, ());
            } else if (node.0 as usize) > end {
                return Err(TensorError::NodeOutOfRange(end_inclusive, node));
            }
        }
    }
    let mut table = BTreeMap::new();
    let mut segment = Vec::with_capacity(crossing.len() + end - first + 1);
    let mut cut_inputs = Vec::with_capacity(crossing.len());
    for (position, node) in crossing.keys().enumerate() {
        let extents = shapes
            .of(*node)
            .iter()
            .map(|extent| Extent::Static(*extent as u32))
            .collect();
        let name = program[node.0 as usize]
            .name()
            .map(String::from)
            .unwrap_or_else(|| format!("__cut_{}", node.0));
        let mapped = NodeId(position as u32);
        table.insert(*node, mapped);
        cut_inputs.push((*node, name.clone()));
        segment.push(Op::Input {
            dtype: program[node.0 as usize].dtype(),
            shape: extents,
            name: Some(name),
        });
    }
    let mut pending: Vec<NodeId> = (first..=end).map(|index| NodeId(index as u32)).collect();
    let mut ordered = Vec::with_capacity(pending.len());
    while !pending.is_empty() {
        let position = pending.iter().position(|candidate| {
            let references = match &program[candidate.0 as usize] {
                Op::Input { .. } | Op::Iota { .. } | Op::Constant { .. } => Vec::new(),
                Op::Elementwise { operands, .. } => {
                    let mut references = Vec::with_capacity(operands.len() * 2);
                    for (node, index_map) in operands {
                        references.push(*node);
                        if let IndexMap::Computed { indices, .. } = index_map {
                            references.push(*indices);
                        }
                    }
                    references
                }
                Op::Reduce(reduce) => {
                    let mut references = alloc::vec![reduce.operand];
                    if let IndexMap::Computed { indices, .. } = &reduce.in_map {
                        references.push(*indices);
                    }
                    if let IndexMap::Computed { indices, .. } = &reduce.out_map {
                        references.push(*indices);
                    }
                    references
                }
            };
            references
                .into_iter()
                .all(|reference| (reference.0 as usize) < first || !pending.contains(&reference))
        });
        let Some(position) = position else {
            return Err(TensorError::NodeOutOfRange(end_inclusive, pending[0]));
        };
        ordered.push(pending.remove(position));
    }
    for (offset, original) in ordered.iter().enumerate() {
        table.insert(*original, NodeId((cut_inputs.len() + offset) as u32));
    }
    for original in &ordered {
        segment.push(remap(&program[original.0 as usize], &table, *original)?);
    }
    Ok((segment, cut_inputs, table))
}

/// Like [`partition_between_with_mapping`], but retains only the nodes that
/// are ancestors of `end_inclusive`. This is the form for a fused boundary
/// whose interval contains unrelated future-layer inputs: interval membership
/// alone must not force those inputs into the executable program.
pub fn partition_between_with_mapping_ancestor_closed(
    program: &[Op],
    symbols: &[u64],
    start_exclusive: Option<NodeId>,
    end_inclusive: NodeId,
) -> Result<SegmentCutsMapping, TensorError> {
    let end = end_inclusive.0 as usize;
    if end >= program.len() {
        return Err(TensorError::UnknownOutput(end_inclusive));
    }
    let start = start_exclusive.map_or(usize::MAX, |node| node.0 as usize);
    if start != usize::MAX && start >= end {
        return Err(TensorError::UnknownOutput(end_inclusive));
    }
    let first = if start == usize::MAX { 0 } else { start + 1 };
    let shapes = shape::infer(&program[..=end], symbols)?;
    let mut reachable = BTreeSet::new();
    let mut visit = vec![end_inclusive];
    let mut crossing = BTreeMap::new();
    while let Some(node) = visit.pop() {
        if !reachable.insert(node) {
            continue;
        }
        let index = node.0 as usize;
        if index < first {
            crossing.insert(node, ());
            continue;
        }
        for reference in op_references(&program[index]) {
            if reference.0 as usize > end {
                return Err(TensorError::NodeOutOfRange(end_inclusive, reference));
            }
            visit.push(reference);
        }
    }
    let mut table = BTreeMap::new();
    let mut segment = Vec::with_capacity(reachable.len());
    let mut cut_inputs = Vec::with_capacity(crossing.len());
    for (position, node) in crossing.keys().enumerate() {
        let extents = shapes
            .of(*node)
            .iter()
            .map(|extent| Extent::Static(*extent as u32))
            .collect();
        let name = program[node.0 as usize]
            .name()
            .map(String::from)
            .unwrap_or_else(|| format!("__cut_{}", node.0));
        let mapped = NodeId(position as u32);
        table.insert(*node, mapped);
        cut_inputs.push((*node, name.clone()));
        segment.push(Op::Input {
            dtype: program[node.0 as usize].dtype(),
            shape: extents,
            name: Some(name),
        });
    }
    let mut pending: Vec<NodeId> = reachable
        .iter()
        .copied()
        .filter(|node| {
            let index = node.0 as usize;
            index >= first && index <= end
        })
        .collect();
    let mut ordered = Vec::with_capacity(pending.len());
    while !pending.is_empty() {
        let position = pending.iter().position(|candidate| {
            op_references(&program[candidate.0 as usize])
                .into_iter()
                .all(|reference| !pending.contains(&reference))
        });
        let Some(position) = position else {
            return Err(TensorError::NodeOutOfRange(end_inclusive, pending[0]));
        };
        ordered.push(pending.remove(position));
    }
    for (offset, original) in ordered.iter().enumerate() {
        table.insert(*original, NodeId((cut_inputs.len() + offset) as u32));
    }
    for original in &ordered {
        segment.push(remap(&program[original.0 as usize], &table, *original)?);
    }
    Ok((segment, cut_inputs, table))
}

fn op_references(op: &Op) -> Vec<NodeId> {
    match op {
        Op::Input { .. } | Op::Iota { .. } | Op::Constant { .. } => Vec::new(),
        Op::Elementwise { operands, .. } => operands
            .iter()
            .flat_map(|(node, index_map)| {
                let computed = match index_map {
                    IndexMap::Affine(_) => None,
                    IndexMap::Computed { indices, .. } => Some(*indices),
                };
                core::iter::once(*node).chain(computed)
            })
            .collect(),
        Op::Reduce(reduce) => {
            let mut references = vec![reduce.operand];
            if let IndexMap::Computed { indices, .. } = &reduce.in_map {
                references.push(*indices);
            }
            if let IndexMap::Computed { indices, .. } = &reduce.out_map {
                references.push(*indices);
            }
            references
        }
    }
}

/// Whether `node` appears, anywhere in `program`, ONLY as the SOLE operand of
/// an `Identity` [`Op::Elementwise`] addressed through an
/// [`IndexMap::Computed`] pattern -- the exact "embedding table" gather shape
/// [`embedding_lookup`](crate::spec::embedding_lookup) builds. The CPU's
/// quantized evaluator reads such a table one packed row at a time
/// (`run_embedding_gather_quantized`), so its f32-only validation exempts it;
/// [`row_gathers`] reads the same shape to find tables a separate-memory
/// device never needs whole. One definition for both.
pub(crate) fn is_gather_only_operand(program: &[Op], node: NodeId) -> bool {
    let mut used_as_gather_operand = false;
    for expr in program {
        let Op::Elementwise { body, operands, .. } = expr else {
            continue;
        };
        if !operands.iter().any(|(source, _)| *source == node) {
            continue;
        }
        let [(source, index_map)] = operands.as_slice() else {
            return false;
        };
        if *source != node || *body != ScalarOp::Identity {
            return false;
        }
        if !matches!(index_map, IndexMap::Computed { .. }) {
            return false;
        }
        used_as_gather_operand = true;
    }
    used_as_gather_operand
}

/// One table leaf read by exactly one row gather: the three nodes
/// [`hoist_row_gather`] rewrites. Plain data, not a pipe: it names nodes of a
/// program, it transforms nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RowGather {
    /// The `Op::Input` table leaf (`[vocab, row...]`).
    pub table: NodeId,
    /// The node supplying the row indices.
    pub indices: NodeId,
    /// The `Identity` elementwise that reads `table` through `indices`.
    pub gather: NodeId,
}

/// Every `Op::Input` leaf whose only reader in `program` is one `Identity`
/// row gather with an [`IndexMap::Computed`] operand map -- the shape
/// [`crate::spec::embedding_lookup_as`] builds. Such a table never has to be
/// resident on a [`crate::bind::DeviceMemory::Separate`] device: the rows a
/// step reads can be gathered on the host (see [`hoist_row_gather`]). A leaf
/// that any other op also reads (a tied output head's matmul) is not returned.
///
/// Composes [`is_gather_only_operand`], the predicate the CPU evaluator also
/// uses, plus a whole-program reference count so a `Reduce` reading the leaf
/// directly refuses it too.
#[must_use]
pub fn row_gathers(program: &[Op]) -> Vec<RowGather> {
    let mut reads = vec![0u32; program.len()];
    for op in program {
        for node in op_references(op) {
            if let Some(count) = reads.get_mut(node.0 as usize) {
                *count += 1;
            }
        }
    }
    program
        .iter()
        .enumerate()
        .filter_map(|(position, op)| {
            let Op::Elementwise {
                body: ScalarOp::Identity,
                operands,
                ..
            } = op
            else {
                return None;
            };
            let [(table, IndexMap::Computed { indices, .. })] = operands.as_slice() else {
                return None;
            };
            let is_leaf = matches!(program.get(table.0 as usize), Some(Op::Input { .. }));
            let sole_reader = reads.get(table.0 as usize) == Some(&1);
            (is_leaf && sole_reader && is_gather_only_operand(program, *table)).then_some(RowGather {
                table: *table,
                indices: *indices,
                gather: NodeId(position as u32),
            })
        })
        .collect()
}

/// What [`hoist_row_gather`] leaves behind for the host to run.
/// Plain data: `producer` is an ordinary program any evaluator runs.
#[derive(Debug, Clone)]
pub struct Hoisted {
    /// A self-contained program (`[ids input, table input, gather]`).
    pub producer: Vec<Op>,
    /// The node of `producer` that evaluates to the gathered rows.
    pub output: NodeId,
    /// The name of the new input leaf (`"{table}@rows"`) the device program
    /// now reads the rows from.
    pub leaf: String,
}

/// Rewrites `program` in place so `gather.table` is no longer a table but the
/// already-gathered rows: the leaf becomes
/// `Input { name: "{table}@rows", shape: indices_shape ++ table_shape[1..] }`
/// at the gather's dtype, and the gather reads it through the identity affine
/// map over its own rank. NodeIds do not move, so roots held by the caller
/// stay valid. The returned [`Hoisted::producer`] evaluates to exactly the
/// rows the unrewritten gather produced.
///
/// # Errors
/// [`TensorError::NotLowerable`] when `gather` does not name a named table
/// `Input`, a shaped indices `Input`, and a gather on axis 0 whose rank is
/// `indices` rank + `table` rank - 1.
pub fn hoist_row_gather(program: &mut [Op], gather: &RowGather) -> Result<Hoisted, TensorError> {
    let refuse = |reason: &'static str| TensorError::NotLowerable {
        node: gather.gather,
        reason,
    };
    let table_op = program.get(gather.table.0 as usize).cloned();
    let indices_op = program.get(gather.indices.0 as usize).cloned();
    let gather_op = program.get(gather.gather.0 as usize).cloned();
    let (
        Some(table_op @ Op::Input { .. }),
        Some(indices_op @ Op::Input { .. }),
        Some(mut gather_op @ Op::Elementwise { .. }),
    ) = (table_op, indices_op, gather_op)
    else {
        return Err(refuse("row gather nodes are not input, input, elementwise"));
    };
    let Op::Input {
        shape: table_shape,
        name: Some(table_name),
        ..
    } = &table_op
    else {
        return Err(refuse("hoisted table input has no name"));
    };
    let Op::Input {
        shape: indices_shape,
        ..
    } = &indices_op
    else {
        return Err(refuse("row gather indices are not an input"));
    };
    let Op::Elementwise {
        dtype: rows_dtype,
        operands,
        ..
    } = &gather_op
    else {
        return Err(refuse("row gather node is not elementwise"));
    };
    let [(_, IndexMap::Computed { base, gathered_dim: 0, .. })] = operands.as_slice() else {
        return Err(refuse("row gather is not a single axis-0 computed read"));
    };
    let rows_shape: Vec<Extent> = indices_shape
        .iter()
        .chain(table_shape.iter().skip(1))
        .cloned()
        .collect();
    if rows_shape.len() != usize::from(base.iter_rank) {
        return Err(refuse("row gather rank is not indices rank plus row rank"));
    }
    let leaf = format!("{table_name}@rows");
    let rows_dtype = *rows_dtype;
    let rank = base.iter_rank;
    if let Op::Elementwise { operands, .. } = &mut gather_op {
        for (node, index_map) in operands.iter_mut() {
            *node = NodeId(1);
            if let IndexMap::Computed { indices, .. } = index_map {
                *indices = NodeId(0);
            }
        }
    }
    let producer = vec![indices_op, table_op, gather_op];

    if let Some(Op::Input { dtype, shape, name }) = program.get_mut(gather.table.0 as usize) {
        *dtype = rows_dtype;
        *shape = rows_shape;
        *name = Some(leaf.clone());
    }
    if let Some(Op::Elementwise { operands, .. }) = program.get_mut(gather.gather.0 as usize)
        && let Some((_, index_map)) = operands.first_mut()
    {
        let axes: Vec<u16> = (0..rank).collect();
        *index_map = IndexMap::Affine(crate::map::projection(rank, &axes));
    }
    Ok(Hoisted {
        producer,
        output: NodeId(2),
        leaf,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::cpu;
    use proxima_numeric::DType;
    use crate::map::{self, IndexMap};
    use crate::op::{ScalarOp, append};

    fn linear_chain() -> Vec<Op> {
        let mut program = Vec::new();
        let left = append(
            &mut program,
            Op::Input {
                dtype: DType::FLOAT32,
                shape: vec![Extent::Static(4)],
                name: Some(String::from("left")),
            },
        );
        let right = append(
            &mut program,
            Op::Input {
                dtype: DType::FLOAT32,
                shape: vec![Extent::Static(4)],
                name: Some(String::from("right")),
            },
        );
        let identity_map = IndexMap::Affine(map::projection(1, &[0]));
        let sum = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
                body: ScalarOp::Add,
                operands: alloc::vec![(left, identity_map.clone()), (right, identity_map.clone())],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
                body: ScalarOp::Negate,
                operands: alloc::vec![(sum, identity_map)],
                name: None,
            },
        );
        program
    }

    #[test]
    fn partition_between_materializes_external_cut_inputs() {
        let program = linear_chain();
        let (segment, cuts) = partition_between(&program, &[], Some(NodeId(0)), NodeId(2))
            .expect("the segment has an explicit external cut");
        assert_eq!(cuts.len(), 1);
        let evaluated = cpu::evaluate_named(
            &segment,
            &[],
            &[
                (cuts[0].1.as_str(), &[2.0_f32; 4]),
                ("right", &[3.0_f32; 4]),
            ],
            &[NodeId(2)],
        )
        .expect("segment evaluates with cut bindings");
        assert_eq!(evaluated.root(), &[5.0_f32; 4]);
    }

    #[test]
    fn ancestor_closed_partition_excludes_dead_interval_nodes() {
        let identity_map = IndexMap::Affine(map::projection(1, &[0]));
        let mut program = Vec::new();
        let input = append(
            &mut program,
            Op::Input {
                dtype: DType::FLOAT32,
                shape: vec![Extent::Static(1)],
                name: Some(String::from("input")),
            },
        );
        let live = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
                body: ScalarOp::Negate,
                operands: vec![(input, identity_map.clone())],
                name: Some(String::from("live")),
            },
        );
        let _dead = append(
            &mut program,
            Op::Input {
                dtype: DType::FLOAT32,
                shape: vec![Extent::Static(1)],
                name: Some(String::from("dead_future_expert")),
            },
        );
        let output = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
                body: ScalarOp::Negate,
                operands: vec![(live, identity_map)],
                name: Some(String::from("output")),
            },
        );

        let (segment, cuts, mapping) =
            partition_between_with_mapping_ancestor_closed(&program, &[], None, output)
                .expect("ancestor-closed partition succeeds");
        assert_eq!(cuts.len(), 0);
        assert!(
            !segment
                .iter()
                .any(|operation| { operation.name() == Some("dead_future_expert") })
        );
        let mapped_output = mapping.get(&output).copied().expect("output is retained");
        let evaluated = cpu::evaluate(&segment, &[], &[&[2.0_f32]], &[mapped_output])
            .expect("ancestor-closed segment evaluates");
        assert_eq!(evaluated.root(), &[2.0_f32]);
    }

    #[test]
    fn round_trip_matches_whole_program_evaluation() {
        let program = linear_chain();
        let symbols: Vec<u64> = Vec::new();
        let left_data = [1.0_f32, 2.0, 3.0, 4.0];
        let right_data = [10.0_f32, 20.0, 30.0, 40.0];
        let blocks: [&[f32]; 2] = [&left_data, &right_data];

        let whole =
            cpu::evaluate(&program, &symbols, &blocks, &[]).expect("whole program evaluates");

        let boundary = NodeId(2);
        let (producer, cut_inputs, consumer) =
            partition_at(&program, &symbols, boundary).expect("partition succeeds");
        assert_eq!(cut_inputs.len(), 1, "one value (the sum) crosses this cut");

        let producer_outputs: Vec<NodeId> = cut_inputs.iter().map(|(node, _)| *node).collect();
        let producer_result = cpu::evaluate(&producer, &symbols, &blocks, &producer_outputs)
            .expect("producer evaluates");

        let named: Vec<(&str, &[f32])> = cut_inputs
            .iter()
            .map(|(node, name)| {
                let (data, _shape) = producer_result
                    .get(*node)
                    .expect("producer emitted the cut value");
                (name.as_str(), data)
            })
            .collect();

        let consumer_result =
            cpu::evaluate_named(&consumer, &symbols, &named, &[]).expect("consumer evaluates");

        assert_eq!(
            consumer_result.root(),
            whole.root(),
            "partitioned run must be bit-identical to the whole program"
        );
    }

    fn xorshift_floats(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
            })
            .collect()
    }

    const TABLE_NAME: &str = "per_layer_token_embd.weight";

    /// ids -> gather from a packed table -> `Negate`, the gemma4
    /// `per_layer_token_embd` shape at test size.
    fn gather_then_negate(vocab: u32, dim: u32, seq: u32) -> (Vec<Op>, NodeId) {
        use crate::spec::{embedding_lookup, input_leaf};

        let mut program = Vec::new();
        let ids = input_leaf(&mut program, DType::INT32, vec![Extent::Static(seq)], "ids");
        let table = input_leaf(
            &mut program,
            DType::UINT8,
            vec![Extent::Static(vocab), Extent::Static(dim)],
            TABLE_NAME,
        );
        let rows = embedding_lookup(&mut program, table, ids);
        let negated = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
                body: ScalarOp::Negate,
                operands: alloc::vec![(rows, IndexMap::Affine(map::projection(2, &[0, 1])))],
                name: None,
            },
        );
        (program, negated)
    }

    #[test]
    fn hoisted_rows_equal_device_gather_on_a_q6k_table() {
        use crate::cpu::{QuantizedBlock, evaluate_quantized_named};
        use proxima_gguf::quant::q6_k::{BLOCK_BYTES, QK_K, quantize};
        use proxima_numeric::codec::Codec;

        let (vocab, dim, seq) = (6u32, 256u32, 3u32);
        let (program, negated) = gather_then_negate(vocab, dim, seq);
        let table_f32 = xorshift_floats(0x2026_1113_0303_beef, (vocab * dim) as usize);
        let mut table_bytes = vec![0u8; table_f32.len() / QK_K * BLOCK_BYTES];
        quantize(&table_f32, &mut table_bytes).expect("six whole q6_k rows quantize");
        let ids_data = [4.0f32, 0.0, 5.0];
        let ids_block = QuantizedBlock::Float32(ids_data.as_slice());
        let table_block = QuantizedBlock::Packed {
            codec: Codec::Q6K,
            bytes: &table_bytes,
        };

        let unhoisted = evaluate_quantized_named(
            &program,
            &[],
            &[("ids", ids_block), (TABLE_NAME, table_block)],
            &[negated],
        )
        .expect("the packed gather evaluates on device-shaped input");
        let (expected, expected_shape) = unhoisted.get(negated).expect("negated output");

        let mut device_program = program.clone();
        let gathers = row_gathers(&device_program);
        assert_eq!(gathers.len(), 1);
        let hoisted = hoist_row_gather(&mut device_program, &gathers[0]).expect("table hoists");
        assert_eq!(hoisted.leaf, "per_layer_token_embd.weight@rows");
        let produced = evaluate_quantized_named(
            &hoisted.producer,
            &[],
            &[("ids", ids_block), (TABLE_NAME, table_block)],
            &[hoisted.output],
        )
        .expect("the producer evaluates on the host");
        let (rows, rows_shape) = produced.get(hoisted.output).expect("rows output");
        assert_eq!(rows_shape, [u64::from(seq), u64::from(dim)]);
        let rows = rows.to_vec();

        let hoisted_run = evaluate_quantized_named(
            &device_program,
            &[],
            &[("ids", ids_block), (hoisted.leaf.as_str(), QuantizedBlock::Float32(&rows))],
            &[negated],
        )
        .expect("the device program evaluates with rows fed in");
        let (actual, actual_shape) = hoisted_run.get(negated).expect("negated output");

        assert_eq!(actual_shape, expected_shape);
        let actual_bits: Vec<u32> = actual.iter().map(|value| value.to_bits()).collect();
        let expected_bits: Vec<u32> = expected.iter().map(|value| value.to_bits()).collect();
        assert_eq!(actual_bits, expected_bits);
    }

    #[test]
    fn a_leaf_also_read_by_a_matmul_is_not_a_row_gather() {
        use crate::op::{Keep, ReduceInit};
        use crate::spec::{embedding_lookup, input_leaf};

        let mut program = Vec::new();
        let ids = input_leaf(&mut program, DType::INT32, vec![Extent::Static(2)], "ids");
        let shape = vec![Extent::Static(4), Extent::Static(8)];
        let gather_only = input_leaf(&mut program, DType::FLOAT32, shape.clone(), "per_layer.weight");
        let tied = input_leaf(&mut program, DType::FLOAT32, shape.clone(), "token_embd.weight");
        let activation = input_leaf(&mut program, DType::FLOAT32, shape, "activation");
        let _per_layer_rows = embedding_lookup(&mut program, gather_only, ids);
        let _head_rows = embedding_lookup(&mut program, tied, ids);
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![
                    (tied, IndexMap::Affine(map::projection(2, &[0, 1]))),
                    (activation, IndexMap::Affine(map::projection(2, &[0, 1]))),
                ],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::FLOAT32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
                out_map: IndexMap::Affine(map::projection(2, &[0])),
                keep: Keep::Reduce,
                name: None,
            }),
        );

        let found = row_gathers(&program);

        assert_eq!(found.len(), 1);
        assert_eq!(found[0].table, gather_only);
        assert_eq!(found[0].indices, ids);
    }
}
