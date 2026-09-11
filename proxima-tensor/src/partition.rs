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

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::error::TensorError;
use crate::map::IndexMap;
use crate::op::{Extent, NodeId, Op, Reduce};
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
fn remap(op: &Op, table: &BTreeMap<NodeId, NodeId>, self_id: NodeId) -> Result<Op, TensorError> {
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
) -> Result<(Vec<Op>, Vec<(NodeId, String)>), TensorError> {
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
) -> Result<(Vec<Op>, Vec<(NodeId, String)>, NodeId), TensorError> {
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
) -> Result<(Vec<Op>, Vec<(NodeId, String)>, BTreeMap<NodeId, NodeId>), TensorError> {
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::cpu;
    use crate::dtype::DType;
    use crate::map::{self, IndexMap};
    use crate::op::{ScalarOp, append};

    fn linear_chain() -> Vec<Op> {
        let mut program = Vec::new();
        let left = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(4)],
                name: Some(String::from("left")),
            },
        );
        let right = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(4)],
                name: Some(String::from("right")),
            },
        );
        let identity_map = IndexMap::Affine(map::projection(1, &[0]));
        let sum = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![(left, identity_map.clone()), (right, identity_map.clone())],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
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
}
