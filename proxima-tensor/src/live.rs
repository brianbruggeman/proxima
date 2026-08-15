//! Exact liveness for a tensor program: which nodes retire at which position.
//!
//! [`annotate`] runs one backward scan over the whole program and returns,
//! per expression, the set of node ids whose *last* use is that expression —
//! after that position nothing in the program reads them again. A node
//! nothing ever references retires at its own position (dead code). A
//! requested output never retires.
//!
//! This is computed where the whole program is in hand — the batch driver,
//! locally, in this crate; a remote producer, before the wire, later, for a
//! partitioned program. [`nest::Lower`](crate::nest::Lower) and
//! [`cpu::evaluate`](crate::cpu::evaluate) are consumers of the result: they
//! obey the kill flag a retire set carries and never guess whether a value is
//! still needed. A different retention policy is a different upstream
//! annotator producing a different `Vec<Vec<NodeId>>` — this crate ships
//! exactly one, the exact one.

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use crate::expr::{Expr, NodeId};
use crate::map::IndexMap;

/// One retire set per program position: `result[p]` is every node whose last
/// use is `program[p]`.
#[must_use]
pub fn annotate(program: &[Expr], outputs: &[NodeId]) -> Vec<Vec<NodeId>> {
    let outputs: BTreeSet<NodeId> = outputs.iter().copied().collect();
    let mut last_use: Vec<Option<usize>> = vec![None; program.len()];

    for (position, expr) in program.iter().enumerate() {
        for used in uses(expr) {
            last_use[used.0 as usize] = Some(position);
        }
    }

    let mut retires: Vec<Vec<NodeId>> = vec![Vec::new(); program.len()];
    for (index, use_position) in last_use.into_iter().enumerate() {
        let node = NodeId(index as u32);
        if outputs.contains(&node) {
            continue;
        }
        let retire_at = use_position.unwrap_or(index);
        retires[retire_at].push(node);
    }
    retires
}

fn uses(expr: &Expr) -> Vec<NodeId> {
    match expr {
        Expr::Block { .. } => Vec::new(),
        Expr::Zip { operands, .. } => operands
            .iter()
            .flat_map(|(node, map)| map_uses(*node, map))
            .collect(),
        Expr::Fold(fold) => {
            let mut used = map_uses(fold.operand, &fold.in_map);
            if let IndexMap::Computed { indices, .. } = &fold.out_map {
                used.push(*indices);
            }
            used
        }
    }
}

fn map_uses(operand: NodeId, map: &IndexMap) -> Vec<NodeId> {
    let mut used = vec![operand];
    if let IndexMap::Computed { indices, .. } = map {
        used.push(*indices);
    }
    used
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::expr::{Extent, Fold, FoldInit, Keep, ScalarOp, append};
    use crate::map;

    fn block(program: &mut Vec<Expr>) -> NodeId {
        append(
            program,
            Expr::Block {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        )
    }

    #[test]
    fn a_single_use_operand_retires_at_its_consumer() {
        let mut program = Vec::new();
        let source = block(&mut program);
        let map = IndexMap::Affine(map::projection(1, &[0]));
        let consumer = append(
            &mut program,
            Expr::Zip {
                dtype: DType::Float32,
                body: ScalarOp::Negate,
                operands: alloc::vec![(source, map)],
                name: None,
            },
        );

        let retires = annotate(&program, &[]);
        assert!(retires[consumer.0 as usize].contains(&source));
    }

    #[test]
    fn a_requested_output_never_retires() {
        let mut program = Vec::new();
        let source = block(&mut program);
        let map = IndexMap::Affine(map::projection(1, &[0]));
        let consumer = append(
            &mut program,
            Expr::Zip {
                dtype: DType::Float32,
                body: ScalarOp::Negate,
                operands: alloc::vec![(source, map)],
                name: None,
            },
        );

        let retires = annotate(&program, &[source]);
        assert!(retires.into_iter().flatten().all(|node| node != source));
        let _ = consumer;
    }

    #[test]
    fn dead_code_retires_at_its_own_position() {
        let mut program = Vec::new();
        let never_used = block(&mut program);

        let retires = annotate(&program, &[]);
        assert_eq!(retires[never_used.0 as usize], alloc::vec![never_used]);
    }

    #[test]
    fn a_fold_operand_retires_at_the_fold() {
        let mut program = Vec::new();
        let source = block(&mut program);
        let in_map = IndexMap::Affine(map::projection(1, &[0]));
        let out_map = IndexMap::Affine(map::projection(1, &[]));
        let folded = append(
            &mut program,
            Expr::Fold(Fold {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: FoldInit::Zero,
                operand: source,
                in_map,
                out_map,
                keep: Keep::Last,
                name: None,
            }),
        );

        let retires = annotate(&program, &[]);
        assert!(retires[folded.0 as usize].contains(&source));
    }

    #[test]
    fn a_multiply_used_operand_retires_at_its_last_use_not_its_first() {
        let mut program = Vec::new();
        let shared = block(&mut program);
        let map = IndexMap::Affine(map::projection(1, &[0]));
        let first = append(
            &mut program,
            Expr::Zip {
                dtype: DType::Float32,
                body: ScalarOp::Negate,
                operands: alloc::vec![(shared, map.clone())],
                name: None,
            },
        );
        let second = append(
            &mut program,
            Expr::Zip {
                dtype: DType::Float32,
                body: ScalarOp::Negate,
                operands: alloc::vec![(shared, map)],
                name: None,
            },
        );

        let retires = annotate(&program, &[]);
        assert!(!retires[first.0 as usize].contains(&shared));
        assert!(retires[second.0 as usize].contains(&shared));
    }
}
