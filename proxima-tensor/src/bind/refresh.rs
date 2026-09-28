//! Refreshing a retained [`BoundOp`] set at a new set of symbols by
//! recomposing each fused window LOCALLY, at the fixed window the original
//! bind already decided — never by re-running the whole-program fusion
//! matchers ([`super::bind_cached_attention_fusion`], reduce-epilogue-fusion,
//! round-batching).
//!
//! The owner's classification (slice 2b brief): a masked position's fusion
//! DECISION does not change across a symbol crossing, only its extents,
//! layouts and kernel scalars do. This module keeps that decision fixed and
//! re-derives the numbers:
//!
//! 1. [`recover_window`] rebuilds the same `held: BTreeMap<NodeId,
//!    HeldElementwise>` [`super::builder_compose_window::BoundOpBuilder`]
//!    itself would have had at this node, purely from `program`'s own
//!    structure (an operand fuses into a window iff the edge map is an
//!    identity projection AND this consumer is that operand's LAST use,
//!    both symbol-independent — [`live::annotate`] is the SAME liveness
//!    pass the original bind used, called once here, not reimplemented).
//! 2. [`super::builder_compose_window::compose_fused_operands`] — the
//!    SAME function `BoundOpBuilder::push` calls for a fusing reduce — is
//!    then run over that fixed window and `new_shapes`, producing a fresh
//!    `(ComposedBody, BoundOperands)`.
//! 3. The recomposed [`ComposedBody`]'s own step-op SEQUENCE (structural,
//!    symbol-independent) must exactly match the retained op's — this is
//!    the validity check task step 3 asks for: if the window this module
//!    recovered was wrong (the real bind quarantined a broadcast operand at
//!    a shape this recovery does not model), the two step sequences
//!    disagree and refresh refuses by name rather than silently binding a
//!    different composition than the one that was actually retained.
//!
//! Reduce-epilogue fusion (a fold's `node` REPLACED by its consumer's own
//! identity) and every matcher-recognized kind
//! (`CachedAttention`/`CachedSoftmaxWeights`/`GatedDeltaNet`/`MoeTopK`/
//! `RoundBatchedReduce`) are NAMED FALLBACKS this slice does not recompose
//! locally — see [`RefreshRefusal`]'s own doc for why each one needs more
//! than this module's window-recovery to reconstruct safely.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::cell::RefCell;

use crate::live;
use crate::map::IndexMap;
use crate::numeric::NumericPolicy;
use crate::op::{NodeId, Op, ScalarOp};
use crate::shape::Shapes;

use super::builder_compose_window::{
    Constants, build_reduce_op, compose, compose_fused_operands, is_identity_projection,
};
use super::types_layout_boundop::{BoundOp, BoundOpKind, ComposedBody, HeldElementwise};

/// Why one retained [`BoundOp`] could not be refreshed in place.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RefreshRefusal {
    /// `position` is out of range of `resolved`/`masks`/`program`.
    #[error("refresh position {position} is out of range ({len} bound ops)")]
    PositionOutOfRange { position: usize, len: usize },

    /// The bound op at `position` is a matcher-recognized kind
    /// (`CachedAttention`/`CachedSoftmaxWeights`/`GatedDeltaNet`/`MoeTopK`/
    /// `RoundBatchedReduce`). Each absorbs a multi-node graph pattern (not a
    /// single fixed window) via a dedicated recognizer
    /// (`dead_code_cached_attention.rs`), and `CachedSoftmaxWeights`/
    /// `CachedAttention`'s own `cached_key_rows`/`new_key_rows` are sourced
    /// from a bind-internal recognizer node (`cached_key_even`) that is
    /// never itself a member of `BoundOp::operands()` — not reachable from a
    /// retained `BoundOp` plus `program` alone (SLICE1B.md's own residual).
    /// A NAMED FALLBACK, not attempted this slice.
    #[error("position {position}, node {node}: {kind} is matcher-recognized, refresh refuses rather than approximate its kernel scalars/operand layouts")]
    MatcherFusedKind {
        position: usize,
        node: NodeId,
        kind: &'static str,
    },

    /// This `Reduce`'s own `node` no longer names an `Op::Reduce` in
    /// `program` — `reduce-epilogue-fusion` already replaced it with a
    /// consumer's identity. Recomposing that shape needs the ORIGINAL
    /// reduce's own node recovered by searching `program` for a single-use
    /// producer whose iteration space this consumer reads identically
    /// (`find_epilogue_source`) plus re-running the epilogue's own body
    /// composition (a second, separate `compose` call this slice's window
    /// recovery does not extend to) — a NAMED FALLBACK, not attempted this
    /// slice.
    #[error("position {position}, node {node}: reduce-epilogue-fusion replaced this node's identity; epilogue recomposition is not attempted this slice")]
    EpilogueFusionNotLocal { position: usize, node: NodeId },

    /// `program[node]` is not the [`Op`] variant this position's
    /// [`BoundOpKind`] expects — the retained position and the immutable
    /// program have drifted out of correspondence.
    #[error("position {position}, node {node}: program op does not match this position's BoundOpKind")]
    OpKindMismatch { position: usize, node: NodeId },

    /// The window [`recover_window`] rebuilt from `program` recomposed to a
    /// DIFFERENT step-op sequence than the retained op's own `element_body`
    /// — the validity check task step 3 requires. Means either this
    /// recovery over- or under-absorbed relative to what the original bind
    /// actually did (most likely `quarantine_broadcast_operands`'s
    /// shape-dependent exception, which this recovery does not model), or a
    /// genuinely different fusion decision was made at the new shapes. In
    /// either case the ONLY safe answer is to refuse rather than bind a
    /// silently different composition.
    #[error("position {position}, node {node}: recomposed element_body step sequence {recomposed:?} does not match retained {retained:?}")]
    RecomposedBodyMismatch {
        position: usize,
        node: NodeId,
        retained: Vec<ScalarOp>,
        recomposed: Vec<ScalarOp>,
    },

    /// The recomposed operand list names a DIFFERENT set of leaf `NodeId`s
    /// than the retained op's own `operands()` — node identity is
    /// symbol-independent, so any difference here means [`recover_window`]
    /// absorbed a node the original bind actually kept materialized (most
    /// likely `quarantine_broadcast_operands`'s shape-dependent exception,
    /// which this recovery does not model — a broadcast operand whose
    /// materialized extent was smaller than the reduce's iteration extent at
    /// bind time). Refuses rather than silently bind a different leaf than
    /// the one the real program actually reads.
    #[error("position {position}, node {node}: recomposed operand set {recomposed:?} does not match retained {retained:?}")]
    RecomposedOperandsDiffer {
        position: usize,
        node: NodeId,
        retained: Vec<NodeId>,
        recomposed: Vec<NodeId>,
    },

    /// Re-deriving this position's extents/layout failed against the new
    /// shapes (a shape-inference fault, not a fusion-eligibility one).
    #[error("position {position}, node {node}: {source}")]
    ShapeError {
        position: usize,
        node: NodeId,
        #[source]
        source: crate::error::TensorError,
    },
}

fn is_masked(mask: u64, changed_symbols: u64) -> bool {
    mask & changed_symbols != 0
}

fn operand_ids(operands: &[(NodeId, super::types_layout_boundop::Layout, Option<super::types_layout_boundop::Lookup>)]) -> Vec<NodeId> {
    operands.iter().map(|(node, _, _)| *node).collect()
}

fn step_ops(body: &ComposedBody) -> Vec<ScalarOp> {
    body.steps.iter().map(|step| step.op).collect()
}

/// Rebuilds the same `held` window [`super::builder_compose_window::BoundOpBuilder`]
/// would have had when it originally composed `node` through `map` into
/// `consumer` — recursing only through edges that are (a) an identity
/// projection and (b) `node`'s own LAST use at `consumer` (`retires`, from
/// [`live::annotate`], the SAME liveness pass the original bind ran).
/// Symbol-independent by construction: neither condition reads a resolved
/// extent, only `program`'s own structure.
fn recover_window(
    program: &[Op],
    retires: &[Vec<NodeId>],
    consumer: NodeId,
    node: NodeId,
    map: &IndexMap,
    window: &mut BTreeMap<NodeId, HeldElementwise>,
) {
    if !is_identity_projection(map) {
        return;
    }
    if !retires
        .get(consumer.0 as usize)
        .is_some_and(|dying| dying.contains(&node))
    {
        return;
    }
    let Some(Op::Elementwise {
        dtype,
        body,
        operands,
        ..
    }) = program.get(node.0 as usize)
    else {
        return;
    };
    window.insert(
        node,
        HeldElementwise {
            dtype: *dtype,
            body: *body,
            operands: operands.clone(),
        },
    );
    for (sub_node, sub_map) in operands {
        recover_window(program, retires, node, *sub_node, sub_map, window);
    }
}

/// `ones`/`values` — the SAME per-position bookkeeping
/// [`super::builder_compose_window::BoundOpBuilder::push`] builds
/// incrementally, rebuilt here in one pass over the whole (symbol-independent)
/// `program` since a refresh call has no running builder to read them from.
fn constant_tables(program: &[Op]) -> (Vec<bool>, Vec<Option<f32>>) {
    let mut ones = Vec::with_capacity(program.len());
    let mut values = Vec::with_capacity(program.len());
    for op in program {
        let value = if let Op::Constant { value, .. } = op {
            Some(*value)
        } else {
            None
        };
        ones.push(value == Some(1.0));
        values.push(value);
    }
    (ones, values)
}

/// Refreshes one retained [`BoundOp`] against `new_shapes`, recomposing a
/// fused window locally rather than re-running the whole-program fusion
/// matchers, or names the exact condition and position it refuses at.
///
/// `masks[position]` is the caller's own per-`BoundOp` dependency mask (see
/// `kv_masked_positions`'s own `refresh_mask`); an unmasked position is
/// returned unchanged. `outputs` must be the SAME requested-output set the
/// original `bind_with_fusion` call used — [`live::annotate`]'s own
/// retirement table depends on it.
#[must_use = "a refusal names a real gap; dropping it silently accepts an unrefreshed op"]
pub fn refresh_bound_ops(
    resolved: &[BoundOp],
    program: &[Op],
    outputs: &[NodeId],
    new_shapes: &Shapes,
    masks: &[u64],
    changed_symbols: u64,
    numeric_policy: NumericPolicy,
) -> Result<Vec<BoundOp>, RefreshRefusal> {
    let retires = live::annotate(program, outputs);
    let (ones, values) = constant_tables(program);
    let mut refreshed = Vec::with_capacity(resolved.len());
    for (position, bound_op) in resolved.iter().enumerate() {
        let mask = masks
            .get(position)
            .copied()
            .ok_or(RefreshRefusal::PositionOutOfRange {
                position,
                len: resolved.len(),
            })?;
        if !is_masked(mask, changed_symbols) {
            refreshed.push(bound_op.clone());
            continue;
        }
        refreshed.push(refresh_one(
            position,
            bound_op,
            program,
            &retires,
            &ones,
            &values,
            numeric_policy,
            new_shapes,
        )?);
    }
    Ok(refreshed)
}

#[allow(clippy::too_many_arguments)]
fn refresh_one(
    position: usize,
    bound_op: &BoundOp,
    program: &[Op],
    retires: &[Vec<NodeId>],
    ones: &[bool],
    values: &[Option<f32>],
    numeric_policy: NumericPolicy,
    new_shapes: &Shapes,
) -> Result<BoundOp, RefreshRefusal> {
    match &bound_op.kind {
        BoundOpKind::Iota | BoundOpKind::Constant { .. } => Ok(BoundOp {
            extents: new_shapes.of(bound_op.node).to_vec(),
            ..bound_op.clone()
        }),
        BoundOpKind::Elementwise { body, .. } => refresh_elementwise(
            position, bound_op, body, program, retires, ones, values, numeric_policy, new_shapes,
        ),
        BoundOpKind::Reduce {
            element_body,
            epilogue_body,
            epilogue_operands,
            epilogue_broadcast_axes,
            out_scatter,
            ..
        } => {
            let has_epilogue =
                step_ops(epilogue_body) != [ScalarOp::Identity] || !epilogue_operands.is_empty();
            if has_epilogue || !epilogue_broadcast_axes.is_empty() || out_scatter.is_some() {
                return Err(RefreshRefusal::EpilogueFusionNotLocal {
                    position,
                    node: bound_op.node,
                });
            }
            refresh_reduce(
                position,
                bound_op,
                element_body,
                program,
                retires,
                ones,
                values,
                numeric_policy,
                new_shapes,
            )
        }
        BoundOpKind::CachedAttention { .. }
        | BoundOpKind::CachedSoftmaxWeights { .. }
        | BoundOpKind::GatedDeltaNet { .. }
        | BoundOpKind::MoeTopK { .. }
        | BoundOpKind::RoundBatchedReduce { .. } => Err(RefreshRefusal::MatcherFusedKind {
            position,
            node: bound_op.node,
            kind: bound_op.kind.name(),
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn refresh_elementwise(
    position: usize,
    bound_op: &BoundOp,
    retained_body: &ComposedBody,
    program: &[Op],
    retires: &[Vec<NodeId>],
    ones: &[bool],
    values: &[Option<f32>],
    numeric_policy: NumericPolicy,
    new_shapes: &Shapes,
) -> Result<BoundOp, RefreshRefusal> {
    let Some(Op::Elementwise {
        body: top_body,
        operands: top_operands,
        ..
    }) = program.get(bound_op.node.0 as usize)
    else {
        return Err(RefreshRefusal::OpKindMismatch {
            position,
            node: bound_op.node,
        });
    };

    let mut window = BTreeMap::new();
    for (operand_node, operand_map) in top_operands {
        recover_window(
            program,
            retires,
            bound_op.node,
            *operand_node,
            operand_map,
            &mut window,
        );
    }
    let constants = Constants {
        ones,
        values,
        numeric_policy,
    };
    let held = RefCell::new(window);
    let (recomposed_body, recomposed_operands) =
        compose(new_shapes, &held, *top_body, top_operands, constants);

    let retained_ops = step_ops(retained_body);
    let recomposed_ops = step_ops(&recomposed_body);
    if retained_ops != recomposed_ops {
        return Err(RefreshRefusal::RecomposedBodyMismatch {
            position,
            node: bound_op.node,
            retained: retained_ops,
            recomposed: recomposed_ops,
        });
    }
    let retained_operand_ids = operand_ids(bound_op.operands());
    let recomposed_operand_ids = operand_ids(&recomposed_operands);
    if retained_operand_ids != recomposed_operand_ids {
        return Err(RefreshRefusal::RecomposedOperandsDiffer {
            position,
            node: bound_op.node,
            retained: retained_operand_ids,
            recomposed: recomposed_operand_ids,
        });
    }

    Ok(BoundOp {
        extents: new_shapes.of(bound_op.node).to_vec(),
        kind: BoundOpKind::Elementwise {
            body: recomposed_body,
            operands: recomposed_operands,
        },
        ..bound_op.clone()
    })
}

#[allow(clippy::too_many_arguments)]
fn refresh_reduce(
    position: usize,
    bound_op: &BoundOp,
    retained_body: &ComposedBody,
    program: &[Op],
    retires: &[Vec<NodeId>],
    ones: &[bool],
    values: &[Option<f32>],
    numeric_policy: NumericPolicy,
    new_shapes: &Shapes,
) -> Result<BoundOp, RefreshRefusal> {
    let Some(Op::Reduce(reduce)) = program.get(bound_op.node.0 as usize) else {
        return Err(RefreshRefusal::OpKindMismatch {
            position,
            node: bound_op.node,
        });
    };

    let mut window = BTreeMap::new();
    recover_window(
        program,
        retires,
        bound_op.node,
        reduce.operand,
        &reduce.in_map,
        &mut window,
    );
    let held = RefCell::new(window);
    let constants = Constants {
        ones,
        values,
        numeric_policy,
    };
    let (recomposed_body, recomposed_operands) =
        compose_fused_operands(new_shapes, &held, reduce.operand, &reduce.in_map, constants);

    let retained_ops = step_ops(retained_body);
    let recomposed_ops = step_ops(&recomposed_body);
    if retained_ops != recomposed_ops {
        return Err(RefreshRefusal::RecomposedBodyMismatch {
            position,
            node: bound_op.node,
            retained: retained_ops,
            recomposed: recomposed_ops,
        });
    }
    let retained_operand_ids = operand_ids(bound_op.operands());
    let recomposed_operand_ids = operand_ids(&recomposed_operands);
    if retained_operand_ids != recomposed_operand_ids {
        return Err(RefreshRefusal::RecomposedOperandsDiffer {
            position,
            node: bound_op.node,
            retained: retained_operand_ids,
            recomposed: recomposed_operand_ids,
        });
    }

    build_reduce_op(bound_op.node, reduce, new_shapes, recomposed_body, recomposed_operands).map_err(
        |source| RefreshRefusal::ShapeError {
            position,
            node: bound_op.node,
            source,
        },
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::map::{self, IndexMap};
    use crate::op::{Extent, Keep, ReduceInit, append};
    use crate::shape;
    use crate::{NumericPolicy, bind_with_fusion};

    fn per_position_masks(program: &[Op], resolved: &[BoundOp]) -> Vec<u64> {
        let node_masks = shape::symbol_dependency_output_masks(program);
        let aggregate = |node: NodeId| node_masks[node.0 as usize].iter().fold(0, |mask, axis| mask | axis);
        resolved
            .iter()
            .map(|op| {
                let mut mask = aggregate(op.node);
                for (node, _, _) in op.all_read_sources() {
                    mask |= aggregate(*node);
                }
                if let BoundOpKind::Reduce { keep: Keep::Reduce, .. } = &op.kind
                    && let Some(Op::Reduce(reduce)) = program.get(op.node.0 as usize)
                {
                    mask |= shape::symbol_dependency_iteration_mask(reduce, &node_masks);
                }
                mask
            })
            .collect()
    }

    /// The matmul fixture `bind/tests.rs`'s own `matmul_program` uses: a
    /// symbolic-batch `lhs`, a static `rhs`, an elementwise `product` whose
    /// ONLY consumer is the reduce that sums it -- exactly the held-prologue
    /// shape this module's window recovery targets.
    fn matmul_program() -> (Vec<Op>, NodeId, NodeId) {
        let mut program = Vec::new();
        let lhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Symbolic(0), Extent::Static(8)],
                name: None,
            },
        );
        let rhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(8), Extent::Static(4)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![
                    (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                    (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
                ],
                name: None,
            },
        );
        let sum = append(
            &mut program,
            Op::Reduce(crate::op::Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        (program, sum, lhs)
    }

    fn bind_at(program: &[Op], outputs: &[NodeId], symbol: u64) -> Vec<BoundOp> {
        let shapes = shape::infer(program, &[symbol]).expect("matmul program infers");
        bind_with_fusion(program, &shapes, outputs, true, NumericPolicy::bit_exact())
            .expect("matmul program binds")
    }

    #[test]
    fn refresh_recomposes_a_reduce_that_absorbed_a_fused_prologue_locally() {
        let (program, sum, _lhs) = matmul_program();
        let outputs = [sum];
        let resolved_low = bind_at(&program, &outputs, 4);
        let resolved_high = bind_at(&program, &outputs, 9);

        let masks = per_position_masks(&program, &resolved_low);
        let new_shapes = shape::infer(&program, &[9]).expect("refresh target infers");

        let refreshed = refresh_bound_ops(
            &resolved_low,
            &program,
            &outputs,
            &new_shapes,
            &masks,
            1,
            NumericPolicy::bit_exact(),
        )
        .expect("a held-prologue reduce recomposes locally over its fixed window");

        assert_eq!(refreshed.len(), resolved_high.len(), "positions compared: {}", refreshed.len());
        assert_eq!(
            refreshed, resolved_high,
            "locally recomposed bound ops must equal a fresh bind_with_fusion"
        );
    }

    #[test]
    fn refresh_carries_forward_a_position_whose_mask_does_not_name_the_changed_symbol() {
        let (program, sum, _lhs) = matmul_program();
        let outputs = [sum];
        let resolved_low = bind_at(&program, &outputs, 4);
        let new_shapes = shape::infer(&program, &[4]).expect("same-symbol infer");
        let masks = alloc::vec![0u64; resolved_low.len()];

        let refreshed = refresh_bound_ops(
            &resolved_low,
            &program,
            &outputs,
            &new_shapes,
            &masks,
            1,
            NumericPolicy::bit_exact(),
        )
        .expect("no position is masked");

        assert_eq!(refreshed, resolved_low, "unmasked positions are carried forward unchanged");
    }
}
