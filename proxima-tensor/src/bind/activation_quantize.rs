//! The backend-neutral structure of a derived activation quantize: the `BlockQuantize` node a plan
//! adds beside the program's own ops, the readers it repoints, and the undo. Which readers a
//! backend can serve is that backend's decision and stays with it.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use proxima_numeric::DType;
use proxima_numeric::block_layout::BlockLayout;

use super::{BoundOp, BoundOpKind, Layout};
use crate::op::NodeId;

/// A quantize node the plan added: its id is at or past the program's length.
#[must_use]
pub fn is_derived(bound: &BoundOp, program_len: usize) -> bool {
    bound.node.0 as usize >= program_len && matches!(bound.kind, BoundOpKind::BlockQuantize { .. })
}

/// Whether the reader's activation operand is one dense `[tokens, depth]` run: token axis kept in
/// the output, every other live axis a contiguous reduced run of `depth` elements.
#[must_use]
pub fn activation_is_flat(bound: &BoundOp, layout: &Layout, tokens: u64, depth: u64) -> bool {
    let BoundOpKind::Reduce { output_axes, .. } = &bound.kind else {
        return false;
    };
    let mut live: Vec<(i64, u16)> = (0..bound.extents.len() as u16)
        .filter(|axis| layout.stride(*axis) != 0)
        .map(|axis| (layout.stride(axis), axis))
        .collect();
    live.sort_by_key(|(stride, _)| Reverse(*stride));
    let Some((&(token_stride, token_axis), depth_axes)) = live.split_first() else {
        return false;
    };
    let dense_run = depth_axes.iter().rev().try_fold(1i64, |expected, (stride, axis)| {
        (*stride == expected).then(|| expected * bound.extents[usize::from(*axis)] as i64)
    });
    layout.base == 0
        && dense_run == Some(depth as i64)
        && token_stride == depth as i64
        && bound.extents[usize::from(token_axis)] == tokens
        && output_axes.contains(&token_axis)
        && depth_axes.iter().all(|(_, axis)| !output_axes.contains(axis))
}

/// `[tokens, depth]` of an activation shape: the leading axis and the product of the rest.
#[must_use]
pub fn collapsed_extent(shape: &[u64]) -> Option<[u64; 2]> {
    let (tokens, trailing) = shape.split_first()?;
    (!trailing.is_empty()).then(|| [*tokens, trailing.iter().product()])
}

/// The `BlockQuantize` node `derived` that holds `activation` in the block layout `held`, over the
/// flat `[tokens, depth]` extent, addressed like `template` with a dense row-major source.
#[must_use]
pub fn quantize_op(derived: NodeId, activation: NodeId, extent: [u64; 2], template: &Layout, held: &BlockLayout) -> BoundOp {
    let [tokens, depth] = extent;
    let mut layout = template.clone();
    layout.base = 0;
    layout.strides.clear();
    layout.strides.extend([depth as i64, 1]);
    let bytes = depth / u64::from(held.elements) * u64::from(held.block_bytes);
    BoundOp {
        node: derived,
        dtype: DType::UINT8,
        extents: vec![tokens, bytes],
        kind: BoundOpKind::BlockQuantize {
            operands: vec![(activation, layout, None)],
            layout: Box::new(*held),
        },
    }
}

/// Points operand `slot` of a reduce at `derived`.
pub fn repoint(bound: &mut BoundOp, slot: usize, derived: NodeId) {
    if let BoundOpKind::Reduce { operands, .. } = &mut bound.kind
        && let Some(operand) = operands.get_mut(slot)
    {
        operand.0 = derived;
    }
}

/// The bound ops without the plan-local quantize nodes, each reader reading its original f32
/// activation again: the form every program-keyed refresh understands.
#[must_use]
pub fn underive_block_quantize(resolved: &[BoundOp], program_len: usize) -> Vec<BoundOp> {
    let sources: BTreeMap<NodeId, NodeId> = resolved
        .iter()
        .filter(|bound| is_derived(bound, program_len))
        .filter_map(|bound| Some((bound.node, bound.operands().first()?.0)))
        .collect();
    resolved
        .iter()
        .filter(|bound| !is_derived(bound, program_len))
        .cloned()
        .map(|mut bound| {
            if let BoundOpKind::Reduce { operands, .. } = &mut bound.kind {
                operands.iter_mut().for_each(|operand| {
                    operand.0 = sources.get(&operand.0).copied().unwrap_or(operand.0);
                });
            }
            bound
        })
        .collect()
}
