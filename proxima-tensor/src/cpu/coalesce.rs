use super::*;
use crate::bind::{Layout, Lookup};
use crate::sized::MAX_INLINE_RANK;
use core::cell::Cell;
use smallvec::SmallVec;

std::thread_local! {
    /// Per-thread switch for [`coalesce_dims`], defaulting to on. Thread-local
    /// so a degenerate-control test or a paired bench can run the uncoalesced
    /// arm without racing every other test sharing the process; the dispatch
    /// that consults it (`run_reduce`, `run_elementwise_dispatch`) runs on
    /// the calling thread.
    static DIM_COALESCING_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Bench/test-only escape valve for the switch above. Not part of the taught
/// public surface: a caller composing tensor programs never needs it.
#[doc(hidden)]
pub fn set_dim_coalescing_enabled(enabled: bool) {
    DIM_COALESCING_ENABLED.set(enabled);
}

/// One run of adjacent iteration axes that address memory as a single axis,
/// plus the stride each layout (operands first, output layout last) sees on
/// the merged axis.
struct AxisGroup {
    first_axis: usize,
    last_axis: usize,
    extent: u64,
    strides: SmallVec<[i64; MAX_INLINE_RANK]>,
}

/// Which side of a fold an iteration axis sits on. Elementwise ops have only
/// `Output`; a reduce never merges an axis it keeps with one it folds away.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AxisClass {
    Output,
    Reduction,
}

/// Merges adjacent iteration axes whose addressing is one affine run in
/// EVERY layout of the op, so the executor sees one wide contiguous axis
/// where the bound op spells several narrow ones. Axes `a` (outer) and `a+1`
/// (inner) merge when, for every layout, `stride[a] == stride[a+1] *
/// extent[a+1]` -- the row-major contiguity chain `axes_flat_chain` already
/// checks for whole outer ranges -- or when either axis has extent 1 (a
/// single legal coordinate contributes nothing to addressing). A broadcast
/// pair (both strides 0) satisfies the equation as written, and negative
/// strides compose the same way, so neither needs a case.
///
/// The merged axis keeps the inner axis' stride and walks the product of the
/// extents in the same order the pair did, so every output position sees the
/// same operand elements in the same reduction order: results are
/// bit-identical to the uncoalesced op, only the loop nest is flatter. This
/// is what turns a delta-net state `[head_k, head_v, kv_heads, group=2]`
/// whose `kv_heads, group` pair is contiguous in every operand from a
/// width-2 row into a width-32 row, before `run_reduce`/`run_elementwise`
/// choose their row width.
///
/// Returns `None` when nothing merges, when the op is not a plain
/// `Elementwise` or `Keep::Reduce` fold, or when any operand is gathered, the
/// fold scatters, or an epilogue is fused in (its operands are addressed in
/// the pre-merge output-axis space). Only the executor's copy is rewritten:
/// the caller's `BoundOp` is untouched, so bind-level consumers and the
/// post-fold epilogue keep their original axes.
pub(super) fn coalesce_dims(resolved: &BoundOp) -> Option<BoundOp> {
    if !DIM_COALESCING_ENABLED.get() || resolved.extents.len() < 2 {
        return None;
    }
    if resolved.extents.contains(&0) {
        return None;
    }
    let (operands, output_fold) = coalescible_layouts(resolved)?;
    let layouts: SmallVec<[&Layout; 8]> = operands
        .iter()
        .map(|(_, layout, _)| layout)
        .chain(output_fold.map(|(layout, _)| layout))
        .collect();
    let output_axes = output_fold.map(|(_, axes)| axes);
    let groups = group_axes(&resolved.extents, &layouts, output_axes);
    if groups.len() == resolved.extents.len() {
        return None;
    }
    Some(rebuild_coalesced(resolved, &groups))
}

/// One bound operand: source node, layout, optional gather.
type Operand = (NodeId, Layout, Option<Lookup>);

/// The fold's `(out_layout, output_axes)` pair.
type FoldView<'a> = (&'a Layout, &'a [u16]);

/// The layouts [`coalesce_dims`] may rewrite, or `None` when the op shape is
/// outside its contract. The second field is the fold's `(out_layout,
/// output_axes)` for a reduce.
fn coalescible_layouts(resolved: &BoundOp) -> Option<(&[Operand], Option<FoldView<'_>>)> {
    match &resolved.kind {
        BoundOpKind::Elementwise { operands, .. }
            if operands.iter().all(|(_, _, gather)| gather.is_none()) =>
        {
            Some((operands, None))
        }
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            operands,
            output_axes,
            out_layout,
            out_scatter: None,
            epilogue_operands,
            epilogue_broadcast_axes,
            ..
        } if operands.iter().all(|(_, _, gather)| gather.is_none())
            && epilogue_operands.is_empty()
            && epilogue_broadcast_axes.is_empty() =>
        {
            Some((operands, Some((out_layout, output_axes.as_slice()))))
        }
        _ => None,
    }
}

/// Greedy outer-to-inner scan: each axis either extends the current group or
/// opens a new one.
fn group_axes(extents: &[u64], layouts: &[&Layout], output_axes: Option<&[u16]>) -> Vec<AxisGroup> {
    let mut groups: Vec<AxisGroup> = Vec::with_capacity(extents.len());
    for (axis, &extent) in extents.iter().enumerate() {
        let strides: SmallVec<[i64; MAX_INLINE_RANK]> = layouts
            .iter()
            .map(|layout| layout.stride(axis as u16))
            .collect();
        match groups.last_mut() {
            Some(group) if can_extend(group, axis, extent, &strides, output_axes) => {
                extend(group, axis, extent, &strides);
            }
            _ => groups.push(AxisGroup {
                first_axis: axis,
                last_axis: axis,
                extent,
                strides,
            }),
        }
    }
    groups
}

fn axis_class(axis: usize, output_axes: Option<&[u16]>) -> AxisClass {
    match output_axes {
        Some(axes) if !axes.contains(&(axis as u16)) => AxisClass::Reduction,
        _ => AxisClass::Output,
    }
}

/// True when `axis` (extent `extent`, per-layout `strides`) can become the
/// new innermost axis of `group`.
fn can_extend(
    group: &AxisGroup,
    axis: usize,
    extent: u64,
    strides: &[i64],
    output_axes: Option<&[u16]>,
) -> bool {
    if axis_class(group.last_axis, output_axes) != axis_class(axis, output_axes) {
        return false;
    }
    if !output_axes_are_adjacent(group.last_axis, axis, output_axes) {
        return false;
    }
    if extent == 1 || group.extent == 1 {
        return true;
    }
    group
        .strides
        .iter()
        .zip(strides)
        .all(|(&outer_stride, &inner_stride)| outer_stride == inner_stride * extent as i64)
}

/// Two kept axes merge only when the fold writes them consecutively, in this
/// order: the output's loop order is `output_axes`' order, and merging across
/// a gap or a swap would change which output position each step writes.
fn output_axes_are_adjacent(outer: usize, inner: usize, output_axes: Option<&[u16]>) -> bool {
    let Some(axes) = output_axes else {
        return true;
    };
    let outer = outer as u16;
    let inner = inner as u16;
    if !axes.contains(&outer) || !axes.contains(&inner) {
        return true;
    }
    axes.windows(2)
        .any(|pair| pair[0] == outer && pair[1] == inner)
}

/// Absorbs `axis` as `group`'s new innermost axis. The merged stride is the
/// inner axis' own stride, except when the inner axis has extent 1 and so
/// contributes no addressing: then the group keeps its existing stride.
fn extend(group: &mut AxisGroup, axis: usize, extent: u64, strides: &[i64]) {
    if extent != 1 {
        if group.extent == 1 {
            group.strides.clear();
            group.strides.extend_from_slice(strides);
        } else {
            for (merged, &inner) in group.strides.iter_mut().zip(strides) {
                *merged = inner;
            }
        }
    }
    group.extent *= extent;
    group.last_axis = axis;
}

fn rebuild_coalesced(resolved: &BoundOp, groups: &[AxisGroup]) -> BoundOp {
    let mut coalesced = resolved.clone();
    coalesced.extents = groups.iter().map(|group| group.extent).collect();
    let layout_count = resolved.operands().len();
    match &mut coalesced.kind {
        BoundOpKind::Elementwise { operands, .. } => {
            rewrite_layouts(operands, None, groups, layout_count);
        }
        BoundOpKind::Reduce {
            operands,
            output_axes,
            out_layout,
            ..
        } => {
            rewrite_layouts(operands, Some(out_layout), groups, layout_count);
            *output_axes = remap_output_axes(output_axes, groups);
        }
        _ => {}
    }
    coalesced
}

fn rewrite_layouts(
    operands: &mut [Operand],
    out_layout: Option<&mut Layout>,
    groups: &[AxisGroup],
    layout_count: usize,
) {
    for (slot, (_, layout, _)) in operands.iter_mut().enumerate() {
        layout.strides = groups.iter().map(|group| group.strides[slot]).collect();
    }
    if let Some(layout) = out_layout {
        layout.strides = groups
            .iter()
            .map(|group| group.strides[layout_count])
            .collect();
    }
}

/// Each kept axis becomes its group's index; merged kept axes were adjacent
/// in `output_axes`, so the duplicates are consecutive.
fn remap_output_axes(
    output_axes: &SmallVec<[u16; MAX_INLINE_RANK]>,
    groups: &[AxisGroup],
) -> SmallVec<[u16; MAX_INLINE_RANK]> {
    let mut remapped: SmallVec<[u16; MAX_INLINE_RANK]> = SmallVec::new();
    for &axis in output_axes {
        let group_index = groups
            .iter()
            .position(|group| (group.first_axis..=group.last_axis).contains(&(axis as usize)))
            .map_or(axis, |index| index as u16);
        if remapped.last() != Some(&group_index) {
            remapped.push(group_index);
        }
    }
    remapped
}
