use super::*;

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
/// fold scatters, or the fold's epilogue is gathered or re-broadcasts over a
/// reduced axis.
///
/// A fused plain epilogue joins the contiguity check: its operand layouts are
/// addressed by position in `output_axes` (`apply_plain_reduce_epilogue`), so
/// two kept axes merge only when every epilogue layout is also one affine run
/// over that adjacent pair, and the epilogue layouts are then rewritten
/// through the same groups. Only the executor's copy is rewritten: the
/// caller's `BoundOp` is untouched, so bind-level consumers keep their
/// original axes.
#[must_use]
pub fn coalesce_dims(resolved: &BoundOp) -> Option<BoundOp> {
    if resolved.extents.len() < 2 {
        return None;
    }
    if resolved.extents.contains(&0) {
        return None;
    }
    let (operands, output_fold) = coalescible_layouts(resolved)?;
    let layouts: SmallVec<[&Layout; 8]> = operands
        .iter()
        .map(|(_, layout, _)| layout)
        .chain(output_fold.map(|(layout, _, _)| layout))
        .chain(
            output_fold
                .into_iter()
                .flat_map(|(_, _, epilogue)| epilogue.iter().map(|(_, layout, _)| layout)),
        )
        .collect();
    let output_axes = output_fold.map(|(_, axes, _)| axes);
    let local_from = operands.len() + 1;
    let groups = group_axes(&resolved.extents, &layouts, output_axes, local_from);
    if groups.len() == resolved.extents.len() {
        return None;
    }
    Some(rebuild_coalesced(resolved, &groups))
}

/// One bound operand: source node, layout, optional gather.
type Operand = (NodeId, Layout, Option<Lookup>);

/// The fold's `(out_layout, output_axes, epilogue_operands)` triple.
type FoldView<'a> = (&'a Layout, &'a [u16], &'a [Operand]);

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
            && epilogue_operands
                .iter()
                .all(|(_, _, gather)| gather.is_none())
            && epilogue_broadcast_axes.is_empty() =>
        {
            Some((
                operands,
                Some((out_layout, output_axes.as_slice(), epilogue_operands)),
            ))
        }
        _ => None,
    }
}

/// Greedy outer-to-inner scan: each axis either extends the current group or
/// opens a new one.
///
/// Layouts from slot `local_from` on are epilogue operands, addressed by
/// position in `output_axes` rather than by iteration axis: their stride on an
/// axis is the one at that axis' position, and `0` on an axis the fold reduces.
fn group_axes(
    extents: &[u64],
    layouts: &[&Layout],
    output_axes: Option<&[u16]>,
    local_from: usize,
) -> Vec<AxisGroup> {
    let mut groups: Vec<AxisGroup> = Vec::with_capacity(extents.len());
    for (axis, &extent) in extents.iter().enumerate() {
        let strides: SmallVec<[i64; MAX_INLINE_RANK]> = layouts
            .iter()
            .enumerate()
            .map(|(slot, layout)| {
                if slot < local_from {
                    return layout.stride(axis as u16);
                }
                output_axes
                    .and_then(|axes| axes.iter().position(|&kept| kept as usize == axis))
                    .map_or(0, |position| layout.stride(position as u16))
            })
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
    if leading_unit_axis_is_a_row_axis(group, axis, extent, output_axes) {
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

/// A fold's extent-1 kept axis in front of a wider kept axis is the row axis
/// `width_tile_plan`/`neon_tile_plan` key on (an `M = 1` matmul written
/// `[m, n, k]`): absorbing it into the axis behind it leaves the fold with
/// only a width axis, and both tiles decline that shape (`AxesShape`).
/// Absorbing it adds no contiguity either -- the merged group keeps the inner
/// axis' strides and extent -- so the pair stays split. An extent-1 axis
/// BEHIND a wider one still merges: that widens the row (`[n, 1]` -> `[n]`).
fn leading_unit_axis_is_a_row_axis(
    group: &AxisGroup,
    axis: usize,
    extent: u64,
    output_axes: Option<&[u16]>,
) -> bool {
    output_axes.is_some()
        && group.extent == 1
        && extent != 1
        && axis_class(axis, output_axes) == AxisClass::Output
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
            epilogue_operands,
            ..
        } => {
            rewrite_layouts(operands, Some(out_layout), groups, layout_count);
            rewrite_epilogue_layouts(epilogue_operands, output_axes, groups, layout_count + 1);
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

/// An epilogue layout has one stride per entry of the remapped `output_axes`,
/// each read from the group that entry names. `first_slot` is the slot of the
/// first epilogue operand in every group's `strides`.
fn rewrite_epilogue_layouts(
    epilogue_operands: &mut [Operand],
    output_axes: &[u16],
    groups: &[AxisGroup],
    first_slot: usize,
) {
    let kept_groups = remap_output_axes(output_axes, groups);
    for (offset, (_, layout, _)) in epilogue_operands.iter_mut().enumerate() {
        layout.strides = kept_groups
            .iter()
            .map(|&group| groups[group as usize].strides[first_slot + offset])
            .collect();
    }
}

/// Each kept axis becomes its group's index; merged kept axes were adjacent
/// in `output_axes`, so the duplicates are consecutive.
fn remap_output_axes(
    output_axes: &[u16],
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
