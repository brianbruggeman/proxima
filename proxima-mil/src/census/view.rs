//! Operand views: how an operand's [`IndexMap`] reaches MIL, and the
//! contraction test. Pure functions of the algebra's patterns and the
//! resolved extents.

use alloc::vec::Vec;

use proxima_tensor::{AxisIndex, AxisTerm, IndexMap, IndexPattern, NodeId, Reduce, Shapes};

use super::Refusal;

pub(super) fn operand_views(
    map: &IndexMap,
    extents: &[u64],
    operand: usize,
) -> Result<Vec<&'static str>, Refusal> {
    match map {
        IndexMap::Affine(pattern) => pattern_views(pattern, extents, None, operand),
        IndexMap::Computed {
            index_map,
            base,
            gathered_dim,
            ..
        } => {
            let skipped = Some(usize::from(*gathered_dim));
            let mut views = pattern_views(base, extents, skipped, operand)?;
            views.extend(pattern_views(index_map, extents, None, operand)?);
            views.push("gather");
            Ok(views)
        }
    }
}

fn pattern_views(
    pattern: &IndexPattern,
    extents: &[u64],
    skipped: Option<usize>,
    operand: usize,
) -> Result<Vec<&'static str>, Refusal> {
    let mut order: Vec<u16> = Vec::new();
    let mut views: Vec<&'static str> = Vec::new();
    for (position, axis) in pattern.axes.iter().enumerate() {
        if skipped == Some(position) {
            continue;
        }
        let used = axis_use(axis, extents, operand)?;
        used.view
            .into_iter()
            .for_each(|name| push_once(&mut views, name));
        order.extend(used.axes);
    }
    if has_repeat(&order) {
        return Err(Refusal::DiagonalRead { operand });
    }
    views.extend(alignment_views(&order, extents.len()));
    Ok(views)
}

struct AxisUse {
    axes: Vec<u16>,
    view: Option<&'static str>,
}

fn axis_use(axis: &AxisIndex, extents: &[u64], operand: usize) -> Result<AxisUse, Refusal> {
    match axis.terms.as_slice() {
        [] => Ok(AxisUse {
            axes: Vec::new(),
            view: Some("slice_by_index"),
        }),
        [term] => Ok(AxisUse {
            axes: alloc::vec![term.axis],
            view: (term.coeff != 1 || axis.offset != 0 || axis.len.is_some())
                .then_some("slice_by_index"),
        }),
        terms => merged_axes(terms, axis.offset, extents)
            .map(|axes| AxisUse {
                axes,
                view: Some("reshape"),
            })
            .ok_or(Refusal::WindowedRead { operand }),
    }
}

fn merged_axes(terms: &[AxisTerm], offset: i32, extents: &[u64]) -> Option<Vec<u16>> {
    let mut ordered = terms.to_vec();
    ordered.sort_by_key(|term| core::cmp::Reverse(term.coeff));
    let row_major = offset == 0
        && ordered.last().is_some_and(|term| term.coeff == 1)
        && ordered
            .windows(2)
            .all(|pair| strides_nest(&pair[0], &pair[1], extents));
    row_major.then(|| ordered.iter().map(|term| term.axis).collect())
}

fn strides_nest(outer: &AxisTerm, inner: &AxisTerm, extents: &[u64]) -> bool {
    extents
        .get(usize::from(inner.axis))
        .and_then(|extent| i64::try_from(*extent).ok())
        .and_then(|extent| extent.checked_mul(i64::from(inner.coeff)))
        .is_some_and(|stride| stride == i64::from(outer.coeff))
}

fn has_repeat(order: &[u16]) -> bool {
    order
        .iter()
        .enumerate()
        .any(|(position, axis)| order[..position].contains(axis))
}

fn alignment_views(order: &[u16], rank: usize) -> Vec<&'static str> {
    let mut views = Vec::new();
    if order.windows(2).any(|pair| pair[0] > pair[1]) {
        views.push("transpose");
    }
    let mut sorted = order.to_vec();
    sorted.sort_unstable();
    let leading = rank.saturating_sub(sorted.len());
    let trailing = sorted
        .iter()
        .enumerate()
        .all(|(position, axis)| usize::from(*axis) == leading + position);
    if !trailing {
        views.push("expand_dims");
    }
    views
}

fn push_once(views: &mut Vec<&'static str>, name: &'static str) {
    if !views.contains(&name) {
        views.push(name);
    }
}

pub(super) fn reduce_extents(
    map: &IndexMap,
    shapes: &Shapes,
    operand: NodeId,
) -> Result<Vec<u64>, Refusal> {
    let mut resolved: Vec<Option<u64>> = alloc::vec![None; usize::from(map.affine().iter_rank)];
    fill_extents(map.affine(), shapes.of(operand), &mut resolved);
    if let IndexMap::Computed {
        indices, index_map, ..
    } = map
    {
        fill_extents(index_map, shapes.of(*indices), &mut resolved);
    }
    resolved
        .into_iter()
        .map(|extent| extent.ok_or(Refusal::WindowedRead { operand: 0 }))
        .collect()
}

fn fill_extents(pattern: &IndexPattern, shape: &[u64], resolved: &mut [Option<u64>]) {
    for (position, axis) in pattern.axes.iter().enumerate() {
        if let ([term], Some(extent)) = (axis.terms.as_slice(), shape.get(position))
            && term.coeff == 1
            && axis.offset == 0
            && let Some(slot) = resolved.get_mut(usize::from(term.axis))
        {
            *slot = Some(*extent);
        }
    }
}

/// A contraction needs a shared reduced axis and a free axis read by exactly one
/// operand; a product whose free axes both operands share (a sum of squares) is a
/// reduce over a multiply, which bge-small has 25 of (its layer norms).
pub(super) fn is_contraction(reduce: &Reduce, left: &IndexMap, right: &IndexMap) -> bool {
    let (
        IndexMap::Affine(left),
        IndexMap::Affine(right),
        IndexMap::Affine(input),
        IndexMap::Affine(output),
    ) = (left, right, &reduce.in_map, &reduce.out_map)
    else {
        return false;
    };
    let rank = usize::from(input.iter_rank);
    let (left_used, right_used) = (used_axes(left, rank), used_axes(right, rank));
    let kept = used_axes(output, rank);
    let reduced: Vec<usize> = (0..rank).filter(|axis| !kept[*axis]).collect();
    let exclusive = (0..rank).any(|axis| kept[axis] && left_used[axis] != right_used[axis]);
    is_identity(input)
        && exclusive
        && !reduced.is_empty()
        && reduced
            .iter()
            .all(|axis| left_used[*axis] && right_used[*axis])
}

fn used_axes(pattern: &IndexPattern, rank: usize) -> Vec<bool> {
    (0..rank)
        .map(|axis| {
            pattern
                .axes
                .iter()
                .flat_map(|index| index.terms.iter())
                .any(|term| usize::from(term.axis) == axis)
        })
        .collect()
}

fn is_identity(pattern: &IndexPattern) -> bool {
    usize::from(pattern.iter_rank) == pattern.axes.len()
        && pattern.axes.iter().enumerate().all(|(position, axis)| {
            axis.offset == 0
                && axis.len.is_none()
                && matches!(axis.terms.as_slice(), [term] if term.coeff == 1 && usize::from(term.axis) == position)
        })
}
