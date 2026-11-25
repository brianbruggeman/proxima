//! Two rewrites that let a multiply-add reduce reach the NEON dot tile
//! ([`neon_tile_decision`]) when its shape is slightly outside the tile's
//! gate. Both are generic over the reduce: nothing here names a model or a
//! shape, and each only fires when the tile gate would then pass.
//!
//! - [`peel_scalar_factor`]: a body `Multiply(Multiply(a, s), b)` (any
//!   operand order) whose `s` is a reduce-invariant scalar becomes the
//!   one-step `Multiply(a, b)` the gate recognises; the caller multiplies the
//!   finished output by `s` once per element ([`scale_output`]). This
//!   reassociates the scale out of the sum; see the policy note on
//!   [`peel_scalar_factor`].
//! - [`split_leading_chunks`]: a reduce with several leading output axes is
//!   cut along its outermost axis into one single-extent chunk per index
//!   (`BoundOp::split`, the machinery the parallel driver already uses), so
//!   each chunk is the 2-D shape the tile takes.
//!
//! Composes: [`BoundOp::split`], [`resolve_reduce_axis_shape`],
//! [`body_shape`], and the tile gate `run_reduce_planned` consults.

use core::ops::MulAssign;

use crate::bind::{BodyStep, ComposedBody, StepArg};

use super::*;

/// Whether the dot-tile gate would admit `op` once the reduction is treated
/// as flat (a non-flat operand is packed by the tile) and the row axis is
/// its innermost leading axis. Decided from strides and shapes only, and
/// through [`neon_tile_decision`] directly so the decline census is not
/// touched by a probe.
fn tile_gate_open(op: &BoundOp) -> bool {
    let BoundOpKind::Reduce {
        reduce_op,
        init,
        output_axes,
        keep: Keep::Reduce,
        out_scatter: None,
        ..
    } = &op.kind
    else {
        return false;
    };
    let axes = resolve_reduce_axis_shape(op, output_axes.as_slice());
    let Some(&innermost) = axes.leading_output_axes.last() else {
        return false;
    };
    let strides: Vec<i64> = op
        .operands()
        .iter()
        .map(|(_, view, _)| axes.last_output_dim.map_or(0, |dim| view.stride(dim)))
        .collect();
    let flat = vec![1i64; op.operands().len()];
    let shape = body_shape(op.element_body());
    // a width-stepped body takes the width tile, not the dot tile this probes
    if body_shape_is_affine_fast_path(op, &shape, &strides) {
        return false;
    }
    neon_tile_decision(
        op,
        &shape,
        *reduce_op,
        !matches!(init, ReduceInit::FirstElement),
        &flat,
        &strides,
        &[innermost],
    )
    .is_ok()
}

/// A reduce-invariant scalar: every stride zero and no gather, so it reads
/// one element at every iteration point.
fn is_invariant_scalar(resolved: &BoundOp, operand: u16) -> bool {
    let (_, view, gather) = &resolved.operands()[operand as usize];
    gather.is_none() && view.strides.iter().all(|&stride| stride == 0)
}

/// The `(a, b, s)` of a two-step all-`Multiply` body with exactly one
/// reduce-invariant scalar `s`, in either association.
fn scaled_multiply_roles(resolved: &BoundOp) -> Option<(u16, u16, u16)> {
    let [inner, outer] = resolved.element_body().steps.as_slice() else {
        return None;
    };
    if inner.op != ScalarOp::Multiply || outer.op != ScalarOp::Multiply {
        return None;
    }
    let [StepArg::Operand(first), StepArg::Operand(second)] = inner.args.as_slice() else {
        return None;
    };
    let outer_operand = match outer.args.as_slice() {
        [StepArg::Step(0), StepArg::Operand(operand)]
        | [StepArg::Operand(operand), StepArg::Step(0)] => *operand,
        _ => return None,
    };
    let scalar = |operand: u16| is_invariant_scalar(resolved, operand);
    match (scalar(*first), scalar(*second), scalar(outer_operand)) {
        (false, false, true) => Some((*first, *second, outer_operand)),
        (true, false, false) => Some((*second, outer_operand, *first)),
        (false, true, false) => Some((*first, outer_operand, *second)),
        _ => None,
    }
}

/// Recognises `Multiply(Multiply(a, s), b)` (see [`scaled_multiply_roles`])
/// as `Binary(Multiply, a, b)`, returning the rewritten op and the operand
/// index holding `s`; the caller reads `s` and calls [`scale_output`] on the
/// finished output. Fires only for an `Add` reduce seeded with zero (a
/// nonzero seed would be scaled too) whose peeled op then opens the tile gate
/// ([`tile_gate_open`], or [`split_leading_chunks`] for several leading
/// axes). Anything else returns `None` and keeps the generic interpreter.
///
/// Numeric policy: pulling `s` out of the sum reassociates it, the
/// permission `NumericPolicy::reassociation` names
/// (`proxima-numeric/src/numeric.rs:37`), which `bind` checks through `admit`
/// (`proxima-tensor/src/bind/mod.rs:79`). `cpu::run_reduce` is handed no
/// policy, and the dot tile this feeds already reorders the sum with no
/// policy check, so the rewrite is not gated here; its bound is the tile
/// tests' own tolerance.
pub(super) fn peel_scalar_factor(resolved: &BoundOp) -> Option<(BoundOp, u16)> {
    let BoundOpKind::Reduce {
        reduce_op: ScalarOp::Add,
        init: ReduceInit::Zero,
        keep: Keep::Reduce,
        out_scatter: None,
        ..
    } = &resolved.kind
    else {
        return None;
    };
    let (first, second, scalar) = scaled_multiply_roles(resolved)?;
    let mut peeled = resolved.clone();
    let BoundOpKind::Reduce { element_body, .. } = &mut peeled.kind else {
        return None;
    };
    *element_body = ComposedBody {
        steps: vec![BodyStep {
            op: ScalarOp::Multiply,
            args: vec![StepArg::Operand(first), StepArg::Operand(second)],
        }],
    };
    let reachable = tile_gate_open(&peeled) || split_leading_chunks(&peeled).is_some();
    reachable.then_some((peeled, scalar))
}

/// Multiplies every element `resolved` wrote to `output` by `scale`: one
/// contiguous pass when the output layout is dense row-major over its
/// output axes, otherwise one write per coordinate.
pub(super) fn scale_output<Element: Copy + MulAssign>(
    resolved: &BoundOp,
    scale: Element,
    output: &mut [Element],
) {
    let BoundOpKind::Reduce {
        output_axes,
        out_layout,
        ..
    } = &resolved.kind
    else {
        return;
    };
    let extents: Vec<u64> = output_axes
        .iter()
        .map(|&axis| resolved.extents[axis as usize])
        .collect();
    let total = odometer_len(&extents) as usize;
    let mut expected = 1i64;
    let dense = output_axes.iter().rev().all(|&axis| {
        let matches = out_layout.stride(axis) == expected;
        expected = expected.saturating_mul(resolved.extents[axis as usize] as i64);
        matches
    });
    let span = usize::try_from(out_layout.base)
        .ok()
        .and_then(|start| output.get_mut(start..start + total));
    if let (true, Some(span)) = (dense, span) {
        span.iter_mut().for_each(|value| *value *= scale);
        return;
    }
    let mut coordinate = vec![0u64; extents.len()];
    let mut full = vec![0u64; resolved.extents.len()];
    for flat in 0..total as u64 {
        unflatten_into(flat, &extents, &mut coordinate);
        for (&axis, &index) in output_axes.iter().zip(&coordinate) {
            full[axis as usize] = index;
        }
        if let Some(value) = usize::try_from(out_layout.offset_of(&full))
            .ok()
            .and_then(|position| output.get_mut(position))
        {
            *value *= scale;
        }
    }
}

/// One single-extent chunk per index of `resolved`'s outermost output axis,
/// with the output stride of that axis (chunk `index` writes the suffix of
/// `output` starting at `index * stride`), when the reduce has several
/// leading output axes and the tile gate opens on a chunk. The gate then
/// reads the innermost leading axis only: B must be row-invariant over it
/// (`row_stride_b == 0`, `gemm_dot_quant.rs`), while the outer axes it may
/// vary over are removed by the split. `None` leaves today's decline reason
/// (`leading_axes`) in place.
pub(super) fn split_leading_chunks(resolved: &BoundOp) -> Option<(Vec<BoundOp>, i64)> {
    let BoundOpKind::Reduce {
        output_axes,
        out_layout,
        keep: Keep::Reduce,
        out_scatter: None,
        epilogue_broadcast_axes,
        ..
    } = &resolved.kind
    else {
        return None;
    };
    let axes = resolve_reduce_axis_shape(resolved, output_axes.as_slice());
    let outer = *output_axes.first()?;
    if axes.leading_output_axes.len() < 2
        || axes.leading_output_axes.first() != Some(&outer)
        || !epilogue_broadcast_axes.is_empty()
    {
        return None;
    }
    let stride = out_layout.stride(outer);
    if stride < 0 {
        return None;
    }
    let extent = usize::try_from(resolved.extents[outer as usize]).ok()?;
    let chunks = resolved.split(extent)?;
    tile_gate_open(chunks.first()?).then_some((chunks, stride))
}
