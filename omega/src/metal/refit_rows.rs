//! Moving a [`Plan`] to another row count in place: the Metal counterpart of
//! `CudaPlan::repack_rows`. The bound ops' integers (extents, strides, bases, lookup extents) are
//! affine in the row count within the wide class, so a [`RowTemplate`] fitted from three binds of
//! the program re-derives them without a bind per refit; the plan-owned uniform buffers and the
//! resolved steps of the ops that moved follow through [`apply_refit`], and the arena, the
//! retirement schedule and the barrier groups stay as built.

use super::*;

/// The fewest rows the prefill kernels share one form for: below it cached attention runs untiled
/// (CUDA's `ROW_CLASS_MIN`).
const ROW_CLASS_MIN: u64 = 16;

/// Moves `plan` to `symbols` when only `symbols[0]` changed and both row counts are in the wide
/// class and within the arena's capacity, returning `true`; `false` leaves the plan untouched.
/// Never a partial refit: every fallible step runs before the first write.
///
/// # Errors
/// Shape inference, binding or kernel resolution failing; the plan is unchanged on `Err`.
pub(super) fn refit_rows(plan: &mut Plan, symbols: &[u64]) -> Result<bool, MetalError> {
    let Some(rows) = symbols.first().copied() else {
        return Ok(false);
    };
    if !admits_rows(plan, symbols, rows) || !learn_template(plan, symbols, rows)? {
        return Ok(false);
    }
    let next_shapes = infer(&plan.program, symbols)?;
    let mut next = plan.prepared.resolved.clone();
    let Some(attention) = apply_template(plan, rows, &mut next) else {
        return Ok(false);
    };
    for position in attention {
        let Some(patched) = refresh_cached_attention(&next[position], &next_shapes) else {
            return Ok(false);
        };
        next[position] = patched;
    }
    let patches: Vec<(usize, BoundOp)> = next
        .into_iter()
        .enumerate()
        .filter(|(position, bound)| *bound != plan.prepared.resolved[*position])
        .collect();
    apply_refit(plan, patches, next_shapes)?;
    #[cfg(feature = "metal-plan-stable-buffers")]
    if plan.arena.get().is_none() {
        plan.row_capacity.set(rows);
    }
    plan.bound_symbols = symbols.to_vec();
    Ok(true)
}

fn bound_rows(plan: &Plan) -> u64 {
    plan.bound_symbols.first().copied().unwrap_or(1)
}

fn admits_rows(plan: &Plan, symbols: &[u64], rows: u64) -> bool {
    let bound = bound_rows(plan);
    let same_rest = symbols.get(1..) == plan.bound_symbols.get(1..);
    // a configured dense-mma layout is chosen from the bound operands, which a rebind does not redo
    let plain = plan.dense_mma_input_layout.is_none();
    plain
        && same_rest
        && rows != bound
        && rows >= ROW_CLASS_MIN
        && bound >= ROW_CLASS_MIN
        && rows <= plan.row_capacity.get()
}

fn apply_template(plan: &Plan, rows: u64, next: &mut [BoundOp]) -> Option<Vec<usize>> {
    let learned = plan.rows_template.borrow();
    let Some(Some(template)) = learned.as_ref() else {
        return None;
    };
    apply_row_template(template, rows, next)
}

/// A third row count in the wide class, distinct from the two the template is fitted across.
fn third_rows(bound: u64, rows: u64, capacity: u64) -> Option<u64> {
    [
        bound / 2 + rows / 2 + (bound & rows & 1),
        bound / 2,
        rows.saturating_add(1),
        rows.saturating_sub(1),
        bound.saturating_sub(1),
    ]
    .into_iter()
    .find(|candidate| {
        *candidate >= ROW_CLASS_MIN
            && *candidate <= capacity
            && *candidate != bound
            && *candidate != rows
    })
}

fn bind_at(plan: &Plan, symbols: &[u64], rows: u64) -> Result<Vec<BoundOp>, MetalError> {
    let mut moved = symbols.to_vec();
    moved[0] = rows;
    rebind_at(&plan.program, &moved, &plan.prepared, &plan.rebind).map(|prepared| prepared.resolved)
}

/// Fits the template on the first rows refit of a plan (three binds: the plan's own, `rows` and a
/// third count) and remembers the verdict. `Ok(false)` when the ops are not row-affine.
fn learn_template(plan: &Plan, symbols: &[u64], rows: u64) -> Result<bool, MetalError> {
    if let Some(verdict) = plan.rows_template.borrow().as_ref() {
        return Ok(verdict.is_some());
    }
    let bound = bound_rows(plan);
    let third = third_rows(bound, rows, plan.row_capacity.get().max(rows));
    let template = match third {
        Some(third) if rows_enumerable(&plan.prepared.resolved) => {
            let first = bind_at(plan, symbols, rows)?;
            let second = bind_at(plan, symbols, third)?;
            row_template(&[
                (bound, plan.prepared.resolved.as_slice()),
                (rows, first.as_slice()),
                (third, second.as_slice()),
            ])
        }
        _ => None,
    };
    let fitted = template.is_some();
    *plan.rows_template.borrow_mut() = Some(template);
    Ok(fitted)
}

/// A row count to fit the template at that is not the plan's own: half of it, else one below, else
/// one above, inside the wide class and the arena's capacity.
fn probe_rows(bound: u64, capacity: u64) -> Option<u64> {
    [bound / 2, bound.saturating_sub(1), bound.saturating_add(1)]
        .into_iter()
        .find(|candidate| *candidate >= ROW_CLASS_MIN && *candidate <= capacity && *candidate != bound)
}

/// Fits and remembers `plan`'s row template without moving the plan, so the first
/// [`refit_rows`] only applies it. Runs the same [`learn_template`] a refit would, at a probe row
/// count inside the wide class and the arena's capacity; the bound ops, uniforms, grids and
/// `bound_symbols` are untouched. `Ok(true)` when the plan is row-affine and the template is
/// cached (already or now); `Ok(false)` when the plan cannot refit by rows.
///
/// # Errors
/// Binding failing at a probe row count; the plan is unchanged on `Err`.
pub(super) fn prime_row_template(plan: &Plan, symbols: &[u64]) -> Result<bool, MetalError> {
    let Some(probe) = probe_rows(bound_rows(plan), plan.row_capacity.get()) else {
        return Ok(false);
    };
    if !admits_rows(plan, symbols, probe) {
        return Ok(false);
    }
    learn_template(plan, symbols, probe)
}
