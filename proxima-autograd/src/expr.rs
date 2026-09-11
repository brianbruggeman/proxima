//! Graph-building primitives shared by [`crate::adjoint`], [`crate::activation`],
//! and [`crate::optimizer`] — every one of them a one-line composition of
//! [`proxima_tensor::op::append`] over [`proxima_tensor::map::projection`],
//! the same two primitives `proxima-tensor/src/spec.rs`'s own private
//! `elementwise`/`reduce` helpers compose (`spec.rs:542`, `spec.rs:567`).
//! This module differs only in *how* it builds an `IndexMap`: `spec.rs`
//! parses an einsum-style string (`"sd->sde"`), a TOML-facing convenience
//! private to that module; this crate builds
//! [`proxima_tensor::map::projection`] patterns directly, exactly as
//! `proxima-tensor/src/cpu.rs:16062`'s own
//! `scatter_add_into_a_known_destination_via_mask_composition` test does —
//! no string grammar is reinvented here.
//!
//! Every helper is infallible: it only ever pushes an `Op` onto the
//! caller's program. Whether the result actually lowers (arity, shape,
//! accumulator width) is judged once, by
//! [`proxima_tensor::shape::infer`], not re-validated per call here — the
//! same division of labour `op::append` itself already draws
//! (`proxima-tensor/src/op.rs:290-298`).

use alloc::vec::Vec;

use proxima_tensor::dtype::DType;
use proxima_tensor::map::{self, IndexMap, IndexPattern};
use proxima_tensor::op::{self, Extent, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp};

/// One operand: which node, and how the iteration space addresses it.
pub(crate) type Operand = (NodeId, IndexMap);

/// `operand[terms[n]] = iter[n]` for every axis, `0..iter_rank` in order —
/// the map an already-iter_rank-shaped node uses to read itself: no
/// broadcast, no permutation, no offset.
pub(crate) fn identity(iter_rank: u16) -> IndexMap {
    let axes: Vec<u16> = (0..iter_rank).collect();
    IndexMap::Affine(map::projection(iter_rank, &axes))
}

/// A rank-0 operand broadcasting into any consumer rank — the `"->stug"`
/// idiom [`Op::Constant`]'s own doc names (`proxima-tensor/src/op.rs:255`).
pub(crate) fn broadcast(iter_rank: u16) -> IndexMap {
    IndexMap::Affine(map::projection(iter_rank, &[]))
}

/// Every axis of `pattern` is exactly one unit-coefficient term — the same
/// rule `proxima-tensor/src/shape.rs:446` (`project_output_shape`) applies
/// to a `Reduce::out_map` before lowering it. An operand map failing this
/// cannot be reused as a backward `Reduce`'s `out_map`.
pub(crate) fn is_pure_projection(pattern: &IndexPattern) -> bool {
    pattern
        .axes
        .iter()
        .all(|axis| matches!(axis.terms.as_slice(), [term] if term.coeff == 1))
}

fn push(program: &mut Vec<Op>, dtype: DType, body: ScalarOp, operands: Vec<Operand>) -> NodeId {
    op::append(
        program,
        Op::Elementwise {
            dtype,
            body,
            operands,
            name: None,
        },
    )
}

pub(crate) fn unary(program: &mut Vec<Op>, dtype: DType, body: ScalarOp, a: Operand) -> NodeId {
    push(program, dtype, body, alloc::vec![a])
}

pub(crate) fn binary(
    program: &mut Vec<Op>,
    dtype: DType,
    body: ScalarOp,
    a: Operand,
    b: Operand,
) -> NodeId {
    push(program, dtype, body, alloc::vec![a, b])
}
/// A rank-0 [`Op::Constant`] — the literal-spelling idiom
/// `proxima-tensor/src/spec.rs:620`'s `scalar_constant` also composes.
pub(crate) fn constant(program: &mut Vec<Op>, dtype: DType, value: f32) -> NodeId {
    op::append(
        program,
        Op::Constant {
            dtype,
            shape: Vec::new(),
            value,
        },
    )
}

/// A same-shaped anchor at a caller-chosen `value`: [`proxima_tensor::op::Op::Constant`] carries
/// a concrete `Vec<Extent>`, not just a rank-0 broadcast (that variant's
/// own doc, `proxima-tensor/src/op.rs:256-258`: "a higher-rank constant is
/// what carries extents a consumer cannot otherwise infer"). A node
/// already reduced down to a lower rank has nothing of its own to anchor
/// every axis of a wider iteration space when broadcast back up alone —
/// `unify_iteration_space` needs at least one operand per axis with a real
/// term, and a fully-broadcast (empty-axes) operand supplies none. Pairing
/// the reduced node with a `value: 0.0` anchor via `Add` (adding zero) gives
/// shape inference that missing per-axis information without changing any
/// value; `crate::adjoint`'s `Select` adjoint rule reuses the same
/// constructor at `value: 1.0` for the identical reason -- a broadcast
/// condition operand can leave an axis just as unconstrained as a reduced
/// one does.
pub(crate) fn broadcast_anchor(
    program: &mut Vec<Op>,
    dtype: DType,
    extents: &[u64],
    value: f32,
) -> NodeId {
    let shape = extents
        .iter()
        .map(|&extent| Extent::Static(extent as u32))
        .collect();
    op::append(
        program,
        Op::Constant {
            dtype,
            shape,
            value,
        },
    )
}

/// Reconstructs the full iteration-space extents an already-validated
/// `Reduce`'s `in_map` (a [`is_pure_projection`] pattern) established —
/// the same per-axis resolution `proxima-tensor/src/shape.rs`'s
/// `unify_iteration_space` performs forward, read back from `shapes`
/// instead of recomputed, since the forward program already proved it
/// resolves (this crate's `differentiate` calls `shape::infer` before
/// touching any node).
pub(crate) fn iter_extents(
    shapes: &proxima_tensor::shape::Shapes,
    operand: NodeId,
    pattern: &IndexPattern,
) -> Vec<u64> {
    let operand_shape = shapes.of(operand);
    let mut extents = alloc::vec![0u64; pattern.iter_rank as usize];
    for (axis_index, axis) in pattern.axes.iter().enumerate() {
        if let [term] = axis.terms.as_slice() {
            extents[term.axis as usize] = operand_shape[axis_index];
        }
    }
    extents
}

/// One `Op::Reduce`, built directly rather than through `spec.rs`'s
/// einsum-string parser (private to that module) — same shape this crate's
/// own `cpu.rs:16096` scatter-add test builds by hand.
pub(crate) fn reduce(
    program: &mut Vec<Op>,
    dtype: DType,
    body: ScalarOp,
    init: ReduceInit,
    operand: NodeId,
    in_map: IndexMap,
    out_map: IndexMap,
) -> NodeId {
    op::append(
        program,
        Op::Reduce(Reduce {
            dtype,
            body,
            init,
            operand,
            in_map,
            out_map,
            keep: Keep::Reduce,
            name: None,
        }),
    )
}

/// [`reduce`], with `keep: Keep::Scan` instead of `Keep::Reduce` — every
/// prefix survives rather than only the final accumulator
/// (`proxima-tensor/src/op.rs:142-147`'s own doc on the two `Keep`
/// variants). `crate::adjoint`'s scan-add adjoint is the only caller today:
/// the reversed-suffix-sum derivation scans a reversed copy of the upstream
/// gradient, so it needs `Keep::Scan` with the same `Add`/`Zero` shape
/// [`reduce`] already builds for a plain reduction.
pub(crate) fn scan(
    program: &mut Vec<Op>,
    dtype: DType,
    body: ScalarOp,
    init: ReduceInit,
    operand: NodeId,
    in_map: IndexMap,
    out_map: IndexMap,
) -> NodeId {
    op::append(
        program,
        Op::Reduce(Reduce {
            dtype,
            body,
            init,
            operand,
            in_map,
            out_map,
            keep: Keep::Scan,
            name: None,
        }),
    )
}

/// A rank-1 axis-reversal read: `reversed[i] = operand[extent - 1 - i]`.
/// Expressed as a plain [`IndexMap::Affine`] with a negative-coefficient
/// [`AxisTerm`] — `proxima-tensor/src/bind.rs:1401-1409`'s `layout_of`
/// already folds any signed `coeff` into a signed stride, so this needs no
/// new expression form; a slice with a negative stride is exactly what
/// convolution's own two-term axis already proved this grammar carries
/// (this module's own doc table, "stride / dilation").
pub(crate) fn reverse_1d(extent: u64) -> Option<IndexMap> {
    let offset = i32::try_from(extent.saturating_sub(1)).ok()?;
    Some(IndexMap::Affine(map::affine(
        1,
        &[(&[map::AxisTerm::scaled(0, -1)], offset)],
    )))
}

/// Eliminates the `Add(narrow, zero_anchor)` pattern [`broadcast_anchor`]
/// builds purely to give shape inference a same-rank operand with a real
/// term on every axis (that function's own doc) -- whenever a `Multiply`
/// reads the add through the plain identity map, the anchor contributes
/// nothing the multiply's OTHER operand cannot supply on its own, so this
/// substitutes `narrow`'s own `(NodeId, IndexMap)` straight into the
/// multiply and leaves the add/constant pair as dead code
/// ([`proxima_tensor::live::annotate`] already drops unreferenced nodes
/// from the live set the multiply-consumer's arena build reads, so no new
/// liveness machinery is needed here).
///
/// `proxima-tensor/docs/discipline.md` ROW 237: this exact pattern is
/// `differentiate_reduce`'s `ScalarOp::Add` arm broadcasting a matmul's
/// upstream gradient back across the contracted axis before the product
/// rule multiplies it against the other factor -- on the mnist MLP lane's
/// `grad_w1` node the materialized add alone cost 26.5% of the step
/// (`train_step_lane --features train-step-profile`, node 87), and left
/// the multiply's own downstream reduce (node 90) reading a physically
/// non-broadcast operand, which is why `width_tile_plan`
/// (`proxima-tensor/src/cpu.rs:9940`) declined it (`AxesShape`, the "b"
/// operand not truly row-invariant) and forced the untiled per-element
/// fallback loop.
///
/// Value-preserving and index-preserving: no `NodeId` is renumbered or
/// deleted, so every existing `grad_of`/`gathered_of`/output reference
/// stays valid; the rewritten operand's `IndexMap` is copied verbatim from
/// `narrow`'s own definition, so the value read at every iteration point
/// is IDENTICAL (`x + 0 == x`), never merely close. Idempotent: a second
/// pass over an already-rewritten program finds nothing left to match.
pub(crate) fn eliminate_broadcast_anchor_adds(program: &mut [Op]) {
    for index in 0..program.len() {
        let Op::Elementwise {
            body: ScalarOp::Multiply,
            operands,
            ..
        } = &program[index]
        else {
            continue;
        };
        let Some(rank) = operands.first().map(|(_, view)| view.affine().iter_rank) else {
            continue;
        };
        let rewritten: Vec<Option<(NodeId, IndexMap)>> = operands
            .iter()
            .map(|(operand_id, operand_view)| {
                resolve_anchor_bypass(program, *operand_id, operand_view)
            })
            .collect();
        if rewritten.iter().all(Option::is_none) {
            continue;
        }
        // reduce-body `Multiply` (`d(prod x)/dx_i`) pairs TWO anchor-adds
        // together (`numerator`'s `output_broadcast * gradient_broadcast`,
        // adjoint.rs's `ScalarOp::Multiply` reduce arm) -- both narrow
        // sides can share the SAME missing axis, so dropping both anchors
        // would leave that axis unconstrained (`shape::infer`'s own
        // `UnconstrainedDim`, caught by `reduce_multiply_gradient_matches_*`
        // regression tests). Union every operand's post-rewrite covered
        // axes and only commit when the whole rank is still covered --
        // exactly the same guarantee the ORIGINAL anchor-bearing operands
        // provided, so this can never turn a previously-resolvable program
        // unresolvable.
        let covered = operands.iter().zip(&rewritten).fold(
            0u64,
            |mask, ((_, original_view), replacement)| {
                let view = replacement.as_ref().map_or(original_view, |(_, view)| view);
                mask | covered_axes(view)
            },
        );
        if covered != full_axis_mask(rank) {
            continue;
        }
        let Op::Elementwise { operands, .. } = &mut program[index] else {
            unreachable!("the match above already proved this index is Elementwise");
        };
        for (slot, replacement) in operands.iter_mut().zip(rewritten) {
            if let Some(narrow) = replacement {
                *slot = narrow;
            }
        }
    }
}

/// Bitmask of axes `view` supplies a real (non-empty) term for.
/// `full_axis_mask`-compatible: axis `i` set means `view`'s `IndexPattern`
/// constrains dimension `i`, the same "real term" test
/// [`is_pure_projection`] applies per axis.
fn covered_axes(view: &IndexMap) -> u64 {
    view.affine()
        .axes
        .iter()
        .enumerate()
        .filter(|(_, axis)| !axis.terms.is_empty())
        .fold(0u64, |mask, (index, _)| mask | (1u64 << index))
}

/// Every axis `0..rank` set -- `rank` never exceeds 64 in this crate's own
/// tensors (`u16` extents, `iter_rank: u16`, and no program in this
/// workspace approaches even a fraction of that), so the bitmask never
/// truncates.
fn full_axis_mask(rank: u16) -> u64 {
    if rank >= 64 {
        u64::MAX
    } else {
        (1u64 << rank) - 1
    }
}

/// `Some((narrow_id, narrow_view))` when `operand_id` names an
/// `Add(narrow, zero_anchor)` node (either operand order) whose zero side
/// is a same-rank identity-mapped [`Op::Constant`] at `value: 0.0`, AND
/// `operand_view` (the caller's own view INTO that add) is the plain
/// identity map over the add's iteration rank -- the one shape this
/// rewrite proves safe without composing index maps. `None` for every
/// other shape, including a permuted or further-broadcast view into the
/// add, which this pass leaves untouched rather than guess a composition.
fn resolve_anchor_bypass(
    program: &[Op],
    operand_id: NodeId,
    operand_view: &IndexMap,
) -> Option<(NodeId, IndexMap)> {
    let Op::Elementwise {
        body: ScalarOp::Add,
        operands: add_operands,
        ..
    } = program.get(operand_id.0 as usize)?
    else {
        return None;
    };
    let [(first_id, first_view), (second_id, second_view)] = add_operands.as_slice() else {
        return None;
    };
    let add_rank = first_view.affine().iter_rank;
    if *operand_view != identity(add_rank) {
        return None;
    }
    if is_zero_anchor(program, *second_id, second_view, add_rank) {
        return Some((*first_id, first_view.clone()));
    }
    if is_zero_anchor(program, *first_id, first_view, add_rank) {
        return Some((*second_id, second_view.clone()));
    }
    None
}

/// Whether `candidate` is a same-rank identity-mapped zero constant --
/// [`broadcast_anchor`]'s own output node, read back through the exact
/// `full` view every one of its call sites pairs it with.
fn is_zero_anchor(program: &[Op], candidate: NodeId, view: &IndexMap, rank: u16) -> bool {
    *view == identity(rank)
        && matches!(
            program.get(candidate.0 as usize),
            Some(Op::Constant { value, .. }) if *value == 0.0
        )
}
