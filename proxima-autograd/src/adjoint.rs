//! The adjoint transform: forward program plus a scalar loss node in,
//! adjoint program out. [`differentiate`] is the direct-call surface — a
//! plain `&[Op] -> Result<Differentiated, AutogradError>` function, no
//! `.await` needed at a straight-line call site. [`Differentiate`] is the
//! same core wearing this workspace's uniform algebra shape: a
//! [`proxima_primitives::pipe::Pipe`] whose `call` delegates straight to
//! `differentiate`, exactly the relationship
//! `proxima_tensor::shape::ShapeTable`'s own `Pipe` impl
//! (`proxima-tensor/src/shape.rs:292-304`) has to `shape::infer`'s
//! free-function loop over it — the incremental/whole-program primitive
//! wears `Pipe`, the batch convenience is a plain function. Both surfaces
//! exist for the same reason `proxima_tensor::spec::ProgramSpec` exposes a
//! builder AND a plain struct: pick whichever fits the call site, they are
//! the same value.
//!
//! One rule per [`Op`] form:
//!
//! - **`Elementwise` is elementwise.** Each operand gets the local partial
//!   derivative of `body` times the incoming gradient, still at the node's
//!   own iteration space, then routed back into that operand's native
//!   shape through its own [`IndexMap`] — see [`route_contribution`].
//! - **`Reduce(Add)` broadcasts** the incoming gradient back across every
//!   iteration point.
//! - **`Reduce(Maximum)`/`Reduce(Minimum)` mask-route** the incoming
//!   gradient to the argmax/argmin position only — the reduce's own
//!   already-computed output is broadcast back and compared against the
//!   operand with `Equal`, then multiplied in: `ScalarOp::Equal` +
//!   `ScalarOp::Multiply` + `Reduce(Add)`, the same three-primitive shape
//!   `proxima-tensor/src/cpu.rs:16062`'s
//!   `scatter_add_into_a_known_destination_via_mask_composition` test
//!   proves for scatter-add. Ties route the full gradient to every tied
//!   position (matches TensorFlow's `reduce_max` gradient; PyTorch instead
//!   picks one argmax kernel-side — either is a defensible convention,
//!   this crate documents the one it picked).
//! - **`Keep::Scan`** has a materially different derivation (a reversed
//!   prefix-sum for `Add`; no known closed form here for `Maximum`), and is
//!   rejected with [`AutogradError::ScanAdjointUnsupported`] rather than
//!   silently mishandled.
//! - **A gathered (`IndexMap::Computed`) operand's adjoint needs no scatter
//!   at all to *compute*.** `differentiate_elementwise`'s per-operand
//!   contribution is already produced at the *consuming* iteration space —
//!   `table[ids[s], d]`'s own `(s, d)` shape — which is exactly the compact
//!   `[len(ids), row_len]` gradient a caller needs, not the operand's full
//!   `[vocab, row_len]` shape. [`route_contribution`] hands that
//!   contribution back as a [`GatheredContribution`] (paired with the same
//!   `indices` node the forward gather read from) instead of forcing it
//!   through `proxima-tensor/src/cpu.rs:16062`'s dense mask-composition
//!   scatter-add, which is `O(destination x source)` — at embedding scale
//!   (vocab 128k, 4k touched rows) that is 524M mask elements to place 4k
//!   rows. *Applying* a [`GatheredContribution`] back onto its full operand
//!   — the step this crate stops short of, the same way it already stops
//!   short of writing a trained parameter buffer back to disk — is
//!   `O(len(ids) x row_len)`, not `O(vocab x len(ids))`, via
//!   [`crate::sparse::dedupe_and_sum_rows`] plus the existing
//!   [`crate::optimizer::adam_step`] run at a rank sized to the *unique*
//!   touched rows, not the full vocab. See this crate's own report for the
//!   worked example and the element count at realistic dims.
//! - **`Reduce` directly over a gathered operand** (`Reduce::in_map` itself
//!   data-dependent — summing gathered rows before this crate's own
//!   `Reduce` adjoint reuses that same `in_map` as the backward node's
//!   `out_map`) is a materially different derivation from the elementwise
//!   case above: it would need shape.rs's data-dependent-`out_map` gate
//!   (`proxima-tensor/src/shape.rs:166-171`) taught a new lowering, not just
//!   a compact host-side buffer, so it stays out of scope and is rejected
//!   with [`AutogradError::ReduceOverGatherUnsupported`] rather than
//!   silently building a backward program shape.rs would only reject later,
//!   at evaluation time, with no adjoint-specific diagnosis.
//! - **A non-pure-projection operand map** (a convolution-style window,
//!   multi-term or non-unit-coefficient) cannot be reused as a backward
//!   `Reduce`'s `out_map` — `proxima-tensor/src/shape.rs:437-453` rejects
//!   any such `out_map` — so it is rejected here too, named precisely
//!   rather than attempted.

use alloc::vec;
use alloc::vec::Vec;
use core::future::Future;

use proxima_primitives::pipe::Pipe;
use proxima_tensor::dtype::DType;
use proxima_tensor::map::IndexMap;
use proxima_tensor::op::{Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp};
use proxima_tensor::shape::{self, Shapes};

use crate::error::AutogradError;
use crate::expr;

/// One gathered operand's adjoint contribution, before anything scatters it
/// back onto the operand's full shape.
///
/// `values` is already the correct, compact gradient — computing it needed
/// no scatter at all, since [`route_contribution`] hands back the
/// contribution exactly as produced at the *consuming* iteration space
/// (`table[ids[s], d]`'s own `(s, d)` shape), not the operand's own
/// `[vocab, row_len]` shape. `indices` is the same [`NodeId`] the forward
/// gather's [`IndexMap::Computed::indices`] read its row selector from, so
/// `values`'s row `n` belongs at the operand row named by `indices`'s
/// element `n`. `gathered_dim` is the operand axis the forward gather
/// selected (`IndexMap::Computed::gathered_dim`) — [`crate::sparse`]'s
/// helpers assume this is `0` (the operand's leading axis, the canonical
/// embedding-table layout); a caller with a non-leading gathered axis must
/// permute before applying.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatheredContribution {
    pub values: NodeId,
    pub indices: NodeId,
    pub gathered_dim: u16,
}

/// The forward program's own nodes (unchanged, same indices) plus every
/// adjoint node this transform appended, the gradient node for each
/// [`Op::Input`] the loss actually depends on densely, and every gathered
/// operand's compact [`GatheredContribution`].
///
/// Bundling `program` and `gradients` in one value — rather than returning
/// a bare `(Vec<Op>, Vec<(NodeId, NodeId)>)` tuple — closes a real hazard:
/// without it, a caller could pass a `gradients` list computed against one
/// program alongside an unrelated, later-edited `program` value, and
/// `NodeId` is a plain index, so that mismatch would not be a compile
/// error. Moving them together removes that failure mode; it is not
/// grouping for its own sake — see this crate's own report for the "what
/// can a caller do that they could not before" check applied here.
pub struct Differentiated {
    pub program: Vec<Op>,
    pub loss: NodeId,
    gradients: Vec<(NodeId, NodeId)>,
    gathered: Vec<(NodeId, GatheredContribution)>,
}

impl Differentiated {
    fn input_named(&self, name: &str) -> Option<NodeId> {
        self.program.iter().enumerate().find_map(|(index, op)| match op {
            Op::Input { name: Some(candidate), .. } if candidate == name => Some(NodeId(index as u32)),
            _ => None,
        })
    }

    /// The dense gradient node for `node`, if the loss depends on it
    /// through at least one non-gathered (`IndexMap::Affine`) read. A node
    /// read *only* through a gather has no entry here — see
    /// [`Self::gathered_gradients_of`] instead.
    #[must_use]
    pub fn gradient_of(&self, node: NodeId) -> Option<NodeId> {
        self.gradients
            .iter()
            .find(|(candidate, _)| *candidate == node)
            .map(|(_, gradient)| *gradient)
    }

    /// The gradient node for the [`Op::Input`] named `name` — the
    /// gradient-to-parameter binding this crate's scope asks for.
    /// `Op::Input::name` is already how weights load
    /// (`proxima-tensor/src/op.rs:181`) and how
    /// [`proxima_tensor::cpu::evaluate_named`] binds values back in; this
    /// is a lookup over that same name, not a second tree structure.
    #[must_use]
    pub fn gradient_of_named(&self, name: &str) -> Option<NodeId> {
        self.input_named(name).and_then(|node| self.gradient_of(node))
    }

    /// Every [`GatheredContribution`] recorded for `node` — one per forward
    /// site that gathered `node` as an `IndexMap::Computed` operand. Most
    /// programs gather a given table exactly once, so this is usually a
    /// single-element iterator, but nothing here assumes that: a table read
    /// by two separate gathers yields two contributions, both legitimate,
    /// neither one silently dropped.
    pub fn gathered_gradients_of(&self, node: NodeId) -> impl Iterator<Item = GatheredContribution> + '_ {
        self.gathered
            .iter()
            .filter(move |(candidate, _)| *candidate == node)
            .map(|(_, contribution)| *contribution)
    }

    /// [`Self::gathered_gradients_of`], looked up by the operand's
    /// [`Op::Input::name`] instead of its [`NodeId`] — the same convenience
    /// [`Self::gradient_of_named`] gives the dense case.
    pub fn gathered_gradients_of_named(&self, name: &str) -> impl Iterator<Item = GatheredContribution> + '_ {
        let node = self.input_named(name);
        self.gathered
            .iter()
            .filter(move |(candidate, _)| Some(*candidate) == node)
            .map(|(_, contribution)| *contribution)
    }
}

/// See this module's own doc for the rule per [`Op`] form.
pub fn differentiate(program: &[Op], loss: NodeId) -> Result<Differentiated, AutogradError> {
    let loss_index = loss.0 as usize;
    if loss_index >= program.len() {
        return Err(AutogradError::UnknownLoss(loss));
    }

    let shapes = shape::infer(program, &[])?;
    let loss_rank = shapes.of(loss).len();
    if loss_rank != 0 {
        return Err(AutogradError::LossNotScalar { node: loss, rank: loss_rank });
    }

    let mut new_program: Vec<Op> = program[..=loss_index].to_vec();
    let mut grad_of: Vec<Option<NodeId>> = vec![None; loss_index + 1];
    let mut gathered_of: Vec<Vec<GatheredContribution>> = vec![Vec::new(); loss_index + 1];
    grad_of[loss_index] = Some(expr::constant(&mut new_program, DType::Float32, 1.0));

    for index in (0..=loss_index).rev() {
        let Some(gradient) = grad_of[index] else { continue };
        let node = NodeId(index as u32);
        match &program[index] {
            Op::Input { .. } | Op::Iota { .. } | Op::Constant { .. } => {}
            Op::Elementwise { dtype, body, operands, .. } => differentiate_elementwise(
                &mut new_program,
                &mut grad_of,
                &mut gathered_of,
                program,
                &shapes,
                node,
                *dtype,
                *body,
                operands,
                gradient,
            )?,
            Op::Reduce(reduce) => differentiate_reduce(
                &mut new_program,
                &mut grad_of,
                program,
                &shapes,
                node,
                reduce,
                gradient,
            )?,
        }
    }

    let gradients = program
        .iter()
        .enumerate()
        .filter(|(_, op)| matches!(op, Op::Input { .. }))
        .filter_map(|(index, _)| grad_of[index].map(|gradient| (NodeId(index as u32), gradient)))
        .collect();

    let gathered = gathered_of
        .into_iter()
        .enumerate()
        .flat_map(|(index, contributions)| {
            contributions.into_iter().map(move |contribution| (NodeId(index as u32), contribution))
        })
        .collect();

    Ok(Differentiated { program: new_program, loss, gradients, gathered })
}

fn accumulate(
    program: &mut Vec<Op>,
    grad_of: &mut [Option<NodeId>],
    dtype: DType,
    rank: u16,
    node_index: usize,
    contribution: NodeId,
) {
    grad_of[node_index] = Some(match grad_of[node_index] {
        None => contribution,
        Some(existing) => {
            let full = expr::identity(rank);
            expr::binary(program, dtype, ScalarOp::Add, (existing, full.clone()), (contribution, full))
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn differentiate_elementwise(
    program: &mut Vec<Op>,
    grad_of: &mut [Option<NodeId>],
    gathered_of: &mut [Vec<GatheredContribution>],
    original_program: &[Op],
    shapes: &Shapes,
    node: NodeId,
    dtype: DType,
    body: ScalarOp,
    operands: &[(NodeId, IndexMap)],
    gradient: NodeId,
) -> Result<(), AutogradError> {
    let iter_rank = shapes.of(node).len() as u16;
    let full = expr::identity(iter_rank);
    let broadcast = expr::broadcast(iter_rank);

    let contributions: Vec<Option<NodeId>> = match body {
        ScalarOp::Identity => vec![Some(gradient)],
        ScalarOp::Negate => {
            vec![Some(expr::unary(program, dtype, ScalarOp::Negate, (gradient, full)))]
        }
        ScalarOp::Add => vec![Some(gradient), Some(gradient)],
        ScalarOp::Subtract => {
            let negated = expr::unary(program, dtype, ScalarOp::Negate, (gradient, full.clone()));
            vec![Some(gradient), Some(negated)]
        }
        ScalarOp::Multiply => {
            let (a, map_a) = operands[0].clone();
            let (b, map_b) = operands[1].clone();
            let grad_a = expr::binary(program, dtype, ScalarOp::Multiply, (gradient, full.clone()), (b, map_b));
            let grad_b = expr::binary(program, dtype, ScalarOp::Multiply, (gradient, full), (a, map_a));
            vec![Some(grad_a), Some(grad_b)]
        }
        ScalarOp::Divide => {
            let (a, map_a) = operands[0].clone();
            let (b, map_b) = operands[1].clone();
            let recip_b = expr::unary(program, dtype, ScalarOp::Reciprocal, (b, map_b.clone()));
            let grad_a = expr::binary(program, dtype, ScalarOp::Multiply, (gradient, full.clone()), (recip_b, full.clone()));
            let b_squared = expr::binary(program, dtype, ScalarOp::Multiply, (b, map_b.clone()), (b, map_b));
            let neg_a = expr::unary(program, dtype, ScalarOp::Negate, (a, map_a));
            let recip_b_squared = expr::unary(program, dtype, ScalarOp::Reciprocal, (b_squared, full.clone()));
            let slope = expr::binary(program, dtype, ScalarOp::Multiply, (neg_a, full.clone()), (recip_b_squared, full.clone()));
            let grad_b = expr::binary(program, dtype, ScalarOp::Multiply, (gradient, full.clone()), (slope, full));
            vec![Some(grad_a), Some(grad_b)]
        }
        ScalarOp::Maximum => maximum_minimum_grads(program, dtype, operands, gradient, &full, &broadcast, true),
        ScalarOp::Minimum => maximum_minimum_grads(program, dtype, operands, gradient, &full, &broadcast, false),
        ScalarOp::Reciprocal => {
            let out_squared = expr::binary(program, dtype, ScalarOp::Multiply, (node, full.clone()), (node, full.clone()));
            let neg_out_squared = expr::unary(program, dtype, ScalarOp::Negate, (out_squared, full.clone()));
            vec![Some(expr::binary(program, dtype, ScalarOp::Multiply, (gradient, full.clone()), (neg_out_squared, full)))]
        }
        ScalarOp::Exponential => {
            vec![Some(expr::binary(program, dtype, ScalarOp::Multiply, (gradient, full.clone()), (node, full)))]
        }
        ScalarOp::Logarithm => {
            let (x, map_x) = operands[0].clone();
            let recip = expr::unary(program, dtype, ScalarOp::Reciprocal, (x, map_x));
            vec![Some(expr::binary(program, dtype, ScalarOp::Multiply, (gradient, full.clone()), (recip, full)))]
        }
        ScalarOp::SquareRoot => {
            let two = expr::constant(program, dtype, 2.0);
            let denominator = expr::binary(program, dtype, ScalarOp::Multiply, (two, broadcast), (node, full.clone()));
            let recip = expr::unary(program, dtype, ScalarOp::Reciprocal, (denominator, full.clone()));
            vec![Some(expr::binary(program, dtype, ScalarOp::Multiply, (gradient, full.clone()), (recip, full)))]
        }
        ScalarOp::Tanh => {
            let squared = expr::binary(program, dtype, ScalarOp::Multiply, (node, full.clone()), (node, full.clone()));
            let one = expr::constant(program, dtype, 1.0);
            let slope = expr::binary(program, dtype, ScalarOp::Subtract, (one, broadcast), (squared, full.clone()));
            vec![Some(expr::binary(program, dtype, ScalarOp::Multiply, (gradient, full.clone()), (slope, full)))]
        }
        ScalarOp::Erf => {
            let (x, map_x) = operands[0].clone();
            let coefficient = expr::constant(program, dtype, 2.0 / libm::sqrtf(core::f32::consts::PI));
            let x_squared = expr::binary(program, dtype, ScalarOp::Multiply, (x, map_x.clone()), (x, map_x));
            let neg_x_squared = expr::unary(program, dtype, ScalarOp::Negate, (x_squared, full.clone()));
            let exponentiated = expr::unary(program, dtype, ScalarOp::Exponential, (neg_x_squared, full.clone()));
            let slope = expr::binary(program, dtype, ScalarOp::Multiply, (coefficient, broadcast), (exponentiated, full.clone()));
            vec![Some(expr::binary(program, dtype, ScalarOp::Multiply, (gradient, full.clone()), (slope, full)))]
        }
        ScalarOp::Greater | ScalarOp::Equal => operands.iter().map(|_| None).collect(),
        ScalarOp::Select => {
            let (condition, condition_map) = operands[0].clone();
            let true_mask = expr::binary(
                program,
                dtype,
                ScalarOp::Multiply,
                (gradient, full.clone()),
                (condition, condition_map.clone()),
            );
            let one = expr::constant(program, dtype, 1.0);
            let inverse_condition =
                expr::binary(program, dtype, ScalarOp::Subtract, (one, broadcast), (condition, condition_map));
            let false_mask =
                expr::binary(program, dtype, ScalarOp::Multiply, (gradient, full.clone()), (inverse_condition, full));
            vec![None, Some(true_mask), Some(false_mask)]
        }
    };

    for (operand, contribution) in operands.iter().zip(contributions) {
        let Some(contribution) = contribution else { continue };
        route_contribution(program, grad_of, gathered_of, original_program, shapes, node, operand, contribution, iter_rank)?;
    }
    Ok(())
}

/// `Maximum`/`Minimum` route the incoming gradient entirely to the operand
/// that produced the result; ties favor the first operand (`a`) for both —
/// see this module's own doc for the convention.
#[allow(clippy::too_many_arguments)]
fn maximum_minimum_grads(
    program: &mut Vec<Op>,
    dtype: DType,
    operands: &[(NodeId, IndexMap)],
    gradient: NodeId,
    full: &IndexMap,
    broadcast: &IndexMap,
    is_maximum: bool,
) -> Vec<Option<NodeId>> {
    let (a, map_a) = operands[0].clone();
    let (b, map_b) = operands[1].clone();
    let one = expr::constant(program, dtype, 1.0);

    let second_operand_wins = if is_maximum {
        expr::binary(program, dtype, ScalarOp::Greater, (b, map_b), (a, map_a))
    } else {
        expr::binary(program, dtype, ScalarOp::Greater, (a, map_a), (b, map_b))
    };
    let first_operand_wins = expr::binary(
        program,
        dtype,
        ScalarOp::Subtract,
        (one, broadcast.clone()),
        (second_operand_wins, full.clone()),
    );

    let grad_a = expr::binary(program, dtype, ScalarOp::Multiply, (gradient, full.clone()), (first_operand_wins, full.clone()));
    let grad_b = expr::binary(program, dtype, ScalarOp::Multiply, (gradient, full.clone()), (second_operand_wins, full.clone()));
    vec![Some(grad_a), Some(grad_b)]
}

/// Routes one operand's local contribution back into `grad_of` (an
/// `IndexMap::Affine` operand) or `gathered_of` (an `IndexMap::Computed`
/// operand) — see this module's own doc for why the two cases differ.
#[allow(clippy::too_many_arguments)]
fn route_contribution(
    program: &mut Vec<Op>,
    grad_of: &mut [Option<NodeId>],
    gathered_of: &mut [Vec<GatheredContribution>],
    original_program: &[Op],
    shapes: &Shapes,
    consumer: NodeId,
    operand: &(NodeId, IndexMap),
    contribution: NodeId,
    iter_rank: u16,
) -> Result<(), AutogradError> {
    let (operand_node, operand_map) = operand;
    if matches!(original_program[operand_node.0 as usize], Op::Constant { .. } | Op::Iota { .. }) {
        return Ok(());
    }

    let dtype = original_program[operand_node.0 as usize].dtype();

    match operand_map {
        IndexMap::Affine(pattern) => {
            if !expr::is_pure_projection(pattern) {
                return Err(AutogradError::NonProjectionOperandMap { node: consumer, operand: *operand_node });
            }
            let operand_rank = shapes.of(*operand_node).len() as u16;
            let routed = expr::reduce(
                program,
                dtype,
                ScalarOp::Add,
                ReduceInit::Zero,
                contribution,
                expr::identity(iter_rank),
                IndexMap::Affine(pattern.clone()),
            );
            accumulate(program, grad_of, dtype, operand_rank, operand_node.0 as usize, routed);
        }
        IndexMap::Computed { indices, index_map, gathered_dim, .. } => {
            if !expr::is_pure_projection(index_map) {
                return Err(AutogradError::NonProjectionIndexMap { node: consumer, operand: *operand_node });
            }
            gathered_of[operand_node.0 as usize].push(GatheredContribution {
                values: contribution,
                indices: *indices,
                gathered_dim: *gathered_dim,
            });
        }
    }
    Ok(())
}

fn differentiate_reduce(
    program: &mut Vec<Op>,
    grad_of: &mut [Option<NodeId>],
    original_program: &[Op],
    shapes: &Shapes,
    node: NodeId,
    reduce: &Reduce,
    gradient: NodeId,
) -> Result<(), AutogradError> {
    if matches!(reduce.keep, Keep::Scan) {
        return Err(AutogradError::ScanAdjointUnsupported { node });
    }
    if reduce.out_map.is_data_dependent() {
        return Err(AutogradError::ScatterOutputUnsupported { node });
    }
    if reduce.in_map.is_data_dependent() {
        return Err(AutogradError::ReduceOverGatherUnsupported { node, operand: reduce.operand });
    }
    let in_pattern = reduce.in_map.affine();
    if !expr::is_pure_projection(in_pattern) {
        return Err(AutogradError::NonProjectionOperandMap { node, operand: reduce.operand });
    }
    let full = expr::identity(in_pattern.iter_rank);
    let out_map_as_operand = IndexMap::Affine(reduce.out_map.affine().clone());

    let anchor_extents = expr::iter_extents(shapes, reduce.operand, in_pattern);
    let anchor = expr::broadcast_anchor(program, reduce.dtype, &anchor_extents);

    let contribution = match reduce.body {
        ScalarOp::Add => expr::binary(
            program,
            reduce.dtype,
            ScalarOp::Add,
            (gradient, out_map_as_operand),
            (anchor, full.clone()),
        ),
        ScalarOp::Maximum | ScalarOp::Minimum => {
            let out_broadcast = expr::binary(
                program,
                reduce.dtype,
                ScalarOp::Add,
                (node, out_map_as_operand.clone()),
                (anchor, full.clone()),
            );
            let mask = expr::binary(
                program,
                reduce.dtype,
                ScalarOp::Equal,
                (reduce.operand, reduce.in_map.clone()),
                (out_broadcast, full.clone()),
            );
            let gradient_broadcast = expr::binary(
                program,
                reduce.dtype,
                ScalarOp::Add,
                (gradient, out_map_as_operand),
                (anchor, full.clone()),
            );
            expr::binary(program, reduce.dtype, ScalarOp::Multiply, (mask, full.clone()), (gradient_broadcast, full.clone()))
        }
        other => return Err(AutogradError::UnsupportedReduceBody { node, body: other }),
    };

    let operand_dtype = original_program[reduce.operand.0 as usize].dtype();
    let routed = expr::reduce(
        program,
        operand_dtype,
        ScalarOp::Add,
        ReduceInit::Zero,
        contribution,
        full,
        reduce.in_map.clone(),
    );
    let operand_rank = shapes.of(reduce.operand).len() as u16;
    accumulate(program, grad_of, operand_dtype, operand_rank, reduce.operand.0 as usize, routed);
    Ok(())
}

/// [`differentiate`] wearing this workspace's uniform algebra shape — see
/// this module's own doc for the exact relationship to
/// `proxima_tensor::shape::ShapeTable`'s own `Pipe` impl. Zero-sized: unlike
/// [`crate::optimizer::AdamStep`] (which holds an `AdamConfig` and a
/// program under construction across many calls), `differentiate` runs
/// once and needs no persistent state between calls, so there is nothing
/// to hold in `&self`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Differentiate;

impl Pipe for Differentiate {
    /// Owned rather than `&[Op]`: `Pipe::call` takes `Self::In` by value
    /// with no lifetime tying it to `&self`, so a borrowed program would
    /// need a lifetime parameter on the impl itself for no benefit here —
    /// this crate's own `differentiate` free function still only borrows
    /// internally.
    type In = (Vec<Op>, NodeId);
    type Out = Differentiated;
    type Err = AutogradError;

    fn call(
        &self,
        (program, loss): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> {
        async move { differentiate(&program, loss) }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod pipe_tests {
    use proxima_tensor::op::Extent;

    use super::*;

    /// `Pipe::call`'s returned future has no `Send` bound (that is the
    /// entire point of the base rung vs `SendPipe`) — `#[proxima::test]`'s
    /// harness requires `Send`, so a plain `#[test]` polling this
    /// single-state-machine future once (it never returns `Pending`, there
    /// is no `.await` inside the `async move` body) is the correct tool
    /// here, not a workaround.
    fn block_on_once<F: Future>(future: F) -> F::Output {
        let mut future = core::pin::pin!(future);
        let mut context = core::task::Context::from_waker(core::task::Waker::noop());
        match future.as_mut().poll(&mut context) {
            core::task::Poll::Ready(output) => output,
            core::task::Poll::Pending => panic!("test future must be ready on first poll"),
        }
    }

    #[test]
    fn the_pipe_form_agrees_with_the_free_function() {
        let mut program = Vec::new();
        let x = proxima_tensor::op::append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(3)],
                name: Some("x".into()),
            },
        );
        let loss = proxima_tensor::op::append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: x,
                in_map: expr::identity(1),
                out_map: IndexMap::Affine(proxima_tensor::map::projection(1, &[])),
                keep: Keep::Reduce,
                name: None,
            }),
        );

        let via_pipe = block_on_once(Differentiate.call((program.clone(), loss)))
            .expect("pipe form differentiates");
        let via_function = differentiate(&program, loss).expect("free function differentiates");

        assert_eq!(
            via_pipe.gradient_of_named("x"),
            via_function.gradient_of_named("x"),
            "the Pipe wrapper must delegate to the exact same transform"
        );
    }
}
