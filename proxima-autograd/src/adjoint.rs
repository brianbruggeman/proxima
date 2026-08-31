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
//!   shape through its own [`IndexMap`] — see `route_contribution` (private).
//! - **`Reduce(Add)` broadcasts** the incoming gradient back across every
//!   iteration point.
//! - **`Reduce(Multiply)` divides.** `d(prod x)/dx_i = (prod x)/x_i`, so the
//!   contribution at each position is `gradient * output / x_i` -- the
//!   reduce's own already-computed output broadcast back the same way
//!   `Add`'s rule broadcasts `gradient`, then divided by that position's own
//!   input. Undefined (produces `inf`/`NaN`) where `x_i` is exactly zero;
//!   see the private `differentiate_reduce`'s own comment on that arm.
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
//!   `[vocab, row_len]` shape. `route_contribution` (private) hands that
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

use alloc::collections::BTreeSet;
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
/// no scatter at all, since `route_contribution` (private) hands back the
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
    differentiate_core(program, loss, None)
}

/// [`differentiate`], scoped to the [`Op::Input`] nodes named in `wanted`.
///
/// Every gradient `differentiate` would compute is still computed here — the
/// backward walk still visits every reachable node, and every wanted input's
/// gradient is bit-identical to what `differentiate` would produce for it
/// (same nodes, same order, same values). The only thing this function omits
/// is the *final* routing step into an [`Op::Input`] that is not in `wanted`:
/// the small `expr::reduce`/`accumulate` call that would otherwise
/// materialize that input's own gradient buffer and then have it go unread.
/// A shared intermediate node feeding both a wanted and an unwanted input
/// (e.g. a matmul's product feeding both a weight's gradient and the layer's
/// data gradient) is untouched — inputs are always leaves in this reverse
/// walk (`Op::Input`'s own match arm never recurses further), so gating only
/// at the leaf-routing sites (`route_contribution`'s two arms and
/// `differentiate_reduce`'s own direct accumulate) is exactly "skip the
/// final routing into unwanted inputs and nothing upstream of it" — no
/// separate liveness pass is needed because inputs never have downstream
/// adjoint work to prune in the first place.
///
/// See `proxima-tensor/docs/discipline.md` ROW 161/162: `differentiate`
/// unconditionally backprops through every reachable [`Op::Input`], including
/// training data the caller's own `rebind` list never reads back
/// (`train::train_step`'s own `grad_x` case) — this is the scoped surface
/// `train_step`/`fit` reach for instead, requesting gradients only for the
/// parameter+state inputs they actually rebind.
///
/// # Errors
///
/// Same as [`differentiate`].
#[must_use = "Result must be checked"]
pub fn differentiate_wanted(program: &[Op], loss: NodeId, wanted: &[NodeId]) -> Result<Differentiated, AutogradError> {
    let wanted: BTreeSet<NodeId> = wanted.iter().copied().collect();
    differentiate_core(program, loss, Some(&wanted))
}

fn differentiate_core(program: &[Op], loss: NodeId, wanted: Option<&BTreeSet<NodeId>>) -> Result<Differentiated, AutogradError> {
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
                wanted,
            )?,
            Op::Reduce(reduce) => differentiate_reduce(
                &mut new_program,
                &mut grad_of,
                program,
                &shapes,
                node,
                reduce,
                gradient,
                wanted,
            )?,
        }
    }

    // `program[..=loss_index]`, not the full `program` slice: `grad_of` is
    // sized `loss_index + 1` on purpose (this crate's own differentiate
    // never needs a node past the loss to compute anything), and a caller
    // is free to hand `differentiate` a `program` that keeps growing PAST
    // `loss` -- e.g. a second, later loss node on the same forward graph
    // (`proxima-autograd/tests/actor_critic.rs`'s two-loss-node actor-critic
    // program). Indexing `grad_of` by every `Op::Input` in the FULL
    // `program` would walk off the end of `grad_of` the moment any such
    // later node existed.
    let gradients = program[..=loss_index]
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
    wanted: Option<&BTreeSet<NodeId>>,
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
            // `condition_map` may broadcast over an axis (a causal mask
            // shared across every attention head, say) that this Select's
            // OTHER operands do not, so anchoring `one` at a bare rank-0
            // broadcast can leave that axis unconstrained even though the
            // forward `Select` itself lowered fine (`scaled`'s full-rank
            // operand covered it there) -- see `broadcast_anchor`'s own doc.
            let one = expr::broadcast_anchor(program, dtype, shapes.of(node), 1.0);
            let inverse_condition =
                expr::binary(program, dtype, ScalarOp::Subtract, (one, full.clone()), (condition, condition_map));
            let false_mask =
                expr::binary(program, dtype, ScalarOp::Multiply, (gradient, full.clone()), (inverse_condition, full));
            vec![None, Some(true_mask), Some(false_mask)]
        }
    };

    for (operand, contribution) in operands.iter().zip(contributions) {
        let Some(contribution) = contribution else { continue };
        route_contribution(program, grad_of, gathered_of, original_program, shapes, node, operand, contribution, iter_rank, wanted)?;
    }
    Ok(())
}

/// `true` when `node` is an [`Op::Input`] `differentiate_wanted` was NOT
/// asked for — the only case a leaf-routing site may skip. `wanted == None`
/// (plain [`differentiate`]) never skips anything, matching its documented
/// "backprop through every reachable `Op::Input`" contract.
fn is_unwanted_input(original_program: &[Op], wanted: Option<&BTreeSet<NodeId>>, node: NodeId) -> bool {
    let Some(wanted) = wanted else { return false };
    matches!(original_program[node.0 as usize], Op::Input { .. }) && !wanted.contains(&node)
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
    wanted: Option<&BTreeSet<NodeId>>,
) -> Result<(), AutogradError> {
    let (operand_node, operand_map) = operand;
    if matches!(original_program[operand_node.0 as usize], Op::Constant { .. } | Op::Iota { .. }) {
        return Ok(());
    }
    if is_unwanted_input(original_program, wanted, *operand_node) {
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

#[allow(clippy::too_many_arguments)]
fn differentiate_reduce(
    program: &mut Vec<Op>,
    grad_of: &mut [Option<NodeId>],
    original_program: &[Op],
    shapes: &Shapes,
    node: NodeId,
    reduce: &Reduce,
    gradient: NodeId,
    wanted: Option<&BTreeSet<NodeId>>,
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
    if is_unwanted_input(original_program, wanted, reduce.operand) {
        return Ok(());
    }
    let in_pattern = reduce.in_map.affine();
    if !expr::is_pure_projection(in_pattern) {
        return Err(AutogradError::NonProjectionOperandMap { node, operand: reduce.operand });
    }
    let full = expr::identity(in_pattern.iter_rank);
    let out_map_as_operand = IndexMap::Affine(reduce.out_map.affine().clone());

    let anchor_extents = expr::iter_extents(shapes, reduce.operand, in_pattern);
    let anchor = expr::broadcast_anchor(program, reduce.dtype, &anchor_extents, 0.0);

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
        // The standard divide-form product-reduction adjoint:
        // `d(prod x)/dx_i = (prod x) / x_i`, so `grad_i = gradient *
        // output / x_i`. `output` (this `Reduce` node's own already-computed
        // result) is broadcast back across the iteration space the same way
        // `Add`/`Maximum`/`Minimum` above broadcast `gradient` -- reusing
        // the forward pass's own product instead of recomputing "product of
        // every OTHER element" per position, which would need an extra
        // reduction this crate's algebra has no cheaper way to express.
        // Caveat: this divides by each input element, so it is undefined
        // (produces `inf`/`NaN`, not silently wrong) at any position where
        // that element is exactly zero -- the same caveat every
        // divide-form product-rule adjoint carries, documented here rather
        // than guarded, since guarding would silently change the value at
        // a legitimate zero input instead of surfacing the singularity.
        ScalarOp::Multiply => {
            let output_broadcast = expr::binary(
                program,
                reduce.dtype,
                ScalarOp::Add,
                (node, out_map_as_operand.clone()),
                (anchor, full.clone()),
            );
            let gradient_broadcast = expr::binary(
                program,
                reduce.dtype,
                ScalarOp::Add,
                (gradient, out_map_as_operand),
                (anchor, full.clone()),
            );
            let numerator =
                expr::binary(program, reduce.dtype, ScalarOp::Multiply, (gradient_broadcast, full.clone()), (output_broadcast, full.clone()));
            let recip_operand = expr::unary(program, reduce.dtype, ScalarOp::Reciprocal, (reduce.operand, reduce.in_map.clone()));
            expr::binary(program, reduce.dtype, ScalarOp::Multiply, (numerator, full.clone()), (recip_operand, full.clone()))
        }
        other => return Err(AutogradError::UnsupportedReduceBody { node, body: other }),
    };

    let operand_dtype = original_program[reduce.operand.0 as usize].dtype();
    // `reduce.in_map` reads every one of `full`'s iteration positions
    // one-to-one (same rank, no axis dropped, no scaling) exactly when it
    // equals `full` itself -- no position ever accumulates with another, so
    // wrapping `contribution` in an `expr::reduce` here would materialize a
    // full [iter_rank] buffer to compute a pure copy. Skipping the wrapper
    // makes `contribution` (an `Op::Elementwise`) `grad_of[operand]`
    // directly, which `bind.rs`'s existing `held`/fusion path (only
    // `Elementwise` nodes are ever held, `Reduce` nodes always retire
    // unconditionally) can then fuse straight into whichever consumer reads
    // it next, instead of a forced intermediate materialize.
    let routed = if reduce.in_map == full {
        contribution
    } else {
        expr::reduce(
            program,
            operand_dtype,
            ScalarOp::Add,
            ReduceInit::Zero,
            contribution,
            full,
            reduce.in_map.clone(),
        )
    };
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod differentiate_wanted_tests {
    use alloc::vec;

    use proxima_tensor::op::Extent;

    use super::*;

    fn leaf(program: &mut Vec<Op>, name: &str, shape: Vec<Extent>) -> NodeId {
        proxima_tensor::op::append(program, Op::Input { dtype: DType::Float32, shape, name: Some(name.into()) })
    }

    fn identity(rank: u16) -> IndexMap {
        IndexMap::Affine(proxima_tensor::map::projection(rank, &(0..rank).collect::<Vec<u16>>()))
    }

    fn axes(rank: u16, selected: &[u16]) -> IndexMap {
        IndexMap::Affine(proxima_tensor::map::projection(rank, selected))
    }

    /// `matmul[i,k] = sum_j x[i,j] * w[j,k]`, `loss = sum(matmul)` -- the
    /// same `Elementwise(Multiply)` -> `Reduce(Add)` shape
    /// `train_step_lane.rs`'s `batched_dense` builds for each MLP layer, so
    /// the same `differentiate_reduce`/`differentiate_elementwise`/
    /// `route_contribution` path ROW 161/162 characterized is what this
    /// fixture exercises, not a simplified stand-in. `x` and `w` share the
    /// same `product` intermediate node (the un-reduce/broadcast materialize
    /// ROW 161 found expensive), so this is exactly the "shared upstream
    /// node feeding both a wanted and an unwanted input" case the design
    /// constraint calls out.
    fn build_matmul_loss() -> (Vec<Op>, NodeId, NodeId, NodeId) {
        let mut program = Vec::new();
        let x = leaf(&mut program, "x", vec![Extent::Static(2), Extent::Static(3)]);
        let w = leaf(&mut program, "w", vec![Extent::Static(3), Extent::Static(4)]);
        let product = proxima_tensor::op::append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![(w, axes(3, &[1, 2])), (x, axes(3, &[0, 1]))],
                name: None,
            },
        );
        let matmul = expr::reduce(&mut program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, product, identity(3), axes(3, &[0, 2]));
        let loss = expr::reduce(&mut program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, matmul, identity(2), axes(2, &[]));
        (program, x, w, loss)
    }

    /// The load-bearing correctness proof: `differentiate_wanted`'s own
    /// `grad_w` is bit-identical to plain `differentiate`'s `grad_w` on the
    /// exact same fixture — the pruning this function does for `x` (not in
    /// `wanted`) must not perturb `w`'s own gradient by even one ULP, since
    /// both reach `w` through the same shared `product` node's un-reduce.
    #[proxima::test]
    async fn wanted_scoped_gradient_is_bit_identical_to_the_full_differentiate() {
        let (program, _x, w, loss) = build_matmul_loss();
        let x_values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w_values = [0.5f32, -1.0, 2.0, 0.25, 1.5, -0.75, 3.0, 0.1, -2.0, 4.0, 0.0, 1.0];

        let full = differentiate(&program, loss).expect("full differentiate");
        let grad_w_full = full.gradient_of_named("w").expect("w feeds the loss");
        let full_evaluated =
            proxima_tensor::cpu::evaluate_named(&full.program, &[], &[("x", &x_values), ("w", &w_values)], &[grad_w_full])
                .expect("full adjoint program evaluates");
        let full_grad_w = full_evaluated.get(grad_w_full).expect("requested").0;

        let wanted = differentiate_wanted(&program, loss, &[w]).expect("wanted-scoped differentiate");
        let grad_w_wanted = wanted.gradient_of_named("w").expect("w feeds the loss");
        let wanted_evaluated =
            proxima_tensor::cpu::evaluate_named(&wanted.program, &[], &[("x", &x_values), ("w", &w_values)], &[grad_w_wanted])
                .expect("wanted-scoped adjoint program evaluates");
        let wanted_grad_w = wanted_evaluated.get(grad_w_wanted).expect("requested").0;

        assert_eq!(full_grad_w, wanted_grad_w, "wanted-scoped grad_w must be bit-identical to the full differentiate's own grad_w");

        assert!(wanted.gradient_of_named("x").is_none(), "x was not in `wanted`, so its gradient must not be routed at all");
        assert!(full.gradient_of_named("x").is_some(), "sanity: plain differentiate still computes grad_x, unaffected by the wanted-scoped path");
        assert!(
            wanted.program.len() < full.program.len(),
            "the wanted-scoped program must be strictly smaller: it never emits x's own final routing/un-reduce node ({} vs {})",
            wanted.program.len(),
            full.program.len()
        );
    }

    /// A caller-side documented consequence of the pruning, not merely
    /// asserted internally: with `x` unrequested, no plausible
    /// evaluation of the wanted-scoped program can hand back a gradient for
    /// it — `gathered_gradients_of_named`/`gradient_of_named` both agree
    /// there is nothing there to ask for.
    #[proxima::test]
    async fn unwanted_gradient_of_named_is_none_on_the_wanted_scoped_path() {
        let (program, _x, w, loss) = build_matmul_loss();
        let wanted = differentiate_wanted(&program, loss, &[w]).expect("wanted-scoped differentiate");
        assert!(wanted.gradient_of_named("x").is_none());
        assert!(wanted.gradient_of_named("w").is_some());
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod select_broadcast_condition_tests {
    use alloc::vec;

    use proxima_tensor::op::Extent;

    use super::*;

    fn leaf(program: &mut Vec<Op>, name: &str, shape: Vec<Extent>) -> NodeId {
        proxima_tensor::op::append(program, Op::Input { dtype: DType::Float32, shape, name: Some(name.into()) })
    }

    fn elementwise(program: &mut Vec<Op>, body: ScalarOp, operands: Vec<(NodeId, IndexMap)>) -> NodeId {
        proxima_tensor::op::append(program, Op::Elementwise { dtype: DType::Float32, body, operands, name: None })
    }

    fn relative_error(analytic: f32, numeric: f32) -> f32 {
        (analytic - numeric).abs() / (analytic.abs().max(numeric.abs()) + 1e-6)
    }

    /// A `Select` whose condition operand broadcasts over an axis (`h`, a
    /// stand-in for an attention head) that the true/false branches do
    /// NOT broadcast over -- `proxima-autograd/tests/language_model.rs`'s
    /// causal mask hit exactly this shape (`is_future[s,t]` selecting
    /// across every `h`) and failed shape inference with
    /// `TensorError::UnconstrainedDim` before this rule anchored `one` at
    /// the consumer's own full extents (`shapes.of(node)`) instead of a
    /// bare rank-0 broadcast. Central difference against the real forward
    /// program is the oracle: it does not know or care that `condition`
    /// broadcasts, so it is a correct check for this shape regardless.
    #[proxima::test]
    async fn select_with_a_condition_broadcast_over_an_axis_the_branches_do_not_share_gradient_checks() {
        let mut program = Vec::new();
        let a = leaf(&mut program, "a", vec![Extent::Static(2), Extent::Static(3)]);
        let b = leaf(&mut program, "b", vec![Extent::Static(2), Extent::Static(3)]);
        let condition = elementwise(&mut program, ScalarOp::Greater, vec![(a, expr::identity(2)), (b, expr::identity(2))]);
        let true_branch = leaf(&mut program, "true_branch", vec![Extent::Static(2), Extent::Static(3), Extent::Static(2)]);
        let false_branch = leaf(&mut program, "false_branch", vec![Extent::Static(2), Extent::Static(3), Extent::Static(2)]);
        let condition_broadcast_over_h = IndexMap::Affine(proxima_tensor::map::projection(3, &[0, 1]));
        let selected = elementwise(
            &mut program,
            ScalarOp::Select,
            vec![(condition, condition_broadcast_over_h), (true_branch, expr::identity(3)), (false_branch, expr::identity(3))],
        );
        let loss = proxima_tensor::op::append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: selected,
                in_map: expr::identity(3),
                out_map: IndexMap::Affine(proxima_tensor::map::projection(3, &[])),
                keep: Keep::Reduce,
                name: None,
            }),
        );

        let differentiated = differentiate(&program, loss).expect(
            "differentiates without TensorError::UnconstrainedDim -- this is the regression this test guards",
        );
        let grad_true = differentiated.gradient_of_named("true_branch").expect("true_branch feeds the loss");
        let grad_false = differentiated.gradient_of_named("false_branch").expect("false_branch feeds the loss");

        let a_values = [3.0f32, 1.0, 5.0, 0.0, 2.0, 4.0];
        let b_values = [1.0f32, 1.0, 2.0, 2.0, 2.0, 1.0];
        let true_values: alloc::vec::Vec<f32> = (0..12).map(|index| (index as f32 - 5.0) / 2.0).collect();
        let false_values: alloc::vec::Vec<f32> = (0..12).map(|index| (index as f32 * 2.0 - 3.0) / 3.0).collect();

        let evaluated = proxima_tensor::cpu::evaluate_named(
            &differentiated.program,
            &[],
            &[("a", &a_values), ("b", &b_values), ("true_branch", &true_values), ("false_branch", &false_values)],
            &[grad_true, grad_false],
        )
        .expect("adjoint program lowers and evaluates");
        let analytic_true = evaluated.get(grad_true).expect("requested").0.to_vec();
        let analytic_false = evaluated.get(grad_false).expect("requested").0.to_vec();

        let step = 1e-3f32;
        let mut worst = (0.0f32, "", 0usize);
        for (label, values, analytic) in
            [("true_branch", true_values.clone(), &analytic_true), ("false_branch", false_values.clone(), &analytic_false)]
        {
            let mut perturbed = values;
            for index in 0..perturbed.len() {
                let original = perturbed[index];
                let evaluate_loss = |perturbed: &[f32]| {
                    let (true_input, false_input) = if label == "true_branch" {
                        (perturbed, false_values.as_slice())
                    } else {
                        (true_values.as_slice(), perturbed)
                    };
                    proxima_tensor::cpu::evaluate_named(
                        &program,
                        &[],
                        &[("a", &a_values), ("b", &b_values), ("true_branch", true_input), ("false_branch", false_input)],
                        &[loss],
                    )
                    .expect("forward program lowers and evaluates")
                    .get(loss)
                    .expect("loss requested")
                    .0[0]
                };
                perturbed[index] = original + step;
                let plus = evaluate_loss(&perturbed);
                perturbed[index] = original - step;
                let minus = evaluate_loss(&perturbed);
                perturbed[index] = original;

                let numeric = (plus - minus) / (2.0 * step);
                let relative = relative_error(analytic[index], numeric);
                if relative > worst.0 {
                    worst = (relative, label, index);
                }
            }
        }
        assert!(worst.0 < 5e-3, "select adjoint disagreed with central difference: {worst:?}");
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod reduce_multiply_tests {
    use alloc::vec;

    use proxima_tensor::op::Extent;

    use super::*;

    fn leaf(program: &mut Vec<Op>, name: &str, extent: usize) -> NodeId {
        proxima_tensor::op::append(
            program,
            Op::Input { dtype: DType::Float32, shape: vec![Extent::Static(extent as u32)], name: Some(name.into()) },
        )
    }

    /// `loss = prod(x)` for `x = [2, 3, -1, 4]`, no zero anywhere in `x`, so
    /// the divide-form rule (`differentiate_reduce`'s `ScalarOp::Multiply`
    /// arm) is well-defined everywhere it is evaluated. Central difference
    /// against the real forward program is the oracle -- it has no notion
    /// of "divide-form adjoint", it simply differentiates the actual
    /// product function.
    #[proxima::test]
    async fn reduce_multiply_gradient_matches_central_difference() {
        let x_values = [2.0f32, 3.0, -1.0, 4.0];
        let mut program = Vec::new();
        let x = leaf(&mut program, "x", x_values.len());
        let loss = proxima_tensor::op::append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                init: ReduceInit::One,
                operand: x,
                in_map: expr::identity(1),
                out_map: expr::broadcast(1),
                keep: Keep::Reduce,
                name: None,
            }),
        );

        let differentiated = differentiate(&program, loss).expect("Reduce(Multiply) differentiates");
        let grad_x = differentiated.gradient_of_named("x").expect("x feeds the loss");
        let evaluated = proxima_tensor::cpu::evaluate_named(&differentiated.program, &[], &[("x", &x_values)], &[grad_x])
            .expect("adjoint program lowers and evaluates");
        let analytic = evaluated.get(grad_x).expect("grad_x requested").0;

        let loss_at = |perturbed: &[f32]| {
            proxima_tensor::cpu::evaluate_named(&program, &[], &[("x", perturbed)], &[loss])
                .expect("forward program lowers and evaluates")
                .get(loss)
                .expect("loss requested")
                .0[0]
        };

        let step = 1e-3f32;
        let mut perturbed = x_values.to_vec();
        for index in 0..x_values.len() {
            let original = perturbed[index];
            perturbed[index] = original + step;
            let plus = loss_at(&perturbed);
            perturbed[index] = original - step;
            let minus = loss_at(&perturbed);
            perturbed[index] = original;

            let numeric = (plus - minus) / (2.0 * step);
            let relative = (analytic[index] - numeric).abs() / (analytic[index].abs().max(numeric.abs()) + 1e-6);
            assert!(relative < 5e-3, "index {index}: analytic={} numeric={numeric}", analytic[index]);
        }
    }

    /// Closed-form check independent of central difference: `x = [2, 5]`,
    /// `prod(x) = 10`, so `grad = [10/2, 10/5] = [5, 2]` exactly -- the
    /// same "known, hand-computable adjoint" shape
    /// `training_loop.rs`'s `maximum_reduce_adjoint_routes_the_full_gradient_to_the_unique_argmax_only`
    /// uses for `Reduce(Maximum)`.
    #[proxima::test]
    async fn reduce_multiply_gradient_matches_the_exact_divide_form() {
        let x_values = [2.0f32, 5.0];
        let mut program = Vec::new();
        let x = leaf(&mut program, "x", x_values.len());
        let loss = proxima_tensor::op::append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                init: ReduceInit::One,
                operand: x,
                in_map: expr::identity(1),
                out_map: expr::broadcast(1),
                keep: Keep::Reduce,
                name: None,
            }),
        );

        let differentiated = differentiate(&program, loss).expect("Reduce(Multiply) differentiates");
        let grad_x = differentiated.gradient_of_named("x").expect("x feeds the loss");
        let evaluated = proxima_tensor::cpu::evaluate_named(&differentiated.program, &[], &[("x", &x_values)], &[grad_x])
            .expect("adjoint program lowers and evaluates");
        let analytic = evaluated.get(grad_x).expect("grad_x requested").0;

        assert!((analytic[0] - 5.0).abs() < 1e-5, "got {analytic:?}");
        assert!((analytic[1] - 2.0).abs() < 1e-5, "got {analytic:?}");
    }
}
