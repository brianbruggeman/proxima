//! The adjoint transform: forward program plus a scalar loss node in,
//! adjoint program out — [`differentiate`] is a pure, synchronous
//! `&[Op] -> Differentiated` function, not a [`proxima_primitives::pipe::Pipe`].
//! `Pipe::call` is `async` (RPITIT, `proxima-primitives/src/pipe/primitives.rs:101`)
//! because the algebra is an I/O-composition surface; graph construction
//! has no I/O behind it, so forcing an async boundary here would be
//! manufacturing one to earn a name, exactly the error this workspace's own
//! `AGENTS.md` names and rejects.
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
//! - **A gathered (`IndexMap::Computed`) operand's adjoint is a
//!   scatter-add** by the same mask-composition recipe, but it is
//!   `O(destination x source)` dense (`cpu.rs:16062`'s own doc: at
//!   embedding scale, 128k x 4k is 524M mask elements to accumulate 4k
//!   values) and is rejected here with
//!   [`AutogradError::GatherAdjointUnsupported`] rather than shipped
//!   half-verified under time pressure — see this crate's own report for
//!   why that trade was made deliberately, not by omission.
//! - **A non-pure-projection operand map** (a convolution-style window,
//!   multi-term or non-unit-coefficient) cannot be reused as a backward
//!   `Reduce`'s `out_map` — `proxima-tensor/src/shape.rs:437-453` rejects
//!   any such `out_map` — so it is rejected here too, named precisely
//!   rather than attempted.

use alloc::vec;
use alloc::vec::Vec;

use proxima_tensor::dtype::DType;
use proxima_tensor::map::IndexMap;
use proxima_tensor::op::{Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp};
use proxima_tensor::shape::{self, Shapes};

use crate::error::AutogradError;
use crate::expr;

/// The forward program's own nodes (unchanged, same indices) plus every
/// adjoint node this transform appended, and the gradient node for each
/// [`Op::Input`] the loss actually depends on.
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
}

impl Differentiated {
    /// The gradient node for `node`, if the loss depends on it.
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
        self.program
            .iter()
            .enumerate()
            .find_map(|(index, op)| match op {
                Op::Input { name: Some(candidate), .. } if candidate == name => {
                    Some(NodeId(index as u32))
                }
                _ => None,
            })
            .and_then(|node| self.gradient_of(node))
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
    grad_of[loss_index] = Some(expr::constant(&mut new_program, DType::Float32, 1.0));

    for index in (0..=loss_index).rev() {
        let Some(gradient) = grad_of[index] else { continue };
        let node = NodeId(index as u32);
        match &program[index] {
            Op::Input { .. } | Op::Iota { .. } | Op::Constant { .. } => {}
            Op::Elementwise { dtype, body, operands, .. } => differentiate_elementwise(
                &mut new_program,
                &mut grad_of,
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

    Ok(Differentiated { program: new_program, loss, gradients })
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
        route_contribution(program, grad_of, original_program, shapes, node, operand, contribution, iter_rank)?;
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

#[allow(clippy::too_many_arguments)]
fn route_contribution(
    program: &mut Vec<Op>,
    grad_of: &mut [Option<NodeId>],
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

    let operand_rank = shapes.of(*operand_node).len() as u16;
    let dtype = original_program[operand_node.0 as usize].dtype();

    let routed = match operand_map {
        IndexMap::Affine(pattern) => {
            if !expr::is_pure_projection(pattern) {
                return Err(AutogradError::NonProjectionOperandMap { node: consumer, operand: *operand_node });
            }
            expr::reduce(
                program,
                dtype,
                ScalarOp::Add,
                ReduceInit::Zero,
                contribution,
                expr::identity(iter_rank),
                IndexMap::Affine(pattern.clone()),
            )
        }
        IndexMap::Computed { .. } => {
            return Err(AutogradError::GatherAdjointUnsupported { node: consumer, operand: *operand_node });
        }
    };

    accumulate(program, grad_of, dtype, operand_rank, operand_node.0 as usize, routed);
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
