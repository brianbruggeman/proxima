//! `train_step` and `fit`: the hand-written host-buffer bookkeeping
//! `proxima-autograd/tests/training_loop.rs:335-382`'s
//! `adam_training_decreases_the_loss_over_the_dataset` already writes by
//! hand (build the `named` list for [`proxima_tensor::cpu::evaluate_named`]
//! every step, then pull every updated buffer back out one field at a
//! time), generalized into two free functions instead of a `Trainer`
//! struct.
//!
//! **No new type here on purpose.** [`train_step`] is one
//! [`proxima_tensor::cpu::evaluate_named`] call (the sole existing
//! primitive this whole module composes) plus the zip between `rebind`'s
//! `NodeId`s and their bound-buffer names that the hand-written loop
//! otherwise repeats once per parameter; [`fit`] is two nested loops
//! calling [`train_step`]. Neither answer changed the shape of anything
//! `proxima-tensor`/`proxima-autograd` already export — see this crate's
//! own report for the "what can a caller do that they could not before"
//! check this module was held to, with both the free-function and a
//! hypothetical `Pipe`-form call site written out.
//!
//! A "batch" is `Vec<(&str, &[f32])>` — named host buffers, exactly what
//! [`proxima_tensor::cpu::evaluate_named`] already takes as its own `named`
//! parameter. No `Dataset`/`DataLoader` type: a caller chunks its own
//! arrays into that shape the same way `training_loop.rs`'s `Dataset`
//! already does today, one `Vec` literal per epoch loop.

use alloc::string::String;
use alloc::vec::Vec;

use proxima_tensor::TensorError;
use proxima_tensor::cpu::evaluate_named;
use proxima_tensor::op::{NodeId, Op};

/// Every `rebind` name's freshly evaluated buffer -- a type alias, not a
/// struct: nothing a caller does with this shape it could not already do
/// with the bare `Vec<(String, Vec<f32>)>` it names (see this module's own
/// doc for the check this module was held to before minting anything).
pub type State = Vec<(String, Vec<f32>)>;

/// Runs `program` once: binds `named` (this step's batch plus every
/// currently-bound state buffer -- parameters, optimizer state, the step
/// counter), evaluates `loss` and every `rebind` node, and hands back the
/// scalar loss plus each `rebind` name's freshly evaluated buffer -- the
/// exact bindings a caller threads into the next call's `named`, keyed by
/// name rather than position so the loop needs no positional bookkeeping
/// between steps.
///
/// `rebind` pairs a node this step just computed (e.g. [`crate::optimizer::adam_step`]'s
/// `new_param`) with the [`Op::Input`] name it rebinds for the next step
/// (e.g. `"w1"`) -- the same name [`proxima_tensor::cpu::evaluate_named`]
/// already binds by, not a second identifier scheme.
///
/// # Errors
///
/// Propagates [`TensorError`] from [`proxima_tensor::cpu::evaluate_named`]
/// unchanged -- a malformed program or a missing binding is the caller's
/// program to fix, not this function's to reinterpret.
pub fn train_step(
    program: &[Op],
    loss: NodeId,
    rebind: &[(NodeId, &str)],
    named: &[(&str, &[f32])],
) -> Result<(f32, State), TensorError> {
    let mut outputs = Vec::with_capacity(rebind.len() + 1);
    outputs.push(loss);
    outputs.extend(rebind.iter().map(|(node, _)| *node));

    let evaluated = evaluate_named(program, &[], named, &outputs)?;
    let loss_value = evaluated.get(loss).and_then(|(data, _)| data.first().copied()).unwrap_or(0.0);
    let next_state = rebind
        .iter()
        .map(|(node, name)| {
            let values = evaluated.get(*node).map_or_else(Vec::new, |(data, _)| data.to_vec());
            (String::from(*name), values)
        })
        .collect();

    Ok((loss_value, next_state))
}

/// `epochs` repetitions of one pass over `batches`, threading `state` (the
/// `rebind` names' bound buffers -- parameters and optimizer state) forward
/// through [`train_step`] one batch at a time, across every epoch boundary.
/// Returns the final `state` and the per-step loss curve (`epochs *
/// batches.len()` entries, in the order stepped) -- the same loss-curve
/// shape `training_loop.rs`'s own `adam_training_decreases_the_loss_over_the_dataset`
/// prints and asserts against.
///
/// # Errors
///
/// Propagates the first [`TensorError`] any [`train_step`] call raises;
/// stops immediately rather than continuing to train on a program that has
/// already failed to evaluate once.
pub fn fit<'a>(
    program: &[Op],
    loss: NodeId,
    rebind: &[(NodeId, &str)],
    mut state: State,
    epochs: u32,
    batches: &[Vec<(&'a str, &'a [f32])>],
) -> Result<(State, Vec<f32>), TensorError> {
    let mut loss_curve = Vec::with_capacity(epochs as usize * batches.len());
    for _epoch in 0..epochs {
        for batch in batches {
            let named: Vec<(&str, &[f32])> = batch
                .iter()
                .copied()
                .chain(state.iter().map(|(name, values)| (name.as_str(), values.as_slice())))
                .collect();
            let (loss_value, next_state) = train_step(program, loss, rebind, &named)?;
            loss_curve.push(loss_value);
            state = next_state;
        }
    }
    Ok((state, loss_curve))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec;

    use proxima_tensor::dtype::DType;
    use proxima_tensor::op::Extent;

    use super::*;
    use crate::adjoint::differentiate;

    fn leaf(program: &mut Vec<Op>, name: &str, extent: u32) -> NodeId {
        proxima_tensor::op::append(
            program,
            Op::Input { dtype: DType::Float32, shape: vec![Extent::Static(extent)], name: Some(name.into()) },
        )
    }

    /// `loss = sum((x - target)^2)` over a single scalar `x`, so one SGD
    /// step is hand-verifiable: `grad = 2*(x - target)`, `x' = x -
    /// lr*grad`.
    #[proxima::test]
    async fn train_step_rebinds_by_name_and_matches_one_hand_computed_sgd_step() {
        let mut program = Vec::new();
        let x = leaf(&mut program, "x", 1);
        let target = leaf(&mut program, "target", 1);
        let diff = proxima_tensor::op::append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: proxima_tensor::op::ScalarOp::Subtract,
                operands: vec![(x, crate::expr::identity(1)), (target, crate::expr::identity(1))],
                name: None,
            },
        );
        let squared = proxima_tensor::op::append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: proxima_tensor::op::ScalarOp::Multiply,
                operands: vec![(diff, crate::expr::identity(1)), (diff, crate::expr::identity(1))],
                name: None,
            },
        );
        let loss = proxima_tensor::op::append(
            &mut program,
            Op::Reduce(proxima_tensor::op::Reduce {
                dtype: DType::Float32,
                body: proxima_tensor::op::ScalarOp::Add,
                init: proxima_tensor::op::ReduceInit::Zero,
                operand: squared,
                in_map: crate::expr::identity(1),
                out_map: crate::expr::broadcast(1),
                keep: proxima_tensor::op::Keep::Reduce,
                name: None,
            }),
        );

        let differentiated = differentiate(&program, loss).expect("scalar loss differentiates");
        let grad_x = differentiated.gradient_of_named("x").expect("x feeds the loss");
        let mut trained_program = differentiated.program;
        let config = crate::optimizer::SgdConfig { learning_rate: 0.1 };
        let new_x = crate::optimizer::sgd_step(&mut trained_program, &config, 1, x, grad_x);

        let x_values = [2.0f32];
        let target_values = [0.0f32];
        let (loss_value, next_state) = train_step(
            &trained_program,
            loss,
            &[(new_x, "x")],
            &[("x", &x_values), ("target", &target_values)],
        )
        .expect("train_step evaluates");

        assert!((loss_value - 4.0).abs() < 1e-5, "loss = (2-0)^2 = 4, got {loss_value}");
        assert_eq!(next_state.len(), 1);
        assert_eq!(next_state[0].0, "x");
        // grad = 2*(2-0) = 4, x' = 2 - 0.1*4 = 1.6
        assert!((next_state[0].1[0] - 1.6).abs() < 1e-5, "got {next_state:?}");
    }

    #[proxima::test]
    async fn fit_decreases_the_loss_over_repeated_epochs() {
        let mut program = Vec::new();
        let x = leaf(&mut program, "x", 1);
        let target = leaf(&mut program, "target", 1);
        let diff = proxima_tensor::op::append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: proxima_tensor::op::ScalarOp::Subtract,
                operands: vec![(x, crate::expr::identity(1)), (target, crate::expr::identity(1))],
                name: None,
            },
        );
        let squared = proxima_tensor::op::append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: proxima_tensor::op::ScalarOp::Multiply,
                operands: vec![(diff, crate::expr::identity(1)), (diff, crate::expr::identity(1))],
                name: None,
            },
        );
        let loss = proxima_tensor::op::append(
            &mut program,
            Op::Reduce(proxima_tensor::op::Reduce {
                dtype: DType::Float32,
                body: proxima_tensor::op::ScalarOp::Add,
                init: proxima_tensor::op::ReduceInit::Zero,
                operand: squared,
                in_map: crate::expr::identity(1),
                out_map: crate::expr::broadcast(1),
                keep: proxima_tensor::op::Keep::Reduce,
                name: None,
            }),
        );

        let differentiated = differentiate(&program, loss).expect("scalar loss differentiates");
        let grad_x = differentiated.gradient_of_named("x").expect("x feeds the loss");
        let mut trained_program = differentiated.program;
        let config = crate::optimizer::SgdConfig { learning_rate: 0.1 };
        let new_x = crate::optimizer::sgd_step(&mut trained_program, &config, 1, x, grad_x);

        let target_values = [0.0f32];
        let batches = vec![vec![("target", target_values.as_slice())]];
        let initial_state = vec![(String::from("x"), vec![2.0f32])];

        let (_final_state, loss_curve) =
            fit(&trained_program, loss, &[(new_x, "x")], initial_state, 20, &batches).expect("fit runs to completion");

        assert_eq!(loss_curve.len(), 20);
        assert!(
            loss_curve.last().expect("at least one step ran") < &loss_curve[0],
            "loss must decrease over training, got {loss_curve:?}"
        );
        assert!(loss_curve.iter().all(|value| value.is_finite()), "loss went non-finite: {loss_curve:?}");
    }
}
