//! `mse` and `cross_entropy`, spelled as graph-building functions over
//! [`proxima_tensor::op::Op`] — the same shape as [`crate::activation::relu`]
//! and [`crate::activation::softmax`], never a new `Op`/`ScalarOp` variant.
//!
//! `cross_entropy` is `proxima-autograd/tests/training_loop.rs:149-152`'s
//! inline `log -> weighted -> sum -> negate` shape, generalized to an
//! arbitrary rank and lifted out of the test into the library surface;
//! `mse` is the same `Subtract`/`Multiply`/`Reduce(Add)` shape with a
//! divide-by-count in place of the negate. `softmax_cross_entropy` composes
//! [`crate::activation::softmax`] in front of `cross_entropy` — the exact
//! two-call sequence `training_loop.rs`'s `build_network` already writes by
//! hand (`softmax(...)` then the inline cross-entropy shape).

use alloc::vec::Vec;

use proxima_tensor::dtype::DType;
use proxima_tensor::op::{NodeId, Op, ReduceInit, ScalarOp};

use crate::activation;
use crate::expr;

/// Mean squared error between `pred` and `target`, reduced to a scalar over
/// every one of `rank` axes: `sum((pred - target)^2) / element_count`.
///
/// `element_count` is the flattened element count of `pred`/`target` (the
/// product of their static extents) — a host-known value at graph-build
/// time, folded in as an [`proxima_tensor::op::Op::Constant`] exactly the
/// way [`crate::optimizer::adam_step`]'s `bias_correction` folds in
/// `ln(beta)` (that module's own doc: the one thing that changes per call is
/// a runtime `Op::Input`, everything host-known is a graph-time constant).
#[must_use]
pub fn mse(program: &mut Vec<Op>, dtype: DType, pred: NodeId, target: NodeId, rank: u16, element_count: u32) -> NodeId {
    let full = expr::identity(rank);
    let scalar = expr::broadcast(rank);

    let diff = expr::binary(program, dtype, ScalarOp::Subtract, (pred, full.clone()), (target, full.clone()));
    let squared = expr::binary(program, dtype, ScalarOp::Multiply, (diff, full.clone()), (diff, full));
    let sum = expr::reduce(program, dtype, ScalarOp::Add, ReduceInit::Zero, squared, expr::identity(rank), scalar);

    let inverse_count = expr::constant(program, dtype, 1.0 / element_count as f32);
    expr::binary(program, dtype, ScalarOp::Multiply, (sum, expr::identity(0)), (inverse_count, expr::identity(0)))
}

/// Cross-entropy between predicted probabilities `probs` and a one-hot (or
/// soft) target distribution `one_hot`, reduced to a scalar over every one
/// of `rank` axes: `-sum(one_hot * log(max(probs, EPSILON)))`.
///
/// Unnormalized (a sum, not a mean) — the same convention
/// `training_loop.rs:149-152`'s inline shape uses for one example at a
/// time; a caller batching examples divides by the batch size itself, the
/// same way [`mse`] takes `element_count` explicitly rather than assuming
/// what "the batch" means for its caller's iteration space.
///
/// The `max(probs, EPSILON)` floor (one existing [`ScalarOp::Maximum`], the
/// same composed primitive [`crate::activation::relu`] uses) exists because
/// [`crate::adjoint::differentiate`] differentiates this literal graph, not
/// a hand-fused closed-form gradient: `Logarithm`'s adjoint is `gradient *
/// (1/x)` (`adjoint.rs:338-341`), and the `Multiply` feeding it puts an
/// exact `0.0` upstream gradient on every non-target class
/// (`one_hot` is `0` there). A confidently-wrong softmax can underflow a
/// non-target class's probability to exactly `0.0f32` deep into training
/// (unbounded steps, e.g. the scaling-ladder run past
/// `tests/real_mnist_training.rs`'s own 4-epoch/8000-example config) —
/// `1/0.0` is `+inf`, and `0.0 * inf` is `NaN` in IEEE754, permanently
/// poisoning every downstream Adam moment. Flooring `probs` away from `0`
/// keeps `1/x` finite so `0 * finite` stays exactly `0`, the same
/// eps-clamped-log fix every incumbent softmax-cross-entropy implementation
/// carries for the identical reason.
#[must_use]
pub fn cross_entropy(program: &mut Vec<Op>, dtype: DType, probs: NodeId, one_hot: NodeId, rank: u16) -> NodeId {
    const PROBABILITY_FLOOR: f32 = 1e-7;

    let full = expr::identity(rank);
    let scalar = expr::broadcast(rank);

    let floor = expr::constant(program, dtype, PROBABILITY_FLOOR);
    let floored_probs = expr::binary(program, dtype, ScalarOp::Maximum, (probs, full.clone()), (floor, scalar.clone()));
    let log_probs = expr::unary(program, dtype, ScalarOp::Logarithm, (floored_probs, full.clone()));
    let weighted = expr::binary(program, dtype, ScalarOp::Multiply, (one_hot, full.clone()), (log_probs, full));
    let sum = expr::reduce(program, dtype, ScalarOp::Add, ReduceInit::Zero, weighted, expr::identity(rank), scalar);
    expr::unary(program, dtype, ScalarOp::Negate, (sum, expr::identity(0)))
}

/// `cross_entropy(softmax(logits, axis), one_hot)` — softmax and
/// cross-entropy composed exactly as `training_loop.rs`'s `build_network`
/// already writes them by hand, spelled here as one call so a caller does
/// not need to thread the intermediate `probabilities` node itself.
#[must_use]
pub fn softmax_cross_entropy(
    program: &mut Vec<Op>,
    dtype: DType,
    logits: NodeId,
    one_hot: NodeId,
    rank: u16,
    axis: u16,
) -> NodeId {
    let probabilities = activation::softmax(program, dtype, logits, rank, axis);
    cross_entropy(program, dtype, probabilities, one_hot, rank)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec;

    use proxima_tensor::op::Extent;

    use super::*;

    fn leaf(program: &mut Vec<Op>, name: &str, extent: u32) -> NodeId {
        proxima_tensor::op::append(
            program,
            Op::Input { dtype: DType::Float32, shape: vec![Extent::Static(extent)], name: Some(name.into()) },
        )
    }

    #[proxima::test]
    async fn mse_matches_hand_computed_value() {
        let mut program = Vec::new();
        let pred = leaf(&mut program, "pred", 3);
        let target = leaf(&mut program, "target", 3);
        let out = mse(&mut program, DType::Float32, pred, target, 1, 3);

        let pred_values = [1.0f32, 2.0, 3.0];
        let target_values = [1.5f32, 2.5, 2.0];
        let evaluated = proxima_tensor::cpu::evaluate_named(
            &program,
            &[],
            &[("pred", &pred_values), ("target", &target_values)],
            &[out],
        )
        .expect("mse program lowers and evaluates");

        // (0.25 + 0.25 + 1.0) / 3
        let expected = 0.5f32;
        assert!((evaluated.get(out).expect("mse requested").0[0] - expected).abs() < 1e-6);
    }

    #[proxima::test]
    async fn cross_entropy_matches_hand_computed_value() {
        let mut program = Vec::new();
        let probs = leaf(&mut program, "probs", 3);
        let one_hot = leaf(&mut program, "one_hot", 3);
        let out = cross_entropy(&mut program, DType::Float32, probs, one_hot, 1);

        let probs_values = [0.2f32, 0.5, 0.3];
        let one_hot_values = [0.0f32, 1.0, 0.0];
        let evaluated = proxima_tensor::cpu::evaluate_named(
            &program,
            &[],
            &[("probs", &probs_values), ("one_hot", &one_hot_values)],
            &[out],
        )
        .expect("cross_entropy program lowers and evaluates");

        let expected = -(0.5f32.ln());
        assert!((evaluated.get(out).expect("cross_entropy requested").0[0] - expected).abs() < 1e-6);
    }

    #[proxima::test]
    async fn softmax_cross_entropy_matches_softmax_then_cross_entropy() {
        let mut logits_program = Vec::new();
        let logits = leaf(&mut logits_program, "logits", 3);
        let one_hot = leaf(&mut logits_program, "one_hot", 3);
        let fused = softmax_cross_entropy(&mut logits_program, DType::Float32, logits, one_hot, 1, 0);

        let logits_values = [1.0f32, 2.0, 3.0];
        let one_hot_values = [0.0f32, 0.0, 1.0];
        let fused_evaluated = proxima_tensor::cpu::evaluate_named(
            &logits_program,
            &[],
            &[("logits", &logits_values), ("one_hot", &one_hot_values)],
            &[fused],
        )
        .expect("fused program lowers and evaluates");
        let fused_value = fused_evaluated.get(fused).expect("fused requested").0[0];

        let mut split_program = Vec::new();
        let split_logits = leaf(&mut split_program, "logits", 3);
        let probs = activation::softmax(&mut split_program, DType::Float32, split_logits, 1, 0);
        let split_one_hot = leaf(&mut split_program, "one_hot", 3);
        let split_loss = cross_entropy(&mut split_program, DType::Float32, probs, split_one_hot, 1);
        let split_evaluated = proxima_tensor::cpu::evaluate_named(
            &split_program,
            &[],
            &[("logits", &logits_values), ("one_hot", &one_hot_values)],
            &[split_loss],
        )
        .expect("split program lowers and evaluates");
        let split_value = split_evaluated.get(split_loss).expect("split requested").0[0];

        assert!((fused_value - split_value).abs() < 1e-6, "fused={fused_value}, split={split_value}");
    }

    /// Central-difference gradient checks proving both losses' gradients
    /// fall out of existing adjoint rules (`Subtract`/`Multiply`/`Logarithm`/
    /// `Negate`/`Reduce(Add)`), no loss-specific adjoint rule anywhere here.
    #[proxima::test]
    async fn mse_gradient_matches_central_difference() {
        let pred_values = [1.0f32, -0.4, 2.3];
        let target_values = [0.2f32, 0.5, 1.9];

        let mut program = Vec::new();
        let pred = leaf(&mut program, "pred", 3);
        let target = leaf(&mut program, "target", 3);
        let loss = mse(&mut program, DType::Float32, pred, target, 1, 3);

        let differentiated = crate::adjoint::differentiate(&program, loss).expect("scalar loss differentiates");
        let grad_pred = differentiated.gradient_of_named("pred").expect("pred feeds the loss");
        let evaluated = proxima_tensor::cpu::evaluate_named(
            &differentiated.program,
            &[],
            &[("pred", &pred_values), ("target", &target_values)],
            &[grad_pred],
        )
        .expect("adjoint program lowers and evaluates");
        let analytic = evaluated.get(grad_pred).expect("grad_pred requested").0;

        let loss_at = |perturbed: &[f32]| {
            proxima_tensor::cpu::evaluate_named(&program, &[], &[("pred", perturbed), ("target", &target_values)], &[loss])
                .expect("forward program lowers and evaluates")
                .get(loss)
                .expect("loss requested")
                .0[0]
        };

        let step = 1e-3f32;
        let mut perturbed = pred_values.to_vec();
        for index in 0..pred_values.len() {
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

    #[proxima::test]
    async fn softmax_cross_entropy_gradient_matches_central_difference() {
        let logits_values = [0.6f32, -1.1, 2.4];
        let one_hot_values = [0.0f32, 1.0, 0.0];

        let mut program = Vec::new();
        let logits = leaf(&mut program, "logits", 3);
        let one_hot = leaf(&mut program, "one_hot", 3);
        let loss = softmax_cross_entropy(&mut program, DType::Float32, logits, one_hot, 1, 0);

        let differentiated = crate::adjoint::differentiate(&program, loss).expect("scalar loss differentiates");
        let grad_logits = differentiated.gradient_of_named("logits").expect("logits feeds the loss");
        let evaluated = proxima_tensor::cpu::evaluate_named(
            &differentiated.program,
            &[],
            &[("logits", &logits_values), ("one_hot", &one_hot_values)],
            &[grad_logits],
        )
        .expect("adjoint program lowers and evaluates");
        let analytic = evaluated.get(grad_logits).expect("grad_logits requested").0;

        let loss_at = |perturbed: &[f32]| {
            proxima_tensor::cpu::evaluate_named(&program, &[], &[("logits", perturbed), ("one_hot", &one_hot_values)], &[loss])
                .expect("forward program lowers and evaluates")
                .get(loss)
                .expect("loss requested")
                .0[0]
        };

        let step = 1e-3f32;
        let mut perturbed = logits_values.to_vec();
        for index in 0..logits_values.len() {
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

    /// Reproduces the scaling-ladder failure: a softmax so confidently wrong
    /// on one class that its probability underflows to exactly `0.0f32`
    /// (the `-60.0` logit here does that -- `exp(-60)` underflows f32 well
    /// before it reaches the softmax denominator). Before the
    /// `max(probs, PROBABILITY_FLOOR)` floor in [`cross_entropy`], this
    /// produced `0.0 * (1/0.0) = NaN` through [`crate::adjoint`]'s literal
    /// `Logarithm` adjoint (`gradient * (1/x)`, `adjoint.rs:338-341`) for
    /// every non-target class -- the mechanism the scaling ladder in
    /// `tests/real_mnist_training.rs`'s own doc comment surfaced past its
    /// shipped 4-epoch/8000-example config.
    #[proxima::test]
    async fn softmax_cross_entropy_gradient_stays_finite_when_a_class_probability_underflows_to_zero() {
        let logits_values = [-60.0f32, 0.0, 1.0];
        let one_hot_values = [0.0f32, 0.0, 1.0];

        let mut program = Vec::new();
        let logits = leaf(&mut program, "logits", 3);
        let one_hot = leaf(&mut program, "one_hot", 3);
        let loss = softmax_cross_entropy(&mut program, DType::Float32, logits, one_hot, 1, 0);

        let differentiated = crate::adjoint::differentiate(&program, loss).expect("scalar loss differentiates");
        let grad_logits = differentiated.gradient_of_named("logits").expect("logits feeds the loss");
        let evaluated = proxima_tensor::cpu::evaluate_named(
            &differentiated.program,
            &[],
            &[("logits", &logits_values), ("one_hot", &one_hot_values)],
            &[loss, grad_logits],
        )
        .expect("adjoint program lowers and evaluates");

        let loss_value = evaluated.get(loss).expect("loss requested").0[0];
        assert!(loss_value.is_finite(), "loss went non-finite: {loss_value}");

        let gradient = evaluated.get(grad_logits).expect("grad_logits requested").0;
        assert!(
            gradient.iter().all(|value| value.is_finite()),
            "expected every logit gradient finite with an underflowed class probability, got {gradient:?}"
        );
    }
}
