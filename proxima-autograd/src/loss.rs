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
/// of `rank` axes: `-sum(one_hot * log(probs))`.
///
/// Unnormalized (a sum, not a mean) — the same convention
/// `training_loop.rs:149-152`'s inline shape uses for one example at a
/// time; a caller batching examples divides by the batch size itself, the
/// same way [`mse`] takes `element_count` explicitly rather than assuming
/// what "the batch" means for its caller's iteration space.
#[must_use]
pub fn cross_entropy(program: &mut Vec<Op>, dtype: DType, probs: NodeId, one_hot: NodeId, rank: u16) -> NodeId {
    let full = expr::identity(rank);
    let scalar = expr::broadcast(rank);

    let log_probs = expr::unary(program, dtype, ScalarOp::Logarithm, (probs, full.clone()));
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
}
