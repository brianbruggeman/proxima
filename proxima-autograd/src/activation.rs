//! `relu` and `softmax`, spelled as graph-building functions over
//! [`proxima_tensor::op::Op`] — never a new `Op`/`ScalarOp` variant.
//!
//! `ScalarOp` is deliberately closed (`proxima-tensor/src/op.rs:52-56`:
//! "these are scalar machine primitives, not an extension point"), and
//! this module is the proof: `relu(x) = max(x, 0)` is one existing
//! [`ScalarOp::Maximum`] against a rank-0 [`proxima_tensor::op::Op::Constant`],
//! and `softmax` is the exact five-node shape
//! `proxima-tensor/src/spec.rs:1058-1064` already builds for causal
//! attention — max-subtract, `Exponential`, sum-`Reduce`, `Reciprocal`,
//! `Multiply` — spelled here with the same `map::projection` primitive
//! `spec.rs`'s einsum strings desugar into, not that private parser.

use alloc::vec::Vec;

use proxima_tensor::dtype::DType;
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::op::{NodeId, Op, ReduceInit, ScalarOp};

use crate::expr;

/// `max(x, 0)`, elementwise, at whatever rank `x` was built at.
///
/// One [`ScalarOp::Maximum`] against a broadcasting rank-0
/// [`Op::Constant`] — see this module's own doc for why that is the whole
/// function and no `Op::Relu` variant exists.
///
/// ```
/// use proxima_autograd::activation::relu;
/// use proxima_tensor::dtype::DType;
/// use proxima_tensor::op::{self, Extent, Op};
///
/// let mut program = Vec::new();
/// let x = op::append(
///     &mut program,
///     Op::Input { dtype: DType::Float32, shape: vec![Extent::Static(4)], name: Some("x".into()) },
/// );
/// let out = relu(&mut program, DType::Float32, x, 1);
///
/// let values = [-2.0f32, -0.0, 0.5, 3.0];
/// let evaluated = proxima_tensor::cpu::evaluate(&program, &[], &[&values], &[out])
///     .expect("relu program lowers and evaluates");
/// assert_eq!(evaluated.root(), &[0.0, 0.0, 0.5, 3.0]);
/// ```
#[must_use]
pub fn relu(program: &mut Vec<Op>, dtype: DType, x: NodeId, rank: u16) -> NodeId {
    let zero = expr::constant(program, dtype, 0.0);
    expr::binary(
        program,
        dtype,
        ScalarOp::Maximum,
        (x, expr::identity(rank)),
        (zero, expr::broadcast(rank)),
    )
}

/// Softmax of `x` over iteration axis `axis`, out of `rank` total axes.
///
/// Max-subtract for numerical stability, `Exponential`, a sum-`Reduce`
/// dropping `axis`, `Reciprocal`, and a broadcasting `Multiply` — the same
/// five expressions `proxima-tensor/src/spec.rs`'s causal-attention softmax
/// builds (`scores_masked` through `probabilities`, `spec.rs:1058-1064`),
/// generalized to an arbitrary rank/axis instead of that call site's fixed
/// `stug` iteration space.
#[must_use]
pub fn softmax(program: &mut Vec<Op>, dtype: DType, x: NodeId, rank: u16, axis: u16) -> NodeId {
    let reduced_axes: Vec<u16> = (0..rank).filter(|candidate| *candidate != axis).collect();
    let reduced_rank = reduced_axes.len() as u16;
    let out_map = IndexMap::Affine(map::projection(rank, &reduced_axes));

    let max_val = expr::reduce(
        program,
        dtype,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        x,
        expr::identity(rank),
        out_map.clone(),
    );
    let shifted = expr::binary(
        program,
        dtype,
        ScalarOp::Subtract,
        (x, expr::identity(rank)),
        (max_val, out_map.clone()),
    );
    let exponentiated = expr::unary(
        program,
        dtype,
        ScalarOp::Exponential,
        (shifted, expr::identity(rank)),
    );
    let sum_exp = expr::reduce(
        program,
        dtype,
        ScalarOp::Add,
        ReduceInit::Zero,
        exponentiated,
        expr::identity(rank),
        out_map.clone(),
    );
    let inverse_sum = expr::unary(
        program,
        dtype,
        ScalarOp::Reciprocal,
        (sum_exp, expr::identity(reduced_rank)),
    );
    expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (exponentiated, expr::identity(rank)),
        (inverse_sum, out_map),
    )
}

/// `1 / (1 + exp(-x))`, elementwise, at whatever rank `x` was built at.
///
/// Four existing [`ScalarOp`]s -- `Negate`, `Exponential`, `Add` against a
/// broadcasting rank-0 [`Op::Constant`], `Reciprocal` -- the same
/// composition-over-extension shape [`relu`]'s own doc argues for: no
/// `ScalarOp::Sigmoid` variant exists because this module's whole job is to
/// prove none is needed.
#[must_use]
pub fn sigmoid(program: &mut Vec<Op>, dtype: DType, x: NodeId, rank: u16) -> NodeId {
    let full = expr::identity(rank);
    let scalar = expr::broadcast(rank);
    let negated = expr::unary(program, dtype, ScalarOp::Negate, (x, full.clone()));
    let exponentiated = expr::unary(program, dtype, ScalarOp::Exponential, (negated, full.clone()));
    let one = expr::constant(program, dtype, 1.0);
    let denominator = expr::binary(program, dtype, ScalarOp::Add, (one, scalar), (exponentiated, full));
    expr::unary(program, dtype, ScalarOp::Reciprocal, (denominator, expr::identity(rank)))
}

/// `x * sigmoid(x)` (SiLU / Swish), elementwise, at whatever rank `x` was
/// built at. One [`ScalarOp::Multiply`] wrapped around [`sigmoid`] -- the
/// gradient falls out of the existing `Multiply` and `Reciprocal` adjoint
/// rules (`adjoint.rs:303-328`) with no rule of its own.
#[must_use]
pub fn silu(program: &mut Vec<Op>, dtype: DType, x: NodeId, rank: u16) -> NodeId {
    let sigmoid_x = sigmoid(program, dtype, x, rank);
    expr::binary(program, dtype, ScalarOp::Multiply, (x, expr::identity(rank)), (sigmoid_x, expr::identity(rank)))
}

/// `0.5 * x * (1 + erf(x / sqrt(2)))` (exact, not the tanh approximation),
/// elementwise, at whatever rank `x` was built at. Built entirely from
/// [`ScalarOp::Erf`] (already in this crate's closed set, `op.rs:74`, with
/// its own adjoint rule at `adjoint.rs:349-357`) plus `Multiply`/`Add`
/// against broadcasting [`Op::Constant`]s -- no `ScalarOp::Gelu` variant.
#[must_use]
pub fn gelu(program: &mut Vec<Op>, dtype: DType, x: NodeId, rank: u16) -> NodeId {
    let full = expr::identity(rank);
    let scalar = expr::broadcast(rank);
    let inverse_sqrt_two = expr::constant(program, dtype, core::f32::consts::FRAC_1_SQRT_2);
    let scaled = expr::binary(program, dtype, ScalarOp::Multiply, (x, full.clone()), (inverse_sqrt_two, scalar.clone()));
    let erf_scaled = expr::unary(program, dtype, ScalarOp::Erf, (scaled, expr::identity(rank)));
    let one = expr::constant(program, dtype, 1.0);
    let one_plus_erf = expr::binary(program, dtype, ScalarOp::Add, (one, scalar.clone()), (erf_scaled, full.clone()));
    let x_times = expr::binary(program, dtype, ScalarOp::Multiply, (x, full), (one_plus_erf, expr::identity(rank)));
    let half = expr::constant(program, dtype, 0.5);
    expr::binary(program, dtype, ScalarOp::Multiply, (half, scalar), (x_times, expr::identity(rank)))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec;

    use proxima_tensor::op::{Extent, Op};

    use super::*;

    #[proxima::test]
    async fn relu_zeroes_negatives_and_passes_positives() {
        let mut program = Vec::new();
        let x = proxima_tensor::op::append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(4)],
                name: Some("x".into()),
            },
        );
        let out = relu(&mut program, DType::Float32, x, 1);

        let values = [-2.0f32, -0.0, 0.5, 3.0];
        let evaluated = proxima_tensor::cpu::evaluate(&program, &[], &[&values], &[out])
            .expect("relu program lowers and evaluates");
        assert_eq!(evaluated.root(), &[0.0, 0.0, 0.5, 3.0]);
    }

    #[proxima::test]
    async fn softmax_sums_to_one_and_matches_hand_computed_values() {
        let mut program = Vec::new();
        let x = proxima_tensor::op::append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(3)],
                name: Some("x".into()),
            },
        );
        let out = softmax(&mut program, DType::Float32, x, 1, 0);

        let values = [1.0f32, 2.0, 3.0];
        let evaluated = proxima_tensor::cpu::evaluate(&program, &[], &[&values], &[out])
            .expect("softmax program lowers and evaluates");
        let result = evaluated.root();

        let denom = (-2.0f32).exp() + (-1.0f32).exp() + 0.0f32.exp();
        let expected = [(-2.0f32).exp() / denom, (-1.0f32).exp() / denom, 1.0 / denom];
        for (got, want) in result.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "got {got}, want {want}, full result {result:?}"
            );
        }
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax must sum to 1, got {sum}");
    }

    #[proxima::test]
    async fn sigmoid_matches_hand_computed_values() {
        let mut program = Vec::new();
        let x = proxima_tensor::op::append(
            &mut program,
            Op::Input { dtype: DType::Float32, shape: vec![Extent::Static(3)], name: Some("x".into()) },
        );
        let out = sigmoid(&mut program, DType::Float32, x, 1);

        let values = [-1.0f32, 0.0, 1.0];
        let evaluated = proxima_tensor::cpu::evaluate(&program, &[], &[&values], &[out])
            .expect("sigmoid program lowers and evaluates");
        let expected = [0.268_941_42f32, 0.5, 0.731_058_6];
        for (got, want) in evaluated.root().iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-5, "got {got}, want {want}");
        }
    }

    #[proxima::test]
    async fn silu_matches_hand_computed_values() {
        let mut program = Vec::new();
        let x = proxima_tensor::op::append(
            &mut program,
            Op::Input { dtype: DType::Float32, shape: vec![Extent::Static(3)], name: Some("x".into()) },
        );
        let out = silu(&mut program, DType::Float32, x, 1);

        let values = [-1.0f32, 0.0, 1.0];
        let evaluated = proxima_tensor::cpu::evaluate(&program, &[], &[&values], &[out])
            .expect("silu program lowers and evaluates");
        let expected = [-0.268_941_42f32, 0.0, 0.731_058_6];
        for (got, want) in evaluated.root().iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-5, "got {got}, want {want}");
        }
    }

    #[proxima::test]
    async fn gelu_matches_hand_computed_values() {
        let mut program = Vec::new();
        let x = proxima_tensor::op::append(
            &mut program,
            Op::Input { dtype: DType::Float32, shape: vec![Extent::Static(3)], name: Some("x".into()) },
        );
        let out = gelu(&mut program, DType::Float32, x, 1);

        let values = [-1.0f32, 0.0, 1.0];
        let evaluated = proxima_tensor::cpu::evaluate(&program, &[], &[&values], &[out])
            .expect("gelu program lowers and evaluates");
        let expected = [-0.158_655_25f32, 0.0, 0.841_344_7];
        for (got, want) in evaluated.root().iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
    }

    /// One central-difference gradient check per activation, proving the
    /// gradient falls out of the existing adjoint rules composed over
    /// `Negate`/`Exponential`/`Add`/`Reciprocal`/`Multiply`/`Erf` -- no
    /// activation-specific adjoint rule exists anywhere in this crate.
    fn activation_gradient_matches_central_difference(activation: fn(&mut Vec<Op>, DType, NodeId, u16) -> NodeId) {
        let x_values = [-1.7f32, -0.3, 0.4, 2.1];
        let mut program = Vec::new();
        let x = proxima_tensor::op::append(
            &mut program,
            Op::Input { dtype: DType::Float32, shape: vec![Extent::Static(x_values.len() as u32)], name: Some("x".into()) },
        );
        let activated = activation(&mut program, DType::Float32, x, 1);
        let loss = proxima_tensor::op::append(
            &mut program,
            Op::Reduce(proxima_tensor::op::Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: proxima_tensor::op::ReduceInit::Zero,
                operand: activated,
                in_map: expr::identity(1),
                out_map: expr::broadcast(1),
                keep: proxima_tensor::op::Keep::Reduce,
                name: None,
            }),
        );

        let differentiated = crate::adjoint::differentiate(&program, loss).expect("scalar loss differentiates");
        let grad_x = differentiated.gradient_of_named("x").expect("x feeds the loss");
        let evaluated = proxima_tensor::cpu::evaluate_named(&differentiated.program, &[], &[("x", &x_values)], &[grad_x])
            .expect("adjoint program lowers and evaluates");
        let analytic = evaluated.get(grad_x).expect("grad_x requested").0;

        let step = 1e-3f32;
        let loss_at = |perturbed: &[f32]| {
            proxima_tensor::cpu::evaluate_named(&program, &[], &[("x", perturbed)], &[loss])
                .expect("forward program lowers and evaluates")
                .get(loss)
                .expect("loss requested")
                .0[0]
        };

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
            assert!(
                relative < 5e-3,
                "index {index}: analytic={} numeric={numeric} relative={relative}",
                analytic[index]
            );
        }
    }

    #[proxima::test]
    async fn sigmoid_gradient_matches_central_difference() {
        activation_gradient_matches_central_difference(sigmoid);
    }

    #[proxima::test]
    async fn silu_gradient_matches_central_difference() {
        activation_gradient_matches_central_difference(silu);
    }

    #[proxima::test]
    async fn gelu_gradient_matches_central_difference() {
        activation_gradient_matches_central_difference(gelu);
    }
}
