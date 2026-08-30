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
}
