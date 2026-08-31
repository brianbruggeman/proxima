//! Dropout as a graph-building function, the same shape
//! [`crate::activation::relu`]/[`crate::activation::softmax`]/
//! [`crate::optimizer::adam_step`] already are: no `Op::Dropout` -- every
//! quantity here is an existing `Op::Elementwise` composed over an existing
//! [`ScalarOp`], and the gradient falls out of the adjoint rule that `Op`
//! variant already carries (`adjoint.rs`'s `differentiate_elementwise`), not
//! a rule of its own.
//!
//! [`dropout`] needs no `Op::Random`: `ScalarOp`'s own closed set
//! (`proxima-tensor/src/op.rs:60-77`) has no Bernoulli sampler and none is
//! added here -- the 0/1 mask is a caller-supplied [`Op::Input`], generated
//! host-side once per training step (`proxima_tensor::test_support::Lcg` in
//! tests, the caller's own RNG in production) and bound by name through
//! [`proxima_tensor::cpu::evaluate_named`], exactly the way
//! [`crate::optimizer::adam_step`]'s `m`/`v` are ordinary re-bound
//! `Op::Input` leaves (`optimizer.rs`'s own module doc). Eval mode inserts
//! no dropout node at all -- "the graph IS the value"
//! (`proxima_tensor`'s own crate doc), so train and eval are two different
//! graphs built from the same forward function, not one graph with a
//! runtime branch.

use alloc::vec::Vec;

use proxima_tensor::dtype::DType;
use proxima_tensor::op::{NodeId, Op, ScalarOp};

use crate::expr;

/// Inverted dropout: `(x * mask) / keep_prob`, elementwise, at whatever
/// rank `x` was built at. `mask` is a caller-supplied [`Op::Input`] the same
/// shape as `x` -- a host-generated 0/1 Bernoulli draw, never a graph-level
/// random source (this module's own doc explains why none exists). Scaling
/// by `1 / keep_prob` at train time ("inverted" dropout) means eval time
/// needs no rescale at all: skip calling this function on the eval graph, or
/// call it with `keep_prob = 1.0` against an all-ones bound `mask` if the
/// caller would rather keep one graph shape across both modes (both are
/// documented, neither is privileged -- see this module's own doc).
///
/// The gradient w.r.t. `x` is `grad_out * mask / keep_prob`, exactly
/// [`ScalarOp::Multiply`]'s existing adjoint rule
/// (`adjoint.rs:309-315`) applied twice, with `mask` treated as a constant
/// input (its own gradient, if requested, is simply never read) -- no
/// dropout-specific adjoint rule exists anywhere in this crate.
///
/// ```
/// use proxima_autograd::norm::dropout;
/// use proxima_tensor::dtype::DType;
/// use proxima_tensor::op::{self, Extent, Op};
///
/// let mut program = Vec::new();
/// let x = op::append(
///     &mut program,
///     Op::Input { dtype: DType::Float32, shape: vec![Extent::Static(4)], name: Some("x".into()) },
/// );
/// let mask = op::append(
///     &mut program,
///     Op::Input { dtype: DType::Float32, shape: vec![Extent::Static(4)], name: Some("mask".into()) },
/// );
/// let out = dropout(&mut program, DType::Float32, x, mask, 1, 0.5);
///
/// let x_values = [1.0f32, 2.0, 3.0, 4.0];
/// let mask_values = [1.0f32, 0.0, 1.0, 0.0];
/// let evaluated = proxima_tensor::cpu::evaluate(&program, &[], &[&x_values, &mask_values], &[out])
///     .expect("dropout program lowers and evaluates");
/// assert_eq!(evaluated.root(), &[2.0, 0.0, 6.0, 0.0]);
/// ```
#[must_use]
pub fn dropout(program: &mut Vec<Op>, dtype: DType, x: NodeId, mask: NodeId, rank: u16, keep_prob: f32) -> NodeId {
    let full = expr::identity(rank);
    let masked = expr::binary(program, dtype, ScalarOp::Multiply, (x, full.clone()), (mask, full.clone()));
    let inverse_keep_prob = expr::constant(program, dtype, 1.0 / keep_prob);
    expr::binary(program, dtype, ScalarOp::Multiply, (masked, full), (inverse_keep_prob, expr::broadcast(rank)))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec;

    use proxima_tensor::op::Extent;

    use super::*;

    fn leaf(program: &mut Vec<Op>, name: &str, shape: Vec<Extent>) -> NodeId {
        proxima_tensor::op::append(program, Op::Input { dtype: DType::Float32, shape, name: Some(name.into()) })
    }

    #[proxima::test]
    async fn dropout_zeroes_masked_positions_and_inverse_scales_the_rest() {
        let mut program = Vec::new();
        let x = leaf(&mut program, "x", vec![Extent::Static(4)]);
        let mask = leaf(&mut program, "mask", vec![Extent::Static(4)]);
        let out = dropout(&mut program, DType::Float32, x, mask, 1, 0.5);

        let x_values = [1.0f32, 2.0, 3.0, 4.0];
        let mask_values = [1.0f32, 0.0, 1.0, 0.0];
        let evaluated = proxima_tensor::cpu::evaluate_named(&program, &[], &[("x", &x_values), ("mask", &mask_values)], &[out])
            .expect("dropout program lowers and evaluates");
        assert_eq!(evaluated.get(out).expect("dropout output present").0, &[2.0, 0.0, 6.0, 0.0]);
    }

    #[proxima::test]
    async fn dropout_at_keep_prob_one_with_an_all_ones_mask_is_the_identity() {
        let mut program = Vec::new();
        let x = leaf(&mut program, "x", vec![Extent::Static(3)]);
        let mask = leaf(&mut program, "mask", vec![Extent::Static(3)]);
        let out = dropout(&mut program, DType::Float32, x, mask, 1, 1.0);

        let x_values = [1.0f32, -2.5, 3.0];
        let mask_values = [1.0f32, 1.0, 1.0];
        let evaluated = proxima_tensor::cpu::evaluate_named(&program, &[], &[("x", &x_values), ("mask", &mask_values)], &[out])
            .expect("dropout program lowers and evaluates");
        assert_eq!(evaluated.get(out).expect("dropout output present").0, &x_values);
    }

    #[proxima::test]
    async fn dropout_rejects_a_mask_shaped_differently_than_x() {
        let mut program = Vec::new();
        let x = leaf(&mut program, "x", vec![Extent::Static(4)]);
        let mask = leaf(&mut program, "mask", vec![Extent::Static(3)]);
        let out = dropout(&mut program, DType::Float32, x, mask, 1, 0.5);

        let x_values = [1.0f32, 2.0, 3.0, 4.0];
        let mask_values = [1.0f32, 0.0, 1.0];
        let result = proxima_tensor::cpu::evaluate_named(&program, &[], &[("x", &x_values), ("mask", &mask_values)], &[out]);
        assert!(result.is_err(), "a mask shaped differently than x must be a named error, not a silent broadcast or panic");
    }
}
