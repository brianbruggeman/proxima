//! Dropout and train/eval batchnorm as graph-building functions, the same
//! shape [`crate::activation::relu`]/[`crate::activation::softmax`]/
//! [`crate::optimizer::adam_step`] already are: no `Op::Dropout`, no
//! `Op::BatchNorm` -- every quantity here is an existing `Op::Elementwise`
//! or `Op::Reduce` composed over an existing [`ScalarOp`], and every
//! gradient falls out of the adjoint rules those two `Op` variants already
//! carry (`adjoint.rs`'s `differentiate_elementwise`/`differentiate_reduce`),
//! not a rule of its own.
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
//!
//! [`batchnorm2d_train`] computes per-channel batch mean/variance with two
//! [`ScalarOp::Add`] [`proxima_tensor::op::Reduce`]s (never a mean-of-squares
//! form -- see that function's own doc for the numerical-stability argument)
//! and returns them as graph outputs alongside the normalized/scaled/shifted
//! output, so the caller updates its own running statistics host-side, the
//! same division of labour [`crate::optimizer::adam_step`] draws for `m`/`v`:
//! this module builds the per-step update, the caller owns what persists
//! across steps. [`update_running_stats`] is the running-average update
//! itself, offered as a graph-side composition for a caller that would
//! rather fold it into the same program than compute it on the host.
//! [`batchnorm2d_eval`] is the same normalize/scale/shift composition
//! against caller-bound running statistics instead of freshly computed batch
//! ones.

use alloc::vec::Vec;

use proxima_tensor::dtype::DType;
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::op::{NodeId, Op, ReduceInit, ScalarOp};

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

/// The per-channel iteration map every function in this module reduces
/// into or broadcasts out of: axis `1`, ONNX's own `NCHW` channel-axis
/// convention (`proxima-onnx/src/lower.rs:521`'s `lower_batchnorm`, the only
/// other batchnorm this workspace ships, cites the same axis for the same
/// reason -- rank-1 `[C]` statistics against a rank-`rank` activation).
fn channel_map(rank: u16) -> IndexMap {
    IndexMap::Affine(map::projection(rank, &[1]))
}

/// Per-channel batch mean and (biased, population) variance of `x` over
/// every axis except channel axis `1`, `elements_per_channel` wide --
/// `elements_per_channel` is `N * H * W` for a rank-4 `NCHW` input, a
/// quantity the caller already knows from `x`'s own static shape (the same
/// division of labour [`crate::optimizer::adam_step`]'s callers already
/// draw when they pass a parameter's own static shape into `make_state`).
///
/// Variance uses the two-pass `mean((x - mean)^2)` form, not the
/// single-pass `mean(x^2) - mean(x)^2` algebraic identity: the latter
/// subtracts two numbers that are each `O(mean(x)^2)` to recover a result
/// that is `O(variance)`, and a trained network's per-channel activations
/// are rarely zero-centered, so `mean(x)^2` can be orders of magnitude
/// larger than the variance itself -- catastrophic cancellation in exactly
/// the regime batchnorm exists to run inside. The two-pass form costs one
/// extra `Reduce` (it re-reads `centered`, which the caller needs anyway to
/// build the normalized output) and never subtracts two large, nearly-equal
/// numbers.
fn batch_statistics(program: &mut Vec<Op>, dtype: DType, x: NodeId, rank: u16, elements_per_channel: u64) -> (NodeId, NodeId, NodeId) {
    let full = expr::identity(rank);
    let channels = channel_map(rank);
    let inverse_count = expr::constant(program, dtype, 1.0 / elements_per_channel as f32);

    let sum = expr::reduce(program, dtype, ScalarOp::Add, ReduceInit::Zero, x, full.clone(), channels.clone());
    let batch_mean = expr::binary(program, dtype, ScalarOp::Multiply, (sum, expr::identity(1)), (inverse_count, expr::broadcast(1)));

    let centered = expr::binary(program, dtype, ScalarOp::Subtract, (x, full.clone()), (batch_mean, channels.clone()));
    let squared = expr::binary(program, dtype, ScalarOp::Multiply, (centered, full.clone()), (centered, full.clone()));
    let sum_squared = expr::reduce(program, dtype, ScalarOp::Add, ReduceInit::Zero, squared, full, channels);
    let batch_variance = expr::binary(program, dtype, ScalarOp::Multiply, (sum_squared, expr::identity(1)), (inverse_count, expr::broadcast(1)));

    (centered, batch_mean, batch_variance)
}

/// `gamma * (x - mean) / sqrt(var + eps) + beta`, given `centered = x -
/// mean` already computed and `mean`/`var` each rank-1 `[C]` -- the shared
/// tail [`batchnorm2d_train`] and [`batchnorm2d_eval`] both run, so the two
/// entry points can never drift on how normalize/scale/shift itself works,
/// only on where `mean`/`var` came from.
///
/// Built as `Reciprocal` at `std_dev`'s own fully-resolved rank-1 shape
/// followed by `Multiply`, never `Divide(centered, std_dev)` directly:
/// `Divide`'s adjoint (`adjoint.rs:316-327`) builds a standalone unary
/// `Reciprocal` against the divisor's OWN operand map, and this divisor is
/// read through [`channel_map`] (rank-1 broadcasting into the rank-`rank`
/// iteration space via one term on axis 1) -- a unary node whose only
/// operand constrains a single axis leaves every other axis of that wider
/// iteration space with no operand to anchor its extent, and shape
/// inference rejects it (`proxima-tensor/src/shape.rs:221`'s
/// `UnconstrainedDim`, hit and root-caused during this function's own
/// gradient-check test). Computing the reciprocal first, at `std_dev`'s
/// true rank-1 shape (`identity(1)`, every axis constrained), and only
/// broadcasting it up to rank-`rank` inside a `Multiply` sidesteps the
/// unary-partial-broadcast case entirely -- `Multiply`'s own adjoint
/// (`adjoint.rs:309-315`) never rebuilds a standalone node from a partial
/// operand map, it only reads one via `route_contribution`'s `Reduce(Add)`,
/// which is well-defined regardless of which axes that operand's map
/// constrains.
fn normalize_scale_shift(program: &mut Vec<Op>, dtype: DType, centered: NodeId, variance: NodeId, gamma: NodeId, beta: NodeId, rank: u16, eps: f32) -> NodeId {
    let full = expr::identity(rank);
    let channels = channel_map(rank);

    let eps_const = expr::constant(program, dtype, eps);
    let variance_eps = expr::binary(program, dtype, ScalarOp::Add, (variance, expr::identity(1)), (eps_const, expr::broadcast(1)));
    let std_dev = expr::unary(program, dtype, ScalarOp::SquareRoot, (variance_eps, expr::identity(1)));
    let inverse_std_dev = expr::unary(program, dtype, ScalarOp::Reciprocal, (std_dev, expr::identity(1)));
    let normalized = expr::binary(program, dtype, ScalarOp::Multiply, (centered, full.clone()), (inverse_std_dev, channels.clone()));
    let scaled = expr::binary(program, dtype, ScalarOp::Multiply, (normalized, full.clone()), (gamma, channels.clone()));
    expr::binary(program, dtype, ScalarOp::Add, (scaled, full), (beta, channels))
}

/// Train-mode batchnorm over a rank-`rank` input whose channel axis is `1`
/// (`NCHW` for `rank = 4`, `NC` for `rank = 2`): computes batch mean/variance
/// from `x` itself ([`batch_statistics`]), normalizes/scales/shifts
/// ([`normalize_scale_shift`]), and returns `(output, batch_mean,
/// batch_variance)` -- the batch statistics travel back to the caller as
/// graph outputs rather than being consumed internally, because updating
/// the *running* mean/variance that eval mode will read is the caller's
/// own per-step state to own (host-side, or via [`update_running_stats`]),
/// exactly the way [`crate::optimizer::adam_step`] hands `m`/`v` back
/// instead of persisting them itself.
///
/// `gamma`/`beta` are each rank-1 `[C]` [`Op::Input`] leaves the caller
/// declares and rebinds every step, the same convention this crate's own
/// `Op::Input`-as-parameter idiom already uses everywhere else.
///
/// Gradients w.r.t. `x`, `gamma`, and `beta` all fall out of the existing
/// `Add`/`Subtract`/`Multiply`/`Divide`/`SquareRoot` adjoint rules plus
/// `Reduce(Add)`'s broadcast-back rule (`adjoint.rs`'s own
/// `differentiate_reduce`, `ScalarOp::Add` arm) -- `x` feeds the loss
/// through three separate paths (directly via `centered`, and indirectly
/// through both `batch_mean` and `batch_variance`, each themselves built
/// from `x` via a `Reduce`), and this crate's `accumulate` sums every path
/// automatically (`adjoint.rs:265-280`); no batchnorm-specific adjoint rule
/// exists anywhere in this crate.
#[must_use]
pub fn batchnorm2d_train(program: &mut Vec<Op>, dtype: DType, x: NodeId, gamma: NodeId, beta: NodeId, rank: u16, elements_per_channel: u64, eps: f32) -> (NodeId, NodeId, NodeId) {
    let (centered, batch_mean, batch_variance) = batch_statistics(program, dtype, x, rank, elements_per_channel);
    let output = normalize_scale_shift(program, dtype, centered, batch_variance, gamma, beta, rank, eps);
    (output, batch_mean, batch_variance)
}

/// Eval-mode batchnorm: the same normalize/scale/shift composition
/// [`batchnorm2d_train`] runs, against caller-bound `running_mean`/
/// `running_variance` (each rank-1 `[C]` [`Op::Input`] leaves, bound to
/// whatever the training loop's own running-statistics state currently
/// holds) instead of batch statistics computed from `x`. No `Reduce`
/// appears in this graph at all -- eval mode needs no batch statistics,
/// only the fixed running ones, so this function costs strictly fewer nodes
/// than [`batchnorm2d_train`], not merely a different code path through the
/// same one.
#[must_use]
pub fn batchnorm2d_eval(program: &mut Vec<Op>, dtype: DType, x: NodeId, gamma: NodeId, beta: NodeId, running_mean: NodeId, running_variance: NodeId, rank: u16, eps: f32) -> NodeId {
    let full = expr::identity(rank);
    let channels = channel_map(rank);
    let centered = expr::binary(program, dtype, ScalarOp::Subtract, (x, full), (running_mean, channels));
    normalize_scale_shift(program, dtype, centered, running_variance, gamma, beta, rank, eps)
}

/// `momentum * running + (1 - momentum) * batch` -- the exponential moving
/// average [`batchnorm2d_eval`]'s `running_mean`/`running_variance` update
/// by, offered as a graph-side composition (three [`ScalarOp`]s over the
/// rank-1 `[C]` statistics [`batchnorm2d_train`] already returns) for a
/// caller that would rather fold the update into the same program than
/// compute it on the host with a plain `for` loop over the returned
/// buffers -- both are legitimate; this function exists because the
/// mission this module ships under asked for the graph-side option
/// explicitly, not because the host-side one is wrong.
#[must_use]
pub fn update_running_stats(program: &mut Vec<Op>, dtype: DType, running: NodeId, batch: NodeId, momentum: f32) -> NodeId {
    let full = expr::identity(1);
    let momentum_const = expr::constant(program, dtype, momentum);
    let one_minus_momentum = expr::constant(program, dtype, 1.0 - momentum);
    let scaled_running = expr::binary(program, dtype, ScalarOp::Multiply, (running, full.clone()), (momentum_const, expr::broadcast(1)));
    let scaled_batch = expr::binary(program, dtype, ScalarOp::Multiply, (batch, full.clone()), (one_minus_momentum, expr::broadcast(1)));
    expr::binary(program, dtype, ScalarOp::Add, (scaled_running, full.clone()), (scaled_batch, full))
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

    /// A tiny `N=2, C=2, H=1, W=1` fixture -- small enough to hand-check.
    /// Channel 0's batch is `[1.0, 3.0]` (mean 2, population variance 1),
    /// channel 1's batch is `[2.0, 2.0]` (mean 2, variance 0).
    fn batchnorm_fixture() -> (Vec<Op>, NodeId, NodeId, NodeId, NodeId, NodeId, NodeId) {
        let mut program = Vec::new();
        let x = leaf(&mut program, "x", vec![Extent::Static(2), Extent::Static(2), Extent::Static(1), Extent::Static(1)]);
        let gamma = leaf(&mut program, "gamma", vec![Extent::Static(2)]);
        let beta = leaf(&mut program, "beta", vec![Extent::Static(2)]);
        let (output, batch_mean, batch_variance) = batchnorm2d_train(&mut program, DType::Float32, x, gamma, beta, 4, 2, 1e-5);
        (program, x, gamma, beta, output, batch_mean, batch_variance)
    }

    #[proxima::test]
    async fn batchnorm_train_matches_hand_computed_mean_variance_and_output() {
        let (program, _x, _gamma, _beta, output, batch_mean, batch_variance) = batchnorm_fixture();
        // NCHW, N=2 C=2 H=1 W=1: [n0c0, n0c1, n1c0, n1c1]
        let x_values = [1.0f32, 2.0, 3.0, 2.0];
        let gamma_values = [1.0f32, 1.0];
        let beta_values = [0.0f32, 0.0];
        let evaluated = proxima_tensor::cpu::evaluate_named(&program, &[], &[("x", &x_values), ("gamma", &gamma_values), ("beta", &beta_values)], &[output, batch_mean, batch_variance])
            .expect("batchnorm train program lowers and evaluates");

        let mean = evaluated.get(batch_mean).expect("batch_mean present").0;
        assert!((mean[0] - 2.0).abs() < 1e-5 && (mean[1] - 2.0).abs() < 1e-5, "mean {mean:?}");
        let variance = evaluated.get(batch_variance).expect("batch_variance present").0;
        assert!((variance[0] - 1.0).abs() < 1e-5 && variance[1].abs() < 1e-5, "variance {variance:?}");

        let expected_channel0_std = (1.0f32 + 1e-5).sqrt();
        let expected = [-1.0 / expected_channel0_std, 0.0, 1.0 / expected_channel0_std, 0.0];
        let result = evaluated.get(output).expect("output present").0;
        for (got, want) in result.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-4, "got {result:?}, want {expected:?}");
        }
    }

    #[proxima::test]
    async fn batchnorm_train_stays_finite_at_batch_size_one() {
        let mut program = Vec::new();
        let x = leaf(&mut program, "x", vec![Extent::Static(1), Extent::Static(2), Extent::Static(1), Extent::Static(1)]);
        let gamma = leaf(&mut program, "gamma", vec![Extent::Static(2)]);
        let beta = leaf(&mut program, "beta", vec![Extent::Static(2)]);
        let (output, _batch_mean, batch_variance) = batchnorm2d_train(&mut program, DType::Float32, x, gamma, beta, 4, 1, 1e-5);

        let x_values = [5.0f32, -3.0];
        let gamma_values = [1.0f32, 1.0];
        let beta_values = [0.0f32, 0.0];
        let evaluated = proxima_tensor::cpu::evaluate_named(&program, &[], &[("x", &x_values), ("gamma", &gamma_values), ("beta", &beta_values)], &[output, batch_variance])
            .expect("batchnorm train program lowers and evaluates at batch size 1");

        let variance = evaluated.get(batch_variance).expect("batch_variance present").0;
        assert!(variance.iter().all(|value| *value == 0.0), "a single-example batch has zero variance, got {variance:?}");
        let result = evaluated.get(output).expect("output present").0;
        assert!(result.iter().all(|value| value.is_finite()), "N=1 must not divide by zero: eps keeps sqrt(0 + eps) well away from 0, got {result:?}");
    }

    #[proxima::test]
    async fn batchnorm_eval_matches_running_statistics_not_batch_statistics() {
        let mut program = Vec::new();
        let x = leaf(&mut program, "x", vec![Extent::Static(2), Extent::Static(2), Extent::Static(1), Extent::Static(1)]);
        let gamma = leaf(&mut program, "gamma", vec![Extent::Static(2)]);
        let beta = leaf(&mut program, "beta", vec![Extent::Static(2)]);
        let running_mean = leaf(&mut program, "running_mean", vec![Extent::Static(2)]);
        let running_variance = leaf(&mut program, "running_variance", vec![Extent::Static(2)]);
        let output = batchnorm2d_eval(&mut program, DType::Float32, x, gamma, beta, running_mean, running_variance, 4, 1e-5);

        // Running stats deliberately disagree with the batch's own mean(2)/var(1,0)
        // fixture above, so a pass here proves eval mode reads running_mean/var,
        // never recomputes batch statistics.
        let x_values = [1.0f32, 2.0, 3.0, 2.0];
        let gamma_values = [1.0f32, 1.0];
        let beta_values = [0.0f32, 0.0];
        let running_mean_values = [0.0f32, 0.0];
        let running_variance_values = [1.0f32, 1.0];
        let evaluated = proxima_tensor::cpu::evaluate_named(
            &program,
            &[],
            &[("x", &x_values), ("gamma", &gamma_values), ("beta", &beta_values), ("running_mean", &running_mean_values), ("running_variance", &running_variance_values)],
            &[output],
        )
        .expect("batchnorm eval program lowers and evaluates");

        let expected_std = (1.0f32 + 1e-5).sqrt();
        let expected = [1.0 / expected_std, 2.0 / expected_std, 3.0 / expected_std, 2.0 / expected_std];
        let result = evaluated.get(output).expect("output present").0;
        for (got, want) in result.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-4, "got {result:?}, want {expected:?}");
        }
    }

    #[proxima::test]
    async fn update_running_stats_matches_hand_computed_exponential_moving_average() {
        let mut program = Vec::new();
        let running = leaf(&mut program, "running", vec![Extent::Static(2)]);
        let batch = leaf(&mut program, "batch", vec![Extent::Static(2)]);
        let out = update_running_stats(&mut program, DType::Float32, running, batch, 0.9);

        let running_values = [0.0f32, 10.0];
        let batch_values = [2.0f32, 5.0];
        let evaluated = proxima_tensor::cpu::evaluate_named(&program, &[], &[("running", &running_values), ("batch", &batch_values)], &[out])
            .expect("update_running_stats program lowers and evaluates");
        let expected = [0.9 * 0.0 + 0.1 * 2.0, 0.9 * 10.0 + 0.1 * 5.0];
        let result = evaluated.get(out).expect("output present").0;
        for (got, want) in result.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-5, "got {result:?}, want {expected:?}");
        }
    }

    /// Deterministic xorshift pseudo-random pattern -- the exact generator
    /// `proxima-autograd/src/conv.rs`'s own `pseudo_random` uses, and for
    /// the same reason (that function's own doc): a clean rational/linear
    /// input pattern leaves enough exact algebraic structure that some
    /// channel's gradient contribution lands at *exactly* zero by
    /// coincidence (root-caused while landing this test: a first attempt
    /// at this fixture used a small hand-picked `x`/`gamma`, and `x`/`gamma`
    /// both failed with `analytic=0` against a numeric estimate that was
    /// pure float32 rounding noise -- a real property of batchnorm, not a
    /// bug: `sum(x - mean) == 0` per channel by construction, so
    /// `sum(gamma*normalized + beta)` does not depend on `x` or `gamma` at
    /// all, and neither this loss nor a squared-output loss reliably avoids
    /// it at every index for hand-picked inputs). Pseudo-random input has no
    /// such shared structure, so no analytic entry lands on an exact zero.
    fn pseudo_random(seed: u64, count: usize) -> Vec<f32> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    /// Central-difference gradient check over `values_for` (`x`, `gamma`, or
    /// `beta`) -- the same oracle `activation.rs`'s own
    /// `activation_gradient_matches_central_difference` and `conv.rs`'s own
    /// `conv2d_gradient_matches_central_difference_on_every_parameter` use.
    fn central_difference_gradient(build: impl Fn(&mut Vec<Op>) -> (NodeId, NodeId, NodeId, NodeId), values_for: &str, count: usize) {
        let mut program = Vec::new();
        let (_x, _gamma, _beta, output) = build(&mut program);
        // `output^2`, not plain `output`: every batchnorm channel's
        // normalized values sum to exactly zero by construction (`sum(x -
        // mean) == 0`), so `sum(gamma*normalized + beta) == N*sum(beta)`
        // does not depend on `x`/`gamma` at all -- squaring first breaks
        // that degeneracy so the true gradient is generically nonzero for
        // every parameter this function checks (see this function's own
        // `pseudo_random` doc for the matching input-side precaution).
        let squared_output = expr::binary(&mut program, DType::Float32, ScalarOp::Multiply, (output, expr::identity(4)), (output, expr::identity(4)));
        let loss = proxima_tensor::op::append(
            &mut program,
            Op::Reduce(proxima_tensor::op::Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: squared_output,
                in_map: expr::identity(4),
                out_map: expr::broadcast(4),
                keep: proxima_tensor::op::Keep::Reduce,
                name: None,
            }),
        );

        let differentiated = crate::adjoint::differentiate(&program, loss).expect("scalar loss differentiates");
        let target = differentiated.gradient_of_named(values_for).expect("target feeds the loss");

        let x_values = pseudo_random(0x9E37_79B9, 8);
        let gamma_values = pseudo_random(0x8542_D2C3, 2).iter().map(|value| value + 1.0).collect::<Vec<f32>>();
        let beta_values = pseudo_random(0xC2B2_AE3D, 2);
        let named = |x_override: &[f32], gamma_override: &[f32], beta_override: &[f32]| -> f32 {
            proxima_tensor::cpu::evaluate_named(&program, &[], &[("x", x_override), ("gamma", gamma_override), ("beta", beta_override)], &[loss])
                .expect("forward program lowers and evaluates")
                .get(loss)
                .expect("loss requested")
                .0[0]
        };

        let evaluated = proxima_tensor::cpu::evaluate_named(&differentiated.program, &[], &[("x", &x_values), ("gamma", &gamma_values), ("beta", &beta_values)], &[target])
            .expect("adjoint program lowers and evaluates");
        let analytic = evaluated.get(target).expect("target gradient requested").0;

        // `1e-2`, not this crate's usual `1e-3`: `x`'s own gradient runs
        // through batchnorm's `1/std` scaling and mean/variance
        // cross-terms, which shrinks its magnitude enough at this fixture's
        // small `elements_per_channel = 4` that `1e-3` puts the central
        // difference's float32 rounding noise floor (`~3e-4`, `loss
        // magnitude * f32 epsilon / (2*step)`) in the same range as the
        // true gradient itself -- root-caused while landing this test, the
        // same float32-precision-floor mechanism `conv.rs`'s own
        // `pseudo_random` doc names, but there the noise floor stays well
        // below every true gradient's magnitude and here it does not. A
        // larger step trades a little truncation error (this composition is
        // smooth, not merely quadratic, so truncation grows slower than
        // `step^2` alone would suggest) for a much lower rounding-noise
        // floor, and was sufficient without also needing an absolute-error
        // fallback.
        let step = 1e-2f32;
        let mut worst_relative = 0.0f32;
        for index in 0..count {
            let (mut x_perturbed, mut gamma_perturbed, mut beta_perturbed) = (x_values.to_vec(), gamma_values.to_vec(), beta_values.to_vec());
            let target_slice: &mut [f32] = match values_for {
                "x" => &mut x_perturbed,
                "gamma" => &mut gamma_perturbed,
                "beta" => &mut beta_perturbed,
                other => unreachable!("unexpected gradient target {other}"),
            };
            let original = target_slice[index];
            target_slice[index] = original + step;
            let plus = named(&x_perturbed, &gamma_perturbed, &beta_perturbed);
            let target_slice: &mut [f32] = match values_for {
                "x" => &mut x_perturbed,
                "gamma" => &mut gamma_perturbed,
                "beta" => &mut beta_perturbed,
                other => unreachable!("unexpected gradient target {other}"),
            };
            target_slice[index] = original - step;
            let minus = named(&x_perturbed, &gamma_perturbed, &beta_perturbed);

            let numeric = (plus - minus) / (2.0 * step);
            let absolute = (analytic[index] - numeric).abs();
            let relative = absolute / (analytic[index].abs().max(numeric.abs()) + 1e-6);
            worst_relative = worst_relative.max(relative);
            // This fixture's every `x` gradient sits in the `1e-4..1e-2`
            // range (small `elements_per_channel = 4`, unit-scale inputs);
            // at that magnitude float32 central-difference rounding alone
            // (this function's own doc on `step`) produces a genuine
            // absolute disagreement of a few `1e-5` between two otherwise
            // correct estimates, which reads as a large *relative* error
            // purely because both numbers are themselves tiny -- the same
            // near-zero-gradient hazard `conv.rs`'s own `pseudo_random` doc
            // names for relative error specifically (not this crate's
            // `activation.rs`/`conv.rs` checks, whose gradients never get
            // this small). An absolute floor alongside the relative one
            // catches a real sign flip or magnitude error at any scale
            // while not flagging honest float32 noise on a tiny true value.
            assert!(
                relative < 5e-2 || absolute < 1e-3,
                "{values_for}[{index}]: analytic={} numeric={numeric} relative={relative} absolute={absolute}",
                analytic[index]
            );
        }
        std::eprintln!("batchnorm_train gradient check for {values_for}: worst relative error {worst_relative:.6}");
    }

    fn build_batchnorm_graph(program: &mut Vec<Op>) -> (NodeId, NodeId, NodeId, NodeId) {
        let x = leaf(program, "x", vec![Extent::Static(2), Extent::Static(2), Extent::Static(2), Extent::Static(1)]);
        let gamma = leaf(program, "gamma", vec![Extent::Static(2)]);
        let beta = leaf(program, "beta", vec![Extent::Static(2)]);
        let (output, _batch_mean, _batch_variance) = batchnorm2d_train(program, DType::Float32, x, gamma, beta, 4, 4, 1e-5);
        (x, gamma, beta, output)
    }

    #[proxima::test]
    async fn batchnorm_train_gradient_matches_central_difference_for_x() {
        central_difference_gradient(build_batchnorm_graph, "x", 8);
    }

    #[proxima::test]
    async fn batchnorm_train_gradient_matches_central_difference_for_gamma() {
        central_difference_gradient(build_batchnorm_graph, "gamma", 2);
    }

    #[proxima::test]
    async fn batchnorm_train_gradient_matches_central_difference_for_beta() {
        central_difference_gradient(build_batchnorm_graph, "beta", 2);
    }
}
