//! Adam as an elementwise update expression over `(param, grad, m, v)` —
//! not an `Optimizer` trait, not a `step` method on anything.
//!
//! `m := b1*m + (1-b1)*g`, `v := b2*v + (1-b2)*g^2`,
//! `p := p - lr*m_hat/(sqrt(v_hat)+eps)` is nine `ScalarOp::{Add,Multiply,
//! Subtract,Reciprocal,SquareRoot}` nodes over existing
//! [`proxima_tensor::op::Op::Input`] leaves — every term genuinely
//! elementwise, so [`adam_step`] is a graph-building function, the same
//! shape [`crate::activation::relu`]/[`crate::activation::softmax`] are.
//! `m`/`v` are ordinary `Op::Input` leaves the caller creates and re-binds
//! by name across steps (`proxima-tensor/src/cpu.rs:413` `evaluate_named`),
//! exactly like `param`.
//!
//! [`AdamStep`] is `adam_step` wearing this workspace's uniform `Pipe`
//! shape (`In = AdamOperands`, `Out = (NodeId, NodeId, NodeId)`), for a
//! caller that wants to compose it with other pipes rather than call it
//! directly. Unlike [`crate::adjoint::Differentiate`] (zero-sized — a
//! transform runs once with nothing to remember between calls), an
//! `AdamStep` genuinely holds state across many calls: the same
//! `AdamConfig`, `rank`, and `step` node are reused for every parameter in
//! a program, and the program under construction grows with every call.
//! `Pipe::call` takes `&self`, so that growth needs interior mutability —
//! the same `RefCell` idiom
//! `proxima_tensor::shape::ShapeTable`'s own `Pipe` impl uses for the
//! identical reason (`proxima-tensor/src/shape.rs:69-73`, citing
//! `proxima_primitives::pipe::isolate`'s `Rc`/`RefCell`/`!Send`
//! per-thread-state pattern, `proxima-primitives/src/pipe/isolate.rs:34-36`).
//!
//! Bias correction needs `beta1^step`/`beta2^step`. `beta1`/`beta2` are
//! compile-time [`AdamConfig`] values, so `ln(beta1)`/`ln(beta2)` are
//! computed once in Rust at graph-construction time and folded in as
//! [`proxima_tensor::op::Op::Constant`]s; `beta1^step = exp(step *
//! ln(beta1))` needs no graph-level `pow` (this crate's closed `ScalarOp`
//! set has none — `op.rs:60-77`). `step` itself is an `Op::Input`, not a
//! `Constant`: the compiled program is built once and evaluated once per
//! training step (this module's own doc, "once-per-program and cold"), so
//! the one value that changes on every call must be a runtime binding, not
//! a literal baked into the graph.

use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::convert::Infallible;
use core::future::Future;

use proxima_primitives::pipe::Pipe;
use proxima_tensor::dtype::DType;
use proxima_tensor::op::{NodeId, Op, ScalarOp};

use crate::expr;

/// Adam's four tunables (Kingma & Ba 2014). Config-as-data and a fluent
/// builder both first-class (guiding principle 4): under the `config`
/// feature this derives the same three-crate stack
/// `proxima-tensor::spec::ProgramSpec` already uses — `bon::Builder` for
/// the fluent surface, `conflaguration::Settings` for env/file-layered
/// config, `serde` for the wire form — rather than a bespoke mechanism.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "config",
    derive(bon::Builder, serde::Serialize, serde::Deserialize, conflaguration::Settings)
)]
#[cfg_attr(feature = "config", settings(prefix = "AUTOGRAD_ADAM"))]
#[cfg_attr(feature = "config", builder(derive(Clone, Debug)))]
pub struct AdamConfig {
    /// Step size. Scales the bias-corrected first-moment estimate before
    /// it is subtracted from the parameter.
    pub learning_rate: f32,
    /// First-moment (mean) decay. Closer to 1 remembers more history.
    pub beta1: f32,
    /// Second-moment (uncentered variance) decay. Closer to 1 remembers
    /// more history and damps the per-parameter step size more slowly.
    pub beta2: f32,
    /// Added to the second-moment square root before dividing, so a
    /// parameter with near-zero gradient history never divides by zero.
    pub epsilon: f32,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

/// A rank-0 `Op::Input` bound by name every step — the one Adam quantity
/// that cannot be a graph-time `Op::Constant` (this module's own doc).
/// Build once, reuse the same [`NodeId`] across every [`adam_step`] call
/// for every parameter in a program.
#[must_use]
pub fn step_input(program: &mut Vec<Op>, name: &str) -> NodeId {
    proxima_tensor::op::append(
        program,
        Op::Input {
            dtype: DType::Float32,
            shape: Vec::new(),
            name: Some(String::from(name)),
        },
    )
}

/// The four tensors one Adam step touches, grouped because they travel
/// together — the same justification [`proxima_tensor::op::Reduce`]'s own
/// doc gives for bundling its own eight fields
/// (`proxima-tensor/src/op.rs:149-151`: "named because eight of them
/// travel together"). This is the `(param, grad, m, v)` input `adam_step`
/// composes into `(param, m, v)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdamOperands {
    pub param: NodeId,
    pub grad: NodeId,
    pub m: NodeId,
    pub v: NodeId,
}

/// Appends one Adam update for a single `rank`-dimensional parameter,
/// returning `(new_param, new_m, new_v)` — the three quantities that
/// travel together into next step's bound inputs. A plain tuple, not a
/// struct: nothing a caller does with three named fields it cannot already
/// do by destructuring `let (p, m, v) = ...` at the call site (see this
/// crate's root doc for why a wrapper earns its keep only when it changes
/// what a caller can do, not just how it reads).
#[must_use]
pub fn adam_step(
    program: &mut Vec<Op>,
    config: &AdamConfig,
    rank: u16,
    operands: AdamOperands,
    step: NodeId,
) -> (NodeId, NodeId, NodeId) {
    let AdamOperands { param, grad, m: m_prev, v: v_prev } = operands;
    let full = expr::identity(rank);
    let scalar = expr::broadcast(rank);
    let dtype = DType::Float32;

    let beta1 = expr::constant(program, dtype, config.beta1);
    let one_minus_beta1 = expr::constant(program, dtype, 1.0 - config.beta1);
    let beta2 = expr::constant(program, dtype, config.beta2);
    let one_minus_beta2 = expr::constant(program, dtype, 1.0 - config.beta2);

    let m_scaled = expr::binary(program, dtype, ScalarOp::Multiply, (beta1, scalar.clone()), (m_prev, full.clone()));
    let grad_scaled = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (one_minus_beta1, scalar.clone()),
        (grad, full.clone()),
    );
    let m_new = expr::binary(program, dtype, ScalarOp::Add, (m_scaled, full.clone()), (grad_scaled, full.clone()));

    let grad_sq = expr::binary(program, dtype, ScalarOp::Multiply, (grad, full.clone()), (grad, full.clone()));
    let v_scaled = expr::binary(program, dtype, ScalarOp::Multiply, (beta2, scalar.clone()), (v_prev, full.clone()));
    let grad_sq_scaled = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (one_minus_beta2, scalar.clone()),
        (grad_sq, full.clone()),
    );
    let v_new = expr::binary(program, dtype, ScalarOp::Add, (v_scaled, full.clone()), (grad_sq_scaled, full.clone()));

    let bias1_denominator = bias_correction(program, step, libm::logf(config.beta1));
    let bias2_denominator = bias_correction(program, step, libm::logf(config.beta2));
    let zeroth = expr::identity(0);

    let recip_bias1 = expr::unary(program, dtype, ScalarOp::Reciprocal, (bias1_denominator, zeroth.clone()));
    let m_hat = expr::binary(program, dtype, ScalarOp::Multiply, (m_new, full.clone()), (recip_bias1, scalar.clone()));

    let recip_bias2 = expr::unary(program, dtype, ScalarOp::Reciprocal, (bias2_denominator, zeroth));
    let v_hat = expr::binary(program, dtype, ScalarOp::Multiply, (v_new, full.clone()), (recip_bias2, scalar.clone()));

    let sqrt_v_hat = expr::unary(program, dtype, ScalarOp::SquareRoot, (v_hat, full.clone()));
    let epsilon = expr::constant(program, dtype, config.epsilon);
    let denominator = expr::binary(program, dtype, ScalarOp::Add, (sqrt_v_hat, full.clone()), (epsilon, scalar.clone()));
    let recip_denominator = expr::unary(program, dtype, ScalarOp::Reciprocal, (denominator, full.clone()));
    let update = expr::binary(program, dtype, ScalarOp::Multiply, (m_hat, full.clone()), (recip_denominator, full.clone()));

    let learning_rate = expr::constant(program, dtype, config.learning_rate);
    let scaled_update = expr::binary(program, dtype, ScalarOp::Multiply, (learning_rate, scalar), (update, full.clone()));
    let new_param = expr::binary(program, dtype, ScalarOp::Subtract, (param, full.clone()), (scaled_update, full));

    (new_param, m_new, v_new)
}

/// `1 - beta^step`, rank 0. `beta^step = exp(step * ln(beta))`; `ln(beta)`
/// is folded in as a host-computed [`proxima_tensor::op::Op::Constant`]
/// since `beta` is a compile-time [`AdamConfig`] field, never a graph
/// value.
fn bias_correction(program: &mut Vec<Op>, step: NodeId, ln_beta: f32) -> NodeId {
    let dtype = DType::Float32;
    let zeroth = expr::identity(0);
    let ln_beta_constant = expr::constant(program, dtype, ln_beta);
    let exponent = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (step, zeroth.clone()),
        (ln_beta_constant, zeroth.clone()),
    );
    let beta_pow_step = expr::unary(program, dtype, ScalarOp::Exponential, (exponent, zeroth.clone()));
    let one = expr::constant(program, dtype, 1.0);
    expr::binary(program, dtype, ScalarOp::Subtract, (one, zeroth.clone()), (beta_pow_step, zeroth))
}

/// [`adam_step`] wearing this workspace's uniform `Pipe` shape — see this
/// module's own doc for why it needs a `RefCell` and [`Differentiate`]
/// (crate::adjoint) does not.
///
/// `In = AdamOperands`, `Out = (new_param, new_m, new_v)`. Construct once
/// per `(config, rank, step)` combination and `.call(operands)` once per
/// parameter that shares them — the same parameter loop the free function
/// already asked a caller to write, now composable as a `Pipe` chain.
pub struct AdamStep {
    program: RefCell<Vec<Op>>,
    config: AdamConfig,
    rank: u16,
    step: NodeId,
}

impl AdamStep {
    /// `program` is the shared graph every call appends to; `step` must
    /// already be an `Op::Input` in it (see [`step_input`]).
    #[must_use]
    pub fn new(program: Vec<Op>, config: AdamConfig, rank: u16, step: NodeId) -> Self {
        Self { program: RefCell::new(program), config, rank, step }
    }

    /// Hands the accumulated program back to the caller once every
    /// parameter in it has been stepped — the same "read the state back
    /// out" shape `proxima_tensor::shape::ShapeTable::finish` uses
    /// (`proxima-tensor/src/shape.rs:111-113`).
    #[must_use]
    pub fn finish(self) -> Vec<Op> {
        self.program.into_inner()
    }
}

impl Pipe for AdamStep {
    type In = AdamOperands;
    type Out = (NodeId, NodeId, NodeId);
    /// `adam_step` cannot fail — every input is an existing `NodeId` and
    /// every output is a fresh `op::append`, so there is nothing to name.
    type Err = Infallible;

    fn call(&self, operands: Self::In) -> impl Future<Output = Result<Self::Out, Infallible>> {
        async move {
            let mut program = self.program.borrow_mut();
            Ok(adam_step(&mut program, &self.config, self.rank, operands, self.step))
        }
    }
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
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(extent)],
                name: Some(String::from(name)),
            },
        )
    }

    #[proxima::test]
    async fn adam_moves_the_parameter_toward_the_gradients_sign() {
        let mut program = Vec::new();
        let param = leaf(&mut program, "param", 2);
        let grad = leaf(&mut program, "grad", 2);
        let m_prev = leaf(&mut program, "m", 2);
        let v_prev = leaf(&mut program, "v", 2);
        let step = step_input(&mut program, "step");
        let config = AdamConfig::default();

        let (new_param, new_m, new_v) =
            adam_step(&mut program, &config, 1, AdamOperands { param, grad, m: m_prev, v: v_prev }, step);

        let param_values = [1.0f32, -1.0];
        let grad_values = [1.0f32, -1.0];
        let zero = [0.0f32, 0.0];
        let one_step = [1.0f32];
        let evaluated = proxima_tensor::cpu::evaluate_named(
            &program,
            &[],
            &[
                ("param", &param_values),
                ("grad", &grad_values),
                ("m", &zero),
                ("v", &zero),
                ("step", &one_step),
            ],
            &[new_param, new_m, new_v],
        )
        .expect("adam program lowers and evaluates");

        let updated_param = evaluated.get(new_param).expect("new_param requested").0;
        assert!(
            updated_param[0] < param_values[0],
            "positive gradient must decrease the parameter, got {updated_param:?}"
        );
        assert!(
            updated_param[1] > param_values[1],
            "negative gradient must increase the parameter, got {updated_param:?}"
        );

        let updated_m = evaluated.get(new_m).expect("new_m requested").0;
        assert!((updated_m[0] - 0.1).abs() < 1e-6, "m = (1-b1)*g, got {updated_m:?}");
    }


    /// Same reasoning as `adjoint::pipe_tests::block_on_once`: `AdamStep`'s
    /// `RefCell` makes its `Pipe::call` future `!Send` by design (base
    /// `Pipe` has no `Send` bound), so this polls it directly rather than
    /// going through `#[proxima::test]`'s `Send`-bound harness.
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
        let param = leaf(&mut program, "param", 2);
        let grad = leaf(&mut program, "grad", 2);
        let m_prev = leaf(&mut program, "m", 2);
        let v_prev = leaf(&mut program, "v", 2);
        let step = step_input(&mut program, "step");
        let config = AdamConfig::default();
        let operands = AdamOperands { param, grad, m: m_prev, v: v_prev };

        let mut via_function_program = program.clone();
        let via_function = adam_step(&mut via_function_program, &config, 1, operands, step);

        let pipe = AdamStep::new(program, config, 1, step);
        let via_pipe = block_on_once(pipe.call(operands)).expect("adam_step never fails");
        let via_pipe_program = pipe.finish();

        assert_eq!(via_pipe, via_function, "the Pipe wrapper must append the exact same nodes");
        assert_eq!(
            via_pipe_program, via_function_program,
            "the Pipe wrapper must grow the shared program identically to the free function"
        );
    }
}
