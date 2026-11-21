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
use core::future::Future;
#[cfg(feature = "config")]
use std::path::Path;

use proxima_numeric::{DType, FloatLayout, NumericError, Overflow, Rounding, narrow_to_layout};
use proxima_primitives::pipe::Pipe;
use proxima_tensor::op::{NodeId, Op, ScalarOp};

use crate::expr;

/// A hyper-parameter as the host holds it: `f64`, the widest scalar the host has, so no layout
/// the scalar codec runs (at most 32 bits) loses anything before the config's own
/// [`FloatLayout`] is applied. The value reaches the graph only through [`narrow_hyper`], which
/// goes through `proxima-numeric`'s codec, so a bf16 or f16 hyper-parameter is a config change
/// and not a type change. Precedents for a layout carried as a config value beside plain
/// numbers: `omega/src/config.rs:152` (`DenseMmaInputLayout` in `RuntimeConfig`) and
/// `proxima-numeric/src/flow_match.rs:11` (`EmpiricalMu`'s compute layout).
type Hyper = f64;

/// A layout the optimizer refused, naming the config field that carried it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum OptimizerError {
    /// The layout is wider than, or outside the domain of, the 32-bit scalar codec.
    #[error("{field}: layout {layout:?} is not runnable by the scalar codec: {source}")]
    UnsupportedLayout {
        field: &'static str,
        layout: FloatLayout,
        #[source]
        source: NumericError,
    },
    /// The layout has no exponent field, so it is an integer layout, not a float one.
    #[error("{field}: layout {layout:?} is not a float layout")]
    NotFloat {
        field: &'static str,
        layout: FloatLayout,
    },
}

/// Narrows `value` into `layout` through the scalar codec, naming `field` on refusal.
fn narrow_into(
    value: f32,
    layout: FloatLayout,
    field: &'static str,
) -> Result<f32, OptimizerError> {
    if !layout.is_float() {
        return Err(OptimizerError::NotFloat { field, layout });
    }
    narrow_to_layout(value, layout, Rounding::NearestTiesEven, Overflow::Format).map_err(
        |source| OptimizerError::UnsupportedLayout {
            field,
            layout,
            source,
        },
    )
}

/// Narrows one hyper-parameter into `layout` and returns it in the codec's `f32` pivot.
///
/// # Errors
/// [`OptimizerError`] when `layout` is not a float layout the scalar codec runs; there is no
/// fallback to the unnarrowed value, so an unvalidated config fails here rather than computing
/// with a format it did not ask for.
fn narrow_hyper(value: Hyper, layout: FloatLayout) -> Result<f32, OptimizerError> {
    narrow_into(value as f32, layout, "hyper_layout")
}

/// `1 - value` in the pivot, narrowed again: the complement is a graph constant of its own, so it
/// is rounded into the layout rather than derived from the rounded value's neighbours.
fn narrow_complement(value: f32, layout: FloatLayout) -> Result<f32, OptimizerError> {
    narrow_hyper(f64::from(1.0 - value), layout)
}

/// Rounds optimizer moment state through `layout` at the host boundary: each value is encoded to
/// the layout's code and decoded back, the same codec path every `proxima-numeric` narrowing
/// takes. Call it on a step's `new_m` / `new_v` read-back before re-binding them, so the state
/// the next step sees is exactly what a `layout`-wide store would hold. `field` names the config
/// field the layout came from, for the error.
///
/// # Errors
/// [`OptimizerError`] when `layout` is not a float layout the scalar codec runs.
pub fn store_moment(
    values: &mut [f32],
    layout: FloatLayout,
    field: &'static str,
) -> Result<(), OptimizerError> {
    narrow_into(0.0, layout, field)?;
    for value in values {
        *value = narrow_into(*value, layout, field)?;
    }
    Ok(())
}

/// The float32 default of every layout field: binary32 is what the graph computes in, so it is
/// the identity narrowing and keeps a default config bit-identical to the pre-config code.
const DEFAULT_LAYOUT: FloatLayout = FloatLayout::BF32;

#[cfg(feature = "config")]
mod layout_name {
    //! `FloatLayout` as its canonical lowercase name on the wire (`"bfloat16"`), for env, TOML and
    //! serde alike; the structural form of an unnamed layout is not a config spelling.
    use alloc::string::{String, ToString};

    use proxima_numeric::FloatLayout;
    use serde::{Deserialize, Deserializer, Serializer, de, ser};

    pub(super) fn serialize<S: Serializer>(layout: &FloatLayout, out: S) -> Result<S::Ok, S::Error> {
        let name = layout
            .name()
            .ok_or_else(|| ser::Error::custom("layout has no canonical name"))?;
        out.serialize_str(name)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(input: D) -> Result<FloatLayout, D::Error> {
        let name = String::deserialize(input)?;
        FloatLayout::from_name(&name).ok_or_else(|| de::Error::custom(unknown(&name)))
    }

    pub(super) fn unknown(name: &str) -> String {
        alloc::format!("unknown float layout '{name}'; expected a canonical layout name such as float32|bfloat16|float16").to_string()
    }
}

/// `resolve_with` target for the env spelling of a layout field.
#[cfg(feature = "config")]
fn parse_layout(value: &str) -> Result<FloatLayout, LayoutParseError> {
    FloatLayout::from_name(value).ok_or_else(|| LayoutParseError(String::from(value)))
}

/// A layout name no [`FloatLayout::from_name`] row spells.
#[cfg(feature = "config")]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{}", layout_name::unknown(.0))]
pub struct LayoutParseError(String);

#[cfg(feature = "config")]
fn check_layout(
    errors: &mut Vec<conflaguration::ValidationMessage>,
    field: &'static str,
    layout: FloatLayout,
) {
    if narrow_into(0.0, layout, field).is_err() {
        errors.push(conflaguration::ValidationMessage::new(
            field,
            "must be a float layout the 32-bit scalar codec can run",
        ));
    }
}

#[cfg(feature = "config")]
fn check_decay(errors: &mut Vec<conflaguration::ValidationMessage>, field: &'static str, value: Hyper) {
    if !(0.0..1.0).contains(&value) {
        errors.push(conflaguration::ValidationMessage::new(field, "must be in [0, 1)"));
    }
}

#[cfg(feature = "config")]
fn check_positive(errors: &mut Vec<conflaguration::ValidationMessage>, field: &'static str, value: Hyper) {
    if !(value.is_finite() && value > 0.0) {
        errors.push(conflaguration::ValidationMessage::new(field, "must be finite and greater than zero"));
    }
}

#[cfg(feature = "config")]
fn finish(errors: Vec<conflaguration::ValidationMessage>) -> conflaguration::Result<()> {
    if errors.is_empty() {
        Ok(())
    } else {
        Err(conflaguration::Error::Validation { errors })
    }
}

/// Adam's four tunables (Kingma & Ba 2014). Config-as-data and a fluent
/// builder both first-class (guiding principle 4): under the `config`
/// feature this derives the same three-crate stack
/// `proxima-tensor::spec::ProgramSpec` already uses — `bon::Builder` for
/// the fluent surface, `conflaguration::Settings` for env/file-layered
/// config, `serde` for the wire form — rather than a bespoke mechanism.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "config",
    derive(
        bon::Builder,
        serde::Serialize,
        serde::Deserialize,
        conflaguration::Settings
    )
)]
#[cfg_attr(feature = "config", settings(prefix = "AUTOGRAD_ADAM"))]
#[cfg_attr(feature = "config", builder(derive(Clone, Debug)))]
#[cfg_attr(feature = "config", serde(default, deny_unknown_fields))]
pub struct AdamConfig {
    /// Step size. Scales the bias-corrected first-moment estimate before
    /// it is subtracted from the parameter.
    #[cfg_attr(feature = "config", setting(default = 0.001))]
    #[cfg_attr(feature = "config", builder(default = 0.001))]
    pub learning_rate: Hyper,
    /// First-moment (mean) decay. Closer to 1 remembers more history.
    #[cfg_attr(feature = "config", setting(default = 0.9))]
    #[cfg_attr(feature = "config", builder(default = 0.9))]
    pub beta1: Hyper,
    /// Second-moment (uncentered variance) decay. Closer to 1 remembers
    /// more history and damps the per-parameter step size more slowly.
    #[cfg_attr(feature = "config", setting(default = 0.999))]
    #[cfg_attr(feature = "config", builder(default = 0.999))]
    pub beta2: Hyper,
    /// Added to the second-moment square root before dividing, so a
    /// parameter with near-zero gradient history never divides by zero.
    #[cfg_attr(feature = "config", setting(default = 1e-8))]
    #[cfg_attr(feature = "config", builder(default = 1e-8))]
    pub epsilon: Hyper,
    /// Layout the hyper-parameters above are rounded into before they become graph constants.
    #[cfg_attr(feature = "config", setting(default_str = "float32", resolve_with = "parse_layout"))]
    #[cfg_attr(feature = "config", builder(default = DEFAULT_LAYOUT))]
    #[cfg_attr(feature = "config", serde(with = "layout_name"))]
    pub hyper_layout: FloatLayout,
    /// Storage layout of the first moment `m` between steps.
    #[cfg_attr(feature = "config", setting(default_str = "float32", resolve_with = "parse_layout"))]
    #[cfg_attr(feature = "config", builder(default = DEFAULT_LAYOUT))]
    #[cfg_attr(feature = "config", serde(with = "layout_name"))]
    pub first_moment_layout: FloatLayout,
    /// Storage layout of the second moment `v` between steps.
    #[cfg_attr(feature = "config", setting(default_str = "float32", resolve_with = "parse_layout"))]
    #[cfg_attr(feature = "config", builder(default = DEFAULT_LAYOUT))]
    #[cfg_attr(feature = "config", serde(with = "layout_name"))]
    pub second_moment_layout: FloatLayout,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            hyper_layout: DEFAULT_LAYOUT,
            first_moment_layout: DEFAULT_LAYOUT,
            second_moment_layout: DEFAULT_LAYOUT,
        }
    }
}

impl AdamConfig {
    /// Load a TOML/JSON/YAML file and validate the resolved config, the same
    /// source layering `omega::config::RuntimeConfig::from_file` provides.
    ///
    /// # Errors
    /// [`conflaguration::Error`] when the file does not parse, and
    /// [`conflaguration::Error::Validation`] naming the field when [`Self`]'s
    /// `Validate` rejects a value.
    #[cfg(feature = "config")]
    pub fn from_file(path: impl AsRef<Path>) -> conflaguration::Result<Self> {
        let config: Self = conflaguration::from_file(path.as_ref())?;
        conflaguration::Validate::validate(&config)?;
        Ok(config)
    }

    /// Rounds a step's `new_m` read-back through [`Self::first_moment_layout`] ([`store_moment`]).
    ///
    /// # Errors
    /// [`OptimizerError`] for a layout the scalar codec cannot run.
    pub fn store_first_moment(&self, values: &mut [f32]) -> Result<(), OptimizerError> {
        store_moment(values, self.first_moment_layout, "first_moment_layout")
    }

    /// Rounds a step's `new_v` read-back through [`Self::second_moment_layout`] ([`store_moment`]).
    ///
    /// # Errors
    /// [`OptimizerError`] for a layout the scalar codec cannot run.
    pub fn store_second_moment(&self, values: &mut [f32]) -> Result<(), OptimizerError> {
        store_moment(values, self.second_moment_layout, "second_moment_layout")
    }
}

#[cfg(feature = "config")]
impl conflaguration::Validate for AdamConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        check_positive(&mut errors, "learning_rate", self.learning_rate);
        check_positive(&mut errors, "epsilon", self.epsilon);
        check_decay(&mut errors, "beta1", self.beta1);
        check_decay(&mut errors, "beta2", self.beta2);
        check_layout(&mut errors, "hyper_layout", self.hyper_layout);
        check_layout(&mut errors, "first_moment_layout", self.first_moment_layout);
        check_layout(&mut errors, "second_moment_layout", self.second_moment_layout);
        finish(errors)
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
            dtype: DType::FLOAT32,
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
pub fn adam_step(
    program: &mut Vec<Op>,
    config: &AdamConfig,
    rank: u16,
    operands: AdamOperands,
    step: NodeId,
) -> Result<(NodeId, NodeId, NodeId), OptimizerError> {
    let AdamOperands {
        param,
        grad,
        m: m_prev,
        v: v_prev,
    } = operands;
    let full = expr::identity(rank);
    let scalar = expr::broadcast(rank);
    let dtype = DType::FLOAT32;

    let layout = config.hyper_layout;
    let beta1_value = narrow_hyper(config.beta1, layout)?;
    let beta2_value = narrow_hyper(config.beta2, layout)?;
    let beta1 = expr::constant(program, dtype, beta1_value);
    let one_minus_beta1 =
        expr::constant(program, dtype, narrow_complement(beta1_value, layout)?);
    let beta2 = expr::constant(program, dtype, beta2_value);
    let one_minus_beta2 =
        expr::constant(program, dtype, narrow_complement(beta2_value, layout)?);

    let m_scaled = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (beta1, scalar.clone()),
        (m_prev, full.clone()),
    );
    let grad_scaled = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (one_minus_beta1, scalar.clone()),
        (grad, full.clone()),
    );
    let m_new = expr::binary(
        program,
        dtype,
        ScalarOp::Add,
        (m_scaled, full.clone()),
        (grad_scaled, full.clone()),
    );

    let grad_sq = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (grad, full.clone()),
        (grad, full.clone()),
    );
    let v_scaled = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (beta2, scalar.clone()),
        (v_prev, full.clone()),
    );
    let grad_sq_scaled = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (one_minus_beta2, scalar.clone()),
        (grad_sq, full.clone()),
    );
    let v_new = expr::binary(
        program,
        dtype,
        ScalarOp::Add,
        (v_scaled, full.clone()),
        (grad_sq_scaled, full.clone()),
    );

    let bias1_denominator = bias_correction(program, step, libm::logf(beta1_value));
    let bias2_denominator = bias_correction(program, step, libm::logf(beta2_value));
    let zeroth = expr::identity(0);

    let recip_bias1 = expr::unary(
        program,
        dtype,
        ScalarOp::Reciprocal,
        (bias1_denominator, zeroth.clone()),
    );
    let m_hat = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (m_new, full.clone()),
        (recip_bias1, scalar.clone()),
    );

    let recip_bias2 = expr::unary(
        program,
        dtype,
        ScalarOp::Reciprocal,
        (bias2_denominator, zeroth),
    );
    let v_hat = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (v_new, full.clone()),
        (recip_bias2, scalar.clone()),
    );

    let sqrt_v_hat = expr::unary(program, dtype, ScalarOp::SquareRoot, (v_hat, full.clone()));
    let epsilon = expr::constant(
        program,
        dtype,
        narrow_hyper(config.epsilon, config.hyper_layout)?,
    );
    let denominator = expr::binary(
        program,
        dtype,
        ScalarOp::Add,
        (sqrt_v_hat, full.clone()),
        (epsilon, scalar.clone()),
    );
    let recip_denominator = expr::unary(
        program,
        dtype,
        ScalarOp::Reciprocal,
        (denominator, full.clone()),
    );
    let update = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (m_hat, full.clone()),
        (recip_denominator, full.clone()),
    );

    let learning_rate = expr::constant(
        program,
        dtype,
        narrow_hyper(config.learning_rate, config.hyper_layout)?,
    );
    let scaled_update = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (learning_rate, scalar),
        (update, full.clone()),
    );
    let new_param = expr::binary(
        program,
        dtype,
        ScalarOp::Subtract,
        (param, full.clone()),
        (scaled_update, full),
    );

    Ok((new_param, m_new, v_new))
}

/// `1 - beta^step`, rank 0. `beta^step = exp(step * ln(beta))`; `ln(beta)`
/// is folded in as a host-computed [`proxima_tensor::op::Op::Constant`]
/// since `beta` is a compile-time [`AdamConfig`] field, never a graph
/// value.
fn bias_correction(program: &mut Vec<Op>, step: NodeId, ln_beta: f32) -> NodeId {
    let dtype = DType::FLOAT32;
    let zeroth = expr::identity(0);
    let ln_beta_constant = expr::constant(program, dtype, ln_beta);
    let exponent = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (step, zeroth.clone()),
        (ln_beta_constant, zeroth.clone()),
    );
    let beta_pow_step = expr::unary(
        program,
        dtype,
        ScalarOp::Exponential,
        (exponent, zeroth.clone()),
    );
    let one = expr::constant(program, dtype, 1.0);
    expr::binary(
        program,
        dtype,
        ScalarOp::Subtract,
        (one, zeroth.clone()),
        (beta_pow_step, zeroth),
    )
}

/// [`adam_step`] wearing this workspace's uniform `Pipe` shape — see this
/// module's own doc for why it needs a `RefCell` and
/// [`crate::adjoint::Differentiate`] does not.
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
        Self {
            program: RefCell::new(program),
            config,
            rank,
            step,
        }
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
    /// Refuses a config whose layout the scalar codec cannot run.
    type Err = OptimizerError;

    fn call(&self, operands: Self::In) -> impl Future<Output = Result<Self::Out, OptimizerError>> {
        async move {
            let mut program = self.program.borrow_mut();
            adam_step(
                &mut program,
                &self.config,
                self.rank,
                operands,
                self.step,
            )
        }
    }
}

/// SGD's one tunable. Config-as-data and a fluent builder both first-class
/// (guiding principle 4), same three-crate stack as [`AdamConfig`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "config",
    derive(
        bon::Builder,
        serde::Serialize,
        serde::Deserialize,
        conflaguration::Settings
    )
)]
#[cfg_attr(feature = "config", settings(prefix = "AUTOGRAD_SGD"))]
#[cfg_attr(feature = "config", builder(derive(Clone, Debug)))]
#[cfg_attr(feature = "config", serde(default, deny_unknown_fields))]
pub struct SgdConfig {
    /// Step size. Scales the gradient before it is subtracted from the
    /// parameter.
    #[cfg_attr(feature = "config", setting(default = 0.01))]
    #[cfg_attr(feature = "config", builder(default = 0.01))]
    pub learning_rate: Hyper,
    /// Layout the step size is rounded into before it becomes a graph constant.
    #[cfg_attr(feature = "config", setting(default_str = "float32", resolve_with = "parse_layout"))]
    #[cfg_attr(feature = "config", builder(default = DEFAULT_LAYOUT))]
    #[cfg_attr(feature = "config", serde(with = "layout_name"))]
    pub hyper_layout: FloatLayout,
}

impl Default for SgdConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            hyper_layout: DEFAULT_LAYOUT,
        }
    }
}

#[cfg(feature = "config")]
impl conflaguration::Validate for SgdConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        check_positive(&mut errors, "learning_rate", self.learning_rate);
        check_layout(&mut errors, "hyper_layout", self.hyper_layout);
        finish(errors)
    }
}

/// Plain gradient descent: `param - lr*grad`. One [`ScalarOp::Multiply`]
/// and one [`ScalarOp::Subtract`] over existing `Op::Input` leaves — the
/// smallest possible instance of this module's "elementwise expression, not
/// a method on anything" shape (this module's own doc).
pub fn sgd_step(
    program: &mut Vec<Op>,
    config: &SgdConfig,
    rank: u16,
    param: NodeId,
    grad: NodeId,
) -> Result<NodeId, OptimizerError> {
    let full = expr::identity(rank);
    let scalar = expr::broadcast(rank);
    let dtype = DType::FLOAT32;

    let learning_rate = expr::constant(
        program,
        dtype,
        narrow_hyper(config.learning_rate, config.hyper_layout)?,
    );
    let scaled_grad = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (learning_rate, scalar),
        (grad, full.clone()),
    );
    Ok(expr::binary(
        program,
        dtype,
        ScalarOp::Subtract,
        (param, full.clone()),
        (scaled_grad, full),
    ))
}

/// [`sgd_step`] wearing this workspace's uniform `Pipe` shape — see
/// [`AdamStep`]'s own doc for why the `RefCell` is needed here and not in
/// [`crate::adjoint::Differentiate`]. `In = (param, grad)`: a plain tuple,
/// not a named struct, because two positional `NodeId`s need no field names
/// to stay unambiguous at the one call site that builds them (contrast
/// [`AdamOperands`]'s four fields, which do).
pub struct SgdStep {
    program: RefCell<Vec<Op>>,
    config: SgdConfig,
    rank: u16,
}

impl SgdStep {
    #[must_use]
    pub fn new(program: Vec<Op>, config: SgdConfig, rank: u16) -> Self {
        Self {
            program: RefCell::new(program),
            config,
            rank,
        }
    }

    #[must_use]
    pub fn finish(self) -> Vec<Op> {
        self.program.into_inner()
    }
}

impl Pipe for SgdStep {
    type In = (NodeId, NodeId);
    type Out = NodeId;
    type Err = OptimizerError;

    fn call(&self, (param, grad): Self::In) -> impl Future<Output = Result<Self::Out, OptimizerError>> {
        async move {
            let mut program = self.program.borrow_mut();
            sgd_step(&mut program, &self.config, self.rank, param, grad)
        }
    }
}

/// RMSprop's three tunables (Hinton, unpublished; Tieleman & Hinton 2012).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "config",
    derive(
        bon::Builder,
        serde::Serialize,
        serde::Deserialize,
        conflaguration::Settings
    )
)]
#[cfg_attr(feature = "config", settings(prefix = "AUTOGRAD_RMSPROP"))]
#[cfg_attr(feature = "config", builder(derive(Clone, Debug)))]
#[cfg_attr(feature = "config", serde(default, deny_unknown_fields))]
pub struct RmspropConfig {
    /// Step size. Scales the gradient, damped by its own running variance,
    /// before it is subtracted from the parameter.
    #[cfg_attr(feature = "config", setting(default = 0.001))]
    #[cfg_attr(feature = "config", builder(default = 0.001))]
    pub learning_rate: Hyper,
    /// Second-moment (uncentered variance) decay. Closer to 1 remembers
    /// more gradient history.
    #[cfg_attr(feature = "config", setting(default = 0.9))]
    #[cfg_attr(feature = "config", builder(default = 0.9))]
    pub decay: Hyper,
    /// Added to the second-moment square root before dividing, so a
    /// parameter with near-zero gradient history never divides by zero.
    #[cfg_attr(feature = "config", setting(default = 1e-8))]
    #[cfg_attr(feature = "config", builder(default = 1e-8))]
    pub epsilon: Hyper,
    /// Layout the hyper-parameters above are rounded into before they become graph constants.
    #[cfg_attr(feature = "config", setting(default_str = "float32", resolve_with = "parse_layout"))]
    #[cfg_attr(feature = "config", builder(default = DEFAULT_LAYOUT))]
    #[cfg_attr(feature = "config", serde(with = "layout_name"))]
    pub hyper_layout: FloatLayout,
    /// Storage layout of the second moment `v` between steps.
    #[cfg_attr(feature = "config", setting(default_str = "float32", resolve_with = "parse_layout"))]
    #[cfg_attr(feature = "config", builder(default = DEFAULT_LAYOUT))]
    #[cfg_attr(feature = "config", serde(with = "layout_name"))]
    pub second_moment_layout: FloatLayout,
}

impl Default for RmspropConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            decay: 0.9,
            epsilon: 1e-8,
            hyper_layout: DEFAULT_LAYOUT,
            second_moment_layout: DEFAULT_LAYOUT,
        }
    }
}

impl RmspropConfig {
    /// Rounds a step's `new_v` read-back through [`Self::second_moment_layout`] ([`store_moment`]).
    ///
    /// # Errors
    /// [`OptimizerError`] for a layout the scalar codec cannot run.
    pub fn store_second_moment(&self, values: &mut [f32]) -> Result<(), OptimizerError> {
        store_moment(values, self.second_moment_layout, "second_moment_layout")
    }
}

#[cfg(feature = "config")]
impl conflaguration::Validate for RmspropConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        check_positive(&mut errors, "learning_rate", self.learning_rate);
        check_positive(&mut errors, "epsilon", self.epsilon);
        check_decay(&mut errors, "decay", self.decay);
        check_layout(&mut errors, "hyper_layout", self.hyper_layout);
        check_layout(&mut errors, "second_moment_layout", self.second_moment_layout);
        finish(errors)
    }
}

/// Appends one RMSprop update for a single `rank`-dimensional parameter,
/// returning `(new_param, new_v)`. `v := decay*v + (1-decay)*g^2`;
/// `p := p - lr*g/(sqrt(v)+eps)` -- [`adam_step`]'s second-moment tracking
/// with neither a first moment nor bias correction, the same reduction
/// [`rmsprop_step`]'s own history (Adam without `m`/bias-correction) makes.
pub fn rmsprop_step(
    program: &mut Vec<Op>,
    config: &RmspropConfig,
    rank: u16,
    param: NodeId,
    grad: NodeId,
    v_prev: NodeId,
) -> Result<(NodeId, NodeId), OptimizerError> {
    let full = expr::identity(rank);
    let scalar = expr::broadcast(rank);
    let dtype = DType::FLOAT32;

    let decay_value = narrow_hyper(config.decay, config.hyper_layout)?;
    let decay = expr::constant(program, dtype, decay_value);
    let one_minus_decay = expr::constant(
        program,
        dtype,
        narrow_complement(decay_value, config.hyper_layout)?,
    );

    let grad_sq = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (grad, full.clone()),
        (grad, full.clone()),
    );
    let v_scaled = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (decay, scalar.clone()),
        (v_prev, full.clone()),
    );
    let grad_sq_scaled = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (one_minus_decay, scalar.clone()),
        (grad_sq, full.clone()),
    );
    let v_new = expr::binary(
        program,
        dtype,
        ScalarOp::Add,
        (v_scaled, full.clone()),
        (grad_sq_scaled, full.clone()),
    );

    let sqrt_v = expr::unary(program, dtype, ScalarOp::SquareRoot, (v_new, full.clone()));
    let epsilon = expr::constant(
        program,
        dtype,
        narrow_hyper(config.epsilon, config.hyper_layout)?,
    );
    let denominator = expr::binary(
        program,
        dtype,
        ScalarOp::Add,
        (sqrt_v, full.clone()),
        (epsilon, scalar.clone()),
    );
    let recip_denominator = expr::unary(
        program,
        dtype,
        ScalarOp::Reciprocal,
        (denominator, full.clone()),
    );
    let update = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (grad, full.clone()),
        (recip_denominator, full.clone()),
    );

    let learning_rate = expr::constant(
        program,
        dtype,
        narrow_hyper(config.learning_rate, config.hyper_layout)?,
    );
    let scaled_update = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (learning_rate, scalar),
        (update, full.clone()),
    );
    let new_param = expr::binary(
        program,
        dtype,
        ScalarOp::Subtract,
        (param, full.clone()),
        (scaled_update, full),
    );

    Ok((new_param, v_new))
}

/// [`rmsprop_step`] wearing this workspace's uniform `Pipe` shape. `In =
/// (param, grad, v)`: a plain tuple, the same reasoning [`SgdStep`]'s own
/// doc gives for not minting a named struct at three positional `NodeId`s.
pub struct RmspropStep {
    program: RefCell<Vec<Op>>,
    config: RmspropConfig,
    rank: u16,
}

impl RmspropStep {
    #[must_use]
    pub fn new(program: Vec<Op>, config: RmspropConfig, rank: u16) -> Self {
        Self {
            program: RefCell::new(program),
            config,
            rank,
        }
    }

    #[must_use]
    pub fn finish(self) -> Vec<Op> {
        self.program.into_inner()
    }
}

impl Pipe for RmspropStep {
    type In = (NodeId, NodeId, NodeId);
    type Out = (NodeId, NodeId);
    type Err = OptimizerError;

    fn call(
        &self,
        (param, grad, v_prev): Self::In,
    ) -> impl Future<Output = Result<Self::Out, OptimizerError>> {
        async move {
            let mut program = self.program.borrow_mut();
            rmsprop_step(
                &mut program,
                &self.config,
                self.rank,
                param,
                grad,
                v_prev,
            )
        }
    }
}

/// AdamW's five tunables (Loshchilov & Hutter 2019) -- [`AdamConfig`] plus
/// decoupled weight decay.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "config",
    derive(
        bon::Builder,
        serde::Serialize,
        serde::Deserialize,
        conflaguration::Settings
    )
)]
#[cfg_attr(feature = "config", settings(prefix = "AUTOGRAD_ADAMW"))]
#[cfg_attr(feature = "config", builder(derive(Clone, Debug)))]
#[cfg_attr(feature = "config", serde(default, deny_unknown_fields))]
pub struct AdamwConfig {
    /// Step size, as [`AdamConfig::learning_rate`].
    #[cfg_attr(feature = "config", setting(default = 0.001))]
    #[cfg_attr(feature = "config", builder(default = 0.001))]
    pub learning_rate: Hyper,
    /// First-moment decay, as [`AdamConfig::beta1`].
    #[cfg_attr(feature = "config", setting(default = 0.9))]
    #[cfg_attr(feature = "config", builder(default = 0.9))]
    pub beta1: Hyper,
    /// Second-moment decay, as [`AdamConfig::beta2`].
    #[cfg_attr(feature = "config", setting(default = 0.999))]
    #[cfg_attr(feature = "config", builder(default = 0.999))]
    pub beta2: Hyper,
    /// Denominator guard, as [`AdamConfig::epsilon`].
    #[cfg_attr(feature = "config", setting(default = 1e-8))]
    #[cfg_attr(feature = "config", builder(default = 1e-8))]
    pub epsilon: Hyper,
    /// Decoupled weight decay coefficient (Loshchilov & Hutter 2019, eq.
    /// 6): applied directly to the parameter, scaled by `learning_rate`,
    /// rather than folded into the gradient the way L2 regularization is --
    /// the "decoupled" half of the name.
    #[cfg_attr(feature = "config", setting(default = 0.01))]
    #[cfg_attr(feature = "config", builder(default = 0.01))]
    pub weight_decay: Hyper,
    /// Layout the hyper-parameters above are rounded into before they become graph constants.
    #[cfg_attr(feature = "config", setting(default_str = "float32", resolve_with = "parse_layout"))]
    #[cfg_attr(feature = "config", builder(default = DEFAULT_LAYOUT))]
    #[cfg_attr(feature = "config", serde(with = "layout_name"))]
    pub hyper_layout: FloatLayout,
    /// Storage layout of the first moment `m` between steps.
    #[cfg_attr(feature = "config", setting(default_str = "float32", resolve_with = "parse_layout"))]
    #[cfg_attr(feature = "config", builder(default = DEFAULT_LAYOUT))]
    #[cfg_attr(feature = "config", serde(with = "layout_name"))]
    pub first_moment_layout: FloatLayout,
    /// Storage layout of the second moment `v` between steps.
    #[cfg_attr(feature = "config", setting(default_str = "float32", resolve_with = "parse_layout"))]
    #[cfg_attr(feature = "config", builder(default = DEFAULT_LAYOUT))]
    #[cfg_attr(feature = "config", serde(with = "layout_name"))]
    pub second_moment_layout: FloatLayout,
}

impl Default for AdamwConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            hyper_layout: DEFAULT_LAYOUT,
            first_moment_layout: DEFAULT_LAYOUT,
            second_moment_layout: DEFAULT_LAYOUT,
        }
    }
}

impl AdamwConfig {
    /// Rounds a step's `new_m` read-back through [`Self::first_moment_layout`] ([`store_moment`]).
    ///
    /// # Errors
    /// [`OptimizerError`] for a layout the scalar codec cannot run.
    pub fn store_first_moment(&self, values: &mut [f32]) -> Result<(), OptimizerError> {
        store_moment(values, self.first_moment_layout, "first_moment_layout")
    }

    /// Rounds a step's `new_v` read-back through [`Self::second_moment_layout`] ([`store_moment`]).
    ///
    /// # Errors
    /// [`OptimizerError`] for a layout the scalar codec cannot run.
    pub fn store_second_moment(&self, values: &mut [f32]) -> Result<(), OptimizerError> {
        store_moment(values, self.second_moment_layout, "second_moment_layout")
    }
}

#[cfg(feature = "config")]
impl conflaguration::Validate for AdamwConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        check_positive(&mut errors, "learning_rate", self.learning_rate);
        check_positive(&mut errors, "epsilon", self.epsilon);
        check_decay(&mut errors, "beta1", self.beta1);
        check_decay(&mut errors, "beta2", self.beta2);
        if !(self.weight_decay.is_finite() && self.weight_decay >= 0.0) {
            errors.push(conflaguration::ValidationMessage::new(
                "weight_decay",
                "must be finite and not negative",
            ));
        }
        check_layout(&mut errors, "hyper_layout", self.hyper_layout);
        check_layout(&mut errors, "first_moment_layout", self.first_moment_layout);
        check_layout(&mut errors, "second_moment_layout", self.second_moment_layout);
        finish(errors)
    }
}

/// Appends one AdamW update for a single `rank`-dimensional parameter,
/// returning `(new_param, new_m, new_v)` -- [`adam_step`]'s exact `m`/`v`
/// update and bias correction, with `p := p - lr*(m_hat/(sqrt(v_hat)+eps) +
/// weight_decay*p)` in place of plain Adam's `p := p - lr*m_hat/(sqrt(v_hat)+eps)`:
/// the weight-decay term is added AFTER the adaptive step is computed, not
/// mixed into `grad` beforehand, which is what "decoupled" means. Reuses
/// [`AdamOperands`] unchanged -- `(param, grad, m, v)` is exactly what this
/// update touches too, so no second operand type exists for it.
pub fn adamw_step(
    program: &mut Vec<Op>,
    config: &AdamwConfig,
    rank: u16,
    operands: AdamOperands,
    step: NodeId,
) -> Result<(NodeId, NodeId, NodeId), OptimizerError> {
    let AdamOperands {
        param,
        grad,
        m: m_prev,
        v: v_prev,
    } = operands;
    let full = expr::identity(rank);
    let scalar = expr::broadcast(rank);
    let dtype = DType::FLOAT32;

    let layout = config.hyper_layout;
    let beta1_value = narrow_hyper(config.beta1, layout)?;
    let beta2_value = narrow_hyper(config.beta2, layout)?;
    let beta1 = expr::constant(program, dtype, beta1_value);
    let one_minus_beta1 =
        expr::constant(program, dtype, narrow_complement(beta1_value, layout)?);
    let beta2 = expr::constant(program, dtype, beta2_value);
    let one_minus_beta2 =
        expr::constant(program, dtype, narrow_complement(beta2_value, layout)?);

    let m_scaled = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (beta1, scalar.clone()),
        (m_prev, full.clone()),
    );
    let grad_scaled = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (one_minus_beta1, scalar.clone()),
        (grad, full.clone()),
    );
    let m_new = expr::binary(
        program,
        dtype,
        ScalarOp::Add,
        (m_scaled, full.clone()),
        (grad_scaled, full.clone()),
    );

    let grad_sq = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (grad, full.clone()),
        (grad, full.clone()),
    );
    let v_scaled = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (beta2, scalar.clone()),
        (v_prev, full.clone()),
    );
    let grad_sq_scaled = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (one_minus_beta2, scalar.clone()),
        (grad_sq, full.clone()),
    );
    let v_new = expr::binary(
        program,
        dtype,
        ScalarOp::Add,
        (v_scaled, full.clone()),
        (grad_sq_scaled, full.clone()),
    );

    let bias1_denominator = bias_correction(program, step, libm::logf(beta1_value));
    let bias2_denominator = bias_correction(program, step, libm::logf(beta2_value));
    let zeroth = expr::identity(0);

    let recip_bias1 = expr::unary(
        program,
        dtype,
        ScalarOp::Reciprocal,
        (bias1_denominator, zeroth.clone()),
    );
    let m_hat = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (m_new, full.clone()),
        (recip_bias1, scalar.clone()),
    );

    let recip_bias2 = expr::unary(
        program,
        dtype,
        ScalarOp::Reciprocal,
        (bias2_denominator, zeroth),
    );
    let v_hat = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (v_new, full.clone()),
        (recip_bias2, scalar.clone()),
    );

    let sqrt_v_hat = expr::unary(program, dtype, ScalarOp::SquareRoot, (v_hat, full.clone()));
    let epsilon = expr::constant(
        program,
        dtype,
        narrow_hyper(config.epsilon, config.hyper_layout)?,
    );
    let denominator = expr::binary(
        program,
        dtype,
        ScalarOp::Add,
        (sqrt_v_hat, full.clone()),
        (epsilon, scalar.clone()),
    );
    let recip_denominator = expr::unary(
        program,
        dtype,
        ScalarOp::Reciprocal,
        (denominator, full.clone()),
    );
    let adaptive_update = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (m_hat, full.clone()),
        (recip_denominator, full.clone()),
    );

    let weight_decay = expr::constant(
        program,
        dtype,
        narrow_hyper(config.weight_decay, config.hyper_layout)?,
    );
    let decay_term = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (weight_decay, scalar.clone()),
        (param, full.clone()),
    );
    let update = expr::binary(
        program,
        dtype,
        ScalarOp::Add,
        (adaptive_update, full.clone()),
        (decay_term, full.clone()),
    );

    let learning_rate = expr::constant(
        program,
        dtype,
        narrow_hyper(config.learning_rate, config.hyper_layout)?,
    );
    let scaled_update = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (learning_rate, scalar),
        (update, full.clone()),
    );
    let new_param = expr::binary(
        program,
        dtype,
        ScalarOp::Subtract,
        (param, full.clone()),
        (scaled_update, full),
    );

    Ok((new_param, m_new, v_new))
}

/// [`adamw_step`] wearing this workspace's uniform `Pipe` shape -- see
/// [`AdamStep`]'s own doc for the `RefCell`. `In = AdamOperands`,
/// `Out = (new_param, new_m, new_v)`, identical to [`AdamStep`]'s shape:
/// the two updates share an operand type because they genuinely touch the
/// same four tensors.
pub struct AdamwStep {
    program: RefCell<Vec<Op>>,
    config: AdamwConfig,
    rank: u16,
    step: NodeId,
}

impl AdamwStep {
    #[must_use]
    pub fn new(program: Vec<Op>, config: AdamwConfig, rank: u16, step: NodeId) -> Self {
        Self {
            program: RefCell::new(program),
            config,
            rank,
            step,
        }
    }

    #[must_use]
    pub fn finish(self) -> Vec<Op> {
        self.program.into_inner()
    }
}

impl Pipe for AdamwStep {
    type In = AdamOperands;
    type Out = (NodeId, NodeId, NodeId);
    type Err = OptimizerError;

    fn call(&self, operands: Self::In) -> impl Future<Output = Result<Self::Out, OptimizerError>> {
        async move {
            let mut program = self.program.borrow_mut();
            adamw_step(
                &mut program,
                &self.config,
                self.rank,
                operands,
                self.step,
            )
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
                dtype: DType::FLOAT32,
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

        let (new_param, new_m, new_v) = adam_step(
            &mut program,
            &config,
            1,
            AdamOperands {
                param,
                grad,
                m: m_prev,
                v: v_prev,
            },
            step,
        ).expect("step layouts are supported");

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
        assert!(
            (updated_m[0] - 0.1).abs() < 1e-6,
            "m = (1-b1)*g, got {updated_m:?}"
        );
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
        let operands = AdamOperands {
            param,
            grad,
            m: m_prev,
            v: v_prev,
        };

        let mut via_function_program = program.clone();
        let via_function = adam_step(&mut via_function_program, &config, 1, operands, step).expect("step layouts are supported");

        let pipe = AdamStep::new(program, config, 1, step);
        let via_pipe = block_on_once(pipe.call(operands)).expect("adam_step never fails");
        let via_pipe_program = pipe.finish();

        assert_eq!(
            via_pipe, via_function,
            "the Pipe wrapper must append the exact same nodes"
        );
        assert_eq!(
            via_pipe_program, via_function_program,
            "the Pipe wrapper must grow the shared program identically to the free function"
        );
    }

    #[proxima::test]
    async fn sgd_subtracts_the_scaled_gradient_exactly() {
        let mut program = Vec::new();
        let param = leaf(&mut program, "param", 2);
        let grad = leaf(&mut program, "grad", 2);
        let config = SgdConfig::default();
        let new_param = sgd_step(&mut program, &config, 1, param, grad).expect("step layouts are supported");

        let param_values = [1.0f32, -1.0];
        let grad_values = [1.0f32, -1.0];
        let evaluated = proxima_tensor::cpu::evaluate_named(
            &program,
            &[],
            &[("param", &param_values), ("grad", &grad_values)],
            &[new_param],
        )
        .expect("sgd program lowers and evaluates");
        let updated = evaluated.get(new_param).expect("new_param requested").0;

        assert!((updated[0] - 0.99).abs() < 1e-6, "got {updated:?}");
        assert!((updated[1] - (-0.99)).abs() < 1e-6, "got {updated:?}");
    }

    #[test]
    fn sgd_pipe_form_agrees_with_the_free_function() {
        let mut program = Vec::new();
        let param = leaf(&mut program, "param", 2);
        let grad = leaf(&mut program, "grad", 2);
        let config = SgdConfig::default();

        let mut via_function_program = program.clone();
        let via_function = sgd_step(&mut via_function_program, &config, 1, param, grad).expect("step layouts are supported");

        let pipe = SgdStep::new(program, config, 1);
        let via_pipe = block_on_once(pipe.call((param, grad))).expect("sgd_step never fails");
        let via_pipe_program = pipe.finish();

        assert_eq!(via_pipe, via_function);
        assert_eq!(via_pipe_program, via_function_program);
    }

    #[proxima::test]
    async fn rmsprop_moves_the_parameter_toward_the_gradients_sign() {
        let mut program = Vec::new();
        let param = leaf(&mut program, "param", 2);
        let grad = leaf(&mut program, "grad", 2);
        let v_prev = leaf(&mut program, "v", 2);
        let config = RmspropConfig::default();
        let (new_param, new_v) = rmsprop_step(&mut program, &config, 1, param, grad, v_prev).expect("step layouts are supported");

        let param_values = [1.0f32, -1.0];
        let grad_values = [1.0f32, -1.0];
        let zero = [0.0f32, 0.0];
        let evaluated = proxima_tensor::cpu::evaluate_named(
            &program,
            &[],
            &[
                ("param", &param_values),
                ("grad", &grad_values),
                ("v", &zero),
            ],
            &[new_param, new_v],
        )
        .expect("rmsprop program lowers and evaluates");
        let updated_param = evaluated.get(new_param).expect("new_param requested").0;
        let updated_v = evaluated.get(new_v).expect("new_v requested").0;

        assert!(
            (updated_param[0] - 0.996_837_7).abs() < 1e-5,
            "got {updated_param:?}"
        );
        assert!(
            (updated_param[1] - (-0.996_837_7)).abs() < 1e-5,
            "got {updated_param:?}"
        );
        assert!(
            (updated_v[0] - 0.1).abs() < 1e-6,
            "v = (1-decay)*g^2, got {updated_v:?}"
        );
    }

    #[test]
    fn rmsprop_pipe_form_agrees_with_the_free_function() {
        let mut program = Vec::new();
        let param = leaf(&mut program, "param", 2);
        let grad = leaf(&mut program, "grad", 2);
        let v_prev = leaf(&mut program, "v", 2);
        let config = RmspropConfig::default();

        let mut via_function_program = program.clone();
        let via_function = rmsprop_step(&mut via_function_program, &config, 1, param, grad, v_prev).expect("step layouts are supported");

        let pipe = RmspropStep::new(program, config, 1);
        let via_pipe =
            block_on_once(pipe.call((param, grad, v_prev))).expect("rmsprop_step never fails");
        let via_pipe_program = pipe.finish();

        assert_eq!(via_pipe, via_function);
        assert_eq!(via_pipe_program, via_function_program);
    }

    #[proxima::test]
    async fn adamw_subtracts_the_adaptive_step_plus_decoupled_decay() {
        let mut program = Vec::new();
        let param = leaf(&mut program, "param", 2);
        let grad = leaf(&mut program, "grad", 2);
        let m_prev = leaf(&mut program, "m", 2);
        let v_prev = leaf(&mut program, "v", 2);
        let step = step_input(&mut program, "step");
        let config = AdamwConfig::default();
        let operands = AdamOperands {
            param,
            grad,
            m: m_prev,
            v: v_prev,
        };
        let (new_param, new_m, new_v) = adamw_step(&mut program, &config, 1, operands, step).expect("step layouts are supported");

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
        .expect("adamw program lowers and evaluates");
        let updated_param = evaluated.get(new_param).expect("new_param requested").0;

        // hand-computed: m_hat=[1,-1], v_hat=[1,1], adaptive_update~=[1,-1],
        // decay_term=weight_decay*param=[0.01,-0.01], update=[1.01,-1.01],
        // new_param = param - lr*update = [1-0.00101, -1+0.00101]
        assert!(
            (updated_param[0] - 0.998_99).abs() < 1e-4,
            "got {updated_param:?}"
        );
        assert!(
            (updated_param[1] - (-0.998_99)).abs() < 1e-4,
            "got {updated_param:?}"
        );
    }

    #[test]
    fn adamw_pipe_form_agrees_with_the_free_function() {
        let mut program = Vec::new();
        let param = leaf(&mut program, "param", 2);
        let grad = leaf(&mut program, "grad", 2);
        let m_prev = leaf(&mut program, "m", 2);
        let v_prev = leaf(&mut program, "v", 2);
        let step = step_input(&mut program, "step");
        let config = AdamwConfig::default();
        let operands = AdamOperands {
            param,
            grad,
            m: m_prev,
            v: v_prev,
        };

        let mut via_function_program = program.clone();
        let via_function = adamw_step(&mut via_function_program, &config, 1, operands, step).expect("step layouts are supported");

        let pipe = AdamwStep::new(program, config, 1, step);
        let via_pipe = block_on_once(pipe.call(operands)).expect("adamw_step never fails");
        let via_pipe_program = pipe.finish();

        assert_eq!(via_pipe, via_function);
        assert_eq!(via_pipe_program, via_function_program);
    }
}
