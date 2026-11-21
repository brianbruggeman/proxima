//! One training step as a plan any `RuntimeConfig` engine can execute.
//!
//! Autograd is not re-implemented here. [`build_optimizer_step`] calls
//! [`proxima_autograd::adjoint::differentiate_wanted`] and the selected optimizer
//! ([`proxima_autograd::optimizer::adam_step`] or
//! [`proxima_autograd::optimizer::adamw_step`]) to turn a forward program plus
//! its loss node into the single forward+backward+optimizer program that
//! [`crate::backend::plan_named_with_config`] already knows how to run.
//! [`TrainingStep`] then owns the named state buffers (`w`, `m_w`, `v_w`, ...)
//! a caller would otherwise have to rebind by hand between steps.
//!
//! # Configuration
//!
//! [`TrainConfig`] is the whole surface: a `conflaguration` object nesting
//! [`crate::config::RuntimeConfig`] (engine, driver, numeric policy, numerics),
//! the optimizer selection and its hyper-parameters, and the loop schedule.
//! Load it from TOML with [`TrainConfig::from_file`], overlay `OMEGA_TRAIN_*`
//! on a file with [`TrainConfig::from_file_then_env`], read the environment
//! alone with [`TrainConfig::from_env`], or build it explicitly with
//! [`TrainConfig::builder`]. Planning goes through
//! [`crate::backend::plan_named_with_config`], so the engine/driver choice is a
//! config edit, not a second set of arguments.
//!
//! ```
//! use omega::train::TrainConfig;
//!
//! let config = TrainConfig::builder().optimizer("adamw").build()?;
//! assert_eq!(config.optimizer, "adamw");
//! assert_eq!(config.schedule.ubatch, 512);
//! # Ok::<(), conflaguration::Error>(())
//! ```
//!
//! Numerics are `proxima-numeric`'s: [`proxima_numeric::NumericPolicy`] inside
//! `RuntimeConfig`, and each optimizer's `FloatLayout` hyper/moment storage
//! in its own config. Nothing here defines a format.
//!
//! # State names
//!
//! Parameters keep their forward names. The first moment is
//! [`adam_first_moment_name`] (`m_<parameter>`) and the second is
//! [`adam_second_moment_name`] (`v_<parameter>`). Bias correction reads the
//! rank-0 [`STEP_INPUT`] input, which the caller supplies every step. All of
//! them are `FLOAT32`.
//!
//! # Boundary
//!
//! `differentiate_wanted` hands a gathered operand's gradient back as a compact
//! host-side [`proxima_autograd::adjoint::GatheredContribution`]; applying it
//! stays the caller's job (`proxima_autograd::sparse::dedupe_and_sum_rows`
//! plus an optimizer step sized to the touched rows). This module therefore
//! covers the dense-gradient case: every parameter passed to
//! [`build_optimizer_step`] must feed the loss through a non-gathered read, or
//! it is rejected with [`TrainError::MissingGradient`] rather than silently
//! skipped. Every extent must be static: the plan binds an empty symbol table.

use std::collections::BTreeSet;
use std::path::Path;

use conflaguration::{Settings, Validate, ValidationMessage};
use proxima_autograd::adjoint::differentiate_wanted;
use proxima_autograd::optimizer::{
    AdamOperands, OptimizerError, adam_step, adamw_step, step_input, store_moment,
};
use proxima_numeric::DType;
use proxima_numeric::float_layout::FloatLayout;
use proxima_tensor::cpu::QuantizedBlock;
use proxima_tensor::op::{Extent, NodeId, Op, append};
use serde::{Deserialize, Serialize};

use crate::backend::{BackendError, Plan, execute_plan_named, plan_named_with_config};
use crate::config::RuntimeConfig;

pub use proxima_autograd::optimizer::{AdamConfig, AdamwConfig};

/// The rank-0 input bias correction reads. Bind it to the current step number
/// (`1.0`, `2.0`, ...) on every [`TrainingStep::step`] call.
pub const STEP_INPUT: &str = "step";

/// The first-moment state name for `parameter`.
#[must_use]
pub fn adam_first_moment_name(parameter: &str) -> String {
    format!("m_{parameter}")
}

/// The second-moment state name for `parameter`.
#[must_use]
pub fn adam_second_moment_name(parameter: &str) -> String {
    format!("v_{parameter}")
}

/// The loop schedule an optimizer step is embedded in. The step itself reads
/// only [`Self::ubatch`]; the rest is carried so a training driver and its
/// benchmark read one config rather than inventing their own.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, Settings)]
#[serde(default, deny_unknown_fields)]
#[settings(prefix = "OMEGA_TRAIN_SCHEDULE")]
pub struct TrainSchedule {
    /// Epochs over the dataset, llama.cpp's `--epochs` (default 2).
    #[setting(default = 2)]
    pub epochs: u32,
    /// Fraction held out for validation, llama.cpp's `--val-split` (default 0.05).
    #[setting(default = 0.05)]
    pub val_split: f64,
    /// Logical batch size, llama.cpp's `--batch-size` (default 2048).
    #[setting(default = 2048)]
    pub batch: u32,
    /// Physical batch one step processes, llama.cpp's `--ubatch-size` (default 512).
    #[setting(default = 512)]
    pub ubatch: u32,
}

impl Default for TrainSchedule {
    fn default() -> Self {
        Self {
            epochs: 2,
            val_split: 0.05,
            batch: 2048,
            ubatch: 512,
        }
    }
}

impl Validate for TrainSchedule {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.epochs == 0 {
            errors.push(ValidationMessage::new(
                "epochs",
                "must be greater than zero",
            ));
        }
        if !(0.0..1.0).contains(&self.val_split) {
            errors.push(ValidationMessage::new(
                "val_split",
                "must be at least 0 and below 1",
            ));
        }
        if self.ubatch == 0 {
            errors.push(ValidationMessage::new(
                "ubatch",
                "must be greater than zero",
            ));
        }
        if self.batch < self.ubatch {
            errors.push(ValidationMessage::new("batch", "must be at least ubatch"));
        }
        finish(errors)
    }
}

/// The whole training surface as one `conflaguration` object.
///
/// `runtime` decides the engine, driver, numeric policy and numerics;
/// `optimizer` names `adam` or `adamw`; the selected optimizer's own config
/// carries its hyper-parameters and `FloatLayout` storage; `schedule` carries
/// the loop shape. Field names are validated, so a typo or an unknown
/// optimizer is refused by name rather than silently defaulted.
#[derive(Debug, Clone, Deserialize, Serialize, Settings)]
#[serde(default, deny_unknown_fields)]
#[settings(prefix = "OMEGA_TRAIN")]
pub struct TrainConfig {
    /// Engine, driver, numeric policy and numerics, the same object inference uses.
    #[setting(nested)]
    pub runtime: RuntimeConfig,
    /// Optimizer name: `adam` or `adamw`.
    #[setting(default = "adam")]
    pub optimizer: String,
    /// Loop shape: epochs, validation split, logical and physical batch.
    #[setting(nested)]
    pub schedule: TrainSchedule,
    /// Adam hyper-parameters and moment layouts, used when `optimizer = "adam"`.
    #[setting(nested)]
    pub adam: AdamConfig,
    /// AdamW hyper-parameters, moment layouts and decoupled weight decay,
    /// used when `optimizer = "adamw"`.
    #[setting(nested)]
    pub adamw: AdamwConfig,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            runtime: RuntimeConfig::default(),
            optimizer: String::from("adam"),
            schedule: TrainSchedule::default(),
            adam: AdamConfig::default(),
            adamw: AdamwConfig::default(),
        }
    }
}

impl Validate for TrainConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        self.runtime.validate()?;
        self.schedule.validate()?;
        self.adam.validate()?;
        self.adamw.validate()?;
        let mut errors = Vec::new();
        if !matches!(self.optimizer.as_str(), "adam" | "adamw") {
            errors.push(ValidationMessage::new(
                "optimizer",
                format!(
                    "unknown optimizer '{}'; expected adam|adamw",
                    self.optimizer
                ),
            ));
        }
        finish(errors)
    }
}

impl TrainConfig {
    /// Load defaults overlaid by `OMEGA_TRAIN_*` values and validate.
    ///
    /// # Errors
    /// [`conflaguration::Error`] from the environment layer or validation.
    pub fn from_env() -> conflaguration::Result<Self> {
        let config = <Self as Settings>::from_env()?;
        config.validate()?;
        Ok(config)
    }

    /// Load a TOML file and validate the resolved config.
    ///
    /// # Errors
    /// [`conflaguration::Error`] from the file reader or validation.
    pub fn from_file(path: impl AsRef<Path>) -> conflaguration::Result<Self> {
        let config: Self = conflaguration::from_file(path.as_ref())?;
        config.validate()?;
        Ok(config)
    }

    /// Load a TOML file, overlay `OMEGA_TRAIN_*` values, and validate.
    ///
    /// # Errors
    /// [`conflaguration::Error`] from the file reader, the environment layer
    /// or validation.
    pub fn from_file_then_env(path: impl AsRef<Path>) -> conflaguration::Result<Self> {
        let config: Self = conflaguration::from_file_then_env(path.as_ref())?;
        config.validate()?;
        Ok(config)
    }

    /// Start an explicit builder seeded from the defaults.
    #[must_use]
    pub fn builder() -> TrainConfigBuilder {
        TrainConfigBuilder {
            inner: Self::default(),
        }
    }
}

/// Explicit overrides over [`TrainConfig::default`].
#[derive(Debug, Clone)]
pub struct TrainConfigBuilder {
    inner: TrainConfig,
}

impl TrainConfigBuilder {
    /// Replace the engine/driver/numeric runtime.
    #[must_use]
    pub fn runtime(mut self, runtime: RuntimeConfig) -> Self {
        self.inner.runtime = runtime;
        self
    }

    /// Name the optimizer (`adam` or `adamw`).
    #[must_use]
    pub fn optimizer(mut self, optimizer: impl Into<String>) -> Self {
        self.inner.optimizer = optimizer.into();
        self
    }

    /// Replace the loop schedule.
    #[must_use]
    pub fn schedule(mut self, schedule: TrainSchedule) -> Self {
        self.inner.schedule = schedule;
        self
    }

    /// Replace the Adam hyper-parameters.
    #[must_use]
    pub fn adam(mut self, adam: AdamConfig) -> Self {
        self.inner.adam = adam;
        self
    }

    /// Replace the AdamW hyper-parameters.
    #[must_use]
    pub fn adamw(mut self, adamw: AdamwConfig) -> Self {
        self.inner.adamw = adamw;
        self
    }

    /// Validate and return the config.
    ///
    /// # Errors
    /// [`conflaguration::Error`] naming the field that failed.
    pub fn build(self) -> conflaguration::Result<TrainConfig> {
        self.inner.validate()?;
        Ok(self.inner)
    }
}

/// Every fault graph construction or one executed step can raise.
#[derive(Debug, thiserror::Error)]
pub enum TrainError {
    #[error("parameter `{name}` is not an `Op::Input` in the forward program")]
    UnknownParameter { name: String },
    #[error("parameter `{name}` is listed more than once")]
    DuplicateParameter { name: String },
    #[error("parameter `{name}` is {dtype:?}, not FLOAT32")]
    ParameterDtype { name: String, dtype: DType },
    #[error("parameter `{name}` has a symbolic extent; its optimizer state needs a static shape")]
    SymbolicParameter { name: String },
    #[error(
        "parameter `{name}` does not feed the loss through a dense read, so it has no gradient"
    )]
    MissingGradient { name: String },
    #[error("parameter `{name}` has rank above u16::MAX")]
    RankOverflow { name: String },
    #[error("state input `{name}` already exists in the forward program")]
    StateNameCollision { name: String },
    #[error("parameter `{name}` collides with the reserved step counter input")]
    ReservedParameterName { name: String },
    #[error("input `{name}` exists but is not a rank-0 FLOAT32 step counter")]
    InvalidStepInput { name: String },
    #[error("state buffer `{name}` is not bound in `named`")]
    MissingStateBinding { name: String },
    #[error("state buffer `{name}` was not bound as FLOAT32 at plan time")]
    StateBinding { name: String },
    #[error("batch binding `{name}` collides with a state buffer the training step owns")]
    BatchOverridesState { name: String },
    #[error("the training step produced no output for node {node:?}")]
    MissingOutput { node: NodeId },
    #[error(transparent)]
    Autograd(#[from] proxima_autograd::AutogradError),
    #[error(transparent)]
    Optimizer(#[from] OptimizerError),
    #[error(transparent)]
    Configuration(#[from] conflaguration::Error),
    #[error(transparent)]
    Backend(#[from] BackendError),
}

/// A ready-to-plan training program: the forward graph, its backward pass and
/// one optimizer update per parameter, plus the state metadata a step rebinds.
///
/// [`build_optimizer_step`] fixes the runtime config, the optimizer and its
/// hyper-parameters; [`Self::plan`] consumes that same config's runtime, so a
/// second config is not a second surface.
#[derive(Debug, Clone)]
pub struct TrainingProgram {
    program: Vec<Op>,
    loss: NodeId,
    rebind: Vec<(NodeId, String)>,
    moment_layouts: Vec<Option<FloatLayout>>,
    runtime: RuntimeConfig,
}

impl TrainingProgram {
    /// Plans this graph, capturing `named`'s state buffers.
    ///
    /// `named` must bind every input the program reads -- the batch, the step
    /// counter and every rebind name -- exactly as
    /// [`crate::backend::plan_named_with_config`] requires. Only the rebind
    /// buffers are retained; batch bindings are supplied again to every
    /// [`TrainingStep::step`] call. Extents must be static: the symbol table is
    /// empty.
    ///
    /// # Errors
    /// [`TrainError::MissingStateBinding`] if a rebind name is absent from
    /// `named`, [`TrainError::StateBinding`] if it is not `FLOAT32`, otherwise
    /// whatever planning the selected backend rejects.
    pub fn plan(&self, named: &[(&str, QuantizedBlock<'_>)]) -> Result<TrainingStep, TrainError> {
        let mut state = Vec::with_capacity(self.rebind.len());
        for (_, name) in &self.rebind {
            let block = named
                .iter()
                .find_map(|(candidate, block)| (*candidate == name.as_str()).then_some(block))
                .ok_or_else(|| TrainError::MissingStateBinding { name: name.clone() })?;
            let QuantizedBlock::Float32(values) = *block else {
                return Err(TrainError::StateBinding { name: name.clone() });
            };
            state.push((name.clone(), values.to_vec()));
        }

        let mut outputs = Vec::with_capacity(self.rebind.len() + 1);
        outputs.push(self.loss);
        outputs.extend(self.rebind.iter().map(|(node, _)| *node));
        let (plan, _measurements) =
            plan_named_with_config(&self.runtime, &self.program, &[], named, &outputs)?;

        Ok(TrainingStep {
            plan,
            loss: self.loss,
            rebind: self.rebind.clone(),
            state,
            moment_layouts: self.moment_layouts.clone(),
        })
    }
}

/// A resolved training step: one [`Plan`] plus the state it rebinds.
///
/// Plan once, then call [`Self::step`] once per batch. The step updates the
/// state in place, so the next call needs only the new batch data and the new
/// step number.
pub struct TrainingStep {
    plan: Plan,
    loss: NodeId,
    rebind: Vec<(NodeId, String)>,
    state: Vec<(String, Vec<f32>)>,
    moment_layouts: Vec<Option<FloatLayout>>,
}

impl TrainingStep {
    /// Runs one step and returns the loss. The owned state is updated in
    /// place, so the next call passes only the fresh batch data and the new
    /// step number.
    ///
    /// # Errors
    /// [`TrainError::BatchOverridesState`] if `batch` names a state buffer,
    /// [`TrainError::MissingOutput`] if the backend omitted a requested
    /// output, otherwise whatever the chosen backend rejects.
    pub fn step(&mut self, batch: &[(&str, QuantizedBlock<'_>)]) -> Result<f32, TrainError> {
        if let Some(name) = batch.iter().find_map(|(name, _)| {
            self.state
                .iter()
                .any(|(state_name, _)| state_name.as_str() == *name)
                .then_some(*name)
        }) {
            return Err(TrainError::BatchOverridesState {
                name: name.to_owned(),
            });
        }

        let mut named = Vec::with_capacity(self.state.len() + batch.len());
        named.extend(
            self.state
                .iter()
                .map(|(name, values)| (name.as_str(), QuantizedBlock::Float32(values.as_slice()))),
        );
        named.extend(batch.iter().map(|(name, block)| (*name, *block)));

        let evaluated = execute_plan_named(&mut self.plan, &named)?;
        let loss = evaluated
            .get(self.loss)
            .and_then(|(values, _)| values.first().copied())
            .ok_or(TrainError::MissingOutput { node: self.loss })?;

        for ((slot, (node, _)), layout) in self
            .state
            .iter_mut()
            .zip(self.rebind.iter())
            .zip(self.moment_layouts.iter())
        {
            let (values, _) = evaluated
                .get(*node)
                .ok_or(TrainError::MissingOutput { node: *node })?;
            slot.1.clear();
            slot.1.extend_from_slice(values);
            if let Some(layout) = *layout {
                store_moment(&mut slot.1, layout, "moment")?;
            }
        }

        Ok(loss)
    }

    /// The state buffers this step owns, in rebind order.
    #[must_use]
    pub fn state(&self) -> &[(String, Vec<f32>)] {
        &self.state
    }

    /// One owned state buffer by name.
    #[must_use]
    pub fn state_value(&self, name: &str) -> Option<&[f32]> {
        self.state
            .iter()
            .find(|(candidate, _)| candidate == name)
            .map(|(_, values)| values.as_slice())
    }
}

/// Builds the forward+backward+optimizer program for `parameters`.
///
/// `program` is the forward graph and `loss` the scalar node inside it. Each
/// parameter must be an `Op::Input` in `program` with a static `FLOAT32`
/// shape and a dense gradient. The optimizer is `config.optimizer` (`adam` or
/// `adamw`). The returned [`TrainingProgram`] appends [`STEP_INPUT`] (unless
/// the forward program already has a rank-0 `FLOAT32` input with that name)
/// and one `m_<parameter>`/`v_<parameter>` input pair per parameter.
///
/// # Errors
/// [`TrainError::UnknownParameter`] when a name is not an `Op::Input`,
/// [`TrainError::DuplicateParameter`] when a name is listed twice,
/// [`TrainError::ReservedParameterName`] for the [`STEP_INPUT`] name,
/// [`TrainError::ParameterDtype`] when it is not `FLOAT32`,
/// [`TrainError::SymbolicParameter`] when its shape is symbolic,
/// [`TrainError::MissingGradient`] when the loss does not depend on it
/// densely, [`TrainError::StateNameCollision`] when the program already
/// defines an `m_<name>`/`v_<name>` input, [`TrainError::InvalidStepInput`]
/// when a `step` input exists but is not rank-0 `FLOAT32`, or the
/// [`TrainError::Autograd`]/[`TrainError::Optimizer`] variants when the
/// adjoint transform or the optimizer refuses the graph.
pub fn build_optimizer_step(
    program: &[Op],
    loss: NodeId,
    parameters: &[&str],
    config: &TrainConfig,
) -> Result<TrainingProgram, TrainError> {
    config.validate()?;

    let mut resolved = Vec::with_capacity(parameters.len());
    let mut seen = BTreeSet::new();
    for name in parameters {
        if !seen.insert(*name) {
            return Err(TrainError::DuplicateParameter {
                name: (*name).to_owned(),
            });
        }
        if *name == STEP_INPUT {
            return Err(TrainError::ReservedParameterName {
                name: STEP_INPUT.to_owned(),
            });
        }
        let Some((param, dtype, shape)) = named_input(program, name) else {
            return Err(TrainError::UnknownParameter {
                name: (*name).to_owned(),
            });
        };
        if dtype != DType::FLOAT32 {
            return Err(TrainError::ParameterDtype {
                name: (*name).to_owned(),
                dtype,
            });
        }
        if shape
            .iter()
            .any(|extent| matches!(extent, Extent::Symbolic(_)))
        {
            return Err(TrainError::SymbolicParameter {
                name: (*name).to_owned(),
            });
        }
        let rank = u16::try_from(shape.len()).map_err(|_| TrainError::RankOverflow {
            name: (*name).to_owned(),
        })?;
        for state_name in [adam_first_moment_name(name), adam_second_moment_name(name)] {
            if named_input(program, &state_name).is_some() {
                return Err(TrainError::StateNameCollision { name: state_name });
            }
        }
        resolved.push(((*name).to_owned(), param, shape, rank));
    }

    validate_forward_step_input(program)?;

    let wanted: Vec<NodeId> = resolved.iter().map(|(_, param, _, _)| *param).collect();
    let differentiated = differentiate_wanted(program, loss, &wanted)?;
    let gradients = resolved
        .iter()
        .map(|(name, _, _, _)| {
            differentiated
                .gradient_of_named(name)
                .ok_or_else(|| TrainError::MissingGradient { name: name.clone() })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut trained = differentiated.program;
    let step = step_node_in(&mut trained)?;
    let (first_moment_layout, second_moment_layout) = match config.optimizer.as_str() {
        "adamw" => (
            config.adamw.first_moment_layout,
            config.adamw.second_moment_layout,
        ),
        _ => (
            config.adam.first_moment_layout,
            config.adam.second_moment_layout,
        ),
    };
    let mut rebind = Vec::with_capacity(resolved.len() * 3);
    let mut moment_layouts = Vec::with_capacity(resolved.len() * 3);
    for ((name, param, shape, rank), gradient) in resolved.into_iter().zip(gradients) {
        let first_moment = append_state_input(&mut trained, &adam_first_moment_name(&name), shape);
        let second_moment =
            append_state_input(&mut trained, &adam_second_moment_name(&name), shape);
        let operands = AdamOperands {
            param,
            grad: gradient,
            m: first_moment,
            v: second_moment,
        };
        let (new_param, new_first, new_second) = match config.optimizer.as_str() {
            "adamw" => adamw_step(&mut trained, &config.adamw, rank, operands, step)?,
            _ => adam_step(&mut trained, &config.adam, rank, operands, step)?,
        };
        rebind.push((new_param, name.clone()));
        rebind.push((new_first, adam_first_moment_name(&name)));
        rebind.push((new_second, adam_second_moment_name(&name)));
        moment_layouts.push(None);
        moment_layouts.push(Some(first_moment_layout));
        moment_layouts.push(Some(second_moment_layout));
    }

    Ok(TrainingProgram {
        program: trained,
        loss,
        rebind,
        moment_layouts,
        runtime: config.runtime.clone(),
    })
}

fn named_input<'program>(
    program: &'program [Op],
    name: &str,
) -> Option<(NodeId, DType, &'program [Extent])> {
    program.iter().enumerate().find_map(|(index, op)| match op {
        Op::Input {
            dtype,
            shape,
            name: Some(candidate),
        } if candidate.as_str() == name => {
            Some((NodeId(u32::try_from(index).ok()?), *dtype, shape.as_slice()))
        }
        _ => None,
    })
}

fn validate_forward_step_input(program: &[Op]) -> Result<(), TrainError> {
    if let Some((_, dtype, shape)) = named_input(program, STEP_INPUT)
        && (dtype != DType::FLOAT32 || !shape.is_empty())
    {
        return Err(TrainError::InvalidStepInput {
            name: STEP_INPUT.to_owned(),
        });
    }
    Ok(())
}

fn step_node_in(program: &mut Vec<Op>) -> Result<NodeId, TrainError> {
    let existing =
        named_input(program, STEP_INPUT).map(|(node, dtype, shape)| (node, dtype, shape.to_vec()));
    if let Some((node, dtype, shape)) = existing {
        if dtype == DType::FLOAT32 && shape.is_empty() {
            return Ok(node);
        }
        return Err(TrainError::InvalidStepInput {
            name: STEP_INPUT.to_owned(),
        });
    }
    Ok(step_input(program, STEP_INPUT))
}

fn append_state_input(program: &mut Vec<Op>, name: &str, shape: &[Extent]) -> NodeId {
    append(
        program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: shape.to_vec(),
            name: Some(name.to_owned()),
        },
    )
}

fn finish(errors: Vec<ValidationMessage>) -> conflaguration::Result<()> {
    if errors.is_empty() {
        Ok(())
    } else {
        Err(conflaguration::Error::Validation { errors })
    }
}

#[cfg(test)]
mod tests {
    use proxima_numeric::DType;
    use proxima_tensor::map::{self, IndexMap};
    use proxima_tensor::op::{Extent, NodeId, Op, Reduce, ReduceInit, ScalarOp, append};

    use super::*;

    fn leaf(program: &mut Vec<Op>, name: &str, shape: Vec<Extent>) -> NodeId {
        append(
            program,
            Op::Input {
                dtype: DType::FLOAT32,
                shape,
                name: Some(name.to_owned()),
            },
        )
    }

    fn identity(rank: u16) -> IndexMap {
        IndexMap::Affine(map::projection(rank, &(0..rank).collect::<Vec<u16>>()))
    }

    fn scalar_loss(program: &mut Vec<Op>, x: NodeId, target: NodeId) -> NodeId {
        let diff = append(
            program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
                body: ScalarOp::Subtract,
                operands: vec![(x, identity(1)), (target, identity(1))],
                name: None,
            },
        );
        let squared = append(
            program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
                body: ScalarOp::Multiply,
                operands: vec![(diff, identity(1)), (diff, identity(1))],
                name: None,
            },
        );
        append(
            program,
            Op::Reduce(Reduce {
                dtype: DType::FLOAT32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: squared,
                in_map: identity(1),
                out_map: IndexMap::Affine(map::projection(1, &[])),
                keep: proxima_tensor::op::Keep::Reduce,
                name: None,
            }),
        )
    }

    fn scalar_program() -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let x = leaf(&mut program, "x", vec![Extent::Static(1)]);
        let target = leaf(&mut program, "target", vec![Extent::Static(1)]);
        let loss = scalar_loss(&mut program, x, target);
        (program, loss)
    }

    #[test]
    fn build_optimizer_step_appends_parameter_moments_and_the_step_input() {
        let (program, loss) = scalar_program();
        let built = build_optimizer_step(&program, loss, &["x"], &TrainConfig::default())
            .expect("a scalar parameter differentiates");

        assert_eq!(built.rebind.len(), 3);
        assert_eq!(built.rebind[0].1, "x");
        assert_eq!(built.rebind[1].1, "m_x");
        assert_eq!(built.rebind[2].1, "v_x");
        assert!(named_input(&built.program, STEP_INPUT).is_some());
        assert!(named_input(&built.program, "m_x").is_some());
        assert!(named_input(&built.program, "v_x").is_some());
    }

    #[test]
    fn build_optimizer_step_supports_adamw() {
        let (program, loss) = scalar_program();
        let config = TrainConfig::builder()
            .optimizer("adamw")
            .build()
            .expect("adamw is a valid optimizer");
        let built = build_optimizer_step(&program, loss, &["x"], &config)
            .expect("adamw builds the same graph shape");
        assert_eq!(built.rebind.len(), 3);
    }

    #[test]
    fn build_optimizer_step_rejects_an_unknown_parameter() {
        let (program, loss) = scalar_program();
        let error = build_optimizer_step(&program, loss, &["nope"], &TrainConfig::default())
            .expect_err("an unknown name is refused");
        assert!(matches!(error, TrainError::UnknownParameter { name } if name == "nope"));
    }

    #[test]
    fn build_optimizer_step_rejects_a_duplicate_parameter() {
        let (program, loss) = scalar_program();
        let error = build_optimizer_step(&program, loss, &["x", "x"], &TrainConfig::default())
            .expect_err("a duplicate parameter is refused");
        assert!(matches!(error, TrainError::DuplicateParameter { name } if name == "x"));
    }

    #[test]
    fn build_optimizer_step_rejects_a_parameter_named_step() {
        let (mut program, loss) = scalar_program();
        leaf(&mut program, STEP_INPUT, Vec::new());

        let error = build_optimizer_step(&program, loss, &[STEP_INPUT], &TrainConfig::default())
            .expect_err("the step counter is not a parameter");
        assert!(matches!(error, TrainError::ReservedParameterName { name } if name == STEP_INPUT));
    }

    #[test]
    fn build_optimizer_step_rejects_a_symbolic_parameter() {
        let mut program = Vec::new();
        let x = leaf(&mut program, "x", vec![Extent::Symbolic(0)]);
        let target = leaf(&mut program, "target", vec![Extent::Static(1)]);
        let loss = scalar_loss(&mut program, x, target);

        let error = build_optimizer_step(&program, loss, &["x"], &TrainConfig::default())
            .expect_err("a symbolic parameter is refused");
        assert!(matches!(error, TrainError::SymbolicParameter { name } if name == "x"));
    }

    #[test]
    fn build_optimizer_step_rejects_a_parameter_the_loss_does_not_read() {
        let (mut program, loss) = scalar_program();
        leaf(&mut program, "spare", vec![Extent::Static(1)]);

        let error = build_optimizer_step(&program, loss, &["spare"], &TrainConfig::default())
            .expect_err("a disconnected parameter is refused");
        assert!(matches!(error, TrainError::MissingGradient { name } if name == "spare"));
    }

    #[test]
    fn build_optimizer_step_rejects_a_colliding_state_name() {
        let (mut program, loss) = scalar_program();
        leaf(&mut program, "m_x", vec![Extent::Static(1)]);

        let error = build_optimizer_step(&program, loss, &["x"], &TrainConfig::default())
            .expect_err("a colliding state name is refused");
        assert!(matches!(error, TrainError::StateNameCollision { name } if name == "m_x"));
    }

    #[test]
    fn build_optimizer_step_rejects_a_non_scalar_step_input() {
        let (mut program, loss) = scalar_program();
        leaf(&mut program, STEP_INPUT, vec![Extent::Static(1)]);

        let error = build_optimizer_step(&program, loss, &["x"], &TrainConfig::default())
            .expect_err("a non-scalar step input is refused");
        assert!(matches!(error, TrainError::InvalidStepInput { name } if name == STEP_INPUT));
    }
}
