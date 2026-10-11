use std::path::PathBuf;

use bon::Builder;
use conflaguration::Settings;
use proxima_tensor::NumericPolicy;
use serde::{Deserialize, Serialize};

use crate::backend::{Engine, GpuDriver};
use crate::door::{ComputeUnits, DoorConfigError, DoorSettings, Precision};
use crate::sized;

/// Where and how the door runs a lowered graph.
///
/// One value is both the `DoorConfig::builder()` result and the
/// `OMEGA_ONNX_DOOR_*` / file configuration, the same pairing
/// [`RuntimeConfig`](crate::config::RuntimeConfig) uses: names are strings so
/// the value serialises, and the `resolve_*` methods turn them into the typed
/// [`Engine`], [`GpuDriver`] and [`NumericPolicy`] the backend takes.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
#[serde(default)]
#[settings(prefix = "OMEGA_ONNX_DOOR")]
#[builder(state_mod(vis = "pub"))]
pub struct DoorConfig {
    /// `cpu` or `gpu`.
    #[setting(default = "cpu")]
    #[builder(into, overwritable, default = String::from("cpu"))]
    pub engine: String,
    /// `auto`, `metal` or `wgpu`; read only when `engine` is `gpu`.
    #[setting(default = "auto")]
    #[builder(into, overwritable, default = String::from("auto"))]
    pub gpu_driver: String,
    /// `bit_exact`, `llama_relaxed` or `fast`: the rewrites the planner may
    /// apply, as the `NumericPolicy` preset of the same name.
    #[setting(default = "bit_exact")]
    #[builder(into, overwritable, default = String::from("bit_exact"))]
    pub numeric_policy: String,
    /// Pin sets the door keeps a plan for before it drops the least recently
    /// used one; 0 selects `[onnx_door].max_pin_sets` of `omega-runtime.toml`,
    /// the build-time value, so the number lives in one place.
    #[setting(default = 0)]
    #[builder(overwritable, default = 0)]
    pub max_pin_sets: usize,
    /// `cpu_only`, `cpu_and_neural_engine` or `all`: the Core ML devices a lane may place operations on.
    #[setting(default = "cpu_and_neural_engine")]
    #[builder(into, overwritable, default = String::from("cpu_and_neural_engine"))]
    pub compute_units: String,
    /// `float16` or `float32`: the interior width of a Core ML lane; the neural engine computes `float16` only.
    #[setting(default = "float16")]
    #[builder(into, overwritable, default = String::from("float16"))]
    pub precision: String,
    /// Directory a lane keeps compiled packages in; empty lets the lane pick its default beneath the user cache.
    #[setting(default = "")]
    #[builder(into, overwritable, default = String::new())]
    pub model_cache_dir: String,
}

impl Default for DoorConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl DoorConfig {
    /// A builder seeded from `config`, so a loaded file or environment can be
    /// tweaked fluently.
    pub fn builder_from(config: &Self) -> DoorConfigBuilder<impl door_config_builder::IsComplete> {
        Self::builder()
            .engine(config.engine.as_str())
            .gpu_driver(config.gpu_driver.as_str())
            .numeric_policy(config.numeric_policy.as_str())
            .max_pin_sets(config.max_pin_sets)
            .compute_units(config.compute_units.as_str())
            .precision(config.precision.as_str())
            .model_cache_dir(config.model_cache_dir.as_str())
    }

    /// Defaults overlaid by `OMEGA_ONNX_DOOR_*` variables.
    ///
    /// # Errors
    /// Whatever [`conflaguration`] rejects in the environment.
    pub fn from_env() -> Result<Self, conflaguration::Error> {
        <Self as Settings>::from_env()
    }

    /// Loads a TOML, JSON or YAML file.
    ///
    /// # Errors
    /// Whatever [`conflaguration`] rejects in the file.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, conflaguration::Error> {
        conflaguration::from_file(path.as_ref())
    }

    /// The [`Engine`] named by `engine`.
    ///
    /// # Errors
    /// [`DoorConfigError::Engine`] when the name is neither `cpu` nor `gpu`.
    pub fn resolve_engine(&self) -> Result<Engine, DoorConfigError> {
        self.engine
            .parse::<Engine>()
            .map_err(DoorConfigError::Engine)
    }

    /// The driver named by `gpu_driver`; `auto` is `None`, which lets the
    /// backend pick the driver compiled for this target.
    ///
    /// # Errors
    /// [`DoorConfigError::UnknownGpuDriver`] for any other name.
    pub fn resolve_gpu_driver(&self) -> Result<Option<GpuDriver>, DoorConfigError> {
        match self.gpu_driver.as_str() {
            "auto" => Ok(None),
            "metal" => Ok(Some(GpuDriver::Metal)),
            "wgpu" => Ok(Some(GpuDriver::Wgpu)),
            other => Err(DoorConfigError::UnknownGpuDriver {
                name: other.to_owned(),
            }),
        }
    }

    /// The [`NumericPolicy`] preset named by `numeric_policy`.
    ///
    /// # Errors
    /// [`DoorConfigError::UnknownNumericPolicy`] for any other name.
    pub fn resolve_numeric_policy(&self) -> Result<NumericPolicy, DoorConfigError> {
        match self.numeric_policy.as_str() {
            "bit_exact" => Ok(NumericPolicy::bit_exact()),
            "llama_relaxed" => Ok(NumericPolicy::llama_relaxed()),
            "fast" => Ok(NumericPolicy::fast()),
            other => Err(DoorConfigError::UnknownNumericPolicy {
                name: other.to_owned(),
            }),
        }
    }

    /// Every field resolved to the typed values the one generic door takes.
    ///
    /// # Errors
    /// The first [`DoorConfigError`] any field raises.
    pub fn settings(&self) -> Result<DoorSettings, DoorConfigError> {
        let compute_units = self.resolve_compute_units()?;
        let precision = self.resolve_precision()?;
        precision.check(compute_units)?;
        Ok(DoorSettings {
            engine: self.resolve_engine()?,
            gpu_driver: self.resolve_gpu_driver()?,
            numeric_policy: self.resolve_numeric_policy()?,
            max_pin_sets: self.resolve_max_pin_sets(),
            compute_units,
            precision,
            model_cache_dir: self.resolve_model_cache_dir(),
        })
    }

    /// The [`ComputeUnits`] named by `compute_units`.
    ///
    /// # Errors
    /// [`DoorConfigError::UnknownComputeUnits`] for any unlisted name.
    pub fn resolve_compute_units(&self) -> Result<ComputeUnits, DoorConfigError> {
        ComputeUnits::parse(&self.compute_units)
    }

    /// The [`Precision`] named by `precision`.
    ///
    /// # Errors
    /// [`DoorConfigError::UnknownPrecision`] for any unlisted name.
    pub fn resolve_precision(&self) -> Result<Precision, DoorConfigError> {
        Precision::parse(&self.precision)
    }

    /// The package directory: `None` when `model_cache_dir` is empty.
    #[must_use]
    pub fn resolve_model_cache_dir(&self) -> Option<PathBuf> {
        (!self.model_cache_dir.is_empty()).then(|| PathBuf::from(&self.model_cache_dir))
    }

    /// The pin-set cap: `max_pin_sets`, or the build-time
    /// `[onnx_door].max_pin_sets` when it is 0.
    #[must_use]
    pub const fn resolve_max_pin_sets(&self) -> usize {
        match self.max_pin_sets {
            0 => sized::ONNX_DOOR_MAX_PIN_SETS,
            count => count,
        }
    }
}
