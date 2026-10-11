use std::path::PathBuf;

use proxima_tensor::NumericPolicy;

use crate::backend::{BackendError, Engine, GpuDriver};

/// Why a format's config string did not name a compiled-in value.
#[derive(Debug, thiserror::Error)]
pub enum DoorConfigError {
    #[error("engine: {0}")]
    Engine(#[source] BackendError),

    #[error("unknown gpu driver `{name}`; known drivers: auto, metal, wgpu")]
    UnknownGpuDriver { name: String },

    #[error("unknown numeric policy `{name}`; known policies: bit_exact, llama_relaxed, fast")]
    UnknownNumericPolicy { name: String },

    #[error("unknown compute units `{name}`; known units: cpu_only, cpu_and_neural_engine, all")]
    UnknownComputeUnits { name: String },

    #[error("unknown precision `{name}`; known precisions: float16, float32")]
    UnknownPrecision { name: String },

    #[error(
        "precision `{precision}` cannot run on compute units `{units}`; the neural engine computes float16 only"
    )]
    PrecisionUnsupported {
        precision: &'static str,
        units: &'static str,
    },
}

/// Which Core ML devices a lane may place operations on. There is no
/// neural-engine-only unit in the framework: `CpuAndNeuralEngine` allows CPU
/// fallback, so how much ran on the engine is measured, not configured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeUnits {
    CpuOnly,
    CpuAndNeuralEngine,
    All,
}

impl ComputeUnits {
    /// The unit named by a config string, e.g. `cpu_and_neural_engine`.
    ///
    /// # Errors
    /// [`DoorConfigError::UnknownComputeUnits`] for any other name.
    pub fn parse(name: &str) -> Result<Self, DoorConfigError> {
        match name {
            "cpu_only" => Ok(Self::CpuOnly),
            "cpu_and_neural_engine" => Ok(Self::CpuAndNeuralEngine),
            "all" => Ok(Self::All),
            other => Err(DoorConfigError::UnknownComputeUnits {
                name: other.to_owned(),
            }),
        }
    }

    /// The config string that [`ComputeUnits::parse`] reads back.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::CpuOnly => "cpu_only",
            Self::CpuAndNeuralEngine => "cpu_and_neural_engine",
            Self::All => "all",
        }
    }
}

/// The floating-point width a lane computes its interior in; inputs and
/// outputs stay `f32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    Float16,
    Float32,
}

impl Precision {
    /// The precision named by a config string, e.g. `float16`.
    ///
    /// # Errors
    /// [`DoorConfigError::UnknownPrecision`] for any other name.
    pub fn parse(name: &str) -> Result<Self, DoorConfigError> {
        match name {
            "float16" => Ok(Self::Float16),
            "float32" => Ok(Self::Float32),
            other => Err(DoorConfigError::UnknownPrecision {
                name: other.to_owned(),
            }),
        }
    }

    /// The config string that [`Precision::parse`] reads back.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Float16 => "float16",
            Self::Float32 => "float32",
        }
    }

    /// Refuses a pairing the device cannot honour instead of letting the
    /// framework silently compute in another width.
    ///
    /// # Errors
    /// [`DoorConfigError::PrecisionUnsupported`] for `float32` on
    /// `cpu_and_neural_engine`.
    pub const fn check(self, units: ComputeUnits) -> Result<(), DoorConfigError> {
        match (self, units) {
            (Self::Float32, ComputeUnits::CpuAndNeuralEngine) => {
                Err(DoorConfigError::PrecisionUnsupported {
                    precision: self.name(),
                    units: units.name(),
                })
            }
            _ => Ok(()),
        }
    }
}

/// Where and how a door runs a lowered program, resolved to the typed values
/// the backend takes. A format's config surface (builder, file, environment)
/// resolves to one of these, so the door itself names no format's prefix.
#[derive(Debug, Clone, PartialEq)]
pub struct DoorSettings {
    /// CPU interpreter or GPU backend.
    pub engine: Engine,
    /// `None` lets the backend pick the driver compiled for this target.
    pub gpu_driver: Option<GpuDriver>,
    /// The rewrites the planner may apply.
    pub numeric_policy: NumericPolicy,
    /// Pin sets kept before the least recently used one is dropped, e.g. 4.
    pub max_pin_sets: usize,
    /// Devices a Core ML lane may place operations on.
    pub compute_units: ComputeUnits,
    /// Interior width of a Core ML lane.
    pub precision: Precision,
    /// Where compiled packages are kept, e.g. `/Users/me/.cache/proxima/coreml-lane`;
    /// `None` lets the lane pick its default beneath the user cache.
    pub model_cache_dir: Option<PathBuf>,
}
