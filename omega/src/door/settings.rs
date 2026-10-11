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
}

/// Where and how a door runs a lowered program, resolved to the typed values
/// the backend takes. A format's config surface (builder, file, environment)
/// resolves to one of these, so the door itself names no format's prefix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DoorSettings {
    /// CPU interpreter or GPU backend.
    pub engine: Engine,
    /// `None` lets the backend pick the driver compiled for this target.
    pub gpu_driver: Option<GpuDriver>,
    /// The rewrites the planner may apply.
    pub numeric_policy: NumericPolicy,
    /// Pin sets kept before the least recently used one is dropped, e.g. 4.
    pub max_pin_sets: usize,
}
