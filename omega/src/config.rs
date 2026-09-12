//! Typed runtime policy for backend selection and correctness-sensitive rewrites.
//!
//! This is the std/alloc configuration tier. The alloc-only emitter remains
//! free of environment and filesystem access; callers pass the resolved
//! policy explicitly to the backend boundary.

use conflaguration::Settings;
use serde::{Deserialize, Serialize};

/// Process/runtime policy loaded from `OMEGA_*` variables or a config file.
///
/// Fusion is opt-in until each rewrite has a model-family parity proof. This
/// default is deliberately correctness-first and can be changed per run.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, Settings)]
#[settings(prefix = "OMEGA")]
pub struct RuntimeConfig {
    /// `cpu` or `gpu`.
    #[setting(default = "cpu")]
    pub engine: String,
    /// `cuda`, `vulkan`, `metal`, or `auto`.
    #[setting(default = "auto")]
    pub gpu_driver: String,
    /// Permit bind-time and execution-time algebraic fusion.
    #[setting(default = false)]
    pub allow_fusion: bool,
    /// Require exact activation mode where the selected evaluator supports it.
    #[setting(default = true)]
    pub exact_activations: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            engine: String::from("cpu"),
            gpu_driver: String::from("auto"),
            allow_fusion: false,
            exact_activations: true,
        }
    }
}

impl RuntimeConfig {
    /// Load the typed runtime policy from defaults overlaid by `OMEGA_*`.
    pub fn from_env() -> Result<Self, conflaguration::Error> {
        <Self as Settings>::from_env()
    }

    /// Load the typed runtime policy from a TOML/JSON/YAML file.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, conflaguration::Error> {
        conflaguration::from_file(path.as_ref())
    }
}
