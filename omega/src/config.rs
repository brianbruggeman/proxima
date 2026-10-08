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
    /// Persist compiled Metal pipelines across processes. A cache entry is
    /// named by a digest of the kernel source, entry point, compile options,
    /// device name, OS version and OS build, so an entry from another
    /// toolchain or device is never read.
    #[setting(default = true)]
    pub pipeline_cache: bool,
    /// Directory holding the on-disk pipeline cache; an empty string selects
    /// the per-user cache directory (`~/Library/Caches/proxima/omega-pipelines`).
    #[setting(default = "")]
    pub pipeline_cache_dir: String,
    /// Upper bound on the bytes the on-disk pipeline cache keeps; the oldest
    /// entries by modification time are removed after a write passes it.
    #[setting(default = 268435456)]
    pub pipeline_cache_max_bytes: u64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            engine: String::from("cpu"),
            gpu_driver: String::from("auto"),
            allow_fusion: false,
            exact_activations: true,
            pipeline_cache: true,
            pipeline_cache_dir: String::new(),
            pipeline_cache_max_bytes: 268_435_456,
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const POLICY_VARIABLES: [&str; 7] = [
        "OMEGA_ENGINE",
        "OMEGA_GPU_DRIVER",
        "OMEGA_ALLOW_FUSION",
        "OMEGA_EXACT_ACTIVATIONS",
        "OMEGA_PIPELINE_CACHE",
        "OMEGA_PIPELINE_CACHE_DIR",
        "OMEGA_PIPELINE_CACHE_MAX_BYTES",
    ];

    #[test]
    fn an_environment_with_no_omega_variables_loads_the_default_policy() {
        let unset: Vec<(&str, Option<&str>)> =
            POLICY_VARIABLES.iter().map(|name| (*name, None)).collect();

        let loaded = temp_env::with_vars(unset, RuntimeConfig::from_env)
            .expect("an empty environment must load the defaults");

        assert_eq!(loaded, RuntimeConfig::default());
        assert!(loaded.pipeline_cache, "the on-disk pipeline cache is on by default");
        assert_eq!(loaded.pipeline_cache_dir, "", "an empty directory selects the per-user cache");
        assert_eq!(loaded.pipeline_cache_max_bytes, 256 * 1024 * 1024);
    }

    #[test]
    fn the_pipeline_cache_knobs_come_from_omega_variables() {
        let mut variables: Vec<(&str, Option<&str>)> =
            POLICY_VARIABLES.iter().map(|name| (*name, None)).collect();
        variables.retain(|(name, _)| !name.starts_with("OMEGA_PIPELINE_CACHE"));
        variables.extend([
            ("OMEGA_PIPELINE_CACHE", Some("false")),
            ("OMEGA_PIPELINE_CACHE_DIR", Some("/var/cache/omega-pipelines")),
            ("OMEGA_PIPELINE_CACHE_MAX_BYTES", Some("1048576")),
        ]);

        let loaded = temp_env::with_vars(variables, RuntimeConfig::from_env)
            .expect("the pipeline cache variables must parse");

        assert!(!loaded.pipeline_cache);
        assert_eq!(loaded.pipeline_cache_dir, "/var/cache/omega-pipelines");
        assert_eq!(loaded.pipeline_cache_max_bytes, 1_048_576);
    }
}
