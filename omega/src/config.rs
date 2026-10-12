//! Runtime configuration for Omega's existing plan and backend boundaries.
//!
//! This module is the std-tier surface over the build-time defaults in
//! [`crate::sized`]. Alloc-only emission continues to consume explicit values
//! and has no environment or file access.

use conflaguration::{FromEnvStr, Settings, Validate, ValidationMessage};
use proxima_tensor::NumericPolicy;
use serde::{Deserialize, Serialize};
use std::error::Error as StdError;
use std::fmt;
use std::num::ParseIntError;
use std::path::Path;

/// Process-level defaults and per-plan policies accepted by omega.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, Settings)]
#[settings(prefix = "OMEGA")]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeConfig {
    /// Backend family for planning, such as `cpu` for the reference evaluator.
    #[setting(default = "cpu")]
    pub engine: String,
    /// GPU driver for planning, such as `metal` on Apple Silicon.
    #[setting(default = "auto")]
    pub gpu_driver: String,
    /// Permit bind-time and execution-time algebraic fusion, such as joining a product and reduce.
    #[setting(default = false)]
    pub allow_fusion: bool,
    /// Require exact activation mode where supported, such as exact CPU activation evaluation.
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
    /// Maximum CPU evaluator workers, such as `4` independent kernel workers.
    #[setting(default_str = "sized", resolve_with = "parse_cpu_workers")]
    pub cpu_workers: u16,
    /// Maximum independent operations in one dependency-ready DAG level, such as `2` sibling nodes.
    #[setting(default_str = "sized", resolve_with = "parse_independent_ops")]
    pub independent_ops: u16,
    /// Maximum live generation streams admitted to shared execution, such as `3` conversations.
    #[setting(default_str = "sized", resolve_with = "parse_parallel_sequences")]
    pub parallel_sequences: u16,
    /// Maximum submissions awaiting backend completion, such as `2` in-flight commands.
    #[setting(default_str = "sized", resolve_with = "parse_max_in_flight")]
    pub max_in_flight: u16,
    /// Allow the selected driver to overlap transfer and compute when it reports that capability.
    #[setting(default = false)]
    pub overlap_transfer_compute: bool,
    /// Hard device-residency ceiling in bytes, such as `Some(64)` for a bounded fixture.
    #[setting(default)]
    pub max_device_bytes: Option<u64>,
    /// Decode projection kernel policy, such as `automatic` for shape-selected execution.
    #[setting(default)]
    pub decode_kernel: DecodeKernelStrategy,
    /// Fusion strategy request, such as `required` when absence must reject planning.
    #[setting(default)]
    pub fusion: FusionStrategy,
    /// Cache placement policy, such as `exact_resident_first` with exact identity validation.
    #[setting(default)]
    pub residency: ResidencyStrategy,
    /// Row formation policy, such as `compatible_rows` for same-shape decode work.
    #[setting(default)]
    pub batching: BatchingStrategy,
    /// Speculative decoding policy, such as `disabled` when no verified draft is configured.
    #[setting(default)]
    pub speculation: SpeculationStrategy,
    /// Converted-weight cache policy, such as `disabled` unless conversion permission is present.
    #[setting(default)]
    pub weight_conversion: WeightConversionStrategy,
    /// Group already-routed expert rows when the model exposes MoE routing.
    #[setting(default = true)]
    pub moe_grouping: bool,
    /// Existing numeric permissions shared with binding and emission, such as contraction for an MMA path.
    #[setting(nested)]
    pub numeric: NumericPolicy,
    /// Hard byte ceilings enforced by payload allocation owners, such as a 4 KiB device-buffer cap.
    #[setting(nested)]
    pub constraints: ResourceConstraints,
    /// Cold-plan selection and optional counts, such as two warmups and seven measured runs.
    #[setting(nested)]
    pub selection: SelectionConfig,
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
            cpu_workers: default_cpu_workers(),
            independent_ops: default_independent_ops(),
            parallel_sequences: default_parallel_sequences(),
            max_in_flight: default_max_in_flight(),
            overlap_transfer_compute: false,
            max_device_bytes: None,
            decode_kernel: DecodeKernelStrategy::default(),
            fusion: FusionStrategy::default(),
            residency: ResidencyStrategy::default(),
            batching: BatchingStrategy::default(),
            speculation: SpeculationStrategy::default(),
            weight_conversion: WeightConversionStrategy::default(),
            moe_grouping: true,
            numeric: NumericPolicy::bit_exact(),
            constraints: ResourceConstraints::default(),
            selection: SelectionConfig::default(),
        }
    }
}

impl Validate for RuntimeConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();

        if !matches!(self.engine.as_str(), "cpu" | "gpu") {
            errors.push(ValidationMessage::new(
                "engine",
                format!("unknown engine '{}'; expected cpu|gpu", self.engine),
            ));
        }
        if !matches!(self.gpu_driver.as_str(), "auto" | "metal" | "wgpu" | "cuda") {
            errors.push(ValidationMessage::new(
                "gpu_driver",
                format!(
                    "unknown gpu driver '{}'; expected auto|metal|wgpu|cuda",
                    self.gpu_driver
                ),
            ));
        }

        for (field, value) in [
            ("cpu_workers", self.cpu_workers),
            ("independent_ops", self.independent_ops),
            ("parallel_sequences", self.parallel_sequences),
            ("max_in_flight", self.max_in_flight),
        ] {
            if value == 0 {
                errors.push(ValidationMessage::new(field, "must be greater than zero"));
            }
        }
        if self.max_device_bytes == Some(0) {
            errors.push(ValidationMessage::new(
                "max_device_bytes",
                "must be greater than zero",
            ));
        }

        self.constraints.append_validation_errors(&mut errors);
        self.selection.append_validation_errors(&mut errors);

        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl RuntimeConfig {
    /// Load defaults overlaid by `OMEGA_*` values and validate the resolved config.
    pub fn from_env() -> conflaguration::Result<Self> {
        let config = <Self as Settings>::from_env()?;
        config.validate()?;
        Ok(config)
    }

    /// Load a TOML/JSON/YAML file and validate the resolved config.
    pub fn from_file(path: impl AsRef<Path>) -> conflaguration::Result<Self> {
        let config: Self = conflaguration::from_file(path.as_ref())?;
        config.validate()?;
        Ok(config)
    }

    /// Start a fluent builder seeded from the runtime defaults.
    #[must_use]
    pub fn builder() -> RuntimeConfigBuilder {
        RuntimeConfigBuilder {
            inner: Self::default(),
        }
    }

    /// Move this value into a fluent builder for explicit overrides.
    #[must_use]
    pub fn into_builder(self) -> RuntimeConfigBuilder {
        RuntimeConfigBuilder { inner: self }
    }
}

fn default_cpu_workers() -> u16 {
    crate::sized::RUNTIME_CPU_WORKERS
}

fn default_independent_ops() -> u16 {
    crate::sized::RUNTIME_INDEPENDENT_OPS
}

fn default_parallel_sequences() -> u16 {
    crate::sized::RUNTIME_PARALLEL_SEQUENCES
}

fn default_max_in_flight() -> u16 {
    crate::sized::RUNTIME_MAX_IN_FLIGHT
}

fn parse_sized_u16(value: &str, sized: u16) -> Result<u16, ParseIntError> {
    if value == "sized" {
        Ok(sized)
    } else {
        value.parse()
    }
}

fn parse_cpu_workers(value: &str) -> Result<u16, ParseIntError> {
    parse_sized_u16(value, crate::sized::RUNTIME_CPU_WORKERS)
}

fn parse_independent_ops(value: &str) -> Result<u16, ParseIntError> {
    parse_sized_u16(value, crate::sized::RUNTIME_INDEPENDENT_OPS)
}

fn parse_parallel_sequences(value: &str) -> Result<u16, ParseIntError> {
    parse_sized_u16(value, crate::sized::RUNTIME_PARALLEL_SEQUENCES)
}

fn parse_max_in_flight(value: &str) -> Result<u16, ParseIntError> {
    parse_sized_u16(value, crate::sized::RUNTIME_MAX_IN_FLIGHT)
}

macro_rules! string_policy {
    ($name:ident, $error:ident, {$first_string:literal => $first_variant:ident $(, $string:literal => $variant:ident)* $(,)?}, $expected:literal) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
        #[serde(rename_all = "snake_case")]
        #[doc = concat!("Execution policy for ", stringify!($name), ".")]
        pub enum $name {
            #[doc = concat!("Select ", stringify!($first_variant), " behavior.")]
            $first_variant,
            $(#[doc = concat!("Select ", stringify!($variant), " behavior.")]
            $variant,)*
        }

        impl Default for $name {
            fn default() -> Self { Self::$first_variant }
        }

        impl FromEnvStr for $name {
            type Err = $error;

            fn from_env_str(value: &str) -> Result<Self, Self::Err> {
                match value {
                    $first_string => Ok(Self::$first_variant),
                    $($string => Ok(Self::$variant),)*
                    _ => Err($error(value.to_owned())),
                }
            }

            fn type_name() -> &'static str { stringify!($name) }
        }

        #[derive(Debug, Clone, PartialEq, Eq)]
        #[doc = concat!("Invalid value for ", stringify!($name), ".")]
        pub struct $error(String);

        impl fmt::Display for $error {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "unknown value '{}'; expected {}", self.0, $expected)
            }
        }

        impl StdError for $error {}
    };
}

string_policy!(DecodeKernelStrategy, DecodeKernelStrategyParseError, {
    "automatic" => Automatic,
    "serial_projection" => SerialProjection,
    "fused_projection" => FusedProjection,
}, "automatic|serial_projection|fused_projection");

string_policy!(FusionStrategy, FusionStrategyParseError, {
    "automatic" => Automatic,
    "disabled" => Disabled,
    "required" => Required,
}, "automatic|disabled|required");

string_policy!(ResidencyStrategy, ResidencyStrategyParseError, {
    "exact_resident_first" => ExactResidentFirst,
}, "exact_resident_first");

string_policy!(BatchingStrategy, BatchingStrategyParseError, {
    "compatible_rows" => CompatibleRows,
}, "compatible_rows");

string_policy!(SpeculationStrategy, SpeculationStrategyParseError, {
    "disabled" => Disabled,
}, "disabled");

string_policy!(WeightConversionStrategy, WeightConversionStrategyParseError, {
    "disabled" => Disabled,
}, "disabled");

/// Byte ceilings enforced at each concrete allocation owner, such as the host input tensor payload.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize, Settings)]
#[serde(default, deny_unknown_fields)]
pub struct ResourceConstraints {
    /// Maximum total owned host tensor payload bytes, for example `56` for a small CPU fixture.
    #[setting(default)]
    pub max_host_tensor_bytes: Option<u64>,
    /// Maximum total device buffer bytes, for example `4096` for a small device execution.
    #[setting(default)]
    pub max_device_buffer_bytes: Option<u64>,
    /// Maximum size of one device buffer, for example `24` for an individual uniform buffer.
    #[setting(default)]
    pub max_single_device_buffer_bytes: Option<u64>,
}

impl ResourceConstraints {
    fn append_validation_errors(&self, errors: &mut Vec<ValidationMessage>) {
        for (field, value) in [
            ("max_host_tensor_bytes", self.max_host_tensor_bytes),
            ("max_device_buffer_bytes", self.max_device_buffer_bytes),
            (
                "max_single_device_buffer_bytes",
                self.max_single_device_buffer_bytes,
            ),
        ] {
            if value == Some(0) {
                errors.push(ValidationMessage::new(field, "must be greater than zero"));
            }
        }
    }

    /// Set the host tensor payload ceiling.
    #[must_use]
    pub fn with_max_host_tensor_bytes(mut self, value: Option<u64>) -> Self {
        self.max_host_tensor_bytes = value;
        self
    }

    /// Set the total device buffer ceiling.
    #[must_use]
    pub fn with_max_device_buffer_bytes(mut self, value: Option<u64>) -> Self {
        self.max_device_buffer_bytes = value;
        self
    }

    /// Set the largest permitted single device buffer.
    #[must_use]
    pub fn with_max_single_device_buffer_bytes(mut self, value: Option<u64>) -> Self {
        self.max_single_device_buffer_bytes = value;
        self
    }
}

/// Candidate-selection mode for cold planning, such as configured-only execution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionMode {
    /// Use the configured backend without timing alternatives, such as `gpu_driver = "metal"`.
    #[default]
    Configured,
    /// Measure available candidates and choose the least admissible mean, such as CPU versus Metal.
    MinimumMeasuredWallTime,
}

impl FromEnvStr for SelectionMode {
    type Err = SelectionModeParseError;

    fn from_env_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "configured" => Ok(Self::Configured),
            "minimum_measured_wall_time" => Ok(Self::MinimumMeasuredWallTime),
            _ => Err(SelectionModeParseError(value.to_owned())),
        }
    }

    fn type_name() -> &'static str {
        "SelectionMode"
    }
}

/// Invalid value supplied for `OMEGA_SELECTION_MODE`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionModeParseError(String);

impl fmt::Display for SelectionModeParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "unknown selection mode '{}'; expected configured|minimum_measured_wall_time",
            self.0
        )
    }
}

impl StdError for SelectionModeParseError {}

/// Candidate-selection mode and optional trial-count overrides, such as two warmups and seven samples.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, Settings)]
#[serde(default, deny_unknown_fields)]
pub struct SelectionConfig {
    /// Select configured-only or measured minimum-wall-time planning, such as `minimum_measured_wall_time`.
    #[setting(default)]
    pub mode: SelectionMode,
    /// Optional warmup count, such as `Some(2)` before timed device runs.
    #[setting(default)]
    pub warmup_runs: Option<u16>,
    /// Optional sample count, such as `Some(7)` synchronized measurements.
    #[setting(default)]
    pub measurement_runs: Option<u16>,
}

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            mode: SelectionMode::Configured,
            warmup_runs: None,
            measurement_runs: None,
        }
    }
}

impl SelectionConfig {
    /// Resolve warmups from the explicit override or generated runtime floor.
    #[must_use]
    pub const fn resolved_warmup_runs(&self) -> u16 {
        match self.warmup_runs {
            Some(value) => value,
            None => crate::sized::RUNTIME_WARMUP_RUNS,
        }
    }

    /// Resolve samples from the explicit override or generated runtime floor.
    #[must_use]
    pub const fn resolved_measurement_runs(&self) -> u16 {
        match self.measurement_runs {
            Some(value) => value,
            None => crate::sized::RUNTIME_MEASUREMENT_RUNS,
        }
    }

    fn append_validation_errors(&self, errors: &mut Vec<ValidationMessage>) {
        if self.mode == SelectionMode::MinimumMeasuredWallTime {
            if self.resolved_warmup_runs() < 1 {
                errors.push(ValidationMessage::new(
                    "selection.warmup_runs",
                    "must be at least one in measured mode",
                ));
            }
            if self.resolved_measurement_runs() < 3 {
                errors.push(ValidationMessage::new(
                    "selection.measurement_runs",
                    "must be at least three in measured mode",
                ));
            }
        } else {
            if self.warmup_runs.is_some() {
                errors.push(ValidationMessage::new(
                    "selection.warmup_runs",
                    "is only consumed in measured mode",
                ));
            }
            if self.measurement_runs.is_some() {
                errors.push(ValidationMessage::new(
                    "selection.measurement_runs",
                    "is only consumed in measured mode",
                ));
            }
        }
    }

    /// Set the candidate-selection mode.
    #[must_use]
    pub fn with_mode(mut self, mode: SelectionMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set an explicit warmup count, or clear it to use the generated floor.
    #[must_use]
    pub fn with_warmup_runs(mut self, value: Option<u16>) -> Self {
        self.warmup_runs = value;
        self
    }

    /// Set an explicit measured sample count, or clear it to use the generated floor.
    #[must_use]
    pub fn with_measurement_runs(mut self, value: Option<u16>) -> Self {
        self.measurement_runs = value;
        self
    }
}

/// Fluent construction and source layering for [`RuntimeConfig`].
#[derive(Debug, Clone)]
pub struct RuntimeConfigBuilder {
    inner: RuntimeConfig,
}

impl RuntimeConfigBuilder {
    /// Apply file values to the current config; later setters win.
    pub fn from_file(mut self, path: impl AsRef<Path>) -> conflaguration::Result<Self> {
        self.inner = conflaguration::builder()
            .value(self.inner)
            .file(path)
            .validate()
            .build()?;
        Ok(self)
    }

    /// Overlay environment values onto the current config; later setters win.
    pub fn from_env(mut self) -> conflaguration::Result<Self> {
        self.inner = conflaguration::builder()
            .value(self.inner)
            .env()
            .validate()
            .build()?;
        Ok(self)
    }

    /// Set the maximum CPU worker count.
    #[must_use]
    pub fn with_cpu_workers(mut self, value: u16) -> Self {
        self.inner.cpu_workers = value;
        self
    }

    /// Set the maximum independent operations in a ready DAG level.
    #[must_use]
    pub fn with_independent_ops(mut self, value: u16) -> Self {
        self.inner.independent_ops = value;
        self
    }

    /// Set the maximum admitted generation streams.
    #[must_use]
    pub fn with_parallel_sequences(mut self, value: u16) -> Self {
        self.inner.parallel_sequences = value;
        self
    }

    /// Set the maximum backend submissions awaiting completion.
    #[must_use]
    pub fn with_max_in_flight(mut self, value: u16) -> Self {
        self.inner.max_in_flight = value;
        self
    }

    /// Set whether this execution may overlap transfers with computation.
    #[must_use]
    pub fn with_overlap_transfer_compute(mut self, value: bool) -> Self {
        self.inner.overlap_transfer_compute = value;
        self
    }

    /// Set the hard device-residency byte ceiling.
    #[must_use]
    pub fn with_max_device_bytes(mut self, value: Option<u64>) -> Self {
        self.inner.max_device_bytes = value;
        self
    }

    /// Set the decode projection kernel strategy.
    #[must_use]
    pub fn with_decode_kernel(mut self, value: DecodeKernelStrategy) -> Self {
        self.inner.decode_kernel = value;
        self
    }

    /// Set the fusion strategy request.
    #[must_use]
    pub fn with_fusion(mut self, value: FusionStrategy) -> Self {
        self.inner.fusion = value;
        self
    }

    /// Set the cache residency strategy.
    #[must_use]
    pub fn with_residency(mut self, value: ResidencyStrategy) -> Self {
        self.inner.residency = value;
        self
    }

    /// Set the compatible-row batching strategy.
    #[must_use]
    pub fn with_batching(mut self, value: BatchingStrategy) -> Self {
        self.inner.batching = value;
        self
    }

    /// Set the target-verified speculation strategy.
    #[must_use]
    pub fn with_speculation(mut self, value: SpeculationStrategy) -> Self {
        self.inner.speculation = value;
        self
    }

    /// Set the converted-weight cache strategy.
    #[must_use]
    pub fn with_weight_conversion(mut self, value: WeightConversionStrategy) -> Self {
        self.inner.weight_conversion = value;
        self
    }

    /// Set whether MoE rows group by their already-selected expert.
    #[must_use]
    pub fn with_moe_grouping(mut self, value: bool) -> Self {
        self.inner.moe_grouping = value;
        self
    }

    /// Set the backend family.
    #[must_use]
    pub fn with_engine(mut self, value: impl Into<String>) -> Self {
        self.inner.engine = value.into();
        self
    }

    /// Set the GPU driver.
    #[must_use]
    pub fn with_gpu_driver(mut self, value: impl Into<String>) -> Self {
        self.inner.gpu_driver = value.into();
        self
    }

    /// Set fusion permission.
    #[must_use]
    pub fn with_allow_fusion(mut self, value: bool) -> Self {
        self.inner.allow_fusion = value;
        self
    }

    /// Set exact-activation mode.
    #[must_use]
    pub fn with_exact_activations(mut self, value: bool) -> Self {
        self.inner.exact_activations = value;
        self
    }

    /// Set whether compiled Metal pipelines persist across processes.
    #[must_use]
    pub fn with_pipeline_cache(mut self, value: bool) -> Self {
        self.inner.pipeline_cache = value;
        self
    }

    /// Set the on-disk pipeline cache directory; empty selects the per-user directory.
    #[must_use]
    pub fn with_pipeline_cache_dir(mut self, value: impl Into<String>) -> Self {
        self.inner.pipeline_cache_dir = value.into();
        self
    }

    /// Set the byte bound the on-disk pipeline cache keeps.
    #[must_use]
    pub fn with_pipeline_cache_max_bytes(mut self, value: u64) -> Self {
        self.inner.pipeline_cache_max_bytes = value;
        self
    }

    /// Set the existing tensor numeric policy.
    #[must_use]
    pub fn with_numeric(mut self, value: NumericPolicy) -> Self {
        self.inner.numeric = value;
        self
    }

    /// Set resource ceilings.
    #[must_use]
    pub fn with_constraints(mut self, value: ResourceConstraints) -> Self {
        self.inner.constraints = value;
        self
    }

    /// Set candidate selection and optional trial counts.
    #[must_use]
    pub fn with_selection(mut self, value: SelectionConfig) -> Self {
        self.inner.selection = value;
        self
    }

    /// Validate and return the built runtime config.
    pub fn build(self) -> conflaguration::Result<RuntimeConfig> {
        self.inner.validate()?;
        Ok(self.inner)
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
        assert!(
            loaded.pipeline_cache,
            "the on-disk pipeline cache is on by default"
        );
        assert_eq!(
            loaded.pipeline_cache_dir, "",
            "an empty directory selects the per-user cache"
        );
        assert_eq!(loaded.pipeline_cache_max_bytes, 256 * 1024 * 1024);
    }

    #[test]
    fn the_pipeline_cache_knobs_come_from_omega_variables() {
        let mut variables: Vec<(&str, Option<&str>)> =
            POLICY_VARIABLES.iter().map(|name| (*name, None)).collect();
        variables.retain(|(name, _)| !name.starts_with("OMEGA_PIPELINE_CACHE"));
        variables.extend([
            ("OMEGA_PIPELINE_CACHE", Some("false")),
            (
                "OMEGA_PIPELINE_CACHE_DIR",
                Some("/var/cache/omega-pipelines"),
            ),
            ("OMEGA_PIPELINE_CACHE_MAX_BYTES", Some("1048576")),
        ]);

        let loaded = temp_env::with_vars(variables, RuntimeConfig::from_env)
            .expect("the pipeline cache variables must parse");

        assert!(!loaded.pipeline_cache);
        assert_eq!(loaded.pipeline_cache_dir, "/var/cache/omega-pipelines");
        assert_eq!(loaded.pipeline_cache_max_bytes, 1_048_576);
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod config_tests;
