//! The serving settings that were measured per checkpoint rather than stated
//! by it: whether to lower the speculative verify program, the decode-time KV
//! cache shape, the command-buffer split count and the activation format.
//!
//! These are not model facts, so they live beside [`crate::ServingSettings`]
//! rather than in a [`proxima_tensor::spec::FamilyProfile`]; a new architecture
//! needs no entry and takes [`MeasuredServingDefaults::default`]. Each value was
//! measured on the checkpoints of the architecture string that keys it
//! (`general.architecture` on the GGUF path, `model_type` on the HF path), and
//! no structural property of the header separates those architectures from the
//! ones left at the default, so the key is the measured architecture, never a
//! guess from shape. An explicit `ServingConfig` value always wins.
//!
//! Teaching pointer: the table is `serving_defaults.toml`, parsed with `toml`
//! the way [`crate::profiles::family_profile`] parses a profile; the fluent
//! surface is [`MeasuredServingDefaults::builder`]; descriptors take the values
//! through [`MeasuredServingDefaults::apply_to_descriptor`].

use std::collections::BTreeMap;

use bon::Builder;
use proxima_numeric::{DType, NumericFormat};
use proxima_telemetry::debug;
use proxima_tensor::spec::{KvCacheShape, ModelDescriptor};
use serde::Deserialize;

use crate::error::InteropError;
use crate::serving::ServingConfig;
use crate::serving_settings::DispatchTypeName;

const MEASURED_TABLE: &str = include_str!("serving_defaults.toml");

const fn default_command_buffer_chunks() -> u32 {
    1
}

const fn default_dispatch() -> DispatchTypeName {
    DispatchTypeName::Serial
}

const fn default_activation() -> NumericFormat {
    DType::FLOAT32.numeric_format()
}

/// The measured serving settings for one architecture.
#[derive(Debug, Clone, Copy, PartialEq, Builder, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredServingDefaults {
    /// Lower the speculative verify program when the caller supplies no
    /// descriptor of its own; set only where the measured verify step costs
    /// less than the tokens its drafts save.
    #[serde(default)]
    #[builder(default)]
    pub speculative_verify: bool,
    /// The decode-time cache shape the lowered program needs from the runtime
    /// loop, read once per load; anything other than a uniform shape keeps the
    /// checkpoint off the placed single-range cached program.
    #[serde(default)]
    #[builder(default)]
    pub kv_cache_shape: KvCacheShape,
    /// The number of command-buffer chunks one decode step is submitted in; at
    /// least one, and an explicit caller value always wins.
    #[serde(default = "default_command_buffer_chunks")]
    #[builder(default = default_command_buffer_chunks())]
    pub command_buffer_chunks: u32,
    /// The Metal dispatch encoding one decode step runs under, `serial` or
    /// `concurrent`; an explicit non-default caller value or `PROXIMA_DISPATCH`
    /// always wins.
    #[serde(default = "default_dispatch")]
    #[builder(default = default_dispatch())]
    pub dispatch: DispatchTypeName,
    /// Whether decode plans run the plan-time wavefront (dependency level)
    /// schedule; `None` leaves the caller's `ServingConfig::wavefront_schedule`
    /// alone (unmeasured), and `PROXIMA_WAVEFRONT` always wins. Unset for every
    /// architecture until a model A/B measures it.
    #[serde(default)]
    pub wavefront_schedule: Option<bool>,
    /// The numeric format activations and intermediate values are produced at,
    /// as a proxima-numeric format name such as `float32`.
    #[serde(default = "default_activation")]
    #[builder(default = default_activation())]
    pub activation: NumericFormat,
}

impl Default for MeasuredServingDefaults {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl MeasuredServingDefaults {
    /// These defaults with every `Some` override on `config` applied; the
    /// operator's per-deployment choice over the measured value.
    #[must_use]
    pub fn with_overrides(self, config: &ServingConfig<'_>) -> Self {
        Self {
            speculative_verify: config
                .speculative_verify_override
                .unwrap_or(self.speculative_verify),
            kv_cache_shape: config.kv_cache_shape_override.unwrap_or(self.kv_cache_shape),
            activation: config.activation_override.unwrap_or(self.activation),
            ..self
        }
    }

    /// `descriptor` with the verify arming and activation format this
    /// architecture was measured with; the descriptor builders in
    /// `proxima-tensor` emit neutral values for both.
    #[must_use]
    pub fn apply_to_descriptor(&self, descriptor: ModelDescriptor) -> ModelDescriptor {
        ModelDescriptor {
            speculative_verify: self.speculative_verify,
            activation: self.activation,
            ..descriptor
        }
    }
}

/// The measured defaults for `architecture`: its table in
/// `serving_defaults.toml`, or [`MeasuredServingDefaults::default`] when it has
/// none.
///
/// # Errors
///
/// [`InteropError::InvalidFamilyProfile`] when the embedded table does not parse.
pub fn measured_serving_defaults(
    architecture: &str,
) -> Result<MeasuredServingDefaults, InteropError> {
    let table: BTreeMap<String, MeasuredServingDefaults> =
        toml::from_str(MEASURED_TABLE).map_err(|error| InteropError::InvalidFamilyProfile {
            family: architecture.into(),
            message: error.to_string(),
        })?;
    Ok(table.get(architecture).copied().unwrap_or_else(|| {
        debug!(
            architecture = ?architecture,
            "no measured serving defaults for this architecture, using the type defaults"
        );
        MeasuredServingDefaults::default()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiles::{FAMILY_PROFILES, family_profile};

    const SERVING_BEFORE: &[(&str, bool, KvCacheShape, u32)] = &[
        // each row is the value `proxima-model-interop/src/profiles/<family>.toml`
        // gave at 161f9c89; a family absent from that file's keys gave the
        // serde default (false, uniform, 1, float32).
        // gemma4.toml:5 speculative_verify = true, :7 custom, :8 chunks 8
        ("gemma4", true, KvCacheShape::Custom, 8),
        // flint.toml:6 custom, :7 chunks 8
        ("flint", false, KvCacheShape::Custom, 8),
        ("llama", false, KvCacheShape::Uniform, 1),
        ("mistral", false, KvCacheShape::Uniform, 1),
        ("mixtral", false, KvCacheShape::Uniform, 1),
        // granitemoe.toml:4 chunks 8
        ("granitemoe", false, KvCacheShape::Uniform, 8),
        ("qwen2", false, KvCacheShape::Uniform, 1),
        ("qwen3", false, KvCacheShape::Uniform, 1),
        ("qwen3moe", false, KvCacheShape::Uniform, 1),
        ("qwen3_moe", false, KvCacheShape::Uniform, 1),
        // qwen35.toml:5 custom
        ("qwen35", false, KvCacheShape::Custom, 1),
        // qwen35moe.toml:5 monolithic
        ("qwen35moe", false, KvCacheShape::Monolithic, 1),
        ("lfm2", false, KvCacheShape::Uniform, 1),
        ("lfm2moe", false, KvCacheShape::Uniform, 1),
        // ouro.toml:5 custom
        ("ouro", false, KvCacheShape::Custom, 1),
    ];

    #[test]
    fn every_compiled_family_keeps_the_serving_values_its_profile_gave() {
        assert_eq!(SERVING_BEFORE.len(), FAMILY_PROFILES.len());
        for (family, speculative_verify, kv_cache_shape, chunks) in SERVING_BEFORE {
            assert!(
                family_profile(family).is_ok(),
                "{family} is one of the compiled families"
            );
            let after = measured_serving_defaults(family).expect("embedded table parses");

            assert_eq!(after.speculative_verify, *speculative_verify, "{family} verify");
            assert_eq!(after.kv_cache_shape, *kv_cache_shape, "{family} cache shape");
            assert_eq!(after.command_buffer_chunks, *chunks, "{family} chunks");
            assert_eq!(
                after.activation,
                DType::FLOAT32.numeric_format(),
                "{family} activation"
            );
        }
    }

    #[test]
    fn an_architecture_with_no_table_takes_the_type_defaults() {
        let after = measured_serving_defaults("granite").expect("embedded table parses");

        assert_eq!(after, MeasuredServingDefaults::default());
        assert!(!after.speculative_verify);
        assert_eq!(after.command_buffer_chunks, 1);
    }

    #[test]
    fn a_serving_override_replaces_the_measured_value_and_none_keeps_it() {
        let measured = measured_serving_defaults("gemma4").expect("embedded table parses");
        let untouched = measured.with_overrides(&ServingConfig::default());
        let disarmed = measured.with_overrides(&ServingConfig {
            speculative_verify_override: Some(false),
            kv_cache_shape_override: Some(KvCacheShape::Uniform),
            activation_override: Some(NumericFormat::Float16),
            ..ServingConfig::default()
        });

        assert_eq!(untouched, measured);
        assert!(!disarmed.speculative_verify);
        assert_eq!(disarmed.kv_cache_shape, KvCacheShape::Uniform);
        assert_eq!(disarmed.activation, NumericFormat::Float16);
        assert_eq!(disarmed.command_buffer_chunks, 8);
    }

    #[test]
    fn only_flint_is_measured_to_dispatch_concurrent() {
        for (family, _, _, _) in SERVING_BEFORE {
            let measured = measured_serving_defaults(family).expect("embedded table parses");
            let expected = if *family == "flint" {
                DispatchTypeName::Concurrent
            } else {
                DispatchTypeName::Serial
            };

            assert_eq!(measured.dispatch, expected, "{family} dispatch");
        }
    }

    #[test]
    fn an_unlisted_architecture_dispatches_the_global_serial_default() {
        let measured = measured_serving_defaults("granite").expect("embedded table parses");

        assert_eq!(measured.dispatch, DispatchTypeName::Serial);
        assert_eq!(
            DispatchTypeName::Serial.or_measured(measured.dispatch),
            DispatchTypeName::Serial
        );
    }

    #[test]
    fn the_default_serial_takes_the_measured_dispatch_and_an_explicit_concurrent_is_kept() {
        let flint = measured_serving_defaults("flint").expect("embedded table parses");
        let gemma4 = measured_serving_defaults("gemma4").expect("embedded table parses");

        assert_eq!(
            DispatchTypeName::Serial.or_measured(flint.dispatch),
            DispatchTypeName::Concurrent
        );
        assert_eq!(
            DispatchTypeName::Concurrent.or_measured(gemma4.dispatch),
            DispatchTypeName::Concurrent
        );
    }

    #[test]
    fn the_builder_overrides_one_field_and_keeps_the_rest() {
        let tuned = MeasuredServingDefaults::builder()
            .command_buffer_chunks(4)
            .build();

        assert_eq!(tuned.command_buffer_chunks, 4);
        assert_eq!(tuned.kv_cache_shape, KvCacheShape::Uniform);
    }

    #[test]
    fn toml_and_builder_build_the_same_gemma4_defaults() {
        let from_builder = MeasuredServingDefaults::builder()
            .speculative_verify(true)
            .kv_cache_shape(KvCacheShape::Custom)
            .command_buffer_chunks(8)
            .build();

        assert_eq!(
            measured_serving_defaults("gemma4").expect("embedded table parses"),
            from_builder
        );
    }
}
