//! The [`FamilyProfile`] of an architecture with no compiled row: the
//! documented dense defaults plus the run-time [`ArchConfig`] file the
//! architecture names, never another family's values.
//!
//! Teaching pointer: [`ArchConfig::load`] supplies the three correctness facts
//! no checkpoint states; `crate::ffn_inference::with_inferred_ffn` and
//! `crate::dense::descriptor_from_gguf`'s header scales then overlay what the
//! checkpoint does state (FFN tensors, `attention.scale`, `embedding_scale`,
//! `residual_scale`). This module only fills the fields neither source reaches
//! with the defaults a dense llama-shaped model has, and logs each one.

use std::path::PathBuf;

use bon::Builder;
use conflaguration::{Settings, Validate};
use proxima_numeric::DType;
use proxima_telemetry::debug;
use proxima_tensor::spec::{
    EmbeddingScale, ExpertGatingFunc, FamilyProfile, FfnCombination,
    KvCacheShape, LayerFfnConfig, ScheduleSource,
};
use serde::{Deserialize, Serialize};

use crate::arch_config::{ArchConfig, EmbeddingScaleOverride, ScoreScaleOverride};
use crate::error::InteropError;

/// Where the run-time architecture config files live. `PROXIMA_ARCH_CONFIG_DIR`
/// sets it; unset means the data directory this crate ships (`src/arch`, one
/// `<general.architecture>.toml` per example), resolved by [`Self::directory`].
/// An unreadable directory is refused naming the path, never replaced by an
/// embedded copy.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings, Validate)]
#[settings(prefix = "PROXIMA_ARCH_CONFIG")]
#[builder(derive(Clone, Debug))]
pub struct ArchConfigSettings {
    /// A directory of `<architecture>.toml` files, for example
    /// `/etc/proxima/arch`. Absent selects the shipped data directory.
    pub dir: Option<String>,
}

impl ArchConfigSettings {
    /// The configured directory, or the shipped data directory.
    #[must_use]
    pub fn directory(&self) -> PathBuf {
        match &self.dir {
            Some(dir) => PathBuf::from(dir),
            None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src").join("arch"),
        }
    }
}

/// `family`'s profile from the config directory [`ArchConfigSettings`]
/// resolves.
///
/// # Errors
///
/// [`InteropError::MissingArchField`] naming every required field the file
/// lacks (all three when the file is absent); [`InteropError::InvalidFamilyProfile`]
/// when the settings, the directory or the file cannot be read.
pub(crate) fn unknown_family_profile(family: &str) -> Result<FamilyProfile, InteropError> {
    let settings =
        ArchConfigSettings::from_env().map_err(|error| InteropError::InvalidFamilyProfile {
            family: family.into(),
            message: error.to_string(),
        })?;
    let config = ArchConfig::load(&settings.directory(), family)?;
    Ok(profile_from_arch_config(family, &config))
}

/// The dense-default profile over `config`'s required fields and overrides.
/// Each value taken from a default, not from `config`, is logged.
pub(crate) fn profile_from_arch_config(family: &str, config: &ArchConfig) -> FamilyProfile {
    let embedding_scale = match config.embedding_scale {
        Some(EmbeddingScaleOverride::Sqrt) => Some(EmbeddingScale::Sqrt),
        None => {
            log_default(family, "embedding_scale", "none");
            None
        }
    };
    let score_scale_inverse_sqrt_head_dim = match config.score_scale {
        Some(ScoreScaleOverride::Unscaled) => false,
        None => {
            log_default(family, "score_scale", "inverse_sqrt_head_dim");
            true
        }
    };
    let routed_gating = config.routed_gating.unwrap_or_else(|| {
        log_default(family, "routed_gating", "softmax");
        ExpertGatingFunc::Softmax
    });
    log_default(family, "ffn.combination", "exclusive");
    log_default(family, "schedule_source", "uniform");
    log_default(family, "kv_cache_shape", "uniform");
    log_default(family, "command_buffer_chunks", "1");
    log_default(family, "speculative_verify", "false");
    log_default(family, "activation_format", "float32");
    FamilyProfile {
        embedding_scale,
        ffn: LayerFfnConfig {
            post_attention_norm: false,
            combination: FfnCombination::Exclusive,
            output_scale: false,
            routed_gating,
            routed_expert_bias: false,
            activation: config.ffn_activation,
            dense_feed_forward: None,
            exclusive_dense_post_norm: false,
            ple: false,
            shared_expert_gate: true,
        },
        parallel_dense_moe: None,
        score_scale_inverse_sqrt_head_dim,
        value_norm: config.value_norm,
        qk_norm: true,
        rope_layout: config.rope_layout,
        recurrence: None,
        speculative_verify: false,
        schedule_source: ScheduleSource::Uniform,
        kv_cache_shape: KvCacheShape::Uniform,
        command_buffer_chunks: 1,
        ssm_v_head_reordered: false,
        activation: DType::FLOAT32.numeric_format(),
    }
}

fn log_default(family: &str, field: &'static str, value: &'static str) {
    debug!(
        family = ?family,
        field = field,
        value = value,
        "default, not stated by the arch config"
    );
}

#[cfg(test)]
mod tests {
    use std::fs;

    use proxima_gguf::value::{MetadataArray, MetadataValue};
    use proxima_tensor::spec::{Activation, RopeLayout, RopePairing};
    use tempfile::TempDir;

    use super::*;

    fn dir_with(name: &str, text: &str) -> TempDir {
        let dir = tempfile::tempdir().expect("tempdir for arch config");
        fs::write(dir.path().join(format!("{name}.toml")), text).expect("write arch file");
        dir
    }

    fn profile_in(dir: &TempDir, family: &str) -> Result<FamilyProfile, InteropError> {
        let dir_text = dir.path().to_str().expect("tempdir path is utf-8");
        temp_env::with_var("PROXIMA_ARCH_CONFIG_DIR", Some(dir_text), || {
            unknown_family_profile(family)
        })
    }

    #[test]
    fn an_unknown_architecture_takes_its_three_fields_from_the_file() {
        let dir = dir_with(
            "phi9",
            "rope_layout = \"split_half\"\nffn_activation = \"GeluTanh\"\nvalue_norm = true\n",
        );

        let profile = profile_in(&dir, "phi9").expect("complete file loads");

        assert_eq!(profile.rope_layout, RopeLayout::SplitHalf);
        assert_eq!(profile.ffn.activation, Activation::GeluTanh);
        assert!(profile.value_norm);
        assert_eq!(profile.schedule_source, ScheduleSource::Uniform);
        assert_eq!(profile.embedding_scale, None);
        assert!(profile.score_scale_inverse_sqrt_head_dim);
    }

    #[test]
    fn an_unknown_architecture_with_no_file_is_refused_naming_all_three_fields() {
        let dir = tempfile::tempdir().expect("empty config dir");

        match profile_in(&dir, "phi9") {
            Err(InteropError::MissingArchField { architecture, field }) => {
                assert_eq!(architecture, "phi9");
                assert_eq!(field, "rope_layout, ffn_activation, value_norm");
            }
            other => panic!("expected MissingArchField, got {other:?}"),
        }
    }

    fn granite_4_header() -> proxima_gguf::pipe::ParsedGguf {
        let names: Vec<(&str, u64)> = include_str!("../../tests/fixtures/tensor_names/granite4.txt")
            .lines()
            .filter_map(|line| line.strip_prefix("tensor name="))
            .filter(|name| *name != "token_embd.weight")
            .map(|name| (name, 1))
            .collect();
        let u32_key = |key: &'static str, value: u32| (key, MetadataValue::U32(value));
        let f32_key = |key: &'static str, value: f32| (key, MetadataValue::F32(value));
        let metadata = vec![
            ("general.architecture", MetadataValue::String("granite".into())),
            u32_key("granite.block_count", 28),
            u32_key("granite.embedding_length", 1024),
            u32_key("granite.feed_forward_length", 2048),
            u32_key("granite.attention.head_count", 16),
            (
                "granite.attention.head_count_kv",
                MetadataValue::Array(MetadataArray::I32(vec![4; 28])),
            ),
            f32_key("granite.rope.freq_base", 10_000_000.0),
            f32_key("granite.attention.layer_norm_rms_epsilon", 1e-5),
            u32_key("granite.expert_count", 0),
            u32_key("granite.expert_used_count", 0),
            u32_key("granite.rope.dimension_count", 64),
            f32_key("granite.attention.scale", 0.015_625),
            f32_key("granite.embedding_scale", 12.0),
            f32_key("granite.residual_scale", 0.263),
            f32_key("granite.logit_scale", 4.0),
        ];
        crate::test_support::parsed_header_with_extra_tensors(metadata, 1024, 32, &names)
    }

    #[test]
    fn the_real_granite_4_header_builds_a_uniform_adjacent_silu_descriptor_from_the_shipped_file() {
        let parsed = granite_4_header();

        let descriptor = temp_env::with_var("PROXIMA_ARCH_CONFIG_DIR", None::<&str>, || {
            crate::lowering::header_descriptor(&parsed, crate::lowering::KvLayout::default())
        })
        .expect("granite header builds through the unlisted-family path");

        assert_eq!(descriptor.block_count, 28);
        assert_eq!(descriptor.layers.len(), 28);
        for layer in &descriptor.layers {
            assert_eq!(layer.attention.rope_pairing, RopePairing::Interleaved);
            assert_eq!(layer.ffn.activation, Activation::Silu);
            assert!(!layer.attention.value_norm);
        }
        assert_eq!(descriptor.embedding_scale, Some(EmbeddingScale::Factor(12.0)));
    }

    #[test]
    fn the_directory_defaults_to_the_shipped_data_and_the_env_overrides_it() {
        let unset = temp_env::with_var("PROXIMA_ARCH_CONFIG_DIR", None::<&str>, || {
            ArchConfigSettings::from_env().expect("settings load").directory()
        });
        assert!(unset.ends_with("src/arch"), "{unset:?}");

        let built = ArchConfigSettings::builder().dir("/etc/proxima/arch".into()).build();
        assert_eq!(built.directory(), PathBuf::from("/etc/proxima/arch"));
    }
}
