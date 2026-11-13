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
//!
//! The config files are found through an ordered search path
//! ([`ArchConfigSettings::search_path`]): `PROXIMA_ARCH_CONFIG_DIR` alone when
//! set; else `$XDG_CONFIG_HOME/proxima/arch` (or `$HOME/.config/proxima/arch`),
//! `<executable dir>/arch`, then the build-time `src/arch`, so an installed
//! binary does not depend on where its source tree was compiled. The first
//! directory with `<architecture>.toml` wins (`llama.toml`, `granite.toml`
//! ship in `src/arch`); none having it is a refusal listing every directory.

use std::env;
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

/// Where the run-time architecture config files live, as an ordered search
/// path resolved by [`Self::search_path`]; the first directory containing
/// `<architecture>.toml` wins (`llama.toml` and `granite.toml` ship in this
/// crate's `src/arch`). An unreadable file is refused naming the path, never
/// replaced by an embedded copy.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings, Validate)]
#[settings(prefix = "PROXIMA_ARCH_CONFIG")]
#[builder(derive(Clone, Debug))]
pub struct ArchConfigSettings {
    /// A directory of `<architecture>.toml` files, for example
    /// `/etc/proxima/arch`. When set it is the only directory searched; absent
    /// selects, in order: `$XDG_CONFIG_HOME/proxima/arch` (else
    /// `$HOME/.config/proxima/arch`), `<executable dir>/arch` for installed
    /// bundles, then the build-time `CARGO_MANIFEST_DIR/src/arch` for
    /// development, last because it names the build checkout, which an installed
    /// binary may not have.
    pub dir: Option<String>,
}

impl ArchConfigSettings {
    /// The directories to search, in order. Reads `XDG_CONFIG_HOME`, `HOME` and
    /// the running executable's path; a source that is unset or unreadable
    /// contributes no entry.
    #[must_use]
    pub fn search_path(&self) -> Vec<PathBuf> {
        if let Some(dir) = &self.dir {
            return vec![PathBuf::from(dir)];
        }
        let user_config = env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
            .map(|base| base.join("proxima").join("arch"));
        let beside_executable = env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(|parent| parent.join("arch")));
        let build_time = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src").join("arch");
        user_config
            .into_iter()
            .chain(beside_executable)
            .chain(Some(build_time))
            .collect()
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
    let config = ArchConfig::load_from(&settings.search_path(), family)?;
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
            Err(InteropError::MissingArchField { architecture, field, searched }) => {
                assert_eq!(architecture, "phi9");
                assert_eq!(field, "rope_layout, ffn_activation, value_norm");
                assert_eq!(searched, dir.path().display().to_string());
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

        let empty_home = tempfile::tempdir().expect("empty home and xdg config home");
        let empty_home_text = empty_home.path().to_str().expect("tempdir path is utf-8");
        let descriptor = temp_env::with_vars(
            [
                ("PROXIMA_ARCH_CONFIG_DIR", None),
                ("XDG_CONFIG_HOME", Some(empty_home_text)),
                ("HOME", Some(empty_home_text)),
            ],
            || crate::lowering::header_descriptor(&parsed, crate::lowering::KvLayout::default()),
        )
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

    const PHI9_FILE: &str = "rope_layout = \"split_half\"\nffn_activation = \"GeluTanh\"\nvalue_norm = true\n";
    const PHI9_OTHER_FILE: &str = "rope_layout = \"adjacent\"\nffn_activation = \"Silu\"\nvalue_norm = false\n";

    fn build_time_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src").join("arch")
    }

    fn profile_with_dirs(
        env_dir: Option<&str>,
        xdg: Option<&str>,
        family: &str,
    ) -> Result<FamilyProfile, InteropError> {
        temp_env::with_vars(
            [
                ("PROXIMA_ARCH_CONFIG_DIR", env_dir),
                ("XDG_CONFIG_HOME", xdg),
                ("HOME", xdg),
            ],
            || unknown_family_profile(family),
        )
    }

    fn xdg_dir_with(name: &str, text: &str) -> TempDir {
        let xdg = tempfile::tempdir().expect("tempdir for xdg config home");
        let arch = xdg.path().join("proxima").join("arch");
        fs::create_dir_all(&arch).expect("create xdg arch dir");
        fs::write(arch.join(format!("{name}.toml")), text).expect("write xdg arch file");
        xdg
    }

    #[test]
    fn the_env_directory_wins_even_when_the_user_config_dir_has_the_file() {
        let env_dir = dir_with("phi9", PHI9_FILE);
        let xdg = xdg_dir_with("phi9", PHI9_OTHER_FILE);

        let profile = profile_with_dirs(
            env_dir.path().to_str(),
            xdg.path().to_str(),
            "phi9",
        )
        .expect("env directory has the file");

        assert_eq!(profile.rope_layout, RopeLayout::SplitHalf);
        assert!(profile.value_norm);
    }

    #[test]
    fn the_env_directory_is_the_only_one_searched_when_it_lacks_the_file() {
        let env_dir = tempfile::tempdir().expect("empty env dir");
        let xdg = xdg_dir_with("phi9", PHI9_FILE);

        let result = profile_with_dirs(env_dir.path().to_str(), xdg.path().to_str(), "phi9");

        assert!(matches!(result, Err(InteropError::MissingArchField { .. })), "{result:?}");
    }

    #[test]
    fn the_xdg_config_home_directory_is_found_when_the_env_is_unset() {
        let xdg = xdg_dir_with("phi9", PHI9_FILE);

        let profile = profile_with_dirs(None, xdg.path().to_str(), "phi9")
            .expect("xdg arch directory has the file");

        assert_eq!(profile.rope_layout, RopeLayout::SplitHalf);
    }

    #[test]
    fn with_nothing_anywhere_the_refusal_lists_every_searched_directory_in_order() {
        let xdg = tempfile::tempdir().expect("empty xdg config home");
        let xdg_text = xdg.path().to_str().expect("tempdir path is utf-8");

        let error = profile_with_dirs(None, Some(xdg_text), "phi9").expect_err("no file anywhere");

        let InteropError::MissingArchField { searched, .. } = &error else {
            panic!("expected MissingArchField, got {error:?}");
        };
        let user = xdg.path().join("proxima").join("arch");
        let exe = env::current_exe().expect("test exe path").parent().expect("exe dir").join("arch");
        let expected = [user, exe, build_time_dir()]
            .iter()
            .map(|dir| dir.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        assert_eq!(searched, &expected);
        assert!(error.to_string().contains(&expected), "{error}");
    }

    #[test]
    fn the_build_time_directory_is_searched_last_and_still_supplies_the_shipped_files() {
        let xdg = tempfile::tempdir().expect("empty xdg config home");
        let settings = ArchConfigSettings::builder().build();

        let path = temp_env::with_vars(
            [("XDG_CONFIG_HOME", xdg.path().to_str()), ("HOME", xdg.path().to_str())],
            || settings.search_path(),
        );
        let granite = profile_with_dirs(None, xdg.path().to_str(), "granite")
            .expect("shipped granite is found in the build-time directory");

        assert_eq!(path.last(), Some(&build_time_dir()));
        assert_eq!(path.len(), 3);
        assert_eq!(granite.rope_layout, RopeLayout::Adjacent);
    }

    #[test]
    fn a_configured_dir_alone_is_the_search_path() {
        let built = ArchConfigSettings::builder().dir("/etc/proxima/arch".into()).build();

        assert_eq!(built.search_path(), vec![PathBuf::from("/etc/proxima/arch")]);
    }
}
