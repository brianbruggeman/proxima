//! `ArchConfig`: the three correctness facts no checkpoint carries, keyed by
//! `general.architecture` and read from a file at run time.
//!
//! llama.cpp hardcodes these per architecture (`llama_model_rope_type`, the
//! per-arch graph files); no GGUF header or tensor states them. Adding an
//! architecture is a `<architecture>.toml` in a config directory, not a
//! recompile, and an architecture with no file is the typed refusal
//! [`InteropError::MissingArchField`], never a guess. Shipped examples live in
//! `src/arch/`.
//!
//! Teaching pointer: this is the config surface of principle 4 on the same
//! pattern as `PromptCacheSettings` (`bon` builder + `serde` + `conflaguration`
//! `Settings`/`Validate`); its fields reuse the descriptor's own enums
//! ([`proxima_tensor::spec::FamilyProfile`] holds the compiled equivalents in
//! `crate::profiles`, which this does not yet replace).

use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use conflaguration::{Settings, Validate};
use proxima_tensor::spec::{Activation, ExpertGatingFunc, RopeLayout};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::error::InteropError;

const REQUIRED_FIELDS: [&str; 3] = ["rope_layout", "ffn_activation", "value_norm"];

/// A value in an environment variable that names no variant.
#[derive(Debug, Error)]
#[error("unrecognized value {0:?}")]
struct ParseValueError(String);

/// `embedding_scale = "sqrt"`: scale the embedding lookup by `sqrt(embedding)`
/// (gemma4, `gemma4.cpp:164`). Absent means the checkpoint's own key decides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingScaleOverride {
    Sqrt,
}

/// `score_scale = "unscaled"`: attention scores are not scaled (gemma4,
/// `gemma4.cpp:11`). Absent means `attention.scale` or `1/sqrt(head_dim)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreScaleOverride {
    Unscaled,
}

/// The runtime config for one architecture. Build it with
/// [`ArchConfig::builder`], load it with [`ArchConfig::load`], or seed a
/// builder from a loaded value with [`ArchConfig::builder_from`]; all three
/// reach the same value.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, Settings, Validate)]
#[serde(deny_unknown_fields)]
#[settings(prefix = "PROXIMA_ARCH")]
pub struct ArchConfig {
    /// Which channels RoPE rotates together: `"adjacent"` for llama and
    /// granite, `"split_half"` for qwen2 and gemma4.
    #[setting(resolve_with = "parse_rope_layout")]
    pub rope_layout: RopeLayout,
    /// The FFN gate nonlinearity: `"Silu"` for llama and granite, `"GeluTanh"`
    /// for gemma4.
    #[setting(resolve_with = "parse_activation")]
    pub ffn_activation: Activation,
    /// Per-kv-head value RMSNorm without a learned scale: `false` for llama,
    /// `true` for gemma4.
    #[setting(resolve_with = "parse_bool")]
    pub value_norm: bool,
    /// `Some(Sqrt)` for an architecture that scales embeddings by
    /// `sqrt(embedding)` without saying so in its header.
    #[serde(default)]
    #[setting(resolve_with = "parse_embedding_scale", default)]
    pub embedding_scale: Option<EmbeddingScaleOverride>,
    /// `Some(Unscaled)` for an architecture whose attention scores take no
    /// scale and whose header has no `attention.scale`.
    #[serde(default)]
    #[setting(resolve_with = "parse_score_scale", default)]
    pub score_scale: Option<ScoreScaleOverride>,
    /// The routed-expert gating when the header has no `expert_gating_func`:
    /// `"Softmax"` for granitemoe, `"Sigmoid"` for flint.
    #[serde(default)]
    #[setting(resolve_with = "parse_routed_gating", default)]
    pub routed_gating: Option<ExpertGatingFunc>,
}

/// The mutable surface of [`ArchConfig`]: from [`ArchConfig::builder`] (all
/// unset) or [`ArchConfig::builder_from`] (seeded from a loaded value). Every
/// setter overwrites and returns `self`; [`Self::build`] is the one place a
/// missing required field is refused. Hand-written rather than `bon` because
/// a seeded field must stay settable, which `bon`'s typestate forbids without
/// its experimental `overwritable` feature.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ArchConfigBuilder {
    rope_layout: Option<RopeLayout>,
    ffn_activation: Option<Activation>,
    value_norm: Option<bool>,
    embedding_scale: Option<EmbeddingScaleOverride>,
    score_scale: Option<ScoreScaleOverride>,
    routed_gating: Option<ExpertGatingFunc>,
}

impl ArchConfigBuilder {
    #[must_use]
    pub fn rope_layout(mut self, rope_layout: RopeLayout) -> Self {
        self.rope_layout = Some(rope_layout);
        self
    }

    #[must_use]
    pub fn ffn_activation(mut self, ffn_activation: Activation) -> Self {
        self.ffn_activation = Some(ffn_activation);
        self
    }

    #[must_use]
    pub fn value_norm(mut self, value_norm: bool) -> Self {
        self.value_norm = Some(value_norm);
        self
    }

    /// `None` clears the override.
    #[must_use]
    pub fn embedding_scale(mut self, embedding_scale: Option<EmbeddingScaleOverride>) -> Self {
        self.embedding_scale = embedding_scale;
        self
    }

    /// `None` clears the override.
    #[must_use]
    pub fn score_scale(mut self, score_scale: Option<ScoreScaleOverride>) -> Self {
        self.score_scale = score_scale;
        self
    }

    /// `None` clears the override.
    #[must_use]
    pub fn routed_gating(mut self, routed_gating: Option<ExpertGatingFunc>) -> Self {
        self.routed_gating = routed_gating;
        self
    }

    /// # Errors
    ///
    /// [`InteropError::MissingArchField`] naming every unset required field;
    /// its `architecture` is empty because a builder does not know one.
    pub fn build(self) -> Result<ArchConfig, InteropError> {
        let (Some(rope_layout), Some(ffn_activation), Some(value_norm)) =
            (self.rope_layout, self.ffn_activation, self.value_norm)
        else {
            let present = [
                self.rope_layout.is_some(),
                self.ffn_activation.is_some(),
                self.value_norm.is_some(),
            ];
            return Err(ArchConfig::missing(
                "",
                &unset_fields(present),
            ));
        };
        Ok(ArchConfig {
            rope_layout,
            ffn_activation,
            value_norm,
            embedding_scale: self.embedding_scale,
            score_scale: self.score_scale,
            routed_gating: self.routed_gating,
        })
    }
}

fn unset_fields(present: [bool; 3]) -> Vec<&'static str> {
    REQUIRED_FIELDS
        .into_iter()
        .zip(present)
        .filter_map(|(name, here)| (!here).then_some(name))
        .collect()
}

/// The same file with every field optional, so one pass can name all that
/// are missing instead of stopping at serde's first.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawArchConfig {
    rope_layout: Option<RopeLayout>,
    ffn_activation: Option<Activation>,
    value_norm: Option<bool>,
    #[serde(default)]
    embedding_scale: Option<EmbeddingScaleOverride>,
    #[serde(default)]
    score_scale: Option<ScoreScaleOverride>,
    #[serde(default)]
    routed_gating: Option<ExpertGatingFunc>,
}

impl RawArchConfig {
    fn missing(&self) -> Vec<&'static str> {
        unset_fields([
            self.rope_layout.is_some(),
            self.ffn_activation.is_some(),
            self.value_norm.is_some(),
        ])
    }
}

fn parse_rope_layout(text: &str) -> Result<RopeLayout, ParseValueError> {
    match text {
        "adjacent" => Ok(RopeLayout::Adjacent),
        "split_half" => Ok(RopeLayout::SplitHalf),
        other => Err(ParseValueError(other.into())),
    }
}

fn parse_activation(text: &str) -> Result<Activation, ParseValueError> {
    match text {
        "Silu" => Ok(Activation::Silu),
        "GeluTanh" => Ok(Activation::GeluTanh),
        other => Err(ParseValueError(other.into())),
    }
}

fn parse_bool(text: &str) -> Result<bool, ParseValueError> {
    text.parse().map_err(|_| ParseValueError(text.into()))
}

fn parse_embedding_scale(text: &str) -> Result<Option<EmbeddingScaleOverride>, ParseValueError> {
    match text {
        "sqrt" => Ok(Some(EmbeddingScaleOverride::Sqrt)),
        other => Err(ParseValueError(other.into())),
    }
}

fn parse_score_scale(text: &str) -> Result<Option<ScoreScaleOverride>, ParseValueError> {
    match text {
        "unscaled" => Ok(Some(ScoreScaleOverride::Unscaled)),
        other => Err(ParseValueError(other.into())),
    }
}

fn parse_routed_gating(text: &str) -> Result<Option<ExpertGatingFunc>, ParseValueError> {
    match text {
        "Softmax" => Ok(Some(ExpertGatingFunc::Softmax)),
        "Sigmoid" => Ok(Some(ExpertGatingFunc::Sigmoid)),
        other => Err(ParseValueError(other.into())),
    }
}

impl ArchConfig {
    /// An empty builder; `build` refuses until the three required fields are set.
    #[must_use]
    pub fn builder() -> ArchConfigBuilder {
        ArchConfigBuilder::default()
    }

    /// A builder seeded from `config` (a loaded file, say); any setter then
    /// overwrites the seeded value.
    #[must_use]
    pub fn builder_from(config: &Self) -> ArchConfigBuilder {
        ArchConfigBuilder {
            rope_layout: Some(config.rope_layout),
            ffn_activation: Some(config.ffn_activation),
            value_norm: Some(config.value_norm),
            embedding_scale: config.embedding_scale,
            score_scale: config.score_scale,
            routed_gating: config.routed_gating,
        }
    }

    /// Reads `<dir>/<architecture>.toml` at run time; there is no embedded
    /// fallback.
    ///
    /// # Errors
    ///
    /// [`InteropError::MissingArchField`] when the file is absent (naming all
    /// three required fields) or lacks some of them (naming exactly those);
    /// [`InteropError::InvalidFamilyProfile`] when `architecture` is not a
    /// plain file stem, the file cannot be read, or it does not parse.
    pub fn load(dir: &Path, architecture: &str) -> Result<Self, InteropError> {
        let invalid = |message: String| InteropError::InvalidFamilyProfile {
            family: architecture.into(),
            message,
        };
        let plain = !architecture.is_empty()
            && !architecture.contains(['/', '\\'])
            && architecture != "."
            && architecture != "..";
        if !plain {
            return Err(invalid("architecture is not a plain file stem".into()));
        }
        let path = dir.join(format!("{architecture}.toml"));
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) if error.kind() == ErrorKind::NotFound => {
                return Err(Self::missing(architecture, &REQUIRED_FIELDS));
            }
            Err(error) => return Err(invalid(format!("{}: {error}", path.display()))),
        };
        let raw: RawArchConfig =
            conflaguration::from_toml_str(&text).map_err(|error| invalid(error.to_string()))?;
        let missing = raw.missing();
        match (raw.rope_layout, raw.ffn_activation, raw.value_norm) {
            (Some(rope_layout), Some(ffn_activation), Some(value_norm)) => Ok(Self {
                rope_layout,
                ffn_activation,
                value_norm,
                embedding_scale: raw.embedding_scale,
                score_scale: raw.score_scale,
                routed_gating: raw.routed_gating,
            }),
            _ => Err(Self::missing(architecture, &missing)),
        }
    }

    fn missing(architecture: &str, fields: &[&str]) -> InteropError {
        InteropError::MissingArchField {
            architecture: architecture.into(),
            field: fields.join(", "),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::path::PathBuf;

    use tempfile::TempDir;

    use super::*;
    use crate::profiles::family_profile;

    const LLAMA_FILE: &str = "rope_layout = \"adjacent\"\nffn_activation = \"Silu\"\nvalue_norm = false\n";

    fn shipped_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src").join("arch")
    }

    fn dir_with(name: &str, text: &str) -> TempDir {
        let dir = tempfile::tempdir().expect("tempdir for arch config");
        fs::write(dir.path().join(format!("{name}.toml")), text).expect("write arch file");
        dir
    }

    fn missing_fields(result: Result<ArchConfig, InteropError>) -> String {
        match result {
            Err(InteropError::MissingArchField { field, .. }) => field,
            other => panic!("expected MissingArchField, got {other:?}"),
        }
    }

    #[test]
    fn arch_config_load_reads_a_complete_file() {
        let dir = dir_with("llama", LLAMA_FILE);

        let config = ArchConfig::load(dir.path(), "llama").expect("complete file loads");

        assert_eq!(config.rope_layout, RopeLayout::Adjacent);
        assert_eq!(config.ffn_activation, Activation::Silu);
        assert!(!config.value_norm);
        assert_eq!(config.embedding_scale, None);
    }

    #[test]
    fn arch_config_load_reads_the_overrides() {
        let text = format!(
            "{LLAMA_FILE}embedding_scale = \"sqrt\"\nscore_scale = \"unscaled\"\nrouted_gating = \"Softmax\"\n"
        );
        let dir = dir_with("gemma4", &text);

        let config = ArchConfig::load(dir.path(), "gemma4").expect("overrides load");

        assert_eq!(config.embedding_scale, Some(EmbeddingScaleOverride::Sqrt));
        assert_eq!(config.score_scale, Some(ScoreScaleOverride::Unscaled));
        assert_eq!(config.routed_gating, Some(ExpertGatingFunc::Softmax));
    }

    #[test]
    fn arch_config_missing_file_names_all_three_fields() {
        let dir = tempfile::tempdir().expect("tempdir");

        let fields = missing_fields(ArchConfig::load(dir.path(), "newarch"));

        assert_eq!(fields, "rope_layout, ffn_activation, value_norm");
    }

    #[test]
    fn arch_config_file_missing_one_field_names_exactly_that_field() {
        let dir = dir_with("newarch", "rope_layout = \"adjacent\"\nvalue_norm = false\n");

        let fields = missing_fields(ArchConfig::load(dir.path(), "newarch"));

        assert_eq!(fields, "ffn_activation");
    }

    #[test]
    fn arch_config_empty_file_names_all_three_fields() {
        let dir = dir_with("newarch", "");

        let fields = missing_fields(ArchConfig::load(dir.path(), "newarch"));

        assert_eq!(fields, "rope_layout, ffn_activation, value_norm");
    }

    #[test]
    fn arch_config_unknown_key_is_an_invalid_profile() {
        let dir = dir_with("llama", &format!("{LLAMA_FILE}rope_scale = 2\n"));

        let result = ArchConfig::load(dir.path(), "llama");

        assert!(matches!(
            result,
            Err(InteropError::InvalidFamilyProfile { .. })
        ));
    }

    #[test]
    fn arch_config_path_traversal_in_architecture_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");

        let result = ArchConfig::load(dir.path(), "../llama");

        assert!(matches!(
            result,
            Err(InteropError::InvalidFamilyProfile { .. })
        ));
    }

    #[test]
    fn arch_config_builder_equals_loaded() {
        let dir = dir_with("llama", LLAMA_FILE);
        let loaded = ArchConfig::load(dir.path(), "llama").expect("loads");

        let built = ArchConfig::builder()
            .rope_layout(RopeLayout::Adjacent)
            .ffn_activation(Activation::Silu)
            .value_norm(false)
            .build()
            .expect("three required fields set");

        assert_eq!(built, loaded);
    }

    #[test]
    fn arch_config_builder_names_every_unset_required_field() {
        let error = ArchConfig::builder()
            .ffn_activation(Activation::Silu)
            .build()
            .expect_err("two required fields unset");

        match error {
            InteropError::MissingArchField { field, .. } => {
                assert_eq!(field, "rope_layout, value_norm");
            }
            other => panic!("expected MissingArchField, got {other:?}"),
        }
    }

    #[test]
    fn arch_config_builder_from_round_trips_a_loaded_file() {
        let loaded = ArchConfig::load(&shipped_dir(), "llama").expect("shipped llama loads");

        let rebuilt = ArchConfig::builder_from(&loaded)
            .build()
            .expect("seeded builder is complete");

        assert_eq!(rebuilt, loaded);
    }

    #[test]
    fn arch_config_builder_from_overrides_a_seeded_field_and_keeps_the_rest() {
        let loaded = ArchConfig::load(&shipped_dir(), "llama").expect("shipped llama loads");

        let tweaked = ArchConfig::builder_from(&loaded)
            .value_norm(true)
            .build()
            .expect("seeded builder is complete");

        assert!(tweaked.value_norm);
        assert_eq!(
            tweaked,
            ArchConfig {
                value_norm: true,
                ..loaded
            }
        );
    }

    #[test]
    fn arch_config_serde_round_trips_through_toml() {
        let config = ArchConfig::builder()
            .rope_layout(RopeLayout::SplitHalf)
            .ffn_activation(Activation::GeluTanh)
            .value_norm(true)
            .embedding_scale(Some(EmbeddingScaleOverride::Sqrt))
            .score_scale(Some(ScoreScaleOverride::Unscaled))
            .build()
            .expect("three required fields set");
        let dir = dir_with(
            "gemma4",
            &toml::to_string(&config).expect("config serializes"),
        );

        let loaded = ArchConfig::load(dir.path(), "gemma4").expect("serialized form loads");

        assert_eq!(loaded, config);
    }

    #[test]
    fn arch_config_env_reaches_the_same_value_as_the_builder() {
        let built = ArchConfig::builder()
            .rope_layout(RopeLayout::SplitHalf)
            .ffn_activation(Activation::GeluTanh)
            .value_norm(true)
            .embedding_scale(Some(EmbeddingScaleOverride::Sqrt))
            .build()
            .expect("three required fields set");

        let from_env = temp_env::with_vars(
            [
                ("PROXIMA_ARCH_ROPE_LAYOUT", Some("split_half")),
                ("PROXIMA_ARCH_FFN_ACTIVATION", Some("GeluTanh")),
                ("PROXIMA_ARCH_VALUE_NORM", Some("true")),
                ("PROXIMA_ARCH_EMBEDDING_SCALE", Some("sqrt")),
                ("PROXIMA_ARCH_SCORE_SCALE", None),
                ("PROXIMA_ARCH_ROUTED_GATING", None),
            ],
            ArchConfig::from_env,
        )
        .expect("env config loads");

        assert_eq!(from_env, built);
    }

    #[test]
    fn arch_config_shipped_llama_equals_the_compiled_profile() {
        let loaded = ArchConfig::load(&shipped_dir(), "llama").expect("shipped llama loads");
        let compiled = family_profile("llama").expect("compiled llama profile");

        assert_eq!(loaded.rope_layout, compiled.rope_layout);
        assert_eq!(loaded.ffn_activation, compiled.ffn.activation);
        assert_eq!(loaded.value_norm, compiled.value_norm);
    }

    #[test]
    fn arch_config_shipped_granite_is_adjacent_silu_without_value_norm() {
        let loaded = ArchConfig::load(&shipped_dir(), "granite").expect("shipped granite loads");

        assert_eq!(loaded.rope_layout, RopeLayout::Adjacent);
        assert_eq!(loaded.ffn_activation, Activation::Silu);
        assert!(!loaded.value_norm);
    }
}
