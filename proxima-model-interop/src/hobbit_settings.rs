//! `HobbitSettings`: the `[hobbit]` section of [`crate::ServingSettings`],
//! the configuration HOBBIT's sensitivity pass and budget allocator read.
//! It follows [`crate::CodecPolicySettings`]'s pattern: one owned data shape
//! carrying `bon::Builder`, `serde` (TOML) and
//! `conflaguration::{Settings, Validate}` (`PROXIMA_HOBBIT_*` env), so all
//! three surfaces build the same value.
//!
//! The fields are the ticket's own list: a byte budget, the candidate
//! codec(s), the calibration prompt set, the minimum number of layers to
//! recode, and the quality bound without which no rule is applied. Nothing
//! here names a model family or a layer: layer prefixes come from the
//! checkpoint's tensor directory.

use bon::Builder;
use conflaguration::{Settings, Validate};
use proxima_gguf::types::GgmlType;
use proxima_numeric::NumericFormat;
use serde::{Deserialize, Serialize};

use crate::bind::{codec_to_ggml_type, target_has_encoder};
use crate::error::InteropError;

fn from_json<T: serde::de::DeserializeOwned>(raw: &str) -> Result<T, serde_json::Error> {
    serde_json::from_str(raw)
}

fn default_prompts() -> Vec<String> {
    [
        "The quick brown fox jumps over the lazy dog.",
        "Precision is chosen per layer by measuring what each layer changes.",
    ]
    .iter()
    .map(|prompt| (*prompt).to_owned())
    .collect()
}

/// The quality bound a recode must hold before it is applied. At least one
/// field must be `Some`, or [`HobbitSettings::quality_bound`] refuses to
/// return a bound and the caller must not apply anything.
#[derive(
    Debug, Clone, Copy, PartialEq, Default, Builder, Deserialize, Serialize, Settings, Validate,
)]
#[settings(prefix = "PROXIMA_HOBBIT_QUALITY")]
#[builder(derive(Clone, Debug))]
#[serde(default)]
pub struct HobbitQualitySettings {
    /// Maximum mean KL(reference || recoded) over the calibration positions.
    pub max_mean_kl: Option<f64>,
    /// Minimum top-1 agreement against the unrecoded run, in `0.0..=1.0`.
    pub min_top1_agreement: Option<f64>,
}

impl HobbitQualitySettings {
    /// The bound as a usable pair, or `None` when neither side is set. A
    /// caller that receives `None` must not apply a rule (R5: no bound, no
    /// apply).
    #[must_use]
    pub const fn bound(&self) -> Option<(Option<f64>, Option<f64>)> {
        match (self.max_mean_kl, self.min_top1_agreement) {
            (None, None) => None,
            (max_mean_kl, min_top1_agreement) => Some((max_mean_kl, min_top1_agreement)),
        }
    }
}

/// The `[hobbit]` section: the budget, candidate codecs, calibration set,
/// minimum recoded-layer count and quality bound HOBBIT uses.
#[derive(Debug, Clone, PartialEq, Builder, Deserialize, Serialize, Settings, Validate)]
#[settings(prefix = "PROXIMA_HOBBIT")]
#[builder(derive(Clone, Debug))]
#[serde(default)]
pub struct HobbitSettings {
    /// Maximum bytes the whole checkpoint's tensors may occupy after
    /// recoding. `None` is unset: the allocator refuses rather than
    /// inventing a budget.
    pub budget_bytes: Option<u64>,
    /// Candidate recode codecs, a JSON list in env, spelled as
    /// [`NumericFormat`] names (`q3_k`, `q4_k`, `q8_0`, ...). A candidate
    /// with no encoder is refused at lowering, never silently skipped.
    #[setting(resolve_with = "from_json", default_str = r#"["q3_k"]"#)]
    #[builder(default = vec![NumericFormat::Q3K])]
    pub candidate_codecs: Vec<NumericFormat>,
    /// Calibration prompts. The example tokenizes them with the
    /// checkpoint's own vocabulary; the sensitivity pass is the same for
    /// every family.
    #[setting(
        resolve_with = "from_json",
        default_str = r#"["The quick brown fox jumps over the lazy dog.","Precision is chosen per layer by measuring what each layer changes."]"#
    )]
    #[builder(default = default_prompts())]
    pub calibration_prompts: Vec<String>,
    /// The minimum number of layers the allocation recodes, even when the
    /// budget is already met.
    #[setting(default = 1)]
    #[builder(default = 1)]
    pub min_layers: usize,
    /// Tokens per calibration prompt the sensitivity pass uses. `0` uses
    /// the whole prompt.
    #[setting(default = 0)]
    #[builder(default = 0)]
    pub max_calibration_tokens: usize,
    #[setting(nested)]
    #[builder(default)]
    pub quality: HobbitQualitySettings,
}

impl Default for HobbitSettings {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl HobbitSettings {
    /// Lowers each candidate `NumericFormat` to the `GgmlType` the recode
    /// binds at, refusing a format with no ggml wire type and a type with no
    /// encoder. Never returns an empty list: an empty candidate set is a
    /// configuration error, not "all layers".
    ///
    /// # Errors
    ///
    /// [`InteropError::UnsupportedCodec`] when a candidate has no ggml wire
    /// type; [`InteropError::UnsupportedWeightPrecisionTarget`] when it has
    /// a wire type but no encoder.
    pub fn candidate_targets(&self) -> Result<Vec<GgmlType>, InteropError> {
        if self.candidate_codecs.is_empty() {
            return Err(InteropError::UnsupportedWeightPrecisionTarget {
                tensor: "hobbit.candidate_codecs".into(),
                target: GgmlType::F32,
            });
        }
        self.candidate_codecs
            .iter()
            .map(|codec| {
                let target = codec_to_ggml_type(*codec)?;
                if target_has_encoder(target) {
                    Ok(target)
                } else {
                    Err(InteropError::UnsupportedWeightPrecisionTarget {
                        tensor: "hobbit.candidate_codecs".into(),
                        target,
                    })
                }
            })
            .collect()
    }

    /// The quality bound, refusing when neither side is configured.
    ///
    /// # Errors
    ///
    /// [`InteropError::UnsupportedWeightPrecisionTarget`] is not the right
    /// shape for this; the caller uses [`HobbitQualitySettings::bound`]'s
    /// `None` directly. This helper exists so a configuration with no bound
    /// can be reported by name at the apply site.
    #[must_use]
    pub const fn quality_bound(&self) -> Option<(Option<f64>, Option<f64>)> {
        self.quality.bound()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const ENV_KEYS: [&str; 6] = [
        "PROXIMA_HOBBIT_BUDGET_BYTES",
        "PROXIMA_HOBBIT_CANDIDATE_CODECS",
        "PROXIMA_HOBBIT_CALIBRATION_PROMPTS",
        "PROXIMA_HOBBIT_MIN_LAYERS",
        "PROXIMA_HOBBIT_MAX_CALIBRATION_TOKENS",
        "PROXIMA_HOBBIT_QUALITY_MAX_MEAN_KL",
    ];

    fn cleared_env() -> Vec<(&'static str, Option<&'static str>)> {
        ENV_KEYS.iter().map(|key| (*key, None)).collect()
    }

    #[test]
    fn hobbit_settings_default_has_no_budget_and_no_quality_bound() {
        let defaults = HobbitSettings::builder().build();
        let from_empty_toml: HobbitSettings =
            conflaguration::from_toml_str("").expect("an empty toml takes every default");
        temp_env::with_vars(cleared_env(), || {
            let from_env = HobbitSettings::from_env().expect("unset env takes every default");
            assert_eq!(from_env, defaults);
        });
        assert_eq!(from_empty_toml, defaults);
        assert_eq!(defaults.budget_bytes, None);
        assert_eq!(defaults.quality_bound(), None);
    }

    #[test]
    fn hobbit_settings_toml_lowers_candidates_and_bound() {
        let toml_text = r#"
budget_bytes = 200000000
candidate_codecs = ["q3_k", "q4_k"]
calibration_prompts = ["one", "two"]
min_layers = 2
max_calibration_tokens = 16

[quality]
max_mean_kl = 0.05
min_top1_agreement = 0.95
"#;
        let settings: HobbitSettings =
            conflaguration::from_toml_str(toml_text).expect("the hobbit toml section loads");
        assert_eq!(settings.budget_bytes, Some(200_000_000));
        assert_eq!(settings.min_layers, 2);
        assert_eq!(settings.max_calibration_tokens, 16);
        assert_eq!(settings.calibration_prompts, vec!["one", "two"]);
        assert_eq!(settings.quality_bound(), Some((Some(0.05), Some(0.95))));
        assert_eq!(
            settings
                .candidate_targets()
                .expect("both candidates encode"),
            vec![GgmlType::Q3_K, GgmlType::Q4_K]
        );
    }

    #[test]
    fn hobbit_settings_refuses_candidate_without_encoder() {
        let settings = HobbitSettings::builder()
            .candidate_codecs(vec![NumericFormat::Iq2Xxs])
            .build();
        let error = settings
            .candidate_targets()
            .expect_err("iq2_xxs has no encoder");
        assert!(
            matches!(error, InteropError::UnsupportedWeightPrecisionTarget { .. }),
            "{error}"
        );

        let empty = HobbitSettings::builder().candidate_codecs(vec![]).build();
        empty
            .candidate_targets()
            .expect_err("an empty candidate set is a refusal");
    }

    #[test]
    fn hobbit_settings_env_toml_builder_agree_on_byte_identity() {
        let via_builder = HobbitSettings::builder()
            .budget_bytes(1024)
            .candidate_codecs(vec![NumericFormat::Q8_0])
            .calibration_prompts(vec!["p".to_owned()])
            .min_layers(1)
            .max_calibration_tokens(8)
            .quality(HobbitQualitySettings::builder().max_mean_kl(0.1).build())
            .build();
        let via_toml: HobbitSettings = conflaguration::from_toml_str(
            "budget_bytes = 1024\ncandidate_codecs = [\"q8_0\"]\ncalibration_prompts = [\"p\"]\nmin_layers = 1\nmax_calibration_tokens = 8\n\n[quality]\nmax_mean_kl = 0.1\n",
        )
        .expect("toml agrees");
        assert_eq!(via_toml, via_builder);
    }
}
