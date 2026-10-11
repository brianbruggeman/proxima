//! `CodecPolicySettings`: the `[codec_policy]` section of [`crate::ServingSettings`],
//! the choice of weight codec the offline conversion (`convert_codec`) applies and
//! the tensor classes it applies to. It follows the pattern of
//! [`crate::PromptCacheSettings`]: one owned data shape carrying `bon::Builder`,
//! `serde` (TOML) and `conflaguration::{Settings, Validate}` (`PROXIMA_CODEC_POLICY_*`
//! env), so the three surfaces build the same value. [`CodecPolicyWeights::codec`]
//! lowers the choice to the [`proxima_primitives::Codec`] the checkpoint stores.

use bon::Builder;
use conflaguration::{Settings, Validate};
use proxima_primitives::Codec;
use serde::{Deserialize, Serialize};

const DEFAULT_CLASSES: [&str; 7] = [
    "ffn_gate_exps",
    "ffn_up_exps",
    "ffn_down_exps",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
];

fn from_json<T: serde::de::DeserializeOwned>(raw: &str) -> Result<T, serde_json::Error> {
    serde_json::from_str(raw)
}

fn from_name<T: serde::de::DeserializeOwned>(raw: &str) -> Result<T, serde_json::Error> {
    serde_json::from_value(serde_json::Value::String(raw.to_owned()))
}

fn default_classes() -> Vec<String> {
    DEFAULT_CLASSES
        .iter()
        .map(|class| (*class).to_owned())
        .collect()
}

/// Which codec the policy converts the named classes to; the serde names are the
/// strings a TOML file and `PROXIMA_CODEC_POLICY_WEIGHTS` accept.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CodecPolicyWeights {
    /// Keep the checkpoint's own codecs.
    #[default]
    #[serde(rename = "shipped")]
    Shipped,
    /// Convert to [`Codec::Fp6E3M2`].
    #[serde(rename = "fp6_e3m2")]
    Fp6E3M2,
}

impl CodecPolicyWeights {
    /// The codec conversion targets, or `None` when the shipped codecs are kept.
    #[must_use]
    pub const fn codec(self) -> Option<Codec> {
        match self {
            Self::Shipped => None,
            Self::Fp6E3M2 => Some(Codec::Fp6E3M2),
        }
    }
}

/// The `[codec_policy]` section: `weights` picks the target codec and `classes`
/// lists the tensor classes (GGUF name stems such as `ffn_up_exps`, `attn_q`)
/// it applies to. Embeddings, norms, routers and the output head are not in the
/// default list and stay on their shipped codec.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings, Validate)]
#[settings(prefix = "PROXIMA_CODEC_POLICY")]
#[builder(derive(Clone, Debug))]
#[serde(default)]
pub struct CodecPolicySettings {
    /// `shipped` or `fp6_e3m2`; an unknown name is a load error naming it.
    #[setting(resolve_with = "from_name", default_str = "shipped")]
    #[builder(default)]
    pub weights: CodecPolicyWeights,
    /// Tensor classes the codec applies to, a JSON list in env.
    #[setting(
        resolve_with = "from_json",
        default_str = r#"["ffn_gate_exps","ffn_up_exps","ffn_down_exps","attn_q","attn_k","attn_v","attn_output"]"#
    )]
    #[builder(default = default_classes())]
    pub classes: Vec<String>,
}

impl Default for CodecPolicySettings {
    fn default() -> Self {
        Self::builder().build()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const ENV_KEYS: [&str; 2] = [
        "PROXIMA_CODEC_POLICY_WEIGHTS",
        "PROXIMA_CODEC_POLICY_CLASSES",
    ];

    fn cleared_env() -> Vec<(&'static str, Option<&'static str>)> {
        ENV_KEYS.iter().map(|key| (*key, None)).collect()
    }

    #[test]
    fn codec_policy_default_is_shipped() {
        let from_builder = CodecPolicySettings::builder().build();
        let from_empty_toml: CodecPolicySettings =
            conflaguration::from_toml_str("").expect("an empty toml takes every default");

        temp_env::with_vars(cleared_env(), || {
            let from_env = CodecPolicySettings::from_env().expect("unset env takes every default");
            assert_eq!(from_env, from_builder);
        });
        assert_eq!(from_empty_toml, from_builder);
        assert_eq!(from_builder.weights, CodecPolicyWeights::Shipped);
        assert_eq!(from_builder.weights.codec(), None);
        assert_eq!(from_builder.classes, default_classes());
    }

    #[test]
    fn codec_policy_env_toml_builder_agree() {
        let via_builder = CodecPolicySettings::builder()
            .weights(CodecPolicyWeights::Fp6E3M2)
            .classes(vec!["ffn_up_exps".to_owned(), "attn_q".to_owned()])
            .build();
        let toml_text = "weights = \"fp6_e3m2\"\nclasses = [\"ffn_up_exps\", \"attn_q\"]\n";
        let via_toml: CodecPolicySettings =
            conflaguration::from_toml_str(toml_text).expect("the toml loader accepts the section");
        let env = vec![
            ("PROXIMA_CODEC_POLICY_WEIGHTS", Some("fp6_e3m2")),
            (
                "PROXIMA_CODEC_POLICY_CLASSES",
                Some(r#"["ffn_up_exps","attn_q"]"#),
            ),
        ];

        temp_env::with_vars(env, || {
            let via_env =
                CodecPolicySettings::from_env().expect("the env loader accepts the section");
            assert_eq!(via_env, via_builder);
        });
        assert_eq!(via_toml, via_builder);
        assert_eq!(via_builder.weights.codec(), Some(Codec::Fp6E3M2));
        let round_trip: CodecPolicySettings =
            serde_json::from_str(&serde_json::to_string(&via_builder).expect("serialize"))
                .expect("a serialized section reloads");
        assert_eq!(round_trip, via_builder);
    }

    #[test]
    fn codec_policy_unknown_codec_name_rejected() {
        let toml_error =
            conflaguration::from_toml_str::<CodecPolicySettings>("weights = \"fp5_e2m2\"\n")
                .expect_err("toml must reject an unknown codec name");
        assert!(toml_error.to_string().contains("fp5_e2m2"), "{toml_error}");

        let mut env = cleared_env();
        env[0].1 = Some("fp5_e2m2");
        temp_env::with_vars(env, || {
            let env_error =
                CodecPolicySettings::from_env().expect_err("env must reject an unknown codec name");
            assert!(env_error.to_string().contains("fp5_e2m2"), "{env_error}");
        });
    }
}
