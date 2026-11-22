//! `MoeExpertPrecisionSettings`: the `[moe_expert_precision]` section of
//! [`crate::ServingSettings`], the HOBBIT token-level expert policy.
//!
//! `enabled` is off by default so the pre-existing on-disk/high path is
//! byte-identical. `high_threshold` is the paper's `T1`: the first selected
//! expert (score 0) and the cumulative head of the normalized gate mass are
//! served high; the tail is served from the low-codec sidecar. `low_codec`
//! is a [`proxima_numeric::NumericFormat`] the offline sidecar writer binds
//! at, never a bespoke numeric type.

use bon::Builder;
use conflaguration::{Settings, Validate};
use proxima_numeric::NumericFormat;
use serde::{Deserialize, Serialize};

use crate::hobbit_settings::HobbitQualitySettings;

fn from_name<T: serde::de::DeserializeOwned>(raw: &str) -> Result<T, serde_json::Error> {
    serde_json::from_value(serde_json::Value::String(raw.to_owned()))
}

/// The `[moe_expert_precision]` section: whether the gate magnitude chooses
/// high/low precision, the cumulative-mass threshold, and the sidecar codec.
#[derive(Debug, Clone, PartialEq, Builder, Deserialize, Serialize, Settings, Validate)]
#[settings(prefix = "PROXIMA_MOE_EXPERT_PRECISION")]
#[builder(derive(Clone, Debug))]
#[serde(default)]
pub struct MoeExpertPrecisionSettings {
    /// `true` turns the gate-magnitude high/low decision on.
    #[setting(default = false)]
    #[builder(default = false)]
    pub enabled: bool,
    /// The paper's `T1`, in `0.0..=1.0`: the cumulative gate mass served
    /// high. The first selected expert always has score 0 and is high.
    #[setting(default = 0.6)]
    #[builder(default = 0.6)]
    pub high_threshold: f32,
    /// The sidecar target codec, spelled as a `proxima-numeric` format name.
    #[setting(resolve_with = "from_name", default_str = "q4_k")]
    #[builder(default = NumericFormat::Q4K)]
    pub low_codec: NumericFormat,
    /// The quality bound the HOBBIT arm must hold before its numbers count.
    #[setting(nested)]
    #[builder(default)]
    pub quality: HobbitQualitySettings,
}

impl Default for MoeExpertPrecisionSettings {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl MoeExpertPrecisionSettings {
    /// The threshold the decode policy reads, or `None` for the unchanged
    /// high-if-resident/low-otherwise path. Callers that can refuse a bad
    /// configuration use [`Self::checked_threshold`].
    #[must_use]
    pub const fn threshold(&self) -> Option<f32> {
        if self.enabled {
            Some(self.high_threshold)
        } else {
            None
        }
    }

    /// The threshold after the range check the feature gate applies. A
    /// caller that cannot proceed on a bad value (the example, a load path
    /// with a typed error) uses this instead of [`Self::threshold`].
    ///
    /// # Errors
    ///
    /// A message naming the field when the value is not a finite number in
    /// `0.0..=1.0`.
    /// The configured quality bound, or `None` when neither side is set.
    #[must_use]
    pub const fn quality_bound(&self) -> Option<(Option<f64>, Option<f64>)> {
        self.quality.bound()
    }

    pub fn checked_threshold(&self) -> Result<Option<f32>, String> {
        if !self.enabled {
            return Ok(None);
        }
        if !(0.0..=1.0).contains(&self.high_threshold) || !self.high_threshold.is_finite() {
            return Err(String::from(
                "moe_expert_precision.high_threshold must be a finite number in 0.0..=1.0",
            ));
        }
        Ok(Some(self.high_threshold))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const KEYS: [&str; 3] = [
        "PROXIMA_MOE_EXPERT_PRECISION_ENABLED",
        "PROXIMA_MOE_EXPERT_PRECISION_HIGH_THRESHOLD",
        "PROXIMA_MOE_EXPERT_PRECISION_LOW_CODEC",
    ];

    #[test]
    fn moe_expert_precision_default_is_off_and_threshold_none() {
        let defaults = MoeExpertPrecisionSettings::default();
        assert!(!defaults.enabled);
        assert_eq!(defaults.threshold(), None);
        assert_eq!(defaults.low_codec, NumericFormat::Q4K);
        let cleared = KEYS.iter().map(|key| (*key, None::<&str>)).collect::<Vec<_>>();
        temp_env::with_vars(cleared, || {
            let from_env = MoeExpertPrecisionSettings::from_env().expect("unset env");
            assert_eq!(from_env, defaults);
        });
    }

    #[test]
    fn moe_expert_precision_toml_env_and_builder_agree() {
        let via_builder = MoeExpertPrecisionSettings::builder()
            .enabled(true)
            .high_threshold(0.6)
            .low_codec(NumericFormat::Q3K)
            .build();
        let via_toml: MoeExpertPrecisionSettings = conflaguration::from_toml_str(
            "enabled = true\nhigh_threshold = 0.6\nlow_codec = \"q3_k\"\n",
        )
        .expect("toml");
        assert_eq!(via_toml, via_builder);
        assert_eq!(via_builder.threshold(), Some(0.6));
        let env = vec![
            ("PROXIMA_MOE_EXPERT_PRECISION_ENABLED", Some("true")),
            ("PROXIMA_MOE_EXPERT_PRECISION_HIGH_THRESHOLD", Some("0.6")),
            ("PROXIMA_MOE_EXPERT_PRECISION_LOW_CODEC", Some("q3_k")),
        ];
        temp_env::with_vars(env, || {
            let from_env = MoeExpertPrecisionSettings::from_env().expect("env");
            assert_eq!(from_env, via_builder);
        });
    }

    #[test]
    fn moe_expert_precision_lowers_through_serving_settings() {
        let toml_text = r#"
[moe_expert_precision]
enabled = true
high_threshold = 0.6
low_codec = "q4_k"
"#;
        let settings: crate::ServingSettings =
            conflaguration::from_toml_str(toml_text).expect("the serving settings load");
        assert_eq!(settings.moe_expert_precision.checked_threshold(), Ok(Some(0.6)));
        assert_eq!(
            settings.as_serving_config(&[]).moe_high_precision_threshold,
            Some(0.6)
        );
        assert_eq!(
            crate::ServingSettings::default()
                .as_serving_config(&[])
                .moe_high_precision_threshold,
            None,
            "the default path is unchanged"
        );
    }

    #[test]
    fn moe_expert_precision_rejects_out_of_range_threshold() {
        let bad = MoeExpertPrecisionSettings::builder()
            .enabled(true)
            .high_threshold(1.5)
            .build();
        let error = bad
            .checked_threshold()
            .expect_err("1.5 is outside 0..=1");
        assert!(error.contains("high_threshold"), "{error}");
        let nan = MoeExpertPrecisionSettings::builder()
            .enabled(true)
            .high_threshold(f32::NAN)
            .build();
        nan.checked_threshold().expect_err("NaN is not a threshold");
    }
}
