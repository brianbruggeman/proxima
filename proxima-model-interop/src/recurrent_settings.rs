//! `RecurrentSettings`: the conflaguration-facing (env/TOML) loader and `bon`
//! fluent builder for [`crate::RecurrentConfig`]'s data, the same split and
//! pattern as [`crate::PromptCacheSettings`] (see that module's doc for why the
//! `Copy`, no_std-safe config lives in its own module and the owned
//! builder/loader shape lives here, `std`-gated).
//!
//! The mode spells the same way in every route -- the builder takes the
//! [`LoopCdMode`], a TOML file and `PROXIMA_RECURRENT_MODE` both say
//! `disabled`, `fixed_logits`, `adaptive_logits` or `fixed_hidden` -- because
//! the env route parses through the type's own serde spelling rather than a
//! second table of names.

use bon::Builder;
use conflaguration::{Settings, Validate};
use proxima_tensor::spec::LoopCdMode;
use serde::de::IntoDeserializer;
use serde::de::value::{Error as SpellingError, StrDeserializer};
use serde::{Deserialize, Serialize};

use crate::error::InteropError;
use crate::recurrent_config::RecurrentConfig;

fn parse_loopcd_mode(value: &str) -> Result<LoopCdMode, SpellingError> {
    let spelling: StrDeserializer<'_, SpellingError> = value.into_deserializer();
    LoopCdMode::deserialize(spelling)
}

/// The conflaguration/`bon` mirror of [`crate::RecurrentConfig`]; the builder,
/// a TOML file and `PROXIMA_RECURRENT_*` env vars all produce the same value,
/// and [`Self::as_recurrent_config`] lowers it into the `Copy` shape
/// [`crate::ServingConfig::recurrent`] carries.
#[derive(Debug, Clone, PartialEq, Builder, Deserialize, Serialize, Settings, Validate)]
#[settings(prefix = "PROXIMA_RECURRENT")]
#[builder(derive(Clone, Debug))]
#[serde(deny_unknown_fields)]
pub struct RecurrentSettings {
    /// See [`crate::RecurrentConfig::iterations`]. Absent takes the
    /// checkpoint's own depth.
    #[serde(default)]
    pub iterations: Option<u32>,
    /// See [`crate::RecurrentConfig::mode`].
    #[setting(resolve_with = "parse_loopcd_mode", default_str = "disabled")]
    #[builder(default)]
    #[serde(default)]
    pub mode: LoopCdMode,
    /// See [`crate::RecurrentConfig::strength`].
    #[setting(default = 0.0)]
    #[builder(default = 0.0)]
    #[serde(default)]
    pub strength: f32,
    /// See [`crate::RecurrentConfig::reference_iteration`].
    #[setting(default = 1)]
    #[builder(default = 1)]
    #[serde(default = "default_reference_iteration")]
    pub reference_iteration: u32,
}

const fn default_reference_iteration() -> u32 {
    1
}

impl RecurrentSettings {
    /// The `Copy` shape `ServingConfig::recurrent` carries. An explicit depth
    /// is validated whole; an open depth is validated against the one thing
    /// known here, a positive reference pass, and left open for the model to
    /// resolve against its checkpoint at load.
    ///
    /// # Errors
    ///
    /// [`InteropError::InvalidRecurrentConfig`] for a policy no depth could
    /// make valid.
    pub fn as_recurrent_config(&self) -> Result<RecurrentConfig, InteropError> {
        let config = RecurrentConfig {
            iterations: self.iterations,
            mode: self.mode,
            strength: self.strength,
            reference_iteration: self.reference_iteration,
        };
        match self.iterations {
            Some(iterations) => config.resolve(iterations),
            None => config
                .resolve(self.reference_iteration)
                .map(|checked| RecurrentConfig {
                    iterations: None,
                    ..checked
                }),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::io::Write;

    use tempfile::NamedTempFile;

    use super::*;

    const ENV_KEYS: [&str; 4] = [
        "PROXIMA_RECURRENT_ITERATIONS",
        "PROXIMA_RECURRENT_MODE",
        "PROXIMA_RECURRENT_STRENGTH",
        "PROXIMA_RECURRENT_REFERENCE_ITERATION",
    ];

    fn cleared_env() -> Vec<(&'static str, Option<&'static str>)> {
        ENV_KEYS.iter().map(|key| (*key, None)).collect()
    }

    fn guided_env() -> Vec<(&'static str, Option<&'static str>)> {
        vec![
            ("PROXIMA_RECURRENT_ITERATIONS", Some("2")),
            ("PROXIMA_RECURRENT_MODE", Some("fixed_hidden")),
            ("PROXIMA_RECURRENT_STRENGTH", Some("0.5")),
            ("PROXIMA_RECURRENT_REFERENCE_ITERATION", Some("1")),
        ]
    }

    fn guided_builder() -> RecurrentSettings {
        RecurrentSettings::builder()
            .iterations(2)
            .mode(LoopCdMode::FixedHidden)
            .strength(0.5)
            .reference_iteration(1)
            .build()
    }

    #[test]
    fn builder_toml_and_env_build_the_same_settings() {
        let mut file = NamedTempFile::with_suffix(".toml").expect("create recurrent config file");
        writeln!(
            file,
            "iterations = 2\nmode = \"fixed_hidden\"\nstrength = 0.5\nreference_iteration = 1"
        )
        .expect("write recurrent config file");

        let from_file: RecurrentSettings =
            conflaguration::from_file(file.path()).expect("load recurrent config file");
        let from_env = temp_env::with_vars(guided_env(), || {
            RecurrentSettings::from_env().expect("load recurrent config env")
        });

        assert_eq!(guided_builder(), from_file);
        assert_eq!(guided_builder(), from_env);
    }

    #[test]
    fn defaults_are_the_neutral_policy_on_every_route() {
        let from_env = temp_env::with_vars(cleared_env(), || {
            RecurrentSettings::from_env().expect("defaults")
        });
        let from_builder = RecurrentSettings::builder().build();
        let from_toml: RecurrentSettings =
            toml::from_str("").expect("an empty file is the default");

        for settings in [from_env, from_builder, from_toml] {
            assert_eq!(
                settings.as_recurrent_config().expect("neutral lowers"),
                RecurrentConfig::neutral()
            );
        }
    }

    #[test]
    fn settings_lower_to_the_serving_shape() {
        let lowered = guided_builder()
            .as_recurrent_config()
            .expect("valid policy");

        assert_eq!(lowered.iterations, Some(2));
        assert_eq!(lowered.mode, LoopCdMode::FixedHidden);
        assert_eq!(lowered.strength.to_bits(), 0.5f32.to_bits());
        assert_eq!(lowered.reference_iteration, 1);
    }

    #[test]
    fn unusable_policies_do_not_lower() {
        let wrong = |settings: RecurrentSettings| settings.as_recurrent_config().is_err();

        assert!(wrong(
            RecurrentSettings::builder()
                .iterations(2)
                .strength(0.5)
                .build()
        ));
        assert!(wrong(RecurrentSettings::builder().iterations(0).build()));
        assert!(wrong(
            RecurrentSettings::builder()
                .mode(LoopCdMode::FixedLogits)
                .strength(f32::NAN)
                .build()
        ));
        assert!(wrong(
            RecurrentSettings::builder()
                .mode(LoopCdMode::FixedLogits)
                .strength(-1.0)
                .build()
        ));
        assert!(wrong(
            RecurrentSettings::builder().reference_iteration(0).build()
        ));
        assert!(wrong(
            RecurrentSettings::builder()
                .iterations(2)
                .reference_iteration(3)
                .build()
        ));
    }

    #[test]
    fn an_unknown_mode_spelling_is_rejected_on_the_env_route() {
        let mut env = guided_env();
        env[1] = ("PROXIMA_RECURRENT_MODE", Some("FixedHidden"));

        let outcome = temp_env::with_vars(env, RecurrentSettings::from_env);

        assert!(outcome.is_err(), "{outcome:?}");
    }
}
