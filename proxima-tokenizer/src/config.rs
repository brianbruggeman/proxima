//! Runtime configuration (`std`-tier only, mirroring
//! `proxima-gguf/src/config.rs`). `conflaguration` is std-only, so this
//! whole module lives behind the `std` feature -- the no_std+alloc floor
//! never sees it and always uses [`crate::sized`]'s constants directly.
//!
//! The bridge: `default_max_input_bytes` seeds from `crate::sized`, never
//! re-declaring the value, so a build-time floor and its std-tier runtime
//! default cannot silently drift apart (`default_tracks_the_sized_floor`
//! below pins the invariant).

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

/// Runtime policy for [`crate::pipe::encode`] and friends. Defaults from
/// [`crate::sized`]'s build-time floor and may be overridden per process
/// (env `TOKENIZER_MAX_INPUT_BYTES`, a config file, or the fluent
/// builder).
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "TOKENIZER")]
#[builder(derive(Clone, Debug))]
pub struct TokenizerConfig {
    /// Largest single `encode` input accepted, in bytes. Build-time
    /// default: `crate::sized::MAX_INPUT_BYTES`.
    #[setting(default = 1_048_576)]
    #[serde(default = "default_max_input_bytes")]
    #[builder(default = default_max_input_bytes())]
    pub max_input_bytes: usize,
}

fn default_max_input_bytes() -> usize {
    crate::sized::MAX_INPUT_BYTES
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl Validate for TokenizerConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = alloc::vec::Vec::new();
        if self.max_input_bytes == 0 {
            errors.push(ValidationMessage::new("max_input_bytes", "must be > 0"));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn default_config_validates() {
        let config = TokenizerConfig::default();
        assert!(config.validate().is_ok(), "default config should validate");
    }

    // the whole bridge: the runtime default must be SEEDED from the
    // build-time sized constant, never duplicated, so the two can never
    // silently drift apart.
    #[test]
    fn default_tracks_the_sized_floor() {
        let config = TokenizerConfig::default();
        assert_eq!(config.max_input_bytes, crate::sized::MAX_INPUT_BYTES);

        // the env-overlay path (from_env, no vars set) must agree too --
        // guards against the #[setting] literal drifting from the const.
        temp_env::with_vars::<&str, &str, _, _>([], || {
            let from_env = TokenizerConfig::from_env().expect("from_env");
            assert_eq!(
                from_env.max_input_bytes,
                crate::sized::MAX_INPUT_BYTES,
                "#[setting] max_input_bytes literal drifted from sized::MAX_INPUT_BYTES"
            );
        });
    }

    #[test]
    fn env_override_takes_effect_and_does_not_leak() {
        temp_env::with_vars([("TOKENIZER_MAX_INPUT_BYTES", Some("128"))], || {
            let config = TokenizerConfig::from_env().expect("from_env");
            assert_eq!(config.max_input_bytes, 128);
        });

        let config = TokenizerConfig::from_env().expect("from_env");
        assert_eq!(
            config.max_input_bytes,
            crate::sized::MAX_INPUT_BYTES,
            "env override leaked past its scope"
        );
    }

    #[test]
    fn zero_max_input_bytes_rejected() {
        let config = TokenizerConfig::builder().max_input_bytes(0).build();
        let err = config
            .validate()
            .expect_err("validate must reject max_input_bytes = 0");
        assert!(format!("{err:?}").contains("max_input_bytes"));
    }
}
