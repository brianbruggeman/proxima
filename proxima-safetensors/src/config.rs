//! Runtime configuration for [`crate::parser::SafetensorsParser`]
//! (`std`-tier only, mirroring `proxima-gguf/src/config.rs`'s "one type =
//! builder result = config" shape). `conflaguration` is std-only (see
//! `~/.claude/rules/rust.md`'s conflag skill and
//! `proxima-telemetry/src/config.rs`, the canonical example), so this
//! whole module lives behind the `std` feature -- the no_std+alloc floor
//! ([`crate::parser`], via [`crate::parser::SafetensorsParser::new`])
//! never sees it and always uses [`crate::sized`]'s constant directly.
//!
//! The bridge: `default_max_header_bytes` seeds from `crate::sized`, never
//! re-declaring the value, so a build-time floor and its std-tier runtime
//! default cannot silently drift apart
//! (`defaults_track_the_sized_floor` below pins the invariant).

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

/// Runtime policy for [`crate::parser::SafetensorsParser`]. Defaults from
/// [`crate::sized::MAX_HEADER_BYTES`] and may be overridden per process
/// (env `SAFETENSORS_MAX_HEADER_BYTES`, a config file, or the fluent
/// builder) -- e.g. a caller with a known-small model corpus lowering the
/// DOS-prevention cap tighter than the reference implementation's own
/// ceiling.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "SAFETENSORS")]
#[builder(derive(Clone, Debug))]
pub struct SafetensorsParserConfig {
    /// Declared header length above which a file is rejected as
    /// [`crate::error::SafetensorsError::HeaderTooLarge`] before any
    /// allocation for it happens. Build-time default:
    /// `crate::sized::MAX_HEADER_BYTES`.
    #[setting(default = 100_000_000)]
    #[serde(default = "default_max_header_bytes")]
    #[builder(default = default_max_header_bytes())]
    pub max_header_bytes: u64,
}

fn default_max_header_bytes() -> u64 {
    crate::sized::MAX_HEADER_BYTES
}

impl Default for SafetensorsParserConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl Validate for SafetensorsParserConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = alloc::vec::Vec::new();
        if self.max_header_bytes == 0 {
            errors.push(ValidationMessage::new("max_header_bytes", "must be > 0"));
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
    use crate::parser::SafetensorsParser;

    #[test]
    fn default_config_validates() {
        let config = SafetensorsParserConfig::default();
        assert!(config.validate().is_ok(), "default config should validate");
    }

    // the whole bridge: the runtime default must be SEEDED from the
    // build-time sized constant, never duplicated, so the two can never
    // silently drift apart.
    #[test]
    fn defaults_track_the_sized_floor() {
        let config = SafetensorsParserConfig::default();
        assert_eq!(config.max_header_bytes, crate::sized::MAX_HEADER_BYTES);

        // the env-overlay path (from_env, no vars set) must agree too --
        // guards against the #[setting] literal drifting from the const.
        temp_env::with_vars::<&str, &str, _, _>([], || {
            let from_env = SafetensorsParserConfig::from_env().expect("from_env");
            assert_eq!(
                from_env.max_header_bytes,
                crate::sized::MAX_HEADER_BYTES,
                "#[setting] max_header_bytes literal drifted from sized::MAX_HEADER_BYTES"
            );
        });
    }

    #[test]
    fn env_override_takes_effect_and_does_not_leak() {
        temp_env::with_vars([("SAFETENSORS_MAX_HEADER_BYTES", Some("64"))], || {
            let config = SafetensorsParserConfig::from_env().expect("from_env");
            assert_eq!(config.max_header_bytes, 64);
        });

        // outside the scoped block, the override must not leak into a
        // fresh read.
        let config = SafetensorsParserConfig::from_env().expect("from_env");
        assert_eq!(
            config.max_header_bytes,
            crate::sized::MAX_HEADER_BYTES,
            "env override leaked past its scope"
        );
    }

    #[test]
    fn zero_max_header_bytes_rejected() {
        let config = SafetensorsParserConfig::builder().max_header_bytes(0).build();
        let err = config
            .validate()
            .expect_err("validate must reject max_header_bytes = 0");
        assert!(format!("{err:?}").contains("max_header_bytes"));
    }

    // proves the config is actually wired into parsing behavior, not just
    // an inert struct: a lowered max_header_bytes rejects a header the
    // sized-floor default would have accepted.
    #[test]
    fn with_config_lowers_max_header_bytes_and_rejects_larger_headers() {
        let config = SafetensorsParserConfig::builder().max_header_bytes(4).build();
        let json = br#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut wire = alloc::vec::Vec::new();
        wire.extend_from_slice(&(json.len() as u64).to_le_bytes());
        wire.extend_from_slice(json);

        let outcome = SafetensorsParser::with_config(&config)
            .push(&wire)
            .and_then(|parser| parser.finish());
        assert!(matches!(
            outcome,
            Err(crate::error::SafetensorsError::HeaderTooLarge { max: 4, .. })
        ));
    }
}
