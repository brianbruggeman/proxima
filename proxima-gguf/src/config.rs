//! Runtime configuration for [`crate::parser::GgufParser`] (`std`-tier
//! only, mirroring `proxima-clock/src/config.rs`'s "one type = builder
//! result = config" shape). `conflaguration` is std-only (see
//! `~/.claude/rules/rust.md`'s conflag skill and
//! `proxima-telemetry/src/config.rs`, the canonical example), so this
//! whole module lives behind the `std` feature -- the no_std+alloc floor
//! ([`crate::parser`], via [`crate::parser::GgufParser::new`]) never sees
//! it and always uses [`crate::sized`]'s constants directly.
//!
//! The bridge: `default_max_supported_version`/`default_alignment` seed
//! from `crate::sized`, never re-declaring the value, so a build-time
//! floor and its std-tier runtime default cannot silently drift apart
//! (`defaults_track_the_sized_floor` below pins the invariant).

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

/// Runtime policy for [`crate::parser::GgufParser`]. Both fields default
/// from [`crate::sized`]'s build-time floor and may be overridden per
/// process (env `GGUF_MAX_SUPPORTED_VERSION` / `GGUF_DEFAULT_ALIGNMENT`,
/// a config file, or the fluent builder).
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "GGUF")]
#[builder(derive(Clone, Debug))]
pub struct GgufParserConfig {
    /// Newest GGUF version accepted; a file declaring a higher version is
    /// rejected as [`crate::error::GgufError::UnsupportedVersion`].
    /// Build-time default: `crate::sized::MAX_SUPPORTED_VERSION`.
    #[setting(default = 3)]
    #[serde(default = "default_max_supported_version")]
    #[builder(default = default_max_supported_version())]
    pub max_supported_version: u32,

    /// Fallback tensor-data alignment used when a file omits
    /// `general.alignment`. Build-time default:
    /// `crate::sized::DEFAULT_ALIGNMENT`.
    #[setting(default = 32)]
    #[serde(default = "default_alignment")]
    #[builder(default = default_alignment())]
    pub default_alignment: u32,
}

fn default_max_supported_version() -> u32 {
    crate::sized::MAX_SUPPORTED_VERSION
}

fn default_alignment() -> u32 {
    crate::sized::DEFAULT_ALIGNMENT
}

impl Default for GgufParserConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl Validate for GgufParserConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = alloc::vec::Vec::new();
        if self.max_supported_version == 0 {
            errors.push(ValidationMessage::new(
                "max_supported_version",
                "must be > 0",
            ));
        }
        if self.default_alignment == 0 || !self.default_alignment.is_power_of_two() {
            errors.push(ValidationMessage::new(
                "default_alignment",
                "must be a nonzero power of two",
            ));
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
    use crate::parser::GgufParser;

    #[test]
    fn default_config_validates() {
        let config = GgufParserConfig::default();
        assert!(config.validate().is_ok(), "default config should validate");
    }

    // the whole bridge: the runtime default must be SEEDED from the
    // build-time sized constant, never duplicated, so the two can never
    // silently drift apart.
    #[test]
    fn defaults_track_the_sized_floor() {
        let config = GgufParserConfig::default();
        assert_eq!(
            config.max_supported_version,
            crate::sized::MAX_SUPPORTED_VERSION
        );
        assert_eq!(config.default_alignment, crate::sized::DEFAULT_ALIGNMENT);

        // the env-overlay path (from_env, no vars set) must agree too --
        // guards against the #[setting] literal drifting from the const.
        temp_env::with_vars::<&str, &str, _, _>([], || {
            let from_env = GgufParserConfig::from_env().expect("from_env");
            assert_eq!(
                from_env.max_supported_version,
                crate::sized::MAX_SUPPORTED_VERSION,
                "#[setting] max_supported_version literal drifted from sized::MAX_SUPPORTED_VERSION"
            );
            assert_eq!(
                from_env.default_alignment,
                crate::sized::DEFAULT_ALIGNMENT,
                "#[setting] default_alignment literal drifted from sized::DEFAULT_ALIGNMENT"
            );
        });
    }

    #[test]
    fn env_override_takes_effect_and_does_not_leak() {
        temp_env::with_vars(
            [
                ("GGUF_MAX_SUPPORTED_VERSION", Some("2")),
                ("GGUF_DEFAULT_ALIGNMENT", Some("64")),
            ],
            || {
                let config = GgufParserConfig::from_env().expect("from_env");
                assert_eq!(config.max_supported_version, 2);
                assert_eq!(config.default_alignment, 64);
            },
        );

        // outside the scoped block, the override must not leak into a
        // fresh read.
        let config = GgufParserConfig::from_env().expect("from_env");
        assert_eq!(
            config.max_supported_version,
            crate::sized::MAX_SUPPORTED_VERSION,
            "env override leaked past its scope"
        );
        assert_eq!(config.default_alignment, crate::sized::DEFAULT_ALIGNMENT);
    }

    #[test]
    fn zero_max_supported_version_rejected() {
        let config = GgufParserConfig::builder().max_supported_version(0).build();
        let err = config
            .validate()
            .expect_err("validate must reject max_supported_version = 0");
        assert!(format!("{err:?}").contains("max_supported_version"));
    }

    #[test]
    fn non_power_of_two_alignment_rejected() {
        let config = GgufParserConfig::builder().default_alignment(3).build();
        let err = config
            .validate()
            .expect_err("validate must reject a non-power-of-two alignment");
        assert!(format!("{err:?}").contains("default_alignment"));
    }

    // proves the config is actually wired into parsing behavior, not just
    // an inert struct: a lowered max_supported_version rejects a version
    // the sized-floor default would have accepted.
    #[test]
    fn with_config_lowers_max_supported_version_and_rejects_newer_files() {
        let config = GgufParserConfig::builder().max_supported_version(2).build();
        let mut parser = GgufParser::with_config(&config);
        let mut bytes = Vec::from(crate::parser::MAGIC);
        bytes.extend_from_slice(&3u32.to_le_bytes()); // version 3
        parser.feed(&bytes);
        let err = parser.poll().expect_err("version 3 must be rejected");
        assert!(matches!(
            err,
            crate::error::GgufError::UnsupportedVersion { version: 3 }
        ));
    }

    // proves default_alignment is actually consulted for the fallback
    // path (no `general.alignment` key present).
    #[test]
    fn with_config_changes_fallback_alignment() {
        let config = GgufParserConfig::builder().default_alignment(64).build();
        let mut parser = GgufParser::with_config(&config);
        let mut bytes = Vec::from(crate::parser::MAGIC);
        bytes.extend_from_slice(&3u32.to_le_bytes()); // version
        bytes.extend_from_slice(&0i64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&0i64.to_le_bytes()); // kv_count
        parser.feed(&bytes);
        // header event
        parser.poll().expect("header");
        // kv_count == 0 -> resolves alignment and moves to tensor phase,
        // then tensor remaining == 0 -> Complete.
        let outcome = parser.poll().expect("complete");
        match outcome {
            Some(crate::parser::GgufEvent::Complete { alignment, .. }) => {
                assert_eq!(alignment, 64, "fallback alignment came from the config");
            }
            other => panic!("expected Complete, got {other:?}"),
        }
    }
}
