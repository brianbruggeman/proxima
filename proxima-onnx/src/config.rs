//! Runtime configuration for [`crate::parser::OnnxParser`] (`std`-tier
//! only, mirroring `proxima-clock/src/config.rs`'s "one type = builder
//! result = config" shape). `conflaguration` is std-only (see
//! `~/.claude/rules/rust.md`'s conflag skill and
//! `proxima-telemetry/src/config.rs`, the canonical example), so this
//! whole module lives behind the `std` feature -- the no_std+alloc floor
//! ([`crate::parser`], via [`crate::parser::OnnxParser::new`]) never sees
//! it and always uses [`crate::sized`]'s constant directly.
//!
//! The bridge: `default_max_len_delimited_field` seeds from
//! [`crate::sized::MAX_LEN_DELIMITED_FIELD`], never re-declaring the
//! value, so the build-time floor and its std-tier runtime default cannot
//! silently drift apart (`defaults_track_the_sized_floor` below pins the
//! invariant).

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

/// Runtime policy for [`crate::parser::OnnxParser`]. Defaults from
/// [`crate::sized::MAX_LEN_DELIMITED_FIELD`] and may be overridden per
/// process (env `ONNX_MAX_LEN_DELIMITED_FIELD`, a config file, or the
/// fluent builder).
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "ONNX")]
#[builder(derive(Clone, Debug))]
pub struct OnnxParserConfig {
    /// Sanity cap on a length-delimited field's declared length -- a file
    /// declaring a longer field fails fast as
    /// [`crate::error::OnnxError::DeclaredLengthTooLarge`] instead of the
    /// FSM buffering forever. Build-time default:
    /// `crate::sized::MAX_LEN_DELIMITED_FIELD`.
    #[setting(default = 1_099_511_627_776)]
    #[serde(default = "default_max_len_delimited_field")]
    #[builder(default = default_max_len_delimited_field())]
    pub max_len_delimited_field: u64,
}

fn default_max_len_delimited_field() -> u64 {
    crate::sized::MAX_LEN_DELIMITED_FIELD
}

impl Default for OnnxParserConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl Validate for OnnxParserConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        if self.max_len_delimited_field == 0 {
            Err(conflaguration::Error::Validation {
                errors: alloc::vec![ValidationMessage::new(
                    "max_len_delimited_field",
                    "must be > 0"
                )],
            })
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::parser::OnnxParser;

    #[test]
    fn default_config_validates() {
        let config = OnnxParserConfig::default();
        assert!(config.validate().is_ok(), "default config should validate");
    }

    // the whole bridge: the runtime default must be SEEDED from the
    // build-time sized constant, never duplicated, so the two can never
    // silently drift apart.
    #[test]
    fn defaults_track_the_sized_floor() {
        let config = OnnxParserConfig::default();
        assert_eq!(
            config.max_len_delimited_field,
            crate::sized::MAX_LEN_DELIMITED_FIELD
        );

        // the env-overlay path (from_env, no vars set) must agree too --
        // guards against the #[setting] literal drifting from the const.
        temp_env::with_vars::<&str, &str, _, _>([], || {
            let from_env = OnnxParserConfig::from_env().expect("from_env");
            assert_eq!(
                from_env.max_len_delimited_field,
                crate::sized::MAX_LEN_DELIMITED_FIELD,
                "#[setting] max_len_delimited_field literal drifted from sized::MAX_LEN_DELIMITED_FIELD"
            );
        });
    }

    #[test]
    fn env_override_takes_effect_and_does_not_leak() {
        temp_env::with_vars([("ONNX_MAX_LEN_DELIMITED_FIELD", Some("4096"))], || {
            let config = OnnxParserConfig::from_env().expect("from_env");
            assert_eq!(config.max_len_delimited_field, 4096);
        });

        // outside the scoped block, the override must not leak into a
        // fresh read.
        let config = OnnxParserConfig::from_env().expect("from_env");
        assert_eq!(
            config.max_len_delimited_field,
            crate::sized::MAX_LEN_DELIMITED_FIELD,
            "env override leaked past its scope"
        );
    }

    #[test]
    fn zero_max_len_delimited_field_rejected() {
        let config = OnnxParserConfig::builder()
            .max_len_delimited_field(0)
            .build();
        let err = config
            .validate()
            .expect_err("validate must reject max_len_delimited_field = 0");
        assert!(format!("{err:?}").contains("max_len_delimited_field"));
    }

    // proves the config is actually wired into parsing behavior, not just
    // an inert struct: a lowered cap rejects a declared length the
    // sized-floor default would have accepted.
    #[test]
    fn with_config_lowers_the_cap_and_rejects_a_previously_accepted_length() {
        let config = OnnxParserConfig::builder()
            .max_len_delimited_field(1024)
            .build();
        let mut parser = OnnxParser::with_config(&config);

        // field 7 (graph), len-delimited, declares 2048 bytes -- under the
        // sized-floor default (2^40) but over this config's 1024-byte cap.
        let mut bytes = alloc::vec::Vec::new();
        push_tag(7, 2, &mut bytes);
        push_varint(2048, &mut bytes);
        parser.feed(&bytes);
        let err = parser
            .poll()
            .expect_err("declared length exceeds config cap");
        assert!(matches!(
            err,
            crate::error::OnnxError::DeclaredLengthTooLarge {
                declared: 2048,
                cap: 1024,
                ..
            }
        ));
    }

    fn push_tag(field_number: u32, wire_type: u8, out: &mut alloc::vec::Vec<u8>) {
        push_varint(u64::from((field_number << 3) | u32::from(wire_type)), out);
    }

    fn push_varint(mut value: u64, out: &mut alloc::vec::Vec<u8>) {
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if value == 0 {
                break;
            }
        }
    }
}
