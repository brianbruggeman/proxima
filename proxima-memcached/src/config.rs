//! `MemcachedServerConfig` — the facade's config-mirror surface (workspace
//! principle 4): one type is the bon builder result, the serde shape, and
//! the conflaguration env surface (`MEMCACHED_*`). Mirrors
//! `proxima_redis::config::RedisServerConfig`'s house pattern.

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

/// The shortest complete memcached command line: a three-character verb
/// plus CRLF (`get\r\n`). A `max_message_bytes` under this cannot hold even
/// that, so every command becomes a `MessageTooLarge` violation the moment
/// it arrives in more than one read — a misconfiguration, not a tight cap.
const MIN_COMMAND_BYTES: usize = 5;

fn default_max_message() -> usize {
    16 * 1024 * 1024
}

/// Memcached text-protocol server configuration.
///
/// Config is first-class in two equivalent forms — the fluent builder and a
/// TOML file loaded through `conflaguration` — and they produce the exact
/// same value:
///
/// ```
/// use std::io::Write;
///
/// use proxima_memcached::MemcachedServerConfig;
///
/// let via_builder = MemcachedServerConfig::builder()
///     .max_message_bytes(1_048_576)
///     .build();
///
/// let mut file = tempfile::Builder::new().suffix(".toml").tempfile().expect("tempfile");
/// write!(file, "max_message_bytes = 1048576\n").expect("write toml");
///
/// let via_toml: MemcachedServerConfig = conflaguration::builder()
///     .file(file.path())
///     .validate()
///     .build()
///     .expect("load from toml");
///
/// assert_eq!(via_builder, via_toml);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Builder, Serialize, Deserialize, Settings)]
#[settings(prefix = "MEMCACHED")]
#[builder(derive(Clone, Debug))]
pub struct MemcachedServerConfig {
    /// hard cap on one still-incomplete inbound command — the DoS guard
    /// [`proxima_protocols::memcached::frame_codec::MemcachedCodec`]'s
    /// `parse_frame` enforces, folding a longer still-incomplete buffer
    /// into a `MessageTooLarge` violation
    #[setting(default = 16777216)]
    #[serde(default = "default_max_message")]
    #[builder(default = default_max_message())]
    pub max_message_bytes: usize,
}

impl Default for MemcachedServerConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl Validate for MemcachedServerConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.max_message_bytes < MIN_COMMAND_BYTES {
            errors.push(ValidationMessage::new(
                "max_message_bytes",
                "must be at least 5 bytes (the shortest complete command line)",
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

    #[test]
    fn default_config_is_valid() {
        let config = MemcachedServerConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.max_message_bytes, 16 * 1024 * 1024);
    }

    #[test]
    fn builder_overrides_defaults() {
        let config = MemcachedServerConfig::builder()
            .max_message_bytes(1024)
            .build();
        assert_eq!(config.max_message_bytes, 1024);
    }

    #[test]
    fn validate_rejects_a_cap_below_the_shortest_complete_command() {
        let too_small = MemcachedServerConfig::builder().max_message_bytes(4).build();
        assert!(too_small.validate().is_err());
        let shortest = MemcachedServerConfig::builder().max_message_bytes(5).build();
        assert!(shortest.validate().is_ok());
    }

    /// The cap the crate's own `any_protocol` e2e suite drives a real
    /// socket with — it must not be a config the validator rejects.
    #[test]
    fn validate_accepts_the_tight_cap_the_end_to_end_suite_uses() {
        let config = MemcachedServerConfig::builder().max_message_bytes(8).build();
        assert!(config.validate().is_ok());
    }
}
