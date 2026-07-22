//! `RedisServerConfig` — the facade's config-mirror surface (workspace
//! principle 4): one type is the bon builder result, the serde shape, and
//! the conflaguration env surface (`REDIS_*`). Mirrors
//! `proxima_pgwire::config::PgServerConfig`'s house pattern.

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

fn default_read_buffer() -> usize {
    8 * 1024
}

fn default_high_water() -> usize {
    64 * 1024
}

fn default_max_message() -> usize {
    16 * 1024 * 1024
}

/// Redis/Valkey wire server configuration.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Serialize, Deserialize, Settings)]
#[settings(prefix = "REDIS")]
#[builder(derive(Clone, Debug))]
pub struct RedisServerConfig {
    /// initial read-buffer size; the connection's buffer grows up to
    /// `max_message_bytes`
    #[setting(default = 8192)]
    #[serde(default = "default_read_buffer")]
    #[builder(default = default_read_buffer())]
    pub read_buffer_bytes: usize,

    /// write buffer flush threshold; replies (and pushed pub/sub frames)
    /// accumulate in the connection's out buffer up to this many bytes
    /// before a socket write
    #[setting(default = 65536)]
    #[serde(default = "default_high_water")]
    #[builder(default = default_high_water())]
    pub write_high_water_bytes: usize,

    /// hard cap on one still-incomplete inbound frame — the DoS guard
    /// `Connection::advance` enforces (`MessageTooLarge`)
    #[setting(default = 16777216)]
    #[serde(default = "default_max_message")]
    #[builder(default = default_max_message())]
    pub max_message_bytes: usize,
}

impl Default for RedisServerConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl Validate for RedisServerConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.read_buffer_bytes < 64 {
            errors.push(ValidationMessage::new(
                "read_buffer_bytes",
                "must be at least 64 bytes",
            ));
        }
        if self.max_message_bytes < self.read_buffer_bytes {
            errors.push(ValidationMessage::new(
                "max_message_bytes",
                "must be at least read_buffer_bytes",
            ));
        }
        if self.write_high_water_bytes < 1024 {
            errors.push(ValidationMessage::new(
                "write_high_water_bytes",
                "must be at least 1024",
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
        let config = RedisServerConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.read_buffer_bytes, 8192);
        assert_eq!(config.max_message_bytes, 16 * 1024 * 1024);
    }

    #[test]
    fn builder_overrides_defaults() {
        let config = RedisServerConfig::builder()
            .max_message_bytes(1024)
            .read_buffer_bytes(512)
            .build();
        assert_eq!(config.max_message_bytes, 1024);
        assert_eq!(config.read_buffer_bytes, 512);
    }

    #[test]
    fn validate_rejects_max_message_below_read_buffer() {
        let config = RedisServerConfig::builder()
            .read_buffer_bytes(4096)
            .max_message_bytes(1024)
            .build();
        assert!(config.validate().is_err());
    }
}
