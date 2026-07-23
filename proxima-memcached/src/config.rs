//! `MemcachedServerConfig` — the facade's config-mirror surface (workspace
//! principle 4): one type is the bon builder result, the serde shape, and
//! the conflaguration env surface (`MEMCACHED_*`). Mirrors
//! `proxima_redis::config::RedisServerConfig`'s house pattern.

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
///     .read_buffer_bytes(4096)
///     .max_message_bytes(1_048_576)
///     .build();
///
/// let mut file = tempfile::Builder::new().suffix(".toml").tempfile().expect("tempfile");
/// write!(
///     file,
///     "read_buffer_bytes = 4096\nwrite_high_water_bytes = 65536\nmax_message_bytes = 1048576\n"
/// )
/// .expect("write toml");
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
    /// initial read-buffer size; the connection's buffer grows up to
    /// `max_message_bytes`
    #[setting(default = 8192)]
    #[serde(default = "default_read_buffer")]
    #[builder(default = default_read_buffer())]
    pub read_buffer_bytes: usize,

    /// write buffer flush threshold; replies accumulate in the
    /// connection's out buffer up to this many bytes before a socket write
    #[setting(default = 65536)]
    #[serde(default = "default_high_water")]
    #[builder(default = default_high_water())]
    pub write_high_water_bytes: usize,

    /// hard cap on one still-incomplete inbound command — the DoS guard
    /// `Connection::advance` enforces (`MessageTooLarge`)
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
        let config = MemcachedServerConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.read_buffer_bytes, 8192);
        assert_eq!(config.max_message_bytes, 16 * 1024 * 1024);
    }

    #[test]
    fn builder_overrides_defaults() {
        let config = MemcachedServerConfig::builder()
            .max_message_bytes(1024)
            .read_buffer_bytes(512)
            .build();
        assert_eq!(config.max_message_bytes, 1024);
        assert_eq!(config.read_buffer_bytes, 512);
    }

    #[test]
    fn validate_rejects_max_message_below_read_buffer() {
        let config = MemcachedServerConfig::builder()
            .read_buffer_bytes(4096)
            .max_message_bytes(1024)
            .build();
        assert!(config.validate().is_err());
    }
}
