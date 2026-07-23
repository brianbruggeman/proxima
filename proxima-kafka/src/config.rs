//! `KafkaServerConfig` — the facade's config-mirror surface (workspace
//! principle 4): one type is the bon builder result, the serde shape, and
//! the conflaguration env surface (`KAFKA_*`). Mirrors
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

fn default_broker_id() -> i32 {
    0
}

fn default_host() -> String {
    "localhost".to_string()
}

fn default_port() -> i32 {
    9092
}

/// Kafka wire server configuration.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Serialize, Deserialize, Settings)]
#[settings(prefix = "KAFKA")]
#[builder(derive(Clone, Debug))]
pub struct KafkaServerConfig {
    /// initial read-buffer size; the connection's buffer grows up to
    /// `max_message_bytes`
    #[setting(default = 8192)]
    #[serde(default = "default_read_buffer")]
    #[builder(default = default_read_buffer())]
    pub read_buffer_bytes: usize,

    /// write buffer flush threshold; a reply accumulates in the
    /// connection's out buffer up to this many bytes before a socket write
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

    /// the `node_id` this facade reports in a `Metadata` response's broker
    /// list — a real client uses it to pick which broker to connect the
    /// next request to; this facade always answers as the one broker it is
    #[setting(default = 0)]
    #[serde(default = "default_broker_id")]
    #[builder(default = default_broker_id())]
    pub broker_id: i32,

    /// the host this facade advertises in `Metadata` responses
    #[setting(default = "localhost")]
    #[serde(default = "default_host")]
    #[builder(default = default_host(), into)]
    pub advertised_host: String,

    /// the port this facade advertises in `Metadata` responses
    #[setting(default = 9092)]
    #[serde(default = "default_port")]
    #[builder(default = default_port())]
    pub advertised_port: i32,
}

impl Default for KafkaServerConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl Validate for KafkaServerConfig {
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
        if self.advertised_host.is_empty() {
            errors.push(ValidationMessage::new(
                "advertised_host",
                "must be non-empty",
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
        let config = KafkaServerConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.read_buffer_bytes, 8192);
        assert_eq!(config.max_message_bytes, 16 * 1024 * 1024);
        assert_eq!(config.advertised_port, 9092);
    }

    #[test]
    fn builder_overrides_defaults() {
        let config = KafkaServerConfig::builder()
            .max_message_bytes(1024)
            .read_buffer_bytes(512)
            .broker_id(7)
            .build();
        assert_eq!(config.max_message_bytes, 1024);
        assert_eq!(config.read_buffer_bytes, 512);
        assert_eq!(config.broker_id, 7);
    }

    #[test]
    fn validate_rejects_max_message_below_read_buffer() {
        let config = KafkaServerConfig::builder()
            .read_buffer_bytes(4096)
            .max_message_bytes(1024)
            .build();
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_empty_advertised_host() {
        let config = KafkaServerConfig::builder().advertised_host("").build();
        assert!(config.validate().is_err());
    }
}
