//! `KafkaServerConfig` — the facade's config-mirror surface (workspace
//! principle 4): one type is the bon builder result, the serde shape, and
//! the conflaguration env surface (`KAFKA_*`). Mirrors
//! `proxima_redis::config::RedisServerConfig`'s house pattern.

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

use crate::any_protocol::MIN_V0_HEADER_BYTES;

/// The 4-byte length prefix plus the smallest v0 request header — a
/// `max_message_bytes` under this rejects every request the facade can
/// receive, so it is a misconfiguration, not a tight cap.
const MIN_V0_REQUEST_BYTES: usize = 4 + MIN_V0_HEADER_BYTES as usize;

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
///
/// Config is first-class in two equivalent forms — the fluent builder and a
/// TOML file loaded through `conflaguration` — and they produce the exact
/// same value:
///
/// ```
/// use std::io::Write;
///
/// use proxima_kafka::KafkaServerConfig;
///
/// let via_builder = KafkaServerConfig::builder()
///     .broker_id(1)
///     .advertised_host("broker.internal")
///     .build();
///
/// let mut file = tempfile::Builder::new().suffix(".toml").tempfile().expect("tempfile");
/// write!(
///     file,
///     "broker_id = 1\nadvertised_host = \"broker.internal\"\n"
/// )
/// .expect("write toml");
///
/// let via_toml: KafkaServerConfig = conflaguration::builder()
///     .file(file.path())
///     .validate()
///     .build()
///     .expect("load from toml");
///
/// assert_eq!(via_builder, via_toml);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Builder, Serialize, Deserialize, Settings)]
#[settings(prefix = "KAFKA")]
#[builder(derive(Clone, Debug))]
pub struct KafkaServerConfig {
    /// hard cap on one still-incomplete inbound frame — the DoS guard
    /// [`crate::frame_codec::KafkaCodec`]'s `parse_frame` enforces, folding
    /// a larger declared frame into a `MessageTooLarge` violation
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
        if self.max_message_bytes < MIN_V0_REQUEST_BYTES {
            errors.push(ValidationMessage::new(
                "max_message_bytes",
                "must be at least 14 bytes (a v0 frame prefix plus its smallest header)",
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
        assert_eq!(config.max_message_bytes, 16 * 1024 * 1024);
        assert_eq!(config.advertised_port, 9092);
    }

    #[test]
    fn builder_overrides_defaults() {
        let config = KafkaServerConfig::builder()
            .max_message_bytes(1024)
            .broker_id(7)
            .build();
        assert_eq!(config.max_message_bytes, 1024);
        assert_eq!(config.broker_id, 7);
    }

    #[test]
    fn validate_rejects_a_cap_below_the_smallest_v0_request() {
        let config = KafkaServerConfig::builder().max_message_bytes(13).build();
        assert!(config.validate().is_err());
        let smallest = KafkaServerConfig::builder().max_message_bytes(14).build();
        assert!(smallest.validate().is_ok());
    }

    #[test]
    fn validate_rejects_empty_advertised_host() {
        let config = KafkaServerConfig::builder().advertised_host("").build();
        assert!(config.validate().is_err());
    }
}
