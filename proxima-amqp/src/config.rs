//! `AmqpServerConfig` — the facade's config-mirror surface (workspace
//! principle 4): one type is the bon builder result, the serde shape, and
//! the conflaguration env surface (`AMQP_*`). Mirrors
//! `proxima_redis::config::RedisServerConfig`'s house pattern; the extra
//! fields (`channel_max`, `frame_max_bytes`, `heartbeat_seconds`) are the
//! values `connection.tune` negotiates — AMQP's own DoS caps, layered on
//! top of the byte-buffer caps redis's config already has.

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

fn default_read_buffer() -> usize {
    8 * 1024
}

fn default_high_water() -> usize {
    64 * 1024
}

fn default_frame_max_bytes() -> usize {
    131_072
}

fn default_message_max_bytes() -> usize {
    16 * 1024 * 1024
}

fn default_channel_max() -> u16 {
    2047
}

fn default_heartbeat_seconds() -> u16 {
    60
}

/// AMQP 0-9-1 broker-listener configuration.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Serialize, Deserialize, Settings)]
#[settings(prefix = "AMQP")]
#[builder(derive(Clone, Debug))]
pub struct AmqpServerConfig {
    /// initial read-buffer size; the connection's buffer grows up to
    /// `message_max_bytes`
    #[setting(default = 8192)]
    #[serde(default = "default_read_buffer")]
    #[builder(default = default_read_buffer())]
    pub read_buffer_bytes: usize,

    /// write buffer flush threshold; replies (and pushed `basic.deliver`
    /// frames) accumulate in the connection's out buffer up to this many
    /// bytes before a socket write
    #[setting(default = 65536)]
    #[serde(default = "default_high_water")]
    #[builder(default = default_high_water())]
    pub write_high_water_bytes: usize,

    /// the `frame-max` this broker advertises in `connection.tune` — the
    /// hard cap on one method/header/body frame's payload, enforced by the
    /// sans-IO [`crate::fsm::Connection`] regardless of what a client's
    /// `tune-ok` claims (a client cannot negotiate a larger cap than the
    /// server offered)
    #[setting(default = 131072)]
    #[serde(default = "default_frame_max_bytes")]
    #[builder(default = default_frame_max_bytes())]
    pub frame_max_bytes: usize,

    /// hard cap on one reassembled `basic.publish` message body (the sum
    /// of every content-body frame between a method and its declared
    /// `body_size`) — the DoS guard against a client declaring a huge
    /// `body_size` and trickling bytes forever
    #[setting(default = 16777216)]
    #[serde(default = "default_message_max_bytes")]
    #[builder(default = default_message_max_bytes())]
    pub message_max_bytes: usize,

    /// the `channel-max` this broker advertises in `connection.tune` — the
    /// hard cap on distinct channel numbers one connection may open
    #[setting(default = 2047)]
    #[serde(default = "default_channel_max")]
    #[builder(default = default_channel_max())]
    pub channel_max: u16,

    /// the `heartbeat` interval (seconds) this broker advertises in
    /// `connection.tune`; `0` disables heartbeats
    #[setting(default = 60)]
    #[serde(default = "default_heartbeat_seconds")]
    #[builder(default = default_heartbeat_seconds())]
    pub heartbeat_seconds: u16,
}

impl Default for AmqpServerConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl Validate for AmqpServerConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.read_buffer_bytes < 64 {
            errors.push(ValidationMessage::new(
                "read_buffer_bytes",
                "must be at least 64 bytes",
            ));
        }
        if self.message_max_bytes < self.read_buffer_bytes {
            errors.push(ValidationMessage::new(
                "message_max_bytes",
                "must be at least read_buffer_bytes",
            ));
        }
        if self.write_high_water_bytes < 1024 {
            errors.push(ValidationMessage::new(
                "write_high_water_bytes",
                "must be at least 1024",
            ));
        }
        if self.frame_max_bytes < 4096 {
            errors.push(ValidationMessage::new(
                "frame_max_bytes",
                "must be at least 4096 bytes (the AMQP 0-9-1 spec floor)",
            ));
        }
        if self.channel_max == 0 {
            errors.push(ValidationMessage::new(
                "channel_max",
                "must be non-zero (0 means unlimited, not implemented here)",
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
        let config = AmqpServerConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.read_buffer_bytes, 8192);
        assert_eq!(config.channel_max, 2047);
        assert_eq!(config.heartbeat_seconds, 60);
    }

    #[test]
    fn builder_overrides_defaults() {
        let config = AmqpServerConfig::builder()
            .message_max_bytes(1024)
            .read_buffer_bytes(512)
            .channel_max(4)
            .build();
        assert_eq!(config.message_max_bytes, 1024);
        assert_eq!(config.read_buffer_bytes, 512);
        assert_eq!(config.channel_max, 4);
    }

    #[test]
    fn validate_rejects_message_max_below_read_buffer() {
        let config = AmqpServerConfig::builder()
            .read_buffer_bytes(4096)
            .message_max_bytes(1024)
            .build();
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_channel_max() {
        let config = AmqpServerConfig::builder().channel_max(0).build();
        assert!(config.validate().is_err());
    }
}
