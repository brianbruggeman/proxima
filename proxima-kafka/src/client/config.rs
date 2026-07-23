//! `KafkaClientConfig` — the declarative half of a Kafka client (workspace
//! principle 4: one type is the bon builder result, the serde shape, and
//! the conflaguration env surface `KAFKA_CLIENT_*`). The live transport
//! (`StreamUpstream`) is a runtime object injected at connect time, not in
//! the config — the same config-vs-runtime split
//! `proxima_redis::client::config::RedisClientConfig` uses.
//!
//! Scoped to a single bootstrap broker: this facade's own `Metadata`
//! response always names the one broker it is (`crate::broker::KafkaBroker`
//! never returns a second broker to redirect to), so a real multi-broker
//! bootstrap-then-redirect dance has nothing to land on yet.

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

fn default_host() -> String {
    "localhost".to_string()
}

fn default_port() -> u16 {
    9092
}

fn default_client_id() -> String {
    "proxima-kafka".to_string()
}

/// Connection parameters for proxima's Kafka client. Maps 1:1 to a TOML
/// `[kafka]` table or `KAFKA_CLIENT_*` env vars, and to the bon builder.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Serialize, Deserialize, Settings)]
#[settings(prefix = "KAFKA_CLIENT")]
#[builder(derive(Clone, Debug))]
pub struct KafkaClientConfig {
    /// Broker host. Resolved to a socket address when the transport
    /// connects.
    #[setting(default = "localhost")]
    #[serde(default = "default_host")]
    #[builder(default = default_host(), into)]
    pub host: String,

    /// Broker port (Kafka's conventional default 9092).
    #[setting(default = 9092)]
    #[serde(default = "default_port")]
    #[builder(default = default_port())]
    pub port: u16,

    /// The `client_id` this client sends in every request header —
    /// broker-side logging/quota identity, not authentication.
    #[setting(default = "proxima-kafka")]
    #[serde(default = "default_client_id")]
    #[builder(default = default_client_id(), into)]
    pub client_id: String,
}

impl Default for KafkaClientConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl KafkaClientConfig {
    /// Parses a `kafka://host[:port]` DSN. A missing field falls back to
    /// its default. This is the ergonomic entry the fluent `.kafka(dsn)`
    /// sugar lowers to.
    ///
    /// # Errors
    /// [`KafkaConfigError::Scheme`] when the scheme is not `kafka`,
    /// [`KafkaConfigError::Port`] on a non-numeric port.
    pub fn from_dsn(dsn: &str) -> Result<Self, KafkaConfigError> {
        let rest = dsn
            .strip_prefix("kafka://")
            .ok_or(KafkaConfigError::Scheme)?;
        let (host, port) = match rest.rsplit_once(':') {
            Some((host, port)) => (
                host,
                port.parse::<u16>()
                    .map_err(|_error| KafkaConfigError::Port)?,
            ),
            None => (rest, default_port()),
        };
        Ok(Self {
            host: if host.is_empty() {
                default_host()
            } else {
                host.to_string()
            },
            port,
            client_id: default_client_id(),
        })
    }

    /// `host:port`, the address the transport dials.
    #[must_use]
    pub fn address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum KafkaConfigError {
    #[error("dsn must start with kafka://")]
    Scheme,
    #[error("dsn port must be a number")]
    Port,
}

impl Validate for KafkaClientConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.host.is_empty() {
            errors.push(ValidationMessage::new("host", "must be non-empty"));
        }
        if self.port == 0 {
            errors.push(ValidationMessage::new("port", "must be non-zero"));
        }
        if self.client_id.is_empty() {
            errors.push(ValidationMessage::new("client_id", "must be non-empty"));
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
    fn default_matches_builder() {
        assert_eq!(
            KafkaClientConfig::default(),
            KafkaClientConfig::builder().build()
        );
        let config = KafkaClientConfig::default();
        assert_eq!((config.host.as_str(), config.port), ("localhost", 9092));
        assert_eq!(config.client_id, "proxima-kafka");
    }

    #[test]
    fn dsn_full_round_trips_host_and_port() {
        let config = KafkaClientConfig::from_dsn("kafka://broker.example.com:9093").unwrap();
        assert_eq!(config.host, "broker.example.com");
        assert_eq!(config.port, 9093);
    }

    #[test]
    fn dsn_minimal_falls_back_to_the_default_port() {
        let config = KafkaClientConfig::from_dsn("kafka://localhost").unwrap();
        assert_eq!(config.port, 9092);
    }

    #[test]
    fn dsn_rejects_foreign_scheme() {
        assert_eq!(
            KafkaClientConfig::from_dsn("redis://host"),
            Err(KafkaConfigError::Scheme)
        );
    }

    #[test]
    fn dsn_rejects_non_numeric_port() {
        assert_eq!(
            KafkaClientConfig::from_dsn("kafka://host:notaport"),
            Err(KafkaConfigError::Port)
        );
    }

    #[test]
    fn builder_overrides_then_serde_round_trips() {
        let config = KafkaClientConfig::builder()
            .host("h")
            .port(9093)
            .client_id("app-1")
            .build();
        let json = serde_json::to_string(&config).expect("ser");
        let back: KafkaClientConfig = serde_json::from_str(&json).expect("de");
        assert_eq!(config, back);
    }
}
