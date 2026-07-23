//! `MemcachedClientConfig` — the declarative half of a memcached client
//! (workspace principle 4: one type is the bon builder result, the serde
//! shape, and the conflaguration env surface `MEMCACHED_CLIENT_*`). The
//! live transport (`StreamUpstream`) is a runtime object injected at
//! connect time, not in the config — the same config-vs-runtime split
//! `proxima_redis::client::RedisClientConfig` uses.
//!
//! Unlike RESP, the base memcached text protocol has no connection-level
//! handshake (no `HELLO`/`AUTH`/`SELECT` equivalent — SASL auth exists in
//! the protocol but is out of scope here, matching the "simplest of the
//! set" charter), so this config carries only the dial target.

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

fn default_host() -> String {
    "localhost".to_string()
}

fn default_port() -> u16 {
    11211
}

/// Connection parameters for proxima's memcached client. Maps 1:1 to a
/// TOML `[memcached_client]` table or `MEMCACHED_CLIENT_*` env vars, and to
/// the bon builder.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Serialize, Deserialize, Settings)]
#[settings(prefix = "MEMCACHED_CLIENT")]
#[builder(derive(Clone, Debug))]
pub struct MemcachedClientConfig {
    /// Server host. Resolved to a socket address when the transport connects.
    #[setting(default = "localhost")]
    #[serde(default = "default_host")]
    #[builder(default = default_host(), into)]
    pub host: String,

    /// Server port (memcached default 11211).
    #[setting(default = 11211)]
    #[serde(default = "default_port")]
    #[builder(default = default_port())]
    pub port: u16,
}

impl Default for MemcachedClientConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl MemcachedClientConfig {
    /// Parses a `memcached://host[:port]` DSN. A missing port falls back
    /// to the default. This is the ergonomic entry the fluent
    /// `.memcached(dsn)` sugar lowers to.
    ///
    /// # Errors
    /// [`MemcachedConfigError::Scheme`] when the scheme is not `memcached`,
    /// or [`MemcachedConfigError::Port`] on a non-numeric port.
    pub fn from_dsn(dsn: &str) -> Result<Self, MemcachedConfigError> {
        let rest = dsn
            .strip_prefix("memcached://")
            .ok_or(MemcachedConfigError::Scheme)?;
        let (host, port) = match rest.rsplit_once(':') {
            Some((host, port)) => (
                host,
                port.parse::<u16>().map_err(|_| MemcachedConfigError::Port)?,
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
        })
    }

    /// `host:port`, the address the transport dials.
    #[must_use]
    pub fn address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MemcachedConfigError {
    #[error("dsn must start with memcached://")]
    Scheme,
    #[error("dsn port must be a number")]
    Port,
}

impl Validate for MemcachedClientConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.host.is_empty() {
            errors.push(ValidationMessage::new("host", "must be non-empty"));
        }
        if self.port == 0 {
            errors.push(ValidationMessage::new("port", "must be non-zero"));
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
            MemcachedClientConfig::default(),
            MemcachedClientConfig::builder().build()
        );
        let config = MemcachedClientConfig::default();
        assert_eq!((config.host.as_str(), config.port), ("localhost", 11211));
    }

    #[test]
    fn dsn_full_round_trips_host_and_port() {
        let config = MemcachedClientConfig::from_dsn("memcached://cache.example.com:11212")
            .expect("dsn parse");
        assert_eq!(config.host, "cache.example.com");
        assert_eq!(config.port, 11212);
        assert_eq!(config.address(), "cache.example.com:11212");
    }

    #[test]
    fn dsn_minimal_falls_back_to_default_port() {
        let config = MemcachedClientConfig::from_dsn("memcached://localhost").expect("dsn parse");
        assert_eq!(config.port, 11211);
    }

    #[test]
    fn dsn_rejects_foreign_scheme() {
        assert_eq!(
            MemcachedClientConfig::from_dsn("redis://host"),
            Err(MemcachedConfigError::Scheme)
        );
    }

    #[test]
    fn dsn_rejects_non_numeric_port() {
        assert_eq!(
            MemcachedClientConfig::from_dsn("memcached://host:abc"),
            Err(MemcachedConfigError::Port)
        );
    }

    #[test]
    fn builder_overrides_then_serde_round_trips() {
        let config = MemcachedClientConfig::builder()
            .host("h")
            .port(11300)
            .build();
        let json = serde_json::to_string(&config).expect("ser");
        let back: MemcachedClientConfig = serde_json::from_str(&json).expect("de");
        assert_eq!(config, back);
    }
}
