//! `AmqpClientConfig` — the declarative half of an AMQP 0-9-1 client
//! (workspace principle 4: one type is the bon builder result, the serde
//! shape, and the conflaguration env surface `AMQP_CLIENT_*`). The live
//! transport (`StreamUpstream`) is a runtime object injected at connect
//! time, not in the config — mirrors `proxima_redis::client::config::RedisClientConfig`.

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

fn default_host() -> String {
    "localhost".to_string()
}

fn default_port() -> u16 {
    5672
}

fn default_user() -> String {
    "guest".to_string()
}

fn default_virtual_host() -> String {
    "/".to_string()
}

/// Connection parameters for proxima's AMQP 0-9-1 client. Maps 1:1 to a
/// TOML `[amqp]` table or `AMQP_CLIENT_*` env vars, and to the bon builder.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Serialize, Deserialize, Settings)]
#[settings(prefix = "AMQP_CLIENT")]
#[builder(derive(Clone, Debug))]
pub struct AmqpClientConfig {
    /// Broker host. Resolved to a socket address when the transport
    /// connects.
    #[setting(default = "localhost")]
    #[serde(default = "default_host")]
    #[builder(default = default_host(), into)]
    pub host: String,

    /// Broker port (AMQP 0-9-1 default 5672).
    #[setting(default = 5672)]
    #[serde(default = "default_port")]
    #[builder(default = default_port())]
    pub port: u16,

    /// SASL PLAIN username.
    #[setting(default = "guest")]
    #[serde(default = "default_user")]
    #[builder(default = default_user(), into)]
    pub username: String,

    /// SASL PLAIN password.
    #[setting(default = "guest", sensitive)]
    #[serde(default = "default_user")]
    #[builder(default = default_user(), into)]
    pub password: String,

    /// The `connection.open` virtual host.
    #[setting(default = "/")]
    #[serde(default = "default_virtual_host")]
    #[builder(default = default_virtual_host(), into)]
    pub virtual_host: String,
}

impl Default for AmqpClientConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl AmqpClientConfig {
    /// Parses an `amqp://[user[:pass]@]host[:port][/vhost]` DSN. A missing
    /// field falls back to its default. This is the ergonomic entry the
    /// fluent `.amqp(dsn)` builder sugar lowers to.
    ///
    /// # Errors
    /// [`AmqpConfigError::Scheme`] when the scheme is not `amqp`,
    /// [`AmqpConfigError::Tls`] for `amqps://` (TLS transport not yet
    /// wired — rejected rather than silently downgraded), or
    /// [`AmqpConfigError::Port`] on a non-numeric port.
    pub fn from_dsn(dsn: &str) -> Result<Self, AmqpConfigError> {
        if dsn.starts_with("amqps://") {
            return Err(AmqpConfigError::Tls);
        }
        let rest = dsn.strip_prefix("amqp://").ok_or(AmqpConfigError::Scheme)?;

        let (credentials, authority) = match rest.rsplit_once('@') {
            Some((credentials, authority)) => (Some(credentials), authority),
            None => (None, rest),
        };
        let (host_port, virtual_host) = match authority.split_once('/') {
            Some((host_port, virtual_host)) => (host_port, Some(virtual_host)),
            None => (authority, None),
        };
        let (host, port) = match host_port.rsplit_once(':') {
            Some((host, port)) => (
                host,
                port.parse::<u16>().map_err(|_| AmqpConfigError::Port)?,
            ),
            None => (host_port, default_port()),
        };
        let (username, password) = match credentials {
            Some(credentials) => match credentials.split_once(':') {
                Some((user, pass)) => (user.to_string(), pass.to_string()),
                None => (credentials.to_string(), default_user()),
            },
            None => (default_user(), default_user()),
        };
        let virtual_host = match virtual_host.filter(|value| !value.is_empty()) {
            Some(value) => format!("/{value}"),
            None => default_virtual_host(),
        };

        Ok(Self {
            host: if host.is_empty() {
                default_host()
            } else {
                host.to_string()
            },
            port,
            username,
            password,
            virtual_host,
        })
    }

    /// `host:port`, the address the transport dials.
    #[must_use]
    pub fn address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AmqpConfigError {
    #[error("dsn must start with amqp://")]
    Scheme,
    #[error("amqps:// (TLS) is not yet supported — use a TLS-terminating transport")]
    Tls,
    #[error("dsn port must be a number")]
    Port,
}

impl Validate for AmqpClientConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.host.is_empty() {
            errors.push(ValidationMessage::new("host", "must be non-empty"));
        }
        if self.port == 0 {
            errors.push(ValidationMessage::new("port", "must be non-zero"));
        }
        if !self.virtual_host.starts_with('/') {
            errors.push(ValidationMessage::new(
                "virtual_host",
                "must start with '/'",
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
    fn default_matches_builder() {
        assert_eq!(
            AmqpClientConfig::default(),
            AmqpClientConfig::builder().build()
        );
        let config = AmqpClientConfig::default();
        assert_eq!((config.host.as_str(), config.port), ("localhost", 5672));
        assert_eq!(config.virtual_host, "/");
        assert_eq!(config.username, "guest");
    }

    #[test]
    fn dsn_full_round_trips_every_field() {
        let config =
            AmqpClientConfig::from_dsn("amqp://alice:s3cr3t@broker.example.com:5673/prod").unwrap();
        assert_eq!(config.username, "alice");
        assert_eq!(config.password, "s3cr3t");
        assert_eq!(config.host, "broker.example.com");
        assert_eq!(config.port, 5673);
        assert_eq!(config.virtual_host, "/prod");
        assert_eq!(config.address(), "broker.example.com:5673");
    }

    #[test]
    fn dsn_minimal_falls_back_to_defaults() {
        let config = AmqpClientConfig::from_dsn("amqp://localhost").unwrap();
        assert_eq!((config.port, config.virtual_host.as_str()), (5672, "/"));
        assert_eq!(config.username, "guest");
    }

    #[test]
    fn dsn_rejects_foreign_scheme() {
        assert_eq!(
            AmqpClientConfig::from_dsn("http://host"),
            Err(AmqpConfigError::Scheme)
        );
    }

    #[test]
    fn dsn_rejects_tls_scheme_rather_than_downgrading() {
        assert_eq!(
            AmqpClientConfig::from_dsn("amqps://host:5671"),
            Err(AmqpConfigError::Tls)
        );
    }

    #[test]
    fn builder_overrides_then_serde_round_trips() {
        let config = AmqpClientConfig::builder()
            .host("h")
            .port(5673)
            .virtual_host("/staging")
            .build();
        let json = serde_json::to_string(&config).expect("ser");
        let back: AmqpClientConfig = serde_json::from_str(&json).expect("de");
        assert_eq!(config, back);
    }
}
