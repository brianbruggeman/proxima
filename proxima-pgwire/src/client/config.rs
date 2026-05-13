//! `PgClientConfig` — the declarative half of a pgwire client (principle 4:
//! one type is the bon builder result, the serde shape, and the conflaguration
//! env surface `PGWIRE_CLIENT_*`). The live transport (`StreamUpstream`) is a
//! runtime object injected at connect time, not in the config — the same
//! config-vs-runtime split `H1ClientConfig` / telemetry's `Recorder` use.

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

fn default_host() -> String {
    "localhost".to_string()
}

fn default_port() -> u16 {
    5432
}

fn default_user() -> String {
    "postgres".to_string()
}

fn default_database() -> String {
    "postgres".to_string()
}

/// Connection parameters for proxima's PostgreSQL client. Maps 1:1 to a TOML
/// `[pgwire]` table or `PGWIRE_CLIENT_*` env vars, and to the bon builder.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Serialize, Deserialize, Settings)]
#[settings(prefix = "PGWIRE_CLIENT")]
#[builder(derive(Clone, Debug))]
pub struct PgClientConfig {
    /// Server host. Resolved to a socket address when the transport connects.
    #[setting(default = "localhost")]
    #[serde(default = "default_host")]
    #[builder(default = default_host(), into)]
    pub host: String,

    /// Server port (PostgreSQL default 5432).
    #[setting(default = 5432)]
    #[serde(default = "default_port")]
    #[builder(default = default_port())]
    pub port: u16,

    /// Role to authenticate as.
    #[setting(default = "postgres")]
    #[serde(default = "default_user")]
    #[builder(default = default_user(), into)]
    pub user: String,

    /// Password for cleartext / SCRAM auth; unused for trust. Held only as
    /// long as the config; the live `ScramClient` zeroizes its own copy.
    #[setting(default = "", sensitive)]
    #[serde(default)]
    #[builder(default, into)]
    pub password: String,

    /// Database to connect to.
    #[setting(default = "postgres")]
    #[serde(default = "default_database")]
    #[builder(default = default_database(), into)]
    pub database: String,
}

impl Default for PgClientConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl PgClientConfig {
    /// Parses a `postgres://[user[:password]@]host[:port][/database]` DSN. A
    /// missing field falls back to its default. This is the ergonomic entry the
    /// fluent `.pgwire(dsn)` sugar lowers to.
    ///
    /// # Errors
    /// [`ConfigError`] when the scheme is not `postgres`/`postgresql` or the
    /// port is non-numeric.
    pub fn from_dsn(dsn: &str) -> Result<Self, ConfigError> {
        let rest = dsn
            .strip_prefix("postgres://")
            .or_else(|| dsn.strip_prefix("postgresql://"))
            .ok_or(ConfigError::Scheme)?;

        // split optional `user:pass@` from `host:port/db`
        let (credentials, authority) = match rest.rsplit_once('@') {
            Some((credentials, authority)) => (Some(credentials), authority),
            None => (None, rest),
        };
        let (host_port, database) = match authority.split_once('/') {
            Some((host_port, database)) => (host_port, Some(database)),
            None => (authority, None),
        };
        let (host, port) = match host_port.rsplit_once(':') {
            Some((host, port)) => {
                let port = port.parse::<u16>().map_err(|_| ConfigError::Port)?;
                (host, port)
            }
            None => (host_port, default_port()),
        };
        let (user, password) = match credentials {
            Some(credentials) => match credentials.split_once(':') {
                Some((user, password)) => (user.to_string(), password.to_string()),
                None => (credentials.to_string(), String::new()),
            },
            None => (default_user(), String::new()),
        };

        Ok(Self {
            host: if host.is_empty() {
                default_host()
            } else {
                host.to_string()
            },
            port,
            user,
            password,
            database: database
                .filter(|value| !value.is_empty())
                .map_or_else(default_database, str::to_string),
        })
    }

    /// `host:port`, the address the transport dials.
    #[must_use]
    pub fn address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ConfigError {
    #[error("dsn must start with postgres:// or postgresql://")]
    Scheme,
    #[error("dsn port must be a number")]
    Port,
}

impl Validate for PgClientConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.host.is_empty() {
            errors.push(ValidationMessage::new("host", "must be non-empty"));
        }
        if self.port == 0 {
            errors.push(ValidationMessage::new("port", "must be non-zero"));
        }
        if self.user.is_empty() {
            errors.push(ValidationMessage::new("user", "must be non-empty"));
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
        assert_eq!(PgClientConfig::default(), PgClientConfig::builder().build());
        let config = PgClientConfig::default();
        assert_eq!((config.host.as_str(), config.port), ("localhost", 5432));
        assert_eq!(
            (config.user.as_str(), config.database.as_str()),
            ("postgres", "postgres")
        );
    }

    #[test]
    fn dsn_full_round_trips_every_field() {
        let config = PgClientConfig::from_dsn("postgres://alice:s3cr3t@db.example.com:6543/appdb")
            .expect("dsn");
        assert_eq!(config.user, "alice");
        assert_eq!(config.password, "s3cr3t");
        assert_eq!(config.host, "db.example.com");
        assert_eq!(config.port, 6543);
        assert_eq!(config.database, "appdb");
        assert_eq!(config.address(), "db.example.com:6543");
    }

    #[test]
    fn dsn_minimal_falls_back_to_defaults() {
        let config = PgClientConfig::from_dsn("postgres://localhost").expect("dsn");
        assert_eq!(config.port, 5432);
        assert_eq!(config.user, "postgres");
        assert_eq!(config.database, "postgres");
        assert_eq!(config.password, "");
    }

    #[test]
    fn dsn_user_without_password() {
        let config = PgClientConfig::from_dsn("postgres://alice@host/db").expect("dsn");
        assert_eq!(config.user, "alice");
        assert_eq!(config.password, "");
        assert_eq!(config.database, "db");
    }

    #[test]
    fn dsn_rejects_foreign_scheme() {
        assert_eq!(
            PgClientConfig::from_dsn("mysql://host/db"),
            Err(ConfigError::Scheme)
        );
    }

    #[cfg(feature = "listen")]
    #[test]
    fn builder_overrides_then_serde_round_trips() {
        let config = PgClientConfig::builder()
            .host("h")
            .port(1234)
            .user("u")
            .build();
        let json = serde_json::to_string(&config).expect("ser");
        let back: PgClientConfig = serde_json::from_str(&json).expect("de");
        assert_eq!(config, back);
    }
}
