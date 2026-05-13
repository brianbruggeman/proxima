//! `RedisClientConfig` — the declarative half of a Redis/Valkey client
//! (workspace principle 4: one type is the bon builder result, the serde shape,
//! and the conflaguration env surface `REDIS_CLIENT_*`). The live transport
//! (`StreamUpstream`) is a runtime object injected at connect time, not in the
//! config — the same config-vs-runtime split pgwire's `PgClientConfig` uses.

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

fn default_host() -> String {
    "localhost".to_string()
}

fn default_port() -> u16 {
    6379
}

fn default_true() -> bool {
    true
}

/// The RESP protocol revision a connection negotiates. RESP3 (Redis 6+ /
/// Valkey) is reached with `HELLO 3` at startup; RESP2 sends no `HELLO` and
/// authenticates with `AUTH` when a password is set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RespProtocol {
    Resp2,
    Resp3,
}

/// Connection parameters for proxima's Redis/Valkey client. Maps 1:1 to a TOML
/// `[redis]` table or `REDIS_CLIENT_*` env vars, and to the bon builder.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Serialize, Deserialize, Settings)]
#[settings(prefix = "REDIS_CLIENT")]
#[builder(derive(Clone, Debug))]
pub struct RedisClientConfig {
    /// Server host. Resolved to a socket address when the transport connects.
    #[setting(default = "localhost")]
    #[serde(default = "default_host")]
    #[builder(default = default_host(), into)]
    pub host: String,

    /// Server port (Redis/Valkey default 6379).
    #[setting(default = 6379)]
    #[serde(default = "default_port")]
    #[builder(default = default_port())]
    pub port: u16,

    /// ACL username (Redis 6+). Empty selects the implicit `default` user, in
    /// which case a non-empty [`password`](Self::password) authenticates with
    /// single-argument `AUTH`.
    #[setting(default = "")]
    #[serde(default)]
    #[builder(default, into)]
    pub username: String,

    /// Password for `AUTH` / `HELLO ... AUTH`; unused when empty. Held only as
    /// long as the config; the live session keeps its working copy in a
    /// `Zeroizing` buffer wiped on drop.
    #[setting(default = "", sensitive)]
    #[serde(default)]
    #[builder(default, into)]
    pub password: String,

    /// Logical database index to `SELECT` after auth (0 unless overridden).
    #[setting(default = 0)]
    #[serde(default)]
    #[builder(default)]
    pub db: u32,

    /// Negotiate RESP3 with `HELLO 3` at startup. `false` stays on RESP2 (no
    /// `HELLO`); auth then rides a bare `AUTH`.
    #[setting(default = true)]
    #[serde(default = "default_true")]
    #[builder(default = true)]
    pub resp3: bool,
}

impl Default for RedisClientConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl RedisClientConfig {
    /// Parses a `redis://[username[:password]@]host[:port][/db]` DSN. A missing
    /// field falls back to its default. This is the ergonomic entry the fluent
    /// `.redis(dsn)` / `.valkey(dsn)` sugar lowers to.
    ///
    /// # Errors
    /// [`RedisConfigError::Scheme`] when the scheme is not `redis`,
    /// [`RedisConfigError::Tls`] for `rediss://` (TLS transport not yet wired —
    /// rejected rather than silently downgraded), or
    /// [`RedisConfigError::Port`] / [`RedisConfigError::Db`] on a non-numeric
    /// port / database index.
    pub fn from_dsn(dsn: &str) -> Result<Self, RedisConfigError> {
        if dsn.starts_with("rediss://") {
            return Err(RedisConfigError::Tls);
        }
        let rest = dsn
            .strip_prefix("redis://")
            .ok_or(RedisConfigError::Scheme)?;

        let (credentials, authority) = match rest.rsplit_once('@') {
            Some((credentials, authority)) => (Some(credentials), authority),
            None => (None, rest),
        };
        let (host_port, database) = match authority.split_once('/') {
            Some((host_port, database)) => (host_port, Some(database)),
            None => (authority, None),
        };
        let (host, port) = match host_port.rsplit_once(':') {
            Some((host, port)) => (
                host,
                port.parse::<u16>().map_err(|_| RedisConfigError::Port)?,
            ),
            None => (host_port, default_port()),
        };
        let (username, password) = match credentials {
            Some(credentials) => match credentials.split_once(':') {
                Some((user, pass)) => (user.to_string(), pass.to_string()),
                None => (credentials.to_string(), String::new()),
            },
            None => (String::new(), String::new()),
        };
        let db = match database.filter(|value| !value.is_empty()) {
            Some(value) => value.parse::<u32>().map_err(|_| RedisConfigError::Db)?,
            None => 0,
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
            db,
            resp3: true,
        })
    }

    /// The negotiated protocol revision.
    #[must_use]
    pub fn protocol(&self) -> RespProtocol {
        if self.resp3 {
            RespProtocol::Resp3
        } else {
            RespProtocol::Resp2
        }
    }

    /// `host:port`, the address the transport dials.
    #[must_use]
    pub fn address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RedisConfigError {
    #[error("dsn must start with redis://")]
    Scheme,
    #[error("rediss:// (TLS) is not yet supported — use a TLS-terminating transport")]
    Tls,
    #[error("dsn port must be a number")]
    Port,
    #[error("dsn database index must be a number")]
    Db,
}

impl Validate for RedisClientConfig {
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
            RedisClientConfig::default(),
            RedisClientConfig::builder().build()
        );
        let config = RedisClientConfig::default();
        assert_eq!((config.host.as_str(), config.port), ("localhost", 6379));
        assert_eq!((config.db, config.resp3), (0, true));
        assert_eq!(config.protocol(), RespProtocol::Resp3);
    }

    #[test]
    fn dsn_full_round_trips_every_field() {
        let config =
            RedisClientConfig::from_dsn("redis://alice:s3cr3t@cache.example.com:6380/2").unwrap();
        assert_eq!(config.username, "alice");
        assert_eq!(config.password, "s3cr3t");
        assert_eq!(config.host, "cache.example.com");
        assert_eq!(config.port, 6380);
        assert_eq!(config.db, 2);
        assert_eq!(config.address(), "cache.example.com:6380");
    }

    #[test]
    fn dsn_password_only_uses_default_user() {
        let config = RedisClientConfig::from_dsn("redis://:hunter2@localhost").unwrap();
        assert_eq!(config.username, "");
        assert_eq!(config.password, "hunter2");
        assert_eq!(config.port, 6379);
    }

    #[test]
    fn dsn_minimal_falls_back_to_defaults() {
        let config = RedisClientConfig::from_dsn("redis://localhost").unwrap();
        assert_eq!(
            (config.port, config.db, config.username.as_str()),
            (6379, 0, "")
        );
    }

    #[test]
    fn dsn_rejects_foreign_scheme() {
        assert_eq!(
            RedisClientConfig::from_dsn("http://host"),
            Err(RedisConfigError::Scheme)
        );
    }

    #[test]
    fn dsn_rejects_tls_scheme_rather_than_downgrading() {
        assert_eq!(
            RedisClientConfig::from_dsn("rediss://host:6380"),
            Err(RedisConfigError::Tls)
        );
    }

    #[test]
    fn builder_overrides_then_serde_round_trips() {
        let config = RedisClientConfig::builder()
            .host("h")
            .port(6380)
            .db(3)
            .resp3(false)
            .build();
        let json = serde_json::to_string(&config).expect("ser");
        let back: RedisClientConfig = serde_json::from_str(&json).expect("de");
        assert_eq!(config, back);
        assert_eq!(back.protocol(), RespProtocol::Resp2);
    }
}
