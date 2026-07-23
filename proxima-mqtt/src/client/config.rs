//! `MqttClientConfig` — the declarative half of an MQTT client (workspace
//! principle 4: one type is the bon builder result, the serde shape, and
//! the conflaguration env surface `MQTT_CLIENT_*`). The live transport
//! (`StreamUpstream`) is a runtime object injected at connect time, not in
//! the config — the same split `proxima_redis::client::config::RedisClientConfig`
//! uses.

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

fn default_host() -> String {
    "localhost".to_string()
}

fn default_port() -> u16 {
    1883
}

fn default_true() -> bool {
    true
}

fn default_keep_alive() -> u16 {
    60
}

/// Connection parameters for proxima's MQTT client. Maps 1:1 to a TOML
/// `[mqtt]` table or `MQTT_CLIENT_*` env vars, and to the bon builder.
///
/// Config is first-class in two equivalent forms — the fluent builder and a
/// TOML file loaded through `conflaguration` — and they produce the exact
/// same value:
///
/// ```
/// use std::io::Write;
///
/// use proxima_mqtt::MqttClientConfig;
///
/// let via_builder = MqttClientConfig::builder()
///     .client_id("sensor-01")
///     .keep_alive(30)
///     .build();
///
/// let mut file = tempfile::Builder::new().suffix(".toml").tempfile().expect("tempfile");
/// write!(file, "client_id = \"sensor-01\"\nkeep_alive = 30\n").expect("write toml");
///
/// let via_toml: MqttClientConfig = conflaguration::builder()
///     .file(file.path())
///     .validate()
///     .build()
///     .expect("load from toml");
///
/// assert_eq!(via_builder, via_toml);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Builder, Serialize, Deserialize, Settings)]
#[settings(prefix = "MQTT_CLIENT")]
#[builder(derive(Clone, Debug))]
pub struct MqttClientConfig {
    /// Broker host. Resolved to a socket address when the transport
    /// connects.
    #[setting(default = "localhost")]
    #[serde(default = "default_host")]
    #[builder(default = default_host(), into)]
    pub host: String,

    /// Broker port (MQTT default 1883).
    #[setting(default = 1883)]
    #[serde(default = "default_port")]
    #[builder(default = default_port())]
    pub port: u16,

    /// `CONNECT`'s client identifier. Empty is only legal alongside
    /// [`clean_session`](Self::clean_session) `= true` — [MQTT-3.1.3-8].
    #[setting(default = "")]
    #[serde(default)]
    #[builder(default, into)]
    pub client_id: String,

    /// `CONNECT`'s Clean Session flag.
    #[setting(default = true)]
    #[serde(default = "default_true")]
    #[builder(default = true)]
    pub clean_session: bool,

    /// `CONNECT`'s keep-alive interval, seconds. `0` disables the
    /// keep-alive timeout.
    #[setting(default = 60)]
    #[serde(default = "default_keep_alive")]
    #[builder(default = default_keep_alive())]
    pub keep_alive: u16,

    /// `CONNECT` username; unused when empty.
    #[setting(default = "")]
    #[serde(default)]
    #[builder(default, into)]
    pub username: String,

    /// `CONNECT` password; unused when empty. Held only as long as the
    /// config; the live session keeps its working copy in a `Zeroizing`
    /// buffer wiped on drop.
    #[setting(default = "", sensitive)]
    #[serde(default)]
    #[builder(default, into)]
    pub password: String,
}

impl Default for MqttClientConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl MqttClientConfig {
    /// Parses an `mqtt://[username[:password]@]host[:port]` DSN. A missing
    /// field falls back to its default. This is the ergonomic entry the
    /// fluent `.mqtt(dsn)` sugar lowers to.
    ///
    /// # Errors
    /// [`MqttConfigError::Scheme`] when the scheme is not `mqtt`,
    /// [`MqttConfigError::Tls`] for `mqtts://` (TLS transport not yet wired
    /// — rejected rather than silently downgraded), or
    /// [`MqttConfigError::Port`] on a non-numeric port.
    pub fn from_dsn(dsn: &str) -> Result<Self, MqttConfigError> {
        if dsn.starts_with("mqtts://") {
            return Err(MqttConfigError::Tls);
        }
        let rest = dsn.strip_prefix("mqtt://").ok_or(MqttConfigError::Scheme)?;

        let (credentials, authority) = match rest.rsplit_once('@') {
            Some((credentials, authority)) => (Some(credentials), authority),
            None => (None, rest),
        };
        let (host, port) = match authority.rsplit_once(':') {
            Some((host, port)) => (
                host,
                port.parse::<u16>().map_err(|_| MqttConfigError::Port)?,
            ),
            None => (authority, default_port()),
        };
        let (username, password) = match credentials {
            Some(credentials) => match credentials.split_once(':') {
                Some((user, pass)) => (user.to_string(), pass.to_string()),
                None => (credentials.to_string(), String::new()),
            },
            None => (String::new(), String::new()),
        };

        Ok(Self {
            host: if host.is_empty() { default_host() } else { host.to_string() },
            port,
            client_id: String::new(),
            clean_session: true,
            keep_alive: default_keep_alive(),
            username,
            password,
        })
    }

    /// `host:port`, the address the transport dials.
    #[must_use]
    pub fn address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MqttConfigError {
    #[error("dsn must start with mqtt://")]
    Scheme,
    #[error("mqtts:// (TLS) is not yet supported — use a TLS-terminating transport")]
    Tls,
    #[error("dsn port must be a number")]
    Port,
}

impl Validate for MqttClientConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.host.is_empty() {
            errors.push(ValidationMessage::new("host", "must be non-empty"));
        }
        if self.port == 0 {
            errors.push(ValidationMessage::new("port", "must be non-zero"));
        }
        if self.client_id.is_empty() && !self.clean_session {
            errors.push(ValidationMessage::new(
                "client_id",
                "must be non-empty unless clean_session is true [MQTT-3.1.3-8]",
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
        assert_eq!(MqttClientConfig::default(), MqttClientConfig::builder().build());
        let config = MqttClientConfig::default();
        assert_eq!((config.host.as_str(), config.port), ("localhost", 1883));
        assert!(config.clean_session);
        assert_eq!(config.keep_alive, 60);
    }

    #[test]
    fn dsn_full_round_trips_every_field() {
        let config = MqttClientConfig::from_dsn("mqtt://alice:s3cr3t@broker.example.com:8883").unwrap();
        assert_eq!(config.username, "alice");
        assert_eq!(config.password, "s3cr3t");
        assert_eq!(config.host, "broker.example.com");
        assert_eq!(config.port, 8883);
        assert_eq!(config.address(), "broker.example.com:8883");
    }

    #[test]
    fn dsn_password_only_uses_default_user() {
        let config = MqttClientConfig::from_dsn("mqtt://:hunter2@localhost").unwrap();
        assert_eq!(config.username, "");
        assert_eq!(config.password, "hunter2");
        assert_eq!(config.port, 1883);
    }

    #[test]
    fn dsn_minimal_falls_back_to_defaults() {
        let config = MqttClientConfig::from_dsn("mqtt://localhost").unwrap();
        assert_eq!((config.port, config.username.as_str()), (1883, ""));
    }

    #[test]
    fn dsn_rejects_foreign_scheme() {
        assert_eq!(MqttClientConfig::from_dsn("http://host"), Err(MqttConfigError::Scheme));
    }

    #[test]
    fn dsn_rejects_tls_scheme_rather_than_downgrading() {
        assert_eq!(MqttClientConfig::from_dsn("mqtts://host:8883"), Err(MqttConfigError::Tls));
    }

    #[test]
    fn builder_overrides_then_serde_round_trips() {
        let config = MqttClientConfig::builder()
            .host("h")
            .port(8883)
            .client_id("c1")
            .clean_session(false)
            .build();
        let json = serde_json::to_string(&config).expect("ser");
        let back: MqttClientConfig = serde_json::from_str(&json).expect("de");
        assert_eq!(config, back);
    }

    #[test]
    fn validate_rejects_empty_client_id_without_clean_session() {
        let config = MqttClientConfig::builder().clean_session(false).build();
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_accepts_empty_client_id_with_clean_session() {
        let config = MqttClientConfig::builder().clean_session(true).build();
        assert!(config.validate().is_ok());
    }
}
