//! `PgServerConfig` — the facade's config-mirror surface (workspace
//! principle 4): one type is the bon builder result, the serde shape,
//! and the conflaguration env surface (`PGWIRE_*`). The listen layer
//! also accepts it inside a listener spec under the `pgwire` key.

use std::sync::Arc;

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};
use zeroize::Zeroizing;

use crate::auth::{PgAuth, StaticCredentials};
use crate::error::ServeError;

/// Authentication section of the config mirror. Custom verifiers are an
/// API-only surface (`PgWireListenProtocol::with_auth`); config selects
/// between trust and a static cleartext identity.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", tag = "mode")]
pub enum AuthConfig {
    #[default]
    Trust,
    Cleartext {
        username: String,
        password: String,
    },
    Md5 {
        username: String,
        password: String,
    },
    Scram {
        username: String,
        password: String,
    },
}

/// ParameterStatus pairs reported after AuthenticationOk. A newtype so
/// every construction surface (env loader, serde, builder) shares the
/// same canonical default set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ReportedParameters(pub Vec<(String, String)>);

impl Default for ReportedParameters {
    fn default() -> Self {
        Self(
            [
                ("server_version", "16.0 (proxima-pgwire 0.1)"),
                ("server_encoding", "UTF8"),
                ("client_encoding", "UTF8"),
                ("DateStyle", "ISO, MDY"),
                ("integer_datetimes", "on"),
                ("standard_conforming_strings", "on"),
                ("TimeZone", "UTC"),
            ]
            .into_iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect(),
        )
    }
}

impl From<Vec<(String, String)>> for ReportedParameters {
    fn from(pairs: Vec<(String, String)>) -> Self {
        Self(pairs)
    }
}

impl ReportedParameters {
    pub fn iter(&self) -> impl Iterator<Item = &(String, String)> {
        self.0.iter()
    }
}

fn default_read_buffer() -> usize {
    8 * 1024
}

fn default_high_water() -> usize {
    64 * 1024
}

fn default_max_message() -> usize {
    16 * 1024 * 1024
}

fn default_max_statements() -> usize {
    256
}

fn default_max_portals() -> usize {
    64
}

/// PostgreSQL wire server configuration.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Serialize, Deserialize, Settings)]
#[settings(prefix = "PGWIRE")]
#[builder(derive(Clone, Debug))]
pub struct PgServerConfig {
    /// initial read-buffer size; grows up to `max_message_bytes`
    #[setting(default = 8192)]
    #[serde(default = "default_read_buffer")]
    #[builder(default = default_read_buffer())]
    pub read_buffer_bytes: usize,

    /// write buffer flush threshold while streaming rows
    #[setting(default = 65536)]
    #[serde(default = "default_high_water")]
    #[builder(default = default_high_water())]
    pub write_high_water_bytes: usize,

    /// hard cap on one inbound message (the protocol allows up to 1 GiB;
    /// raising this raises the per-connection memory ceiling)
    #[setting(default = 16777216)]
    #[serde(default = "default_max_message")]
    #[builder(default = default_max_message())]
    pub max_message_bytes: usize,

    /// prepared-statement slots per connection
    #[setting(default = 256)]
    #[serde(default = "default_max_statements")]
    #[builder(default = default_max_statements())]
    pub max_statements: usize,

    /// portal slots per connection
    #[setting(default = 64)]
    #[serde(default = "default_max_portals")]
    #[builder(default = default_max_portals())]
    pub max_portals: usize,

    /// ParameterStatus pairs reported after AuthenticationOk; clients
    /// key behavior off `client_encoding`, `standard_conforming_strings`,
    /// `DateStyle`, and `integer_datetimes`
    #[setting(skip)]
    #[serde(default)]
    #[builder(default, into)]
    pub parameters: ReportedParameters,

    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub auth: AuthConfig,
}

impl Default for PgServerConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl Validate for PgServerConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.read_buffer_bytes < 1024 {
            errors.push(ValidationMessage::new(
                "read_buffer_bytes",
                "must be at least 1024 (a startup packet)",
            ));
        }
        if self.max_message_bytes < self.read_buffer_bytes {
            errors.push(ValidationMessage::new(
                "max_message_bytes",
                "must be at least read_buffer_bytes",
            ));
        }
        if self.write_high_water_bytes < 4096 {
            errors.push(ValidationMessage::new(
                "write_high_water_bytes",
                "must be at least 4096 (one wide row)",
            ));
        }
        if self.max_statements == 0 || self.max_portals == 0 {
            errors.push(ValidationMessage::new(
                "max_statements",
                "max_statements and max_portals must be non-zero",
            ));
        }
        for (name, value) in &self.parameters.0 {
            if name.as_bytes().contains(&0) || value.as_bytes().contains(&0) {
                errors.push(ValidationMessage::new(
                    "parameters",
                    "names and values must not embed nul bytes",
                ));
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl PgServerConfig {
    /// Lowers the auth section into the runtime policy.
    ///
    /// # Errors
    /// [`ServeError::Config`] when the cleartext identity is empty.
    pub fn build_auth(&self) -> Result<PgAuth, ServeError> {
        match &self.auth {
            AuthConfig::Trust => Ok(PgAuth::Trust),
            AuthConfig::Cleartext { username, password } => Ok(PgAuth::Cleartext(Arc::new(
                static_identity("cleartext", username, password)?,
            ))),
            AuthConfig::Md5 { username, password } => {
                let _credentials = static_identity("md5", username, password)?;
                #[cfg(feature = "md5-auth")]
                {
                    Ok(PgAuth::Md5(Arc::new(_credentials)))
                }
                #[cfg(not(feature = "md5-auth"))]
                {
                    Err(ServeError::Config(
                        "md5 auth requires the md5-auth feature".into(),
                    ))
                }
            }
            AuthConfig::Scram { username, password } => {
                let _credentials = static_identity("scram", username, password)?;
                #[cfg(feature = "scram")]
                {
                    Ok(PgAuth::Scram(Arc::new(_credentials)))
                }
                #[cfg(not(feature = "scram"))]
                {
                    Err(ServeError::Config(
                        "scram auth requires the scram feature".into(),
                    ))
                }
            }
        }
    }
}

fn static_identity(
    method: &str,
    username: &str,
    password: &str,
) -> Result<StaticCredentials, ServeError> {
    if username.is_empty() {
        return Err(ServeError::Config(format!(
            "{method} auth requires a non-empty username"
        )));
    }
    Ok(StaticCredentials {
        username: username.to_owned(),
        password: Zeroizing::new(password.to_owned()),
    })
}
