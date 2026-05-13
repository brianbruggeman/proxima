//! Typed settings for proxima. Two responsibilities cleanly split:
//!
//! - **Map-keyed registries** (`listeners`, `upstreams`, `middlewares`,
//!   `pipes`) — file-driven via serde. No env-var shape ("how do
//!   you spell `PROXIMA_LISTENER_PUBLIC_TLS_CERT` cleanly?"). Plugin
//!   variants are `serde_json::Value`-shaped at the top level and
//!   late-typed at the factory-registry boundary.
//!
//! - **Tunables** (`http`, `zstd`, `buffer_pool`) — env-var-friendly
//!   tuning knobs. `#[derive(conflaguration::Settings)]` gives free
//!   env-var overrides (`PROXIMA_HTTP_BUFFER_BYTES=32768`), defaults,
//!   validation, and sensitive masking.
//!
//! Each typed struct also derives `bon::Builder`, so fluent
//! construction is `T::builder().field(...).build()` and round-trip
//! editing is `existing.builder().field(new).build()`. Builder ⇄
//! TOML ⇄ Builder is an identity round-trip — that's the load-bearing
//! invariant for the fluent ⇄ Settings story.

pub mod chain;
pub mod listener;
pub mod middleware;
pub mod tuning;
pub mod upstream;

use std::path::Path;

use crate::config_format::default_config_format_registry;
use crate::error::ProximaError;

pub use chain::{Chain, Composable};
pub use listener::HttpListener;
#[cfg(unix)]
pub use listener::HttpUdsListener;
#[cfg(feature = "tls")]
pub use listener::HttpsListener;
pub use middleware::{BearerAuth, ClientAuth, DigestAuth, OauthAuth, RateLimit, SigV4Auth};
pub use upstream::HttpUpstream;

use std::collections::BTreeMap;

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

pub use tuning::{BufferPoolTuning, HttpTuning, ZstdTuning};

/// Map-keyed registry entry. Holds the discriminator (`type`) plus
/// the rest of the section as untyped JSON. The factory registry
/// resolves `type` → typed config struct at load time. Plugin
/// crates register their own factories the same way built-ins do.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RegistryEntry {
    /// Discriminator that selects the typed config struct in the
    /// factory registry.
    pub r#type: String,

    /// Remaining fields, late-deserialized by the registered factory.
    #[serde(flatten)]
    pub spec: serde_json::Value,
}

/// Top-level proxima configuration. The shape the daemon, CLI, and
/// fluent builder all converge on. TOML ⇄ fluent are isomorphic via
/// `bon::Builder` + `serde::Deserialize`.
#[derive(Debug, Clone, Default, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA")]
#[builder(derive(Clone, Debug))]
pub struct ProximaSettings {
    /// Named listeners. `listener.public`, `listener.admin`, etc.
    /// File-driven; no env-var pattern (skip for env-var resolution).
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub listeners: BTreeMap<String, RegistryEntry>,

    /// Named upstreams. `upstream.backend`, etc. File-driven.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub upstreams: BTreeMap<String, RegistryEntry>,

    /// Named middlewares. `middleware.auth`, `middleware.rate-limit`,
    /// etc. File-driven; plugin-extensible via factory registry.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub middlewares: BTreeMap<String, RegistryEntry>,

    /// Named pipes. `pipe.api`, etc. File-driven; references
    /// listeners / middlewares / upstreams by name.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub pipes: BTreeMap<String, RegistryEntry>,

    /// Named producers. `producer.heartbeat`, etc. File-driven. Producers
    /// are self-starting Pipes — they declare continuous-loop work via
    /// `Pipe::background_tasks()` and are NOT mounted on any listener.
    ///
    /// S3 of the proxima-notify initiative (see `docs/proxima-notify/SUBSTRATE.md`).
    /// Active only when the `producer-graph-config` feature is enabled on
    /// proxima; without it, this map is ignored at `App::apply_settings`
    /// time but still parses from TOML (forward-compatible with future
    /// configs).
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub producers: BTreeMap<String, RegistryEntry>,

    /// HTTP framing-layer tunables. Env-overridable per field
    /// (`PROXIMA_HTTP_RESPONSE_BUFFER_BYTES=32768`).
    /// `nested` accumulates parent prefix + field name uppercased,
    /// so the composed env-var key is `PROXIMA_HTTP_<FIELD>`.
    #[setting(nested)]
    #[serde(default)]
    #[builder(default)]
    pub http: HttpTuning,

    /// zstd compression tunables (recording sinks etc.).
    #[setting(nested)]
    #[serde(default)]
    #[builder(default)]
    pub zstd: ZstdTuning,

    /// Per-worker pooled-buffer tunables.
    #[setting(nested)]
    #[serde(default)]
    #[builder(default)]
    pub buffer_pool: BufferPoolTuning,
}

impl ProximaSettings {
    /// Load typed settings from a config file. Format is sniffed from
    /// the file extension via the default config-format registry
    /// (`toml`, `json`, `yaml`, `ron`, `json5`, `xml`).
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ProximaError> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path).map_err(ProximaError::Io)?;
        let hint = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(str::to_string);
        let registry = default_config_format_registry()?;
        let value = registry.parse_with_hint(&raw, hint.as_deref())?;
        serde_json::from_value(value)
            .map_err(|err| ProximaError::Config(format!("decode ProximaSettings: {err}")))
    }
}

impl Validate for ProximaSettings {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        self.http
            .validate()
            .map_err(|err| collect_validation_errors(&mut errors, "http", err))
            .ok();
        self.zstd
            .validate()
            .map_err(|err| collect_validation_errors(&mut errors, "zstd", err))
            .ok();
        self.buffer_pool
            .validate()
            .map_err(|err| collect_validation_errors(&mut errors, "buffer_pool", err))
            .ok();
        for (name, entry) in &self.listeners {
            if entry.r#type.is_empty() {
                errors.push(ValidationMessage::new(
                    format!("listener.{name}.type"),
                    "must not be empty",
                ));
            }
        }
        for (name, entry) in &self.upstreams {
            if entry.r#type.is_empty() {
                errors.push(ValidationMessage::new(
                    format!("upstream.{name}.type"),
                    "must not be empty",
                ));
            }
        }
        for (name, entry) in &self.middlewares {
            if entry.r#type.is_empty() {
                errors.push(ValidationMessage::new(
                    format!("middleware.{name}.type"),
                    "must not be empty",
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

fn collect_validation_errors(
    target: &mut Vec<ValidationMessage>,
    section: &str,
    error: conflaguration::Error,
) {
    if let conflaguration::Error::Validation { errors } = error {
        for mut message in errors {
            message.prepend_path(section);
            target.push(message);
        }
    }
}
