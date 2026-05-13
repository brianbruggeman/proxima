//! `config` — typed config with `conflaguration`: the layered fluent builder,
//! and the serialize <-> deserialize round trip.
//!
//! `transform` teaches you to write one `Pipe`. Every pipe you compose still
//! needs to be configured — ring sizes, timeouts, endpoints — and proxima's
//! house pattern for that is one struct that is simultaneously:
//!   - a `bon::Builder` (explicit `.field(value)` construction),
//!   - a `serde::{Deserialize, Serialize}` (files, wire formats),
//!   - a `conflaguration::Settings` (env vars, prefixed keys), and
//!   - a `conflaguration::Validate` (rejects bad values after construction).
//!
//! A small `layered()` builder on top composes those sources: start from
//! `ServerConfig::default()`, then `.from_path(...)` or `.from_env()` — each
//! call RE-RESOLVES THE WHOLE STRUCT from that source (file/env fields it
//! sets plus `#[setting]`/`#[serde(default)]` for the rest) — and finally
//! `.with_*(...)` mutates just the touched field. So precedence is call
//! order: put `.with_*` after `.from_path`/`.from_env` for an explicit
//! override to win.
//!
//! Run: `cargo run --example config`

use std::path::Path;

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "EXAMPLE")]
#[builder(derive(Clone, Debug))]
struct ServerConfig {
    #[setting(default = "0.0.0.0")]
    #[serde(default = "default_host")]
    #[builder(default = default_host())]
    host: String,

    #[setting(default = 8080)]
    #[serde(default = "default_port")]
    #[builder(default = default_port())]
    port: u16,

    #[setting(default = 64)]
    #[serde(default = "default_max_connections")]
    #[builder(default = default_max_connections())]
    max_connections: usize,

    #[setting(default = 5000)]
    #[serde(default = "default_request_timeout_ms")]
    #[builder(default = default_request_timeout_ms())]
    request_timeout_ms: u64,
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8080
}

fn default_max_connections() -> usize {
    64
}

fn default_request_timeout_ms() -> u64 {
    5000
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl Validate for ServerConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.host.is_empty() {
            errors.push(ValidationMessage::new("host", "must not be empty"));
        }
        if self.port == 0 {
            errors.push(ValidationMessage::new("port", "must be > 0"));
        }
        if self.max_connections == 0 {
            errors.push(ValidationMessage::new("max_connections", "must be > 0"));
        }
        if self.request_timeout_ms == 0 {
            errors.push(ValidationMessage::new("request_timeout_ms", "must be > 0"));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

/// Fluent builder for [`ServerConfig`] with call-order precedence.
struct ServerConfigLayers {
    inner: ServerConfig,
}

impl ServerConfig {
    fn layered() -> ServerConfigLayers {
        ServerConfigLayers {
            inner: ServerConfig::default(),
        }
    }
}

impl ServerConfigLayers {
    // `from_*` consuming `self` is the intentional fluent-chain shape (mirrors
    // proxima-telemetry's `TelemetryLayerBuilder`), not a `Self::from_x()` constructor.
    #[allow(clippy::wrong_self_convention)]
    fn from_path<PathRef: AsRef<Path>>(
        mut self,
        path: PathRef,
    ) -> Result<Self, conflaguration::Error> {
        self.inner = conflaguration::from_file(path.as_ref())?;
        Ok(self)
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_env(mut self) -> Result<Self, conflaguration::Error> {
        self.inner = ServerConfig::from_env()?;
        Ok(self)
    }

    fn with_host(mut self, host: impl Into<String>) -> Self {
        self.inner.host = host.into();
        self
    }

    fn with_port(mut self, port: u16) -> Self {
        self.inner.port = port;
        self
    }

    fn with_max_connections(mut self, max_connections: usize) -> Self {
        self.inner.max_connections = max_connections;
        self
    }

    fn with_request_timeout_ms(mut self, request_timeout_ms: u64) -> Self {
        self.inner.request_timeout_ms = request_timeout_ms;
        self
    }

    fn build(self) -> ServerConfig {
        self.inner
    }
}

fn print_config(label: &str, config: &ServerConfig) {
    println!(
        "{label}: host={} port={} max_connections={} request_timeout_ms={}",
        config.host, config.port, config.max_connections, config.request_timeout_ms
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- round 0: defaults ---");
    let defaults = ServerConfig::default();
    print_config("defaults", &defaults);
    assert_eq!(defaults.host, "0.0.0.0");
    assert_eq!(defaults.port, 8080);

    println!("\n--- round 1: layered().from_path(...) overlays a file ---");
    let config_dir = tempfile::TempDir::new()?;
    let config_path = config_dir.path().join("server.toml");
    std::fs::write(&config_path, "host = \"10.0.0.5\"\nport = 9000\n")?;
    let from_file = ServerConfig::layered().from_path(&config_path)?.build();
    print_config("from file", &from_file);
    assert_eq!(from_file.host, "10.0.0.5");
    assert_eq!(from_file.port, 9000);
    assert_eq!(
        from_file.max_connections, defaults.max_connections,
        "the file didn't set max_connections, so it fell back to #[serde(default)] \
         which matches the builder default"
    );

    println!("\n--- round 2: layered().from_env() re-resolves fresh from the environment ---");
    let from_env = temp_env::with_vars(
        [
            ("EXAMPLE_MAX_CONNECTIONS", Some("256")),
            ("EXAMPLE_PORT", Some("7000")),
        ],
        || ServerConfig::layered().from_env(),
    )?
    .build();
    print_config("from env", &from_env);
    assert_eq!(from_env.max_connections, 256);
    assert_eq!(from_env.port, 7000);
    assert_eq!(
        from_env.host, defaults.host,
        "EXAMPLE_HOST was never set, so from_env resolved #[setting(default)] for it — \
         round 1's file value does not survive into round 2, from_env replaces the whole struct"
    );

    println!("\n--- round 3: with_* after from_env wins (call-order precedence) ---");
    let layered_from_env = temp_env::with_vars([("EXAMPLE_MAX_CONNECTIONS", Some("256"))], || {
        ServerConfig::layered().from_env()
    })?;
    let layered = layered_from_env
        .with_host("override.local")
        .with_request_timeout_ms(1500)
        .build();
    print_config("layered: env + explicit overrides", &layered);
    assert_eq!(
        layered.max_connections, 256,
        "env value kept — with_* never touched this field"
    );
    assert_eq!(
        layered.host, "override.local",
        "explicit with_host wins — it was called after from_env"
    );
    assert_eq!(layered.request_timeout_ms, 1500);
    layered.validate()?;

    println!("\n--- round 4: serialize <-> deserialize round trip ---");
    let serialized = toml::to_string(&layered)?;
    print!("{serialized}");
    let restored: ServerConfig = toml::from_str(&serialized)?;
    assert_eq!(
        restored, layered,
        "deserializing must reproduce the exact struct"
    );
    let reserialized = toml::to_string(&restored)?;
    assert_eq!(
        reserialized, serialized,
        "re-serializing the restored value must reproduce the exact bytes"
    );
    println!(
        "round trip: OK ({} bytes, byte-identical)",
        serialized.len()
    );

    println!("\n--- round 5: Validate rejects an invalid config ---");
    let invalid = ServerConfig::layered()
        .with_port(0)
        .with_max_connections(0)
        .build();
    match invalid.validate() {
        Ok(()) => unreachable!("port=0 and max_connections=0 must fail validation"),
        Err(conflaguration::Error::Validation { errors }) => {
            assert_eq!(
                errors.len(),
                2,
                "both bad fields are collected, not just the first"
            );
            for error in &errors {
                println!("  rejected: {error}");
            }
        }
        Err(other) => unreachable!("expected a Validation error, got {other:?}"),
    }

    Ok(())
}
