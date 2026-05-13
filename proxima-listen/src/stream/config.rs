//! `ListenerStreamConfig` — the runtime config + fluent builder surface (P4:
//! first-class conflaguration AND first-class fluent builder, BOTH).
//!
//! Mirrors `proxima_telemetry::log_buffer`'s `LogBufferConfig`: `#[derive(Builder,
//! Deserialize, Serialize, Settings)]` + [`Validate`], a
//! `ListenerStreamLayerBuilder` with call-order precedence, and a typed env
//! surface. The defaults come FROM the `sized` consts that `build.rs`
//! generates from `proxima-listeners-stream.toml`, so the runtime default
//! follows the compile-time / no_std+no_alloc floor — there is no double
//! source of truth.
//!
//! This is the process-level default surface for [`super::StreamListenerProtocol`]
//! and [`super::StreamListenProtocol`]. It is distinct from the per-listener
//! JSON `spec` that `ListenProtocol::serve` still accepts: `spec` overrides a
//! single listener registration at runtime; `ListenerStreamConfig` sets the
//! process-wide default those registrations fall back to.
//!
//! Tier: std (conflaguration, fs, env). Once this crate lifts off std, the
//! `sized` consts become the no_std+no_alloc floor's only knob and this layer
//! becomes the std runtime override on top, unchanged.
//!
//! `ListenerStreamLayerBuilder` supports call-order precedence:
//!
//! - **Operator config wins**: put `.with_*` BEFORE `.from_path` / `.from_env`.
//! - **Code overrides win**: put `.with_*` AFTER `.from_path` / `.from_env`.

use std::path::Path;

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

fn default_method() -> String {
    super::sized::LISTENER_METHOD_DEFAULT.to_string()
}

fn default_path() -> String {
    super::sized::LISTENER_PATH_DEFAULT.to_string()
}

fn default_chunk_bytes() -> usize {
    super::sized::LISTENER_CHUNK_BYTES_DEFAULT
}

/// Runtime configuration for [`super::StreamListenerProtocol`] /
/// [`super::StreamListenProtocol`]. One built `ListenerStreamConfig` == one
/// serialisable config == one stream-listener default policy.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "STREAM_LISTENER")]
#[builder(derive(Clone, Debug))]
pub struct ListenerStreamConfig {
    /// Synthetic request method assigned to every accepted stream connection.
    /// Defaults from `proxima-listeners-stream.toml`'s `[listener] method`
    /// (the `sized` floor).
    #[setting(default = "STREAM")]
    #[serde(default = "default_method")]
    #[builder(default = default_method())]
    pub method: String,

    /// Synthetic request path assigned to every accepted stream connection.
    /// Defaults from `proxima-listeners-stream.toml`'s `[listener] path`
    /// (the `sized` floor).
    #[setting(default = "/")]
    #[serde(default = "default_path")]
    #[builder(default = default_path())]
    pub path: String,

    /// Read-buffer chunk size (bytes) used to stream a connection's body.
    /// Defaults from `proxima-listeners-stream.toml`'s `[listener]
    /// chunk_bytes` (the `sized` floor).
    #[setting(default = 65536)]
    #[serde(default = "default_chunk_bytes")]
    #[builder(default = default_chunk_bytes())]
    pub chunk_bytes: usize,
}

impl Default for ListenerStreamConfig {
    fn default() -> Self {
        ListenerStreamConfig::builder().build()
    }
}

impl Validate for ListenerStreamConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.method.is_empty() {
            errors.push(ValidationMessage::new("method", "must not be empty"));
        }
        if self.path.is_empty() {
            errors.push(ValidationMessage::new("path", "must not be empty"));
        }
        if self.chunk_bytes == 0 {
            errors.push(ValidationMessage::new("chunk_bytes", "must be > 0"));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl ListenerStreamConfig {
    /// Start a layered builder from the `sized`-seeded defaults.
    #[must_use]
    pub fn layered() -> ListenerStreamLayerBuilder {
        ListenerStreamLayerBuilder {
            inner: ListenerStreamConfig::default(),
            method_set: false,
            path_set: false,
            chunk_bytes_set: false,
        }
    }
}

/// Partial view of [`ListenerStreamConfig`] used by `.from_path`/`.underlay_path`
/// — only fields actually present in the file are applied, so a file setting
/// one field never clobbers the other with a re-resolved default.
#[derive(Debug, Default, Deserialize)]
struct ListenerStreamConfigPartial {
    method: Option<String>,
    path: Option<String>,
    chunk_bytes: Option<usize>,
}

/// Fluent builder for [`ListenerStreamConfig`]. Every source (`.from_path`,
/// `.from_env`, `.underlay_path`, `.underlay_env`, `.with_*`) contributes only
/// the fields it actually specifies, merged onto the accumulated config — a
/// field a source doesn't touch falls through to whatever prior layers set.
/// `.from_path`/`.from_env` override (last writer wins per field);
/// `.underlay_path`/`.underlay_env` fill only fields still unset; `.with_*`
/// always acts as an override at its call position.
pub struct ListenerStreamLayerBuilder {
    inner: ListenerStreamConfig,
    method_set: bool,
    path_set: bool,
    chunk_bytes_set: bool,
}

impl ListenerStreamLayerBuilder {
    /// Merge a TOML/JSON file's fields onto the accumulated config; the file
    /// wins for every field it specifies.
    pub fn from_path<P: AsRef<Path>>(mut self, path: P) -> Result<Self, conflaguration::Error> {
        let partial: ListenerStreamConfigPartial = conflaguration::from_file(path.as_ref())?;
        if let Some(method) = partial.method {
            self.inner.method = method;
            self.method_set = true;
        }
        if let Some(path) = partial.path {
            self.inner.path = path;
            self.path_set = true;
        }
        if let Some(chunk_bytes) = partial.chunk_bytes {
            self.inner.chunk_bytes = chunk_bytes;
            self.chunk_bytes_set = true;
        }
        Ok(self)
    }

    /// Fill any still-unset fields from a TOML/JSON file; already-set fields
    /// are left untouched.
    pub fn underlay_path<P: AsRef<Path>>(mut self, path: P) -> Result<Self, conflaguration::Error> {
        let partial: ListenerStreamConfigPartial = conflaguration::from_file(path.as_ref())?;
        if !self.method_set
            && let Some(method) = partial.method
        {
            self.inner.method = method;
            self.method_set = true;
        }
        if !self.path_set
            && let Some(path) = partial.path
        {
            self.inner.path = path;
            self.path_set = true;
        }
        if !self.chunk_bytes_set
            && let Some(chunk_bytes) = partial.chunk_bytes
        {
            self.inner.chunk_bytes = chunk_bytes;
            self.chunk_bytes_set = true;
        }
        Ok(self)
    }

    /// Merge `STREAM_LISTENER_*` env-set fields onto the accumulated config;
    /// env wins for every field it sets. Unset env vars leave the current
    /// value untouched.
    pub fn from_env(mut self) -> Result<Self, conflaguration::Error> {
        let resolved = ListenerStreamConfig::from_env()?;
        if env_is_set("STREAM_LISTENER_METHOD") {
            self.inner.method = resolved.method;
            self.method_set = true;
        }
        if env_is_set("STREAM_LISTENER_PATH") {
            self.inner.path = resolved.path;
            self.path_set = true;
        }
        if env_is_set("STREAM_LISTENER_CHUNK_BYTES") {
            self.inner.chunk_bytes = resolved.chunk_bytes;
            self.chunk_bytes_set = true;
        }
        Ok(self)
    }

    /// Fill any still-unset fields from `STREAM_LISTENER_*` env vars;
    /// already-set fields are left untouched even if the matching env var is
    /// set.
    pub fn underlay_env(mut self) -> Result<Self, conflaguration::Error> {
        let resolved = ListenerStreamConfig::from_env()?;
        if !self.method_set && env_is_set("STREAM_LISTENER_METHOD") {
            self.inner.method = resolved.method;
            self.method_set = true;
        }
        if !self.path_set && env_is_set("STREAM_LISTENER_PATH") {
            self.inner.path = resolved.path;
            self.path_set = true;
        }
        if !self.chunk_bytes_set && env_is_set("STREAM_LISTENER_CHUNK_BYTES") {
            self.inner.chunk_bytes = resolved.chunk_bytes;
            self.chunk_bytes_set = true;
        }
        Ok(self)
    }

    /// Set the synthetic request method assigned to accepted stream
    /// connections.
    #[must_use]
    pub fn with_method(mut self, method: impl Into<String>) -> Self {
        self.inner.method = method.into();
        self.method_set = true;
        self
    }

    /// Set the synthetic request path assigned to accepted stream
    /// connections.
    #[must_use]
    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.inner.path = path.into();
        self.path_set = true;
        self
    }

    /// Set the read-buffer chunk size (bytes).
    #[must_use]
    pub fn with_chunk_bytes(mut self, chunk_bytes: usize) -> Self {
        self.inner.chunk_bytes = chunk_bytes.max(1);
        self.chunk_bytes_set = true;
        self
    }

    /// The built immutable config.
    #[must_use]
    pub fn build(self) -> ListenerStreamConfig {
        self.inner
    }
}

fn env_is_set(name: &str) -> bool {
    std::env::var(name).is_ok()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // the fluent builder and the conflaguration surface agree on defaults —
    // both seeded by the `sized` floor (build.rs), the single source. pins
    // the bridge: if the runtime default ever diverges from the sized const,
    // this test fails.
    #[test]
    fn defaults_track_the_sized_floor() {
        let config = ListenerStreamConfig::default();
        assert_eq!(config.method, super::super::sized::LISTENER_METHOD_DEFAULT);
        assert_eq!(config.path, super::super::sized::LISTENER_PATH_DEFAULT);
        assert_eq!(
            config.chunk_bytes,
            super::super::sized::LISTENER_CHUNK_BYTES_DEFAULT
        );
        // the env-overlay (from_env, no vars set) must agree with the const too.
        temp_env::with_vars::<&str, &str, _, _>([], || {
            let from_env = ListenerStreamConfig::from_env().expect("from_env");
            assert_eq!(from_env.method, super::super::sized::LISTENER_METHOD_DEFAULT);
            assert_eq!(from_env.path, super::super::sized::LISTENER_PATH_DEFAULT);
            assert_eq!(
                from_env.chunk_bytes,
                super::super::sized::LISTENER_CHUNK_BYTES_DEFAULT
            );
        });
    }

    // the runtime default equals the old hand-rolled magic constants
    // (DEFAULT_METHOD = "STREAM", DEFAULT_PATH = "/",
    // DEFAULT_CHUNK_BYTES = 64 * 1024) — this refactor is behavior-preserving.
    #[test]
    fn defaults_match_the_former_magic_constants() {
        let config = ListenerStreamConfig::default();
        assert_eq!(config.method, "STREAM");
        assert_eq!(config.path, "/");
        assert_eq!(config.chunk_bytes, 64 * 1024);
    }

    #[test]
    fn default_config_validates() {
        let config = ListenerStreamConfig::default();
        assert!(config.validate().is_ok(), "default config should validate");
    }

    #[test]
    fn zero_chunk_bytes_rejected() {
        let config = ListenerStreamConfig::builder().chunk_bytes(0).build();
        let error = config.validate().expect_err("validate must reject 0");
        assert!(format!("{error:?}").contains("chunk_bytes"));
    }

    #[test]
    fn empty_method_rejected() {
        let config = ListenerStreamConfig::builder()
            .method(String::new())
            .build();
        let error = config.validate().expect_err("validate must reject empty");
        assert!(format!("{error:?}").contains("method"));
    }

    #[test]
    fn builder_starts_at_default() {
        let from_layered = ListenerStreamConfig::layered().build();
        let from_default = ListenerStreamConfig::default();
        assert_eq!(from_layered, from_default);
    }

    #[test]
    fn with_overrides_default() {
        let config = ListenerStreamConfig::layered()
            .with_chunk_bytes(4096)
            .build();
        assert_eq!(config.chunk_bytes, 4096);
        assert_eq!(ListenerStreamConfig::default().chunk_bytes, 64 * 1024);
    }

    #[test]
    fn from_path_overrides_default() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("listeners-stream.toml");
        std::fs::write(&path, "chunk_bytes = 8192\n").expect("write toml");
        let config = ListenerStreamConfig::layered()
            .from_path(&path)
            .expect("from_path")
            .build();
        assert_eq!(config.chunk_bytes, 8192);
        assert_eq!(config.method, "STREAM", "untouched field");
    }

    // env override demonstration: STREAM_LISTENER_* vars flow through
    // from_env() and are picked up in the built config.
    #[test]
    fn env_override_demonstration() {
        temp_env::with_vars(
            [
                ("STREAM_LISTENER_CHUNK_BYTES", Some("777")),
                ("STREAM_LISTENER_METHOD", Some("CUSTOM")),
            ],
            || {
                let config = ListenerStreamConfig::from_env().expect("from_env");
                assert_eq!(config.chunk_bytes, 777);
                assert_eq!(config.method, "CUSTOM");
            },
        );
    }

    #[test]
    fn from_env_overlays_via_conflaguration() {
        temp_env::with_vars([("STREAM_LISTENER_CHUNK_BYTES", Some("2048"))], || {
            let config = ListenerStreamConfig::layered()
                .from_env()
                .expect("from_env")
                .build();
            assert_eq!(config.chunk_bytes, 2048);
            assert_eq!(config.method, "STREAM", "untouched field");
        });
    }

    // underlay never clobbers an already-set field; it DOES fill an unset one.
    #[test]
    fn underlay_path_fills_only_unset_fields() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("listeners-stream.toml");
        std::fs::write(&path, "chunk_bytes = 1024\nmethod = \"OTHER\"\n").expect("write toml");
        let config = ListenerStreamConfig::layered()
            .with_chunk_bytes(64)
            .underlay_path(&path)
            .expect("underlay_path")
            .build();
        assert_eq!(
            config.chunk_bytes, 64,
            "already set by with_*; the file's value is dropped"
        );
        assert_eq!(
            config.method, "OTHER",
            "unset before underlay; the file fills it"
        );
    }

    #[test]
    fn config_round_trips_through_serde() {
        let built = ListenerStreamConfig::layered()
            .with_chunk_bytes(2048)
            .with_method("CUSTOM")
            .build();
        let json = serde_json::to_string(&built).expect("serialize");
        let from_json: ListenerStreamConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(from_json, built);
    }
}
