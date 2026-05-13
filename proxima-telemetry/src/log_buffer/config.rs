//! `LogBufferConfig` — the runtime config + fluent builder surface (P4:
//! first-class conflaguration AND first-class fluent builder, BOTH).
//!
//! Mirrors proxima-telemetry's `InstrumentConfig`: `#[derive(Builder,
//! Deserialize, Serialize, Settings)]` + [`Validate`], a `LogBufferLayerBuilder`
//! with call-order precedence, and a typed env surface. The defaults come FROM
//! the `sized` consts that `build.rs` generates from `proxima-log-buffer.toml`,
//! so the runtime default follows the compile-time / no_std+no_alloc floor —
//! there is no double source of truth.
//!
//! Tier: std (conflaguration, fs, env). The retained-ring sans-IO core lives
//! in the `ring` module (no_std + alloc clean); this crate stays std by
//! design for the subscriber fanout (`arc-swap`'s no_std path needs a nightly
//! feature; `dashmap`'s registry has no no_std story at all) and for the
//! conflaguration-backed runtime config surface itself. The `sized` consts
//! remain the single source of truth for both this layer's runtime defaults
//! and the core crate's no_std + alloc floor.
//!
//! `LogBufferLayerBuilder` supports call-order precedence:
//!
//! - **Operator config wins**: put `.with_*` BEFORE `.from_path` / `.from_env`.
//! - **Code overrides win**: put `.with_*` AFTER `.from_path` / `.from_env`.

use std::path::Path;

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

fn default_capacity() -> usize {
    super::sized::LOG_BUFFER_CAPACITY_DEFAULT
}

fn default_live_tail_channel_capacity() -> usize {
    super::sized::LIVE_TAIL_CHANNEL_CAPACITY_DEFAULT
}

/// Runtime configuration for a [`crate::LogBuffer`]. One built `LogBufferConfig`
/// == one serialisable config == one buffer sizing policy.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "LOG_BUFFER")]
#[builder(derive(Clone, Debug))]
pub struct LogBufferConfig {
    /// Retained ring-buffer capacity in lines. Oldest line is evicted once
    /// full. Defaults from `proxima-log-buffer.toml`'s `[buffer] capacity`
    /// (the `sized` floor).
    #[setting(default = 1024)]
    #[serde(default = "default_capacity")]
    #[builder(default = default_capacity())]
    pub capacity: usize,

    /// Per-subscriber live-tail queue capacity. A slow subscriber whose queue
    /// fills has its newest line dropped. Defaults from
    /// `proxima-log-buffer.toml`'s `[live_tail] channel_capacity` (the
    /// `sized` floor).
    #[setting(default = 256)]
    #[serde(default = "default_live_tail_channel_capacity")]
    #[builder(default = default_live_tail_channel_capacity())]
    pub live_tail_channel_capacity: usize,
}

impl Default for LogBufferConfig {
    fn default() -> Self {
        LogBufferConfig::builder().build()
    }
}

impl Validate for LogBufferConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.capacity == 0 {
            errors.push(ValidationMessage::new("capacity", "must be > 0"));
        }
        if self.live_tail_channel_capacity == 0 {
            errors.push(ValidationMessage::new(
                "live_tail_channel_capacity",
                "must be > 0",
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl LogBufferConfig {
    /// Start a layered builder from the `sized`-seeded defaults.
    #[must_use]
    pub fn layered() -> LogBufferLayerBuilder {
        LogBufferLayerBuilder {
            inner: LogBufferConfig::default(),
            capacity_set: false,
            live_tail_channel_capacity_set: false,
        }
    }
}

/// Partial view of [`LogBufferConfig`] used by `.from_path`/`.underlay_path` —
/// only fields actually present in the file are applied, so a file setting
/// one field never clobbers the other with a re-resolved default.
#[derive(Debug, Default, Deserialize)]
struct LogBufferConfigPartial {
    capacity: Option<usize>,
    live_tail_channel_capacity: Option<usize>,
}

/// Fluent builder for [`LogBufferConfig`]. Every source (`.from_path`,
/// `.from_env`, `.underlay_path`, `.underlay_env`, `.with_*`) contributes only
/// the fields it actually specifies, merged onto the accumulated config — a
/// field a source doesn't touch falls through to whatever prior layers set.
/// `.from_path`/`.from_env` override (last writer wins per field);
/// `.underlay_path`/`.underlay_env` fill only fields still unset; `.with_*`
/// always acts as an override at its call position.
pub struct LogBufferLayerBuilder {
    inner: LogBufferConfig,
    capacity_set: bool,
    live_tail_channel_capacity_set: bool,
}

impl LogBufferLayerBuilder {
    /// Merge a TOML/JSON file's fields onto the accumulated config; the file
    /// wins for every field it specifies.
    pub fn from_path<P: AsRef<Path>>(mut self, path: P) -> Result<Self, conflaguration::Error> {
        let partial: LogBufferConfigPartial = conflaguration::from_file(path.as_ref())?;
        if let Some(capacity) = partial.capacity {
            self.inner.capacity = capacity;
            self.capacity_set = true;
        }
        if let Some(channel_capacity) = partial.live_tail_channel_capacity {
            self.inner.live_tail_channel_capacity = channel_capacity;
            self.live_tail_channel_capacity_set = true;
        }
        Ok(self)
    }

    /// Fill any still-unset fields from a TOML/JSON file; already-set fields
    /// are left untouched.
    pub fn underlay_path<P: AsRef<Path>>(mut self, path: P) -> Result<Self, conflaguration::Error> {
        let partial: LogBufferConfigPartial = conflaguration::from_file(path.as_ref())?;
        if !self.capacity_set
            && let Some(capacity) = partial.capacity
        {
            self.inner.capacity = capacity;
            self.capacity_set = true;
        }
        if !self.live_tail_channel_capacity_set
            && let Some(channel_capacity) = partial.live_tail_channel_capacity
        {
            self.inner.live_tail_channel_capacity = channel_capacity;
            self.live_tail_channel_capacity_set = true;
        }
        Ok(self)
    }

    /// Merge `LOG_BUFFER_*` env-set fields onto the accumulated config; env
    /// wins for every field it sets. Unset env vars leave the current value
    /// untouched.
    pub fn from_env(mut self) -> Result<Self, conflaguration::Error> {
        let resolved = LogBufferConfig::from_env()?;
        if env_is_set("LOG_BUFFER_CAPACITY") {
            self.inner.capacity = resolved.capacity;
            self.capacity_set = true;
        }
        if env_is_set("LOG_BUFFER_LIVE_TAIL_CHANNEL_CAPACITY") {
            self.inner.live_tail_channel_capacity = resolved.live_tail_channel_capacity;
            self.live_tail_channel_capacity_set = true;
        }
        Ok(self)
    }

    /// Fill any still-unset fields from `LOG_BUFFER_*` env vars; already-set
    /// fields are left untouched even if the matching env var is set.
    pub fn underlay_env(mut self) -> Result<Self, conflaguration::Error> {
        let resolved = LogBufferConfig::from_env()?;
        if !self.capacity_set && env_is_set("LOG_BUFFER_CAPACITY") {
            self.inner.capacity = resolved.capacity;
            self.capacity_set = true;
        }
        if !self.live_tail_channel_capacity_set
            && env_is_set("LOG_BUFFER_LIVE_TAIL_CHANNEL_CAPACITY")
        {
            self.inner.live_tail_channel_capacity = resolved.live_tail_channel_capacity;
            self.live_tail_channel_capacity_set = true;
        }
        Ok(self)
    }

    /// Set the retained ring-buffer capacity (lines).
    #[must_use]
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.inner.capacity = capacity;
        self.capacity_set = true;
        self
    }

    /// Set the per-subscriber live-tail queue capacity.
    #[must_use]
    pub fn with_live_tail_channel_capacity(mut self, capacity: usize) -> Self {
        self.inner.live_tail_channel_capacity = capacity;
        self.live_tail_channel_capacity_set = true;
        self
    }

    /// The built immutable config.
    #[must_use]
    pub fn build(self) -> LogBufferConfig {
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
    // both seeded by the `sized` floor (build.rs), the single source. pins the
    // bridge: if the runtime default ever diverges from the sized const, this
    // test fails.
    #[test]
    fn defaults_track_the_sized_floor() {
        let config = LogBufferConfig::default();
        assert_eq!(config.capacity, super::super::sized::LOG_BUFFER_CAPACITY_DEFAULT);
        assert_eq!(
            config.live_tail_channel_capacity,
            super::super::sized::LIVE_TAIL_CHANNEL_CAPACITY_DEFAULT
        );
        // the env-overlay (from_env, no vars set) must agree with the const too.
        temp_env::with_vars::<&str, &str, _, _>([], || {
            let from_env = LogBufferConfig::from_env().expect("from_env");
            assert_eq!(from_env.capacity, super::super::sized::LOG_BUFFER_CAPACITY_DEFAULT);
            assert_eq!(
                from_env.live_tail_channel_capacity,
                super::super::sized::LIVE_TAIL_CHANNEL_CAPACITY_DEFAULT
            );
        });
    }

    // the runtime default equals the old hand-rolled magic constants
    // (DEFAULT_LOG_BUFFER_CAPACITY = 1024, LIVE_TAIL_CHANNEL_CAPACITY = 256) —
    // this refactor is behavior-preserving.
    #[test]
    fn defaults_match_the_former_magic_constants() {
        let config = LogBufferConfig::default();
        assert_eq!(config.capacity, 1024);
        assert_eq!(config.live_tail_channel_capacity, 256);
    }

    #[test]
    fn default_config_validates() {
        let config = LogBufferConfig::default();
        assert!(config.validate().is_ok(), "default config should validate");
    }

    #[test]
    fn zero_capacity_rejected() {
        let config = LogBufferConfig::builder().capacity(0).build();
        let error = config.validate().expect_err("validate must reject 0");
        assert!(format!("{error:?}").contains("capacity"));
    }

    #[test]
    fn zero_live_tail_channel_capacity_rejected() {
        let config = LogBufferConfig::builder()
            .live_tail_channel_capacity(0)
            .build();
        let error = config.validate().expect_err("validate must reject 0");
        assert!(format!("{error:?}").contains("live_tail_channel_capacity"));
    }

    #[test]
    fn builder_starts_at_default() {
        let from_layered = LogBufferConfig::layered().build();
        let from_default = LogBufferConfig::default();
        assert_eq!(from_layered, from_default);
    }

    #[test]
    fn with_overrides_default() {
        let config = LogBufferConfig::layered().with_capacity(2048).build();
        assert_eq!(config.capacity, 2048);
        assert_eq!(LogBufferConfig::default().capacity, 1024);
    }

    #[test]
    fn from_path_overrides_default() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("log-buffer.toml");
        std::fs::write(&path, "capacity = 4096\n").expect("write toml");
        let config = LogBufferConfig::layered()
            .from_path(&path)
            .expect("from_path")
            .build();
        assert_eq!(config.capacity, 4096);
        assert_eq!(config.live_tail_channel_capacity, 256, "untouched field");
    }

    #[test]
    fn from_env_overlays_via_conflaguration() {
        temp_env::with_vars([("LOG_BUFFER_CAPACITY", Some("8192"))], || {
            let config = LogBufferConfig::layered()
                .from_env()
                .expect("from_env")
                .build();
            assert_eq!(config.capacity, 8192);
            assert_eq!(config.live_tail_channel_capacity, 256, "untouched field");
        });
    }

    // env override demonstration: PROXIMA_LOG_BUFFER_* vars flow through
    // from_env() and are picked up in the built config.
    #[test]
    fn env_override_demonstration() {
        temp_env::with_vars(
            [
                ("LOG_BUFFER_CAPACITY", Some("777")),
                ("LOG_BUFFER_LIVE_TAIL_CHANNEL_CAPACITY", Some("42")),
            ],
            || {
                let config = LogBufferConfig::from_env().expect("from_env");
                assert_eq!(config.capacity, 777);
                assert_eq!(config.live_tail_channel_capacity, 42);
            },
        );
    }

    // seam-#3: a file sets TWO fields, env sets only ONE — the file's other
    // field must survive `.from_path().from_env()`.
    #[test]
    fn seam_3_from_path_then_from_env_preserves_files_untouched_field() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("log-buffer.toml");
        std::fs::write(&path, "capacity = 4096\nlive_tail_channel_capacity = 128\n")
            .expect("write toml");
        temp_env::with_vars([("LOG_BUFFER_CAPACITY", Some("8192"))], || {
            let config = LogBufferConfig::layered()
                .from_path(&path)
                .expect("from_path")
                .from_env()
                .expect("from_env")
                .build();
            assert_eq!(config.capacity, 8192, "env wins the field it sets");
            assert_eq!(
                config.live_tail_channel_capacity, 128,
                "the file's field must survive"
            );
        });
    }

    // full stack: defaults < file < env < with_*.
    #[test]
    fn full_stack_defaults_file_env_with_override_each_field() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("log-buffer.toml");
        std::fs::write(&path, "live_tail_channel_capacity = 64\n").expect("write toml");
        temp_env::with_vars([("LOG_BUFFER_CAPACITY", Some("8192"))], || {
            let config = LogBufferConfig::layered()
                .from_path(&path)
                .expect("from_path")
                .from_env()
                .expect("from_env")
                .with_live_tail_channel_capacity(999)
                .build();
            assert_eq!(config.capacity, 8192, "env layer");
            assert_eq!(config.live_tail_channel_capacity, 999, "with_* layer wins");
        });
    }

    // underlay never clobbers an already-set field; it DOES fill an unset one.
    #[test]
    fn underlay_path_fills_only_unset_fields() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("log-buffer.toml");
        std::fs::write(
            &path,
            "capacity = 1024\nlive_tail_channel_capacity = 1024\n",
        )
        .expect("write toml");
        let config = LogBufferConfig::layered()
            .with_capacity(64)
            .underlay_path(&path)
            .expect("underlay_path")
            .build();
        assert_eq!(
            config.capacity, 64,
            "already set by with_*; the file's value is dropped"
        );
        assert_eq!(
            config.live_tail_channel_capacity, 1024,
            "unset before underlay; the file fills it"
        );
    }

    #[test]
    fn underlay_env_fills_only_unset_fields() {
        temp_env::with_vars(
            [
                ("LOG_BUFFER_CAPACITY", Some("1024")),
                ("LOG_BUFFER_LIVE_TAIL_CHANNEL_CAPACITY", Some("2048")),
            ],
            || {
                let config = LogBufferConfig::layered()
                    .with_capacity(64)
                    .underlay_env()
                    .expect("underlay_env")
                    .build();
                assert_eq!(config.capacity, 64, "already set; env's value is dropped");
                assert_eq!(
                    config.live_tail_channel_capacity, 2048,
                    "unset before underlay_env; env fills it"
                );
            },
        );
    }

    #[test]
    fn config_round_trips_through_serde() {
        let built = LogBufferConfig::layered()
            .with_capacity(2048)
            .with_live_tail_channel_capacity(512)
            .build();
        let json = serde_json::to_string(&built).expect("serialize");
        let from_json: LogBufferConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(from_json, built);
    }
}
