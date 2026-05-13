//! `ListenTuningConfig` — the runtime config + fluent builder surface (P4:
//! first-class conflaguration AND first-class fluent builder, BOTH).
//!
//! Mirrors `proxima_telemetry::log_buffer`'s `LogBufferConfig`: `#[derive(Builder,
//! Deserialize, Serialize, Settings)]` + [`Validate`], a
//! `ListenTuningLayerBuilder` with call-order precedence, and a typed env
//! surface. The `backlog` / `drain_timeout_ms` defaults come FROM the
//! `sized` consts that `build.rs` generates from `proxima-listen.toml`, so
//! the runtime default follows the compile-time / no_std+no_alloc floor —
//! there is no double source of truth.
//!
//! `http_handler_spread` is a runtime-only policy toggle (not a build-time
//! size), preserved with its historical env var name
//! (`PROXIMA_HTTP_HANDLER_SPREAD`) and its historical "only the literal `1`
//! is truthy" parse semantics via `#[setting(resolve_with = ...)]`, so this
//! refactor stays behavior-preserving for existing deployments.
//!
//! Tier: std (conflaguration, fs, env). Once this crate lifts off std, the
//! `sized` consts become the no_std+no_alloc floor's only knob and this
//! layer becomes the std runtime override on top, unchanged.
//!
//! `ListenTuningLayerBuilder` supports call-order precedence:
//!
//! - **Operator config wins**: put `.with_*` BEFORE `.from_path` / `.from_env`.
//! - **Code overrides win**: put `.with_*` AFTER `.from_path` / `.from_env`.

use std::convert::Infallible;
use std::path::Path;

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

fn default_backlog() -> i32 {
    crate::sized::LISTENER_BACKLOG_DEFAULT
}

fn default_drain_timeout_ms() -> u64 {
    crate::sized::LISTENER_DRAIN_TIMEOUT_MS_DEFAULT
}

/// Mirrors the historical `std::env::var("PROXIMA_HTTP_HANDLER_SPREAD")
/// .is_ok_and(|v| v == "1")` check: only the literal `"1"` is truthy, and an
/// unset or otherwise-valued env var resolves to `false` rather than an
/// error.
fn parse_legacy_spread_flag(raw: &str) -> Result<bool, Infallible> {
    Ok(raw == "1")
}

/// Runtime configuration for [`crate::handle::Listener::run_with_runtime`]
/// and [`crate::handle::bind_reuseport_listener_with_options`]. One built
/// `ListenTuningConfig` == one serialisable config == one listener-tuning
/// policy.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_LISTEN")]
#[builder(derive(Clone, Debug))]
pub struct ListenTuningConfig {
    /// TCP `listen()` backlog (SYN accept queue depth) for reuseport-bound
    /// sockets. Defaults from `proxima-listen.toml`'s `[listener] backlog`
    /// (the `sized` floor).
    #[setting(default = 1024)]
    #[serde(default = "default_backlog")]
    #[builder(default = default_backlog())]
    pub backlog: i32,

    /// Graceful-drain timeout (ms) used by `ShutdownPolicy::drain_30s()`.
    /// Defaults from `proxima-listen.toml`'s `[listener] drain_timeout_ms`
    /// (the `sized` floor).
    #[setting(default = 30_000)]
    #[serde(default = "default_drain_timeout_ms")]
    #[builder(default = default_drain_timeout_ms())]
    pub drain_timeout_ms: u64,

    /// Spread accepted HTTP connections to peer worker cores instead of
    /// handling them inline on the accepting core. Read from the historical
    /// `PROXIMA_HTTP_HANDLER_SPREAD` env var (only `"1"` is truthy); a
    /// platform default (macOS/BSD) is applied separately by the caller via
    /// `cfg!(...)`, not by this field.
    #[setting(
        envs = "PROXIMA_HTTP_HANDLER_SPREAD",
        r#override,
        resolve_with = "parse_legacy_spread_flag",
        default = false
    )]
    #[serde(default)]
    #[builder(default)]
    pub http_handler_spread: bool,
}

impl Default for ListenTuningConfig {
    fn default() -> Self {
        ListenTuningConfig::builder().build()
    }
}

impl Validate for ListenTuningConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.backlog <= 0 {
            errors.push(ValidationMessage::new("backlog", "must be > 0"));
        }
        if self.drain_timeout_ms == 0 {
            errors.push(ValidationMessage::new("drain_timeout_ms", "must be > 0"));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl ListenTuningConfig {
    /// Start a layered builder from the `sized`-seeded defaults.
    #[must_use]
    pub fn layered() -> ListenTuningLayerBuilder {
        ListenTuningLayerBuilder {
            inner: ListenTuningConfig::default(),
            backlog_set: false,
            drain_timeout_ms_set: false,
            http_handler_spread_set: false,
        }
    }
}

/// Partial view of [`ListenTuningConfig`] used by `.from_path`/`.underlay_path`
/// — only fields actually present in the file are applied, so a file setting
/// one field never clobbers the other with a re-resolved default.
#[derive(Debug, Default, Deserialize)]
struct ListenTuningConfigPartial {
    backlog: Option<i32>,
    drain_timeout_ms: Option<u64>,
    http_handler_spread: Option<bool>,
}

/// Fluent builder for [`ListenTuningConfig`]. Every source (`.from_path`,
/// `.from_env`, `.underlay_path`, `.underlay_env`, `.with_*`) contributes only
/// the fields it actually specifies, merged onto the accumulated config — a
/// field a source doesn't touch falls through to whatever prior layers set.
/// `.from_path`/`.from_env` override (last writer wins per field);
/// `.underlay_path`/`.underlay_env` fill only fields still unset; `.with_*`
/// always acts as an override at its call position.
pub struct ListenTuningLayerBuilder {
    inner: ListenTuningConfig,
    backlog_set: bool,
    drain_timeout_ms_set: bool,
    http_handler_spread_set: bool,
}

impl ListenTuningLayerBuilder {
    /// Merge a TOML/JSON file's fields onto the accumulated config; the file
    /// wins for every field it specifies.
    pub fn from_path<P: AsRef<Path>>(mut self, path: P) -> Result<Self, conflaguration::Error> {
        let partial: ListenTuningConfigPartial = conflaguration::from_file(path.as_ref())?;
        if let Some(backlog) = partial.backlog {
            self.inner.backlog = backlog;
            self.backlog_set = true;
        }
        if let Some(drain_timeout_ms) = partial.drain_timeout_ms {
            self.inner.drain_timeout_ms = drain_timeout_ms;
            self.drain_timeout_ms_set = true;
        }
        if let Some(http_handler_spread) = partial.http_handler_spread {
            self.inner.http_handler_spread = http_handler_spread;
            self.http_handler_spread_set = true;
        }
        Ok(self)
    }

    /// Fill any still-unset fields from a TOML/JSON file; already-set fields
    /// are left untouched.
    pub fn underlay_path<P: AsRef<Path>>(mut self, path: P) -> Result<Self, conflaguration::Error> {
        let partial: ListenTuningConfigPartial = conflaguration::from_file(path.as_ref())?;
        if !self.backlog_set
            && let Some(backlog) = partial.backlog
        {
            self.inner.backlog = backlog;
            self.backlog_set = true;
        }
        if !self.drain_timeout_ms_set
            && let Some(drain_timeout_ms) = partial.drain_timeout_ms
        {
            self.inner.drain_timeout_ms = drain_timeout_ms;
            self.drain_timeout_ms_set = true;
        }
        if !self.http_handler_spread_set
            && let Some(http_handler_spread) = partial.http_handler_spread
        {
            self.inner.http_handler_spread = http_handler_spread;
            self.http_handler_spread_set = true;
        }
        Ok(self)
    }

    /// Merge env-set fields onto the accumulated config; env wins for every
    /// field it sets. Unset env vars leave the current value untouched.
    pub fn from_env(mut self) -> Result<Self, conflaguration::Error> {
        let resolved = ListenTuningConfig::from_env()?;
        if env_is_set("PROXIMA_LISTEN_BACKLOG") {
            self.inner.backlog = resolved.backlog;
            self.backlog_set = true;
        }
        if env_is_set("PROXIMA_LISTEN_DRAIN_TIMEOUT_MS") {
            self.inner.drain_timeout_ms = resolved.drain_timeout_ms;
            self.drain_timeout_ms_set = true;
        }
        if env_is_set("PROXIMA_HTTP_HANDLER_SPREAD") {
            self.inner.http_handler_spread = resolved.http_handler_spread;
            self.http_handler_spread_set = true;
        }
        Ok(self)
    }

    /// Fill any still-unset fields from env vars; already-set fields are
    /// left untouched even if the matching env var is set.
    pub fn underlay_env(mut self) -> Result<Self, conflaguration::Error> {
        let resolved = ListenTuningConfig::from_env()?;
        if !self.backlog_set && env_is_set("PROXIMA_LISTEN_BACKLOG") {
            self.inner.backlog = resolved.backlog;
            self.backlog_set = true;
        }
        if !self.drain_timeout_ms_set && env_is_set("PROXIMA_LISTEN_DRAIN_TIMEOUT_MS") {
            self.inner.drain_timeout_ms = resolved.drain_timeout_ms;
            self.drain_timeout_ms_set = true;
        }
        if !self.http_handler_spread_set && env_is_set("PROXIMA_HTTP_HANDLER_SPREAD") {
            self.inner.http_handler_spread = resolved.http_handler_spread;
            self.http_handler_spread_set = true;
        }
        Ok(self)
    }

    /// Set the TCP `listen()` backlog.
    #[must_use]
    pub fn with_backlog(mut self, backlog: i32) -> Self {
        self.inner.backlog = backlog;
        self.backlog_set = true;
        self
    }

    /// Set the graceful-drain timeout (ms).
    #[must_use]
    pub fn with_drain_timeout_ms(mut self, drain_timeout_ms: u64) -> Self {
        self.inner.drain_timeout_ms = drain_timeout_ms;
        self.drain_timeout_ms_set = true;
        self
    }

    /// Set the HTTP-handler-spread policy toggle.
    #[must_use]
    pub fn with_http_handler_spread(mut self, spread: bool) -> Self {
        self.inner.http_handler_spread = spread;
        self.http_handler_spread_set = true;
        self
    }

    /// The built immutable config.
    #[must_use]
    pub fn build(self) -> ListenTuningConfig {
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
        let config = ListenTuningConfig::default();
        assert_eq!(config.backlog, crate::sized::LISTENER_BACKLOG_DEFAULT);
        assert_eq!(
            config.drain_timeout_ms,
            crate::sized::LISTENER_DRAIN_TIMEOUT_MS_DEFAULT
        );
        // the env-overlay (from_env, no vars set) must agree with the const too.
        temp_env::with_vars::<&str, &str, _, _>([], || {
            let from_env = ListenTuningConfig::from_env().expect("from_env");
            assert_eq!(from_env.backlog, crate::sized::LISTENER_BACKLOG_DEFAULT);
            assert_eq!(
                from_env.drain_timeout_ms,
                crate::sized::LISTENER_DRAIN_TIMEOUT_MS_DEFAULT
            );
            assert!(!from_env.http_handler_spread);
        });
    }

    // the runtime default equals the old hand-rolled magic constants
    // (socket.listen(1024), ShutdownPolicy::drain_30s() == 30_000ms,
    // PROXIMA_HTTP_HANDLER_SPREAD unset == false) — this refactor is
    // behavior-preserving.
    #[test]
    fn defaults_match_the_former_magic_constants() {
        let config = ListenTuningConfig::default();
        assert_eq!(config.backlog, 1024);
        assert_eq!(config.drain_timeout_ms, 30_000);
        assert!(!config.http_handler_spread);
    }

    #[test]
    fn default_config_validates() {
        let config = ListenTuningConfig::default();
        assert!(config.validate().is_ok(), "default config should validate");
    }

    #[test]
    fn zero_backlog_rejected() {
        let config = ListenTuningConfig::builder().backlog(0).build();
        let error = config.validate().expect_err("validate must reject 0");
        assert!(format!("{error:?}").contains("backlog"));
    }

    #[test]
    fn zero_drain_timeout_rejected() {
        let config = ListenTuningConfig::builder().drain_timeout_ms(0).build();
        let error = config.validate().expect_err("validate must reject 0");
        assert!(format!("{error:?}").contains("drain_timeout_ms"));
    }

    #[test]
    fn builder_starts_at_default() {
        let from_layered = ListenTuningConfig::layered().build();
        let from_default = ListenTuningConfig::default();
        assert_eq!(from_layered, from_default);
    }

    #[test]
    fn with_overrides_default() {
        let config = ListenTuningConfig::layered().with_backlog(4096).build();
        assert_eq!(config.backlog, 4096);
        assert_eq!(ListenTuningConfig::default().backlog, 1024);
    }

    #[test]
    fn from_path_overrides_default() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("listen.toml");
        std::fs::write(&path, "backlog = 2048\n").expect("write toml");
        let config = ListenTuningConfig::layered()
            .from_path(&path)
            .expect("from_path")
            .build();
        assert_eq!(config.backlog, 2048);
        assert_eq!(config.drain_timeout_ms, 30_000, "untouched field");
    }

    // env override demonstration: PROXIMA_LISTEN_* and the historical
    // PROXIMA_HTTP_HANDLER_SPREAD both flow through from_env() and are
    // picked up in the built config.
    #[test]
    fn env_override_demonstration() {
        temp_env::with_vars(
            [
                ("PROXIMA_LISTEN_BACKLOG", Some("777")),
                ("PROXIMA_HTTP_HANDLER_SPREAD", Some("1")),
            ],
            || {
                let config = ListenTuningConfig::from_env().expect("from_env");
                assert_eq!(config.backlog, 777);
                assert!(config.http_handler_spread);
            },
        );
    }

    // only the literal "1" is truthy — matches the historical
    // `v == "1"` check exactly, unlike a generic bool parse.
    #[test]
    fn spread_flag_only_treats_literal_one_as_true() {
        temp_env::with_vars([("PROXIMA_HTTP_HANDLER_SPREAD", Some("true"))], || {
            let config = ListenTuningConfig::from_env().expect("from_env");
            assert!(
                !config.http_handler_spread,
                "\"true\" is not \"1\"; must stay false to match the old behavior"
            );
        });
    }

    #[test]
    fn from_env_overlays_via_conflaguration() {
        temp_env::with_vars([("PROXIMA_LISTEN_BACKLOG", Some("2048"))], || {
            let config = ListenTuningConfig::layered()
                .from_env()
                .expect("from_env")
                .build();
            assert_eq!(config.backlog, 2048);
            assert_eq!(config.drain_timeout_ms, 30_000, "untouched field");
        });
    }

    // underlay never clobbers an already-set field; it DOES fill an unset one.
    #[test]
    fn underlay_path_fills_only_unset_fields() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("listen.toml");
        std::fs::write(&path, "backlog = 1024\ndrain_timeout_ms = 5000\n").expect("write toml");
        let config = ListenTuningConfig::layered()
            .with_backlog(64)
            .underlay_path(&path)
            .expect("underlay_path")
            .build();
        assert_eq!(
            config.backlog, 64,
            "already set by with_*; the file's value is dropped"
        );
        assert_eq!(
            config.drain_timeout_ms, 5000,
            "unset before underlay; the file fills it"
        );
    }

    #[test]
    fn config_round_trips_through_serde() {
        let built = ListenTuningConfig::layered()
            .with_backlog(2048)
            .with_drain_timeout_ms(5000)
            .build();
        let json = serde_json::to_string(&built).expect("serialize");
        let from_json: ListenTuningConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(from_json, built);
    }
}
