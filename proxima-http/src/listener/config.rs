//! `HttpListenerConfig` — the runtime config + fluent builder surface (P4:
//! first-class conflaguration AND first-class fluent builder, BOTH).
//!
//! Mirrors `proxima_telemetry::log_buffer`'s `LogBufferConfig`: `#[derive(Builder,
//! Deserialize, Serialize, Settings)]` + [`Validate`], an
//! `HttpListenerLayerBuilder` with call-order precedence, and a typed env
//! surface. The defaults come FROM the `sized` consts that `build.rs`
//! generates from `proxima-listeners-http.toml`, so the runtime default
//! follows the compile-time / no_std+no_alloc floor — there is no double
//! source of truth.
//!
//! # What this reaches today — read before using
//!
//! Two config surfaces exist for [`crate::listener::HttpListenProtocol`] and
//! they are not equally wired:
//!
//! - the per-listener JSON `spec` passed to `ListenProtocol::serve`, which
//!   overrides one listener registration at runtime (`drain_timeout_ms`,
//!   `quiesce_status`, `quiesce_retry_after`, `proxy_protocol`, `name`);
//! - `HttpListenerConfig`, this type, which supplies the values a spec falls
//!   back to when it omits a key.
//!
//! The second is narrower than it looks. `HttpListenProtocol`'s serve path
//! reads `HttpListenerConfig::default()` — and nothing else. `default()` is
//! `builder().build()`, which resolves to the `sized` consts; it never
//! consults env or a file. So an `HttpListenerConfig` you construct through
//! the layered builder below — however you source it — has no way to reach a
//! running listener: there is no `HttpListenProtocol::from_config` and no
//! process-wide install point. `HTTP_LISTENER_*` env vars are read by
//! [`HttpListenerConfig::from_env`] and are honored by every test in this
//! file, but they do not change a live listener's behavior.
//!
//! What that leaves working, and genuinely useful:
//!
//! - the defaults a live listener falls back to, which come from the `sized`
//!   floor and are therefore tunable at BUILD time via
//!   `PROXIMA_LISTENERS_HTTP_*` (see `build.rs`);
//! - per-listener runtime overrides, via the JSON `spec`;
//! - this type as a standalone, loadable, validated config value — which a
//!   caller can load, validate, and then translate into the `spec` it passes
//!   to `serve`.
//!
//! Closing the gap means an injection seam on the protocol, which is an API
//! change, not a docs one. Until then this doc states the reach precisely
//! rather than implying more.
//!
//! Tier: std (conflaguration, fs, env). Once this crate lifts off std, the
//! `sized` consts become the no_std+no_alloc floor's only knob and this
//! layer becomes the std runtime override on top, unchanged.
//!
//! `HttpListenerLayerBuilder` supports call-order precedence:
//!
//! - **Operator config wins**: put `.with_*` BEFORE `.from_path` / `.from_env`.
//! - **Code overrides win**: put `.with_*` AFTER `.from_path` / `.from_env`.

use std::path::Path;

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

fn default_name() -> String {
    crate::listener::sized::LISTENER_NAME_DEFAULT.to_string()
}

fn default_drain_timeout_ms() -> u64 {
    crate::listener::sized::LISTENER_DRAIN_TIMEOUT_MS_DEFAULT
}

fn default_quiesce_status() -> u16 {
    crate::listener::sized::LISTENER_QUIESCE_STATUS_DEFAULT
}

fn default_quiesce_retry_after() -> String {
    crate::listener::sized::LISTENER_QUIESCE_RETRY_AFTER_DEFAULT.to_string()
}

fn default_proxy_protocol_enabled() -> bool {
    crate::listener::sized::LISTENER_PROXY_PROTOCOL_ENABLED_DEFAULT
}

/// Runtime configuration for [`crate::listener::HttpListenProtocol`]. One built
/// `HttpListenerConfig` == one serialisable config == one HTTP-listener
/// default policy.
///
/// Read the [module docs](crate::listener) first for what this currently reaches: a
/// config built here does not yet have an injection point on the protocol.
///
/// # Two surfaces, both first-class
///
/// | source | how |
/// |---|---|
/// | code, fluently | [`HttpListenerConfig::builder`] |
/// | code, layered | [`HttpListenerConfig::layered`] |
/// | a file | [`HttpListenerLayerBuilder::from_path`] |
/// | the environment | [`HttpListenerConfig::from_env`] |
///
/// `builder()` is the plain fluent surface: set fields, `build()`.
/// `layered()` is the same thing plus config sources, and it is the one to
/// reach for when values come from more than one place.
///
/// # Layering, and why call order is the API
///
/// Every source contributes only the fields it actually specifies, merged
/// onto what came before. A field nobody sets keeps its default. There are
/// two kinds of source, and the difference is the whole design:
///
/// - **override** — `.from_path`, `.from_env`, and every `.with_*`. Wins for
///   each field it specifies, at the position it is called.
/// - **underlay** — `.underlay_path`, `.underlay_env`. Fills only fields
///   still unset; never clobbers.
///
/// So precedence is not a fixed policy you have to memorize — it is where
/// you put the call:
///
/// - operator config should win → `.with_*` BEFORE `.from_path` / `.from_env`
/// - code should win → `.with_*` AFTER `.from_path` / `.from_env`
///
/// # Building one
///
/// Compiled and run by `cargo test`.
///
/// ```
/// use conflaguration::Validate;
/// use proxima_http::listener::HttpListenerConfig;
///
/// // ── fluent ────────────────────────────────────────────────────────────
/// let built = HttpListenerConfig::builder()
///     .drain_timeout_ms(5_000)
///     .quiesce_status(429)
///     .build();
/// assert_eq!(built.drain_timeout_ms, 5_000);
/// assert_eq!(built.name, "http", "untouched, so the sized default");
/// assert!(built.validate().is_ok());
///
/// // validation is a separate step and has real opinions.
/// let nonsense = HttpListenerConfig::builder().quiesce_status(0).build();
/// assert!(nonsense.validate().is_err(), "0 is not an HTTP status");
///
/// // ── layered: a file, then code on top ─────────────────────────────────
/// let dir = tempfile::TempDir::new().unwrap();
/// let path = dir.path().join("listener.toml");
/// std::fs::write(&path, "drain_timeout_ms = 9000\nname = \"edge\"\n").unwrap();
///
/// let config = HttpListenerConfig::layered()
///     .from_path(&path).unwrap()
///     .with_drain_timeout_ms(250)   // AFTER the file, so code wins
///     .build();
/// assert_eq!(config.drain_timeout_ms, 250, "the .with_* came later");
/// assert_eq!(config.name, "edge", "only the file set this");
/// assert_eq!(config.quiesce_status, 503, "nobody set this; sized default");
///
/// // ── underlay: the file fills gaps but never overwrites ────────────────
/// let config = HttpListenerConfig::layered()
///     .with_drain_timeout_ms(250)
///     .underlay_path(&path).unwrap()
///     .build();
/// assert_eq!(config.drain_timeout_ms, 250, "already set; the file is ignored");
/// assert_eq!(config.name, "edge", "was unset, so the file fills it");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "HTTP_LISTENER")]
#[builder(derive(Clone, Debug))]
pub struct HttpListenerConfig {
    /// Fallback telemetry label for a listener whose spec doesn't set
    /// `name`. Defaults from `proxima-listeners-http.toml`'s `[listener]
    /// name` (the `sized` floor).
    #[setting(default = "http")]
    #[serde(default = "default_name")]
    #[builder(default = default_name())]
    pub name: String,

    /// Fallback in-flight-drain timeout (ms) for a listener whose spec
    /// doesn't set `drain_timeout_ms`. Defaults from
    /// `proxima-listeners-http.toml`'s `[listener] drain_timeout_ms` (the
    /// `sized` floor).
    #[setting(default = 30_000)]
    #[serde(default = "default_drain_timeout_ms")]
    #[builder(default = default_drain_timeout_ms())]
    pub drain_timeout_ms: u64,

    /// Fallback HTTP status returned while quiescing, for a listener whose
    /// spec doesn't set `quiesce_status`. Defaults from
    /// `proxima-listeners-http.toml`'s `[listener] quiesce_status` (the
    /// `sized` floor).
    #[setting(default = 503)]
    #[serde(default = "default_quiesce_status")]
    #[builder(default = default_quiesce_status())]
    pub quiesce_status: u16,

    /// Fallback `Retry-After` header value (seconds) returned while
    /// quiescing, for a listener whose spec doesn't set
    /// `quiesce_retry_after`. Defaults from
    /// `proxima-listeners-http.toml`'s `[listener] quiesce_retry_after`
    /// (the `sized` floor).
    #[setting(default = "5")]
    #[serde(default = "default_quiesce_retry_after")]
    #[builder(default = default_quiesce_retry_after())]
    pub quiesce_retry_after: String,

    /// Fallback PROXY-protocol requirement for a listener whose spec
    /// doesn't set `proxy_protocol`. Defaults from
    /// `proxima-listeners-http.toml`'s `[listener]
    /// proxy_protocol_enabled` (the `sized` floor).
    #[setting(default = false)]
    #[serde(default = "default_proxy_protocol_enabled")]
    #[builder(default = default_proxy_protocol_enabled())]
    pub proxy_protocol_enabled: bool,
}

impl Default for HttpListenerConfig {
    fn default() -> Self {
        HttpListenerConfig::builder().build()
    }
}

impl Validate for HttpListenerConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.name.is_empty() {
            errors.push(ValidationMessage::new("name", "must not be empty"));
        }
        if !(100..=599).contains(&self.quiesce_status) {
            errors.push(ValidationMessage::new(
                "quiesce_status",
                "must be a valid HTTP status code (100-599)",
            ));
        }
        if self.quiesce_retry_after.is_empty() {
            errors.push(ValidationMessage::new(
                "quiesce_retry_after",
                "must not be empty",
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl HttpListenerConfig {
    /// Start a layered builder from the `sized`-seeded defaults.
    #[must_use]
    pub fn layered() -> HttpListenerLayerBuilder {
        HttpListenerLayerBuilder {
            inner: HttpListenerConfig::default(),
            name_set: false,
            drain_timeout_ms_set: false,
            quiesce_status_set: false,
            quiesce_retry_after_set: false,
            proxy_protocol_enabled_set: false,
        }
    }
}

/// Partial view of [`HttpListenerConfig`] used by `.from_path`/`.underlay_path`
/// — only fields actually present in the file are applied, so a file setting
/// one field never clobbers the other with a re-resolved default.
#[derive(Debug, Default, Deserialize)]
struct HttpListenerConfigPartial {
    name: Option<String>,
    drain_timeout_ms: Option<u64>,
    quiesce_status: Option<u16>,
    quiesce_retry_after: Option<String>,
    proxy_protocol_enabled: Option<bool>,
}

/// Fluent builder for [`HttpListenerConfig`]. Every source (`.from_path`,
/// `.from_env`, `.underlay_path`, `.underlay_env`, `.with_*`) contributes only
/// the fields it actually specifies, merged onto the accumulated config — a
/// field a source doesn't touch falls through to whatever prior layers set.
/// `.from_path`/`.from_env` override (last writer wins per field);
/// `.underlay_path`/`.underlay_env` fill only fields still unset; `.with_*`
/// always acts as an override at its call position.
pub struct HttpListenerLayerBuilder {
    inner: HttpListenerConfig,
    name_set: bool,
    drain_timeout_ms_set: bool,
    quiesce_status_set: bool,
    quiesce_retry_after_set: bool,
    proxy_protocol_enabled_set: bool,
}

impl HttpListenerLayerBuilder {
    /// Merge a TOML/JSON file's fields onto the accumulated config; the file
    /// wins for every field it specifies.
    pub fn from_path<P: AsRef<Path>>(mut self, path: P) -> Result<Self, conflaguration::Error> {
        let partial: HttpListenerConfigPartial = conflaguration::from_file(path.as_ref())?;
        if let Some(name) = partial.name {
            self.inner.name = name;
            self.name_set = true;
        }
        if let Some(drain_timeout_ms) = partial.drain_timeout_ms {
            self.inner.drain_timeout_ms = drain_timeout_ms;
            self.drain_timeout_ms_set = true;
        }
        if let Some(quiesce_status) = partial.quiesce_status {
            self.inner.quiesce_status = quiesce_status;
            self.quiesce_status_set = true;
        }
        if let Some(quiesce_retry_after) = partial.quiesce_retry_after {
            self.inner.quiesce_retry_after = quiesce_retry_after;
            self.quiesce_retry_after_set = true;
        }
        if let Some(proxy_protocol_enabled) = partial.proxy_protocol_enabled {
            self.inner.proxy_protocol_enabled = proxy_protocol_enabled;
            self.proxy_protocol_enabled_set = true;
        }
        Ok(self)
    }

    /// Fill any still-unset fields from a TOML/JSON file; already-set fields
    /// are left untouched.
    pub fn underlay_path<P: AsRef<Path>>(mut self, path: P) -> Result<Self, conflaguration::Error> {
        let partial: HttpListenerConfigPartial = conflaguration::from_file(path.as_ref())?;
        if !self.name_set
            && let Some(name) = partial.name
        {
            self.inner.name = name;
            self.name_set = true;
        }
        if !self.drain_timeout_ms_set
            && let Some(drain_timeout_ms) = partial.drain_timeout_ms
        {
            self.inner.drain_timeout_ms = drain_timeout_ms;
            self.drain_timeout_ms_set = true;
        }
        if !self.quiesce_status_set
            && let Some(quiesce_status) = partial.quiesce_status
        {
            self.inner.quiesce_status = quiesce_status;
            self.quiesce_status_set = true;
        }
        if !self.quiesce_retry_after_set
            && let Some(quiesce_retry_after) = partial.quiesce_retry_after
        {
            self.inner.quiesce_retry_after = quiesce_retry_after;
            self.quiesce_retry_after_set = true;
        }
        if !self.proxy_protocol_enabled_set
            && let Some(proxy_protocol_enabled) = partial.proxy_protocol_enabled
        {
            self.inner.proxy_protocol_enabled = proxy_protocol_enabled;
            self.proxy_protocol_enabled_set = true;
        }
        Ok(self)
    }

    /// Merge `HTTP_LISTENER_*` env-set fields onto the accumulated config;
    /// env wins for every field it sets. Unset env vars leave the current
    /// value untouched.
    pub fn from_env(mut self) -> Result<Self, conflaguration::Error> {
        let resolved = HttpListenerConfig::from_env()?;
        if env_is_set("HTTP_LISTENER_NAME") {
            self.inner.name = resolved.name;
            self.name_set = true;
        }
        if env_is_set("HTTP_LISTENER_DRAIN_TIMEOUT_MS") {
            self.inner.drain_timeout_ms = resolved.drain_timeout_ms;
            self.drain_timeout_ms_set = true;
        }
        if env_is_set("HTTP_LISTENER_QUIESCE_STATUS") {
            self.inner.quiesce_status = resolved.quiesce_status;
            self.quiesce_status_set = true;
        }
        if env_is_set("HTTP_LISTENER_QUIESCE_RETRY_AFTER") {
            self.inner.quiesce_retry_after = resolved.quiesce_retry_after;
            self.quiesce_retry_after_set = true;
        }
        if env_is_set("HTTP_LISTENER_PROXY_PROTOCOL_ENABLED") {
            self.inner.proxy_protocol_enabled = resolved.proxy_protocol_enabled;
            self.proxy_protocol_enabled_set = true;
        }
        Ok(self)
    }

    /// Fill any still-unset fields from `HTTP_LISTENER_*` env vars;
    /// already-set fields are left untouched even if the matching env var is
    /// set.
    pub fn underlay_env(mut self) -> Result<Self, conflaguration::Error> {
        let resolved = HttpListenerConfig::from_env()?;
        if !self.name_set && env_is_set("HTTP_LISTENER_NAME") {
            self.inner.name = resolved.name;
            self.name_set = true;
        }
        if !self.drain_timeout_ms_set && env_is_set("HTTP_LISTENER_DRAIN_TIMEOUT_MS") {
            self.inner.drain_timeout_ms = resolved.drain_timeout_ms;
            self.drain_timeout_ms_set = true;
        }
        if !self.quiesce_status_set && env_is_set("HTTP_LISTENER_QUIESCE_STATUS") {
            self.inner.quiesce_status = resolved.quiesce_status;
            self.quiesce_status_set = true;
        }
        if !self.quiesce_retry_after_set && env_is_set("HTTP_LISTENER_QUIESCE_RETRY_AFTER") {
            self.inner.quiesce_retry_after = resolved.quiesce_retry_after;
            self.quiesce_retry_after_set = true;
        }
        if !self.proxy_protocol_enabled_set && env_is_set("HTTP_LISTENER_PROXY_PROTOCOL_ENABLED") {
            self.inner.proxy_protocol_enabled = resolved.proxy_protocol_enabled;
            self.proxy_protocol_enabled_set = true;
        }
        Ok(self)
    }

    /// Set the fallback telemetry label.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.inner.name = name.into();
        self.name_set = true;
        self
    }

    /// Set the fallback in-flight-drain timeout (ms).
    #[must_use]
    pub fn with_drain_timeout_ms(mut self, drain_timeout_ms: u64) -> Self {
        self.inner.drain_timeout_ms = drain_timeout_ms;
        self.drain_timeout_ms_set = true;
        self
    }

    /// Set the fallback quiesce HTTP status.
    #[must_use]
    pub fn with_quiesce_status(mut self, quiesce_status: u16) -> Self {
        self.inner.quiesce_status = quiesce_status;
        self.quiesce_status_set = true;
        self
    }

    /// Set the fallback `Retry-After` header value.
    #[must_use]
    pub fn with_quiesce_retry_after(mut self, quiesce_retry_after: impl Into<String>) -> Self {
        self.inner.quiesce_retry_after = quiesce_retry_after.into();
        self.quiesce_retry_after_set = true;
        self
    }

    /// Set the fallback PROXY-protocol requirement.
    #[must_use]
    pub fn with_proxy_protocol_enabled(mut self, enabled: bool) -> Self {
        self.inner.proxy_protocol_enabled = enabled;
        self.proxy_protocol_enabled_set = true;
        self
    }

    /// The built immutable config.
    #[must_use]
    pub fn build(self) -> HttpListenerConfig {
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
        let config = HttpListenerConfig::default();
        assert_eq!(config.name, crate::listener::sized::LISTENER_NAME_DEFAULT);
        assert_eq!(
            config.drain_timeout_ms,
            crate::listener::sized::LISTENER_DRAIN_TIMEOUT_MS_DEFAULT
        );
        assert_eq!(
            config.quiesce_status,
            crate::listener::sized::LISTENER_QUIESCE_STATUS_DEFAULT
        );
        assert_eq!(
            config.quiesce_retry_after,
            crate::listener::sized::LISTENER_QUIESCE_RETRY_AFTER_DEFAULT
        );
        assert_eq!(
            config.proxy_protocol_enabled,
            crate::listener::sized::LISTENER_PROXY_PROTOCOL_ENABLED_DEFAULT
        );
        // the env-overlay (from_env, no vars set) must agree with the const too.
        temp_env::with_vars::<&str, &str, _, _>([], || {
            let from_env = HttpListenerConfig::from_env().expect("from_env");
            assert_eq!(from_env.name, crate::listener::sized::LISTENER_NAME_DEFAULT);
            assert_eq!(
                from_env.drain_timeout_ms,
                crate::listener::sized::LISTENER_DRAIN_TIMEOUT_MS_DEFAULT
            );
            assert_eq!(
                from_env.quiesce_status,
                crate::listener::sized::LISTENER_QUIESCE_STATUS_DEFAULT
            );
        });
    }

    // the runtime default equals the old hand-rolled magic constants
    // (drain_timeout_ms = 30_000, quiesce_status = 503,
    // quiesce_retry_after = "5", proxy_protocol_enabled = false,
    // name fallback = "http") — this refactor is behavior-preserving.
    #[test]
    fn defaults_match_the_former_magic_constants() {
        let config = HttpListenerConfig::default();
        assert_eq!(config.name, "http");
        assert_eq!(config.drain_timeout_ms, 30_000);
        assert_eq!(config.quiesce_status, 503);
        assert_eq!(config.quiesce_retry_after, "5");
        assert!(!config.proxy_protocol_enabled);
    }

    #[test]
    fn default_config_validates() {
        let config = HttpListenerConfig::default();
        assert!(config.validate().is_ok(), "default config should validate");
    }

    #[test]
    fn invalid_quiesce_status_rejected() {
        let config = HttpListenerConfig::builder().quiesce_status(0).build();
        let error = config.validate().expect_err("validate must reject 0");
        assert!(format!("{error:?}").contains("quiesce_status"));
    }

    #[test]
    fn empty_name_rejected() {
        let config = HttpListenerConfig::builder().name(String::new()).build();
        let error = config.validate().expect_err("validate must reject empty");
        assert!(format!("{error:?}").contains("name"));
    }

    #[test]
    fn builder_starts_at_default() {
        let from_layered = HttpListenerConfig::layered().build();
        let from_default = HttpListenerConfig::default();
        assert_eq!(from_layered, from_default);
    }

    #[test]
    fn with_overrides_default() {
        let config = HttpListenerConfig::layered()
            .with_drain_timeout_ms(5000)
            .build();
        assert_eq!(config.drain_timeout_ms, 5000);
        assert_eq!(HttpListenerConfig::default().drain_timeout_ms, 30_000);
    }

    #[test]
    fn from_path_overrides_default() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("listeners-http.toml");
        std::fs::write(&path, "drain_timeout_ms = 9000\n").expect("write toml");
        let config = HttpListenerConfig::layered()
            .from_path(&path)
            .expect("from_path")
            .build();
        assert_eq!(config.drain_timeout_ms, 9000);
        assert_eq!(config.quiesce_status, 503, "untouched field");
    }

    // env override demonstration: HTTP_LISTENER_* vars flow through
    // from_env() and are picked up in the built config.
    #[test]
    fn env_override_demonstration() {
        temp_env::with_vars(
            [
                ("HTTP_LISTENER_DRAIN_TIMEOUT_MS", Some("777")),
                ("HTTP_LISTENER_QUIESCE_STATUS", Some("429")),
            ],
            || {
                let config = HttpListenerConfig::from_env().expect("from_env");
                assert_eq!(config.drain_timeout_ms, 777);
                assert_eq!(config.quiesce_status, 429);
            },
        );
    }

    #[test]
    fn from_env_overlays_via_conflaguration() {
        temp_env::with_vars([("HTTP_LISTENER_DRAIN_TIMEOUT_MS", Some("2048"))], || {
            let config = HttpListenerConfig::layered()
                .from_env()
                .expect("from_env")
                .build();
            assert_eq!(config.drain_timeout_ms, 2048);
            assert_eq!(config.quiesce_status, 503, "untouched field");
        });
    }

    // underlay never clobbers an already-set field; it DOES fill an unset one.
    #[test]
    fn underlay_path_fills_only_unset_fields() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("listeners-http.toml");
        std::fs::write(&path, "drain_timeout_ms = 1000\nquiesce_status = 429\n")
            .expect("write toml");
        let config = HttpListenerConfig::layered()
            .with_drain_timeout_ms(64)
            .underlay_path(&path)
            .expect("underlay_path")
            .build();
        assert_eq!(
            config.drain_timeout_ms, 64,
            "already set by with_*; the file's value is dropped"
        );
        assert_eq!(
            config.quiesce_status, 429,
            "unset before underlay; the file fills it"
        );
    }

    #[test]
    fn config_round_trips_through_serde() {
        let built = HttpListenerConfig::layered()
            .with_drain_timeout_ms(2048)
            .with_quiesce_status(429)
            .build();
        let json = serde_json::to_string(&built).expect("serialize");
        let from_json: HttpListenerConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(from_json, built);
    }
}
