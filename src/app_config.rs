//! `App`'s configuration surface. Replaces the old `App::new()` internals
//! that read `std::env::var("PROXIMA_RUNTIME_CORES")` directly — a
//! hand-rolled env read is not config, it's a global side-channel, and it's
//! why nine examples resorted to `unsafe { std::env::set_var(...) }` to
//! influence it.
//!
//! Mirrors `cassette_config.rs`'s house pattern: `#[derive(Builder,
//! Deserialize, Serialize, Settings)]` + [`Validate`], and a `layered()`
//! fluent loader with call-order precedence (defaults -> file -> env ->
//! explicit `.with_*` overrides). `RuntimeConfig` nests under [`AppConfig`]
//! rather than living as a flat bag, so more App-level config sections have
//! a home to nest under as they earn a surface.
//!
//! `App::new()` only resolves this when no runtime is already installed
//! ambiently — see `crate::runtime::installed_runtime`. A
//! `#[proxima::main(cores = N)]`-booted runtime always wins over
//! this config; `RuntimeConfig` is the fallback path for `App::new()` /
//! `AppBuilder::build()` calls made outside that macro (a custom entry
//! point, or a test).

use std::collections::BTreeSet;
use std::fmt;
use std::path::Path;
use std::str::FromStr;

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::runtime::RuntimeSelection;

use crate::error::ProximaError;

fn default_cores() -> usize {
    0
}

fn default_backend() -> RuntimeBackendSelection {
    RuntimeBackendSelection::Auto
}

/// Parse error for `RuntimeConfig` fields. conflaguration's `resolve_with`
/// plumbing demands a `std::error::Error` impl on the parser's error type —
/// mirrors `cassette_config::ParseError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError(String);

impl fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ParseError {}

/// Which runtime backend `App::new()`/`AppBuilder::build()` resolves to —
/// the config-shaped sibling of `App::builder().runtime(RuntimeSelection)`
/// (principle 4: both a fluent AND a config surface). `Auto` (the default)
/// keeps today's backward-compatible fallback (prime-first-if-linked, else
/// tokio — see `resolve_default_runtime_selection` in `src/app.rs`);
/// `Prime`/`Tokio` name the SAME two backends `RuntimeSelection::prime`/
/// `::tokio` and `#[proxima::main(runtime = "prime"|"tokio")]` do, so a
/// runtime chosen from TOML/env resolves through the identical vocabulary
/// as the fluent and macro surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RuntimeBackendSelection {
    /// No explicit choice — fall through to `resolve_default_runtime_selection`.
    Auto,
    Prime,
    Tokio,
}

impl FromStr for RuntimeBackendSelection {
    type Err = ParseError;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw {
            "auto" => Ok(Self::Auto),
            "prime" => Ok(Self::Prime),
            "tokio" => Ok(Self::Tokio),
            other => Err(ParseError(format!(
                "unknown runtime backend `{other}` (expected auto|prime|tokio)"
            ))),
        }
    }
}

impl RuntimeBackendSelection {
    /// Build the `RuntimeSelection` this config value resolves to, sized by
    /// `cores`. `Auto` returns `None` — the caller falls through to its own
    /// default-resolution path.
    ///
    /// # Errors
    /// `ProximaError::Config` when `Prime`/`Tokio` is selected but that
    /// backend's Cargo features are not linked, or if the selected
    /// backend's runtime fails to build.
    pub fn resolve(self, cores: usize) -> Result<Option<RuntimeSelection>, ProximaError> {
        match self {
            Self::Auto => Ok(None),
            Self::Prime => resolve_prime_selection(cores).map(Some),
            Self::Tokio => resolve_tokio_selection(cores).map(Some),
        }
    }
}

#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool",
    any(target_os = "linux", target_os = "macos")
))]
fn resolve_prime_selection(cores: usize) -> Result<RuntimeSelection, ProximaError> {
    RuntimeSelection::prime(cores)
}

#[cfg(not(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool",
    any(target_os = "linux", target_os = "macos")
)))]
fn resolve_prime_selection(_cores: usize) -> Result<RuntimeSelection, ProximaError> {
    Err(ProximaError::Config(
        "runtime config selected backend `prime`, but the prime runtime bundle is not linked \
         (enable `serve-prime` or the four `runtime-prime-*` features)"
            .into(),
    ))
}

#[cfg(feature = "runtime-tokio")]
fn resolve_tokio_selection(cores: usize) -> Result<RuntimeSelection, ProximaError> {
    RuntimeSelection::tokio(cores)
}

#[cfg(not(feature = "runtime-tokio"))]
fn resolve_tokio_selection(_cores: usize) -> Result<RuntimeSelection, ProximaError> {
    Err(ProximaError::Config(
        "runtime config selected backend `tokio`, but `runtime-tokio` is not linked".into(),
    ))
}

/// The App's runtime sizing + backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_RUNTIME")]
#[builder(derive(Clone, Debug))]
pub struct RuntimeConfig {
    /// Worker core count for the App's default runtime (whichever backend —
    /// prime or tokio — the build resolves to). `0` (the default) means
    /// "auto": resolved to the host's CPU count by
    /// [`resolved_cores`](Self::resolved_cores) at use time, not baked in
    /// here, so a config loaded once stays portable across hosts.
    /// `PROXIMA_RUNTIME_CORES` overrides via the env layer.
    #[setting(default = 0)]
    #[serde(default = "default_cores")]
    #[builder(default = default_cores())]
    pub cores: usize,

    /// Which runtime backend to use — `auto` (default, backward-compatible
    /// prime-first-if-linked fallback), `prime`, or `tokio`.
    /// `PROXIMA_RUNTIME_BACKEND` overrides via the env layer.
    #[setting(default_str = "auto", resolve_with = "parse_backend")]
    #[serde(default = "default_backend")]
    #[builder(default = default_backend())]
    pub backend: RuntimeBackendSelection,
}

fn parse_backend(raw: &str) -> Result<RuntimeBackendSelection, ParseError> {
    raw.parse()
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl Validate for RuntimeConfig {
    // `cores` has no invalid representation: `0` is the documented "auto"
    // sentinel, any other value is a literal worker count. `backend` is
    // parse-validated by `resolve_with`; nothing structural left to check.
    fn validate(&self) -> conflaguration::Result<()> {
        Ok(())
    }
}

impl RuntimeConfig {
    /// Resolve `cores` for actual use: `0` (auto, the default) becomes the
    /// host's CPU count; an explicit value is honored as-is. Both are
    /// clamped to at least 1 so a hand-written `cores = 0` in a config file
    /// never spins up a zero-worker runtime.
    #[must_use]
    pub fn resolved_cores(&self) -> usize {
        let cores = if self.cores == 0 {
            num_cpus::get()
        } else {
            self.cores
        };
        cores.max(1)
    }

    /// Build the `RuntimeSelection` this config resolves to — `None` for
    /// `backend = auto`, the caller's own default-resolution fallback.
    /// The config-shaped sibling of `RuntimeSelection::prime`/`::tokio`; see
    /// `RuntimeBackendSelection::resolve` for the errors this can return.
    pub fn resolve_selection(&self) -> Result<Option<RuntimeSelection>, ProximaError> {
        self.backend.resolve(self.resolved_cores())
    }

    /// Layered fluent loader (call-order precedence: a later layer wins per
    /// field it sets). Mirrors `CassetteConfig::layered`.
    #[must_use]
    pub fn layered() -> RuntimeConfigLayerBuilder {
        RuntimeConfigLayerBuilder {
            inner: Self::default(),
            touched: BTreeSet::new(),
        }
    }

    /// Resolve the effective config: defaults <- `PROXIMA_RUNTIME_*` env.
    /// The fallback `App::new()` / `AppBuilder::build()` use when no runtime
    /// is already installed ambiently (see `crate::runtime::installed_runtime`).
    ///
    /// # Errors
    /// Returns `ProximaError::Config` on a malformed env value (e.g.
    /// `PROXIMA_RUNTIME_CORES` set to a non-integer, or
    /// `PROXIMA_RUNTIME_BACKEND` set to an unknown name).
    pub fn resolve_from_env() -> Result<Self, ProximaError> {
        conflaguration::builder()
            .value(Self::default())
            .env()
            .build()
            .map_err(|error| ProximaError::Config(format!("runtime config: {error}")))
    }
}

/// Fluent layer builder for [`RuntimeConfig`]. Every source (`.from_path`,
/// `.from_env`, `.underlay_path`, `.underlay_env`, `.with_cores`)
/// contributes only the fields it actually specifies, merged onto the
/// accumulated config — a field a source doesn't touch falls through to
/// whatever prior layers set. `.from_path`/`.from_env` override (last writer
/// wins per field); `.underlay_path`/`.underlay_env` fill only fields still
/// unset; `.with_cores` always acts as an override at its call position.
#[derive(Debug, Clone)]
pub struct RuntimeConfigLayerBuilder {
    inner: RuntimeConfig,
    touched: BTreeSet<String>,
}

impl RuntimeConfigLayerBuilder {
    /// Merge a config file's fields onto the accumulated config; the file
    /// wins for every field it specifies.
    ///
    /// # Errors
    /// Propagates the conflaguration file/parse error.
    pub fn from_path<P: AsRef<Path>>(mut self, path: P) -> Result<Self, conflaguration::Error> {
        let incoming: Value = conflaguration::from_file(path.as_ref())?;
        apply_layer(
            &mut self.inner,
            &mut self.touched,
            incoming,
            MergeMode::Override,
        )?;
        Ok(self)
    }

    /// Fill any still-unset fields from a config file; already-set fields
    /// are left untouched.
    ///
    /// # Errors
    /// Propagates the conflaguration file/parse error.
    pub fn underlay_path<P: AsRef<Path>>(mut self, path: P) -> Result<Self, conflaguration::Error> {
        let incoming: Value = conflaguration::from_file(path.as_ref())?;
        apply_layer(
            &mut self.inner,
            &mut self.touched,
            incoming,
            MergeMode::Underlay,
        )?;
        Ok(self)
    }

    /// Merge `PROXIMA_RUNTIME_*` env-set fields onto the accumulated config;
    /// env wins for every field it sets. Unset env vars leave the current
    /// value untouched.
    ///
    /// # Errors
    /// Propagates the conflaguration env resolution error.
    pub fn from_env(mut self) -> Result<Self, conflaguration::Error> {
        let incoming = runtime_env_partial()?;
        apply_layer(
            &mut self.inner,
            &mut self.touched,
            incoming,
            MergeMode::Override,
        )?;
        Ok(self)
    }

    /// Fill any still-unset fields from env vars; already-set fields are
    /// left untouched even if the matching env var is set.
    ///
    /// # Errors
    /// Propagates the conflaguration env resolution error.
    pub fn underlay_env(mut self) -> Result<Self, conflaguration::Error> {
        let incoming = runtime_env_partial()?;
        apply_layer(
            &mut self.inner,
            &mut self.touched,
            incoming,
            MergeMode::Underlay,
        )?;
        Ok(self)
    }

    #[must_use]
    pub fn with_cores(mut self, cores: usize) -> Self {
        self.inner.cores = cores;
        self.touched.insert("cores".to_string());
        self
    }

    #[must_use]
    pub fn with_backend(mut self, backend: RuntimeBackendSelection) -> Self {
        self.inner.backend = backend;
        self.touched.insert("backend".to_string());
        self
    }

    #[must_use]
    pub fn build(self) -> RuntimeConfig {
        self.inner
    }
}

/// Top-level `App` configuration. `runtime` is nested — `App` grows more
/// config-driven sections here (listener defaults, etc.) as they earn a
/// surface; this is not a flat bag.
#[derive(Debug, Clone, Default, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_APP")]
#[builder(derive(Clone, Debug))]
pub struct AppConfig {
    /// Runtime sizing. `override_prefix` keeps the env surface at the
    /// pre-existing `PROXIMA_RUNTIME_*` names, not `PROXIMA_APP_RUNTIME_*`.
    #[setting(nested, override_prefix = "PROXIMA_RUNTIME")]
    #[serde(default)]
    #[builder(default)]
    pub runtime: RuntimeConfig,
}

impl Validate for AppConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        self.runtime.validate()
    }
}

/// Whether an incoming layer's fields win over an already-touched field
/// (`Override`, last writer wins) or only fill a field nothing has set yet
/// (`Underlay`, fill-only — never clobbers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MergeMode {
    Override,
    Underlay,
}

/// Merge `incoming`'s present fields onto `inner`, tracking which top-level
/// fields have been touched so `Underlay` layers never clobber an
/// already-set value. `RuntimeConfig` has exactly one field, so a one-level
/// merge covers it in full — see `cassette_config.rs`'s `apply_layer` for
/// the same primitive over a multi-field config.
fn apply_layer<T>(
    inner: &mut T,
    touched: &mut BTreeSet<String>,
    incoming: Value,
    mode: MergeMode,
) -> Result<(), conflaguration::Error>
where
    T: Serialize + DeserializeOwned,
{
    let Value::Object(incoming_map) = incoming else {
        return Ok(());
    };
    let mut base = to_value(inner)?;
    let Value::Object(base_map) = &mut base else {
        return Ok(());
    };
    for (key, value) in incoming_map {
        let should_apply = match mode {
            MergeMode::Override => true,
            MergeMode::Underlay => !touched.contains(&key),
        };
        if should_apply {
            touched.insert(key.clone());
            base_map.insert(key, value);
        }
    }
    *inner = from_value(base)?;
    Ok(())
}

/// The env-set subset of [`RuntimeConfig`]'s fields, as a partial JSON
/// object containing only the fields whose env var is actually present —
/// never the ones `Settings::from_env` filled with a default.
fn runtime_env_partial() -> Result<Value, conflaguration::Error> {
    let resolved = RuntimeConfig::from_env()?;
    let mut partial = Map::new();
    if std::env::var("PROXIMA_RUNTIME_CORES").is_ok() {
        partial.insert("cores".to_string(), to_value(&resolved.cores)?);
    }
    if std::env::var("PROXIMA_RUNTIME_BACKEND").is_ok() {
        partial.insert("backend".to_string(), to_value(&resolved.backend)?);
    }
    Ok(Value::Object(partial))
}

fn to_value<T: Serialize>(value: &T) -> Result<Value, conflaguration::Error> {
    serde_json::to_value(value).map_err(|error| conflaguration::Error::Validation {
        errors: vec![ValidationMessage::new(
            "layered",
            format!("serialize failed: {error}"),
        )],
    })
}

fn from_value<T: DeserializeOwned>(value: Value) -> Result<T, conflaguration::Error> {
    serde_json::from_value(value).map_err(|error| conflaguration::Error::Validation {
        errors: vec![ValidationMessage::new(
            "layered",
            format!("deserialize failed: {error}"),
        )],
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn default_cores_is_auto_sentinel() {
        let config = RuntimeConfig::default();
        assert_eq!(config.cores, 0);
        assert!(config.resolved_cores() >= 1);
    }

    #[test]
    fn explicit_cores_pass_through_resolved_cores() {
        let config = RuntimeConfig::builder().cores(4).build();
        assert_eq!(config.resolved_cores(), 4);
    }

    #[test]
    fn zero_cores_written_by_hand_clamps_to_at_least_one() {
        let config = RuntimeConfig {
            cores: 0,
            backend: RuntimeBackendSelection::Auto,
        };
        assert!(config.resolved_cores() >= 1);
    }

    #[test]
    fn env_overrides_default() {
        temp_env::with_vars([("PROXIMA_RUNTIME_CORES", Some("3"))], || {
            let config = RuntimeConfig::from_env().expect("env config");
            assert_eq!(config.cores, 3);
        });
    }

    #[test]
    fn malformed_env_value_is_a_loud_error() {
        temp_env::with_vars([("PROXIMA_RUNTIME_CORES", Some("not-a-number"))], || {
            assert!(RuntimeConfig::from_env().is_err());
        });
    }

    #[test]
    fn resolve_from_env_without_var_falls_through_to_default() {
        temp_env::with_vars_unset(["PROXIMA_RUNTIME_CORES"], || {
            let config = RuntimeConfig::resolve_from_env().expect("resolve");
            assert_eq!(config.cores, 0);
        });
    }

    #[test]
    fn layered_with_cores_wins_without_env() {
        temp_env::with_vars_unset(["PROXIMA_RUNTIME_CORES"], || {
            let config = RuntimeConfig::layered().with_cores(7).build();
            assert_eq!(config.cores, 7);
        });
    }

    #[test]
    fn layered_from_env_overrides_with_cores_set_before_it() {
        temp_env::with_vars([("PROXIMA_RUNTIME_CORES", Some("9"))], || {
            let config = RuntimeConfig::layered()
                .with_cores(2)
                .from_env()
                .expect("from_env")
                .build();
            assert_eq!(config.cores, 9, "env applied after with_cores wins");
        });
    }

    #[test]
    fn layered_underlay_env_never_clobbers_already_set_field() {
        temp_env::with_vars([("PROXIMA_RUNTIME_CORES", Some("9"))], || {
            let config = RuntimeConfig::layered()
                .with_cores(2)
                .underlay_env()
                .expect("underlay_env")
                .build();
            assert_eq!(config.cores, 2, "with_cores already set it; env is dropped");
        });
    }

    #[test]
    fn layered_from_path_overrides_default() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("runtime.toml");
        std::fs::write(&path, "cores = 5\n").expect("write toml");
        let config = RuntimeConfig::layered()
            .from_path(&path)
            .expect("from_path")
            .build();
        assert_eq!(config.cores, 5);
    }

    #[test]
    fn app_config_nests_runtime_and_resolves_prefixed_env() {
        temp_env::with_vars([("PROXIMA_RUNTIME_CORES", Some("6"))], || {
            let config = AppConfig::from_env().expect("app config from env");
            assert_eq!(config.runtime.cores, 6);
        });
    }

    #[test]
    fn app_config_default_matches_runtime_default() {
        let config = AppConfig::default();
        assert_eq!(config.runtime, RuntimeConfig::default());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn default_backend_is_auto() {
        let config = RuntimeConfig::default();
        assert_eq!(config.backend, RuntimeBackendSelection::Auto);
    }

    #[test]
    fn auto_backend_resolves_to_none() {
        let config = RuntimeConfig::default();
        assert!(
            config.resolve_selection().expect("resolve").is_none(),
            "auto must defer to the caller's own default-resolution fallback"
        );
    }

    #[test]
    fn backend_env_override_parses() {
        temp_env::with_vars([("PROXIMA_RUNTIME_BACKEND", Some("tokio"))], || {
            let config = RuntimeConfig::from_env().expect("env config");
            assert_eq!(config.backend, RuntimeBackendSelection::Tokio);
        });
    }

    #[test]
    fn malformed_backend_env_value_is_a_loud_error() {
        temp_env::with_vars([("PROXIMA_RUNTIME_BACKEND", Some("glommio"))], || {
            let error = RuntimeConfig::from_env().expect_err("unknown backend must error");
            assert!(format!("{error}").contains("glommio"));
        });
    }

    #[test]
    fn layered_with_backend_wins_without_env() {
        temp_env::with_vars_unset(["PROXIMA_RUNTIME_BACKEND"], || {
            let config = RuntimeConfig::layered()
                .with_backend(RuntimeBackendSelection::Prime)
                .build();
            assert_eq!(config.backend, RuntimeBackendSelection::Prime);
        });
    }

    // P4's headline fixture: a runtime selected from config produces the
    // SAME App state (backend + cores + capability shape) as the fluent
    // `RuntimeSelection::tokio(..)` constructor — the config and fluent
    // surfaces are isomorphic, not two independent paths that can drift.
    #[cfg(feature = "runtime-tokio")]
    #[test]
    fn config_selected_backend_round_trips_to_the_same_selection_as_the_fluent_builder() {
        let from_config = RuntimeConfig::builder()
            .cores(1)
            .backend(RuntimeBackendSelection::Tokio)
            .build()
            .resolve_selection()
            .expect("resolve_selection")
            .expect("backend = tokio must resolve to Some");
        let from_fluent = RuntimeSelection::tokio(1).expect("RuntimeSelection::tokio");

        assert_eq!(from_config.backend, from_fluent.backend);
        assert_eq!(
            from_config.runtime.num_cores(),
            from_fluent.runtime.num_cores()
        );
        assert_eq!(
            from_config.datagram_factory.is_some(),
            from_fluent.datagram_factory.is_some(),
            "capability shape (which factories are Some) must match backend-for-backend"
        );
        assert_eq!(
            from_config.unix_upstream_factory.is_some(),
            from_fluent.unix_upstream_factory.is_some()
        );
        assert_eq!(
            from_config.packet_listener_factory.is_some(),
            from_fluent.packet_listener_factory.is_some()
        );
    }

    // the TOML-file half of the same round trip — a config FILE selecting
    // `backend = "tokio"` resolves identically to the env/builder paths.
    #[cfg(feature = "runtime-tokio")]
    #[test]
    fn config_file_selected_backend_resolves_a_runtime_selection() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("runtime.toml");
        std::fs::write(&path, "cores = 1\nbackend = \"tokio\"\n").expect("write toml");
        let config = RuntimeConfig::layered()
            .from_path(&path)
            .expect("from_path")
            .build();
        assert_eq!(config.backend, RuntimeBackendSelection::Tokio);
        let selection = config
            .resolve_selection()
            .expect("resolve_selection")
            .expect("backend = tokio must resolve to Some");
        assert_eq!(selection.backend, crate::runtime::RuntimeBackend::Tokio);
    }

    // selecting a backend whose Cargo features are not linked is a runtime
    // Config error, not a panic or a silent fallback to a different backend.
    #[cfg(not(feature = "runtime-tokio"))]
    #[test]
    fn tokio_backend_without_the_feature_is_a_config_error() {
        // `RuntimeSelection` carries `Arc<dyn Trait>` fields with no `Debug`
        // impl (the trait objects aren't Debug), so `Result::expect_err`
        // (which requires `T: Debug`) can't be used here — match instead.
        let config = RuntimeConfig::builder()
            .backend(RuntimeBackendSelection::Tokio)
            .build();
        match config.resolve_selection() {
            Err(error) => assert!(format!("{error}").contains("tokio")),
            Ok(_) => panic!("tokio backend selected without runtime-tokio must error"),
        }
    }
}
