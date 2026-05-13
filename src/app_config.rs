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
use std::path::Path;

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::ProximaError;

fn default_cores() -> usize {
    0
}

/// The App's runtime sizing. One field today; more App-level runtime knobs
/// land here as they earn a config surface.
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
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl Validate for RuntimeConfig {
    // `cores` has no invalid representation: `0` is the documented "auto"
    // sentinel, any other value is a literal worker count.
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
    /// `PROXIMA_RUNTIME_CORES` set to a non-integer).
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
        let config = RuntimeConfig { cores: 0 };
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
}
