//! `InstrumentConfig` — the unified-instrument config + fluent builder surface
//! (P4: first-class conflaguration AND first-class fluent builder, BOTH).
//!
//! Mirrors [`crate::config::TelemetryConfig`] and [`crate::emit::config::EmitConfig`]
//! exactly: `#[derive(Builder, Deserialize, Serialize, Settings)]` + [`Validate`],
//! an `InstrumentLayerBuilder` with call-order precedence, and a typed env surface.
//! The defaults come FROM the `sized` consts that `build.rs` generates from
//! `[instrument]` in `proxima-telemetry.toml`, so the runtime default follows the
//! compile-time / no_std+no_alloc floor — there is no double source of truth.
//!
//! Tier: std (conflaguration, fs, env). At no_std+no_alloc the `sized` consts ARE
//! the config; this layer is the std runtime override on top.

use std::collections::BTreeSet;

use bon::Builder;
use conflaguration::{Settings, Validate};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::config_merge::{MergeMode, apply_layer, insert_if_env_set};

fn default_metrics() -> bool {
    crate::sized::INSTRUMENT_METRICS_DEFAULT
}
fn default_budget_micros() -> u64 {
    crate::sized::INSTRUMENT_DEFAULT_BUDGET_MICROS
}

/// Unified-instrument runtime config. One built `InstrumentConfig` ==
/// one serialisable config == one recorder policy (via [`InstrumentConfig::apply`]).
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "INSTRUMENT")]
#[builder(derive(Clone, Debug))]
pub struct InstrumentConfig {
    /// Does a recorder consume the span-duration metric by default (the consumer
    /// gate)? `false` = a span close records no metric until something subscribes;
    /// `true` = a metric-consuming deployment, recorder auto-subscribes at build.
    /// Defaults from `[instrument] metrics` (the `sized` floor).
    #[setting(default = false)]
    #[serde(default = "default_metrics")]
    #[builder(default = default_metrics())]
    pub metrics: bool,

    /// Default tail-sampling budget (microseconds) for spans with no explicit
    /// `#[span(budget)]`. A span overrunning it force-keeps its trace past head
    /// sampling. `0` = none. Defaults from `[instrument] default_budget_micros`.
    #[setting(default = 0)]
    #[serde(default = "default_budget_micros")]
    #[builder(default = default_budget_micros())]
    pub default_budget_micros: u64,
}

impl Default for InstrumentConfig {
    fn default() -> Self {
        InstrumentConfig::builder().build()
    }
}

impl Validate for InstrumentConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        // booleans + a non-negative budget — nothing to reject; the type IS the
        // validation. Present for parity with the other config surfaces.
        Ok(())
    }
}

impl InstrumentConfig {
    /// The default budget as nanoseconds (the unit the span builder takes), or
    /// `None` when no default budget is set.
    #[must_use]
    pub fn default_budget_ns(&self) -> Option<u64> {
        match self.default_budget_micros {
            0 => None,
            micros => Some(micros.saturating_mul(1_000)),
        }
    }

    /// Apply this policy to a recorder: subscribe the metric consumer when
    /// `metrics`, and install the default tail-sampling budget. The single seam a
    /// config-driven deployment uses; `#[span]` annotations need not change.
    #[cfg(feature = "instrument-metrics")]
    pub fn apply(&self, recorder: &crate::recorder::Recorder) {
        if self.metrics {
            recorder.enable_span_metrics();
        }
        recorder.set_default_budget_ns(self.default_budget_ns().unwrap_or(0));
    }

    /// Start a layered builder from the `sized`-seeded defaults.
    #[must_use]
    pub fn layered() -> InstrumentLayerBuilder {
        InstrumentLayerBuilder {
            inner: InstrumentConfig::default(),
            touched: BTreeSet::new(),
        }
    }
}

/// Fluent builder for [`InstrumentConfig`] (mirrors `EmitLayerBuilder`).
/// Every source contributes only the fields it actually specifies, merged
/// onto the accumulated config. `.from_path`/`.from_env` override (last
/// writer wins per field); `.underlay_path`/`.underlay_env` fill only
/// fields still unset; `.with_*` always acts as an override at its call
/// position.
pub struct InstrumentLayerBuilder {
    inner: InstrumentConfig,
    touched: BTreeSet<String>,
}

impl InstrumentLayerBuilder {
    /// Merge a TOML/JSON file's fields onto the accumulated config; the file
    /// wins for every field it specifies.
    pub fn from_path<P: AsRef<std::path::Path>>(
        mut self,
        path: P,
    ) -> Result<Self, conflaguration::Error> {
        let incoming: Value = conflaguration::from_file(path.as_ref())?;
        apply_layer(
            &mut self.inner,
            &mut self.touched,
            incoming,
            MergeMode::Override,
            &[],
        )?;
        Ok(self)
    }

    /// Fill any still-unset fields from a TOML/JSON file; already-set fields
    /// are left untouched.
    pub fn underlay_path<P: AsRef<std::path::Path>>(
        mut self,
        path: P,
    ) -> Result<Self, conflaguration::Error> {
        let incoming: Value = conflaguration::from_file(path.as_ref())?;
        apply_layer(
            &mut self.inner,
            &mut self.touched,
            incoming,
            MergeMode::Underlay,
            &[],
        )?;
        Ok(self)
    }

    /// Merge `INSTRUMENT_*` env-set fields onto the accumulated config; env
    /// wins for every field it sets.
    pub fn from_env(mut self) -> Result<Self, conflaguration::Error> {
        let incoming = instrument_env_partial()?;
        apply_layer(
            &mut self.inner,
            &mut self.touched,
            incoming,
            MergeMode::Override,
            &[],
        )?;
        Ok(self)
    }

    /// Fill any still-unset fields from `INSTRUMENT_*` env; already-set
    /// fields are left untouched even if the matching env var is set.
    pub fn underlay_env(mut self) -> Result<Self, conflaguration::Error> {
        let incoming = instrument_env_partial()?;
        apply_layer(
            &mut self.inner,
            &mut self.touched,
            incoming,
            MergeMode::Underlay,
            &[],
        )?;
        Ok(self)
    }

    /// Enable or disable the span-duration metric consumer.
    #[must_use]
    pub fn with_metrics(mut self, metrics: bool) -> Self {
        self.inner.metrics = metrics;
        self.touched.insert("metrics".to_string());
        self
    }

    /// Set the default tail-sampling budget in microseconds (`0` = none).
    #[must_use]
    pub fn with_default_budget_micros(mut self, micros: u64) -> Self {
        self.inner.default_budget_micros = micros;
        self.touched.insert("default_budget_micros".to_string());
        self
    }

    /// The built immutable config.
    #[must_use]
    pub fn build(self) -> InstrumentConfig {
        self.inner
    }
}

/// The env-set subset of [`InstrumentConfig`]'s fields, as a partial JSON
/// object containing only the fields whose env var is actually present —
/// never the ones `Settings::from_env` filled with a default.
fn instrument_env_partial() -> Result<Value, conflaguration::Error> {
    let resolved = InstrumentConfig::from_env()?;
    let mut partial = Map::new();
    insert_if_env_set(
        &mut partial,
        "metrics",
        &["INSTRUMENT_METRICS"],
        &resolved.metrics,
    )?;
    insert_if_env_set(
        &mut partial,
        "default_budget_micros",
        &["INSTRUMENT_DEFAULT_BUDGET_MICROS"],
        &resolved.default_budget_micros,
    )?;
    Ok(Value::Object(partial))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::InstrumentConfig;

    // P4 parity: a built config round-trips through serde unchanged (the
    // conflaguration wire form == the fluent-built struct).
    #[test]
    fn config_round_trips_through_serde() {
        let built = InstrumentConfig::layered()
            .with_metrics(true)
            .with_default_budget_micros(500)
            .build();

        let json = serde_json::to_string(&built).unwrap();
        let from_json: InstrumentConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(from_json, built);
    }

    // the fluent builder and the conflaguration surface agree on defaults — both
    // seeded by the `sized` floor (build.rs `[instrument]`), the single source.
    #[test]
    fn defaults_track_the_sized_floor() {
        let cfg = InstrumentConfig::default();
        assert_eq!(cfg.metrics, crate::sized::INSTRUMENT_METRICS_DEFAULT);
        assert_eq!(
            cfg.default_budget_micros,
            crate::sized::INSTRUMENT_DEFAULT_BUDGET_MICROS
        );
    }

    // micros lower to nanoseconds; 0 means "no default budget", not 0 ns.
    #[test]
    fn budget_micros_lower_to_ns() {
        let none = InstrumentConfig::layered()
            .with_default_budget_micros(0)
            .build();
        assert_eq!(none.default_budget_ns(), None);
        let some = InstrumentConfig::layered()
            .with_default_budget_micros(2)
            .build();
        assert_eq!(some.default_budget_ns(), Some(2_000));
    }

    // the config seam actually drives a recorder: apply() subscribes the metric
    // consumer (so closes record) and installs the default budget — the
    // config-driven deployment path, no #[span] annotation change.
    #[test]
    fn apply_drives_the_recorder() {
        use crate::metric::MetricSample;
        use crate::pipes::InMemoryPipe;
        use crate::recorder::Recorder;

        let pipe = InMemoryPipe::new();
        let recorder = Recorder::builder()
            .pipe(pipe.clone())
            .core_count(1)
            .start()
            .expect("recorder build");

        // default (no consumer): a close records no metric.
        drop(recorder.span("pre_apply").start());

        InstrumentConfig::layered()
            .with_metrics(true)
            .with_default_budget_micros(5)
            .build()
            .apply(&recorder);

        // after apply: the consumer is subscribed, so the close records.
        drop(recorder.span("post_apply").start());

        // fold (deferred: at drain via Block/assist; inline: already folded) and
        // export, then read the exported sample — the live `count()` is snapshotted
        // and reset by the drain, so it is not the tier-agnostic observable.
        while recorder.drain() > 0 {}
        let histograms = pipe
            .metrics()
            .into_iter()
            .filter(|sample| matches!(sample, MetricSample::Histogram(_)))
            .count();
        assert_eq!(
            histograms, 1,
            "with_metrics(true) drives one duration histogram; the pre-apply close (no consumer) records none"
        );
    }

    // OPEN: a real TOML loads through the actual conflaguration loader (not just
    // serde) — file config is first-class.
    #[test]
    fn loads_from_toml_via_conflaguration() {
        let toml = "metrics = true\ndefault_budget_micros = 750\n";
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("instrument.toml");
        std::fs::write(&path, toml).unwrap();

        let cfg = InstrumentConfig::layered()
            .from_path(&path)
            .unwrap()
            .build();
        assert!(cfg.metrics);
        assert_eq!(cfg.default_budget_micros, 750);
        assert_eq!(cfg.default_budget_ns(), Some(750_000));
    }

    // the exact seam-#3 case: a file sets TWO fields, env sets only ONE.
    #[test]
    fn seam_3_from_path_then_from_env_preserves_files_untouched_field() {
        let toml = "metrics = true\ndefault_budget_micros = 750\n";
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("instrument.toml");
        std::fs::write(&path, toml).unwrap();

        temp_env::with_vars([("INSTRUMENT_DEFAULT_BUDGET_MICROS", Some("42"))], || {
            let cfg = InstrumentConfig::layered()
                .from_path(&path)
                .unwrap()
                .from_env()
                .unwrap()
                .build();
            assert_eq!(cfg.default_budget_micros, 42, "env wins the field it sets");
            assert!(cfg.metrics, "the file's field must survive");
        });
    }

    // order-independence: the same two sources, built both orders.
    #[test]
    fn order_independence_file_then_env_vs_env_then_file() {
        let toml = "metrics = true\n";
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("instrument.toml");
        std::fs::write(&path, toml).unwrap();

        temp_env::with_vars([("INSTRUMENT_DEFAULT_BUDGET_MICROS", Some("42"))], || {
            let file_then_env = InstrumentConfig::layered()
                .from_path(&path)
                .unwrap()
                .from_env()
                .unwrap()
                .build();
            assert!(file_then_env.metrics, "file's field survives");
            assert_eq!(
                file_then_env.default_budget_micros, 42,
                "env's field applies"
            );

            let env_then_file = InstrumentConfig::layered()
                .from_env()
                .unwrap()
                .from_path(&path)
                .unwrap()
                .build();
            assert_eq!(
                env_then_file.default_budget_micros, 42,
                "env's field survives"
            );
            assert!(env_then_file.metrics, "file's field applies");
        });
    }

    // full stack: defaults < file < env < with_*.
    #[test]
    fn full_stack_defaults_file_env_with_override_each_field() {
        let toml = "metrics = true\n";
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("instrument.toml");
        std::fs::write(&path, toml).unwrap();

        temp_env::with_vars([("INSTRUMENT_DEFAULT_BUDGET_MICROS", Some("42"))], || {
            let cfg = InstrumentConfig::layered()
                .from_path(&path)
                .unwrap()
                .from_env()
                .unwrap()
                .with_metrics(false)
                .build();
            assert!(!cfg.metrics, "with_* layer wins over the file's true");
            assert_eq!(cfg.default_budget_micros, 42, "env layer");
        });
    }

    // underlay never clobbers an already-set field; it DOES fill an unset one.
    #[test]
    fn underlay_path_fills_only_unset_fields() {
        let toml = "metrics = true\ndefault_budget_micros = 750\n";
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("instrument.toml");
        std::fs::write(&path, toml).unwrap();

        let cfg = InstrumentConfig::layered()
            .with_metrics(false)
            .underlay_path(&path)
            .unwrap()
            .build();
        assert!(
            !cfg.metrics,
            "already set by with_*; the file's value is dropped"
        );
        assert_eq!(
            cfg.default_budget_micros, 750,
            "unset before underlay; the file fills it"
        );
    }

    // combined: defaults -> underlay(file) -> override(env) -> override(with_*).
    #[test]
    fn combined_underlay_file_then_override_env_then_with() {
        let toml = "metrics = true\ndefault_budget_micros = 750\n";
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("instrument.toml");
        std::fs::write(&path, toml).unwrap();

        temp_env::with_vars([("INSTRUMENT_METRICS", Some("false"))], || {
            let cfg = InstrumentConfig::layered()
                .underlay_path(&path)
                .unwrap()
                .from_env()
                .unwrap()
                .with_default_budget_micros(9)
                .build();
            assert!(!cfg.metrics, "override(env) wins over underlay(file)");
            assert_eq!(
                cfg.default_budget_micros, 9,
                "the later with_* overrides underlay(file)"
            );
        });
    }
}
