//! Telemetry trait surface destined for `proxima-pipe`.
//!
//! Lifted from `telemetry/legacy.rs` during Phase 1.5 of the
//! decomposition (see `docs/decomposition/discipline.md`). The trait +
//! `Labels` value type + `NoopTelemetry` default impl + `TelemetryHandle`
//! alias live here so `request.rs` can reference them without pulling
//! the full metrics impl (dashmap, hdrhistogram, thread_local).
//!
//! On Phase 2 (proxima-pipe extraction) this file's contents move to the
//! new `proxima-pipe` crate. The full `Metrics` impl in `legacy.rs` moves
//! to `proxima-telemetry` and depends on this trait surface.

#![cfg(feature = "alloc")]

use alloc::string::String;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Labels {
    entries: Vec<(String, String)>,
}

impl Labels {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    #[must_use]
    pub fn from_pairs(pairs: &[(&str, &str)]) -> Self {
        let mut entries: Vec<(String, String)> = pairs
            .iter()
            .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
            .collect();
        entries.sort_by(|left, right| left.0.cmp(&right.0));
        Self { entries }
    }

    #[must_use]
    pub fn entries(&self) -> &[(String, String)] {
        &self.entries
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl<const SIZE: usize> From<&[(&str, &str); SIZE]> for Labels {
    fn from(pairs: &[(&str, &str); SIZE]) -> Self {
        Self::from_pairs(pairs)
    }
}

pub trait Telemetry: Send + Sync + 'static {
    fn counter_inc(&self, metric: &str, labels: &Labels, by: u64);
    fn gauge_set(&self, metric: &str, labels: &Labels, value: i64);
    fn histogram_record(&self, metric: &str, labels: &Labels, value: f64);

    /// Hot-path gate: when this returns `false`, a caller on a tight loop may
    /// skip building `Labels` and computing values entirely — the records
    /// would be dropped anyway. Defaults to `true`; the no-op handle returns
    /// `false` so a telemetry-less client pays nothing per call.
    fn is_active(&self) -> bool {
        true
    }
}

pub type TelemetryHandle = Arc<dyn Telemetry>;

pub struct NoopTelemetry;

impl NoopTelemetry {
    #[must_use]
    pub fn handle() -> TelemetryHandle {
        Arc::new(NoopTelemetry)
    }
}

impl Telemetry for NoopTelemetry {
    fn counter_inc(&self, _metric: &str, _labels: &Labels, _by: u64) {}
    fn gauge_set(&self, _metric: &str, _labels: &Labels, _value: i64) {}
    fn histogram_record(&self, _metric: &str, _labels: &Labels, _value: f64) {}
    fn is_active(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistogramSummary {
    pub count: u64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub p50: f64,
    pub p90: f64,
    pub p99: f64,
    pub p99_9: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub counters: Vec<(String, Labels, u64)>,
    pub gauges: Vec<(String, Labels, i64)>,
    pub histograms: Vec<(String, Labels, HistogramSummary)>,
}
