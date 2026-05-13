use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

use dashmap::DashMap;
use hdrhistogram::Histogram;
#[allow(unused_imports)]
use serde::{Deserialize, Serialize};
use thread_local::ThreadLocal;

pub use proxima_primitives::pipe::telemetry_surface::{Labels, NoopTelemetry, Telemetry, TelemetryHandle};

/// Process-wide cache of `metric_name -> Arc<str>`. Hot-path callers
/// pass the same `&'static str` literal billions of times; this
/// turns the per-record allocation into an Arc-clone (pointer bump)
/// after the first observation of each name.
///
/// The cache grows monotonically to the number of distinct metric
/// names in use (~10 in proxima today). No eviction; entries are
/// 'static-lifetime by construction.
static METRIC_NAME_CACHE: LazyLock<DashMap<String, Arc<str>>> = LazyLock::new(DashMap::new);

fn intern_metric_name(name: &str) -> Arc<str> {
    if let Some(existing) = METRIC_NAME_CACHE.get(name) {
        return Arc::clone(existing.value());
    }
    METRIC_NAME_CACHE
        .entry(name.to_string())
        .or_insert_with(|| Arc::from(name))
        .clone()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MetricKey {
    name: Arc<str>,
    labels: Labels,
}

impl MetricKey {
    fn new(name: &str, labels: &Labels) -> Self {
        Self {
            name: intern_metric_name(name),
            labels: labels.clone(),
        }
    }
}

pub struct Metrics {
    counters: DashMap<MetricKey, AtomicU64>,
    gauges: DashMap<MetricKey, AtomicI64>,
    histograms: DashMap<MetricKey, ShardedHistogram>,
    significant_digits: u8,
    max_value: u64,
}

/// Per-thread sharded histogram. Each writer thread gets its own
/// `Mutex<Histogram<u64>>` slot via `ThreadLocal`, so the steady-state
/// record path holds an uncontended thread-private mutex — concurrent
/// writers on different threads never collide. Read paths
/// (`histogram_summary`, `snapshot`) iterate all per-thread shards and
/// merge them into a fresh combined histogram. Reads are rare; writes
/// are hot. This replaces the prior single `Mutex<Histogram>` which
/// serialized every record across all threads.
pub struct ShardedHistogram {
    shards: ThreadLocal<HistogramShard>,
    significant_digits: u8,
    max_value: u64,
}

impl ShardedHistogram {
    fn new(significant_digits: u8, max_value: u64) -> Self {
        Self {
            shards: ThreadLocal::new(),
            significant_digits,
            max_value,
        }
    }

    fn record(&self, value: u64) {
        let shard = self
            .shards
            .get_or(|| HistogramShard::new(self.significant_digits, self.max_value));
        if let HistogramShard::Ready(mutex) = shard
            && let Ok(mut histogram) = mutex.lock()
        {
            let _ = histogram.record(value);
        }
    }

    fn merged(&self) -> Option<Histogram<u64>> {
        let mut combined = build_histogram(self.significant_digits, self.max_value)?;
        for shard in self.shards.iter() {
            if let HistogramShard::Ready(mutex) = shard
                && let Ok(histogram) = mutex.lock()
                && combined.add(&*histogram).is_err()
            {
                tracing::warn!("histogram shard merge failed; partial summary may result");
            }
        }
        if combined.is_empty() {
            return None;
        }
        Some(combined)
    }
}

enum HistogramShard {
    // WHY Mutex here:
    //   `hdrhistogram::Histogram<u64>` requires `&mut self` for
    //   `record()` (it updates internal bucket counters + auto-resizes
    //   on overflow). It's not internally synchronized.
    //
    // WHY NOT removable:
    //   - Per-thread shard via `ThreadLocal<HistogramShard>` fronts
    //     this Mutex (see the `ShardedHistogram` struct above). Each
    //     OS thread owns one shard, so the Mutex is uncontested in
    //     steady state — single-writer-per-instance.
    //   - Replacing the Mutex with `RefCell` would require
    //     `Send + Sync` to be unsafely asserted (HistogramShard
    //     crosses thread boundaries via `ThreadLocal` even though
    //     access stays single-thread). Mutex preserves Send + Sync
    //     properly.
    //   - Atomic histogram (e.g., crossbeam) doesn't exist for hdr;
    //     would need to roll our own. hdrhistogram is the de facto
    //     latency-histogram crate.
    //
    // WHY this is right:
    //   `benches/histogram_record.rs` measures the record path:
    //   single-thread record = 472 ns macOS / 242 ns Linux. Most of
    //   that is the histogram bucket math, NOT the Mutex acquire
    //   (~5ns uncontested). The 8-worker shared-histogram bench
    //   (18 µs macOS / 10 µs Linux) shows what happens WITHOUT
    //   sharding — which is why ShardedHistogram exists.
    //
    //   The Mutex is only contended on `merged()` (the read/snapshot
    //   path), which is rare (export endpoint, debugging).
    Ready(Mutex<Histogram<u64>>),
    Unavailable,
}

impl HistogramShard {
    fn new(significant_digits: u8, max_value: u64) -> Self {
        match build_histogram(significant_digits, max_value) {
            Some(histogram) => Self::Ready(Mutex::new(histogram)),
            None => Self::Unavailable,
        }
    }
}

fn build_histogram(significant_digits: u8, max_value: u64) -> Option<Histogram<u64>> {
    Histogram::<u64>::new_with_max(max_value, significant_digits)
        .or_else(|_| Histogram::<u64>::new(significant_digits.min(2)))
        .or_else(|_| Histogram::<u64>::new(0))
        .ok()
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new(3, 60_000_000)
    }
}

impl Metrics {
    #[must_use]
    pub fn new(significant_digits: u8, max_value: u64) -> Self {
        Self {
            counters: DashMap::new(),
            gauges: DashMap::new(),
            histograms: DashMap::new(),
            significant_digits,
            max_value,
        }
    }

    #[must_use]
    pub fn counter(&self, metric: &str, labels: &Labels) -> Option<u64> {
        self.counters
            .get(&MetricKey::new(metric, labels))
            .map(|entry| entry.load(Ordering::Relaxed))
    }

    #[must_use]
    pub fn gauge(&self, metric: &str, labels: &Labels) -> Option<i64> {
        self.gauges
            .get(&MetricKey::new(metric, labels))
            .map(|entry| entry.load(Ordering::Relaxed))
    }

    #[must_use]
    pub fn histogram_summary(&self, metric: &str, labels: &Labels) -> Option<HistogramSummary> {
        let entry = self.histograms.get(&MetricKey::new(metric, labels))?;
        let histogram = entry.merged()?;
        Some(summary_from_histogram(&histogram))
    }

    #[must_use]
    pub fn snapshot(&self) -> MetricsSnapshot {
        let counters = self
            .counters
            .iter()
            .map(|entry| {
                (
                    entry.key().name.as_ref().to_string(),
                    entry.key().labels.clone(),
                    entry.value().load(Ordering::Relaxed),
                )
            })
            .collect();
        let gauges = self
            .gauges
            .iter()
            .map(|entry| {
                (
                    entry.key().name.as_ref().to_string(),
                    entry.key().labels.clone(),
                    entry.value().load(Ordering::Relaxed),
                )
            })
            .collect();
        let histograms = self
            .histograms
            .iter()
            .filter_map(|entry| {
                let histogram = entry.value().merged()?;
                Some((
                    entry.key().name.as_ref().to_string(),
                    entry.key().labels.clone(),
                    summary_from_histogram(&histogram),
                ))
            })
            .collect();
        MetricsSnapshot {
            counters,
            gauges,
            histograms,
        }
    }

    fn ensure_histogram(&self, key: MetricKey) {
        if self.histograms.contains_key(&key) {
            return;
        }
        let significant_digits = self.significant_digits;
        let max_value = self.max_value;
        self.histograms
            .entry(key)
            .or_insert_with(|| ShardedHistogram::new(significant_digits, max_value));
    }
}

fn summary_from_histogram(histogram: &Histogram<u64>) -> HistogramSummary {
    HistogramSummary {
        count: histogram.len(),
        min: histogram.min() as f64,
        max: histogram.max() as f64,
        mean: histogram.mean(),
        p50: histogram.value_at_percentile(50.0) as f64,
        p90: histogram.value_at_percentile(90.0) as f64,
        p99: histogram.value_at_percentile(99.0) as f64,
        p99_9: histogram.value_at_percentile(99.9) as f64,
    }
}

impl Telemetry for Metrics {
    fn counter_inc(&self, metric: &str, labels: &Labels, by: u64) {
        let key = MetricKey::new(metric, labels);
        let entry = self
            .counters
            .entry(key)
            .or_insert_with(|| AtomicU64::new(0));
        entry.fetch_add(by, Ordering::Relaxed);
    }

    fn gauge_set(&self, metric: &str, labels: &Labels, value: i64) {
        let key = MetricKey::new(metric, labels);
        let entry = self.gauges.entry(key).or_insert_with(|| AtomicI64::new(0));
        entry.store(value, Ordering::Relaxed);
    }

    fn histogram_record(&self, metric: &str, labels: &Labels, value: f64) {
        let bucket = if value.is_finite() && value >= 0.0 {
            value.round() as u64
        } else {
            return;
        };
        let key = MetricKey::new(metric, labels);
        self.ensure_histogram(key.clone());
        if let Some(entry) = self.histograms.get(&key) {
            let clamped = bucket.min(self.max_value);
            entry.record(clamped);
        }
    }
}

pub use proxima_primitives::pipe::telemetry_surface::{HistogramSummary, MetricsSnapshot};

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    fn labels_sort_by_name_for_stable_keys() {
        let one = Labels::from_pairs(&[("zone", "a"), ("region", "us")]);
        let two = Labels::from_pairs(&[("region", "us"), ("zone", "a")]);
        assert_eq!(
            one, two,
            "label order at construction must not affect identity"
        );
    }

    #[test]
    fn counter_inc_accumulates() {
        let metrics = Metrics::default();
        let labels = Labels::from_pairs(&[("pipe", "echo")]);
        metrics.counter_inc("requests_total", &labels, 1);
        metrics.counter_inc("requests_total", &labels, 4);
        assert_eq!(metrics.counter("requests_total", &labels), Some(5));
    }

    #[test]
    fn gauge_set_overwrites_previous_value() {
        let metrics = Metrics::default();
        let labels = Labels::from_pairs(&[("listener", "main")]);
        metrics.gauge_set("connections_active", &labels, 12);
        metrics.gauge_set("connections_active", &labels, 7);
        assert_eq!(metrics.gauge("connections_active", &labels), Some(7));
    }

    #[test]
    fn histogram_records_compute_percentiles() {
        let metrics = Metrics::default();
        let labels = Labels::from_pairs(&[("pipe", "echo")]);
        for sample in 1..=1000u64 {
            metrics.histogram_record("latency_ms", &labels, sample as f64);
        }
        let summary = metrics
            .histogram_summary("latency_ms", &labels)
            .expect("summary present after samples");
        assert_eq!(summary.count, 1000);
        assert!(
            summary.p50 >= 499.0 && summary.p50 <= 501.0,
            "p50={}",
            summary.p50
        );
        assert!(
            summary.p90 >= 899.0 && summary.p90 <= 901.0,
            "p90={}",
            summary.p90
        );
        assert!(
            summary.p99 >= 989.0 && summary.p99 <= 1000.0,
            "p99={}",
            summary.p99
        );
    }

    #[test]
    fn histogram_summary_returns_none_before_first_sample() {
        let metrics = Metrics::default();
        let labels = Labels::empty();
        assert!(metrics.histogram_summary("latency_ms", &labels).is_none());
    }

    #[test]
    fn snapshot_returns_all_recorded_metrics() {
        let metrics = Metrics::default();
        let labels = Labels::from_pairs(&[("pipe", "echo")]);
        metrics.counter_inc("requests_total", &labels, 3);
        metrics.gauge_set("connections", &labels, 5);
        metrics.histogram_record("latency_ms", &labels, 12.0);
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.counters.len(), 1);
        assert_eq!(snapshot.gauges.len(), 1);
        assert_eq!(snapshot.histograms.len(), 1);
    }

    #[rstest]
    #[case::nan(f64::NAN)]
    #[case::neg_infinity(f64::NEG_INFINITY)]
    #[case::negative(-5.0)]
    fn histogram_record_rejects_invalid_values(#[case] value: f64) {
        let metrics = Metrics::default();
        let labels = Labels::empty();
        metrics.histogram_record("anything", &labels, value);
        assert!(metrics.histogram_summary("anything", &labels).is_none());
    }

    #[test]
    fn noop_telemetry_records_nothing_visible() {
        let telemetry = NoopTelemetry;
        let labels = Labels::empty();
        telemetry.counter_inc("foo", &labels, 100);
        telemetry.gauge_set("foo", &labels, 100);
        telemetry.histogram_record("foo", &labels, 100.0);
    }

    #[test]
    fn histogram_records_merge_across_threads_with_no_data_loss() {
        let metrics = Arc::new(Metrics::default());
        let labels = Labels::from_pairs(&[("pipe", "stress")]);
        let workers = 8;
        let per_worker = 1_000_u64;
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let metrics = metrics.clone();
            let labels = labels.clone();
            handles.push(std::thread::spawn(move || {
                for sample in 1..=per_worker {
                    metrics.histogram_record("latency_ms", &labels, sample as f64);
                }
            }));
        }
        for handle in handles {
            handle.join().expect("worker thread");
        }
        let summary = metrics
            .histogram_summary("latency_ms", &labels)
            .expect("merged summary present");
        assert_eq!(
            summary.count,
            per_worker * workers as u64,
            "per-thread shards must sum without dropping records",
        );
    }

    #[test]
    fn distinct_label_sets_keep_separate_counters() {
        let metrics = Metrics::default();
        let echo = Labels::from_pairs(&[("pipe", "echo")]);
        let stripe = Labels::from_pairs(&[("pipe", "stripe")]);
        metrics.counter_inc("requests_total", &echo, 7);
        metrics.counter_inc("requests_total", &stripe, 3);
        assert_eq!(metrics.counter("requests_total", &echo), Some(7));
        assert_eq!(metrics.counter("requests_total", &stripe), Some(3));
    }

    #[test]
    fn metrics_snapshot_round_trips_through_json() {
        let metrics = Metrics::default();
        let labels = Labels::from_pairs(&[("pipe", "echo"), ("region", "us")]);
        metrics.counter_inc("proxima.requests_total", &labels, 42);
        metrics.gauge_set("proxima.queue_depth", &labels, 7);
        for sample in 1..=10 {
            metrics.histogram_record("proxima.latency_ms", &labels, sample as f64);
        }
        let snapshot = metrics.snapshot();
        let encoded = serde_json::to_string(&snapshot).expect("serialize snapshot");
        let decoded: MetricsSnapshot =
            serde_json::from_str(&encoded).expect("deserialize snapshot");
        assert_eq!(decoded.counters.len(), snapshot.counters.len());
        assert_eq!(decoded.gauges.len(), snapshot.gauges.len());
        assert_eq!(decoded.histograms.len(), snapshot.histograms.len());
        let (decoded_name, decoded_labels, decoded_value) = &decoded.counters[0];
        let (orig_name, orig_labels, orig_value) = &snapshot.counters[0];
        assert_eq!(decoded_name, orig_name);
        assert_eq!(decoded_labels, orig_labels);
        assert_eq!(decoded_value, orig_value);
        let (_, _, decoded_hist) = &decoded.histograms[0];
        let (_, _, orig_hist) = &snapshot.histograms[0];
        assert_eq!(decoded_hist, orig_hist);
    }
}
