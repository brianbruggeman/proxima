// opt-sweep candidate: replace HashMap + Mutex with a lock-free trie or
// CSR-slot slab once the registry becomes a hot path during burst registration.
// v1 baseline: O(1) avg via HashMap; registration is once-per-instrument-name.

extern crate std;

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, Ordering};
use std::collections::HashMap;

use parking_lot::Mutex;

use crate::metric::counter::Counter;
use crate::metric::gauge::Gauge;
use crate::metric::updown::UpDownCounter;
use crate::metric::{MetricSample, NumberDataPoint};
use crate::tag::{ScalarValue, Tag};

#[cfg(feature = "histogram")]
use crate::metric::histogram::Histogram;

#[cfg(feature = "instrument-metrics")]
use crate::metric::exemplar::ExemplarCell;

pub struct InstrumentRegistry {
    by_name: Mutex<HashMap<&'static str, u32>>,
    next: AtomicU32,
    counters: Mutex<Vec<Arc<Counter>>>,
    gauges: Mutex<Vec<Arc<Gauge>>>,
    updown_counters: Mutex<Vec<Arc<UpDownCounter>>>,
    #[cfg(feature = "histogram")]
    histograms: Mutex<Vec<Arc<Histogram<f64>>>>,
    // C4: one exemplar cell per instrument index, parallel to the histograms.
    #[cfg(feature = "instrument-metrics")]
    exemplars: Mutex<Vec<Arc<ExemplarCell>>>,
}

impl Default for InstrumentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl InstrumentRegistry {
    pub fn new() -> Self {
        Self {
            by_name: Mutex::new(HashMap::new()),
            next: AtomicU32::new(0),
            counters: Mutex::new(Vec::new()),
            gauges: Mutex::new(Vec::new()),
            updown_counters: Mutex::new(Vec::new()),
            #[cfg(feature = "histogram")]
            histograms: Mutex::new(Vec::new()),
            #[cfg(feature = "instrument-metrics")]
            exemplars: Mutex::new(Vec::new()),
        }
    }

    pub fn register(&self, name: &'static str) -> u32 {
        let mut guard = self.by_name.lock();
        if let Some(&existing) = guard.get(name) {
            return existing;
        }
        let index = self.next.fetch_add(1, Ordering::Relaxed);
        guard.insert(name, index);
        index
    }

    pub fn lookup(&self, name: &'static str) -> Option<u32> {
        self.by_name.lock().get(name).copied()
    }

    /// Register a counter instrument for drain-time snapshot.
    ///
    /// If a counter with this name already exists, returns the existing Arc.
    /// Registration is expected once at startup, not on the hot emit path.
    pub fn register_counter(&self, name: &'static str) -> Arc<Counter> {
        let mut by_name = self.by_name.lock();
        if let Some(&index) = by_name.get(name) {
            let counters = self.counters.lock();
            if let Some(existing) = counters.get(index as usize) {
                return Arc::clone(existing);
            }
        }

        let counter = Arc::new(Counter::new(name));
        let index = self.next.fetch_add(1, Ordering::Relaxed);
        by_name.insert(name, index);
        let mut counters = self.counters.lock();
        // fill any gap with sentinel so index == position
        while counters.len() < index as usize {
            counters.push(Arc::new(Counter::new("")));
        }
        counters.push(Arc::clone(&counter));
        counter
    }

    /// Register a gauge instrument (last-value) for drain-time snapshot.
    ///
    /// Idempotent by name. Registration is once-at-startup, so the linear scan is
    /// not a hot path; `next` is bumped so a gauge-only recorder isn't skipped by
    /// the snapshot early-out.
    pub fn register_gauge(&self, name: &'static str) -> Arc<Gauge> {
        let mut gauges = self.gauges.lock();
        if let Some(existing) = gauges.iter().find(|gauge| gauge.name == name) {
            return Arc::clone(existing);
        }
        self.next.fetch_add(1, Ordering::Relaxed);
        let gauge = Arc::new(Gauge::new(name));
        gauges.push(Arc::clone(&gauge));
        gauge
    }

    /// Register an up-down counter (cumulative signed sum) for drain-time snapshot.
    pub fn register_updown_counter(&self, name: &'static str) -> Arc<UpDownCounter> {
        let mut updowns = self.updown_counters.lock();
        if let Some(existing) = updowns.iter().find(|updown| updown.name == name) {
            return Arc::clone(existing);
        }
        self.next.fetch_add(1, Ordering::Relaxed);
        let updown = Arc::new(UpDownCounter::new(name));
        updowns.push(Arc::clone(&updown));
        updown
    }

    /// Register a histogram instrument for drain-time snapshot.
    ///
    /// Reuses the name's existing `by_name` index if one is already present (e.g.
    /// its exemplar registered first — the order the deferred-fold path produces),
    /// so histogram and exemplar stay index-aligned regardless of which registered
    /// first. Placing at the exact index (not `push`) is what makes it order-safe.
    #[cfg(feature = "histogram")]
    pub fn register_histogram(&self, name: &'static str) -> Arc<Histogram<f64>> {
        let mut by_name = self.by_name.lock();
        let mut histograms = self.histograms.lock();
        let index = match by_name.get(name) {
            Some(&existing) => {
                if let Some(existing_hist) = histograms.get(existing as usize)
                    && !existing_hist.name.is_empty()
                {
                    return Arc::clone(existing_hist);
                }
                existing as usize
            }
            None => {
                let next = self.next.fetch_add(1, Ordering::Relaxed);
                by_name.insert(name, next);
                next as usize
            }
        };
        let histogram = Arc::new(Histogram::<f64>::new(name));
        while histograms.len() <= index {
            histograms.push(Arc::new(Histogram::<f64>::new("")));
        }
        histograms[index] = Arc::clone(&histogram);
        histogram
    }

    /// Register the C4 exemplar cell for `name`, sharing the instrument index with
    /// its histogram. Idempotent: the same name returns the same cell, so the emit
    /// path and a reader both reach one `ExemplarCell`. Gaps are filled with fresh
    /// cells (`ExemplarCell` carries no name, so a placeholder is a usable cell).
    #[cfg(feature = "instrument-metrics")]
    pub fn register_exemplar(&self, name: &'static str) -> Arc<ExemplarCell> {
        let mut by_name = self.by_name.lock();
        let index = match by_name.get(name) {
            Some(&existing) => existing as usize,
            None => {
                let index = self.next.fetch_add(1, Ordering::Relaxed);
                by_name.insert(name, index);
                index as usize
            }
        };
        let mut exemplars = self.exemplars.lock();
        while exemplars.len() <= index {
            exemplars.push(Arc::new(ExemplarCell::new()));
        }
        Arc::clone(&exemplars[index])
    }

    /// Snapshot every non-zero instrument into export requests, resetting each as
    /// it goes (observations racing the swap appear in the NEXT window). Shared by
    /// the sync [`drain_instruments`](Self::drain_instruments) and async
    /// [`drain_instruments_async`](Self::drain_instruments_async) so both reset
    /// exactly once per pass — only the dispatch (block_on vs await) differs. Each
    /// mutex is held only for its Vec clone, never across a snapshot or dispatch.
    fn snapshot_instruments(&self, ts_ns: u64) -> Vec<crate::pipes::TelemetryRequest> {
        use crate::pipes::metric_request;

        // cheap atomic early-out: a trace/log-only recorder registers no
        // instruments, so skip the mutex locks + vec clones every drain pass.
        if self.next.load(Ordering::Relaxed) == 0 {
            return Vec::new();
        }

        let mut requests = Vec::new();

        let counters: Vec<Arc<Counter>> = {
            let guard = self.counters.lock();
            guard
                .iter()
                .filter(|counter| !counter.name.is_empty())
                .cloned()
                .collect()
        };
        for counter in &counters {
            let delta = counter.snapshot_and_reset();
            if delta > 0 {
                let sample = MetricSample::Counter(NumberDataPoint {
                    value: ScalarValue::U64(delta),
                    attrs: name_attrs(counter.name, counter.unit),
                    ts_ns,
                    start_ts_ns: 0,
                });
                requests.push(metric_request(sample));
            }
        }

        // gauges are last-value: report the current typed value every window (no
        // reset), so a steady gauge still publishes its level.
        let gauges: Vec<Arc<Gauge>> = {
            let guard = self.gauges.lock();
            guard
                .iter()
                .filter(|gauge| !gauge.name.is_empty())
                .cloned()
                .collect()
        };
        for gauge in &gauges {
            if let Some(value) = gauge.snapshot_if_changed() {
                let sample = MetricSample::Gauge(NumberDataPoint {
                    value,
                    attrs: name_attrs(gauge.name, gauge.unit),
                    ts_ns,
                    start_ts_ns: 0,
                });
                requests.push(metric_request(sample));
            }
        }

        // up-down counters are a cumulative signed sum: report the running total
        // every window (no reset), unlike the delta-temporality Counter above.
        let updowns: Vec<Arc<UpDownCounter>> = {
            let guard = self.updown_counters.lock();
            guard
                .iter()
                .filter(|updown| !updown.name.is_empty())
                .cloned()
                .collect()
        };
        for updown in &updowns {
            if let Some(value) = updown.snapshot_if_changed() {
                let sample = MetricSample::UpDownCounter(NumberDataPoint {
                    value: ScalarValue::I64(value),
                    attrs: name_attrs(updown.name, updown.unit),
                    ts_ns,
                    start_ts_ns: 0,
                });
                requests.push(metric_request(sample));
            }
        }

        #[cfg(feature = "histogram")]
        {
            use crate::metric::HistogramDataPoint;
            use crate::metric::histogram::MAX_BUCKETS;

            let histograms: Vec<Arc<Histogram<f64>>> = {
                let guard = self.histograms.lock();
                guard
                    .iter()
                    .filter(|histogram| !histogram.name.is_empty())
                    .cloned()
                    .collect()
            };
            for histogram in &histograms {
                let (count, sum_bits, bucket_counts) = histogram.snapshot_and_reset();
                if count > 0 {
                    let sample = MetricSample::Histogram(HistogramDataPoint {
                        count,
                        sum: f64::from_bits(sum_bits),
                        bucket_counts: bucket_counts[..MAX_BUCKETS].to_vec(),
                        bounds: &HISTOGRAM_DRAIN_BOUNDS,
                        attrs: name_attrs(histogram.name, histogram.unit),
                        ts_ns,
                        start_ts_ns: 0,
                    });
                    requests.push(metric_request(sample));
                }
            }
        }

        requests
    }

    /// Walk all registered instruments and dispatch non-zero snapshots via pipe.
    ///
    /// Called by the Drainer once per SYNC drain pass. Each instrument's atomics
    /// are swapped to 0 — observations racing the swap appear in the NEXT window.
    pub fn drain_instruments(
        &self,
        ts_ns: u64,
        pipe: &dyn proxima_primitives::pipe::alloc_tier::SendDynPipe<crate::pipes::TelemetryRequest, proxima_primitives::pipe::request::Response<bytes::Bytes>>,
    ) -> usize {
        let requests = self.snapshot_instruments(ts_ns);
        let total = requests.len();
        for request in requests {
            call_pipe_sync(pipe, request);
        }
        total
    }

    /// Async counterpart of [`drain_instruments`](Self::drain_instruments):
    /// `.await`s the terminal pipe instead of `block_on`-ing it, so a metrics
    /// recorder exports its counters/histograms over an async (network) sink
    /// driven from a prime executor thread — where a `block_on` would deadlock
    /// that executor. The prime drain pump calls this alongside
    /// [`Recorder::drain_async`](crate::recorder::Recorder::drain_async), which
    /// covers only the ring signals.
    pub async fn drain_instruments_async(
        &self,
        ts_ns: u64,
        pipe: &dyn proxima_primitives::pipe::alloc_tier::SendDynPipe<crate::pipes::TelemetryRequest, proxima_primitives::pipe::request::Response<bytes::Bytes>>,
    ) -> usize {
        let requests = self.snapshot_instruments(ts_ns);
        let total = requests.len();
        for request in requests {
            if let Err(error) = pipe.call_dyn(request).await {
                tracing::error!(error = %error, "pipe dispatch error during async instrument drain");
            }
        }
        total
    }
}

// the registry drops the instrument name when it snapshots into a sample;
// downstream pipes (cdb persist, otlp) need it to key the series, so carry it
// as a zero-alloc tag — name/unit are &'static str, no copy.
fn name_attrs(name: &'static str, unit: &'static str) -> smallvec::SmallVec<[Tag; 4]> {
    let mut attrs = smallvec::SmallVec::new();
    attrs.push(Tag::Scalar {
        key: "metric.name",
        value: ScalarValue::Str(name),
    });
    if !unit.is_empty() {
        attrs.push(Tag::Scalar {
            key: "metric.unit",
            value: ScalarValue::Str(unit),
        });
    }
    attrs
}

fn call_pipe_sync(
    pipe: &dyn proxima_primitives::pipe::alloc_tier::SendDynPipe<crate::pipes::TelemetryRequest, proxima_primitives::pipe::request::Response<bytes::Bytes>>,
    request: crate::pipes::TelemetryRequest,
) {
    let future = pipe.call_dyn(request);
    if let Err(error) = futures::executor::block_on(future) {
        tracing::error!(error = %error, "pipe dispatch error during instrument drain");
    }
}

// static bounds slice for drain-time histogram snapshots — base-2 exponential
// layout matching the default Histogram<f64> bucket configuration.
#[cfg(feature = "histogram")]
static HISTOGRAM_DRAIN_BOUNDS: [f64; 32] = [
    9.765_625e-4, // 2^-10
    1.953_125e-3, // 2^-9
    3.906_25e-3,  // 2^-8
    7.8125e-3,    // 2^-7
    0.015_625,    // 2^-6
    0.031_25,     // 2^-5
    0.0625,       // 2^-4
    0.125,        // 2^-3
    0.25,         // 2^-2
    0.5,          // 2^-1
    1.0,          // 2^0
    2.0,          // 2^1
    4.0,          // 2^2
    8.0,          // 2^3
    16.0,         // 2^4
    32.0,         // 2^5
    64.0,         // 2^6
    128.0,        // 2^7
    256.0,        // 2^8
    512.0,        // 2^9
    1_024.0,      // 2^10
    2_048.0,      // 2^11
    4_096.0,      // 2^12
    8_192.0,      // 2^13
    16_384.0,     // 2^14
    32_768.0,     // 2^15
    65_536.0,     // 2^16
    131_072.0,    // 2^17
    262_144.0,    // 2^18
    524_288.0,    // 2^19
    1_048_576.0,  // 2^20
    2_097_152.0,  // 2^21
];

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use core::sync::atomic::Ordering;

    use crate::pipes::{CountingPipe, into_telemetry_handle};

    use super::InstrumentRegistry;

    // registered gauge + up-down snapshot to a metric request on the first drain,
    // then go QUIET (report-on-change) — so the managed drainer's drain-until-empty
    // loop terminates instead of spinning on a never-emptying instrument.
    #[test]
    fn gauge_and_updown_snapshot_then_quiet() {
        let registry = InstrumentRegistry::new();
        registry.register_gauge("temp").set_f64(36.6, &[]);
        registry.register_updown_counter("conns").add(5, &[]);

        let (pipe, _, _, _, metrics, _) = CountingPipe::new();
        let handle = into_telemetry_handle(pipe);

        // first drain: both changed -> two metric requests dispatched.
        assert_eq!(registry.drain_instruments(0, handle.as_ref()), 2);
        assert_eq!(metrics.load(Ordering::Relaxed), 2);

        // second drain: nothing changed -> zero (the loop-terminating property).
        assert_eq!(registry.drain_instruments(0, handle.as_ref()), 0);
        assert_eq!(metrics.load(Ordering::Relaxed), 2);

        // a change re-reports.
        registry.register_gauge("temp").set_f64(40.0, &[]);
        assert_eq!(registry.drain_instruments(0, handle.as_ref()), 1);
        assert_eq!(metrics.load(Ordering::Relaxed), 3);
    }
}
