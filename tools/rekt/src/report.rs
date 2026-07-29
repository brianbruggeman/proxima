//! Measurements go into telemetry; the report reads them back out.
//!
//! There is no recorder type here any more. `Recorder<Out>` was a bespoke
//! accumulator — an unbounded `Vec<f64>` of every latency, a hand-rolled sorted
//! bucket list, and a hand-rolled percentile — sitting next to
//! `proxima_telemetry::Metrics`, which is the same thing done properly:
//!
//! - **keyed**, `DashMap<(name, Labels), _>`, so "count by status" is a label
//!   rather than a `Vec<(Out, u64)>` rekt maintains itself;
//! - **bounded and precise**, hdrhistogram at 3 significant digits rather than
//!   8 bytes of resident memory per arrival;
//! - **shard-per-thread**, `ThreadLocal<HistogramShard>` merged on read, so
//!   writers on different cores never collide and there is no tally to ship
//!   over an mpsc and sum.
//!
//! `MetricsSnapshot`/`HistogramSummary` already carry exactly what a load report
//! needs (count, min, max, mean, p50/p90/p99/p99_9 per keyed series), so the
//! read side is a projection, not a computation.
//!
//! **Latency is recorded in MICROSECONDS.** `Telemetry::histogram_record` takes
//! `f64` and immediately does `value.round() as u64`; recording milliseconds
//! would floor every sub-millisecond latency to zero, which for a load
//! generator is most of them.

use core::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

#[cfg(feature = "scheduler")]
use bytes::Bytes;
#[cfg(feature = "scheduler")]
use proxima::runtime::Runtime;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::fanout::{FanOut, IgnoreErrors};
#[cfg(feature = "scheduler")]
use proxima_recording::pipe::{AccumulatingSink, BoundedRecordingSink, DynRecordingSink, FailMode, FormatKind, LazyFanOut, RECORD_DROP_METRIC, RecordingSink, SinkSpec, deferred_runtime};
#[cfg(feature = "scheduler")]
use proxima_recording::{InteractionId, ProtocolEvent, RecordingEvent};
#[cfg(feature = "scheduler")]
use proxima_telemetry::TelemetryHandle;
use proxima_telemetry::{HistogramSummary, Labels, Metrics, MetricsSnapshot, Telemetry};

#[cfg(feature = "scheduler")]
use crate::error::Error;
#[cfg(feature = "scheduler")]
use crate::scenario::Dump;
use crate::scenario::LoadPlan;

/// Metric names, in one place so the report reads back exactly what the loop
/// wrote.
const LATENCY: &str = "rekt.latency";
const REPLIES: &str = "rekt.replies";
const FAILURES: &str = "rekt.failures";
const ELAPSED: &str = "rekt.elapsed_nanos";
const CONNECTIONS: &str = "rekt.connections";
const CORES: &str = "rekt.cores";

/// The measurement store for a run. `Metrics::default()` is 3 significant
/// digits up to 60_000_000 — sixty seconds' worth of microseconds.
#[must_use]
pub fn store() -> Metrics {
    Metrics::default()
}

/// One observation, in the form every sink can take.
///
/// The reply is rendered to its bucket key here and nowhere else, because a fan
/// must hand the *same* observation to every arm — and `Conn::Out`/`Conn::Err`
/// are neither `Clone` nor uniform across protocols. This keeps both sides of
/// the monad (`Ok` bucket vs `Err` bucket); it is a rendering, not a reduction.
#[derive(Clone)]
pub struct Observed {
    pub scenario: Arc<str>,
    pub stage: usize,
    pub latency: Duration,
    /// The reply's bucket key, or the failure's.
    pub bucket: Result<Arc<str>, Arc<str>>,
}

impl Observed {
    /// Render a pipe's reply into an observation.
    pub fn of<Out, Err>(scenario: &Arc<str>, stage: usize, latency: Duration, reply: Result<Out, Err>) -> Self
    where
        Out: core::fmt::Debug,
        Err: core::fmt::Debug,
    {
        Self {
            scenario: Arc::clone(scenario),
            stage,
            latency,
            bucket: match reply {
                Ok(out) => Ok(Arc::from(format!("{out:?}").as_str())),
                Err(failure) => Err(Arc::from(format!("{failure:?}").as_str())),
            },
        }
    }
}

/// One arm of the measurement fan.
///
/// An enum rather than `Arc<dyn SendDynPipe<_, _>>`: the arms are a closed set,
/// and rust.md's box-free rule says a discriminated enum + match before a boxed
/// trait object. `FanOut<S, Policy>` holds `Vec<S>` — homogeneous by design, so
/// that it stays monomorphised with no dyn dispatch on the hot path — which
/// means the arms must share a type anyway.
///
/// Each arm owns the failure policy appropriate to it, which is what "a backoff
/// strategy per fan" resolves to for an instrument:
///
/// - [`Sink::Series`] cannot fail. Atomics and a `DashMap`; there is nothing to
///   back off from.
/// - [`Sink::Window`] cannot fail either, and is deliberately *selective*: only
///   replies that arrived, because a target failing fast must not read as faster
///   to the hillclimb controller.
/// - a durable arm (the dump) is bounded-queue-with-drop, NOT inline retry.
///   Retrying inside the send loop would park the load generator on a slow disk
///   and silently lower the offered rate — the instrument perturbing what it
///   measures. `Backoff`/`Jitter` belong on the drain side, off this path.
#[derive(Clone)]
pub enum Sink {
    /// The run's report: every observation, bucketed.
    Series(Arc<Metrics>),
    /// The adaptive controller's window: arrived-reply latencies only.
    Window(Arc<Metrics>),
    /// Durable capture of every arrival, behind a bounded queue.
    ///
    /// `workload` is the id of the event that recorded WHAT this connection
    /// sends. rekt sends the same prepared item every arrival, so the payload is
    /// dumped once and each arrival links back to it by `parent` — the field
    /// `RecordingEvent` has for exactly this. Carrying the bytes per arrival
    /// would be the same value written a million times.
    #[cfg(feature = "scheduler")]
    Dump { sink: Arc<BoundedRecordingSink>, workload: InteractionId },
}

impl SendPipe for Sink {
    type In = Observed;
    type Out = ();
    type Err = Infallible;

    async fn call(&self, observed: Observed) -> Result<(), Infallible> {
        match self {
            Sink::Series(metrics) => write_series(metrics, &observed),
            Sink::Window(metrics) => {
                if observed.bucket.is_ok() {
                    metrics.histogram_record(LATENCY, &Labels::empty(), micros_of(observed.latency));
                }
            }
            // `append` returns the queue's verdict; a full queue is a DROP, not a
            // stall, and the drop is already counted in `record_dropped_total`.
            // Swallowing it here is the policy, not an oversight: an instrument
            // that blocks on its own dump has stopped measuring the target.
            #[cfg(feature = "scheduler")]
            Sink::Dump { sink, workload } => {
                let _ = sink
                    .append(arrival_event(&observed, *workload))
                    .await;
            }
        }
        Ok(())
    }
}

#[cfg(feature = "scheduler")]
/// One arrival as a `RecordingEvent`.
///
/// `ProtocolEvent::Custom` rather than `HttpEvent`: the h1 raw path returns a
/// status and never materialises a `Response`, so an `HttpEvent::ResponseStarted`
/// would have to invent headers it never read. `Custom` is the event enum's own
/// extension point and records exactly what rekt actually observed.
fn arrival_event(observed: &Observed, workload: InteractionId) -> RecordingEvent {
    let (outcome, bucket) = match &observed.bucket {
        Ok(bucket) => ("reply", bucket),
        Err(bucket) => ("no_reply", bucket),
    };
    RecordingEvent {
        id: InteractionId::new(),
        ts_ms: 0,
        parent: Some(workload),
        event: ProtocolEvent::Custom {
            kind: "rekt.arrival".to_string(),
            payload: serde_json::json!({
                "scenario": observed.scenario.as_ref(),
                "stage": observed.stage,
                "latency_us": micros_of(observed.latency) as u64,
                "outcome": outcome,
                "bucket": bucket.as_ref(),
            }),
        },
    }
}

#[cfg(feature = "scheduler")]
/// Build the durable dump chain a [`Sink::Dump`] arm wraps.
///
/// The shape is proxima-recording's own: a spigot-gated `LazyFanOut` terminal
/// (no file is opened until the first event, so an unused dump costs nothing), an
/// `AccumulatingSink` coalescing events into blocks, and a
/// `BoundedRecordingSink` in front holding the queue bound and the drop policy.
pub fn dump_sink(spec: &Dump, runtime: Arc<dyn Runtime>, metrics: &Arc<Metrics>) -> Result<Arc<BoundedRecordingSink>, Error> {
    let format = match spec.format.as_str() {
        "bin" => FormatKind::Bin,
        "json" => FormatKind::Json,
        other => return Err(Error::Config(format!("dump format {other:?}: want \"bin\" or \"json\""))),
    };
    let on_full = match spec.on_full.as_str() {
        "drop_newest" => FailMode::DropNewest,
        "drop_oldest" => FailMode::DropOldest,
        "fail_closed" => FailMode::FailClosed,
        other => {
            return Err(Error::Config(format!("dump on_full {other:?}: want \"drop_newest\", \"drop_oldest\" or \"fail_closed\"")));
        }
    };

    let spigot = deferred_runtime();
    let _ = spigot.set(runtime);
    let durable = Arc::new(LazyFanOut::new(vec![SinkSpec::new(spec.path.clone(), format)], spigot));
    let batched: DynRecordingSink = Arc::new(AccumulatingSink::new(durable, spec.batch.max(1)));

    // `with_telemetry`, not `new`: `new` sends drops to `NoopTelemetry`, so a
    // queue that overflows loses events SILENTLY and the log looks complete. The
    // drop counter goes into the run's own store, which makes the loss part of
    // the report — `dumped + dropped == arrivals` is then a checkable invariant
    // rather than a hope.
    Ok(Arc::new(BoundedRecordingSink::with_telemetry(
        batched,
        spec.capacity.max(1),
        on_full,
        Arc::clone(metrics) as TelemetryHandle,
        Labels::from_pairs(&[("sink", "dump")]),
    )))
}

/// The measurement fan: one observation, every arm, and a broken arm can neither
/// fail nor stall the load loop.
///
/// `IgnoreErrors` is the load-generator policy. `AllOrNothing` would let a full
/// disk abort a run and `BestEffort` would surface the error into the send path;
/// an instrument must keep measuring while a sink is unhappy, and say so in its
/// own drop counter instead.
pub type Fan = FanOut<Sink, IgnoreErrors>;

/// The window's rtt distribution — the controller's input.
#[must_use]
pub fn window_rtt(window: &Metrics) -> Option<HistogramSummary> {
    window.histogram_summary(LATENCY, &Labels::empty())
}

/// The fan a staged run records through.
#[must_use]
pub fn series_fan(metrics: &Arc<Metrics>) -> Fan {
    FanOut::new(vec![Sink::Series(Arc::clone(metrics))])
}

#[cfg(feature = "scheduler")]
/// Record WHAT a connection will send, once, and return the id arrivals link to.
///
/// This is the "what we sent" half of the dump. It is one event per connection
/// rather than one per arrival because the item is prepared once and re-sent
/// unchanged — writing it a million times would say nothing new.
pub async fn dump_workload(sink: &Arc<BoundedRecordingSink>, scenario: &str, sent: &Bytes) -> InteractionId {
    let workload = InteractionId::new();
    let _ = sink
        .append(RecordingEvent {
            id: workload,
            ts_ms: 0,
            parent: None,
            event: ProtocolEvent::Custom {
                kind: "rekt.workload".to_string(),
                payload: serde_json::json!({
                    "scenario": scenario,
                    "bytes": String::from_utf8_lossy(sent).into_owned(),
                }),
            },
        })
        .await;
    workload
}

#[cfg(feature = "scheduler")]
/// The same fan with a durable capture arm appended.
#[must_use]
pub fn dumping_fan(metrics: &Arc<Metrics>, sink: &Arc<BoundedRecordingSink>, workload: InteractionId) -> Fan {
    FanOut::new(vec![Sink::Series(Arc::clone(metrics)), Sink::Dump { sink: Arc::clone(sink), workload }])
}

/// The fan the adaptive drive records through: the run's report and the
/// controller's window, from one call.
#[must_use]
pub fn windowed_fan(run: &Arc<Metrics>, window: &Arc<Metrics>) -> Fan {
    FanOut::new(vec![Sink::Series(Arc::clone(run)), Sink::Window(Arc::clone(window))])
}

fn write_series(metrics: &Metrics, observed: &Observed) {
    let stage_label = observed.stage.to_string();
    let keys: [(&str, &str); 2] = [("scenario", &observed.scenario), ("stage", &stage_label)];
    let micros = micros_of(observed.latency);

    metrics.histogram_record(LATENCY, &Labels::from_pairs(&keys), micros);
    metrics.histogram_record(LATENCY, &Labels::empty(), micros);

    match &observed.bucket {
        Ok(bucket) => metrics.counter_inc(REPLIES, &Labels::from_pairs(&[keys[0], keys[1], ("reply", bucket)]), 1),
        Err(bucket) => metrics.counter_inc(FAILURES, &Labels::from_pairs(&[keys[0], keys[1], ("failure", bucket)]), 1),
    }
}

fn micros_of(latency: Duration) -> f64 {
    latency.as_secs_f64() * 1_000_000.0
}

/// The scenario name the throughput drivers record under. They have no scenario
/// file — the whole run is one workload — but the label keeps their series the
/// same shape as a staged run's, which is what let `Throughput` go away.
pub const RUN: &str = "run";

/// Gauges describing the shape of the run, stamped once when it ends.
///
/// `Throughput` used to be a struct carrying these plus `completed`/`errors`.
/// The counts are counters and the shape is gauges, so the struct was a fifth
/// copy of data telemetry already held — and its `completed`/`errors` pair was
/// the reply-collapse that made a 500 read as clean throughput.
pub fn stamp_run(metrics: &Metrics, elapsed: Duration, connections: usize, cores: usize) {
    let nanos = i64::try_from(elapsed.as_nanos()).unwrap_or(i64::MAX);
    metrics.gauge_set(ELAPSED, &Labels::empty(), nanos);
    metrics.gauge_set(CONNECTIONS, &Labels::empty(), i64::try_from(connections).unwrap_or(i64::MAX));
    metrics.gauge_set(CORES, &Labels::empty(), i64::try_from(cores).unwrap_or(i64::MAX));
}

/// A connection that never opened: no reply, and no latency to record either.
pub fn record_setup_failure(metrics: &Metrics, reason: &str) {
    metrics.counter_inc(FAILURES, &Labels::from_pairs(&[("scenario", RUN), ("stage", "0"), ("failure", reason)]), 1);
}

/// Replies per second over the stamped run duration.
///
/// Reads back out of telemetry rather than being a method on a result struct —
/// `completed / elapsed` where both are recorded series.
#[must_use]
pub fn per_sec(metrics: &Metrics) -> f64 {
    let elapsed = metrics
        .gauge(ELAPSED, &Labels::empty())
        .unwrap_or(0);
    if elapsed <= 0 {
        return 0.0;
    }
    let seconds = elapsed as f64 / 1_000_000_000.0;
    completed(metrics) as f64 / seconds
}

/// Total replies that arrived, across every bucket.
#[must_use]
pub fn completed(metrics: &Metrics) -> u64 {
    total_of(metrics, REPLIES)
}

/// Arrivals the dump could not keep up with.
///
/// Non-zero means the capture is incomplete and by exactly how much — the
/// difference between a lossy instrument and a lying one.
#[cfg(feature = "scheduler")]
#[must_use]
pub fn dump_dropped(metrics: &Metrics) -> u64 {
    metrics
        .counter(RECORD_DROP_METRIC, &Labels::from_pairs(&[("sink", "dump")]))
        .or_else(|| {
            metrics
                .snapshot()
                .counters
                .iter()
                .find(|(name, _, _)| name == RECORD_DROP_METRIC)
                .map(|(_, _, count)| *count)
        })
        .unwrap_or(0)
}

/// Total arrivals that got nothing back.
#[must_use]
pub fn failed(metrics: &Metrics) -> u64 {
    total_of(metrics, FAILURES)
}

fn total_of(metrics: &Metrics, metric: &str) -> u64 {
    metrics
        .snapshot()
        .counters
        .iter()
        .filter(|(name, _, _)| name == metric)
        .map(|(_, _, count)| count)
        .sum()
}

/// Render the recorded series, and say whether the run passed.
///
/// No report type: the data is already in `Metrics`, keyed and summarised, so a
/// `Report`/`StageReport` pair would only be a second copy of it shaped for
/// printing. This walks the same series once and emits the text plus the
/// verdict — the only two things a caller actually wants.
#[must_use]
pub fn render(metrics: &Metrics, plan: &LoadPlan) -> (String, bool) {
    let snapshot = metrics.snapshot();
    let mut out = String::new();
    let mut total = 0u64;
    let mut total_failures = 0u64;

    for scenario in &plan.scenarios {
        out.push_str(&format!("{} -> {}\n", scenario.name, scenario.url));
        for stage in 0..scenario.stages.len() {
            let stage_label = stage.to_string();
            let keys: [(&str, &str); 2] = [("scenario", &scenario.name), ("stage", &stage_label)];
            let latency = metrics.histogram_summary(LATENCY, &Labels::from_pairs(&keys));
            let count = latency
                .as_ref()
                .map_or(0, |summary| summary.count);
            let percentile = |pick: fn(&HistogramSummary) -> f64| fmt_micros(latency.as_ref().map_or(0.0, pick));

            out.push_str(&format!(
                "  stage {stage}: {count} sent  p50 {p50}  p90 {p90}  p95 {p95}  p99 {p99}  p999 {p999}\n",
                p50 = percentile(|summary| summary.p50),
                p90 = percentile(|summary| summary.p90),
                p95 = percentile(|summary| summary.p95),
                p99 = percentile(|summary| summary.p99),
                p999 = percentile(|summary| summary.p99_9),
            ));
            for (bucket, hits) in counters_for(&snapshot, REPLIES, "reply", &keys) {
                out.push_str(&format!("    {bucket}: {hits}\n"));
            }
            for (bucket, hits) in counters_for(&snapshot, FAILURES, "failure", &keys) {
                out.push_str(&format!("    no reply ({bucket}): {hits}\n"));
                total_failures += hits;
            }
            total += count;
        }
    }

    let overall_p99 = metrics
        .histogram_summary(LATENCY, &Labels::empty())
        .map_or(0.0, |summary| summary.p99);
    let failure_rate = ratio(total_failures, total);
    let passed = plan
        .thresholds
        .p99
        .is_none_or(|limit| micros_to_duration(overall_p99) <= limit)
        && plan
            .thresholds
            .failure_rate
            .is_none_or(|limit| failure_rate <= limit);

    out.push_str(&format!(
        "overall: p99 {p99}  no reply {rate:.2}%  -> {verdict}\n",
        p99 = fmt_micros(overall_p99),
        rate = failure_rate * 100.0,
        verdict = if passed { "pass" } else { "fail" },
    ));
    (out, passed)
}

/// One stage's slice of a counter series, keyed by `dimension`, sorted.
fn counters_for(snapshot: &MetricsSnapshot, metric: &str, dimension: &str, keys: &[(&str, &str); 2]) -> Vec<(String, u64)> {
    let mut out: Vec<(String, u64)> = snapshot
        .counters
        .iter()
        .filter(|(name, labels, _)| {
            name == metric
                && keys
                    .iter()
                    .all(|(key, value)| label(labels, key) == Some(*value))
        })
        .filter_map(|(_, labels, count)| label(labels, dimension).map(|bucket| (bucket.to_string(), *count)))
        .collect();
    out.sort();
    out
}

fn label<'labels>(labels: &'labels Labels, name: &str) -> Option<&'labels str> {
    labels
        .entries()
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.as_str())
}

fn ratio(num: u64, den: u64) -> f64 {
    if den == 0 { 0.0 } else { num as f64 / den as f64 }
}

fn micros_to_duration(value: f64) -> Duration {
    Duration::from_secs_f64((value / 1_000_000.0).max(0.0))
}

fn fmt_micros(value: f64) -> String {
    format!("{:.1}ms", value / 1000.0)
}

/// Read one reply bucket back, for callers that want the number rather than the
/// text — the tests, and anything scripting a run.
#[must_use]
pub fn replies(metrics: &Metrics, scenario: &str, stage: usize, bucket: &str) -> u64 {
    metrics
        .counter(REPLIES, &Labels::from_pairs(&[("scenario", scenario), ("stage", &stage.to_string()), ("reply", bucket)]))
        .unwrap_or(0)
}

/// Arrivals recorded for a stage.
#[must_use]
pub fn arrivals(metrics: &Metrics, scenario: &str, stage: usize) -> u64 {
    metrics
        .histogram_summary(LATENCY, &Labels::from_pairs(&[("scenario", scenario), ("stage", &stage.to_string())]))
        .map_or(0, |summary| summary.count)
}

#[cfg(test)]
mod tests {
    // asserting on a recorded series; unwrap/expect is the clearer failure here
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// Tests record the way the drivers do — through a fan, not a back door.
    fn observe(metrics: &Arc<Metrics>, stage: usize, latency: Duration, reply: Result<u16, Error>) {
        let fan = series_fan(metrics);
        let scenario: Arc<str> = Arc::from("load");
        futures::executor::block_on(fan.call(Observed::of(&scenario, stage, latency, reply))).expect("infallible");
    }
    use crate::error::Error;
    use crate::scenario::{Arrival, PayloadSpec, Scenario, Stage, Thresholds};

    fn open() -> Thresholds {
        Thresholds { p99: None, failure_rate: None }
    }

    /// The smallest plan that names one scenario with one stage — enough for
    /// `render` to know what series to walk.
    fn plan_of(thresholds: &Thresholds) -> LoadPlan {
        LoadPlan::builder()
            .thresholds(thresholds.clone())
            .scenarios(vec![Scenario {
                name: "load".to_string(),
                url: "http://127.0.0.1:8080/".to_string(),
                payload: PayloadSpec::default(),
                stages: vec![Stage {
                    rate_per_sec: None,
                    duration: Duration::from_secs(1),
                    arrival: Arrival::Even,
                }],
            }])
            .build()
    }

    fn replied(status: u16) -> Result<u16, Error> {
        Ok(status)
    }

    fn no_reply() -> Result<u16, Error> {
        Err(Error::Config("connection reset".into()))
    }

    #[test]
    fn every_distinct_status_gets_its_own_bucket() {
        let metrics = Arc::new(store());
        for _ in 0..90 {
            observe(&metrics, 0, Duration::from_millis(10), replied(200));
        }
        for _ in 0..7 {
            observe(&metrics, 0, Duration::from_millis(10), replied(500));
        }
        observe(&metrics, 0, Duration::from_millis(10), replied(501));
        observe(&metrics, 0, Duration::from_millis(10), replied(403));
        observe(&metrics, 0, Duration::from_millis(30), no_reply());

        assert_eq!(arrivals(&metrics, "load", 0), 100);
        assert_eq!(replies(&metrics, "load", 0, "200"), 90);
        assert_eq!(replies(&metrics, "load", 0, "500"), 7);
        assert_eq!(replies(&metrics, "load", 0, "501"), 1);
        assert_eq!(replies(&metrics, "load", 0, "403"), 1);

        let (text, _) = render(&metrics, &plan_of(&open()));
        assert!(text.contains("no reply ("), "the Err side buckets on its own");
    }

    #[test]
    fn a_target_answering_500_to_everything_is_visible() {
        let metrics = Arc::new(store());
        for _ in 0..100 {
            observe(&metrics, 0, Duration::from_millis(1), replied(500));
        }

        let (text, passed) = render(&metrics, &plan_of(&open()));
        assert!(text.contains("500: 100"));
        assert!(
            text.contains("no reply 0.00%"),
            "every request DID get a reply — a true statement; the bucket is what says the run was bad"
        );
        assert!(passed, "with no thresholds set, nothing here decides a 500 is a failure");
    }

    #[test]
    fn sub_millisecond_latency_survives_the_histogram() {
        // the unit trap: `histogram_record` rounds f64 to u64, so recording
        // milliseconds would floor 200us to 0 and report p50 = 0.0ms for a fast
        // target — which is most of them.
        let metrics = Arc::new(store());
        for _ in 0..1000 {
            observe(&metrics, 0, Duration::from_micros(200), replied(200));
        }

        let summary = metrics
            .histogram_summary(LATENCY, &Labels::from_pairs(&[("scenario", "load"), ("stage", "0")]))
            .expect("recorded");
        assert!((190.0..=210.0).contains(&summary.p50), "p50 was {}us, expected ~200us", summary.p50);
    }

    #[test]
    fn p99_threshold_fails_the_run() {
        let metrics = Arc::new(store());
        for _ in 0..98 {
            observe(&metrics, 0, Duration::from_millis(10), replied(200));
        }
        observe(&metrics, 0, Duration::from_millis(500), replied(200));
        observe(&metrics, 0, Duration::from_millis(500), replied(200));

        let (_, passed) = render(
            &metrics,
            &plan_of(&Thresholds {
                p99: Some(Duration::from_millis(250)),
                failure_rate: None,
            }),
        );
        assert!(!passed);
    }

    #[test]
    fn the_failure_rate_counts_arrivals_that_got_nothing_back() {
        let metrics = Arc::new(store());
        for _ in 0..99 {
            observe(&metrics, 0, Duration::from_millis(10), replied(200));
        }
        observe(&metrics, 0, Duration::from_millis(30), no_reply());

        let (text, passed) = render(&metrics, &plan_of(&Thresholds { p99: None, failure_rate: Some(0.005) }));
        assert!(text.contains("no reply 1.00%"));
        assert!(!passed);
    }

    #[test]
    fn stages_stay_separate_series() {
        let metrics = Arc::new(store());
        for _ in 0..3 {
            observe(&metrics, 0, Duration::from_millis(1), replied(200));
        }
        for _ in 0..5 {
            observe(&metrics, 1, Duration::from_millis(1), replied(200));
        }

        assert_eq!(arrivals(&metrics, "load", 0), 3);
        assert_eq!(arrivals(&metrics, "load", 1), 5);
    }

    #[test]
    fn clean_run_passes() {
        let metrics = Arc::new(store());
        for _ in 0..100 {
            observe(&metrics, 0, Duration::from_millis(10), replied(200));
        }

        let (_, passed) = render(
            &metrics,
            &plan_of(&Thresholds {
                p99: Some(Duration::from_millis(250)),
                failure_rate: Some(0.01),
            }),
        );
        assert!(passed);
    }
}
