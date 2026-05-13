//! Dead-simple telemetry capture for tests.
//!
//! The whole point: a human writes THREE lines to capture and assert on any
//! signal — spans, logs, metrics, events, links — with no builder boilerplate
//! and no remembering to drain.
//!
//! ```
//! # use proxima_telemetry::capture::capture;
//! # // capture() builds a real Recorder, using proxima-core's Ring/StaticRing
//! # // internally -- cfg-swapped to loom under `--features loom`, only usable
//! # // inside an actual loom::model(...) closure, which this doctest isn't.
//! # if cfg!(feature = "loom") { return; }
//! let tel = capture(|rec| {
//!     rec.span("load_user").tag("id", 42u64).start();
//!     rec.log().message("cache miss").emit();
//!     rec.counter("db.queries").add(1, &[]);
//! });
//! assert_eq!(tel.spans().len(), 1);
//! assert_eq!(tel.spans()[0].name, "load_user");
//! assert_eq!(tel.logs().len(), 1);
//! // >= 1, not ==: capture() subscribes to span-duration metrics when
//! // instrument-metrics is compiled in, so "load_user"'s own duration
//! // lands here alongside the explicit db.queries counter.
//! assert!(tel.metrics().len() >= 1);
//! // and when a test fails, SEE everything that was emitted:
//! println!("{}", tel.dump());
//! ```

use std::fmt::Write as _;
use std::string::String;
use std::vec::Vec;

use crate::log::LogRecord;
use crate::metric::MetricSample;
use crate::pipes::InMemoryPipe;
use crate::recorder::Recorder;
use crate::tag::{ScalarValue, Tag};
use crate::trace::status::Status;
use crate::trace::{EventRecord, SpanLink, SpanRecord};

/// Everything a recorder emitted during [`capture`], as typed records plus a
/// human-readable [`dump`](Captured::dump).
pub struct Captured {
    pipe: InMemoryPipe,
}

impl Captured {
    #[must_use]
    pub fn spans(&self) -> Vec<SpanRecord> {
        self.pipe.spans()
    }

    #[must_use]
    pub fn logs(&self) -> Vec<LogRecord> {
        self.pipe.logs()
    }

    #[must_use]
    pub fn metrics(&self) -> Vec<MetricSample> {
        self.pipe.metrics()
    }

    #[must_use]
    pub fn events(&self) -> Vec<EventRecord> {
        self.pipe.events()
    }

    #[must_use]
    pub fn links(&self) -> Vec<SpanLink> {
        self.pipe.links()
    }

    /// Total records captured across all signal kinds.
    #[must_use]
    pub fn total(&self) -> usize {
        self.pipe.total()
    }

    /// One readable line per record — for `println!` when a test fails so a
    /// human can see exactly what was emitted without decoding wire bytes.
    #[must_use]
    pub fn dump(&self) -> String {
        let mut out = String::new();
        for span in self.spans() {
            let _ = writeln!(
                out,
                "span  {:?} dur={}ns status={}{}",
                span.name,
                span.duration_ns,
                status_str(&span.status),
                tags(&span.attrs)
            );
        }
        for log in self.logs() {
            let _ = writeln!(
                out,
                "log   {} {}{}",
                log.level,
                body_str(&log.body),
                tags(&log.attrs)
            );
        }
        for metric in self.metrics() {
            let (kind, value) = metric_kv(&metric);
            let _ = writeln!(out, "metric {kind}={value}{}", metric_tags(&metric));
        }
        for event in self.events() {
            let _ = writeln!(out, "event {:?}{}", event.name, tags(&event.attrs));
        }
        for link in self.links() {
            let _ = writeln!(out, "link  {link:?}");
        }
        out
    }
}

/// Run `body` against a recorder whose telemetry is captured in memory, drain,
/// and hand back the records. No builder, no `core_count`, no manual drain.
// test/dev helper: a recorder that fails to build is a misconfiguration the
// caller must see immediately, so panic rather than swallow into an empty capture.
#[allow(clippy::expect_used)]
#[must_use]
pub fn capture(body: impl FnOnce(&Recorder)) -> Captured {
    let pipe = InMemoryPipe::new();
    let recorder = Recorder::builder()
        .pipe(pipe.clone())
        .core_count(1)
        .start()
        .expect("capture recorder build");
    // capture buffers every pillar — it IS a span-metric consumer, so subscribe.
    #[cfg(feature = "instrument-metrics")]
    recorder.enable_span_metrics();
    body(&recorder);
    recorder.drain();
    Captured { pipe }
}

fn status_str(status: &Status) -> &'static str {
    match status {
        Status::Unset => "unset",
        Status::Ok => "ok",
        Status::Error { .. } => "error",
    }
}

fn body_str(body: &crate::log::body::LogBody) -> String {
    match body {
        crate::log::body::LogBody::Text(text) => (*text).to_string(),
        other => std::format!("{other:?}"),
    }
}

fn metric_kv(sample: &MetricSample) -> (&'static str, String) {
    match sample {
        MetricSample::Counter(point) => ("counter", scalar(&point.value)),
        MetricSample::Gauge(point) => ("gauge", scalar(&point.value)),
        MetricSample::UpDownCounter(point) => ("updown", scalar(&point.value)),
        #[cfg(feature = "histogram")]
        MetricSample::Histogram(_) => ("histogram", String::new()),
    }
}

fn metric_tags(sample: &MetricSample) -> String {
    match sample {
        MetricSample::Counter(point)
        | MetricSample::Gauge(point)
        | MetricSample::UpDownCounter(point) => tags(&point.attrs),
        #[cfg(feature = "histogram")]
        MetricSample::Histogram(_) => String::new(),
    }
}

fn tags(attrs: &[Tag]) -> String {
    if attrs.is_empty() {
        return String::new();
    }
    let mut out = String::from(" {");
    for (index, tag) in attrs.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        match tag {
            Tag::Scalar { key, value } => {
                let _ = write!(out, "{key}={}", scalar(value));
            }
            Tag::Structured { key, .. } => {
                let _ = write!(out, "{key}=<structured>");
            }
        }
    }
    out.push('}');
    out
}

fn scalar(value: &ScalarValue) -> String {
    match value {
        ScalarValue::I64(raw) => raw.to_string(),
        ScalarValue::U64(raw) => raw.to_string(),
        ScalarValue::F64(raw) => raw.to_string(),
        ScalarValue::Bool(raw) => raw.to_string(),
        ScalarValue::Str(text) => (*text).to_string(),
        ScalarValue::Bytes(raw) => String::from_utf8_lossy(raw).into_owned(),
    }
}

// capture() builds a real Recorder, which uses proxima-core's Ring/
// StaticRing internally -- those are cfg-swapped to loom under
// `--features loom` (forwarded via proxima-core/loom) and only work
// inside an actual loom::model(...) closure, which these plain #[test]
// functions don't provide.
#[cfg(all(test, not(feature = "loom")))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn captures_a_span_with_attrs() {
        let tel = capture(|rec| {
            rec.span("load_user").tag("id", 42u64).start();
        });
        let spans = tel.spans();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].name, "load_user");
        assert!(spans[0].attrs.iter().any(|tag| matches!(
            tag,
            Tag::Scalar {
                key: "id",
                value: ScalarValue::U64(42)
            }
        )));
    }

    #[test]
    fn captures_logs_and_metrics_together() {
        let tel = capture(|rec| {
            rec.log().message("cache miss").emit();
            rec.counter("db.queries").add(3, &[]);
            rec.span("op").start();
        });
        assert_eq!(tel.logs().len(), 1);
        assert_eq!(tel.spans().len(), 1);
        // Unified instrument: with the `instrument-metrics` seam on, the span "op"
        // also registers+records a duration histogram, which drains as a second
        // metric (alongside the db.queries counter). So the metric/total counts are
        // feature-dependent — that extra metric IS the point of the seam.
        #[cfg(not(feature = "instrument-metrics"))]
        {
            assert_eq!(tel.metrics().len(), 1);
            assert_eq!(tel.total(), 3);
        }
        #[cfg(feature = "instrument-metrics")]
        {
            assert_eq!(
                tel.metrics().len(),
                2,
                "db.queries counter + span-duration histogram"
            );
            assert_eq!(tel.total(), 4);
        }
    }

    #[test]
    fn empty_capture_is_empty() {
        let tel = capture(|_rec| {});
        assert_eq!(tel.total(), 0);
        assert!(tel.dump().is_empty());
    }

    // repro: does the #[instrument] MACRO (not recorder.span()) record a duration
    // metric? Explicit recorder = rec isolates the macro/guard path from ambient
    // current() resolution.
    #[cfg(feature = "instrument-metrics")]
    #[test]
    fn instrument_macro_records_duration_metric() {
        #[crate::instrument(recorder = rec)]
        fn instrumented(rec: &Recorder) -> u32 {
            42
        }
        let tel = capture(|rec| {
            let _ = instrumented(rec);
        });
        assert_eq!(
            tel.metrics().len(),
            1,
            "instrument macro must record a duration histogram; dump: {}",
            tel.dump()
        );
        assert_eq!(tel.spans().len(), 1, "and a trace span");
    }

    // repro for the ambient path: an installed process-default recorder is the
    // #[instrument] target AND a span-metric consumer (its drain exports the
    // histogram), so #[instrument] must record a duration without a manual
    // enable_span_metrics(). nextest isolates each test in its own process, so the
    // process-global default recorder here does not leak.
    #[cfg(feature = "instrument-metrics")]
    #[test]
    fn ambient_install_records_instrument_duration() {
        #[crate::instrument]
        fn ambient_work() -> u32 {
            7
        }

        let pipe = InMemoryPipe::new();
        let recorder = Recorder::builder()
            .pipe(pipe.clone())
            .core_count(1)
            .start()
            .expect("recorder build");
        crate::export::set_default_recorder(alloc::sync::Arc::new(recorder));

        ambient_work();

        // drain to empty, then assert the recorder EXPORTED a duration histogram.
        // Tier-agnostic: the inline path folds on emit, the deferred path folds at
        // drain (Block/producer-assist, lossless); either way the drain snapshots
        // the histogram to the pipe. Reading the live `count()` would race the
        // snapshot-and-reset, so assert on the exported sample.
        let ambient = Recorder::current().expect("ambient recorder installed");
        while ambient.drain() > 0 {}
        let captured = Captured { pipe };
        assert!(
            captured
                .metrics()
                .iter()
                .any(|sample| matches!(sample, MetricSample::Histogram(_))),
            "installed ambient recorder must export a #[instrument] duration histogram; dump: {}",
            captured.dump()
        );
    }

    #[test]
    fn dump_is_human_readable() {
        let tel = capture(|rec| {
            rec.span("load_user").tag("id", 42u64).start();
            rec.counter("db.queries").add(1, &[]);
        });
        let dump = tel.dump();
        assert!(dump.contains("span  \"load_user\""), "dump: {dump}");
        assert!(dump.contains("id=42"), "dump: {dump}");
        assert!(dump.contains("metric counter=1"), "dump: {dump}");
    }
}
