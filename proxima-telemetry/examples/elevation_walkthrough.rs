#![allow(clippy::unwrap_used, clippy::expect_used)]
// error-elevation, walked by hand.
//
// The problem: at INFO in production you don't see the TRACE/DEBUG breadcrumbs
// that would explain a failure, because turning those levels on for everyone,
// all the time, is too expensive and too noisy. But if you only turn them on
// AFTER you see the error, the breadcrumbs that led up to it are already gone.
//
// error-elevation resolves that by buffering, not by widening the live floor:
// - a FLOOR level (say `info`) always reaches the normal exporter, unchanged.
// - a small SAMPLED fraction of traces (`sample_ratio`) are also admitted to
//   VERBOSE-BUFFERED mode: their below-floor records (down to `elevated`, say
//   `trace`) are still built and pushed, but into a bounded per-trace ring
//   instead of the normal exporter.
// - if that trace ever emits a record at or above `trigger_level` (default
//   `error`), the WHOLE buffered tree — floor+ and below-floor, in time order —
//   replays to a separate ELEVATED exporter. A healthy sampled trace's buffer
//   is just dropped when its root span closes; nothing extra was ever sent.
//
// So the cost you pay is bounded to the sampled fraction, and only traces that
// actually go wrong ever produce their verbose tree anywhere.
//
// Run: cargo run -p proxima-telemetry --features elevation --example elevation_walkthrough

use std::sync::{Arc, Mutex};

use bytes::Bytes;
use conflaguration::Validate;
use futures::executor::block_on;
use proxima_primitives::pipe::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::request::Response;
use proxima_telemetry::config::{Elevation, ExporterChoice, Retention, TelemetryConfig};
use proxima_telemetry::id::{SpanId, TraceFlags, TraceId};
use proxima_telemetry::level::Level;
use proxima_telemetry::log::{LogBody, LogRecord};
use proxima_telemetry::pipes::{
    ElevationSink, TelemetryRecord, TelemetryRequest, into_telemetry_handle, log_batch_request,
};

// the elevated sink under test: a terminal Pipe that just remembers every
// LogRecord it is handed. In production this arm would be a real exporter
// (OTLP, a forensic file sink, whatever); a real ElevationSink doesn't care —
// it only needs `SendPipe<In = TelemetryRequest>`. This mirrors the `Capture`
// pipe in `src/pipes.rs`'s `mod elevation_sink_tests`.
#[derive(Clone)]
struct Capture {
    seen: Arc<Mutex<Vec<LogRecord>>>,
}

impl Capture {
    fn new() -> Self {
        Self {
            seen: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn records(&self) -> Vec<LogRecord> {
        self.seen.lock().expect("capture lock").clone()
    }
}

impl SendPipe for Capture {
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: TelemetryRequest,
    ) -> impl std::future::Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let seen = Arc::clone(&self.seen);
        async move {
            if let TelemetryRecord::LogBatch(records) = &request.payload {
                seen.lock().expect("capture lock").extend(records.iter().cloned());
            }
            Ok(Response::ok(Bytes::new()))
        }
    }
}

// a record as it looks when it was actually emitted inside a verbose-sampled
// trace: SAMPLED (W3C) + VERBOSE_BUFFERED (proxima-local, stamped only when the
// trace was chosen by the verbose sampler) carrying its trace id and the
// wall-clock-free `ts_ns` that orders the eventual replay.
fn verbose_log(trace: TraceId, level: Level, ts_ns: u64, message: &'static str) -> LogRecord {
    LogRecord {
        ts_ns,
        observed_ts_ns: ts_ns,
        level,
        body: LogBody::Text(message),
        attrs: smallvec::SmallVec::new(),
        trace_id: Some(trace),
        span_id: Some(SpanId::from_bytes([1; 8])),
        trace_flags: TraceFlags::SAMPLED.with_verbose_buffered(),
        module_path: "elevation_walkthrough",
        file_line: (0, 0),
    }
}

fn main() {
    // step 1: the policy, exactly as an operator would write it in config.
    // floor=info: the normal exporter's contract is unchanged. elevated=trace:
    // on trigger, replay goes all the way down to trace. sample_ratio=1.0 here
    // so THIS walkthrough's one trace is deterministically verbose (production
    // would use something like 0.01 -- 1% of traces pay the buffering cost).
    let elevation = Elevation {
        floor: Level::INFO,
        elevated: Some(Level::TRACE),
        sample_ratio: 1.0,
        trigger_level: Level::ERROR,
        exporter: ExporterChoice::Noop,
        retention: Retention::default(),
    };
    let config = TelemetryConfig::builder()
        .elevation(elevation.clone())
        .build();
    config.validate().expect("a sane elevation policy validates");
    println!(
        "policy: floor={} elevated={} sample_ratio={} trigger={}",
        elevation.floor,
        elevation.resolved_elevated(),
        elevation.sample_ratio,
        elevation.trigger_level
    );

    // step 2: build the sink by hand the way `install_elevation` (src/config.rs)
    // builds it from a config -- a `0` retention field resolves to the
    // build-time `sized` default so there is one source of truth for "how much
    // memory can a trace flood cost".
    let capture = Capture::new();
    let per_trace_ring = if elevation.retention.per_trace_ring == 0 {
        proxima_telemetry::sized::ELEVATION_PER_TRACE_RING
    } else {
        elevation.retention.per_trace_ring
    };
    let max_traces = if elevation.retention.max_traces == 0 {
        proxima_telemetry::sized::ELEVATION_MAX_TRACES
    } else {
        elevation.retention.max_traces
    };
    let sink = ElevationSink::new(
        into_telemetry_handle(capture.clone()),
        elevation.trigger_level,
        per_trace_ring,
        max_traces,
        elevation.retention.ttl_millis.saturating_mul(1_000_000),
        elevation.retention.drain_on_root_close,
    );
    println!("sink: per_trace_ring={per_trace_ring} max_traces={max_traces}");

    // step 3: drive one verbose-sampled trace's records through the sink,
    // out of ts_ns order -- exactly how concurrent emit arrives in practice.
    // Every one of these is BELOW the info floor (debug/trace), so the normal
    // exporter never sees them; only the buffer does.
    let trace = TraceId::from_bytes([0x42; 16]);
    let below_floor = log_batch_request(vec![
        verbose_log(trace, Level::INFO, 300, "cache miss for key=user:42"),
        verbose_log(trace, Level::TRACE, 100, "handler entered: GET /users/42"),
        verbose_log(trace, Level::DEBUG, 200, "querying users table"),
    ]);
    block_on(SendPipe::call(&sink, below_floor)).expect("buffer accepted");
    println!(
        "buffered 3 records (1 floor+, 2 below-floor) -- elevated sink saw {} so far",
        capture.records().len()
    );

    // step 4: the trigger. An ERROR record at or above `trigger_level` fires
    // the replay -- the WHOLE buffered tree, not just this record, goes to the
    // elevated exporter, ordered by ts_ns.
    let trigger = log_batch_request(vec![verbose_log(
        trace,
        Level::ERROR,
        400,
        "downstream timeout: connection reset",
    )]);
    block_on(SendPipe::call(&sink, trigger)).expect("trigger accepted");

    // step 5: read back what the elevated sink actually received -- the full
    // ordered tree, floor+ and below-floor together, exactly as the trace
    // happened.
    println!("\nreplayed tree (ordered by ts_ns):");
    for record in capture.records() {
        println!(
            "  [{:>4}ns] {:<5} {}",
            record.ts_ns,
            record.level.name(),
            match record.body {
                LogBody::Text(text) => text,
                _ => "<non-text body>",
            }
        );
    }

    // a second, HEALTHY trace never triggers: its buffer is simply dropped
    // when its root span closes, and the elevated sink never sees a byte of it.
    let healthy_trace = TraceId::from_bytes([0x99; 16]);
    let healthy = log_batch_request(vec![verbose_log(
        healthy_trace,
        Level::DEBUG,
        10,
        "handler entered: GET /users/7 (this one succeeds)",
    )]);
    block_on(SendPipe::call(&sink, healthy)).expect("buffer accepted");
    let root_close = proxima_telemetry::pipes::span_batch_request(vec![root_span(healthy_trace)]);
    block_on(SendPipe::call(&sink, root_close)).expect("root close accepted");
    println!(
        "\nhealthy trace's root closed without a trigger: elevated sink still shows {} total records (unchanged)",
        capture.records().len()
    );
}

fn root_span(trace: TraceId) -> proxima_telemetry::trace::SpanRecord {
    proxima_telemetry::trace::SpanRecord {
        trace_id: trace,
        span_id: SpanId::from_bytes([1; 8]),
        parent_span_id: None,
        name: "root",
        kind: proxima_telemetry::trace::SpanKind::Internal,
        start_ns: 0,
        duration_ns: 10,
        status: proxima_telemetry::trace::Status::Unset,
        attrs: smallvec::SmallVec::new(),
        events: smallvec::SmallVec::new(),
        links: smallvec::SmallVec::new(),
        tracestate: proxima_telemetry::trace::TraceState(None),
        module_path: "elevation_walkthrough",
        file_line: (0, 0),
    }
}
