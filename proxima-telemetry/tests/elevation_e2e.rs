//! End-to-end proof of the error-elevation runtime path that the unit tests in
//! `src/pipes.rs` (`elevation_sink_tests`) stub past: they drive `ElevationSink`
//! directly with hand-built `LogRecord`s. This test drives it from the OTHER
//! end — a real `trace!`/`info!`/`error!` macro call, inside a real span, on a
//! real `Recorder` — so the chain that actually runs in production is proven
//! end to end:
//!
//!   macro call -> current-span verbose stamp (`log/builder.rs`) -> the
//!   macro's below-floor admit branch (`emit/macros.rs`) -> per-core ring ->
//!   drainer -> fan-out -> `FloorFilter` / `ElevationSink` -> on `error!`, the
//!   elevated sink receives the trace's full ordered tree.
//!
//! `Recorder::from_config`/`install_elevation` can't be used as-is here: the
//! elevated sink is resolved from `Elevation.exporter: ExporterChoice`, which
//! only lowers to `Noop`/`OtlpHttp`/`OtlpGrpc` — none of them capturable in a
//! test. So the terminal pipe is wired by hand, mirroring
//! `config.rs::install_elevation`'s fan (`[FloorFilter -> normal,
//! ElevationSink -> elevated]`), with both arms landing on a capturing sink;
//! `cfg.elevation` stays `None` so `from_config_with_pipe` doesn't ALSO wrap
//! it, and the verbose sampler is armed directly via `current::*`.

#![cfg(feature = "elevation")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::{Arc, Mutex};

use bytes::Bytes;
use proxima_primitives::pipe::request::Response;
use proxima_primitives::pipe::{ProximaError, SendPipe};

use proxima_telemetry::config::{RecordSharing, TelemetryConfig};
use proxima_telemetry::current;
use proxima_telemetry::emit::{EnvFilter, global};
use proxima_telemetry::level::Level;
use proxima_telemetry::log::{LogBody, LogRecord};
use proxima_telemetry::pipes::{
    ElevationSink, FloorFilter, TelemetryPipeHandle, TelemetryRecord, TelemetryRequest,
    fan_exporters, into_telemetry_handle,
};
use proxima_telemetry::recorder::Recorder;
use proxima_telemetry::{error, info, trace};

// global::install and the verbose-sampler statics (current::set_verbose_ratio /
// set_verbose_admit_floor) are process-wide, like emit/macros.rs's
// recorder_routing tests — this file has a single #[test], but the lock keeps
// the state change explicit and safe against a future second test landing here.
static GLOBAL_STATE_LOCK: Mutex<()> = Mutex::new(());

fn lock_global_state() -> std::sync::MutexGuard<'static, ()> {
    GLOBAL_STATE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Terminal pipe that records every `LogRecord` it is handed, across both wire
/// forms the drainer can produce: `LogBatch` (inline — what `ElevationSink`'s
/// replay sends) and `LogBatchArc` (what the drainer sends when
/// `record_sharing = Arc`, which the fan-out here requires).
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
            let mut sink = seen.lock().expect("capture lock");
            match request.payload {
                TelemetryRecord::Log(record) => sink.push(record),
                TelemetryRecord::LogBatch(records) => sink.extend(records),
                TelemetryRecord::LogBatchArc(records) => {
                    sink.extend(records.iter().map(|record| (**record).clone()));
                }
                _ => {}
            }
            Ok(Response::ok(Bytes::new()))
        }
    }
}

fn message(record: &LogRecord) -> &'static str {
    match record.body {
        LogBody::Text(text) => text,
        _ => "<non-text body>",
    }
}

/// Build a recorder whose terminal pipe mirrors `config.rs::install_elevation`'s
/// fan by hand, so both arms are capturable: arm A is `FloorFilter -> normal`
/// (floor+ only, matching the exporter's contract today); arm B is
/// `ElevationSink -> elevated` (buffers verbose-sampled traces, replays the
/// whole tree on `error`). `record_sharing = Arc` because the fan needs it
/// (normally derived automatically inside `install_elevation`, which this
/// test bypasses).
fn recorder_with_capturable_elevation(normal: &Capture, elevated: &Capture) -> Recorder {
    let floor_arm: TelemetryPipeHandle = into_telemetry_handle(FloorFilter::new(
        Level::INFO,
        into_telemetry_handle(normal.clone()),
    ));
    let elevation_arm: TelemetryPipeHandle = into_telemetry_handle(ElevationSink::new(
        into_telemetry_handle(elevated.clone()),
        Level::ERROR,
        64,
        16,
        0,
        true,
    ));
    let fan = fan_exporters(vec![floor_arm, elevation_arm]);

    let cfg = TelemetryConfig::builder()
        .core_count(1)
        .record_sharing(RecordSharing::Arc)
        .build();
    assert!(cfg.elevation.is_none(), "install_elevation must not double-wrap the fan");

    Recorder::from_config_with_pipe(&cfg, fan)
        .start()
        .expect("recorder build")
}

fn drain_fully(recorder: &Recorder) {
    while recorder.drain() > 0 {}
}

// the headline path: a real trace!/info!/error! sequence, emitted inside a
// real span of a verbose-sampled trace, ends up replayed whole to the elevated
// sink — and a non-verbose trace's records never do, at floor+ cost only.
#[test]
fn error_inside_verbose_span_replays_full_tree_end_to_end() {
    let _guard = lock_global_state();
    global::install(EnvFilter::parse(""));

    // case 1: a verbose-sampled trace. ratio=1.0 admits (effectively) every
    // trace; admit_floor=TRACE means below-the-callsite-floor records (trace,
    // info — the default floor is error) are still constructed and buffered.
    let normal = Capture::new();
    let elevated = Capture::new();
    let recorder = recorder_with_capturable_elevation(&normal, &elevated);
    current::set_verbose_ratio(1.0);
    current::set_verbose_admit_floor(Level::TRACE);

    {
        let _span = recorder.span("verbose-request").start();
        trace!(recorder = &recorder, "handler entered");
        info!(recorder = &recorder, "cache miss");
        error!(recorder = &recorder, "downstream timeout");
    }
    drain_fully(&recorder);

    let normal_logs = normal.records();
    assert_eq!(
        normal_logs.len(),
        2,
        "normal sink is floor+ only (info, error) — never the below-floor trace"
    );
    assert!(
        normal_logs.iter().all(|record| record.level.severity() >= Level::INFO.severity()),
        "FloorFilter must retain only floor+ records"
    );
    assert!(
        !normal_logs.iter().any(|record| record.level == Level::TRACE),
        "the below-floor trace record must never reach the normal sink"
    );

    let elevated_logs = elevated.records();
    assert_eq!(
        elevated_logs.len(),
        3,
        "the error trigger must replay the WHOLE tree — below-floor trace included"
    );
    assert!(
        elevated_logs.iter().all(|record| record.trace_flags.is_verbose_buffered()),
        "every replayed record must carry the VERBOSE_BUFFERED stamp from log/builder.rs"
    );
    assert!(
        elevated_logs.iter().any(|record| record.level == Level::TRACE && message(record) == "handler entered"),
        "the below-floor trace record must have been admitted, built, and replayed"
    );
    assert!(
        elevated_logs.iter().any(|record| record.level == Level::INFO && message(record) == "cache miss")
    );
    assert!(
        elevated_logs.iter().any(|record| record.level == Level::ERROR && message(record) == "downstream timeout")
    );
    let timestamps: Vec<u64> = elevated_logs.iter().map(|record| record.ts_ns).collect();
    let mut sorted = timestamps.clone();
    sorted.sort_unstable();
    assert_eq!(timestamps, sorted, "the replay is ordered by ts_ns");
    let mut unique = timestamps.clone();
    unique.dedup();
    if unique.len() == timestamps.len() {
        // distinct clock reads: trace must precede info must precede error,
        // matching emission order.
        let trace_ts = elevated_logs
            .iter()
            .find(|record| record.level == Level::TRACE)
            .expect("trace record present")
            .ts_ns;
        let error_ts = elevated_logs
            .iter()
            .find(|record| record.level == Level::ERROR)
            .expect("error record present")
            .ts_ns;
        assert!(trace_ts < error_ts, "trace must have been recorded before the error that triggered it");
    }

    // case 2: a non-verbose trace (ratio 0.0). trace!/info! are never even
    // admitted below the callsite floor, so they never construct a record; the
    // error still reaches the normal sink (floor+, unconditional on sampling),
    // but nothing is ever buffered or replayed — the healthy path stays free.
    let normal2 = Capture::new();
    let elevated2 = Capture::new();
    let recorder2 = recorder_with_capturable_elevation(&normal2, &elevated2);
    current::set_verbose_ratio(0.0);
    current::set_verbose_admit_floor(Level::TRACE);

    {
        let _span = recorder2.span("healthy-request").start();
        trace!(recorder = &recorder2, "handler entered (non-verbose)");
        info!(recorder = &recorder2, "cache miss (non-verbose)");
        error!(recorder = &recorder2, "downstream timeout (non-verbose)");
    }
    drain_fully(&recorder2);

    let normal_logs2 = normal2.records();
    assert_eq!(
        normal_logs2.len(),
        1,
        "only the error crosses the callsite floor for a non-verbose trace"
    );
    assert_eq!(normal_logs2[0].level, Level::ERROR);
    assert!(
        !normal_logs2[0].trace_flags.is_verbose_buffered(),
        "a non-verbose trace's records are never stamped VERBOSE_BUFFERED"
    );

    assert!(
        elevated2.records().is_empty(),
        "a non-verbose trace never buffers or replays, even on error"
    );
}
