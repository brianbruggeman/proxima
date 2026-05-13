#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Logging is not special machinery. It is the same three primitives the
//! `fan-out`, `filter`, and `gate` examples already taught, applied to one
//! more payload shape: a log record instead of an HTTP request or a job.
//!
//! - **structured logging** — `proxima_telemetry`'s `error!`/`warn!`/`info!`/
//!   `debug!`/`trace!` macros carry typed fields, and every callsite is gated
//!   by a runtime level filter (`RUST_LOG`) before it ever reaches a recorder.
//!   That gate IS `filter`: a decision pipe (level >= floor, `In ->
//!   Result<In, Err>`) run before the "inner pipe" (the recorder) is ever
//!   called.
//! - **fan-out to sinks** — one log event, delivered to console AND a file,
//!   through `proxima_telemetry::pipes::fan_exporters` — the same "one input,
//!   N sinks, N-1 clones" shape `FanOut` taught, applied to `TelemetryRequest`
//!   instead of `Message`. Each sink additionally gets its own level filter,
//!   so fan-out and filter compose exactly as the `gate` example's SHED shape
//!   composes a gate-reading decision pipe in front of a gate.
//! - **backpressure is a choice, not hidden machinery** — a bounded queue sits
//!   in front of a sink. `proxima_telemetry::ring::HeapBoundedQueue` (the same
//!   primitive the real per-core log ring is built from) exposes the tradeoff
//!   directly: `FailMode::DropNewest` / `DropOldest` (lossy — shed under
//!   overload) versus `enqueue_assisting` (lossless — the producer becomes a
//!   momentary consumer to make room, the same shape `OverflowPolicy::Block`'s
//!   elastic producer-assist uses). Nothing here is an "async appender"
//!   swallowing the decision — the choice is made explicitly, in the open.
//!
//! Run: `cargo run --example logs`

use std::fs::File;
use std::future::Future;
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use bytes::Bytes;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::{ProximaError, Response};
use proxima_telemetry::emit::EnvFilter;
use proxima_telemetry::emit::global::install as install_emit_filter;
use proxima_telemetry::export::Exporter;
use proxima_telemetry::level::Level;
use proxima_telemetry::pipes::{
    FormatterPipe, LogFormat, TelemetryPipeHandle, TelemetryRecord, TelemetryRequest,
    fan_exporters, into_telemetry_handle,
};
use proxima_telemetry::recorder::Recorder;
use proxima_telemetry::ring::{EnqueueOutcome, FailMode, HeapBoundedQueue};
use proxima_telemetry::{debug, error, info, trace, warn};

fn main() {
    println!("structured logging: real macros + RUST_LOG level discipline");
    run_structured_logging();

    println!("\nfan-out to sinks: one log event, console AND file, filtered per sink");
    run_fanout_sinks();

    println!("\nbackpressure: bounded queue in front of a sink, lossless vs lossy");
    run_backpressure_tradeoff();
}

// ── 1. structured logging: the real macros, gated by RUST_LOG ──────────────

fn run_structured_logging() {
    // The emit filter is read lazily on the first emit and cached per
    // callsite, so install it before anything fires: this run's floor is
    // deterministic regardless of the caller's shell environment, and
    // installed directly (`EnvFilter::parse` — the same grammar `RUST_LOG`
    // uses, applied to a literal string instead of the process env) rather
    // than mutating a global env var. debug lets trace! stay filtered while
    // debug!/info!/warn!/error! all pass.
    install_emit_filter(EnvFilter::parse("debug"));

    let recorder = Recorder::builder()
        .export(Exporter::stdout())
        .expect("stdout exporter composes")
        .core_count(1)
        .install()
        .expect("recorder installs as the process default");

    let peer = "10.0.0.7:51422";
    let attempt = 3u64;
    let err = "connection reset by peer";

    // filtered before it ever reaches the recorder: no "trace" line below.
    trace!(%peer, "per-datagram noise nobody asked for");
    debug!(handle = 7u64, %peer, "worker picked up job");
    info!(route = "checkout", jobs_processed = 42u64, "batch complete");
    warn!(?err, attempt, "retrying after transient failure");
    error!(reason = "max_retries_exceeded", "job abandoned");

    let exported = recorder.drain();
    println!("drained {exported} records (trace! never reached the ring)");
    assert_eq!(
        exported, 4,
        "RUST_LOG=debug passes debug/info/warn/error; trace is filtered before the recorder sees it"
    );
}

// ── 2. fan-out to sinks, each with its own level filter ─────────────────────

/// The `filter` half of "fan-out + filter" applied to a telemetry sink: wraps
/// a `TelemetryPipeHandle`, admits a log record only at or above `threshold`,
/// otherwise short-circuits without ever calling the inner sink — the exact
/// decision-pipe shape `filter.rs`'s `FilterConfig`/`Predicate` teaches, over
/// a payload (`TelemetryRequest`) `filter.rs`'s own decision pipes aren't
/// pinned to (they compose over `Request<Bytes>`) — the same reason
/// `gate.rs`'s `Gated<G>` hand-composes its own decision pipe instead of
/// reusing one of `filter.rs`'s HTTP-specific types.
struct LevelGate {
    inner: TelemetryPipeHandle,
    threshold: Level,
    passed: Arc<AtomicUsize>,
    dropped: Arc<AtomicUsize>,
}

impl SendPipe for LevelGate {
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: TelemetryRequest,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        // the drainer delivers a batch per drain cycle (RecordSharing::Inline
        // default), not one record per call — filter the level per record
        // inside the batch, and forward only the survivors.
        let floor = self.threshold.severity();
        let (kept_payload, admitted, refused) = match &request.payload {
            TelemetryRecord::Log(record) if record.level.severity() >= floor => {
                (Some(TelemetryRecord::Log(record.clone())), 1, 0)
            }
            TelemetryRecord::Log(_) => (None, 0, 1),
            TelemetryRecord::LogBatch(records) => {
                let (kept, refused): (Vec<_>, Vec<_>) = records
                    .iter()
                    .cloned()
                    .partition(|record| record.level.severity() >= floor);
                let admitted = kept.len();
                let refused_count = refused.len();
                let payload = (!kept.is_empty()).then_some(TelemetryRecord::LogBatch(kept));
                (payload, admitted, refused_count)
            }
            TelemetryRecord::LogBatchArc(records) => {
                let (kept, refused): (Vec<_>, Vec<_>) = records
                    .iter()
                    .cloned()
                    .partition(|record| record.level.severity() >= floor);
                let admitted = kept.len();
                let refused_count = refused.len();
                let payload = (!kept.is_empty()).then_some(TelemetryRecord::LogBatchArc(kept));
                (payload, admitted, refused_count)
            }
            _ => (None, 0, 0),
        };

        let inner = Arc::clone(&self.inner);
        let passed = Arc::clone(&self.passed);
        let dropped = Arc::clone(&self.dropped);
        let mut forwarded = request;

        async move {
            passed.fetch_add(admitted, Ordering::Relaxed);
            dropped.fetch_add(refused, Ordering::Relaxed);
            match kept_payload {
                Some(payload) => {
                    forwarded.payload = payload;
                    inner.call_dyn(forwarded).await
                }
                None => Ok(Response::ok(Bytes::new())),
            }
        }
    }
}

fn run_fanout_sinks() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let file_path = temp_dir.path().join("proxima-logs-fanout.log");

    let stdout_handle = into_telemetry_handle(FormatterPipe::new(io::stdout(), LogFormat::Human));
    let file_handle = into_telemetry_handle(FormatterPipe::new(
        File::create(&file_path).expect("create log file"),
        LogFormat::Human,
    ));

    let console_passed = Arc::new(AtomicUsize::new(0));
    let console_dropped = Arc::new(AtomicUsize::new(0));
    let file_passed = Arc::new(AtomicUsize::new(0));
    let file_dropped = Arc::new(AtomicUsize::new(0));

    let console_gate = into_telemetry_handle(LevelGate {
        inner: stdout_handle,
        threshold: Level::WARN,
        passed: Arc::clone(&console_passed),
        dropped: Arc::clone(&console_dropped),
    });
    let file_gate = into_telemetry_handle(LevelGate {
        inner: file_handle,
        threshold: Level::DEBUG,
        passed: Arc::clone(&file_passed),
        dropped: Arc::clone(&file_dropped),
    });

    // one event in, N sinks, concurrently — proxima-telemetry's own fan-out
    // primitive, the same shape the standalone `fan_out` example teaches over
    // `FanOut<S, Policy>`, specialized here to telemetry's TelemetryRequest.
    let fanned = fan_exporters(vec![console_gate, file_gate]);

    let recorder = Recorder::builder()
        .export(Exporter::pipe(fanned))
        .expect("fanned exporter composes")
        .core_count(1)
        .start()
        .expect("recorder starts (not installed as the process default)");

    recorder
        .log()
        .level(Level::DEBUG)
        .message("cache warmed")
        .tag("entries", 4_096u64)
        .emit();
    recorder
        .log()
        .level(Level::INFO)
        .message("request served")
        .tag("route", "checkout")
        .emit();
    recorder
        .log()
        .level(Level::WARN)
        .message("latency budget exceeded")
        .tag("elapsed_ms", 812u64)
        .emit();

    let exported = recorder.drain();
    println!("fanned {exported} log events to 2 sinks");

    let file_contents = std::fs::read_to_string(&file_path).expect("read fanned file");
    println!(
        "--- file sink ({}) ---\n{file_contents}",
        file_path.display()
    );

    assert_eq!(
        file_passed.load(Ordering::Relaxed),
        3,
        "file threshold is DEBUG: all 3 events pass"
    );
    assert_eq!(
        file_dropped.load(Ordering::Relaxed),
        0,
        "file drops nothing at DEBUG"
    );
    assert_eq!(
        console_passed.load(Ordering::Relaxed),
        1,
        "console threshold is WARN: only the warn event passes"
    );
    assert_eq!(
        console_dropped.load(Ordering::Relaxed),
        2,
        "console drops the debug and info events, same fanned event, independent decision"
    );
    for message in ["cache warmed", "request served", "latency budget exceeded"] {
        assert_eq!(
            file_contents.matches(message).count(),
            1,
            "file received every fanned event exactly once"
        );
    }
}

// ── 3. backpressure: the explicit lossless-vs-lossy choice ─────────────────

#[derive(Debug, Clone, Copy)]
struct LogLine {
    #[allow(dead_code)]
    level: Level,
    message: &'static str,
}

const BURST: [&str; 6] = [
    "worker 1 started",
    "worker 2 started",
    "worker 3 started",
    "worker 4 started",
    "worker 5 started",
    "worker 6 started",
];

fn run_backpressure_tradeoff() {
    println!("-- lossy (FailMode::DropNewest): the incoming record is refused --");
    let drop_newest = HeapBoundedQueue::<LogLine>::new(4, FailMode::DropNewest);
    for message in BURST {
        let outcome = drop_newest.enqueue(LogLine {
            level: Level::INFO,
            message,
        });
        println!("  enqueue {message:?}: {outcome:?}");
    }
    println!("  dropped: {}", drop_newest.dropped());
    assert_eq!(
        drop_newest.dropped(),
        2,
        "6 records into a 4-slot queue under DropNewest: 2 refused"
    );
    let kept_newest: Vec<&str> =
        std::iter::from_fn(|| drop_newest.dequeue().map(|line| line.message)).collect();
    assert_eq!(
        kept_newest,
        BURST[..4],
        "DropNewest keeps the oldest 4 already queued, refuses the rest"
    );

    println!(
        "\n-- lossy, other flavor (FailMode::DropOldest): evict the oldest to admit the newest --"
    );
    let drop_oldest = HeapBoundedQueue::<LogLine>::new(4, FailMode::DropOldest);
    for message in BURST {
        let outcome = drop_oldest.enqueue(LogLine {
            level: Level::INFO,
            message,
        });
        assert!(
            matches!(
                outcome,
                EnqueueOutcome::Enqueued | EnqueueOutcome::DroppedOldest
            ),
            "DropOldest never refuses the newest record"
        );
    }
    println!("  dropped: {}", drop_oldest.dropped());
    assert_eq!(drop_oldest.dropped(), 2, "the 2 oldest records are evicted");
    let kept_oldest: Vec<&str> =
        std::iter::from_fn(|| drop_oldest.dequeue().map(|line| line.message)).collect();
    assert_eq!(
        kept_oldest,
        BURST[2..],
        "DropOldest keeps the newest 4, evicting to make room instead of refusing"
    );

    println!("\n-- lossless: enqueue_assisting makes room by draining, nothing is dropped --");
    // fail_mode is irrelevant here: enqueue_assisting bypasses it entirely,
    // looping through an explicit make-room step instead. This is the same
    // shape OverflowPolicy::Block's elastic producer-assist runs under a full
    // per-core ring — the producer becomes a momentary consumer to free a
    // slot, then retries, so nothing is ever lost.
    let lossless = HeapBoundedQueue::<LogLine>::new(4, FailMode::FailClosed);
    let mut delivered: Vec<&str> = Vec::new();
    for message in BURST {
        lossless
            .enqueue_assisting(
                LogLine {
                    level: Level::INFO,
                    message,
                },
                || match lossless.dequeue() {
                    Some(line) => {
                        delivered.push(line.message);
                        true
                    }
                    None => false,
                },
            )
            .expect("a 4-slot queue always has room to make by draining one item");
    }
    while let Some(line) = lossless.dequeue() {
        delivered.push(line.message);
    }
    println!("  delivered, in order: {delivered:?}");
    assert_eq!(
        delivered, BURST,
        "lossless: every record eventually reaches the sink, in order, none dropped"
    );
    assert_eq!(
        lossless.dropped(),
        0,
        "enqueue_assisting never counts a drop — the tradeoff is throughput, not data"
    );

    println!(
        "\nthe choice is explicit: lossy bounds memory and latency at the cost of dropped signal; \
         lossless guarantees delivery at the cost of throttling the producer to the sink's real speed."
    );
}
