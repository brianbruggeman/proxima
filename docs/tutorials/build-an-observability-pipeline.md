# Build an observability pipeline

**Prerequisites:** [Foundations](./00-foundations.md) — the `Pipe` trait, `filter`, and `into_handle`.
**You will:** build a logging pipeline that filters by level, fans one event out to console **and** file sinks (each independently filtered), and makes the backpressure tradeoff explicit. The point: observability is not special machinery — it is the pipe algebra aimed at telemetry.
**New concepts (in order):** structured logging as a level **filter** (an `EnvFilter` floor) · **fan-out** to sinks (`fan_exporters`) · **backpressure** as an explicit choice (`HeapBoundedQueue` + `FailMode`).
**Answer key:** [`examples/logs/main.rs`](../../examples/logs/main.rs) — `cargo run --example logs`.

The example says it: *"Logging is not special machinery. It is the same three primitives the fan-out, filter, and gate examples already taught, applied to one more payload shape: a log record instead of an HTTP request."*

## 1. Structured logging is a filter

`proxima::telemetry`'s `trace!`/`debug!`/`info!`/`warn!`/`error!` macros carry typed fields, and every callsite is gated by a runtime level filter before it ever reaches a recorder — that gate **is** `filter`, a `Decide` (level ≥ floor) run before the inner pipe (`examples/logs/main.rs:64-98`).

A `Recorder` is the telemetry sink registry — install one, wire in an `Exporter`, then emit through the macros. `Exporter::stdout()` sends records to the console; `.core_count(1)` sizes the recorder's worker pool (1 is fine here); `.drain()` flushes the recorder and returns how many records it exported. `EnvFilter::parse` reads the same grammar `RUST_LOG` does, applied to a literal string instead of the process environment — useful here because it makes the floor deterministic regardless of the caller's shell:

```rust
use proxima::telemetry::emit::EnvFilter;
use proxima::telemetry::emit::global::install as install_emit_filter;
use proxima::telemetry::export::Exporter;
use proxima::telemetry::recorder::Recorder;
use proxima::telemetry::{debug, error, info, trace, warn};

install_emit_filter(EnvFilter::parse("debug"));   // floor: debug

let recorder = Recorder::builder()
    .export(Exporter::stdout())
    .expect("stdout exporter composes")
    .core_count(1)
    .install()
    .expect("recorder installs as the process default");

let peer = "10.0.0.7:51422";
let attempt = 3u64;
let err = "connection reset by peer";

trace!(%peer, "per-datagram noise");        // filtered — never reaches the ring
debug!(handle = 7u64, %peer, "worker picked up job");
info!(route = "checkout", jobs_processed = 42u64, "batch complete");
warn!(?err, attempt, "retrying after transient failure");
error!(reason = "max_retries_exceeded", "job abandoned");

assert_eq!(recorder.drain(), 4);   // trace filtered; debug/info/warn/error passed
```

The level floor short-circuits below-threshold records before any recorder work — the same `Decide`-then-delegate shape from [`examples/filter`](../../examples/filter). The `%` and `?` sigils in `%peer` and `?err` attach the field via `Display` or `Debug` formatting — shorthand for `peer = %peer` and `err = ?err`, the same typed-field mechanism as `route = "checkout"` above.

## 2. Fan-out to sinks, each independently filtered

One log event, delivered to console **and** a file via `fan_exporters` — the "one input, N sinks, N-1 clones" `FanOut` shape applied to telemetry. Each sink gets its *own* level filter, so fan-out and filter compose (`examples/logs/main.rs:100-272`). `into_telemetry_handle` is `into_handle` for the telemetry pipe shape — it wraps a sink in the same kind of uniform handle Foundations introduced, sized for `TelemetryRequest` instead of an HTTP request.

There is no ready-made `Filter` type for this: proxima's own decision pipes (`PipeExt::filter`, `AndThen::new(predicate, self)` — `proxima-primitives/src/pipe/ext.rs:57-62`) compose over `Request<Bytes>`, and this pipeline's payload is a `TelemetryRequest`. So `LevelGate` writes the same decision-then-delegate shape out by hand — admit a record at or above `threshold`, otherwise short-circuit without ever calling the inner sink:

```rust
use std::fs::File;
use std::future::Future;
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use bytes::Bytes;
use proxima::telemetry::level::Level;
use proxima::telemetry::pipes::{
    FormatterPipe, LogFormat, TelemetryPipeHandle, TelemetryRecord, TelemetryRequest,
    fan_exporters, into_telemetry_handle,
};
use proxima_primitives::pipe::{ProximaError, Response, SendPipe};

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
        // the drainer delivers a batch per drain cycle, not one record per
        // call — filter the level per record inside the batch, and forward
        // only the survivors.
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

// one event in, N sinks, concurrently — the same shape the standalone
// `fan_out` example teaches over `FanOut<S, Policy>`, specialized to
// telemetry's TelemetryRequest.
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
let fanned = fan_exporters(vec![console_gate, file_gate]);

// `.start()` runs a recorder without installing it as the process default —
// unlike `.install()` in §1, which the macros above reach through implicitly.
let recorder = Recorder::builder()
    .export(Exporter::pipe(fanned))
    .expect("fanned exporter composes")
    .core_count(1)
    .start()
    .expect("recorder starts (not installed as the process default)");

recorder.log().level(Level::DEBUG).message("cache warmed").tag("entries", 4_096u64).emit();
recorder.log().level(Level::INFO).message("request served").tag("route", "checkout").emit();
recorder
    .log()
    .level(Level::WARN)
    .message("latency budget exceeded")
    .tag("elapsed_ms", 812u64)
    .emit();

assert_eq!(recorder.drain(), 3);   // the same fanned event, independent decisions

assert_eq!(file_passed.load(Ordering::Relaxed), 3, "file threshold is DEBUG: all 3 pass");
assert_eq!(file_dropped.load(Ordering::Relaxed), 0);
assert_eq!(console_passed.load(Ordering::Relaxed), 1, "console threshold is WARN: only the warn event passes");
assert_eq!(console_dropped.load(Ordering::Relaxed), 2, "console drops the debug and info events");
```

`LevelGate` is named "Gate" but is really a filter: it decides per record (level ≥ threshold) and forwards only survivors — not the armed/disarmed readiness switch Foundations calls `gate`. See [`examples/fan_out`](../../examples/fan_out).

## 3. Backpressure is a choice you make in the open

A bounded queue sits in front of a sink. A bounded queue is the concrete backpressure primitive; a gate (Foundations §9) is the readiness switch in front of it — same idea, control flow under load, two tools that compose. `HeapBoundedQueue` + `FailMode` exposes the lossless-vs-lossy tradeoff directly — no "async appender" swallowing the decision (`examples/logs/main.rs:274-386`):

- **Lossy, `DropNewest`** — 6 records into a 4-slot queue → 2 refused; keeps the oldest 4.
- **Lossy, `DropOldest`** — never refuses the newest; evicts the 2 oldest to make room.
- **Lossless, `enqueue_assisting`** — the producer becomes a momentary consumer to free a slot, so nothing is dropped, at the cost of throttling to the sink's real speed.

```rust
use proxima::telemetry::level::Level;
use proxima::telemetry::ring::{EnqueueOutcome, FailMode, HeapBoundedQueue};

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

let drop_oldest = HeapBoundedQueue::<LogLine>::new(4, FailMode::DropOldest);
for message in BURST {
    let outcome = drop_oldest.enqueue(LogLine {
        level: Level::INFO,
        message,
    });
    assert!(
        matches!(outcome, EnqueueOutcome::Enqueued | EnqueueOutcome::DroppedOldest),
        "DropOldest never refuses the newest record"
    );
}
assert_eq!(drop_oldest.dropped(), 2, "the 2 oldest records are evicted to admit the newest");

let kept: Vec<&str> = std::iter::from_fn(|| drop_oldest.dequeue().map(|line| line.message)).collect();
assert_eq!(kept, BURST[2..], "DropOldest keeps the newest 4");
```

`FailMode::DropNewest` is the other lossy flavor — it refuses the *incoming* record instead of evicting a queued one, so a 4-slot queue fed the same 6-record burst also drops 2, but keeps the oldest 4 rather than the newest 4. `enqueue_assisting` is the lossless third option: it bypasses `fail_mode` entirely, looping through an explicit make-room step — the producer becomes a momentary consumer, draining one record to free a slot, then retries — so nothing is ever lost, the same shape a full per-core ring's elastic producer-assist runs under overload. See [`examples/backpressure`](../../examples/backpressure).

The tradeoff is explicit: lossy bounds memory and latency at the cost of dropped signal; lossless guarantees delivery at the cost of throttling the producer.

## What you built, and the one idea

An observability pipeline from three primitives you already know:

- **filter** — a level floor short-circuits below-threshold records before the recorder.
- **fan-out** — `fan_exporters` delivers one event to N sinks, each independently filtered.
- **backpressure** — a bounded queue makes the lossless-vs-lossy choice explicit, in the open.

Observability is the pipe algebra aimed at telemetry — but only half of it, and the honest half matters. The *shipping* side composes: one record is fanned out to console + file + OTLP together ([`examples/export`](../../examples/export)), each arm a pipe. The *recording* side deliberately does not: a metric is `Counter::add(&self, delta, tags)`, a direct call on a handle, and a span is not a pipe at all ([`examples/metrics`](../../examples/metrics), [`examples/traces`](../../examples/traces)). That is a design decision, not an omission — a counter bump sits on the hottest path in the program, and a pipe chain per increment would allocate and compose to record a single integer. A pipe is for things worth composing; when the answer is "increment this number, now", reach for a function. One `#[proxima::instrument]` still yields metric + trace + log ([`examples/instrument`](../../examples/instrument)).
