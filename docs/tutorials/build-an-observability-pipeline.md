# Build an observability pipeline

**Prerequisites:** [Foundations](./00-foundations.md) — the **filter**, **observe**, and **sink** roles.
**You will:** build a logging pipeline that filters by level, fans one event out to console **and** file sinks (each independently filtered), and makes the backpressure tradeoff explicit. The point: observability is not special machinery — it is the pipe algebra aimed at telemetry.
**New concepts (in order):** structured logging as a level **filter** (`RUST_LOG`) · **fan-out** to sinks (`fan_exporters`) · **backpressure** as an explicit choice (`HeapBoundedQueue` + `FailMode`).
**Answer key:** [`examples/logs/main.rs`](../../examples/logs/main.rs) — `cargo run --example logs`.

The example says it: *"Logging is not special machinery. It is the same three primitives the fan-out, filter, and gate examples already taught, applied to one more payload shape: a log record instead of an HTTP request."*

## 1. Structured logging is a filter

The telemetry macros carry typed fields, and every callsite is gated by a runtime level filter (`RUST_LOG`) before it reaches the recorder — that gate **is** `filter`, a `Decide` (level ≥ floor) run before the inner pipe (`logs/main.rs:59-93`).

A `Recorder` is the telemetry sink registry — the `App`-equivalent for logs: build one, wire in an `Exporter`, then emit through it. `Exporter::stdout()` sends records to the console; `.core_count(1)` sizes the recorder's worker pool (1 is fine for this example); `.drain()` flushes the recorder and returns how many records it exported:

```rust
std::env::set_var("RUST_LOG", "debug");   // floor: debug
let recorder = Recorder::builder().export(Exporter::stdout())?.core_count(1).install()?;

trace!(%peer, "per-datagram noise");        // filtered — never reaches the ring
debug!(handle = 7u64, %peer, "worker picked up job");
info!(route = "checkout", jobs_processed = 42u64, "batch complete");
warn!(?err, attempt, "retrying after transient failure");
error!(reason = "max_retries_exceeded", "job abandoned");

assert_eq!(recorder.drain(), 4);   // trace filtered; debug/info/warn/error passed
```

The level floor short-circuits below-threshold records before any recorder work — the same `Decide`-then-delegate shape from [`examples/filter`](../../examples/filter). The `%` and `?` sigils in `%peer` and `?err` attach the field via `Display` or `Debug` formatting — shorthand for `peer = %peer` and `err = ?err`, the same typed-field mechanism as `route = "checkout"` above.

## 2. Fan-out to sinks, each independently filtered

One log event, delivered to console **and** a file via `fan_exporters` — the "one input, N sinks, N-1 clones" `FanOut` shape applied to telemetry. Each sink gets its *own* level filter, so fan-out and filter compose (`logs/main.rs:169-200`). `into_telemetry_handle` is `into_handle` for the telemetry pipe shape: it wraps a sink in the same kind of uniform handle Foundations introduced, sized for `TelemetryRequest` instead of an HTTP request:

```rust
let console_gate = into_telemetry_handle(LevelGate { inner: stdout_handle, threshold: Level::WARN, .. });   // .. = fields elided (passed/dropped counters)
let file_gate    = into_telemetry_handle(LevelGate { inner: file_handle,   threshold: Level::DEBUG, .. });   // .. = fields elided (passed/dropped counters)
let fanned = fan_exporters(vec![console_gate, file_gate]);
let recorder = Recorder::builder().export(Exporter::pipe(fanned))?.core_count(1).start()?;
```

This recorder calls `.start()` instead of `.install()` from §1: `install` sets the recorder as the process-wide default (so the macros in §1 reach it); `start` runs a recorder without installing it as that default, which matters once more than one recorder exists.

Emit three events (DEBUG/INFO/WARN): the file sink (floor DEBUG) keeps all three; the console sink (floor WARN) keeps one — the *same* fanned event, independent decisions (`logs/main.rs:234-253`). `LevelGate` is named "Gate" but is actually a hand-composed `Filter`: it decides per record (level ≥ threshold) and forwards only survivors, the same `Decide`-then-delegate shape as `filter` — not the armed/disarmed readiness switch Foundations calls `gate`. It's hand-composed rather than using `proxima_pipe::Filter` directly because Rust won't let this example crate implement an external trait (`Rejectable`, the sink/reject trait) on an external type (`TelemetryRequest`) that it doesn't own — so it defines its own adapter instead, the same reason `gate.rs`'s `Gated<G>` hand-composes its `Decide` adapter (`logs/main.rs:98-105`). See [`examples/fan_out`](../../examples/fan_out).

## 3. Backpressure is a choice you make in the open

A bounded queue sits in front of a sink. A bounded queue is the concrete backpressure primitive; a gate (Foundations §9) is the readiness switch in front of it — same idea, control flow under load, two tools that compose. `HeapBoundedQueue` + `FailMode` exposes the lossless-vs-lossy tradeoff directly — no "async appender" swallowing the decision (`logs/main.rs:263-360`):

- **Lossy, `DropNewest`** — 6 records into a 4-slot queue → 2 refused; keeps the oldest 4.
- **Lossy, `DropOldest`** — never refuses the newest; evicts the 2 oldest to make room.
- **Lossless, `enqueue_assisting`** — the producer becomes a momentary consumer to free a slot, so nothing is dropped, at the cost of throttling to the sink's real speed.

```rust
let q = HeapBoundedQueue::<LogLine>::new(4, FailMode::DropOldest);
for message in BURST { q.enqueue(LogLine { .. }); }   // evicts oldest to admit newest
```

The tradeoff is explicit: lossy bounds memory and latency at the cost of dropped signal; lossless guarantees delivery at the cost of throttling the producer. See [`examples/backpressure`](../../examples/backpressure).

## What you built, and the one idea

An observability pipeline from three primitives you already know:

- **filter** — a level floor (`RUST_LOG`) short-circuits below-threshold records before the recorder.
- **fan-out** — `fan_exporters` delivers one event to N sinks, each independently filtered.
- **backpressure** — a bounded queue makes the lossless-vs-lossy choice explicit, in the open.

Observability is the pipe algebra aimed at telemetry — but only half of it, and the honest half matters. The *shipping* side composes: one record is fanned out to console + file + OTLP together ([`examples/export`](../../examples/export)), each arm a pipe. The *recording* side deliberately does not: a metric is `Counter::add(&self, delta, tags)`, a direct call on a handle, and a span is not a pipe at all ([`examples/metrics`](../../examples/metrics), [`examples/traces`](../../examples/traces)). That is a design decision, not an omission — a counter bump sits on the hottest path in the program, and a pipe chain per increment would allocate and compose to record a single integer. A pipe is for things worth composing; when the answer is "increment this number, now", reach for a function. One `#[proxima::instrument]` still yields metric + trace + log ([`examples/instrument`](../../examples/instrument)).
