---
name: proxima-log
description: How proxima's native observability works — the unified `#[proxima::instrument]` / `#[proxima::span]` annotation, the log macros (`error!`/`warn!`/`info!`/`debug!`/`trace!`), and the metric instruments (`Counter`/`Gauge`/`UpDownCounter`/`Histogram` + `counter!`/`gauge!`/`histogram!`/`updown!`). Covers the call syntax, the three pillars (metric/trace/log) from one annotation, the RUST_LOG runtime filter, the ambient-recorder no-op contract, the `instrument-metrics` feature firewall, and the consumer gate. Use when instrumenting proxima/csr/ragd code to debug, when a log line or metric isn't appearing, when adding `#[instrument]` to a hot function, when wiring a recorder, or when the user says "instrument this", "add logging", "add a metric", "why isn't my log showing", "how does proxima::log work", "proxima tracing". This is proxima's OWN telemetry — it is NOT the `tracing` crate, NOT the `log` crate, NOT the `metrics` crate.
---

# proxima-log

proxima has its own observability substrate (`proxima-telemetry` + the
`proxima_macros` attribute macros). It is a drop-in for the `tracing` call
syntax but records into proxima's own per-core lock-free recorder, gated by
proxima's own runtime filter. No `tracing` dependency on the hot path.

Three surfaces, one recorder:

- **`#[proxima::instrument]` / `#[proxima::span]`** — one annotation makes a
  function an observable unit of work across all three pillars.
- **log macros** — `error!` / `warn!` / `info!` / `debug!` / `trace!`.
- **metric instruments** — `Counter` / `Gauge` / `UpDownCounter` / `Histogram`
  with the `counter!` / `gauge!` / `histogram!` / `updown!` macros.

## The two non-negotiable preconditions

A log line, span, or span-metric records ONLY when **both** are true. When a
debug session shows "my instrumentation produced nothing," it is almost always
one of these:

1. **A recorder is installed.** Emit sites resolve the process-wide recorder via
   `export::default_recorder()`; `#[instrument]` resolves it via
   `Recorder::current()`. With none installed the call is a no-op and the body
   runs span-free — by design, not a bug. Install one:

   ```rust
   // simplest: console logging to stderr
   let _recorder = proxima_telemetry::export::install_console_logging()?;

   // or build explicitly and set as process default
   let recorder = proxima_telemetry::recorder::Recorder::builder()
       .pipe(my_sink)
       .core_count(num_cores)
       .start()?;
   proxima_telemetry::export::set_default_recorder(recorder);
   // RecorderBuilder::install() does .start() + set_default in one call
   ```

2. **The level passes the runtime filter.** The default floor is `error` — with
   no `RUST_LOG` set, only `error!` records and an `info!`/`debug!` is dropped
   before it formats a single field. Raise it:

   ```bash
   RUST_LOG=debug                       # global floor
   RUST_LOG=proxima_h3=trace,warn       # per-target, tracing-subscriber grammar
   ```

   Read lazily on first emit from `RUST_LOG`. To set it programmatically:
   `proxima_telemetry::emit::global::install_from_env()` or
   `::install(EnvFilter::parse("debug"))`.

   Gotcha worth remembering: `RUST_LOG=debug` is known to BREAK the native-h3
   handshake under load — benchmark at `warn`. Debug-level logging is not free at
   volume even though a single dropped callsite is. That is a behavioural fact,
   not a measurement; if you need the cost, measure it yourself and record it in
   a discipline log.

## Log macros

```rust
use proxima_telemetry::{debug, info, warn, error, trace};

error!("a contract violation worth surfacing");        // leading message literal
debug!(?err, "debug-formatted field");                 // ?expr  -> Debug
debug!(label = %peer, "display-formatted field");      // %expr  -> Display
debug!(handle = handle, "typed scalar field");         // key = expr -> typed ScalarValue
debug!(len, %peer, "bare-ident shorthand plus display"); // bare ident -> key=value
warn!(label = %peer, len, reason = ?err, "mixed");     // any combination
```

Same field grammar as `tracing`: `?x` Debug-formats, `%x` Display-formats, a
bare `key = expr` is a **typed** scalar tag (not a Debug string), a bare ident
captures the variable under its own name, a leading literal is the message.

Why this is cheap: each callsite owns a `static CallsiteGate`. The first hit at
a given filter generation runs the filter once and caches the decision; every
later hit is two atomic loads. A disabled callsite never touches the recorder,
never formats a field, never allocates. The callsite stays compiled in (compile
floor is `trace`), so raising `RUST_LOG` lights it up with no rebuild — leave
`trace!` lines in hot code and pay nothing until you ask for them.

Under the hood: `LogRecord`, `LogBody` (`Text` / `Owned` bytes / `Structured`),
`LogBuilder` (`.level().message().module_path().file_line().trace(trace_id,
span_id).trace_flags().tag().emit()`), and the `log_record!` macro. Custom
severities via `Level::custom(name, severity)`. The level macros wrap all this —
you rarely touch it directly.

## Metric instruments

Independent of spans: declare an instrument (const-constructible, so a `static`
is the idiom) and drive it with the macros. All state lives in the struct —
`.add` / `.set` / `.record` are atomic, zero-alloc.

```rust
use proxima_telemetry::{counter, gauge, histogram, updown};
use proxima_telemetry::metric::{Counter, Gauge, UpDownCounter, Histogram};

static REQUESTS: Counter = Counter::new("http.requests").unit("{request}");
static IN_FLIGHT: UpDownCounter = UpDownCounter::new("http.in_flight");
static QUEUE_DEPTH: Gauge = Gauge::new("queue.depth");
static LATENCY: Histogram = Histogram::new("http.latency_ns");   // feature = "histogram"

counter!(REQUESTS, 1);                       // monotonic add
counter!(REQUESTS, 1, "route" = path);       // with tags
updown!(IN_FLIGHT, 1);  /* ... */  updown!(IN_FLIGHT, -1);
gauge!(QUEUE_DEPTH, depth);                  // last-value observation
histogram!(LATENCY, elapsed_ns);             // bucketed distribution
```

- `Counter` — monotonically increasing `u64` (`.add`).
- `UpDownCounter` — signed `i64` delta (`.add`, positive increments / negative decrements).
- `Gauge` — last-value, stored as bits (`.set_u64` / `.set_f64` / `.set_i64`).
- `Histogram` — fixed 32-bucket atomic distribution (`.record`); behind
  `feature = "histogram"`. The same `Histogram` type backs `#[instrument]`'s
  auto duration metric.
- Tags are accepted on every form; in v1 the accumulator is a single atomic (not
  yet sharded per attr-set — the documented opt-sweep target), so tags ride the
  sample but aren't bucketed yet.

## `#[instrument]` / `#[span]` — and its automatic metric

Same expansion, two names for intent: `span` when you mean "open a span,"
`instrument` when you mean "instrument this function." Re-exported as
`proxima::telemetry::{instrument, span}` (and `proxima_telemetry::{instrument,
span}`); with a `use`, the bare `#[instrument]` / `#[span]` works.

```rust
#[proxima::instrument]                       // name = fn name, level = info, ambient recorder
fn do_work(input: &str) -> usize { input.len() }

#[span(name = "explicit", level = "warn", kind = "server")]
async fn fetch(url: &str) -> Result<String, Error> { /* guard crosses .await */ }

#[span(err)]                                 // Result-returning: marks error status on Err
fn parse(buf: &[u8]) -> Result<Frame, DecodeError> { ... }

#[span(fields(peer = peer_addr, "http.method" = method), budget = 900_000)]
fn dispatch(...) { ... }
```

Arguments:

- `name = "..."` — span name; defaults to the function name.
- `level = "trace|debug|info|warn|error"` — defaults to `info`.
- `kind = "internal|server|client|producer|consumer"` — span kind.
- `recorder = <expr>` — a `&Recorder`; defaults to the ambient `Recorder::current()`.
- `fields(key = expr, "dotted.key" = expr, bare_ident)` — values must be
  `Into<ScalarValue>` (typed scalars, not Debug strings); a non-convertible expr
  is a compile error at the call site, which is the point.
- `err` — on a `Result` body, set error status on the `Err` path; flows every
  return (including early `return`) through the status check.
- `budget = <ns>` — tail-sampling force-keep: a head-sampled-OUT span whose
  `duration_ns` exceeds the budget keeps its trace anyway (the slow outlier is
  the trace worth keeping).

The guard is always `Option<guard>`: the ambient path resolves
`Recorder::current()` and runs the body span-free (`None`) when no recorder is
installed — the same no-op contract as the log macros. On an `async fn` the RAII
guard crosses `.await` and computes `duration_ns` on drop. Cannot be applied to a
`const fn`.

### The three pillars (the unifier)

One `#[instrument]` fans out to all three pillars from a single declaration,
replacing three ceremonies (`#[span]` for traces, `info!`/`debug!` for logs,
hand-rolled `Histogram::new()` + `Instant` deltas for metrics):

- **metric** — an always-on per-**name** duration histogram, recorded on span
  close (even sampled-OUT closes). This is the `#[instrument]`-automatic metric;
  it is alloc-free per-core (atomic-bucket rings). For ANY OTHER metric (a count,
  a gauge, a non-duration distribution) use the explicit instruments above —
  `#[instrument]` only auto-emits duration.
- **trace** — the span, sampled via the `decide()` gate.
- **log** — entry/exit, sampled (this sub-pillar is deferred; the
  always-on-metric / sampled-trace split is the load-bearing piece that landed).

Cost model facts that matter on hot paths:

- The span-metric pillar is behind a **compile-time default-off feature,
  `instrument-metrics`** (a firewall). Feature-off, `#[span]` is trace-only.
- **Consumer gate:** even feature-on, a span's duration metric records ONLY when
  something consumes it. With no consumer it is plain `Noop` — not even a clock
  read. Turn it on by subscribing: `recorder.enable_span_metrics()` (for an
  exporter / drain / test reader) or
  `recorder.set_duration_observer(|name, duration_ns| { ... })` (the C6
  control-loop feed). The `capture()` test helper subscribes automatically.
  (This gate is specific to the span duration metric; the explicit `Counter` etc.
  accumulate unconditionally.)
- **Overhead ordering** (no numbers here on purpose — a skill that quotes a
  measurement drifts from it silently, and the reader cannot tell): an unconsumed
  span is free; a sampled-out metric-only span is cheaper than a full trace; the
  span metric when consumed costs a registry by-name lock on top of the trace
  baseline, which is a known optimization target. **For the actual numbers, read
  the bench seal in `docs/unified-instrument/discipline.md`** — that log carries
  the host, the run count, and the CoV, which is what makes a number a result
  rather than an anecdote.
- Exemplars: the slowest kept span stamps its trace id onto the duration
  histogram, so the metric's slow bucket points at a retrievable trace.

## Debugging recipe (for proxima-debugger)

To make a hard bug tell you the truth instead of guessing from code:

1. Install a recorder at process start (`install_console_logging()` for a quick
   look, or `Recorder::builder().pipe(collecting_sink)` to assert on records in a
   test).
2. Add `debug!`/`trace!` at the decision point with the ACTUAL payload data as
   typed fields — the slot id, edge key, score, confidence, payload words — not a
   prose summary. Drop `#[instrument]` on the function to time-and-span it with
   zero manual wiring. Add a `Counter`/`Histogram` when you need an aggregate
   across many iterations (how often a branch fires, the distribution of a value).
3. Run with `RUST_LOG` raised to the level you emitted at (default `error` drops
   everything below). Per-target (`RUST_LOG=mycrate::module=trace`) keeps the
   noise down.
4. Drain and read: in a test, `while recorder.drain() > 0 {}` then assert on the
   captured `LogRecord`s / `SpanRecord`s / metric samples. Live, the configured
   pipe exports them.
5. If a span's duration metric is missing, check the `instrument-metrics` feature
   AND that a consumer is subscribed (`enable_span_metrics`) — both are required
   for the span-metric pillar to fire. Explicit instruments don't need either.

This is proxima's substrate for the AGENTS.md "do not guess; instrument and use
actual execution payload data" discipline — the tool that makes that rule
executable.

## Source pointers

- macros: `proxima/proxima-macros/src/span_attr.rs` (the `#[span]`/`#[instrument]`
  expansion), `proxima-macros/src/lib.rs` (proc-macro entry points).
- log macros + callsite gate: `proxima/proxima-telemetry/src/emit/macros.rs`,
  `emit/global.rs` (RUST_LOG filter), `emit/gate.rs`.
- log records: `proxima/proxima-telemetry/src/log/`.
- metric instruments + macros: `proxima/proxima-telemetry/src/metric/` (`mod.rs`
  has the `counter!`/`gauge!`/`histogram!`/`updown!` macros; `counter.rs`,
  `gauge.rs`, `updown.rs`, `histogram.rs`, `exemplar.rs`).
- recorder + install: `proxima/proxima-telemetry/src/recorder/mod.rs`,
  `src/export.rs` (`set_default_recorder`, `install_console_logging`).
- the full design + bench seal: `proxima/docs/unified-instrument/discipline.md`.
