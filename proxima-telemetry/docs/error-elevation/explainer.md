# error-elevation: a teaching walkthrough

Who this is for: you know general tracing/logging concepts — levels, spans,
trace ids, "turn on debug logging when something breaks" — but you have never
looked inside `proxima-telemetry`. Every file:line citation below is checked
against the worktree branch `feat/telemetry-error-elevation` as of this
writing; run `git log --oneline -1` if you want the exact commit context —
this feature has not landed on `main` yet.

Every code block in this document is either drawn directly from the source
(with its `file:line`) or is real output from a command that was actually run
while writing this doc — nothing here is invented or paraphrased-then-drifted.
A few source blocks are reformatted for width (a multi-line struct literal
collapsed to one line, a `#[serde(default = "...")]` attribute annotated
inline with the default it resolves to) or trimmed to the relevant lines —
every trim is marked `// ...` or "(trimmed to ...)" at the point it happens,
and no reformatting changes what the code does.

---

## 0. The vocabulary you need before "elevation" means anything

Skip this section if you already know what a `Recorder`, a drain pass, a
`Pipe`, and a `trace_id` mean in this codebase. Everyone else, read it first —
the rest of the document assumes it.

### 0.1 Levels are `severity: u8`, not an enum you match on

```rust
// src/level.rs:19-49
pub struct Level {
    severity: u8,
    name: &'static str,
}

impl Level {
    pub const TRACE: Level = Level { severity: 1, name: "trace" };
    pub const DEBUG: Level = Level { severity: 5, name: "debug" };
    pub const INFO:  Level = Level { severity: 9, name: "info" };
    pub const WARN:  Level = Level { severity: 13, name: "warn" };
    pub const ERROR: Level = Level { severity: 17, name: "error" };
    pub const FATAL: Level = Level { severity: 21, name: "fatal" };
}
```

Higher number = more severe. "At or above floor" is always `severity() >=
floor.severity()` — a plain integer comparison, not a match arm. Keep that
direction straight, because elevation's config field is called `elevated` and
it is the *fine* end (small number, e.g. `trace`), while `floor` is the
*coarse* end (bigger number, e.g. `info`) that always gets through. It reads
backwards from "floor is the bottom" if you're not paying attention: `floor`
being coarser than `elevated` is the whole point — floor is what always ships,
elevated is what you only get on request. (`Elevation::elevated` is actually
typed `Option<Level>`; `None`, the default, resolves through `Elevation::
resolved_elevated()` to `floor` — "no extra depth" rather than always
`trace`. Section 5.1 covers the resolution rule; everywhere below that says
"`elevated`" means the *resolved* value unless stated otherwise.)

### 0.2 The emit macros build a `LogRecord` only if a callsite gate says keep

`proxima_telemetry::{error!, warn!, info!, debug!, trace!}` (`src/emit/
macros.rs:156-179`) are the logging call sites — drop-in for `tracing`'s
macros syntactically, but they never touch `tracing`. Each callsite (literally
each place you write `trace!(...)` in your code) owns a `static`
[`CallsiteGate`](../../src/emit/gate.rs) (`src/emit/gate.rs:57-99`) that
caches a keep/drop decision:

```rust
// src/emit/macros.rs:32-46 (the recorder=rec form; the ambient-default form is the same shape)
macro_rules! __emit {
    ($level:expr, recorder = $recorder:expr $(, $($args:tt)*)?) => {{
        static __GATE: $crate::emit::CallsiteGate = $crate::emit::CallsiteGate::new();
        if __GATE.is_enabled(
            $crate::emit::global::current_generation(),
            || $crate::emit::global::decide(::core::module_path!(), $crate::emit::Coord::from($level)),
        ) {
            $crate::__emit_collect!(@scan $level, "", [], [$recorder], $($($args)*)?);
        } else {
            $crate::__emit_admit!($level, [$recorder], $($($args)*)?);
        }
    }};
    // ...
}
```

The important thing this buys: on a disabled callsite, the `else` branch runs
instead of `__emit_collect!` — and (until elevation exists) that `else` branch
is nothing. **A dropped record is never built.** No `LogRecord` allocation, no
field formatting, no ring push. This is different from OTel-style
"filter-at-export" designs, where every record is built and the filter runs
downstream at the exporter. proxima's filter runs at the emit call site
(`src/emit/global.rs:1-10` calls this "the std-tier half of proxima's emit
gate" and the runtime decision is `emit::global::decide`, consulted only on a
`CallsiteGate` cache miss).

The default runtime floor with no `RUST_LOG` set is `error` (`src/emit/
global.rs:6`: *"Default floor is `error`: with no `RUST_LOG` set, only
`error!` records."*). `RUST_LOG=info cargo run ...` (or `emit::global::install`
in code) raises it, same grammar as `tracing-subscriber`'s `EnvFilter`.

### 0.3 A `Recorder` is a per-core ring plus a drain pass plus a terminal `Pipe`

Emitting a log/span/metric does not send it anywhere synchronously. It pushes
onto a lock-free per-core ring (`RingSet`, `src/recorder/ring_set.rs:27`); some
later "drain pass" — a background thread, or the producer itself under
backpressure — pops a batch off the rings and hands it to the recorder's
**terminal pipe**: `TelemetryPipeHandle` (`src/pipes.rs:65`), a runtime-erased
handle over anything implementing [`SendPipe`](#04-pipe-and-sendpipe-the-two-tiers-of-the-algebra)
for `TelemetryRequest` (`src/pipes.rs:59`). `Recorder::drain` (`src/
recorder/mod.rs:1712`) runs one drain pass by hand; a `managed_drainer` (a
config flag, `src/config.rs:117`) can also run it on a background thread.

That terminal pipe is the seam elevation hooks into: instead of "one exporter,"
elevation makes the terminal pipe "two exporters wired together" (section 3).

### 0.4 `Pipe` and `SendPipe`: the two tiers of the algebra

Everything downstream of "record is built" in proxima is a `Pipe` — an
`In -> Result<Out, Err>` async function object, defined at the no_std+no-alloc
floor:

```rust
// proxima-primitives/src/pipe/primitives.rs:89-99
pub trait Pipe {
    type In;
    type Out;
    type Err: Debug + 'static;
    fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>>;
}
```

Note what is *missing*: no `Send` bound on `Self` or the returned future. This
is deliberate and stated right in the doc comment — `Pipe` is root, `Send` is
additive ("RPITIT" below is just a name for the `-> impl Future<...>` return
syntax you see above — "return-position impl trait in trait"):

> `[Pipe]` is the root form — typed In/Out/Err, RPITIT, NO `Send` bound. This
> INVERTS the legacy arrangement: local is root, `Send` is the additive
> constraint (`[SendPipe]`), because Send-everywhere is a work-stealing
> assumption and prime is per-core shared-nothing.
> — `proxima-primitives/src/pipe/primitives.rs:3-6`

`SendPipe` (`proxima-primitives/src/pipe/primitives.rs:124-147`) is the
additive cross-core form: same shape, plus `Send + Sync + 'static` on the type
and `Send` on the returned future, so a pipe can be handed to another core's
drain thread. proxima-telemetry's pipes (`FanExporter`, `FloorFilter`,
`ElevationSink`, all of `src/pipes.rs`) are `SendPipe`, because a recorder's
terminal pipe genuinely does need to run from whichever thread drains — the
managed drainer thread, or a producer's own elastic-assist drain. This is why
you will see `impl SendPipe for ElevationSink` rather than `impl Pipe`.

`Pipe::and_then` (`proxima-primitives/src/pipe/primitives.rs:104-111`) chains
two pipes into a two-stage pipeline; `SendPipe::and_then` is the same shape for
the cross-core form. Composition — "run this pipe, then that one" — is a
method call, not a new type per pairing.

### 0.5 Trace ids, span ids, and "the current span"

`TraceId` (16 bytes) and `SpanId` (8 bytes) (`src/id.rs:35-39`) identify a
trace and a span within it, W3C-`traceparent`-compatible
(`parse_traceparent`/`format_traceparent`, `src/id.rs:197-248`). A `LogRecord`
built inside a span picks up that span's `(TraceId, SpanId)` automatically —
not by the caller passing them, but by reading a thread-local "current span"
cell:

```rust
// src/log/builder.rs:88-105 (LogBuilder::emit)
pub fn emit(mut self) {
    if self.span_id.is_none()
        && let Some((trace_id, span_id)) = crate::current::current()
    {
        self.trace_id = Some(trace_id);
        self.span_id = Some(span_id);
        self.trace_flags = TraceFlags::SAMPLED;
        // ...
    }
    // ...
}
```

`crate::current` (`src/current.rs`) holds that cell: `enter`/`restore` push
and pop `(TraceId, SpanId)` as spans open and close (sync spans bracket their
whole scope — `Span::enter` calls `current::enter` when the guard is created
(`src/trace/span.rs:180`), and `Drop for SpanGuard` calls `current::restore`
when it's dropped (`src/trace/span.rs:340`); async spans bracket per-poll
instead, per `src/current.rs:13-17`). This is what "current span" means
everywhere below — a single `Cell`, not a stack data structure, with nesting
expressed by each caller holding the parent it displaced.

That's the whole vocabulary. Now the actual feature.

---

## 1. The problem

Production logging lives with a permanent trade-off: log a lot and you pay for
storage/throughput/noise on every request, all the time, forever; log a
little (say, only `info`+) and the one time you actually need to debug a
failure, the breadcrumbs that explain *why* it failed were never recorded —
they were `debug`/`trace` records that got dropped before you knew you'd want
them.

The usual answer, "just turn `debug` on," doesn't work retroactively: by the
time you notice the error and raise the log level, the request that failed is
over. You get verbosity on the *next* failure, not the one you're looking at.

error-elevation's answer: **keep a permanent floor for everyone (so the
common case stays cheap), but for a small sampled fraction of traces, quietly
build and buffer the verbose detail too — not send it anywhere, just hold it
— and if that trace ever produces an error, replay its whole buffered history,
in order, to a separate sink.** A healthy sampled trace's buffer is simply
discarded when the trace finishes. You pay the buffering cost only for the
sampled fraction, and you only ever *send* the verbose detail for traces that
actually went wrong.

The example walkthrough (`examples/elevation_walkthrough.rs:1-21`) states this
in its own header comment, and running it produces the real payoff at the end
— the full ordered tree, replayed:

```
$ cargo run -p proxima-telemetry --features elevation --example elevation_walkthrough
policy: floor=info elevated=trace sample_ratio=1 trigger=error
sink: per_trace_ring=256 max_traces=1024
buffered 3 records (1 floor+, 2 below-floor) -- elevated sink saw 0 so far

replayed tree (ordered by ts_ns):
  [ 100ns] trace handler entered: GET /users/42
  [ 200ns] debug querying users table
  [ 300ns] info  cache miss for key=user:42
  [ 400ns] error downstream timeout: connection reset

healthy trace's root closed without a trigger: elevated sink still shows 4 total records (unchanged)
```

(That is the actual output of that command, run against this worktree while
writing this doc.) Three below-floor and floor+ records went in out of
timestamp order — `300, 100, 200` — exactly as concurrent emit would produce
them; the replay came out sorted by `ts_ns` (`100, 200, 300, 400`), the
emission order, not the arrival order. And the second, healthy trace's single
`debug` record never shows up anywhere in the elevated sink at all — its
buffer was simply dropped.

`elevation: None` (the config default) or the `elevation` feature turned off
at compile time collapses this exactly back to today's behavior: one floor,
one exporter, no buffering, nothing extra. Section 5 covers both collapses in
detail; section 7 gives their actual measured cost.

---

## 2. Why tail-sampling is *forced*, not chosen

Here is the part that is easy to get wrong if you reason about this from an
OTel/tracing background instead of from this codebase's actual mechanism.

Section 0.2 established the load-bearing fact: proxima's filter runs **at the
emit call site**, before a `LogRecord` is ever constructed. `CallsiteGate`
(`src/emit/gate.rs`) plus the macro's `if __GATE.is_enabled(...) { build } else
{ nothing }` branch (`src/emit/macros.rs:35-46`) mean that at floor=`info`,
a `debug!("querying users table")` call produces **no `LogRecord` at all** —
not one that gets filtered downstream, one that was never allocated.

This is the opposite of "filter-at-export": there, every record gets built
and shipped to the exporter, which then decides what to keep. proxima's
design is deliberately the other way around — see `src/emit/gate.rs:1-13`'s
own framing: *"proxima's filter otherwise runs at drain — meaning a disabled
record is built, ringed, drained, and only then dropped. A `CallsiteGate`
moves the decision to the emit site."* That's a real perf win for the
common case (a disabled `trace!()` costs two atomic loads and a branch,
nothing more), but it has a consequence for this feature: **the full
trace/debug tree "the full picture of what happened" simply does not exist
in memory anywhere once floor=`info` has dropped it at the call site.** You
cannot "replay what you didn't record."

So error-elevation cannot be "buffer everything, decide what to send at
export time" — that would mean building every `trace!`/`debug!` record for
every request, all the time, which is exactly the cost the record-time gate
exists to avoid. Building the full tree for every trace, just in case one of
them errors, re-creates the cost this whole substrate was built to eliminate.

The only way to get "verbose detail available on error, without paying for it
on every request" is to admit a **bounded, sampled fraction** of traces into
verbose mode, and only pay the build+buffer cost for those. This is exactly
what distributed-tracing "tail sampling" means: sample after you know the
outcome (as opposed to "head sampling," deciding at trace start with no idea
whether it'll matter) — except here the "sample" decision has to happen at
trace start (an emit-time decision can't consult a future outcome), so what
actually happens is: sample a fraction of traces *up front* into "build the
verbose detail and hold it," and decide *later*, per-trace, whether to ship
what was held.

The knob that bounds this is `Elevation::sample_ratio` (`src/config.rs:213-
215`), a deterministic `TraceId`-keyed ratio — the same math
`sampler::TraceIdRatioBased` already uses:

```rust
// src/sampler.rs:102-121 — the existing sampler, for comparison
pub struct TraceIdRatioBased {
    p: f64,
    threshold: u64,
}
impl TraceIdRatioBased {
    pub fn new(p: f64) -> Self {
        let clamped = p.clamp(0.0, 1.0);
        let threshold = (clamped * u64::MAX as f64) as u64;
        Self { p: clamped, threshold }
    }
}
impl Sampler for TraceIdRatioBased {
    fn should_sample(&self, ctx: SamplingContext<'_>) -> Decision {
        let value = /* low 8 bytes of ctx.trace_id, as u64 */;
        if value < self.threshold { Decision::Keep } else { Decision::Drop }
    }
}
```

Elevation's own verbose-admission check (`src/current.rs:71-82`,
`is_verbose_trace`) reuses this *exact* threshold arithmetic against a
per-trace threshold set from `sample_ratio` — same distribution, same
determinism (same `trace_id` always gets the same verbose/not-verbose
decision), just consulted from a different place (once per span-enter,
cached — section 4) instead of from `Sampler::should_sample`. This is not a
new sampler type; it is the same math, reused, because the shape of the
problem — "deterministically admit a fraction of trace ids" — is identical.

---

## 3. The shape: an existing fan-out over two pipes, not a new "tee" type

Once you accept tail-sampling is forced, the question becomes: what actually
changes in the recorder's plumbing? The answer, verified from source, is
*less than you'd guess* — nothing new is invented at the composition level.

### 3.1 The recorder already knew how to fan out to N exporters

Before elevation existed, `proxima-telemetry` already had a way to send one
record to more than one exporter: `FanExporter` (`src/pipes.rs:81-122`),
installed by `fan_exporters` (`src/pipes.rs:426-434`) whenever
`TelemetryConfig::exporters` names two or more sinks (`src/config.rs:68-81`,
doc comment: *"Fan each record to these exporters ... a multi-sink deployment
is a TOML list, not new Rust."*).

```rust
// src/pipes.rs:81-92 — pre-existing, not part of this feature
pub struct FanExporter {
    exporters: Arc<Vec<TelemetryPipeHandle>>,
}
impl FanExporter {
    pub fn new(exporters: Vec<TelemetryPipeHandle>) -> Self { /* ... */ }
}
```

```rust
// src/pipes.rs:426-434 — pre-existing
pub fn fan_exporters(mut exporters: Vec<TelemetryPipeHandle>) -> TelemetryPipeHandle {
    match exporters.len() {
        0 => into_telemetry_handle(NullPipe::new()),
        1 => exporters.pop().unwrap_or_else(|| into_telemetry_handle(NullPipe::new())),
        _ => into_telemetry_handle(FanExporter::new(exporters)),
    }
}
```

`FanExporter::call` (`src/pipes.rs:99-121`) runs every exporter *concurrently*
(`futures::future::join`), waits on all of them, and returns the primary's
result — a broken secondary exporter is best-effort, it does not fail the
others. Elevation needed exactly this: "send one record down two independent
arms, wait on both, don't let one break the other." So it reuses
`fan_exporters` directly — `install_elevation` (`src/config.rs:549-586`, shown
in full in section 5) is literally `fan_exporters(vec![floor_arm, sink])`. No
new fan-out type was written for this feature. The discipline log for this
component names this explicitly: *"No bespoke `tee` type ... elevation is
another consumer of the same primitive, not a new one"* (`docs/error-
elevation/discipline.md:35-40`).

### 3.2 Arm A: `FloorFilter` — the normal exporter's contract, unchanged

The fan's first arm has to guarantee the normal exporter still only ever sees
floor-and-above records — exactly what it saw before elevation existed, even
though (as you'll see in section 4) below-floor records for verbose traces
now *do* reach the fan. `FloorFilter` (`src/pipes.rs:146-203`) is the pipe
that enforces that:

```rust
// src/pipes.rs:146-150
#[cfg(feature = "elevation")]
pub struct FloorFilter {
    inner: TelemetryPipeHandle,
    floor_severity: u8,
}
```

```rust
// src/pipes.rs:187-203 (SendPipe impl, trimmed to the call path)
impl SendPipe for FloorFilter {
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(&self, request: TelemetryRequest) -> impl StdFuture<...> + Send {
        self.inner.call_dyn(retain_floor(request, self.floor_severity))
    }
}
```

`retain_floor` (`src/pipes.rs:163-184`) filters a batch down to
`record.level.severity() >= floor_severity` and forwards the rest unchanged —
non-log payloads (spans, metrics) pass through untouched. `FloorFilter` wraps
whatever the *normal* exporter would have been (`Elevation::exporter` is a
completely separate field for the elevated sink, section 5), so the fan-out is
"filter, then the exporter you already had" on arm A.

The doc comment on `FloorFilter` is explicit about why it is a specific leaf
pipe and not a general filter combinator library type:

> It is a specific telemetry leaf pipe (like `FanExporter` / `NullPipe`), not
> a general filter combinator — the pipe algebra composes it via
> `and_then`/fan-out; it does not need a new library primitive.
> — `src/pipes.rs:142-144`

### 3.3 Arm B: `ElevationSink` — the buffer, the trigger, the replay

`ElevationSink` (`src/pipes.rs:229-239`) is the one genuinely new behavior in
this feature: it buffers verbose-sampled traces' records per `trace_id`, and
on a trigger-level record, replays that trace's whole ordered buffer to a
*separate* exporter (`Elevation::exporter`, not the normal one).

Its state (below) keys a map by `TraceId` in a `DashMap` — a sharded
concurrent hashmap (the `dashmap` crate): a `HashMap` that many threads can
read and write at once without one single lock guarding the whole map, which
matters here because records from many concurrent requests all land on this
same map at once.

```rust
// src/pipes.rs:237-239
#[cfg(feature = "elevation")]
pub struct ElevationSink {
    state: Arc<ElevationState>,
}

struct ElevationState {                      // src/pipes.rs:242-251
    elevated: TelemetryPipeHandle,
    buffers: DashMap<TraceId, Arc<TraceBuffer>>,
    trigger_severity: u8,
    per_trace_ring: usize,
    max_traces: usize,
    ttl_ns: u64,
    drain_on_root_close: bool,
    latest_ts_ns: AtomicU64,
    sweep_counter: AtomicU64,
}
```

Notice the shape: `ElevationSink` itself is a thin, `Clone`-cheap handle
(one `Arc`) wrapping `SendPipe`; all the real state — the per-trace map, the
knobs — lives behind that one `Arc<ElevationState>`. This is the same
structural pattern `proxima-primitives::pipe::FanIn` uses (a small `Pipe`
struct on the outside, shared concurrent state on the inside), which is why
the doc comment calls it out by name even though `ElevationSink` does not
literally implement the `FanIn` trait:

> Pipe outside, atomic state inside (the `FanIn` pattern): the `SendPipe`
> composes by type; the shared `Arc<ElevationState>` holds the concurrent
> per-trace map.
> — `src/pipes.rs:229-232` (doc comment on `ElevationSink`)

The per-trace buffer itself is a `TraceBuffer` (`src/pipes.rs:204-207`):

```rust
// src/pipes.rs:204-207
struct TraceBuffer {
    ring: LogRing<LogRecord>,
    last_touch_ns: AtomicU64,
}
```

`LogRing<LogRecord>` is where the "reuse, don't invent" discipline is most
visible. `LogRing<T>` already existed for a completely different purpose — a
live-tail log buffer, `LogRing<String>`, folded into this crate from a former
satellite crate (`src/log_buffer/ring.rs:1-14`). It was generic-*capable* but
only ever instantiated with `String`:

```rust
// src/log_buffer/ring.rs:22-28
/// Generic over the element `T`, defaulting to `String` so the live-tail
/// log path (`LogBuffer`) is unchanged. The elevation path reuses it as
/// `LogRing<LogRecord>` — a per-trace replay ring — rather than minting a
/// second bounded-ring primitive.
pub struct LogRing<T = String> {
    lines: ArrayQueue<T>,
}
```

Elevation's per-trace buffer is `LogRing<LogRecord>` — the *same* bounded,
oldest-evicting ring, safe for multiple producers and consumers to share
concurrently ("MPMC," multi-producer multi-consumer — `crossbeam_queue::
ArrayQueue` underneath), just holding a different element type. Nothing about `LogRing`'s logic changed;
its type parameter was already there, unused for anything but `String` until
this feature used it. This is principle-1 RISC reuse in the most literal
sense: not "a ring shaped like `LogRing`," the actual `LogRing`.

The ingest+replay logic lives on `ElevationState` as plain methods (not
trait methods — this is internal, not a public API surface):

- `buffer_for` (`src/pipes.rs:285-297`) gets-or-creates a trace's buffer,
  calling `enforce_cap` first if the map is full (section 6).
- `ingest_log` (`src/pipes.rs:324-343`) is the per-record hot path: skip
  anything not `VERBOSE_BUFFERED` (section 4), push it onto that trace's
  ring, and if its severity is at or above `trigger_severity`, *remove* the
  buffer from the map and hand its drained, `ts_ns`-sorted contents to the
  caller as a replay request.
- `observe_span` (`src/pipes.rs:345-350`) watches for root-span closes —
  the semantic completion signal (section 6).
- `ingest` (`src/pipes.rs:351-378`) dispatches a whole `TelemetryRequest`
  (which may be a batch) to `ingest_log`/`observe_span` per record.
- `maybe_sweep` (`src/pipes.rs:381-399`) is the amortized TTL sweep, run
  once every `SWEEP_EVERY = 64` calls (`src/pipes.rs:221`), not per record.

`ElevationSink::call` (`src/pipes.rs:402-421`) ties it together: ingest the
request, and for every replay it produced, best-effort-send it to the
elevated exporter (*"a broken elevated sink must not fail the drain"*, mirror
of `FanExporter`'s own best-effort-secondaries policy from section 3.1).

### 3.4 Composition: `and_then`/`SendPipe`, not a bespoke wiring type

Nowhere in this shape is there a new combinator trait. `FloorFilter` and
`ElevationSink` are each an ordinary `SendPipe<In = TelemetryRequest, Out =
Response<Bytes>, Err = ProximaError>` — the exact same contract every other
telemetry exporter pipe in this crate satisfies (`OtlpHttpPipe`,
`OtlpGrpcPipe`, `NullPipe`, all in `src/pipes.rs`). The fan-out that combines
them is `fan_exporters` — a function, called with a two-element `Vec`
(`src/config.rs:585`, shown in section 5). There is no `Elevation<Floor,
Sink>` wrapper type, no new trait to implement, no macro. This is what
`Pipe`/`SendPipe`'s `and_then` (section 0.4) and the pre-existing
`fan_exporters` buy: composing new *behavior* out of old *pieces*.

### 3.5 Why this is the disciplined answer, not just a convenient one

This codebase has a standing rule (stated in the workspace's binding
guiding-principles document, principle 1, "RISC reuse first"): before adding
any new type, ask whether an existing primitive already does the job, and if
in doubt, don't add the type. Every piece of elevation's shape is an answer to
that question, decided in the affirmative for reuse:

| need | reused primitive | new code |
|---|---|---|
| send one record to two arms | `FanExporter`/`fan_exporters` (already existed) | none |
| "fire when level >= X" | a bare `Level` comparison (`Elevation::trigger_level: Level`) | none — no `TriggerSpec` type |
| a bounded per-trace log buffer | `LogRing<T>`, generic parameter already present, only ever used with `T=String` before | none — same type, new instantiation |
| per-trace shared state behind a cheap pipe handle | the same "Pipe outside, Arc state inside" shape as `FanIn` | `ElevationSink`/`ElevationState` — the one genuinely new type, because "buffer per trace_id, replay on trigger" has no existing home |
| below-floor-but-verbose admission | the emit macro's existing gate structure, extended with one more branch | `__emit_admit!` (feature-gated) |
| deciding which traces are verbose | the same threshold math as `TraceIdRatioBased` | a parallel, thread-local-cached ("TLS," thread-local storage) copy of that math (section 4 explains why it couldn't just call the sampler) |

`ElevationSink` is the one component that is a genuinely new struct, because
nothing in the codebase already did "buffer per trace_id with layered
eviction and replay on trigger." Everything else is either literally reused
or a one-line reuse of an existing shape (a `Level` comparison instead of a
predicate type).

---

## 4. The hot-path mechanism: one `Cell::get`, decided once per span

Section 2 established *why* verbose admission has to be a sampled, per-trace
decision. This section is *how* that decision reaches the log macro cheaply
enough to not undo the callsite gate's whole reason for existing.

### 4.1 The naive version would be too expensive

If a below-floor `trace!()` call had to ask "is my trace verbose-sampled?"
by hashing the trace id and comparing against a ratio threshold on *every
single call*, that would mean re-doing `TraceIdRatioBased`-style work on
every dropped record — exactly the kind of per-record cost the whole
record-time-gate design exists to avoid paying on the disabled path.

### 4.2 The actual version: decide once per span-enter, cache the bit

`crate::current` (introduced in section 0.5 for "current span") is extended,
under the `elevation` feature only, with a second thread-local: a `Cell<bool>`
that says "is the currently-entered trace verbose."

```rust
// src/current.rs:43-46
#[cfg(feature = "elevation")]
std::thread_local! {
    static CURRENT_VERBOSE: Cell<bool> = const { Cell::new(false) };
}
```

It is set exactly where the span cell itself is set — inside `enter`/
`restore`, i.e. once per span-enter/exit (sync scope, or once per poll for
async — same granularity `current` itself already uses), never once per
record:

```rust
// src/current.rs:107-117
pub fn enter(trace: TraceId, span: SpanId) -> Option<(TraceId, SpanId)> {
    #[cfg(feature = "elevation")]
    set_verbose_for(Some((trace, span)));
    CURRENT.with(|cell| cell.replace(Some((trace, span))))
}

pub fn restore(parent: Option<(TraceId, SpanId)>) {
    #[cfg(feature = "elevation")]
    set_verbose_for(parent);
    CURRENT.with(|cell| cell.set(parent));
}
```

`set_verbose_for` (`src/current.rs:84-88`) calls `is_verbose_trace`
(`src/current.rs:71-82`) — the same `TraceIdRatioBased`-shaped threshold
check from section 2, against a `VERBOSE_THRESHOLD` atomic set once, at
elevation install time, from `Elevation::sample_ratio` (`current::
set_verbose_ratio`, `src/current.rs:161-163`, called from `install_elevation`
in `src/config.rs:561`) — and stores the result in `CURRENT_VERBOSE`.

The log macro's below-floor admit check then costs exactly one `Cell::get`
plus (only if that read is `true`) one relaxed atomic load against a second
static, `VERBOSE_ADMIT_FLOOR` (the `elevated` depth, `src/current.rs:60`),
also set once at install:

```rust
// src/current.rs:102-105
pub fn should_admit_below_floor(severity: u8) -> bool {
    is_current_verbose() && severity >= VERBOSE_ADMIT_FLOOR.load(Ordering::Relaxed)
}
```

No re-hash of the trace id. No map lookup. No sampler recompute. The
`&&` short-circuits before the atomic load on the (overwhelmingly common)
`false` case. Section 7 shows this really does cost what it claims to.

### 4.3 The macro's below-floor admit branch

Recall from section 0.2 that a gated-off callsite's `else` branch used to be
nothing. With `elevation` compiled in, it is one more check:

```rust
// src/emit/macros.rs:67-76
#[cfg(feature = "elevation")]
macro_rules! __emit_admit {
    ($level:expr, [$($sink:tt)*], $($args:tt)*) => {
        if $crate::current::should_admit_below_floor($level) {
            $crate::__emit_collect!(@scan $level, "", [], [$($sink)*], $($args)*);
        }
    };
}
```

```rust
// src/emit/macros.rs:78-83 — the feature-off form
#[cfg(not(feature = "elevation"))]
macro_rules! __emit_admit {
    ($($args:tt)*) => {};
}
```

With `elevation` off, `__emit_admit!` expands to literally nothing — the same
empty `else` branch as before this feature existed, byte-for-byte. With
`elevation` on but no trace currently verbose, it costs the one `Cell::get`
from section 4.2 and stops. Only for a genuinely verbose-sampled trace does it
fall through to `__emit_collect!` and actually build the `LogRecord` — the
exact same builder path a normal, floor-passing `error!()` call uses.

### 4.4 Marking the record so `ElevationSink` knows to keep it

A record built via the admit branch (or a floor+ record from inside a
verbose trace) needs to carry a marker so `ElevationSink` (section 3.3) knows
to buffer it, and `FloorFilter` (section 3.2) knows the normal exporter must
never see it if it's below floor. That marker is a bit in `TraceFlags`
(`src/id.rs`), stamped at the point the record actually correlates to the
current span:

```rust
// src/id.rs:44-64
impl TraceFlags {
    pub const SAMPLED: Self = Self(0x01);
    pub const NOT_SAMPLED: Self = Self(0x00);
    /// proxima-local marker (bit `0x02`): this record belongs to a
    /// verbose-buffered (error-elevation) trace, so `ElevationSink` retains
    /// it for a possible replay. NOT a W3C-standard flag — it is stamped on
    /// the `LogRecord` only, never on the span context that serializes to
    /// the outbound `traceparent`, so it stays in-process.
    pub const VERBOSE_BUFFERED: Self = Self(0x02);

    pub const fn with_verbose_buffered(self) -> Self { Self(self.0 | Self::VERBOSE_BUFFERED.0) }
    pub const fn is_verbose_buffered(self) -> bool { self.0 & Self::VERBOSE_BUFFERED.0 != 0 }
}
```

The stamp happens in `LogBuilder::emit` (introduced in section 0.5), right
next to where `trace_id`/`span_id` get filled in from the current span:

```rust
// src/log/builder.rs:93-105
if self.span_id.is_none()
    && let Some((trace_id, span_id)) = crate::current::current()
{
    self.trace_id = Some(trace_id);
    self.span_id = Some(span_id);
    self.trace_flags = TraceFlags::SAMPLED;
    #[cfg(feature = "elevation")]
    if crate::current::is_current_verbose() {
        self.trace_flags = self.trace_flags.with_verbose_buffered();
    }
}
```

The doc comment on the constant is explicit that this bit is process-local
only — it is never encoded into the W3C `traceparent` bytes that leave the
process (`format_traceparent`, section 0.5), so a downstream service never
sees or has to understand it. It rides the in-memory `LogRecord` only, from
the moment it's built to the moment `ElevationSink` reads it.

### 4.5 Proof, end to end, with real macro calls

Everything above is individually true, but a fair question is whether it
actually composes: does a real `trace!()` call, inside a real span, on a real
`Recorder`, actually get buffered and replayed? `tests/elevation_e2e.rs`
answers that directly (its own header explains why it exists — the unit
tests in `src/pipes.rs` drive `ElevationSink` with hand-built `LogRecord`s,
this test drives the *macro* end instead):

```rust
// tests/elevation_e2e.rs:162-167
let _span = recorder.span("verbose-request").start();
trace!(recorder = &recorder, "handler entered");
info!(recorder = &recorder, "cache miss");
error!(recorder = &recorder, "downstream timeout");
```

Running it (`cargo nextest run -p proxima-telemetry --features elevation
--test elevation_e2e`, actually run against this worktree):

```
PASS [   0.011s] proxima-telemetry::elevation_e2e error_inside_verbose_span_replays_full_tree_end_to_end
```

The test's own assertions (`tests/elevation_e2e.rs:170-224`) confirm: the
normal sink gets exactly the 2 floor+ records (`info`, `error`), never the
`trace` one; the elevated sink gets all 3, ordered by `ts_ns`, every one
carrying `VERBOSE_BUFFERED`. A second, non-verbose trace (`sample_ratio=0.0`)
run through the identical macro sequence produces only the single `error` on
the normal sink and *nothing* on the elevated sink — the below-floor calls
were never even admitted, so no below-floor `LogRecord` for that trace ever
existed at all.

### 4.6 Two independent floors, and why that's the documented design, not a bug

Read `tests/elevation_e2e.rs` closely and you'll notice something that is
easy to miss: `global::install(EnvFilter::parse(""))` (the *process-wide*
callsite floor, section 0.2, default `error`) and `FloorFilter::new(Level::
INFO, ...)` (`Elevation::floor`, section 3.2) are set **independently**, and
`install_elevation` (`src/config.rs:549-586`, section 5) never calls
`emit::global::install` itself — confirmed by grep, there is no
`global::install` call anywhere outside `src/emit/macros.rs`'s own tests and
`emit::global` itself.

That means these are genuinely two separate knobs:

1. the **process-wide runtime callsite floor** (`RUST_LOG`, or
   `emit::global::install` in code) — decides, for a *non-verbose* trace,
   whether a record is built at all;
2. **`Elevation::floor`** — decides, for records that *did* reach the
   fan-out (via the normal Keep path, or via a verbose trace's below-floor
   admit path), which ones the normal exporter arm keeps.

If you set `Elevation { floor: Level::INFO, .. }` but leave `RUST_LOG` (or
the compiled-in default) at `error`, a non-verbose trace's `info!()` calls
never even reach the fan — they were dropped at the callsite gate before
`FloorFilter` ever saw them, exactly as they would be without elevation at
all. `Elevation::floor` only controls what happens to records that got past
the *first* gate; it does not raise that first gate for you.

This was flagged as an open gotcha in an earlier revision of this document;
it is now the **documented design**, decided deliberately rather than left
unresolved. `Elevation::floor`'s own doc comment (`src/config.rs:200-205`)
states the rule plainly: it SHOULD equal the operator's effective emit gate
floor, and the two are **not auto-synced** — not because nobody got around to
it, but because `RUST_LOG`/`EnvFilter` is inherently **per-module**
(`module_path => level` pairs), so there is no single scalar "the process
floor" for `install_elevation` to read and mirror. Aligning the two is the
operator's responsibility, the same way an operator is already responsible
for keeping any two related config knobs consistent; `Elevation::floor`
defaulting to `info` (`default_elevation_floor`, `src/config.rs:236-238`)
gives that alignment a sane starting point rather than leaving it to a wider
mismatch by omission — but it cannot chase an arbitrary per-module
`RUST_LOG` automatically. A mismatch doesn't corrupt anything; it just makes
the normal sink's *effective* floor the coarser of the two knobs, silently,
which is exactly why this section exists — know to check both.

(There is a third, unrelated floor you may see mentioned elsewhere in this
crate: `sized::EMIT_COMPILE_FLOOR` (`src/lib.rs:32-53`), a *compile-time*
floor below which a callsite's code is `const`-folded away entirely — default
`trace`, meaning it essentially never matters unless you rebuild with a raised
`[emit] max_level` in `proxima-telemetry.toml`. It is orthogonal to both
floors above; mentioned here only so you don't confuse a third name for
"floor" with either of the two that actually interact with elevation.)

---

## 5. The config surface

### 5.1 `Elevation` and `Retention`

```rust
// src/config.rs:199-230
pub struct Elevation {
    #[serde(default = "default_elevation_floor")] // = Level::INFO
    pub floor: Level,
    #[serde(default)]                              // = None -> resolves to floor
    pub elevated: Option<Level>,
    pub sample_ratio: f64,
    #[serde(default = "default_trigger_level")]   // = Level::ERROR
    pub trigger_level: Level,
    #[serde(default)]                              // = ExporterChoice::Noop
    pub exporter: ExporterChoice,
    #[serde(default)]
    pub retention: Retention,
}

impl Elevation {
    pub fn resolved_elevated(&self) -> Level {
        self.elevated.unwrap_or(self.floor)
    }
}
```

```rust
// src/config.rs:253-271
pub struct Retention {
    #[serde(default = "default_true")]     // = true
    pub drain_on_root_close: bool,
    #[serde(default = "default_ttl_millis")] // = 60_000
    pub ttl_millis: u64,
    #[serde(default)]                        // = 0 -> build-time sized default
    pub max_traces: usize,
    #[serde(default)]                        // = 0 -> build-time sized default
    pub per_trace_ring: usize,
}
```

`floor` and `elevated` each getting their own field-level `#[serde(default)]`
(rather than the whole `Elevation` struct being all-or-nothing) is what makes
an `[elevation]` TOML table naming only `sample_ratio` deserialize
successfully — `sample_ratio` is the one field with no default, so it's the
only one an operator's config file is actually required to set. `elevated:
Option<Level>` defaulting to `None`, resolved through `resolved_elevated()` to
`floor`, is a deliberate "coupled defaults" choice, not an oversight: a
separately-drifting default for `elevated` would have meant two levels an
operator has to keep in sync by hand instead of one (see section 4.6).

Both derive `Serialize`/`Deserialize`; `Level` itself gained a hand-written
serde impl for this feature so a config file reads level *names*, not raw
severity integers:

```rust
// src/level.rs (diff against pre-elevation main)
impl serde::Serialize for Level {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.name)
    }
}
```

Confirmed round-tripping by `config::tests::elevation_level_names_are_the_
wire_form` (`src/config.rs:1582-1588`) — a serialized `Elevation` literally
contains the substrings `"info"` and `"trace"`, not `9`/`1`. The default
resolution itself is proven by `config::tests::elevated_defaults_to_floor`
(unset `elevated` resolves to `floor`) and `config::tests::
elevation_loads_from_toml_with_defaults` (a partial `[elevation]` table
naming only `sample_ratio` loads, with `floor` defaulting to `info` and
`elevated` resolving to it) — both in the same `src/config.rs mod tests`.

### 5.2 `elevation: Option<Elevation>` on `TelemetryConfig`

```rust
// src/config.rs:92-99
/// Error-elevation policy. `None` (default) is the **simple form**: today's
/// record-time gate, no per-trace buffer stage installed — genuinely
/// zero-cost. `Some` installs the tail-sampled replay: a floor level always
/// emits; a sampled fraction of traces, on an error trigger, replay their
/// full tree down to `elevated` to a separate exporter. See [`Elevation`].
#[setting(skip)]
#[serde(default)]
pub elevation: Option<Elevation>,
```

The builder surface (this crate's config surface always pairs a fluent
builder with the serde form, per this workspace's config-and-builder
discipline): `TelemetryConfig::layered().with_elevation(elevation)` /
`.with_no_elevation()` (`src/config.rs:743-756`) — the latter explicitly
exists so a later layer in a composed config chain can turn elevation back
off, verified by `config::tests::layered_with_elevation_then_no_elevation_
collapses` (`src/config.rs:1617-1624`): call `with_elevation` then
`with_no_elevation`, and the later call wins.

### 5.3 A real config, both ways

TOML (an operator's config file, every field named explicitly):

```toml
[elevation]
floor = "info"
elevated = "trace"
sample_ratio = 0.01
trigger_level = "error"

[elevation.exporter]
kind = "otlp_http"
endpoint = "https://forensics-collector:4318"

[elevation.retention]
drain_on_root_close = true
ttl_millis = 60000
max_traces = 1024
per_trace_ring = 256
```

Or the minimal form an operator who only wants the always-on floor behaviour
would actually write — `floor`, `elevated`, `trigger_level`, `exporter`, and
`retention` all have serde defaults (`floor` -> `info`, `elevated` -> `None`
-> resolves to `floor`), so `sample_ratio` is the only field a config file is
required to set:

```toml
[elevation]
sample_ratio = 0.01
```

`config::tests::elevation_loads_from_toml_with_defaults` (section 5.1) proves
exactly this minimal form loads, with `floor` resolving to `info` and
`elevated` resolving to it too.

(`ExporterChoice`'s `OtlpHttp` variant, and the `kind = "otlp_http"` tag above,
only exist when the *separate* `otlp-http` Cargo feature is also enabled —
`#[cfg(feature = "otlp-http")] OtlpHttp { endpoint: String }`, `src/
config.rs:354`; `elevation` alone gets you `ExporterChoice::Noop`, which
discards every replay. `elevation` and `otlp-http` are independent features
you turn on together for a real elevated sink.)

Fluent (composed in code, mirroring the example walkthrough,
`examples/elevation_walkthrough.rs:107-117`):

```rust
let elevation = Elevation {
    floor: Level::INFO,
    elevated: Some(Level::TRACE),
    sample_ratio: 0.01,
    trigger_level: Level::ERROR,
    exporter: ExporterChoice::OtlpHttp { endpoint: "https://forensics-collector:4318".into() },
    retention: Retention::default(),
};
let config = TelemetryConfig::builder().elevation(elevation).build();
```

Both forms round-trip through serde (`config::tests::elevation_round_trips_
through_serde`, `src/config.rs:1496-1512`) — build one, serialize, deserialize,
get the same policy back.

### 5.4 What actually gets installed: `install_elevation`

```rust
// src/config.rs:549-586
#[cfg(feature = "elevation")]
fn install_elevation(cfg: &TelemetryConfig, terminal: TelemetryPipeHandle) -> TelemetryPipeHandle {
    let Some(elevation) = &cfg.elevation else {
        return terminal;
    };
    crate::current::set_verbose_ratio(elevation.sample_ratio);
    crate::current::set_verbose_admit_floor(elevation.resolved_elevated());
    let floor_arm =
        into_telemetry_handle(crate::pipes::FloorFilter::new(elevation.floor, terminal));
    let elevated = pipe_from_choice(&elevation.exporter);
    let max_traces = if elevation.retention.max_traces == 0 {
        crate::sized::ELEVATION_MAX_TRACES
    } else {
        elevation.retention.max_traces
    };
    let per_trace_ring = if elevation.retention.per_trace_ring == 0 {
        crate::sized::ELEVATION_PER_TRACE_RING
    } else {
        elevation.retention.per_trace_ring
    };
    let ttl_ns = elevation.retention.ttl_millis.saturating_mul(1_000_000);
    let sink = into_telemetry_handle(crate::pipes::ElevationSink::new(
        elevated, elevation.trigger_level, per_trace_ring, max_traces, ttl_ns,
        elevation.retention.drain_on_root_close,
    ));
    fan_exporters(alloc::vec![floor_arm, sink])
}
```

Read top to bottom, this is section 3 and section 4's wiring made concrete:
arm the thread-local verbose-sampling threshold and admit floor (section 4.2), wrap
whatever the normal exporter was in `FloorFilter` (section 3.2), resolve
`0` retention fields to build-time `sized` defaults (section 5.5), build the
`ElevationSink` (section 3.3), and fan the two arms together with the
pre-existing `fan_exporters` (section 3.1). `None` short-circuits at the top
and returns the `terminal` pipe completely untouched — the simple form.

This is called from `Recorder::from_config_with_pipe` (`src/config.rs:440-
508`), feature-gated:

```rust
// src/config.rs:459-463
// elevation (when configured) rewraps the terminal pipe as a fan-out over
// [FloorFilter -> terminal, ElevationSink] and arms the verbose sampler.
// None / feature-off leaves the terminal pipe exactly as passed in.
#[cfg(feature = "elevation")]
let pipe = install_elevation(cfg, pipe);
```

Note also `src/config.rs:485-495`: a fan (either the pre-existing
multi-exporter fan or elevation's own fan) needs `RecordSharing::Arc` so the
fan clones the drained record cheaply (a refcount bump on the `*BatchArc`
drain form) instead of deep-copying it per arm — and `from_config_with_pipe`
detects `cfg.elevation.is_some()` and selects `Arc` sharing automatically for
that reason, the same way it already did for `cfg.exporters.len() >= 2`.

### 5.5 Build-time sizing: one source of truth for "how much memory can this cost"

`Retention::max_traces` and `Retention::per_trace_ring` default to `0`, and
`0` is resolved (section 5.4) to build-time constants generated by `build.rs`
from `proxima-telemetry.toml`:

```toml
# proxima-telemetry.toml:79-92
[elevation]
# Max concurrently-buffered traces (the hard count-cap backstop).
max_traces = 1024
# Per-trace replay ring capacity (records).
per_trace_ring = 256
```

```rust
// build.rs:170-174 (the generated output, at src/lib.rs:28-30's OUT_DIR include)
pub const ELEVATION_MAX_TRACES: usize = 1024;
pub const ELEVATION_PER_TRACE_RING: usize = 256;
```

Both are overridable per-build via `PROXIMA_TELEMETRY_ELEVATION_MAX_TRACES=...
cargo build` (`build.rs`'s own comment) and per-*process* by setting a
non-zero `Retention` field — "one source of truth" here means the compile-time
default and the runtime override both resolve through the same `sized`
module, not two independently-drifting numbers.

### 5.6 The `elevation` Cargo feature: default-off

```toml
# Cargo.toml:148-155
# error-elevation: tail-sampled trace buffering. A configurable floor level is
# always emitted; for a sampled fraction of traces, an error trigger replays that
# trace's full tree down to a configurable elevated level to a separate exporter.
# Default-off (disciplined-component gate 1): a build without it is byte-identical
# to today — the widened current-span verbose bit, the macro admit branch, and the
# ElevationSink/FloorFilter pipes do not exist. Implies std (DashMap per-trace map,
# the pipe chain, config). See docs/error-elevation/discipline.md.
elevation = ["std"]
```

`elevation = ["std"]` — it implies `std` (not the other way around), because
the per-trace map (`DashMap`) and the pipe/config machinery are inherently
std-tier. A build that never enables `elevation` compiles none of
`FloorFilter`, `ElevationSink`, `CURRENT_VERBOSE`, `VERBOSE_THRESHOLD`,
`VERBOSE_ADMIT_FLOOR`, or the non-empty `__emit_admit!` arm — they are
`#[cfg(feature = "elevation")]`-gated out of existence, not merely unused.

### 5.7 Two collapses, verified, not asserted

Both size-and-cost claims in section 1 and section 3 are things you can check
yourself, and were checked while writing this doc:

**Feature-off is compile-time byte-identical.** With `elevation` uncompiled,
`__emit_admit!` (section 4.3) expands to nothing, so a disabled `trace!()`
callsite is exactly what it was before this feature existed — no new code
path exists to be identical *to*, because none of the elevation types compile
in the first place.

**`None` at runtime is a no-op fan.** `install_elevation` (section 5.4)
returns `terminal` untouched when `cfg.elevation` is `None` — no
`FloorFilter`, no `ElevationSink`, no fan-out wrapper — and
`VERBOSE_THRESHOLD` defaults to `AtomicU64::new(0)` (`src/current.rs:54`), so
`is_verbose_trace` returns `false` before it even reads a trace id
(`src/current.rs:73-76`: `if threshold == 0 { return false; }`). The
gate-disabled admit check's cost is exactly the one `Cell::get` from section
4.2, whether or not elevation was ever configured for this process.

Test-count proof, both feature states, actually run against this worktree
while writing this doc:

```
$ cargo nextest run -p proxima-telemetry
     Summary [   0.587s] 425 tests run: 425 passed, 0 skipped

$ cargo nextest run -p proxima-telemetry --features elevation
     Summary [   0.568s] 431 tests run: 431 passed, 0 skipped
```

(`docs/error-elevation/discipline.md:132-138` records the current +6 delta
breakdown: 4 in `pipes::elevation_sink_tests` (`src/pipes.rs`) + 1
`current::tests::verbose_bit_follows_current_trace` (`src/current.rs`) + 1
`tests/elevation_e2e.rs` end-to-end test — `src/config.rs`'s elevation tests
swap one `#[cfg]`-gated test for another between the two feature states,
contributing net zero to the delta either way. Two more tests,
`config::tests::elevated_defaults_to_floor` and `config::tests::
elevation_loads_from_toml_with_defaults`, are `#[cfg]`-unconditional and so
count identically in both totals — they prove the `elevated`-defaults-to-
`floor` resolution rule, not anything feature-gated. Every one of the other
425 tests (423 pre-existing + these 2 new unconditional ones) is the
identical test passing in both runs — proof the feature adds behavior
without changing anything that already existed.)

### 5.8 Validation: a policy the build can't honor fails loud

`TelemetryConfig::validate` (`Validate` impl, `src/config.rs:366-427`) checks,
when `elevation` is `Some`:

- `sample_ratio` is in `0.0..=1.0` (`config::tests::validate_rejects_out_of_
  range_sample_ratio`, `src/config.rs:1608-1615`);
- `resolved_elevated()` is no coarser than `floor` (`elevated.severity() <=
  floor.severity()` — recall section 0.1's direction; a `floor=info,
  elevated=warn` policy would buffer *nothing* extra, so it's rejected as
  presumably a mistake — `config::tests::validate_rejects_elevated_coarser_
  than_floor`, `src/config.rs:1591-1606`);
- (feature-off only) `elevation` being `Some` at all is itself an error —
  *"fail loud rather than silently ignore a policy the build can't
  honour"* (`src/config.rs:414-419`), proven by `config::tests::elevation_
  set_without_feature_is_rejected` (`src/config.rs:1570-1581`).

---

## 6. Eviction: three layers, three different failure modes

A per-trace buffer has to be reclaimed eventually, or `ElevationSink`'s map
grows without bound. `Retention` (section 5.1) encodes three independent
reclaim mechanisms, each answering a different question:

**Root-span close — the expected case.** `ElevationState::observe_span`
(`src/pipes.rs:345-350`) watches `SpanRecord`s flowing through the fan; when
one has `parent_span_id: None` (i.e. it's a root span — recall `SpanRecord`,
`src/trace/span.rs:15-30`) and `drain_on_root_close` is set (default `true`),
its trace's buffer is simply removed from the map. This is "the request
finished, and it never triggered — throw away what we held." Test:
`root_span_close_drops_untriggered_buffer` (`src/pipes.rs:4736-4757`) proves
a record emitted *after* the root close, on the same trace id, only replays
that later record — the pre-close tree it dropped never comes back, even on
a subsequent error.

**TTL — the crash / lost-root fallback.** If the root span's close was never
observed (the process crashed mid-request, a span was dropped, whatever),
`Retention::ttl_millis` (default 60 seconds) bounds how long a stale buffer
survives. `maybe_sweep` (`src/pipes.rs:381-399`) runs an amortized cutoff scan
every 64 calls (`SWEEP_EVERY`, `src/pipes.rs:221`) — not per record, since a
`DashMap::retain` scan over every buffered trace on every call would defeat
the point of section 4's cheap-hot-path design. `ttl_millis: 0` disables the
sweep entirely (`if self.ttl_ns == 0 { return; }`, `src/pipes.rs:382-384`).

**Count-cap — the hard OOM backstop.** Even with root-close and TTL both
working correctly, a flood of *concurrently open* verbose-sampled traces
could still grow the map arbitrarily large before either mechanism fires.
`enforce_cap` (`src/pipes.rs:300-313`) is checked on every `buffer_for` call
(i.e. every time a *new* trace id is first seen): once the map reaches
`max_traces`, it evicts the least-recently-touched trace (by `last_touch_ns`)
before inserting the new one — a single linear scan, but only when the map is
already full. `count_cap_bounds_concurrent_traces` (`src/pipes.rs:4761-4777`)
proves this directly: cap the sink at 2 traces, push records for 3 distinct
trace ids, and the map never exceeds 2.

All three exist together because each covers a gap the others don't: root-
close is precise but only fires on the happy path; TTL covers the case
root-close never fires but is slow (minutes, by default); count-cap is the
only one that bounds memory under a true flood, instantly, regardless of
whether any individual trace ever finishes or times out.

---

## 7. The perf trade, honestly labeled

Per this workspace's rule that a load-bearing performance claim needs a
measurement in the same breath as the claim (not "crypto is probably fine" —
an actual number), here is what was actually measured, on this machine,
while writing this doc — not copied from anywhere without re-running it:

```
$ cargo bench -p proxima-telemetry --features elevation --bench bench_elevation -- --quick
elevation_admit_check/not_verbose            time: [778.33 ps 784.66 ps 786.24 ps]
elevation_admit_check/verbose                time: [777.75 ps 779.52 ps 779.96 ps]
elevation_macro_emit/none_not_verbose        time: [68.117 ns  68.264 ns  68.851 ns]
elevation_macro_emit/verbose_admit_below_floor time: [1.6762 µs 1.6805 µs 1.6816 µs]
```

This is a dev laptop, not a quiet dedicated bench host — treat the digits as
"same order of magnitude, reproducible," not as a number to publish. (The
component's discipline log, `docs/error-elevation/discipline.md:126-147`,
recorded very similar numbers on an earlier run — 753.8/768.2 ps and 68.0
ns/1.659 µs — so this reproduces.)

**Reading the two groups honestly** (the bench file's own doc comment,
`benches/bench_elevation.rs:1-17`, states the intent — this is what the
numbers actually show):

- `elevation_admit_check` isolates *only* the branch section 4.2 added:
  `should_admit_below_floor`, not-verbose vs verbose. The two numbers
  (~778 ps vs ~780 ps) are within noise of each other — both pay one
  `Cell::get`; verbose additionally pays one relaxed atomic load, and that
  load is too cheap to separate from measurement noise at this sample count.
  This is exactly what section 4.2's design predicts: no map, no recompute,
  no hash.
- `elevation_macro_emit` runs the same delta through a *real* `trace!()`
  call into a real recorder, so it also carries whatever happens *after* the
  admit check passes: building a `LogRecord`, tagging it, pushing it through
  the ring. That's where the real cost lives — `none_not_verbose` (~68 ns,
  the callsite gate rejects before any of that runs) vs
  `verbose_admit_below_floor` (~1.68 µs, ~24x) — and that ~24x is paid **only
  for the sampled fraction of traces** (`sample_ratio`), never for the
  overwhelming majority that aren't verbose-admitted. The ~68 ns baseline is
  not a cost this feature adds; it is what a gated-off `trace!()` already
  cost before elevation existed at all — the callsite gate, doing its job.

**What this does and does not buy you, stated plainly:**

- The *normal* export path — `FloorFilter`'s arm, and the drain cadence
  around it — is unchanged. Elevation does not make normal floor+ logging
  faster.
- A latency/throughput win only shows up if you *use* elevation to justify
  **raising** your normal floor (e.g. run at `info` instead of `debug`
  everywhere, because you no longer lose the below-`info` detail on a
  failure) — that's a `RUST_LOG`/callsite-gate change (section 0.2), not
  something elevation does automatically. Elevation is what makes raising the
  floor *safe to do* without losing forensic detail on the traces that break.
- Memory is spikier under elevation (a burst of concurrently-verbose,
  never-triggering traces holds their buffers until root-close/TTL/cap), but
  it is hard-bounded: `max_traces * per_trace_ring` records, times
  `sample_ratio`'s effect on how many traces are ever verbose at once. With
  the build-time defaults (section 5.5), that's at most `1024 * 256 =
  262,144` `LogRecord`s buffered at any instant, regardless of total request
  volume.

If you want to see this yourself rather than trust this document: `cargo run
-p proxima-telemetry --features elevation --example elevation_walkthrough`
(section 1's output) and `cargo bench -p proxima-telemetry --features
elevation --bench bench_elevation` are both real, runnable, and were both
actually run to produce the output quoted in this document.

---

## 8. Where this fits, and what to read next

- `docs/error-elevation/discipline.md` — the component's own discipline-log
  entry: which guiding principles it engaged, a component-by-component
  status table, and the test-gate/bench numbers as of its own last update
  (some of which this document re-verified and, in section 5.7, found one
  test-count off by one — the discipline log is a point-in-time record, this
  document's numbers are what's true of the worktree right now).
- `docs/tracing/exporter-composition.md` — the wider "an exporter is a config
  point, not a type" argument this feature's `Elevation::exporter` field
  participates in (it resolves through the same `pipe_from_choice`,
  `src/config.rs:623-635`, as the normal `exporter` field).
- `proxima-primitives/src/pipe/primitives.rs` — the `Pipe`/`SendPipe` root
  forms this entire document's "everything downstream is a pipe" claim rests
  on; read its module doc for the four-forms table (transform/source/sink/
  observe) if you want the general vocabulary, not just this feature's use
  of it.
- `proxima-primitives/src/pipe/fan_in.rs` — the actual `FanIn` type section
  3.3 draws the "pipe outside, state inside" analogy from; worth reading if
  you want to see the same structural pattern used for a literal N-source
  merge instead of a 2-arm buffer/replay sink.
- `examples/elevation_walkthrough.rs` and `tests/elevation_e2e.rs` — the two
  runnable worked examples this document quotes from; both are short enough
  to read start to finish in a few minutes and will show you the whole
  feature moving, in order, on your own machine.

Suggested places to link *to* this document (not added by this pass — the
crate's docs directory has no existing "link from" convention to follow, so
this is a recommendation for whoever owns `docs/error-elevation/`): the
top-of-file doc comment on `src/pipes.rs` (currently a one-line comment,
`src/pipes.rs:1-2`), the `Elevation`/`Retention` struct doc comments in
`src/config.rs`, and `docs/error-elevation/discipline.md`'s own header,
which currently only points a reader at itself.
