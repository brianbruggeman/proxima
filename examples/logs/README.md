# logs — logging is fan-out + filter + gate, applied to log events

## Builds on

[fan-out](../fan_out/README.md) — destinations: one log event delivered to every sink, not a special "multi-appender."

[filter](../filter/README.md) — level: a `Decide`-shaped predicate (level >= floor) gates a record before the sink ever runs, exactly like `MinAmount` gates an `Order`.

[gate](../gate/README.md) — backpressure: the bounded queue in front of a sink is the same `DemandGate`-flavored choice `gate`'s SHED/WAIT shapes teach — admit, shed, or park, decided explicitly, not hidden in an async appender.

## What it demonstrates

Logging is not special machinery bolted onto the side of the substrate. It is
the same three primitives the curriculum already covers, applied to one more
payload shape — a `LogRecord` instead of an HTTP request or a job:

- **structured logging** — `proxima_telemetry`'s `error!`/`warn!`/`info!`/
  `debug!`/`trace!` macros carry typed fields (`?x` debug, `%x` display,
  `key = expr` typed scalar, bare-ident shorthand). Every callsite is gated by
  a runtime level filter (`RUST_LOG`, default floor `error`) before it ever
  reaches a recorder — that gate is `filter`'s `Decide`-then-delegate shape,
  just implemented as a cached per-callsite atomic check instead of a `Filter`
  struct.
- **fan-out to sinks** — `proxima_telemetry::pipes::fan_exporters` delivers one
  log event to N sink handles, the primary and secondaries running
  concurrently — the exact "one input, N sinks" shape the `fan-out` example
  teaches over `FanOut<S, Policy>`, specialized to `TelemetryRequest`. Each
  sink additionally wraps a level filter (`LevelGate`, hand-composed the same
  way `gate.rs`'s `Gated<G>` hand-composes a `Decide` adapter — `Rejectable`
  can't be implemented on `TelemetryRequest` from an example crate, orphan
  rules), so the console sink and the file sink can each admit a different
  floor from the SAME fanned event.
- **backpressure is a choice, not hidden machinery** — a bounded queue sits in
  front of a sink: `proxima_telemetry::ring::HeapBoundedQueue`, the exact
  primitive the real per-core log ring is built from
  (`recorder::deliver`/`OverflowPolicy`). The tradeoff is explicit and
  inspectable: `FailMode::DropNewest` / `DropOldest` (lossy — shed under
  overload, counted via `EnqueueOutcome`/`.dropped()`) versus
  `enqueue_assisting` (lossless — the producer becomes a momentary consumer to
  free a slot, the same shape `OverflowPolicy::Block`'s elastic
  producer-assist runs under a full ring). Nothing here is an async appender
  quietly deciding for you; the choice is made in the open, by name.

## Run

```
cargo run --example logs
```

## What you'll see

```
structured logging: real macros + RUST_LOG level discipline
... DEBUG logs: worker picked up job handle=7 peer=10.0.0.7:51422
... INFO logs: batch complete route=checkout jobs_processed=42
... WARN logs: retrying after transient failure err="connection reset by peer" attempt=3
... ERROR logs: job abandoned reason=max_retries_exceeded
drained 4 records (trace! never reached the ring)

fan-out to sinks: one log event, console AND file, filtered per sink
... WARN : latency budget exceeded elapsed_ms=812
fanned 3 log events to 2 sinks
--- file sink (/tmp/.../proxima-logs-fanout.log) ---
... DEBUG : cache warmed entries=4096
... INFO : request served route=checkout
... WARN : latency budget exceeded elapsed_ms=812

backpressure: bounded queue in front of a sink, lossless vs lossy
-- lossy (FailMode::DropNewest): the incoming record is refused --
  enqueue "worker 1 started": Enqueued
  ...
  enqueue "worker 5 started": DroppedNewest
  enqueue "worker 6 started": DroppedNewest
  dropped: 2

-- lossy, other flavor (FailMode::DropOldest): evict the oldest to admit the newest --
  dropped: 2

-- lossless: enqueue_assisting makes room by draining, nothing is dropped --
  delivered, in order: ["worker 1 started", ..., "worker 6 started"]
```

Part 1: the `debug` floor is installed (`EnvFilter::parse("debug")` — the same
grammar `RUST_LOG` uses, applied directly rather than through the env var)
before the first emit. Five macro calls fire
(`trace!` through `error!`); `recorder.drain()` returns exactly 4 — `trace!`
never reached the ring, filtered at the callsite before the recorder ever saw
it. A real regression (the filter passing `trace!`, or dropping `debug!`)
fails the `assert_eq!`, not just the eyeball check.

Part 2: three log events (`DEBUG`, `INFO`, `WARN`) go through one recorder
whose pipe is `fan_exporters([console_gate, file_gate])`. The file sink's
threshold is `DEBUG` (admits all 3); the console sink's threshold is `WARN`
(admits only 1). The file, read back after `drain()`, contains all three
messages exactly once; `LevelGate`'s own atomic counters prove the split
(`file_passed == 3`, `console_passed == 1`, `console_dropped == 2`) — the SAME
fanned event, two independent per-sink decisions.

Part 3: six log lines pushed into a 4-slot `HeapBoundedQueue`. Under
`FailMode::DropNewest`, the last two are refused (`dropped() == 2`) and the
first four are what a dequeuing sink would see. Under `FailMode::DropOldest`,
the same two are dropped, but the *newest* four survive — the sink sees the
most recent traffic, not the oldest. Under `enqueue_assisting` (the lossless
path), every one of the six is eventually delivered, in original order,
`dropped() == 0` — at the cost of the producer doing a `dequeue` itself each
time the queue is full, i.e. throttling to the sink's real speed instead of
losing signal.

## Gap

`Exporter::file(path)` (the sink used above) truncates once at recorder start
and then writes forever. `proxima-telemetry` does now ship a rotating
alternative — `Exporter::file_rotating(path, max_bytes, max_files)`, a
size-triggered roll checked on every write, `path` -> `path.1` -> `path.2` ...
up to `max_files` — but it is size-only (no time/schedule trigger) and this
example doesn't reach for it: the backpressure section already demonstrates
the composable-decorator shape (`LevelGate` wrapping a sink) that rotation
itself is built the same way as, so this example leaves it to `Exporter`'s
own docs rather than duplicating a fourth demonstration of the same shape.
