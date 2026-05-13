# export — telemetry leaves via sinks, and a sink is never alone

## Builds on

[transform](../transform/README.md) (sink form) — `Exporter`/`OtlpHttpPipe`/`FormatterPipe` are all `Pipe<In = TelemetryRecord, Out = ()>`, the same degenerate sink shape `transform.rs`'s `Print` taught, just aimed at a real destination instead of stdout.

[fan-out](../fan_out/README.md) — `proxima_telemetry::pipes::fan_exporters` is `FanOut` specialized to `TelemetryRequest`: one record in, N sinks, the primary and secondaries delivered concurrently.

## What it demonstrates

Export is not bespoke telemetry machinery — it is `transform`'s `sink` form
(`In -> ()`, consumes and produces nothing) fanned to multiple destinations.
`Exporter` composes the sink: `stdout()` / `stderr()` / `std()`
(severity-split) / `file(path)` / `writer(w)` / `pipe(handle)` — the last is
the escape hatch that lets ANY pre-built `Pipe`, including an OTLP encoder,
compose the same way.

The rule this example makes concrete: **never wire OTLP alone.** A collector
outage, a misconfigured endpoint, or a network partition takes the whole
signal down with it if OTLP is the only sink. Console and file are free,
in-process, and have no network dependency — they compose ALONGSIDE OTLP, not
instead of it. `fan_exporters` is what makes that composition free: one
record in, every sink gets it, and one broken exporter doesn't take the
others down (`FanExporter`'s secondaries are best-effort; only the primary's
result propagates). The wiring is one sink built from another: each
destination is its own `Exporter` (`Exporter::stdout()`, `Exporter::file(path)`,
...), those go into `fan_exporters([...])` to produce one fanned handle, and
that handle is handed right back to `Exporter::pipe(fanned)` — the same
`pipe(handle)` escape hatch, just fed the fan-out's own output — so the
recorder installs one `Exporter` whether it's talking to one sink or five.

The example fans a span, two logs, a counter, and a gauge to:

- **console** (`FormatterPipe<Stdout>`) — always free, always in-process.
- **file** (`FormatterPipe<File>`, a tempdir path) — read back and asserted
  after `drain()`, proving every record actually reached a second,
  independent sink.
- **OTLP** (`OtlpHttpPipe`, behind `--features otlp-http`) — real OTLP
  protobuf encoding (`ExportLogsServiceRequest` / `ExportTraceServiceRequest`
  / `ExportMetricsServiceRequest`), flushed and decoded back with `prost` to
  prove the SAME records reached it too. `OtlpHttpPipe`/`OtlpGrpcPipe` are
  encode-only in `proxima-telemetry` — no socket opens on `.call()` — so this
  proves the sink side with no live collector required. The actual network
  POST is a separate, composed stage (`OtlpHttpCodec` -> an HTTP client; see
  `proxima::otlp::OtlpClient` and `tests/otlp_send_prime_e2e.rs`, which sends
  real OTLP over a loopback prime HTTP server).

Metrics export for real here: `recorder.counter(name)` / `.gauge(name)`
register into the RECORDER's own `InstrumentRegistry`, which `recorder.drain()`
snapshots and routes through the exact same fanned pipe as the logs and span
— proven by the file read-back containing `COUNTER value=U64(3)` and `GAUGE
value=U64(7)`, and (with `otlp-http`) by the decoded OTLP metrics batch
containing both points. This is a DIFFERENT registry from the ambient
`counter!`/`gauge!`/`histogram!` macros' static instruments (see
`examples/metrics.rs`): those back onto the global registry stub in
`proxima-telemetry/src/metric/registry.rs` (`// instrument registry — v1
stub; C9 will wire a global recorder here`), which has no recorder and no
sink at all — read only via `.get()`/`.sum()`, never exported. See Gap below.

## Run

```
cargo run --example export
```

With the OTLP arm (adds a third sink, encode-verified, no collector needed):

```
cargo run --example export --features otlp-http
```

## What you'll see

```
export = a sink (Pipe<In = TelemetryRecord, Out = ()>) fanned to 2 destinations
SPAN  checkout: duration_ns=1000
... INFO : request served route=checkout
... WARN : latency budget exceeded elapsed_ms=812
COUNTER value=U64(3)
GAUGE value=U64(7)
drained 5 records (span + 2 logs + counter + gauge) to 2 sinks
--- file sink (/tmp/.../proxima-export-demo.log) ---
SPAN  checkout: duration_ns=1000
... INFO : request served route=checkout
... WARN : latency budget exceeded elapsed_ms=812
COUNTER value=U64(3)
GAUGE value=U64(7)

file sink read back: span, both logs, the counter AND the gauge all landed

the never-otlp-only rule, made concrete: console and file received every record
above with zero network dependency; had OTLP been the ONLY sink and the collector
been unreachable, every one of them would have been lost with it.
```

With `--features otlp-http`, an extra line appears between the file read-back
and the closing rule:

```
otlp sink decoded back: 2 logs, 1 span, 2 metric points (real OTLP protobuf,
no live collector — a network POST would need one)
```

One span (`checkout`), two logs (`INFO`, `WARN`), one counter add, and one
gauge set are emitted through a recorder whose pipe is
`fan_exporters([console, file])` (or `[console, file, otlp]`).
`recorder.drain()` returns 5 — every record reached the fan. The file, read
back from disk, contains all five record lines exactly once, asserted by
name/value, not eyeballed. With the OTLP arm built, the SAME five records
decode back out of real OTLP protobuf bytes: 2 log records, 1 span, 2 metric
points — `assert_eq!` against exact counts, not just "non-empty."

## Gap

Two distinct "metrics" surfaces exist, and only one of them exports:

- **Recorder-scoped** (`recorder.counter(name)`/`.gauge(name)`/`.histogram(name)`,
  used in this example) — a real per-recorder `InstrumentRegistry`, drained
  and exported through the recorder's pipe on every `drain()`/`drain_async()`.
  This works, and this example proves it (file read-back + OTLP decode both
  contain the counter and gauge).
- **Ambient macros** (`counter!`/`gauge!`/`histogram!` over a `static Counter`
  /`Gauge`/`Histogram`, used in `examples/metrics.rs`) — back onto the global
  registry stub in `proxima-telemetry/src/metric/registry.rs` (`// instrument
  registry — v1 stub; C9 will wire a global recorder here`). There is no
  recorder behind it, so there is no sink, no drain, and no export path —
  only `.get()`/`.sum()` local reads. `examples/metrics.rs` never claims
  otherwise; this example calls it out explicitly so the two surfaces aren't
  conflated. Wiring the ambient registry to a recorder is C9's job, not this
  example's.
