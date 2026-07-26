# the resilient OTLP sink: what happens when your collector goes down

Audience: an **operator** running a proxima service that exports telemetry to
an OTLP collector — not a library author. This is a new capability
(`proxima_telemetry::out::resilient`, added in `34df4978` on `origin/main`'s
history, with its config validation + layered loader completed in the tip
commit `a3b4d3a3` — the same commit range as the zero-feature init fix
covered by the companion doc) that shipped with no user-facing docs before
this page. Everything below is verified against source and its own test
suite (`proxima-telemetry/src/out/resilient/{mod,config,queue,worker,tests}.rs`)
by `file:line` citation.

## What problem this solves

Without this sink, an OTLP exporter is a plain `Pipe`: one send, one
await. If your collector is down, slow, or unreachable, that `await` blocks
whatever calls it — and in proxima's design, that caller is the **shared
telemetry drain**, the same thread every other sink (console, file) depends
on to flush. A wedged collector would silently freeze all of your telemetry,
not just the OTLP leg.

`ResilientSink` is a drop-in terminal pipe that owns its **own** buffer and a
**dedicated background thread**. `Pipe::call` only enqueues (a short mutex
section, no I/O) and returns immediately
(`proxima-telemetry/src/out/resilient/mod.rs:270-278`); the network send,
retries, backoff, and reconnects all happen off to the side. This is proven
directly by a test that asserts the enqueue call **resolves on the very
first `poll`, never `Pending`**
(`call_resolves_ready_on_first_poll_never_pending`, `tests.rs:227-250`) — the
exact property that guarantees the shared drain can never block on this
sink, no matter what the collector is doing.

## The two guarantees, stated precisely

**Guarantees: liveness.**

- Reconnects forever. There is **no terminal give-up state** — a batch that
  keeps failing to send is retried indefinitely, proven directly by a test
  that runs 200 retry iterations against an always-refusing collector and
  asserts nothing was ever dropped as a result
  (`retries_never_reach_a_terminal_give_up_state`, `tests.rs:315-339`).
- Backoff is exponential with jitter, capped at `backoff_cap_ms` (default
  **30 seconds**, `config.rs:38,207-210`) — so once a collector recovers, it
  is noticed within about one interval, never left waiting behind an
  ever-growing delay (`backoff_grows_is_jittered_and_caps_at_configured_ceiling`,
  `tests.rs:254-311`).
- Any transport error (not just an HTTP error status) tears down and rebuilds
  the downstream connection via the `factory` closure you supplied, before
  the next attempt — so a dead-but-still-"connected" channel (the classic
  gRPC/H1 keepalive-looks-fine-but-every-send-fails case) is never retried
  against itself forever (`worker.rs:116-161`,
  `transport_error_triggers_reconnect_before_the_next_attempt`,
  `tests.rs:930-978`).
- The background worker is **supervised**: its whole per-iteration body runs
  inside `catch_unwind`. A panic is counted and logged, never lets the thread
  exit — an unsupervised panic here would otherwise silently and permanently
  kill export for the rest of the process's life
  (`worker.rs:1-10,296-327`, proven by
  `worker_iteration_panic_is_caught_and_the_loop_can_continue`,
  `tests.rs:864-926`).
- End-to-end proof under a real outage: a test hammers the sink with 200
  records while a fake collector refuses every attempt, asserts producing
  those 200 records stays under 500 ms (never blocks on the outage), then
  confirms all 200 arrive once the collector recovers
  (`outage_then_recovery_flushes_backlog_and_floor_sink_never_stalls`,
  `tests.rs:792-860`).

**Does NOT guarantee: losslessness.**

- The buffer is bounded (`buffer_capacity`, default **65,536** records,
  `config.rs:46,180-183`). Space is finite; a long enough outage, or a high
  enough ingest rate, **will** drop records. This is an accepted, announced
  outcome, not a bug — see the shedding strategy below.
- At-least-once delivery, not exactly-once: a retry after a send that
  actually landed but whose response was lost or timed out can duplicate.
  No deduplication is performed (`mod.rs:123-124`).
- `shutdown()` signals the worker to stop after its current iteration; it
  does **not** wait for the buffer to drain to zero first. There is no
  "flush and stop" guarantee — call it at planned process shutdown, not
  mid-outage, if you need to be sure a specific batch got out first
  (`mod.rs:240-247`).

## The graceful-decline strategy

This is the heart of the design: when the buffer fills faster than it can be
drained (collector down, or slow, or your service is emitting faster than
the network can absorb), the sink sheds the **least severe, oldest** records
first, and it protects errors longest.

Every record is bucketed into one of five severity lanes and given a
**per-severity retention horizon** — the maximum age it is allowed to reach
before it is evicted regardless of remaining space:

| severity | default horizon | config field | env var |
|---|---|---|---|
| trace | 10 min (600s) | `horizons.trace_secs` | `PROXIMA_OTLP_RESILIENT_HORIZONS_TRACE_SECS` |
| debug | 20 min (1200s) | `horizons.debug_secs` | `PROXIMA_OTLP_RESILIENT_HORIZONS_DEBUG_SECS` |
| info | 30 min (1800s) | `horizons.info_secs` | `PROXIMA_OTLP_RESILIENT_HORIZONS_INFO_SECS` |
| warn | 35 min (2100s) | `horizons.warn_secs` | `PROXIMA_OTLP_RESILIENT_HORIZONS_WARN_SECS` |
| error | 40 min (2400s) | `horizons.error_secs` | `PROXIMA_OTLP_RESILIENT_HORIZONS_ERROR_SECS` |

Defaults: `proxima-telemetry/src/out/resilient/config.rs:29-33` (the
`DEFAULT_*_HORIZON_SECS` constants) and `:63-83` (the struct + its `#[setting(default = ...)]`
attributes). A log's own level picks its lane directly; a span's **terminal
status** picks its lane — an errored span is treated as valuable as an error
log (`error` lane), a healthy span defaults to `info`
(`queue.rs:56-72`, `span_severity_bucket_follows_terminal_status_not_a_fixed_default`,
`tests.rs:210-223`). Events/metrics/links, which carry no severity of their
own, default to the `info` lane.

**Two independent eviction paths, whichever binds first:**

1. **Age past horizon.** Swept periodically (amortized, not per-record): any
   record older than its own lane's horizon is evicted, oldest lanes first —
   so at t=11s with the ladder `[10, 20, 30, 35, 40]`s only the trace record
   is gone; by t=41s everything is
   (`sweep_horizons`, `queue.rs:230-253`;
   `horizon_sweep_evicts_in_the_configured_per_severity_ladder`,
   `tests.rs:535-579`, which drives exactly this timeline and asserts each
   step).
2. **Space pressure.** If the buffer is at capacity, the **shortest-horizon
   nonempty lane** is evicted first (trace before debug before ... before
   error) — space pressure can never touch the error lane while any less
   severe lane still holds anything
   (`push`, `queue.rs:201-228`;
   `space_pressure_evicts_by_severity_before_touching_higher_lanes`,
   `tests.rs:497-531`, which fills 2 trace + 2 debug + 2 error records into a
   5-slot buffer, at the same instant so no aging is possible, and confirms
   exactly one trace record is evicted, nothing else).

**Frame it the way you actually think about an outage:** "how long can my
collector be down before I start losing signal, and what goes first?" The
answer, read straight off the table above with default config: roughly 10
minutes of trace-level noise is gone before anything else is touched; you
have a 40-minute margin before you lose a single error. Tune the ladder to
match how long a maintenance window you actually need to survive at each
severity (`config.rs:25-28`, the design rationale comment).

**The ladder is enforced, not just documented.** `RetentionHorizons`
implements `Validate`: every horizon must be non-zero, and the ladder must be
**non-decreasing** (`trace_secs <= debug_secs <= info_secs <= warn_secs <=
error_secs`) — an inverted table (e.g. a shorter error horizon than debug)
would silently defeat the entire design, so it is rejected at config-load
time with each offending field named
(`config.rs:114-152`;
`inverted_ladder_is_rejected_and_names_the_fields`, `tests.rs:982-1009`,
which sets `debug_secs=2400` above the default `info_secs=1800` and
`error_secs=600` below the default `warn_secs=2100` in one edit and asserts
**both** inversions are reported, not just the first).

## The survivable-window figures

Per severity, the sink computes and self-exports a **survivable seconds**
gauge: `min(configured_horizon, available_capacity / current_ingest_rate)`,
where "available capacity" assumes every strictly-more-severe lane keeps
growing (they always win eviction over you), so it is the pessimistic,
worst-case figure, not an average
(`survivable_seconds`, `queue.rs:380-399`;
`survivable_seconds_shrinks_as_higher_lanes_occupy_more_capacity`/
`survivable_seconds_shrinks_as_ingest_rate_rises`, `tests.rs:635-676`).

**Gauge names** (`worker.rs:32-38`, one per severity):

```
proxima.otlp.survivable_seconds.trace
proxima.otlp.survivable_seconds.debug
proxima.otlp.survivable_seconds.info
proxima.otlp.survivable_seconds.warn
proxima.otlp.survivable_seconds.error
```

**How to read them:** each gauge answers "if the collector went down right
now and ingest kept up at the recent rate, how many seconds until this
severity starts losing records?" A number pinned at the configured horizon
means capacity isn't the binding constraint yet (you'd hit the age horizon
first); a number well below the horizon means a busy neighbor lane (or your
own ingest rate) is the thing actually limiting your margin — that is your
signal to either raise `buffer_capacity`, shorten a less-important lane's
horizon to free headroom, or reduce ingest rate at that severity. These
gauges are recomputed and re-exported every `drop_announce_interval_ms`
(default 5s, see below) — they are a live dial, not a one-time estimate
(`export_self_metrics`, `worker.rs:198-249`).

## How you know you're losing data

Two independent surfaces, neither of which is "one log line per drop" —
that would itself become a second outage under a real drop storm:

**1. An aggregated, rate-limited announcement.** At most once per
`drop_announce_interval_ms` (default **5000ms**, `config.rs:47,214-217`), if
anything was dropped since the last tick, one `WARN` log line summarizes the
whole window:

```rust
recorder
    .log()
    .level(Level::WARN)
    .message("otlp sink shedding telemetry under buffer pressure")
    .tag("dropped_trace", drop_deltas[0])
    .tag("dropped_debug", drop_deltas[1])
    .tag("dropped_info", drop_deltas[2])
    .tag("dropped_warn", drop_deltas[3])
    .tag("dropped_error", drop_deltas[4])
    .tag("backlog", shared.queue.total_len() as u64)
    .tag("oldest_ms", oldest_ms)
    .tag("window_secs", elapsed.as_secs())
    .emit();
```

(`announce_drops`, `worker.rs:251-278`, quoted verbatim). Proven aggregated
under load: a test drives a 500-record drop storm into a 4-slot buffer and
asserts the number of announcement lines is far fewer than the number of
records actually dropped
(`drop_announcements_are_aggregated_across_a_drop_storm`, `tests.rs:687-743`
— in that run, hundreds of drops produce a small handful of log lines).

**2. Self-exported counters/gauges.** Every tick, alongside the survivable
gauges above, the sink exports (`worker.rs:25-43,198-249`):

| name | kind | meaning |
|---|---|---|
| `proxima.otlp.dropped.trace` / `.debug` / `.info` / `.warn` / `.error` | counter | records evicted from that lane since start (space **or** age) |
| `proxima.otlp.retried_total` | counter | send attempts that did not succeed on the first try |
| `proxima.otlp.reconnected_total` | counter | times the downstream transport was torn down and rebuilt |
| `proxima.otlp.sent_total` | counter | records successfully handed to the collector |
| `proxima.otlp.worker_panics_total` | counter | background-worker iterations that panicked and were recovered |
| `proxima.otlp.backlog_depth` | gauge | records currently buffered, across all severities |

Proven landing on a floor sink (not just incrementing in memory):
`self_exported_counters_and_gauges_reach_the_floor_sink`, `tests.rs:747-788`.

**Operator gotcha: these only export if a recorder is reachable.** Both the
drop announcement and the self-metrics export bail out silently
(`let Some(recorder) = shared.self_recorder() else { return; }`,
`worker.rs:204-206,261-263`) if there is no recorder to report through. That
recorder is either one you passed explicitly via `spawn_with_recorder`
(below), or, failing that, whatever is currently the **process-default
ambient recorder** (`mod.rs:61-68`, `crate::export::default_recorder`). If
you build the sink with plain `spawn()` and never install an ambient
recorder (e.g. via `proxima::init_telemetry()` — see
`ai_docs/projections/telemetry-init.md`) before or after, you get **zero**
visibility into drops/retries/reconnects — the sink still runs correctly,
you just can't see any of the surfaces above. Always either install an
ambient recorder or use `spawn_with_recorder`.

**Also available without any recorder at all:** `sink.stats()` returns a
plain `Stats` struct (`sent`, `retried`, `reconnected`, `panics_recovered`,
`backlog_depth`, `dropped_by_bucket: [u64; 5]`) read directly off the sink's
own atomics — the test/health-check surface that doesn't depend on telemetry
being wired up at all (`mod.rs:216-238,250-260`).

## Building one

Illustrative (names a real collector transport but is not itself a compiling
standalone snippet — the "how it composes" doctest is in the next section):

```rust,ignore
use proxima_telemetry::out::resilient::{ResilientOtlpConfig, ResilientSink};
use proxima_telemetry::pipes::{OtlpHttpCodec, into_telemetry_handle};

// `factory` builds (or rebuilds, on reconnect) the downstream transport.
// `OtlpHttpCodec::new(downstream)` (feature `otlp-http`, `pipes.rs:850-865`)
// is the real collector-facing shape: `downstream` is an HTTP client pipe
// (or a retry/TLS wrapper around one) that already knows the collector's URL.
let sink = ResilientSink::spawn(
    move || into_telemetry_handle(OtlpHttpCodec::new(downstream_client_handle.clone())),
    ResilientOtlpConfig::default(),
);
```

- `ResilientSink::spawn(factory, config)` — production entry point, self-
  instruments through the ambient recorder if one is installed
  (`mod.rs:134-139`).
- `ResilientSink::spawn_with_recorder(factory, config, recorder)` — same, but
  self-instruments through an explicit `Arc<Recorder>` instead of the
  ambient default — the seam to use if this sink lives inside a non-default
  recorder, or you simply don't want to depend on install order
  (`mod.rs:146-152`).
- `ResilientSink::spawn_with_clock(factory, config, clock)` — the
  deterministic-test seam (an injected `Clock` makes backoff/horizon timing
  assertable without a real sleep); this is how every test above drives the
  sink (`mod.rs:159-171`).
- `sink.shutdown()` — stop the background worker after its current
  iteration; see the no-flush-guarantee note above.

`ResilientSink<Clk>` implements `SendPipe<In = TelemetryRequest, Out =
Response<Bytes>>` (`mod.rs:262-279`), so `into_telemetry_handle(sink)` turns
it into a `TelemetryPipeHandle` like any other terminal — it composes into
`fan_exporters` exactly like a console or file sink.

## The house rule: never OTLP-only

Export must never depend solely on a collector you might not be able to
reach. Pair the resilient sink with a local floor sink via
`fan_exporters` — this is a real, compiling doctest
(`proxima-telemetry/src/out/resilient/mod.rs`, run by
`cargo test --doc -p proxima-telemetry --features otlp-http`):

```rust
use proxima_telemetry::export::Exporter;
use proxima_telemetry::level::Level;
use proxima_telemetry::out::resilient::{ResilientOtlpConfig, ResilientSink};
use proxima_telemetry::pipes::{
    FormatterPipe, InMemoryPipe, LogFormat, fan_exporters, into_telemetry_handle,
};
use proxima_telemetry::recorder::Recorder;

// stand-in for a real OTLP transport factory -- any SendPipe factory works,
// the sink is transport-agnostic.
let collector = InMemoryPipe::new();
let factory_collector = collector.clone();
let resilient = into_telemetry_handle(ResilientSink::spawn(
    move || into_telemetry_handle(factory_collector.clone()),
    ResilientOtlpConfig::default(),
));
let console = into_telemetry_handle(FormatterPipe::new(std::io::stderr(), LogFormat::Human));
let fanned = fan_exporters(vec![console, resilient]);

let recorder = Recorder::builder()
    .export(Exporter::pipe(fanned))
    .expect("compose console + resilient fan")
    .core_count(1)
    .start()
    .expect("start recorder");

recorder.log().level(Level::ERROR).message("hello").emit();
recorder.drain(); // enqueues into the resilient sink's own buffer; the
                   // background worker (not this call) does the send.
```

Console output keeps working even if the OTLP leg is entirely down — the two
sinks are independent (`fan_exporters`, `proxima-telemetry/src/pipes.rs:426`
onward).

## Config: fluent builder and layered loader

`ResilientOtlpConfig` (`config.rs:170-233`) has every knob referenced above
(`buffer_capacity`, `max_batch_bytes`, `max_batch_items`, `backoff_base_ms`,
`backoff_cap_ms`, `drop_announce_interval_ms`, `idle_poll_ms`, and the nested
`horizons: RetentionHorizons`), plus `RetentionHorizons` itself. Both derive
`Builder` (fluent), `Deserialize`/`Serialize` + `Settings` (conflaguration
env/file), and `Validate`.

**Fluent:**

```rust
let config = ResilientOtlpConfig::builder()
    .buffer_capacity(4096)
    .backoff_cap_ms(5_000)
    .build();
```

**Layered loader** — `.layered()` starts from the compiled-in defaults; each
source (`.with_*`, `.from_path`, `.from_env`, `.underlay_path`,
`.underlay_env`) contributes only the fields it actually sets, so an
untouched field always falls through rather than getting re-defaulted over a
prior layer (`config.rs:343-357,381-582`):

```rust
let config = ResilientOtlpConfig::layered()
    .with_buffer_capacity(4096)      // code override
    .from_path("resilient.toml")?    // file wins for fields it sets
    .from_env()?                     // env wins for fields it sets
    .build();
config.validate()?; // NOT run automatically -- caller decides hard-fail vs fallback
```

`.build()` deliberately does **not** validate — call `.validate()` yourself,
same contract as the rest of the crate's layered configs
(`config.rs:640-647`). This composition order (`.with_*` then `.from_path`
then `.from_env`) is exactly what
`layered_underlay_path_never_clobbers_an_already_set_field`/
`layered_from_env_overlays_scalars_and_the_nested_horizons_table`
(`tests.rs:1167-1219`) assert.

**TOML example** (this exact shape is what
`builder_serde_and_layered_loader_agree`, `tests.rs:1223-1259`, round-trips
against the builder and asserts equal):

```toml
buffer_capacity = 2048

[horizons]
trace_secs = 60
debug_secs = 120
info_secs = 180
warn_secs = 240
error_secs = 300
```

Loaded with `ResilientOtlpConfig::layered().from_path("resilient.toml")?.build()`.
Setting `[horizons]` is whole-table — a file that touches the ladder at all
must specify all five fields (`config.rs:360-368`).

**Env vars** use the `PROXIMA_OTLP_RESILIENT_*` prefix for the flat fields
and `PROXIMA_OTLP_RESILIENT_HORIZONS_*` for the ladder (table above); e.g.
`PROXIMA_OTLP_RESILIENT_BUFFER_CAPACITY=777` and
`PROXIMA_OTLP_RESILIENT_HORIZONS_ERROR_SECS=9999`
(`layered_from_env_overlays_scalars_and_the_nested_horizons_table`,
`tests.rs:1188-1207`).

**An inverted or zeroed ladder is rejected, not silently accepted** — see
"the ladder is enforced" above. Both the flat config and the nested horizons
report every problem at once, not one `Err` per field
(`resilient_config_reports_all_problems_at_once`, `tests.rs:1067-1100`).

## Other tunables worth knowing about

- `max_batch_items` (default 512) / `max_batch_bytes` (default ~3 MiB,
  sized to leave margin under the OTel Collector's default 4 MiB
  `otlpreceiver` message-size ceiling, `config.rs:41-44,187-197`) — a backlog
  larger than either limit is chunked into multiple sends on flush, always
  highest-severity-nonempty-lane first
  (`pull_batch`, `queue.rs:255-289`;
  `backlog_larger_than_one_batch_flushes_as_multiple_correctly_sized_chunks`/
  `backlog_flush_respects_the_byte_ceiling_not_just_item_count`,
  `tests.rs:343-439`).
- `idle_poll_ms` (default 1000ms) — the safety-net interval the worker falls
  back to when nothing wakes it via `notify`; bounds staleness of the
  horizon sweep and the metrics tick (`config.rs:219-224`).
- `backoff_base_ms` (default 200ms) — the first retry delay, before jitter
  and before it grows toward `backoff_cap_ms`.

## Where this lives / what's still not covered

- Module: `proxima-telemetry/src/out/resilient/{mod,config,queue,worker}.rs`
  behind the `otlp-http` feature (also covers the `otlp-grpc` transport — the
  sink is transport-agnostic, `mod.rs:10-12`).
- Not covered here: the OTLP wire encoding itself (`out::otlp_http`,
  `out::otlp_grpc`) — this page is about the resilience layer sitting in
  front of either transport, not the codec.
- FEATURES.md had no mention of this capability before this page; see the
  report accompanying this doc for the exact gap and the entry added there.
