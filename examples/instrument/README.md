# instrument — one annotation, three pillars

## Builds on

[logs](../logs/README.md) — the log pillar: `Recorder::log()`, level-tagged, fanned to a pipe.

[metrics](../metrics/README.md) — the metric pillar: a per-name `Histogram`, read back as an aggregate, not a return value.

[traces](../traces/README.md) — the trace pillar: a span observes an operation's scope, and the `#[proxima::telemetry::instrument]` sugar this example uses is the same macro `traces.rs`'s `validate_request` is built with.

## What it demonstrates

`logs`, `metrics`, and `traces` each taught one signal kind in isolation. This
example is the unification: `#[proxima::telemetry::instrument]` on a function
opens a span (trace) around its body, and — when a metric consumer is
subscribed — folds that span's duration into a per-name histogram (metric) on
close. A log emitted from inside the body lands in the identical `Recorder`,
through the identical pipe. Nothing new is wired for the third pillar; there
was never a second telemetry system to wire it into.

One function, one annotation:

```rust
#[proxima::telemetry::instrument(name = "load_user", recorder = rec)]
fn load_user(rec: &Recorder, user_id: u64) -> u64 {
    rec.log().message("loaded user record").tag("user_id", user_id).emit();
    user_id * 2
}
```

`recorder = rec` names the target explicitly — the same explicit-data rule
`traces`' `parent = <expr>` follows. There is no ambient `Recorder::current()`
this expands into; the macro just lowers to `rec.span(name)...start()`, held
for the function body's duration.

**The honest part — the metric pillar is not unconditional.** proxima's own
comment on the mechanism (`proxima-telemetry/src/recorder/mod.rs`, `C2`) says
it plainly: *"the always-on metric is on only when observed."*
`Recorder::span_metrics_live` is a real consumer gate — a span's duration
folds into a histogram only when something has subscribed to read it
(`enable_span_metrics()`, or the ambient `.install()` path, which calls it for
you). This is a *design decision* (skip the histogram lookup and fold
entirely when nothing consumes it — a metric-only recorder pays one relaxed
atomic load, not a full fold), not the dead-code stub the task briefing
expected: `proxima-telemetry/src/metric/registry.rs` ("v1 stub; C9 will wire a
global recorder here") has **zero callers anywhere in the workspace** — it is
orphaned. The instrument that's actually live is
`proxima-telemetry/src/recorder/registry.rs`'s `InstrumentRegistry`, reached
through `Recorder::span_metrics_live` / `enable_span_metrics`, and it is fully
wired — just opt-in, and OFF by default at build time
(`sized::INSTRUMENT_METRICS_DEFAULT`, generated the same way the `conflag`
no_std tier bakes any other default constant) — a fresh `Recorder` starts
with the gate closed until something opens it.

Section 2 proves both sides of the gate on the *identical* annotated
function: same span, same log, metric count 0 with the gate closed, 1 with it
open. Section 1 uses `capture()` (a span-metric consumer by construction —
building the private buffered recorder calls `enable_span_metrics()` as part
of its own setup) so all three pillars show up together on the first call.
Section 3 reads the same call back on stdout, next to an in-memory drain
count, so a human sees what the assertions already proved.

## Run

```
cargo run --example instrument --features instrument-metrics
```

`macros` (the `#[instrument]` attribute itself) is already in the default
feature set. `instrument-metrics` is the default-off firewall this example
exists to demonstrate, so it is not — the `required-features` gate in
`Cargo.toml` means a plain `cargo run --example instrument` with no extra
flags silently skips this example rather than failing to compile.

## What you'll see

```
proxima #[instrument]: one annotation, three pillars

--- 1. one call, one capture: span + log + metric together ---
  trace  (span)      : 1 record(s), name=Some("load_user")
  log                : 1 record(s)
  metric (histogram) : 1 record(s)
  -> ONE #[instrument] annotation produced all three signal kinds from ONE call.

--- 2. the metric pillar is a consumer gate, not unconditional ---
  gate closed : span=1 log=1 metric=0
  gate open   : span=1 log=1 metric=1
  -> trace and log pillars are unconditional; the metric pillar is a real, deliberate
     opt-in (Recorder::enable_span_metrics / .install()), not a hidden bug and not automatic.

--- 3. reading it back on stdout (console + in-memory, same recorder) ---
SPAN instrument load_user: duration_ns=5000
2026-07-08T16:40:51.204006000Z INFO : loaded user record user_id=99
HISTOGRAM count=1
  (above: the console lines a human sees for one #[instrument] call)
  drained 3 telemetry records from the installed recorder

PASS: #[instrument] always yields span + log through the shared recorder;
      the duration metric is real and correct, but opt-in via the C2
      consumer gate (enable_span_metrics / install) -- not automatic.
```

**Verdict: two pillars fire unconditionally from the annotation (trace, and
whatever you log inside it); the third (metric) fires from the same
annotation only once a consumer opens the gate.** All three land in the same
recorder either way — there is no missing plumbing, only a missing
subscriber, and subscribing is one line.

## Gap

The metric pillar is real and correctly implemented, but it is **opt-in**,
not automatic — a plain `Recorder::builder()....start()` with no
`enable_span_metrics()`/`.install()` records the span and any logs, and
silently records ZERO duration metrics, regardless of the `instrument-metrics`
feature being compiled in. This is a legitimate design tradeoff (a
metric-only recorder pays nothing when nobody reads its metrics), but it
means "one annotation, three pillars" is only true once, somewhere in the
process, a consumer has opened the gate — the annotation site itself cannot
tell you whether that has happened. There is no compile-time or run-time
warning today if it hasn't; `#[instrument]` on a hot path with no installed
recorder and no `enable_span_metrics()` call will look like it's working
(span shows up fine) while its metric pillar quietly does nothing.
