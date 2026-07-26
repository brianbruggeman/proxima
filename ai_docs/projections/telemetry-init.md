# telemetry init: turning on `info!`/`#[instrument]` with zero setup

**Supersedes** the unmerged `docs/telemetry-init` branch's
`ai_docs/projections/telemetry-init.md` (branch tip commit `cbac56c9`). That
branch never landed on `origin/main`
(`git merge-base --is-ancestor docs/telemetry-init origin/main` fails, exit
1) and was written against a real bug that has since been fixed on
`origin/main`'s own history. Its warning — **"avoid
`proxima::init_tracing_default`, it is dead code"** — is now **false**: the
bug it documented was fixed in two steps on `origin/main`,
`01e6b207` (removed the broken function entirely) then `ccd755fe` (restored
it as a working, `#[deprecated]` delegation, since proxima is a public crate
and "no callers in this repo" isn't evidence nobody depends on the name),
with `61dff5eb` immediately after making the whole `init_telemetry` family
work with **zero required features** (before that commit, `init_telemetry`
itself still required `--features tracing-init`). This page replaces the old
branch's doc; do not read the old branch's copy.

Everything below is verified against `origin/main` at `a3b4d3a3` (which
contains all of the above, plus the resilient-sink work in
`34df4978` covered by the companion doc
`ai_docs/projections/otlp-resilient-sink.md`), by `file:line` citation or by
actually running the command shown.

## The trap the old doc described (still real) and the fix (new)

`proxima_telemetry::{info!, debug!, warn!, error!, trace!}` and
`#[proxima::instrument]`/`#[proxima::span]` compile and run with **zero
setup**. With nothing installed, every one of those calls is a silent,
successful no-op — this part of the old doc's "mental model" was correct and
still is: a **recorder** is what a call site needs to find in order to go
anywhere (`proxima_telemetry::export::default_recorder`,
`proxima-telemetry/src/export.rs:49-53`), and nothing fills that slot for you
before your code runs.

What changed is what to do about it. The old doc's fix was "call
`Recorder::builder()...install()` yourself, and whatever you do, don't reach
for `init_tracing_default` — it silently discards everything into a
`NullPipe` and never installs itself." That was true at `f74ba40f`. As of
`origin/main` at `a3b4d3a3` (via `61dff5eb`), the crate root ships the
one-liner the old doc said didn't exist and couldn't be trusted:

```rust
proxima::init_telemetry().expect("install console telemetry");

proxima_telemetry::info!("service starting");
```

This is `proxima/src/tracing_init.rs:1-10` (crate doc) and `:77-79`
(the function). Verified by actually running it — `examples/init_telemetry.rs`
under **default features, no `--features` flag**:

```
$ RUST_LOG=info cargo run --example init_telemetry
2026-07-26T16:33:55.467255000Z INFO init_telemetry: service starting
2026-07-26T16:33:55.467257000Z ERROR init_telemetry: this always shows, even with RUST_LOG unset (default floor)
(tracing:: events are NOT bridged without --features tracing-init; proxima_telemetry:: events above still work)
```

(Line order above is process stdout/stderr interleaved differently than the
source order — `error!` goes to stderr, `info!` to stdout, and the "not
bridged" `println!` is unrelated stdout emitted after both. Every line did
print, from a plain `cargo run` with no feature flags.)

## The distinction that makes the zero-feature promise true

There are **two separable things** "telemetry init" could mean, and only one
of them needs an extra dependency:

1. **proxima's own telemetry** — the `proxima_telemetry::*` macros,
   `#[proxima::instrument]`. These need a `Recorder` + a sink + something
   draining it. That is all native code in `proxima-telemetry`; it does not
   touch the `tracing` crate or `tracing-subscriber` at all.
2. **bridging the `tracing` crate itself** — so a third-party dependency's own
   `tracing::info!()`/`tracing::span!()` calls (hyper, rustls, tokio, ...)
   also land in your recorder. This genuinely needs `tracing-subscriber`
   (`TracingLayer`, `proxima-telemetry/src/tracing_bridge.rs`), which is why
   it is gated behind the `tracing-init` feature.

`init_telemetry()`/`init_telemetry_with(format)` only ever need (1), so they
work with **zero required features** — confirmed by the crate doc comment
(`tracing_init.rs:1-19`) and by the `#[cfg]` split in the source: without
`tracing-init` they delegate to
`proxima_telemetry::export::install_console_recorder_with`
(`tracing_init.rs:27-28`, `:97-100`); with it on, to
`install_console_logging_with` (`tracing_init.rs:35-38`, `:86-89`), which adds
step (2) on top. Either branch is a straight delegation — `tracing_init.rs`
carries no parallel reimplementation of either (see the doc comment at
`export.rs:297-305`).

## The four names, what each does, and which is deprecated

| name | features required | what it does | deprecated? |
|---|---|---|---|
| `proxima::init_telemetry()` | **none** | console output (level-routed: trace/debug/info→stdout, warn/error→stderr), `RUST_LOG` honored, ambient recorder installed, drain thread running | no — this is the name to reach for |
| `proxima::init_telemetry_with(LogFormat)` | none | same, with an explicit format (`LogFormat::Json` for structured console output) | no |
| `proxima::init_tracing_default(LogFormat)` | none | thin delegation to `init_telemetry_with` — kept for source compatibility with code written against the old name | **yes** — `#[deprecated(note = "renamed: use proxima::init_telemetry_with ...")]`, `tracing_init.rs:105` |
| `proxima::init_tracing(recorder, format)` | **`tracing-init`** (genuinely) | bridges `tracing::` events into a recorder **the caller already built and owns** | no — this is the one case `init_telemetry` structurally cannot express |

Citations: `init_telemetry`/`init_telemetry_with` at `tracing_init.rs:77-100`;
`init_tracing_default` at `tracing_init.rs:102-108`; `init_tracing` at
`tracing_init.rs:110-159` (the `tracing-init` branch) and `:153-159` (the
loud-failure branch when the feature is off).

Why `init_tracing` is not deprecated even though its name looks like the
"old" pair: `init_telemetry`/`init_telemetry_with` always build their **own**
recorder. `init_tracing(recorder, format)` takes one **the caller already
built** (with whatever sinks, capacity, `core_count`, etc. it wants) and
bridges `tracing::` events into that specific instance. There is no way to
express "bridge into a recorder I already have" through the zero-argument
one-liner — so the two names serve genuinely different callers, not an
old/new pair (doc comment, `tracing_init.rs:110-129`). Its `format` parameter
is accepted but has **no effect** — the recorder's formatting was already
fixed by the caller when they built it, before this function ever sees it;
it is kept only for signature compatibility (`tracing_init.rs:120-123`,
explicitly documented, not silently dropped).

Regression coverage for all four names — including the specific "must not
regress back to the `NullPipe`/no-`set_default_recorder` trap" contract —
lives in `tests/console_logging_regression.rs`, run here for real:

```
$ cargo test --test console_logging_regression                    # default features
running 4 tests ... test result: ok. 4 passed; 0 failed
$ cargo test --test console_logging_regression --features tracing-init
running 7 tests ... test result: ok. 7 passed; 0 failed
```

## `install_console_recorder{,_with}` vs `install_console_logging{,_with}`

Both pairs live in `proxima-telemetry/src/export.rs` and are what
`init_telemetry`/`init_telemetry_with` delegate to. Reach for them directly
(instead of the `proxima::` crate-root names) when you're already inside
`proxima_telemetry` rather than the `proxima` facade — e.g. writing an
example or test for the telemetry crate itself, as every `examples/*.rs` in
this repo does.

- **`install_console_recorder()` / `_with(Formatter)`**
  (`export.rs:297-340`) — the always-available, zero-extra-feature half:
  builds a recorder over `Exporter::std()` (level-routed console), installs
  it as the process default, spawns the background drain thread. This is what
  `init_telemetry` calls when `tracing-init` is off.
- **`install_console_logging()` / `_with(Formatter)`**
  (`export.rs:342-383`, `#[cfg(feature = "tracing-init")]`) —
  `install_console_recorder_with` **plus** the `tracing::`-crate bridge
  (`TracingLayer`, filtered by the same `RUST_LOG`-driven floor, default
  `"warn,proxima=info"` when unset — `export.rs:376-377`). What
  `init_telemetry` calls when `tracing-init` is on.

Use `install_console_recorder{,_with}` when you only care about proxima's own
macros and want to keep `tracing-subscriber` out of your dependency tree.
Reach for `install_console_logging{,_with}` (or just `init_telemetry` with
`tracing-init` enabled) the moment a dependency's own `tracing::` calls need
to show up too.

## Copy-paste patterns (each run, not just read)

### (a) a binary

Verbatim `examples/init_telemetry.rs` (see its full source and the
transcript above) — `proxima::init_telemetry()` is the entire setup.

### (b) a test — capture, no ambient state touched

Verbatim doctest, `proxima-telemetry/src/capture.rs:7-27` (compiled and run
by `cargo test --doc -p proxima-telemetry`):

```rust
use proxima_telemetry::capture::capture;

let tel = capture(|rec| {
    rec.span("load_user").tag("id", 42u64).start();
    rec.log().message("cache miss").emit();
    rec.counter("db.queries").add(1, &[]);
});
assert_eq!(tel.spans().len(), 1);
assert_eq!(tel.spans()[0].name, "load_user");
assert_eq!(tel.logs().len(), 1);
assert!(tel.metrics().len() >= 1);
println!("{}", tel.dump()); // when an assertion fails, see everything emitted
```

`capture()` (`capture.rs:123-136`) builds a private, in-memory recorder,
never touches `default_recorder()`, drains it for you, and hands back typed
records — this is the right tool for a unit test's assertions, not
`init_telemetry` (which installs an ambient, process-wide recorder you'd then
have to avoid cross-contaminating between tests).

### (c) a library crate's own example — explicit console + file fan-out

`examples/logs/main.rs` (`cargo run --example logs`) is the canonical
worked example: real `RUST_LOG`-gated macros, fan-out to console AND a file
via `proxima_telemetry::pipes::fan_exporters`, and the explicit lossless-vs-
lossy backpressure choice. Real transcript (this repo, this commit):

```
$ cargo run --example logs
...
fan-out to sinks: one log event, console AND file, filtered per sink
2026-07-26T16:34:46.363222000Z WARN : latency budget exceeded elapsed_ms=812
fanned 3 log events to 2 sinks
--- file sink (/.../proxima-logs-fanout.log) ---
2026-07-26T16:34:46.363220000Z DEBUG : cache warmed entries=4096
2026-07-26T16:34:46.363222000Z INFO : request served route=checkout
2026-07-26T16:34:46.363222000Z WARN : latency budget exceeded elapsed_ms=812
```

This is the pattern to copy when you want more than one sink — it composes
`Exporter`/`FormatterPipe`/`fan_exporters` directly rather than going through
`init_telemetry` (which only ever builds one console sink).

## Decision table

| Where you are | Does telemetry auto-init? | What to write |
|---|---|---|
| `#[proxima::main]` / a bare `fn main` | **No.** Neither the macro nor a plain `main` installs anything for you. | `proxima::init_telemetry()` as the first line — snippet (a). |
| `#[proxima::test]` | **No.** | `capture()` for assertions — snippet (b). Install a console recorder inside the test only if you need to watch it live. |
| A library crate's own example/binary | **No.** | `install_console_recorder_with`/`install_console_logging_with` directly, or compose your own sinks — snippet (c). |
| A bench | **No**, and usually you don't want one — keep a bench's hot path isolated from console I/O. | Build a private `Recorder` scoped to the bench; don't install it as ambient. |
| A library you don't own the `main` of | **No**, and it's not your job to install one. | Just call the macros / add `#[instrument]`; the binary that links you in owns the install decision. |

## Troubleshooting

- **Nothing at all, ever.** No recorder installed. Add `init_telemetry()` (or
  one of the `install_console_*` calls) before the first emit.
- **Only `error!` lines show, everything else is silent.** Correct, not
  broken: the runtime emit floor defaults to `error`-only with no `RUST_LOG`
  set (`proxima-telemetry/src/emit/global.rs:6-9`, backed by
  `EnvFilter::from_default_env()`). Raise it with `RUST_LOG=debug`, or in
  code via `proxima_telemetry::emit::global::install(EnvFilter::parse("debug"))`
  before the first emit (see `examples/logs/main.rs:65-72` for exactly this
  pattern, with the rationale for installing explicitly rather than relying
  on the shell's env).
- **Logs and spans show up, but my `Counter`/`Gauge` never do.**
  `Counter::add` (`proxima-telemetry/src/metric/counter.rs:44`) only ever
  touches its own `AtomicU64` — the plain `counter!(INSTRUMENT, delta)` form
  never reaches any recorder, installed or not. This is still true and is
  unrelated to the fixes above: to get a metric exported, either read
  `.get()`/`.snapshot_and_reset()` yourself, or pass
  `counter!(INSTRUMENT, delta, recorder = rec)`
  (`proxima-telemetry/src/metric/mod.rs:40-44`) to mirror the delta into that
  recorder explicitly.
- **`init_tracing(recorder, format)` returns an error immediately.** Without
  `tracing-init`, this function's whole purpose (the `tracing::`-crate
  bridge) doesn't exist, so it fails loudly rather than silently succeeding —
  by design (`tracing_init.rs:153-159`, covered by
  `init_tracing_fails_loudly_without_tracing_init` in
  `tests/console_logging_regression.rs`). Enable `tracing-init` if you need
  it, or use `init_telemetry` if you only need proxima's own macros.

## What this page does NOT cover

The `proxima-log` skill (`~/.claude/skills/proxima-log/SKILL.md`) is the
reference for call-site syntax once a recorder exists: the log-macro field
grammar, metric instrument macros, `#[instrument]`/`#[span]`'s three-pillar
expansion, the `instrument-metrics` consumer gate, and the `RUST_LOG`
grammar. Read it after this page, not instead of it.

**Found, not fixed:** the `proxima-log` skill cites
`proxima/docs/unified-instrument/discipline.md` as its design/bench-seal
reference for `#[instrument]`. That path still does not exist anywhere in
this repository at `a3b4d3a3` (`find . -iname 'unified-instrument*'` and
`find . -iname discipline.md`, both run from the repo root, return nothing).
This was already
true when the superseded branch reported it and remains true now — the
skill's pointer is stale. Not fixed here; reported for the owner to triage.

## How this page was verified

Every command below was actually run against this repository at `origin/main`
`a3b4d3a3` (whose linear history includes `01e6b207`, `ccd755fe`, `61dff5eb`,
and `34df4978`, the telemetry commits this page and its companion
`otlp-resilient-sink.md` describe):

- `RUST_LOG=info cargo run --example init_telemetry` (default features, no
  `--features` flag) — output pasted above.
- `cargo run --example logs` — output pasted above.
- `cargo test --test console_logging_regression` (default features): 4
  passed.
- `cargo test --test console_logging_regression --features tracing-init`: 7
  passed.
- `cargo test --doc -p proxima-telemetry` (default features): 3 passed,
  including the `capture()` doctest quoted above.
- Every `file:line` citation above was read back from source at this commit.
