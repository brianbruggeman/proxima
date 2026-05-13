# proxima by example

proxima is an algebra: everything is a `Pipe`, and big things are small things composed.
So these examples compose too. **Read top to bottom** — each teaches one new primitive or
combinator. When a later example looks big, follow its **Builds on** links back to the
pieces you already know; that decomposition *is* how it's built, in the code and here.

Each example is `examples/<name>/main.rs` with a `examples/<name>/README.md` beside it.

**Config convention:** proxima is config-driven, so any example with real knobs (thresholds, policies,
routes, sinks, TTLs) configures them via `conflaguration` (`Settings` + the layered builder), never
hardcoded — and the `scenarios/` dir is the fully-declarative twin. The atomic algebra demos stay
config-free to keep one concept in focus.

## Start here
- **hello** — a service is a pipe behind a listener. *(builds on: nothing)*

## The pipe algebra — learn these and you can compose anything
- **transform** — write one `Pipe` (`In → Out`), plus its degenerate forms — **source** (`() → Out`), **sink** (`In → ()`), **observe** (`In → In`). One trait, four roles by choosing the types. *(hello)*
- **send** — the same pipe, two flavors: local `Pipe` (no-Send, **borrow-OK**, the default) vs `SendPipe` (`Send + Sync + 'static`). The bounds are a **ladder, all additive**: `Pipe` (borrow, `!Send`) → `+'static` (erase into `DynPipe`/spawn) → `+Send` (cross a core). Climb only as far as your use demands — **zero-copy borrowing pipes are the permissive root**, `'static` and `Send` are costs you opt into. Parallel traits, not sub/super (stable-Rust RPITIT). *(transform)*
- **filter** — pass or drop by a `Decide`. *(transform)*
- **fan-out** — one request out to many (the dual of fan-in). *(transform)*
- **fan-in** — merge many sources into one; pull only the ready. *(transform)*
- **gate** — readiness & backpressure as composition, never a method on the pipe. *(filter, fan-in)*
- **signal** — fully-async fire-once completion: await a condition (end-of-stream, a drain going quiescent) with no polls, waits, or sleeps. observe → filter the terminal condition → fire a `Signal` → await it. The completion sibling of `gate`. *(filter, gate)*

## Configure it — proxima is config-driven
- **config** — typed config with `conflaguration`: the layered **fluent** builder (defaults → file → env → `with_*` overrides), `Validate`, and the serialize↔deserialize **round-trip** (parse a config, get it back byte-faithful). *(transform)*

## Make it resilient — reliability as combinators (the resilience layer, already in main)
- **clock** — time as an injectable seam: schedule against a `Clock`, never `sleep`; deterministic, sleepless. *(transform)*
- **retry** — re-run a pipe on a retryable error. *(filter)*
- **backoff** — retry with delay: constant → exponential → jitter, against the clock. *(retry, clock)*
- **rate-limit** — admit under a rate (`TokenBucket` over the clock). *(gate, clock)*
- **circuit-breaker** — a gate that opens after a failure threshold. *(gate)*
- **deadline** — a timeout as a fired `Signal`. *(signal, clock)*
- **fallback** — try an alternate on failure. *(retry)*

## Flow & delivery — backpressure, cancellation, and the guarantees they compose
- **backpressure** — the strategy space when a producer outruns a consumer: block · drop-newest · drop-oldest · sample · coalesce · batch · pull/demand · credit. *(gate, fan-in)*
- **cancellation** — cooperative (`Signal`) · deadline · drop-to-cancel · propagating · cancel-with-cleanup. *(signal, deadline)*
- **delivery** — at-most-once · at-least-once · exactly-once, *composed* from per-stage strategy choices. *(backpressure, cancellation)*
- **best-effort** — the composite: drop locally so a *presence* guarantee holds globally (tracing's model — the app never stalls, output degrades-but-present). *(delivery)*

## Prove it holds — validate the strategy under adversity
- **chaos** — inject delay/drop/error; assert graceful degradation (`chaos.rs` is itself a pipe). *(the pipe under test)*
- **fuzz** — random + property inputs against the state machines (no-panic, round-trip invariants). *(the codec under test)*
- **differential** — run against a reference oracle (smoltcp for the TCP stack). *(the pipe under test)*

## Fake & replay — front, fake, or replay any dependency
- **record** — capture live traffic. *(transform)*
- **replay** — serve it back, byte-identical. *(record)*
- **cache** — fallthrough + write-back. *(filter, transform)*

## Observe it — o11y, the three pillars, native
*No special machinery: o11y is the algebra aimed at telemetry. logs = fan-out + filter + gate (backpressure); export = sink + fan-out; metrics/traces = observe. The only genuinely new ideas here are the three pillars as data, the unified `instrument`, and cross-boundary propagation — and the lossy-vs-lossless tradeoffs are explicit, not hidden.*
- **logs** — structured logging fanned out to sinks (console, file, file-with-rotate), level-**filtered**, each sink **gated** by an explicit backpressure tradeoff: lossless (block) vs lossy (drop — `OldestEvicted`/`NewestDiscarded`) vs sample. Logging is fan-out + filter + gate; the tradeoff is yours, not hidden in an async appender. *(fan-out, filter, gate)*
- **metrics** — Counter / Gauge / Histogram. *(transform)*
- **traces** — spans + propagation across async boundaries. *(transform)*
- **instrument** — one `#[proxima::instrument]` → all three pillars. *(logs, metrics, traces)*
- **export** — telemetry leaves via **sinks**: console + file + OTLP together (never OTLP-only). The `sink` form aimed at destinations, fanned out. *(sink, fan-out)*
- **distributed-trace** — follow one request across two proxima instances; both spans, ONE trace. *(traces, instrument, export)*

## Runtimes
- **runtime-select** — the same pipe on prime or tokio. *(hello)*
- **multi-runtime** — prime + tokio at once, one process, shared state. *(runtime-select, transform)*

## The frontier — where a tokio stack can't follow
- **no-std** — the sans-IO core on bare metal; config becomes **build-time constants** (`build.rs` codegen), the no-runtime tier of `conflaguration`. *(transform, config)*
- **new-platform** — bring up a new target (os/arch/board): a `PROXIMA_PROFILE` + a `build.rs` that reads it via `conflaguration`/`proxima_build` and emits per-platform `pub const`s — config baked at build, no runtime. The porting workflow. *(config, no-std)*
- **wasm** — proxima at the edge. *(transform)*
- **dpdk** — kernel-bypass **networking**: userspace NIC rx/tx rings, poll-mode. *(runtime-select)*
- **spdk** — kernel-bypass **storage**: userspace NVMe, a sans-IO SQE/CQE codec + ring FSM. *(dpdk)*
- **pmem** — **persistent memory**: byte-addressable, crash-consistent cells, no block layer. *(transform)*

## Extend it
- **plugin** — package a `Pipe` others compose. *(transform)*
- **codec** — teach proxima a new wire protocol. *(transform)*

## Applied — build a real thing
- **proxy** — forward to an upstream. *(transform)*
- **gateway** — proxy + policy (auth · route · rate-limit). *(proxy, gate, filter)*
- **load-balance** — distribute across N healthy backends. *(proxy, fan-in)*
- **crud** — proxima *is* the origin: a small REST service. *(transform, filter)*
- **integration** — front + fake + replay a third-party API. *(proxy, record, replay)*
- **load** — a load generator built on proxima itself: the **`rekt`** crate (its `rek` binary + `rekt_load` example), proxima load-testing proxima. *(transform)*

## Reference — baselines & tools, not lessons
- **floor** — raw-prime baseline (what the abstraction costs).
- **boot-time** — cold-start latency.
- **benches** — throughput/latency harnesses.
- **proxima-main** — `#[proxima::main]` boots the prime or tokio runtime and drives
  the async `main` body to completion; the macro's own proof, not a pipe/listener
  lesson (`proxima_main_demo.rs` / `proxima_main_tokio_demo.rs`).

---
*Status: rungs land as they're built; a rung with no `<name>/main.rs` next to it hasn't
landed yet. Live today: `hello`, the pipe algebra (`transform` through `signal`),
`config`, the resilience layer (`clock` through `fallback`), the full Flow & delivery
unit (`backpressure` · `cancellation` · `delivery` · `best-effort`),
`record`/`replay`/`cache`, the o11y rungs (`logs` through `distributed-trace`), the
prove-it-holds rungs (`chaos`/`fuzz`/`differential`), `runtime-select`, `multi-runtime`,
the frontier `no-std`/`wasm`/`new-platform`/`dpdk` (`dpdk_tcp_connect`/`dpdk_tcp_echo`/
`dpdk_udp_echo`, `required-features = ["dpdk"]`), `codec`, `plugin` (`examples/plugin-skeleton`),
and the applied rungs
(`proxy`/`gateway`/`load-balance`/`crud`/`integration`/`load`). Not yet built:
`spdk`/`pmem`.*
