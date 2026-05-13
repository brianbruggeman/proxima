# proxima-notify substrate gaps (S1, S2, S3)

This document captures three architecture gaps in proxima that the proxima-notify initiative absorbs as substrate components (S1, S2, S3). The user explicitly chose to fold the substrate work into this initiative rather than split it into a separate workspace-level effort.

The findings below were confirmed by reading source at the cited file:line references before this initiative branched from `main`.

## S1 — `BinSink` is not a real crash-safe WAL

**Symptom.** The earlier plan claimed `BinSink` was a "WAL with fsync". That's wrong.

**Evidence.**
```
$ rg sync_all|sync_data|fsync proxima-recording-core/src/binary/sink.rs
(zero hits)
```

`BinSink::append` allocates an encoded frame, calls `tokio::fs::File::write_all`, and `BinSink::flush` calls `tokio::fs::File::flush` (`sink.rs:74,113`). `flush` is buffer-level only — it does NOT call `sync_all` or `sync_data`. A crash between the buffer flush and the OS page cache write loses data. Replay on restart sees a possibly-incomplete file.

**Consequence.** C7 (OffsetCursor) and C8 (DurableConsumerPipe) cannot land on this substrate as designed. The "at-least-once" claim requires the substrate to actually persist before acknowledging.

**S1 fix.**
- Add `BinSink::sync_now() -> Result<()>` that calls `sync_all()` on the data file AND the `.idx` index file, in the correct order (data first per the algorithm-development paper proof below).
- Add `BinSink::set_sync_mode(SyncMode)` with `Always | Batch(N) | Manual` (default `Manual`, opt-in for durability).
- Add `BinSource::seek_to_offset(byte_offset: u64) -> Result<()>` (resume from a known offset, validating frame alignment).
- Add `BinSource::tail_from_offset(byte_offset: u64) -> Pin<Box<dyn Stream<Item = Result<(u64, RecordingEvent), ProximaError>> + Send>>` (live-tail with each event paired with its byte offset).
- Crash-point property test on real disk: spawn process, write N events, kill mid-flush at random point, restart, assert `seek_to_offset` advances past the partial frame and never serves a corrupted event.

**Fsync ordering proof (`/algorithm-development`, lands in S1 phase).**
- Invariant: a `.idx` entry referring to byte offset O is only visible after the data file has at least O+frame_len bytes durable on disk.
- Algorithm:
  1. Append frame to data file → write_all to OS buffer
  2. `data_file.sync_all()` → forces OS buffer + drive cache to disk
  3. Append index entry to .idx file → write_all to OS buffer
  4. `idx_file.sync_all()` → forces OS buffer to disk
- Crash window analysis: a crash between step 2 and step 4 leaves the data on disk but the index missing; on restart, replay must scan-forward from last `.idx` entry to recover. A crash between step 1 and step 2 may leave a partial frame on disk; `seek_to_offset` must validate frame length + checksum before yielding.

**Markers.** `BinSink` upgraded is NOT `IsPure` (touches filesystem). NOT `WithoutTime` (timestamps). The new sync APIs are `#[must_use]` on Result.

## S2 — No producer lifecycle in `src/app.rs`

**Symptom.** Earlier drafts described `ScheduledTriggerPipe` and `TelegramLongPollPipe` as registering background tasks via `Pipe::background_tasks()`. That mechanism exists but nothing spawns the returned tasks.

**Evidence.**
- `Pipe::background_tasks()` is declared at `proxima-pipe/src/pipe.rs:188`:
  ```rust
  pub struct BackgroundTask {
      pub name: String,
      pub future: Pin<Box<dyn Future<Output = ()> + Send>>,
  }
  ```
- `src/app.rs` composes request-serving Pipes via `App::mount` + `App::listen` at line 386. The mounted Pipe is invoked per request. Nothing walks the registry and spawns `background_tasks()`.
- The only existing pattern that runs continuously is `ProcessUpstream` at `src/upstreams/process.rs:225`, which spawns its supervisor directly via `tokio::spawn` inside `ProcessUpstream::spawn()` (line 141) — NOT via `background_tasks()`. The shape exists but isn't used as the universal lifecycle hook.

**Consequence.** Producer Pipes (C3 scheduled-trigger, C10 Telegram long-poll) have no way to run their loops without S2.

**S2 fix.**
- Add `ProducerLifecycle` struct in `proxima/src/app.rs` that holds a `Vec<(String, BackgroundTask)>` collected by walking the configured Pipe set during `App::build`.
- On `App::run`, spawn each task with `tokio::spawn(task.future)` wrapped in a panic catcher that surfaces task panics on a structured `ProximaError::ProducerPanicked { name, source }` channel.
- Integrate a workspace-shared `CancellationToken` for shutdown: `ProducerLifecycle::shutdown()` cancels the token, awaits all spawned `JoinHandle`s with a configurable grace period (default 5s), and reports per-task drain status.
- Tests:
  - `producer_lifecycle_spawns_background_tasks_from_configured_pipes` — happy path.
  - `producer_lifecycle_drains_in_flight_on_shutdown` — graceful shutdown.
  - `producer_lifecycle_surfaces_task_panic_on_error_channel` — sad path.
  - `producer_lifecycle_meets_grace_period_on_uncooperative_task` — forced abort after deadline.

**Markers.** `ProducerLifecycle` is NOT `WithoutSpawn` (the whole point is to spawn). NOT `WithoutTime` (deadline arithmetic).

**Home-turf bench arm.** The closest in-tree incumbent is the bespoke `while runtime.timer_at(next).await { ... }` loop at `src/scenarios/orchestrator.rs:815-838`. Bench: spawn 1000 background tasks via our lifecycle vs. drive equivalent work via the orchestrator's coupled loop. Document the regime difference (our lifecycle is general-purpose; the orchestrator is workload-coupled).

## S3 — `ProximaSettings` has no producer-graph schema

**Symptom.** The earlier `[[pipeline.heartbeat_to_stdout]]` TOML examples assumed a `pipeline.<name>` table syntax that the actual config model does not support.

**Evidence.** `ProximaSettings` at `src/settings/mod.rs:69`:
```rust
pub struct ProximaSettings {
    pub listeners: BTreeMap<String, RegistryEntry>,    // listener.public, listener.admin
    pub upstreams: BTreeMap<String, RegistryEntry>,    // upstream.backend
    pub middlewares: BTreeMap<String, RegistryEntry>,  // middleware.auth, middleware.rate-limit
    pub pipes: BTreeMap<String, RegistryEntry>,        // pipe.api, references listeners + middlewares + upstreams
}
```

Composition (per `src/settings_to_app.rs:18`) is: a listener accepts requests, hands them to a pipe, the pipe chains middlewares wrapping an upstream leaf. There is no shape for a self-starting `scheduled_trigger → stdout_alert` graph. A producer Pipe has no listener; the existing schema cannot express it.

**S3 fix (`/research-rigor` first to settle the interface).**

Three plausible interfaces, settled by a tournament before implementation:

- **Option A.** Add `producers: BTreeMap<String, ProducerSpec>` to `ProximaSettings` directly. `ProducerSpec { chain: Vec<RegistryEntry>, source: RegistryEntry }` where `source` is a Pipe with `background_tasks()` driving its inputs. Reuses the existing factory registry.
- **Option B.** Sibling registry crate `proxima-producers` with its own config struct, composed at the top-level config via `merge_into(&mut ProximaSettings)`.
- **Option C.** Separate config file `producers.toml` loaded into a sibling struct alongside `ProximaSettings`.

`/research-rigor` evaluates each on: invasiveness to existing config consumers, factory-registry reuse, principle 1 (RISC reuse) compliance, surface area, future composability with the hierarchical-guidance-routing initiative.

**Tentative recommendation pending tournament:** Option A. It reuses every existing primitive (factory registry, RegistryEntry, BTreeMap-based map), adds one field, and matches the convention. Options B and C add file-loading or merge complexity for marginal isolation benefit.

**TOML shape (subject to S3 outcome):**
```toml
[producers.heartbeat_to_stdout]
schedule = { interval_ms = 5000 }
source   = { type = "scheduled_trigger", event_kind = "heartbeat" }
chain    = ["stdout_alert"]

[middlewares.stdout_alert]
type   = "stdout_alert"
format = "human"
```

**Tests.**
- A TOML file with a `[producers.x]` section parses into `ProximaSettings`.
- The factory registry instantiates the producer chain.
- S2's lifecycle driver discovers + spawns the producer's background task.
- An end-to-end test loads the heartbeat TOML, advances tokio paused time 5s × 3, observes 3 stdout lines.

## Why these three components belong in this initiative

The user's audit identified all three substrate gaps at once; folding them in is the pragmatic call:

1. **They're tightly coupled.** S1 enables C7+C8. S2 enables C3+C10. S3 makes C3, C4, C6 configurable at all. Shipping notify components without these is shipping dead code.
2. **They're scoped enough.** Each is a focused PR-sized change: S1 is ~3 new methods + crash-point property tests; S2 is ~150 LoC in app.rs + lifecycle tests; S3 is one new field + factory adapter + tests.
3. **One coherent shipping unit.** Per principle 7 (discipline over momentum) and the disciplined-component contract, each lands with its own discipline-log row + bench + parity test. The fact that they happen to be substrate doesn't change the gate.

A future workspace-level audit may decide these substrate primitives should have been their own initiative. If so, this initiative becomes a documented "monolithic land + later refactor split" — but principle 15 (do the correct thing, no defer) makes "ship the substrate now where it actually plugs in" the disciplined call given the user's chosen path.

## Architectural consolidation (post-2ad062c push)

After the initial branch push, an architectural audit collapsed two over-fragmentations:

### One crate, not two

`proxima-notify-proto` was originally landed as a sibling crate at commit `557c290`. Per principle-1 audit, the tier-3-vs-tier-2 split that justified two crates is actually delivered cheaper by feature gates inside one crate:

- `cargo build -p proxima-notify --no-default-features --features proto` — tier-3 (no_std + no_alloc + heapless), only the `event` module compiles.
- `cargo build -p proxima-notify` (default) — full std, includes Pipe facade modules.

The proto crate has been folded into `proxima-notify` as the `event` module. Same types, same wire format, same const-generic caps from the (renamed) `proxima-notify.toml`. Net: one crate instead of two.

**Further consolidation (2026-07, grep-verified):** `proxima-notify` itself has
since been folded again into `proxima-patterns` as the `alert` module/feature —
renamed from `notify` to `alert` to resolve a name collision with the
`proxima_primitives::sync::Notify` primitive (see
`proxima-patterns/Cargo.toml`'s crate description, which lists `alert` as
"formerly proxima-notify" alongside sibling folds `balancer`, `middleware`,
`control_plane`, `kv`). Current path: `proxima-patterns/src/alert/event.rs`;
sizing TOML: `proxima-patterns/proxima-notify.toml` (the file itself kept its
name through this second fold). See `docs/proxima-notify/discipline.md`'s top
note for the full current-path mapping, including which downstream components
(scheduled-trigger, stdout-alert, stdio-guidance) also landed under
`proxima-patterns/src/alert/` and which (Telegram, webhook, durable-consumer)
remain not built.

### Per-protocol integrations are CONFIG, not Rust modules

The original plan named four follow-up initiatives — `proxima-notify-telegram` (C2+C9+C10), `proxima-notify-http-sinks` (C6), `proxima-notify-mcp-integration`, `proxima-notify-durable-delivery` (C7+C8). Architectural audit: per-protocol integrations like Telegram, Slack, Discord, PagerDuty, ntfy are NOT new Pipe impls. They are TOML compositions of existing primitives:

```toml
[pipes.telegram_send]
chain = ["body_template", "validate_body", "retry", "isolate"]
upstream = "telegram_api"

[middlewares.body_template]
type = "transform"
body_template = '{"chat_id":"${env:TELEGRAM_CHAT_ID}","text":"${event.kind}"}'

[middlewares.validate_body]
type = "validate"
schema = "schemas/telegram_send_message.toml"   # proxima-schema-format

[middlewares.retry]
max_attempts = 3

[middlewares.isolate]
timeout_ms = 10000

[upstreams.telegram_api]
type = "http"
url = "https://api.telegram.org/bot${env:TELEGRAM_BOT_TOKEN}/sendMessage"
method = "POST"
```

The same chain points at a local sim listener for offline testing — flipping modes is one URL line. The sim is itself a normal listener + `RoutingPipe` + `SynthUpstream` with body templates / sequence-of-responses for poll-style APIs.

This means the four follow-up initiatives collapse into ONE: `proxima-template-dsl-and-sim`. The net new Rust is:

- ~200 LoC: templating DSL (`${event.kind}`, `${env:X}`, `${counter:N}`, `${captured:F}`) in `proxima-compose`.
- ~40 LoC: `Transform` middleware gains `body_template` + `content_type` fields.
- ~70 LoC: `SynthUpstream` gains `body_template` + `body_sequence` fields.
- ~150 LoC: sim round-trip integration test driven by example TOML configs (Telegram, Slack, PagerDuty, Discord, ntfy).
- TOML schemas + example configs (no Rust): per-integration request/response schemas in `proxima-schema` format; real + sim example configs in `docs/proxima-notify/examples/`.

Zero new Pipes, zero new crates, zero per-integration Rust. Adding a new integration target (Discord, Slack, PagerDuty, custom internal API) = adding TOML.

### Durability (C7+C8) is still its own follow-up but inside proxima-recording-pipe

`OffsetCursor` + `DurableConsumerPipe` remain as a focused follow-up since they touch the S1 durable WAL surface and need crash-point property tests. They extend the existing `proxima-recording-pipe` crate — no new crate. Follow-up initiative renamed `proxima-recording-pipe-durable-consumer`.

### MCP `ask_for_guidance` tool

Lives in the downstream consumer daemon as application glue — outside this initiative's scope and outside the proxima workspace. Follow-up `proxima-notify-mcp-integration` retained as a small consumer-side change (~80 LoC tool handler).

### Net follow-ups after consolidation

| Initiative | What lands | New Rust | New crates |
|---|---|---|---|
| `proxima-template-dsl-and-sim` | Template DSL + Transform/Synth extensions + integration TOML schemas + example configs + sim round-trip tests | ~460 LoC across `proxima-compose` and `proxima-middleware` | 0 |
| `proxima-recording-pipe-durable-consumer` | C7 OffsetCursor + C8 DurableConsumerPipe + crash-recovery property test | ~400 LoC in `proxima-recording-pipe` | 0 |
| `proxima-notify-mcp-integration` | MCP `ask_for_guidance` tool + claude integration test | ~80 LoC in the consumer daemon | 0 |

Total new crates across the entire proxima-notify story going forward: **1** (`proxima-notify`, already landed) — **update (2026-07): 0**, since that crate was itself subsequently folded into `proxima-patterns` as the `alert` feature (see the consolidation note above), so nothing under this initiative is a standalone crate today. Telegram, Slack, Discord, PagerDuty, ntfy, MCP, durable delivery, webhook, simulator, and any future 3rd-party integration ride on existing primitives.
