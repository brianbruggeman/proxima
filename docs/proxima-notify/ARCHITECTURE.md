# proxima-notify architecture

**Relocation note (2026-07, grep-verified):** the crate names below
(`proxima-notify`, `proxima-notify-proto`) describe the plan as originally
written. What actually landed relocated into `proxima-patterns` as the
`alert` module/feature (renamed to avoid colliding with the
`proxima_primitives::sync::Notify` primitive) — see `docs/proxima-notify/discipline.md`'s
top note for the grep-confirmed current paths. `proxima-telegram-proto` /
`proxima-telegram` were never built as Rust crates at all: per this doc's own
"per-protocol integrations are CONFIG" section and `SUBSTRATE.md`'s
"Architectural consolidation", Telegram/Slack/Discord/PagerDuty/ntfy
integrations were redirected to TOML composition of existing primitives
instead of dedicated crates.

Teaching surface per principle 2 of `/guiding-principles`. Names the primitives this initiative composes and explains why each wrapper exists vs. composing directly.

## One sentence

proxima-notify lands two user-visible features (one-way alerting, request/response guidance) on top of three substrate components that fix missing architecture in proxima itself — durable WAL semantics in `BinSink`/`BinSource`, producer-pipe lifecycle in `src/app.rs`, and a producer-graph schema in `ProximaSettings`.

## Why everything is a Pipe

The user direction crystallized over three refinement rounds: do not invent `AlertSink`/`AlertSource`/`GuidanceTransport` traits when `Pipe` already covers them. `Pipe` is at `proxima/proxima-pipe/src/pipe.rs:267`:

```rust
pub trait Pipe: Send + Sync + 'static {
    fn call(&self, request: Request) -> impl Future<Output = Result<Response, ProximaError>> + Send;
    fn name(&self) -> &str { "anonymous" }
    fn background_tasks(&self) -> Vec<BackgroundTask> { Vec::new() }
}
```

Sink-shaped components (return `Response { body: Body::Empty }`) and request/response components (return `Body::typed(T)`) both fit. Routing/fanout/retry/timeout come from existing middleware in `proxima-compose` and `proxima-middleware`.

## The two-tier split

Per principle 3, performance-sensitive primitives are two-tier by design:

- **tier-3 sans-IO proto crates** (`proxima-notify-proto`, `proxima-telegram-proto`): `#![no_std]`, no per-call alloc, `heapless::{String<N>, Vec<T,N>, IndexMap<K,V,N>}` with const-generic caps from per-crate `<crate>.toml` per principle 12. Compile under `cargo build -p <crate> --no-default-features`.
- **tier-2 std facade crates** (`proxima-notify`, `proxima-telegram`): wrap proto types in `Pipe` impls; integrate `tokio`, `hyper`, `std::io`. Compile under default features.

## Method-byte discriminant

Every Pipe accepting a typed `Body::typed<T>` payload dispatches on `request.method.as_ref()` per the telemetry convention at `proxima-telemetry/src/pipes.rs:322`:

- `b"ALERT"` — `Body::typed(AlertEvent)`, response `Body::Empty`
- `b"GUIDANCE_QUESTION"` — `Body::typed(GuidanceQuestion)`, response `Body::typed(GuidanceAnswer)`
- `b"SCHEDULED_TICK"` — `Body::typed(ScheduledTick)`, response `Body::Empty`
- `b"OFFSET_COMMIT"` — `Body::typed(u64)`, response `Body::Empty`
- `b"DURABLE_REPLAY"` — `Body::typed(EventBatch)`, response `Body::Empty`

A Pipe rejecting a method returns `ProximaError::method_not_supported`. The method-byte constants live in each facade crate's `pub mod methods`.

## Body::typed is NOT zero-copy

`Body::typed<T>` at `proxima-pipe/src/body.rs:199` creates an `Arc<dyn Any + Send + Sync>`. That allocates one `Arc` per construction. What it does avoid is serialization of the payload and downstream payload-byte copies on fan-out (cheap `Arc::clone`). Plan and code language consistently describe it as "in-process typed payload, heap-allocated, cheap clone, avoids serialization" — never "zero-copy" or "zero-allocation."

## The 12 marker traits

`proxima-core/src/markers.rs` defines 12 ZST marker traits that compose via AND through blanket impls:

- **Tier:** `NoStd`, `AllocFree`
- **Effect absence (negative):** `WithoutFilesystem`, `WithoutNetwork`, `WithoutSpawn`, `WithoutTime`, `WithoutRandom`
- **Umbrella:** `IsPure`
- **Determinism:** `Deterministic`, `Reproducible`, `IdempotentSideEffectFree`, `Commutative`

Each new type explicitly impls the subset it qualifies for. Audit-corrected over-claims removed: `ScheduledTriggerPipe` is NOT `IdempotentSideEffectFree` (reads clock; `fired_at` varies per dispatch), and `WebhookOutPipe` is NOT `IdempotentSideEffectFree` even with an idempotency-key header (the local Pipe cannot enforce remote endpoint behavior).

## TOML composition vocabulary

S3 extends `ProximaSettings` with a `producers: BTreeMap<String, ProducerSpec>` registry. Each `ProducerSpec` declares a chain rooted at a source pipe (no listener). Existing `listeners + upstreams + middlewares + pipes` registries stay untouched — they handle request-serving Pipes; the new `producers` registry handles self-starting Pipes.

S2 walks both registries at startup. For each Pipe, it calls `Pipe::background_tasks()` and spawns the returned futures with a `CancellationToken` for shutdown propagation. Task panics surface on a structured error path.

Concrete TOML examples land in this doc and in `tests/fixtures/` AFTER Phase 2 (S3) seals — the schema is settled by S3's `/research-rigor` tournament.

## Reused primitives (do not reinvent)

**Path caveat (2026-07):** the `proxima/proxima-pipe`, `proxima-compose`,
`proxima-middleware`, `proxima-recording-core`, `proxima-time`, `proxima-h1`,
`proxima-state-store`, `proxima-task` crate paths below predate a broader
workspace consolidation and no longer exist as separate crates in this
worktree (grep-confirmed: none of those directories are present). `Pipe`,
`Retry`, `Isolate`, `RoutingPipe`, and `SwappablePipe`/`SwapRegistry` are now
under `proxima-primitives/src/pipe/{primitives,retry,isolate,routing,
swap_registry}.rs`. A full re-audit of every path in this list is out of
scope for the alert/notify relocation this note set out to fix — treat the
specific file:line citations below as unverified until re-checked.

This initiative explicitly leans on:

- `Pipe` trait — `proxima/proxima-pipe/src/pipe.rs:267`
- `Body::typed<T>` — `proxima/proxima-pipe/src/body.rs:199` (NOT zero-copy)
- `PipeFactoryRegistry` — `proxima/proxima-pipe/src/pipe_factory.rs:43`
- `RoutingPipe` — `proxima/proxima-compose/src/routing_pipe.rs:11`
- `Tee` + `SharedRingTee<T>` — `proxima/proxima-compose/src/tee/`
- `Retry` — `proxima/proxima-middleware/src/retry.rs:101`
- `Isolate` — `proxima/proxima-compose/src/isolate.rs:34`
- `SwappablePipe` / `SwapRegistry` — `proxima/proxima-compose/src/swap.rs`
- `proxima_time::Interval` — `proxima/proxima-time/src/lib.rs:160` (two-tier no_std-capable)
- `Runtime::timer_at` — `proxima/proxima-runtime/src/lib.rs:255`
- `proxima-h1::SharedHttpClient` — `proxima/proxima-h1/src/shared_http.rs:75`
- 12 marker traits — `proxima/proxima-core/src/markers.rs`
- `OtlpHttpPipe` template (HTTP-out) — `proxima/proxima-telemetry/src/out/otlp_http/`
- `FormatterPipe` template (formatted write) — `proxima/proxima-telemetry/src/pipes.rs:1266`
- `BinSink` / `BinSource` — `proxima/proxima-recording-core/src/binary/{sink,source}.rs` (S1 upgrades these)
- `proxima-state-store` atomic-rename pattern — `proxima/proxima-state-store/src/lib.rs` (C7 mirrors this)
- `ProtocolEvent::Custom` — `proxima/proxima-recording-core/src/event.rs:160` (alert event home for persistence/replay)

## What this initiative explicitly does NOT do

See `discipline.md`'s out-of-scope section for the full list. Most notable:

- No exactly-once delivery (at-least-once only; sink-side dedup is a future initiative).
- No Discord/Slack/SMTP/SNMP/ntfy sinks beyond what's in the plan — each is its own follow-up component.
- No hierarchical multi-agent guidance routing — `GuidanceQuestion::parent_id` is reserved for it but ignored in C10.
- No Kafka backing for at-least-once — `proxima-kafka` is sans-IO parser only; not a usable queue today.
