# workspace SRP audit (2026-05-29)

snapshot of the single-responsibility audit across the 71-crate workspace,
including the decisions that landed and the ones explicitly deferred.

source plan: `~/.claude/plans/within-proxima-we-have-eager-steele.md`.

## landed

- **proxima-agent-fabric retirement.** branch was strictly behind main; only
  surviving work was uncommitted edits salvaged as `feat(upstreams):
  stream_passthrough upstream with conflaguration-driven transport`. local
  branch + worktree dropped; no remote ref existed.
- **proxima-integrations-core extraction.** four shared modules in core
  consumed by claude + copilot:
  - `core::credentials::CredentialStore<T>` — generic over vendor credentials
    shape. claude / copilot expose a type alias.
  - `core::tls::connect_tls(host)` — shared rustls + webpki-roots + tokio-rustls
    connect kernel; gated behind core's `client` feature.
  - `core::http::decode_response_body` + `core::http::decode_chunked` — shared
    HTTP/1.1 response head/body splitter and RFC 7230 chunked decoder.
  - (`error::IntegrationError` not extracted — `~8 LOC saved` doesn't justify
    the wrapping overhead over per-vendor `proxima-error`-derived enums.)
- **copilot symmetry fix.** TLS / session-client surface now gated behind
  a `client` feature matching claude's design. dead deps (`bytes`, `flate2`,
  `tracing`) dropped from the copilot crate.
- **intercept diamond cleanup.** `proxima-recording-core` and `proxima-replay`
  flipped from `optional = true` to unconditional so the dep-graph is
  legible without chasing feature flags. `intercept-capture` /
  `intercept-replay` features continue to gate functionality.
- **clarified Cargo.toml descriptions.** proxima-pipe, proxima-stream,
  and proxima-intercept descriptions now name the layering relationships
  rather than just the type list.

## deferred (won't-fix-until-forced)

each entry below was scoped during the audit and explicitly deferred. they
are not bugs; they are decisions to NOT act today, with the reason and the
trigger that would re-open the question.

### codec / proto / wire trait family

scoped, not started: a `MessageCodec` / `FrameCodec` / `StatefulCodec` /
`WireCodec` trait family across 9 sub-crates (proxima-h1-codec,
proxima-h2-codec, proxima-h3-proto, proxima-quic-proto, proxima-hpack,
proxima-protobuf-wire, proxima-grpc-framing, proxima-websocket-frame, and
the already-existing proxima-codec). proxima-framing-json-codec stays out
of scope (pure utility fns).

`/disciplined-component`-scale work; ~355 LOC across the 9 crates;
mandatory bench gate (`docs/codec-trait/discipline.md` + `baselines.md` +
`scripts/codec-gate.sh`) before the first impl lands. one tweak per
bench cycle, default-off flag per sub-crate, meet-or-beat each incumbent
on its design point.

trigger: when proxima-codec's traits attract a real second consumer (e.g.
a JSON-RPC streamer reusing the FrameCodec shape), spin up the bench
gate and land the family.

### recording / intercept / replay as compose + conflaguration factories

scoped, not started: `RecordingConfig`, `InterceptConfig`, `ReplayConfig`
defined via `conflaguration`'s `#[derive(Settings, Validate)]`, with
`TeeFactory::from_config`, `InterceptPipe::from_config`, and
`ReplayUpstream::from_config` producing the Pipe graph from config.

crate boundary stays — each has a different dep cone (recording-core is
trait substrate, recording-pipe carries causal byte-tracking + crossbeam
sink, intercept carries always-on rcgen + rustls + tokio-rustls + copilot
deps, replay is independent JSONL/Bin source matcher). collapsing all
four into one crate would bloat replay-only and record-only users with
TLS deps they don't need.

trigger: when the next recording or intercept config flag would otherwise
live in a hand-rolled Rust struct rather than `conflaguration`, take the
factory path. don't write more bespoke config structs.

### runtime-shaped sync / task / time (Option B)

scoped, not started: make `proxima_sync::Mutex<T>`, `proxima_task::JoinSet<T>`,
`proxima_time::sleep` generic over `R: Runtime = TokioRuntime`, with real
prime impls in `proxima-runtime-prime`. backwards compatible via the
default generic parameter; prime users opt in with the type annotation.

`/disciplined-component`-scale. discipline log at
`docs/runtime-shaped/discipline.md`. each primitive (Mutex, RwLock, Notify,
mpsc, JoinSet, Sleep) lands with its own feature flag + micro-bench
comparing tokio impl against current async-lock direct impl, plus a
prime-impl bench on prime's home turf.

trigger: when prime ships its first production workload that needs
sync / task / time primitives. today prime is experimental, so the
current "tokio-shaped under the hood" is acceptable.

### compose → graph, selection → balancer renames

scoped, not started. proxima-compose is a graph of Tee / Mount /
RoutingPipe / Diff / Isolate / Swap operators — "compose" is generic;
"graph" lands the shape. proxima-selection's failover / round-robin /
weighted strategies are load balancing, not "selection."

10+ consumer crates each. land via deprecation alias shim
(`pub use proxima_graph as proxima_compose;`) for one release cycle
before deleting the old name.

trigger: explicit user confirmation. names landed truthfully so the
audit can defer them without surprising readers.

### streaming-shaped Pipe (defer indefinitely)

`SharedRingPipe` lives at `proxima-compose/src/tee/shared_ring.rs`. it
is the multi-consumer broadcast ring crossbeam-CachePadded substrate
used INSIDE `Tee` — not a Pipe replacement.

`Pipe` stays request/response shaped. composition primitives
(`SharedRingPipe`, `Tee`, `RoutingPipe`, `Mount`) wrap request/response
Pipes and provide streaming-shaped semantics where needed. switching
`Pipe` itself to `process(impl Stream<Item=Frame>) -> impl Stream<Item=Frame>`
would relocate the same machinery into the plugin surface, force HTTP/1
(the dominant intercept use case) to wrap every request as
`stream::once`, and push backpressure modeling into application code.

trigger: a real second consumer that demands a streaming Pipe shape and
where the SharedRingPipe + Tee composition genuinely fails. today there
isn't one.

### app-protocol-to-transport gap

kafka, amqp, mqtt, redis, dns, jsonrpc, memcached are sans-IO parser
crates with zero dependency on proxima-stream / proxima-pipe / proxima-net.
to listen on Kafka wire today, callers hand-roll a tokio TcpListener loop.

this is a real gap, not a bug. Pipe's request/response shape doesn't fit
stateful subscription protocols (Kafka fetch loops, AMQP channels, MQTT
subscriptions) — they want a `Session` abstraction, which is its own
initiative.

trigger: when a concrete consumer needs Kafka or AMQP wire integration
on top of the proxima substrate. defer the Session abstraction design
until then.

### sub-1000-LOC consolidations (defer)

- `proxima-sugar` (373 LOC) + `proxima-templates` (211 LOC): both thin
  expansion helpers. merging buys nothing and couples unrelated lifetimes.
- `proxima-compose` vs `proxima-selection`: selection could be a compose
  submodule but the separation is defensible while both APIs are
  still settling.
- `PacketListener` (proxima-net) vs `StreamListener` (proxima-stream):
  two listener traits at different OSI layers. not a bug.

### scope discovered mid-audit

- `proxima-integrations-codex/` is WIP per user — do not touch.
- `proxima-integrations-codex/src/` exists but lacks a `Cargo.toml`;
  workspace member declaration is broken until the crate is finished.
- `proxima-integrations-openai/` exists and is wired into intercept; same
  duplication patterns as claude / copilot likely apply, but the
  refactor was out of the original audit scope. pick up after step 5 (or
  as a side task).
- `proxima-intercept/examples/decode-codex-ws.rs` references
  `proxima_recording_core::BinSource` without enabling
  `--features intercept-capture`. pre-existing bug; flag for follow-up.
