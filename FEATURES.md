# Proxima Features

What proxima does today. Reference for evaluating adoption.

For the pitch + audience-specific framing see the marketing doc; this file is the structural inventory.

---

## Table of contents

- [Core abstractions](#core-abstractions)
- [Runtime](#runtime)
- [Listeners (server-side)](#listeners-server-side)
- [Upstreams (client-side)](#upstreams-client-side)
- [Middlewares](#middlewares)
- [Observability](#observability)
- [Recording + replay](#recording--replay)
- [Configuration](#configuration)
- [Control plane](#control-plane)
- [CLI](#cli)
- [Lock-free primitives + per-core architecture](#lock-free-primitives--per-core-architecture)
- [Performance](#performance)
- [Feature flags](#feature-flags)
- [Test + bench coverage](#test--bench-coverage)
- [What's NOT in the box yet](#whats-not-in-the-box-yet)

---

## Core abstractions

### `Pipe` trait
The universal unit. Any type that implements `Pipe` is a node in the chain. The base trait is generic over its input/output/error and is `!Send`-rooted — local is the root of the ladder, and `Send` is added by a separate, additive form.

```rust
pub trait Pipe {
    type In;
    type Out;
    type Err: Debug + 'static;

    fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>>;

    fn and_then<Next>(self, next: Next) -> AndThen<Self, Next>
    where
        Self: Sized,
        Next: Pipe<In = Self::Out>,
        Next::Err: From<Self::Err>;
}
```

- The returned future is NOT required to be `Send`; per-core dispatch is the default.
- `SendPipe: Send + Sync + 'static` is the additive cross-core form (its own `In`/`Out`/`Err`, with `Err: Send`). It is standalone, not `SendPipe: Pipe` — an RPITIT future's `Send`-ness can't be strengthened by a subtrait until return-type notation stabilises (rust#109417); when it does, these forms collapse back into `Pipe` plus a use-site bound. `UnpinPipe`/`UnpinSendPipe` add `Unpin` the same way.
- `SendDynPipe` / `DynPipe` for type erasure; `PipeHandle = Arc<dyn SendDynPipe<Request<Bytes>, Response<Bytes>>>` is the serve-shaped handle, and `ThreadLocalPipeHandle` is the `!Send` erased handle used on the per-core path.
- `Handler` is blanket-implemented for any `SendPipe<In = Request<Bytes>, Out = Response<Bytes>, Err = ProximaError>` — there is no separate "serve" trait to opt into.

### `Request` / `Response` / `Body` / `RequestContext`
- `Request { method, path, query, headers, body, context }`
- `Response { status, headers, body, upgrade }` — `upgrade` is the H/1 Upgrade hijack handler.
- `Body`: bytes-or-stream, optional cancellation token, optional trailers slot.
- `RequestContext`: telemetry, deadline, trace_id (W3C traceparent), cancel token, capture sidecar, path params, custom labels, io_uring upgrade ticket.

### `Runtime` trait
Per-core executor abstraction.

- `spawn_on_current_core(future: ?Send + 'static)` — per-core task
- `spawn_on_core(CoreId, Send + 'static)` — cross-core dispatch
- `spawn_factory_on_core(CoreId, FnOnce -> ?Send Future)` — Send factory ships cross-core, ?Send future constructed at destination
- `spawn_background_blocking(...)` — CPU-bound work off the chain runtime (delegates to `BackgroundPool`)
- `timer_at(Instant)` — runtime-driven timer
- `num_cores()`, `current_core()`

### `BackgroundPool` trait
Pluggable cross-thread CPU-bound work backend.
- Default: `tokio::task::spawn_blocking`
- `RayonBackgroundPool` (feature `rayon`): work-stealing fork-join pool

### `App` / `AppBuilder`
- `App::new()` creates a runtime + pipe registry + listen registry + load context
- `app.pipe(name, spec)`, `app.mount(pattern, target)`
- `app.run_until_signal(RunConfig)` — bind a listener, returns `Shutdown` handle
- `app.serve(impl Into<RunConfig>)` — fluent terminal, returns a `Server` handle (`Clone`, `IntoFuture`)
- `app.apply_settings(&ProximaSettings)` — materialize typed Settings into the App (upstreams + composed pipes)
- `app.build_listener(spec)` — multi-listener apps
- `AppBuilder` for fluent construction; `Spec` accepts inline JSON/TOML/YAML/etc., file paths, or constructed `PipeHandle`s

### `Server` (fluent terminal, returned by `App::serve(...)`)
- Wraps the listener-lifecycle `Shutdown` plus an `Arc<dyn ControlPlane>` operator surface
- `Clone` via internal `Arc`; clones share control-plane state, listener-loop drive is single-owner
- Three drive shapes: `.await` (terminal future via `IntoFuture`), `.run_until_signal()`, `.run_until_signal_with_drain()`, `.stop()`, `.drain()`
- Impls `ControlPlane` so any clone can `list_pipes`, `status`, `start`, `stop`, `restart`, `apply`, `snapshot_metrics`, `shutdown`, `logs`
- Same trait surface that `ControlPlanePipe` exposes over HTTP — one concept, two access modes

### `ProximaSettings` (typed configuration)
- Top-level shape: `BTreeMap<String, RegistryEntry>` for listeners / upstreams / middlewares / pipes + nested `HttpTuning` / `ZstdTuning` / `BufferPoolTuning`
- `bon::Builder` derive on every struct — fluent `T::builder().field(...).build()`
- `conflaguration::Settings` derive on tunables — env-var overrides for free (`PROXIMA_HTTP_RESPONSE_BUFFER_BYTES=32768`)
- `ProximaSettings::from_path("proxima.toml")` — supports TOML, JSON, YAML, RON, JSON5, XML by extension sniff
- Round-trip property: Settings ⇄ TOML ⇄ Settings is identity (tested in `tests/units/settings_round_trip.rs` and `tests/units/settings_to_app.rs`)

### Typed fluent configs (`proxima::settings::*`)
- Listeners: `HttpListener`, `HttpsListener` (feature `tls`), `HttpUdsListener` (cfg `unix`) — each `Into<RunConfig>`
- Upstreams: `HttpUpstream` — `Into<Spec>` emits `{ "type": "http", ... }`
- Middlewares: `BearerAuth`, `RateLimit` — `Into<Spec>` emits `{ "type": "<tag>", ... }`
- `Composable::then(layer)` — blanket trait impl on any `T: Into<Spec>`, produces a `Chain`. Top-down code order = request execution order.

---

## Runtime

### `PrimeRuntime` (default, feature `serve-prime`)
- From-scratch, tokio-free per-core runtime: one thread per core, no work-stealing
- `serve-prime` pulls the four `runtime-prime-*` primitives (`runtime-prime-executor`,
  `runtime-prime-inbox-alloc`, `runtime-prime-reactor`, `runtime-prime-bgpool`) plus
  the prime HTTP upstream dep bundle (`http-prime-deps`) — this is what the default
  build pulls so `PrimeRuntime` is the serve+chain runtime out of the box
- `tokio` is NOT in the default dependency graph at all (verify with
  `cargo tree -e normal -i tokio`)

### `TokioPerCoreRuntime` (opt-in, feature `runtime-tokio`, requires `tokio`)
- N OS threads, one tokio current-thread runtime + `LocalSet` per thread
- Threads pinned to CPU cores via `core_affinity` (best-effort)
- Cross-core dispatch via per-core `flume` MPSC channels
- `tokio_uring::start(...)` swap-in under `io-uring` feature on Linux

### `io_uring` (feature `io-uring`, Linux only)
- Per-worker `tokio_uring::start` replaces tokio current-thread
- Listener accepts via `tokio_uring::net::TcpListener`
- Owned-buffer I/O via `tokio_uring::net::TcpStream::read`/`write`
- Streaming bodies dispatched via mpsc + `spawn_local`'d `Pipe::call`
- TLS termination via `UringAsyncStream` adapter + `tokio_rustls`
- Upgrade hijack via `LocalUpgradeHandler` + thread-local ticket registry
- HTTPS upstream over `tokio_uring::net::TcpStream` (Stage 5e/5f)

### `ShutdownBarrier`
- 4-phase graceful shutdown: quiesce → drain → drop → exit
- `register_per_core_resource(name, on_drop)` — Pipe authors stash `!Send` cleanup hooks
- `Shutdown::drain()` awaits listener completion before broadcasting drop hooks
- Drop hooks fire on the OS thread that owns the resource (LIFO registration order)

---

## Listeners (server-side)

| Listener | Module | Feature |
|---|---|---|
| HTTP/1.1 (native state machine, not hyper) | `listeners::http` | default |
| HTTP/1.1 over io_uring | `listeners::http_uring` | `io-uring` |
| HTTP/2 (native state machine, no `h2` crate) | `h2::server` | `http2` |
| HTTP/2 ALPN multiplex (`h2` crate, transitional) | `listeners::h2` | `http2` |
| MCP (Model Context Protocol via stdio) | `listeners::mcp` | default |
| WebSocket | `listeners::websocket` | `websocket` |
| QUIC + HTTP/3 (via `quinn` + `h3`) | `listeners::quic` | `quic` |
| Generic TCP stream | `listeners::tokio_stream` | `tcp` |
| Generic Unix domain socket | `listeners::tokio_stream` | `unix` |
| Generic UDP packet | `listeners::tokio_packet` | `udp` |
| AF_XDP / XDP (skeleton) | `listeners::xdp_packet` | `xdp` |
| DPDK (skeleton) | `listeners::dpdk_packet`, `listeners::dpdk_stream` | `dpdk` |

### HTTP listener (`listeners::http`)
- Own connection state machine (`h1_connection.rs`) using `httparse` for head, custom body decoder for chunked + Content-Length, response writer for chunked + CL framing
- TLS termination via `tokio-rustls` (feature `tls`)
- Upgrade hijack — H/1 protocol switch (WebSocket, h2c, CONNECT tunnel, custom binary protocols)
- `Connection: close` + keep-alive
- HTTP/1.1 Expect: 100-continue with optional reject policy
- Chunked transfer encoding (request + response)
- Auto-stream policy: body >1 MiB Content-Length or chunked → streaming dispatch
- Trailers (request + response)
- Quiesce window on shutdown (`quiesce_duration_ms` + configurable status + Retry-After)
- Drain timeout
- Cancel-on-disconnect (default listener path; io_uring buffered path tracked as Stage 5e+)
- `proxima.connections_accepted_total` + `proxima.requests_total` + `proxima.request.latency_ms` metrics

### io_uring listener (`listeners::http_uring`)
- All of the above on Linux ≥ 6.0 with `--features io-uring`
- Owned-buffer reads/writes (no epoll)
- Per-core SO_REUSEPORT accept
- Streaming bodies (Stage 5c-1)
- Upgrade hijack via `LocalUpgradeHandler` (Stage 5c-2)
- HTTPS upstream forwarding over `tokio_uring::net::TcpStream` (Stage 5f)

### HTTP/2 native listener (`h2::server::serve_h2_connection`)
- Sans-IO state machine, no `h2` crate dependency on the protocol path
- Full RFC 7541 HPACK: integer codec, Huffman (zero-unsafe, fixed-table), static + dynamic tables, encoder + decoder
- Stream state machine + per-stream flow control (RFC 7540 §5)
- Connection-level flow control with auto-WINDOW_UPDATE replenishment
- Send-side flow-control wait/resume (responses larger than the initial 65,535-byte window pause + resume on peer WINDOW_UPDATE)
- Connection lifecycle: preface, SETTINGS exchange, PING, GOAWAY
- Compatible with the existing `Pipe` / `PipeHandle` dispatch
- Measured 24% faster than hyper, 28% faster than pingora on default tokio at single-stream warm-GET; 52-64% faster at conn=64 on the per-core runtime (see `benches/RESULTS_linux.md`)

### MCP listener
- stdio framing for Model Context Protocol
- Request/response cycle hosted via the same `Pipe` trait

---

## Upstreams (client-side)

| Upstream | Module | Use case |
|---|---|---|
| HTTP forwarder (hyper-util pooled / io_uring direct) | `upstreams::http` | reverse-proxy to backend HTTP services |
| TCP raw byte stream | `upstreams::tokio_stream` | proxy to length-prefixed protocols |
| Unix domain socket | `upstreams::tokio_stream` (cfg unix) | local IPC |
| Subprocess pipe (stdio) | `upstreams::process` | spawn child, talk via stdin/stdout |
| Subprocess RPC (length-framed JSON) | `upstreams::process_rpc` | child subprocess speaking framed JSON-RPC |
| Filesystem (read static / template) | `upstreams::fs` | serve files |
| KV cache | `upstreams::kv_cache`, `upstreams::kv_upstream`, `upstreams::kv_file` | KV-backed cache layer |
| Synthetic (synth/replay) | `upstreams::synth`, `upstreams::replay` | testing |
| Recording wrapper | `upstreams::record` | capture upstream calls for replay |
| Stream passthrough | `upstreams::stream_passthrough` | bidirectional raw bytes |
| Callback (in-process closure) | `upstreams::callback`, `upstreams::callback_registry` | unit-test fixtures |

### HTTP upstream
- Pooled hyper-util `Client` (default) — one process-wide pool, shared across `HttpUpstream` instances
- TLS via `hyper-rustls` (feature `tls`)
- io_uring path (feature `io-uring`): `tokio_uring::net::TcpStream` + raw `hyper::client::conn::http1::handshake` (per-request connect today; pool deferred)
- HTTPS upstream over io_uring: webpki-roots `ClientConfig` + `tokio_rustls::TlsConnector`
- Per-call cancellation (parks on `request.context.cancel`)
- Configurable timeout, method override, header forwarding (allow-list), header injection (templates), auto-traceparent propagation

---

## Middlewares

All composable — each wraps an `inner: PipeHandle` and is itself a `Pipe`.

| Middleware | Module | What |
|---|---|---|
| `Auth` | `middlewares::auth` | Pluggable auth via configurable verifier (bearer, basic, custom) |
| `RateLimit` | `middlewares::rate_limit` | Token-bucket, lock-free (AtomicU64 CAS-loop refill+consume) |
| `Retry` | `middlewares::retry` | Configurable backoff (constant, exponential, jitter), max attempts, retry-on policies |
| `Transform` | `middlewares::transform` | Request/response shape changes (header rewrites, body templates, method overrides) |
| `Validate` | `middlewares::validate` | JSON schema validation (request / response) |
| `WriteBack` | `middlewares::write_back` | Fan-in to KV cache after upstream success |
| `Isolate` | `middlewares::isolate` | `catch_unwind` boundary; panic in Pipe → ProximaError instead of process crash |
| `Diff` | `middlewares::diff` | Tee + replay against a candidate pipe; emit byte-level diff |
| `SwappablePipe` | `swap` | Atomic mid-flight pipe swap (hot-swap via `proxima apply`) |
| `Tee` | `tee` | Body fan-out for selection fall-through + recording sinks. Bounded backpressure via ArrayQueue + AtomicWaker |
| `Selection` | `selection` | Fallthrough chains (least-conn, round-robin, weighted-least-conn) |
| `Mount router` | `mount`, `path_pattern` | Path-pattern routing with `{param}` and `{*wildcard}` extraction |
| `Causal` (causality wrapper) | `causality` | Records byte-range causal edges around an inner pipe call |
| `ContextInjector` | `context_inject` | Injects telemetry handle, defaults, etc. into RequestContext |

---

## Observability

### Telemetry (`telemetry` module) — Pipe-shaped redesign

The telemetry substrate is built on the same `Pipe` primitive as every other proxima component. Every telemetry record (span, log, metric, event) is wrapped in a `Request` envelope and dispatched through a terminal `Pipe`. Exporters are Pipes; fanout is `Tee<T>`; the recorder drains into a `PipeHandle`.

**Structural substrate (always-on, no feature flag):**
- `ring` — per-core SPSC lock-free ring buffer; 11.6× faster than crossbeam at 1k, 4.4× at 1M
- `id` — `TraceId` / `SpanId` + W3C traceparent SIMD-branchless parse; 5.4× faster than OTel's parser
- `level` — custom severity levels (sub-ns compare; supports user-defined levels beyond the 5 built-ins)
- `tag` — domain-agnostic `Tag` + `TagSink` trait; 64-byte stack value, no `Box<dyn>` dispatch
- `trace` — `SpanRecord` / `SpanBuilder` / `SpanGuard`; RAII drop emits into per-core ring
- `metric` — `Counter` + direct-instrument `AtomicU64::fetch_add` (2.19 ns, at parity with prometheus); `Gauge`; `UpDownCounter`
- `log` — `LogRecord` + `LogBuilder`; 17 ns bare emit; 381 ps filter rejection
- `recorder` — `Recorder` + per-core `RingSet` + typed bags (`Resource` / `ScopeHandle` / `Baggage`) + `Drainer` + `InstrumentRegistry`
- `native` — `NativePipe<S: FrameSink>` postcard-framed DPU↔host wire format; 27% smaller wire than OTLP
- `config` — `TelemetryConfig` (conflaguration `Settings` + `Validate` + `bon::Builder`); 3.22 ns default, 5.79 ns validate

**Terminal Pipes (`telemetry::pipes`):**
- `NullPipe` — no-op; default when no real exporter is configured
- `NativePipe<S>` — postcard wire format over any `FrameSink`
- `OtlpHttpPipe` (feature `otlp-http`) — OTLP/HTTP protobuf encoding; 12% within OTel SDK on encode speed; 1.5% larger wire
- `OtlpGrpcPipe` (feature `otlp-grpc`) — OTLP/gRPC with single-alloc backpatch framing; 10.3% faster than OTel SDK on home-turf encode+frame arm
- `CountingPipe` — test helper; per-record-type atomic counters

**Fanout:**
- `Tee<T>` (feature `tee-generic`) — generic record fanout; `ArrayQueue`-backed per-sink queue with backpressure; replay buffer for late-arriving consumers

**TracingLayer bridge (feature `tracing-init`):**
- `TracingLayer` — `tracing_subscriber::Layer` that bridges upstream `tracing::info!()` / `tracing::span!()` calls from hyper, rustls, tokio, and third-party crates into proxima's per-core `Recorder`
- 1.7× faster than `tracing_subscriber::fmt` to `io::sink` (319 ns vs 546 ns per event); defers field formatting to drain time
- Install: `registry().with(EnvFilter).with(TracingLayer::new(Arc::clone(&recorder)))`

**End-to-end composition results (Phase K + L, Darwin aarch64):**
- Traces: proxima 350 ns vs OTel SDK 723 ns — **proxima 2.06×**
- Metrics: proxima 5.78 ns vs prometheus 54.5 ns — **proxima 9.4×**
- Logs: proxima 105 ns vs tracing fmt 910 ns — **proxima 8.7×**
- Full 5-signal: proxima 663 ns vs 4-crate stack 1,740 ns — **proxima 2.62×**

**OTel SDK speed gate (Phase P6, Darwin aarch64, N=10 000 spans):**
- proxima `recorder.span(...).tag(...).start()` + drain: **150 ns/span** (1.5 ms / 10k)
- OTel SDK `tracer.start(...) + set_attribute + end` (InMemorySpanExporter): **351 ns/span** (3.5 ms / 10k)
- **proxima 2.34× faster than OTel SDK on span emit with one tagged attribute**

- `Metrics` implementation — counters, gauges, histograms with HDR-histogram backing
- Per-thread sharded histograms (`ShardedHistogram`) — uncontested record path via `ThreadLocal<Mutex<Histogram>>`
- Built-in metrics: `proxima.requests_total`, `proxima.connections_accepted_total`, `proxima.connections.active`, `proxima.upstream.calls_total`, `proxima.upstream.errors_total`, `proxima.upstream.latency_ms`, `proxima.request.latency_ms`, `proxima.requests.in_flight`
- `Labels` for dimensional labeling; `MetricsSnapshot` for export
- `NoopTelemetry` for tests

### Capture (`recording::capture`)
- `CaptureContext` per-request: Pipe-attached key/value pairs ride with recorded frames
- Lock-free via `SegQueue<(String, Bytes)>`
- Last-write-wins per key on `drain()` into `FrameMetadata`
- Drained at frame-emission boundaries by `RecordUpstream::ChunkRecorder`

### Causal (`causality` module)
- `CausalIndex` — byte-range edge graph (per-thread sharded `Vec<Mutex<Vec<CausalEdge>>>`, Stage 3c)
- `Causal` middleware records edges around inner Pipe calls
- `explain(node_id, offset)` walks ancestors backward from a recorded output byte
- `write_jsonl` + `read_jsonl` for offline analysis
- Default slot count `min(num_cpus, 64)`; explicit `with_slots(N)` for high-core DPDK
- 57 ns/record uncontested Linux, 168 ns/record at 16 concurrent recorders

### Tracing (`tracing_init` module, feature `tracing-init`)
- `TracingLayer` adapter bridges `tracing::` events into the per-core `Recorder` (see Telemetry section above)
- `init_tracing(recorder, format)` installs `TracingLayer` + `EnvFilter` via `registry().with(filter).with(layer).try_init()`
- `init_tracing_default(format)` convenience wrapper backed by a `NullPipe` recorder
- Spans across `tokio::spawn` boundaries via `Instrument`
- Per-request span carrying trace_id from `traceparent`

### Log buffer (`log_buffer` module)
- Per-supervised-pipe stdout/stderr ring buffer (`ArrayQueue<String>`)
- Live-tail subscribers via per-subscriber `ArrayQueue<String>` + `Notify`
- `LogBufferRegistry` keyed by pipe name (DashMap)
- Used by `process` upstream + `daemon control plane` for `proxima logs` CLI

### Determinism (`determinism` module)
- `check_determinism(build, request)` — property-test substrate
- Runs the same request through a Pipe N times and asserts byte-identical output
- Catches non-determinism: clocks, RNGs, HashMap iteration order, etc.

---

## Recording + replay

### Capture (`recording/cap.rs`, `recording/capture.rs`, `recording/event.rs`)
- `RecordingEvent` enum: RequestBegin, RequestEnd, ChunkRecorded, etc.
- `FrameMetadata` carries Pipe-attached key/values per frame
- Recorded sessions are interaction-id-keyed (`InteractionId` = 16 bytes)

### Sinks (write)
- `recording::jsonl::JsonlSink` — one JSON-encoded event per line
- `recording::bin::BinSink` — length-prefixed binary frames + index file (zstd-compressed)
- Pluggable via `RecordingSink` trait + `RecordingSinkFactory` registry

### Sources (read)
- `recording::jsonl::JsonlSource` — reads JSONL recordings
- `recording::bin::BinSource` — reads binary recordings with index-based seek
- Pluggable via `RecordingSource` trait

### Replay (`upstreams::replay`)
- `ReplaySource` Pipe replays a recorded interaction by interaction-id
- Determinism harness asserts replayed output matches recorded output

### `Tee` + recording
- `Tee::wrap(body, cap_bytes)` returns a primary stream + replayable Body
- `Tee::sink()` returns a Body subscriber; backpressure stalls the primary at sink_queue depth
- Selection fall-through uses Tee to retry a captured body against a fallback pipe

---

## Configuration

For the user-facing walkthrough see [`docs/configuration.md`](docs/configuration.md).

### `ProximaSettings` (typed top-level config)
- Map-keyed registries: `listeners`, `upstreams`, `middlewares`, `pipes` — each a `BTreeMap<String, RegistryEntry>` where `RegistryEntry = { type, spec }` for plugin-extensible late-typed deserialization
- Nested tunables: `HttpTuning`, `ZstdTuning`, `BufferPoolTuning` — env-overridable via `conflaguration::Settings` derive (`PROXIMA_HTTP_RESPONSE_BUFFER_BYTES=32768`)
- Every struct derives `bon::Builder` for fluent construction + serde for TOML/JSON/YAML/RON/JSON5/XML round-trip
- `ProximaSettings::from_path("proxima.toml")` — extension-sniffed loader through the config-format registry
- `App::apply_settings(&settings)` — materializes upstreams + composed pipes into the App
- Round-trip property tested: Settings ⇄ TOML ⇄ Settings is identity

### `Spec` (load module — lower-level)
Multiple sources, autodetect:
- `Spec::Inline(Value)` — `serde_json::Value`
- `Spec::Path(PathBuf)` — file on disk, format inferred from extension
- `Spec::Raw { content, format }` — inline source text with optional format hint
- `Spec::Toml(String)` — backward-compat TOML
- `Spec::Handle(PipeHandle)` — pre-built pipe (test fixtures, in-process pipes)

### Config formats (`config_format` module)
- JSON, TOML, YAML, JSON5, RON, XML
- `ConfigFormatRegistry` (lock-free via `ArcSwap`) — pluggable parser registration
- Format sniffing when no explicit hint

### Schema validation (`schema` module)
- JSON schema for spec validation
- `SchemaRegistry` (lock-free via `ArcSwap`)

### Pipe factories (`pipe_factory` module)
- `PipeFactory` trait — builds a `PipeHandle` from a spec value
- `PipeFactoryRegistry` (lock-free via `ArcSwap`)
- Built-in factories: HTTP upstream, KV cache, fs, process, process-rpc, synth, replay, record, callback, stream-passthrough
- User-defined factories register at app construction

### Listen protocols (`listen` module)
- `ListenProtocol` trait — bind + serve
- `ListenRegistry` (lock-free via `ArcSwap`)
- Built-in: http, mcp, websocket, quic, stream, packet, http_uring

### Templates (`templates` module)
- `{{request.trace_id}}`, `{{request.path}}`, `{{request.pipe}}` expansion in injected headers / response bodies
- Used by `Transform` middleware

### Codecs (`codec` + `codec_factory`)
- `Codec` trait for body encoding/decoding (currently zstd; gzip planned)
- `CodecFactory` registry (lock-free via `ArcSwap`)

---

## Control plane

### `ControlPlane` trait (`control_plane` module)
- `swap(node_id, factory_spec)` — mid-flight pipe swap
- `status(name)` — pipe health snapshot
- `metrics()` — telemetry snapshot
- Lock-free pipe status map via `ArcSwap<HashMap>`

### Daemon control plane (`daemon_control_plane` module)
- Long-running daemon variant of proxima
- Unix-socket RPC for `proxima apply`, `proxima logs`, `proxima status`
- Spawns supervised process upstreams + collects their stdout/stderr into `LogBufferRegistry`
- Per-pipe lifecycle: start, stop, restart, status

---

## CLI

`proxima` (in `tools/proxima-cli/src/main.rs`):

| Command | What |
|---|---|
| `proxima call --config <path> [--method] [--path] [--body] [--header ...]` | One-shot `Pipe::call` from CLI; output to stdout |
| `proxima serve [spec] [--config] [--addr] [--mount]` | Load a pipe config, bind an HTTP listener, serve until SIGINT/SIGTERM; prints `READY <addr>` |
| `proxima describe --config <path> [--format json-schema\|openapi\|toml]` | Emit the config's registered `[[schema]]` blocks in the chosen format |
| `proxima daemon [--socket <path>] <action>` | Talk to a running daemon's control-plane UDS; actions: `list`, `status <name>`, `metrics`, `logs <name> [--max-lines]`, `start <name>`, `stop <name>`, `restart <name>`, `reload`, `apply <name> --spec <path>` |
| `proxima apply <name> --spec <path> [--socket]` | Convenience alias for `proxima daemon apply` — same wire protocol |
| `proxima explain --index <path> --node <id> --offset <n>` | Walk the causal graph backward from a recorded output byte |
| `proxima pipeline [--socket \| --host] <action>` | Submit/inspect/replay pipelines on a running `proximad`; actions: `submit`, `list`, `resolve`, `inspect`, `tail`, `events`, `explain --stage`, `replay --substitute`, `artifact --stage --path [--output]` |
| `proxima verify [spec] [--policy] [--format] [--strict] [--repair]` | Walk a parsed spec and report policy violations; exit 0/1/2 |
| `proxima replay [--recording] [--verify <policy>] [--spec] [--format] [--strict] [--repair]` | Stream a recorded `.bin` session, run replay-policy assertions |
| `proxima load [scenario] [--name] [--json] [--rps] [--duration] [--remote] [--record]` | Drive a load scenario from a TOML file (open-loop or closed-loop, from the workload spec) |

`proxima daemon <action>` covers what used to be modeled as separate top-level
`status`/`logs`/`start`/`stop`/`restart` commands — they are subcommands of
`daemon`, not top-level commands.

---

## Lock-free primitives + per-core architecture

| Primitive | Where | What |
|---|---|---|
| `ArcSwap<T>` (single Arc) | `SwappablePipe::delegate` | Atomic mid-flight handle swap; ~12ns read Linux |
| `ArcSwap<HashMap>` / `ArcSwap<BTreeMap>` | All registries (PipeFactory, Listen, Schema, Codec, Recording) | Lock-free reads, CAS-loop CoW on write |
| `DashMap` | `LogBufferRegistry` | Sharded RwLock; sub-process log buffer mapping |
| Per-thread sharded `Vec<Mutex<Vec<T>>>` | `CausalIndex` | Stage 3c — 2.7-15× faster than Mutex<Vec> at concurrency |
| `ThreadLocal<Mutex<Histogram>>` | `ShardedHistogram` | Uncontested per-thread record; merge fallback for snapshots |
| `SegQueue<T>` | `CaptureContext::pending` | Lock-free MPMC queue for attached key/values |
| `ArrayQueue<T>` | `LogBuffer`, `Tee::sinks`, `log_buffer::Subscriber` | Bounded MPMC, lock-free |
| `AtomicWaker` | `Tee::SinkSlot` | Cross-poll cross-thread single-waker for backpressure |
| `AtomicU64` (CAS-loop) | `RateLimit::token_bucket`, sequence counters | Lock-free atomic state |
| `thread_local!` + `RefCell` | `ShutdownBarrier::RESOURCES`, `CausalIndex::THREAD_SLOT` | Per-core ownership |

### Why every Mutex is justified

Each remaining `Mutex`/`RwLock` in production code carries a three-part comment:
```
// WHY Mutex here:    <invariant the lock guards>
// WHY NOT removable: <bench-cited or structurally ruled-out alternatives>
// WHY this is right: <bench citation OR structural reasoning>
```

Bench-validated remaining sites:
- `tee.rs::Mutex<TeeState>` — multi-field state machine atomic transition (cites `tee_backpressure.rs`)
- `telemetry.rs::Mutex<Histogram>` — hdrhistogram needs `&mut self`, fronted by ThreadLocal (cites `histogram_record.rs`)
- `CausalIndex` per-thread slots `Mutex<Vec<CausalEdge>>` — Stage 3c (cites `causal_record_primitives.rs`)
- Recording sinks `Mutex<File>` — file write atomicity for >PIPE_BUF records (cites `recording_sink_primitives.rs` showing SegQueue+writer-task is 30-40× faster, deferred)

Structurally required (no bench applies):
- `Mutex<Option<Future>>` interior-mutability in framing / upstreams / quic / websocket — `&self` trait API + futures not movable through atomics + RefCell would force !Send
- `WebSocketConnection::read_buffer: Mutex<Vec>` — message-vs-byte mismatch in AsyncRead surface

---

## Performance

### Hot path (Linux host-b, criterion `--quick`)

| Bench | Time |
|---|---|
| `h1_parse_head` small GET, 5 headers | 183 ns |
| `Connection` round-trip, no body | 298 ns |
| `Connection` round-trip, POST + 5-byte body | 293 ns |
| `h1_streaming` cl_256 buffered | 295 ns |
| `h1_streaming` cl_256 streaming | 327 ns |
| `substrate_dispatch` 1 pipe | 344 ns |
| `substrate_dispatch` 4 pipes | 690 ns |
| `substrate_dispatch` 16 pipes | 1.10 µs |
| `per_core_vs_arcswap` thread-local read | 2.5 ns |
| `per_core_vs_arcswap` ArcSwap read uncontended | 12 ns |
| `swap_under_load` swap() | 100 ns |
| `capture_drain` attach + drain | 121 ns |
| `causal_record` single edge | 85 ns |
| `tee_backpressure` wrap+drain 256B | 197 ns |

### HTTP/1.1 vs hyper, same loopback transport

| | proxima | hyper |
|---|---|---|
| macOS | 67.4 µs | 70.0 µs |
| Linux | 62.0 µs | 68.2 µs |

### HTTP/1.1 vs pingora (Cloudflare), same loopback transport

| | proxima | pingora |
|---|---|---|
| macOS | 67.4 µs | 74.9 µs (11% slower) |
| Linux | 62.0 µs | 76.4 µs (23% slower) |

Caveat: ~95% of the 60-75 µs is kernel-side (socket bind, accept, connect, TCP handshake, EOF detect). The proxima/hyper/pingora user-space connection driver is 100-500 ns of that total. The relative gap is the substrate-driver lead; the absolute number is dominated by kernel TCP. DPDK Stage 11 eliminates the kernel side.

### HTTP/2 head-to-head (host-b, Linux 6.15, 4 server cores)

Warm h2 client; sequential GETs over **independent TCP connections**;
5-run RPS medians; bodies = `b"ok"`. Full data + variance + tail
percentiles in `benches/RESULTS_linux.md`.

| connections | proxima native + per_core | proxima native + default tokio | hyper | pingora |
|---|---|---|---|---|
| 1 | **37,346** | 33,202 | 23,467 | 23,353 |
| 4 | **96,501** | 91,463 | 66,428 | 68,097 |
| 16 | **246,170** | 142,848 | 174,420 | 164,121 |
| 64 | **286,447** | 189,219 | 188,670 | 175,207 |

Per-core proxima native is the production deployment. At conn=64 it is
52% faster than hyper, 64% faster than pingora — with coefficient of
variation 2.0% across 5 trials (vs 4.6-9.0% for the default-tokio
variants).

Single-stream warm h2 (multi-stream over one TCP connection):

| | proxima native + per_core | hyper | pingora |
|---|---|---|---|
| 1 stream | **41,103 rps, p99 29 µs** | 23,947 rps, p99 78 µs | 21,639 rps, p99 85 µs |
| 100 streams | 198,891 rps, p99 571 µs | 182,257 rps, p99 996 µs | 178,857 rps, p99 967 µs |

HPACK microbenches (apples-to-apples vs the `h2` crate's HPACK):

| layer | proxima native | h2 crate |
|---|---|---|
| Huffman decode | **2.5-3× faster** | baseline |
| Huffman encode | tied or +5-17% | baseline |
| Integer codec | tied | baseline |
| Static table lookup | tied | baseline |

### Why the per-core runtime matters

Default tokio multi-thread uses work-stealing. Under high concurrency
that adds cross-CPU wake-ups, cache-line bouncing, and run-to-run
scheduler jitter. The `runtime-tokio` per-core runtime pins one tokio
current-thread runtime per CPU via `core_affinity` and dispatches
connections to cores via `flume`. Each connection's state stays on
one core's L1/L2 — no work-stealing — and the RPS distribution
tightens 2-3× across trials.

macOS doesn't show the same advantage because Darwin's
`core_affinity::set_for_current` is best-effort and tokio's
work-stealing is well-tuned for M-class. Linux is where the per-core
architecture pays off.

---

## Feature flags

```toml
default = ["serve-prime", "http2", "http3", "histogram", "macros", "http-prime-deps"]
```

`serve-prime` makes `PrimeRuntime` the default serve+chain runtime — `tokio` is
NOT in the default dependency graph at all. `http2`/`http3` resolve to the
native, tokio-free drivers. Opt into the tokio-backed capability set
(sister-tokio serve runtime, hyper, quinn-compat h3, h1) with `--features
tokio`; `http1` specifically needs `tokio` because its connection driver has
no sans-IO implementation yet (h2/h3 do).

Telemetry substrate primitives (`ring`, `id`, `level`, `tag`, `trace`, `metric`, `log`, `recorder`, `native`, `config`) are structural — always-on, no feature flag. Only consumer-facing toggles that change the dependency closure remain as features.

| Flag | What it enables |
|---|---|
| `serve-prime` | Default on — `runtime-prime-{executor,inbox-alloc,reactor,bgpool}` + `http-prime-deps`; `PrimeRuntime` is the serve+chain runtime out of the box |
| `http2` | Default on — native h2c listener, no `tokio`, no external `h2` crate on the protocol path |
| `http3` | Default on — native sans-IO QUIC+H3 stack, no `tokio`, no `quinn` |
| `http-prime-deps` | Default on — prime-native HTTP/1.1 client stack; prime is the default `"http"` upstream backend on unix |
| `tokio` | The single opt-in that restores the entire tokio capability set (`tokio`/`tokio-util` deps, `tokio::process` upstreams, MCP stdio listener, OS signal handling, tokio-native accept-loop fallbacks, sister-tokio-compat prime worker). Required explicitly by anything that still needs a tokio runtime: `http1`, `tls`, `tcp`/`udp`/`unix`, `quic`, `websocket`, `http-hyper`, `io-uring` |
| `tokio-runtime` | Gates the tokio-specific listener / upstream impls (`listeners::tokio_stream`, `upstreams::tokio_stream`); implied by `tokio` |
| `runtime-tokio` | `TokioPerCoreRuntime` (per-core dispatch); opt-in, implied by `tokio` |
| `tcp` / `unix` / `udp` | Byte-stream / packet protocol families (require `tokio`) |
| `tls` | TLS termination (listener) + HTTPS upstream (`rustls` + `hyper-rustls` + `tokio-rustls` + `rcgen`); requires `tokio` |
| `io-uring` | Linux-only: tokio_uring listener + HTTPS upstream over uring + webpki-roots |
| `websocket` | WebSocket listener (`async-tungstenite`) |
| `quic` | QUIC listener (`quinn` + `rustls`); native sans-IO QUIC is `quic-native` (part of default `http3`) |
| `xdp` | AF_XDP packet listener (skeleton — returns Unsupported today) |
| `dpdk` | DPDK packet/stream listener (skeleton — Stage 11) |
| `rayon` | `RayonBackgroundPool` for cross-thread CPU work |
| `histogram` | HDR-histogram instrument (`Histogram<f64>`); default on (dep weight justified by the instrument shape) |
| `otlp-http` | `OtlpHttpPipe` OTLP/HTTP protobuf exporter (pulls `prost`) |
| `otlp-grpc` | `OtlpGrpcPipe` OTLP/gRPC framed exporter (implies `otlp-http`) |
| `macros` | `#[span]` proc-macro + `#[derive(SpanCarrier)]`; default on — 3.16× faster than `#[instrument]` |
| `tracing-init` | `TracingLayer` adapter that bridges `tracing::` events into the per-core `Recorder`; `init_tracing` helper |
| `tee-generic` | Generic `Tee<T>` record-fanout primitive with replay and backpressure |
| `runtime-prime-full` | Experimental — additional substrate hooks for span-carry across `prime::spawn`, beyond what `serve-prime` already provides by default |

---

## Test + bench coverage

### Test sweeps (current main)

- **macOS default**: 514 lib + 18 e2e + 1 shutdown_barrier + 2 background_pool
- **Linux `--features io-uring`**: 510 lib + 17 e2e + 2 streaming + 1 upgrade + 1 shutdown_barrier
- **Linux `--features io-uring,tls`**: builds clean

### Integration tests

| File | What |
|---|---|
| `tests/e2e/end_to_end.rs` | Full app + listener + upstream + middlewares |
| `tests/e2e/listener_streaming_iouring.rs` | Chunked + 2 MiB CL upload via io_uring |
| `tests/e2e/listener_upgrade_iouring.rs` | CONNECT/Upgrade over io_uring hijacks + echoes |
| `tests/e2e/shutdown_barrier.rs` | Per-core resource drops on owning thread after drain |
| `tests/units/background_pool.rs` | CPU-bound work on rayon pool doesn't block chain dispatch |
| `tests/e2e/end_to_end.rs::listener_drains_in_flight_request_before_shutdown` | Drain timing |
| `tests/e2e/end_to_end.rs::quiesce_returns_503_during_window` | Quiesce window with configurable status |

### Property tests (in-module)

- `merged_edges_aggregates_concurrent_recorders` (`causality.rs`) — 8 threads × 64 edges → 512 edges merged, set equality, per-thread order preserved
- `explain_walks_chain_across_recorder_threads` (`causality.rs`) — cross-thread chain walks

### Benches

Hot path:
- `h1_dispatch`, `h1_streaming`, `h1_vs_hyper`, `h1_vs_pingora`
- `substrate_dispatch` (1-16 chain depths)
- `per_core_vs_arcswap` (3 primitives × 3 contention regimes)
- `swap_under_load`, `network_throughput`, `request_path`

Substrate primitives:
- `capture_drain`, `causal_record`, `causal_record_primitives`
- `tee_backpressure`, `tee_sink_primitives`
- `recording_sink_primitives`
- `histogram_record`, `simd_json_decode`, `perf_audit`

---

## What's NOT in the box yet

### Plan-prescribed (deferred or partial)

| Stage | What | State |
|---|---|---|
| 8 | Reproducible bench CI gates | benches exist, no CI pipeline |
| 10 | `LocalPipe` middleware fork (30-50 sites) | substrate primitive done; deferred-by-plan until DPDK |
| 11 | DPDK Runtime + L2/L3/L4 trait surface | months out |

### Bench-validated optimizations (parked with explicit trigger)

| Item | Bench-validated win | Trigger to land |
|---|---|---|
| Recording sink: SegQueue + writer task | 30-40× vs Mutex<File> | Recording becomes measured bottleneck |
| `SwappablePipe::delegate` per-core cache | ~10ns/dispatch (3% of substrate_dispatch) | Substrate_dispatch becomes the bottleneck |
| CausalIndex `SmallVec<[Edge; 4]>` inline | 15% on N=1 recordings | Allocator pressure shows up in profiles |
| Continuous-read serve loop on io_uring buffered | Cancel-on-disconnect parity | Serve-loop restructure planned |
| Per-core connection pool for io_uring upstream | Production load perf | Per-request connect is functionally correct |

### Production parity vs pingora (not bench-relevant, gap items)

- PROXY protocol v1+v2 (HAProxy / AWS NLB chain)
- Per-connection TLS session digest (cipher / SNI / ALPN / peer cert via typed accessor)
- Per-connection timing digest (handshake markers, first-byte-recv, response-write-start)
- Per-connection UniqueID (cheap — ~20 LoC AtomicU64 + RequestContext slot)
- MSG_PEEK on io_uring buffered path (blocked on tokio_uring 0.5 API)

### Strategic gap

- **Anchor application**: concrete `#[test]` demonstrating end-to-end security composition (capture session → swap cipher → replay → assert byte-identical causal chains). The substrate primitives are all built; the demo proves the moat. ~300-500 LoC + a tiny MCP harness. Tracked in `parking-lot.md`.

### Language bindings (planned, not started)

| Language | Binding | Effort |
|---|---|---|
| Python | PyO3 + maturin | ~600-1000 LoC |
| TypeScript | NAPI-RS (+ wasm later) | ~600-1000 LoC |
| Go | cgo over a C ABI | ~800-1200 LoC |

---

## Key files

- `parking-lot.md` — every deferred item with bench evidence + trigger condition
- `docs/bench_baselines.md` — current bench numbers
- `examples/` — config-only patterns + plugin skeleton
- `src/lib.rs` — public re-exports (the API surface)
