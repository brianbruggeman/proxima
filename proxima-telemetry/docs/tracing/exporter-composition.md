# Telemetry exporter = config composition (gap map)

An exporter is **not a type, it's a config point**. A vendor/endpoint is a point
in `auth × transport × protocol`, and integrations are TOML compositions of
compiled primitives (`proxima.notify` precedent:
`proxima-notify/src/lib.rs` — "Integrations are TOML compositions of existing
primitives: HttpUpstream, Transform, Validate, Retry, Isolate. No per-integration
Rust required."). The telemetry exporter is the same shape. conflaguration-first,
fluent fallback (guiding-principle 4); reuse before build (principle 1).

This is the gap map: the composition, each knob's compiled primitive, and whether
it EXISTS / PARTIAL / MISSING — grounded in an audit (2026-06-08), with citations.

## The pipeline

```
Recorder.drain ─▶ [batch] ─▶ [retry] ─▶ [auth] ─▶ [fanout|failover{ep…}] ─▶ [transport] ─▶ collector
                                                          └─ on permanent fail ─▶ [WAL/replay] (durable tail)

backpressure (already end-to-end, no new primitive):
   transport await  ─▶  drain call_pipe block_on  ─▶  ring fills  ─▶  producer-assist  ─▶  producer throttles
```

Every bracket is a `Pipe` and a config knob. The chain is assembled by the
existing composition mechanism (`.then()` / `PipeFactory`), lowered from config.

## Config shape (conflaguration-first, fluent fallback)

```toml
[telemetry.exporter]
codec   = "otlp_http"          # otlp_http | otlp_grpc | native
batch   = { max_records = 512, max_delay_ms = 200 }
retry   = { max_attempts = 5, base_delay_ms = 50, max_delay_ms = 5000 }
timeout_ms = 10000
fanout  = "failover"           # failover | broadcast
replay  = { wal = "/var/spool/telemetry-otlp", max_bytes = "1GiB" }   # durable tail

[[telemetry.exporter.endpoints]]
url = "https://otlp-us:4318"   auth = "prod_us"
[[telemetry.exporter.endpoints]]
url = "https://otlp-eu:4318"   auth = "prod_eu"

[credentials.prod_us]  kind = "bearer"  token = "env:OTLP_US_TOKEN"
[credentials.prod_eu]  kind = "mtls"    cert = "..."  key = "..."
```

Fluent fallback (when you assemble in code) composes the same primitives:
`exporter().codec(OtlpHttp).batch(512, 200ms).retry(5).timeout(10s).endpoints([...]).failover().replay(wal)`.

## Per-knob primitive map (EXISTS / PARTIAL / MISSING)

| knob | compiled primitive | status | citation |
|---|---|---|---|
| **codec** otlp_http | `OtlpHttpExporter` (prost, wire-verified) | EXISTS | `src/out/otlp_http/mod.rs` (byte-parity test vs OTel) |
| codec otlp_grpc | `OtlpGrpcExporter` (gRPC framing) | EXISTS | `src/out/otlp_grpc.rs` |
| codec native | `NativeExporter` (postcard, zero-alloc) | EXISTS | `src/out/native.rs` |
| **transport** send HTTP/1.1 | `H1ClientUpstream` (impl `Pipe`, prod-ready, benched vs hyper) | EXISTS (unwired) | `proxima-h1/src/client.rs:52` |
| transport HTTP/3 | `proxima_h3::native::Client` | PARTIAL (experimental; collectors are :4318 H1/H2) | `proxima-h3/src/native/client.rs:54` |
| **collector** (dogfood E2E) | `HttpListenProtocol::serve` | EXISTS | `proxima-listeners-http/src/lib.rs:40` |
| **chaining/lowering** | `.then()`/`Composable` + `PipeFactory` registry | EXISTS | `src/settings/chain.rs:103`, `proxima-pipe/src/pipe_factory.rs:24` |
| config-select (codec) | `ExporterChoice` `#[serde(tag="kind")]` + `pipe_from_choice` | EXISTS | `src/config.rs` |
| **retry** | `Retry<Inner>` (backoff, budget, predicates, Tee replay) | EXISTS | `proxima-middleware/src/retry.rs:101` |
| **timeout** | `proxima_time::timeout(dur, fut)` (runtime-agnostic, no_std-tiered, used in 9 crates); `Isolate` composes it | EXISTS | `proxima-time/src/lib.rs:135`, `proxima-graph/src/isolate.rs:88` |
| rate-limit | `RateLimit<Inner>` (token bucket) | EXISTS | `proxima-middleware/src/rate_limit.rs:97` |
| **fanout** (in-proc) | `Tee` / `SharedRingTee` (ArrayQueue, real backpressure) | EXISTS | `proxima-graph/src/tee/body.rs:20`, `tee/shared_ring.rs:17` |
| **failover** (ep A→B) | — | **SCOPED — punt general combinator (P1)** | exporter-local ordered-endpoint first-success loop, not a workspace primitive — see "Failover scoping" below |
| **batch** (N/T → one) | — | MISSING (drain batches at 512; no export-side N/T window) | — |
| circuit-breaker | — | MISSING (optional; `Retry` budget covers most of the need) | could extend `Isolate` |
| **transport TLS** (https server-cert) | `TlsStreamUpstream` (rustls, aws-lc-rs, webpki) ∘ `H1ClientUpstream` | EXISTS (prod) | `proxima-tls/src/connector.rs:42` |
| **auth** static bearer/header | `injected_request_headers` (template-expanded) | EXISTS-by-config | `proxima-h1/src/upstream.rs:356` |
| **auth** mTLS client-cert | an external crypto crate's `TlsConfig::with_client_auth_cert` → `ClientConfig` → `TlsStreamUpstream::new` | COMPOSE-EXISTING (wiring only, no new crypto) | `proxima-tls/src/connector.rs:120` |
| auth request-signing (Ed25519/HMAC) | an external crypto crate's signing / Blake3 mac | EXISTS (bonus option) | — |
| auth OAuth2 client-creds / refresh | — | **GENUINELY-NEW** (only real new auth work) | token endpoint + cache + refresh |
| **durable spool / replay** | `BinSink`/`BinSource` (WAL, offset-replay, crash-safe) | EXISTS (unwired; needs OTLP-bytes event wrapper) | `proxima-recording-core/src/binary/{sink,source}.rs` |
| **backpressure propagation** | async-await + `call_pipe` block_on + ring producer-assist | EXISTS (works; single drain blocks all → use `drain_range`/multi) | `src/recorder/drainer.rs:237`, ring producer-assist (this initiative) |

**Headline (revised after digging timeout + security): ~90% is existing
primitives.** timeout = `proxima_time::timeout` (solved); https/TLS = `proxima-tls`
(prod); mTLS = compose an external crypto crate (wiring, no new crypto); bearer = config;
request-signing = the external crypto crate. The genuinely-NEW list shrank to: **OAuth2-refresh
auth, a batch(N/T) combinator, the WIRING, and the drain-runtime decision.**
failover is scoped OUT as a general primitive (below); circuit-breaker is optional
(Retry's budget covers most of it).

## The one keystone: an async export runtime

timeout (`proxima_time::timeout`), transport-send (`H1ClientUpstream`), retry, and
backpressure ALL need exactly one thing: **the export side runs as an async task on
an executor**, not the current sync drain (`call_pipe` poll-once + `block_on` on a
plain thread with no reactor/timer driver — `drainer.rs:237`). proxima-time's driver
is std-thread-backed (no tokio needed) but still needs a *poller*; `H1ClientUpstream`
needs a reactor. **Prime-first** (tokio is the escape-hatch): the reactor is the
**prime reactor** (`prime::os::core_shard`) — `PrimeTcpUpstream` is the production
transport, proven over the prime wire in `tests/otlp_send_e2e.rs`. So the
drain-runtime decision is the single choice that unblocks timeout + transport +
retry + end-to-end backpressure together. Options (decide in slice 3b): (a) the
managed-drainer thread runs a **prime core-shard** and the drain `await`s the export
chain; (b) N `drain_range` async drainers on prime; (c) a reactor-backed transport
the sync drain can drive. (a) is cleanest and makes every async knob compose for
free. tokio only via the documented `runtime = "tokio"` escape-hatch.

## Failover scoping (P1: don't add a type unless justified)

Failover is a *pattern*, not a clean reusable primitive. A general `Failover` Pipe
would have to own a lot of app-specific policy: which errors trip failover, how long
to mark an endpoint down, probe/recovery interval, sticky vs round-robin, health
state. Building that generality now is speculative (P1: "when in doubt, don't add
the type; the right answer might be the caller composes"). For the exporter,
failover IS just: iterate the configured `endpoints` in order, send through each
endpoint's sub-chain (`auth ∘ timeout ∘ retry ∘ transport`), stop at first success;
on all-fail → durable replay. That's exporter config (the ordered `endpoints` list)
+ a ~10-line first-success loop in the exporter's send path — NOT a workspace
combinator. **Decision: punt the general `Failover` combinator; express failover as
exporter-local ordered-endpoint policy.** If a second consumer ever needs
failover-as-a-Pipe, extract it then (P1: wait for the second consumer).

## Backpressure model (decided — not contested, no tournament)

The `Pipe` trait (`proxima-pipe/src/pipe.rs:267`) has **no `poll_ready`**;
backpressure is "the call future awaits." That composes end-to-end for free:

1. transport `H1ClientUpstream::call().await` is pending while the socket/collector
   is not ready;
2. the drain's `call_pipe` (`drainer.rs:237`) drives it (poll-once, else
   `block_on`) — so a slow transport stalls the drain pass;
3. the ring fills; under `Block`, **elastic producer-assist** (this initiative)
   makes the producer throttle to real downstream throughput.

So the end-to-end backpressure the whole thread asked about is the *existing*
async chain meeting the ring work already landed. No readiness/credit primitive to
invent (P13: no `/algorithm-development` or `/algorithm-rigor` — the design is RISC
reuse + config lowering, mechanical given the primitives; recorded here instead of
performing a ceremony tournament).

**Known caveat (carry into the DAG):** `call_pipe`'s `block_on` stalls the *whole*
drain pass on one slow endpoint. Fix is the already-built `drain_range`
partitioning (N drainers) and/or driving the transport on an async runtime — a
real sub-decision (drain-runtime model) sequenced in the DAG, not hand-waved.

## Build DAG (each slice fully proven before the next — P16)

1. **Exporter config surface** — `ExporterChoice` → an `ExporterPipeline` config
   (codec, batch, retry, timeout, endpoints[+auth], fanout/failover, replay):
   conflaguration + serde + fluent + round-trip + validation tests. Config-first
   half of P4. *Proof:* round-trip + validate tests. **Must lower to real pipes in
   slice 2 — config alone is scaffolding (P15), so 1+2 land together.**
2. **Composition lowering** — config → a real `Pipe` chain via existing combinators
   (`Retry`∘`Isolate`∘codec) terminating in a pluggable terminal. *Proof:*
   in-process fault-injecting terminal; assert retry/timeout actually engage. No
   network needed — proves the thesis.
3. **Transport-send terminal** — `OtlpHttpExporter` ∘ `H1ClientUpstream` → endpoint;
   drain-runtime model decided. *Proof:* E2E vs a dogfood collector
   (`HttpListenProtocol::serve` on 127.0.0.1): emit N, assert N received +
   `dropped()==0`. This is the real OTLP-over-wire + the grade vs OTel.
4. **endpoint policy** (ordered first-success "failover") — exporter-local, lands
   with the endpoints config (slices 1+2) over each endpoint's slice-3 sub-chain;
   NOT a standalone combinator (scoped out above). *Proof:* endpoint A errors → B
   receives; all-fail → replay (slice 7).
5. **batch** combinator (N/T window) — the one genuinely-new combinator. *Proof:*
   timing + count.
6. **auth** per-endpoint — mostly compose-existing: bearer/header (config today),
   https/TLS (`proxima-tls`), mTLS (compose the external crypto crate's ClientConfig →
   `TlsStreamUpstream`); only OAuth2-refresh is new. *Proof:* dogfood collector
   asserts the header / client-cert.
7. **durable replay** — `BinSink` spool on permanent-fail, replay on recovery.
   *Proof:* kill collector, emit, restart, assert replayed (lossless dead-sink —
   closes the axis flagged unsolved earlier).
8. **grade vs incumbent** (P14) — writes×reads×shape vs OTel `BatchSpanProcessor`
   (drops) + tracing-fmt (blocks), same per-record cost, measuring loss/throughput/
   latency. The reads axis (N drainers) is the edge OTel structurally lacks.

## No new primitives — full decomposition

Even the "genuinely-new" items decompose into existing primitives + thin glue. The
whole production exporter is assembly, not invention (the proxima thesis):

| capability | composes from (existing primitives) | the only new code |
|---|---|---|
| composition/wiring | `.then()` / `PipeFactory` / `ExporterChoice` | config→chain lowering (glue) |
| drain-runtime host | **prime** `core_shard` reactor (tokio = escape-hatch) | choose + host the export task (wiring) |
| codec | `OtlpHttpExporter` / `NativeExporter` | — |
| transport http/1.1 | `H1ClientUpstream` | — |
| TLS https | `proxima-tls TlsStreamUpstream` | — |
| mTLS | the external crypto crate's ClientConfig → `TlsStreamUpstream::new` | cert/key config wiring |
| bearer/header | `injected_request_headers` (templates) | config |
| request-signing | the external crypto crate's Ed25519 / Blake3 | — |
| timeout | `proxima_time::timeout` / `Isolate` | — |
| retry | `proxima-middleware Retry` | — |
| rate-limit | `proxima-middleware RateLimit` | — |
| **batch(N/T)** | drain `batch_size` (N) + `IntervalPipe` (T window) | interval-flush wiring (no primitive) |
| **failover** | ordered `endpoints` config + first-success loop | ~10-line exporter loop |
| **OAuth2-refresh** | `H1ClientUpstream` (token POST) + `serde_json` (parse) + `proxima_time` (expiry/refresh) + `injected_request_headers` (attach) | thin token cache+refresh glue (no primitive) |
| durable replay | `BinSink`/`BinSource` (WAL) | OTLP-bytes event wrapper (wiring) |
| backpressure | async-await + drain + ring producer-assist | already done |
| collector (test) | prime `os::net::TcpListener` (proven) / `HttpListenProtocol::serve` | test harness |

**Zero new foundational primitives.** Remaining work = (1) the drain-runtime host
choice + wiring, (2) config→pipe-chain lowering, (3) thin orchestration glue
(exporter first-success loop, OAuth refresh cache, batch interval trigger, BinSink
event wrapper) + tests. "No new primitives" ≠ "no code" — there is real assembly,
glue, and proof to land — but nothing foundational is missing.

## Where the ring work fits

The lossless ring + producer-assist + MPMC + `assisted()` this initiative landed is
the **head** of this pipeline (slices already done, proven). This gap map is the
**body+tail**. The two meet at the backpressure chain above.
