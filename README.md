# proxima

> ## WORK IN PROGRESS — EXPERIMENTAL
>
> **proxima is under heavy active development: not stable, not production-ready.**
> APIs, behavior, semantics, and internals change without notice or a migration
> path, and whole subsystems may be reworked or removed. Do not depend on it in
> production. Use at your own risk.

**A sans-IO core where everything is a `Pipe`, and big things are small
things composed.** One trait — async `In -> Result<Out, Err>` — and four roles
(source, sink, transform, observe) express every part. A proxy, a gateway, a cache,
a load balancer, a rate limiter, a codec, a telemetry pipeline: each is a handful
of pipes stacked. A production gateway is roughly `Auth -> Route -> RateLimit -> Forward`.

Because protocol logic and composition are separated from I/O, the same pipe runs
behind the from-scratch **Prime** per-core runtime (the default), a hosted **Tokio**
backend, config-driven services, and sans-IO protocol crates. `no_std`, wasm, and
the kernel-bypass floor (DPDK/pmem, plus a custom userspace NVMe queue-pair
engine — no SPDK dependency) are frontier surfaces, not the start-here
path. Config-driven throughout — one spec drives the library, the CLI, and an MCP
(Model Context Protocol) control plane; dependencies can be faked, recorded, or replayed.

## learn proxima

Start with the **example curriculum** — a scope-and-sequence where each rung
teaches one piece, and the whole algebra bottoms out at eight of them: **one
form** worn four ways (transform `In -> Out`, source `() -> Out`, sink
`In -> ()`, observe `In -> In`), **one chain** that joins them (`and_then`), and
**three primitives** built on the pipe (filter, fan-out, fan-in). Everything
else — retries, gates, proxies, a gateway — is those eight composed. Read it top
to bottom.

- **[`examples/`](examples/README.md)** — one runnable `examples/<name>/` per
  rung (README + `main.rs`). Start at `hello`, follow the **Builds on** links.
- **[`docs/tutorials/00-foundations.md`](docs/tutorials/00-foundations.md)** —
  the same path in prose, from zero: what a pipe is, its four shapes, why
  there are four related `Pipe` traits, and how to serve one.
- **the book's algebra** — [`book/src/algebra/index.md`](book/src/algebra/index.md)
  is the map (form, chain, primitive, pattern — four layers, each built on the
  one before); [`patterns.md`](book/src/algebra/patterns.md) is the pattern
  gallery. `mdbook serve book` renders the curriculum from the verified
  example source (chapters embed the real files, so prose and code never
  drift).

### hello, running

The smallest proxima service is one pipe mounted behind a listener. This is
`examples/hello/main.rs` verbatim, trimmed to the two functions that matter
(the file also has the doc comments explaining each line and a client
round-trip proving the response crossed a real socket — see
[`examples/hello/`](examples/hello/main.rs) for all of it):

```rust
#[proxima::instrument]
async fn hello(_request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    Ok(Response::ok("hello, proxima\n"))
}

#[proxima::main]
async fn main() -> Result<(), ProximaError> {
    let bind = SocketAddr::from((Ipv4Addr::LOCALHOST, 8080));

    let app = App::new()?;
    app.mount("/", hello)?;

    // `serve` spawns the listener and returns once it is actually accepting —
    // no polling, no sleeping, no discovering ECONNREFUSED the hard way.
    let server = app.serve(RunConfig::http(bind)).await?;
    println!("listening on http://{bind}");

    server.run_until_signal().await;
    Ok(())
}
```

`hello` is just an `async fn` — no attribute makes it a pipe. `App::mount`
takes a bare `async fn(Request<Bytes>) -> Result<Response<Bytes>, ProximaError>`
directly (via `IntoMountTarget`), so `app.mount("/", hello)` compiles with no
wrapper type and no manual boxing. `#[proxima::instrument]` is orthogonal: it
wraps the handler in a span so every call is traced — one attribute yields
trace + metric + log. Reach for `#[proxima::piped]` only when you want a
*named*, reusable pipe type instead of a one-off handler.
`#[proxima::main]` boots the runtime that `App::new()` then adopts; `app.serve(...)`
spawns the listener and returns only once it is actually accepting.

```bash
cargo run --example hello --features http1-native
```

```
listening on http://127.0.0.1:8080
```

`server.run_until_signal().await` then blocks until SIGINT/SIGTERM, stops
accepting, and lets in-flight requests finish before the process exits —
the whole shutdown story in one line.

Then keep going:

```bash
cargo run --example transform   # one pipe, and its four roles
cargo run --example gateway     # proxy + auth + route + rate-limit
```

## the model

Everything is a `Pipe`: an async `In -> Result<Out, Err>`. Choose the types and
the same trait becomes a **source** (`() -> Out`), a **sink** (`In -> ()`), a
**transform** (`In -> Out`), or an **observe** (`In -> In`). Join them with
`and_then`, reach for the three primitives that wrap a pipe — filter, fan-out,
fan-in — and the bigger thing falls out. A gate, a retry, a proxy: each is
those composed, not another thing to learn. A pipe mounted behind a
**listener** answers requests; an **app** binds listeners and runs them.

A "cache" is just a multi-upstream pipe where one upstream is kv-typed and the
selection is `fallthrough`. A "load balancer" is the same shape with a different
selection. A "mock" has one synthetic upstream. A "TCP proxy" is a stream
listener over a passthrough pipe. **One core, every pattern.**

Config and code are isomorphic: any setup expressible in TOML is expressible via
the fluent builder, and the two round-trip through serde. Recording, replay,
hot-swap, and observability all key off the same named entries either way.

```toml
# proxima.toml — a cache in front of an origin
[upstreams.cache]
type = "kv"
kv   = "cache"
ttl  = "1h"

[upstreams.origin]
type = "http"
url  = "https://api.example.com"

[pipes.cached]
mount = "/{*path}"
chain = ["cache", "origin"]
```

The equivalent fluent-builder and CLI forms are in
[`examples/`](examples/README.md) — the examples are the source of truth for
current API.

## runtime model

A pipe is sans-IO, so it doesn't care which executor drives it. The `Runtime`
trait abstracts spawn / timer / blocking-pool, and more than one runtime can run
in one process:

- **Prime** (default) — a from-scratch, tokio-free per-core runtime: one thread
  per core, no work-stealing, no locks in core code (hot-path reads are
  thread-local or lock-free via `ArcSwap`).
- **Tokio** (`runtime-tokio`) — N pinned current-thread runtimes, one per core;
  the hosted backend and perf baseline. `io-uring` on Linux.
- **Kernel-bypass floor** — DPDK, pmem, and a custom userspace NVMe queue-pair
  engine (no SPDK dependency) drive the same sans-IO pipes with no OS on the
  hot path. `no_std` + alloc is the price of admission to that floor.

Four tiers of the same contract keep one pipe portable across all of them.
`Pipe` is the root and the one to reach for: `!Send`, so a per-core worker can
hold `Rc`/`RefCell`/a SQLite or GPU handle across an await. `SendPipe` crosses
a core, `UnpinPipe` returns a future you can poll in place, and
`UnpinSendPipe` does both — each is a cost you name, never one inferred for
you. They are four standalone traits and not a hierarchy for one reason:
`impl<P: Pipe + Send> SendPipe for P` needs a bound on an RPITIT's returned
future, which is return-type notation, still unstable (rust#109417). When it
lands, all three collapse back into `Pipe` plus a bound at the use site and
get deleted. `docs/tutorials/00-foundations.md` walks the ladder.

## the crate map

35 crates, in seven groups. Each group only depends on the ones above it, so you
can read down the list and never meet a name you have not been given yet. The
one thing to hold onto: `Pipe` lives in `proxima-primitives`, and every other
crate is either something it stands on, something shaped out of it, or somewhere
to plug it in.

**primitives** — the algebra and what it stands on.

| crate | what it is |
|---|---|
| `proxima-core` | error, ring, arena, time, markers. Depends on no other proxima crate — the root of the tree. |
| `proxima-primitives` | **`Pipe`** and its three tier siblings (`SendPipe`, `UnpinPipe`, `UnpinSendPipe`), `Handler`, `Body`, `HeaderList`, the sans-IO byte-stream traits, runtime-agnostic concurrency. |
| `proxima-runtime` | the `Runtime` trait: per-core spawn, cross-core dispatch, background pool. |
| `proxima-clock` | time as a pipe — tick sources, wall-clock anchoring, a coarse shared-time cell. |
| `proxima-telemetry` | traces, metrics, logs — proxima's own surface, not `tracing`. |
| `proxima-macros` | `#[proxima::piped]`, `#[proxima::instrument]`, `#[proxima::main]`, `#[proxima::test]`. |
| `proxima-config` | format registry (JSON, JSON5, TOML, YAML, RON, XML), typed schema IR, desired-state store. |
| `proxima-codec` | body marshalling registry — JSON and bytes-passthrough by default. |
| `proxima-build` | `build.rs` library: resolve a typed profile, emit consts and cfgs for the crate being built. |
| `proxima-test` | the `#[proxima::test]` harness — prime and tokio drivers, panic capture, cassettes. |

**patterns** — pipes wired into a shape. No wire of their own.

| crate | what it is |
|---|---|
| `proxima-patterns` | one feature per pattern: `alert`, `balancer`, `middleware`, `control_plane`, `kv`. |
| `proxima-auth` | sans-IO auth cores — token lifecycle, SigV4, RFC 7616 Digest, SPNEGO. Depends on no proxima crate: an FSM you put pipes around, not one that holds them. |
| `proxima-recording` | event model, binary and JSONL formats, capture, causal-order replay. |

**protocols** — sans-IO. Bytes in, frames out; never touches a socket.

| crate | what it is |
|---|---|
| `proxima-protocols` | 24 feature-gated codecs: HTTP/1, HTTP/2 + HPACK, HTTP/3, QUIC, TCP/IP, WebSocket, redis, pgwire, DNS, MQTT, AMQP, Kafka, memcached, gRPC framing, protobuf wire, JSON-RPC, PROXY protocol, NVMe. |
| `proxima-centauri` | IKE-style key agreement, rekey, stateless cookies, and an ESP child SA with AEAD + anti-replay. No-alloc; entropy and time are pipe *inputs*, never held capabilities. |

**stacks** — a codec plus a listener plus a client, so `proxima::Client` and
`proxima::Listener` speak the wire. Each of the protocol stacks splits into a
`client` and a `listen` feature, and `--no-default-features` leaves the bare
codec.

| crate | what it is |
|---|---|
| `proxima-listen` | the listener registry: `ListenProtocol`, `ListenRegistry`, `ServeContext`, and the `.any()` preface classifier. |
| `proxima-http` | HTTP/1.1, HTTP/2, HTTP/3, WebSocket. |
| `proxima-quic` | runtime-agnostic QUIC I/O facade over `proxima-protocols::quic`. |
| `proxima-tls` | `TlsConfig`, rustls integration, SNI, self-signed dev certs. |
| `proxima-redis` `proxima-pgwire` `proxima-dns` `proxima-kafka` `proxima-mqtt` `proxima-amqp` `proxima-memcached` | one per wire. |

**backends, as features** — a backend is a feature flag, never a second API.

| crate | what it is |
|---|---|
| `prime` | the from-scratch per-core runtime: one thread per core, no work-stealing, bounded SPSC inbox, reactor + executor + timer + background pool. |
| `proxima-net` | UDP `PacketListener` and addressing, plus every platform backend — `prime`, `tokio`, `wasm`, `dpdk`, `xdp` — as a feature-gated module. |

**storage & process** — where bytes land, where processes run.

| crate | what it is |
|---|---|
| `proxima-storage` | the NVMe queue-pair engine (`nvme`), the pmem crash-consistency leaf (unconditional, `no_std` + no-alloc), the DAX mmap facade (`dax`). |
| `proxima-process` | capability-typed commands, PTY and FD pipes, libc shim dispatch, host grounds. |

**apps** — things you run.

| crate | what it is |
|---|---|
| `proxima` | the umbrella library, and `proximad`. |
| `proxima-cli` | the `proxima` binary: call, serve, describe, daemon control, hot-swap, replay. |
| `proxima-intercept` | TLS-terminating CONNECT proxy with per-host cert generation. |
| `rekt` | the `rek` binary — a load tester built on proxima. |
| `proxima-vm` | a VM-backed `Pipe` proof surface over KVM or Hypervisor.framework. |

## more

- **feature flags** — default serves on Prime with HTTP/1–3; `runtime-tokio`,
  `tls`, `websocket`, `quic`, `unix`, `io-uring` are opt-in.
  `--no-default-features` compiles the trait + protocol layers with no listener
  impls.
- **hot-swap** — `proxima apply <name> --spec <file>` swaps a running pipe;
  in-flight requests finish on the old impl, new ones hit the new.
- **pipeline runner** — `proximad` runs declarative pipelines (DAGs of
  child-process stages), records every event, and replays / mutates / explains
  them over UDS, SSH-stdio, or MCP JSON-RPC. See
  [`scenarios/remote_pipeline_demo/`](scenarios/remote_pipeline_demo/README.md).
- **benchmarks** — the in-tree sans-IO HTTP/2 stack beats hyper and pingora on
  RPS and tail variance; Prime-vs-Tokio and the full numbers live in
  [`benches/`](benches/).

## license

Dual-licensed under either MIT or Apache-2.0, at your option.
