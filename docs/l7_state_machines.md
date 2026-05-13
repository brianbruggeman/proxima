# L7 state machines

Architecture + scope of proxima's HTTP/1, HTTP/2, HTTP/3 implementations.

## Principles

1. **Sans-IO state machines, owned I/O loop.** Each protocol's
   semantic state machine (framing, headers, streams, flow control)
   is independent of how bytes get to/from the wire. We feed bytes
   in and read bytes out; the socket / TLS / kernel-bypass layer
   below is swappable.

2. **Compile-time invalid state prevention.** Rust's type system
   represents each state as a distinct type (typestate pattern).
   `Connection<ReadingHead>` cannot call `begin_response` —
   compile error, not runtime check. Every state transition is a
   method that consumes one typestate and returns another.

3. **Per-protocol listeners, not unified.** `h1::Listener`,
   `h2::Listener`, `h3::Listener` share NOTHING in their hot path.
   The unifier is the `Pipe` trait above them (`impl Future`,
   monomorphized — not the boxed `tower::Service` hyper uses).

4. **Feature-gated at the Cargo level.** `h2`, `h3`, `trailers`,
   `upgrade`, `expect-100` are opt-in features. Unused features
   compile out entirely — no per-request cost.

5. **Per-listener config gating at runtime.** With features
   compiled in, the listener config (`HttpListenerSpec`) gates the
   feature paths. Hot path uses inline `if false { ... }` branches
   that the compiler dead-code-eliminates.

## Crate choices

| Layer | Crate / impl | Why |
|---|---|---|
| HTTP/1 parser | `httparse` | Sans-io, SIMD-validated, used by hyper. Battle-tested. No reason to reimplement. |
| HTTP/2 frames + HPACK | **`h2` (in-tree)** | Sans-io state machine + HPACK codec + connection driver, all in `src/h2/`. **Zero unsafe** on the protocol path; benchmarks 2-3× faster Huffman decode than the `h2` crate, ~50-60% higher RPS than hyper/pingora at 64 concurrent connections on the per-core runtime. The `h2` crate is retained as an optional dependency for the ALPN-multiplex listener (`http2` feature) until the native path becomes the ALPN default; it disappears entirely on `http1`-only builds. |
| HTTP/3 + QUIC | `h3` + `quinn-proto` | Both sans-io. We own the I/O event loop and integrate with the substrate's runtime. |
| TLS | `rustls` (sans-io) + `tokio-rustls` (today) | Same pattern: sans-io core, async wrapper. Switchable when the runtime changes. |

### Why a native HTTP/2 stack

The `h2` crate is excellent but pulls in ~25k lines and the `h2 + fnv
+ slab + futures-{core,sink,util} + http-body + tokio-util` dependency
fan. `h2` is the same protocol scope in ~6k lines (4× smaller)
with `bytes + smallvec + thiserror` as the only protocol deps.

The native stack shipped with measured wins on real-shaped workloads:

- HPACK Huffman decode: **2-3× faster** than the `h2` crate's decoder
  (zero-unsafe fixed-table implementation; LLVM elides bounds checks)
- HPACK encode / integer codec: matched or marginally faster
- End-to-end h2 single-stream: **24% faster than hyper, 28% faster
  than pingora** on default tokio
- With proxima's per-core runtime layered on:
  - conn=1 single-stream: **+58% RPS vs hyper, +59% vs pingora**
  - conn=64 multi-connection: **+52% vs hyper, +64% vs pingora**, with
    coefficient of variation 2.0% (2-3× tighter than competitors)

Full data: `rust/benches/RESULTS_linux.md`.

## State-machine surface area

Every item below has to be representable as a typestate with
compile-time invalid-state prevention.

### Request/response framing + lifecycle

- Status: **h1 done.** `Connection` transitions `ReadingHead →
  ReadingBody → AwaitingResponse → AfterResponse → ReadingHead (if
  keep-alive)`.
- Today these are runtime-checked via the `State` enum. Typestate
  refactor pending — see "compile-time invariants" below.

### Body streaming with substrate flow control

- Status: **partial.** Body decoder is a streaming state machine
  (`h1_body::BodyDecoder`); it surfaces chunks via a callback.
- `Connection` currently buffers the full body before
  `RequestReady` fires — Pipe::call gets the body upfront. This
  is wrong for streaming uploads / SSE-style ingestion.
- Pending: extend `Poll` to include `RequestHeadReady` (head parsed,
  body streaming) and `RequestBodyChunk(&[u8])`. The Pipe trait
  already supports streaming bodies via `Body` stream; the listener
  needs to start emitting chunks before End.
- Backpressure: the chunk-emission rate must respect the
  Pipe's body sink consumption rate (don't read from socket
  faster than the Pipe consumes). Today the Pipe's body sink
  is via `Body::from_bytes` — for true streaming the listener has to
  hold a channel between the read side and the body stream.

### Keep-alive / pipelining / half-closed connection handling

- Status: **partial.** Keep-alive works (RFC 7230 §6.3, version
  default + Connection header). Pipelining works for the simple
  request-after-request case (reset moves the cursor; overflow
  bytes in the same buffer are picked up by the next read).
- Half-close: not handled. If the client closes its write half
  after sending a request but keeps the read half open, we should
  send the response then close. Currently we'd return on the first
  EOF.
- Pipelined-while-writing: NOT supported. RFC 7230 §6.3.2 allows
  the client to send the next request while the previous response
  is being written. Our serial-loop architecture doesn't allow
  this; would require splitting read and write into separate tasks.
- Compile-time discipline: pipeline-state should be a typestate
  so we can't accidentally `begin_response` for request N+1 before
  finishing the response for request N.

### Expect / 100-continue semantics

- Status: **pending.** RFC 7231 §5.1.1. If the request head has
  `Expect: 100-continue`, the server should write a `100 Continue`
  status line BEFORE reading the body, indicating the client may
  proceed. If the server wants to reject before the body arrives, it
  writes the rejection status and closes.
- Implementation: `Connection::poll` returns a new `Expect100Continue
  { method, path, headers }` variant when the header is present. The
  listener decides: write `100 Continue` (continue reading body),
  write `4xx` (reject), or just ignore (defaults to continue per spec).
- Feature-gated: `expect-100` Cargo feature. With it disabled, the
  Expect header is ignored (matches "MAY ignore" per RFC).

### Upgrade (websocket, h2c)

- Status: **pending.** RFC 7230 §6.7. Request has
  `Connection: Upgrade` + `Upgrade: <protocol>` headers. Response
  is `101 Switching Protocols`, then the socket is handed off to
  the new protocol implementation.
- Implementation: `Connection::poll` returns
  `UpgradeRequested { protocol, sink }`. The listener writes the
  `101` response and hands the underlying socket to the new
  protocol's listener. Connection's state becomes terminal —
  we no longer drive it.
- Feature-gated: `upgrade` Cargo feature.

### CONNECT method semantics

- Status: **pending.** RFC 7231 §4.3.6. CONNECT establishes a
  tunnel through the proxy to the target. After the `200 OK`
  response, bytes flow bidirectionally between client and target
  without HTTP framing.
- Implementation: similar to Upgrade — connection enters a
  "tunnel" state where reads and writes pass through unmodified.
- Feature-gated: `connect` Cargo feature.

### Trailers

- Status: **pending.** RFC 7230 §4.1.2. Headers that arrive AFTER
  the chunked body's terminating `0\r\n` chunk. Used for fields
  whose value depends on the body content (digest, content-md5).
- Body decoder already skips them (parses but discards). Surfacing
  them through the Pipe::call API requires extending Response
  with a `trailers: HeaderList` field that the Pipe can populate.
- Feature-gated: `trailers` Cargo feature.

### Connection draining for graceful shutdown

- Status: **done.** `ShutdownPolicy::Drain { timeout }` /
  `Quiesce { duration, then }` on the listener. `ShutdownBarrier`
  coordinates per-core resource teardown LIFO.

### Integration with recording / causality / swap primitives

- Status: **automatic.** The substrate primitives are middleware on
  the `Pipe` trait — they operate on `(method, path, headers,
  body)` byte-stream tuples without caring how the request got there.
  Both the hyper-based listener (gone) and our `Connection`-driven
  listener route to the same `Pipe::call` — recording / replay /
  swap work either way.
- Edge case: when streaming bodies land, the recording sink has to
  see chunks as they arrive, not just the final buffered body.
  Already handled — `RecordingEvent::RequestChunk` is per-chunk.

## Compile-time invariants (typestate pattern)

Today: `Connection.state: State` enum, runtime `matches!` checks at
every accessor. Goal: each state is a distinct type, methods only
defined on the right type, invalid call = compile error.

Sketch:

```rust
pub struct Connection<S: ConnectionState> { ... _state: PhantomData<S> }

pub struct ReadingHead;
pub struct ReadingBody { body_start: usize }
pub struct AwaitingResponse;
pub struct Responding;
pub struct Closed;

impl Connection<ReadingHead> {
    pub fn poll(&mut self) -> Result<HeadPoll<'_>, ReadError>;
}

pub enum HeadPoll<'a> {
    NeedInput,
    HeadReady(Connection<AwaitingResponse>),  // consumed self
    BodyChunkInline(...),  // streaming body variant
}

impl Connection<AwaitingResponse> {
    pub fn method(&self) -> &[u8];
    pub fn path(&self) -> &[u8];
    pub fn header_value(&self, name: &[u8]) -> Option<&[u8]>;
    pub fn body(&self) -> &[u8];
    pub fn begin_response(self, ...) -> Connection<Responding>;
}

impl Connection<Responding> {
    pub fn write_chunk(&mut self, data: &[u8], out: &mut Vec<u8>);
    pub fn end_response(self) -> Connection<AfterResponse>;
}

impl Connection<AfterResponse> {
    pub fn keep_alive(&self) -> bool;
    pub fn reset_for_next_request(self) -> Connection<ReadingHead>;
    pub fn close(self) -> Connection<Closed>;
}
```

Cost: ~300 LoC of type plumbing, replaces ~50 LoC of runtime checks.
Benefit: misuse fails at compile time. Each state has only the
operations valid for that state.

Pending — not done in the initial Connection commit because the
shape was still settling. Now that the listener integration is
proven, refactor is the right move.

## Benchmark coverage requirement

Per user direction: bench everything that ships. Current coverage:

| What | Bench | Status |
|---|---|---|
| Substrate dispatch overhead | `substrate_dispatch.rs` | ✓ |
| Full request path | `request_path.rs` | ✓ |
| Hot-path microbenches | `perf_audit.rs` | ✓ |
| Per-core vs ArcSwap | `per_core_vs_arcswap.rs` | ✓ |
| H1 parse + connection | `h1_dispatch.rs` | ✓ |
| H1 connection vs hyper | `h1_vs_hyper.rs` | ✓ |
| TLS handshake | (pending) | — |
| SO_REUSEPORT accept rate | (pending) | — |
| Body streaming throughput | (pending — needs streaming bodies) | — |
| Recording cap throughput | (pending) | — |
| Swap latency | (pending) | — |
| H/2 round-trip (when shipped) | (future) | — |
| H/3 round-trip (when shipped) | (future) | — |

Each pending bench gets added before or with the feature it measures.
No shipping unmeasured perf claims.
