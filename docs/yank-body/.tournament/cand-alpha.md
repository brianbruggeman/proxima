# Streaming contract design — Spec A (incumbent)

Status: incumbent spec, pre-tournament. This is the input to
`/spec-rigor`. It will be critiqued, an independent Spec B authored, the
two synthesized, and a Borda panel will judge. The winner becomes the
binding streaming contract; the result lands in `edges.md`.

Scope: the spine-vs-streaming boundary ONLY. The `Typed -> TypedPipe`
move and the `Body -> Bytes` collapse are mechanical once this boundary
is nailed (per the initiative brief). This spec must answer, with worked
examples: buffered-vs-streaming, where the Send/!Send stream lives, how
`StreamingPipe` composes with the bytes spine, and how proxy
pass-through / SSE / PTY / websocket each map onto it.

---

## 0. Vocabulary

- **Spine** — the central abstraction everything composes through:
  `Pipe`, `Request`, `Response`. After this initiative the spine is
  **bytes-only**: `Request`/`Response` carry `Bytes` (via a thin
  `Payload(Bytes)` newtype), and `Pipe` is effectively `Bytes -> Bytes`.
  No `Body` enum.
- **Edge** — a static-dispatch layer ON TOP of the spine that adds
  typing (`TypedPipe<In,Out>`) or streaming (`StreamingPipe`). Edges are
  where the 4-arm `Body` union is replaced.
- **Buffered** — the whole request/reply fits in one `Bytes` and is
  handed across the spine in one shot. The 80% case (per the
  framed-listener finding: the consumer's framed RPC, every middleware,
  every proxy that doesn't need incremental delivery).
- **Streaming** — the body is delivered as an ordered sequence of
  `Bytes` chunks over time, possibly unbounded (SSE, PTY output, WS
  frames, chunked proxy pass-through). Today this is `BodyInner::Stream`.

---

## 1. Axioms

**A1 — Dyn dispatch forces exactly one erased payload type; make it
`Bytes`.** `PipeHandle = Arc<dyn DynPipe>` exists; erasure is
unavoidable at that boundary. The erased payload is `Bytes` (wrapped as
`Payload(Bytes)`), never a union. A union in the erased type is the
disease; a single concrete byte buffer is the cure.

**A2 — The spine pays zero streaming tax.** Buffered RPC — the 80% case
— must not allocate a `Pin<Box<dyn Stream>>`, must not carry a
cancellation token tree, must not carry a trailers `Arc<Mutex>`. Those
costs exist only on the explicit streaming edge, paid only by the ~27
sites that actually stream.

**A3 — Typing is static-dispatch at the edge.** `TypedPipe<In,Out>` is
generic and monomorphized; it decodes `Bytes -> In` at ingress and
encodes `Out -> Bytes` at egress via a `proxima-codec` `MessageCodec`.
The telemetry zero-copy `Arc<record>` fan-out is preserved by making the
in-process telemetry path a `TypedPipe` whose `In`/`Out` ARE the
`Arc<record>` types — it never round-trips through `Bytes` in-process
(see §4).

**A4 — Streaming is an explicit, separate contract — not a spine
variant.** `StreamingPipe` is its own trait. It is the ONLY place the
Send-vs-!Send stream question is decided, so the decision is *contained*
rather than smeared across every buffered RPC (which is what
`BodyInner::Stream` did).

**A5 — Send is the only genuine trait split; std/alloc are `#[cfg]` on
one trait; minimize `Box<dyn>` boundaries.** (Inherited axis discipline.)
The streaming contract must not introduce a second orthogonal split that
multiplies into 2^N traits.

**A6 — `ThreadLocalPipe` coherence is real and must hold.** The
`impl<T: Pipe> ThreadLocalPipe for T` blanket was removed (orphan
conflict). The streaming contract therefore mirrors the *parallel-impl*
reality, not a blanket: a Send streaming trait and a `!Send` streaming
trait are declared in parallel, exactly as `Pipe`/`ThreadLocalPipe` are.

---

## 2. The bytes spine (settled — restated for completeness)

```rust
// proxima-pipe/src/payload.rs (new, ~replaces body.rs husk)
/// The erased spine payload. A buffered byte buffer; zero-copy clone
/// (Bytes refcount bump). No union, no stream, no cancel, no trailers.
#[derive(Clone, Debug, Default)]
pub struct Payload(pub Bytes);

// Request/Response carry Payload (== Bytes) instead of Body.
pub struct Request  { /* head ... */ pub payload: Payload }
pub struct Response { /* head ... */ pub payload: Payload }

// Pipe is unchanged in shape; its body is now bytes.
pub trait Pipe: Send + Sync + 'static {
    fn call(&self, request: Request)
        -> impl Future<Output = Result<Response, ProximaError>> + Send;
    fn name(&self) -> &str { "anonymous" }
    fn background_tasks(&self) -> Vec<BackgroundTask> { Vec::new() }
}
// ThreadLocalPipe sibling unchanged (no Send), parallel-impl per A6.
```

Migration mechanic (strangler-fig): first make `Body` a `Bytes` newtype
alias so the hundreds of `from_bytes`/`payload` pass-through sites are
mechanically unaffected; then inline `Payload` and delete the husk. The
buffered methods (`from_bytes`, the `Bytes` accessor) survive as
inherent methods on `Payload`.

**Cancel + trailers do NOT live on the spine.** They were
`Body::{cancel, trailers}` and are streaming-completion concerns. They
move onto the streaming edge (§3): a streamed response carries its own
cancel token and trailers slot; a buffered `Payload` carries neither.

---

## 3. The streaming edge — `StreamingPipe` (THE contested design)

### 3.1 Chunk contract

A stream is an ordered sequence of `Result<Bytes, ProximaError>` chunks,
terminated by `None`, optionally followed by trailers. This is a
sans-IO-shaped pull contract (poll-based), NOT a `Pin<Box<dyn Stream>>`
baked into the spine:

```rust
/// One streamed body, pulled chunk-by-chunk. Send variant (mainstream).
pub trait ByteStream: Send + 'static {
    /// Poll the next chunk. `Ready(None)` = end of body.
    fn poll_chunk(self: Pin<&mut Self>, cx: &mut Context<'_>)
        -> Poll<Option<Result<Bytes, ProximaError>>>;
    /// Trailers, available only AFTER poll_chunk returned Ready(None).
    fn trailers(&self) -> Option<HeaderList> { None }
}

/// !Send sibling for per-core / Prime streaming (Rc/RefCell chunk
/// sources). Declared in parallel per A6 — NOT a blanket.
pub trait LocalByteStream: 'static {
    fn poll_chunk(self: Pin<&mut Self>, cx: &mut Context<'_>)
        -> Poll<Option<Result<Bytes, ProximaError>>>;
    fn trailers(&self) -> Option<HeaderList> { None }
}
```

A blanket `impl<S: ByteStream> LocalByteStream for S` is the natural
"Send is a strict superset" wiring — but A6 warns it may hit the same
orphan conflict the `Pipe` blanket did. **Spec A's position:** declare
both in parallel and provide a free adapter `fn local(self) ->
impl LocalByteStream` on `ByteStream` instead of a blanket, sidestepping
coherence. (This is a primary critique target — see §6.)

### 3.2 The streaming Pipe

`StreamingPipe` is the explicit streaming counterpart to `Pipe`. It
takes a (possibly streamed) request body and returns a streamed
response body. It is a SEPARATE trait; a `Pipe` is the buffered case and
does not implement it.

```rust
pub trait StreamingPipe: Send + Sync + 'static {
    type RespStream: ByteStream;
    fn call_streaming(&self, head: RequestHead, body: RequestBody)
        -> impl Future<Output = Result<(ResponseHead, Self::RespStream), ProximaError>> + Send;
}
// RequestBody is an enum { Buffered(Bytes), Streamed(Box<dyn ByteStream>) }
// — the ONE Box<dyn> boundary, paid only on the streaming edge (A5).
```

Per A4 the Send/!Send decision is contained to this trait family:
`StreamingPipe`/`ByteStream` (Send, mainstream) and
`LocalStreamingPipe`/`LocalByteStream` (!Send, per-core), declared in
parallel.

### 3.3 Bridging buffered <-> streaming on the spine

The spine is `Pipe` (buffered). A `StreamingPipe` is adapted INTO the
spine at the I/O facade by a `BufferStreaming` adapter that `collect()`s
the response stream into one `Payload` when a buffered consumer asks —
and, conversely, a buffered `Pipe` is lifted to a degenerate
single-chunk `StreamingPipe` by `StreamOnce`. The two adapters are the
only glue; middleware that genuinely needs incremental delivery (Tee,
SSE) speaks `StreamingPipe` directly.

**Spec A's position on Send v1:** every concrete streaming site today is
Send (re-measured: 27/27). So v1 ships `StreamingPipe`/`ByteStream`
(Send) as the *only wired* streaming path; `Local*` traits are declared
(so the contract is complete and the per-core extension is not a future
trait split) but have zero in-tree consumers until a Prime per-core
streaming need is *measured*. This is NOT a punt under principle 15: the
contract is complete and total; the `Local*` path has no consumer
because no consumer exists, and adding one is "implement the parallel
trait," not "redesign the contract." (Contrast option-b parametric-Send,
which Spec B will argue — see §6.)

---

## 4. The typed edge — `TypedPipe<In,Out>` (preserving Arc fan-out)

```rust
pub trait TypedPipe<In, Out>: Send + Sync + 'static
where In: Send + 'static, Out: Send + 'static {
    fn call_typed(&self, input: In)
        -> impl Future<Output = Result<Out, ProximaError>> + Send;
}
```

Two ways a `TypedPipe` meets the bytes spine:

1. **Codec edge (cross-process / wire):** a `CodecAdapter<C: MessageCodec,
   P: TypedPipe<C::Input, C::Output>>` decodes `Payload -> In` at ingress
   and encodes `Out -> Payload` at egress. This replaces `Body::typed`
   for any payload that must cross the wire.

2. **In-process typed dispatch (telemetry — the perf-critical case):**
   the telemetry record pipes become `TypedPipe<Arc<Record>, ...>` (or
   `TypedPipe<Vec<Arc<Record>>, ...>`). The drainer builds `Vec<Arc<T>>`
   ONCE and calls `call_typed` directly — no `Bytes`, no codec, no
   downcast. The Arc fan-out (consumers borrowing each Arc) is preserved
   verbatim because `In = Vec<Arc<T>>` is passed by value/ref through
   static dispatch, never erased to `Arc<dyn Any>`. The `as_typed`
   `downcast_ref` (one TypeId compare) is *eliminated*, not replaced —
   the type is known at the call site.

This is strictly faster than `Body::typed` (no `Arc<dyn Any>` box, no
downcast) AND preserves zero-copy fan-out. A bench arm proves parity-or-
better against the current `Body::typed` drainer path (principle 14;
gate point 13 home-turf arm = the telemetry drainer at realistic batch
size).

---

## 5. Worked examples

### 5.1 Buffered RPC (framed JSON from a downstream consumer — the 80% case)

```
frame bytes --[listener]--> Request{payload: Payload(bytes)}
  --> Pipe::call --> Response{payload: Payload(bytes)} --> one frame
```

Zero streaming machinery. No Box, no cancel, no trailers, no stream
poll. This is the framed-listener path, now spine-native. `TypedPipe` +
`JsonCodec` decode/encode at the edge if the Pipe wants typed access.

### 5.2 SSE (HTTP -> server-sent events; agui-http-stream)

```
upstream Response stream (ByteStream of chunks)
  --[HttpStreamToAgUi: StreamingPipe]--> ByteStream that maps each
     inner chunk to an SSE `data: ...\n\n` Bytes frame
  --[listener]--> writes each chunk to the socket as it arrives
```

`HttpStreamToAgUi` is a `StreamingPipe`: its `RespStream` is a
chunk-mapping `ByteStream`. No buffering; backpressure flows through
`poll_chunk`. Today this is `Body::from_stream(async_stream::stream!)`;
post-migration it is an explicit `ByteStream` impl (enum-state, no
async_stream macro needed).

### 5.3 PTY (process master output; pty_pipe)

```
PTY master fd --[reactor readiness]--> ByteStream::poll_chunk reads
  available bytes --> StreamingPipe response stream
  + size-follower / child-cleanup ride as ByteStream state
```

Today: `Body::from_stream(stream::unfold((stream, cancel, dispatched)))`.
Post-migration: a `PtyByteStream` struct holding the fd reader + cancel
token + dispatched-child guard, implementing `ByteStream`. The cancel
token that was `Body::cancel` is now a field of the stream (§2). Genuine
incremental I/O; this is the `Send` mainstream today and would be the
first candidate for `LocalByteStream` IF Prime pins PTY to a core with
`!Send` fd state — but that need is not yet measured (§3.3).

### 5.4 WebSocket (frame relay; pipe-agui ws)

```
upstream ByteStream of payloads
  --[WsAdapter]--> writes each payload chunk as a WS data frame on the
     upgraded socket
inbound WS frames --> ByteStream of payloads --> Request streamed body
```

`WsAdapter` is a `StreamingPipe` on the response side and feeds a
`RequestBody::Streamed` on the inbound side. The upgrade handshake is a
buffered `Pipe` call (101 response with empty `Payload`); the post-
upgrade frame relay is `StreamingPipe`. Clean split: handshake = spine,
relay = streaming edge.

### 5.5 Proxy pass-through (chunked; stream_passthrough)

```
upstream read-half --[reader_to_byte_stream]--> ByteStream
  --[StreamPassthrough: StreamingPipe]--> response ByteStream
  (no transform, no buffering — pure relay)
```

The latent `!Send` risk (AsyncRead impl) is now visible at the
`ByteStream: Send` bound: if a future backend's AsyncRead is `!Send`, it
implements `LocalByteStream` instead and runs on the per-core
`LocalStreamingPipe` path. The Send-ness is a compile-time choice at the
stream impl, contained to this one adapter — exactly A4.

---

## 6. The contested question (what spec-rigor must settle)

**Spec A's position:** v1 wires only the Send streaming path
(`StreamingPipe`/`ByteStream`), declares the `Local*` parallel traits for
completeness, and uses a free `.local()` adapter instead of a blanket to
dodge the orphan conflict. Cancel + trailers migrate onto the stream.
The one `Box<dyn>` is `RequestBody::Streamed(Box<dyn ByteStream>)`.

**Anticipated Spec B (option b — parametric Send):** instead of two
parallel trait families, parameterize the stream over a `Maybe-Send`
marker (`ByteStream<S: SendMarker>`), so a single trait covers both
Send and !Send via a type parameter — avoiding the parallel-impl
duplication entirely, at the cost of a marker-type-parameter on every
streaming signature.

**Borda axes (spec-rigor will score):** completeness (does it cover all
5 worked examples + cancel/trailers + the typed edge?), consistency
(does the Send story contradict A5/A6 anywhere?), minimality (parallel
traits vs marker param vs Send-only), worked-example coverage (do
SSE/PTY/WS/proxy/buffered each map without a special case?),
composability with the bytes spine + `TypedPipe`, soundness (does
`!Send` actually stay out of the Send path at compile time?), interface
boundary (how many `Box<dyn>`; is the spine truly bytes-only?).

**Open sub-questions for the tournament:**
- Does the `.local()` adapter actually dodge the orphan conflict, or
  does it resurface it? (Spec A asserts it does; needs a compile check.)
- Is `RequestBody` an enum on the spine (leaking streaming into the
  buffered path) or strictly an argument to `StreamingPipe::call_streaming`
  only? Spec A puts it ONLY on the streaming trait — verify no spine
  leak.
- Trailers: slot-on-stream (Spec A) vs a third return value. Slot keeps
  the `poll_chunk` signature clean; a return value is more explicit.
