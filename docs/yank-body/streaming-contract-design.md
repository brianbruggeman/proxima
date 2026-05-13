# Streaming contract — BINDING (yank-body initiative)

> **Crate consolidation / path note (2026-07).** `proxima-pipe` (named
> below, e.g. `proxima-pipe/src/byte_stream.rs`) no longer exists as a
> standalone crate — it folded into `proxima-primitives`. `Pipe`/
> `ThreadLocalPipe`/`DynPipe` referenced below now live under
> `proxima-primitives/src/pipe/`.

Status: **converged** via `/spec-rigor` (1 round, unanimous 6/0 Borda sweep;
two judges scored it 8/0/0). This is the binding spine-vs-streaming
contract for removing the concrete `Body` type from proxima. Provenance,
composition map, and risk register at the end.

Scope: the spine-vs-streaming boundary. The `Body::Typed -> TypedPipe`
move and the `Body -> Bytes` collapse are mechanical once this is fixed.

---

## 0. Vocabulary

Names checked against the real tree (no existing `ByteStream`/`Carry`/
`ResponseStream`/`LocalByteStream`; reused as-is: `Body`, `BodyStream`,
`Bytes`, `Request`, `Response`, `HeaderList`, `UpgradeHandler`,
`MessageCodec`, `DynCodec`, `CodecFactory`).

- **Spine** — `Pipe`/`ThreadLocalPipe`, `Request`, `Response`. The body
  becomes **bytes-only**: a flat `Bytes` field. No `Body` enum, no union.
  `Request`/`Response` stay FLAT structs (there is no `RequestHead`/
  `ResponseHead` split in the tree).
- **Edge** — static-dispatch typing (`TypedPipe<In,Out,C>`) layered on the
  spine. Replaces `Body::Typed` at the wire boundary.
- **Stream** — `ByteStream` (the Send chunk-source trait) carried as an
  `Option` field on `Request`/`Response`. There is **no separate
  `StreamingPipe` trait**: a streaming Pipe is an ordinary `Pipe` whose
  `Response.stream` is `Some`.
- **Buffered** — `stream == None`; the whole body is one `Bytes`. The 80%
  case.
- **Carry** — an optional erased `Arc<dyn Any+Send+Sync>` field beside the
  bytes (on BOTH `Request` and `Response`), for in-process telemetry
  fan-out. Replaces `Body::Typed` at runtime; `TypedPipe` is the
  static-dispatch front door.

---

## 1. Axioms

- **A1** — Dyn dispatch forces one erased payload; the wire payload is
  `Bytes`, never a union (`PipeHandle = Arc<dyn DynPipe>` is real).
- **A2** — The spine pays zero streaming tax: a buffered `Response`
  allocates no `Pin<Box<dyn Stream>>`, carries no trailers `Mutex`, no
  cancel tree.
- **A3** — Send is the only genuine trait split; std/alloc are `#[cfg]` on
  one trait. No Send-marker type-parameter smeared across every signature.
- **A4** — The Send/!Send streaming decision is contained in ONE named
  trait, `ByteStream: Send` — not smeared across `Pipe` (which already
  decided Send).
- **A5** — Streaming composes with the spine by FIELD, not a parallel
  pipe-trait. A streaming response is a normal `Pipe` returning
  `Response { stream: Some(..) }`. ⇒ NO `DynStreamingPipe` erasure
  boundary (`Arc<dyn DynPipe>` already erases it).
- **A6** — Cancel and request-trailers already have spine homes:
  `RequestContext.cancel: CancellationToken` exists (`request.rs:86`);
  cancellation is per-request, not per-body, so it does NOT move onto the
  stream. Request trailers fold into `Request.headers` at chunked-decode
  end.
- **A7** — Response trailers are stream-completion metadata; they live ON
  the stream behind `#[cfg(feature = "std")]` (`TrailersSlot = Arc<Mutex>`
  is std-gated), never on a non-gated trait method.
- **A8** — `ThreadLocalPipe` coherence is real (the `impl<T:Pipe>
  ThreadLocalPipe for T` blanket was removed for orphan conflict). v1
  ships `ByteStream: Send` as the only wired source. `LocalByteStream`
  (!Send) is NOT declared vacuously; it is named as deferred with a
  concrete unblocking dependency (§7).

---

## 2. The bytes spine

`Request`/`Response` stay FLAT. `carry` is on BOTH (the telemetry drainer
builds a `Request` carrying records; the exporter reads it):

```rust
pub struct Request {
    pub method: Bytes,
    pub path: Bytes,
    pub query: HeaderList,
    pub headers: HeaderList,        // request trailers fold in here at decode end (A6)
    pub body: Bytes,                // was `body: Body`
    pub carry: Option<Carry>,       // in-process typed payload (telemetry); None for wire
    #[cfg(feature = "std")]
    pub stream: Option<RequestStream>, // streamed request body (uploads, WS inbound relay)
    pub context: RequestContext,    // .cancel lives here (A6) — unchanged
}

pub struct Response {
    pub status: u16,
    pub headers: HeaderList,
    pub body: Bytes,                // buffered; empty Bytes == Body::empty()
    pub carry: Option<Carry>,       // in-process typed fan-out; None for wire
    #[cfg(feature = "std")]
    pub stream: Option<ResponseStream>, // streaming + trailers ride HERE, not the spine
    #[cfg(feature = "std")]
    pub upgrade: Option<UpgradeHandler>, // pre-existing raw-socket handoff; not body/stream
}
```

**Three mutually-exclusive body shapes by construction convention, NOT an
enum** (this is what kills the union without re-introducing an enum tag):

| shape | `body` | `stream` | `carry` |
|---|---|---|---|
| buffered | the bytes | `None` | `None` |
| streamed | empty | `Some` | `None` |
| typed (in-proc) | empty | `None` | `Some` |

Constructors enforce the convention (`Response::ok(bytes)`,
`Response::streamed(s)`, `Response::carry(arc)`); no public path sets two
at once. A buffered response is literally `{body: Bytes, stream: None,
carry: None}` — zero streaming tax (A2).

`Body::{cancel, trailers}` fields are deleted: cancel →
`RequestContext.cancel`, response trailers → `ResponseStream`, request
trailers → `Request.headers`. `name()`/`background_tasks()` are untouched
(no pipe-trait is added, so parity is preserved by construction).

`body` is raw `Bytes`, not a newtype — the spine has no buffered-side
invariant to hang on a wrapper, and raw `Bytes` is the tightest erased
payload (risk-register item R2 resolved: keep raw).

---

## 3. The streaming contract + Send/!Send decision

```rust
// proxima-pipe/src/byte_stream.rs (new)

/// One streamed body, pulled chunk-by-chunk. THE single place the
/// Send/!Send stream question is decided (A4): the source is `Send`.
pub trait ByteStream: Send + 'static {
    /// `Ready(None)` = end of body. After it, `trailers()` may be read.
    fn poll_chunk(self: Pin<&mut Self>, cx: &mut Context<'_>)
        -> Poll<Option<Result<Bytes, ProximaError>>>;

    /// Response trailers, valid only AFTER poll_chunk returned Ready(None).
    /// std-only (TrailersSlot needs Mutex, A7). Under no_std the method
    /// does not exist, so std never becomes a second trait axis.
    #[cfg(feature = "std")]
    fn trailers(&self) -> Option<HeaderList> { None }
}

/// Erased streamed response: the ONE Box<dyn> on the streaming path.
/// Carried as `Response.stream`. No `DynStreamingPipe` exists because the
/// stream rides inside the `Response` that `Arc<dyn DynPipe>` already
/// returns (A5).
#[cfg(feature = "std")]
pub struct ResponseStream(pub Pin<Box<dyn ByteStream>>);
#[cfg(feature = "std")]
pub struct RequestStream(pub Pin<Box<dyn ByteStream>>);
```

Cancellation is NOT a method/field on `ByteStream`: a streaming Pipe reads
`request.context.cancel` and selects against it inside its driving task
(exactly as `Body::collect` did, `body.rs:264`). The cancel contract
surface is `RequestContext::{cancel, child_token}` — real methods.

**No separate streaming Pipe trait (the structural pivot).** A streaming
Pipe returns `Response { stream: Some(..) }`. `RoutingPipe`,
`SwappablePipe`, `Tee`, `Diff` keep working unchanged — they dispatch
`Pipe::call` and forward the `Response`; the stream is a field they don't
touch. Result: exactly ONE `Box<dyn>` on the streaming path, ZERO new
erasure traits.

**Send-only v1; `!Send` explicitly deferred (principle 15, option ii).**
27/27 streaming sites are Send today. `LocalByteStream` is NOT declared in
v1 — declaring it empty is the vacuous punt the tournament rejected. It is
named in §7 with a concrete unblocking dependency: a *measured* `!Send`
per-core streaming source on Prime. When it arrives, it is a parallel
`LocalByteStream` + `Response.local_stream` field gated behind the same
feature as `ThreadLocalPipe` use — paid only at that one `Box<dyn>`, never
on the Send path. **Correction to a tournament input:**
`RequestContext.local_upgrade_ticket: Option<u64>` (`request.rs:98`) is an
UPGRADE-HANDLER ticket for a `!Send` socket, NOT a ready `!Send`
streaming-body channel; the `!Send` stream field is new work gated on the
measured need, parallel-impl per the `ThreadLocalPipe` precedent.

---

## 4. The typed edge — `TypedPipe<In,Out,C>` + `Carry`

### 4.1 TypedPipe is a monomorphic Pipe (no new Box<dyn>)

```rust
impl<In, Out, Handler, C> Pipe for TypedPipe<In, Out, Handler, C>
where
    C: MessageCodec<Input = In, Output = Out>,
    In: Send + Sync + 'static, Out: Send + Sync + 'static,
{
    fn call(&self, request: Request)
        -> impl Future<Output = Result<Response, ProximaError>> + Send {
        async move {
            let input = self.codec.decode_input(&request.body)?;  // &Bytes derefs to &[u8]
            let output = (self.handler)(input).await?;
            let bytes = self.codec.encode_output(&output)?;       // -> Bytes
            Ok(Response::ok(bytes))
        }
    }
}
```

Real codec signatures: `MessageCodec::decode_input(&[u8]) -> Input`,
`encode_output(&Output) -> Bytes` (`proxima-codec/src/lib.rs:37-38`). No
fabricated `Payload`; the codec operates on `&request.body`. `In`/`Out`
are stack locals, never erased.

### 4.2 Carry — in-process fan-out, refcount accounting

```rust
#[derive(Clone)]
pub struct Carry(Arc<dyn Any + Send + Sync>);
impl Carry {
    pub fn new<T: Any + Send + Sync>(v: T) -> Self { Self(Arc::new(v)) }
    pub fn from_arc<T: Any + Send + Sync>(v: Arc<T>) -> Self { Self(v) }
    pub fn downcast_ref<T: Any + Send + Sync>(&self) -> Option<&T> { self.0.downcast_ref() }
}
```

Preserves `Body::typed` fan-out verbatim:
- drainer builds `Vec<Arc<T>>` ONCE (one `Arc::new` per record);
- `Carry::from_arc` is one move (`Arc<T> -> Arc<dyn Any>`, like
  `body.rs:214`); wrapping a batch is one bump per batch;
- fan-out (`Tee`/`Diff` cloning the `Request`/`Response`): `Carry::clone`
  is one atomic bump of the outer Arc per branch (NOT per record);
- consumer `downcast_ref::<T>()` = one TypeId compare + `as_ref` borrow,
  ZERO per-record bumps.

Net: zero additional allocations or bumps vs `Body::typed`. The static
`TypedPipe` front door eliminates the downcast entirely when `In`/`Out`
are known at the call site. **Gate point 13 home-turf bench arm = the
telemetry drainer at realistic batch size, proving parity-or-better vs the
`Body::typed` path (principle 14).**

---

## 5. Worked examples (re-derived against §2-§4)

### 5.1 Buffered RPC (framed JSON from a downstream consumer — 80%)

```rust
struct EchoJson;
impl Pipe for EchoJson {
    fn call(&self, request: Request)
        -> impl Future<Output = Result<Response, ProximaError>> + Send {
        async move { Ok(Response::ok(request.body)) }   // Bytes in, Bytes out
    }
}
```
`stream: None, carry: None`. No Box, no trailers, no cancel-on-body. Wrap
in `TypedPipe<In, Out, _, JsonCodec<In, Out>>` for typed access.

### 5.2 SSE — concrete poll_chunk state machine

```rust
enum SseState { Reading(Pin<Box<dyn ByteStream>>), Done }
struct SseStream { state: SseState }

impl ByteStream for SseStream {
    fn poll_chunk(self: Pin<&mut Self>, cx: &mut Context<'_>)
        -> Poll<Option<Result<Bytes, ProximaError>>> {
        let this = self.get_mut();
        loop {
            match &mut this.state {
                SseState::Done => return Poll::Ready(None),
                SseState::Reading(upstream) => match upstream.as_mut().poll_chunk(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(None) => { this.state = SseState::Done; return Poll::Ready(None); }
                    Poll::Ready(Some(Err(error))) => return Poll::Ready(Some(Err(error))),
                    Poll::Ready(Some(Ok(chunk))) => {
                        let mut framed = bytes::BytesMut::with_capacity(chunk.len() + 8);
                        framed.extend_from_slice(b"data: ");
                        framed.extend_from_slice(&chunk);
                        framed.extend_from_slice(b"\n\n");
                        return Poll::Ready(Some(Ok(framed.freeze())));
                    }
                },
            }
        }
    }
    #[cfg(feature = "std")]
    fn trailers(&self) -> Option<HeaderList> { None }
}
```
Wrapping Pipe: `Ok(Response { body: Bytes::new(), stream:
Some(ResponseStream(Box::pin(SseStream{..}))), carry: None, .. })`. No
`StreamingPipe`, no `async_stream!` macro.

### 5.3 PTY — cancel + trailers demonstrated

```rust
struct PtyStream { reader: PtyReader, cancel: CancellationToken, child_guard: ChildCleanup }
impl ByteStream for PtyStream {
    fn poll_chunk(self: Pin<&mut Self>, cx: &mut Context<'_>)
        -> Poll<Option<Result<Bytes, ProximaError>>> {
        let this = self.get_mut();
        if this.cancel.is_cancelled() {
            return Poll::Ready(Some(Err(ProximaError::Body("pty cancelled".into()))));
        }
        match Pin::new(&mut this.reader).poll_read_available(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Ok(chunk)) if chunk.is_empty() => Poll::Ready(None),   // EOF
            Poll::Ready(Ok(chunk)) => Poll::Ready(Some(Ok(chunk))),
            Poll::Ready(Err(error)) => Poll::Ready(Some(Err(error.into()))),
        }
    }
    #[cfg(feature = "std")]
    fn trailers(&self) -> Option<HeaderList> {
        self.child_guard.exit_status().map(|code| {
            let mut headers = HeaderList::new();
            headers.insert(Bytes::from_static(b"x-exit-code"),
                           Bytes::from(code.to_string().into_bytes()));
            headers
        })
    }
}
```
The Pipe builds `PtyStream { cancel: request.context.child_token(), .. }`
— cancel from the spine context (A6, replacing `Body::cancel`), trailers
(exit code) from the stream after `Ready(None)` (A7).

### 5.4 WebSocket — upgrade, not stream

```rust
fn ws_handshake(request: Request) -> Result<Response, ProximaError> {
    let handler = WsHandler::from_request(&request)?;
    Ok(Response::new(101).with_upgrade(handler))   // upgrade field, stream: None
}
```
Post-upgrade frame relay runs inside the `UpgradeHandler` against the raw
socket — NOT a `ByteStream`, NOT on `Response.stream`. Clean split:
handshake = buffered spine; relay = upgrade seam (`request.rs:362`).

### 5.5 Proxy pass-through — where the latent `!Send` risk surfaces

```rust
struct PassthroughStream { upstream: Pin<Box<dyn ByteStream>> }
impl ByteStream for PassthroughStream {
    fn poll_chunk(self: Pin<&mut Self>, cx: &mut Context<'_>)
        -> Poll<Option<Result<Bytes, ProximaError>>> {
        self.get_mut().upstream.as_mut().poll_chunk(cx)   // pure relay
    }
}
```
A future `!Send` AsyncRead backend fails to satisfy `ByteStream: Send` at
compile time and must use the deferred `LocalByteStream` path (§7) —
contained to this one adapter, never leaking into the Send path. A4
realized as a compile-time bound.

---

## 6. Properties (follow from the axioms)

- **P1** no spine union (A1). **P2** zero buffered tax (A2). **P3** one
  streaming `Box<dyn>`, no `DynStreamingPipe` (A5). **P4** Send contained
  — `!Send` fails to compile (A3/A4, §5.5). **P5** std not a second axis —
  only `ByteStream::trailers` is `#[cfg]`; no_std drops the method (A7).
  **P6** cancel has a real surface, `RequestContext::{cancel,
  child_token}` (A6, §5.3). **P7** trailers both sides (response via
  `ByteStream::trailers` after `Ready(None)`; request via
  `Request.headers`). **P8** Arc fan-out preserved — one outer bump per
  branch, zero per-record (§4.2). **P9** composition unchanged
  (`Tee`/`Diff`/`Routing`/`Swappable` + `name()`/`background_tasks()`
  parity). **P10** codec types real (`MessageCodec` verbatim, no fabricated
  `Payload`).

---

## 7. In / out of scope (named dependencies, principle 15)

**In scope (complete-as-contract):** bytes-only spine (`Request.body`/
`Response.body: Bytes`); `{Request,Response}.stream` + `ByteStream: Send`
with std-gated `trailers`; `{Request,Response}.carry` + `TypedPipe`; cancel
→ `RequestContext.cancel`; response trailers → `ByteStream::trailers`;
request trailers → `Request.headers`; all 5 worked examples with real
`poll_chunk`; deletion of `Body`/`BodyInner`/`BodyStream`/`Body::{cancel,
trailers}` after strangler-fig migration.

**Out of scope, with named unblocking dependency:**
- **`!Send` per-core streaming (`LocalByteStream` + `*.local_stream`)** —
  unblocked by a *measured* `!Send` streaming source on Prime. Until then
  not declared (empty declaration = vacuous punt). Parallel-impl per the
  `ThreadLocalPipe` precedent.
- **WS/CONNECT framing internals** — the existing `Response.upgrade` /
  `UpgradeHandler` seam; not a `ByteStream` concern.
- **`DynCodec`/`CodecFactory` plugin erasure** — already exists
  (`proxima-codec/src/factory.rs`); `TypedPipe`'s `C` defaults to
  `JsonCodec`; orthogonal.
- **Migration sequencing** — mechanical strangler-fig (alias `Body` →
  bytes first so the ~694 buffered pass-through refs are unaffected, then
  delete the husk); a plan concern, in `discipline.md`.

---

## Provenance

- Produced by `/spec-rigor`, round 1. Incumbent Spec A (hand-authored with
  full codebase grounding) vs blind independent Spec B vs synthesis.
- **Result: synthesis won unanimously, Borda 6/0** (3/3 first-place; two
  judges 8/0/0, one 8/0/0-equivalent). B second (3), A third (0).
- Convergence declared on the unanimous single-round sweep: the design
  space collapsed to one natural basis (bytes spine + `ByteStream` field +
  `Carry` + cancel-on-`RequestContext`), which is the skill's
  abort-the-contest signal — a second round's blind author would re-derive
  this basis. Not the strict two-consecutive-wins rule; noted honestly.

## Composition map (sibling vocabulary touchpoints)

- `proxima-pipe`: `Pipe`/`ThreadLocalPipe`/`DynPipe` (unchanged shape),
  `RequestContext.cancel` + `child_token()` (cancel home),
  `Response.upgrade`/`UpgradeHandler` (WS home), `HeaderList`.
- `proxima-codec`: `MessageCodec::{decode_input, encode_output}`,
  `JsonCodec<In,Out>` (TypedPipe default `C`); `DynCodec`/`CodecFactory`
  orthogonal.
- `proxima-telemetry`: `recorder/drainer.rs` (the `Vec<Arc<T>>` fan-out
  `Carry` must preserve), `pipes.rs` (exporter `downcast_ref` consumers).

## Risk register (weakest points even in the winner)

- **R1 (resolved here):** `carry` asymmetry — judge flagged it on
  `Response` only. RESOLVED: `carry` is on BOTH `Request` and `Response`
  (the drainer builds a `Request` carrying records). Reflected in §2.
- **R2 (resolved here):** `body: Bytes` raw vs newtype. RESOLVED: keep raw
  `Bytes` — no buffered-side invariant to hang; tightest erased payload.
- **R3 (open, low):** request-side streaming (`RequestStream`) has thinner
  worked-example coverage than the response side (only WS-inbound /
  uploads). Validate during the C-streaming migration slice with a real
  upload + WS-inbound test. No contract change expected.
- **R4 (open, low):** the `!Send` deferral assumes no Prime per-core
  streaming need lands during this initiative. If one is measured
  mid-migration, the parallel `LocalByteStream` slice is added then (its
  shape is specified; only the consumer is deferred).
