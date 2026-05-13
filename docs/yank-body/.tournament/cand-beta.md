# proxima yank-body — Spine vs Streaming Contract Specification

## (0) Vocabulary

- **Spine** — the bytes-only core: `Pipe` / `ThreadLocalPipe`, `Request`, `Response`, and the erased payload `Arc<dyn DynPipe>`. The spine knows only `Bytes` and `Frames` (a finite chunk sequence). It does not know about typing, streaming, trailers, or codecs.
- **Frames** — the spine payload newtype. A `Bytes` plus a "is-complete" bit. Replaces `BodyInner::{Empty,Bytes}`. Buffered RPC and buffered-then-wrapped sites use this exclusively.
- **`ByteStream`** — the streaming primitive: a poll-based source of `Result<Bytes, ProximaError>` chunks. Lives in a *separate* trait, never in the payload enum.
- **`TypedPipe<In,Out>`** — a static-dispatch typing layer above the spine. Owns a `MessageCodec` and decodes request bytes into `In`, encodes `Out` into response bytes. Replaces `Body::Typed` for the *wire-bound* case.
- **`Carry<T>`** — the in-process zero-copy typed payload (telemetry records). A thin alias for `Arc<dyn Any + Send + Sync>` carried *beside* the bytes payload, not inside it. Replaces `Body::Typed` for the *never-serialized* case. (The two former roles of `Body::Typed` — "decode-at-boundary" and "never-serialize in-process" — are genuinely different and split here deliberately.)
- **Completion** — the post-stream metadata: trailers. A `CompletionSlot` attached to a streaming response, resolved when the stream ends.
- **Drainer** — telemetry's per-core ring drainer (`recorder/drainer.rs`) that builds `Vec<Arc<T>>` once and fans out.

## (1) Axioms

- **A1 (Send is the only split).** The only trait bifurcation is `Send`. std/alloc are `#[cfg]` on a *single* trait, never a parallel trait. No 2^N families.
- **A2 (Spine is bytes-only).** The erased payload at `Arc<dyn DynPipe>` is `Frames` (a `Bytes` newtype). No enum with a `Typed` or `Stream` variant.
- **A3 (Streaming is an explicit capability, not a payload variant).** A `Pipe` that streams *declares* it by implementing a streaming trait. A buffered `Pipe` pays nothing for streaming's existence.
- **A4 (Typing is static-dispatch and lives above the spine).** `TypedPipe<In,Out>` decodes/encodes at the boundary via `MessageCodec`. The decoded value never enters the erased payload.
- **A5 (One `Arc::new` per record, borrow thereafter).** The telemetry fan-out builds `Vec<Arc<T>>` exactly once at the drainer; every consumer borrows. The contract must offer a path that introduces zero refcount bumps and zero `downcast` per consumer beyond one `TypeId` compare.
- **A6 (No punting — principle 15).** Every path declared here is fully specified; there are no stub seams. If a capability is out of scope it is named in §7, not stubbed.
- **A7 (Cross-cutting request metadata lives on the envelope, not the payload).** Cancellation is request-scoped → `RequestContext`. Socket handoff is response-scoped → `Response.upgrade` (already true today). Trailers are stream-completion-scoped → the streaming response, not the buffered spine.

## (2) The bytes spine

The spine keeps the existing `Pipe`/`ThreadLocalPipe`/`DynPipe`/`ThreadLocalDynPipe` traits **unchanged in shape**. Only the payload type on `Request`/`Response` changes.

```rust
#[derive(Clone, Debug, Default)]
pub struct Frames(Bytes);
impl Frames {
    pub fn empty() -> Self { Self(Bytes::new()) }
    pub fn from_bytes(bytes: Bytes) -> Self { Self(bytes) }
    pub fn as_bytes(&self) -> &Bytes { &self.0 }
    pub fn into_bytes(self) -> Bytes { self.0 }
}
```

`Request`/`Response` change their `body` field type; carry slot is a SEPARATE field defaulted None:

```rust
pub struct Request {
    pub method: Bytes, pub path: Bytes, pub query: HeaderList, pub headers: HeaderList,
    pub body: Frames,                       // was Body
    pub carry: Option<Carry>,               // was Body::Typed — in-process zero-copy
    pub stream: Option<ByteStreamHandle>,   // was Body::Stream — explicit streaming
    pub context: RequestContext,            // already carries cancel: CancellationToken
}
pub struct Response {
    pub status: u16, pub headers: HeaderList,
    pub body: Frames, pub carry: Option<Carry>,
    pub stream: Option<ResponseStream>,     // streaming + completion (trailers) here
    #[cfg(feature = "std")] pub upgrade: Option<UpgradeHandler>,  // unchanged
}
```

Three mutually-exclusive shapes by construction convention (not an enum): buffered (body, carry None, stream None); typed in-process (carry Some, body empty); streaming (stream Some, body empty). A buffered consumer reads `request.body.as_bytes()`. `Pipe`/`DynPipe`/handles byte-for-byte unchanged; no new spine trait.

## (3) The streaming contract — Send/!Send decision

**Decision: (c) streaming-always-Send; !Send is buffered-only**, plus a single feature-gated escape hatch `LocalByteStream` that lives OUT of the spine, not a parallel trait family.

Justification: edges.md is decisive — all ~27 streaming sites are Send, no !Send streaming exists, only latent !Send risk in AsyncRead adapters that are Send today. Genuinely-!Send work (Rc/RefCell/per-core/GPU) is state held across the call body, not flowing through a stream. A per-core ThreadLocalPipe can hold Rc state and still produce a Send byte stream because the bytes are owned Bytes (Send); the !Send-ness stays behind a channel whose receiver yields Send Bytes.

```rust
pub trait ByteStream: Send + 'static {
    fn poll_chunk(self: Pin<&mut Self>, cx: &mut Context<'_>)
        -> Poll<Option<Result<Bytes, ProximaError>>>;
}
pub type ByteStreamHandle = Pin<Box<dyn ByteStream>>;
```

This is `Stream<Item=Result<Bytes,ProximaError>> + Send` re-expressed as ONE named trait so the +Send is one documented decision, not 27 scattered bounds.

Why not (a) parallel families: 2^N trap; every streaming Pipe would pick a family, bridging needs 2x adapters, buys nothing today. Why not (b) Send-marker param: cleanest in theory but poisons the erased boundary — `Response.stream` would be `Option<Pin<Box<dyn ByteStream<?>>>>` and you cannot erase the marker without re-introducing a dyn split = (a) in costume. Markers shine for static code; they fight erasure, and the spine IS an erasure boundary.

Escape hatch fully specified: a feature-gated `LocalByteStream: 'static` (no blanket to ByteStream → no orphan recurrence) that rides the existing ThreadLocalPipe/local_upgrade_ticket path and never touches Response.stream; gated off until a !Send streaming need on Prime is measured. Mirrors the existing Response.upgrade (Send) vs local_upgrade_ticket (!Send) asymmetry.

### Cancellation and trailers

- **Cancellation → RequestContext.cancel (already there).** request.rs already carries a per-request CancellationToken. Body::cancel was redundant. The streaming bridge collect() takes the token from the request context. Body::from_stream_with_cancel/with_cancel deleted.
- **Trailers → ResponseStream completion slot.** Trailers are stream-end metadata; meaningless for buffered Frames.

```rust
pub struct ResponseStream { stream: ByteStreamHandle,
    #[cfg(feature="std")] trailers: Option<CompletionSlot> }
impl ResponseStream {
    pub fn collect(self, cancel: &CancellationToken) -> /* async */ Result<Bytes, ProximaError> { /* select! biased on cancel */ }
    pub fn once(b: Bytes) -> Self { /* single-chunk */ }
}
```

Bridging: streaming→buffered via collect(&cancel); buffered→streaming via once(bytes). Listener dispatch total: emit stream if Some, else frame body fixed-length.

## (4) TypedPipe + Carry

Body::Typed served TWO roles, split here. Role 1 wire-bound: `TypedPipe<In,Out>` is a Pipe (monomorphic), decode_input(body.as_bytes())->In, handler->Out, encode_output->Bytes, JsonCodec default; In/Out are stack locals, never erased. Role 2 in-process never-serialized: `Carry(Arc<dyn Any+Send+Sync>)` carried beside bytes.

```rust
pub struct Carry(Arc<dyn Any + Send + Sync>);
impl Carry {
    pub fn new<T: Any+Send+Sync>(v: T) -> Self { Self(Arc::new(v)) }
    pub fn from_arc<T: Any+Send+Sync>(v: Arc<T>) -> Self { Self(v) }
    pub fn downcast_ref<T: Any+Send+Sync>(&self) -> Option<&T> { self.0.downcast_ref() }
}
```

Fan-out proof: drainer does Vec<Arc<T>> once (one Arc::new/record); Carry::new adds one Arc around the Vec (same single bump Body::typed did); Tee fan-out clones Carry = one Arc::clone of the Vec-Arc per branch (NOT per record); each exporter borrows via downcast_ref + as_ref = one TypeId compare, zero per-record bumps. Identical to today. Carry is a separate field (not folded into Frames) because A2 — forcing SpanRecord->Bytes at the drainer burns exactly the serialization A5 avoids.

## (5) Five worked examples
- **5.1 Buffered RPC**: TypedPipe<QueryIn,QueryOut> over JsonCodec; framed-listener fills body=Frames(bytes), carry None, stream None; decode→handle→encode→Response{body:Frames(json)}. Zero stream alloc, zero Arc<dyn Any>, zero collect.
- **5.2 SSE**: SseUpstream impl Pipe returns Response{stream:Some(ResponseStream::new(SseChunks))}; SseChunks: ByteStream yields data:…\n\n per event; cancel via listener pump select! on request.context.cancel.
- **5.3 PTY**: PtyChunks holds cancel token + reads 4KiB chunks, stops on cancel; exit code rides a trailer slot written at EOF — correct home for trailers (A7).
- **5.4 WebSocket**: NOT a streaming-body case — socket handoff via Response.upgrade (Send) / local_upgrade_ticket (!Send). No ByteStream. Validates the spine split.
- **5.5 Proxy pass-through**: HttpProxy returns upstream Response.stream unchanged (zero-copy, no collect); Validate middleware bridges stream→buffered via collect(&cancel), inspects, re-wraps.

## (6) Properties
P1 buffered pays zero streaming tax (A2+A3); P2 one named Send decision (grep returns one site); P3 Arc fan-out invariant (1:1 Arc::new at drainer, 1 bump/batch via Carry, 1 bump/branch on fan-out, 0 on read); P4 no payload ambiguous (construction convention + total listener dispatch); P5 cancel single-sourced (RequestContext.cancel); P6 trailers iff streaming; P7 typing adds no erasure (TypedPipe monomorphic); P8 no orphan reintroduction (no blanket on ByteStream/LocalByteStream).

## (7) Scope
In: Frames + field changes; ByteStream + ResponseStream + decision (c) + gated LocalByteStream; Carry + drainer/exporter migration; TypedPipe + TypedHandler over MessageCodec/JsonCodec; cancel relocation; trailers relocation; both bridges; listener total-dispatch; all 5 examples.
Out (named, not stubbed): WS/CONNECT framing internals (existing upgrade seam); plugin DynCodec/CodecFactory erasure (existing); the concrete !Send per-core streaming CONSUMER (LocalByteStream contract defined + feature-gated, unblocked by measured Prime need); migration sequencing.
