# Streaming contract — merged spec

Status: merges two prior specs + a critique, re-derived against verified code at `proxima-pipe/src/{pipe,body,request}.rs` and `proxima-codec/src/lib.rs`.

## 0. Vocabulary
Names checked against the real tree (no existing `ByteStream`/`Carry`/`ResponseStream`/`LocalByteStream`; reused as-is: `Body`,`BodyStream`,`Bytes`,`Request`,`Response`,`HeaderList`,`UpgradeHandler`,`MessageCodec`,`DynCodec`).
- **Spine** — `Pipe`/`ThreadLocalPipe`, `Request`, `Response`. Body becomes bytes-only: a flat `Bytes` field, no `Body` enum, no union. `Request`/`Response` stay FLAT structs (no `RequestHead`/`ResponseHead` split — that was fabricated by an input spec; corrected).
- **Edge** — static-dispatch typing (`TypedPipe<In,Out,C>`) on the spine. Replaces `Body::Typed`.
- **Stream** — `ByteStream` (Send chunk-source trait) carried as `Option` field on `Response`. NO separate `StreamingPipe` trait. A streaming Pipe is an ordinary `Pipe` whose `Response.stream` is `Some`.
- **Buffered** — `Response.stream == None`; whole body is one `Bytes`. The 80% case.
- **Carry** — optional erased `Arc<dyn Any+Send+Sync>` field beside the bytes, for in-process telemetry fan-out. Replaces `Body::Typed` at runtime; `TypedPipe` is the static front door.

## 1. Axioms
- **A1** Dyn dispatch forces one erased payload; the wire payload is `Bytes`, never a union (`PipeHandle=Arc<dyn DynPipe>` real).
- **A2** The spine pays zero streaming tax: a buffered Response allocates no `Pin<Box<dyn Stream>>`, no trailers Mutex, no cancel tree.
- **A3** Send is the only genuine trait split; std/alloc are `#[cfg]` on one trait. No marker type-param on every signature.
- **A4** The Send/!Send streaming decision is contained in ONE named trait `ByteStream: Send` — not smeared across `Pipe` (which already decided Send).
- **A5** Streaming composes with the spine by FIELD, not a parallel pipe-trait. A streaming response is a normal `Pipe` returning `Response{stream:Some}`. ⇒ NO `DynStreamingPipe` erasure boundary (`Arc<dyn DynPipe>` already erases it).
- **A6** Cancel and request-trailers already have homes: `RequestContext.cancel` exists (request.rs:86), cancellation is per-request not per-body so does NOT move onto the stream; request trailers fold into `Request.headers` at decode time.
- **A7** Response trailers are stream-completion metadata; they live ON the stream behind `#[cfg(feature="std")]` (TrailersSlot=Arc<Mutex> is std-gated), never on a non-gated trait method.
- **A8** `ThreadLocalPipe` coherence is real (blanket removed, orphan conflict). v1 ships `ByteStream: Send` as the only wired source. `LocalByteStream` (!Send) is NOT declared vacuously — named as deferred with a concrete unblocking dependency.

## 2. The bytes spine
`Request`/`Response` stay FLAT (no head/body split):
```rust
pub struct Request { pub method: Bytes, pub path: Bytes, pub query: HeaderList,
    pub headers: HeaderList,   // request trailers fold in here at chunked-decode end (A6)
    pub body: Bytes,           // was Body
    pub context: RequestContext } // .cancel lives here (A6)
pub struct Response { pub status: u16, pub headers: HeaderList,
    pub body: Bytes,           // buffered; empty Bytes == Body::empty()
    #[cfg(feature="std")] pub stream: Option<ResponseStream>,  // streaming + trailers here, not spine
    pub carry: Option<Carry>,  // in-process typed fan-out; None for wire
    #[cfg(feature="std")] pub upgrade: Option<UpgradeHandler> } // pre-existing, not body/stream
```
Three mutually-exclusive shapes by construction convention, NOT an enum: buffered (bytes / None / None); streamed (empty / Some / None); typed (empty / None / Some). Constructors enforce it (`Response::ok`, `::streamed`, `::carry`); no public path sets two. Kills the union without an enum tag. Cancel→RequestContext.cancel; response trailers→ResponseStream; request trailers→Request.headers. `Body::{cancel,trailers}` deleted. `name()`/`background_tasks()` untouched (no pipe-trait added).

## 3. Streaming contract + Send/!Send
```rust
pub trait ByteStream: Send + 'static {
    fn poll_chunk(self: Pin<&mut Self>, cx: &mut Context<'_>)
        -> Poll<Option<Result<Bytes, ProximaError>>>;
    #[cfg(feature="std")]
    fn trailers(&self) -> Option<HeaderList> { None }  // valid only after Ready(None); std-only (A7)
}
#[cfg(feature="std")]
pub struct ResponseStream(pub Pin<Box<dyn ByteStream>>);  // THE one Box<dyn> on the streaming path
```
Cancellation is NOT on ByteStream: a streaming Pipe reads `request.context.cancel` and selects against it (as Body::collect did, body.rs:264). Cancel surface = `RequestContext::{cancel, child_token}` (real methods).

**No separate streaming Pipe trait** (the structural pivot): a streaming Pipe returns `Response{stream:Some(...)}`. Routing/Swappable/Tee/Diff all keep working (they dispatch `Pipe::call`, forward the Response, never touch the stream field). ⇒ exactly ONE `Box<dyn>` on the streaming path, ZERO new erasure traits.

**Send-only v1; !Send deferred (principle 15, option ii — honest).** 27/27 sites Send today. `LocalByteStream` is NOT declared in v1 (declaring it empty is the vacuous punt). Named in §7 with concrete dependency: a measured !Send per-core streaming source on Prime. CORRECTION to an input spec: `RequestContext.local_upgrade_ticket` is an UPGRADE-HANDLER ticket for a !Send socket, NOT a ready !Send streaming-body channel; the !Send stream field is new work gated on the measured need, parallel-impl per the ThreadLocalPipe precedent.

## 4. TypedPipe + Carry
TypedPipe is a monomorphic Pipe (no new Box<dyn>):
```rust
impl<In,Out,Handler,C> Pipe for TypedPipe<In,Out,Handler,C>
where C: MessageCodec<Input=In,Output=Out>, In: Send+Sync+'static, Out: Send+Sync+'static {
    fn call(&self, request: Request) -> impl Future<Output=Result<Response,ProximaError>> + Send {
        async move {
            let input = self.codec.decode_input(&request.body)?;   // codec sig is &[u8] (real)
            let output = (self.handler)(input).await?;
            let bytes = self.codec.encode_output(&output)?;        // -> Bytes (real)
            Ok(Response::ok(bytes)) } }
}
```
Real signatures: `MessageCodec::decode_input(&[u8])->Input`, `encode_output(&Output)->Bytes`. No fabricated `Payload`; codec operates on `&request.body` (&Bytes derefs to &[u8]).

Carry — in-process fan-out:
```rust
#[derive(Clone)] pub struct Carry(Arc<dyn Any+Send+Sync>);
impl Carry { pub fn new<T:Any+Send+Sync>(v:T)->Self{Self(Arc::new(v))}
    pub fn from_arc<T:Any+Send+Sync>(v:Arc<T>)->Self{Self(v)}
    pub fn downcast_ref<T:Any+Send+Sync>(&self)->Option<&T>{self.0.downcast_ref()} }
```
Refcount accounting (preserves Body::typed verbatim): drainer builds Vec<Arc<T>> ONCE; `from_arc` = one move (Arc<T>->Arc<dyn Any>, like body.rs:214); fan-out Tee/Diff clone = one atomic bump of outer Arc per branch; consumer downcast_ref = one TypeId compare + as_ref borrow, ZERO per-record bumps. Net zero additional allocs/bumps vs Body::typed; the static TypedPipe front door eliminates the downcast when In/Out known at call site. Bench arm (principle 14, home-turf = telemetry drainer at realistic batch) proves parity-or-better.

## 5. Worked examples
- **5.1 Buffered**: `impl Pipe for EchoJson { call → Ok(Response::ok(request.body)) }`; stream None, carry None. Wrap in TypedPipe+JsonCodec for typed.
- **5.2 SSE (concrete poll_chunk state machine)**: `enum SseState { Reading(Pin<Box<dyn ByteStream>>), Done }`; poll_chunk loops: Done→Ready(None); Reading→match upstream.poll_chunk {Pending→Pending; Ready(None)→set Done, Ready(None); Err→Err; Ok(chunk)→build `data: `+chunk+`\n\n`, Ready(Some(Ok(framed)))}; trailers None. Pipe returns Response{body:empty, stream:Some(ResponseStream(Box::pin(SseStream)))}.
- **5.3 PTY (cancel+trailers demonstrated)**: `PtyStream{reader, cancel:child_token, _child_guard}`; poll_chunk: if cancel.is_cancelled()→Err; else poll_read_available → empty=EOF Ready(None), chunk→Ready(Some(Ok)); trailers() returns x-exit-code header from child_guard.exit_status() after EOF. cancel from request.context.child_token() (A6), trailers from stream after Ready(None) (A7).
- **5.4 WebSocket**: handshake = ordinary buffered Pipe → `Response::new(101).with_upgrade(handler)`, stream None. Post-upgrade relay runs in UpgradeHandler against raw socket — NOT a ByteStream, NOT on Response.stream. Corrects forcing WS onto a streaming trait.
- **5.5 Proxy pass-through**: `PassthroughStream{upstream}` poll_chunk delegates to upstream.poll_chunk (pure relay). The AsyncRead !Send risk surfaces exactly at `ByteStream: Send` — a future !Send backend fails to compile and must use the deferred LocalByteStream path. A4 as a compile-time bound.

## 6. Properties
P1 no spine union (A1); P2 zero buffered tax (A2); P3 one streaming Box<dyn>, no DynStreamingPipe (A5); P4 Send contained — !Send fails to compile (A3/A4); P5 std not a 2nd axis (only ByteStream::trailers is #[cfg], no_std drops the method) (A7); P6 cancel has a real surface RequestContext::{cancel,child_token} (A6); P7 trailers both sides (response via ByteStream::trailers after Ready(None); request via Request.headers); P8 Arc fan-out preserved — one outer bump/branch, zero per-record (verbatim Body::typed); P9 composition unchanged (Tee/Diff/Routing/Swappable + name()/background_tasks() parity); P10 codec types real (MessageCodec verbatim, no fabricated Payload).

## 7. Scope
In (complete-as-contract): bytes-only spine (Request.body/Response.body: Bytes); Response.stream + ByteStream + std-gated trailers; Response.carry + TypedPipe; cancel→RequestContext.cancel (verified); response trailers→ByteStream::trailers; request trailers→Request.headers; 5 examples with real poll_chunk; deletion of Body/BodyInner/BodyStream/Body::{cancel,trailers} after migration.
Out (named dependency, principle 15 option ii): !Send per-core streaming (LocalByteStream + Response.local_stream) — unblocked by a MEASURED !Send streaming source on Prime; until then not declared (empty declaration = vacuous punt). WS/CONNECT framing internals (existing upgrade seam). DynCodec/CodecFactory plugin erasure (exists in factory.rs; TypedPipe's C defaults to JsonCodec, orthogonal). Migration sequencing (mechanical strangler-fig).
