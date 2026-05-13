# Critique (auxiliary — of cand-alpha only; NOT a candidate)

Adversarial review of cand-alpha against the eight axes. Auxiliary input
for judges (helps scoring); it is NOT one of the ranked candidates.

- **completeness WEAK**: `RequestHead`/`ResponseHead`/`Payload` undefined or fabricated (Request/Response are flat in the real tree, not head+body split); cancel has no contract METHOD on the stream trait (only "rides as a field"); request-side trailers missing; DynCodec/CodecFactory plugin path uncovered; name()/background_tasks() parity dropped on the streaming/typed traits.
- **consistency WEAK**: trailers()->HeaderList puts a std-gated dependency in a non-#[cfg] trait (makes std a 2nd axis, violating "Send is the only split"); a StreamOnce lift of every buffered Pipe contradicts "spine pays zero streaming tax"; "ONLY place Send is decided" is false (Pipe/ThreadLocalPipe already decide Send).
- **minimality ADEQUATE/weak**: 4 traits where 1 is wired + 3 are consumer-less per its own evidence; a manual `.local()` adapter where a derivable superset relation exists.
- **worked-example WEAK**: examples are ASCII narration; no concrete `poll_chunk` state machine; PTY/proxy decline the !Send case they exist to stress; cancel+trailers demonstrated for 0 of 5.
- **composability WEAK**: `.local()` only MOVES the orphan conflict (asserted settled in one section, listed open in another — contradiction); a `RequestBody{Buffered(Bytes),Streamed(...)}` enum argument RESURRECTS the deleted union as a 2-arm union and smears bytes across two homes; codec sigs (&[u8]/Bytes) conflated with a fabricated Payload type.
- **soundness WEAK (central flaw)**: declaring Local* traits with zero consumers is VACUOUS — the `Send` bound on the wired trait does all the compile-time work; the !Send pipe-trait body is never written; a single `RequestBody` enum hardcodes `Box<dyn ByteStream>`(Send) so it can never carry the !Send arm — latent unsoundness in "the ONE Box<dyn>".
- **interface boundary ADEQUATE**: spine is mostly bytes-only and scope is explicit, BUT `RequestBody::Buffered` smears bytes across two homes; the real Box<dyn> count is 3-4 (a DynStreamingPipe erasure for hot-swap/routing is unmentioned), not the claimed 1.
- **notation WEAK**: `Payload`/`RequestBody`/`ByteStream` collide with or fabricate against existing proxima types; `RequestHead`/`ResponseHead` fabricated; std-gated `HeaderList`/`TrailersSlot`/`CancellationToken` used unqualified in trait signatures.

Verified code facts (for judging composability/soundness of ALL candidates):
- `RequestContext.cancel: CancellationToken` (NOT Option) exists, request.rs:86, std-gated; has child_token()/with_cancel().
- `RequestContext.local_upgrade_ticket: Option<u64>` exists (request.rs:98) — it is an UPGRADE-HANDLER ticket for a !Send socket, NOT a !Send streaming-body channel.
- `Response.upgrade: Option<UpgradeHandler>` exists (request.rs:362), std-gated, a separate raw-socket field.
- Request/Response are FLAT structs (no head/body split). Body currently carries cancel+trailers (std-gated).
- The `impl<T:Pipe> ThreadLocalPipe for T` blanket was REMOVED (orphan conflict, pipe.rs:383); the `impl<T:ThreadLocalPipe> ThreadLocalDynPipe for T` blanket was KEPT.
- `MessageCodec::decode_input(&[u8])->Input`, `encode_output(&Output)->Bytes` (codec lib.rs:37-38).
- No existing `Payload`/`Frames`/`Carry`/`ByteStream`/`ResponseStream`/`LocalByteStream`. Existing: Body, BodyStream, Bytes, Request, Response, HeaderList, UpgradeHandler, MessageCodec, DynCodec, CodecFactory.
