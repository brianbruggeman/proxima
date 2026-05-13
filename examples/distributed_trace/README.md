# distributed-trace — cross-instance propagation

Follow one request across two proxima instances and check whether both hops land in ONE trace.

## Builds on

- [`traces`](../traces/README.md) — spans + propagation across async boundaries.
- [`instrument`](../instrument/README.md) — one `#[proxima::instrument]` → all three pillars.
- [`export`](../export/README.md) — console + file + OTLP sinks together (never OTLP-only).

(`traces`/`instrument`/`export` are cataloged in `examples/README.md` but not yet built as
standalone examples — this capstone exercises the same primitives directly.)

## What it demonstrates

Two `App` instances in one process (per `multi_runtime.rs`'s pattern of standing up several
`App`s side by side): instance A ("front") receives the client's request and forwards it to
instance B ("origin") over a real TCP/HTTP hop. A injects its `RequestContext`'s W3C
`traceparent` onto the outbound request; B's own H1 listener extracts it on ingress
(`RequestContext::extract_propagation`, already wired in `proxima-http/src/http1/serve.rs`). Each side
also wraps a function in `#[proxima::telemetry::instrument(parent = request.context.traceparent())]`,
and a `Recorder` fans every resulting span out to a console sink and an in-memory capture buffer
so the exported `SpanRecord`s can be inspected after the request completes.

The question this answers empirically, not by inspection: do A's span and B's span come out as
one connected trace (same `trace_id`, B parented under A), or two disconnected traces?

**Answer: connected, at both layers, including the literal parent chain.** The header layer was
already wired (the egress-inject fix this example itself adds); the span layer connects via the
`parent = <expr>` argument on `#[instrument]`/`#[span]`, which routes to
`Recorder::span_from_traceparent` instead of opening a fresh root; and the h1 listener now opens
its own `h1_request` span with `parent = request.context.traceparent()`
(`proxima-http/src/http1/serve.rs`), so an inbound request continues the caller's trace instead of minting
a fresh root at the protocol boundary. See "What you'll see" below for the precise mechanism.

## Run

```sh
cargo run --example distributed_trace
```

No extra features needed — `serve-prime`, `http-prime-deps`, and `macros` are all in the
default feature set. (`required-features` is still declared explicitly in `Cargo.toml` so a
`--no-default-features` build gates cleanly.)

## What you'll see

```
origin (B) listening on 127.0.0.1:8092
front  (A) listening on 127.0.0.1:8091, forwards to 127.0.0.1:8092

client -> front raw response:
HTTP/1.1 200 OK
traceparent: 00-5add75d5c99cce59c5233a49dc98ce05-dc9fd05b6ad5db43-01
content-length: 149

front_traceparent=00-5add75d5c99cce59c5233a49dc98ce05-dc9fd05b6ad5db43-01
origin_traceparent=00-5add75d5c99cce59c5233a49dc98ce05-dc9fd05b6ad5db43-01

SPAN distributed_trace front: duration_ns=0
SPAN distributed_trace origin: duration_ns=1000
SPAN proxima_http::http1::serve h1_request: duration_ns=224000
SPAN proxima_http::http1::serve h1_request: duration_ns=855000
drained 4 telemetry records, 4 spans captured:
  name=front    trace_id=5add75d5c99cce59c5233a49dc98ce05 span_id=1ad58c2640eb8fb1 parent_span_id=dc9fd05b6ad5db43
  name=origin   trace_id=5add75d5c99cce59c5233a49dc98ce05 span_id=52b5bb6fa1c5349f parent_span_id=dc9fd05b6ad5db43
  name=h1_request trace_id=5add75d5c99cce59c5233a49dc98ce05 span_id=f3e025963dc36a13 parent_span_id=dc9fd05b6ad5db43
  name=h1_request trace_id=5add75d5c99cce59c5233a49dc98ce05 span_id=0852c912a1e5b2d1 parent_span_id=dc9fd05b6ad5db43

--- validation ---
W3C header layer (RequestContext.trace_id via inject_propagation/establish_trace_context):
  front  traceparent = 00-5add75d5c99cce59c5233a49dc98ce05-dc9fd05b6ad5db43-01
  origin traceparent = 00-5add75d5c99cce59c5233a49dc98ce05-dc9fd05b6ad5db43-01
  -> CONNECTED: same trace_id crossed the A -> B hop

telemetry span layer (#[proxima::telemetry::instrument(parent = ...)] on each pipe):
  front  span trace_id = 5add75d5c99cce59c5233a49dc98ce05
  origin span trace_id = 5add75d5c99cce59c5233a49dc98ce05
  -> CONNECTED: one trace, two spans

literal parent_span_id chain (establish_trace_context preserves the inbound span-id):
  front header span-id  = Some(SpanId([220, 159, 208, 91, 106, 213, 219, 67]))
  origin span parent_span_id = Some(SpanId([220, 159, 208, 91, 106, 213, 219, 67]))

PASS: distributed tracing across two proxima instances lands in ONE trace.
      Both layers agree: the header layer via inject_propagation/establish_trace_context,
      the span layer via #[instrument(parent = request.context.traceparent())] routing
      to `Recorder::span_from_traceparent` instead of a fresh root.
      The literal parent_span_id chain crosses the wire hop too: establish_trace_context
      preserves the inbound span-id instead of discarding it.

origin drained: cores_acked=2 hooks_drained=0
front  drained: cores_acked=2 hooks_drained=0
```

Trace/span IDs, span durations, and the exact interleaving of the two apps'
async span exports are non-deterministic between runs (fresh random IDs,
real scheduling) — the invariant `cargo run --example distributed_trace`
always proves is 4 spans captured, one shared `trace_id`, and
`origin_span.parent_span_id == front_header_span`, not this literal
transcript.

Two extra `h1_request` spans now appear that weren't in earlier captures of this example: h1's
listener boundary (`proxima-http/src/http1/serve.rs`) opens one per inbound request via
`#[proxima_telemetry::instrument(name = "h1_request", parent = request.context.traceparent(), err)]`
around the Pipe dispatch — the seam this README used to describe as still-needed wiring.

Three things connected now:

- **W3C header layer — CONNECTED.** `front_traceparent` and `origin_traceparent` are now
  byte-identical: `RequestContext::extract_propagation` (`proxima-primitives/src/pipe/request.rs`) preserves
  the inbound span-id instead of minting its own when a request carries one, so origin's restamped
  traceparent is exactly what front put on the wire. This required one line of wiring this example
  adds and notes explicitly: `request.context.inject_propagation(&mut outbound_headers)` in
  `FrontPipe::call`, before the forward. Nothing else in the proxima tree calls
  `inject_propagation` on any real forwarding path today — every existing caller is a unit test in
  `proxima-primitives/src/pipe/request.rs`. The example asserts this connects; a failure here would mean the
  wire-level propagation itself is broken, not just unwired.
- **Telemetry span layer — CONNECTED.** `front_hop`/`origin_hop` are each
  `#[proxima::telemetry::instrument(name = "...", parent = parent)]` with `parent: Option<&[u8]>`
  passed as `request.context.traceparent()` at the call site. When `Some`, the macro opens the
  span via `Recorder::span_from_traceparent` instead of `Recorder::span` — inheriting the caller's
  `trace_id` and recording its span-id as `parent_span_id` — instead of an unconditional fresh
  root. The example asserts `front_span.trace_id == origin_span.trace_id`.
- **Literal parent_span_id chain — CONNECTED.** `origin_span.parent_span_id` is now front's own
  wire-level span-id, not a self-referential placeholder: `extract_propagation` used to mint a
  fresh `span_id` per hop and discard the inbound header's span-id component, so origin's recorded
  parent was origin's own restamped context id, matching only the span-id component of
  `origin_traceparent`, never front's. Preserving the inbound span-id fixes that — the example
  asserts `origin_span.parent_span_id == front_header_span` (the span-id parsed out of
  `front_traceparent`), a strict single-tree reconstruction from `parent_span_id` alone, not just a
  shared `trace_id`.
- **h1 listener boundary — WIRED.** `proxima-http/src/http1/serve.rs`'s `dispatch_request` opens an
  `h1_request` span with `parent = request.context.traceparent()` around every inbound request's
  Pipe dispatch, so a request arriving with a `traceparent` header continues that trace instead of
  minting a fresh root at the protocol boundary. `proxima-http`'s http2/http3 servers mirror the
  same seam (`h2_request` / `h3_request`) — not exercised by this two-hop h1-only example, but
  covered by each crate's own test suite.

Exit code is 0 when all layers connect — the fixed, current, reproducible result.
