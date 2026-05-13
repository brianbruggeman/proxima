#![cfg(feature = "alloc")]

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;

use bytes::Bytes;
#[cfg(feature = "std")]
use proxima_core::signal::Signal;
#[cfg(feature = "std")]
use std::sync::OnceLock;

use proxima_core::time::Instant;

use crate::pipe::body::{ChunkStream, RequestStream, ResponseStream};
use crate::pipe::capture_surface::CaptureContext;
use crate::pipe::endpoint::PeerInfo;
use crate::pipe::header_list::HeaderList;
use crate::pipe::header_name::HeaderName;
use crate::pipe::telemetry_surface::{Labels, NoopTelemetry, TelemetryHandle};
#[cfg(feature = "std")]
use crate::pipe::upgrade::UpgradeHandler;
use proxima_core::ProximaError;

// Process-wide shared defaults so `RequestContext::default()` skips per-
// call heap allocation. Each handle is an Arc clone (atomic increment).
//
// Signal is shared by reference but each clone tracks the same
// cancel state — callers that need an independent token (per-request
// timeout, deadline cancel) must call `.with_cancel(Signal::new())`
// explicitly. The default is "never-cancelled" — matches RequestContext's
// pre-existing semantics where `default()` produced a fresh token nobody
// signalled.
#[cfg(feature = "std")]
static NOOP_TELEMETRY: OnceLock<TelemetryHandle> = OnceLock::new();
#[cfg(feature = "std")]
static NEVER_CANCEL: OnceLock<Signal> = OnceLock::new();

#[cfg(feature = "std")]
fn noop_telemetry() -> TelemetryHandle {
    NOOP_TELEMETRY
        .get_or_init(|| Arc::new(NoopTelemetry))
        .clone()
}

#[cfg(not(feature = "std"))]
fn noop_telemetry() -> TelemetryHandle {
    Arc::new(NoopTelemetry)
}

#[cfg(feature = "std")]
fn never_cancel() -> Signal {
    NEVER_CANCEL.get_or_init(Signal::new).clone()
}

#[derive(Clone)]
pub struct RequestContext {
    pub telemetry: TelemetryHandle,
    /// Optional request deadline, on the bound timer driver's clock. Build it
    /// from `proxima_core::time::now()` so the origin agrees with the reader's
    /// (`src/load.rs` races dispatch against it).
    pub deadline: Option<Instant>,
    /// trace identifier — typically W3C traceparent ASCII bytes.
    /// `Arc<[u8]>` so per-request clones are refcount bumps. Populated by
    /// [`adopt_trace_context`](Self::adopt_trace_context); proxima-pipe
    /// carries only this byte form — the typed trace/span identifier parse +
    /// generate logic lives in `proxima_telemetry::propagation`.
    pub trace_id: Option<Arc<[u8]>>,
    /// W3C `baggage` header bytes lifted off the inbound request, re-stamped
    /// onto the outbound request so application context survives the hop.
    /// `Arc<[u8]>` so per-request clones are refcount bumps. `None` when the
    /// inbound request carried no (valid) baggage.
    pub baggage: Option<Arc<[u8]>>,
    pub pipe_label: Option<Arc<[u8]>>,
    pub upstream_label: Option<Arc<[u8]>>,
    pub extra_labels: Labels,
    /// path-pattern parameters extracted by the mount router. `{id}` in the
    /// mount path becomes `path_params["id"]`. empty when the mount has no
    /// params or the request was not routed through one.
    pub path_params: BTreeMap<String, String>,
    /// per-request cancellation token. Only available under std (requires tokio).
    /// cancelled when the listener observes connection close, deadline expiry,
    /// or operator-initiated shutdown. child Pipes should call `.child()`
    /// for sub-operations they need to cancel independently.
    #[cfg(feature = "std")]
    pub cancel: Signal,
    /// per-call sidecar a wrapping RecordUpstream installs when the chain
    /// is being recorded. `None` when no recording is active — Pipes
    /// must handle that case explicitly to avoid silently dropping data.
    pub capture: Option<Arc<dyn CaptureContext>>,
    /// io_uring upgrade ticket. The io_uring listener mints this per
    /// request before dispatching; Pipe authors use it with
    /// `crate::pipe::upgrade::local_slots::install` to register a
    /// `LocalUpgradeHandler` that runs against the `!Send` socket
    /// after the listener writes the upgrade response head. `None` on
    /// the default (tokio) listener path where the Send-bound
    /// `Response.upgrade` field is the correct seam instead.
    pub local_upgrade_ticket: Option<u64>,
    /// Peer address for this request. When PROXY protocol is active
    /// this is the original client (not the load balancer); without
    /// PROXY it's the raw socket peer. `None` on substrate paths
    /// that don't carry an address (purely in-process Pipe calls,
    /// for example).
    pub peer: Option<PeerInfo>,
}

impl Default for RequestContext {
    fn default() -> Self {
        Self {
            telemetry: noop_telemetry(),
            deadline: None,
            trace_id: None,
            baggage: None,
            pipe_label: None,
            upstream_label: None,
            extra_labels: Labels::empty(),
            path_params: BTreeMap::new(),
            #[cfg(feature = "std")]
            cancel: never_cancel(),
            capture: None,
            local_upgrade_ticket: None,
            peer: None,
        }
    }
}

impl core::fmt::Debug for RequestContext {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut debug = formatter.debug_struct("RequestContext");
        #[cfg(feature = "std")]
        debug.field("deadline", &self.deadline);
        debug
            .field("trace_id", &self.trace_id)
            .field("baggage", &self.baggage)
            .field("pipe_label", &self.pipe_label)
            .field("upstream_label", &self.upstream_label)
            .field("extra_labels", &self.extra_labels.entries())
            .finish_non_exhaustive()
    }
}

impl RequestContext {
    #[must_use]
    pub fn with_telemetry(mut self, telemetry: TelemetryHandle) -> Self {
        self.telemetry = telemetry;
        self
    }

    #[cfg(feature = "std")]
    #[must_use]
    pub fn with_cancel(mut self, cancel: Signal) -> Self {
        self.cancel = cancel;
        self
    }

    /// Derive a child scope from this context's `cancel`. Firing the
    /// child does not affect the parent; firing the parent fires
    /// every descendant.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn child_signal(&self) -> Signal {
        self.cancel.child()
    }

    #[must_use]
    pub fn with_pipe_label(mut self, label: impl AsRef<[u8]>) -> Self {
        self.pipe_label = Some(Arc::from(label.as_ref()));
        self
    }

    #[must_use]
    pub fn with_upstream_label(mut self, label: impl AsRef<[u8]>) -> Self {
        self.upstream_label = Some(Arc::from(label.as_ref()));
        self
    }

    #[must_use]
    pub fn with_peer(mut self, peer: PeerInfo) -> Self {
        self.peer = Some(peer);
        self
    }

    /// Adopt the byte-form trace id + baggage payload a listener already
    /// established at ingress via
    /// `proxima_telemetry::propagation::establish_trace_context`. Proxima
    /// acts as a real hop: that establishment step adopts the inbound trace
    /// id (or originates a fresh trace id + span id when none is present)
    /// and PRESERVES the inbound span id rather than discarding it — a span
    /// later opened with `parent = self.traceparent()` therefore records the
    /// real inbound span as its `parent_span_id`, so the literal parent
    /// chain crosses the wire hop instead of only the shared `trace_id`.
    ///
    /// `trace_id` replaces the existing value whenever present (the
    /// establishment step always produces one under `std`); `baggage` only
    /// overwrites when present, so an already-set baggage value is not
    /// clobbered by a re-adoption carrying no inbound baggage header.
    ///
    /// proxima-pipe carries no trace identifier type (no header parsing, no
    /// generation) — that logic lives in the telemetry substrate's own
    /// propagation module, which already depends on this crate for
    /// [`HeaderList`]. This method is the seam that keeps the dependency
    /// one-directional: the listener (already depending on that substrate)
    /// does the establishment and hands this crate only the resulting bytes.
    pub fn adopt_trace_context(&mut self, trace_id: Option<Bytes>, baggage: Option<Bytes>) {
        if let Some(trace_id) = trace_id {
            self.trace_id = Some(Arc::from(trace_id.as_ref()));
        }
        if let Some(baggage) = baggage {
            self.baggage = Some(Arc::from(baggage.as_ref()));
        }
    }

    /// The raw W3C `traceparent` bytes this request carries (this hop's own
    /// span, restamped by
    /// [`adopt_trace_context`](Self::adopt_trace_context)) — ready to hand
    /// straight to `#[proxima::instrument(parent = ...)]` or
    /// `Recorder::span_from_traceparent`. `None` before `adopt_trace_context`
    /// runs, or on a request that predates trace context. This is the
    /// boundary seam: a handler passes `request.context.traceparent()` so its
    /// span continues the inbound trace instead of opening a fresh root —
    /// proxima has no ambient "current span" for it to inherit otherwise.
    #[must_use]
    pub fn traceparent(&self) -> Option<&[u8]> {
        self.trace_id.as_deref()
    }

    /// Stamp the carried `traceparent` + `baggage` onto outbound headers.
    /// Called at upstream egress. `insert_if_absent` semantics — a value the
    /// caller already set wins.
    pub fn inject_propagation(&self, headers: &mut HeaderList) {
        if let Some(traceparent) = self.trace_id.as_deref() {
            headers.insert_if_absent(HeaderName::Traceparent, traceparent);
        }
        if let Some(baggage) = self.baggage.as_deref() {
            headers.insert_if_absent(HeaderName::Baggage, baggage);
        }
    }

    #[must_use]
    pub fn metric_labels(&self, additional: &[(&str, &str)]) -> Labels {
        let mut pairs: Vec<(String, String)> = self.extra_labels.entries().to_vec();
        if let Some(pipe) = &self.pipe_label {
            pairs.push(("pipe".into(), String::from_utf8_lossy(pipe).into_owned()));
        }
        if let Some(upstream) = &self.upstream_label {
            pairs.push((
                "upstream".into(),
                String::from_utf8_lossy(upstream).into_owned(),
            ));
        }
        for (name, value) in additional {
            pairs.push(((*name).to_string(), (*value).to_string()));
        }
        let pair_refs: Vec<(&str, &str)> = pairs
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        Labels::from_pairs(&pair_refs)
    }
}

#[derive(Debug)]
pub struct Request<P> {
    /// HTTP method. Standard methods are unit variants (zero-alloc construct +
    /// integer compare); a non-standard method carries its wire bytes. See
    /// [`Method`](crate::pipe::method::Method).
    pub method: crate::pipe::method::Method,
    /// URL path bytes. percent-decoded form is in here; the listener
    /// boundary is responsible for normalization.
    pub path: Bytes,
    pub query: HeaderList,
    pub metadata: HeaderList,
    /// Buffered request payload. `Bytes` is the default wire case.
    pub payload: P,
    /// Streamed request body (uploads, WebSocket-inbound relay). `None`
    /// for the buffered 80% case. Request trailers fold into `metadata`
    /// at chunked-decode end; cancellation is `context.cancel`.
    pub stream: Option<RequestStream>,
    pub context: RequestContext,
}

impl<P: Clone> Clone for Request<P> {
    /// Clone a **buffered** request so a caller can build once and re-send many
    /// times (a load generator). Every field is a cheap copy — `Method`/`Bytes`
    /// are refcount bumps, `RequestContext` is `Arc`-backed — so a clone-per-send
    /// avoids the per-send `path` copy + `RequestContext::default()` of a fresh
    /// build. A streamed body is **not** carried: the clone has `stream: None`
    /// and the buffered `payload` (a live stream is single-use, not clonable).
    fn clone(&self) -> Self {
        debug_assert!(
            self.stream.is_none(),
            "Request::clone drops a live stream; clone only buffered requests"
        );
        Self {
            method: self.method.clone(),
            path: self.path.clone(),
            query: self.query.clone(),
            metadata: self.metadata.clone(),
            payload: self.payload.clone(),
            stream: None,
            context: self.context.clone(),
        }
    }
}

impl Request<Bytes> {
    #[must_use]
    pub fn builder() -> RequestBuilder {
        RequestBuilder::default()
    }

    /// Materialize the payload as `Bytes`, draining a streamed body if one
    /// is present, and return `(self, bytes)` with the buffered bytes
    /// installed on `self.payload` for reuse.
    pub async fn body_bytes(self) -> Result<(Self, Bytes), ProximaError> {
        let mut this = self;
        if let Some(stream) = this.stream.take() {
            // Grab the trailers slot (an Arc) before `collect` consumes the
            // stream; the chunked decoder populates it at body-end, so it's
            // readable once the drain completes.
            #[cfg(feature = "std")]
            let trailers_slot = stream.trailers_slot().cloned();
            let bytes = stream.collect().await?;
            // Request trailers fold into `metadata` at chunked-decode end.
            #[cfg(feature = "std")]
            if let Some(slot) = trailers_slot
                && let Ok(guard) = slot.lock()
                && let Some(trailers) = guard.clone()
            {
                for (name, value) in &trailers {
                    this.metadata
                        .insert(Bytes::clone(name), Bytes::clone(value));
                }
            }
            this.payload = Bytes::clone(&bytes);
            return Ok((this, bytes));
        }
        let echo = Bytes::clone(&this.payload);
        Ok((this, echo))
    }

    /// Uniform chunk-stream view of the request payload: the streamed body
    /// if present, else a one-chunk stream of the buffered bytes. Lets a
    /// listener pump any request body without matching on the shape.
    #[must_use]
    pub fn into_chunk_stream(self) -> ChunkStream {
        match self.stream {
            Some(stream) => stream.into_chunk_stream(),
            None => {
                let bytes = self.payload;
                alloc::boxed::Box::pin(futures::stream::once(async move { Ok(bytes) }))
            }
        }
    }

    /// Drain a [`PartSource`](crate::pipe::part::PartSource) into an owned
    /// `Request` — the opt-in materialization step of the `Part` model
    /// (`docs/proxima-pipe/part-source-sink-design.md`). Every
    /// `Part::Method` / `Part::Path` / `Part::Header` is copied into owned
    /// storage and every `Part::Chunk` accumulates into the payload; this is
    /// the allocation cost a handler opts INTO by calling this instead of
    /// stepping the source itself. `Part::End` stops the drain; a source
    /// that never yields it is drained to exhaustion. `query` and
    /// `RequestContext` are not populated — the `Part` model carries neither.
    #[cfg(feature = "part-source")]
    #[must_use]
    pub fn from_source(source: &mut impl crate::pipe::part::PartSource) -> Self {
        use crate::pipe::part::Part;

        let mut method = crate::pipe::method::Method::default();
        let mut path = Bytes::new();
        let mut metadata = HeaderList::new();
        let mut chunks: Vec<u8> = Vec::new();

        while let Some(part) = source.next() {
            match part {
                Part::Method(bytes) => method = crate::pipe::method::Method::from_bytes(bytes),
                Part::Path(bytes) => path = Bytes::copy_from_slice(bytes),
                Part::Header(name, value) => {
                    metadata.insert(name, value);
                }
                Part::Chunk(bytes) => chunks.extend_from_slice(bytes),
                Part::End => break,
            }
        }

        Self {
            method,
            path,
            query: HeaderList::new(),
            metadata,
            payload: Bytes::from(chunks),
            stream: None,
            context: RequestContext::default(),
        }
    }
}

#[derive(Default)]
pub struct RequestBuilder {
    method: Option<crate::pipe::method::Method>,
    path: Option<Bytes>,
    query: HeaderList,
    metadata: HeaderList,
    payload: Option<Bytes>,
    stream: Option<RequestStream>,
    context: RequestContext,
}

impl core::fmt::Debug for RequestBuilder {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("RequestBuilder")
            .field("method", &self.method)
            .field("path", &self.path)
            .field("query", &self.query)
            .field("metadata", &self.metadata)
            .field("has_payload", &self.payload.is_some())
            .field("context", &self.context)
            .finish()
    }
}

impl RequestBuilder {
    #[must_use]
    pub fn method(mut self, method: impl Into<crate::pipe::method::Method>) -> Self {
        self.method = Some(method.into());
        self
    }

    #[must_use]
    pub fn path(mut self, path: impl crate::pipe::header_list::IntoHeaderBytes) -> Self {
        self.path = Some(path.into_header_bytes());
        self
    }

    #[must_use]
    pub fn header(
        mut self,
        name: impl crate::pipe::header_list::IntoHeaderBytes,
        value: impl crate::pipe::header_list::IntoHeaderBytes,
    ) -> Self {
        self.metadata.insert(name, value);
        self
    }

    #[must_use]
    pub fn query_param(
        mut self,
        name: impl crate::pipe::header_list::IntoHeaderBytes,
        value: impl crate::pipe::header_list::IntoHeaderBytes,
    ) -> Self {
        self.query.insert(name, value);
        self
    }

    #[must_use]
    pub fn payload(mut self, payload: impl Into<Bytes>) -> Self {
        self.payload = Some(payload.into());
        self
    }

    #[must_use]
    pub fn body(mut self, body: impl Into<Bytes>) -> Self {
        self.payload = Some(body.into());
        self
    }

    /// Attach a streamed request body.
    #[must_use]
    pub fn stream(mut self, stream: RequestStream) -> Self {
        self.stream = Some(stream);
        self
    }

    #[must_use]
    pub fn context(mut self, context: RequestContext) -> Self {
        self.context = context;
        self
    }

    #[must_use]
    pub fn telemetry(mut self, telemetry: TelemetryHandle) -> Self {
        self.context.telemetry = telemetry;
        self
    }

    pub fn build(self) -> Result<Request<Bytes>, ProximaError> {
        let method = self
            .method
            .ok_or_else(|| ProximaError::Config("request method required".into()))?;
        let path = self
            .path
            .ok_or_else(|| ProximaError::Config("request path required".into()))?;
        Ok(Request {
            method,
            path,
            query: self.query,
            metadata: self.metadata,
            payload: self.payload.unwrap_or_default(),
            stream: self.stream,
            context: self.context,
        })
    }
}

#[derive(Debug)]
pub struct Response<P> {
    pub status: u16,
    pub metadata: HeaderList,
    /// Buffered response payload. `Bytes` is the default wire case.
    pub payload: P,
    /// Streamed response body. `None` for the buffered 80% case.
    /// Response trailers (RFC 7230 §4.1.2) ride on the `ResponseStream`;
    /// the listener emits them after the final 0-length chunk.
    pub stream: Option<ResponseStream>,
    /// When set, the listener hands the raw socket to this handler
    /// after writing the response head. Body framing is suppressed
    /// (no `Transfer-Encoding`, no body bytes) — the next bytes on
    /// the wire belong to the handler's protocol (CONNECT tunnel,
    /// WebSocket frames, h2c SETTINGS, …).
    #[cfg(feature = "std")]
    pub upgrade: Option<UpgradeHandler>,
}

impl<P> Response<P> {
    /// Construct a typed response with an explicit payload value.
    #[must_use]
    pub fn typed(status: u16, payload: P) -> Self {
        Self {
            status,
            metadata: HeaderList::new(),
            payload,
            stream: None,
            #[cfg(feature = "std")]
            upgrade: None,
        }
    }

    /// Attach a streamed response body.
    #[must_use]
    pub fn with_stream(mut self, stream: ResponseStream) -> Self {
        self.stream = Some(stream);
        self
    }

    /// Attach an upgrade handler. The listener writes this response's
    /// head, then hands the raw socket to the handler. Typically
    /// paired with status 200 (CONNECT) or 101 (Upgrade).
    #[cfg(feature = "std")]
    #[must_use]
    pub fn with_upgrade(mut self, handler: UpgradeHandler) -> Self {
        self.upgrade = Some(handler);
        self
    }

    #[must_use]
    pub fn with_header(
        mut self,
        name: impl crate::pipe::header_list::IntoHeaderBytes,
        value: impl crate::pipe::header_list::IntoHeaderBytes,
    ) -> Self {
        self.metadata.insert(name, value);
        self
    }

    #[must_use]
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    #[must_use]
    pub fn is_no_data(&self) -> bool {
        self.status == 204
    }
}

impl Response<Bytes> {
    #[must_use]
    pub fn new(status: u16) -> Self {
        Self {
            status,
            metadata: HeaderList::new(),
            payload: Bytes::new(),
            stream: None,
            #[cfg(feature = "std")]
            upgrade: None,
        }
    }

    #[must_use]
    pub fn with_payload(mut self, payload: impl Into<Bytes>) -> Self {
        self.payload = payload.into();
        self
    }

    #[must_use]
    pub fn with_body(mut self, body: impl Into<Bytes>) -> Self {
        self.payload = body.into();
        self
    }

    /// Construct a streamed response (status 200).
    #[must_use]
    pub fn streamed(stream: ResponseStream) -> Self {
        Self::new(200).with_stream(stream)
    }

    #[must_use]
    pub fn ok(payload: impl Into<Bytes>) -> Self {
        Self::new(200).with_payload(payload)
    }

    #[must_use]
    pub fn not_found() -> Self {
        Self::new(404)
    }

    #[must_use]
    pub fn no_data() -> Self {
        Self::new(204)
    }

    /// Uniform chunk-stream view of the response payload: the streamed body
    /// if present, else a one-chunk stream of the buffered bytes. Lets a
    /// listener pump any response without matching on the shape.
    #[must_use]
    pub fn into_chunk_stream(self) -> ChunkStream {
        match self.stream {
            Some(stream) => stream.into_chunk_stream(),
            None => {
                let bytes = self.payload;
                alloc::boxed::Box::pin(futures::stream::once(async move { Ok(bytes) }))
            }
        }
    }

    /// Drain the response payload to `Bytes` — the streamed body collected,
    /// else the buffered bytes. Not cancellation-aware; for that,
    /// take `self.stream` and call `ResponseStream::collect(Some(&token))`.
    pub async fn collect_body(self) -> Result<Bytes, ProximaError> {
        match self.stream {
            Some(stream) => {
                stream
                    .collect(
                        #[cfg(feature = "std")]
                        None,
                    )
                    .await
            }
            None => Ok(self.payload),
        }
    }
}

// `#[proxima::test]` pulls in the `proxima` dev-dependency, which the
// loom build keeps out of the graph (see
// `[target.'cfg(not(loom))'.dev-dependencies]` in Cargo.toml); these
// tests are unrelated to the Notify/watch loom protocol.
#[cfg(all(test, not(loom)))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn builder_requires_method() {
        let outcome = Request::builder().path("/foo").build();
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[test]
    fn builder_requires_path() {
        let outcome = Request::builder().method("GET").build();
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[test]
    fn builder_assembles_request_with_required_fields() {
        let request = Request::builder()
            .method("GET")
            .path("/users/42")
            .header("accept", "application/json")
            .query_param("limit", "10")
            .build()
            .expect("builder should succeed");
        assert_eq!(request.method, "GET");
        assert_eq!(request.path, "/users/42");
        assert_eq!(request.metadata.get_str("accept"), Some("application/json"));
        assert_eq!(request.query.get_str("limit"), Some("10"));
    }

    #[proxima::test]
    async fn body_bytes_returns_collected_bytes_and_replaces_body() {
        let request = Request::builder()
            .method("POST")
            .path("/")
            .body("hello")
            .build()
            .expect("builder should succeed");
        let (request, bytes) = request
            .body_bytes()
            .await
            .expect("body_bytes should succeed");
        assert_eq!(&bytes[..], b"hello");
        assert_eq!(
            &request.payload[..],
            b"hello",
            "echo payload should be available for reuse"
        );
    }

    #[test]
    fn response_helpers_set_status() {
        assert_eq!(Response::ok("body").status, 200);
        assert_eq!(Response::not_found().status, 404);
        assert_eq!(Response::no_data().status, 204);
    }

    #[test]
    fn is_success_only_2xx() {
        assert!(Response::new(200).is_success());
        assert!(Response::new(299).is_success());
        assert!(!Response::new(199).is_success());
        assert!(!Response::new(300).is_success());
    }

    #[test]
    fn is_no_data_only_204() {
        assert!(Response::new(204).is_no_data());
        assert!(!Response::new(200).is_no_data());
    }

    #[test]
    fn default_context_is_noop_telemetry_with_empty_labels() {
        let context = RequestContext::default();
        assert!(context.deadline.is_none());
        assert!(context.trace_id.is_none());
        assert!(context.pipe_label.is_none());
        assert!(context.extra_labels.is_empty());
    }

    // Header parsing + trace id generation now live in
    // `proxima_telemetry::propagation::establish_trace_context` (see its
    // tests in that crate); these tests exercise the byte-only seam
    // proxima-pipe still owns: adopting already-established bytes and
    // stamping them back onto outbound headers.

    #[test]
    fn adopt_trace_context_sets_trace_id_and_baggage() {
        let mut context = RequestContext::default();
        context.adopt_trace_context(
            Some(Bytes::from_static(
                b"00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            )),
            Some(Bytes::from_static(b"userId=alice")),
        );
        assert_eq!(
            context.trace_id.as_deref(),
            Some(b"00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01".as_slice())
        );
        assert_eq!(context.baggage.as_deref(), Some(b"userId=alice".as_slice()));
    }

    #[test]
    fn adopt_trace_context_does_not_clobber_baggage_when_absent() {
        let mut context = RequestContext::default();
        context.adopt_trace_context(
            Some(Bytes::from_static(b"trace-a")),
            Some(Bytes::from_static(b"k=v")),
        );
        context.adopt_trace_context(Some(Bytes::from_static(b"trace-b")), None);
        assert_eq!(context.trace_id.as_deref(), Some(b"trace-b".as_slice()));
        assert_eq!(
            context.baggage.as_deref(),
            Some(b"k=v".as_slice()),
            "baggage from an earlier adoption survives a re-adoption with no baggage"
        );
    }

    #[test]
    fn inject_propagation_emits_context_traceparent_and_baggage() {
        let mut context = RequestContext::default();
        context.adopt_trace_context(
            Some(Bytes::from_static(
                b"00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            )),
            Some(Bytes::from_static(b"k=v")),
        );

        let mut outbound = HeaderList::new();
        context.inject_propagation(&mut outbound);
        assert_eq!(
            outbound.get_str("traceparent"),
            Some("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01")
        );
        assert_eq!(outbound.get_str("baggage"), Some("k=v"));
    }

    #[test]
    fn metric_labels_includes_pipe_and_upstream_when_set() {
        let context = RequestContext::default()
            .with_pipe_label("echo-cached")
            .with_upstream_label("origin");
        let labels = context.metric_labels(&[("status_class", "2xx")]);
        let entries = labels.entries();
        assert!(
            entries
                .iter()
                .any(|(name, value)| name == "pipe" && value == "echo-cached")
        );
        assert!(
            entries
                .iter()
                .any(|(name, value)| name == "upstream" && value == "origin")
        );
        assert!(
            entries
                .iter()
                .any(|(name, value)| name == "status_class" && value == "2xx")
        );
    }

    #[test]
    fn builder_telemetry_overrides_default_noop() {
        use crate::pipe::telemetry_surface::Telemetry;
        use core::sync::atomic::{AtomicU64, Ordering};

        struct CountingTelemetry {
            count: AtomicU64,
        }

        impl Telemetry for CountingTelemetry {
            fn counter_inc(&self, _metric: &str, _labels: &Labels, by: u64) {
                self.count.fetch_add(by, Ordering::Relaxed);
            }
            fn gauge_set(&self, _: &str, _: &Labels, _: i64) {}
            fn histogram_record(&self, _: &str, _: &Labels, _: f64) {}
        }

        let counting = Arc::new(CountingTelemetry {
            count: AtomicU64::new(0),
        });
        let telemetry: TelemetryHandle = counting.clone();
        let request = Request::builder()
            .method("GET")
            .path("/")
            .telemetry(telemetry)
            .build()
            .expect("builder should succeed");
        request
            .context
            .telemetry
            .counter_inc("test_counter", &Labels::empty(), 1);
        assert_eq!(counting.count.load(Ordering::Relaxed), 1);
    }

    /// Request and Response can be constructed with no std-only features.
    /// This test exercises the alloc-only code paths available even without
    /// the std feature.
    #[test]
    fn request_and_response_constructible_under_alloc_only() {
        let request = Request::builder()
            .method("POST")
            .path("/items")
            .header("content-type", "application/json")
            .build()
            .expect("request should build");
        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/items");

        let response = Response::new(201)
            .with_body("created")
            .with_header("x-id", "42");
        assert_eq!(response.status, 201);
        assert_eq!(response.metadata.get_str("x-id"), Some("42"));
        assert!(response.is_success());
    }

    /// RequestContext can be constructed and cloned without std features.
    #[test]
    fn request_context_clone_and_peer_label_work_without_std() {
        let context = RequestContext::default()
            .with_pipe_label("my-pipe")
            .with_upstream_label("my-upstream");
        let cloned = context.clone();
        assert_eq!(cloned.pipe_label, context.pipe_label);
        assert_eq!(cloned.upstream_label, context.upstream_label);
    }

    #[test]
    fn deadline_field_is_none_by_default() {
        let context = RequestContext::default();
        assert!(context.deadline.is_none());
    }

    /// Under std, a cancel signal can be attached and child scopes derived.
    #[cfg(feature = "std")]
    #[test]
    fn cancel_signal_child_derivation_under_std() {
        let context = RequestContext::default();
        let child = context.child_signal();
        // child scope is not fired when parent is not
        assert!(!child.is_fired());
    }
}
