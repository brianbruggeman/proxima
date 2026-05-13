use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock, Weak};
use std::task::{Context, Poll};
use std::time::Instant;

use bon::Builder;
use bytes::Bytes;
use conflaguration::{Settings, Validate, ValidationMessage};
use futures::{FutureExt, Stream, select_biased};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use time::OffsetDateTime;
use proxima_primitives::sync::mpsc;

use crate::body::{ChunkStream, RequestStream, ResponseStream};
use crate::capture_surface::CaptureContext;
use crate::error::ProximaError;
use proxima_primitives::pipe::{Pipe, SendPipe};

use crate::pipe::{Handler, PipeHandle, ThreadLocalHandler, ThreadLocalPipeHandle, into_handle};
use crate::pipe_factory::{PipeFactory, PipeFactoryRegistry};
use crate::recording::LiveCaptureContext;
use crate::recording::event::{
    HttpEvent, InteractionId, ProtocolEvent, RecordMeta, RecordingEvent, RequestHeader,
};
use crate::recording::sink::DynRecordingSink;
use crate::recording::{
    AccumulatingSink, DeferredRuntime, FormatKind, LazyFanOut, SinkSpec, deferred_runtime,
};
use crate::request::{Request, Response};
use crate::runtime::{CoreId, Runtime};

// memoized sender to a RecordUpstream's single, long-lived drainer.
type DrainerCell = Arc<OnceLock<mpsc::UnboundedSender<RecordingEvent>>>;

/// Proxy that tees every (request, response) interaction into a
/// recording sink. Per-chunk events preserve inter-chunk timing for
/// replay; sink writes drain on a background task so the request hot
/// path doesn't block on I/O.
///
/// Generic over the inner handle: `RecordUpstream<PipeHandle>` impls
/// `Handler`; `RecordUpstream<ThreadLocalPipeHandle>` impls
/// `ThreadLocalHandler`. Dispatch unifies through the
/// `ThreadLocalHandler` blanket so a single body pipes both paths.
pub struct RecordUpstream<Inner = PipeHandle> {
    label: String,
    inner: Inner,
    sink: DynRecordingSink,
    pipe_label: String,
    protocol: String,
    // armed by the App at serve; once set, the drainer spawns once instead
    // of per call (see `instance_drainer_sender`).
    spigot: DeferredRuntime,
    drainer: DrainerCell,
}

impl<Inner> RecordUpstream<Inner> {
    #[must_use]
    pub fn new(
        label: impl Into<String>,
        inner: Inner,
        sink: DynRecordingSink,
        pipe_label: impl Into<String>,
    ) -> Self {
        Self {
            label: label.into(),
            inner,
            sink,
            pipe_label: pipe_label.into(),
            protocol: "http".into(),
            spigot: deferred_runtime(),
            drainer: Arc::new(OnceLock::new()),
        }
    }

    /// This upstream's label, set at construction (TARGET 3 — served-Handler
    /// naming now lives at the mount-site label, not the handle).
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    #[must_use]
    pub fn with_protocol(mut self, protocol: impl Into<String>) -> Self {
        self.protocol = protocol.into();
        self
    }

    /// Inject the runtime spigot so the drainer spawns once on the runtime
    /// instead of per call. Unarmed spigots (tests, direct construction)
    /// keep the legacy per-call drainer.
    #[must_use]
    pub fn with_runtime(mut self, spigot: DeferredRuntime) -> Self {
        self.spigot = spigot;
        self
    }
}

#[derive(Debug, Clone, Copy)]
enum Phase {
    Request,
    Response,
}

// spawns ONE long-lived drainer on the runtime the first time it's called and
// memoizes its sender; every later call reuses it — no spawn, no channel, per
// call.
fn instance_drainer_sender(
    drainer: &DrainerCell,
    runtime: &Arc<dyn Runtime>,
    sink: &DynRecordingSink,
) -> mpsc::UnboundedSender<RecordingEvent> {
    // spawn_on_core, not spawn_on_current_core: the recording pipe is driven
    // from arbitrary call sites, not necessarily a runtime worker thread.
    drainer
        .get_or_init(|| {
            let (sender, receiver) = mpsc::unbounded_channel::<RecordingEvent>();
            if let Err(error) =
                runtime.spawn_on_core(CoreId(0), Box::pin(drain_forever(receiver, sink.clone())))
            {
                tracing::error!(error = ?error, "recording drainer spawn failed");
            }
            sender
        })
        .clone()
}

// append each event, drain any burst already queued, then flush once caught
// up — durability amortized across the burst instead of once per call.
async fn drain_forever(
    mut receiver: mpsc::UnboundedReceiver<RecordingEvent>,
    sink: DynRecordingSink,
) {
    while let Some(event) = receiver.recv().await {
        if let Err(error) = sink.append(event).await {
            tracing::error!(error = %error, "recording sink append failed");
        }
        // drain any burst already queued without waiting for more: a
        // `now_or_never` immediate poll stands in for tokio's `try_recv`
        // (proxima's mpsc doesn't shim that non-blocking probe — see its
        // module doc's "Non-coverage" list).
        while let Some(Some(event)) = receiver.recv().now_or_never() {
            if let Err(error) = sink.append(event).await {
                tracing::error!(error = %error, "recording sink append failed");
            }
        }
        if let Err(error) = sink.flush().await {
            tracing::error!(error = %error, "recording sink flush failed");
        }
    }
    if let Err(error) = sink.flush().await {
        tracing::error!(error = %error, "recording sink flush failed");
    }
}

impl<Inner> SendPipe for RecordUpstream<Inner>
where
    Inner: Handler + Clone,
{
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        record_dispatch(
            self.inner.clone(),
            request,
            self.sink.clone(),
            self.pipe_label.clone(),
            self.protocol.clone(),
            self.spigot.clone(),
            self.drainer.clone(),
        )
    }
}


impl Pipe for RecordUpstream<ThreadLocalPipeHandle> {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        record_dispatch_local(
            self.inner.clone(),
            request,
            self.sink.clone(),
            self.pipe_label.clone(),
            self.protocol.clone(),
            self.spigot.clone(),
            self.drainer.clone(),
        )
    }
}


/// Shared body for both Handler and ThreadLocalHandler impls. Dispatches
/// the inner call via `ThreadLocalHandler::call` — the blanket impl makes
/// every `Handler` automatically a `ThreadLocalHandler`, so a Send Inner
/// still produces a Send future here and an Rc-based Inner produces a
/// !Send one.
async fn record_dispatch<Inner>(
    inner: Inner,
    request: Request<Bytes>,
    sink: DynRecordingSink,
    pipe_label: String,
    protocol: String,
    spigot: DeferredRuntime,
    drainer: DrainerCell,
) -> Result<Response<Bytes>, ProximaError>
where
    Inner: Handler + Clone,
{
    let cancel = request.context.cancel.clone();
    let id = InteractionId::new();
    let ts_start = OffsetDateTime::now_utc();
    let started = Instant::now();

    let sender = match spigot.get() {
        // armed: one drainer per RecordUpstream, spawned once on the
        // injected runtime — not a `tokio::spawn` per call.
        Some(runtime) => instance_drainer_sender(&drainer, runtime, &sink),
        // unarmed (tests / direct construction): the legacy per-call
        // drainer on the ambient tokio runtime, cancellable with the request.
        None => {
            let (sender, mut receiver) = mpsc::unbounded_channel::<RecordingEvent>();
            let sink_for_task = sink.clone();
            let drainer_cancel = cancel.clone();
            // no injected runtime to spawn_on_core against: a dedicated OS
            // thread driving `block_on` gives real background progress
            // without requiring any particular async runtime (mirrors
            // `proxima_primitives::sync::task`'s portable `JoinSet`).
            std::thread::spawn(move || {
                futures::executor::block_on(async move {
                    loop {
                        select_biased! {
                            _ = drainer_cancel.fired().fuse() => break,
                            event = receiver.recv().fuse() => match event {
                                Some(event) => {
                                    if let Err(error) = sink_for_task.append(event).await {
                                        tracing::error!(error = %error, "recording sink append failed");
                                    }
                                }
                                None => break,
                            },
                        }
                    }
                    if let Err(error) = sink_for_task.flush().await {
                        tracing::error!(error = %error, "recording sink flush failed");
                    }
                });
            });
            sender
        }
    };

    let Request {
        method,
        path,
        query,
        metadata,
        payload,
        stream,
        mut context,
        ..
    } = request;
    let req_chunks: ChunkStream = match stream {
        Some(request_stream) => request_stream.into_chunk_stream(),
        None => Box::pin(futures::stream::once(async move { Ok(payload) })),
    };
    let capture = Arc::new(LiveCaptureContext::new());
    context.capture = Some(capture.clone() as Arc<dyn CaptureContext>);

    let req_headers_for_record: std::collections::BTreeMap<String, String> = metadata
        .iter()
        .map(|(name, value)| {
            (
                String::from_utf8_lossy(name).into_owned(),
                String::from_utf8_lossy(value).into_owned(),
            )
        })
        .collect();
    let req_query_for_record: std::collections::BTreeMap<String, String> = query
        .iter()
        .map(|(name, value)| {
            (
                String::from_utf8_lossy(name).into_owned(),
                String::from_utf8_lossy(value).into_owned(),
            )
        })
        .collect();
    // `protocol` is implied by ProtocolEvent::Http; the previous `protocol: String` field is dropped.
    let _ = protocol;
    let _ = sender.send(RecordingEvent {
        id,
        ts_ms: 0,
        parent: None,
        event: ProtocolEvent::Http(HttpEvent::Started {
            ts: ts_start,
            pipe: pipe_label,
            request: RequestHeader {
                method: String::from_utf8_lossy(method.as_bytes()).into_owned(),
                path: String::from_utf8_lossy(&path).into_owned(),
                headers: req_headers_for_record,
                query: req_query_for_record,
            },
            meta: None,
        }),
    });

    let req_body = wrap_chunked(
        req_chunks,
        started,
        id,
        sender.clone(),
        Phase::Request,
        capture.clone(),
    );

    let inbound = Request {
        method,
        path,
        query,
        metadata,
        payload: Bytes::new(),
        stream: Some(RequestStream::from_chunk_stream(req_body)),
        context,
    };
    let response = SendPipe::call(&inner, inbound).await?;

    let resp_started_ms = started.elapsed().as_millis() as u64;
    let header_pairs: Vec<(String, String)> = response
        .metadata
        .iter()
        .map(|(name, value)| {
            (
                String::from_utf8_lossy(name).into_owned(),
                String::from_utf8_lossy(value).into_owned(),
            )
        })
        .collect();
    let _ = sender.send(RecordingEvent {
        id,
        ts_ms: resp_started_ms,
        parent: None,
        event: ProtocolEvent::Http(HttpEvent::ResponseStarted {
            status: response.status,
            headers: header_pairs,
        }),
    });

    let status = response.status;
    let headers = response.metadata.clone();
    let resp_chunks = response.into_chunk_stream();
    let resp_body = wrap_chunked(resp_chunks, started, id, sender, Phase::Response, capture);
    let mut rebuilt =
        Response::new(status).with_stream(ResponseStream::from_chunk_stream(resp_body));
    for (name, value) in headers {
        rebuilt = rebuilt.with_header(name, value);
    }
    Ok(rebuilt)
}

// !Send variant for `impl ThreadLocalHandler for RecordUpstream<ThreadLocalPipeHandle>`.
// Identical body to `record_dispatch` modulo the dispatch trait. Lives separately
// because the previous `impl<T: Handler> ThreadLocalHandler for T` blanket was removed
// during the proxima-pipe extraction (coherence issue with downstream wrappers).
async fn record_dispatch_local<Inner>(
    inner: Inner,
    request: Request<Bytes>,
    sink: DynRecordingSink,
    pipe_label: String,
    protocol: String,
    spigot: DeferredRuntime,
    drainer: DrainerCell,
) -> Result<Response<Bytes>, ProximaError>
where
    Inner: ThreadLocalHandler + Clone,
{
    let cancel = request.context.cancel.clone();
    let id = InteractionId::new();
    let ts_start = OffsetDateTime::now_utc();
    let started = Instant::now();

    let sender = match spigot.get() {
        // armed: one drainer per RecordUpstream, spawned once on the
        // injected runtime — not a `tokio::spawn` per call.
        Some(runtime) => instance_drainer_sender(&drainer, runtime, &sink),
        // unarmed (tests / direct construction): the legacy per-call
        // drainer on the ambient tokio runtime, cancellable with the request.
        None => {
            let (sender, mut receiver) = mpsc::unbounded_channel::<RecordingEvent>();
            let sink_for_task = sink.clone();
            let drainer_cancel = cancel.clone();
            // no injected runtime to spawn_on_core against: a dedicated OS
            // thread driving `block_on` gives real background progress
            // without requiring any particular async runtime (mirrors
            // `proxima_primitives::sync::task`'s portable `JoinSet`).
            std::thread::spawn(move || {
                futures::executor::block_on(async move {
                    loop {
                        select_biased! {
                            _ = drainer_cancel.fired().fuse() => break,
                            event = receiver.recv().fuse() => match event {
                                Some(event) => {
                                    if let Err(error) = sink_for_task.append(event).await {
                                        tracing::error!(error = %error, "recording sink append failed");
                                    }
                                }
                                None => break,
                            },
                        }
                    }
                    if let Err(error) = sink_for_task.flush().await {
                        tracing::error!(error = %error, "recording sink flush failed");
                    }
                });
            });
            sender
        }
    };

    let Request {
        method,
        path,
        query,
        metadata,
        payload,
        stream,
        mut context,
        ..
    } = request;
    let req_chunks: ChunkStream = match stream {
        Some(request_stream) => request_stream.into_chunk_stream(),
        None => Box::pin(futures::stream::once(async move { Ok(payload) })),
    };
    let capture = Arc::new(LiveCaptureContext::new());
    context.capture = Some(capture.clone() as Arc<dyn CaptureContext>);

    let req_headers_for_record: std::collections::BTreeMap<String, String> = metadata
        .iter()
        .map(|(name, value)| {
            (
                String::from_utf8_lossy(name).into_owned(),
                String::from_utf8_lossy(value).into_owned(),
            )
        })
        .collect();
    let req_query_for_record: std::collections::BTreeMap<String, String> = query
        .iter()
        .map(|(name, value)| {
            (
                String::from_utf8_lossy(name).into_owned(),
                String::from_utf8_lossy(value).into_owned(),
            )
        })
        .collect();
    let _ = protocol;
    let _ = sender.send(RecordingEvent {
        id,
        ts_ms: 0,
        parent: None,
        event: ProtocolEvent::Http(HttpEvent::Started {
            ts: ts_start,
            pipe: pipe_label,
            request: RequestHeader {
                method: String::from_utf8_lossy(method.as_bytes()).into_owned(),
                path: String::from_utf8_lossy(&path).into_owned(),
                headers: req_headers_for_record,
                query: req_query_for_record,
            },
            meta: None,
        }),
    });

    let req_body = wrap_chunked(
        req_chunks,
        started,
        id,
        sender.clone(),
        Phase::Request,
        capture.clone(),
    );

    let inbound = Request {
        method,
        path,
        query,
        metadata,
        payload: Bytes::new(),
        stream: Some(RequestStream::from_chunk_stream(req_body)),
        context,
    };
    let response = Pipe::call(&inner, inbound).await?;

    let resp_started_ms = started.elapsed().as_millis() as u64;
    let header_pairs: Vec<(String, String)> = response
        .metadata
        .iter()
        .map(|(name, value)| {
            (
                String::from_utf8_lossy(name).into_owned(),
                String::from_utf8_lossy(value).into_owned(),
            )
        })
        .collect();
    let _ = sender.send(RecordingEvent {
        id,
        ts_ms: resp_started_ms,
        parent: None,
        event: ProtocolEvent::Http(HttpEvent::ResponseStarted {
            status: response.status,
            headers: header_pairs,
        }),
    });

    let status = response.status;
    let headers = response.metadata.clone();
    let resp_chunks = response.into_chunk_stream();
    let resp_body = wrap_chunked(resp_chunks, started, id, sender, Phase::Response, capture);
    let mut rebuilt =
        Response::new(status).with_stream(ResponseStream::from_chunk_stream(resp_body));
    for (name, value) in headers {
        rebuilt = rebuilt.with_header(name, value);
    }
    Ok(rebuilt)
}

fn wrap_chunked(
    inner: ChunkStream,
    started: Instant,
    id: InteractionId,
    sender: mpsc::UnboundedSender<RecordingEvent>,
    phase: Phase,
    capture: Arc<LiveCaptureContext>,
) -> ChunkStream {
    Box::pin(ChunkRecorder {
        inner,
        started,
        id,
        sender: Some(sender),
        phase,
        end_emitted: false,
        capture,
    })
}

struct ChunkRecorder {
    inner: ChunkStream,
    started: Instant,
    id: InteractionId,
    sender: Option<mpsc::UnboundedSender<RecordingEvent>>,
    phase: Phase,
    end_emitted: bool,
    capture: Arc<LiveCaptureContext>,
}

impl ChunkRecorder {
    fn elapsed_ms(&self) -> u64 {
        self.started.elapsed().as_millis() as u64
    }

    fn emit_chunk(&mut self, chunk: &Bytes) {
        if let Some(sender) = self.sender.as_ref() {
            let metadata = self.capture.drain();
            let ts_ms = self.elapsed_ms();
            let http_event = match self.phase {
                Phase::Request => HttpEvent::RequestChunk {
                    data: chunk.clone(),
                    metadata,
                },
                Phase::Response => HttpEvent::ResponseChunk {
                    data: chunk.clone(),
                    metadata,
                },
            };
            let _ = sender.send(RecordingEvent {
                id: self.id,
                ts_ms,
                parent: None,
                event: ProtocolEvent::Http(http_event),
            });
        }
    }

    fn emit_end(&mut self) {
        if self.end_emitted {
            return;
        }
        self.end_emitted = true;
        if let Some(sender) = self.sender.take() {
            let ts_end = self.elapsed_ms();
            let http_event = match self.phase {
                Phase::Request => HttpEvent::RequestEnded,
                Phase::Response => HttpEvent::Ended {
                    latency_ms: ts_end,
                    meta: RecordMeta::default(),
                },
            };
            let _ = sender.send(RecordingEvent {
                id: self.id,
                ts_ms: ts_end,
                parent: None,
                event: ProtocolEvent::Http(http_event),
            });
        }
    }
}

impl Drop for ChunkRecorder {
    fn drop(&mut self) {
        // emit end-of-interaction even if the consumer drops mid-stream.
        self.emit_end();
    }
}

impl Stream for ChunkRecorder {
    type Item = Result<Bytes, ProximaError>;

    fn poll_next(mut self: Pin<&mut Self>, ctx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.inner).poll_next(ctx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => {
                self.emit_end();
                Poll::Ready(None)
            }
            Poll::Ready(Some(Err(error))) => {
                // upstream stream errored; drop the sender so the drainer exits.
                self.sender = None;
                self.end_emitted = true;
                Poll::Ready(Some(Err(error)))
            }
            Poll::Ready(Some(Ok(chunk))) => {
                self.emit_chunk(&chunk);
                Poll::Ready(Some(Ok(chunk)))
            }
        }
    }
}

/// `Weak<PipeFactoryRegistry>` because `inner` resolves through the same
/// registry that owns this factory — Arc would cycle.
pub struct RecordPipeFactory {
    upstreams: Weak<PipeFactoryRegistry>,
    spigot: DeferredRuntime,
}

impl RecordPipeFactory {
    #[must_use]
    pub fn new(upstreams: Weak<PipeFactoryRegistry>, spigot: DeferredRuntime) -> Self {
        Self { upstreams, spigot }
    }
}

impl PipeFactory for RecordPipeFactory {
    fn name(&self) -> &str {
        "record"
    }

    fn build(
        &self,
        spec: &Value,
        _inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        let upstreams = self.upstreams.clone();
        let spigot = self.spigot.clone();
        Box::pin(async move {
            let config: RecordConfig = serde_json::from_value(spec)
                .map_err(|err| ProximaError::Config(format!("record config: {err}")))?;
            config
                .validate()
                .map_err(|err| ProximaError::Config(format!("{err}")))?;
            let label = config.name.clone();
            let pipe_label = config.pipe.clone().unwrap_or_else(|| label.clone());
            let sink_spec = config.sink.into_sink_spec()?;
            let durable = Arc::new(LazyFanOut::new(vec![sink_spec], spigot.clone()));
            let sink: DynRecordingSink = Arc::new(AccumulatingSink::with_defaults(durable));
            let inner = resolve_inner(&config.inner, &upstreams).await?;
            let upstream = RecordUpstream::new(label, inner, sink, pipe_label)
                .with_protocol(config.protocol)
                .with_runtime(spigot.clone());
            Ok(into_handle(upstream))
        })
    }
}

/// Serialisable recording format — the config mirror of [`FormatKind`].
/// Accepts `bin`, `jsonl`, or `json` (the last two both → JSON) matching the
/// historical hand-parser; defaults to `bin`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormatChoice {
    #[default]
    Bin,
    Jsonl,
    Json,
}

impl From<FormatChoice> for FormatKind {
    fn from(choice: FormatChoice) -> Self {
        match choice {
            FormatChoice::Bin => FormatKind::Bin,
            FormatChoice::Jsonl | FormatChoice::Json => FormatKind::Json,
        }
    }
}

fn default_format() -> FormatChoice {
    FormatChoice::Bin
}

/// Typed config for a recording sink — the destination file + format the
/// interactions are written to. Mirrors [`SinkSpec`].
#[derive(Debug, Clone, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_RECORD_SINK")]
#[builder(derive(Clone, Debug), on(String, into))]
pub struct SinkConfig {
    /// Destination path for the recording.
    pub path: String,

    /// Output format (`bin` | `jsonl` | `json`). Defaults to `bin`. The wire
    /// form also accepts the legacy `format` key as an alias for `type`.
    #[setting(skip)]
    #[serde(default = "default_format", alias = "format", rename = "type")]
    #[builder(default = default_format())]
    pub format: FormatChoice,

    /// Optional zstd compression level for the `bin` format.
    #[setting(default)]
    #[serde(default)]
    pub zstd_level: Option<i32>,
}

impl SinkConfig {
    /// Lower the wire config to the runtime [`SinkSpec`].
    pub fn into_sink_spec(self) -> Result<SinkSpec, ProximaError> {
        let mut sink_spec = SinkSpec::new(&self.path, self.format.into());
        if let Some(level) = self.zstd_level {
            sink_spec = sink_spec.with_zstd_level(level);
        }
        Ok(sink_spec)
    }
}

fn default_record_label() -> String {
    "record".to_string()
}

fn default_protocol() -> String {
    "http".to_string()
}

/// Typed config surface for the `record` upstream — a tee that records every
/// interaction through `inner` to a `sink`. `inner` stays a recursive pipe
/// spec (resolved via the registry like `load.rs::build_pipe`), so it is held
/// as a neutral [`Value`] rather than flattened.
#[derive(Debug, Clone, Builder, Deserialize, Serialize)]
#[builder(derive(Clone, Debug), on(String, into))]
pub struct RecordConfig {
    /// Recording sink (destination + format).
    pub sink: SinkConfig,

    /// The inner pipe spec to wrap and record (recursive).
    pub inner: Value,

    /// Handler / upstream label.
    #[serde(default = "default_record_label")]
    #[builder(default = default_record_label())]
    pub name: String,

    /// Logical pipe label stamped into the recording. Defaults to `name`.
    #[serde(default)]
    pub pipe: Option<String>,

    /// Recorded protocol tag. Defaults to `http`.
    #[serde(default = "default_protocol")]
    #[builder(default = default_protocol())]
    pub protocol: String,
}

impl Validate for RecordConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.sink.path.is_empty() {
            errors.push(ValidationMessage::new("sink.path", "must not be empty"));
        }
        if self.inner.is_null() {
            errors.push(ValidationMessage::new("inner", "must not be null"));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

async fn resolve_inner(
    spec: &Value,
    upstreams: &Weak<PipeFactoryRegistry>,
) -> Result<PipeHandle, ProximaError> {
    let registry = upstreams.upgrade().ok_or_else(|| {
        ProximaError::Registry("upstream registry dropped before record build".into())
    })?;
    // mirror the shorthand dispatch in load.rs::build_pipe.
    if let Some(http) = spec.get("http")
        && let Some(url) = http.as_str()
    {
        let factory = registry.get("http")?;
        let inner_spec = serde_json::json!({ "url": url });
        return factory.build(&inner_spec, None).await;
    }
    if let Some(synth) = spec.get("synth") {
        let factory = registry.get("synth")?;
        return factory.build(synth, None).await;
    }
    if let Some(callback) = spec.get("callback") {
        let factory = registry.get("callback")?;
        return factory.build(callback, None).await;
    }
    if let Some(replay) = spec.get("replay") {
        let factory = registry.get("replay")?;
        return factory.build(replay, None).await;
    }
    if let Some(process) = spec.get("process") {
        let factory = registry.get("process")?;
        return factory.build(process, None).await;
    }
    if let Some(rpc) = spec.get("process_rpc") {
        let factory = registry.get("process_rpc")?;
        return factory.build(rpc, None).await;
    }
    if let Some(type_field) = spec.get("type").and_then(Value::as_str) {
        let factory = registry.get(type_field)?;
        return factory.build(spec, None).await;
    }
    Err(ProximaError::Config(
        "record.inner needs http / synth / callback / replay / process / process_rpc / `type`"
            .into(),
    ))
}

#[cfg(all(
    test,
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::recording::JsonlSource;
    use crate::recording::source::RecordingSource;
    use crate::upstreams::synth::SynthPipeFactory;
    use futures::StreamExt;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn build_pipe_factory_registry() -> Arc<PipeFactoryRegistry> {
        let registry = Arc::new(PipeFactoryRegistry::new());
        registry
            .register(Arc::new(SynthPipeFactory))
            .expect("register synth");
        registry
    }

    fn armed_spigot() -> crate::recording::DeferredRuntime {
        let spigot = crate::recording::deferred_runtime();
        spigot
            .set(
                std::sync::Arc::new(crate::runtime::PrimeRuntime::new(1).expect("prime runtime"))
                    as std::sync::Arc<dyn crate::runtime::Runtime>,
            )
            .ok();
        spigot
    }

    async fn drain_events(path: &std::path::Path) -> Vec<RecordingEvent> {
        if !path.exists() {
            return Vec::new();
        }
        let runtime: std::sync::Arc<dyn crate::runtime::Runtime> =
            std::sync::Arc::new(crate::runtime::PrimeRuntime::new(1).expect("prime runtime"));
        let source = JsonlSource::new(path, runtime);
        let mut events = source.events();
        let mut collected: Vec<RecordingEvent> = Vec::new();
        while let Some(event) = events.next().await {
            match event {
                Ok(recording_event) => collected.push(recording_event),
                Err(_) => break,
            }
        }
        collected
    }

    fn recording_complete(events: &[RecordingEvent]) -> bool {
        events
            .iter()
            .any(|event| matches!(event.event, ProtocolEvent::Http(HttpEvent::Ended { .. })))
    }

    // the sink drains on a background task; poll the trace until the recording
    // is terminal rather than guessing a fixed yield count (raced on slow CI)
    async fn drain_until(
        path: &std::path::Path,
        ready: fn(&[RecordingEvent]) -> bool,
    ) -> Vec<RecordingEvent> {
        for _ in 0..1024 {
            let collected = drain_events(path).await;
            if ready(&collected) {
                return collected;
            }
            proxima_primitives::sync::task::yield_now().await;
        }
        drain_events(path).await
    }

    // principle-4 parity: the fluent builder and the config value must lower to
    // identical SinkSpec state (path, format, zstd level).
    #[test]
    fn parity_fluent_builder_and_config_value_match() {
        let from_value: SinkConfig = serde_json::from_value(serde_json::json!({
            "path": "/var/trace.bin",
            "type": "bin",
            "zstd_level": 7,
        }))
        .expect("from_value");
        let from_value = from_value.into_sink_spec().expect("into_sink_spec value");

        let from_builder = SinkConfig::builder()
            .path("/var/trace.bin")
            .format(FormatChoice::Bin)
            .zstd_level(7)
            .build()
            .into_sink_spec()
            .expect("into_sink_spec builder");

        assert_eq!(from_value, from_builder);
    }

    #[test]
    fn sink_format_alias_maps_jsonl_to_json() {
        let via_format: SinkConfig =
            serde_json::from_value(serde_json::json!({"path": "/x", "format": "jsonl"}))
                .expect("from_value");
        assert_eq!(
            via_format.into_sink_spec().expect("spec").format,
            FormatKind::Json
        );
    }

    #[proxima::test]
    async fn record_factory_resolves_inner_via_registry_and_records_round_trip() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("trace.jsonl");
        let upstreams = build_pipe_factory_registry();
        let factory = RecordPipeFactory::new(Arc::downgrade(&upstreams), armed_spigot());
        let spec = serde_json::json!({
            "name": "echo_recorded",
            "pipe": "echo",
            "sink":  { "type": "jsonl", "path": path.to_string_lossy() },
            "inner": { "synth": { "status": 200, "body": "from-inner" } },
        });
        let handle = factory.build(&spec, None).await.expect("build");
        let request = Request::builder()
            .method("POST")
            .path("/v1/chat")
            .body("hello")
            .build()
            .expect("request");
        let response = SendPipe::call(&handle, request).await.expect("call");
        assert_eq!(response.status, 200);
        let body = response.collect_body().await.expect("collect");
        assert_eq!(&body[..], b"from-inner");

        let collected = drain_until(&path, recording_complete).await;
        assert!(matches!(
            collected[0].event,
            ProtocolEvent::Http(HttpEvent::Started { .. })
        ));
        // streaming order: request side may emit RequestChunk(s) +
        // RequestEnded before ResponseStarted. assert the structural shape.
        let mut idx = 1;
        while matches!(
            collected.get(idx).map(|event| &event.event),
            Some(ProtocolEvent::Http(HttpEvent::RequestChunk { .. }))
        ) {
            idx += 1;
        }
        assert!(matches!(
            collected[idx].event,
            ProtocolEvent::Http(HttpEvent::RequestEnded)
        ));
        idx += 1;
        assert!(matches!(
            collected[idx].event,
            ProtocolEvent::Http(HttpEvent::ResponseStarted { .. })
        ));
        idx += 1;
        while matches!(
            collected.get(idx).map(|event| &event.event),
            Some(ProtocolEvent::Http(HttpEvent::ResponseChunk { .. }))
        ) {
            idx += 1;
        }
        assert!(matches!(
            collected[idx].event,
            ProtocolEvent::Http(HttpEvent::Ended { .. })
        ));
    }

    #[proxima::test]
    async fn pipe_attached_metadata_round_trips_to_recorded_response_chunk() {
        use crate::pipe::into_handle;

        // a Handler that stashes a "clock_at_call" entropy fingerprint into
        // the per-call capture context. mirrors what an entropy wrapper
        // elsewhere in the workspace would do at a nondeterministic seam.
        struct ClockCapturingPipe {
            clock_value: u64,
        }
        impl SendPipe for ClockCapturingPipe {
            type In = Request<Bytes>;
            type Out = Response<Bytes>;
            type Err = ProximaError;

            fn call(
                &self,
                request: Request<Bytes>,
            ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
                let clock_value = self.clock_value;
                async move {
                    if let Some(capture) = request.context.capture.as_ref() {
                        capture.attach(
                            "clock_at_call",
                            bytes::Bytes::copy_from_slice(&clock_value.to_be_bytes()),
                        );
                    }
                    Ok(Response::ok(bytes::Bytes::from_static(b"recorded-body")))
                }
            }
        }


        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("trace.jsonl");
        let durable = Arc::new(LazyFanOut::new(
            vec![SinkSpec::new(path.to_string_lossy(), FormatKind::Json)],
            armed_spigot(),
        ));
        let sink: DynRecordingSink = Arc::new(AccumulatingSink::with_defaults(durable));
        let inner = into_handle(ClockCapturingPipe {
            clock_value: 0x0123_4567_89AB_CDEF,
        });
        let recorder = RecordUpstream::new("recorded", inner, sink, "echo");
        let request = Request::builder()
            .method("POST")
            .path("/v1/chat")
            .body("ignored")
            .build()
            .expect("request");
        let response = SendPipe::call(&recorder, request).await.expect("call");
        let body = response.collect_body().await.expect("collect");
        assert_eq!(&body[..], b"recorded-body");

        let collected = drain_until(&path, recording_complete).await;
        let chunk_metadata = collected
            .iter()
            .find_map(|event| match &event.event {
                ProtocolEvent::Http(HttpEvent::ResponseChunk { metadata, .. })
                    if !metadata.is_empty() =>
                {
                    Some(metadata)
                }
                _ => None,
            })
            .expect("response chunk with metadata must be recorded");
        let recorded = chunk_metadata
            .get("clock_at_call")
            .expect("clock_at_call key present");
        assert_eq!(
            recorded.as_ref(),
            &0x0123_4567_89AB_CDEF_u64.to_be_bytes(),
            "recorded entropy must match what the Handler attached",
        );
    }

    #[proxima::test]
    async fn record_factory_missing_sink_returns_config_error() {
        let upstreams = build_pipe_factory_registry();
        let factory = RecordPipeFactory::new(Arc::downgrade(&upstreams), armed_spigot());
        let spec = serde_json::json!({
            "inner": { "synth": { "status": 200, "body": "x" } },
        });
        let outcome = factory.build(&spec, None).await;
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[proxima::test]
    async fn record_factory_missing_inner_returns_config_error() {
        let upstreams = build_pipe_factory_registry();
        let factory = RecordPipeFactory::new(Arc::downgrade(&upstreams), armed_spigot());
        let spec = serde_json::json!({
            "sink": { "type": "jsonl", "path": "/tmp/x.jsonl" },
        });
        let outcome = factory.build(&spec, None).await;
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[proxima::test]
    async fn record_upstream_directly_constructed_works_without_factory() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("direct.jsonl");
        let durable = Arc::new(LazyFanOut::new(
            vec![SinkSpec::new(path.to_string_lossy(), FormatKind::Json)],
            armed_spigot(),
        ));
        let sink: DynRecordingSink = Arc::new(AccumulatingSink::with_defaults(durable));
        let inner_factory = SynthPipeFactory;
        let inner = inner_factory
            .build(&serde_json::json!({ "status": 200, "body": "ok" }), None)
            .await
            .expect("inner");
        let upstream = RecordUpstream::new("rec", inner, sink.clone(), "rec");
        let request = Request::builder()
            .method("GET")
            .path("/")
            .build()
            .expect("request");
        let response = SendPipe::call(&upstream, request).await.expect("call");
        assert_eq!(response.status, 200);
        // drain response body so chunk events are emitted before flush.
        let _ = response.collect_body().await.expect("collect");
        for _ in 0..16 {
            proxima_primitives::sync::task::yield_now().await;
        }
        sink.flush().await.expect("flush");
    }

    #[proxima::test]
    async fn armed_spigot_spawns_drainer_once_across_calls() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("armed.jsonl");
        let spigot = armed_spigot();
        let durable = Arc::new(LazyFanOut::new(
            vec![SinkSpec::new(path.to_string_lossy(), FormatKind::Json)],
            spigot.clone(),
        ));
        let sink: DynRecordingSink = Arc::new(AccumulatingSink::with_defaults(durable));
        let inner_factory = SynthPipeFactory;
        let inner = inner_factory
            .build(&serde_json::json!({ "status": 200, "body": "ok" }), None)
            .await
            .expect("inner");
        let upstream =
            RecordUpstream::new("rec-armed", inner, sink.clone(), "rec").with_runtime(spigot);

        for _ in 0..3 {
            let request = Request::builder()
                .method("GET")
                .path("/")
                .build()
                .expect("request");
            let response = SendPipe::call(&upstream, request).await.expect("call");
            assert_eq!(response.status, 200);
            let _ = response.collect_body().await.expect("collect");
        }

        let collected = drain_until(&path, |events| {
            events
                .iter()
                .filter(|event| matches!(event.event, ProtocolEvent::Http(HttpEvent::Ended { .. })))
                .count()
                == 3
        })
        .await;
        let ended_count = collected
            .iter()
            .filter(|event| matches!(event.event, ProtocolEvent::Http(HttpEvent::Ended { .. })))
            .count();
        assert_eq!(
            ended_count, 3,
            "all three calls recorded through the single armed drainer"
        );
    }
}
