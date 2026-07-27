use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::task::{Context, Poll};
use std::time::Duration;

use bon::Builder;
use bytes::Bytes;
use conflaguration::{Settings, Validate, ValidationMessage};
use futures::{FutureExt, Stream, select_biased};
use proxima_primitives::pipe::capabilities::Clock;
use proxima_primitives::pipe::clock::TimeClock;
use proxima_primitives::sync::mpsc;
use proxima_primitives::sync::oneshot;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use time::OffsetDateTime;
use ulid::Ulid;

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
type DrainerCell = Arc<OnceLock<mpsc::UnboundedSender<DrainerMessage>>>;

/// A message on the drainer's ordered channel: either a recording event to
/// append, or a barrier asking "is everything sent before me durable yet".
/// Channel-only — never serialized, unlike [`RecordingEvent`] — so a
/// barrier can never leak into the recorded wire format.
// `Event` is the hot-path variant (every request sends several); boxing it
// to shrink `Barrier` would put a heap allocation on every recorded event
// to save stack space on a variant sent at most once per `flush()` call.
#[allow(clippy::large_enum_variant)]
enum DrainerMessage {
    Event(RecordingEvent),
    /// Reports the outcome of the flush the barrier triggered, so a caller
    /// awaiting [`RecordUpstream::flush`] observes a real I/O error rather
    /// than a bare "done".
    Barrier(oneshot::Sender<Result<(), ProximaError>>),
}

/// Proxy that tees every (request, response) interaction into a
/// recording sink. Per-chunk events preserve inter-chunk timing for
/// replay; sink writes drain on a background task so the request hot
/// path doesn't block on I/O.
///
/// Generic over the inner handle: `RecordUpstream<PipeHandle>` impls
/// `Handler`; `RecordUpstream<ThreadLocalPipeHandle>` impls
/// `ThreadLocalHandler`. Dispatch unifies through the
/// `ThreadLocalHandler` blanket so a single body pipes both paths.
pub struct RecordUpstream<Inner = PipeHandle, Clk = TimeClock> {
    label: String,
    inner: Inner,
    sink: DynRecordingSink,
    pipe_label: String,
    protocol: String,
    // armed by the App at serve; once set, the drainer spawns once instead
    // of per call (see `instance_drainer_sender`).
    spigot: DeferredRuntime,
    drainer: DrainerCell,
    // the injected time seam `TimedReplay` already uses on the replay side
    // (`proxima_primitives::pipe::capabilities::Clock`); production defaults
    // to `TimeClock` (the real driver), `with_clock` swaps in a deterministic
    // double so record and replay share one clock instead of record minting
    // its own `Instant`/`OffsetDateTime::now_utc()` reads.
    clock: Clk,
    // a wall-clock reading captured once at construction, paired with the
    // clock's own reading at that same instant. Every request's `ts_start`
    // is `wall_epoch + (clock.now_nanos() - epoch_nanos)` — a pure function
    // of the injected clock (so it can be made deterministic) that still
    // tracks real wall time under the production `TimeClock`, instead of a
    // fresh `now_utc()` read per request.
    wall_epoch: OffsetDateTime,
    epoch_nanos: u64,
    // seeded per-instance ULID randomness feeding `InteractionId`. Production
    // seeds this once from OS entropy (`rand::random`); `with_clock` accepts
    // an explicit seed so a test reproduces the exact same id sequence.
    // Mutex-guarded (never held across an `.await`) rather than lock-free,
    // matching this crate's existing short-critical-section convention (see
    // `pipe/causality.rs`'s `Arc<Vec<std::sync::Mutex<..>>>` slots).
    rng: Arc<Mutex<StdRng>>,
}

impl<Inner> RecordUpstream<Inner, TimeClock> {
    #[must_use]
    pub fn new(
        label: impl Into<String>,
        inner: Inner,
        sink: DynRecordingSink,
        pipe_label: impl Into<String>,
    ) -> Self {
        let clock = TimeClock;
        let epoch_nanos = clock.now_nanos();
        Self {
            label: label.into(),
            inner,
            sink,
            pipe_label: pipe_label.into(),
            protocol: "http".into(),
            spigot: deferred_runtime(),
            drainer: Arc::new(OnceLock::new()),
            clock,
            wall_epoch: OffsetDateTime::now_utc(),
            epoch_nanos,
            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(rand::random()))),
        }
    }
}

impl<Inner, Clk> RecordUpstream<Inner, Clk>
where
    Clk: Clock,
{
    /// Build over an explicit [`Clock`] and a seeded RNG — the seam a
    /// deterministic test injects so two runs of the same scenario mint
    /// byte-identical `InteractionId`s and `ts_ms`/`ts_start` values.
    /// Mirrors `TimedReplay::with_clock` on the replay side (see
    /// `proxima-recording/src/pipe/replay.rs`).
    #[must_use]
    pub fn with_clock(
        label: impl Into<String>,
        inner: Inner,
        sink: DynRecordingSink,
        pipe_label: impl Into<String>,
        clock: Clk,
        wall_epoch: OffsetDateTime,
        rng_seed: u64,
    ) -> Self {
        let epoch_nanos = clock.now_nanos();
        Self {
            label: label.into(),
            inner,
            sink,
            pipe_label: pipe_label.into(),
            protocol: "http".into(),
            spigot: deferred_runtime(),
            drainer: Arc::new(OnceLock::new()),
            clock,
            wall_epoch,
            epoch_nanos,
            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(rng_seed))),
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

    /// Await durability of every interaction enqueued before this call.
    ///
    /// Enqueues a barrier on the SAME ordered channel recording events flow
    /// through, so it resolves only once every event sent strictly before
    /// it has been appended and the sink flushed — never a sleep, never a
    /// filesystem poll. Ordering, not timing, is what makes this
    /// deterministic. Calling it before any request has gone through starts
    /// the drainer exactly as the first real request would (the same
    /// `get_or_init`), so there is still exactly one channel and one
    /// ordering to reason about — no separate "is it running yet" check to
    /// get wrong.
    ///
    /// # Errors
    ///
    /// Returns `Err` rather than resolving if there is no persistent
    /// drainer to barrier against (no runtime armed via
    /// [`RecordUpstream::with_runtime`]) or if the drainer task is gone —
    /// spawn failed, panicked, or was cancelled mid-flight. A barrier that
    /// quietly "succeeds" against a missing or dead drainer would hide
    /// exactly the failure this API exists to surface.
    ///
    /// ```
    /// use bytes::Bytes;
    /// use proxima::pipe::{PipeHandle, into_handle};
    /// use proxima::{
    ///     AccumulatingSink, DynRecordingSink, LazyFanOut, ProximaError, RecordUpstream, Request,
    ///     Response, SendPipe, deferred_runtime,
    /// };
    ///
    /// struct EchoPipe;
    ///
    /// impl SendPipe for EchoPipe {
    ///     type In = Request<Bytes>;
    ///     type Out = Response<Bytes>;
    ///     type Err = ProximaError;
    ///     async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    ///         Ok(Response::ok("pong"))
    ///     }
    /// }
    ///
    /// # #[proxima::main]
    /// # async fn main() -> Result<(), ProximaError> {
    /// let sink: DynRecordingSink = std::sync::Arc::new(AccumulatingSink::with_defaults(
    ///     std::sync::Arc::new(LazyFanOut::new(Vec::new(), deferred_runtime())),
    /// ));
    /// let inner: PipeHandle = into_handle(EchoPipe);
    /// let upstream = RecordUpstream::new("example", inner, sink, "example");
    ///
    /// // no runtime armed: flush reports the gap instead of hanging forever.
    /// assert!(upstream.flush().await.is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn flush(&self) -> impl Future<Output = Result<(), ProximaError>> + Send + 'static {
        let sender = self
            .spigot
            .get()
            .map(|runtime| instance_drainer_sender(&self.drainer, runtime, &self.sink));
        async move {
            let sender = sender.ok_or_else(|| {
                ProximaError::Record(
                    "recording drainer has no armed runtime to flush (call with_runtime first)"
                        .into(),
                )
            })?;
            let (ack_sender, ack_receiver) = oneshot::channel();
            sender
                .send(DrainerMessage::Barrier(ack_sender))
                .map_err(|_| ProximaError::Record("recording drainer is gone".into()))?;
            ack_receiver.await.map_err(|_| {
                ProximaError::Record("recording drainer dropped before flushing".into())
            })?
        }
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
) -> mpsc::UnboundedSender<DrainerMessage> {
    // spawn_on_core, not spawn_on_current_core: the recording pipe is driven
    // from arbitrary call sites, not necessarily a runtime worker thread.
    drainer
        .get_or_init(|| {
            let (sender, receiver) = mpsc::unbounded_channel::<DrainerMessage>();
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
    mut receiver: mpsc::UnboundedReceiver<DrainerMessage>,
    sink: DynRecordingSink,
) {
    while let Some(message) = receiver.recv().await {
        drain_message(&sink, message).await;
        // drain any burst already queued without waiting for more: a
        // `now_or_never` immediate poll stands in for tokio's `try_recv`
        // (proxima's mpsc doesn't shim that non-blocking probe — see its
        // module doc's "Non-coverage" list).
        while let Some(Some(message)) = receiver.recv().now_or_never() {
            drain_message(&sink, message).await;
        }
        if let Err(error) = sink.flush().await {
            tracing::error!(error = %error, "recording sink flush failed");
        }
    }
    if let Err(error) = sink.flush().await {
        tracing::error!(error = %error, "recording sink flush failed");
    }
}

// appends a durable event, or — for a barrier — flushes right away and
// replies with the real outcome, so an awaiting caller never observes a
// later, unrelated flush's result in place of its own.
async fn drain_message(sink: &DynRecordingSink, message: DrainerMessage) {
    match message {
        DrainerMessage::Event(event) => {
            if let Err(error) = sink.append(event).await {
                tracing::error!(error = %error, "recording sink append failed");
            }
        }
        DrainerMessage::Barrier(ack) => {
            let result = sink.flush().await;
            if let Err(ref error) = result {
                tracing::error!(error = %error, "recording sink flush failed");
            }
            let _ = ack.send(result);
        }
    }
}

impl<Inner, Clk> SendPipe for RecordUpstream<Inner, Clk>
where
    Inner: Handler + Clone,
    Clk: Clock + Clone + Send + Sync + Unpin + 'static,
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
            self.clock.clone(),
            self.wall_epoch,
            self.epoch_nanos,
            self.rng.clone(),
        )
    }
}


impl<Clk> Pipe for RecordUpstream<ThreadLocalPipeHandle, Clk>
where
    Clk: Clock + Clone + Send + Unpin + 'static,
{
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
            self.clock.clone(),
            self.wall_epoch,
            self.epoch_nanos,
            self.rng.clone(),
        )
    }
}

// derives the `InteractionId`'s wall-clock-timestamp component and the
// `HttpEvent::Started.ts` field from the injected clock, so both are a pure
// function of `(wall_epoch, epoch_nanos, clock.now_nanos())` instead of a
// fresh `OffsetDateTime::now_utc()` read — the seam that makes two runs of
// the same scenario mint identical timestamps under a deterministic clock,
// while still tracking real wall time under the production `TimeClock`.
fn wall_clock_now(wall_epoch: OffsetDateTime, epoch_nanos: u64, now_nanos: u64) -> OffsetDateTime {
    wall_epoch + Duration::from_nanos(now_nanos.saturating_sub(epoch_nanos))
}

// two `next_u64` draws under one lock acquisition give the 80 bits of
// randomness `Ulid::from_parts` keeps (it masks away the rest) — see
// `InteractionId::new()`'s `ulid::Ulid::from_datetime_with_source`, whose
// same-shaped msb/lsb draw this seeded path replaces.
fn draw_interaction_random(rng: &Mutex<StdRng>) -> u128 {
    let mut guard = rng.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let high = u128::from(guard.next_u64());
    let low = u128::from(guard.next_u64());
    (high << 64) | low
}

fn mint_interaction_id(
    wall_epoch: OffsetDateTime,
    epoch_nanos: u64,
    now_nanos: u64,
    rng: &Mutex<StdRng>,
) -> (InteractionId, OffsetDateTime) {
    let ts_start = wall_clock_now(wall_epoch, epoch_nanos, now_nanos);
    let timestamp_ms = u64::try_from(ts_start.unix_timestamp_nanos() / 1_000_000).unwrap_or(0);
    let random = draw_interaction_random(rng);
    (
        InteractionId::from_ulid(Ulid::from_parts(timestamp_ms, random)),
        ts_start,
    )
}

fn elapsed_ms(now_nanos: u64, started_nanos: u64) -> u64 {
    now_nanos.saturating_sub(started_nanos) / 1_000_000
}


/// Shared body for both Handler and ThreadLocalHandler impls. Dispatches
/// the inner call via `ThreadLocalHandler::call` — the blanket impl makes
/// every `Handler` automatically a `ThreadLocalHandler`, so a Send Inner
/// still produces a Send future here and an Rc-based Inner produces a
/// !Send one.
// four extra params (clock/wall_epoch/epoch_nanos/rng) over the pre-existing
// seven thread the injected time+rng seam through from `RecordUpstream`'s
// fields; a wrapper struct would just relocate the same plumbing.
#[allow(clippy::too_many_arguments)]
async fn record_dispatch<Inner, Clk>(
    inner: Inner,
    request: Request<Bytes>,
    sink: DynRecordingSink,
    pipe_label: String,
    protocol: String,
    spigot: DeferredRuntime,
    drainer: DrainerCell,
    clock: Clk,
    wall_epoch: OffsetDateTime,
    epoch_nanos: u64,
    rng: Arc<Mutex<StdRng>>,
) -> Result<Response<Bytes>, ProximaError>
where
    Inner: Handler + Clone,
    Clk: Clock + Clone + Send + Sync + Unpin + 'static,
{
    let cancel = request.context.cancel.clone();
    let started_nanos = clock.now_nanos();
    let (id, ts_start) = mint_interaction_id(wall_epoch, epoch_nanos, started_nanos, &rng);

    let sender = match spigot.get() {
        // armed: one drainer per RecordUpstream, spawned once on the
        // injected runtime — not a `tokio::spawn` per call.
        Some(runtime) => instance_drainer_sender(&drainer, runtime, &sink),
        // unarmed (tests / direct construction): the legacy per-call
        // drainer on the ambient tokio runtime, cancellable with the request.
        None => {
            let (sender, mut receiver) = mpsc::unbounded_channel::<DrainerMessage>();
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
                            message = receiver.recv().fuse() => match message {
                                Some(message) => drain_message(&sink_for_task, message).await,
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
    let _ = sender.send(DrainerMessage::Event(RecordingEvent {
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
    }));

    let req_body = wrap_chunked(
        req_chunks,
        clock.clone(),
        started_nanos,
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

    let resp_started_ms = elapsed_ms(clock.now_nanos(), started_nanos);
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
    let _ = sender.send(DrainerMessage::Event(RecordingEvent {
        id,
        ts_ms: resp_started_ms,
        parent: None,
        event: ProtocolEvent::Http(HttpEvent::ResponseStarted {
            status: response.status,
            headers: header_pairs,
        }),
    }));

    let status = response.status;
    let headers = response.metadata.clone();
    let resp_chunks = response.into_chunk_stream();
    let resp_body = wrap_chunked(
        resp_chunks,
        clock,
        started_nanos,
        id,
        sender,
        Phase::Response,
        capture,
    );
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
// same shape as `record_dispatch` above — see its comment.
#[allow(clippy::too_many_arguments)]
async fn record_dispatch_local<Inner, Clk>(
    inner: Inner,
    request: Request<Bytes>,
    sink: DynRecordingSink,
    pipe_label: String,
    protocol: String,
    spigot: DeferredRuntime,
    drainer: DrainerCell,
    clock: Clk,
    wall_epoch: OffsetDateTime,
    epoch_nanos: u64,
    rng: Arc<Mutex<StdRng>>,
) -> Result<Response<Bytes>, ProximaError>
where
    Inner: ThreadLocalHandler + Clone,
    Clk: Clock + Clone + Send + Unpin + 'static,
{
    let cancel = request.context.cancel.clone();
    let started_nanos = clock.now_nanos();
    let (id, ts_start) = mint_interaction_id(wall_epoch, epoch_nanos, started_nanos, &rng);

    let sender = match spigot.get() {
        // armed: one drainer per RecordUpstream, spawned once on the
        // injected runtime — not a `tokio::spawn` per call.
        Some(runtime) => instance_drainer_sender(&drainer, runtime, &sink),
        // unarmed (tests / direct construction): the legacy per-call
        // drainer on the ambient tokio runtime, cancellable with the request.
        None => {
            let (sender, mut receiver) = mpsc::unbounded_channel::<DrainerMessage>();
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
                            message = receiver.recv().fuse() => match message {
                                Some(message) => drain_message(&sink_for_task, message).await,
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
    let _ = sender.send(DrainerMessage::Event(RecordingEvent {
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
    }));

    let req_body = wrap_chunked(
        req_chunks,
        clock.clone(),
        started_nanos,
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

    let resp_started_ms = elapsed_ms(clock.now_nanos(), started_nanos);
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
    let _ = sender.send(DrainerMessage::Event(RecordingEvent {
        id,
        ts_ms: resp_started_ms,
        parent: None,
        event: ProtocolEvent::Http(HttpEvent::ResponseStarted {
            status: response.status,
            headers: header_pairs,
        }),
    }));

    let status = response.status;
    let headers = response.metadata.clone();
    let resp_chunks = response.into_chunk_stream();
    let resp_body = wrap_chunked(
        resp_chunks,
        clock,
        started_nanos,
        id,
        sender,
        Phase::Response,
        capture,
    );
    let mut rebuilt =
        Response::new(status).with_stream(ResponseStream::from_chunk_stream(resp_body));
    for (name, value) in headers {
        rebuilt = rebuilt.with_header(name, value);
    }
    Ok(rebuilt)
}

fn wrap_chunked<Clk>(
    inner: ChunkStream,
    clock: Clk,
    started_nanos: u64,
    id: InteractionId,
    sender: mpsc::UnboundedSender<DrainerMessage>,
    phase: Phase,
    capture: Arc<LiveCaptureContext>,
) -> ChunkStream
where
    Clk: Clock + Send + Unpin + 'static,
{
    Box::pin(ChunkRecorder {
        inner,
        clock,
        started_nanos,
        id,
        sender: Some(sender),
        phase,
        end_emitted: false,
        capture,
    })
}

struct ChunkRecorder<Clk: Clock> {
    inner: ChunkStream,
    clock: Clk,
    started_nanos: u64,
    id: InteractionId,
    sender: Option<mpsc::UnboundedSender<DrainerMessage>>,
    phase: Phase,
    end_emitted: bool,
    capture: Arc<LiveCaptureContext>,
}

impl<Clk: Clock> ChunkRecorder<Clk> {
    fn elapsed_ms(&self) -> u64 {
        elapsed_ms(self.clock.now_nanos(), self.started_nanos)
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
            let _ = sender.send(DrainerMessage::Event(RecordingEvent {
                id: self.id,
                ts_ms,
                parent: None,
                event: ProtocolEvent::Http(http_event),
            }));
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
            let _ = sender.send(DrainerMessage::Event(RecordingEvent {
                id: self.id,
                ts_ms: ts_end,
                parent: None,
                event: ProtocolEvent::Http(http_event),
            }));
        }
    }
}

impl<Clk: Clock> Drop for ChunkRecorder<Clk> {
    fn drop(&mut self) {
        // emit end-of-interaction even if the consumer drops mid-stream.
        self.emit_end();
    }
}

impl<Clk: Clock + Send + Unpin + 'static> Stream for ChunkRecorder<Clk> {
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

    // shared by the `PipeFactory` impl (which must return the erased
    // `PipeHandle` for the heterogeneous registry) and tests that need the
    // concrete `RecordUpstream` back — e.g. to await `RecordUpstream::flush`,
    // a capability `PipeHandle`'s object-safe erasure has no room to carry.
    async fn build_typed(&self, spec: &Value) -> Result<RecordUpstream<PipeHandle>, ProximaError> {
        let config: RecordConfig = serde_json::from_value(spec.clone())
            .map_err(|err| ProximaError::Config(format!("record config: {err}")))?;
        config
            .validate()
            .map_err(|err| ProximaError::Config(format!("{err}")))?;
        let label = config.name.clone();
        let pipe_label = config.pipe.clone().unwrap_or_else(|| label.clone());
        let sink_spec = config.sink.into_sink_spec()?;
        let durable = Arc::new(LazyFanOut::new(vec![sink_spec], self.spigot.clone()));
        let sink: DynRecordingSink = Arc::new(AccumulatingSink::with_defaults(durable));
        let inner = resolve_inner(&config.inner, &self.upstreams).await?;
        Ok(RecordUpstream::new(label, inner, sink, pipe_label)
            .with_protocol(config.protocol)
            .with_runtime(self.spigot.clone()))
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
        Box::pin(async move { Ok(into_handle(self.build_typed(&spec).await?)) })
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

    // production arms `PrimeRuntime` through `prime::config::Builder`, which
    // defaults `background_pool` to `PoolKind::Rayon` (prime/src/config.rs)
    // — every durable write in the recording pipe routes through
    // `Runtime::spawn_background_blocking`, and a real pool is always
    // attached. Bypassing the builder here (bare `PrimeRuntime::new`) left
    // `background_pool: None`, so `spawn_background_blocking` fell back to
    // spawning a brand-new OS thread per append/flush call (prime/src/os/runtime.rs) —
    // thread-creation latency that balloons under host contention. A fixed
    // retry-budget poll used to race that latency and lose; `RecordUpstream::flush`
    // now awaits a real barrier instead, but a warm pool still keeps these tests
    // fast. `ProximaBackgroundPool` mirrors production's persistent-pool shape
    // without pulling in `rayon`.
    fn armed_spigot() -> crate::recording::DeferredRuntime {
        let spigot = crate::recording::deferred_runtime();
        let background_pool: std::sync::Arc<dyn crate::runtime::BackgroundPool> =
            std::sync::Arc::new(
                crate::runtime::prime::os::background::ProximaBackgroundPool::new()
                    .expect("background pool"),
            );
        let runtime: std::sync::Arc<dyn crate::runtime::Runtime> = std::sync::Arc::new(
            crate::runtime::PrimeRuntime::new(1)
                .expect("prime runtime")
                .with_background_pool(background_pool),
        );
        // block here (setup, off the request/assertion critical path) until
        // core 0's worker thread has actually run one task. A freshly
        // `thread::Builder::spawn`'d worker is kernel-scheduled but not yet
        // CPU-scheduled; under host contention it can sit unscheduled for
        // longer than a bounded polling loop's whole retry budget, which is
        // exactly the race a since-removed filesystem-polling test helper
        // lost (the drainer task never got a first turn before the poll gave
        // up). A real blocking handshake here proves the worker is warm before any recording
        // event is ever sent to it.
        let (ready_sender, ready_receiver) = std::sync::mpsc::channel::<()>();
        runtime
            .spawn_on_core(
                crate::runtime::CoreId(0),
                Box::pin(async move {
                    let _ = ready_sender.send(());
                }),
            )
            .expect("warm up drainer core");
        ready_receiver
            .recv()
            .expect("drainer core never scheduled a task");
        spigot.set(runtime).ok();
        spigot
    }

    // a single, non-looping read: only ever called after `RecordUpstream::flush`
    // has resolved, so the file already holds everything the barrier ordered
    // ahead of it — no retry budget to race, nothing left to poll for.
    async fn read_recorded_events(path: &std::path::Path) -> Vec<RecordingEvent> {
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
        // `build_typed` (not the `PipeFactory::build` trait method) returns
        // the concrete `RecordUpstream` instead of an erased `PipeHandle` —
        // the erased handle has no room to carry `flush`, the capability this
        // test needs to await durability deterministically.
        let upstream = factory.build_typed(&spec).await.expect("build");
        let request = Request::builder()
            .method("POST")
            .path("/v1/chat")
            .body("hello")
            .build()
            .expect("request");
        let response = SendPipe::call(&upstream, request).await.expect("call");
        assert_eq!(response.status, 200);
        let body = response.collect_body().await.expect("collect");
        assert_eq!(&body[..], b"from-inner");

        upstream.flush().await.expect("flush");
        let collected = read_recorded_events(&path).await;
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
        // one spigot shared by the sink AND the upstream: `RecordUpstream`
        // needs its own runtime armed (not just the sink's) for `flush` to
        // have a persistent drainer to barrier against.
        let spigot = armed_spigot();
        let durable = Arc::new(LazyFanOut::new(
            vec![SinkSpec::new(path.to_string_lossy(), FormatKind::Json)],
            spigot.clone(),
        ));
        let sink: DynRecordingSink = Arc::new(AccumulatingSink::with_defaults(durable));
        let inner = into_handle(ClockCapturingPipe {
            clock_value: 0x0123_4567_89AB_CDEF,
        });
        let recorder = RecordUpstream::new("recorded", inner, sink, "echo").with_runtime(spigot);
        let request = Request::builder()
            .method("POST")
            .path("/v1/chat")
            .body("ignored")
            .build()
            .expect("request");
        let response = SendPipe::call(&recorder, request).await.expect("call");
        let body = response.collect_body().await.expect("collect");
        assert_eq!(&body[..], b"recorded-body");

        recorder.flush().await.expect("flush");
        let collected = read_recorded_events(&path).await;
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

        upstream.flush().await.expect("flush");
        let collected = read_recorded_events(&path).await;
        let ended_count = collected
            .iter()
            .filter(|event| matches!(event.event, ProtocolEvent::Http(HttpEvent::Ended { .. })))
            .count();
        assert_eq!(
            ended_count, 3,
            "all three calls recorded through the single armed drainer"
        );
    }

    // ── single-connection determinism proof ───────────────────────────────
    //
    // One connection, one request/response, through the FULL listener
    // stack (in-memory duplex socket -> h1 codec -> `RecordUpstream` tee ->
    // handler -> response) on a virtual clock and a seeded RNG. Runs the
    // identical scenario `DETERMINISM_RUNS` times and asserts the recorded
    // JSONL trace is byte-identical every time.
    //
    // Scope: single connection, single request/response — proves neither
    // multi-connection ordering (kernel-determined reactor readiness, a
    // separate months-scale component) nor prime's reactor timers (this
    // scenario never awaits one: the h1 driver runs under a bare
    // `futures::executor::block_on`, no runtime installed, and the only
    // "clock" consulted anywhere on this path is `RecordUpstream`'s
    // injected `RecordingClock`).
    #[cfg(feature = "http1-native")]
    mod determinism {
        use super::*;
        use futures::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
        use proxima_primitives::pipe::clock::testing::RecordingClock;
        use std::collections::VecDeque;
        use std::pin::Pin;
        use std::task::{Context, Poll, Waker};
        use time::macros::datetime;

        const DETERMINISM_RUNS: usize = 500;
        const FIXED_SEED: u64 = 0xC0FF_EE99_DEAD_BEEF;
        const REQUEST_WIRE: &[u8] = b"POST /echo HTTP/1.1\r\nHost: determinism.local\r\nContent-Length: 13\r\nConnection: close\r\n\r\nhello, proxima";

        // fixed wall-clock literal the deterministic `RecordingClock` is
        // anchored to (see `RecordUpstream::with_clock`'s `wall_epoch` /
        // `epoch_nanos` pair) — arbitrary but constant across every run.
        fn fixed_wall_epoch() -> OffsetDateTime {
            datetime!(2024-01-01 00:00:00 UTC)
        }

        // echoes the request body back as the response body — exercises the
        // request-chunk AND response-chunk recording paths, not just a bare
        // status/headers round trip.
        struct EchoPipe;

        impl SendPipe for EchoPipe {
            type In = Request<Bytes>;
            type Out = Response<Bytes>;
            type Err = ProximaError;

            fn call(
                &self,
                request: Request<Bytes>,
            ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
                async move {
                    let (_, body) = request.body_bytes().await?;
                    Ok(Response::new(200)
                        .with_body(body)
                        .with_header("content-type", "text/plain"))
                }
            }
        }

        /// Byte queue shared by one direction of [`test_duplex`]. Mirrors
        /// `proxima-http/src/http1/serve.rs`'s private test-only duplex —
        /// `futures` (unlike `tokio`) ships no `io::duplex`, and that one is
        /// `#[cfg(test)]`-private to a different crate, so this integration
        /// test needs its own copy of the same minimal scaffolding.
        #[derive(Default)]
        struct DuplexBuf {
            bytes: VecDeque<u8>,
            closed: bool,
            waker: Option<Waker>,
        }

        struct DuplexHalf {
            read_buf: Arc<std::sync::Mutex<DuplexBuf>>,
            write_buf: Arc<std::sync::Mutex<DuplexBuf>>,
        }

        fn test_duplex() -> (DuplexHalf, DuplexHalf) {
            let a_to_b = Arc::new(std::sync::Mutex::new(DuplexBuf::default()));
            let b_to_a = Arc::new(std::sync::Mutex::new(DuplexBuf::default()));
            (
                DuplexHalf {
                    read_buf: b_to_a.clone(),
                    write_buf: a_to_b.clone(),
                },
                DuplexHalf {
                    read_buf: a_to_b,
                    write_buf: b_to_a,
                },
            )
        }

        impl AsyncRead for DuplexHalf {
            fn poll_read(
                self: Pin<&mut Self>,
                cx: &mut Context<'_>,
                buf: &mut [u8],
            ) -> Poll<std::io::Result<usize>> {
                let mut guard = self.read_buf.lock().unwrap();
                if guard.bytes.is_empty() {
                    if guard.closed {
                        return Poll::Ready(Ok(0));
                    }
                    guard.waker = Some(cx.waker().clone());
                    return Poll::Pending;
                }
                let take = guard.bytes.len().min(buf.len());
                for slot in buf.iter_mut().take(take) {
                    *slot = guard.bytes.pop_front().expect("checked len above");
                }
                Poll::Ready(Ok(take))
            }
        }

        impl AsyncWrite for DuplexHalf {
            fn poll_write(
                self: Pin<&mut Self>,
                _cx: &mut Context<'_>,
                buf: &[u8],
            ) -> Poll<std::io::Result<usize>> {
                let mut guard = self.write_buf.lock().unwrap();
                guard.bytes.extend(buf.iter().copied());
                if let Some(waker) = guard.waker.take() {
                    waker.wake();
                }
                Poll::Ready(Ok(buf.len()))
            }

            fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
                Poll::Ready(Ok(()))
            }

            fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
                let mut guard = self.write_buf.lock().unwrap();
                guard.closed = true;
                if let Some(waker) = guard.waker.take() {
                    waker.wake();
                }
                Poll::Ready(Ok(()))
            }
        }

        impl Drop for DuplexHalf {
            fn drop(&mut self) {
                let mut guard = self.write_buf.lock().unwrap();
                guard.closed = true;
                if let Some(waker) = guard.waker.take() {
                    waker.wake();
                }
            }
        }

        // one full pass of the scenario: fresh tempdir, fresh `RecordingClock`
        // pinned to nanos=0, fresh armed spigot (real background pool, no
        // per-call `std::thread::spawn` fallback), one connection, one
        // request/response, then the raw recorded JSONL bytes off disk.
        fn run_once() -> Vec<u8> {
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("trace.jsonl");
            let spigot = armed_spigot();
            let durable = Arc::new(LazyFanOut::new(
                vec![SinkSpec::new(path.to_string_lossy(), FormatKind::Json)],
                spigot.clone(),
            ));
            let sink: DynRecordingSink = Arc::new(AccumulatingSink::with_defaults(durable));
            let inner: PipeHandle = into_handle(EchoPipe);
            let clock = RecordingClock::new();
            let concrete = Arc::new(
                RecordUpstream::with_clock(
                    "determinism",
                    inner,
                    sink,
                    "determinism",
                    clock,
                    fixed_wall_epoch(),
                    FIXED_SEED,
                )
                .with_runtime(spigot),
            );
            let dispatch: PipeHandle = into_handle(concrete.clone());

            let (server_half, mut client_half) = test_duplex();
            let server = crate::serve_h1_connection(server_half, dispatch, None, None);
            let client = async move {
                client_half
                    .write_all(REQUEST_WIRE)
                    .await
                    .expect("client write");
                let mut response = Vec::new();
                client_half
                    .read_to_end(&mut response)
                    .await
                    .expect("client read");
                response
            };
            let (server_result, response) =
                futures::executor::block_on(futures::future::join(server, client));
            server_result.expect("serve_h1_connection should complete");
            assert!(
                String::from_utf8_lossy(&response).starts_with("HTTP/1.1 200"),
                "sanity: the scenario must actually produce a 200 response"
            );

            futures::executor::block_on(concrete.flush()).expect("flush");
            std::fs::read(&path).expect("read recorded trace")
        }

        // first index at which two byte buffers differ (or the length of the
        // shorter one, if one is a prefix of the other) — the pinpoint this
        // proof needs to name a real diverging leak instead of just failing.
        fn first_divergence(expected: &[u8], actual: &[u8]) -> Option<usize> {
            if expected == actual {
                return None;
            }
            let shared = expected.len().min(actual.len());
            (0..shared)
                .find(|&index| expected[index] != actual[index])
                .or(Some(shared))
        }

        #[test]
        fn single_connection_trace_is_byte_identical_across_n_runs() {
            let baseline = run_once();
            assert!(!baseline.is_empty(), "recorded trace must not be empty");

            for run_index in 1..DETERMINISM_RUNS {
                let candidate = run_once();
                if let Some(offset) = first_divergence(&baseline, &candidate) {
                    let window = 32;
                    let baseline_start = offset.saturating_sub(window);
                    let candidate_start = offset.saturating_sub(window);
                    let baseline_snippet = String::from_utf8_lossy(
                        &baseline[baseline_start..(offset + window).min(baseline.len())],
                    );
                    let candidate_snippet = String::from_utf8_lossy(
                        &candidate[candidate_start..(offset + window).min(candidate.len())],
                    );
                    panic!(
                        "run {run_index} diverged from run 0 at byte {offset}\n  \
                         run 0:  ...{baseline_snippet}...\n  \
                         run {run_index}: ...{candidate_snippet}..."
                    );
                }
            }
        }

        // ── multi-connection interleaving proof ────────────────────────────
        //
        // `DETERMINISM_RUNS` single connections above prove the h1 + record
        // path is deterministic in isolation. This extends the SAME harness
        // to N concurrent connections sharing one `RecordUpstream` (one
        // trace), all driven by a single `futures::executor::block_on` call
        // — no epoll/kqueue, no OS thread hand-off for the connections
        // themselves (the recording drainer's background thread only ever
        // DEQUEUES; every `sender.send` that decides record ORDER happens on
        // this one foreground thread). If the combined trace is still
        // byte-identical across runs and processes, the only ordering input
        // the single-connection proof couldn't exercise — the executor's own
        // ready-queue, deciding which connection's next step runs first — is
        // deterministic too.
        #[cfg(feature = "http-prime-deps")]
        mod multi_connection {
            use super::*;
            use proxima_primitives::stream::{PeerInfo, StreamConnection, StreamUpstream};

            const MULTI_CONN_RUNS: usize = 300;

            // one-shot transport: hands back the pre-built duplex half
            // exactly once, matching `H1ClientUpstream`'s own keep-alive
            // contract (it only ever calls `connect()` again after an
            // error, which this scenario never produces).
            struct SingleConnUpstream {
                conn: std::sync::Mutex<Option<DuplexHalf>>,
            }

            impl SingleConnUpstream {
                fn new(conn: DuplexHalf) -> Self {
                    Self {
                        conn: std::sync::Mutex::new(Some(conn)),
                    }
                }
            }

            impl StreamConnection for DuplexHalf {
                fn peer(&self) -> Option<PeerInfo> {
                    None
                }
            }

            impl StreamUpstream for SingleConnUpstream {
                type Conn = DuplexHalf;

                fn poll_connect(
                    &self,
                    _ctx: &mut Context<'_>,
                ) -> Poll<std::io::Result<Self::Conn>> {
                    let mut guard = self
                        .conn
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    match guard.take() {
                        Some(conn) => Poll::Ready(Ok(conn)),
                        None => Poll::Ready(Err(std::io::Error::other(
                            "single-connection upstream already connected once",
                        ))),
                    }
                }
            }

            // deliberately uneven: connection 0 sends 1 request, connection
            // 1 sends 2, ... wrapping every 4 connections, so a run mixes
            // short- and long-lived connections instead of N identical ones.
            fn request_count_for(connection_index: usize) -> usize {
                1 + (connection_index % 4)
            }

            // varies 40..=290 bytes as a function of (connection, request)
            // — no two requests in a run share a size by coincidence.
            fn byte_count_for(connection_index: usize, request_index: usize) -> usize {
                40 + (connection_index * 31 + request_index * 17) % 251
            }

            // a readable, real-sentence payload padded/truncated to
            // `byte_count` — not `b"AAAA"`: the point is to prove ordering,
            // and a legible trace is easier to eyeball if this ever fails.
            fn scripted_body(
                connection_index: usize,
                request_index: usize,
                byte_count: usize,
            ) -> Bytes {
                let marker = format!("conn={connection_index} req={request_index} ");
                let filler = "the quick brown fox jumps over the lazy dog; ";
                let mut body = marker.into_bytes();
                while body.len() < byte_count {
                    body.extend_from_slice(filler.as_bytes());
                }
                body.truncate(byte_count);
                Bytes::from(body)
            }

            // drives one connection's whole request script, in program
            // order, over its own keep-alive client. The interleaving
            // ACROSS connections comes from running N of these concurrently
            // (see `run_multi_once`), never from anything inside this fn.
            async fn drive_client_connection(
                client: crate::H1ClientUpstream<SingleConnUpstream>,
                connection_index: usize,
                request_count: usize,
            ) {
                for request_index in 0..request_count {
                    let byte_count = byte_count_for(connection_index, request_index);
                    let body = scripted_body(connection_index, request_index, byte_count);
                    let request = Request::builder()
                        .method("POST")
                        .path(format!("/echo/conn-{connection_index}/req-{request_index}"))
                        .body(body.clone())
                        .build()
                        .expect("request");
                    let response = client.call(request).await.expect("client call");
                    assert_eq!(response.status, 200);
                    let echoed = response.collect_body().await.expect("collect body");
                    assert_eq!(
                        echoed, body,
                        "echo pipe must return the exact request body for conn {connection_index} req {request_index}"
                    );
                }
            }

            // one full pass: `connection_count` connections sharing ONE
            // `RecordUpstream` (one trace), driven concurrently by a single
            // `futures::executor::block_on` call. Returns the raw recorded
            // bytes (for the byte-identical comparison) alongside the
            // parsed events (for the interleaving check) — both read back
            // before the tempdir drops.
            fn run_multi_once(connection_count: usize) -> (Vec<u8>, Vec<RecordingEvent>) {
                let dir = tempdir().expect("tempdir");
                let path = dir.path().join("trace.jsonl");
                let spigot = armed_spigot();
                let durable = Arc::new(LazyFanOut::new(
                    vec![SinkSpec::new(path.to_string_lossy(), FormatKind::Json)],
                    spigot.clone(),
                ));
                let sink: DynRecordingSink = Arc::new(AccumulatingSink::with_defaults(durable));
                let inner: PipeHandle = into_handle(EchoPipe);
                let clock = RecordingClock::new();
                let concrete = Arc::new(
                    RecordUpstream::with_clock(
                        "determinism-multi",
                        inner,
                        sink,
                        "determinism-multi",
                        clock,
                        fixed_wall_epoch(),
                        FIXED_SEED,
                    )
                    .with_runtime(spigot),
                );
                let dispatch: PipeHandle = into_handle(concrete.clone());

                let mut servers = Vec::with_capacity(connection_count);
                let mut clients = Vec::with_capacity(connection_count);
                for connection_index in 0..connection_count {
                    let (server_half, client_half) = test_duplex();
                    servers.push(crate::serve_h1_connection(
                        server_half,
                        dispatch.clone(),
                        None,
                        None,
                    ));
                    let client = crate::H1ClientUpstream::new(
                        SingleConnUpstream::new(client_half),
                        "determinism.local",
                        format!("conn-{connection_index}"),
                    );
                    clients.push(drive_client_connection(
                        client,
                        connection_index,
                        request_count_for(connection_index),
                    ));
                }

                let (server_results, _client_results) =
                    futures::executor::block_on(futures::future::join(
                        futures::future::join_all(servers),
                        futures::future::join_all(clients),
                    ));
                for result in server_results {
                    result.expect("serve_h1_connection should complete");
                }

                futures::executor::block_on(concrete.flush()).expect("flush");
                let bytes = std::fs::read(&path).expect("read recorded trace");
                let events = futures::executor::block_on(read_recorded_events(&path));
                (bytes, events)
            }

            fn connection_index_from_path(path: &str) -> usize {
                path.strip_prefix("/echo/conn-")
                    .and_then(|rest| rest.split('/').next())
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or_else(|| {
                        panic!("recorded path did not encode a connection index: {path}")
                    })
            }

            // the connection each `Started` event belongs to, in the order
            // they were recorded — the sequence genuine interleaving must
            // NOT reduce to N contiguous blocks.
            fn started_connection_sequence(events: &[RecordingEvent]) -> Vec<usize> {
                events
                    .iter()
                    .filter_map(|event| match &event.event {
                        ProtocolEvent::Http(HttpEvent::Started { request, .. }) => {
                            Some(connection_index_from_path(&request.path))
                        }
                        _ => None,
                    })
                    .collect()
            }

            // number of maximal contiguous same-value runs. A fully
            // sequential schedule (connection 0 draining entirely, then
            // connection 1, ...) produces exactly `connection_count` runs;
            // genuine interleaving produces strictly more, smaller ones.
            fn count_runs(sequence: &[usize]) -> usize {
                if sequence.is_empty() {
                    return 0;
                }
                1 + sequence
                    .windows(2)
                    .filter(|pair| pair[0] != pair[1])
                    .count()
            }

            fn assert_multi_connection_determinism(connection_count: usize) {
                let (baseline, baseline_events) = run_multi_once(connection_count);
                assert!(!baseline.is_empty(), "recorded trace must not be empty");

                let sequence = started_connection_sequence(&baseline_events);
                let observed_runs = count_runs(&sequence);
                assert!(
                    observed_runs > connection_count,
                    "connections did not genuinely interleave: {connection_count} connections \
                     produced only {observed_runs} contiguous per-connection run(s) of Started \
                     events (a fully-sequential schedule -- connection 0 draining completely \
                     before connection 1 starts -- would itself produce exactly \
                     {connection_count}); recorded sequence was {sequence:?}"
                );

                for run_index in 1..MULTI_CONN_RUNS {
                    let (candidate, _candidate_events) = run_multi_once(connection_count);
                    if let Some(offset) = first_divergence(&baseline, &candidate) {
                        let window = 32;
                        let baseline_start = offset.saturating_sub(window);
                        let candidate_start = offset.saturating_sub(window);
                        let baseline_snippet = String::from_utf8_lossy(
                            &baseline[baseline_start..(offset + window).min(baseline.len())],
                        );
                        let candidate_snippet = String::from_utf8_lossy(
                            &candidate[candidate_start..(offset + window).min(candidate.len())],
                        );
                        panic!(
                            "run {run_index} diverged from run 0 at byte {offset} ({connection_count} connections)\n  \
                             run 0:  ...{baseline_snippet}...\n  \
                             run {run_index}: ...{candidate_snippet}..."
                        );
                    }
                }
            }

            #[test]
            fn multi_connection_trace_is_byte_identical_across_n_runs_four_connections() {
                assert_multi_connection_determinism(4);
            }

            #[test]
            fn multi_connection_trace_is_byte_identical_across_n_runs_sixteen_connections() {
                assert_multi_connection_determinism(16);
            }
        }
    }
}
