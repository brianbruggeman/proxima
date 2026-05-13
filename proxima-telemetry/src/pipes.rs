// terminal Pipes for the telemetry drainer. each pipe receives a typed
// Request<TelemetryRecord> and dispatches on the enum variant. no type erasure.

use alloc::sync::Arc;
use core::sync::atomic::Ordering;
use std::future::Future as StdFuture;

use bytes::Bytes;
use smallvec::SmallVec;

use crate::level::Level;
use crate::log::LogRecord;
use crate::metric::MetricSample;
use crate::tag::{ScalarValue, Tag};
use crate::trace::{EventRecord, SpanLink, SpanRecord};
#[cfg(feature = "elevation")]
use crate::id::TraceId;
#[cfg(feature = "elevation")]
use crate::log_buffer::ring::LogRing;
#[cfg(feature = "elevation")]
use core::sync::atomic::AtomicU64;
#[cfg(feature = "elevation")]
use dashmap::DashMap;
use proxima_primitives::pipe::Method;
use proxima_primitives::pipe::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::alloc_tier::SendDynPipe;
#[cfg(feature = "otlp-http")]
use proxima_primitives::pipe::handler::PipeHandle;
use proxima_primitives::pipe::request::{Request, RequestContext, Response};

/// Typed payload enum for all telemetry record shapes that flow through the
/// internal telemetry Pipe channel. Replaces the old `carry` type-erasure.
// single-record variants intentionally large; boxing would add heap alloc per span/event on the hot path
// Clone is for the multi-exporter fan ONLY: it clones the *BatchArc fan form
// (Vec<Arc<T>> spine + refcount bumps, cheap) to deliver one batch to N
// exporters. The drain hot path moves records and never clones.
#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
pub enum TelemetryRecord {
    Span(SpanRecord),
    Event(EventRecord),
    Log(LogRecord),
    Metric(MetricSample),
    Link(SpanLink),
    SpanBatch(alloc::vec::Vec<SpanRecord>),
    EventBatch(alloc::vec::Vec<EventRecord>),
    LogBatch(alloc::vec::Vec<LogRecord>),
    MetricBatch(alloc::vec::Vec<MetricSample>),
    LinkBatch(alloc::vec::Vec<SpanLink>),
    SpanBatchArc(alloc::vec::Vec<Arc<SpanRecord>>),
    EventBatchArc(alloc::vec::Vec<Arc<EventRecord>>),
    LogBatchArc(alloc::vec::Vec<Arc<LogRecord>>),
    MetricBatchArc(alloc::vec::Vec<Arc<MetricSample>>),
    LinkBatchArc(alloc::vec::Vec<Arc<SpanLink>>),
}

/// The request type for all internal telemetry pipes.
pub type TelemetryRequest = Request<TelemetryRecord>;

/// Runtime-erased handle for telemetry pipes. An instantiation of the
/// generic erased form `proxima_primitives::pipe::PipeHandle<In, Out>` — parallel
/// to the HTTP-shaped `proxima_primitives::pipe::handler::PipeHandle` but typed for
/// `TelemetryRequest` instead of `Request<Bytes>`.
pub type TelemetryPipeHandle = proxima_primitives::pipe::alloc_tier::PipeHandle<TelemetryRequest, Response<Bytes>>;

pub use proxima_primitives::pipe::alloc_tier::into_handle as into_telemetry_handle;

/// Fan one telemetry request to N exporters concurrently, behind the single
/// handle the recorder dispatches to. The N sinks are independent, so they run
/// at once — the drain waits on the slowest exporter, not the sum of them
/// (sequential delivery would throttle the drainer behind serial network
/// waits). The recorder and drainer are unchanged; the single-exporter path is
/// untouched.
///
/// `TelemetryRequest` is not `Clone` (its `stream` field is not), so the
/// PRIMARY exporter (index 0) receives the original by move and ITS result is
/// the fan's result; each SECONDARY receives a request rebuilt from a cloned
/// record (cheap for the `*BatchArc` drain form). Secondary errors are
/// best-effort — one broken exporter must not fail the others.
pub struct FanExporter {
    exporters: Arc<Vec<TelemetryPipeHandle>>,
}

impl FanExporter {
    #[must_use]
    pub fn new(exporters: Vec<TelemetryPipeHandle>) -> Self {
        Self {
            exporters: Arc::new(exporters),
        }
    }
}

impl SendPipe for FanExporter {
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: TelemetryRequest,
    ) -> impl StdFuture<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let exporters = Arc::clone(&self.exporters);
        async move {
            let Some((primary, secondaries)) = exporters.split_first() else {
                return Err(ProximaError::Config("fan_exporters: no exporters".into()));
            };
            let secondary_calls: Vec<_> = secondaries
                .iter()
                .map(|exporter| {
                    exporter.call_dyn(rebuild_request(&request, request.payload.clone()))
                })
                .collect();
            let (primary_result, _) = futures::future::join(
                primary.call_dyn(request),
                futures::future::join_all(secondary_calls),
            )
            .await;
            primary_result
        }
    }
}

fn rebuild_request(template: &TelemetryRequest, record: TelemetryRecord) -> TelemetryRequest {
    Request {
        method: template.method.clone(),
        path: template.path.clone(),
        query: template.query.clone(),
        metadata: template.metadata.clone(),
        payload: record,
        stream: None,
        context: template.context.clone(),
    }
}

/// Retain only floor-and-above log records in a request, then forward to the
/// normal exporter. Installed as arm A of the elevation fan-out: below-floor
/// records — admitted into the shared ring only for verbose-sampled traces —
/// reach this arm too, and this is where they are dropped so the normal sink
/// stays floor+ exactly as it is today. Non-log payloads pass through untouched.
///
/// It is a specific telemetry leaf pipe (like [`FanExporter`] / [`NullPipe`]),
/// not a general filter combinator — the pipe algebra composes it via
/// `and_then`/fan-out; it does not need a new library primitive.
#[cfg(feature = "elevation")]
pub struct FloorFilter {
    inner: TelemetryPipeHandle,
    floor_severity: u8,
}

#[cfg(feature = "elevation")]
impl FloorFilter {
    #[must_use]
    pub fn new(floor: Level, inner: TelemetryPipeHandle) -> Self {
        Self {
            inner,
            floor_severity: floor.severity(),
        }
    }
}

#[cfg(feature = "elevation")]
fn retain_floor(mut request: TelemetryRequest, floor_severity: u8) -> TelemetryRequest {
    let payload = core::mem::replace(&mut request.payload, TelemetryRecord::LogBatch(Vec::new()));
    request.payload = match payload {
        TelemetryRecord::LogBatch(records) => TelemetryRecord::LogBatch(
            records
                .into_iter()
                .filter(|record| record.level.severity() >= floor_severity)
                .collect(),
        ),
        TelemetryRecord::LogBatchArc(records) => TelemetryRecord::LogBatchArc(
            records
                .into_iter()
                .filter(|record| record.level.severity() >= floor_severity)
                .collect(),
        ),
        TelemetryRecord::Log(record) if record.level.severity() < floor_severity => {
            TelemetryRecord::LogBatch(Vec::new())
        }
        other => other,
    };
    request
}

#[cfg(feature = "elevation")]
impl SendPipe for FloorFilter {
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: TelemetryRequest,
    ) -> impl StdFuture<Output = Result<Response<Bytes>, ProximaError>> + Send {
        // filtering is synchronous; the returned future is the inner handle's own.
        self.inner.call_dyn(retain_floor(request, self.floor_severity))
    }
}

/// Per-trace replay buffer: a bounded ring of the trace's records plus its last
/// activity timestamp (record `ts_ns`, used for TTL / LRU — no wall clock).
#[cfg(feature = "elevation")]
struct TraceBuffer {
    ring: LogRing<LogRecord>,
    last_touch_ns: AtomicU64,
}

#[cfg(feature = "elevation")]
impl TraceBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            ring: LogRing::new(capacity),
            last_touch_ns: AtomicU64::new(0),
        }
    }
}

// sweep the TTL map once every N calls — amortised, off the per-record path.
#[cfg(feature = "elevation")]
const SWEEP_EVERY: u64 = 64;

/// The elevation buffer sink — arm B of the fan-out. It buffers the records of
/// verbose-sampled traces (marked [`TraceFlags::VERBOSE_BUFFERED`]) per
/// `trace_id`; a trigger-level record replays that trace's full ordered tree to
/// a separate elevated exporter. Non-verbose records cost it a flag check and a
/// drop.
///
/// Pipe outside, atomic state inside (the [`FanIn`] pattern): the `SendPipe`
/// composes by type; the shared `Arc<ElevationState>` holds the concurrent
/// per-trace map. Eviction is layered — root-span close (semantic completion),
/// TTL (lost-root fallback), and a hard count-cap (OOM backstop).
///
/// [`FanIn`]: proxima_primitives::pipe::FanIn
/// [`TraceFlags::VERBOSE_BUFFERED`]: crate::id::TraceFlags::VERBOSE_BUFFERED
#[cfg(feature = "elevation")]
pub struct ElevationSink {
    state: Arc<ElevationState>,
}

#[cfg(feature = "elevation")]
struct ElevationState {
    elevated: TelemetryPipeHandle,
    buffers: DashMap<TraceId, Arc<TraceBuffer>>,
    trigger_severity: u8,
    per_trace_ring: usize,
    max_traces: usize,
    ttl_ns: u64,
    drain_on_root_close: bool,
    latest_ts_ns: AtomicU64,
    sweep_counter: AtomicU64,
}

#[cfg(feature = "elevation")]
impl ElevationSink {
    /// `ttl_ns == 0` disables the TTL sweep; `max_traces`/`per_trace_ring` are
    /// already resolved (0 replaced with the build-time sized default upstream).
    #[must_use]
    pub fn new(
        elevated: TelemetryPipeHandle,
        trigger: Level,
        per_trace_ring: usize,
        max_traces: usize,
        ttl_ns: u64,
        drain_on_root_close: bool,
    ) -> Self {
        Self {
            state: Arc::new(ElevationState {
                elevated,
                buffers: DashMap::new(),
                trigger_severity: trigger.severity(),
                per_trace_ring,
                max_traces,
                ttl_ns,
                drain_on_root_close,
                latest_ts_ns: AtomicU64::new(0),
                sweep_counter: AtomicU64::new(0),
            }),
        }
    }
}

#[cfg(feature = "elevation")]
impl ElevationState {
    fn buffer_for(&self, trace_id: TraceId) -> Arc<TraceBuffer> {
        if let Some(existing) = self.buffers.get(&trace_id) {
            return Arc::clone(existing.value());
        }
        self.enforce_cap();
        Arc::clone(
            self.buffers
                .entry(trace_id)
                .or_insert_with(|| Arc::new(TraceBuffer::new(self.per_trace_ring)))
                .value(),
        )
    }

    // hard OOM backstop: at cap, evict the least-recently-touched trace (one
    // scan, only when full). TraceId is Copy, so no map guard is held over remove.
    fn enforce_cap(&self) {
        if self.buffers.len() < self.max_traces {
            return;
        }
        let victim = self
            .buffers
            .iter()
            .min_by_key(|entry| entry.value().last_touch_ns.load(Ordering::Relaxed))
            .map(|entry| *entry.key());
        if let Some(trace_id) = victim {
            self.buffers.remove(&trace_id);
        }
    }

    // drain a trace's ring as an ordered (ts_ns) replay request, if non-empty.
    fn drain_trace(&self, buffer: &TraceBuffer) -> Option<TelemetryRequest> {
        let mut records = buffer.ring.snapshot(None);
        if records.is_empty() {
            return None;
        }
        records.sort_by_key(|record| record.ts_ns);
        Some(log_batch_request(records))
    }

    fn ingest_log(&self, record: &LogRecord, replays: &mut Vec<TelemetryRequest>) {
        if !record.trace_flags.is_verbose_buffered() {
            return;
        }
        let Some(trace_id) = record.trace_id else {
            return;
        };
        self.latest_ts_ns.fetch_max(record.ts_ns, Ordering::Relaxed);
        let buffer = self.buffer_for(trace_id);
        buffer.ring.push(record.clone());
        buffer.last_touch_ns.store(record.ts_ns, Ordering::Relaxed);
        if record.level.severity() >= self.trigger_severity
            && let Some((_, triggered)) = self.buffers.remove(&trace_id)
            && let Some(replay) = self.drain_trace(&triggered)
        {
            replays.push(replay);
        }
    }

    // root-span close is the semantic completion signal: a completed trace that
    // never triggered drops its buffer (a triggered one is already removed).
    fn observe_span(&self, span: &SpanRecord) {
        if self.drain_on_root_close && span.parent_span_id.is_none() {
            self.buffers.remove(&span.trace_id);
        }
    }

    fn ingest(&self, request: &TelemetryRequest) -> Vec<TelemetryRequest> {
        let mut replays = Vec::new();
        match &request.payload {
            TelemetryRecord::Log(record) => self.ingest_log(record, &mut replays),
            TelemetryRecord::LogBatch(records) => {
                for record in records {
                    self.ingest_log(record, &mut replays);
                }
            }
            TelemetryRecord::LogBatchArc(records) => {
                for record in records {
                    self.ingest_log(record, &mut replays);
                }
            }
            TelemetryRecord::Span(span) => self.observe_span(span),
            TelemetryRecord::SpanBatch(spans) => {
                for span in spans {
                    self.observe_span(span);
                }
            }
            TelemetryRecord::SpanBatchArc(spans) => {
                for span in spans {
                    self.observe_span(span);
                }
            }
            _ => {}
        }
        replays
    }

    fn maybe_sweep(&self) {
        if self.ttl_ns == 0 {
            return;
        }
        if !self
            .sweep_counter
            .fetch_add(1, Ordering::Relaxed)
            .is_multiple_of(SWEEP_EVERY)
        {
            return;
        }
        let cutoff = self
            .latest_ts_ns
            .load(Ordering::Relaxed)
            .saturating_sub(self.ttl_ns);
        self.buffers
            .retain(|_, buffer| buffer.last_touch_ns.load(Ordering::Relaxed) >= cutoff);
    }
}

#[cfg(feature = "elevation")]
impl SendPipe for ElevationSink {
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: TelemetryRequest,
    ) -> impl StdFuture<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let state = Arc::clone(&self.state);
        async move {
            for replay in state.ingest(&request) {
                // best-effort: a broken elevated sink must not fail the drain.
                let _ = state.elevated.call_dyn(replay).await;
            }
            state.maybe_sweep();
            Ok(ok_response())
        }
    }
}

/// Compose N exporters into one handle: 0 -> no-op `NullPipe`, 1 -> the handle
/// itself (zero fan overhead), >=2 -> `FanExporter`.
#[must_use]
pub fn fan_exporters(mut exporters: Vec<TelemetryPipeHandle>) -> TelemetryPipeHandle {
    match exporters.len() {
        0 => into_telemetry_handle(NullPipe::new()),
        1 => exporters
            .pop()
            .unwrap_or_else(|| into_telemetry_handle(NullPipe::new())),
        _ => into_telemetry_handle(FanExporter::new(exporters)),
    }
}

pub const METHOD_LOG: &[u8] = b"LOG";
pub const METHOD_SPAN_START: &[u8] = b"SPAN_START";
pub const METHOD_SPAN_END: &[u8] = b"SPAN_END";
pub const METHOD_EVENT: &[u8] = b"EVENT";
pub const METHOD_COUNTER_ADD: &[u8] = b"COUNTER_ADD";
pub const METHOD_HIST_RECORD: &[u8] = b"HIST_RECORD";
pub const METHOD_LINK: &[u8] = b"LINK";

pub const METHOD_SPAN_BATCH: &[u8] = b"SPAN_BATCH";
pub const METHOD_EVENT_BATCH: &[u8] = b"EVENT_BATCH";
pub const METHOD_LOG_BATCH: &[u8] = b"LOG_BATCH";
pub const METHOD_METRIC_BATCH: &[u8] = b"METRIC_BATCH";
pub const METHOD_LINK_BATCH: &[u8] = b"LINK_BATCH";

pub const METHOD_SPAN_BATCH_ARC: &[u8] = b"SPAN_BATCH_ARC";
pub const METHOD_EVENT_BATCH_ARC: &[u8] = b"EVENT_BATCH_ARC";
pub const METHOD_LOG_BATCH_ARC: &[u8] = b"LOG_BATCH_ARC";
pub const METHOD_METRIC_BATCH_ARC: &[u8] = b"METRIC_BATCH_ARC";
pub const METHOD_LINK_BATCH_ARC: &[u8] = b"LINK_BATCH_ARC";

pub const PATH_LOG: &[u8] = b"/log";
pub const PATH_SPAN: &[u8] = b"/span";
pub const PATH_EVENT: &[u8] = b"/event";
pub const PATH_METRIC_COUNTER: &[u8] = b"/metric/counter";
pub const PATH_METRIC_HISTOGRAM: &[u8] = b"/metric/histogram";
pub const PATH_LINK: &[u8] = b"/link";

fn make_request(
    method: &'static [u8],
    path: &'static [u8],
    record: TelemetryRecord,
) -> TelemetryRequest {
    Request {
        method: Method::from_wire(Bytes::from_static(method)),
        path: Bytes::from_static(path),
        query: proxima_primitives::pipe::header_list::HeaderList::new(),
        metadata: proxima_primitives::pipe::header_list::HeaderList::new(),
        payload: record,
        stream: None,
        context: RequestContext::default(),
    }
}

/// Build a telemetry Request envelope for a SpanRecord (end of span).
#[must_use]
pub fn span_request(record: SpanRecord) -> TelemetryRequest {
    make_request(METHOD_SPAN_END, PATH_SPAN, TelemetryRecord::Span(record))
}

/// Build a telemetry Request envelope for an EventRecord.
#[must_use]
pub fn event_request(record: EventRecord) -> TelemetryRequest {
    make_request(METHOD_EVENT, PATH_EVENT, TelemetryRecord::Event(record))
}

/// Build a telemetry Request envelope for a LogRecord.
#[must_use]
pub fn log_request(record: LogRecord) -> TelemetryRequest {
    make_request(METHOD_LOG, PATH_LOG, TelemetryRecord::Log(record))
}

/// Build a telemetry Request envelope for a MetricSample.
#[must_use]
pub fn metric_request(sample: MetricSample) -> TelemetryRequest {
    make_request(
        METHOD_COUNTER_ADD,
        PATH_METRIC_COUNTER,
        TelemetryRecord::Metric(sample),
    )
}

/// Build a telemetry Request envelope for a SpanLink.
#[must_use]
pub fn link_request(link: SpanLink) -> TelemetryRequest {
    make_request(METHOD_LINK, PATH_LINK, TelemetryRecord::Link(link))
}

/// Build a batch telemetry Request carrying a Vec<SpanRecord>.
#[must_use]
pub fn span_batch_request(records: alloc::vec::Vec<SpanRecord>) -> TelemetryRequest {
    make_request(
        METHOD_SPAN_BATCH,
        PATH_SPAN,
        TelemetryRecord::SpanBatch(records),
    )
}

/// Build a batch telemetry Request carrying a Vec<EventRecord>.
#[must_use]
pub fn event_batch_request(records: alloc::vec::Vec<EventRecord>) -> TelemetryRequest {
    make_request(
        METHOD_EVENT_BATCH,
        PATH_EVENT,
        TelemetryRecord::EventBatch(records),
    )
}

/// Build a batch telemetry Request carrying a Vec<LogRecord>.
#[must_use]
pub fn log_batch_request(records: alloc::vec::Vec<LogRecord>) -> TelemetryRequest {
    make_request(
        METHOD_LOG_BATCH,
        PATH_LOG,
        TelemetryRecord::LogBatch(records),
    )
}

/// Build a batch telemetry Request carrying a Vec<MetricSample>.
#[must_use]
pub fn metric_batch_request(samples: alloc::vec::Vec<MetricSample>) -> TelemetryRequest {
    make_request(
        METHOD_METRIC_BATCH,
        PATH_METRIC_COUNTER,
        TelemetryRecord::MetricBatch(samples),
    )
}

/// Build a batch telemetry Request carrying a Vec<SpanLink>.
#[must_use]
pub fn link_batch_request(links: alloc::vec::Vec<SpanLink>) -> TelemetryRequest {
    make_request(
        METHOD_LINK_BATCH,
        PATH_LINK,
        TelemetryRecord::LinkBatch(links),
    )
}

/// Build an Arc-shared batch telemetry Request carrying a Vec<Arc<SpanRecord>>.
///
/// Fan-out via Tee becomes N Arc bumps instead of N record memcpys. Produced by
/// the drainer when `RecordSharing::Arc` is configured.
#[must_use]
pub fn span_batch_arc_request(records: alloc::vec::Vec<Arc<SpanRecord>>) -> TelemetryRequest {
    make_request(
        METHOD_SPAN_BATCH_ARC,
        PATH_SPAN,
        TelemetryRecord::SpanBatchArc(records),
    )
}

/// Build an Arc-shared batch telemetry Request carrying a Vec<Arc<EventRecord>>.
#[must_use]
pub fn event_batch_arc_request(records: alloc::vec::Vec<Arc<EventRecord>>) -> TelemetryRequest {
    make_request(
        METHOD_EVENT_BATCH_ARC,
        PATH_EVENT,
        TelemetryRecord::EventBatchArc(records),
    )
}

/// Build an Arc-shared batch telemetry Request carrying a Vec<Arc<LogRecord>>.
#[must_use]
pub fn log_batch_arc_request(records: alloc::vec::Vec<Arc<LogRecord>>) -> TelemetryRequest {
    make_request(
        METHOD_LOG_BATCH_ARC,
        PATH_LOG,
        TelemetryRecord::LogBatchArc(records),
    )
}

/// Build an Arc-shared batch telemetry Request carrying a Vec<Arc<MetricSample>>.
#[must_use]
pub fn metric_batch_arc_request(samples: alloc::vec::Vec<Arc<MetricSample>>) -> TelemetryRequest {
    make_request(
        METHOD_METRIC_BATCH_ARC,
        PATH_METRIC_COUNTER,
        TelemetryRecord::MetricBatchArc(samples),
    )
}

/// Build an Arc-shared batch telemetry Request carrying a Vec<Arc<SpanLink>>.
#[must_use]
pub fn link_batch_arc_request(links: alloc::vec::Vec<Arc<SpanLink>>) -> TelemetryRequest {
    make_request(
        METHOD_LINK_BATCH_ARC,
        PATH_LINK,
        TelemetryRecord::LinkBatchArc(links),
    )
}

fn ok_response() -> Response<Bytes> {
    Response::ok(bytes::Bytes::new())
}

/// No-op terminal sink. Returns Ok immediately without touching the record.
/// Default when no real exporter is configured.
pub struct NullPipe;

impl NullPipe {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for NullPipe {
    fn default() -> Self {
        Self::new()
    }
}

impl SendPipe for NullPipe {
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: TelemetryRequest,
    ) -> impl StdFuture<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move { Ok(ok_response()) }
    }
}

/// Terminal sink that writes telemetry records to a `FrameSink` using the
/// native postcard wire format.
pub struct NativePipe<S: crate::out::native::FrameSink> {
    inner: crate::out::native::NativeExporter<S>,
}

impl<S: crate::out::native::FrameSink + 'static> NativePipe<S> {
    #[must_use]
    pub fn new(sink: S) -> Self {
        Self {
            inner: crate::out::native::NativeExporter::new(sink),
        }
    }

    #[must_use]
    pub fn schema_version(self, version: u8) -> Self {
        Self {
            inner: self.inner.schema_version(version),
        }
    }
}

impl<S: crate::out::native::FrameSink + 'static> SendPipe for NativePipe<S> {
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: TelemetryRequest,
    ) -> impl StdFuture<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let result = dispatch_native(&self.inner, request);
        async move { result }
    }
}

fn dispatch_native<S: crate::out::native::FrameSink>(
    exporter: &crate::out::native::NativeExporter<S>,
    request: TelemetryRequest,
) -> Result<Response<Bytes>, ProximaError> {
    use crate::out::native::{
        NativePayloadRef, event_to_native_ref, link_to_native_ref, log_to_native_ref,
        metric_to_native_ref, span_to_native_ref,
    };

    match request.payload {
        TelemetryRecord::SpanBatch(records) => {
            for record in &records {
                exporter.encode_and_emit_payload_ref(NativePayloadRef::Span(span_to_native_ref(
                    record,
                )));
            }
        }
        TelemetryRecord::SpanBatchArc(records) => {
            for record in &records {
                exporter.encode_and_emit_payload_ref(NativePayloadRef::Span(span_to_native_ref(
                    record.as_ref(),
                )));
            }
        }
        TelemetryRecord::EventBatch(records) => {
            for record in &records {
                exporter.encode_and_emit_payload_ref(NativePayloadRef::Event(event_to_native_ref(
                    record,
                )));
            }
        }
        TelemetryRecord::EventBatchArc(records) => {
            for record in &records {
                exporter.encode_and_emit_payload_ref(NativePayloadRef::Event(event_to_native_ref(
                    record.as_ref(),
                )));
            }
        }
        TelemetryRecord::LogBatch(records) => {
            for record in &records {
                exporter
                    .encode_and_emit_payload_ref(NativePayloadRef::Log(log_to_native_ref(record)));
            }
        }
        TelemetryRecord::LogBatchArc(records) => {
            for record in &records {
                exporter.encode_and_emit_payload_ref(NativePayloadRef::Log(log_to_native_ref(
                    record.as_ref(),
                )));
            }
        }
        TelemetryRecord::MetricBatch(samples) => {
            for sample in &samples {
                exporter.encode_and_emit_payload_ref(NativePayloadRef::Metric(
                    metric_to_native_ref(sample),
                ));
            }
        }
        TelemetryRecord::MetricBatchArc(samples) => {
            for sample in &samples {
                exporter.encode_and_emit_payload_ref(NativePayloadRef::Metric(
                    metric_to_native_ref(sample.as_ref()),
                ));
            }
        }
        TelemetryRecord::LinkBatch(links) => {
            for link in &links {
                exporter
                    .encode_and_emit_payload_ref(NativePayloadRef::Link(link_to_native_ref(link)));
            }
        }
        TelemetryRecord::LinkBatchArc(links) => {
            for link in &links {
                exporter.encode_and_emit_payload_ref(NativePayloadRef::Link(link_to_native_ref(
                    link.as_ref(),
                )));
            }
        }
        TelemetryRecord::Span(record) => {
            exporter
                .encode_and_emit_payload_ref(NativePayloadRef::Span(span_to_native_ref(&record)));
        }
        TelemetryRecord::Event(record) => {
            exporter
                .encode_and_emit_payload_ref(NativePayloadRef::Event(event_to_native_ref(&record)));
        }
        TelemetryRecord::Log(record) => {
            exporter.encode_and_emit_payload_ref(NativePayloadRef::Log(log_to_native_ref(&record)));
        }
        TelemetryRecord::Metric(sample) => {
            exporter.encode_and_emit_payload_ref(NativePayloadRef::Metric(metric_to_native_ref(
                &sample,
            )));
        }
        TelemetryRecord::Link(link) => {
            exporter.encode_and_emit_payload_ref(NativePayloadRef::Link(link_to_native_ref(&link)));
        }
    }

    Ok(ok_response())
}

/// Terminal sink that buffers telemetry records in OTLP/HTTP protobuf format.
/// Call `flush()` to encode and return the pending payload bytes.
#[cfg(feature = "otlp-http")]
pub struct OtlpHttpPipe {
    inner: crate::out::otlp_http::OtlpHttpExporter,
}

#[cfg(feature = "otlp-http")]
impl OtlpHttpPipe {
    #[must_use]
    pub fn new(endpoint: impl Into<alloc::string::String>) -> Self {
        Self {
            inner: crate::out::otlp_http::OtlpHttpExporter::new(endpoint),
        }
    }

    /// Encode and drain pending spans/logs/metrics as OTLP protobuf Bytes.
    #[must_use]
    pub fn flush_spans(&self) -> bytes::Bytes {
        self.inner.encode_spans()
    }

    #[must_use]
    pub fn flush_logs(&self) -> bytes::Bytes {
        self.inner.encode_logs()
    }

    #[must_use]
    pub fn flush_metrics(&self) -> bytes::Bytes {
        self.inner.encode_metrics()
    }
}

#[cfg(feature = "otlp-http")]
impl SendPipe for OtlpHttpPipe {
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: TelemetryRequest,
    ) -> impl StdFuture<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let result = dispatch_otlp_http(&self.inner, request);
        async move { result }
    }
}

/// OTLP/HTTP protocol codec stage. Pure protocol transform: it encodes the
/// drained batch to OTLP protobuf, rewrites it into a `POST /v1/{traces,logs,
/// metrics}` request, and calls the next stage (`downstream`). It owns no
/// transport — `downstream` is whatever the config chain put next (an HTTP
/// client, or a retry / tls / timeout wrapper around one).
///
/// This is the *protocol* axis of the exporter (protocol × transport × auth):
/// the only part that is irreducibly OTLP-specific. Transport and the rest are
/// sibling stages the caller composes downstream — so the leaf crate never
/// depends on an HTTP client, and "OTLP over the wire" is config, not a bespoke
/// send type. The caller injects `downstream` via
/// [`into_handle`](proxima_primitives::pipe::alloc_tier::into_handle); the umbrella's config
/// builder is what wires `codec -> transport` from `ExporterChoice::OtlpHttp`.
#[cfg(feature = "otlp-http")]
pub struct OtlpHttpCodec {
    encoder: crate::out::otlp_http::OtlpHttpExporter,
    downstream: PipeHandle,
}

#[cfg(feature = "otlp-http")]
impl OtlpHttpCodec {
    /// `downstream` is the next pipe stage (an HTTP client, or a retry / tls
    /// wrapper around one). Each drained batch is encoded to OTLP protobuf and
    /// POSTed through it.
    #[must_use]
    pub fn new(downstream: PipeHandle) -> Self {
        Self {
            encoder: crate::out::otlp_http::OtlpHttpExporter::new(""),
            downstream,
        }
    }
}

#[cfg(feature = "otlp-http")]
impl SendPipe for OtlpHttpCodec {
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: TelemetryRequest,
    ) -> impl StdFuture<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let otlp_path: Option<&'static str> = match &request.payload {
            TelemetryRecord::SpanBatch(_)
            | TelemetryRecord::SpanBatchArc(_)
            | TelemetryRecord::Span(_) => Some("/v1/traces"),
            TelemetryRecord::LogBatch(_)
            | TelemetryRecord::LogBatchArc(_)
            | TelemetryRecord::Log(_) => Some("/v1/logs"),
            TelemetryRecord::MetricBatch(_)
            | TelemetryRecord::MetricBatchArc(_)
            | TelemetryRecord::Metric(_) => Some("/v1/metrics"),
            _ => None,
        };
        let buffered = dispatch_otlp_http(&self.encoder, request);
        let body_fn: Option<(&'static str, bytes::Bytes)> = otlp_path.map(|path| {
            let body = match path {
                "/v1/traces" => self.encoder.encode_spans(),
                "/v1/logs" => self.encoder.encode_logs(),
                _ => self.encoder.encode_metrics(),
            };
            (path, body)
        });
        let downstream = Arc::clone(&self.downstream);
        async move {
            buffered?;
            let Some((path, body)) = body_fn else {
                return Ok(Response::ok(Bytes::new()));
            };
            if body.is_empty() {
                return Ok(Response::ok(Bytes::new()));
            }
            let post = Request::builder()
                .method("POST")
                .path(path)
                .header("content-type", "application/x-protobuf")
                .payload(body)
                .build()?;
            SendPipe::call(downstream.as_ref(), post).await
        }
    }
}

#[cfg(feature = "otlp-http")]
fn dispatch_otlp_http(
    exporter: &crate::out::otlp_http::OtlpHttpExporter,
    request: TelemetryRequest,
) -> Result<Response<Bytes>, ProximaError> {
    use crate::out::otlp_http::conv::{
        event_to_proto, link_to_proto, log_to_proto, metric_to_proto, span_to_proto,
    };
    use crate::out::otlp_http::proto;

    match request.payload {
        TelemetryRecord::SpanBatch(records) => {
            let mut locked = exporter.pending_spans.lock();
            for record in &records {
                locked.push(span_to_proto(record));
            }
        }
        TelemetryRecord::SpanBatchArc(records) => {
            let mut locked = exporter.pending_spans.lock();
            for record in &records {
                locked.push(span_to_proto(record.as_ref()));
            }
        }
        TelemetryRecord::EventBatch(records) => {
            let mut locked = exporter.pending_spans.lock();
            for record in &records {
                let event = event_to_proto(record);
                let proto_event = proto::SpanEvent {
                    time_unix_nano: event.time_unix_nano,
                    name: event.name,
                    attributes: event.attributes,
                    dropped_attributes_count: 0,
                };
                if let Some(last_span) = locked.last_mut() {
                    last_span.events.push(proto_event);
                }
            }
        }
        TelemetryRecord::EventBatchArc(records) => {
            let mut locked = exporter.pending_spans.lock();
            for record in &records {
                let event = event_to_proto(record.as_ref());
                let proto_event = proto::SpanEvent {
                    time_unix_nano: event.time_unix_nano,
                    name: event.name,
                    attributes: event.attributes,
                    dropped_attributes_count: 0,
                };
                if let Some(last_span) = locked.last_mut() {
                    last_span.events.push(proto_event);
                }
            }
        }
        TelemetryRecord::LogBatch(records) => {
            let mut locked = exporter.pending_logs.lock();
            for record in &records {
                locked.push(log_to_proto(record));
            }
        }
        TelemetryRecord::LogBatchArc(records) => {
            let mut locked = exporter.pending_logs.lock();
            for record in &records {
                locked.push(log_to_proto(record.as_ref()));
            }
        }
        TelemetryRecord::MetricBatch(samples) => {
            let mut locked = exporter.pending_metrics.lock();
            for sample in &samples {
                locked.push(metric_to_proto(sample));
            }
        }
        TelemetryRecord::MetricBatchArc(samples) => {
            let mut locked = exporter.pending_metrics.lock();
            for sample in &samples {
                locked.push(metric_to_proto(sample.as_ref()));
            }
        }
        TelemetryRecord::LinkBatch(links) => {
            let mut locked = exporter.pending_spans.lock();
            for link in &links {
                let proto_link = link_to_proto(link);
                if let Some(last_span) = locked.last_mut() {
                    last_span.links.push(proto_link);
                }
            }
        }
        TelemetryRecord::LinkBatchArc(links) => {
            let mut locked = exporter.pending_spans.lock();
            for link in &links {
                let proto_link = link_to_proto(link.as_ref());
                if let Some(last_span) = locked.last_mut() {
                    last_span.links.push(proto_link);
                }
            }
        }
        TelemetryRecord::Span(record) => {
            exporter.pending_spans.lock().push(span_to_proto(&record));
        }
        TelemetryRecord::Event(record) => {
            let event = event_to_proto(&record);
            let proto_event = proto::SpanEvent {
                time_unix_nano: event.time_unix_nano,
                name: event.name,
                attributes: event.attributes,
                dropped_attributes_count: 0,
            };
            let mut locked = exporter.pending_spans.lock();
            if let Some(last_span) = locked.last_mut() {
                last_span.events.push(proto_event);
            }
        }
        TelemetryRecord::Log(record) => {
            exporter.pending_logs.lock().push(log_to_proto(&record));
        }
        TelemetryRecord::Metric(sample) => {
            exporter
                .pending_metrics
                .lock()
                .push(metric_to_proto(&sample));
        }
        TelemetryRecord::Link(link) => {
            let proto_link = link_to_proto(&link);
            let mut locked = exporter.pending_spans.lock();
            if let Some(last_span) = locked.last_mut() {
                last_span.links.push(proto_link);
            }
        }
    }

    Ok(ok_response())
}

/// Terminal sink that buffers telemetry records in OTLP/gRPC framed format.
/// Call `flush_*` to encode and return the pending payload bytes with gRPC framing.
#[cfg(feature = "otlp-grpc")]
pub struct OtlpGrpcPipe {
    inner: crate::out::otlp_grpc::OtlpGrpcExporter,
}

#[cfg(feature = "otlp-grpc")]
impl OtlpGrpcPipe {
    #[must_use]
    pub fn new(endpoint: impl Into<alloc::string::String>) -> Self {
        Self {
            inner: crate::out::otlp_grpc::OtlpGrpcExporter::new(endpoint),
        }
    }

    #[must_use]
    pub fn flush_spans(&self) -> bytes::Bytes {
        self.inner.encode_spans()
    }

    #[must_use]
    pub fn flush_logs(&self) -> bytes::Bytes {
        self.inner.encode_logs()
    }

    #[must_use]
    pub fn flush_metrics(&self) -> bytes::Bytes {
        self.inner.encode_metrics()
    }
}

#[cfg(feature = "otlp-grpc")]
impl SendPipe for OtlpGrpcPipe {
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: TelemetryRequest,
    ) -> impl StdFuture<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let result = dispatch_otlp_grpc(&self.inner, request);
        async move { result }
    }
}

#[cfg(feature = "otlp-grpc")]
fn dispatch_otlp_grpc(
    exporter: &crate::out::otlp_grpc::OtlpGrpcExporter,
    request: TelemetryRequest,
) -> Result<Response<Bytes>, ProximaError> {
    use crate::out::otlp_http::conv::{
        event_to_proto, link_to_proto, log_to_proto, metric_to_proto, span_to_proto,
    };
    use crate::out::otlp_http::proto;

    let inner = exporter.inner();
    match request.payload {
        TelemetryRecord::SpanBatch(records) => {
            let mut locked = inner.pending_spans.lock();
            for record in &records {
                locked.push(span_to_proto(record));
            }
        }
        TelemetryRecord::SpanBatchArc(records) => {
            let mut locked = inner.pending_spans.lock();
            for record in &records {
                locked.push(span_to_proto(record.as_ref()));
            }
        }
        TelemetryRecord::EventBatch(records) => {
            let mut locked = inner.pending_spans.lock();
            for record in &records {
                let event = event_to_proto(record);
                let proto_event = proto::SpanEvent {
                    time_unix_nano: event.time_unix_nano,
                    name: event.name,
                    attributes: event.attributes,
                    dropped_attributes_count: 0,
                };
                if let Some(last_span) = locked.last_mut() {
                    last_span.events.push(proto_event);
                }
            }
        }
        TelemetryRecord::EventBatchArc(records) => {
            let mut locked = inner.pending_spans.lock();
            for record in &records {
                let event = event_to_proto(record.as_ref());
                let proto_event = proto::SpanEvent {
                    time_unix_nano: event.time_unix_nano,
                    name: event.name,
                    attributes: event.attributes,
                    dropped_attributes_count: 0,
                };
                if let Some(last_span) = locked.last_mut() {
                    last_span.events.push(proto_event);
                }
            }
        }
        TelemetryRecord::LogBatch(records) => {
            let mut locked = inner.pending_logs.lock();
            for record in &records {
                locked.push(log_to_proto(record));
            }
        }
        TelemetryRecord::LogBatchArc(records) => {
            let mut locked = inner.pending_logs.lock();
            for record in &records {
                locked.push(log_to_proto(record.as_ref()));
            }
        }
        TelemetryRecord::MetricBatch(samples) => {
            let mut locked = inner.pending_metrics.lock();
            for sample in &samples {
                locked.push(metric_to_proto(sample));
            }
        }
        TelemetryRecord::MetricBatchArc(samples) => {
            let mut locked = inner.pending_metrics.lock();
            for sample in &samples {
                locked.push(metric_to_proto(sample.as_ref()));
            }
        }
        TelemetryRecord::LinkBatch(links) => {
            let mut locked = inner.pending_spans.lock();
            for link in &links {
                let proto_link = link_to_proto(link);
                if let Some(last_span) = locked.last_mut() {
                    last_span.links.push(proto_link);
                }
            }
        }
        TelemetryRecord::LinkBatchArc(links) => {
            let mut locked = inner.pending_spans.lock();
            for link in &links {
                let proto_link = link_to_proto(link.as_ref());
                if let Some(last_span) = locked.last_mut() {
                    last_span.links.push(proto_link);
                }
            }
        }
        TelemetryRecord::Span(record) => {
            inner.pending_spans.lock().push(span_to_proto(&record));
        }
        TelemetryRecord::Event(record) => {
            let event = event_to_proto(&record);
            let proto_event = proto::SpanEvent {
                time_unix_nano: event.time_unix_nano,
                name: event.name,
                attributes: event.attributes,
                dropped_attributes_count: 0,
            };
            let mut locked = inner.pending_spans.lock();
            if let Some(last_span) = locked.last_mut() {
                last_span.events.push(proto_event);
            }
        }
        TelemetryRecord::Log(record) => {
            inner.pending_logs.lock().push(log_to_proto(&record));
        }
        TelemetryRecord::Metric(sample) => {
            inner.pending_metrics.lock().push(metric_to_proto(&sample));
        }
        TelemetryRecord::Link(link) => {
            let proto_link = link_to_proto(&link);
            let mut locked = inner.pending_spans.lock();
            if let Some(last_span) = locked.last_mut() {
                last_span.links.push(proto_link);
            }
        }
    }

    Ok(ok_response())
}

/// Test-only pipe that counts records by type.
///
/// Shared counters are returned at construction time for assertions.
pub struct CountingPipe {
    pub spans: Arc<core::sync::atomic::AtomicU64>,
    pub events: Arc<core::sync::atomic::AtomicU64>,
    pub logs: Arc<core::sync::atomic::AtomicU64>,
    pub metrics: Arc<core::sync::atomic::AtomicU64>,
    pub links: Arc<core::sync::atomic::AtomicU64>,
}

/// A [`CountingPipe`] paired with its five shared counters
/// (spans, events, logs, metrics, links) for assertion access.
pub type CountingPipeWithCounters = (
    CountingPipe,
    Arc<core::sync::atomic::AtomicU64>,
    Arc<core::sync::atomic::AtomicU64>,
    Arc<core::sync::atomic::AtomicU64>,
    Arc<core::sync::atomic::AtomicU64>,
    Arc<core::sync::atomic::AtomicU64>,
);

impl CountingPipe {
    // returns the pipe plus its five shared counters for test assertions
    #[allow(clippy::type_complexity)]
    #[must_use]
    pub fn new() -> CountingPipeWithCounters {
        let spans = Arc::new(core::sync::atomic::AtomicU64::new(0));
        let events = Arc::new(core::sync::atomic::AtomicU64::new(0));
        let logs = Arc::new(core::sync::atomic::AtomicU64::new(0));
        let metrics = Arc::new(core::sync::atomic::AtomicU64::new(0));
        let links = Arc::new(core::sync::atomic::AtomicU64::new(0));
        (
            Self {
                spans: Arc::clone(&spans),
                events: Arc::clone(&events),
                logs: Arc::clone(&logs),
                metrics: Arc::clone(&metrics),
                links: Arc::clone(&links),
            },
            spans,
            events,
            logs,
            metrics,
            links,
        )
    }
}

impl Default for CountingPipe {
    fn default() -> Self {
        Self::new().0
    }
}

impl SendPipe for CountingPipe {
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: TelemetryRequest,
    ) -> impl StdFuture<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let payload = request.payload;
        let spans = Arc::clone(&self.spans);
        let events = Arc::clone(&self.events);
        let logs = Arc::clone(&self.logs);
        let metrics = Arc::clone(&self.metrics);
        let links = Arc::clone(&self.links);

        async move {
            match payload {
                TelemetryRecord::SpanBatch(records) => {
                    spans.fetch_add(records.len() as u64, Ordering::Relaxed)
                }
                TelemetryRecord::SpanBatchArc(records) => {
                    spans.fetch_add(records.len() as u64, Ordering::Relaxed)
                }
                TelemetryRecord::EventBatch(records) => {
                    events.fetch_add(records.len() as u64, Ordering::Relaxed)
                }
                TelemetryRecord::EventBatchArc(records) => {
                    events.fetch_add(records.len() as u64, Ordering::Relaxed)
                }
                TelemetryRecord::LogBatch(records) => {
                    logs.fetch_add(records.len() as u64, Ordering::Relaxed)
                }
                TelemetryRecord::LogBatchArc(records) => {
                    logs.fetch_add(records.len() as u64, Ordering::Relaxed)
                }
                TelemetryRecord::MetricBatch(samples) => {
                    metrics.fetch_add(samples.len() as u64, Ordering::Relaxed)
                }
                TelemetryRecord::MetricBatchArc(samples) => {
                    metrics.fetch_add(samples.len() as u64, Ordering::Relaxed)
                }
                TelemetryRecord::LinkBatch(batch) => {
                    links.fetch_add(batch.len() as u64, Ordering::Relaxed)
                }
                TelemetryRecord::LinkBatchArc(batch) => {
                    links.fetch_add(batch.len() as u64, Ordering::Relaxed)
                }
                TelemetryRecord::Span(_) => spans.fetch_add(1, Ordering::Relaxed),
                TelemetryRecord::Event(_) => events.fetch_add(1, Ordering::Relaxed),
                TelemetryRecord::Log(_) => logs.fetch_add(1, Ordering::Relaxed),
                TelemetryRecord::Metric(_) => metrics.fetch_add(1, Ordering::Relaxed),
                TelemetryRecord::Link(_) => links.fetch_add(1, Ordering::Relaxed),
            };
            Ok(ok_response())
        }
    }
}

/// Terminal sink that stores records in memory — matched sink for OTel's InMemorySpanExporter.
///
/// Clones each arriving record into a parking_lot Mutex-guarded Vec. Intended for
/// benches that compare proxima against `opentelemetry_sdk::trace::InMemorySpanExporter`
/// so both sides do equivalent work (lock + clone + push). Also useful in tests that
/// need to inspect records after drain.
///
/// `Clone` shares the same backing buffers (every field is an `Arc`): one handle
/// goes into the recorder, the other inspects after drain.
#[derive(Clone)]
pub struct InMemoryPipe {
    spans: Arc<parking_lot::Mutex<alloc::vec::Vec<SpanRecord>>>,
    events: Arc<parking_lot::Mutex<alloc::vec::Vec<EventRecord>>>,
    logs: Arc<parking_lot::Mutex<alloc::vec::Vec<LogRecord>>>,
    metrics: Arc<parking_lot::Mutex<alloc::vec::Vec<MetricSample>>>,
    links: Arc<parking_lot::Mutex<alloc::vec::Vec<SpanLink>>>,
}

impl InMemoryPipe {
    #[must_use]
    pub fn new() -> Self {
        Self {
            spans: Arc::new(parking_lot::Mutex::new(alloc::vec::Vec::new())),
            events: Arc::new(parking_lot::Mutex::new(alloc::vec::Vec::new())),
            logs: Arc::new(parking_lot::Mutex::new(alloc::vec::Vec::new())),
            metrics: Arc::new(parking_lot::Mutex::new(alloc::vec::Vec::new())),
            links: Arc::new(parking_lot::Mutex::new(alloc::vec::Vec::new())),
        }
    }

    #[must_use]
    pub fn spans(&self) -> alloc::vec::Vec<SpanRecord> {
        self.spans.lock().clone()
    }

    #[must_use]
    pub fn events(&self) -> alloc::vec::Vec<EventRecord> {
        self.events.lock().clone()
    }

    #[must_use]
    pub fn logs(&self) -> alloc::vec::Vec<LogRecord> {
        self.logs.lock().clone()
    }

    #[must_use]
    pub fn metrics(&self) -> alloc::vec::Vec<MetricSample> {
        self.metrics.lock().clone()
    }

    #[must_use]
    pub fn links(&self) -> alloc::vec::Vec<SpanLink> {
        self.links.lock().clone()
    }

    pub fn clear(&self) {
        self.spans.lock().clear();
        self.events.lock().clear();
        self.logs.lock().clear();
        self.metrics.lock().clear();
        self.links.lock().clear();
    }

    #[must_use]
    pub fn total(&self) -> usize {
        self.spans.lock().len()
            + self.events.lock().len()
            + self.logs.lock().len()
            + self.metrics.lock().len()
            + self.links.lock().len()
    }
}

impl Default for InMemoryPipe {
    fn default() -> Self {
        Self::new()
    }
}

impl SendPipe for InMemoryPipe {
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: TelemetryRequest,
    ) -> impl StdFuture<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let result = dispatch_in_memory(self, request);
        async move { result }
    }
}

fn dispatch_in_memory(
    pipe: &InMemoryPipe,
    request: TelemetryRequest,
) -> Result<Response<Bytes>, ProximaError> {
    match request.payload {
        TelemetryRecord::SpanBatch(records) => {
            pipe.spans.lock().extend(records);
        }
        TelemetryRecord::SpanBatchArc(records) => {
            pipe.spans
                .lock()
                .extend(records.iter().map(|arc| (**arc).clone()));
        }
        TelemetryRecord::EventBatch(records) => {
            pipe.events.lock().extend(records);
        }
        TelemetryRecord::EventBatchArc(records) => {
            pipe.events
                .lock()
                .extend(records.iter().map(|arc| (**arc).clone()));
        }
        TelemetryRecord::LogBatch(records) => {
            pipe.logs.lock().extend(records);
        }
        TelemetryRecord::LogBatchArc(records) => {
            pipe.logs
                .lock()
                .extend(records.iter().map(|arc| (**arc).clone()));
        }
        TelemetryRecord::MetricBatch(samples) => {
            pipe.metrics.lock().extend(samples);
        }
        TelemetryRecord::MetricBatchArc(samples) => {
            pipe.metrics
                .lock()
                .extend(samples.iter().map(|arc| (**arc).clone()));
        }
        TelemetryRecord::LinkBatch(links) => {
            pipe.links.lock().extend(links);
        }
        TelemetryRecord::LinkBatchArc(links) => {
            pipe.links
                .lock()
                .extend(links.iter().map(|arc| (**arc).clone()));
        }
        TelemetryRecord::Span(record) => {
            pipe.spans.lock().push(record);
        }
        TelemetryRecord::Event(record) => {
            pipe.events.lock().push(record);
        }
        TelemetryRecord::Log(record) => {
            pipe.logs.lock().push(record);
        }
        TelemetryRecord::Metric(sample) => {
            pipe.metrics.lock().push(sample);
        }
        TelemetryRecord::Link(link) => {
            pipe.links.lock().push(link);
        }
    }
    Ok(ok_response())
}

/// Format choice for `FormatterPipe`.
#[derive(Clone, Copy, Debug)]
pub enum LogFormat {
    /// Human-readable single-line format, similar to `tracing_subscriber::fmt`.
    Human,
    /// JSON one-object-per-line format.
    Json,
}

/// Terminal sink that formats records to a `std::io::Write` sink — matched sink for
/// `tracing_subscriber::fmt::layer`. Used as the terminal in benches that compare
/// proxima against tracing_subscriber::fmt so both sides do equivalent work (format +
/// write to the configured writer).
pub struct FormatterPipe<W: std::io::Write + Send + Sync + 'static> {
    writer: Arc<parking_lot::Mutex<W>>,
    format: LogFormat,
    capacity_bytes: usize,
}

impl<W: std::io::Write + Send + Sync + 'static> FormatterPipe<W> {
    #[must_use]
    pub fn new(writer: W, format: LogFormat) -> Self {
        Self {
            writer: Arc::new(parking_lot::Mutex::new(writer)),
            format,
            capacity_bytes: crate::sized::SINK_CAPACITY_BYTES,
        }
    }

    /// Override the per-drain-batch buffer preallocation (default the build-time
    /// floor [`crate::sized::SINK_CAPACITY_BYTES`]).
    #[must_use]
    pub fn with_capacity_bytes(mut self, capacity_bytes: usize) -> Self {
        self.capacity_bytes = capacity_bytes;
        self
    }
}

impl<W: std::io::Write + Send + Sync + 'static> SendPipe for FormatterPipe<W> {
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: TelemetryRequest,
    ) -> impl StdFuture<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let result = dispatch_formatter(self, request);
        async move { result }
    }
}

/// Severity at/above which a log is an operational problem (warn). The
/// `Level` severities are trace=1 debug=5 info=9 warn=13 error=17.
const STDERR_SEVERITY_FLOOR: u8 = 13;

/// Terminal log sink that ROUTES BY SEVERITY: trace/debug/info → stdout,
/// warn/error (and above) → stderr — the standard convention so diagnostics
/// and problems can be redirected independently. Non-log records
/// (spans/metrics/links) are dropped: those belong on the OTLP/native export
/// path, not the console. Use via [`crate::export::Exporter::std`].
pub struct StdSplitPipe<O = std::io::Stdout, E = std::io::Stderr> {
    out: Arc<parking_lot::Mutex<O>>,
    err: Arc<parking_lot::Mutex<E>>,
    format: LogFormat,
    capacity_bytes: usize,
}

impl StdSplitPipe<std::io::Stdout, std::io::Stderr> {
    #[must_use]
    pub fn new(format: LogFormat) -> Self {
        Self {
            out: Arc::new(parking_lot::Mutex::new(std::io::stdout())),
            err: Arc::new(parking_lot::Mutex::new(std::io::stderr())),
            format,
            capacity_bytes: crate::sized::SINK_CAPACITY_BYTES,
        }
    }
}

impl<O, E> StdSplitPipe<O, E>
where
    O: std::io::Write + Send + Sync + 'static,
    E: std::io::Write + Send + Sync + 'static,
{
    /// Construct over explicit writers — lets a test drive routing + batching
    /// against in-memory sinks instead of the process stdout/stderr.
    #[must_use]
    pub fn with_writers(out: O, err: E, format: LogFormat) -> Self {
        Self {
            out: Arc::new(parking_lot::Mutex::new(out)),
            err: Arc::new(parking_lot::Mutex::new(err)),
            format,
            capacity_bytes: crate::sized::SINK_CAPACITY_BYTES,
        }
    }

    /// Override the per-drain-batch buffer preallocation (per writer; default the
    /// build-time floor [`crate::sized::SINK_CAPACITY_BYTES`]).
    #[must_use]
    pub fn with_capacity_bytes(mut self, capacity_bytes: usize) -> Self {
        self.capacity_bytes = capacity_bytes;
        self
    }

    // route one log into the stdout- or stderr-bound buffer by severity. the
    // write itself is deferred to `dispatch` so the whole batch goes out in one
    // syscall per writer rather than one per record.
    fn route_log(
        &self,
        out_buf: &mut proxima_core::batch::BatchBuffer,
        err_buf: &mut proxima_core::batch::BatchBuffer,
        record: &LogRecord,
    ) {
        if record.level.severity() >= STDERR_SEVERITY_FLOOR {
            format_log(err_buf, record, self.format);
        } else {
            format_log(out_buf, record, self.format);
        }
    }

    fn dispatch(&self, request: TelemetryRequest) -> Result<Response<Bytes>, ProximaError> {
        let mut out_buf = proxima_core::batch::BatchBuffer::with_capacity(self.capacity_bytes);
        let mut err_buf = proxima_core::batch::BatchBuffer::with_capacity(self.capacity_bytes);
        match request.payload {
            TelemetryRecord::Log(record) => self.route_log(&mut out_buf, &mut err_buf, &record),
            TelemetryRecord::LogBatch(records) => {
                for record in &records {
                    self.route_log(&mut out_buf, &mut err_buf, record);
                }
            }
            TelemetryRecord::LogBatchArc(records) => {
                for record in &records {
                    self.route_log(&mut out_buf, &mut err_buf, record.as_ref());
                }
            }
            // spans / events / metrics / links are OTLP-bound, not console logs.
            _ => {}
        }
        // one write per writer per drain batch (stdout-bound + stderr-bound),
        // preserving severity routing while amortizing the syscall over the batch.
        if !out_buf.is_empty() {
            out_buf
                .flush_to(&mut proxima_core::batch::WriteSink(&mut *self.out.lock()))
                .map_err(|err| ProximaError::Body(alloc::format!("formatter write: {err}")))?;
        }
        if !err_buf.is_empty() {
            err_buf
                .flush_to(&mut proxima_core::batch::WriteSink(&mut *self.err.lock()))
                .map_err(|err| ProximaError::Body(alloc::format!("formatter write: {err}")))?;
        }
        Ok(ok_response())
    }
}

impl<O, E> SendPipe for StdSplitPipe<O, E>
where
    O: std::io::Write + Send + Sync + 'static,
    E: std::io::Write + Send + Sync + 'static,
{
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: TelemetryRequest,
    ) -> impl StdFuture<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let result = self.dispatch(request);
        async move { result }
    }
}

fn dispatch_formatter<W: std::io::Write + Send + Sync + 'static>(
    pipe: &FormatterPipe<W>,
    request: TelemetryRequest,
) -> Result<Response<Bytes>, ProximaError> {
    let format = pipe.format;
    let mut buf = proxima_core::batch::BatchBuffer::with_capacity(pipe.capacity_bytes);

    match request.payload {
        TelemetryRecord::Log(record) => format_log(&mut buf, &record, format),
        TelemetryRecord::LogBatch(records) => {
            for record in &records {
                format_log(&mut buf, record, format);
            }
        }
        TelemetryRecord::LogBatchArc(records) => {
            for record in &records {
                format_log(&mut buf, record.as_ref(), format);
            }
        }
        TelemetryRecord::Span(record) => format_span(&mut buf, &record, format),
        TelemetryRecord::SpanBatch(records) => {
            for record in &records {
                format_span(&mut buf, record, format);
            }
        }
        TelemetryRecord::SpanBatchArc(records) => {
            for record in &records {
                format_span(&mut buf, record.as_ref(), format);
            }
        }
        TelemetryRecord::Event(record) => format_event(&mut buf, &record, format),
        TelemetryRecord::EventBatch(records) => {
            for record in &records {
                format_event(&mut buf, record, format);
            }
        }
        TelemetryRecord::EventBatchArc(records) => {
            for record in &records {
                format_event(&mut buf, record.as_ref(), format);
            }
        }
        TelemetryRecord::Metric(sample) => format_metric(&mut buf, &sample, format),
        TelemetryRecord::MetricBatch(samples) => {
            for sample in &samples {
                format_metric(&mut buf, sample, format);
            }
        }
        TelemetryRecord::MetricBatchArc(samples) => {
            for sample in &samples {
                format_metric(&mut buf, sample.as_ref(), format);
            }
        }
        TelemetryRecord::Link(link) => format_link(&mut buf, &link, format),
        TelemetryRecord::LinkBatch(links) => {
            for link in &links {
                format_link(&mut buf, link, format);
            }
        }
        TelemetryRecord::LinkBatchArc(links) => {
            for link in &links {
                format_link(&mut buf, link.as_ref(), format);
            }
        }
    }

    // one write per drain batch: the whole request's formatted bytes go out in a
    // single syscall instead of one per record. this amortization is what keeps
    // the drain ahead of emit, so a producer is never conscripted into per-record
    // synchronous writes — the debug-under-load death spiral. lossless: every
    // record is still written, just batched.
    if !buf.is_empty() {
        buf.flush_to(&mut proxima_core::batch::WriteSink(
            &mut *pipe.writer.lock(),
        ))
        .map_err(|err| ProximaError::Body(alloc::format!("formatter write: {err}")))?;
    }

    Ok(ok_response())
}

// tracing-convention uppercase level name, no alloc. Custom levels keep their
// own name.
fn level_name_upper(level: crate::level::Level) -> &'static str {
    match level.name() {
        "trace" => "TRACE",
        "debug" => "DEBUG",
        "info" => "INFO",
        "warn" => "WARN",
        "error" => "ERROR",
        "fatal" => "FATAL",
        other => other,
    }
}

// nanoseconds-since-epoch -> RFC 3339 UTC (`2026-06-30T14:07:02.444554000Z`),
// dep-free: seconds split for the clock, Howard Hinnant's civil-from-days for the
// date. Matches the `ts` a tracing fmt line carries, without pulling chrono/time.
fn format_rfc3339(ts_ns: u64) -> alloc::string::String {
    let secs = ts_ns / 1_000_000_000;
    let nanos = ts_ns % 1_000_000_000;
    let days = (secs / 86_400) as i64;
    let sod = secs % 86_400;
    let (hour, minute, second) = (sod / 3600, (sod % 3600) / 60, sod % 60);
    // civil_from_days: days since 1970-01-01 -> (year, month, day).
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe as i64 + era * 400 + i64::from(month <= 2);
    alloc::format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{nanos:09}Z")
}

fn format_tags(tags: &[Tag]) -> alloc::string::String {
    tags.iter()
        .map(|tag| match tag {
            // `{value}` (Display) renders the bare value — `frame_len=5`, not the
            // `Debug` `frame_len=U64(5)`. Structured values keep Debug (no Display).
            Tag::Scalar { key, value } => alloc::format!("{key}={value}"),
            Tag::Structured { key, value } => alloc::format!("{key}={value:?}"),
        })
        .collect::<alloc::vec::Vec<_>>()
        .join(" ")
}

fn format_log(buf: &mut proxima_core::batch::BatchBuffer, record: &LogRecord, format: LogFormat) {
    let line = match format {
        LogFormat::Human => {
            let body = match &record.body {
                crate::log::LogBody::Empty => alloc::string::String::new(),
                crate::log::LogBody::Text(text) => (*text).to_string(),
                crate::log::LogBody::Owned(bytes) => {
                    alloc::string::String::from_utf8_lossy(bytes.as_ref()).into_owned()
                }
                crate::log::LogBody::Structured(val) => {
                    alloc::format!("{val:?}")
                }
            };
            let tags = format_tags(&record.attrs);
            let ts = format_rfc3339(record.ts_ns);
            let level = level_name_upper(record.level);
            // correlation: the span this log was emitted inside, when present.
            let span = match record.span_id {
                Some(id) => alloc::format!(" span={id}"),
                None => alloc::string::String::new(),
            };
            if tags.is_empty() {
                alloc::format!("{ts} {level} {}:{span} {}\n", record.module_path, body)
            } else {
                alloc::format!("{ts} {level} {}:{span} {} {}\n", record.module_path, body, tags)
            }
        }
        LogFormat::Json => {
            let body = match &record.body {
                crate::log::LogBody::Empty => alloc::string::String::new(),
                crate::log::LogBody::Text(text) => (*text).to_string(),
                crate::log::LogBody::Owned(bytes) => {
                    alloc::string::String::from_utf8_lossy(bytes.as_ref()).into_owned()
                }
                crate::log::LogBody::Structured(val) => {
                    alloc::format!("{val:?}")
                }
            };
            alloc::format!(
                "{{\
                    \"severity\":\"{}\",\
                    \"body\":\"{}\",\
                    \"ts_ns\":{},\
                    \"module_path\":\"{}\"\
                }}\n",
                record.level.name(),
                body.replace('"', "\\\""),
                record.ts_ns,
                record.module_path,
            )
        }
    };
    buf.push_str(&line);
}

fn format_span(buf: &mut proxima_core::batch::BatchBuffer, record: &SpanRecord, format: LogFormat) {
    let line = match format {
        LogFormat::Human => {
            alloc::format!(
                "SPAN {} {}: duration_ns={}\n",
                record.module_path,
                record.name,
                record.duration_ns,
            )
        }
        LogFormat::Json => {
            alloc::format!(
                "{{\
                    \"kind\":\"span\",\
                    \"name\":\"{}\",\
                    \"duration_ns\":{},\
                    \"module_path\":\"{}\"\
                }}\n",
                record.name,
                record.duration_ns,
                record.module_path,
            )
        }
    };
    buf.push_str(&line);
}

fn format_event(
    buf: &mut proxima_core::batch::BatchBuffer,
    record: &EventRecord,
    format: LogFormat,
) {
    let line = match format {
        LogFormat::Human => {
            alloc::format!("EVENT {}: ts_ns={}\n", record.name, record.ts_ns)
        }
        LogFormat::Json => {
            alloc::format!(
                "{{\
                    \"kind\":\"event\",\
                    \"name\":\"{}\",\
                    \"ts_ns\":{}\
                }}\n",
                record.name,
                record.ts_ns,
            )
        }
    };
    buf.push_str(&line);
}

fn format_metric(
    buf: &mut proxima_core::batch::BatchBuffer,
    sample: &MetricSample,
    format: LogFormat,
) {
    let line = match format {
        LogFormat::Human => match sample {
            MetricSample::Counter(point) => {
                alloc::format!("COUNTER value={:?}\n", point.value)
            }
            MetricSample::Gauge(point) => {
                alloc::format!("GAUGE value={:?}\n", point.value)
            }
            MetricSample::UpDownCounter(point) => {
                alloc::format!("UPDOWN value={:?}\n", point.value)
            }
            #[cfg(feature = "histogram")]
            MetricSample::Histogram(point) => {
                alloc::format!("HISTOGRAM count={}\n", point.count)
            }
        },
        LogFormat::Json => match sample {
            MetricSample::Counter(point) => {
                alloc::format!("{{\"kind\":\"counter\",\"value\":\"{:?}\"}}\n", point.value)
            }
            MetricSample::Gauge(point) => {
                alloc::format!("{{\"kind\":\"gauge\",\"value\":\"{:?}\"}}\n", point.value)
            }
            MetricSample::UpDownCounter(point) => {
                alloc::format!("{{\"kind\":\"updown\",\"value\":\"{:?}\"}}\n", point.value)
            }
            #[cfg(feature = "histogram")]
            MetricSample::Histogram(point) => {
                alloc::format!("{{\"kind\":\"histogram\",\"count\":{}}}\n", point.count)
            }
        },
    };
    buf.push_str(&line);
}

fn format_link(buf: &mut proxima_core::batch::BatchBuffer, link: &SpanLink, format: LogFormat) {
    let line = match format {
        LogFormat::Human => {
            alloc::format!("LINK span_id={:?}\n", link.span_id)
        }
        LogFormat::Json => {
            alloc::format!("{{\"kind\":\"link\",\"span_id\":\"{:?}\"}}\n", link.span_id)
        }
    };
    buf.push_str(&line);
}

// helper: get the scalar value of a named attr key from a slice of Tags.
// returns None when the key is absent (record passes; no opinion).
fn find_attr_value<'tag>(attrs: &'tag [Tag], key: &str) -> Option<&'tag ScalarValue> {
    attrs.iter().find_map(|tag| match tag {
        Tag::Scalar {
            key: tag_key,
            value,
        } if *tag_key == key => Some(value),
        _ => None,
    })
}

// filter attrs slice, removing any Tag::Scalar/Structured whose key is in `keys`.
fn strip_attrs(attrs: SmallVec<[Tag; 4]>, keys: &[&'static str]) -> SmallVec<[Tag; 4]> {
    attrs
        .into_iter()
        .filter(|tag| {
            let tag_key = match tag {
                Tag::Scalar { key, .. } => *key,
                Tag::Structured { key, .. } => *key,
            };
            !keys.contains(&tag_key)
        })
        .collect()
}

// SpanRecord, EventRecord, LogRecord don't derive Clone — manual helpers.

fn clone_span(record: &SpanRecord) -> SpanRecord {
    SpanRecord {
        trace_id: record.trace_id,
        span_id: record.span_id,
        parent_span_id: record.parent_span_id,
        name: record.name,
        kind: record.kind,
        start_ns: record.start_ns,
        duration_ns: record.duration_ns,
        status: record.status.clone(),
        attrs: record.attrs.clone(),
        events: record.events.clone(),
        links: record.links.clone(),
        tracestate: record.tracestate.clone(),
        module_path: record.module_path,
        file_line: record.file_line,
    }
}

fn clone_event(record: &EventRecord) -> EventRecord {
    EventRecord {
        parent_span_id: record.parent_span_id,
        name: record.name,
        ts_ns: record.ts_ns,
        attrs: record.attrs.clone(),
        module_path: record.module_path,
        file_line: record.file_line,
    }
}

fn clone_log(record: &LogRecord) -> LogRecord {
    LogRecord {
        ts_ns: record.ts_ns,
        observed_ts_ns: record.observed_ts_ns,
        level: record.level,
        body: record.body.clone(),
        attrs: record.attrs.clone(),
        trace_id: record.trace_id,
        span_id: record.span_id,
        trace_flags: record.trace_flags,
        module_path: record.module_path,
        file_line: record.file_line,
    }
}

fn strip_metric_attrs(sample: MetricSample, keys: &[&'static str]) -> MetricSample {
    match sample {
        MetricSample::Counter(mut point) => {
            point.attrs = strip_attrs(point.attrs, keys);
            MetricSample::Counter(point)
        }
        MetricSample::Gauge(mut point) => {
            point.attrs = strip_attrs(point.attrs, keys);
            MetricSample::Gauge(point)
        }
        MetricSample::UpDownCounter(mut point) => {
            point.attrs = strip_attrs(point.attrs, keys);
            MetricSample::UpDownCounter(point)
        }
        #[cfg(feature = "histogram")]
        MetricSample::Histogram(mut point) => {
            point.attrs = strip_attrs(point.attrs, keys);
            MetricSample::Histogram(point)
        }
    }
}

fn attr_passes(attrs: &[Tag], key: &str, predicate: fn(&ScalarValue) -> bool) -> bool {
    match find_attr_value(attrs, key) {
        Some(value) => !predicate(value),
        None => true,
    }
}

fn metric_attrs(sample: &MetricSample) -> &[Tag] {
    match sample {
        MetricSample::Counter(point)
        | MetricSample::Gauge(point)
        | MetricSample::UpDownCounter(point) => point.attrs.as_slice(),
        #[cfg(feature = "histogram")]
        MetricSample::Histogram(point) => point.attrs.as_slice(),
    }
}

/// Randomly drops a fraction of telemetry records before they reach the inner Pipe.
///
/// `keep_ratio` of 1.0 passes everything; 0.0 drops everything.
/// Handles both per-record and batched shapes for all record types.
///
/// Clone cost: none for dropped records; batches rebuild a filtered Vec (one allocation per
/// filtered batch). Per-record drops pay no clone at all.
pub struct RandomDropPipe<P> {
    inner: P,
    keep_ratio: f64,
}

impl<P> RandomDropPipe<P> {
    #[must_use]
    pub fn new(inner: P, keep_ratio: f64) -> Self {
        Self { inner, keep_ratio }
    }
}

fn random_drop_filter(request: TelemetryRequest, keep_ratio: f64) -> Option<TelemetryRequest> {
    let Request {
        method,
        path,
        query,
        metadata,
        payload,
        stream,
        context,
    } = request;
    let payload = match payload {
        TelemetryRecord::SpanBatch(records) => {
            let kept: alloc::vec::Vec<SpanRecord> = records
                .into_iter()
                .filter(|_| fastrand::f64() < keep_ratio)
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::SpanBatch(kept)
        }
        TelemetryRecord::SpanBatchArc(records) => {
            let kept: alloc::vec::Vec<Arc<SpanRecord>> = records
                .into_iter()
                .filter(|_| fastrand::f64() < keep_ratio)
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::SpanBatchArc(kept)
        }
        TelemetryRecord::EventBatch(records) => {
            let kept: alloc::vec::Vec<EventRecord> = records
                .into_iter()
                .filter(|_| fastrand::f64() < keep_ratio)
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::EventBatch(kept)
        }
        TelemetryRecord::EventBatchArc(records) => {
            let kept: alloc::vec::Vec<Arc<EventRecord>> = records
                .into_iter()
                .filter(|_| fastrand::f64() < keep_ratio)
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::EventBatchArc(kept)
        }
        TelemetryRecord::LogBatch(records) => {
            let kept: alloc::vec::Vec<LogRecord> = records
                .into_iter()
                .filter(|_| fastrand::f64() < keep_ratio)
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::LogBatch(kept)
        }
        TelemetryRecord::LogBatchArc(records) => {
            let kept: alloc::vec::Vec<Arc<LogRecord>> = records
                .into_iter()
                .filter(|_| fastrand::f64() < keep_ratio)
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::LogBatchArc(kept)
        }
        TelemetryRecord::MetricBatch(samples) => {
            let kept: alloc::vec::Vec<MetricSample> = samples
                .into_iter()
                .filter(|_| fastrand::f64() < keep_ratio)
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::MetricBatch(kept)
        }
        TelemetryRecord::MetricBatchArc(samples) => {
            let kept: alloc::vec::Vec<Arc<MetricSample>> = samples
                .into_iter()
                .filter(|_| fastrand::f64() < keep_ratio)
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::MetricBatchArc(kept)
        }
        TelemetryRecord::LinkBatch(links) => {
            let kept: alloc::vec::Vec<SpanLink> = links
                .into_iter()
                .filter(|_| fastrand::f64() < keep_ratio)
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::LinkBatch(kept)
        }
        TelemetryRecord::LinkBatchArc(links) => {
            let kept: alloc::vec::Vec<Arc<SpanLink>> = links
                .into_iter()
                .filter(|_| fastrand::f64() < keep_ratio)
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::LinkBatchArc(kept)
        }
        other => {
            if fastrand::f64() >= keep_ratio {
                return None;
            }
            other
        }
    };
    Some(Request {
        method,
        path,
        query,
        metadata,
        payload,
        stream,
        context,
    })
}

impl<P: SendPipe<In = TelemetryRequest, Out = Response<Bytes>, Err = ProximaError> + Send + Sync>
    SendPipe for RandomDropPipe<P>
{
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: TelemetryRequest,
    ) -> impl StdFuture<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let keep_ratio = self.keep_ratio;
        let filtered = random_drop_filter(request, keep_ratio);
        let inner = &self.inner;
        async move {
            match filtered {
                Some(req) => SendPipe::call(inner, req).await,
                None => Ok(ok_response()),
            }
        }
    }
}

/// Passes only log records at or above `min_level`; non-log requests pass through.
///
/// Handles both `METHOD_LOG` (per-record) and `METHOD_LOG_BATCH` (batched).
/// Other record types (spans, metrics, events, links) are always forwarded unchanged.
pub struct FilterByLevelPipe<P> {
    inner: P,
    min_level: Level,
}

impl<P> FilterByLevelPipe<P> {
    #[must_use]
    pub fn new(inner: P, min_level: Level) -> Self {
        Self { inner, min_level }
    }
}

fn filter_by_level_transform(
    request: TelemetryRequest,
    min_level: Level,
) -> Option<TelemetryRequest> {
    let Request {
        method,
        path,
        query,
        metadata,
        payload,
        stream,
        context,
    } = request;
    let payload = match payload {
        TelemetryRecord::Log(record) => {
            if record.level < min_level {
                return None;
            }
            TelemetryRecord::Log(record)
        }
        TelemetryRecord::LogBatch(records) => {
            let kept: alloc::vec::Vec<LogRecord> = records
                .into_iter()
                .filter(|r| r.level >= min_level)
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::LogBatch(kept)
        }
        TelemetryRecord::LogBatchArc(records) => {
            let kept: alloc::vec::Vec<Arc<LogRecord>> = records
                .into_iter()
                .filter(|arc| arc.level >= min_level)
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::LogBatchArc(kept)
        }
        other => other,
    };
    Some(Request {
        method,
        path,
        query,
        metadata,
        payload,
        stream,
        context,
    })
}

impl<P: SendPipe<In = TelemetryRequest, Out = Response<Bytes>, Err = ProximaError> + Send + Sync>
    SendPipe for FilterByLevelPipe<P>
{
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: TelemetryRequest,
    ) -> impl StdFuture<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let min_level = self.min_level;
        let filtered = filter_by_level_transform(request, min_level);
        let inner = &self.inner;
        async move {
            match filtered {
                Some(req) => SendPipe::call(inner, req).await,
                None => Ok(ok_response()),
            }
        }
    }
}

/// Applies a compiled hierarchical [`crate::emit::CompiledEmit`] filter per
/// signal: logs resolve by `(module_path, level)`; spans/events are gated by
/// their module path at a synthetic `span_band` coordinate (they carry no
/// level); metrics/links pass through (filtered by name elsewhere). The
/// hierarchical, target-aware superset of [`FilterByLevelPipe`].
#[cfg(feature = "emit")]
pub struct EmitFilterPipe<P> {
    inner: P,
    compiled: Arc<crate::emit::CompiledEmit>,
    span_band: crate::emit::Coord,
}

#[cfg(feature = "emit")]
impl<P> EmitFilterPipe<P> {
    /// Wrap `inner` with a compiled filter. `span_band` is the coordinate
    /// spans/events are gated at, since they carry no `Level`.
    #[must_use]
    pub fn new(
        inner: P,
        compiled: Arc<crate::emit::CompiledEmit>,
        span_band: crate::emit::Coord,
    ) -> Self {
        Self {
            inner,
            compiled,
            span_band,
        }
    }
}

#[cfg(feature = "emit")]
fn emit_filter_transform(
    request: TelemetryRequest,
    compiled: &crate::emit::CompiledEmit,
    span_band: crate::emit::Coord,
) -> Option<TelemetryRequest> {
    let Request {
        method,
        path,
        query,
        metadata,
        payload,
        stream,
        context,
    } = request;
    let keep_log = |record: &LogRecord| {
        compiled
            .decide(record.module_path, crate::emit::Coord::from(record.level))
            .is_keep()
    };
    let keep_target = |module_path: &'static str| compiled.decide(module_path, span_band).is_keep();
    let payload = match payload {
        TelemetryRecord::Log(record) => {
            if keep_log(&record) {
                TelemetryRecord::Log(record)
            } else {
                return None;
            }
        }
        TelemetryRecord::LogBatch(records) => {
            let kept: alloc::vec::Vec<LogRecord> = records
                .into_iter()
                .filter(|record| keep_log(record))
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::LogBatch(kept)
        }
        TelemetryRecord::LogBatchArc(records) => {
            let kept: alloc::vec::Vec<Arc<LogRecord>> = records
                .into_iter()
                .filter(|record| keep_log(record))
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::LogBatchArc(kept)
        }
        TelemetryRecord::Span(record) => {
            if keep_target(record.module_path) {
                TelemetryRecord::Span(record)
            } else {
                return None;
            }
        }
        TelemetryRecord::SpanBatch(records) => {
            let kept: alloc::vec::Vec<SpanRecord> = records
                .into_iter()
                .filter(|record| keep_target(record.module_path))
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::SpanBatch(kept)
        }
        TelemetryRecord::SpanBatchArc(records) => {
            let kept: alloc::vec::Vec<Arc<SpanRecord>> = records
                .into_iter()
                .filter(|record| keep_target(record.module_path))
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::SpanBatchArc(kept)
        }
        TelemetryRecord::Event(record) => {
            if keep_target(record.module_path) {
                TelemetryRecord::Event(record)
            } else {
                return None;
            }
        }
        TelemetryRecord::EventBatch(records) => {
            let kept: alloc::vec::Vec<EventRecord> = records
                .into_iter()
                .filter(|record| keep_target(record.module_path))
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::EventBatch(kept)
        }
        TelemetryRecord::EventBatchArc(records) => {
            let kept: alloc::vec::Vec<Arc<EventRecord>> = records
                .into_iter()
                .filter(|record| keep_target(record.module_path))
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::EventBatchArc(kept)
        }
        other => other,
    };
    Some(Request {
        method,
        path,
        query,
        metadata,
        payload,
        stream,
        context,
    })
}

#[cfg(feature = "emit")]
impl<P: SendPipe<In = TelemetryRequest, Out = Response<Bytes>, Err = ProximaError> + Send + Sync>
    SendPipe for EmitFilterPipe<P>
{
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: TelemetryRequest,
    ) -> impl StdFuture<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let filtered = emit_filter_transform(request, &self.compiled, self.span_band);
        let inner = &self.inner;
        async move {
            match filtered {
                Some(req) => SendPipe::call(inner, req).await,
                None => Ok(ok_response()),
            }
        }
    }
}

/// Drops records where a named attr's value matches a predicate.
///
/// Records with no attr named `key` always pass (no opinion).
/// Records where `key` exists and `predicate(&value)` returns `true` are dropped.
/// Handles both per-record and batched shapes for all record types.
pub struct FilterByAttrPipe<P> {
    inner: P,
    key: &'static str,
    predicate: fn(&ScalarValue) -> bool,
}

impl<P> FilterByAttrPipe<P> {
    #[must_use]
    pub fn new(inner: P, key: &'static str, predicate: fn(&ScalarValue) -> bool) -> Self {
        Self {
            inner,
            key,
            predicate,
        }
    }
}

fn filter_by_attr_transform(
    request: TelemetryRequest,
    key: &'static str,
    predicate: fn(&ScalarValue) -> bool,
) -> Option<TelemetryRequest> {
    let Request {
        method,
        path,
        query,
        metadata,
        payload,
        stream,
        context,
    } = request;
    let payload = match payload {
        TelemetryRecord::Span(record) => {
            if !attr_passes(&record.attrs, key, predicate) {
                return None;
            }
            TelemetryRecord::Span(record)
        }
        TelemetryRecord::Event(record) => {
            if !attr_passes(&record.attrs, key, predicate) {
                return None;
            }
            TelemetryRecord::Event(record)
        }
        TelemetryRecord::Log(record) => {
            if !attr_passes(&record.attrs, key, predicate) {
                return None;
            }
            TelemetryRecord::Log(record)
        }
        TelemetryRecord::Metric(sample) => {
            if !attr_passes(metric_attrs(&sample), key, predicate) {
                return None;
            }
            TelemetryRecord::Metric(sample)
        }
        TelemetryRecord::Link(link) => {
            if !attr_passes(&link.attrs, key, predicate) {
                return None;
            }
            TelemetryRecord::Link(link)
        }
        TelemetryRecord::SpanBatch(records) => {
            let kept: alloc::vec::Vec<SpanRecord> = records
                .into_iter()
                .filter(|r| attr_passes(&r.attrs, key, predicate))
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::SpanBatch(kept)
        }
        TelemetryRecord::SpanBatchArc(records) => {
            let kept: alloc::vec::Vec<Arc<SpanRecord>> = records
                .into_iter()
                .filter(|arc| attr_passes(&arc.attrs, key, predicate))
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::SpanBatchArc(kept)
        }
        TelemetryRecord::EventBatch(records) => {
            let kept: alloc::vec::Vec<EventRecord> = records
                .into_iter()
                .filter(|r| attr_passes(&r.attrs, key, predicate))
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::EventBatch(kept)
        }
        TelemetryRecord::EventBatchArc(records) => {
            let kept: alloc::vec::Vec<Arc<EventRecord>> = records
                .into_iter()
                .filter(|arc| attr_passes(&arc.attrs, key, predicate))
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::EventBatchArc(kept)
        }
        TelemetryRecord::LogBatch(records) => {
            let kept: alloc::vec::Vec<LogRecord> = records
                .into_iter()
                .filter(|r| attr_passes(&r.attrs, key, predicate))
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::LogBatch(kept)
        }
        TelemetryRecord::LogBatchArc(records) => {
            let kept: alloc::vec::Vec<Arc<LogRecord>> = records
                .into_iter()
                .filter(|arc| attr_passes(&arc.attrs, key, predicate))
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::LogBatchArc(kept)
        }
        TelemetryRecord::MetricBatch(samples) => {
            let kept: alloc::vec::Vec<MetricSample> = samples
                .into_iter()
                .filter(|s| attr_passes(metric_attrs(s), key, predicate))
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::MetricBatch(kept)
        }
        TelemetryRecord::MetricBatchArc(samples) => {
            let kept: alloc::vec::Vec<Arc<MetricSample>> = samples
                .into_iter()
                .filter(|arc| attr_passes(metric_attrs(arc.as_ref()), key, predicate))
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::MetricBatchArc(kept)
        }
        TelemetryRecord::LinkBatch(links) => {
            let kept: alloc::vec::Vec<SpanLink> = links
                .into_iter()
                .filter(|l| attr_passes(&l.attrs, key, predicate))
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::LinkBatch(kept)
        }
        TelemetryRecord::LinkBatchArc(links) => {
            let kept: alloc::vec::Vec<Arc<SpanLink>> = links
                .into_iter()
                .filter(|arc| attr_passes(&arc.attrs, key, predicate))
                .collect();
            if kept.is_empty() {
                return None;
            }
            TelemetryRecord::LinkBatchArc(kept)
        }
    };
    Some(Request {
        method,
        path,
        query,
        metadata,
        payload,
        stream,
        context,
    })
}

impl<P: SendPipe<In = TelemetryRequest, Out = Response<Bytes>, Err = ProximaError> + Send + Sync>
    SendPipe for FilterByAttrPipe<P>
{
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: TelemetryRequest,
    ) -> impl StdFuture<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let key = self.key;
        let predicate = self.predicate;
        let filtered = filter_by_attr_transform(request, key, predicate);
        let inner = &self.inner;
        async move {
            match filtered {
                Some(req) => SendPipe::call(inner, req).await,
                None => Ok(ok_response()),
            }
        }
    }
}

/// Strips named attrs from every record before forwarding to the inner Pipe.
///
/// Clone cost: each record is cloned with its attrs filtered — unavoidable since the
/// payload is owned and may be shared. Clone cost is one SmallVec iteration + filter + collect per record.
///
/// Handles both per-record and batched body shapes.
pub struct DropAttrPipe<P> {
    inner: P,
    keys: SmallVec<[&'static str; 4]>,
}

impl<P> DropAttrPipe<P> {
    #[must_use]
    pub fn new(inner: P, keys: &[&'static str]) -> Self {
        Self {
            inner,
            keys: SmallVec::from_slice(keys),
        }
    }
}

fn drop_attr_transform(request: TelemetryRequest, keys: &[&'static str]) -> TelemetryRequest {
    let Request {
        method,
        path,
        query,
        metadata,
        payload,
        stream,
        context,
    } = request;
    let payload = match payload {
        TelemetryRecord::Span(mut record) => {
            record.attrs = strip_attrs(record.attrs, keys);
            TelemetryRecord::Span(record)
        }
        TelemetryRecord::Event(mut record) => {
            record.attrs = strip_attrs(record.attrs, keys);
            TelemetryRecord::Event(record)
        }
        TelemetryRecord::Log(mut record) => {
            record.attrs = strip_attrs(record.attrs, keys);
            TelemetryRecord::Log(record)
        }
        TelemetryRecord::Metric(sample) => {
            TelemetryRecord::Metric(strip_metric_attrs(sample, keys))
        }
        TelemetryRecord::Link(mut link) => {
            link.attrs = strip_attrs(link.attrs, keys);
            TelemetryRecord::Link(link)
        }
        TelemetryRecord::SpanBatch(records) => {
            let stripped: alloc::vec::Vec<SpanRecord> = records
                .into_iter()
                .map(|mut r| {
                    r.attrs = strip_attrs(r.attrs, keys);
                    r
                })
                .collect();
            TelemetryRecord::SpanBatch(stripped)
        }
        TelemetryRecord::SpanBatchArc(records) => {
            let stripped: alloc::vec::Vec<Arc<SpanRecord>> = records
                .iter()
                .map(|arc| {
                    let mut cloned = clone_span(arc.as_ref());
                    cloned.attrs = strip_attrs(cloned.attrs, keys);
                    Arc::new(cloned)
                })
                .collect();
            TelemetryRecord::SpanBatchArc(stripped)
        }
        TelemetryRecord::EventBatch(records) => {
            let stripped: alloc::vec::Vec<EventRecord> = records
                .into_iter()
                .map(|mut r| {
                    r.attrs = strip_attrs(r.attrs, keys);
                    r
                })
                .collect();
            TelemetryRecord::EventBatch(stripped)
        }
        TelemetryRecord::EventBatchArc(records) => {
            let stripped: alloc::vec::Vec<Arc<EventRecord>> = records
                .iter()
                .map(|arc| {
                    let mut cloned = clone_event(arc.as_ref());
                    cloned.attrs = strip_attrs(cloned.attrs, keys);
                    Arc::new(cloned)
                })
                .collect();
            TelemetryRecord::EventBatchArc(stripped)
        }
        TelemetryRecord::LogBatch(records) => {
            let stripped: alloc::vec::Vec<LogRecord> = records
                .into_iter()
                .map(|mut r| {
                    r.attrs = strip_attrs(r.attrs, keys);
                    r
                })
                .collect();
            TelemetryRecord::LogBatch(stripped)
        }
        TelemetryRecord::LogBatchArc(records) => {
            let stripped: alloc::vec::Vec<Arc<LogRecord>> = records
                .iter()
                .map(|arc| {
                    let mut cloned = clone_log(arc.as_ref());
                    cloned.attrs = strip_attrs(cloned.attrs, keys);
                    Arc::new(cloned)
                })
                .collect();
            TelemetryRecord::LogBatchArc(stripped)
        }
        TelemetryRecord::MetricBatch(samples) => {
            let stripped: alloc::vec::Vec<MetricSample> = samples
                .into_iter()
                .map(|s| strip_metric_attrs(s, keys))
                .collect();
            TelemetryRecord::MetricBatch(stripped)
        }
        TelemetryRecord::MetricBatchArc(samples) => {
            let stripped: alloc::vec::Vec<Arc<MetricSample>> = samples
                .iter()
                .map(|arc| Arc::new(strip_metric_attrs((**arc).clone(), keys)))
                .collect();
            TelemetryRecord::MetricBatchArc(stripped)
        }
        TelemetryRecord::LinkBatch(links) => {
            let stripped: alloc::vec::Vec<SpanLink> = links
                .into_iter()
                .map(|mut l| {
                    l.attrs = strip_attrs(l.attrs, keys);
                    l
                })
                .collect();
            TelemetryRecord::LinkBatch(stripped)
        }
        TelemetryRecord::LinkBatchArc(links) => {
            let stripped: alloc::vec::Vec<Arc<SpanLink>> = links
                .iter()
                .map(|arc| {
                    let mut cloned = (**arc).clone();
                    cloned.attrs = strip_attrs(cloned.attrs, keys);
                    Arc::new(cloned)
                })
                .collect();
            TelemetryRecord::LinkBatchArc(stripped)
        }
    };
    Request {
        method,
        path,
        query,
        metadata,
        payload,
        stream,
        context,
    }
}

impl<P: SendPipe<In = TelemetryRequest, Out = Response<Bytes>, Err = ProximaError> + Send + Sync>
    SendPipe for DropAttrPipe<P>
{
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: TelemetryRequest,
    ) -> impl StdFuture<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let keys_owned: alloc::vec::Vec<&'static str> = self.keys.to_vec();
        let transformed = drop_attr_transform(request, &keys_owned);
        let inner = &self.inner;
        async move { SendPipe::call(inner, transformed).await }
    }
}

/// Renames a metric by name in-flight, rewriting matching `MetricSample` records.
///
/// **Limitation (v1):** MetricSample is registry-keyed; the canonical name lives in
/// the registry, not in the sample itself. Callers who want rename to work must
/// include a `"metric.name"` tag on the sample's attrs at emission time.
pub struct RenameMetricPipe<P> {
    inner: P,
    from: &'static str,
    to: &'static str,
}

impl<P> RenameMetricPipe<P> {
    #[must_use]
    pub fn new(inner: P, from: &'static str, to: &'static str) -> Self {
        Self { inner, from, to }
    }
}

fn rename_in_sample(sample: MetricSample, from: &'static str, to: &'static str) -> MetricSample {
    let rewrite = |mut attrs: SmallVec<[Tag; 4]>| -> SmallVec<[Tag; 4]> {
        for tag in &mut attrs {
            if let Tag::Scalar {
                key: "metric.name",
                value: ScalarValue::Str(name),
            } = tag
                && *name == from
            {
                *name = to;
            }
        }
        attrs
    };
    match sample {
        MetricSample::Counter(mut point) => {
            point.attrs = rewrite(point.attrs);
            MetricSample::Counter(point)
        }
        MetricSample::Gauge(mut point) => {
            point.attrs = rewrite(point.attrs);
            MetricSample::Gauge(point)
        }
        MetricSample::UpDownCounter(mut point) => {
            point.attrs = rewrite(point.attrs);
            MetricSample::UpDownCounter(point)
        }
        #[cfg(feature = "histogram")]
        MetricSample::Histogram(mut point) => {
            point.attrs = rewrite(point.attrs);
            MetricSample::Histogram(point)
        }
    }
}

fn rename_metric_transform(
    request: TelemetryRequest,
    from: &'static str,
    to: &'static str,
) -> TelemetryRequest {
    let Request {
        method,
        path,
        query,
        metadata,
        payload,
        stream,
        context,
    } = request;
    let payload = match payload {
        TelemetryRecord::Metric(sample) => {
            TelemetryRecord::Metric(rename_in_sample(sample, from, to))
        }
        TelemetryRecord::MetricBatch(samples) => {
            let renamed: alloc::vec::Vec<MetricSample> = samples
                .into_iter()
                .map(|s| rename_in_sample(s, from, to))
                .collect();
            TelemetryRecord::MetricBatch(renamed)
        }
        TelemetryRecord::MetricBatchArc(samples) => {
            let renamed: alloc::vec::Vec<Arc<MetricSample>> = samples
                .iter()
                .map(|arc| Arc::new(rename_in_sample((**arc).clone(), from, to)))
                .collect();
            TelemetryRecord::MetricBatchArc(renamed)
        }
        other => other,
    };
    Request {
        method,
        path,
        query,
        metadata,
        payload,
        stream,
        context,
    }
}

impl<P: SendPipe<In = TelemetryRequest, Out = Response<Bytes>, Err = ProximaError> + Send + Sync>
    SendPipe for RenameMetricPipe<P>
{
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: TelemetryRequest,
    ) -> impl StdFuture<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let from = self.from;
        let to = self.to;
        let transformed = rename_metric_transform(request, from, to);
        let inner = &self.inner;
        async move { SendPipe::call(inner, transformed).await }
    }
}

/// Re-buckets histogram data points onto new bucket bounds in-flight.
///
/// Applies only to `METHOD_HIST_RECORD` / `METHOD_METRIC_BATCH` bodies carrying
/// `MetricSample::Histogram`. Other record types and metric variants pass through.
///
/// **Non-histogram features:** when the `histogram` feature is off, this pipe is a
/// no-op transparent pass-through.
pub struct RebucketHistogramPipe<P> {
    inner: P,
    new_bounds: &'static [f64],
}

impl<P> RebucketHistogramPipe<P> {
    #[must_use]
    pub fn new(inner: P, new_bounds: &'static [f64]) -> Self {
        Self { inner, new_bounds }
    }
}

#[cfg(feature = "histogram")]
fn rebucket(sample: MetricSample, new_bounds: &'static [f64]) -> MetricSample {
    use crate::metric::sample::HistogramDataPoint;
    let MetricSample::Histogram(point) = sample else {
        return sample;
    };
    let mut new_counts = alloc::vec![0u64; new_bounds.len() + 1];
    for (source_index, &count) in point.bucket_counts.iter().enumerate() {
        let source_bound = point
            .bounds
            .get(source_index)
            .copied()
            .unwrap_or(f64::INFINITY);
        let target_index = new_bounds
            .iter()
            .position(|&bound| source_bound <= bound)
            .unwrap_or(new_bounds.len());
        new_counts[target_index] += count;
    }
    MetricSample::Histogram(HistogramDataPoint {
        count: point.count,
        sum: point.sum,
        bucket_counts: new_counts,
        bounds: new_bounds,
        attrs: point.attrs,
        ts_ns: point.ts_ns,
        start_ts_ns: point.start_ts_ns,
    })
}

fn rebucket_transform(request: TelemetryRequest, new_bounds: &'static [f64]) -> TelemetryRequest {
    let Request {
        method,
        path,
        query,
        metadata,
        payload,
        stream,
        context,
    } = request;
    let payload = match payload {
        #[cfg(feature = "histogram")]
        TelemetryRecord::Metric(sample) => TelemetryRecord::Metric(rebucket(sample, new_bounds)),
        #[cfg(feature = "histogram")]
        TelemetryRecord::MetricBatch(samples) => {
            let rebucketed: alloc::vec::Vec<MetricSample> = samples
                .into_iter()
                .map(|s| rebucket(s, new_bounds))
                .collect();
            TelemetryRecord::MetricBatch(rebucketed)
        }
        #[cfg(feature = "histogram")]
        TelemetryRecord::MetricBatchArc(samples) => {
            let rebucketed: alloc::vec::Vec<Arc<MetricSample>> = samples
                .iter()
                .map(|arc| Arc::new(rebucket((**arc).clone(), new_bounds)))
                .collect();
            TelemetryRecord::MetricBatchArc(rebucketed)
        }
        other => {
            let _ = new_bounds;
            other
        }
    };
    Request {
        method,
        path,
        query,
        metadata,
        payload,
        stream,
        context,
    }
}

impl<P: SendPipe<In = TelemetryRequest, Out = Response<Bytes>, Err = ProximaError> + Send + Sync>
    SendPipe for RebucketHistogramPipe<P>
{
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: TelemetryRequest,
    ) -> impl StdFuture<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let new_bounds = self.new_bounds;
        let transformed = rebucket_transform(request, new_bounds);
        let inner = &self.inner;
        async move { SendPipe::call(inner, transformed).await }
    }
}

/// Windowed counter aggregation Pipe.
///
/// Accumulates `MetricSample::Counter` records over a configurable time window, grouped by
/// a configured set of attribute keys. At window close, emits one aggregate per unique
/// attribute group to the inner Pipe. All other record types pass through unchanged.
pub struct SumByPipe<P> {
    inner: P,
    window: std::time::Duration,
    group_by: SmallVec<[&'static str; 4]>,
    state: parking_lot::Mutex<SumState>,
}

struct SumState {
    window_start: std::time::Instant,
    accumulator: rustc_hash::FxHashMap<GroupKey, AccumulatedSum>,
}

#[derive(Hash, PartialEq, Eq, Clone)]
struct GroupKey {
    attrs: SmallVec<[(&'static str, alloc::vec::Vec<u8>); 4]>,
}

#[derive(Clone)]
struct AccumulatedSum {
    sum_u64: u64,
    sum_f64: f64,
    count: u64,
    last_ts_ns: u64,
    start_ts_ns: u64,
    sample_attrs: SmallVec<[Tag; 4]>,
}

impl<P> SumByPipe<P> {
    #[must_use]
    pub fn new(
        inner: P,
        window: std::time::Duration,
        group_by: impl IntoIterator<Item = &'static str>,
    ) -> Self {
        let mut keys: SmallVec<[&'static str; 4]> = group_by.into_iter().collect();
        keys.sort_unstable();
        Self {
            inner,
            window,
            group_by: keys,
            state: parking_lot::Mutex::new(SumState {
                window_start: std::time::Instant::now(),
                accumulator: rustc_hash::FxHashMap::default(),
            }),
        }
    }
}

fn scalar_to_canonical_bytes(value: &ScalarValue) -> alloc::vec::Vec<u8> {
    alloc::format!("{value:?}").into_bytes()
}

fn build_group_key(attrs: &[Tag], group_by: &[&'static str]) -> GroupKey {
    let mut pairs: SmallVec<[(&'static str, alloc::vec::Vec<u8>); 4]> = group_by
        .iter()
        .map(|key| {
            let bytes = attrs
                .iter()
                .find_map(|tag| match tag {
                    Tag::Scalar {
                        key: tag_key,
                        value,
                    } if tag_key == key => Some(scalar_to_canonical_bytes(value)),
                    _ => None,
                })
                .unwrap_or_default();
            (*key, bytes)
        })
        .collect();
    pairs.sort_unstable_by_key(|(key, _)| *key);
    GroupKey { attrs: pairs }
}

fn accumulate_counter(
    acc: &mut rustc_hash::FxHashMap<GroupKey, AccumulatedSum>,
    point: &crate::metric::sample::NumberDataPoint,
    group_by: &[&'static str],
) {
    let key = build_group_key(&point.attrs, group_by);
    let entry = acc.entry(key).or_insert_with(|| AccumulatedSum {
        sum_u64: 0,
        sum_f64: 0.0,
        count: 0,
        last_ts_ns: point.ts_ns,
        start_ts_ns: point.start_ts_ns,
        sample_attrs: point.attrs.clone(),
    });
    match &point.value {
        ScalarValue::U64(value) => entry.sum_u64 = entry.sum_u64.saturating_add(*value),
        ScalarValue::F64(value) => entry.sum_f64 += value,
        ScalarValue::I64(value) if *value >= 0 => {
            entry.sum_u64 = entry.sum_u64.saturating_add(*value as u64);
        }
        ScalarValue::I64(_) => {}
        _ => {}
    }
    entry.count += 1;
    entry.last_ts_ns = entry.last_ts_ns.max(point.ts_ns);
}

fn drain_to_aggregates(
    state: &mut SumState,
    value_hint: &ScalarValue,
) -> alloc::vec::Vec<MetricSample> {
    use crate::metric::sample::NumberDataPoint;
    std::mem::take(&mut state.accumulator)
        .into_values()
        .map(|entry| {
            let value = match value_hint {
                ScalarValue::F64(_) => ScalarValue::F64(entry.sum_f64),
                _ => ScalarValue::U64(entry.sum_u64),
            };
            MetricSample::Counter(NumberDataPoint {
                value,
                attrs: entry.sample_attrs,
                ts_ns: entry.last_ts_ns,
                start_ts_ns: entry.start_ts_ns,
            })
        })
        .collect()
}

impl<P: SendPipe<In = TelemetryRequest, Out = Response<Bytes>, Err = ProximaError> + Send + Sync>
    SumByPipe<P>
{
    /// Force-flush any accumulated data to the inner Pipe regardless of window state.
    pub async fn flush(&self) -> Result<Response<Bytes>, ProximaError> {
        let aggregates = {
            let mut state = self.state.lock();
            state.window_start = std::time::Instant::now();
            std::mem::take(&mut state.accumulator)
                .into_values()
                .map(|entry| {
                    use crate::metric::sample::NumberDataPoint;
                    MetricSample::Counter(NumberDataPoint {
                        value: ScalarValue::U64(entry.sum_u64),
                        attrs: entry.sample_attrs,
                        ts_ns: entry.last_ts_ns,
                        start_ts_ns: entry.start_ts_ns,
                    })
                })
                .collect::<alloc::vec::Vec<_>>()
        };
        if aggregates.is_empty() {
            return Ok(ok_response());
        }
        SendPipe::call(&self.inner, metric_batch_request(aggregates)).await
    }
}

fn sum_by_dispatch<
    P: SendPipe<In = TelemetryRequest, Out = Response<Bytes>, Err = ProximaError>,
>(
    pipe: &SumByPipe<P>,
    request: TelemetryRequest,
) -> (
    Option<alloc::vec::Vec<MetricSample>>,
    Option<TelemetryRequest>,
) {
    use crate::metric::sample::NumberDataPoint;

    let group_by: alloc::vec::Vec<&'static str> = pipe.group_by.to_vec();
    let window = pipe.window;

    let points: Option<alloc::vec::Vec<NumberDataPoint>> = match &request.payload {
        TelemetryRecord::Metric(MetricSample::Counter(point)) => Some(alloc::vec![point.clone()]),
        TelemetryRecord::MetricBatch(samples) => Some(
            samples
                .iter()
                .filter_map(|s| {
                    if let MetricSample::Counter(point) = s {
                        Some(point.clone())
                    } else {
                        None
                    }
                })
                .collect(),
        ),
        _ => None,
    };

    let Some(points) = points else {
        return (None, Some(request));
    };

    let flush_batch: Option<alloc::vec::Vec<MetricSample>> = {
        let mut state = pipe.state.lock();
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(state.window_start);

        let flushed = if elapsed >= window && !state.accumulator.is_empty() {
            let value_hint = points
                .first()
                .map(|point| point.value.clone())
                .unwrap_or(ScalarValue::U64(0));
            let batch = drain_to_aggregates(&mut state, &value_hint);
            state.window_start = now;
            Some(batch)
        } else {
            if elapsed >= window {
                state.window_start = now;
            }
            None
        };

        for point in &points {
            accumulate_counter(&mut state.accumulator, point, &group_by);
        }

        flushed
    };

    (flush_batch, None)
}

impl<P: SendPipe<In = TelemetryRequest, Out = Response<Bytes>, Err = ProximaError> + Send + Sync>
    SendPipe for SumByPipe<P>
{
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: TelemetryRequest,
    ) -> impl StdFuture<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let (flush_batch, pass_through) = sum_by_dispatch(self, request);
        let inner = &self.inner;
        async move {
            if let Some(req) = pass_through {
                return SendPipe::call(inner, req).await;
            }
            if let Some(batch) = flush_batch
                && !batch.is_empty()
            {
                SendPipe::call(inner, metric_batch_request(batch)).await?;
            }
            Ok(ok_response())
        }
    }
}

/// Composability sugar: chain filter + view Pipes onto any inner Pipe via fluent methods.
pub trait TelemetryPipeExt:
    SendPipe<In = TelemetryRequest, Out = Response<Bytes>, Err = ProximaError> + Sized
{
    /// Wrap with `RandomDropPipe`: keep only `keep_ratio` fraction of records (0.0 = drop all).
    fn filter_random_drop(self, keep_ratio: f64) -> RandomDropPipe<Self> {
        RandomDropPipe::new(self, keep_ratio)
    }

    /// Wrap with `FilterByLevelPipe`: pass only log records at or above `min_level`.
    fn filter_by_level(self, min_level: Level) -> FilterByLevelPipe<Self> {
        FilterByLevelPipe::new(self, min_level)
    }

    /// Wrap with `EmitFilterPipe`: apply a compiled hierarchical emit filter —
    /// logs by `(target, level)`, spans/events by target subtree. The
    /// hierarchical, target-aware superset of [`Self::filter_by_level`].
    #[cfg(feature = "emit")]
    fn emit_filter(
        self,
        compiled: Arc<crate::emit::CompiledEmit>,
        span_band: crate::emit::Coord,
    ) -> EmitFilterPipe<Self> {
        EmitFilterPipe::new(self, compiled, span_band)
    }

    /// Wrap with `FilterByAttrPipe`: drop records where attr `key` satisfies `predicate`.
    fn filter_by_attr(
        self,
        key: &'static str,
        predicate: fn(&ScalarValue) -> bool,
    ) -> FilterByAttrPipe<Self> {
        FilterByAttrPipe::new(self, key, predicate)
    }

    /// Wrap with `DropAttrPipe`: strip named attrs from every record.
    fn drop_attrs(self, keys: &[&'static str]) -> DropAttrPipe<Self> {
        DropAttrPipe::new(self, keys)
    }

    /// Wrap with `RenameMetricPipe`: rename metric samples carrying `"metric.name" = from`.
    fn rename_metric(self, from: &'static str, to: &'static str) -> RenameMetricPipe<Self> {
        RenameMetricPipe::new(self, from, to)
    }

    /// Wrap with `RebucketHistogramPipe`: re-bucket histogram observations onto `new_bounds`.
    fn rebucket_histogram(self, new_bounds: &'static [f64]) -> RebucketHistogramPipe<Self> {
        RebucketHistogramPipe::new(self, new_bounds)
    }

    /// Wrap with `SumByPipe`: aggregate counter records over a time window grouped by attrs.
    fn sum_by(
        self,
        window: std::time::Duration,
        group_by: impl IntoIterator<Item = &'static str>,
    ) -> SumByPipe<Self> {
        SumByPipe::new(self, window, group_by)
    }
}

impl<P: SendPipe<In = TelemetryRequest, Out = Response<Bytes>, Err = ProximaError>> TelemetryPipeExt
    for P
{
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod filter_view_tests {
    use core::sync::atomic::Ordering;

    use futures::executor::block_on;

    use super::*;
    use crate::id::{SpanId, TraceFlags, TraceId};
    use crate::level::Level;
    use crate::log::LogBody;
    use crate::metric::sample::NumberDataPoint;
    use crate::tag::{ScalarValue, Tag};
    use crate::trace::kind::SpanKind;
    use crate::trace::status::Status;
    use crate::trace::tracestate::TraceState;
    use proxima_primitives::pipe::SendPipe;

    fn make_log(level: Level) -> LogRecord {
        LogRecord {
            ts_ns: 0,
            observed_ts_ns: 0,
            level,
            body: LogBody::Text("test"),
            attrs: smallvec::SmallVec::new(),
            trace_id: None,
            span_id: None,
            trace_flags: TraceFlags(0),
            module_path: "",
            file_line: (0, 0),
        }
    }

    fn make_log_with_attr(level: Level, key: &'static str, val: ScalarValue) -> LogRecord {
        let mut record = make_log(level);
        record.attrs.push(Tag::Scalar { key, value: val });
        record
    }

    fn make_span() -> SpanRecord {
        SpanRecord {
            trace_id: TraceId::INVALID,
            span_id: SpanId::INVALID,
            parent_span_id: None,
            name: "test",
            kind: SpanKind::Internal,
            start_ns: 0,
            duration_ns: 0,
            status: Status::Unset,
            attrs: smallvec::SmallVec::new(),
            events: smallvec::SmallVec::new(),
            links: smallvec::SmallVec::new(),
            tracestate: TraceState::empty(),
            module_path: "",
            file_line: (0, 0),
        }
    }

    fn make_metric_counter(key: &'static str, val: ScalarValue) -> MetricSample {
        let mut point = NumberDataPoint {
            value: ScalarValue::U64(1),
            attrs: smallvec::SmallVec::new(),
            ts_ns: 0,
            start_ts_ns: 0,
        };
        point.attrs.push(Tag::Scalar { key, value: val });
        MetricSample::Counter(point)
    }

    // A `std::io::Write` that records every `write` call so a test can prove the
    // batch left the formatter in a single syscall (Clone shares the inner Arc, so
    // the handle kept for inspection sees what the pipe's copy wrote).
    #[derive(Clone)]
    struct CountingWriter {
        inner: std::sync::Arc<std::sync::Mutex<(usize, alloc::vec::Vec<u8>)>>,
    }

    impl CountingWriter {
        fn new() -> Self {
            Self {
                inner: std::sync::Arc::new(std::sync::Mutex::new((0, alloc::vec::Vec::new()))),
            }
        }
        fn writes(&self) -> usize {
            self.inner.lock().unwrap().0
        }
        fn bytes(&self) -> alloc::vec::Vec<u8> {
            self.inner.lock().unwrap().1.clone()
        }
    }

    impl std::io::Write for CountingWriter {
        fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
            let mut guard = self.inner.lock().unwrap();
            guard.0 += 1;
            guard.1.extend_from_slice(data);
            Ok(data.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    // The batched-flush contract: a five-record drain batch leaves the formatter
    // in ONE write, not five. This is the syscall amortization that keeps the drain
    // ahead of emit so a producer is never conscripted into per-record synchronous
    // writes (the debug-under-load death spiral). Lossless: all five lines present.
    #[test]
    fn log_batch_flushes_in_a_single_write() {
        let writer = CountingWriter::new();
        let pipe = FormatterPipe::new(writer.clone(), LogFormat::Human);
        let records = vec![
            make_log(Level::INFO),
            make_log(Level::WARN),
            make_log(Level::ERROR),
            make_log(Level::INFO),
            make_log(Level::DEBUG),
        ];
        let request = log_batch_request(records);

        block_on(SendPipe::call(&pipe, request)).expect("call ok");

        assert_eq!(
            writer.writes(),
            1,
            "five records must flush in a single write"
        );
        let out = writer.bytes();
        assert_eq!(
            out.iter().filter(|&&byte| byte == b'\n').count(),
            5,
            "all five lines present, none lost to batching",
        );
    }

    // The console-path contract: the severity-routed sink batches each half of a
    // mixed batch into ONE write per writer, and routes by severity. This is the
    // path `install_console_logging` uses — the one the debug-under-load fix is
    // for. INFO/DEBUG → stdout, WARN/ERROR → stderr.
    #[test]
    fn console_split_batches_per_writer_and_routes_by_severity() {
        let out_writer = CountingWriter::new();
        let err_writer = CountingWriter::new();
        let pipe =
            StdSplitPipe::with_writers(out_writer.clone(), err_writer.clone(), LogFormat::Human);
        let records = vec![
            make_log(Level::INFO),
            make_log(Level::WARN),
            make_log(Level::DEBUG),
            make_log(Level::ERROR),
        ];
        let request = log_batch_request(records);

        block_on(SendPipe::call(&pipe, request)).expect("call ok");

        assert_eq!(out_writer.writes(), 1, "stdout half flushes in one write");
        assert_eq!(err_writer.writes(), 1, "stderr half flushes in one write");
        let out_lines = out_writer
            .bytes()
            .iter()
            .filter(|&&byte| byte == b'\n')
            .count();
        let err_lines = err_writer
            .bytes()
            .iter()
            .filter(|&&byte| byte == b'\n')
            .count();
        assert_eq!(out_lines, 2, "INFO + DEBUG routed to stdout");
        assert_eq!(err_lines, 2, "WARN + ERROR routed to stderr");
    }

    // RandomDropPipe tests

    #[test]
    fn random_drop_keep_all_passes_every_record() {
        let (inner, _, _, logs, _, _) = CountingPipe::new();
        let pipe = RandomDropPipe::new(inner, 1.0);
        let request = log_request(make_log(Level::INFO));
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(logs.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn random_drop_keep_none_drops_every_record() {
        let (inner, _, _, logs, _, _) = CountingPipe::new();
        let pipe = RandomDropPipe::new(inner, 0.0);
        let request = log_request(make_log(Level::INFO));
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(logs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn random_drop_keep_all_batch_passes_all() {
        let (inner, _, _, logs, _, _) = CountingPipe::new();
        let pipe = RandomDropPipe::new(inner, 1.0);
        let records = vec![make_log(Level::INFO), make_log(Level::WARN)];
        let request = log_batch_request(records);
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(logs.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn random_drop_keep_none_batch_short_circuits() {
        let (inner, _, _, logs, _, _) = CountingPipe::new();
        let pipe = RandomDropPipe::new(inner, 0.0);
        let records = vec![make_log(Level::INFO), make_log(Level::WARN)];
        let request = log_batch_request(records);
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(logs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn random_drop_composition_both_apply() {
        let (inner, _, _, logs, _, _) = CountingPipe::new();
        let pipe = RandomDropPipe::new(RandomDropPipe::new(inner, 1.0), 1.0);
        let request = log_request(make_log(Level::INFO));
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(logs.load(Ordering::Relaxed), 1);
    }

    // FilterByLevelPipe tests

    #[test]
    fn filter_by_level_passes_record_at_or_above_threshold() {
        let (inner, _, _, logs, _, _) = CountingPipe::new();
        let pipe = FilterByLevelPipe::new(inner, Level::WARN);
        let request = log_request(make_log(Level::ERROR));
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(logs.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn filter_by_level_drops_record_below_threshold() {
        let (inner, _, _, logs, _, _) = CountingPipe::new();
        let pipe = FilterByLevelPipe::new(inner, Level::WARN);
        let request = log_request(make_log(Level::DEBUG));
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(logs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn filter_by_level_passes_non_log_requests_unchanged() {
        let (inner, spans, _, _, _, _) = CountingPipe::new();
        let pipe = FilterByLevelPipe::new(inner, Level::ERROR);
        let request = span_request(make_span());
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(spans.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn filter_by_level_batch_keeps_only_at_or_above_threshold() {
        let (inner, _, _, logs, _, _) = CountingPipe::new();
        let pipe = FilterByLevelPipe::new(inner, Level::WARN);
        let records = vec![
            make_log(Level::DEBUG),
            make_log(Level::WARN),
            make_log(Level::ERROR),
        ];
        let request = log_batch_request(records);
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(logs.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn filter_by_level_composition_with_random_drop() {
        let (inner, _, _, logs, _, _) = CountingPipe::new();
        let pipe = RandomDropPipe::new(FilterByLevelPipe::new(inner, Level::WARN), 1.0);
        let request = log_request(make_log(Level::ERROR));
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(logs.load(Ordering::Relaxed), 1);
    }

    // FilterByAttrPipe tests

    #[test]
    fn filter_by_attr_drops_record_when_predicate_matches() {
        let (inner, _, _, logs, _, _) = CountingPipe::new();
        let pipe =
            FilterByAttrPipe::new(inner, "env", |val| matches!(val, ScalarValue::Str("dev")));
        let record = make_log_with_attr(Level::INFO, "env", ScalarValue::Str("dev"));
        let request = log_request(record);
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(logs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn filter_by_attr_passes_record_when_predicate_does_not_match() {
        let (inner, _, _, logs, _, _) = CountingPipe::new();
        let pipe =
            FilterByAttrPipe::new(inner, "env", |val| matches!(val, ScalarValue::Str("dev")));
        let record = make_log_with_attr(Level::INFO, "env", ScalarValue::Str("prod"));
        let request = log_request(record);
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(logs.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn filter_by_attr_passes_record_when_key_absent() {
        let (inner, _, _, logs, _, _) = CountingPipe::new();
        let pipe =
            FilterByAttrPipe::new(inner, "env", |val| matches!(val, ScalarValue::Str("dev")));
        let request = log_request(make_log(Level::INFO));
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(logs.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn filter_by_attr_batch_removes_matching_records() {
        let (inner, _, _, logs, _, _) = CountingPipe::new();
        let pipe =
            FilterByAttrPipe::new(inner, "env", |val| matches!(val, ScalarValue::Str("dev")));
        let records = vec![
            make_log_with_attr(Level::INFO, "env", ScalarValue::Str("dev")),
            make_log_with_attr(Level::INFO, "env", ScalarValue::Str("prod")),
            make_log(Level::INFO),
        ];
        let request = log_batch_request(records);
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(logs.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn filter_by_attr_composition_with_level_filter() {
        let (inner, _, _, logs, _, _) = CountingPipe::new();
        let pipe =
            FilterByAttrPipe::new(FilterByLevelPipe::new(inner, Level::WARN), "env", |val| {
                matches!(val, ScalarValue::Str("dev"))
            });
        let request = log_request(make_log_with_attr(
            Level::ERROR,
            "env",
            ScalarValue::Str("prod"),
        ));
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(logs.load(Ordering::Relaxed), 1);
    }

    // DropAttrPipe tests

    #[test]
    fn drop_attr_removes_named_attr_from_log_record() {
        let (inner, _, _, logs, _, _) = CountingPipe::new();
        let pipe = DropAttrPipe::new(inner, &["user_id"]);
        let record = make_log_with_attr(Level::INFO, "user_id", ScalarValue::Str("abc"));
        let request = log_request(record);
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(logs.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn drop_attr_passes_through_when_key_absent() {
        let (inner, _, _, logs, _, _) = CountingPipe::new();
        let pipe = DropAttrPipe::new(inner, &["user_id"]);
        let request = log_request(make_log(Level::INFO));
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(logs.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn drop_attr_passes_non_log_non_typed_requests() {
        let (inner, spans, _, _, _, _) = CountingPipe::new();
        let mut span = make_span();
        span.attrs.push(Tag::Scalar {
            key: "user_id",
            value: ScalarValue::Str("x"),
        });
        let pipe = DropAttrPipe::new(inner, &["user_id"]);
        let request = span_request(span);
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(spans.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn drop_attr_batch_strips_attr_from_all_records() {
        let (inner, _, _, logs, _, _) = CountingPipe::new();
        let pipe = DropAttrPipe::new(inner, &["user_id"]);
        let records = vec![
            make_log_with_attr(Level::INFO, "user_id", ScalarValue::Str("a")),
            make_log_with_attr(Level::INFO, "user_id", ScalarValue::Str("b")),
        ];
        let request = log_batch_request(records);
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(logs.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn drop_attr_composition_with_level_filter() {
        let (inner, _, _, logs, _, _) = CountingPipe::new();
        let pipe = DropAttrPipe::new(FilterByLevelPipe::new(inner, Level::INFO), &["user_id"]);
        let record = make_log_with_attr(Level::INFO, "user_id", ScalarValue::Str("abc"));
        let request = log_request(record);
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(logs.load(Ordering::Relaxed), 1);
    }

    // RenameMetricPipe tests

    #[test]
    fn rename_metric_rewrites_metric_name_attr() {
        let (inner, _, _, _, metrics, _) = CountingPipe::new();
        let pipe = RenameMetricPipe::new(inner, "http.requests", "requests.count");
        let sample = make_metric_counter("metric.name", ScalarValue::Str("http.requests"));
        let request = metric_request(sample);
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(metrics.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn rename_metric_passes_through_non_matching() {
        let (inner, _, _, _, metrics, _) = CountingPipe::new();
        let pipe = RenameMetricPipe::new(inner, "http.requests", "requests.count");
        let sample = make_metric_counter("metric.name", ScalarValue::Str("other.metric"));
        let request = metric_request(sample);
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(metrics.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn rename_metric_passes_non_metric_requests() {
        let (inner, _, _, logs, _, _) = CountingPipe::new();
        let pipe = RenameMetricPipe::new(inner, "http.requests", "requests.count");
        let request = log_request(make_log(Level::INFO));
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(logs.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn rename_metric_composition_with_drop_attr() {
        let (inner, _, _, _, metrics, _) = CountingPipe::new();
        let pipe = RenameMetricPipe::new(
            DropAttrPipe::new(inner, &["debug.tag"]),
            "old.name",
            "new.name",
        );
        let sample = make_metric_counter("metric.name", ScalarValue::Str("old.name"));
        let request = metric_request(sample);
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(metrics.load(Ordering::Relaxed), 1);
    }

    // RebucketHistogramPipe tests

    #[cfg(feature = "histogram")]
    mod histogram_tests {
        use super::*;
        use crate::metric::sample::HistogramDataPoint;

        fn make_histogram_request(bounds: &'static [f64], counts: Vec<u64>) -> TelemetryRequest {
            let sample = MetricSample::Histogram(HistogramDataPoint {
                count: counts.iter().sum(),
                sum: 10.0,
                bucket_counts: counts,
                bounds,
                attrs: smallvec::SmallVec::new(),
                ts_ns: 0,
                start_ts_ns: 0,
            });
            make_request(
                METHOD_HIST_RECORD,
                PATH_METRIC_HISTOGRAM,
                TelemetryRecord::Metric(sample),
            )
        }

        static OLD_BOUNDS: &[f64] = &[0.001, 0.01, 0.1, 1.0];
        static NEW_BOUNDS: &[f64] = &[0.01, 1.0];

        #[test]
        fn rebucket_histogram_remaps_buckets_and_forwards() {
            let (inner, _, _, _, metrics, _) = CountingPipe::new();
            let pipe = RebucketHistogramPipe::new(inner, NEW_BOUNDS);
            let request = make_histogram_request(OLD_BOUNDS, vec![1, 2, 3, 4, 5]);
            block_on(SendPipe::call(&pipe, request)).expect("call ok");
            assert_eq!(metrics.load(Ordering::Relaxed), 1);
        }

        #[test]
        fn rebucket_histogram_non_histogram_passes_through() {
            let (inner, _, _, _, metrics, _) = CountingPipe::new();
            let pipe = RebucketHistogramPipe::new(inner, NEW_BOUNDS);
            let sample = MetricSample::Counter(crate::metric::sample::NumberDataPoint {
                value: ScalarValue::U64(5),
                attrs: smallvec::SmallVec::new(),
                ts_ns: 0,
                start_ts_ns: 0,
            });
            let request = metric_request(sample);
            block_on(SendPipe::call(&pipe, request)).expect("call ok");
            assert_eq!(metrics.load(Ordering::Relaxed), 1);
        }

        #[test]
        fn rebucket_histogram_composition_with_drop_attr() {
            let (inner, _, _, _, metrics, _) = CountingPipe::new();
            let pipe = RebucketHistogramPipe::new(DropAttrPipe::new(inner, &["debug"]), NEW_BOUNDS);
            let request = make_histogram_request(OLD_BOUNDS, vec![1, 2, 3, 4, 5]);
            block_on(SendPipe::call(&pipe, request)).expect("call ok");
            assert_eq!(metrics.load(Ordering::Relaxed), 1);
        }
    }

    // InMemoryPipe tests

    #[test]
    fn happy_log_lands_in_store() {
        let pipe = InMemoryPipe::new();
        let record = make_log(Level::INFO);
        let request = log_request(record);
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        let stored = pipe.logs();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].level, Level::INFO);
        assert_eq!(stored[0].module_path, "");
    }

    #[test]
    fn happy_span_lands_in_store() {
        let pipe = InMemoryPipe::new();
        let record = make_span();
        let request = span_request(record);
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        let stored = pipe.spans();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].name, "test");
    }

    #[test]
    fn batch_logs_all_land_in_store() {
        let pipe = InMemoryPipe::new();
        let records = vec![
            make_log(Level::INFO),
            make_log(Level::WARN),
            make_log(Level::ERROR),
        ];
        let request = log_batch_request(records);
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(pipe.logs().len(), 3);
    }

    #[test]
    fn clear_drops_all_records() {
        let pipe = InMemoryPipe::new();
        for _ in 0..5 {
            block_on(SendPipe::call(&pipe, log_request(make_log(Level::INFO)))).expect("call ok");
        }
        assert_eq!(pipe.total(), 5);
        pipe.clear();
        assert_eq!(pipe.total(), 0);
    }

    #[test]
    fn total_counts_all_record_types() {
        let pipe = InMemoryPipe::new();
        block_on(SendPipe::call(&pipe, log_request(make_log(Level::INFO)))).expect("call ok");
        block_on(SendPipe::call(&pipe, span_request(make_span()))).expect("call ok");
        let event_req = event_request(EventRecord {
            parent_span_id: SpanId::INVALID,
            name: "ev",
            ts_ns: 0,
            attrs: smallvec::SmallVec::new(),
            module_path: "",
            file_line: (0, 0),
        });
        block_on(SendPipe::call(&pipe, event_req)).expect("call ok");
        let metric_req = metric_request(make_metric_counter("k", ScalarValue::U64(1)));
        block_on(SendPipe::call(&pipe, metric_req)).expect("call ok");
        let link_req = link_request(SpanLink::new(TraceId::INVALID, SpanId::INVALID));
        block_on(SendPipe::call(&pipe, link_req)).expect("call ok");
        assert_eq!(pipe.total(), 5);
    }

    #[test]
    fn concurrent_emits_dont_lose_records() {
        use std::sync::Arc;
        use std::thread;

        let pipe = Arc::new(InMemoryPipe::new());
        let mut handles = alloc::vec::Vec::new();
        for _ in 0..8 {
            let pipe_clone = Arc::clone(&pipe);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let request = log_request(make_log(Level::INFO));
                    block_on(SendPipe::call(&*pipe_clone, request)).expect("call ok");
                }
            }));
        }
        for handle in handles {
            handle.join().expect("thread ok");
        }
        assert_eq!(pipe.logs().len(), 800);
    }

    // FormatterPipe tests

    #[test]
    fn rfc3339_formats_known_epochs() {
        assert_eq!(super::format_rfc3339(0), "1970-01-01T00:00:00.000000000Z");
        assert_eq!(
            super::format_rfc3339(1_000_000_000),
            "1970-01-01T00:00:01.000000000Z"
        );
        // 2000-01-01T00:00:00Z = 946_684_800 s (leap-year boundary).
        assert_eq!(
            super::format_rfc3339(946_684_800_000_000_000),
            "2000-01-01T00:00:00.000000000Z"
        );
        // sub-second nanos preserved, full precision.
        assert_eq!(
            super::format_rfc3339(1_609_459_200_123_456_789),
            "2021-01-01T00:00:00.123456789Z"
        );
    }

    #[test]
    fn happy_log_writes_to_sink() {
        let sink = alloc::vec::Vec::<u8>::new();
        let pipe = FormatterPipe::new(sink, LogFormat::Human);
        let record = LogRecord {
            ts_ns: 0,
            observed_ts_ns: 0,
            level: Level::WARN,
            body: crate::log::LogBody::Text("hello world"),
            attrs: smallvec::SmallVec::new(),
            trace_id: None,
            span_id: None,
            trace_flags: TraceFlags(0),
            module_path: "mymod",
            file_line: (0, 0),
        };
        block_on(SendPipe::call(&pipe, log_request(record))).expect("call ok");
        let output = alloc::string::String::from_utf8(pipe.writer.lock().clone()).expect("utf8");
        assert!(
            output.contains("WARN"),
            "level missing from output: {output}"
        );
        assert!(output.contains("hello world"), "message missing: {output}");
    }

    #[test]
    fn happy_span_writes_to_sink() {
        let sink = alloc::vec::Vec::<u8>::new();
        let pipe = FormatterPipe::new(sink, LogFormat::Human);
        let record = make_span();
        block_on(SendPipe::call(&pipe, span_request(record))).expect("call ok");
        let output = alloc::string::String::from_utf8(pipe.writer.lock().clone()).expect("utf8");
        assert!(output.contains("SPAN"), "span marker missing: {output}");
        assert!(output.contains("test"), "span name missing: {output}");
    }

    #[test]
    fn json_format_produces_valid_json() {
        let sink = alloc::vec::Vec::<u8>::new();
        let pipe = FormatterPipe::new(sink, LogFormat::Json);
        let record = LogRecord {
            ts_ns: 12345,
            observed_ts_ns: 0,
            level: Level::ERROR,
            body: crate::log::LogBody::Text("oops"),
            attrs: smallvec::SmallVec::new(),
            trace_id: None,
            span_id: None,
            trace_flags: TraceFlags(0),
            module_path: "mod",
            file_line: (0, 0),
        };
        block_on(SendPipe::call(&pipe, log_request(record))).expect("call ok");
        let output = alloc::string::String::from_utf8(pipe.writer.lock().clone()).expect("utf8");
        let trimmed = output.trim();
        let parsed: serde_json::Value = serde_json::from_str(trimmed).expect("valid json");
        assert!(parsed.get("severity").is_some(), "severity field missing");
        assert!(parsed.get("body").is_some(), "body field missing");
    }

    #[test]
    fn batch_logs_each_written_once() {
        let sink = alloc::vec::Vec::<u8>::new();
        let pipe = FormatterPipe::new(sink, LogFormat::Human);
        let records = vec![
            make_log(Level::INFO),
            make_log(Level::WARN),
            make_log(Level::ERROR),
        ];
        block_on(SendPipe::call(&pipe, log_batch_request(records))).expect("call ok");
        let output = alloc::string::String::from_utf8(pipe.writer.lock().clone()).expect("utf8");
        assert_eq!(output.lines().count(), 3, "expected 3 lines: {output}");
    }

    #[test]
    fn human_format_includes_level_and_message() {
        let sink = alloc::vec::Vec::<u8>::new();
        let pipe = FormatterPipe::new(sink, LogFormat::Human);
        let record = LogRecord {
            ts_ns: 0,
            observed_ts_ns: 0,
            level: Level::WARN,
            body: crate::log::LogBody::Text("watch out"),
            attrs: smallvec::SmallVec::new(),
            trace_id: None,
            span_id: None,
            trace_flags: TraceFlags(0),
            module_path: "",
            file_line: (0, 0),
        };
        block_on(SendPipe::call(&pipe, log_request(record))).expect("call ok");
        let output = alloc::string::String::from_utf8(pipe.writer.lock().clone()).expect("utf8");
        // new tracing-shape line: `<rfc3339> LEVEL <module_path>: <message>`.
        assert!(output.contains("WARN"), "uppercase level missing: {output}");
        assert!(output.contains("watch out"), "message missing: {output}");
        assert!(
            output.starts_with("1970-01-01T00:00:00"),
            "rfc3339 ts prefix: {output}"
        );
    }

    // TelemetryPipeExt fluent API test

    #[test]
    fn telemetry_pipe_ext_chain_compiles_and_runs() {
        let (inner, spans, _, logs, metrics, _) = CountingPipe::new();
        let pipe = inner
            .filter_by_level(Level::WARN)
            .filter_random_drop(1.0)
            .drop_attrs(&["user_id"])
            .rename_metric("old", "new");

        let request = span_request(make_span());
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(spans.load(Ordering::Relaxed), 1);

        let request = log_request(make_log(Level::WARN));
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(logs.load(Ordering::Relaxed), 1);

        let request = log_request(make_log(Level::DEBUG));
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(logs.load(Ordering::Relaxed), 1);

        let request = metric_request(make_metric_counter("x", ScalarValue::U64(1)));
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        assert_eq!(metrics.load(Ordering::Relaxed), 1);
    }

    // SumByPipe tests

    fn make_counter_point(route: &'static str, value: u64, ts_ns: u64) -> MetricSample {
        let mut point = NumberDataPoint {
            value: ScalarValue::U64(value),
            attrs: smallvec::SmallVec::new(),
            ts_ns,
            start_ts_ns: 0,
        };
        point.attrs.push(Tag::Scalar {
            key: "route",
            value: ScalarValue::Str(route),
        });
        MetricSample::Counter(point)
    }

    #[test]
    fn sum_by_groups_increments_by_route() {
        use std::sync::Arc;
        use std::time::Duration;

        let inner = Arc::new(InMemoryPipe::new());
        let pipe = SumByPipe::new(Arc::clone(&inner), Duration::from_secs(60), ["route"]);

        for value in [10u64, 20, 30] {
            let req = metric_request(make_counter_point("/a", value, 1000));
            block_on(SendPipe::call(&pipe, req)).expect("call ok");
        }
        for value in [100u64, 200] {
            let req = metric_request(make_counter_point("/b", value, 2000));
            block_on(SendPipe::call(&pipe, req)).expect("call ok");
        }

        block_on(pipe.flush()).expect("flush ok");

        let metrics = inner.metrics();
        assert_eq!(metrics.len(), 2, "expected 2 aggregate groups");

        let mut sums: alloc::vec::Vec<u64> = metrics
            .iter()
            .map(|sample| {
                if let MetricSample::Counter(point) = sample {
                    if let ScalarValue::U64(val) = point.value {
                        val
                    } else {
                        0
                    }
                } else {
                    0
                }
            })
            .collect();
        sums.sort_unstable();
        assert_eq!(sums, &[60, 300], "expected /a=60 and /b=300");
    }

    #[test]
    fn sum_by_emits_after_window_expires() {
        use std::sync::Arc;
        use std::time::Duration;

        let inner = Arc::new(InMemoryPipe::new());
        let pipe = SumByPipe::new(Arc::clone(&inner), Duration::from_millis(1), ["route"]);

        let req = metric_request(make_counter_point("/a", 5, 1000));
        block_on(SendPipe::call(&pipe, req)).expect("call ok");

        std::thread::sleep(Duration::from_millis(5));

        // this second record triggers the window flush before accumulating itself
        let req = metric_request(make_counter_point("/a", 7, 2000));
        block_on(SendPipe::call(&pipe, req)).expect("call ok");

        // flush remaining (the second record's window)
        block_on(pipe.flush()).expect("flush ok");

        let metrics = inner.metrics();
        assert!(
            !metrics.is_empty(),
            "expected at least one aggregate emitted"
        );

        // the first window flush should have emitted sum=5 for /a
        let first = &metrics[0];
        if let MetricSample::Counter(point) = first {
            assert!(
                matches!(point.value, ScalarValue::U64(5)),
                "expected first window aggregate of 5, got {:?}",
                point.value
            );
        } else {
            panic!("expected Counter variant");
        }
    }

    #[test]
    fn flush_method_forces_emit() {
        use std::sync::Arc;
        use std::time::Duration;

        let inner = Arc::new(InMemoryPipe::new());
        let pipe = SumByPipe::new(Arc::clone(&inner), Duration::from_secs(60), ["route"]);

        for value in [1u64, 2, 3] {
            let req = metric_request(make_counter_point("/x", value, 1000));
            block_on(SendPipe::call(&pipe, req)).expect("call ok");
        }

        assert_eq!(inner.metrics().len(), 0, "no flush yet");
        block_on(pipe.flush()).expect("flush ok");
        assert_eq!(
            inner.metrics().len(),
            1,
            "flush should have emitted one aggregate"
        );

        if let MetricSample::Counter(point) = &inner.metrics()[0] {
            assert_eq!(point.value, ScalarValue::U64(6));
        } else {
            panic!("expected Counter");
        }
    }

    #[test]
    fn non_counter_records_pass_through() {
        use std::time::Duration;
        let (inner, _, _, logs, _, _) = CountingPipe::new();
        let pipe = SumByPipe::new(inner, Duration::from_secs(60), ["route"]);

        let req = log_request(make_log(Level::INFO));
        block_on(SendPipe::call(&pipe, req)).expect("call ok");

        assert_eq!(
            logs.load(Ordering::Relaxed),
            1,
            "log should pass through unchanged"
        );
    }

    #[test]
    fn batched_counter_input_accumulates() {
        use std::sync::Arc;
        use std::time::Duration;

        let inner = Arc::new(InMemoryPipe::new());
        let pipe = SumByPipe::new(Arc::clone(&inner), Duration::from_secs(60), ["route"]);

        let batch: alloc::vec::Vec<MetricSample> = (0..100)
            .map(|i| make_counter_point("/batch", 1, i as u64 * 1000))
            .collect();
        let req = metric_batch_request(batch);
        block_on(SendPipe::call(&pipe, req)).expect("call ok");

        block_on(pipe.flush()).expect("flush ok");

        let metrics = inner.metrics();
        assert_eq!(metrics.len(), 1, "all 100 should collapse to one group");
        if let MetricSample::Counter(point) = &metrics[0] {
            assert_eq!(point.value, ScalarValue::U64(100));
        } else {
            panic!("expected Counter");
        }
    }

    #[test]
    fn zero_group_by_aggregates_everything() {
        use std::sync::Arc;
        use std::time::Duration;

        let inner = Arc::new(InMemoryPipe::new());
        // empty group_by: all counters map to one group
        let pipe = SumByPipe::new(Arc::clone(&inner), Duration::from_secs(60), []);

        for route in ["/a", "/b", "/c"] {
            let req = metric_request(make_counter_point(route, 10, 1000));
            block_on(SendPipe::call(&pipe, req)).expect("call ok");
        }

        block_on(pipe.flush()).expect("flush ok");

        let metrics = inner.metrics();
        assert_eq!(
            metrics.len(),
            1,
            "all routes should collapse into one group"
        );
        if let MetricSample::Counter(point) = &metrics[0] {
            assert_eq!(point.value, ScalarValue::U64(30));
        } else {
            panic!("expected Counter");
        }
    }

    // Arc-batch tests (P17)

    #[test]
    fn arc_batch_lands_in_in_memory_pipe() {
        let pipe = InMemoryPipe::new();
        let records: alloc::vec::Vec<Arc<SpanRecord>> = (0..3)
            .map(|i| {
                Arc::new(SpanRecord {
                    trace_id: TraceId::INVALID,
                    span_id: SpanId::INVALID,
                    parent_span_id: None,
                    name: "arc-test",
                    kind: SpanKind::Internal,
                    start_ns: i,
                    duration_ns: 0,
                    status: Status::Unset,
                    attrs: smallvec::SmallVec::new(),
                    events: smallvec::SmallVec::new(),
                    links: smallvec::SmallVec::new(),
                    tracestate: TraceState::empty(),
                    module_path: "",
                    file_line: (0, 0),
                })
            })
            .collect();
        let request = span_batch_arc_request(records);
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        let stored = pipe.spans();
        assert_eq!(
            stored.len(),
            3,
            "all 3 Arc<SpanRecord>s must land in InMemoryPipe"
        );
        assert!(
            stored.iter().all(|r| r.name == "arc-test"),
            "names must match"
        );
    }

    #[test]
    fn arc_batch_filter_via_random_drop() {
        let (inner, _, _, logs, _, _) = CountingPipe::new();
        let pipe = RandomDropPipe::new(inner, 0.5);
        let records: alloc::vec::Vec<Arc<LogRecord>> =
            (0..1000).map(|_| Arc::new(make_log(Level::INFO))).collect();
        let request = log_batch_arc_request(records);
        block_on(SendPipe::call(&pipe, request)).expect("call ok");
        let count = logs.load(Ordering::Relaxed);
        assert!(
            count > 300 && count < 700,
            "RandomDropPipe 0.5 should pass ~500/1000; got {count}"
        );
    }

    #[test]
    fn arc_batch_compatible_with_inline_pipes() {
        let pipe = InMemoryPipe::new();

        let inline_spans: alloc::vec::Vec<SpanRecord> = (0..2).map(|_| make_span()).collect();
        block_on(SendPipe::call(&pipe, span_batch_request(inline_spans))).expect("inline ok");

        let arc_spans: alloc::vec::Vec<Arc<SpanRecord>> =
            (0..3).map(|_| Arc::new(make_span())).collect();
        block_on(SendPipe::call(&pipe, span_batch_arc_request(arc_spans))).expect("arc ok");

        assert_eq!(
            pipe.spans().len(),
            5,
            "both inline and arc batches must reach the store"
        );
    }

    #[test]
    fn arc_batch_counting_pipe_counts_correctly() {
        let (pipe, spans, _, logs, _, _) = CountingPipe::new();

        let arc_spans: alloc::vec::Vec<Arc<SpanRecord>> =
            (0..7).map(|_| Arc::new(make_span())).collect();
        block_on(SendPipe::call(&pipe, span_batch_arc_request(arc_spans))).expect("call ok");

        let arc_logs: alloc::vec::Vec<Arc<LogRecord>> =
            (0..4).map(|_| Arc::new(make_log(Level::INFO))).collect();
        block_on(SendPipe::call(&pipe, log_batch_arc_request(arc_logs))).expect("call ok");

        assert_eq!(
            spans.load(Ordering::Relaxed),
            7,
            "7 Arc spans must be counted"
        );
        assert_eq!(
            logs.load(Ordering::Relaxed),
            4,
            "4 Arc logs must be counted"
        );
    }

    struct FailingPipe;

    impl SendPipe for FailingPipe {
        type In = TelemetryRequest;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: TelemetryRequest,
        ) -> impl StdFuture<Output = Result<Response<Bytes>, ProximaError>> + Send {
            async { Err(ProximaError::Upstream("exporter down".into())) }
        }
    }

    #[test]
    fn fan_exporters_delivers_to_every_exporter() {
        let (first, _, _, first_logs, _, _) = CountingPipe::new();
        let (second, _, _, second_logs, _, _) = CountingPipe::new();
        let fan = fan_exporters(vec![
            into_telemetry_handle(first),
            into_telemetry_handle(second),
        ]);

        block_on(fan.call_dyn(log_request(make_log(Level::INFO)))).expect("fan call ok");

        assert_eq!(first_logs.load(Ordering::Relaxed), 1, "primary received");
        assert_eq!(second_logs.load(Ordering::Relaxed), 1, "secondary received");
    }

    #[test]
    fn fan_exporters_single_is_passthrough_handle() {
        let (only, _, _, logs, _, _) = CountingPipe::new();
        let fan = fan_exporters(vec![into_telemetry_handle(only)]);

        block_on(fan.call_dyn(log_request(make_log(Level::INFO)))).expect("call ok");

        assert_eq!(logs.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn fan_exporters_empty_is_noop() {
        let fan = fan_exporters(vec![]);
        block_on(fan.call_dyn(log_request(make_log(Level::INFO)))).expect("null pipe accepts");
    }

    #[test]
    fn fan_exporters_secondary_error_does_not_fail_the_fan() {
        let (primary, _, _, primary_logs, _, _) = CountingPipe::new();
        let fan = fan_exporters(vec![
            into_telemetry_handle(primary),
            into_telemetry_handle(FailingPipe),
        ]);

        block_on(fan.call_dyn(log_request(make_log(Level::INFO))))
            .expect("primary ok despite secondary failure");

        assert_eq!(
            primary_logs.load(Ordering::Relaxed),
            1,
            "primary still received"
        );
    }

    // each exporter records the peak in-flight count, then yields once so its
    // siblings get polled before it finishes. Concurrent delivery drives all N
    // into flight at once (peak == N); a sequential fan would complete each
    // before starting the next (peak == 1). Runtime-agnostic, no sleeps.
    struct InflightProbe {
        inflight: alloc::sync::Arc<core::sync::atomic::AtomicUsize>,
        peak: alloc::sync::Arc<core::sync::atomic::AtomicUsize>,
    }

    impl SendPipe for InflightProbe {
        type In = TelemetryRequest;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: TelemetryRequest,
        ) -> impl StdFuture<Output = Result<Response<Bytes>, ProximaError>> + Send {
            let inflight = alloc::sync::Arc::clone(&self.inflight);
            let peak = alloc::sync::Arc::clone(&self.peak);
            async move {
                let now = inflight.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                yield_once().await;
                inflight.fetch_sub(1, Ordering::SeqCst);
                Ok(ok_response())
            }
        }
    }

    fn yield_once() -> impl StdFuture<Output = ()> {
        let mut yielded = false;
        core::future::poll_fn(move |cx| {
            if yielded {
                core::task::Poll::Ready(())
            } else {
                yielded = true;
                cx.waker().wake_by_ref();
                core::task::Poll::Pending
            }
        })
    }

    #[proxima::test]
    async fn fan_exporters_delivers_concurrently() {
        let exporter_count = 4;
        let inflight = alloc::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0));
        let peak = alloc::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0));
        let fan = fan_exporters(
            (0..exporter_count)
                .map(|_| {
                    into_telemetry_handle(InflightProbe {
                        inflight: alloc::sync::Arc::clone(&inflight),
                        peak: alloc::sync::Arc::clone(&peak),
                    })
                })
                .collect(),
        );

        fan.call_dyn(log_request(make_log(Level::INFO)))
            .await
            .expect("fan ok");

        assert_eq!(
            peak.load(Ordering::SeqCst),
            exporter_count,
            "every exporter must be in flight at once; sequential delivery would peak at 1"
        );
    }
}

#[cfg(all(test, feature = "elevation", not(feature = "loom")))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod elevation_sink_tests {
    use super::*;
    use crate::id::{SpanId, TraceFlags, TraceId};
    use crate::log::body::LogBody;
    use crate::trace::{SpanKind, Status, TraceState};
    use futures::executor::block_on;
    use std::sync::Mutex;

    // a terminal pipe that records the log records it is handed — the elevated
    // forensic sink under test.
    #[derive(Clone)]
    struct Capture {
        seen: Arc<Mutex<Vec<LogRecord>>>,
    }

    impl Capture {
        fn new() -> Self {
            Self {
                seen: Arc::new(Mutex::new(Vec::new())),
            }
        }
        fn timestamps(&self) -> Vec<u64> {
            self.seen.lock().unwrap().iter().map(|record| record.ts_ns).collect()
        }
    }

    impl SendPipe for Capture {
        type In = TelemetryRequest;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            request: TelemetryRequest,
        ) -> impl StdFuture<Output = Result<Response<Bytes>, ProximaError>> + Send {
            let seen = Arc::clone(&self.seen);
            async move {
                if let TelemetryRecord::LogBatch(records) = &request.payload {
                    seen.lock().unwrap().extend(records.iter().cloned());
                }
                Ok(ok_response())
            }
        }
    }

    fn trace_id(byte: u8) -> TraceId {
        TraceId::from_bytes([byte; 16])
    }

    // a record as it looks emitted inside a verbose-sampled trace: SAMPLED +
    // VERBOSE_BUFFERED, carrying its trace id and event timestamp.
    fn verbose_log(trace: TraceId, level: Level, ts_ns: u64) -> LogRecord {
        LogRecord {
            ts_ns,
            observed_ts_ns: ts_ns,
            level,
            body: LogBody::Empty,
            attrs: SmallVec::new(),
            trace_id: Some(trace),
            span_id: Some(SpanId::from_bytes([1; 8])),
            trace_flags: TraceFlags::SAMPLED.with_verbose_buffered(),
            module_path: "test",
            file_line: (0, 0),
        }
    }

    fn root_span(trace: TraceId) -> SpanRecord {
        SpanRecord {
            trace_id: trace,
            span_id: SpanId::from_bytes([1; 8]),
            parent_span_id: None,
            name: "root",
            kind: SpanKind::Internal,
            start_ns: 0,
            duration_ns: 10,
            status: Status::Unset,
            attrs: SmallVec::new(),
            events: SmallVec::new(),
            links: SmallVec::new(),
            tracestate: TraceState(None),
            module_path: "test",
            file_line: (0, 0),
        }
    }

    fn sink_to(capture: &Capture, max_traces: usize) -> ElevationSink {
        ElevationSink::new(
            into_telemetry_handle(capture.clone()),
            Level::ERROR,
            256,
            max_traces,
            0,
            true,
        )
    }

    // the headline behaviour: floor+ AND below-floor records of a verbose trace
    // are buffered, and an error replays the WHOLE tree ordered by ts_ns.
    #[test]
    fn error_replays_full_ordered_tree_to_elevated_sink() {
        let capture = Capture::new();
        let sink = sink_to(&capture, 1024);
        let trace = trace_id(9);
        block_on(SendPipe::call(
            &sink,
            log_batch_request(alloc::vec![
                verbose_log(trace, Level::INFO, 300),
                verbose_log(trace, Level::TRACE, 100),
                verbose_log(trace, Level::DEBUG, 200),
            ]),
        ))
        .expect("buffer ok");
        assert!(capture.timestamps().is_empty(), "no trigger yet: nothing replayed");

        block_on(SendPipe::call(
            &sink,
            log_batch_request(alloc::vec![verbose_log(trace, Level::ERROR, 400)]),
        ))
        .expect("trigger ok");
        assert_eq!(
            capture.timestamps(),
            alloc::vec![100, 200, 300, 400],
            "the full tree (trace..error) replays in ts_ns order"
        );
    }

    // a non-verbose trace's records — even an error — are never buffered or
    // replayed: the healthy path is untouched.
    #[test]
    fn non_verbose_records_are_ignored() {
        let capture = Capture::new();
        let sink = sink_to(&capture, 1024);
        let trace = trace_id(3);
        let mut record = verbose_log(trace, Level::ERROR, 100);
        record.trace_flags = TraceFlags::SAMPLED; // not verbose-buffered
        block_on(SendPipe::call(&sink, log_batch_request(alloc::vec![record]))).expect("ok");
        assert!(capture.timestamps().is_empty(), "non-verbose error is not replayed");
    }

    // root-span close is the completion signal: an untriggered trace's buffer is
    // dropped, so a later error can't resurrect the pre-close tree.
    #[test]
    fn root_span_close_drops_untriggered_buffer() {
        let capture = Capture::new();
        let sink = sink_to(&capture, 1024);
        let trace = trace_id(5);
        block_on(SendPipe::call(
            &sink,
            log_batch_request(alloc::vec![verbose_log(trace, Level::INFO, 100)]),
        ))
        .expect("ok");
        block_on(SendPipe::call(&sink, span_batch_request(alloc::vec![root_span(trace)]))).expect("ok");
        block_on(SendPipe::call(
            &sink,
            log_batch_request(alloc::vec![verbose_log(trace, Level::ERROR, 200)]),
        ))
        .expect("ok");
        assert_eq!(
            capture.timestamps(),
            alloc::vec![200],
            "only the post-close error replays; the dropped tree is gone"
        );
    }

    // the count-cap is the OOM backstop: buffering beyond max_traces evicts the
    // least-recently-touched trace, so memory is bounded regardless.
    #[test]
    fn count_cap_bounds_concurrent_traces() {
        let capture = Capture::new();
        let sink = sink_to(&capture, 2);
        for (index, byte) in [10u8, 11, 12].into_iter().enumerate() {
            block_on(SendPipe::call(
                &sink,
                log_batch_request(alloc::vec![verbose_log(trace_id(byte), Level::INFO, index as u64 + 1)]),
            ))
            .expect("ok");
        }
        assert_eq!(
            sink.state.buffers.len(),
            2,
            "the cap holds: the oldest trace was evicted, never a third buffer"
        );
    }

    // the load-bearing occupancy proof: drive far more concurrent traces than
    // max_traces, each pushing far more records than per_trace_ring, and assert
    // BOTH caps hold at every step (not just at the end) — the worst-case
    // steady-state occupancy is exactly max_traces * per_trace_ring records,
    // never more, regardless of load shape.
    #[test]
    fn occupancy_is_hard_bounded_by_max_traces_and_per_trace_ring() {
        let capture = Capture::new();
        let max_traces = 4;
        let per_trace_ring = 8;
        let sink = ElevationSink::new(
            into_telemetry_handle(capture.clone()),
            Level::ERROR,
            per_trace_ring,
            max_traces,
            0,
            true,
        );

        let trace_count = max_traces * 5; // far more concurrent traces than the cap
        let records_per_trace = per_trace_ring * 3; // far more records per trace than the ring
        let mut ts_ns = 0u64;

        for byte in 0..trace_count {
            let trace = trace_id(byte as u8);
            for _ in 0..records_per_trace {
                ts_ns += 1;
                block_on(SendPipe::call(
                    &sink,
                    log_batch_request(alloc::vec![verbose_log(trace, Level::INFO, ts_ns)]),
                ))
                .expect("buffer ok");

                assert!(
                    sink.state.buffers.len() <= max_traces,
                    "buffers.len()={} exceeded max_traces={max_traces}",
                    sink.state.buffers.len(),
                );
                for entry in sink.state.buffers.iter() {
                    assert!(
                        entry.value().ring.len() <= per_trace_ring,
                        "trace ring len={} exceeded per_trace_ring={per_trace_ring}",
                        entry.value().ring.len(),
                    );
                }
            }
        }

        assert_eq!(
            sink.state.buffers.len(),
            max_traces,
            "steady state settles at exactly the cap under sustained trace overflow"
        );
        for entry in sink.state.buffers.iter() {
            assert_eq!(
                entry.value().ring.len(),
                per_trace_ring,
                "each surviving trace's ring settles at exactly its cap under sustained record overflow"
            );
        }
    }
}
