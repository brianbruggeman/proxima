extern crate std;

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::task::{Context, Poll, Waker};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::pipes::{TelemetryPipeHandle, TelemetryRequest};

use bytes::Bytes;

use crate::config::RecordSharing;
use crate::log::LogRecord;
use crate::metric::MetricSample;
use crate::pipes::{
    event_batch_arc_request, event_batch_request, link_batch_arc_request, link_batch_request,
    log_batch_arc_request, log_batch_request, metric_batch_arc_request, metric_batch_request,
    span_batch_arc_request, span_batch_request,
};
use crate::ring::HeapBoundedQueue;
use crate::trace::{EventRecord, SpanLink, SpanRecord};
use proxima_primitives::pipe::{BatchSource, SendDynPipe};
use proxima_primitives::pipe::request::Response;

use super::EmitShared;
use super::ring_set::OverflowAttr;
#[cfg(feature = "deferred-metric-fold")]
use super::ring_set::SpanObservation;

/// Batch-drain consumer over an [`EmitShared`]'s rings + instrument registry.
///
/// A thin owned-`Arc` wrapper so the public `drain`/`drain_range` API and a
/// managed drainer thread can drive the shared state. The work lives in the
/// `&EmitShared` free functions below — the same functions the emit/overflow
/// path's elastic producer-assist calls per-ring, and that `EmitShared::drop`
/// calls for the lossless shutdown flush.
pub struct Drainer {
    shared: Arc<EmitShared>,
}

impl Drainer {
    pub fn new(shared: Arc<EmitShared>) -> Self {
        Self { shared }
    }

    /// Drain one pass across all cores plus the instrument registry.
    pub fn drain_pass(&self) -> usize {
        drain_pass(&self.shared)
    }

    /// Drain the ring records of cores `[start, end)` only (no registry
    /// instruments). The rings are multi-consumer, so calling this from several
    /// threads — over disjoint ranges OR the same range — never reads a cell
    /// twice; partitioning is the parallel-drain primitive that lifts the
    /// single-drainer ceiling.
    pub fn drain_cores(&self, start: usize, end: usize) -> usize {
        drain_cores(&self.shared, start, end)
    }
}

/// Drain one pass across all cores plus the instrument registry. Returns total
/// records exported. Drain order: span -> event -> log -> metric -> link ->
/// overflow_attr, then instruments. Called by [`Drainer`] and by
/// `EmitShared::drop` (final flush).
pub(crate) fn drain_pass(shared: &EmitShared) -> usize {
    let mut total = drain_cores(shared, 0, shared.cores().count());
    let pipe_handle: TelemetryPipeHandle = shared.pipe().read(TelemetryPipeHandle::clone);
    let ts_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|dur| dur.as_nanos() as u64)
        .unwrap_or(0);
    total += shared
        .registry()
        .drain_instruments(ts_ns, pipe_handle.as_ref());
    total
}

/// Drain + export the ring records of cores `[start, end)` (no instruments).
pub(crate) fn drain_cores(shared: &EmitShared, start: usize, end: usize) -> usize {
    let pipe_handle: TelemetryPipeHandle = shared.pipe().read(TelemetryPipeHandle::clone);
    let pipe = pipe_handle.as_ref();
    let sharing = shared.sharing();
    let batch = shared.drain_batch();

    let mut total = 0usize;
    for core_index in start..end.min(shared.cores().count()) {
        let ring_set = shared.cores().slot(core_index);
        total += drain_one(
            shared,
            pipe,
            &ring_set.spans,
            sharing,
            batch,
            span_sentinel,
            span_batch_request,
            span_batch_arc_request,
        );
        total += drain_one(
            shared,
            pipe,
            &ring_set.events,
            sharing,
            batch,
            event_sentinel,
            event_batch_request,
            event_batch_arc_request,
        );
        total += drain_one(
            shared,
            pipe,
            &ring_set.logs,
            sharing,
            batch,
            log_sentinel,
            log_batch_request,
            log_batch_arc_request,
        );
        total += drain_one(
            shared,
            pipe,
            &ring_set.metrics,
            sharing,
            batch,
            metric_sentinel,
            metric_batch_request,
            metric_batch_arc_request,
        );
        total += drain_one(
            shared,
            pipe,
            &ring_set.links,
            sharing,
            batch,
            link_sentinel,
            link_batch_request,
            link_batch_arc_request,
        );
        total += drain_overflow_attrs(&ring_set.overflow_attrs, batch);
        #[cfg(feature = "deferred-metric-fold")]
        {
            total += fold_span_observations(shared, &ring_set.span_obs, batch);
        }
    }
    total
}

// dequeue one batch (frees the ring slots), wake any parked producer, THEN export.
// notifying before the slow export — not after the whole pass — is what bounds a
// parked producer's wakeup to dequeue speed instead of the batch's export time.
#[allow(clippy::too_many_arguments)]
fn drain_one<Record>(
    shared: &EmitShared,
    pipe: &dyn SendDynPipe<TelemetryRequest, Response<Bytes>>,
    ring: &HeapBoundedQueue<Record>,
    sharing: RecordSharing,
    batch: usize,
    make_sentinel: impl Fn() -> Record,
    inline: fn(Vec<Record>) -> TelemetryRequest,
    arc: fn(Vec<Arc<Record>>) -> TelemetryRequest,
) -> usize {
    match drain_to_request(ring, sharing, batch, make_sentinel, inline, arc) {
        Some((count, request)) => {
            shared.notify_slots_freed();
            call_pipe(pipe, request);
            count
        }
        None => 0,
    }
}

/// Async sibling of [`drain_cores`]: drains the ring records of cores
/// `[start, end)` and `.await`s the terminal pipe instead of `block_on`-ing it.
///
/// This is the prime-first export path — a managed drainer running as a prime
/// task (or any prime caller) drives this so a network terminal's I/O is awaited
/// on the reactor, never blocked. The record-drain out of each ring is the same
/// sync, slot-freeing step; only the export is awaited. Registry instruments are
/// not drained here (see [`drain_pass`]); this covers the ring signals.
pub(crate) async fn drain_cores_async(shared: &EmitShared, start: usize, end: usize) -> usize {
    let pipe_handle: TelemetryPipeHandle = shared.pipe().read(TelemetryPipeHandle::clone);
    let pipe = pipe_handle.as_ref();
    let sharing = shared.sharing();
    let batch = shared.drain_batch();

    let mut total = 0usize;
    for core_index in start..end.min(shared.cores().count()) {
        let ring_set = shared.cores().slot(core_index);
        total += drain_one_async(
            shared,
            pipe,
            &ring_set.spans,
            sharing,
            batch,
            span_sentinel,
            span_batch_request,
            span_batch_arc_request,
        )
        .await;
        total += drain_one_async(
            shared,
            pipe,
            &ring_set.events,
            sharing,
            batch,
            event_sentinel,
            event_batch_request,
            event_batch_arc_request,
        )
        .await;
        total += drain_one_async(
            shared,
            pipe,
            &ring_set.logs,
            sharing,
            batch,
            log_sentinel,
            log_batch_request,
            log_batch_arc_request,
        )
        .await;
        total += drain_one_async(
            shared,
            pipe,
            &ring_set.metrics,
            sharing,
            batch,
            metric_sentinel,
            metric_batch_request,
            metric_batch_arc_request,
        )
        .await;
        total += drain_one_async(
            shared,
            pipe,
            &ring_set.links,
            sharing,
            batch,
            link_sentinel,
            link_batch_request,
            link_batch_arc_request,
        )
        .await;
        total += drain_overflow_attrs(&ring_set.overflow_attrs, batch);
        #[cfg(feature = "deferred-metric-fold")]
        {
            total += fold_span_observations(shared, &ring_set.span_obs, batch);
        }
    }
    total
}

/// Drain this core's deferred span-duration observations, folding each into the
/// registry histogram + observer via [`EmitShared::fold_span_obs`]. Bounded to
/// `batch` per pass like the other rings; the drain-until-empty pump loops. The
/// per-fold registry Mutex is uncontended here — one drainer, off the hot path.
#[cfg(feature = "deferred-metric-fold")]
fn fold_span_observations(
    shared: &EmitShared,
    ring: &HeapBoundedQueue<SpanObservation>,
    batch: usize,
) -> usize {
    let mut folded = 0;
    while folded < batch {
        let Some(observation) = ring.dequeue() else {
            break;
        };
        shared.fold_span_obs(observation);
        folded += 1;
    }
    folded
}

// Per-record-type "drain one batch from this ring and export it via the pipe"
// helpers. Each is callable from the drain pass above AND from the emit path's
// elastic producer-assist (a full ring => the producer drains+exports a batch
// itself to free a slot). The generic core differs per type only in the
// sentinel and the inline/arc request builders, so each helper is a one-liner.

// drain one batch out of the ring (the sync, slot-freeing step) and build the
// export request. shared by the sync and async export paths; returns None when
// the ring had nothing to drain.
fn drain_to_request<Record>(
    ring: &HeapBoundedQueue<Record>,
    sharing: RecordSharing,
    batch: usize,
    make_sentinel: impl Fn() -> Record,
    inline: fn(Vec<Record>) -> TelemetryRequest,
    arc: fn(Vec<Arc<Record>>) -> TelemetryRequest,
) -> Option<(usize, TelemetryRequest)> {
    let records = drain_owned(ring, batch, make_sentinel);
    let count = records.len();
    if count == 0 {
        return None;
    }
    let request = match sharing {
        RecordSharing::Inline => inline(records),
        RecordSharing::Arc => arc(records.into_iter().map(Arc::new).collect()),
    };
    Some((count, request))
}

// await the terminal for a drained batch (the prime-first export step).
async fn export_async(
    pipe: &dyn SendDynPipe<TelemetryRequest, Response<Bytes>>,
    drained: Option<(usize, TelemetryRequest)>,
) -> usize {
    match drained {
        Some((count, request)) => {
            if let Err(error) = pipe.call_dyn(request).await {
                tracing::error!(error = %error, "pipe dispatch error during async drain");
            }
            count
        }
        None => 0,
    }
}

// async sibling of `drain_one`: dequeue (frees slots) -> wake parkers -> await
// the export. notifying before the await keeps a parked producer's wakeup bound
// to dequeue speed, not the awaited export round-trip.
#[allow(clippy::too_many_arguments)]
async fn drain_one_async<Record>(
    shared: &EmitShared,
    pipe: &dyn SendDynPipe<TelemetryRequest, Response<Bytes>>,
    ring: &HeapBoundedQueue<Record>,
    sharing: RecordSharing,
    batch: usize,
    make_sentinel: impl Fn() -> Record,
    inline: fn(Vec<Record>) -> TelemetryRequest,
    arc: fn(Vec<Arc<Record>>) -> TelemetryRequest,
) -> usize {
    let drained = drain_to_request(ring, sharing, batch, make_sentinel, inline, arc);
    if drained.is_some() {
        shared.notify_slots_freed();
    }
    export_async(pipe, drained).await
}

fn drain_export<Record>(
    ring: &HeapBoundedQueue<Record>,
    pipe: &dyn SendDynPipe<TelemetryRequest, Response<Bytes>>,
    sharing: RecordSharing,
    batch: usize,
    make_sentinel: impl Fn() -> Record,
    inline: fn(Vec<Record>) -> TelemetryRequest,
    arc: fn(Vec<Arc<Record>>) -> TelemetryRequest,
) -> usize {
    match drain_to_request(ring, sharing, batch, make_sentinel, inline, arc) {
        Some((count, request)) => {
            call_pipe(pipe, request);
            count
        }
        None => 0,
    }
}

pub(crate) fn drain_export_spans(
    ring: &HeapBoundedQueue<SpanRecord>,
    pipe: &dyn SendDynPipe<TelemetryRequest, Response<Bytes>>,
    sharing: RecordSharing,
    batch: usize,
) -> usize {
    drain_export(
        ring,
        pipe,
        sharing,
        batch,
        span_sentinel,
        span_batch_request,
        span_batch_arc_request,
    )
}

pub(crate) fn drain_export_logs(
    ring: &HeapBoundedQueue<LogRecord>,
    pipe: &dyn SendDynPipe<TelemetryRequest, Response<Bytes>>,
    sharing: RecordSharing,
    batch: usize,
) -> usize {
    drain_export(
        ring,
        pipe,
        sharing,
        batch,
        log_sentinel,
        log_batch_request,
        log_batch_arc_request,
    )
}

pub(crate) fn drain_export_metrics(
    ring: &HeapBoundedQueue<MetricSample>,
    pipe: &dyn SendDynPipe<TelemetryRequest, Response<Bytes>>,
    sharing: RecordSharing,
    batch: usize,
) -> usize {
    drain_export(
        ring,
        pipe,
        sharing,
        batch,
        metric_sentinel,
        metric_batch_request,
        metric_batch_arc_request,
    )
}

// overflow attrs are internal accounting; drained to free slots, never exported.
fn drain_overflow_attrs(ring: &HeapBoundedQueue<OverflowAttr>, batch: usize) -> usize {
    let want = batch.min(ring.len());
    if want == 0 {
        return 0;
    }
    let mut buf: Vec<OverflowAttr> = Vec::with_capacity(want);
    buf.resize_with(want, || OverflowAttr {
        span_id: crate::id::SpanId::INVALID,
        tag: crate::tag::Tag::Scalar {
            key: "",
            value: crate::tag::ScalarValue::Bool(false),
        },
    });
    ring.drain_into(&mut buf[..])
}

fn span_sentinel() -> SpanRecord {
    SpanRecord {
        trace_id: crate::id::TraceId::INVALID,
        span_id: crate::id::SpanId::INVALID,
        parent_span_id: None,
        name: "",
        kind: crate::trace::SpanKind::Internal,
        start_ns: 0,
        duration_ns: 0,
        status: crate::trace::Status::Unset,
        attrs: smallvec::SmallVec::new(),
        events: smallvec::SmallVec::new(),
        links: smallvec::SmallVec::new(),
        tracestate: crate::trace::TraceState::empty(),
        module_path: "",
        file_line: (0, 0),
    }
}

fn event_sentinel() -> EventRecord {
    EventRecord {
        parent_span_id: crate::id::SpanId::INVALID,
        name: "",
        ts_ns: 0,
        attrs: smallvec::SmallVec::new(),
        module_path: "",
        file_line: (0, 0),
    }
}

fn log_sentinel() -> LogRecord {
    LogRecord {
        ts_ns: 0,
        observed_ts_ns: 0,
        level: crate::level::Level::INFO,
        body: crate::log::LogBody::Empty,
        attrs: smallvec::SmallVec::new(),
        trace_id: None,
        span_id: None,
        trace_flags: crate::id::TraceFlags::NOT_SAMPLED,
        module_path: "",
        file_line: (0, 0),
    }
}

fn metric_sentinel() -> MetricSample {
    MetricSample::Counter(crate::metric::NumberDataPoint {
        value: crate::tag::ScalarValue::U64(0),
        attrs: smallvec::SmallVec::new(),
        ts_ns: 0,
        start_ts_ns: 0,
    })
}

fn link_sentinel() -> SpanLink {
    SpanLink::new(crate::id::TraceId::INVALID, crate::id::SpanId::INVALID)
}

// generic over the pipe-forms `BatchSource` (a `&self` multi-consumer owned-batch
// pull), not a concrete queue: the recorder's drain drives the algebra's source
// trait, so any BatchSource — a BoundedQueue today, another bounded structure
// tomorrow — feeds the same drain. Owned batch (not the borrow-visitor
// `DrainSource`) because the export request needs to own its records.
fn drain_owned<Source, MakeSentinel>(
    source: &Source,
    batch_size: usize,
    make_sentinel: MakeSentinel,
) -> Vec<Source::Item>
where
    Source: BatchSource,
    MakeSentinel: Fn() -> Source::Item,
{
    // size to actual occupancy, not the batch cap: the old code constructed
    // `batch_size` (512) sentinel records EVERY drain for EVERY ring — most
    // drains leave most signal rings empty, so that was 512 wasted record
    // constructions per empty ring per drain. `len()` is a consumer-side lower
    // bound on the source (a producer racing in just drains next pass — same
    // as the pre-existing batch-cap behaviour), so no record is lost.
    let want = batch_size.min(source.len());
    if want == 0 {
        return Vec::new();
    }
    let mut buf: Vec<Source::Item> = Vec::with_capacity(want);
    buf.resize_with(want, &make_sentinel);
    let count = source.drain_batch(buf.as_mut_slice());
    buf.truncate(count);
    buf
}

fn call_pipe(pipe: &dyn SendDynPipe<TelemetryRequest, Response<Bytes>>, request: TelemetryRequest) {
    let mut future = pipe.call_dyn(request);
    // the terminal telemetry pipes do their work synchronously and hand back a
    // ready future. poll once with a noop waker to skip block_on's parker
    // setup/park machinery — that was the bulk of the residual fixed per-pass
    // cost. only a genuinely-pending (async) pipe falls back to block_on.
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let result = match future.as_mut().poll(&mut context) {
        Poll::Ready(result) => result,
        Poll::Pending => futures::executor::block_on(future),
    };
    if let Err(error) = result {
        tracing::error!(error = %error, "pipe dispatch error during drain");
    }
}

// these tests drive EmitShared's rings directly, which are proxima-core's
// Ring/StaticRing -- cfg-swapped to loom under `--features loom`
// (forwarded via proxima-core/loom), only usable inside an actual
// loom::model(...) closure, which these plain #[test] functions don't
// provide.
#[cfg(all(test, not(feature = "loom")))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::sync::Arc;

    use crate::config::RecordSharing;
    use crate::pipes::{
        METHOD_SPAN_BATCH, METHOD_SPAN_BATCH_ARC, TelemetryRequest, into_telemetry_handle,
    };
    use crate::recorder::EmitShared;
    use crate::recorder::PerCore;
    use crate::recorder::registry::InstrumentRegistry;
    use crate::recorder::ring_set::RingSet;

    use super::Drainer;

    fn make_drainer_with<
        P: proxima_primitives::pipe::SendPipe<
                In = TelemetryRequest,
                Out = proxima_primitives::pipe::request::Response<bytes::Bytes>,
                Err = proxima_primitives::pipe::ProximaError,
            > + Send
            + Sync
            + 'static,
    >(
        pipe: P,
        sharing: RecordSharing,
    ) -> (Drainer, Arc<EmitShared>) {
        let caps = crate::recorder::RingCapacities {
            spans: 256,
            events: 256,
            logs: 256,
            metrics: 256,
            links: 256,
            overflow_attrs: 256,
            #[cfg(feature = "deferred-metric-fold")]
            span_obs: 256,
        };
        let cores = PerCore::new_with(1, |_| RingSet::new(&caps).expect("ring set init in test"));
        let shared = Arc::new(EmitShared::new(
            cores,
            InstrumentRegistry::new(),
            into_telemetry_handle(pipe),
            sharing,
            256,
            64,
            core::time::Duration::from_millis(1),
        ));
        let drainer = Drainer::new(Arc::clone(&shared));
        (drainer, shared)
    }

    // The bounded-tail invariant: a single elastic-assist drain frees AT MOST
    // `assist_batch` slots, regardless of how full the ring is. This is what
    // bounds the worst-case emit stall (assist cost ≈ assist_batch × per-record
    // sink latency) on the producer/request thread — the producer frees just
    // enough headroom and returns to work, it does not flush the whole ring.
    #[test]
    fn assist_drains_at_most_assist_batch() {
        use crate::ring::{FailMode, HeapBoundedQueue};

        let ring = HeapBoundedQueue::<crate::trace::SpanRecord>::new(64, FailMode::DropNewest);
        for _ in 0..64 {
            assert!(ring.try_enqueue(super::span_sentinel()).is_ok());
        }
        let pipe = into_telemetry_handle(crate::pipes::NullPipe::new());
        let dyn_pipe: &dyn proxima_primitives::pipe::alloc_tier::SendDynPipe<
            TelemetryRequest,
            proxima_primitives::pipe::request::Response<bytes::Bytes>,
        > = pipe.as_ref();

        // ring holds 64; one assist with assist_batch = 8 drains exactly 8.
        let freed = super::drain_export_spans(&ring, dyn_pipe, RecordSharing::Inline, 8);
        assert_eq!(
            freed, 8,
            "a single assist frees exactly assist_batch, not the whole ring"
        );
        assert_eq!(
            ring.len(),
            56,
            "the rest stays buffered for the background drainer"
        );
    }

    #[test]
    fn drainer_with_inline_sharing_emits_inline_batch() {
        let method_tracker = Arc::new(parking_lot::Mutex::new(
            alloc::vec::Vec::<bytes::Bytes>::new(),
        ));
        let tracker_clone = Arc::clone(&method_tracker);

        struct MethodCapturePipe {
            methods: Arc<parking_lot::Mutex<alloc::vec::Vec<bytes::Bytes>>>,
        }

        impl proxima_primitives::pipe::SendPipe for MethodCapturePipe {
            type In = TelemetryRequest;
            type Out = proxima_primitives::pipe::request::Response<bytes::Bytes>;
            type Err = proxima_primitives::pipe::ProximaError;

            fn call(
                &self,
                request: TelemetryRequest,
            ) -> impl std::future::Future<
                Output = Result<
                    proxima_primitives::pipe::request::Response<bytes::Bytes>,
                    proxima_primitives::pipe::ProximaError,
                >,
            > + Send {
                self.methods.lock().push(request.method.to_bytes());
                async move { Ok(proxima_primitives::pipe::request::Response::ok(bytes::Bytes::new())) }
            }
        }

        let pipe = MethodCapturePipe {
            methods: tracker_clone,
        };
        let (drainer, cores) = make_drainer_with(pipe, RecordSharing::Inline);

        let ring_set = cores.cores().slot(0);
        for _ in 0..5 {
            let _ = ring_set.spans.enqueue(crate::trace::SpanRecord {
                trace_id: crate::id::TraceId::INVALID,
                span_id: crate::id::SpanId::INVALID,
                parent_span_id: None,
                name: "test",
                kind: crate::trace::SpanKind::Internal,
                start_ns: 0,
                duration_ns: 0,
                status: crate::trace::Status::Unset,
                attrs: smallvec::smallvec![],
                events: smallvec::smallvec![],
                links: smallvec::smallvec![],
                tracestate: crate::trace::TraceState::empty(),
                module_path: "",
                file_line: (0, 0),
            });
        }

        drainer.drain_pass();

        let methods = method_tracker.lock();
        assert!(
            methods
                .iter()
                .any(|method| method.as_ref() == METHOD_SPAN_BATCH),
            "expected METHOD_SPAN_BATCH for Inline sharing, got: {:?}",
            methods
        );
        assert!(
            !methods
                .iter()
                .any(|method| method.as_ref() == METHOD_SPAN_BATCH_ARC),
            "did not expect METHOD_SPAN_BATCH_ARC for Inline sharing"
        );
    }

    #[test]
    fn drainer_with_arc_sharing_emits_arc_batch() {
        let method_tracker = Arc::new(parking_lot::Mutex::new(
            alloc::vec::Vec::<bytes::Bytes>::new(),
        ));
        let tracker_clone = Arc::clone(&method_tracker);

        struct MethodCapturePipe {
            methods: Arc<parking_lot::Mutex<alloc::vec::Vec<bytes::Bytes>>>,
        }

        impl proxima_primitives::pipe::SendPipe for MethodCapturePipe {
            type In = TelemetryRequest;
            type Out = proxima_primitives::pipe::request::Response<bytes::Bytes>;
            type Err = proxima_primitives::pipe::ProximaError;

            fn call(
                &self,
                request: TelemetryRequest,
            ) -> impl std::future::Future<
                Output = Result<
                    proxima_primitives::pipe::request::Response<bytes::Bytes>,
                    proxima_primitives::pipe::ProximaError,
                >,
            > + Send {
                self.methods.lock().push(request.method.to_bytes());
                async move { Ok(proxima_primitives::pipe::request::Response::ok(bytes::Bytes::new())) }
            }
        }

        let pipe = MethodCapturePipe {
            methods: tracker_clone,
        };
        let (drainer, cores) = make_drainer_with(pipe, RecordSharing::Arc);

        let ring_set = cores.cores().slot(0);
        for _ in 0..10 {
            let _ = ring_set.spans.enqueue(crate::trace::SpanRecord {
                trace_id: crate::id::TraceId::INVALID,
                span_id: crate::id::SpanId::INVALID,
                parent_span_id: None,
                name: "test",
                kind: crate::trace::SpanKind::Internal,
                start_ns: 0,
                duration_ns: 0,
                status: crate::trace::Status::Unset,
                attrs: smallvec::smallvec![],
                events: smallvec::smallvec![],
                links: smallvec::smallvec![],
                tracestate: crate::trace::TraceState::empty(),
                module_path: "",
                file_line: (0, 0),
            });
        }

        drainer.drain_pass();

        let methods = method_tracker.lock();
        assert!(
            methods
                .iter()
                .any(|method| method.as_ref() == METHOD_SPAN_BATCH_ARC),
            "expected METHOD_SPAN_BATCH_ARC for Arc sharing, got: {:?}",
            methods
        );
        assert!(
            !methods
                .iter()
                .any(|method| method.as_ref() == METHOD_SPAN_BATCH),
            "did not expect METHOD_SPAN_BATCH for Arc sharing"
        );
    }
}
