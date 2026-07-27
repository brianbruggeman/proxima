//! A resilient OTLP terminal sink: owns its own bounded buffer and a
//! detached background sender, so a collector outage never stalls the
//! shared drain (`crate::recorder::drainer`) that all other sinks share.
//!
//! `Pipe::call` only enqueues (a short mutex section, no I/O) and returns
//! immediately — the network send, retry/backoff, and reconnect all happen
//! on a dedicated background thread. See `docs` on [`ResilientSink`] for the
//! full contract.
//!
//! Composes onto ANY [`TelemetryPipeHandle`]-producing factory — the OTLP/
//! HTTP codec (`crate::out::otlp_http`) and OTLP/gRPC pipe (`crate::out::
//! otlp_grpc`) both qualify, so this one sink covers both transports.

mod config;
mod queue;
mod worker;

pub use config::{ResilientOtlpConfig, RetentionHorizons};

use alloc::sync::Arc;
use core::future::Future as StdFuture;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use bytes::Bytes;
use parking_lot::RwLock;
use proxima_primitives::pipe::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::capabilities::Clock;
use proxima_primitives::pipe::clock::TimeClock;
use proxima_primitives::pipe::request::Response;
use proxima_primitives::sync::Notify;

use crate::level::Level;
use crate::pipes::{TelemetryPipeHandle, TelemetryRecord, TelemetryRequest};
use crate::recorder::Recorder;

use queue::{QueuedRecord, SeverityBucketedQueue};

/// The state shared between [`ResilientSink::call`] (the producer, on the
/// shared drain) and the background worker (the sole consumer). Everything
/// here is cheap/synchronous to touch from the producer side; anything that
/// can block on I/O lives only in `worker`.
pub(crate) struct Shared<Clk> {
    queue: SeverityBucketedQueue,
    factory: alloc::boxed::Box<dyn Fn() -> TelemetryPipeHandle + Send + Sync>,
    current: RwLock<TelemetryPipeHandle>,
    clock: Clk,
    config: ResilientOtlpConfig,
    notify: Notify,
    shutdown: AtomicBool,
    last_tick_nanos: AtomicU64,
    since_tick_drops: [AtomicU64; queue::SEVERITY_BUCKETS],
    counters: worker::SelfCounters,
    /// Explicit self-instrumentation target — the `recorder = rec` seam this
    /// crate's own log/metric macros use (`crate::export::default_recorder`
    /// is the ambient fallback). Explicit by default here too: two sinks (or
    /// two tests) built with different recorders must never cross-report.
    self_recorder: Option<Arc<Recorder>>,
}

impl<Clk> Shared<Clk> {
    /// The recorder self-instrumentation reports through: the explicit one
    /// this sink was built with, else whatever is process-default. The two
    /// candidates are different concrete `Recorder<_>` types (an explicit
    /// recorder is the ordinary `SystemClock`; the ambient default is
    /// `GlobalClock` — see `export::default_recorder`), so the explicit one
    /// is rewrapped via [`Recorder::to_global`] to unify on one return type.
    fn self_recorder(&self) -> Option<Arc<Recorder<crate::clock::GlobalClock>>> {
        match &self.self_recorder {
            Some(recorder) => Some(Arc::new(recorder.to_global())),
            None => crate::export::default_recorder(),
        }
    }
}

/// A resilient OTLP terminal — bounded buffer, capped-backoff reconnecting
/// sender, severity/age-horizon shedding under sustained pressure, and
/// self-exported drop/retry/reconnect/backlog telemetry. Operator-facing
/// guide: `ai_docs/projections/otlp-resilient-sink.md`.
///
/// Composes into any fan-out like every other terminal sink — the workspace
/// rule is export is never OTLP-only, so a real deployment pairs this with a
/// local floor sink via [`crate::pipes::fan_exporters`] (`collector` below
/// stands in for a real OTLP transport factory, e.g. `crate::out::otlp_http`;
/// any `SendPipe` factory works, the sink is transport-agnostic):
///
/// ```
/// use proxima_telemetry::export::Exporter;
/// use proxima_telemetry::level::Level;
/// use proxima_telemetry::out::resilient::{ResilientOtlpConfig, ResilientSink};
/// use proxima_telemetry::pipes::{
///     FormatterPipe, InMemoryPipe, LogFormat, fan_exporters, into_telemetry_handle,
/// };
/// use proxima_telemetry::recorder::Recorder;
///
/// let collector = InMemoryPipe::new();
/// let factory_collector = collector.clone();
/// let resilient = into_telemetry_handle(ResilientSink::spawn(
///     move || into_telemetry_handle(factory_collector.clone()),
///     ResilientOtlpConfig::default(),
/// ));
/// let console = into_telemetry_handle(FormatterPipe::new(std::io::stderr(), LogFormat::Human));
/// let fanned = fan_exporters(vec![console, resilient]);
///
/// let recorder = Recorder::builder()
///     .export(Exporter::pipe(fanned))
///     .expect("compose console + resilient fan")
///     .core_count(1)
///     .start()
///     .expect("start recorder");
///
/// recorder.log().level(Level::ERROR).message("hello").emit();
/// recorder.drain(); // enqueues into the resilient sink's own buffer; the
///                    // background worker (not this call) does the send.
/// ```
///
/// # Contract
/// - `Pipe::call` never waits on the network: it enqueues and returns.
/// - The background sender retries forever (no terminal give-up state);
///   backoff is exponential with jitter, capped (`backoff_cap_ms`, default
///   30s) so a recovered collector is noticed within about one interval.
/// - Any transport error triggers a reconnect (the downstream handle is
///   rebuilt via `factory`) before the next attempt — a stale channel is
///   never retried forever.
/// - The buffer sheds by severity + age horizon before it drops anything
///   from higher lanes; only once even the error lane is over its horizon
///   does space pressure evict it. See [`RetentionHorizons`].
/// - At-least-once: a retry after a send that actually landed (but whose
///   response was lost/timed out) can duplicate. No dedup is performed.
pub struct ResilientSink<Clk = TimeClock> {
    shared: Arc<Shared<Clk>>,
}

impl ResilientSink<TimeClock> {
    /// Build and spawn the background sender on the production clock.
    /// `factory` builds (or rebuilds, on reconnect) the downstream transport
    /// pipe — e.g. `move || into_telemetry_handle(OtlpHttpCodec::new(client))`.
    #[must_use]
    pub fn spawn(
        factory: impl Fn() -> TelemetryPipeHandle + Send + Sync + 'static,
        config: ResilientOtlpConfig,
    ) -> Self {
        Self::spawn_with(factory, config, TimeClock, None)
    }

    /// [`Self::spawn`], self-instrumenting into `recorder` explicitly instead
    /// of the process-default — the seam this crate's own macros use
    /// (`recorder = rec`), so composing this sink into a non-default
    /// [`Recorder`] still gets its counters/logs where they belong.
    #[must_use]
    pub fn spawn_with_recorder(
        factory: impl Fn() -> TelemetryPipeHandle + Send + Sync + 'static,
        config: ResilientOtlpConfig,
        recorder: Arc<Recorder>,
    ) -> Self {
        Self::spawn_with(factory, config, TimeClock, Some(recorder))
    }
}

impl<Clk> ResilientSink<Clk>
where
    Clk: Clock + Send + Sync + 'static,
{
    /// Build and spawn the background sender on an injected clock — the
    /// deterministic-test seam (a fake clock makes backoff assertable
    /// without a real sleep). The worker thread is supervised: a panic
    /// inside one send iteration is caught and counted, never lets the
    /// thread exit (see `worker::run`).
    #[must_use]
    pub fn spawn_with_clock(
        factory: impl Fn() -> TelemetryPipeHandle + Send + Sync + 'static,
        config: ResilientOtlpConfig,
        clock: Clk,
    ) -> Self {
        Self::spawn_with(factory, config, clock, None)
    }

    #[must_use]
    pub fn spawn_with(
        factory: impl Fn() -> TelemetryPipeHandle + Send + Sync + 'static,
        config: ResilientOtlpConfig,
        clock: Clk,
        self_recorder: Option<Arc<Recorder>>,
    ) -> Self {
        let initial = factory();
        let shared = Arc::new(Shared {
            queue: SeverityBucketedQueue::new(config.buffer_capacity, config.horizons),
            factory: alloc::boxed::Box::new(factory),
            current: RwLock::new(initial),
            clock,
            config,
            notify: Notify::new(),
            shutdown: AtomicBool::new(false),
            last_tick_nanos: AtomicU64::new(0),
            since_tick_drops: core::array::from_fn(|_| AtomicU64::new(0)),
            counters: worker::SelfCounters::default(),
            self_recorder,
        });
        let background = Arc::clone(&shared);
        let spawned = std::thread::Builder::new()
            .name("proxima-otlp-resilient".into())
            .spawn(move || worker::run(background));
        // a spawn failure here means the process is out of OS threads —
        // already in serious trouble. Fall back to leaving the sink
        // enqueue-only (degrade honestly per the no-runtime-tiers rule)
        // rather than panicking the caller's install path.
        if spawned.is_err()
            && let Some(recorder) = shared.self_recorder()
        {
            recorder
                .log()
                .level(Level::ERROR)
                .message(
                    "proxima-otlp-resilient worker thread failed to spawn; sink is enqueue-only",
                )
                .emit();
        }
        Self { shared }
    }

    /// Snapshot of lifetime counters — the test/diagnostic surface that does
    /// not depend on an installed ambient recorder.
    #[must_use]
    pub fn stats(&self) -> Stats {
        Stats {
            sent: self.shared.counters.sent_lifetime.load(Ordering::Relaxed),
            retried: self
                .shared
                .counters
                .retried_lifetime
                .load(Ordering::Relaxed),
            reconnected: self
                .shared
                .counters
                .reconnected_lifetime
                .load(Ordering::Relaxed),
            panics_recovered: self.shared.counters.panics.load(Ordering::Relaxed),
            backlog_depth: self.shared.queue.total_len(),
            dropped_by_bucket: core::array::from_fn(|bucket| {
                self.shared.queue.dropped_total(bucket)
            }),
        }
    }

    /// Signal the background worker to stop after its current iteration.
    /// Does not wait for buffered records to flush (there is no terminal
    /// "flush and stop" here — an outage-surviving sink has no drain-to-zero
    /// guarantee by design; call this at process shutdown, not mid-outage).
    pub fn shutdown(&self) {
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.notify.notify_waiters();
    }
}

/// Lifetime counters, for assertions that don't want to depend on the
/// ambient recorder being installed.
#[derive(Debug, Clone, Copy, Default)]
pub struct Stats {
    pub sent: u64,
    pub retried: u64,
    pub reconnected: u64,
    pub panics_recovered: u64,
    pub backlog_depth: usize,
    pub dropped_by_bucket: [u64; queue::SEVERITY_BUCKETS],
}

impl<Clk> SendPipe for ResilientSink<Clk>
where
    Clk: Clock + Send + Sync + 'static,
{
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: TelemetryRequest,
    ) -> impl StdFuture<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let now_nanos = self.shared.clock.now_nanos();
        enqueue(&self.shared, request, now_nanos);
        self.shared.notify.notify_one();
        async move { Ok(Response::ok(Bytes::new())) }
    }
}

/// Destructure one drained `TelemetryRequest` into individual queue entries
/// so severity/age eviction (and later, re-batching) operates at record
/// granularity rather than whatever batch size the drain happened to use.
fn enqueue<Clk>(shared: &Shared<Clk>, request: TelemetryRequest, now_nanos: u64) {
    let push_one = |record: QueuedRecord| {
        if let Some(event) = shared.queue.push(record, now_nanos) {
            shared.since_tick_drops[event.bucket].fetch_add(1, Ordering::Relaxed);
        }
    };
    match request.payload {
        TelemetryRecord::Span(record) => push_one(QueuedRecord::Span(Arc::new(record))),
        TelemetryRecord::Event(record) => push_one(QueuedRecord::Event(Arc::new(record))),
        TelemetryRecord::Log(record) => push_one(QueuedRecord::Log(Arc::new(record))),
        TelemetryRecord::Metric(record) => push_one(QueuedRecord::Metric(Arc::new(record))),
        TelemetryRecord::Link(record) => push_one(QueuedRecord::Link(Arc::new(record))),
        TelemetryRecord::SpanBatch(records) => {
            for record in records {
                push_one(QueuedRecord::Span(Arc::new(record)));
            }
        }
        TelemetryRecord::EventBatch(records) => {
            for record in records {
                push_one(QueuedRecord::Event(Arc::new(record)));
            }
        }
        TelemetryRecord::LogBatch(records) => {
            for record in records {
                push_one(QueuedRecord::Log(Arc::new(record)));
            }
        }
        TelemetryRecord::MetricBatch(records) => {
            for record in records {
                push_one(QueuedRecord::Metric(Arc::new(record)));
            }
        }
        TelemetryRecord::LinkBatch(records) => {
            for record in records {
                push_one(QueuedRecord::Link(Arc::new(record)));
            }
        }
        TelemetryRecord::SpanBatchArc(records) => {
            for record in records {
                push_one(QueuedRecord::Span(record));
            }
        }
        TelemetryRecord::EventBatchArc(records) => {
            for record in records {
                push_one(QueuedRecord::Event(record));
            }
        }
        TelemetryRecord::LogBatchArc(records) => {
            for record in records {
                push_one(QueuedRecord::Log(record));
            }
        }
        TelemetryRecord::MetricBatchArc(records) => {
            for record in records {
                push_one(QueuedRecord::Metric(record));
            }
        }
        TelemetryRecord::LinkBatchArc(records) => {
            for record in records {
                push_one(QueuedRecord::Link(record));
            }
        }
    }
}

#[cfg(test)]
mod tests;
