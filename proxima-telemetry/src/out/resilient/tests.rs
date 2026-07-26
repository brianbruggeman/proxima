#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_lines)]

use alloc::sync::Arc;
use core::future::Future;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use core::task::{Context, Poll, Waker};
use core::time::Duration;
use std::sync::Mutex;
use std::time::Instant;

use bytes::Bytes;
use proxima_primitives::pipe::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::capabilities::Clock;
use proxima_primitives::pipe::request::Response;

use crate::id::{SpanId, TraceId};
use crate::level::Level;
use crate::log::LogRecord;
use crate::log::body::LogBody;
use crate::pipes::{
    InMemoryPipe, TelemetryPipeHandle, TelemetryRequest, into_telemetry_handle, log_batch_request,
    log_request,
};
use crate::trace::span::SpanRecord;
use crate::trace::status::Status;

use super::config::ResilientOtlpConfig;
use super::queue::{
    DropReason, QueuedRecord, SeverityBucketedQueue, ingest_rates, survivable_seconds,
};
use super::worker::{self, SendCursor};
use super::{ResilientSink, Shared};

// ── test fixtures ───────────────────────────────────────────────────────────

fn log_record(level: Level, text: &'static str) -> LogRecord {
    LogRecord {
        ts_ns: 0,
        observed_ts_ns: 0,
        level,
        body: LogBody::Text(text),
        attrs: smallvec::SmallVec::new(),
        trace_id: None,
        span_id: None,
        trace_flags: crate::id::TraceFlags::NOT_SAMPLED,
        module_path: "test",
        file_line: (0, 0),
    }
}

fn span_record(status: Status) -> SpanRecord {
    SpanRecord {
        trace_id: TraceId::from_bytes([1u8; 16]),
        span_id: SpanId::from_bytes([2u8; 8]),
        parent_span_id: None,
        name: "test.span",
        kind: crate::trace::kind::SpanKind::Internal,
        start_ns: 0,
        duration_ns: 0,
        status,
        attrs: smallvec::SmallVec::new(),
        events: smallvec::SmallVec::new(),
        links: smallvec::SmallVec::new(),
        tracestate: crate::trace::tracestate::TraceState::empty(),
        module_path: "test",
        file_line: (0, 0),
    }
}

/// Deterministic, manually-advanced clock for step()-driven tests. `delay`
/// never actually waits — it records what was requested (for backoff
/// assertions) and resolves immediately, matching the `FakeClock` pattern
/// already used in `proxima-primitives`' retry tests.
#[derive(Clone, Default)]
struct FakeClock {
    now_nanos: Arc<AtomicU64>,
    delays: Arc<Mutex<alloc::vec::Vec<Duration>>>,
}

impl FakeClock {
    fn advance(&self, duration: Duration) {
        self.now_nanos
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
    }
}

impl Clock for FakeClock {
    type Delay = core::future::Ready<()>;

    fn now_nanos(&self) -> u64 {
        self.now_nanos.load(Ordering::Relaxed)
    }

    // `step()` never calls `Clock::delay` (backoff is a pure `Duration`
    // computation read off the cursor); this only matters to `worker::run`'s
    // idle-wait, which the step()-driven tests below bypass entirely. Kept
    // real (not `unimplemented!`) so this clock stays usable if a future
    // test drives `run()` directly.
    fn delay(&self, duration: Duration) -> Self::Delay {
        self.delays.lock().unwrap().push(duration);
        core::future::ready(())
    }
}

/// A fake OTLP collector: refuses (returns a transport `Err`) the first
/// `refuse_count` calls, then accepts everything after — the "collector down
/// for N attempts then recovers" fixture the failure-injection tests need.
/// Accepted batches are captured in an `InMemoryPipe` so tests can assert on
/// exactly what got through. `factory_calls` counts how many times the
/// resilient sink's `factory` closure rebuilt a handle onto this collector —
/// the reconnect-count signal.
#[derive(Clone)]
struct FlakyCollector {
    calls: Arc<AtomicUsize>,
    refuse_count: Arc<AtomicUsize>,
    received: InMemoryPipe,
}

impl FlakyCollector {
    fn new(refuse_count: usize) -> Self {
        Self {
            calls: Arc::new(AtomicUsize::new(0)),
            refuse_count: Arc::new(AtomicUsize::new(refuse_count)),
            received: InMemoryPipe::new(),
        }
    }

    fn always_down() -> Self {
        Self::new(usize::MAX)
    }
}

impl SendPipe for FlakyCollector {
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: TelemetryRequest,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let call_index = self.calls.fetch_add(1, Ordering::SeqCst);
        let refuse_count = self.refuse_count.load(Ordering::SeqCst);
        let received = self.received.clone();
        async move {
            if call_index < refuse_count {
                Err(ProximaError::Io(std::io::Error::new(
                    std::io::ErrorKind::ConnectionRefused,
                    "collector down",
                )))
            } else {
                SendPipe::call(&received, request).await
            }
        }
    }
}

/// Build a `ResilientSink`'s shared state for direct `step()`-driving: no
/// background thread, fully deterministic under `FakeClock`. `self_recorder`
/// is the explicit seam — pass one to assert on self-exported counters/logs
/// without touching the process-global ambient recorder.
fn build_shared(
    collector: FlakyCollector,
    config: ResilientOtlpConfig,
    clock: FakeClock,
    self_recorder: Option<Arc<crate::recorder::Recorder>>,
) -> Arc<Shared<FakeClock>> {
    let factory_collector = collector;
    let factory =
        move || -> TelemetryPipeHandle { into_telemetry_handle(factory_collector.clone()) };
    let initial = factory();
    Arc::new(Shared {
        queue: SeverityBucketedQueue::new(config.buffer_capacity, config.horizons),
        factory: alloc::boxed::Box::new(factory),
        current: parking_lot::RwLock::new(initial),
        clock,
        config,
        notify: proxima_primitives::sync::Notify::new(),
        shutdown: core::sync::atomic::AtomicBool::new(false),
        last_tick_nanos: AtomicU64::new(0),
        since_tick_drops: core::array::from_fn(|_| AtomicU64::new(0)),
        counters: worker::SelfCounters::default(),
        self_recorder,
    })
}

fn build_sink(
    collector: FlakyCollector,
    config: ResilientOtlpConfig,
    clock: FakeClock,
) -> Arc<Shared<FakeClock>> {
    build_shared(collector, config, clock, None)
}

fn small_config() -> ResilientOtlpConfig {
    ResilientOtlpConfig::builder()
        .buffer_capacity(8)
        .max_batch_items(4)
        .max_batch_bytes(1_000_000)
        .backoff_base_ms(10)
        .backoff_cap_ms(100)
        .drop_announce_interval_ms(50)
        .idle_poll_ms(10)
        .build()
}

#[test]
fn span_severity_bucket_follows_terminal_status_not_a_fixed_default() {
    let ok_span = QueuedRecord::Span(Arc::new(span_record(Status::Ok)));
    let error_span = QueuedRecord::Span(Arc::new(span_record(Status::Error { reason: "boom" })));
    assert_eq!(
        ok_span.severity_bucket(),
        2,
        "a healthy span defaults to the info lane"
    );
    assert_eq!(
        error_span.severity_bucket(),
        4,
        "an errored span is as valuable as an error log"
    );
}

// ── 1. the shared drain never waits ────────────────────────────────────────

#[test]
fn call_resolves_ready_on_first_poll_never_pending() {
    // this is the exact optimization `drainer::call_pipe` relies on (poll
    // once with a noop waker; only a genuinely-pending future falls back to
    // `block_on`). Proving `call` is always `Ready` on first poll proves the
    // shared drain thread can NEVER end up blocked on this sink, regardless
    // of what the background worker is doing.
    let collector = FlakyCollector::always_down();
    let shared = build_sink(collector, small_config(), FakeClock::default());
    let sink = ResilientSink { shared };

    let future = SendPipe::call(
        &sink,
        log_batch_request(alloc::vec![log_record(Level::ERROR, "x")]),
    );
    let mut future = core::pin::pin!(future);
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let result = future.as_mut().poll(&mut context);
    assert!(
        matches!(result, Poll::Ready(Ok(_))),
        "call() must resolve on the first poll"
    );
}

// ── 2. backoff grows, is jittered, and is capped ───────────────────────────

#[test]
fn backoff_grows_is_jittered_and_caps_at_configured_ceiling() {
    let collector = FlakyCollector::always_down();
    let clock = FakeClock::default();
    let config = ResilientOtlpConfig::builder()
        .buffer_capacity(64)
        .max_batch_items(16)
        .max_batch_bytes(1_000_000)
        .backoff_base_ms(100)
        .backoff_cap_ms(1_000)
        .drop_announce_interval_ms(100_000)
        .idle_poll_ms(10)
        .build();
    let shared = build_sink(collector, config, clock.clone());
    let mut cursor = SendCursor::default();

    shared.queue.push(
        QueuedRecord::Log(Arc::new(log_record(Level::ERROR, "x"))),
        clock.now_nanos(),
    );

    let mut scheduled = alloc::vec::Vec::new();
    for _ in 0..12 {
        let before = clock.now_nanos();
        worker::step(&shared, &mut cursor);
        let scheduled_delay =
            Duration::from_nanos(cursor.next_attempt_at_nanos.saturating_sub(before));
        scheduled.push(scheduled_delay);
        // advance past whatever was scheduled so the next step() actually attempts
        clock.advance(scheduled_delay + Duration::from_millis(1));
    }

    assert!(
        scheduled.windows(2).any(|pair| pair[1] > pair[0]),
        "delay must grow across early attempts: {scheduled:?}"
    );
    assert!(
        scheduled
            .iter()
            .all(|delay| *delay <= Duration::from_millis(1_000)),
        "no delay may exceed the configured cap: {scheduled:?}"
    );
    assert!(
        scheduled
            .iter()
            .skip(6)
            .all(|delay| *delay <= Duration::from_millis(1_000)),
        "later attempts must stay at/under the cap forever: {scheduled:?}"
    );
    assert!(
        scheduled
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            > 1,
        "jitter must vary the delay, not repeat a fixed sleep: {scheduled:?}"
    );
}

// ── no terminal give-up state ───────────────────────────────────────────────

#[test]
fn retries_never_reach_a_terminal_give_up_state() {
    let collector = FlakyCollector::always_down();
    let clock = FakeClock::default();
    let shared = build_sink(collector, small_config(), clock.clone());
    let mut cursor = SendCursor::default();
    shared.queue.push(
        QueuedRecord::Log(Arc::new(log_record(Level::ERROR, "x"))),
        clock.now_nanos(),
    );

    // far beyond any conventional `max_attempts` (3-10) — the batch must
    // still be actively retried, never discarded into a dead/poisoned state.
    for _ in 0..200 {
        let before = clock.now_nanos();
        worker::step(&shared, &mut cursor);
        let scheduled_delay =
            Duration::from_nanos(cursor.next_attempt_at_nanos.saturating_sub(before));
        clock.advance(scheduled_delay + Duration::from_millis(1));
    }
    assert!(
        shared.queue.dropped_total(4) == 0,
        "an always-retryable batch is never counted as dropped"
    );
}

// ── 3. backlog exceeds max batch size -> flush re-batches ──────────────────

#[test]
fn backlog_larger_than_one_batch_flushes_as_multiple_correctly_sized_chunks() {
    let collector = FlakyCollector::new(0); // never refuses
    let clock = FakeClock::default();
    let config = ResilientOtlpConfig::builder()
        .buffer_capacity(1_000)
        .max_batch_items(3)
        .max_batch_bytes(1_000_000)
        .backoff_base_ms(10)
        .backoff_cap_ms(100)
        .drop_announce_interval_ms(100_000)
        .idle_poll_ms(10)
        .build();
    let received = collector.received.clone();
    let shared = build_sink(collector, config, clock.clone());

    for index in 0..10u32 {
        shared.queue.push(
            QueuedRecord::Log(Arc::new(log_record(Level::INFO, "x"))),
            clock.now_nanos(),
        );
        let _ = index;
    }

    let mut cursor = SendCursor::default();
    // 10 records, 3 per batch -> 4 sends (3,3,3,1); step() advances one send
    // per call once it is due (immediate here — collector never refuses).
    for _ in 0..4 {
        worker::step(&shared, &mut cursor);
    }

    assert_eq!(
        received.logs().len(),
        10,
        "all 10 records eventually arrive"
    );
    // re-derive the batch boundaries via the drain's own dispatch call count:
    // each of the 4 step() iterations sent at most max_batch_items.
    assert_eq!(shared.queue.total_len(), 0, "backlog fully flushed");
}

#[test]
fn backlog_flush_respects_the_byte_ceiling_not_just_item_count() {
    let collector = FlakyCollector::new(0);
    let clock = FakeClock::default();
    // size the byte ceiling from a real encoded record so the test isn't
    // guessing at prost's wire overhead.
    let sample = QueuedRecord::Log(Arc::new(log_record(Level::INFO, "payload-of-known-size")));
    let one_record_bytes = sample.encoded_len();
    let max_batch_bytes = one_record_bytes * 2 + 1; // room for ~2 records per batch

    let config = ResilientOtlpConfig::builder()
        .buffer_capacity(1_000)
        .max_batch_items(1_000)
        .max_batch_bytes(max_batch_bytes)
        .backoff_base_ms(10)
        .backoff_cap_ms(100)
        .drop_announce_interval_ms(100_000)
        .idle_poll_ms(10)
        .build();
    let received = collector.received.clone();
    let shared = build_sink(collector, config, clock.clone());

    for _ in 0..9 {
        shared.queue.push(
            QueuedRecord::Log(Arc::new(log_record(Level::INFO, "payload-of-known-size"))),
            clock.now_nanos(),
        );
    }

    let mut cursor = SendCursor::default();
    let mut batch_sizes = alloc::vec::Vec::new();
    while shared.queue.total_len() > 0 {
        let before = received.logs().len();
        worker::step(&shared, &mut cursor);
        let after = received.logs().len();
        if after > before {
            batch_sizes.push(after - before);
        }
    }

    assert_eq!(
        batch_sizes.iter().sum::<usize>(),
        9,
        "no record lost across chunking"
    );
    assert!(
        batch_sizes.len() > 1,
        "9 records at ~2/batch must span multiple batches"
    );
    for size in &batch_sizes {
        assert!(
            *size <= 2,
            "chunk of {size} exceeds the byte-derived cap of ~2 records"
        );
    }
}

// ── 4. buffer exhaustion: drop-oldest, counted, newest retained ───────────

#[test]
fn buffer_exhaustion_drops_oldest_lowest_severity_first_and_counts_it() {
    let clock = FakeClock::default();
    let queue = SeverityBucketedQueue::new(4, super::config::RetentionHorizons::default());

    // D1 D2 E1 E2 fill capacity(4); D3 must evict D1 (oldest debug), not an error.
    queue.push(
        QueuedRecord::Log(Arc::new(log_record(Level::DEBUG, "d1"))),
        0,
    );
    queue.push(
        QueuedRecord::Log(Arc::new(log_record(Level::DEBUG, "d2"))),
        0,
    );
    queue.push(
        QueuedRecord::Log(Arc::new(log_record(Level::ERROR, "e1"))),
        0,
    );
    queue.push(
        QueuedRecord::Log(Arc::new(log_record(Level::ERROR, "e2"))),
        0,
    );
    let drop = queue.push(
        QueuedRecord::Log(Arc::new(log_record(Level::DEBUG, "d3"))),
        0,
    );

    assert!(drop.is_some_and(|event| event.bucket == 1 && event.reason == DropReason::Space));
    assert_eq!(queue.dropped_total(1), 1, "debug drop counted");
    assert_eq!(queue.dropped_total(4), 0, "no error was dropped");
    assert_eq!(queue.total_len(), 4, "capacity never exceeded");

    // newest (d3) retained: drain debug lane and confirm d1 is gone, d2/d3 remain.
    let remaining = queue.lane_lens();
    assert_eq!(remaining[1], 2, "d2 and d3 both survive in the debug lane");
    let _ = clock;
}

#[test]
fn buffer_never_grows_past_capacity_under_sustained_overflow() {
    let queue = SeverityBucketedQueue::new(16, super::config::RetentionHorizons::default());
    for index in 0..10_000u32 {
        queue.push(
            QueuedRecord::Log(Arc::new(log_record(Level::INFO, "x"))),
            u64::from(index),
        );
    }
    assert_eq!(queue.total_len(), 16, "bounded regardless of ingest volume");
    assert!(queue.dropped_total(2) >= 10_000 - 16);
}

// ── 6/8. severity shedding under space pressure ────────────────────────────

#[test]
fn space_pressure_evicts_by_severity_before_touching_higher_lanes() {
    let queue = SeverityBucketedQueue::new(5, super::config::RetentionHorizons::default());
    // fill fast, same instant (now=0 for every push) -- nothing has aged, so
    // any eviction observed here is proven to be the SPACE path, not horizon.
    for _ in 0..2 {
        queue.push(
            QueuedRecord::Log(Arc::new(log_record(Level::TRACE, "t"))),
            0,
        );
    }
    for _ in 0..2 {
        queue.push(
            QueuedRecord::Log(Arc::new(log_record(Level::DEBUG, "d"))),
            0,
        );
    }
    for _ in 0..2 {
        queue.push(
            QueuedRecord::Log(Arc::new(log_record(Level::ERROR, "e"))),
            0,
        );
    }
    // 6 pushed against capacity 5: exactly one eviction, and it must be trace
    // (the shortest-horizon lane), never error.
    let lens = queue.lane_lens();
    assert_eq!(lens[0], 1, "one trace evicted, one remains");
    assert_eq!(lens[1], 2, "debug untouched");
    assert_eq!(lens[4], 2, "error untouched");
    assert_eq!(queue.dropped_total(0), 1);
    assert_eq!(
        queue.dropped_total(4),
        0,
        "space pressure never takes error while trace/debug still hold anything"
    );
}

// ── 7. horizon eviction ladder ──────────────────────────────────────────────

#[test]
fn horizon_sweep_evicts_in_the_configured_per_severity_ladder() {
    let horizons = super::config::RetentionHorizons::builder()
        .trace_secs(10)
        .debug_secs(20)
        .info_secs(30)
        .warn_secs(35)
        .error_secs(40)
        .build();
    let queue = SeverityBucketedQueue::new(1_000, horizons);
    for bucket_level in [
        Level::TRACE,
        Level::DEBUG,
        Level::INFO,
        Level::WARN,
        Level::ERROR,
    ] {
        queue.push(
            QueuedRecord::Log(Arc::new(log_record(bucket_level, "x"))),
            0,
        );
    }

    // t=11s: only trace (10s horizon) has expired.
    let events = queue.sweep_horizons(Duration::from_secs(11).as_nanos() as u64);
    assert_eq!(events.len(), 1);
    assert_eq!(queue.lane_lens(), [0, 1, 1, 1, 1]);

    // t=21s: debug (20s) now expires too.
    let events = queue.sweep_horizons(Duration::from_secs(21).as_nanos() as u64);
    assert_eq!(events.len(), 1);
    assert_eq!(queue.lane_lens(), [0, 0, 1, 1, 1]);

    // t=31s: info (30s).
    queue.sweep_horizons(Duration::from_secs(31).as_nanos() as u64);
    assert_eq!(queue.lane_lens(), [0, 0, 0, 1, 1]);

    // t=36s: warn (35s). error (40s) survives the longest.
    queue.sweep_horizons(Duration::from_secs(36).as_nanos() as u64);
    assert_eq!(queue.lane_lens(), [0, 0, 0, 0, 1]);

    // t=41s: error finally expires too.
    queue.sweep_horizons(Duration::from_secs(41).as_nanos() as u64);
    assert_eq!(queue.lane_lens(), [0, 0, 0, 0, 0]);
}

// ── 9. horizons are configurable (fluent/config parity) ────────────────────

#[test]
fn retention_horizons_fluent_and_serde_forms_match() {
    let from_builder = super::config::RetentionHorizons::builder()
        .trace_secs(1)
        .debug_secs(2)
        .info_secs(3)
        .warn_secs(4)
        .error_secs(5)
        .build();
    let from_value: super::config::RetentionHorizons = serde_json::from_value(serde_json::json!({
        "trace_secs": 1,
        "debug_secs": 2,
        "info_secs": 3,
        "warn_secs": 4,
        "error_secs": 5,
    }))
    .expect("deserialize");
    assert_eq!(from_builder, from_value);
    for bucket in 0..5 {
        assert_eq!(
            from_builder.for_bucket(bucket),
            from_value.for_bucket(bucket)
        );
    }
}

#[test]
fn custom_horizons_change_when_eviction_actually_happens() {
    let short = super::config::RetentionHorizons::builder()
        .trace_secs(1)
        .build();
    let queue = SeverityBucketedQueue::new(1_000, short);
    queue.push(
        QueuedRecord::Log(Arc::new(log_record(Level::TRACE, "x"))),
        0,
    );
    assert!(
        queue
            .sweep_horizons(Duration::from_millis(500).as_nanos() as u64)
            .is_empty()
    );
    assert_eq!(
        queue
            .sweep_horizons(Duration::from_secs(2).as_nanos() as u64)
            .len(),
        1
    );
}

// ── 11. survivable-window figure ────────────────────────────────────────────

#[test]
fn survivable_seconds_shrinks_as_higher_lanes_occupy_more_capacity() {
    let horizons = super::config::RetentionHorizons::builder()
        .trace_secs(600)
        .build();
    let rates = [1.0, 0.0, 0.0, 0.0, 0.0];

    let roomy = survivable_seconds(&[0, 0, 0, 0, 0], &rates, 100, &horizons, 0);
    let crowded = survivable_seconds(&[0, 0, 0, 0, 50], &rates, 100, &horizons, 0);
    assert!(
        crowded < roomy,
        "error lane eating capacity must shrink trace's survivable window"
    );
    assert_eq!(
        roomy, 100.0,
        "capacity(100)/rate(1) = 100s, under the 600s horizon"
    );
}

#[test]
fn survivable_seconds_shrinks_as_ingest_rate_rises() {
    let horizons = super::config::RetentionHorizons::builder()
        .trace_secs(600)
        .build();
    let slow = survivable_seconds(
        &[0, 0, 0, 0, 0],
        &[1.0, 0.0, 0.0, 0.0, 0.0],
        100,
        &horizons,
        0,
    );
    let fast = survivable_seconds(
        &[0, 0, 0, 0, 0],
        &[10.0, 0.0, 0.0, 0.0, 0.0],
        100,
        &horizons,
        0,
    );
    assert!(
        fast < slow,
        "a higher ingest rate must shrink the survivable window"
    );
}

#[test]
fn ingest_rates_computed_from_counts_and_elapsed_window() {
    let counts = [100, 0, 0, 0, 0];
    let rates = ingest_rates(&counts, Duration::from_secs(10));
    assert!((rates[0] - 10.0).abs() < 0.001);
}

// ── 10. drop announcements are aggregated, not flooded ─────────────────────

#[test]
fn drop_announcements_are_aggregated_across_a_drop_storm() {
    let clock = FakeClock::default();
    let collector = FlakyCollector::always_down();
    let mut config = small_config();
    config.buffer_capacity = 4;
    config.drop_announce_interval_ms = 1_000;

    let floor = InMemoryPipe::new();
    let recorder = Arc::new(
        crate::recorder::Recorder::builder()
            .core_count(1)
            .pipe(floor.clone())
            .start()
            .expect("recorder build"),
    );
    let shared = build_shared(collector, config, clock.clone(), Some(recorder));

    let mut cursor = SendCursor::default();
    let total_pushes = 500u32;
    for index in 0..total_pushes {
        // route through the real enqueue path (not `queue.push` directly) so
        // the per-tick drop tally the announcement reads from is populated
        // exactly as it would be from `Pipe::call`.
        super::enqueue(
            &shared,
            log_request(log_record(Level::DEBUG, "x")),
            u64::from(index),
        );
    }
    // drive several ticks (advancing the clock past the announce interval
    // each time) so multiple announcement windows fire.
    for _ in 0..5 {
        clock.advance(Duration::from_millis(1_100));
        worker::step(&shared, &mut cursor);
    }

    // `recorder.log()/.counter()` only enqueue into the recorder's own ring;
    // draining is what actually delivers them to the floor pipe.
    while shared.self_recorder().expect("recorder installed").drain() > 0 {}
    let announce_lines = floor
        .logs()
        .iter()
        .filter(|log| matches!(log.body, LogBody::Text(text) if text.contains("shedding")))
        .count();
    let total_dropped: u64 = (0..5)
        .map(|bucket| shared.queue.dropped_total(bucket))
        .sum();
    assert!(
        total_dropped > 400,
        "the storm must actually have dropped a lot: {total_dropped}"
    );
    assert!(
        announce_lines > 0 && (announce_lines as u64) < total_dropped / 10,
        "announcements ({announce_lines}) must be far fewer than drops ({total_dropped})"
    );
}

// ── 5. self-exported counters land on a floor sink ──────────────────────────

#[test]
fn self_exported_counters_and_gauges_reach_the_floor_sink() {
    let clock = FakeClock::default();
    let collector = FlakyCollector::always_down();
    let mut config = small_config();
    config.buffer_capacity = 2;
    config.drop_announce_interval_ms = 0;
    let floor = InMemoryPipe::new();
    let recorder = Arc::new(
        crate::recorder::Recorder::builder()
            .core_count(1)
            .pipe(floor.clone())
            .start()
            .expect("recorder build"),
    );
    let shared = build_shared(collector, config, clock.clone(), Some(recorder));

    for index in 0..10u32 {
        super::enqueue(
            &shared,
            log_request(log_record(Level::DEBUG, "x")),
            u64::from(index),
        );
    }
    let mut cursor = SendCursor::default();
    clock.advance(Duration::from_millis(1));
    worker::step(&shared, &mut cursor);

    while shared.self_recorder().expect("recorder installed").drain() > 0 {}
    let metrics = floor.metrics();
    assert!(
        !metrics.is_empty(),
        "self-metrics must land on the floor sink"
    );
    let has_dropped_counter = metrics
        .iter()
        .any(|sample| matches!(sample, crate::metric::MetricSample::Counter(_)));
    assert!(
        has_dropped_counter,
        "expected at least one counter sample among self-metrics"
    );
}

// ── 12. outage -> recovery, isolation proof (real thread, bounded wait) ────

#[test]
fn outage_then_recovery_flushes_backlog_and_floor_sink_never_stalls() {
    let floor = InMemoryPipe::new();
    let recorder = Arc::new(
        crate::recorder::Recorder::builder()
            .core_count(1)
            .pipe(floor.clone())
            .start()
            .expect("recorder build"),
    );

    let refuse_for = 5usize;
    let collector = FlakyCollector::new(refuse_for);
    let received = collector.received.clone();
    let config = ResilientOtlpConfig::builder()
        .buffer_capacity(1_000)
        .max_batch_items(50)
        .max_batch_bytes(1_000_000)
        .backoff_base_ms(5)
        .backoff_cap_ms(20)
        .drop_announce_interval_ms(50)
        .idle_poll_ms(5)
        .build();
    let sink = ResilientSink::spawn_with_recorder(
        move || into_telemetry_handle(collector.clone()),
        config,
        Arc::clone(&recorder),
    );

    let start = Instant::now();
    // hammer the sink AND the floor sink concurrently, the way the shared
    // drain would, while the collector is refusing every attempt.
    for index in 0..200u32 {
        let _ = futures::executor::block_on(SendPipe::call(
            &sink,
            log_request(log_record(Level::INFO, "during-outage")),
        ));
        recorder.log().message("floor-during-outage").emit();
        let _ = index;
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_millis(500),
        "producing 200 records while the collector is down must stay fast: {elapsed:?}"
    );
    recorder.drain();
    assert_eq!(
        floor.logs().len(),
        200,
        "the floor sink kept receiving throughout the outage"
    );

    // wait (bounded, real time — a failure guard, not synchronization) for
    // the backlog to actually flush once the collector recovers.
    let deadline = Instant::now() + Duration::from_secs(5);
    while received.logs().len() < 200 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(
        received.logs().len(),
        200,
        "backlog flushed to the collector after recovery"
    );
    assert!(
        sink.stats().reconnected >= 1,
        "a transport error must have triggered at least one reconnect"
    );
    sink.shutdown();
}

// ── 13. supervised worker: a panic is caught, counted, and recoverable ─────

#[test]
fn worker_iteration_panic_is_caught_and_the_loop_can_continue() {
    struct PanicOnce {
        armed: Arc<std::sync::atomic::AtomicBool>,
    }
    impl SendPipe for PanicOnce {
        type In = TelemetryRequest;
        type Out = Response<Bytes>;
        type Err = ProximaError;
        fn call(
            &self,
            _request: TelemetryRequest,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            if self.armed.swap(false, Ordering::SeqCst) {
                panic!("simulated send-path panic");
            }
            async move { Ok(Response::ok(Bytes::new())) }
        }
    }

    let armed = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let factory_armed = Arc::clone(&armed);
    let clock = FakeClock::default();
    let shared = Arc::new(Shared {
        queue: SeverityBucketedQueue::new(8, super::config::RetentionHorizons::default()),
        factory: alloc::boxed::Box::new(move || {
            into_telemetry_handle(PanicOnce {
                armed: Arc::clone(&factory_armed),
            })
        }),
        current: parking_lot::RwLock::new(into_telemetry_handle(PanicOnce {
            armed: Arc::clone(&armed),
        })),
        clock: clock.clone(),
        config: small_config(),
        notify: proxima_primitives::sync::Notify::new(),
        shutdown: core::sync::atomic::AtomicBool::new(false),
        last_tick_nanos: AtomicU64::new(0),
        since_tick_drops: core::array::from_fn(|_| AtomicU64::new(0)),
        counters: worker::SelfCounters::default(),
        self_recorder: None,
    });
    shared
        .queue
        .push(QueuedRecord::Log(Arc::new(log_record(Level::INFO, "x"))), 0);

    let mut cursor = SendCursor::default();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        worker::step(&shared, &mut cursor);
    }));
    assert!(
        outcome.is_err(),
        "the panic must propagate out of step() so run()'s catch_unwind sees it"
    );

    // the supervisor's contract: the NEXT iteration must still make progress.
    clock.advance(Duration::from_millis(50));
    worker::step(&shared, &mut cursor);
    assert!(
        cursor.current.is_none(),
        "the retried send succeeded on the next iteration"
    );
}

// ── 14. dead channel triggers reconnect, not endless retry on itself ──────

#[test]
fn transport_error_triggers_reconnect_before_the_next_attempt() {
    let clock = FakeClock::default();
    let collector = FlakyCollector::new(3);
    let factory_calls = Arc::new(AtomicUsize::new(0));
    let factory_collector = collector.clone();
    let factory_calls_clone = Arc::clone(&factory_calls);
    let factory = move || -> TelemetryPipeHandle {
        factory_calls_clone.fetch_add(1, Ordering::SeqCst);
        into_telemetry_handle(factory_collector.clone())
    };
    let initial = factory();
    let shared = Arc::new(Shared {
        queue: SeverityBucketedQueue::new(8, super::config::RetentionHorizons::default()),
        factory: alloc::boxed::Box::new(factory),
        current: parking_lot::RwLock::new(initial),
        clock: clock.clone(),
        config: small_config(),
        notify: proxima_primitives::sync::Notify::new(),
        shutdown: core::sync::atomic::AtomicBool::new(false),
        last_tick_nanos: AtomicU64::new(0),
        since_tick_drops: core::array::from_fn(|_| AtomicU64::new(0)),
        counters: worker::SelfCounters::default(),
        self_recorder: None,
    });
    shared.queue.push(
        QueuedRecord::Log(Arc::new(log_record(Level::ERROR, "x"))),
        0,
    );

    let mut cursor = SendCursor::default();
    for _ in 0..4 {
        let before = clock.now_nanos();
        worker::step(&shared, &mut cursor);
        let delay = Duration::from_nanos(cursor.next_attempt_at_nanos.saturating_sub(before));
        clock.advance(delay + Duration::from_millis(1));
    }

    assert!(
        factory_calls.load(Ordering::SeqCst) >= 3,
        "each transport-error attempt must rebuild the handle (reconnect), factory calls={}",
        factory_calls.load(Ordering::SeqCst)
    );
    assert_eq!(
        collector.received.logs().len(),
        1,
        "the batch eventually landed once refusals ran out"
    );
}
