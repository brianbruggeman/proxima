//! Dispatch/wake-integrity tests: a task dispatched to a `core_shard`
//! worker resumes correctly after a cross-thread reactor-style wake
//! (oneshot channel fired from another thread).
//!
//! These tests predate Wave D Phase 1 under the name "span id stable
//! across wake", but they only ever captured a plain `u64` by value into
//! the dispatched future's closure — they never touched the (now-deleted)
//! ambient span-carry machinery. Renamed to describe what they actually
//! prove: a value captured before a cross-thread wake is still correct
//! after it. The genuine span-carry contract — a real `SpanRecord`
//! surviving a real cross-thread wake — is proven by the new test at the
//! bottom of this file, which wraps the dispatched future in
//! `telemetry::Spanned<T>`.
#![cfg(all(
    feature = "runtime-prime-full",
    feature = "runtime-prime-executor",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-inbox-alloc",
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use proxima::runtime::CoreId;
use proxima::runtime::prime::os::core_shard;
use proxima::telemetry::clock::MonotonicCounter;
use proxima::telemetry::id::{SpanId, TraceId};
use proxima::telemetry::spanned::Spanned;
use proxima::telemetry::trace::{SpanBuilder, SpanGuard, SpanRecord, SpanSink};

fn make_span_id(byte: u8) -> SpanId {
    SpanId::from_bytes([byte; 8])
}

/// a value captured before an `.await` is unchanged after a
/// cross-thread wake resumes the task (renamed from
/// `span_stable_after_cross_thread_wake`).
#[test]
fn captured_value_is_unchanged_after_cross_thread_wake() {
    use futures::channel::oneshot;

    let captured_value = make_span_id(0xef);
    let captured_bytes = u64::from_le_bytes(captured_value.to_bytes());

    let before_await = Arc::new(AtomicU64::new(u64::MAX));
    let after_wake = Arc::new(AtomicU64::new(u64::MAX));

    let before_copy = before_await.clone();
    let after_copy = after_wake.clone();

    let (sender, receiver) = oneshot::channel::<()>();

    let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 64).expect("launch core 0");

    handle
        .dispatch_send(Box::pin(async move {
            before_copy.store(captured_bytes, Ordering::Release);
            let _ = receiver.await;
            after_copy.store(captured_bytes, Ordering::Release);
        }))
        .expect("dispatch");

    let deadline = Instant::now() + Duration::from_secs(2);
    while before_await.load(Ordering::Acquire) == u64::MAX && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_ne!(
        before_await.load(Ordering::Acquire),
        u64::MAX,
        "task must have started",
    );

    std::thread::sleep(Duration::from_millis(50));
    sender.send(()).expect("oneshot send");

    let deadline2 = Instant::now() + Duration::from_secs(2);
    while after_wake.load(Ordering::Acquire) == u64::MAX && Instant::now() < deadline2 {
        std::thread::sleep(Duration::from_millis(5));
    }

    handle.shutdown_and_join().expect("shutdown");

    assert_eq!(
        after_wake.load(Ordering::Acquire),
        captured_bytes,
        "captured value must be identical before and after the wake",
    );
}

/// multiple tasks with different captured values; after cross-thread
/// wakes both observe their own value, not each other's (renamed from
/// `multiple_tasks_span_isolation_across_wakes`).
#[test]
fn multiple_tasks_have_isolated_captured_values_across_wakes() {
    use futures::channel::oneshot;

    let value_a = make_span_id(0x1a);
    let value_b = make_span_id(0x2b);
    let a_bytes = u64::from_le_bytes(value_a.to_bytes());
    let b_bytes = u64::from_le_bytes(value_b.to_bytes());

    let after_a = Arc::new(AtomicU64::new(u64::MAX));
    let after_b = Arc::new(AtomicU64::new(u64::MAX));

    let (tx_a, rx_a) = oneshot::channel::<()>();
    let (tx_b, rx_b) = oneshot::channel::<()>();

    let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 64).expect("launch core 0");

    let after_a2 = after_a.clone();
    handle
        .dispatch_send(Box::pin(async move {
            let _ = rx_a.await;
            after_a2.store(a_bytes, Ordering::Release);
        }))
        .expect("dispatch a");

    let after_b2 = after_b.clone();
    handle
        .dispatch_send(Box::pin(async move {
            let _ = rx_b.await;
            after_b2.store(b_bytes, Ordering::Release);
        }))
        .expect("dispatch b");

    std::thread::sleep(Duration::from_millis(100));
    tx_a.send(()).expect("send a");
    tx_b.send(()).expect("send b");

    let deadline = Instant::now() + Duration::from_secs(2);
    while (after_a.load(Ordering::Acquire) == u64::MAX
        || after_b.load(Ordering::Acquire) == u64::MAX)
        && Instant::now() < deadline
    {
        std::thread::sleep(Duration::from_millis(5));
    }

    handle.shutdown_and_join().expect("shutdown");

    assert_eq!(after_a.load(Ordering::Acquire), a_bytes, "task a value");
    assert_eq!(after_b.load(Ordering::Acquire), b_bytes, "task b value");
}

/// Test-double sink shared across the dispatching thread and the
/// remote core's worker thread.
#[derive(Clone, Default)]
struct RecordingSink {
    records: Arc<Mutex<Vec<SpanRecord>>>,
}

impl SpanSink for RecordingSink {
    fn emit(&mut self, record: SpanRecord) {
        self.records
            .lock()
            .expect("recording sink mutex poisoned")
            .push(record);
    }
}

fn spanned_guard(
    sink: RecordingSink,
    trace_id: TraceId,
    span_id: SpanId,
) -> SpanGuard<RecordingSink, MonotonicCounter> {
    SpanBuilder::new("wake-child", trace_id, span_id)
        .start(&MonotonicCounter::new(0), sink)
        .enter(MonotonicCounter::new(0))
}

/// THE genuine span-carry proof: a `Spanned` future dispatched to a
/// remote core, suspended on a cross-thread reactor wake (oneshot
/// channel), still finishes its span — emitting exactly one
/// `SpanRecord` with the right id — once the wake resumes it and it
/// resolves.
#[test]
fn spanned_future_suspended_on_a_cross_thread_wake_still_emits_its_record() {
    use futures::channel::oneshot;

    let sink = RecordingSink::default();
    let trace_id = TraceId::from_bytes([0x66; 16]);
    let span_id = make_span_id(0x77);
    let guard = spanned_guard(sink.clone(), trace_id, span_id);

    let (sender, receiver) = oneshot::channel::<()>();

    let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 64).expect("launch core 0");
    handle
        .dispatch_send(Box::pin(Spanned::new(
            async move {
                let _ = receiver.await;
            },
            guard,
        )))
        .expect("dispatch spanned future");

    // give the worker time to poll once and park on the oneshot —
    // the span must still be open, no record emitted yet.
    std::thread::sleep(Duration::from_millis(100));
    assert!(
        sink.records
            .lock()
            .expect("recording sink mutex poisoned")
            .is_empty(),
        "the span must stay open while the task is parked on the cross-thread wake",
    );

    sender.send(()).expect("oneshot send");

    let deadline = Instant::now() + Duration::from_secs(2);
    while sink
        .records
        .lock()
        .expect("recording sink mutex poisoned")
        .is_empty()
        && Instant::now() < deadline
    {
        std::thread::sleep(Duration::from_millis(5));
    }

    handle.shutdown_and_join().expect("shutdown");

    let records = sink.records.lock().expect("recording sink mutex poisoned");
    assert_eq!(
        records.len(),
        1,
        "span must finish exactly once, after the cross-thread wake resolves the future"
    );
    assert_eq!(
        records[0].span_id, span_id,
        "span id must survive the cross-thread wake"
    );
}
