//! Dispatch-integrity tests for `core_shard`'s cross-core `dispatch_send` /
//! `dispatch_factory`.
//!
//! These tests predate Wave D Phase 1 under the name "cross-core span
//! carry", but they only ever captured a plain `u64` by value into the
//! dispatched future's closure — they never touched the (now-deleted)
//! ambient span-carry machinery. Renamed to describe what they actually
//! prove: a value captured by a future dispatched to a remote core arrives
//! intact and isolated from other concurrent dispatches. The genuine
//! span-carry contract — a real `SpanRecord` surviving a real cross-core
//! hop — is proven by the new test at the bottom of this file, which wraps
//! the dispatched future in `telemetry::Spanned<T>`.
#![cfg(all(
    feature = "runtime-prime-full",
    feature = "runtime-prime-executor",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-inbox-alloc",
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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

/// a value captured by a dispatched future arrives intact on the
/// remote core (renamed from `cross_core_dispatch_carries_span` — the
/// dispatched closure captures a plain `u64` by value, which is
/// dispatch-integrity, not span propagation).
#[test]
fn dispatch_send_delivers_the_captured_value_to_the_remote_core() {
    let captured_value = make_span_id(0xde);
    let received = Arc::new(AtomicU64::new(u64::MAX));
    let received_copy = received.clone();

    let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 64).expect("launch core 0");

    let captured_bytes = u64::from_le_bytes(captured_value.to_bytes());

    handle
        .dispatch_send(Box::pin(async move {
            received_copy.store(captured_bytes, Ordering::Release);
        }))
        .expect("dispatch");

    let deadline = Instant::now() + Duration::from_secs(2);
    while received.load(Ordering::Acquire) == u64::MAX && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }

    handle.shutdown_and_join().expect("shutdown");

    let stored = received.load(Ordering::Acquire);
    assert_ne!(stored, u64::MAX, "task on remote core must have run");
    assert_eq!(
        stored, captured_bytes,
        "the value captured at dispatch time must arrive unchanged on the remote core",
    );
}

/// consecutive dispatches deliver independent captured values — no
/// bleed between them (renamed from
/// `consecutive_dispatches_have_independent_spans`).
#[test]
fn consecutive_dispatch_send_calls_deliver_independent_captured_values() {
    let value_a = make_span_id(0x0a);
    let value_b = make_span_id(0x0b);
    let a_bytes = u64::from_le_bytes(value_a.to_bytes());
    let b_bytes = u64::from_le_bytes(value_b.to_bytes());

    let observed_a = Arc::new(AtomicU64::new(u64::MAX));
    let observed_b = Arc::new(AtomicU64::new(u64::MAX));

    let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 64).expect("launch");

    let obs_a = observed_a.clone();
    handle
        .dispatch_send(Box::pin(async move {
            obs_a.store(a_bytes, Ordering::Release);
        }))
        .expect("send a");

    let obs_b = observed_b.clone();
    handle
        .dispatch_send(Box::pin(async move {
            obs_b.store(b_bytes, Ordering::Release);
        }))
        .expect("send b");

    let deadline = Instant::now() + Duration::from_secs(2);
    while (observed_a.load(Ordering::Acquire) == u64::MAX
        || observed_b.load(Ordering::Acquire) == u64::MAX)
        && Instant::now() < deadline
    {
        std::thread::sleep(Duration::from_millis(5));
    }

    handle.shutdown_and_join().expect("shutdown");

    assert_eq!(observed_a.load(Ordering::Acquire), a_bytes, "dispatch a");
    assert_eq!(observed_b.load(Ordering::Acquire), b_bytes, "dispatch b");
}

/// factory dispatch delivers its captured value the same as `Send`
/// dispatch (renamed from `factory_dispatch_carries_span`).
#[test]
fn dispatch_factory_delivers_its_captured_value() {
    let done = Arc::new(AtomicBool::new(false));
    let done_for_task = done.clone();

    let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 64).expect("launch");

    handle
        .dispatch_factory(Box::new(move || {
            let done = done_for_task.clone();
            Box::pin(async move {
                done.store(true, Ordering::Release);
            })
        }))
        .expect("factory dispatch");

    let deadline = Instant::now() + Duration::from_secs(2);
    while !done.load(Ordering::Acquire) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }

    handle.shutdown_and_join().expect("shutdown");
    assert!(done.load(Ordering::Acquire), "factory task must complete");
}

/// Test-double sink shared across the dispatching thread and the
/// remote core's worker thread — a real `Arc<Mutex<..>>` (not
/// `Rc<RefCell<..>>`) because this genuinely crosses OS threads.
#[derive(Clone, Default)]
struct RecordingSink {
    records: Arc<Mutex<Vec<SpanRecord>>>,
}

impl SpanSink for RecordingSink {
    fn emit(&mut self, record: SpanRecord) {
        self.records.lock().expect("recording sink mutex poisoned").push(record);
    }
}

fn spanned_guard(
    sink: RecordingSink,
    trace_id: TraceId,
    span_id: SpanId,
    parent_span_id: Option<SpanId>,
) -> SpanGuard<RecordingSink, MonotonicCounter> {
    let mut builder = SpanBuilder::new("cross-core-child", trace_id, span_id);
    if let Some(parent) = parent_span_id {
        builder = builder.parent(parent);
    }
    builder
        .start(&MonotonicCounter::new(0), sink)
        .enter(MonotonicCounter::new(0))
}

/// THE genuine span-carry proof: wrap the dispatched future in
/// `telemetry::Spanned<T>` — its guard travels WITH the future across
/// the cross-core hop and finishes (emits its `SpanRecord`) on the
/// RECEIVING core's worker thread once the future resolves there.
#[test]
fn spanned_future_dispatched_cross_core_emits_its_record_on_the_remote_core() {
    let sink = RecordingSink::default();
    let trace_id = TraceId::from_bytes([0x33; 16]);
    let parent_span = make_span_id(0x44);
    let child_span = make_span_id(0x55);
    let guard = spanned_guard(sink.clone(), trace_id, child_span, Some(parent_span));

    let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 64).expect("launch core 0");
    handle
        .dispatch_send(Box::pin(Spanned::new(async {}, guard)))
        .expect("dispatch spanned future");

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
        "the remote core must emit exactly one record for the dispatched span"
    );
    assert_eq!(
        records[0].trace_id, trace_id,
        "trace id must survive the cross-core hop"
    );
    assert_eq!(
        records[0].span_id, child_span,
        "span id must survive the cross-core hop"
    );
    assert_eq!(
        records[0].parent_span_id,
        Some(parent_span),
        "parent span id must survive the cross-core hop"
    );
}
