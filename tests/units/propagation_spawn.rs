//! Wave D Phase 1: span context no longer rides on an ambient
//! `LocalExecutor::current_span_id()` reader (that API, and the `_with_span`
//! spawn variants, were deleted along with prime's C15 span-carry
//! subsystem — proxima has no ambient "current span" state by design).
//!
//! The replacement contract this file pins: a span travels WITH the future
//! it belongs to, via `telemetry::Spanned<T>` wrapping the future before
//! it's spawned. These tests observe that contract the way a real consumer
//! would — by inspecting the `SpanRecord` a test-double `SpanSink` receives
//! when the wrapped future resolves — instead of reaching into the
//! executor's internals through a raw pointer.
#![cfg(all(feature = "runtime-prime-full", feature = "runtime-prime-executor"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::cell::RefCell;
use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll, Waker};

use proxima::runtime::prime::core::local_executor::LocalExecutor;
use proxima::telemetry::clock::MonotonicCounter;
use proxima::telemetry::id::{SpanId, TraceId};
use proxima::telemetry::spanned::Spanned;
use proxima::telemetry::trace::{SpanBuilder, SpanGuard, SpanRecord, SpanSink};

/// Test-double sink: every `SpanRecord` a `SpanGuard` finishes gets
/// pushed here. `Rc<RefCell<..>>` (not `Arc<Mutex<..>>`) is enough
/// because `LocalExecutor` is `!Send` and every test below drives it
/// on a single thread.
#[derive(Clone, Default)]
struct RecordingSink {
    records: Rc<RefCell<Vec<SpanRecord>>>,
}

impl SpanSink for RecordingSink {
    fn emit(&mut self, record: SpanRecord) {
        self.records.borrow_mut().push(record);
    }
}

fn spanned_guard(
    sink: RecordingSink,
    trace_id: TraceId,
    span_id: SpanId,
    parent_span_id: Option<SpanId>,
) -> SpanGuard<RecordingSink, MonotonicCounter> {
    let mut builder = SpanBuilder::new("child", trace_id, span_id);
    if let Some(parent) = parent_span_id {
        builder = builder.parent(parent);
    }
    builder
        .start(&MonotonicCounter::new(0), sink)
        .enter(MonotonicCounter::new(0))
}

#[test]
fn spawned_future_emits_a_record_carrying_its_trace_and_parent_span_ids() {
    let sink = RecordingSink::default();
    let trace_id = TraceId::from_bytes([0xab; 16]);
    let parent_span = SpanId::from_bytes([0xaa; 8]);
    let child_span = SpanId::from_bytes([0xbb; 8]);
    let guard = spanned_guard(sink.clone(), trace_id, child_span, Some(parent_span));

    let executor = LocalExecutor::new();
    executor.arm();
    executor.spawn_local_pin(Box::pin(Spanned::new(async {}, guard)));
    executor.tick();
    executor.disarm();

    let records = sink.records.borrow();
    assert_eq!(records.len(), 1, "exactly one span record must be emitted");
    assert_eq!(
        records[0].trace_id, trace_id,
        "record must carry the parent's trace id — a child stays in the same trace"
    );
    assert_eq!(
        records[0].span_id, child_span,
        "record must carry the child's own span id"
    );
    assert_eq!(
        records[0].parent_span_id,
        Some(parent_span),
        "record must record the parent span id the child was created under"
    );
}

#[test]
fn spawn_local_without_spanned_emits_no_span_record() {
    // Ordinary spawn_local (no Spanned wrapper) never manufactures a
    // span behind the caller's back — proxima has no ambient auto-span.
    let sink = RecordingSink::default();
    let executor = LocalExecutor::new();
    executor.arm();
    executor.spawn_local(async {});
    executor.tick();
    executor.disarm();
    assert!(
        sink.records.borrow().is_empty(),
        "spawn_local with no Spanned wrapper must not emit any span record",
    );
}

#[test]
fn two_spanned_tasks_emit_independent_records() {
    let sink = RecordingSink::default();
    let trace_id = TraceId::from_bytes([0x11; 16]);
    let span_a = SpanId::from_bytes([0x01; 8]);
    let span_b = SpanId::from_bytes([0x02; 8]);

    let executor = LocalExecutor::new();
    executor.arm();
    executor.spawn_local_pin(Box::pin(Spanned::new(
        async {},
        spanned_guard(sink.clone(), trace_id, span_a, None),
    )));
    executor.spawn_local_pin(Box::pin(Spanned::new(
        async {},
        spanned_guard(sink.clone(), trace_id, span_b, None),
    )));
    executor.tick();
    executor.disarm();

    let records = sink.records.borrow();
    let span_ids: HashSet<_> = records.iter().map(|record| record.span_id).collect();
    assert_eq!(records.len(), 2, "both tasks must emit their own record");
    assert!(span_ids.contains(&span_a), "task a's span id must appear");
    assert!(span_ids.contains(&span_b), "task b's span id must appear");
}

#[test]
fn spanned_future_that_yields_still_emits_exactly_one_record_at_resolution() {
    // Registers its waker on first poll and returns Pending WITHOUT
    // self-waking (unlike a `wake_by_ref`-then-Pending future, which
    // `tick()` drains to completion within a single call — see
    // `LocalExecutor::tick`'s doc: it re-polls `local_ready` until
    // quiescent). Suspending on an externally-fired waker is the only
    // way to observe the executor genuinely parking the task BETWEEN
    // `tick()` calls, mirroring `pending_future_resumes_after_explicit_wake`
    // in `local_executor.rs`'s own test suite.
    struct SuspendUntilWoken {
        waker_slot: Rc<RefCell<Option<Waker>>>,
        resumed: bool,
    }

    impl Future for SuspendUntilWoken {
        type Output = ();
        fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
            let this = self.get_mut();
            if this.resumed {
                return Poll::Ready(());
            }
            this.resumed = true;
            *this.waker_slot.borrow_mut() = Some(context.waker().clone());
            Poll::Pending
        }
    }

    let sink = RecordingSink::default();
    let trace_id = TraceId::from_bytes([0x22; 16]);
    let span_id = SpanId::from_bytes([0xcc; 8]);
    let guard = spanned_guard(sink.clone(), trace_id, span_id, None);
    let waker_slot: Rc<RefCell<Option<Waker>>> = Rc::new(RefCell::new(None));

    let executor = LocalExecutor::new();
    executor.arm();
    executor.spawn_local_pin(Box::pin(Spanned::new(
        SuspendUntilWoken {
            waker_slot: waker_slot.clone(),
            resumed: false,
        },
        guard,
    )));

    // first tick: registers the waker and suspends — the span must
    // still be open, no record emitted from the intermediate suspend.
    executor.tick();
    assert!(
        sink.records.borrow().is_empty(),
        "the span must stay open while the task is genuinely parked — \
         a suspend must not finish it early",
    );

    // fire the captured waker (mirrors a cross-thread or reactor wake)
    // then tick again — the task resolves, closing the span exactly
    // once, carrying the same span id it was wrapped with (nothing
    // else could have replaced it — there's no ambient state to bleed
    // in).
    waker_slot
        .borrow_mut()
        .take()
        .expect("future must have registered its waker on first poll")
        .wake();
    executor.tick();
    executor.disarm();

    let records = sink.records.borrow();
    assert_eq!(
        records.len(),
        1,
        "span must finish exactly once, at resolution, not per poll"
    );
    assert_eq!(records[0].span_id, span_id, "span id must survive the suspend");
}
