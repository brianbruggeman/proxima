use std::sync::Arc;

use proxima_primitives::sync::Notify;

use crate::event::RecordingEvent;
use crate::pipe::event_sink::{AppendFuture, DynRecordingSink, RecordingSink};
use proxima_core::ProximaError;
use proxima_primitives::pipe::telemetry_surface::{Labels, NoopTelemetry, TelemetryHandle};
use proxima_primitives::pipe::{BoundedQueue, EnqueueOutcome, SinkCounters};

// the overflow drop policy is the generic `proxima_primitives::pipe::FailMode`, re-exported
// so callers keep `crate::pipe::FailMode`.
pub use proxima_primitives::pipe::FailMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropReason {
    QueueFull,
}

impl DropReason {
    fn as_label(self) -> &'static str {
        match self {
            Self::QueueFull => "queue_full",
        }
    }
}

pub const RECORD_DROP_METRIC: &str = "record_dropped_total";

pub struct BoundedRecordingSink {
    inner: Arc<BoundedInner>,
}

struct BoundedInner {
    backend: DynRecordingSink,
    /// bounded lock-free ring with an overflow drop policy + drop counter
    /// (multiple chain tasks enqueue; the worker dequeues).
    queue: BoundedQueue<RecordingEvent>,
    pending_wakeup: Notify,
    progress_signal: Notify,
    // append/drain accounting + the `appended == drained + drops` quiescence
    // invariant as a NAMED predicate (was two scattered atomics + an inline
    // check repeated in `drain`).
    counters: SinkCounters,
    telemetry: TelemetryHandle,
    drop_labels: Labels,
}

impl BoundedRecordingSink {
    pub fn new(backend: DynRecordingSink, capacity: usize, fail_mode: FailMode) -> Self {
        Self::with_telemetry(
            backend,
            capacity,
            fail_mode,
            Arc::new(NoopTelemetry),
            Labels::empty(),
        )
    }

    pub fn with_telemetry(
        backend: DynRecordingSink,
        capacity: usize,
        fail_mode: FailMode,
        telemetry: TelemetryHandle,
        drop_labels: Labels,
    ) -> Self {
        let inner = Arc::new(BoundedInner {
            backend,
            queue: BoundedQueue::new(capacity, fail_mode),
            pending_wakeup: Notify::new(),
            progress_signal: Notify::new(),
            counters: SinkCounters::new(),
            telemetry,
            drop_labels,
        });
        let worker = inner.clone();
        // one dedicated OS thread per sink instance (not per-call) driving
        // the drain loop via `block_on` — real background progress with no
        // particular async runtime required.
        std::thread::spawn(move || {
            futures::executor::block_on(run_worker(worker));
        });
        Self { inner }
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner.queue.capacity()
    }

    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.inner.queue.dropped()
    }

    #[must_use]
    pub fn drained(&self) -> u64 {
        self.inner.counters.drained()
    }

    fn is_quiescent(&self) -> bool {
        self.inner
            .counters
            .is_quiescent(self.inner.queue.len(), self.inner.queue.dropped())
    }

    pub async fn drain(&self) {
        loop {
            if self.is_quiescent() {
                return;
            }
            let waiter = self.inner.progress_signal.notified();
            futures::pin_mut!(waiter);
            waiter.as_mut().enable();
            // re-check after enabling the waiter to avoid lost-wakeup races.
            if self.is_quiescent() {
                return;
            }
            waiter.await;
        }
    }
}

impl RecordingSink for BoundedRecordingSink {
    fn append<'lifetime>(&'lifetime self, event: RecordingEvent) -> AppendFuture<'lifetime> {
        let inner = self.inner.clone();
        Box::pin(async move { enqueue(inner, event).await })
    }

    fn flush<'lifetime>(&'lifetime self) -> AppendFuture<'lifetime> {
        let inner = self.inner.clone();
        let drain_fut = self.drain();
        Box::pin(async move {
            drain_fut.await;
            inner.backend.flush().await
        })
    }
}

async fn enqueue(inner: Arc<BoundedInner>, event: RecordingEvent) -> Result<(), ProximaError> {
    inner.counters.record_append();
    // the generic queue applies the drop policy and counts the drop; we react to
    // the outcome here (wake the worker / drain waiters, emit the drop metric).
    match inner.queue.enqueue(event) {
        EnqueueOutcome::Enqueued => {
            inner.pending_wakeup.notify_one();
            Ok(())
        }
        EnqueueOutcome::DroppedOldest => {
            record_drop(&inner, DropReason::QueueFull);
            inner.pending_wakeup.notify_one();
            Ok(())
        }
        EnqueueOutcome::DroppedNewest => {
            record_drop(&inner, DropReason::QueueFull);
            inner.progress_signal.notify_waiters();
            Ok(())
        }
        EnqueueOutcome::Refused => {
            record_drop(&inner, DropReason::QueueFull);
            inner.progress_signal.notify_waiters();
            Err(ProximaError::Record(format!(
                "recording ring buffer full at capacity {}",
                inner.queue.capacity()
            )))
        }
    }
}

fn record_drop(inner: &BoundedInner, reason: DropReason) {
    let pairs: Vec<(&str, &str)> = inner
        .drop_labels
        .entries()
        .iter()
        .map(|(name, value)| (name.as_str(), value.as_str()))
        .chain(std::iter::once(("reason", reason.as_label())))
        .collect();
    let labels = Labels::from_pairs(&pairs);
    inner.telemetry.counter_inc(RECORD_DROP_METRIC, &labels, 1);
}

async fn run_worker(inner: Arc<BoundedInner>) {
    loop {
        match inner.queue.dequeue() {
            Some(event) => {
                if let Err(error) = inner.backend.append(event).await {
                    tracing::error!(error = %error, "bounded recording backend failed");
                }
                inner.counters.record_drain();
                inner.progress_signal.notify_waiters();
            }
            None => {
                inner.pending_wakeup.notified().await;
            }
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
mod tests {
    use super::*;
    use crate::event::InteractionId;
    use crate::pipe::event_sink::AppendFuture;
    use proxima_primitives::pipe::telemetry_surface::Telemetry;
    use std::collections::HashMap;
    use std::sync::Mutex as StdMutex;

    #[derive(Default)]
    struct Metrics {
        counters: StdMutex<HashMap<(String, Vec<(String, String)>), u64>>,
    }

    impl Telemetry for Metrics {
        fn counter_inc(&self, metric: &str, labels: &Labels, by: u64) {
            let key = (metric.to_string(), labels.entries().to_vec());
            *self.counters.lock().unwrap().entry(key).or_insert(0) += by;
        }
        fn gauge_set(&self, _: &str, _: &Labels, _: i64) {}
        fn histogram_record(&self, _: &str, _: &Labels, _: f64) {}
    }

    impl Metrics {
        fn counter(&self, metric: &str, labels: &Labels) -> Option<u64> {
            let key = (metric.to_string(), labels.entries().to_vec());
            self.counters.lock().unwrap().get(&key).copied()
        }
    }

    #[derive(Default)]
    struct MemorySink {
        events: StdMutex<Vec<RecordingEvent>>,
    }

    impl RecordingSink for MemorySink {
        fn append<'lifetime>(&'lifetime self, event: RecordingEvent) -> AppendFuture<'lifetime> {
            Box::pin(async move {
                self.events.lock().expect("memory sink lock").push(event);
                Ok(())
            })
        }

        fn flush<'lifetime>(&'lifetime self) -> AppendFuture<'lifetime> {
            Box::pin(async { Ok(()) })
        }
    }

    fn req_end(seq: u64) -> RecordingEvent {
        use crate::event::{HttpEvent, ProtocolEvent};
        RecordingEvent {
            id: InteractionId::from_bytes([(seq % 256) as u8; 16]),
            ts_ms: seq,
            parent: None,
            event: ProtocolEvent::Http(HttpEvent::RequestEnded),
        }
    }

    #[proxima::test]
    async fn enqueue_then_drain_emits_all_events_in_order() {
        let backend: Arc<MemorySink> = Arc::new(MemorySink::default());
        let cap_sink = BoundedRecordingSink::new(backend.clone(), 8, FailMode::FailClosed);
        for seq in 0..5 {
            cap_sink.append(req_end(seq)).await.expect("append");
        }
        cap_sink.flush().await.expect("flush");
        let events = backend.events.lock().expect("memory events");
        assert_eq!(events.len(), 5);
        for (seq, event) in events.iter().enumerate() {
            assert_eq!(event.ts_ms(), seq as u64);
        }
    }

    // `BoundedQueue::new(capacity, ..)` rounds up to the ring's minimum (a
    // power of two >= 2), so a nominal capacity of 1 still holds 2 items —
    // pre-fill to `cap_sink.capacity()`, not the nominal argument, to
    // deterministically reach the ACTUAL full state.
    fn fill_to_capacity(cap_sink: &BoundedRecordingSink) {
        for seq in 0..cap_sink.capacity() as u64 {
            assert_eq!(
                cap_sink.inner.queue.enqueue(req_end(seq)),
                EnqueueOutcome::Enqueued,
                "pre-fill must fit"
            );
        }
    }

    #[proxima::test]
    async fn fail_closed_returns_typed_error_when_full() {
        let backend: Arc<MemorySink> = Arc::new(MemorySink::default());
        let cap_sink = BoundedRecordingSink::new(backend.clone(), 1, FailMode::FailClosed);
        // pre-fill the ring directly to deterministically force the overflow
        // path without racing the worker.
        fill_to_capacity(&cap_sink);
        let result = cap_sink.append(req_end(cap_sink.capacity() as u64)).await;
        assert!(matches!(result, Err(ProximaError::Record(_))));
        assert_eq!(cap_sink.dropped(), 1);
    }

    #[proxima::test]
    async fn drop_oldest_evicts_oldest_event_when_queue_is_full() {
        let backend: Arc<MemorySink> = Arc::new(MemorySink::default());
        let cap_sink = BoundedRecordingSink::new(backend.clone(), 1, FailMode::DropOldest);
        fill_to_capacity(&cap_sink);
        let capacity = cap_sink.capacity() as u64;
        cap_sink
            .append(req_end(capacity))
            .await
            .expect("append should succeed under DropOldest");
        // the oldest pre-filled event (ts_ms=0) must have been evicted; every
        // other pre-filled event plus the new one remain, in FIFO order.
        let mut remaining = Vec::new();
        while let Some(event) = cap_sink.inner.queue.dequeue() {
            remaining.push(event.ts_ms());
        }
        assert_eq!(remaining, (1..=capacity).collect::<Vec<_>>());
        assert_eq!(cap_sink.dropped(), 1);
    }

    #[proxima::test]
    async fn telemetry_counter_increments_on_drop() {
        let metrics: Arc<Metrics> = Arc::new(Metrics::default());
        let telemetry: TelemetryHandle = metrics.clone();
        let backend: Arc<MemorySink> = Arc::new(MemorySink::default());
        let labels = Labels::from_pairs(&[("pipe", "echo")]);
        let cap_sink = BoundedRecordingSink::with_telemetry(
            backend.clone(),
            1,
            FailMode::DropNewest,
            telemetry,
            labels,
        );
        fill_to_capacity(&cap_sink);
        cap_sink
            .append(req_end(cap_sink.capacity() as u64))
            .await
            .expect("append must succeed under DropNewest");
        let read_labels = Labels::from_pairs(&[("pipe", "echo"), ("reason", "queue_full")]);
        assert_eq!(
            metrics.counter(RECORD_DROP_METRIC, &read_labels),
            Some(1),
            "drop counter must report exactly one queue_full event"
        );
    }
}
