use core::sync::atomic::{AtomicU64, Ordering};

use crate::id::SpanId;
use crate::log::LogRecord;
use crate::metric::MetricSample;
use crate::ring::{FailMode, HeapBoundedQueue};
use crate::tag::Tag;
use crate::trace::{EventRecord, SpanLink, SpanRecord};

pub struct OverflowAttr {
    pub span_id: SpanId,
    pub tag: Tag,
}

/// A span-duration observation deferred to the drain thread: the span name (a
/// `&'static str` — `Copy`, no owned string) and its measured `duration_ns`.
/// 24-byte POD. The emit hot path pushes one to a per-core lock-free ring
/// instead of taking the registry Mutex to fold inline; the by-name→histogram
/// resolution + fold happen once per drain batch off the hot path.
#[cfg(feature = "deferred-metric-fold")]
#[derive(Debug, Clone, Copy)]
pub struct SpanObservation {
    pub name: &'static str,
    pub duration_ns: u64,
}

pub struct RingSet {
    pub spans: HeapBoundedQueue<SpanRecord>,
    pub events: HeapBoundedQueue<EventRecord>,
    pub logs: HeapBoundedQueue<LogRecord>,
    pub metrics: HeapBoundedQueue<MetricSample>,
    pub links: HeapBoundedQueue<SpanLink>,
    pub overflow_attrs: HeapBoundedQueue<OverflowAttr>,
    /// Per-core deferred span-duration observations (see [`SpanObservation`]).
    /// Lock-free, preallocated; the emit path pushes here mutex-free + zero-alloc.
    #[cfg(feature = "deferred-metric-fold")]
    pub span_obs: HeapBoundedQueue<SpanObservation>,
    /// Records this core's producers exported themselves via elastic
    /// producer-assist (a full ring forced the emitter to drain+export before it
    /// could push). The leading indicator that drain is not keeping up with
    /// emit: it climbs under `Block` BEFORE any latency cliff, while `dropped`
    /// stays 0. A nonzero/rising assist rate means "add drain capacity or shed".
    pub assisted: AtomicU64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingCapacities {
    pub spans: usize,
    pub events: usize,
    pub logs: usize,
    pub metrics: usize,
    pub links: usize,
    pub overflow_attrs: usize,
    #[cfg(feature = "deferred-metric-fold")]
    pub span_obs: usize,
}

impl Default for RingCapacities {
    fn default() -> Self {
        Self {
            spans: 4096,
            events: 4096,
            logs: 4096,
            metrics: 8192,
            links: 1024,
            overflow_attrs: 2048,
            #[cfg(feature = "deferred-metric-fold")]
            span_obs: 8192,
        }
    }
}

impl RingSet {
    pub fn new(caps: &RingCapacities) -> Result<Self, crate::error::Error> {
        // each stream is a bounded queue with drop-newest overflow; the recorder
        // routes Block vs Drop per-emit on top via `try_enqueue`, and counts a
        // Drop through the queue's own counter (`note_drop`), so `dropped()` sums
        // the per-stream totals rather than a separate per-core atomic.
        Ok(Self {
            spans: HeapBoundedQueue::new(caps.spans, FailMode::DropNewest),
            events: HeapBoundedQueue::new(caps.events, FailMode::DropNewest),
            logs: HeapBoundedQueue::new(caps.logs, FailMode::DropNewest),
            metrics: HeapBoundedQueue::new(caps.metrics, FailMode::DropNewest),
            links: HeapBoundedQueue::new(caps.links, FailMode::DropNewest),
            overflow_attrs: HeapBoundedQueue::new(caps.overflow_attrs, FailMode::DropNewest),
            #[cfg(feature = "deferred-metric-fold")]
            span_obs: HeapBoundedQueue::new(caps.span_obs, FailMode::DropNewest),
            assisted: AtomicU64::new(0),
        })
    }

    /// Records dropped on this core (a full ring at emit under `Drop`), summed
    /// across the streams — each queue owns its own drop count.
    pub fn dropped(&self) -> u64 {
        let total = self.spans.dropped()
            + self.events.dropped()
            + self.logs.dropped()
            + self.metrics.dropped()
            + self.links.dropped()
            + self.overflow_attrs.dropped();
        #[cfg(feature = "deferred-metric-fold")]
        let total = total + self.span_obs.dropped();
        total
    }

    /// Count `records` exported by a producer via elastic producer-assist.
    pub fn note_assist(&self, records: usize) {
        self.assisted.fetch_add(records as u64, Ordering::Relaxed);
    }

    pub fn assisted(&self) -> u64 {
        self.assisted.load(Ordering::Relaxed)
    }
}
