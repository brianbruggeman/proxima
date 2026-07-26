//! The resilient sink's own buffer: five per-severity FIFO lanes (trace,
//! debug, info, warn, error) under one capacity, with two independent
//! eviction paths — an age horizon per lane, and a shortest-horizon-first
//! space backstop when the total exceeds capacity. Shape follows
//! `ElevationSink`'s TTL-sweep + hard-cap-backstop idiom (`crate::pipes`),
//! applied to severity lanes instead of a per-trace map.

use alloc::collections::VecDeque;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};
use core::time::Duration;

use parking_lot::Mutex;
use prost::Message;

use crate::level::Level;
use crate::log::LogRecord;
use crate::metric::MetricSample;
use crate::out::otlp_http::conv::{
    event_to_proto, link_to_proto, log_to_proto, metric_to_proto, span_to_proto,
};
use crate::pipes::{
    TelemetryRequest, event_batch_arc_request, link_batch_arc_request, log_batch_arc_request,
    metric_batch_arc_request, span_batch_arc_request,
};
use crate::trace::{EventRecord, SpanLink, SpanRecord, Status};

use super::config::RetentionHorizons;

pub(super) const SEVERITY_BUCKETS: usize = 5;

/// Map a log [`Level`] to one of the five retention lanes.
pub(super) fn bucket_of_level(level: Level) -> usize {
    match level.severity() {
        severity if severity < Level::DEBUG.severity() => 0,
        severity if severity < Level::INFO.severity() => 1,
        severity if severity < Level::WARN.severity() => 2,
        severity if severity < Level::ERROR.severity() => 3,
        _ => 4,
    }
}

/// One retained record. `Arc`-wrapped uniformly (whether it arrived owned or
/// already `Arc`-shared) so a retry clones a refcount, never the payload, and
/// so re-batching reuses the existing `*_batch_arc_request` builders.
#[derive(Clone)]
pub(super) enum QueuedRecord {
    Span(Arc<SpanRecord>),
    Event(Arc<EventRecord>),
    Log(Arc<LogRecord>),
    Metric(Arc<MetricSample>),
    Link(Arc<SpanLink>),
}

impl QueuedRecord {
    /// Which of the five retention lanes this record belongs in. Only
    /// `LogRecord` carries an explicit [`Level`]; spans use their terminal
    /// status (an errored span is as valuable as an error log), and the
    /// remaining kinds (event/metric/link) default to the `info` lane — a
    /// simple, defensible default rather than inventing severity for signals
    /// that carry none.
    pub(super) fn severity_bucket(&self) -> usize {
        match self {
            QueuedRecord::Log(record) => bucket_of_level(record.level),
            QueuedRecord::Span(record) => match record.status {
                Status::Error { .. } => 4,
                _ => 2,
            },
            QueuedRecord::Event(_) | QueuedRecord::Metric(_) | QueuedRecord::Link(_) => 2,
        }
    }

    /// Exact OTLP-encoded size of this one record, used to chunk a flush
    /// batch under the collector's message-size ceiling. Re-derives the
    /// proto form the codec would build anyway; acceptable cost on the
    /// backlog/outage path, not the hot path.
    pub(super) fn encoded_len(&self) -> usize {
        match self {
            QueuedRecord::Span(record) => span_to_proto(record).encoded_len(),
            QueuedRecord::Log(record) => log_to_proto(record).encoded_len(),
            QueuedRecord::Metric(record) => metric_to_proto(record).encoded_len(),
            QueuedRecord::Event(record) => event_to_proto(record).encoded_len(),
            QueuedRecord::Link(record) => link_to_proto(record).encoded_len(),
        }
    }

    fn kind(&self) -> RecordKind {
        match self {
            QueuedRecord::Span(_) => RecordKind::Span,
            QueuedRecord::Event(_) => RecordKind::Event,
            QueuedRecord::Log(_) => RecordKind::Log,
            QueuedRecord::Metric(_) => RecordKind::Metric,
            QueuedRecord::Link(_) => RecordKind::Link,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RecordKind {
    Span,
    Event,
    Log,
    Metric,
    Link,
}

struct Entry {
    record: QueuedRecord,
    enqueued_at_nanos: u64,
}

/// A pulled, homogeneous-by-kind batch ready to encode into one
/// `TelemetryRequest`, plus the severity lane it was drawn from (for
/// telemetry/labeling only — the send path treats all kinds alike).
pub(super) enum PulledBatch {
    Spans(Vec<Arc<SpanRecord>>),
    Events(Vec<Arc<EventRecord>>),
    Logs(Vec<Arc<LogRecord>>),
    Metrics(Vec<Arc<MetricSample>>),
    Links(Vec<Arc<SpanLink>>),
}

impl PulledBatch {
    pub(super) fn len(&self) -> usize {
        match self {
            PulledBatch::Spans(items) => items.len(),
            PulledBatch::Events(items) => items.len(),
            PulledBatch::Logs(items) => items.len(),
            PulledBatch::Metrics(items) => items.len(),
            PulledBatch::Links(items) => items.len(),
        }
    }

    /// Build a fresh `TelemetryRequest` from this batch. Called once per send
    /// ATTEMPT (cloning the `Arc<T>` spine, not the payloads) so a retry
    /// resends byte-identical content without holding the request itself
    /// across the attempt.
    pub(super) fn to_request(&self) -> TelemetryRequest {
        match self {
            PulledBatch::Spans(items) => span_batch_arc_request(items.clone()),
            PulledBatch::Events(items) => event_batch_arc_request(items.clone()),
            PulledBatch::Logs(items) => log_batch_arc_request(items.clone()),
            PulledBatch::Metrics(items) => metric_batch_arc_request(items.clone()),
            PulledBatch::Links(items) => link_batch_arc_request(items.clone()),
        }
    }
}

/// One overflow/expiry event, for the aggregated announcement + counters.
#[derive(Clone, Copy)]
pub(super) struct DropEvent {
    pub(super) bucket: usize,
    pub(super) reason: DropReason,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum DropReason {
    Space,
    Expired,
}

struct State {
    lanes: [VecDeque<Entry>; SEVERITY_BUCKETS],
    total_len: usize,
    ingest_since_tick: [u64; SEVERITY_BUCKETS],
}

impl State {
    fn new() -> Self {
        Self {
            lanes: core::array::from_fn(|_| VecDeque::new()),
            total_len: 0,
            ingest_since_tick: [0; SEVERITY_BUCKETS],
        }
    }
}

/// The bounded, severity-lane buffer backing the resilient OTLP sink. Cheap
/// to enqueue into (a short mutex section, no I/O) — this is what lets the
/// sink's `Pipe::call` return immediately to the shared drain.
pub(super) struct SeverityBucketedQueue {
    state: Mutex<State>,
    capacity: usize,
    horizons: RetentionHorizons,
    dropped_total: [AtomicU64; SEVERITY_BUCKETS],
}

impl SeverityBucketedQueue {
    pub(super) fn new(capacity: usize, horizons: RetentionHorizons) -> Self {
        Self {
            state: Mutex::new(State::new()),
            capacity,
            horizons,
            dropped_total: core::array::from_fn(|_| AtomicU64::new(0)),
        }
    }

    /// Enqueue one record, evicting under space pressure if needed. Returns
    /// the eviction this push caused, if any.
    pub(super) fn push(&self, record: QueuedRecord, now_nanos: u64) -> Option<DropEvent> {
        let bucket = record.severity_bucket();
        let mut state = self.state.lock();
        state.lanes[bucket].push_back(Entry {
            record,
            enqueued_at_nanos: now_nanos,
        });
        state.total_len += 1;
        state.ingest_since_tick[bucket] += 1;

        if state.total_len <= self.capacity {
            return None;
        }
        // shortest-horizon-first: evict from the lowest-severity nonempty
        // lane (trace, then debug, ...) so space pressure never takes an
        // error while a lower lane still holds anything.
        for victim_bucket in 0..SEVERITY_BUCKETS {
            if state.lanes[victim_bucket].pop_front().is_some() {
                state.total_len -= 1;
                self.dropped_total[victim_bucket].fetch_add(1, Ordering::Relaxed);
                return Some(DropEvent {
                    bucket: victim_bucket,
                    reason: DropReason::Space,
                });
            }
        }
        None
    }

    /// Evict every entry older than its own lane's horizon. Run periodically
    /// (amortized) by the worker tick, not per-enqueue — mirrors
    /// `ElevationSink`'s sweep-every-N-calls pattern.
    pub(super) fn sweep_horizons(&self, now_nanos: u64) -> Vec<DropEvent> {
        let mut events = Vec::new();
        let mut state = self.state.lock();
        for bucket in 0..SEVERITY_BUCKETS {
            let horizon_nanos = self.horizons.for_bucket(bucket).as_nanos() as u64;
            while let Some(front) = state.lanes[bucket].front() {
                let age = now_nanos.saturating_sub(front.enqueued_at_nanos);
                if age <= horizon_nanos {
                    break;
                }
                state.lanes[bucket].pop_front();
                state.total_len -= 1;
                self.dropped_total[bucket].fetch_add(1, Ordering::Relaxed);
                events.push(DropEvent {
                    bucket,
                    reason: DropReason::Expired,
                });
            }
        }
        events
    }

    /// Pull the next batch to send: highest-severity nonempty lane first (so
    /// a backlog flush drains errors before debug), grouped by the SAME
    /// record kind (OTLP batches are signal-typed) and capped by both
    /// `max_items` and `max_bytes` (the collector's message-size ceiling).
    /// A single record that alone exceeds `max_bytes` is still returned
    /// alone rather than starving the lane forever.
    pub(super) fn pull_batch(&self, max_items: usize, max_bytes: usize) -> Option<PulledBatch> {
        let mut state = self.state.lock();
        for bucket in (0..SEVERITY_BUCKETS).rev() {
            let Some(kind) = state.lanes[bucket].front().map(|front| front.record.kind()) else {
                continue;
            };
            let mut collected: Vec<QueuedRecord> = Vec::new();
            let mut bytes_used = 0usize;
            while let Some(front) = state.lanes[bucket].front() {
                if front.record.kind() != kind || collected.len() >= max_items {
                    break;
                }
                let item_len = front.record.encoded_len();
                if !collected.is_empty() && bytes_used + item_len > max_bytes {
                    break;
                }
                let Some(entry) = state.lanes[bucket].pop_front() else {
                    break;
                };
                bytes_used += item_len;
                collected.push(entry.record);
                state.total_len -= 1;
            }
            if !collected.is_empty() {
                return Some(group_by_kind(kind, collected));
            }
        }
        None
    }

    pub(super) fn total_len(&self) -> usize {
        self.state.lock().total_len
    }

    pub(super) fn lane_lens(&self) -> [usize; SEVERITY_BUCKETS] {
        let state = self.state.lock();
        core::array::from_fn(|bucket| state.lanes[bucket].len())
    }

    /// Records enqueued per lane since the last call, and the elapsed window
    /// — the raw material for a rate estimate. Resets the counters.
    pub(super) fn take_ingest_counts(&self) -> [u64; SEVERITY_BUCKETS] {
        let mut state = self.state.lock();
        let counts = state.ingest_since_tick;
        state.ingest_since_tick = [0; SEVERITY_BUCKETS];
        counts
    }

    /// Age (in nanos) of the globally oldest retained entry, if any — for
    /// the drop-announcement line's `oldest_ms` field.
    pub(super) fn oldest_age_nanos(&self, now_nanos: u64) -> Option<u64> {
        let state = self.state.lock();
        state
            .lanes
            .iter()
            .filter_map(|lane| lane.front())
            .map(|entry| now_nanos.saturating_sub(entry.enqueued_at_nanos))
            .max()
    }

    pub(super) fn dropped_total(&self, bucket: usize) -> u64 {
        self.dropped_total[bucket].load(Ordering::Relaxed)
    }

    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }
}

fn group_by_kind(kind: RecordKind, records: Vec<QueuedRecord>) -> PulledBatch {
    match kind {
        RecordKind::Span => PulledBatch::Spans(
            records
                .into_iter()
                .map(|record| match record {
                    QueuedRecord::Span(span) => span,
                    _ => unreachable!("pull_batch only groups matching kinds"),
                })
                .collect(),
        ),
        RecordKind::Event => PulledBatch::Events(
            records
                .into_iter()
                .map(|record| match record {
                    QueuedRecord::Event(event) => event,
                    _ => unreachable!("pull_batch only groups matching kinds"),
                })
                .collect(),
        ),
        RecordKind::Log => PulledBatch::Logs(
            records
                .into_iter()
                .map(|record| match record {
                    QueuedRecord::Log(log) => log,
                    _ => unreachable!("pull_batch only groups matching kinds"),
                })
                .collect(),
        ),
        RecordKind::Metric => PulledBatch::Metrics(
            records
                .into_iter()
                .map(|record| match record {
                    QueuedRecord::Metric(metric) => metric,
                    _ => unreachable!("pull_batch only groups matching kinds"),
                })
                .collect(),
        ),
        RecordKind::Link => PulledBatch::Links(
            records
                .into_iter()
                .map(|record| match record {
                    QueuedRecord::Link(link) => link,
                    _ => unreachable!("pull_batch only groups matching kinds"),
                })
                .collect(),
        ),
    }
}

/// Per-severity survivable window: the shorter of the configured horizon and
/// how long the space actually available to this lane lasts at its current
/// ingest rate. "Available" assumes every strictly-higher lane keeps growing
/// (they always win eviction), so it is the worst-case, not the average.
pub(super) fn survivable_seconds(
    lane_lens: &[usize; SEVERITY_BUCKETS],
    ingest_rates: &[f64; SEVERITY_BUCKETS],
    capacity: usize,
    horizons: &RetentionHorizons,
    bucket: usize,
) -> f64 {
    let occupied_by_higher: usize = lane_lens[(bucket + 1)..SEVERITY_BUCKETS].iter().sum();
    let available = capacity.saturating_sub(occupied_by_higher);
    let horizon_secs = horizons.for_bucket(bucket).as_secs_f64();
    let rate = ingest_rates[bucket];
    if rate <= 0.0 {
        return horizon_secs;
    }
    horizon_secs.min(available as f64 / rate)
}

pub(super) fn ingest_rates(
    counts: &[u64; SEVERITY_BUCKETS],
    elapsed: Duration,
) -> [f64; SEVERITY_BUCKETS] {
    let elapsed_secs = elapsed.as_secs_f64().max(f64::EPSILON);
    core::array::from_fn(|bucket| counts[bucket] as f64 / elapsed_secs)
}
