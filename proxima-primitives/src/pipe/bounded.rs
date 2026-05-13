//! `BoundedQueue<T>` — the alloc-tier bounded queue, re-exported from
//! [`proxima_core::ring`].
//!
//! The type and its overflow algorithm moved to `proxima-core` so the same
//! `FailMode` policy drives both the heap [`Ring`](proxima_core::ring::Ring)
//! (this alias) and the no-alloc
//! [`StaticRing`](proxima_core::ring::StaticRing) tiers from one implementation.
//! `SinkFront` and the recording/telemetry sinks compose this alias unchanged.

pub use proxima_core::ring::{EnqueueOutcome, FailMode};

/// A heap-backed bounded queue with a [`FailMode`] overflow policy — the alloc
/// tier of [`proxima_core::ring::BoundedQueue`]. `BoundedQueue::new(capacity,
/// fail_mode)` is unchanged; the no-alloc peer is
/// [`proxima_core::ring::StaticBoundedQueue`].
pub type BoundedQueue<T> = proxima_core::ring::HeapBoundedQueue<T>;
