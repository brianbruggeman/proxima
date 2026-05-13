//! `SinkFront<Item>` — the alloc-tier, cloneable, `Arc`-shared handle over the
//! generic [`crate::pipe::SinkFront`] engine (heap [`Ring`] storage).
//!
//! The engine — `emit` / `emit_lossless` / `drain_one` / lifecycle FSM / demand
//! flag — is tier-generic (the same code runs the no-alloc
//! [`crate::pipe::StaticSinkFront`]). This alias adds only what the alloc tier
//! needs: shared ownership. A producer and its drainer each hold a clone
//! (`Arc`), arm/disarm and emit/drain reach the engine through `Deref`.
//! Construction is a single `new(capacity, fail_mode)` — the demand flag
//! is intrinsic to the sink, so there is no separate controller handle.
//!
//! TIER: T1 — no_std + alloc (`Arc<SinkFront<Ring<Item>>>`). [`Admission`],
//! [`DropReason`], [`SinkLifecycle`], [`SinkCounters`] are re-exported from
//! the engine module.

use alloc::sync::Arc;
use core::ops::Deref;

use proxima_core::ring::FailMode;
use crate::pipe::sink_front::HeapSinkFront as Engine;

pub use crate::pipe::sink_front::{Admission, DropReason, SinkCounters, SinkLifecycle};

/// A cloneable, `Arc`-shared sync producer facade over a heap [`Ring`]. Clone it
/// to hand the drainer a handle; both reach the same engine. All behaviour
/// (`emit`, `emit_lossless`, `drain_one`, `arm`/`disarm`, `lifecycle`,
/// `is_quiescent`) is the engine's, reached through [`Deref`].
pub struct SinkFront<Item> {
    inner: Arc<Engine<Item>>,
}

impl<Item> SinkFront<Item> {
    /// A sink front (starts disarmed) holding at most `capacity` items (rounded
    /// up to a power of two). The consumer [`arm`](Engine::arm)s it once running.
    #[must_use]
    pub fn new(capacity: usize, fail_mode: FailMode) -> Self {
        Self {
            inner: Arc::new(Engine::new(capacity, fail_mode)),
        }
    }
}

impl<Item> Clone for SinkFront<Item> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<Item> Deref for SinkFront<Item> {
    type Target = Engine<Item>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn new_starts_dormant_and_arms() {
        let sink = SinkFront::<u32>::new(8, FailMode::DropOldest);
        assert_eq!(sink.lifecycle(), SinkLifecycle::Dormant);
        assert_eq!(sink.emit(42), Admission::Dormant);
        sink.arm();
        assert_eq!(sink.lifecycle(), SinkLifecycle::Accepting);
        assert_eq!(sink.emit(7), Admission::Accepted);
        assert_eq!(sink.drain_one(), Some(7));
        assert!(sink.is_quiescent());
    }

    // the Arc handle is shared: a producer clone and a drainer clone reach the
    // same engine — emit on one is visible to drain on the other.
    #[test]
    fn clone_shares_the_same_engine() {
        let producer = SinkFront::<u32>::new(8, FailMode::DropOldest);
        producer.arm();
        let drainer = producer.clone();
        assert_eq!(producer.emit(99), Admission::Accepted);
        assert_eq!(
            drainer.drain_one(),
            Some(99),
            "clone sees the producer's item"
        );
        assert!(drainer.is_quiescent());
    }

    // producer-assist through the alloc handle: the assist closure gets the
    // engine (Deref target) and drains to make room, so nothing is dropped.
    #[test]
    fn emit_lossless_through_the_handle() {
        let sink = SinkFront::<u32>::new(2, FailMode::FailClosed);
        sink.arm();
        assert_eq!(sink.emit(1), Admission::Accepted);
        assert_eq!(sink.emit(2), Admission::Accepted);
        let admission = sink.emit_lossless(3, |engine| engine.drain_one().is_some());
        assert_eq!(admission, Admission::Accepted);
        assert_eq!(sink.dropped(), 0);
    }
}
