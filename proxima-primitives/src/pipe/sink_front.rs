//! `SinkFront<R>` — the owned-push producer facade + explicit lifecycle FSM,
//! generic over a [`RingStorage`] so one engine serves every tier: the no-alloc
//! [`StaticSinkFront`] (inline [`StaticRing`]) and the alloc heap form share ONE
//! emit / drain / lifecycle implementation.
//!
//! `emit` is sync, never awaited, never blocks: it checks the demand flag and
//! enqueues into a [`BoundedQueue`], returning an [`Admission`]. `emit_lossless`
//! adds producer-assist — on a full ring it calls a caller-supplied sync `assist`
//! (drain + export) to make room and retries, so nothing is dropped while the
//! consumer can keep up; no clock, no park, so it is legal at T0. The drain
//! (dequeue → exporter) is the consumer's runtime concern (it holds the front,
//! calls [`drain_one`](SinkFront::drain_one)) — no drain worker, no sleep-poll,
//! is baked in here.
//!
//! The demand flag is an inline [`AtomicBool`], not an `Arc`-shared gate: it is
//! intrinsic to the sink's own lifecycle (a sink nobody drains stays `Dormant`),
//! and staying inline is what keeps the whole type no-alloc. The alloc tier gets
//! shared ownership by wrapping the whole `SinkFront` in an `Arc`, not by
//! Arc-wrapping the flag.
//!
//! [`SinkCounters`] uses `portable_atomic::AtomicU64` rather than
//! `core::sync::atomic::AtomicU64`: targets without native 64-bit atomics
//! (thumbv7em-none-eabihf, thumbv7m-none-eabi) don't have the latter at all,
//! which would block this no-alloc leaf from building bare-metal. On targets
//! that DO have native 64-bit atomics (every host tier), `portable_atomic`
//! resolves to the native instruction — zero cost, same layout.

use core::sync::atomic::{AtomicBool, Ordering};

use portable_atomic::AtomicU64;

#[cfg(feature = "alloc")]
use proxima_core::ring::Ring;
use proxima_core::ring::{BoundedQueue, EnqueueOutcome, FailMode, RingStorage, StaticRing};

/// The sync result of a [`SinkFront::emit`]. Never a `Future`, never a `Result`
/// error — a drop is policy, not a fault. Three variants, not two: the wake
/// decision needs `Dormant` (nothing enqueued → don't wake the drainer) distinct
/// from `Dropped` (built then shed). See [`Admission::leaves_item_queued`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum Admission {
    /// Item entered the queue; the drainer will deliver it.
    Accepted,
    /// Demand closed (no consumer) — nothing was enqueued.
    Dormant,
    /// Item was shed under the overflow policy.
    Dropped(DropReason),
}

/// Why an item was shed on a full queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DropReason {
    /// Queue was full; the oldest item was evicted and THIS item queued.
    OldestEvicted,
    /// Queue was full; THIS item was discarded; queue unchanged.
    NewestDiscarded,
    /// `FailClosed`; THIS item refused; queue unchanged.
    Refused,
}

impl Admission {
    #[must_use]
    pub fn is_accepted(self) -> bool {
        matches!(self, Self::Accepted)
    }

    /// Whether this outcome left a *new* item in the queue (so a drainer woken on
    /// this outcome has work). `Accepted` and `OldestEvicted` queue an item;
    /// `Dormant`/`NewestDiscarded`/`Refused` leave the queue unchanged.
    #[must_use]
    pub fn leaves_item_queued(self) -> bool {
        matches!(
            self,
            Self::Accepted | Self::Dropped(DropReason::OldestEvicted)
        )
    }
}

fn admission_of(outcome: EnqueueOutcome) -> Admission {
    match outcome {
        EnqueueOutcome::Enqueued => Admission::Accepted,
        EnqueueOutcome::DroppedOldest => Admission::Dropped(DropReason::OldestEvicted),
        EnqueueOutcome::DroppedNewest => Admission::Dropped(DropReason::NewestDiscarded),
        EnqueueOutcome::Refused => Admission::Dropped(DropReason::Refused),
    }
}

/// The sink's lifecycle, as an explicit exhaustive enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SinkLifecycle {
    /// Demand closed; items are declined (`Admission::Dormant`), nothing queued.
    Dormant,
    /// Demand open, queue has room — items are accepted.
    Accepting,
    /// Demand open, queue at capacity — new items hit the overflow policy.
    Backpressured,
}

/// Monotonic append/drain counters with the quiescence invariant as a NAMED
/// predicate ([`SinkCounters::is_quiescent`]).
pub struct SinkCounters {
    appended: AtomicU64,
    drained: AtomicU64,
}

impl Default for SinkCounters {
    fn default() -> Self {
        Self::new()
    }
}

impl SinkCounters {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            appended: AtomicU64::new(0),
            drained: AtomicU64::new(0),
        }
    }

    pub fn record_append(&self) {
        self.appended.fetch_add(1, Ordering::SeqCst);
    }

    pub fn record_drain(&self) {
        self.drained.fetch_add(1, Ordering::SeqCst);
    }

    #[must_use]
    pub fn appended(&self) -> u64 {
        self.appended.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn drained(&self) -> u64 {
        self.drained.load(Ordering::Relaxed)
    }

    /// Every appended item is now drained or dropped: `appended == drained +
    /// drops` AND the queue is empty.
    #[must_use]
    pub fn is_quiescent(&self, queue_len: usize, total_drops: u64) -> bool {
        let appended = self.appended.load(Ordering::SeqCst);
        let drained = self.drained.load(Ordering::SeqCst);
        queue_len == 0 && drained + total_drops >= appended
    }
}

/// The sync producer facade over any [`RingStorage`]. Construct a tier through a
/// tier alias: [`StaticSinkFront`] (no-alloc) or the heap `new` (alloc). Shared
/// between producer and drainer by borrow (T0) or `Arc` (alloc).
pub struct SinkFront<R: RingStorage> {
    queue: BoundedQueue<R>,
    armed: AtomicBool,
    counters: SinkCounters,
}

impl<R: RingStorage> SinkFront<R> {
    /// Wrap an already-constructed storage; the sink starts `Dormant` (disarmed)
    /// until a consumer [`arm`](Self::arm)s it.
    pub fn from_storage(storage: R, fail_mode: FailMode) -> Self {
        Self {
            queue: BoundedQueue::from_storage(storage, fail_mode),
            armed: AtomicBool::new(false),
            counters: SinkCounters::new(),
        }
    }

    /// Signal that a consumer is draining — items are now accepted.
    pub fn arm(&self) {
        self.armed.store(true, Ordering::Release);
    }

    /// Signal that the consumer is gone — the sink goes `Dormant`.
    pub fn disarm(&self) {
        self.armed.store(false, Ordering::Release);
    }

    #[must_use]
    pub fn is_armed(&self) -> bool {
        self.armed.load(Ordering::Acquire)
    }

    /// Push one item. Sync, never awaited, never blocks. The demand flag is
    /// checked INSIDE `emit`, so there is no check-then-act TOCTOU window.
    pub fn emit(&self, item: R::Item) -> Admission {
        if !self.is_armed() {
            return Admission::Dormant;
        }
        self.counters.record_append();
        admission_of(self.queue.enqueue(item))
    }

    /// Lossless producer-assist: on a full ring, call `assist` (a sync closure
    /// that drains + exports to make room) and retry, so a producer that can be
    /// kept up with never loses an item. `assist` returns `false` to give up, at
    /// which point the configured [`FailMode`] applies (and counts the drop). No
    /// clock, no park — legal at T0. The retry bound lives in `assist`, not here.
    pub fn emit_lossless(&self, item: R::Item, mut assist: impl FnMut(&Self) -> bool) -> Admission {
        if !self.is_armed() {
            return Admission::Dormant;
        }
        self.counters.record_append();
        // the producer-assist loop lives in the queue primitive; here `on_full`
        // is `assist(self)` (make room, `false` = give up). on give-up the item
        // comes back and the configured FailMode applies (counting the drop).
        match self.queue.enqueue_assisting(item, || assist(self)) {
            Ok(()) => Admission::Accepted,
            Err(item) => admission_of(self.queue.enqueue(item)),
        }
    }

    /// Dequeue one item for the drainer, accounting the drain. `None` when empty.
    #[must_use]
    pub fn drain_one(&self) -> Option<R::Item> {
        let item = self.queue.dequeue()?;
        self.counters.record_drain();
        Some(item)
    }

    /// Current lifecycle, computed from the demand flag + queue — not stored.
    #[must_use]
    pub fn lifecycle(&self) -> SinkLifecycle {
        if !self.is_armed() {
            SinkLifecycle::Dormant
        } else if self.queue.len() >= self.queue.capacity() {
            SinkLifecycle::Backpressured
        } else {
            SinkLifecycle::Accepting
        }
    }

    /// Every appended item drained or dropped (the explicit quiescence state).
    #[must_use]
    pub fn is_quiescent(&self) -> bool {
        self.counters
            .is_quiescent(self.queue.len(), self.queue.dropped())
    }

    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.queue.dropped()
    }

    #[must_use]
    pub fn drained(&self) -> u64 {
        self.counters.drained()
    }
}

/// The no-alloc tier: a sink front over an inline [`StaticRing`] (`N` a power of
/// two ≥ 2). Builds bare-metal.
pub type StaticSinkFront<T, const N: usize> = SinkFront<StaticRing<T, N>>;

impl<T, const N: usize> SinkFront<StaticRing<T, N>> {
    /// A no-alloc sink front of capacity `N`.
    #[must_use]
    pub fn new(fail_mode: FailMode) -> Self {
        Self::from_storage(StaticRing::new(), fail_mode)
    }
}

/// The alloc tier: a sink front over a heap [`Ring`], runtime capacity. Named as
/// an alias (like [`StaticSinkFront`]) so `new` resolves without ambiguity.
#[cfg(feature = "alloc")]
pub type HeapSinkFront<T> = SinkFront<Ring<T>>;

#[cfg(feature = "alloc")]
impl<T> SinkFront<Ring<T>> {
    /// A heap-backed sink front holding at most `capacity` items (rounded up to a
    /// power of two).
    #[must_use]
    pub fn new(capacity: usize, fail_mode: FailMode) -> Self {
        Self::from_storage(Ring::with_capacity(capacity), fail_mode)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn dormant_emit_never_queues() {
        let sink = StaticSinkFront::<u32, 8>::new(FailMode::DropOldest);
        assert_eq!(sink.lifecycle(), SinkLifecycle::Dormant);
        let outcome = sink.emit(42);
        assert_eq!(outcome, Admission::Dormant);
        assert!(!outcome.leaves_item_queued());
        assert!(sink.drain_one().is_none());
    }

    #[test]
    fn armed_emit_accepts_and_drains_to_quiescence() {
        let sink = StaticSinkFront::<u32, 8>::new(FailMode::DropOldest);
        sink.arm();
        assert_eq!(sink.lifecycle(), SinkLifecycle::Accepting);
        assert_eq!(sink.emit(7), Admission::Accepted);
        assert_eq!(sink.drain_one(), Some(7));
        assert!(sink.is_quiescent());
    }

    #[test]
    fn full_static_sink_is_backpressured_and_drop_oldest_keeps_item() {
        let sink = StaticSinkFront::<u32, 2>::new(FailMode::DropOldest);
        sink.arm();
        assert_eq!(sink.emit(1), Admission::Accepted);
        assert_eq!(sink.emit(2), Admission::Accepted);
        assert_eq!(sink.lifecycle(), SinkLifecycle::Backpressured);
        let evicted = sink.emit(3);
        assert_eq!(evicted, Admission::Dropped(DropReason::OldestEvicted));
        assert!(evicted.leaves_item_queued());
        assert_eq!(sink.drain_one(), Some(2));
        assert_eq!(sink.drain_one(), Some(3));
    }

    // producer-assist: a full ring is made room for by the assist closure
    // (draining one), so the item lands instead of dropping — zero loss.
    #[test]
    fn emit_lossless_drains_to_make_room_no_drop() {
        let sink = StaticSinkFront::<u32, 2>::new(FailMode::FailClosed);
        sink.arm();
        assert_eq!(sink.emit(1), Admission::Accepted);
        assert_eq!(sink.emit(2), Admission::Accepted);
        // ring full; assist drains one to free a slot, so 3 is accepted losslessly.
        let admission = sink.emit_lossless(3, |front| front.drain_one().is_some());
        assert_eq!(admission, Admission::Accepted);
        assert_eq!(sink.dropped(), 0, "producer-assist dropped nothing");
    }

    // when assist gives up, the configured FailMode applies and counts the drop.
    #[test]
    fn emit_lossless_falls_back_to_policy_when_assist_gives_up() {
        let sink = StaticSinkFront::<u32, 2>::new(FailMode::FailClosed);
        sink.arm();
        assert_eq!(sink.emit(1), Admission::Accepted);
        assert_eq!(sink.emit(2), Admission::Accepted);
        let admission = sink.emit_lossless(3, |_front| false);
        assert_eq!(admission, Admission::Dropped(DropReason::Refused));
        assert_eq!(sink.dropped(), 1);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn heap_tier_matches_static_behavior() {
        let sink = HeapSinkFront::<u32>::new(2, FailMode::DropNewest);
        sink.arm();
        assert_eq!(sink.emit(1), Admission::Accepted);
        assert_eq!(sink.emit(2), Admission::Accepted);
        assert_eq!(
            sink.emit(3),
            Admission::Dropped(DropReason::NewestDiscarded)
        );
        assert_eq!(sink.drain_one(), Some(1));
        assert_eq!(sink.drain_one(), Some(2));
    }
}
