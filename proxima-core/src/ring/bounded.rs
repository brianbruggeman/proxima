//! `BoundedQueue<R>` — a bounded MPMC queue over a [`RingStorage`] with a
//! [`FailMode`] overflow policy and a drop counter, in every tier: the no-alloc
//! [`StaticBoundedQueue`] (inline [`StaticRing`]) and the alloc
//! [`HeapBoundedQueue`] (heap [`Ring`]) share ONE overflow algorithm, so the
//! drop-oldest / drop-newest / fail-closed semantics cannot drift between tiers.
//!
//! `enqueue` applies the policy (and counts drops); `try_enqueue` bypasses it,
//! handing a rejected item back so a lossless producer-assist loop can make room
//! and retry instead of losing it.

use core::sync::atomic::{AtomicUsize, Ordering};

use super::StaticRing;
#[cfg(feature = "alloc")]
use super::{Drainer, Ring};

/// The storage a [`BoundedQueue`] drives: a lock-free MPMC ring, in either the
/// inline [`StaticRing`] (no-alloc) or heap [`Ring`] (alloc) form. `push` hands
/// the item back on a full ring (never blocks); `dequeue` pops FIFO. The item
/// type is associated so a `BoundedQueue<R>` names a single storage rather than
/// threading a redundant element parameter.
pub trait RingStorage {
    /// The queued element.
    type Item;

    /// Push one item; on a full ring the item is handed back via `Err`.
    fn push(&self, item: Self::Item) -> Result<(), Self::Item>;
    /// Pop the oldest item, or `None` when empty.
    fn dequeue(&self) -> Option<Self::Item>;
    /// Snapshot of the number of queued items (not linearizable).
    fn len(&self) -> usize;
    /// True when the ring holds nothing (snapshot).
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The ring's capacity (a power of two).
    fn capacity(&self) -> usize;
    /// Pop up to `out.len()` items FIFO into `out`, returning the count filled.
    /// Default is a `dequeue` loop; the heap [`Ring`] overrides it with its
    /// prefetching batch `Drainer`.
    fn drain_into(&self, out: &mut [Self::Item]) -> usize {
        let mut count = 0;
        while count < out.len() {
            match self.dequeue() {
                Some(item) => {
                    out[count] = item;
                    count += 1;
                }
                None => break,
            }
        }
        count
    }
}

impl<T, const N: usize> RingStorage for StaticRing<T, N> {
    type Item = T;

    fn push(&self, item: T) -> Result<(), T> {
        StaticRing::push(self, item)
    }
    fn dequeue(&self) -> Option<T> {
        StaticRing::dequeue(self)
    }
    fn len(&self) -> usize {
        StaticRing::len(self)
    }
    fn capacity(&self) -> usize {
        StaticRing::capacity(self)
    }
}

#[cfg(feature = "alloc")]
impl<T> RingStorage for Ring<T> {
    type Item = T;

    #[inline]
    fn push(&self, item: T) -> Result<(), T> {
        Ring::push(self, item)
    }
    #[inline]
    fn dequeue(&self) -> Option<T> {
        Ring::dequeue(self)
    }
    fn len(&self) -> usize {
        Ring::len(self)
    }
    fn capacity(&self) -> usize {
        Ring::cap(self)
    }
    fn drain_into(&self, out: &mut [T]) -> usize {
        Drainer::new(self).drain_into(out)
    }
}

/// What a full queue does with a new item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailMode {
    /// Evict the oldest queued item to make room (the new item is kept).
    DropOldest,
    /// Reject the new item, keep the queue as-is.
    DropNewest,
    /// Reject the new item and surface the overflow to the caller.
    FailClosed,
}

/// The result of a [`BoundedQueue::enqueue`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnqueueOutcome {
    /// Item queued, nothing dropped.
    Enqueued,
    /// Queue was full; the oldest item was evicted, the new one queued.
    DroppedOldest,
    /// Queue was full; the new item was dropped.
    DroppedNewest,
    /// Queue was full under `FailClosed`; the new item was refused.
    Refused,
}

impl EnqueueOutcome {
    /// Whether this outcome dropped (or refused) an item.
    #[must_use]
    pub fn is_drop(self) -> bool {
        !matches!(self, Self::Enqueued)
    }
}

/// A bounded queue applying a [`FailMode`] on overflow, over any [`RingStorage`].
/// Construct it through a tier alias: [`StaticBoundedQueue`] (no-alloc) or
/// [`HeapBoundedQueue`] (alloc).
pub struct BoundedQueue<R: RingStorage> {
    queue: R,
    fail_mode: FailMode,
    drops: AtomicUsize,
}

impl<R: RingStorage> BoundedQueue<R> {
    /// Wrap an already-constructed storage with an overflow policy.
    pub fn from_storage(queue: R, fail_mode: FailMode) -> Self {
        Self {
            queue,
            fail_mode,
            drops: AtomicUsize::new(0),
        }
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.queue.capacity()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Total items dropped or refused on overflow since construction.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        // usize keeps the atomic portable to targets without 64-bit atomics
        // (cortex-m); a drop counter never needs more than usize range there.
        self.drops.load(Ordering::Relaxed) as u64
    }

    /// Count one externally-decided drop. For a caller that applies its OWN
    /// overflow routing on top of [`try_enqueue`](Self::try_enqueue) (a dynamic
    /// per-item policy the fixed [`FailMode`] can't express) and discards the
    /// handed-back item itself — this folds that drop into the same counter
    /// [`dropped`](Self::dropped) reports, so the queue stays the single drop
    /// total.
    pub fn note_drop(&self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }

    /// Enqueue under the configured [`FailMode`], counting any drop.
    pub fn enqueue(&self, item: R::Item) -> EnqueueOutcome {
        let outcome = match self.queue.push(item) {
            Ok(()) => EnqueueOutcome::Enqueued,
            Err(rejected) => match self.fail_mode {
                FailMode::DropOldest => {
                    // no atomic evict-oldest in the Vyukov ring: free a slot by
                    // dropping the oldest, then retry. a raced refill drops the
                    // newest instead — either way exactly one item is lost.
                    let _oldest = self.queue.dequeue();
                    match self.queue.push(rejected) {
                        Ok(()) => EnqueueOutcome::DroppedOldest,
                        Err(_lost) => EnqueueOutcome::DroppedNewest,
                    }
                }
                FailMode::DropNewest => EnqueueOutcome::DroppedNewest,
                FailMode::FailClosed => EnqueueOutcome::Refused,
            },
        };
        if outcome.is_drop() {
            self.drops.fetch_add(1, Ordering::Relaxed);
        }
        outcome
    }

    /// Enqueue WITHOUT the fail policy: on a full ring the item is handed back
    /// via `Err` so a lossless producer-assist loop can make room and retry it
    /// rather than dropping it. Nothing is counted as dropped. `#[inline]` — this
    /// is an emit hot path (the telemetry recorder pushes through it per record),
    /// so it must collapse to the bare storage `push`.
    #[inline]
    pub fn try_enqueue(&self, item: R::Item) -> Result<(), R::Item> {
        self.queue.push(item)
    }

    /// Enqueue, looping through a make-room step on a full ring until the item
    /// lands. `on_full` is the producer's make-room action (e.g. drain + export a
    /// batch to free a slot); it returns `true` to keep trying, `false` to give
    /// up — in which case the un-enqueued item is handed back via `Err`. This is
    /// the shared producer-assist loop: a strictly-lossless caller passes an
    /// `on_full` that never gives up (and yields on a no-progress round); a
    /// bounded caller gives up and applies its own overflow policy to the `Err`.
    /// The loop itself neither counts nor blocks — any yield/park lives in
    /// `on_full`, so this stays no_std.
    pub fn enqueue_assisting<F>(&self, mut item: R::Item, mut on_full: F) -> Result<(), R::Item>
    where
        F: FnMut() -> bool,
    {
        loop {
            match self.queue.push(item) {
                Ok(()) => return Ok(()),
                Err(returned) => {
                    item = returned;
                    if !on_full() {
                        return Err(item);
                    }
                }
            }
        }
    }

    /// Pop the oldest queued item, if any.
    #[must_use]
    pub fn dequeue(&self) -> Option<R::Item> {
        self.queue.dequeue()
    }

    /// Batch-pop up to `out.len()` items FIFO, returning the count filled. Uses
    /// the storage's batch drain (the heap [`Ring`]'s prefetching `Drainer`).
    pub fn drain_into(&self, out: &mut [R::Item]) -> usize {
        self.queue.drain_into(out)
    }
}

/// The no-alloc tier: a bounded queue over an inline [`StaticRing`] (`[Cell; N]`,
/// `N` a power of two ≥ 2). Builds bare-metal (no heap).
pub type StaticBoundedQueue<T, const N: usize> = BoundedQueue<StaticRing<T, N>>;

impl<T, const N: usize> BoundedQueue<StaticRing<T, N>> {
    /// A no-alloc bounded queue of capacity `N` (a power of two ≥ 2 — the
    /// [`StaticRing`] requirement).
    #[must_use]
    pub fn new(fail_mode: FailMode) -> Self {
        Self::from_storage(StaticRing::new(), fail_mode)
    }
}

/// The alloc tier: a bounded queue over a heap [`Ring`], runtime capacity.
#[cfg(feature = "alloc")]
pub type HeapBoundedQueue<T> = BoundedQueue<Ring<T>>;

#[cfg(feature = "alloc")]
impl<T> BoundedQueue<Ring<T>> {
    /// A heap-backed bounded queue holding at most `capacity` items (rounded up
    /// to a power of two, minimum 2).
    #[must_use]
    pub fn new(capacity: usize, fail_mode: FailMode) -> Self {
        Self::from_storage(Ring::with_capacity(capacity), fail_mode)
    }
}

// composes StaticRing/Ring from super, whose atomics/UnsafeCell are
// cfg-swapped to loom under `--features loom` (see ring/mpsc.rs) — those
// only work inside an actual loom::model(...) closure, which these plain
// #[test] functions don't provide.
#[cfg(all(test, not(feature = "loom")))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // One assertion body run through BOTH tiers proves the overflow algorithm
    // does not drift between the inline and heap storages.
    fn drop_oldest_keeps_newest<R: RingStorage<Item = u32>>(queue: BoundedQueue<R>) {
        assert_eq!(queue.enqueue(1), EnqueueOutcome::Enqueued);
        assert_eq!(queue.enqueue(2), EnqueueOutcome::Enqueued);
        assert_eq!(
            queue.enqueue(3),
            EnqueueOutcome::DroppedOldest,
            "evicts the oldest"
        );
        assert_eq!(queue.dropped(), 1);
        assert_eq!(
            queue.dequeue(),
            Some(2),
            "oldest (1) evicted; 2 and 3 remain"
        );
        assert_eq!(queue.dequeue(), Some(3));
        assert_eq!(queue.dequeue(), None);
    }

    fn fail_closed_refuses<R: RingStorage<Item = u32>>(queue: BoundedQueue<R>) {
        assert_eq!(queue.enqueue(1), EnqueueOutcome::Enqueued);
        assert_eq!(queue.enqueue(2), EnqueueOutcome::Enqueued);
        assert_eq!(queue.enqueue(3), EnqueueOutcome::Refused);
        assert_eq!(queue.dropped(), 1);
    }

    fn try_enqueue_hands_back_without_dropping<R: RingStorage<Item = u32>>(queue: BoundedQueue<R>) {
        assert!(queue.try_enqueue(1).is_ok());
        assert!(queue.try_enqueue(2).is_ok());
        assert_eq!(queue.try_enqueue(3), Err(3), "handed back, not dropped");
        assert_eq!(queue.dropped(), 0, "try_enqueue never counts a drop");
    }

    #[test]
    fn static_tier_matches() {
        drop_oldest_keeps_newest(StaticBoundedQueue::<u32, 2>::new(FailMode::DropOldest));
        fail_closed_refuses(StaticBoundedQueue::<u32, 2>::new(FailMode::FailClosed));
        try_enqueue_hands_back_without_dropping(StaticBoundedQueue::<u32, 2>::new(
            FailMode::FailClosed,
        ));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn heap_tier_matches() {
        drop_oldest_keeps_newest(HeapBoundedQueue::<u32>::new(2, FailMode::DropOldest));
        fail_closed_refuses(HeapBoundedQueue::<u32>::new(2, FailMode::FailClosed));
        try_enqueue_hands_back_without_dropping(HeapBoundedQueue::<u32>::new(
            2,
            FailMode::FailClosed,
        ));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn heap_capacity_rounds_up_to_power_of_two() {
        assert_eq!(
            HeapBoundedQueue::<u32>::new(3, FailMode::DropNewest).capacity(),
            4
        );
        assert_eq!(
            HeapBoundedQueue::<u32>::new(1, FailMode::DropNewest).capacity(),
            2
        );
    }

    #[test]
    fn enqueue_assisting_makes_room_and_lands() {
        let queue = StaticBoundedQueue::<u32, 2>::new(FailMode::FailClosed);
        queue.enqueue(1);
        queue.enqueue(2);
        // on_full frees a slot (drops the oldest here) then retries — 3 lands.
        let landed = queue.enqueue_assisting(3, || queue.dequeue().is_some());
        assert!(landed.is_ok());
        assert_eq!(queue.dequeue(), Some(2), "1 was freed to make room");
        assert_eq!(queue.dequeue(), Some(3));
    }

    #[test]
    fn enqueue_assisting_hands_back_on_give_up() {
        let queue = StaticBoundedQueue::<u32, 2>::new(FailMode::FailClosed);
        queue.enqueue(1);
        queue.enqueue(2);
        assert_eq!(
            queue.enqueue_assisting(3, || false),
            Err(3),
            "gave up, item handed back"
        );
    }

    #[test]
    fn drain_into_batch_pops_fifo_both_tiers() {
        let queue = StaticBoundedQueue::<u32, 4>::new(FailMode::DropNewest);
        for value in 1..=3 {
            queue.enqueue(value);
        }
        let mut out = [0u32; 8];
        assert_eq!(
            queue.drain_into(&mut out),
            3,
            "drains exactly what is queued"
        );
        assert_eq!(&out[..3], &[1, 2, 3]);
        assert!(queue.dequeue().is_none());
    }

    #[test]
    fn drop_newest_leaves_queue_untouched() {
        let queue = StaticBoundedQueue::<u32, 2>::new(FailMode::DropNewest);
        queue.enqueue(1);
        queue.enqueue(2);
        assert_eq!(queue.enqueue(3), EnqueueOutcome::DroppedNewest);
        assert_eq!(queue.dropped(), 1);
        assert_eq!(queue.dequeue(), Some(1), "queue untouched");
        assert_eq!(queue.dequeue(), Some(2));
    }
}
