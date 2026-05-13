//! `BatchSource` — the multi-consumer (`&self`) owned-batch source.
//!
//! The algebra's third source flavor, filling the gap between the two
//! single-consumer sources:
//! - [`DrainSource`](crate::pipe::drain_source::DrainSource) — `&mut self`, zero-copy
//!   borrow-visitor (the `*DK` frame relay).
//! - [`FanIn`](crate::pipe::fan_in::FanIn)'s `UnpinPipe` sources — `&self`, owned async pull.
//! - **`BatchSource`** — `&self`, owned batch pull, **safe for many concurrent
//!   drainers**. A lock-free MPMC ring ([`proxima_core::ring::BoundedQueue`]) is
//!   the canonical impl: several threads can each pull a disjoint batch at once,
//!   which is what lets a drainer partition work across cores and lift the
//!   single-drainer ceiling. The single-consumer sources cannot express that —
//!   `&mut self` is one owner by construction.
//!
//! A consumer drives it by handing a scratch slice to fill: `drain_batch(&mut
//! [Item]) -> count`. No alloc, no waker, no `&mut self` — legal at T0.

use proxima_core::ring::{BoundedQueue, RingStorage};

/// A source drained in owned batches through a shared `&self`, so multiple
/// consumers may pull disjoint batches concurrently.
pub trait BatchSource {
    /// The owned item pulled from the source.
    type Item;

    /// Pull up to `out.len()` items FIFO into `out`, returning the count filled.
    /// Safe to call from several threads at once — each caller gets a disjoint
    /// batch (the impl linearises the dequeue).
    fn drain_batch(&self, out: &mut [Self::Item]) -> usize;

    /// A snapshot lower bound on the items available (a racing producer just
    /// defers to the next pull). Consumers size their scratch buffer from it.
    fn len(&self) -> usize;

    /// True when the snapshot shows nothing to pull.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<R: RingStorage> BatchSource for BoundedQueue<R> {
    type Item = R::Item;

    #[inline]
    fn drain_batch(&self, out: &mut [R::Item]) -> usize {
        self.drain_into(out)
    }

    fn len(&self) -> usize {
        BoundedQueue::len(self)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_core::ring::{FailMode, StaticBoundedQueue};

    #[test]
    fn bounded_queue_is_a_batch_source() {
        let queue = StaticBoundedQueue::<u32, 4>::new(FailMode::DropNewest);
        for value in 1..=3 {
            queue.enqueue(value);
        }
        // drive it through the trait, not the inherent method — proving the
        // recorder's drain can be generic over any BatchSource.
        fn drain_via_trait<Source: BatchSource<Item = u32>>(
            source: &Source,
            out: &mut [u32],
        ) -> usize {
            source.drain_batch(out)
        }
        assert_eq!(queue.len(), 3);
        assert!(!queue.is_empty());
        let mut out = [0u32; 8];
        assert_eq!(drain_via_trait(&queue, &mut out), 3);
        assert_eq!(&out[..3], &[1, 2, 3]);
        assert!(queue.is_empty());
    }
}
