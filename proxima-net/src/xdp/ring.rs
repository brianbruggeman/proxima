//! Sans-IO index arithmetic for the four AF_XDP rings.
//!
//! Each AF_XDP ring is a single-producer/single-consumer queue whose `producer`
//! and `consumer` counters live in mmap'd memory shared with the kernel. These
//! two types own only the *cached* copies of those counters plus the modular
//! index math, so the whole reserve/commit/peek/release protocol is pure and
//! unit-testable without a socket. The linux datapath layers atomic loads/stores
//! of the shared counters and the descriptor-array writes on top.
//!
//! The counters are free-running `u32`s that wrap; the number of outstanding
//! entries is `producer - consumer` under wrapping subtraction, which stays
//! correct across the `u32::MAX` boundary because the outstanding count never
//! exceeds the ring size.

use super::error::XdpError;

/// Producer side of a ring we fill (the FILL and TX rings). We advance
/// `producer`; the kernel advances `consumer`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProducerIndex {
    size: u32,
    mask: u32,
    cached_producer: u32,
    cached_consumer: u32,
}

impl ProducerIndex {
    /// Create the bookkeeping for a ring of `size` entries.
    ///
    /// # Errors
    /// [`XdpError::RingSizeNotPowerOfTwo`] unless `size` is a non-zero power of two.
    pub fn new(size: u32) -> Result<Self, XdpError> {
        require_power_of_two(size)?;
        Ok(Self {
            size,
            mask: size - 1,
            cached_producer: 0,
            cached_consumer: 0,
        })
    }

    /// Reserve `want` contiguous slots, returning the first slot's (unmasked)
    /// index. `live_consumer` is the current shared consumer counter, read only
    /// if the cached view looks full — the kernel may have drained since. Returns
    /// `None` if fewer than `want` slots are free even after the refresh.
    pub fn reserve(&mut self, want: u32, live_consumer: u32) -> Option<u32> {
        if self.free() < want {
            self.cached_consumer = live_consumer;
            if self.free() < want {
                return None;
            }
        }
        let start = self.cached_producer;
        self.cached_producer = self.cached_producer.wrapping_add(want);
        Some(start)
    }

    /// Reserve up to `want` contiguous slots for a batch, returning the first
    /// slot's (unmasked) index and how many were granted — `min(want, free)`
    /// after a live-consumer refresh. Unlike [`reserve`](Self::reserve) this
    /// grants a partial batch instead of all-or-nothing, so one atomic commit
    /// publishes a whole burst.
    pub fn reserve_up_to(&mut self, want: u32, live_consumer: u32) -> (u32, u32) {
        if self.free() < want {
            self.cached_consumer = live_consumer;
        }
        let granted = want.min(self.free());
        let start = self.cached_producer;
        self.cached_producer = self.cached_producer.wrapping_add(granted);
        (start, granted)
    }

    /// The producer counter to publish into shared memory after filling the
    /// reserved slots, making them visible to the kernel.
    #[must_use]
    pub fn commit(&self) -> u32 {
        self.cached_producer
    }

    /// Map an unmasked index to its slot in the descriptor array.
    #[must_use]
    pub fn slot(&self, index: u32) -> usize {
        (index & self.mask) as usize
    }

    fn free(&self) -> u32 {
        self.size - self.cached_producer.wrapping_sub(self.cached_consumer)
    }
}

/// Consumer side of a ring we drain (the RX and COMPLETION rings). The kernel
/// advances `producer`; we advance `consumer`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsumerIndex {
    mask: u32,
    cached_producer: u32,
    cached_consumer: u32,
}

impl ConsumerIndex {
    /// Create the bookkeeping for a ring of `size` entries.
    ///
    /// # Errors
    /// [`XdpError::RingSizeNotPowerOfTwo`] unless `size` is a non-zero power of two.
    pub fn new(size: u32) -> Result<Self, XdpError> {
        require_power_of_two(size)?;
        Ok(Self {
            mask: size - 1,
            cached_producer: 0,
            cached_consumer: 0,
        })
    }

    /// Peek up to `want` ready entries, returning the first entry's (unmasked)
    /// index and how many are ready. `live_producer` is the current shared
    /// producer counter, read only if the cached view looks empty. The entries
    /// are not consumed until [`ConsumerIndex::release`].
    pub fn peek(&mut self, want: u32, live_producer: u32) -> (u32, u32) {
        let mut available = self.available();
        if available < want {
            self.cached_producer = live_producer;
            available = self.available();
        }
        (self.cached_consumer, want.min(available))
    }

    /// Consume `count` entries and return the consumer counter to publish into
    /// shared memory, handing those frames back to the kernel.
    pub fn release(&mut self, count: u32) -> u32 {
        self.cached_consumer = self.cached_consumer.wrapping_add(count);
        self.cached_consumer
    }

    /// Map an unmasked index to its slot in the descriptor array.
    #[must_use]
    pub fn slot(&self, index: u32) -> usize {
        (index & self.mask) as usize
    }

    fn available(&self) -> u32 {
        self.cached_producer.wrapping_sub(self.cached_consumer)
    }
}

fn require_power_of_two(size: u32) -> Result<(), XdpError> {
    if size == 0 || (size & (size - 1)) != 0 {
        return Err(XdpError::RingSizeNotPowerOfTwo(size));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]
    use super::*;

    #[test]
    fn new_rejects_non_power_of_two() {
        assert!(ProducerIndex::new(0).is_err());
        assert!(ProducerIndex::new(3).is_err());
        assert!(ConsumerIndex::new(6).is_err());
        assert!(ProducerIndex::new(2048).is_ok());
    }

    #[test]
    fn reserve_hands_out_contiguous_slots_from_zero() {
        let mut producer = ProducerIndex::new(8).expect("power of two");
        let start = producer.reserve(3, 0).expect("room on an empty ring");
        assert_eq!(start, 0);
        assert_eq!(producer.slot(start), 0);
        assert_eq!(producer.slot(start + 2), 2);
        assert_eq!(
            producer.commit(),
            3,
            "producer counter advanced by the reservation"
        );
    }

    #[test]
    fn producer_blocks_at_capacity_until_consumer_drains() {
        let mut producer = ProducerIndex::new(4).expect("power of two");
        assert!(producer.reserve(4, 0).is_some(), "fill the ring");
        assert!(
            producer.reserve(1, 0).is_none(),
            "full ring with an unchanged consumer rejects"
        );
        assert!(
            producer.reserve(1, 4).is_some(),
            "a drained consumer frees a slot"
        );
    }

    #[test]
    fn consumer_peek_is_non_destructive_until_release() {
        let mut consumer = ConsumerIndex::new(8).expect("power of two");
        let (start, ready) = consumer.peek(8, 5);
        assert_eq!(
            (start, ready),
            (0, 5),
            "sees the kernel's 5 produced entries"
        );
        let (again, still) = consumer.peek(8, 5);
        assert_eq!((again, still), (0, 5), "peek did not consume");
        assert_eq!(
            consumer.release(5),
            5,
            "release advances the consumer counter"
        );
        assert_eq!(consumer.peek(8, 5), (5, 0), "nothing left after release");
    }

    #[test]
    fn peek_clamps_to_available() {
        let mut consumer = ConsumerIndex::new(16).expect("power of two");
        assert_eq!(consumer.peek(16, 3), (0, 3), "asked 16, only 3 are ready");
    }

    #[test]
    fn roundtrip_wraps_slots_past_ring_size() {
        let mut producer = ProducerIndex::new(4).expect("power of two");
        let mut consumer = ConsumerIndex::new(4).expect("power of two");

        assert_eq!(producer.reserve(4, 0), Some(0));
        let (start, ready) = consumer.peek(4, producer.commit());
        assert_eq!((start, ready), (0, 4));
        consumer.release(4);

        let start = producer
            .reserve(4, consumer.release(0))
            .expect("consumer drained");
        assert_eq!(start, 4, "counter keeps climbing");
        assert_eq!(
            producer.slot(start),
            0,
            "but the slot wrapped back to the ring head"
        );
        assert_eq!(producer.slot(start + 3), 3);
    }

    #[test]
    fn outstanding_count_survives_u32_boundary() {
        let near_max = u32::MAX - 1;
        let mut producer = ProducerIndex {
            size: 4,
            mask: 3,
            cached_producer: near_max,
            cached_consumer: near_max,
        };
        let start = producer
            .reserve(3, near_max)
            .expect("empty ring near the wrap");
        assert_eq!(start, near_max);
        assert_eq!(producer.slot(start), (near_max & 3) as usize);
        assert_eq!(
            producer.slot(start.wrapping_add(2)),
            0,
            "third slot wraps across u32::MAX"
        );
        assert_eq!(
            producer.free(),
            1,
            "one slot left, computed under wrapping subtraction"
        );
        assert_eq!(
            producer.commit(),
            1,
            "producer counter wrapped past u32::MAX"
        );
    }

    #[test]
    fn batched_reserve_up_to_and_peek_wrap_past_ring_size() {
        let mut producer = ProducerIndex::new(4).expect("power of two");
        let mut consumer = ConsumerIndex::new(4).expect("power of two");

        // a batch bigger than the ring clamps to the free space (partial grant),
        // and one commit publishes the whole batch.
        let (start, granted) = producer.reserve_up_to(6, 0);
        assert_eq!(
            (start, granted),
            (0, 4),
            "reserve_up_to clamps to free space"
        );
        assert_eq!(producer.commit(), 4, "one commit for the whole batch");

        // the consumer peeks the batch and releases it in one step.
        let (peek_start, ready) = consumer.peek(6, producer.commit());
        assert_eq!((peek_start, ready), (0, 4), "peek clamps to available");
        assert_eq!(consumer.release(4), 4, "one release for the whole batch");

        // drained: a fresh batch keeps the counter climbing but wraps its slots
        // back to the ring head.
        let (start, granted) = producer.reserve_up_to(3, consumer.release(0));
        assert_eq!(
            (start, granted),
            (4, 3),
            "counter climbs past the ring size"
        );
        assert_eq!(producer.slot(start), 0, "slot wrapped back to the head");
        assert_eq!(producer.slot(start.wrapping_add(2)), 2);

        // a second batch on the same cycle gets only the last free slot.
        let (start, granted) = producer.reserve_up_to(3, consumer.release(0));
        assert_eq!(granted, 1, "one slot left after the 3-slot batch");
        assert_eq!(producer.slot(start), 3);
    }
}
