//! Retransmission queue (RFC 9293 §3.7, RFC 6298 §5).
//!
//! Holds descriptors for sent-but-unacknowledged segments in send order. The
//! payload bytes are NOT copied here — `payload_offset` locates them in the
//! caller's send buffer, so a retransmit is a zero-copy re-send of
//! `send_buf[offset..offset+len]`. No-alloc: a fixed `[RetxSegment; MAX]`.
//!
//! On ACK the queue prunes every fully-acknowledged segment from the front
//! (segments are ordered, so acks clear a prefix). A segment retransmitted more
//! than [`MaxRetransmit::ABANDON_THRESHOLD`] times signals the connection
//! should be torn down (RFC 9293 §3.7: give up after R2).

use super::seq::SeqNum;

/// Per-segment retransmit counter with the RFC 9293 §3.7 abandon ceiling.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MaxRetransmit(u8);

impl MaxRetransmit {
    /// Give-up threshold (RFC 9293 §3.7 R2; the common default is ~15 tries).
    pub const ABANDON_THRESHOLD: u8 = 15;

    #[must_use]
    pub const fn zero() -> Self {
        Self(0)
    }

    #[must_use]
    pub const fn count(self) -> u8 {
        self.0
    }

    #[must_use]
    const fn increment(self) -> Self {
        Self(self.0.saturating_add(1))
    }

    #[must_use]
    pub const fn should_abandon(self) -> bool {
        self.0 >= Self::ABANDON_THRESHOLD
    }
}

/// A sent-but-unacknowledged segment descriptor (no payload bytes — see
/// `payload_offset`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetxSegment {
    pub seq: SeqNum,
    pub len: u32,
    /// Byte offset of this segment's payload in the caller's send buffer.
    pub payload_offset: u32,
    pub retransmits: MaxRetransmit,
}

impl RetxSegment {
    #[must_use]
    pub const fn new(seq: SeqNum, len: u32, payload_offset: u32) -> Self {
        Self {
            seq,
            len,
            payload_offset,
            retransmits: MaxRetransmit::zero(),
        }
    }

    #[must_use]
    const fn end(&self) -> SeqNum {
        self.seq.wrapping_add(self.len)
    }
}

/// What a retransmit attempt on the oldest segment resolved to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RetransmitDecision {
    /// Re-send this segment; its retransmit counter was incremented.
    Resend(RetxSegment),
    /// The oldest segment hit the abandon threshold — tear the connection down.
    Abandon,
    /// Nothing outstanding.
    Empty,
}

/// Fixed-capacity retransmission queue, ordered oldest-first.
#[derive(Debug, Clone)]
pub struct RetxQueue<const MAX: usize> {
    segments: [RetxSegment; MAX],
    len: usize,
}

impl<const MAX: usize> RetxQueue<MAX> {
    #[must_use]
    pub const fn new() -> Self {
        let zero = RetxSegment::new(SeqNum(0), 0, 0);
        Self {
            segments: [zero; MAX],
            len: 0,
        }
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Enqueue a freshly sent segment. Returns `false` (and does not enqueue)
    /// when the queue is full — the caller must stop sending until ACKs drain it.
    #[must_use]
    pub fn push(&mut self, segment: RetxSegment) -> bool {
        if self.len == MAX {
            return false;
        }
        self.segments[self.len] = segment;
        self.len += 1;
        true
    }

    /// Prune every fully-acknowledged segment (end at or before `ack`) from the
    /// front. Returns the number of payload bytes pruned.
    pub fn on_ack(&mut self, ack: SeqNum) -> u32 {
        let mut pruned_to = 0;
        let mut bytes = 0;
        while pruned_to < self.len {
            let segment = self.segments[pruned_to];
            let end = segment.end();
            // fully acked when `end` is at or before `ack` in serial order.
            let fully_acked = end == ack || end.precedes(ack);
            if fully_acked {
                bytes += segment.len;
                pruned_to += 1;
            } else {
                break;
            }
        }
        if pruned_to > 0 {
            for slot in 0..(self.len - pruned_to) {
                self.segments[slot] = self.segments[slot + pruned_to];
            }
            self.len -= pruned_to;
        }
        bytes
    }

    /// Peek the oldest outstanding segment without altering it.
    #[must_use]
    pub fn oldest(&self) -> Option<RetxSegment> {
        (self.len > 0).then(|| self.segments[0])
    }

    /// Attempt to retransmit the oldest segment: increments its counter and
    /// returns it, or [`RetransmitDecision::Abandon`] once it crosses the
    /// give-up threshold.
    pub fn retransmit_oldest(&mut self) -> RetransmitDecision {
        if self.len == 0 {
            return RetransmitDecision::Empty;
        }
        let bumped = self.segments[0].retransmits.increment();
        if bumped.should_abandon() {
            return RetransmitDecision::Abandon;
        }
        self.segments[0].retransmits = bumped;
        RetransmitDecision::Resend(self.segments[0])
    }
}

impl<const MAX: usize> Default for RetxQueue<MAX> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Queued segment count never exceeds MAX; push beyond capacity returns
        /// `false` (the must-use capacity signal) and never panics.
        #[test]
        fn push_beyond_capacity_returns_false_never_panics(
            segs in prop::collection::vec((any::<u32>(), 1_u32..=65535, any::<u32>()), 0..20),
        ) {
            let mut queue = RetxQueue::<8>::new();
            for (seq, len, offset) in segs {
                let accepted = queue.push(RetxSegment::new(SeqNum(seq), len, offset));
                if !accepted {
                    prop_assert_eq!(
                        queue.len(),
                        8,
                        "push returned false but queue is not full"
                    );
                }
                prop_assert!(
                    queue.len() <= 8,
                    "queue length {} exceeded MAX=8", queue.len()
                );
            }
        }

        /// `on_ack` and `retransmit_oldest` never panic for arbitrary inputs.
        #[test]
        fn retx_queue_operations_never_panic(
            initial_segs in prop::collection::vec((any::<u32>(), 1_u32..=1000, any::<u32>()), 0..8),
            acks in prop::collection::vec(any::<u32>(), 0..8),
            retransmits in 0_usize..=20,
        ) {
            let mut queue = RetxQueue::<8>::new();
            for (seq, len, offset) in initial_segs {
                let _ = queue.push(RetxSegment::new(SeqNum(seq), len, offset));
            }
            for ack in acks {
                let _ = queue.on_ack(SeqNum(ack));
            }
            for _ in 0..retransmits {
                let _ = queue.retransmit_oldest();
            }
        }
    }

    fn seg(seq: u32, len: u32, offset: u32) -> RetxSegment {
        RetxSegment::new(SeqNum(seq), len, offset)
    }

    #[test]
    fn push_until_full_then_refuse() {
        let mut queue = RetxQueue::<2>::new();
        assert!(queue.push(seg(1000, 100, 0)));
        assert!(queue.push(seg(1100, 100, 100)));
        assert!(!queue.push(seg(1200, 100, 200)));
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn ack_prunes_fully_acked_prefix() {
        let mut queue = RetxQueue::<4>::new();
        assert!(queue.push(seg(1000, 100, 0)));
        assert!(queue.push(seg(1100, 100, 100)));
        assert!(queue.push(seg(1200, 100, 200)));
        // ack 1200 fully acks the first two segments (ends 1100, 1200).
        assert_eq!(queue.on_ack(SeqNum(1200)), 200);
        assert_eq!(queue.len(), 1);
        assert_eq!(queue.oldest(), Some(seg(1200, 100, 200)));
    }

    #[test]
    fn partial_ack_keeps_unacked_segment() {
        let mut queue = RetxQueue::<4>::new();
        assert!(queue.push(seg(1000, 100, 0)));
        assert!(queue.push(seg(1100, 100, 100)));
        // ack 1150 fully acks [1000,1100) only; [1100,1200) stays.
        assert_eq!(queue.on_ack(SeqNum(1150)), 100);
        assert_eq!(queue.len(), 1);
        assert_eq!(queue.oldest(), Some(seg(1100, 100, 100)));
    }

    #[test]
    fn retransmit_increments_counter() {
        let mut queue = RetxQueue::<2>::new();
        assert!(queue.push(seg(1000, 100, 0)));
        match queue.retransmit_oldest() {
            RetransmitDecision::Resend(segment) => assert_eq!(segment.retransmits.count(), 1),
            other => panic!("expected resend, got {other:?}"),
        }
    }

    #[test]
    fn retransmit_abandons_at_threshold() {
        let mut queue = RetxQueue::<2>::new();
        assert!(queue.push(seg(1000, 100, 0)));
        let mut last = RetransmitDecision::Empty;
        for _ in 0..MaxRetransmit::ABANDON_THRESHOLD + 1 {
            last = queue.retransmit_oldest();
        }
        assert_eq!(last, RetransmitDecision::Abandon);
    }

    #[test]
    fn retransmit_empty_queue_is_empty() {
        let mut queue = RetxQueue::<2>::new();
        assert_eq!(queue.retransmit_oldest(), RetransmitDecision::Empty);
    }

    #[test]
    fn zero_copy_payload_offset_survives_pruning() {
        let mut queue = RetxQueue::<4>::new();
        assert!(queue.push(seg(1000, 100, 4096)));
        assert!(queue.push(seg(1100, 100, 4196)));
        queue.on_ack(SeqNum(1100));
        // the surviving segment keeps its caller-buffer offset.
        assert_eq!(queue.oldest().expect("one left").payload_offset, 4196);
    }
}
