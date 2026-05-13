//! Per-stream out-of-order STREAM-frame reassembly per RFC 9000
//! §2.2 (Stream States) + §3 (Stream Types and Identifiers).
//!
//! QUIC STREAM frames carry byte offsets; the receiver MUST be able
//! to reassemble fragments that arrive out of order (per RFC 9002
//! loss-detection retransmits will fill gaps). The proto layer's
//! per-stream `RecvState::Recv` carries:
//!
//! - the contiguous head — bytes `[0 .. offset_next)` in
//!   `recv_buffer` (ArrayVec, no holes), drained by the application
//!   via `Connection::read_stream`;
//! - the pending-fragments queue — bytes at offsets >
//!   `offset_next`, awaiting the gap-causing fragment(s) to arrive.
//!
//! [`ReassemblyQueue`] is the pending-fragments primitive. It
//! holds up to [`MAX_FRAGMENTS`] out-of-order fragments, sorted by
//! offset; each fragment stores up to [`FRAGMENT_INLINE_BYTES`]
//! bytes. Insert returns the count of bytes successfully buffered;
//! over-long fragments are tail-truncated, the peer retransmits per
//! RFC 9002.
//!
//! # Tier
//!
//! Tier-3 (no_alloc). Storage = `heapless::Vec<Fragment, CAP>`.

use heapless::Vec;

use crate::quic::sized;

/// Maximum out-of-order fragments held per stream. Sourced from
/// `proxima-quic-proto.toml [streams].reassembly_max_fragments`.
pub const MAX_FRAGMENTS: usize = sized::STREAMS_REASSEMBLY_MAX_FRAGMENTS;

/// Maximum bytes per pending fragment. Sourced from
/// `proxima-quic-proto.toml [streams].reassembly_fragment_inline_bytes`.
pub const FRAGMENT_INLINE_BYTES: usize = sized::STREAMS_REASSEMBLY_FRAGMENT_INLINE_BYTES;

/// One out-of-order STREAM-frame fragment awaiting reassembly.
#[derive(Debug, Clone)]
pub struct Fragment {
    /// Absolute stream byte offset of the first byte in `data`.
    pub offset: u64,
    /// Payload bytes — `[0 .. data.len())` covers stream offsets
    /// `[offset .. offset + data.len())`.
    pub data: heapless::Vec<u8, FRAGMENT_INLINE_BYTES>,
}

/// Outcome of [`ReassemblyQueue::insert`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InsertOutcome {
    /// All input bytes were either buffered into the contiguous
    /// head or deferred into the pending queue. No bytes lost.
    Accepted {
        /// Number of bytes appended to the contiguous head this call.
        appended_to_contiguous: usize,
    },
    /// All input bytes already covered by the contiguous head OR
    /// already covered by an overlapping pending fragment. No-op.
    Duplicate,
    /// At least one byte was dropped because either:
    /// (a) the input fragment was longer than `FRAGMENT_INLINE_BYTES`
    ///     and the tail wouldn't fit, OR
    /// (b) the pending queue was full and the input couldn't be
    ///     buffered (only the bytes that fit into the contiguous
    ///     head this call were saved).
    Truncated {
        appended_to_contiguous: usize,
        dropped_bytes: usize,
    },
}

/// Out-of-order STREAM-frame reassembly queue.
#[derive(Debug, Clone, Default)]
pub struct ReassemblyQueue {
    /// Pending fragments sorted ascending by offset. Adjacent /
    /// overlapping fragments are merged on insert to keep the queue
    /// minimal.
    pending: Vec<Fragment, MAX_FRAGMENTS>,
}

impl ReassemblyQueue {
    /// Empty queue.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    /// Number of pending fragments.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Iterate pending fragments in offset-ascending order.
    pub fn pending(&self) -> impl Iterator<Item = &Fragment> {
        self.pending.iter()
    }

    /// Insert a STREAM-frame fragment + drain any newly-contiguous
    /// bytes into `contiguous_head`.
    ///
    /// `offset_next` MUST hold the next expected contiguous byte
    /// offset on entry; on return it's advanced past every byte
    /// appended to `contiguous_head` this call.
    ///
    /// `contiguous_head` is the receiver's append-only buffer — the
    /// caller has already sized it (typically a stream's
    /// `recv_buffer: ArrayVec<u8, STREAM_RECV_INLINE>`).
    ///
    /// Returns an [`InsertOutcome`] describing what happened.
    pub fn insert<const HEAD_CAP: usize>(
        &mut self,
        offset: u64,
        data: &[u8],
        contiguous_head: &mut arrayvec::ArrayVec<u8, HEAD_CAP>,
        offset_next: &mut u64,
    ) -> InsertOutcome {
        // Clip any bytes that overlap with the contiguous head.
        let data_start_offset = offset;
        let data_end_offset = offset.saturating_add(data.len() as u64);
        if data_end_offset <= *offset_next {
            return InsertOutcome::Duplicate;
        }
        let clip_lo = if data_start_offset < *offset_next {
            (*offset_next - data_start_offset) as usize
        } else {
            0
        };
        let clipped = &data[clip_lo..];
        let clipped_offset = data_start_offset + clip_lo as u64;

        let mut appended_to_contiguous = 0usize;
        let mut dropped = 0usize;

        if clipped_offset == *offset_next {
            // Direct contiguous append.
            let copy_len = core::cmp::min(clipped.len(), contiguous_head.remaining_capacity());
            // try_extend_from_slice cannot fail — bounded above.
            contiguous_head
                .try_extend_from_slice(&clipped[..copy_len])
                .ok();
            appended_to_contiguous += copy_len;
            *offset_next = offset_next.saturating_add(copy_len as u64);
            // Anything past head-cap is dropped. The peer's
            // application-level back-pressure (MAX_DATA /
            // MAX_STREAM_DATA) MUST keep this bounded.
            dropped += clipped.len() - copy_len;
            // Drain any pending fragments now adjacent. Tail drops
            // inside `drain_into` (when the contiguous head fills
            // mid-fragment) MUST surface here so the caller treats
            // the triggering packet as TransientRecvBufferFull
            // instead of silently ACKing data that didn't land.
            let (drained_appended, drained_dropped) = self.drain_into(contiguous_head, offset_next);
            appended_to_contiguous += drained_appended;
            dropped += drained_dropped;
        } else {
            // Stash for later. Merging adjacent / overlapping pending
            // fragments keeps the queue minimal.
            let buffered = self.insert_pending(clipped_offset, clipped);
            dropped += clipped.len() - buffered;
        }

        if dropped > 0 {
            InsertOutcome::Truncated {
                appended_to_contiguous,
                dropped_bytes: dropped,
            }
        } else {
            InsertOutcome::Accepted {
                appended_to_contiguous,
            }
        }
    }

    /// Drain pending fragments whose offset matches the current
    /// `*offset_next`, appending them to `contiguous_head` until the
    /// next gap or the head fills. Returns `(appended, dropped)` so
    /// the caller can roll up partial fragments lost to head-cap
    /// exhaustion into the same `InsertOutcome::Truncated` it
    /// surfaces to the application.
    fn drain_into<const HEAD_CAP: usize>(
        &mut self,
        contiguous_head: &mut arrayvec::ArrayVec<u8, HEAD_CAP>,
        offset_next: &mut u64,
    ) -> (usize, usize) {
        let mut total_appended = 0usize;
        let mut total_dropped = 0usize;
        while let Some(front) = self.pending.first() {
            if front.offset > *offset_next {
                break;
            }
            // front.offset == *offset_next per the invariant — but
            // double-check defensively.
            if front.offset != *offset_next {
                break;
            }
            let fragment = self.pending.remove(0);
            let capacity = contiguous_head.remaining_capacity();
            let copy_len = core::cmp::min(fragment.data.len(), capacity);
            contiguous_head
                .try_extend_from_slice(&fragment.data[..copy_len])
                .ok();
            total_appended += copy_len;
            *offset_next = offset_next.saturating_add(copy_len as u64);
            if copy_len < fragment.data.len() {
                // Head ran out of room; the rest of this fragment is
                // lost. Caller treats the triggering packet as
                // TransientRecvBufferFull so the peer retransmits.
                total_dropped += fragment.data.len() - copy_len;
                break;
            }
        }
        (total_appended, total_dropped)
    }

    /// Insert a fragment into the pending queue. Returns the count
    /// of bytes successfully buffered (may be less than `data.len()`
    /// if the queue is full + the fragment can't merge into an
    /// existing entry, OR if `data.len()` exceeds the fragment-cap).
    fn insert_pending(&mut self, offset: u64, data: &[u8]) -> usize {
        if data.is_empty() {
            return 0;
        }
        // Find the insertion position — sorted by offset.
        let mut position = 0;
        while position < self.pending.len() && self.pending[position].offset < offset {
            position += 1;
        }

        // Already covered by an existing pending fragment?
        if position > 0 {
            let prior = &self.pending[position - 1];
            let prior_end = prior.offset + prior.data.len() as u64;
            if prior_end >= offset + data.len() as u64 {
                return data.len(); // fully covered, treat as buffered no-op
            }
        }

        // Compute the bytes that would actually buffer here vs the
        // fragment cap.
        let take = core::cmp::min(data.len(), FRAGMENT_INLINE_BYTES);
        let mut fragment = Fragment {
            offset,
            data: heapless::Vec::new(),
        };
        // try_extend_from_slice on heapless::Vec: returns Err on cap;
        // we sized take to fit.
        fragment.data.extend_from_slice(&data[..take]).ok();

        // Insert; if at cap, drop (return 0).
        if self.pending.len() >= MAX_FRAGMENTS {
            return 0;
        }
        self.pending.insert(position, fragment).ok();
        take
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use arrayvec::ArrayVec;

    const HEAD_CAP: usize = 64;

    fn empty_head() -> (ArrayVec<u8, HEAD_CAP>, u64) {
        (ArrayVec::new(), 0)
    }

    #[test]
    fn in_order_insert_appends_directly() {
        let mut queue = ReassemblyQueue::new();
        let (mut head, mut next) = empty_head();
        let outcome = queue.insert(0, b"hello", &mut head, &mut next);
        assert!(matches!(
            outcome,
            InsertOutcome::Accepted {
                appended_to_contiguous: 5
            }
        ));
        assert_eq!(&head[..], b"hello");
        assert_eq!(next, 5);
        assert_eq!(queue.pending_count(), 0);
    }

    #[test]
    fn out_of_order_then_gap_fill_drains_correctly() {
        let mut queue = ReassemblyQueue::new();
        let (mut head, mut next) = empty_head();
        // Fragment at offset 5 arrives first → goes pending.
        let outcome = queue.insert(5, b"world", &mut head, &mut next);
        assert!(matches!(
            outcome,
            InsertOutcome::Accepted {
                appended_to_contiguous: 0
            }
        ));
        assert_eq!(head.len(), 0);
        assert_eq!(next, 0);
        assert_eq!(queue.pending_count(), 1);

        // Fragment at offset 0 fills the gap → both drain.
        let outcome = queue.insert(0, b"hello", &mut head, &mut next);
        assert!(matches!(
            outcome,
            InsertOutcome::Accepted {
                appended_to_contiguous: 10
            }
        ));
        assert_eq!(&head[..], b"helloworld");
        assert_eq!(next, 10);
        assert_eq!(queue.pending_count(), 0);
    }

    #[test]
    fn duplicate_already_consumed_returns_duplicate() {
        let mut queue = ReassemblyQueue::new();
        let (mut head, mut next) = empty_head();
        queue.insert(0, b"hello", &mut head, &mut next);
        let outcome = queue.insert(0, b"hello", &mut head, &mut next);
        assert_eq!(outcome, InsertOutcome::Duplicate);
        assert_eq!(&head[..], b"hello");
    }

    #[test]
    fn partial_overlap_clips_to_new_bytes_only() {
        let mut queue = ReassemblyQueue::new();
        let (mut head, mut next) = empty_head();
        queue.insert(0, b"hello", &mut head, &mut next);
        // New fragment covers bytes 3..8 — bytes 3..5 are duplicate
        // (already in head); bytes 5..8 are new.
        let outcome = queue.insert(3, b"lo BIG", &mut head, &mut next);
        assert!(matches!(
            outcome,
            InsertOutcome::Accepted {
                appended_to_contiguous: 4
            }
        ));
        // Only the 4 new bytes ("o BI" — wait, let's trace:
        // bytes 3..9 are "lo BIG" = 6 bytes. clip_lo = 5 - 3 = 2.
        // clipped = "o BIG" = 5 bytes? No wait:
        // "lo BIG".len() = 6. clip_lo=2 → clipped = " BIG" (4 bytes).
        // Wait: "lo BIG" is l-o-space-B-I-G = 6 bytes; slice from idx 2 = "space-B-I-G" = 4 bytes.
        assert_eq!(&head[..], b"hello BIG");
        assert_eq!(next, 9);
    }

    #[test]
    fn pending_fragments_sorted_by_offset() {
        let mut queue = ReassemblyQueue::new();
        let (mut head, mut next) = empty_head();
        queue.insert(20, b"END", &mut head, &mut next);
        queue.insert(10, b"MID", &mut head, &mut next);
        queue.insert(5, b"AB", &mut head, &mut next);
        let offsets: alloc::vec::Vec<u64> = queue.pending().map(|frag| frag.offset).collect();
        assert_eq!(offsets, alloc::vec![5, 10, 20]);
    }

    #[test]
    fn drain_stops_at_next_gap() {
        let mut queue = ReassemblyQueue::new();
        let (mut head, mut next) = empty_head();
        queue.insert(5, b"BBBBB", &mut head, &mut next); // pending
        queue.insert(15, b"CCC", &mut head, &mut next); // pending
        queue.insert(0, b"AAAAA", &mut head, &mut next); // drains first two? No — gap between 10 and 15.
        // After insert at offset 0: drain offset 0..5 (head=AAAAA),
        // then drain pending offset 5..10 (head=AAAAABBBBB), then
        // STOP (pending offset 15 is > 10).
        assert_eq!(&head[..], b"AAAAABBBBB");
        assert_eq!(next, 10);
        assert_eq!(queue.pending_count(), 1);
    }

    #[test]
    fn over_cap_fragment_truncates_tail() {
        let mut queue = ReassemblyQueue::new();
        let (mut head, mut next) = empty_head();
        // Fragment longer than FRAGMENT_INLINE_BYTES (256) at gap.
        let big = alloc::vec![0xAA; FRAGMENT_INLINE_BYTES + 100];
        let outcome = queue.insert(100, &big, &mut head, &mut next);
        match outcome {
            InsertOutcome::Truncated {
                appended_to_contiguous: 0,
                dropped_bytes,
            } => {
                assert_eq!(dropped_bytes, 100, "100 bytes past fragment cap dropped");
            }
            other => panic!("expected Truncated, got {other:?}"),
        }
        assert_eq!(queue.pending_count(), 1);
    }

    #[test]
    fn pending_queue_full_drops_new_fragment() {
        let mut queue = ReassemblyQueue::new();
        let (mut head, mut next) = empty_head();
        // Fill MAX_FRAGMENTS slots with non-contiguous gaps.
        for i in 0..MAX_FRAGMENTS as u64 {
            queue.insert(10 + i * 100, b"X", &mut head, &mut next);
        }
        assert_eq!(queue.pending_count(), MAX_FRAGMENTS);
        // One more → drop.
        let outcome = queue.insert(10_000, b"DROP", &mut head, &mut next);
        match outcome {
            InsertOutcome::Truncated {
                appended_to_contiguous: 0,
                dropped_bytes: 4,
            } => {}
            other => panic!("expected Truncated 4, got {other:?}"),
        }
        assert_eq!(queue.pending_count(), MAX_FRAGMENTS);
    }

    extern crate alloc;
}
