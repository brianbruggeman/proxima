//! Bounded out-of-order reassembly (RFC 9293 §3.4 receive processing).
//!
//! Tracks which sequence ranges above `RCV.NXT` have arrived, coalesces them,
//! and reports how many bytes become in-order-deliverable as gaps fill. The
//! payload bytes themselves live in the caller's buffer (zero-copy); this
//! module is pure sequence-space bookkeeping.
//!
//! No-alloc: the gap table is a fixed `[Gap; MAX_GAPS]`. When a new gap would
//! overflow a full table it returns [`InsertOutcome::WindowShrinkRequired`] so
//! the caller closes the receive window (RFC 9293 §3.8.6.2 zero-window) rather
//! than silently dropping data — a silent drop is an undetectable hole in the
//! delivered byte stream.
//!
//! All ordering is done on the offset-from-`rcv_nxt` (`distance_from`), which is
//! ordinary `u32` arithmetic bounded by the receive window — this sidesteps the
//! 2^32 serial-number wrap that bare comparison of absolute sequence numbers
//! would get wrong.

use super::seq::SeqNum;

/// A buffered out-of-order range `[start, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Gap {
    start: SeqNum,
    end: SeqNum,
}

/// Result of inserting a received segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InsertOutcome {
    /// In-order (possibly gap-filling): `RCV.NXT` advanced by `bytes`.
    Delivered { bytes: u32 },
    /// Buffered above a gap; `RCV.NXT` unchanged.
    OutOfOrder,
    /// Entirely at or below `RCV.NXT`; nothing new.
    Duplicate,
    /// Gap table full and the segment opens a new gap — caller must shrink the
    /// receive window to zero to stop the sender.
    WindowShrinkRequired,
}

/// Fixed-capacity reassembly tracker over the receive sequence space.
#[derive(Debug, Clone)]
pub struct Reassembler<const MAX_GAPS: usize> {
    rcv_nxt: SeqNum,
    gaps: [Gap; MAX_GAPS],
    len: usize,
}

impl<const MAX_GAPS: usize> Reassembler<MAX_GAPS> {
    #[must_use]
    pub fn new(rcv_nxt: SeqNum) -> Self {
        Self {
            rcv_nxt,
            gaps: [Gap {
                start: rcv_nxt,
                end: rcv_nxt,
            }; MAX_GAPS],
            len: 0,
        }
    }

    #[must_use]
    pub const fn rcv_nxt(&self) -> SeqNum {
        self.rcv_nxt
    }

    #[must_use]
    pub const fn pending_gaps(&self) -> usize {
        self.len
    }

    /// Offset of `seq` ahead of `rcv_nxt`. Only meaningful when `seq` is at or
    /// ahead of `rcv_nxt` (offset `< 2^31`); a value `>= 2^31` means `seq` is
    /// behind `rcv_nxt`.
    fn offset(&self, seq: SeqNum) -> u32 {
        seq.distance_from(self.rcv_nxt)
    }

    fn is_behind(&self, seq: SeqNum) -> bool {
        self.offset(seq) >= (1 << 31)
    }

    /// Insert a received segment `[seg_start, seg_start + seg_len)`.
    pub fn insert(&mut self, seg_start: SeqNum, seg_len: u32) -> InsertOutcome {
        if seg_len == 0 {
            return InsertOutcome::Duplicate;
        }
        let seg_end = seg_start.wrapping_add(seg_len);

        // Fully old: end at or behind rcv_nxt.
        if seg_end == self.rcv_nxt || self.is_behind(seg_end) {
            return InsertOutcome::Duplicate;
        }

        // Clip a front that overlaps already-delivered bytes.
        let start = if self.is_behind(seg_start) {
            self.rcv_nxt
        } else {
            seg_start
        };

        if start == self.rcv_nxt {
            self.rcv_nxt = seg_end;
            self.drain_contiguous();
            return InsertOutcome::Delivered {
                bytes: self.rcv_nxt.distance_from(start),
            };
        }

        self.insert_gap(Gap {
            start,
            end: seg_end,
        })
    }

    /// After `rcv_nxt` advances, absorb any leading gaps now contiguous.
    fn drain_contiguous(&mut self) {
        let mut index = 0;
        while index < self.len {
            let gap = self.gaps[index];
            // gap fully behind the new rcv_nxt: drop it.
            if gap.end == self.rcv_nxt || self.is_behind(gap.end) {
                index += 1;
                continue;
            }
            // gap starts at or before rcv_nxt and extends past it: absorb.
            if gap.start == self.rcv_nxt || self.is_behind(gap.start) {
                self.rcv_nxt = gap.end;
                index += 1;
                continue;
            }
            break;
        }
        if index > 0 {
            self.remove_front(index);
        }
    }

    fn remove_front(&mut self, count: usize) {
        let remaining = self.len - count;
        for slot in 0..remaining {
            self.gaps[slot] = self.gaps[slot + count];
        }
        self.len = remaining;
    }

    /// Insert an out-of-order gap, coalescing overlapping/adjacent gaps.
    fn insert_gap(&mut self, mut incoming: Gap) -> InsertOutcome {
        // Merge any existing gap that overlaps or touches `incoming`, compacting
        // survivors into place as we go.
        let mut write = 0;
        let mut merged_any = false;
        for read in 0..self.len {
            let existing = self.gaps[read];
            if self.touches(existing, incoming) {
                incoming = self.union(existing, incoming);
                merged_any = true;
            } else {
                self.gaps[write] = existing;
                write += 1;
            }
        }
        self.len = write;

        if !merged_any && self.len == MAX_GAPS {
            return InsertOutcome::WindowShrinkRequired;
        }
        self.push_sorted(incoming);
        InsertOutcome::OutOfOrder
    }

    /// Two ranges overlap or are adjacent (touching end-to-start).
    fn touches(&self, lhs: Gap, rhs: Gap) -> bool {
        let lhs_start = self.offset(lhs.start);
        let lhs_end = self.offset(lhs.end);
        let rhs_start = self.offset(rhs.start);
        let rhs_end = self.offset(rhs.end);
        lhs_start <= rhs_end && rhs_start <= lhs_end
    }

    fn union(&self, lhs: Gap, rhs: Gap) -> Gap {
        let start = if self.offset(lhs.start) <= self.offset(rhs.start) {
            lhs.start
        } else {
            rhs.start
        };
        let end = if self.offset(lhs.end) >= self.offset(rhs.end) {
            lhs.end
        } else {
            rhs.end
        };
        Gap { start, end }
    }

    fn push_sorted(&mut self, gap: Gap) {
        let key = self.offset(gap.start);
        let mut position = self.len;
        while position > 0 && self.offset(self.gaps[position - 1].start) > key {
            self.gaps[position] = self.gaps[position - 1];
            position -= 1;
        }
        self.gaps[position] = gap;
        self.len += 1;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use proptest::prelude::*;

    const RCV: u32 = 1000;

    fn reassembler() -> Reassembler<4> {
        Reassembler::new(SeqNum(RCV))
    }

    #[test]
    fn in_order_delivers_immediately() {
        let mut reasm = reassembler();
        assert_eq!(
            reasm.insert(SeqNum(1000), 100),
            InsertOutcome::Delivered { bytes: 100 }
        );
        assert_eq!(reasm.rcv_nxt(), SeqNum(1100));
        assert_eq!(reasm.pending_gaps(), 0);
    }

    #[test]
    fn out_of_order_buffers_then_gap_fill_delivers_all() {
        let mut reasm = reassembler();
        assert_eq!(reasm.insert(SeqNum(1100), 100), InsertOutcome::OutOfOrder);
        assert_eq!(reasm.rcv_nxt(), SeqNum(1000));
        assert_eq!(reasm.pending_gaps(), 1);
        // filling [1000,1100) makes [1100,1200) contiguous -> deliver 200.
        assert_eq!(
            reasm.insert(SeqNum(1000), 100),
            InsertOutcome::Delivered { bytes: 200 }
        );
        assert_eq!(reasm.rcv_nxt(), SeqNum(1200));
        assert_eq!(reasm.pending_gaps(), 0);
    }

    #[test]
    fn adjacent_out_of_order_segments_coalesce() {
        let mut reasm = reassembler();
        reasm.insert(SeqNum(1100), 100);
        reasm.insert(SeqNum(1200), 100);
        assert_eq!(reasm.pending_gaps(), 1);
    }

    #[test]
    fn overlapping_out_of_order_segments_coalesce() {
        let mut reasm = reassembler();
        reasm.insert(SeqNum(1100), 100);
        reasm.insert(SeqNum(1150), 100);
        assert_eq!(reasm.pending_gaps(), 1);
        // fill the front gap; whole coalesced run [1100,1250) delivers.
        assert_eq!(
            reasm.insert(SeqNum(1000), 100),
            InsertOutcome::Delivered { bytes: 250 }
        );
        assert_eq!(reasm.rcv_nxt(), SeqNum(1250));
    }

    #[test]
    fn fully_old_segment_is_duplicate() {
        let mut reasm = reassembler();
        assert_eq!(reasm.insert(SeqNum(900), 50), InsertOutcome::Duplicate);
        assert_eq!(reasm.rcv_nxt(), SeqNum(1000));
    }

    #[test]
    fn front_overlap_is_clipped_and_delivered() {
        let mut reasm = reassembler();
        // [950,1050): 50 bytes already delivered, 50 new past rcv_nxt.
        assert_eq!(
            reasm.insert(SeqNum(950), 100),
            InsertOutcome::Delivered { bytes: 50 }
        );
        assert_eq!(reasm.rcv_nxt(), SeqNum(1050));
    }

    #[test]
    fn full_gap_table_requires_window_shrink() {
        let mut reasm = reassembler();
        // four non-adjacent gaps fill the MAX_GAPS=4 table.
        reasm.insert(SeqNum(1100), 50);
        reasm.insert(SeqNum(1200), 50);
        reasm.insert(SeqNum(1300), 50);
        reasm.insert(SeqNum(1400), 50);
        assert_eq!(reasm.pending_gaps(), 4);
        // a fifth disjoint gap cannot be stored.
        assert_eq!(
            reasm.insert(SeqNum(1500), 50),
            InsertOutcome::WindowShrinkRequired
        );
    }

    #[test]
    fn full_table_still_accepts_a_coalescing_segment() {
        let mut reasm = reassembler();
        reasm.insert(SeqNum(1100), 50);
        reasm.insert(SeqNum(1200), 50);
        reasm.insert(SeqNum(1300), 50);
        reasm.insert(SeqNum(1400), 50);
        // extends ONE existing gap ([1100,1150)->[1100,1180)) rather than
        // opening a new one: allowed, count unchanged.
        assert_eq!(reasm.insert(SeqNum(1150), 30), InsertOutcome::OutOfOrder);
        assert_eq!(reasm.pending_gaps(), 4);
    }

    #[test]
    fn bridging_segment_coalesces_two_gaps() {
        let mut reasm = reassembler();
        reasm.insert(SeqNum(1100), 50);
        reasm.insert(SeqNum(1200), 50);
        reasm.insert(SeqNum(1300), 50);
        assert_eq!(reasm.pending_gaps(), 3);
        // [1150,1200) bridges [1100,1150) and [1200,1250) into one run.
        assert_eq!(reasm.insert(SeqNum(1150), 50), InsertOutcome::OutOfOrder);
        assert_eq!(reasm.pending_gaps(), 2);
    }

    proptest! {
        /// `rcv_nxt` is monotonically non-decreasing across arbitrary insert sequences.
        /// This is the core delivery invariant: bytes can only be consumed forward.
        #[test]
        fn rcv_nxt_is_monotone_over_arbitrary_inserts(
            initial in any::<u32>(),
            inserts in prop::collection::vec((any::<u32>(), 1_u32..=65535), 0..32),
        ) {
            let mut reasm = Reassembler::<8>::new(SeqNum(initial));
            let mut last_rcv_nxt = reasm.rcv_nxt().0;
            for (seq, len) in inserts {
                reasm.insert(SeqNum(seq), len);
                let current = reasm.rcv_nxt().0;
                let advanced = current.wrapping_sub(last_rcv_nxt);
                prop_assert!(
                    advanced < (1 << 31),
                    "rcv_nxt moved backward: was {last_rcv_nxt} now {current}"
                );
                last_rcv_nxt = current;
            }
        }

        /// Total delivered bytes reported across all inserts must not exceed
        /// the span of distinct sequence space actually inserted.
        #[test]
        fn delivered_bytes_never_exceed_inserted_span(
            initial in any::<u32>(),
            inserts in prop::collection::vec((any::<u32>(), 1_u32..=1024), 0..16),
        ) {
            let mut reasm = Reassembler::<8>::new(SeqNum(initial));
            let mut total_delivered: u64 = 0;
            let mut max_seq_end: u64 = initial as u64;

            for (seq, len) in &inserts {
                let seq_end = (*seq as u64).wrapping_add(*len as u64) & 0xFFFF_FFFF;
                let end_forward = (seq_end.wrapping_sub(initial as u64)) & 0xFFFF_FFFF;
                if end_forward < (1 << 31) {
                    max_seq_end = max_seq_end.max(
                        (initial as u64).wrapping_add(end_forward) & 0xFFFF_FFFF,
                    );
                }
                if let InsertOutcome::Delivered { bytes } = reasm.insert(SeqNum(*seq), *len) {
                    total_delivered += bytes as u64;
                }
            }

            let max_possible = (reasm.rcv_nxt().0 as u64)
                .wrapping_sub(initial as u64)
                & 0xFFFF_FFFF;
            prop_assert!(
                total_delivered <= max_possible + 1,
                "delivered {total_delivered} exceeds inserted span {max_possible}"
            );
        }

        /// `insert` never panics regardless of seq/len including wrap-around.
        #[test]
        fn insert_never_panics_on_arbitrary_inputs(
            initial in any::<u32>(),
            inserts in prop::collection::vec((any::<u32>(), any::<u32>()), 0..24),
        ) {
            let mut reasm = Reassembler::<4>::new(SeqNum(initial));
            for (seq, len) in inserts {
                let _ = reasm.insert(SeqNum(seq), len);
            }
        }

        /// When the gap table is full a disjoint segment must return
        /// `WindowShrinkRequired`, never panic.
        #[test]
        fn full_gap_table_returns_window_shrink_required_not_panic(
            initial in 0_u32..=(u32::MAX - 1024 * 8),
        ) {
            let mut reasm = Reassembler::<4>::new(SeqNum(initial));
            // fill all 4 slots with non-adjacent gaps above initial.
            for slot in 0..4_u32 {
                reasm.insert(SeqNum(initial.wrapping_add(100 + slot * 200)), 50);
            }
            prop_assert_eq!(reasm.pending_gaps(), 4, "expected 4 pending gaps");
            let outcome = reasm.insert(SeqNum(initial.wrapping_add(1000 + 4 * 200)), 50);
            prop_assert_eq!(
                outcome,
                InsertOutcome::WindowShrinkRequired,
                "full table must yield WindowShrinkRequired"
            );
        }
    }

    #[test]
    fn wraparound_in_order_delivery() {
        let mut reasm = Reassembler::<4>::new(SeqNum(0xFFFF_FFE0));
        // 32 bytes spanning the 2^32 wrap.
        assert_eq!(
            reasm.insert(SeqNum(0xFFFF_FFE0), 64),
            InsertOutcome::Delivered { bytes: 64 }
        );
        assert_eq!(reasm.rcv_nxt(), SeqNum(0x0000_0020));
    }
}
