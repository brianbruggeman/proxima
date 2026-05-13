//! TCP sequence-number arithmetic (RFC 1982 serial numbers) and the RFC 9293
//! §3.4 segment-acceptability test.
//!
//! Sequence numbers are modular over 2^32, so a bare `<` is wrong for any pair
//! that straddles the wrap. [`SeqNum`] deliberately derives no `PartialOrd`:
//! ordering is only available through [`SeqNum::precedes`], which makes a
//! misuse a compile error rather than a silent bulk-transfer bug at 4 GiB.

/// A 32-bit TCP sequence number under RFC 1982 serial arithmetic.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SeqNum(pub u32);

impl SeqNum {
    /// `self` strictly precedes `other` in serial-number order (RFC 1982):
    /// the forward distance falls in `[1, 2^31)`. Equality is not precedence.
    #[must_use]
    pub const fn precedes(self, other: Self) -> bool {
        let forward = other.0.wrapping_sub(self.0);
        forward != 0 && forward < (1 << 31)
    }

    #[must_use]
    pub const fn wrapping_add(self, count: u32) -> Self {
        Self(self.0.wrapping_add(count))
    }

    /// Forward distance from `earlier` to `self`, modulo 2^32.
    #[must_use]
    pub const fn distance_from(self, earlier: Self) -> u32 {
        self.0.wrapping_sub(earlier.0)
    }
}

/// True when `seq` lies in the half-open window `[lo, lo + wnd)` under serial
/// arithmetic — the modular distance is the only comparison that survives the
/// 2^32 wrap.
#[must_use]
const fn in_window(seq: SeqNum, lo: SeqNum, wnd: u32) -> bool {
    seq.distance_from(lo) < wnd
}

/// RFC 9293 §3.4 Table 6: is an incoming segment acceptable given the receiver's
/// `rcv_nxt` and `rcv_wnd`? `seg_len` counts the sequence space the segment
/// occupies (payload plus SYN/FIN flags).
#[must_use]
pub fn segment_acceptable(seg_seq: SeqNum, seg_len: u32, rcv_nxt: SeqNum, rcv_wnd: u32) -> bool {
    match (seg_len, rcv_wnd) {
        // zero-window probe: only the exact next byte is acceptable, so a
        // keep-alive ACK is processed while any data is refused (RFC 9293 §3.4
        // tightening of RFC 793).
        (0, 0) => seg_seq == rcv_nxt,
        (0, _) => in_window(seg_seq, rcv_nxt, rcv_wnd),
        (_, 0) => false,
        (_, _) => {
            let seg_end = seg_seq.wrapping_add(seg_len - 1);
            in_window(seg_seq, rcv_nxt, rcv_wnd) || in_window(seg_end, rcv_nxt, rcv_wnd)
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use proptest::prelude::*;
    use rstest::rstest;

    #[rstest]
    #[case::forward(1, 2, true)]
    #[case::backward(2, 1, false)]
    #[case::equal(5, 5, false)]
    #[case::wrap_forward(0xFFFF_FFFF, 0, true)]
    #[case::wrap_backward(0, 0xFFFF_FFFF, false)]
    fn precedes_follows_rfc1982(#[case] left: u32, #[case] right: u32, #[case] expected: bool) {
        assert_eq!(SeqNum(left).precedes(SeqNum(right)), expected);
    }

    proptest! {
        /// `segment_acceptable` must never panic for any combination of inputs,
        /// including values that wrap the 2^32 boundary.
        #[test]
        fn segment_acceptable_never_panics(
            rcv_nxt in any::<u32>(),
            rcv_wnd in any::<u32>(),
            seg_seq in any::<u32>(),
            seg_len in any::<u32>(),
        ) {
            let _ = segment_acceptable(SeqNum(seg_seq), seg_len, SeqNum(rcv_nxt), rcv_wnd);
        }

        /// The RFC 9293 §3.4 Table 6 predicate cross-checked against an
        /// independent in_window formulation derived directly from serial
        /// arithmetic.  Both expressions must agree for every input.
        #[test]
        fn segment_acceptable_matches_independent_rfc9293_predicate(
            rcv_nxt in any::<u32>(),
            rcv_wnd in any::<u32>(),
            seg_seq in any::<u32>(),
            seg_len in any::<u32>(),
        ) {
            let got = segment_acceptable(SeqNum(seg_seq), seg_len, SeqNum(rcv_nxt), rcv_wnd);

            let expected = match (seg_len, rcv_wnd) {
                (0, 0) => seg_seq == rcv_nxt,
                (0, _) => SeqNum(seg_seq).distance_from(SeqNum(rcv_nxt)) < rcv_wnd,
                (_, 0) => false,
                (_, _) => {
                    let seg_end = SeqNum(seg_seq).wrapping_add(seg_len - 1);
                    SeqNum(seg_seq).distance_from(SeqNum(rcv_nxt)) < rcv_wnd
                        || seg_end.distance_from(SeqNum(rcv_nxt)) < rcv_wnd
                }
            };

            prop_assert_eq!(got, expected,
                "mismatch: rcv_nxt={} rcv_wnd={} seg_seq={} seg_len={}",
                rcv_nxt, rcv_wnd, seg_seq, seg_len);
        }

        /// `precedes` is antisymmetric: when a != b, exactly one of a.precedes(b)
        /// or b.precedes(a) can be true (they cannot both be true simultaneously).
        #[test]
        fn precedes_is_antisymmetric_for_distinct_values(
            left in any::<u32>(),
            right in any::<u32>(),
        ) {
            prop_assume!(left != right);
            let forward = SeqNum(left).precedes(SeqNum(right));
            let backward = SeqNum(right).precedes(SeqNum(left));
            prop_assert!(
                !(forward && backward),
                "both directions claimed to precede: left={left} right={right}"
            );
        }
    }

    // RFC 9293 §3.4 Table 6 worked examples (see docs/tcp-data-path/discipline.md).
    #[rstest]
    #[case::left_edge(100, 10, 100, 0, true)]
    #[case::right_edge_inclusive(100, 10, 109, 0, true)]
    #[case::past_window(100, 10, 110, 0, false)]
    #[case::zero_window_probe_exact(100, 0, 100, 0, true)]
    #[case::zero_window_probe_off_by_one(100, 0, 101, 0, false)]
    #[case::data_into_zero_window(100, 0, 100, 5, false)]
    #[case::tail_lands_in_window(100, 10, 95, 10, true)]
    #[case::wraparound(0xFFFF_FFFE, 8, 2, 0, true)]
    fn segment_acceptable_matches_table6(
        #[case] rcv_nxt: u32,
        #[case] rcv_wnd: u32,
        #[case] seg_seq: u32,
        #[case] seg_len: u32,
        #[case] expected: bool,
    ) {
        assert_eq!(
            segment_acceptable(SeqNum(seg_seq), seg_len, SeqNum(rcv_nxt), rcv_wnd),
            expected
        );
    }
}
