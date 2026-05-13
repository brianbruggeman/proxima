//! RFC 9293 §3.3.1 send/receive window bookkeeping and RFC 5681 §2 duplicate
//! ACK detection.
//!
//! The send sequence variables are `SND.UNA` (oldest unacknowledged),
//! `SND.NXT` (next to send), `SND.WND` (peer's advertised window); the receive
//! variables are `RCV.NXT` (next expected) and `RCV.WND` (our advertised
//! window). An ACK is acceptable when `SND.UNA < SEG.ACK <= SND.NXT` under
//! serial arithmetic. A duplicate ACK (the fast-retransmit trigger) is one that
//! acks no new data while data is outstanding and the window is unchanged.

use super::seq::SeqNum;

/// What an incoming ACK did to the send window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AckOutcome {
    /// Acked previously-unacknowledged data; `SND.UNA` advanced.
    Acked { bytes_acked: u32 },
    /// RFC 5681 §2 duplicate ACK; `count` is the running run length. The caller
    /// must additionally confirm the segment was a pure ACK (no payload, no
    /// SYN/FIN) before treating this as a fast-retransmit signal.
    Duplicate { count: u8 },
    /// ACK is beyond `SND.NXT` — acknowledges unsent data; caller should ACK.
    Unsent,
    /// Stale ACK or a bare window update; no retransmit action.
    Ignored,
}

/// Send/receive window state for one connection.
#[derive(Debug, Clone, Copy)]
pub struct WindowTracker {
    snd_una: SeqNum,
    snd_nxt: SeqNum,
    snd_wnd: u32,
    rcv_nxt: SeqNum,
    rcv_wnd: u32,
    dup_ack_count: u8,
}

impl WindowTracker {
    #[must_use]
    pub const fn new(
        snd_una: SeqNum,
        snd_nxt: SeqNum,
        snd_wnd: u32,
        rcv_nxt: SeqNum,
        rcv_wnd: u32,
    ) -> Self {
        Self {
            snd_una,
            snd_nxt,
            snd_wnd,
            rcv_nxt,
            rcv_wnd,
            dup_ack_count: 0,
        }
    }

    /// Process an incoming ACK number with the peer's advertised window.
    pub fn on_ack(&mut self, ack: SeqNum, wnd: u32) -> AckOutcome {
        // SND.UNA < ACK <= SND.NXT  => acknowledges new data.
        let acks_new =
            self.snd_una.precedes(ack) && (ack == self.snd_nxt || ack.precedes(self.snd_nxt));
        if acks_new {
            let bytes_acked = ack.distance_from(self.snd_una);
            self.snd_una = ack;
            self.snd_wnd = wnd;
            self.dup_ack_count = 0;
            return AckOutcome::Acked { bytes_acked };
        }

        // ACK > SND.NXT => acknowledges something never sent.
        if self.snd_nxt.precedes(ack) {
            return AckOutcome::Unsent;
        }

        // ACK == SND.UNA and data is outstanding and window unchanged => dup.
        if ack == self.snd_una && self.snd_una != self.snd_nxt && wnd == self.snd_wnd {
            self.dup_ack_count = self.dup_ack_count.saturating_add(1);
            return AckOutcome::Duplicate {
                count: self.dup_ack_count,
            };
        }

        // Stale ACK or a pure window update: absorb the window, no retransmit.
        self.snd_wnd = wnd;
        self.dup_ack_count = 0;
        AckOutcome::Ignored
    }

    /// Advance `SND.NXT` after queueing `len` bytes for transmission.
    pub fn on_data_sent(&mut self, len: u32) {
        self.snd_nxt = self.snd_nxt.wrapping_add(len);
    }

    /// Advance `RCV.NXT` after delivering `len` contiguous in-order bytes.
    pub fn on_data_received(&mut self, len: u32) {
        self.rcv_nxt = self.rcv_nxt.wrapping_add(len);
    }

    /// Bytes the peer's advertised window still permits in flight.
    #[must_use]
    pub fn usable_send_window(&self) -> u32 {
        let in_flight = self.snd_nxt.distance_from(self.snd_una);
        self.snd_wnd.saturating_sub(in_flight)
    }

    #[must_use]
    pub const fn snd_una(&self) -> SeqNum {
        self.snd_una
    }

    #[must_use]
    pub const fn snd_nxt(&self) -> SeqNum {
        self.snd_nxt
    }

    #[must_use]
    pub const fn rcv_nxt(&self) -> SeqNum {
        self.rcv_nxt
    }

    #[must_use]
    pub const fn rcv_wnd(&self) -> u32 {
        self.rcv_wnd
    }

    /// Set our advertised receive window (e.g. shrink to zero on OOO overflow).
    pub fn set_rcv_wnd(&mut self, wnd: u32) {
        self.rcv_wnd = wnd;
    }

    #[must_use]
    pub const fn dup_ack_count(&self) -> u8 {
        self.dup_ack_count
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// After any sequence of ACKs, `snd_una` must never pass `snd_nxt`
        /// in the forward direction: you cannot acknowledge more than was sent.
        #[test]
        fn snd_una_never_passes_snd_nxt_over_arbitrary_acks(
            initial_una in any::<u32>(),
            sent_bytes in 0_u32..=65535,
            acks in prop::collection::vec((any::<u32>(), any::<u32>()), 0..32),
        ) {
            let snd_nxt = SeqNum(initial_una).wrapping_add(sent_bytes);
            let mut window = WindowTracker::new(
                SeqNum(initial_una),
                snd_nxt,
                65535,
                SeqNum(0),
                65535,
            );
            for (ack_val, wnd) in acks {
                window.on_ack(SeqNum(ack_val), wnd);
                let una = window.snd_una().0;
                let nxt = window.snd_nxt().0;
                let una_past_nxt = SeqNum(una).precedes(SeqNum(nxt)) || una == nxt;
                prop_assert!(
                    una_past_nxt,
                    "snd_una={una} passed snd_nxt={nxt}"
                );
            }
        }

        /// `on_ack` never panics for any ACK value and window size.
        #[test]
        fn on_ack_never_panics(
            initial_una in any::<u32>(),
            sent_bytes in 0_u32..=65535,
            ack_val in any::<u32>(),
            peer_wnd in any::<u32>(),
        ) {
            let snd_nxt = SeqNum(initial_una).wrapping_add(sent_bytes);
            let mut window = WindowTracker::new(
                SeqNum(initial_una),
                snd_nxt,
                65535,
                SeqNum(0),
                65535,
            );
            let _ = window.on_ack(SeqNum(ack_val), peer_wnd);
        }
    }

    fn tracker() -> WindowTracker {
        // SND.UNA=1000, SND.NXT=1500 (500 bytes outstanding), peer wnd 8000.
        WindowTracker::new(SeqNum(1000), SeqNum(1500), 8000, SeqNum(5000), 16384)
    }

    #[test]
    fn ack_advances_snd_una_on_new_data() {
        let mut window = tracker();
        assert_eq!(
            window.on_ack(SeqNum(1200), 8000),
            AckOutcome::Acked { bytes_acked: 200 }
        );
        assert_eq!(window.snd_una(), SeqNum(1200));
    }

    #[test]
    fn full_ack_clears_outstanding() {
        let mut window = tracker();
        assert_eq!(
            window.on_ack(SeqNum(1500), 8000),
            AckOutcome::Acked { bytes_acked: 500 }
        );
        assert_eq!(window.snd_una(), window.snd_nxt());
    }

    #[test]
    fn three_duplicate_acks_count_up() {
        let mut window = tracker();
        assert_eq!(
            window.on_ack(SeqNum(1000), 8000),
            AckOutcome::Duplicate { count: 1 }
        );
        assert_eq!(
            window.on_ack(SeqNum(1000), 8000),
            AckOutcome::Duplicate { count: 2 }
        );
        assert_eq!(
            window.on_ack(SeqNum(1000), 8000),
            AckOutcome::Duplicate { count: 3 }
        );
    }

    #[test]
    fn new_data_ack_resets_dup_count() {
        let mut window = tracker();
        window.on_ack(SeqNum(1000), 8000);
        window.on_ack(SeqNum(1000), 8000);
        assert_eq!(window.dup_ack_count(), 2);
        window.on_ack(SeqNum(1200), 8000);
        assert_eq!(window.dup_ack_count(), 0);
    }

    #[test]
    fn ack_beyond_snd_nxt_is_unsent() {
        let mut window = tracker();
        assert_eq!(window.on_ack(SeqNum(1600), 8000), AckOutcome::Unsent);
    }

    #[test]
    fn repeated_ack_with_nothing_outstanding_is_ignored() {
        let mut window = WindowTracker::new(SeqNum(1500), SeqNum(1500), 8000, SeqNum(5000), 16384);
        assert_eq!(window.on_ack(SeqNum(1500), 8000), AckOutcome::Ignored);
    }

    #[test]
    fn window_change_on_same_ack_is_not_duplicate() {
        let mut window = tracker();
        assert_eq!(window.on_ack(SeqNum(1000), 9000), AckOutcome::Ignored);
        assert_eq!(window.dup_ack_count(), 0);
    }

    #[test]
    fn usable_send_window_subtracts_in_flight() {
        let window = tracker();
        // 500 bytes in flight against an 8000 window.
        assert_eq!(window.usable_send_window(), 7500);
    }
}
