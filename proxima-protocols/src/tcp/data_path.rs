//! Capstone data-path FSM (RFC 9293 §3): composes the connection control FSM
//! with the window, retransmit queue, RTT estimator, reassembler, and
//! congestion controller into one sans-IO unit.
//!
//! One ingress, [`DataPath::on_segment`], returns an aggregate [`SegmentOutput`]
//! because a single TCP segment can require several actions at once (deliver
//! data AND ack AND fast-retransmit). [`DataPath::try_send`] gates application
//! data by both the peer window and the congestion window. [`DataPath::poll`]
//! drives the RTO timer.
//!
//! Sequence-space only: payload bytes live in the caller's buffers (zero-copy);
//! `payload_offset` threads through the retransmit queue.
//!
//! RTT is timed one segment at a time (RFC 6298 §3): the path remembers a single
//! `(seq, sent_at)` and samples when an un-retransmitted ACK covers it (Karn).
//!
//! SEAL STATUS (principle 16): the per-module worked-example tests below prove
//! the composition's core paths on the mac. The FULL correctness oracle for the
//! integration layer is the smoltcp differential + packetdrill (edges.md E1
//! tiers 2 + 5), which are network/host-b-gated and NOT yet run. C8 is
//! therefore CORE-PROVEN, not E2E-sealed.

use super::congestion::TcpCongestionControl;
use super::connection::{Action, Connection, Input, Segment};
use super::reassembly::{InsertOutcome, Reassembler};
use super::retx::{RetransmitDecision, RetxSegment};
use super::rtt::RtoEstimator;
use super::seq::SeqNum;
use super::time::Instant;
use super::window::{AckOutcome, WindowTracker};

/// The aggregate of everything one inbound segment asked the caller to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SegmentOutput {
    /// In-order bytes now deliverable to the application (`RCV.NXT` advanced).
    pub delivered: u32,
    /// An ACK must be sent (data was delivered, or the control FSM asked).
    pub ack_required: bool,
    /// The 3rd duplicate ACK triggered a fast retransmit of this segment.
    pub fast_retransmit: Option<RetxSegment>,
    /// The peer reset the connection.
    pub connection_reset: bool,
    /// The reassembly table overflowed — advertise a zero receive window.
    pub window_shrink: bool,
}

/// A composed TCP connection data path.
pub struct DataPath<const OOO_GAPS: usize, const RETX_CAP: usize, CC>
where
    CC: TcpCongestionControl,
{
    control: Connection,
    window: WindowTracker,
    retx: super::retx::RetxQueue<RETX_CAP>,
    reasm: Reassembler<OOO_GAPS>,
    rtt: RtoEstimator,
    cc: CC,
    smss: u32,
    timed: Option<(SeqNum, Instant)>,
    rto_deadline: Option<Instant>,
}

impl<const OOO_GAPS: usize, const RETX_CAP: usize, CC> DataPath<OOO_GAPS, RETX_CAP, CC>
where
    CC: TcpCongestionControl,
{
    /// Build a data path for a connection that has completed its handshake.
    /// `iss`/`irs` are the negotiated initial send/receive sequence numbers,
    /// `peer_wnd`/`our_wnd` the advertised windows, `smss` the segment size.
    #[must_use]
    pub fn established(
        iss: SeqNum,
        irs: SeqNum,
        peer_wnd: u32,
        our_wnd: u32,
        smss: u32,
        congestion: CC,
    ) -> Self {
        let mut control = Connection::new();
        // Drive the control FSM through a passive open to ESTABLISHED so the
        // lifecycle state is correct for the data-transfer phase.
        let _ = control.step(Input::OpenActive);
        let _ = control.step(Input::Segment(Segment {
            syn: true,
            ack: true,
            ..Segment::default()
        }));
        Self {
            control,
            window: WindowTracker::new(iss, iss, peer_wnd, irs, our_wnd),
            retx: super::retx::RetxQueue::new(),
            reasm: Reassembler::new(irs),
            rtt: RtoEstimator::new(),
            cc: congestion,
            smss,
            timed: None,
            rto_deadline: None,
        }
    }

    #[must_use]
    pub fn state(&self) -> super::connection::State {
        self.control.state()
    }

    #[must_use]
    fn flight_size(&self) -> u32 {
        self.window.snd_nxt().distance_from(self.window.snd_una())
    }

    /// Try to send up to `max_len` bytes of application data starting at
    /// `SND.NXT`, gated by the usable peer window and the congestion window.
    /// `payload_offset` locates the bytes in the caller's send buffer. Returns
    /// the segment to transmit, or `None` when blocked.
    pub fn try_send(
        &mut self,
        max_len: u32,
        payload_offset: u32,
        now: Instant,
    ) -> Option<RetxSegment> {
        let budget = self
            .window
            .usable_send_window()
            .min(self.cc.cwnd().saturating_sub(self.flight_size()))
            .min(self.smss)
            .min(max_len);
        if budget == 0 {
            return None;
        }
        let seq = self.window.snd_nxt();
        let segment = RetxSegment::new(seq, budget, payload_offset);
        if !self.retx.push(segment) {
            return None;
        }
        self.window.on_data_sent(budget);
        if self.timed.is_none() {
            self.timed = Some((seq, now));
        }
        if self.rto_deadline.is_none() {
            self.rto_deadline = Some(now.saturating_add(self.rtt.rto()));
        }
        Some(segment)
    }

    /// Process one inbound segment. `payload_len` is the sequence space the
    /// segment's data occupies (the bytes themselves are the caller's).
    pub fn on_segment(
        &mut self,
        control: Segment,
        seq: SeqNum,
        ack: SeqNum,
        peer_wnd: u32,
        payload_len: u32,
        now: Instant,
    ) -> SegmentOutput {
        let mut output = SegmentOutput::default();

        match self.control.step(Input::Segment(control)) {
            Action::ConnectionReset => {
                output.connection_reset = true;
                return output;
            }
            Action::SendAck | Action::SendSynAck => output.ack_required = true,
            _ => {}
        }

        if control.ack {
            self.process_ack(ack, peer_wnd, now, &mut output);
        }

        if payload_len > 0 {
            match self.reasm.insert(seq, payload_len) {
                InsertOutcome::Delivered { bytes } => {
                    self.window.on_data_received(bytes);
                    output.delivered = bytes;
                    output.ack_required = true;
                }
                InsertOutcome::OutOfOrder => output.ack_required = true,
                InsertOutcome::WindowShrinkRequired => {
                    self.window.set_rcv_wnd(0);
                    output.window_shrink = true;
                    output.ack_required = true;
                }
                InsertOutcome::Duplicate => output.ack_required = true,
            }
        }

        output
    }

    fn process_ack(
        &mut self,
        ack: SeqNum,
        peer_wnd: u32,
        now: Instant,
        output: &mut SegmentOutput,
    ) {
        match self.window.on_ack(ack, peer_wnd) {
            AckOutcome::Acked { bytes_acked } => {
                self.retx.on_ack(ack);
                self.cc.on_ack(bytes_acked);
                self.sample_rtt(ack, now);
                self.rto_deadline =
                    (!self.retx.is_empty()).then(|| now.saturating_add(self.rtt.rto()));
            }
            AckOutcome::Duplicate { count } => {
                if self.cc.on_dup_ack(count, self.flight_size())
                    && let RetransmitDecision::Resend(segment) = self.retx.retransmit_oldest()
                {
                    output.fast_retransmit = Some(segment);
                }
            }
            AckOutcome::Unsent => output.ack_required = true,
            AckOutcome::Ignored => {}
        }
    }

    /// Sample RTT (RFC 6298) when the timed segment is acknowledged. Karn: the
    /// timer is only armed on first transmission, never on a retransmit.
    fn sample_rtt(&mut self, ack: SeqNum, now: Instant) {
        if let Some((timed_seq, sent_at)) = self.timed
            && (ack == timed_seq || timed_seq.precedes(ack))
            && let Some(sample) = now.duration_since(sent_at)
        {
            self.rtt.on_sample(sample, false);
            self.timed = None;
        }
    }

    /// Drive the RTO timer. When `now` reaches the deadline, retransmit the
    /// oldest segment, back off the RTO, and signal congestion (RFC 5681 §3.1).
    pub fn poll(&mut self, now: Instant) -> Option<RetxSegment> {
        let deadline = self.rto_deadline?;
        // proceed only once `now` has reached the deadline (Some duration).
        now.duration_since(deadline)?;
        match self.retx.retransmit_oldest() {
            RetransmitDecision::Resend(segment) => {
                self.cc.on_rto(self.flight_size());
                self.rtt.backoff();
                self.timed = None; // Karn: do not time a retransmit
                self.rto_deadline = Some(now.saturating_add(self.rtt.rto()));
                Some(segment)
            }
            RetransmitDecision::Abandon | RetransmitDecision::Empty => {
                self.rto_deadline = None;
                None
            }
        }
    }

    #[must_use]
    pub fn cwnd(&self) -> u32 {
        self.cc.cwnd()
    }

    #[must_use]
    pub fn rcv_nxt(&self) -> SeqNum {
        self.window.rcv_nxt()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use super::super::congestion::Reno;
    use super::super::connection::State;
    use super::super::time::Instant;
    use proptest::prelude::*;

    const SMSS: u32 = 1000;
    const ISS: u32 = 10_000;
    const IRS: u32 = 50_000;

    proptest! {
        /// `on_segment` never panics for arbitrary control bits, seq/ack values,
        /// peer window, payload length, and timestamp — the ingress must be
        /// unconditionally safe on an established data path.
        #[test]
        fn on_segment_never_panics_for_arbitrary_inputs(
            syn in any::<bool>(),
            ack in any::<bool>(),
            fin in any::<bool>(),
            rst in any::<bool>(),
            seq in any::<u32>(),
            ack_val in any::<u32>(),
            peer_wnd in any::<u32>(),
            payload_len in 0_u32..=65535,
            now_micros in any::<u64>(),
        ) {
            let mut conn: DataPath<8, 16, Reno> = DataPath::established(
                SeqNum(ISS),
                SeqNum(IRS),
                64_000,
                64_000,
                SMSS,
                Reno::new(SMSS),
            );
            let control = super::super::connection::Segment { syn, ack, fin, rst };
            let _ = conn.on_segment(
                control,
                SeqNum(seq),
                SeqNum(ack_val),
                peer_wnd,
                payload_len,
                Instant::from_micros(now_micros),
            );
        }
    }

    fn path() -> DataPath<8, 16, Reno> {
        DataPath::established(
            SeqNum(ISS),
            SeqNum(IRS),
            64_000,
            64_000,
            SMSS,
            Reno::new(SMSS),
        )
    }

    fn data() -> Segment {
        Segment {
            ack: true,
            ..Segment::default()
        }
    }

    #[test]
    fn established_after_construction() {
        assert_eq!(path().state(), State::Established);
    }

    #[test]
    fn in_order_data_is_delivered_and_acked() {
        let mut conn = path();
        let out = conn.on_segment(data(), SeqNum(IRS), SeqNum(ISS), 64_000, 500, Instant::ZERO);
        assert_eq!(out.delivered, 500);
        assert!(out.ack_required);
        assert_eq!(conn.rcv_nxt(), SeqNum(IRS + 500));
    }

    #[test]
    fn out_of_order_then_gap_fill_delivers_all() {
        let mut conn = path();
        // gap: [IRS+500, IRS+1000) arrives first.
        let ooo = conn.on_segment(
            data(),
            SeqNum(IRS + 500),
            SeqNum(ISS),
            64_000,
            500,
            Instant::ZERO,
        );
        assert_eq!(ooo.delivered, 0);
        // fill [IRS, IRS+500): both become contiguous.
        let fill = conn.on_segment(data(), SeqNum(IRS), SeqNum(ISS), 64_000, 500, Instant::ZERO);
        assert_eq!(fill.delivered, 1000);
        assert_eq!(conn.rcv_nxt(), SeqNum(IRS + 1000));
    }

    #[test]
    fn send_is_gated_and_queues_for_retransmit() {
        let mut conn = path();
        let segment = conn
            .try_send(500, 0, Instant::from_micros(1))
            .expect("window open");
        assert_eq!(segment.seq, SeqNum(ISS));
        assert_eq!(segment.len, 500);
    }

    #[test]
    fn ack_of_sent_data_clears_retransmit_and_samples_rtt() {
        let mut conn = path();
        conn.try_send(500, 0, Instant::from_micros(0))
            .expect("window open");
        // peer acks the 500 bytes at t=100ms.
        let out = conn.on_segment(
            data(),
            SeqNum(IRS),
            SeqNum(ISS + 500),
            64_000,
            0,
            Instant::from_micros(100_000),
        );
        assert!(!out.connection_reset);
        // slow start grew cwnd by one acked segment: 10·SMSS + 500 = 10500.
        assert_eq!(conn.cwnd(), 10_500);
    }

    #[test]
    fn three_dup_acks_trigger_fast_retransmit() {
        let mut conn = path();
        conn.try_send(500, 0, Instant::from_micros(0))
            .expect("window open");
        conn.try_send(500, 500, Instant::from_micros(1))
            .expect("window open");
        // three dup acks for the first segment (ack stays at ISS).
        let dup = |conn: &mut DataPath<8, 16, Reno>| {
            conn.on_segment(
                data(),
                SeqNum(IRS),
                SeqNum(ISS),
                64_000,
                0,
                Instant::from_micros(2),
            )
        };
        assert!(dup(&mut conn).fast_retransmit.is_none());
        assert!(dup(&mut conn).fast_retransmit.is_none());
        let third = dup(&mut conn);
        assert!(third.fast_retransmit.is_some());
        assert_eq!(third.fast_retransmit.expect("retransmit").seq, SeqNum(ISS));
    }

    #[test]
    fn rst_segment_reports_reset() {
        let mut conn = path();
        let out = conn.on_segment(
            Segment {
                rst: true,
                ..Segment::default()
            },
            SeqNum(IRS),
            SeqNum(ISS),
            64_000,
            0,
            Instant::ZERO,
        );
        assert!(out.connection_reset);
    }

    #[test]
    fn rto_poll_retransmits_oldest() {
        let mut conn = path();
        conn.try_send(500, 0, Instant::from_micros(0))
            .expect("window open");
        // before the deadline: nothing.
        assert!(conn.poll(Instant::from_micros(1)).is_none());
        // RTO_MIN is 1s; poll well past it.
        let retransmit = conn.poll(Instant::from_micros(2_000_000));
        assert!(retransmit.is_some());
        assert_eq!(retransmit.expect("retransmit").seq, SeqNum(ISS));
    }
}

#[cfg(all(test, feature = "std"))]
#[path = "smoltcp_differential.rs"]
mod smoltcp_differential;
