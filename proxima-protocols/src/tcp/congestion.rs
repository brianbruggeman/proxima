//! TCP congestion control (RFC 5681 NewReno baseline).
//!
//! A trait plus the `Reno` reference implementation: slow-start until
//! `cwnd >= ssthresh`, then congestion avoidance; 3 duplicate ACKs trigger fast
//! retransmit + fast recovery; an RTO collapses to a one-segment loss window
//! and restarts slow start. In-flight bytes are NOT tracked here — the window
//! module owns `SND.NXT - SND.UNA` as the single source of truth and passes
//! `flight_size` in (see docs/tcp-data-path/edges.md E2).
//!
//! Equations cite RFC 5681 §3.1 (slow start / CA / RTO) and §3.2 (fast
//! retransmit / fast recovery). PRR (RFC 6937) and ABC (RFC 3465) are
//! deliberate out-of-scope enhancements.

/// Sans-IO congestion controller. The data path drives these on ACK events.
pub trait TcpCongestionControl {
    /// New data acknowledged (`bytes_acked` > 0). Grows cwnd or deflates out of
    /// fast recovery.
    fn on_ack(&mut self, bytes_acked: u32);

    /// A duplicate ACK arrived; `count` is the running run length and
    /// `flight_size` the currently outstanding bytes. Returns `true` exactly
    /// once — on the 3rd dup ACK — to tell the caller to fast-retransmit.
    fn on_dup_ack(&mut self, count: u8, flight_size: u32) -> bool;

    /// A retransmission timeout fired with `flight_size` outstanding.
    fn on_rto(&mut self, flight_size: u32);

    fn cwnd(&self) -> u32;
    fn ssthresh(&self) -> Option<u32>;
}

/// RFC 5681 NewReno controller.
#[derive(Debug, Clone, Copy)]
pub struct Reno {
    cwnd: u32,
    ssthresh: Option<u32>,
    smss: u32,
    in_fast_recovery: bool,
}

impl Reno {
    /// RFC 6928 initial window of 10 segments.
    const INITIAL_WINDOW_SEGMENTS: u32 = 10;

    /// Construct with the sender maximum segment size in bytes.
    #[must_use]
    pub const fn new(smss: u32) -> Self {
        Self {
            cwnd: smss.saturating_mul(Self::INITIAL_WINDOW_SEGMENTS),
            ssthresh: None,
            smss,
            in_fast_recovery: false,
        }
    }

    /// RFC 5681 §3.1 eq 4 / §3.2 step 2: `ssthresh = max(FlightSize/2, 2·SMSS)`.
    fn reduced_ssthresh(&self, flight_size: u32) -> u32 {
        (flight_size / 2).max(self.smss.saturating_mul(2))
    }
}

impl TcpCongestionControl for Reno {
    fn on_ack(&mut self, bytes_acked: u32) {
        if bytes_acked == 0 {
            return;
        }
        if self.in_fast_recovery {
            // RFC 5681 §3.2 step 6: the ACK that covers the retransmit ends
            // recovery; deflate cwnd back to ssthresh.
            self.in_fast_recovery = false;
            self.cwnd = self.ssthresh.unwrap_or(self.cwnd);
            return;
        }
        let threshold = self.ssthresh.unwrap_or(u32::MAX);
        if self.cwnd < threshold {
            // Slow start (RFC 5681 §3.1): cwnd += min(bytes_acked, SMSS).
            self.cwnd = self.cwnd.saturating_add(bytes_acked.min(self.smss));
        } else {
            // Congestion avoidance (RFC 5681 §3.1 eq 2):
            // cwnd += SMSS * bytes_acked / cwnd, at least 1 byte of progress.
            let cwnd = self.cwnd.max(1);
            let increment = (self.smss.saturating_mul(bytes_acked) / cwnd).max(1);
            self.cwnd = self.cwnd.saturating_add(increment);
        }
    }

    fn on_dup_ack(&mut self, count: u8, flight_size: u32) -> bool {
        match count {
            // RFC 5681 §3.2 steps 2-3: enter fast recovery, fast-retransmit.
            3 => {
                let ssthresh = self.reduced_ssthresh(flight_size);
                self.ssthresh = Some(ssthresh);
                self.cwnd = ssthresh.saturating_add(self.smss.saturating_mul(3));
                self.in_fast_recovery = true;
                true
            }
            // §3.2 step 4: inflate by one SMSS per additional dup ACK.
            n if n > 3 && self.in_fast_recovery => {
                self.cwnd = self.cwnd.saturating_add(self.smss);
                false
            }
            _ => false,
        }
    }

    fn on_rto(&mut self, flight_size: u32) {
        // RFC 5681 §3.1: collapse to a one-segment loss window, restart slow
        // start. ssthresh is set from the flight size at the time of loss.
        self.ssthresh = Some(self.reduced_ssthresh(flight_size));
        self.cwnd = self.smss;
        self.in_fast_recovery = false;
    }

    fn cwnd(&self) -> u32 {
        self.cwnd
    }

    fn ssthresh(&self) -> Option<u32> {
        self.ssthresh
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use proptest::prelude::*;

    const SMSS: u32 = 1000;

    /// An arbitrary sequence of congestion events that a proptest can apply.
    #[derive(Debug, Clone)]
    enum CongEvent {
        Ack(u32),
        DupAck { count: u8, flight: u32 },
        Rto(u32),
    }

    fn arb_cong_event() -> impl Strategy<Value = CongEvent> {
        prop_oneof![
            (1_u32..=65535).prop_map(CongEvent::Ack),
            (1_u8..=10, 0_u32..=65535)
                .prop_map(|(count, flight)| CongEvent::DupAck { count, flight }),
            (0_u32..=65535).prop_map(CongEvent::Rto),
        ]
    }

    proptest! {
        /// After a loss event `ssthresh` must be >= 2*SMSS (RFC 5681 §3.2 floor).
        #[test]
        fn ssthresh_is_at_least_two_smss_after_loss(
            smss in 1_u32..=9000,
            flight_size in 0_u32..=65535,
        ) {
            let mut control = Reno::new(smss);
            control.on_dup_ack(3, flight_size);
            if let Some(ssthresh) = control.ssthresh() {
                prop_assert!(
                    ssthresh >= smss.saturating_mul(2),
                    "ssthresh={ssthresh} below 2*smss={}", smss.saturating_mul(2)
                );
            }
        }

        /// `cwnd` must always be >= SMSS: a congestion window of zero would
        /// freeze the connection permanently.
        #[test]
        fn cwnd_never_drops_below_smss_over_arbitrary_events(
            smss in 1_u32..=9000,
            events in prop::collection::vec(arb_cong_event(), 0..32),
        ) {
            let mut control = Reno::new(smss);
            for event in events {
                match event {
                    CongEvent::Ack(bytes) => control.on_ack(bytes),
                    CongEvent::DupAck { count, flight } => { let _ = control.on_dup_ack(count, flight); }
                    CongEvent::Rto(flight) => control.on_rto(flight),
                }
                prop_assert!(
                    control.cwnd() >= smss,
                    "cwnd={} dropped below smss={smss}", control.cwnd()
                );
            }
        }

        /// `on_rto` / `on_dup_ack` / `on_ack` never panic for arbitrary inputs.
        #[test]
        fn congestion_events_never_panic(
            smss in 1_u32..=9000,
            events in prop::collection::vec(arb_cong_event(), 0..32),
        ) {
            let mut control = Reno::new(smss);
            for event in events {
                match event {
                    CongEvent::Ack(bytes) => control.on_ack(bytes),
                    CongEvent::DupAck { count, flight } => { let _ = control.on_dup_ack(count, flight); }
                    CongEvent::Rto(flight) => control.on_rto(flight),
                }
            }
        }
    }

    fn reno() -> Reno {
        Reno::new(SMSS)
    }

    #[test]
    fn initial_window_is_ten_segments() {
        assert_eq!(reno().cwnd(), 10_000);
        assert_eq!(reno().ssthresh(), None);
    }

    // Slow start: cwnd += min(bytes_acked, SMSS) (RFC 5681 §3.1).
    #[test]
    fn slow_start_grows_by_segment() {
        let mut control = reno();
        control.on_ack(1000);
        assert_eq!(control.cwnd(), 11_000);
        control.on_ack(4000); // capped at one SMSS per ACK
        assert_eq!(control.cwnd(), 12_000);
    }

    // Fast retransmit (RFC 5681 §3.2): flight 8000 -> ssthresh=4000,
    // cwnd=ssthresh+3·SMSS=7000, signals retransmit exactly on the 3rd dup ACK.
    #[test]
    fn third_dup_ack_enters_fast_recovery() {
        let mut control = reno();
        assert!(!control.on_dup_ack(1, 8000));
        assert!(!control.on_dup_ack(2, 8000));
        assert!(control.on_dup_ack(3, 8000));
        assert_eq!(control.ssthresh(), Some(4000));
        assert_eq!(control.cwnd(), 7000);
    }

    #[test]
    fn further_dup_acks_inflate_by_one_segment() {
        let mut control = reno();
        control.on_dup_ack(3, 8000);
        assert!(!control.on_dup_ack(4, 8000));
        assert_eq!(control.cwnd(), 8000);
        control.on_dup_ack(5, 8000);
        assert_eq!(control.cwnd(), 9000);
    }

    #[test]
    fn recovering_ack_deflates_to_ssthresh() {
        let mut control = reno();
        control.on_dup_ack(3, 8000);
        assert_eq!(control.cwnd(), 7000);
        control.on_ack(1000);
        assert_eq!(control.cwnd(), 4000); // ssthresh
        assert_eq!(control.ssthresh(), Some(4000));
    }

    // ssthresh floor is 2·SMSS even for tiny flight sizes (RFC 5681 §3.2).
    #[test]
    fn ssthresh_floors_at_two_segments() {
        let mut control = reno();
        control.on_dup_ack(3, 1000);
        assert_eq!(control.ssthresh(), Some(2000));
    }

    // RTO (RFC 5681 §3.1): ssthresh=max(flight/2,2·SMSS), cwnd=1·SMSS.
    #[test]
    fn rto_collapses_to_loss_window() {
        let mut control = reno();
        control.on_ack(1000); // cwnd 11000
        control.on_rto(8000);
        assert_eq!(control.ssthresh(), Some(4000));
        assert_eq!(control.cwnd(), 1000);
    }

    // Congestion avoidance: once cwnd >= ssthresh, growth is sub-linear.
    // After RTO: cwnd=1000, ssthresh=4000. Slow start climbs 1000->4000 in 3
    // ACKs (cwnd == ssthresh); the 4th ACK is CA: +SMSS·1000/4000 = +250.
    #[test]
    fn congestion_avoidance_grows_sublinearly() {
        let mut control = reno();
        control.on_rto(8000); // ssthresh=4000, cwnd=1000
        for _ in 0..3 {
            control.on_ack(1000);
        }
        assert_eq!(control.cwnd(), 4000);
        control.on_ack(1000);
        assert_eq!(control.cwnd(), 4250);
    }
}
