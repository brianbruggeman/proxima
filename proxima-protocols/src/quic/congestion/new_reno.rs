//! NewReno congestion control per [RFC 9002 §7.3].
//!
//! Implements the RFC 9002 reference algorithm: slow-start until
//! `cwnd >= ssthresh`, then congestion-avoidance growth at
//! `(max_datagram_size × bytes_acked) / cwnd` per ACK. Loss events
//! halve cwnd (clamped at `kMinimumWindow`) and start a
//! recovery window that filters out follow-on losses from the same
//! congestion burst. Persistent congestion (RFC 9002 §7.6) resets
//! cwnd to `kMinimumWindow` and exits recovery.
//!
//! [RFC 9002 §7.3]: https://www.rfc-editor.org/rfc/rfc9002#section-7.3

use crate::quic::loss::SentPacket;
use crate::quic::time::{Duration, Instant};

use super::CongestionController;
use super::constants::{
    DEFAULT_MAX_DATAGRAM_SIZE, K_INITIAL_WINDOW_DATAGRAMS, K_LOSS_REDUCTION_DENOM,
    K_LOSS_REDUCTION_NUM, K_MIN_WINDOW_DATAGRAMS, K_PERSISTENT_CONGESTION_THRESHOLD,
};

/// Reference NewReno controller.
#[derive(Debug, Clone, Copy)]
pub struct NewReno {
    pub cwnd: u64,
    pub ssthresh: Option<u64>,
    pub bytes_in_flight: u64,
    pub max_datagram_size: u64,
    /// `Some(t)` while we're in the congestion-recovery window starting
    /// at `t`. Per RFC 9002 §7.3.2, additional loss events whose
    /// `sent_time` falls inside this window do NOT trigger a fresh
    /// cwnd reduction.
    pub congestion_recovery_start_time: Option<Instant>,
}

impl Default for NewReno {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_DATAGRAM_SIZE)
    }
}

impl NewReno {
    /// Construct a fresh controller with `max_datagram_size = mtu`.
    #[must_use]
    pub const fn new(mtu: u64) -> Self {
        Self {
            cwnd: K_INITIAL_WINDOW_DATAGRAMS * mtu,
            ssthresh: None,
            bytes_in_flight: 0,
            max_datagram_size: mtu,
            congestion_recovery_start_time: None,
        }
    }

    /// `kMinimumWindow` = 2 × max_datagram_size per RFC 9002 §7.2.
    #[must_use]
    pub const fn minimum_window(&self) -> u64 {
        K_MIN_WINDOW_DATAGRAMS * self.max_datagram_size
    }

    /// Returns `true` if the controller is currently inside a recovery
    /// window started no later than `sent_time` per RFC 9002 §7.3.2.
    #[must_use]
    pub fn in_congestion_recovery(&self, sent_time: Instant) -> bool {
        match self.congestion_recovery_start_time {
            Some(start) => sent_time <= start,
            None => false,
        }
    }
}

impl CongestionController for NewReno {
    fn on_packet_sent(&mut self, bytes: u64) {
        self.bytes_in_flight = self.bytes_in_flight.saturating_add(bytes);
    }

    fn on_packet_acked(&mut self, packet: &SentPacket, _now: Instant) {
        let bytes = u64::from(packet.size_bytes);
        self.bytes_in_flight = self.bytes_in_flight.saturating_sub(bytes);
        // Do not grow cwnd for packets that were sent during congestion
        // recovery (RFC 9002 §7.3.2).
        if self.in_congestion_recovery(packet.sent_time) {
            return;
        }
        if self.cwnd < self.ssthresh.unwrap_or(u64::MAX) {
            // Slow start (RFC 9002 §7.3.1).
            self.cwnd = self.cwnd.saturating_add(bytes);
        } else {
            // Congestion avoidance (RFC 9002 §7.3.3).
            // cwnd += (max_datagram_size * bytes_acked) / cwnd
            let cwnd = self.cwnd.max(1);
            let increment = self.max_datagram_size.saturating_mul(bytes) / cwnd;
            self.cwnd = self.cwnd.saturating_add(increment);
        }
    }

    fn on_packets_lost(&mut self, lost: &[SentPacket], now: Instant, pto: Duration) {
        for packet in lost {
            self.bytes_in_flight = self
                .bytes_in_flight
                .saturating_sub(u64::from(packet.size_bytes));
        }
        let earliest = match lost.iter().map(|p| p.sent_time).min() {
            Some(t) => t,
            None => return,
        };
        if !self.in_congestion_recovery(earliest) {
            self.congestion_recovery_start_time = Some(now);
            let halved = self.cwnd * K_LOSS_REDUCTION_NUM / K_LOSS_REDUCTION_DENOM;
            let floor = self.minimum_window();
            let new_cwnd = halved.max(floor);
            self.ssthresh = Some(new_cwnd);
            self.cwnd = new_cwnd;
        }
        if lost.len() >= 2 {
            // `lost.iter().max()` on a non-empty slice never returns None;
            // unwrap_or(earliest) keeps the math defensive AND clippy-clean.
            let latest = lost.iter().map(|p| p.sent_time).max().unwrap_or(earliest);
            // duration_since is None when latest < earliest (shouldn't
            // happen because we computed earliest=min above, but guard).
            let span = latest.duration_since(earliest).unwrap_or(Duration::ZERO);
            let threshold = pto.saturating_mul(K_PERSISTENT_CONGESTION_THRESHOLD);
            if span.as_micros() >= threshold.as_micros() {
                self.cwnd = self.minimum_window();
                self.congestion_recovery_start_time = None;
            }
        }
    }

    fn send_budget(&self) -> u64 {
        self.cwnd.saturating_sub(self.bytes_in_flight)
    }

    fn bytes_in_flight(&self) -> u64 {
        self.bytes_in_flight
    }

    fn cwnd(&self) -> u64 {
        self.cwnd
    }

    fn ssthresh(&self) -> Option<u64> {
        self.ssthresh
    }

    fn on_packet_number_space_discarded(&mut self, total_bytes: u64) {
        // RFC 9002 §A.4 — release in-flight bytes from the discarded
        // PN space without invoking the loss/cwnd-reduction path.
        self.bytes_in_flight = self.bytes_in_flight.saturating_sub(total_bytes);
    }

    fn on_ecn_ce_seen(&mut self, now: Instant, _pto: Duration) {
        // RFC 9000 §13.4.2 — treat CE as a congestion event; halve cwnd
        // gated by the same recovery-window filter as loss events.
        // Synthesise the "event sent_time" as `now` so the recovery
        // window will block follow-on CE events in the same burst.
        if self.in_congestion_recovery(now) {
            return;
        }
        self.congestion_recovery_start_time = Some(now);
        let halved = self.cwnd * K_LOSS_REDUCTION_NUM / K_LOSS_REDUCTION_DENOM;
        let floor = self.minimum_window();
        let new_cwnd = halved.max(floor);
        self.ssthresh = Some(new_cwnd);
        self.cwnd = new_cwnd;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn at(micros: u64) -> Instant {
        Instant::from_micros(micros)
    }

    fn packet(pn: u64, sent_time: Instant) -> SentPacket {
        SentPacket {
            packet_number: pn,
            sent_time,
            size_bytes: 1200,
            is_ack_eliciting: true,
            in_flight: true,
        }
    }

    #[test]
    fn new_controller_uses_initial_window() {
        let nr = NewReno::new(1200);
        assert_eq!(nr.cwnd, 12_000); // 10 * 1200
        assert_eq!(nr.ssthresh, None);
        assert_eq!(nr.bytes_in_flight, 0);
        assert_eq!(nr.send_budget(), 12_000);
    }

    #[test]
    fn slow_start_growth_walked_example() {
        // docs/proxima-quic/c15-newreno-design.md slow-start example.
        let mut nr = NewReno::new(1200);
        nr.on_packet_sent(12_000); // saturate cwnd
        let pkt = packet(0, at(1_000_000));
        nr.on_packet_acked(&pkt, at(1_050_000));
        assert_eq!(nr.bytes_in_flight, 10_800);
        assert_eq!(nr.cwnd, 13_200);
        assert_eq!(nr.ssthresh, None);
    }

    #[test]
    fn congestion_avoidance_growth_walked_example() {
        // Set up: cwnd=13200, ssthresh=Some(13200), bif=13200.
        let mut nr = NewReno::new(1200);
        nr.cwnd = 13_200;
        nr.ssthresh = Some(13_200);
        nr.bytes_in_flight = 13_200;
        let pkt = packet(0, at(1_000_000));
        nr.on_packet_acked(&pkt, at(1_050_000));
        assert_eq!(nr.bytes_in_flight, 12_000);
        // Expected: cwnd += (1200 * 1200) / 13200 = 109 (integer).
        assert_eq!(nr.cwnd, 13_309);
    }

    #[test]
    fn loss_event_halves_cwnd_walked_example() {
        let mut nr = NewReno::new(1200);
        nr.cwnd = 20_000;
        nr.bytes_in_flight = 20_000;
        let lost = [packet(0, at(1_000_000))];
        nr.on_packets_lost(&lost, at(1_100_000), Duration::from_millis(300));
        assert_eq!(nr.bytes_in_flight, 18_800);
        assert_eq!(nr.cwnd, 10_000);
        assert_eq!(nr.ssthresh, Some(10_000));
        assert!(nr.congestion_recovery_start_time.is_some());
    }

    #[test]
    fn second_loss_in_same_recovery_window_does_not_reduce_again() {
        let mut nr = NewReno::new(1200);
        nr.cwnd = 20_000;
        nr.bytes_in_flight = 20_000;
        let lost = [packet(0, at(1_000_000))];
        nr.on_packets_lost(&lost, at(1_100_000), Duration::from_millis(300));
        assert_eq!(nr.cwnd, 10_000);
        // Another loss whose sent_time is BEFORE recovery_start (1_100_000)
        // — same congestion burst, no fresh reduction.
        let same_burst = [packet(1, at(1_050_000))];
        nr.on_packets_lost(&same_burst, at(1_150_000), Duration::from_millis(300));
        assert_eq!(nr.cwnd, 10_000);
        assert_eq!(nr.ssthresh, Some(10_000));
    }

    #[test]
    fn loss_after_recovery_window_ends_reduces_again() {
        let mut nr = NewReno::new(1200);
        nr.cwnd = 20_000;
        nr.bytes_in_flight = 20_000;
        // First loss: recovery_start = 1_100_000.
        nr.on_packets_lost(
            &[packet(0, at(1_000_000))],
            at(1_100_000),
            Duration::from_millis(300),
        );
        assert_eq!(nr.cwnd, 10_000);
        // Bump bytes_in_flight back up + send a NEW packet AFTER recovery start.
        nr.bytes_in_flight = 8_000;
        // New loss whose sent_time is AFTER recovery_start.
        nr.on_packets_lost(
            &[packet(2, at(1_200_000))],
            at(1_300_000),
            Duration::from_millis(300),
        );
        assert_eq!(nr.cwnd, 5_000);
    }

    #[test]
    fn persistent_congestion_resets_cwnd_to_minimum_window() {
        // docs/proxima-quic/c15-newreno-design.md persistent-congestion example.
        let mut nr = NewReno::new(1200);
        nr.cwnd = 8_000;
        nr.ssthresh = Some(10_000);
        nr.bytes_in_flight = 8_000;
        let lost = [
            packet(0, at(2_000_000)),
            packet(1, at(2_000_000 + 1_000_000)), // 1000 ms later
        ];
        nr.on_packets_lost(&lost, at(3_000_000), Duration::from_millis(300));
        // Span = 1000 ms, threshold = 3 * 300 = 900 ms → triggers persistent.
        assert_eq!(nr.cwnd, nr.minimum_window());
        assert_eq!(nr.cwnd, 2_400);
        assert!(nr.congestion_recovery_start_time.is_none());
    }

    #[test]
    fn send_budget_clips_at_zero_when_over_in_flight() {
        let mut nr = NewReno::new(1200);
        nr.cwnd = 5_000;
        nr.bytes_in_flight = 8_000;
        assert_eq!(nr.send_budget(), 0);
    }

    #[test]
    fn recovery_filter_blocks_ack_during_recovery() {
        let mut nr = NewReno::new(1200);
        // Enter recovery.
        nr.on_packets_lost(
            &[packet(0, at(1_000_000))],
            at(1_100_000),
            Duration::from_millis(300),
        );
        let cwnd_before = nr.cwnd;
        // ACK for a packet sent BEFORE recovery_start — should NOT grow cwnd.
        nr.on_packet_acked(&packet(1, at(1_050_000)), at(1_200_000));
        assert_eq!(nr.cwnd, cwnd_before);
    }

    #[test]
    fn on_packet_number_space_discarded_releases_bytes_without_loss_event() {
        // Regression for RFC 9002 §A.4 — Initial/Handshake key
        // discard must release in-flight bytes WITHOUT triggering
        // the loss/recovery path. Otherwise cwnd is permanently
        // understated by the discarded packets' size.
        let mut nr = NewReno::new(1200);
        nr.on_packet_sent(3 * 1200); // 3 Handshake-sized packets in flight
        assert_eq!(nr.bytes_in_flight, 3_600);
        let cwnd_before = nr.cwnd;
        let ssthresh_before = nr.ssthresh;
        nr.on_packet_number_space_discarded(3_600);
        assert_eq!(nr.bytes_in_flight, 0, "in-flight released");
        assert_eq!(nr.cwnd, cwnd_before, "cwnd NOT halved on discard");
        assert_eq!(nr.ssthresh, ssthresh_before, "ssthresh untouched");
        assert!(
            nr.congestion_recovery_start_time.is_none(),
            "no recovery window entered"
        );
        assert_eq!(nr.send_budget(), nr.cwnd, "full cwnd available again");
    }

    #[test]
    fn on_packet_number_space_discarded_saturates_at_zero() {
        // Defensive: passing more than in-flight should saturate,
        // not panic / underflow.
        let mut nr = NewReno::new(1200);
        nr.on_packet_sent(1200);
        nr.on_packet_number_space_discarded(9_999);
        assert_eq!(nr.bytes_in_flight, 0);
    }
}
