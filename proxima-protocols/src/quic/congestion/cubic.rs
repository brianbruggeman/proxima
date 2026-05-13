//! CUBIC congestion control per [RFC 9438].
//!
//! Implements the cubic-growth congestion-avoidance + 0.7
//! multiplicative-decrease per the RFC. All arithmetic is integer
//! (`u64`); the `cbrt` helper uses Newton-Raphson iteration. The
//! TCP-friendly fallback (RFC 9438 §4.3) is documented as a follow-on
//! (C16.1) and not implemented in this first slice.
//!
//! [RFC 9438]: https://www.rfc-editor.org/rfc/rfc9438

use crate::quic::loss::SentPacket;
use crate::quic::time::{Duration, Instant};

use super::CongestionController;
use super::constants::{
    DEFAULT_MAX_DATAGRAM_SIZE, K_INITIAL_WINDOW_DATAGRAMS, K_MIN_WINDOW_DATAGRAMS,
    K_PERSISTENT_CONGESTION_THRESHOLD,
};

/// Multiplicative-decrease numerator per RFC 9438 §5.1 (β = 0.7).
pub const CUBIC_BETA_NUM: u64 = 7;
/// Multiplicative-decrease denominator.
pub const CUBIC_BETA_DENOM: u64 = 10;
/// Cubic scaling-constant numerator per RFC 9438 §5.1 (C = 0.4 per s³).
pub const CUBIC_C_NUM: u64 = 4;
/// Cubic scaling-constant denominator.
pub const CUBIC_C_DENOM: u64 = 10;

/// CUBIC controller.
#[derive(Debug, Clone, Copy)]
pub struct Cubic {
    pub cwnd: u64,
    pub ssthresh: Option<u64>,
    pub bytes_in_flight: u64,
    pub max_datagram_size: u64,
    pub congestion_recovery_start_time: Option<Instant>,
    /// W_max from RFC 9438 §5.2 — cwnd in BYTES at the most recent
    /// loss event.
    pub w_max_bytes: u64,
    /// K from RFC 9438 §4.2 — time-to-recover, in milliseconds.
    pub k_ms: u64,
    /// Instant of the most recent loss event (for `t` in the cubic
    /// function).
    pub last_loss_instant: Option<Instant>,
    /// Latest RTT estimate from the C14 RTT estimator. Used in the
    /// `W_cubic(t + RTT)` lookahead.
    pub rtt: Duration,
}

impl Default for Cubic {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_DATAGRAM_SIZE)
    }
}

impl Cubic {
    /// Construct a fresh controller with `max_datagram_size = mtu`.
    #[must_use]
    pub const fn new(mtu: u64) -> Self {
        Self {
            cwnd: K_INITIAL_WINDOW_DATAGRAMS * mtu,
            ssthresh: None,
            bytes_in_flight: 0,
            max_datagram_size: mtu,
            congestion_recovery_start_time: None,
            w_max_bytes: 0,
            k_ms: 0,
            last_loss_instant: None,
            rtt: Duration::from_micros(333_000),
        }
    }

    /// Update the cached RTT estimate (called from the FSM after each
    /// `LossDetection::on_ack_received`).
    pub fn update_rtt(&mut self, rtt: Duration) {
        self.rtt = rtt;
    }

    /// kMinimumWindow = 2 × MTU per RFC 9002 §7.2.
    #[must_use]
    pub const fn minimum_window(&self) -> u64 {
        K_MIN_WINDOW_DATAGRAMS * self.max_datagram_size
    }

    /// Are we currently inside a congestion-recovery window started no
    /// later than `sent_time` per RFC 9002 §7.3.2?
    #[must_use]
    pub fn in_congestion_recovery(&self, sent_time: Instant) -> bool {
        match self.congestion_recovery_start_time {
            Some(start) => sent_time <= start,
            None => false,
        }
    }
}

impl CongestionController for Cubic {
    fn on_packet_sent(&mut self, bytes: u64) {
        self.bytes_in_flight = self.bytes_in_flight.saturating_add(bytes);
    }

    fn on_packet_acked(&mut self, packet: &SentPacket, now: Instant) {
        let bytes = u64::from(packet.size_bytes);
        self.bytes_in_flight = self.bytes_in_flight.saturating_sub(bytes);
        if self.in_congestion_recovery(packet.sent_time) {
            return;
        }
        if self.cwnd < self.ssthresh.unwrap_or(u64::MAX) {
            // Slow start, same as NewReno.
            self.cwnd = self.cwnd.saturating_add(bytes);
            return;
        }
        // CUBIC CA path.
        let mtu = self.max_datagram_size;
        let elapsed_ms = self
            .last_loss_instant
            .and_then(|loss| now.duration_since(loss))
            .map(|delta| delta.as_micros() / 1_000)
            .unwrap_or(0);
        let rtt_ms = self.rtt.as_micros() / 1_000;
        let t_ms = elapsed_ms.saturating_add(rtt_ms);
        let w_max_segs = self.w_max_bytes / mtu.max(1);
        let target_segs = w_cubic_segments(t_ms, self.k_ms, w_max_segs);
        let cwnd_segs = self.cwnd / mtu.max(1);
        let increment = growth_per_ack(target_segs, cwnd_segs, mtu);
        self.cwnd = self.cwnd.saturating_add(increment);
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
            self.last_loss_instant = Some(now);
            let mtu = self.max_datagram_size.max(1);
            let cwnd_segs = self.cwnd / mtu;
            self.w_max_bytes = self.cwnd;
            // cwnd *= beta
            let reduced_segs = cwnd_segs.saturating_mul(CUBIC_BETA_NUM) / CUBIC_BETA_DENOM;
            let reduced_bytes = reduced_segs.saturating_mul(mtu);
            let floor = self.minimum_window();
            let new_cwnd = reduced_bytes.max(floor);
            self.ssthresh = Some(new_cwnd);
            self.cwnd = new_cwnd;
            // K_ms = cbrt(W_max_segs * (1 - β) * 1e9 / C)
            //      = cbrt(W_max_segs * (BETA_DENOM - BETA_NUM) * CUBIC_C_DENOM * 1e9
            //             / (CUBIC_C_NUM * BETA_DENOM))
            let numerator = cwnd_segs
                .saturating_mul(CUBIC_BETA_DENOM - CUBIC_BETA_NUM)
                .saturating_mul(CUBIC_C_DENOM)
                .saturating_mul(1_000_000_000);
            let denominator = CUBIC_C_NUM.saturating_mul(CUBIC_BETA_DENOM);
            self.k_ms = cbrt_u64(numerator / denominator.max(1));
        }
        if lost.len() >= 2 {
            let latest = lost.iter().map(|p| p.sent_time).max().unwrap_or(earliest);
            let span = latest.duration_since(earliest).unwrap_or(Duration::ZERO);
            let threshold = pto.saturating_mul(K_PERSISTENT_CONGESTION_THRESHOLD);
            if span.as_micros() >= threshold.as_micros() {
                self.cwnd = self.minimum_window();
                self.congestion_recovery_start_time = None;
                self.w_max_bytes = 0;
                self.k_ms = 0;
                self.last_loss_instant = None;
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
        // last_loss_instant intentionally untouched: even though the
        // anchor might point at a discarded packet, leaving it
        // produces only a longer CUBIC `t` value (conservative), not
        // incorrect math.
        self.bytes_in_flight = self.bytes_in_flight.saturating_sub(total_bytes);
    }

    fn on_ecn_ce_seen(&mut self, now: Instant, _pto: Duration) {
        if self.in_congestion_recovery(now) {
            return;
        }
        self.congestion_recovery_start_time = Some(now);
        self.last_loss_instant = Some(now);
        let mtu = self.max_datagram_size.max(1);
        let cwnd_segs = self.cwnd / mtu;
        self.w_max_bytes = self.cwnd;
        let reduced_segs = cwnd_segs.saturating_mul(CUBIC_BETA_NUM) / CUBIC_BETA_DENOM;
        let reduced_bytes = reduced_segs.saturating_mul(mtu);
        let floor = self.minimum_window();
        let new_cwnd = reduced_bytes.max(floor);
        self.ssthresh = Some(new_cwnd);
        self.cwnd = new_cwnd;
        let numerator = cwnd_segs
            .saturating_mul(CUBIC_BETA_DENOM - CUBIC_BETA_NUM)
            .saturating_mul(CUBIC_C_DENOM)
            .saturating_mul(1_000_000_000);
        let denominator = CUBIC_C_NUM.saturating_mul(CUBIC_BETA_DENOM);
        self.k_ms = cbrt_u64(numerator / denominator.max(1));
    }
}

/// Integer cube root via Newton-Raphson. Converges in ≤10 iterations
/// for any `u64` input (proven by the leading-zeros seed).
#[must_use]
pub fn cbrt_u64(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let bits = 64 - n.leading_zeros();
    let mut x: u64 = 1u64 << (bits / 3 + 1);
    for _ in 0..10 {
        let x2 = x.saturating_mul(x).max(1);
        let next = (2u64.saturating_mul(x) + n / x2) / 3;
        if next >= x {
            return x;
        }
        x = next;
    }
    x
}

/// Compute `W_cubic(t_ms)` in segments per RFC 9438 §4.1.
#[must_use]
pub fn w_cubic_segments(t_ms: u64, k_ms: u64, w_max_segs: u64) -> u64 {
    let delta_ms = t_ms.saturating_sub(k_ms);
    let delta_cubed = delta_ms.saturating_mul(delta_ms).saturating_mul(delta_ms);
    let growth =
        delta_cubed.saturating_mul(CUBIC_C_NUM) / CUBIC_C_DENOM.saturating_mul(1_000_000_000);
    w_max_segs.saturating_add(growth)
}

/// Compute the per-ACK cwnd increment in BYTES per RFC 9438 §4.2.
#[must_use]
pub fn growth_per_ack(target_segs: u64, cwnd_segs: u64, mss: u64) -> u64 {
    if target_segs <= cwnd_segs {
        // Slow probing past W_max; RFC 9438 §4.2 falls back to
        // cwnd += mss / (100 * cwnd_segs) — capped at zero in integer
        // arithmetic for any cwnd above mss.
        return mss / (100u64.saturating_mul(cwnd_segs.max(1)));
    }
    let cnt_recip = target_segs - cwnd_segs;
    mss.saturating_mul(cnt_recip) / cwnd_segs.max(1)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn cbrt_perfect_cubes() {
        assert_eq!(cbrt_u64(0), 0);
        assert_eq!(cbrt_u64(1), 1);
        assert_eq!(cbrt_u64(8), 2);
        assert_eq!(cbrt_u64(27), 3);
        assert_eq!(cbrt_u64(1_000_000_000_000), 10_000);
        assert_eq!(cbrt_u64(125_000_000_000_000), 50_000);
    }

    #[test]
    fn cbrt_non_perfect_within_one() {
        // Cube root of 1_000_000_000 = exactly 1000.
        assert_eq!(cbrt_u64(1_000_000_000), 1_000);
        // Cube root of 1_000_000_001 should be 1000 (truncated).
        let result = cbrt_u64(1_000_000_001);
        assert!(
            result == 1_000 || result == 1_001,
            "cbrt(1_000_000_001) = {result}; expected 1000 or 1001"
        );
    }

    #[test]
    fn k_ms_for_wmax_100() {
        // docs/proxima-quic/c16-cubic-design.md K computation:
        //   W_max=100, beta=7/10 → K_ms = cbrt(100 * 3/10 * 10/4 * 1e9)
        //   numerator = 100 * 3 * 10 * 1e9 = 3e12
        //   denominator = 4 * 10 = 40
        //   K_ms = cbrt(3e12 / 40) = cbrt(75_000_000_000) ≈ 4217
        let numerator =
            100u64 * (CUBIC_BETA_DENOM - CUBIC_BETA_NUM) * CUBIC_C_DENOM * 1_000_000_000;
        let denominator = CUBIC_C_NUM * CUBIC_BETA_DENOM;
        let k_ms = cbrt_u64(numerator / denominator);
        // 4217^3 = 74_998_137_113
        // 4218^3 = 75_051_491_592
        // → cbrt(75_000_000_000) should land at 4217.
        assert!(
            (4216..=4218).contains(&k_ms),
            "expected K_ms ≈ 4217; got {k_ms}",
        );
    }

    #[test]
    fn w_cubic_at_k_equals_wmax() {
        let w_max_segs = 100u64;
        let k_ms = 4217u64;
        assert_eq!(w_cubic_segments(k_ms, k_ms, w_max_segs), w_max_segs);
    }

    #[test]
    fn w_cubic_growth_after_3s_past_k() {
        // delta=3000, delta^3 = 2.7e10, growth = 2.7e10 * 4 / (10 * 1e9) = 10
        let growth = w_cubic_segments(3000, 0, 0);
        assert_eq!(growth, 10);
    }

    #[test]
    fn growth_per_ack_returns_positive_when_target_exceeds_cwnd() {
        let mss = 1200u64;
        let cwnd_segs = 70u64;
        let target_segs = 80u64;
        // cnt_recip = 10; growth = 1200 * 10 / 70 = 171 (truncated).
        assert_eq!(growth_per_ack(target_segs, cwnd_segs, mss), 171);
    }

    #[test]
    fn growth_per_ack_minimal_when_target_below_cwnd() {
        let mss = 1200u64;
        let cwnd_segs = 100u64;
        // fallback: mss / (100 * cwnd_segs) = 1200 / 10000 = 0
        assert_eq!(growth_per_ack(50, cwnd_segs, mss), 0);
    }

    #[test]
    fn loss_event_halves_via_beta_factor() {
        // cwnd=120000 (100 segs); beta=7/10 → cwnd=84000 (70 segs).
        let mut c = Cubic::new(1200);
        c.cwnd = 120_000;
        c.bytes_in_flight = 120_000;
        let lost = [SentPacket {
            packet_number: 0,
            sent_time: Instant::from_micros(1_000_000),
            size_bytes: 1200,
            is_ack_eliciting: true,
            in_flight: true,
        }];
        c.on_packets_lost(
            &lost,
            Instant::from_micros(2_000_000),
            Duration::from_millis(300),
        );
        assert_eq!(c.cwnd, 84_000);
        assert_eq!(c.ssthresh, Some(84_000));
        assert_eq!(c.w_max_bytes, 120_000);
        // K_ms ≈ 4217 from the walked example.
        assert!(
            (4216..=4218).contains(&c.k_ms),
            "K_ms should be ~4217; got {}",
            c.k_ms,
        );
    }

    #[test]
    fn slow_start_grows_like_newreno_before_first_loss() {
        let mut c = Cubic::new(1200);
        let pkt = SentPacket {
            packet_number: 0,
            sent_time: Instant::from_micros(1_000_000),
            size_bytes: 1200,
            is_ack_eliciting: true,
            in_flight: true,
        };
        c.on_packet_sent(12_000);
        c.on_packet_acked(&pkt, Instant::from_micros(1_050_000));
        assert_eq!(c.bytes_in_flight, 10_800);
        assert_eq!(c.cwnd, 13_200);
    }

    #[test]
    fn persistent_congestion_resets_state_like_newreno() {
        let mut c = Cubic::new(1200);
        c.cwnd = 8_000;
        c.bytes_in_flight = 8_000;
        c.w_max_bytes = 10_000;
        c.k_ms = 1234;
        let lost = [
            SentPacket {
                packet_number: 0,
                sent_time: Instant::from_micros(2_000_000),
                size_bytes: 1200,
                is_ack_eliciting: true,
                in_flight: true,
            },
            SentPacket {
                packet_number: 1,
                sent_time: Instant::from_micros(3_000_000),
                size_bytes: 1200,
                is_ack_eliciting: true,
                in_flight: true,
            },
        ];
        c.on_packets_lost(
            &lost,
            Instant::from_micros(4_000_000),
            Duration::from_millis(300),
        );
        assert_eq!(c.cwnd, c.minimum_window());
        assert_eq!(c.w_max_bytes, 0);
        assert_eq!(c.k_ms, 0);
        assert!(c.last_loss_instant.is_none());
    }
}
