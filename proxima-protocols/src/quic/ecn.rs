//! ECN (Explicit Congestion Notification) per [RFC 9000 §13.4] +
//! [RFC 8311]. C18.
//!
//! [`EcnState`] is one-per-epoch state tracking outbound ECT codepoint
//! counts + inbound CE-ack reconciliation per RFC 9000 §13.4.2. The
//! sender marks outbound packets with [`EcnCodepoint::Ect0`]; routers
//! may rewrite to [`EcnCodepoint::Ce`] on congestion; the receiver
//! echoes per-codepoint counts in the ACK_ECN frame; on each ACK we
//! validate the counts and surface CE deltas as congestion events to
//! the [`CongestionController`].
//!
//! [RFC 9000 §13.4]: https://www.rfc-editor.org/rfc/rfc9000#section-13.4
//! [RFC 8311]: https://www.rfc-editor.org/rfc/rfc8311
//! [`CongestionController`]: crate::quic::congestion::CongestionController

use crate::quic::congestion::CongestionController;
use crate::quic::frame::EcnCounts;
use crate::quic::time::{Duration, Instant};

/// IP-level ECN codepoint per RFC 3168 §5 + RFC 8311.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EcnCodepoint {
    /// `0b00` — packet is NOT ECN-capable.
    NotEct,
    /// `0b01` — ECN-capable (legacy). QUIC senders rarely use this.
    Ect1,
    /// `0b10` — ECN-capable. QUIC's recommended default per
    /// RFC 9000 §13.4.1.
    Ect0,
    /// `0b11` — Congestion Experienced (router-set on congestion).
    Ce,
}

impl EcnCodepoint {
    /// Encode as the 2-bit IP-TOS field per RFC 3168.
    #[must_use]
    pub const fn as_bits(self) -> u8 {
        match self {
            Self::NotEct => 0b00,
            Self::Ect1 => 0b01,
            Self::Ect0 => 0b10,
            Self::Ce => 0b11,
        }
    }

    /// Decode from the 2-bit IP-TOS field; defaults to `NotEct` for
    /// unknown patterns (impossible — exhausted by 2 bits — but
    /// keeps the type signature total).
    #[must_use]
    pub const fn from_bits(bits: u8) -> Self {
        match bits & 0b11 {
            0b00 => Self::NotEct,
            0b01 => Self::Ect1,
            0b10 => Self::Ect0,
            _ => Self::Ce,
        }
    }
}

/// Per-epoch ECN validation state per RFC 9000 §13.4.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EcnMode {
    /// Sending ECT(0) outbound; validating the peer's counts.
    Attempting,
    /// Validation succeeded (3 consecutive valid ACK_ECNs). Continue
    /// sending ECT(0).
    Capable,
    /// Validation failed. Send `NotEct` and ignore CE counts.
    Disabled,
}

/// One ECN state per epoch.
#[derive(Debug, Clone, Copy)]
pub struct EcnState {
    pub mode: EcnMode,
    pub ce_acked: u64,
    pub ect0_sent: u64,
    pub ect1_sent: u64,
    /// Consecutive valid ACK_ECNs while in `Attempting` mode.
    pub validation_consecutive_oks: u8,
}

impl Default for EcnState {
    fn default() -> Self {
        Self::new()
    }
}

impl EcnState {
    /// Construct a fresh tracker in `Attempting` mode.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            mode: EcnMode::Attempting,
            ce_acked: 0,
            ect0_sent: 0,
            ect1_sent: 0,
            validation_consecutive_oks: 0,
        }
    }

    /// What codepoint to mark the next outbound packet with.
    #[must_use]
    pub const fn outbound_codepoint(&self) -> EcnCodepoint {
        match self.mode {
            EcnMode::Disabled => EcnCodepoint::NotEct,
            EcnMode::Attempting | EcnMode::Capable => EcnCodepoint::Ect0,
        }
    }

    /// Record that the FSM just sent a packet with the given codepoint
    /// in this epoch.
    pub fn on_packet_sent(&mut self, codepoint: EcnCodepoint) {
        match codepoint {
            EcnCodepoint::Ect0 => self.ect0_sent = self.ect0_sent.saturating_add(1),
            EcnCodepoint::Ect1 => self.ect1_sent = self.ect1_sent.saturating_add(1),
            EcnCodepoint::NotEct | EcnCodepoint::Ce => {}
        }
    }

    /// Process the ECN counts from an inbound ACK_ECN frame. Returns
    /// the count of new CE marks (delta) AND whether the ECN mode
    /// changed. Any CE delta > 0 has already been signalled to
    /// `controller.on_ecn_ce_seen` before this function returns.
    pub fn on_ack_with_ecn(
        &mut self,
        counts: EcnCounts,
        controller: &mut impl CongestionController,
        now: Instant,
        pto: Duration,
    ) -> EcnAckOutcome {
        if matches!(self.mode, EcnMode::Disabled) {
            return EcnAckOutcome {
                new_ce: 0,
                mode_changed: false,
            };
        }
        // Validation per RFC 9000 §13.4.2.
        let ok_a = counts.ect0.saturating_add(counts.ecn_ce) >= self.ect0_sent;
        let ok_b = counts.ect1 <= self.ect1_sent;
        let ok_c = counts.ecn_ce >= self.ce_acked;
        if !(ok_a && ok_b && ok_c) {
            self.mode = EcnMode::Disabled;
            return EcnAckOutcome {
                new_ce: 0,
                mode_changed: true,
            };
        }
        let new_ce = counts.ecn_ce - self.ce_acked;
        if new_ce > 0 {
            controller.on_ecn_ce_seen(now, pto);
        }
        self.ce_acked = counts.ecn_ce;
        let prior_mode = self.mode;
        if matches!(self.mode, EcnMode::Attempting) {
            self.validation_consecutive_oks = self.validation_consecutive_oks.saturating_add(1);
            if self.validation_consecutive_oks >= 3 {
                self.mode = EcnMode::Capable;
            }
        }
        EcnAckOutcome {
            new_ce,
            mode_changed: prior_mode != self.mode,
        }
    }
}

/// Outcome of an [`EcnState::on_ack_with_ecn`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EcnAckOutcome {
    pub new_ce: u64,
    pub mode_changed: bool,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::quic::congestion::NewReno;

    fn at(micros: u64) -> Instant {
        Instant::from_micros(micros)
    }

    #[test]
    fn codepoint_bit_round_trip() {
        for codepoint in [
            EcnCodepoint::NotEct,
            EcnCodepoint::Ect1,
            EcnCodepoint::Ect0,
            EcnCodepoint::Ce,
        ] {
            assert_eq!(EcnCodepoint::from_bits(codepoint.as_bits()), codepoint);
        }
    }

    #[test]
    fn new_state_attempts_ect0() {
        let state = EcnState::new();
        assert_eq!(state.mode, EcnMode::Attempting);
        assert_eq!(state.outbound_codepoint(), EcnCodepoint::Ect0);
    }

    #[test]
    fn validation_passes_then_ce_signals_congestion() {
        // docs/proxima-quic/c18-ecn-design.md worked example.
        let mut state = EcnState::new();
        let mut controller = NewReno::new(1200);
        controller.cwnd = 20_000;
        // Send PN 0 + PN 1 (both ECT(0)).
        state.on_packet_sent(EcnCodepoint::Ect0);
        state.on_packet_sent(EcnCodepoint::Ect0);
        assert_eq!(state.ect0_sent, 2);

        // ACK_ECN: {ect0:2, ect1:0, ecn_ce:0} → validate ✓, no CE.
        let outcome = state.on_ack_with_ecn(
            EcnCounts {
                ect0: 2,
                ect1: 0,
                ecn_ce: 0,
            },
            &mut controller,
            at(1_000_000),
            Duration::from_millis(300),
        );
        assert_eq!(outcome.new_ce, 0);
        assert!(!outcome.mode_changed);
        // cwnd unchanged.
        assert_eq!(controller.cwnd(), 20_000);

        // Send PN 2.
        state.on_packet_sent(EcnCodepoint::Ect0);

        // ACK_ECN: {ect0:2, ect1:0, ecn_ce:1} → validate ✓ (2+1=3 ≥ 3),
        // CE delta = 1 → controller.on_ecn_ce_seen → cwnd halves.
        let outcome = state.on_ack_with_ecn(
            EcnCounts {
                ect0: 2,
                ect1: 0,
                ecn_ce: 1,
            },
            &mut controller,
            at(2_000_000),
            Duration::from_millis(300),
        );
        assert_eq!(outcome.new_ce, 1);
        // cwnd was 20000, halved to 10000.
        assert_eq!(controller.cwnd(), 10_000);
        assert_eq!(state.ce_acked, 1);
    }

    #[test]
    fn under_count_disables_ecn() {
        let mut state = EcnState::new();
        let mut controller = NewReno::new(1200);
        state.on_packet_sent(EcnCodepoint::Ect0);
        state.on_packet_sent(EcnCodepoint::Ect0);
        state.on_packet_sent(EcnCodepoint::Ect0);
        // Peer claims only 1 ECT(0) — under-counting → disable.
        let outcome = state.on_ack_with_ecn(
            EcnCounts {
                ect0: 1,
                ect1: 0,
                ecn_ce: 0,
            },
            &mut controller,
            at(1_000_000),
            Duration::from_millis(300),
        );
        assert!(outcome.mode_changed);
        assert_eq!(state.mode, EcnMode::Disabled);
        assert_eq!(state.outbound_codepoint(), EcnCodepoint::NotEct);
    }

    #[test]
    fn unexpected_ect1_disables_ecn() {
        let mut state = EcnState::new();
        let mut controller = NewReno::new(1200);
        // We never sent ECT(1).
        state.on_packet_sent(EcnCodepoint::Ect0);
        let outcome = state.on_ack_with_ecn(
            EcnCounts {
                ect0: 0,
                ect1: 1,
                ecn_ce: 1,
            },
            &mut controller,
            at(1_000_000),
            Duration::from_millis(300),
        );
        assert!(outcome.mode_changed);
        assert_eq!(state.mode, EcnMode::Disabled);
    }

    #[test]
    fn decreasing_ce_count_disables_ecn() {
        let mut state = EcnState::new();
        let mut controller = NewReno::new(1200);
        state.on_packet_sent(EcnCodepoint::Ect0);
        // First ACK: ce_ack = 1.
        state.on_ack_with_ecn(
            EcnCounts {
                ect0: 0,
                ect1: 0,
                ecn_ce: 1,
            },
            &mut controller,
            at(1_000_000),
            Duration::from_millis(300),
        );
        assert_eq!(state.ce_acked, 1);
        // Second ACK: ce decreased to 0 → invalid.
        let outcome = state.on_ack_with_ecn(
            EcnCounts {
                ect0: 0,
                ect1: 0,
                ecn_ce: 0,
            },
            &mut controller,
            at(2_000_000),
            Duration::from_millis(300),
        );
        assert!(outcome.mode_changed);
        assert_eq!(state.mode, EcnMode::Disabled);
    }

    #[test]
    fn three_consecutive_validations_promote_to_capable() {
        let mut state = EcnState::new();
        let mut controller = NewReno::new(1200);
        for _ in 0..3 {
            state.on_packet_sent(EcnCodepoint::Ect0);
            state.on_ack_with_ecn(
                EcnCounts {
                    ect0: state.ect0_sent,
                    ect1: 0,
                    ecn_ce: 0,
                },
                &mut controller,
                at(1_000_000),
                Duration::from_millis(300),
            );
        }
        assert_eq!(state.mode, EcnMode::Capable);
    }

    #[test]
    fn disabled_mode_ignores_subsequent_acks() {
        let mut state = EcnState::new();
        let mut controller = NewReno::new(1200);
        state.mode = EcnMode::Disabled;
        let prior_cwnd = controller.cwnd();
        let outcome = state.on_ack_with_ecn(
            EcnCounts {
                ect0: 5,
                ect1: 0,
                ecn_ce: 3,
            },
            &mut controller,
            at(1_000_000),
            Duration::from_millis(300),
        );
        assert_eq!(outcome.new_ce, 0);
        assert!(!outcome.mode_changed);
        assert_eq!(controller.cwnd(), prior_cwnd);
    }
}
