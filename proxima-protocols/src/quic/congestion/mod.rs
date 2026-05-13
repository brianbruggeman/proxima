//! Congestion control per [RFC 9002 §7].
//!
//! C15 lands the [`CongestionController`] trait + the
//! [`NewReno`] reference implementation. [`Cubic`] (RFC 9438) is
//! C16; [`Bbr2`] (draft-ietf-ccwg-bbr) is C17. All implementations
//! plug into the trait.
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`). State is plain `u64` fields +
//! `Option<Instant>` recovery anchor.
//!
//! [RFC 9002 §7]: https://www.rfc-editor.org/rfc/rfc9002#section-7

pub mod bbr;
pub mod constants;
pub mod cubic;
pub mod new_reno;

pub use constants::{
    DEFAULT_MAX_DATAGRAM_SIZE, K_INITIAL_WINDOW_DATAGRAMS, K_LOSS_REDUCTION_DENOM,
    K_LOSS_REDUCTION_NUM, K_MIN_WINDOW_DATAGRAMS, K_PERSISTENT_CONGESTION_THRESHOLD,
};
pub use cubic::Cubic;
pub use new_reno::NewReno;

use crate::quic::loss::SentPacket;
use crate::quic::time::{Duration, Instant};

/// Sans-IO congestion-controller trait. Per the C15 design pass each
/// implementation owns its `cwnd` + `bytes_in_flight` + recovery
/// state; the proto layer drives via the methods below.
pub trait CongestionController {
    /// Called when a packet is queued for transmission.
    /// `bytes` should be `0` for packets that are NOT in flight
    /// (RFC 9002 §A.1) — typically pure-ACK datagrams.
    fn on_packet_sent(&mut self, bytes: u64);

    /// Called for each acknowledged packet (one call per packet).
    /// Drives slow-start vs CA growth.
    fn on_packet_acked(&mut self, packet: &SentPacket, now: Instant);

    /// Called once per loss event with the full list of lost packets
    /// (the same `LossOutcome.lost` slice from C14). The controller
    /// applies the cwnd reduction once and checks persistent congestion.
    fn on_packets_lost(&mut self, lost: &[SentPacket], now: Instant, pto: Duration);

    /// Bytes the caller is allowed to send right now (cwnd minus
    /// in-flight, saturating at zero).
    fn send_budget(&self) -> u64;

    /// Current `bytes_in_flight` per RFC 9002 §A.1.
    fn bytes_in_flight(&self) -> u64;

    /// Current congestion window in bytes.
    fn cwnd(&self) -> u64;

    /// Current slow-start threshold; `None` until the first loss event.
    fn ssthresh(&self) -> Option<u64>;

    /// Called when an ACK_ECN frame's `ecn_ce` count increases (RFC
    /// 9000 §13.4.2). Treats as one congestion event — equivalent to
    /// declaring a single packet lost from the cwnd-reduction
    /// perspective. Default impl is a no-op for controllers that
    /// don't care about ECN.
    fn on_ecn_ce_seen(&mut self, _now: Instant, _pto: Duration) {}

    /// Called when an entire PN space's loss state is discarded per
    /// RFC 9001 §4.9 / RFC 9002 §A.4 (Initial/Handshake keys are
    /// dropped). Release any `bytes_in_flight` charged to the
    /// discarded packets WITHOUT treating it as a congestion event —
    /// these packets are neither acknowledged nor lost; they are
    /// "no longer trackable" because the PN space is gone.
    ///
    /// `total_bytes` is the sum of `size_bytes` across every still-
    /// in-flight ack-eliciting packet that was in the discarded epoch.
    /// Default impl is a saturating decrement; controllers with extra
    /// state (recovery anchors anchored on a discarded packet) may
    /// override.
    fn on_packet_number_space_discarded(&mut self, total_bytes: u64);
}
