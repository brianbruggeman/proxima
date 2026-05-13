//! Per-RFC-9000 §8.1 anti-amplification budget tracking.
//!
//! Until a server has verified the client's source address (via either
//! receipt of a Handshake-encrypted packet from the client or a successful
//! address-validation token), the server MUST NOT send more than 3 times
//! as many bytes as it has received from the client.
//!
//! The client is always considered to have a validated address (by
//! virtue of initiating the connection from an address it can receive
//! on); for client-side connections this budget is effectively infinite.
//!
//! C11 wires the counter into [`InitialState`] and [`HandshakeState`];
//! once the connection reaches [`EstablishedState`] the budget is no
//! longer relevant (address-validation has happened by then via the
//! Handshake-encrypted ACK).
//!
//! [`InitialState`]: crate::quic::connection::state::InitialState
//! [`HandshakeState`]: crate::quic::connection::state::HandshakeState
//! [`EstablishedState`]: crate::quic::connection::state::EstablishedState

use crate::quic::side::Side;

/// Tracks the RFC 9000 §8.1 anti-amplification budget for one connection
/// while address-validation is still pending.
#[derive(Debug, Clone, Copy)]
pub struct AntiAmplificationCounter {
    side: Side,
    received_from_peer: u64,
    sent_to_peer: u64,
    address_validated: bool,
}

impl AntiAmplificationCounter {
    /// Construct a fresh counter at zero bytes in each direction.
    ///
    /// Client-side counters are constructed with `address_validated = true`
    /// because the client's own address is implicitly validated by
    /// the act of receiving the server's responses.
    #[must_use]
    pub const fn new(side: Side) -> Self {
        Self {
            side,
            received_from_peer: 0,
            sent_to_peer: 0,
            address_validated: matches!(side, Side::Client),
        }
    }

    /// Record bytes received from the peer (the inbound datagram payload).
    pub fn record_received(&mut self, bytes: u64) {
        self.received_from_peer = self.received_from_peer.saturating_add(bytes);
    }

    /// Record bytes sent to the peer.
    pub fn record_sent(&mut self, bytes: u64) {
        self.sent_to_peer = self.sent_to_peer.saturating_add(bytes);
    }

    /// Mark the peer's address as validated; the budget no longer constrains.
    ///
    /// Called when:
    /// - The server receives a Handshake-encrypted packet from the client
    ///   (RFC 9000 §8.1: "[the client's address is] validated when the
    ///   server receives a Handshake packet from the client").
    /// - The server validates an address-validation token (RFC 9000 §8.1.3).
    pub fn mark_address_validated(&mut self) {
        self.address_validated = true;
    }

    /// Return the remaining bytes that may be sent right now.
    ///
    /// Returns `u64::MAX` once the address is validated (no constraint),
    /// otherwise `3 * received - sent` clipped at zero.
    #[must_use]
    pub fn send_budget(&self) -> u64 {
        if self.address_validated {
            return u64::MAX;
        }
        self.received_from_peer
            .saturating_mul(3)
            .saturating_sub(self.sent_to_peer)
    }

    /// Convenience: would sending `bytes` exceed the budget?
    #[must_use]
    pub fn can_send(&self, bytes: u64) -> bool {
        self.send_budget() >= bytes
    }

    /// Has the peer's address been validated?
    #[must_use]
    pub const fn address_validated(&self) -> bool {
        self.address_validated
    }

    /// Total bytes received from the peer to date.
    #[must_use]
    pub const fn received_from_peer(&self) -> u64 {
        self.received_from_peer
    }

    /// Total bytes sent to the peer to date.
    #[must_use]
    pub const fn sent_to_peer(&self) -> u64 {
        self.sent_to_peer
    }

    /// Which side owns this counter.
    #[must_use]
    pub const fn side(&self) -> Side {
        self.side
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn client_side_starts_unconstrained() {
        let counter = AntiAmplificationCounter::new(Side::Client);
        assert!(counter.address_validated());
        assert_eq!(counter.send_budget(), u64::MAX);
        assert!(counter.can_send(10_000));
    }

    #[test]
    fn server_side_starts_constrained() {
        let counter = AntiAmplificationCounter::new(Side::Server);
        assert!(!counter.address_validated());
        assert_eq!(counter.send_budget(), 0);
        assert!(!counter.can_send(1));
    }

    #[test]
    fn server_send_budget_is_three_times_received_minus_sent() {
        let mut counter = AntiAmplificationCounter::new(Side::Server);
        counter.record_received(1200);
        assert_eq!(counter.send_budget(), 3 * 1200);
        counter.record_sent(2000);
        assert_eq!(counter.send_budget(), 3 * 1200 - 2000);
    }

    #[test]
    fn server_send_budget_clips_at_zero_when_over_sent() {
        let mut counter = AntiAmplificationCounter::new(Side::Server);
        counter.record_received(100);
        counter.record_sent(1000);
        assert_eq!(counter.send_budget(), 0);
        assert!(!counter.can_send(1));
    }

    #[test]
    fn mark_address_validated_lifts_constraint() {
        let mut counter = AntiAmplificationCounter::new(Side::Server);
        counter.record_received(100);
        counter.record_sent(500);
        assert_eq!(counter.send_budget(), 0);
        counter.mark_address_validated();
        assert_eq!(counter.send_budget(), u64::MAX);
    }

    #[test]
    fn record_received_saturates_at_u64_max() {
        let mut counter = AntiAmplificationCounter::new(Side::Server);
        counter.record_received(u64::MAX);
        counter.record_received(1);
        assert_eq!(counter.received_from_peer(), u64::MAX);
    }
}
