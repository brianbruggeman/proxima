//! RFC 9221 unreliable DATAGRAM extension — per-direction queues +
//! the surface API the FSM exposes.
//!
//! Wire format (frame type 0x30/0x31) is already in C3's
//! [`crate::quic::frame::Frame::Datagram`]. This module adds the bounded
//! send/recv queues + the transport-parameter negotiation gating.
//!
//! # Tier
//!
//! Tier-1 (alloc). Per-payload size is variable, so each queued
//! datagram is a `Vec<u8>`. Queues are `heapless::Deque` with
//! const-generic caps from `proxima-quic-proto.toml [datagram]`.

extern crate alloc;

use alloc::vec::Vec;

use heapless::Deque;

use crate::quic::sized;

/// Maximum queued outbound DATAGRAMs. Sourced from
/// `proxima-quic-proto.toml [datagram].send_queue_cap`.
pub const SEND_QUEUE_CAP: usize = sized::DATAGRAM_SEND_QUEUE_CAP;

/// Maximum queued inbound DATAGRAMs awaiting `recv_datagram`.
pub const RECV_QUEUE_CAP: usize = sized::DATAGRAM_RECV_QUEUE_CAP;

/// Local-side `max_datagram_frame_size` per RFC 9221 §3 (in bytes).
/// 0 disables the DATAGRAM extension (we don't advertise; can't send;
/// rejects inbound).
pub const LOCAL_MAX_DATAGRAM_FRAME_SIZE: u64 = sized::DATAGRAM_LOCAL_MAX_DATAGRAM_FRAME_SIZE;

/// Errors from [`DatagramQueues::send`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DatagramSendError {
    /// Peer's `max_datagram_frame_size = 0` (or the TP wasn't sent)
    /// per RFC 9221 §3 — DATAGRAM not enabled for this connection.
    NotEnabled,
    /// Payload exceeds the peer's advertised `max_datagram_frame_size`.
    TooLarge { max: u64, attempted: usize },
    /// Send queue at capacity ([`SEND_QUEUE_CAP`]).
    QueueFull { cap: usize },
}

/// Per-connection DATAGRAM state — both queues + the
/// peer-advertised cap.
#[derive(Debug, Default, Clone)]
pub struct DatagramQueues {
    /// Outbound queue — `poll_transmit_established` drains as
    /// budget allows.
    send: Deque<Vec<u8>, SEND_QUEUE_CAP>,
    /// Inbound queue — `parse_and_apply_established` appends on
    /// receipt of `Frame::Datagram`; caller drains via
    /// `recv_datagram`.
    recv: Deque<Vec<u8>, RECV_QUEUE_CAP>,
    /// Peer's advertised `max_datagram_frame_size` (RFC 9221 §3).
    /// `None` until the peer-TP parse fires; once set, frozen for the
    /// connection lifetime.
    peer_max_datagram_frame_size: Option<u64>,
    /// OUR advertised `max_datagram_frame_size` from local TPs.
    /// RFC 9221 §5.2 — a peer sending DATAGRAM when we advertised
    /// 0 (or absent) is PROTOCOL_VIOLATION.
    local_max_datagram_frame_size: u64,
}

impl DatagramQueues {
    /// Construct empty queues with no peer cap.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            send: Deque::new(),
            recv: Deque::new(),
            peer_max_datagram_frame_size: None,
            local_max_datagram_frame_size: 0,
        }
    }

    /// Record our own advertised `max_datagram_frame_size`.
    pub fn set_local_max_datagram_frame_size(&mut self, local_max: u64) {
        self.local_max_datagram_frame_size = local_max;
    }

    /// Our advertised `max_datagram_frame_size` (0 = not advertised).
    #[must_use]
    pub fn local_max_datagram_frame_size(&self) -> u64 {
        self.local_max_datagram_frame_size
    }

    /// Record the peer's advertised `max_datagram_frame_size` once the
    /// TLS provider fires `PeerTransportParameters`. `0` is the RFC
    /// signal for "not enabled."
    pub fn set_peer_max_datagram_frame_size(&mut self, peer_max: u64) {
        self.peer_max_datagram_frame_size = Some(peer_max);
    }

    /// True iff DATAGRAMs are negotiated for this direction (peer
    /// advertised a non-zero `max_datagram_frame_size`).
    #[must_use]
    pub fn enabled_for_send(&self) -> bool {
        matches!(self.peer_max_datagram_frame_size, Some(cap) if cap > 0)
    }

    /// Bytes the peer claims it can accept per individual DATAGRAM.
    /// `None` if not yet negotiated.
    #[must_use]
    pub const fn peer_max(&self) -> Option<u64> {
        self.peer_max_datagram_frame_size
    }

    /// Enqueue an outbound DATAGRAM. Caller-owned bytes copied into
    /// the queue (variable size — alloc is unavoidable for tier-1).
    ///
    /// # Errors
    ///
    /// - [`DatagramSendError::NotEnabled`] when the peer hasn't
    ///   advertised the extension OR advertised `max=0`.
    /// - [`DatagramSendError::TooLarge`] when the payload exceeds
    ///   the peer's advertised limit.
    /// - [`DatagramSendError::QueueFull`] when the queue is at cap.
    pub fn send(&mut self, payload: &[u8]) -> Result<(), DatagramSendError> {
        let cap = match self.peer_max_datagram_frame_size {
            None | Some(0) => return Err(DatagramSendError::NotEnabled),
            Some(max) => max,
        };
        if (payload.len() as u64) > cap {
            return Err(DatagramSendError::TooLarge {
                max: cap,
                attempted: payload.len(),
            });
        }
        if self.send.is_full() {
            return Err(DatagramSendError::QueueFull {
                cap: SEND_QUEUE_CAP,
            });
        }
        let mut owned: Vec<u8> = Vec::with_capacity(payload.len());
        owned.extend_from_slice(payload);
        // try_push_back cannot fail because of the is_full check above.
        let _ = self.send.push_back(owned);
        Ok(())
    }

    /// Drain one outbound DATAGRAM if available (used by
    /// `poll_transmit_established`).
    pub fn pop_send(&mut self) -> Option<Vec<u8>> {
        self.send.pop_front()
    }

    /// Append an inbound DATAGRAM (used by
    /// `parse_and_apply_established` on receipt of `Frame::Datagram`).
    /// Returns `true` if accepted; `false` if the recv queue is full
    /// (caller should drop or log).
    pub fn push_recv(&mut self, payload: Vec<u8>) -> bool {
        if self.recv.is_full() {
            return false;
        }
        let _ = self.recv.push_back(payload);
        true
    }

    /// Drain one inbound DATAGRAM if available (caller API
    /// `Connection::recv_datagram`).
    pub fn recv(&mut self) -> Option<Vec<u8>> {
        self.recv.pop_front()
    }

    /// Number of outbound DATAGRAMs awaiting transmit.
    #[must_use]
    pub fn pending_send(&self) -> usize {
        self.send.len()
    }

    /// Number of inbound DATAGRAMs awaiting `recv_datagram`.
    #[must_use]
    pub fn pending_recv(&self) -> usize {
        self.recv.len()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn send_before_peer_tp_returns_not_enabled() {
        let mut queues = DatagramQueues::new();
        assert_eq!(queues.send(b"hello"), Err(DatagramSendError::NotEnabled));
    }

    #[test]
    fn send_after_peer_advertises_zero_returns_not_enabled() {
        let mut queues = DatagramQueues::new();
        queues.set_peer_max_datagram_frame_size(0);
        assert_eq!(queues.send(b"hello"), Err(DatagramSendError::NotEnabled));
    }

    #[test]
    fn send_with_payload_over_peer_cap_returns_too_large() {
        let mut queues = DatagramQueues::new();
        queues.set_peer_max_datagram_frame_size(3);
        let result = queues.send(b"hello");
        assert_eq!(
            result,
            Err(DatagramSendError::TooLarge {
                max: 3,
                attempted: 5,
            })
        );
    }

    #[test]
    fn send_within_cap_enqueues_and_pops_in_order() {
        let mut queues = DatagramQueues::new();
        queues.set_peer_max_datagram_frame_size(1200);
        queues.send(b"hello").expect("ok");
        queues.send(b"world").expect("ok");
        assert_eq!(queues.pending_send(), 2);
        assert_eq!(queues.pop_send().as_deref(), Some(&b"hello"[..]));
        assert_eq!(queues.pop_send().as_deref(), Some(&b"world"[..]));
        assert!(queues.pop_send().is_none());
    }

    #[test]
    fn send_at_capacity_returns_queue_full() {
        let mut queues = DatagramQueues::new();
        queues.set_peer_max_datagram_frame_size(1200);
        for _ in 0..SEND_QUEUE_CAP {
            queues.send(b"x").expect("within cap");
        }
        assert_eq!(
            queues.send(b"x"),
            Err(DatagramSendError::QueueFull {
                cap: SEND_QUEUE_CAP
            })
        );
    }

    #[test]
    fn push_recv_at_capacity_returns_false() {
        let mut queues = DatagramQueues::new();
        for _ in 0..RECV_QUEUE_CAP {
            assert!(queues.push_recv(alloc::vec![0u8]));
        }
        assert!(!queues.push_recv(alloc::vec![0u8]));
    }

    #[test]
    fn recv_pops_in_order() {
        let mut queues = DatagramQueues::new();
        queues.push_recv(b"first".to_vec());
        queues.push_recv(b"second".to_vec());
        assert_eq!(queues.recv().as_deref(), Some(&b"first"[..]));
        assert_eq!(queues.recv().as_deref(), Some(&b"second"[..]));
        assert!(queues.recv().is_none());
    }

    #[test]
    fn enabled_for_send_flips_on_nonzero_peer_max() {
        let mut queues = DatagramQueues::new();
        assert!(!queues.enabled_for_send());
        queues.set_peer_max_datagram_frame_size(0);
        assert!(!queues.enabled_for_send());
        queues.set_peer_max_datagram_frame_size(1);
        assert!(queues.enabled_for_send());
    }
}
