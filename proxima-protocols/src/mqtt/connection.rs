//! Sans-IO MQTT connection state machine — bytes in, [`Packet`] out.
//!
//! Mirrors [`crate::redis::connection::Connection`]'s `feed_bytes`/
//! `advance`/`consume` shape: one growing read buffer, a cursor so
//! pipelined packet bytes don't memcpy, and a typed [`Advanced`] outcome
//! the driver matches on. No socket, no tokio, no `.await` anywhere in
//! this file — the I/O edge (`proxima-mqtt`'s driver) owns reading bytes
//! off the wire and feeding them in.
//!
//! Unlike RESP, [`super::parse_packet`] already declares its own
//! "remaining length" up front, so there is no attacker-controlled-length
//! DoS gap the way `redis::connection` closes for `parse_blob` — but a
//! peer can still trickle a huge declared length one byte at a time,
//! growing this connection's buffer without bound while
//! [`super::ParseError::PartialPacket`] keeps returning "not yet".
//! [`Connection::advance`] catches that once the buffered-but-still-
//! incomplete bytes exceed [`Limits::max_message_bytes`], the same DoS
//! guard shape.

use alloc::vec::Vec;

use super::{ParseError, Packet, parse_packet};

/// A connection stays under this many buffered-but-unparsed bytes before a
/// still-incomplete packet is treated as oversized. 16 MiB matches
/// `redis::connection`'s default.
const DEFAULT_MAX_MESSAGE_BYTES: usize = 16 * 1024 * 1024;

/// Once the consumed prefix exceeds this many bytes, [`Connection::consume`]
/// compacts the buffer (drains the dead prefix) instead of just moving the
/// cursor — mirrors `redis::connection::Connection`'s identical threshold.
const COMPACT_THRESHOLD_BYTES: usize = 8 * 1024;

/// Byte caps a [`Connection`] enforces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    pub max_message_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_message_bytes: DEFAULT_MAX_MESSAGE_BYTES,
        }
    }
}

/// Typed outcome of [`Connection::advance`]. `Command`'s `packet` borrows
/// from the connection's internal buffer — the driver must extract
/// whatever it needs before calling [`Connection::consume`] or `advance`
/// again.
pub enum Advanced<'a> {
    /// The buffer holds a prefix of a packet; read more bytes and retry.
    NeedMore,
    /// One full MQTT packet parsed. `consumed` is the byte length to pass
    /// to [`Connection::consume`] once the driver is done with `packet`.
    Command { packet: Packet<'a>, consumed: usize },
    /// The buffered bytes violate MQTT framing. `consumed` is always 0 (a
    /// framing violation leaves no trustworthy packet boundary to skip
    /// past) — the driver closes the connection rather than trying to
    /// resync.
    ProtocolError { reason: &'static str, consumed: usize },
    /// A still-incomplete packet already exceeds
    /// [`Limits::max_message_bytes`] — the DoS guard tripped. The driver
    /// closes the connection.
    MessageTooLarge,
}

impl core::fmt::Debug for Advanced<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let variant = match self {
            Self::NeedMore => "NeedMore",
            Self::Command { .. } => "Command",
            Self::ProtocolError { .. } => "ProtocolError",
            Self::MessageTooLarge => "MessageTooLarge",
        };
        formatter
            .debug_struct("Advanced")
            .field("variant", &variant)
            .finish_non_exhaustive()
    }
}

/// Sans-IO MQTT connection state machine.
pub struct Connection {
    buffer: Vec<u8>,
    /// Logical start of the not-yet-consumed region. Advances on
    /// [`Self::consume`]; pipelined packet bytes past it are re-parsed in
    /// place, no memcpy.
    cursor: usize,
    limits: Limits,
}

impl Default for Connection {
    fn default() -> Self {
        Self::new()
    }
}

impl Connection {
    #[must_use]
    pub fn new() -> Self {
        Self::with_limits(Limits::default())
    }

    #[must_use]
    pub fn with_limits(limits: Limits) -> Self {
        Self {
            buffer: Vec::new(),
            cursor: 0,
            limits,
        }
    }

    /// Append bytes read off the wire.
    pub fn feed_bytes(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    /// Drive the state machine one step: try to parse one MQTT packet from
    /// the unconsumed buffer region.
    pub fn advance(&mut self) -> Advanced<'_> {
        match parse_packet(&self.buffer[self.cursor..]) {
            Ok((packet, consumed)) => Advanced::Command { packet, consumed },
            Err(ParseError::InvalidPacketType(_)) => Advanced::ProtocolError {
                reason: "packet type is reserved or invalid",
                consumed: 0,
            },
            Err(ParseError::Malformed(reason)) => Advanced::ProtocolError { reason, consumed: 0 },
            Err(ParseError::RemainingLengthOverflow) => Advanced::ProtocolError {
                reason: "remaining-length varint exceeds 4 bytes",
                consumed: 0,
            },
            Err(ParseError::Short | ParseError::PartialPacket(_)) => {
                if self.buffer.len() - self.cursor > self.limits.max_message_bytes {
                    Advanced::MessageTooLarge
                } else {
                    Advanced::NeedMore
                }
            }
        }
    }

    /// Advance past a parsed packet's bytes (the `consumed` a
    /// [`Advanced::Command`] carried). Compacts the buffer once the
    /// consumed prefix grows past [`COMPACT_THRESHOLD_BYTES`], or clears it
    /// outright once every buffered byte is consumed.
    pub fn consume(&mut self, amount: usize) {
        self.cursor += amount;
        if self.cursor >= self.buffer.len() {
            self.buffer.clear();
            self.cursor = 0;
        } else if self.cursor > COMPACT_THRESHOLD_BYTES {
            self.buffer.drain(..self.cursor);
            self.cursor = 0;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use alloc::vec;

    fn pingreq() -> Vec<u8> {
        vec![0xC0, 0x00]
    }

    fn publish_qos0(topic: &str, payload: &[u8]) -> Vec<u8> {
        let mut out = vec![0x30];
        let body_len = 2 + topic.len() + payload.len();
        crate::mqtt::encode::encode_remaining_length(body_len as u32, &mut out);
        out.extend_from_slice(&(topic.len() as u16).to_be_bytes());
        out.extend_from_slice(topic.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn partial_packet_returns_need_more() {
        let mut connection = Connection::new();
        connection.feed_bytes(&[0xC0]);
        assert!(matches!(connection.advance(), Advanced::NeedMore));
    }

    #[test]
    fn complete_packet_returns_command_with_consumed_length() {
        let mut connection = Connection::new();
        let wire = pingreq();
        connection.feed_bytes(&wire);
        match connection.advance() {
            Advanced::Command { packet, consumed } => {
                assert_eq!(consumed, wire.len());
                assert!(matches!(packet, Packet::PingReq));
            }
            other => panic!("expected Command, got {other:?}"),
        }
    }

    #[test]
    fn invalid_packet_type_returns_protocol_error_with_zero_consumed() {
        let mut connection = Connection::new();
        connection.feed_bytes(&[0x00, 0x00]);
        match connection.advance() {
            Advanced::ProtocolError { consumed, .. } => assert_eq!(consumed, 0),
            other => panic!("expected ProtocolError, got {other:?}"),
        }
    }

    #[test]
    fn oversized_incomplete_packet_trips_message_too_large() {
        let mut connection = Connection::with_limits(Limits { max_message_bytes: 10 });
        // declares a huge remaining length, but only a few payload bytes
        // actually sent — PartialPacket forever unless the guard trips.
        connection.feed_bytes(&[0x30, 0xFF, 0xFF, 0xFF, 0x7F]);
        connection.feed_bytes(&[0_u8; 20]);
        assert!(matches!(connection.advance(), Advanced::MessageTooLarge));
    }

    #[test]
    fn small_incomplete_packet_stays_need_more_under_the_cap() {
        let mut connection = Connection::with_limits(Limits { max_message_bytes: 16 });
        connection.feed_bytes(&[0x30, 0x05, 0x00, 0x02, b'a']);
        assert!(matches!(connection.advance(), Advanced::NeedMore));
    }

    #[test]
    fn consume_advances_cursor_and_pipelined_bytes_reparse_in_place() {
        let mut connection = Connection::new();
        let first = pingreq();
        let second = publish_qos0("a/b", b"hi");
        connection.feed_bytes(&first);
        connection.feed_bytes(&second);

        let Advanced::Command { consumed, .. } = connection.advance() else {
            panic!("expected first Command");
        };
        connection.consume(consumed);

        match connection.advance() {
            Advanced::Command { packet, .. } => {
                assert!(matches!(packet, Packet::Publish { topic: b"a/b", .. }));
            }
            other => panic!("expected second Command, got {other:?}"),
        }
    }

    #[test]
    fn consume_clears_buffer_once_fully_drained() {
        let mut connection = Connection::new();
        let wire = pingreq();
        connection.feed_bytes(&wire);
        let Advanced::Command { consumed, .. } = connection.advance() else {
            panic!("expected Command");
        };
        connection.consume(consumed);
        assert!(matches!(connection.advance(), Advanced::NeedMore));
    }
}
