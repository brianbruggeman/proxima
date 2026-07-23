//! MQTT v3.1.1 wire-format parser (sans-IO).
//!
//! Tracked as P2 in `docs/protocol-gap/discipline.md`. MQTT packets
//! are: 1-byte type+flags + variable-length "remaining length" field
//! (1-4 bytes, custom encoding) + variable header + payload.
//!
//! This primitive parses the **fixed header** and the
//! **routing-critical variable header** for each packet type
//! (topic + packet_id for PUBLISH, client_id for CONNECT, etc.).
//! The deep RFC-mandated payload structure for less common packets
//! is exposed as a borrowed `&[u8]` for the caller to deal with.
//!
//! MQTT v5 adds property tables and authentication exchanges; it's
//! a sibling primitive, not in scope here.
//!
//! Reference crates: `rumqttd`, `ntex-mqtt`. Both are full broker
//! impls — the substrate parity baseline is a scope-matched
//! hand-rolled parser in the bench harness.
//!
//! Sub-flag: `mqtt-listener` (default off).
//!
//! [`connection`] wraps [`parse_packet`] in a sans-IO state machine
//! ([`connection::Connection`] — `feed_bytes`/`advance`/`consume`, DoS-capped
//! by [`connection::Limits`]) mirroring [`crate::redis::connection::Connection`]'s
//! shape. [`encode`] builds the server-to-client and client-to-server wire
//! packets [`parse_packet`] cannot itself produce. [`pipe_contract`] maps a
//! packet onto a `proxima_primitives::pipe::Pipe` request the same way
//! [`crate::redis::pipe_contract`] does for RESP — the std client/listener
//! facade (`proxima-mqtt`) builds on both.

pub mod connection;
pub mod encode;
pub mod pipe_contract;

pub use connection::{Advanced, Connection, Limits};
pub use pipe_contract::{MqttReply, MqttRequest, is_streaming, verb};

/// MQTT control packet types (low nibble of the first byte after
/// shifting right by 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PacketType {
    Connect = 1,
    ConnAck = 2,
    Publish = 3,
    PubAck = 4,
    PubRec = 5,
    PubRel = 6,
    PubComp = 7,
    Subscribe = 8,
    SubAck = 9,
    Unsubscribe = 10,
    UnsubAck = 11,
    PingReq = 12,
    PingResp = 13,
    Disconnect = 14,
}

/// PUBLISH-specific flags decoded from the first byte's low nibble.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PublishFlags {
    /// QoS level: 0, 1, or 2.
    pub qos: u8,
    /// True if the packet is a re-delivery (DUP).
    pub dup: bool,
    /// True if the broker should retain this message on the topic.
    pub retain: bool,
}

/// Decoded MQTT packet. Borrows variable header / payload slices
/// from the source buffer.
#[derive(Debug, Clone)]
pub enum Packet<'a> {
    /// CONNECT packet — extracted enough for routing decisions.
    Connect {
        /// Protocol name (e.g. "MQTT").
        protocol_name: &'a [u8],
        /// Protocol level (4 for v3.1.1, 5 for v5).
        protocol_level: u8,
        /// Connect flags byte.
        connect_flags: u8,
        /// Keep-alive interval in seconds.
        keep_alive: u16,
        /// Client identifier.
        client_id: &'a [u8],
        /// Remaining payload after client_id (will/username/password —
        /// caller decodes if needed).
        rest: &'a [u8],
    },
    /// PUBLISH packet — the routing-critical case.
    Publish {
        flags: PublishFlags,
        topic: &'a [u8],
        /// Only present when QoS > 0.
        packet_id: Option<u16>,
        payload: &'a [u8],
    },
    /// SUBSCRIBE packet — packet_id + topic filters payload (caller
    /// walks for individual filters if needed).
    Subscribe {
        packet_id: u16,
        topic_filters: &'a [u8],
    },
    /// UNSUBSCRIBE packet.
    Unsubscribe {
        packet_id: u16,
        topic_filters: &'a [u8],
    },
    /// Ack variants share the same 2-byte payload (the packet ID).
    Ack {
        packet_type: PacketType,
        packet_id: u16,
    },
    /// Heartbeats — no payload.
    PingReq,
    PingResp,
    /// DISCONNECT — no payload in v3.1.1.
    Disconnect,
    /// CONNACK — session-present flag + return code.
    ConnAck {
        session_present: bool,
        return_code: u8,
    },
    /// SUBACK — packet_id + return-code list (length-derived from rem len).
    SubAck {
        packet_id: u16,
        return_codes: &'a [u8],
    },
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("buffer ended mid-packet")]
    Short,
    #[error("remaining-length varint exceeds 4 bytes")]
    RemainingLengthOverflow,
    #[error("packet type {0} is reserved or invalid")]
    InvalidPacketType(u8),
    #[error("declared remaining length {0} exceeds buffer")]
    PartialPacket(u32),
    #[error("malformed packet: {0}")]
    Malformed(&'static str),
}

/// Parse one full MQTT packet starting at `buf[0]`. Returns the
/// decoded packet plus total bytes consumed (fixed header + variable
/// header + payload).
#[inline]
pub fn parse_packet(buf: &[u8]) -> Result<(Packet<'_>, usize), ParseError> {
    if buf.is_empty() {
        return Err(ParseError::Short);
    }
    let first = buf[0];
    let packet_type_bits = first >> 4;
    let flags = first & 0x0F;

    let (remaining_length, rem_len_bytes) = decode_remaining_length(&buf[1..])?;
    let header_len = 1 + rem_len_bytes;
    let total_len = header_len + remaining_length as usize;
    if buf.len() < total_len {
        return Err(ParseError::PartialPacket(remaining_length));
    }
    let body = &buf[header_len..total_len];

    let packet = match packet_type_bits {
        1 => parse_connect(body)?,
        2 => parse_connack(body)?,
        3 => parse_publish(flags, body)?,
        4 => Packet::Ack {
            packet_type: PacketType::PubAck,
            packet_id: parse_packet_id(body)?,
        },
        5 => Packet::Ack {
            packet_type: PacketType::PubRec,
            packet_id: parse_packet_id(body)?,
        },
        6 => Packet::Ack {
            packet_type: PacketType::PubRel,
            packet_id: parse_packet_id(body)?,
        },
        7 => Packet::Ack {
            packet_type: PacketType::PubComp,
            packet_id: parse_packet_id(body)?,
        },
        8 => parse_subscribe(body)?,
        9 => parse_suback(body)?,
        10 => parse_unsubscribe(body)?,
        11 => Packet::Ack {
            packet_type: PacketType::UnsubAck,
            packet_id: parse_packet_id(body)?,
        },
        12 => Packet::PingReq,
        13 => Packet::PingResp,
        14 => Packet::Disconnect,
        other => return Err(ParseError::InvalidPacketType(other)),
    };
    Ok((packet, total_len))
}

/// Decode the MQTT "remaining length" varint. Up to 4 bytes, low 7
/// bits of each byte are data, MSB is continuation. Hard error if
/// the 4th byte still has its continuation bit set.
#[inline(always)]
pub fn decode_remaining_length(buf: &[u8]) -> Result<(u32, usize), ParseError> {
    let mut value: u32 = 0;
    let mut multiplier: u32 = 1;
    for (idx, &byte) in buf.iter().take(4).enumerate() {
        value += u32::from(byte & 0x7F) * multiplier;
        if byte & 0x80 == 0 {
            return Ok((value, idx + 1));
        }
        multiplier *= 128;
    }
    if buf.len() < 4 {
        Err(ParseError::Short)
    } else {
        Err(ParseError::RemainingLengthOverflow)
    }
}

#[inline(always)]
fn read_u16(buf: &[u8]) -> Result<(u16, &[u8]), ParseError> {
    if buf.len() < 2 {
        return Err(ParseError::Short);
    }
    let value = u16::from_be_bytes([buf[0], buf[1]]);
    Ok((value, &buf[2..]))
}

/// Reads one length-prefixed MQTT UTF-8 string field (`u16` BE length +
/// bytes). Exposed beyond this module because the CONNECT payload's
/// remainder (`Packet::Connect::rest` — Will Topic/Message, username,
/// password) is left for the caller to walk per the connect-flags bits,
/// the same "less common payload structure is a raw slice" reasoning the
/// module doc describes.
#[inline(always)]
pub fn read_string(buf: &[u8]) -> Result<(&[u8], &[u8]), ParseError> {
    let (len, rest) = read_u16(buf)?;
    let len = len as usize;
    if rest.len() < len {
        return Err(ParseError::Short);
    }
    Ok((&rest[..len], &rest[len..]))
}

#[inline]
fn parse_packet_id(body: &[u8]) -> Result<u16, ParseError> {
    let (packet_id, _) = read_u16(body)?;
    Ok(packet_id)
}

#[inline(always)]
fn parse_connect(body: &[u8]) -> Result<Packet<'_>, ParseError> {
    let (protocol_name, rest) = read_string(body)?;
    if rest.len() < 4 {
        return Err(ParseError::Short);
    }
    let protocol_level = rest[0];
    let connect_flags = rest[1];
    let keep_alive = u16::from_be_bytes([rest[2], rest[3]]);
    let rest = &rest[4..];
    let (client_id, rest) = read_string(rest)?;
    Ok(Packet::Connect {
        protocol_name,
        protocol_level,
        connect_flags,
        keep_alive,
        client_id,
        rest,
    })
}

#[inline]
fn parse_connack(body: &[u8]) -> Result<Packet<'_>, ParseError> {
    if body.len() < 2 {
        return Err(ParseError::Short);
    }
    Ok(Packet::ConnAck {
        session_present: body[0] & 0x01 != 0,
        return_code: body[1],
    })
}

#[inline(always)]
fn parse_publish(flags: u8, body: &[u8]) -> Result<Packet<'_>, ParseError> {
    let qos = (flags >> 1) & 0x03;
    let dup = flags & 0x08 != 0;
    let retain = flags & 0x01 != 0;
    if qos > 2 {
        return Err(ParseError::Malformed("publish qos > 2"));
    }
    let (topic, rest) = read_string(body)?;
    let (packet_id, payload) = if qos > 0 {
        let (id, after) = read_u16(rest)?;
        (Some(id), after)
    } else {
        (None, rest)
    };
    Ok(Packet::Publish {
        flags: PublishFlags { qos, dup, retain },
        topic,
        packet_id,
        payload,
    })
}

#[inline]
fn parse_subscribe(body: &[u8]) -> Result<Packet<'_>, ParseError> {
    let (packet_id, rest) = read_u16(body)?;
    Ok(Packet::Subscribe {
        packet_id,
        topic_filters: rest,
    })
}

#[inline]
fn parse_unsubscribe(body: &[u8]) -> Result<Packet<'_>, ParseError> {
    let (packet_id, rest) = read_u16(body)?;
    Ok(Packet::Unsubscribe {
        packet_id,
        topic_filters: rest,
    })
}

#[inline]
fn parse_suback(body: &[u8]) -> Result<Packet<'_>, ParseError> {
    let (packet_id, rest) = read_u16(body)?;
    Ok(Packet::SubAck {
        packet_id,
        return_codes: rest,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn remaining_length_single_byte() {
        let (value, used) = decode_remaining_length(&[0]).unwrap();
        assert_eq!((value, used), (0, 1));
        let (value, used) = decode_remaining_length(&[127]).unwrap();
        assert_eq!((value, used), (127, 1));
    }

    #[test]
    fn remaining_length_multi_byte() {
        // 128 = 0x80, 0x01
        let (value, used) = decode_remaining_length(&[0x80, 0x01]).unwrap();
        assert_eq!((value, used), (128, 2));
        // 16384 = 0x80, 0x80, 0x01
        let (value, used) = decode_remaining_length(&[0x80, 0x80, 0x01]).unwrap();
        assert_eq!((value, used), (16_384, 3));
        // 2097152 = 0x80, 0x80, 0x80, 0x01
        let (value, used) = decode_remaining_length(&[0x80, 0x80, 0x80, 0x01]).unwrap();
        assert_eq!((value, used), (2_097_152, 4));
    }

    #[test]
    fn remaining_length_overflow() {
        // 4 bytes all with continuation set ⇒ overflow.
        let buf = [0x80, 0x80, 0x80, 0x80];
        match decode_remaining_length(&buf) {
            Err(ParseError::RemainingLengthOverflow) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parses_publish_qos0() {
        // PUBLISH, qos 0, topic "a/b", payload "hello"
        let mut buf = vec![0x30]; // type=3, flags=0
        let body_len = 2 + 3 + 5; // topic_len(2) + topic(3) + payload(5)
        buf.push(body_len as u8);
        buf.extend_from_slice(&[0, 3]); // topic length
        buf.extend_from_slice(b"a/b");
        buf.extend_from_slice(b"hello");
        let (pkt, used) = parse_packet(&buf).unwrap();
        assert_eq!(used, buf.len());
        match pkt {
            Packet::Publish {
                flags,
                topic,
                packet_id,
                payload,
            } => {
                assert_eq!(flags.qos, 0);
                assert!(!flags.dup);
                assert!(!flags.retain);
                assert_eq!(topic, b"a/b");
                assert!(packet_id.is_none());
                assert_eq!(payload, b"hello");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parses_publish_qos1() {
        // PUBLISH, qos 1 (flags = 0x02), topic "t", packet_id 7, payload "x"
        let mut buf = vec![0x32];
        let body_len = 2 + 1 + 2 + 1;
        buf.push(body_len as u8);
        buf.extend_from_slice(&[0, 1]);
        buf.extend_from_slice(b"t");
        buf.extend_from_slice(&[0, 7]);
        buf.extend_from_slice(b"x");
        let (pkt, _) = parse_packet(&buf).unwrap();
        match pkt {
            Packet::Publish {
                flags,
                topic,
                packet_id,
                payload,
            } => {
                assert_eq!(flags.qos, 1);
                assert_eq!(topic, b"t");
                assert_eq!(packet_id, Some(7));
                assert_eq!(payload, b"x");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parses_connect() {
        let mut buf = vec![0x10];
        // body = protocol_name "MQTT" + lvl 4 + flags 2 + ka 60 + client_id "c1"
        let body_len = 2 + 4 + 1 + 1 + 2 + 2 + 2;
        buf.push(body_len as u8);
        buf.extend_from_slice(&[0, 4]);
        buf.extend_from_slice(b"MQTT");
        buf.push(4);
        buf.push(0x02);
        buf.extend_from_slice(&[0, 60]);
        buf.extend_from_slice(&[0, 2]);
        buf.extend_from_slice(b"c1");
        let (pkt, _) = parse_packet(&buf).unwrap();
        match pkt {
            Packet::Connect {
                protocol_name,
                protocol_level,
                connect_flags,
                keep_alive,
                client_id,
                ..
            } => {
                assert_eq!(protocol_name, b"MQTT");
                assert_eq!(protocol_level, 4);
                assert_eq!(connect_flags, 0x02);
                assert_eq!(keep_alive, 60);
                assert_eq!(client_id, b"c1");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parses_pingreq() {
        let buf = [0xC0, 0x00];
        let (pkt, used) = parse_packet(&buf).unwrap();
        assert_eq!(used, 2);
        assert!(matches!(pkt, Packet::PingReq));
    }

    #[test]
    fn parses_disconnect() {
        let buf = [0xE0, 0x00];
        let (pkt, _) = parse_packet(&buf).unwrap();
        assert!(matches!(pkt, Packet::Disconnect));
    }

    #[test]
    fn parses_puback() {
        let buf = [0x40, 0x02, 0x00, 0x05];
        let (pkt, _) = parse_packet(&buf).unwrap();
        match pkt {
            Packet::Ack {
                packet_type,
                packet_id,
            } => {
                assert_eq!(packet_type, PacketType::PubAck);
                assert_eq!(packet_id, 5);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn short_returns_short() {
        let buf = [0x30]; // PUBLISH header without rem len
        match parse_packet(&buf) {
            Err(ParseError::Short) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn partial_packet_returns_partial() {
        let buf = [0x30, 10]; // declares 10 byte body, none supplied
        match parse_packet(&buf) {
            Err(ParseError::PartialPacket(10)) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn invalid_packet_type_rejected() {
        let buf = [0x00, 0x00]; // type 0 is reserved
        match parse_packet(&buf) {
            Err(ParseError::InvalidPacketType(0)) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }
}
