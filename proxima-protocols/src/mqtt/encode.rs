//! MQTT wire encoders — the write side [`super::parse_packet`] does not
//! itself provide. Bytes-out only, no I/O: the caller (`proxima-mqtt`'s
//! connection driver / client session) owns the socket.
//!
//! One free function per packet the broker or the client needs to build:
//! server-to-client (`CONNACK`, outbound `PUBLISH`, `PUBACK`/`PUBREC`/
//! `PUBREL`/`PUBCOMP`, `SUBACK`, `UNSUBACK`, `PINGRESP`) and
//! client-to-server (`CONNECT`, outbound `PUBLISH`, `SUBSCRIBE`,
//! `UNSUBSCRIBE`, `PINGREQ`, `DISCONNECT`). Mirrors
//! `crate::redis::encode`/`encode_command`'s bytes-out shape.

use alloc::vec::Vec;

use super::PacketType;

/// Encode the MQTT "remaining length" varint (the inverse of
/// [`super::decode_remaining_length`]): up to 4 bytes, low 7 bits of each
/// byte are data, MSB is continuation.
pub fn encode_remaining_length(mut value: u32, out: &mut Vec<u8>) {
    loop {
        let mut byte = (value % 128) as u8;
        value /= 128;
        if value > 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}

fn push_string(bytes: &[u8], out: &mut Vec<u8>) {
    out.extend_from_slice(&(bytes.len() as u16).to_be_bytes());
    out.extend_from_slice(bytes);
}

fn push_fixed_header(packet_type: PacketType, flags: u8, remaining_length: usize, out: &mut Vec<u8>) {
    out.push(((packet_type as u8) << 4) | flags);
    encode_remaining_length(remaining_length as u32, out);
}

/// `CONNECT` — the client's session-open packet. `clean_session` maps to
/// connect-flag bit 1; `keep_alive` is seconds. `username`/`password` are
/// omitted from the connect-flags byte when both are empty.
#[allow(clippy::too_many_arguments)]
pub fn encode_connect(
    client_id: &[u8],
    clean_session: bool,
    keep_alive: u16,
    username: Option<&[u8]>,
    password: Option<&[u8]>,
    out: &mut Vec<u8>,
) {
    let mut body = Vec::new();
    push_string(b"MQTT", &mut body);
    body.push(4); // protocol level 4 == v3.1.1
    let mut flags = 0_u8;
    if clean_session {
        flags |= 0x02;
    }
    if username.is_some() {
        flags |= 0x80;
    }
    if password.is_some() {
        flags |= 0x40;
    }
    body.push(flags);
    body.extend_from_slice(&keep_alive.to_be_bytes());
    push_string(client_id, &mut body);
    if let Some(username) = username {
        push_string(username, &mut body);
    }
    if let Some(password) = password {
        push_string(password, &mut body);
    }
    push_fixed_header(PacketType::Connect, 0, body.len(), out);
    out.extend_from_slice(&body);
}

/// `CONNACK` — the broker's reply to `CONNECT`.
pub fn encode_connack(session_present: bool, return_code: u8, out: &mut Vec<u8>) {
    push_fixed_header(PacketType::ConnAck, 0, 2, out);
    out.push(u8::from(session_present));
    out.push(return_code);
}

/// `PUBLISH`, either direction. `packet_id` must be `Some` for `qos > 0`
/// and `None` for `qos == 0` — the caller (not this encoder) enforces that
/// invariant, matching [`super::parse_publish`]'s own read side.
pub fn encode_publish(
    topic: &[u8],
    packet_id: Option<u16>,
    payload: &[u8],
    qos: u8,
    dup: bool,
    retain: bool,
    out: &mut Vec<u8>,
) {
    let mut flags = (qos & 0x03) << 1;
    if dup {
        flags |= 0x08;
    }
    if retain {
        flags |= 0x01;
    }
    let mut remaining = 2 + topic.len() + payload.len();
    if packet_id.is_some() {
        remaining += 2;
    }
    push_fixed_header(PacketType::Publish, flags, remaining, out);
    push_string(topic, out);
    if let Some(packet_id) = packet_id {
        out.extend_from_slice(&packet_id.to_be_bytes());
    }
    out.extend_from_slice(payload);
}

/// The four 2-byte-payload acknowledgements sharing one shape: `PUBACK`,
/// `PUBREC`, `PUBREL` (flags `0x02` per the spec), `PUBCOMP`, `UNSUBACK`.
pub fn encode_ack(packet_type: PacketType, packet_id: u16, out: &mut Vec<u8>) {
    let flags = u8::from(matches!(packet_type, PacketType::PubRel));
    push_fixed_header(packet_type, flags, 2, out);
    out.extend_from_slice(&packet_id.to_be_bytes());
}

/// `SUBSCRIBE` — the client's subscribe request. `filters` pairs each
/// topic filter with its requested QoS.
pub fn encode_subscribe(packet_id: u16, filters: &[(&[u8], u8)], out: &mut Vec<u8>) {
    let mut body = Vec::new();
    body.extend_from_slice(&packet_id.to_be_bytes());
    for (filter, qos) in filters {
        push_string(filter, &mut body);
        body.push(*qos);
    }
    push_fixed_header(PacketType::Subscribe, 0x02, body.len(), out);
    out.extend_from_slice(&body);
}

/// `SUBACK` — the broker's reply. `return_codes[i]` is the granted QoS for
/// `filters[i]` in the matching `SUBSCRIBE`, or `0x80` to refuse that one
/// filter (real clients treat a `SUBSCRIBE` as per-filter, not atomic).
pub fn encode_suback(packet_id: u16, return_codes: &[u8], out: &mut Vec<u8>) {
    push_fixed_header(PacketType::SubAck, 0, 2 + return_codes.len(), out);
    out.extend_from_slice(&packet_id.to_be_bytes());
    out.extend_from_slice(return_codes);
}

/// `UNSUBSCRIBE` — the client's unsubscribe request.
pub fn encode_unsubscribe(packet_id: u16, filters: &[&[u8]], out: &mut Vec<u8>) {
    let mut body = Vec::new();
    body.extend_from_slice(&packet_id.to_be_bytes());
    for filter in filters {
        push_string(filter, &mut body);
    }
    push_fixed_header(PacketType::Unsubscribe, 0x02, body.len(), out);
    out.extend_from_slice(&body);
}

/// `PINGREQ` — no payload.
pub fn encode_pingreq(out: &mut Vec<u8>) {
    push_fixed_header(PacketType::PingReq, 0, 0, out);
}

/// `PINGRESP` — no payload.
pub fn encode_pingresp(out: &mut Vec<u8>) {
    push_fixed_header(PacketType::PingResp, 0, 0, out);
}

/// `DISCONNECT` — no payload in v3.1.1.
pub fn encode_disconnect(out: &mut Vec<u8>) {
    push_fixed_header(PacketType::Disconnect, 0, 0, out);
}

/// One `(topic_filter, requested_qos)` pair from a `SUBSCRIBE` packet's
/// payload — [`super::Packet::Subscribe`] exposes the raw remainder bytes
/// deliberately (mirrors `Frame::Array` leaving arg extraction to the
/// driver); this walks them.
pub fn iter_subscribe_filters(payload: &[u8]) -> SubscribeFilters<'_> {
    SubscribeFilters { rest: payload }
}

pub struct SubscribeFilters<'a> {
    rest: &'a [u8],
}

impl<'a> Iterator for SubscribeFilters<'a> {
    type Item = (&'a [u8], u8);

    fn next(&mut self) -> Option<Self::Item> {
        if self.rest.len() < 3 {
            return None;
        }
        let len = u16::from_be_bytes([self.rest[0], self.rest[1]]) as usize;
        if self.rest.len() < 2 + len + 1 {
            return None;
        }
        let filter = &self.rest[2..2 + len];
        let qos = self.rest[2 + len];
        self.rest = &self.rest[2 + len + 1..];
        Some((filter, qos))
    }
}

/// One topic filter from an `UNSUBSCRIBE` packet's payload (no trailing
/// QoS byte, unlike [`iter_subscribe_filters`]).
pub fn iter_unsubscribe_filters(payload: &[u8]) -> UnsubscribeFilters<'_> {
    UnsubscribeFilters { rest: payload }
}

pub struct UnsubscribeFilters<'a> {
    rest: &'a [u8],
}

impl<'a> Iterator for UnsubscribeFilters<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.rest.len() < 2 {
            return None;
        }
        let len = u16::from_be_bytes([self.rest[0], self.rest[1]]) as usize;
        if self.rest.len() < 2 + len {
            return None;
        }
        let filter = &self.rest[2..2 + len];
        self.rest = &self.rest[2 + len..];
        Some(filter)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::mqtt::{Packet, parse_packet};
    use alloc::vec;

    #[test]
    fn remaining_length_round_trips_through_decode() {
        for value in [0_u32, 127, 128, 16_384, 2_097_152, 268_435_455] {
            let mut out = Vec::new();
            encode_remaining_length(value, &mut out);
            let (decoded, used) = super::super::decode_remaining_length(&out).unwrap();
            assert_eq!((decoded, used), (value, out.len()));
        }
    }

    #[test]
    fn connack_round_trips_through_parse() {
        let mut out = Vec::new();
        encode_connack(true, 0, &mut out);
        let (packet, used) = parse_packet(&out).unwrap();
        assert_eq!(used, out.len());
        match packet {
            Packet::ConnAck { session_present, return_code } => {
                assert!(session_present);
                assert_eq!(return_code, 0);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn publish_qos1_round_trips_through_parse() {
        let mut out = Vec::new();
        encode_publish(b"a/b", Some(7), b"hello", 1, false, true, &mut out);
        let (packet, used) = parse_packet(&out).unwrap();
        assert_eq!(used, out.len());
        match packet {
            Packet::Publish { flags, topic, packet_id, payload } => {
                assert_eq!(flags.qos, 1);
                assert!(flags.retain);
                assert!(!flags.dup);
                assert_eq!(topic, b"a/b");
                assert_eq!(packet_id, Some(7));
                assert_eq!(payload, b"hello");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn puback_round_trips_through_parse() {
        let mut out = Vec::new();
        encode_ack(PacketType::PubAck, 42, &mut out);
        let (packet, _) = parse_packet(&out).unwrap();
        match packet {
            Packet::Ack { packet_type, packet_id } => {
                assert_eq!(packet_type, PacketType::PubAck);
                assert_eq!(packet_id, 42);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn suback_round_trips_through_parse() {
        let mut out = Vec::new();
        encode_suback(9, &[0, 1, 0x80], &mut out);
        let (packet, _) = parse_packet(&out).unwrap();
        match packet {
            Packet::SubAck { packet_id, return_codes } => {
                assert_eq!(packet_id, 9);
                assert_eq!(return_codes, &[0, 1, 0x80]);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn pingresp_round_trips_through_parse() {
        let mut out = Vec::new();
        encode_pingresp(&mut out);
        let (packet, used) = parse_packet(&out).unwrap();
        assert_eq!(used, 2);
        assert!(matches!(packet, Packet::PingResp));
    }

    #[test]
    fn connect_round_trips_through_parse() {
        let mut out = Vec::new();
        encode_connect(b"client-1", true, 30, Some(b"alice"), Some(b"hunter2"), &mut out);
        let (packet, used) = parse_packet(&out).unwrap();
        assert_eq!(used, out.len());
        match packet {
            Packet::Connect { protocol_name, protocol_level, client_id, keep_alive, .. } => {
                assert_eq!(protocol_name, b"MQTT");
                assert_eq!(protocol_level, 4);
                assert_eq!(client_id, b"client-1");
                assert_eq!(keep_alive, 30);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn subscribe_filters_walk_topic_and_requested_qos_pairs() {
        let mut out = Vec::new();
        encode_subscribe(3, &[(b"a/+", 0), (b"b/#", 1)], &mut out);
        let (packet, _) = parse_packet(&out).unwrap();
        let Packet::Subscribe { packet_id, topic_filters } = packet else {
            panic!("expected Subscribe");
        };
        assert_eq!(packet_id, 3);
        let filters: Vec<(&[u8], u8)> = iter_subscribe_filters(topic_filters).collect();
        assert_eq!(filters, vec![(b"a/+".as_slice(), 0), (b"b/#".as_slice(), 1)]);
    }

    #[test]
    fn unsubscribe_filters_walk_bare_topic_strings() {
        let mut out = Vec::new();
        encode_unsubscribe(4, &[b"a/+", b"b/#"], &mut out);
        let (packet, _) = parse_packet(&out).unwrap();
        let Packet::Unsubscribe { packet_id, topic_filters } = packet else {
            panic!("expected Unsubscribe");
        };
        assert_eq!(packet_id, 4);
        let filters: Vec<&[u8]> = iter_unsubscribe_filters(topic_filters).collect();
        assert_eq!(filters, vec![b"a/+".as_slice(), b"b/#".as_slice()]);
    }
}
