//! Frame envelope *encode* — the write-side counterpart to
//! `proxima_protocols::amqp::parse_frame`, which only decodes. AMQP 0-9-1
//! wraps every method/header/body/heartbeat payload in the same 7-byte
//! header (`type` octet + `channel` short + `length` long) and trailing
//! [`proxima_protocols::amqp::FRAME_END`] marker; this module is that
//! wrapper plus the three outbound frame shapes
//! [`crate::fsm::Connection`]/[`crate::broker::AmqpBroker`] build.

use proxima_protocols::amqp::{FRAME_END, FrameType};

use crate::method::Method;

/// Appends one framed `(type, channel, payload)` triple to `out`.
pub fn encode_frame(out: &mut Vec<u8>, frame_type: FrameType, channel: u16, payload: &[u8]) {
    out.push(frame_type as u8);
    out.extend_from_slice(&channel.to_be_bytes());
    out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    out.extend_from_slice(payload);
    out.push(FRAME_END);
}

/// Encodes `method` as a `Frame::Method` on `channel`.
pub fn encode_method_frame(out: &mut Vec<u8>, channel: u16, method: &Method) {
    let (class_id, method_id, args) = crate::method::encode(method);
    let mut payload = Vec::with_capacity(4 + args.len());
    payload.extend_from_slice(&class_id.to_be_bytes());
    payload.extend_from_slice(&method_id.to_be_bytes());
    payload.extend_from_slice(&args);
    encode_frame(out, FrameType::Method, channel, &payload);
}

/// Encodes a content-header frame (always `weight = 0`; AMQP 0-9-1 never
/// assigns it a meaning beyond the reserved zero).
pub fn encode_header_frame(
    out: &mut Vec<u8>,
    channel: u16,
    class_id: u16,
    body_size: u64,
    properties: &[u8],
) {
    let mut payload = Vec::with_capacity(12 + properties.len());
    payload.extend_from_slice(&class_id.to_be_bytes());
    payload.extend_from_slice(&0_u16.to_be_bytes());
    payload.extend_from_slice(&body_size.to_be_bytes());
    payload.extend_from_slice(properties);
    encode_frame(out, FrameType::Header, channel, &payload);
}

/// Splits `body` across one or more content-body frames, each capped at
/// `max_chunk` bytes — the negotiated `frame-max` minus the 8-byte frame
/// envelope overhead. Emits nothing for an empty body (a zero-length
/// message has no body frame at all, per spec).
pub fn encode_body_frames(out: &mut Vec<u8>, channel: u16, body: &[u8], max_chunk: usize) {
    if body.is_empty() {
        return;
    }
    for chunk in body.chunks(max_chunk.max(1)) {
        encode_frame(out, FrameType::Body, channel, chunk);
    }
}

pub fn encode_heartbeat_frame(out: &mut Vec<u8>) {
    encode_frame(out, FrameType::Heartbeat, 0, &[]);
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_protocols::amqp::{Frame, parse_frame};

    #[test]
    fn encoded_method_frame_round_trips_through_parse_frame() {
        let mut out = Vec::new();
        encode_method_frame(&mut out, 1, &Method::ChannelOpenOk);
        let (frame, consumed) = parse_frame(&out).expect("parse");
        assert_eq!(consumed, out.len());
        match frame {
            Frame::Method {
                channel,
                class_id,
                method_id,
                ..
            } => {
                assert_eq!(channel, 1);
                assert_eq!(class_id, crate::method::id::CHANNEL);
                assert_eq!(method_id, crate::method::id::CHANNEL_OPEN_OK);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn body_frames_split_at_max_chunk() {
        let mut out = Vec::new();
        encode_body_frames(&mut out, 2, b"hello world", 4);
        let mut cursor = out.as_slice();
        let mut chunks = Vec::new();
        while !cursor.is_empty() {
            let (frame, consumed) = parse_frame(cursor).expect("parse");
            match frame {
                Frame::Body { payload, .. } => chunks.push(payload.to_vec()),
                other => panic!("unexpected: {other:?}"),
            }
            cursor = &cursor[consumed..];
        }
        assert_eq!(
            chunks,
            vec![b"hell".to_vec(), b"o wo".to_vec(), b"rld".to_vec()]
        );
    }

    #[test]
    fn empty_body_emits_no_frame() {
        let mut out = Vec::new();
        encode_body_frames(&mut out, 2, b"", 4);
        assert!(out.is_empty());
    }
}
