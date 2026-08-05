//! Frame envelope *encode* — the write-side counterpart to
//! `proxima_protocols::amqp::parse_frame`, which only decodes. AMQP 0-9-1
//! wraps every method/header/body/heartbeat payload in the same 7-byte
//! header (`type` octet + `channel` short + `length` long) and trailing
//! [`proxima_protocols::amqp::FRAME_END`] marker; this module is that
//! wrapper plus the three outbound frame shapes
//! [`crate::fsm::Connection`]/[`crate::broker::AmqpBroker`] build.

use proxima_protocols::amqp::{FRAME_END, FrameType};

use crate::method::Method;

/// What one frame costs on the wire beyond its payload: the 7-byte header
/// (`type` octet + `channel` short + `length` long) plus the
/// [`FRAME_END`] marker. AMQP 0-9-1's negotiated `frame-max` counts both
/// (§connection.tune), so a body chunk may only be `frame-max` minus this.
pub const FRAME_ENVELOPE_BYTES: usize = 8;

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

/// Splits `body` across content-body frames no larger than the negotiated
/// `frame_max` — envelope included, so the payload cap is `frame_max` minus
/// [`FRAME_ENVELOPE_BYTES`]. Subtracting here rather than at each call site
/// is deliberate: this function emits the envelope, so it is the only place
/// that can get the arithmetic right, and a caller that passed the raw
/// `frame-max` used to overshoot the cap it had just advertised. Emits
/// nothing for an empty body (a zero-length message has no body frame at
/// all, per spec).
pub fn encode_body_frames(out: &mut Vec<u8>, channel: u16, body: &[u8], frame_max: usize) {
    if body.is_empty() {
        return;
    }
    let payload_max = frame_max.saturating_sub(FRAME_ENVELOPE_BYTES).max(1);
    for chunk in body.chunks(payload_max) {
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

    fn body_frame_payloads(wire: &[u8]) -> Vec<Vec<u8>> {
        let mut cursor = wire;
        let mut chunks = Vec::new();
        while !cursor.is_empty() {
            let (frame, consumed) = parse_frame(cursor).expect("parse");
            match frame {
                Frame::Body { payload, .. } => chunks.push(payload.to_vec()),
                other => panic!("unexpected: {other:?}"),
            }
            cursor = &cursor[consumed..];
        }
        chunks
    }

    #[test]
    fn body_frames_split_at_the_frame_max_minus_its_envelope() {
        let mut out = Vec::new();
        encode_body_frames(&mut out, 2, b"hello world", 4 + FRAME_ENVELOPE_BYTES);
        assert_eq!(
            body_frame_payloads(&out),
            vec![b"hell".to_vec(), b"o wo".to_vec(), b"rld".to_vec()]
        );
    }

    // the interop bug this guards: a body chunk sized at the raw `frame-max`
    // puts `frame-max + 8` bytes on the wire, which a conforming peer (and
    // this crate's own `Connection`) rejects as a frame error.
    #[test]
    fn no_emitted_frame_exceeds_the_negotiated_frame_max() {
        let frame_max = 64;
        let mut out = Vec::new();
        encode_body_frames(&mut out, 2, &vec![b'x'; 512], frame_max);

        let mut cursor = out.as_slice();
        while !cursor.is_empty() {
            let (_frame, consumed) = parse_frame(cursor).expect("parse");
            assert!(
                consumed <= frame_max,
                "emitted a {consumed}-byte frame past the {frame_max}-byte frame-max"
            );
            cursor = &cursor[consumed..];
        }
    }

    #[test]
    fn empty_body_emits_no_frame() {
        let mut out = Vec::new();
        encode_body_frames(&mut out, 2, b"", 4);
        assert!(out.is_empty());
    }
}
