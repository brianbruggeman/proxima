//! AMQP 0-9-1 frame parser (sans-IO).
//!
//! Tracked as P3 in `docs/protocol-gap/discipline.md`. AMQP 0-9-1
//! is RabbitMQ's native wire protocol. Every frame on the wire is:
//!
//! ```text
//! +--------+--------+----------+-----------+--------+
//! |  type  |   channel   |  payload size  | payload |  0xCE  |
//! | 1 byte |   2 bytes   |    4 bytes     | N bytes | 1 byte |
//! +--------+-------------+----------------+---------+--------+
//! ```
//!
//! Frame types:
//! - 1 = METHOD (class_id u16 + method_id u16 + method args)
//! - 2 = HEADER (class_id u16 + weight u16 + body_size u64 + props)
//! - 3 = BODY  (raw bytes)
//! - 8 = HEARTBEAT (zero payload)
//!
//! Reference crate: `lapin` (client-side). The parity baseline lives
//! inline in the bench harness with the same scope.
//!
//! Sub-flag: `amqp-listener` (default off).


/// AMQP frame end-marker byte (final byte of every frame).
pub const FRAME_END: u8 = 0xCE;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameType {
    Method = 1,
    Header = 2,
    Body = 3,
    Heartbeat = 8,
}

/// One parsed AMQP frame. The payload is borrowed from the source
/// buffer; the caller decodes the inner method args / properties /
/// body bytes per type.
#[derive(Debug, Clone)]
pub enum Frame<'a> {
    Method {
        channel: u16,
        class_id: u16,
        method_id: u16,
        /// Method arguments — the remainder of the payload after
        /// class_id + method_id. Caller decodes per AMQP spec
        /// (depends on the specific method).
        args: &'a [u8],
    },
    Header {
        channel: u16,
        class_id: u16,
        weight: u16,
        body_size: u64,
        /// Property flags + values. Caller decodes per the property
        /// list spec.
        properties: &'a [u8],
    },
    Body {
        channel: u16,
        payload: &'a [u8],
    },
    Heartbeat {
        channel: u16,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("buffer ended mid-frame")]
    Short,
    #[error("frame type {0} is reserved or invalid")]
    InvalidFrameType(u8),
    #[error("declared payload size {0} exceeds buffer")]
    PartialFrame(u32),
    #[error("frame-end marker missing or wrong (got 0x{0:02X}, expected 0xCE)")]
    BadFrameEnd(u8),
    #[error("method frame too short for class_id + method_id")]
    MalformedMethod,
    #[error("header frame too short for class_id + weight + body_size")]
    MalformedHeader,
}

/// Parse one full AMQP frame starting at `buf[0]`. Returns the frame
/// (with borrowed slices into `buf`) and the total bytes consumed
/// (1 + 2 + 4 + payload + 1 for the 0xCE marker).
#[inline(always)]
pub fn parse_frame(buf: &[u8]) -> Result<(Frame<'_>, usize), ParseError> {
    if buf.len() < 7 {
        return Err(ParseError::Short);
    }
    let frame_type = buf[0];
    let channel = u16::from_be_bytes([buf[1], buf[2]]);
    let size = u32::from_be_bytes([buf[3], buf[4], buf[5], buf[6]]);
    let payload_start = 7;
    let payload_end = payload_start + size as usize;
    let total = payload_end + 1; // +1 for the 0xCE byte
    if buf.len() < total {
        return Err(ParseError::PartialFrame(size));
    }
    if buf[payload_end] != FRAME_END {
        return Err(ParseError::BadFrameEnd(buf[payload_end]));
    }
    let payload = &buf[payload_start..payload_end];

    let frame = match frame_type {
        1 => {
            // method frame: class_id + method_id + args
            if payload.len() < 4 {
                return Err(ParseError::MalformedMethod);
            }
            Frame::Method {
                channel,
                class_id: u16::from_be_bytes([payload[0], payload[1]]),
                method_id: u16::from_be_bytes([payload[2], payload[3]]),
                args: &payload[4..],
            }
        }
        2 => {
            // header frame: class_id + weight + body_size + properties
            if payload.len() < 12 {
                return Err(ParseError::MalformedHeader);
            }
            Frame::Header {
                channel,
                class_id: u16::from_be_bytes([payload[0], payload[1]]),
                weight: u16::from_be_bytes([payload[2], payload[3]]),
                body_size: u64::from_be_bytes([
                    payload[4],
                    payload[5],
                    payload[6],
                    payload[7],
                    payload[8],
                    payload[9],
                    payload[10],
                    payload[11],
                ]),
                properties: &payload[12..],
            }
        }
        3 => Frame::Body { channel, payload },
        8 => Frame::Heartbeat { channel },
        other => return Err(ParseError::InvalidFrameType(other)),
    };
    Ok((frame, total))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn make_frame(frame_type: u8, channel: u16, payload: &[u8]) -> Vec<u8> {
        let mut buf = vec![frame_type];
        buf.extend_from_slice(&channel.to_be_bytes());
        buf.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        buf.extend_from_slice(payload);
        buf.push(FRAME_END);
        buf
    }

    #[test]
    fn parses_method_frame() {
        // class_id=60 (basic), method_id=40 (publish), args 4 zero bytes
        let mut payload = Vec::new();
        payload.extend_from_slice(&60u16.to_be_bytes());
        payload.extend_from_slice(&40u16.to_be_bytes());
        payload.extend_from_slice(&[0, 0, 0, 0]);
        let buf = make_frame(1, 1, &payload);
        let (frame, used) = parse_frame(&buf).unwrap();
        assert_eq!(used, buf.len());
        match frame {
            Frame::Method {
                channel,
                class_id,
                method_id,
                args,
            } => {
                assert_eq!(channel, 1);
                assert_eq!(class_id, 60);
                assert_eq!(method_id, 40);
                assert_eq!(args, &[0, 0, 0, 0]);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parses_body_frame() {
        let buf = make_frame(3, 2, b"hello world");
        let (frame, _) = parse_frame(&buf).unwrap();
        match frame {
            Frame::Body { channel, payload } => {
                assert_eq!(channel, 2);
                assert_eq!(payload, b"hello world");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parses_header_frame() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&60u16.to_be_bytes()); // class_id
        payload.extend_from_slice(&0u16.to_be_bytes()); // weight
        payload.extend_from_slice(&1024u64.to_be_bytes()); // body_size
        payload.extend_from_slice(&[0xAB, 0xCD]); // properties placeholder
        let buf = make_frame(2, 3, &payload);
        let (frame, _) = parse_frame(&buf).unwrap();
        match frame {
            Frame::Header {
                class_id,
                weight,
                body_size,
                properties,
                ..
            } => {
                assert_eq!(class_id, 60);
                assert_eq!(weight, 0);
                assert_eq!(body_size, 1024);
                assert_eq!(properties, &[0xAB, 0xCD]);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parses_heartbeat() {
        let buf = make_frame(8, 0, &[]);
        let (frame, _) = parse_frame(&buf).unwrap();
        assert!(matches!(frame, Frame::Heartbeat { channel: 0 }));
    }

    #[test]
    fn short_returns_short() {
        let buf = [1u8, 0, 0]; // partial header
        match parse_frame(&buf) {
            Err(ParseError::Short) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn partial_payload_returns_partial() {
        let buf = [1u8, 0, 0, 0, 0, 0, 10]; // declares 10 byte payload, none supplied
        match parse_frame(&buf) {
            Err(ParseError::PartialFrame(10)) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn missing_frame_end_rejected() {
        let mut buf = vec![3u8, 0, 1, 0, 0, 0, 3];
        buf.extend_from_slice(b"abc");
        buf.push(0xFF); // wrong end marker
        match parse_frame(&buf) {
            Err(ParseError::BadFrameEnd(0xFF)) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn invalid_frame_type_rejected() {
        let mut buf = vec![99u8, 0, 0, 0, 0, 0, 0];
        buf.push(FRAME_END);
        match parse_frame(&buf) {
            Err(ParseError::InvalidFrameType(99)) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }
}
