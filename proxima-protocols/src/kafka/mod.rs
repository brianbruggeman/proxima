//! Kafka wire-format parser (sans-IO).
//!
//! Tracked as P4 in `docs/protocol-gap/discipline.md`. Kafka's
//! wire protocol frames every request/response as:
//!
//! ```text
//! +--------------------+-----------------+
//! |  size (4 bytes BE) |  payload (size) |
//! +--------------------+-----------------+
//! ```
//!
//! The payload of a request starts with a header — version 0 looks like:
//!
//! ```text
//! +----------+--------------+-----------------+----------------+
//! | api_key  | api_version  | correlation_id  |   client_id    |
//! | i16 BE   |   i16 BE     |     i32 BE      |  nullable str  |
//! +----------+--------------+-----------------+----------------+
//! ```
//!
//! Nullable strings: 2-byte BE length prefix, where -1 means null,
//! otherwise N bytes follow.
//!
//! Header v1 adds nothing for request-side besides the v0 fields
//! (clients still send v0 most of the time). Header v2 (flexible)
//! uses varint lengths + tagged fields — out of scope here, follow-up.
//!
//! Sub-flag: `kafka-listener` (default off).

#[cfg(feature = "kafka-codec-trait")]
pub mod codec_trait;
#[cfg(feature = "kafka-codec-trait")]
pub use codec_trait::{FrameError as KafkaFrameError, KafkaFrameCodec};

/// One parsed Kafka request header. Borrows `client_id` from the
/// source buffer.
#[derive(Debug, Clone, Copy)]
pub struct RequestHeader<'a> {
    pub api_key: i16,
    pub api_version: i16,
    pub correlation_id: i32,
    /// `None` if the wire encoding sent a null string (-1 length).
    pub client_id: Option<&'a [u8]>,
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("buffer ended before full length-prefixed frame")]
    Short,
    #[error("declared frame size {0} is negative or absurdly large")]
    InvalidSize(i32),
    #[error("declared frame size {0} exceeds buffer")]
    PartialFrame(u32),
    #[error("nullable string length {0} is invalid (must be -1 or >= 0)")]
    InvalidStringLength(i16),
}

/// Peek the 4-byte length prefix without consuming it. Useful for
/// reader-side flow control before allocating.
#[inline(always)]
pub fn peek_frame_size(buf: &[u8]) -> Result<u32, ParseError> {
    if buf.len() < 4 {
        return Err(ParseError::Short);
    }
    let size = i32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    if size < 0 {
        return Err(ParseError::InvalidSize(size));
    }
    Ok(size as u32)
}

/// Parse a length-prefixed Kafka frame. Returns the payload slice
/// (without the 4-byte size prefix) and the total bytes consumed
/// (4 + size).
#[inline(always)]
pub fn parse_frame(buf: &[u8]) -> Result<(&[u8], usize), ParseError> {
    let size = peek_frame_size(buf)?;
    let total = 4 + size as usize;
    if buf.len() < total {
        return Err(ParseError::PartialFrame(size));
    }
    Ok((&buf[4..total], total))
}

/// Parse a Kafka request header (v0/v1 — same layout) from the
/// start of `payload`. Returns the parsed header and the offset
/// at which the request body begins.
#[inline(always)]
pub fn parse_request_header(payload: &[u8]) -> Result<(RequestHeader<'_>, usize), ParseError> {
    if payload.len() < 8 {
        return Err(ParseError::Short);
    }
    let api_key = i16::from_be_bytes([payload[0], payload[1]]);
    let api_version = i16::from_be_bytes([payload[2], payload[3]]);
    let correlation_id = i32::from_be_bytes([payload[4], payload[5], payload[6], payload[7]]);
    let (client_id, body_offset) = read_nullable_string(&payload[8..])?;
    Ok((
        RequestHeader {
            api_key,
            api_version,
            correlation_id,
            client_id,
        },
        8 + body_offset,
    ))
}

/// Read a Kafka nullable string: 2-byte BE length prefix where -1
/// means null, otherwise N bytes follow.
#[inline(always)]
fn read_nullable_string(buf: &[u8]) -> Result<(Option<&[u8]>, usize), ParseError> {
    if buf.len() < 2 {
        return Err(ParseError::Short);
    }
    let len = i16::from_be_bytes([buf[0], buf[1]]);
    if len == -1 {
        return Ok((None, 2));
    }
    if len < -1 {
        return Err(ParseError::InvalidStringLength(len));
    }
    let len = len as usize;
    let end = 2 + len;
    if buf.len() < end {
        return Err(ParseError::Short);
    }
    Ok((Some(&buf[2..end]), end))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn make_request(
        api_key: i16,
        api_version: i16,
        correlation_id: i32,
        client_id: Option<&[u8]>,
        body: &[u8],
    ) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&api_key.to_be_bytes());
        payload.extend_from_slice(&api_version.to_be_bytes());
        payload.extend_from_slice(&correlation_id.to_be_bytes());
        match client_id {
            Some(s) => {
                payload.extend_from_slice(&(s.len() as i16).to_be_bytes());
                payload.extend_from_slice(s);
            }
            None => payload.extend_from_slice(&(-1i16).to_be_bytes()),
        }
        payload.extend_from_slice(body);

        let mut frame = Vec::new();
        frame.extend_from_slice(&(payload.len() as i32).to_be_bytes());
        frame.extend_from_slice(&payload);
        frame
    }

    #[test]
    fn parses_full_frame() {
        let frame = make_request(0, 11, 42, Some(b"client-1"), b"body");
        let (payload, used) = parse_frame(&frame).unwrap();
        assert_eq!(used, frame.len());
        let (header, body_offset) = parse_request_header(payload).unwrap();
        assert_eq!(header.api_key, 0); // Produce
        assert_eq!(header.api_version, 11);
        assert_eq!(header.correlation_id, 42);
        assert_eq!(header.client_id, Some(&b"client-1"[..]));
        assert_eq!(&payload[body_offset..], b"body");
    }

    #[test]
    fn parses_null_client_id() {
        let frame = make_request(1, 0, 100, None, b"");
        let (payload, _) = parse_frame(&frame).unwrap();
        let (header, _) = parse_request_header(payload).unwrap();
        assert!(header.client_id.is_none());
    }

    #[test]
    fn short_frame_returns_short() {
        let buf = [0u8, 0, 0]; // partial size prefix
        match parse_frame(&buf) {
            Err(ParseError::Short) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn partial_frame_returns_partial() {
        let buf = [0u8, 0, 0, 100]; // declares 100 byte payload, none supplied
        match parse_frame(&buf) {
            Err(ParseError::PartialFrame(100)) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn negative_size_rejected() {
        let buf = [0xFFu8, 0xFF, 0xFF, 0xFF]; // -1
        match parse_frame(&buf) {
            Err(ParseError::InvalidSize(-1)) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn peek_frame_size_works() {
        let buf = [0u8, 0, 0x10, 0]; // 4096
        assert_eq!(peek_frame_size(&buf).unwrap(), 4096);
    }

    #[test]
    fn empty_client_id_is_zero_length_string() {
        let frame = make_request(0, 0, 1, Some(b""), b"");
        let (payload, _) = parse_frame(&frame).unwrap();
        let (header, _) = parse_request_header(payload).unwrap();
        assert_eq!(header.client_id, Some(&[][..]));
    }
}
