//! `proxima_codec::FrameCodec` impl for HTTP/2 frames.
//!
//! Gated behind the `codec-trait` feature so the no_std + alloc cliff
//! stays clean: the proxima-codec dependency is the only thing this
//! module imports.
//!
//! Frame shape: a `(FrameHeader, FramePayload)` pair. The 9-byte
//! header is read from the buffer first; its `length` field selects
//! the payload slice that `parse_payload` then decodes. `Bytes` is
//! refcounted: this impl pays a single `Bytes::copy_from_slice` for
//! the payload bytes, since the FrameCodec contract takes `&[u8]` and
//! the existing `parse_payload` expects `&Bytes`. Callers on the hot
//! path that already own a `Bytes` should use `parse_payload`
//! directly to avoid the copy.

use core::fmt;

use alloc::vec::Vec;
use bytes::Bytes;
use proxima_codec::FrameCodec;

use crate::http2_codec::frame::{FRAME_HEADER_LEN, FrameHeader, FramePayload, encode_frame, parse_payload};

/// HTTP/2 frame [`FrameCodec`]. Zero-sized; clone freely.
#[derive(Debug, Clone, Copy, Default)]
pub struct H2FrameCodec;

/// Owned frame value: header + decoded payload. The codec uses owned
/// `Bytes` rather than borrowed slices because `FramePayload` already
/// stores `Bytes` internally (no point introducing a parallel borrowed
/// variant just to thread a lifetime).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H2Frame {
    pub header: FrameHeader,
    pub payload: FramePayload,
}

/// Error surface for [`H2FrameCodec`]. Wraps the existing
/// [`crate::http2_codec::frame::FrameError`] plus a `Partial` marker so the
/// FrameCodec `Result` shape can express "need more bytes" without
/// conflating it with a hard parse error.
///
/// `Clone` is intentionally not derived: the wrapped
/// `crate::http2_codec::frame::FrameError` does not implement Clone, and the
/// FrameCodec contract does not require it.
#[derive(Debug, PartialEq, Eq)]
pub enum FrameError {
    /// Buffer did not contain a complete frame (header or payload
    /// missing); caller should read more bytes and retry.
    Partial,
    /// Hard parse failure inside the payload — the input is malformed
    /// for the indicated frame type.
    Payload(crate::http2_codec::frame::FrameError),
}

impl fmt::Display for FrameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Partial => formatter.write_str("partial frame: need more bytes"),
            Self::Payload(inner) => write!(formatter, "payload: {inner}"),
        }
    }
}

impl core::error::Error for FrameError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Payload(inner) => Some(inner),
            Self::Partial => None,
        }
    }
}

impl From<crate::http2_codec::frame::FrameError> for FrameError {
    fn from(error: crate::http2_codec::frame::FrameError) -> Self {
        Self::Payload(error)
    }
}

impl FrameCodec for H2FrameCodec {
    type Frame<'a> = H2Frame;
    type Error = FrameError;

    fn parse_frame<'a>(&self, buf: &'a [u8]) -> Result<(Self::Frame<'a>, usize), Self::Error> {
        let header = FrameHeader::parse(buf).ok_or(FrameError::Partial)?;
        let payload_len = header.length as usize;
        let total = FRAME_HEADER_LEN + payload_len;
        if buf.len() < total {
            return Err(FrameError::Partial);
        }
        // copy_from_slice is unavoidable here: the codec-trait contract
        // takes &[u8] and parse_payload wants &Bytes. callers with a
        // pre-existing Bytes should use parse_payload directly to skip
        // this copy.
        let payload_bytes = Bytes::copy_from_slice(&buf[FRAME_HEADER_LEN..total]);
        let payload = parse_payload(&header, &payload_bytes)?;
        Ok((H2Frame { header, payload }, total))
    }

    fn encode_frame(&self, frame: &Self::Frame<'_>, dest: &mut Vec<u8>) -> Result<(), Self::Error> {
        encode_frame(
            frame.header.frame_type,
            frame.header.flags,
            frame.header.stream_id,
            &frame.payload,
            dest,
        );
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::http2_codec::frame::FrameType;
    use proxima_codec::FrameCodec;

    fn build_ping_frame() -> Vec<u8> {
        // SETTINGS frame is the simplest fully-parseable wire shape
        // we can hand-build: zero payload + ACK flag.
        let mut buf = Vec::new();
        encode_frame(
            FrameType::Settings,
            0x01, // ACK
            0,    // connection-level stream id
            &FramePayload::Settings(crate::http2_codec::frame::StandardSettings::default()),
            &mut buf,
        );
        buf
    }

    #[test]
    fn parse_frame_full_buffer_returns_frame_and_consumed() {
        let codec = H2FrameCodec;
        let buf = build_ping_frame();
        let (frame, consumed) = codec.parse_frame(&buf).expect("complete frame");
        assert_eq!(frame.header.frame_type, FrameType::Settings);
        assert_eq!(frame.header.stream_id, 0);
        assert!(frame.header.has_flag(0x01)); // ACK
        assert_eq!(consumed, buf.len());
    }

    #[test]
    fn parse_frame_short_header_returns_partial() {
        let codec = H2FrameCodec;
        let buf = build_ping_frame();
        let truncated = &buf[..FRAME_HEADER_LEN - 1];
        assert_eq!(
            codec.parse_frame(truncated).err(),
            Some(FrameError::Partial)
        );
    }

    #[test]
    fn parse_frame_short_payload_returns_partial() {
        let codec = H2FrameCodec;
        // SETTINGS frame with 6-byte payload claim but only 3 bytes supplied
        let mut buf = vec![
            0x00, 0x00, 0x06, // length = 6
            0x04, // SETTINGS
            0x00, // flags
            0x00, 0x00, 0x00, 0x00, // stream id 0
            0xde, 0xad, 0xbe, // only 3 bytes of payload
        ];
        buf.truncate(FRAME_HEADER_LEN + 3);
        assert_eq!(codec.parse_frame(&buf).err(), Some(FrameError::Partial));
    }

    #[test]
    fn encode_frame_round_trips_through_parse() {
        let codec = H2FrameCodec;
        let buf = build_ping_frame();
        let (frame, _) = codec.parse_frame(&buf).expect("parse");
        let mut encoded = Vec::with_capacity(buf.len());
        codec.encode_frame(&frame, &mut encoded).expect("encode");
        let (round_tripped, _) = codec.parse_frame(&encoded).expect("re-parse");
        assert_eq!(round_tripped.header.frame_type, frame.header.frame_type);
        assert_eq!(round_tripped.header.flags, frame.header.flags);
        assert_eq!(round_tripped.header.stream_id, frame.header.stream_id);
        assert_eq!(round_tripped.payload, frame.payload);
    }

    #[test]
    fn frame_error_partial_displays_human_readable_message() {
        let displayed = format!("{}", FrameError::Partial);
        assert!(displayed.contains("partial"));
    }
}
