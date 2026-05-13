//! `proxima_codec::FrameCodec` impl for HTTP/3 wire frames.
//!
//! Gated behind the `http3_codec-codec-trait` feature so the no_std + alloc cliff
//! stays clean: the proxima-codec dependency is the only thing this
//! module imports.
//!
//! Frame shape: a borrowed [`H3Frame<'a>`] (zero-copy view into the
//! input buffer). This is the only sub-crate where the existing
//! upstream surface already matched the FrameCodec contract almost
//! exactly — `parse` already returns `(H3Frame<'_>, usize)`. The
//! only adaptation is wrapping [`H3FrameError`] (which lives in
//! `frame.rs`) in a [`H3CodecError`] that adds `core::error::Error` +
//! `Display`, and bridging `encode` (writes to `&mut [u8]`) to
//! `encode_frame` (extends a `Vec<u8>`).

use core::fmt;

use alloc::vec::Vec;
use proxima_codec::FrameCodec;

use crate::http3_codec::frame::{FrameError as H3FrameError, H3Frame, encode, parse};

/// HTTP/3 frame [`FrameCodec`]. Zero-sized; clone freely.
#[derive(Debug, Clone, Copy, Default)]
pub struct H3FrameCodec;

/// Error surface for [`H3FrameCodec`]. Wraps the existing
/// [`H3FrameError`] and adds `Display` + `core::error::Error` impls
/// (the upstream type derives neither, since the wire layer is
/// no_std-strict). The `Truncated` variant of `H3FrameError` already
/// means "buffer ran out before the frame could be parsed", so this
/// wrapper does NOT add a separate `Partial` variant the way the h1
/// and h2 wrappers do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct H3CodecError(pub H3FrameError);

impl fmt::Display for H3CodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            H3FrameError::Truncated => formatter.write_str("truncated: need more bytes"),
            H3FrameError::InvalidVarint => formatter.write_str("invalid varint"),
            H3FrameError::PayloadTooLong => formatter.write_str("payload too long"),
            H3FrameError::BufferTooSmall { needed } => {
                write!(formatter, "encode buffer too small: need {needed} bytes")
            }
        }
    }
}

impl core::error::Error for H3CodecError {}

impl From<H3FrameError> for H3CodecError {
    fn from(error: H3FrameError) -> Self {
        Self(error)
    }
}

impl FrameCodec for H3FrameCodec {
    type Frame<'a> = H3Frame<'a>;
    type Error = H3CodecError;

    fn parse_frame<'a>(&self, buf: &'a [u8]) -> Result<(Self::Frame<'a>, usize), Self::Error> {
        parse(buf).map_err(H3CodecError::from)
    }

    fn encode_frame(&self, frame: &Self::Frame<'_>, dest: &mut Vec<u8>) -> Result<(), Self::Error> {
        // bridge: encode writes into a fixed-size slice; we grow the
        // dest buffer first, then encode into the new tail, then
        // truncate to the actual bytes written.
        let start = dest.len();
        // worst-case h3 frame: 8-byte varint type + 8-byte varint length
        // + payload. payload size lives on each H3Frame variant; rather
        // than re-walking it here, we grow generously and trim.
        let estimated = frame_payload_len(frame) + 16;
        dest.resize(start + estimated, 0);
        let written = encode(frame, &mut dest[start..])?;
        dest.truncate(start + written);
        Ok(())
    }
}

/// Cheap upper-bound estimate of the encoded payload size for a frame.
/// Used to size the dest slice before calling `encode`; over-estimates
/// are harmless because the buffer is truncated to the actual written
/// length.
fn frame_payload_len(frame: &H3Frame<'_>) -> usize {
    match frame {
        H3Frame::Data { payload } | H3Frame::Settings { payload } => payload.len(),
        H3Frame::Headers { header_block } => header_block.len(),
        H3Frame::PushPromise { header_block, .. } => header_block.len() + 8,
        H3Frame::CancelPush { .. } | H3Frame::GoAway { .. } | H3Frame::MaxPushId { .. } => 8,
        H3Frame::Reserved { payload, .. } => payload.len(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_codec::FrameCodec;

    #[test]
    fn parse_frame_data_round_trip() {
        let codec = H3FrameCodec;
        // build a DATA frame manually: type=0x00, length=0x05, payload="hello"
        let buf = b"\x00\x05hello";
        let (frame, consumed) = codec.parse_frame(buf).expect("parse");
        match frame {
            H3Frame::Data { payload } => assert_eq!(payload, b"hello"),
            other => panic!("expected DATA, got {other:?}"),
        }
        assert_eq!(consumed, buf.len());
    }

    #[test]
    fn parse_frame_truncated_returns_codec_error() {
        let codec = H3FrameCodec;
        let buf = b"\x00"; // type only, length missing
        let outcome = codec.parse_frame(buf);
        assert_eq!(outcome.err(), Some(H3CodecError(H3FrameError::Truncated)));
    }

    #[test]
    fn encode_frame_round_trips_through_parse() {
        let codec = H3FrameCodec;
        let original = H3Frame::Data {
            payload: b"round-trip",
        };
        let mut encoded = Vec::new();
        codec.encode_frame(&original, &mut encoded).expect("encode");
        let (round_tripped, consumed) = codec.parse_frame(&encoded).expect("re-parse");
        assert_eq!(consumed, encoded.len());
        match round_tripped {
            H3Frame::Data { payload } => assert_eq!(payload, b"round-trip"),
            other => panic!("expected DATA, got {other:?}"),
        }
    }

    #[test]
    fn codec_error_displays_human_readable_message() {
        let displayed = format!("{}", H3CodecError(H3FrameError::Truncated));
        assert!(displayed.contains("truncated"));
    }
}
