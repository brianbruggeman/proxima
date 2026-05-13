//! `proxima_codec::FrameCodec` impl for QUIC wire frames.
//!
//! Gated behind the `codec-trait` feature so the no_std + alloc cliff
//! stays clean: the proxima-codec dependency is the only thing this
//! module imports.
//!
//! Frame shape: a borrowed [`Frame<'a>`] from the existing
//! [`crate::quic::frame`] module. `parse_frame` delegates to the existing
//! `parse` (which already returns `(Frame<'_>, usize)`); `encode_frame`
//! bridges the upstream `Frame::encode(&mut [u8])` signature to the
//! FrameCodec `&mut Vec<u8>` contract by growing dest and trimming.

use core::fmt;

use alloc::vec::Vec;
use proxima_codec::FrameCodec;

use crate::quic::frame::{DecodeError, EncodeError, Frame, parse};

/// QUIC frame [`FrameCodec`]. Zero-sized; clone freely.
#[derive(Debug, Clone, Copy, Default)]
pub struct QuicFrameCodec;

/// Error surface for [`QuicFrameCodec`]. Wraps the upstream
/// [`DecodeError`] and [`EncodeError`] and adds `Display` +
/// `core::error::Error` (the upstream types are no_std-strict). The
/// `Truncated` variant of `DecodeError` already means "need more
/// bytes", so this wrapper does NOT add a separate `Partial` variant.
#[derive(Debug)]
pub enum QuicCodecError {
    Decode(DecodeError),
    Encode(EncodeError),
}

impl fmt::Display for QuicCodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(inner) => write!(formatter, "decode: {inner:?}"),
            Self::Encode(inner) => write!(formatter, "encode: {inner:?}"),
        }
    }
}

impl core::error::Error for QuicCodecError {}

impl From<DecodeError> for QuicCodecError {
    fn from(error: DecodeError) -> Self {
        Self::Decode(error)
    }
}

impl From<EncodeError> for QuicCodecError {
    fn from(error: EncodeError) -> Self {
        Self::Encode(error)
    }
}

impl FrameCodec for QuicFrameCodec {
    type Frame<'a> = Frame<'a>;
    type Error = QuicCodecError;

    fn parse_frame<'a>(&self, buf: &'a [u8]) -> Result<(Self::Frame<'a>, usize), Self::Error> {
        parse(buf).map_err(QuicCodecError::from)
    }

    fn encode_frame(&self, frame: &Self::Frame<'_>, dest: &mut Vec<u8>) -> Result<(), Self::Error> {
        // bridge: upstream encode writes into a fixed-size slice; we
        // grow the dest first, encode into the tail, then trim to the
        // actual bytes written. cap is the worst-case QUIC frame size
        // — anything beyond should be carried as Datagram or split.
        const MAX_ENCODED: usize = 65_536;
        let start = dest.len();
        dest.resize(start + MAX_ENCODED, 0);
        let written = frame
            .encode(&mut dest[start..])
            .map_err(QuicCodecError::from)?;
        dest.truncate(start + written);
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_codec::FrameCodec;

    #[test]
    fn parse_frame_ping_round_trips() {
        // PING is the simplest QUIC frame — a single 0x01 byte, no payload.
        let codec = QuicFrameCodec;
        let buf = b"\x01";
        let (frame, consumed) = codec.parse_frame(buf).expect("parse PING");
        assert_eq!(consumed, 1);
        let mut encoded = Vec::new();
        codec.encode_frame(&frame, &mut encoded).expect("encode");
        let (round_tripped, _) = codec.parse_frame(&encoded).expect("re-parse");
        assert_eq!(frame, round_tripped);
    }

    #[test]
    fn parse_frame_padding_round_trips() {
        // PADDING is 0x00. parse may consume one or more zeros depending
        // on implementation; just verify we get something back and the
        // round-trip is stable.
        let codec = QuicFrameCodec;
        let buf = b"\x00";
        let (frame, consumed) = codec.parse_frame(buf).expect("parse PADDING");
        assert!(consumed >= 1);
        let mut encoded = Vec::new();
        codec.encode_frame(&frame, &mut encoded).expect("encode");
        let (round_tripped, _) = codec.parse_frame(&encoded).expect("re-parse");
        assert_eq!(frame, round_tripped);
    }

    #[test]
    fn parse_frame_truncated_returns_decode_error() {
        let codec = QuicFrameCodec;
        // empty buffer can't even hold a frame type byte
        let outcome = codec.parse_frame(b"");
        assert!(matches!(outcome, Err(QuicCodecError::Decode(_))));
    }

    #[test]
    fn codec_error_displays_human_readable_message() {
        let displayed = format!("{}", QuicCodecError::Decode(DecodeError::Truncated));
        assert!(displayed.contains("decode"));
    }
}
