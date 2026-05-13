//! `proxima_codec::FrameCodec` impl for the gRPC 5-byte length-prefix
//! framing.

use alloc::vec::Vec;

use proxima_codec::FrameCodec;

use super::{Frame, ParseError, encode, parse};

/// gRPC framing [`FrameCodec`]. Zero-sized; clone freely.
#[derive(Debug, Clone, Copy, Default)]
pub struct GrpcFrameCodec;

impl FrameCodec for GrpcFrameCodec {
    type Frame<'a> = Frame<'a>;
    type Error = ParseError;

    fn parse_frame<'a>(&self, buf: &'a [u8]) -> Result<(Self::Frame<'a>, usize), Self::Error> {
        parse(buf)
    }

    fn encode_frame(&self, frame: &Self::Frame<'_>, dest: &mut Vec<u8>) -> Result<(), Self::Error> {
        encode(frame.payload, frame.compression, dest);
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use super::super::Compression;
    use proxima_codec::FrameCodec;

    #[test]
    fn parse_frame_uncompressed_round_trips() {
        let codec = GrpcFrameCodec;
        let mut buf = Vec::new();
        encode(b"hello world", Compression::None, &mut buf);
        let (frame, consumed) = codec.parse_frame(&buf).expect("parse");
        assert_eq!(consumed, buf.len());
        assert_eq!(frame.payload, b"hello world");
        assert_eq!(frame.compression, Compression::None);
    }

    #[test]
    fn encode_frame_round_trips_through_parse() {
        let codec = GrpcFrameCodec;
        let original = Frame {
            compression: Compression::None,
            payload: b"data",
        };
        let mut encoded = Vec::new();
        codec.encode_frame(&original, &mut encoded).expect("encode");
        let (round_tripped, _) = codec.parse_frame(&encoded).expect("re-parse");
        assert_eq!(round_tripped.payload, original.payload);
        assert_eq!(round_tripped.compression, original.compression);
    }

    #[test]
    fn parse_frame_short_buffer_returns_short_error() {
        let codec = GrpcFrameCodec;
        let outcome = codec.parse_frame(b"\x00\x00\x00");
        assert!(matches!(outcome, Err(ParseError::Short)));
    }
}
