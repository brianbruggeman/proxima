//! `proxima_codec::FrameCodec` impl for the WebSocket (RFC 6455)
//! frame codec.

use alloc::vec::Vec;

use proxima_codec::FrameCodec;

use super::{Frame, ParseError, encode_header, parse_frame, unmask_in_place};

/// WebSocket frame [`FrameCodec`]. Zero-sized; clone freely.
///
/// `parse_frame` runs the strict RFC 6455 §5.2 path with no
/// negotiated extensions — RSV bits trigger `ReservedBits`. Callers
/// that have negotiated permessage-deflate (RFC 7692) should keep
/// using [`crate::parse_frame_with_extensions`] directly instead of
/// the trait shape.
#[derive(Debug, Clone, Copy, Default)]
pub struct WebSocketFrameCodec;

impl FrameCodec for WebSocketFrameCodec {
    type Frame<'a> = Frame<'a>;
    type Error = ParseError;

    fn parse_frame<'a>(&self, buf: &'a [u8]) -> Result<(Self::Frame<'a>, usize), Self::Error> {
        parse_frame(buf)
    }

    fn encode_frame(&self, frame: &Self::Frame<'_>, dest: &mut Vec<u8>) -> Result<(), Self::Error> {
        encode_header(
            frame.fin,
            frame.opcode,
            frame.payload.len(),
            frame.mask,
            dest,
        );
        match frame.mask {
            Some(key) => {
                let start = dest.len();
                dest.extend_from_slice(frame.payload);
                unmask_in_place(&mut dest[start..], key);
            }
            None => dest.extend_from_slice(frame.payload),
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use super::super::Opcode;
    use proxima_codec::FrameCodec;

    #[test]
    fn parse_frame_unmasked_text_round_trips() {
        let codec = WebSocketFrameCodec;
        let mut buf = vec![0x81, 0x05];
        buf.extend_from_slice(b"hello");
        let (frame, consumed) = codec.parse_frame(&buf).expect("parse");
        assert_eq!(consumed, buf.len());
        assert_eq!(frame.opcode, Opcode::Text);
        assert_eq!(frame.payload, b"hello");
        assert!(frame.fin);
        assert!(frame.mask.is_none());
    }

    #[test]
    fn encode_frame_unmasked_text_round_trips_through_parse() {
        let codec = WebSocketFrameCodec;
        let original = Frame {
            fin: true,
            opcode: Opcode::Text,
            compressed: false,
            mask: None,
            payload: b"hi",
        };
        let mut encoded = Vec::new();
        codec.encode_frame(&original, &mut encoded).expect("encode");
        let (round_tripped, _) = codec.parse_frame(&encoded).expect("re-parse");
        assert_eq!(round_tripped.payload, original.payload);
        assert_eq!(round_tripped.opcode, original.opcode);
        assert_eq!(round_tripped.fin, original.fin);
        assert!(round_tripped.mask.is_none());
    }

    #[test]
    fn parse_frame_short_buffer_returns_short_error() {
        let codec = WebSocketFrameCodec;
        let outcome = codec.parse_frame(b"\x81");
        assert!(matches!(outcome, Err(ParseError::Short)));
    }
}
