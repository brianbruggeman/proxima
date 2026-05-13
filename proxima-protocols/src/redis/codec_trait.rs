//! `proxima_codec::FrameCodec` impl for RESP2/RESP3 — the
//! plug-and-play-floor sweep's second spot-check that "could cleanly
//! impl `FrameCodec`" is the right call for this codec. [`super::parse`]
//! already returns `(Frame<'_>, usize)` and [`super::encode_into`]
//! already writes a `Frame` into a caller `&mut Vec<u8>` — this impl
//! wires the two EXISTING functions into the trait, no new parsing or
//! encoding logic.

use alloc::vec::Vec;

use proxima_codec::FrameCodec;

use super::{Frame, ParseError, encode_into, parse};

/// RESP2/RESP3 [`FrameCodec`]. Zero-sized; clone freely.
#[derive(Debug, Clone, Copy, Default)]
pub struct RedisFrameCodec;

impl FrameCodec for RedisFrameCodec {
    type Frame<'a> = Frame<'a>;
    type Error = ParseError;

    fn parse_frame<'a>(&self, buf: &'a [u8]) -> Result<(Frame<'a>, usize), ParseError> {
        parse(buf)
    }

    fn encode_frame(&self, frame: &Frame<'_>, dest: &mut Vec<u8>) -> Result<(), ParseError> {
        encode_into(frame, dest);
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn complete_simple_string_frame_round_trips() {
        // real RESP2 wire bytes (P9): the exact `+PONG\r\n` a Redis/Valkey
        // server sends in reply to PING.
        let codec = RedisFrameCodec;
        let wire = b"+PONG\r\n";
        let (frame, consumed) = codec.parse_frame(wire).expect("real RESP frame parses");
        assert_eq!(frame, Frame::SimpleString(b"PONG"));
        assert_eq!(consumed, wire.len());

        let mut dest = Vec::new();
        codec.encode_frame(&frame, &mut dest).expect("encode");
        assert_eq!(dest, wire);
    }

    #[test]
    fn short_buffer_returns_need_more_not_error() {
        let codec = RedisFrameCodec;
        let outcome = codec.parse_frame(b"+PONG");
        assert_eq!(outcome.unwrap_err(), ParseError::NeedMore);
    }

    #[test]
    fn array_command_frame_round_trips() {
        // real RESP2 client request bytes: `*2\r\n$3\r\nGET\r\n$3\r\nfoo\r\n`.
        let codec = RedisFrameCodec;
        let mut wire = Vec::new();
        encode_into(
            &Frame::Array(alloc::vec![
                Frame::BlobString(b"GET"),
                Frame::BlobString(b"foo"),
            ]),
            &mut wire,
        );
        let (frame, consumed) = codec.parse_frame(&wire).expect("real RESP array parses");
        assert_eq!(consumed, wire.len());
        match frame {
            Frame::Array(items) => {
                assert_eq!(items, alloc::vec![
                    Frame::BlobString(b"GET"),
                    Frame::BlobString(b"foo"),
                ]);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }
}
