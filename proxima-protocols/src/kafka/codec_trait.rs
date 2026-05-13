//! `proxima_codec::FrameCodec` impl for Kafka's `[i32 BE size][payload]`
//! request/response framing — the plug-and-play-floor sweep classified
//! [`super::parse_frame`] as already returning the `(Frame<'_>, usize)`
//! shape `FrameCodec::parse_frame` needs. The Kafka "frame" IS the
//! length-prefixed payload, opaque past the wire envelope (header /
//! body decoding is the separate [`super::parse_request_header`], out
//! of scope for framing). `encode_frame` is the mechanical inverse of
//! the size-prefix math [`super::peek_frame_size`] already performs —
//! the same `[len][payload]` shape `proxima_codec::LengthDelimitedCodec`
//! hand-implements inline — not new protocol logic.

use alloc::vec::Vec;
use core::fmt;

use proxima_codec::FrameCodec;

use super::{ParseError, parse_frame};

/// Kafka wire-framing [`FrameCodec`]. Zero-sized; clone freely.
#[derive(Debug, Clone, Copy, Default)]
pub struct KafkaFrameCodec;

/// Sum of [`ParseError`] (the decode path) plus the one error
/// `encode_frame` can hit that `ParseError` has no reason to model
/// itself: a payload too large for Kafka's signed 32-bit length
/// prefix. `ParseError` itself carries no `PartialEq` (thiserror-only
/// derive upstream), so this wrapper matches that and tests assert via
/// `matches!` like the rest of the sweep's kafka test suite.
#[derive(Debug)]
pub enum FrameError {
    Parse(ParseError),
    FrameTooLarge { len: usize },
}

impl fmt::Display for FrameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(error) => write!(formatter, "{error}"),
            Self::FrameTooLarge { len } => {
                write!(formatter, "kafka frame length {len} exceeds i32::MAX")
            }
        }
    }
}

impl core::error::Error for FrameError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Parse(error) => Some(error),
            Self::FrameTooLarge { .. } => None,
        }
    }
}

impl FrameCodec for KafkaFrameCodec {
    type Frame<'a> = &'a [u8];
    type Error = FrameError;

    fn parse_frame<'a>(&self, buf: &'a [u8]) -> Result<(&'a [u8], usize), FrameError> {
        parse_frame(buf).map_err(FrameError::Parse)
    }

    fn encode_frame(&self, frame: &&[u8], dest: &mut Vec<u8>) -> Result<(), FrameError> {
        let length = i32::try_from(frame.len())
            .map_err(|_error| FrameError::FrameTooLarge { len: frame.len() })?;
        dest.extend_from_slice(&length.to_be_bytes());
        dest.extend_from_slice(frame);
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn real_produce_request_frame_round_trips() {
        // real Kafka wire bytes (P9): a v0 Produce request header
        // (api_key=0, api_version=11, correlation_id=42, client_id
        // "client-1") followed by a request body, framed with the
        // 4-byte BE length prefix.
        let codec = KafkaFrameCodec;
        let mut payload = Vec::new();
        payload.extend_from_slice(&0i16.to_be_bytes());
        payload.extend_from_slice(&11i16.to_be_bytes());
        payload.extend_from_slice(&42i32.to_be_bytes());
        payload.extend_from_slice(&8i16.to_be_bytes());
        payload.extend_from_slice(b"client-1");
        payload.extend_from_slice(b"body");

        let mut wire = Vec::new();
        codec
            .encode_frame(&payload.as_slice(), &mut wire)
            .expect("encode");

        let (frame, consumed) = codec.parse_frame(&wire).expect("real kafka frame parses");
        assert_eq!(frame, payload.as_slice());
        assert_eq!(consumed, wire.len());
    }

    #[test]
    fn short_length_prefix_is_incomplete_not_error() {
        let codec = KafkaFrameCodec;
        let outcome = codec.parse_frame(&[0, 0, 0]);
        assert!(matches!(outcome, Err(FrameError::Parse(ParseError::Short))));
    }

    #[test]
    fn partial_payload_signals_partial_frame() {
        let codec = KafkaFrameCodec;
        // declares 10 bytes of payload, supplies none.
        let outcome = codec.parse_frame(&[0, 0, 0, 10]);
        assert!(matches!(
            outcome,
            Err(FrameError::Parse(ParseError::PartialFrame(10)))
        ));
    }

    #[test]
    fn negative_size_is_rejected() {
        let codec = KafkaFrameCodec;
        let outcome = codec.parse_frame(&[0xFF, 0xFF, 0xFF, 0xFF]);
        assert!(matches!(
            outcome,
            Err(FrameError::Parse(ParseError::InvalidSize(-1)))
        ));
    }

    #[test]
    fn encode_then_parse_round_trips_an_empty_payload() {
        let codec = KafkaFrameCodec;
        let mut wire = Vec::new();
        codec
            .encode_frame(&[].as_slice(), &mut wire)
            .expect("encode");
        let (frame, consumed) = codec.parse_frame(&wire).expect("parse back");
        assert!(frame.is_empty());
        assert_eq!(consumed, wire.len());
    }
}
