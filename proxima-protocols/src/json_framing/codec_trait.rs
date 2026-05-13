//! `proxima_codec::FrameCodec` impl for the `[u32 BE len][payload]`
//! length-prefixed JSON framing — the plug-and-play-floor sweep's
//! spot-check that "could cleanly impl `FrameCodec`" is the right call
//! for this codec, not a guess. [`codec::encode_header`]/
//! [`codec::decode_header`] already have the EXACT shape
//! `proxima_codec::LengthDelimitedCodec` hand-implements inline (a
//! `[u32 BE len][payload]` frame is the length-delimited shape by
//! definition); this impl is a thin wrapper reusing them, not new
//! parsing logic.

use alloc::vec::Vec;
use core::fmt;

use proxima_codec::FrameCodec;

use super::codec::{DecodeError, EncodeError, HEADER_BYTES, decode_header, encode_header};

/// Length-prefixed-JSON [`FrameCodec`]. `Frame<'a> = &'a [u8]` — the raw
/// JSON payload bytes, undecoded (mirrors
/// `proxima_codec::LengthDelimitedCodec`; a downstream stage decodes the
/// JSON). Zero-sized; clone freely.
#[derive(Debug, Clone, Copy, Default)]
pub struct JsonFrameCodec;

/// Sum of the codec's existing `EncodeError`/`DecodeError` plus the
/// "need more bytes" signal `FrameCodec::parse_frame` needs but
/// `decode_header` (which always receives a full 4-byte array) has no
/// reason to model itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameError {
    /// Buffer does not yet hold a complete frame — read more and retry.
    Incomplete,
    Decode(DecodeError),
    Encode(EncodeError),
}

impl fmt::Display for FrameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Incomplete => formatter.write_str("incomplete frame"),
            Self::Decode(error) => write!(formatter, "{error}"),
            Self::Encode(error) => write!(formatter, "{error}"),
        }
    }
}

impl core::error::Error for FrameError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Decode(error) => Some(error),
            Self::Encode(error) => Some(error),
            Self::Incomplete => None,
        }
    }
}

impl FrameCodec for JsonFrameCodec {
    type Frame<'a> = &'a [u8];
    type Error = FrameError;

    fn parse_frame<'a>(&self, buf: &'a [u8]) -> Result<(&'a [u8], usize), FrameError> {
        if buf.len() < HEADER_BYTES {
            return Err(FrameError::Incomplete);
        }
        let mut header = [0_u8; HEADER_BYTES];
        header.copy_from_slice(&buf[..HEADER_BYTES]);
        let len = decode_header(header).map_err(FrameError::Decode)?;
        let total = HEADER_BYTES + len;
        if buf.len() < total {
            return Err(FrameError::Incomplete);
        }
        Ok((&buf[HEADER_BYTES..total], total))
    }

    fn encode_frame(&self, frame: &&[u8], dest: &mut Vec<u8>) -> Result<(), FrameError> {
        let header = encode_header(frame.len()).map_err(FrameError::Encode)?;
        dest.extend_from_slice(&header);
        dest.extend_from_slice(frame);
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // real length-prefixed JSON wire bytes (P9): the exact shape a
    // sidecar peer sends — 4-byte BE length + a genuine JSON payload.
    fn wire_frame(payload: &[u8]) -> Vec<u8> {
        let codec = JsonFrameCodec;
        let mut dest = Vec::new();
        codec.encode_frame(&payload, &mut dest).expect("encode");
        dest
    }

    #[test]
    fn complete_frame_returns_payload_and_consumed() {
        let codec = JsonFrameCodec;
        let payload = br#"{"jsonrpc":"2.0","method":"ping","id":1}"#;
        let wire = wire_frame(payload);
        let (frame, consumed) = codec.parse_frame(&wire).expect("real JSON frame parses");
        assert_eq!(frame, payload);
        assert_eq!(consumed, wire.len());
    }

    #[test]
    fn short_header_returns_incomplete_not_error() {
        let codec = JsonFrameCodec;
        let outcome = codec.parse_frame(&[0, 0]);
        assert_eq!(outcome, Err(FrameError::Incomplete));
    }

    #[test]
    fn partial_payload_returns_incomplete_not_error() {
        let codec = JsonFrameCodec;
        let mut wire = wire_frame(b"{}");
        wire.truncate(wire.len() - 1);
        let outcome = codec.parse_frame(&wire);
        assert_eq!(outcome, Err(FrameError::Incomplete));
    }

    #[test]
    fn encode_then_parse_round_trips_a_real_payload() {
        let codec = JsonFrameCodec;
        let payload = br#"{"result":{"ok":true}}"#;
        let mut dest = Vec::new();
        codec.encode_frame(&payload.as_slice(), &mut dest).expect("encode");
        let (frame, consumed) = codec.parse_frame(&dest).expect("parse back");
        assert_eq!(frame, payload);
        assert_eq!(consumed, dest.len());
    }
}
