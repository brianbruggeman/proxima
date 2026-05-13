//! Sans-IO codec for `[u32 BE len][payload]` length-prefixed framing.
//!
//! Folded in from the former `proxima-framing-json-codec` satellite crate
//! (single consumer: this crate). Pure byte-level encode/decode of the
//! 4-byte length prefix used by the JSON sidecar framing (and any other
//! length-prefixed protocol happy with a u32-BE frame size).
//!
//! Tier: no_std + alloc — always available regardless of this crate's
//! `std` feature. Caller owns the transport (`AsyncRead` / `AsyncWrite` /
//! direct socket / DPDK ring — all the codec's problem). Same shape as
//! proxima-h1-codec / proxima-h2-codec.

use core::fmt;

/// Max payload bytes per frame. 64 MiB matches the original umbrella
/// constant — large enough for typical JSON sidecar messages,
/// tight enough that a malformed length prefix can't trick a server
/// into multi-GB allocations.
pub const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

/// Length-prefix header size in bytes.
pub const HEADER_BYTES: usize = 4;

/// Encode the length prefix for `payload_len`. Returns the 4 BE bytes
/// the caller writes before the payload.
///
/// # Errors
///
/// Returns [`EncodeError::FrameTooLarge`] when `payload_len` exceeds
/// either `u32::MAX` or [`MAX_FRAME_BYTES`].
pub fn encode_header(payload_len: usize) -> Result<[u8; HEADER_BYTES], EncodeError> {
    if payload_len > MAX_FRAME_BYTES {
        return Err(EncodeError::FrameTooLarge { len: payload_len });
    }
    let len_u32 =
        u32::try_from(payload_len).map_err(|_| EncodeError::FrameTooLarge { len: payload_len })?;
    Ok(len_u32.to_be_bytes())
}

/// Decode the 4 BE bytes of the length prefix into a payload length.
/// Validates against [`MAX_FRAME_BYTES`] so callers don't have to.
///
/// # Errors
///
/// Returns [`DecodeError::FrameTooLarge`] when the declared length
/// exceeds [`MAX_FRAME_BYTES`].
pub fn decode_header(bytes: [u8; HEADER_BYTES]) -> Result<usize, DecodeError> {
    let len = u32::from_be_bytes(bytes) as usize;
    if len > MAX_FRAME_BYTES {
        return Err(DecodeError::FrameTooLarge { len });
    }
    Ok(len)
}

/// Errors from [`encode_header`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeError {
    FrameTooLarge { len: usize },
}

impl fmt::Display for EncodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FrameTooLarge { len } => {
                write!(formatter, "frame size {len} exceeds MAX_FRAME_BYTES")
            }
        }
    }
}

impl core::error::Error for EncodeError {}

/// Errors from [`decode_header`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    FrameTooLarge { len: usize },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FrameTooLarge { len } => {
                write!(
                    formatter,
                    "declared frame size {len} exceeds MAX_FRAME_BYTES"
                )
            }
        }
    }
}

impl core::error::Error for DecodeError {}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default,
        clippy::type_complexity,
        clippy::useless_vec,
        clippy::needless_range_loop,
        clippy::default_constructed_unit_structs
    )]
    use super::*;

    #[test]
    fn encode_round_trip_small() {
        let header = encode_header(7).expect("encode");
        assert_eq!(header, [0, 0, 0, 7]);
        assert_eq!(decode_header(header).expect("decode"), 7);
    }

    #[test]
    fn encode_round_trip_at_max() {
        let header = encode_header(MAX_FRAME_BYTES).expect("encode");
        assert_eq!(decode_header(header).expect("decode"), MAX_FRAME_BYTES);
    }

    #[test]
    fn encode_rejects_above_max() {
        let outcome = encode_header(MAX_FRAME_BYTES + 1);
        assert!(matches!(outcome, Err(EncodeError::FrameTooLarge { .. })));
    }

    #[test]
    fn decode_rejects_above_max() {
        // u32::MAX is much larger than MAX_FRAME_BYTES, so any
        // u32-BE that exceeds the cap should reject.
        let big = (MAX_FRAME_BYTES as u32 + 1).to_be_bytes();
        let outcome = decode_header(big);
        assert!(matches!(outcome, Err(DecodeError::FrameTooLarge { .. })));
    }

    #[test]
    fn header_bytes_constant_matches_array_size() {
        let header = encode_header(0).expect("encode");
        assert_eq!(header.len(), HEADER_BYTES);
    }
}
