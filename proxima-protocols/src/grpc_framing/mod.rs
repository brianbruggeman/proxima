//! gRPC length-prefix framing (sans-IO).
//!
//! Tracked as P9 in `docs/protocol-gap/discipline.md`. The
//! `application/grpc` body framing per RFC: every message is
//! prefixed with **5 bytes** — 1-byte compression flag + 4-byte
//! big-endian message length — followed by `len` bytes of opaque
//! payload (typically a protobuf message, but the framing layer
//! is content-agnostic).
//!
//! This module is the **sans-IO** codec. The h2 wire is the existing
//! `http2` feature; gRPC over HTTP/2 = h2 DATA frames carrying these
//! 5-byte-prefixed messages. The h2 listener/upstream layers can
//! consume / produce these frames without depending on `tonic` or
//! `prost`.
//!
//! Sub-flag: `grpc-framing` (default off — only callers that build
//! gRPC pipes/clients need this).


use alloc::vec::Vec;

/// gRPC compression flag (the 1-byte prefix at the start of every
/// frame). Per spec, only 0/1 are defined; other values are reserved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Compression {
    /// Uncompressed payload.
    None = 0,
    /// Payload compressed per the `grpc-encoding` header.
    Compressed = 1,
}

/// One gRPC frame. Borrows the payload directly from the source buffer.
#[derive(Debug, Clone, Copy)]
pub struct Frame<'a> {
    pub compression: Compression,
    pub payload: &'a [u8],
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("buffer shorter than 5-byte gRPC frame header")]
    Short,
    #[error("buffer holds header but is shorter than declared payload length {0}")]
    PartialPayload(u32),
    #[error("compression flag {0} not defined in gRPC spec (only 0/1)")]
    ReservedCompression(u8),
}

/// Maximum gRPC message size proxima accepts in one frame.
///
/// gRPC has no hard wire limit, but most peers cap at 4 MiB to bound
/// memory; we mirror the same default. Callers that need larger
/// messages should parse manually using [`peek_length`] + a custom
/// allocator path.
pub const MAX_FRAME_BYTES: usize = 4 * 1024 * 1024;

/// Read the length field of a frame without claiming the payload.
/// Cheap (5 byte reads) — useful for size-budget checks before
/// allocating.
#[inline]
pub fn peek_length(buf: &[u8]) -> Result<(Compression, u32), ParseError> {
    if buf.len() < 5 {
        return Err(ParseError::Short);
    }
    let compression = match buf[0] {
        0 => Compression::None,
        1 => Compression::Compressed,
        other => return Err(ParseError::ReservedCompression(other)),
    };
    let length = u32::from_be_bytes([buf[1], buf[2], buf[3], buf[4]]);
    Ok((compression, length))
}

/// Parse one full frame at the start of `buf`. Returns the borrowed
/// frame plus the number of bytes consumed (always `5 + payload.len()`).
#[inline]
pub fn parse(buf: &[u8]) -> Result<(Frame<'_>, usize), ParseError> {
    let (compression, length) = peek_length(buf)?;
    let len = length as usize;
    if buf.len() < 5 + len {
        return Err(ParseError::PartialPayload(length));
    }
    Ok((
        Frame {
            compression,
            payload: &buf[5..5 + len],
        },
        5 + len,
    ))
}

/// Encode a frame into `dest`. Single allocation strategy is the
/// caller's: we just push the 5-byte header + payload onto the
/// destination buffer.
#[inline]
pub fn encode(message: &[u8], compression: Compression, dest: &mut Vec<u8>) {
    dest.reserve(5 + message.len());
    dest.push(compression as u8);
    dest.extend_from_slice(&(message.len() as u32).to_be_bytes());
    dest.extend_from_slice(message);
}

#[cfg(feature = "grpc_framing-codec-trait")]
pub mod codec_trait;
#[cfg(feature = "grpc_framing-codec-trait")]
pub use codec_trait::GrpcFrameCodec;

#[cfg(feature = "grpc_framing-frame-pipe")]
pub mod frame_codec_pipe;
#[cfg(feature = "grpc_framing-frame-pipe")]
pub use frame_codec_pipe::OwnedGrpcFrame;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn round_trips_empty_payload() {
        let mut buf = Vec::new();
        encode(&[], Compression::None, &mut buf);
        assert_eq!(buf, vec![0, 0, 0, 0, 0]);
        let (frame, used) = parse(&buf).unwrap();
        assert_eq!(used, 5);
        assert_eq!(frame.compression, Compression::None);
        assert_eq!(frame.payload, &[] as &[u8]);
    }

    #[test]
    fn round_trips_small_message() {
        let message = b"hello, grpc world";
        let mut buf = Vec::new();
        encode(message, Compression::None, &mut buf);
        assert_eq!(buf[0], 0);
        assert_eq!(&buf[1..5], &(message.len() as u32).to_be_bytes());
        let (frame, used) = parse(&buf).unwrap();
        assert_eq!(used, 5 + message.len());
        assert_eq!(frame.payload, message);
    }

    #[test]
    fn round_trips_compressed_flag() {
        let mut buf = Vec::new();
        encode(b"payload", Compression::Compressed, &mut buf);
        let (frame, _) = parse(&buf).unwrap();
        assert_eq!(frame.compression, Compression::Compressed);
    }

    #[test]
    fn short_buffer_returns_short() {
        let buf = [0u8; 4];
        assert!(matches!(parse(&buf), Err(ParseError::Short)));
    }

    #[test]
    fn partial_payload_returns_partial() {
        // header declares 100 bytes; buffer only has 50.
        let mut buf = vec![0, 0, 0, 0, 100];
        buf.extend_from_slice(&[0xAB; 50]);
        match parse(&buf) {
            Err(ParseError::PartialPayload(100)) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn reserved_compression_rejected() {
        let buf = [2u8, 0, 0, 0, 0];
        match parse(&buf) {
            Err(ParseError::ReservedCompression(2)) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn peek_length_does_not_require_payload() {
        let buf = [0u8, 0, 0, 0x10, 0]; // declares 4096 byte payload
        let (comp, len) = peek_length(&buf).unwrap();
        assert_eq!(comp, Compression::None);
        assert_eq!(len, 4096);
    }
}
