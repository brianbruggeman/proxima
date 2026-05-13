//! [`GrpcFrameCodec`] plugs into the GENERIC
//! [`crate::codec_pipe::FrameCodecPipe`] — the second instantiation (after
//! [`crate::http1_codec::codec_trait::H1RequestCodec`]) proving the
//! adapter is written ONCE and reused per codec, not rewritten per
//! protocol. Part of the plug-and-play-floor sweep
//! (`validate/pipe-transform-sweep`).
//!
//! Supplies the two per-codec seams: [`OwnFrame`] (re-own the borrowed
//! `payload: &[u8]` as `Bytes` via [`Bytes::slice_ref`] — zero-copy, a
//! refcount bump over the same backing storage) and [`Incomplete`]
//! (`ParseError::Short`/`PartialPayload` both mean "read more bytes";
//! `ReservedCompression` is a hard parse failure).

use bytes::Bytes;

use crate::codec_pipe::{Incomplete, OwnFrame};
use crate::grpc_framing::codec_trait::GrpcFrameCodec;
use crate::grpc_framing::{Compression, Frame, ParseError};

/// Owned counterpart of [`Frame`]: the payload re-owned as [`Bytes`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedGrpcFrame {
    pub compression: Compression,
    pub payload: Bytes,
}

impl OwnFrame for GrpcFrameCodec {
    type Owned = OwnedGrpcFrame;

    fn own_frame(source: &Bytes, frame: &Frame<'_>) -> OwnedGrpcFrame {
        OwnedGrpcFrame {
            compression: frame.compression,
            payload: source.slice_ref(frame.payload),
        }
    }
}

impl Incomplete for ParseError {
    fn is_incomplete(&self) -> bool {
        matches!(self, ParseError::Short | ParseError::PartialPayload(_))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::codec_pipe::FrameCodecPipe;
    use alloc::vec::Vec;
    use core::future::Future;
    use proxima_primitives::pipe::Pipe;

    /// Dependency-free executor for the always-ready probe futures (mirrors
    /// `http1_codec::frame_codec_pipe`'s own test helper).
    fn block_on<Fut: Future>(future: Fut) -> Fut::Output {
        let mut pinned = core::pin::pin!(future);
        let mut context = core::task::Context::from_waker(core::task::Waker::noop());
        loop {
            if let core::task::Poll::Ready(output) = pinned.as_mut().poll(&mut context) {
                return output;
            }
        }
    }

    // real gRPC wire framing (P9): the exact 5-byte-prefix shape a gRPC peer
    // sends — 1-byte compression flag + 4-byte BE length + payload.
    fn encode_frame(message: &[u8], compression: Compression) -> Vec<u8> {
        let mut buf = Vec::new();
        crate::grpc_framing::encode(message, compression, &mut buf);
        buf
    }

    #[test]
    fn complete_frame_returns_owned_frame_and_consumed() {
        let codec: FrameCodecPipe<GrpcFrameCodec> = FrameCodecPipe::default();
        let wire = encode_frame(b"hello, grpc world", Compression::None);
        let input = Bytes::copy_from_slice(&wire);
        let outcome = block_on(Pipe::call(&codec, input.clone())).expect("real gRPC frame parses");
        let (frame, consumed) = outcome.expect("complete frame, not partial");
        assert_eq!(&frame.payload[..], b"hello, grpc world");
        assert_eq!(frame.compression, Compression::None);
        assert_eq!(consumed, wire.len());
    }

    #[test]
    fn owned_frame_payload_shares_the_input_bytes_allocation() {
        let codec: FrameCodecPipe<GrpcFrameCodec> = FrameCodecPipe::default();
        let wire = encode_frame(b"zero-copy", Compression::None);
        let input = Bytes::copy_from_slice(&wire);
        let outcome = block_on(Pipe::call(&codec, input.clone()))
            .expect("parse")
            .expect("complete");
        assert_eq!(outcome.0.payload.as_ptr(), input[5..].as_ptr());
    }

    #[test]
    fn short_buffer_returns_none_not_error() {
        let codec: FrameCodecPipe<GrpcFrameCodec> = FrameCodecPipe::default();
        let truncated = Bytes::copy_from_slice(&[0u8, 0, 0]);
        let outcome = block_on(Pipe::call(&codec, truncated)).expect("short is Ok(None), not Err");
        assert!(outcome.is_none());
    }

    #[test]
    fn partial_payload_returns_none_not_error() {
        let codec: FrameCodecPipe<GrpcFrameCodec> = FrameCodecPipe::default();
        // header declares 100 bytes; buffer only has 50 — a realistic
        // mid-stream read, not a synthetic edge case.
        let mut buf = alloc::vec![0u8, 0, 0, 0, 100];
        buf.extend_from_slice(&[0xAB; 50]);
        let outcome = block_on(Pipe::call(&codec, Bytes::from(buf)))
            .expect("partial payload is Ok(None), not Err");
        assert!(outcome.is_none());
    }

    #[test]
    fn reserved_compression_returns_hard_error() {
        let codec: FrameCodecPipe<GrpcFrameCodec> = FrameCodecPipe::default();
        let bad = Bytes::copy_from_slice(&[2u8, 0, 0, 0, 0]);
        let outcome = block_on(Pipe::call(&codec, bad));
        assert!(matches!(outcome, Err(ParseError::ReservedCompression(2))));
    }
}
