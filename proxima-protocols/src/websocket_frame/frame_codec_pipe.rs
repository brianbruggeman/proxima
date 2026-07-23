//! [`WebSocketFrameCodec`] plugs into the GENERIC
//! [`crate::codec_pipe::FrameCodecPipe`] — the third instantiation (after
//! [`crate::http1_codec::codec_trait::H1RequestCodec`] and
//! [`crate::grpc_framing::codec_trait::GrpcFrameCodec`]), the sweep's
//! ≥3-codec proof that the adapter is written ONCE and reused per codec.
//! Part of the plug-and-play-floor sweep (`validate/pipe-transform-sweep`).
//!
//! Supplies the two per-codec seams: [`OwnFrame`] (re-own the borrowed
//! `payload: &[u8]` as `Bytes`; the mask key is `Copy` already) and
//! [`Incomplete`] (`ParseError::Short`/`PartialPayload` mean "read more
//! bytes"; `ReservedBits`/`UnknownOpcode`/`OversizedControl`/
//! `PayloadTooLarge` are hard parse failures — RFC 6455 violations, not a
//! buffering state).

use bytes::Bytes;

use crate::codec_pipe::{Incomplete, OwnFrame};
use crate::websocket_frame::codec_trait::WebSocketFrameCodec;
use crate::websocket_frame::{Frame, Opcode, ParseError};

/// Owned counterpart of [`Frame`]: the payload re-owned as [`Bytes`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedWsFrame {
    pub fin: bool,
    pub opcode: Opcode,
    pub compressed: bool,
    pub mask: Option<[u8; 4]>,
    pub payload: Bytes,
}

impl OwnFrame for WebSocketFrameCodec {
    type Source = Bytes;
    type Owned = OwnedWsFrame;

    fn own_frame(source: &Bytes, frame: &Frame<'_>) -> OwnedWsFrame {
        OwnedWsFrame {
            fin: frame.fin,
            opcode: frame.opcode,
            compressed: frame.compressed,
            mask: frame.mask,
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

    // real unmasked-text WebSocket frame bytes (P9): 0x81 = FIN + text
    // opcode, 0x05 = unmasked 5-byte payload length — the exact wire shape
    // a server sends per RFC 6455 §5.2.
    fn real_text_frame() -> Vec<u8> {
        let mut buf = alloc::vec![0x81, 0x05];
        buf.extend_from_slice(b"hello");
        buf
    }

    #[test]
    fn complete_frame_returns_owned_frame_and_consumed() {
        let codec: FrameCodecPipe<WebSocketFrameCodec> = FrameCodecPipe::default();
        let wire = real_text_frame();
        let input = Bytes::copy_from_slice(&wire);
        let outcome =
            block_on(Pipe::call(&codec, input.clone())).expect("real WS frame parses");
        let (frame, consumed) = outcome.expect("complete frame, not partial");
        assert_eq!(&frame.payload[..], b"hello");
        assert_eq!(frame.opcode, Opcode::Text);
        assert!(frame.fin);
        assert!(frame.mask.is_none());
        assert_eq!(consumed, wire.len());
    }

    #[test]
    fn owned_frame_payload_shares_the_input_bytes_allocation() {
        let codec: FrameCodecPipe<WebSocketFrameCodec> = FrameCodecPipe::default();
        let wire = real_text_frame();
        let input = Bytes::copy_from_slice(&wire);
        let outcome = block_on(Pipe::call(&codec, input.clone()))
            .expect("parse")
            .expect("complete");
        assert_eq!(outcome.0.payload.as_ptr(), input[2..].as_ptr());
    }

    #[test]
    fn short_buffer_returns_none_not_error() {
        let codec: FrameCodecPipe<WebSocketFrameCodec> = FrameCodecPipe::default();
        let truncated = Bytes::copy_from_slice(&[0x81]);
        let outcome = block_on(Pipe::call(&codec, truncated)).expect("short is Ok(None), not Err");
        assert!(outcome.is_none());
    }

    #[test]
    fn reserved_bits_return_hard_error() {
        let codec: FrameCodecPipe<WebSocketFrameCodec> = FrameCodecPipe::default();
        // RSV1 set (0xC1 = FIN + RSV1 + text opcode) with no extension
        // negotiated — a real, RFC-invalid wire byte, not a fabricated enum.
        let mut wire = real_text_frame();
        wire[0] = 0xC1;
        let outcome = block_on(Pipe::call(&codec, Bytes::copy_from_slice(&wire)));
        assert!(matches!(outcome, Err(ParseError::ReservedBits)));
    }
}
