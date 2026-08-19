//! Stateless `[u64 LE header_len][header_len bytes JSON]` frame codec —
//! the same [`proxima_codec::FrameCodec`] shape as
//! `proxima_codec::LengthDelimitedCodec` and
//! `proxima_protocols::json_framing::codec_trait::JsonFrameCodec`
//! (`proxima-protocols/src/json_framing/codec_trait.rs`), patterned
//! after both: an 8-byte little-endian length prefix — the safetensors
//! spec's own width and endianness, vs. `LengthDelimitedCodec`'s 4-byte
//! big-endian one — returning the undecoded header JSON bytes as the
//! `Frame`; `crate::parser` owns turning that into a [`crate::Manifest`],
//! mirroring how `JsonFrameCodec` leaves JSON decoding to its caller.
//!
//! Also exposed as a [`proxima_primitives::pipe::Pipe`]: the ONE-SHOT
//! "does this buffer hold a complete header frame yet" step is stateless
//! (same input always yields the same answer), which is exactly the shape
//! `proxima_protocols::codec_pipe::FrameCodecPipe<C>` already proves any
//! `FrameCodec` composes as. This crate inlines that one instantiation
//! rather than depending on `proxima-protocols` (a wire-protocol suite,
//! not a file-format reader) for a single generic struct — the dependency
//! would outweigh the few lines it saves.
//!
//! The stateful byte-accumulation loop across chunks stays OUTSIDE this
//! type, in `crate::parser::SafetensorsParser` — a self-consuming `Self ->
//! Self` state transition is not expressible as `Pipe::call(&self, ..)`,
//! the same reason `proxima_codec::DelimiterFraming` (this workspace's
//! other chunk-boundary FSM) is not a `Pipe` impl either.

use alloc::vec::Vec;
use core::future::Future;

use bytes::Bytes;
use proxima_codec::FrameCodec;
use proxima_primitives::pipe::Pipe;

use crate::error::SafetensorsError;
use crate::sized::{HEADER_LEN_BYTES, MAX_HEADER_BYTES};

/// Stateless header-frame codec. Zero-sized; clone freely. Always applies
/// [`crate::sized::MAX_HEADER_BYTES`], the build-time floor -- a caller
/// that needs a per-process override goes through
/// [`crate::parser::SafetensorsParser::with_config`] instead, which reads
/// the same header shape via [`HeaderCodec::parse_frame_with_limit`].
#[derive(Debug, Clone, Copy, Default)]
pub struct HeaderCodec;

impl HeaderCodec {
    /// Same wire parsing [`FrameCodec::parse_frame`] does, with the
    /// declared-header-length cap as an explicit parameter instead of the
    /// build-time floor. The trait method is a thin wrapper over this with
    /// `max_header_bytes = crate::sized::MAX_HEADER_BYTES`; this is the
    /// hook a `std`-tier caller with a [`crate::config::SafetensorsParserConfig`]
    /// override runs instead.
    pub fn parse_frame_with_limit<'a>(
        &self,
        buf: &'a [u8],
        max_header_bytes: u64,
    ) -> Result<(&'a [u8], usize), SafetensorsError> {
        if buf.len() < HEADER_LEN_BYTES {
            return Err(SafetensorsError::TruncatedInput {
                needed: HEADER_LEN_BYTES as u64,
                available: buf.len() as u64,
            });
        }
        let mut len_bytes = [0_u8; HEADER_LEN_BYTES];
        len_bytes.copy_from_slice(&buf[..HEADER_LEN_BYTES]);
        let header_len = u64::from_le_bytes(len_bytes);
        if header_len > max_header_bytes {
            return Err(SafetensorsError::HeaderTooLarge {
                declared: header_len,
                max: max_header_bytes,
            });
        }
        let total = HEADER_LEN_BYTES as u64 + header_len;
        if (buf.len() as u64) < total {
            return Err(SafetensorsError::TruncatedInput {
                needed: total,
                available: buf.len() as u64,
            });
        }
        // safe: `total <= buf.len()` was just proven above, and `buf.len()`
        // already fits `usize` by construction.
        let end = total as usize;
        Ok((&buf[HEADER_LEN_BYTES..end], end))
    }
}

impl FrameCodec for HeaderCodec {
    type Frame<'a> = &'a [u8];
    type Error = SafetensorsError;

    fn parse_frame<'a>(&self, buf: &'a [u8]) -> Result<(&'a [u8], usize), SafetensorsError> {
        self.parse_frame_with_limit(buf, MAX_HEADER_BYTES)
    }

    fn encode_frame(&self, frame: &&[u8], dest: &mut Vec<u8>) -> Result<(), SafetensorsError> {
        let len = frame.len() as u64;
        dest.extend_from_slice(&len.to_le_bytes());
        dest.extend_from_slice(frame);
        Ok(())
    }
}

impl Pipe for HeaderCodec {
    type In = Bytes;
    type Out = Option<(Bytes, usize)>;
    type Err = SafetensorsError;

    /// `None` means "not a hard failure, not enough bytes yet" — the
    /// caller reads more and calls again, the same Incomplete-as-`Ok`
    /// convention `FrameCodecPipe` uses for every other sans-IO codec in
    /// this workspace.
    fn call(&self, input: Bytes) -> impl Future<Output = Result<Self::Out, Self::Err>> {
        async move {
            match self.parse_frame(&input) {
                Ok((frame, consumed)) => Ok(Some((input.slice_ref(frame), consumed))),
                Err(SafetensorsError::TruncatedInput { .. }) => Ok(None),
                Err(error) => Err(error),
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use alloc::vec;

    fn wire_header(json: &[u8]) -> Vec<u8> {
        let codec = HeaderCodec;
        let mut dest = Vec::new();
        codec.encode_frame(&json, &mut dest).expect("encode");
        dest
    }

    #[test]
    fn complete_header_parses_and_reports_consumed() {
        let json = br#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let wire = wire_header(json);
        let (frame, consumed) = HeaderCodec.parse_frame(&wire).expect("parses");
        assert_eq!(frame, json);
        assert_eq!(consumed, wire.len());
    }

    #[test]
    fn short_length_prefix_is_truncated_not_error() {
        let outcome = HeaderCodec.parse_frame(&[0, 0, 0]);
        assert!(matches!(
            outcome,
            Err(SafetensorsError::TruncatedInput { .. })
        ));
    }

    #[test]
    fn partial_json_is_truncated_not_error() {
        let mut wire = wire_header(b"{}");
        wire.truncate(wire.len() - 1);
        let outcome = HeaderCodec.parse_frame(&wire);
        assert!(matches!(
            outcome,
            Err(SafetensorsError::TruncatedInput { .. })
        ));
    }

    #[test]
    fn oversized_declared_header_is_rejected_immediately() {
        let mut wire = vec![0_u8; HEADER_LEN_BYTES];
        wire[..HEADER_LEN_BYTES].copy_from_slice(&(MAX_HEADER_BYTES + 1).to_le_bytes());
        let outcome = HeaderCodec.parse_frame(&wire);
        assert!(matches!(
            outcome,
            Err(SafetensorsError::HeaderTooLarge { .. })
        ));
    }

    // The RPITIT future `HeaderCodec::call` returns is ready on first
    // poll (pure `match`, no real `.await` inside) — proven by the fact
    // this poll-once-and-expect-`Ready` helper never panics below,
    // matching the workspace's own measured claim that a ready-on-
    // first-poll `Pipe::call` is free. No executor / `proxima::test`
    // needed just to observe that.
    fn block_on_ready<F: Future>(future: F) -> F::Output {
        use core::pin::pin;
        use core::task::{Context, Poll, Waker};

        let mut future = pin!(future);
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(output) => output,
            Poll::Pending => panic!("pipe future was not ready on first poll"),
        }
    }

    #[test]
    fn pipe_call_returns_none_when_incomplete_and_some_when_complete() {
        let json = br#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let wire = Bytes::from(wire_header(json));

        let partial = wire.slice(..HEADER_LEN_BYTES + 2);
        let incomplete = block_on_ready(HeaderCodec.call(partial)).expect("no hard error");
        assert!(incomplete.is_none());

        let complete = block_on_ready(HeaderCodec.call(wire.clone())).expect("parses");
        let (frame, consumed) = complete.expect("complete frame");
        assert_eq!(&frame[..], json);
        assert_eq!(consumed, wire.len());
    }
}
