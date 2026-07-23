//! [`H1RequestCodec`] plugs into the GENERIC
//! [`crate::codec_pipe::FrameCodecPipe`] — proves an `H1RequestCodec` (a
//! `proxima_codec::FrameCodec`) composes directly as a
//! `proxima_primitives::pipe::Pipe` with no codec rewrite. The
//! adapter half of the pipe-shaped-connection reshape spike: the byte-ring
//! read side (`proxima-net`'s `DrainSource`/`Readiness` pump) hands the
//! pipe an owned [`Bytes`] window; it returns one parsed frame plus the
//! bytes consumed, or `None` when the window holds only a partial head.
//!
//! This module supplies the two per-codec seams the generic adapter is
//! generic over (see `crate::codec_pipe` for why they are per-codec, not
//! reinvented here): [`OwnFrame`] (re-own a borrowed [`RequestHead`] as
//! [`OwnedFrame`]) and [`Incomplete`] (`FrameError::Partial` means "read
//! more bytes"). This module used to hold a hardcoded, H1-only
//! `FrameCodecPipe` struct duplicating what is now the shared adapter;
//! generalizing it is the C1 deliverable of the plug-and-play-floor sweep
//! (`validate/pipe-transform-sweep`).
//!
//! `OwnedFrame` re-owns [`RequestHead`]'s borrowed method/path/header
//! slices via [`Bytes::slice_ref`] — an `Arc` refcount bump over the SAME
//! backing storage, not a byte copy. The one real copy in the read path
//! happens earlier, at the T0 (stack ring) -> T1 (owned `Bytes`) crossing
//! (`proxima-net`'s `poll_next_owned`); by the time bytes reach this pipe
//! they are already refcounted, so re-owning a borrowed sub-slice is free.
//!
//! [`OwnedFrame`] also implements the `proxima-primitives` capability
//! traits `Replayable` / `Idempotent` / `Labeled` — the seam that lets a
//! parsed frame be wrapped by an ordinary pipe combinator (`Retry`, ...)
//! with zero extra plumbing, proving the composed `AndThen<FrameCodecPipe<
//! H1RequestCodec>, App>` is not a closed box: any stage downstream of the
//! codec can be decorated like any other `Pipe`. See `proxima-net`'s
//! `pipe_connection` module for the end-to-end proof.

use alloc::vec::Vec;
use core::fmt;
use core::future::Future;

use bytes::Bytes;
use proxima_codec::FrameCodec;
use proxima_core::ProximaError;
use proxima_primitives::pipe::capabilities::{Idempotent, Replayable};
use proxima_primitives::pipe::{Pipe, SendPipe};
use proxima_primitives::pipe::labeled::Labeled;
use proxima_primitives::pipe::telemetry_surface::{Labels, NoopTelemetry, TelemetryHandle};

use crate::codec_pipe::{Incomplete, OwnFrame};
use crate::http1_codec::codec_trait::{FrameError, H1RequestCodec};
use crate::http1_codec::h1::{HttpVersion, RequestHead};
use crate::http1_codec::h1_body::{
    BodyDecoder, BodyFraming, DecodeError, DecoderLimits, Status as BodyStatus,
};

/// One re-owned header: [`Bytes::slice_ref`] over the same backing
/// storage as the request line — zero-copy, just a refcount + range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedHeader {
    pub name: Bytes,
    pub value: Bytes,
}

/// Owned counterpart of [`RequestHead`]. Exists because `Pipe::Out`
/// cannot borrow from `Pipe::In` once `call` returns it by value — the
/// borrow crossing pays exactly one re-own step (all zero-copy slices
/// of the same input `Bytes`), never a fresh byte copy.
///
/// `headers` stays a plain `Vec`, NOT the `SmallVec` shape
/// [`crate::http1_codec::h1::HeaderVec`] uses on the short-lived
/// parser side: `OwnedHeader` is two `Bytes` (4 machine words each),
/// twice `Header`'s size, and `OwnedFrame`/`H1RequestFrame` are moved
/// by value all the way through the downstream pipe (`Pipe::Out`,
/// handler dispatch, `Replayable::fork`) rather than dying at the end
/// of one parse call — measured (`bench_http1_pipe_serve.rs`):
/// inlining headers here nearly tripled `OwnedFrame` (96 -> 344 bytes)
/// and regressed per-request latency 2-13% even though it also cut
/// one heap allocation, a net loss on this hot path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedFrame {
    pub method: Bytes,
    pub path: Bytes,
    pub version: HttpVersion,
    pub headers: Vec<OwnedHeader>,
}

impl OwnedFrame {
    fn from_head(source: &Bytes, head: &RequestHead<'_>) -> Self {
        let headers = head
            .headers
            .iter()
            .map(|header| OwnedHeader {
                name: source.slice_ref(header.name()),
                value: source.slice_ref(header.value()),
            })
            .collect();
        Self {
            method: source.slice_ref(head.method),
            path: source.slice_ref(head.path),
            version: head.version,
            headers,
        }
    }
}

/// The per-codec re-owning seam [`crate::codec_pipe::FrameCodecPipe`] is
/// generic over.
impl OwnFrame for H1RequestCodec {
    type Source = Bytes;
    type Owned = OwnedFrame;

    fn own_frame(source: &Bytes, frame: &RequestHead<'_>) -> OwnedFrame {
        OwnedFrame::from_head(source, frame)
    }
}

/// The per-codec "need more bytes" seam [`crate::codec_pipe::FrameCodecPipe`]
/// is generic over: only `Partial` means "not enough bytes yet", a hard
/// `Parse`/`EncodeOverrun` error stays an `Err`.
impl Incomplete for FrameError {
    fn is_incomplete(&self) -> bool {
        matches!(self, FrameError::Partial)
    }
}

// `Replayable::fork`/`replay` clone `OwnedFrame`: every field is either a
// `Bytes` (refcount bump) or a `Vec<OwnedHeader>` of `Bytes` pairs, so the
// clone is cheap (no fresh byte copy) even though it is not zero-alloc (the
// header `Vec` reallocates) — acceptable because `Retry`'s replay path is
// inherently a cold/error path, not the steady-state hot loop.
impl Replayable for OwnedFrame {
    type Source = OwnedFrame;

    fn fork(self, _replay_cap_bytes: usize) -> (OwnedFrame, OwnedFrame) {
        (self.clone(), self)
    }

    fn replay(source: &OwnedFrame) -> Result<OwnedFrame, ProximaError> {
        Ok(source.clone())
    }
}

// A parsed HTTP/1 request head is safe to re-feed to a downstream handler as
// many times as a retry policy asks — parsing has already happened; only the
// handler's OWN side effects (not modeled here) would make replay unsafe, and
// that is the handler's contract to enforce, not the frame's.
impl Idempotent for OwnedFrame {
    fn is_idempotent(&self) -> bool {
        true
    }
}

impl Labeled for OwnedFrame {
    fn telemetry(&self) -> TelemetryHandle {
        NoopTelemetry::handle()
    }

    fn labels(&self) -> Labels {
        Labels::empty()
    }
}

/// `OwnedFrame` plus its decoded body — the typed value a
/// bytes-in/bytes-out connection Pipe hands to the "http domain"
/// handler stage. `H1RequestCodec::parse_frame` (see `codec_trait`)
/// stops at the blank line terminating the head; this codec is the
/// seam that ALSO drains the framed body (Content-Length or chunked,
/// per RFC 7230 §3.3) via [`BodyDecoder`] — the SAME sans-IO body
/// decoder [`crate::http1_codec::h1_connection::Connection`] uses, not
/// a rewritten one — so `body` is never empty just because the codec
/// only knows how to read a head.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H1RequestFrame {
    pub head: OwnedFrame,
    pub body: Bytes,
}

/// Error surface for [`H1RequestFrameCodec`]: either stage (head parse,
/// framing derivation, body decode) can fail. `AmbiguousFraming` /
/// `BadContentLength` / `UnsupportedTransferEncoding` mirror
/// `h1_connection::ReadError`'s exact taxonomy (RFC 7230 §3.3.3
/// request-smuggling defense: a request MUST NOT declare both
/// `Content-Length` and `Transfer-Encoding: chunked`) — duplicated
/// here rather than reused because `ReadError` is computed from
/// `h1_connection`'s private offset table, not from a borrowed
/// [`RequestHead`]; the two call sites parse the SAME rule from
/// differently-shaped inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum H1RequestFrameError {
    Head(FrameError),
    AmbiguousFraming,
    BadContentLength,
    UnsupportedTransferEncoding,
    Body(DecodeError),
}

impl fmt::Display for H1RequestFrameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Head(error) => write!(formatter, "head: {error}"),
            Self::AmbiguousFraming => formatter.write_str(
                "ambiguous body framing: both content-length and transfer-encoding chunked present",
            ),
            Self::BadContentLength => formatter.write_str("invalid content-length header value"),
            Self::UnsupportedTransferEncoding => {
                formatter.write_str("unsupported transfer-encoding")
            }
            Self::Body(error) => write!(formatter, "body: {error}"),
        }
    }
}

impl core::error::Error for H1RequestFrameError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Head(error) => Some(error),
            Self::Body(error) => Some(error),
            Self::AmbiguousFraming | Self::BadContentLength | Self::UnsupportedTransferEncoding => {
                None
            }
        }
    }
}

/// Pick the request's body framing from its parsed headers, per RFC
/// 7230 §3.3 — the request-side counterpart of
/// `h1_client::framing_from_response`. `Transfer-Encoding: chunked`
/// and `Content-Length` both present is a hard error (request
/// smuggling defense), not "chunked wins": the incumbent's
/// `h1_connection::body_framing_from_offsets` rejects the same
/// ambiguity for the same reason.
fn framing_from_request(head: &RequestHead<'_>) -> Result<BodyFraming, H1RequestFrameError> {
    let mut content_length: Option<u64> = None;
    let mut chunked = false;
    let mut other_encoding = false;
    for header in &head.headers {
        if header.name().eq_ignore_ascii_case(b"content-length") {
            let text = core::str::from_utf8(header.value())
                .map_err(|_| H1RequestFrameError::BadContentLength)?
                .trim();
            let parsed: u64 = text
                .parse()
                .map_err(|_| H1RequestFrameError::BadContentLength)?;
            content_length = Some(parsed);
        } else if header.name().eq_ignore_ascii_case(b"transfer-encoding") {
            let text = core::str::from_utf8(header.value()).unwrap_or("");
            for token in text.split(',') {
                let trimmed = token.trim();
                if trimmed.eq_ignore_ascii_case("chunked") {
                    chunked = true;
                } else if !trimmed.is_empty() && !trimmed.eq_ignore_ascii_case("identity") {
                    other_encoding = true;
                }
            }
        }
    }
    if chunked && content_length.is_some() {
        return Err(H1RequestFrameError::AmbiguousFraming);
    }
    if other_encoding {
        return Err(H1RequestFrameError::UnsupportedTransferEncoding);
    }
    if chunked {
        Ok(BodyFraming::Chunked)
    } else if let Some(length) = content_length {
        Ok(BodyFraming::ContentLength(length))
    } else {
        Ok(BodyFraming::None)
    }
}

/// `Pipe<In = Bytes, Out = Option<(H1RequestFrame, usize)>, Err =
/// H1RequestFrameError>` — head parse (via [`H1RequestCodec`]) THEN
/// body decode (via [`BodyDecoder`]) over the SAME accumulator
/// snapshot, so `consumed` reflects head+body together. This is why
/// it is hand-written rather than plugged into the generic
/// `crate::codec_pipe::FrameCodecPipe<C>` adapter: that adapter's
/// `OwnFrame::own_frame` seam only ever sees the codec's OWN
/// `consumed` (head-only for `H1RequestCodec`), with no way to extend
/// the frame boundary past it.
#[derive(Debug, Clone, Copy, Default)]
pub struct H1RequestFrameCodec {
    inner: H1RequestCodec,
    decoder_limits: DecoderLimits,
}

impl H1RequestFrameCodec {
    #[must_use]
    pub const fn new(decoder_limits: DecoderLimits) -> Self {
        Self {
            inner: H1RequestCodec,
            decoder_limits,
        }
    }

    fn decode(
        &self,
        input: &Bytes,
    ) -> Result<Option<(H1RequestFrame, usize)>, H1RequestFrameError> {
        let (head, head_consumed) = match self.inner.parse_frame(input) {
            Ok(parsed) => parsed,
            Err(error) if error.is_incomplete() => return Ok(None),
            Err(error) => return Err(H1RequestFrameError::Head(error)),
        };
        let framing = framing_from_request(&head)?;
        let body_window = &input[head_consumed..];
        let mut decoder = BodyDecoder::with_limits(framing, self.decoder_limits);
        let (body, body_consumed, status) = match framing {
            // `ContentLengthRemaining`'s `feed` arm sinks the whole
            // remainder in one call and returns immediately, so the
            // body is always a single contiguous span here — re-own
            // it via `slice_ref` (an `Arc` refcount bump, same trick
            // `OwnedFrame::from_head` uses for headers) instead of
            // copying into a fresh `Vec`.
            BodyFraming::None | BodyFraming::ContentLength(_) => {
                let mut span: Option<Bytes> = None;
                let (consumed, status) = decoder
                    .feed(body_window, |chunk| span = Some(input.slice_ref(chunk)))
                    .map_err(H1RequestFrameError::Body)?;
                (span.unwrap_or_default(), consumed, status)
            }
            // chunked reassembly discards the framing bytes between
            // chunks, so the decoded body is not a contiguous
            // sub-slice of the input — the copy here is inherent, not
            // incidental.
            BodyFraming::Chunked => {
                let mut body_buffer: Vec<u8> = Vec::new();
                let (consumed, status) = decoder
                    .feed(body_window, |chunk| body_buffer.extend_from_slice(chunk))
                    .map_err(H1RequestFrameError::Body)?;
                (Bytes::from(body_buffer), consumed, status)
            }
        };
        if matches!(status, BodyStatus::NeedMore) {
            // partial body — the head reparses too on the next call,
            // matching `FrameCodecPipe`'s own "None = wait for more
            // bytes" contract (see that type's doc).
            return Ok(None);
        }
        let owned_head = OwnedFrame::from_head(input, &head);
        let frame = H1RequestFrame {
            head: owned_head,
            body,
        };
        Ok(Some((frame, head_consumed + body_consumed)))
    }
}

impl Pipe for H1RequestFrameCodec {
    type In = Bytes;
    type Out = Option<(H1RequestFrame, usize)>;
    type Err = H1RequestFrameError;

    fn call(&self, input: Bytes) -> impl Future<Output = Result<Self::Out, Self::Err>> {
        async move { self.decode(&input) }
    }
}

impl SendPipe for H1RequestFrameCodec {
    type In = Bytes;
    type Out = Option<(H1RequestFrame, usize)>;
    type Err = H1RequestFrameError;

    fn call(&self, input: Bytes) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move { self.decode(&input) }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::codec_pipe::FrameCodecPipe;
    use core::future::Future;
    use proxima_primitives::pipe::Pipe;

    /// Dependency-free executor for the always-ready probe futures — keeps
    /// this leaf module no_std-pure (mirrors `primitives.rs`'s own `block_on`
    /// test helper rather than pulling `futures::executor` into the
    /// no_std + alloc tier just for tests). `FrameCodecPipe::call` is
    /// synchronous logic wrapped in `async move`, so it resolves on the
    /// first poll; a spin loop is sufficient.
    fn block_on<Fut: Future>(future: Fut) -> Fut::Output {
        let mut pinned = core::pin::pin!(future);
        let mut context = core::task::Context::from_waker(core::task::Waker::noop());
        loop {
            if let core::task::Poll::Ready(output) = pinned.as_mut().poll(&mut context) {
                return output;
            }
        }
    }

    // real HTTP/1.1 request bytes (P9): the exact wire shape a browser or
    // curl sends, not a synthetic fixture.
    const SIMPLE_GET: &[u8] =
        b"GET /v1/messages HTTP/1.1\r\nHost: api.example.com\r\nContent-Length: 0\r\n\r\n";

    #[test]
    fn complete_frame_returns_owned_frame_and_consumed() {
        let codec: FrameCodecPipe<H1RequestCodec> = FrameCodecPipe::default();
        let input = Bytes::copy_from_slice(SIMPLE_GET);
        let outcome = block_on(Pipe::call(&codec, input.clone())).expect("real GET request parses");
        let (frame, consumed) = outcome.expect("complete head, not partial");
        assert_eq!(&frame.method[..], b"GET");
        assert_eq!(&frame.path[..], b"/v1/messages");
        assert_eq!(frame.version, HttpVersion::Http11);
        assert_eq!(consumed, input.len());
        assert_eq!(frame.headers.len(), 2);
        assert_eq!(&frame.headers[0].name[..], b"Host");
        assert_eq!(&frame.headers[0].value[..], b"api.example.com");
    }

    #[test]
    fn owned_frame_slices_share_the_input_bytes_allocation() {
        let codec: FrameCodecPipe<H1RequestCodec> = FrameCodecPipe::default();
        let input = Bytes::copy_from_slice(SIMPLE_GET);
        let outcome = block_on(Pipe::call(&codec, input.clone()))
            .expect("parse")
            .expect("complete");
        // zero-copy claim: the owned method slice points INTO the same
        // backing allocation as the input, not a fresh copy.
        assert_eq!(outcome.0.method.as_ptr(), input[0..3].as_ptr());
    }

    #[test]
    fn partial_head_returns_none_not_error() {
        let codec: FrameCodecPipe<H1RequestCodec> = FrameCodecPipe::default();
        let truncated = Bytes::copy_from_slice(&SIMPLE_GET[..SIMPLE_GET.len() - 4]);
        let outcome =
            block_on(Pipe::call(&codec, truncated)).expect("partial is Ok(None), not Err");
        assert!(outcome.is_none());
    }

    #[test]
    fn malformed_head_returns_frame_error() {
        let codec: FrameCodecPipe<H1RequestCodec> = FrameCodecPipe::default();
        let bad = Bytes::copy_from_slice(b"not even a request line\r\n\r\n");
        let outcome = block_on(Pipe::call(&codec, bad));
        assert!(matches!(outcome, Err(FrameError::Parse(_))));
    }

    #[test]
    fn owned_frame_replays_to_an_equal_clone() {
        let codec: FrameCodecPipe<H1RequestCodec> = FrameCodecPipe::default();
        let input = Bytes::copy_from_slice(SIMPLE_GET);
        let (frame, _) = block_on(Pipe::call(&codec, input))
            .expect("parse")
            .expect("complete");
        let (first, source) = Replayable::fork(frame.clone(), 0);
        assert_eq!(first, frame);
        let replayed = OwnedFrame::replay(&source).expect("replay");
        assert_eq!(replayed, frame);
        assert!(frame.is_idempotent());
    }

    // ── H1RequestFrameCodec: head + body threaded together ──────────

    #[test]
    fn bodyless_get_yields_empty_body() {
        let codec = H1RequestFrameCodec::default();
        let input = Bytes::copy_from_slice(SIMPLE_GET);
        let (frame, consumed) = block_on(Pipe::call(&codec, input.clone()))
            .expect("parses")
            .expect("complete, not partial");
        assert_eq!(&frame.head.method[..], b"GET");
        assert!(frame.body.is_empty());
        assert_eq!(consumed, input.len());
    }

    #[test]
    fn content_length_body_is_threaded_into_the_frame() {
        let codec = H1RequestFrameCodec::default();
        let raw = b"POST /echo HTTP/1.1\r\nHost: x\r\nContent-Length: 11\r\n\r\nhello world";
        let input = Bytes::copy_from_slice(raw);
        let (frame, consumed) = block_on(Pipe::call(&codec, input.clone()))
            .expect("parses")
            .expect("complete body, not partial");
        assert_eq!(&frame.body[..], b"hello world");
        assert_eq!(consumed, input.len());
    }

    #[test]
    fn chunked_body_is_dechunked_into_the_frame() {
        let codec = H1RequestFrameCodec::default();
        let raw = b"POST /echo HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n";
        let input = Bytes::copy_from_slice(raw);
        let (frame, consumed) = block_on(Pipe::call(&codec, input.clone()))
            .expect("parses")
            .expect("complete body, not partial");
        assert_eq!(&frame.body[..], b"hello world");
        assert_eq!(consumed, input.len());
    }

    #[test]
    fn partial_body_returns_none_not_error() {
        let codec = H1RequestFrameCodec::default();
        // head is complete but only 5 of the declared 11 body bytes arrived.
        let raw = b"POST /echo HTTP/1.1\r\nHost: x\r\nContent-Length: 11\r\n\r\nhello";
        let input = Bytes::copy_from_slice(raw);
        let outcome = block_on(Pipe::call(&codec, input)).expect("partial is Ok(None), not Err");
        assert!(outcome.is_none());
    }

    #[test]
    fn ambiguous_framing_is_rejected() {
        let codec = H1RequestFrameCodec::default();
        let raw = b"POST /echo HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\nTransfer-Encoding: chunked\r\n\r\nhello";
        let input = Bytes::copy_from_slice(raw);
        let outcome = block_on(Pipe::call(&codec, input));
        assert_eq!(outcome, Err(H1RequestFrameError::AmbiguousFraming));
    }
}
