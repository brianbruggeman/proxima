//! `proxima_codec::FrameCodec` impl for HTTP/1.x request heads.
//!
//! Gated behind the `codec-trait` feature so the no_std + alloc cliff
//! stays clean: the proxima-codec dependency is the only thing this
//! module imports, and a downstream that only wants the existing
//! `parse_head` / `BodyDecoder` surface gets none of it.
//!
//! Frame shape: a [`RequestHead<'a>`] (borrowed from the input buffer).
//! `parse_frame` returns the head + bytes consumed; an in-progress
//! parse surfaces as [`FrameError::Partial`] so the caller can request
//! more bytes. `encode_frame` writes the request line and header block
//! back to the wire, terminated by the canonical CRLF + empty CRLF.

use alloc::vec::Vec;
use core::fmt;

use proxima_codec::FrameCodec;

use crate::http1_codec::h1::{HttpVersion, ParseError, RequestHead, Status, parse_head};

/// HTTP/1.x request-head [`FrameCodec`]. Zero-sized; clone freely.
#[derive(Debug, Clone, Copy, Default)]
pub struct H1RequestCodec;

/// Error surface for [`H1RequestCodec`]. Wraps the existing
/// [`ParseError`] taxonomy plus a `Partial` marker so the [`FrameCodec`]
/// `Result` shape can express "need more bytes" without conflating it
/// with a hard parse error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameError {
    /// Buffer did not contain a complete head; caller should read more
    /// bytes and retry.
    Partial,
    /// Hard parse failure — the input is malformed.
    Parse(ParseError),
    /// Encoder buffer overrun — currently unused because `encode_frame`
    /// extends `dest` rather than writing into a fixed-size slice.
    /// Reserved so adding a fixed-buffer variant later does not require
    /// a breaking enum extension.
    EncodeOverrun,
}

impl fmt::Display for FrameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Partial => formatter.write_str("partial frame: need more bytes"),
            Self::Parse(inner) => write!(formatter, "parse: {inner}"),
            Self::EncodeOverrun => formatter.write_str("encode buffer overrun"),
        }
    }
}

impl core::error::Error for FrameError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Parse(inner) => Some(inner),
            Self::Partial | Self::EncodeOverrun => None,
        }
    }
}

impl From<ParseError> for FrameError {
    fn from(error: ParseError) -> Self {
        Self::Parse(error)
    }
}

impl FrameCodec for H1RequestCodec {
    type Frame<'a> = RequestHead<'a>;
    type Error = FrameError;

    fn parse_frame<'a>(&self, buf: &'a [u8]) -> Result<(Self::Frame<'a>, usize), Self::Error> {
        match parse_head(buf)? {
            Status::Complete { head, consumed } => Ok((head, consumed)),
            Status::Partial => Err(FrameError::Partial),
        }
    }

    fn encode_frame(&self, frame: &Self::Frame<'_>, dest: &mut Vec<u8>) -> Result<(), Self::Error> {
        dest.extend_from_slice(frame.method);
        dest.push(b' ');
        dest.extend_from_slice(frame.path);
        dest.push(b' ');
        dest.extend_from_slice(version_bytes(frame.version));
        dest.extend_from_slice(b"\r\n");
        for header in &frame.headers {
            dest.extend_from_slice(header.name());
            dest.extend_from_slice(b": ");
            dest.extend_from_slice(header.value());
            dest.extend_from_slice(b"\r\n");
        }
        dest.extend_from_slice(b"\r\n");
        Ok(())
    }
}

fn version_bytes(version: HttpVersion) -> &'static [u8] {
    match version {
        HttpVersion::Http10 => b"HTTP/1.0",
        HttpVersion::Http11 => b"HTTP/1.1",
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_codec::FrameCodec;

    const SIMPLE_GET: &[u8] =
        b"GET /v1/messages HTTP/1.1\r\nHost: api.example.com\r\nContent-Length: 0\r\n\r\n";

    #[test]
    fn parse_frame_complete_returns_head_and_consumed() {
        let codec = H1RequestCodec;
        let (head, consumed) = codec.parse_frame(SIMPLE_GET).expect("complete head");
        assert_eq!(head.method, b"GET");
        assert_eq!(head.path, b"/v1/messages");
        assert_eq!(head.version, HttpVersion::Http11);
        assert_eq!(consumed, SIMPLE_GET.len());
        assert_eq!(head.headers.len(), 2);
    }

    #[test]
    fn parse_frame_partial_returns_partial_error() {
        let codec = H1RequestCodec;
        // truncate before the empty CRLF that signals end-of-head
        let truncated = &SIMPLE_GET[..SIMPLE_GET.len() - 4];
        let outcome = codec.parse_frame(truncated);
        assert_eq!(outcome.err(), Some(FrameError::Partial));
    }

    #[test]
    fn parse_frame_malformed_returns_parse_error() {
        let codec = H1RequestCodec;
        let bad = b"not even a request line\r\n\r\n";
        let outcome = codec.parse_frame(bad);
        assert!(matches!(outcome, Err(FrameError::Parse(_))));
    }

    #[test]
    fn encode_frame_round_trips_through_parse() {
        let codec = H1RequestCodec;
        let (head, _) = codec.parse_frame(SIMPLE_GET).expect("parse");
        let mut encoded = Vec::with_capacity(SIMPLE_GET.len());
        codec.encode_frame(&head, &mut encoded).expect("encode");
        let (round_tripped, _) = codec.parse_frame(&encoded).expect("re-parse");
        assert_eq!(round_tripped.method, head.method);
        assert_eq!(round_tripped.path, head.path);
        assert_eq!(round_tripped.version, head.version);
        assert_eq!(round_tripped.headers.len(), head.headers.len());
        for (lhs, rhs) in round_tripped.headers.iter().zip(head.headers.iter()) {
            assert_eq!(lhs.name(), rhs.name());
            assert_eq!(lhs.value(), rhs.value());
        }
    }

    #[test]
    fn encode_frame_terminates_with_double_crlf() {
        let codec = H1RequestCodec;
        let (head, _) = codec.parse_frame(SIMPLE_GET).expect("parse");
        let mut encoded = Vec::new();
        codec.encode_frame(&head, &mut encoded).expect("encode");
        assert!(encoded.ends_with(b"\r\n\r\n"));
    }

    #[test]
    fn frame_error_partial_displays_human_readable_message() {
        let displayed = format!("{}", FrameError::Partial);
        assert!(displayed.contains("partial"));
    }
}
