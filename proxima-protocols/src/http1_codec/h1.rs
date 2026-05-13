//! HTTP/1.1 request head parsing — delegates to `httparse`.
//!
//! Rationale: `httparse` is the canonical Rust HTTP/1 head parser
//! (used by `hyper`, `reqwest`, `actix`, etc.). It's SIMD-validated
//! on x86/ARM, SWAR-fallback otherwise, allocation-free, and has
//! seen years of fuzzing + security audits in production. Building
//! our own would be reinventing the wheel.
//!
//! What this module provides:
//! - Zero-copy `RequestHead<'a>` / `Header<'a>` types borrowed into
//!   the caller's buffer.
//! - `parse_head_streaming(buffer, limits, on_header)` — surfaces
//!   each header through a callback so the caller (`Connection`)
//!   pushes offsets straight into pre-allocated storage. No Vec is
//!   allocated inside the parser.
//! - `HttpVersion` enum used by the response writer.
//!
//! Body decoding, response writing, and the connection state machine
//! live in sibling modules (`h1_body`, `h1_response`, `h1_connection`)
//! because httparse covers parsing only. Those modules ARE the
//! substrate's own work — they implement state machines httparse
//! doesn't.

use core::fmt;

#[cfg(not(feature = "http1_codec-no-alloc"))]
use smallvec::SmallVec;

#[cfg(not(feature = "http1_codec-no-alloc"))]
use crate::http1_codec::sized;

const MAX_HEADERS: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HttpVersion {
    Http10,
    #[default]
    Http11,
}

impl HttpVersion {
    fn from_httparse(version: Option<u8>) -> Result<Self, ParseError> {
        match version {
            Some(0) => Ok(Self::Http10),
            Some(1) => Ok(Self::Http11),
            _ => Err(ParseError::BadVersion),
        }
    }
}

/// One header as a pair of slices borrowed from the input buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header<'a> {
    name: &'a [u8],
    value: &'a [u8],
}

impl<'a> Header<'a> {
    /// Construct a Header from two slices borrowing into the input
    /// buffer. Used by `Connection::head` to rebuild a typed handle
    /// from its cached `(start, end)` offsets.
    #[must_use]
    pub fn new(name: &'a [u8], value: &'a [u8]) -> Self {
        Self { name, value }
    }

    #[must_use]
    pub fn name(&self) -> &'a [u8] {
        self.name
    }

    #[must_use]
    pub fn value(&self) -> &'a [u8] {
        self.value
    }

    #[must_use]
    pub fn name_str(&self) -> &'a str {
        core::str::from_utf8(self.name).unwrap_or("")
    }

    #[must_use]
    pub fn value_str(&self) -> Option<&'a str> {
        core::str::from_utf8(self.value).ok()
    }
}

/// Header container backing [`RequestHead::headers`]: inline storage
/// for up to [`sized::HEADER_INLINE_CAP`] headers, spilling to the heap
/// (same growth as a plain `Vec`, and no worse) only past that cap — a
/// request with that many headers or fewer (e.g. a bodyless health
/// check or an internal RPC-over-HTTP1 call) parses with zero heap
/// allocation for the container itself (the name/value byte slices
/// were already zero-copy). The cap is deliberately small: each inline
/// slot embeds a `Header<'a>` (two fat pointers) directly in
/// `RequestHead`'s own bytes, so a larger cap would grow `RequestHead`
/// past clippy's `large_enum_variant` threshold for the
/// `Status::{Partial,Complete}` enum it lives in — see
/// `http1_codec.toml`.
#[cfg(not(feature = "http1_codec-no-alloc"))]
pub type HeaderVec<'a> = SmallVec<[Header<'a>; sized::HEADER_INLINE_CAP]>;

/// Typed request head. Lives as long as the input buffer the parser
/// was called with. `headers` is the only allocation, and only past
/// [`sized::HEADER_INLINE_CAP`] headers — callers on the hot path
/// should use `parse_head_streaming` to surface headers via callback
/// into pre-allocated storage instead.
#[cfg(not(feature = "http1_codec-no-alloc"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestHead<'a> {
    pub method: &'a [u8],
    pub path: &'a [u8],
    pub version: HttpVersion,
    pub headers: HeaderVec<'a>,
}

#[cfg(not(feature = "http1_codec-no-alloc"))]
impl<'a> RequestHead<'a> {
    /// Case-insensitive header lookup per RFC 7230 §3.2.
    #[must_use]
    pub fn header(&self, name: &str) -> Option<&Header<'a>> {
        let lower = name.as_bytes();
        self.headers
            .iter()
            .find(|header| eq_ignore_ascii_case(header.name, lower))
    }
}

#[cfg(not(feature = "http1_codec-no-alloc"))]
fn eq_ignore_ascii_case(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right.iter())
        .all(|(left_byte, right_byte)| left_byte.eq_ignore_ascii_case(right_byte))
}

#[cfg(not(feature = "http1_codec-no-alloc"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status<'a> {
    Partial,
    Complete {
        head: RequestHead<'a>,
        consumed: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingStatus<'a> {
    Partial,
    Complete {
        method: &'a [u8],
        path: &'a [u8],
        version: HttpVersion,
        consumed: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    BadVersion,
    InvalidHeaderName,
    InvalidHeaderValue,
    MalformedRequestLine,
    BadLineEnding,
    MethodTooLong,
    PathTooLong,
    HeaderLineTooLong,
    TooManyHeaders,
}

impl fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadVersion => write!(formatter, "unsupported HTTP version"),
            Self::InvalidHeaderName => write!(formatter, "invalid header name"),
            Self::InvalidHeaderValue => write!(formatter, "invalid header value"),
            Self::MalformedRequestLine => write!(formatter, "malformed request line"),
            Self::BadLineEnding => write!(formatter, "CR not followed by LF"),
            Self::MethodTooLong => write!(formatter, "method exceeds budget"),
            Self::PathTooLong => write!(formatter, "request-target exceeds budget"),
            Self::HeaderLineTooLong => write!(formatter, "header line exceeds budget"),
            Self::TooManyHeaders => write!(formatter, "too many headers"),
        }
    }
}

impl core::error::Error for ParseError {}

fn map_httparse_error(error: httparse::Error) -> ParseError {
    match error {
        httparse::Error::HeaderName => ParseError::InvalidHeaderName,
        httparse::Error::HeaderValue => ParseError::InvalidHeaderValue,
        httparse::Error::NewLine => ParseError::BadLineEnding,
        httparse::Error::Status => ParseError::MalformedRequestLine,
        httparse::Error::Token => ParseError::MalformedRequestLine,
        httparse::Error::TooManyHeaders => ParseError::TooManyHeaders,
        httparse::Error::Version => ParseError::BadVersion,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ParserLimits {
    pub max_method_bytes: usize,
    pub max_path_bytes: usize,
    pub max_header_line_bytes: usize,
    pub max_headers: usize,
}

impl Default for ParserLimits {
    fn default() -> Self {
        Self {
            max_method_bytes: 16,
            max_path_bytes: 8192,
            max_header_line_bytes: 8192,
            max_headers: 100,
        }
    }
}

/// Parse a request head from `buffer`. Returns `Partial` if incomplete,
/// or `Complete` with a typed `RequestHead` + the body offset.
///
/// Allocates only past `sized::HEADER_INLINE_CAP` headers (a `HeaderVec`
/// spill). Hot-path callers should use `parse_head_streaming` instead —
/// it routes headers through a callback into pre-allocated storage.
#[cfg(not(feature = "http1_codec-no-alloc"))]
pub fn parse_head(buffer: &[u8]) -> Result<Status<'_>, ParseError> {
    parse_head_with_limits(buffer, ParserLimits::default())
}

#[cfg(not(feature = "http1_codec-no-alloc"))]
pub fn parse_head_with_limits(
    buffer: &[u8],
    limits: ParserLimits,
) -> Result<Status<'_>, ParseError> {
    let mut headers = [httparse::EMPTY_HEADER; MAX_HEADERS];
    let cap = limits.max_headers.min(MAX_HEADERS);
    let mut request = httparse::Request::new(&mut headers[..cap]);
    let consumed = match request.parse(buffer).map_err(map_httparse_error)? {
        httparse::Status::Partial => return Ok(Status::Partial),
        httparse::Status::Complete(consumed) => consumed,
    };
    check_limits(&request, buffer, consumed, &limits)?;
    let method = request
        .method
        .ok_or(ParseError::MalformedRequestLine)?
        .as_bytes();
    let path = request
        .path
        .ok_or(ParseError::MalformedRequestLine)?
        .as_bytes();
    let version = HttpVersion::from_httparse(request.version)?;
    let head_headers: HeaderVec<'_> = request
        .headers
        .iter()
        .map(|header| Header::new(header.name.as_bytes(), header.value))
        .collect();
    Ok(Status::Complete {
        head: RequestHead {
            method,
            path,
            version,
            headers: head_headers,
        },
        consumed,
    })
}

/// Hot-path parser. Surfaces each header through `on_header` instead
/// of collecting into a Vec. Returns the request-line summary +
/// `consumed`. Zero allocations per call: the parser scratch is a
/// stack `[EMPTY_HEADER; MAX_HEADERS]` array.
pub fn parse_head_streaming<'a, F>(
    buffer: &'a [u8],
    limits: ParserLimits,
    mut on_header: F,
) -> Result<StreamingStatus<'a>, ParseError>
where
    F: FnMut(Header<'a>),
{
    let mut headers = [httparse::EMPTY_HEADER; MAX_HEADERS];
    let cap = limits.max_headers.min(MAX_HEADERS);
    let mut request = httparse::Request::new(&mut headers[..cap]);
    let consumed = match request.parse(buffer).map_err(map_httparse_error)? {
        httparse::Status::Partial => return Ok(StreamingStatus::Partial),
        httparse::Status::Complete(consumed) => consumed,
    };
    check_limits(&request, buffer, consumed, &limits)?;
    let method_str = request.method.ok_or(ParseError::MalformedRequestLine)?;
    let path_str = request.path.ok_or(ParseError::MalformedRequestLine)?;
    let version = HttpVersion::from_httparse(request.version)?;
    // `request.method` / `request.path` are `&str` slices into the
    // input buffer; convert to `&[u8]` that borrow the same memory.
    let method: &'a [u8] = byte_slice_from_str(method_str, buffer);
    let path: &'a [u8] = byte_slice_from_str(path_str, buffer);
    for header in request.headers.iter() {
        on_header(Header::new(
            byte_slice_from_str(header.name, buffer),
            slice_in_buffer(header.value, buffer),
        ));
    }
    Ok(StreamingStatus::Complete {
        method,
        path,
        version,
        consumed,
    })
}

/// Reattach a `&str` returned by httparse to the input buffer's
/// lifetime. httparse hands back `&str` borrowing into `buffer` —
/// we re-borrow at the buffer's lifetime so the result outlives
/// the httparse `Request` (which itself borrows the same buffer).
fn byte_slice_from_str<'a>(borrowed: &str, buffer: &'a [u8]) -> &'a [u8] {
    let start = borrowed.as_ptr() as usize - buffer.as_ptr() as usize;
    &buffer[start..start + borrowed.len()]
}

fn slice_in_buffer<'a>(borrowed: &[u8], buffer: &'a [u8]) -> &'a [u8] {
    let start = borrowed.as_ptr() as usize - buffer.as_ptr() as usize;
    &buffer[start..start + borrowed.len()]
}

fn check_limits(
    request: &httparse::Request<'_, '_>,
    buffer: &[u8],
    consumed: usize,
    limits: &ParserLimits,
) -> Result<(), ParseError> {
    if let Some(method) = request.method
        && method.len() > limits.max_method_bytes
    {
        return Err(ParseError::MethodTooLong);
    }
    if let Some(path) = request.path
        && path.len() > limits.max_path_bytes
    {
        return Err(ParseError::PathTooLong);
    }
    // Per-header-line budget: longest single name + value pair must
    // fit within max_header_line_bytes. The `name: value\r\n` overhead
    // (4 chars) is included so the budget reflects the on-wire size.
    for header in request.headers.iter() {
        if header.name.len() + header.value.len() + 4 > limits.max_header_line_bytes {
            return Err(ParseError::HeaderLineTooLong);
        }
    }
    // Cap by the buffer slice we actually saw — defensive.
    let _ = (buffer, consumed);
    Ok(())
}

#[cfg(all(test, not(feature = "http1_codec-no-alloc")))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn complete(input: &[u8]) -> (RequestHead<'_>, usize) {
        match parse_head(input).expect("parse") {
            Status::Complete { head, consumed } => (head, consumed),
            Status::Partial => panic!("expected Complete, got Partial"),
        }
    }

    #[test]
    fn simple_get_borrows_method_and_path_into_input_buffer() {
        let input: &[u8] = b"GET /hello HTTP/1.1\r\n\r\n";
        let (head, consumed) = complete(input);
        assert_eq!(head.method, b"GET");
        assert_eq!(head.path, b"/hello");
        assert_eq!(head.version, HttpVersion::Http11);
        assert!(head.headers.is_empty());
        assert_eq!(consumed, input.len());
        assert_eq!(head.method.as_ptr(), input[..3].as_ptr());
        assert_eq!(head.path.as_ptr(), input[4..10].as_ptr());
    }

    #[test]
    fn header_slices_point_into_the_input_buffer() {
        let input: &[u8] = b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n";
        let (head, _consumed) = complete(input);
        assert_eq!(head.headers.len(), 1);
        let header = &head.headers[0];
        assert_eq!(header.name(), b"Host");
        assert_eq!(header.value(), b"example.com");
        let host_start = input
            .windows(b"Host".len())
            .position(|window| window == b"Host")
            .expect("found");
        assert_eq!(header.name().as_ptr(), input[host_start..].as_ptr());
    }

    #[test]
    fn case_insensitive_header_lookup() {
        let input: &[u8] = b"GET / HTTP/1.1\r\nContent-Length: 5\r\n\r\n";
        let (head, _consumed) = complete(input);
        let found = head.header("content-length").expect("found");
        assert_eq!(found.value(), b"5");
        let alt = head.header("CONTENT-LENGTH").expect("found alt-case");
        assert_eq!(alt.value(), b"5");
    }

    #[test]
    fn partial_input_returns_partial_status_with_no_error() {
        let input: &[u8] = b"GET /hello HT";
        let status = parse_head(input).expect("parse");
        assert!(matches!(status, Status::Partial));
    }

    #[test]
    fn body_bytes_after_consumed_marker_are_left_untouched() {
        let input: &[u8] = b"GET / HTTP/1.1\r\n\r\nBODYBYTES";
        let (_head, consumed) = complete(input);
        assert_eq!(&input[consumed..], b"BODYBYTES");
    }

    #[rstest]
    #[case::bad_version(b"GET / HTTP/2.0\r\n\r\n", ParseError::BadVersion)]
    #[case::invalid_method_byte(b"G\x00ET / HTTP/1.1\r\n\r\n", ParseError::MalformedRequestLine)]
    #[case::header_name_with_control(
        b"GET / HTTP/1.1\r\nBad\x00Name: x\r\n\r\n",
        ParseError::InvalidHeaderName
    )]
    fn malformed_input_returns_typed_error(#[case] input: &[u8], #[case] expected: ParseError) {
        let outcome = parse_head(input);
        assert_eq!(outcome, Err(expected));
    }

    #[test]
    fn method_longer_than_budget_is_rejected() {
        let outcome = parse_head_with_limits(
            b"OPTIONSO / HTTP/1.1\r\n\r\n",
            ParserLimits {
                max_method_bytes: 4,
                ..ParserLimits::default()
            },
        );
        assert_eq!(outcome, Err(ParseError::MethodTooLong));
    }

    #[test]
    fn header_count_above_limit_rejected() {
        let outcome = parse_head_with_limits(
            b"GET / HTTP/1.1\r\nA: 1\r\nB: 2\r\nC: 3\r\n\r\n",
            ParserLimits {
                max_headers: 2,
                ..ParserLimits::default()
            },
        );
        assert_eq!(outcome, Err(ParseError::TooManyHeaders));
    }
}

// split from `tests`: exercises only the tier-3-safe streaming parser,
// so it stays compiled when `no-alloc` gates the owned-head tests out
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod streaming_tests {
    use super::*;

    #[test]
    fn streaming_variant_surfaces_each_header_via_callback() {
        let input: &[u8] = b"POST /v1 HTTP/1.1\r\nHost: x\r\nContent-Length: 3\r\n\r\nabc";
        // fixed-size, no heap: this module must stay no-alloc-tier safe
        let mut headers: [(&[u8], &[u8]); 2] = [(b"", b""), (b"", b"")];
        let mut header_count = 0usize;
        let status = parse_head_streaming(input, ParserLimits::default(), |header| {
            headers[header_count] = (header.name(), header.value());
            header_count += 1;
        })
        .expect("parse");
        match status {
            StreamingStatus::Complete {
                method,
                path,
                version,
                consumed,
            } => {
                assert_eq!(method, b"POST");
                assert_eq!(path, b"/v1");
                assert_eq!(version, HttpVersion::Http11);
                assert_eq!(consumed, input.len() - 3);
                assert_eq!(header_count, 2);
                assert_eq!(headers[0], (b"Host".as_slice(), b"x".as_slice()));
                assert_eq!(headers[1], (b"Content-Length".as_slice(), b"3".as_slice()));
            }
            StreamingStatus::Partial => panic!("expected Complete"),
        }
    }
}
