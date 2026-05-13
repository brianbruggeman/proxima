//! Client-side HTTP/1.1 head codec — the inverse of [`h1`] +
//! [`h1_response`]. Encodes a REQUEST head (the client writes) and
//! parses a RESPONSE head (the client reads), delegating the parse to
//! `httparse::Response` exactly as [`h1::parse_head`] delegates to
//! `httparse::Request`.
//!
//! Body framing for responses is decoded by reusing
//! [`h1_body::BodyDecoder`]; [`framing_from_response`] picks the
//! framing from a parsed [`ResponseHead`]'s Content-Length /
//! Transfer-Encoding headers per RFC 7230 §3.3.

use alloc::vec::Vec;

use crate::http1_codec::h1::{Header, HttpVersion, ParseError, ParserLimits};
use crate::http1_codec::h1_body::BodyFraming;

const MAX_HEADERS: usize = 100;

/// Serialize a request head: `METHOD PATH HTTP/1.1\r\nName: value\r\n...\r\n\r\n`.
///
/// Mirrors [`h1_response::write_response_head`] in style — appends into
/// the caller's `out` buffer, no allocation of its own. Headers are
/// written in the order given so the caller controls precedence
/// (Host, Content-Length, Connection, etc.).
pub fn encode_request_head<N, V>(method: &str, path: &str, headers: &[(N, V)], out: &mut Vec<u8>)
where
    N: AsRef<[u8]>,
    V: AsRef<[u8]>,
{
    out.extend_from_slice(method.as_bytes());
    out.push(b' ');
    out.extend_from_slice(path.as_bytes());
    out.extend_from_slice(b" HTTP/1.1\r\n");
    for (name, value) in headers {
        out.extend_from_slice(name.as_ref());
        out.extend_from_slice(b": ");
        out.extend_from_slice(value.as_ref());
        out.extend_from_slice(b"\r\n");
    }
    out.extend_from_slice(b"\r\n");
}

/// Typed response head, borrowing into the input buffer the parser was
/// called with. Symmetric to [`h1::RequestHead`] for the response
/// direction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponseHead<'a> {
    pub status: u16,
    pub version: HttpVersion,
    pub reason: &'a str,
    pub headers: Vec<Header<'a>>,
}

impl<'a> ResponseHead<'a> {
    /// Case-insensitive header lookup per RFC 7230 §3.2.
    #[must_use]
    pub fn header(&self, name: &str) -> Option<&Header<'a>> {
        let lower = name.as_bytes();
        self.headers
            .iter()
            .find(|header| eq_ignore_ascii_case(header.name(), lower))
    }
}

fn eq_ignore_ascii_case(left: &[u8], right: &[u8]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(left_byte, right_byte)| left_byte.eq_ignore_ascii_case(right_byte))
}

/// Parse result for a response head. `Complete` carries the parsed head
/// plus the byte offset where the body begins (mirrors the request-side
/// `Status::Complete { head, consumed }`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResponseStatus<'a> {
    Partial,
    Complete {
        head: ResponseHead<'a>,
        body_offset: usize,
    },
}

fn map_httparse_error(error: httparse::Error) -> ParseError {
    match error {
        httparse::Error::HeaderName => ParseError::InvalidHeaderName,
        httparse::Error::HeaderValue => ParseError::InvalidHeaderValue,
        httparse::Error::NewLine => ParseError::BadLineEnding,
        httparse::Error::Status | httparse::Error::Token => ParseError::MalformedRequestLine,
        httparse::Error::TooManyHeaders => ParseError::TooManyHeaders,
        httparse::Error::Version => ParseError::BadVersion,
    }
}

fn version_from_httparse(version: Option<u8>) -> Result<HttpVersion, ParseError> {
    match version {
        Some(0) => Ok(HttpVersion::Http10),
        Some(1) => Ok(HttpVersion::Http11),
        _ => Err(ParseError::BadVersion),
    }
}

/// Parse a response head from `buffer`. Returns `Partial` if the head
/// is incomplete, or `Complete` with a borrowed [`ResponseHead`] and
/// the body byte-offset. Wraps `httparse::Response::parse` exactly like
/// [`h1::parse_head`] wraps `httparse::Request`.
pub fn parse_response_head(buffer: &[u8]) -> Result<ResponseStatus<'_>, ParseError> {
    parse_response_head_with_limits(buffer, ParserLimits::default())
}

pub fn parse_response_head_with_limits(
    buffer: &[u8],
    limits: ParserLimits,
) -> Result<ResponseStatus<'_>, ParseError> {
    let mut headers = [httparse::EMPTY_HEADER; MAX_HEADERS];
    let cap = limits.max_headers.min(MAX_HEADERS);
    let mut response = httparse::Response::new(&mut headers[..cap]);
    let body_offset = match response.parse(buffer).map_err(map_httparse_error)? {
        httparse::Status::Partial => return Ok(ResponseStatus::Partial),
        httparse::Status::Complete(offset) => offset,
    };
    for header in response.headers.iter() {
        if header.name.len() + header.value.len() + 4 > limits.max_header_line_bytes {
            return Err(ParseError::HeaderLineTooLong);
        }
    }
    let status = response.code.ok_or(ParseError::MalformedRequestLine)?;
    let version = version_from_httparse(response.version)?;
    // `reason` borrows into `buffer`; httparse hands back a `&str` view.
    let reason: &str = response.reason.unwrap_or("");
    let parsed_headers: Vec<Header<'_>> = response
        .headers
        .iter()
        .map(|header| Header::new(header.name.as_bytes(), header.value))
        .collect();
    Ok(ResponseStatus::Complete {
        head: ResponseHead {
            status,
            version,
            reason,
            headers: parsed_headers,
        },
        body_offset,
    })
}

/// Pick the body framing for a parsed response head per RFC 7230 §3.3.
/// `Transfer-Encoding: chunked` wins over `Content-Length`; a numeric
/// `Content-Length` yields `ContentLength(n)`; neither present means
/// `None` (the response either has no body or — for HTTP/1.0 — runs to
/// connection close, which this leaf does not model).
#[must_use]
pub fn framing_from_response(head: &ResponseHead<'_>) -> BodyFraming {
    if let Some(encoding) = head.header("transfer-encoding")
        && let Some(value) = encoding.value_str()
        && value
            .split(',')
            .any(|token| token.trim().eq_ignore_ascii_case("chunked"))
    {
        return BodyFraming::Chunked;
    }
    if let Some(length) = head.header("content-length")
        && let Some(value) = length.value_str()
        && let Ok(parsed) = value.trim().parse::<u64>()
    {
        return BodyFraming::ContentLength(parsed);
    }
    BodyFraming::None
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn complete(buffer: &[u8]) -> (ResponseHead<'_>, usize) {
        match parse_response_head(buffer).expect("parse") {
            ResponseStatus::Complete { head, body_offset } => (head, body_offset),
            ResponseStatus::Partial => panic!("expected Complete, got Partial"),
        }
    }

    #[test]
    fn encode_get_emits_exact_request_wire_bytes() {
        let mut out = Vec::new();
        encode_request_head(
            "GET",
            "/hello",
            &[("host", "example.com"), ("connection", "keep-alive")],
            &mut out,
        );
        assert_eq!(
            core::str::from_utf8(&out).expect("ascii"),
            "GET /hello HTTP/1.1\r\nhost: example.com\r\nconnection: keep-alive\r\n\r\n"
        );
    }

    #[test]
    fn encode_request_with_no_headers_is_request_line_and_blank_line() {
        let mut out = Vec::new();
        let headers: &[(&str, &str)] = &[];
        encode_request_head("HEAD", "/", headers, &mut out);
        assert_eq!(out, b"HEAD / HTTP/1.1\r\n\r\n");
    }

    #[test]
    fn parse_content_length_response_reports_status_and_body_offset() {
        let buffer = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello";
        let (head, body_offset) = complete(buffer);
        assert_eq!(head.status, 200);
        assert_eq!(head.version, HttpVersion::Http11);
        assert_eq!(head.reason, "OK");
        assert_eq!(head.header("content-length").unwrap().value(), b"5");
        assert_eq!(&buffer[body_offset..], b"hello");
        assert_eq!(framing_from_response(&head), BodyFraming::ContentLength(5));
    }

    #[test]
    fn parse_chunked_response_selects_chunked_framing() {
        let buffer =
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n";
        let (head, body_offset) = complete(buffer);
        assert_eq!(head.status, 200);
        assert_eq!(framing_from_response(&head), BodyFraming::Chunked);
        assert_eq!(&buffer[body_offset..], b"5\r\nhello\r\n0\r\n\r\n");
    }

    #[test]
    fn parse_multiple_headers_borrows_each_into_buffer() {
        let buffer =
            b"HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\nContent-Length: 0\r\n\r\n";
        let (head, _) = complete(buffer);
        assert_eq!(head.status, 404);
        assert_eq!(head.reason, "Not Found");
        assert_eq!(head.headers.len(), 2);
        assert_eq!(head.header("content-type").unwrap().value(), b"text/plain");
        assert_eq!(framing_from_response(&head), BodyFraming::ContentLength(0));
    }

    #[test]
    fn parse_partial_head_returns_partial() {
        let buffer = b"HTTP/1.1 200 O";
        assert_eq!(
            parse_response_head(buffer).expect("parse"),
            ResponseStatus::Partial
        );
    }

    #[test]
    fn parse_bad_version_is_rejected() {
        let buffer = b"HTTP/2.0 200 OK\r\n\r\n";
        assert_eq!(parse_response_head(buffer), Err(ParseError::BadVersion));
    }

    #[test]
    fn parse_malformed_status_line_is_rejected() {
        let buffer = b"NOTHTTP 200 OK\r\n\r\n";
        assert!(parse_response_head(buffer).is_err());
    }

    #[test]
    fn framing_defaults_to_none_without_length_or_encoding() {
        let buffer = b"HTTP/1.1 204 No Content\r\n\r\n";
        let (head, _) = complete(buffer);
        assert_eq!(framing_from_response(&head), BodyFraming::None);
    }
}
