//! HTTP/1.1 response writer state machine.
//!
//! Stage 3 of the L7 H1 state-machine track. Output side of the
//! handshake: serializes a status line + headers (one shot) and
//! frames body chunks according to Content-Length or chunked
//! transfer-encoding (stateful, called per chunk).
//!
//! Like the parser and body decoder, this module is allocation-free
//! per call beyond the caller's output buffer growth: the writer
//! appends bytes to a `Vec<u8>` the caller supplies, and any
//! framing overhead (chunk size lines, terminator) is encoded
//! directly into that buffer.

use alloc::string::String;
use alloc::vec::Vec;

use crate::http1_codec::h1::HttpVersion;
use crate::http1_codec::h1_body::BodyFraming;

/// Render a 3-digit decimal status code (100..=599) into the buffer.
/// Replaces `write!(out, "{status}")` which would need std::io::Write.
fn write_status_decimal(out: &mut Vec<u8>, status: u16) {
    let hundreds = ((status / 100) % 10) as u8 + b'0';
    let tens = ((status / 10) % 10) as u8 + b'0';
    let ones = (status % 10) as u8 + b'0';
    out.push(hundreds);
    out.push(tens);
    out.push(ones);
}

/// Write `value` as ASCII decimal digits into `out` with no heap allocation.
/// Uses a 20-byte stack buffer (max u64 decimal width is 20 digits).
fn write_u64_decimal(out: &mut Vec<u8>, value: u64) {
    if value == 0 {
        out.push(b'0');
        return;
    }
    let mut digits = [0_u8; 20];
    let mut count = 0;
    let mut remaining = value;
    while remaining > 0 {
        digits[count] = b'0' + (remaining % 10) as u8;
        count += 1;
        remaining /= 10;
    }
    for index in 0..count {
        out.push(digits[count - 1 - index]);
    }
}

/// Serialize the response head: `HTTP/1.x STATUS REASON\r\nName: Value\r\n...\r\n\r\n`.
///
/// `reason` falls back to a minimal status-text table when blank;
/// callers who want strict RFC reasons should pass them explicitly.
/// Headers are written in the order provided — most servers preserve
/// insertion order, and that's what makes Set-Cookie and similar
/// repeated headers behave.
///
/// `framing` is authoritative for content-length: when
/// `BodyFraming::ContentLength(len)` is present, exactly one
/// `content-length: <len>` header is emitted after the caller-supplied
/// headers (any content-length entry already in `headers` is skipped).
/// This prevents double-emission and removes the `to_string()` call.
pub fn write_response_head(
    out: &mut Vec<u8>,
    version: HttpVersion,
    status: u16,
    reason: &str,
    headers: &[(String, String)],
    framing: BodyFraming,
) {
    let version_bytes: &[u8] = match version {
        HttpVersion::Http10 => b"HTTP/1.0",
        HttpVersion::Http11 => b"HTTP/1.1",
    };
    out.extend_from_slice(version_bytes);
    out.push(b' ');
    write_status_decimal(out, status);
    out.push(b' ');
    let reason_bytes = if reason.is_empty() {
        default_reason(status)
    } else {
        reason.as_bytes()
    };
    out.extend_from_slice(reason_bytes);
    out.extend_from_slice(b"\r\n");
    for (name, value) in headers {
        if name.as_bytes().eq_ignore_ascii_case(b"content-length") {
            continue;
        }
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(b": ");
        out.extend_from_slice(value.as_bytes());
        out.extend_from_slice(b"\r\n");
    }
    if let BodyFraming::ContentLength(len) = framing {
        out.extend_from_slice(b"content-length: ");
        write_u64_decimal(out, len);
        out.extend_from_slice(b"\r\n");
    }
    out.extend_from_slice(b"\r\n");
}

/// Minimal default reason phrases for the common statuses. Callers
/// who care about specific text should supply it explicitly via
/// `write_response_head`; this table only covers the codes the
/// substrate itself produces (synth, control plane, internal errors).
fn default_reason(status: u16) -> &'static [u8] {
    match status {
        100 => b"Continue",
        101 => b"Switching Protocols",
        200 => b"OK",
        201 => b"Created",
        202 => b"Accepted",
        204 => b"No Content",
        206 => b"Partial Content",
        301 => b"Moved Permanently",
        302 => b"Found",
        303 => b"See Other",
        304 => b"Not Modified",
        307 => b"Temporary Redirect",
        308 => b"Permanent Redirect",
        400 => b"Bad Request",
        401 => b"Unauthorized",
        403 => b"Forbidden",
        404 => b"Not Found",
        405 => b"Method Not Allowed",
        408 => b"Request Timeout",
        409 => b"Conflict",
        410 => b"Gone",
        413 => b"Payload Too Large",
        414 => b"URI Too Long",
        415 => b"Unsupported Media Type",
        429 => b"Too Many Requests",
        500 => b"Internal Server Error",
        501 => b"Not Implemented",
        502 => b"Bad Gateway",
        503 => b"Pipe Unavailable",
        504 => b"Gateway Timeout",
        505 => b"HTTP Version Not Supported",
        _ => b"",
    }
}

/// Frames body chunks for the wire according to the selected
/// transfer encoding. Construct with the framing decision the head
/// committed to (Content-Length vs chunked); call `encode_chunk`
/// once per body chunk and `encode_end` exactly once when the body
/// is complete.
pub struct BodyEncoder {
    framing: BodyFraming,
    closed: bool,
}

impl BodyEncoder {
    #[must_use]
    pub fn new(framing: BodyFraming) -> Self {
        Self {
            framing,
            closed: false,
        }
    }

    /// Append `data` to `out` framed for the active encoding.
    /// - ContentLength: raw bytes (caller is responsible for not
    ///   exceeding the declared length).
    /// - Chunked: `hex(len)\r\n<data>\r\n`. A zero-length input is
    ///   skipped (a 0-length chunk is reserved for the terminator).
    /// - None: no-op.
    pub fn encode_chunk(&self, data: &[u8], out: &mut Vec<u8>) {
        if self.closed || data.is_empty() {
            return;
        }
        match self.framing {
            BodyFraming::None => {}
            BodyFraming::ContentLength(_) => {
                out.extend_from_slice(data);
            }
            BodyFraming::Chunked => {
                let mut size_buf = [0_u8; 16];
                let written = format_hex(data.len() as u64, &mut size_buf);
                out.extend_from_slice(&size_buf[..written]);
                out.extend_from_slice(b"\r\n");
                out.extend_from_slice(data);
                out.extend_from_slice(b"\r\n");
            }
        }
    }

    /// Write the terminator. For chunked, `0\r\n\r\n` (no trailers).
    /// For Content-Length or None, no-op. Idempotent.
    pub fn encode_end(&mut self, out: &mut Vec<u8>) {
        self.encode_end_with_trailers(&[], out);
    }

    /// Like `encode_end`, but if framing is Chunked AND `trailers`
    /// is non-empty, emits each `Name: value\r\n` between the
    /// 0-length chunk-size line and the terminating CRLF — RFC 7230
    /// §4.1.2. Trailers ignored for Content-Length / None framings
    /// (they have no wire slot for trailers). Idempotent on
    /// subsequent calls.
    pub fn encode_end_with_trailers(&mut self, trailers: &[(&[u8], &[u8])], out: &mut Vec<u8>) {
        if self.closed {
            return;
        }
        self.closed = true;
        if !matches!(self.framing, BodyFraming::Chunked) {
            return;
        }
        out.extend_from_slice(b"0\r\n");
        for (name, value) in trailers {
            out.extend_from_slice(name);
            out.extend_from_slice(b": ");
            out.extend_from_slice(value);
            out.extend_from_slice(b"\r\n");
        }
        out.extend_from_slice(b"\r\n");
    }
}

/// Write `value` as lowercase ASCII hex into `out`, returning the
/// number of bytes written. Used for chunk size lines; matches what
/// every other HTTP/1.1 server emits (`5\r\n`, `a\r\n`, `100\r\n`).
fn format_hex(value: u64, out: &mut [u8]) -> usize {
    if value == 0 {
        out[0] = b'0';
        return 1;
    }
    // collect digits low-to-high then reverse.
    let mut digits = [0_u8; 16];
    let mut count = 0;
    let mut remaining = value;
    while remaining > 0 {
        let nibble = (remaining & 0xF) as u8;
        digits[count] = if nibble < 10 {
            b'0' + nibble
        } else {
            b'a' + nibble - 10
        };
        count += 1;
        remaining >>= 4;
    }
    for index in 0..count {
        out[index] = digits[count - 1 - index];
    }
    count
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    #[test]
    fn write_head_emits_status_line_and_headers_in_order() {
        let headers = vec![("content-type".to_string(), "application/json".to_string())];
        let mut out = Vec::new();
        write_response_head(
            &mut out,
            HttpVersion::Http11,
            200,
            "OK",
            &headers,
            BodyFraming::ContentLength(12),
        );
        assert_eq!(
            core::str::from_utf8(&out).expect("ascii"),
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 12\r\n\r\n"
        );
    }

    #[test]
    fn write_head_skips_content_length_in_headers_list_and_emits_from_framing() {
        let headers = vec![
            ("content-type".to_string(), "text/plain".to_string()),
            ("content-length".to_string(), "99".to_string()),
        ];
        let mut out = Vec::new();
        write_response_head(
            &mut out,
            HttpVersion::Http11,
            200,
            "OK",
            &headers,
            BodyFraming::ContentLength(7),
        );
        let text = core::str::from_utf8(&out).expect("ascii");
        let count = text.matches("content-length").count();
        assert_eq!(count, 1, "exactly one content-length on the wire");
        assert!(text.contains("content-length: 7\r\n"), "framing value wins");
    }

    #[test]
    fn write_head_no_content_length_for_chunked_framing() {
        let headers = vec![("transfer-encoding".to_string(), "chunked".to_string())];
        let mut out = Vec::new();
        write_response_head(
            &mut out,
            HttpVersion::Http11,
            200,
            "OK",
            &headers,
            BodyFraming::Chunked,
        );
        let text = core::str::from_utf8(&out).expect("ascii");
        assert!(
            !text.contains("content-length"),
            "no content-length for chunked"
        );
    }

    #[test]
    fn write_head_uses_default_reason_when_blank() {
        let mut out = Vec::new();
        write_response_head(
            &mut out,
            HttpVersion::Http11,
            404,
            "",
            &[],
            BodyFraming::None,
        );
        assert!(out.starts_with(b"HTTP/1.1 404 Not Found\r\n"));
    }

    #[test]
    fn write_head_emits_http10_version_when_requested() {
        let mut out = Vec::new();
        write_response_head(
            &mut out,
            HttpVersion::Http10,
            200,
            "OK",
            &[],
            BodyFraming::None,
        );
        assert!(out.starts_with(b"HTTP/1.0 200 OK\r\n"));
    }

    #[test]
    fn write_u64_decimal_handles_zero_and_large_values() {
        let mut out = Vec::new();
        write_u64_decimal(&mut out, 0);
        assert_eq!(&out, b"0");

        out.clear();
        write_u64_decimal(&mut out, 12345);
        assert_eq!(&out, b"12345");

        out.clear();
        write_u64_decimal(&mut out, u64::MAX);
        assert_eq!(&out, b"18446744073709551615");
    }

    #[test]
    fn content_length_encoder_emits_raw_bytes() {
        let encoder = BodyEncoder::new(BodyFraming::ContentLength(5));
        let mut out = Vec::new();
        encoder.encode_chunk(b"hello", &mut out);
        assert_eq!(out, b"hello");
    }

    #[test]
    fn chunked_encoder_wraps_each_chunk_with_size_and_crlf() {
        let encoder = BodyEncoder::new(BodyFraming::Chunked);
        let mut out = Vec::new();
        encoder.encode_chunk(b"hello", &mut out);
        encoder.encode_chunk(b" world", &mut out);
        assert_eq!(out, b"5\r\nhello\r\n6\r\n world\r\n");
    }

    #[test]
    fn chunked_encoder_emits_terminator_on_end() {
        let mut encoder = BodyEncoder::new(BodyFraming::Chunked);
        let mut out = Vec::new();
        encoder.encode_chunk(b"x", &mut out);
        encoder.encode_end(&mut out);
        assert_eq!(out, b"1\r\nx\r\n0\r\n\r\n");
    }

    #[test]
    fn content_length_encoder_skips_terminator() {
        let mut encoder = BodyEncoder::new(BodyFraming::ContentLength(1));
        let mut out = Vec::new();
        encoder.encode_chunk(b"x", &mut out);
        encoder.encode_end(&mut out);
        assert_eq!(out, b"x");
    }

    #[test]
    fn chunked_encoder_skips_empty_chunk_to_protect_terminator() {
        let encoder = BodyEncoder::new(BodyFraming::Chunked);
        let mut out = Vec::new();
        encoder.encode_chunk(b"", &mut out);
        assert!(out.is_empty(), "empty chunk must be a no-op");
    }

    #[test]
    fn chunked_encoder_end_is_idempotent() {
        let mut encoder = BodyEncoder::new(BodyFraming::Chunked);
        let mut out = Vec::new();
        encoder.encode_end(&mut out);
        encoder.encode_end(&mut out);
        assert_eq!(out, b"0\r\n\r\n");
    }

    #[test]
    fn chunked_encoder_emits_trailers_between_zero_chunk_and_terminator() {
        let mut encoder = BodyEncoder::new(BodyFraming::Chunked);
        let mut out = Vec::new();
        encoder.encode_chunk(b"foo", &mut out);
        encoder.encode_end_with_trailers(&[(b"X-Result", b"ok"), (b"X-Count", b"42")], &mut out);
        assert_eq!(
            core::str::from_utf8(&out).expect("ascii"),
            "3\r\nfoo\r\n0\r\nX-Result: ok\r\nX-Count: 42\r\n\r\n"
        );
    }

    #[test]
    fn trailers_ignored_for_content_length_framing() {
        let mut encoder = BodyEncoder::new(BodyFraming::ContentLength(3));
        let mut out = Vec::new();
        encoder.encode_chunk(b"foo", &mut out);
        encoder.encode_end_with_trailers(&[(b"X", b"y")], &mut out);
        assert_eq!(out, b"foo");
    }

    #[test]
    fn format_hex_lowercases_and_handles_zero() {
        let mut buf = [0_u8; 16];
        assert_eq!(format_hex(0, &mut buf), 1);
        assert_eq!(&buf[..1], b"0");
        assert_eq!(format_hex(0xA, &mut buf), 1);
        assert_eq!(&buf[..1], b"a");
        assert_eq!(format_hex(0x1FF, &mut buf), 3);
        assert_eq!(&buf[..3], b"1ff");
        assert_eq!(format_hex(0xDEADBEEF, &mut buf), 8);
        assert_eq!(&buf[..8], b"deadbeef");
    }
}
