//! Sans-IO RESP (Redis Serialization Protocol) codec — borrowed-view parser,
//! encoder, and the client→server command framing.
//!
//! This is the codec substrate (workspace principle 11): an enum-shaped frame
//! model, bytes-in / borrowed-views-out, no I/O in the core. The async client
//! (`proxima-redis`'s `RedisClientUpstream`) and the blocking driver
//! (`RedisClient`) both drive the sans-IO `ClientSession` over this codec
//! from the `proxima-redis` facade crate, so the wire transport is
//! pluggable (prime, tokio, TLS-wrapped) — the same split pgwire uses.
//!
//! RESP2 + RESP3 (the wire is identical except the extra RESP3 type tags;
//! Valkey speaks the same protocol). Coverage:
//!
//! | tag  | type                | variant                          |
//! |------|---------------------|----------------------------------|
//! | `+`  | simple string       | [`Frame::SimpleString`]          |
//! | `-`  | simple error        | [`Frame::Error`]                 |
//! | `:`  | integer             | [`Frame::Integer`]               |
//! | `$`  | blob string         | [`Frame::BlobString`] / `NullBlob` |
//! | `*`  | array               | [`Frame::Array`] / `NullArray`   |
//! | `_`  | null (RESP3)        | [`Frame::Null`]                  |
//! | `,`  | double (RESP3)      | [`Frame::Double`]                |
//! | `#`  | boolean (RESP3)     | [`Frame::Boolean`]               |
//! | `(`  | big number (RESP3)  | [`Frame::BigNumber`]             |
//! | `=`  | verbatim (RESP3)    | [`Frame::VerbatimString`]        |
//! | `!`  | blob error (RESP3)  | [`Frame::BlobError`]             |
//! | `%`  | map (RESP3)         | [`Frame::Map`]                   |
//! | `~`  | set (RESP3)         | [`Frame::Set`]                   |
//! | `>`  | push (RESP3)        | [`Frame::Push`]                  |
//! | `\|` | attribute (RESP3)   | [`Frame::Attribute`]             |
//!
//! Streamed aggregates (`$?` / `*?` chunked) are out of scope — real Redis /
//! Valkey never emit them for the request/response and HELLO/AUTH flows this
//! client drives; they are rejected as malformed rather than mis-parsed.

use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::io;

pub mod pipe_contract;
pub mod value;

#[cfg(feature = "redis-codec-trait")]
pub mod codec_trait;
#[cfg(feature = "redis-codec-trait")]
pub use codec_trait::RedisFrameCodec;

pub use pipe_contract::{RedisRequest, verb};
pub use value::RespValue;

/// Decoded RESP frame. Borrowed variant — payload bytes are `&'a [u8]` slices
/// pointing into the caller's parse buffer. Zero allocation per frame on the
/// scalar hot path; aggregates allocate a `Vec` for their element list
/// (recursion would be awkward on the type).
///
/// For an owned, `'static` value that can ride a `Carry` across an async
/// boundary, lower to [`RespValue`] via [`RespValue::from_frame`].
#[derive(Debug, Clone, PartialEq)]
pub enum Frame<'a> {
    SimpleString(&'a [u8]),
    Error(&'a [u8]),
    Integer(i64),
    BlobString(&'a [u8]),
    /// `$-1\r\n` legacy nil; RESP3 prefers `_\r\n` Null but blob -1 is still
    /// wire-valid from RESP2 servers.
    NullBlob,
    Array(Vec<Frame<'a>>),
    /// `*-1\r\n` legacy nil array.
    NullArray,
    /// `_\r\n` RESP3 null.
    Null,
    /// `,3.14\r\n` RESP3 double (also `inf` / `-inf` / `nan`).
    Double(f64),
    /// `#t\r\n` / `#f\r\n` RESP3 boolean.
    Boolean(bool),
    /// `(12345...\r\n` RESP3 big number — ASCII digits, possibly beyond i64.
    BigNumber(&'a [u8]),
    /// `=15\r\ntxt:...\r\n` RESP3 verbatim string: a 3-byte `format`, a `:`,
    /// then the text. `format` is e.g. `txt` / `mkd`.
    VerbatimString {
        format: &'a [u8],
        text: &'a [u8],
    },
    /// `!21\r\nSYNTAX ...\r\n` RESP3 blob error (binary-safe error payload).
    BlobError(&'a [u8]),
    /// `%<n>\r\n` RESP3 map — `n` key/value pairs.
    Map(Vec<(Frame<'a>, Frame<'a>)>),
    /// `~<n>\r\n` RESP3 set.
    Set(Vec<Frame<'a>>),
    /// `><n>\r\n` RESP3 out-of-band push (pub/sub messages, invalidation).
    Push(Vec<Frame<'a>>),
    /// `|<n>\r\n` RESP3 attribute — metadata that prefixes the next reply.
    Attribute(Vec<(Frame<'a>, Frame<'a>)>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// Buffer too short to complete a frame; caller reads more bytes and
    /// retries.
    NeedMore,
    /// Wire byte sequence violates RESP framing.
    Malformed(&'static str),
}

impl core::fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NeedMore => formatter.write_str("incomplete RESP frame: need more bytes"),
            Self::Malformed(reason) => write!(formatter, "malformed RESP frame: {reason}"),
        }
    }
}

impl core::error::Error for ParseError {}

#[cfg(feature = "std")]
impl From<ParseError> for io::Error {
    fn from(value: ParseError) -> Self {
        match value {
            ParseError::NeedMore => {
                io::Error::new(io::ErrorKind::UnexpectedEof, "resp: short read")
            }
            ParseError::Malformed(reason) => io::Error::other(format!("resp: malformed: {reason}")),
        }
    }
}

/// Parse one RESP frame from `buf`. On success returns the frame and the number
/// of bytes consumed. The returned `Frame<'a>` references slices of `buf`; the
/// caller must not free `buf` until done with the frame.
///
/// # Errors
/// [`ParseError::NeedMore`] when `buf` is a prefix of a frame, or
/// [`ParseError::Malformed`] when the bytes violate framing.
pub fn parse(buf: &[u8]) -> Result<(Frame<'_>, usize), ParseError> {
    if buf.is_empty() {
        return Err(ParseError::NeedMore);
    }
    let tag = buf[0];
    let rest = &buf[1..];
    match tag {
        b'+' => {
            let crlf = find_crlf(rest)?;
            Ok((Frame::SimpleString(&rest[..crlf]), 1 + crlf + 2))
        }
        b'-' => {
            let crlf = find_crlf(rest)?;
            Ok((Frame::Error(&rest[..crlf]), 1 + crlf + 2))
        }
        b':' => {
            let crlf = find_crlf(rest)?;
            let value = parse_i64(&rest[..crlf])?;
            Ok((Frame::Integer(value), 1 + crlf + 2))
        }
        b'_' => {
            let crlf = find_crlf(rest)?;
            if crlf != 0 {
                return Err(ParseError::Malformed("null token carries data"));
            }
            Ok((Frame::Null, 1 + crlf + 2))
        }
        b'#' => {
            let crlf = find_crlf(rest)?;
            let value = match &rest[..crlf] {
                b"t" => true,
                b"f" => false,
                _ => return Err(ParseError::Malformed("boolean not t/f")),
            };
            Ok((Frame::Boolean(value), 1 + crlf + 2))
        }
        b',' => {
            let crlf = find_crlf(rest)?;
            let value = parse_f64(&rest[..crlf])?;
            Ok((Frame::Double(value), 1 + crlf + 2))
        }
        b'(' => {
            let crlf = find_crlf(rest)?;
            Ok((Frame::BigNumber(&rest[..crlf]), 1 + crlf + 2))
        }
        b'$' => parse_blob(buf, rest, Frame::BlobString, Frame::NullBlob),
        b'!' => parse_blob(buf, rest, Frame::BlobError, Frame::Null),
        b'=' => {
            let (frame, consumed) = parse_blob(buf, rest, Frame::BlobString, Frame::Null)?;
            match frame {
                Frame::BlobString(payload) => match payload.iter().position(|&byte| byte == b':') {
                    Some(index) => Ok((
                        Frame::VerbatimString {
                            format: &payload[..index],
                            text: &payload[index + 1..],
                        },
                        consumed,
                    )),
                    None => Err(ParseError::Malformed(
                        "verbatim string missing format prefix",
                    )),
                },
                _ => Err(ParseError::Malformed("verbatim string cannot be null")),
            }
        }
        b'*' => parse_sequence(buf, rest, Frame::Array, Frame::NullArray),
        b'~' => parse_sequence(buf, rest, Frame::Set, Frame::Null),
        b'>' => parse_sequence(buf, rest, Frame::Push, Frame::Null),
        b'%' => parse_pairs(buf, rest, Frame::Map),
        b'|' => parse_pairs(buf, rest, Frame::Attribute),
        _ => Err(ParseError::Malformed("unknown frame tag")),
    }
}

/// Shared `$`/`!`/`=` blob body parse: `<len>\r\n<payload>\r\n`, with a `-1`
/// length producing `null`.
fn parse_blob<'a>(
    buf: &'a [u8],
    rest: &'a [u8],
    make: fn(&'a [u8]) -> Frame<'a>,
    null: Frame<'a>,
) -> Result<(Frame<'a>, usize), ParseError> {
    let crlf = find_crlf(rest)?;
    let len = parse_i64(&rest[..crlf])?;
    if len == -1 {
        return Ok((null, 1 + crlf + 2));
    }
    if len < 0 {
        return Err(ParseError::Malformed("blob length negative (not -1)"));
    }
    let len = len as usize;
    let payload_start = 1 + crlf + 2;
    let payload_end = payload_start + len;
    let total = payload_end + 2;
    if buf.len() < total {
        return Err(ParseError::NeedMore);
    }
    if buf[payload_end] != b'\r' || buf[payload_end + 1] != b'\n' {
        return Err(ParseError::Malformed("blob missing trailing CRLF"));
    }
    Ok((make(&buf[payload_start..payload_end]), total))
}

/// Shared `*`/`~`/`>` aggregate parse: `<count>\r\n<elem>...`, with a `-1`
/// count producing `null` (legacy nil array).
fn parse_sequence<'a>(
    buf: &'a [u8],
    rest: &'a [u8],
    make: fn(Vec<Frame<'a>>) -> Frame<'a>,
    null: Frame<'a>,
) -> Result<(Frame<'a>, usize), ParseError> {
    let crlf = find_crlf(rest)?;
    let len = parse_i64(&rest[..crlf])?;
    if len == -1 {
        return Ok((null, 1 + crlf + 2));
    }
    if len < 0 {
        return Err(ParseError::Malformed("aggregate count negative (not -1)"));
    }
    let len = len as usize;
    let mut elements = Vec::with_capacity(len);
    let mut cursor = 1 + crlf + 2;
    for _ in 0..len {
        let (element, used) = parse(&buf[cursor..])?;
        elements.push(element);
        cursor += used;
    }
    Ok((make(elements), cursor))
}

/// Shared `%`/`|` pair-aggregate parse: `<count>\r\n<key><value>...`, where
/// `count` is the number of pairs (so `2 * count` sub-frames).
fn parse_pairs<'a>(
    buf: &'a [u8],
    rest: &'a [u8],
    make: fn(Vec<(Frame<'a>, Frame<'a>)>) -> Frame<'a>,
) -> Result<(Frame<'a>, usize), ParseError> {
    let crlf = find_crlf(rest)?;
    let count = parse_i64(&rest[..crlf])?;
    if count < 0 {
        return Err(ParseError::Malformed("map/attribute count negative"));
    }
    let count = count as usize;
    let mut pairs = Vec::with_capacity(count);
    let mut cursor = 1 + crlf + 2;
    for _ in 0..count {
        let (key, used) = parse(&buf[cursor..])?;
        cursor += used;
        let (value, used) = parse(&buf[cursor..])?;
        cursor += used;
        pairs.push((key, value));
    }
    Ok((make(pairs), cursor))
}

/// Parse an ASCII-decimal i64 without going through `str::from_utf8` +
/// `str::parse`. RESP wire bytes are ASCII; we own the validation surface, so
/// the manual scan saves a UTF-8 check plus the parser-state-machine overhead.
fn parse_i64(buf: &[u8]) -> Result<i64, ParseError> {
    if buf.is_empty() {
        return Err(ParseError::Malformed("empty integer"));
    }
    let (negative, digits) = match buf[0] {
        b'-' => (true, &buf[1..]),
        b'+' => (false, &buf[1..]),
        _ => (false, buf),
    };
    if digits.is_empty() {
        return Err(ParseError::Malformed("integer missing digits"));
    }
    let mut value: i64 = 0;
    for &byte in digits {
        if !byte.is_ascii_digit() {
            return Err(ParseError::Malformed("integer non-digit"));
        }
        value = value
            .checked_mul(10)
            .and_then(|acc| acc.checked_add(i64::from(byte - b'0')))
            .ok_or(ParseError::Malformed("integer overflow"))?;
    }
    if negative { Ok(-value) } else { Ok(value) }
}

/// Parse a RESP3 double line. Redis emits `inf` / `-inf` / `nan` for the
/// non-finite cases, which `f64::from_str` accepts directly.
fn parse_f64(buf: &[u8]) -> Result<f64, ParseError> {
    let text = core::str::from_utf8(buf).map_err(|_| ParseError::Malformed("double not utf-8"))?;
    text.parse::<f64>()
        .map_err(|_| ParseError::Malformed("double not a number"))
}

fn find_crlf(buf: &[u8]) -> Result<usize, ParseError> {
    // SIMD `\r\n` scan; RESP is CRLF-delimited and this runs per frame, so the
    // scalar byte-at-a-time loop was hot. Mirrors the pgwire codec's memchr use.
    memchr::memmem::find(buf, b"\r\n").ok_or(ParseError::NeedMore)
}

/// Encode a command as a RESP array of bulk strings — the canonical
/// client→server request framing (`*<argc>\r\n$<len>\r\n<arg>\r\n...`). Binary
/// safe; every argument (command verb included) is a length-prefixed blob.
pub fn encode_command(args: &[&[u8]], out: &mut Vec<u8>) {
    out.push(b'*');
    push_usize(out, args.len());
    out.extend_from_slice(b"\r\n");
    for arg in args {
        out.push(b'$');
        push_usize(out, arg.len());
        out.extend_from_slice(b"\r\n");
        out.extend_from_slice(arg);
        out.extend_from_slice(b"\r\n");
    }
}

/// Serialize a frame into RESP wire bytes. Inverse of [`parse`] for every
/// finite-valued variant; round-trip preserves the supported types.
#[must_use]
pub fn encode(frame: &Frame<'_>) -> Vec<u8> {
    let mut out = Vec::new();
    encode_into(frame, &mut out);
    out
}

pub fn encode_into(frame: &Frame<'_>, out: &mut Vec<u8>) {
    match frame {
        Frame::SimpleString(bytes) => prefixed_line(out, b'+', bytes),
        Frame::Error(bytes) => prefixed_line(out, b'-', bytes),
        Frame::Integer(value) => {
            out.push(b':');
            push_i64(out, *value);
            out.extend_from_slice(b"\r\n");
        }
        Frame::BlobString(bytes) => blob(out, b'$', bytes),
        Frame::BlobError(bytes) => blob(out, b'!', bytes),
        Frame::NullBlob => out.extend_from_slice(b"$-1\r\n"),
        Frame::NullArray => out.extend_from_slice(b"*-1\r\n"),
        Frame::Null => out.extend_from_slice(b"_\r\n"),
        Frame::Boolean(value) => out.extend_from_slice(if *value { b"#t\r\n" } else { b"#f\r\n" }),
        Frame::Double(value) => {
            out.push(b',');
            push_f64(out, *value);
            out.extend_from_slice(b"\r\n");
        }
        Frame::BigNumber(digits) => prefixed_line(out, b'(', digits),
        Frame::VerbatimString { format, text } => {
            let length = format.len() + 1 + text.len();
            out.push(b'=');
            push_usize(out, length);
            out.extend_from_slice(b"\r\n");
            out.extend_from_slice(format);
            out.push(b':');
            out.extend_from_slice(text);
            out.extend_from_slice(b"\r\n");
        }
        Frame::Array(elements) => aggregate(out, b'*', elements),
        Frame::Set(elements) => aggregate(out, b'~', elements),
        Frame::Push(elements) => aggregate(out, b'>', elements),
        Frame::Map(pairs) => pair_aggregate(out, b'%', pairs),
        Frame::Attribute(pairs) => pair_aggregate(out, b'|', pairs),
    }
}

fn prefixed_line(out: &mut Vec<u8>, tag: u8, bytes: &[u8]) {
    out.push(tag);
    out.extend_from_slice(bytes);
    out.extend_from_slice(b"\r\n");
}

fn blob(out: &mut Vec<u8>, tag: u8, bytes: &[u8]) {
    out.push(tag);
    push_usize(out, bytes.len());
    out.extend_from_slice(b"\r\n");
    out.extend_from_slice(bytes);
    out.extend_from_slice(b"\r\n");
}

fn aggregate(out: &mut Vec<u8>, tag: u8, elements: &[Frame<'_>]) {
    out.push(tag);
    push_usize(out, elements.len());
    out.extend_from_slice(b"\r\n");
    for element in elements {
        encode_into(element, out);
    }
}

fn pair_aggregate(out: &mut Vec<u8>, tag: u8, pairs: &[(Frame<'_>, Frame<'_>)]) {
    out.push(tag);
    push_usize(out, pairs.len());
    out.extend_from_slice(b"\r\n");
    for (key, value) in pairs {
        encode_into(key, out);
        encode_into(value, out);
    }
}

fn push_i64(out: &mut Vec<u8>, value: i64) {
    if value < 0 {
        out.push(b'-');
        push_u64(out, value.unsigned_abs());
    } else {
        push_u64(out, value as u64);
    }
}

fn push_usize(out: &mut Vec<u8>, value: usize) {
    push_u64(out, value as u64);
}

fn push_u64(out: &mut Vec<u8>, mut value: u64) {
    if value == 0 {
        out.push(b'0');
        return;
    }
    // u64::MAX is 20 digits; scratch fits with margin.
    let mut scratch = [0_u8; 20];
    let mut index = scratch.len();
    while value > 0 {
        index -= 1;
        scratch[index] = b'0' + (value % 10) as u8;
        value /= 10;
    }
    out.extend_from_slice(&scratch[index..]);
}

fn push_f64(out: &mut Vec<u8>, value: f64) {
    if value.is_infinite() {
        out.extend_from_slice(if value > 0.0 { b"inf" } else { b"-inf" });
    } else if value.is_nan() {
        out.extend_from_slice(b"nan");
    } else {
        // alloc::format is the no_std-friendly route to a decimal rendering;
        // the double path is cold (replies, not the request hot loop).
        let rendered = alloc::format!("{value}");
        out.extend_from_slice(rendered.as_bytes());
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn parses_simple_string() {
        let buf = b"+OK\r\n";
        let (frame, used) = parse(buf).unwrap();
        assert_eq!(frame, Frame::SimpleString(b"OK"));
        assert_eq!(used, buf.len());
    }

    #[test]
    fn parses_error() {
        let buf = b"-ERR unknown command 'fizz'\r\n";
        let (frame, used) = parse(buf).unwrap();
        assert_eq!(frame, Frame::Error(b"ERR unknown command 'fizz'"));
        assert_eq!(used, buf.len());
    }

    #[test]
    fn parses_integer() {
        let buf = b":42\r\n";
        let (frame, used) = parse(buf).unwrap();
        assert_eq!(frame, Frame::Integer(42));
        assert_eq!(used, buf.len());
    }

    #[test]
    fn parses_negative_integer() {
        let (frame, _) = parse(b":-100\r\n").unwrap();
        assert_eq!(frame, Frame::Integer(-100));
    }

    #[test]
    fn parses_blob_string() {
        let buf = b"$5\r\nhello\r\n";
        let (frame, used) = parse(buf).unwrap();
        assert_eq!(frame, Frame::BlobString(b"hello"));
        assert_eq!(used, buf.len());
    }

    #[test]
    fn parses_null_blob() {
        let (frame, _) = parse(b"$-1\r\n").unwrap();
        assert_eq!(frame, Frame::NullBlob);
    }

    #[test]
    fn parses_array_of_blobs() {
        let buf = b"*2\r\n$5\r\nhello\r\n$5\r\nworld\r\n";
        let (frame, used) = parse(buf).unwrap();
        assert_eq!(
            frame,
            Frame::Array(vec![
                Frame::BlobString(b"hello"),
                Frame::BlobString(b"world")
            ])
        );
        assert_eq!(used, buf.len());
    }

    #[test]
    fn parses_null_array() {
        let (frame, _) = parse(b"*-1\r\n").unwrap();
        assert_eq!(frame, Frame::NullArray);
    }

    #[test]
    fn parses_resp3_null() {
        let (frame, used) = parse(b"_\r\n").unwrap();
        assert_eq!(frame, Frame::Null);
        assert_eq!(used, 3);
    }

    #[test]
    fn parses_resp3_boolean() {
        assert_eq!(parse(b"#t\r\n").unwrap().0, Frame::Boolean(true));
        assert_eq!(parse(b"#f\r\n").unwrap().0, Frame::Boolean(false));
    }

    #[test]
    fn parses_resp3_double() {
        assert_eq!(parse(b",2.5\r\n").unwrap().0, Frame::Double(2.5));
        assert_eq!(parse(b",inf\r\n").unwrap().0, Frame::Double(f64::INFINITY));
        assert_eq!(
            parse(b",-inf\r\n").unwrap().0,
            Frame::Double(f64::NEG_INFINITY)
        );
    }

    #[test]
    fn parses_resp3_big_number() {
        let (frame, _) = parse(b"(3492890328409238509324850943850943825024385\r\n").unwrap();
        assert_eq!(
            frame,
            Frame::BigNumber(b"3492890328409238509324850943850943825024385")
        );
    }

    #[test]
    fn parses_resp3_verbatim_string() {
        let buf = b"=15\r\ntxt:Some string\r\n";
        let (frame, used) = parse(buf).unwrap();
        assert_eq!(
            frame,
            Frame::VerbatimString {
                format: b"txt",
                text: b"Some string"
            }
        );
        assert_eq!(used, buf.len());
    }

    #[test]
    fn parses_resp3_blob_error() {
        let (frame, _) = parse(b"!21\r\nSYNTAX invalid syntax\r\n").unwrap();
        assert_eq!(frame, Frame::BlobError(b"SYNTAX invalid syntax"));
    }

    #[test]
    fn parses_resp3_map() {
        let buf = b"%2\r\n+first\r\n:1\r\n+second\r\n:2\r\n";
        let (frame, used) = parse(buf).unwrap();
        assert_eq!(
            frame,
            Frame::Map(vec![
                (Frame::SimpleString(b"first"), Frame::Integer(1)),
                (Frame::SimpleString(b"second"), Frame::Integer(2)),
            ])
        );
        assert_eq!(used, buf.len());
    }

    #[test]
    fn parses_resp3_set() {
        let buf = b"~2\r\n+a\r\n+b\r\n";
        let (frame, _) = parse(buf).unwrap();
        assert_eq!(
            frame,
            Frame::Set(vec![Frame::SimpleString(b"a"), Frame::SimpleString(b"b")])
        );
    }

    #[test]
    fn parses_resp3_push() {
        let buf = b">3\r\n$7\r\nmessage\r\n$3\r\nchn\r\n$5\r\nhello\r\n";
        let (frame, _) = parse(buf).unwrap();
        assert_eq!(
            frame,
            Frame::Push(vec![
                Frame::BlobString(b"message"),
                Frame::BlobString(b"chn"),
                Frame::BlobString(b"hello"),
            ])
        );
    }

    #[test]
    fn need_more_on_truncated_blob() {
        assert_eq!(parse(b"$10\r\nhel"), Err(ParseError::NeedMore));
    }

    #[test]
    fn malformed_on_unknown_tag() {
        assert!(matches!(parse(b"@nope\r\n"), Err(ParseError::Malformed(_))));
    }

    #[test]
    fn encodes_get_command() {
        let mut out = Vec::new();
        encode_command(&[b"GET", b"mykey"], &mut out);
        assert_eq!(out, b"*2\r\n$3\r\nGET\r\n$5\r\nmykey\r\n");
    }

    #[test]
    fn encodes_set_with_binary_value() {
        let mut out = Vec::new();
        encode_command(&[b"SET", b"k", &[0x00, 0xff, b'\r', b'\n']], &mut out);
        assert_eq!(out, b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$4\r\n\x00\xff\r\n\r\n");
    }

    #[test]
    fn round_trip_simple_string() {
        let frame = Frame::SimpleString(b"PONG");
        let bytes = encode(&frame);
        let (parsed, _) = parse(&bytes).unwrap();
        assert_eq!(parsed, frame);
    }

    #[test]
    fn round_trip_nested_array() {
        let frame = Frame::Array(vec![
            Frame::SimpleString(b"OK"),
            Frame::Integer(7),
            Frame::Array(vec![Frame::BlobString(b"nested"), Frame::NullBlob]),
        ]);
        let bytes = encode(&frame);
        let (parsed, used) = parse(&bytes).unwrap();
        assert_eq!(parsed, frame);
        assert_eq!(used, bytes.len());
    }

    #[test]
    fn round_trip_resp3_map_and_double() {
        let frame = Frame::Map(vec![
            (Frame::SimpleString(b"pi"), Frame::Double(3.5)),
            (Frame::SimpleString(b"ok"), Frame::Boolean(true)),
            (Frame::SimpleString(b"nil"), Frame::Null),
        ]);
        let bytes = encode(&frame);
        let (parsed, used) = parse(&bytes).unwrap();
        assert_eq!(parsed, frame);
        assert_eq!(used, bytes.len());
    }

    #[test]
    fn round_trip_verbatim_string() {
        let frame = Frame::VerbatimString {
            format: b"mkd",
            text: b"# title",
        };
        let bytes = encode(&frame);
        let (parsed, _) = parse(&bytes).unwrap();
        assert_eq!(parsed, frame);
    }
}
