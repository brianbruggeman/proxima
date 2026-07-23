//! Memcached text-protocol parser (sans-IO).
//!
//! Tracked as P6 in `docs/protocol-gap/discipline.md`. The ASCII
//! protocol is line-oriented (CRLF-terminated commands). Binary
//! protocol is a follow-up; see [protocol spec][1].
//!
//! [1]: https://github.com/memcached/memcached/blob/master/doc/protocol.txt
//!
//! The parser is the substrate primitive — it returns a borrowed
//! [`Command<'a>`] describing what the peer sent. Listener wire-up
//! (TCP accept loop + dispatch + response writing) is straightforward
//! tokio code on top.
//!
//! Sub-flag: `memcached-listener` (default off).


use alloc::vec::Vec;

/// One parsed memcached request line plus any associated body.
#[derive(Debug, Clone)]
pub enum Command<'a> {
    /// `get <key1> [<key2> ...]\r\n` (also `gets` — same shape with CAS-aware reply).
    Get { keys: &'a [u8], gets: bool },
    /// Storage commands — `set` / `add` / `replace` / `append` / `prepend`.
    Store {
        mode: StoreMode,
        key: &'a [u8],
        flags: u32,
        exptime: u32,
        value: &'a [u8],
        noreply: bool,
    },
    /// `cas <key> <flags> <exptime> <bytes> <cas_unique> [noreply]\r\n<data>\r\n`
    Cas {
        key: &'a [u8],
        flags: u32,
        exptime: u32,
        cas_unique: u64,
        value: &'a [u8],
        noreply: bool,
    },
    /// `delete <key> [noreply]\r\n`
    Delete { key: &'a [u8], noreply: bool },
    /// `incr <key> <value> [noreply]\r\n` / `decr <key> <value> [noreply]\r\n`
    Counter {
        increment: bool,
        key: &'a [u8],
        delta: u64,
        noreply: bool,
    },
    /// `touch <key> <exptime> [noreply]\r\n`
    Touch {
        key: &'a [u8],
        exptime: u32,
        noreply: bool,
    },
    /// `flush_all [<delay>] [noreply]\r\n`
    FlushAll { delay: Option<u32>, noreply: bool },
    /// `stats [<args>]\r\n` — args is the rest of the line, borrowed.
    Stats { args: &'a [u8] },
    /// `version\r\n`
    Version,
    /// `quit\r\n`
    Quit,
}

/// Storage modes per memcached protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreMode {
    Set,
    Add,
    Replace,
    Append,
    Prepend,
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("buffer ended before CRLF")]
    Short,
    #[error("unknown command verb {0:?}")]
    UnknownCommand(Vec<u8>),
    #[error("malformed command: {0}")]
    Malformed(&'static str),
    #[error("invalid integer in field {0}")]
    InvalidInt(&'static str),
    #[error("declared value bytes {0} exceeds buffer")]
    PartialValue(u32),
    #[error("udp datagram ended before the 8-byte header")]
    DatagramHeaderShort,
}

/// Parse one full command from the start of `buf`. Returns the
/// command and the number of bytes consumed (always ends at the
/// final CRLF — for storage commands, that's *after* the value's
/// trailing CRLF).
///
/// Returns `ParseError::Short` when more bytes are needed; the
/// caller buffers and retries.
#[inline]
pub fn parse_command(buf: &[u8]) -> Result<(Command<'_>, usize), ParseError> {
    let line_end = find_crlf(buf).ok_or(ParseError::Short)?;
    let line = &buf[..line_end];
    let after_line = line_end + 2; // skip CRLF

    let (verb, rest) = split_token(line).ok_or(ParseError::Malformed("empty line"))?;

    match verb {
        b"get" | b"gets" => Ok((
            Command::Get {
                keys: rest,
                gets: verb == b"gets",
            },
            after_line,
        )),
        b"set" => parse_store(StoreMode::Set, rest, buf, after_line),
        b"add" => parse_store(StoreMode::Add, rest, buf, after_line),
        b"replace" => parse_store(StoreMode::Replace, rest, buf, after_line),
        b"append" => parse_store(StoreMode::Append, rest, buf, after_line),
        b"prepend" => parse_store(StoreMode::Prepend, rest, buf, after_line),
        b"cas" => parse_cas(rest, buf, after_line),
        b"delete" => parse_delete(rest, after_line),
        b"incr" => parse_counter(true, rest, after_line),
        b"decr" => parse_counter(false, rest, after_line),
        b"touch" => parse_touch(rest, after_line),
        b"flush_all" => parse_flush_all(rest, after_line),
        b"stats" => Ok((Command::Stats { args: rest }, after_line)),
        b"version" => Ok((Command::Version, after_line)),
        b"quit" => Ok((Command::Quit, after_line)),
        other => Err(ParseError::UnknownCommand(other.to_vec())),
    }
}

// `pub(crate)` (not private) — `connection`, `pipe_contract`, and `reply`
// reuse these exact tokenizer/integer primitives on the reply-parsing and
// request-model side instead of re-deriving them; the request parser
// (`parse_command`, above) stays the only place that OWNS command framing.
#[inline]
pub(crate) fn find_crlf(buf: &[u8]) -> Option<usize> {
    let mut i = 0;
    while i + 1 < buf.len() {
        if buf[i] == b'\r' && buf[i + 1] == b'\n' {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[inline]
pub(crate) fn split_token(line: &[u8]) -> Option<(&[u8], &[u8])> {
    let space = line.iter().position(|&b| b == b' ');
    match space {
        Some(idx) => {
            let mut rest_start = idx + 1;
            while rest_start < line.len() && line[rest_start] == b' ' {
                rest_start += 1;
            }
            Some((&line[..idx], &line[rest_start..]))
        }
        None => Some((line, &[])),
    }
}

#[inline]
pub(crate) fn parse_u32(token: &[u8], field: &'static str) -> Result<u32, ParseError> {
    let mut value: u32 = 0;
    for &byte in token {
        if !byte.is_ascii_digit() {
            return Err(ParseError::InvalidInt(field));
        }
        value = value
            .checked_mul(10)
            .and_then(|v| v.checked_add(u32::from(byte - b'0')))
            .ok_or(ParseError::InvalidInt(field))?;
    }
    if token.is_empty() {
        return Err(ParseError::InvalidInt(field));
    }
    Ok(value)
}

#[inline]
pub(crate) fn parse_u64(token: &[u8], field: &'static str) -> Result<u64, ParseError> {
    let mut value: u64 = 0;
    for &byte in token {
        if !byte.is_ascii_digit() {
            return Err(ParseError::InvalidInt(field));
        }
        value = value
            .checked_mul(10)
            .and_then(|v| v.checked_add(u64::from(byte - b'0')))
            .ok_or(ParseError::InvalidInt(field))?;
    }
    if token.is_empty() {
        return Err(ParseError::InvalidInt(field));
    }
    Ok(value)
}

fn parse_store<'a>(
    mode: StoreMode,
    rest: &'a [u8],
    buf: &'a [u8],
    after_line: usize,
) -> Result<(Command<'a>, usize), ParseError> {
    let (key, rest) = split_token(rest).ok_or(ParseError::Malformed("set missing key"))?;
    let (flags_tok, rest) = split_token(rest).ok_or(ParseError::Malformed("set missing flags"))?;
    let (exptime_tok, rest) =
        split_token(rest).ok_or(ParseError::Malformed("set missing exptime"))?;
    let (bytes_tok, rest) = split_token(rest).ok_or(ParseError::Malformed("set missing bytes"))?;
    let flags = parse_u32(flags_tok, "flags")?;
    let exptime = parse_u32(exptime_tok, "exptime")?;
    let value_len = parse_u32(bytes_tok, "bytes")? as usize;
    let noreply = is_noreply(rest);

    let value_start = after_line;
    let value_end = value_start + value_len;
    if buf.len() < value_end + 2 {
        return Err(ParseError::PartialValue(value_len as u32));
    }
    if &buf[value_end..value_end + 2] != b"\r\n" {
        return Err(ParseError::Malformed("value not terminated by CRLF"));
    }
    Ok((
        Command::Store {
            mode,
            key,
            flags,
            exptime,
            value: &buf[value_start..value_end],
            noreply,
        },
        value_end + 2,
    ))
}

fn parse_cas<'a>(
    rest: &'a [u8],
    buf: &'a [u8],
    after_line: usize,
) -> Result<(Command<'a>, usize), ParseError> {
    let (key, rest) = split_token(rest).ok_or(ParseError::Malformed("cas missing key"))?;
    let (flags_tok, rest) = split_token(rest).ok_or(ParseError::Malformed("cas missing flags"))?;
    let (exptime_tok, rest) =
        split_token(rest).ok_or(ParseError::Malformed("cas missing exptime"))?;
    let (bytes_tok, rest) = split_token(rest).ok_or(ParseError::Malformed("cas missing bytes"))?;
    let (cas_tok, rest) =
        split_token(rest).ok_or(ParseError::Malformed("cas missing cas_unique"))?;
    let flags = parse_u32(flags_tok, "flags")?;
    let exptime = parse_u32(exptime_tok, "exptime")?;
    let value_len = parse_u32(bytes_tok, "bytes")? as usize;
    let cas_unique = parse_u64(cas_tok, "cas_unique")?;
    let noreply = is_noreply(rest);

    let value_start = after_line;
    let value_end = value_start + value_len;
    if buf.len() < value_end + 2 {
        return Err(ParseError::PartialValue(value_len as u32));
    }
    if &buf[value_end..value_end + 2] != b"\r\n" {
        return Err(ParseError::Malformed("cas value not terminated by CRLF"));
    }
    Ok((
        Command::Cas {
            key,
            flags,
            exptime,
            cas_unique,
            value: &buf[value_start..value_end],
            noreply,
        },
        value_end + 2,
    ))
}

fn parse_delete(rest: &[u8], after_line: usize) -> Result<(Command<'_>, usize), ParseError> {
    let (key, rest) = split_token(rest).ok_or(ParseError::Malformed("delete missing key"))?;
    Ok((
        Command::Delete {
            key,
            noreply: is_noreply(rest),
        },
        after_line,
    ))
}

fn parse_counter(
    increment: bool,
    rest: &[u8],
    after_line: usize,
) -> Result<(Command<'_>, usize), ParseError> {
    let (key, rest) = split_token(rest).ok_or(ParseError::Malformed("counter missing key"))?;
    let (delta_tok, rest) =
        split_token(rest).ok_or(ParseError::Malformed("counter missing value"))?;
    let delta = parse_u64(delta_tok, "delta")?;
    Ok((
        Command::Counter {
            increment,
            key,
            delta,
            noreply: is_noreply(rest),
        },
        after_line,
    ))
}

fn parse_touch(rest: &[u8], after_line: usize) -> Result<(Command<'_>, usize), ParseError> {
    let (key, rest) = split_token(rest).ok_or(ParseError::Malformed("touch missing key"))?;
    let (exptime_tok, rest) =
        split_token(rest).ok_or(ParseError::Malformed("touch missing exptime"))?;
    let exptime = parse_u32(exptime_tok, "exptime")?;
    Ok((
        Command::Touch {
            key,
            exptime,
            noreply: is_noreply(rest),
        },
        after_line,
    ))
}

fn parse_flush_all(rest: &[u8], after_line: usize) -> Result<(Command<'_>, usize), ParseError> {
    if rest.is_empty() {
        return Ok((
            Command::FlushAll {
                delay: None,
                noreply: false,
            },
            after_line,
        ));
    }
    let (first, after_first) = split_token(rest).unwrap_or((rest, &[]));
    if first == b"noreply" {
        return Ok((
            Command::FlushAll {
                delay: None,
                noreply: true,
            },
            after_line,
        ));
    }
    let delay = parse_u32(first, "delay")?;
    Ok((
        Command::FlushAll {
            delay: Some(delay),
            noreply: is_noreply(after_first),
        },
        after_line,
    ))
}

#[inline]
fn is_noreply(rest: &[u8]) -> bool {
    let mut cursor = rest;
    while !cursor.is_empty() {
        let Some((tok, after)) = split_token(cursor) else {
            return false;
        };
        if tok == b"noreply" {
            return true;
        }
        if tok.is_empty() {
            return false;
        }
        cursor = after;
    }
    false
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn parses_get_single_key() {
        let buf = b"get mykey\r\n";
        let (cmd, used) = parse_command(buf).unwrap();
        assert_eq!(used, buf.len());
        match cmd {
            Command::Get { keys, gets } => {
                assert_eq!(keys, b"mykey");
                assert!(!gets);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parses_get_multi_key() {
        let buf = b"get a b c\r\n";
        let (cmd, _) = parse_command(buf).unwrap();
        match cmd {
            Command::Get { keys, .. } => assert_eq!(keys, b"a b c"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parses_set_with_value() {
        let buf = b"set foo 0 0 5\r\nhello\r\n";
        let (cmd, used) = parse_command(buf).unwrap();
        assert_eq!(used, buf.len());
        match cmd {
            Command::Store {
                mode,
                key,
                flags,
                exptime,
                value,
                noreply,
            } => {
                assert_eq!(mode, StoreMode::Set);
                assert_eq!(key, b"foo");
                assert_eq!(flags, 0);
                assert_eq!(exptime, 0);
                assert_eq!(value, b"hello");
                assert!(!noreply);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parses_set_noreply() {
        let buf = b"set k 3 60 3 noreply\r\nabc\r\n";
        let (cmd, _) = parse_command(buf).unwrap();
        match cmd {
            Command::Store {
                key,
                flags,
                exptime,
                value,
                noreply,
                ..
            } => {
                assert_eq!(key, b"k");
                assert_eq!(flags, 3);
                assert_eq!(exptime, 60);
                assert_eq!(value, b"abc");
                assert!(noreply);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parses_cas() {
        let buf = b"cas k 0 0 3 12345\r\nabc\r\n";
        let (cmd, _) = parse_command(buf).unwrap();
        match cmd {
            Command::Cas {
                key,
                cas_unique,
                value,
                ..
            } => {
                assert_eq!(key, b"k");
                assert_eq!(cas_unique, 12345);
                assert_eq!(value, b"abc");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parses_delete() {
        let buf = b"delete foo\r\n";
        let (cmd, _) = parse_command(buf).unwrap();
        match cmd {
            Command::Delete { key, noreply } => {
                assert_eq!(key, b"foo");
                assert!(!noreply);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parses_incr() {
        let buf = b"incr counter 5\r\n";
        let (cmd, _) = parse_command(buf).unwrap();
        match cmd {
            Command::Counter {
                increment,
                key,
                delta,
                ..
            } => {
                assert!(increment);
                assert_eq!(key, b"counter");
                assert_eq!(delta, 5);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn short_buffer_returns_short() {
        let buf = b"get incomplete";
        match parse_command(buf) {
            Err(ParseError::Short) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn partial_set_value_returns_partial() {
        let buf = b"set k 0 0 10\r\nabc"; // declares 10 bytes, only 3 supplied
        match parse_command(buf) {
            Err(ParseError::PartialValue(10)) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn unknown_verb_returns_error() {
        let buf = b"flarble x\r\n";
        match parse_command(buf) {
            Err(ParseError::UnknownCommand(verb)) => assert_eq!(verb, b"flarble"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parses_version() {
        let (cmd, _) = parse_command(b"version\r\n").unwrap();
        assert!(matches!(cmd, Command::Version));
    }

    #[test]
    fn parses_quit() {
        let (cmd, _) = parse_command(b"quit\r\n").unwrap();
        assert!(matches!(cmd, Command::Quit));
    }
}

#[cfg(feature = "memcached-codec-trait")]
pub mod codec_trait;
#[cfg(feature = "memcached-codec-trait")]
pub use codec_trait::{DatagramHeader, MemcachedDatagramCodec};

/// Sans-IO connection state machine (bytes in, [`Command`] out) — the
/// server-side idiom `proxima-memcached` drives. Mirrors
/// `crate::redis::connection`'s shape.
pub mod connection;
/// The memcached-over-`Pipe` contract: [`MemcachedRequest`] (the owned,
/// `'static` mirror of [`Command`]) plus its wire encoder. Mirrors
/// `crate::redis::pipe_contract`'s role.
pub mod pipe_contract;
/// Owned server-reply model (`STORED`/`VALUE ... END`/...) plus
/// `encode_reply`/`parse_reply` — the encode-direction counterpart
/// `parse_command` has none of, needed by both `proxima-memcached`'s
/// listener (encode) and client (parse).
pub mod reply;

pub use connection::{Advanced, Connection, Limits};
pub use pipe_contract::{MemcachedRequest, encode_request, verb};
pub use reply::{Reply, ReplyHint, StoredValue, encode_reply, parse_reply};
