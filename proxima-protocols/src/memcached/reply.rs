//! Sans-IO memcached reply model: the server -> client wire shapes
//! [`super::parse_command`] has no counterpart for (it only parses the
//! client -> server request direction). `parse_command` stays untouched —
//! this module adds only what the reply direction needs: an owned,
//! `'static` [`Reply`] (mirrors `crate::redis::RespValue`'s role as the
//! async client's carry) plus [`encode_reply`] / [`parse_reply`].
//!
//! Unlike RESP (one recursive [`crate::redis::Frame`] shape serves both
//! directions), memcached's reply framing is per-command and genuinely
//! ambiguous from the bytes alone: a bare `END\r\n` means "no rows" for
//! both a `get` miss and an empty `stats` reply. A real client resolves
//! this by remembering which command it just sent; [`parse_reply`] takes
//! that same context explicitly as a [`ReplyHint`] instead of guessing.
//!
//! No borrowed/zero-copy reply type exists here (unlike [`Command`]'s
//! borrowed fields) — a reply is a short status line or a handful of
//! `VALUE`/`STAT` blocks, so the zero-copy discipline that pays for itself
//! on `parse_command`'s hot ingest path has no comparable payoff on this
//! side; [`parse_reply`] allocates directly into the owned [`Reply`] it
//! returns.

use alloc::vec::Vec;

use super::{ParseError, find_crlf, parse_u32, parse_u64, split_token};

/// One `VALUE` block from a `get`/`gets` reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredValue {
    pub key: Vec<u8>,
    pub flags: u32,
    pub data: Vec<u8>,
    /// Present only for a `gets` reply (the CAS-aware variant of `get`).
    pub cas_unique: Option<u64>,
}

/// Owned server reply — the `'static` carry a client session hands back
/// once a reply's framing is fully parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reply {
    Stored,
    NotStored,
    Exists,
    NotFound,
    Deleted,
    Touched,
    Ok,
    Error,
    ClientError(Vec<u8>),
    ServerError(Vec<u8>),
    /// The new value after `incr`/`decr`.
    Counter(u64),
    Version(Vec<u8>),
    /// Zero or more `VALUE` blocks (a `get`/`gets` reply; empty on a miss).
    Values(Vec<StoredValue>),
    /// Zero or more `STAT name value` pairs (a `stats` reply).
    Stats(Vec<(Vec<u8>, Vec<u8>)>),
}

/// Which reply shape a caller expects, driven by the command it just
/// sent — resolves the `END\r\n`-alone ambiguity [`parse_reply`] cannot
/// resolve from the bytes on the wire (see module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplyHint {
    /// A status line, counter, version, or error — every command whose
    /// reply is never a `VALUE`/`STAT` block sequence.
    Simple,
    /// `get`/`gets`: zero or more `VALUE` blocks then `END`.
    Get,
    /// `stats`: zero or more `STAT` lines then `END`.
    Stats,
}

/// Parse one reply from the start of `buf`, per `hint`. Returns the reply
/// and the number of bytes consumed.
///
/// # Errors
/// [`ParseError::Short`] / [`ParseError::PartialValue`] when more bytes are
/// needed; the caller buffers and retries. [`ParseError::Malformed`] /
/// [`ParseError::InvalidInt`] on a genuine framing violation.
pub fn parse_reply(buf: &[u8], hint: ReplyHint) -> Result<(Reply, usize), ParseError> {
    match hint {
        ReplyHint::Simple => parse_simple(buf),
        ReplyHint::Get => parse_blocks(buf, BlockKind::Get),
        ReplyHint::Stats => parse_blocks(buf, BlockKind::Stats),
    }
}

fn parse_simple(buf: &[u8]) -> Result<(Reply, usize), ParseError> {
    let line_end = find_crlf(buf).ok_or(ParseError::Short)?;
    let line = &buf[..line_end];
    let consumed = line_end + 2;
    let (status, rest) = split_token(line).ok_or(ParseError::Malformed("empty reply line"))?;
    let reply = match status {
        b"STORED" => Reply::Stored,
        b"NOT_STORED" => Reply::NotStored,
        b"EXISTS" => Reply::Exists,
        b"NOT_FOUND" => Reply::NotFound,
        b"DELETED" => Reply::Deleted,
        b"TOUCHED" => Reply::Touched,
        b"OK" => Reply::Ok,
        b"ERROR" => Reply::Error,
        b"CLIENT_ERROR" => Reply::ClientError(rest.to_vec()),
        b"SERVER_ERROR" => Reply::ServerError(rest.to_vec()),
        b"VERSION" => Reply::Version(rest.to_vec()),
        digits if !digits.is_empty() && digits.iter().all(u8::is_ascii_digit) => {
            Reply::Counter(parse_u64(digits, "counter")?)
        }
        _ => return Err(ParseError::Malformed("unrecognized reply status")),
    };
    Ok((reply, consumed))
}

#[derive(Clone, Copy)]
enum BlockKind {
    Get,
    Stats,
}

/// Shared `0+ <row> ... END` parse for `get`/`gets` (`VALUE` rows) and
/// `stats` (`STAT` rows) — both end on a bare `END\r\n`, and both surface
/// `CLIENT_ERROR`/`SERVER_ERROR`/`ERROR` in place of the row list on a
/// server-side failure.
fn parse_blocks(buf: &[u8], kind: BlockKind) -> Result<(Reply, usize), ParseError> {
    let mut cursor = 0_usize;
    let mut values = Vec::new();
    let mut stats = Vec::new();
    loop {
        let remaining = &buf[cursor..];
        let line_end = find_crlf(remaining).ok_or(ParseError::Short)?;
        let line = &remaining[..line_end];
        let after_line = cursor + line_end + 2;
        let (verb, rest) = split_token(line).ok_or(ParseError::Malformed("empty reply line"))?;
        match verb {
            b"END" => {
                return Ok((
                    match kind {
                        BlockKind::Get => Reply::Values(values),
                        BlockKind::Stats => Reply::Stats(stats),
                    },
                    after_line,
                ));
            }
            b"VALUE" if matches!(kind, BlockKind::Get) => {
                let (data_end, value) = parse_value_row(buf, rest, after_line)?;
                values.push(value);
                cursor = data_end + 2;
            }
            b"STAT" if matches!(kind, BlockKind::Stats) => {
                let (name, value) =
                    split_token(rest).ok_or(ParseError::Malformed("stat missing name"))?;
                stats.push((name.to_vec(), value.to_vec()));
                cursor = after_line;
            }
            b"CLIENT_ERROR" => return Ok((Reply::ClientError(rest.to_vec()), after_line)),
            b"SERVER_ERROR" => return Ok((Reply::ServerError(rest.to_vec()), after_line)),
            b"ERROR" => return Ok((Reply::Error, after_line)),
            _ => return Err(ParseError::Malformed("unexpected reply row")),
        }
    }
}

/// `VALUE <key> <flags> <bytes> [<cas_unique>]\r\n<data>\r\n`, starting
/// just past the `VALUE ` token (`rest` is `<key> <flags> ...`).
fn parse_value_row(
    buf: &[u8],
    rest: &[u8],
    after_line: usize,
) -> Result<(usize, StoredValue), ParseError> {
    let (key, rest) = split_token(rest).ok_or(ParseError::Malformed("value missing key"))?;
    let (flags_tok, rest) =
        split_token(rest).ok_or(ParseError::Malformed("value missing flags"))?;
    let (bytes_tok, rest) =
        split_token(rest).ok_or(ParseError::Malformed("value missing bytes"))?;
    let flags = parse_u32(flags_tok, "flags")?;
    let value_len = parse_u32(bytes_tok, "bytes")? as usize;
    let cas_unique = if rest.is_empty() {
        None
    } else {
        let (cas_tok, _) = split_token(rest).ok_or(ParseError::Malformed("value cas_unique"))?;
        Some(parse_u64(cas_tok, "cas_unique")?)
    };

    let data_start = after_line;
    let data_end = data_start + value_len;
    if buf.len() < data_end + 2 {
        return Err(ParseError::PartialValue(value_len as u32));
    }
    if &buf[data_end..data_end + 2] != b"\r\n" {
        return Err(ParseError::Malformed("value not terminated by CRLF"));
    }
    Ok((
        data_end,
        StoredValue {
            key: key.to_vec(),
            flags,
            data: buf[data_start..data_end].to_vec(),
            cas_unique,
        },
    ))
}

fn write_decimal_u32(dest: &mut Vec<u8>, value: u32) {
    write_decimal_u64(dest, u64::from(value));
}

fn write_decimal_u64(dest: &mut Vec<u8>, mut value: u64) {
    let mut digits = [0_u8; 20];
    let mut count = 0;
    loop {
        digits[count] = b'0' + (value % 10) as u8;
        value /= 10;
        count += 1;
        if value == 0 {
            break;
        }
    }
    dest.extend(digits[..count].iter().rev());
}

/// Render one [`Reply`] as the ASCII line(s) [`parse_reply`] accepts back
/// (with the matching [`ReplyHint`]) — the listener's encode side.
pub fn encode_reply(reply: &Reply, dest: &mut Vec<u8>) {
    match reply {
        Reply::Stored => dest.extend_from_slice(b"STORED\r\n"),
        Reply::NotStored => dest.extend_from_slice(b"NOT_STORED\r\n"),
        Reply::Exists => dest.extend_from_slice(b"EXISTS\r\n"),
        Reply::NotFound => dest.extend_from_slice(b"NOT_FOUND\r\n"),
        Reply::Deleted => dest.extend_from_slice(b"DELETED\r\n"),
        Reply::Touched => dest.extend_from_slice(b"TOUCHED\r\n"),
        Reply::Ok => dest.extend_from_slice(b"OK\r\n"),
        Reply::Error => dest.extend_from_slice(b"ERROR\r\n"),
        Reply::ClientError(message) => {
            dest.extend_from_slice(b"CLIENT_ERROR ");
            dest.extend_from_slice(message);
            dest.extend_from_slice(b"\r\n");
        }
        Reply::ServerError(message) => {
            dest.extend_from_slice(b"SERVER_ERROR ");
            dest.extend_from_slice(message);
            dest.extend_from_slice(b"\r\n");
        }
        Reply::Counter(value) => {
            write_decimal_u64(dest, *value);
            dest.extend_from_slice(b"\r\n");
        }
        Reply::Version(version) => {
            dest.extend_from_slice(b"VERSION ");
            dest.extend_from_slice(version);
            dest.extend_from_slice(b"\r\n");
        }
        Reply::Values(values) => {
            for value in values {
                dest.extend_from_slice(b"VALUE ");
                dest.extend_from_slice(&value.key);
                dest.push(b' ');
                write_decimal_u32(dest, value.flags);
                dest.push(b' ');
                write_decimal_u32(dest, value.data.len() as u32);
                if let Some(cas_unique) = value.cas_unique {
                    dest.push(b' ');
                    write_decimal_u64(dest, cas_unique);
                }
                dest.extend_from_slice(b"\r\n");
                dest.extend_from_slice(&value.data);
                dest.extend_from_slice(b"\r\n");
            }
            dest.extend_from_slice(b"END\r\n");
        }
        Reply::Stats(stats) => {
            for (name, value) in stats {
                dest.extend_from_slice(b"STAT ");
                dest.extend_from_slice(name);
                dest.push(b' ');
                dest.extend_from_slice(value);
                dest.extend_from_slice(b"\r\n");
            }
            dest.extend_from_slice(b"END\r\n");
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn parses_stored() {
        let (reply, used) = parse_reply(b"STORED\r\n", ReplyHint::Simple).unwrap();
        assert_eq!(reply, Reply::Stored);
        assert_eq!(used, 8);
    }

    #[test]
    fn parses_client_error_with_message() {
        let (reply, _) =
            parse_reply(b"CLIENT_ERROR bad command line\r\n", ReplyHint::Simple).unwrap();
        assert_eq!(reply, Reply::ClientError(b"bad command line".to_vec()));
    }

    #[test]
    fn parses_counter_reply() {
        let (reply, _) = parse_reply(b"42\r\n", ReplyHint::Simple).unwrap();
        assert_eq!(reply, Reply::Counter(42));
    }

    #[test]
    fn parses_get_miss_as_empty_values() {
        let (reply, used) = parse_reply(b"END\r\n", ReplyHint::Get).unwrap();
        assert_eq!(reply, Reply::Values(Vec::new()));
        assert_eq!(used, 5);
    }

    #[test]
    fn parses_get_hit_with_one_value() {
        let (reply, used) =
            parse_reply(b"VALUE mykey 5 5\r\nhello\r\nEND\r\n", ReplyHint::Get).unwrap();
        assert_eq!(
            reply,
            Reply::Values(vec![StoredValue {
                key: b"mykey".to_vec(),
                flags: 5,
                data: b"hello".to_vec(),
                cas_unique: None,
            }])
        );
        assert_eq!(used, b"VALUE mykey 5 5\r\nhello\r\nEND\r\n".len());
    }

    #[test]
    fn parses_gets_hit_with_cas_unique() {
        let (reply, _) =
            parse_reply(b"VALUE k 0 3 77\r\nabc\r\nEND\r\n", ReplyHint::Get).unwrap();
        match reply {
            Reply::Values(values) => {
                assert_eq!(values.len(), 1);
                assert_eq!(values[0].cas_unique, Some(77));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parses_multiple_values_before_end() {
        let (reply, _) = parse_reply(
            b"VALUE a 0 1\r\nx\r\nVALUE b 0 1\r\ny\r\nEND\r\n",
            ReplyHint::Get,
        )
        .unwrap();
        match reply {
            Reply::Values(values) => {
                assert_eq!(values.len(), 2);
                assert_eq!(values[0].key, b"a");
                assert_eq!(values[1].key, b"b");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn get_partial_value_returns_short() {
        let outcome = parse_reply(b"VALUE k 0 10\r\nabc", ReplyHint::Get);
        assert!(matches!(outcome, Err(ParseError::PartialValue(10))));
    }

    #[test]
    fn parses_stats_rows() {
        let (reply, _) =
            parse_reply(b"STAT pid 123\r\nSTAT uptime 45\r\nEND\r\n", ReplyHint::Stats).unwrap();
        assert_eq!(
            reply,
            Reply::Stats(vec![
                (b"pid".to_vec(), b"123".to_vec()),
                (b"uptime".to_vec(), b"45".to_vec()),
            ])
        );
    }

    #[test]
    fn empty_stats_is_bare_end() {
        let (reply, _) = parse_reply(b"END\r\n", ReplyHint::Stats).unwrap();
        assert_eq!(reply, Reply::Stats(Vec::new()));
    }

    #[test]
    fn round_trips_stored() {
        let mut out = Vec::new();
        encode_reply(&Reply::Stored, &mut out);
        let (parsed, used) = parse_reply(&out, ReplyHint::Simple).unwrap();
        assert_eq!(parsed, Reply::Stored);
        assert_eq!(used, out.len());
    }

    #[test]
    fn round_trips_values_with_cas() {
        let reply = Reply::Values(vec![StoredValue {
            key: b"mykey".to_vec(),
            flags: 9,
            data: b"payload".to_vec(),
            cas_unique: Some(555),
        }]);
        let mut out = Vec::new();
        encode_reply(&reply, &mut out);
        let (parsed, used) = parse_reply(&out, ReplyHint::Get).unwrap();
        assert_eq!(parsed, reply);
        assert_eq!(used, out.len());
    }

    #[test]
    fn round_trips_stats() {
        let reply = Reply::Stats(vec![(b"curr_items".to_vec(), b"3".to_vec())]);
        let mut out = Vec::new();
        encode_reply(&reply, &mut out);
        let (parsed, used) = parse_reply(&out, ReplyHint::Stats).unwrap();
        assert_eq!(parsed, reply);
        assert_eq!(used, out.len());
    }

    #[test]
    fn round_trips_counter() {
        let mut out = Vec::new();
        encode_reply(&Reply::Counter(9001), &mut out);
        assert_eq!(out, b"9001\r\n");
        let (parsed, _) = parse_reply(&out, ReplyHint::Simple).unwrap();
        assert_eq!(parsed, Reply::Counter(9001));
    }

    #[test]
    fn malformed_status_line_is_hard_error() {
        let outcome = parse_reply(b"BOGUS\r\n", ReplyHint::Simple);
        assert!(matches!(outcome, Err(ParseError::Malformed(_))));
    }
}
