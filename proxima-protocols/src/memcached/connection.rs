//! Sans-IO memcached (text protocol) connection state machine — bytes in,
//! [`Command`] out.
//!
//! Mirrors [`crate::redis::connection::Connection`]'s `feed_bytes`/
//! `advance`/`consume` shape: one growing read buffer, a cursor so
//! pipelined command bytes don't memcpy, and a typed [`Advanced`] outcome
//! the driver matches on. No socket, no tokio, no `.await` anywhere in
//! this file — the I/O edge (`proxima-memcached`'s driver) owns reading
//! bytes off the wire and feeding them in.
//!
//! Simpler than the RESP connection it mirrors: memcached has no
//! CLIENT-REPLY-mode gate (no pub/sub, so no equivalent of RESP's
//! `ConnMode::Subscriber`) — the only protocol state this FSM owns beyond
//! the buffer is the DoS guard. [`super::parse_command`] itself trusts an
//! attacker-controlled `<bytes>` length on every storage command with no
//! cap of its own (it only checks the buffer already holds the declared
//! total, surfacing [`ParseError::PartialValue`] otherwise); an attacker
//! who declares a huge length and trickles bytes would otherwise grow this
//! connection's buffer without bound. [`Connection::advance`] catches that
//! once the buffered-but-still-incomplete bytes exceed
//! [`Limits::max_message_bytes`], the same guard shape
//! `crate::redis::connection::Connection` and `proxima-pgwire` apply.

use alloc::vec::Vec;

use super::{Command, ParseError, parse_command};

/// A connection stays under this many buffered-but-unparsed bytes before a
/// still-incomplete command is treated as an oversized message. Matches
/// `crate::redis::connection`'s default.
const DEFAULT_MAX_MESSAGE_BYTES: usize = 16 * 1024 * 1024;

/// Once the consumed prefix exceeds this many bytes, [`Connection::consume`]
/// compacts the buffer (drains the dead prefix) instead of just moving the
/// cursor — mirrors `crate::redis::connection::Connection`'s identical
/// threshold-triggered compaction.
const COMPACT_THRESHOLD_BYTES: usize = 8 * 1024;

/// Byte caps a [`Connection`] enforces. `max_message_bytes` is the DoS
/// guard described in the module doc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    pub max_message_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_message_bytes: DEFAULT_MAX_MESSAGE_BYTES,
        }
    }
}

/// Typed outcome of [`Connection::advance`]. `Command`'s `command` borrows
/// from the connection's internal buffer (same borrow shape as
/// `crate::redis::connection::Advanced::Command`) — the driver must extract
/// whatever it needs before calling [`Connection::consume`] or `advance`
/// again.
#[derive(Debug)]
pub enum Advanced<'a> {
    /// The buffer holds a prefix of a command; read more bytes and retry.
    NeedMore,
    /// One full command parsed. `consumed` is the byte length to pass to
    /// [`Connection::consume`] once the driver is done with `command`.
    Command { command: Command<'a>, consumed: usize },
    /// The buffered bytes violate the text protocol's framing (unknown
    /// verb, malformed field, bad integer). The driver closes the
    /// connection rather than trying to resync — there is no trustworthy
    /// command boundary to skip past.
    ProtocolError { error: ParseError },
    /// A still-incomplete command already exceeds
    /// [`Limits::max_message_bytes`] — the DoS guard tripped. The driver
    /// closes the connection.
    MessageTooLarge,
}

/// Sans-IO memcached connection state machine.
pub struct Connection {
    buffer: Vec<u8>,
    /// Logical start of the not-yet-consumed region. Advances on
    /// [`Self::consume`]; pipelined command bytes past it are re-parsed in
    /// place, no memcpy.
    cursor: usize,
    limits: Limits,
}

impl Default for Connection {
    fn default() -> Self {
        Self::new()
    }
}

impl Connection {
    #[must_use]
    pub fn new() -> Self {
        Self::with_limits(Limits::default())
    }

    #[must_use]
    pub fn with_limits(limits: Limits) -> Self {
        Self {
            buffer: Vec::new(),
            cursor: 0,
            limits,
        }
    }

    /// Append bytes read off the wire.
    pub fn feed_bytes(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    /// Drive the state machine one step: try to parse one command from the
    /// unconsumed buffer region. [`ParseError::Short`] (no CRLF yet) and
    /// [`ParseError::PartialValue`] (the declared value length exceeds
    /// what's buffered) both mean "need more bytes", not a protocol
    /// violation — the DoS guard below is what turns an unbounded wait
    /// into a hard error.
    pub fn advance(&mut self) -> Advanced<'_> {
        match parse_command(&self.buffer[self.cursor..]) {
            Ok((command, consumed)) => Advanced::Command { command, consumed },
            Err(ParseError::Short | ParseError::PartialValue(_)) => {
                if self.buffer.len() - self.cursor > self.limits.max_message_bytes {
                    Advanced::MessageTooLarge
                } else {
                    Advanced::NeedMore
                }
            }
            Err(error) => Advanced::ProtocolError { error },
        }
    }

    /// Advance past a parsed command's bytes (the `consumed` an
    /// [`Advanced::Command`] carried). Compacts the buffer once the
    /// consumed prefix grows past [`COMPACT_THRESHOLD_BYTES`], or clears it
    /// outright once every buffered byte is consumed, so a long-lived
    /// pipelined connection's buffer stays bounded.
    pub fn consume(&mut self, amount: usize) {
        self.cursor += amount;
        if self.cursor >= self.buffer.len() {
            self.buffer.clear();
            self.cursor = 0;
        } else if self.cursor > COMPACT_THRESHOLD_BYTES {
            self.buffer.drain(..self.cursor);
            self.cursor = 0;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::memcached::StoreMode;

    fn set_command(key: &str, value: &str) -> Vec<u8> {
        alloc::format!("set {key} 0 0 {}\r\n{value}\r\n", value.len()).into_bytes()
    }

    fn get_command(key: &str) -> Vec<u8> {
        alloc::format!("get {key}\r\n").into_bytes()
    }

    #[test]
    fn partial_line_returns_need_more() {
        let mut connection = Connection::new();
        connection.feed_bytes(b"get incomp");
        assert!(matches!(connection.advance(), Advanced::NeedMore));
    }

    #[test]
    fn partial_value_returns_need_more() {
        let mut connection = Connection::new();
        connection.feed_bytes(b"set k 0 0 10\r\nabc");
        assert!(matches!(connection.advance(), Advanced::NeedMore));
    }

    #[test]
    fn complete_get_returns_command_with_consumed_length() {
        let mut connection = Connection::new();
        let wire = get_command("mykey");
        connection.feed_bytes(&wire);
        match connection.advance() {
            Advanced::Command { command, consumed } => {
                assert_eq!(consumed, wire.len());
                match command {
                    Command::Get { keys, gets } => {
                        assert_eq!(keys, b"mykey");
                        assert!(!gets);
                    }
                    other => panic!("unexpected: {other:?}"),
                }
            }
            other => panic!("expected Command, got {other:?}"),
        }
    }

    #[test]
    fn unknown_verb_returns_protocol_error() {
        let mut connection = Connection::new();
        connection.feed_bytes(b"flarble x\r\n");
        match connection.advance() {
            Advanced::ProtocolError {
                error: ParseError::UnknownCommand(verb),
            } => assert_eq!(verb.as_bytes(), b"flarble"),
            other => panic!("expected ProtocolError, got {other:?}"),
        }
    }

    #[test]
    fn oversized_incomplete_value_trips_message_too_large() {
        let mut connection = Connection::with_limits(Limits { max_message_bytes: 10 });
        connection.feed_bytes(b"set k 0 0 1000000000\r\nabc");
        assert!(matches!(connection.advance(), Advanced::MessageTooLarge));
    }

    #[test]
    fn small_incomplete_value_stays_need_more_under_the_cap() {
        let mut connection = Connection::with_limits(Limits { max_message_bytes: 32 });
        connection.feed_bytes(b"set k 0 0 5\r\nhel");
        assert!(matches!(connection.advance(), Advanced::NeedMore));
    }

    #[test]
    fn consume_advances_cursor_and_pipelined_bytes_reparse_in_place() {
        let mut connection = Connection::new();
        let first = set_command("a", "1");
        let second = get_command("a");
        connection.feed_bytes(&first);
        connection.feed_bytes(&second);

        let Advanced::Command { consumed, command } = connection.advance() else {
            panic!("expected first Command");
        };
        assert!(matches!(
            command,
            Command::Store {
                mode: StoreMode::Set,
                ..
            }
        ));
        connection.consume(consumed);

        match connection.advance() {
            Advanced::Command { command, .. } => match command {
                Command::Get { keys, .. } => assert_eq!(keys, b"a"),
                other => panic!("unexpected: {other:?}"),
            },
            other => panic!("expected second Command, got {other:?}"),
        }
    }

    #[test]
    fn consume_clears_buffer_once_fully_drained() {
        let mut connection = Connection::new();
        let wire = get_command("k");
        connection.feed_bytes(&wire);
        let Advanced::Command { consumed, .. } = connection.advance() else {
            panic!("expected Command");
        };
        connection.consume(consumed);
        assert!(matches!(connection.advance(), Advanced::NeedMore));
    }
}
