//! Sans-IO RESP connection state machine — bytes in, [`Frame`] out.
//!
//! Mirrors [`crate::http1_codec::h1_connection::Connection`]'s
//! `feed_bytes`/`advance`/`consume` shape: one growing read buffer, a
//! cursor so pipelined command bytes don't memcpy, and a typed `Advanced`
//! outcome the driver matches on. No socket, no tokio, no `.await`
//! anywhere in this file — the I/O edge (`proxima-redis`'s driver) owns
//! reading bytes off the wire and feeding them in.
//!
//! Also owns the two pieces of protocol state a bare frame parser cannot:
//! the CLIENT REPLY mode gate ([`ConnMode`] — a connection that has
//! SUBSCRIBEd may only run the pub/sub-family commands plus PING/QUIT/RESET
//! until it UNSUBSCRIBEs from everything) and the DoS guard `parse`'s
//! `parse_blob` does not itself apply: `parse_blob` trusts an
//! attacker-controlled length with no cap of its own (it only checks the
//! buffer already holds the declared total), so an attacker who declares a
//! huge length and trickles bytes would otherwise grow this connection's
//! buffer without bound. [`Connection::advance`] catches that once the
//! buffered-but-still-incomplete bytes exceed
//! [`Limits::max_message_bytes`], mirroring pgwire's `MessageTooLarge` guard
//! (`proxima-pgwire/src/connection.rs`'s `config.max_message_bytes` check
//! ahead of every `read_some`).

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use super::{Frame, ParseError, parse};

/// A connection stays under this many buffered-but-unparsed bytes before a
/// still-incomplete frame is treated as an oversized message. 16 MiB matches
/// `proxima-pgwire`'s default `max_message_bytes` — generous for any
/// legitimate GET/SET-shaped command, small enough to bound a malicious
/// connection's memory footprint.
const DEFAULT_MAX_MESSAGE_BYTES: usize = 16 * 1024 * 1024;

/// Once the consumed prefix exceeds this many bytes, [`Connection::consume`]
/// compacts the buffer (drains the dead prefix) instead of just moving the
/// cursor — mirrors `h1_connection::Connection::reset_for_next_request`'s
/// same threshold-triggered compaction so a long-lived pipelined connection
/// doesn't grow its buffer unbounded.
const COMPACT_THRESHOLD_BYTES: usize = 8 * 1024;

/// Byte caps a [`Connection`] enforces. `max_message_bytes` is the DoS guard
/// described in the module doc.
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

/// CLIENT REPLY mode: `Command` admits every command; `Subscriber` (entered
/// by a successful SUBSCRIBE/PSUBSCRIBE) admits only the pub/sub family plus
/// PING/QUIT/RESET, per the real Redis subscriber-context restriction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnMode {
    Command,
    Subscriber,
}

/// Typed outcome of [`Connection::advance`]. `Command`'s `frame` borrows
/// from the connection's internal buffer (same borrow shape as
/// `h1_connection::Advanced<'a>`) — the driver must extract whatever it
/// needs (verb, args) before calling [`Connection::consume`] or `advance`
/// again.
pub enum Advanced<'a> {
    /// The buffer holds a prefix of a frame; read more bytes and retry.
    NeedMore,
    /// One full RESP frame parsed. `consumed` is the byte length to pass to
    /// [`Connection::consume`] once the driver is done with `frame`.
    Command { frame: Frame<'a>, consumed: usize },
    /// The buffered bytes violate RESP framing. `reason` is the malformed-
    /// frame detail; `consumed` is always 0 (a framing violation leaves no
    /// trustworthy frame boundary to skip past) — the driver closes the
    /// connection rather than trying to resync.
    ProtocolError { reason: &'static str, consumed: usize },
    /// A still-incomplete frame already exceeds
    /// [`Limits::max_message_bytes`] — the DoS guard tripped. The driver
    /// closes the connection.
    MessageTooLarge,
}

impl core::fmt::Debug for Advanced<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let variant = match self {
            Self::NeedMore => "NeedMore",
            Self::Command { .. } => "Command",
            Self::ProtocolError { .. } => "ProtocolError",
            Self::MessageTooLarge => "MessageTooLarge",
        };
        formatter
            .debug_struct("Advanced")
            .field("variant", &variant)
            .finish_non_exhaustive()
    }
}

/// Commands a subscriber-mode connection may still run: the pub/sub family
/// (so it can keep managing its own subscriptions) plus PING/QUIT/RESET —
/// exactly the real Redis subscriber-context allowance.
const SUBSCRIBER_SAFE_VERBS: &[&[u8]] = &[
    b"SUBSCRIBE",
    b"UNSUBSCRIBE",
    b"PSUBSCRIBE",
    b"PUNSUBSCRIBE",
    b"SSUBSCRIBE",
    b"SUNSUBSCRIBE",
    b"PING",
    b"QUIT",
    b"RESET",
];

/// Sans-IO RESP connection state machine.
pub struct Connection {
    buffer: Vec<u8>,
    /// Logical start of the not-yet-consumed region. Advances on
    /// [`Self::consume`]; pipelined command bytes past it are re-parsed
    /// in place, no memcpy.
    cursor: usize,
    mode: ConnMode,
    subscriptions: BTreeSet<Vec<u8>>,
    psubscriptions: BTreeSet<Vec<u8>>,
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
            mode: ConnMode::Command,
            subscriptions: BTreeSet::new(),
            psubscriptions: BTreeSet::new(),
            limits,
        }
    }

    /// Append bytes read off the wire.
    pub fn feed_bytes(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    /// Drive the state machine one step: try to parse one RESP frame from
    /// the unconsumed buffer region.
    pub fn advance(&mut self) -> Advanced<'_> {
        match parse(&self.buffer[self.cursor..]) {
            Ok((frame, consumed)) => Advanced::Command { frame, consumed },
            Err(ParseError::Malformed(reason)) => Advanced::ProtocolError { reason, consumed: 0 },
            Err(ParseError::NeedMore) => {
                if self.buffer.len() - self.cursor > self.limits.max_message_bytes {
                    Advanced::MessageTooLarge
                } else {
                    Advanced::NeedMore
                }
            }
        }
    }

    /// Advance past a parsed frame's bytes (the `consumed` a
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

    /// Pure mode-gate predicate: does the current [`ConnMode`] admit
    /// `verb`? `Command` mode admits everything; `Subscriber` mode admits
    /// only [`SUBSCRIBER_SAFE_VERBS`].
    #[must_use]
    pub fn admits(&self, verb: &[u8]) -> bool {
        match self.mode {
            ConnMode::Command => true,
            ConnMode::Subscriber => SUBSCRIBER_SAFE_VERBS
                .iter()
                .any(|allowed| allowed.eq_ignore_ascii_case(verb)),
        }
    }

    #[must_use]
    pub fn mode(&self) -> ConnMode {
        self.mode
    }

    /// Force the connection into `Subscriber` mode. Called by the driver
    /// once a SUBSCRIBE/PSUBSCRIBE reply has been queued; also flipped
    /// implicitly by [`Self::subscribe`]/[`Self::psubscribe`].
    pub fn enter_subscriber_mode(&mut self) {
        self.mode = ConnMode::Subscriber;
    }

    /// Record an exact-channel subscription; returns whether it is new.
    /// Enters `Subscriber` mode.
    pub fn subscribe(&mut self, channel: Vec<u8>) -> bool {
        let inserted = self.subscriptions.insert(channel);
        self.enter_subscriber_mode();
        inserted
    }

    /// Remove an exact-channel subscription; returns whether it existed.
    /// Falls back to `Command` mode once every subscription (exact and
    /// pattern) is gone.
    pub fn unsubscribe(&mut self, channel: &[u8]) -> bool {
        let removed = self.subscriptions.remove(channel);
        self.exit_subscriber_mode_if_idle();
        removed
    }

    /// Record a pattern subscription; returns whether it is new. Enters
    /// `Subscriber` mode.
    pub fn psubscribe(&mut self, pattern: Vec<u8>) -> bool {
        let inserted = self.psubscriptions.insert(pattern);
        self.enter_subscriber_mode();
        inserted
    }

    /// Remove a pattern subscription; returns whether it existed. Falls
    /// back to `Command` mode once every subscription is gone.
    pub fn punsubscribe(&mut self, pattern: &[u8]) -> bool {
        let removed = self.psubscriptions.remove(pattern);
        self.exit_subscriber_mode_if_idle();
        removed
    }

    fn exit_subscriber_mode_if_idle(&mut self) {
        if self.subscriptions.is_empty() && self.psubscriptions.is_empty() {
            self.mode = ConnMode::Command;
        }
    }

    /// Total subscription count (exact + pattern) — the real Redis
    /// `SUBSCRIBE`/`UNSUBSCRIBE` reply's third element.
    #[must_use]
    pub fn subscription_count(&self) -> usize {
        self.subscriptions.len() + self.psubscriptions.len()
    }

    /// The exact channels this connection is currently subscribed to.
    #[must_use]
    pub fn subscriptions(&self) -> &BTreeSet<Vec<u8>> {
        &self.subscriptions
    }

    /// The patterns this connection is currently subscribed to.
    #[must_use]
    pub fn psubscriptions(&self) -> &BTreeSet<Vec<u8>> {
        &self.psubscriptions
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use alloc::vec;

    fn ping_command() -> Vec<u8> {
        let mut out = Vec::new();
        crate::redis::encode_command(&[b"PING"], &mut out);
        out
    }

    fn get_command(key: &str) -> Vec<u8> {
        let mut out = Vec::new();
        crate::redis::encode_command(&[b"GET", key.as_bytes()], &mut out);
        out
    }

    #[test]
    fn partial_frame_returns_need_more() {
        let mut connection = Connection::new();
        connection.feed_bytes(b"*1\r\n$4\r\nPIN");
        assert!(matches!(connection.advance(), Advanced::NeedMore));
    }

    #[test]
    fn complete_frame_returns_command_with_consumed_length() {
        let mut connection = Connection::new();
        let wire = ping_command();
        connection.feed_bytes(&wire);
        match connection.advance() {
            Advanced::Command { frame, consumed } => {
                assert_eq!(consumed, wire.len());
                assert_eq!(
                    frame,
                    Frame::Array(vec![Frame::BlobString(b"PING")])
                );
            }
            other => panic!("expected Command, got {other:?}"),
        }
    }

    #[test]
    fn malformed_bytes_return_protocol_error_with_zero_consumed() {
        let mut connection = Connection::new();
        connection.feed_bytes(b"@nope\r\n");
        match connection.advance() {
            Advanced::ProtocolError { consumed, .. } => assert_eq!(consumed, 0),
            other => panic!("expected ProtocolError, got {other:?}"),
        }
    }

    #[test]
    fn oversized_incomplete_frame_trips_message_too_large() {
        let mut connection = Connection::with_limits(Limits { max_message_bytes: 10 });
        // a bulk string declaring a huge length, but only a few payload
        // bytes actually sent — NeedMore forever unless the guard trips.
        // 16 buffered bytes > the 10-byte cap.
        connection.feed_bytes(b"$1000000000\r\nabc");
        assert!(matches!(connection.advance(), Advanced::MessageTooLarge));
    }

    #[test]
    fn small_incomplete_frame_stays_need_more_under_the_cap() {
        let mut connection = Connection::with_limits(Limits { max_message_bytes: 16 });
        connection.feed_bytes(b"$5\r\nhel");
        assert!(matches!(connection.advance(), Advanced::NeedMore));
    }

    #[test]
    fn consume_advances_cursor_and_pipelined_bytes_reparse_in_place() {
        let mut connection = Connection::new();
        let first = ping_command();
        let second = get_command("k");
        connection.feed_bytes(&first);
        connection.feed_bytes(&second);

        let Advanced::Command { consumed, .. } = connection.advance() else {
            panic!("expected first Command");
        };
        connection.consume(consumed);

        match connection.advance() {
            Advanced::Command { frame, .. } => {
                assert_eq!(
                    frame,
                    Frame::Array(vec![Frame::BlobString(b"GET"), Frame::BlobString(b"k")])
                );
            }
            other => panic!("expected second Command, got {other:?}"),
        }
    }

    #[test]
    fn consume_clears_buffer_once_fully_drained() {
        let mut connection = Connection::new();
        let wire = ping_command();
        connection.feed_bytes(&wire);
        let Advanced::Command { consumed, .. } = connection.advance() else {
            panic!("expected Command");
        };
        connection.consume(consumed);
        assert!(matches!(connection.advance(), Advanced::NeedMore));
    }

    #[test]
    fn command_mode_admits_every_verb() {
        let connection = Connection::new();
        assert!(connection.admits(b"GET"));
        assert!(connection.admits(b"SUBSCRIBE"));
    }

    #[test]
    fn subscribe_enters_subscriber_mode_and_gates_admission() {
        let mut connection = Connection::new();
        connection.subscribe(b"news".to_vec());

        assert_eq!(connection.mode(), ConnMode::Subscriber);
        assert!(connection.admits(b"PING"));
        assert!(connection.admits(b"SUBSCRIBE"));
        assert!(connection.admits(b"unsubscribe"), "case-insensitive");
        assert!(!connection.admits(b"GET"), "GET is not subscriber-safe");
    }

    #[test]
    fn unsubscribing_the_last_channel_returns_to_command_mode() {
        let mut connection = Connection::new();
        connection.subscribe(b"news".to_vec());
        assert!(connection.unsubscribe(b"news"));

        assert_eq!(connection.mode(), ConnMode::Command);
        assert!(connection.admits(b"GET"));
    }

    #[test]
    fn psubscribe_and_punsubscribe_track_pattern_mode_independently() {
        let mut connection = Connection::new();
        connection.subscribe(b"news".to_vec());
        connection.psubscribe(b"chat.*".to_vec());
        assert_eq!(connection.subscription_count(), 2);

        connection.unsubscribe(b"news");
        assert_eq!(
            connection.mode(),
            ConnMode::Subscriber,
            "a live pattern subscription keeps the connection gated"
        );

        connection.punsubscribe(b"chat.*");
        assert_eq!(connection.mode(), ConnMode::Command);
        assert_eq!(connection.subscription_count(), 0);
    }

    #[test]
    fn unsubscribe_of_an_absent_channel_reports_false() {
        let mut connection = Connection::new();
        assert!(!connection.unsubscribe(b"never-subscribed"));
    }
}
