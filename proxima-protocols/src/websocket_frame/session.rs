//! Sans-IO RFC 6455 SERVER session state machine — bytes in, [`Event`]
//! out; queued reply bytes drained via [`Session::poll_transmit`].
//!
//! Mirrors [`crate::redis::connection::Connection`]'s `feed_bytes` /
//! `advance` shape (a byte-STREAM sans-IO connection, not a per-datagram
//! one — WebSocket rides TCP, so the h1/redis shape fits, not
//! `proxima_listen`'s `DatagramProtocol`, which exists for connectionless
//! transports). Unlike `redis::Connection::advance`, [`Session::poll_event`]
//! never needs a separate `consume()` step: a reassembled [`Message`] borrows
//! either straight from the wire buffer (the common single-frame case, zero
//! copy) or from `Session`'s own `completed_message` field (the fragmented
//! case, one copy per completed message — unavoidable, since fragments are
//! not contiguous in the wire buffer once a control frame interleaves per
//! RFC 6455 §5.4) — both borrows are tied to `&mut self`, so the borrow
//! checker itself enforces "read the event before calling `poll_event`
//! again," the same discipline `redis::Advanced<'a>` gets from its explicit
//! `consumed` field.
//!
//! Built entirely on the existing frame codec
//! ([`super::parse_frame`], [`super::encode_header`],
//! [`super::unmask_in_place`]) — this module adds none of its own framing,
//! only the RFC business rules a bare frame parser cannot express:
//! fragmentation reassembly, control-frame automation (PING -> automatic
//! PONG, CLOSE -> the closing handshake), §5.1 masking enforcement (a
//! server MUST reject an unmasked client frame and MUST NOT mask its own
//! frames), §8.1 UTF-8 validation, and §7.4 close-code semantics.
//!
//! Auto-generated replies (a PONG echo, a CLOSE echo, a CLOSE triggered by
//! a protocol violation) are control frames, capped at 127 bytes on the
//! wire (2-byte header + <=125-byte payload per §5.5) — queuing them costs
//! one small, bounded `Vec` allocation on this COLD path (RFC violations
//! and keepalive pings are not the hot path; the existing
//! [`super::encode_header`] signature already takes `&mut Vec<u8>`, so
//! reusing it as-is here — rather than hand-rolling a second, fixed-buffer
//! encoder just to dodge one cold-path alloc — is the RISC-reuse answer,
//! principle 1). The hot path (a complete unfragmented text/binary
//! message) allocates nothing: [`Event::Message`] borrows the wire buffer
//! directly.
//!
//! No timers, no sockets, no async, no runtime symbols anywhere in this
//! file — a caller on prime, tokio, a bare epoll loop, or a fuzzer drives
//! it identically: `feed` inbound bytes, `poll_event` in a loop until
//! `Event::Incomplete`, `poll_transmit` to drain any queued reply, repeat.

use alloc::vec::Vec;
use core::str;

use super::{Opcode, ParseError, encode_header, parse_frame, unmask_in_place};

/// A connection stays under this many buffered bytes of a single
/// still-incomplete frame before it is treated as oversized (RFC 6455
/// §7.4.1 code 1009). Chosen so one legitimate frame comfortably fits
/// (1 MiB) while an attacker declaring a multi-gigabyte frame length is
/// rejected before the buffer grows to hold it.
pub const DEFAULT_MAX_FRAME_BYTES: usize = 1024 * 1024;

/// Total bytes a reassembled (possibly multi-fragment) message may reach
/// before it is treated as oversized. Matches
/// `crate::redis::connection::DEFAULT_MAX_MESSAGE_BYTES` — same DoS-guard
/// register, same default magnitude.
pub const DEFAULT_MAX_MESSAGE_BYTES: usize = 16 * 1024 * 1024;

/// Once the consumed prefix of the wire buffer exceeds this many bytes,
/// [`Session::poll_event`] compacts (drains the dead prefix) instead of
/// leaving it in place — mirrors `redis::connection::Connection`'s
/// identical threshold so a long-lived connection's buffer stays bounded.
const COMPACT_THRESHOLD_BYTES: usize = 8 * 1024;

/// Byte caps a [`Session`] enforces. `max_frame_bytes` bounds any single
/// frame's payload (including one fragment of a reassembled message);
/// `max_message_bytes` bounds the total of a (possibly multi-fragment)
/// reassembled message. A sane configuration keeps
/// `max_frame_bytes <= max_message_bytes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    pub max_frame_bytes: usize,
    pub max_message_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_frame_bytes: DEFAULT_MAX_FRAME_BYTES,
            max_message_bytes: DEFAULT_MAX_MESSAGE_BYTES,
        }
    }
}

/// RFC 6455 §7.4.1 WebSocket close status code. The named variants are
/// the codes the base RFC (§7.4.1) and its IANA-registered follow-ons
/// (1012-1014) define; [`CloseCode::Other`] carries any registered
/// (3000-3999) or private-use (4000-4999) code a caller passes to
/// [`Session::close`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloseCode {
    /// 1000 — normal, successful completion.
    Normal,
    /// 1001 — endpoint going away (server shutdown, browser navigation).
    GoingAway,
    /// 1002 — a framing/protocol violation.
    ProtocolError,
    /// 1003 — endpoint received a data type it cannot accept.
    Unsupported,
    /// 1005 — reserved: local-use sentinel for "no status code was
    /// present." MUST NOT be sent on the wire; [`Session`] only ever
    /// produces it as the observed code when a peer's Close frame carried
    /// an empty payload.
    NoStatus,
    /// 1006 — reserved: local-use sentinel for "the connection dropped
    /// without a close handshake." Never sent or observed on the wire by
    /// this module; listed for completeness with the RFC's registry.
    Abnormal,
    /// 1007 — payload data does not match its declared type (e.g.
    /// invalid UTF-8 in a text message).
    InvalidPayload,
    /// 1008 — a generic policy violation not covered by a more specific
    /// code.
    PolicyViolation,
    /// 1009 — message too big to process.
    MessageTooBig,
    /// 1010 — client expected the server to negotiate an extension.
    MandatoryExtension,
    /// 1011 — server encountered an unexpected condition. Primarily for
    /// the application layer above [`Session`] to report its own
    /// failures via [`Session::close`]; the RFC framing layer itself
    /// only reaches this defensively (see [`Session::poll_event`]).
    InternalError,
    /// 1012 — server is restarting (IANA-registered, not in the base RFC).
    ServiceRestart,
    /// 1013 — server is temporarily overloaded (IANA-registered).
    TryAgainLater,
    /// 1014 — gateway/proxy got an invalid response from upstream
    /// (IANA-registered).
    BadGateway,
    /// 1015 — reserved: local-use sentinel for "TLS handshake failed."
    /// Never sent or observed on the wire.
    TlsHandshake,
    /// Any other code — registered (3000-3999) or private-use
    /// (4000-4999), or an out-of-range/undefined value observed from a
    /// peer (see [`CloseCode::valid_on_wire`]).
    Other(u16),
}

impl CloseCode {
    #[must_use]
    pub const fn as_u16(self) -> u16 {
        match self {
            Self::Normal => 1000,
            Self::GoingAway => 1001,
            Self::ProtocolError => 1002,
            Self::Unsupported => 1003,
            Self::NoStatus => 1005,
            Self::Abnormal => 1006,
            Self::InvalidPayload => 1007,
            Self::PolicyViolation => 1008,
            Self::MessageTooBig => 1009,
            Self::MandatoryExtension => 1010,
            Self::InternalError => 1011,
            Self::ServiceRestart => 1012,
            Self::TryAgainLater => 1013,
            Self::BadGateway => 1014,
            Self::TlsHandshake => 1015,
            Self::Other(code) => code,
        }
    }

    #[must_use]
    pub const fn from_u16(code: u16) -> Self {
        match code {
            1000 => Self::Normal,
            1001 => Self::GoingAway,
            1002 => Self::ProtocolError,
            1003 => Self::Unsupported,
            1005 => Self::NoStatus,
            1006 => Self::Abnormal,
            1007 => Self::InvalidPayload,
            1008 => Self::PolicyViolation,
            1009 => Self::MessageTooBig,
            1010 => Self::MandatoryExtension,
            1011 => Self::InternalError,
            1012 => Self::ServiceRestart,
            1013 => Self::TryAgainLater,
            1014 => Self::BadGateway,
            1015 => Self::TlsHandshake,
            other => Self::Other(other),
        }
    }

    /// RFC 6455 §7.4.1: codes legal to appear ON THE WIRE in a Close
    /// frame. 1005/1006/1015 are local-use-only sentinels (never sent);
    /// codes below 1000 and in 1016..3000 are undefined/unassigned.
    #[must_use]
    pub const fn valid_on_wire(self) -> bool {
        matches!(self.as_u16(), 1000..=1003 | 1007..=1014 | 3000..=4999)
    }
}

/// One complete, reassembled application message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Message<'a> {
    /// UTF-8-validated per RFC 6455 §8.1.
    Text(&'a str),
    Binary(&'a [u8]),
}

/// Outcome of [`Session::poll_event`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event<'a> {
    /// Nothing to report yet: the buffer holds no complete frame, the
    /// frame processed was an interior fragment, or it was a control
    /// frame fully handled internally (a reply, if any, is already
    /// queued for [`Session::poll_transmit`]).
    Incomplete,
    /// A complete message, reassembled if it arrived fragmented.
    Message(Message<'a>),
    /// A PING arrived. Informational only — the automatic PONG reply is
    /// already queued for [`Session::poll_transmit`]; most callers only
    /// need this to reset an idle/liveness timer.
    Ping(&'a [u8]),
    /// A PONG arrived. RFC 6455 leaves its use to the application (e.g.
    /// round-trip latency, idle-timer reset).
    Pong(&'a [u8]),
    /// The closing handshake reached its terminal state — either the
    /// peer initiated it (our echo, if any, is already queued for
    /// [`Session::poll_transmit`]) or a protocol violation forced it.
    /// [`Session::is_closed`] is `true` once this is observed; the
    /// caller drains `poll_transmit` and then closes the transport.
    Closed { code: CloseCode, reason: &'a str },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DataOpcode {
    Text,
    Binary,
}

struct Reassembly {
    opcode: DataOpcode,
    buffer: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionState {
    Open,
    /// We have sent our Close frame (either [`Session::close`] was
    /// called, or the peer closed first and we already echoed — that
    /// path goes straight to `Closed` instead, see
    /// [`Session::poll_event`]). Waiting for the peer's Close to
    /// complete the handshake; per RFC 6455 §7.1.2 no further data
    /// frames are sent once ours went out, so incoming data/control
    /// frames other than Close are discarded here rather than acted on.
    Closing,
    /// Both directions have sent a Close frame (or a protocol violation
    /// terminated the session unilaterally). The caller closes the
    /// underlying transport once [`Session::poll_transmit`] drains.
    Closed,
}

struct PendingFrame {
    bytes: Vec<u8>,
    cursor: usize,
}

/// Sans-IO RFC 6455 WebSocket SERVER session. See the module doc for the
/// `feed` / `poll_event` / `poll_transmit` driving shape.
pub struct Session {
    buffer: Vec<u8>,
    cursor: usize,
    state: SessionState,
    reassembly: Option<Reassembly>,
    /// Backing storage for the most recently completed FRAGMENTED
    /// message — reused across calls (no drop+realloc per message) and
    /// borrowed from directly by the `Event::Message` a fragmented
    /// completion returns. Empty and unused on the (common) single-frame
    /// fast path, which borrows the wire buffer instead.
    completed_message: Vec<u8>,
    limits: Limits,
    pending_outbound: Option<PendingFrame>,
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

impl Session {
    #[must_use]
    pub fn new() -> Self {
        Self::with_limits(Limits::default())
    }

    #[must_use]
    pub fn with_limits(limits: Limits) -> Self {
        Self {
            buffer: Vec::new(),
            cursor: 0,
            state: SessionState::Open,
            reassembly: None,
            completed_message: Vec::new(),
            limits,
            pending_outbound: None,
        }
    }

    /// Append bytes read off the wire.
    pub fn feed(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    /// `true` once the closing handshake has completed in either
    /// direction (see [`Event::Closed`]) — the caller should stop
    /// feeding new bytes and close the transport once
    /// [`Session::poll_transmit`] drains.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.state == SessionState::Closed
    }

    /// Drive the state machine one step: try to turn the buffered,
    /// unconsumed bytes into one [`Event`]. Call in a loop until
    /// [`Event::Incomplete`] before feeding more bytes.
    pub fn poll_event(&mut self) -> Event<'_> {
        self.compact();
        if self.pending_outbound.is_some() || self.state == SessionState::Closed {
            return Event::Incomplete;
        }
        match parse_frame(&self.buffer[self.cursor..]) {
            // Extracted into plain `Copy` locals (not passed as `&Frame`)
            // before any further call: `frame`/`used` borrow `self.buffer`
            // immutably, and every step below needs `&mut self` —
            // holding a `Frame` across that would conflict.
            Ok((frame, used)) => {
                let fin = frame.fin;
                let opcode = frame.opcode;
                let mask = frame.mask;
                let payload_len = frame.payload.len();
                self.on_parsed(fin, opcode, mask, payload_len, used)
            }
            Err(ParseError::Short) => Event::Incomplete,
            Err(ParseError::PartialPayload(declared_len)) => {
                if declared_len > self.limits.max_frame_bytes as u64 {
                    self.fail(
                        CloseCode::MessageTooBig,
                        "frame payload exceeds max_frame_bytes",
                    )
                } else {
                    Event::Incomplete
                }
            }
            Err(ParseError::PayloadTooLarge(_)) => self.fail(
                CloseCode::MessageTooBig,
                "declared payload length exceeds platform usize",
            ),
            Err(ParseError::ReservedBits) => self.fail(
                CloseCode::ProtocolError,
                "reserved bits set with no negotiated extension",
            ),
            Err(ParseError::UnknownOpcode(_)) => {
                self.fail(CloseCode::ProtocolError, "unknown opcode")
            }
            Err(ParseError::OversizedControl(_)) => self.fail(
                CloseCode::ProtocolError,
                "control frame payload exceeds 125 bytes",
            ),
        }
    }

    /// Drain up to `buf.len()` bytes of a queued outbound reply (an
    /// automatic PONG, a CLOSE echo, or a CLOSE this session initiated
    /// via [`Session::close`] or a protocol violation). `None` once
    /// nothing is pending. Call after every [`Session::poll_event`] that
    /// didn't return `Event::Incomplete` on a fresh buffer, and after
    /// every [`Session::close`] call, before feeding more input —
    /// `poll_event` refuses to process another frame while a reply is
    /// still undrained, so a caller that never drains stalls rather than
    /// silently losing the reply.
    pub fn poll_transmit(&mut self, buf: &mut [u8]) -> Option<usize> {
        let pending = self.pending_outbound.as_mut()?;
        let remaining = &pending.bytes[pending.cursor..];
        let count = remaining.len().min(buf.len());
        buf[..count].copy_from_slice(&remaining[..count]);
        pending.cursor += count;
        if pending.cursor >= pending.bytes.len() {
            self.pending_outbound = None;
        }
        Some(count)
    }

    /// Initiate the closing handshake (RFC 6455 §7.1.2). Queues a Close
    /// frame carrying `code` + `reason` for [`Session::poll_transmit`].
    /// Returns `false` (no-op) if the session is not `Open`, or if a
    /// reply is already queued and undrained — the caller drains
    /// [`Session::poll_transmit`] and retries. Returns `true` once
    /// queued.
    #[must_use]
    pub fn close(&mut self, code: CloseCode, reason: &str) -> bool {
        if self.state != SessionState::Open || self.pending_outbound.is_some() {
            return false;
        }
        let mut payload = Vec::with_capacity(2 + reason.len());
        payload.extend_from_slice(&code.as_u16().to_be_bytes());
        payload.extend_from_slice(reason.as_bytes());
        self.queue_outbound(Opcode::Close, &payload);
        self.state = SessionState::Closing;
        true
    }

    /// Drop the already-consumed prefix of the wire buffer. Safe to call
    /// at the top of `poll_event` (rather than right after consuming a
    /// frame) because any borrow the PREVIOUS `poll_event` call returned
    /// is guaranteed dead by the time a new `&mut self` call is made.
    fn compact(&mut self) {
        if self.cursor >= self.buffer.len() {
            self.buffer.clear();
            self.cursor = 0;
        } else if self.cursor > COMPACT_THRESHOLD_BYTES {
            self.buffer.drain(..self.cursor);
            self.cursor = 0;
        }
    }

    fn on_parsed(
        &mut self,
        fin: bool,
        opcode: Opcode,
        mask: Option<[u8; 4]>,
        payload_len: usize,
        used: usize,
    ) -> Event<'_> {
        let frame_start = self.cursor;
        let payload_start = frame_start + (used - payload_len);
        let payload_end = frame_start + used;
        self.cursor = frame_start + used;

        let Some(mask_key) = mask else {
            return self.fail(
                CloseCode::ProtocolError,
                "client frame must be masked (RFC 6455 §5.1)",
            );
        };
        if payload_len > self.limits.max_frame_bytes {
            return self.fail(
                CloseCode::MessageTooBig,
                "frame payload exceeds max_frame_bytes",
            );
        }
        unmask_in_place(&mut self.buffer[payload_start..payload_end], mask_key);
        self.handle_frame(fin, opcode, payload_start, payload_end)
    }

    fn handle_frame(&mut self, fin: bool, opcode: Opcode, start: usize, end: usize) -> Event<'_> {
        match opcode {
            Opcode::Ping => self.on_ping(fin, start, end),
            Opcode::Pong => self.on_pong(fin, start, end),
            Opcode::Close => self.on_close(fin, start, end),
            Opcode::Text => self.on_data_start(fin, DataOpcode::Text, start, end),
            Opcode::Binary => self.on_data_start(fin, DataOpcode::Binary, start, end),
            Opcode::Continuation => self.on_continuation(fin, start, end),
        }
    }

    fn on_ping(&mut self, fin: bool, start: usize, end: usize) -> Event<'_> {
        if !fin {
            return self.fail(
                CloseCode::ProtocolError,
                "control frame must not be fragmented (RFC 6455 §5.5)",
            );
        }
        // Once we've sent our own Close, RFC 6455 §7.1.2 forbids further
        // data frames from us; a PONG reply is withheld too rather than
        // relying on the wire-level distinction between "data" and
        // "control" to justify still replying mid-close.
        if self.state == SessionState::Open {
            self.queue_outbound_echo(Opcode::Pong, start, end);
        }
        Event::Ping(&self.buffer[start..end])
    }

    fn on_pong(&mut self, fin: bool, start: usize, end: usize) -> Event<'_> {
        if !fin {
            return self.fail(
                CloseCode::ProtocolError,
                "control frame must not be fragmented (RFC 6455 §5.5)",
            );
        }
        Event::Pong(&self.buffer[start..end])
    }

    fn on_close(&mut self, fin: bool, start: usize, end: usize) -> Event<'_> {
        if !fin {
            return self.fail(
                CloseCode::ProtocolError,
                "control frame must not be fragmented (RFC 6455 §5.5)",
            );
        }
        let len = end - start;
        if len == 1 {
            return self.fail(
                CloseCode::ProtocolError,
                "close frame payload must be empty or >= 2 bytes",
            );
        }
        let code = if len >= 2 {
            let raw = u16::from_be_bytes([self.buffer[start], self.buffer[start + 1]]);
            let close_code = CloseCode::from_u16(raw);
            if !close_code.valid_on_wire() {
                return self.fail(
                    CloseCode::ProtocolError,
                    "close status code is reserved or undefined",
                );
            }
            close_code
        } else {
            CloseCode::NoStatus
        };
        if len > 2 && str::from_utf8(&self.buffer[start + 2..end]).is_err() {
            return self.fail(
                CloseCode::InvalidPayload,
                "close reason is not valid UTF-8 (RFC 6455 §8.1)",
            );
        }
        let we_already_sent_close = self.state == SessionState::Closing;
        if !we_already_sent_close {
            self.queue_outbound_echo(Opcode::Close, start, end);
        }
        self.state = SessionState::Closed;
        self.reassembly = None;
        let reason = if len > 2 {
            str::from_utf8(&self.buffer[start + 2..end]).unwrap_or("")
        } else {
            ""
        };
        Event::Closed { code, reason }
    }

    fn on_data_start(
        &mut self,
        fin: bool,
        opcode: DataOpcode,
        start: usize,
        end: usize,
    ) -> Event<'_> {
        if self.state == SessionState::Closing {
            return Event::Incomplete;
        }
        if self.reassembly.is_some() {
            return self.fail(
                CloseCode::ProtocolError,
                "data frame started before the previous fragmented message finished (RFC 6455 §5.4)",
            );
        }
        let payload_len = end - start;
        if payload_len > self.limits.max_message_bytes {
            return self.fail(
                CloseCode::MessageTooBig,
                "message exceeds max_message_bytes",
            );
        }
        if fin {
            return self.finish_unfragmented(opcode, start, end);
        }
        let mut buffer = Vec::with_capacity(payload_len);
        buffer.extend_from_slice(&self.buffer[start..end]);
        self.reassembly = Some(Reassembly { opcode, buffer });
        Event::Incomplete
    }

    fn finish_unfragmented(&mut self, opcode: DataOpcode, start: usize, end: usize) -> Event<'_> {
        // Validated (and, on failure, `self.fail`'d) BEFORE the borrow
        // that the returned `Event` carries is taken: an arm that both
        // returns `&self.buffer[..]` for one branch's lifetime and calls
        // `&mut self` (`fail`) for the other cannot borrow-check as a
        // single match, since the Ok arm's lifetime requirement pins the
        // scrutinee's borrow region for the whole expression.
        if opcode == DataOpcode::Text && str::from_utf8(&self.buffer[start..end]).is_err() {
            return self.fail(
                CloseCode::InvalidPayload,
                "text message is not valid UTF-8 (RFC 6455 §8.1)",
            );
        }
        match opcode {
            DataOpcode::Binary => Event::Message(Message::Binary(&self.buffer[start..end])),
            // Validated above — `unwrap_or_default` never actually
            // fires; it exists so this arm makes zero further `&mut
            // self` calls (which is what the borrow-check above buys).
            DataOpcode::Text => Event::Message(Message::Text(
                str::from_utf8(&self.buffer[start..end]).unwrap_or_default(),
            )),
        }
    }

    fn on_continuation(&mut self, fin: bool, start: usize, end: usize) -> Event<'_> {
        if self.state == SessionState::Closing {
            return Event::Incomplete;
        }
        let Some(reassembly) = self.reassembly.as_mut() else {
            return self.fail(
                CloseCode::ProtocolError,
                "continuation frame with no fragmented message in progress (RFC 6455 §5.4)",
            );
        };
        let payload_len = end - start;
        if reassembly.buffer.len() + payload_len > self.limits.max_message_bytes {
            self.reassembly = None;
            return self.fail(
                CloseCode::MessageTooBig,
                "reassembled message exceeds max_message_bytes",
            );
        }
        reassembly
            .buffer
            .extend_from_slice(&self.buffer[start..end]);
        if !fin {
            return Event::Incomplete;
        }
        self.finish_fragmented()
    }

    fn finish_fragmented(&mut self) -> Event<'_> {
        let Some(Reassembly { opcode, buffer }) = self.reassembly.take() else {
            return self.fail(
                CloseCode::InternalError,
                "reassembly state lost between append and completion",
            );
        };
        // `buffer` is a local, owned `Vec<u8>` here (moved out of
        // `self.reassembly`) — validating against it borrows nothing
        // from `self`, so this `self.fail` call (unlike
        // `finish_unfragmented`'s) never conflicts with anything.
        if opcode == DataOpcode::Text && str::from_utf8(&buffer).is_err() {
            self.completed_message = buffer;
            return self.fail(
                CloseCode::InvalidPayload,
                "text message is not valid UTF-8 (RFC 6455 §8.1)",
            );
        }
        self.completed_message = buffer;
        match opcode {
            DataOpcode::Binary => Event::Message(Message::Binary(&self.completed_message)),
            DataOpcode::Text => Event::Message(Message::Text(
                str::from_utf8(&self.completed_message).unwrap_or_default(),
            )),
        }
    }

    /// Force the session to `Closed` and queue a Close frame carrying
    /// `code`, built fresh (not echoed from the wire buffer) — the
    /// generic path for every RFC-violation termination.
    fn fail(&mut self, code: CloseCode, reason: &'static str) -> Event<'static> {
        let payload = code.as_u16().to_be_bytes();
        self.queue_outbound(Opcode::Close, &payload);
        self.state = SessionState::Closed;
        self.reassembly = None;
        Event::Closed { code, reason }
    }

    /// Queue an outbound control frame whose payload is caller-owned
    /// (not derived from `self.buffer`) — the general form used by
    /// [`Session::fail`] and [`Session::close`].
    fn queue_outbound(&mut self, opcode: Opcode, payload: &[u8]) {
        let mut bytes = Vec::with_capacity(2 + payload.len());
        encode_header(true, opcode, payload.len(), None, &mut bytes);
        bytes.extend_from_slice(payload);
        self.pending_outbound = Some(PendingFrame { bytes, cursor: 0 });
    }

    /// Queue an outbound control frame whose payload is an EXACT echo of
    /// `self.buffer[start..end]` (a received PING's payload for the
    /// PONG reply, or a received CLOSE's code+reason for the echo) —
    /// reads `self.buffer` and writes `self.pending_outbound` in one
    /// method body so the call site never needs to hold a `self.buffer`
    /// borrow across a `&mut self` call.
    fn queue_outbound_echo(&mut self, opcode: Opcode, start: usize, end: usize) {
        let payload_len = end - start;
        let mut bytes = Vec::with_capacity(2 + payload_len);
        encode_header(true, opcode, payload_len, None, &mut bytes);
        bytes.extend_from_slice(&self.buffer[start..end]);
        self.pending_outbound = Some(PendingFrame { bytes, cursor: 0 });
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec;

    use super::*;

    /// Build a CLIENT (masked) wire frame the way a conformant client
    /// encoder would — reuses the exact primitives `Session` itself is
    /// built on ([`encode_header`], [`unmask_in_place`], which is its
    /// own XOR-based inverse per RFC 6455 §5.3) rather than a
    /// hand-rolled second encoder.
    fn client_frame(fin: bool, opcode: Opcode, payload: &[u8], key: [u8; 4]) -> Vec<u8> {
        let mut buf = Vec::new();
        encode_header(fin, opcode, payload.len(), Some(key), &mut buf);
        let mut masked_payload = payload.to_vec();
        unmask_in_place(&mut masked_payload, key);
        buf.extend_from_slice(&masked_payload);
        buf
    }

    /// Drain every byte `Session::poll_transmit` currently has queued.
    fn drain_transmit(session: &mut Session) -> Vec<u8> {
        let mut out = Vec::new();
        let mut scratch = [0_u8; 256];
        while let Some(count) = session.poll_transmit(&mut scratch) {
            if count == 0 {
                break;
            }
            out.extend_from_slice(&scratch[..count]);
        }
        out
    }

    /// A server frame (what `Session` itself emits) must never carry the
    /// mask bit — RFC 6455 §5.1: "a server MUST NOT mask any frames."
    fn assert_unmasked_server_frame(bytes: &[u8]) {
        assert!(!bytes.is_empty(), "empty server frame");
        assert_eq!(
            bytes[1] & 0x80,
            0,
            "server frame must not set the MASK bit: {bytes:?}"
        );
    }

    // ---- RFC 6455 §5.7 worked examples ("Example[s]") -----------------

    #[test]
    fn rfc6455_section_5_7_single_frame_masked_text_hello() {
        // Byte-exact copy of the RFC's own vector: a single-frame masked
        // text message containing "Hello".
        let wire = [
            0x81, 0x85, 0x37, 0xfa, 0x21, 0x3d, 0x7f, 0x9f, 0x4d, 0x51, 0x58,
        ];
        // Cross-check: our own client-frame builder reproduces the exact
        // same bytes from the plaintext payload + mask key, proving the
        // helper every other test below leans on matches the primary
        // source, not just this one hard-coded array.
        assert_eq!(
            client_frame(true, Opcode::Text, b"Hello", [0x37, 0xfa, 0x21, 0x3d]),
            wire
        );

        let mut session = Session::new();
        session.feed(&wire);
        match session.poll_event() {
            Event::Message(Message::Text(text)) => assert_eq!(text, "Hello"),
            other => panic!("expected Message(Text(\"Hello\")), got {other:?}"),
        }
    }

    #[test]
    fn rfc6455_section_5_7_fragmented_unmasked_text_hello_adapted_masked() {
        // RFC 6455 §5.7's fragmented example is illustrative and shown
        // UNMASKED (`0x01 0x03 0x48 0x65 0x6c` then `0x80 0x02 0x6c
        // 0x6f`); a conformant client frame is always masked (§5.1), so
        // this test keeps the RFC's exact fragmentation shape (fin=0
        // Text "Hel", then fin=1 Continuation "lo") and applies real
        // masking via the same primitive the RFC's own masked example
        // (tested above) uses.
        let key = [0x12, 0x34, 0x56, 0x78];
        let mut session = Session::new();

        session.feed(&client_frame(false, Opcode::Text, b"Hel", key));
        assert!(matches!(session.poll_event(), Event::Incomplete));

        session.feed(&client_frame(true, Opcode::Continuation, b"lo", key));
        match session.poll_event() {
            Event::Message(Message::Text(text)) => assert_eq!(text, "Hello"),
            other => panic!("expected Message(Text(\"Hello\")), got {other:?}"),
        }
    }

    #[test]
    fn rfc6455_section_5_7_256_byte_binary_single_frame() {
        // RFC 6455 §5.7's "256 bytes binary message in a single unmasked
        // frame" example is illustrating the 16-bit extended-length
        // encoding (marker 126); adapted masked + with real payload
        // bytes (a deterministic non-constant pattern, not all-zero, so
        // a byte-order bug in the extended-length path would show up as
        // wrong content, not just wrong length).
        let payload: Vec<u8> = (0..256u32).map(|index| (index % 251) as u8).collect();
        let key = [0xde, 0xad, 0xbe, 0xef];
        let wire = client_frame(true, Opcode::Binary, &payload, key);
        assert_eq!(
            wire[1] & 0x7f,
            126,
            "256-byte payload must use the 16-bit extended-length marker"
        );

        let mut session = Session::new();
        session.feed(&wire);
        match session.poll_event() {
            Event::Message(Message::Binary(data)) => assert_eq!(data, payload.as_slice()),
            other => panic!("expected Message(Binary(..)), got {other:?}"),
        }
    }

    #[test]
    fn rfc6455_section_5_7_64kib_binary_single_frame() {
        // RFC 6455 §5.7's 64 KiB example illustrates the 64-bit
        // extended-length encoding (marker 127).
        let payload: Vec<u8> = (0..65_536u32).map(|index| (index % 251) as u8).collect();
        let key = [0x01, 0x02, 0x03, 0x04];
        let wire = client_frame(true, Opcode::Binary, &payload, key);
        assert_eq!(
            wire[1] & 0x7f,
            127,
            "64 KiB payload must use the 64-bit extended-length marker"
        );

        let mut session = Session::with_limits(Limits {
            max_frame_bytes: 128 * 1024,
            max_message_bytes: 128 * 1024,
        });
        session.feed(&wire);
        match session.poll_event() {
            Event::Message(Message::Binary(data)) => assert_eq!(data.len(), 65_536),
            other => panic!("expected Message(Binary(..)), got {other:?}"),
        }
    }

    #[test]
    fn rfc6455_section_5_7_masked_ping_and_pong_hello() {
        let key = [0x9a, 0x1b, 0x2c, 0x3d];
        let mut session = Session::new();

        session.feed(&client_frame(true, Opcode::Ping, b"Hello", key));
        match session.poll_event() {
            Event::Ping(payload) => assert_eq!(payload, b"Hello"),
            other => panic!("expected Event::Ping, got {other:?}"),
        }
        let reply = drain_transmit(&mut session);
        assert_unmasked_server_frame(&reply);
        let mut expected = Vec::new();
        encode_header(true, Opcode::Pong, 5, None, &mut expected);
        expected.extend_from_slice(b"Hello");
        assert_eq!(reply, expected);

        session.feed(&client_frame(true, Opcode::Pong, b"Hello", key));
        match session.poll_event() {
            Event::Pong(payload) => assert_eq!(payload, b"Hello"),
            other => panic!("expected Event::Pong, got {other:?}"),
        }
    }

    // ---- fragmentation + interleaving -----------------------------

    #[test]
    fn fragmentation_across_three_frames_reassembles_in_order() {
        let key = [0x01, 0x02, 0x03, 0x04];
        let mut session = Session::new();
        session.feed(&client_frame(false, Opcode::Binary, b"ab", key));
        assert!(matches!(session.poll_event(), Event::Incomplete));
        session.feed(&client_frame(false, Opcode::Continuation, b"cd", key));
        assert!(matches!(session.poll_event(), Event::Incomplete));
        session.feed(&client_frame(true, Opcode::Continuation, b"ef", key));
        match session.poll_event() {
            Event::Message(Message::Binary(data)) => assert_eq!(data, b"abcdef"),
            other => panic!("expected Message(Binary(\"abcdef\")), got {other:?}"),
        }
    }

    #[test]
    fn interleaved_ping_mid_fragmentation_is_answered_without_disturbing_reassembly() {
        let key = [0xaa, 0xbb, 0xcc, 0xdd];
        let mut session = Session::new();
        session.feed(&client_frame(false, Opcode::Text, b"Hel", key));
        assert!(matches!(session.poll_event(), Event::Incomplete));

        session.feed(&client_frame(true, Opcode::Ping, b"hi", key));
        match session.poll_event() {
            Event::Ping(payload) => assert_eq!(payload, b"hi"),
            other => panic!("expected Event::Ping, got {other:?}"),
        }
        assert!(
            !drain_transmit(&mut session).is_empty(),
            "pong reply should be queued"
        );

        session.feed(&client_frame(true, Opcode::Continuation, b"lo", key));
        match session.poll_event() {
            Event::Message(Message::Text(text)) => assert_eq!(text, "Hello"),
            other => panic!("expected Message(Text(\"Hello\")), got {other:?}"),
        }
    }

    #[test]
    fn continuation_without_a_started_message_is_a_protocol_error() {
        let key = [0x11, 0x22, 0x33, 0x44];
        let mut session = Session::new();
        session.feed(&client_frame(true, Opcode::Continuation, b"stray", key));
        match session.poll_event() {
            Event::Closed {
                code: CloseCode::ProtocolError,
                ..
            } => {}
            other => panic!("expected Closed(ProtocolError), got {other:?}"),
        }
    }

    #[test]
    fn a_second_data_frame_before_the_first_fragmented_message_finishes_is_a_protocol_error() {
        let key = [0x11, 0x22, 0x33, 0x44];
        let mut session = Session::new();
        session.feed(&client_frame(false, Opcode::Text, b"Hel", key));
        assert!(matches!(session.poll_event(), Event::Incomplete));
        session.feed(&client_frame(true, Opcode::Binary, b"oops", key));
        match session.poll_event() {
            Event::Closed {
                code: CloseCode::ProtocolError,
                ..
            } => {}
            other => panic!("expected Closed(ProtocolError), got {other:?}"),
        }
    }

    // ---- masking enforcement (RFC 6455 §5.1) -----------------------

    #[test]
    fn unmasked_client_frame_is_rejected_with_protocol_error() {
        // A well-formed but UNMASKED text frame — exactly the shape a
        // buggy or malicious client would send. Built with
        // `encode_header(.., None, ..)`, the same helper a SERVER uses
        // to encode ITS OWN frames — proving the parser can't tell mask
        // absence from a server frame, which is exactly why Session must
        // enforce it as policy.
        let mut wire = Vec::new();
        encode_header(true, Opcode::Text, 5, None, &mut wire);
        wire.extend_from_slice(b"Hello");

        let mut session = Session::new();
        session.feed(&wire);
        match session.poll_event() {
            Event::Closed {
                code: CloseCode::ProtocolError,
                ..
            } => {}
            other => panic!("expected Closed(ProtocolError), got {other:?}"),
        }
        assert!(session.is_closed());
        assert_unmasked_server_frame(&drain_transmit(&mut session));
    }

    #[test]
    fn server_reply_frames_never_set_the_mask_bit() {
        let key = [0x55, 0x66, 0x77, 0x88];
        let mut session = Session::new();
        session.feed(&client_frame(true, Opcode::Ping, b"ping", key));
        let _ = session.poll_event();
        let reply = drain_transmit(&mut session);
        assert_unmasked_server_frame(&reply);
    }

    // ---- control-frame limits (RFC 6455 §5.5) -----------------------

    #[test]
    fn oversized_control_frame_is_rejected() {
        let key = [0x01, 0x02, 0x03, 0x04];
        let oversized_payload = vec![0x41_u8; 126];
        let wire = client_frame(true, Opcode::Ping, &oversized_payload, key);

        let mut session = Session::new();
        session.feed(&wire);
        match session.poll_event() {
            Event::Closed {
                code: CloseCode::ProtocolError,
                ..
            } => {}
            other => panic!("expected Closed(ProtocolError), got {other:?}"),
        }
    }

    // ---- UTF-8 validation (RFC 6455 §8.1) ----------------------------

    #[test]
    fn invalid_utf8_in_a_single_frame_text_message_closes_1007() {
        let key = [0xf0, 0x0d, 0xba, 0xbe];
        // 0xFF is never valid in any position of a UTF-8 sequence.
        let invalid = [b'H', b'i', 0xFF, 0xFF];
        let wire = client_frame(true, Opcode::Text, &invalid, key);

        let mut session = Session::new();
        session.feed(&wire);
        match session.poll_event() {
            Event::Closed {
                code: CloseCode::InvalidPayload,
                ..
            } => {}
            other => panic!("expected Closed(InvalidPayload), got {other:?}"),
        }
    }

    #[test]
    fn invalid_utf8_split_across_fragments_closes_1007() {
        // A 2-byte UTF-8 sequence (0xC2 0xA9, the copyright sign) split
        // so the FIRST fragment ends mid-sequence — only detectable once
        // the message is fully reassembled, proving validation happens
        // on the completed message, not per-fragment.
        let key = [0x0a, 0x0b, 0x0c, 0x0d];
        let mut session = Session::new();
        session.feed(&client_frame(false, Opcode::Text, &[0xC2], key));
        assert!(matches!(session.poll_event(), Event::Incomplete));
        // Continuation supplies an invalid second byte (not a valid UTF-8
        // continuation byte), corrupting the sequence.
        session.feed(&client_frame(true, Opcode::Continuation, &[0x00], key));
        match session.poll_event() {
            Event::Closed {
                code: CloseCode::InvalidPayload,
                ..
            } => {}
            other => panic!("expected Closed(InvalidPayload), got {other:?}"),
        }
    }

    // ---- size limits (RFC 6455 §7.4.1 code 1009) ---------------------

    #[test]
    fn a_frame_declaring_more_than_max_frame_bytes_closes_1009_before_buffering_it() {
        let mut session = Session::with_limits(Limits {
            max_frame_bytes: 16,
            max_message_bytes: 1024,
        });
        // Header alone declares a 1000-byte payload; only a few payload
        // bytes are actually fed — proves the guard fires on the
        // DECLARED length, not only once the buffer already holds it
        // (the DoS case: an attacker who never finishes sending).
        let mut header = Vec::new();
        encode_header(true, Opcode::Binary, 1000, Some([1, 2, 3, 4]), &mut header);
        let mut session_feed = header;
        session_feed.extend_from_slice(&[0_u8; 4]);
        session.feed(&session_feed);
        match session.poll_event() {
            Event::Closed {
                code: CloseCode::MessageTooBig,
                ..
            } => {}
            other => panic!("expected Closed(MessageTooBig), got {other:?}"),
        }
    }

    #[test]
    fn reassembled_message_exceeding_max_message_bytes_closes_1009() {
        // Neither individual fragment exceeds `max_frame_bytes` (64) or
        // `max_message_bytes` (10) on its own — only the CUMULATIVE
        // total across fragments does, proving the check tracks
        // reassembly progress rather than re-checking each frame alone.
        let key = [0x01, 0x02, 0x03, 0x04];
        let mut session = Session::with_limits(Limits {
            max_frame_bytes: 64,
            max_message_bytes: 10,
        });
        session.feed(&client_frame(false, Opcode::Text, b"012345", key));
        assert!(matches!(session.poll_event(), Event::Incomplete));
        session.feed(&client_frame(true, Opcode::Continuation, b"6789ab", key));
        match session.poll_event() {
            Event::Closed {
                code: CloseCode::MessageTooBig,
                ..
            } => {}
            other => panic!("expected Closed(MessageTooBig), got {other:?}"),
        }
    }

    // ---- closing handshake (RFC 6455 §7.4) ---------------------------

    #[test]
    fn peer_initiated_close_is_echoed_and_completes_the_handshake() {
        let key = [0x01, 0x02, 0x03, 0x04];
        let mut payload = Vec::new();
        payload.extend_from_slice(&1000_u16.to_be_bytes());
        payload.extend_from_slice(b"bye");
        let wire = client_frame(true, Opcode::Close, &payload, key);

        let mut session = Session::new();
        session.feed(&wire);
        match session.poll_event() {
            Event::Closed {
                code: CloseCode::Normal,
                reason,
            } => assert_eq!(reason, "bye"),
            other => panic!("expected Closed(Normal, \"bye\"), got {other:?}"),
        }
        assert!(session.is_closed());

        let echoed = drain_transmit(&mut session);
        assert_unmasked_server_frame(&echoed);
        // Echo carries the SAME code+reason bytes the peer sent.
        assert_eq!(&echoed[2..], payload.as_slice());
    }

    #[test]
    fn server_initiated_close_then_peer_echo_completes_the_handshake_with_no_second_send() {
        let mut session = Session::new();
        assert!(session.close(CloseCode::GoingAway, "shutting down"));

        let sent = drain_transmit(&mut session);
        assert_unmasked_server_frame(&sent);
        assert_eq!(u16::from_be_bytes([sent[2], sent[3]]), 1001);
        assert_eq!(&sent[4..], b"shutting down");
        assert!(
            !session.is_closed(),
            "handshake not complete until the peer echoes"
        );

        let key = [0x0a, 0x0b, 0x0c, 0x0d];
        session.feed(&client_frame(
            true,
            Opcode::Close,
            &1001_u16.to_be_bytes(),
            key,
        ));
        match session.poll_event() {
            Event::Closed {
                code: CloseCode::GoingAway,
                ..
            } => {}
            other => panic!("expected Closed(GoingAway), got {other:?}"),
        }
        assert!(session.is_closed());
        // We already sent our Close — no second one goes out for the echo.
        assert!(drain_transmit(&mut session).is_empty());
    }

    #[test]
    fn no_data_frames_are_reported_once_closed() {
        let key = [0x01, 0x02, 0x03, 0x04];
        let mut session = Session::new();
        session.feed(&client_frame(true, Opcode::Close, &[], key));
        let _ = session.poll_event();
        assert!(session.is_closed());
        let _ = drain_transmit(&mut session);

        session.feed(&client_frame(true, Opcode::Text, b"too late", key));
        assert!(matches!(session.poll_event(), Event::Incomplete));
    }

    #[test]
    fn close_frame_with_invalid_status_code_is_rejected() {
        let key = [0x01, 0x02, 0x03, 0x04];
        // 1006 is a reserved local-use sentinel — MUST NOT appear on the
        // wire per RFC 6455 §7.4.1.
        let wire = client_frame(true, Opcode::Close, &1006_u16.to_be_bytes(), key);
        let mut session = Session::new();
        session.feed(&wire);
        match session.poll_event() {
            Event::Closed {
                code: CloseCode::ProtocolError,
                ..
            } => {}
            other => panic!("expected Closed(ProtocolError), got {other:?}"),
        }
    }

    #[test]
    fn close_frame_with_one_byte_payload_is_rejected() {
        let key = [0x01, 0x02, 0x03, 0x04];
        let wire = client_frame(true, Opcode::Close, &[0x03], key);
        let mut session = Session::new();
        session.feed(&wire);
        match session.poll_event() {
            Event::Closed {
                code: CloseCode::ProtocolError,
                ..
            } => {}
            other => panic!("expected Closed(ProtocolError), got {other:?}"),
        }
    }

    #[test]
    fn close_frame_with_invalid_utf8_reason_closes_1007() {
        let key = [0x01, 0x02, 0x03, 0x04];
        let mut payload = Vec::new();
        payload.extend_from_slice(&1000_u16.to_be_bytes());
        payload.push(0xFF);
        let wire = client_frame(true, Opcode::Close, &payload, key);
        let mut session = Session::new();
        session.feed(&wire);
        match session.poll_event() {
            Event::Closed {
                code: CloseCode::InvalidPayload,
                ..
            } => {}
            other => panic!("expected Closed(InvalidPayload), got {other:?}"),
        }
    }

    // ---- reserved bits (delegated to the base parser) -----------------

    #[test]
    fn reserved_bits_set_closes_1002() {
        let wire = [0xC1_u8, 0x80, 0x01, 0x02, 0x03, 0x04];
        let mut session = Session::new();
        session.feed(&wire);
        match session.poll_event() {
            Event::Closed {
                code: CloseCode::ProtocolError,
                ..
            } => {}
            other => panic!("expected Closed(ProtocolError), got {other:?}"),
        }
    }

    // ---- close-code round trip -----------------------------------------

    #[test]
    fn close_code_round_trips_through_u16() {
        for code in [
            CloseCode::Normal,
            CloseCode::GoingAway,
            CloseCode::ProtocolError,
            CloseCode::InvalidPayload,
            CloseCode::MessageTooBig,
            CloseCode::InternalError,
            CloseCode::Other(4100),
        ] {
            assert_eq!(CloseCode::from_u16(code.as_u16()), code);
        }
    }

    #[test]
    fn reserved_wire_codes_are_rejected_by_valid_on_wire() {
        for reserved in [0_u16, 999, 1004, 1005, 1006, 1015, 1016, 2999, 5000] {
            assert!(
                !CloseCode::from_u16(reserved).valid_on_wire(),
                "{reserved} must be invalid on the wire"
            );
        }
        for legal in [1000, 1002, 1009, 1011, 1014, 3000, 4999] {
            assert!(
                CloseCode::from_u16(legal).valid_on_wire(),
                "{legal} must be valid on the wire"
            );
        }
    }
}
