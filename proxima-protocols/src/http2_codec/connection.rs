//! HTTP/2 connection driver — sans-IO (RFC 7540 §3 + §6.5).
//!
//! [`Connection`] is fed wire bytes via [`Connection::feed`] and
//! exposes pending output bytes via [`Connection::take_output`]. The
//! I/O layer (sync or async) just shuttles bytes to/from the socket
//! and calls these methods. No async, no executor coupling, fully
//! testable with synthetic byte sequences.
//!
//! ## Lifecycle
//!
//! ```text
//!  AwaitingPreface ──read 24-byte client preface──> AwaitingClientSettings
//!                                                        │
//!                                                read SETTINGS (non-ACK)
//!                                                        │
//!                                                        v
//!                                                     Running
//!                                                        │
//!                                              recv GOAWAY / send GOAWAY
//!                                                        │
//!                                                        v
//!                                                     Closing
//! ```
//!
//! At construction time `Connection` queues its **own** server preface:
//! a SETTINGS frame announcing local parameters. The I/O layer flushes
//! that to the wire before reading the client preface.
//!
//! ## Layering
//!
//! Frame parsing lives in [`super::frame`]; stream + flow-control
//! state in [`super::stream_table`] and [`super::stream`]; HPACK in
//! [`super::hpack`]. This module wires them together and owns the
//! handshake choreography.

use alloc::collections::VecDeque;
use alloc::vec::Vec;

use bytes::{Buf, BufMut, Bytes, BytesMut};
use thiserror::Error;

use crate::http2_codec::frame::{
    CONNECTION_PREFACE, FRAME_HEADER_LEN, FrameError, FrameHeader, FramePayload, FrameType,
    StandardSettings, encode_frame, flags, parse_payload,
};
use crate::http2_codec::stream::{StreamError, StreamEvent};
use crate::http2_codec::stream_table::{StreamTable, TableError};
use crate::hpack::{DecodeError, DynamicTable, decode_block, encode_block};

const HPACK_DEFAULT_TABLE_SIZE: u32 = 4096;

#[derive(Debug, Error)]
pub enum ConnectionError {
    #[error("client preface mismatch at byte {offset}")]
    PrefaceMismatch { offset: usize },
    #[error("first client frame must be SETTINGS (non-ACK); got {got:?}")]
    MissingClientSettings { got: FrameType },
    #[error("SETTINGS frame on stream {0}; SETTINGS is connection-scoped (stream 0 only)")]
    SettingsOnNonZeroStream(u32),
    #[error("SETTINGS ACK frame must have empty payload")]
    SettingsAckNonEmpty,
    #[error("frame parse error: {0}")]
    Frame(#[from] FrameError),
    #[error("flow / stream error: {0}")]
    Table(#[from] TableError),
    #[error("hpack decode error: {0}")]
    Hpack(#[from] DecodeError),
    #[error("stream {0}: {1}")]
    Stream(u32, StreamError),
    #[error("CONTINUATION on stream {got}, expected stream {expected} (in-progress HEADERS)")]
    ContinuationStreamMismatch { got: u32, expected: u32 },
    #[error("frame {got:?} interrupted in-progress HEADERS on stream {stream}")]
    HeadersInterrupted { stream: u32, got: FrameType },
    #[error("CONTINUATION with no in-progress HEADERS")]
    UnexpectedContinuation,
    #[error("frame header parse failed despite length check")]
    InternalParseFailure,
}

/// RFC 7540 §8.1.2 header block validation for a server-received
/// request. Returns an error reason string if malformed; the caller
/// emits RST_STREAM(PROTOCOL_ERROR) and skips dispatch.
///
/// Rules enforced:
/// - header field names must be lowercase
/// - pseudo-headers (`:` prefix) must precede regular headers
/// - request pseudo-headers limited to `:method`, `:scheme`, `:path`,
///   `:authority`. `:status` is response-only.
/// - exactly one each of `:method`, `:scheme`, `:path`; at most one
///   `:authority`. (CONNECT method exemptions skipped for MVP.)
/// - connection-specific headers forbidden: `connection`,
///   `keep-alive`, `proxy-connection`, `transfer-encoding`, `upgrade`
/// - empty header names are malformed
fn validate_request_headers(headers: &[(Bytes, Bytes)]) -> Result<(), &'static str> {
    let mut seen_regular = false;
    let mut method_count = 0_u32;
    let mut scheme_count = 0_u32;
    let mut path_count = 0_u32;
    let mut authority_count = 0_u32;
    for (name, _value) in headers {
        let name_bytes = name.as_ref();
        if name_bytes.is_empty() {
            return Err("empty header name");
        }
        let is_pseudo = name_bytes[0] == b':';
        if is_pseudo {
            if seen_regular {
                return Err("pseudo-header after regular header");
            }
            match name_bytes {
                b":method" => method_count += 1,
                b":scheme" => scheme_count += 1,
                b":path" => path_count += 1,
                b":authority" => authority_count += 1,
                b":status" => return Err(":status invalid in request"),
                b":protocol" => {} // RFC 8441 extended CONNECT — allowed
                _ => return Err("unknown pseudo-header"),
            }
        } else {
            seen_regular = true;
            // Header names must be lowercase ASCII (RFC §8.1.2).
            if name_bytes.iter().any(u8::is_ascii_uppercase) {
                return Err("uppercase header name");
            }
            // Connection-specific headers are forbidden (RFC §8.1.2.2).
            match name_bytes {
                b"connection" | b"keep-alive" | b"proxy-connection" | b"transfer-encoding"
                | b"upgrade" => return Err("connection-specific header forbidden"),
                _ => {}
            }
        }
    }
    if method_count != 1 {
        return Err(":method must appear exactly once");
    }
    if scheme_count != 1 {
        return Err(":scheme must appear exactly once");
    }
    if path_count != 1 {
        return Err(":path must appear exactly once");
    }
    if authority_count > 1 {
        return Err(":authority must appear at most once");
    }
    Ok(())
}

/// RFC 7540 §8.1.2 header validation for a CLIENT-received response head (or a
/// trailers block). `:status` is the only legal response pseudo-header; request
/// pseudo-headers are forbidden; pseudo-headers precede regular; names lowercase;
/// connection-specific headers forbidden. `:status` count is 0 or 1 — a response
/// head carries exactly one, a trailers block (HEADERS after DATA) carries none.
fn validate_response_headers(headers: &[(Bytes, Bytes)]) -> Result<(), &'static str> {
    let mut seen_regular = false;
    let mut status_count = 0_u32;
    for (name, _value) in headers {
        let name_bytes = name.as_ref();
        if name_bytes.is_empty() {
            return Err("empty header name");
        }
        let is_pseudo = name_bytes[0] == b':';
        if is_pseudo {
            if seen_regular {
                return Err("pseudo-header after regular header");
            }
            match name_bytes {
                b":status" => status_count += 1,
                b":method" | b":scheme" | b":path" | b":authority" => {
                    return Err("request pseudo-header in response");
                }
                _ => return Err("unknown response pseudo-header"),
            }
        } else {
            seen_regular = true;
            if name_bytes.iter().any(u8::is_ascii_uppercase) {
                return Err("uppercase header name");
            }
            match name_bytes {
                b"connection" | b"keep-alive" | b"proxy-connection" | b"transfer-encoding"
                | b"upgrade" => return Err("connection-specific header forbidden"),
                _ => {}
            }
        }
    }
    if status_count > 1 {
        return Err(":status must appear at most once");
    }
    Ok(())
}

impl ConnectionError {
    /// Map the connection error to a GOAWAY error code per RFC 7540 §7.
    /// The wrapper queues GOAWAY with this code before dropping the
    /// socket so the peer knows why.
    #[must_use]
    pub fn goaway_code(&self) -> u32 {
        use crate::http2_codec::frame::error_code as code;
        match self {
            Self::PrefaceMismatch { .. } => code::PROTOCOL_ERROR,
            Self::MissingClientSettings { .. } => code::PROTOCOL_ERROR,
            Self::SettingsOnNonZeroStream(_) => code::PROTOCOL_ERROR,
            Self::SettingsAckNonEmpty => code::FRAME_SIZE_ERROR,
            Self::Frame(_) => code::PROTOCOL_ERROR,
            Self::Table(table_err) => match table_err {
                TableError::ConnRecvWindowExceeded { .. }
                | TableError::ConnSendWindowExceeded { .. }
                | TableError::ConnWindowOverflow { .. } => code::FLOW_CONTROL_ERROR,
                _ => code::PROTOCOL_ERROR,
            },
            Self::Hpack(_) => code::COMPRESSION_ERROR,
            Self::Stream(_, _) => code::PROTOCOL_ERROR,
            Self::ContinuationStreamMismatch { .. } => code::PROTOCOL_ERROR,
            Self::HeadersInterrupted { .. } => code::PROTOCOL_ERROR,
            Self::UnexpectedContinuation => code::PROTOCOL_ERROR,
            Self::InternalParseFailure => code::INTERNAL_ERROR,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// (Server) reading the 24-byte magic from the client.
    AwaitingPreface,
    /// (Server) preface received; awaiting the client's initial SETTINGS frame.
    AwaitingClientSettings,
    /// (Client) preface + SETTINGS sent; awaiting the server's initial SETTINGS.
    /// The client never reads the 24-byte magic (only the client SENDS it), so
    /// it starts here rather than in `AwaitingPreface`.
    AwaitingServerSettings,
    /// Both sides handshook. DATA / HEADERS / etc. flow.
    Running,
    /// GOAWAY observed or queued. No new streams. Existing
    /// streams may complete.
    Closing,
}

/// Which side of the connection this `Connection` drives. The frame / HPACK /
/// flow-control / stream machinery is symmetric; `Role` selects only the
/// handshake direction (who sends the preface) and the request/response polarity
/// (a server receives `RequestHead` + sends responses; a client sends requests +
/// receives `ResponseHead`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Server,
    Client,
}

/// Sans-IO HTTP/2 connection state machine (server or client — see [`Role`]).
#[derive(Debug)]
pub struct Connection {
    /// Server or client. Selects handshake direction + request/response polarity.
    role: Role,
    /// (Client) next odd stream id to hand out from [`next_local_stream_id`].
    next_local_stream_id: u32,
    state: ConnectionState,
    /// Bytes the I/O layer should write to the peer. Vec because
    /// `encode_frame` takes `&mut Vec<u8>`.
    output: Vec<u8>,
    /// Bytes waiting to be parsed (partial frames).
    input: BytesMut,
    /// Preface bytes received so far (resets after 24).
    preface_received: usize,
    /// Frame buffer accumulator while a frame's payload is in flight.
    in_flight_payload: Option<InFlightFrame>,
    /// Connection-wide stream table + flow control.
    streams: StreamTable,
    /// HPACK decoder dynamic table (peer's encoded headers). Used
    /// once HEADERS dispatch lands in the next layer.
    #[allow(dead_code)]
    decoder_table: DynamicTable,
    /// HPACK encoder dynamic table (our encoded responses).
    encoder_table: DynamicTable,
    /// Settings we've announced to the peer. Kept for reference /
    /// future re-negotiation; consulted once we add per-peer caps.
    #[allow(dead_code)]
    local_settings: StandardSettings,
    /// Settings the peer announced to us. Updated each non-ACK SETTINGS.
    remote_settings: StandardSettings,
    /// Effective max frame size we'll accept (SETTINGS_MAX_FRAME_SIZE
    /// from the peer, or RFC default 16,384).
    peer_max_frame_size: u32,
    /// Pending control events to surface to the I/O layer / dispatcher.
    events: VecDeque<ConnectionEvent>,
    /// Header block in progress (HEADERS w/o END_HEADERS, awaiting
    /// CONTINUATION frames).
    pending_headers: Option<PendingHeaders>,
    /// Reusable scratch buffer for HPACK block encoding in
    /// `send_response_head`. Reset (cleared, capacity retained) per
    /// response so we don't malloc a fresh BytesMut every time.
    encoder_scratch: BytesMut,
    /// Spare `Vec<(Bytes, Bytes)>` recycled across `complete_headers`
    /// calls — same "reset, capacity retained" idea as
    /// `encoder_scratch`, applied to the DECODE side's owned-headers
    /// event payload (`DC-H2-EVENTS-OWN`, see
    /// `docs/proxima-quic/alloc-budget.md`). Populated by
    /// [`Connection::return_headers_buffer`] once a consumer (the I/O
    /// facade's event-drain loop) finishes reading a queued
    /// `RequestHead`/`ResponseHead` event's `headers`. `None` means no
    /// spare is currently available (first request on this connection,
    /// or a prior buffer hasn't been returned yet because more than
    /// one HEADERS frame's worth of events is queued ahead of the
    /// drain loop) — `complete_headers` falls back to a fresh
    /// `Vec::with_capacity` in that case.
    headers_scratch: Option<Vec<(Bytes, Bytes)>>,
}

#[derive(Debug)]
struct PendingHeaders {
    stream_id: u32,
    buffer: BytesMut,
    end_stream: bool,
}

#[derive(Debug)]
struct InFlightFrame {
    header: FrameHeader,
}

/// Outcome of a [`Connection::send_body`] call.
#[derive(Debug)]
pub enum SendOutcome {
    /// All bytes emitted; stream advanced through `SendData` events.
    Done,
    /// Send window exhausted (stream or connection). Caller holds the
    /// remainder until a [`ConnectionEvent::WindowGranted`] event
    /// fires for this `stream_id` or for `stream_id == 0` (connection-
    /// wide), then calls `send_body` again with the remainder.
    WindowExhausted { remainder: Bytes, end_stream: bool },
}

/// Side-band events the connection emits as it processes input.
/// The I/O layer drains these via [`Connection::next_event`] each
/// turn of its loop.
#[derive(Debug)]
pub enum ConnectionEvent {
    /// Peer asked the connection to close. last_stream_id signals
    /// the highest stream id the peer will pipe.
    PeerGoaway {
        last_stream_id: u32,
        error_code: u32,
        debug_data: Bytes,
    },
    /// PING received; we've already queued the ACK in output.
    PingAcked { opaque_data: [u8; 8] },
    /// Peer's SETTINGS were applied (after our ACK got queued).
    SettingsApplied,
    /// A complete request head (HEADERS [+ CONTINUATION]) has been
    /// assembled and decoded. `end_stream` indicates whether DATA
    /// frames will follow (false) or this is the entire request
    /// (true — typical GET).
    RequestHead {
        stream_id: u32,
        headers: Vec<(Bytes, Bytes)>,
        end_stream: bool,
    },
    /// (Client) a complete response head (HEADERS [+ CONTINUATION]) for a stream
    /// we opened has been assembled + decoded. `end_stream` true means a
    /// headers-only response (no DATA to follow — e.g. 204, or a gRPC
    /// trailers-only reply); false means response DATA follows.
    ResponseHead {
        stream_id: u32,
        headers: Vec<(Bytes, Bytes)>,
        end_stream: bool,
    },
    /// Inbound DATA payload for a stream. `end_stream` signals the
    /// end of the request body.
    BodyData {
        stream_id: u32,
        data: Bytes,
        end_stream: bool,
    },
    /// Peer reset a stream. Caller cancels any in-flight handler.
    StreamReset { stream_id: u32, error_code: u32 },
    /// WINDOW_UPDATE applied. `stream_id == 0` means connection-wide.
    WindowGranted { stream_id: u32, increment: u32 },
}

impl Connection {
    /// New connection in the initial state. The server preface
    /// (a SETTINGS frame announcing `local_settings`) is queued into
    /// the output buffer so the I/O layer's first `take_output()`
    /// drains it.
    #[must_use]
    pub fn new(local_settings: StandardSettings) -> Self {
        let mut connection = Self::bare(
            Role::Server,
            ConnectionState::AwaitingPreface,
            &local_settings,
        );
        connection.queue_settings_frame(&local_settings, false);
        connection
    }

    /// New CLIENT connection. Queues the 24-byte connection preface magic
    /// (RFC 7540 §3.5) followed by the client's initial SETTINGS into the output
    /// buffer — the I/O layer flushes both before the server replies. The client
    /// never reads a preface (only the client sends the magic), so it starts in
    /// [`AwaitingServerSettings`](ConnectionState::AwaitingServerSettings).
    #[must_use]
    pub fn new_client(local_settings: StandardSettings) -> Self {
        let mut connection = Self::bare(
            Role::Client,
            ConnectionState::AwaitingServerSettings,
            &local_settings,
        );
        connection.output.extend_from_slice(CONNECTION_PREFACE);
        connection.queue_settings_frame(&local_settings, false);
        connection
    }

    fn bare(role: Role, state: ConnectionState, local_settings: &StandardSettings) -> Self {
        Self {
            role,
            next_local_stream_id: 1,
            state,
            output: Vec::with_capacity(256),
            input: BytesMut::with_capacity(4096),
            preface_received: 0,
            in_flight_payload: None,
            streams: StreamTable::new(),
            decoder_table: DynamicTable::new(HPACK_DEFAULT_TABLE_SIZE as usize),
            encoder_table: DynamicTable::new(HPACK_DEFAULT_TABLE_SIZE as usize),
            local_settings: local_settings.clone(),
            remote_settings: StandardSettings::default(),
            peer_max_frame_size: 16_384,
            events: VecDeque::new(),
            pending_headers: None,
            encoder_scratch: BytesMut::with_capacity(512),
            headers_scratch: None,
        }
    }

    /// Return a drained `headers` buffer from a consumed
    /// `RequestHead`/`ResponseHead` event so the NEXT `complete_headers`
    /// call can reuse its allocation instead of paying a fresh
    /// `Vec::with_capacity`. Callers should `buffer.clear()` (or
    /// `drain(..)`, which leaves the Vec empty with capacity retained)
    /// before returning it — this method clears defensively either way,
    /// so a caller that forgets can't leak stale header data into the
    /// next request.
    ///
    /// Only ONE spare is kept: if a buffer is already parked (multiple
    /// HEADERS frames decoded within one `feed()` batch, events not yet
    /// drained), the newer one is simply dropped — correctness never
    /// depends on this succeeding, only steady-state allocation count
    /// does.
    pub fn return_headers_buffer(&mut self, mut buffer: Vec<(Bytes, Bytes)>) {
        buffer.clear();
        self.headers_scratch = Some(buffer);
    }

    /// The connection's role (server or client).
    #[must_use]
    pub fn role(&self) -> Role {
        self.role
    }

    /// (Client) allocate the next odd, monotonically-increasing stream id
    /// (RFC 7540 §5.1.1). Pass the result to [`send_request_head`](Self::send_request_head).
    pub fn next_local_stream_id(&mut self) -> u32 {
        let id = self.next_local_stream_id;
        self.next_local_stream_id = id.saturating_add(2);
        id
    }

    /// Current state.
    #[must_use]
    pub fn state(&self) -> ConnectionState {
        self.state
    }

    /// Connection stream-table view.
    #[must_use]
    pub fn streams(&self) -> &StreamTable {
        &self.streams
    }

    /// Reap closed streams. Call periodically so the table doesn't
    /// grow unbounded over a long-lived connection.
    pub fn gc_closed_streams(&mut self) -> usize {
        self.streams.gc_closed()
    }

    /// Bytes pending for the wire. Caller drains by writing then
    /// calling again on the next loop iteration.
    ///
    /// Preserves the buffer's allocated capacity across calls so we
    /// don't pay malloc on the first push after every flush. The
    /// freed `Vec` becomes `Bytes` zero-copy (Bytes::from(Vec)); the
    /// replacement allocates exactly once with the prior capacity.
    pub fn take_output(&mut self) -> Bytes {
        if self.output.is_empty() {
            return Bytes::new();
        }
        let capacity = self.output.capacity();
        let taken = core::mem::replace(&mut self.output, Vec::with_capacity(capacity));
        Bytes::from(taken)
    }

    /// Drain the next event, if any.
    pub fn next_event(&mut self) -> Option<ConnectionEvent> {
        self.events.pop_front()
    }

    /// Feed wire bytes into the parser. Returns when all `bytes` are
    /// consumed and any complete frames are dispatched.
    pub fn feed(&mut self, bytes: &[u8]) -> Result<(), ConnectionError> {
        self.input.extend_from_slice(bytes);
        loop {
            match self.state {
                ConnectionState::AwaitingPreface => {
                    if !self.consume_preface()? {
                        return Ok(());
                    }
                }
                ConnectionState::AwaitingClientSettings
                | ConnectionState::AwaitingServerSettings
                | ConnectionState::Running
                | ConnectionState::Closing => {
                    if !self.consume_frame()? {
                        return Ok(());
                    }
                }
            }
        }
    }

    fn consume_preface(&mut self) -> Result<bool, ConnectionError> {
        let needed = CONNECTION_PREFACE.len() - self.preface_received;
        let available = self.input.len();
        let take = needed.min(available);
        for index in 0..take {
            let expected = CONNECTION_PREFACE[self.preface_received + index];
            if self.input[index] != expected {
                return Err(ConnectionError::PrefaceMismatch {
                    offset: self.preface_received + index,
                });
            }
        }
        self.input.advance(take);
        self.preface_received += take;
        if self.preface_received == CONNECTION_PREFACE.len() {
            self.state = ConnectionState::AwaitingClientSettings;
            return Ok(true);
        }
        Ok(false)
    }

    fn consume_frame(&mut self) -> Result<bool, ConnectionError> {
        if self.in_flight_payload.is_none() {
            if self.input.len() < FRAME_HEADER_LEN {
                return Ok(false);
            }
            let header = FrameHeader::parse(&self.input[..FRAME_HEADER_LEN])
                .ok_or(ConnectionError::InternalParseFailure)?;
            self.input.advance(FRAME_HEADER_LEN);
            self.in_flight_payload = Some(InFlightFrame { header });
        }
        let Some(in_flight) = &self.in_flight_payload else {
            return Ok(false);
        };
        let payload_len = in_flight.header.length as usize;
        if self.input.len() < payload_len {
            return Ok(false);
        }
        let header = in_flight.header;
        let payload = self.input.split_to(payload_len).freeze();
        self.in_flight_payload = None;
        self.process_frame(header, payload)?;
        Ok(true)
    }

    fn process_frame(
        &mut self,
        header: FrameHeader,
        payload: Bytes,
    ) -> Result<(), ConnectionError> {
        let is_settings_ack =
            header.frame_type == FrameType::Settings && header.flags & flags::ACK != 0;
        if matches!(
            self.state,
            ConnectionState::AwaitingClientSettings | ConnectionState::AwaitingServerSettings
        ) && (header.frame_type != FrameType::Settings || is_settings_ack)
        {
            return Err(ConnectionError::MissingClientSettings {
                got: header.frame_type,
            });
        }
        // Once a HEADERS arrives without END_HEADERS, only CONTINUATION
        // frames on the same stream are legal until END_HEADERS lands
        // (RFC §6.10).
        if let Some(pending) = &self.pending_headers
            && header.frame_type != FrameType::Continuation
        {
            return Err(ConnectionError::HeadersInterrupted {
                stream: pending.stream_id,
                got: header.frame_type,
            });
        }
        let parsed = parse_payload(&header, &payload)?;
        match parsed {
            FramePayload::Settings(settings) => {
                self.handle_settings(&header, settings, is_settings_ack)?;
            }
            FramePayload::Ping { opaque } => {
                let is_ack = header.flags & flags::ACK != 0;
                if !is_ack {
                    self.queue_ping_ack(opaque);
                }
                self.events.push_back(ConnectionEvent::PingAcked {
                    opaque_data: opaque,
                });
            }
            FramePayload::GoAway {
                last_stream_id,
                error_code,
                debug_data,
            } => {
                self.state = ConnectionState::Closing;
                self.events.push_back(ConnectionEvent::PeerGoaway {
                    last_stream_id,
                    error_code,
                    debug_data,
                });
            }
            FramePayload::Headers { block_fragment, .. } => {
                self.handle_headers(&header, block_fragment)?;
            }
            FramePayload::Continuation { block_fragment } => {
                self.handle_continuation(&header, block_fragment)?;
            }
            FramePayload::Data { data } => {
                self.handle_data(&header, data)?;
            }
            FramePayload::RstStream { error_code } => {
                self.handle_rst(&header, error_code)?;
            }
            FramePayload::WindowUpdate { increment } => {
                self.handle_window_update(&header, increment)?;
            }
            FramePayload::Priority(_) | FramePayload::Unknown { .. } => {
                // PRIORITY is deprecated, treated as no-op. Unknown
                // frame types MUST be ignored (§4.1).
            }
            FramePayload::PushPromise { .. } => {
                // PUSH_PROMISE from client is illegal — server-only
                // signal. The framer accepted it (the protocol layer
                // is responsible for rejecting); we surface as a
                // protocol error in a future revision. For now,
                // silently ignore.
            }
        }
        Ok(())
    }

    fn handle_headers(
        &mut self,
        header: &FrameHeader,
        fragment: Bytes,
    ) -> Result<(), ConnectionError> {
        let end_stream = header.flags & flags::END_STREAM != 0;
        let end_headers = header.flags & flags::END_HEADERS != 0;
        let stream_id = header.stream_id;
        // SERVER role: a HEADERS opens a new peer-initiated stream. RFC §5.1.2:
        // peers exceeding our advertised MAX_CONCURRENT_STREAMS are rejected with
        // RST_STREAM(REFUSED_STREAM) (they MAY retry per §8.1.4); the connection
        // stays alive. Then register the stream up front so the state machine
        // sees recv-headers when END_HEADERS lands.
        //
        // CLIENT role: this HEADERS is a RESPONSE on a stream WE opened via
        // `send_request_head` — it already exists; no accept, no concurrency cap.
        if self.role == Role::Server {
            let local_max = self
                .local_settings
                .max_concurrent_streams
                .unwrap_or(u32::MAX) as usize;
            if self.streams.count_active() >= local_max {
                let payload = FramePayload::RstStream {
                    error_code: crate::http2_codec::frame::error_code::REFUSED_STREAM,
                };
                encode_frame(
                    FrameType::RstStream,
                    0,
                    stream_id,
                    &payload,
                    &mut self.output,
                );
                return Ok(());
            }
            self.streams.accept_client_stream(stream_id)?;
        }
        if end_headers {
            self.complete_headers(stream_id, fragment, end_stream)?;
        } else {
            let mut buffer = BytesMut::with_capacity(fragment.len() * 2);
            buffer.put_slice(&fragment);
            self.pending_headers = Some(PendingHeaders {
                stream_id,
                buffer,
                end_stream,
            });
        }
        Ok(())
    }

    fn handle_continuation(
        &mut self,
        header: &FrameHeader,
        fragment: Bytes,
    ) -> Result<(), ConnectionError> {
        let Some(mut pending) = self.pending_headers.take() else {
            return Err(ConnectionError::UnexpectedContinuation);
        };
        if header.stream_id != pending.stream_id {
            let expected = pending.stream_id;
            self.pending_headers = Some(pending);
            return Err(ConnectionError::ContinuationStreamMismatch {
                got: header.stream_id,
                expected,
            });
        }
        pending.buffer.put_slice(&fragment);
        let end_headers = header.flags & flags::END_HEADERS != 0;
        if end_headers {
            let block = pending.buffer.freeze();
            self.complete_headers(pending.stream_id, block, pending.end_stream)?;
        } else {
            self.pending_headers = Some(pending);
        }
        Ok(())
    }

    fn complete_headers(
        &mut self,
        stream_id: u32,
        block: Bytes,
        end_stream: bool,
    ) -> Result<(), ConnectionError> {
        // Reuse a returned buffer from a prior request on this same
        // connection when one's available (see `headers_scratch`'s
        // doc) — steady-state on a warm connection pays zero fresh
        // allocations here. First request (or events piled up ahead
        // of the drain loop) falls back to a fresh, pre-sized Vec:
        // typical requests have 4-10 headers (pseudo + accept/UA/...),
        // so pre-sizing skips the Vec's own double-realloc growth.
        let mut headers = self
            .headers_scratch
            .take()
            .unwrap_or_else(|| Vec::with_capacity(16));
        let settings_max = self
            .local_settings
            .header_table_size
            .unwrap_or(HPACK_DEFAULT_TABLE_SIZE) as usize;
        decode_block(
            &block,
            &mut self.decoder_table,
            settings_max,
            |name, value| {
                headers.push((name, value));
            },
        )?;
        // RFC §8.1.2: malformed head -> stream error of PROTOCOL_ERROR. Send
        // RST_STREAM and skip dispatch; the connection survives, only the
        // offending stream dies. Validation polarity follows the role: a server
        // validates an inbound REQUEST, a client an inbound RESPONSE.
        let validation = match self.role {
            Role::Server => validate_request_headers(&headers),
            Role::Client => validate_response_headers(&headers),
        };
        if let Err(reason) = validation {
            tracing::debug!(stream_id, %reason, "h2 native rejected malformed head");
            let payload = FramePayload::RstStream {
                error_code: crate::http2_codec::frame::error_code::PROTOCOL_ERROR,
            };
            encode_frame(
                FrameType::RstStream,
                0,
                stream_id,
                &payload,
                &mut self.output,
            );
            // Force-close the stream locally so we ignore further frames.
            if let Some(stream) = self.streams.get_mut(stream_id) {
                let _ = stream.on_event(StreamEvent::SendRst);
            }
            return Ok(());
        }
        // Advance per-stream state machine.
        let event = StreamEvent::RecvHeaders { end_stream };
        if let Some(stream) = self.streams.get_mut(stream_id) {
            stream
                .on_event(event)
                .map_err(|err| ConnectionError::Stream(stream_id, err))?;
        }
        let connection_event = match self.role {
            Role::Server => ConnectionEvent::RequestHead {
                stream_id,
                headers,
                end_stream,
            },
            Role::Client => ConnectionEvent::ResponseHead {
                stream_id,
                headers,
                end_stream,
            },
        };
        self.events.push_back(connection_event);
        Ok(())
    }

    fn handle_data(&mut self, header: &FrameHeader, data: Bytes) -> Result<(), ConnectionError> {
        let stream_id = header.stream_id;
        let len = u32::try_from(data.len()).unwrap_or(u32::MAX);
        self.streams.flow_mut().consume_recv(len)?;
        let end_stream = header.flags & flags::END_STREAM != 0;
        let stream = self
            .streams
            .get_mut(stream_id)
            .ok_or(ConnectionError::Stream(
                stream_id,
                StreamError::InvalidState {
                    event: StreamEvent::RecvData { end_stream, len },
                    state: crate::http2_codec::stream::StreamState::Closed,
                },
            ))?;
        stream
            .on_event(StreamEvent::RecvData { end_stream, len })
            .map_err(|err| ConnectionError::Stream(stream_id, err))?;
        // Auto-replenish flow control: if either window has dropped
        // below half the initial value, top it up. Without this we
        // deadlock on bodies > initial-window-size since the peer
        // waits for credit we never grant. (RFC §6.9.1 obliges us;
        // we choose a half-window trigger to keep frame count low.)
        let initial = i64::from(super::stream::DEFAULT_INITIAL_WINDOW_SIZE);
        let conn_window = self.streams.flow().recv_window();
        if conn_window < initial / 2 {
            let increment = (initial - conn_window) as u32;
            self.streams.flow_mut().grant_recv(increment)?;
            self.queue_window_update(0, increment);
        }
        if let Some(stream) = self.streams.get_mut(stream_id) {
            let stream_window = stream.recv_window();
            if !stream.is_closed() && stream_window < initial / 2 {
                let increment = (initial - stream_window) as u32;
                stream
                    .grant_recv_window(increment)
                    .map_err(|err| ConnectionError::Stream(stream_id, err))?;
                self.queue_window_update(stream_id, increment);
            }
        }
        self.events.push_back(ConnectionEvent::BodyData {
            stream_id,
            data,
            end_stream,
        });
        Ok(())
    }

    fn queue_window_update(&mut self, stream_id: u32, increment: u32) {
        let payload = FramePayload::WindowUpdate { increment };
        encode_frame(
            FrameType::WindowUpdate,
            0,
            stream_id,
            &payload,
            &mut self.output,
        );
    }

    fn handle_rst(&mut self, header: &FrameHeader, error_code: u32) -> Result<(), ConnectionError> {
        let stream_id = header.stream_id;
        if let Some(stream) = self.streams.get_mut(stream_id) {
            // Force-close per RFC §6.4: any state -> Closed.
            stream
                .on_event(StreamEvent::RecvRst)
                .map_err(|err| ConnectionError::Stream(stream_id, err))?;
        }
        self.events.push_back(ConnectionEvent::StreamReset {
            stream_id,
            error_code,
        });
        Ok(())
    }

    fn handle_window_update(
        &mut self,
        header: &FrameHeader,
        increment: u32,
    ) -> Result<(), ConnectionError> {
        let stream_id = header.stream_id;
        if stream_id == 0 {
            self.streams.flow_mut().grant_send(increment)?;
        } else if let Some(stream) = self.streams.get_mut(stream_id) {
            stream
                .grant_send_window(increment)
                .map_err(|err| ConnectionError::Stream(stream_id, err))?;
        }
        self.events.push_back(ConnectionEvent::WindowGranted {
            stream_id,
            increment,
        });
        Ok(())
    }

    fn handle_settings(
        &mut self,
        header: &FrameHeader,
        settings: StandardSettings,
        ack: bool,
    ) -> Result<(), ConnectionError> {
        if header.stream_id != 0 {
            return Err(ConnectionError::SettingsOnNonZeroStream(header.stream_id));
        }
        if ack {
            if header.length != 0 {
                return Err(ConnectionError::SettingsAckNonEmpty);
            }
            return Ok(());
        }
        if let Some(initial) = settings.initial_window_size {
            self.streams.apply_initial_window_change(initial)?;
        }
        if let Some(max_frame) = settings.max_frame_size {
            self.peer_max_frame_size = max_frame;
        }
        if let Some(header_table) = settings.header_table_size {
            self.encoder_table.set_max_size(header_table as usize);
        }
        self.remote_settings = settings;
        self.queue_settings_ack();
        if matches!(
            self.state,
            ConnectionState::AwaitingClientSettings | ConnectionState::AwaitingServerSettings
        ) {
            self.state = ConnectionState::Running;
        }
        self.events.push_back(ConnectionEvent::SettingsApplied);
        Ok(())
    }

    /// Queue a response head (HEADERS frame). The `headers` list must
    /// start with `:status` followed by zero-or-more regular header
    /// pairs. `end_stream = true` ends the response with no body.
    ///
    /// Drives the per-stream state machine accordingly.
    pub fn send_response_head<I>(
        &mut self,
        stream_id: u32,
        headers: I,
        end_stream: bool,
    ) -> Result<(), ConnectionError>
    where
        I: IntoIterator<Item = (Bytes, Bytes)>,
    {
        self.emit_headers(stream_id, headers, end_stream)
    }

    /// (Client) queue a REQUEST head — the mirror of [`send_response_head`].
    /// Opens a locally-initiated stream (allocate `stream_id` via
    /// [`next_local_stream_id`](Self::next_local_stream_id)), then encodes the
    /// HEADERS. The list must start with the request pseudo-headers (`:method`,
    /// `:scheme`, `:path`, `:authority`) before regular headers. `end_stream =
    /// true` is a bodyless request (e.g. GET); `false` means a request body
    /// follows via [`send_body`](Self::send_body).
    pub fn send_request_head<I>(
        &mut self,
        stream_id: u32,
        headers: I,
        end_stream: bool,
    ) -> Result<(), ConnectionError>
    where
        I: IntoIterator<Item = (Bytes, Bytes)>,
    {
        // register the outbound stream as Idle so SendHeaders advances it.
        self.streams.open_local_stream(stream_id)?;
        self.emit_headers(stream_id, headers, end_stream)
    }

    /// Encode a header list as a HEADERS frame (splitting into HEADERS +
    /// CONTINUATION across `peer_max_frame_size`, RFC §6.10) and advance the
    /// stream's send-side state machine. Shared by `send_response_head` (server)
    /// and `send_request_head` (client) — the framing is role-symmetric.
    fn emit_headers<I>(
        &mut self,
        stream_id: u32,
        headers: I,
        end_stream: bool,
    ) -> Result<(), ConnectionError>
    where
        I: IntoIterator<Item = (Bytes, Bytes)>,
    {
        // Reuse the encoder scratch buffer: clear keeps the
        // allocation, `split().freeze()` hands out a Bytes view that
        // refcount-shares the chunk and leaves the buffer ready for
        // the next head (BytesMut::split keeps remaining capacity
        // on the source).
        self.encoder_scratch.clear();
        encode_block(headers, &mut self.encoder_table, &mut self.encoder_scratch);
        let block = self.encoder_scratch.split().freeze();
        let max_frame = self.peer_max_frame_size as usize;
        let total = block.len();
        if total <= max_frame {
            // Single HEADERS frame — common case for typical
            // response sizes (status + ~5 headers ≈ 30-100 bytes).
            let payload = FramePayload::Headers {
                priority: None,
                block_fragment: block,
            };
            let mut flag_bits = flags::END_HEADERS;
            if end_stream {
                flag_bits |= flags::END_STREAM;
            }
            encode_frame(
                FrameType::Headers,
                flag_bits,
                stream_id,
                &payload,
                &mut self.output,
            );
        } else {
            // Block exceeds peer max frame size — split per RFC §6.10:
            // first chunk goes in a HEADERS frame with END_STREAM (if
            // requested) but WITHOUT END_HEADERS; remaining chunks
            // emit as CONTINUATION frames, with END_HEADERS on the
            // last one. No other frames on this connection may be
            // interleaved between the HEADERS and its CONTINUATIONs.
            let first_chunk = block.slice(0..max_frame);
            let mut flag_bits = 0_u8;
            if end_stream {
                flag_bits |= flags::END_STREAM;
            }
            let payload = FramePayload::Headers {
                priority: None,
                block_fragment: first_chunk,
            };
            encode_frame(
                FrameType::Headers,
                flag_bits,
                stream_id,
                &payload,
                &mut self.output,
            );
            let mut offset = max_frame;
            while offset < total {
                let end = (offset + max_frame).min(total);
                let chunk = block.slice(offset..end);
                let is_last = end == total;
                let cont_flags = if is_last { flags::END_HEADERS } else { 0 };
                let cont_payload = FramePayload::Continuation {
                    block_fragment: chunk,
                };
                encode_frame(
                    FrameType::Continuation,
                    cont_flags,
                    stream_id,
                    &cont_payload,
                    &mut self.output,
                );
                offset = end;
            }
        }
        if let Some(stream) = self.streams.get_mut(stream_id) {
            stream
                .on_event(StreamEvent::SendHeaders { end_stream })
                .map_err(|err| ConnectionError::Stream(stream_id, err))?;
        }
        Ok(())
    }

    /// Queue body bytes as one or more DATA frames. Window-aware:
    /// emits up to the minimum of (peer max frame size, stream send
    /// window, connection send window). If window is exhausted before
    /// all bytes are emitted, returns
    /// [`SendOutcome::WindowExhausted`] with the remainder; the
    /// caller holds it and retries once a `WindowGranted` event
    /// indicates new credit.
    ///
    /// `end_stream = true` is set on the **last** emitted DATA frame
    /// (whether that's in this call or a future resume).
    pub fn send_body(
        &mut self,
        stream_id: u32,
        data: Bytes,
        end_stream: bool,
    ) -> Result<SendOutcome, ConnectionError> {
        let total = data.len();
        let chunk_max = self.peer_max_frame_size as usize;
        if total == 0 {
            let payload = FramePayload::Data { data: Bytes::new() };
            let flag_bits = if end_stream { flags::END_STREAM } else { 0 };
            encode_frame(
                FrameType::Data,
                flag_bits,
                stream_id,
                &payload,
                &mut self.output,
            );
            if let Some(stream) = self.streams.get_mut(stream_id) {
                stream
                    .on_event(StreamEvent::SendData { end_stream, len: 0 })
                    .map_err(|err| ConnectionError::Stream(stream_id, err))?;
            }
            return Ok(SendOutcome::Done);
        }
        let mut offset = 0usize;
        while offset < total {
            let stream_credit = self
                .streams
                .get(stream_id)
                .map(crate::http2_codec::stream::Stream::send_window)
                .unwrap_or(0);
            let conn_credit = self.streams.flow().send_window();
            if stream_credit <= 0 || conn_credit <= 0 {
                return Ok(SendOutcome::WindowExhausted {
                    remainder: data.slice(offset..total),
                    end_stream,
                });
            }
            let max_chunk = stream_credit.min(conn_credit).min(chunk_max as i64) as usize;
            let take = (total - offset).min(max_chunk);
            let next = offset + take;
            let is_last = next == total;
            let chunk = data.slice(offset..next);
            let chunk_len = take as u32;
            let flag_bits = if is_last && end_stream {
                flags::END_STREAM
            } else {
                0
            };
            let payload = FramePayload::Data { data: chunk };
            encode_frame(
                FrameType::Data,
                flag_bits,
                stream_id,
                &payload,
                &mut self.output,
            );
            // Per RFC §5.2.1: DATA frames count against both the
            // stream send-window AND the connection send-window.
            self.streams.flow_mut().consume_send(chunk_len)?;
            if let Some(stream) = self.streams.get_mut(stream_id) {
                stream
                    .on_event(StreamEvent::SendData {
                        end_stream: is_last && end_stream,
                        len: chunk_len,
                    })
                    .map_err(|err| ConnectionError::Stream(stream_id, err))?;
            }
            offset = next;
        }
        Ok(SendOutcome::Done)
    }

    /// Reset a stream (we initiate). Encodes RST_STREAM and force-
    /// closes the per-stream state.
    pub fn send_rst(&mut self, stream_id: u32, error_code: u32) -> Result<(), ConnectionError> {
        let payload = FramePayload::RstStream { error_code };
        encode_frame(
            FrameType::RstStream,
            0,
            stream_id,
            &payload,
            &mut self.output,
        );
        if let Some(stream) = self.streams.get_mut(stream_id) {
            stream
                .on_event(StreamEvent::SendRst)
                .map_err(|err| ConnectionError::Stream(stream_id, err))?;
        }
        Ok(())
    }

    /// Begin graceful shutdown: queue GOAWAY and mark the connection
    /// Closing. Existing streams may continue until they reach
    /// `Closed`; no new streams will be accepted.
    pub fn send_goaway(&mut self, error_code: u32, debug_data: Bytes) {
        let payload = FramePayload::GoAway {
            last_stream_id: self.streams.last_processed_id(),
            error_code,
            debug_data,
        };
        encode_frame(FrameType::GoAway, 0, 0, &payload, &mut self.output);
        self.state = ConnectionState::Closing;
    }

    fn queue_settings_frame(&mut self, settings: &StandardSettings, ack: bool) {
        let flags_value = if ack { flags::ACK } else { 0 };
        let payload = if ack {
            FramePayload::Settings(StandardSettings::default())
        } else {
            FramePayload::Settings(settings.clone())
        };
        encode_frame(
            FrameType::Settings,
            flags_value,
            0,
            &payload,
            &mut self.output,
        );
    }

    fn queue_settings_ack(&mut self) {
        let payload = FramePayload::Settings(StandardSettings::default());
        encode_frame(
            FrameType::Settings,
            flags::ACK,
            0,
            &payload,
            &mut self.output,
        );
    }

    fn queue_ping_ack(&mut self, opaque_data: [u8; 8]) {
        let payload = FramePayload::Ping {
            opaque: opaque_data,
        };
        encode_frame(FrameType::Ping, flags::ACK, 0, &payload, &mut self.output);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::http2_codec::frame::settings_id;
    use alloc::vec;

    fn server_with_defaults() -> Connection {
        let settings = StandardSettings {
            header_table_size: Some(4096),
            initial_window_size: Some(65_535),
            max_frame_size: Some(16_384),
            ..StandardSettings::default()
        };
        Connection::new(settings)
    }

    fn build_settings(pairs: &[(u16, u32)]) -> StandardSettings {
        let mut settings = StandardSettings::default();
        for (id, value) in pairs {
            match *id {
                settings_id::HEADER_TABLE_SIZE => settings.header_table_size = Some(*value),
                settings_id::ENABLE_PUSH => settings.enable_push = Some(*value != 0),
                settings_id::MAX_CONCURRENT_STREAMS => {
                    settings.max_concurrent_streams = Some(*value);
                }
                settings_id::INITIAL_WINDOW_SIZE => settings.initial_window_size = Some(*value),
                settings_id::MAX_FRAME_SIZE => settings.max_frame_size = Some(*value),
                settings_id::MAX_HEADER_LIST_SIZE => {
                    settings.max_header_list_size = Some(*value);
                }
                _ => unreachable!("test should use known settings ids"),
            }
        }
        settings
    }

    /// Build a SETTINGS frame as a client would send it: type=4,
    /// flags=0, stream=0, payload = (id, value) pairs.
    fn client_settings_frame(pairs: &[(u16, u32)]) -> Vec<u8> {
        let settings = build_settings(pairs);
        let mut wire = Vec::new();
        encode_frame(
            FrameType::Settings,
            0,
            0,
            &FramePayload::Settings(settings),
            &mut wire,
        );
        wire
    }

    fn settings_ack_frame() -> Vec<u8> {
        let mut wire = Vec::new();
        let payload = FramePayload::Settings(StandardSettings::default());
        encode_frame(FrameType::Settings, flags::ACK, 0, &payload, &mut wire);
        wire
    }

    #[test]
    fn server_queues_initial_settings_on_construction() {
        let mut conn = server_with_defaults();
        let out = conn.take_output();
        assert!(!out.is_empty(), "server SETTINGS must be queued");
        // Wire: 9-byte header + payload. First byte of header is
        // length high; type byte at offset 3 must be SETTINGS (4).
        assert_eq!(out[3], FrameType::Settings.to_u8());
    }

    #[test]
    fn preface_match_advances_state() {
        let mut conn = server_with_defaults();
        let _ = conn.take_output();
        assert_eq!(conn.state(), ConnectionState::AwaitingPreface);
        conn.feed(CONNECTION_PREFACE).unwrap();
        assert_eq!(conn.state(), ConnectionState::AwaitingClientSettings);
    }

    #[test]
    fn preface_in_two_chunks() {
        let mut conn = server_with_defaults();
        let _ = conn.take_output();
        let (head, tail) = CONNECTION_PREFACE.split_at(10);
        conn.feed(head).unwrap();
        assert_eq!(conn.state(), ConnectionState::AwaitingPreface);
        conn.feed(tail).unwrap();
        assert_eq!(conn.state(), ConnectionState::AwaitingClientSettings);
    }

    #[test]
    fn preface_mismatch_errors_with_offset() {
        let mut conn = server_with_defaults();
        let _ = conn.take_output();
        let mut bad = CONNECTION_PREFACE.to_vec();
        bad[5] = b'X';
        let err = conn.feed(&bad).unwrap_err();
        assert!(matches!(
            err,
            ConnectionError::PrefaceMismatch { offset: 5 }
        ));
    }

    #[test]
    fn first_frame_must_be_settings() {
        let mut conn = server_with_defaults();
        let _ = conn.take_output();
        conn.feed(CONNECTION_PREFACE).unwrap();
        let mut wire = Vec::new();
        encode_frame(
            FrameType::Ping,
            0,
            0,
            &FramePayload::Ping { opaque: [0u8; 8] },
            &mut wire,
        );
        let err = conn.feed(&wire).unwrap_err();
        assert!(matches!(
            err,
            ConnectionError::MissingClientSettings {
                got: FrameType::Ping
            }
        ));
    }

    #[test]
    fn client_settings_acknowledged_and_state_advances() {
        let mut conn = server_with_defaults();
        let _server_initial = conn.take_output();
        conn.feed(CONNECTION_PREFACE).unwrap();
        let settings_wire = client_settings_frame(&[
            (settings_id::INITIAL_WINDOW_SIZE, 131_072),
            (settings_id::MAX_FRAME_SIZE, 32_768),
        ]);
        conn.feed(&settings_wire).unwrap();
        assert_eq!(conn.state(), ConnectionState::Running);
        let out = conn.take_output();
        // ACK frame: type 4, flags ACK, length 0, stream 0.
        assert_eq!(out.len(), FRAME_HEADER_LEN);
        assert_eq!(out[3], FrameType::Settings.to_u8());
        assert_eq!(out[4], flags::ACK);
        // SETTINGS-applied event surfaced.
        let event = conn.next_event().unwrap();
        assert!(matches!(event, ConnectionEvent::SettingsApplied));
    }

    #[test]
    fn settings_apply_initial_window_to_streams() {
        let mut conn = server_with_defaults();
        let _ = conn.take_output();
        conn.feed(CONNECTION_PREFACE).unwrap();
        conn.feed(&client_settings_frame(&[(
            settings_id::INITIAL_WINDOW_SIZE,
            131_072,
        )]))
        .unwrap();
        assert_eq!(conn.state(), ConnectionState::Running);
        // peer_max_frame_size still default (16,384) since we didn't bump it
        assert_eq!(conn.peer_max_frame_size, 16_384);
    }

    #[test]
    fn settings_ack_consumed_silently() {
        let mut conn = server_with_defaults();
        let _ = conn.take_output();
        conn.feed(CONNECTION_PREFACE).unwrap();
        conn.feed(&client_settings_frame(&[])).unwrap();
        let _ack = conn.take_output();
        // Now feed an ACK from client (acking our server SETTINGS).
        conn.feed(&settings_ack_frame()).unwrap();
        assert_eq!(conn.state(), ConnectionState::Running);
        // No new output for an ACK we received.
        assert!(conn.take_output().is_empty());
    }

    #[test]
    fn ping_triggers_ack() {
        let mut conn = server_with_defaults();
        let _ = conn.take_output();
        conn.feed(CONNECTION_PREFACE).unwrap();
        conn.feed(&client_settings_frame(&[])).unwrap();
        let _ack = conn.take_output();
        let opaque = *b"PINGPING";
        let mut wire = Vec::new();
        encode_frame(
            FrameType::Ping,
            0,
            0,
            &FramePayload::Ping { opaque },
            &mut wire,
        );
        conn.feed(&wire).unwrap();
        let out = conn.take_output();
        assert_eq!(out[3], FrameType::Ping.to_u8());
        assert_eq!(out[4], flags::ACK);
        assert_eq!(&out[FRAME_HEADER_LEN..], &opaque[..]);
    }

    #[test]
    fn ping_ack_does_not_re_ack() {
        let mut conn = server_with_defaults();
        let _ = conn.take_output();
        conn.feed(CONNECTION_PREFACE).unwrap();
        conn.feed(&client_settings_frame(&[])).unwrap();
        let _ack = conn.take_output();
        let mut wire = Vec::new();
        encode_frame(
            FrameType::Ping,
            flags::ACK,
            0,
            &FramePayload::Ping { opaque: [0u8; 8] },
            &mut wire,
        );
        conn.feed(&wire).unwrap();
        assert!(
            conn.take_output().is_empty(),
            "no output for received PING ACK"
        );
    }

    #[test]
    fn goaway_drives_state_to_closing() {
        let mut conn = server_with_defaults();
        let _ = conn.take_output();
        conn.feed(CONNECTION_PREFACE).unwrap();
        conn.feed(&client_settings_frame(&[])).unwrap();
        let _ack = conn.take_output();
        let _settings_event = conn.next_event();
        let payload = FramePayload::GoAway {
            last_stream_id: 0,
            error_code: 0,
            debug_data: Bytes::new(),
        };
        let mut wire = Vec::new();
        encode_frame(FrameType::GoAway, 0, 0, &payload, &mut wire);
        conn.feed(&wire).unwrap();
        assert_eq!(conn.state(), ConnectionState::Closing);
        let event = conn.next_event().unwrap();
        assert!(matches!(event, ConnectionEvent::PeerGoaway { .. }));
    }

    #[test]
    fn feed_handles_partial_frames_across_calls() {
        let mut conn = server_with_defaults();
        let _ = conn.take_output();
        let mut buf = Vec::new();
        buf.extend_from_slice(CONNECTION_PREFACE);
        buf.extend_from_slice(&client_settings_frame(&[]));
        // Feed 3 bytes at a time.
        for chunk in buf.chunks(3) {
            conn.feed(chunk).unwrap();
        }
        assert_eq!(conn.state(), ConnectionState::Running);
    }

    fn handshake(conn: &mut Connection) {
        let _ = conn.take_output();
        conn.feed(CONNECTION_PREFACE).unwrap();
        conn.feed(&client_settings_frame(&[])).unwrap();
        let _ = conn.take_output();
        while conn.next_event().is_some() {}
    }

    /// Encode a HEADERS frame: HPACK block, flags, stream id.
    fn headers_frame(stream_id: u32, block: &[u8], end_stream: bool, end_headers: bool) -> Vec<u8> {
        let mut flag_bits = 0u8;
        if end_stream {
            flag_bits |= flags::END_STREAM;
        }
        if end_headers {
            flag_bits |= flags::END_HEADERS;
        }
        let payload = FramePayload::Headers {
            priority: None,
            block_fragment: Bytes::copy_from_slice(block),
        };
        let mut wire = Vec::new();
        encode_frame(
            FrameType::Headers,
            flag_bits,
            stream_id,
            &payload,
            &mut wire,
        );
        wire
    }

    fn continuation_frame(stream_id: u32, block: &[u8], end_headers: bool) -> Vec<u8> {
        let payload = FramePayload::Continuation {
            block_fragment: Bytes::copy_from_slice(block),
        };
        let flag_bits = if end_headers { flags::END_HEADERS } else { 0 };
        let mut wire = Vec::new();
        encode_frame(
            FrameType::Continuation,
            flag_bits,
            stream_id,
            &payload,
            &mut wire,
        );
        wire
    }

    fn data_frame(stream_id: u32, data: &[u8], end_stream: bool) -> Vec<u8> {
        let payload = FramePayload::Data {
            data: Bytes::copy_from_slice(data),
        };
        let flag_bits = if end_stream { flags::END_STREAM } else { 0 };
        let mut wire = Vec::new();
        encode_frame(FrameType::Data, flag_bits, stream_id, &payload, &mut wire);
        wire
    }

    fn rst_frame(stream_id: u32, error_code: u32) -> Vec<u8> {
        let payload = FramePayload::RstStream { error_code };
        let mut wire = Vec::new();
        encode_frame(FrameType::RstStream, 0, stream_id, &payload, &mut wire);
        wire
    }

    fn window_update_frame(stream_id: u32, increment: u32) -> Vec<u8> {
        let payload = FramePayload::WindowUpdate { increment };
        let mut wire = Vec::new();
        encode_frame(FrameType::WindowUpdate, 0, stream_id, &payload, &mut wire);
        wire
    }

    /// HPACK block encoding `:method GET :scheme https :path / :authority example.com`.
    fn hpack_get_root_example_com() -> Vec<u8> {
        use crate::hpack::{DynamicTable, encode_block};
        let headers = vec![
            (Bytes::from_static(b":method"), Bytes::from_static(b"GET")),
            (Bytes::from_static(b":scheme"), Bytes::from_static(b"https")),
            (Bytes::from_static(b":path"), Bytes::from_static(b"/")),
            (
                Bytes::from_static(b":authority"),
                Bytes::from_static(b"example.com"),
            ),
        ];
        let mut buf = BytesMut::new();
        let mut table = DynamicTable::new(4096);
        encode_block(headers, &mut table, &mut buf);
        buf.to_vec()
    }

    #[test]
    fn headers_one_shot_emits_request_head() {
        let mut conn = server_with_defaults();
        handshake(&mut conn);
        let block = hpack_get_root_example_com();
        conn.feed(&headers_frame(1, &block, true, true)).unwrap();
        let event = conn.next_event().unwrap();
        let ConnectionEvent::RequestHead {
            stream_id,
            headers,
            end_stream,
        } = event
        else {
            panic!("expected RequestHead, got {event:?}");
        };
        assert_eq!(stream_id, 1);
        assert!(end_stream);
        assert_eq!(headers.len(), 4);
        assert_eq!(headers[0].0.as_ref(), b":method");
        assert_eq!(headers[0].1.as_ref(), b"GET");
        assert_eq!(headers[3].1.as_ref(), b"example.com");
    }

    #[test]
    fn headers_with_continuation_assembles_block() {
        let mut conn = server_with_defaults();
        handshake(&mut conn);
        let block = hpack_get_root_example_com();
        let (first, second) = block.split_at(block.len() / 2);
        conn.feed(&headers_frame(1, first, false, false)).unwrap();
        assert!(conn.next_event().is_none(), "no event mid-CONTINUATION");
        conn.feed(&continuation_frame(1, second, true)).unwrap();
        let event = conn.next_event().unwrap();
        assert!(matches!(
            event,
            ConnectionEvent::RequestHead { stream_id: 1, .. }
        ));
    }

    #[test]
    fn interrupting_headers_with_data_errors() {
        let mut conn = server_with_defaults();
        handshake(&mut conn);
        let block = hpack_get_root_example_com();
        let (first, _) = block.split_at(block.len() / 2);
        conn.feed(&headers_frame(1, first, false, false)).unwrap();
        let err = conn.feed(&data_frame(1, b"junk", false)).unwrap_err();
        assert!(matches!(
            err,
            ConnectionError::HeadersInterrupted { stream: 1, .. }
        ));
    }

    #[test]
    fn continuation_on_wrong_stream_errors() {
        let mut conn = server_with_defaults();
        handshake(&mut conn);
        let block = hpack_get_root_example_com();
        let (first, second) = block.split_at(block.len() / 2);
        conn.feed(&headers_frame(1, first, false, false)).unwrap();
        let err = conn.feed(&continuation_frame(3, second, true)).unwrap_err();
        assert!(matches!(
            err,
            ConnectionError::ContinuationStreamMismatch {
                got: 3,
                expected: 1
            }
        ));
    }

    #[test]
    fn data_consumes_connection_recv_window() {
        let mut conn = server_with_defaults();
        handshake(&mut conn);
        let block = hpack_get_root_example_com();
        conn.feed(&headers_frame(1, &block, false, true)).unwrap();
        let _request_head = conn.next_event().unwrap();
        let starting = conn.streams().flow().recv_window();
        conn.feed(&data_frame(1, b"hello world", true)).unwrap();
        let event = conn.next_event().unwrap();
        assert!(matches!(
            event,
            ConnectionEvent::BodyData {
                stream_id: 1,
                end_stream: true,
                ..
            }
        ));
        assert_eq!(conn.streams().flow().recv_window(), starting - 11);
    }

    #[test]
    fn rst_closes_stream_and_emits_event() {
        let mut conn = server_with_defaults();
        handshake(&mut conn);
        let block = hpack_get_root_example_com();
        conn.feed(&headers_frame(1, &block, false, true)).unwrap();
        let _ = conn.next_event();
        conn.feed(&rst_frame(1, 8)).unwrap();
        let event = conn.next_event().unwrap();
        assert!(matches!(
            event,
            ConnectionEvent::StreamReset {
                stream_id: 1,
                error_code: 8
            }
        ));
        assert!(conn.streams().get(1).unwrap().is_closed());
    }

    #[test]
    fn window_update_stream_zero_grants_connection() {
        let mut conn = server_with_defaults();
        handshake(&mut conn);
        let starting = conn.streams().flow().send_window();
        conn.feed(&window_update_frame(0, 10_000)).unwrap();
        let event = conn.next_event().unwrap();
        assert!(matches!(
            event,
            ConnectionEvent::WindowGranted {
                stream_id: 0,
                increment: 10_000
            }
        ));
        assert_eq!(conn.streams().flow().send_window(), starting + 10_000);
    }

    #[test]
    fn window_update_per_stream_grants_stream_send_window() {
        let mut conn = server_with_defaults();
        handshake(&mut conn);
        let block = hpack_get_root_example_com();
        conn.feed(&headers_frame(1, &block, false, true)).unwrap();
        let _ = conn.next_event();
        let starting = conn.streams().get(1).unwrap().send_window();
        conn.feed(&window_update_frame(1, 5_000)).unwrap();
        let _ = conn.next_event();
        assert_eq!(
            conn.streams().get(1).unwrap().send_window(),
            starting + 5_000
        );
    }

    #[test]
    fn send_response_head_advances_stream_state() {
        let mut conn = server_with_defaults();
        handshake(&mut conn);
        let block = hpack_get_root_example_com();
        conn.feed(&headers_frame(1, &block, true, true)).unwrap();
        let _ = conn.next_event();
        let response_headers = vec![
            (Bytes::from_static(b":status"), Bytes::from_static(b"200")),
            (
                Bytes::from_static(b"content-type"),
                Bytes::from_static(b"text/plain"),
            ),
        ];
        conn.send_response_head(1, response_headers, true).unwrap();
        let out = conn.take_output();
        assert!(!out.is_empty());
        assert_eq!(out[3], FrameType::Headers.to_u8());
        // END_HEADERS + END_STREAM both set.
        assert_eq!(out[4], flags::END_HEADERS | flags::END_STREAM);
        // Stream should be Closed (request END_STREAM + response END_STREAM).
        assert!(conn.streams().get(1).unwrap().is_closed());
    }

    #[test]
    fn send_body_splits_at_peer_max_frame_size() {
        let mut conn = server_with_defaults();
        handshake(&mut conn);
        let block = hpack_get_root_example_com();
        conn.feed(&headers_frame(1, &block, true, true)).unwrap();
        let _ = conn.next_event();
        let response_headers = vec![(Bytes::from_static(b":status"), Bytes::from_static(b"200"))];
        conn.send_response_head(1, response_headers, false).unwrap();
        let _ = conn.take_output();
        // 40,000 bytes fits in the 65,535 send window; chunked at the
        // 16,384 peer max frame size -> 3 DATA frames.
        let body = Bytes::from(vec![b'x'; 40_000]);
        let outcome = conn.send_body(1, body, true).unwrap();
        assert!(matches!(outcome, SendOutcome::Done));
        let out = conn.take_output();
        assert_eq!(out[3], FrameType::Data.to_u8());
        assert!(conn.streams().get(1).unwrap().is_closed());
    }

    #[test]
    fn send_body_stalls_when_send_window_exhausted() {
        let mut conn = server_with_defaults();
        handshake(&mut conn);
        let block = hpack_get_root_example_com();
        conn.feed(&headers_frame(1, &block, true, true)).unwrap();
        let _ = conn.next_event();
        let response_headers = vec![(Bytes::from_static(b":status"), Bytes::from_static(b"200"))];
        conn.send_response_head(1, response_headers, false).unwrap();
        let _ = conn.take_output();
        // 100 KiB body, send window 65,535 -> exhausts before all sent.
        let body = Bytes::from(vec![b'y'; 100 * 1024]);
        let outcome = conn.send_body(1, body, true).unwrap();
        match outcome {
            SendOutcome::WindowExhausted {
                remainder,
                end_stream,
            } => {
                assert_eq!(remainder.len(), 100 * 1024 - 65_535);
                assert!(end_stream);
            }
            SendOutcome::Done => panic!("expected window exhaustion at 65,535 / 100 KiB"),
        }
        // Stream is still HalfClosedRemote (we haven't sent END_STREAM yet).
        assert!(!conn.streams().get(1).unwrap().is_closed());
    }

    #[test]
    fn send_body_resumes_after_window_grant() {
        let mut conn = server_with_defaults();
        handshake(&mut conn);
        let block = hpack_get_root_example_com();
        conn.feed(&headers_frame(1, &block, true, true)).unwrap();
        let _ = conn.next_event();
        let response_headers = vec![(Bytes::from_static(b":status"), Bytes::from_static(b"200"))];
        conn.send_response_head(1, response_headers, false).unwrap();
        let _ = conn.take_output();
        let body = Bytes::from(vec![b'z'; 100 * 1024]);
        let outcome = conn.send_body(1, body, true).unwrap();
        let SendOutcome::WindowExhausted {
            remainder,
            end_stream,
        } = outcome
        else {
            panic!("expected window exhaustion");
        };
        // Peer grants more credit (both stream and connection windows).
        conn.feed(&window_update_frame(0, 100 * 1024)).unwrap();
        conn.feed(&window_update_frame(1, 100 * 1024)).unwrap();
        while conn.next_event().is_some() {}
        // Resume.
        let outcome = conn.send_body(1, remainder, end_stream).unwrap();
        assert!(matches!(outcome, SendOutcome::Done));
        assert!(conn.streams().get(1).unwrap().is_closed());
    }

    #[test]
    fn send_rst_force_closes_stream() {
        let mut conn = server_with_defaults();
        handshake(&mut conn);
        let block = hpack_get_root_example_com();
        conn.feed(&headers_frame(1, &block, false, true)).unwrap();
        let _ = conn.next_event();
        conn.send_rst(1, 1 /* PROTOCOL_ERROR */).unwrap();
        assert!(conn.streams().get(1).unwrap().is_closed());
        let out = conn.take_output();
        assert_eq!(out[3], FrameType::RstStream.to_u8());
    }

    #[test]
    fn send_goaway_marks_closing_and_emits_frame() {
        let mut conn = server_with_defaults();
        handshake(&mut conn);
        let block = hpack_get_root_example_com();
        conn.feed(&headers_frame(1, &block, true, true)).unwrap();
        let _ = conn.next_event();
        conn.send_goaway(0, Bytes::new());
        assert_eq!(conn.state(), ConnectionState::Closing);
        let out = conn.take_output();
        assert_eq!(out[3], FrameType::GoAway.to_u8());
    }

    #[test]
    fn request_response_roundtrip_through_two_connections() {
        // Client side acts as a peer to drive the bytes.
        let mut server = server_with_defaults();
        let mut server_output = server.take_output();
        // Client preface + SETTINGS.
        let mut client_buf = Vec::new();
        client_buf.extend_from_slice(CONNECTION_PREFACE);
        client_buf.extend_from_slice(&client_settings_frame(&[]));
        // Drive server through handshake.
        server.feed(&client_buf).unwrap();
        // Drain the server's SETTINGS-ACK from output.
        let _ = server.take_output();
        // Drain SettingsApplied event.
        while server.next_event().is_some() {}
        // Acknowledge server's initial SETTINGS so it doesn't accumulate.
        server.feed(&settings_ack_frame()).unwrap();

        // Client sends a GET.
        let request_block = hpack_get_root_example_com();
        server
            .feed(&headers_frame(1, &request_block, true, true))
            .unwrap();
        let event = server.next_event().unwrap();
        assert!(matches!(
            event,
            ConnectionEvent::RequestHead { stream_id: 1, .. }
        ));

        // Server emits a response with a body.
        let response = vec![
            (Bytes::from_static(b":status"), Bytes::from_static(b"200")),
            (
                Bytes::from_static(b"content-type"),
                Bytes::from_static(b"text/plain"),
            ),
            (
                Bytes::from_static(b"content-length"),
                Bytes::from_static(b"11"),
            ),
        ];
        server.send_response_head(1, response, false).unwrap();
        server
            .send_body(1, Bytes::from_static(b"hello world"), true)
            .unwrap();

        // Server output now contains HEADERS + DATA. Stream is Closed.
        let wire = server.take_output();
        assert!(!wire.is_empty());
        assert!(server.streams().get(1).unwrap().is_closed());

        // Sanity: discard initial SETTINGS prefix from server_output.
        let _ = server_output.split_off(0);
    }

    /// DC-H2-HEADERS-SCRATCH-REUSE — mechanically re-provable proof
    /// (P16) that `return_headers_buffer` actually gets a request's
    /// second-and-later `Vec<(Bytes,Bytes)>` allocation reused rather
    /// than fresh: two GET requests over the SAME connection, with the
    /// first request's `headers` returned to the pool before the
    /// second is fed, must yield the SAME backing pointer for both.
    /// This is the substrate behind the e2e alloc-count claim in
    /// `docs/proxima-quic/discipline.md` — that claim's "steady-state
    /// warm connection" precondition IS this behavior.
    #[test]
    fn headers_scratch_buffer_is_reused_across_requests_on_same_connection() {
        let mut server = server_with_defaults();
        let mut client_buf = Vec::new();
        client_buf.extend_from_slice(CONNECTION_PREFACE);
        client_buf.extend_from_slice(&client_settings_frame(&[]));
        server.feed(&client_buf).unwrap();
        let _ = server.take_output();
        while server.next_event().is_some() {}
        server.feed(&settings_ack_frame()).unwrap();

        // First request (stream 1) — no spare buffer exists yet, so
        // this one is a fresh Vec::with_capacity(16).
        let request_block = hpack_get_root_example_com();
        server
            .feed(&headers_frame(1, &request_block, true, true))
            .unwrap();
        let ConnectionEvent::RequestHead {
            headers: first_headers,
            ..
        } = server.next_event().unwrap()
        else {
            panic!("expected RequestHead");
        };
        let first_capacity = first_headers.capacity();
        let first_ptr = first_headers.as_ptr();
        assert!(first_capacity >= 4, "pre-sized for the typical case");

        // Consumer (the I/O facade) is done reading `first_headers` —
        // return it to the connection's pool.
        server.return_headers_buffer(first_headers);

        // Second request (stream 3) — must reuse the returned buffer's
        // allocation instead of paying a fresh Vec::with_capacity.
        let request_block_two = hpack_get_root_example_com();
        server
            .feed(&headers_frame(3, &request_block_two, true, true))
            .unwrap();
        let ConnectionEvent::RequestHead {
            headers: second_headers,
            ..
        } = server.next_event().unwrap()
        else {
            panic!("expected RequestHead");
        };

        assert_eq!(
            second_headers.as_ptr(),
            first_ptr,
            "second request's headers Vec must reuse the SAME backing allocation \
             the first request's headers Vec returned — this is the mechanism \
             behind the measured allocs/req reduction, not just 'didn't panic'"
        );
        assert_eq!(
            second_headers.capacity(),
            first_capacity,
            "reused allocation's capacity must be unchanged (no growth needed \
             for a same-shaped request)"
        );
    }

    /// CLIENT↔SERVER loopback worked example (RFC 7540 §3.5 handshake + §8.1
    /// request/response). A `new_client` Connection and the proven `new` server
    /// Connection exchange real wire bytes through a full unary POST with bodies.
    /// This is the parity proof (principle 14): the client role is byte-compatible
    /// with the server it will talk to. The grpc-shaped DATA (`flag||len||msg`)
    /// is opaque to h2 — it's realistic payload (principle 9), the gRPC codec
    /// frames it one layer up.
    fn pump(from: &mut Connection, to: &mut Connection) {
        let out = from.take_output();
        if !out.is_empty() {
            to.feed(&out).unwrap();
        }
    }

    #[test]
    fn client_server_loopback_unary_request_response() {
        let settings = StandardSettings {
            header_table_size: Some(4096),
            initial_window_size: Some(65_535),
            max_frame_size: Some(16_384),
            ..StandardSettings::default()
        };
        let mut client = Connection::new_client(settings.clone());
        let mut server = Connection::new(settings);

        // Handshake: client preface+SETTINGS -> server; server SETTINGS+ACK ->
        // client; client ACK -> server. Both reach Running.
        pump(&mut client, &mut server);
        pump(&mut server, &mut client);
        pump(&mut client, &mut server);
        assert_eq!(client.state(), ConnectionState::Running);
        assert_eq!(server.state(), ConnectionState::Running);
        while client.next_event().is_some() {}
        while server.next_event().is_some() {}

        // Client opens stream 1 with a unary POST + a grpc-framed body.
        let stream_id = client.next_local_stream_id();
        assert_eq!(stream_id, 1, "client streams are odd, starting at 1");
        let request = vec![
            (Bytes::from_static(b":method"), Bytes::from_static(b"POST")),
            (Bytes::from_static(b":scheme"), Bytes::from_static(b"https")),
            (
                Bytes::from_static(b":path"),
                Bytes::from_static(b"/svc.Service/Method"),
            ),
            (
                Bytes::from_static(b":authority"),
                Bytes::from_static(b"collector"),
            ),
            (
                Bytes::from_static(b"content-type"),
                Bytes::from_static(b"application/grpc"),
            ),
        ];
        client.send_request_head(stream_id, request, false).unwrap();
        client
            .send_body(
                stream_id,
                Bytes::from_static(b"\x00\x00\x00\x00\x05hello"),
                true,
            )
            .unwrap();
        pump(&mut client, &mut server);

        let mut server_head = None;
        let mut server_body = None;
        while let Some(event) = server.next_event() {
            match event {
                ConnectionEvent::RequestHead {
                    stream_id,
                    headers,
                    end_stream,
                } => {
                    server_head = Some((stream_id, headers, end_stream));
                }
                ConnectionEvent::BodyData {
                    stream_id,
                    data,
                    end_stream,
                } => {
                    server_body = Some((stream_id, data, end_stream));
                }
                _ => {}
            }
        }
        let (sid, headers, end_stream) = server_head.expect("server saw the request head");
        assert_eq!(sid, 1);
        assert!(!end_stream, "request body follows");
        assert_eq!(headers[0].1.as_ref(), b"POST");
        let (bsid, data, bend) = server_body.expect("server saw the request body");
        assert_eq!(bsid, 1);
        assert!(bend);
        assert_eq!(data.as_ref(), b"\x00\x00\x00\x00\x05hello");

        // Server responds with status + a grpc-framed body.
        let response = vec![
            (Bytes::from_static(b":status"), Bytes::from_static(b"200")),
            (
                Bytes::from_static(b"content-type"),
                Bytes::from_static(b"application/grpc"),
            ),
        ];
        server.send_response_head(1, response, false).unwrap();
        server
            .send_body(1, Bytes::from_static(b"\x00\x00\x00\x00\x03bye"), true)
            .unwrap();
        pump(&mut server, &mut client);

        let mut client_head = None;
        let mut client_body = None;
        while let Some(event) = client.next_event() {
            match event {
                ConnectionEvent::ResponseHead {
                    stream_id,
                    headers,
                    end_stream,
                } => {
                    client_head = Some((stream_id, headers, end_stream));
                }
                ConnectionEvent::BodyData {
                    stream_id,
                    data,
                    end_stream,
                } => {
                    client_body = Some((stream_id, data, end_stream));
                }
                _ => {}
            }
        }
        let (sid, headers, end_stream) = client_head.expect("client saw the response head");
        assert_eq!(sid, 1);
        assert!(!end_stream, "response body follows");
        assert_eq!(headers[0].0.as_ref(), b":status");
        assert_eq!(headers[0].1.as_ref(), b"200");
        let (bsid, data, bend) = client_body.expect("client saw the response body");
        assert_eq!(bsid, 1);
        assert!(bend);
        assert_eq!(data.as_ref(), b"\x00\x00\x00\x00\x03bye");

        assert!(
            client.streams().get(1).unwrap().is_closed(),
            "client stream closed"
        );
        assert!(
            server.streams().get(1).unwrap().is_closed(),
            "server stream closed"
        );
    }
}
