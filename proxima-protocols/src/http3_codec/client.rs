//! Client-side HTTP/3 connection state machine per [RFC 9114].
//!
//! Symmetric to [`crate::http3_codec::server::ServerConnection`] — same SETTINGS
//! exchange, same per-request state machines but the directions are
//! swapped: the client sends a request (HEADERS + DATA + optional
//! trailers + FIN) and receives a response.
//!
//! Per-request state lives in a stream-id-keyed
//! `heapless::FnvIndexMap<u64, StreamEntry, MAX>` (boxed). Cap is
//! [`crate::sized::PROXIMA_PROTOCOLS_HTTP3_CODEC_CLIENT_MAX_CONCURRENT_REQUESTS`];
//! `open_request` returns [`ClientError::RequestMapFull`] when the
//! local cap would be exceeded.
//!
//! [RFC 9114]: https://www.rfc-editor.org/rfc/rfc9114

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::vec::Vec;

/// Map cap — pulled from `sized.rs`. heapless requires a power of two.
const STREAMS_CAP: usize = crate::sized::PROXIMA_PROTOCOLS_HTTP3_CODEC_CLIENT_MAX_CONCURRENT_REQUESTS;
/// Ceiling on recycled outbound buffers kept in the free-list.
const OUTBOUND_POOL_CAP: usize = 64;
type StreamMap = heapless::index_map::FnvIndexMap<u64, StreamEntry, STREAMS_CAP>;

use crate::http3_codec::frame::{self, H3Frame};
use crate::http3_codec::qpack;
use crate::http3_codec::request::{RecvState, RequestError, SendState};
use crate::http3_codec::server::StreamId;
use crate::http3_codec::settings::{Settings, SettingsError};

/// Client-connection state.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ClientState {
    Negotiating,
    Established,
    Closing { peer_max_stream_id: u64 },
}

/// Response-HEADERS delivery mode (`part-source` feature only). See
/// [`ClientConnection::enable_header_source_mode`]. `Owned` (the default)
/// preserves the pre-existing `H3ClientEvent::ResponseHeaders` event
/// byte-for-byte; every connection starts here, so a caller that never
/// opts in sees no behavior change from this row.
#[cfg(feature = "http3_codec-part-source")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResponseHeaderMode {
    #[default]
    Owned,
    /// Response HEADERS field sections are queued RAW (copied once into a
    /// recycled per-connection buffer — steady-state 0 heap allocations)
    /// and decoded lazily: [`ClientConnection::poll_response_header_source`]
    /// hands back a borrowed [`qpack::part_source::FieldSectionSource`]
    /// that steps one field per call, instead of the owned
    /// `ResponseHeaders` event. Deferred-validation contract: a malformed
    /// section surfaces when the CALLER steps the source
    /// ([`qpack::part_source::FieldSectionSource::error`]), not at
    /// `feed_response` time — the caller MUST drain the source queue and
    /// treat a reported error as a QPACK decompression failure
    /// (connection-fatal, RFC 9204 §2.2.3).
    Source,
}

/// Client-side connection event.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum H3ClientEvent {
    SettingsEstablished {
        peer: Settings,
    },
    /// Server sent the response HEADERS frame.
    ResponseHeaders {
        stream_id: StreamId,
        /// Extracted `:status` pseudo-header (RFC 9114 §4.3.2) — captured
        /// inline while validating the field section via
        /// [`qpack::decoder::decode_into`] (the 0-alloc borrowing engine),
        /// so this Copy field costs nothing to produce. `None` if `:status`
        /// was absent or not a valid `u16`. Most callers only need this.
        status: Option<u16>,
        /// Owned copy of the still-QPACK-encoded field-section bytes, for
        /// callers that need full header enumeration beyond `:status` (e.g.
        /// forwarding response headers through a proxy). Decode with
        /// [`qpack::decoder::decode_into`] (borrowing, 0-alloc) or
        /// [`qpack::decoder::decode_bounded`] (owned `Vec<DecodedField>`).
        /// This is ONE allocation — needed regardless to carry the bytes
        /// across the event-queue boundary (`pending_events` outlives the
        /// caller's inbound-byte borrow) — versus the prior `1 + 2 *
        /// field_count` from eagerly materializing a `Vec<DecodedField>`
        /// for every response whether or not the caller ever reads past
        /// `:status`. Closes the client half of `DC-H3-FACADE-EVENTS-OWN`
        /// (see `docs/proxima-quic/alloc-budget.md`); the server half
        /// (C35) is a separate, still-open redesign.
        header_block: Vec<u8>,
    },
    /// Server sent body bytes.
    ResponseData {
        stream_id: StreamId,
        bytes: Vec<u8>,
    },
    /// Server sent optional trailing HEADERS. No consumer reads trailer
    /// fields today, so this is a signal event only — the field section is
    /// still validated (cap-enforced, malformed-input-rejected) exactly as
    /// before, the decoded fields just aren't retained. A future caller
    /// that needs trailers can be added the same way `ResponseHeaders`
    /// carries `header_block`.
    ResponseTrailers {
        stream_id: StreamId,
    },
    /// Server set FIN on the response stream.
    ResponseFinished {
        stream_id: StreamId,
    },
    /// Server sent GOAWAY.
    GoAway {
        peer_max_stream_id: u64,
    },
}

/// Errors from client-connection operations.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ClientError {
    Settings(SettingsError),
    Request(RequestError),
    Qpack(qpack::decoder::DecodeError),
    IllegalInState {
        state: &'static str,
        method: &'static str,
    },
    /// Per-connection request table is at its build-time cap
    /// ([`crate::sized::PROXIMA_PROTOCOLS_HTTP3_CODEC_CLIENT_MAX_CONCURRENT_REQUESTS`]).
    /// The caller must wait for an in-flight request to finish before
    /// opening another, or rebuild with a larger cap via the env
    /// override `PROXIMA_PROTOCOLS_HTTP3_CODEC_CLIENT_MAX_CONCURRENT_REQUESTS`.
    RequestMapFull {
        cap: usize,
    },
}

impl From<SettingsError> for ClientError {
    fn from(err: SettingsError) -> Self {
        Self::Settings(err)
    }
}
impl From<RequestError> for ClientError {
    fn from(err: RequestError) -> Self {
        Self::Request(err)
    }
}
impl From<qpack::decoder::DecodeError> for ClientError {
    fn from(err: qpack::decoder::DecodeError) -> Self {
        Self::Qpack(err)
    }
}

struct StreamEntry {
    recv: RecvState,
    send: SendState,
    outbound: Vec<u8>,
    fin_pending: bool,
}

/// HTTP/3 client connection.
pub struct ClientConnection {
    state: ClientState,
    local_settings: Settings,
    peer_settings: Option<Settings>,
    /// boxed for the same reason as `ServerConnection::requests` —
    /// the inline map is fat and would push the connection past
    /// test-thread stack capacity.
    streams: Box<StreamMap>,
    next_client_bidi_stream_id: u64,
    pending_events: VecDeque<H3ClientEvent>,
    control_outbound: Vec<u8>,
    /// Reused scratch for QPACK-encoding request header blocks — cleared and
    /// refilled per `open_request` so the hot path allocates once (grows to
    /// the largest header block seen), not a fresh `Vec` per request.
    request_encode_scratch: Vec<u8>,
    /// The last request's fully-encoded H3 HEADERS frame, kept alongside the
    /// header set that produced it ([`cached_request_headers`]). QPACK
    /// encoding here is stateless (static-table only, no dynamic table), so an
    /// identical header set always yields identical bytes — a client sending
    /// the same request repeatedly (the common keep-alive / bench shape) hits
    /// this cache and memcpys the frame instead of re-encoding it per request.
    cached_request_frame: Vec<u8>,
    /// The header set [`cached_request_frame`] corresponds to (empty = no
    /// frame cached yet). Compared by value against each `open_request`'s
    /// headers; a byte-equal match reuses the cached frame.
    cached_request_headers: Vec<(Vec<u8>, Vec<u8>)>,
    /// Free-list of drained per-stream outbound buffers (alloc tier: reuse the
    /// heap allocation rather than `Vec::new()` per request). A completed
    /// stream's empty-but-capacity buffer is recycled here in `gc_completed`
    /// and handed back out by `open_request`. Bounded so it can't grow without
    /// limit. (At no_std+no_alloc this whole connection is absent — the
    /// no-alloc tier exposes only the leaf parse subset.)
    outbound_pool: Vec<Vec<u8>>,
    settings_emitted: bool,
    /// RFC 9114 §7.2.4 — see [`super::server::ServerConnection`] for
    /// the same invariant on the symmetric path.
    peer_settings_seen: bool,
    /// See [`ResponseHeaderMode`]. Default `Owned` — this row is fully
    /// opt-in (`part-source` feature only).
    #[cfg(feature = "http3_codec-part-source")]
    response_header_mode: ResponseHeaderMode,
    /// Raw response-HEADERS field sections queued while
    /// `response_header_mode` is `Source`, drained via
    /// [`Self::poll_response_header_source`] — pool-recycled small
    /// values, steady-state 0 allocations (the queued-by-value
    /// `HeaderBlockPartSource` this replaces moved its multi-KB inline
    /// arena per queue hop, which cost more than the one allocation it
    /// removed — `docs/proxima-pipe/discipline.md` C2's honest negative,
    /// fixed by C3).
    #[cfg(feature = "http3_codec-part-source")]
    header_source_queue: qpack::part_source::HeaderBlockQueue<StreamId>,
}

impl ClientConnection {
    /// Construct with local SETTINGS values.
    #[must_use]
    pub fn new(local_settings: Settings) -> Self {
        Self {
            state: ClientState::Negotiating,
            local_settings,
            peer_settings: None,
            streams: Box::new(StreamMap::new()),
            // Client-initiated bidi streams use IDs 0, 4, 8, 12, …
            // (RFC 9000 §2.1).
            next_client_bidi_stream_id: 0,
            pending_events: VecDeque::new(),
            control_outbound: Vec::new(),
            request_encode_scratch: Vec::new(),
            cached_request_frame: Vec::new(),
            cached_request_headers: Vec::new(),
            outbound_pool: Vec::new(),
            settings_emitted: false,
            peer_settings_seen: false,
            #[cfg(feature = "http3_codec-part-source")]
            response_header_mode: ResponseHeaderMode::default(),
            #[cfg(feature = "http3_codec-part-source")]
            header_source_queue: qpack::part_source::HeaderBlockQueue::new(),
        }
    }

    /// Opt this connection into [`ResponseHeaderMode::Source`] — every
    /// subsequent response's HEADERS frame is queued for
    /// [`Self::poll_response_header_source`] instead of the owned
    /// `H3ClientEvent::ResponseHeaders` event. Call once, before the first
    /// response arrives; switching mid-stream leaves in-flight streams on
    /// whichever mode was active when their HEADERS landed. Sizes the
    /// Huffman scratch once (the mode's only setup allocation).
    #[cfg(feature = "http3_codec-part-source")]
    pub fn enable_header_source_mode(&mut self) {
        self.response_header_mode = ResponseHeaderMode::Source;
        self.header_source_queue
            .size_scratch(crate::sized::PROXIMA_PROTOCOLS_HTTP3_CODEC_QPACK_DECODE_BOUNDED_SCRATCH_LEN);
    }

    /// Drain one queued response-HEADERS field section as a borrowed
    /// [`qpack::part_source::FieldSectionSource`] — only populated when
    /// [`ResponseHeaderMode::Source`] is active (see
    /// [`Self::enable_header_source_mode`]). Step the returned source via
    /// [`proxima_primitives::pipe::part::PartSource::next`] to read `:status` +
    /// headers at 0 heap allocations, then CHECK
    /// [`qpack::part_source::FieldSectionSource::error`] — decode
    /// failures are deferred to stepping (see [`ResponseHeaderMode`]) and
    /// a reported error is connection-fatal. The source borrows this
    /// connection; drop it before the next poll.
    #[cfg(feature = "http3_codec-part-source")]
    #[must_use]
    pub fn poll_response_header_source(
        &mut self,
    ) -> Option<(StreamId, qpack::part_source::FieldSectionSource<'_>)> {
        let cap = self.local_settings.max_field_section_size;
        self.header_source_queue.poll(cap)
    }

    /// Current connection state.
    #[must_use]
    pub fn state(&self) -> &ClientState {
        &self.state
    }

    /// Locally-advertised SETTINGS.
    #[must_use]
    pub fn local_settings(&self) -> &Settings {
        &self.local_settings
    }

    /// Peer-advertised SETTINGS, available after SETTINGS exchange.
    #[must_use]
    pub fn peer_settings(&self) -> Option<&Settings> {
        self.peer_settings.as_ref()
    }

    /// Feed inbound bytes from the server's control stream.
    ///
    /// # Errors
    ///
    /// See [`ClientError`].
    pub fn feed_control(&mut self, bytes: &[u8]) -> Result<(), ClientError> {
        let mut cursor = 0;
        while cursor < bytes.len() {
            let (frame_obj, consumed) = frame::parse(&bytes[cursor..])
                .map_err(|err| ClientError::Request(RequestError::Frame(err)))?;
            cursor += consumed;
            match frame_obj {
                H3Frame::Settings { payload } => {
                    if self.peer_settings_seen {
                        return Err(ClientError::Settings(SettingsError::DuplicateFrame));
                    }
                    let mut peer = Settings::default();
                    peer.apply_payload(payload)?;
                    self.peer_settings = Some(peer);
                    self.peer_settings_seen = true;
                    if matches!(self.state, ClientState::Negotiating) {
                        self.state = ClientState::Established;
                    }
                    self.pending_events
                        .push_back(H3ClientEvent::SettingsEstablished { peer });
                }
                _ if !self.peer_settings_seen => {
                    let observed_id = match frame_obj {
                        H3Frame::GoAway { .. } => 0x07,
                        H3Frame::CancelPush { .. } => 0x03,
                        H3Frame::MaxPushId { .. } => 0x0d,
                        H3Frame::Data { .. } => 0x00,
                        H3Frame::Headers { .. } => 0x01,
                        H3Frame::PushPromise { .. } => 0x05,
                        H3Frame::Reserved { frame_type, .. } => frame_type,
                        H3Frame::Settings { .. } => 0x04, // unreachable
                    };
                    return Err(ClientError::Settings(SettingsError::MissingSettings {
                        observed_id,
                    }));
                }
                H3Frame::GoAway { id } => {
                    self.state = ClientState::Closing {
                        peer_max_stream_id: id,
                    };
                    self.pending_events.push_back(H3ClientEvent::GoAway {
                        peer_max_stream_id: id,
                    });
                }
                H3Frame::CancelPush { .. } => {
                    // Allowed on the server's control stream;
                    // client doesn't act on it yet (push disabled).
                }
                H3Frame::MaxPushId { .. } => {
                    // RFC 9114 §7.2.7 — MAX_PUSH_ID is sent by the
                    // client to the server. A server MUST NOT send it;
                    // a client receiving it MUST close with
                    // H3_FRAME_UNEXPECTED.
                    return Err(ClientError::Request(RequestError::UnexpectedFrame));
                }
                H3Frame::Reserved { frame_type, .. } => {
                    // RFC 9114 §11.2.1 — HTTP/2-reserved types MUST
                    // be a connection error; §7.2.8 — other reserved
                    // types MUST be ignored.
                    if frame::is_http2_reserved(frame_type) {
                        return Err(ClientError::Request(RequestError::UnexpectedFrame));
                    }
                }
                H3Frame::Data { .. } | H3Frame::Headers { .. } | H3Frame::PushPromise { .. } => {
                    return Err(ClientError::Request(RequestError::UnexpectedFrame));
                }
            }
        }
        self.ensure_settings_emitted();
        Ok(())
    }

    /// Feed inbound bytes from a response stream.
    ///
    /// # Errors
    ///
    /// See [`ClientError`].
    pub fn feed_response(
        &mut self,
        stream_id: StreamId,
        bytes: &[u8],
        fin: bool,
    ) -> Result<(), ClientError> {
        if !self.streams.contains_key(&stream_id.0) {
            let fresh = StreamEntry {
                recv: RecvState::Idle,
                send: SendState::Idle,
                outbound: Vec::new(),
                fin_pending: false,
            };
            self.streams
                .insert(stream_id.0, fresh)
                .map_err(|_| ClientError::RequestMapFull { cap: STREAMS_CAP })?;
        }
        let max_field_section_size = self.local_settings.max_field_section_size;
        #[cfg(feature = "http3_codec-part-source")]
        let response_header_mode = self.response_header_mode;
        let entry = self
            .streams
            .get_mut(&stream_id.0)
            .ok_or(ClientError::RequestMapFull { cap: STREAMS_CAP })?;
        let mut cursor = 0;
        while cursor < bytes.len() {
            let (frame_obj, consumed) = frame::parse(&bytes[cursor..])
                .map_err(|err| ClientError::Request(RequestError::Frame(err)))?;
            cursor += consumed;
            // Opt-in reroute (default mode `Owned` never takes this branch, so
            // `apply_response_frame`'s behavior below is unchanged): a
            // connection in `ResponseHeaderMode::Source` queues its response
            // HEADERS field section raw on `pending_header_blocks`, decoded
            // lazily at `poll_response_header_source` time instead of built
            // into the owned `ResponseHeaders` event.
            #[cfg(feature = "http3_codec-part-source")]
            if response_header_mode == ResponseHeaderMode::Source
                && matches!(entry.recv, RecvState::Idle)
                && let H3Frame::Headers { header_block } = frame_obj
            {
                // the client never re-reads the state copy (only the server
                // accumulates via recv.headers()) — carry an empty vec, the
                // section rides the block queue instead. see
                // `HeaderBlockQueue` for the deferred-validation contract.
                entry.recv = RecvState::HeadersReceived {
                    headers: Vec::new(),
                };
                self.header_source_queue.push(stream_id, header_block);
                continue;
            }
            apply_response_frame(
                stream_id,
                &mut entry.recv,
                frame_obj,
                &mut self.pending_events,
                max_field_section_size,
            )?;
        }
        if fin {
            entry.recv = RecvState::Done;
            self.pending_events
                .push_back(H3ClientEvent::ResponseFinished { stream_id });
        }
        Ok(())
    }

    /// Send a request — opens a fresh client-initiated bidi stream,
    /// queues HEADERS, returns the new stream ID. The caller follows
    /// up with `send_request_data` + `finish_request` as needed.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::IllegalInState`] when called before
    /// SETTINGS exchange completes.
    pub fn open_request(&mut self, headers: &[(&[u8], &[u8])]) -> Result<StreamId, ClientError> {
        // RFC 9114 §5.2 — after receiving GOAWAY the client MUST NOT
        // open new requests on streams above the indicated ID; the
        // server will not process them. ClientState::Closing carries
        // the peer_max_stream_id but we reject ALL new requests in
        // that state for simplicity.
        if !matches!(self.state, ClientState::Established) {
            return Err(ClientError::IllegalInState {
                state: state_label(&self.state),
                method: "open_request",
            });
        }
        let stream_id = StreamId(self.next_client_bidi_stream_id);
        self.next_client_bidi_stream_id = self.next_client_bidi_stream_id.saturating_add(4);
        let outbound = self.take_outbound();
        let mut entry = StreamEntry {
            recv: RecvState::Idle,
            send: SendState::Idle,
            outbound,
            fin_pending: false,
        };
        if !headers_match(&self.cached_request_headers, headers) {
            self.request_encode_scratch.clear();
            qpack::encoder::encode_refs(headers.iter().copied(), &mut self.request_encode_scratch)
                .map_err(|_| {
                    ClientError::Request(RequestError::Frame(frame::FrameError::BufferTooSmall {
                        needed: 0,
                    }))
                })?;
            self.cached_request_frame.clear();
            encode_h3_frame(
                &H3Frame::Headers {
                    header_block: &self.request_encode_scratch,
                },
                &mut self.cached_request_frame,
            )?;
            self.cached_request_headers.clear();
            self.cached_request_headers.extend(
                headers
                    .iter()
                    .map(|(name, value)| (name.to_vec(), value.to_vec())),
            );
        }
        entry.outbound.extend_from_slice(&self.cached_request_frame);
        entry.send = SendState::HeadersSent;
        self.streams
            .insert(stream_id.0, entry)
            .map_err(|_| ClientError::RequestMapFull { cap: STREAMS_CAP })?;
        Ok(stream_id)
    }

    /// Append body data to a request.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::IllegalInState`] when the stream isn't in
    /// `HeadersSent` / `BodyStreaming`.
    pub fn send_request_data(
        &mut self,
        stream_id: StreamId,
        bytes: &[u8],
    ) -> Result<(), ClientError> {
        let entry = self
            .streams
            .get_mut(&stream_id.0)
            .ok_or(ClientError::IllegalInState {
                state: state_label(&self.state),
                method: "send_request_data (unknown stream)",
            })?;
        match entry.send {
            SendState::HeadersSent => {
                entry.send = SendState::BodyStreaming {
                    bytes_sent: bytes.len() as u64,
                };
            }
            SendState::BodyStreaming { ref mut bytes_sent } => {
                *bytes_sent = bytes_sent.saturating_add(bytes.len() as u64);
            }
            _ => {
                return Err(ClientError::IllegalInState {
                    state: send_label(&entry.send),
                    method: "send_request_data",
                });
            }
        }
        encode_h3_frame(&H3Frame::Data { payload: bytes }, &mut entry.outbound)?;
        Ok(())
    }

    /// Mark the request complete — FIN bit on next take.
    ///
    /// # Errors
    ///
    /// See [`ClientError::IllegalInState`].
    pub fn finish_request(&mut self, stream_id: StreamId) -> Result<(), ClientError> {
        let entry = self
            .streams
            .get_mut(&stream_id.0)
            .ok_or(ClientError::IllegalInState {
                state: state_label(&self.state),
                method: "finish_request (unknown stream)",
            })?;
        if matches!(entry.send, SendState::Done) {
            return Err(ClientError::IllegalInState {
                state: send_label(&entry.send),
                method: "finish_request",
            });
        }
        entry.fin_pending = true;
        entry.send = SendState::Done;
        Ok(())
    }

    /// Drain pending events.
    #[must_use]
    pub fn poll_event(&mut self) -> Option<H3ClientEvent> {
        self.pending_events.pop_front()
    }

    /// Drain outbound control-stream bytes.
    ///
    /// RFC 9114 §6.2.1 — both endpoints MUST initiate their control
    /// stream + ship SETTINGS as the first frame. We stage SETTINGS
    /// proactively on first drain so callers don't need to wait for
    /// peer input to flush their own.
    pub fn take_control_outbound(&mut self) -> Vec<u8> {
        self.ensure_settings_emitted();
        core::mem::take(&mut self.control_outbound)
    }

    /// Drain outbound bytes + FIN bit for a request stream.
    pub fn take_request_outbound(&mut self, stream_id: StreamId) -> Option<(Vec<u8>, bool)> {
        let entry = self.streams.get_mut(&stream_id.0)?;
        let bytes = core::mem::take(&mut entry.outbound);
        let fin = core::mem::replace(&mut entry.fin_pending, false);
        Some((bytes, fin))
    }

    /// Remove all fully-completed stream entries. See
    /// [`super::server::ServerConnection::gc_completed`].
    /// Hand out a cleared outbound buffer, reusing a recycled allocation when
    /// one is free (alloc tier: avoid a per-request `Vec::new()`).
    fn take_outbound(&mut self) -> Vec<u8> {
        self.outbound_pool
            .pop()
            .map(|mut buf| {
                buf.clear();
                buf
            })
            .unwrap_or_default()
    }

    /// Return a drained outbound buffer to the free-list, up to the cap.
    fn recycle_outbound(&mut self, buf: Vec<u8>) {
        if self.outbound_pool.len() < OUTBOUND_POOL_CAP {
            self.outbound_pool.push(buf);
        }
    }

    pub fn gc_completed(&mut self) {
        let mut done: heapless::Vec<u64, STREAMS_CAP> = heapless::Vec::new();
        for key in self.streams.keys().copied() {
            if self.streams.get(&key).is_some_and(|entry| {
                matches!(entry.recv, RecvState::Done)
                    && matches!(entry.send, SendState::Done)
                    && entry.outbound.is_empty()
                    && !entry.fin_pending
            }) {
                let _ = done.push(key);
            }
        }
        for key in done {
            if let Some(entry) = self.streams.swap_remove(&key) {
                self.recycle_outbound(entry.outbound);
            }
        }
    }

    /// Iterate the IDs of every request stream this client has opened.
    /// Used by the per-connection driver to know which QUIC bidi
    /// streams it needs to bridge.
    pub fn request_stream_ids(&self) -> impl Iterator<Item = StreamId> + '_ {
        self.streams.keys().copied().map(StreamId)
    }

    fn ensure_settings_emitted(&mut self) {
        if self.settings_emitted {
            return;
        }
        let mut payload = [0u8; 64];
        let mut cursor = 0;
        use crate::quic::varint;
        cursor += varint::encode(
            crate::http3_codec::settings::SETTINGS_QPACK_MAX_TABLE_CAPACITY,
            &mut payload[cursor..],
        )
        .unwrap_or(0);
        cursor += varint::encode(
            self.local_settings.qpack_max_table_capacity,
            &mut payload[cursor..],
        )
        .unwrap_or(0);
        cursor += varint::encode(
            crate::http3_codec::settings::SETTINGS_MAX_FIELD_SECTION_SIZE,
            &mut payload[cursor..],
        )
        .unwrap_or(0);
        cursor += varint::encode(
            self.local_settings.max_field_section_size,
            &mut payload[cursor..],
        )
        .unwrap_or(0);
        // RFC 9297 §3 — both endpoints MUST send
        // SETTINGS_H3_DATAGRAM=1 before DATAGRAM use.
        if self.local_settings.h3_datagram {
            cursor += varint::encode(
                crate::http3_codec::settings::SETTINGS_H3_DATAGRAM,
                &mut payload[cursor..],
            )
            .unwrap_or(0);
            cursor += varint::encode(1, &mut payload[cursor..]).unwrap_or(0);
        }
        // RFC 9220 §3 — ENABLE_CONNECT_PROTOCOL=1 for extended CONNECT.
        if self.local_settings.enable_connect_protocol {
            cursor += varint::encode(
                crate::http3_codec::settings::SETTINGS_ENABLE_CONNECT_PROTOCOL,
                &mut payload[cursor..],
            )
            .unwrap_or(0);
            cursor += varint::encode(1, &mut payload[cursor..]).unwrap_or(0);
        }
        let frame = H3Frame::Settings {
            payload: &payload[..cursor],
        };
        let _ = encode_h3_frame(&frame, &mut self.control_outbound);
        self.settings_emitted = true;
    }
}

fn apply_response_frame(
    stream_id: StreamId,
    recv: &mut RecvState,
    frame_obj: H3Frame<'_>,
    events: &mut VecDeque<H3ClientEvent>,
    max_field_section_size: u64,
) -> Result<(), ClientError> {
    match (core::mem::take(recv), frame_obj) {
        (RecvState::Idle, H3Frame::Headers { header_block }) => {
            let status = decode_status(header_block, max_field_section_size)?;
            // the client delivers headers via the event and never re-reads the
            // state copy (only the server accumulates via recv.headers()), so
            // carry an empty vec here — the decoded field section rides the
            // event as `header_block` instead.
            *recv = RecvState::HeadersReceived {
                headers: Vec::new(),
            };
            events.push_back(H3ClientEvent::ResponseHeaders {
                stream_id,
                status,
                header_block: header_block.to_vec(),
            });
        }
        (RecvState::HeadersReceived { headers }, H3Frame::Data { payload }) => {
            // client delivers the chunk via the event; it does NOT accumulate
            // body_so_far (only the server reads that), so the chunk is copied
            // once into the event, not a second time into an unread buffer.
            *recv = RecvState::BodyReceiving {
                headers,
                body_so_far: Vec::new(),
            };
            events.push_back(H3ClientEvent::ResponseData {
                stream_id,
                bytes: payload.to_vec(),
            });
        }
        (
            RecvState::BodyReceiving {
                headers,
                body_so_far,
            },
            H3Frame::Data { payload },
        ) => {
            *recv = RecvState::BodyReceiving {
                headers,
                body_so_far,
            };
            events.push_back(H3ClientEvent::ResponseData {
                stream_id,
                bytes: payload.to_vec(),
            });
        }
        (RecvState::HeadersReceived { headers }, H3Frame::Headers { header_block }) => {
            validate_trailers(header_block, max_field_section_size)?;
            *recv = RecvState::TrailersReceived {
                headers,
                body: Vec::new(),
                trailers: Vec::new(),
            };
            events.push_back(H3ClientEvent::ResponseTrailers { stream_id });
        }
        (
            RecvState::BodyReceiving {
                headers,
                body_so_far,
            },
            H3Frame::Headers { header_block },
        ) => {
            validate_trailers(header_block, max_field_section_size)?;
            *recv = RecvState::TrailersReceived {
                headers,
                body: body_so_far,
                trailers: Vec::new(),
            };
            events.push_back(H3ClientEvent::ResponseTrailers { stream_id });
        }
        (prior, H3Frame::Reserved { frame_type, .. }) => {
            // RFC 9114 §11.2.1 — HTTP/2-reserved types are a
            // connection error; §7.2.8 — other reserved types
            // (GREASE / future-assigned) MUST be ignored.
            if frame::is_http2_reserved(frame_type) {
                return Err(ClientError::Request(RequestError::UnexpectedFrame));
            }
            *recv = prior;
            return Ok(());
        }
        (_, _) => {
            return Err(ClientError::Request(RequestError::UnexpectedFrame));
        }
    }
    Ok(())
}

/// Drive [`qpack::decoder::decode_into`] (the 0-alloc borrowing engine) over
/// a response HEADERS field section, extracting `:status` (RFC 9114
/// §4.3.2) as a `u16`. `:status` is always static-table-indexed or a raw
/// (non-Huffman) literal in practice, but the sink still runs over every
/// field so the SAME cap enforcement / malformed-input rejection
/// `decode_bounded` performed happens here too — only the "own every field"
/// step is skipped. `None` if `:status` is absent or not a valid `u16`.
fn decode_status(header_block: &[u8], cap: u64) -> Result<Option<u16>, ClientError> {
    let mut scratch = [0u8; crate::sized::PROXIMA_PROTOCOLS_HTTP3_CODEC_QPACK_DECODE_BOUNDED_SCRATCH_LEN];
    let mut status = None;
    let mut sink = |name: &[u8], value: &[u8]| -> Result<(), qpack::decoder::DecodeError> {
        if name == b":status" {
            status = core::str::from_utf8(value)
                .ok()
                .and_then(|text| text.parse().ok());
        }
        Ok(())
    };
    qpack::decoder::decode_into(header_block, cap, &mut scratch, &mut sink)?;
    Ok(status)
}

/// Validate a trailing HEADERS field section under `cap` without owning any
/// field. No consumer reads response trailers today (see
/// [`H3ClientEvent::ResponseTrailers`]); this exists purely to preserve the
/// same cap-enforcement / malformed-input rejection `decode_bounded`
/// provided before this redesign.
fn validate_trailers(header_block: &[u8], cap: u64) -> Result<(), ClientError> {
    let mut scratch = [0u8; crate::sized::PROXIMA_PROTOCOLS_HTTP3_CODEC_QPACK_DECODE_BOUNDED_SCRATCH_LEN];
    let mut sink = |_: &[u8], _: &[u8]| -> Result<(), qpack::decoder::DecodeError> { Ok(()) };
    qpack::decoder::decode_into(header_block, cap, &mut scratch, &mut sink)?;
    Ok(())
}

/// Byte-equality of a cached header set against an incoming one — the guard
/// for reusing [`ClientConnection::cached_request_frame`]. Cheaper than a QPACK
/// re-encode; a value compare (not a hash) so there is no collision risk.
fn headers_match(cached: &[(Vec<u8>, Vec<u8>)], incoming: &[(&[u8], &[u8])]) -> bool {
    cached.len() == incoming.len()
        && cached
            .iter()
            .zip(incoming)
            .all(|((cached_name, cached_value), (name, value))| {
                cached_name == name && cached_value == value
            })
}

fn encode_h3_frame(frame_obj: &H3Frame<'_>, out: &mut Vec<u8>) -> Result<(), ClientError> {
    let initial_len = out.len();
    out.resize(initial_len + 4096, 0);
    match frame::encode(frame_obj, &mut out[initial_len..]) {
        Ok(written) => {
            out.truncate(initial_len + written);
            Ok(())
        }
        Err(frame::FrameError::BufferTooSmall { needed }) => {
            out.truncate(initial_len);
            out.resize(initial_len + needed, 0);
            let written = frame::encode(frame_obj, &mut out[initial_len..])
                .map_err(|err| ClientError::Request(RequestError::Frame(err)))?;
            out.truncate(initial_len + written);
            Ok(())
        }
        Err(other) => Err(ClientError::Request(RequestError::Frame(other))),
    }
}

fn state_label(state: &ClientState) -> &'static str {
    match state {
        ClientState::Negotiating => "Negotiating",
        ClientState::Established => "Established",
        ClientState::Closing { .. } => "Closing",
    }
}

fn send_label(state: &SendState) -> &'static str {
    match state {
        SendState::Idle => "Idle",
        SendState::HeadersSent => "HeadersSent",
        SendState::BodyStreaming { .. } => "BodyStreaming",
        SendState::TrailersSent => "TrailersSent",
        SendState::Done => "Done",
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn encode_settings(pairs: &[(u64, u64)]) -> Vec<u8> {
        let mut payload = [0u8; 32];
        let mut cursor = 0;
        use crate::quic::varint;
        for (id, value) in pairs {
            cursor += varint::encode(*id, &mut payload[cursor..]).unwrap();
            cursor += varint::encode(*value, &mut payload[cursor..]).unwrap();
        }
        let mut out = alloc::vec![0u8; 64];
        let written = frame::encode(
            &H3Frame::Settings {
                payload: &payload[..cursor],
            },
            &mut out,
        )
        .unwrap();
        out.truncate(written);
        out
    }

    fn encode_headers(pairs: &[(&[u8], &[u8])]) -> Vec<u8> {
        let mut header_block = Vec::new();
        qpack::encoder::encode_refs(pairs.iter().copied(), &mut header_block).unwrap();
        let mut out = alloc::vec![0u8; header_block.len() + 16];
        let written = frame::encode(
            &H3Frame::Headers {
                header_block: &header_block,
            },
            &mut out,
        )
        .unwrap();
        out.truncate(written);
        out
    }

    #[test]
    fn open_request_assigns_client_bidi_stream_ids() {
        let mut client = ClientConnection::new(Settings::default());
        client.feed_control(&encode_settings(&[])).unwrap();
        let _ = client.poll_event();
        let s0 = client
            .open_request(&[(b":method", b"GET"), (b":path", b"/")])
            .unwrap();
        let s1 = client
            .open_request(&[(b":method", b"GET"), (b":path", b"/")])
            .unwrap();
        assert_eq!(s0, StreamId(0));
        assert_eq!(s1, StreamId(4));
    }

    #[test]
    fn full_request_response_cycle() {
        let mut client = ClientConnection::new(Settings::default());
        // Server's SETTINGS comes in over the control stream.
        client
            .feed_control(&encode_settings(&[(0x06, 65536)]))
            .unwrap();
        let _ = client.poll_event();
        // Client opens a request, sends body, finishes.
        let stream = client
            .open_request(&[
                (b":method", b"POST"),
                (b":scheme", b"https"),
                (b":authority", b"example.com"),
                (b":path", b"/api"),
            ])
            .unwrap();
        client.send_request_data(stream, b"payload").unwrap();
        client.finish_request(stream).unwrap();
        let (outbound, fin) = client.take_request_outbound(stream).unwrap();
        assert!(fin);
        let (first, c1) = frame::parse(&outbound).unwrap();
        assert!(matches!(first, H3Frame::Headers { .. }));
        let (second, _) = frame::parse(&outbound[c1..]).unwrap();
        let H3Frame::Data { payload } = second else {
            panic!("expected data");
        };
        assert_eq!(payload, b"payload");

        // Server responds with HEADERS + DATA.
        let mut server_response = encode_headers(&[(b":status", b"200")]);
        let data_frame = {
            let mut out = alloc::vec![0u8; 32];
            let written = frame::encode(&H3Frame::Data { payload: b"hello" }, &mut out).unwrap();
            out.truncate(written);
            out
        };
        server_response.extend_from_slice(&data_frame);
        client
            .feed_response(stream, &server_response, true)
            .unwrap();
        let H3ClientEvent::ResponseHeaders {
            status,
            header_block,
            ..
        } = client.poll_event().unwrap()
        else {
            panic!("expected headers");
        };
        assert_eq!(status, Some(200));
        // header_block still carries the full field section — a caller
        // that needs more than :status decodes it directly.
        let fields = qpack::decoder::decode_bounded(&header_block, u64::MAX).unwrap();
        assert_eq!(fields[0].name, b":status".to_vec());
        assert_eq!(fields[0].value, b"200".to_vec());
        let H3ClientEvent::ResponseData { bytes, .. } = client.poll_event().unwrap() else {
            panic!("expected data");
        };
        assert_eq!(bytes, b"hello");
        assert!(matches!(
            client.poll_event().unwrap(),
            H3ClientEvent::ResponseFinished { .. }
        ));
    }

    #[test]
    fn open_request_before_settings_rejected() {
        let mut client = ClientConnection::new(Settings::default());
        let err = client.open_request(&[(b":method", b"GET")]).unwrap_err();
        assert!(matches!(err, ClientError::IllegalInState { .. }));
    }

    /// C36-R alloc-count claim (`DC-H3-FACADE-EVENTS-OWN`, client half): a
    /// captured nginx-shaped `200` response (`:status` + 4 regular
    /// headers — the same shape as the QPACK decoder's own C34 fixture)
    /// drives `apply_response_frame`'s `:status` extraction through
    /// `decode_status` at 0 heap allocations, and the whole call (which
    /// still copies the field section once into `header_block` to cross
    /// the event-queue boundary) at exactly 1 — down from the pre-redesign
    /// `decode_bounded` path's `1 + 2 * field_count` (11 for this 5-field
    /// fixture).
    #[cfg(feature = "std")]
    #[test]
    fn alloc_count_apply_response_frame_headers_is_one_not_one_plus_two_per_field() {
        let wire = encode_headers(&[
            (b":status", b"200"),
            (b"server", b"nginx/1.27.0"),
            (b"date", b"Tue, 30 Jun 2026 00:00:00 GMT"),
            (b"content-type", b"text/html"),
            (b"content-length", b"612"),
        ]);
        let (frame_obj, _) = frame::parse(&wire).unwrap();

        // warm-up call: absorbs the VecDeque's one-time first-push
        // allocation so the measured call below isolates
        // apply_response_frame's OWN marginal cost.
        let mut warm_recv = RecvState::Idle;
        let mut warm_events: VecDeque<H3ClientEvent> = VecDeque::new();
        apply_response_frame(
            StreamId(0),
            &mut warm_recv,
            frame_obj,
            &mut warm_events,
            u64::MAX,
        )
        .expect("warm-up decode");

        let mut recv = RecvState::Idle;
        let mut events: VecDeque<H3ClientEvent> = warm_events;
        events.clear();

        let region = crate::alloc_test::exclusive_region();
        let before = region.change();
        apply_response_frame(StreamId(4), &mut recv, frame_obj, &mut events, u64::MAX)
            .expect("measured decode");
        let after = region.change();
        assert_eq!(
            after.allocations - before.allocations,
            1,
            "apply_response_frame must perform exactly 1 heap allocation \
             (the header_block.to_vec() copy) for a 5-field response — \
             the pre-redesign decode_bounded path performed 1 + 2*5 = 11"
        );

        let H3ClientEvent::ResponseHeaders {
            status,
            header_block,
            ..
        } = events.pop_front().expect("headers event queued")
        else {
            panic!("expected ResponseHeaders");
        };
        assert_eq!(status, Some(200));
        let fields = qpack::decoder::decode_bounded(&header_block, u64::MAX)
            .expect("header_block still decodes to the full field set");
        assert_eq!(fields.len(), 5);
        assert_eq!(fields[0].name, b":status".to_vec());
        assert_eq!(fields[0].value, b"200".to_vec());
    }

    /// The isolated `:status` extraction step — [`decode_status`] itself —
    /// performs 0 heap allocations regardless of field count, since it
    /// drives the borrowing `decode_into` engine directly.
    #[cfg(feature = "std")]
    #[test]
    fn alloc_count_decode_status_is_zero() {
        let wire = encode_headers(&[
            (b":status", b"200"),
            (b"server", b"nginx/1.27.0"),
            (b"date", b"Tue, 30 Jun 2026 00:00:00 GMT"),
            (b"content-type", b"text/html"),
            (b"content-length", b"612"),
        ]);
        let (frame_obj, _) = frame::parse(&wire).unwrap();
        let H3Frame::Headers { header_block } = frame_obj else {
            panic!("expected headers frame");
        };

        let region = crate::alloc_test::exclusive_region();
        let before = region.change();
        let status = decode_status(header_block, u64::MAX).expect("decode_status");
        let after = region.change();
        assert_eq!(status, Some(200));
        assert_eq!(
            after.allocations - before.allocations,
            0,
            "decode_status must perform 0 heap allocations"
        );
    }

    /// `docs/proxima-pipe/part-source-sink-design.md` step 3, h3 client
    /// response path: a connection opted into
    /// [`ResponseHeaderMode::Source`] delivers response HEADERS as a
    /// `PartSource` on [`ClientConnection::poll_response_header_source`],
    /// NOT the owned `ResponseHeaders` event — the two are mutually
    /// exclusive per connection (see [`ResponseHeaderMode`] docs). Stepping
    /// the source reads `:status` correctly, same as the owned path.
    #[cfg(feature = "http3_codec-part-source")]
    #[test]
    fn header_source_mode_emits_part_source_not_owned_event() {
        use proxima_primitives::pipe::part::{Part, PartSource as _};

        let wire = encode_headers(&[(b":status", b"200"), (b"content-type", b"text/plain")]);

        let mut client = ClientConnection::new(Settings::default());
        client.enable_header_source_mode();
        client
            .feed_response(StreamId(0), &wire, false)
            .expect("feed response headers");

        assert!(
            client.poll_event().is_none(),
            "Source mode must not also emit the owned ResponseHeaders event"
        );

        let (stream_id, mut source) = client
            .poll_response_header_source()
            .expect("headers queued on the source path");
        assert_eq!(stream_id, StreamId(0));

        let mut status = None;
        let mut saw_end = false;
        while let Some(part) = source.next() {
            match part {
                Part::Header(name, value) if name == b":status" => {
                    status = core::str::from_utf8(value)
                        .ok()
                        .and_then(|text| text.parse::<u16>().ok());
                }
                Part::End => saw_end = true,
                _ => {}
            }
        }
        assert_eq!(status, Some(200));
        assert!(saw_end, "source must yield exactly one Part::End");
        assert_eq!(
            source.error(),
            None,
            "a well-formed section must decode cleanly through the lazy source"
        );
    }

    /// DC-H3-FACADE-EVENTS-OWN, client half, step 3: `ClientConnection`
    /// itself (not just the `qpack::part_source` adapter in isolation)
    /// performs 0 heap allocations decoding response HEADERS when
    /// `ResponseHeaderMode::Source` is active — versus > 0 in the
    /// pre-existing `Owned` default (the `header_block.to_vec()` copy
    /// crossing the event-queue boundary, see
    /// `alloc_count_apply_response_frame_headers_is_one_not_one_plus_two_per_field`
    /// above). Each mode warms up its own connection's VecDeque on stream 0
    /// (a fresh `VecDeque`'s first push always allocates) before measuring
    /// stream 4's decode in isolation — same discipline as the pre-existing
    /// warm-up in this module.
    #[cfg(all(feature = "http3_codec-part-source", feature = "std"))]
    #[test]
    fn alloc_count_feed_response_source_mode_is_zero_owned_mode_is_greater_than_zero() {
        let wire = encode_headers(&[
            (b":status", b"200"),
            (b"server", b"nginx/1.27.0"),
            (b"date", b"Tue, 30 Jun 2026 00:00:00 GMT"),
            (b"content-type", b"text/html"),
            (b"content-length", b"612"),
        ]);

        let mut source_client = ClientConnection::new(Settings::default());
        source_client.enable_header_source_mode();
        source_client
            .feed_response(StreamId(0), &wire, false)
            .expect("warm-up feed");
        // two polls: the first hands out stream 0's block, the second
        // recycles it into the pool (recycling happens at the START of the
        // next poll) so the measured feed below reuses it — the
        // steady-state shape.
        let _ = source_client.poll_response_header_source();
        let _ = source_client.poll_response_header_source();

        let region = crate::alloc_test::exclusive_region();
        let before_source = region.change();
        source_client
            .feed_response(StreamId(4), &wire, false)
            .expect("measured source-mode feed");
        let after_source = region.change();
        assert_eq!(
            after_source.allocations - before_source.allocations,
            0,
            "Source-mode feed_response must perform 0 heap allocations for a 5-field response"
        );
        let before_poll = region.change();
        {
            let (_, mut source) = source_client
                .poll_response_header_source()
                .expect("headers queued on the source path");
            assert!(matches!(
                proxima_primitives::pipe::part::PartSource::next(&mut source),
                Some(proxima_primitives::pipe::part::Part::Header(name, value)) if name == b":status" && value == b"200"
            ));
            while proxima_primitives::pipe::part::PartSource::next(&mut source).is_some() {}
            assert_eq!(source.error(), None);
        }
        let after_poll = region.change();
        assert_eq!(
            after_poll.allocations - before_poll.allocations,
            0,
            "polling + stepping the lazy source must also perform 0 heap allocations"
        );

        let mut owned_client = ClientConnection::new(Settings::default());
        owned_client
            .feed_response(StreamId(0), &wire, false)
            .expect("warm-up decode");
        let _ = owned_client.poll_event();

        let before_owned = region.change();
        owned_client
            .feed_response(StreamId(4), &wire, false)
            .expect("measured owned-mode decode");
        let after_owned = region.change();
        assert!(
            after_owned.allocations - before_owned.allocations > 0,
            "Owned-mode feed_response still allocates header_block.to_vec() — \
             the cost Source mode opts out of"
        );
    }
}
