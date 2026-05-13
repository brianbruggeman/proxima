//! Server-side HTTP/3 connection state machine per [RFC 9114].
//!
//! Sans-IO. Caller hands inbound stream bytes in (`feed_request`,
//! `feed_control`), drains outbound bytes (`take_outbound`) for the
//! QUIC layer to send.
//!
//! Connection-level state is a discriminated enum per principle 11:
//! [`ServerState`]. Per-request state lives in a stream-id-keyed
//! `heapless::FnvIndexMap<u64, RequestEntry, MAX>` (boxed so the
//! inline map doesn't fatten `ServerConnection`) and uses [`RecvState`]
//! / [`SendState`] enums per request. Cap is
//! [`crate::sized::PROXIMA_PROTOCOLS_HTTP3_CODEC_SERVER_MAX_CONCURRENT_REQUESTS`];
//! a request stream that arrives after the map is full is rejected
//! with [`ServerError::RequestMapFull`].
//!
//! [RFC 9114]: https://www.rfc-editor.org/rfc/rfc9114

use alloc::boxed::Box;
use alloc::vec::Vec;

/// Map cap — pulled from `sized.rs` so the build-time TOML drives the
/// type. heapless requires a power of two.
const REQUESTS_CAP: usize = crate::sized::PROXIMA_PROTOCOLS_HTTP3_CODEC_SERVER_MAX_CONCURRENT_REQUESTS;
type RequestMap = heapless::index_map::FnvIndexMap<u64, RequestEntry, REQUESTS_CAP>;

use crate::http3_codec::frame::{self, H3Frame};
use crate::http3_codec::qpack;
use crate::http3_codec::request::{RecvState, RequestError, SendState};
use crate::http3_codec::settings::{Settings, SettingsError};

/// QUIC stream ID newtype. Mirrors `crate::quic::streams::StreamId`
/// but lives here to keep proto-h3 independently citable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StreamId(pub u64);

/// Top-level server-connection state.
///
/// Discriminated enum per principle 11. SETTINGS exchange transitions
/// [`ServerState::Negotiating`] → [`ServerState::Established`]; a sent
/// or received GOAWAY adds a [`ServerState::Closing`] marker (we
/// continue serving in-flight requests until they drain).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ServerState {
    Negotiating,
    Established,
    Closing { peer_max_stream_id: u64 },
}

/// Request-HEADERS delivery mode (`part-source` feature only) — the
/// server half of the client's
/// [`crate::http3_codec::client::ResponseHeaderMode`], same contract: `Owned` (the
/// default) preserves the pre-existing owned
/// [`H3ServerEvent::RequestHeaders`] event byte-for-byte; `Source`
/// queues the still-encoded section raw (pool-recycled, steady-state 0
/// allocations) for lazy stepping via
/// [`ServerConnection::poll_request_header_source`]. Deferred-validation
/// contract as on the client: decode failures surface when the CALLER
/// steps the source and are connection-fatal (RFC 9204 §2.2.3).
#[cfg(feature = "http3_codec-part-source")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RequestHeaderMode {
    #[default]
    Owned,
    Source,
}

/// Connection-level event emitted by [`ServerConnection::poll_event`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum H3ServerEvent {
    /// SETTINGS exchange complete; the connection is fully established.
    SettingsEstablished { peer: Settings },
    /// A new request received its full header block.
    RequestHeaders {
        stream_id: StreamId,
        headers: Vec<qpack::decoder::DecodedField>,
    },
    /// Body bytes arrived on an established request.
    RequestData { stream_id: StreamId, bytes: Vec<u8> },
    /// Optional trailing HEADERS frame received.
    RequestTrailers {
        stream_id: StreamId,
        trailers: Vec<qpack::decoder::DecodedField>,
    },
    /// FIN bit observed on the request stream — peer is done sending.
    RequestFinished { stream_id: StreamId },
    /// Peer sent a GOAWAY on the control stream — graceful shutdown.
    GoAway { peer_max_stream_id: u64 },
}

/// Errors from server-connection operations.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ServerError {
    /// SETTINGS exchange complete but the frame violated §7.2.4.1.
    Settings(SettingsError),
    /// Request stream frame failed RFC 9114 §4.1.
    Request(RequestError),
    /// QPACK decode failed.
    Qpack(qpack::decoder::DecodeError),
    /// Method call invoked outside the FSM's legal state.
    IllegalInState {
        state: &'static str,
        method: &'static str,
    },
    /// A new request stream arrived but the per-connection request
    /// table is at its build-time cap
    /// ([`crate::sized::PROXIMA_PROTOCOLS_HTTP3_CODEC_SERVER_MAX_CONCURRENT_REQUESTS`]).
    /// The peer is violating the SETTINGS we advertised; per RFC 9114
    /// §5.2 the caller should respond with a connection error
    /// (H3_EXCESSIVE_LOAD).
    RequestMapFull { cap: usize },
}

impl From<SettingsError> for ServerError {
    fn from(err: SettingsError) -> Self {
        Self::Settings(err)
    }
}
impl From<RequestError> for ServerError {
    fn from(err: RequestError) -> Self {
        Self::Request(err)
    }
}
impl From<qpack::decoder::DecodeError> for ServerError {
    fn from(err: qpack::decoder::DecodeError) -> Self {
        Self::Qpack(err)
    }
}

/// Per-request state slot.
struct RequestEntry {
    recv: RecvState,
    send: SendState,
    /// Outbound bytes queued for this stream (response HEADERS / DATA /
    /// trailers). The QUIC layer drains via
    /// [`ServerConnection::take_request_outbound`].
    outbound: Vec<u8>,
    /// FIN should be set on the QUIC layer's next stream write.
    fin_pending: bool,
}

/// HTTP/3 server connection.
pub struct ServerConnection {
    state: ServerState,
    local_settings: Settings,
    peer_settings: Option<Settings>,
    /// boxed because the inline `FnvIndexMap<u64, RequestEntry, 1024>`
    /// would push `ServerConnection` past test-thread stack capacity.
    /// one alloc per connection, then in-place ops; entries live inline
    /// inside the box (no per-entry heap alloc, unlike the prior
    /// `BTreeMap` shape).
    requests: Box<RequestMap>,
    pending_events: alloc::collections::VecDeque<H3ServerEvent>,
    /// Bytes queued for the control stream (SETTINGS + later GOAWAY /
    /// MAX_PUSH_ID). The QUIC layer reads via
    /// [`ServerConnection::take_control_outbound`].
    control_outbound: Vec<u8>,
    /// Have we already emitted our SETTINGS frame onto the control
    /// stream's outbound buffer?
    settings_emitted: bool,
    /// RFC 9114 §7.2.4 — track whether the peer's SETTINGS frame has
    /// already been received on the control stream. A second SETTINGS
    /// frame is a connection error; any non-SETTINGS frame arriving
    /// before the first SETTINGS is H3_MISSING_SETTINGS.
    peer_settings_seen: bool,
    /// See [`RequestHeaderMode`]. Default `Owned` — fully opt-in.
    #[cfg(feature = "http3_codec-part-source")]
    request_header_mode: RequestHeaderMode,
    /// Raw request-HEADERS field sections queued while
    /// `request_header_mode` is `Source`, drained via
    /// [`Self::poll_request_header_source`]. Same shared substrate as
    /// the client half — see
    /// [`qpack::part_source::HeaderBlockQueue`].
    #[cfg(feature = "http3_codec-part-source")]
    header_source_queue: qpack::part_source::HeaderBlockQueue<StreamId>,
}

impl ServerConnection {
    /// Construct with the locally-advertised SETTINGS.
    #[must_use]
    pub fn new(local_settings: Settings) -> Self {
        Self {
            state: ServerState::Negotiating,
            local_settings,
            peer_settings: None,
            requests: Box::new(RequestMap::new()),
            pending_events: alloc::collections::VecDeque::new(),
            control_outbound: Vec::new(),
            settings_emitted: false,
            peer_settings_seen: false,
            #[cfg(feature = "http3_codec-part-source")]
            request_header_mode: RequestHeaderMode::default(),
            #[cfg(feature = "http3_codec-part-source")]
            header_source_queue: qpack::part_source::HeaderBlockQueue::new(),
        }
    }

    /// Opt this connection into [`RequestHeaderMode::Source`] — every
    /// subsequent request's first HEADERS frame is queued for
    /// [`Self::poll_request_header_source`] instead of the owned
    /// [`H3ServerEvent::RequestHeaders`] event. Call once, before the
    /// first request arrives. Sizes the Huffman scratch once (the mode's
    /// only setup allocation).
    #[cfg(feature = "http3_codec-part-source")]
    pub fn enable_header_source_mode(&mut self) {
        self.request_header_mode = RequestHeaderMode::Source;
        self.header_source_queue
            .size_scratch(crate::sized::PROXIMA_PROTOCOLS_HTTP3_CODEC_QPACK_DECODE_BOUNDED_SCRATCH_LEN);
    }

    /// Drain one queued request-HEADERS field section as a borrowed
    /// [`qpack::part_source::FieldSectionSource`] — only populated when
    /// [`RequestHeaderMode::Source`] is active. Step it, then CHECK
    /// [`qpack::part_source::FieldSectionSource::error`] — a reported
    /// error is a QPACK decompression failure (connection-fatal). The
    /// source borrows this connection; drop it before the next poll.
    #[cfg(feature = "http3_codec-part-source")]
    #[must_use]
    pub fn poll_request_header_source(
        &mut self,
    ) -> Option<(StreamId, qpack::part_source::FieldSectionSource<'_>)> {
        let cap = self.local_settings.max_field_section_size;
        self.header_source_queue.poll(cap)
    }

    /// Current connection state.
    #[must_use]
    pub fn state(&self) -> &ServerState {
        &self.state
    }

    /// Locally-advertised SETTINGS.
    #[must_use]
    pub fn local_settings(&self) -> &Settings {
        &self.local_settings
    }

    /// Peer-advertised SETTINGS (None until SETTINGS exchange completes).
    #[must_use]
    pub fn peer_settings(&self) -> Option<&Settings> {
        self.peer_settings.as_ref()
    }

    /// Feed inbound bytes received on the peer's control stream.
    ///
    /// # Errors
    ///
    /// See [`ServerError`].
    pub fn feed_control(&mut self, bytes: &[u8]) -> Result<(), ServerError> {
        let mut cursor = 0;
        while cursor < bytes.len() {
            let (frame_obj, consumed) = frame::parse(&bytes[cursor..])
                .map_err(|err| ServerError::Request(RequestError::Frame(err)))?;
            cursor += consumed;
            match frame_obj {
                H3Frame::Settings { payload } => {
                    // RFC 9114 §7.2.4 — exactly one SETTINGS frame per
                    // control stream. Duplicate is a connection error.
                    if self.peer_settings_seen {
                        return Err(ServerError::Settings(SettingsError::DuplicateFrame));
                    }
                    let mut peer = Settings::default();
                    peer.apply_payload(payload)?;
                    self.peer_settings = Some(peer);
                    self.peer_settings_seen = true;
                    if matches!(self.state, ServerState::Negotiating) {
                        self.state = ServerState::Established;
                    }
                    self.pending_events
                        .push_back(H3ServerEvent::SettingsEstablished { peer });
                }
                // RFC 9114 §6.2.1 + §7.2.4 — SETTINGS MUST be the first
                // frame on the control stream. Any other frame arriving
                // before peer_settings_seen is H3_MISSING_SETTINGS.
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
                    return Err(ServerError::Settings(SettingsError::MissingSettings {
                        observed_id,
                    }));
                }
                H3Frame::GoAway { id } => {
                    self.state = ServerState::Closing {
                        peer_max_stream_id: id,
                    };
                    self.pending_events.push_back(H3ServerEvent::GoAway {
                        peer_max_stream_id: id,
                    });
                }
                H3Frame::CancelPush { .. } | H3Frame::MaxPushId { .. } => {
                    // Allowed on control stream but our v1 server doesn't
                    // emit pushes — drop CANCEL_PUSH / MAX_PUSH_ID.
                }
                H3Frame::Reserved { frame_type, .. } => {
                    // RFC 9114 §11.2.1 — the four HTTP/2-reserved
                    // types (0x02 / 0x06 / 0x08 / 0x09) MUST be
                    // treated as a connection error, on ANY stream
                    // class. §7.2.8 GREASE / other unknown reserved
                    // types are ignored.
                    if frame::is_http2_reserved(frame_type) {
                        return Err(ServerError::Request(RequestError::UnexpectedFrame));
                    }
                    // Ignore (RFC 9114 §7.2.8).
                }
                H3Frame::Data { .. } | H3Frame::Headers { .. } | H3Frame::PushPromise { .. } => {
                    return Err(ServerError::Request(RequestError::UnexpectedFrame));
                }
            }
        }
        // Emit our own SETTINGS on first call so the peer sees it.
        self.ensure_settings_emitted();
        Ok(())
    }

    /// Feed inbound bytes received on a peer-initiated request stream.
    /// `fin` is `true` when this byte slice ends with the QUIC FIN bit.
    ///
    /// # Errors
    ///
    /// See [`ServerError`].
    pub fn feed_request(
        &mut self,
        stream_id: StreamId,
        bytes: &[u8],
        fin: bool,
    ) -> Result<(), ServerError> {
        if !self.requests.contains_key(&stream_id.0) {
            let fresh = RequestEntry {
                recv: RecvState::Idle,
                send: SendState::Idle,
                outbound: Vec::new(),
                fin_pending: false,
            };
            self.requests
                .insert(stream_id.0, fresh)
                .map_err(|_| ServerError::RequestMapFull { cap: REQUESTS_CAP })?;
        }
        // Snapshot the cap BEFORE the mutable borrow so we can pass it
        // into the apply helper alongside the &mut to self.pending_events.
        let max_field_section_size = self.local_settings.max_field_section_size;
        #[cfg(feature = "http3_codec-part-source")]
        let request_header_mode = self.request_header_mode;
        // Just inserted or already present; get_mut cannot return None.
        let entry = self
            .requests
            .get_mut(&stream_id.0)
            .ok_or(ServerError::RequestMapFull { cap: REQUESTS_CAP })?;
        let mut cursor = 0;
        while cursor < bytes.len() {
            let (frame_obj, consumed) = frame::parse(&bytes[cursor..])
                .map_err(|err| ServerError::Request(RequestError::Frame(err)))?;
            cursor += consumed;
            // Opt-in reroute, server half of the client's Source mode (the
            // default `Owned` never takes this branch — the apply helper
            // below is byte-for-byte unchanged): the first HEADERS on a
            // request stream is queued raw for lazy stepping.
            #[cfg(feature = "http3_codec-part-source")]
            if request_header_mode == RequestHeaderMode::Source
                && matches!(entry.recv, RecvState::Idle)
                && let H3Frame::Headers { header_block } = frame_obj
            {
                // the state copy is not re-read on this path (the facade
                // consumer builds its request from the stepped source, not
                // from recv.headers()) — carry an empty vec.
                entry.recv = RecvState::HeadersReceived {
                    headers: Vec::new(),
                };
                self.header_source_queue.push(stream_id, header_block);
                continue;
            }
            apply_inbound_request_frame(
                stream_id,
                &mut entry.recv,
                frame_obj,
                &mut self.pending_events,
                max_field_section_size,
            )?;
        }
        if fin {
            // Promote to Done if the FSM has already reached a terminal
            // recv-side; otherwise transition into Done anyway (peer
            // closed early).
            entry.recv = RecvState::Done;
            self.pending_events
                .push_back(H3ServerEvent::RequestFinished { stream_id });
        }
        Ok(())
    }

    /// Drain the next pending H3 event, if any.
    #[must_use]
    pub fn poll_event(&mut self) -> Option<H3ServerEvent> {
        self.pending_events.pop_front()
    }

    /// Drain outbound bytes queued for the control stream. The caller
    /// MUST forward these bytes to the peer over its unidirectional
    /// control stream.
    ///
    /// RFC 9114 §6.2.1 — both endpoints MUST initiate their control
    /// stream + ship SETTINGS as the first frame. We stage SETTINGS
    /// proactively on first drain so callers don't need to wait for
    /// peer input to flush their own.
    pub fn take_control_outbound(&mut self) -> Vec<u8> {
        self.ensure_settings_emitted();
        core::mem::take(&mut self.control_outbound)
    }

    /// Drain outbound bytes queued for a per-request stream. Returns
    /// the bytes + whether the FIN bit should be set on this write.
    /// Returns `None` if the stream isn't known.
    pub fn take_request_outbound(&mut self, stream_id: StreamId) -> Option<(Vec<u8>, bool)> {
        let entry = self.requests.get_mut(&stream_id.0)?;
        let bytes = core::mem::take(&mut entry.outbound);
        let fin = core::mem::replace(&mut entry.fin_pending, false);
        Some((bytes, fin))
    }

    /// Request streams that still have response bytes or a FIN queued.
    /// The driver iterates these instead of every known stream, keeping
    /// the send pass O(active-with-output) rather than O(all-streams) —
    /// without it, throughput stays flat as concurrency rises instead
    /// of climbing.
    #[must_use]
    pub fn streams_with_outbound(&self) -> alloc::vec::Vec<u64> {
        self.requests
            .iter()
            .filter(|(_, entry)| !entry.outbound.is_empty() || entry.fin_pending)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Remove all fully-completed request entries from the map so
    /// the fixed-cap slot is freed for future requests.
    pub fn gc_completed(&mut self) {
        // heapless IndexMap doesn't support retain; scan + collect
        // into a stack-cap'd heapless Vec (bounded by the table cap
        // so no alloc). In practice the number of simultaneously-
        // completing streams per driver pass is small (~1–2).
        let mut done: heapless::Vec<u64, REQUESTS_CAP> = heapless::Vec::new();
        for key in self.requests.keys().copied() {
            if self.requests.get(&key).is_some_and(|entry| {
                matches!(entry.recv, RecvState::Done)
                    && matches!(entry.send, SendState::Done)
                    && entry.outbound.is_empty()
                    && !entry.fin_pending
            }) {
                let _ = done.push(key);
            }
        }
        for key in done {
            self.requests.swap_remove(&key);
        }
    }

    /// Queue the response HEADERS frame for `stream_id`.
    ///
    /// # Errors
    ///
    /// Returns [`ServerError::IllegalInState`] when called before
    /// SETTINGS exchange completes or before the request's HEADERS
    /// have been received.
    pub fn send_response_headers(
        &mut self,
        stream_id: StreamId,
        headers: &[(&[u8], &[u8])],
    ) -> Result<(), ServerError> {
        if !matches!(
            self.state,
            ServerState::Established | ServerState::Closing { .. }
        ) {
            return Err(ServerError::IllegalInState {
                state: state_label(&self.state),
                method: "send_response_headers",
            });
        }
        let entry = self
            .requests
            .get_mut(&stream_id.0)
            .ok_or(ServerError::IllegalInState {
                state: state_label(&self.state),
                method: "send_response_headers (unknown stream)",
            })?;
        if !matches!(entry.send, SendState::Idle) {
            return Err(ServerError::IllegalInState {
                state: send_label(&entry.send),
                method: "send_response_headers",
            });
        }
        let mut header_block = Vec::new();
        let header_iter = headers.iter().copied();
        qpack::encoder::encode_refs(header_iter, &mut header_block).map_err(|_| {
            ServerError::Request(RequestError::Frame(frame::FrameError::BufferTooSmall {
                needed: 0,
            }))
        })?;
        encode_h3_frame_to_vec(
            &H3Frame::Headers {
                header_block: &header_block,
            },
            &mut entry.outbound,
        )?;
        entry.send = SendState::HeadersSent;
        Ok(())
    }

    /// Queue response body bytes.
    ///
    /// # Errors
    ///
    /// Returns [`ServerError::IllegalInState`] when called before
    /// `send_response_headers`.
    pub fn send_response_data(
        &mut self,
        stream_id: StreamId,
        bytes: &[u8],
    ) -> Result<(), ServerError> {
        let entry = self
            .requests
            .get_mut(&stream_id.0)
            .ok_or(ServerError::IllegalInState {
                state: state_label(&self.state),
                method: "send_response_data (unknown stream)",
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
                return Err(ServerError::IllegalInState {
                    state: send_label(&entry.send),
                    method: "send_response_data",
                });
            }
        }
        encode_h3_frame_to_vec(&H3Frame::Data { payload: bytes }, &mut entry.outbound)?;
        Ok(())
    }

    /// Mark the response complete — sets the FIN bit on the next
    /// take_request_outbound and transitions send-state to Done.
    ///
    /// # Errors
    ///
    /// Returns [`ServerError::IllegalInState`] when invoked twice.
    pub fn finish_response(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        let entry = self
            .requests
            .get_mut(&stream_id.0)
            .ok_or(ServerError::IllegalInState {
                state: state_label(&self.state),
                method: "finish_response (unknown stream)",
            })?;
        if matches!(entry.send, SendState::Done) {
            return Err(ServerError::IllegalInState {
                state: send_label(&entry.send),
                method: "finish_response",
            });
        }
        entry.fin_pending = true;
        entry.send = SendState::Done;
        Ok(())
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
        if self.local_settings.h3_datagram {
            cursor += varint::encode(
                crate::http3_codec::settings::SETTINGS_H3_DATAGRAM,
                &mut payload[cursor..],
            )
            .unwrap_or(0);
            cursor += varint::encode(1, &mut payload[cursor..]).unwrap_or(0);
        }
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
        let _ = encode_h3_frame_to_vec(&frame, &mut self.control_outbound);
        self.settings_emitted = true;
    }
}

fn apply_inbound_request_frame(
    stream_id: StreamId,
    recv: &mut RecvState,
    frame_obj: H3Frame<'_>,
    events: &mut alloc::collections::VecDeque<H3ServerEvent>,
    max_field_section_size: u64,
) -> Result<(), ServerError> {
    match (core::mem::take(recv), frame_obj) {
        (RecvState::Idle, H3Frame::Headers { header_block }) => {
            let headers = qpack::decoder::decode_bounded(header_block, max_field_section_size)?;
            *recv = RecvState::HeadersReceived {
                headers: headers.clone(),
            };
            events.push_back(H3ServerEvent::RequestHeaders { stream_id, headers });
        }
        (RecvState::HeadersReceived { headers }, H3Frame::Data { payload }) => {
            let bytes = payload.to_vec();
            *recv = RecvState::BodyReceiving {
                headers,
                body_so_far: bytes.clone(),
            };
            events.push_back(H3ServerEvent::RequestData { stream_id, bytes });
        }
        (
            RecvState::BodyReceiving {
                headers,
                mut body_so_far,
            },
            H3Frame::Data { payload },
        ) => {
            body_so_far.extend_from_slice(payload);
            *recv = RecvState::BodyReceiving {
                headers,
                body_so_far,
            };
            events.push_back(H3ServerEvent::RequestData {
                stream_id,
                bytes: payload.to_vec(),
            });
        }
        (RecvState::HeadersReceived { headers }, H3Frame::Headers { header_block }) => {
            // Trailers without any body — legal per RFC 9114 §4.1.
            let trailers = qpack::decoder::decode_bounded(header_block, max_field_section_size)?;
            *recv = RecvState::TrailersReceived {
                headers,
                body: Vec::new(),
                trailers: trailers.clone(),
            };
            events.push_back(H3ServerEvent::RequestTrailers {
                stream_id,
                trailers,
            });
        }
        (
            RecvState::BodyReceiving {
                headers,
                body_so_far,
            },
            H3Frame::Headers { header_block },
        ) => {
            let trailers = qpack::decoder::decode_bounded(header_block, max_field_section_size)?;
            *recv = RecvState::TrailersReceived {
                headers,
                body: body_so_far,
                trailers: trailers.clone(),
            };
            events.push_back(H3ServerEvent::RequestTrailers {
                stream_id,
                trailers,
            });
        }
        (prior, H3Frame::Reserved { frame_type, .. }) => {
            // RFC 9114 §11.2.1 — the four HTTP/2-reserved types MUST
            // be a connection error on ANY stream class. §7.2.8 says
            // every other reserved type (GREASE 0x21, 0x40, 0x5f, …;
            // any future-assigned type) MUST be ignored even on a
            // request stream. The prior shape rejected all of them,
            // which was an inversion of the spec.
            if frame::is_http2_reserved(frame_type) {
                return Err(ServerError::Request(RequestError::UnexpectedFrame));
            }
            // Ignore + restore the recv-state we took out.
            *recv = prior;
        }
        (_, _) => {
            return Err(ServerError::Request(RequestError::UnexpectedFrame));
        }
    }
    Ok(())
}

fn encode_h3_frame_to_vec(frame_obj: &H3Frame<'_>, out: &mut Vec<u8>) -> Result<(), ServerError> {
    // First call with a generous buffer; on BufferTooSmall, resize +
    // retry once (the FrameError carries the needed length).
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
                .map_err(|err| ServerError::Request(RequestError::Frame(err)))?;
            out.truncate(initial_len + written);
            Ok(())
        }
        Err(other) => Err(ServerError::Request(RequestError::Frame(other))),
    }
}

fn state_label(state: &ServerState) -> &'static str {
    match state {
        ServerState::Negotiating => "Negotiating",
        ServerState::Established => "Established",
        ServerState::Closing { .. } => "Closing",
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

    fn encode_settings_frame_with(pairs: &[(u64, u64)]) -> Vec<u8> {
        let mut payload = [0u8; 64];
        let mut cursor = 0;
        use crate::quic::varint;
        for (id, value) in pairs {
            cursor += varint::encode(*id, &mut payload[cursor..]).unwrap();
            cursor += varint::encode(*value, &mut payload[cursor..]).unwrap();
        }
        let mut out = alloc::vec![0u8; cursor + 4];
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

    fn encode_headers_frame(pairs: &[(&[u8], &[u8])]) -> Vec<u8> {
        let mut header_block = Vec::new();
        qpack::encoder::encode_refs(pairs.iter().copied(), &mut header_block).unwrap();
        let mut out = alloc::vec![0u8; header_block.len() + 4];
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

    fn encode_data_frame(payload: &[u8]) -> Vec<u8> {
        let mut out = alloc::vec![0u8; payload.len() + 4];
        let written = frame::encode(&H3Frame::Data { payload }, &mut out).unwrap();
        out.truncate(written);
        out
    }

    #[test]
    fn settings_exchange_transitions_negotiating_to_established() {
        let mut conn = ServerConnection::new(Settings::default());
        assert_eq!(conn.state(), &ServerState::Negotiating);
        let settings_frame = encode_settings_frame_with(&[(0x01, 4096), (0x06, 16384)]);
        conn.feed_control(&settings_frame).unwrap();
        assert_eq!(conn.state(), &ServerState::Established);
        let event = conn.poll_event().unwrap();
        let H3ServerEvent::SettingsEstablished { peer } = event else {
            panic!("expected SettingsEstablished");
        };
        assert_eq!(peer.qpack_max_table_capacity, 4096);
        assert_eq!(peer.max_field_section_size, 16384);
    }

    fn encode_goaway_frame(id: u64) -> Vec<u8> {
        let mut out = alloc::vec![0u8; 16];
        let written = frame::encode(&H3Frame::GoAway { id }, &mut out).unwrap();
        out.truncate(written);
        out
    }

    #[test]
    fn control_stream_rejects_frame_before_settings() {
        // RFC 9114 §6.2.1 — SETTINGS MUST be the first frame on the
        // control stream. A GoAway arriving first must be rejected.
        let mut conn = ServerConnection::new(Settings::default());
        let err = conn.feed_control(&encode_goaway_frame(0)).unwrap_err();
        assert!(
            matches!(
                err,
                ServerError::Settings(SettingsError::MissingSettings { observed_id: 0x07 })
            ),
            "expected MissingSettings, got {err:?}"
        );
    }

    #[test]
    fn control_stream_rejects_duplicate_settings_frame() {
        // RFC 9114 §7.2.4 — exactly one SETTINGS frame per control
        // stream; a second is H3_FRAME_UNEXPECTED.
        let mut conn = ServerConnection::new(Settings::default());
        conn.feed_control(&encode_settings_frame_with(&[(0x06, 1024)]))
            .unwrap();
        let err = conn
            .feed_control(&encode_settings_frame_with(&[(0x06, 2048)]))
            .unwrap_err();
        assert!(
            matches!(err, ServerError::Settings(SettingsError::DuplicateFrame)),
            "expected DuplicateFrame, got {err:?}"
        );
    }

    #[test]
    fn local_settings_emitted_on_control_outbound_after_first_feed() {
        let mut conn = ServerConnection::new(Settings::default());
        let settings_frame = encode_settings_frame_with(&[(0x06, 1024)]);
        conn.feed_control(&settings_frame).unwrap();
        let outbound = conn.take_control_outbound();
        assert!(!outbound.is_empty(), "server must emit its SETTINGS frame");
        let (parsed, _) = frame::parse(&outbound).unwrap();
        assert!(matches!(parsed, H3Frame::Settings { .. }));
    }

    #[test]
    fn request_headers_emit_event_with_decoded_fields() {
        let mut conn = ServerConnection::new(Settings::default());
        conn.feed_control(&encode_settings_frame_with(&[])).unwrap();
        let _ = conn.poll_event(); // drain SettingsEstablished
        let request_frame = encode_headers_frame(&[
            (b":method", b"GET"),
            (b":scheme", b"https"),
            (b":authority", b"example.com"),
            (b":path", b"/"),
        ]);
        conn.feed_request(StreamId(0), &request_frame, true)
            .unwrap();
        let H3ServerEvent::RequestHeaders { stream_id, headers } = conn.poll_event().unwrap()
        else {
            panic!("expected RequestHeaders");
        };
        assert_eq!(stream_id, StreamId(0));
        assert_eq!(headers.len(), 4);
        assert_eq!(headers[0].name, b":method".to_vec());
        assert_eq!(headers[0].value, b"GET".to_vec());
        let finished = conn.poll_event().unwrap();
        assert!(matches!(finished, H3ServerEvent::RequestFinished { .. }));
    }

    #[test]
    fn send_response_headers_then_data_then_finish() {
        let mut conn = ServerConnection::new(Settings::default());
        conn.feed_control(&encode_settings_frame_with(&[])).unwrap();
        let _ = conn.poll_event();
        // Create a request first so the stream entry exists.
        conn.feed_request(
            StreamId(0),
            &encode_headers_frame(&[(b":method", b"GET"), (b":path", b"/")]),
            false,
        )
        .unwrap();
        let _ = conn.poll_event(); // drain RequestHeaders.

        conn.send_response_headers(
            StreamId(0),
            &[(b":status", b"200"), (b"content-type", b"text/plain")],
        )
        .unwrap();
        conn.send_response_data(StreamId(0), b"hello\n").unwrap();
        conn.finish_response(StreamId(0)).unwrap();
        let (outbound, fin) = conn.take_request_outbound(StreamId(0)).unwrap();
        assert!(fin, "FIN bit must be set on the final write");
        // Parse the queued bytes back: HEADERS then DATA.
        let (first, consumed) = frame::parse(&outbound).unwrap();
        assert!(matches!(first, H3Frame::Headers { .. }));
        let (second, _) = frame::parse(&outbound[consumed..]).unwrap();
        let H3Frame::Data { payload } = second else {
            panic!("expected Data");
        };
        assert_eq!(payload, b"hello\n");
    }

    #[test]
    fn request_body_emits_data_event() {
        let mut conn = ServerConnection::new(Settings::default());
        conn.feed_control(&encode_settings_frame_with(&[])).unwrap();
        let _ = conn.poll_event();
        let mut inbound = encode_headers_frame(&[(b":method", b"POST"), (b":path", b"/upload")]);
        inbound.extend_from_slice(&encode_data_frame(b"chunk1"));
        inbound.extend_from_slice(&encode_data_frame(b"chunk2"));
        conn.feed_request(StreamId(0), &inbound, true).unwrap();
        // drain events: headers, data, data, finished.
        assert!(matches!(
            conn.poll_event().unwrap(),
            H3ServerEvent::RequestHeaders { .. }
        ));
        let H3ServerEvent::RequestData { bytes, .. } = conn.poll_event().unwrap() else {
            panic!("expected data");
        };
        assert_eq!(bytes, b"chunk1");
        let H3ServerEvent::RequestData { bytes, .. } = conn.poll_event().unwrap() else {
            panic!("expected data");
        };
        assert_eq!(bytes, b"chunk2");
        assert!(matches!(
            conn.poll_event().unwrap(),
            H3ServerEvent::RequestFinished { .. }
        ));
    }

    #[test]
    fn goaway_received_transitions_to_closing() {
        let mut conn = ServerConnection::new(Settings::default());
        conn.feed_control(&encode_settings_frame_with(&[])).unwrap();
        let _ = conn.poll_event();

        let mut goaway = alloc::vec![0u8; 8];
        let written = frame::encode(&H3Frame::GoAway { id: 16 }, &mut goaway).unwrap();
        goaway.truncate(written);
        conn.feed_control(&goaway).unwrap();
        assert_eq!(
            conn.state(),
            &ServerState::Closing {
                peer_max_stream_id: 16
            }
        );
    }

    /// C35, server half of `docs/proxima-pipe/part-source-sink-design.md`
    /// step 3: a connection opted into [`RequestHeaderMode::Source`]
    /// delivers request HEADERS as a lazy `PartSource` on
    /// [`ServerConnection::poll_request_header_source`], NOT the owned
    /// `RequestHeaders` event — same mutual exclusion + routing contract
    /// as the client half.
    #[cfg(feature = "http3_codec-part-source")]
    #[test]
    fn request_header_source_mode_emits_part_source_not_owned_event() {
        use proxima_primitives::pipe::part::{Part, PartSource as _};

        let frame_bytes = encode_headers_frame(&[
            (b":method", b"GET"),
            (b":path", b"/v1/items"),
            (b"user-agent", b"curl/8.7.1"),
        ]);
        let mut conn = ServerConnection::new(Settings::default());
        conn.enable_header_source_mode();
        conn.feed_request(StreamId(0), &frame_bytes, false)
            .expect("feed request headers");

        assert!(
            conn.poll_event().is_none(),
            "Source mode must not also emit the owned RequestHeaders event"
        );

        let (stream_id, mut source) = conn
            .poll_request_header_source()
            .expect("headers queued on the source path");
        assert_eq!(stream_id, StreamId(0));
        assert_eq!(source.next(), Some(Part::Method(b"GET")));
        assert_eq!(source.next(), Some(Part::Path(b"/v1/items")));
        assert_eq!(
            source.next(),
            Some(Part::Header(b"user-agent", b"curl/8.7.1"))
        );
        assert_eq!(source.next(), Some(Part::End));
        assert_eq!(source.next(), None);
        assert_eq!(source.error(), None);
    }

    /// DC-H3-FACADE-EVENTS-OWN, server half: `feed_request` in `Source`
    /// mode performs 0 heap allocations steady-state (pool-recycled block
    /// buffer), and polling + stepping the lazy source is also 0 — versus
    /// the `Owned` default's `decode_bounded` (`1 + 2*field_count`) plus
    /// the recv-state clone. Warm-up discipline as on the client: feed
    /// one stream, poll twice (the second poll recycles the block into
    /// the pool), then measure a second stream.
    #[cfg(all(feature = "http3_codec-part-source", feature = "std"))]
    #[test]
    fn alloc_count_feed_request_source_mode_is_zero_owned_mode_is_greater_than_zero() {
        let frame_bytes = encode_headers_frame(&[
            (b":method", b"GET"),
            (b":scheme", b"https"),
            (b":authority", b"bench.local"),
            (b":path", b"/"),
            (b"user-agent", b"rekt/0.1"),
        ]);

        let mut source_conn = ServerConnection::new(Settings::default());
        source_conn.enable_header_source_mode();
        source_conn
            .feed_request(StreamId(0), &frame_bytes, false)
            .expect("warm-up feed");
        let _ = source_conn.poll_request_header_source();
        let _ = source_conn.poll_request_header_source();

        let region = crate::alloc_test::exclusive_region();
        let before_feed = region.change();
        source_conn
            .feed_request(StreamId(4), &frame_bytes, false)
            .expect("measured source-mode feed");
        let after_feed = region.change();
        assert_eq!(
            after_feed.allocations - before_feed.allocations,
            0,
            "Source-mode feed_request must perform 0 heap allocations for a 5-field request"
        );

        let before_poll = region.change();
        {
            use proxima_primitives::pipe::part::{Part, PartSource as _};
            let (_, mut source) = source_conn
                .poll_request_header_source()
                .expect("headers queued on the source path");
            let mut saw_method = false;
            while let Some(part) = source.next() {
                if part == Part::Method(b"GET") {
                    saw_method = true;
                }
            }
            assert!(saw_method);
            assert_eq!(source.error(), None);
        }
        let after_poll = region.change();
        assert_eq!(
            after_poll.allocations - before_poll.allocations,
            0,
            "polling + stepping the lazy source must also perform 0 heap allocations"
        );

        let mut owned_conn = ServerConnection::new(Settings::default());
        owned_conn
            .feed_request(StreamId(0), &frame_bytes, false)
            .expect("warm-up feed");
        while owned_conn.poll_event().is_some() {}

        let before_owned = region.change();
        owned_conn
            .feed_request(StreamId(4), &frame_bytes, false)
            .expect("measured owned-mode feed");
        let after_owned = region.change();
        assert!(
            after_owned.allocations - before_owned.allocations > 0,
            "Owned-mode feed_request still pays decode_bounded's 1 + 2*field_count \
             plus the recv-state clone — the cost Source mode opts out of"
        );
    }
}
