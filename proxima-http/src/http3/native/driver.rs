//! Per-connection H3 driver — routes QUIC stream bytes through the
//! sans-IO [`ServerConnection`] / [`ClientConnection`] state machines.
//!
//! H3 over QUIC uses three classes of streams (RFC 9114 §6):
//!
//! - **Control stream**: one local-initiated unidirectional stream per
//!   endpoint, type byte `0x00`, carries SETTINGS / GOAWAY / etc.
//! - **QPACK encoder / decoder streams**: types `0x02` / `0x03`. We
//!   don't drive a dynamic QPACK table in v1 so we observe + discard.
//! - **Request streams**: bidirectional, opened by the client; carry
//!   HEADERS + DATA + trailing HEADERS.
//!
//! The driver tracks the routing per stream so [`drive_server_step`]
//! / [`drive_client_step`] can be called in a loop to advance work
//! without the consumer needing to know any of the wire-level rules.
//!
//! No async, no I/O — pure sans-IO over the proto layer's bytes-in /
//! bytes-out API.

use std::collections::BTreeMap;

use proxima_protocols::http3_codec::client::ClientConnection;
use proxima_protocols::http3_codec::frame as h3_frame;
use proxima_protocols::http3_codec::server::{ServerConnection, StreamId as H3StreamId};
use proxima_protocols::quic::connection::{Connection, ConnectionError};
use proxima_protocols::quic::streams::{StreamDirection, StreamId};
use proxima_protocols::quic::tls::TlsProvider;
use proxima_protocols::quic::varint;

/// Per-stream routing decision made by the driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamKind {
    /// Peer-opened uni stream identified as the H3 control stream
    /// (first varint byte = 0x00).
    PeerControl,
    /// Peer-opened uni stream identified as a QPACK encoder/decoder
    /// stream (types 0x02 / 0x03). Bytes discarded in v1.
    QpackOther,
    /// Peer-opened bidi stream — an H3 request.
    Request,
    /// Local-initiated uni stream we use as our control stream.
    LocalControl,
    /// Some peer-opened uni stream type we don't recognise. Discard.
    Unknown,
}

/// Everything the driver tracks about one QUIC stream, keyed by stream
/// id in `DriverState::streams`. Consolidating what used to be seven
/// parallel `BTreeMap`/`BTreeSet` lookups into one entry means a stream
/// touch is a single hash lookup instead of up to seven O(log n) tree
/// walks.
#[derive(Default)]
struct StreamEntry {
    /// This stream's identified routing kind. `None` until the
    /// type-byte varint has been consumed (was: absence from the old
    /// `kinds` map).
    kind: Option<StreamKind>,
    /// Bytes buffered while we wait for the full type-byte varint on a
    /// peer-opened uni stream. Almost always 0–1 bytes since the
    /// common types are single-byte varints, but the varint API is
    /// generic.
    type_byte_buf: Vec<u8>,
    /// Accumulator for a peer-control-stream: H3 control frames arrive
    /// over a stream of bytes that may not align with QUIC stream
    /// segment boundaries. feed_control errors with Truncated on a
    /// half-arrived frame, so the driver buffers + retries until the
    /// parser consumes the whole prefix.
    control_recv_buf: Vec<u8>,
    /// Same pattern as `control_recv_buf` for a request stream — the
    /// driver accumulates bytes locally and only feeds the H3 layer
    /// once enough has arrived for a full frame to parse.
    request_recv_buf: Vec<u8>,
    /// Outbound bytes that QUIC's send buffer rejected on a prior
    /// driver pass (`send_application` returned `accepted < len`). The
    /// H3 layer has already considered these bytes "taken" — if the
    /// driver dropped them the H3 stream would be silently truncated.
    /// Drained head-first on every drive_*_step before any new bytes
    /// are pulled from `take_*_outbound`.
    outbound_pending: Vec<u8>,
    /// Set once `feed_request(..., fin=true)` / `feed_response(...,
    /// fin=true)` has been reported so we don't double-feed FIN.
    fin_reported: bool,
    /// Set when this stream's FIN was deferred because
    /// `outbound_pending` still held un-shipped bytes. Cleared once
    /// the buffer empties — FIN fires only then, otherwise we'd close
    /// the send half on bytes the peer never received.
    fin_pending: bool,
}

/// Routing state for one connection's stream multiplex.
pub struct DriverState {
    /// Local-initiated H3 control stream (our SETTINGS).
    local_control: Option<StreamId>,
    /// Bytes left to write before we've emitted the 0x00 type byte +
    /// any staged outbound for our control stream.
    local_control_type_byte_sent: bool,
    /// Per-stream bookkeeping, keyed by QUIC stream id. See
    /// [`StreamEntry`].
    streams: BTreeMap<u64, StreamEntry>,
    /// RFC 9114 §6.2.1 — a peer MUST open exactly one control stream.
    /// We record the stream-id of the first one we see classified as
    /// `PeerControl` and treat a second classification as a protocol
    /// violation (the driver returns ProtocolViolation; the facade
    /// maps to H3_STREAM_CREATION_ERROR).
    peer_control_stream: Option<u64>,
}

impl Default for DriverState {
    fn default() -> Self {
        Self::new()
    }
}

impl DriverState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            local_control: None,
            local_control_type_byte_sent: false,
            streams: BTreeMap::new(),
            peer_control_stream: None,
        }
    }

    /// The local stream id we use for our outbound control stream, or
    /// `None` if we haven't opened it yet.
    #[must_use]
    pub fn local_control_stream(&self) -> Option<StreamId> {
        self.local_control
    }

    /// Drop all per-stream bookkeeping for a completed request stream.
    /// Without this the per-stream map grows with lifetime request count
    /// and the listener's per-burst scans of it go O(lifetime) — the
    /// dominant cost under sustained multiplexed load.
    pub fn forget_stream(&mut self, stream_id: u64) {
        self.streams.remove(&stream_id);
    }

    /// Whether `stream_id`'s FIN was deferred pending outbound drain,
    /// clearing the flag if so — mirrors `BTreeSet::remove`'s "was it
    /// present" return, but on a per-stream bool field.
    fn take_fin_pending(&mut self, stream_id: u64) -> bool {
        match self.streams.get_mut(&stream_id) {
            Some(entry) if entry.fin_pending => {
                entry.fin_pending = false;
                true
            }
            _ => false,
        }
    }
}

/// One driver step against a server-side connection. Routes any newly-
/// readable bytes from QUIC streams into the H3 state machine and
/// drains any newly-queued H3 bytes back to QUIC.
///
/// Idempotent: safe to call repeatedly from the consumer's event loop;
/// returns early if there's nothing to move.
///
/// # Errors
///
/// Bubbles [`ConnectionError`] from the underlying QUIC layer and any
/// H3 layer error wrapped as [`ConnectionError::ProtocolViolation`].
pub fn drive_server_step<P: TlsProvider>(
    connection: &mut Connection<P>,
    h3: &mut ServerConnection,
    state: &mut DriverState,
) -> Result<(), ConnectionError> {
    open_local_control_if_needed(connection, state, /* is_client */ false)?;
    drain_local_control_server(connection, h3, state)?;
    route_inbound_streams_server(connection, h3, state)?;
    drain_request_streams_server(connection, h3, state)?;
    // Free completed request entries so the fixed-cap H3 map
    // reflects concurrent streams, not lifetime total.
    h3.gc_completed();
    connection.reap_closed_streams();
    Ok(())
}

/// One driver step against a client-side connection. Symmetric to
/// [`drive_server_step`]; on the client side we read responses on
/// open request streams + ship request bytes on the same.
///
/// # Errors
///
/// Same as [`drive_server_step`].
pub fn drive_client_step<P: TlsProvider>(
    connection: &mut Connection<P>,
    h3: &mut ClientConnection,
    state: &mut DriverState,
) -> Result<(), ConnectionError> {
    open_local_control_if_needed(connection, state, /* is_client */ true)?;
    drain_local_control_client(connection, h3, state)?;
    route_inbound_streams_client(connection, h3, state)?;
    drain_request_streams_client(connection, h3, state)?;
    h3.gc_completed();
    // Reap AFTER the read pass: route_inbound_streams_client's read_stream is
    // what flips a fully-consumed response stream to terminal, so freeing
    // slots must happen here, not in the earlier in-frame reap.
    connection.reap_closed_streams();
    Ok(())
}

fn open_local_control_if_needed<P: TlsProvider>(
    connection: &mut Connection<P>,
    state: &mut DriverState,
    _is_client: bool,
) -> Result<(), ConnectionError> {
    if state.local_control.is_some() {
        return Ok(());
    }
    // Only legal once the connection has reached Established; the
    // caller is responsible for not calling drive_* before then.
    let id = connection.open_stream(StreamDirection::Uni)?;
    state.local_control = Some(id);
    state.streams.entry(id.as_u64()).or_default().kind = Some(StreamKind::LocalControl);
    Ok(())
}

fn drain_local_control_server<P: TlsProvider>(
    connection: &mut Connection<P>,
    h3: &mut ServerConnection,
    state: &mut DriverState,
) -> Result<(), ConnectionError> {
    let Some(local) = state.local_control else {
        return Ok(());
    };
    let _ = drain_pending(connection, state, local)?;
    let mut bytes = h3.take_control_outbound();
    if !state.local_control_type_byte_sent {
        // Prepend the H3 control-stream type byte (0x00 — single-byte
        // varint).
        let mut prefixed = Vec::with_capacity(bytes.len() + 1);
        prefixed.push(0x00);
        prefixed.append(&mut bytes);
        bytes = prefixed;
        state.local_control_type_byte_sent = true;
    }
    if bytes.is_empty() {
        return Ok(());
    }
    send_all_buffered(connection, state, local, &bytes)
}

fn drain_local_control_client<P: TlsProvider>(
    connection: &mut Connection<P>,
    h3: &mut ClientConnection,
    state: &mut DriverState,
) -> Result<(), ConnectionError> {
    let Some(local) = state.local_control else {
        return Ok(());
    };
    let _ = drain_pending(connection, state, local)?;
    let mut bytes = h3.take_control_outbound();
    if !state.local_control_type_byte_sent {
        let mut prefixed = Vec::with_capacity(bytes.len() + 1);
        prefixed.push(0x00);
        prefixed.append(&mut bytes);
        bytes = prefixed;
        state.local_control_type_byte_sent = true;
    }
    if bytes.is_empty() {
        return Ok(());
    }
    send_all_buffered(connection, state, local, &bytes)
}

fn route_inbound_streams_server<P: TlsProvider>(
    connection: &mut Connection<P>,
    h3: &mut ServerConnection,
    state: &mut DriverState,
) -> Result<(), ConnectionError> {
    // Only streams that received data/FIN since the last step, not the
    // whole table — O(active) per datagram, the difference between flat
    // and inverse throughput scaling. Drains the readable set (owned
    // Vec) so the loop body can take &mut connection freely.
    let ids = connection.take_readable()?;
    for id in ids {
        if state.streams.get(&id.as_u64()).and_then(|entry| entry.kind)
            == Some(StreamKind::LocalControl)
        {
            continue;
        }
        let is_local = id.is_local(side_of(connection));
        if is_local {
            continue;
        }
        let mut scratch = [0u8; 4096];
        let read_len = connection.read_stream(id, &mut scratch)?;
        // Scratch filled: more may remain buffered — finish next step.
        if read_len == scratch.len() {
            connection.mark_readable(id);
        }
        if read_len == 0 && !connection.stream_recv_finished(id).unwrap_or(false) {
            continue;
        }
        let bytes = &scratch[..read_len];
        let entry = state.streams.entry(id.as_u64()).or_default();
        let kind_now = entry.kind;
        match (id.direction(), kind_now) {
            (StreamDirection::Bidi, _) => {
                entry.kind = Some(StreamKind::Request);
                let fin = connection.stream_recv_finished(id).unwrap_or(false);
                let h3_id = H3StreamId(id.as_u64());
                let fresh = !entry.fin_reported;
                entry.request_recv_buf.extend_from_slice(bytes);
                let to_feed = drain_complete_frames(&mut entry.request_recv_buf)?;
                if !to_feed.is_empty() || (fin && fresh && entry.request_recv_buf.is_empty()) {
                    let feed_fin = fin && fresh && entry.request_recv_buf.is_empty();
                    h3.feed_request(h3_id, &to_feed, feed_fin).map_err(|_| {
                        ConnectionError::ProtocolViolation {
                            reason: "h3 feed_request failed",
                        }
                    })?;
                    let entry = state.streams.entry(id.as_u64()).or_default();
                    if fin && entry.request_recv_buf.is_empty() {
                        entry.fin_reported = true;
                    }
                }
                // RFC 9114 §7.1 — FIN with remaining bytes in the
                // buffer means a truncated H3 frame; this is
                // H3_FRAME_ERROR, not "wait for more data".
                let entry = state.streams.entry(id.as_u64()).or_default();
                if fin && fresh && !entry.request_recv_buf.is_empty() {
                    return Err(ConnectionError::ProtocolViolation {
                        reason: "h3 request stream ended with truncated frame (H3_FRAME_ERROR)",
                    });
                }
            }
            (StreamDirection::Uni, Some(StreamKind::PeerControl)) if !bytes.is_empty() => {
                entry.control_recv_buf.extend_from_slice(bytes);
                let to_feed = drain_complete_frames(&mut entry.control_recv_buf)?;
                if !to_feed.is_empty() {
                    h3.feed_control(&to_feed)
                        .map_err(|_| ConnectionError::ProtocolViolation {
                            reason: "h3 feed_control failed",
                        })?;
                }
            }
            (StreamDirection::Uni, Some(StreamKind::PeerControl)) => {
                // RFC 9114 §6.2.1 — closure of the peer control
                // stream MUST be H3_CLOSED_CRITICAL_STREAM.
                let fin = connection.stream_recv_finished(id).unwrap_or(false);
                if fin {
                    return Err(ConnectionError::ProtocolViolation {
                        reason: "h3 peer closed control stream (H3_CLOSED_CRITICAL_STREAM)",
                    });
                }
            }
            (StreamDirection::Uni, Some(StreamKind::QpackOther)) => {
                // RFC 9114 §6.2.1 — closure of QPACK encoder/decoder
                // streams MUST be H3_CLOSED_CRITICAL_STREAM.
                let fin = connection.stream_recv_finished(id).unwrap_or(false);
                if fin {
                    return Err(ConnectionError::ProtocolViolation {
                        reason: "h3 peer closed QPACK critical stream (H3_CLOSED_CRITICAL_STREAM)",
                    });
                }
            }
            (StreamDirection::Uni, Some(StreamKind::Unknown)) => {
                // discard
            }
            (StreamDirection::Uni, _) => {
                // First bytes on a peer-opened uni stream: accumulate
                // until we've parsed the type varint.
                entry.type_byte_buf.extend_from_slice(bytes);
                if let Some((stream_type, consumed)) = try_parse_varint(&entry.type_byte_buf) {
                    let kind = classify_uni_type(stream_type);
                    // RFC 9114 §6.2.1 — exactly one peer control stream.
                    if matches!(kind, StreamKind::PeerControl) {
                        if let Some(prior) = state.peer_control_stream
                            && prior != id.as_u64()
                        {
                            return Err(ConnectionError::ProtocolViolation {
                                reason: "h3 peer opened a second control stream",
                            });
                        }
                        state.peer_control_stream = Some(id.as_u64());
                    }
                    let entry = state.streams.entry(id.as_u64()).or_default();
                    entry.kind = Some(kind);
                    let leftover: Vec<u8> = entry.type_byte_buf[consumed..].to_vec();
                    entry.type_byte_buf = Vec::new();
                    let _ = stream_type;
                    if matches!(kind, StreamKind::PeerControl) && !leftover.is_empty() {
                        entry.control_recv_buf.extend_from_slice(&leftover);
                        let to_feed = drain_complete_frames(&mut entry.control_recv_buf)?;
                        if !to_feed.is_empty() {
                            h3.feed_control(&to_feed).map_err(|_| {
                                ConnectionError::ProtocolViolation {
                                    reason: "h3 feed_control failed",
                                }
                            })?;
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Pop complete H3 frames off the head of `buf` and return them as a
/// single contiguous slice (caller forwards to feed_control /
/// feed_request). `buf` retains any trailing partial frame for the
/// next driver pass.
///
/// Distinguishes incomplete frames (`FrameError::Truncated` — wait
/// for more bytes, buffer retained as-is) from real parse errors
/// (`FrameError::InvalidVarint`, `FrameError::PayloadTooLong` —
/// connection error, surfaced as `Err` so the caller maps to
/// `ProtocolViolation`). Without this split, a complete-but-malformed
/// frame blocks every subsequent frame on the stream indefinitely
/// because the prior shape treated every parse error as "wait for
/// more".
///
/// # Errors
///
/// Returns `Err(ConnectionError::ProtocolViolation)` when the head of
/// `buf` is a complete-but-malformed frame.
fn drain_complete_frames(buf: &mut Vec<u8>) -> Result<Vec<u8>, ConnectionError> {
    let mut consumed = 0;
    loop {
        if consumed == buf.len() {
            break;
        }
        match h3_frame::parse(&buf[consumed..]) {
            Ok((_, 0)) => break, // belt-and-suspenders: zero-length parse would loop forever
            Ok((_, frame_len)) => consumed += frame_len,
            // Truncated = we need more bytes from the QUIC layer;
            // retain the buffer as-is and try again next pass.
            Err(h3_frame::FrameError::Truncated) => break,
            // Anything else is a complete-but-malformed frame; the
            // bytes are already in our buffer, so "wait for more"
            // would block this stream forever. Surface as a
            // connection-level error.
            Err(h3_frame::FrameError::InvalidVarint) => {
                return Err(ConnectionError::ProtocolViolation {
                    reason: "h3 frame: invalid varint",
                });
            }
            Err(h3_frame::FrameError::PayloadTooLong) => {
                return Err(ConnectionError::ProtocolViolation {
                    reason: "h3 frame: payload length overflows usize",
                });
            }
            Err(h3_frame::FrameError::BufferTooSmall { .. }) => {
                // BufferTooSmall is an encoder error; should never
                // appear from a parse call. Treat as a protocol
                // violation rather than panic.
                return Err(ConnectionError::ProtocolViolation {
                    reason: "h3 frame: parse returned encoder error",
                });
            }
            // FrameError is #[non_exhaustive] — any future variant
            // is also "not truncation", so it's also a protocol
            // violation (better to refuse than to block the stream).
            Err(_) => {
                return Err(ConnectionError::ProtocolViolation {
                    reason: "h3 frame: parse failed",
                });
            }
        }
    }
    if consumed == 0 {
        return Ok(Vec::new());
    }
    Ok(buf.drain(..consumed).collect())
}

fn route_inbound_streams_client<P: TlsProvider>(
    connection: &mut Connection<P>,
    h3: &mut ClientConnection,
    state: &mut DriverState,
) -> Result<(), ConnectionError> {
    // Readable streams only (responses + peer control), not the whole
    // table — O(active) per datagram. See the server route for why.
    let ids = connection.take_readable()?;
    for id in ids {
        if state.streams.get(&id.as_u64()).and_then(|entry| entry.kind)
            == Some(StreamKind::LocalControl)
        {
            continue;
        }
        let is_local = id.is_local(side_of(connection));
        // For the client, request streams are LOCAL-initiated bidi —
        // we feed response bytes on those into the H3 client.
        let mut scratch = [0u8; 4096];
        let read_len = connection.read_stream(id, &mut scratch)?;
        // Scratch filled: more may remain buffered — finish next step.
        if read_len == scratch.len() {
            connection.mark_readable(id);
        }
        if read_len == 0 && !connection.stream_recv_finished(id).unwrap_or(false) {
            continue;
        }
        let bytes = &scratch[..read_len];
        match id.direction() {
            StreamDirection::Bidi if is_local => {
                let entry = state.streams.entry(id.as_u64()).or_default();
                let fin = connection.stream_recv_finished(id).unwrap_or(false);
                let h3_id = H3StreamId(id.as_u64());
                let fresh = !entry.fin_reported;
                entry.request_recv_buf.extend_from_slice(bytes);
                let to_feed = drain_complete_frames(&mut entry.request_recv_buf)?;
                if !to_feed.is_empty() || (fin && fresh && entry.request_recv_buf.is_empty()) {
                    h3.feed_response(
                        h3_id,
                        &to_feed,
                        fin && fresh && entry.request_recv_buf.is_empty(),
                    )
                    .map_err(|_| ConnectionError::ProtocolViolation {
                        reason: "h3 feed_response failed",
                    })?;
                    let entry = state.streams.entry(id.as_u64()).or_default();
                    if fin && entry.request_recv_buf.is_empty() {
                        entry.fin_reported = true;
                    }
                }
                let entry = state.streams.entry(id.as_u64()).or_default();
                if fin && fresh && !entry.request_recv_buf.is_empty() {
                    return Err(ConnectionError::ProtocolViolation {
                        reason: "h3 response stream ended with truncated frame (H3_FRAME_ERROR)",
                    });
                }
            }
            StreamDirection::Bidi if !is_local => {
                // RFC 9114 §6.1 — server-initiated bidi streams
                // are not permitted without a negotiated extension.
                // MUST close with H3_STREAM_CREATION_ERROR.
                return Err(ConnectionError::ProtocolViolation {
                    reason: "h3 server-initiated bidirectional stream (H3_STREAM_CREATION_ERROR)",
                });
            }
            StreamDirection::Bidi => {
                // This arm shouldn't be reachable (non-local was
                // caught above, local was caught by the first arm).
            }
            StreamDirection::Uni if is_local => continue,
            StreamDirection::Uni => {
                let entry = state.streams.entry(id.as_u64()).or_default();
                match entry.kind {
                    Some(StreamKind::PeerControl) => {
                        if !bytes.is_empty() {
                            entry.control_recv_buf.extend_from_slice(bytes);
                            let to_feed = drain_complete_frames(&mut entry.control_recv_buf)?;
                            if !to_feed.is_empty() {
                                h3.feed_control(&to_feed).map_err(|_| {
                                    ConnectionError::ProtocolViolation {
                                        reason: "h3 feed_control failed",
                                    }
                                })?;
                            }
                        }
                        let fin = connection.stream_recv_finished(id).unwrap_or(false);
                        if fin {
                            return Err(ConnectionError::ProtocolViolation {
                                reason: "h3 peer closed control stream (H3_CLOSED_CRITICAL_STREAM)",
                            });
                        }
                    }
                    Some(StreamKind::QpackOther) => {
                        let fin = connection.stream_recv_finished(id).unwrap_or(false);
                        if fin {
                            return Err(ConnectionError::ProtocolViolation {
                                reason: "h3 peer closed QPACK stream (H3_CLOSED_CRITICAL_STREAM)",
                            });
                        }
                    }
                    Some(StreamKind::Unknown) => {}
                    _ => {
                        entry.type_byte_buf.extend_from_slice(bytes);
                        if let Some((stream_type, consumed)) =
                            try_parse_varint(&entry.type_byte_buf)
                        {
                            let kind = classify_uni_type(stream_type);
                            if matches!(kind, StreamKind::PeerControl) {
                                if let Some(prior) = state.peer_control_stream
                                    && prior != id.as_u64()
                                {
                                    return Err(ConnectionError::ProtocolViolation {
                                        reason: "h3 peer opened a second control stream",
                                    });
                                }
                                state.peer_control_stream = Some(id.as_u64());
                            }
                            let entry = state.streams.entry(id.as_u64()).or_default();
                            entry.kind = Some(kind);
                            let leftover: Vec<u8> = entry.type_byte_buf[consumed..].to_vec();
                            entry.type_byte_buf = Vec::new();
                            if matches!(kind, StreamKind::PeerControl) && !leftover.is_empty() {
                                entry.control_recv_buf.extend_from_slice(&leftover);
                                let to_feed = drain_complete_frames(&mut entry.control_recv_buf)?;
                                if !to_feed.is_empty() {
                                    h3.feed_control(&to_feed).map_err(|_| {
                                        ConnectionError::ProtocolViolation {
                                            reason: "h3 feed_control failed",
                                        }
                                    })?;
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn drain_request_streams_server<P: TlsProvider>(
    connection: &mut Connection<P>,
    h3: &mut ServerConnection,
    state: &mut DriverState,
) -> Result<(), ConnectionError> {
    // First-pass: streams whose prior outbound was partially shipped (or
    // whose FIN is still pending) get a retry before fresh bytes. Both
    // sets are gathered from the single per-stream map and filtered to
    // only the entries that need work, so this is O(pending) — not a
    // scan of every known stream. Kept as two separately-sorted, chained
    // passes (pending-buffer streams first, then fin-pending streams,
    // ascending by id, duplicates allowed) to match the prior
    // `BTreeMap`/`BTreeSet` iteration order exactly.
    let retry_ids: Vec<u64> = {
        let mut with_pending: Vec<u64> = state
            .streams
            .iter()
            .filter(|(_, entry)| !entry.outbound_pending.is_empty())
            .map(|(id, _)| *id)
            .collect();
        with_pending.sort_unstable();
        let mut with_fin_pending: Vec<u64> = state
            .streams
            .iter()
            .filter(|(_, entry)| entry.fin_pending)
            .map(|(id, _)| *id)
            .collect();
        with_fin_pending.sort_unstable();
        with_pending.into_iter().chain(with_fin_pending).collect()
    };
    for raw in retry_ids {
        let id = StreamId(raw);
        let drained = drain_pending(connection, state, id)?;
        if drained && state.take_fin_pending(raw) {
            connection.close_send(id)?;
        }
    }
    // Fresh response bytes — only streams the H3 layer has queued output
    // for, not every request stream. This is what lets server throughput
    // climb with concurrency instead of staying flat.
    for raw in h3.streams_with_outbound() {
        let id = StreamId(raw);
        let Some((bytes, fin)) = h3.take_request_outbound(H3StreamId(raw)) else {
            continue;
        };
        if !bytes.is_empty() {
            send_all_buffered(connection, state, id, &bytes)?;
        }
        if fin {
            // Only safe to half-close the send side once the per-stream
            // pending buffer is empty — otherwise close_send fires on
            // bytes the peer hasn't received and h3 truncates.
            let has_pending = state
                .streams
                .get(&id.as_u64())
                .is_some_and(|entry| !entry.outbound_pending.is_empty());
            if has_pending {
                state.streams.entry(id.as_u64()).or_default().fin_pending = true;
            } else {
                connection.close_send(id)?;
            }
        }
    }
    Ok(())
}

fn drain_request_streams_client<P: TlsProvider>(
    connection: &mut Connection<P>,
    h3: &mut ClientConnection,
    state: &mut DriverState,
) -> Result<(), ConnectionError> {
    // Open a QUIC bidi for every H3 request stream that doesn't have one yet.
    // The facade (`Client::open_request`) opens QUIC at request-creation, but
    // the proto-direct path (`ClientConnection::open_request` + drive) leaves
    // it to the driver, so this MUST stay — dropping it broke the direct path
    // with "send_application on unknown stream". The prior code rebuilt a
    // BTreeSet of ALL stream ids every pass (O(N log N) + an alloc — the real
    // "more streams hurts" cost); `has_stream` is an O(1) per-request check
    // with no allocation, keeping the perf win.
    let h3_request_ids: Vec<u64> = h3.request_stream_ids().map(|id| id.0).collect();
    for h3_id in &h3_request_ids {
        let quic_id = StreamId(*h3_id);
        if !connection.has_stream(quic_id) {
            let opened = connection.open_stream(StreamDirection::Bidi)?;
            // Both sides hand out the same next-id (RFC 9000 §2.1); a mismatch
            // means a stream was opened out-of-band.
            debug_assert_eq!(opened.as_u64(), *h3_id);
        }
    }
    // First-pass: retry per-stream pending bytes from a prior backpressure
    // event before pulling fresh outbound from the H3 layer.
    for h3_id in &h3_request_ids {
        let quic_id = StreamId(*h3_id);
        let drained = drain_pending(connection, state, quic_id)?;
        if drained && state.take_fin_pending(*h3_id) {
            connection.close_send(quic_id)?;
        }
    }
    for h3_id in h3_request_ids {
        let Some((bytes, fin)) = h3.take_request_outbound(H3StreamId(h3_id)) else {
            continue;
        };
        let quic_id = StreamId(h3_id);
        if !bytes.is_empty() {
            send_all_buffered(connection, state, quic_id, &bytes)?;
        }
        if fin {
            let has_pending = state
                .streams
                .get(&h3_id)
                .is_some_and(|entry| !entry.outbound_pending.is_empty());
            if has_pending {
                state.streams.entry(h3_id).or_default().fin_pending = true;
            } else {
                connection.close_send(quic_id)?;
            }
        }
    }
    Ok(())
}

/// Push `bytes` to a QUIC stream, retaining any unaccepted suffix in
/// the stream's `outbound_pending` buffer so a future driver pass can
/// retry. NEVER
/// drops bytes on the floor — the caller has already drained these
/// from the H3 layer's take_*_outbound buffer, so silently losing them
/// would truncate the H3 stream.
fn send_all_buffered<P: TlsProvider>(
    connection: &mut Connection<P>,
    state: &mut DriverState,
    stream: StreamId,
    bytes: &[u8],
) -> Result<(), ConnectionError> {
    let mut cursor = 0;
    while cursor < bytes.len() {
        let accepted = connection.send_application(stream, &bytes[cursor..])?;
        if accepted == 0 {
            break;
        }
        cursor += accepted;
    }
    if cursor < bytes.len() {
        state
            .streams
            .entry(stream.as_u64())
            .or_default()
            .outbound_pending
            .extend_from_slice(&bytes[cursor..]);
    }
    Ok(())
}

/// Try to ship the per-stream `outbound_pending` buffer to QUIC. On
/// `accepted < len` the unconsumed tail stays buffered; on full drain
/// the buffer is cleared. Returns `true` when the buffer is now empty
/// (caller-deferred FIN may fire).
fn drain_pending<P: TlsProvider>(
    connection: &mut Connection<P>,
    state: &mut DriverState,
    stream: StreamId,
) -> Result<bool, ConnectionError> {
    let key = stream.as_u64();
    let Some(entry) = state.streams.get_mut(&key) else {
        return Ok(true);
    };
    if entry.outbound_pending.is_empty() {
        return Ok(true);
    }
    let buf = &mut entry.outbound_pending;
    let mut cursor = 0;
    while cursor < buf.len() {
        let accepted = connection.send_application(stream, &buf[cursor..])?;
        if accepted == 0 {
            break;
        }
        cursor += accepted;
    }
    if cursor == 0 {
        return Ok(false);
    }
    if cursor == buf.len() {
        buf.clear();
        Ok(true)
    } else {
        buf.drain(..cursor);
        Ok(false)
    }
}

fn side_of<P: TlsProvider>(_connection: &Connection<P>) -> proxima_protocols::quic::side::Side {
    P::SIDE
}

fn classify_uni_type(stream_type: u64) -> StreamKind {
    match stream_type {
        0x00 => StreamKind::PeerControl,
        0x02 | 0x03 => StreamKind::QpackOther,
        _ => StreamKind::Unknown,
    }
}

fn try_parse_varint(bytes: &[u8]) -> Option<(u64, usize)> {
    varint::decode(bytes).ok()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// A `GoAway` frame whose declared length holds more bytes than
    /// the varint id consumes — RFC 9114 §7.2.6 says the entire
    /// payload must be a single varint, so trailing bytes are a
    /// connection error. Prior shape: drain_complete_frames
    /// swallowed the Err and left the bad frame buffered — every
    /// subsequent driver pass tried to re-parse it, blocking the
    /// stream forever. Fixed shape: surfaces as ProtocolViolation
    /// so the listener can close the connection.
    #[test]
    fn drain_complete_frames_surfaces_malformed_frame_as_protocol_violation() {
        // GoAway frame: type=0x07 (1 byte), length=5 (1 byte),
        // payload=[0x00, 0xAA, 0xBB, 0xCC, 0xDD]. The varint decode
        // of payload[0]=0x00 consumes 1 byte (id=0), but payload.len()=5
        // — frame.rs:166 enforces consumed == payload.len(), so this
        // returns FrameError::InvalidVarint.
        let mut buf = vec![0x07u8, 0x05, 0x00, 0xAA, 0xBB, 0xCC, 0xDD];
        let err = drain_complete_frames(&mut buf)
            .expect_err("malformed GOAWAY must surface as ProtocolViolation");
        assert!(matches!(err, ConnectionError::ProtocolViolation { .. }));
    }

    /// Truncated frames (less data than the declared length) MUST
    /// stay buffered for a future driver pass — this is the
    /// fragmentation case across QUIC datagrams.
    #[test]
    fn drain_complete_frames_retains_truncated_frame() {
        // Type=0x07, declared length=4, but only 2 payload bytes
        // present.
        let mut buf = vec![0x07u8, 0x04, 0x00, 0x01];
        let drained = drain_complete_frames(&mut buf).expect("truncation must not error");
        assert!(drained.is_empty(), "no complete frames yet");
        assert_eq!(buf.len(), 4, "truncated bytes retained for next pass");
    }
}
