//! Multi-connection QUIC server listener over the sans-IO proto.
//!
//! I/O-free DCID-keyed [`EndpointDemux`] + per-connection [`Connection<P>`]
//! state machines. New peers classified as `NewInitial` get a freshly
//! constructed connection + a push notification on the accept channel the
//! caller drains independently.
//!
//! # Driver pattern
//!
//! [`Listener<P>`] never touches a socket, a clock, or a timer — it
//! implements [`proxima_listen::stream::DatagramProtocol`], the sans-IO seam
//! [`proxima_listen::stream::DatagramProtocolListenProtocol`] drives: fed
//! `now` and borrowed inbound bytes via [`DatagramProtocol::on_datagram`],
//! it fills a caller-owned buffer with outbound bytes via
//! [`DatagramProtocol::poll_transmit`], and [`DatagramProtocol::on_timeout`]
//! fires exactly at [`DatagramProtocol::next_deadline`] — the QUIC-handshake-
//! retransmit shape (PTO firing with no inbound datagram) that seam exists
//! for. [`Listener::listen_protocol`] is the single reference point that
//! wires this state machine onto that driver for a real `serve()` call.
//!
//! ```ignore
//! let (protocol, mut accept_rx) = Listener::listen_protocol("quic", accept_fn);
//! // protocol implements proxima_listen::ListenProtocol; register it with
//! // a serve() call the way any other ListenProtocol is registered.
//! while let Some(handle) = accept_rx.next().await {
//!     // per-connection setup (e.g. open H3 control stream).
//! }
//! ```
//!
//! # The generic trait discards information; the concrete type doesn't
//!
//! [`DatagramProtocol::on_datagram`]'s fixed signature (`Result<(), Err>`)
//! is right-sized for the generic, connectionless driver — it has no
//! per-connection state of its own to report back. `Listener<P>` DOES:
//! which handle a datagram routed to, whether it was newly accepted, and
//! whether `Connection::handle_datagram` surfaced a genuine per-connection
//! protocol error. Rather than stash that on `Listener<P>` behind a
//! side-channel drain method the generic caller doesn't need, the CONCRETE
//! [`Listener::ingest_datagram`] returns it directly as [`DatagramIngest`];
//! `on_datagram` calls it and discards the rich part. A caller that wants
//! the rich return (H3, layering its own per-connection driving and error
//! escalation on top) calls `ingest_datagram` directly instead of going
//! through the trait.

use std::collections::BTreeMap;
use std::io;
use std::net::SocketAddr;
use std::sync::Arc;

use futures::channel::mpsc;
use proxima_listen::stream::{DatagramProtocol, DatagramProtocolListenProtocol};
use proxima_protocols::quic::connection::{Connection, ConnectionError, ConnectionState, DatagramWrite, TimerOutcome};
use proxima_protocols::quic::endpoint::{
    ConnectionHandle, ConnectionIdBytes, DatagramClassification, DropReason, EndpointDemux,
};
use proxima_protocols::quic::packet::header::Header;
use proxima_protocols::quic::time::Instant;
use proxima_protocols::quic::tls::TlsProvider;

/// QUIC version 1 (RFC 9000), big-endian, for the supported-versions list
/// of an outbound Version Negotiation packet.
const QUIC_V1_VERSION: [u8; 4] = [0x00, 0x00, 0x00, 0x01];

/// Listener-level errors.
#[derive(Debug)]
#[non_exhaustive]
pub enum ListenerError {
    Io(io::Error),
    Connection(ConnectionError),
    /// Per-path/per-connection table at capacity; new Initial dropped.
    AcceptTableFull,
}

impl core::fmt::Display for ListenerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "io: {err}"),
            Self::Connection(err) => write!(f, "connection: {err:?}"),
            Self::AcceptTableFull => f.write_str("accept table at capacity"),
        }
    }
}

impl std::error::Error for ListenerError {}

impl From<io::Error> for ListenerError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<ConnectionError> for ListenerError {
    fn from(err: ConnectionError) -> Self {
        Self::Connection(err)
    }
}

/// Per-connection slot.
struct ConnEntry<P: TlsProvider> {
    connection: Connection<P>,
    peer: SocketAddr,
    /// The server-generated local SCID registered in the demux
    /// table. Retained so `remove_connection` can call
    /// `demux.unregister` — without it, normal connection turnover
    /// fills the fixed demux table and prevents new accepts.
    local_scid: [u8; LOCAL_SCID_LEN],
    /// The client's ORIGINAL DCID (0..=20 bytes per RFC 9000 §17.2 —
    /// quinn uses the full 20), also registered in the demux at accept
    /// time. Until the client receives our SCID it addresses every
    /// Initial — the CRYPTO-fragmented ClientHello continuations AND
    /// retransmits — to its OWN chosen DCID; without this second
    /// registration those fragments classify as a fresh `NewInitial`
    /// each time and spawn a phantom half-handshake connection that
    /// never completes. `None` once freed after the connection reaches
    /// Established (the client now addresses us by `local_scid` — RFC
    /// 9000 §7.2 — so the ODCID route is dead weight in the bounded
    /// demux table) or if the client's DCID was too long to register
    /// (matches the demux's own reject).
    original_dcid: Option<ConnectionIdBytes>,
}

/// Per-connection accept policy — caller supplies this when binding
/// so the Listener knows how to construct a fresh `Connection<P>`
/// from a `NewInitial` classification.
pub type AcceptFn<P> = Arc<dyn Fn(&[u8], &[u8], &[u8], Instant) -> Result<Connection<P>, ConnectionError> + Send + Sync>;

/// Multi-connection, I/O-free QUIC server listener — a
/// [`DatagramProtocol`] state machine driven by
/// [`DatagramProtocolListenProtocol`]. See [`Listener::listen_protocol`]
/// for the single reference point that wires it onto that driver.
pub struct Listener<P: TlsProvider> {
    demux: EndpointDemux,
    connections: BTreeMap<u32, ConnEntry<P>>,
    next_handle: u32,
    /// Live handle order, mirrored against `connections`, used to
    /// round-robin `poll_transmit` fairly across connections without
    /// reallocating a scan buffer on every call.
    handle_order: Vec<u32>,
    /// Index into `handle_order` (mod its length) the next
    /// `poll_transmit` call resumes draining from.
    transmit_cursor: usize,
    /// Function the listener calls when a NewInitial datagram arrives —
    /// it constructs a fresh `Connection<P>` keyed by the client's CIDs.
    accept_fn: AcceptFn<P>,
    /// Push side of the accept notification: one send per freshly
    /// accepted connection. The paired receiver is what the caller
    /// awaits to learn of new connections (replaces a pull-style
    /// `take_accepted` queue — the caller no longer needs to poll the
    /// listener itself to observe accepts).
    accept_tx: mpsc::UnboundedSender<ConnectionHandle>,
    /// Queued Version Negotiation replies (RFC 9000 §6/§17.2.1) — a
    /// peer offering a version we don't speak gets a connectionless VN
    /// packet, staged here by `on_datagram` (which cannot itself emit
    /// wire bytes — only `transmit` does) and drained FIRST by
    /// `transmit`, ahead of the round-robin connection scan, so a busy
    /// connection can never starve a version-probing peer.
    pending_vn: Vec<(Vec<u8>, SocketAddr)>,
}

/// Outcome of routing one inbound datagram through the demux and (when it
/// classifies to a connection) `Connection::handle_datagram` — the rich
/// return [`Listener::ingest_datagram`] gives a caller that needs to know
/// WHICH connection a datagram touched and WHETHER it surfaced an error,
/// neither of which the generic [`DatagramProtocol::on_datagram`] trait
/// method's fixed `Result<(), Err>` signature can carry.
#[derive(Debug)]
#[non_exhaustive]
pub enum DatagramIngest {
    /// Routed to an existing connection.
    ///
    /// `error` is `Some` only for a GENUINE per-connection protocol error
    /// — never for the RFC 9000 §10.3 packet-level noise (header parse,
    /// decrypt/AEAD failure, packet-number error, or a transient local
    /// reassembly-buffer-full condition) a listener facing the open
    /// internet sees routinely; those are silently dropped per spec and
    /// never surfaced as connection errors. When `error` names a
    /// [`ConnectionError::FlowControlError`], [`ConnectionError::ProtocolViolation`],
    /// or [`ConnectionError::Frame`] — RFC 9000's own unambiguous
    /// transport-error classification — `Listener` has ALREADY closed
    /// the connection with the matching transport `CONNECTION_CLOSE`
    /// code before returning; for every other (ambiguous / application-
    /// flavored) error the connection is left OPEN for the caller to
    /// close with whatever code its own layer's semantics call for
    /// (`Connection::close`/`close_transport` are idempotent — the
    /// FIRST close wins, so a caller MAY always attempt its own close
    /// on `Some(error)` without checking whether `Listener` already
    /// acted).
    Existing {
        handle: ConnectionHandle,
        error: Option<ConnectionError>,
    },
    /// A `NewInitial` classification accepted a fresh connection. `error`
    /// mirrors `Existing`'s — the newly-created connection's OWN first
    /// `handle_datagram` call can surface the exact same error shapes.
    Accepted {
        handle: ConnectionHandle,
        error: Option<ConnectionError>,
    },
    /// The datagram carried an unsupported QUIC version; a Version
    /// Negotiation reply was queued (drained by the next `transmit`
    /// call). No connection is involved.
    VersionNegotiated,
    /// Silently dropped per RFC 9000 §10.3 (malformed header) or a
    /// future non-exhaustive drop category. No connection is involved.
    Dropped,
}

impl<P: TlsProvider> Listener<P> {
    /// Construct an I/O-free listener. `accept_tx` is the push side of
    /// the accept-notification channel; a `NewInitial` classification
    /// sends the freshly accepted handle into it.
    #[must_use]
    pub fn new(accept_fn: AcceptFn<P>, accept_tx: mpsc::UnboundedSender<ConnectionHandle>) -> Self {
        Self {
            // Fixed 8-byte local SCIDs (see generate_local_scid below)
            // → EndpointDemux short-header dispatch goes through the
            // O(1) hash path. The prior EndpointDemux::new fell back
            // to the O(N) prefix-scan path even though every CID we
            // issue is the same length.
            demux: EndpointDemux::with_local_cid_len(
                proxima_protocols::quic::connection::SUPPORTED_VERSIONS,
                LOCAL_SCID_LEN as u8,
            ),
            connections: BTreeMap::new(),
            next_handle: 0,
            handle_order: Vec::new(),
            transmit_cursor: 0,
            accept_fn,
            accept_tx,
            pending_vn: Vec::new(),
        }
    }

    /// Build the [`DatagramProtocolListenProtocol`] driving a fresh
    /// [`Listener<P>`] per `serve()` invocation — the single reference
    /// point wiring native QUIC onto the runtime-agnostic
    /// [`DatagramProtocol`] seam. `accept_fn` is cloned into every
    /// `Listener` the returned protocol's `build` closure constructs;
    /// the returned receiver observes every connection any of those
    /// listeners accepts.
    #[must_use]
    pub fn listen_protocol(
        label: impl Into<String>,
        accept_fn: AcceptFn<P>,
    ) -> (DatagramProtocolListenProtocol<impl Fn() -> Self + Send + Sync + 'static, Self>, mpsc::UnboundedReceiver<ConnectionHandle>)
    where
        P: Send + 'static,
    {
        let (accept_tx, accept_rx) = mpsc::unbounded();
        let build = move || Self::new(Arc::clone(&accept_fn), accept_tx.clone());
        (DatagramProtocolListenProtocol::new(label, build), accept_rx)
    }

    /// Borrow the connection for `handle`.
    pub fn connection_mut(&mut self, handle: ConnectionHandle) -> Option<&mut Connection<P>> {
        self.connections
            .get_mut(&handle.0)
            .map(|entry| &mut entry.connection)
    }

    /// Borrow a snapshot of currently-live handles (cheap clone for
    /// loop iteration).
    pub fn connection_handles(&self) -> Vec<ConnectionHandle> {
        self.connections
            .keys()
            .copied()
            .map(ConnectionHandle)
            .collect()
    }

    /// Advance per-connection timers up to `now`. Returns the set of
    /// handles that transitioned to Closed (caller should remove them
    /// via `remove_connection`).
    pub fn handle_timeout(&mut self, now: Instant) -> Vec<ConnectionHandle> {
        let mut closed = Vec::new();
        for (handle, entry) in self.connections.iter_mut() {
            if let Ok(TimerOutcome::IdleClosed) | Ok(TimerOutcome::Drained) =
                entry.connection.handle_timeout(now)
            {
                closed.push(ConnectionHandle(*handle));
            }
        }
        closed
    }

    /// Drop a connection from the listener (e.g. after IdleClosed).
    /// Also unregisters its SCID from the demux table so the slot is
    /// freed for future connections.
    pub fn remove_connection(&mut self, handle: ConnectionHandle) {
        if let Some(entry) = self.connections.remove(&handle.0) {
            let _ = self.demux.unregister(&entry.local_scid);
            if let Some(odcid) = entry.original_dcid {
                let _ = self.demux.unregister(&odcid);
            }
        }
        self.handle_order.retain(|&candidate| candidate != handle.0);
    }

    /// Route one inbound datagram through the demux and (when it
    /// classifies to a connection) `Connection::handle_datagram`. Returns
    /// which connection was touched and whether a genuine per-connection
    /// protocol error surfaced — see [`DatagramIngest`]. The generic
    /// [`DatagramProtocol::on_datagram`] trait method calls this and
    /// discards the rich return; a caller that needs it (H3, to know
    /// which handle to drive and whether to layer its own error
    /// escalation) calls this directly.
    ///
    /// # Errors
    ///
    /// Only for LISTENER-level failures that happen before any
    /// connection exists to attribute a per-connection error to: the
    /// caller-supplied `accept_fn` rejecting a `NewInitial`, or the
    /// demux's accept table being full. Per-connection protocol errors
    /// are never propagated here — see [`DatagramIngest::Existing`] /
    /// [`DatagramIngest::Accepted`]'s `error` field instead.
    pub fn ingest_datagram(&mut self, now: Instant, peer: SocketAddr, datagram: &[u8]) -> Result<DatagramIngest, ListenerError> {
        let class = self.demux.classify_datagram(datagram);
        match class {
            DatagramClassification::Existing { handle, .. } => {
                let Some(entry) = self.connections.get_mut(&handle.0) else {
                    return Ok(DatagramIngest::Dropped);
                };
                // Per-connection error isolation: a bad peer never
                // propagates to the listener caller as a whole (see
                // `act_and_surface`'s doc for the act/surface split).
                let error = match entry.connection.handle_datagram(now, datagram) {
                    Ok(()) => None,
                    Err(err) => act_and_surface(&mut entry.connection, handle.0, peer, err),
                };
                // Established: the client now addresses us by our SCID
                // (RFC 9000 §7.2), so the handshake-only ODCID route is
                // dead weight in the bounded demux table. Free it once
                // to keep long-lived connections at one table entry
                // each.
                if matches!(entry.connection.state(), ConnectionState::Established(_))
                    && let Some(odcid) = entry.original_dcid.take()
                {
                    let _ = self.demux.unregister(&odcid);
                }
                Ok(DatagramIngest::Existing { handle, error })
            }
            DatagramClassification::NewInitial { dcid, scid, .. } => {
                let local_scid = generate_local_scid(dcid);
                let connection = (self.accept_fn)(dcid, scid, &local_scid, now)?;
                let handle = ConnectionHandle(self.next_handle);
                self.next_handle = self.next_handle.saturating_add(1);
                self.demux
                    .register(&local_scid, handle)
                    .map_err(|_| ListenerError::AcceptTableFull)?;
                // Route the client's ORIGINAL DCID to this same
                // connection too. Until the client receives our SCID it
                // addresses every Initial — including CRYPTO-fragmented
                // ClientHello continuations and retransmits — to its
                // own chosen DCID; without this second registration
                // fragment 2 classifies as a fresh NewInitial and spawns
                // a phantom half-ClientHello connection that never
                // completes the handshake. The client's DCID is
                // 0..=20 bytes (RFC 9000 §17.2; quinn uses the full 20)
                // — `client_dcid_for_demux` preserves the actual length
                // rather than narrowing to a fixed 8 bytes, which would
                // silently drop every non-8-byte client's ODCID.
                let original_dcid = client_dcid_for_demux(dcid);
                if let Some(ref odcid) = original_dcid {
                    let _ = self.demux.register(odcid, handle);
                }
                let mut entry = ConnEntry {
                    connection,
                    peer,
                    local_scid,
                    original_dcid,
                };
                // Same act/surface classification as an existing
                // connection's datagram — the freshly-created connection
                // is just as capable of a real protocol violation on its
                // OWN first packet (e.g. a malformed CRYPTO frame) as an
                // established one, and leaving it open silently would
                // strand a zombie connection the reap pass never catches
                // (it's healthy from `handle_timeout`'s perspective).
                let error = match entry.connection.handle_datagram(now, datagram) {
                    Ok(()) => None,
                    Err(err) => act_and_surface(&mut entry.connection, handle.0, peer, err),
                };
                self.connections.insert(handle.0, entry);
                self.handle_order.push(handle.0);
                let _ = self.accept_tx.unbounded_send(handle);
                Ok(DatagramIngest::Accepted { handle, error })
            }
            DatagramClassification::UnsupportedVersion { dcid, scid, peer_version } => {
                // RFC 9000 §6 / §17.2.1 — the peer offered a version we
                // don't speak (commonly a GREASE probe, e.g.
                // 0x?a?a?a?a). Reply with a Version Negotiation packet
                // listing v1 so it retries. The CIDs are echoed SWAPPED
                // (VN.dcid = peer's scid, VN.scid = peer's dcid).
                // Without this, version-probing clients (cloudflare
                // quiche) never learn we speak v1 and stall. `on_datagram`
                // cannot itself emit wire bytes — only `transmit` does —
                // so the encoded packet is staged in `pending_vn` and
                // drained on the next `transmit` call.
                tracing::debug!(
                    peer_version,
                    "listener: unsupported version; replying with Version Negotiation"
                );
                let mut buf = [0u8; 64];
                match build_version_negotiation(dcid, scid, &mut buf) {
                    Some(written) => self.pending_vn.push((buf[..written].to_vec(), peer)),
                    None => tracing::warn!("listener: version-negotiation encode failed"),
                }
                Ok(DatagramIngest::VersionNegotiated)
            }
            DatagramClassification::Drop {
                reason: DropReason::MalformedHeader,
            } => {
                // v1 server: silently drop unknown/malformed packets.
                Ok(DatagramIngest::Dropped)
            }
            DatagramClassification::Drop { .. } => {
                // Future drop categories — silent.
                Ok(DatagramIngest::Dropped)
            }
            _ => {
                // Future non_exhaustive variants — silent drop.
                Ok(DatagramIngest::Dropped)
            }
        }
    }
}

impl<P: TlsProvider + Send + 'static> DatagramProtocol for Listener<P> {
    type Err = ListenerError;

    // `Listener<P>` never actually awaits anything — the sans-IO
    // `Connection<P>` methods it calls are all synchronous — so every
    // method here is `async fn` wrapping the same synchronous body as
    // before; the `async` keyword alone is enough to satisfy the trait's
    // RPITIT + `Send` signature (an immediately-`Ready` future is trivially
    // `Send` when every value it closes over is `Send`, which holds here:
    // `Connection<P>: Send` requires `P: Send`, added to the impl bound).
    async fn on_datagram(&mut self, now: proxima_core::time::Instant, peer: SocketAddr, datagram: &[u8]) -> Result<(), Self::Err> {
        // Discard the rich `DatagramIngest` return — the generic,
        // connectionless driver has no per-connection state to drive off
        // the routed handle or a surfaced error; a caller that wants
        // those (H3) calls `ingest_datagram` directly instead of going
        // through this trait method.
        self.ingest_datagram(quic_instant(now), peer, datagram).map(|_ingest| ())
    }

    async fn on_timeout(&mut self, now: proxima_core::time::Instant) -> Result<(), Self::Err> {
        for handle in self.handle_timeout(quic_instant(now)) {
            self.remove_connection(handle);
        }
        Ok(())
    }

    fn next_deadline(&self) -> Option<proxima_core::time::Instant> {
        self.connections
            .values()
            .filter_map(|entry| entry.connection.next_timeout())
            .min()
            .map(core_instant)
    }

    async fn transmit(&mut self, now: proxima_core::time::Instant, buf: &mut [u8]) -> Result<Option<(usize, SocketAddr)>, Self::Err> {
        // Version Negotiation replies are connectionless and staged
        // eagerly by `on_datagram`; drain them FIRST so a version-probing
        // peer can never be starved behind a busy connection's egress.
        if let Some((bytes, peer)) = self.pending_vn.pop() {
            let len = bytes.len().min(buf.len());
            buf[..len].copy_from_slice(&bytes[..len]);
            return Ok(Some((len, peer)));
        }

        let now = quic_instant(now);
        let total = self.handle_order.len();
        for _ in 0..total {
            let Some(length) = core::num::NonZeroUsize::new(self.handle_order.len()) else {
                break;
            };
            let index = self.transmit_cursor % length.get();
            let handle = self.handle_order[index];
            match self.connections.get_mut(&handle) {
                Some(entry) => match entry.connection.poll_transmit(now, buf) {
                    Ok(Some(DatagramWrite { len, .. })) => {
                        return Ok(Some((len, entry.peer)));
                    }
                    Ok(None) => {
                        self.transmit_cursor = self.transmit_cursor.wrapping_add(1);
                    }
                    Err(err) => {
                        tracing::warn!(
                            ?err,
                            handle,
                            "listener: transmit failed; skipping connection this tick"
                        );
                        self.transmit_cursor = self.transmit_cursor.wrapping_add(1);
                    }
                },
                // Stale handle_order entry (should not happen if kept
                // in sync with `connections`); drop it defensively.
                None => {
                    self.handle_order.remove(index);
                }
            }
        }
        Ok(None)
    }
}

/// Convert the driver's runtime-agnostic clock reading into the QUIC
/// proto layer's microsecond-resolution [`Instant`]. Truncates to
/// [`u64::MAX`] microseconds on overflow — unreachable in practice (that
/// horizon is ~584 000 years).
fn quic_instant(now: proxima_core::time::Instant) -> Instant {
    let micros = now.into_monotonic().as_micros();
    Instant::from_micros(u64::try_from(micros).unwrap_or(u64::MAX))
}

/// Convert a QUIC proto layer [`Instant`] back into the driver's
/// runtime-agnostic clock type.
fn core_instant(instant: Instant) -> proxima_core::time::Instant {
    proxima_core::time::Instant::from_monotonic(core::time::Duration::from_micros(instant.as_micros()))
}

/// Act on and surface a `Connection::handle_datagram` error — the single
/// place both `ingest_datagram` call sites (an existing connection's
/// datagram, and a freshly-accepted connection's own first datagram)
/// route through, so the classification can never drift between them.
///
/// Three buckets, per RFC 9000's own error classification:
///
/// 1. **Unambiguous transport violations** (`FlowControlError` → `0x03`,
///    `ProtocolViolation` → `0x0a`, `Frame` → `0x07`, RFC 9000 §20.1) —
///    ACT: `close_transport` with the exact documented code. SURFACE: a
///    `proxima_telemetry::error!` event plus `Some(err)` in the return so
///    the caller's `DatagramIngest` carries it.
/// 2. **RFC 9000 §10.3 packet-level noise** (header parse, decrypt/AEAD
///    failure, packet-number error, or a transient local
///    reassembly-buffer-full condition) — the connection stays healthy
///    and MUST NOT be closed; this is routine background noise for a
///    listener facing the open internet, not a connection error. Neither
///    acted on nor surfaced (returns `None`) — a debug-level log is all
///    it gets.
/// 3. **Everything else** (TLS errors, capacity/`NotImplemented`,
///    version-negotiation, future `#[non_exhaustive]` variants, …) — a
///    GENUINE error, but `Listener<P>` (QUIC-only, no H3 knowledge) has
///    no RFC-defined transport code for it. SURFACE only: telemetry +
///    `Some(err)`; the connection is left OPEN for the caller's own layer
///    to close with whatever code its semantics call for. Safe to do
///    unconditionally because `Connection::close`/`close_transport` are
///    idempotent (first call wins) — a caller MAY always attempt its own
///    close on `Some(err)` without checking whether bucket 1 already
///    acted; on a bucket-1 error that attempt is simply a no-op.
fn act_and_surface<P: TlsProvider>(
    connection: &mut Connection<P>,
    handle: u32,
    peer: SocketAddr,
    err: ConnectionError,
) -> Option<ConnectionError> {
    match &err {
        ConnectionError::FlowControlError { reason } => {
            let _ = connection.close_transport(0x03, 0x08, reason.as_bytes());
            proxima_telemetry::error!(
                connection_id = handle,
                peer = %peer,
                reason = %reason,
                "listener: flow-control violation; closing connection"
            );
            Some(err)
        }
        ConnectionError::ProtocolViolation { reason } => {
            let _ = connection.close_transport(0x0a, 0, reason.as_bytes());
            proxima_telemetry::error!(
                connection_id = handle,
                peer = %peer,
                reason = %reason,
                "listener: protocol violation; closing connection"
            );
            Some(err)
        }
        ConnectionError::Frame(_) => {
            // RFC 9000 §12.4 — malformed frame after successful
            // decryption is FRAME_ENCODING_ERROR (0x07).
            let _ = connection.close_transport(0x07, 0, b"frame encoding error");
            proxima_telemetry::error!(
                connection_id = handle,
                peer = %peer,
                ?err,
                "listener: frame encoding error; closing connection"
            );
            Some(err)
        }
        ConnectionError::Header(_)
        | ConnectionError::PacketProtection(_)
        | ConnectionError::Aead(_)
        | ConnectionError::PacketNumber(_)
        | ConnectionError::TransientRecvBufferFull { .. } => {
            tracing::debug!(?err, handle, "listener: packet-level error; dropping packet");
            None
        }
        _ => {
            proxima_telemetry::error!(
                connection_id = handle,
                peer = %peer,
                ?err,
                "listener: connection-level error surfaced; no listener-level transport code \
                 applies, connection left open for the caller's own layer to close"
            );
            Some(err)
        }
    }
}

/// Per-server local SCID length in bytes — fixed so the EndpointDemux
/// classifier can dispatch short-header packets in O(1) via a hash
/// lookup keyed by the exact `datagram[1..1+LOCAL_SCID_LEN]` slice.
const LOCAL_SCID_LEN: usize = 8;

/// Generate a per-connection server SCID. RFC 9000 §5.3 requires this
/// be **unpredictable to anyone other than the generating endpoint** —
/// a deterministic derivation from the peer's DCID (which travels in
/// the clear) lets any on-path observer pre-compute it, enabling
/// targeted spoofing and blocking future hardening (CID rotation,
/// retry-token integrity). `SysRng` reads straight from the OS
/// entropy source; on an OS-RNG failure (which would also break TLS
/// entirely) we fall back to a thread RNG so the listener doesn't
/// have a unique panic surface, but log loud.
fn generate_local_scid(_dcid: &[u8]) -> [u8; LOCAL_SCID_LEN] {
    use rand::{RngExt, TryRng};
    let mut out = [0u8; LOCAL_SCID_LEN];
    if let Err(err) = rand::rngs::SysRng.try_fill_bytes(&mut out) {
        tracing::warn!(
            ?err,
            "proxima-quic listener SysRng failed; falling back to thread RNG"
        );
        rand::rng().fill(&mut out[..]);
    }
    out
}

/// Store the CLIENT's chosen original DCID for demux routing. The client
/// picks its own length (0..=20 bytes per RFC 9000 §17.2 — quinn uses the
/// full 20), so this preserves the ACTUAL length: narrowing to a fixed
/// `[u8; 8]` would silently drop every non-8-byte client's ODCID, so the
/// demux would never learn it and the client's fragmented/retransmitted
/// Initials couldn't route home. `None` for an over-length CID matches
/// the demux's own `register` reject.
fn client_dcid_for_demux(dcid: &[u8]) -> Option<ConnectionIdBytes> {
    let mut cid = ConnectionIdBytes::new();
    cid.try_extend_from_slice(dcid).ok().map(|()| cid)
}

/// Build a Version Negotiation packet (RFC 9000 §17.2.1) replying to a
/// peer that offered an unsupported version. The CIDs are echoed
/// SWAPPED — the VN's DCID is the peer's SCID and the VN's SCID is the
/// peer's DCID — and the supported-versions list offers QUIC v1. Returns
/// the written length, or `None` if encoding failed.
fn build_version_negotiation(peer_dcid: &[u8], peer_scid: &[u8], out: &mut [u8]) -> Option<usize> {
    Header::VersionNegotiation {
        dcid: peer_scid,
        scid: peer_dcid,
        supported_versions_raw: &QUIC_V1_VERSION,
    }
    .encode(out)
    .ok()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
