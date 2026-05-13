//! Multi-connection QUIC server listener over the sans-IO proto.
//!
//! Single UDP socket + DCID-keyed `EndpointDemux` + per-connection
//! [`Connection<P>`] state machines. New peers classified as
//! `NewInitial` get a freshly-constructed connection + handle the
//! caller can drive independently.
//!
//! # Driver pattern
//!
//! The Listener owns the socket — connections cannot move out. The
//! consumer's loop is:
//!
//! ```ignore
//! listener.poll_drive(cx, now).await;
//! while let Some(handle) = listener.take_accepted() {
//!     // per-connection setup (e.g. open H3 control stream).
//! }
//! for handle in listener.connection_handles() {
//!     // per-connection app-level work via listener.connection_mut(handle).
//! }
//! ```

use std::collections::{BTreeMap, VecDeque};
use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::task::{Context, Poll};

use prime::os::net::UdpSocket;
use proxima_protocols::quic::connection::{Connection, ConnectionError, DatagramWrite, TimerOutcome};
use proxima_protocols::quic::endpoint::{
    ConnectionHandle, DatagramClassification, DropReason, EndpointDemux, MAX_UDP_PAYLOAD_SIZE,
};
use proxima_protocols::quic::time::Instant;
use proxima_protocols::quic::tls::TlsProvider;

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
}

/// Per-connection accept policy — caller supplies this when binding
/// so the Listener knows how to construct a fresh `Connection<P>`
/// from a `NewInitial` classification.
pub type AcceptFn<P> = std::sync::Arc<
    dyn Fn(&[u8], &[u8], &[u8], Instant) -> Result<Connection<P>, ConnectionError> + Send + Sync,
>;

/// Multi-connection QUIC server listener.
pub struct Listener<P: TlsProvider> {
    socket: UdpSocket,
    demux: EndpointDemux,
    connections: BTreeMap<u32, ConnEntry<P>>,
    next_handle: u32,
    /// Handles that haven't been observed via `take_accepted` yet.
    pending_accepted: VecDeque<ConnectionHandle>,
    /// Function the listener calls when a NewInitial datagram arrives —
    /// it constructs a fresh `Connection<P>` keyed by the client's CIDs.
    accept_fn: AcceptFn<P>,
    /// Per-connection outbound queue (PN bytes + peer addr). Drained
    /// in poll_drive.
    out_queue: VecDeque<(usize, SocketAddr, Vec<u8>)>,
    scratch: Vec<u8>,
    /// Reusable inbound-datagram buffer, sized to the advertised
    /// `max_udp_payload_size` so a client filling its allowance is never
    /// truncated. Separate from `scratch` (outbound staging, transmit-MTU
    /// bound); same source-of-truth const as the advertised transport
    /// parameter.
    recv_buf: Vec<u8>,
}

impl<P: TlsProvider> Listener<P> {
    /// Bind a non-blocking UDP socket on the proxima worker thread.
    ///
    /// # Errors
    ///
    /// Bubbles up [`std::io::Error`] from `bind(2)`.
    pub fn bind(bind: SocketAddr, accept_fn: AcceptFn<P>) -> Result<Self, ListenerError> {
        let socket = UdpSocket::bind(bind)?;
        Ok(Self {
            // Fixed 8-byte local SCIDs (see generate_local_scid below)
            // → EndpointDemux short-header dispatch goes through the
            // O(1) hash path. The prior EndpointDemux::new fell back
            // to the O(N) prefix-scan path even though every CID we
            // issue is the same length.
            demux: EndpointDemux::with_local_cid_len(
                proxima_protocols::quic::connection::SUPPORTED_VERSIONS,
                LOCAL_SCID_LEN as u8,
            ),
            socket,
            connections: BTreeMap::new(),
            next_handle: 0,
            pending_accepted: VecDeque::new(),
            accept_fn,
            out_queue: VecDeque::new(),
            scratch: vec![0u8; 2048],
            recv_buf: vec![0u8; MAX_UDP_PAYLOAD_SIZE],
        })
    }

    /// Local bind address.
    ///
    /// # Errors
    ///
    /// Bubbles `getsockname(2)` failures.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.socket.local_addr()
    }

    /// Drain the next freshly-accepted connection handle, if any.
    pub fn take_accepted(&mut self) -> Option<ConnectionHandle> {
        self.pending_accepted.pop_front()
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

    /// Drive one I/O step:
    /// 1. Read one inbound datagram (if any). Classify + dispatch.
    /// 2. Drain pending outbound datagrams from every connection.
    ///
    /// Returns `Poll::Pending` when no I/O is ready and waker is
    /// registered.
    ///
    /// # Errors
    ///
    /// See [`ListenerError`].
    pub fn poll_drive(
        &mut self,
        cx: &mut Context<'_>,
        now: Instant,
    ) -> Poll<Result<(), ListenerError>> {
        // 1) Try one inbound datagram.
        match Pin::new(&mut self.socket).poll_recv_from(cx, &mut self.recv_buf) {
            Poll::Ready(Ok((len, peer))) => {
                // take the filled buffer out so `handle_inbound(&mut self, ..)`
                // doesn't conflict with borrowing `recv_buf`; restore after so
                // the allocation is reused (no per-datagram alloc).
                let buf = core::mem::take(&mut self.recv_buf);
                let result = self.handle_inbound(&buf[..len], peer, now);
                self.recv_buf = buf;
                result?;
            }
            Poll::Ready(Err(err)) => return Poll::Ready(Err(err.into())),
            Poll::Pending => {
                // No inbound — fall through to draining outbound.
            }
        }
        // 2) Drain outbound from every connection.
        // Iterate by handle to avoid borrow-conflict with the
        // socket-send below.
        let handles: Vec<u32> = self.connections.keys().copied().collect();
        for handle in handles {
            while let Some(entry) = self.connections.get_mut(&handle) {
                match entry.connection.poll_transmit(now, &mut self.scratch) {
                    Ok(Some(DatagramWrite { len, .. })) => {
                        let peer = entry.peer;
                        self.out_queue
                            .push_back((len, peer, self.scratch[..len].to_vec()));
                    }
                    Ok(None) => break,
                    Err(err) => return Poll::Ready(Err(err.into())),
                }
            }
        }
        // 3) Flush outbound queue.
        while let Some((len, peer, bytes)) = self.out_queue.pop_front() {
            match Pin::new(&mut self.socket).poll_send_to(cx, &bytes[..len], peer) {
                Poll::Ready(Ok(_)) => {}
                Poll::Pending => {
                    // Re-queue at the front so we retry next call.
                    self.out_queue.push_front((len, peer, bytes));
                    return Poll::Pending;
                }
                Poll::Ready(Err(err)) => return Poll::Ready(Err(err.into())),
            }
        }
        Poll::Pending
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
        }
    }

    fn handle_inbound(
        &mut self,
        datagram: &[u8],
        peer: SocketAddr,
        now: Instant,
    ) -> Result<(), ListenerError> {
        let class = self.demux.classify_datagram(datagram);
        match class {
            DatagramClassification::Existing { handle, .. } => {
                // Per-connection error isolation: don't propagate to
                // the listener caller. BUT distinguish transport
                // violations that require CONNECTION_CLOSE from
                // packet-level discard (header parse, decrypt fail).
                if let Some(entry) = self.connections.get_mut(&handle.0)
                    && let Err(err) = entry.connection.handle_datagram(now, datagram)
                {
                    use proxima_protocols::quic::connection::ConnectionError;
                    match &err {
                        ConnectionError::FlowControlError { reason } => {
                            let _ = entry
                                .connection
                                .close_transport(0x03, 0x08, reason.as_bytes());
                            tracing::warn!(
                                ?err,
                                handle = handle.0,
                                "listener: flow-control violation; closing connection"
                            );
                        }
                        ConnectionError::ProtocolViolation { reason } => {
                            let _ = entry.connection.close_transport(0x0a, 0, reason.as_bytes());
                            tracing::warn!(
                                ?err,
                                handle = handle.0,
                                "listener: protocol violation; closing connection"
                            );
                        }
                        ConnectionError::Frame(_) => {
                            // RFC 9000 §12.4 — malformed frame after
                            // successful decryption is
                            // FRAME_ENCODING_ERROR (0x07).
                            let _ =
                                entry
                                    .connection
                                    .close_transport(0x07, 0, b"frame encoding error");
                            tracing::warn!(
                                ?err,
                                handle = handle.0,
                                "listener: frame encoding error; closing connection"
                            );
                        }
                        // Header parse, decrypt, AEAD, PN,
                        // TransientRecvBufferFull — silent discard
                        // per RFC 9000 §10.3.
                        _ => {
                            tracing::debug!(
                                ?err,
                                handle = handle.0,
                                "listener: packet-level error; dropping packet"
                            );
                        }
                    }
                }
            }
            DatagramClassification::NewInitial { dcid, scid, .. } => {
                let local_scid = generate_local_scid(dcid);
                let connection = (self.accept_fn)(dcid, scid, &local_scid, now)?;
                let handle = ConnectionHandle(self.next_handle);
                self.next_handle = self.next_handle.saturating_add(1);
                self.demux
                    .register(&local_scid, handle)
                    .map_err(|_| ListenerError::AcceptTableFull)?;
                let mut entry = ConnEntry {
                    connection,
                    peer,
                    local_scid,
                };
                if let Err(err) = entry.connection.handle_datagram(now, datagram) {
                    tracing::debug!(
                        ?err,
                        handle = handle.0,
                        "listener first datagram error; dropping packet"
                    );
                }
                self.connections.insert(handle.0, entry);
                self.pending_accepted.push_back(handle);
            }
            DatagramClassification::UnsupportedVersion { .. }
            | DatagramClassification::Drop {
                reason: DropReason::MalformedHeader,
            } => {
                // v1 server: silently drop unknown/malformed packets.
            }
            DatagramClassification::Drop { .. } => {
                // Future drop categories — silent.
            }
            _ => {
                // Future non_exhaustive variants — silent drop.
            }
        }
        Ok(())
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn listener_error_displays_io_variant() {
        let err = ListenerError::Io(io::Error::other("oops"));
        let formatted = format!("{err}");
        assert!(formatted.contains("io:"));
    }
}
