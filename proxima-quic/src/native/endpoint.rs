//! Runtime-agnostic Endpoint over `proxima_protocols::quic`.
//!
//! Holds the UDP socket + a `proxima_protocols::quic::Connection<P>`
//! state machine; drives them via `poll_*` methods so the caller's
//! executor decides how to schedule.
//!
//! This is the C29 minimal-viable shape — a single-connection
//! endpoint (client OR one inbound server connection) suitable for the
//! integration smoke test. The full DCID-demux multi-connection
//! endpoint with `accept` futures lands as B1.1 once consumers need it.

use std::net::SocketAddr;
use std::pin::Pin;
use std::task::{Context, Poll};

use arrayvec::ArrayVec;
use prime::os::net::UdpSocket;
use proxima_core::datagram_batch::DatagramBatch;
use proxima_protocols::quic::connection::{Connection, ConnectionError, DatagramWrite, TimerOutcome};
use proxima_protocols::quic::endpoint::MAX_UDP_PAYLOAD_SIZE;
use proxima_protocols::quic::time::Instant;
use proxima_protocols::quic::tls::TlsProvider;

use super::config::EndpointConfig;

/// `recvmmsg` batch sized for QUIC: each recv slot holds one whole UDP datagram
/// up to the advertised [`MAX_UDP_PAYLOAD_SIZE`] (RFC 9000 §18.2), so a peer
/// filling its allowance is never truncated. Only the recv-slot byte size is
/// QUIC-specific; the slab depth and send-arena params stay at `proxima-core`'s
/// generic batch defaults (the send arena is bounded by our transmit MTU, a
/// separate axis). Same source-of-truth const as the park-path recv buffer and
/// the advertised transport parameter — they cannot drift.
type QuicDatagramBatch = DatagramBatch<
    MAX_UDP_PAYLOAD_SIZE,
    { proxima_core::sized::BATCH_RECV_INITIAL_CAP },
    { proxima_core::sized::BATCH_SEND_ARENA_INITIAL_BYTES },
    { proxima_core::sized::BATCH_SEND_SPAN_CAP },
>;

/// Max datagrams staged per `poll_send_batch` send burst. Bounded by
/// the stack iovec array the `sendmmsg` implementation allocates per call;
/// 64 comfortably covers 32 in-flight streams × up to 2 datagrams each.
const CLIENT_SEND_BATCH_CAP: usize = 64;

/// Max datagrams drained per `poll_recv_batch` recv burst — matches
/// `DefaultDatagramBatch`'s initial slot count so the slab never resizes
/// during a normal response burst.
const CLIENT_RECV_BATCH_CAP: usize = 32;

/// Endpoint errors.
#[derive(Debug)]
#[non_exhaustive]
pub enum EndpointError {
    /// I/O error from the underlying UDP socket.
    Io(std::io::Error),
    /// Sans-IO state machine error from
    /// [`proxima_protocols::quic::connection::Connection`].
    Connection(ConnectionError),
    /// Endpoint configured without the side needed for the requested
    /// operation (e.g. `connect` on a server-only endpoint).
    UnconfiguredSide,
}

impl core::fmt::Display for EndpointError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "io: {err}"),
            Self::Connection(err) => write!(f, "connection: {err:?}"),
            Self::UnconfiguredSide => f.write_str("endpoint missing config for this operation"),
        }
    }
}

impl std::error::Error for EndpointError {}

impl From<std::io::Error> for EndpointError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<ConnectionError> for EndpointError {
    fn from(err: ConnectionError) -> Self {
        Self::Connection(err)
    }
}

/// Single-connection runtime-agnostic endpoint over the sans-IO proto.
///
/// Drive the connection by alternating `poll_send` (drains pending
/// outbound datagrams from the proto state machine to the socket) and
/// `poll_recv` (drains inbound datagrams from the socket into the
/// proto state machine). Both are `Poll`-typed so any executor with
/// a `Future` shape can compose them.
pub struct Endpoint<P: TlsProvider> {
    config: EndpointConfig,
    socket: UdpSocket,
    /// Boxed: `Connection<P>` is large (≈160 KB with the rustls provider),
    /// so keeping it inline would (a) overflow the small per-core worker
    /// stack on every move through construction and (b) bloat every type
    /// that holds an `Endpoint`. The box keeps moves pointer-sized.
    connection: Box<Connection<P>>,
    peer: Option<SocketAddr>,
    scratch: Vec<u8>,
    /// Reusable inbound-datagram buffer for the single-recv park path. Sized to
    /// the advertised `max_udp_payload_size` so a peer that fills its allowance
    /// (e.g. a server on a 64K-MTU loopback) is never truncated — a short read
    /// would mangle the AEAD tag and kill the connection mid-stream.
    recv_buf: Vec<u8>,
    /// Set when a build-side datagram is ready to write but the socket
    /// returned WouldBlock. Re-sent on the next `poll_send`.
    pending_out: Option<(usize, SocketAddr)>,
    /// Reusable datagram I/O batch buffers — pre-allocated once, reused every
    /// `poll_send_batch`/`poll_recv_batch` call so the hot path allocates
    /// nothing. `recv` holds the iovec-addressable recv slots (`recvmmsg`
    /// writes straight in); `send` is the contiguous staging arena for
    /// `sendmmsg`. Both Vecs grow to the high-water mark then never reallocate.
    batch: QuicDatagramBatch,
}

impl<P: TlsProvider> Endpoint<P> {
    /// Bind a non-blocking UDP socket on the proxima worker thread and
    /// wrap the given proto-side connection. The proto connection is
    /// constructed by the caller (it owns the TLS provider choice +
    /// initial CID generation).
    ///
    /// # Errors
    ///
    /// Returns [`EndpointError::Io`] if the UDP bind fails.
    pub fn new(config: EndpointConfig, connection: Connection<P>) -> Result<Self, EndpointError> {
        let socket = UdpSocket::bind(config.bind)?;
        Ok(Self {
            config,
            socket,
            connection: Box::new(connection),
            peer: None,
            scratch: vec![0u8; 1500],
            recv_buf: vec![0u8; proxima_protocols::quic::endpoint::MAX_UDP_PAYLOAD_SIZE],
            pending_out: None,
            batch: QuicDatagramBatch::new(),
        })
    }

    /// Set the destination address for outbound datagrams (client-side:
    /// the server's `SocketAddr`).
    pub fn set_peer(&mut self, peer: SocketAddr) {
        self.peer = Some(peer);
    }

    /// Borrow the loaded config (for diagnostics + introspection).
    #[must_use]
    pub fn config(&self) -> &EndpointConfig {
        &self.config
    }

    /// Borrow the inner [`Connection`] for state introspection +
    /// stream operations (open_stream, send_application, etc.).
    pub fn connection(&self) -> &Connection<P> {
        &self.connection
    }

    /// Mutably borrow the connection.
    pub fn connection_mut(&mut self) -> &mut Connection<P> {
        &mut self.connection
    }

    /// Local bind address (post-bind, resolves ephemeral port).
    ///
    /// # Errors
    ///
    /// Bubbles up [`std::io::Error`] from `getsockname(2)`.
    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.socket.local_addr()
    }

    /// Drive at most one outbound datagram from the proto state machine
    /// to the UDP socket.
    ///
    /// Returns:
    /// - `Poll::Ready(Ok(true))` — one datagram successfully sent.
    /// - `Poll::Ready(Ok(false))` — proto has nothing to send.
    /// - `Poll::Pending` — socket would block; waker registered.
    /// - `Poll::Ready(Err(_))` — I/O or proto error.
    ///
    /// # Errors
    ///
    /// See [`EndpointError`].
    pub fn poll_send(
        &mut self,
        cx: &mut Context<'_>,
        now: Instant,
    ) -> Poll<Result<bool, EndpointError>> {
        // Drain any pending-from-last-call datagram first.
        if let Some((len, peer)) = self.pending_out.take() {
            match Pin::new(&mut self.socket).poll_send_to(cx, &self.scratch[..len], peer) {
                Poll::Ready(Ok(_)) => return Poll::Ready(Ok(true)),
                Poll::Pending => {
                    self.pending_out = Some((len, peer));
                    return Poll::Pending;
                }
                Poll::Ready(Err(err)) => return Poll::Ready(Err(err.into())),
            }
        }
        let Some(peer) = self.peer else {
            return Poll::Ready(Ok(false));
        };
        match self.connection.poll_transmit(now, &mut self.scratch) {
            Ok(Some(DatagramWrite { len, .. })) => {
                match Pin::new(&mut self.socket).poll_send_to(cx, &self.scratch[..len], peer) {
                    Poll::Ready(Ok(_)) => Poll::Ready(Ok(true)),
                    Poll::Pending => {
                        self.pending_out = Some((len, peer));
                        Poll::Pending
                    }
                    Poll::Ready(Err(err)) => Poll::Ready(Err(err.into())),
                }
            }
            Ok(None) => Poll::Ready(Ok(false)),
            Err(err) => Poll::Ready(Err(err.into())),
        }
    }

    /// Drive at most one inbound datagram from the UDP socket into the
    /// proto state machine.
    ///
    /// Returns:
    /// - `Poll::Ready(Ok(true))` — one datagram processed.
    /// - `Poll::Pending` — socket has no datagram; waker registered.
    /// - `Poll::Ready(Err(_))` — I/O or proto error.
    ///
    /// # Errors
    ///
    /// See [`EndpointError`].
    pub fn poll_recv(
        &mut self,
        cx: &mut Context<'_>,
        now: Instant,
    ) -> Poll<Result<bool, EndpointError>> {
        match Pin::new(&mut self.socket).poll_recv_from(cx, &mut self.recv_buf) {
            Poll::Ready(Ok((len, peer))) => {
                if self.peer.is_none() {
                    self.peer = Some(peer);
                }
                self.connection
                    .handle_datagram(now, &self.recv_buf[..len])?;
                Poll::Ready(Ok(true))
            }
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(err)) => Poll::Ready(Err(err.into())),
        }
    }

    /// Drain all pending outbound datagrams from the proto state machine
    /// into a staging arena and ship them in ONE `sendmmsg` call (Linux) or
    /// the fewest possible `sendto` calls (non-Linux fallback). Replaces the
    /// `loop { poll_send }` pattern: instead of N separate `sendto` syscalls
    /// for N pending datagrams, this stages the whole burst then ships it
    /// atomically.
    ///
    /// Returns the number of datagrams sent. `Poll::Pending` when the kernel
    /// send buffer is full and nothing was sent (waker registered); the caller
    /// should await readiness and retry. A partial send (some but not all
    /// staged) returns `Ready(Ok(sent))` — QUIC loss recovery retransmits the
    /// unsent ones on the next PTO timeout.
    ///
    /// # Errors
    ///
    /// See [`EndpointError`].
    pub fn poll_send_batch(
        &mut self,
        cx: &mut Context<'_>,
        now: Instant,
    ) -> Poll<Result<usize, EndpointError>> {
        let peer = match self.peer {
            Some(peer) => peer,
            None => return Poll::Ready(Ok(0)),
        };
        // Stage all pending datagrams: one poll_transmit per datagram (each
        // writes into self.scratch), copied into the send arena (one small
        // memcpy per packet — the cost of a contiguous sendmmsg layout).
        self.batch.send.reset();
        loop {
            match self.connection.poll_transmit(now, &mut self.scratch) {
                Ok(Some(DatagramWrite { len, .. })) => {
                    // arena full → drop this datagram; QUIC loss recovery
                    // retransmits it on the next PTO. Log-and-drop is safe here.
                    let _ = self.batch.send.try_append(&self.scratch[..len], peer);
                }
                Ok(None) => break,
                Err(err) => return Poll::Ready(Err(err.into())),
            }
        }
        let staged = self.batch.send.len();
        if staged == 0 {
            return Poll::Ready(Ok(0));
        }
        // Build the iov slice from the staged spans and ship in one call.
        // Field-splitting: iov borrows self.batch (via spans), the send call
        // borrows self.socket — different fields, both borrows are valid.
        let spans = self.batch.send.spans();
        let iov: ArrayVec<(&[u8], SocketAddr), CLIENT_SEND_BATCH_CAP> = spans
            .iter()
            .take(CLIENT_SEND_BATCH_CAP)
            .map(|span| (self.batch.send.slice_for(span), span.peer))
            .collect();
        match Pin::new(&mut self.socket).poll_send_batch(cx, iov.as_slice()) {
            Poll::Ready(Ok(sent)) => Poll::Ready(Ok(sent)),
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(err)) => Poll::Ready(Err(err.into())),
        }
    }

    /// Drain a burst of inbound datagrams in ONE `recvmmsg` syscall (Linux)
    /// or the minimum `recvfrom` calls (non-Linux fallback) and feed each
    /// through the proto state machine. Replaces the `now_or_never(poll_recv)`
    /// drain loop: instead of creating N `poll_fn` closures and calling
    /// `poll_recv_from` N times, this reads up to [`CLIENT_RECV_BATCH_CAP`]
    /// datagrams in a single call then feeds them all before returning.
    ///
    /// Returns the number of datagrams processed. `Poll::Pending` when the
    /// socket is empty and the caller's waker has been registered.
    ///
    /// # Errors
    ///
    /// See [`EndpointError`].
    pub fn poll_recv_batch(
        &mut self,
        cx: &mut Context<'_>,
        now: Instant,
    ) -> Poll<Result<usize, EndpointError>> {
        self.batch.recv.clear();
        let _ = self.batch.recv.ensure_capacity(CLIENT_RECV_BATCH_CAP);
        let received = {
            // Field-splitting: bufs/meta borrow self.batch.recv; the socket
            // call borrows self.socket — disjoint fields, both borrows valid.
            let (slots, meta) = self.batch.recv.unfilled_mut();
            let batch_size = slots.len().min(meta.len()).min(CLIENT_RECV_BATCH_CAP);
            if batch_size == 0 {
                return Poll::Ready(Ok(0));
            }
            let mut bufs: ArrayVec<&mut [u8], CLIENT_RECV_BATCH_CAP> = slots
                .iter_mut()
                .take(batch_size)
                .map(|slot| slot.as_mut_slice())
                .collect();
            match Pin::new(&mut self.socket).poll_recv_batch(
                cx,
                bufs.as_mut_slice(),
                &mut meta[..batch_size],
            ) {
                Poll::Ready(Ok(count)) => count,
                Poll::Ready(Err(err)) => return Poll::Ready(Err(err.into())),
                Poll::Pending => return Poll::Pending,
            }
        };
        self.batch.recv.commit(received);
        for view in self.batch.recv.filled_datagrams() {
            if self.peer.is_none() {
                self.peer = Some(view.peer);
            }
            if let Err(err) = self.connection.handle_datagram(now, view.bytes) {
                return Poll::Ready(Err(err.into()));
            }
        }
        Poll::Ready(Ok(received))
    }

    /// Advance the connection's timers up to `now`. Returns the
    /// resulting [`TimerOutcome`] (Continue / IdleClosed / etc.).
    ///
    /// # Errors
    ///
    /// Bubbles [`EndpointError::Connection`] from the proto state
    /// machine.
    pub fn handle_timeout(&mut self, now: Instant) -> Result<TimerOutcome, EndpointError> {
        Ok(self.connection.handle_timeout(now)?)
    }

    /// Next timer deadline from the proto state machine (PTO,
    /// close, drain, idle).
    #[must_use]
    pub fn next_timeout(&self) -> Option<Instant> {
        self.connection.next_timeout()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use core::future::poll_fn;
    use prime::os::core_shard;
    use proxima_protocols::quic::tls::mock::{MockStep, MockTlsProvider};
    use proxima_runtime::CoreId;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    #[test]
    fn endpoint_error_displays_io_variant() {
        let err = EndpointError::Io(std::io::Error::other("oops"));
        let formatted = format!("{err}");
        assert!(formatted.contains("io:"));
    }

    #[test]
    fn endpoint_error_displays_unconfigured_side() {
        let err = EndpointError::UnconfiguredSide;
        let formatted = format!("{err}");
        assert!(formatted.contains("missing"));
    }

    /// C29 worked example: bind an Endpoint on a proxima worker; drive
    /// `poll_send` to emit the first ClientHello-bearing Initial datagram
    /// over the loopback UDP socket; verify a blocking std::net::UdpSocket
    /// peer receives ≥ 1200 bytes (RFC 9000 §14.1 — Initial padding).
    #[test]
    fn endpoint_poll_send_emits_initial_to_loopback_peer() {
        let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        let done = Arc::new(AtomicBool::new(false));
        let done_for_factory = done.clone();
        let peer_chan = Arc::new(std::sync::Mutex::new(None::<SocketAddr>));
        let peer_for_factory = peer_chan.clone();

        // Bind the inbound std socket first so we have a fixed peer addr.
        let peer_socket = std::net::UdpSocket::bind("127.0.0.1:0").expect("peer bind");
        peer_socket
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("read timeout");
        let peer_addr = peer_socket.local_addr().expect("peer local_addr");

        handle
            .dispatch_factory(Box::new(move || {
                let done = done_for_factory.clone();
                let peer_handle = peer_for_factory.clone();
                Box::pin(async move {
                    let dcid = [0x83u8, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08];
                    let scid = [0xc0u8, 0xff, 0xee, 0xba, 0xbe, 0x12, 0x34, 0x56];
                    let client_hello: Vec<u8> = vec![0xDE, 0xAD, 0xBE, 0xEF];
                    let config =
                        MockTlsProvider::script_client(vec![MockStep::EmitHandshakeBytes {
                            epoch: proxima_protocols::quic::tls::Epoch::Initial,
                            bytes: client_hello,
                        }]);
                    let connection = Connection::<MockTlsProvider>::new_client(
                        config,
                        b"",
                        &dcid,
                        &scid,
                        Instant::from_micros(1_000_000),
                    )
                    .expect("new_client");
                    let endpoint_config = super::EndpointConfig {
                        bind: "127.0.0.1:0".parse().unwrap(),
                        client: Some(super::super::config::ClientConfig::default()),
                        server: None,
                    };
                    let mut endpoint = Endpoint::new(endpoint_config, connection).expect("bind");
                    let local = endpoint.local_addr().expect("local_addr");
                    *peer_handle.lock().unwrap() = Some(local);
                    endpoint.set_peer(peer_addr);
                    // Drain the first datagram.
                    let sent =
                        poll_fn(|cx| endpoint.poll_send(cx, Instant::from_micros(1_000_001))).await;
                    let sent = sent.expect("poll_send ok");
                    assert!(sent, "expected one datagram emitted");
                    done.store(true, Ordering::Release);
                }) as Pin<Box<dyn core::future::Future<Output = ()> + 'static>>
            }))
            .expect("dispatch_factory");

        // Wait for the worker to bind + report its addr.
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        loop {
            if peer_chan.lock().unwrap().is_some() {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "endpoint never bound");
            std::thread::sleep(Duration::from_millis(5));
        }

        // The peer should receive at least one Initial datagram.
        let mut buf = [0u8; 2048];
        let (len, _src) = peer_socket.recv_from(&mut buf).expect("peer recv");
        assert!(
            len >= 1200,
            "Initial datagram must be padded to ≥ 1200 bytes per RFC 9000 §14.1; got {len}"
        );

        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while !done.load(Ordering::Acquire) {
            assert!(
                std::time::Instant::now() < deadline,
                "endpoint task did not finish"
            );
            std::thread::sleep(Duration::from_millis(5));
        }

        handle.shutdown_and_join().expect("shutdown");
    }
}
