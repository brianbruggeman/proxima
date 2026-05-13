//! `tokio::net::UdpSocket`-backed Endpoint variant.
//!
//! Symmetric to [`super::endpoint::Endpoint`] but uses
//! `tokio::net::UdpSocket` as the I/O source. Behind the
//! **`tokio-compat`** feature flag (off by default per the workspace
//! tokio-free-production rule) — for consumers already on a tokio
//! runtime who want to drive the sans-IO proto from a tokio task.
//!
//! Production proxima endpoints use [`super::endpoint::Endpoint`].

use std::net::SocketAddr;
use std::task::{Context, Poll};

use proxima_protocols::quic::connection::{Connection, DatagramWrite, TimerOutcome};
use proxima_protocols::quic::time::Instant;
use proxima_protocols::quic::tls::TlsProvider;
use tokio::net::UdpSocket as TokioUdpSocket;

use super::config::EndpointConfig;
use super::endpoint::EndpointError;

/// `tokio::net::UdpSocket`-backed Endpoint.
///
/// API parity with [`super::endpoint::Endpoint`] — same `poll_send` /
/// `poll_recv` / `handle_timeout` / `next_timeout` shape. Drivable
/// from any tokio task; `tokio::main` not required (works inside
/// `tokio::task::spawn_local` too).
pub struct TokioEndpoint<P: TlsProvider> {
    config: EndpointConfig,
    socket: TokioUdpSocket,
    connection: Connection<P>,
    peer: Option<SocketAddr>,
    scratch: Vec<u8>,
    /// Reusable inbound-datagram buffer, sized to the advertised
    /// `max_udp_payload_size` so a peer filling its allowance is never
    /// truncated (a short read mangles the AEAD tag). Same source-of-truth
    /// const as the advertised transport parameter.
    recv_buf: Vec<u8>,
    pending_out: Option<(usize, SocketAddr)>,
}

impl<P: TlsProvider> TokioEndpoint<P> {
    /// Bind a tokio UDP socket and wrap the proto-side connection.
    /// Requires a tokio runtime in scope.
    ///
    /// # Errors
    ///
    /// [`EndpointError::Io`] on bind failure.
    pub async fn new(
        config: EndpointConfig,
        connection: Connection<P>,
    ) -> Result<Self, EndpointError> {
        let socket = TokioUdpSocket::bind(config.bind).await?;
        Ok(Self {
            config,
            socket,
            connection,
            peer: None,
            scratch: vec![0u8; 1500],
            recv_buf: vec![0u8; proxima_protocols::quic::endpoint::MAX_UDP_PAYLOAD_SIZE],
            pending_out: None,
        })
    }

    /// Set the destination peer.
    pub fn set_peer(&mut self, peer: SocketAddr) {
        self.peer = Some(peer);
    }

    /// Loaded config (introspection).
    #[must_use]
    pub fn config(&self) -> &EndpointConfig {
        &self.config
    }

    /// Borrow the underlying sans-IO connection.
    pub fn connection(&self) -> &Connection<P> {
        &self.connection
    }

    /// Mutably borrow the underlying sans-IO connection.
    pub fn connection_mut(&mut self) -> &mut Connection<P> {
        &mut self.connection
    }

    /// Local bind address.
    ///
    /// # Errors
    ///
    /// Bubbles `getsockname(2)` failure.
    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.socket.local_addr()
    }

    /// Drain one outbound proto datagram to the tokio UDP socket.
    ///
    /// # Errors
    ///
    /// See [`EndpointError`].
    pub fn poll_send(
        &mut self,
        cx: &mut Context<'_>,
        now: Instant,
    ) -> Poll<Result<bool, EndpointError>> {
        if let Some((len, peer)) = self.pending_out.take() {
            match self.socket.poll_send_to(cx, &self.scratch[..len], peer) {
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
                match self.socket.poll_send_to(cx, &self.scratch[..len], peer) {
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

    /// Drain one inbound datagram from the tokio UDP socket into the
    /// proto state machine.
    ///
    /// # Errors
    ///
    /// See [`EndpointError`].
    pub fn poll_recv(
        &mut self,
        cx: &mut Context<'_>,
        now: Instant,
    ) -> Poll<Result<bool, EndpointError>> {
        let mut read_buf = tokio::io::ReadBuf::new(&mut self.recv_buf);
        match self.socket.poll_recv_from(cx, &mut read_buf) {
            Poll::Ready(Ok(peer)) => {
                if self.peer.is_none() {
                    self.peer = Some(peer);
                }
                let len = read_buf.filled().len();
                self.connection
                    .handle_datagram(now, &self.recv_buf[..len])?;
                Poll::Ready(Ok(true))
            }
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(err)) => Poll::Ready(Err(err.into())),
        }
    }

    /// Advance the connection's timers.
    ///
    /// # Errors
    ///
    /// Bubbles [`EndpointError::Connection`].
    pub fn handle_timeout(&mut self, now: Instant) -> Result<TimerOutcome, EndpointError> {
        Ok(self.connection.handle_timeout(now)?)
    }

    /// Next pending timer deadline.
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
    use proxima_protocols::quic::tls::Epoch;
    use proxima_protocols::quic::tls::mock::{MockStep, MockTlsProvider};

    /// C31 tokio-compat worked example: bind a TokioEndpoint, drive
    /// poll_send once, verify a blocking std UdpSocket peer receives
    /// a ≥1200 byte Initial datagram.
    #[proxima::test(runtime = "tokio")]
    async fn tokio_endpoint_poll_send_emits_initial_to_loopback() {
        // The connection enum is large (multipath + stream tables);
        // spawn a thread with a generous stack to host the future.
        let handle = std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("rt");
                rt.block_on(async {
                    let peer_socket = std::net::UdpSocket::bind("127.0.0.1:0").expect("peer bind");
                    peer_socket
                        .set_read_timeout(Some(std::time::Duration::from_secs(2)))
                        .expect("read timeout");
                    let peer_addr = peer_socket.local_addr().expect("peer addr");

                    let dcid = [0x83u8, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08];
                    let scid = [0xc0u8, 0xff, 0xee, 0xba, 0xbe, 0x12, 0x34, 0x56];
                    let client_hello: Vec<u8> = vec![0xDE, 0xAD, 0xBE, 0xEF];
                    let config =
                        MockTlsProvider::script_client(vec![MockStep::EmitHandshakeBytes {
                            epoch: Epoch::Initial,
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
                    let endpoint_config = EndpointConfig {
                        bind: "127.0.0.1:0".parse().unwrap(),
                        client: Some(super::super::config::ClientConfig::default()),
                        server: None,
                    };
                    let mut endpoint: Box<TokioEndpoint<MockTlsProvider>> = Box::new(
                        TokioEndpoint::new(endpoint_config, connection)
                            .await
                            .expect("bind"),
                    );
                    endpoint.set_peer(peer_addr);
                    let sent =
                        poll_fn(|cx| endpoint.poll_send(cx, Instant::from_micros(1_000_001)))
                            .await
                            .expect("poll_send ok");
                    assert!(sent);

                    let mut buf = [0u8; 2048];
                    let (len, _src) = peer_socket.recv_from(&mut buf).expect("peer recv");
                    assert!(len >= 1200, "RFC 9000 §14.1 Initial padding; got {len}");
                });
            })
            .expect("spawn");
        handle.join().expect("join");
    }
}
