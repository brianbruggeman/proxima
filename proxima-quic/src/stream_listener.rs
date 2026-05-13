//! QUIC stream listener. Each accepted QUIC connection collapses to
//! a single bidirectional stream so it implements `StreamListener` —
//! useful for non-HTTP protocols that want QUIC's transport
//! properties (encryption, 0-RTT, migration) without h3 framing.
//!
//! For HTTP/3, use [`crate::listeners::h3`] (rides on the full QUIC
//! multiplexer at [`crate::quic`]). These two are sibling concerns:
//! stream-per-connection vs full-multiplexer-per-connection.
//!
//! TLS is mandatory — pass a pre-built `quinn::ServerConfig`.

use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use quinn::{Endpoint, RecvStream, SendStream, ServerConfig};
use tokio_util::compat::{Compat, TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use proxima_primitives::stream::{BindAddr, PeerInfo, StreamConnection, StreamListener};

pub struct QuicStreamConnection {
    send: Compat<SendStream>,
    recv: Compat<RecvStream>,
    peer: Option<SocketAddr>,
}

impl QuicStreamConnection {
    fn new(send: SendStream, recv: RecvStream, peer: Option<SocketAddr>) -> Self {
        Self {
            send: send.compat_write(),
            recv: recv.compat(),
            peer,
        }
    }
}

impl futures::io::AsyncRead for QuicStreamConnection {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.get_mut().recv).poll_read(cx, buf)
    }
}

impl futures::io::AsyncWrite for QuicStreamConnection {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.get_mut().send).poll_write(cx, buf)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().send).poll_flush(cx)
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().send).poll_close(cx)
    }
}

impl StreamConnection for QuicStreamConnection {
    fn peer(&self) -> Option<PeerInfo> {
        self.peer.map(PeerInfo::Tcp)
    }
}

type QuicAcceptFut =
    Pin<Box<dyn std::future::Future<Output = io::Result<QuicStreamConnection>> + Send>>;

/// QUIC listener. One bidirectional stream per connection
/// (HTTP/3-style request/reply); multi-stream is not supported.
pub struct QuicListener {
    endpoint: Endpoint,
    local_addr: Option<SocketAddr>,
    // WHY Mutex here / WHY NOT removable / WHY right: same pattern
    // as `TokioTcpUpstream::in_flight` (`src/upstreams/tokio_stream.rs`)
    // — interior mutability for a poll-resumable future, &self trait
    // API, future not movable through atomics, RefCell would force
    // !Send. Per-listener (not per-connection), uncontested between
    // accept polls.
    in_flight: Mutex<Option<QuicAcceptFut>>,
}

impl QuicListener {
    pub fn bind(addr: SocketAddr, server_config: ServerConfig) -> io::Result<Self> {
        let endpoint = Endpoint::server(server_config, addr)?;
        let local_addr = endpoint.local_addr().ok();
        Ok(Self {
            endpoint,
            local_addr,
            in_flight: Mutex::new(None),
        })
    }
}

impl StreamListener for QuicListener {
    type Conn = QuicStreamConnection;

    fn poll_accept(&self, cx: &mut Context<'_>) -> Poll<io::Result<Self::Conn>> {
        let Ok(mut slot) = self.in_flight.lock() else {
            return Poll::Ready(Err(io::Error::other("quic in-flight lock poisoned")));
        };
        let endpoint = self.endpoint.clone();
        let future = slot.get_or_insert_with(|| {
            Box::pin(async move {
                let connecting = endpoint
                    .accept()
                    .await
                    .ok_or_else(|| io::Error::other("quic endpoint closed"))?;
                let connection = connecting
                    .await
                    .map_err(|err| io::Error::other(format!("quic handshake: {err}")))?;
                let peer = connection.remote_address();
                let (send, recv) = connection
                    .accept_bi()
                    .await
                    .map_err(|err| io::Error::other(format!("quic accept_bi: {err}")))?;
                Ok(QuicStreamConnection::new(send, recv, Some(peer)))
            })
        });
        match future.as_mut().poll(cx) {
            Poll::Ready(result) => {
                *slot = None;
                Poll::Ready(result)
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn local_addr(&self) -> Option<BindAddr> {
        self.local_addr.map(BindAddr::Tcp)
    }
}

/// Build a self-signed `ServerConfig` for tests / dev. Generates a
/// fresh certificate for `localhost`. Production should plug in real
/// certs.
pub fn dev_server_config() -> io::Result<ServerConfig> {
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
        .map_err(|err| io::Error::other(format!("rcgen: {err}")))?;
    let cert_der = cert.cert.der().clone();
    let key_der =
        quinn::rustls::pki_types::PrivateKeyDer::Pkcs8(cert.signing_key.serialize_der().into());
    let mut server_config = ServerConfig::with_single_cert(vec![cert_der], key_der)
        .map_err(|err| io::Error::other(format!("server config: {err}")))?;
    let transport = Arc::get_mut(&mut server_config.transport)
        .ok_or_else(|| io::Error::other("server config transport not unique"))?;
    transport.max_concurrent_uni_streams(0_u8.into());
    Ok(server_config)
}
