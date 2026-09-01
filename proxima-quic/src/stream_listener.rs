//! QUIC stream listener. Each accepted QUIC connection collapses to
//! a single bidirectional stream so it implements `StreamListener` —
//! useful for non-HTTP protocols that want QUIC's transport
//! properties (encryption, 0-RTT, migration) without h3 framing.
//!
//! For HTTP/3, use `proxima::listeners::h3`, which rides the full QUIC
//! multiplexer at [`crate::endpoint`]. These two are sibling concerns:
//! stream-per-connection vs full-multiplexer-per-connection.
//!
//! TLS is mandatory — pass a pre-built `quinn::ServerConfig`.
//! [`crate::dev_server_config`] builds a self-signed one for tests and
//! local dev.

use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;
use std::task::{Context, Poll};

use quinn::ClientConfig;
use quinn::{Endpoint, RecvStream, SendStream, ServerConfig};
use tokio_util::compat::{Compat, TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use proxima_primitives::stream::{
    BindAddr, PeerInfo, StreamConnection, StreamListener, StreamUpstream,
};

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

// boxed because `poll_connect` takes `&self`, so the asynchronous QUIC
// connect/open-bi sequence must live across polls in the same way as the
// listener's accept sequence below.
type QuicConnectFut = Pin<
    Box<
        dyn std::future::Future<Output = io::Result<(QuicStreamConnection, quinn::Connection)>>
            + Send,
    >,
>;

/// QUIC client adapter for stream protocols such as DNS-over-QUIC.
///
/// Each `poll_connect` opens one bidirectional application stream, reusing a
/// bounded pool of authenticated QUIC connections. The caller supplies a TLS
/// config whose ALPN is appropriate for its protocol (DoQ uses `doq`). The
/// returned stream uses the existing bounded protocol framing; this adapter
/// owns only endpoint, handshake, stream setup, and bounded connection reuse.
pub struct QuicUpstream {
    endpoint: Endpoint,
    server_addr: SocketAddr,
    server_name: String,
    in_flight: Mutex<Option<QuicConnectFut>>,
    connections: Mutex<Vec<quinn::Connection>>,
    max_connections: usize,
}

impl QuicUpstream {
    /// Build a QUIC client endpoint using the caller's rustls-backed QUIC
    /// configuration. No network activity occurs until `poll_connect`.
    pub fn with_client_config(
        server_addr: SocketAddr,
        server_name: impl Into<String>,
        tls_config: rustls::ClientConfig,
    ) -> io::Result<Self> {
        Self::with_client_config_and_limit(server_addr, server_name, tls_config, 1)
    }

    /// Build a client endpoint with a bounded connection pool. A limit of
    /// zero is rejected so configuration mistakes fail before any network
    /// activity.
    pub fn with_client_config_and_limit(
        server_addr: SocketAddr,
        server_name: impl Into<String>,
        tls_config: rustls::ClientConfig,
        max_connections: usize,
    ) -> io::Result<Self> {
        if max_connections == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "quic connection pool limit must be non-zero",
            ));
        }
        let local = if server_addr.is_ipv4() {
            SocketAddr::from(([0u8; 4], 0))
        } else {
            SocketAddr::from(([0u16; 8], 0))
        };
        let mut endpoint = Endpoint::client(local)?;
        let quic_tls = quinn::crypto::rustls::QuicClientConfig::try_from(tls_config)
            .map_err(|error| io::Error::other(format!("quic tls config: {error}")))?;
        endpoint.set_default_client_config(ClientConfig::new(Arc::new(quic_tls)));
        Ok(Self {
            endpoint,
            server_addr,
            server_name: server_name.into(),
            in_flight: Mutex::new(None),
            connections: Mutex::new(Vec::with_capacity(max_connections.min(4))),
            max_connections,
        })
    }

    #[cfg(test)]
    #[allow(clippy::expect_used)]
    fn pooled_connection_count(&self) -> usize {
        self.connections.lock().expect("quic pool lock").len()
    }
}

impl StreamUpstream for QuicUpstream {
    type Conn = Box<dyn StreamConnection>;

    fn poll_connect(&self, cx: &mut Context<'_>) -> Poll<io::Result<Self::Conn>> {
        let Ok(mut slot) = self.in_flight.lock() else {
            return Poll::Ready(Err(io::Error::other("quic in-flight lock poisoned")));
        };
        let endpoint = self.endpoint.clone();
        let server_addr = self.server_addr;
        let server_name = self.server_name.clone();
        let pooled = self
            .connections
            .lock()
            .ok()
            .and_then(|mut connections| connections.pop());
        let future = slot.get_or_insert_with(|| {
            Box::pin(async move {
                let connect = || async {
                    let connecting = endpoint
                        .connect(server_addr, &server_name)
                        .map_err(|error| io::Error::other(format!("quic connect: {error}")))?;
                    connecting
                        .await
                        .map_err(|error| io::Error::other(format!("quic handshake: {error}")))
                };
                let connection = if let Some(connection) = pooled {
                    connection
                } else {
                    connect().await?
                };
                let peer = connection.remote_address();
                let (send, recv) = match connection.open_bi().await {
                    Ok(stream) => stream,
                    Err(_) => {
                        let connection = connect().await?;
                        let peer = connection.remote_address();
                        let (send, recv) = connection.open_bi().await.map_err(|error| {
                            io::Error::other(format!("quic open stream: {error}"))
                        })?;
                        return Ok((
                            QuicStreamConnection::new(send, recv, Some(peer)),
                            connection,
                        ));
                    }
                };
                Ok((
                    QuicStreamConnection::new(send, recv, Some(peer)),
                    connection,
                ))
            })
        });
        match future.as_mut().poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(result) => {
                slot.take();
                Poll::Ready(result.map(|(connection, pooled)| {
                    if let Ok(mut connections) = self.connections.lock()
                        && connections.len() < self.max_connections
                    {
                        connections.push(pooled);
                    }
                    Box::new(connection) as Self::Conn
                }))
            }
        }
    }
}

// boxed because the accept sequence (accept → handshake → accept_bi) is an
// async block with no nameable type, and `poll_accept` takes `&self`, so it
// has to be stored across polls rather than held on the stack.
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

#[cfg(all(test, feature = "tokio-compat"))]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use futures::io::{AsyncReadExt, AsyncWriteExt};
    use proxima_primitives::stream::{StreamListener, StreamUpstream};
    use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
    use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
    use rustls::{DigitallySignedStruct, SignatureScheme};

    #[derive(Debug)]
    struct AcceptAnyCertificate;

    impl ServerCertVerifier for AcceptAnyCertificate {
        fn verify_server_cert(
            &self,
            _end_entity: &CertificateDer<'_>,
            _intermediates: &[CertificateDer<'_>],
            _server_name: &ServerName<'_>,
            _ocsp_response: &[u8],
            _now: UnixTime,
        ) -> Result<ServerCertVerified, rustls::Error> {
            Ok(ServerCertVerified::assertion())
        }

        fn verify_tls12_signature(
            &self,
            _message: &[u8],
            _cert: &CertificateDer<'_>,
            _dss: &DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, rustls::Error> {
            Ok(HandshakeSignatureValid::assertion())
        }

        fn verify_tls13_signature(
            &self,
            _message: &[u8],
            _cert: &CertificateDer<'_>,
            _dss: &DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, rustls::Error> {
            Ok(HandshakeSignatureValid::assertion())
        }

        fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
            vec![
                SignatureScheme::ECDSA_NISTP256_SHA256,
                SignatureScheme::ECDSA_NISTP384_SHA384,
                SignatureScheme::RSA_PKCS1_SHA256,
                SignatureScheme::RSA_PSS_SHA256,
                SignatureScheme::ED25519,
            ]
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn quic_upstream_and_listener_exchange_one_stream() {
        let server_config =
            crate::dev_server_config(vec!["localhost".into()], &[b"doq"]).expect("server config");
        let listener =
            QuicListener::bind("127.0.0.1:0".parse().expect("bind address"), server_config)
                .expect("quic listener");
        let BindAddr::Tcp(server_addr) = listener.local_addr().expect("local address") else {
            panic!("quic listener returned a non-stream address")
        };

        let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
        let mut tls = rustls::ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .expect("TLS versions")
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(AcceptAnyCertificate))
            .with_no_client_auth();
        tls.alpn_protocols = vec![b"doq".to_vec()];
        let upstream = Arc::new(
            QuicUpstream::with_client_config(server_addr, "localhost", tls).expect("quic upstream"),
        );

        let upstream_for_client = Arc::clone(&upstream);
        let client_task = tokio::spawn(async move {
            let mut client = std::future::poll_fn(|cx| upstream_for_client.poll_connect(cx))
                .await
                .expect("connect stream");
            client.write_all(b"doq-frame").await.expect("client write");
            client
        });
        let mut server = std::future::poll_fn(|cx| listener.poll_accept(cx))
            .await
            .expect("accept stream");
        let mut client = client_task.await.expect("client task");
        assert_eq!(upstream.pooled_connection_count(), 1);
        let mut request = [0u8; 9];
        server.read_exact(&mut request).await.expect("server read");
        assert_eq!(&request, b"doq-frame");
        server.write_all(b"doq-reply").await.expect("server write");
        let mut response = [0u8; 9];
        client.read_exact(&mut response).await.expect("client read");
        assert_eq!(&response, b"doq-reply");
    }
}
