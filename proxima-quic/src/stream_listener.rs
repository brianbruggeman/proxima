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

use futures::lock::Mutex as AsyncMutex;
use quinn::ClientConfig;
use quinn::{Endpoint, RecvStream, SendStream, ServerConfig};
use tokio_util::compat::{Compat, TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use proxima_primitives::stream::{
    BindAddr, ConnectFuture, PeerInfo, StreamConnection, StreamListener, StreamUpstream,
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

/// QUIC client adapter for stream protocols such as DNS-over-QUIC.
///
/// Each connection future opens one bidirectional application stream, reusing a
/// bounded pool of authenticated QUIC connections. The caller supplies a TLS
/// config whose ALPN is appropriate for its protocol (DoQ uses `doq`). The
/// returned stream uses the existing bounded protocol framing; this adapter
/// owns only endpoint, handshake, stream setup, and bounded connection reuse.
pub struct QuicUpstream {
    endpoint: Endpoint,
    server_addr: SocketAddr,
    server_name: String,
    connections: AsyncMutex<Vec<quinn::Connection>>,
    /// maximum connection handles retained for subsequent stream setup.
    max_connections: usize,
}

impl QuicUpstream {
    /// Build a QUIC client endpoint using the caller's rustls-backed QUIC
    /// configuration. No network activity occurs until its connection future is polled.
    pub fn with_client_config(
        server_addr: SocketAddr,
        server_name: impl Into<String>,
        tls_config: rustls::ClientConfig,
    ) -> io::Result<Self> {
        Self::with_client_config_and_limit(server_addr, server_name, tls_config, 1)
    }

    /// build a client endpoint retaining at most `max_connections` handles
    /// for reuse. concurrent dials can create additional live connections.
    /// zero is rejected before network activity.
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
            connections: AsyncMutex::new(Vec::with_capacity(max_connections.min(4))),
            max_connections,
        })
    }

    #[cfg(all(test, feature = "tokio-compat"))]
    #[allow(clippy::expect_used)]
    fn pooled_connection_count(&self) -> usize {
        self.connections.try_lock().expect("quic pool lock").len()
    }
}

impl StreamUpstream for QuicUpstream {
    type Conn = Box<dyn StreamConnection>;

    fn connect_future(&self) -> ConnectFuture<'_, Self::Conn> {
        Box::pin(async move {
            let connect = || async {
                self.endpoint
                    .connect(self.server_addr, &self.server_name)
                    .map_err(|error| io::Error::other(format!("quic connect: {error}")))?
                    .await
                    .map_err(|error| io::Error::other(format!("quic handshake: {error}")))
            };
            let pooled = self.connections.lock().await.pop();
            let mut connection = match pooled {
                Some(connection) => connection,
                None => connect().await?,
            };
            let (send, recv) = match connection.open_bi().await {
                Ok(stream) => stream,
                Err(_) => {
                    connection = connect().await?;
                    connection
                        .open_bi()
                        .await
                        .map_err(|error| io::Error::other(format!("quic open stream: {error}")))?
                }
            };
            let peer = connection.remote_address();
            let mut connections = self.connections.lock().await;
            if connections.len() < self.max_connections {
                connections.push(connection);
            }
            Ok(Box::new(QuicStreamConnection::new(send, recv, Some(peer))) as Self::Conn)
        })
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
    // the listener retains its single accept operation between polls.
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
    use futures::channel::oneshot;
    use futures::future::poll_fn;
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

        let (started, both_pending) = oneshot::channel();
        let upstream_for_client = Arc::clone(&upstream);
        let client_task = tokio::spawn(async move {
            let mut first = upstream_for_client.connect_future();
            let mut second = upstream_for_client.connect_future();
            poll_fn(|context| {
                assert!(first.as_mut().poll(context).is_pending());
                assert!(second.as_mut().poll(context).is_pending());
                Poll::Ready(())
            })
            .await;
            started.send(()).expect("announce both pending handshakes");
            let exchange = async |future: ConnectFuture<'_, Box<dyn StreamConnection>>,
                                  request: &[u8; 9],
                                  expected: &[u8; 9]| {
                let mut client = future.await.expect("connect stream");
                client.write_all(request).await.expect("client write");
                let mut response = [0u8; 9];
                client.read_exact(&mut response).await.expect("client read");
                assert_eq!(&response, expected);
            };
            futures::join!(
                exchange(first, b"doq-first", b"ack-first"),
                exchange(second, b"doq-other", b"ack-other")
            );
        });
        both_pending
            .await
            .expect("both calls own a pending handshake");
        let mut first = poll_fn(|context| listener.poll_accept(context))
            .await
            .expect("accept first stream");
        let mut second = poll_fn(|context| listener.poll_accept(context))
            .await
            .expect("accept second stream");
        assert_eq!(upstream.pooled_connection_count(), 1);
        let mut requests = Vec::new();
        for connection in [&mut first, &mut second] {
            let mut request = [0u8; 9];
            connection
                .read_exact(&mut request)
                .await
                .expect("server read");
            let response = match &request {
                b"doq-first" => b"ack-first",
                b"doq-other" => b"ack-other",
                other => panic!("unexpected request: {other:?}"),
            };
            connection.write_all(response).await.expect("server write");
            requests.push(request);
        }
        requests.sort();
        assert_eq!(requests, vec![*b"doq-first", *b"doq-other"]);
        client_task.await.expect("client task");
    }
}
