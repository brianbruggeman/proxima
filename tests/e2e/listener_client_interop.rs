//! The payoff: `Listener` and `Client` are two faces of the same
//! `SpecBuilder` coin. `Listener::builder()` stands up a REAL loopback HTTP
//! listener (through the existing `App::serve` / `HttpListenProtocol`
//! path — no new driver, no new socket code); `Client::builder()` dials it
//! over a REAL socket. Both builders compose the identical
//! `TransportSugar`/`ProtocolSugar` axes.

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(all(
    feature = "http1",
    any(
        feature = "runtime-tokio",
        all(
            feature = "serve-prime",
            feature = "runtime-prime-reactor",
            any(target_os = "linux", target_os = "macos")
        )
    )
))]

use std::future::Future;
use std::net::{SocketAddr, TcpListener as StdTcpListener, TcpStream};
use std::time::Duration;

use bytes::Bytes;

use proxima::error::ProximaError;
use proxima::pipe::into_handle;
use proxima::request::{Request, Response};
use proxima::{Client, Listener, ListenerBuilderEntry, ProtocolSugar, SendPipe, TransportSugar};

/// Answers every request with a fixed 200 — the "real pipe" the listener
/// dispatches to.
struct FixedOk;

impl SendPipe for FixedOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move { Ok(Response::ok("listener-client-interop")) }
    }
}

/// Resolve an ephemeral loopback port the same way `tests/serve_parity.rs`'s
/// `pick_free_addr` does: bind once to let the OS assign a port, then drop so
/// the real listener can rebind it. `App::serve`'s `Server` has no
/// `bind_addr()` accessor (unlike the lower-level `ListenerHandle`), so the
/// concrete address is resolved here rather than inside the builder.
fn free_loopback_addr() -> SocketAddr {
    let probe = StdTcpListener::bind("127.0.0.1:0").expect("probe bind");
    let addr = probe.local_addr().expect("probe addr");
    drop(probe);
    addr
}

/// Poll a raw connect until the listener's real `bind`/`listen` syscalls have
/// run — `App::serve` (unlike `proxima_listen::handle::Listener::run_with_runtime`)
/// returns before the spawned lane's first poll, so a caller must not assume
/// the socket is live the instant `.serve()` resolves. Mirrors
/// `examples/hello/main.rs`'s `wait_until_listening`.
fn wait_until_listening(addr: SocketAddr) {
    for _ in 0..200 {
        if TcpStream::connect(addr).is_ok() {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("listener at {addr} never came up");
}

/// The symmetric proof: `Listener::builder()` (server side) and
/// `Client::builder()` (client side) interoperate over a real loopback
/// socket, both composing `TransportSugar`'s `.tcp()` from the SAME
/// `SpecBuilder` seam. Dials BOTH the literal root and a nested path —
/// `ListenerBuilder::serve` mounts at the `"/{*path}"` catch-all convention
/// (`src/app.rs:925,981`), not the literal `"/"`, so a non-root path must
/// answer too, not 404.
#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn listener_builder_serves_what_client_builder_dials() {
    let bind = free_loopback_addr();

    let server = Listener::builder()
        .bind(bind)
        .tcp()
        .handle(into_handle(FixedOk))
        .serve()
        .await
        .expect("listener builder serves");

    wait_until_listening(bind);

    let client = Client::builder()
        .http(format!("http://{bind}"))
        .tcp()
        .build()
        .expect("client builder builds");

    let root_response = client.call("GET", "/").send().await.expect("client send");
    assert_eq!(root_response.status(), 200);
    let root_body = root_response.text().await.expect("response text");
    assert_eq!(root_body, "listener-client-interop");

    let nested_response = client
        .call("GET", "/health/x")
        .send()
        .await
        .expect("client send nested path");
    assert_eq!(
        nested_response.status(),
        200,
        "non-root path must reach the mounted handle, not 404 the literal \"/\" mount"
    );
    let nested_body = nested_response.text().await.expect("response text");
    assert_eq!(nested_body, "listener-client-interop");

    server.stop();
}

/// Proves `.tls(TlsConfig)` REALLY terminates TLS — not a silent plaintext
/// no-op. A raw `tokio_rustls::TlsConnector` (accept-any-cert, test-only)
/// completes a genuine handshake against the socket `Listener::builder()`
/// bound, then a raw HTTP/1.1 request tunneled through that TLS stream gets
/// the same response the plaintext test above gets — proving the cert/key
/// wired through `.tls()` reached `HttpListenProtocol::serve_default`'s
/// `tokio_rustls::TlsAcceptor` (`proxima-http/src/listener/mod.rs:195`) and
/// terminated a real session, not just set an inert spec key.
#[cfg(feature = "tls")]
#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn listener_builder_tls_terminates_a_real_handshake() {
    use std::sync::Arc;

    use proxima::tls::TlsConfig;
    use rustls::client::ClientConfig;
    use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
    use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
    use rustls::{DigitallySignedStruct, SignatureScheme};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpStream;
    use tokio_rustls::TlsConnector;

    // rustls verifier that accepts any server cert — test-only, mirrors
    // `tests/e2e/listener_tls.rs`'s `AcceptAnyServerCert`; the point of this
    // test is that a handshake completes at all, not certificate trust.
    #[derive(Debug)]
    struct AcceptAnyServerCert;

    impl ServerCertVerifier for AcceptAnyServerCert {
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

    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    let bind = free_loopback_addr();

    let server = Listener::builder()
        .bind(bind)
        .tls(TlsConfig::self_signed())
        .handle(into_handle(FixedOk))
        .serve()
        .await
        .expect("listener builder serves tls");

    wait_until_listening(bind);

    let client_config = ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(AcceptAnyServerCert))
        .with_no_client_auth();
    let connector = TlsConnector::from(Arc::new(client_config));
    let tcp = TcpStream::connect(bind).await.expect("connect tcp");
    let server_name = ServerName::try_from("localhost").expect("server name");
    let mut tls_stream = connector
        .connect(server_name, tcp)
        .await
        .expect("client tls handshake");

    tls_stream
        .write_all(b"GET /health/x HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .await
        .expect("write request over tls");
    // the h1 listener closes the TCP socket after `Connection: close`
    // without a TLS `close_notify` alert; rustls treats that as
    // unexpected-eof (truncation-attack safety), but by the time it fires
    // the Content-Length-bounded HTTP response is already fully buffered in
    // `raw_response`, so only THIS specific error is tolerated here.
    let mut raw_response = Vec::new();
    match tls_stream.read_to_end(&mut raw_response).await {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => {}
        Err(error) => panic!("read response over tls: {error}"),
    }
    let response_text = String::from_utf8_lossy(&raw_response);

    assert!(
        response_text.starts_with("HTTP/1.1 200"),
        "got: {response_text}"
    );
    assert!(
        response_text.contains("listener-client-interop"),
        "got: {response_text}"
    );

    server.stop();
}
