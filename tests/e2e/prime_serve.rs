//! End-to-end tests for `PrimeRuntime::serve_http` and
//! `PrimeRuntime::serve_https_with_tls`. Both use prime's native
//! `net::TcpListener` (futures-io). The HTTPS variant bridges
//! `tokio_rustls::TlsAcceptor` (tokio-io) into prime's stream via
//! `tokio_util::compat`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
#![cfg(all(
    all(
        feature = "runtime-prime-executor",
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-reactor",
        feature = "runtime-prime-bgpool"
    ),
    feature = "http1"
))]

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use proxima::SendPipe;
use proxima::error::ProximaError;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::prime::PrimeRuntime;
use proxima::request::{Request, Response};
use proxima::runtime::PrimeServeExt;
use tokio::net::TcpListener;

struct SynthPipe {
    body: &'static str,
}

impl SendPipe for SynthPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let body = self.body;
        async move { Ok(Response::ok(body)) }
    }
}


async fn pick_free_addr() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    drop(listener);
    addr
}

async fn wait_until_listening(addr: SocketAddr) {
    for _ in 0..100 {
        if tokio::net::TcpStream::connect(addr).await.is_ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("listener at {addr} never came up");
}

struct ClientResponse {
    status: u16,
    body: Vec<u8>,
}

async fn http_get(addr: SocketAddr, path: &str) -> ClientResponse {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
    let request = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes()).await.expect("write");
    let mut bytes = Vec::new();
    stream.read_to_end(&mut bytes).await.expect("read");
    let header_end = bytes
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .expect("header terminator");
    let header_text = std::str::from_utf8(&bytes[..header_end]).expect("header utf8");
    let status_line = header_text.split("\r\n").next().expect("status line");
    let status = status_line
        .split(' ')
        .nth(1)
        .expect("status code")
        .parse::<u16>()
        .expect("status parses");
    let body = bytes[header_end + 4..].to_vec();
    ClientResponse { status, body }
}

#[proxima::test(flavor = "multi_thread", worker_threads = 4)]
async fn serve_http_dispatches_synth_pipe_response() {
    let runtime = Arc::new(
        PrimeRuntime::builder()
            .cores(2)
            .background_inline()
            .build()
            .expect("build runtime"),
    );
    let pipe: PipeHandle = into_handle(SynthPipe {
        body: "hello from synth",
    });

    let addr = pick_free_addr().await;
    let _handle = runtime
        .serve_http(addr, pipe)
        .expect("serve_http should bind");
    wait_until_listening(addr).await;

    let response = http_get(addr, "/").await;
    assert_eq!(response.status, 200, "status must be 200");
    let body_text = std::str::from_utf8(&response.body).expect("body utf8");
    // Response body may be chunk-framed (Transfer-Encoding: chunked).
    // The synth body bytes are intact inside the framing; assert
    // they appear as a substring rather than parsing chunks here.
    assert!(
        body_text.contains("hello from synth"),
        "synth body not found in response; raw body: {body_text:?}",
    );
}

#[cfg(feature = "tls")]
mod https {
    use super::*;
    use rustls::client::ClientConfig;
    use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
    use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
    use rustls::{DigitallySignedStruct, SignatureScheme};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio_rustls::TlsConnector;

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
                SignatureScheme::RSA_PKCS1_SHA256,
                SignatureScheme::RSA_PSS_SHA256,
                SignatureScheme::ED25519,
            ]
        }
    }

    #[proxima::test(flavor = "multi_thread", worker_threads = 4)]
    async fn serve_https_with_tls_terminates_handshake_and_dispatches_to_pipe() {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

        let runtime = Arc::new(
            PrimeRuntime::builder()
                .cores(2)
                .background_inline()
                .build()
                .expect("build runtime"),
        );
        let pipe: PipeHandle = into_handle(SynthPipe {
            body: "hello via tls",
        });

        let addr = pick_free_addr().await;
        let tls = proxima::tls::TlsConfig::self_signed();
        let _handle = runtime
            .serve_https_with_tls(addr, tls, pipe)
            .expect("serve_https_with_tls should bind");
        wait_until_listening(addr).await;

        let client_config = ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(AcceptAnyServerCert))
            .with_no_client_auth();
        let connector = TlsConnector::from(Arc::new(client_config));

        let tcp = tokio::net::TcpStream::connect(addr).await.expect("connect");
        let server_name = ServerName::try_from("localhost").expect("server name");
        let mut tls_stream = connector
            .connect(server_name, tcp)
            .await
            .expect("tls handshake");

        let request = b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
        tls_stream.write_all(request).await.expect("write");
        let mut bytes = Vec::new();
        // rustls returns UnexpectedEof when the peer closes the underlying
        // TCP without a TLS close_notify alert. Our `Connection: close`
        // request triggers exactly that shape. The body is already in the
        // buffer at that point; tolerate the soft close.
        match tls_stream.read_to_end(&mut bytes).await {
            Ok(_) => {}
            Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => {}
            Err(err) => panic!("unexpected read error: {err}"),
        }

        let body_text = String::from_utf8_lossy(&bytes);
        assert!(
            body_text.contains("hello via tls"),
            "synth body not found in TLS-terminated response: {body_text:?}",
        );
    }
}
