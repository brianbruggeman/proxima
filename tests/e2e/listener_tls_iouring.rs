//! End-to-end TLS over io_uring on Linux. Boots an HttpListenProtocol
//! configured with a self-signed cert + the `__proxima_tls` spec key,
//! runs everything inside a `tokio_uring::start` worker, then
//! connects with `tokio_rustls::TlsConnector` (accepting any cert)
//! and verifies the response round-trips. Proves the full path:
//! tokio_uring::net::TcpListener accept → UringAsyncStream adapter
//! → tokio_rustls::TlsAcceptor handshake → Connection state machine
//! → Pipe dispatch → response written back through the adapter.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
#![cfg(all(target_os = "linux", feature = "io-uring", feature = "tls"))]

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures::channel::oneshot;
use proxima::listeners::HttpListenerSpec;
use proxima::{PipeHandle, ProximaError, Request, Response, into_handle};
use rustls::client::ClientConfig;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, SignatureScheme};
use serde_json::json;
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

struct StaticOk;

impl proxima::SendPipe for StaticOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move {
            Ok(Response::new(200)
                .with_header("content-length", "2")
                .with_body(Bytes::from_static(b"ok")))
        }
    }
}

#[test]
fn iouring_tls_round_trip() {
    // tokio_uring::start runs a single-thread current-thread runtime
    // with the io_uring driver attached. Everything in this block —
    // the listener, the client, the spawn_local'd tasks — share that
    // runtime.
    tokio_uring::start(async move {
        let port = 28443_u16;
        let bind: SocketAddr = format!("127.0.0.1:{port}").parse().expect("addr");

        let raw_spec = json!({
            "__proxima_tls": {
                "mode": { "kind": "self_signed" }
            }
        });

        let dispatch: PipeHandle = into_handle(StaticOk);
        let spec = Arc::new(HttpListenerSpec {
            max_body_bytes: None,
        });
        let (shutdown_tx, shutdown_rx) = oneshot::channel();

        // Drive serve_uring directly — it's the public entry point
        // for the io_uring path. spawn_local since the future is
        // !Send (Rc<TcpStream>).
        let raw_spec_for_serve = raw_spec.clone();
        let telemetry: proxima::TelemetryHandle = Arc::new(proxima::NoopTelemetry);
        tokio::task::spawn_local(async move {
            if let Err(error) = proxima::listeners::http_uring::serve_uring(
                bind,
                dispatch,
                spec,
                &raw_spec_for_serve,
                telemetry,
                shutdown_rx,
            )
            .await
            {
                eprintln!("serve_uring error: {error:?}");
            }
        });

        // Wait briefly for the listener to bind.
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Build a rustls ClientConfig that accepts any server cert.
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let mut client_config = ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(AcceptAnyServerCert))
            .with_no_client_auth();
        client_config.alpn_protocols.push(b"http/1.1".to_vec());
        let connector = TlsConnector::from(Arc::new(client_config));

        let plain_stream = tokio::net::TcpStream::connect(bind)
            .await
            .expect("tcp connect");
        let server_name = ServerName::try_from("localhost").expect("server name");
        let mut tls_stream = connector
            .connect(server_name, plain_stream)
            .await
            .expect("tls handshake");

        tls_stream
            .write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .await
            .expect("write request");

        let mut response = Vec::with_capacity(256);
        let _ = tls_stream.read_to_end(&mut response).await;
        let text = String::from_utf8_lossy(&response);
        assert!(text.starts_with("HTTP/1.1 200"), "response: {text}");
        assert!(text.contains("ok"), "response body not echoed: {text}");

        let _ = shutdown_tx.send(());
    });
}
