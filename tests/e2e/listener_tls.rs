//! End-to-end TLS termination test: a real `tokio::net::TcpListener`
//! wrapped by `proxima::tls::build_acceptor` accepts a real TLS
//! handshake from a `tokio_rustls::TlsConnector`. Proves the cert
//! chain + private key + ServerConfig assembly are wired correctly.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
#![cfg(feature = "tls")]

use std::sync::Arc;

use proxima::tls::{TlsConfig, build_acceptor};
use rustls::client::ClientConfig;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, SignatureScheme};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio_rustls::TlsConnector;

/// rustls verifier that accepts any server cert. Test-only — never use
/// in production. The point of this test is the handshake mechanics,
/// not certificate validation.
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

#[proxima::test]
async fn self_signed_acceptor_completes_tls_handshake_with_test_client() {
    // make sure rustls' aws_lc_rs CryptoProvider is installed for ClientConfig
    // builders that don't take an explicit provider.
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

    let acceptor = build_acceptor(&TlsConfig::self_signed()).expect("build acceptor");

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local addr");

    let server = tokio::spawn(async move {
        let (socket, _peer) = listener.accept().await.expect("accept tcp");
        let mut tls_stream = acceptor.accept(socket).await.expect("server handshake");
        // Echo a single byte to prove the tunnel works post-handshake.
        let mut byte = [0_u8; 1];
        tls_stream.read_exact(&mut byte).await.expect("read");
        tls_stream.write_all(&byte).await.expect("write");
        tls_stream.shutdown().await.expect("shutdown");
    });

    let client_config = ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(AcceptAnyServerCert))
        .with_no_client_auth();
    let connector = TlsConnector::from(Arc::new(client_config));
    let tcp = TcpStream::connect(addr).await.expect("connect tcp");
    let server_name = ServerName::try_from("localhost").expect("server name");
    let mut tls_stream = connector
        .connect(server_name, tcp)
        .await
        .expect("client handshake");

    tls_stream.write_all(b"!").await.expect("write");
    let mut echo = [0_u8; 1];
    tls_stream.read_exact(&mut echo).await.expect("read echo");
    assert_eq!(echo, *b"!", "TLS tunnel must round-trip a byte");

    server.await.expect("server task");
}
