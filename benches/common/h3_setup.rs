//! Shared h3 bench plumbing. Spins up a `proxima::quic::Endpoint`
//! server, hands each accepted connection to
//! [`proxima::h3::serve_h3_connection`]. Provides a warm h3 client
//! (quinn + h3-quinn + h3 client crate, no hyper).
//!
//! Benches use this so they measure proxima's h3 surface, not
//! cert-generation or accept-loop overhead.

#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use bytes::Bytes;
use proxima::pipe::PipeHandle;

pub fn build_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("tokio runtime")
}

pub fn install_crypto_provider() {
    let _ = quinn::rustls::crypto::aws_lc_rs::default_provider().install_default();
}

/// Boot a proxima h3 server on a loopback ephemeral port. Returns the
/// bound addr; the spawned accept loop runs until the runtime drops.
pub fn start_h3_server(runtime: &tokio::runtime::Runtime, dispatch: PipeHandle) -> SocketAddr {
    install_crypto_provider();
    runtime.block_on(async {
        let server_config =
            proxima::quic::dev_server_config(vec!["localhost".to_string()], &[b"h3"])
                .expect("dev server config");
        let endpoint = Arc::new(
            proxima::quic::Endpoint::server(
                (std::net::Ipv4Addr::LOCALHOST, 0).into(),
                server_config,
            )
            .expect("quic bind"),
        );
        let addr = endpoint.local_addr().expect("local addr");

        let endpoint_for_accept = endpoint.clone();
        tokio::spawn(async move {
            loop {
                match endpoint_for_accept.accept().await {
                    Some(Ok(connection)) => {
                        let dispatch = dispatch.clone();
                        let in_flight = Arc::new(AtomicU64::new(0));
                        tokio::spawn(async move {
                            let _ =
                                proxima::h3::serve_h3_connection(connection, dispatch, in_flight)
                                    .await;
                        });
                    }
                    Some(Err(_)) => continue,
                    None => break,
                }
            }
        });
        addr
    })
}

/// Trust-anything server cert verifier for the bench client side —
/// we generate a fresh self-signed cert per server start.
#[derive(Debug)]
struct AcceptAnyCert;

impl quinn::rustls::client::danger::ServerCertVerifier for AcceptAnyCert {
    fn verify_server_cert(
        &self,
        _end_entity: &quinn::rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[quinn::rustls::pki_types::CertificateDer<'_>],
        _server_name: &quinn::rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: quinn::rustls::pki_types::UnixTime,
    ) -> Result<quinn::rustls::client::danger::ServerCertVerified, quinn::rustls::Error> {
        Ok(quinn::rustls::client::danger::ServerCertVerified::assertion())
    }
    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &quinn::rustls::pki_types::CertificateDer<'_>,
        _dss: &quinn::rustls::DigitallySignedStruct,
    ) -> Result<quinn::rustls::client::danger::HandshakeSignatureValid, quinn::rustls::Error> {
        Ok(quinn::rustls::client::danger::HandshakeSignatureValid::assertion())
    }
    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &quinn::rustls::pki_types::CertificateDer<'_>,
        _dss: &quinn::rustls::DigitallySignedStruct,
    ) -> Result<quinn::rustls::client::danger::HandshakeSignatureValid, quinn::rustls::Error> {
        Ok(quinn::rustls::client::danger::HandshakeSignatureValid::assertion())
    }
    fn supported_verify_schemes(&self) -> Vec<quinn::rustls::SignatureScheme> {
        quinn::rustls::crypto::aws_lc_rs::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

pub fn client_config() -> quinn::ClientConfig {
    let mut tls = quinn::rustls::ClientConfig::builder_with_protocol_versions(&[
        &quinn::rustls::version::TLS13,
    ])
    .dangerous()
    .with_custom_certificate_verifier(Arc::new(AcceptAnyCert))
    .with_no_client_auth();
    tls.alpn_protocols = vec![b"h3".to_vec()];
    let crypto =
        quinn::crypto::rustls::QuicClientConfig::try_from(tls).expect("quic client config");
    quinn::ClientConfig::new(Arc::new(crypto))
}

/// Build a fresh client endpoint bound to an ephemeral local port.
pub fn make_client_endpoint() -> quinn::Endpoint {
    let mut endpoint =
        quinn::Endpoint::client((std::net::Ipv4Addr::UNSPECIFIED, 0).into()).expect("quic client");
    endpoint.set_default_client_config(client_config());
    endpoint
}

/// Open one h3 connection to `addr` and return the `SendRequest` handle
/// callers reuse for warm-connection iterations. Drives the h3
/// connection task in the background so it stays alive for the bench.
pub async fn warm_h3_client(
    endpoint: &quinn::Endpoint,
    addr: SocketAddr,
) -> h3::client::SendRequest<h3_quinn::OpenStreams, Bytes> {
    let connecting = endpoint.connect(addr, "localhost").expect("connect");
    let connection = connecting.await.expect("handshake");
    let h3_conn = h3_quinn::Connection::new(connection);
    let (mut driver, send_request) = h3::client::builder()
        .build::<_, _, Bytes>(h3_conn)
        .await
        .expect("h3 build");
    tokio::spawn(async move {
        let _ = std::future::poll_fn(|cx| driver.poll_close(cx)).await;
    });
    send_request
}
