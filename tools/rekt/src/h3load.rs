//! HTTP/3 load over proxima's NATIVE QUIC (`H3NativeUpstream`, prime UDP — not
//! quinn). The native client reuses one persistent QUIC connection per upstream
//! but is request-at-a-time (stream multiplexing is a documented substrate
//! follow-on), so concurrency comes from N connections per core, each firing
//! `GET /` back-to-back. Reuses the h1/h2 drive harness (per-core prime
//! factories, `Throughput`).
//!
//! Bench TLS: the server runs a dev self-signed cert; the client accepts any
//! server cert (localhost bench only — never the production path).

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use proxima::h3::native::bench_multiplexed;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, SignatureScheme};

use crate::engine::{Throughput, drive_replicated};
use crate::error::Error;

/// Bench-only verifier: accept any server certificate. Localhost h3 bench against
/// a dev self-signed server; NEVER the production path.
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
    fn verify_tls12_signature(&self, _message: &[u8], _cert: &CertificateDer<'_>, _dss: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }
    fn verify_tls13_signature(&self, _message: &[u8], _cert: &CertificateDer<'_>, _dss: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }
    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        rustls::crypto::aws_lc_rs::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

fn bench_client_config() -> rustls::ClientConfig {
    let mut config = rustls::ClientConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(AcceptAnyServerCert))
        .with_no_client_auth();
    config.alpn_protocols = vec![b"h3".to_vec()];
    config
}

/// Closed-loop native-h3 drive: `cores` prime cores, each opening
/// `connections_per_core` persistent native-QUIC connections, each firing `GET /`
/// request-at-a-time until the deadline. Composes
/// [`crate::engine::drive_replicated`] — see its doc-comment for why this
/// fans via `FuturesUnordered` and not
/// [`proxima_primitives::pipe::FanOut`]/[`proxima_primitives::pipe::ScatterGather`].
pub fn drive_h3(server_addr: SocketAddr, server_name: &str, connections_per_core: usize, cores: usize, duration: Duration, streams_per_conn: usize) -> Result<Throughput, Error> {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    let streams_per_conn = streams_per_conn.max(1);
    let rustls_config = Arc::new(bench_client_config());
    let server_name = server_name.to_string();
    drive_replicated(cores, connections_per_core, duration, move |deadline| {
        let server_name = server_name.clone();
        let rustls_config = Arc::clone(&rustls_config);
        async move { bench_multiplexed(server_addr, &server_name, rustls_config, streams_per_conn, deadline).await }
    })
}
