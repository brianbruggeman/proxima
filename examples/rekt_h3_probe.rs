//! rekt_h3_probe — drive the native proxima h3 client (`bench_multiplexed`)
//! against a real h3 server (e.g. nginx-h3) on the prime runtime and print
//! completed/errors. With `REKT_PROBE=1` the client loop also prints its exit
//! reason — the discriminator for the recv-buffer truncation fix: a healthy
//! run exits on the deadline, a truncated connection exits on a pump error.
//!
//!   cargo run --release --features http3,tracing-init --example rekt_h3_probe \
//!     -- 127.0.0.1:9141 localhost 32 10

use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use proxima::runtime::{CoreId, PrimeRuntime, Runtime};
use rustls::DigitallySignedStruct;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::CryptoProvider;
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};

// dev-cert probe: trust whatever the server presents (nginx self-signed). The
// fix under test is on the QUIC recv path, not certificate validation.
#[derive(Debug)]
struct TrustAnyServer(Arc<CryptoProvider>);

impl ServerCertVerifier for TrustAnyServer {
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

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.0.signature_verification_algorithms.supported_schemes()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // level-routed console logging; RUST_LOG controls the native client's emits.
    let telemetry = proxima_telemetry::export::install_console_logging()?;
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    // default groups (incl. post-quantum X25519MLKEM768): a server preferring it
    // sends a HelloRetryRequest, which the native client now follows.
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());

    let mut args = std::env::args().skip(1);
    let addr: SocketAddr = args
        .next()
        .unwrap_or_else(|| "127.0.0.1:9141".to_string())
        .parse()?;
    let server_name = args.next().unwrap_or_else(|| "localhost".to_string());
    let streams: usize = args.next().and_then(|raw| raw.parse().ok()).unwrap_or(32);
    let secs: u64 = args.next().and_then(|raw| raw.parse().ok()).unwrap_or(10);

    let mut tls = rustls::ClientConfig::builder_with_provider(provider.clone())
        .with_protocol_versions(&[&rustls::version::TLS13])?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(TrustAnyServer(provider)))
        .with_no_client_auth();
    tls.alpn_protocols = vec![b"h3".to_vec()];
    let rustls_config = Arc::new(tls);

    let runtime = Arc::new(PrimeRuntime::new(1)?);
    let deadline = Instant::now() + Duration::from_secs(secs);
    let (tx, rx) = mpsc::channel();

    let factory: Box<dyn FnOnce() -> Pin<Box<dyn Future<Output = ()>>> + Send + 'static> =
        Box::new(move || {
            Box::pin(async move {
                let result = proxima::h3::native::bench_multiplexed(
                    addr,
                    &server_name,
                    rustls_config,
                    streams,
                    deadline,
                )
                .await;
                let _ = tx.send(result);
            })
        });
    runtime.spawn_factory_on_core(CoreId(0), factory)?;

    let (completed, errors) = rx.recv()?;
    // drain() is synchronous and safe alongside the background console-drain
    // thread (the rings are multi-consumer); looping to exhaustion flushes
    // every record the run produced before the summary prints.
    while telemetry.drain() > 0 {}
    println!(
        "rekt h3: {completed} completed, {errors} errors (1 conn x {streams} streams, {secs}s) -> {addr}"
    );
    Ok(())
}
