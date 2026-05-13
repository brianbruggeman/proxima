//! rekt_h3_load — multi-connection native h3 load driver: opens `connections`
//! QUIC connections (each keeping `streams` requests in flight) spread across
//! `client_cores` prime cores, and sums completed/errors. A single-connection
//! driver only ever hits one reuseport server core; N connections are required
//! to exercise an N-core h3 server. Default (post-quantum) key exchange —
//! the client follows a HelloRetryRequest, so no group workaround is needed.
//!
//!   cargo run --release --features h3-native-upstream,tracing-init \
//!     --example rekt_h3_load -- 127.0.0.1:9141 localhost <connections> <streams> <secs> <client_cores>

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

/// Runs one connection's load loop. Behind `h3-part-source`, `REKT_PART_SOURCE`
/// (any value) switches to [`proxima::h3::native::bench_multiplexed_part_source`]
/// — the design doc's step 3 h3-client response path
/// (`docs/proxima-pipe/part-source-sink-design.md`), which reads response
/// `:status`/headers via a `PartSource` (0 heap allocations) instead of the
/// owned `ResponseHeaders` event. Unset (or built without the feature), this
/// is exactly the pre-existing [`proxima::h3::native::bench_multiplexed`] call
/// — no behavior change to the default path.
#[cfg(feature = "h3-part-source")]
async fn run_connection(
    addr: SocketAddr,
    server_name: &str,
    rustls_config: Arc<rustls::ClientConfig>,
    streams: usize,
    deadline: Instant,
) -> (u64, u64) {
    if std::env::var_os("REKT_PART_SOURCE").is_some() {
        proxima::h3::native::bench_multiplexed_part_source(
            addr,
            server_name,
            rustls_config,
            streams,
            deadline,
        )
        .await
    } else {
        proxima::h3::native::bench_multiplexed(addr, server_name, rustls_config, streams, deadline)
            .await
    }
}

#[cfg(not(feature = "h3-part-source"))]
async fn run_connection(
    addr: SocketAddr,
    server_name: &str,
    rustls_config: Arc<rustls::ClientConfig>,
    streams: usize,
    deadline: Instant,
) -> (u64, u64) {
    proxima::h3::native::bench_multiplexed(addr, server_name, rustls_config, streams, deadline)
        .await
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let telemetry = proxima_telemetry::export::install_console_logging()?;
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());

    let mut args = std::env::args().skip(1);
    let addr: SocketAddr = args
        .next()
        .unwrap_or_else(|| "127.0.0.1:9141".to_string())
        .parse()?;
    let server_name = args.next().unwrap_or_else(|| "localhost".to_string());
    let connections: usize = args.next().and_then(|raw| raw.parse().ok()).unwrap_or(8);
    let streams: usize = args.next().and_then(|raw| raw.parse().ok()).unwrap_or(32);
    let secs: u64 = args.next().and_then(|raw| raw.parse().ok()).unwrap_or(10);
    let client_cores: usize = args
        .next()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(1)
        .max(1);

    let mut tls = rustls::ClientConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(TrustAnyServer(provider)))
        .with_no_client_auth();
    tls.alpn_protocols = vec![b"h3".to_vec()];
    let rustls_config = Arc::new(tls);

    let runtime = Arc::new(PrimeRuntime::new(client_cores)?);
    let deadline = Instant::now() + Duration::from_secs(secs);
    let (tx, rx) = mpsc::channel();

    for conn in 0..connections {
        let name = server_name.clone();
        let cfg = rustls_config.clone();
        let tx = tx.clone();
        let factory: Box<dyn FnOnce() -> Pin<Box<dyn Future<Output = ()>>> + Send + 'static> =
            Box::new(move || {
                Box::pin(async move {
                    let result = run_connection(addr, &name, cfg, streams, deadline).await;
                    let _ = tx.send(result);
                })
            });
        runtime.spawn_factory_on_core(CoreId(conn % client_cores), factory)?;
    }
    drop(tx);

    let (mut completed, mut errors) = (0u64, 0u64);
    for _ in 0..connections {
        if let Ok((c, e)) = rx.recv() {
            completed += c;
            errors += e;
        }
    }
    // drain() is synchronous and safe alongside the background console-drain
    // thread (the rings are multi-consumer); looping to exhaustion flushes
    // every record the run produced before the summary prints.
    while telemetry.drain() > 0 {}
    println!(
        "rekt h3: {completed} completed, {errors} errors ({connections} conns x {streams} streams, {client_cores} client-cores, {secs}s) -> {addr}"
    );
    Ok(())
}
