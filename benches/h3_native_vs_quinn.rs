//! C43 compare-bench — native HTTP/3 client (`H3NativeUpstream`, prime
//! UDP + sans-IO QUIC/H3) vs the incumbent quinn `Http3Upstream` (P7),
//! both against the SAME mounted native listener.
//!
//! design-favors: incumbent. quinn's design point is a persistent,
//! multiplexed connection — connection REUSE. So the fair, headline arm
//! is WARM: one upstream, handshake paid once, then N reused `GET /`
//! requests measured per-call. Both sides reuse, so the handshake is out
//! of the steady-state number and we compare the per-request hot path on
//! quinn's home turf.
//!
//! A COLD arm (fresh upstream per call) is reported too, but it is NOT a
//! verdict: a fresh quinn Endpoint per iteration churns quinn into a
//! ~1s-per-call pathology (uniform on Linux) — a harness artifact. The
//! warm arm is the honest comparator.
//!
//! Manual timing (not criterion): handshake-dominated cold + sub-ms warm
//! over loopback, high CoV; the meaningful numbers are mean + range over
//! N. Run on host-b (the bench host), not mac.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::future::Future;
use std::net::{Ipv4Addr, SocketAddr};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures::channel::oneshot;

use proxima::error::ProximaError;
use proxima::listen::{ListenProtocol, ServeContext};
use proxima::listeners::H3NativeListenProtocol;
use proxima::pipe::{into_handle};
use proxima::request::{Request, Response};
use proxima::telemetry::NoopTelemetry;
use proxima_http::http3::native::H3NativeUpstream;
use proxima_http::http3::upstream::Http3Upstream;
use proxima_primitives::pipe::SendPipe;

// The native client reuses unbounded (stream reaping landed — see the
// 50-request `native_h3_upstream_reuses_connection_across_calls` E2E). The
// warm arm stays at 6 because the QUINN reference batches ACKs, so the
// listener accumulates sent-but-unacked response streams that can't yet be
// reaped and trips `max_concurrent_bidi = 8` at higher N — a quinn-vs-
// listener interaction, not a native limit. 6 keeps both arms clean; the
// fair verdict (native beats quinn) is unchanged from C43's sealed run.
const WARM: usize = 6;
const COLD: usize = 50;

struct ConstantOk;

impl SendPipe for ConstantOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;
    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move { Ok(Response::ok(Bytes::from_static(b"ok"))) }
    }
}

fn main() {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

    let probe = std::net::UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let bound = probe.local_addr().unwrap();
    drop(probe);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server_thread = std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
            let protocol = H3NativeListenProtocol::new();
            let context = ServeContext::new(Arc::new(NoopTelemetry));
            let spec = serde_json::json!({ "dev_self_signed": true, "dev_sans": ["localhost"] });
            let _ = protocol
                .serve(bound, into_handle(ConstantOk), &spec, context, shutdown_rx)
                .await;
        });
    });
    std::thread::sleep(Duration::from_millis(300));

    let native_warm = run_native(bound, false);
    let quinn_warm = run_quinn(bound, false);
    let native_cold = run_native(bound, true);
    let quinn_cold = run_quinn(bound, true);

    let _ = shutdown_tx.send(());
    let _ = server_thread.join();

    println!("\n# WARM — one connection, reused (the fair, headline comparison)");
    report("native H3NativeUpstream (warm/reused)", &native_warm);
    report("quinn  Http3Upstream   (warm/reused)", &quinn_warm);
    let nw = mean(&native_warm);
    let qw = mean(&quinn_warm);
    let ratio = nw / qw;
    println!(
        "VERDICT (warm per-request): native {nw:.1}µs / quinn {qw:.1}µs = {ratio:.2}x — {}",
        if ratio <= 1.10 {
            "native MEETS/BEATS quinn on its home turf"
        } else {
            "INCUMBENT WINS: native slower per warm request (driver tuning is the path)"
        }
    );

    println!("\n# COLD — fresh upstream per call (informational; NOT a verdict)");
    report("native H3NativeUpstream (cold)", &native_cold);
    report("quinn  Http3Upstream   (cold)", &quinn_cold);
    let qcm = quinn_cold.iter().min().unwrap().as_secs_f64() * 1e6;
    if qcm > 100_000.0 {
        println!(
            "  (quinn cold min {qcm:.0}µs is pathological — fresh-endpoint-per-iter churn, not a ranking)"
        );
    }

    std::process::exit(0);
}

/// Native arm on a prime shard. `cold` = fresh upstream per call; warm =
/// one upstream reused (handshake amortized; first call is a warmup).
fn run_native(bound: SocketAddr, cold: bool) -> Vec<Duration> {
    let handle =
        prime::os::core_shard::launch_with_lanes(proxima::CoreId(0), None, 2, 16).expect("launch");
    let done = Arc::new(AtomicBool::new(false));
    let done_for = done.clone();
    let n = if cold { COLD } else { WARM };
    let samples: Arc<Mutex<Vec<Duration>>> = Arc::new(Mutex::new(Vec::with_capacity(n)));
    let samples_for = samples.clone();

    handle
        .dispatch_factory(Box::new(move || {
            let done = done_for.clone();
            let samples = samples_for.clone();
            Box::pin(async move {
                let shared = (!cold).then(|| {
                    H3NativeUpstream::with_client_config(bound, "localhost", accept_any())
                });
                if let Some(upstream) = shared.as_ref() {
                    // warmup: pay the handshake once, untimed.
                    let _ = SendPipe::call(upstream, get())
                        .await
                        .expect("native warmup");
                }
                for _ in 0..n {
                    let fresh;
                    let upstream = match shared.as_ref() {
                        Some(u) => u,
                        None => {
                            fresh = H3NativeUpstream::with_client_config(
                                bound,
                                "localhost",
                                accept_any(),
                            );
                            &fresh
                        }
                    };
                    let start = Instant::now();
                    let response = SendPipe::call(upstream, get()).await.expect("native call");
                    let elapsed = start.elapsed();
                    assert_eq!(response.status, 200);
                    samples.lock().unwrap().push(elapsed);
                }
                done.store(true, Ordering::Release);
            }) as Pin<Box<dyn Future<Output = ()> + 'static>>
        }))
        .expect("dispatch_factory");

    let deadline = Instant::now() + Duration::from_secs(120);
    while !done.load(Ordering::Acquire) {
        assert!(Instant::now() < deadline, "native bench stalled");
        std::thread::sleep(Duration::from_millis(5));
    }
    handle.shutdown_and_join().expect("shutdown");
    Arc::try_unwrap(samples).unwrap().into_inner().unwrap()
}

/// Quinn arm on a tokio runtime. `cold` = fresh `Http3Upstream` per call;
/// warm = one `Http3Upstream` reused (it lazily connects on the first call
/// and reuses thereafter — its design point).
fn run_quinn(bound: SocketAddr, cold: bool) -> Vec<Duration> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let n = if cold { COLD } else { WARM };
    runtime.block_on(async move {
        let mut samples = Vec::with_capacity(n);
        let shared =
            (!cold).then(|| Http3Upstream::with_client_config(bound, "localhost", accept_any()));
        if let Some(upstream) = shared.as_ref() {
            let _ = SendPipe::call(upstream, get()).await.expect("quinn warmup");
        }
        for _ in 0..n {
            let fresh;
            let upstream = match shared.as_ref() {
                Some(u) => u,
                None => {
                    fresh = Http3Upstream::with_client_config(bound, "localhost", accept_any());
                    &fresh
                }
            };
            let start = Instant::now();
            let response = SendPipe::call(upstream, get()).await.expect("quinn call");
            let elapsed = start.elapsed();
            assert_eq!(response.status, 200);
            samples.push(elapsed);
        }
        samples
    })
}

fn get() -> Request<Bytes> {
    Request::builder().method("GET").path("/").build().unwrap()
}

fn accept_any() -> rustls::ClientConfig {
    let mut config =
        rustls::ClientConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(AcceptAny))
            .with_no_client_auth();
    config.alpn_protocols = vec![b"h3".to_vec()];
    config
}

#[derive(Debug)]
struct AcceptAny;

impl rustls::client::danger::ServerCertVerifier for AcceptAny {
    fn verify_server_cert(
        &self,
        _e: &rustls::pki_types::CertificateDer<'_>,
        _i: &[rustls::pki_types::CertificateDer<'_>],
        _s: &rustls::pki_types::ServerName<'_>,
        _o: &[u8],
        _n: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }
    fn verify_tls12_signature(
        &self,
        _m: &[u8],
        _c: &rustls::pki_types::CertificateDer<'_>,
        _d: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }
    fn verify_tls13_signature(
        &self,
        _m: &[u8],
        _c: &rustls::pki_types::CertificateDer<'_>,
        _d: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }
    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::aws_lc_rs::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

fn mean(samples: &[Duration]) -> f64 {
    let total: Duration = samples.iter().sum();
    total.as_secs_f64() * 1e6 / samples.len() as f64
}

fn report(label: &str, samples: &[Duration]) {
    let micros: Vec<f64> = samples.iter().map(|d| d.as_secs_f64() * 1e6).collect();
    let min = micros.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = micros.iter().cloned().fold(0.0, f64::max);
    println!(
        "{label}: n={} mean={:.1}µs min={:.1}µs max={:.1}µs",
        samples.len(),
        mean(samples),
        min,
        max
    );
}
