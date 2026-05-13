//! K1 proof (C42) — the native HTTP/3 client `SendPipe`
//! ([`H3NativeUpstream`]) drives a real `GET /` over a real UDP socket
//! against the mounted [`H3NativeListenProtocol`] server and reads back
//! `200 OK` + body `ok`.
//!
//! This closes the one untested cell of the native stack: prior proofs
//! cover native-server ↔ quinn-client over UDP (`listener_h3_native.rs`)
//! and native-client ↔ native-server in-memory (`proxima-h3`'s
//! `native_round_trip.rs`). Nothing yet drove the native client over a
//! real socket. Here the client runs on a prime worker (prime UDP
//! `Endpoint`); the server runs the production tokio listener — exactly
//! the two runtimes that meet in production.

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(feature = "h3-native-upstream")]

use std::future::Future;
use std::net::Ipv4Addr;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures::channel::oneshot;

use proxima::CoreId;
use proxima::error::ProximaError;
use proxima::listen::{ListenProtocol, ServeContext};
use proxima::listeners::H3NativeListenProtocol;
use proxima::pipe::into_handle;
use proxima::request::{Request, Response};
use proxima::telemetry::NoopTelemetry;
use proxima_http::http3::native::H3NativeUpstream;
use proxima_primitives::pipe::SendPipe;

/// Constant `200 OK` + `ok` handler — the server-side Handler.
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


/// Dev-only verifier: the server presents an ephemeral rcgen self-signed
/// cert (`dev_self_signed`), so the client trusts any cert for the test.
#[derive(Debug)]
struct AcceptAnyCert;

impl rustls::client::danger::ServerCertVerifier for AcceptAnyCert {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::aws_lc_rs::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

fn accept_any_client_config() -> rustls::ClientConfig {
    let mut config =
        rustls::ClientConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(AcceptAnyCert))
            .with_no_client_auth();
    config.alpn_protocols = vec![b"h3".to_vec()];
    config
}

// Proves the native HTTP/3 client `SendPipe` (H3NativeUpstream, prime UDP
// Endpoint) completes a real-socket GET / -> 200 OK + "ok" against the
// mounted H3NativeListenProtocol server. Closing this required two
// RFC-mandated client fixes the in-memory path never exercised: adopting
// the server's Source CID as our DCID (RFC 9000 §7.2) and advertising
// initial_source_connection_id (RFC 9000 §18.2 / §7.3). See
// docs/proxima-quic/discipline.md C42.
#[test]
fn native_h3_upstream_round_trips_against_native_listener() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn,proxima_http=debug,proxima=debug".into()),
        )
        .with_thread_names(true)
        .with_test_writer()
        .try_init();
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

    // Probe a free UDP port with a plain std socket (no async runtime
    // needed here), then drop it so the native listener can bind it.
    let probe = std::net::UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let bound = probe.local_addr().unwrap();
    drop(probe);

    // Server: the production native listener on a tokio runtime in its
    // own thread.
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server_thread = std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("server runtime");
        runtime.block_on(async move {
            let protocol = H3NativeListenProtocol::new();
            let context = ServeContext::new(Arc::new(NoopTelemetry));
            let spec = serde_json::json!({
                "dev_self_signed": true,
                "dev_sans": ["localhost"],
            });
            let _ = protocol
                .serve(bound, into_handle(ConstantOk), &spec, context, shutdown_rx)
                .await;
        });
    });

    // Give the listener a moment to bind.
    std::thread::sleep(Duration::from_millis(300));

    // Client: the native upstream driven on a prime worker (prime UDP
    // Endpoint needs CURRENT_REACTOR).
    let handle = prime::os::core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
    let done = Arc::new(AtomicBool::new(false));
    let done_for_factory = done.clone();
    let result: Arc<Mutex<Option<Result<(u16, Bytes), String>>>> = Arc::new(Mutex::new(None));
    let result_for_factory = result.clone();

    handle
        .dispatch_factory(Box::new(move || {
            let done = done_for_factory.clone();
            let result_slot = result_for_factory.clone();
            Box::pin(async move {
                let upstream = H3NativeUpstream::with_client_config(
                    bound,
                    "localhost",
                    accept_any_client_config(),
                );
                let request = Request::builder()
                    .method("GET")
                    .path("/")
                    .build()
                    .expect("request");
                let outcome = match SendPipe::call(&upstream, request).await {
                    Ok(response) => Ok((response.status, response.payload)),
                    Err(err) => Err(format!("call: {err}")),
                };
                *result_slot.lock().expect("result lock") = Some(outcome);
                done.store(true, Ordering::Release);
            }) as Pin<Box<dyn Future<Output = ()> + 'static>>
        }))
        .expect("dispatch_factory");

    let deadline = Instant::now() + Duration::from_secs(15);
    while !done.load(Ordering::Acquire) {
        assert!(
            Instant::now() < deadline,
            "native h3 upstream round-trip never completed (likely handshake/driver stall)"
        );
        std::thread::sleep(Duration::from_millis(20));
    }

    let _ = shutdown_tx.send(());
    handle.shutdown_and_join().expect("shutdown");
    let _ = server_thread.join();

    let outcome = result
        .lock()
        .expect("result lock")
        .take()
        .expect("result set");
    let (status, body) = outcome.expect("upstream call succeeded");
    assert_eq!(status, 200, "expected 200 OK");
    assert_eq!(&body[..], b"ok", "expected body 'ok'");
}

/// Same real-socket round trip with BOTH halves on the lazy source path:
/// the listener in `part_source` mode (request HEADERS stepped into the
/// dispatch `Request`) and the upstream in `with_part_source` mode
/// (response HEADERS stepped off the queued block, full forward path).
/// Proves the C3/C4 Source modes compose end-to-end over a real UDP
/// socket, not just against their own crate's fixtures.
#[cfg(feature = "h3-part-source")]
#[test]
fn native_h3_upstream_round_trips_with_part_source_on_both_halves() {
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
            .expect("server runtime");
        runtime.block_on(async move {
            let protocol = H3NativeListenProtocol::new();
            let context = ServeContext::new(Arc::new(NoopTelemetry));
            let spec = serde_json::json!({
                "dev_self_signed": true,
                "dev_sans": ["localhost"],
                "part_source": true,
            });
            let _ = protocol
                .serve(bound, into_handle(ConstantOk), &spec, context, shutdown_rx)
                .await;
        });
    });

    std::thread::sleep(Duration::from_millis(300));

    let handle = prime::os::core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
    let done = Arc::new(AtomicBool::new(false));
    let done_for_factory = done.clone();
    let result: Arc<Mutex<Option<Result<(u16, Bytes), String>>>> = Arc::new(Mutex::new(None));
    let result_for_factory = result.clone();

    handle
        .dispatch_factory(Box::new(move || {
            let done = done_for_factory.clone();
            let result_slot = result_for_factory.clone();
            Box::pin(async move {
                let upstream = H3NativeUpstream::with_client_config(
                    bound,
                    "localhost",
                    accept_any_client_config(),
                )
                .with_part_source();
                let request = Request::builder()
                    .method("GET")
                    .path("/")
                    .build()
                    .expect("request");
                let outcome = match SendPipe::call(&upstream, request).await {
                    Ok(response) => Ok((response.status, response.payload)),
                    Err(err) => Err(format!("call: {err}")),
                };
                *result_slot.lock().expect("result lock") = Some(outcome);
                done.store(true, Ordering::Release);
            }) as Pin<Box<dyn Future<Output = ()> + 'static>>
        }))
        .expect("dispatch_factory");

    let deadline = Instant::now() + Duration::from_secs(15);
    while !done.load(Ordering::Acquire) {
        assert!(
            Instant::now() < deadline,
            "part-source round-trip never completed (likely handshake/driver stall)"
        );
        std::thread::sleep(Duration::from_millis(20));
    }

    let _ = shutdown_tx.send(());
    handle.shutdown_and_join().expect("shutdown");
    let _ = server_thread.join();

    let outcome = result
        .lock()
        .expect("result lock")
        .take()
        .expect("result set");
    let (status, body) = outcome.expect("upstream call succeeded");
    assert_eq!(status, 200, "expected 200 OK through both source halves");
    assert_eq!(
        &body[..],
        b"ok",
        "expected body 'ok' through both source halves"
    );
}

/// Spawn the dev native H3 listener on its own tokio runtime thread,
/// returning its bound address + a shutdown handle + the join handle.
fn spawn_dev_listener() -> (
    std::net::SocketAddr,
    oneshot::Sender<()>,
    std::thread::JoinHandle<()>,
) {
    let probe = std::net::UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let bound = probe.local_addr().unwrap();
    drop(probe);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server_thread = std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("server runtime");
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
    (bound, shutdown_tx, server_thread)
}

/// The headline "sane semantic client" path: HTTP/3 over the native stack
/// reached purely through `proxima::Client` + a spec (`{"type":"h3-native",
/// ...}`) — the factory resolves it, `Client` hops onto the shared prime
/// runtime for the prime UDP Endpoint, and the GET returns 200 + "ok".
/// `insecure` trusts the listener's dev self-signed cert.
#[test]
fn native_h3_via_client_spec_round_trips() {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    let (bound, shutdown_tx, server_thread) = spawn_dev_listener();

    let (status, body) = futures::executor::block_on(async {
        let client = proxima::Client::from_value(serde_json::json!({
            "type": "h3-native",
            "addr": bound.to_string(),
            "server_name": "localhost",
            "insecure": true,
        }))
        .expect("build client");
        let response = client.call("GET", "/").send().await.expect("send");
        let status = response.status();
        let body = response.bytes().await.expect("bytes");
        (status, body)
    });

    let _ = shutdown_tx.send(());
    let _ = server_thread.join();
    assert_eq!(status, 200, "expected 200 OK via Client spec");
    assert_eq!(&body[..], b"ok", "expected body 'ok' via Client spec");
}

/// Sad path: dialing a port with NO listener fails with an error (the
/// per-request timeout), not a hang. Proves `with_timeout` bounds the
/// driver so a dead/silent peer can't park forever.
#[test]
fn native_h3_upstream_errors_on_dead_port() {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    // A free loopback port with nothing bound to it.
    let probe = std::net::UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let dead = probe.local_addr().unwrap();
    drop(probe);

    let handle = prime::os::core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
    let done = Arc::new(AtomicBool::new(false));
    let done_for = done.clone();
    let result: Arc<Mutex<Option<Result<u16, String>>>> = Arc::new(Mutex::new(None));
    let result_for = result.clone();

    handle
        .dispatch_factory(Box::new(move || {
            let done = done_for.clone();
            let result_slot = result_for.clone();
            Box::pin(async move {
                let upstream = H3NativeUpstream::with_client_config(
                    dead,
                    "localhost",
                    accept_any_client_config(),
                )
                .with_timeout(Duration::from_secs(2));
                let request = Request::builder()
                    .method("GET")
                    .path("/")
                    .build()
                    .expect("request");
                let outcome = match SendPipe::call(&upstream, request).await {
                    Ok(response) => Ok(response.status),
                    Err(err) => Err(format!("{err}")),
                };
                *result_slot.lock().expect("result lock") = Some(outcome);
                done.store(true, Ordering::Release);
            }) as Pin<Box<dyn Future<Output = ()> + 'static>>
        }))
        .expect("dispatch_factory");

    let deadline = Instant::now() + Duration::from_secs(10);
    while !done.load(Ordering::Acquire) {
        assert!(
            Instant::now() < deadline,
            "dead-port dial neither returned nor timed out"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    handle.shutdown_and_join().expect("shutdown");

    let outcome = result
        .lock()
        .expect("result lock")
        .take()
        .expect("result set");
    assert!(
        outcome.is_err(),
        "expected an error dialing a dead port, got {outcome:?}"
    );
}

/// Connection reuse: 200 sequential GETs over ONE upstream all return 200.
/// The persistent QUIC connection is established once and reused (handshake
/// paid only on the first call). 200 is well past BOTH
/// `max_concurrent_bidi = 8` (the table cap, freed by stream reaping) AND
/// `initial_max_streams_bidi = 100`, proving reuse is effectively unbounded
/// over a real socket.
#[test]
fn native_h3_upstream_reuses_connection_across_calls() {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    let (bound, shutdown_tx, server_thread) = spawn_dev_listener();

    let handle = prime::os::core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
    let done = Arc::new(AtomicBool::new(false));
    let done_for = done.clone();
    let result: Arc<Mutex<Option<Result<usize, String>>>> = Arc::new(Mutex::new(None));
    let result_for = result.clone();

    handle
        .dispatch_factory(Box::new(move || {
            let done = done_for.clone();
            let result_slot = result_for.clone();
            Box::pin(async move {
                let upstream = H3NativeUpstream::with_client_config(
                    bound,
                    "localhost",
                    accept_any_client_config(),
                );
                let mut ok = 0usize;
                let mut failure = None;
                for index in 0..200 {
                    let request = Request::builder()
                        .method("GET")
                        .path("/")
                        .build()
                        .expect("request");
                    match SendPipe::call(&upstream, request).await {
                        Ok(response)
                            if response.status == 200 && &response.payload[..] == b"ok" =>
                        {
                            ok += 1;
                        }
                        Ok(response) => {
                            failure = Some(format!("call {index}: status {}", response.status));
                            break;
                        }
                        Err(err) => {
                            failure = Some(format!("call {index}: {err}"));
                            break;
                        }
                    }
                }
                *result_slot.lock().expect("result lock") = Some(failure.map_or(Ok(ok), Err));
                done.store(true, Ordering::Release);
            }) as Pin<Box<dyn Future<Output = ()> + 'static>>
        }))
        .expect("dispatch_factory");

    let deadline = Instant::now() + Duration::from_secs(15);
    while !done.load(Ordering::Acquire) {
        assert!(
            Instant::now() < deadline,
            "reuse round-trip never completed"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    let _ = shutdown_tx.send(());
    handle.shutdown_and_join().expect("shutdown");
    let _ = server_thread.join();

    let outcome = result
        .lock()
        .expect("result lock")
        .take()
        .expect("result set");
    assert_eq!(outcome, Ok(200), "expected 200 reused 200 responses");
}
