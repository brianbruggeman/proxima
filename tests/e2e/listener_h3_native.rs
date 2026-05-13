//! End-to-end test for the NATIVE HTTP/3 listener.
//!
//! Mirrors [`tests/listener_h3.rs`] but mounts
//! [`proxima::listeners::H3NativeListenProtocol`] on the server side —
//! the native sans-IO QUIC + H3 stack with no quinn, no h3-quinn.
//! Client still uses h3 + h3-quinn (legacy) to prove the native stack
//! is wire-compatible with the incumbent.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
#![cfg(feature = "http3")]

use std::future::Future;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use proxima::error::ProximaError;
use proxima::listeners::H3NativeListenProtocol;
use proxima::pipe::{into_handle};
use proxima::request::{Request, Response};
use proxima::runtime::{PrimeRuntime, Runtime};
use proxima::telemetry::NoopTelemetry;
use proxima::{ListenRegistry, ListenerSpec};
use proxima_net::prime::PrimeDatagramFactory;
use proxima_primitives::pipe::SendPipe;

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

fn client_config() -> quinn::ClientConfig {
    let mut tls = quinn::rustls::ClientConfig::builder_with_protocol_versions(&[
        &quinn::rustls::version::TLS13,
    ])
    .dangerous()
    .with_custom_certificate_verifier(Arc::new(AcceptAnyCert))
    .with_no_client_auth();
    tls.alpn_protocols = vec![b"h3".to_vec()];
    let crypto = quinn::crypto::rustls::QuicClientConfig::try_from(tls).unwrap();
    quinn::ClientConfig::new(Arc::new(crypto))
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn h3_native_listener_round_trip() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn,proxima=debug".into()),
        )
        .with_test_writer()
        .try_init();
    let _ = quinn::rustls::crypto::aws_lc_rs::default_provider().install_default();

    let dispatch = into_handle(ConstantOk);

    // Grab a free UDP port with a plain std socket (closes SYNCHRONOUSLY on
    // drop, unlike an async endpoint whose driver lingers), then hand the
    // vacated port to the native listener.
    let probe = std::net::UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let bound: SocketAddr = probe.local_addr().unwrap();
    drop(probe);

    // The native h3 listener binds its UDP socket through a prime-reactor-backed
    // DatagramFactory (CURRENT_REACTOR must be live), so it runs on a
    // PrimeRuntime; the quinn client below runs on this test's tokio runtime.
    // They bridge over real localhost UDP — the true independent-client interop
    // proof the in-memory native_round_trip test can't give.
    let runtime = Arc::new(PrimeRuntime::new(1).unwrap());
    let registry = ListenRegistry::new();
    registry
        .register(Arc::new(H3NativeListenProtocol::new()))
        .unwrap();
    let runtime_dyn: Arc<dyn Runtime> = runtime.clone();
    let mut listener_spec = ListenerSpec::http(bound);
    listener_spec.protocol_name = "h3-native".into();
    let listener_spec = listener_spec.with_spec(serde_json::json!({
        "dev_self_signed": true,
        "dev_sans": ["localhost"],
    }));
    let server_handle = listener_spec
        .attach(dispatch)
        .run_with_runtime(
            &registry,
            NoopTelemetry::handle(),
            Some(runtime_dyn),
            None,
            Some(Arc::new(PrimeDatagramFactory)),
        )
        .unwrap();

    // No bind-wait sleep: quinn PTO-retransmits its Initial for up to 30s, so a
    // server that finishes binding a beat later still catches the retransmit.

    let mut client_endpoint = quinn::Endpoint::client((Ipv4Addr::UNSPECIFIED, 0).into()).unwrap();
    client_endpoint.set_default_client_config(client_config());

    let connecting = client_endpoint.connect(bound, "localhost").unwrap();
    let connection = connecting.await.unwrap();

    let h3_conn = h3_quinn::Connection::new(connection);
    let (mut driver, mut send_request) = h3::client::builder()
        .build::<_, _, Bytes>(h3_conn)
        .await
        .unwrap();

    let driver_task = tokio::spawn(async move {
        let _ = std::future::poll_fn(|cx| driver.poll_close(cx)).await;
    });

    let req = http::Request::builder()
        .method("GET")
        .uri(format!("https://localhost:{}/", bound.port()))
        .body(())
        .unwrap();
    let mut stream = send_request.send_request(req).await.unwrap();
    stream.finish().await.unwrap();

    let response = stream.recv_response().await.unwrap();
    assert_eq!(response.status(), 200);

    let mut body = bytes::BytesMut::new();
    while let Some(mut chunk) = stream.recv_data().await.unwrap() {
        while bytes::Buf::has_remaining(&chunk) {
            let bytes = bytes::Buf::chunk(&chunk);
            body.extend_from_slice(bytes);
            let advance = bytes.len();
            bytes::Buf::advance(&mut chunk, advance);
        }
    }
    assert_eq!(&body[..], b"ok");

    server_handle.shutdown();
    client_endpoint.close(0u32.into(), b"done");
    let _ = tokio::time::timeout(Duration::from_secs(1), driver_task).await;
}
