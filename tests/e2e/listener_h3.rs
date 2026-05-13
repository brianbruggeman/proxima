//! End-to-end test for the HTTP/3 listener. Server side is the
//! native proxima::h3 driver mounted via `H3ListenProtocol` with a
//! self-signed cert. Client uses the `h3` + `h3-quinn` pair driving
//! a single GET round-trip — no hyper anywhere.

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
use futures::channel::oneshot;
use proxima::error::ProximaError;
use proxima::listen::ListenProtocol;
use proxima::listeners::H3ListenProtocol;
use proxima::pipe::{into_handle};
use proxima::request::{Request, Response};
use proxima::telemetry::NoopTelemetry;
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
async fn h3_listener_round_trip() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn,h3=trace,quinn=info,proxima=debug".into()),
        )
        .with_test_writer()
        .try_init();
    let _ = quinn::rustls::crypto::aws_lc_rs::default_provider().install_default();

    let dispatch = into_handle(ConstantOk);
    let bind: SocketAddr = (Ipv4Addr::LOCALHOST, 0).into();

    // ephemeral-port discovery: bind via the same dev helper the
    // listener uses, capture the assigned port, then close the
    // probe endpoint so the listener can take it. there's no
    // accept on `H3ListenProtocol::serve` that returns the bound
    // port, so we hand it an explicit port up front.
    let probe_config =
        proxima::quic::dev_server_config(vec!["localhost".to_string()], &[b"h3"]).unwrap();
    let probe = proxima::quic::Endpoint::server(bind, probe_config).unwrap();
    let bound = probe.local_addr().unwrap();
    drop(probe);

    let spec = serde_json::json!({
        "dev_self_signed": true,
        "dev_sans": ["localhost"],
    });
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let protocol = H3ListenProtocol::new();
    let context = proxima::listen::ServeContext::new(Arc::new(NoopTelemetry));

    let server = {
        let dispatch = dispatch.clone();
        let spec = spec.clone();
        async move {
            protocol
                .serve(bound, dispatch, &spec, context, shutdown_rx)
                .await
        }
    };
    let server_handle = tokio::spawn(server);

    // give the listener a moment to bind. without this the client
    // attempt can race the bind on a cold runtime.
    tokio::time::sleep(Duration::from_millis(200)).await;

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
        // ConnectionError doesn't matter here — the test ends on
        // body received, then driver returns when the server closes.
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

    let _ = shutdown_tx.send(());
    let _ = tokio::time::timeout(Duration::from_secs(2), server_handle).await;
    client_endpoint.close(0u32.into(), b"done");
    let _ = tokio::time::timeout(Duration::from_secs(1), driver_task).await;
}

/// Mirrors what `predicate.and_then(inner)` produces at the pipe edge:
/// `/reject` returns the exact `ProximaError::Forbidden` a filter's
/// `RejectMode::Drop` emits, `/internal` returns a genuinely internal
/// failure, everything else is admitted. One pipe drives every
/// filter-over-h3 regression scenario below.
struct FilterRoutedPipe;

impl SendPipe for FilterRoutedPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move {
            match request.path.as_ref() {
                b"/reject" => Err(ProximaError::Forbidden("blocked by filter".into())),
                b"/internal" => Err(ProximaError::Upstream("boom".into())),
                _ => Ok(Response::ok(Bytes::from_static(b"ok"))),
            }
        }
    }
}

async fn drain_body(stream: &mut h3::client::RequestStream<h3_quinn::BidiStream<Bytes>, Bytes>) -> Vec<u8> {
    let mut body = bytes::BytesMut::new();
    while let Some(mut chunk) = stream.recv_data().await.unwrap() {
        while bytes::Buf::has_remaining(&chunk) {
            let bytes = bytes::Buf::chunk(&chunk);
            body.extend_from_slice(bytes);
            let advance = bytes.len();
            bytes::Buf::advance(&mut chunk, advance);
        }
    }
    body.to_vec()
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn h3_listener_filter_rejection_renders_403_and_connection_survives() {
    // Regression for the h3 filter-rejection bug (proxima-http/src/http3/server.rs):
    // the handler future's `Err` used to propagate out of the pushed
    // task into the connection driver's `result?`, which could tear
    // down the whole multiplexed QUIC connection over one rejected
    // request. A rejection must render as a real 403, and a later
    // request on the SAME connection must still succeed.
    let _ = quinn::rustls::crypto::aws_lc_rs::default_provider().install_default();

    let dispatch = into_handle(FilterRoutedPipe);
    let bind: SocketAddr = (Ipv4Addr::LOCALHOST, 0).into();

    let probe_config =
        proxima::quic::dev_server_config(vec!["localhost".to_string()], &[b"h3"]).unwrap();
    let probe = proxima::quic::Endpoint::server(bind, probe_config).unwrap();
    let bound = probe.local_addr().unwrap();
    drop(probe);

    let spec = serde_json::json!({
        "dev_self_signed": true,
        "dev_sans": ["localhost"],
    });
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let protocol = H3ListenProtocol::new();
    let context = proxima::listen::ServeContext::new(Arc::new(NoopTelemetry));

    let server = {
        let dispatch = dispatch.clone();
        let spec = spec.clone();
        async move {
            protocol
                .serve(bound, dispatch, &spec, context, shutdown_rx)
                .await
        }
    };
    let server_handle = tokio::spawn(server);

    tokio::time::sleep(Duration::from_millis(200)).await;

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

    let rejected = http::Request::builder()
        .method("GET")
        .uri(format!("https://localhost:{}/reject", bound.port()))
        .body(())
        .unwrap();
    let mut rejected_stream = send_request.send_request(rejected).await.unwrap();
    rejected_stream.finish().await.unwrap();
    let response = rejected_stream.recv_response().await.unwrap();
    assert_eq!(response.status(), 403, "filter rejection renders as 403");
    assert_eq!(&drain_body(&mut rejected_stream).await[..], b"blocked by filter");

    // Same connection, next request: proves the rejection didn't
    // kill the underlying QUIC connection — the h3 bug's actual
    // signature.
    let ok = http::Request::builder()
        .method("GET")
        .uri(format!("https://localhost:{}/ok", bound.port()))
        .body(())
        .unwrap();
    let mut ok_stream = send_request.send_request(ok).await.unwrap();
    ok_stream.finish().await.unwrap();
    let ok_response = ok_stream
        .recv_response()
        .await
        .expect("connection must survive a prior filter rejection");
    assert_eq!(ok_response.status(), 200);
    assert_eq!(&drain_body(&mut ok_stream).await[..], b"ok");

    let _ = shutdown_tx.send(());
    let _ = tokio::time::timeout(Duration::from_secs(2), server_handle).await;
    client_endpoint.close(0u32.into(), b"done");
    let _ = tokio::time::timeout(Duration::from_secs(1), driver_task).await;
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn h3_listener_internal_error_resets_stream_and_connection_survives() {
    // Unchanged-behaviour regression: a genuinely internal (non-filter)
    // handler error must NOT render a response — it resets just its
    // own stream (the h3 analogue of h2's RST_STREAM) — and, exactly
    // like the filter-rejection case, must not take the connection
    // down with it.
    let _ = quinn::rustls::crypto::aws_lc_rs::default_provider().install_default();

    let dispatch = into_handle(FilterRoutedPipe);
    let bind: SocketAddr = (Ipv4Addr::LOCALHOST, 0).into();

    let probe_config =
        proxima::quic::dev_server_config(vec!["localhost".to_string()], &[b"h3"]).unwrap();
    let probe = proxima::quic::Endpoint::server(bind, probe_config).unwrap();
    let bound = probe.local_addr().unwrap();
    drop(probe);

    let spec = serde_json::json!({
        "dev_self_signed": true,
        "dev_sans": ["localhost"],
    });
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let protocol = H3ListenProtocol::new();
    let context = proxima::listen::ServeContext::new(Arc::new(NoopTelemetry));

    let server = {
        let dispatch = dispatch.clone();
        let spec = spec.clone();
        async move {
            protocol
                .serve(bound, dispatch, &spec, context, shutdown_rx)
                .await
        }
    };
    let server_handle = tokio::spawn(server);

    tokio::time::sleep(Duration::from_millis(200)).await;

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

    let internal = http::Request::builder()
        .method("GET")
        .uri(format!("https://localhost:{}/internal", bound.port()))
        .body(())
        .unwrap();
    let mut internal_stream = send_request.send_request(internal).await.unwrap();
    internal_stream.finish().await.unwrap();
    let response_result = internal_stream.recv_response().await;
    assert!(
        response_result.is_err(),
        "internal error must reset the stream, not render a response: {response_result:?}"
    );

    // Same connection, next request still succeeds — the internal
    // error must not have taken the whole connection down either.
    let ok = http::Request::builder()
        .method("GET")
        .uri(format!("https://localhost:{}/ok", bound.port()))
        .body(())
        .unwrap();
    let mut ok_stream = send_request.send_request(ok).await.unwrap();
    ok_stream.finish().await.unwrap();
    let ok_response = ok_stream
        .recv_response()
        .await
        .expect("connection must survive an internal error on another stream");
    assert_eq!(ok_response.status(), 200);
    assert_eq!(&drain_body(&mut ok_stream).await[..], b"ok");

    let _ = shutdown_tx.send(());
    let _ = tokio::time::timeout(Duration::from_secs(2), server_handle).await;
    client_endpoint.close(0u32.into(), b"done");
    let _ = tokio::time::timeout(Duration::from_secs(1), driver_task).await;
}
