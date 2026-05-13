#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
#![cfg(feature = "h3-upstream")]

//! P7 — HTTP/3 upstream microbench. Apples-to-apples: both arms hit
//! the same in-process h3 echo server over the same self-signed
//! cert, and both arms reuse one persistent QUIC connection (the
//! per-call cost is opening a new bidi stream + req/resp round-trip).
//!
//! Arms:
//!
//! - `proxima_h3_upstream` — `Http3Upstream::call(Request)` with a
//!   pre-warmed connection. Substrate cost = Request → http::Request
//!   translation + body drain + Response build.
//! - `parity_h3_quinn` — direct `h3::client::SendRequest::send_request`
//!   doing the same scope: build `http::Request`, send_data, finish,
//!   recv_response, recv_data loop, return body Bytes. No Pipe
//!   trait, no Request/Response translation. Same SendRequest is
//!   reused; same connection.
//!
//! Both arms ignore the actual response payload — black_box on the
//! returned bytes only. The bench measures one round-trip per
//! iteration over QUIC stream 0 → stream N.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use bytes::{Buf, Bytes};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};
use tokio::runtime::Builder;

use proxima::SendPipe;
use proxima::request::{Request, RequestContext};
use proxima::upstreams::Http3Upstream;

const REQUEST_PAYLOAD: &[u8] = b"hello-from-proxima-h3-bench";

fn make_self_signed() -> (Vec<CertificateDer<'static>>, PrivateKeyDer<'static>) {
    let names = vec!["localhost".to_string(), "127.0.0.1".to_string()];
    let generated = rcgen::generate_simple_self_signed(names).expect("rcgen");
    let cert = CertificateDer::from(generated.cert.der().to_vec());
    let key = PrivateKeyDer::Pkcs8(generated.signing_key.serialize_der().into());
    (vec![cert], key)
}

async fn spawn_h3_echo_server(
    cert_chain: Vec<CertificateDer<'static>>,
    key: PrivateKeyDer<'static>,
) -> SocketAddr {
    let mut server_crypto = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(cert_chain, key)
        .expect("rustls server cfg");
    server_crypto.alpn_protocols = vec![b"h3".to_vec()];

    let quic_cfg =
        quinn::crypto::rustls::QuicServerConfig::try_from(server_crypto).expect("quic server cfg");
    let server_cfg = quinn::ServerConfig::with_crypto(Arc::new(quic_cfg));

    let bind: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let endpoint = quinn::Endpoint::server(server_cfg, bind).expect("endpoint");
    let local = endpoint.local_addr().expect("local addr");

    tokio::spawn(async move {
        while let Some(incoming) = endpoint.accept().await {
            tokio::spawn(async move {
                let connection = match incoming.await {
                    Ok(conn) => conn,
                    Err(_) => return,
                };
                let h3_conn = h3_quinn::Connection::new(connection);
                let mut server = match h3::server::builder().build(h3_conn).await {
                    Ok(server) => server,
                    Err(_) => return,
                };
                loop {
                    let resolver = match server.accept().await {
                        Ok(Some(resolver)) => resolver,
                        Ok(None) => return,
                        Err(_) => return,
                    };
                    tokio::spawn(async move {
                        let (_req, mut stream) = match resolver.resolve_request().await {
                            Ok(pair) => pair,
                            Err(_) => return,
                        };
                        let mut body = bytes::BytesMut::new();
                        while let Ok(Some(mut chunk)) = stream.recv_data().await {
                            let slice = chunk.chunk();
                            body.extend_from_slice(slice);
                            let len = slice.len();
                            chunk.advance(len);
                        }
                        let resp = http::Response::builder().status(200).body(()).unwrap();
                        if stream.send_response(resp).await.is_err() {
                            return;
                        }
                        if !body.is_empty() {
                            let _ = stream.send_data(body.freeze()).await;
                        }
                        let _ = stream.finish().await;
                    });
                }
            });
        }
    });

    local
}

#[derive(Debug)]
struct AcceptAnyServer;

impl rustls::client::danger::ServerCertVerifier for AcceptAnyServer {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::RSA_PSS_SHA256,
            rustls::SignatureScheme::ED25519,
        ]
    }
}

fn make_client_config() -> rustls::ClientConfig {
    let mut cfg = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(AcceptAnyServer))
        .with_no_client_auth();
    cfg.alpn_protocols = vec![b"h3".to_vec()];
    cfg
}

async fn build_client(
    server_addr: SocketAddr,
) -> (
    quinn::Endpoint,
    h3::client::SendRequest<h3_quinn::OpenStreams, Bytes>,
) {
    let tls = make_client_config();
    let quic_cfg = quinn::crypto::rustls::QuicClientConfig::try_from(tls).expect("quic cfg");
    let client_cfg = quinn::ClientConfig::new(Arc::new(quic_cfg));

    let mut endpoint = quinn::Endpoint::client("127.0.0.1:0".parse().unwrap()).expect("endpoint");
    endpoint.set_default_client_config(client_cfg);

    let connection = endpoint
        .connect(server_addr, "localhost")
        .expect("connect")
        .await
        .expect("handshake");
    let h3_conn = h3_quinn::Connection::new(connection);
    let (mut driver, send_request) = h3::client::new(h3_conn).await.expect("h3 init");

    tokio::spawn(async move {
        let _ = futures::future::poll_fn(|cx| driver.poll_close(cx)).await;
    });

    (endpoint, send_request)
}

fn proxima_arm(criterion: &mut Criterion) {
    let runtime = Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("runtime");

    let upstream: Arc<Http3Upstream> = runtime.block_on(async {
        let (cert_chain, key) = make_self_signed();
        let server_addr = spawn_h3_echo_server(cert_chain, key).await;
        let upstream = Arc::new(Http3Upstream::with_client_config(
            server_addr,
            "localhost",
            make_client_config(),
        ));
        let warmup = Request {
            method: Bytes::from_static(b"POST").into(),
            path: Bytes::from_static(b"/echo"),
            query: proxima::header_list::HeaderList::new(),
            metadata: proxima::header_list::HeaderList::new(),
            payload: Bytes::from_static(REQUEST_PAYLOAD),
            stream: None,
            context: RequestContext::default(),
        };
        upstream.call(warmup).await.expect("warmup");
        upstream
    });

    let mut group = criterion.benchmark_group("h3_upstream_proxima");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));
    group.bench_function("call", |bencher| {
        bencher.to_async(&runtime).iter(|| {
            let upstream = Arc::clone(&upstream);
            async move {
                let request = Request {
                    method: Bytes::from_static(b"POST").into(),
                    path: Bytes::from_static(b"/echo"),
                    query: proxima::header_list::HeaderList::new(),
                    metadata: proxima::header_list::HeaderList::new(),
                    payload: Bytes::from_static(REQUEST_PAYLOAD),
                    stream: None,
                    context: RequestContext::default(),
                };
                let response = upstream.call(request).await.expect("call");
                std::hint::black_box(response);
            }
        });
    });
    group.finish();
    drop(upstream);
    runtime.shutdown_timeout(Duration::from_millis(50));
}

fn parity_arm(criterion: &mut Criterion) {
    let runtime = Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("runtime");

    // Mirror proxima's lazy-init shape: Mutex<Option<SendRequest>>.
    // Apples-to-apples — both arms pay the same Mutex acquire + clone
    // on the hot path. The remaining gap (if any) is then the Pipe
    // trait + Body abstraction overhead, which is what the comparison
    // is supposed to measure.
    let state: Arc<
        tokio::sync::Mutex<Option<h3::client::SendRequest<h3_quinn::OpenStreams, Bytes>>>,
    > = runtime.block_on(async {
        let (cert_chain, key) = make_self_signed();
        let server_addr = spawn_h3_echo_server(cert_chain, key).await;
        let (_endpoint, send_request) = build_client(server_addr).await;
        // Pre-warm so the bench loop only measures stream open + rt.
        let mut warm_send = send_request.clone();
        let warm_req = http::Request::builder()
            .method("POST")
            .uri("https://localhost/echo")
            .body(())
            .unwrap();
        let mut stream = warm_send.send_request(warm_req).await.unwrap();
        stream
            .send_data(Bytes::from_static(REQUEST_PAYLOAD))
            .await
            .unwrap();
        stream.finish().await.unwrap();
        let _ = stream.recv_response().await.unwrap();
        while let Ok(Some(mut chunk)) = stream.recv_data().await {
            let len = chunk.chunk().len();
            chunk.advance(len);
        }
        Arc::new(tokio::sync::Mutex::new(Some(send_request)))
    });
    let send_request = state;

    let mut group = criterion.benchmark_group("h3_upstream_parity");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));
    // Parity mirrors what a hand-rolled h3 client would do per call:
    // take method + path + body in, format the URI, build the request.
    // Matches proxima's per-call work exactly so the comparison is
    // apples-to-apples (same alloc, same UTF-8 validation, same builder
    // work, same body Bytes round-trip).
    let server_name = "localhost".to_string();
    let server_name = Arc::new(server_name);
    group.bench_function("call", |bencher| {
        bencher.to_async(&runtime).iter(|| {
            let send_request = Arc::clone(&send_request);
            let server_name = Arc::clone(&server_name);
            async move {
                let method_bytes = Bytes::from_static(b"POST");
                let path_bytes = Bytes::from_static(b"/echo");
                let payload = Bytes::from_static(REQUEST_PAYLOAD);
                let method = std::str::from_utf8(method_bytes.as_ref()).expect("utf8");
                let path = std::str::from_utf8(path_bytes.as_ref()).expect("utf8");
                let uri = format!("https://{server_name}{path}");
                let http_req = http::Request::builder()
                    .method(method)
                    .uri(&uri)
                    .body(())
                    .expect("build");
                let mut send_request: h3::client::SendRequest<h3_quinn::OpenStreams, Bytes> = {
                    let guard = send_request.lock().await;
                    guard.as_ref().expect("send_request initialized").clone()
                };
                let mut stream = send_request.send_request(http_req).await.expect("send");
                stream.send_data(payload).await.expect("send_data");
                stream.finish().await.expect("finish");
                let _response = stream.recv_response().await.expect("recv resp");
                let mut body = bytes::BytesMut::new();
                while let Some(mut chunk) = stream.recv_data().await.expect("recv data") {
                    let slice = chunk.chunk();
                    body.extend_from_slice(slice);
                    let len = slice.len();
                    chunk.advance(len);
                }
                std::hint::black_box(body);
            }
        });
    });
    group.finish();
    drop(send_request);
    runtime.shutdown_timeout(Duration::from_millis(50));
}

fn benches(criterion: &mut Criterion) {
    proxima_arm(criterion);
    parity_arm(criterion);
}

criterion_group!(group, benches);
criterion_main!(group);
