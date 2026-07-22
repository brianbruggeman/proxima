#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
#![cfg(all(feature = "http2", feature = "tls"))]

//! Three-way h2 comparison on warm connections: proxima's
//! `serve_h2_connection` vs hyper's `http2::Builder` vs Cloudflare's
//! Pingora. All three driven by the same h2 client over loopback TCP
//! using h2 prior-knowledge (no TLS / ALPN). One handshake per server
//! at bench setup; iter measures per-request cost on the warm
//! connection — h2's design point.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use http_body_util::Full;
use hyper::server::conn::http2;
use hyper::service::service_fn as pipe_fn;
use hyper_util::rt::{TokioExecutor, TokioIo};
use proxima::error::ProximaError;
use proxima::h2::serve_h2_connection;
use tokio_util::compat::TokioAsyncReadCompatExt;
#[path = "../common/h2_external.rs"]
mod h2_external;
use proxima::listeners::http::QuiesceResponse;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::request::{Request, Response};
use proxima_primitives::pipe::SendPipe;
use tokio::net::{TcpListener, TcpStream};

const RESPONSE_BODY: &[u8] = b"ok";

fn build_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("tokio runtime")
}

async fn hyper_handler(
    _request: hyper::Request<hyper::body::Incoming>,
) -> Result<hyper::Response<Full<Bytes>>, Infallible> {
    Ok(hyper::Response::builder()
        .status(200)
        .header("content-type", "text/plain")
        .body(Full::new(Bytes::from_static(RESPONSE_BODY)))
        .expect("response"))
}

struct ConstantOk;

impl SendPipe for ConstantOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl std::future::Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move { Ok(Response::ok(Bytes::from_static(RESPONSE_BODY))) }
    }
}


fn start_proxima_server(
    runtime: &tokio::runtime::Runtime,
) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
    let (addr, listener) = runtime.block_on(async {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        (addr, listener)
    });
    let dispatch: PipeHandle = into_handle(ConstantOk);
    let join = runtime.spawn(async move {
        loop {
            let (socket, _) = match listener.accept().await {
                Ok(value) => value,
                Err(_) => break,
            };
            let _ = socket.set_nodelay(true);
            let dispatch = dispatch.clone();
            tokio::spawn(async move {
                let in_flight = Arc::new(AtomicU64::new(0));
                let quiesce = Arc::new(QuiesceResponse {
                    status: 503,
                    retry_after: "1".into(),
                });
                let _ =
                    h2_external::serve_h2_connection(socket, dispatch, in_flight, quiesce).await;
            });
        }
    });
    (addr, join)
}

fn start_hyper_server(
    runtime: &tokio::runtime::Runtime,
) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
    let (addr, listener) = runtime.block_on(async {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        (addr, listener)
    });
    let join = runtime.spawn(async move {
        loop {
            let (socket, _) = match listener.accept().await {
                Ok(value) => value,
                Err(_) => break,
            };
            let _ = socket.set_nodelay(true);
            tokio::spawn(async move {
                let io = TokioIo::new(socket);
                let _ = http2::Builder::new(TokioExecutor::new())
                    .serve_connection(io, pipe_fn(hyper_handler))
                    .await;
            });
        }
    });
    (addr, join)
}

fn start_proxima_native_server(
    runtime: &tokio::runtime::Runtime,
) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
    let (addr, listener) = runtime.block_on(async {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        (addr, listener)
    });
    let dispatch: PipeHandle = into_handle(ConstantOk);
    let join = runtime.spawn(async move {
        loop {
            let (socket, _) = match listener.accept().await {
                Ok(value) => value,
                Err(_) => break,
            };
            let _ = socket.set_nodelay(true);
            let dispatch = dispatch.clone();
            tokio::spawn(async move {
                let _ = serve_h2_connection(
                    socket.compat(),
                    dispatch,
                    proxima_listen::admission::ConnAdmission::unbounded(),
                    None,
                )
                .await;
            });
        }
    });
    (addr, join)
}

fn start_pingora_server(
    runtime: &tokio::runtime::Runtime,
) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
    use pingora_core::protocols::Digest;
    use pingora_core::protocols::http::v2::server::{HttpSession, handshake as h2c_handshake};
    use pingora_http::ResponseHeader;

    let (addr, listener) = runtime.block_on(async {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        (addr, listener)
    });
    let join = runtime.spawn(async move {
        loop {
            let (socket, _) = match listener.accept().await {
                Ok(value) => value,
                Err(_) => break,
            };
            let _ = socket.set_nodelay(true);
            tokio::spawn(async move {
                let l4_stream: pingora_core::protocols::l4::stream::Stream = socket.into();
                let pingora_stream: pingora_core::protocols::Stream = Box::new(l4_stream);
                let mut h2_conn = match h2c_handshake(pingora_stream, None).await {
                    Ok(conn) => conn,
                    Err(_) => return,
                };
                let digest = Arc::new(Digest::default());
                loop {
                    let session =
                        match HttpSession::from_h2_conn(&mut h2_conn, digest.clone()).await {
                            Ok(Some(session)) => session,
                            Ok(None) | Err(_) => break,
                        };
                    tokio::spawn(async move {
                        let mut session = session;
                        let mut header = match ResponseHeader::build(200, None) {
                            Ok(value) => value,
                            Err(_) => return,
                        };
                        let _ = header.append_header("content-type", "text/plain");
                        if session
                            .write_response_header(Box::new(header), false)
                            .is_err()
                        {
                            return;
                        }
                        let _ = session
                            .write_body(Bytes::from_static(RESPONSE_BODY), true)
                            .await;
                        let _ = session.finish();
                    });
                }
            });
        }
    });
    (addr, join)
}

fn warm_client(
    runtime: &tokio::runtime::Runtime,
    addr: std::net::SocketAddr,
) -> h2::client::SendRequest<Bytes> {
    runtime.block_on(async move {
        let socket = TcpStream::connect(addr).await.expect("connect");
        let _ = socket.set_nodelay(true);
        let (h2_client, h2_conn) = h2::client::handshake(socket).await.expect("handshake");
        tokio::spawn(async move {
            let _ = h2_conn.await;
        });
        h2_client
    })
}

async fn one_request(mut h2_client: h2::client::SendRequest<Bytes>) {
    let request = http::Request::builder()
        .method("GET")
        .uri("http://localhost/")
        .body(())
        .expect("request");
    let (response_future, _) = h2_client.send_request(request, true).expect("send_request");
    let response = response_future.await.expect("response");
    std::hint::black_box(response.status());
    let mut body = response.into_body();
    while let Some(chunk) = body.data().await {
        let chunk = chunk.expect("chunk");
        let len = chunk.len();
        body.flow_control()
            .release_capacity(len)
            .expect("flow control");
        std::hint::black_box(chunk);
    }
}

fn h2_vs_pingora(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_vs_pingora_warm");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1));
    let runtime = build_runtime();

    let (proxima_addr, _proxima_join) = start_proxima_server(&runtime);
    let proxima_client = warm_client(&runtime, proxima_addr);
    group.bench_function(
        "proxima::serve_h2_connection (external harness, warm)",
        |bencher| {
            bencher.to_async(&runtime).iter(|| {
                let client = proxima_client.clone();
                one_request(client)
            });
        },
    );

    let (native_addr, _native_join) = start_proxima_native_server(&runtime);
    let native_client = warm_client(&runtime, native_addr);
    group.bench_function("proxima::serve_h2_connection (native, warm)", |bencher| {
        bencher.to_async(&runtime).iter(|| {
            let client = native_client.clone();
            one_request(client)
        });
    });

    let (hyper_addr, _hyper_join) = start_hyper_server(&runtime);
    let hyper_client = warm_client(&runtime, hyper_addr);
    group.bench_function("hyper::http2::Builder (warm)", |bencher| {
        bencher.to_async(&runtime).iter(|| {
            let client = hyper_client.clone();
            one_request(client)
        });
    });

    let (pingora_addr, _pingora_join) = start_pingora_server(&runtime);
    let pingora_client = warm_client(&runtime, pingora_addr);
    group.bench_function("pingora::http::v2::HttpSession (warm)", |bencher| {
        bencher.to_async(&runtime).iter(|| {
            let client = pingora_client.clone();
            one_request(client)
        });
    });

    group.finish();
}

criterion_group!(benches, h2_vs_pingora);
criterion_main!(benches);
