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

//! H2 hot-path microbench. Mirrors `h1_dispatch` for HTTP/2.
//!
//! Three measurements:
//! 1. `handshake_only` — one full TCP-level bidirectional h2 handshake
//!    per iter. The connection-setup tax: preface + SETTINGS exchange.
//! 2. `request_on_warm_connection` — one request/response on a
//!    persistent h2 connection. This is the per-request cost h2 was
//!    designed to minimize (vs per-connection). Comparable to
//!    `h1_dispatch::connection_round_trip_no_body`.
//! 3. `post_with_body_on_warm_connection` — request with a 5-byte body
//!    that the server echoes. Same connection reused across iters.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima::error::ProximaError;
#[path = "../common/h2_external.rs"]
mod h2_external;
use proxima::listeners::http::QuiesceResponse;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::request::{Request, Response};
use proxima_primitives::pipe::SendPipe;
use tokio::net::{TcpListener, TcpStream};

const RESPONSE_BODY: &[u8] = b"ok";

fn build_runtime() -> tokio::runtime::Runtime {
    // Multi-thread: server task + client conn driver + main iter share
    // 3 cooperative tasks per bench; multi-thread avoids starvation.
    // serve_h2_connection itself is runtime-agnostic (FuturesUnordered +
    // select!), so the listener works on any executor.
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("tokio runtime")
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


struct EchoBody;

impl SendPipe for EchoBody {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl std::future::Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move {
            let (_request, bytes) = request.body_bytes().await?;
            Ok(Response::ok(bytes))
        }
    }
}


/// Boot a loopback h2 server. Returns the bound addr + a handle to
/// keep the accept-loop alive while the bench runs.
fn start_server(
    runtime: &tokio::runtime::Runtime,
    dispatch: PipeHandle,
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

fn handshake_only(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_handshake_only");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));
    let runtime = build_runtime();
    let dispatch: PipeHandle = into_handle(ConstantOk);
    let (addr, _server) = start_server(&runtime, dispatch);
    group.bench_function("tcp_handshake_then_drop", |bencher| {
        bencher.to_async(&runtime).iter(|| async {
            let socket = TcpStream::connect(addr).await.expect("connect");
            let _ = socket.set_nodelay(true);
            let (h2_client, h2_conn) = h2::client::handshake(socket).await.expect("handshake");
            let conn = tokio::spawn(async move {
                let _ = h2_conn.await;
            });
            drop(h2_client);
            let _ = conn.await;
        });
    });
    group.finish();
}

fn request_on_warm_connection(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_request_on_warm_connection");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1));
    let runtime = build_runtime();
    let dispatch: PipeHandle = into_handle(ConstantOk);
    let (addr, _server) = start_server(&runtime, dispatch);

    // Set up ONE h2 connection up front; bench measures per-request
    // cost on that warm connection.
    let h2_client = runtime.block_on(async {
        let socket = TcpStream::connect(addr).await.expect("connect");
        let _ = socket.set_nodelay(true);
        let (h2_client, h2_conn) = h2::client::handshake(socket).await.expect("handshake");
        tokio::spawn(async move {
            let _ = h2_conn.await;
        });
        h2_client
    });

    group.bench_function("get_then_200_ok_on_warm_connection", |bencher| {
        let h2_client = h2_client.clone();
        bencher.to_async(&runtime).iter(|| {
            let mut h2_client = h2_client.clone();
            async move {
                let request = http::Request::builder()
                    .method("GET")
                    .uri("http://localhost/")
                    .body(())
                    .expect("request");
                let (response_future, _) =
                    h2_client.send_request(request, true).expect("send_request");
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
        });
    });
    group.finish();
}

fn post_with_body_on_warm_connection(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_post_with_body_on_warm_connection");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1));
    let runtime = build_runtime();
    let dispatch: PipeHandle = into_handle(EchoBody);
    let (addr, _server) = start_server(&runtime, dispatch);

    let h2_client = runtime.block_on(async {
        let socket = TcpStream::connect(addr).await.expect("connect");
        let _ = socket.set_nodelay(true);
        let (h2_client, h2_conn) = h2::client::handshake(socket).await.expect("handshake");
        tokio::spawn(async move {
            let _ = h2_conn.await;
        });
        h2_client
    });
    let payload: Bytes = Bytes::from_static(b"hello");

    group.bench_function("post_5_bytes_then_echo_on_warm_connection", |bencher| {
        let h2_client = h2_client.clone();
        let payload = payload.clone();
        bencher.to_async(&runtime).iter(|| {
            let mut h2_client = h2_client.clone();
            let payload = payload.clone();
            async move {
                let request = http::Request::builder()
                    .method("POST")
                    .uri("http://localhost/echo")
                    .body(())
                    .expect("request");
                let (response_future, mut send_body) = h2_client
                    .send_request(request, false)
                    .expect("send_request");
                send_body.send_data(payload, true).expect("send body");
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
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    handshake_only,
    request_on_warm_connection,
    post_with_body_on_warm_connection
);
criterion_main!(benches);
