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

//! Head-to-head h2: proxima's `serve_h2_connection` vs hyper's
//! `server::conn::http2::Builder`. Same h2 state-machine crate
//! underneath; the delta isolates substrate Request/Response wiring
//! vs hyper's Body conversions.
//!
//! Both servers run over loopback TCP with a warm h2 connection
//! (handshake done once per bench, not per-iter) so we measure
//! per-request cost — h2's whole point is connection reuse +
//! multiplexing.

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

fn h2_end_to_end(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_end_to_end_warm");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1));

    let runtime = build_runtime();

    let (proxima_addr, _proxima_join) = start_proxima_server(&runtime);
    let proxima_client = warm_client(&runtime, proxima_addr);
    group.bench_function("proxima::serve_h2_connection (warm)", |bencher| {
        bencher.to_async(&runtime).iter(|| {
            let client = proxima_client.clone();
            one_request(client)
        });
    });

    let (hyper_addr, _hyper_join) = start_hyper_server(&runtime);
    let hyper_client = warm_client(&runtime, hyper_addr);
    group.bench_function("hyper::server::conn::http2 (warm)", |bencher| {
        bencher.to_async(&runtime).iter(|| {
            let client = hyper_client.clone();
            one_request(client)
        });
    });

    group.finish();
}

criterion_group!(benches, h2_end_to_end);
criterion_main!(benches);
