#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
#![cfg(feature = "http2")]

//! End-to-end head-to-head: proxima's **native** HTTP/2 listener
//! vs. the same listener built on the **h2 crate**. Identical
//! Pipe impls on both sides; identical h2-crate client driving
//! both. Plain TCP (no TLS) so we measure the protocol-stack delta,
//! not handshake / cipher cost.
//!
//! Three workload regimes:
//!
//! ## Best case (`get_minimal`)
//! Minimal GET → minimal 200 response. Body = `b"ok"`, no extra
//! headers. Exercises the hot path: 1-byte HPACK indexed pseudo
//! headers in both directions, tiny DATA frame, near-zero per-
//! request work. Native's HPACK decode wins should dominate here.
//!
//! ## Balanced (`get_browser_headers`)
//! Realistic browser-style GET: 4 pseudo-headers + user-agent,
//! accept, accept-language, accept-encoding, cookie. Response
//! includes content-type + content-length + cache-control. Most
//! browser/server-mesh traffic looks like this.
//!
//! ## Worst case (`post_echo_32kib`)
//! 32 KiB request body + 32 KiB response body echoed back. Stress
//! tests: HEADERS encode/decode, DATA chunking at peer max frame
//! size (multiple frames), in-process body buffering. The native
//! impl's first-cut buffers full body before dispatch (no streaming
//! channel yet) — this is the workload that exposes that cost.
//!
//! Capped at 32 KiB so both bodies stay within the default 65,535
//! flow-control window. Larger bodies hit the native server's
//! pending send-side flow-control gap (separate work item;
//! tracked in tests/listener_h2.rs via the `#[ignore]`'d
//! `initial_window_plus_one` test).
//!
//! All scenarios use a warm h2 connection (handshake done once per
//! bench, not per-iter) so we measure per-request cost on a
//! multiplexed connection — h2's whole point.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
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

const SMALL_RESPONSE_BODY: &[u8] = b"ok";

fn build_runtime() -> tokio::runtime::Runtime {
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
        async move { Ok(Response::ok(Bytes::from_static(SMALL_RESPONSE_BODY))) }
    }
}


struct BrowserStyleResponse;

impl SendPipe for BrowserStyleResponse {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl std::future::Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move {
            let mut response = Response::ok(Bytes::from_static(b"{\"ok\":true,\"id\":42}"));
            let _ = response.metadata.insert("content-type", "application/json");
            let _ = response.metadata.insert("content-length", "19");
            let _ = response.metadata.insert("cache-control", "no-store");
            let _ = response.metadata.insert("server", "proxima-bench/0.1");
            Ok(response)
        }
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


fn start_h2_crate_server(
    runtime: &tokio::runtime::Runtime,
    dispatch: PipeHandle,
) -> std::net::SocketAddr {
    let (addr, listener) = runtime.block_on(async {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        (addr, listener)
    });
    runtime.spawn(async move {
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
    addr
}

fn start_native_server(
    runtime: &tokio::runtime::Runtime,
    dispatch: PipeHandle,
) -> std::net::SocketAddr {
    let (addr, listener) = runtime.block_on(async {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        (addr, listener)
    });
    runtime.spawn(async move {
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
                    serve_h2_connection(socket.compat(), dispatch, in_flight, quiesce, None).await;
            });
        }
    });
    addr
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

async fn one_minimal_get(mut h2_client: h2::client::SendRequest<Bytes>) {
    let request = http::Request::builder()
        .method("GET")
        .uri("http://localhost/")
        .body(())
        .expect("request");
    let (response_future, _) = h2_client.send_request(request, true).expect("send_request");
    let response = response_future.await.expect("response");
    std::hint::black_box(response.status());
    drain_body(response.into_body()).await;
}

async fn one_browser_get(mut h2_client: h2::client::SendRequest<Bytes>) {
    let request = http::Request::builder()
        .method("GET")
        .uri("http://localhost/api/v1/items/42")
        .header("user-agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        .header("accept", "text/html,application/xhtml+xml,application/xml;q=0.9")
        .header("accept-language", "en-US,en;q=0.9")
        .header("accept-encoding", "gzip, deflate")
        .header("cookie", "session=abc123; user=42; locale=en_US")
        .body(())
        .expect("request");
    let (response_future, _) = h2_client.send_request(request, true).expect("send_request");
    let response = response_future.await.expect("response");
    std::hint::black_box(response.status());
    drain_body(response.into_body()).await;
}

async fn one_post_echo(mut h2_client: h2::client::SendRequest<Bytes>, payload: Bytes) {
    let request = http::Request::builder()
        .method("POST")
        .uri("http://localhost/echo")
        .body(())
        .expect("request");
    let (response_future, mut send) = h2_client
        .send_request(request, false)
        .expect("send_request");
    send.send_data(payload, true).expect("send body");
    let response = response_future.await.expect("response");
    std::hint::black_box(response.status());
    drain_body(response.into_body()).await;
}

async fn drain_body(mut body: h2::RecvStream) {
    while let Some(chunk) = body.data().await {
        let chunk = chunk.expect("chunk");
        let len = chunk.len();
        body.flow_control()
            .release_capacity(len)
            .expect("flow control");
        std::hint::black_box(chunk);
    }
}

fn best_case_get_minimal(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_e2e_best_case_get_minimal");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1));

    let runtime = build_runtime();

    let h2_addr = start_h2_crate_server(&runtime, into_handle(ConstantOk));
    let h2_client = warm_client(&runtime, h2_addr);
    group.bench_function("h2_crate", |bencher| {
        bencher.to_async(&runtime).iter(|| {
            let client = h2_client.clone();
            one_minimal_get(client)
        });
    });

    let native_addr = start_native_server(&runtime, into_handle(ConstantOk));
    let native_client = warm_client(&runtime, native_addr);
    group.bench_function("proxima_native", |bencher| {
        bencher.to_async(&runtime).iter(|| {
            let client = native_client.clone();
            one_minimal_get(client)
        });
    });

    group.finish();
}

fn balanced_browser_get(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_e2e_balanced_browser_get");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1));

    let runtime = build_runtime();

    let h2_addr = start_h2_crate_server(&runtime, into_handle(BrowserStyleResponse));
    let h2_client = warm_client(&runtime, h2_addr);
    group.bench_function("h2_crate", |bencher| {
        bencher.to_async(&runtime).iter(|| {
            let client = h2_client.clone();
            one_browser_get(client)
        });
    });

    let native_addr = start_native_server(&runtime, into_handle(BrowserStyleResponse));
    let native_client = warm_client(&runtime, native_addr);
    group.bench_function("proxima_native", |bencher| {
        bencher.to_async(&runtime).iter(|| {
            let client = native_client.clone();
            one_browser_get(client)
        });
    });

    group.finish();
}

fn worst_case_post_echo_32kib(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_e2e_worst_case_post_echo_32kib");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Bytes(32 * 1024));

    let runtime = build_runtime();
    let payload = Bytes::from(vec![b'p'; 32 * 1024]);

    let h2_addr = start_h2_crate_server(&runtime, into_handle(EchoBody));
    let h2_client = warm_client(&runtime, h2_addr);
    let payload_h2 = payload.clone();
    group.bench_function("h2_crate", |bencher| {
        bencher.to_async(&runtime).iter(|| {
            let client = h2_client.clone();
            let payload = payload_h2.clone();
            one_post_echo(client, payload)
        });
    });

    let native_addr = start_native_server(&runtime, into_handle(EchoBody));
    let native_client = warm_client(&runtime, native_addr);
    let payload_native = payload.clone();
    group.bench_function("proxima_native", |bencher| {
        bencher.to_async(&runtime).iter(|| {
            let client = native_client.clone();
            let payload = payload_native.clone();
            one_post_echo(client, payload)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    best_case_get_minimal,
    balanced_browser_get,
    worst_case_post_echo_32kib,
);
criterion_main!(benches);
