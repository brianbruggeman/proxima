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

//! H3 hot-path microbench. Mirrors `h2_dispatch`:
//!
//! 1. `handshake_only` — full QUIC + h3 handshake per iter (connection
//!    setup tax: TLS 1.3 + initial QUIC packets + h3 SETTINGS exchange).
//! 2. `request_on_warm_connection` — one request/response on a
//!    persistent h3 connection. Per-request cost h3 is meant to amortize.
//! 3. `post_with_body_on_warm_connection` — small body echoed end-to-end
//!    on a warm connection.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima::error::ProximaError;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::request::{Request, Response};
use proxima_primitives::pipe::SendPipe;

#[path = "../common/h3_setup.rs"]
mod h3_setup;

const RESPONSE_BODY: &[u8] = b"ok";

struct ConstantOk;

impl SendPipe for ConstantOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
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
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move {
            let (_request, bytes) = request.body_bytes().await?;
            Ok(Response::ok(bytes))
        }
    }
}


fn handshake_only(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h3_handshake_only");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));
    let runtime = h3_setup::build_runtime();
    let dispatch: PipeHandle = into_handle(ConstantOk);
    let addr = h3_setup::start_h3_server(&runtime, dispatch);

    group.bench_function("quic_handshake_then_drop", |bencher| {
        bencher.to_async(&runtime).iter(|| async move {
            let client = h3_setup::make_client_endpoint();
            let connecting = client.connect(addr, "localhost").expect("connect");
            let connection = connecting.await.expect("handshake");
            let h3_conn = h3_quinn::Connection::new(connection);
            let (mut driver, send_request) = h3::client::builder()
                .build::<_, _, Bytes>(h3_conn)
                .await
                .expect("h3 build");
            std::hint::black_box(send_request);
            tokio::spawn(async move {
                let _ = std::future::poll_fn(|cx| driver.poll_close(cx)).await;
            });
            client.close(0u32.into(), b"done");
        });
    });
    group.finish();
}

fn request_on_warm_connection(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h3_request_on_warm_connection");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1));
    let runtime = h3_setup::build_runtime();
    let dispatch: PipeHandle = into_handle(ConstantOk);
    let addr = h3_setup::start_h3_server(&runtime, dispatch);

    let (client_endpoint, send_request) = runtime.block_on(async {
        let endpoint = h3_setup::make_client_endpoint();
        let send = h3_setup::warm_h3_client(&endpoint, addr).await;
        (endpoint, send)
    });
    let _client_endpoint = client_endpoint;

    group.bench_function("get_then_200_ok_on_warm_connection", |bencher| {
        let send_request = send_request.clone();
        let uri = format!("https://localhost:{}/", addr.port());
        bencher.to_async(&runtime).iter(|| {
            let uri = uri.clone();
            let mut send_request = send_request.clone();
            async move {
                let request = http::Request::builder()
                    .method("GET")
                    .uri(uri)
                    .body(())
                    .expect("request");
                let mut stream = send_request.send_request(request).await.expect("send");
                stream.finish().await.expect("finish");
                let response = stream.recv_response().await.expect("response");
                std::hint::black_box(response.status());
                while let Some(chunk) = stream.recv_data().await.expect("recv_data") {
                    std::hint::black_box(chunk);
                }
            }
        });
    });
    group.finish();
}

fn post_with_body_on_warm_connection(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h3_post_with_body_on_warm_connection");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1));
    let runtime = h3_setup::build_runtime();
    let dispatch: PipeHandle = into_handle(EchoBody);
    let addr = h3_setup::start_h3_server(&runtime, dispatch);

    let (client_endpoint, send_request) = runtime.block_on(async {
        let endpoint = h3_setup::make_client_endpoint();
        let send = h3_setup::warm_h3_client(&endpoint, addr).await;
        (endpoint, send)
    });
    let _client_endpoint = client_endpoint;

    let payload: Bytes = Bytes::from_static(b"hello");

    group.bench_function("post_5_bytes_then_echo_on_warm_connection", |bencher| {
        let send_request = send_request.clone();
        let uri = format!("https://localhost:{}/echo", addr.port());
        let payload = payload.clone();
        bencher.to_async(&runtime).iter(|| {
            let uri = uri.clone();
            let payload = payload.clone();
            let mut send_request = send_request.clone();
            async move {
                let request = http::Request::builder()
                    .method("POST")
                    .uri(uri)
                    .body(())
                    .expect("request");
                let mut stream = send_request.send_request(request).await.expect("send");
                stream.send_data(payload).await.expect("send body");
                stream.finish().await.expect("finish");
                let response = stream.recv_response().await.expect("response");
                std::hint::black_box(response.status());
                let mut total = 0usize;
                while let Some(mut chunk) = stream.recv_data().await.expect("recv_data") {
                    total += bytes::Buf::remaining(&chunk);
                    while bytes::Buf::has_remaining(&chunk) {
                        let slice = bytes::Buf::chunk(&chunk);
                        let advance = slice.len();
                        std::hint::black_box(slice);
                        bytes::Buf::advance(&mut chunk, advance);
                    }
                }
                std::hint::black_box(total);
            }
        });
    });
    let _ = Arc::new(AtomicU64::new(0));
    group.finish();
}

criterion_group!(
    benches,
    handshake_only,
    request_on_warm_connection,
    post_with_body_on_warm_connection
);
criterion_main!(benches);
