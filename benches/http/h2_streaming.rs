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

//! H2 body-size sweep on a warm connection.
//!
//! Three body shapes echoed through proxima's h2 listener:
//! - 256 bytes (small body; per-request overhead dominates)
//! - 64 KiB (single-frame medium body; per-byte work dominates)
//! - 16x4 KiB (multi-frame body; exercises h2 flow-control credit
//!   cycles in both directions)
//!
//! Warm-connection pattern: one h2 handshake at bench setup; the
//! `iter` measures only the per-request echo cost. The h2 design is
//! connection reuse + multiplexing; per-iter handshake would mask
//! the actual per-request signal in connection-setup tax.

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

fn build_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("tokio runtime")
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


fn start_echo_server(
    runtime: &tokio::runtime::Runtime,
) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
    let (addr, listener) = runtime.block_on(async {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        (addr, listener)
    });
    let dispatch: PipeHandle = into_handle(EchoBody);
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

async fn echo_one(h2_client: h2::client::SendRequest<Bytes>, body: Bytes, chunks: usize) {
    let mut h2_client = h2_client;
    let request = http::Request::builder()
        .method("POST")
        .uri("http://localhost/echo")
        .body(())
        .expect("request");
    let (response_future, mut send_body) = h2_client
        .send_request(request, false)
        .expect("send_request");
    let total = body.len();
    let chunk_size = total.div_ceil(chunks).max(1);
    let mut offset = 0;
    while offset < total {
        let end = (offset + chunk_size).min(total);
        let slice = body.slice(offset..end);
        let end_of_stream = end == total;
        send_body
            .send_data(slice, end_of_stream)
            .expect("send body");
        offset = end;
    }
    let response = response_future.await.expect("response");
    std::hint::black_box(response.status());
    let mut body_recv = response.into_body();
    let mut total_recv = 0;
    while let Some(chunk) = body_recv.data().await {
        let chunk = chunk.expect("chunk");
        total_recv += chunk.len();
        body_recv
            .flow_control()
            .release_capacity(chunk.len())
            .expect("flow control");
    }
    std::hint::black_box(total_recv);
}

fn echo_round_trip(criterion: &mut Criterion, label: &str, body: Bytes, chunks: usize) {
    let mut group = criterion.benchmark_group(format!("h2_echo_{label}"));
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Bytes(body.len() as u64));
    let runtime = build_runtime();
    let (addr, _server) = start_echo_server(&runtime);
    let client = warm_client(&runtime, addr);
    group.bench_function("echo_on_warm_connection", |bencher| {
        let client = client.clone();
        let body = body.clone();
        bencher.to_async(&runtime).iter(|| {
            let client = client.clone();
            let body = body.clone();
            async move { echo_one(client, body, chunks).await }
        });
    });
    group.finish();
}

fn small_body_256(criterion: &mut Criterion) {
    let body = Bytes::from(vec![b'A'; 256]);
    echo_round_trip(criterion, "256b", body, 1);
}

fn medium_body_64kib(criterion: &mut Criterion) {
    let body = Bytes::from(vec![b'A'; 64 * 1024]);
    echo_round_trip(criterion, "64kib", body, 1);
}

fn multi_chunk_16x4kib(criterion: &mut Criterion) {
    let body = Bytes::from(vec![b'A'; 16 * 4 * 1024]);
    echo_round_trip(criterion, "16x4kib", body, 16);
}

criterion_group!(
    benches,
    small_body_256,
    medium_body_64kib,
    multi_chunk_16x4kib
);
criterion_main!(benches);
