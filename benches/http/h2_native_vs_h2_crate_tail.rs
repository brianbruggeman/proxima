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

//! Tail-latency + concurrent-stream sweep: proxima native vs. h2
//! crate, both end-to-end through a warm h2 client. Records every
//! request latency in an HDR histogram so we can report p50/p90/p99/
//! p999/max — not just means.
//!
//! Concurrency levels: 1 (single-stream baseline), 10 (typical
//! browser pipelining), 100 (heavy multiplex). Each (impl ×
//! concurrency) combo runs for a fixed wall-clock duration.
//!
//! Not a criterion bench — we drive our own loop because criterion
//! reports mean/median, not tail percentiles. Single binary, runs
//! as `cargo bench --bench h2_native_vs_h2_crate_tail`.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::{Duration, Instant};

use bytes::Bytes;
use hdrhistogram::Histogram;
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
/// How long each (impl × concurrency) combo runs.
const SAMPLE_WINDOW: Duration = Duration::from_secs(3);
/// Histogram range: 1ns to 60s, 3 sig figs. Auto-resizes if needed.
fn fresh_histogram() -> Histogram<u64> {
    Histogram::<u64>::new_with_bounds(1, 60_000_000_000, 3).expect("hdr bounds")
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


async fn spawn_h2_crate(dispatch: PipeHandle) -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
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

async fn spawn_native(dispatch: PipeHandle) -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
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
    addr
}

async fn warm_client(addr: std::net::SocketAddr) -> h2::client::SendRequest<Bytes> {
    let socket = TcpStream::connect(addr).await.expect("connect");
    let _ = socket.set_nodelay(true);
    let (h2_client, h2_conn) = h2::client::handshake(socket).await.expect("handshake");
    tokio::spawn(async move {
        let _ = h2_conn.await;
    });
    h2_client
}

async fn one_request(client: &mut h2::client::SendRequest<Bytes>) {
    let request = http::Request::builder()
        .method("GET")
        .uri("http://localhost/")
        .body(())
        .expect("request");
    let (response_future, _) = client.send_request(request, true).expect("send_request");
    let response = response_future.await.expect("response");
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

/// Spawn `concurrency` parallel request loops sharing one h2 client.
/// Each loop records its own per-request latency into a histogram.
/// At the end, merge them and return the combined distribution.
async fn run_workload(addr: std::net::SocketAddr, concurrency: usize) -> Histogram<u64> {
    let client = warm_client(addr).await;
    // Prime the connection so handshake settles before timing.
    {
        let mut warmup = client.clone();
        for _ in 0..10 {
            one_request(&mut warmup).await;
        }
    }
    let deadline = Instant::now() + SAMPLE_WINDOW;
    let mut tasks = Vec::with_capacity(concurrency);
    for _ in 0..concurrency {
        let mut client = client.clone();
        tasks.push(tokio::spawn(async move {
            let mut histogram = fresh_histogram();
            while Instant::now() < deadline {
                let start = Instant::now();
                one_request(&mut client).await;
                let elapsed_ns = start.elapsed().as_nanos() as u64;
                let _ = histogram.record(elapsed_ns.max(1));
            }
            histogram
        }));
    }
    let mut combined = fresh_histogram();
    for task in tasks {
        let histogram = task.await.expect("join task");
        combined.add(&histogram).expect("merge histogram");
    }
    combined
}

fn format_ns(value: u64) -> String {
    if value < 1_000 {
        format!("{value} ns")
    } else if value < 1_000_000 {
        format!("{:.2} us", value as f64 / 1_000.0)
    } else if value < 1_000_000_000 {
        format!("{:.2} ms", value as f64 / 1_000_000.0)
    } else {
        format!("{:.2} s", value as f64 / 1_000_000_000.0)
    }
}

fn report(label: &str, histogram: &Histogram<u64>) {
    let p50 = histogram.value_at_quantile(0.50);
    let p90 = histogram.value_at_quantile(0.90);
    let p99 = histogram.value_at_quantile(0.99);
    let p999 = histogram.value_at_quantile(0.999);
    let max = histogram.max();
    let count = histogram.len();
    let mean = histogram.mean();
    println!(
        "{label:<32} count={count:<8} mean={:<10} p50={:<10} p90={:<10} p99={:<10} p999={:<10} max={}",
        format_ns(mean as u64),
        format_ns(p50),
        format_ns(p90),
        format_ns(p99),
        format_ns(p999),
        format_ns(max),
    );
}

fn main() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("tokio runtime");

    println!(
        "tail-latency sweep — warm h2 client, {}s per scenario, minimal GET workload",
        SAMPLE_WINDOW.as_secs(),
    );
    println!();

    let (h2_addr, native_addr) = runtime.block_on(async {
        let h2_addr = spawn_h2_crate(into_handle(ConstantOk)).await;
        let native_addr = spawn_native(into_handle(ConstantOk)).await;
        (h2_addr, native_addr)
    });

    for concurrency in [1, 10, 100] {
        let h2_histogram = runtime.block_on(run_workload(h2_addr, concurrency));
        let native_histogram = runtime.block_on(run_workload(native_addr, concurrency));
        println!("concurrency={concurrency}:");
        report("  h2_crate", &h2_histogram);
        report("  proxima_native", &native_histogram);
        println!();
    }
}
