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

//! Per-request allocation cost: proxima native vs h2 crate vs hyper
//! vs pingora. Wraps the global allocator with a counter so every
//! `alloc` / `dealloc` bumps an atomic. Workload runs once per impl;
//! delta-in-bytes / delta-in-count divided by request count gives
//! the per-request numbers.
//!
//! Counter overhead is two `Relaxed` atomic adds per allocation —
//! the bench numbers will be slightly inflated vs an un-instrumented
//! run but the **deltas** between impls are still meaningful.
//!
//! Not a criterion bench — we drive our own loop because we need
//! direct access to allocator counters at workload boundaries.

use std::alloc::{GlobalAlloc, Layout, System};
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use bytes::Bytes;
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

static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn snapshot() -> (u64, u64) {
    (
        ALLOCATED_BYTES.load(Ordering::Relaxed),
        ALLOCATIONS.load(Ordering::Relaxed),
    )
}

const RESPONSE_BODY: &[u8] = b"ok";
const WORKLOAD_DURATION: Duration = Duration::from_secs(3);

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


async fn spawn_proxima_h2(dispatch: PipeHandle) -> std::net::SocketAddr {
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

async fn spawn_proxima_native(dispatch: PipeHandle) -> std::net::SocketAddr {
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
                    serve_h2_connection(socket.compat(), dispatch, in_flight, quiesce, None).await;
            });
        }
    });
    addr
}

async fn spawn_hyper() -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
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

struct WorkloadResult {
    label: &'static str,
    requests: u64,
    elapsed: Duration,
    bytes_delta: u64,
    allocs_delta: u64,
}

async fn run_workload(label: &'static str, addr: std::net::SocketAddr) -> WorkloadResult {
    let mut client = warm_client(addr).await;
    // Prime the handshake + h2 init machinery so the snapshot
    // captures steady-state per-request cost, not connection setup.
    for _ in 0..100 {
        one_request(&mut client).await;
    }
    let (start_bytes, start_allocs) = snapshot();
    let start = Instant::now();
    let mut requests = 0u64;
    let deadline = start + WORKLOAD_DURATION;
    while Instant::now() < deadline {
        one_request(&mut client).await;
        requests += 1;
    }
    let elapsed = start.elapsed();
    let (end_bytes, end_allocs) = snapshot();
    WorkloadResult {
        label,
        requests,
        elapsed,
        bytes_delta: end_bytes - start_bytes,
        allocs_delta: end_allocs - start_allocs,
    }
}

fn report(result: &WorkloadResult) {
    let bytes_per_req = result.bytes_delta / result.requests;
    let allocs_per_req = result.allocs_delta as f64 / result.requests as f64;
    let rps = result.requests as f64 / result.elapsed.as_secs_f64();
    println!(
        "  {:<20}  rps={:<8.0}  bytes/req={:<6}  allocs/req={:<6.1}  total_bytes={:.1} MiB  total_allocs={}",
        result.label,
        rps,
        bytes_per_req,
        allocs_per_req,
        result.bytes_delta as f64 / (1024.0 * 1024.0),
        result.allocs_delta,
    );
}

fn main() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("tokio runtime");

    println!(
        "allocation cost per h2 request — counting global allocator, {}s steady-state window",
        WORKLOAD_DURATION.as_secs(),
    );
    println!(
        "(allocator wrapper adds ~2 atomic ops per allocation; absolute numbers slightly inflated, deltas meaningful)"
    );
    println!();

    let (proxima_addr, native_addr, hyper_addr) = runtime.block_on(async {
        let proxima_addr = spawn_proxima_h2(into_handle(ConstantOk)).await;
        let native_addr = spawn_proxima_native(into_handle(ConstantOk)).await;
        let hyper_addr = spawn_hyper().await;
        (proxima_addr, native_addr, hyper_addr)
    });

    let proxima_result = runtime.block_on(run_workload("proxima_h2_crate", proxima_addr));
    let native_result = runtime.block_on(run_workload("proxima_native", native_addr));
    let hyper_result = runtime.block_on(run_workload("hyper_http2", hyper_addr));

    println!("h2 server, warm client, minimal GET workload:");
    report(&proxima_result);
    report(&native_result);
    report(&hyper_result);
}
