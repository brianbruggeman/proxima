//! Concurrent-throughput gate for `PrimeServeExt::serve_http` — the actual
//! daemon :9091 path (NOT the `HttpListenProtocol` factory path that
//! `bench_serve_prime_vs_tokio` drives). Measures aggregate request/s across
//! C kept-alive connections, so the round-robin handler dispatch (one accept
//! loop on core 0, handlers spread across cores 1..N) is actually exercised.
//!
//! The before/after for the round-robin change is `cores=1` vs `cores=10`:
//! at `cores=1` every handler runs on core 0 (identical to the pre-change
//! single-core behavior); at `cores=10` handlers fan out. A round-robin that
//! helps shows near-linear scaling from 1→10 cores under concurrency; a
//! regression shows 10-core aggregate at or below 1-core.
//!
//! Body is 1 KiB: small enough that per-request dispatch cost (the thing the
//! round-robin changes) dominates over bandwidth.

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool",
    feature = "http1"
))]

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use proxima::error::ProximaError;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::prime::PrimeRuntime;
use proxima::request::{Request, Response};
use proxima::runtime::PrimeServeExt;
use proxima_primitives::pipe::SendPipe;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

const BODY_LEN: usize = 1024;

struct FixedBody {
    body: Arc<Vec<u8>>,
}

impl SendPipe for FixedBody {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let body = self.body.clone();
        async move {
            if let Some(stream) = request.stream {
                let _ = stream.collect().await?;
            }
            Ok(Response::ok(body.as_slice().to_vec()))
        }
    }
}


fn dispatch_handle() -> PipeHandle {
    into_handle(FixedBody {
        body: Arc::new(vec![b'x'; BODY_LEN]),
    })
}

async fn connect_retry(addr: SocketAddr) -> TcpStream {
    for _ in 0..400 {
        if let Ok(stream) = TcpStream::connect(addr).await {
            stream.set_nodelay(true).expect("nodelay");
            return stream;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("listener at {addr} never accepted");
}

/// One keep-alive GET round-trip reading the Content-Length / chunked body.
async fn keepalive_get(stream: &mut TcpStream, scratch: &mut Vec<u8>) {
    stream
        .write_all(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
        .await
        .expect("write");
    stream.flush().await.expect("flush");
    scratch.clear();
    let mut chunk = [0_u8; 8192];
    let header_end = loop {
        if let Some(position) = scratch.windows(4).position(|w| w == b"\r\n\r\n") {
            break position;
        }
        let read = stream.read(&mut chunk).await.expect("read head");
        assert!(read != 0, "eof before head");
        scratch.extend_from_slice(&chunk[..read]);
    };
    let head = String::from_utf8_lossy(&scratch[..header_end]).to_ascii_lowercase();
    let body_start = header_end + 4;
    if let Some(length) = head
        .split("\r\n")
        .find_map(|line| line.strip_prefix("content-length:"))
        .and_then(|value| value.trim().parse::<usize>().ok())
    {
        while scratch.len() < body_start + length {
            let read = stream.read(&mut chunk).await.expect("read body");
            assert!(read != 0, "eof before body");
            scratch.extend_from_slice(&chunk[..read]);
        }
    } else {
        while !scratch[body_start..].ends_with(b"0\r\n\r\n") {
            let read = stream.read(&mut chunk).await.expect("read chunk");
            assert!(read != 0, "eof before chunk end");
            scratch.extend_from_slice(&chunk[..read]);
        }
    }
}

/// Run `total_iters` requests spread across `concurrency` kept-alive
/// connections; return the elapsed wall time of the timed loop. Connect +
/// one warm-up GET per connection happen before the clock starts.
fn run_concurrent(
    client_runtime: &tokio::runtime::Runtime,
    serve_runtime: &Arc<PrimeRuntime>,
    concurrency: usize,
    total_iters: u64,
) -> Duration {
    client_runtime.block_on(async move {
        let handle = serve_runtime
            .serve_http("127.0.0.1:0".parse().unwrap(), dispatch_handle())
            .expect("serve_http");
        let addr = handle.bind_addr().expect("bound addr");

        let per_conn = total_iters / concurrency as u64;
        let tasks: Vec<_> = (0..concurrency)
            .map(|_| {
                tokio::spawn(async move {
                    let mut stream = connect_retry(addr).await;
                    let mut scratch = Vec::with_capacity(BODY_LEN + 512);
                    keepalive_get(&mut stream, &mut scratch).await; // warm-up
                    let start = Instant::now();
                    for _ in 0..per_conn {
                        keepalive_get(&mut stream, &mut scratch).await;
                    }
                    start.elapsed()
                })
            })
            .collect();

        let mut max_elapsed = Duration::ZERO;
        for task in tasks {
            max_elapsed = max_elapsed.max(task.await.expect("client task"));
        }
        handle.shutdown();
        max_elapsed
    })
}

fn client_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("client runtime")
}

fn prime_runtime(cores: usize) -> Arc<PrimeRuntime> {
    Arc::new(
        PrimeRuntime::builder()
            .cores(cores)
            .background_inline()
            .build()
            .expect("prime runtime"),
    )
}

fn bench_concurrent(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("serve_http_concurrent");
    group.throughput(Throughput::Elements(1));
    group.sample_size(30);

    let client = client_runtime();
    // cores=1 reproduces the pre-round-robin single-core behavior (all
    // handlers on core 0); cores=10 fans handlers across cores 1..10.
    for cores in [1_usize, 10] {
        let runtime = prime_runtime(cores);
        for concurrency in [1_usize, 8, 32] {
            group.bench_function(format!("cores{cores}/conc{concurrency}"), |bencher| {
                bencher.iter_custom(|iters| {
                    let iters = iters.max(concurrency as u64);
                    run_concurrent(&client, &runtime, concurrency, iters)
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_concurrent);
criterion_main!(benches);
