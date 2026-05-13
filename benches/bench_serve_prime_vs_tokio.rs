//! Compare-bench gate for the prime-default serve path: drives the REAL
//! `HttpListenProtocol::serve` on BOTH backends — tokio (reference) and
//! prime (under test) — over a loopback HTTP/1.1 GET round-trip, and
//! reports time/iter + MB/s via criterion.
//!
//! Structure mirrors `tests/serve_parity.rs`: the tokio backend serves
//! inside a `LocalSet` (`runtime = None` → `spawn_local`); the prime
//! backend binds + accepts on a prime `CoreShard` worker via
//! `spawn_factory_on_core(CoreId(0), ...)` with the serve future leaked to
//! `'static`. The pipe returns a fixed 64 KiB body (the AGENTS.md-relevant
//! size, >=55 MB/s / sub-1ms p99 targets).
//!
//! Each benched iteration is ONE client GET round-trip on a kept-alive
//! connection against an already-bound + warmed listener: bind + warm-up
//! happen once outside the timing loop, so we time only request/response.
//!
//! prime ships single-core for serve (`CoreId(0)`), so this is a
//! single-core comparison.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::type_complexity)]
#![cfg(all(
    all(
        feature = "runtime-prime-executor",
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-reactor",
        feature = "runtime-prime-bgpool"
    ),
    feature = "http1"
))]

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use futures::channel::oneshot;

use proxima::error::ProximaError;
use proxima::listen::{ListenProtocol, ServeContext};
use proxima::listeners::HttpListenProtocol;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::prime::{CoreId, PrimeRuntime};
use proxima::request::{Request, Response};
use proxima::runtime::Runtime;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::telemetry_surface::NoopTelemetry;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

const BODY_LEN: usize = 64 * 1024;

// ---- pipe under test --------------------------------------------------

/// Returns a fixed 64 KiB body and fully drains any streamed request body
/// so the serve pump completes rather than parking on back-pressure.
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
    let body = Arc::new(vec![b'x'; BODY_LEN]);
    into_handle(FixedBody { body })
}

// ---- raw keep-alive client -------------------------------------------

async fn pick_free_addr() -> SocketAddr {
    let probe = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("probe bind");
    let addr = probe.local_addr().expect("probe addr");
    drop(probe);
    addr
}

async fn connect_retry(addr: SocketAddr) -> TcpStream {
    for _ in 0..200 {
        match TcpStream::connect(addr).await {
            Ok(stream) => {
                stream.set_nodelay(true).expect("nodelay");
                return stream;
            }
            Err(_) => tokio::time::sleep(Duration::from_millis(20)).await,
        }
    }
    panic!("listener at {addr} never accepted a connection");
}

/// One keep-alive GET round-trip: write the request, read the framed
/// 64 KiB Content-Length body off the same socket without closing it.
async fn keepalive_get(stream: &mut TcpStream) {
    stream
        .write_all(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
        .await
        .expect("client write");
    stream.flush().await.expect("client flush");

    let mut raw = Vec::with_capacity(BODY_LEN + 512);
    let mut scratch = [0_u8; 16 * 1024];
    let header_end = loop {
        if let Some(position) = raw.windows(4).position(|window| window == b"\r\n\r\n") {
            break position;
        }
        let read = stream.read(&mut scratch).await.expect("client read head");
        assert!(read != 0, "eof before response head completed");
        raw.extend_from_slice(&scratch[..read]);
    };
    let head = std::str::from_utf8(&raw[..header_end]).expect("headers utf8");
    let lower = head.to_ascii_lowercase();
    let chunked = lower.contains("transfer-encoding: chunked");
    let content_length = lower
        .split("\r\n")
        .find_map(|line| line.strip_prefix("content-length:"))
        .map(|value| {
            value
                .trim()
                .parse::<usize>()
                .expect("content-length parses")
        });
    let body_start = header_end + 4;
    if let Some(length) = content_length {
        while raw.len() < body_start + length {
            let read = stream.read(&mut scratch).await.expect("client read body");
            assert!(read != 0, "eof before content-length body completed");
            raw.extend_from_slice(&scratch[..read]);
        }
        assert_eq!(length, BODY_LEN, "server returned the fixed 64 KiB body");
    } else if chunked {
        while !raw[body_start..].ends_with(b"0\r\n\r\n") {
            let read = stream.read(&mut scratch).await.expect("client read chunk");
            assert!(read != 0, "eof before chunked body terminated");
            raw.extend_from_slice(&scratch[..read]);
        }
    } else {
        panic!("keep-alive response is neither content-length nor chunked framed");
    }
}

// ---- tokio reference backend -----------------------------------------

/// Serve `HttpListenProtocol::serve` with `TokioAcceptorFactory` and
/// `runtime = None` inside a `LocalSet`. The serve future borrows the
/// protocol + spec so it cannot be spawned; it races the client on the
/// same LocalSet task. Returns once `iters` round-trips have completed.
fn run_tokio_iters(runtime: &tokio::runtime::Runtime, iters: u64) -> Duration {
    let local = tokio::task::LocalSet::new();
    local.block_on(runtime, async move {
        let addr = pick_free_addr().await;
        let context = ServeContext::new(NoopTelemetry::handle())
            .with_acceptor_factory(Arc::new(proxima_net::tokio::TokioAcceptorFactory));
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let spec = serde_json::json!({ "name": "http" });
        let protocol = HttpListenProtocol::new();
        let serve = protocol.serve(addr, dispatch_handle(), &spec, context, shutdown_rx);

        let elapsed = tokio::select! {
            serve_result = serve => panic!("tokio serve returned early: {serve_result:?}"),
            elapsed = drive_iters(addr, iters) => elapsed,
        };
        drop(shutdown_tx);
        elapsed
    })
}

// ---- prime backend under test ----------------------------------------

/// Bind + accept the real `HttpListenProtocol::serve` on a prime worker
/// via `spawn_factory_on_core(CoreId(0), ...)`. Protocol + spec are leaked
/// to `'static` so the serve future's `'_` borrow lives inside the
/// `'static` factory closure (exactly as `serve_parity::spawn_prime_serve`).
fn spawn_prime_serve(runtime: &Arc<PrimeRuntime>, addr: SocketAddr) -> oneshot::Sender<()> {
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let runtime_for_context: Arc<dyn Runtime> = runtime.clone();

    let protocol: &'static HttpListenProtocol = Box::leak(Box::new(HttpListenProtocol::new()));
    let spec: &'static serde_json::Value =
        Box::leak(Box::new(serde_json::json!({ "name": "http" })));

    runtime
        .spawn_factory_on_core(
            CoreId(0),
            Box::new(move || {
                let context = ServeContext::new(NoopTelemetry::handle())
                    .with_runtime(runtime_for_context)
                    .with_acceptor_factory(Arc::new(proxima_net::prime::PrimeAcceptorFactory));
                Box::pin(async move {
                    let _ = protocol
                        .serve(addr, dispatch_handle(), spec, context, shutdown_rx)
                        .await;
                }) as std::pin::Pin<Box<dyn Future<Output = ()> + 'static>>
            }),
        )
        .expect("spawn prime serve factory on core 0");

    shutdown_tx
}

/// Drive `iters` keep-alive round-trips on one connection, returning the
/// total elapsed time of just the request/response loop (connect + one
/// warm-up GET happen before the clock starts).
async fn drive_iters(addr: SocketAddr, iters: u64) -> Duration {
    let mut stream = connect_retry(addr).await;
    keepalive_get(&mut stream).await; // warm-up, untimed
    let start = std::time::Instant::now();
    for _ in 0..iters {
        keepalive_get(&mut stream).await;
    }
    start.elapsed()
}

fn run_prime_iters(
    client_runtime: &tokio::runtime::Runtime,
    prime_runtime: &Arc<PrimeRuntime>,
    iters: u64,
) -> Duration {
    client_runtime.block_on(async move {
        let addr = pick_free_addr().await;
        let shutdown_tx = spawn_prime_serve(prime_runtime, addr);
        let elapsed = drive_iters(addr, iters).await;
        drop(shutdown_tx);
        elapsed
    })
}

// ---- criterion entry point -------------------------------------------

fn client_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("client tokio runtime")
}

fn bench_serve(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("serve");
    group.throughput(Throughput::Bytes(BODY_LEN as u64));

    let tokio_runtime = client_runtime();
    group.bench_function("tokio", |bencher| {
        bencher.iter_custom(|iters| run_tokio_iters(&tokio_runtime, iters));
    });

    let prime_runtime = Arc::new(
        PrimeRuntime::builder()
            .cores(2)
            .background_inline()
            .build()
            .expect("build prime runtime"),
    );
    let prime_client_runtime = client_runtime();
    group.bench_function("prime", |bencher| {
        bencher.iter_custom(|iters| run_prime_iters(&prime_client_runtime, &prime_runtime, iters));
    });

    group.finish();
}

criterion_group!(benches, bench_serve);
criterion_main!(benches);
