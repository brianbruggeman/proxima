//! Fairness under skewed connection load. Exercises the work-stealing
//! vs pinned-per-core tradeoff: when a service receives more connections
//! than cores AND the workload is CPU-bound enough to saturate a single
//! core, can the runtime redistribute work?
//!
//! Workload:
//!   - Server has 2 cores.
//!   - `N` concurrent client connections each issue `M` requests
//!     sequentially.
//!   - Each request triggers a blake3 hash of 4 KiB on the server,
//!     producing per-request CPU cost ~5 μs on M1. With 4 connections ×
//!     50 requests this is ~1 ms of CPU work total, enough to saturate
//!     a single core if all connections pin to it.
//!
//! Compared backends:
//!   - `tokio_multi_thread` — default tokio, work-stealing across 2
//!     worker threads. Connections accepted on whichever thread tokio
//!     picks; the scheduler redistributes work under load. This is the
//!     fairness baseline.
//!   - `prime` — PrimeRuntime with explicit round-robin connection
//!     dispatch via `spawn_on_core(connection_index % cores, …)`. We
//!     are NOT testing the worst-case (all connections on core 0); we
//!     ARE testing whether prime can match tokio's fairness *when the
//!     application explicitly fans out across cores*. The gap between
//!     these two arms is the "cost of not having work-stealing":
//!     prime requires manual fanout, tokio gets it for free.
//!
//! The "fairness" of an arm is reflected in BOTH the mean and the tail:
//! if a single core's queue blocks others, total wall time stretches
//! out and tail (last connection to finish) dominates the mean. We
//! report mean per-request latency; criterion's stddev/CV captures the
//! fairness leakage.
//!
//! required-features: runtime-tokio, runtime-prime-full, http2, tcp,
//! http1.

#![cfg(all(
    feature = "runtime-tokio",
    all(
        feature = "runtime-prime-executor",
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-reactor",
        feature = "runtime-prime-bgpool"
    ),
    feature = "http2",
))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

#[path = "common/hdr_phased.rs"]
mod hdr_phased;
use hdr_phased::HdrQuartet;
use proxima::error::ProximaError;
use proxima::h2::serve_h2_connection;
use proxima::listeners::http::QuiesceResponse;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::request::{Request, Response};
use proxima::runtime::prime::os::net::TcpListener as ProximaTcpListener;
use proxima::runtime::{CoreId, PrimeRuntime, Runtime, TokioPerCoreRuntime};
use proxima_primitives::pipe::SendPipe;
use tokio::net::TcpListener;
use tokio_util::compat::TokioAsyncReadCompatExt;

const HASH_INPUT_LEN: usize = 4096;
const CORES: usize = 2;

#[inline(never)]
fn cpu_work(payload: &[u8]) -> u64 {
    let mut acc: u64 = 0;
    for _ in 0..32 {
        for byte in payload {
            acc = acc
                .wrapping_mul(0x100000001b3)
                .wrapping_add(u64::from(*byte));
        }
    }
    acc
}
const CONNECTIONS: usize = 4;
const REQUESTS_PER_CONNECTION: usize = 50;
const TOTAL_REQUESTS: usize = CONNECTIONS * REQUESTS_PER_CONNECTION;

struct HashPipe {
    payload: Arc<[u8; HASH_INPUT_LEN]>,
}

impl SendPipe for HashPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl std::future::Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let payload = self.payload.clone();
        async move {
            // Inline CPU work — the whole point of this bench is to see
            // whether the runtime can keep N connections running in
            // parallel when each saturates the core it lands on.
            let digest = cpu_work(payload.as_ref());
            Ok(Response::ok(Bytes::from(digest.to_le_bytes().to_vec())))
        }
    }
}


fn payload() -> Arc<[u8; HASH_INPUT_LEN]> {
    let mut buf = [0u8; HASH_INPUT_LEN];
    for (idx, slot) in buf.iter_mut().enumerate() {
        *slot = (idx & 0xff) as u8;
    }
    Arc::new(buf)
}

fn build_client_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("client runtime")
}

async fn one_request(mut client: h2::client::SendRequest<Bytes>) {
    let request = http::Request::builder()
        .method("GET")
        .uri("http://localhost/")
        .body(())
        .expect("request");
    let (response_future, _) = client.send_request(request, true).expect("send");
    let response = response_future.await.expect("response");
    let _ = std::hint::black_box(response.status());
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

async fn warm_connections(
    addr: std::net::SocketAddr,
    count: usize,
) -> Vec<h2::client::SendRequest<Bytes>> {
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let socket = tokio::net::TcpStream::connect(addr).await.expect("connect");
        let _ = socket.set_nodelay(true);
        let (client, conn) = h2::client::handshake(socket).await.expect("handshake");
        tokio::spawn(async move {
            let _ = conn.await;
        });
        out.push(client);
    }
    out
}

/// Drive `N` connections concurrently. Each connection sends
/// `REQUESTS_PER_CONNECTION` requests sequentially.
async fn drive_concurrent(connections: Vec<h2::client::SendRequest<Bytes>>) {
    let mut handles = Vec::with_capacity(connections.len());
    for client in connections {
        handles.push(tokio::spawn(async move {
            for _ in 0..REQUESTS_PER_CONNECTION {
                one_request(client.clone()).await;
            }
        }));
    }
    for handle in handles {
        let _ = handle.await;
    }
}

fn start_tokio_multi_thread() -> std::net::SocketAddr {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(CORES)
        .enable_all()
        .build()
        .expect("server runtime");
    let pipe = HashPipe { payload: payload() };
    let dispatch: PipeHandle = into_handle(pipe);
    let (addr_tx, addr_rx) = std::sync::mpsc::channel::<std::net::SocketAddr>();
    runtime.spawn(async move {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        addr_tx.send(addr).expect("addr send");
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
    let addr = addr_rx.recv().expect("addr");
    std::mem::forget(runtime);
    addr
}

/// Prime with explicit round-robin connection dispatch via `spawn_on_core`.
/// The accept loop runs on core 0, but each connection task is dispatched
/// to a core chosen by index — this exercises prime's cross-core spawn
/// path under the connection-lifetime workload (vs `cross_core_spawn`'s
/// fire-and-forget pattern).
fn start_prime_fanout() -> std::net::SocketAddr {
    let runtime: Arc<dyn Runtime> = Arc::new(PrimeRuntime::new(CORES).expect("prime runtime"));
    let pipe = HashPipe { payload: payload() };
    let dispatch: PipeHandle = into_handle(pipe);
    let (addr_tx, addr_rx) = std::sync::mpsc::channel::<std::net::SocketAddr>();
    let connection_counter = Arc::new(AtomicUsize::new(0));
    let runtime_clone = runtime.clone();
    runtime
        .spawn_factory_on_core(
            CoreId(0),
            Box::new(move || {
                let dispatch = dispatch;
                let connection_counter = connection_counter;
                let runtime_inner = runtime_clone;
                Box::pin(async move {
                    let mut listener =
                        ProximaTcpListener::bind("127.0.0.1:0".parse().unwrap()).expect("bind");
                    let addr = listener.local_addr().expect("local_addr");
                    addr_tx.send(addr).expect("addr send");
                    loop {
                        let (socket, _peer) = match listener.accept().await {
                            Ok(value) => value,
                            Err(_) => break,
                        };
                        let dispatch = dispatch.clone();
                        let idx = connection_counter.fetch_add(1, Ordering::AcqRel);
                        let target_core = CoreId(idx % CORES);
                        // NOTE: this is the round-robin variant. Prime's
                        // TcpStream is `Send` so we can hand it across cores
                        // via spawn_factory_on_core. Each connection's I/O
                        // events then fire on the target core's reactor.
                        if target_core.0 == 0 {
                            // Same core: skip the cross-core hop entirely.
                            proxima::runtime::prime::os::core_shard::spawn_on_current_core(
                                Box::pin(async move {
                                    let in_flight = Arc::new(AtomicU64::new(0));
                                    let quiesce = Arc::new(QuiesceResponse {
                                        status: 503,
                                        retry_after: "1".into(),
                                    });
                                    let _ = serve_h2_connection(
                                        socket, dispatch, in_flight, quiesce, None,
                                    )
                                    .await;
                                }),
                            );
                        } else {
                            // Cross-core: send the socket to the target
                            // core's executor. The socket's reactor pointer
                            // will be re-established on first poll on the
                            // new core (per the net.rs SAFETY contract).
                            // Connection dispatch under load — if the target
                            // core's inbox is saturated, log and drop. Bench
                            // measures behavior under bounded queueing.
                            if let Err(err) = runtime_inner.spawn_factory_on_core(
                                target_core,
                                Box::new(move || {
                                    let dispatch = dispatch;
                                    Box::pin(async move {
                                        let in_flight = Arc::new(AtomicU64::new(0));
                                        let quiesce = Arc::new(QuiesceResponse {
                                            status: 503,
                                            retry_after: "1".into(),
                                        });
                                        let _ = serve_h2_connection(
                                            socket, dispatch, in_flight, quiesce, None,
                                        )
                                        .await;
                                    })
                                        as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
                                }),
                            ) {
                                eprintln!("fairness bench: cross-core dispatch dropped: {err}");
                            }
                        }
                    }
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }),
        )
        .expect("bench setup: spawn listener factory on fresh runtime");
    let addr = addr_rx.recv().expect("addr");
    std::mem::forget(runtime);
    addr
}

fn benches(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("fairness_imbalanced");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));
    // throughput-per-element here is per-request, so we can read off
    // per-request latency from criterion's element timing.
    group.throughput(Throughput::Elements(TOTAL_REQUESTS as u64));

    let client_runtime = build_client_runtime();

    let tokio_addr = start_tokio_multi_thread();
    let tokio_pool =
        client_runtime.block_on(async move { warm_connections(tokio_addr, CONNECTIONS).await });
    group.bench_function("tokio_multi_thread", |bencher| {
        bencher.to_async(&client_runtime).iter(|| {
            let pool: Vec<_> = tokio_pool.to_vec();
            drive_concurrent(pool)
        });
    });

    let prime_addr = start_prime_fanout();
    let prime_pool =
        client_runtime.block_on(async move { warm_connections(prime_addr, CONNECTIONS).await });
    group.bench_function("prime_round_robin_fanout", |bencher| {
        bencher.to_async(&client_runtime).iter(|| {
            let pool: Vec<_> = prime_pool.to_vec();
            drive_concurrent(pool)
        });
    });

    group.finish();
}

// Skewed-cost fairness bench.
//
// Uniform CPU cost (the `benches` group above) hides a real scheduling
// tradeoff: when 10% of requests are 100x more expensive than the other
// 90%, a work-stealing scheduler can absorb the expensive tail by
// moving cheap work off a blocked core. A per-core scheduler cannot —
// it pays the tail in isolation.
//
// Workload:
//   - SKEW_CONNECTIONS concurrent h2 connections to one server.
//   - SKEW_TOTAL_REQUESTS total requests, distributed round-robin across
//     connections.
//   - 90% of requests: allocate 1 KB and return immediately (cheap).
//   - 10% of requests: busy-loop for ~100 µs (expensive).
//   - Request identity (cheap vs expensive) is determined by request
//     counter mod 10 — deterministic, not random, so results are
//     reproducible across runs.
//   - Measure: completion time of the last request (tail-completion)
//     via iter_custom; p99/p999 latency via hdrhistogram.
//
// Arms:
//   - proxima_per_core: TokioPerCoreRuntime — work-stealing-free,
//     per-core queues. Cheap mean, expensive tail.
//   - proxima_prime_native: PrimeRuntime — same per-core model via
//     prime's executor. Validates prime vs tokio-per-core parity.
//   - tokio_multi_thread: default tokio with work-stealing. Should
//     win on tail because stolen cheap tasks unblock the expensive ones.

const SKEW_CONNECTIONS: usize = 16;
const SKEW_TOTAL_REQUESTS: usize = 10_000;

// busy-loop targeting ~100 µs on a modern core (calibrated for M1/M2).
// the loop body is pure integer work — no sleeps, no yields — so the
// core is genuinely saturated for the duration. thread::sleep would
// yield the OS thread, making this look like an I/O wait, not CPU cost.
#[inline(never)]
fn expensive_work() {
    let target = Duration::from_micros(100);
    let start = Instant::now();
    let mut acc: u64 = 1;
    while start.elapsed() < target {
        for _ in 0..256 {
            acc = acc
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
        }
    }
    std::hint::black_box(acc);
}

struct SkewedCostPipe {
    counter: Arc<AtomicUsize>,
}

impl SendPipe for SkewedCostPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl std::future::Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let index = self.counter.fetch_add(1, Ordering::Relaxed);
        async move {
            if index.is_multiple_of(10) {
                // 10% expensive: busy-loop ~100 µs
                expensive_work();
            }
            // 90% cheap: allocate 1 KB response body
            let body = vec![0u8; 1024];
            Ok(Response::ok(Bytes::from(body)))
        }
    }
}


async fn drive_skewed(connections: Vec<h2::client::SendRequest<Bytes>>) {
    let requests_per_conn = SKEW_TOTAL_REQUESTS / connections.len();
    let mut handles = Vec::with_capacity(connections.len());
    for client in connections {
        handles.push(tokio::spawn(async move {
            for _ in 0..requests_per_conn {
                one_request(client.clone()).await;
            }
        }));
    }
    for handle in handles {
        let _ = handle.await;
    }
}

fn start_skewed_tokio_multi_thread() -> std::net::SocketAddr {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(CORES)
        .enable_all()
        .build()
        .expect("server runtime");
    let counter = Arc::new(AtomicUsize::new(0));
    let pipe: PipeHandle = into_handle(SkewedCostPipe { counter });
    let (addr_tx, addr_rx) = std::sync::mpsc::channel::<std::net::SocketAddr>();
    runtime.spawn(async move {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        addr_tx.send(addr).expect("addr send");
        loop {
            let (socket, _) = match listener.accept().await {
                Ok(value) => value,
                Err(_) => break,
            };
            let _ = socket.set_nodelay(true);
            let pipe = pipe.clone();
            tokio::spawn(async move {
                let in_flight = Arc::new(AtomicU64::new(0));
                let quiesce = Arc::new(QuiesceResponse {
                    status: 503,
                    retry_after: "1".into(),
                });
                let _ = serve_h2_connection(socket.compat(), pipe, in_flight, quiesce, None).await;
            });
        }
    });
    let addr = addr_rx.recv().expect("addr");
    std::mem::forget(runtime);
    addr
}

fn start_skewed_per_core() -> std::net::SocketAddr {
    let runtime = TokioPerCoreRuntime::new(CORES).expect("per-core runtime");
    let counter = Arc::new(AtomicUsize::new(0));
    let connection_counter = Arc::new(AtomicUsize::new(0));
    let (addr_tx, addr_rx) = std::sync::mpsc::channel::<std::net::SocketAddr>();
    for core_index in 0..CORES {
        let pipe: PipeHandle = into_handle(SkewedCostPipe {
            counter: counter.clone(),
        });
        let addr_tx = if core_index == 0 {
            Some(addr_tx.clone())
        } else {
            None
        };
        let conn_counter = connection_counter.clone();
        runtime
            .spawn_factory_on_core(
                CoreId(core_index),
                Box::new(move || {
                    Box::pin(async move {
                        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
                        if let Some(tx) = addr_tx {
                            let addr = listener.local_addr().expect("addr");
                            tx.send(addr).expect("addr send");
                        }
                        loop {
                            let (socket, _) = match listener.accept().await {
                                Ok(value) => value,
                                Err(_) => break,
                            };
                            let _ = socket.set_nodelay(true);
                            let pipe = pipe.clone();
                            let _ = conn_counter.fetch_add(1, Ordering::Relaxed);
                            tokio::task::spawn_local(async move {
                                let in_flight = Arc::new(AtomicU64::new(0));
                                let quiesce = Arc::new(QuiesceResponse {
                                    status: 503,
                                    retry_after: "1".into(),
                                });
                                let _ = serve_h2_connection(
                                    socket.compat(),
                                    pipe,
                                    in_flight,
                                    quiesce,
                                    None,
                                )
                                .await;
                            });
                        }
                    })
                        as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
                }),
            )
            .expect("spawn per-core skewed accept");
    }
    let addr = addr_rx.recv().expect("addr");
    std::mem::forget(runtime);
    addr
}

fn start_skewed_prime_native() -> std::net::SocketAddr {
    let runtime: Arc<dyn Runtime> = Arc::new(PrimeRuntime::new(CORES).expect("prime runtime"));
    let counter = Arc::new(AtomicUsize::new(0));
    let pipe: PipeHandle = into_handle(SkewedCostPipe { counter });
    let (addr_tx, addr_rx) = std::sync::mpsc::channel::<std::net::SocketAddr>();
    let connection_counter = Arc::new(AtomicUsize::new(0));
    let runtime_clone = runtime.clone();
    runtime
        .spawn_factory_on_core(
            CoreId(0),
            Box::new(move || {
                let dispatch = pipe;
                let conn_counter = connection_counter;
                let rt = runtime_clone;
                Box::pin(async move {
                    let mut listener =
                        ProximaTcpListener::bind("127.0.0.1:0".parse().unwrap()).expect("bind");
                    let addr = listener.local_addr().expect("local_addr");
                    addr_tx.send(addr).expect("addr send");
                    loop {
                        let (socket, _) = match listener.accept().await {
                            Ok(value) => value,
                            Err(_) => break,
                        };
                        let dispatch = dispatch.clone();
                        let idx = conn_counter.fetch_add(1, Ordering::AcqRel);
                        let target_core = CoreId(idx % CORES);
                        if target_core.0 == 0 {
                            proxima::runtime::prime::os::core_shard::spawn_on_current_core(
                                Box::pin(async move {
                                    let in_flight = Arc::new(AtomicU64::new(0));
                                    let quiesce = Arc::new(QuiesceResponse {
                                        status: 503,
                                        retry_after: "1".into(),
                                    });
                                    let _ = serve_h2_connection(
                                        socket, dispatch, in_flight, quiesce, None,
                                    )
                                    .await;
                                }),
                            );
                        } else {
                            if let Err(err) = rt.spawn_factory_on_core(
                                target_core,
                                Box::new(move || {
                                    let dispatch = dispatch;
                                    Box::pin(async move {
                                        let in_flight = Arc::new(AtomicU64::new(0));
                                        let quiesce = Arc::new(QuiesceResponse {
                                            status: 503,
                                            retry_after: "1".into(),
                                        });
                                        let _ = serve_h2_connection(
                                            socket, dispatch, in_flight, quiesce, None,
                                        )
                                        .await;
                                    })
                                        as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
                                }),
                            ) {
                                eprintln!("skewed-cost prime: cross-core dispatch dropped: {err}");
                            }
                        }
                    }
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }),
        )
        .expect("spawn skewed prime listener");
    let addr = addr_rx.recv().expect("addr");
    std::mem::forget(runtime);
    addr
}

fn bench_fairness_skewed_cost(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("fairness_skewed_cost");
    group.sample_size(15);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(6));
    group.throughput(Throughput::Elements(SKEW_TOTAL_REQUESTS as u64));

    let client_runtime = build_client_runtime();

    let tokio_addr = start_skewed_tokio_multi_thread();
    let per_core_addr = start_skewed_per_core();
    let prime_addr = start_skewed_prime_native();

    macro_rules! skew_arm {
        ($name:expr, $addr:expr) => {{
            let addr = $addr;
            let pool = client_runtime.block_on(warm_connections(addr, SKEW_CONNECTIONS));
            let mut quartet = HdrQuartet::new();
            group.bench_function($name, |bencher| {
                bencher.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for idx in 0..iters {
                        let connections: Vec<_> = pool.iter().cloned().collect();
                        let start = Instant::now();
                        client_runtime.block_on(drive_skewed(connections));
                        let elapsed = start.elapsed();
                        quartet.record(idx, elapsed.as_nanos().max(1) as u64);
                        total += elapsed;
                    }
                    quartet.finalize(iters);
                    total
                });
            });
            quartet.report($name);
        }};
    }

    skew_arm!("tokio_multi_thread", tokio_addr);
    skew_arm!("proxima_per_core", per_core_addr);
    skew_arm!("proxima_prime_native", prime_addr);

    group.finish();
}

criterion_group!(fairness, benches, bench_fairness_skewed_cost);
criterion_main!(fairness);
