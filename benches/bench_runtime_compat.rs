//! P2 — Prime+Tokio compat mode bench harness.
//!
//! Three arms (tokio multi-thread baseline / pure prime / prime-tokio-compat)
//! × five workloads (per-core spawn / h2 fan-in / streaming response /
//! tokio::sync::Mutex contention / single-stream GET). The third arm is
//! gated behind `feature = "prime-tokio-compat"` and only attaches once
//! P2.b plumbing has landed.
//!
//! The harness exists to drive the ship-criteria decision in P2.e — every
//! number here goes straight into `rust/docs/runtime-prime/discipline-prime-tokio-compat.md`
//! with the delta vs. prior. CI is the source of truth; this file produces
//! the inputs.
//!
//! Workloads cover the dimensions compat is expected to cost on and the
//! dimensions where it must not regress past pure-prime / tokio:
//!
//! - **W1 per-core spawn** — compat's expected weakness (nested scheduler).
//! - **W2 h2 multi-stream fan-in** — proxima's production wedge; must not
//!   regress.
//! - **W3 streaming response throughput** — reactor-bound; compat should
//!   approach pure-prime.
//! - **W4 tokio::sync::Mutex contention** — nested-scheduler cost on
//!   contended sync.
//! - **W5 single-stream simple GET** — prime's known weakness; sanity
//!   that compat doesn't make it worse.
//!
//! ```bash
//! cargo bench -p proxima --bench bench_runtime_compat
//! cargo bench -p proxima --bench bench_runtime_compat -- w1_per_core_spawn
//! ```

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
#![cfg(all(
    feature = "http2",
    feature = "tcp",
    feature = "http1",
    feature = "runtime-tokio",
    all(
        feature = "runtime-prime-executor",
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-reactor",
        feature = "runtime-prime-bgpool"
    ),
))]

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use futures::future::join_all;
use http_body_util::Full;
use hyper::server::conn::http2;
use hyper::service::service_fn as pipe_fn;
use hyper_util::rt::{TokioExecutor, TokioIo};
use proxima::error::ProximaError;
use proxima::h2::serve_h2_connection;
use proxima::listeners::http::QuiesceResponse;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::request::{Request, Response};
use proxima::runtime::prime::os::net::TcpListener as ProximaTcpListener;
use proxima::runtime::{
    CoreId, PrimeRuntime, Runtime, TokioPerCoreRuntime, spawn_on_core_blocking_with,
};
use proxima_primitives::pipe::SendPipe;
use tokio::net::{TcpListener, TcpStream};
use tokio_util::compat::TokioAsyncReadCompatExt;

const CORES: usize = 4;
const STREAMING_BODY_BYTES: usize = 64 * 1024;
const FANIN_STREAMS: usize = 32;
const MUTEX_OPS_PER_THREAD: usize = 4_000;
const MUTEX_THREADS: usize = 4;

fn configure_group<M: criterion::measurement::Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
) {
    group.sample_size(15);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
}

// ---------------------------------------------------------------------
// W1 — per-core spawn throughput.
//
// Each arm uses its runtime's NATIVE spawn API — measuring what a user
// of that runtime would actually call. For compat, that means dispatching
// to the sister tokio handles directly (the "tokio::spawn from inside
// a compat-mode prime task" path), NOT calling prime's spawn_on_core
// (which would route through prime's inbox and leave the EnterGuard
// unused).
//
// - tokio_per_core: external thread → `runtime.spawn_on_core` → tokio
//   current-thread executor on that core.
// - prime: external thread → `runtime.spawn_on_core` → prime inbox →
//   per-core executor.
// - prime_tokio_compat: external thread → `tokio_handle.spawn` (per-core
//   sister handle) → sister current-thread executor on its own thread.
// ---------------------------------------------------------------------

const SPAWN_BURST_COUNT: usize = 4_000;

fn spawn_burst_dyn(runtime: &Arc<dyn Runtime>) -> Duration {
    let counter = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    for index in 0..SPAWN_BURST_COUNT {
        let counter = counter.clone();
        let core = CoreId(index % CORES);
        let _ = spawn_on_core_blocking_with(runtime.as_ref(), core, move || {
            let counter = counter.clone();
            Box::pin(async move {
                counter.fetch_add(1, Ordering::AcqRel);
            })
        });
    }
    while counter.load(Ordering::Acquire) < SPAWN_BURST_COUNT {
        std::hint::spin_loop();
    }
    started.elapsed()
}

/// Compat-native spawn burst — dispatches via the per-core sister
/// tokio `Handle`, not via prime's inbox. This is the path user code
/// using `tokio::spawn` actually takes inside a compat-mode runtime
/// (the EnterGuard routes `tokio::spawn` to whichever sister handle
/// the current prime worker has entered).
#[cfg(feature = "prime-tokio-compat")]
fn spawn_burst_compat(runtime: &Arc<PrimeRuntime>) -> Duration {
    let handles = runtime
        .tokio_compat_handles()
        .expect("compat runtime missing sister handles")
        .clone();
    let counter = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    for index in 0..SPAWN_BURST_COUNT {
        let core = CoreId(index % CORES);
        let handle = handles.handle(core).expect("sister handle for core");
        let counter = counter.clone();
        // detached: completion is observed via the shared counter, not the
        // join handle. dropping the handle keeps the task running.
        drop(handle.spawn(async move {
            counter.fetch_add(1, Ordering::AcqRel);
        }));
    }
    while counter.load(Ordering::Acquire) < SPAWN_BURST_COUNT {
        std::hint::spin_loop();
    }
    started.elapsed()
}

/// Compat BATCHED spawn burst — dispatches via `TokioCompatHandles::spawn_on_core`
/// (per-core channel → the sister's own drain loop `tokio::task::spawn`s each task
/// locally) instead of `Handle::spawn` per task. The syscall-light path: ~one
/// `unpark` per burst instead of one `kevent` per task, mirroring how
/// `TokioPerCoreRuntime` batches through flume. `spawn_burst_compat` above keeps
/// the unbatched `Handle::spawn`-per-task arm so the delta is visible.
#[cfg(feature = "prime-tokio-compat")]
fn spawn_burst_compat_batched(runtime: &Arc<PrimeRuntime>) -> Duration {
    let handles = runtime
        .tokio_compat_handles()
        .expect("compat runtime missing sister handles")
        .clone();
    let counter = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    for index in 0..SPAWN_BURST_COUNT {
        let core = CoreId(index % CORES);
        let counter = counter.clone();
        let _ = handles.spawn_on_core(
            core,
            Box::pin(async move {
                counter.fetch_add(1, Ordering::AcqRel);
            }),
        );
    }
    while counter.load(Ordering::Acquire) < SPAWN_BURST_COUNT {
        std::hint::spin_loop();
    }
    started.elapsed()
}

/// Compat INVERTED spawn burst (design D2) — the REAL transparent path: a
/// prime task dispatched onto each inverted worker loops raw `tokio::spawn`
/// of the counter-increment task. Because the inverted worker ticks the prime
/// executor inside `sister.block_on(...)`, those `tokio::spawn`s take tokio's
/// LOCAL fast path (no per-spawn kevent). This is NOT the opt-in
/// `spawn_on_core` / sister-`Handle::spawn` path — it is what user code with
/// raw `tokio::spawn` actually hits inside an inverted-compat runtime.
#[cfg(feature = "prime-tokio-compat-inverted")]
fn spawn_burst_compat_inverted(runtime: &Arc<PrimeRuntime>) -> Duration {
    let counter = Arc::new(AtomicUsize::new(0));
    let per_core = SPAWN_BURST_COUNT / CORES;
    let started = Instant::now();
    for core_index in 0..CORES {
        let counter = counter.clone();
        let _ = runtime.spawn_factory_on_core(
            CoreId(core_index),
            Box::new(move || {
                Box::pin(async move {
                    for _ in 0..per_core {
                        let counter = counter.clone();
                        // raw tokio::spawn from inside a prime task — LOCAL on
                        // the inverted worker's own sister runtime.
                        tokio::spawn(async move {
                            counter.fetch_add(1, Ordering::AcqRel);
                        });
                    }
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }),
        );
    }
    while counter.load(Ordering::Acquire) < per_core * CORES {
        std::hint::spin_loop();
    }
    started.elapsed()
}

fn bench_w1_per_core_spawn(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("w1_per_core_spawn");
    configure_group(&mut group);
    group.throughput(Throughput::Elements(SPAWN_BURST_COUNT as u64));

    group.bench_function("tokio_per_core", |bencher| {
        let runtime: Arc<dyn Runtime> =
            Arc::new(TokioPerCoreRuntime::new(CORES).expect("tokio_per_core"));
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                total += spawn_burst_dyn(&runtime);
            }
            total
        });
    });

    group.bench_function("prime", |bencher| {
        let runtime: Arc<dyn Runtime> = Arc::new(PrimeRuntime::new(CORES).expect("prime"));
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                total += spawn_burst_dyn(&runtime);
            }
            total
        });
    });

    #[cfg(feature = "prime-tokio-compat")]
    group.bench_function("prime_tokio_compat", |bencher| {
        let runtime: Arc<PrimeRuntime> = Arc::new(
            PrimeRuntime::builder()
                .cores(CORES)
                .background_inline()
                .tokio_compat()
                .build()
                .expect("prime-tokio-compat"),
        );
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                total += spawn_burst_compat(&runtime);
            }
            total
        });
    });

    #[cfg(feature = "prime-tokio-compat")]
    group.bench_function("prime_tokio_compat_batched", |bencher| {
        let runtime: Arc<PrimeRuntime> = Arc::new(
            PrimeRuntime::builder()
                .cores(CORES)
                .background_inline()
                .tokio_compat()
                .build()
                .expect("prime-tokio-compat"),
        );
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                total += spawn_burst_compat_batched(&runtime);
            }
            total
        });
    });

    #[cfg(feature = "prime-tokio-compat-inverted")]
    group.bench_function("prime_tokio_compat_inverted", |bencher| {
        let runtime: Arc<PrimeRuntime> = Arc::new(
            PrimeRuntime::builder()
                .cores(CORES)
                .background_inline()
                .tokio_compat_inverted()
                .build()
                .expect("prime-tokio-compat-inverted"),
        );
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                total += spawn_burst_compat_inverted(&runtime);
            }
            total
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------
// W4 — tokio::sync::Mutex contention across CORES (4 by default).
//
// MUTEX_THREADS producer threads each loop MUTEX_OPS_PER_THREAD times
// over `runtime.spawn_on_core(...)` of a closure that locks the
// shared `tokio::sync::Mutex`, increments, and releases. The compat
// arm runs these locks INSIDE a nested tokio current-thread runtime —
// the contention path picks up the nested-scheduler cost on every wake.
//
// For the pure-prime arm we substitute `futures::lock::Mutex` (no tokio
// context required). The comparison is "what's the steady-state cost of
// contended async sync under each runtime's natural primitive". Compat
// vs. tokio is the ship-criteria delta (compat must be ≥ tokio).
// ---------------------------------------------------------------------

const MUTEX_TOTAL_OPS: u64 = (MUTEX_THREADS * MUTEX_OPS_PER_THREAD) as u64;

fn run_mutex_tokio(runtime: &Arc<dyn Runtime>) -> Duration {
    let lock = Arc::new(tokio::sync::Mutex::new(0u64));
    let done = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    for thread_index in 0..MUTEX_THREADS {
        let lock_outer = lock.clone();
        let done_outer = done.clone();
        let core = CoreId(thread_index % CORES);
        let _ = spawn_on_core_blocking_with(runtime.as_ref(), core, move || {
            let lock_inner = lock_outer.clone();
            let done_inner = done_outer.clone();
            Box::pin(async move {
                for _ in 0..MUTEX_OPS_PER_THREAD {
                    let mut guard = lock_inner.lock().await;
                    *guard += 1;
                    drop(guard);
                }
                done_inner.fetch_add(1, Ordering::AcqRel);
            })
        });
    }
    while done.load(Ordering::Acquire) < MUTEX_THREADS {
        std::hint::spin_loop();
    }
    started.elapsed()
}

fn run_mutex_futures(runtime: &Arc<dyn Runtime>) -> Duration {
    let lock = Arc::new(futures::lock::Mutex::new(0u64));
    let done = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    for thread_index in 0..MUTEX_THREADS {
        let lock_outer = lock.clone();
        let done_outer = done.clone();
        let core = CoreId(thread_index % CORES);
        let _ = spawn_on_core_blocking_with(runtime.as_ref(), core, move || {
            let lock_inner = lock_outer.clone();
            let done_inner = done_outer.clone();
            Box::pin(async move {
                for _ in 0..MUTEX_OPS_PER_THREAD {
                    let mut guard = lock_inner.lock().await;
                    *guard += 1;
                    drop(guard);
                }
                done_inner.fetch_add(1, Ordering::AcqRel);
            })
        });
    }
    while done.load(Ordering::Acquire) < MUTEX_THREADS {
        std::hint::spin_loop();
    }
    started.elapsed()
}

fn bench_w4_mutex_contention(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("w4_mutex_contention");
    configure_group(&mut group);
    group.throughput(Throughput::Elements(MUTEX_TOTAL_OPS));

    group.bench_function("tokio_per_core", |bencher| {
        let runtime: Arc<dyn Runtime> =
            Arc::new(TokioPerCoreRuntime::new(CORES).expect("tokio_per_core"));
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                total += run_mutex_tokio(&runtime);
            }
            total
        });
    });

    group.bench_function("prime", |bencher| {
        let runtime: Arc<dyn Runtime> = Arc::new(PrimeRuntime::new(CORES).expect("prime"));
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                total += run_mutex_futures(&runtime);
            }
            total
        });
    });

    #[cfg(feature = "prime-tokio-compat")]
    group.bench_function("prime_tokio_compat", |bencher| {
        let runtime: Arc<dyn Runtime> = Arc::new(
            PrimeRuntime::builder()
                .cores(CORES)
                .background_inline()
                .tokio_compat()
                .build()
                .expect("prime-tokio-compat"),
        );
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                total += run_mutex_tokio(&runtime);
            }
            total
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------
// H/2 server starters. Cribbed from `h2_runtime_swap.rs`; each starter
// returns the bound address. The accompanying tokio runtime is leaked
// to keep server tasks alive for the lifetime of the bench process.
// ---------------------------------------------------------------------

const SMALL_BODY: &[u8] = b"ok";
const STREAMING_BODY: &[u8] = &[0xAB; STREAMING_BODY_BYTES];

struct ConstantOk;
impl SendPipe for ConstantOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl std::future::Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move { Ok(Response::ok(Bytes::from_static(SMALL_BODY))) }
    }
}


struct StreamingOk;
impl SendPipe for StreamingOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl std::future::Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move { Ok(Response::ok(Bytes::from_static(STREAMING_BODY))) }
    }
}


async fn hyper_small_handler(
    _request: hyper::Request<hyper::body::Incoming>,
) -> Result<hyper::Response<Full<Bytes>>, Infallible> {
    Ok(hyper::Response::builder()
        .status(200)
        .body(Full::new(Bytes::from_static(SMALL_BODY)))
        .expect("response"))
}

fn build_client_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("client runtime")
}

/// Starts the proxima native h2 server on a dedicated tokio multi-thread
/// runtime — the "tokio baseline" for h2-shaped workloads.
fn start_h2_tokio(pipe: PipeHandle) -> std::net::SocketAddr {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(CORES)
        .enable_all()
        .build()
        .expect("tokio server runtime");
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
            let pipe_for_conn = pipe.clone();
            tokio::spawn(async move {
                let in_flight = Arc::new(AtomicU64::new(0));
                let quiesce = Arc::new(QuiesceResponse {
                    status: 503,
                    retry_after: "1".into(),
                });
                let _ =
                    serve_h2_connection(socket.compat(), pipe_for_conn, in_flight, quiesce, None)
                        .await;
            });
        }
    });
    std::mem::forget(runtime);
    addr
}

/// Starts proxima native h2 on `PrimeRuntime`. No tokio runtime context;
/// the prime executor drives accept + connection futures.
fn start_h2_prime(pipe: PipeHandle) -> std::net::SocketAddr {
    let runtime = PrimeRuntime::new(CORES).expect("prime runtime");
    let (addr_tx, addr_rx) = std::sync::mpsc::channel::<std::net::SocketAddr>();
    runtime
        .spawn_factory_on_core(
            CoreId(0),
            Box::new(move || {
                let pipe = pipe;
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
                        let pipe = pipe.clone();
                        proxima::runtime::prime::os::core_shard::spawn_on_current_core(Box::pin(
                            async move {
                                let in_flight = Arc::new(AtomicU64::new(0));
                                let quiesce = Arc::new(QuiesceResponse {
                                    status: 503,
                                    retry_after: "1".into(),
                                });
                                let _ = serve_h2_connection(socket, pipe, in_flight, quiesce, None)
                                    .await;
                            },
                        ));
                    }
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }),
        )
        .expect("spawn h2 listener factory");
    let addr = addr_rx.recv().expect("addr from prime worker");
    std::mem::forget(runtime);
    addr
}

/// Starts proxima native h2 on a compat-mode `PrimeRuntime`. Critically,
/// this uses `tokio::net::TcpListener` + `tokio::spawn` for accept +
/// connection handling — the natural tokio code path. The prime worker
/// task that hosts the accept loop has the sister tokio `Handle`
/// entered via the runtime's per-worker `EnterGuard`; `tokio::net::bind`
/// registers with the sister's mio reactor; `tokio::spawn` dispatches
/// handlers to the sister's task queue. This is what user code drops
/// onto compat — and what we are measuring.
#[cfg(feature = "prime-tokio-compat")]
fn start_h2_prime_tokio_compat(pipe: PipeHandle) -> std::net::SocketAddr {
    let runtime = PrimeRuntime::builder()
        .cores(CORES)
        .background_inline()
        .tokio_compat()
        .build()
        .expect("prime-tokio-compat runtime");
    let (addr_tx, addr_rx) = std::sync::mpsc::channel::<std::net::SocketAddr>();
    runtime
        .spawn_factory_on_core(
            CoreId(0),
            Box::new(move || {
                let pipe = pipe;
                Box::pin(async move {
                    // tokio::net::TcpListener::bind requires a current tokio
                    // runtime context. compat mode's per-worker EnterGuard
                    // satisfies that — Handle::current() returns the sister.
                    let listener = TcpListener::bind("127.0.0.1:0")
                        .await
                        .expect("compat tokio TcpListener bind");
                    let addr = listener.local_addr().expect("local_addr");
                    addr_tx.send(addr).expect("addr send");
                    loop {
                        let (socket, _peer) = match listener.accept().await {
                            Ok(value) => value,
                            Err(_) => break,
                        };
                        let _ = socket.set_nodelay(true);
                        let pipe = pipe.clone();
                        // tokio::spawn → sister tokio current-thread executor
                        // (via EnterGuard's published Handle). The connection
                        // future runs on the sister OS thread, not this prime
                        // worker thread.
                        tokio::spawn(async move {
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
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }),
        )
        .expect("spawn h2 listener factory");
    let addr = addr_rx.recv().expect("addr from prime-tokio-compat worker");
    std::mem::forget(runtime);
    addr
}

/// E2E h2 serve on the INVERTED compat runtime (D2). Identical user code to
/// `start_h2_prime_tokio_compat` — `tokio::net::TcpListener` + `tokio::spawn`
/// inside a prime task — but the runtime is built with `tokio_compat_inverted`,
/// so the sister tokio runtime is driven IN-THREAD by the prime worker. This is
/// the gate's E2E arm: it exercises tokio::net accept + per-connection I/O on
/// the inverted worker (not just the spawn micro-bench).
#[cfg(feature = "prime-tokio-compat-inverted")]
fn start_h2_prime_tokio_compat_inverted(pipe: PipeHandle) -> std::net::SocketAddr {
    let runtime = PrimeRuntime::builder()
        .cores(CORES)
        .background_inline()
        .tokio_compat_inverted()
        .build()
        .expect("prime-tokio-compat-inverted runtime");
    let (addr_tx, addr_rx) = std::sync::mpsc::channel::<std::net::SocketAddr>();
    runtime
        .spawn_factory_on_core(
            CoreId(0),
            Box::new(move || {
                let pipe = pipe;
                Box::pin(async move {
                    let listener = TcpListener::bind("127.0.0.1:0")
                        .await
                        .expect("inverted compat tokio TcpListener bind");
                    let addr = listener.local_addr().expect("local_addr");
                    addr_tx.send(addr).expect("addr send");
                    loop {
                        let (socket, _peer) = match listener.accept().await {
                            Ok(value) => value,
                            Err(_) => break,
                        };
                        let _ = socket.set_nodelay(true);
                        let pipe = pipe.clone();
                        // tokio::spawn from a prime task on the inverted worker
                        // is LOCAL (runs as a sister task driven in-thread).
                        tokio::spawn(async move {
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
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }),
        )
        .expect("spawn inverted h2 listener factory");
    let addr = addr_rx.recv().expect("addr from inverted compat worker");
    std::mem::forget(runtime);
    addr
}

fn warm_h2_client(
    runtime: &tokio::runtime::Runtime,
    addr: std::net::SocketAddr,
) -> h2::client::SendRequest<Bytes> {
    runtime.block_on(async move {
        let socket = TcpStream::connect(addr).await.expect("connect");
        let _ = socket.set_nodelay(true);
        let (client, conn) = h2::client::handshake(socket).await.expect("handshake");
        tokio::spawn(async move {
            let _ = conn.await;
        });
        client
    })
}

async fn one_h2_request(mut client: h2::client::SendRequest<Bytes>) {
    let request = http::Request::builder()
        .method("GET")
        .uri("http://localhost/")
        .body(())
        .expect("request");
    let (response_future, _) = client.send_request(request, true).expect("send_request");
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

// ---------------------------------------------------------------------
// W2 — h2 multi-stream fan-in. Single connection, `FANIN_STREAMS`
// concurrent in-flight requests, measured per batch.
// ---------------------------------------------------------------------

fn bench_w2_h2_fanin(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("w2_h2_fanin");
    group.measurement_time(Duration::from_secs(4));
    group.sample_size(15);
    group.throughput(Throughput::Elements(FANIN_STREAMS as u64));

    let client_runtime = build_client_runtime();

    let tokio_addr = start_h2_tokio(into_handle(ConstantOk));
    let tokio_client = warm_h2_client(&client_runtime, tokio_addr);
    group.bench_function("tokio_per_core", |bencher| {
        bencher.to_async(&client_runtime).iter(|| {
            let client = tokio_client.clone();
            async move {
                let mut futs = Vec::with_capacity(FANIN_STREAMS);
                for _ in 0..FANIN_STREAMS {
                    futs.push(one_h2_request(client.clone()));
                }
                join_all(futs).await;
            }
        });
    });

    let prime_addr = start_h2_prime(into_handle(ConstantOk));
    let prime_client = warm_h2_client(&client_runtime, prime_addr);
    group.bench_function("prime", |bencher| {
        bencher.to_async(&client_runtime).iter(|| {
            let client = prime_client.clone();
            async move {
                let mut futs = Vec::with_capacity(FANIN_STREAMS);
                for _ in 0..FANIN_STREAMS {
                    futs.push(one_h2_request(client.clone()));
                }
                join_all(futs).await;
            }
        });
    });

    #[cfg(feature = "prime-tokio-compat")]
    {
        let compat_addr = start_h2_prime_tokio_compat(into_handle(ConstantOk));
        let compat_client = warm_h2_client(&client_runtime, compat_addr);
        group.bench_function("prime_tokio_compat", |bencher| {
            bencher.to_async(&client_runtime).iter(|| {
                let client = compat_client.clone();
                async move {
                    let mut futs = Vec::with_capacity(FANIN_STREAMS);
                    for _ in 0..FANIN_STREAMS {
                        futs.push(one_h2_request(client.clone()));
                    }
                    join_all(futs).await;
                }
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------
// W3 — streaming response throughput. Single h2 stream returning
// `STREAMING_BODY_BYTES` per request. Measures MB/s; reactor + frame
// writer cost dominates spawn overhead at this body size.
// ---------------------------------------------------------------------

fn bench_w3_streaming_response(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("w3_streaming_response");
    group.measurement_time(Duration::from_secs(4));
    group.sample_size(15);
    group.throughput(Throughput::Bytes(STREAMING_BODY_BYTES as u64));

    let client_runtime = build_client_runtime();

    let tokio_addr = start_h2_tokio(into_handle(StreamingOk));
    let tokio_client = warm_h2_client(&client_runtime, tokio_addr);
    group.bench_function("tokio_per_core", |bencher| {
        bencher.to_async(&client_runtime).iter(|| {
            let client = tokio_client.clone();
            one_h2_request(client)
        });
    });

    let prime_addr = start_h2_prime(into_handle(StreamingOk));
    let prime_client = warm_h2_client(&client_runtime, prime_addr);
    group.bench_function("prime", |bencher| {
        bencher.to_async(&client_runtime).iter(|| {
            let client = prime_client.clone();
            one_h2_request(client)
        });
    });

    #[cfg(feature = "prime-tokio-compat")]
    {
        let compat_addr = start_h2_prime_tokio_compat(into_handle(StreamingOk));
        let compat_client = warm_h2_client(&client_runtime, compat_addr);
        group.bench_function("prime_tokio_compat", |bencher| {
            bencher.to_async(&client_runtime).iter(|| {
                let client = compat_client.clone();
                one_h2_request(client)
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------
// W5 — single-stream simple GET. The smallest viable h2 workload —
// one stream, one ~2-byte body, sequential issuance. Prime's known
// weakness (cross-core wake cost dominates).
// ---------------------------------------------------------------------

fn bench_w5_single_stream_get(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("w5_single_stream_get");
    group.measurement_time(Duration::from_secs(4));
    group.sample_size(15);
    group.throughput(Throughput::Elements(1));

    let client_runtime = build_client_runtime();

    let tokio_addr = start_h2_tokio(into_handle(ConstantOk));
    let tokio_client = warm_h2_client(&client_runtime, tokio_addr);
    group.bench_function("tokio_per_core", |bencher| {
        bencher.to_async(&client_runtime).iter(|| {
            let client = tokio_client.clone();
            one_h2_request(client)
        });
    });

    let prime_addr = start_h2_prime(into_handle(ConstantOk));
    let prime_client = warm_h2_client(&client_runtime, prime_addr);
    group.bench_function("prime", |bencher| {
        bencher.to_async(&client_runtime).iter(|| {
            let client = prime_client.clone();
            one_h2_request(client)
        });
    });

    #[cfg(feature = "prime-tokio-compat")]
    {
        let compat_addr = start_h2_prime_tokio_compat(into_handle(ConstantOk));
        let compat_client = warm_h2_client(&client_runtime, compat_addr);
        group.bench_function("prime_tokio_compat", |bencher| {
            bencher.to_async(&client_runtime).iter(|| {
                let client = compat_client.clone();
                one_h2_request(client)
            });
        });
    }

    // E2E gate arm: the same h2 serve on the INVERTED runtime — proves the
    // inverted worker drives tokio::net accept + per-connection I/O, not just
    // the spawn micro-bench.
    #[cfg(feature = "prime-tokio-compat-inverted")]
    {
        let inverted_addr = start_h2_prime_tokio_compat_inverted(into_handle(ConstantOk));
        let inverted_client = warm_h2_client(&client_runtime, inverted_addr);
        group.bench_function("prime_tokio_compat_inverted", |bencher| {
            bencher.to_async(&client_runtime).iter(|| {
                let client = inverted_client.clone();
                one_h2_request(client)
            });
        });
    }

    // hyper baseline keeps the comparison honest — proxima's h2 server on
    // its native tokio multi-thread runtime should beat hyper, otherwise
    // any compat-arm gap is a runtime issue, not a serve-loop issue.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(CORES)
        .enable_all()
        .build()
        .expect("hyper baseline runtime");
    let (hyper_addr, listener) = runtime.block_on(async {
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
            tokio::spawn(async move {
                let io = TokioIo::new(socket);
                let _ = http2::Builder::new(TokioExecutor::new())
                    .serve_connection(io, pipe_fn(hyper_small_handler))
                    .await;
            });
        }
    });
    std::mem::forget(runtime);
    let hyper_client = warm_h2_client(&client_runtime, hyper_addr);
    group.bench_function("hyper_tokio_baseline", |bencher| {
        bencher.to_async(&client_runtime).iter(|| {
            let client = hyper_client.clone();
            one_h2_request(client)
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------
// Bench registration. Five workloads × two-or-three arms depending on
// whether `prime-tokio-compat` is enabled.
// ---------------------------------------------------------------------

criterion_group!(
    benches,
    bench_w1_per_core_spawn,
    bench_w4_mutex_contention,
    bench_w2_h2_fanin,
    bench_w3_streaming_response,
    bench_w5_single_stream_get,
);
criterion_main!(benches);

// keep an unused-import suppressor for `thread` when the compat feature
// is off — `thread` is referenced only by helper paths added for the
// optimization loop.
#[allow(dead_code)]
fn _keep_thread_import_alive() {
    let _ = thread::current().id();
}
