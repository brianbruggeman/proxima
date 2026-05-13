//! Library × runtime matrix: can tokio-using server libraries (hyper,
//! pingora, proxima) actually serve traffic when their accept loop sits
//! on `PrimeRuntime::tokio_compat()`?
//!
//! Bench A (`bench_runtime_compat.rs`) measures the internal cost of
//! compat to proxima itself — same workload, three runtime arms. This
//! bench measures the **external utility** of compat — what does it
//! mean for someone running hyper or pingora on top of prime+compat?
//! The arms hold the workload (h2 GET) constant and vary both the
//! server library AND the runtime hosting it.
//!
//! Six arms, one workload:
//!
//! | server library | default tokio multi-thread | prime + compat |
//! |---|---|---|
//! | hyper          | `hyper_default_tokio`      | `hyper_prime_compat`   |
//! | pingora        | `pingora_default_tokio`    | `pingora_prime_compat` |
//! | proxima h2     | `proxima_default_tokio`    | `proxima_prime_compat` |
//!
//! For each row: the server library is unchanged. The accept loop
//! uses `tokio::net::TcpListener::accept`; the per-connection handler
//! is spawned via `tokio::spawn`. On the **prime_compat** column, the
//! whole accept-loop future runs inside a prime task that holds an
//! `EnterGuard` into the per-core sister tokio runtime. tokio's
//! `bind` / `accept` / `spawn` all resolve against the sister; the
//! handler future executes on the sister OS thread.
//!
//! ```bash
//! cargo bench -p proxima --features "..." --bench bench_compat_libraries
//! cargo bench -p proxima --features "..." --bench bench_compat_libraries -- hyper
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
    feature = "prime-tokio-compat",
))]

use std::convert::Infallible;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::{Duration, Instant};

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use futures::future::join_all;
use http_body_util::Full;

#[path = "common/hdr_phased.rs"]
mod hdr_phased;
use hdr_phased::HdrQuartet;
use hyper::server::conn::http2;
use hyper::service::service_fn as pipe_fn;
use hyper_util::rt::{TokioExecutor, TokioIo};
use proxima::error::ProximaError;
use proxima::h2::serve_h2_connection;
use proxima::listen_handle::bind_reuseport_listener;
use proxima::listeners::http::QuiesceResponse;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::request::{Request, Response};
use proxima::runtime::{CoreId, PrimeRuntime, Runtime, TokioPerCoreRuntime};
use proxima_primitives::pipe::SendPipe;
use tokio::net::TcpListener;
use tokio::net::TcpStream;
use tokio_util::compat::TokioAsyncReadCompatExt;

const CORES: usize = 4;
const RESPONSE_BODY: &[u8] = b"ok";
const FANIN_STREAMS: usize = 32;

fn configure_group<M: criterion::measurement::Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
) {
    group.sample_size(15);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(4));
}

fn build_client_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("client runtime")
}

// ---------------------------------------------------------------------
// Shared pipe + handler types.
// ---------------------------------------------------------------------

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


async fn hyper_handler(
    _request: hyper::Request<hyper::body::Incoming>,
) -> Result<hyper::Response<Full<Bytes>>, Infallible> {
    Ok(hyper::Response::builder()
        .status(200)
        .body(Full::new(Bytes::from_static(RESPONSE_BODY)))
        .expect("response"))
}

// ---------------------------------------------------------------------
// Hyper × default-tokio (baseline).
// ---------------------------------------------------------------------

fn start_hyper_default_tokio() -> std::net::SocketAddr {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(CORES)
        .enable_all()
        .build()
        .expect("hyper default-tokio server runtime");
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
            tokio::spawn(async move {
                let io = TokioIo::new(socket);
                let _ = http2::Builder::new(TokioExecutor::new())
                    .serve_connection(io, pipe_fn(hyper_handler))
                    .await;
            });
        }
    });
    std::mem::forget(runtime);
    addr
}

// ---------------------------------------------------------------------
// Hyper × tokio current-thread (control). Same accept-loop code as
// the multi-thread arm, but the runtime is a single-threaded
// current-thread runtime. This is the apples-to-apples baseline for
// the prime+compat arm — compat's sister is also a single
// current-thread runtime. If this matches compat, the original
// multi-thread vs compat "win" was current_thread-beats-multi_thread
// on a single-connection workload, NOT a compat-mode win.
// ---------------------------------------------------------------------

fn start_hyper_current_thread() -> std::net::SocketAddr {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("hyper current-thread server runtime");
    let (addr, listener) = runtime.block_on(async {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        (addr, listener)
    });
    let server_runtime = std::sync::Arc::new(runtime);
    let server_runtime_for_driver = server_runtime.clone();
    // current-thread runtimes don't run unless block_on or
    // a spawn() driven by an alive Handle exists. spawn the driver
    // loop on a dedicated OS thread that block_on's the accept loop.
    std::thread::Builder::new()
        .name("hyper-current-thread-driver".into())
        .spawn(move || {
            server_runtime_for_driver.block_on(async move {
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
        })
        .expect("spawn hyper current-thread driver");
    std::mem::forget(server_runtime);
    addr
}

// ---------------------------------------------------------------------
// Hyper × prime+compat — the same accept loop body runs inside a prime
// task with an EnterGuard active. `TcpListener::bind`, `accept`, and
// `tokio::spawn` all resolve against the per-core sister tokio runtime.
// ---------------------------------------------------------------------

fn start_hyper_prime_compat() -> std::net::SocketAddr {
    let runtime = PrimeRuntime::builder()
        .cores(CORES)
        .background_inline()
        .tokio_compat()
        .build()
        .expect("hyper prime-compat runtime");
    let (addr_tx, addr_rx) = std::sync::mpsc::channel::<std::net::SocketAddr>();
    runtime
        .spawn_factory_on_core(
            CoreId(0),
            Box::new(move || {
                Box::pin(async move {
                    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
                    let addr = listener.local_addr().expect("addr");
                    addr_tx.send(addr).expect("addr send");
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
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }),
        )
        .expect("spawn hyper accept loop on prime worker");
    let addr = addr_rx.recv().expect("addr from prime+compat hyper");
    std::mem::forget(runtime);
    addr
}

// ---------------------------------------------------------------------
// Pingora × default-tokio (baseline).
// ---------------------------------------------------------------------

async fn pingora_serve_one_connection(socket: TcpStream) {
    use pingora_core::protocols::Digest;
    use pingora_core::protocols::http::v2::server::{HttpSession, handshake as h2c_handshake};
    use pingora_http::ResponseHeader;

    let l4_stream: pingora_core::protocols::l4::stream::Stream = socket.into();
    let pingora_stream: pingora_core::protocols::Stream = Box::new(l4_stream);
    let mut h2_conn = match h2c_handshake(pingora_stream, None).await {
        Ok(conn) => conn,
        Err(_) => return,
    };
    let digest = Arc::new(Digest::default());
    loop {
        let session = match HttpSession::from_h2_conn(&mut h2_conn, digest.clone()).await {
            Ok(Some(session)) => session,
            Ok(None) | Err(_) => break,
        };
        tokio::spawn(async move {
            let mut session = session;
            let mut header = match ResponseHeader::build(200, None) {
                Ok(value) => value,
                Err(_) => return,
            };
            let _ = header.append_header("content-type", "text/plain");
            if session
                .write_response_header(Box::new(header), false)
                .is_err()
            {
                return;
            }
            let _ = session
                .write_body(Bytes::from_static(RESPONSE_BODY), true)
                .await;
            let _ = session.finish();
        });
    }
}

fn start_pingora_default_tokio() -> std::net::SocketAddr {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(CORES)
        .enable_all()
        .build()
        .expect("pingora default-tokio server runtime");
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
            tokio::spawn(async move {
                pingora_serve_one_connection(socket).await;
            });
        }
    });
    std::mem::forget(runtime);
    addr
}

// ---------------------------------------------------------------------
// Pingora × tokio current-thread (control). See hyper_current_thread.
// ---------------------------------------------------------------------

fn start_pingora_current_thread() -> std::net::SocketAddr {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("pingora current-thread server runtime");
    let (addr, listener) = runtime.block_on(async {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        (addr, listener)
    });
    let server_runtime = std::sync::Arc::new(runtime);
    let server_runtime_for_driver = server_runtime.clone();
    std::thread::Builder::new()
        .name("pingora-current-thread-driver".into())
        .spawn(move || {
            server_runtime_for_driver.block_on(async move {
                loop {
                    let (socket, _) = match listener.accept().await {
                        Ok(value) => value,
                        Err(_) => break,
                    };
                    let _ = socket.set_nodelay(true);
                    tokio::spawn(async move {
                        pingora_serve_one_connection(socket).await;
                    });
                }
            });
        })
        .expect("spawn pingora current-thread driver");
    std::mem::forget(server_runtime);
    addr
}

// ---------------------------------------------------------------------
// Pingora × prime+compat.
// ---------------------------------------------------------------------

fn start_pingora_prime_compat() -> std::net::SocketAddr {
    let runtime = PrimeRuntime::builder()
        .cores(CORES)
        .background_inline()
        .tokio_compat()
        .build()
        .expect("pingora prime-compat runtime");
    let (addr_tx, addr_rx) = std::sync::mpsc::channel::<std::net::SocketAddr>();
    runtime
        .spawn_factory_on_core(
            CoreId(0),
            Box::new(move || {
                Box::pin(async move {
                    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
                    let addr = listener.local_addr().expect("addr");
                    addr_tx.send(addr).expect("addr send");
                    loop {
                        let (socket, _) = match listener.accept().await {
                            Ok(value) => value,
                            Err(_) => break,
                        };
                        let _ = socket.set_nodelay(true);
                        tokio::spawn(async move {
                            pingora_serve_one_connection(socket).await;
                        });
                    }
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }),
        )
        .expect("spawn pingora accept loop on prime worker");
    let addr = addr_rx.recv().expect("addr from prime+compat pingora");
    std::mem::forget(runtime);
    addr
}

// ---------------------------------------------------------------------
// Proxima × default-tokio (baseline).
// ---------------------------------------------------------------------

fn start_proxima_default_tokio(pipe: PipeHandle) -> std::net::SocketAddr {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(CORES)
        .enable_all()
        .build()
        .expect("proxima default-tokio server runtime");
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
    std::mem::forget(runtime);
    addr
}

// ---------------------------------------------------------------------
// Proxima × tokio current-thread (control). See hyper_current_thread.
// ---------------------------------------------------------------------

fn start_proxima_current_thread(pipe: PipeHandle) -> std::net::SocketAddr {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("proxima current-thread server runtime");
    let (addr, listener) = runtime.block_on(async {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        (addr, listener)
    });
    let server_runtime = std::sync::Arc::new(runtime);
    let server_runtime_for_driver = server_runtime.clone();
    std::thread::Builder::new()
        .name("proxima-current-thread-driver".into())
        .spawn(move || {
            server_runtime_for_driver.block_on(async move {
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
                        let _ =
                            serve_h2_connection(socket.compat(), pipe, in_flight, quiesce, None)
                                .await;
                    });
                }
            });
        })
        .expect("spawn proxima current-thread driver");
    std::mem::forget(server_runtime);
    addr
}

// ---------------------------------------------------------------------
// Proxima × prime+compat.
// ---------------------------------------------------------------------

fn start_proxima_prime_compat(pipe: PipeHandle) -> std::net::SocketAddr {
    let runtime = PrimeRuntime::builder()
        .cores(CORES)
        .background_inline()
        .tokio_compat()
        .build()
        .expect("proxima prime-compat runtime");
    let (addr_tx, addr_rx) = std::sync::mpsc::channel::<std::net::SocketAddr>();
    runtime
        .spawn_factory_on_core(
            CoreId(0),
            Box::new(move || {
                let pipe = pipe;
                Box::pin(async move {
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
        .expect("spawn proxima accept loop on prime worker");
    let addr = addr_rx.recv().expect("addr from prime+compat proxima");
    std::mem::forget(runtime);
    addr
}

// ---------------------------------------------------------------------
// Client driver + h2 request.
// ---------------------------------------------------------------------

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
// Bench groups. Single-stream GET (simplest, most diagnostic) and
// h2 fan-in (more realistic — exercises concurrent handler dispatch
// through the compat path on every connection).
// ---------------------------------------------------------------------

fn bench_single_stream(criterion: &mut Criterion) {
    // known leak: 9 server runtimes leaked via std::mem::forget in start_* helpers,
    // bounded by the arm count (9) for this group's process lifetime.
    let mut group = criterion.benchmark_group("compat_libs_single_stream");
    configure_group(&mut group);
    group.throughput(Throughput::Elements(1));

    let client_runtime = build_client_runtime();

    macro_rules! single_arm {
        ($name:expr, $client:expr) => {{
            let client = $client;
            let mut quartet = HdrQuartet::new();
            group.bench_function($name, |bencher| {
                bencher.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for idx in 0..iters {
                        let start = Instant::now();
                        client_runtime.block_on(one_h2_request(client.clone()));
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

    single_arm!(
        "hyper_default_tokio",
        warm_h2_client(&client_runtime, start_hyper_default_tokio())
    );
    single_arm!(
        "hyper_current_thread",
        warm_h2_client(&client_runtime, start_hyper_current_thread())
    );
    single_arm!(
        "hyper_prime_compat",
        warm_h2_client(&client_runtime, start_hyper_prime_compat())
    );
    single_arm!(
        "pingora_default_tokio",
        warm_h2_client(&client_runtime, start_pingora_default_tokio())
    );
    single_arm!(
        "pingora_current_thread",
        warm_h2_client(&client_runtime, start_pingora_current_thread())
    );
    single_arm!(
        "pingora_prime_compat",
        warm_h2_client(&client_runtime, start_pingora_prime_compat())
    );
    single_arm!(
        "proxima_default_tokio",
        warm_h2_client(
            &client_runtime,
            start_proxima_default_tokio(into_handle(ConstantOk))
        )
    );
    single_arm!(
        "proxima_current_thread",
        warm_h2_client(
            &client_runtime,
            start_proxima_current_thread(into_handle(ConstantOk))
        )
    );
    single_arm!(
        "proxima_prime_compat",
        warm_h2_client(
            &client_runtime,
            start_proxima_prime_compat(into_handle(ConstantOk))
        )
    );

    group.finish();
}

fn bench_h2_fanin(criterion: &mut Criterion) {
    // known leak: 9 server runtimes leaked via std::mem::forget in start_* helpers,
    // bounded by the arm count (9) for this group's process lifetime.
    let mut group = criterion.benchmark_group("compat_libs_h2_fanin");
    configure_group(&mut group);
    group.throughput(Throughput::Elements(FANIN_STREAMS as u64));

    let client_runtime = build_client_runtime();

    macro_rules! fanin_arm {
        ($name:expr, $client:expr) => {{
            let client = $client;
            let mut quartet = HdrQuartet::new();
            group.bench_function($name, |bencher| {
                bencher.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for idx in 0..iters {
                        let start = Instant::now();
                        client_runtime.block_on(async {
                            let mut futs = Vec::with_capacity(FANIN_STREAMS);
                            for _ in 0..FANIN_STREAMS {
                                futs.push(one_h2_request(client.clone()));
                            }
                            join_all(futs).await;
                        });
                        let elapsed = start.elapsed();
                        let per_stream_ns =
                            (elapsed.as_nanos() / FANIN_STREAMS as u128).max(1) as u64;
                        quartet.record(idx, per_stream_ns);
                        total += elapsed;
                    }
                    quartet.finalize(iters);
                    total
                });
            });
            quartet.report($name);
        }};
    }

    fanin_arm!(
        "hyper_default_tokio",
        warm_h2_client(&client_runtime, start_hyper_default_tokio())
    );
    fanin_arm!(
        "hyper_current_thread",
        warm_h2_client(&client_runtime, start_hyper_current_thread())
    );
    fanin_arm!(
        "hyper_prime_compat",
        warm_h2_client(&client_runtime, start_hyper_prime_compat())
    );
    fanin_arm!(
        "pingora_default_tokio",
        warm_h2_client(&client_runtime, start_pingora_default_tokio())
    );
    fanin_arm!(
        "pingora_current_thread",
        warm_h2_client(&client_runtime, start_pingora_current_thread())
    );
    fanin_arm!(
        "pingora_prime_compat",
        warm_h2_client(&client_runtime, start_pingora_prime_compat())
    );
    fanin_arm!(
        "proxima_default_tokio",
        warm_h2_client(
            &client_runtime,
            start_proxima_default_tokio(into_handle(ConstantOk))
        )
    );
    fanin_arm!(
        "proxima_current_thread",
        warm_h2_client(
            &client_runtime,
            start_proxima_current_thread(into_handle(ConstantOk))
        )
    );
    fanin_arm!(
        "proxima_prime_compat",
        warm_h2_client(
            &client_runtime,
            start_proxima_prime_compat(into_handle(ConstantOk))
        )
    );

    group.finish();
}

// MULTICORE — multi-port + multi-connection workload.
//
// The single-stream and h2_fanin groups above use a single TCP
// connection. h2 streams over one connection serialize through one
// muxer, which means multi-thread tokio's extra workers are idle.
// That made the original Bench B numbers misleading (current-thread
// beat multi-thread for trivial scheduling reasons, not for any
// compat-mode mechanism).
//
// This group serves CORES ports (one per core) and the client opens
// CORES TCP connections, one per port. Each iter dispatches
// PER_CONN_STREAMS concurrent h2 streams per connection. Now:
// - multi_thread tokio can spread accept + handlers across its
//   workers
// - per_core_runtime (TokioPerCoreRuntime — pingora's actual
//   production model) gets one connection per pinned worker
// - prime_compat spawns one accept loop per prime core, each binding
//   on a separate port via the sister tokio's mio
//
// CORES connections × PER_CONN_STREAMS streams = effective parallelism
// budget. If compat actually delivers cross-core scaling, this is
// where it shows.

const PER_CONN_STREAMS: usize = 8;

// ---------------------------------------------------------------------
// hyper × multi_thread tokio — N listeners on one shared runtime.
// ---------------------------------------------------------------------

fn start_hyper_multi_thread_multicore() -> Vec<std::net::SocketAddr> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(CORES)
        .enable_all()
        .build()
        .expect("hyper multi-thread multicore runtime");
    let mut addrs = Vec::with_capacity(CORES);
    for _ in 0..CORES {
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
                tokio::spawn(async move {
                    let io = TokioIo::new(socket);
                    let _ = http2::Builder::new(TokioExecutor::new())
                        .serve_connection(io, pipe_fn(hyper_handler))
                        .await;
                });
            }
        });
        addrs.push(addr_rx.recv().expect("addr"));
    }
    std::mem::forget(runtime);
    addrs
}

// ---------------------------------------------------------------------
// hyper × TokioPerCoreRuntime (pingora's production model — N pinned
// current-thread tokio runtimes, one per core).
// ---------------------------------------------------------------------

fn start_hyper_per_core_multicore() -> Vec<std::net::SocketAddr> {
    let runtime = TokioPerCoreRuntime::new(CORES).expect("per-core runtime");
    let mut addrs = Vec::with_capacity(CORES);
    for core_index in 0..CORES {
        let (addr_tx, addr_rx) = std::sync::mpsc::channel::<std::net::SocketAddr>();
        runtime
            .spawn_factory_on_core(
                CoreId(core_index),
                Box::new(move || {
                    Box::pin(async move {
                        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
                        let addr = listener.local_addr().expect("addr");
                        addr_tx.send(addr).expect("addr send");
                        loop {
                            let (socket, _) = match listener.accept().await {
                                Ok(value) => value,
                                Err(_) => break,
                            };
                            let _ = socket.set_nodelay(true);
                            tokio::task::spawn_local(async move {
                                let io = TokioIo::new(socket);
                                let _ = http2::Builder::new(TokioExecutor::new())
                                    .serve_connection(io, pipe_fn(hyper_handler))
                                    .await;
                            });
                        }
                    })
                        as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
                }),
            )
            .expect("spawn hyper accept on per-core runtime");
        addrs.push(addr_rx.recv().expect("addr"));
    }
    std::mem::forget(runtime);
    addrs
}

// ---------------------------------------------------------------------
// hyper × prime+compat — N accept loops, one per prime core.
// ---------------------------------------------------------------------

fn start_hyper_prime_compat_multicore() -> Vec<std::net::SocketAddr> {
    let runtime = PrimeRuntime::builder()
        .cores(CORES)
        .background_inline()
        .tokio_compat()
        .build()
        .expect("hyper prime-compat multicore runtime");
    let mut addrs = Vec::with_capacity(CORES);
    for core_index in 0..CORES {
        let (addr_tx, addr_rx) = std::sync::mpsc::channel::<std::net::SocketAddr>();
        runtime
            .spawn_factory_on_core(
                CoreId(core_index),
                Box::new(move || {
                    Box::pin(async move {
                        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
                        let addr = listener.local_addr().expect("addr");
                        addr_tx.send(addr).expect("addr send");
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
                    })
                        as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
                }),
            )
            .expect("spawn hyper accept on prime+compat");
        addrs.push(addr_rx.recv().expect("addr"));
    }
    std::mem::forget(runtime);
    addrs
}

// ---------------------------------------------------------------------
// pingora × the three runtimes — same shape as hyper above.
// ---------------------------------------------------------------------

fn start_pingora_multi_thread_multicore() -> Vec<std::net::SocketAddr> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(CORES)
        .enable_all()
        .build()
        .expect("pingora multi-thread multicore runtime");
    let mut addrs = Vec::with_capacity(CORES);
    for _ in 0..CORES {
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
                tokio::spawn(async move {
                    pingora_serve_one_connection(socket).await;
                });
            }
        });
        addrs.push(addr_rx.recv().expect("addr"));
    }
    std::mem::forget(runtime);
    addrs
}

fn start_pingora_per_core_multicore() -> Vec<std::net::SocketAddr> {
    let runtime = TokioPerCoreRuntime::new(CORES).expect("per-core runtime");
    let mut addrs = Vec::with_capacity(CORES);
    for core_index in 0..CORES {
        let (addr_tx, addr_rx) = std::sync::mpsc::channel::<std::net::SocketAddr>();
        runtime
            .spawn_factory_on_core(
                CoreId(core_index),
                Box::new(move || {
                    Box::pin(async move {
                        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
                        let addr = listener.local_addr().expect("addr");
                        addr_tx.send(addr).expect("addr send");
                        loop {
                            let (socket, _) = match listener.accept().await {
                                Ok(value) => value,
                                Err(_) => break,
                            };
                            let _ = socket.set_nodelay(true);
                            tokio::task::spawn_local(async move {
                                pingora_serve_one_connection(socket).await;
                            });
                        }
                    })
                        as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
                }),
            )
            .expect("spawn pingora accept on per-core runtime");
        addrs.push(addr_rx.recv().expect("addr"));
    }
    std::mem::forget(runtime);
    addrs
}

fn start_pingora_prime_compat_multicore() -> Vec<std::net::SocketAddr> {
    let runtime = PrimeRuntime::builder()
        .cores(CORES)
        .background_inline()
        .tokio_compat()
        .build()
        .expect("pingora prime-compat multicore runtime");
    let mut addrs = Vec::with_capacity(CORES);
    for core_index in 0..CORES {
        let (addr_tx, addr_rx) = std::sync::mpsc::channel::<std::net::SocketAddr>();
        runtime
            .spawn_factory_on_core(
                CoreId(core_index),
                Box::new(move || {
                    Box::pin(async move {
                        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
                        let addr = listener.local_addr().expect("addr");
                        addr_tx.send(addr).expect("addr send");
                        loop {
                            let (socket, _) = match listener.accept().await {
                                Ok(value) => value,
                                Err(_) => break,
                            };
                            let _ = socket.set_nodelay(true);
                            tokio::spawn(async move {
                                pingora_serve_one_connection(socket).await;
                            });
                        }
                    })
                        as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
                }),
            )
            .expect("spawn pingora accept on prime+compat");
        addrs.push(addr_rx.recv().expect("addr"));
    }
    std::mem::forget(runtime);
    addrs
}

// ---------------------------------------------------------------------
// proxima × the three runtimes.
// ---------------------------------------------------------------------

fn start_proxima_multi_thread_multicore(pipe: PipeHandle) -> Vec<std::net::SocketAddr> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(CORES)
        .enable_all()
        .build()
        .expect("proxima multi-thread multicore runtime");
    let mut addrs = Vec::with_capacity(CORES);
    for _ in 0..CORES {
        let (addr_tx, addr_rx) = std::sync::mpsc::channel::<std::net::SocketAddr>();
        let pipe = pipe.clone();
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
                    let _ =
                        serve_h2_connection(socket.compat(), pipe, in_flight, quiesce, None).await;
                });
            }
        });
        addrs.push(addr_rx.recv().expect("addr"));
    }
    std::mem::forget(runtime);
    addrs
}

fn start_proxima_per_core_multicore(pipe: PipeHandle) -> Vec<std::net::SocketAddr> {
    let runtime = TokioPerCoreRuntime::new(CORES).expect("per-core runtime");
    let mut addrs = Vec::with_capacity(CORES);
    for core_index in 0..CORES {
        let (addr_tx, addr_rx) = std::sync::mpsc::channel::<std::net::SocketAddr>();
        let pipe = pipe.clone();
        runtime
            .spawn_factory_on_core(
                CoreId(core_index),
                Box::new(move || {
                    Box::pin(async move {
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
            .expect("spawn proxima accept on per-core runtime");
        addrs.push(addr_rx.recv().expect("addr"));
    }
    std::mem::forget(runtime);
    addrs
}

fn start_proxima_prime_compat_multicore(pipe: PipeHandle) -> Vec<std::net::SocketAddr> {
    let runtime = PrimeRuntime::builder()
        .cores(CORES)
        .background_inline()
        .tokio_compat()
        .build()
        .expect("proxima prime-compat multicore runtime");
    let mut addrs = Vec::with_capacity(CORES);
    for core_index in 0..CORES {
        let (addr_tx, addr_rx) = std::sync::mpsc::channel::<std::net::SocketAddr>();
        let pipe = pipe.clone();
        runtime
            .spawn_factory_on_core(
                CoreId(core_index),
                Box::new(move || {
                    Box::pin(async move {
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
            .expect("spawn proxima accept on prime+compat");
        addrs.push(addr_rx.recv().expect("addr"));
    }
    std::mem::forget(runtime);
    addrs
}

// ---------------------------------------------------------------------
// Multicore client: one warm h2 client per addr, fans PER_CONN_STREAMS
// concurrent streams across each connection. Total = CORES * PER_CONN_STREAMS.
// ---------------------------------------------------------------------

fn warm_h2_clients(
    runtime: &tokio::runtime::Runtime,
    addrs: &[std::net::SocketAddr],
) -> Vec<h2::client::SendRequest<Bytes>> {
    addrs
        .iter()
        .map(|addr| warm_h2_client(runtime, *addr))
        .collect()
}

fn bench_multicore_fanin(criterion: &mut Criterion) {
    // known leak: 9 server runtimes leaked via std::mem::forget in start_* helpers,
    // bounded by the arm count (9) for this group's process lifetime.
    let mut group = criterion.benchmark_group("compat_libs_multicore_fanin");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(15);
    group.throughput(Throughput::Elements((CORES * PER_CONN_STREAMS) as u64));

    let client_runtime = build_client_runtime();
    let total_streams = CORES * PER_CONN_STREAMS;

    macro_rules! multicore_arm {
        ($name:expr, $clients:expr) => {{
            let clients = $clients;
            let mut quartet = HdrQuartet::new();
            group.bench_function($name, |bencher| {
                bencher.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for idx in 0..iters {
                        let start = Instant::now();
                        client_runtime.block_on(run_multicore_fanin(&clients));
                        let elapsed = start.elapsed();
                        let per_stream_ns =
                            (elapsed.as_nanos() / total_streams as u128).max(1) as u64;
                        quartet.record(idx, per_stream_ns);
                        total += elapsed;
                    }
                    quartet.finalize(iters);
                    total
                });
            });
            quartet.report($name);
        }};
    }

    multicore_arm!(
        "hyper_multi_thread",
        warm_h2_clients(&client_runtime, &start_hyper_multi_thread_multicore())
    );
    multicore_arm!(
        "hyper_per_core",
        warm_h2_clients(&client_runtime, &start_hyper_per_core_multicore())
    );
    multicore_arm!(
        "hyper_prime_compat",
        warm_h2_clients(&client_runtime, &start_hyper_prime_compat_multicore())
    );
    multicore_arm!(
        "pingora_multi_thread",
        warm_h2_clients(&client_runtime, &start_pingora_multi_thread_multicore())
    );
    multicore_arm!(
        "pingora_per_core",
        warm_h2_clients(&client_runtime, &start_pingora_per_core_multicore())
    );
    multicore_arm!(
        "pingora_prime_compat",
        warm_h2_clients(&client_runtime, &start_pingora_prime_compat_multicore())
    );
    multicore_arm!(
        "proxima_multi_thread",
        warm_h2_clients(
            &client_runtime,
            &start_proxima_multi_thread_multicore(into_handle(ConstantOk))
        )
    );
    multicore_arm!(
        "proxima_per_core",
        warm_h2_clients(
            &client_runtime,
            &start_proxima_per_core_multicore(into_handle(ConstantOk))
        )
    );
    multicore_arm!(
        "proxima_prime_compat",
        warm_h2_clients(
            &client_runtime,
            &start_proxima_prime_compat_multicore(into_handle(ConstantOk))
        )
    );

    group.finish();
}

async fn run_multicore_fanin(clients: &[h2::client::SendRequest<Bytes>]) {
    let total = clients.len() * PER_CONN_STREAMS;
    let mut futs = Vec::with_capacity(total);
    for client in clients {
        for _ in 0..PER_CONN_STREAMS {
            futs.push(one_h2_request(client.clone()));
        }
    }
    join_all(futs).await;
}

// SO_REUSEPORT — single port, N accept lanes.
//
// On Linux the kernel hashes the 4-tuple (src_ip, src_port, dst_ip,
// dst_port) to distribute incoming connections across the sockets in
// the SO_REUSEPORT group — each core's socket gets its own queue, so
// there is no lock on the accept path.
//
// On macOS the distribution is round-robin (not hash-based), which
// means accept events land on the sockets in rotation regardless of
// which CPU is calling accept. The bench still runs correctly; the
// distribution characteristic just differs from production Linux.
//
// We skip `current_thread + SO_REUSEPORT` — a single thread calling
// accept on all N sockets serially gets none of the kernel-side
// distribution benefit. The useful combinations are:
//   - per-core (TokioPerCoreRuntime): each pinned worker owns one socket
//   - prime+compat: each prime worker owns one socket via the sister tokio

fn bind_reuseport_addr() -> SocketAddr {
    // bind a throwaway standard socket to get a free port, then close it.
    // the port stays reserved just long enough for the caller to re-bind
    // CORES sockets to the same (addr, port) with SO_REUSEPORT.
    let probe = std::net::TcpListener::bind("127.0.0.1:0").expect("probe bind");
    probe.local_addr().expect("probe addr")
}

// ---------------------------------------------------------------------
// hyper × SO_REUSEPORT × per-core (TokioPerCoreRuntime)
// ---------------------------------------------------------------------

fn start_hyper_reuseport_per_core() -> Vec<SocketAddr> {
    let addr = bind_reuseport_addr();
    let runtime = TokioPerCoreRuntime::new(CORES).expect("per-core runtime");
    let mut addrs = Vec::with_capacity(CORES);
    for core_index in 0..CORES {
        let (addr_tx, addr_rx) = std::sync::mpsc::channel::<SocketAddr>();
        runtime
            .spawn_factory_on_core(
                CoreId(core_index),
                Box::new(move || {
                    Box::pin(async move {
                        let listener = bind_reuseport_listener(addr).expect("reuseport bind");
                        let bound = listener.local_addr().expect("addr");
                        addr_tx.send(bound).expect("addr send");
                        loop {
                            let (socket, _) = match listener.accept().await {
                                Ok(value) => value,
                                Err(_) => break,
                            };
                            let _ = socket.set_nodelay(true);
                            tokio::task::spawn_local(async move {
                                let io = TokioIo::new(socket);
                                let _ = http2::Builder::new(TokioExecutor::new())
                                    .serve_connection(io, pipe_fn(hyper_handler))
                                    .await;
                            });
                        }
                    })
                        as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
                }),
            )
            .expect("spawn hyper reuseport accept on per-core runtime");
        addrs.push(addr_rx.recv().expect("addr"));
    }
    std::mem::forget(runtime);
    addrs
}

// ---------------------------------------------------------------------
// hyper × SO_REUSEPORT × prime+compat
// ---------------------------------------------------------------------

fn start_hyper_reuseport_prime_compat() -> Vec<SocketAddr> {
    let addr = bind_reuseport_addr();
    let runtime = PrimeRuntime::builder()
        .cores(CORES)
        .background_inline()
        .tokio_compat()
        .build()
        .expect("hyper reuseport prime-compat runtime");
    let mut addrs = Vec::with_capacity(CORES);
    for core_index in 0..CORES {
        let (addr_tx, addr_rx) = std::sync::mpsc::channel::<SocketAddr>();
        runtime
            .spawn_factory_on_core(
                CoreId(core_index),
                Box::new(move || {
                    Box::pin(async move {
                        let listener = bind_reuseport_listener(addr).expect("reuseport bind");
                        let bound = listener.local_addr().expect("addr");
                        addr_tx.send(bound).expect("addr send");
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
                    })
                        as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
                }),
            )
            .expect("spawn hyper reuseport accept on prime+compat");
        addrs.push(addr_rx.recv().expect("addr"));
    }
    std::mem::forget(runtime);
    addrs
}

// ---------------------------------------------------------------------
// pingora × SO_REUSEPORT × per-core
// ---------------------------------------------------------------------

fn start_pingora_reuseport_per_core() -> Vec<SocketAddr> {
    let addr = bind_reuseport_addr();
    let runtime = TokioPerCoreRuntime::new(CORES).expect("per-core runtime");
    let mut addrs = Vec::with_capacity(CORES);
    for core_index in 0..CORES {
        let (addr_tx, addr_rx) = std::sync::mpsc::channel::<SocketAddr>();
        runtime
            .spawn_factory_on_core(
                CoreId(core_index),
                Box::new(move || {
                    Box::pin(async move {
                        let listener = bind_reuseport_listener(addr).expect("reuseport bind");
                        let bound = listener.local_addr().expect("addr");
                        addr_tx.send(bound).expect("addr send");
                        loop {
                            let (socket, _) = match listener.accept().await {
                                Ok(value) => value,
                                Err(_) => break,
                            };
                            let _ = socket.set_nodelay(true);
                            tokio::task::spawn_local(async move {
                                pingora_serve_one_connection(socket).await;
                            });
                        }
                    })
                        as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
                }),
            )
            .expect("spawn pingora reuseport accept on per-core runtime");
        addrs.push(addr_rx.recv().expect("addr"));
    }
    std::mem::forget(runtime);
    addrs
}

// ---------------------------------------------------------------------
// pingora × SO_REUSEPORT × prime+compat
// ---------------------------------------------------------------------

fn start_pingora_reuseport_prime_compat() -> Vec<SocketAddr> {
    let addr = bind_reuseport_addr();
    let runtime = PrimeRuntime::builder()
        .cores(CORES)
        .background_inline()
        .tokio_compat()
        .build()
        .expect("pingora reuseport prime-compat runtime");
    let mut addrs = Vec::with_capacity(CORES);
    for core_index in 0..CORES {
        let (addr_tx, addr_rx) = std::sync::mpsc::channel::<SocketAddr>();
        runtime
            .spawn_factory_on_core(
                CoreId(core_index),
                Box::new(move || {
                    Box::pin(async move {
                        let listener = bind_reuseport_listener(addr).expect("reuseport bind");
                        let bound = listener.local_addr().expect("addr");
                        addr_tx.send(bound).expect("addr send");
                        loop {
                            let (socket, _) = match listener.accept().await {
                                Ok(value) => value,
                                Err(_) => break,
                            };
                            let _ = socket.set_nodelay(true);
                            tokio::spawn(async move {
                                pingora_serve_one_connection(socket).await;
                            });
                        }
                    })
                        as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
                }),
            )
            .expect("spawn pingora reuseport accept on prime+compat");
        addrs.push(addr_rx.recv().expect("addr"));
    }
    std::mem::forget(runtime);
    addrs
}

// ---------------------------------------------------------------------
// proxima × SO_REUSEPORT × per-core
// ---------------------------------------------------------------------

fn start_proxima_reuseport_per_core(pipe: PipeHandle) -> Vec<SocketAddr> {
    let addr = bind_reuseport_addr();
    let runtime = TokioPerCoreRuntime::new(CORES).expect("per-core runtime");
    let mut addrs = Vec::with_capacity(CORES);
    for core_index in 0..CORES {
        let (addr_tx, addr_rx) = std::sync::mpsc::channel::<SocketAddr>();
        let pipe = pipe.clone();
        runtime
            .spawn_factory_on_core(
                CoreId(core_index),
                Box::new(move || {
                    Box::pin(async move {
                        let listener = bind_reuseport_listener(addr).expect("reuseport bind");
                        let bound = listener.local_addr().expect("addr");
                        addr_tx.send(bound).expect("addr send");
                        loop {
                            let (socket, _) = match listener.accept().await {
                                Ok(value) => value,
                                Err(_) => break,
                            };
                            let _ = socket.set_nodelay(true);
                            let pipe = pipe.clone();
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
            .expect("spawn proxima reuseport accept on per-core runtime");
        addrs.push(addr_rx.recv().expect("addr"));
    }
    std::mem::forget(runtime);
    addrs
}

// ---------------------------------------------------------------------
// proxima × SO_REUSEPORT × prime+compat
// ---------------------------------------------------------------------

fn start_proxima_reuseport_prime_compat(pipe: PipeHandle) -> Vec<SocketAddr> {
    let addr = bind_reuseport_addr();
    let runtime = PrimeRuntime::builder()
        .cores(CORES)
        .background_inline()
        .tokio_compat()
        .build()
        .expect("proxima reuseport prime-compat runtime");
    let mut addrs = Vec::with_capacity(CORES);
    for core_index in 0..CORES {
        let (addr_tx, addr_rx) = std::sync::mpsc::channel::<SocketAddr>();
        let pipe = pipe.clone();
        runtime
            .spawn_factory_on_core(
                CoreId(core_index),
                Box::new(move || {
                    let pipe = pipe;
                    Box::pin(async move {
                        let listener = bind_reuseport_listener(addr).expect("reuseport bind");
                        let bound = listener.local_addr().expect("addr");
                        addr_tx.send(bound).expect("addr send");
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
            .expect("spawn proxima reuseport accept on prime+compat");
        addrs.push(addr_rx.recv().expect("addr"));
    }
    std::mem::forget(runtime);
    addrs
}

// ---------------------------------------------------------------------
// SO_REUSEPORT bench group.
//
// Client side: all CORES connections go to the SAME port. The kernel's
// SO_REUSEPORT distribution decides which accept lane gets each SYN.
// PER_CONN_STREAMS concurrent h2 streams per connection — same budget
// as the multi-port group so numbers are comparable.
// ---------------------------------------------------------------------

fn bench_multicore_fanin_reuseport(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("compat_libs_multicore_fanin_reuseport");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(15);
    group.throughput(Throughput::Elements((CORES * PER_CONN_STREAMS) as u64));

    let client_runtime = build_client_runtime();
    let total_streams = CORES * PER_CONN_STREAMS;

    macro_rules! reuseport_arm {
        ($name:expr, $addrs:expr) => {{
            let addrs = $addrs;
            // all connections go to the same port (SO_REUSEPORT group
            // shares one (addr, port); the first entry is representative)
            let target = addrs[0];
            let mut clients = Vec::with_capacity(CORES);
            for _ in 0..CORES {
                clients.push(warm_h2_client(&client_runtime, target));
            }
            let mut quartet = HdrQuartet::new();
            group.bench_function($name, |bencher| {
                bencher.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for idx in 0..iters {
                        let start = Instant::now();
                        client_runtime.block_on(run_multicore_fanin(&clients));
                        let elapsed = start.elapsed();
                        let per_stream_ns =
                            (elapsed.as_nanos() / total_streams as u128).max(1) as u64;
                        quartet.record(idx, per_stream_ns);
                        total += elapsed;
                    }
                    quartet.finalize(iters);
                    total
                });
            });
            quartet.report($name);
        }};
    }

    reuseport_arm!("hyper_reuseport_per_core", start_hyper_reuseport_per_core());
    reuseport_arm!(
        "hyper_reuseport_prime_compat",
        start_hyper_reuseport_prime_compat()
    );
    reuseport_arm!(
        "pingora_reuseport_per_core",
        start_pingora_reuseport_per_core()
    );
    reuseport_arm!(
        "pingora_reuseport_prime_compat",
        start_pingora_reuseport_prime_compat()
    );
    reuseport_arm!(
        "proxima_reuseport_per_core",
        start_proxima_reuseport_per_core(into_handle(ConstantOk))
    );
    reuseport_arm!(
        "proxima_reuseport_prime_compat",
        start_proxima_reuseport_prime_compat(into_handle(ConstantOk))
    );

    group.finish();
}

criterion_group!(
    benches,
    bench_single_stream,
    bench_h2_fanin,
    bench_multicore_fanin,
    bench_multicore_fanin_reuseport
);
criterion_main!(benches);

// keeps Runtime import alive when we are not directly constructing a
// `dyn Runtime` here (PrimeRuntime is used via its concrete type and
// builder); Runtime is needed for the trait-bound spawn_factory_on_core
// call below.
#[allow(dead_code)]
fn _keep_runtime_trait_alive() {
    let _: Option<Arc<dyn Runtime>> = None;
}
