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
    feature = "runtime-tokio",
    all(
        feature = "runtime-prime-executor",
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-reactor",
        feature = "runtime-prime-bgpool"
    )
))]

//! H/2 load 5-way comparison: `pingora` vs `tokio` (hyper) vs three flavors
//! of proxima native HTTP/2. Cross-platform (macOS + Linux).
//!
//! Arms:
//!   1. **pingora** — pingora's h2 server on default tokio multi-thread.
//!   2. **tokio_hyper** — hyper's h2 server on default tokio multi-thread.
//!      This is what people usually mean by "the tokio h2 baseline."
//!   3. **proxima_on_tokio** — proxima native h2 server on default tokio
//!      multi-thread (work-stealing, N worker threads).
//!   4. **proxima_on_flume** — proxima native h2 server on `TokioPerCoreRuntime`
//!      (pinned tokio current-thread per CPU; cross-core dispatch via flume).
//!   5. **proxima_on_prime** — proxima native h2 server on `PrimeRuntime`
//!      (our from-scratch per-core runtime). **NOT YET — requires futures::io
//!      `AsyncRead`/`AsyncWrite` TcpListener + TcpStream built on
//!      proxima::runtime::prime::os::reactor::Reactor.** See the stub at the
//!      bottom of this file + TODO docs in src/runtime/prime/os/.
//!
//! All servers run on isolated runtimes; the criterion client driver runs
//! on its own dedicated tokio multi-thread runtime so we measure pure
//! server-side cost.
//!
//! ## Running
//!
//! ```bash
//! # macOS / Linux:
//! cargo bench --features http2,runtime-tokio --bench h2_runtime_swap
//!
//! # filter to just the 5-way:
//! cargo bench --features http2,runtime-tokio --bench h2_runtime_swap -- h2_load_5way
//! ```

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use http_body_util::Full;
use hyper::server::conn::http2;
// hyper's tower-style adapter is still named service_fn upstream — the
// proxima-internal `Service → Pipe` rename doesn't reach hyper.
use hyper::service::service_fn as pipe_fn;
use hyper_util::rt::{TokioExecutor, TokioIo};
use proxima::error::ProximaError;
use proxima::h2::serve_h2_connection;
use proxima::listeners::http::QuiesceResponse;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::request::{Request, Response};
use proxima::runtime::prime::os::net::TcpListener as ProximaTcpListener;
use proxima::runtime::{CoreId, PrimeRuntime, Runtime, TokioPerCoreRuntime};
use proxima_primitives::pipe::SendPipe;
use tokio::net::{TcpListener, TcpStream};
use tokio_util::compat::TokioAsyncReadCompatExt;

const RESPONSE_BODY: &[u8] = b"ok";

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
        .header("content-type", "text/plain")
        .body(Full::new(Bytes::from_static(RESPONSE_BODY)))
        .expect("response"))
}

fn build_client_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("client runtime")
}

/// Start proxima native on **default tokio multi-thread** in its own
/// dedicated runtime. Returns the bound address.
fn start_native_default_tokio() -> std::net::SocketAddr {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("default tokio server runtime");
    let (addr, listener) = runtime.block_on(async {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        (addr, listener)
    });
    let dispatch: PipeHandle = into_handle(ConstantOk);
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
    // Leak the runtime: it must outlive the bench. Server tasks run on
    // its threads. Process exit cleans up.
    std::mem::forget(runtime);
    addr
}

/// Start proxima native on the **per-core runtime** (pinned tokio
/// current-thread per CPU). Returns the bound address.
fn start_native_per_core() -> std::net::SocketAddr {
    let runtime = TokioPerCoreRuntime::new(2).expect("per-core runtime");
    // Bind on the per-core runtime's core 0 — TcpListener::bind needs
    // a tokio context, which the per-core worker provides. We use a
    // setup channel to ferry the addr back out.
    let (addr_tx, addr_rx) = std::sync::mpsc::channel::<std::net::SocketAddr>();
    let dispatch: PipeHandle = into_handle(ConstantOk);
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
                        let dispatch = dispatch.clone();
                        // Spawn each connection on the same core — no
                        // cross-core hops for the connection's lifetime.
                        tokio::task::spawn_local(async move {
                            let in_flight = Arc::new(AtomicU64::new(0));
                            let quiesce = Arc::new(QuiesceResponse {
                                status: 503,
                                retry_after: "1".into(),
                            });
                            let _ = serve_h2_connection(
                                socket.compat(),
                                dispatch,
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
        .expect("bench setup: spawn listener factory on fresh runtime");
    let addr = addr_rx.recv().expect("addr from per-core worker");
    std::mem::forget(runtime);
    addr
}

fn start_hyper_default_tokio() -> std::net::SocketAddr {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("hyper server runtime");
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

/// Start proxima native h2 on `PrimeRuntime` — our from-scratch per-core
/// runtime with native futures::io TCP I/O. No tokio context. Returns the
/// bound address.
fn start_native_proxima_runtime() -> std::net::SocketAddr {
    // 1 core is sufficient — the bench's client opens ONE persistent h2
    // connection. listener + serve_h2_connection live on the same worker.
    // 2 cores adds an extra idle worker thread with no benefit.
    let runtime = PrimeRuntime::new(1).expect("proxima runtime");
    let (addr_tx, addr_rx) = std::sync::mpsc::channel::<std::net::SocketAddr>();
    let dispatch: PipeHandle = into_handle(ConstantOk);
    runtime
        .spawn_factory_on_core(
            CoreId(0),
            Box::new(move || {
                let dispatch = dispatch;
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
                        // spawn each connection on the same core. proxima
                        // executor will poll this via its !Send slab.
                        proxima::runtime::prime::os::core_shard::spawn_on_current_core(Box::pin(
                            async move {
                                let in_flight = Arc::new(AtomicU64::new(0));
                                let quiesce = Arc::new(QuiesceResponse {
                                    status: 503,
                                    retry_after: "1".into(),
                                });
                                // socket is futures::io natively — no compat
                                // shim needed (vs the tokio_util::compat used
                                // by the other arms).
                                let _ =
                                    serve_h2_connection(socket, dispatch, in_flight, quiesce, None)
                                        .await;
                            },
                        ));
                    }
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }),
        )
        .expect("bench setup: spawn listener factory on fresh runtime");
    let addr = addr_rx.recv().expect("addr from proxima worker");
    std::mem::forget(runtime);
    addr
}

fn start_pingora_default_tokio() -> std::net::SocketAddr {
    use pingora_core::protocols::Digest;
    use pingora_core::protocols::http::v2::server::{HttpSession, handshake as h2c_handshake};
    use pingora_http::ResponseHeader;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("pingora server runtime");
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
                let l4_stream: pingora_core::protocols::l4::stream::Stream = socket.into();
                let pingora_stream: pingora_core::protocols::Stream = Box::new(l4_stream);
                let mut h2_conn = match h2c_handshake(pingora_stream, None).await {
                    Ok(conn) => conn,
                    Err(_) => return,
                };
                let digest = Arc::new(Digest::default());
                loop {
                    let session =
                        match HttpSession::from_h2_conn(&mut h2_conn, digest.clone()).await {
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
            });
        }
    });
    std::mem::forget(runtime);
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

async fn one_request(mut h2_client: h2::client::SendRequest<Bytes>) {
    let request = http::Request::builder()
        .method("GET")
        .uri("http://localhost/")
        .body(())
        .expect("request");
    let (response_future, _) = h2_client.send_request(request, true).expect("send_request");
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

fn runtime_swap_proxima(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_runtime_swap_proxima_native");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1));

    let client_runtime = build_client_runtime();

    let default_addr = start_native_default_tokio();
    let default_client = warm_client(&client_runtime, default_addr);
    group.bench_function("default_tokio", |bencher| {
        bencher.to_async(&client_runtime).iter(|| {
            let client = default_client.clone();
            one_request(client)
        });
    });

    let per_core_addr = start_native_per_core();
    let per_core_client = warm_client(&client_runtime, per_core_addr);
    group.bench_function("per_core_runtime", |bencher| {
        bencher.to_async(&client_runtime).iter(|| {
            let client = per_core_client.clone();
            one_request(client)
        });
    });

    group.finish();
}

fn mic_drop_vs_hyper_pingora(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_per_core_vs_hyper_pingora");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1));

    let client_runtime = build_client_runtime();

    let per_core_addr = start_native_per_core();
    let per_core_client = warm_client(&client_runtime, per_core_addr);
    group.bench_function("proxima_native_per_core", |bencher| {
        bencher.to_async(&client_runtime).iter(|| {
            let client = per_core_client.clone();
            one_request(client)
        });
    });

    let hyper_addr = start_hyper_default_tokio();
    let hyper_client = warm_client(&client_runtime, hyper_addr);
    group.bench_function("hyper_default_tokio", |bencher| {
        bencher.to_async(&client_runtime).iter(|| {
            let client = hyper_client.clone();
            one_request(client)
        });
    });

    let pingora_addr = start_pingora_default_tokio();
    let pingora_client = warm_client(&client_runtime, pingora_addr);
    group.bench_function("pingora_default_tokio", |bencher| {
        bencher.to_async(&client_runtime).iter(|| {
            let client = pingora_client.clone();
            one_request(client)
        });
    });

    group.finish();
}

/// 5-way head-to-head: pingora / tokio_hyper / proxima_on_tokio /
/// proxima_on_flume / proxima_on_prime.
///
/// Cross-platform. macOS uses kqueue I/O via tokio's mio reactor;
/// Linux uses epoll. Both via the standard tokio runtime path. No
/// special platform handling required.
fn h2_load_5way(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_load_5way");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1));

    let client_runtime = build_client_runtime();

    // arm 1: pingora native h2
    let pingora_addr = start_pingora_default_tokio();
    let pingora_client = warm_client(&client_runtime, pingora_addr);
    group.bench_function("pingora", |bencher| {
        bencher.to_async(&client_runtime).iter(|| {
            let client = pingora_client.clone();
            one_request(client)
        });
    });

    // arm 2: tokio_hyper — hyper h2 on default tokio multi-thread
    let hyper_addr = start_hyper_default_tokio();
    let hyper_client = warm_client(&client_runtime, hyper_addr);
    group.bench_function("tokio_hyper", |bencher| {
        bencher.to_async(&client_runtime).iter(|| {
            let client = hyper_client.clone();
            one_request(client)
        });
    });

    // arm 3: proxima_on_tokio — proxima native h2 on default tokio multi-thread
    let default_addr = start_native_default_tokio();
    let default_client = warm_client(&client_runtime, default_addr);
    group.bench_function("proxima_on_tokio", |bencher| {
        bencher.to_async(&client_runtime).iter(|| {
            let client = default_client.clone();
            one_request(client)
        });
    });

    // arm 4: proxima_on_flume — proxima native h2 on TokioPerCoreRuntime
    //   (per-core pinned tokio current-thread; cross-core dispatch via flume).
    let per_core_addr = start_native_per_core();
    let per_core_client = warm_client(&client_runtime, per_core_addr);
    group.bench_function("proxima_on_flume", |bencher| {
        bencher.to_async(&client_runtime).iter(|| {
            let client = per_core_client.clone();
            one_request(client)
        });
    });

    // arm 5: proxima_on_prime — proxima native h2 on PrimeRuntime.
    //   uses the native proxima::runtime::prime::os::net::TcpListener
    //   which implements futures::io::AsyncRead/AsyncWrite on top of
    //   PrimeRuntime's Reactor. NO tokio runtime context — pure proxima.
    let proxima_addr = start_native_proxima_runtime();
    let proxima_client = warm_client(&client_runtime, proxima_addr);
    group.bench_function("proxima_on_prime", |bencher| {
        bencher.to_async(&client_runtime).iter(|| {
            let client = proxima_client.clone();
            one_request(client)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    h2_load_5way,
    runtime_swap_proxima,
    mic_drop_vs_hyper_pingora
);
criterion_main!(benches);
