#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
#![cfg(all(feature = "http2", feature = "runtime-tokio"))]

//! Multi-connection tail-latency sweep. Where single-connection benches
//! pin all work to one server core (so a per-core runtime can't shine),
//! this bench opens **N independent TCP connections** to each server.
//! Each connection fans out one request per connection per criterion iter.
//!
//! The per-core proxima variant round-robins accepted connections
//! across cores via [`Runtime::spawn_factory_on_core`] — so different
//! connections actually land on different CPUs. Hyper and pingora
//! servers run on default tokio multi-thread; tokio's work-stealing
//! schedules connections wherever a worker is idle.
//!
//! Concurrency dimension here = **number of TCP connections**, not
//! streams. Each connection runs one request at a time; with 4
//! connections you get 4 concurrent in-flight requests across 4
//! different streams on 4 different sockets.
//!
//! Per-request latencies are recorded into an HDR histogram so that
//! p50/p99/p999 are available alongside criterion's mean. Criterion
//! receives the p50 as the timing measurement; p99 and p999 are
//! emitted as separate bench_function entries per arm per conn count.

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use bytes::Bytes;
use criterion::{Criterion, criterion_group, criterion_main};
use futures::future::join_all;
use hdrhistogram::Histogram;

#[path = "../common/hdr_phased.rs"]
mod hdr_phased;
use hdr_phased::HdrQuartet;
use http_body_util::Full;
use hyper::server::conn::http2;
use hyper::service::service_fn as pipe_fn;
use hyper_util::rt::{TokioExecutor, TokioIo};
use proxima::error::ProximaError;
use proxima::h2::serve_h2_connection;
use proxima::listeners::http::QuiesceResponse;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::request::{Request, Response};
use proxima::runtime::{CoreId, Runtime, TokioPerCoreRuntime};
use proxima_primitives::pipe::SendPipe;
use tokio::net::{TcpListener, TcpStream};
use tokio_util::compat::TokioAsyncReadCompatExt;

const RESPONSE_BODY: &[u8] = b"ok";
const SERVER_CORES: usize = 4;
const WARMUP_REQUESTS: usize = 20;
const SAMPLE_WINDOW: Duration = Duration::from_secs(3);

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


async fn hyper_handler(
    _request: hyper::Request<hyper::body::Incoming>,
) -> Result<hyper::Response<Full<Bytes>>, Infallible> {
    Ok(hyper::Response::builder()
        .status(200)
        .header("content-type", "text/plain")
        .body(Full::new(Bytes::from_static(RESPONSE_BODY)))
        .expect("response"))
}

fn start_native_default_tokio() -> std::net::SocketAddr {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(SERVER_CORES)
        .enable_all()
        .build()
        .expect("server runtime");
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
    std::mem::forget(runtime);
    addr
}

fn start_native_per_core_round_robin() -> std::net::SocketAddr {
    let runtime = Arc::new(TokioPerCoreRuntime::new(SERVER_CORES).expect("per-core runtime"));
    let (addr_tx, addr_rx) = std::sync::mpsc::channel::<std::net::SocketAddr>();
    let dispatch: PipeHandle = into_handle(ConstantOk);
    let runtime_for_accept = Arc::clone(&runtime);
    let _ = runtime.spawn_factory_on_core(
        CoreId(0),
        Box::new(move || {
            Box::pin(async move {
                let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
                let addr = listener.local_addr().expect("addr");
                addr_tx.send(addr).expect("addr send");
                let next_core = Arc::new(AtomicUsize::new(0));
                loop {
                    let (socket, _) = match listener.accept().await {
                        Ok(value) => value,
                        Err(_) => break,
                    };
                    let _ = socket.set_nodelay(true);
                    let dispatch = dispatch.clone();
                    let core_index =
                        next_core.fetch_add(1, Ordering::Relaxed) % runtime_for_accept.num_cores();
                    let _ = runtime_for_accept.spawn_factory_on_core(
                        CoreId(core_index),
                        Box::new(move || {
                            Box::pin(async move {
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
                            })
                                as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
                        }),
                    );
                }
            }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
        }),
    );
    let addr = addr_rx.recv().expect("addr from per-core worker");
    std::mem::forget(runtime);
    addr
}

fn start_hyper() -> std::net::SocketAddr {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(SERVER_CORES)
        .enable_all()
        .build()
        .expect("hyper runtime");
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

fn start_pingora() -> std::net::SocketAddr {
    use pingora_core::protocols::Digest;
    use pingora_core::protocols::http::v2::server::{HttpSession, handshake as h2c_handshake};
    use pingora_http::ResponseHeader;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(SERVER_CORES)
        .enable_all()
        .build()
        .expect("pingora runtime");
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

fn build_client_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(SERVER_CORES)
        .enable_all()
        .build()
        .expect("client runtime")
}

async fn make_warm_clients(
    addr: std::net::SocketAddr,
    connections: usize,
) -> Vec<h2::client::SendRequest<Bytes>> {
    let mut clients = Vec::with_capacity(connections);
    for _ in 0..connections {
        let socket = TcpStream::connect(addr).await.expect("connect");
        let _ = socket.set_nodelay(true);
        let (mut h2_client, h2_conn) = h2::client::handshake(socket).await.expect("handshake");
        tokio::spawn(async move {
            let _ = h2_conn.await;
        });
        for _ in 0..WARMUP_REQUESTS {
            one_request(&mut h2_client).await;
        }
        clients.push(h2_client);
    }
    clients
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

async fn timed_iter(clients: Vec<h2::client::SendRequest<Bytes>>, iters: u64) -> Histogram<u64> {
    let mut hist = fresh_histogram();
    for _ in 0..iters {
        let request_tasks = clients.iter().cloned().map(|mut client| async move {
            let request = http::Request::builder()
                .method("GET")
                .uri("http://localhost/")
                .body(())
                .expect("request");
            let started = Instant::now();
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
            started.elapsed().as_nanos() as u64
        });
        let per_conn_ns = join_all(request_tasks).await;
        for measurement in per_conn_ns {
            let _ = hist.record(measurement.max(1));
        }
    }
    hist
}

fn bench_arm(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    client_runtime: &tokio::runtime::Runtime,
    clients: Vec<h2::client::SendRequest<Bytes>>,
    arm: &str,
    connections: usize,
) {
    let label_p50 = format!("{arm}/conn={connections}/p50");
    let label_p99 = format!("{arm}/conn={connections}/p99");
    let label_p999 = format!("{arm}/conn={connections}/p999");
    let phased_label = format!("{arm}/conn={connections}");

    group.bench_function(&label_p50, |bencher| {
        let clients = clients.clone();
        let arm_label = phased_label.clone();
        let mut quartet = HdrQuartet::new();
        bencher.to_async(client_runtime).iter_custom(|iters| {
            let clients = clients.clone();
            let arm_label = arm_label.clone();
            async move {
                let hist = timed_iter(clients, iters).await;
                // record each measurement into the quartet using the histogram
                // samples are not available individually here; approximate by
                // recording the p50 once per iter as a representative sample.
                // per-sample phased decomposition is in timed_iter_phased below.
                let _ = &arm_label;
                Duration::from_nanos(hist.value_at_quantile(0.5).max(1))
            }
        });
        // report phased decomposition after criterion finishes sampling this arm
        let clients_for_phased = clients.clone();
        let samples: Vec<u64> =
            client_runtime.block_on(async { timed_iter_samples(clients_for_phased, 100).await });
        for (idx, latency_ns) in samples.iter().enumerate() {
            quartet.record(idx as u64, *latency_ns);
        }
        quartet.finalize(samples.len() as u64);
        quartet.report(&phased_label);
    });

    group.bench_function(&label_p99, |bencher| {
        let clients = clients.clone();
        bencher.to_async(client_runtime).iter_custom(|iters| {
            let clients = clients.clone();
            async move {
                let hist = timed_iter(clients, iters).await;
                Duration::from_nanos(hist.value_at_quantile(0.99).max(1))
            }
        });
    });

    group.bench_function(&label_p999, |bencher| {
        let clients = clients.clone();
        bencher.to_async(client_runtime).iter_custom(|iters| {
            let clients = clients.clone();
            async move {
                let hist = timed_iter(clients, iters).await;
                Duration::from_nanos(hist.value_at_quantile(0.999).max(1))
            }
        });
    });
}

async fn timed_iter_samples(clients: Vec<h2::client::SendRequest<Bytes>>, iters: u64) -> Vec<u64> {
    let mut samples = Vec::with_capacity((iters * clients.len() as u64) as usize);
    for _ in 0..iters {
        let request_tasks = clients.iter().cloned().map(|mut client| async move {
            let request = http::Request::builder()
                .method("GET")
                .uri("http://localhost/")
                .body(())
                .expect("request");
            let started = Instant::now();
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
            started.elapsed().as_nanos() as u64
        });
        let per_conn_ns = join_all(request_tasks).await;
        samples.extend(per_conn_ns);
    }
    samples
}

fn h2_tail_multi_conn(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_tail_multi_conn");
    group.measurement_time(SAMPLE_WINDOW);

    let client_runtime = build_client_runtime();

    let native_default_addr = start_native_default_tokio();
    let native_per_core_addr = start_native_per_core_round_robin();
    let hyper_addr = start_hyper();
    let pingora_addr = start_pingora();

    std::thread::sleep(Duration::from_millis(200));

    for connections in [1_usize, 4, 16, 64] {
        let native_default_clients =
            client_runtime.block_on(make_warm_clients(native_default_addr, connections));
        bench_arm(
            &mut group,
            &client_runtime,
            native_default_clients,
            "proxima_native_default_tokio",
            connections,
        );

        let native_per_core_clients =
            client_runtime.block_on(make_warm_clients(native_per_core_addr, connections));
        bench_arm(
            &mut group,
            &client_runtime,
            native_per_core_clients,
            "proxima_native_per_core",
            connections,
        );

        let hyper_clients = client_runtime.block_on(make_warm_clients(hyper_addr, connections));
        bench_arm(
            &mut group,
            &client_runtime,
            hyper_clients,
            "hyper_default_tokio",
            connections,
        );

        let pingora_clients = client_runtime.block_on(make_warm_clients(pingora_addr, connections));
        bench_arm(
            &mut group,
            &client_runtime,
            pingora_clients,
            "pingora_default_tokio",
            connections,
        );
    }

    group.finish();
}

criterion_group!(benches, h2_tail_multi_conn);
criterion_main!(benches);
