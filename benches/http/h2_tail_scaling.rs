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

//! Full tail-latency sweep: **four** server impls × **three**
//! concurrency levels. Closes the loop on the substrate composition
//! claim — does the protocol-stack win compose with the runtime win
//! as concurrency rises?
//!
//! Impls:
//! - proxima native + default tokio (multi-thread, work-stealing)
//! - proxima native + per-core runtime (pinned tokio current-thread)
//! - hyper            + default tokio
//! - pingora          + default tokio
//!
//! Concurrency: 1, 10, 100 streams multiplexed on one warm h2 client.
//!
//! Each criterion iter fires one round of `concurrency` concurrent
//! requests on a pre-warmed connection. Criterion's mean + CI is the
//! regression signal; per-iter median latency is what we capture.

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use http_body_util::Full;
use hyper::server::conn::http2;
use hyper::service::service_fn as pipe_fn;
use hyper_util::rt::{TokioExecutor, TokioIo};
use proxima::error::ProximaError;
use proxima::h2::serve_h2_connection;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::request::{Request, Response};
use proxima::runtime::{CoreId, Runtime, TokioPerCoreRuntime};
use proxima_primitives::pipe::SendPipe;
use tokio::net::{TcpListener, TcpStream};
use tokio_util::compat::TokioAsyncReadCompatExt;

const RESPONSE_BODY: &[u8] = b"ok";
const WARMUP_REQUESTS: usize = 50;

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
        .worker_threads(2)
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
    std::mem::forget(runtime);
    addr
}

fn start_native_per_core() -> std::net::SocketAddr {
    let runtime = TokioPerCoreRuntime::new(2).expect("per-core runtime");
    let (addr_tx, addr_rx) = std::sync::mpsc::channel::<std::net::SocketAddr>();
    let dispatch: PipeHandle = into_handle(ConstantOk);
    let _ = runtime.spawn_factory_on_core(
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
                    tokio::task::spawn_local(async move {
                        let _ = serve_h2_connection(
                            socket.compat(),
                            dispatch,
                            proxima_listen::admission::ConnAdmission::unbounded(),
                            None,
                        )
                        .await;
                    });
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
        .worker_threads(2)
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
        .worker_threads(2)
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
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("client runtime")
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

async fn run_one_iter(client: h2::client::SendRequest<Bytes>, concurrency: usize) {
    let mut tasks = Vec::with_capacity(concurrency);
    for _ in 0..concurrency {
        let mut client = client.clone();
        tasks.push(tokio::spawn(async move {
            one_request(&mut client).await;
        }));
    }
    for task in tasks {
        task.await.expect("join task");
    }
}

fn h2_tail_scaling(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_tail_scaling");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1));

    let client_runtime = build_client_runtime();

    let native_default_addr = start_native_default_tokio();
    let native_per_core_addr = start_native_per_core();
    let hyper_addr = start_hyper();
    let pingora_addr = start_pingora();

    std::thread::sleep(Duration::from_millis(200));

    let native_default_client = client_runtime.block_on(async {
        let mut client = warm_client(native_default_addr).await;
        for _ in 0..WARMUP_REQUESTS {
            one_request(&mut client).await;
        }
        client
    });

    let native_per_core_client = client_runtime.block_on(async {
        let mut client = warm_client(native_per_core_addr).await;
        for _ in 0..WARMUP_REQUESTS {
            one_request(&mut client).await;
        }
        client
    });

    let hyper_client = client_runtime.block_on(async {
        let mut client = warm_client(hyper_addr).await;
        for _ in 0..WARMUP_REQUESTS {
            one_request(&mut client).await;
        }
        client
    });

    let pingora_client = client_runtime.block_on(async {
        let mut client = warm_client(pingora_addr).await;
        for _ in 0..WARMUP_REQUESTS {
            one_request(&mut client).await;
        }
        client
    });

    for concurrency in [1_usize, 10, 100] {
        group.bench_function(
            format!("proxima_native_default_tokio/streams={concurrency}"),
            |bencher| {
                bencher
                    .to_async(&client_runtime)
                    .iter(|| run_one_iter(native_default_client.clone(), concurrency));
            },
        );

        group.bench_function(
            format!("proxima_native_per_core/streams={concurrency}"),
            |bencher| {
                bencher
                    .to_async(&client_runtime)
                    .iter(|| run_one_iter(native_per_core_client.clone(), concurrency));
            },
        );

        group.bench_function(
            format!("hyper_default_tokio/streams={concurrency}"),
            |bencher| {
                bencher
                    .to_async(&client_runtime)
                    .iter(|| run_one_iter(hyper_client.clone(), concurrency));
            },
        );

        group.bench_function(
            format!("pingora_default_tokio/streams={concurrency}"),
            |bencher| {
                bencher
                    .to_async(&client_runtime)
                    .iter(|| run_one_iter(pingora_client.clone(), concurrency));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, h2_tail_scaling);
criterion_main!(benches);
