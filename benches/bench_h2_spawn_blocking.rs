//! End-to-end h2 request bench where the handler offloads CPU work to
//! the runtime's background-blocking pool. Measures the *integration*
//! cost of `spawn_background_blocking`: the await-side park + cross-
//! thread join + result delivery, not just the pool's raw throughput.
//!
//! Workload per request:
//!   - h2 handler calls `runtime.spawn_background_blocking(|| blake3::hash(4KB))`.
//!   - Awaits the handle.
//!   - Returns the hash as the response body.
//!
//! Compared backends (both serve the same `serve_h2_connection`):
//!   - `tokio_per_core`: TokioPerCoreRuntime; spawn_background_blocking
//!     delegates to `tokio::task::spawn_blocking`.
//!   - `prime`: PrimeRuntime; uses its own BackgroundPool (when one is
//!     attached) or falls back to a one-shot std::thread per call (when
//!     not — tested here intentionally to surface the cost).
//!
//! required-features: runtime-tokio, runtime-prime-full, http2, tcp, http1.

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

use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima::error::ProximaError;
use proxima::h2::serve_h2_connection;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::request::{Request, Response};
use proxima::runtime::prime::os::background::ProximaBackgroundPool;
use proxima::runtime::prime::os::net::TcpListener as ProximaTcpListener;
use proxima::runtime::{BackgroundPool, CoreId, PrimeRuntime, Runtime, TokioPerCoreRuntime};
use proxima_primitives::pipe::SendPipe;
use tokio::net::TcpListener;
use tokio_util::compat::TokioAsyncReadCompatExt;

const PAYLOAD_LEN: usize = 4096;

/// Stand-in for "CPU work that benefits from offloading to a blocking
/// pool" — sum bytes with multiply+xor, repeated enough times to take
/// ~5μs on M1. Real workloads would be e.g. crypto, image decoding,
/// regex matching; we just need consistent CPU saturation that the
/// optimizer can't elide.
#[inline(never)]
fn cpu_work(payload: &[u8]) -> u64 {
    let mut acc: u64 = 0;
    // ~32 passes over a 4 KiB buffer is ~5μs at 50 GB/s memory bandwidth
    // amortized; the multiply chains keep the CPU pipeline busy.
    for _ in 0..32 {
        for byte in payload {
            acc = acc
                .wrapping_mul(0x100000001b3)
                .wrapping_add(u64::from(*byte));
        }
    }
    acc
}

/// Pipe whose `call()` offloads CPU work via the runtime's
/// background-blocking pool, then returns the digest as a body.
struct BlockingHashPipe {
    runtime: Arc<dyn Runtime>,
    payload: Arc<[u8; PAYLOAD_LEN]>,
}

impl SendPipe for BlockingHashPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl std::future::Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let runtime = self.runtime.clone();
        let payload = self.payload.clone();
        async move {
            let work: Box<dyn FnOnce() -> Result<Box<dyn Any + Send>, ProximaError> + Send> =
                Box::new(move || {
                    let digest = cpu_work(payload.as_ref());
                    Ok(Box::new(digest) as Box<dyn Any + Send>)
                });
            let result_any = runtime.spawn_background_blocking(work).await?;
            let digest: Box<u64> = result_any
                .downcast::<u64>()
                .map_err(|_| ProximaError::Body("downcast failed".into()))?;
            Ok(Response::ok(Bytes::from(digest.to_le_bytes().to_vec())))
        }
    }
}


fn build_client_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("client runtime")
}

fn warm_client(
    runtime: &tokio::runtime::Runtime,
    addr: std::net::SocketAddr,
) -> h2::client::SendRequest<Bytes> {
    runtime.block_on(async move {
        let socket = tokio::net::TcpStream::connect(addr).await.expect("connect");
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
    let (response_future, _) = h2_client.send_request(request, true).expect("send");
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

fn payload() -> Arc<[u8; PAYLOAD_LEN]> {
    let mut buf = [0u8; PAYLOAD_LEN];
    for (idx, slot) in buf.iter_mut().enumerate() {
        *slot = (idx & 0xff) as u8;
    }
    Arc::new(buf)
}

fn start_per_core() -> std::net::SocketAddr {
    let runtime: Arc<dyn Runtime> =
        Arc::new(TokioPerCoreRuntime::new(2).expect("per-core runtime"));
    let pipe = BlockingHashPipe {
        runtime: runtime.clone(),
        payload: payload(),
    };
    let dispatch: PipeHandle = into_handle(pipe);
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
        )
        .expect("bench setup: spawn listener factory on fresh runtime");
    let addr = addr_rx.recv().expect("addr");
    std::mem::forget(runtime);
    addr
}

fn start_prime_with_pool() -> std::net::SocketAddr {
    // Attach the ProximaBackgroundPool so spawn_background_blocking goes
    // through the disciplined work-stealing pool (the production path).
    // Without this, prime falls back to a per-call std::thread::spawn
    // which is also valid but not the recommended config.
    let pool: Arc<dyn BackgroundPool> =
        Arc::new(ProximaBackgroundPool::new().expect("background pool"));
    let runtime: Arc<dyn Runtime> = Arc::new(
        PrimeRuntime::new(1)
            .expect("prime runtime")
            .with_background_pool(pool),
    );
    let pipe = BlockingHashPipe {
        runtime: runtime.clone(),
        payload: payload(),
    };
    let dispatch: PipeHandle = into_handle(pipe);
    let (addr_tx, addr_rx) = std::sync::mpsc::channel::<std::net::SocketAddr>();
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
                        proxima::runtime::prime::os::core_shard::spawn_on_current_core(Box::pin(
                            async move {
                                let _ = serve_h2_connection(
                                    socket,
                                    dispatch,
                                    proxima_listen::admission::ConnAdmission::unbounded(),
                                    None,
                                )
                                .await;
                            },
                        ));
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
    let mut group = criterion.benchmark_group("h2_spawn_blocking");
    group.sample_size(60);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1));

    let client_runtime = build_client_runtime();

    let per_core_addr = start_per_core();
    let per_core_client = warm_client(&client_runtime, per_core_addr);
    group.bench_function("tokio_per_core", |bencher| {
        bencher.to_async(&client_runtime).iter(|| {
            let client = per_core_client.clone();
            one_request(client)
        });
    });

    let prime_addr = start_prime_with_pool();
    let prime_client = warm_client(&client_runtime, prime_addr);
    group.bench_function("prime", |bencher| {
        bencher.to_async(&client_runtime).iter(|| {
            let client = prime_client.clone();
            one_request(client)
        });
    });

    group.finish();
}

criterion_group!(h2_spawn_blocking, benches);
criterion_main!(h2_spawn_blocking);
