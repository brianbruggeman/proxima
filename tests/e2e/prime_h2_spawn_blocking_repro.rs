//! End-to-end repro for the h2-on-prime body-delivery hang.
//!
//! Mirrors the exact shape of
//! `bench_h2_spawn_blocking::start_prime_with_pool` + `one_request` in
//! a deterministic, time-bounded integration test (5s deadline). The
//! hang predates the runtime-prime cross-thread wake fix; the
//! cross-thread fix in `local_executor.rs` / `core_shard.rs` is
//! necessary for legitimate BgPool wake-chains but does NOT close this
//! hang.
//!
//! Diagnostic findings (kept inline so the next investigator has the
//! trail without re-instrumenting):
//!
//! - The handler future runs to completion (sync + async variants both
//!   reach the response). HEADERS, DATA(N bytes), and DATA(empty,
//!   END_STREAM) are all encoded into `Connection::output` and
//!   `write_all`'d to the socket in a single combined buffer.
//! - Hex dump of the combined buffer is valid h2:
//!   `HEADERS(stream=1, END_HEADERS, :status=200) | DATA(stream=1, 8 bytes) | DATA(stream=1, END_STREAM, 0 bytes)`.
//! - `socket.send` returns `Ok(36)` for the combined write — bytes
//!   reach the kernel TCP buffer.
//! - The tokio h2 client receives the HEADERS frame and resolves
//!   `response_future`. But `body.data().await` never yields the DATA
//!   chunk, and the test times out at 5s. After timeout, the dropped
//!   client closes the connection; the server's `serve_h2_connection`
//!   exits cleanly with `Ok(())`.
//!
//! Hypothesis (un-validated): the prime `TcpStream::poll_write` returns
//! `Ok(36)` but the bytes are not actually flushed to the wire in a
//! way the peer's tokio h2 client can consume. Could be a `socket2`
//! interaction with `TCP_NODELAY` (already set on accept), a sub-batch
//! `MSG_NOSIGNAL` issue, or a deeper kernel-side queueing artifact.
//! The h2 client's `body.data()` is awaiting the DATA frame; the
//! server side has nothing left to do; classic stalemate.
//!
//! Resume: next step is to (a) capture a tcpdump of localhost on the
//! bound port for ONE request and confirm the server's 36-byte write
//! actually appears on the wire, OR (b) replace the prime server's
//! `ProximaTcpStream` with `tokio::net::TcpStream` and re-run this
//! test — if it passes, the bug is isolated to `ProximaTcpStream` /
//! the prime reactor's write path. If it still hangs, the bug is in
//! `serve_h2_connection`'s loop shape under prime's executor.

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
use proxima::error::ProximaError;
use proxima::h2::serve_h2_connection;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::request::{Request, Response};
use proxima::runtime::prime::os::background::ProximaBackgroundPool;
use proxima::runtime::prime::os::net::TcpListener as ProximaTcpListener;
use proxima::runtime::{BackgroundPool, CoreId, PrimeRuntime, Runtime};
use proxima_primitives::pipe::SendPipe;

const PAYLOAD_LEN: usize = 4096;

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


/// Same wiring as BlockingHashPipe but the handler is sync — does NOT
/// hit the BgPool. If the prime repro test hangs with this handler too,
/// the bug is in the h2-on-prime path itself, not the BgPool/oneshot
/// cross-thread wake.
struct SyncHashPipe {
    payload: Arc<[u8; PAYLOAD_LEN]>,
}

impl SendPipe for SyncHashPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl std::future::Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let payload = self.payload.clone();
        async move {
            eprintln!("[repro/pipe] SyncHashPipe::call POLLED");
            let digest = cpu_work(payload.as_ref());
            eprintln!("[repro/pipe] returning response with body");
            Ok(Response::ok(Bytes::from(digest.to_le_bytes().to_vec())))
        }
    }
}


fn payload() -> Arc<[u8; PAYLOAD_LEN]> {
    let mut buf = [0u8; PAYLOAD_LEN];
    for (index, slot) in buf.iter_mut().enumerate() {
        *slot = (index & 0xff) as u8;
    }
    Arc::new(buf)
}

fn start_prime_with_pool(use_bg_pool: bool) -> std::net::SocketAddr {
    eprintln!("[repro] building runtime (use_bg_pool={use_bg_pool})");
    let runtime: Arc<dyn Runtime> = if use_bg_pool {
        let pool: Arc<dyn BackgroundPool> =
            Arc::new(ProximaBackgroundPool::new().expect("background pool"));
        Arc::new(
            PrimeRuntime::new(1)
                .expect("prime runtime")
                .with_background_pool(pool),
        )
    } else {
        Arc::new(PrimeRuntime::new(1).expect("prime runtime"))
    };
    let dispatch: PipeHandle = if use_bg_pool {
        into_handle(BlockingHashPipe {
            runtime: runtime.clone(),
            payload: payload(),
        })
    } else {
        into_handle(SyncHashPipe { payload: payload() })
    };
    let (addr_tx, addr_rx) = std::sync::mpsc::channel::<std::net::SocketAddr>();
    eprintln!("[repro] dispatching listener factory");
    runtime
        .spawn_factory_on_core(
            CoreId(0),
            Box::new(move || {
                let dispatch = dispatch;
                Box::pin(async move {
                    eprintln!("[repro/listener] future polled, binding");
                    let mut listener =
                        ProximaTcpListener::bind("127.0.0.1:0".parse().expect("parse listen addr"))
                            .expect("bind");
                    let addr = listener.local_addr().expect("local_addr");
                    eprintln!("[repro/listener] bound at {addr}");
                    addr_tx.send(addr).expect("addr send");
                    loop {
                        eprintln!("[repro/listener] awaiting accept");
                        let (socket, peer) = match listener.accept().await {
                            Ok(value) => value,
                            Err(error) => {
                                eprintln!("[repro/listener] accept error: {error}");
                                break;
                            }
                        };
                        eprintln!("[repro/listener] accepted peer={peer}");
                        let dispatch = dispatch.clone();
                        proxima::runtime::prime::os::core_shard::spawn_on_current_core(Box::pin(
                            async move {
                                let admission =
                                    proxima_listen::admission::ConnAdmission::unbounded();
                                eprintln!("[repro/serve] entering serve_h2_connection");
                                let result =
                                    serve_h2_connection(socket, dispatch, admission, None)
                                        .await;
                                eprintln!("[repro/serve] serve_h2_connection -> {result:?}");
                            },
                        ));
                        eprintln!("[repro/listener] handler spawned");
                    }
                    eprintln!("[repro/listener] exiting accept loop");
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }),
        )
        .expect("spawn listener factory");
    eprintln!("[repro] waiting for listener addr");
    let addr = addr_rx.recv().expect("addr");
    eprintln!("[repro] got addr={addr}");
    std::mem::forget(runtime);
    addr
}

fn run_one_request(addr: std::net::SocketAddr) {
    let outer = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("client runtime");
    outer.block_on(async move {
        let result = tokio::time::timeout(Duration::from_secs(5), async move {
            eprintln!("[repro/client] connecting to {addr}");
            let socket = tokio::net::TcpStream::connect(addr).await.expect("connect");
            let _ = socket.set_nodelay(true);
            eprintln!("[repro/client] h2 handshake");
            let (mut h2_client, h2_conn) = h2::client::handshake(socket).await.expect("handshake");
            let conn_handle = tokio::spawn(async move {
                let outcome = h2_conn.await;
                eprintln!("[repro/client] h2_conn -> {outcome:?}");
            });
            let request = http::Request::builder()
                .method("GET")
                .uri("http://localhost/")
                .body(())
                .expect("request");
            eprintln!("[repro/client] sending request");
            let (response_future, _) = h2_client.send_request(request, true).expect("send request");
            eprintln!("[repro/client] awaiting response");
            let response = response_future.await.expect("response");
            eprintln!("[repro/client] got response status");
            assert_eq!(response.status().as_u16(), 200, "expected 200 OK");
            let mut body = response.into_body();
            let mut total = 0usize;
            let mut iterations = 0usize;
            loop {
                iterations += 1;
                eprintln!("[repro/client] body.data() iter {iterations} START");
                let next = body.data().await;
                eprintln!(
                    "[repro/client] body.data() iter {iterations} returned: {}",
                    match &next {
                        Some(Ok(chunk)) => format!("Some(Ok({} bytes))", chunk.len()),
                        Some(Err(error)) => format!("Some(Err({error}))"),
                        None => "None".into(),
                    },
                );
                match next {
                    None => break,
                    Some(Err(error)) => panic!("body chunk error: {error}"),
                    Some(Ok(chunk)) => {
                        total += chunk.len();
                        body.flow_control()
                            .release_capacity(chunk.len())
                            .expect("flow control");
                    }
                }
            }
            // Second request on the SAME connection — mirrors the
            // bench's `bencher.iter(|| one_request(client.clone()))`
            // pattern. If the second request hangs, the bench would
            // hang in warmup.
            eprintln!("[repro/client] second request on same connection");
            let request2 = http::Request::builder()
                .method("GET")
                .uri("http://localhost/")
                .body(())
                .expect("request2");
            let (response_future2, _) = h2_client.send_request(request2, true).expect("send req2");
            let response2 = response_future2.await.expect("response2");
            eprintln!(
                "[repro/client] second response status = {}",
                response2.status()
            );
            let mut body2 = response2.into_body();
            let mut iter2 = 0;
            loop {
                iter2 += 1;
                eprintln!("[repro/client] req2 body.data() iter {iter2} START");
                let next = body2.data().await;
                eprintln!(
                    "[repro/client] req2 body.data() iter {iter2} returned: {}",
                    match &next {
                        Some(Ok(chunk)) => format!("Some(Ok({} bytes))", chunk.len()),
                        Some(Err(error)) => format!("Some(Err({error}))"),
                        None => "None".into(),
                    }
                );
                match next {
                    None => break,
                    Some(Err(error)) => panic!("body2 chunk error: {error}"),
                    Some(Ok(chunk)) => {
                        total += chunk.len();
                        body2
                            .flow_control()
                            .release_capacity(chunk.len())
                            .expect("flow control 2");
                    }
                }
            }
            drop(conn_handle);
            drop(h2_client);
            total
        })
        .await;
        match result {
            Ok(total) => assert!(total > 0, "expected non-empty response body"),
            Err(_) => panic!(
                "prime h2 body-delivery hang: HEADERS received by client but DATA chunk \
                 never delivered within 5s (see file-level doc for the diagnostic trail)",
            ),
        }
    });
}

#[allow(dead_code)]
fn enable_h2_tracing_once() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("h2=trace")),
            )
            .with_test_writer()
            .try_init();
    });
}

#[test]
fn prime_h2_with_bg_pool_completes_one_request_in_five_seconds() {
    run_one_request(start_prime_with_pool(true));
}

#[test]
fn prime_h2_with_sync_handler_completes_one_request_in_five_seconds() {
    run_one_request(start_prime_with_pool(false));
}

/// Mirror the bench's exact pattern: ONE warm_client, then a tight
/// loop of `one_request(client.clone())` reusing the same connection.
/// This is what `cargo bench --bench bench_h2_spawn_blocking` does.
/// If this test completes 200 requests in <10s, the bench shouldn't
/// hang. If it hangs, the bug is in the bench-style request reuse
/// pattern.
async fn one_request_like_bench(mut h2_client: h2::client::SendRequest<bytes::Bytes>) -> usize {
    let request = http::Request::builder()
        .method("GET")
        .uri("http://localhost/")
        .body(())
        .expect("request");
    let (response_future, _) = h2_client.send_request(request, true).expect("send");
    let response = response_future.await.expect("response");
    let mut body = response.into_body();
    let mut total = 0usize;
    while let Some(chunk) = body.data().await {
        let chunk = chunk.expect("chunk");
        total += chunk.len();
        body.flow_control()
            .release_capacity(chunk.len())
            .expect("flow control");
    }
    total
}

async fn run_bench_pattern_200_requests(addr: std::net::SocketAddr, label: &'static str) {
    let result = tokio::time::timeout(Duration::from_secs(15), async move {
        let socket = tokio::net::TcpStream::connect(addr).await.expect("connect");
        let _ = socket.set_nodelay(true);
        let (h2_client, h2_conn) = h2::client::handshake(socket).await.expect("handshake");
        let conn = tokio::spawn(async move {
            let _ = h2_conn.await;
        });
        const ITERS: usize = 200;
        let start = std::time::Instant::now();
        for index in 0..ITERS {
            let body_len = one_request_like_bench(h2_client.clone()).await;
            if index < 3 || index == ITERS - 1 {
                eprintln!(
                    "[{label}] iter {index}: body_len={body_len} elapsed={:?}",
                    start.elapsed()
                );
            }
            assert!(body_len > 0, "{label} iter {index}: empty body");
        }
        let elapsed = start.elapsed();
        eprintln!("[{label}] {ITERS} requests in {elapsed:?}");
        drop(conn);
        drop(h2_client);
        elapsed
    })
    .await;
    match result {
        Ok(_) => {}
        Err(_) => panic!("{label} hung past 15s on 200 sequential requests"),
    }
}

#[test]
fn prime_h2_bench_pattern_sync_pipe() {
    let addr = start_prime_with_pool(false);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("driver runtime");
    runtime.block_on(run_bench_pattern_200_requests(addr, "sync-pipe"));
}

#[test]
fn prime_h2_bench_pattern_bg_pool_pipe() {
    let addr = start_prime_with_pool(true);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("driver runtime");
    runtime.block_on(run_bench_pattern_200_requests(addr, "bg-pool-pipe"));
}
