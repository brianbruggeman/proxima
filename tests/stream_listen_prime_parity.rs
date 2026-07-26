//! Regression proof for the `"stream"` listener panicking on the Prime
//! runtime: `StreamListenProtocol::spawn_handler` used to hardcode
//! `tokio::task::spawn_local`, which needs a tokio `LocalSet` context that
//! a Prime `CoreShard` worker never enters (Prime has no tokio dependency
//! at all). The fix threads the installed `ServeContext::runtime` through
//! to `spawn_handler`, which dispatches via `Runtime::spawn_on_current_core`
//! when present — the same pattern `HttpListenProtocol` already uses (see
//! `serve_parity.rs`).
//!
//! Drives the real `StreamListenProtocol::serve` path bound through
//! `PrimeAcceptorFactory` on a `PrimeRuntime` worker and round-trips bytes
//! over a raw TCP connection — the exact shape `mc-server` (Prime + the
//! `"stream"` protocol) hit in practice.

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool",
    feature = "tcp"
))]

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;

use bytes::Bytes;
use futures::channel::oneshot;

use proxima::error::ProximaError;
use proxima::listen::{ListenProtocol, ServeContext};
use proxima::listeners::StreamListenProtocol;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::prime::{CoreId, PrimeRuntime};
use proxima::request::{Request, Response};
use proxima::runtime::Runtime;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::telemetry_surface::NoopTelemetry;

#[cfg(not(feature = "tokio"))]
use futures::io::{AsyncReadExt, AsyncWriteExt};
#[cfg(not(feature = "tokio"))]
use prime::os::net::TcpStream;
#[cfg(feature = "tokio")]
use tokio::io::{AsyncReadExt, AsyncWriteExt};
#[cfg(feature = "tokio")]
use tokio::net::TcpStream;

/// Echoes the request body back — proves the per-connection handler ran
/// to completion, not just that the connection was accepted.
struct EchoPipe;

impl SendPipe for EchoPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move {
            let (_, bytes) = request.body_bytes().await?;
            Ok(Response::ok(bytes))
        }
    }
}

// sync std bind-then-drop — no reactor needed either way, so this one
// helper serves both the tokio and the prime-hosted test below.
fn pick_free_addr() -> SocketAddr {
    let probe = std::net::TcpListener::bind("127.0.0.1:0").expect("probe bind");
    probe.local_addr().expect("probe addr")
}

/// Bind + accept the real `StreamListenProtocol::serve` ON a prime worker
/// via `spawn_factory_on_core(CoreId(0), ...)`, mirroring
/// `serve_parity.rs::spawn_prime_serve`. The serve future borrows the
/// protocol + spec (`'_`), so both are leaked to `'static`.
fn spawn_prime_serve(runtime: &Arc<PrimeRuntime>, addr: SocketAddr) -> oneshot::Sender<()> {
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let dispatch: PipeHandle = into_handle(EchoPipe);
    let runtime_for_context: Arc<dyn Runtime> = runtime.clone();

    let protocol: &'static StreamListenProtocol = Box::leak(Box::new(StreamListenProtocol::new()));
    let spec: &'static serde_json::Value = Box::leak(Box::new(serde_json::Value::Null));

    runtime
        .spawn_factory_on_core(
            CoreId(0),
            Box::new(move || {
                let context = ServeContext::new(NoopTelemetry::handle())
                    .with_runtime(runtime_for_context)
                    .with_acceptor_factory(Arc::new(proxima_net::prime::PrimeAcceptorFactory));
                Box::pin(async move {
                    // bind happens here, on the worker with CURRENT_REACTOR
                    // live — the prime TcpListener requires it.
                    let _ = protocol
                        .serve(addr, dispatch, spec, context, shutdown_rx)
                        .await;
                }) as std::pin::Pin<Box<dyn Future<Output = ()> + 'static>>
            }),
        )
        .expect("spawn prime serve factory on core 0");

    shutdown_tx
}

async fn connect_retry(addr: SocketAddr) -> TcpStream {
    for _ in 0..200 {
        match TcpStream::connect(addr).await {
            Ok(stream) => return stream,
            #[cfg(feature = "tokio")]
            Err(_) => tokio::time::sleep(std::time::Duration::from_millis(20)).await,
            #[cfg(not(feature = "tokio"))]
            Err(_) => proxima::time::sleep(std::time::Duration::from_millis(20)).await,
        }
    }
    panic!("listener at {addr} never accepted a connection");
}

// Before the fix, the first accepted connection panicked the Prime worker
// thread inside `spawn_handler` (`tokio::task::spawn_local` outside any
// tokio LocalSet) — this test would hang waiting for a response that never
// comes, rather than complete. After the fix, the connection is handled via
// `Runtime::spawn_on_current_core` and the echo round-trips normally.
#[cfg(feature = "tokio")]
#[proxima::test(flavor = "multi_thread", worker_threads = 4)]
async fn stream_listener_round_trips_bytes_on_prime() {
    let runtime = Arc::new(
        PrimeRuntime::builder()
            .cores(1)
            .background_inline()
            .build()
            .expect("build prime runtime"),
    );
    let addr = pick_free_addr();
    let shutdown_tx = spawn_prime_serve(&runtime, addr);

    let mut client = connect_retry(addr).await;
    client
        .write_all(b"the quick brown fox")
        .await
        .expect("client write");
    client.shutdown().await.expect("client shutdown write");
    let mut response = Vec::new();
    client
        .read_to_end(&mut response)
        .await
        .expect("client read");

    drop(shutdown_tx);
    assert_eq!(response, b"the quick brown fox");
}

// tokio-free sibling: same regression proof, `prime::os::net::TcpStream`
// as the raw client instead of tokio's — exercises the exact tokio-free
// build (`--no-default-features --features serve-prime,tcp,macros`) this
// crate now supports (see the umbrella Cargo.toml's `tcp` doc). Half-close
// via `AsyncWriteExt::close()` (write-only on `prime::os::net::TcpStream`
// since the `poll_close` fix below — it used to shut down both directions,
// which would have zeroed this test's own response read too).
#[cfg(not(feature = "tokio"))]
#[proxima::test]
async fn stream_listener_round_trips_bytes_on_prime() {
    let runtime = Arc::new(
        PrimeRuntime::builder()
            .cores(1)
            .background_inline()
            .build()
            .expect("build prime runtime"),
    );
    let addr = pick_free_addr();
    let shutdown_tx = spawn_prime_serve(&runtime, addr);

    let mut client = connect_retry(addr).await;
    client
        .write_all(b"the quick brown fox")
        .await
        .expect("client write");
    client.close().await.expect("client shutdown write");
    let mut response = Vec::new();
    client
        .read_to_end(&mut response)
        .await
        .expect("client read");

    drop(shutdown_tx);
    assert_eq!(response, b"the quick brown fox");
}
