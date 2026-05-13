#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
#![cfg(all(feature = "http3", feature = "runtime-tokio"))]

//! H3 runtime-swap. Mirrors `h2_runtime_swap` Story A.
//!
//! Same proxima h3 listener served under two runtimes:
//!
//! - **default tokio multi-thread** (work-stealing, N worker threads,
//!   `tokio::spawn` for each per-connection driver).
//! - **proxima `TokioPerCoreRuntime`** (pinned tokio current-thread
//!   per CPU via `core_affinity`; accept loop + every per-connection
//!   driver runs on the same core; `tokio::task::spawn_local` for
//!   `?Send`-friendly dispatch).
//!
//! Story B (vs hyper / pingora) is N/A for h3 — hyper 1.9 and pingora
//! 0.8 ship no native h3 server. Everyone in the Rust ecosystem who
//! wants h3 today calls the same `h3 + h3-quinn` pair proxima wraps,
//! so a cross-stack column would have ≈0 signal. When a native h3
//! lands here (parking-lot `proxima::h3` native rewrite) Story B
//! becomes proxima-native-h3 vs h3-crate, mirroring `h2_native_vs_h2_crate`.

use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima::error::ProximaError;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::request::{Request, Response};
use proxima::runtime::{CoreId, Runtime, TokioPerCoreRuntime};
use proxima_primitives::pipe::SendPipe;

#[path = "common/h3_setup.rs"]
mod h3_setup;

const RESPONSE_BODY: &[u8] = b"ok";

struct ConstantOk;

impl SendPipe for ConstantOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move { Ok(Response::ok(Bytes::from_static(RESPONSE_BODY))) }
    }
}


fn build_client_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("client runtime")
}

/// Boot proxima h3 on the **default tokio multi-thread** runtime.
/// Each accepted connection is driven on a `tokio::spawn`-spawned
/// task — work-stealing across the runtime's worker threads.
fn start_proxima_default_tokio() -> SocketAddr {
    h3_setup::install_crypto_provider();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("default tokio server runtime");
    let dispatch: PipeHandle = into_handle(ConstantOk);
    let addr = runtime.block_on(async move {
        let server_config =
            proxima::quic::dev_server_config(vec!["localhost".to_string()], &[b"h3"])
                .expect("dev server config");
        let endpoint = Arc::new(
            proxima::quic::Endpoint::server(
                (std::net::Ipv4Addr::LOCALHOST, 0).into(),
                server_config,
            )
            .expect("quic bind"),
        );
        let bound = endpoint.local_addr().expect("local addr");
        let endpoint_for_accept = endpoint.clone();
        tokio::spawn(async move {
            loop {
                match endpoint_for_accept.accept().await {
                    Some(Ok(connection)) => {
                        let dispatch = dispatch.clone();
                        let in_flight = Arc::new(AtomicU64::new(0));
                        tokio::spawn(async move {
                            let _ =
                                proxima::h3::serve_h3_connection(connection, dispatch, in_flight)
                                    .await;
                        });
                    }
                    Some(Err(_)) => continue,
                    None => break,
                }
            }
        });
        bound
    });
    // Server runtime must outlive the bench. Process exit cleans up.
    std::mem::forget(runtime);
    addr
}

/// Boot proxima h3 on the **per-core runtime** (`TokioPerCoreRuntime`).
/// The accept loop + every per-connection h3 driver runs on the same
/// pinned core via `spawn_local`. Two cores total so the runtime
/// matches `start_proxima_default_tokio`'s thread budget for a fair
/// comparison; only core 0 hosts the listener — core 1 is idle here
/// (matches `h2_runtime_swap`'s shape).
fn start_proxima_per_core() -> SocketAddr {
    h3_setup::install_crypto_provider();
    let runtime = TokioPerCoreRuntime::new(2).expect("per-core runtime");
    let (addr_tx, addr_rx) = std::sync::mpsc::channel::<SocketAddr>();
    let dispatch: PipeHandle = into_handle(ConstantOk);
    let _ = runtime.spawn_factory_on_core(
        CoreId(0),
        Box::new(move || {
            Box::pin(async move {
                let server_config =
                    proxima::quic::dev_server_config(vec!["localhost".to_string()], &[b"h3"])
                        .expect("dev server config");
                let endpoint = Arc::new(
                    proxima::quic::Endpoint::server(
                        (std::net::Ipv4Addr::LOCALHOST, 0).into(),
                        server_config,
                    )
                    .expect("quic bind"),
                );
                let bound = endpoint.local_addr().expect("local addr");
                addr_tx.send(bound).expect("addr send");
                let endpoint_for_accept = endpoint.clone();
                loop {
                    match endpoint_for_accept.accept().await {
                        Some(Ok(connection)) => {
                            let dispatch = dispatch.clone();
                            let in_flight = Arc::new(AtomicU64::new(0));
                            // Same-core spawn — no cross-core hops
                            // for the connection's lifetime.
                            tokio::task::spawn_local(async move {
                                let _ = proxima::h3::serve_h3_connection(
                                    connection, dispatch, in_flight,
                                )
                                .await;
                            });
                        }
                        Some(Err(_)) => continue,
                        None => break,
                    }
                }
            }) as Pin<Box<dyn Future<Output = ()> + 'static>>
        }),
    );
    let addr = addr_rx.recv().expect("addr from per-core worker");
    std::mem::forget(runtime);
    addr
}

fn run_warm_bench(criterion: &mut Criterion, group_label: &str, addr: SocketAddr) {
    let client_runtime = build_client_runtime();
    let (_client_endpoint, send_request) = client_runtime.block_on(async {
        let endpoint = h3_setup::make_client_endpoint();
        let send = h3_setup::warm_h3_client(&endpoint, addr).await;
        (endpoint, send)
    });
    let uri = format!("https://localhost:{}/", addr.port());

    let mut group = criterion.benchmark_group(group_label);
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1));
    group.bench_function("get_then_200_ok_on_warm_connection", |bencher| {
        let send_request = send_request.clone();
        let uri = uri.clone();
        bencher.to_async(&client_runtime).iter(|| {
            let uri = uri.clone();
            let mut send_request = send_request.clone();
            async move {
                let request = http::Request::builder()
                    .method("GET")
                    .uri(&uri)
                    .body(())
                    .expect("request");
                let mut stream = send_request.send_request(request).await.expect("send");
                stream.finish().await.expect("finish");
                let response = stream.recv_response().await.expect("response");
                std::hint::black_box(response.status());
                while let Some(chunk) = stream.recv_data().await.expect("recv_data") {
                    std::hint::black_box(chunk);
                }
            }
        });
    });
    group.finish();
}

fn proxima_default_tokio(criterion: &mut Criterion) {
    let addr = start_proxima_default_tokio();
    run_warm_bench(criterion, "h3_runtime_swap_default_tokio", addr);
}

fn proxima_per_core(criterion: &mut Criterion) {
    let addr = start_proxima_per_core();
    run_warm_bench(criterion, "h3_runtime_swap_per_core", addr);
}

criterion_group!(benches, proxima_default_tokio, proxima_per_core);
criterion_main!(benches);
