#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
#![cfg(feature = "websocket-upstream")]

//! P8 — WebSocket upstream micro-bench. Disciplined-component shape
//! per `docs/protocol-gap/discipline.md`: measure proxima's
//! `WebSocketUpstream` round-trip cost against direct
//! `tokio-tungstenite`-style client usage to size the abstraction
//! tax proxima adds on top of the raw WS framing.
//!
//! Two arms today; more land as the impl matures:
//!
//! - `proxima_send_recv` — `WebSocketUpstream::call(request)` against
//!   an echo server hosted on the same runtime.
//! - `direct_async_tungstenite_send_recv` — same echo, but the
//!   client side is hand-written against `async_tungstenite::tokio`
//!   with no proxima wrapper. measures the floor.
//!
//! delta gives the substrate-tax number we record in the
//! discipline.md changelog row for every tweak.

use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

const PAYLOAD: &[u8] = b"ping";

fn build_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("tokio runtime")
}

/// Spin up an echo WebSocket server on a loopback ephemeral port.
/// Returns `ws://host:port` so both bench arms can connect.
async fn start_echo_server() -> String {
    use async_tungstenite::tokio::accept_async;
    use async_tungstenite::tungstenite::Message;
    use futures::StreamExt;
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        while let Ok((tcp, _peer)) = listener.accept().await {
            tokio::spawn(async move {
                let Ok(mut ws) = accept_async(tcp).await else {
                    return;
                };
                while let Some(Ok(msg)) = ws.next().await {
                    match msg {
                        Message::Binary(_) | Message::Text(_) => {
                            // echo unmodified
                            let _ = futures::sink::SinkExt::send(&mut ws, msg).await;
                        }
                        Message::Close(_) => break,
                        _ => {}
                    }
                }
            });
        }
    });
    format!("ws://{addr}")
}

fn proxima_round_trip(criterion: &mut Criterion) {
    use proxima::SendPipe;
    use proxima::request::Request;
    use proxima::upstreams::WebSocketUpstream;

    let mut group = criterion.benchmark_group("ws_upstream_proxima");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(PAYLOAD.len() as u64));
    let runtime = build_runtime();
    let url = runtime.block_on(start_echo_server());

    group.bench_function("send_recv_warm_connection", |bencher| {
        let upstream = WebSocketUpstream::new(url.clone());
        bencher.to_async(&runtime).iter(|| {
            let upstream = &upstream;
            async move {
                let request = Request::builder()
                    .method("POST")
                    .path("/")
                    .body(Bytes::from_static(PAYLOAD))
                    .build()
                    .expect("request");
                let response = upstream.call(request).await.expect("response");
                std::hint::black_box(response.status);
            }
        });
    });
    group.finish();
}

fn direct_tungstenite_round_trip(criterion: &mut Criterion) {
    use async_tungstenite::tokio::connect_async;
    use async_tungstenite::tungstenite::Message;
    use futures::StreamExt;

    let mut group = criterion.benchmark_group("ws_upstream_direct_async_tungstenite");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(PAYLOAD.len() as u64));
    let runtime = build_runtime();
    let url = runtime.block_on(start_echo_server());

    group.bench_function("send_recv_warm_connection", |bencher| {
        let stream = runtime.block_on(async {
            let (stream, _resp) = connect_async(&url).await.expect("connect");
            stream
        });
        // Lock per-iter so the bench is fair: same single-connection
        // serial-send semantics as the proxima upstream.
        let stream = std::sync::Arc::new(tokio::sync::Mutex::new(stream));
        bencher.to_async(&runtime).iter(|| {
            let stream = stream.clone();
            async move {
                let mut guard = stream.lock().await;
                let payload: Vec<u8> = PAYLOAD.to_vec();
                futures::sink::SinkExt::send(&mut *guard, Message::Binary(payload.into()))
                    .await
                    .expect("send");
                let msg = guard.next().await.expect("frame").expect("ok");
                std::hint::black_box(msg);
            }
        });
    });
    group.finish();
}

criterion_group!(benches, proxima_round_trip, direct_tungstenite_round_trip);
criterion_main!(benches);
