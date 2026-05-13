#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! `Tee::wrap` fan-out and replay throughput. Tee splits one Body
//! into a primary path and a sink path, with a bounded replay cap;
//! used by the Selection fall-through path so a second upstream
//! can re-read a request body the first upstream already consumed.
//! Two measurements:
//!
//! 1. `wrap_then_drain_primary` — pure primary drain cost when the
//!    sink isn't actively read (the common selection-success path).
//! 2. `wrap_and_replay` — drain primary, then re-read via `replay`
//!    (the fall-through path — body is buffered in the tee up to
//!    `cap_bytes`).

use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use futures::StreamExt;
use proxima::{ChunkStream, DEFAULT_REPLAY_CAP_BYTES, ProximaError, Replay};
use tokio::runtime::Runtime;

fn build_runtime() -> Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime")
}

/// One-chunk ChunkStream over a buffered payload — Tee now splits a
/// `ChunkStream`, not a `Body`.
fn chunk_stream(payload: Bytes) -> ChunkStream {
    Box::pin(futures::stream::once(async move {
        Ok::<Bytes, ProximaError>(payload)
    }))
}

/// Drain a ChunkStream, returning the total byte count (replaces the old
/// `Body::collect().len()`).
async fn drain(mut stream: ChunkStream) -> usize {
    let mut total = 0;
    while let Some(chunk) = stream.next().await {
        total += chunk.expect("chunk").len();
    }
    total
}

fn wrap_then_drain_primary(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("tee_wrap_drain_primary");
    group.measurement_time(Duration::from_secs(3));
    let runtime = build_runtime();
    for &size in &[256_usize, 4 * 1024, 64 * 1024] {
        group.throughput(Throughput::Bytes(size as u64));
        let payload = Bytes::from(vec![b'A'; size]);
        group.bench_function(format!("size_{size}"), |bencher| {
            bencher.to_async(&runtime).iter(|| {
                let payload = payload.clone();
                async move {
                    let (_tee, primary) =
                        Replay::wrap(chunk_stream(payload), DEFAULT_REPLAY_CAP_BYTES);
                    let len = drain(primary).await;
                    std::hint::black_box(len);
                }
            });
        });
    }
    group.finish();
}

fn wrap_and_replay(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("tee_wrap_and_replay");
    group.measurement_time(Duration::from_secs(3));
    let runtime = build_runtime();
    for &size in &[256_usize, 4 * 1024] {
        group.throughput(Throughput::Bytes(size as u64 * 2));
        let payload = Bytes::from(vec![b'A'; size]);
        group.bench_function(format!("size_{size}"), |bencher| {
            bencher.to_async(&runtime).iter(|| {
                let payload = payload.clone();
                async move {
                    let (tee, primary) =
                        Replay::wrap(chunk_stream(payload), DEFAULT_REPLAY_CAP_BYTES);
                    let _ = drain(primary).await;
                    let replay = tee.replay().expect("replay");
                    let len = drain(replay).await;
                    std::hint::black_box(len);
                }
            });
        });
    }
    group.finish();
}

criterion_group!(benches, wrap_then_drain_primary, wrap_and_replay);
criterion_main!(benches);
