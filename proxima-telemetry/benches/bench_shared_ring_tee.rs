//! bench_shared_ring_tee — fan-out throughput baseline.
//!
//! The original suite compared the `proxima` umbrella crate's `SharedRingTee<T>`
//! and `PerSinkTee<T>` against `tokio::sync::broadcast`. proxima-telemetry does
//! not depend on the `proxima` umbrella crate and has no local `SharedRingTee` /
//! `PerSinkTee` equivalent, so the four arms that referenced them are dropped.
//! The self-contained `tokio_broadcast` incumbent arm is retained as the
//! meaningful fan-out throughput baseline.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use std::sync::Arc;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

const LARGE_N: usize = 1_000_000;
const SINK_QUEUE: usize = 256;
const N_CONSUMERS_LARGE: usize = 8;

// design-favors: incumbent — tokio broadcast home turf.
fn tokio_broadcast_8c_1m(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("tokio_broadcast_8c_1M");
    group.measurement_time(Duration::from_secs(10));
    group.throughput(Throughput::Elements(
        LARGE_N as u64 * N_CONSUMERS_LARGE as u64,
    ));

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(N_CONSUMERS_LARGE + 1)
        .enable_all()
        .build()
        .expect("multi-thread runtime");

    group.bench_function("tokio_broadcast", |bencher| {
        bencher.to_async(&runtime).iter(|| async {
            let (tx, _rx) = tokio::sync::broadcast::channel::<u64>(SINK_QUEUE);
            let barrier = Arc::new(tokio::sync::Barrier::new(N_CONSUMERS_LARGE + 1));

            let mut consumer_tasks = Vec::new();
            for _ in 0..N_CONSUMERS_LARGE {
                let mut consumer_rx = tx.subscribe();
                let barrier_clone = barrier.clone();
                consumer_tasks.push(tokio::spawn(async move {
                    barrier_clone.wait().await;
                    let mut count = 0usize;
                    loop {
                        match consumer_rx.recv().await {
                            Ok(_) => count += 1,
                            Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                            Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                                count += skipped as usize;
                            }
                        }
                        if count >= LARGE_N {
                            break;
                        }
                    }
                    count
                }));
            }

            barrier.wait().await;
            for item in 0..LARGE_N as u64 {
                let _ = tx.send(item);
            }
            drop(tx);

            let mut total = 0usize;
            for task in consumer_tasks {
                total += task.await.expect("consumer join");
            }
            std::hint::black_box(total);
        });
    });
    group.finish();
}

criterion_group!(benches, tokio_broadcast_8c_1m);
criterion_main!(benches);
