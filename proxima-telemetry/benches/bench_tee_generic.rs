#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! Stream fan-out throughput bench.
//!
//! The original five-arm suite compared the `proxima` umbrella crate's
//! generic `Tee<T>` (single-consumer drain, one-sink, 8-sink, replay) against
//! `tokio::sync::broadcast`. proxima-telemetry does not depend on the `proxima`
//! umbrella crate and has no local `Tee<T>` equivalent, so the four
//! `Tee`-dependent arms are dropped. The self-contained `tokio_broadcast_8c_1M`
//! incumbent arm is retained as the meaningful fan-out throughput baseline.

use std::sync::Arc;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

const LARGE_N: usize = 1_000_000;
const SINK_QUEUE: usize = 256;
const N_CONSUMERS: usize = 8;

// design-favors: incumbent — tokio_broadcast's home-turf workload.
// 1 producer, 8 consumers, 1M items each. bounded channel (256).
fn tokio_broadcast_8c_1m(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("tokio_broadcast_8c_1M");
    group.measurement_time(Duration::from_secs(10));
    group.throughput(Throughput::Elements(LARGE_N as u64 * N_CONSUMERS as u64));

    // multi-thread runtime required: 8 consumer tasks must actually run
    // while the producer pushes. current-thread would deadlock because
    // a synchronous push blocks until a consumer yields.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(N_CONSUMERS + 1)
        .enable_all()
        .build()
        .expect("multi-thread runtime");

    group.bench_function(BenchmarkId::from_parameter(LARGE_N), |bencher| {
        bencher.to_async(&runtime).iter(|| async {
            let (tx, _rx) = tokio::sync::broadcast::channel::<u64>(SINK_QUEUE);
            let barrier = Arc::new(tokio::sync::Barrier::new(N_CONSUMERS + 1));

            let mut consumer_tasks = Vec::new();
            for _ in 0..N_CONSUMERS {
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
                // send returns Err only when there are no receivers
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
