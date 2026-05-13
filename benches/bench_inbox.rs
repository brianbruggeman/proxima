//! micro-bench for the proxima MPSC inbox vs flume / tokio / std baselines.
//!
//! incumbents (versions pinned in Cargo.toml):
//!   - flume 0.12 (bounded + unbounded) — MPMC channel under contention
//!   - tokio::sync::mpsc 1.x — async bounded MPSC for runtime task dispatch
//!   - std::sync::mpsc — std bounded MPSC, single-producer optimized
//!
//! groups (and design-favors per workload):
//!   - inbox_spsc_throughput     design-favors: prime  (per-core SPSC lane is proxima's design)
//!   - inbox_mpsc_fanin_4        design-favors: incumbent (4-producer MPMC contention — flume/tokio design point)
//!   - inbox_mpsc_fanin_8        design-favors: incumbent (8-producer)
//!   - inbox_mpsc_fanin_16       design-favors: incumbent (16-producer, peak contention before oversub)
//!   - inbox_mpsc_fanin_32       design-favors: incumbent (32-producer, 3× oversub on M1)
//!
//! capacity = 1024 across all backends. Throughput::Elements(N) per group so
//! results render as items/sec.
//!
//! requires-features: runtime-prime-inbox-alloc, runtime-tokio (for flume +
//! tokio::sync::mpsc).

#![cfg(all(feature = "runtime-prime-inbox-alloc", feature = "runtime-tokio"))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use std::thread;
use std::time::{Duration, Instant};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima::runtime::prime::core::inbox;

const CAPACITY: usize = 1024;
const ITEMS_PER_TRIAL: usize = 20_000;

fn configure_group<M: criterion::measurement::Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
) {
    // keep each bench fn under ~30s wall — sample_size * (warm_up + measurement)
    // + per-iter cost. 30 samples × (1s + 2s) = 90s ceiling at worst, typically
    // far less. caller can override per-fn if needed.
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
}

fn spsc_round_trip<S, R>(items: usize, send: S, mut recv: R) -> Duration
where
    S: Fn(u64) -> bool + Send + 'static,
    R: FnMut() -> Option<u64>,
{
    let started = Instant::now();
    let producer = thread::spawn(move || {
        for index in 0..items {
            while !send(index as u64) {
                thread::yield_now();
            }
        }
    });
    let mut consumed = 0;
    while consumed < items {
        if recv().is_some() {
            consumed += 1;
        } else {
            thread::yield_now();
        }
    }
    producer.join().unwrap();
    started.elapsed()
}

/// run a many-producer fan-in. `make_senders(n)` returns N independent send
/// closures — each must be Send and own its own backend handle (not share!).
/// for proxima this means a Producer per thread (each owns a lane); for
/// flume/tokio it means a Sender per thread (cheap clone).
fn mpsc_fanin<F, S, R>(
    producers: usize,
    per_producer: usize,
    make_senders: F,
    mut recv: R,
) -> Duration
where
    F: FnOnce(usize) -> Vec<S>,
    S: Fn(u64) -> bool + Send + 'static,
    R: FnMut() -> Option<u64>,
{
    let senders = make_senders(producers);
    assert_eq!(senders.len(), producers);
    let started = Instant::now();
    let mut threads = Vec::with_capacity(producers);
    for send in senders {
        threads.push(thread::spawn(move || {
            for index in 0..per_producer {
                while !send(index as u64) {
                    thread::yield_now();
                }
            }
        }));
    }
    let total = producers * per_producer;
    let mut consumed = 0;
    while consumed < total {
        if recv().is_some() {
            consumed += 1;
        } else {
            thread::yield_now();
        }
    }
    for thread in threads {
        thread.join().unwrap();
    }
    started.elapsed()
}

// design-favors: prime — one producer, one consumer. proxima's per-core SPSC
// lane design is the dominant primitive shape; flume's MPMC machinery
// (cross-thread atomics for multi-producer ordering) is unused here, so flume
// is being measured outside its design point. Std mpsc is purpose-built for
// this case and remains a fair reference. A win here is expected, not the
// load-bearing claim.
fn bench_spsc_throughput(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("inbox_spsc_throughput");
    configure_group(&mut group);
    group.throughput(Throughput::Elements(ITEMS_PER_TRIAL as u64));

    group.bench_function("proxima", |bencher| {
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let (producer, consumer) = inbox::channel::<u64>(1, CAPACITY);
                total += spsc_round_trip(
                    ITEMS_PER_TRIAL,
                    move |value| producer.try_send(value).is_ok(),
                    || consumer.try_recv().ok(),
                );
            }
            total
        });
    });

    group.bench_function("flume_unbounded", |bencher| {
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let (tx, rx) = flume::unbounded::<u64>();
                let tx_for_send = tx.clone();
                total += spsc_round_trip(
                    ITEMS_PER_TRIAL,
                    move |value| tx_for_send.send(value).is_ok(),
                    || rx.try_recv().ok(),
                );
                drop(tx);
            }
            total
        });
    });

    group.bench_function("flume_bounded", |bencher| {
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let (tx, rx) = flume::bounded::<u64>(CAPACITY);
                let tx_for_send = tx.clone();
                total += spsc_round_trip(
                    ITEMS_PER_TRIAL,
                    move |value| tx_for_send.try_send(value).is_ok(),
                    || rx.try_recv().ok(),
                );
                drop(tx);
            }
            total
        });
    });

    group.bench_function("std_sync_mpsc", |bencher| {
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let (tx, rx) = std::sync::mpsc::sync_channel::<u64>(CAPACITY);
                let tx_for_send = tx.clone();
                total += spsc_round_trip(
                    ITEMS_PER_TRIAL,
                    move |value| tx_for_send.try_send(value).is_ok(),
                    || rx.try_recv().ok(),
                );
                drop(tx);
            }
            total
        });
    });

    group.bench_function("tokio_mpsc", |bencher| {
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let (tx, mut rx) = tokio::sync::mpsc::channel::<u64>(CAPACITY);
                let tx_for_send = tx.clone();
                total += spsc_round_trip(
                    ITEMS_PER_TRIAL,
                    move |value| tx_for_send.try_send(value).is_ok(),
                    || rx.try_recv().ok(),
                );
                drop(tx);
            }
            total
        });
    });

    group.finish();
}

// design-favors: incumbent — N producers contending for a single consumer is
// flume's and tokio mpsc's design point (multi-producer bounded channel with
// cross-thread ordering). A win here is the load-bearing gate-13 claim because
// the incumbent's machinery is fully engaged. (Note: tokio mpsc is async; this
// bench's `try_send` keeps it on the sync fast path, matching proxima's
// shape — async-aware tokio paths are exercised at the runtime layer in
// h2_runtime_swap, not here.)
fn bench_mpsc_fanin(criterion: &mut Criterion, producers: usize) {
    let group_name = format!("inbox_mpsc_fanin_{producers}");
    let mut group = criterion.benchmark_group(&group_name);
    configure_group(&mut group);
    let per_producer = ITEMS_PER_TRIAL / producers;
    let total_items = producers * per_producer;
    group.throughput(Throughput::Elements(total_items as u64));

    group.bench_function("proxima", |bencher| {
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let (producer, consumer) = inbox::channel::<u64>(producers, CAPACITY);
                total += mpsc_fanin(
                    producers,
                    per_producer,
                    |n| {
                        let mut handles = Vec::with_capacity(n);
                        let first = producer;
                        for _ in 1..n {
                            handles.push(first.clone());
                        }
                        handles.push(first);
                        handles
                            .into_iter()
                            .map(|handle| {
                                let send: Box<dyn Fn(u64) -> bool + Send> =
                                    Box::new(move |value| handle.try_send(value).is_ok());
                                send
                            })
                            .collect()
                    },
                    || consumer.try_recv().ok(),
                );
            }
            total
        });
    });

    group.bench_function("flume_bounded", |bencher| {
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let (tx, rx) = flume::bounded::<u64>(CAPACITY);
                total += mpsc_fanin(
                    producers,
                    per_producer,
                    |n| {
                        (0..n)
                            .map(|_| {
                                let tx = tx.clone();
                                let send: Box<dyn Fn(u64) -> bool + Send> =
                                    Box::new(move |value| tx.try_send(value).is_ok());
                                send
                            })
                            .collect()
                    },
                    || rx.try_recv().ok(),
                );
                drop(tx);
            }
            total
        });
    });

    group.bench_function("tokio_mpsc", |bencher| {
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let (tx, mut rx) = tokio::sync::mpsc::channel::<u64>(CAPACITY);
                total += mpsc_fanin(
                    producers,
                    per_producer,
                    |n| {
                        (0..n)
                            .map(|_| {
                                let tx = tx.clone();
                                let send: Box<dyn Fn(u64) -> bool + Send> =
                                    Box::new(move |value| tx.try_send(value).is_ok());
                                send
                            })
                            .collect()
                    },
                    || rx.try_recv().ok(),
                );
                drop(tx);
            }
            total
        });
    });

    group.finish();
}

fn bench_mpsc_fanin_4(criterion: &mut Criterion) {
    bench_mpsc_fanin(criterion, 4);
}

fn bench_mpsc_fanin_8(criterion: &mut Criterion) {
    bench_mpsc_fanin(criterion, 8);
}

fn bench_mpsc_fanin_16(criterion: &mut Criterion) {
    bench_mpsc_fanin(criterion, 16);
}

fn bench_mpsc_fanin_32(criterion: &mut Criterion) {
    bench_mpsc_fanin(criterion, 32);
}

criterion_group!(
    benches,
    bench_spsc_throughput,
    bench_mpsc_fanin_4,
    bench_mpsc_fanin_8,
    bench_mpsc_fanin_16,
    bench_mpsc_fanin_32,
);
criterion_main!(benches);
