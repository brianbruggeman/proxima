#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
use std::sync::{Arc, Mutex};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use proxima_core::ring::{Ring, StaticRing};

const SIZES: &[usize] = &[1_000, 1_000_000];

// All arms measure PURE queue throughput on equal footing: allocate the
// container inline, push N, then drain to a register (`black_box`). The earlier
// proxima arm drained via `drain_into(&mut out)` into an 8 MB buffer while the
// incumbents popped to a register — an unfair apples-to-oranges comparison whose
// out-buffer-materialization cost (queue-independent — see bench_ring_decompose:
// crossbeam cliffs identically into a buffer) swamped the queue mechanics and
// produced a false 4.6x "loss" at 1M. Register-to-register, the proxima ring is
// ~2x FASTER than crossbeam and flat across the cache hierarchy.
fn proxima_ring_single_producer(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("c1_ring");

    for size in SIZES {
        group.bench_with_input(
            BenchmarkId::new("proxima_ring_single_producer", size),
            size,
            |bench, &cap| {
                bench.iter(|| {
                    let ring = Ring::<u64>::new(cap).expect("ring init failed");

                    for item in 0..cap as u64 {
                        while ring.push(std::hint::black_box(item)).is_err() {}
                    }

                    let mut total = 0usize;
                    while let Some(value) = ring.dequeue() {
                        std::hint::black_box(value);
                        total += 1;
                    }

                    std::hint::black_box(total)
                });
            },
        );
    }

    group.finish();
}

fn flume_bounded_single(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("c1_ring");

    for size in SIZES {
        group.bench_with_input(
            BenchmarkId::new("flume_bounded_single", size),
            size,
            |bench, &cap| {
                bench.iter(|| {
                    let (tx, rx) = flume::bounded::<u64>(cap);

                    for item in 0..cap as u64 {
                        let _ = tx.send(item);
                    }

                    let mut total = 0usize;
                    while let Ok(value) = rx.try_recv() {
                        std::hint::black_box(value);
                        total += 1;
                    }

                    std::hint::black_box(total)
                });
            },
        );
    }

    group.finish();
}

fn crossbeam_array_queue(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("c1_ring");

    for size in SIZES {
        group.bench_with_input(
            BenchmarkId::new("crossbeam_array_queue", size),
            size,
            |bench, &cap| {
                bench.iter(|| {
                    let queue = crossbeam_queue::ArrayQueue::<u64>::new(cap);

                    for item in 0..cap as u64 {
                        let _ = queue.push(item);
                    }

                    let mut total = 0usize;
                    while let Some(value) = queue.pop() {
                        std::hint::black_box(value);
                        total += 1;
                    }

                    std::hint::black_box(total)
                });
            },
        );
    }

    group.finish();
}

fn tokio_mpsc_bounded(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("c1_ring");

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime init failed");

    for size in SIZES {
        group.bench_with_input(
            BenchmarkId::new("tokio_mpsc_bounded", size),
            size,
            |bench, &cap| {
                bench.iter(|| {
                    rt.block_on(async {
                        let (tx, mut rx) = tokio::sync::mpsc::channel::<u64>(cap);

                        for item in 0..cap as u64 {
                            let _ = tx.send(item).await;
                        }
                        drop(tx);

                        let mut total = 0usize;
                        while let Some(value) = rx.recv().await {
                            std::hint::black_box(value);
                            total += 1;
                        }

                        total
                    })
                });
            },
        );
    }

    group.finish();
}

fn causality_index_pattern(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("c1_ring");

    for size in SIZES {
        group.bench_with_input(
            BenchmarkId::new("causality_index_pattern", size),
            size,
            |bench, &cap| {
                bench.iter(|| {
                    // minimal repro of Vec<Mutex<Vec<_>>> per-thread sharding
                    let slots: Vec<Mutex<Vec<u64>>> =
                        (0..cap).map(|_| Mutex::new(Vec::new())).collect();
                    let slots = Arc::new(slots);

                    let writer = Arc::clone(&slots);
                    for item in 0..cap as u64 {
                        let slot_idx = (item as usize) % writer.len();
                        if let Ok(mut guard) = writer[slot_idx].lock() {
                            guard.push(item);
                        }
                    }

                    let mut total = 0usize;
                    for slot in slots.iter() {
                        if let Ok(guard) = slot.lock() {
                            total += guard.len();
                        }
                    }

                    std::hint::black_box(total)
                });
            },
        );
    }

    group.finish();
}

fn proxima_static_ring_single_producer(crit: &mut Criterion) {
    // no-alloc const-cap ring (inline [Cell; 1024]); compare vs the alloc Ring
    // arm at size 1000 — does inline storage match Box on the same algorithm?
    let mut group = crit.benchmark_group("c1_ring");
    group.bench_function("proxima_static_ring_single_producer/1024", |bench| {
        bench.iter(|| {
            let ring = StaticRing::<u64, 1024>::new();
            for item in 0..1024u64 {
                while ring.push(std::hint::black_box(item)).is_err() {}
            }
            let mut total = 0usize;
            while let Some(value) = ring.dequeue() {
                std::hint::black_box(value);
                total += 1;
            }
            std::hint::black_box(total)
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    proxima_ring_single_producer,
    proxima_static_ring_single_producer,
    flume_bounded_single,
    crossbeam_array_queue,
    tokio_mpsc_bounded,
    causality_index_pattern,
);
criterion_main!(benches);
