//! BucketTable vs the dashmap incumbent on the rate-limit workload.
//!
//! The hot path of a rate limiter is read-dominated: every request does one
//! get-or-insert of an already-present key. The arms below isolate that
//! (single-thread + contended) plus the cold insert, a realistic mixed mix,
//! and the O(CAP) maintenance scans. dashmap is the incumbent (P14); both arms
//! store `Arc<V>` for a fair handle comparison. Measured on an M1 Max / 10
//! cores — relative ratios are valid here; absolute contended numbers want the
//! host-b bench host.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dashmap::DashMap;
use proxima_primitives::pipe::bucket_table::BucketTable;

// bucket-sized value, mirroring AtomicBucket (5 words). fields are padding to
// match the real footprint; only last_access is read by the eviction metric.
#[allow(dead_code)]
struct Bucket {
    tokens: AtomicU64,
    last_refill: AtomicU64,
    last_access: AtomicU64,
    capacity: u64,
    refill: u64,
}

impl Bucket {
    fn new() -> Self {
        Self {
            tokens: AtomicU64::new(1_000_000),
            last_refill: AtomicU64::new(0),
            last_access: AtomicU64::new(0),
            capacity: 1_000_000,
            refill: 1_000_000,
        }
    }
}

#[inline]
fn access(bucket: &Bucket) -> u64 {
    bucket.last_access.load(Relaxed)
}

// realistic rate-limit key shapes (tenant ids + method/path), P9.
fn keys(count: usize) -> Vec<Vec<u8>> {
    (0..count)
        .map(|index| {
            if index % 3 == 0 {
                format!("tenant-{index:06}").into_bytes()
            } else if index % 3 == 1 {
                format!("GET /v1/items/{index}").into_bytes()
            } else {
                format!("apikey-{:016x}", index as u64 * 0x9e37_79b9).into_bytes()
            }
        })
        .collect()
}

fn fill_bucket_table(size: usize) -> BucketTable<Bucket> {
    let table = BucketTable::with_max_keys(size * 2);
    for key in keys(size) {
        let _ = table.get_or_insert(&key, Bucket::new);
    }
    table
}

fn fill_dashmap(size: usize) -> DashMap<Vec<u8>, Arc<Bucket>> {
    let map = DashMap::new();
    for key in keys(size) {
        map.entry(key).or_insert_with(|| Arc::new(Bucket::new()));
    }
    map
}

#[inline]
fn dash_get_or_insert(map: &DashMap<Vec<u8>, Arc<Bucket>>, key: &[u8]) -> Arc<Bucket> {
    if let Some(found) = map.get(key) {
        return found.value().clone();
    }
    map.entry(key.to_vec())
        .or_insert_with(|| Arc::new(Bucket::new()))
        .value()
        .clone()
}

// ── arm 1: read-hot, single thread (the per-request cost) ──────────────────────
fn read_hot_single(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("read_hot_single");
    for size in [16usize, 256, 4096, 100_000] {
        let probe = keys(size);
        let table = fill_bucket_table(size);
        let map = fill_dashmap(size);
        group.bench_with_input(
            BenchmarkId::new("bucket_table", size),
            &size,
            |bencher, _| {
                let mut next = 0usize;
                bencher.iter(|| {
                    let key = &probe[next % probe.len()];
                    next = next.wrapping_add(1);
                    black_box(table.get_or_insert(black_box(key), Bucket::new))
                });
            },
        );
        group.bench_with_input(BenchmarkId::new("dashmap", size), &size, |bencher, _| {
            let mut next = 0usize;
            bencher.iter(|| {
                let key = &probe[next % probe.len()];
                next = next.wrapping_add(1);
                black_box(dash_get_or_insert(black_box(&map), black_box(key)))
            });
        });
    }
    group.finish();
}

// ── arm 2: read-hot, contended (lock-free read vs sharded RwLock) ──────────────
fn read_hot_contended(criterion: &mut Criterion) {
    const SIZE: usize = 4096;
    const OPS_PER_THREAD: usize = 50_000;
    let mut group = criterion.benchmark_group("read_hot_contended");
    for threads in [2usize, 4, 8] {
        group.throughput(Throughput::Elements((threads * OPS_PER_THREAD) as u64));
        let table = Arc::new(fill_bucket_table(SIZE));
        group.bench_with_input(
            BenchmarkId::new("bucket_table", threads),
            &threads,
            |bencher, &threads| {
                bencher.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let barrier = Arc::new(Barrier::new(threads));
                        let start = Instant::now();
                        std::thread::scope(|scope| {
                            for tid in 0..threads {
                                let table = table.clone();
                                let barrier = barrier.clone();
                                scope.spawn(move || {
                                    let probe = keys(SIZE);
                                    barrier.wait();
                                    let mut next = tid * 97;
                                    for _ in 0..OPS_PER_THREAD {
                                        let key = &probe[next % probe.len()];
                                        next = next.wrapping_add(1);
                                        black_box(table.get_or_insert(key, Bucket::new));
                                    }
                                });
                            }
                        });
                        total += start.elapsed();
                    }
                    total
                });
            },
        );
        let map = Arc::new(fill_dashmap(SIZE));
        group.bench_with_input(
            BenchmarkId::new("dashmap", threads),
            &threads,
            |bencher, &threads| {
                bencher.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let barrier = Arc::new(Barrier::new(threads));
                        let start = Instant::now();
                        std::thread::scope(|scope| {
                            for tid in 0..threads {
                                let map = map.clone();
                                let barrier = barrier.clone();
                                scope.spawn(move || {
                                    let probe = keys(SIZE);
                                    barrier.wait();
                                    let mut next = tid * 97;
                                    for _ in 0..OPS_PER_THREAD {
                                        let key = &probe[next % probe.len()];
                                        next = next.wrapping_add(1);
                                        black_box(dash_get_or_insert(&map, key));
                                    }
                                });
                            }
                        });
                        total += start.elapsed();
                    }
                    total
                });
            },
        );
    }
    group.finish();
}

// ── arm 3: insert cold (BucketTable allocs Arc + Claiming CAS; expected to lose) ─
fn insert_cold(criterion: &mut Criterion) {
    const SIZE: usize = 4096;
    let probe = keys(SIZE);
    let mut group = criterion.benchmark_group("insert_cold");
    group.bench_function("bucket_table", |bencher| {
        bencher.iter_batched(
            || BucketTable::<Bucket>::with_max_keys(SIZE * 2),
            |table| {
                for key in &probe {
                    black_box(table.get_or_insert(key, Bucket::new));
                }
                table
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("dashmap", |bencher| {
        bencher.iter_batched(
            DashMap::<Vec<u8>, Arc<Bucket>>::new,
            |map| {
                for key in &probe {
                    map.entry(key.clone())
                        .or_insert_with(|| Arc::new(Bucket::new()));
                }
                map
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

// ── arm 4: maintenance scans (O(CAP) array vs dashmap iter) ─────────────────────
fn maintenance(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("maintenance");
    for size in [4096usize, 100_000] {
        let table = fill_bucket_table(size);
        group.bench_with_input(
            BenchmarkId::new("bucket_table_evict_lru", size),
            &size,
            |bencher, _| {
                bencher.iter(|| table.evict_one_lru(black_box(access)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("bucket_table_sweep", size),
            &size,
            |bencher, _| {
                bencher.iter(|| table.sweep_idle(black_box(0), black_box(0), access));
            },
        );
        let map = fill_dashmap(size);
        group.bench_with_input(
            BenchmarkId::new("dashmap_min_scan", size),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    let victim = map
                        .iter()
                        .min_by_key(|entry| entry.value().last_access.load(Relaxed))
                        .map(|entry| entry.key().clone());
                    black_box(victim)
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    read_hot_single,
    read_hot_contended,
    insert_cold,
    maintenance
);
criterion_main!(benches);
