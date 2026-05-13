#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! Per-core thread-local read vs `ArcSwap` read under writer contention.
//!
//! The architectural pivot's central claim: chain dispatch reads on a
//! per-core executor should not pay cross-core synchronization cost.
//! This bench measures the read-side cost of both shapes:
//!
//! - **ArcSwap path**: `Arc<ArcSwap<T>>` shared across N writer threads
//!   doing concurrent `store` operations. The reader thread loads on
//!   every iteration. Each load is a synchronized atomic + reference
//!   bump; under heavy writer contention, the cache line bounces.
//!
//! - **Per-core path**: `thread_local!` `RefCell<T>` accessed from the
//!   same thread that owns it. No atomics, no cross-core traffic. A
//!   parallel "writer" loop running on other threads is invisible to
//!   the reader.
//!
//! The number we want to see: per-core read latency stays flat as
//! writer contention rises; ArcSwap read latency grows. The gap
//! justifies the per-core substrate.

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use arc_swap::ArcSwap;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dashmap::DashMap;

#[derive(Clone)]
struct ChainConfig {
    upstream_id: u32,
    sample_bit: u64,
}

impl ChainConfig {
    fn baseline() -> Self {
        Self {
            upstream_id: 7,
            sample_bit: 0,
        }
    }
}

thread_local! {
    static LOCAL_CONFIG: RefCell<ChainConfig> = RefCell::new(ChainConfig::baseline());
}

fn arcswap_read_under_contention(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("arcswap_read_under_writer_contention");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));

    for writer_threads in [0_usize, 1, 4, 16] {
        let shared = Arc::new(ArcSwap::from_pointee(ChainConfig::baseline()));
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();
        for index in 0..writer_threads {
            let shared_for_writer = shared.clone();
            let stop_for_writer = stop.clone();
            handles.push(thread::spawn(move || {
                let mut tick: u64 = index as u64;
                while !stop_for_writer.load(Ordering::Relaxed) {
                    shared_for_writer.store(Arc::new(ChainConfig {
                        upstream_id: 7,
                        sample_bit: tick,
                    }));
                    tick = tick.wrapping_add(1);
                }
            }));
        }
        group.bench_with_input(
            BenchmarkId::from_parameter(writer_threads),
            &shared,
            |bencher, shared| {
                bencher.iter(|| {
                    let snapshot = shared.load();
                    std::hint::black_box((snapshot.upstream_id, snapshot.sample_bit));
                });
            },
        );
        stop.store(true, Ordering::Relaxed);
        for handle in handles {
            handle.join().expect("writer thread joined");
        }
    }
    group.finish();
}

fn per_core_read_under_writer_traffic(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("per_core_read_under_writer_traffic");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));

    // Per-core reads should be invariant to other-thread activity. We
    // spin parallel "noise" threads that simulate other cores doing
    // work; the reader's thread_local is untouched by them. The bench
    // confirms read latency stays flat across noise levels.
    for noise_threads in [0_usize, 1, 4, 16] {
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();
        for _ in 0..noise_threads {
            let stop_for_noise = stop.clone();
            handles.push(thread::spawn(move || {
                let mut tick: u64 = 0;
                while !stop_for_noise.load(Ordering::Relaxed) {
                    tick = tick.wrapping_add(1);
                    std::hint::black_box(tick);
                }
            }));
        }
        group.bench_with_input(
            BenchmarkId::from_parameter(noise_threads),
            &noise_threads,
            |bencher, _| {
                bencher.iter(|| {
                    LOCAL_CONFIG.with(|cell| {
                        let config = cell.borrow();
                        std::hint::black_box((config.upstream_id, config.sample_bit));
                    });
                });
            },
        );
        stop.store(true, Ordering::Relaxed);
        for handle in handles {
            handle.join().expect("noise thread joined");
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Registry-shaped benches: SwapRegistry / PipeFactoryRegistry / ListenRegistry
// access pattern is "map of string-keyed handles, look up by key per dispatch."
// All three primitives below answer the same question — given an Arc<T> map
// keyed by &str, how fast can we look up an entry while writers concurrently
// register/replace? We exercise three contention regimes:
//
// - read_only_no_writers — baseline lookup cost
// - read_under_writer_contention — typical: many reads, occasional swap
// - write_under_reader_contention — mid-life churn: many concurrent installs
//
// Comparators:
// - per_core_thread_local: each thread has its own HashMap; writes broadcast
//   to every thread_local (simulated by N parallel hashmap updates)
// - arcswap_map: ArcSwap<HashMap<String, Arc<T>>>, copy-on-write
// - dashmap: sharded RwLock map
// ---------------------------------------------------------------------------

const REGISTRY_KEYS: &[&str] = &[
    "auth",
    "cache",
    "rate-limit",
    "compress",
    "retry",
    "isolate",
    "router",
    "echo",
    "transform",
    "log",
    "trace",
    "metrics",
    "tls",
    "swap",
    "validate",
    "fan-out",
];

thread_local! {
    static LOCAL_REGISTRY: RefCell<std::collections::HashMap<String, Arc<ChainConfig>>> =
        RefCell::new(std::collections::HashMap::new());
}

fn seed_local_registry() {
    LOCAL_REGISTRY.with(|cell| {
        let mut map = cell.borrow_mut();
        for &key in REGISTRY_KEYS {
            map.insert(key.to_string(), Arc::new(ChainConfig::baseline()));
        }
    });
}

fn registry_read_only(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("registry_read_only_no_writers");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));

    // per-core thread-local map
    seed_local_registry();
    group.bench_function("per_core_thread_local", |bencher| {
        let mut index: usize = 0;
        bencher.iter(|| {
            let key = REGISTRY_KEYS[index % REGISTRY_KEYS.len()];
            index = index.wrapping_add(1);
            LOCAL_REGISTRY.with(|cell| {
                let map = cell.borrow();
                let entry = map.get(key);
                std::hint::black_box(entry.map(|cfg| cfg.upstream_id));
            });
        });
    });

    // ArcSwap<HashMap>
    let arcswap_map: Arc<ArcSwap<std::collections::HashMap<String, Arc<ChainConfig>>>> = {
        let mut initial = std::collections::HashMap::new();
        for &key in REGISTRY_KEYS {
            initial.insert(key.to_string(), Arc::new(ChainConfig::baseline()));
        }
        Arc::new(ArcSwap::from_pointee(initial))
    };
    group.bench_function("arcswap_map", |bencher| {
        let mut index: usize = 0;
        bencher.iter(|| {
            let key = REGISTRY_KEYS[index % REGISTRY_KEYS.len()];
            index = index.wrapping_add(1);
            let snapshot = arcswap_map.load();
            let entry = snapshot.get(key);
            std::hint::black_box(entry.map(|cfg| cfg.upstream_id));
        });
    });

    // DashMap
    let dashmap: Arc<DashMap<String, Arc<ChainConfig>>> = Arc::new(DashMap::new());
    for &key in REGISTRY_KEYS {
        dashmap.insert(key.to_string(), Arc::new(ChainConfig::baseline()));
    }
    group.bench_function("dashmap", |bencher| {
        let mut index: usize = 0;
        bencher.iter(|| {
            let key = REGISTRY_KEYS[index % REGISTRY_KEYS.len()];
            index = index.wrapping_add(1);
            let entry = dashmap.get(key);
            std::hint::black_box(entry.as_ref().map(|cfg| cfg.upstream_id));
        });
    });

    group.finish();
}

fn registry_read_under_writer_contention(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("registry_read_under_writer_contention");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));

    for writer_threads in [1_usize, 4, 16] {
        // ArcSwap<HashMap>
        let arcswap_map: Arc<ArcSwap<std::collections::HashMap<String, Arc<ChainConfig>>>> = {
            let mut initial = std::collections::HashMap::new();
            for &key in REGISTRY_KEYS {
                initial.insert(key.to_string(), Arc::new(ChainConfig::baseline()));
            }
            Arc::new(ArcSwap::from_pointee(initial))
        };
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();
        for thread_index in 0..writer_threads {
            let shared = arcswap_map.clone();
            let stop = stop.clone();
            handles.push(thread::spawn(move || {
                let mut tick: u64 = thread_index as u64;
                while !stop.load(Ordering::Relaxed) {
                    let current = shared.load_full();
                    let mut next: std::collections::HashMap<String, Arc<ChainConfig>> =
                        (*current).clone();
                    next.insert(
                        format!("dyn-{}", tick % 4),
                        Arc::new(ChainConfig {
                            upstream_id: 7,
                            sample_bit: tick,
                        }),
                    );
                    shared.store(Arc::new(next));
                    tick = tick.wrapping_add(1);
                }
            }));
        }
        group.bench_with_input(
            BenchmarkId::new("arcswap_map", writer_threads),
            &arcswap_map,
            |bencher, shared| {
                let mut index: usize = 0;
                bencher.iter(|| {
                    let key = REGISTRY_KEYS[index % REGISTRY_KEYS.len()];
                    index = index.wrapping_add(1);
                    let snapshot = shared.load();
                    let entry = snapshot.get(key);
                    std::hint::black_box(entry.map(|cfg| cfg.upstream_id));
                });
            },
        );
        stop.store(true, Ordering::Relaxed);
        for handle in handles {
            handle.join().expect("arcswap writer joined");
        }

        // DashMap
        let dashmap: Arc<DashMap<String, Arc<ChainConfig>>> = Arc::new(DashMap::new());
        for &key in REGISTRY_KEYS {
            dashmap.insert(key.to_string(), Arc::new(ChainConfig::baseline()));
        }
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();
        for thread_index in 0..writer_threads {
            let shared = dashmap.clone();
            let stop = stop.clone();
            handles.push(thread::spawn(move || {
                let mut tick: u64 = thread_index as u64;
                while !stop.load(Ordering::Relaxed) {
                    shared.insert(
                        format!("dyn-{}", tick % 4),
                        Arc::new(ChainConfig {
                            upstream_id: 7,
                            sample_bit: tick,
                        }),
                    );
                    tick = tick.wrapping_add(1);
                }
            }));
        }
        group.bench_with_input(
            BenchmarkId::new("dashmap", writer_threads),
            &dashmap,
            |bencher, shared| {
                let mut index: usize = 0;
                bencher.iter(|| {
                    let key = REGISTRY_KEYS[index % REGISTRY_KEYS.len()];
                    index = index.wrapping_add(1);
                    let entry = shared.get(key);
                    std::hint::black_box(entry.as_ref().map(|cfg| cfg.upstream_id));
                });
            },
        );
        stop.store(true, Ordering::Relaxed);
        for handle in handles {
            handle.join().expect("dashmap writer joined");
        }
    }

    // per-core thread-local — reads are invariant to writer count by
    // construction (writers don't touch the reader's thread-local). We
    // include one parameterization to make this explicit in the report.
    seed_local_registry();
    let stop = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();
    for _ in 0..16 {
        let stop = stop.clone();
        handles.push(thread::spawn(move || {
            let mut tick: u64 = 0;
            while !stop.load(Ordering::Relaxed) {
                tick = tick.wrapping_add(1);
                std::hint::black_box(tick);
            }
        }));
    }
    group.bench_with_input(
        BenchmarkId::new("per_core_thread_local", 16),
        &(),
        |bencher, _| {
            let mut index: usize = 0;
            bencher.iter(|| {
                let key = REGISTRY_KEYS[index % REGISTRY_KEYS.len()];
                index = index.wrapping_add(1);
                LOCAL_REGISTRY.with(|cell| {
                    let map = cell.borrow();
                    let entry = map.get(key);
                    std::hint::black_box(entry.map(|cfg| cfg.upstream_id));
                });
            });
        },
    );
    stop.store(true, Ordering::Relaxed);
    for handle in handles {
        handle.join().expect("noise thread joined");
    }

    group.finish();
}

fn registry_write_under_reader_contention(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("registry_write_under_reader_contention");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));

    for reader_threads in [1_usize, 4, 16] {
        // ArcSwap<HashMap> — writer measured under N concurrent readers
        let arcswap_map: Arc<ArcSwap<std::collections::HashMap<String, Arc<ChainConfig>>>> = {
            let mut initial = std::collections::HashMap::new();
            for &key in REGISTRY_KEYS {
                initial.insert(key.to_string(), Arc::new(ChainConfig::baseline()));
            }
            Arc::new(ArcSwap::from_pointee(initial))
        };
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();
        for thread_index in 0..reader_threads {
            let shared = arcswap_map.clone();
            let stop = stop.clone();
            handles.push(thread::spawn(move || {
                let mut tick: usize = thread_index;
                while !stop.load(Ordering::Relaxed) {
                    let key = REGISTRY_KEYS[tick % REGISTRY_KEYS.len()];
                    let snapshot = shared.load();
                    let entry = snapshot.get(key);
                    std::hint::black_box(entry.map(|cfg| cfg.upstream_id));
                    tick = tick.wrapping_add(1);
                }
            }));
        }
        group.bench_with_input(
            BenchmarkId::new("arcswap_map", reader_threads),
            &arcswap_map,
            |bencher, shared| {
                let mut tick: u64 = 0;
                bencher.iter(|| {
                    let current = shared.load_full();
                    let mut next: std::collections::HashMap<String, Arc<ChainConfig>> =
                        (*current).clone();
                    next.insert(
                        format!("dyn-{}", tick % 4),
                        Arc::new(ChainConfig {
                            upstream_id: 7,
                            sample_bit: tick,
                        }),
                    );
                    shared.store(Arc::new(next));
                    tick = tick.wrapping_add(1);
                });
            },
        );
        stop.store(true, Ordering::Relaxed);
        for handle in handles {
            handle.join().expect("arcswap reader joined");
        }

        // DashMap — writer under N concurrent readers
        let dashmap: Arc<DashMap<String, Arc<ChainConfig>>> = Arc::new(DashMap::new());
        for &key in REGISTRY_KEYS {
            dashmap.insert(key.to_string(), Arc::new(ChainConfig::baseline()));
        }
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();
        for thread_index in 0..reader_threads {
            let shared = dashmap.clone();
            let stop = stop.clone();
            handles.push(thread::spawn(move || {
                let mut tick: usize = thread_index;
                while !stop.load(Ordering::Relaxed) {
                    let key = REGISTRY_KEYS[tick % REGISTRY_KEYS.len()];
                    let entry = shared.get(key);
                    std::hint::black_box(entry.as_ref().map(|cfg| cfg.upstream_id));
                    tick = tick.wrapping_add(1);
                }
            }));
        }
        group.bench_with_input(
            BenchmarkId::new("dashmap", reader_threads),
            &dashmap,
            |bencher, shared| {
                let mut tick: u64 = 0;
                bencher.iter(|| {
                    shared.insert(
                        format!("dyn-{}", tick % 4),
                        Arc::new(ChainConfig {
                            upstream_id: 7,
                            sample_bit: tick,
                        }),
                    );
                    tick = tick.wrapping_add(1);
                });
            },
        );
        stop.store(true, Ordering::Relaxed);
        for handle in handles {
            handle.join().expect("dashmap reader joined");
        }
    }

    group.finish();
}

fn dashmap_read_under_contention(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("dashmap_read_under_writer_contention");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));

    // DashMap is the sharded-RwLock alternative we audited in Stage 3b.
    // ArcSwap loads atomically and replaces the whole map on write; DashMap
    // selects a shard by hash and acquires a per-shard read/write lock.
    // For read-mostly map-shaped data the two should be close;
    // write-contended workloads should favor DashMap (shard isolation,
    // no full-map clone); broadcast-replace workloads should favor
    // ArcSwap. This bench gives us the empirical numbers proxima
    // didn't have when the original "RwLock<HashMap> → ArcSwap<HashMap>"
    // prescription was written.
    let key = "chain";
    for writer_threads in [0_usize, 1, 4, 16] {
        let shared: Arc<DashMap<&'static str, ChainConfig>> = Arc::new(DashMap::new());
        shared.insert(key, ChainConfig::baseline());
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();
        for index in 0..writer_threads {
            let shared_for_writer = shared.clone();
            let stop_for_writer = stop.clone();
            handles.push(thread::spawn(move || {
                let mut tick: u64 = index as u64;
                while !stop_for_writer.load(Ordering::Relaxed) {
                    shared_for_writer.insert(
                        key,
                        ChainConfig {
                            upstream_id: 7,
                            sample_bit: tick,
                        },
                    );
                    tick = tick.wrapping_add(1);
                }
            }));
        }
        group.bench_with_input(
            BenchmarkId::from_parameter(writer_threads),
            &shared,
            |bencher, shared| {
                bencher.iter(|| {
                    if let Some(entry) = shared.get(key) {
                        std::hint::black_box((entry.upstream_id, entry.sample_bit));
                    }
                });
            },
        );
        stop.store(true, Ordering::Relaxed);
        for handle in handles {
            handle.join().expect("writer thread joined");
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    arcswap_read_under_contention,
    per_core_read_under_writer_traffic,
    dashmap_read_under_contention,
    registry_read_only,
    registry_read_under_writer_contention,
    registry_write_under_reader_contention,
);
criterion_main!(benches);
