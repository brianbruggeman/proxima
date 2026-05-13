//! bench for `runtime-prime-inbox-dynamic` — the `[floor, ceiling]` lane pool.
//!
//! **Design point:** cross-core MPSC fan-in to a per-core consumer; 80% case =
//! single-producer sticky-lane try_send/try_recv (spawn_burst hot path).
//!
//! **Named incumbents:**
//!
//! 1. `inbox-alloc` (single-producer SPSC sticky-lane) — `design-favors: incumbent`
//!    The prime 80%-case. Dynamic inbox MUST meet-or-beat on this arm or it does
//!    not land. A regression here is a gate blocker.
//!
//! 2. `inbox-alloc` (many-producer fan-in 8/16/64) — `design-favors: neutral`
//!    Validates dynamic inbox's claim of no lane-count amplification.
//!
//! 3. Memory arms (idle 1-producer; N=64 producers) — `design-favors: proxima`
//!    Dynamic inbox allocates floor × ring; incumbent allocates num_lanes × ring.
//!    Memory wins are the primary motivation for the variant.
//!
//! **Workload:** synchronous produce-then-consume (no async runtime). No sleeps.
//! Each bench iteration pushes ITEMS through the channel and consumes them.
//! Cross-thread fan-in arms use rayon scope to drive N producers simultaneously.
//!
//! **CoV guidance:** run with `--measurement-time 10 --warm-up-time 3` and
//! compare baselines. If CoV > 5%, run 3–5 criterion passes with
//! `--save-baseline` and average. Numbers in the discipline log are from
//! the run with lowest CoV.

use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use prime::core::inbox as alloc_inbox;
use prime::core::inbox_dynamic::{InboxDynamicConfig, ReleasePolicy, channel as dyn_channel};

const ITEMS: usize = 100_000;
const LANE_CAP: usize = 1024;

// ---- helpers ----

fn alloc_config(num_lanes: usize) -> (alloc_inbox::Producer<u64>, alloc_inbox::Consumer<u64>) {
    alloc_inbox::channel::<u64>(num_lanes, LANE_CAP)
}

fn dyn_config_floor_n(floor: usize, ceiling: usize) -> InboxDynamicConfig {
    InboxDynamicConfig {
        floor,
        ceiling,
        release: ReleasePolicy::Never,
        lane_capacity: LANE_CAP,
    }
}

fn dyn_config_always(floor: usize, ceiling: usize) -> InboxDynamicConfig {
    InboxDynamicConfig {
        floor,
        ceiling,
        release: ReleasePolicy::Always,
        lane_capacity: LANE_CAP,
    }
}

// ---- arm 1: single-producer sticky-lane try_send/try_recv (HOME TURF) ----
//
// design-favors: incumbent (inbox-alloc).
// meet-or-beat is a gate condition. a regression here blocks landing.

fn bench_single_producer_spsc(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("single_producer_spsc");
    group.throughput(Throughput::Elements(ITEMS as u64));

    group.bench_function("inbox_alloc_incumbent", |bench| {
        bench.iter(|| {
            let (producer, consumer) = alloc_config(1);
            let mut count = 0usize;
            let mut index = 0u64;
            while count < ITEMS {
                let batch = LANE_CAP.min(ITEMS - count);
                let mut pushed = 0;
                while pushed < batch {
                    if producer.try_send(black_box(index)).is_ok() {
                        index += 1;
                        pushed += 1;
                    } else {
                        break;
                    }
                }
                let mut drained = 0;
                while drained < pushed {
                    if consumer.try_recv().is_ok() {
                        drained += 1;
                        count += 1;
                    } else {
                        break;
                    }
                }
            }
            black_box(count)
        });
    });

    group.bench_function("inbox_dynamic_floor1", |bench| {
        let config = dyn_config_floor_n(1, 64);
        bench.iter(|| {
            let (producer, consumer) = dyn_channel::<u64>(&config);
            let mut count = 0usize;
            let mut index = 0u64;
            while count < ITEMS {
                let batch = LANE_CAP.min(ITEMS - count);
                let mut pushed = 0;
                while pushed < batch {
                    if producer.try_send(black_box(index)).is_ok() {
                        index += 1;
                        pushed += 1;
                    } else {
                        break;
                    }
                }
                let mut drained = 0;
                while drained < pushed {
                    if consumer.try_recv().is_ok() {
                        drained += 1;
                        count += 1;
                    } else {
                        break;
                    }
                }
            }
            black_box(count)
        });
    });

    group.bench_function("inbox_dynamic_floor96", |bench| {
        let config = dyn_config_floor_n(96, 128);
        bench.iter(|| {
            let (producer, consumer) = dyn_channel::<u64>(&config);
            let mut count = 0usize;
            let mut index = 0u64;
            while count < ITEMS {
                let batch = LANE_CAP.min(ITEMS - count);
                let mut pushed = 0;
                while pushed < batch {
                    if producer.try_send(black_box(index)).is_ok() {
                        index += 1;
                        pushed += 1;
                    } else {
                        break;
                    }
                }
                let mut drained = 0;
                while drained < pushed {
                    if consumer.try_recv().is_ok() {
                        drained += 1;
                        count += 1;
                    } else {
                        break;
                    }
                }
            }
            black_box(count)
        });
    });

    // same single-producer workload, release=Always (default-candidate). should
    // tie floor1 (Never): reclamation only runs on a fast-path MISS, and the
    // single-producer sticky path never misses. confirms the perf-best default.
    group.bench_function("inbox_dynamic_floor1_always", |bench| {
        let config = dyn_config_always(1, 64);
        bench.iter(|| {
            let (producer, consumer) = dyn_channel::<u64>(&config);
            let mut count = 0usize;
            let mut index = 0u64;
            while count < ITEMS {
                let batch = LANE_CAP.min(ITEMS - count);
                let mut pushed = 0;
                while pushed < batch {
                    if producer.try_send(black_box(index)).is_ok() {
                        index += 1;
                        pushed += 1;
                    } else {
                        break;
                    }
                }
                let mut drained = 0;
                while drained < pushed {
                    if consumer.try_recv().is_ok() {
                        drained += 1;
                        count += 1;
                    } else {
                        break;
                    }
                }
            }
            black_box(count)
        });
    });

    group.finish();
}

// ---- arm 2: many-producer fan-in (8 / 16 / 64 producers) ----
//
// design-favors: neutral. both variants use per-thread SPSC lanes.

fn fanin_alloc(num_producers: usize, items_per_producer: usize) {
    let (producer, consumer) = alloc_config(num_producers + 1);
    let producer = Arc::new(producer);
    let received = Arc::new(AtomicUsize::new(0));
    let total = num_producers * items_per_producer;
    let mut handles = Vec::with_capacity(num_producers);
    for thread_id in 0..num_producers {
        let prod = producer.clone();
        handles.push(thread::spawn(move || {
            for index in 0..items_per_producer {
                let value = (thread_id * items_per_producer + index) as u64;
                loop {
                    match prod.try_send_mpsc(black_box(value)) {
                        Ok(()) => break,
                        Err(alloc_inbox::SendError::Full(_)) => thread::yield_now(),
                        Err(alloc_inbox::SendError::NoLanes(_)) => thread::yield_now(),
                        Err(other) => panic!("fanin_alloc: {other}"),
                    }
                }
            }
        }));
    }
    let rec = received.clone();
    let drain_handle = thread::spawn(move || {
        while rec.load(Ordering::Acquire) < total {
            match consumer.try_recv() {
                Ok(_) => {
                    rec.fetch_add(1, Ordering::Release);
                }
                Err(alloc_inbox::TryRecvError::Empty) => thread::yield_now(),
                Err(alloc_inbox::TryRecvError::Disconnected) => break,
            }
        }
    });
    for handle in handles {
        handle.join().expect("producer join");
    }
    drain_handle.join().expect("drain join");
    assert_eq!(received.load(Ordering::Acquire), total);
}

fn fanin_dynamic(config: &InboxDynamicConfig, num_producers: usize, items_per_producer: usize) {
    let (producer, consumer) = dyn_channel::<u64>(config);
    let producer = Arc::new(producer);
    let received = Arc::new(AtomicUsize::new(0));
    let total = num_producers * items_per_producer;
    let mut handles = Vec::with_capacity(num_producers);
    for thread_id in 0..num_producers {
        let prod = producer.clone();
        handles.push(thread::spawn(move || {
            for index in 0..items_per_producer {
                let value = (thread_id * items_per_producer + index) as u64;
                loop {
                    match prod.try_send_mpsc(black_box(value)) {
                        Ok(()) => break,
                        Err(prime::core::inbox_dynamic::SendError::Full(_))
                        | Err(prime::core::inbox_dynamic::SendError::Busy(_)) => {
                            thread::yield_now();
                        }
                        Err(other) => panic!("fanin_dynamic: {other}"),
                    }
                }
            }
        }));
    }
    let rec = received.clone();
    let drain_handle = thread::spawn(move || {
        while rec.load(Ordering::Acquire) < total {
            match consumer.try_recv() {
                Ok(_) => {
                    rec.fetch_add(1, Ordering::Release);
                }
                Err(prime::core::inbox_dynamic::TryRecvError::Empty) => thread::yield_now(),
                Err(prime::core::inbox_dynamic::TryRecvError::Disconnected) => break,
            }
        }
    });
    for handle in handles {
        handle.join().expect("producer join");
    }
    drain_handle.join().expect("drain join");
    assert_eq!(received.load(Ordering::Acquire), total);
}

fn bench_fanin(criterion: &mut Criterion) {
    for num_producers in [8usize, 16, 64] {
        let items_per = 1000;
        let label = format!("p{num_producers}");

        criterion.bench_with_input(
            BenchmarkId::new("fanin_alloc_incumbent", &label),
            &num_producers,
            |bench, &num_prod| {
                bench.iter(|| fanin_alloc(num_prod, items_per));
            },
        );

        let config = dyn_config_floor_n(num_producers, num_producers * 2);
        criterion.bench_with_input(
            BenchmarkId::new("fanin_dynamic_never", &label),
            &num_producers,
            |bench, &num_prod| {
                bench.iter(|| fanin_dynamic(&config, num_prod, items_per));
            },
        );

        // Always: producers drop each iteration -> lanes abandoned -> consumer
        // reclaims. this is the worst case for the reclamation cost; if it ties
        // Never here, Always is the perf-safe default (and frees idle memory).
        let config_always = dyn_config_always(num_producers, num_producers * 2);
        criterion.bench_with_input(
            BenchmarkId::new("fanin_dynamic_always", &label),
            &num_producers,
            |bench, &num_prod| {
                bench.iter(|| fanin_dynamic(&config_always, num_prod, items_per));
            },
        );
    }
}

// ---- arm 3: MEMORY — allocated rings × ring_bytes ----
//
// design-favors: proxima (dynamic allocates floor; incumbent allocates num_lanes).
// these are not timing benches — they report the SIZE of what channel() allocates.
// we compute it manually and print it for the discipline log.

fn ring_bytes<T>(lane_capacity: usize) -> usize {
    lane_capacity * core::mem::size_of::<core::mem::MaybeUninit<T>>()
}

fn bench_memory(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("memory_channel_construction");

    // idle (1 producer): incumbent allocates num_lanes rings; dynamic allocates floor rings.
    let incumbent_lanes = 64usize;
    let bytes_per_ring = ring_bytes::<u64>(LANE_CAP);
    let incumbent_bytes = incumbent_lanes * bytes_per_ring;
    let dynamic_floor = 1usize;
    let dynamic_bytes = dynamic_floor * bytes_per_ring;
    eprintln!(
        "[memory/idle-1-producer] incumbent={incumbent_bytes}B ({incumbent_lanes} rings × {bytes_per_ring}B) | dynamic={dynamic_bytes}B ({dynamic_floor} floor rings × {bytes_per_ring}B)"
    );

    // N=64 producers: incumbent still holds num_lanes rings; dynamic holds 64 rings.
    let n_producers = 64usize;
    let dynamic_n_bytes = n_producers * bytes_per_ring;
    eprintln!(
        "[memory/N=64-producers] incumbent={incumbent_bytes}B ({incumbent_lanes} rings × {bytes_per_ring}B) | dynamic={dynamic_n_bytes}B ({n_producers} rings × {bytes_per_ring}B)"
    );

    // bench the channel() construction cost as a proxy for alloc overhead.
    group.bench_function("alloc_channel_construct", |bench| {
        bench.iter(|| {
            let (producer, consumer) = alloc_config(64);
            black_box((producer, consumer))
        });
    });

    let config_floor1 = dyn_config_floor_n(1, 128);
    group.bench_function("dynamic_channel_construct_floor1", |bench| {
        bench.iter(|| {
            let (producer, consumer) = dyn_channel::<u64>(&config_floor1);
            black_box((producer, consumer))
        });
    });

    let config_floor64 = dyn_config_floor_n(64, 128);
    group.bench_function("dynamic_channel_construct_floor64", |bench| {
        bench.iter(|| {
            let (producer, consumer) = dyn_channel::<u64>(&config_floor64);
            black_box((producer, consumer))
        });
    });

    group.finish();
}

fn bench_all(criterion: &mut Criterion) {
    bench_single_producer_spsc(criterion);
    bench_fanin(criterion);
    bench_memory(criterion);
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
