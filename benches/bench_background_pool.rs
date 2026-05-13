//! micro-bench for ProximaBackgroundPool vs RayonBackgroundPool vs
//! `tokio::task::spawn_blocking`.
//!
//! incumbents (versions pinned in Cargo.toml):
//!   - rayon 1.12 — data-parallel work-stealing thread pool; design point is
//!     fork-join + recursive split (par_iter) + work-stealing under imbalance
//!   - tokio::task::spawn_blocking 1.x — fixed-size blocking task pool for
//!     offloading sync work from the async runtime
//!
//! groups (and design-favors per workload):
//!   - bg_pool_tiny_jobs         design-favors: neither (dispatch microbench)
//!   - bg_pool_cpu_imbalanced    design-favors: incumbent (10% heavy / 90% light — rayon work-stealing design point)
//!   - bg_pool_latency           design-favors: neither (single-shot latency floor)
//!   - bg_pool_multi_producer    design-favors: incumbent (4 producer threads — Injector contention)
//!   - bg_pool_fork_join         design-favors: incumbent (100k elements, par_iter / par_reduce — rayon's primary design point)
//!
//! required-features: runtime-prime-bgpool-rayon, rayon, runtime-tokio.

#![cfg(all(
    feature = "runtime-prime-bgpool",
    feature = "rayon",
    feature = "runtime-tokio"
))]
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
use std::time::{Duration, Instant};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima::runtime::prime::os::background::ProximaBackgroundPool;
use proxima::runtime::{BackgroundPool, RayonBackgroundPool};

const JOBS_PER_TRIAL: usize = 1000;

fn configure_group<M: criterion::measurement::Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
) {
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
}

fn make_work(
    index: u32,
) -> Box<dyn FnOnce() -> Result<Box<dyn std::any::Any + Send>, proxima::ProximaError> + Send> {
    Box::new(move || Ok(Box::new(index * 2) as Box<dyn std::any::Any + Send>))
}

// design-favors: neither — tiny jobs (one mul) measure pure dispatch overhead.
// Both rayon and proxima collapse to "push a closure on the queue, pop, run"
// with no opportunity for work-stealing or load-balancing to matter. Useful as
// a noise floor; not a verdict for either side.
fn bench_tiny_jobs_throughput(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("bg_pool_tiny_jobs");
    configure_group(&mut group);
    group.throughput(Throughput::Elements(JOBS_PER_TRIAL as u64));

    // proxima via the trait method (slow path) — keeps API-level Box<dyn FnOnce>
    // for parity with the dyn-dispatch case.
    group.bench_function("proxima_dyn", |bencher| {
        let pool = Arc::new(ProximaBackgroundPool::with_threads(4).expect("build pool"));
        let pool_dyn: Arc<dyn BackgroundPool> = pool.clone();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            runtime.block_on(async {
                for _ in 0..iters {
                    let started = Instant::now();
                    let mut handles = Vec::with_capacity(JOBS_PER_TRIAL);
                    for index in 0..JOBS_PER_TRIAL as u32 {
                        handles.push(BackgroundPool::spawn(&*pool_dyn, make_work(index)));
                    }
                    for handle in handles {
                        let _ = handle.await;
                    }
                    total += started.elapsed();
                }
            });
            total
        });
    });

    // proxima via the typed fast path — skips the API-level Box<dyn FnOnce>.
    // this is the "trait refactor" win: callers with a concrete type get a
    // direct generic call with no API-level allocation.
    group.bench_function("proxima_typed", |bencher| {
        let pool = Arc::new(ProximaBackgroundPool::with_threads(4).expect("build pool"));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            runtime.block_on(async {
                for _ in 0..iters {
                    let started = Instant::now();
                    let mut handles = Vec::with_capacity(JOBS_PER_TRIAL);
                    for index in 0..JOBS_PER_TRIAL as u32 {
                        handles
                            .push(pool.spawn(move || Ok::<u32, proxima::ProximaError>(index * 2)));
                    }
                    for handle in handles {
                        let _ = handle.await;
                    }
                    total += started.elapsed();
                }
            });
            total
        });
    });

    group.bench_function("rayon", |bencher| {
        let pool = Arc::new(RayonBackgroundPool::with_threads(4).expect("build pool"));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            runtime.block_on(async {
                for _ in 0..iters {
                    let started = Instant::now();
                    let mut handles = Vec::with_capacity(JOBS_PER_TRIAL);
                    for index in 0..JOBS_PER_TRIAL as u32 {
                        handles.push(pool.spawn(make_work(index)));
                    }
                    for handle in handles {
                        let _ = handle.await;
                    }
                    total += started.elapsed();
                }
            });
            total
        });
    });

    // RayonBackgroundPool via the typed fast-path (runtime-prime-bgpool-rayon).
    // no API-level Box<dyn FnOnce>; closure pushed directly into rayon's deque.
    #[cfg(feature = "runtime-prime-bgpool-rayon")]
    group.bench_function("proxima_rayon_backed", |bencher| {
        let pool = Arc::new(RayonBackgroundPool::with_threads(4).expect("build pool"));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            runtime.block_on(async {
                for _ in 0..iters {
                    let started = Instant::now();
                    let mut handles = Vec::with_capacity(JOBS_PER_TRIAL);
                    for index in 0..JOBS_PER_TRIAL as u32 {
                        handles
                            .push(pool.spawn(move || Ok::<u32, proxima::ProximaError>(index * 2)));
                    }
                    for handle in handles {
                        let _ = handle.await;
                    }
                    total += started.elapsed();
                }
            });
            total
        });
    });

    // RayonBackgroundPool via the dyn-compatible trait path under the new flag.
    // same dyn allocation overhead as the existing rayon arm; kept for symmetry
    // with the proxima_typed/proxima_dyn split.
    #[cfg(feature = "runtime-prime-bgpool-rayon")]
    group.bench_function("proxima_rayon_backed_dyn", |bencher| {
        let pool = Arc::new(RayonBackgroundPool::with_threads(4).expect("build pool"));
        let pool_dyn: Arc<dyn BackgroundPool> = pool.clone();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            runtime.block_on(async {
                for _ in 0..iters {
                    let started = Instant::now();
                    let mut handles = Vec::with_capacity(JOBS_PER_TRIAL);
                    for index in 0..JOBS_PER_TRIAL as u32 {
                        handles.push(BackgroundPool::spawn(&*pool_dyn, make_work(index)));
                    }
                    for handle in handles {
                        let _ = handle.await;
                    }
                    total += started.elapsed();
                }
            });
            total
        });
    });

    group.bench_function("tokio_spawn_blocking", |bencher| {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .max_blocking_threads(4)
            .build()
            .unwrap();
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            runtime.block_on(async {
                for _ in 0..iters {
                    let started = Instant::now();
                    let mut handles = Vec::with_capacity(JOBS_PER_TRIAL);
                    for index in 0..JOBS_PER_TRIAL as u32 {
                        handles.push(tokio::task::spawn_blocking(move || index * 2));
                    }
                    for handle in handles {
                        let _ = handle.await;
                    }
                    total += started.elapsed();
                }
            });
            total
        });
    });

    group.finish();
}

// CPU-burn helper. busy-wait `cycles` iterations of a sum that the optimizer
// cannot eliminate (black_box guards). returns the accumulated value to
// prevent dead-code elimination of the call site.
#[inline(never)]
fn cpu_burn(cycles: u64) -> u64 {
    let mut accumulator: u64 = 0;
    for index in 0..cycles {
        accumulator = std::hint::black_box(accumulator.wrapping_add(index ^ 0x9E37_79B9_7F4A_7C15));
    }
    std::hint::black_box(accumulator)
}

// per-job cycle counts. on M1 P-core each cpu_burn iteration is ~1-2 ns
// (XOR + add + black_box barrier). previous values (HEAVY=20k) gave
// ~20-40 µs heavy jobs, dwarfed by dispatch overhead — load-balancing
// strength couldn't show. new values: HEAVY=2M ≈ 2-4 ms per heavy job;
// LIGHT=5k ≈ 5-10 µs per light job. with 10 heavy + 90 light jobs:
// total heavy work ≈ 20-40 ms; total light ≈ 0.5-1 ms; perfect 4-way
// parallel ≈ 5-10 ms; static-dispatch imbalance can stretch that to
// 6-12 ms depending on which worker grabs which heavy job. work-stealing
// rebalances; static dispatch (our model) doesn't.
const CYCLES_LIGHT: u64 = 5_000;
const CYCLES_HEAVY: u64 = 2_000_000;
const IMBALANCED_JOBS: usize = 100;

// design-favors: incumbent — 10% heavy / 90% light is rayon's bread and
// butter: work-stealing rebalances heavy jobs across idle workers. Static
// dispatch (proxima's model) cannot. Meet-or-beat here is the load-bearing
// gate-13 claim.
fn bench_cpu_imbalanced(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("bg_pool_cpu_imbalanced");
    configure_group(&mut group);
    group.throughput(Throughput::Elements(IMBALANCED_JOBS as u64));

    // 10% heavy, 90% light. workers should distribute heavy jobs across
    // themselves; a static-dispatch pool concentrates them on whichever
    // workers stole them first and stalls. work-stealing rebalances.
    let cycles_for = |index: usize| -> u64 {
        if index % 10 == 0 {
            CYCLES_HEAVY
        } else {
            CYCLES_LIGHT
        }
    };

    group.bench_function("proxima_dyn", |bencher| {
        let pool = Arc::new(ProximaBackgroundPool::with_threads(4).expect("build pool"));
        let pool_dyn: Arc<dyn BackgroundPool> = pool.clone();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            runtime.block_on(async {
                for _ in 0..iters {
                    let started = Instant::now();
                    let mut handles = Vec::with_capacity(IMBALANCED_JOBS);
                    for index in 0..IMBALANCED_JOBS {
                        let cycles = cycles_for(index);
                        let job: Box<
                            dyn FnOnce() -> Result<
                                    Box<dyn std::any::Any + Send>,
                                    proxima::ProximaError,
                                > + Send,
                        > = Box::new(move || {
                            Ok(Box::new(cpu_burn(cycles)) as Box<dyn std::any::Any + Send>)
                        });
                        handles.push(BackgroundPool::spawn(&*pool_dyn, job));
                    }
                    for handle in handles {
                        let _ = handle.await;
                    }
                    total += started.elapsed();
                }
            });
            total
        });
    });

    group.bench_function("proxima_typed", |bencher| {
        let pool = Arc::new(ProximaBackgroundPool::with_threads(4).expect("build pool"));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            runtime.block_on(async {
                for _ in 0..iters {
                    let started = Instant::now();
                    let mut handles = Vec::with_capacity(IMBALANCED_JOBS);
                    for index in 0..IMBALANCED_JOBS {
                        let cycles = cycles_for(index);
                        handles.push(
                            pool.spawn(move || Ok::<u64, proxima::ProximaError>(cpu_burn(cycles))),
                        );
                    }
                    for handle in handles {
                        let _ = handle.await;
                    }
                    total += started.elapsed();
                }
            });
            total
        });
    });

    group.bench_function("rayon", |bencher| {
        let pool = Arc::new(RayonBackgroundPool::with_threads(4).expect("build pool"));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            runtime.block_on(async {
                for _ in 0..iters {
                    let started = Instant::now();
                    let mut handles = Vec::with_capacity(IMBALANCED_JOBS);
                    for index in 0..IMBALANCED_JOBS {
                        let cycles = cycles_for(index);
                        let job: Box<
                            dyn FnOnce() -> Result<
                                    Box<dyn std::any::Any + Send>,
                                    proxima::ProximaError,
                                > + Send,
                        > = Box::new(move || {
                            Ok(Box::new(cpu_burn(cycles)) as Box<dyn std::any::Any + Send>)
                        });
                        handles.push(pool.spawn(job));
                    }
                    for handle in handles {
                        let _ = handle.await;
                    }
                    total += started.elapsed();
                }
            });
            total
        });
    });

    #[cfg(feature = "runtime-prime-bgpool-rayon")]
    group.bench_function("proxima_rayon_backed", |bencher| {
        let pool = Arc::new(RayonBackgroundPool::with_threads(4).expect("build pool"));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            runtime.block_on(async {
                for _ in 0..iters {
                    let started = Instant::now();
                    let mut handles = Vec::with_capacity(IMBALANCED_JOBS);
                    for index in 0..IMBALANCED_JOBS {
                        let cycles = cycles_for(index);
                        handles.push(
                            pool.spawn(move || Ok::<u64, proxima::ProximaError>(cpu_burn(cycles))),
                        );
                    }
                    for handle in handles {
                        let _ = handle.await;
                    }
                    total += started.elapsed();
                }
            });
            total
        });
    });

    group.bench_function("tokio_spawn_blocking", |bencher| {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .max_blocking_threads(4)
            .build()
            .unwrap();
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            runtime.block_on(async {
                for _ in 0..iters {
                    let started = Instant::now();
                    let mut handles = Vec::with_capacity(IMBALANCED_JOBS);
                    for index in 0..IMBALANCED_JOBS {
                        let cycles = cycles_for(index);
                        handles.push(tokio::task::spawn_blocking(move || cpu_burn(cycles)));
                    }
                    for handle in handles {
                        let _ = handle.await;
                    }
                    total += started.elapsed();
                }
            });
            total
        });
    });

    group.finish();
}

const LATENCY_TRIPS: usize = 200;

// design-favors: neither — single-shot spawn+await round-trip measures
// dispatch + wake latency floor. Neither pool's design point is engaged.
fn bench_latency(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("bg_pool_latency");
    configure_group(&mut group);
    group.throughput(Throughput::Elements(LATENCY_TRIPS as u64));

    // round-trip latency floor: spawn one job, await, repeat. measures the
    // dispatch + wake + await cost per job WITHOUT amortization from
    // batched awaits. proxima_typed should show the per-spawn floor;
    // proxima_dyn adds the Box<dyn FnOnce> + return-Box::pin overhead per
    // round trip.

    group.bench_function("proxima_dyn", |bencher| {
        let pool = Arc::new(ProximaBackgroundPool::with_threads(4).expect("build pool"));
        let pool_dyn: Arc<dyn BackgroundPool> = pool.clone();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            runtime.block_on(async {
                for _ in 0..iters {
                    let started = Instant::now();
                    for index in 0..LATENCY_TRIPS as u32 {
                        let job: Box<
                            dyn FnOnce() -> Result<
                                    Box<dyn std::any::Any + Send>,
                                    proxima::ProximaError,
                                > + Send,
                        > = Box::new(move || {
                            Ok(Box::new(index * 2) as Box<dyn std::any::Any + Send>)
                        });
                        let _ = BackgroundPool::spawn(&*pool_dyn, job).await;
                    }
                    total += started.elapsed();
                }
            });
            total
        });
    });

    group.bench_function("proxima_typed", |bencher| {
        let pool = Arc::new(ProximaBackgroundPool::with_threads(4).expect("build pool"));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            runtime.block_on(async {
                for _ in 0..iters {
                    let started = Instant::now();
                    for index in 0..LATENCY_TRIPS as u32 {
                        let _ = pool
                            .spawn(move || Ok::<u32, proxima::ProximaError>(index * 2))
                            .await;
                    }
                    total += started.elapsed();
                }
            });
            total
        });
    });

    group.bench_function("rayon", |bencher| {
        let pool = Arc::new(RayonBackgroundPool::with_threads(4).expect("build pool"));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            runtime.block_on(async {
                for _ in 0..iters {
                    let started = Instant::now();
                    for index in 0..LATENCY_TRIPS as u32 {
                        let job: Box<
                            dyn FnOnce() -> Result<
                                    Box<dyn std::any::Any + Send>,
                                    proxima::ProximaError,
                                > + Send,
                        > = Box::new(move || {
                            Ok(Box::new(index * 2) as Box<dyn std::any::Any + Send>)
                        });
                        let _ = pool.spawn(job).await;
                    }
                    total += started.elapsed();
                }
            });
            total
        });
    });

    #[cfg(feature = "runtime-prime-bgpool-rayon")]
    group.bench_function("proxima_rayon_backed", |bencher| {
        let pool = Arc::new(RayonBackgroundPool::with_threads(4).expect("build pool"));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            runtime.block_on(async {
                for _ in 0..iters {
                    let started = Instant::now();
                    for index in 0..LATENCY_TRIPS as u32 {
                        let _ = pool
                            .spawn(move || Ok::<u32, proxima::ProximaError>(index * 2))
                            .await;
                    }
                    total += started.elapsed();
                }
            });
            total
        });
    });

    group.bench_function("tokio_spawn_blocking", |bencher| {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .max_blocking_threads(4)
            .build()
            .unwrap();
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            runtime.block_on(async {
                for _ in 0..iters {
                    let started = Instant::now();
                    for index in 0..LATENCY_TRIPS as u32 {
                        let _ = tokio::task::spawn_blocking(move || index * 2).await;
                    }
                    total += started.elapsed();
                }
            });
            total
        });
    });

    group.finish();
}

const MULTI_PRODUCER_THREADS: usize = 4;
const MULTI_PRODUCER_JOBS_EACH: usize = 250;

// design-favors: incumbent — 4 producer threads × 250 jobs each contends on
// the job-injection primitive. Rayon's Injector is the queue rayon ships for
// external-producer contention; proxima uses the same primitive. Hitting their
// shared design point.
fn bench_multi_producer(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("bg_pool_multi_producer");
    configure_group(&mut group);
    let total_jobs = MULTI_PRODUCER_THREADS * MULTI_PRODUCER_JOBS_EACH;
    group.throughput(Throughput::Elements(total_jobs as u64));

    // M producer threads each push K jobs concurrently through one
    // pool. measures contention on the producer side — for proxima
    // this hits the Injector::push atomic; for rayon's external-push
    // path same primitive (Injector). For typed paths, proxima skips
    // the API-level Box<dyn FnOnce>.

    group.bench_function("proxima_dyn", |bencher| {
        let pool = Arc::new(ProximaBackgroundPool::with_threads(4).expect("build pool"));
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let started = Instant::now();
                let mut threads = Vec::with_capacity(MULTI_PRODUCER_THREADS);
                for _ in 0..MULTI_PRODUCER_THREADS {
                    let pool_dyn: Arc<dyn BackgroundPool> = pool.clone();
                    threads.push(std::thread::spawn(move || {
                        let runtime = tokio::runtime::Builder::new_current_thread()
                            .build()
                            .unwrap();
                        runtime.block_on(async {
                            let mut handles = Vec::with_capacity(MULTI_PRODUCER_JOBS_EACH);
                            for index in 0..MULTI_PRODUCER_JOBS_EACH as u32 {
                                let job: Box<
                                    dyn FnOnce() -> Result<
                                            Box<dyn std::any::Any + Send>,
                                            proxima::ProximaError,
                                        > + Send,
                                > = Box::new(move || {
                                    Ok(Box::new(index * 2) as Box<dyn std::any::Any + Send>)
                                });
                                handles.push(BackgroundPool::spawn(&*pool_dyn, job));
                            }
                            for handle in handles {
                                let _ = handle.await;
                            }
                        });
                    }));
                }
                for thread in threads {
                    let _ = thread.join();
                }
                total += started.elapsed();
            }
            total
        });
    });

    group.bench_function("proxima_typed", |bencher| {
        let pool = Arc::new(ProximaBackgroundPool::with_threads(4).expect("build pool"));
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let started = Instant::now();
                let mut threads = Vec::with_capacity(MULTI_PRODUCER_THREADS);
                for _ in 0..MULTI_PRODUCER_THREADS {
                    let pool = pool.clone();
                    threads.push(std::thread::spawn(move || {
                        let runtime = tokio::runtime::Builder::new_current_thread()
                            .build()
                            .unwrap();
                        runtime.block_on(async {
                            let mut handles = Vec::with_capacity(MULTI_PRODUCER_JOBS_EACH);
                            for index in 0..MULTI_PRODUCER_JOBS_EACH as u32 {
                                handles.push(
                                    pool.spawn(move || Ok::<u32, proxima::ProximaError>(index * 2)),
                                );
                            }
                            for handle in handles {
                                let _ = handle.await;
                            }
                        });
                    }));
                }
                for thread in threads {
                    let _ = thread.join();
                }
                total += started.elapsed();
            }
            total
        });
    });

    group.bench_function("rayon", |bencher| {
        let pool = Arc::new(RayonBackgroundPool::with_threads(4).expect("build pool"));
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let started = Instant::now();
                let mut threads = Vec::with_capacity(MULTI_PRODUCER_THREADS);
                for _ in 0..MULTI_PRODUCER_THREADS {
                    let pool = pool.clone();
                    threads.push(std::thread::spawn(move || {
                        let runtime = tokio::runtime::Builder::new_current_thread()
                            .build()
                            .unwrap();
                        runtime.block_on(async {
                            let mut handles = Vec::with_capacity(MULTI_PRODUCER_JOBS_EACH);
                            for index in 0..MULTI_PRODUCER_JOBS_EACH as u32 {
                                let job: Box<
                                    dyn FnOnce() -> Result<
                                            Box<dyn std::any::Any + Send>,
                                            proxima::ProximaError,
                                        > + Send,
                                > = Box::new(move || {
                                    Ok(Box::new(index * 2) as Box<dyn std::any::Any + Send>)
                                });
                                handles.push(pool.spawn(job));
                            }
                            for handle in handles {
                                let _ = handle.await;
                            }
                        });
                    }));
                }
                for thread in threads {
                    let _ = thread.join();
                }
                total += started.elapsed();
            }
            total
        });
    });

    #[cfg(feature = "runtime-prime-bgpool-rayon")]
    group.bench_function("proxima_rayon_backed", |bencher| {
        let pool = Arc::new(RayonBackgroundPool::with_threads(4).expect("build pool"));
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let started = Instant::now();
                let mut threads = Vec::with_capacity(MULTI_PRODUCER_THREADS);
                for _ in 0..MULTI_PRODUCER_THREADS {
                    let pool = pool.clone();
                    threads.push(std::thread::spawn(move || {
                        let runtime = tokio::runtime::Builder::new_current_thread()
                            .build()
                            .unwrap();
                        runtime.block_on(async {
                            let mut handles = Vec::with_capacity(MULTI_PRODUCER_JOBS_EACH);
                            for index in 0..MULTI_PRODUCER_JOBS_EACH as u32 {
                                handles.push(
                                    pool.spawn(move || Ok::<u32, proxima::ProximaError>(index * 2)),
                                );
                            }
                            for handle in handles {
                                let _ = handle.await;
                            }
                        });
                    }));
                }
                for thread in threads {
                    let _ = thread.join();
                }
                total += started.elapsed();
            }
            total
        });
    });

    group.bench_function("tokio_spawn_blocking", |bencher| {
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(MULTI_PRODUCER_THREADS)
                .max_blocking_threads(4)
                .build()
                .unwrap(),
        );
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let started = Instant::now();
                let mut threads = Vec::with_capacity(MULTI_PRODUCER_THREADS);
                for _ in 0..MULTI_PRODUCER_THREADS {
                    let runtime = runtime.clone();
                    threads.push(std::thread::spawn(move || {
                        runtime.block_on(async {
                            let mut handles = Vec::with_capacity(MULTI_PRODUCER_JOBS_EACH);
                            for index in 0..MULTI_PRODUCER_JOBS_EACH as u32 {
                                handles.push(tokio::task::spawn_blocking(move || index * 2));
                            }
                            for handle in handles {
                                let _ = handle.await;
                            }
                        });
                    }));
                }
                for thread in threads {
                    let _ = thread.join();
                }
                total += started.elapsed();
            }
            total
        });
    });

    group.finish();
}

// fork-join workload constants. 100k elements, 10% heavy (10k elements
// each ~500-1000 ns of cpu_burn), 90% light (1 sum op). manual chunking
// into N chunks (NOT auto-split via rayon's iterator machinery — that
// would make the comparison "rayon par_iter's intelligence" rather than
// "the underlying executor"). using N=16 chunks for 4 workers means
// some workers process 4 chunks; the imbalance comes from heavy elements
// being unevenly distributed across chunks (the data layout has heavy
// every 10 elements, so chunks are roughly balanced — but contention
// in dispatch + per-chunk completion ordering still differentiate the
// pools).
const FORK_JOIN_ELEMENTS: usize = 100_000;
const FORK_JOIN_CHUNKS: usize = 16;
const FORK_JOIN_HEAVY_CYCLES: u64 = 500;

#[inline(always)]
fn fork_join_compute(value: u32) -> u64 {
    if value.is_multiple_of(10) {
        cpu_burn(FORK_JOIN_HEAVY_CYCLES) ^ u64::from(value)
    } else {
        u64::from(value)
    }
}

// design-favors: incumbent — fork-join over 100k elements via par_iter /
// par_reduce / par_chunks_mut / par_sort_by / par_bridge / par_filter is the
// shape rayon was built around. Recursive splitting + work-stealing + tree
// reduce is rayon's load-bearing design point. Hitting all of these.
fn bench_fork_join(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("bg_pool_fork_join");
    configure_group(&mut group);
    group.throughput(Throughput::Elements(FORK_JOIN_ELEMENTS as u64));

    // shared input data. `Arc<[u32]>` (not `Arc<Vec<_>>`) because
    // par_reduce expects `Arc<[Item]>` directly; the other arms only
    // need `Deref<Target = [u32]>` which both support, so this also
    // works for all manual-chunk arms.
    let data: Arc<[u32]> = (0..FORK_JOIN_ELEMENTS as u32).collect::<Vec<_>>().into();

    // gold standard: rayon's actual par_iter. uses rayon's recursive
    // splitting + work-stealing on the underlying ThreadPool. measures
    // the upper bound of what work-stealing buys for this workload.
    group.bench_function("rayon_par_iter", |bencher| {
        use rayon::prelude::*;
        let rayon_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("rayon pool");
        bencher.iter(|| {
            let data = data.clone();
            rayon_pool.install(|| {
                let sum: u64 = data.par_iter().map(|&value| fork_join_compute(value)).sum();
                std::hint::black_box(sum)
            })
        });
    });

    // manual chunking via proxima typed API: split data into N chunks,
    // spawn one job per chunk, await all, sum partials. comparable to
    // rayon's `pool.spawn` arm — same dispatch shape, different backend.
    group.bench_function("proxima_typed", |bencher| {
        let pool = Arc::new(ProximaBackgroundPool::with_threads(4).expect("build pool"));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        bencher.iter_custom(|iters| {
            let chunk_size = FORK_JOIN_ELEMENTS.div_ceil(FORK_JOIN_CHUNKS);
            let mut total = Duration::ZERO;
            runtime.block_on(async {
                for _ in 0..iters {
                    let started = Instant::now();
                    let mut handles = Vec::with_capacity(FORK_JOIN_CHUNKS);
                    for chunk_index in 0..FORK_JOIN_CHUNKS {
                        let data = data.clone();
                        let start = chunk_index * chunk_size;
                        let end = (start + chunk_size).min(FORK_JOIN_ELEMENTS);
                        handles.push(pool.spawn(move || {
                            let sum: u64 = data[start..end]
                                .iter()
                                .map(|&value| fork_join_compute(value))
                                .sum();
                            Ok::<u64, proxima::ProximaError>(sum)
                        }));
                    }
                    let mut grand_total: u64 = 0;
                    for handle in handles {
                        grand_total = grand_total.wrapping_add(handle.await.unwrap_or(0));
                    }
                    std::hint::black_box(grand_total);
                    total += started.elapsed();
                }
            });
            total
        });
    });

    group.bench_function("proxima_dyn", |bencher| {
        let pool = Arc::new(ProximaBackgroundPool::with_threads(4).expect("build pool"));
        let pool_dyn: Arc<dyn BackgroundPool> = pool.clone();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        bencher.iter_custom(|iters| {
            let chunk_size = FORK_JOIN_ELEMENTS.div_ceil(FORK_JOIN_CHUNKS);
            let mut total = Duration::ZERO;
            runtime.block_on(async {
                for _ in 0..iters {
                    let started = Instant::now();
                    let mut handles = Vec::with_capacity(FORK_JOIN_CHUNKS);
                    for chunk_index in 0..FORK_JOIN_CHUNKS {
                        let data = data.clone();
                        let start = chunk_index * chunk_size;
                        let end = (start + chunk_size).min(FORK_JOIN_ELEMENTS);
                        let job: Box<
                            dyn FnOnce() -> Result<
                                    Box<dyn std::any::Any + Send>,
                                    proxima::ProximaError,
                                > + Send,
                        > = Box::new(move || {
                            let sum: u64 = data[start..end]
                                .iter()
                                .map(|&value| fork_join_compute(value))
                                .sum();
                            Ok(Box::new(sum) as Box<dyn std::any::Any + Send>)
                        });
                        handles.push(BackgroundPool::spawn(&*pool_dyn, job));
                    }
                    let mut grand_total: u64 = 0;
                    for handle in handles {
                        if let Ok(value) = handle.await
                            && let Ok(sum) = value.downcast::<u64>()
                        {
                            grand_total = grand_total.wrapping_add(*sum);
                        }
                    }
                    std::hint::black_box(grand_total);
                    total += started.elapsed();
                }
            });
            total
        });
    });

    // rayon backend through proxima's BackgroundPool API (dyn-trait):
    // same chunking + spawn pattern, different backend.
    group.bench_function("rayon_via_bgpool", |bencher| {
        let pool = Arc::new(RayonBackgroundPool::with_threads(4).expect("build pool"));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        bencher.iter_custom(|iters| {
            let chunk_size = FORK_JOIN_ELEMENTS.div_ceil(FORK_JOIN_CHUNKS);
            let mut total = Duration::ZERO;
            runtime.block_on(async {
                for _ in 0..iters {
                    let started = Instant::now();
                    let mut handles = Vec::with_capacity(FORK_JOIN_CHUNKS);
                    for chunk_index in 0..FORK_JOIN_CHUNKS {
                        let data = data.clone();
                        let start = chunk_index * chunk_size;
                        let end = (start + chunk_size).min(FORK_JOIN_ELEMENTS);
                        let job: Box<
                            dyn FnOnce() -> Result<
                                    Box<dyn std::any::Any + Send>,
                                    proxima::ProximaError,
                                > + Send,
                        > = Box::new(move || {
                            let sum: u64 = data[start..end]
                                .iter()
                                .map(|&value| fork_join_compute(value))
                                .sum();
                            Ok(Box::new(sum) as Box<dyn std::any::Any + Send>)
                        });
                        handles.push(pool.spawn(job));
                    }
                    let mut grand_total: u64 = 0;
                    for handle in handles {
                        if let Ok(value) = handle.await
                            && let Ok(sum) = value.downcast::<u64>()
                        {
                            grand_total = grand_total.wrapping_add(*sum);
                        }
                    }
                    std::hint::black_box(grand_total);
                    total += started.elapsed();
                }
            });
            total
        });
    });

    #[cfg(feature = "runtime-prime-bgpool-rayon")]
    group.bench_function("proxima_rayon_backed", |bencher| {
        let pool = Arc::new(RayonBackgroundPool::with_threads(4).expect("build pool"));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        bencher.iter_custom(|iters| {
            let chunk_size = FORK_JOIN_ELEMENTS.div_ceil(FORK_JOIN_CHUNKS);
            let mut total = Duration::ZERO;
            runtime.block_on(async {
                for _ in 0..iters {
                    let started = Instant::now();
                    let mut handles = Vec::with_capacity(FORK_JOIN_CHUNKS);
                    for chunk_index in 0..FORK_JOIN_CHUNKS {
                        let data = data.clone();
                        let start = chunk_index * chunk_size;
                        let end = (start + chunk_size).min(FORK_JOIN_ELEMENTS);
                        handles.push(pool.spawn(move || {
                            let sum: u64 = data[start..end]
                                .iter()
                                .map(|&value| fork_join_compute(value))
                                .sum();
                            Ok::<u64, proxima::ProximaError>(sum)
                        }));
                    }
                    let mut grand_total: u64 = 0;
                    for handle in handles {
                        grand_total = grand_total.wrapping_add(handle.await.unwrap_or(0));
                    }
                    std::hint::black_box(grand_total);
                    total += started.elapsed();
                }
            });
            total
        });
    });

    // par_reduce on top of ProximaBackgroundPool: recursive split + tree
    // reduce. compares directly against rayon_par_iter on the same input.
    // sweeping threshold lets us find the elbow vs spawn overhead.
    #[cfg(feature = "runtime-prime-bgpool-par")]
    for chunk_threshold in &[512usize, 2048, 8192] {
        let chunk_threshold = *chunk_threshold;
        let arm_name = format!("proxima_par_reduce_{chunk_threshold}");
        let data_for_arm = data.clone();
        group.bench_function(&arm_name, move |bencher| {
            use proxima::runtime::prime::os::par::par_reduce_with_threshold;
            let pool = Arc::new(ProximaBackgroundPool::with_threads(4).expect("build pool"));
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            bencher.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                runtime.block_on(async {
                    for _ in 0..iters {
                        let started = Instant::now();
                        let sum = par_reduce_with_threshold(
                            &pool,
                            data_for_arm.clone(),
                            || 0u64,
                            |&value| fork_join_compute(value),
                            |left, right| left.wrapping_add(right),
                            chunk_threshold,
                        )
                        .await;
                        std::hint::black_box(sum);
                        total += started.elapsed();
                    }
                });
                total
            });
        });
    }

    // trait-API arm using auto-derived threshold via `data.par_iter(&pool)`.
    // measures the tax of going through `ProximaParIter` / `ProximaParMap`
    // vs constructing par_reduce_with_threshold by hand. should match
    // the proxima_par_reduce_* arm whose threshold lands closest to the
    // auto-derived value (4 leaves/worker = ~6250 elements for 100K/4).
    #[cfg(feature = "runtime-prime-bgpool-par")]
    {
        let data_for_arm = data.clone();
        group.bench_function("proxima_par_iter_map_sum", move |bencher| {
            use proxima::runtime::prime::os::par::ProximaParIter;
            let pool = Arc::new(ProximaBackgroundPool::with_threads(4).expect("build pool"));
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            bencher.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                runtime.block_on(async {
                    for _ in 0..iters {
                        let started = Instant::now();
                        let sum: u64 = data_for_arm
                            .clone()
                            .par_iter(&pool)
                            .map(|&value| fork_join_compute(value))
                            .sum()
                            .await;
                        std::hint::black_box(sum);
                        total += started.elapsed();
                    }
                });
                total
            });
        });
    }

    // par_map_collect arm: builds a Vec<u64> of mapped outputs. measures
    // the tree-concatenation cost on top of the recursive split. expect
    // a constant overhead vs par_iter_map_sum because the leaf builds a
    // local Vec + the merges concatenate.
    #[cfg(feature = "runtime-prime-bgpool-par")]
    {
        let data_for_arm = data.clone();
        group.bench_function("proxima_par_iter_map_collect", move |bencher| {
            use proxima::runtime::prime::os::par::ProximaParIter;
            let pool = Arc::new(ProximaBackgroundPool::with_threads(4).expect("build pool"));
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            bencher.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                runtime.block_on(async {
                    for _ in 0..iters {
                        let started = Instant::now();
                        let collected: Vec<u64> = data_for_arm
                            .clone()
                            .par_iter(&pool)
                            .map(|&value| fork_join_compute(value))
                            .collect()
                            .await;
                        std::hint::black_box(collected);
                        total += started.elapsed();
                    }
                });
                total
            });
        });
    }

    // map_async on the same workload: closure is wrapped in `async move`
    // but does no real awaits beyond the workload itself. measures the
    // pure overhead of the polling-worker path vs sync map_sum. expect
    // a small constant tax for the per-leaf tokio block_on + the per-
    // item future state machine.
    #[cfg(all(
        feature = "runtime-prime-bgpool-par",
        feature = "runtime-prime-bgpool-async"
    ))]
    {
        let data_for_arm = data.clone();
        group.bench_function("proxima_par_iter_map_async_sum", move |bencher| {
            use proxima::runtime::prime::os::par::ProximaParIter;
            let pool = Arc::new(ProximaBackgroundPool::with_threads(4).expect("build pool"));
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            bencher.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                runtime.block_on(async {
                    for _ in 0..iters {
                        let started = Instant::now();
                        let sum: u64 = data_for_arm
                            .clone()
                            .par_iter(&pool)
                            .map_async(|value| async move { fork_join_compute(value) })
                            .sum()
                            .await;
                        std::hint::black_box(sum);
                        total += started.elapsed();
                    }
                });
                total
            });
        });
    }

    // par_stream unordered: N futures in flight, output collected in
    // completion order. different shape from par_iter — this is the
    // "buffered concurrency" pattern for I/O-bound workloads. on a
    // CPU-bound workload like fork_join_compute, expect higher overhead
    // than par_iter because no recursive split / tree reduce.
    #[cfg(feature = "runtime-prime-bgpool-async")]
    for concurrency in &[16usize, 64] {
        let concurrency = *concurrency;
        let arm_name = format!("proxima_par_stream_then_{concurrency}");
        let data_for_arm = data.clone();
        group.bench_function(&arm_name, move |bencher| {
            use futures::StreamExt;
            use proxima::runtime::prime::os::par::ProximaParStreamExt;
            let pool = Arc::new(ProximaBackgroundPool::with_threads(4).expect("build pool"));
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            bencher.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                runtime.block_on(async {
                    for _ in 0..iters {
                        let started = Instant::now();
                        let collected: Vec<u64> = data_for_arm
                            .clone()
                            .par_stream(&pool, concurrency)
                            .then(|value| async move { fork_join_compute(value) })
                            .collect()
                            .await;
                        std::hint::black_box(collected);
                        total += started.elapsed();
                    }
                });
                total
            });
        });
    }

    // par_stream ordered: same shape but reorder buffer keeps output in
    // input order. measures the reorder-buffer tax vs the unordered
    // variant (HashMap insert/remove + sliding-window emission). expect
    // a small constant cost on top of `then`.
    #[cfg(feature = "runtime-prime-bgpool-async")]
    {
        let data_for_arm = data.clone();
        group.bench_function("proxima_par_stream_then_ordered_64", move |bencher| {
            use futures::StreamExt;
            use proxima::runtime::prime::os::par::ProximaParStreamExt;
            let pool = Arc::new(ProximaBackgroundPool::with_threads(4).expect("build pool"));
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            bencher.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                runtime.block_on(async {
                    for _ in 0..iters {
                        let started = Instant::now();
                        let collected: Vec<u64> = data_for_arm
                            .clone()
                            .par_stream(&pool, 64)
                            .then_ordered(|value| async move { fork_join_compute(value) })
                            .collect()
                            .await;
                        std::hint::black_box(collected);
                        total += started.elapsed();
                    }
                });
                total
            });
        });
    }

    // ---- par_filter bench arms ----

    #[cfg(feature = "runtime-prime-bgpool-par")]
    {
        let data_for_arm = data.clone();
        group.bench_function("proxima_par_filter", move |bencher| {
            use proxima::runtime::prime::os::par::par_filter_with_threshold;
            let pool = Arc::new(ProximaBackgroundPool::with_threads(4).expect("build pool"));
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            bencher.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                runtime.block_on(async {
                    for _ in 0..iters {
                        let started = Instant::now();
                        let result: Vec<u32> = par_filter_with_threshold(
                            &pool,
                            data_for_arm.clone(),
                            |&value| value % 2 == 0,
                            1024,
                        )
                        .await;
                        std::hint::black_box(result);
                        total += started.elapsed();
                    }
                });
                total
            });
        });
    }

    group.bench_function("rayon_par_iter_filter", |bencher| {
        use rayon::prelude::*;
        let rayon_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("rayon pool");
        bencher.iter(|| {
            let data = data.clone();
            rayon_pool.install(|| {
                let result: Vec<u32> = data
                    .par_iter()
                    .filter(|&&value| value % 2 == 0)
                    .copied()
                    .collect();
                std::hint::black_box(result)
            })
        });
    });

    // ---- par_sort_by bench arms ----

    #[cfg(feature = "runtime-prime-bgpool-par")]
    {
        let sort_data: Vec<u32> = (0..FORK_JOIN_ELEMENTS as u32).rev().collect();
        group.bench_function("proxima_par_sort_by", move |bencher| {
            use proxima::runtime::prime::os::par::par_sort_by;
            let pool = Arc::new(ProximaBackgroundPool::with_threads(4).expect("build pool"));
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            bencher.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                runtime.block_on(async {
                    for _ in 0..iters {
                        let started = Instant::now();
                        let result = par_sort_by(&pool, sort_data.clone(), |a, b| a.cmp(b)).await;
                        std::hint::black_box(result);
                        total += started.elapsed();
                    }
                });
                total
            });
        });
    }

    group.bench_function("rayon_par_sort_by", |bencher| {
        use rayon::prelude::*;
        let rayon_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("rayon pool");
        let sort_data: Vec<u32> = (0..FORK_JOIN_ELEMENTS as u32).rev().collect();
        bencher.iter(|| {
            let mut data = sort_data.clone();
            rayon_pool.install(|| {
                data.par_sort_by(|a, b| a.cmp(b));
                std::hint::black_box(&data);
            })
        });
    });

    // ---- par_chunks_mut bench arms ----

    #[cfg(feature = "runtime-prime-bgpool-par")]
    {
        let mut chunks_data: Vec<u32> = (0..FORK_JOIN_ELEMENTS as u32).collect();
        group.bench_function("proxima_par_chunks_mut", move |bencher| {
            use proxima::runtime::prime::os::par::par_chunks_mut;
            let pool = Arc::new(ProximaBackgroundPool::with_threads(4).expect("build pool"));
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            bencher.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                runtime.block_on(async {
                    for _ in 0..iters {
                        let started = Instant::now();
                        par_chunks_mut(&pool, &mut chunks_data, 1024, |chunk| {
                            for item in chunk.iter_mut() {
                                *item = item.wrapping_add(1);
                            }
                        })
                        .await;
                        std::hint::black_box(&chunks_data);
                        total += started.elapsed();
                    }
                });
                total
            });
        });
    }

    group.bench_function("rayon_par_chunks_mut", |bencher| {
        use rayon::prelude::*;
        let rayon_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("rayon pool");
        let mut chunks_data: Vec<u32> = (0..FORK_JOIN_ELEMENTS as u32).collect();
        bencher.iter(|| {
            rayon_pool.install(|| {
                chunks_data.par_chunks_mut(1024).for_each(|chunk| {
                    for item in chunk.iter_mut() {
                        *item = item.wrapping_add(1);
                    }
                });
                std::hint::black_box(&chunks_data);
            })
        });
    });

    // ---- par_bridge bench arms ----

    #[cfg(feature = "runtime-prime-bgpool-par")]
    {
        group.bench_function("proxima_par_bridge", move |bencher| {
            use proxima::runtime::prime::os::par::par_bridge;
            use std::sync::atomic::{AtomicU64, Ordering};
            let pool = Arc::new(ProximaBackgroundPool::with_threads(4).expect("build pool"));
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            bencher.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                runtime.block_on(async {
                    for _ in 0..iters {
                        let started = Instant::now();
                        let acc = Arc::new(AtomicU64::new(0));
                        let acc_for_work = acc.clone();
                        par_bridge(&pool, 0u32..FORK_JOIN_ELEMENTS as u32, 4, move |value| {
                            acc_for_work.fetch_add(fork_join_compute(value), Ordering::Relaxed);
                        })
                        .await;
                        std::hint::black_box(acc.load(Ordering::Relaxed));
                        total += started.elapsed();
                    }
                });
                total
            });
        });
    }

    group.bench_function("rayon_par_bridge", |bencher| {
        use rayon::prelude::*;
        let rayon_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("rayon pool");
        bencher.iter(|| {
            let sum: u64 = rayon_pool.install(|| {
                (0u32..FORK_JOIN_ELEMENTS as u32)
                    .par_bridge()
                    .map(fork_join_compute)
                    .sum()
            });
            std::hint::black_box(sum)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_tiny_jobs_throughput,
    bench_cpu_imbalanced,
    bench_latency,
    bench_multi_producer,
    bench_fork_join,
);
criterion_main!(benches);
