//! Cross-core spawn workload coverage. Exercises the full producer-side
//! `spawn_on_core` → cross-core inbox → target-core executor → task body
//! → completion path under several patterns:
//!
//!   1. `spawn_burst_1k` — one burst of 1000 spawns from one producer
//!      thread. The original baseline (also in `bench_runtime_swap.rs`).
//!   2. `spawn_burst_10k` — same shape, 10× the count. Reveals whether
//!      the gap is per-spawn linear or has fixed-cost components.
//!   3. `spawn_sustained_10k_per_sec` — sustained spawning at a measured
//!      rate over a longer window. Closest to a production load
//!      generator's behavior; reveals tail/steady-state cost.
//!   4. `spawn_fanin_N_producers` — N producer threads concurrently
//!      spawning 1000 tasks each. N = 1, 2, 4, 8. Tests inbox-lane
//!      contention and producer-side cache pressure.
//!
//! Compared backends:
//!   - `tokio_per_core` (pinned tokio current-thread per CPU; cross-core
//!     dispatch via flume).
//!   - `prime` (`PrimeRuntime` — our from-scratch per-core runtime).
//!
//! required-features: runtime-tokio, runtime-prime-full.

#![cfg(all(
    feature = "runtime-tokio",
    all(
        feature = "runtime-prime-executor",
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-reactor",
        feature = "runtime-prime-bgpool"
    )
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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima::runtime::{
    CoreId, PrimeRuntime, Runtime, TokioPerCoreRuntime, spawn_on_core_blocking_with,
};

const CORES: usize = 2;

fn configure_group<M: criterion::measurement::Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
) {
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
}

/// One-burst workload: spawn `count` tasks round-robin across cores from
/// a single producer thread, wait for all to complete, measure wall time.
///
/// Uses `spawn_on_core_blocking_with` so the producer yield-loops on
/// inbox saturation (`SpawnError::InboxFull`) rather than silently
/// dropping tasks. This is the canonical batch-dispatch pattern for
/// load generators; production code that genuinely needs back-pressure
/// awareness uses `spawn_on_core` directly and matches on `SpawnError`.
fn run_burst(runtime: &Arc<dyn Runtime>, count: usize) -> Duration {
    let counter = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    for index in 0..count {
        let counter = counter.clone();
        let core = CoreId(index % CORES);
        let _ = spawn_on_core_blocking_with(runtime.as_ref(), core, move || {
            let counter = counter.clone();
            Box::pin(async move {
                counter.fetch_add(1, Ordering::AcqRel);
            })
        });
    }
    while counter.load(Ordering::Acquire) < count {
        std::hint::spin_loop();
    }
    started.elapsed()
}

/// typed-task variant: PrimeRuntime-only fast path that skips the
/// per-spawn `Box::pin(dyn Future)`. Caller hands the concrete future
/// to `spawn_typed_on_core`; the runtime stamps it into an `InlineTask`
/// (inline byte buffer when small, single `Box<F>` when oversized) and
/// dispatches via a per-`F` vtable, no `dyn Future` indirection.
///
/// Held as a separate run_burst variant because the trait-typed
/// `Arc<dyn Runtime>` cannot expose `spawn_typed_on_core<F>` (would
/// re-introduce the very fat-pointer cost the fast path exists to
/// avoid). The arm takes `Arc<PrimeRuntime>` directly.
fn run_burst_typed(runtime: &Arc<PrimeRuntime>, count: usize) -> Duration {
    let counter = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    for index in 0..count {
        let counter = counter.clone();
        let core = CoreId(index % CORES);
        // typed dispatch — no Box::pin at the call site; the runtime
        // composes the InlineTask. spawn-burst's async block captures
        // one Arc<AtomicUsize> (~16 bytes), well within the inline
        // 56-byte budget — no heap alloc on either side.
        loop {
            match runtime.spawn_typed_on_core(core, {
                let counter = counter.clone();
                async move {
                    counter.fetch_add(1, Ordering::AcqRel);
                }
            }) {
                Ok(()) => break,
                Err(proxima::runtime::SpawnError::InboxFull) => std::thread::yield_now(),
                Err(proxima::runtime::SpawnError::Disconnected) => break,
            }
        }
    }
    while counter.load(Ordering::Acquire) < count {
        std::hint::spin_loop();
    }
    started.elapsed()
}

/// Sustained-rate workload: spawn at the target rate for the duration
/// window. Returns wall time and observed count (target is best-effort —
/// scheduler delays slip below target). Reports the actual rate achieved.
fn run_sustained(
    runtime: &Arc<dyn Runtime>,
    duration: Duration,
    target_per_sec: u32,
) -> (Duration, usize) {
    let counter = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    let interval = Duration::from_nanos(1_000_000_000 / u64::from(target_per_sec));
    let mut next = started;
    let mut spawned = 0_usize;
    while started.elapsed() < duration {
        let now = Instant::now();
        if now >= next {
            let counter = counter.clone();
            let core = CoreId(spawned % CORES);
            let _ = spawn_on_core_blocking_with(runtime.as_ref(), core, move || {
                let counter = counter.clone();
                Box::pin(async move {
                    counter.fetch_add(1, Ordering::AcqRel);
                })
            });
            spawned += 1;
            next += interval;
        } else {
            std::hint::spin_loop();
        }
    }
    while counter.load(Ordering::Acquire) < spawned {
        std::hint::spin_loop();
    }
    (started.elapsed(), spawned)
}

/// Fan-in workload: `producers` threads concurrently each spawn `per_thread`
/// tasks. Total = `producers * per_thread`. All tasks complete on the runtime
/// before timing stops.
fn run_fanin(runtime: &Arc<dyn Runtime>, producers: usize, per_thread: usize) -> Duration {
    let counter = Arc::new(AtomicUsize::new(0));
    let total = producers * per_thread;
    let started = Instant::now();
    let mut handles = Vec::with_capacity(producers);
    for thread_index in 0..producers {
        let runtime = runtime.clone();
        let counter = counter.clone();
        let handle = thread::spawn(move || {
            for spawn_index in 0..per_thread {
                let counter = counter.clone();
                let core = CoreId((thread_index + spawn_index) % CORES);
                let _ = spawn_on_core_blocking_with(runtime.as_ref(), core, move || {
                    let counter = counter.clone();
                    Box::pin(async move {
                        counter.fetch_add(1, Ordering::AcqRel);
                    })
                });
            }
        });
        handles.push(handle);
    }
    for handle in handles {
        let _ = handle.join();
    }
    while counter.load(Ordering::Acquire) < total {
        std::hint::spin_loop();
    }
    started.elapsed()
}

fn bench_burst(criterion: &mut Criterion, name: &str, count: usize) {
    let mut group = criterion.benchmark_group(name);
    configure_group(&mut group);
    group.throughput(Throughput::Elements(count as u64));

    group.bench_function("tokio_per_core", |bencher| {
        let runtime: Arc<dyn Runtime> =
            Arc::new(TokioPerCoreRuntime::new(CORES).expect("tokio_per_core"));
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                total += run_burst(&runtime, count);
            }
            total
        });
    });

    group.bench_function("prime", |bencher| {
        let runtime: Arc<dyn Runtime> = Arc::new(PrimeRuntime::new(CORES).expect("prime"));
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                total += run_burst(&runtime, count);
            }
            total
        });
    });

    group.bench_function("prime_typed", |bencher| {
        let runtime: Arc<PrimeRuntime> = Arc::new(PrimeRuntime::new(CORES).expect("prime"));
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                total += run_burst_typed(&runtime, count);
            }
            total
        });
    });

    group.finish();
}

fn bench_sustained(criterion: &mut Criterion, name: &str, target_per_sec: u32) {
    let mut group = criterion.benchmark_group(name);
    group.sample_size(15);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(4));
    let window = Duration::from_millis(500);

    group.bench_function("tokio_per_core", |bencher| {
        let runtime: Arc<dyn Runtime> =
            Arc::new(TokioPerCoreRuntime::new(CORES).expect("tokio_per_core"));
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let (elapsed, _) = run_sustained(&runtime, window, target_per_sec);
                total += elapsed;
            }
            total
        });
    });

    group.bench_function("prime", |bencher| {
        let runtime: Arc<dyn Runtime> = Arc::new(PrimeRuntime::new(CORES).expect("prime"));
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let (elapsed, _) = run_sustained(&runtime, window, target_per_sec);
                total += elapsed;
            }
            total
        });
    });

    group.finish();
}

fn bench_fanin(criterion: &mut Criterion, name: &str, producers: usize, per_thread: usize) {
    let mut group = criterion.benchmark_group(name);
    configure_group(&mut group);
    group.throughput(Throughput::Elements((producers * per_thread) as u64));

    group.bench_function("tokio_per_core", |bencher| {
        let runtime: Arc<dyn Runtime> =
            Arc::new(TokioPerCoreRuntime::new(CORES).expect("tokio_per_core"));
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                total += run_fanin(&runtime, producers, per_thread);
            }
            total
        });
    });

    group.bench_function("prime", |bencher| {
        let runtime: Arc<dyn Runtime> = Arc::new(PrimeRuntime::new(CORES).expect("prime"));
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                total += run_fanin(&runtime, producers, per_thread);
            }
            total
        });
    });

    group.finish();
}

fn benches(criterion: &mut Criterion) {
    bench_burst(criterion, "spawn_burst_1k", 1_000);
    bench_burst(criterion, "spawn_burst_10k", 10_000);
    bench_sustained(criterion, "spawn_sustained_10k_per_sec", 10_000);
    bench_fanin(criterion, "spawn_fanin_1", 1, 1_000);
    // fanin_{2,4,8} previously hung at 100% CPU because `Arc<dyn Runtime>`
    // shared one `Producer` across producer threads and `try_send` is
    // SPSC-only — concurrent calls on a single lane raced on head/tail
    // atomics. Bug B fix: `CoreShardHandle::dispatch_send` now calls
    // `try_send_mpsc`, which lazily assigns each thread its own SPSC
    // lane via a thread-local cache. The 1024-slot default lane count
    // sized at `core_shard::launch` accommodates fan-in up to 1023
    // distinct producer threads per shard.
    bench_fanin(criterion, "spawn_fanin_2", 2, 1_000);
    bench_fanin(criterion, "spawn_fanin_4", 4, 1_000);
    bench_fanin(criterion, "spawn_fanin_8", 8, 1_000);
}

criterion_group!(spawn_burst, benches);
criterion_main!(spawn_burst);
