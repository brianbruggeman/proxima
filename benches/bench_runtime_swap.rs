//! end-to-end bench against the `Runtime` trait. measures cross-core
//! spawn-and-complete throughput on each backend with identical workload.
//!
//! workload: spawn `N` tiny tasks distributed round-robin across the
//! runtime's `num_cores` workers; each task increments a shared counter
//! and exits. measure wall time until all tasks have completed.
//!
//! this is the smallest viable e2e bench that exercises the full per-core
//! dispatch path: producer-side `spawn_on_core` → cross-core inbox →
//! target-core executor → task body → completion. it does NOT exercise
//! the I/O reactor (HTTP-level swap defers until proxima has a hyper-
//! compatible I/O adapter; for now, hyper's tight tokio coupling blocks
//! the full HTTP/2 swap).
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
use std::time::{Duration, Instant};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima::runtime::{
    CoreId, PrimeRuntime, Runtime, TokioPerCoreRuntime, spawn_on_core_blocking_with,
};

const TASKS_PER_TRIAL: usize = 1000;
const CORES: usize = 2;

fn configure_group<M: criterion::measurement::Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
) {
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
}

fn run_workload(runtime: &Arc<dyn Runtime>) -> Duration {
    let counter = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    for index in 0..TASKS_PER_TRIAL {
        let counter = counter.clone();
        let core = CoreId(index % CORES);
        // Use the blocking helper so producer yields on inbox saturation
        // rather than silently dropping tasks (which used to hang the bench
        // forever on the prime runtime's bounded SPSC lane).
        let _ = spawn_on_core_blocking_with(runtime.as_ref(), core, move || {
            let counter = counter.clone();
            Box::pin(async move {
                counter.fetch_add(1, Ordering::AcqRel);
            })
        });
    }
    while counter.load(Ordering::Acquire) < TASKS_PER_TRIAL {
        std::hint::spin_loop();
    }
    started.elapsed()
}

fn bench_cross_core_spawn(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("runtime_swap_cross_core_spawn");
    configure_group(&mut group);
    group.throughput(Throughput::Elements(TASKS_PER_TRIAL as u64));

    group.bench_function("tokio_per_core", |bencher| {
        let runtime: Arc<dyn Runtime> =
            Arc::new(TokioPerCoreRuntime::new(CORES).expect("tokio_per_core"));
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                total += run_workload(&runtime);
            }
            total
        });
    });

    group.bench_function("proxima_runtime", |bencher| {
        let runtime: Arc<dyn Runtime> =
            Arc::new(PrimeRuntime::new(CORES).expect("proxima_runtime"));
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                total += run_workload(&runtime);
            }
            total
        });
    });

    group.finish();
}

criterion_group!(benches, bench_cross_core_spawn);
criterion_main!(benches);
