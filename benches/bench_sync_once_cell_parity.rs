//! once-cell parity baseline — A3.a of the tokio-parity plan.
//!
//! two workloads mirroring real proxima usage:
//!
//! 1. **cold init race** (`client/handle.rs:22,62` pattern) — 100
//!    concurrent tokio tasks racing `get_or_try_init` (or `get_or_init`
//!    for the sync floor) on a cold `OnceCell` where the init body sleeps
//!    500µs. each criterion iteration = fresh cell + 100 tasks + drain.
//!    three arms: `proxima::sync::OnceCell` (the user-visible surface,
//!    backed by async-lock), `tokio::sync::OnceCell`, `once_cell::sync::OnceCell`
//!    (sync floor — single-thread init, no race; measures the init cost
//!    without the async wait-queue).
//!
//! 2. **hot warm-get** (`huffman.rs:249,283` pattern) — 1M `get().unwrap()`
//!    iterations on a pre-initialized cell. huffman.rs uses `std::sync::OnceLock`
//!    rather than `tokio::sync::OnceCell`; we bench all three here to
//!    confirm the async-crate warm-get is at-parity with the sync floor,
//!    which the hot HPACK path needs.
//!
//! run:
//! ```bash
//! cargo bench -p proxima --features sync-wrappers --bench bench_sync_once_cell_parity
//! cargo bench -p proxima --features sync-wrappers --bench bench_sync_once_cell_parity -- warm_get
//! ```

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
use tokio::runtime::Builder as TokioBuilder;

const RACE_TASKS: usize = 100;
const INIT_SLEEP_US: u64 = 500;
const WARM_GET_ITERS: u64 = 1_000_000;

fn current_thread_runtime() -> tokio::runtime::Runtime {
    TokioBuilder::new_current_thread()
        .enable_all()
        .build()
        .expect("current thread runtime")
}

// ---------- workload 1: cold init race (client/handle.rs pattern) ----------
//
// 100 tasks race get_or_try_init on a cold cell. init body sleeps 500µs,
// then returns Ok(42u64). only one init body should run; the rest wait
// for it. each criterion iteration creates a fresh cell so every sample
// starts cold.
//
// once_cell::sync arm: the sync OnceLock can't race async init. instead we
// measure single-thread get_or_init to establish a sync floor — the cost
// of init alone without any async wait-queue overhead.

fn bench_workload1_cold_init_race(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("once_cell_parity_w1_cold_init_race");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("proxima_sync_once_cell", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let cell = Arc::new(proxima::sync::OnceCell::<u64>::new());
                let mut handles = Vec::with_capacity(RACE_TASKS);

                for _ in 0..RACE_TASKS {
                    let cell = cell.clone();
                    handles.push(tokio::spawn(async move {
                        let value = cell
                            .get_or_try_init(|| async {
                                tokio::time::sleep(Duration::from_micros(INIT_SLEEP_US)).await;
                                Ok::<u64, ()>(42)
                            })
                            .await
                            .expect("init");
                        std::hint::black_box(*value);
                    }));
                }

                for handle in handles {
                    handle.await.expect("task join");
                }

                let final_value = cell.get().expect("initialized");
                std::hint::black_box(*final_value);
            });
        });
    });

    group.bench_function("tokio_sync_once_cell", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let cell = Arc::new(tokio::sync::OnceCell::<u64>::new());
                let mut handles = Vec::with_capacity(RACE_TASKS);

                for _ in 0..RACE_TASKS {
                    let cell = cell.clone();
                    handles.push(tokio::spawn(async move {
                        let value = cell
                            .get_or_try_init(|| async {
                                tokio::time::sleep(Duration::from_micros(INIT_SLEEP_US)).await;
                                Ok::<u64, ()>(42)
                            })
                            .await
                            .expect("init");
                        std::hint::black_box(*value);
                    }));
                }

                for handle in handles {
                    handle.await.expect("task join");
                }

                let final_value = cell.get().expect("initialized");
                std::hint::black_box(*final_value);
            });
        });
    });

    // sync floor: no concurrent async init. single-thread init via get_or_init
    // measures the pure init cost (500µs sleep) without any wait-queue.
    // wall time should equal ~500µs — confirms the sync path is purely init-bound.
    group.bench_function("once_cell_sync", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let cell = once_cell::sync::OnceCell::<u64>::new();

                let value = cell.get_or_init(|| {
                    std::thread::sleep(Duration::from_micros(INIT_SLEEP_US));
                    42u64
                });
                std::hint::black_box(*value);

                let final_value = cell.get().expect("initialized");
                std::hint::black_box(*final_value);
            });
        });
    });

    group.finish();
}

// ---------- workload 2: hot warm-get (huffman.rs pattern) ----------
//
// single-threaded. 1M get().unwrap() calls on a pre-initialized cell.
// criterion::Throughput::Elements(1_000_000) so the report shows per-op
// throughput — compare with the expected ~1ns/op for a pure atomic load.
//
// huffman.rs uses std::sync::OnceLock, not tokio::sync::OnceCell. we
// bench all three to confirm the async crate's warm-get stays at-parity
// with the sync floor. regression here would be visible on every HPACK
// header decode.

fn bench_workload2_hot_warm_get(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("once_cell_parity_w2_hot_warm_get");
    group.throughput(Throughput::Elements(WARM_GET_ITERS));
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("proxima_sync_once_cell", |bench| {
        let runtime = current_thread_runtime();
        let cell = proxima::sync::OnceCell::<u64>::new();
        runtime.block_on(async {
            cell.get_or_init(|| async { 42u64 }).await;
        });

        bench.iter(|| {
            runtime.block_on(async {
                for _ in 0..WARM_GET_ITERS {
                    let value = cell.get().expect("initialized");
                    std::hint::black_box(*value);
                }
            });
        });
    });

    group.bench_function("tokio_sync_once_cell", |bench| {
        let runtime = current_thread_runtime();
        let cell = tokio::sync::OnceCell::<u64>::new();
        runtime.block_on(async {
            cell.get_or_init(|| async { 42u64 }).await;
        });

        bench.iter(|| {
            runtime.block_on(async {
                for _ in 0..WARM_GET_ITERS {
                    let value = cell.get().expect("initialized");
                    std::hint::black_box(*value);
                }
            });
        });
    });

    group.bench_function("once_cell_sync", |bench| {
        let runtime = current_thread_runtime();
        let cell = once_cell::sync::OnceCell::<u64>::new();
        cell.get_or_init(|| 42u64);

        bench.iter(|| {
            runtime.block_on(async {
                for _ in 0..WARM_GET_ITERS {
                    let value = cell.get().expect("initialized");
                    std::hint::black_box(*value);
                }
            });
        });
    });

    group.finish();
}

criterion_group!(
    bench_once_cell_parity_workloads,
    bench_workload1_cold_init_race,
    bench_workload2_hot_warm_get,
);
criterion_main!(bench_once_cell_parity_workloads);
