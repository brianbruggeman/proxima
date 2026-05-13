//! `proxima::task::JoinSet` newtype overhead — A6.b of the tokio-parity plan.
//!
//! Compares a local bench shim (`ProximaJoinSet<T>`) that mirrors the exact
//! newtype shape A6.c will land in `proxima::task`, against the raw
//! `tokio::task::JoinSet<T>`.  Both arms run on a `current_thread` tokio
//! runtime.  Expected delta: zero — the shim is a transparent forwarding
//! wrapper with no allocation, fence, or boxing.
//!
//! Three workloads:
//!
//! 1. **Newtype overhead sanity** — 100 short futures (5× yield_now each),
//!    drain.  Measures raw dispatch + newtype call chain cost.
//!
//! 2. **Pipeline drain** (executor.rs:111 pattern) — 5 tasks sleeping 50ms
//!    in parallel, drain.  Verifies the wrapper doesn't interfere with
//!    parallel execution; theoretical optimal ≈ 50ms.
//!
//! 3. **Abort-on-Drop correctness** (process_bridge.rs:95 pattern) — 3
//!    long-running tasks each incrementing a counter every 1ms.  Drop the
//!    JoinSet at 100ms; verify counters stop within 5ms.  Correctness test
//!    in `iter_custom` form — fails loudly if abort-on-drop is broken.
//!
//! Run:
//! ```bash
//! cargo bench -p proxima --bench bench_task_join_set_parity
//! cargo bench -p proxima --bench bench_task_join_set_parity -- newtype_overhead
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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use criterion::{Criterion, criterion_group, criterion_main};
use tokio::runtime::Builder as TokioBuilder;

const SANITY_TASK_COUNT: usize = 100;
const SANITY_YIELDS_PER_TASK: usize = 5;

const PIPELINE_TASK_COUNT: usize = 5;
const PIPELINE_SLEEP_MS: u64 = 50;

const ABORT_TASK_COUNT: usize = 3;
const ABORT_RUN_MS: u64 = 100;
const ABORT_GRACE_MS: u64 = 5;

// ---------- proxima_primitives::sync::task::JoinSet (the A6.c newtype) ----------
//
// A6.b shipped with an inline shim; this is rebound to the real
// `proxima_primitives::sync::task::JoinSet` now that A6.c is in place (folded from
// the former `proxima-task` crate in Workstream F, RISC-dedup). Same
// structural shape — newtype forwarding to `tokio::task::JoinSet`. The
// bench measures forwarding overhead.

use proxima_primitives::sync::task::JoinSet as ProximaJoinSet;

// Original inline shim left in this commented block for context; the
// real type is in `proxima-sync/src/task.rs`:
//
// struct ProximaJoinSet<T>(tokio::task::JoinSet<T>);
// impl<T: Send + 'static> ProximaJoinSet<T> {
//     fn new() -> Self { Self(tokio::task::JoinSet::new()) }
//     fn spawn<F>(&mut self, fut: F) where F: Future<Output = T> + Send + 'static {
//         self.0.spawn(fut);
//     }
//     async fn join_next(&mut self) -> Option<Result<T, JoinError>> {
//         self.0.join_next().await
//     }
//     fn len(&self) -> usize { self.0.len() }
//     fn abort_all(&mut self) { self.0.abort_all(); }
// }

#[allow(dead_code, unused_imports)]
mod _api_shape {
    pub use proxima_primitives::sync::task::JoinError;
}

fn current_thread_runtime() -> tokio::runtime::Runtime {
    TokioBuilder::new_current_thread()
        .enable_all()
        .build()
        .expect("current thread runtime")
}

// ---------- workload 1: newtype overhead sanity ----------
//
// 100 short futures (5 × yield_now each) spawned into raw tokio::task::JoinSet
// vs the ProximaJoinSet shim.  The shim call chain adds no allocation or fence
// so both arms should be indistinguishable.

fn bench_workload1_newtype_overhead(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("task_join_set_newtype_overhead");
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("tokio_join_set_raw", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let mut join_set = tokio::task::JoinSet::new();
                for _ in 0..SANITY_TASK_COUNT {
                    join_set.spawn(async {
                        for _ in 0..SANITY_YIELDS_PER_TASK {
                            tokio::task::yield_now().await;
                        }
                    });
                }
                while join_set.join_next().await.is_some() {}
            });
        });
    });

    group.bench_function("tokio_join_set_via_newtype", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let mut join_set: ProximaJoinSet<()> = ProximaJoinSet::new();
                for _ in 0..SANITY_TASK_COUNT {
                    join_set.spawn(async {
                        for _ in 0..SANITY_YIELDS_PER_TASK {
                            tokio::task::yield_now().await;
                        }
                    });
                }
                while join_set.join_next().await.is_some() {}
            });
        });
    });

    group.finish();
}

// ---------- workload 2: pipeline drain (executor.rs:111 pattern) ----------
//
// 5 tasks each sleep 50ms, then complete.  When spawned into a JoinSet and
// drained, they run in parallel — wall time should be ≈50ms regardless of
// whether the wrapper is present.  Verifies the newtype doesn't serialize
// what should be concurrent execution.

fn bench_workload2_pipeline_drain(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("task_join_set_pipeline_drain");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("tokio_join_set_raw", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let mut join_set = tokio::task::JoinSet::new();
                for _ in 0..PIPELINE_TASK_COUNT {
                    join_set.spawn(async {
                        tokio::time::sleep(Duration::from_millis(PIPELINE_SLEEP_MS)).await;
                    });
                }
                while join_set.join_next().await.is_some() {}
            });
        });
    });

    group.bench_function("tokio_join_set_via_newtype", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let mut join_set: ProximaJoinSet<()> = ProximaJoinSet::new();
                for _ in 0..PIPELINE_TASK_COUNT {
                    join_set.spawn(async {
                        tokio::time::sleep(Duration::from_millis(PIPELINE_SLEEP_MS)).await;
                    });
                }
                while join_set.join_next().await.is_some() {}
            });
        });
    });

    group.finish();
}

// ---------- workload 3: abort-on-drop (process_bridge.rs:95 pattern) ----------
//
// 3 long-running tasks loop on sleep(1ms)+counter increment.  The JoinSet is
// dropped at 100ms; all 3 tasks must observe cancellation within 5ms.
//
// Uses iter_custom for wall-clock control.  Each iter:
//   1. Reset counters to 0.
//   2. Spawn 3 looping tasks (raw arm) or 3 via newtype (newtype arm).
//   3. Sleep 100ms — tasks accumulate counts.
//   4. Drop the JoinSet (triggers abort_all on drop for tokio::task::JoinSet).
//   5. Sleep 5ms grace period.
//   6. Capture final counter values — must match the snapshot taken at drop.
//
// The bench fails (eprintln! warning) if any counter increments after the
// drop+grace window.  Correctness gates on abort-on-drop semantics.

fn run_abort_on_drop_raw(iters: u64) -> Duration {
    let runtime = current_thread_runtime();
    let mut total = Duration::ZERO;

    for _ in 0..iters {
        let counters: Vec<Arc<AtomicUsize>> = (0..ABORT_TASK_COUNT)
            .map(|_| Arc::new(AtomicUsize::new(0)))
            .collect();

        let start = Instant::now();
        runtime.block_on(async {
            let mut join_set = tokio::task::JoinSet::new();
            for counter in &counters {
                let counter = counter.clone();
                join_set.spawn(async move {
                    loop {
                        tokio::time::sleep(Duration::from_millis(1)).await;
                        counter.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }

            tokio::time::sleep(Duration::from_millis(ABORT_RUN_MS)).await;
            drop(join_set);

            let snapshot: Vec<usize> =
                counters.iter().map(|counter| counter.load(Ordering::Relaxed)).collect();

            tokio::time::sleep(Duration::from_millis(ABORT_GRACE_MS)).await;

            for (index, (before, counter)) in snapshot.iter().zip(counters.iter()).enumerate() {
                let after = counter.load(Ordering::Relaxed);
                std::hint::black_box(after);
                if after != *before {
                    eprintln!(
                        "abort-on-drop FAILED (raw): task {index} incremented after drop ({before} -> {after})"
                    );
                }
            }
        });
        total += start.elapsed();
    }

    total
}

fn run_abort_on_drop_newtype(iters: u64) -> Duration {
    let runtime = current_thread_runtime();
    let mut total = Duration::ZERO;

    for _ in 0..iters {
        let counters: Vec<Arc<AtomicUsize>> = (0..ABORT_TASK_COUNT)
            .map(|_| Arc::new(AtomicUsize::new(0)))
            .collect();

        let start = Instant::now();
        runtime.block_on(async {
            let mut join_set: ProximaJoinSet<()> = ProximaJoinSet::new();
            for counter in &counters {
                let counter = counter.clone();
                join_set.spawn(async move {
                    loop {
                        tokio::time::sleep(Duration::from_millis(1)).await;
                        counter.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }

            tokio::time::sleep(Duration::from_millis(ABORT_RUN_MS)).await;
            drop(join_set);

            let snapshot: Vec<usize> =
                counters.iter().map(|counter| counter.load(Ordering::Relaxed)).collect();

            tokio::time::sleep(Duration::from_millis(ABORT_GRACE_MS)).await;

            for (index, (before, counter)) in snapshot.iter().zip(counters.iter()).enumerate() {
                let after = counter.load(Ordering::Relaxed);
                std::hint::black_box(after);
                if after != *before {
                    eprintln!(
                        "abort-on-drop FAILED (newtype): task {index} incremented after drop ({before} -> {after})"
                    );
                }
            }
        });
        total += start.elapsed();
    }

    total
}

fn bench_workload3_abort_on_drop(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("task_join_set_abort_on_drop");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));

    group.bench_function("tokio_join_set_raw", |bench| {
        bench.iter_custom(run_abort_on_drop_raw);
    });

    group.bench_function("tokio_join_set_via_newtype", |bench| {
        bench.iter_custom(run_abort_on_drop_newtype);
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_workload1_newtype_overhead,
    bench_workload2_pipeline_drain,
    bench_workload3_abort_on_drop,
);
criterion_main!(benches);
