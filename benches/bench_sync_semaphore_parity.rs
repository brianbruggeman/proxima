//! semaphore parity baseline — A2.a of the tokio-parity plan.
//!
//! one workload: bounded concurrency over slow workers (orchestrator-style).
//! harness only — numbers come in the next step; see
//! discipline-tokio-parity.md for the log.
//!
//! context: Semaphore has zero production call sites in proxima today.
//! `orchestrator.rs:690` explicitly notes it does NOT use Semaphore —
//! concurrency is bounded by `AtomicUsize in_flight + concurrency_cap`.
//! this bench is forward-looking parity, not a perf-critical comparison.
//!
//! fairness (max-wait minus min-wait across the 8 tasks) is load-bearing
//! correctness for bounded concurrency but hard to capture inside criterion's
//! `iter` model without per-task timing hooks. omitted here; a dedicated
//! fairness probe could use `iter_custom` with per-task instants if it
//! becomes critical.

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
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use tokio::runtime::Builder as TokioBuilder;

const WORKER_TASKS: usize = 8;
const ROUNDS_PER_TASK: usize = 10;
const WORKER_RTT_MS: u64 = 1;

const PERMIT_COUNTS: &[u64] = &[1, 8, 64];

fn current_thread_runtime() -> tokio::runtime::Runtime {
    TokioBuilder::new_current_thread()
        .enable_all()
        .build()
        .expect("current thread runtime")
}

// ---------- workload: bounded concurrency over slow workers ----------
//
// N-permit semaphore; 8 tasks race to acquire 1 permit each, hold 1ms
// (simulated upstream RTT), release; ROUNDS_PER_TASK rounds per task
// per criterion iteration.
//
// three arms:
//   proxima_sync_semaphore — async_lock::Semaphore re-export (current shape)
//   tokio_sync_semaphore   — tokio::sync::Semaphore baseline
//   atomic_usize_cap       — the orchestrator pattern: AtomicUsize counter +
//                            bounded spin with yield_now; no semaphore at all

fn bench_semaphore_bounded_concurrency(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("semaphore_bounded_concurrency");
    group.measurement_time(Duration::from_secs(5));

    for &permits in PERMIT_COUNTS {
        group.throughput(Throughput::Elements(permits));

        group.bench_with_input(
            BenchmarkId::new("proxima_sync_semaphore", permits),
            &permits,
            |bench, &permits| {
                let runtime = current_thread_runtime();
                bench.iter(|| {
                    runtime.block_on(async {
                        let sem = Arc::new(proxima::sync::Semaphore::new(permits as usize));
                        let mut handles = Vec::with_capacity(WORKER_TASKS);
                        for _ in 0..WORKER_TASKS {
                            let sem = sem.clone();
                            handles.push(tokio::spawn(async move {
                                for _ in 0..ROUNDS_PER_TASK {
                                    let _guard = sem.acquire().await;
                                    tokio::time::sleep(Duration::from_millis(WORKER_RTT_MS)).await;
                                }
                            }));
                        }
                        for handle in handles {
                            handle.await.expect("task join");
                        }
                    });
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("tokio_sync_semaphore", permits),
            &permits,
            |bench, &permits| {
                let runtime = current_thread_runtime();
                bench.iter(|| {
                    runtime.block_on(async {
                        let sem = Arc::new(tokio::sync::Semaphore::new(permits as usize));
                        let mut handles = Vec::with_capacity(WORKER_TASKS);
                        for _ in 0..WORKER_TASKS {
                            let sem = sem.clone();
                            handles.push(tokio::spawn(async move {
                                for _ in 0..ROUNDS_PER_TASK {
                                    let _permit = sem.acquire().await.expect("semaphore acquire");
                                    tokio::time::sleep(Duration::from_millis(WORKER_RTT_MS)).await;
                                }
                            }));
                        }
                        for handle in handles {
                            handle.await.expect("task join");
                        }
                    });
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("atomic_usize_cap", permits),
            &permits,
            |bench, &permits| {
                let runtime = current_thread_runtime();
                bench.iter(|| {
                    runtime.block_on(async {
                        let in_flight = Arc::new(AtomicUsize::new(0));
                        let cap = permits as usize;
                        let mut handles = Vec::with_capacity(WORKER_TASKS);
                        for _ in 0..WORKER_TASKS {
                            let in_flight = in_flight.clone();
                            handles.push(tokio::spawn(async move {
                                for _ in 0..ROUNDS_PER_TASK {
                                    loop {
                                        let current = in_flight.load(Ordering::Acquire);
                                        if current < cap {
                                            if in_flight
                                                .compare_exchange(
                                                    current,
                                                    current + 1,
                                                    Ordering::AcqRel,
                                                    Ordering::Acquire,
                                                )
                                                .is_ok()
                                            {
                                                break;
                                            }
                                        }
                                        tokio::task::yield_now().await;
                                    }
                                    tokio::time::sleep(Duration::from_millis(WORKER_RTT_MS)).await;
                                    in_flight.fetch_sub(1, Ordering::Release);
                                }
                            }));
                        }
                        for handle in handles {
                            handle.await.expect("task join");
                        }
                    });
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    bench_semaphore_parity_workloads,
    bench_semaphore_bounded_concurrency,
);
criterion_main!(bench_semaphore_parity_workloads);
