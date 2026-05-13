//! `proxima::time` vs `tokio::time` shootout.
//!
//! Compares `proxima::time::{sleep, timeout, interval}` (futures-timer backed)
//! against their `tokio::time` analogues on representative workloads. Both arms
//! run inside a `tokio::runtime::current_thread` so the executor is fixed and
//! only the timer primitive varies.
//!
//! Groups:
//! - sleep_short: 100 × sleep(1ms) — per-sleep overhead and jitter
//! - timeout_success: 1024 × timeout(50ms, immediate) — setup+teardown cost when timer never fires
//! - timeout_elapsed: 100 × timeout(1ms, sleep(5ms)) — elapsed-path cost when timer fires
//! - interval_drift: 100 ticks at 1ms cadence — total-elapsed fidelity
//! - timeout_concurrent_n10000: 10K concurrent timeouts, mixed Ready+Pending inner futures
//! - sleep_concurrent_n10000: 10K concurrent sleeps — tokio time-wheel home turf
//! - interval_concurrent_n1000_x10_ticks: 1K concurrent 1ms intervals × 10 ticks — tokio time-wheel home turf
//!
//! Run:
//! ```bash
//! cargo bench -p proxima --bench bench_sync_time
//! cargo bench -p proxima --bench bench_sync_time -- sleep_short
//! cargo bench -p proxima --bench bench_sync_time -- concurrent
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

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use futures::StreamExt;
use tokio::runtime::Builder as TokioBuilder;

const SLEEP_ITERS: usize = 100;
const SLEEP_DURATION_MS: u64 = 1;
const TIMEOUT_SUCCESS_ITERS: usize = 1024;
const TIMEOUT_SUCCESS_BUDGET_MS: u64 = 50;
const TIMEOUT_ELAPSED_ITERS: usize = 100;
const TIMEOUT_ELAPSED_BUDGET_MS: u64 = 1;
const TIMEOUT_ELAPSED_FUTURE_MS: u64 = 5;
const INTERVAL_TICKS: usize = 100;
const INTERVAL_PERIOD_MS: u64 = 1;
const CONCURRENT_N: usize = 10_000;
const CONCURRENT_BUDGET_MS: u64 = 50;
const SLEEP_CONCURRENT_N: usize = 10_000;
const INTERVAL_CONCURRENT_N: usize = 1_000;
const INTERVAL_CONCURRENT_TICKS: usize = 10;

fn current_thread_runtime() -> tokio::runtime::Runtime {
    TokioBuilder::new_current_thread()
        .enable_all()
        .build()
        .expect("current thread runtime")
}

// ---------- sleep_short ----------

fn bench_sleep_short(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("sleep_short_1ms_x100");
    group.throughput(Throughput::Elements(SLEEP_ITERS as u64));

    // design-favors: proxima (single timer; futures-timer helper thread shines)
    group.bench_function("tokio", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                for _ in 0..SLEEP_ITERS {
                    tokio::time::sleep(Duration::from_millis(SLEEP_DURATION_MS)).await;
                }
            });
        });
    });

    // design-favors: proxima (single timer; futures-timer helper thread shines)
    group.bench_function("proxima", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                for _ in 0..SLEEP_ITERS {
                    proxima::time::sleep(Duration::from_millis(SLEEP_DURATION_MS)).await;
                }
            });
        });
    });

    group.finish();
}

// ---------- timeout_success ----------

fn bench_timeout_success(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("timeout_success_50ms_x1024");
    group.throughput(Throughput::Elements(TIMEOUT_SUCCESS_ITERS as u64));

    // design-favors: proxima (Ready-first-poll regime; E.2.1 short-circuit fires)
    group.bench_function("tokio", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                for _ in 0..TIMEOUT_SUCCESS_ITERS {
                    let _ = tokio::time::timeout(
                        Duration::from_millis(TIMEOUT_SUCCESS_BUDGET_MS),
                        async { 42_u32 },
                    )
                    .await;
                }
            });
        });
    });

    group.bench_function("proxima", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                for _ in 0..TIMEOUT_SUCCESS_ITERS {
                    let _ = proxima::time::timeout(
                        Duration::from_millis(TIMEOUT_SUCCESS_BUDGET_MS),
                        async { 42_u32 },
                    )
                    .await;
                }
            });
        });
    });

    group.finish();
}

// ---------- timeout_elapsed ----------

fn bench_timeout_elapsed(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("timeout_elapsed_1ms_over_5ms_x100");
    group.throughput(Throughput::Elements(TIMEOUT_ELAPSED_ITERS as u64));

    // design-favors: neutral (partial) — timer fire exercised but 1 at a time
    group.bench_function("tokio", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                for _ in 0..TIMEOUT_ELAPSED_ITERS {
                    let _ = tokio::time::timeout(
                        Duration::from_millis(TIMEOUT_ELAPSED_BUDGET_MS),
                        tokio::time::sleep(Duration::from_millis(TIMEOUT_ELAPSED_FUTURE_MS)),
                    )
                    .await;
                }
            });
        });
    });

    group.bench_function("proxima", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                for _ in 0..TIMEOUT_ELAPSED_ITERS {
                    let _ = proxima::time::timeout(
                        Duration::from_millis(TIMEOUT_ELAPSED_BUDGET_MS),
                        proxima::time::sleep(Duration::from_millis(TIMEOUT_ELAPSED_FUTURE_MS)),
                    )
                    .await;
                }
            });
        });
    });

    group.finish();
}

// ---------- interval_drift ----------

fn bench_interval_drift(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("interval_drift_1ms_x100");
    group.throughput(Throughput::Elements(INTERVAL_TICKS as u64));

    // design-favors: proxima (single interval; E.2.2 mechanism targets this)
    group.bench_function("tokio", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let mut interval = tokio::time::interval(Duration::from_millis(INTERVAL_PERIOD_MS));
                for _ in 0..INTERVAL_TICKS {
                    interval.tick().await;
                }
            });
        });
    });

    // design-favors: proxima (single interval; E.2.2 mechanism targets this)
    group.bench_function("proxima", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let mut interval =
                    proxima::time::interval(Duration::from_millis(INTERVAL_PERIOD_MS));
                for _ in 0..INTERVAL_TICKS {
                    interval.next().await;
                }
            });
        });
    });

    group.finish();
}

// ---------- timeout_concurrent_n10000 ----------

/// Future that returns `Pending` for the first `needed` polls, then `Ready`.
///
/// Used to simulate inner futures that require a small number of wakeup cycles
/// before resolving — represents the 50% Pending-half of the concurrent mix.
struct CountdownFuture {
    remaining: Arc<AtomicUsize>,
}

impl Future for CountdownFuture {
    type Output = ();

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
        let prev = self
            .remaining
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |count| {
                Some(count.saturating_sub(1))
            });
        match prev {
            Ok(0) | Ok(1) => Poll::Ready(()),
            _ => {
                context.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }
}

fn make_mixed_tokio_futures(
    count: usize,
) -> Vec<impl Future<Output = Result<(), tokio::time::error::Elapsed>>> {
    (0..count)
        .map(|index| {
            let duration = Duration::from_millis(CONCURRENT_BUDGET_MS);
            if index % 2 == 0 {
                // design-favors: incumbent — half of futures need timer allocation (Pending first poll)
                let countdown = Arc::new(AtomicUsize::new(3));
                tokio::time::timeout(
                    duration,
                    CountdownFuture {
                        remaining: countdown,
                    },
                )
            } else {
                // immediately Ready — E.2.1-equivalent: tokio still allocates a timer entry here
                let countdown = Arc::new(AtomicUsize::new(0));
                tokio::time::timeout(
                    duration,
                    CountdownFuture {
                        remaining: countdown,
                    },
                )
            }
        })
        .collect()
}

fn make_mixed_proxima_futures(
    count: usize,
) -> Vec<impl Future<Output = Result<(), proxima::time::Elapsed>>> {
    (0..count)
        .map(|index| {
            let duration = Duration::from_millis(CONCURRENT_BUDGET_MS);
            if index % 2 == 0 {
                // design-favors: incumbent — Pending first poll triggers Delay allocation
                let countdown = Arc::new(AtomicUsize::new(3));
                proxima::time::timeout(
                    duration,
                    CountdownFuture {
                        remaining: countdown,
                    },
                )
            } else {
                // immediately Ready — E.2.1 short-circuit fires, no Delay allocated
                let countdown = Arc::new(AtomicUsize::new(0));
                proxima::time::timeout(
                    duration,
                    CountdownFuture {
                        remaining: countdown,
                    },
                )
            }
        })
        .collect()
}

fn bench_timeout_concurrent_n10000(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("timeout_concurrent_n10000");
    group.throughput(Throughput::Elements(CONCURRENT_N as u64));

    // design-favors: incumbent (tokio's time-wheel home turf — N=10K concurrent timers)
    group.bench_function("tokio", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let futures = make_mixed_tokio_futures(CONCURRENT_N);
                let results = futures::future::join_all(futures).await;
                assert_eq!(results.len(), CONCURRENT_N);
            });
        });
    });

    // design-favors: incumbent (tokio's time-wheel home turf — N=10K concurrent timers)
    group.bench_function("proxima", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let futures = make_mixed_proxima_futures(CONCURRENT_N);
                let results = futures::future::join_all(futures).await;
                assert_eq!(results.len(), CONCURRENT_N);
            });
        });
    });

    group.finish();
}

// ---------- sleep_concurrent_n10000 ----------
//
// Durations: index i → 1 + (i % 50) ms, spread across [1ms, 50ms] deterministically.

fn bench_sleep_concurrent_n10000(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("sleep_concurrent_n10000");
    group.throughput(Throughput::Elements(SLEEP_CONCURRENT_N as u64));

    // design-favors: incumbent (tokio time-wheel home turf — 10K concurrent timers)
    group.bench_function("tokio", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let futs: Vec<_> = (0..SLEEP_CONCURRENT_N)
                    .map(|index| {
                        let ms = 1 + (index % 50) as u64;
                        tokio::time::sleep(Duration::from_millis(ms))
                    })
                    .collect();
                futures::future::join_all(futs).await;
            });
        });
    });

    // design-favors: incumbent (tokio time-wheel home turf — 10K concurrent timers)
    group.bench_function("proxima", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let futs: Vec<_> = (0..SLEEP_CONCURRENT_N)
                    .map(|index| {
                        let ms = 1 + (index % 50) as u64;
                        proxima::time::sleep(Duration::from_millis(ms))
                    })
                    .collect();
                futures::future::join_all(futs).await;
            });
        });
    });

    group.finish();
}

// ---------- interval_concurrent_n1000_x10_ticks ----------

fn bench_interval_concurrent_n1000_x10_ticks(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("interval_concurrent_n1000_x10_ticks");
    group.throughput(Throughput::Elements(
        (INTERVAL_CONCURRENT_N * INTERVAL_CONCURRENT_TICKS) as u64,
    ));

    // design-favors: incumbent (tokio time-wheel home turf — 1K concurrent intervals)
    group.bench_function("tokio", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let tasks: Vec<_> = (0..INTERVAL_CONCURRENT_N)
                    .map(|_| {
                        tokio::spawn(async move {
                            let mut interval =
                                tokio::time::interval(Duration::from_millis(INTERVAL_PERIOD_MS));
                            for _ in 0..INTERVAL_CONCURRENT_TICKS {
                                interval.tick().await;
                            }
                        })
                    })
                    .collect();
                for task in tasks {
                    task.await.expect("task");
                }
            });
        });
    });

    // design-favors: incumbent (tokio time-wheel home turf — 1K concurrent intervals)
    group.bench_function("proxima", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let tasks: Vec<_> = (0..INTERVAL_CONCURRENT_N)
                    .map(|_| {
                        tokio::spawn(async move {
                            let mut interval =
                                proxima::time::interval(Duration::from_millis(INTERVAL_PERIOD_MS));
                            for _ in 0..INTERVAL_CONCURRENT_TICKS {
                                interval.next().await;
                            }
                        })
                    })
                    .collect();
                for task in tasks {
                    task.await.expect("task");
                }
            });
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_sleep_short,
    bench_timeout_success,
    bench_timeout_elapsed,
    bench_interval_drift,
    bench_timeout_concurrent_n10000,
    bench_sleep_concurrent_n10000,
    bench_interval_concurrent_n1000_x10_ticks,
);
criterion_main!(benches);
