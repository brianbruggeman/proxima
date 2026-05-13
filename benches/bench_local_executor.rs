//! micro-bench for the proxima LocalExecutor vs tokio LocalSet baseline.
//!
//! incumbents (versions pinned in Cargo.toml):
//!   - tokio::task::LocalSet 1.x — single-thread async task executor; design
//!     point is spawn_local + cooperative scheduling on one runtime thread
//!
//! groups (and design-favors per workload):
//!   - local_exec_ready_throughput   design-favors: incumbent
//!     (spawn N + drain — exactly LocalSet's design point: spawn_local +
//!     join handle awaiting on the same runtime thread)
//!   - local_exec_yield_pingpong     design-favors: incumbent
//!     (YieldNTimes future driven by LocalSet's cooperative scheduling)
//!
//! required-features: runtime-prime-executor, runtime-prime-inbox-alloc, runtime-tokio.

#![cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
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

use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima::runtime::prime::core::local_executor::LocalExecutor;

const ITEMS_PER_TRIAL: usize = 10_000;

fn configure_group<M: criterion::measurement::Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
) {
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
}

// design-favors: incumbent — LocalSet's spawn_local + JoinHandle::await is
// the canonical single-thread async task pattern. proxima skips JoinHandle
// bookkeeping (spawn returns nothing awaitable) — that's the structural
// difference. A win here engages LocalSet on its design point.
fn bench_ready_throughput(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("local_exec_ready_throughput");
    configure_group(&mut group);
    group.throughput(Throughput::Elements(ITEMS_PER_TRIAL as u64));

    group.bench_function("proxima", |bencher| {
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let executor = LocalExecutor::new();
                let counter = Rc::new(Cell::new(0_u32));
                for _ in 0..ITEMS_PER_TRIAL {
                    let counter = counter.clone();
                    executor.spawn_local(async move {
                        counter.set(counter.get() + 1);
                    });
                }
                let started = Instant::now();
                executor.block_on(async {});
                total += started.elapsed();
                assert_eq!(counter.get(), ITEMS_PER_TRIAL as u32);
            }
            total
        });
    });

    group.bench_function("tokio_localset", |bencher| {
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .build()
                    .unwrap();
                let local_set = tokio::task::LocalSet::new();
                let counter = Rc::new(Cell::new(0_u32));
                local_set.block_on(&runtime, async {
                    let started = Instant::now();
                    let mut joins = Vec::with_capacity(ITEMS_PER_TRIAL);
                    for _ in 0..ITEMS_PER_TRIAL {
                        let counter = counter.clone();
                        joins.push(tokio::task::spawn_local(async move {
                            counter.set(counter.get() + 1);
                        }));
                    }
                    for join in joins {
                        join.await.unwrap();
                    }
                    total += started.elapsed();
                });
                assert_eq!(counter.get(), ITEMS_PER_TRIAL as u32);
            }
            total
        });
    });

    group.finish();
}

struct YieldNTimes {
    remaining: u32,
}

impl Future for YieldNTimes {
    type Output = ();
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        if this.remaining == 0 {
            Poll::Ready(())
        } else {
            this.remaining -= 1;
            context.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

// design-favors: incumbent — YieldNTimes driven by the LocalSet polling loop
// IS LocalSet's cooperative scheduling design. Both executors poll a future
// that wake_by_ref() schedules a re-poll. proxima's flat ready-queue is the
// structural alternative to tokio's notify-driven queue.
fn bench_yield_pingpong(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("local_exec_yield_pingpong");
    configure_group(&mut group);
    group.throughput(Throughput::Elements(ITEMS_PER_TRIAL as u64));

    group.bench_function("proxima", |bencher| {
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let executor = LocalExecutor::new();
                let started = Instant::now();
                executor.block_on(YieldNTimes {
                    remaining: ITEMS_PER_TRIAL as u32,
                });
                total += started.elapsed();
            }
            total
        });
    });

    group.bench_function("tokio_localset", |bencher| {
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .build()
                    .unwrap();
                let local_set = tokio::task::LocalSet::new();
                let started = Instant::now();
                local_set.block_on(&runtime, async {
                    YieldNTimes {
                        remaining: ITEMS_PER_TRIAL as u32,
                    }
                    .await;
                });
                total += started.elapsed();
            }
            total
        });
    });

    group.finish();
}

criterion_group!(benches, bench_ready_throughput, bench_yield_pingpong);
criterion_main!(benches);
