//! micro-bench for the proxima TimerWheel vs tokio::time baseline.
//!
//! incumbents (versions pinned in Cargo.toml):
//!   - tokio::time 1.x — async timer wheel integrated with the tokio runtime;
//!     design point is sleep_until + .await driven by a runtime poll, with
//!     timers firing lazily during runtime ticks
//!
//! groups (and design-favors per workload):
//!   - timer_register_throughput     design-favors: neither
//!     (registration overhead only, drop-without-drive — symmetric workload)
//!   - timer_drain_throughput        design-favors: incumbent
//!     (tokio_time_joinall arm IS tokio on its design point: sleep_until +
//!     join_all + await driven by the current-thread runtime. proxima's
//!     `advance(horizon)` is the structurally-different counterpart.)
//!   - timer_register_then_cancel    design-favors: prime
//!     (proxima-only; tokio doesn't expose comparable register+cancel)
//!
//! N is `ITEMS_PER_TRIAL`; deadlines are spread modulo bottom-wheel slots so
//! we touch many slots, not just one. tokio baseline uses the current-thread
//! tokio runtime and `tokio::time::sleep_until` for an equivalent shape.
//!
//! required-features: runtime-prime-timer, runtime-tokio.

#![cfg(all(feature = "runtime-prime-timer", feature = "runtime-tokio"))]
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
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima::runtime::prime::core::timer::{Clock, TimerWheel};

const ITEMS_PER_TRIAL: usize = 10_000;

fn configure_group<M: criterion::measurement::Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
) {
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
}

struct CounterClock(Arc<AtomicU64>);
impl Clock for CounterClock {
    fn now(&self) -> u64 {
        self.0.load(Ordering::Acquire)
    }
}

fn proxima_register(wheel: &mut TimerWheel<CounterClock>, count: usize) {
    let waker: std::task::Waker = std::task::Waker::noop().clone();
    for index in 0..count {
        // spread deadlines so they hit many slots; some land in bottom wheel,
        // some in far_future via the cascade boundary.
        let deadline = (index as u64).wrapping_mul(13).wrapping_add(1);
        wheel.register(deadline, waker.clone());
    }
}

// design-favors: neither — both arms measure pure registration; tokio's
// drop-without-drive path skips its design point (lazy fire-during-poll).
// proxima writes directly into the bottom wheel. Symmetric microbench.
fn bench_register_throughput(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("timer_register_throughput");
    configure_group(&mut group);
    group.throughput(Throughput::Elements(ITEMS_PER_TRIAL as u64));

    group.bench_function("proxima", |bencher| {
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut wheel = TimerWheel::new(CounterClock(Arc::new(AtomicU64::new(0))));
                let started = Instant::now();
                proxima_register(&mut wheel, ITEMS_PER_TRIAL);
                total += started.elapsed();
            }
            total
        });
    });

    group.bench_function("tokio_time", |bencher| {
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_time()
                .build()
                .unwrap();
            for _ in 0..iters {
                let started = Instant::now();
                runtime.block_on(async {
                    let base = tokio::time::Instant::now();
                    let mut sleeps = Vec::with_capacity(ITEMS_PER_TRIAL);
                    for index in 0..ITEMS_PER_TRIAL {
                        let deadline = base
                            + Duration::from_micros(
                                (index as u64).wrapping_mul(13).wrapping_add(1),
                            );
                        sleeps.push(Box::pin(tokio::time::sleep_until(deadline)));
                    }
                    // drop sleeps without driving — measures pure registration.
                    drop(sleeps);
                });
                total += started.elapsed();
            }
            total
        });
    });

    group.finish();
}

// design-favors: incumbent — the tokio_time_joinall arm engages tokio's
// design point: N sleep_until futures driven concurrently by the runtime
// until completion. proxima's `advance(horizon)` is a structurally-different
// primitive (explicit O(slots) sweep). The +200x delta is architectural
// shape, not a like-for-like win; tokio's home turf here is the
// sleep_until/await/join_all pattern, which IS being engaged.
fn bench_drain_throughput(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("timer_drain_throughput");
    configure_group(&mut group);
    group.throughput(Throughput::Elements(ITEMS_PER_TRIAL as u64));

    group.bench_function("proxima", |bencher| {
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut wheel = TimerWheel::new(CounterClock(Arc::new(AtomicU64::new(0))));
                proxima_register(&mut wheel, ITEMS_PER_TRIAL);
                // advance well past every registered deadline.
                let horizon = (ITEMS_PER_TRIAL as u64).wrapping_mul(13).wrapping_add(2);
                let started = Instant::now();
                let fired = wheel.advance(horizon);
                total += started.elapsed();
                assert_eq!(fired, ITEMS_PER_TRIAL);
            }
            total
        });
    });

    // tokio doesn't expose a comparable "fire all" drain primitive — its
    // wheel fires lazily when the runtime polls. measure the closest analog:
    // sleep_until + .await on every registered timer in a JoinAll.
    group.bench_function("tokio_time_joinall", |bencher| {
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_time()
                .build()
                .unwrap();
            for _ in 0..iters {
                let started = Instant::now();
                runtime.block_on(async {
                    let base = tokio::time::Instant::now();
                    let mut futures = Vec::with_capacity(ITEMS_PER_TRIAL);
                    for index in 0..ITEMS_PER_TRIAL {
                        let deadline = base
                            + Duration::from_micros(
                                (index as u64).wrapping_mul(13).wrapping_add(1),
                            );
                        futures.push(tokio::time::sleep_until(deadline));
                    }
                    futures::future::join_all(futures).await;
                });
                total += started.elapsed();
            }
            total
        });
    });

    group.finish();
}

// design-favors: prime — proxima-only register+cancel pattern. No comparable
// tokio API (tokio's Sleep cancellation is via dropping the future). The
// O(1) cancel via lazy waker-drop is a proxima-specific invariant.
fn bench_register_then_cancel(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("timer_register_then_cancel");
    configure_group(&mut group);
    group.throughput(Throughput::Elements(ITEMS_PER_TRIAL as u64 * 2));

    group.bench_function("proxima", |bencher| {
        bencher.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            let waker: std::task::Waker = std::task::Waker::noop().clone();
            for _ in 0..iters {
                let mut wheel = TimerWheel::new(CounterClock(Arc::new(AtomicU64::new(0))));
                let mut keys = Vec::with_capacity(ITEMS_PER_TRIAL);
                let started = Instant::now();
                for index in 0..ITEMS_PER_TRIAL {
                    let deadline = (index as u64).wrapping_mul(13).wrapping_add(1);
                    keys.push(wheel.register(deadline, waker.clone()));
                }
                for key in &keys {
                    wheel.cancel(*key);
                }
                total += started.elapsed();
            }
            total
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_register_throughput,
    bench_drain_throughput,
    bench_register_then_cancel,
);
criterion_main!(benches);
