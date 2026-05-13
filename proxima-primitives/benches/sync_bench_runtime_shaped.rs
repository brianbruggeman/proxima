#![allow(clippy::unwrap_used, clippy::expect_used)]

//! R3 + R4 of the runtime-shaped initiative.
//!
//! Compares three arms on a current-thread tokio runtime:
//!
//! - `tokio_direct` — `tokio::sync::Mutex` / `tokio::sync::Notify` —
//!   the workspace's already-imported async-lock-shaped baseline.
//! - `async_lock_direct` — `async_lock::Mutex` / `event_listener` —
//!   the backing crate used by the workspace-default `proxima_primitives::sync::Mutex`
//!   and `proxima_primitives::sync::Notify`. R3's "vs async-lock" Compare-bench
//!   target per discipline.md.
//! - `runtime_shaped_tokio` — `proxima_primitives::sync::runtime_shaped::Mutex<T,
//!   TokioPerCoreRuntime>` / `Notify<TokioPerCoreRuntime>` —
//!   trait-routed via R1's `RuntimeFactory`. Should match
//!   `tokio_direct` within CoV because it forwards to the same
//!   underlying `tokio::sync` primitive.
//!
//! Workloads:
//! - `mutex_uncontended_round_trip` — single locker, 100 lock/unlock.
//! - `mutex_two_serialized_lockers` — two tasks racing 50 locks each.
//! - `notify_park_wake_round_trip` — waiter parks; producer notifies.

use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use proxima_primitives::sync::runtime_shaped::{Mutex as ShapedMutex, Notify as ShapedNotify};
use proxima_runtime::tokio::TokioPerCoreRuntime;

fn current_thread_tokio() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("current-thread runtime")
}

fn bench_mutex_uncontended_round_trip(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("r3_mutex_uncontended_round_trip");
    group.measurement_time(Duration::from_secs(5));
    let runtime = current_thread_tokio();

    group.bench_function("tokio_direct", |bencher| {
        bencher.iter(|| {
            runtime.block_on(async {
                let mutex = tokio::sync::Mutex::new(0u64);
                for index in 0..100u64 {
                    let mut guard = mutex.lock().await;
                    *guard = std::hint::black_box(index);
                }
            });
        });
    });

    group.bench_function("async_lock_direct", |bencher| {
        bencher.iter(|| {
            runtime.block_on(async {
                let mutex = async_lock::Mutex::new(0u64);
                for index in 0..100u64 {
                    let mut guard = mutex.lock().await;
                    *guard = std::hint::black_box(index);
                }
            });
        });
    });

    group.bench_function("runtime_shaped_tokio", |bencher| {
        bencher.iter(|| {
            runtime.block_on(async {
                let mutex: ShapedMutex<u64, TokioPerCoreRuntime> = ShapedMutex::new(0u64);
                for index in 0..100u64 {
                    let mut guard = mutex.lock().await;
                    *guard = std::hint::black_box(index);
                }
            });
        });
    });

    group.finish();
}

fn bench_mutex_two_serialized_lockers(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("r3_mutex_two_serialized_lockers");
    group.measurement_time(Duration::from_secs(5));
    let runtime = current_thread_tokio();

    group.bench_function("tokio_direct", |bencher| {
        bencher.iter(|| {
            runtime.block_on(async {
                let mutex = std::sync::Arc::new(tokio::sync::Mutex::new(0u64));
                let mut handles = Vec::new();
                for _ in 0..2 {
                    let shared = std::sync::Arc::clone(&mutex);
                    handles.push(tokio::task::spawn(async move {
                        for _ in 0..50u64 {
                            let mut guard = shared.lock().await;
                            *guard += 1;
                        }
                    }));
                }
                for handle in handles {
                    handle.await.unwrap();
                }
            });
        });
    });

    group.bench_function("async_lock_direct", |bencher| {
        bencher.iter(|| {
            runtime.block_on(async {
                let mutex = std::sync::Arc::new(async_lock::Mutex::new(0u64));
                let mut handles = Vec::new();
                for _ in 0..2 {
                    let shared = std::sync::Arc::clone(&mutex);
                    handles.push(tokio::task::spawn(async move {
                        for _ in 0..50u64 {
                            let mut guard = shared.lock().await;
                            *guard += 1;
                        }
                    }));
                }
                for handle in handles {
                    handle.await.unwrap();
                }
            });
        });
    });

    group.bench_function("runtime_shaped_tokio", |bencher| {
        bencher.iter(|| {
            runtime.block_on(async {
                let mutex: std::sync::Arc<ShapedMutex<u64, TokioPerCoreRuntime>> =
                    std::sync::Arc::new(ShapedMutex::new(0u64));
                let mut handles = Vec::new();
                for _ in 0..2 {
                    let shared = std::sync::Arc::clone(&mutex);
                    handles.push(tokio::task::spawn(async move {
                        for _ in 0..50u64 {
                            let mut guard = shared.lock().await;
                            *guard += 1;
                        }
                    }));
                }
                for handle in handles {
                    handle.await.unwrap();
                }
            });
        });
    });

    group.finish();
}

fn bench_notify_park_wake_round_trip(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("r4_notify_park_wake_round_trip");
    group.measurement_time(Duration::from_secs(5));
    let runtime = current_thread_tokio();

    group.bench_function("tokio_direct", |bencher| {
        bencher.iter(|| {
            runtime.block_on(async {
                let notify = std::sync::Arc::new(tokio::sync::Notify::new());
                let waker = std::sync::Arc::clone(&notify);
                let waiter = tokio::task::spawn(async move {
                    waker.notified().await;
                });
                tokio::task::yield_now().await;
                notify.notify_one();
                waiter.await.unwrap();
            });
        });
    });

    group.bench_function("runtime_shaped_tokio", |bencher| {
        bencher.iter(|| {
            runtime.block_on(async {
                let notify: std::sync::Arc<ShapedNotify<TokioPerCoreRuntime>> =
                    std::sync::Arc::new(ShapedNotify::new());
                let waker = std::sync::Arc::clone(&notify);
                let waiter = tokio::task::spawn(async move {
                    waker.notified().await;
                });
                tokio::task::yield_now().await;
                notify.notify_one();
                waiter.await.unwrap();
            });
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_mutex_uncontended_round_trip,
    bench_mutex_two_serialized_lockers,
    bench_notify_park_wake_round_trip,
);
criterion_main!(benches);
