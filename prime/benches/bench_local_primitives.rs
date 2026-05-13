#![allow(clippy::unwrap_used, clippy::expect_used)]

//! R8 of the runtime-shaped initiative — prime-pinned non-Send primitives.
//!
//! Three arms on the same single-core current-thread workload:
//!
//! - `tokio_send_mutex` — `tokio::sync::Mutex<T>` on a current-thread
//!   runtime — the Send-shaped incumbent that uses atomic ops for lock
//!   state even when the runtime never moves tasks across cores.
//! - `prime_local_mutex` — `PrimeLocalMutex<T>` via the
//!   `LocalRuntimeFactory` impl — `Cell<bool>` lock + `RefCell<T>` value,
//!   no atomic ops on the fast path.
//! - `prime_local_notify` — `PrimeLocalNotify` notify/notified round-trip
//!   vs `tokio::sync::Notify`.
//!
//! Workloads:
//! - `lock_uncontended_round_trip` — single locker, 100 lock/unlock cycles.
//!   This is the case the Local primitive should win on (no atomics, no
//!   cross-core fence).
//! - `lock_two_serialized_lockers` — two `spawn_local` tasks racing the
//!   same lock 50 times each. Serializes via the runtime queue; tests that
//!   the wake path doesn't regress vs tokio.
//! - `notify_park_wake_round_trip` — waiter parks on `notified()`, second
//!   task calls `notify_one()`. Exercises the per-waiter woken-flag path.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use prime::PrimeRuntime;
use proxima_runtime::primitives::{LocalMutexLike, LocalNotifyLike, LocalRuntimeFactory};

fn current_thread_tokio() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("current-thread runtime")
}

fn bench_lock_uncontended_round_trip(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("r8_lock_uncontended_round_trip");
    group.measurement_time(Duration::from_secs(5));
    let runtime = current_thread_tokio();

    group.bench_function("tokio_send_mutex", |bencher| {
        bencher.iter(|| {
            runtime.block_on(async {
                let mutex = tokio::sync::Mutex::new(0u64);
                for index in 0..100u64 {
                    let mut guard = mutex.lock().await;
                    *guard = std::hint::black_box(index);
                }
                let final_guard = mutex.lock().await;
                std::hint::black_box(*final_guard);
            });
        });
    });

    group.bench_function("prime_local_mutex", |bencher| {
        bencher.iter(|| {
            runtime.block_on(async {
                let mutex = PrimeRuntime::new_local_mutex(0u64);
                for index in 0..100u64 {
                    let mut guard = mutex.lock().await;
                    *guard = std::hint::black_box(index);
                }
                let final_guard = mutex.lock().await;
                std::hint::black_box(*final_guard);
            });
        });
    });

    group.finish();
}

fn bench_lock_two_serialized_lockers(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("r8_lock_two_serialized_lockers");
    group.measurement_time(Duration::from_secs(5));
    let runtime = current_thread_tokio();

    group.bench_function("tokio_send_mutex", |bencher| {
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

    group.bench_function("prime_local_mutex", |bencher| {
        bencher.iter(|| {
            runtime.block_on(async {
                let mutex = Rc::new(PrimeRuntime::new_local_mutex(0u64));
                let local = tokio::task::LocalSet::new();
                local
                    .run_until(async {
                        let mut handles = Vec::new();
                        for _ in 0..2 {
                            let shared = Rc::clone(&mutex);
                            handles.push(tokio::task::spawn_local(async move {
                                for _ in 0..50u64 {
                                    let mut guard = shared.lock().await;
                                    *guard += 1;
                                }
                            }));
                        }
                        for handle in handles {
                            handle.await.unwrap();
                        }
                    })
                    .await;
            });
        });
    });

    group.finish();
}

fn bench_notify_park_wake_round_trip(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("r8_notify_park_wake_round_trip");
    group.measurement_time(Duration::from_secs(5));
    let runtime = current_thread_tokio();

    group.bench_function("tokio_send_notify", |bencher| {
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

    group.bench_function("prime_local_notify", |bencher| {
        bencher.iter(|| {
            runtime.block_on(async {
                let notify = Rc::new(PrimeRuntime::new_local_notify());
                let waker = Rc::clone(&notify);
                let local = tokio::task::LocalSet::new();
                local
                    .run_until(async move {
                        let waiter = tokio::task::spawn_local(async move {
                            waker.notified().await;
                        });
                        tokio::task::yield_now().await;
                        notify.notify_one();
                        waiter.await.unwrap();
                    })
                    .await;
            });
        });
    });

    group.finish();
}

fn bench_lock_inner_loop_alloc_check(criterion: &mut Criterion) {
    // sanity bench: 10k uncontended lock acquisitions in one go, comparing
    // peak transient allocations is out-of-scope here (no allocator
    // instrumentation), but the throughput number IS the proxy — if the
    // Local impl is allocating inside the hot loop, throughput will tank
    // hard. AGENTS.md no-inner-loop-heap-alloc invariant is enforced by
    // shape, not by harness.
    let mut group = criterion.benchmark_group("r8_lock_inner_loop_alloc_check");
    group.measurement_time(Duration::from_secs(5));
    let runtime = current_thread_tokio();

    group.bench_function("tokio_send_mutex_10k", |bencher| {
        bencher.iter(|| {
            runtime.block_on(async {
                let mutex = tokio::sync::Mutex::new(RefCell::new(0u64));
                for index in 0..10_000u64 {
                    let guard = mutex.lock().await;
                    *guard.borrow_mut() = std::hint::black_box(index);
                }
            });
        });
    });

    group.bench_function("prime_local_mutex_10k", |bencher| {
        bencher.iter(|| {
            runtime.block_on(async {
                let mutex = PrimeRuntime::new_local_mutex(0u64);
                for index in 0..10_000u64 {
                    let mut guard = mutex.lock().await;
                    *guard = std::hint::black_box(index);
                }
            });
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_lock_uncontended_round_trip,
    bench_lock_two_serialized_lockers,
    bench_notify_park_wake_round_trip,
    bench_lock_inner_loop_alloc_check,
);
criterion_main!(benches);
