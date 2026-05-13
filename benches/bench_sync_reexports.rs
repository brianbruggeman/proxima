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
use std::time::{Duration, Instant};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use tokio::runtime::{Builder as TokioBuilder, Runtime};

const MUTEX_TASKS: usize = 4;
const MUTEX_ITERS: usize = 256;
const MUTEX_CANCEL_TASKS: usize = 16;
const MUTEX_CANCEL_ITERS: usize = 256;
const CANCEL_TIMEOUT_US: u64 = 50;
const SEMAPHORE_CANCEL_TASKS: usize = 16;
const SEMAPHORE_CANCEL_ITERS: usize = 128;
const RWLOCK_READERS: usize = 4;
const RWLOCK_READS_PER_READER: usize = 256;
const RWLOCK_WRITES: usize = 64;
const SEMAPHORE_TASKS: usize = 8;
const SEMAPHORE_ITERS: usize = 128;
const SEMAPHORE_CAPACITY: usize = 4;
const ONESHOT_ROUNDTRIPS: usize = 1024;
const ONESHOT_CLOSE_RACE_ITERS: usize = 4096;
const ONESHOT_CLOSED_POLL_ITERS: usize = 4096;

fn current_thread_runtime() -> Runtime {
    TokioBuilder::new_current_thread()
        .enable_all()
        .build()
        .expect("current thread runtime")
}

// ---------- Mutex ----------
//
// Feature-gap audit (incumbent: tokio::sync::Mutex)
//
// MUTEX ARM B: mutex_lock_owned_feature_gap
//   design-favors: incumbent — feature gap
//   tokio::sync::Mutex::lock_owned() returns OwnedMutexGuard<T>, a guard whose
//   lifetime is tied to Arc<Mutex<T>> rather than any borrow. Useful for
//   self-contained handles passed into spawned tasks.
//   futures::lock::Mutex has no equivalent. proxima::sync::Mutex cannot be
//   benched for this arm.
//   Source: src/sync/mutex.rs non-coverage section; futures 0.3 API.
//   Cannot run the arm — feature gap — deliberate trade-off.

async fn mutex_tokio() {
    let mutex = Arc::new(tokio::sync::Mutex::new(0u64));
    let handles: Vec<_> = (0..MUTEX_TASKS)
        .map(|_| {
            let mutex = Arc::clone(&mutex);
            tokio::spawn(async move {
                for _ in 0..MUTEX_ITERS {
                    let mut guard = mutex.lock().await;
                    *guard += 1;
                    drop(guard);
                }
            })
        })
        .collect();
    for handle in handles {
        handle.await.expect("mutex task join");
    }
}

async fn mutex_proxima() {
    let mutex = Arc::new(proxima::sync::Mutex::new(0u64));
    let handles: Vec<_> = (0..MUTEX_TASKS)
        .map(|_| {
            let mutex = Arc::clone(&mutex);
            tokio::spawn(async move {
                for _ in 0..MUTEX_ITERS {
                    let mut guard = mutex.lock().await;
                    *guard += 1;
                    drop(guard);
                }
            })
        })
        .collect();
    for handle in handles {
        handle.await.expect("mutex task join");
    }
}

fn bench_mutex(criterion: &mut Criterion) {
    // design-favors: neutral (uniform hold; cancellation never fires)
    let total_ops = (MUTEX_TASKS * MUTEX_ITERS) as u64;
    let mut group = criterion.benchmark_group("mutex_4task_contention");
    group.throughput(Throughput::Elements(total_ops));
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("tokio", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| runtime.block_on(mutex_tokio()));
    });

    group.bench_function("proxima", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| runtime.block_on(mutex_proxima()));
    });

    group.finish();
}

// ---------- Mutex cancellation under contention ----------
// design-favors: incumbent (cancellation under contention home turf)
//
// 16 tasks race for a single mutex; each iter wraps lock() in a 50µs timeout.
// ~50% of acquires succeed (short hold); ~50% time out and cancel mid-wait.
// tokio::sync::Mutex was specifically engineered to be cancel-safe: dropping
// the lock future before resolution leaves the mutex clean and unblocks the
// next waiter. futures::lock::Mutex has the same property per its design,
// but this is tokio's documented and tested design point.

async fn mutex_cancellation_tokio() {
    let mutex = Arc::new(tokio::sync::Mutex::new(0u64));
    let handles: Vec<_> = (0..MUTEX_CANCEL_TASKS)
        .map(|_| {
            let mutex = Arc::clone(&mutex);
            tokio::spawn(async move {
                for _ in 0..MUTEX_CANCEL_ITERS {
                    let timeout = Duration::from_micros(CANCEL_TIMEOUT_US);
                    if let Ok(mut guard) = tokio::time::timeout(timeout, mutex.lock()).await {
                        *guard += 1;
                        drop(guard);
                    }
                }
            })
        })
        .collect();
    for handle in handles {
        handle.await.expect("mutex cancellation task join");
    }
}

async fn mutex_cancellation_proxima() {
    let mutex = Arc::new(proxima::sync::Mutex::new(0u64));
    let handles: Vec<_> = (0..MUTEX_CANCEL_TASKS)
        .map(|_| {
            let mutex = Arc::clone(&mutex);
            tokio::spawn(async move {
                for _ in 0..MUTEX_CANCEL_ITERS {
                    let timeout = Duration::from_micros(CANCEL_TIMEOUT_US);
                    if let Ok(mut guard) = tokio::time::timeout(timeout, mutex.lock()).await {
                        *guard += 1;
                        drop(guard);
                    }
                }
            })
        })
        .collect();
    for handle in handles {
        handle.await.expect("mutex cancellation task join");
    }
}

fn bench_mutex_cancellation(criterion: &mut Criterion) {
    // design-favors: incumbent (cancellation under contention home turf)
    let total_ops = (MUTEX_CANCEL_TASKS * MUTEX_CANCEL_ITERS) as u64;
    let mut group = criterion.benchmark_group("mutex_cancellation_under_contention");
    group.throughput(Throughput::Elements(total_ops));
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("tokio", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| runtime.block_on(mutex_cancellation_tokio()));
    });

    group.bench_function("proxima", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| runtime.block_on(mutex_cancellation_proxima()));
    });

    group.finish();
}

// ---------- RwLock ----------

async fn rwlock_tokio() {
    let rwlock = Arc::new(tokio::sync::RwLock::new(0u64));

    let writer = {
        let rwlock = Arc::clone(&rwlock);
        tokio::spawn(async move {
            for _ in 0..RWLOCK_WRITES {
                let mut guard = rwlock.write().await;
                *guard += 1;
                drop(guard);
            }
        })
    };

    let readers: Vec<_> = (0..RWLOCK_READERS)
        .map(|_| {
            let rwlock = Arc::clone(&rwlock);
            tokio::spawn(async move {
                for _ in 0..RWLOCK_READS_PER_READER {
                    let guard = rwlock.read().await;
                    let _ = *guard;
                    drop(guard);
                }
            })
        })
        .collect();

    writer.await.expect("rwlock writer join");
    for handle in readers {
        handle.await.expect("rwlock reader join");
    }
}

async fn rwlock_proxima() {
    let rwlock = Arc::new(proxima::sync::RwLock::new(0u64));

    let writer = {
        let rwlock = Arc::clone(&rwlock);
        tokio::spawn(async move {
            for _ in 0..RWLOCK_WRITES {
                let mut guard = rwlock.write().await;
                *guard += 1;
                drop(guard);
            }
        })
    };

    let readers: Vec<_> = (0..RWLOCK_READERS)
        .map(|_| {
            let rwlock = Arc::clone(&rwlock);
            tokio::spawn(async move {
                for _ in 0..RWLOCK_READS_PER_READER {
                    let guard = rwlock.read().await;
                    let _ = *guard;
                    drop(guard);
                }
            })
        })
        .collect();

    writer.await.expect("rwlock writer join");
    for handle in readers {
        handle.await.expect("rwlock reader join");
    }
}

fn bench_rwlock(criterion: &mut Criterion) {
    // design-favors: neutral (fixed 1w/4r ratio; uniform hold; starvation regime not stressed)
    let total_ops = (RWLOCK_WRITES + RWLOCK_READERS * RWLOCK_READS_PER_READER) as u64;
    let mut group = criterion.benchmark_group("rwlock_1w4r_contention");
    group.throughput(Throughput::Elements(total_ops));
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("tokio", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| runtime.block_on(rwlock_tokio()));
    });

    group.bench_function("proxima", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| runtime.block_on(rwlock_proxima()));
    });

    group.finish();
}

// ---------- RwLock write-starvation (16r/2w, MT4) ----------
// design-favors: incumbent (write-starvation under heavy reader load home turf)
//
// 16 readers × 512 iters = 8192 read ops; 2 writers × 32 iters = 64 write ops (128:1 ratio).
// Tokio's RwLock has explicit write-starvation prevention. async_lock::RwLock may not.
// Metric: writer_elapsed vs reader_elapsed. If writers finish well before readers, no starvation.
// If writer_elapsed >> reader_elapsed, proxima's RwLock allows starvation.

const STARVE_READERS: usize = 16;
const STARVE_READS_PER_READER: usize = 512;
const STARVE_WRITERS: usize = 2;
const STARVE_WRITES_PER_WRITER: usize = 32;

fn multi_thread_runtime() -> Runtime {
    TokioBuilder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("multi thread runtime")
}

// Returns (writer_elapsed_ms, reader_elapsed_ms) to expose starvation ratio.
async fn rwlock_starvation_tokio() -> (u128, u128) {
    let rwlock = Arc::new(tokio::sync::RwLock::new(0u64));
    let start = Instant::now();

    let writer_handles: Vec<_> = (0..STARVE_WRITERS)
        .map(|_| {
            let rwlock = Arc::clone(&rwlock);
            tokio::spawn(async move {
                for _ in 0..STARVE_WRITES_PER_WRITER {
                    let mut guard = rwlock.write().await;
                    *guard += 1;
                    drop(guard);
                    tokio::time::sleep(Duration::from_nanos(1000)).await;
                }
            })
        })
        .collect();

    let reader_handles: Vec<_> = (0..STARVE_READERS)
        .map(|_| {
            let rwlock = Arc::clone(&rwlock);
            tokio::spawn(async move {
                for _ in 0..STARVE_READS_PER_READER {
                    let guard = rwlock.read().await;
                    let _ = *guard;
                    drop(guard);
                    tokio::time::sleep(Duration::from_nanos(100)).await;
                }
            })
        })
        .collect();

    for handle in writer_handles {
        handle.await.expect("starvation writer join");
    }
    let writer_elapsed = start.elapsed().as_millis();

    for handle in reader_handles {
        handle.await.expect("starvation reader join");
    }
    let reader_elapsed = start.elapsed().as_millis();

    (writer_elapsed, reader_elapsed)
}

async fn rwlock_starvation_proxima() -> (u128, u128) {
    let rwlock = Arc::new(proxima::sync::RwLock::new(0u64));
    let start = Instant::now();

    let writer_handles: Vec<_> = (0..STARVE_WRITERS)
        .map(|_| {
            let rwlock = Arc::clone(&rwlock);
            tokio::spawn(async move {
                for _ in 0..STARVE_WRITES_PER_WRITER {
                    let mut guard = rwlock.write().await;
                    *guard += 1;
                    drop(guard);
                    tokio::time::sleep(Duration::from_nanos(1000)).await;
                }
            })
        })
        .collect();

    let reader_handles: Vec<_> = (0..STARVE_READERS)
        .map(|_| {
            let rwlock = Arc::clone(&rwlock);
            tokio::spawn(async move {
                for _ in 0..STARVE_READS_PER_READER {
                    let guard = rwlock.read().await;
                    let _ = *guard;
                    drop(guard);
                    tokio::time::sleep(Duration::from_nanos(100)).await;
                }
            })
        })
        .collect();

    for handle in writer_handles {
        handle.await.expect("starvation writer join");
    }
    let writer_elapsed = start.elapsed().as_millis();

    for handle in reader_handles {
        handle.await.expect("starvation reader join");
    }
    let reader_elapsed = start.elapsed().as_millis();

    (writer_elapsed, reader_elapsed)
}

fn bench_rwlock_write_starvation_16r_2w(criterion: &mut Criterion) {
    // design-favors: incumbent (write-starvation under heavy reader load home turf)
    let total_ops = (STARVE_WRITERS * STARVE_WRITES_PER_WRITER
        + STARVE_READERS * STARVE_READS_PER_READER) as u64;
    let mut group = criterion.benchmark_group("rwlock_write_starvation_16r_2w");
    group.throughput(Throughput::Elements(total_ops));
    group.measurement_time(Duration::from_secs(15));

    group.bench_function("tokio", |bench| {
        let runtime = multi_thread_runtime();
        bench.iter(|| {
            let (writer_ms, reader_ms) = runtime.block_on(rwlock_starvation_tokio());
            // starvation ratio: writer should complete long before readers
            let _ = (writer_ms, reader_ms);
        });
    });

    group.bench_function("proxima", |bench| {
        let runtime = multi_thread_runtime();
        bench.iter(|| {
            let (writer_ms, reader_ms) = runtime.block_on(rwlock_starvation_proxima());
            let _ = (writer_ms, reader_ms);
        });
    });

    group.finish();
}

// ---------- Semaphore ----------
//
// Feature-gap audit (incumbent: tokio::sync::Semaphore)
//
// SEMAPHORE ARM A: semaphore_acquire_many_feature_gap
//   design-favors: incumbent — feature gap
//   tokio::sync::Semaphore::acquire_many(n) / try_acquire_many(n) atomically
//   reserve N permits, enabling permit-batching for back-pressure.
//   async_lock::Semaphore has no equivalent; implementing it correctly requires
//   retry-with-backoff atomic logic to preserve the all-or-nothing guarantee.
//   proxima::sync::Semaphore cannot be benched for this arm.
//   Source: src/sync/semaphore.rs non-coverage section; async-lock 3.4.2 API.
//   Cannot run the arm — feature gap — deliberate trade-off.

// design-favors: neutral (uniform contention; no cancellation)
async fn semaphore_tokio() {
    let semaphore = Arc::new(tokio::sync::Semaphore::new(SEMAPHORE_CAPACITY));
    let handles: Vec<_> = (0..SEMAPHORE_TASKS)
        .map(|_| {
            let semaphore = Arc::clone(&semaphore);
            tokio::spawn(async move {
                for _ in 0..SEMAPHORE_ITERS {
                    let permit = semaphore.acquire().await.expect("semaphore acquire");
                    drop(permit);
                }
            })
        })
        .collect();
    for handle in handles {
        handle.await.expect("semaphore task join");
    }
}

async fn semaphore_proxima() {
    let semaphore = Arc::new(proxima::sync::Semaphore::new(SEMAPHORE_CAPACITY));
    let handles: Vec<_> = (0..SEMAPHORE_TASKS)
        .map(|_| {
            let semaphore = Arc::clone(&semaphore);
            tokio::spawn(async move {
                for _ in 0..SEMAPHORE_ITERS {
                    let guard = semaphore.acquire().await;
                    drop(guard);
                }
            })
        })
        .collect();
    for handle in handles {
        handle.await.expect("semaphore task join");
    }
}

fn bench_semaphore(criterion: &mut Criterion) {
    // design-favors: neutral (uniform contention; no cancellation)
    let total_ops = (SEMAPHORE_TASKS * SEMAPHORE_ITERS) as u64;
    let mut group = criterion.benchmark_group("semaphore_8task_cap4");
    group.throughput(Throughput::Elements(total_ops));
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("tokio", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| runtime.block_on(semaphore_tokio()));
    });

    group.bench_function("proxima", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| runtime.block_on(semaphore_proxima()));
    });

    group.finish();
}

// ---------- Semaphore cancellation under contention ----------
// design-favors: incumbent (cancellation under contention home turf)
//
// 16 tasks race on a cap-4 semaphore; each iter wraps acquire() in a 50µs
// timeout. ~50% succeed; ~50% cancel mid-wait. tokio specifically designed its
// semaphore acquisition future to be cancel-safe: a dropped future never holds
// a permit, guaranteeing forward progress for other waiters.
// async_lock::Semaphore has the same property via its event_listener design,
// but this is tokio's documented design point.

async fn semaphore_cancellation_tokio() {
    let semaphore = Arc::new(tokio::sync::Semaphore::new(SEMAPHORE_CAPACITY));
    let handles: Vec<_> = (0..SEMAPHORE_CANCEL_TASKS)
        .map(|_| {
            let semaphore = Arc::clone(&semaphore);
            tokio::spawn(async move {
                for _ in 0..SEMAPHORE_CANCEL_ITERS {
                    let timeout = Duration::from_micros(CANCEL_TIMEOUT_US);
                    if let Ok(permit) = tokio::time::timeout(timeout, semaphore.acquire()).await {
                        drop(permit.expect("semaphore permit"));
                    }
                }
            })
        })
        .collect();
    for handle in handles {
        handle.await.expect("semaphore cancellation task join");
    }
}

async fn semaphore_cancellation_proxima() {
    let semaphore = Arc::new(proxima::sync::Semaphore::new(SEMAPHORE_CAPACITY));
    let handles: Vec<_> = (0..SEMAPHORE_CANCEL_TASKS)
        .map(|_| {
            let semaphore = Arc::clone(&semaphore);
            tokio::spawn(async move {
                for _ in 0..SEMAPHORE_CANCEL_ITERS {
                    let timeout = Duration::from_micros(CANCEL_TIMEOUT_US);
                    if let Ok(guard) = tokio::time::timeout(timeout, semaphore.acquire()).await {
                        drop(guard);
                    }
                }
            })
        })
        .collect();
    for handle in handles {
        handle.await.expect("semaphore cancellation task join");
    }
}

fn bench_semaphore_cancellation(criterion: &mut Criterion) {
    // design-favors: incumbent (cancellation under contention home turf)
    let total_ops = (SEMAPHORE_CANCEL_TASKS * SEMAPHORE_CANCEL_ITERS) as u64;
    let mut group = criterion.benchmark_group("semaphore_cancellation_under_contention");
    group.throughput(Throughput::Elements(total_ops));
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("tokio", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| runtime.block_on(semaphore_cancellation_tokio()));
    });

    group.bench_function("proxima", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| runtime.block_on(semaphore_cancellation_proxima()));
    });

    group.finish();
}

// ---------- oneshot ----------

async fn oneshot_tokio() {
    for _ in 0..ONESHOT_ROUNDTRIPS {
        let (tx, rx) = tokio::sync::oneshot::channel::<u64>();
        tx.send(42).expect("oneshot send");
        let _ = rx.await.expect("oneshot recv");
    }
}

async fn oneshot_proxima() {
    for _ in 0..ONESHOT_ROUNDTRIPS {
        let (tx, rx) = proxima::sync::oneshot::channel::<u64>();
        tx.send(42).expect("oneshot send");
        let _ = rx.await.expect("oneshot recv");
    }
}

fn bench_oneshot(criterion: &mut Criterion) {
    // design-favors: proxima (sequential send/recv; tokio's close-race machinery never fires)
    let mut group = criterion.benchmark_group("oneshot_roundtrip");
    group.throughput(Throughput::Elements(ONESHOT_ROUNDTRIPS as u64));
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("tokio", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| runtime.block_on(oneshot_tokio()));
    });

    group.bench_function("proxima", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| runtime.block_on(oneshot_proxima()));
    });

    group.finish();
}

// ---------- oneshot close-race ----------
// design-favors: incumbent (close-race detection home turf)
//
// 4096 iterations of: spawn sender + receiver tasks; receiver drops
// immediately (cancellation); sender's send() returns Err. Exercises
// tokio's close-race state machine vs futures::channel::oneshot's
// simpler drop-detection path.

async fn oneshot_close_race_tokio() {
    for _ in 0..ONESHOT_CLOSE_RACE_ITERS {
        let (tx, rx) = tokio::sync::oneshot::channel::<u64>();
        let sender_task = tokio::spawn(async move {
            tokio::task::yield_now().await;
            let _ = tx.send(42);
        });
        drop(rx);
        sender_task.await.expect("close-race sender join");
    }
}

async fn oneshot_close_race_proxima() {
    for _ in 0..ONESHOT_CLOSE_RACE_ITERS {
        let (tx, rx) = proxima::sync::oneshot::channel::<u64>();
        let sender_task = tokio::spawn(async move {
            tokio::task::yield_now().await;
            let _ = tx.send(42);
        });
        drop(rx);
        sender_task.await.expect("close-race sender join");
    }
}

fn bench_oneshot_close_race(criterion: &mut Criterion) {
    // design-favors: incumbent (close-race detection home turf)
    let mut group = criterion.benchmark_group("oneshot_close_race");
    group.throughput(Throughput::Elements(ONESHOT_CLOSE_RACE_ITERS as u64));
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("tokio", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| runtime.block_on(oneshot_close_race_tokio()));
    });

    group.bench_function("proxima", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| runtime.block_on(oneshot_close_race_proxima()));
    });

    group.finish();
}

// ---------- oneshot closed-poll ----------
// design-favors: incumbent — feature gap
//
// Tokio provides `Sender::closed().await` — a future that resolves when
// the receiver is dropped. This is the await-for-cancellation path that
// tokio's state machine machinery exists to serve.
//
// `futures::channel::oneshot::Sender` does NOT have `.closed()`.
// The equivalent is `.cancellation()` (wraps `poll_canceled`), but
// `proxima::sync::oneshot` does not re-export it because no internal
// caller used it at re-export time (see src/sync/oneshot.rs Non-coverage
// section). This is a deliberate trade-off: zero-overhead re-export at
// the cost of not surfacing cancellation-polling to callers.
//
// Callers needing this on proxima can use:
//   futures::channel::oneshot::Sender::cancellation()
// directly, but it requires importing futures::channel::oneshot alongside
// proxima::sync::oneshot — the APIs are not unified.
//
// The tokio arm is included to document the regime and measured cost.
// The proxima arm is omitted: the feature is not surfaced, so the
// comparison cannot be made. This is documented as a feature gap, not
// a performance loss.

async fn oneshot_closed_poll_tokio() {
    for _ in 0..ONESHOT_CLOSED_POLL_ITERS {
        let (mut tx, rx) = tokio::sync::oneshot::channel::<u64>();
        let sender_task = tokio::spawn(async move {
            tx.closed().await;
        });
        tokio::task::yield_now().await;
        drop(rx);
        sender_task.await.expect("closed-poll sender join");
    }
}

fn bench_oneshot_closed_poll(criterion: &mut Criterion) {
    // design-favors: incumbent — feature gap (futures::channel::oneshot::Sender lacks .closed())
    // proxima arm omitted: Sender::closed() is not re-exported; feature gap, not a perf loss.
    let mut group = criterion.benchmark_group("oneshot_closed_poll");
    group.throughput(Throughput::Elements(ONESHOT_CLOSED_POLL_ITERS as u64));
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("tokio", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| runtime.block_on(oneshot_closed_poll_tokio()));
    });

    // proxima arm: OMITTED
    // futures::channel::oneshot::Sender has no .closed() method.
    // proxima::sync::oneshot re-exports futures::channel::oneshot directly
    // and does not wrap or add .closed(). The regime cannot be benchmarked
    // on proxima without either: (a) adding a wrapper that surfaces
    // .cancellation() as .closed(), or (b) using the futures crate directly,
    // which defeats the point of the comparison. Documented as feature gap.

    group.finish();
}

criterion_group!(
    benches,
    bench_mutex,
    bench_mutex_cancellation,
    bench_rwlock,
    bench_rwlock_write_starvation_16r_2w,
    bench_semaphore,
    bench_semaphore_cancellation,
    bench_oneshot,
    bench_oneshot_close_race,
    bench_oneshot_closed_poll,
);
criterion_main!(benches);
