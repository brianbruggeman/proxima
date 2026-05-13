//! mutex parity baseline — A1.a of the tokio-parity plan.
//!
//! three workloads mirroring real proxima usage (process_rpc,
//! daemon_control_plane, mcp.rs). harness only — numbers come in
//! the next step; see discipline-tokio-parity.md for the log.

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

use criterion::{Criterion, criterion_group, criterion_main};
use tokio::runtime::Builder as TokioBuilder;

const IO_MUTEX_TASKS: usize = 4;
const IO_MUTEX_ROUNDS: usize = 10;
const IO_BODY_SLEEP_US: u64 = 500;

const ADMIN_READER_TASKS: usize = 7;
const ADMIN_WRITER_CYCLES: usize = 100;
const ADMIN_HOLD_SPIN_NS: u64 = 10_000;

const WRITE_FLUSH_CYCLES: usize = 100;
const WRITE_PAYLOAD_BYTES: usize = 1024;

fn current_thread_runtime() -> tokio::runtime::Runtime {
    TokioBuilder::new_current_thread()
        .enable_all()
        .build()
        .expect("current thread runtime")
}

fn busy_spin_ns(nanos: u64) {
    let deadline = std::time::Instant::now() + Duration::from_nanos(nanos);
    while std::time::Instant::now() < deadline {
        std::hint::spin_loop();
    }
}

// ---------- workload 1: I/O-serializing mutex (process_rpc pattern) ----------
//
// 4 tasks compete for 1 mutex. each task locks, sleeps 500µs (simulated body
// I/O), releases. contention is by design — the mutex serializes dispatch,
// identical to `state.child.lock().await` in process_rpc.rs:85.
//
// parking_lot is omitted here: it cannot `.await` across a held lock, so the
// workload shape is fundamentally incompatible with a sync floor reference.

fn bench_workload1_io_serializing(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("mutex_parity_w1_io_serializing");
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("proxima_sync_mutex", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let mutex = Arc::new(proxima::sync::Mutex::new(0u64));
                let mut handles = Vec::with_capacity(IO_MUTEX_TASKS);
                for _ in 0..IO_MUTEX_TASKS {
                    let mutex = mutex.clone();
                    handles.push(tokio::spawn(async move {
                        for _ in 0..IO_MUTEX_ROUNDS {
                            let mut guard = mutex.lock().await;
                            tokio::time::sleep(Duration::from_micros(IO_BODY_SLEEP_US)).await;
                            *guard += 1;
                        }
                    }));
                }
                for handle in handles {
                    handle.await.expect("task join");
                }
                let final_count = *mutex.lock().await;
                std::hint::black_box(final_count);
            });
        });
    });

    group.bench_function("tokio_sync_mutex", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let mutex = Arc::new(tokio::sync::Mutex::new(0u64));
                let mut handles = Vec::with_capacity(IO_MUTEX_TASKS);
                for _ in 0..IO_MUTEX_TASKS {
                    let mutex = mutex.clone();
                    handles.push(tokio::spawn(async move {
                        for _ in 0..IO_MUTEX_ROUNDS {
                            let mut guard = mutex.lock().await;
                            tokio::time::sleep(Duration::from_micros(IO_BODY_SLEEP_US)).await;
                            *guard += 1;
                        }
                    }));
                }
                for handle in handles {
                    handle.await.expect("task join");
                }
                let final_count = *mutex.lock().await;
                std::hint::black_box(final_count);
            });
        });
    });

    group.finish();
}

// ---------- workload 2: hot-read under admin write (daemon_control_plane) ----------
//
// 7 reader tasks spin try_lock in a tight loop with yield_now between attempts.
// 1 writer task acquires, busy-spins ~10µs of CPU work (simulating a state
// mutation), releases. 100 writer cycles per iteration.
//
// the benched op is the writer side — does reader pressure delay the writer's
// lock acquisition? parking_lot::Mutex is the sync floor: pure spin under no
// async, reveals the async-wait-queue overhead.

fn bench_workload2_hot_read_admin_write(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("mutex_parity_w2_hot_read_admin_write");
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("proxima_sync_mutex", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let mutex = Arc::new(proxima::sync::Mutex::new(0u64));
                let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

                let mut reader_handles = Vec::with_capacity(ADMIN_READER_TASKS);
                for _ in 0..ADMIN_READER_TASKS {
                    let mutex = mutex.clone();
                    let stop = stop.clone();
                    reader_handles.push(tokio::spawn(async move {
                        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                            if let Ok(guard) = mutex.try_lock() {
                                std::hint::black_box(*guard);
                                drop(guard);
                            }
                            tokio::task::yield_now().await;
                        }
                    }));
                }

                for _ in 0..ADMIN_WRITER_CYCLES {
                    let mut guard = mutex.lock().await;
                    busy_spin_ns(ADMIN_HOLD_SPIN_NS);
                    *guard += 1;
                    drop(guard);
                    tokio::task::yield_now().await;
                }

                stop.store(true, std::sync::atomic::Ordering::Relaxed);
                for handle in reader_handles {
                    handle.await.expect("reader join");
                }
                let final_count = *mutex.lock().await;
                std::hint::black_box(final_count);
            });
        });
    });

    group.bench_function("tokio_sync_mutex", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let mutex = Arc::new(tokio::sync::Mutex::new(0u64));
                let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

                let mut reader_handles = Vec::with_capacity(ADMIN_READER_TASKS);
                for _ in 0..ADMIN_READER_TASKS {
                    let mutex = mutex.clone();
                    let stop = stop.clone();
                    reader_handles.push(tokio::spawn(async move {
                        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                            if let Ok(guard) = mutex.try_lock() {
                                std::hint::black_box(*guard);
                                drop(guard);
                            }
                            tokio::task::yield_now().await;
                        }
                    }));
                }

                for _ in 0..ADMIN_WRITER_CYCLES {
                    let mut guard = mutex.lock().await;
                    busy_spin_ns(ADMIN_HOLD_SPIN_NS);
                    *guard += 1;
                    drop(guard);
                    tokio::task::yield_now().await;
                }

                stop.store(true, std::sync::atomic::Ordering::Relaxed);
                for handle in reader_handles {
                    handle.await.expect("reader join");
                }
                let final_count = *mutex.lock().await;
                std::hint::black_box(final_count);
            });
        });
    });

    group.bench_function("parking_lot_mutex", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let mutex = Arc::new(parking_lot::Mutex::new(0u64));
                let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

                let mut reader_handles = Vec::with_capacity(ADMIN_READER_TASKS);
                for _ in 0..ADMIN_READER_TASKS {
                    let mutex = mutex.clone();
                    let stop = stop.clone();
                    reader_handles.push(tokio::spawn(async move {
                        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                            if let Some(guard) = mutex.try_lock() {
                                std::hint::black_box(*guard);
                                drop(guard);
                            }
                            tokio::task::yield_now().await;
                        }
                    }));
                }

                for _ in 0..ADMIN_WRITER_CYCLES {
                    {
                        let mut guard = mutex.lock();
                        busy_spin_ns(ADMIN_HOLD_SPIN_NS);
                        *guard += 1;
                    }
                    tokio::task::yield_now().await;
                }

                stop.store(true, std::sync::atomic::Ordering::Relaxed);
                for handle in reader_handles {
                    handle.await.expect("reader join");
                }
                let final_count = *mutex.lock();
                std::hint::black_box(final_count);
            });
        });
    });

    group.finish();
}

// ---------- workload 3: async write + flush (mcp.rs pattern) ----------
//
// 1 task, near-uncontended. lock → write 1KB to a Vec<u8> inside the mutex →
// release. 100 cycles per iter. this is the mcp.rs stdout/UDS write path.
// parking_lot is included here — sync floor for the overhead-free acquire
// on an uncontended mutex.

fn bench_workload3_async_write_flush(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("mutex_parity_w3_async_write_flush");
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("proxima_sync_mutex", |bench| {
        let runtime = current_thread_runtime();
        let payload = vec![0xABu8; WRITE_PAYLOAD_BYTES];
        bench.iter(|| {
            runtime.block_on(async {
                let mutex = Arc::new(proxima::sync::Mutex::new(Vec::<u8>::new()));
                for _ in 0..WRITE_FLUSH_CYCLES {
                    let mut guard = mutex.lock().await;
                    guard.extend_from_slice(&payload);
                    std::hint::black_box(guard.len());
                    guard.clear();
                }
            });
        });
    });

    group.bench_function("tokio_sync_mutex", |bench| {
        let runtime = current_thread_runtime();
        let payload = vec![0xABu8; WRITE_PAYLOAD_BYTES];
        bench.iter(|| {
            runtime.block_on(async {
                let mutex = Arc::new(tokio::sync::Mutex::new(Vec::<u8>::new()));
                for _ in 0..WRITE_FLUSH_CYCLES {
                    let mut guard = mutex.lock().await;
                    guard.extend_from_slice(&payload);
                    std::hint::black_box(guard.len());
                    guard.clear();
                }
            });
        });
    });

    group.bench_function("parking_lot_mutex", |bench| {
        let runtime = current_thread_runtime();
        let payload = vec![0xABu8; WRITE_PAYLOAD_BYTES];
        bench.iter(|| {
            runtime.block_on(async {
                let mutex = Arc::new(parking_lot::Mutex::new(Vec::<u8>::new()));
                for _ in 0..WRITE_FLUSH_CYCLES {
                    let mut guard = mutex.lock();
                    guard.extend_from_slice(&payload);
                    std::hint::black_box(guard.len());
                    guard.clear();
                }
            });
        });
    });

    group.finish();
}

criterion_group!(
    bench_mutex_parity_workloads,
    bench_workload1_io_serializing,
    bench_workload2_hot_read_admin_write,
    bench_workload3_async_write_flush,
);
criterion_main!(bench_mutex_parity_workloads);
