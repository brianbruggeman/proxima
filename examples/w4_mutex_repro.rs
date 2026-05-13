//! W4 root-cause repro: 4 prime tasks contending on a shared
//! `futures::lock::Mutex`, no Criterion overhead. If this hangs, the
//! bug is in prime's cross-core wake path, not in Criterion's warmup.
//!
//! Run:
//!     cargo run --release --example w4_mutex_repro --features \
//!         "runtime-tokio runtime-prime-full"
//!
//! Reports the wall-clock time for each task to finish 4000 lock
//! cycles. Prints "DONE" if all 4 tasks complete, "HUNG" if the
//! 30-second deadline elapses with tasks still pending.

#![cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
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

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use futures::FutureExt;
use proxima::runtime::{CoreId, PrimeRuntime, Runtime, spawn_on_core_blocking_with};

const TASKS: usize = 4;
const OPS_PER_TASK: usize = 4_000;
const DEADLINE_SECS: u64 = 30;

/// One iter: fresh lock + 4 tasks, busy-spin until all 4 finish.
fn run_iter(runtime: &Arc<dyn Runtime>) -> Duration {
    let lock = Arc::new(futures::lock::Mutex::new(0u64));
    let done = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();

    // started[i] = 1 once task i's body actually ran (proves the
    // worker pulled the task out of the inbox). if `done < TASKS` but
    // `started == TASKS`, the stuck task is wedged in lock.lock().
    // if `started < TASKS`, the stuck task was never pulled from the
    // inbox — that's the spawn-side race close gap.
    let started_marks: Vec<Arc<AtomicUsize>> =
        (0..TASKS).map(|_| Arc::new(AtomicUsize::new(0))).collect();

    for thread_index in 0..TASKS {
        let lock_outer = lock.clone();
        let done_outer = done.clone();
        let started_outer = started_marks[thread_index].clone();
        let core = CoreId(thread_index);
        let _ = spawn_on_core_blocking_with(runtime.as_ref(), core, move || {
            let lock_inner = lock_outer.clone();
            let done_inner = done_outer.clone();
            let started_inner = started_outer.clone();
            Box::pin(async move {
                started_inner.store(1, Ordering::Release);
                for _ in 0..OPS_PER_TASK {
                    let mut guard = lock_inner.lock().await;
                    *guard += 1;
                    drop(guard);
                }
                done_inner.fetch_add(1, Ordering::AcqRel);
            })
        });
    }

    let deadline = Instant::now() + Duration::from_secs(DEADLINE_SECS);
    let mut deadline_check: u32 = 0;
    loop {
        let count = done.load(Ordering::Acquire);
        if count == TASKS {
            break;
        }
        deadline_check = deadline_check.wrapping_add(1);
        if deadline_check >= 1_000_000 {
            deadline_check = 0;
            if Instant::now() >= deadline {
                let lock_value = lock.lock().now_or_never().map(|guard| *guard).unwrap_or(0);
                let started: Vec<usize> = started_marks
                    .iter()
                    .map(|atomic| atomic.load(Ordering::Acquire))
                    .collect();
                eprintln!(
                    "HUNG: only {} of {} tasks done after {}s; lock value = {}; \
                     started per core = {:?}",
                    count, TASKS, DEADLINE_SECS, lock_value, started
                );
                std::process::exit(2);
            }
        }
        std::hint::spin_loop();
    }

    started.elapsed()
}

fn main() {
    let iter_count: usize = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(1);

    let runtime: Arc<dyn Runtime> = Arc::new(PrimeRuntime::new(TASKS).expect("prime runtime"));
    let started = Instant::now();
    for iter in 0..iter_count {
        let elapsed = run_iter(&runtime);
        if iter < 5 || iter % 10 == 0 {
            println!("iter {}: {:?}", iter, elapsed);
        }
    }
    let total = started.elapsed();
    let total_ops = TASKS * OPS_PER_TASK * iter_count;
    println!(
        "DONE: {} iters x {} tasks x {} ops in {:?} ({:.0} ops/sec)",
        iter_count,
        TASKS,
        OPS_PER_TASK,
        total,
        total_ops as f64 / total.as_secs_f64()
    );
}
