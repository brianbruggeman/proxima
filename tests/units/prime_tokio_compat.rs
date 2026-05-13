//! P2.b integration test — `tokio::spawn` from inside a prime task on
//! a runtime built via `PrimeRuntime::builder().tokio_compat()` reaches
//! the sister tokio current-thread runtime, executes, and completes.
//!
//! Two properties under test:
//!
//! 1. `tokio::spawn(future)` from inside a prime task does not panic
//!    (a vanilla `PrimeRuntime::new(...)` causes
//!    `Handle::current()` to panic on the spawn call). The presence
//!    of the EnterGuard on the prime worker is the contract here.
//! 2. The spawned future actually runs to completion. We observe an
//!    atomic counter to prove the body executed; we await the spawned
//!    `JoinHandle` from inside the same prime task to prove the join
//!    side is also functional.
//!
//! Additionally: `tokio::time::sleep` reaches the sister's timer driver
//! (covered indirectly by the second test that sleeps before counting).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
#![cfg(feature = "prime-tokio-compat")]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use proxima::prime::PrimeRuntime;
use proxima::runtime::{CoreId, Runtime};

/// Sanity: vanilla `PrimeRuntime::new(...)` (no compat) does NOT have
/// a tokio context on its workers, so `tokio::spawn` would panic. This
/// test asserts the assumption — if it changes, the compat path is no
/// longer load-bearing.
#[test]
fn vanilla_prime_runtime_has_no_tokio_context() {
    let runtime = PrimeRuntime::builder()
        .cores(1)
        .background_inline()
        .build()
        .expect("build vanilla prime");
    let observed_current: Arc<std::sync::Mutex<Option<bool>>> =
        Arc::new(std::sync::Mutex::new(None));
    let observed_for_task = observed_current.clone();
    runtime
        .spawn_on_core(
            CoreId(0),
            Box::pin(async move {
                let has_current = tokio::runtime::Handle::try_current().is_ok();
                *observed_for_task.lock().expect("mutex") = Some(has_current);
            }),
        )
        .expect("spawn on vanilla prime");
    let deadline = Instant::now() + Duration::from_secs(2);
    while observed_current.lock().expect("mutex").is_none() && Instant::now() < deadline {
        std::thread::yield_now();
    }
    let value = observed_current.lock().expect("mutex").expect("task ran");
    assert!(
        !value,
        "vanilla prime worker should NOT have a current tokio runtime"
    );
}

#[test]
fn tokio_spawn_inside_prime_task_runs_to_completion_via_compat() {
    let runtime = PrimeRuntime::builder()
        .cores(2)
        .background_inline()
        .tokio_compat()
        .build()
        .expect("build prime-tokio-compat");
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_for_outer = counter.clone();
    let done = Arc::new(AtomicUsize::new(0));
    let done_for_outer = done.clone();
    runtime
        .spawn_on_core(
            CoreId(0),
            Box::pin(async move {
                // tokio::spawn looks up Handle::current() — must be Ok on
                // a compat-mode worker.
                let join = tokio::spawn({
                    let counter = counter_for_outer.clone();
                    async move {
                        // tokio::time::sleep reaches the sister timer
                        // driver — sleep elapses, returns control to the
                        // task body.
                        tokio::time::sleep(Duration::from_millis(5)).await;
                        counter.fetch_add(1, Ordering::AcqRel);
                    }
                });
                // Await the JoinHandle from the prime task. The
                // JoinHandle future is runtime-aware; its wake fires
                // from the sister runtime when the spawned task ends.
                let _ = join.await;
                done_for_outer.store(1, Ordering::Release);
            }),
        )
        .expect("spawn on prime-tokio-compat");

    let deadline = Instant::now() + Duration::from_secs(5);
    while done.load(Ordering::Acquire) == 0 && Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert_eq!(
        done.load(Ordering::Acquire),
        1,
        "outer prime task never completed: tokio::spawn awaitable from prime?",
    );
    assert_eq!(
        counter.load(Ordering::Acquire),
        1,
        "tokio-spawned inner task body did not run",
    );
}

#[test]
fn tokio_sync_mutex_acquires_inside_prime_task() {
    // tokio::sync::Mutex does NOT need a tokio runtime context to
    // function — but on a compat-mode prime worker, lock + release
    // round-trips should be observed correctly. tests that compat
    // mode doesn't break the parts of tokio that were going to work
    // anyway.
    let runtime = PrimeRuntime::builder()
        .cores(1)
        .background_inline()
        .tokio_compat()
        .build()
        .expect("build prime-tokio-compat");
    let lock = Arc::new(tokio::sync::Mutex::new(0u64));
    let lock_for_task = lock.clone();
    let done = Arc::new(AtomicUsize::new(0));
    let done_for_task = done.clone();
    runtime
        .spawn_on_core(
            CoreId(0),
            Box::pin(async move {
                for _ in 0..100 {
                    let mut guard = lock_for_task.lock().await;
                    *guard += 1;
                    drop(guard);
                }
                done_for_task.store(1, Ordering::Release);
            }),
        )
        .expect("spawn on prime-tokio-compat");

    let deadline = Instant::now() + Duration::from_secs(5);
    while done.load(Ordering::Acquire) == 0 && Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert_eq!(done.load(Ordering::Acquire), 1);
    // separate runtime to safely lock the tokio::Mutex from the main
    // test thread — does NOT need to be the compat runtime.
    let probe = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("probe runtime");
    let final_value = probe.block_on(async {
        let guard = lock.lock().await;
        *guard
    });
    assert_eq!(final_value, 100);
}
