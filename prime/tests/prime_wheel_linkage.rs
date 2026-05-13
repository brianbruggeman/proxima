//! Gate row 3: runtime linkage proof for the prime-wheel timer driver.
//!
//! This binary links prime (which exports `proxima_time_external_now_millis` and
//! `proxima_time_external_schedule_wake` in `timer_driver`) against
//! `proxima_core::time` compiled with `time-driver-prime-wheel`. The linker
//! must resolve the two extern symbols at link time; if it cannot, the
//! binary fails to build — the correct loud failure for "time-driver-prime-wheel
//! requested but no runtime provides it".
//!
//! The test proves the full call chain:
//!   `proxima_core::time::sleep` → `Sleep::poll` → `ExternalDriver::schedule_wake`
//!   → `proxima_time_external_schedule_wake` (symbol) → `core_shard::schedule_wake`
//!   → worker `TimerWheel::register` → wheel advance fires waker → future resolves.
//!
//! Required features: runtime-prime-executor, runtime-prime-inbox-alloc,
//!   runtime-prime-reactor, runtime-prime-bgpool (all needed for PrimeRuntime
//!   + timer_driver symbol export).

#![cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool",
))]
#![allow(clippy::expect_used)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use prime::os::runtime::PrimeRuntime;
use proxima_runtime::{CoreId, Runtime};

/// Proves that:
///   1. the linker resolved both extern "Rust" timer symbols (build-time proof),
///   2. `proxima_core::time::sleep` fires correctly on a prime worker (runtime proof).
///
/// `sleep` is constructed INSIDE the factory: `ExternalDriver::now()` reads
/// `CURRENT_TIMER` (the worker thread-local), which is null outside a worker.
/// The factory closure runs on core 0's worker thread, where the TL is set.
///
/// Assertion: no wall-clock exact timing — only that the flag is set (future
/// resolved) within a 2-second deadline, making the test deterministic.
#[test]
fn prime_wheel_extern_symbols_resolve_and_sleep_fires() {
    let runtime: Arc<dyn Runtime> =
        Arc::new(PrimeRuntime::new(1).expect("build 1-core prime runtime"));
    let fired = Arc::new(AtomicBool::new(false));
    let fired_factory = fired.clone();

    runtime
        .spawn_factory_on_core(
            CoreId(0),
            Box::new(move || {
                let sleep_future = proxima_core::time::sleep(Duration::from_millis(10));
                Box::pin(async move {
                    sleep_future.await;
                    fired_factory.store(true, Ordering::Release);
                })
            }),
        )
        .expect("dispatch factory to prime core 0");

    let deadline = Instant::now() + Duration::from_secs(2);
    while !fired.load(Ordering::Acquire) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(
        fired.load(Ordering::Acquire),
        "prime-wheel Sleep did not fire within 2s — \
         extern symbol resolved but timer wheel did not advance (check worker loop)"
    );
}

/// Second assertion: `proxima_core::time::now()` on a prime worker returns a
/// non-zero monotonic value and advances over time. This proves that
/// `proxima_time_external_now_millis` routes to the correct worker clock
/// and not a stale or epoch-zero value.
#[test]
fn prime_wheel_now_advances_on_worker() {
    let runtime: Arc<dyn Runtime> =
        Arc::new(PrimeRuntime::new(1).expect("build 1-core prime runtime"));

    let (sender, receiver) = std::sync::mpsc::channel::<(u64, u64)>();

    runtime
        .spawn_factory_on_core(
            CoreId(0),
            Box::new(move || {
                Box::pin(async move {
                    let first = proxima_core::time::now().into_monotonic().as_millis() as u64;
                    proxima_core::time::sleep(Duration::from_millis(20)).await;
                    let second = proxima_core::time::now().into_monotonic().as_millis() as u64;
                    let _ = sender.send((first, second));
                })
            }),
        )
        .expect("dispatch factory to prime core 0");

    let (first, second) = receiver
        .recv_timeout(Duration::from_secs(2))
        .expect("now() readings must arrive within 2s");

    assert!(
        second > first,
        "proxima_core::time::now() must advance between two reads separated by sleep(20ms); \
         got first={first}ms second={second}ms"
    );
}
