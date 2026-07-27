//! Cross-crate proof that prime's virtual clock is a real downstream
//! capability, not just something reachable from prime's own
//! `#[cfg(test)]` tree. This crate (`proxima`) sees `prime` only through
//! its published Cargo dependency edge — the `runtime-prime-virtual-clock`
//! feature this test relies on is forced on via this crate's
//! `[dev-dependencies]` entry for `prime` (see root `Cargo.toml`), not via
//! any special access to prime's internals.
#![cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-bgpool",
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use proxima::runtime::CoreId;
use proxima::runtime::prime::os::core_shard;
use proxima_clock::coarse::TickCell;
use proxima_clock::ticks::Ticks;

/// mirrors prime's own `virtual_clock_fires_timer_when_cell_is_advanced_with_zero_wall_clock_sleep`
/// test, but calls `launch_with_virtual_clock` from OUTSIDE prime: proof
/// that promoting `StdClock::Virtual` from `#[cfg(test)]` to the
/// `runtime-prime-virtual-clock` feature actually reaches a downstream
/// crate, not just prime's own test binary.
#[test]
fn virtual_clock_fires_timer_from_outside_prime_with_zero_wall_clock_sleep() {
    const SIMULATED_SECONDS: u64 = 25;
    const SIMULATED_MILLIS: u64 = SIMULATED_SECONDS * 1_000;

    let cell = Arc::new(TickCell::new(Ticks::ZERO));
    let handle = core_shard::launch_with_virtual_clock(CoreId(210), None, 2, 16, cell.clone())
        .expect("launch virtual-clock worker from outside prime");

    let (done_tx, done_rx) = mpsc::channel::<()>();
    handle
        .dispatch_factory(Box::new(move || {
            Box::pin(async move {
                core_shard::timer_at(SIMULATED_MILLIS).await;
                let _ = done_tx.send(());
            }) as Pin<Box<dyn Future<Output = ()> + 'static>>
        }))
        .expect("dispatch timer factory");

    // advance the SAME cell the worker's wheel reads, then nudge the
    // worker so a parked reactor rechecks the timer against the new tick
    // immediately instead of waiting out a stale real-ms timeout.
    cell.set(Ticks::from_raw(SIMULATED_MILLIS));
    handle
        .dispatch_send(Box::pin(async {}))
        .expect("nudge worker to recheck the timer wheel");

    done_rx
        .recv_timeout(Duration::from_secs(2))
        .expect(
            "timer must fire from the virtual advance alone; a real 25s wait would \
             exceed this 2s bound, proving no wall-clock sleep drove the fire",
        );
    handle.shutdown_and_join().expect("shutdown");
}

/// Part 2, proven from outside prime: three tasks sleeping for 1s, 5s and
/// 30s of SIMULATED time all complete in single-digit-to-low-hundreds of
/// milliseconds of WALL time, in deadline order — with NO `cell.set` call
/// anywhere in this test. The worker's own idle-park hook advances the
/// clock to the earliest pending deadline automatically; a downstream
/// caller gets this for free just by using `launch_with_virtual_clock`.
#[test]
fn auto_advance_orders_multiple_simulated_sleeps_from_outside_prime() {
    use std::sync::Mutex;
    use std::time::Instant;

    const ONE_SECOND_MILLIS: u64 = 1_000;
    const FIVE_SECONDS_MILLIS: u64 = 5_000;
    const THIRTY_SECONDS_MILLIS: u64 = 30_000;

    let cell = Arc::new(TickCell::new(Ticks::ZERO));
    let handle = core_shard::launch_with_virtual_clock(CoreId(211), None, 2, 16, cell)
        .expect("launch virtual-clock worker from outside prime");

    let order: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
    let (done_tx, done_rx) = mpsc::channel::<()>();

    // dispatch order deliberately scrambled relative to deadline order —
    // completion order must reflect DEADLINE order, not dispatch order.
    for deadline in [THIRTY_SECONDS_MILLIS, ONE_SECOND_MILLIS, FIVE_SECONDS_MILLIS] {
        let order_for_task = order.clone();
        let done_tx_for_task = done_tx.clone();
        handle
            .dispatch_factory(Box::new(move || {
                Box::pin(async move {
                    core_shard::timer_at(deadline).await;
                    order_for_task
                        .lock()
                        .expect("order mutex poisoned")
                        .push(deadline);
                    let _ = done_tx_for_task.send(());
                }) as Pin<Box<dyn Future<Output = ()> + 'static>>
            }))
            .expect("dispatch sleeper factory");
    }
    drop(done_tx);

    let started = Instant::now();
    for _ in 0..3 {
        done_rx
            .recv_timeout(Duration::from_secs(2))
            .expect(
                "all three simulated sleeps (1s/5s/30s) must complete via auto-advance \
                 alone; a real wait would take 36s and blow well past this 2s bound",
            );
    }
    let elapsed = started.elapsed();

    handle.shutdown_and_join().expect("shutdown");

    assert_eq!(
        *order.lock().expect("order mutex poisoned"),
        vec![ONE_SECOND_MILLIS, FIVE_SECONDS_MILLIS, THIRTY_SECONDS_MILLIS],
        "completion order must match simulated deadline order (1s, then 5s, then 30s)",
    );
    assert!(
        elapsed < Duration::from_millis(500),
        "36 simulated seconds must resolve in well under 500ms of wall clock (got {elapsed:?})",
    );
}
