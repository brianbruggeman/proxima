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
