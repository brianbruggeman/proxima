//! Link-time timer hooks that back `proxima_core::time`'s
//! `time-driver-prime-wheel`.
//!
//! proxima-core cannot cargo-depend on prime — `prime -> proxima-pipe ->
//! proxima-core` already exists, so a `proxima-core -> prime` edge would
//! cycle. The prime-wheel driver is therefore wired by LINKAGE, not by a
//! dep: proxima-core's `time` module declares two `extern "Rust"` symbols
//! and calls them through its `ExternalDriver`; prime defines them here
//! with `#[unsafe(no_mangle)]`. The linker ties the two crates together in
//! the final binary with zero dependency edge — the `#[global_allocator]`
//! pattern applied to the timer driver.
//!
//! Each call routes to the CALLING worker's per-core timer wheel via
//! prime's thread-local, so the symbols are global yet every call stays
//! per-core — the same Send-but-per-worker contract as the prime TCP
//! acceptor.

use core::task::Waker;

use crate::os::core_shard;

/// Backs `proxima_core::time::now()` under prime-wheel — milliseconds since
/// the calling worker's shard launched.
#[unsafe(no_mangle)]
pub extern "Rust" fn proxima_time_external_now_millis() -> u64 {
    // on a prime worker: the per-core wheel (hot path, unchanged). off a
    // worker — a tokio-hosted client in a mixed-runtime binary that links
    // prime — the wheel is unreachable, so read a monotonic wall clock
    // instead of aborting. no_std keeps the strict contract by preserving
    // the panic (you are always on a worker there).
    match core_shard::current_tick_checked() {
        Some(tick) => tick,
        None => fallback_now_millis(),
    }
}

#[cfg(feature = "std")]
fn fallback_now_millis() -> u64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_millis() as u64
}

#[cfg(not(feature = "std"))]
fn fallback_now_millis() -> u64 {
    core_shard::current_tick()
}

/// Backs `proxima_core::time`'s `schedule_wake` — registers `waker` on the
/// calling worker's timer wheel to fire at `deadline_millis`.
#[unsafe(no_mangle)]
pub extern "Rust" fn proxima_time_external_schedule_wake(deadline_millis: u64, waker: Waker) {
    // on a worker: the per-core wheel. off a worker (a tokio-hosted client
    // whose binary links prime): a one-shot std timer thread, mirroring
    // proxima_core::time's own std_thread driver. no_std keeps the strict
    // contract.
    if core_shard::on_worker() {
        core_shard::schedule_wake(deadline_millis, waker);
    } else {
        fallback_schedule_wake(deadline_millis, waker);
    }
}

#[cfg(feature = "std")]
fn fallback_schedule_wake(deadline_millis: u64, waker: Waker) {
    let delay = deadline_millis.saturating_sub(fallback_now_millis());
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(delay));
        waker.wake();
    });
}

#[cfg(not(feature = "std"))]
fn fallback_schedule_wake(deadline_millis: u64, waker: Waker) {
    core_shard::schedule_wake(deadline_millis, waker);
}
