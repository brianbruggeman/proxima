//! Link-time timer hooks that back `proxima_core::time`'s
//! `time-driver-prime-wheel`.
//!
//! proxima-core cannot cargo-depend on prime — prime depends on
//! proxima-core directly (`prime/Cargo.toml`), so a `proxima-core -> prime`
//! edge would cycle. The prime-wheel driver is therefore wired by LINKAGE, not by a
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
//!
//! `std` is unconditional here despite the crate being no_std-capable:
//! `lib.rs` gates this module on `runtime-prime-reactor`, and that feature
//! itself names `std` (`Cargo.toml`), so a no_std build never reaches this
//! file. Writing `#[cfg(not(feature = "std"))]` fallbacks would be dead
//! arms describing a configuration that cannot be selected.
//!
//! `#[cfg(not(test))]`: `cargo test -p prime` dev-depends (transitively,
//! through the `proxima` umbrella crate needed for the `#[proxima::test]`
//! macro) on a second, normally-compiled copy of this same `prime` package
//! alongside the `--test`-compiled crate under test — mirrors
//! `proxima-core`'s own `external.rs` test host, which self-hosts these
//! same symbols under `#[cfg(test)]` for the opposite reason (no real
//! provider linked there). Here a real provider IS linked (the other prime
//! copy), so the `--test` copy must NOT also define these `#[unsafe(no_mangle)]`
//! symbols — two definitions of an unmangled symbol in one binary is a
//! linker error, not a Rust-level conflict (rustc's per-instantiation name
//! hashing doesn't apply to `no_mangle`), and GNU ld/rust-lld enforce it
//! even where the two are never both needed.

#[cfg(not(test))]
use core::task::Waker;

#[cfg(not(test))]
use crate::os::core_shard;

/// Backs `proxima_core::time::now()` under prime-wheel — milliseconds since
/// the calling worker's shard launched.
#[cfg(not(test))]
#[unsafe(no_mangle)]
pub extern "Rust" fn proxima_time_external_now_millis() -> u64 {
    // on a prime worker: the per-core wheel (hot path, unchanged). off a
    // worker — a tokio-hosted client in a mixed-runtime binary that links
    // prime — the wheel is unreachable, so read a monotonic wall clock
    // instead of aborting.
    match core_shard::current_tick_checked() {
        Some(tick) => tick,
        None => fallback_now_millis(),
    }
}

#[cfg(not(test))]
fn fallback_now_millis() -> u64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_millis() as u64
}

/// Backs `proxima_core::time`'s `schedule_wake` — registers `waker` on the
/// calling worker's timer wheel to fire at `deadline_millis`.
#[cfg(not(test))]
#[unsafe(no_mangle)]
pub extern "Rust" fn proxima_time_external_schedule_wake(deadline_millis: u64, waker: Waker) {
    // on a worker: the per-core wheel. off a worker (a tokio-hosted client
    // whose binary links prime): a one-shot std timer thread, mirroring
    // proxima_core::time's own std_thread driver.
    if core_shard::on_worker() {
        core_shard::schedule_wake(deadline_millis, waker);
    } else {
        fallback_schedule_wake(deadline_millis, waker);
    }
}

#[cfg(not(test))]
fn fallback_schedule_wake(deadline_millis: u64, waker: Waker) {
    let delay = deadline_millis.saturating_sub(fallback_now_millis());
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(delay));
        waker.wake();
    });
}
