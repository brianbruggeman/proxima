//! Wasm host-clock driver. [`now`](Driver::now) and
//! [`schedule_wake`](Driver::schedule_wake) delegate to three host
//! imports the embedder supplies — there is no ambient monotonic clock
//! or timer on `wasm32-unknown-unknown`, so the host (browser glue or a
//! wasi shim) owns both. This mirrors the external-HAL model in
//! [`drivers`](crate::drivers): the host IS the hardware here.
//!
//! # Host ABI
//!
//! The embedder provides two imports and calls one export:
//!
//! ```text
//! // imports the wasm module expects from the host environment
//! fn proxima_time_now_micros() -> u64;        // monotonic micros since an epoch the host picks
//! fn proxima_time_request_wake(delay_micros: u64); // ask the host to call the export after delay
//!
//! // export the host calls when a requested wake is due (browser: setTimeout cb; wasi: after poll)
//! fn proxima_time_fire();
//! ```
//!
//! Over-firing is always safe: every [`Sleep`](crate::Sleep) re-checks
//! `now() >= deadline` on each poll, so a spurious `proxima_time_fire`
//! costs one extra poll and nothing else. The host need only fire at or
//! after the earliest requested deadline.

use alloc::vec::Vec;
use core::task::Waker;

#[cfg(target_arch = "wasm32")]
use crate::time::{Driver, Instant};
#[cfg(target_arch = "wasm32")]
use core::sync::atomic::{AtomicBool, Ordering};

/// Pending `(deadline_micros, waker)` registry. A single-slot spinlock
/// guards an `alloc` vector; wasm is single-threaded so the lock is
/// never actually contended, but the `Sync` bound on `BOUND_DRIVER`
/// requires interior synchronization to be sound on paper.
#[cfg(target_arch = "wasm32")]
struct Registry {
    lock: AtomicBool,
    entries: core::cell::UnsafeCell<Vec<(u64, Waker)>>,
}

// safety: every access goes through `with`, which holds `lock` for the
// duration of the `&mut` borrow; wasm32 has no second thread to race.
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for Registry {}

#[cfg(target_arch = "wasm32")]
impl Registry {
    const fn new() -> Self {
        Self {
            lock: AtomicBool::new(false),
            entries: core::cell::UnsafeCell::new(Vec::new()),
        }
    }

    fn with<R>(&self, body: impl FnOnce(&mut Vec<(u64, Waker)>) -> R) -> R {
        while self
            .lock
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            core::hint::spin_loop();
        }
        let entries = unsafe { &mut *self.entries.get() };
        let result = body(entries);
        self.lock.store(false, Ordering::Release);
        result
    }
}

#[cfg(target_arch = "wasm32")]
static REGISTRY: Registry = Registry::new();

/// Remove and return every waker whose deadline has passed, leaving the
/// still-pending entries in place. Pure so the wake fan-out is testable
/// without the host imports.
#[cfg_attr(
    not(target_arch = "wasm32"),
    allow(dead_code) // exercised by host unit tests + the wasm-gated Driver impl
)]
fn drain_due(entries: &mut Vec<(u64, Waker)>, now_micros: u64) -> Vec<Waker> {
    let mut due = Vec::new();
    entries.retain(|(deadline, waker)| {
        if *deadline <= now_micros {
            due.push(waker.clone());
            false
        } else {
            true
        }
    });
    due
}

/// Earliest still-pending deadline, if any — what the host should be
/// asked to fire next after a drain.
#[cfg_attr(
    not(target_arch = "wasm32"),
    allow(dead_code) // see drain_due
)]
fn next_deadline(entries: &[(u64, Waker)]) -> Option<u64> {
    entries.iter().map(|(deadline, _)| *deadline).min()
}

#[cfg(target_arch = "wasm32")]
unsafe extern "C" {
    fn proxima_time_now_micros() -> u64;
    fn proxima_time_request_wake(delay_micros: u64);
}

/// Singleton — referenced by the static binding `proxima-core`'s
/// `build.rs` emits when `time-driver-wasm` is the active driver.
#[cfg(target_arch = "wasm32")]
pub static DRIVER: WasmDriver = WasmDriver;

/// Host-delegated clock for wasm targets.
#[cfg(target_arch = "wasm32")]
pub struct WasmDriver;

#[cfg(target_arch = "wasm32")]
impl Driver for WasmDriver {
    fn now(&self) -> Instant {
        let micros = unsafe { proxima_time_now_micros() };
        Instant::from_monotonic(core::time::Duration::from_micros(micros))
    }

    fn schedule_wake(&self, deadline: Instant, waker: Waker) {
        let deadline_micros = deadline.into_monotonic().as_micros() as u64;
        let now_micros = unsafe { proxima_time_now_micros() };
        REGISTRY.with(|entries| entries.push((deadline_micros, waker)));
        unsafe { proxima_time_request_wake(deadline_micros.saturating_sub(now_micros)) };
    }
}

/// Host entry point: invoked when a requested wake comes due. Wakes
/// every now-elapsed waiter and re-arms the host for the next pending
/// deadline. `#[no_mangle]` so the embedder can call it by name.
#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn proxima_time_fire() {
    let now_micros = unsafe { proxima_time_now_micros() };
    let due = REGISTRY.with(|entries| drain_due(entries, now_micros));
    for waker in due {
        waker.wake();
    }
    if let Some(next) = REGISTRY.with(|entries| next_deadline(entries)) {
        unsafe { proxima_time_request_wake(next.saturating_sub(now_micros)) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::task::{RawWaker, RawWakerVTable, Waker};

    fn noop_waker() -> Waker {
        fn clone(_: *const ()) -> RawWaker {
            RawWaker::new(core::ptr::null(), &VTABLE)
        }
        fn noop(_: *const ()) {}
        static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
        unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VTABLE)) }
    }

    #[test]
    fn drain_due_returns_elapsed_and_keeps_pending() {
        let mut entries = alloc::vec![(100_u64, noop_waker()), (300, noop_waker())];

        let woken = drain_due(&mut entries, 200);

        assert_eq!(woken.len(), 1);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, 300);
    }

    #[test]
    fn drain_due_at_exact_deadline_is_elapsed() {
        let mut entries = alloc::vec![(250_u64, noop_waker())];

        let woken = drain_due(&mut entries, 250);

        assert_eq!(woken.len(), 1);
        assert!(entries.is_empty());
    }

    #[test]
    fn next_deadline_is_the_minimum_pending() {
        let entries = alloc::vec![
            (900_u64, noop_waker()),
            (400, noop_waker()),
            (700, noop_waker())
        ];

        assert_eq!(next_deadline(&entries), Some(400));
    }

    #[test]
    fn next_deadline_is_none_when_empty() {
        let entries: Vec<(u64, Waker)> = Vec::new();

        assert_eq!(next_deadline(&entries), None);
    }
}
