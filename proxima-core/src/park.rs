//! `SlotPark` — a futex park for "wait until a bounded resource frees a slot".
//!
//! The lossless-backpressure primitive: when a bounded resource (a full ring, a
//! saturated pool, a closed window) can't admit a producer, the producer parks
//! here instead of dropping or busy-spinning; whoever frees a slot calls
//! [`wake_all`](SlotPark::wake_all). It is the wait/wake half of any
//! block-until-room design — decoupled from *what* the resource is, so a queue,
//! an arena, a rate limiter, or a connection pool can all reuse it. proxima's
//! telemetry recorder is the first consumer, not the only one.
//!
//! ## Why a futex, not a `Condvar`
//!
//! A `std::sync::Condvar` needs a `Mutex` companion and is std-only. This park
//! is a raw futex (`atomic-wait`: Linux `futex`, Windows `WaitOnAddress`, macOS
//! `__ulock_wait`) over a single `AtomicU32` epoch word — **mutex-free and
//! no_std** (it needs an OS for the syscall, but not Rust `std`). That matters
//! because the lossless guarantee it backs must be available on the hosted
//! no_std / kernel-bypass tiers, not just under `std`.
//!
//! It is behind the `park` feature (which pulls `atomic-wait`) and is therefore
//! **absent on bare-metal no-OS targets** — there is no futex without a kernel,
//! and "park a thread" is a category error with no scheduler. Those targets
//! spin, use a hardware wait, or an RTOS primitive instead.
//!
//! ## The epoch + the lost-wakeup window
//!
//! The single `AtomicU32` is a wake **epoch**, not the resource state. A waker
//! bumps it before waking; a parker snapshots it before the final resource
//! re-check and passes the snapshot to [`wait`](SlotPark::wait), which returns
//! immediately if the epoch already moved. That closes the classic lost-wakeup
//! window (a free that lands between the parker's last full-check and its park):
//! either the re-check sees the freed slot, or the epoch moved (wait returns at
//! once), or the parker is genuinely parked and the next free wakes it.
//!
//! The caller drives the loop (this type does not own the resource re-check),
//! in this exact order — announce, snapshot, RE-CHECK, wait:
//!
//! ```ignore
//! loop {
//!     if let Some(v) = try_take() { return v; }   // fast path: room now
//!     before_park();                              // e.g. nudge a drain pump
//!     let epoch = park.begin_wait();              // announce + snapshot
//!     if let Some(v) = try_take() {               // RE-CHECK closes the window
//!         park.end_wait();
//!         return v;
//!     }
//!     park.wait(epoch);                           // futex-park until epoch moves
//!     park.end_wait();
//! }
//! ```
//!
//! `waiters` is a pure optimisation: [`wake_all`](SlotPark::wake_all) skips the
//! epoch bump + wake syscall entirely when it is zero (the steady state), so a
//! consumer that never exerts backpressure pays nothing.

use core::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

/// A futex wait-for-a-freed-slot park. Shared by `&`, so many producers park on
/// one instance and any thread can [`wake_all`](Self::wake_all) them. See the
/// module docs for the caller's announce → snapshot → re-check → wait loop.
pub struct SlotPark {
    /// Wake epoch (the futex word). Bumped by `wake_all`, snapshotted by
    /// `begin_wait`, and the value `wait` blocks against.
    epoch: AtomicU32,
    /// Live parkers, so `wake_all` can no-op when none are waiting.
    waiters: AtomicUsize,
    /// Cumulative parks — the backpressure observability signal: nonzero/rising
    /// means producers are stalling on a slow consumer.
    parked: AtomicU64,
}

impl Default for SlotPark {
    fn default() -> Self {
        Self::new()
    }
}

impl SlotPark {
    /// A fresh, empty park. `const` so it can back a `static` with no lazy init.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            epoch: AtomicU32::new(0),
            waiters: AtomicUsize::new(0),
            parked: AtomicU64::new(0),
        }
    }

    /// Announce this thread as a waiter and snapshot the wake epoch. Call it
    /// AFTER the fast-path check fails and BEFORE the final resource re-check;
    /// pass the returned epoch to [`wait`](Self::wait), and pair every call with
    /// [`end_wait`](Self::end_wait). Announcing first is what lets a concurrent
    /// [`wake_all`](Self::wake_all) see us and bump the epoch rather than skip.
    #[must_use]
    pub fn begin_wait(&self) -> u32 {
        self.waiters.fetch_add(1, Ordering::AcqRel);
        self.epoch.load(Ordering::Acquire)
    }

    /// Park until the epoch moves off `epoch` (a waker fired) — or a spurious
    /// futex wake, which the caller's loop absorbs by re-checking. `epoch` MUST
    /// come from a preceding [`begin_wait`](Self::begin_wait), with a resource
    /// re-check in between, or the lost-wakeup window is open.
    pub fn wait(&self, epoch: u32) {
        self.parked.fetch_add(1, Ordering::Relaxed);
        atomic_wait::wait(&self.epoch, epoch);
    }

    /// Stop being a waiter — after [`wait`](Self::wait) returns, or after the
    /// re-check between `begin_wait` and `wait` succeeded (so `wait` is skipped).
    pub fn end_wait(&self) {
        self.waiters.fetch_sub(1, Ordering::AcqRel);
    }

    /// Wake every parked producer after freeing a slot. Bumping the epoch before
    /// the wake closes the lost-wakeup window; the `waiters == 0` guard skips the
    /// syscall entirely in the steady state (no producer is parked).
    pub fn wake_all(&self) {
        if self.waiters.load(Ordering::Acquire) == 0 {
            return;
        }
        self.epoch.fetch_add(1, Ordering::Release);
        atomic_wait::wake_all(&self.epoch);
    }

    /// Cumulative parks since construction — the "consumer isn't keeping up"
    /// backpressure indicator (it climbs before any drop under a lossless policy).
    #[must_use]
    pub fn parked(&self) -> u64 {
        self.parked.load(Ordering::Relaxed)
    }

    /// Producers parked right now (a snapshot).
    #[must_use]
    pub fn waiters(&self) -> usize {
        self.waiters.load(Ordering::Acquire)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    extern crate std;

    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::thread;

    // a producer parks on a "full" flag; a consumer clears it + wakes. proves the
    // announce -> re-check -> wait -> wake handshake delivers the wakeup (no hang,
    // no lost wakeup) with real threads.
    #[test]
    fn parks_until_woken_then_proceeds() {
        let park = Arc::new(SlotPark::new());
        let full = Arc::new(AtomicBool::new(true));

        let producer = {
            let park = Arc::clone(&park);
            let full = Arc::clone(&full);
            thread::spawn(move || {
                loop {
                    if !full.load(Ordering::Acquire) {
                        return;
                    }
                    let epoch = park.begin_wait();
                    if !full.load(Ordering::Acquire) {
                        park.end_wait();
                        return;
                    }
                    park.wait(epoch);
                    park.end_wait();
                }
            })
        };

        // let the producer reach the park, then free the slot + wake it.
        while park.waiters() == 0 {
            thread::yield_now();
        }
        assert!(park.parked() >= 1, "producer parked");
        full.store(false, Ordering::Release);
        park.wake_all();

        producer.join().expect("producer returns after wake");
    }

    #[test]
    fn wake_all_is_a_noop_with_no_waiters() {
        let park = SlotPark::new();
        park.wake_all(); // must not panic / block when nobody is parked
        assert_eq!(park.waiters(), 0);
        assert_eq!(park.parked(), 0);
    }
}
