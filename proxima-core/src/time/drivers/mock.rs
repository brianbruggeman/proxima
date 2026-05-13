//! Deterministic driver for tests. Time advances only when the test
//! explicitly calls [`MockDriver::advance`]; pending wakers fire when
//! their deadline crosses the mock clock.
//!
//! Currently requires `std` (uses `std::sync::Mutex`). An alloc-only
//! mock is a follow-up — would need `spin::Mutex` or
//! `critical_section::Mutex` as a no_std-compatible substitute.

use core::task::Waker;
use core::time::Duration;
use std::sync::Mutex;
use std::vec::Vec;

use crate::time::{Driver, Instant};

/// Singleton — referenced by the static binding when `timer = "mock"`.
pub static DRIVER: MockDriver = MockDriver::new();

/// Deterministic clock + waker registry.
pub struct MockDriver {
    state: Mutex<MockState>,
}

struct MockState {
    now: Duration,
    pending: Vec<(Instant, Waker)>,
}

impl MockDriver {
    /// Construct in a static context — both inner fields are
    /// const-constructible.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: Mutex::new(MockState {
                now: Duration::ZERO,
                pending: Vec::new(),
            }),
        }
    }

    /// Advance the mock clock by `delta` and wake any registrations
    /// whose deadline has matured.
    pub fn advance(&self, delta: Duration) {
        let to_wake = {
            let mut state = self.state.lock().unwrap_or_else(|err| err.into_inner());
            state.now = state.now.saturating_add(delta);
            let cutoff = Instant::from_monotonic(state.now);
            let mut still_pending = Vec::with_capacity(state.pending.len());
            let mut fired = Vec::new();
            for entry in state.pending.drain(..) {
                if entry.0 <= cutoff {
                    fired.push(entry.1);
                } else {
                    still_pending.push(entry);
                }
            }
            state.pending = still_pending;
            fired
        };
        for waker in to_wake {
            waker.wake();
        }
    }

    /// Reset the mock clock to zero and discard all registrations.
    /// Useful between tests.
    pub fn reset(&self) {
        let mut state = self.state.lock().unwrap_or_else(|err| err.into_inner());
        state.now = Duration::ZERO;
        state.pending.clear();
    }
}

impl Default for MockDriver {
    fn default() -> Self {
        Self::new()
    }
}

impl Driver for MockDriver {
    fn now(&self) -> Instant {
        let state = self.state.lock().unwrap_or_else(|err| err.into_inner());
        Instant::from_monotonic(state.now)
    }

    fn schedule_wake(&self, deadline: Instant, waker: Waker) {
        let mut state = self.state.lock().unwrap_or_else(|err| err.into_inner());
        let now = Instant::from_monotonic(state.now);
        if deadline <= now {
            drop(state);
            waker.wake();
        } else {
            state.pending.push((deadline, waker));
        }
    }
}
