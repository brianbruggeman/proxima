//! Always-available fallback driver. Selected when no `time-driver-*`
//! feature is active *and* no `PROXIMA_PROFILE` is set — most often during
//! alloc-only cross-compile verification of downstream crates (the
//! consumer wants to prove its source compiles for a no_std target;
//! it doesn't actually exercise the time primitives at runtime).
//!
//! Any call to [`now`](Driver::now) or
//! [`schedule_wake`](Driver::schedule_wake) panics with a message
//! pointing at the missing driver configuration. This is intentional:
//! shipping a build that links the unbound driver and *does* use timer
//! primitives is a configuration error, and silent zero-time / silent
//! never-fires would mask it.

use core::task::Waker;

use crate::time::{Driver, Instant};

/// Singleton — referenced by the build.rs fallback when no other
/// driver is selectable.
pub static DRIVER: UnboundDriver = UnboundDriver;

/// Zero-sized panics-on-use driver.
pub struct UnboundDriver;

impl Driver for UnboundDriver {
    fn now(&self) -> Instant {
        panic!(
            "proxima_core::time has no driver bound; activate a `time-driver-*` \
             feature (e.g. time-driver-std-thread, time-driver-mock) or set \
             PROXIMA_PROFILE"
        );
    }

    fn schedule_wake(&self, _deadline: Instant, _waker: Waker) {
        panic!(
            "proxima_core::time has no driver bound; activate a `time-driver-*` \
             feature (e.g. time-driver-std-thread, time-driver-mock) or set \
             PROXIMA_PROFILE"
        );
    }
}
