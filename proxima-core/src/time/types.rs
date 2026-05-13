//! Leaf types for `proxima_core::time`: the [`Driver`] trait + [`Instant`].
//!
//! Kept dependency-free and no_std + alloc clean so a runtime crate (e.g.
//! `prime`) can `impl Driver` and export a `TIMER_DRIVER` static without
//! reaching into the rest of this module's std-tier machinery. Folded in
//! from the former `proxima-time-types` satellite crate (single consumer:
//! this crate) — `proxima_core::time::{Driver, Instant}` is unchanged for
//! callers.
//!
//! `Instant::now()` / `elapsed()` are NOT here — they read the bound
//! driver, which lives in the `time` module root; use
//! `proxima_core::time::now()` for those.

use core::cmp::Ordering;
use core::ops::{Add, AddAssign, Sub, SubAssign};
use core::task::Waker;
use core::time::Duration;

/// The timer driver interface every backend implements. Selected at build time
/// by the `timer` axis of the active profile; bound at link time via the static
/// `BOUND_DRIVER` symbol `proxima-time`'s build emits.
///
/// Implementations must be `Sync` because `BOUND_DRIVER` is a global `&'static`.
/// They are typically zero-sized or use interior mutability.
pub trait Driver: Sync {
    /// Current monotonic time according to this driver.
    fn now(&self) -> Instant;

    /// Arrange for `waker` to be woken at or after `deadline`. Drivers must be
    /// tolerant of multiple registrations from the same waker; `Waker::wake` is
    /// idempotent for already-pending tasks.
    fn schedule_wake(&self, deadline: Instant, waker: Waker);
}

/// Monotonic point in time. Internally a `Duration` since the active driver's
/// epoch — comparisons and arithmetic are well-defined as long as both operands
/// came from the same driver instance, which the link-time-bound `BOUND_DRIVER`
/// guarantees.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Instant {
    monotonic: Duration,
}

impl Instant {
    /// Construct from a raw monotonic duration. Driver implementations use this;
    /// application code uses `proxima_core::time::now()`.
    #[must_use]
    pub const fn from_monotonic(monotonic: Duration) -> Self {
        Self { monotonic }
    }

    /// The raw monotonic duration since the driver's epoch.
    #[must_use]
    pub const fn into_monotonic(self) -> Duration {
        self.monotonic
    }

    /// Duration since `earlier`. Saturates at zero if `earlier` is in the future.
    #[must_use]
    pub fn saturating_duration_since(self, earlier: Self) -> Duration {
        self.monotonic.saturating_sub(earlier.monotonic)
    }

    /// Duration since `earlier`; zero if `earlier` is in the future.
    #[must_use]
    pub fn duration_since(self, earlier: Self) -> Duration {
        self.saturating_duration_since(earlier)
    }

    /// Add `duration`, returning `None` on overflow.
    #[must_use]
    pub fn checked_add(self, duration: Duration) -> Option<Self> {
        self.monotonic
            .checked_add(duration)
            .map(Self::from_monotonic)
    }

    /// Subtract `duration`, returning `None` on underflow.
    #[must_use]
    pub fn checked_sub(self, duration: Duration) -> Option<Self> {
        self.monotonic
            .checked_sub(duration)
            .map(Self::from_monotonic)
    }
}

impl PartialOrd for Instant {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Instant {
    fn cmp(&self, other: &Self) -> Ordering {
        self.monotonic.cmp(&other.monotonic)
    }
}

impl Add<Duration> for Instant {
    type Output = Self;
    fn add(self, duration: Duration) -> Self {
        Self::from_monotonic(self.monotonic + duration)
    }
}

impl AddAssign<Duration> for Instant {
    fn add_assign(&mut self, duration: Duration) {
        self.monotonic += duration;
    }
}

impl Sub<Duration> for Instant {
    type Output = Self;
    fn sub(self, duration: Duration) -> Self {
        Self::from_monotonic(self.monotonic - duration)
    }
}

impl SubAssign<Duration> for Instant {
    fn sub_assign(&mut self, duration: Duration) {
        self.monotonic -= duration;
    }
}

impl Sub<Instant> for Instant {
    type Output = Duration;
    fn sub(self, earlier: Instant) -> Duration {
        self.saturating_duration_since(earlier)
    }
}
