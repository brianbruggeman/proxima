use core::time::Duration;

use crate::pipe::capabilities::Clock;
use proxima_core::time::Sleep;

/// Production [`Clock`] backed by `proxima-time`'s link-bound driver.
///
/// `now_nanos` reads the monotonic clock; `delay` hands back proxima-time's
/// concrete `Sleep` future, so a `Retry` built over this stays unboxed and
/// no-alloc. Build the controller's `Deadline` from the same clock so the
/// nanos origins agree.
#[derive(Debug, Clone, Copy, Default)]
pub struct TimeClock;

impl Clock for TimeClock {
    type Delay = Sleep;

    fn now_nanos(&self) -> u64 {
        u64::try_from(proxima_core::time::now().into_monotonic().as_nanos()).unwrap_or(u64::MAX)
    }

    fn delay(&self, dur: Duration) -> Sleep {
        proxima_core::time::sleep(dur)
    }
}

/// Canonical [`Clock`] test doubles — reachable from any crate via
/// `proxima-primitives`'s `test-support` feature (same convention as
/// `proxima-core`'s `time-driver-mock`: a lower crate ships the fake once,
/// behind a feature flag, instead of every caller reinventing an
/// `Arc<AtomicU64>`-backed clock).
///
/// Two doubles, not one, because they prove two different things:
///
/// - [`MockClock`] proves a combinator's async pend/wake state machine —
///   `delay` genuinely pends until [`MockClock::advance`] crosses the
///   deadline, waking the task through the real driver's `schedule_wake`.
///   Collapsing this into `RecordingClock` would make every test that needs
///   real pend/wake behavior (does the combinator park at the right time,
///   does it wake exactly once) pass trivially without exercising it.
/// - [`RecordingClock`] proves what a combinator *computed* (a backoff
///   schedule, a rate-limit refill) — `delay` never pends; it records the
///   requested [`Duration`] and resolves on first poll, so a sequential test
///   never has to drive a concurrent `advance()` to unblock it.
#[cfg(any(test, feature = "test-support"))]
pub mod testing {
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use core::future::{Ready, ready};
    use core::pin::Pin;
    use core::task::{Context, Poll};
    use core::time::Duration;
    use std::sync::Mutex;

    use portable_atomic::{AtomicU64, Ordering};
    use proxima_core::time::drivers::mock::MockDriver;
    use proxima_core::time::{Driver, Instant};

    use crate::pipe::capabilities::Clock;

    /// Deterministic `Clock` wrapping [`MockDriver`]. See the module doc for
    /// when to reach for this over [`RecordingClock`].
    #[derive(Clone, Default)]
    pub struct MockClock {
        driver: Arc<MockDriver>,
    }

    impl MockClock {
        #[must_use]
        pub fn new() -> Self {
            Self::default()
        }

        /// Move virtual time forward, firing any pending [`MockClock::delay`]
        /// whose deadline the new time now crosses.
        pub fn advance(&self, delta: Duration) {
            self.driver.advance(delta);
        }
    }

    /// [`MockClock::delay`]'s future: pends until the driver's clock reaches
    /// `deadline`, registering a real wake via `MockDriver::schedule_wake`
    /// — never a busy-poll.
    pub struct MockSleep {
        driver: Arc<MockDriver>,
        deadline: Instant,
    }

    impl core::future::Future for MockSleep {
        type Output = ();

        fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
            if self.driver.now() >= self.deadline {
                Poll::Ready(())
            } else {
                self.driver
                    .schedule_wake(self.deadline, context.waker().clone());
                Poll::Pending
            }
        }
    }

    impl Clock for MockClock {
        type Delay = MockSleep;

        fn now_nanos(&self) -> u64 {
            u64::try_from(self.driver.now().into_monotonic().as_nanos()).unwrap_or(u64::MAX)
        }

        fn delay(&self, duration: Duration) -> MockSleep {
            let deadline = self.driver.now() + duration;
            MockSleep {
                driver: self.driver.clone(),
                deadline,
            }
        }
    }

    /// Deterministic, never-pending `Clock`. See the module doc for when to
    /// reach for this over [`MockClock`].
    #[derive(Clone, Default)]
    pub struct RecordingClock {
        now_nanos: Arc<AtomicU64>,
        delays: Arc<Mutex<Vec<Duration>>>,
    }

    impl RecordingClock {
        /// `now_nanos` starts at zero; advance it with [`RecordingClock::advance`].
        #[must_use]
        pub fn new() -> Self {
            Self::default()
        }

        /// Start `now_nanos` at an arbitrary value instead of zero — needed
        /// when the code under test cares about the clock's absolute
        /// magnitude (e.g. distinguishing an elapsed-since-origin reading
        /// from an absolute-epoch reading).
        #[must_use]
        pub fn at(now_nanos: u64) -> Self {
            Self {
                now_nanos: Arc::new(AtomicU64::new(now_nanos)),
                delays: Arc::new(Mutex::new(Vec::new())),
            }
        }

        /// Move `now_nanos` forward. `delay` never needs this to resolve —
        /// only code that reads `now_nanos` (deadline/refill arithmetic)
        /// observes it.
        pub fn advance(&self, duration: Duration) {
            let elapsed_nanos = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
            self.now_nanos.fetch_add(elapsed_nanos, Ordering::Relaxed);
        }

        /// Every duration a caller has asked `delay` to wait, in call order —
        /// the backoff/refill schedule a test asserts against.
        #[must_use]
        pub fn delays(&self) -> Vec<Duration> {
            self.delays
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone()
        }
    }

    impl Clock for RecordingClock {
        type Delay = Ready<()>;

        fn now_nanos(&self) -> u64 {
            self.now_nanos.load(Ordering::Relaxed)
        }

        fn delay(&self, duration: Duration) -> Ready<()> {
            self.delays
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(duration);
            ready(())
        }
    }
}
