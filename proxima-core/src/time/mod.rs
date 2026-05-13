//! `proxima_core::time` (re-exported as `proxima::time`) — runtime-agnostic
//! timer primitives shaped like `tokio::time`. The active backend (host
//! thread, embassy, prime's per-core wheel, deterministic mock, or a
//! user-supplied hardware driver) is selected by the active [`Profile`] via
//! the `timer` axis; `build.rs` bakes a `&'static dyn Driver` symbol into
//! the build, so `Delay::poll`'s only runtime cost is one virtual dispatch.
//!
//! Folded in from the former `proxima-time` satellite crate (single
//! consumer: the workspace's timer surface) — `proxima_core::time::{sleep,
//! timeout, interval, Driver, Instant}` is unchanged for callers that used
//! to spell it `proxima_time::{..}`.
//!
//! [`Profile`]: proxima_build::Profile

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};
use core::time::Duration;

use futures::stream::Stream;
use pin_project_lite::pin_project;

pub mod drivers;
mod missed_tick;
mod types;

// `Driver` + `Instant` live in the dependency-free `types` module so a
// runtime crate (prime) can impl `Driver` + export `TIMER_DRIVER` without a
// cycle back through this crate's std-tier machinery. Re-exported so
// `proxima_core::time::{Driver, Instant}` is unchanged for callers.
pub use missed_tick::MissedTickBehavior;
pub use types::{Driver, Instant};

include!(concat!(env!("OUT_DIR"), "/proxima_time_bound_driver.rs"));

/// Capture the current monotonic time from the link-time-bound driver. (The
/// driver-free `Instant` in the leaf cannot reach `BOUND_DRIVER`, so `now()`
/// lives here.)
#[must_use]
pub fn now() -> Instant {
    BOUND_DRIVER.now()
}

/// Returned by [`timeout`] when the supplied duration passes before the
/// wrapped future completes. Matches `tokio::time::error::Elapsed`.
#[derive(Debug, thiserror::Error)]
#[error("deadline has elapsed")]
pub struct Elapsed(());

/// Future returned by [`sleep`] / [`sleep_until`]. Resolves when the
/// active driver's `now()` reaches `deadline`.
pub struct Sleep {
    deadline: Instant,
}

impl Sleep {
    /// Construct from an absolute deadline.
    #[must_use]
    pub fn until(deadline: Instant) -> Self {
        Self { deadline }
    }

    /// The deadline this `Sleep` is waiting on.
    #[must_use]
    pub fn deadline(&self) -> Instant {
        self.deadline
    }
}

impl Future for Sleep {
    type Output = ();
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
        if BOUND_DRIVER.now() >= self.deadline {
            Poll::Ready(())
        } else {
            BOUND_DRIVER.schedule_wake(self.deadline, context.waker().clone());
            Poll::Pending
        }
    }
}

/// Resolves after `duration` has elapsed on the bound driver's clock.
/// Shape-compatible with `tokio::time::sleep`.
#[must_use]
pub fn sleep(duration: Duration) -> Sleep {
    Sleep::until(now() + duration)
}

/// Resolves at the given deadline. Shape-compatible with
/// `tokio::time::sleep_until`. If the deadline is in the past, the
/// returned future resolves on first poll.
#[must_use]
pub fn sleep_until(deadline: Instant) -> Sleep {
    Sleep::until(deadline)
}

pin_project! {
    /// Future returned by [`timeout`] / [`timeout_at`].
    ///
    /// First poll checks the inner future before scheduling any wake.
    /// `deadline` is captured up front; on each `Pending` from the inner
    /// future, we re-check the clock and schedule a wake.
    struct Timeout<Fut> {
        #[pin]
        future: Fut,
        deadline: Instant,
    }
}

impl<Fut: Future> Future for Timeout<Fut> {
    type Output = Result<Fut::Output, Elapsed>;
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();

        if let Poll::Ready(value) = this.future.poll(context) {
            return Poll::Ready(Ok(value));
        }

        if BOUND_DRIVER.now() >= *this.deadline {
            return Poll::Ready(Err(Elapsed(())));
        }

        BOUND_DRIVER.schedule_wake(*this.deadline, context.waker().clone());
        Poll::Pending
    }
}

/// Race `future` against a timer of `duration`. Returns `Err(Elapsed)`
/// if the timer fires first.
pub fn timeout<Fut>(
    duration: Duration,
    future: Fut,
) -> impl Future<Output = Result<Fut::Output, Elapsed>>
where
    Fut: Future,
{
    Timeout {
        future,
        deadline: now() + duration,
    }
}

/// Race `future` against a deadline. Shape-compatible with
/// `tokio::time::timeout_at`.
pub fn timeout_at<Fut>(
    deadline: Instant,
    future: Fut,
) -> impl Future<Output = Result<Fut::Output, Elapsed>>
where
    Fut: Future,
{
    Timeout { future, deadline }
}

/// Stream + tick-future hybrid that yields every `period`, with the
/// first item delivered immediately. Matches `tokio::time::interval`
/// ergonomically (`.tick().await -> Instant`) AND exposes
/// `Stream<Item = ()>` for callers that prefer that shape.
#[must_use]
pub fn interval(period: Duration) -> Interval {
    Interval {
        period,
        next_deadline: None,
        missed_tick_behavior: MissedTickBehavior::default(),
    }
}

/// Like [`interval`] but the first tick fires at `start` rather than
/// immediately.
#[must_use]
pub fn interval_at(start: Instant, period: Duration) -> Interval {
    Interval {
        period,
        next_deadline: Some(start),
        missed_tick_behavior: MissedTickBehavior::default(),
    }
}

/// Periodic timer. `.tick().await` returns the [`Instant`] at which the
/// tick fired (matching tokio); also implements `Stream<Item = ()>`.
pub struct Interval {
    period: Duration,
    /// Absolute deadline of the next pending tick. `None` means
    /// "first tick should fire immediately" (post-construction state).
    next_deadline: Option<Instant>,
    missed_tick_behavior: MissedTickBehavior,
}

impl Interval {
    /// Set the missed-tick behavior. Default is `Burst`, matching tokio.
    pub fn set_missed_tick_behavior(&mut self, behavior: MissedTickBehavior) {
        self.missed_tick_behavior = behavior;
    }

    /// Current missed-tick behavior.
    #[must_use]
    pub fn missed_tick_behavior(&self) -> MissedTickBehavior {
        self.missed_tick_behavior
    }

    /// Configured period.
    #[must_use]
    pub fn period(&self) -> Duration {
        self.period
    }

    /// Future resolving to the [`Instant`] at which the tick fires.
    pub fn tick(&mut self) -> Tick<'_> {
        Tick { interval: self }
    }
}

/// Future returned by [`Interval::tick`].
pub struct Tick<'a> {
    interval: &'a mut Interval,
}

impl Future for Tick<'_> {
    type Output = Instant;
    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Instant> {
        let now = now();
        let interval = &mut *self.interval;

        // First tick fires immediately; arm the deadline for the next.
        if interval.next_deadline.is_none() {
            interval.next_deadline = Some(now + interval.period);
            return Poll::Ready(now);
        }

        let Some(deadline) = interval.next_deadline else {
            return Poll::Pending;
        };

        if now < deadline {
            BOUND_DRIVER.schedule_wake(deadline, context.waker().clone());
            return Poll::Pending;
        }

        // Tick has matured. Advance the deadline per missed-tick behavior.
        let fired_at = deadline;
        let next_deadline = match interval.missed_tick_behavior {
            MissedTickBehavior::Burst => fired_at + interval.period,
            MissedTickBehavior::Delay => now + interval.period,
            MissedTickBehavior::Skip => {
                let elapsed_periods = now
                    .saturating_duration_since(fired_at)
                    .as_nanos()
                    .checked_div(interval.period.as_nanos().max(1))
                    .unwrap_or(0);
                let skip = u32::try_from(elapsed_periods).unwrap_or(u32::MAX);
                fired_at + interval.period.saturating_mul(skip.saturating_add(1))
            }
        };
        interval.next_deadline = Some(next_deadline);
        Poll::Ready(fired_at)
    }
}

impl Stream for Interval {
    type Item = ();
    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<()>> {
        let interval: &mut Interval = &mut self;
        let mut tick = Tick { interval };
        match Pin::new(&mut tick).poll(context) {
            Poll::Ready(_instant) => Poll::Ready(Some(())),
            Poll::Pending => Poll::Pending,
        }
    }
}

// sleep/timeout/interval all dispatch through the bound driver (see
// drivers/unbound.rs); these tests assert on real elapsed wall-clock
// time, so they need time-driver-std-thread specifically, not just any
// driver (time-driver-mock's virtual clock wouldn't actually delay).
#[cfg(all(test, feature = "time-driver-std-thread"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use futures::stream::StreamExt;

    #[test]
    fn sleep_resolves_after_duration() {
        let start = now();
        block_on(sleep(Duration::from_millis(30)));
        assert!(now().saturating_duration_since(start) >= Duration::from_millis(25));
    }

    #[test]
    fn timeout_returns_ok_when_future_completes_in_time() {
        let outcome = block_on(timeout(Duration::from_millis(100), async { 7_u32 }));
        assert_eq!(outcome.expect("ok"), 7);
    }

    #[test]
    fn timeout_returns_elapsed_when_timer_fires_first() {
        let outcome: Result<u32, Elapsed> = block_on(timeout(Duration::from_millis(10), async {
            sleep(Duration::from_secs(1)).await;
            7_u32
        }));
        assert!(outcome.is_err());
    }

    #[test]
    fn interval_emits_first_tick_immediately_then_periodic() {
        block_on(async {
            let mut interval = interval(Duration::from_millis(20));
            let start = now();
            for _ in 0..3_u32 {
                interval.next().await.expect("tick");
            }
            // first tick is immediate; remaining 2 take ~40ms
            assert!(now().saturating_duration_since(start) >= Duration::from_millis(30));
        });
    }
}
