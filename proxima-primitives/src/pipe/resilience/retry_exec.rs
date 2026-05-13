use core::future::Future;
use core::time::Duration;

use crate::pipe::capabilities::{Clock, Retryable};
use crate::pipe::primitives::Pipe;
use crate::pipe::resilience::retry::{RetryAction, RetryController};

/// Per-attempt jitter seed. `RetryController` owns the decision (how much to
/// jitter); this only turns one stable seed into an uncorrelated stream so
/// concurrent retriers don't march in lockstep. Deterministic given the seed.
fn splitmix64(state: u64) -> u64 {
    let state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let state = (state ^ (state >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let state = (state ^ (state >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    state ^ (state >> 31)
}

/// Drives an inner [`Pipe`] under a [`RetryController`], sleeping between
/// attempts via the injected [`Clock`]. No alloc: the inner call future and the
/// clock's `Delay` live inline in the returned state machine — nothing boxed.
#[derive(Debug, Clone)]
pub struct Retry<Inner, Clk> {
    inner: Inner,
    controller: RetryController,
    clock: Clk,
    rand_seed: u64,
}

impl<Inner, Clk> Retry<Inner, Clk> {
    #[must_use]
    pub fn new(inner: Inner, controller: RetryController, clock: Clk, rand_seed: u64) -> Self {
        Self {
            inner,
            controller,
            clock,
            rand_seed,
        }
    }
}

impl<Inner, Clk> Pipe for Retry<Inner, Clk>
where
    Inner: Pipe,
    Inner::In: Clone,
    Inner::Out: Retryable,
    Clk: Clock,
{
    type In = Inner::In;
    type Out = Inner::Out;
    type Err = Inner::Err;

    fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>> {
        async move {
            let mut prev_delay = Duration::ZERO;
            let mut attempt = 0u32;
            loop {
                let outcome = self.inner.call(input.clone()).await;
                let rand = splitmix64(self.rand_seed ^ u64::from(attempt));
                let now_nanos = self.clock.now_nanos();
                match self
                    .controller
                    .on_outcome(attempt, &outcome, now_nanos, rand, prev_delay)
                {
                    RetryAction::Done | RetryAction::Exhausted => return outcome,
                    RetryAction::Retry { after } => {
                        self.clock.delay(after).await;
                        prev_delay = after;
                        attempt += 1;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::pipe::resilience::backoff::{Backoff, Jitter};
    use crate::pipe::resilience::deadline::Deadline;
    use crate::pipe::retry_rules::RetryRules;
    use core::cell::RefCell;
    use core::future::{Ready, ready};
    use core::task::Poll;
    use std::rc::Rc;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::vec::Vec;

    fn block_on<Fut: Future>(future: Fut) -> Fut::Output {
        let mut pinned = core::pin::pin!(future);
        let mut cx = core::task::Context::from_waker(core::task::Waker::noop());
        loop {
            if let Poll::Ready(output) = pinned.as_mut().poll(&mut cx) {
                return output;
            }
        }
    }

    /// Deterministic `Clock`: `delay` resolves immediately (no wall-clock — Retry
    /// is sequential, not a race) and records each requested duration so a test
    /// can assert the backoff schedule.
    #[derive(Clone)]
    struct MockClock {
        now_nanos: u64,
        delays: Rc<RefCell<Vec<Duration>>>,
    }

    impl MockClock {
        fn new() -> Self {
            Self::at(0)
        }
        fn at(now_nanos: u64) -> Self {
            Self {
                now_nanos,
                delays: Rc::new(RefCell::new(Vec::new())),
            }
        }
        fn delays(&self) -> Vec<Duration> {
            self.delays.borrow().clone()
        }
    }

    impl Clock for MockClock {
        type Delay = Ready<()>;
        fn now_nanos(&self) -> u64 {
            self.now_nanos
        }
        fn delay(&self, dur: Duration) -> Ready<()> {
            self.delays.borrow_mut().push(dur);
            ready(())
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct StatusOut(Option<u16>);

    impl Retryable for StatusOut {
        fn retry_status(&self) -> Option<u16> {
            self.0
        }
        fn is_success(&self) -> bool {
            self.0.is_none()
        }
    }

    /// Fails (status 503) until `succeed_at` calls have been made, then succeeds.
    struct FlakyPipe {
        calls: Arc<AtomicU32>,
        succeed_at: u32,
    }

    impl Pipe for FlakyPipe {
        type In = u32;
        type Out = StatusOut;
        type Err = &'static str;

        fn call(&self, _input: u32) -> impl Future<Output = Result<StatusOut, &'static str>> {
            let seen = self.calls.fetch_add(1, Ordering::Relaxed);
            let succeed = seen >= self.succeed_at;
            async move {
                if succeed {
                    Ok(StatusOut(None))
                } else {
                    Ok(StatusOut(Some(503)))
                }
            }
        }
    }

    fn controller(max_attempts: u32, backoff: Backoff) -> RetryController {
        RetryController {
            rules: RetryRules::default(),
            backoff,
            jitter: Jitter::None,
            max_attempts,
            deadline: None,
        }
    }

    #[test]
    fn success_on_first_attempt_is_one_call_and_no_delays() {
        let calls = Arc::new(AtomicU32::new(0));
        let pipe = FlakyPipe {
            calls: calls.clone(),
            succeed_at: 0,
        };
        let clock = MockClock::new();
        let retry = Retry::new(
            pipe,
            controller(3, Backoff::Constant(Duration::from_millis(50))),
            clock,
            0,
        );

        let result = block_on(Pipe::call(&retry, 0));

        assert_eq!(result, Ok(StatusOut(None)));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn two_failures_then_success_is_three_calls_and_two_delays() {
        let calls = Arc::new(AtomicU32::new(0));
        let pipe = FlakyPipe {
            calls: calls.clone(),
            succeed_at: 2,
        };
        let retry = Retry::new(
            pipe,
            controller(5, Backoff::Constant(Duration::from_millis(50))),
            MockClock::new(),
            0,
        );

        let result = block_on(Pipe::call(&retry, 0));

        assert_eq!(result, Ok(StatusOut(None)));
        assert_eq!(
            calls.load(Ordering::Relaxed),
            3,
            "two failures then a success"
        );
    }

    #[test]
    fn exhausted_at_max_attempts_returns_last_outcome() {
        let calls = Arc::new(AtomicU32::new(0));
        let pipe = FlakyPipe {
            calls: calls.clone(),
            succeed_at: u32::MAX,
        };
        let retry = Retry::new(
            pipe,
            controller(3, Backoff::Constant(Duration::from_millis(50))),
            MockClock::new(),
            0,
        );

        let result = block_on(Pipe::call(&retry, 0));

        assert_eq!(
            result,
            Ok(StatusOut(Some(503))),
            "final retryable outcome propagates"
        );
        assert_eq!(
            calls.load(Ordering::Relaxed),
            3,
            "exactly max_attempts calls"
        );
    }

    #[test]
    fn past_deadline_from_clock_exhausts_before_any_retry() {
        let calls = Arc::new(AtomicU32::new(0));
        let pipe = FlakyPipe {
            calls: calls.clone(),
            succeed_at: u32::MAX,
        };
        let mut controller = controller(10, Backoff::Constant(Duration::from_millis(50)));
        controller.deadline = Some(Deadline::new(0, Duration::from_secs(1)));
        // clock reads 2s — past the 1s deadline, so the first failure exhausts.
        let retry = Retry::new(pipe, controller, MockClock::at(2_000_000_000), 0);

        let result = block_on(Pipe::call(&retry, 0));

        assert_eq!(result, Ok(StatusOut(Some(503))));
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "deadline cut it off before retrying"
        );
    }

    #[test]
    fn delays_follow_the_exponential_schedule() {
        let calls = Arc::new(AtomicU32::new(0));
        let pipe = FlakyPipe {
            calls: calls.clone(),
            succeed_at: u32::MAX,
        };
        let clock = MockClock::new();
        let probe = clock.clone();
        let backoff = Backoff::Exponential {
            initial: Duration::from_millis(100),
            factor: 2,
            max: Duration::from_millis(2000),
        };
        let retry = Retry::new(pipe, controller(4, backoff), clock, 0);

        let _ = block_on(Pipe::call(&retry, 0));

        // 4 attempts → 3 sleeps between them: 100ms, 200ms, 400ms.
        assert_eq!(
            probe.delays(),
            std::vec![
                Duration::from_millis(100),
                Duration::from_millis(200),
                Duration::from_millis(400),
            ],
        );
    }
}
