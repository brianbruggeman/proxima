//! `deadline` — a timeout as a fired completion: bound an operation by a
//! clock, not a sleep.
//!
//! `signal` taught fire-once completion (`Signal::fire`, sticky, awaited
//! once); `clock` taught time as an injectable seam (`now_nanos`, `delay`,
//! driven by hand with `FakeClock`). A deadline is the two combined: a
//! [`Deadline`](proxima_primitives::pipe::resilience::Deadline) is a plain
//! timestamp comparison (`expired(now_nanos)`), and wrapping an operation in
//! one turns "has the clock passed this instant" into a completion that
//! fires exactly once — either the inner operation finishes first (the
//! deadline never fires), or the clock crosses the deadline first (the
//! deadline fires, the inner operation is dropped without finishing, and a
//! `Signal` records that it happened, exactly once, for good).
//!
//! No real sleeps anywhere: both outcomes are driven by hand, advancing a
//! `FakeClock` between polls exactly like `clock` did. "Slow" for the inner
//! operation is modeled as a poll count, not real time, so it advances on
//! its own schedule, independent of the clock.
//!
//! Run: `cargo run --example deadline`

use core::cell::Cell;
use core::fmt;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};
use core::time::Duration;
use std::rc::Rc;

use proxima_core::signal::Signal;
use proxima_primitives::pipe::capabilities::Clock;
use proxima_primitives::pipe::resilience::Deadline;

/// A `Clock` backed by a shared `Cell<u64>` instead of the wall. `now_nanos`
/// reads the cell; `advance` is the only thing that ever moves it forward —
/// the same mechanism `clock` used, reused here so the deadline's notion of
/// "now" is exactly as deterministic as any other timer-driven combinator.
#[derive(Clone, Default)]
struct FakeClock {
    now_nanos: Rc<Cell<u64>>,
}

impl FakeClock {
    fn advance(&self, elapsed: Duration) {
        let elapsed_nanos = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        self.now_nanos
            .set(self.now_nanos.get().saturating_add(elapsed_nanos));
    }
}

impl Clock for FakeClock {
    type Delay = FakeDelay;

    fn now_nanos(&self) -> u64 {
        self.now_nanos.get()
    }

    fn delay(&self, dur: Duration) -> FakeDelay {
        let wait_nanos = u64::try_from(dur.as_nanos()).unwrap_or(u64::MAX);
        FakeDelay {
            now_nanos: self.now_nanos.clone(),
            deadline: self.now_nanos.get().saturating_add(wait_nanos),
        }
    }
}

/// The future `FakeClock::delay` hands back. Unused by `DeadlineGuard`
/// below (it checks `Deadline::expired` against `now_nanos` directly, not a
/// racing timer future) but required to make `FakeClock` a real `Clock` —
/// the same seam `Retry`/`Backoff` build on.
struct FakeDelay {
    now_nanos: Rc<Cell<u64>>,
    deadline: u64,
}

impl Future for FakeDelay {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<()> {
        if self.now_nanos.get() >= self.deadline {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

/// A pretend slow operation. "Slow" is modeled as a poll count, not real
/// time: it returns `Pending` `pending_polls` times, then `Ready` on the
/// poll after that. Driving it this way keeps it independent of the clock —
/// the example advances the clock and the work on separate schedules, so
/// either one can win the race.
struct SlowWork {
    pending_polls: Cell<u32>,
}

impl SlowWork {
    fn new(pending_polls: u32) -> Self {
        Self {
            pending_polls: Cell::new(pending_polls),
        }
    }
}

impl Future for SlowWork {
    type Output = &'static str;

    fn poll(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<&'static str> {
        let remaining = self.pending_polls.get();
        if remaining == 0 {
            return Poll::Ready("operation complete");
        }
        self.pending_polls.set(remaining - 1);
        Poll::Pending
    }
}

/// The error a [`DeadlineGuard`] resolves to when the clock crosses the
/// deadline before the inner future does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DeadlineExceeded {
    now_nanos: u64,
}

impl fmt::Display for DeadlineExceeded {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "deadline exceeded at {} ns", self.now_nanos)
    }
}

/// Wraps an inner future with a [`Deadline`] checked against an injected
/// [`Clock`]. Polls the inner future first; if it isn't ready yet, checks
/// whether the clock has crossed the deadline. The moment it has, the guard
/// fires `fired` — once, for good, the same sticky semantics `signal`
/// taught — and resolves `Err`, dropping the inner future without polling
/// it again. A timeout, seen this way, is nothing but a completion that
/// fires on elapsed time instead of on inner readiness.
struct DeadlineGuard<InnerFuture, ClockImpl: Clock> {
    inner: InnerFuture,
    clock: ClockImpl,
    deadline: Deadline,
    fired: Signal,
}

impl<InnerFuture, ClockImpl> Future for DeadlineGuard<InnerFuture, ClockImpl>
where
    InnerFuture: Future<Output = &'static str> + Unpin,
    ClockImpl: Clock + Unpin,
{
    type Output = Result<&'static str, DeadlineExceeded>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();

        if let Poll::Ready(output) = Pin::new(&mut this.inner).poll(context) {
            return Poll::Ready(Ok(output));
        }

        let now_nanos = this.clock.now_nanos();
        if this.deadline.expired(now_nanos) {
            this.fired.fire();
            return Poll::Ready(Err(DeadlineExceeded { now_nanos }));
        }

        Poll::Pending
    }
}

/// Polls a future once against a no-op waker and returns whatever it
/// reports — `Pending` included. Driving it by hand, one poll per call,
/// is what lets the example interleave `FakeClock::advance` between polls.
fn poll_once<Fut: Future>(future: Pin<&mut Fut>) -> Poll<Fut::Output> {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    future.poll(&mut context)
}

fn run_case_finishes_in_time() {
    let clock = FakeClock::default();
    let deadline = Deadline::new(clock.now_nanos(), Duration::from_secs(3));
    let fired = Signal::new();
    let guard = DeadlineGuard {
        inner: SlowWork::new(2),
        clock: clock.clone(),
        deadline,
        fired: fired.clone(),
    };
    let mut guard = core::pin::pin!(guard);

    println!("budget: 3s, inner needs 2 polls to finish");

    match poll_once(guard.as_mut()) {
        Poll::Pending => {
            println!("  poll #1 (t=0s): Pending  — inner working, deadline not crossed")
        }
        Poll::Ready(_) => unreachable!("inner needs 2 polls, not ready on poll #1"),
    }
    clock.advance(Duration::from_secs(1));
    println!("  advance(+1s) -> now_nanos = {}", clock.now_nanos());

    match poll_once(guard.as_mut()) {
        Poll::Pending => {
            println!("  poll #2 (t=1s): Pending  — inner working, deadline not crossed")
        }
        Poll::Ready(_) => unreachable!("inner needs 2 polls, not ready on poll #2"),
    }
    clock.advance(Duration::from_secs(1));
    println!("  advance(+1s) -> now_nanos = {}", clock.now_nanos());

    match poll_once(guard.as_mut()) {
        Poll::Ready(Ok(outcome)) => println!(
            "  poll #3 (t=2s): Ready(Ok(\"{outcome}\"))  — inner finished first, deadline never fired (budget was 3s)"
        ),
        other => unreachable!("inner should have finished by poll #3, got {other:?}"),
    }

    assert!(
        !fired.is_fired(),
        "deadline must not fire when the inner op wins the race"
    );
    println!("  fired.is_fired() = false  — confirmed\n");
}

fn run_case_deadline_fires() {
    let clock = FakeClock::default();
    let deadline = Deadline::new(clock.now_nanos(), Duration::from_secs(2));
    let fired = Signal::new();
    let guard = DeadlineGuard {
        inner: SlowWork::new(5),
        clock: clock.clone(),
        deadline,
        fired: fired.clone(),
    };
    let mut guard = core::pin::pin!(guard);

    println!("budget: 2s, inner needs 5 polls to finish — too slow");

    match poll_once(guard.as_mut()) {
        Poll::Pending => {
            println!("  poll #1 (t=0s): Pending  — inner working, deadline not crossed")
        }
        Poll::Ready(_) => unreachable!("inner needs 5 polls, not ready on poll #1"),
    }
    clock.advance(Duration::from_secs(1));
    println!("  advance(+1s) -> now_nanos = {}", clock.now_nanos());

    match poll_once(guard.as_mut()) {
        Poll::Pending => {
            println!("  poll #2 (t=1s): Pending  — inner still working, still under budget")
        }
        Poll::Ready(_) => unreachable!("inner needs 5 polls, not ready on poll #2"),
    }
    clock.advance(Duration::from_secs(2));
    println!(
        "  advance(+2s) -> now_nanos = {} (past the 2s budget)",
        clock.now_nanos()
    );

    match poll_once(guard.as_mut()) {
        Poll::Ready(Err(exceeded)) => println!(
            "  poll #3 (t=3s): Ready(Err({exceeded}))  — deadline crossed, inner cancelled with work still left"
        ),
        other => unreachable!("the clock crossed the deadline, got {other:?}"),
    }

    assert!(
        fired.is_fired(),
        "deadline must fire once the clock crosses it"
    );
    println!("  fired.is_fired() = true  — confirmed, and it stays fired (sticky, like signal)");
}

fn main() {
    println!("deadline: a timeout as a fired completion\n");

    println!("--- case 1: inner finishes before the clock passes the deadline ---");
    run_case_finishes_in_time();

    println!("--- case 2: the clock passes the deadline before inner finishes ---");
    run_case_deadline_fires();

    println!("\nboth cases: zero real sleeps — the fake clock made both outcomes deterministic.");
}
