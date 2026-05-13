//! Time as an injectable seam. `proxima_primitives::pipe::capabilities::Clock` is
//! the trait every timer-driven combinator (`Retry`, and later
//! `Backoff`/`RateLimit`/`Deadline`) is generic over — `now_nanos` for "what
//! time is it", `delay(dur)` for "wait this long", both through `&self`, not
//! a bare thread- or timer-`sleep()` call baked into the logic.
//!
//! Production gets a `Clock` backed by the real monotonic clock (`TimeClock`,
//! wrapping `proxima-time`). Tests get a `Clock` backed by a `Cell<u64>` that
//! only moves when `advance` is called. Same trait, same timer logic on top
//! — the only thing that changes is which clock you inject. That's the whole
//! reason the resilience layer never sleeps in a test: it schedules against
//! whatever `Clock` it was handed, and a test hands it one it can drive by
//! hand.
//!
//! Run: `cargo run --example clock`

use core::cell::Cell;
use core::convert::Infallible;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};
use core::time::Duration;
use std::rc::Rc;

use proxima_primitives::pipe::clock::TimeClock;
use proxima_primitives::pipe::Pipe;
use proxima_primitives::pipe::capabilities::Clock;

/// A `Clock` backed by a shared `Cell<u64>` instead of the wall. `now_nanos`
/// reads the cell; nothing else can move it forward — `advance` is the only
/// seam a test uses to make time pass, and it makes it pass instantly.
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

/// The future `FakeClock::delay` hands back. `poll` compares the shared cell
/// to `deadline`: `Ready` once `advance` has pushed it past due, `Pending`
/// otherwise. No timer thread, no OS wait — readiness is driven entirely by
/// whoever calls `advance` and polls again.
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

/// transform: schedules a wait against whichever `Clock` it was built with,
/// then reports that it fired. The pipe never calls `sleep` itself — it
/// awaits `Clock::delay`, so the clock decides what "waiting" means.
struct Timer<ClockImpl: Clock> {
    clock: ClockImpl,
}

impl<ClockImpl: Clock> Pipe for Timer<ClockImpl> {
    type In = Duration;
    type Out = &'static str;
    type Err = Infallible;

    fn call(&self, wait: Duration) -> impl Future<Output = Result<&'static str, Infallible>> {
        let delay = self.clock.delay(wait);
        async move {
            delay.await;
            Ok("fired")
        }
    }
}

/// Polls a future once against a no-op waker and returns whatever it reports
/// — `Pending` included. Awaiting (as `transform` does) polls to completion;
/// here we want to inspect `Pending` between polls, so we drive the state
/// machine by hand instead.
fn poll_once<Fut: Future>(future: Pin<&mut Fut>) -> Poll<Fut::Output> {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    future.poll(&mut context)
}

fn main() {
    // constructed, never called: `TimeClock` reads the actual monotonic
    // clock, and this example's whole point is that the logic below never
    // touches real time — that's what makes it deterministic and sleepless.
    let real_clock = TimeClock;
    println!("real clock (TimeClock, backs Retry/Backoff/Deadline in production): {real_clock:?}");

    let fake_clock = FakeClock::default();
    println!("\nfake clock (FakeClock, starts at 0, only moves when told):");
    println!("  now_nanos = {}", fake_clock.now_nanos());

    let timer = Timer {
        clock: fake_clock.clone(),
    };
    let call = timer.call(Duration::from_secs(30));
    let mut call = core::pin::pin!(call);

    println!("\nTimer scheduled for 30s against the fake clock. No thread sleeps, ever:");

    match poll_once(call.as_mut()) {
        Poll::Pending => println!(
            "  poll #1 (t=0s):  Pending  — 30s hasn't happened, nothing is waiting on a clock tick"
        ),
        Poll::Ready(_) => unreachable!("fired before any time was advanced"),
    }

    fake_clock.advance(Duration::from_secs(15));
    println!("  advance(+15s) -> now_nanos = {}", fake_clock.now_nanos());
    match poll_once(call.as_mut()) {
        Poll::Pending => println!("  poll #2 (t=15s): Pending  — halfway there, still not due"),
        Poll::Ready(_) => unreachable!("fired at half the interval"),
    }

    fake_clock.advance(Duration::from_secs(15));
    println!("  advance(+15s) -> now_nanos = {}", fake_clock.now_nanos());
    match poll_once(call.as_mut()) {
        Poll::Ready(Ok(outcome)) => {
            println!("  poll #3 (t=30s): Ready(\"{outcome}\")  — deadline crossed, timer fires")
        }
        Poll::Ready(Err(never)) => match never {},
        Poll::Pending => unreachable!("30s have elapsed on the fake clock"),
    }

    println!(
        "\nzero real time passed. zero sleeps. zero threads parked — the fake clock made it deterministic."
    );
}
