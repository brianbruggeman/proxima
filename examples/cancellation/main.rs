//! `cancellation` — cooperative, deadline, drop, propagating, and
//! cancel-with-cleanup: five modes, one primitive.
//!
//! `signal` taught fire-once completion; `deadline` taught a timeout as a
//! signal fired by an injected clock. Cancellation is not a runtime
//! feature bolted on top of those — it is what you get from composing a
//! fired [`Signal`] with the pipe algebra already in hand:
//!
//! - **cooperative** — a checkpoint reads `Signal::is_fired()` between
//!   units of work and stops when it sees it, releasing what it holds
//!   before returning.
//! - **deadline** — the same fired `Signal`, except the thing that decides
//!   to fire it is `Deadline::expired(clock.now_nanos())` instead of a
//!   caller.
//! - **drop-to-cancel** — `Signal::guard()` ties a scope's level to a
//!   value's lifetime; dropping that value fires the scope, so abandoning
//!   the driver *is* the cancel.
//! - **propagating** — `Signal::child()` merges ancestor levels into a
//!   descendant's own; firing a parent is visible to every descendant
//!   without visiting them.
//! - **cancel-with-cleanup** — a `Drop` finalizer held by the cancelled
//!   future. Not `Signal`-specific: the guarantee ("exactly once") comes
//!   from the language's drop semantics, and `Signal` only decides *when*
//!   the value carrying that finalizer gets dropped.
//!
//! No sleeps anywhere: cooperative and cancel-with-cleanup model "slow"
//! work as a poll count, and deadline drives a `Cell`-backed fake clock by
//! hand, exactly like `clock` and `deadline` did.
//!
//! Run: `cargo run --example cancellation`

use core::cell::Cell;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};
use core::time::Duration;
use std::rc::Rc;

use proxima_core::signal::Signal;
use proxima_primitives::pipe::resilience::Deadline;

/// Polls a future once against a no-op waker and returns whatever it
/// reports, `Pending` included — the same hand-driving `deadline` and
/// `clock` use so every step stays observable and nothing ever sleeps.
fn poll_once<Fut: Future>(future: Pin<&mut Fut>) -> Poll<Fut::Output> {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    future.poll(&mut context)
}

// ── cooperative: checked at each checkpoint, never preempted ───────────────

/// "Slow" work modeled as a poll count. Each checkpoint first asks whether
/// the signal fired; only if it hasn't does it do a unit of work.
struct CooperativeWork {
    signal: Signal,
    steps_remaining: u32,
    completed: Rc<Cell<u32>>,
    resource_open: Rc<Cell<bool>>,
}

impl Future for CooperativeWork {
    type Output = &'static str;

    fn poll(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<&'static str> {
        let this = self.get_mut();
        if this.signal.is_fired() {
            this.resource_open.set(false);
            return Poll::Ready("cancelled at checkpoint");
        }
        if this.steps_remaining == 0 {
            this.resource_open.set(false);
            return Poll::Ready("finished");
        }
        this.steps_remaining -= 1;
        this.completed.set(this.completed.get() + 1);
        Poll::Pending
    }
}

fn run_cooperative() {
    println!("--- cooperative: Signal::is_fired() checked at each checkpoint ---");
    let signal = Signal::new();
    let completed = Rc::new(Cell::new(0_u32));
    let resource_open = Rc::new(Cell::new(true));
    let work = CooperativeWork {
        signal: signal.clone(),
        steps_remaining: 5,
        completed: Rc::clone(&completed),
        resource_open: Rc::clone(&resource_open),
    };
    let mut work = core::pin::pin!(work);

    for step in 1..=2 {
        match poll_once(work.as_mut()) {
            Poll::Pending => println!(
                "  checkpoint {step}: pending, {} step(s) done",
                completed.get()
            ),
            Poll::Ready(outcome) => unreachable!("5 steps queued, got {outcome} early"),
        }
    }

    println!(
        "  signal.fire() — cancel after {} of 5 steps",
        completed.get()
    );
    signal.fire();

    match poll_once(work.as_mut()) {
        Poll::Ready(outcome) => println!("  checkpoint 3: {outcome}"),
        Poll::Pending => unreachable!("the next checkpoint must observe the fired signal"),
    }

    assert_eq!(
        completed.get(),
        2,
        "work stopped at the checkpoint, not after all 5 steps"
    );
    assert!(
        !resource_open.get(),
        "the checkpoint released its resource before returning"
    );
    println!(
        "  completed = {}, resource_open = {}\n",
        completed.get(),
        resource_open.get()
    );
}

// ── deadline: the same fired signal, driven by a clock instead of a caller ──

/// A `Cell`-backed clock, moved only by `advance` — the same shape `clock`
/// and `deadline` use, kept minimal here since `Deadline` itself needs
/// nothing but a `u64` to compare against.
#[derive(Clone, Default)]
struct FakeClock {
    now_nanos: Rc<Cell<u64>>,
}

impl FakeClock {
    fn now_nanos(&self) -> u64 {
        self.now_nanos.get()
    }

    fn advance(&self, elapsed: Duration) {
        let elapsed_nanos = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        self.now_nanos
            .set(self.now_nanos.get().saturating_add(elapsed_nanos));
    }
}

fn run_deadline() {
    println!("--- deadline: Deadline::expired(clock.now_nanos()) fires the signal ---");
    let clock = FakeClock::default();
    let deadline = Deadline::new(clock.now_nanos(), Duration::from_secs(5));
    let signal = Signal::new();

    clock.advance(Duration::from_secs(3));
    println!(
        "  advance(+3s) -> now_nanos = {} — under the 5s budget",
        clock.now_nanos()
    );
    if deadline.expired(clock.now_nanos()) {
        signal.fire();
    }
    assert!(
        !signal.is_fired(),
        "3s elapsed against a 5s budget: not expired yet"
    );

    clock.advance(Duration::from_secs(4));
    println!(
        "  advance(+4s) -> now_nanos = {} — past the 5s budget",
        clock.now_nanos()
    );
    if deadline.expired(clock.now_nanos()) {
        signal.fire();
    }
    assert!(
        signal.is_fired(),
        "7s elapsed against a 5s budget: the deadline must have fired"
    );
    println!("  signal.is_fired() = true — the clock cancelled, not a caller\n");
}

// ── drop-to-cancel: abandoning the driver handle fires the signal ──────────

/// Work bound to a driver's scope. It only commits its side effect on the
/// step after its last poll — cancelling one poll early means the commit
/// never happens.
struct DriverBoundWork {
    signal: Signal,
    steps_remaining: u32,
    committed: Rc<Cell<bool>>,
}

impl Future for DriverBoundWork {
    type Output = &'static str;

    fn poll(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<&'static str> {
        let this = self.get_mut();
        if this.signal.is_fired() {
            return Poll::Ready("cancelled: driver dropped");
        }
        if this.steps_remaining == 0 {
            this.committed.set(true);
            return Poll::Ready("committed");
        }
        this.steps_remaining -= 1;
        Poll::Pending
    }
}

fn run_drop_to_cancel() {
    println!("--- drop-to-cancel: Signal::guard() fires its scope when dropped ---");
    let scope = Signal::new();
    let observer = scope.clone();
    let committed = Rc::new(Cell::new(false));
    let work = DriverBoundWork {
        signal: observer.clone(),
        steps_remaining: 3,
        committed: Rc::clone(&committed),
    };
    let mut work = core::pin::pin!(work);
    let driver = scope.guard();

    for step in 1..=2 {
        match poll_once(work.as_mut()) {
            Poll::Pending => println!("  step {step}: pending, driver still alive"),
            Poll::Ready(outcome) => unreachable!("3 steps queued, got {outcome} early"),
        }
    }

    println!("  drop(driver) — the caller abandons the operation before it commits");
    drop(driver);
    assert!(
        observer.is_fired(),
        "dropping the guard fires the scope it owns"
    );

    match poll_once(work.as_mut()) {
        Poll::Ready(outcome) => println!("  next poll: {outcome}"),
        Poll::Pending => unreachable!("the fired signal must resolve on the next poll"),
    }
    assert!(
        !committed.get(),
        "the side effect never ran — cancelled one step before its commit"
    );
    println!("  committed = {}\n", committed.get());
}

// ── propagating: a fired parent is visible to every descendant ─────────────

fn run_propagating() {
    println!("--- propagating: Signal::child() merges ancestor levels into a descendant ---");
    let parent = Signal::new();
    let child_a = parent.child();
    let child_b = parent.child();
    let grandchild = child_a.child();
    let unrelated = Signal::new();

    assert!(!parent.is_fired());
    assert!(!child_a.is_fired());
    assert!(!child_b.is_fired());
    assert!(!grandchild.is_fired());

    parent.fire();
    println!("  parent.fire()");

    assert!(child_a.is_fired(), "a child observes the parent's fire");
    assert!(
        child_b.is_fired(),
        "every child observes the same fire, not just the first"
    );
    assert!(
        grandchild.is_fired(),
        "a grandchild observes it two levels up, through the merge"
    );
    assert!(
        !unrelated.is_fired(),
        "an unrelated scope, sharing no ancestor, is untouched"
    );
    println!(
        "  child_a={} child_b={} grandchild={} unrelated={}",
        child_a.is_fired(),
        child_b.is_fired(),
        grandchild.is_fired(),
        unrelated.is_fired()
    );

    let root = Signal::new();
    let branch = root.child();
    branch.fire();
    assert!(branch.is_fired());
    assert!(
        !root.is_fired(),
        "cancellation is directional: firing a child never touches its parent"
    );
    println!(
        "  branch.fire() leaves root untouched — propagation flows root -> leaves, never back up\n"
    );
}

// ── cancel-with-cleanup: a finalizer the language guarantees runs once ─────

/// Runs `cleanup` when dropped — the RAII half of the composition. The
/// guarantee that it runs *exactly* once is not something this example
/// builds; it is what `Drop` already promises for every value.
struct CleanupGuard<CleanupFn: FnMut()> {
    cleanup: CleanupFn,
}

impl<CleanupFn: FnMut()> Drop for CleanupGuard<CleanupFn> {
    fn drop(&mut self) {
        (self.cleanup)();
    }
}

/// Work that carries its own finalizer. Observing `Ready("cancelled")` is
/// not the same event as the finalizer running — the finalizer only runs
/// when this whole value is dropped.
struct GuardedWork<CleanupFn: FnMut()> {
    signal: Signal,
    steps_remaining: u32,
    _finalizer: CleanupGuard<CleanupFn>,
}

impl<CleanupFn: FnMut() + Unpin> Future for GuardedWork<CleanupFn> {
    type Output = &'static str;

    fn poll(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<&'static str> {
        let this = self.get_mut();
        if this.signal.is_fired() {
            return Poll::Ready("cancelled");
        }
        if this.steps_remaining == 0 {
            return Poll::Ready("finished");
        }
        this.steps_remaining -= 1;
        Poll::Pending
    }
}

fn run_cancel_with_cleanup() {
    println!("--- cancel-with-cleanup: a Drop finalizer, guaranteed exactly once ---");
    let cleanup_runs = Rc::new(Cell::new(0_u32));
    let signal = Signal::new();

    {
        let counter = Rc::clone(&cleanup_runs);
        let work = GuardedWork {
            signal: signal.clone(),
            steps_remaining: 5,
            _finalizer: CleanupGuard {
                cleanup: move || counter.set(counter.get() + 1),
            },
        };
        let mut work = core::pin::pin!(work);

        for step in 1..=2 {
            match poll_once(work.as_mut()) {
                Poll::Pending => println!("  step {step}: pending, finalizer not run yet"),
                Poll::Ready(outcome) => unreachable!("5 steps queued, got {outcome} early"),
            }
        }

        signal.fire();
        match poll_once(work.as_mut()) {
            Poll::Ready(outcome) => {
                println!("  {outcome}, but the finalizer runs on drop, not on Ready")
            }
            Poll::Pending => unreachable!("the fired signal must resolve on the next poll"),
        }
        assert_eq!(
            cleanup_runs.get(),
            0,
            "observing Ready(\"cancelled\") is not the same as being dropped"
        );
    }

    assert_eq!(
        cleanup_runs.get(),
        1,
        "the guard dropped exactly once when the cancelled work went out of scope"
    );
    println!(
        "  cleanup_runs = {} — exactly once, on the cancel path\n",
        cleanup_runs.get()
    );
}

fn main() {
    println!("cancellation: a fired Signal, composed five ways\n");

    run_cooperative();
    run_deadline();
    run_drop_to_cancel();
    run_propagating();
    run_cancel_with_cleanup();

    println!("all five modes: one primitive (Signal), five compositions, zero sleeps.");
}
