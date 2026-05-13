#![cfg(feature = "loom")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Loom model-check of [`proxima_core::signal::Signal`]'s fire-vs-register
//! race — the "register-then-check closes the lost-wakeup race" invariant
//! `signal.rs`'s `Fired::poll` doc-comment already claims, proven under
//! every thread interleaving rather than merely asserted by a single-shot
//! test. Exercises exactly the storage this change touched: the per-level
//! `Mutex<SmallVec<[Arc<WakerSlot>; N]>>` push/drain race between one
//! thread's `Signal::fire()` and another thread's `Fired::poll` register.
//!
//! `futures::task::AtomicWaker` (inside `WakerSlot`) is NOT re-modeled here
//! — it is a trusted, independently-audited primitive (see `signal.rs`'s
//! module-level rationale); this test targets OUR protocol layered on top
//! of it, matching `ring/mpsc.rs`'s "only the racy primitives we wrote"
//! loom scope.
//!
//! The waiter side keeps its `Fired` future ALIVE (registered) while it
//! waits to be woken, matching how a real executor holds a pending future
//! on its task stack rather than dropping it between polls — dropping
//! `Fired` immediately after one `Pending` poll would unregister the slot
//! before `fire()` gets a chance to see it, which is a test bug, not a
//! `Signal` bug (caught while developing this test: an early draft did
//! exactly that and "found" a false lost-wakeup because it had already
//! unregistered by the time the firer thread ran).
//!
//! run:   cargo test -p proxima-core --features loom --test loom_signal --release
//! (~56s for both tests on the CI host loadout measured at row-seal time).
//! `LOOM_MAX_PREEMPTIONS` is deliberately NOT set here: the waiter's
//! spin-wait-for-wake loop (a test-harness necessity — a real executor
//! parks on the waker instead of spinning) trips loom's "exceeded maximum
//! branches" guard under a tight preemption bound on the two-waiter model;
//! that guard is loom's own documented spin-lock caveat, not a correctness
//! signal, so the unbounded run is the correct invocation for this file.

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};
use std::sync::Arc as StdArc;
use std::task::{Wake, Waker};

use loom::sync::atomic::{AtomicBool, Ordering};
use loom::thread;

use proxima_core::signal::Signal;

/// Sets a loom-tracked flag when woken, so the waiter thread can block on
/// it (spin + yield) instead of busy-polling `Signal` directly — the latter
/// would trivially "pass" without ever exercising the wake callback.
struct FlagWaker(AtomicBool);

impl Wake for FlagWaker {
    fn wake(self: StdArc<Self>) {
        self.0.store(true, Ordering::Release);
    }

    fn wake_by_ref(self: &StdArc<Self>) {
        self.0.store(true, Ordering::Release);
    }
}

/// Polls `fired` once (registering); if not immediately ready, blocks on
/// `flag` (set only by this waiter's own `wake()`) before re-polling to
/// confirm — the registration stays alive across the wait, matching a
/// real executor holding a pending future on its task stack.
fn poll_then_wait_if_pending(
    fired: &mut (impl Future<Output = ()> + Unpin),
    cx: &mut Context<'_>,
    flag: &AtomicBool,
) {
    if Pin::new(&mut *fired).poll(cx) == Poll::Ready(()) {
        return;
    }
    while !flag.load(Ordering::Acquire) {
        thread::yield_now();
    }
    let second_poll = Pin::new(&mut *fired).poll(cx);
    assert_eq!(
        second_poll,
        Poll::Ready(()),
        "woken waiter must observe the fire on re-poll"
    );
}

// one Level (Signal::new() depth 1), one waiter, one fire — the minimal
// model that still exercises the full push-then-drain race; loom's state
// space is combinatorial in threads x ops (see `loom_ring.rs`'s own
// precedent for keeping models deliberately tiny).
#[test]
fn fire_and_register_never_lose_the_wakeup() {
    loom::model(|| {
        let signal = Signal::new();
        let waiter_signal = signal.clone();

        let waker_flag = StdArc::new(FlagWaker(AtomicBool::new(false)));
        let poll_waker_flag = StdArc::clone(&waker_flag);

        let waiter = thread::spawn(move || {
            let waker = Waker::from(poll_waker_flag);
            let mut fired = waiter_signal.fired();
            let mut cx = Context::from_waker(&waker);
            poll_then_wait_if_pending(&mut fired, &mut cx, &waker_flag.0);
        });

        let firer = thread::spawn(move || {
            signal.fire();
        });

        waiter.join().expect("waiter thread completes");
        firer.join().expect("firer thread completes");
    });
}

/// Two concurrent waiters against the same level, one firer — the inline
/// `SmallVec` registry must hold both slots (well under the default
/// `sized::SIGNAL_WAITERS_INLINE_CAP`) and drain both on fire.
#[test]
fn fire_wakes_every_concurrent_waiter() {
    loom::model(|| {
        let signal = Signal::new();
        let waiter_a_signal = signal.clone();
        let waiter_b_signal = signal.clone();

        let flag_a = StdArc::new(FlagWaker(AtomicBool::new(false)));
        let flag_b = StdArc::new(FlagWaker(AtomicBool::new(false)));
        let poll_flag_a = StdArc::clone(&flag_a);
        let poll_flag_b = StdArc::clone(&flag_b);

        let waiter_a = thread::spawn(move || {
            let waker = Waker::from(poll_flag_a);
            let mut fired = waiter_a_signal.fired();
            let mut cx = Context::from_waker(&waker);
            poll_then_wait_if_pending(&mut fired, &mut cx, &flag_a.0);
        });

        let waiter_b = thread::spawn(move || {
            let waker = Waker::from(poll_flag_b);
            let mut fired = waiter_b_signal.fired();
            let mut cx = Context::from_waker(&waker);
            poll_then_wait_if_pending(&mut fired, &mut cx, &flag_b.0);
        });

        let firer = thread::spawn(move || {
            signal.fire();
        });

        waiter_a.join().expect("waiter a completes");
        waiter_b.join().expect("waiter b completes");
        firer.join().expect("firer thread completes");
    });
}
