#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Circuit breaker: a [`gate`](gate.rs) that opens itself, from failure
//! evidence instead of an external controller.
//!
//! `gate`'s `AtomicGate` is armed and disarmed by whoever holds the
//! `AtomicGateController` — the pipe never decides its own admission state.
//! `CircuitBreaker` (`proxima_primitives::pipe::resilience::circuit_breaker`) is the
//! same shed-vs-pass decision, but the pipe drives it: every failure and
//! success feeds back into the breaker, and the breaker decides when to shed.
//!
//! `CircuitBreaker` is sans-IO — three methods, no wall-clock reads:
//! - `allow(now_nanos) -> bool` — may this call proceed right now?
//! - `on_success()` — record a success.
//! - `on_failure(now_nanos)` — record a failure.
//!
//! Three states (`CircuitState`):
//! 1. CLOSED    — calls pass through; consecutive failures are counted.
//! 2. OPEN      — tripped after `failure_threshold` failures; every call is
//!    refused before it reaches the dependency until `cooldown` elapses.
//! 3. HALF-OPEN — cooldown elapsed; a bounded number of probe calls are let
//!    through to test recovery. Enough successes closes the circuit again;
//!    any failure re-opens it immediately.
//!
//! `Breaker<Inner>` below is the missing wire: a `Pipe` wrapper that calls
//! `allow`/`on_success`/`on_failure` around an inner pipe. Because the
//! cooldown is `now_nanos`-driven rather than a real timer, this example
//! drives it with a manually-advanced clock — the same idiom as `clock` and
//! `backoff` — so the whole run is deterministic and sleeps zero times.
//!
//! Run: `cargo run --example circuit_breaker`

use core::cell::{Cell, RefCell};
use core::future::Future;
use core::task::{Context, Poll, Waker};
use core::time::Duration;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use proxima_primitives::pipe::{CircuitBreaker, CircuitState, Pipe};

fn main() {
    println!(
        "circuit breaker: closed -> open (short-circuit) -> half-open (probe) -> closed"
    );
    run_circuit_breaker();
}

// ── shared driver ───────────────────────────────────────────────────────────

// every future in this example resolves on its first poll (the dependency
// and the breaker are both synchronous under the hood), so a one-shot poll
// is a legitimate `block_on` — no executor dependency needed to prove it.
fn block_on_ready<F: Future>(future: F) -> F::Output {
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    let mut future = core::pin::pin!(future);
    match future.as_mut().poll(&mut cx) {
        Poll::Ready(value) => value,
        Poll::Pending => unreachable!("circuit breaker example futures resolve on first poll"),
    }
}

/// A time source backed by a shared `Cell<u64>` instead of the wall.
/// `CircuitBreaker` takes `now_nanos` as a plain argument (it never reads a
/// clock itself), so `advance` is the only seam that moves time — and it
/// moves it instantly, with no real sleep anywhere in this example.
#[derive(Clone, Default)]
struct ManualClock {
    now_nanos: Rc<Cell<u64>>,
}

impl ManualClock {
    fn now_nanos(&self) -> u64 {
        self.now_nanos.get()
    }

    fn advance(&self, elapsed: Duration) {
        let elapsed_nanos = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        self.now_nanos
            .set(self.now_nanos.get().saturating_add(elapsed_nanos));
    }
}

// ── the dependency being protected ──────────────────────────────────────────

/// A flaky downstream call. `healthy` toggles whether it succeeds or fails;
/// `calls` counts every time it actually runs — the counter the breaker's
/// short-circuit claim is proven against.
struct FlakyDependency {
    healthy: Rc<Cell<bool>>,
    calls: Arc<AtomicUsize>,
}

impl Pipe for FlakyDependency {
    type In = u32;
    type Out = &'static str;
    type Err = &'static str;

    fn call(&self, _request: u32) -> impl Future<Output = Result<&'static str, &'static str>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let healthy = self.healthy.get();
        async move {
            if healthy {
                Ok("ok")
            } else {
                Err("dependency unavailable")
            }
        }
    }
}

// ── the breaker wrapper ─────────────────────────────────────────────────────

/// `Breaker<Inner>`'s own error: refused by the circuit, or the inner pipe's
/// real error — kept distinct so a caller can tell "shed" from "it ran and
/// failed".
#[derive(Debug, Clone, PartialEq)]
enum BreakerError<InnerErr> {
    Open,
    Inner(InnerErr),
}

/// Wraps any `Pipe` with a `CircuitBreaker`. `allow` is checked synchronously
/// at `call` time, before the inner pipe's future is ever constructed — an
/// open circuit never touches `inner`.
struct Breaker<Inner> {
    inner: Inner,
    breaker: RefCell<CircuitBreaker>,
    clock: ManualClock,
}

impl<Inner> Breaker<Inner> {
    fn new(inner: Inner, breaker: CircuitBreaker, clock: ManualClock) -> Self {
        Self {
            inner,
            breaker: RefCell::new(breaker),
            clock,
        }
    }

    fn circuit_state(&self) -> CircuitState {
        self.breaker.borrow().state()
    }
}

impl<Inner: Pipe> Pipe for Breaker<Inner> {
    type In = Inner::In;
    type Out = Inner::Out;
    type Err = BreakerError<Inner::Err>;

    fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>> {
        let now = self.clock.now_nanos();
        let allowed = self.breaker.borrow_mut().allow(now);
        async move {
            if !allowed {
                return Err(BreakerError::Open);
            }
            match self.inner.call(input).await {
                Ok(out) => {
                    self.breaker.borrow_mut().on_success();
                    Ok(out)
                }
                Err(err) => {
                    self.breaker.borrow_mut().on_failure(self.clock.now_nanos());
                    Err(BreakerError::Inner(err))
                }
            }
        }
    }
}

// ── the demo ─────────────────────────────────────────────────────────────────

fn run_circuit_breaker() {
    let clock = ManualClock::default();
    let healthy = Rc::new(Cell::new(false));
    let calls = Arc::new(AtomicUsize::new(0));
    let dependency = FlakyDependency {
        healthy: Rc::clone(&healthy),
        calls: Arc::clone(&calls),
    };

    // 3 failures trip it, 1s cooldown, 2 successful probes required to close.
    let breaker = Breaker::new(
        dependency,
        CircuitBreaker::new(3, Duration::from_secs(1), 2),
        clock.clone(),
    );
    assert_eq!(
        breaker.circuit_state(),
        CircuitState::Closed,
        "starts closed"
    );

    println!("-- closed: dependency failing, calls pass through until the threshold --");
    for attempt in 1..=2_u32 {
        let outcome = block_on_ready(Pipe::call(&breaker, attempt));
        println!(
            "  call {attempt}: {outcome:?} (state={:?})",
            breaker.circuit_state()
        );
        assert_eq!(outcome, Err(BreakerError::Inner("dependency unavailable")));
        assert_eq!(
            breaker.circuit_state(),
            CircuitState::Closed,
            "under threshold, still closed"
        );
    }

    let outcome = block_on_ready(Pipe::call(&breaker, 3));
    println!(
        "  call 3: {outcome:?} (state={:?})",
        breaker.circuit_state()
    );
    assert_eq!(outcome, Err(BreakerError::Inner("dependency unavailable")));
    assert_eq!(
        breaker.circuit_state(),
        CircuitState::Open,
        "third failure trips the breaker"
    );
    assert_eq!(
        calls.load(Ordering::Relaxed),
        3,
        "all three attempts reached the dependency"
    );

    println!("-- open: cooldown not elapsed, calls are refused before the dependency --");
    for attempt in 4..=5_u32 {
        let outcome = block_on_ready(Pipe::call(&breaker, attempt));
        println!(
            "  call {attempt}: {outcome:?} (state={:?})",
            breaker.circuit_state()
        );
        assert_eq!(
            outcome,
            Err(BreakerError::Open),
            "circuit open -> refused, not the dependency's own error"
        );
    }
    assert_eq!(
        calls.load(Ordering::Relaxed),
        3,
        "open circuit short-circuits: call count unchanged since the trip"
    );

    println!("-- cooldown elapses: next call probes in half-open --");
    clock.advance(Duration::from_secs(1));
    healthy.set(true);

    let outcome = block_on_ready(Pipe::call(&breaker, 6));
    println!(
        "  call 6 (probe 1): {outcome:?} (state={:?})",
        breaker.circuit_state()
    );
    assert_eq!(outcome, Ok("ok"));
    assert_eq!(
        breaker.circuit_state(),
        CircuitState::HalfOpen,
        "one success of two required probes: still half-open"
    );

    let outcome = block_on_ready(Pipe::call(&breaker, 7));
    println!(
        "  call 7 (probe 2): {outcome:?} (state={:?})",
        breaker.circuit_state()
    );
    assert_eq!(outcome, Ok("ok"));
    assert_eq!(
        breaker.circuit_state(),
        CircuitState::Closed,
        "second probe succeeds -> closed"
    );
    assert_eq!(
        calls.load(Ordering::Relaxed),
        5,
        "3 failures + 2 probes reached the dependency"
    );

    println!("-- closed again: dependency reached normally --");
    let outcome = block_on_ready(Pipe::call(&breaker, 8));
    println!(
        "  call 8: {outcome:?} (state={:?})",
        breaker.circuit_state()
    );
    assert_eq!(outcome, Ok("ok"));
    assert_eq!(calls.load(Ordering::Relaxed), 6, "normal operation resumed");

    println!(
        "\nclosed -> open -> half-open -> closed, proved by state and by a call count the open circuit never moved."
    );
}
