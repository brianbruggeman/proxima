#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Backoff: the delay *schedule* between retry attempts, and how it is driven
//! against a `Clock` without ever calling a thread- or timer-`sleep()`.
//!
//! `Backoff` (`proxima_primitives::pipe::resilience::backoff`) is pure math: given an
//! attempt number it returns a `Duration`, nothing else. Three shapes:
//!
//! 1. CONSTANT     — the same delay every attempt.
//! 2. EXPONENTIAL  — `initial * factor^attempt`, saturating at `max`.
//! 3. JITTER       — a `Jitter` variant randomises the base delay with
//!    caller-supplied entropy (`rand: u64`), never a global RNG.
//!
//! `Retry` (`proxima_primitives::pipe::resilience::retry_exec`) is what actually
//! *schedules* those delays: after a retryable outcome it calls
//! `clock.delay(after)` and awaits it. The clock in this example is a
//! `ManualClock` that resolves every `delay` immediately (no real sleep) and
//! advances its own `now_nanos` by the requested duration, so the whole
//! example runs in microseconds while still proving a real elapsed-time
//! schedule — the same trick the crate's own tests use (see
//! `MockClock` in `retry_exec.rs`).
//!
//! Run: `cargo run --example backoff`

use core::cell::{Cell, RefCell};
use core::future::{Future, Ready, ready};
use core::time::Duration;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use proxima_primitives::block_on;
use proxima_primitives::pipe::capabilities::Clock;
use proxima_primitives::pipe::resilience::Retry;
use proxima_primitives::pipe::{Backoff, Jitter, Pipe, RetryController, RetryRules, Retryable};

fn main() {
    println!("constant: the same delay every attempt");
    run_constant();

    println!("\nexponential: delay doubles, saturates at max");
    run_exponential();

    println!("\njitter: randomised on top of the exponential base");
    run_jitter();
}

/// Deterministic, injectable `Clock`. `delay` never sleeps — it records the
/// requested duration into the schedule and advances `now_nanos` by that same
/// amount, so downstream deadline math sees elapsed virtual time with zero
/// wall-clock cost. This is the manual advance the whole example turns on.
#[derive(Clone)]
struct ManualClock {
    now_nanos: Rc<Cell<u64>>,
    schedule: Rc<RefCell<Vec<Duration>>>,
}

impl ManualClock {
    fn new() -> Self {
        Self {
            now_nanos: Rc::new(Cell::new(0)),
            schedule: Rc::new(RefCell::new(Vec::new())),
        }
    }

    fn schedule(&self) -> Vec<Duration> {
        self.schedule.borrow().clone()
    }
}

impl Clock for ManualClock {
    type Delay = Ready<()>;

    fn now_nanos(&self) -> u64 {
        self.now_nanos.get()
    }

    fn delay(&self, duration: Duration) -> Ready<()> {
        self.schedule.borrow_mut().push(duration);
        let duration_nanos = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
        self.now_nanos
            .set(self.now_nanos.get().saturating_add(duration_nanos));
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

/// Always answers 503 — every attempt is retryable, so a run drives exactly
/// `max_attempts` calls and `max_attempts - 1` scheduled delays.
struct AlwaysUnavailable {
    calls: Arc<AtomicU32>,
}

impl Pipe for AlwaysUnavailable {
    type In = u32;
    type Out = StatusOut;
    type Err = &'static str;

    fn call(&self, _input: u32) -> impl Future<Output = Result<StatusOut, &'static str>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        async move { Ok(StatusOut(Some(503))) }
    }
}

fn controller(max_attempts: u32, backoff: Backoff, jitter: Jitter) -> RetryController {
    RetryController {
        rules: RetryRules::default(),
        backoff,
        jitter,
        max_attempts,
        deadline: None,
    }
}

// ── 1. CONSTANT ──────────────────────────────────────────────────────────────

fn run_constant() {
    let clock = ManualClock::new();
    let calls = Arc::new(AtomicU32::new(0));
    let pipe = AlwaysUnavailable {
        calls: Arc::clone(&calls),
    };
    let retry = Retry::new(
        pipe,
        controller(
            4,
            Backoff::Constant(Duration::from_millis(50)),
            Jitter::None,
        ),
        clock.clone(),
        0,
    );

    let outcome = block_on(Pipe::call(&retry, 0));

    let schedule = clock.schedule();
    println!("attempts made: {}", calls.load(Ordering::Relaxed));
    println!("delay schedule: {schedule:?}");
    println!(
        "clock advanced to: {}ns (no real sleep occurred)",
        clock.now_nanos()
    );

    assert_eq!(
        outcome,
        Ok(StatusOut(Some(503))),
        "exhausted after max_attempts"
    );
    assert_eq!(
        calls.load(Ordering::Relaxed),
        4,
        "4 attempts, 3 gaps between them"
    );
    assert_eq!(
        schedule,
        std::vec![Duration::from_millis(50); 3],
        "constant backoff repeats the same delay between every attempt"
    );
    assert_eq!(
        clock.now_nanos(),
        150_000_000,
        "3 * 50ms advanced, zero wall-clock time"
    );
}

// ── 2. EXPONENTIAL ───────────────────────────────────────────────────────────

fn run_exponential() {
    let clock = ManualClock::new();
    let calls = Arc::new(AtomicU32::new(0));
    let pipe = AlwaysUnavailable {
        calls: Arc::clone(&calls),
    };
    let backoff = Backoff::Exponential {
        initial: Duration::from_millis(100),
        factor: 2,
        max: Duration::from_millis(2000),
    };
    let retry = Retry::new(pipe, controller(8, backoff, Jitter::None), clock.clone(), 0);

    let outcome = block_on(Pipe::call(&retry, 0));

    let schedule = clock.schedule();
    println!("attempts made: {}", calls.load(Ordering::Relaxed));
    println!("delay schedule: {schedule:?}");
    println!(
        "clock advanced to: {}ns (no real sleep occurred)",
        clock.now_nanos()
    );

    assert_eq!(
        outcome,
        Ok(StatusOut(Some(503))),
        "exhausted after max_attempts"
    );
    assert_eq!(
        calls.load(Ordering::Relaxed),
        8,
        "8 attempts, 7 gaps between them"
    );
    let expected = [100u64, 200, 400, 800, 1600, 2000, 2000]
        .map(Duration::from_millis)
        .to_vec();
    assert_eq!(
        schedule, expected,
        "base = 100ms, doubles each attempt (100, 200, 400, 800, 1600), saturates at max = 2000ms"
    );
}

// ── 3. JITTER ─────────────────────────────────────────────────────────────────

fn run_jitter() {
    let backoff = Backoff::Exponential {
        initial: Duration::from_millis(100),
        factor: 2,
        max: Duration::from_millis(2000),
    };

    println!("-- bit-exact: Backoff::delay with caller-supplied rand, Jitter::Full --");
    let direct_clock = ManualClock::new();
    let rands = [0u64, 150_000, 999_999_999];
    let mut prev = Duration::ZERO;
    for (attempt, rand) in rands.into_iter().enumerate() {
        let attempt = attempt as u32;
        let base = backoff.base_delay(attempt);
        let jittered = backoff.delay(attempt, Jitter::Full, prev, rand);
        block_on(direct_clock.delay(jittered));
        println!("  attempt {attempt}: base={base:?} rand={rand} -> jittered={jittered:?}");
        assert!(
            jittered <= base,
            "Full jitter is never larger than its base delay"
        );
        prev = jittered;
    }
    println!(
        "  clock advanced to {}ns scheduling {} delays, zero real sleeps",
        direct_clock.now_nanos(),
        direct_clock.schedule().len()
    );

    // Full jitter ignores `prev`, so the same (attempt, rand) reproduces the
    // same delay regardless of what came before it — no hidden RNG state.
    let replay = backoff.delay(1, Jitter::Full, Duration::ZERO, rands[1]);
    assert_eq!(
        replay,
        direct_clock.schedule()[1],
        "same (attempt, rand) always reproduces the same jittered delay"
    );

    println!("-- integrated: Retry drives Jitter::Equal end-to-end over the injected Clock --");
    let integration_clock = ManualClock::new();
    let calls = Arc::new(AtomicU32::new(0));
    let pipe = AlwaysUnavailable {
        calls: Arc::clone(&calls),
    };
    let retry = Retry::new(
        pipe,
        controller(4, backoff, Jitter::Equal),
        integration_clock.clone(),
        7,
    );

    let _ = block_on(Pipe::call(&retry, 0));

    let schedule = integration_clock.schedule();
    println!("  Jitter::Equal schedule: {schedule:?}");
    for (attempt, delay) in schedule.iter().enumerate() {
        let base = backoff.base_delay(attempt as u32);
        let half = Duration::from_millis(u64::try_from(base.as_millis()).unwrap_or(u64::MAX) / 2);
        assert!(
            *delay >= half && *delay <= base,
            "Equal jitter attempt {attempt}: {delay:?} not in [{half:?}, {base:?}]"
        );
    }
    assert_eq!(schedule.len(), 3, "4 attempts, 3 scheduled gaps");
    println!("  every delay landed in [base/2, base] — jittered, still bounded, still no sleep");
}
