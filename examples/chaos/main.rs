#![allow(clippy::unwrap_used, clippy::expect_used)]

//! `chaos` — fault injection as a pipe.
//!
//! Chaos testing in proxima is not a framework bolted on from outside; it is
//! a [`Pipe`] you compose IN FRONT of the system under test. `Chaos<Inner>`
//! below is a decorator: on every call it rolls a SEEDED, deterministic
//! xorshift PRNG against a `ChaosPolicy` and injects exactly one of three
//! fault kinds — an `Error` (the call never reaches `inner`), a `Dropped`
//! response (blackholed the same way — `inner` is never called, the caller
//! just sees a distinct terminal error), or a `Delay` (a fake clock is
//! advanced, `inner` still runs and still succeeds). No real sleeps and no
//! real randomness anywhere: the same seed reproduces the exact same fault
//! sequence every run, which is what makes the assertions below provable
//! instead of eyeballed.
//!
//! `retry` and `fallback` already taught the two shapes that absorb faults:
//! `RetryController` re-runs the SAME pipe on a retryable outcome,
//! `Fallback` routes to a DIFFERENT pipe on any failure. This example stacks
//! each in front of a `Chaos`-wrapped, otherwise-healthy upstream and drives
//! a fixed batch of requests through:
//!
//! 1. `Chaos(error + drop)` in front of `upstream_service`, `retry(4)` in
//!    front of that — a 50% direct-fault rate per attempt, absorbed by
//!    re-attempting, so every request in the batch still resolves `Ok`.
//! 2. `Chaos(error + drop + delay)` in front of `upstream_service` as
//!    `Fallback`'s primary, a reliable `Cache` as its secondary — an 80%
//!    direct-fault rate, absorbed by routing to the secondary instead of
//!    retrying, so every request resolves `Ok` regardless of how hostile the
//!    policy is (`Fallback`'s guarantee does not depend on tuning luck).
//!
//! Run: `cargo run --example chaos`

use core::cell::{Cell, RefCell};
use core::convert::Infallible;
use core::future::Future;
use core::task::{Context, Poll, Waker};
use core::time::Duration;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use proxima_primitives::pipe::{
    Backoff, Fallback, Jitter, Pipe, RetryAction, RetryController, RetryRules, Retryable,
};

fn main() {
    println!("chaos: fault injection as a pipe, absorbed by retry + fallback\n");
    run_retry_absorbs_faults();
    println!();
    run_fallback_absorbs_faults();
}

// ── the pure fault-injection core ───────────────────────────────────────────

/// xorshift64* — a small, seeded, deterministic PRNG. Never real entropy:
/// the same seed always produces the same stream, which is what lets this
/// example's assertions hold on every run instead of only on lucky ones.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut state = self.state;
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        self.state = state;
        state
    }
}

/// The fault a single call drew. `Clean` means the roll landed outside every
/// configured bucket — `inner` runs exactly as if `Chaos` were not there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FaultKind {
    Clean,
    Error,
    Dropped,
    Delay,
}

/// The fault policy as pure data — three percentage buckets out of 100 plus
/// the latency a `Delay` fault adds. Buckets are checked in order
/// (error, drop, delay); whatever roll is left over is `Clean`.
#[derive(Debug, Clone, Copy)]
struct ChaosPolicy {
    error_percent: u64,
    drop_percent: u64,
    delay_percent: u64,
    delay: Duration,
}

impl ChaosPolicy {
    fn classify(&self, roll: u64) -> FaultKind {
        let roll = roll % 100;
        let error_edge = self.error_percent;
        let drop_edge = error_edge + self.drop_percent;
        let delay_edge = drop_edge + self.delay_percent;
        if roll < error_edge {
            FaultKind::Error
        } else if roll < drop_edge {
            FaultKind::Dropped
        } else if roll < delay_edge {
            FaultKind::Delay
        } else {
            FaultKind::Clean
        }
    }
}

/// Per-kind fault counters, shared out to the caller so the example can
/// report and assert on exactly what was injected — not just what was
/// eyeballed in the per-call print lines.
#[derive(Default)]
struct ChaosStats {
    errors: AtomicU32,
    drops: AtomicU32,
    delays: AtomicU32,
    clean: AtomicU32,
}

impl ChaosStats {
    fn record(&self, fault: FaultKind) {
        let counter = match fault {
            FaultKind::Error => &self.errors,
            FaultKind::Dropped => &self.drops,
            FaultKind::Delay => &self.delays,
            FaultKind::Clean => &self.clean,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self) -> FaultCounts {
        FaultCounts {
            errors: self.errors.load(Ordering::Relaxed),
            drops: self.drops.load(Ordering::Relaxed),
            delays: self.delays.load(Ordering::Relaxed),
            clean: self.clean.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FaultCounts {
    errors: u32,
    drops: u32,
    delays: u32,
    clean: u32,
}

/// A fake clock a `Delay` fault advances. `Chaos` never reads the wall clock
/// and never sleeps — `advance` moves time instantly, the same injected-clock
/// idiom `clock`, `backoff`, and `circuit_breaker` already use.
#[derive(Default)]
struct FaultClock {
    now_nanos: Cell<u64>,
}

impl FaultClock {
    fn advance(&self, elapsed: Duration) {
        let elapsed_nanos = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        self.now_nanos
            .set(self.now_nanos.get().saturating_add(elapsed_nanos));
    }

    fn now_nanos(&self) -> u64 {
        self.now_nanos.get()
    }
}

/// `Chaos<Inner>`'s own error: an injected error, a blackholed (dropped)
/// response, or `inner`'s real error passed through unchanged. Kept distinct
/// so a caller can tell "chaos struck" from "the system under test failed on
/// its own" — the same separation `Breaker<Inner>` draws in `circuit_breaker`.
#[derive(Debug, Clone, PartialEq)]
enum ChaosFault<InnerErr> {
    Injected,
    Dropped,
    Inner(InnerErr),
}

/// The chaos decorator: wraps any `Pipe` and, per `policy`, injects a fault
/// before `inner` ever runs — or lets `inner` run, possibly after advancing
/// the fake clock. `rng` is a `RefCell` because `Pipe::call` only borrows
/// `&self`, the same interior-mutability shape `Breaker<Inner>` uses for its
/// `CircuitBreaker`.
struct Chaos<Inner> {
    inner: Inner,
    policy: ChaosPolicy,
    rng: RefCell<Xorshift64>,
    clock: FaultClock,
    stats: Arc<ChaosStats>,
}

impl<Inner> Chaos<Inner> {
    fn new(inner: Inner, policy: ChaosPolicy, seed: u64, stats: Arc<ChaosStats>) -> Self {
        Self {
            inner,
            policy,
            rng: RefCell::new(Xorshift64::new(seed)),
            clock: FaultClock::default(),
            stats,
        }
    }

    fn simulated_delay(&self) -> Duration {
        Duration::from_nanos(self.clock.now_nanos())
    }
}

impl<Inner: Pipe> Pipe for Chaos<Inner> {
    type In = Inner::In;
    type Out = Inner::Out;
    type Err = ChaosFault<Inner::Err>;

    fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>> {
        let fault = {
            let mut rng = self.rng.borrow_mut();
            self.policy.classify(rng.next_u64())
        };
        self.stats.record(fault);
        if fault == FaultKind::Delay {
            self.clock.advance(self.policy.delay);
        }
        async move {
            match fault {
                FaultKind::Error => Err(ChaosFault::Injected),
                FaultKind::Dropped => Err(ChaosFault::Dropped),
                FaultKind::Delay | FaultKind::Clean => {
                    self.inner.call(input).await.map_err(ChaosFault::Inner)
                }
            }
        }
    }
}

// ── shared driver ───────────────────────────────────────────────────────────

// every future in this example resolves on its first poll (chaos and the
// system under test are both synchronous under the hood), so a one-shot poll
// is a legitimate block_on — the same idiom as retry/fallback/circuit_breaker.
fn block_on_ready<F: Future>(future: F) -> F::Output {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut future = core::pin::pin!(future);
    match future.as_mut().poll(&mut context) {
        Poll::Ready(value) => value,
        Poll::Pending => unreachable!("chaos example futures resolve on first poll"),
    }
}

// ── the pipe under test ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct Request {
    id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    Upstream,
    Cache,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Response {
    id: u32,
    source: Source,
}

// the only retry surface Response needs: it never carries a retryable
// status of its own, because every failure this example retries comes from
// Chaos, not from the upstream itself.
impl Retryable for Response {
    fn retry_status(&self) -> Option<u16> {
        None
    }

    fn is_success(&self) -> bool {
        true
    }
}

/// The healthy system under test. It never fails on its own — every failure
/// downstream examples observe comes from the `Chaos` wrapper in front of it,
/// which is the whole point: chaos is injected at the seam, not baked into
/// the service. Stateless, so `#[proxima::piped]` writes the `Pipe` impl.
#[proxima::piped]
async fn upstream_service(request: Request) -> Result<Response, Infallible> {
    Ok(Response {
        id: request.id,
        source: Source::Upstream,
    })
}

// ── scenario 1: retry absorbs error + drop ──────────────────────────────────

fn retry_call(
    pipe: &Chaos<upstream_service>,
    request: Request,
    controller: &RetryController,
) -> (Result<Response, ChaosFault<Infallible>>, u32) {
    let mut attempt = 0;
    let mut prev_delay = Duration::ZERO;
    let mut outcome = block_on_ready(pipe.call(request));
    loop {
        match controller.on_outcome(attempt, &outcome, 0, 0, prev_delay) {
            RetryAction::Done | RetryAction::Exhausted => return (outcome, attempt + 1),
            RetryAction::Retry { after } => {
                prev_delay = after;
                attempt += 1;
                outcome = block_on_ready(pipe.call(request));
            }
        }
    }
}

fn run_retry_absorbs_faults() {
    println!("-- chaos(50% fault) + retry(4): every request still resolves --");

    const REQUEST_COUNT: u32 = 16;
    let policy = ChaosPolicy {
        error_percent: 35,
        drop_percent: 15,
        delay_percent: 10,
        delay: Duration::from_millis(75),
    };
    let stats = Arc::new(ChaosStats::default());
    let chaos = Chaos::new(
        upstream_service,
        policy,
        0xA5A5_1234_9E37_79B9,
        Arc::clone(&stats),
    );
    let controller = RetryController {
        rules: RetryRules::default(),
        backoff: Backoff::Exponential {
            initial: Duration::from_millis(20),
            factor: 2,
            max: Duration::from_millis(500),
        },
        jitter: Jitter::None,
        max_attempts: 4,
        deadline: None,
    };

    let mut successes = 0;
    let mut total_attempts = 0;
    for id in 0..REQUEST_COUNT {
        let (outcome, attempts) = retry_call(&chaos, Request { id }, &controller);
        total_attempts += attempts;
        match outcome {
            Ok(response) => {
                successes += 1;
                println!("  request {id}: resolved Ok({response:?}) after {attempts} attempt(s)");
            }
            Err(fault) => {
                println!(
                    "  request {id}: resolved Err({fault:?}) after {attempts} attempt(s) (retry budget exhausted)"
                );
            }
        }
    }

    let counts = stats.snapshot();
    println!(
        "\n  faults injected: {} error, {} drop, {} delay, {} clean ({total_attempts} attempts over {REQUEST_COUNT} requests)",
        counts.errors, counts.drops, counts.delays, counts.clean
    );
    println!(
        "  simulated chaos-clock advance: {:?} (no real sleep)",
        chaos.simulated_delay()
    );

    assert_eq!(
        successes, REQUEST_COUNT,
        "retry(4) recovers every request despite a 50% direct-fault rate per attempt"
    );
    println!("  {successes}/{REQUEST_COUNT} requests recovered — graceful degradation via retry\n");
}

// ── scenario 2: fallback absorbs total primary failure ──────────────────────

/// The reliable secondary. It never fails and never sees a chaos policy —
/// `hits` proves how often `Fallback` actually needed it.
struct Cache {
    hits: Arc<AtomicU32>,
}

impl Pipe for Cache {
    type In = Request;
    type Out = Response;
    type Err = ChaosFault<Infallible>;

    fn call(
        &self,
        request: Request,
    ) -> impl Future<Output = Result<Response, ChaosFault<Infallible>>> {
        self.hits.fetch_add(1, Ordering::Relaxed);
        async move {
            Ok(Response {
                id: request.id,
                source: Source::Cache,
            })
        }
    }
}

fn run_fallback_absorbs_faults() {
    println!("-- chaos(80% fault) + fallback: every request still resolves --");

    const REQUEST_COUNT: u32 = 16;
    let policy = ChaosPolicy {
        error_percent: 30,
        drop_percent: 30,
        delay_percent: 20,
        delay: Duration::from_millis(120),
    };
    let stats = Arc::new(ChaosStats::default());
    let chaos = Chaos::new(
        upstream_service,
        policy,
        0xC0FF_EE00_1357_2468,
        Arc::clone(&stats),
    );
    let cache_hits = Arc::new(AtomicU32::new(0));
    let composite = Fallback {
        primary: chaos,
        secondary: Cache {
            hits: Arc::clone(&cache_hits),
        },
    };

    let mut successes = 0;
    for id in 0..REQUEST_COUNT {
        let outcome = block_on_ready(Pipe::call(&composite, Request { id }));
        let response = outcome.expect("fallback's secondary never fails, so this always resolves");
        successes += 1;
        println!(
            "  request {id}: resolved Ok({response:?}) via {:?}",
            response.source
        );
    }

    let counts = stats.snapshot();
    println!(
        "\n  faults injected: {} error, {} drop, {} delay, {} clean over {REQUEST_COUNT} requests",
        counts.errors, counts.drops, counts.delays, counts.clean
    );
    println!(
        "  simulated chaos-clock advance: {:?} (no real sleep)",
        composite.primary.simulated_delay()
    );
    println!(
        "  cache served {} of {REQUEST_COUNT} requests (primary's faults routed here)",
        cache_hits.load(Ordering::Relaxed)
    );

    assert_eq!(
        successes, REQUEST_COUNT,
        "fallback resolves every request regardless of chaos intensity — no tuning required"
    );
    println!(
        "  {successes}/{REQUEST_COUNT} requests recovered — graceful degradation via fallback"
    );
}
