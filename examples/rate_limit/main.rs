#![allow(clippy::unwrap_used, clippy::expect_used)]

//! `rate-limit` — the gate pattern applied to *rate*: admission gated not by
//! open/closed (see `gate`) but by whether the token bucket inside
//! `proxima_primitives::pipe::RateLimit` still has a token, where the bucket is refilled
//! by an injected [`Clock`].
//!
//! Two halves, one concept:
//!
//! 1. ADMISSION — `proxima_primitives::pipe::RateLimit` wraps an inner pipe and answers
//!    every call from a per-key token bucket: a token available admits
//!    (call passes through), an empty bucket refuses (429, `Ingest` never
//!    runs) — the same shed shape `gate`'s SHED example built from
//!    `DemandGate`, just keyed on "has a token" instead of "is armed".
//! 2. REFILL — the bucket regains capacity over time, driven by a clock.
//!    `RateLimit<Inner, Extractor, Clk>` is generic over `Clk: Clock` — the
//!    same injected-clock seam `Retry` reads via `clock.now_nanos()` before
//!    consulting its backoff schedule (see the `retry`/`backoff` examples).
//!    `RateLimit::new`/`with_caps` default `Clk` to the production
//!    `TimeClock`; `RateLimit::with_clock` takes any `Clock` impl. To prove
//!    "refilled by the clock" deterministically — no sleeps — this example
//!    drives the REAL `RateLimit` over a `FakeClock`: real time only moves
//!    when `advance` is called.
//!
//! Run: `cargo run --example rate_limit`

#[cfg(feature = "serve-prime")]
extern crate prime as _;

use core::future::Future;
use core::task::{Context, Poll, Waker};
use core::time::Duration;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use proxima_macros::pipe;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::capabilities::Clock;
use proxima_primitives::pipe::rate_limit::{CapsConfig, KeyConfig, RateLimitConfig};
use proxima_primitives::pipe::{
    KeyExtractor, ProximaError, RateLimit, RateLimitCaps, Request, Response, TokenBucketConfig,
    into_handle,
};

fn main() {
    println!("admit under the rate, refuse once the bucket is empty");
    run_admit_and_refuse();

    println!("\nthe rate is the knob: same numbers, via conflaguration");
    run_config_is_the_knob();

    println!("\nrefill: advance an injected Clock, admission resumes");
    run_clock_refill();
}

// ── shared driver ───────────────────────────────────────────────────────────

// RateLimit's call resolves on its first poll here (bucket check + an inner
// pipe that never awaits), so a one-shot poll is a legitimate block_on — see
// gate.rs for the same argument.
fn block_on_ready<Fut: Future>(future: Fut) -> Fut::Output {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut future = core::pin::pin!(future);
    match future.as_mut().poll(&mut context) {
        Poll::Ready(value) => value,
        Poll::Pending => unreachable!("rate_limit example futures resolve on first poll"),
    }
}

fn request(path: &str) -> Request<bytes::Bytes> {
    Request::builder()
        .method("GET")
        .path(path)
        .build()
        .expect("request builder")
}

// `#[pipe(send)]` generates the struct AND its `SendPipe` impl, deriving
// `Clone` unconditionally — the one bound `RateLimit::new`'s `Inner: Clone`
// needs. `name = Backend` keeps the generated struct's name matching every
// call site below, while the fn itself stays named for what it does.
#[pipe(send, name = Backend)]
async fn respond_ok(
    _request: Request<bytes::Bytes>,
) -> Result<Response<bytes::Bytes>, ProximaError> {
    Ok(Response::ok("ok"))
}

// ── 1. admission: token available admits, empty bucket refuses ─────────────

fn run_admit_and_refuse() {
    // refill_per_sec: 0 keeps this half deterministic and clock-free — the
    // point here is the boundary (capacity in, capacity+1 refused), not the
    // refill, which section 3 proves on its own.
    let stack = RateLimit::new(
        Backend,
        TokenBucketConfig {
            capacity: 2,
            refill_per_sec: 0,
        },
        KeyExtractor::ConstantKey("global".into()),
    );

    for attempt in 1..=2 {
        let response =
            block_on_ready(SendPipe::call(&stack, request("/orders"))).expect("call never errors");
        println!("attempt {attempt}: status {}", response.status);
        assert_eq!(response.status, 200, "attempt {attempt} is under capacity");
    }

    let refused =
        block_on_ready(SendPipe::call(&stack, request("/orders"))).expect("call never errors");
    println!("attempt 3: status {} (bucket empty)", refused.status);
    assert_eq!(
        refused.status, 429,
        "capacity is exhausted, no refill configured"
    );
    assert_eq!(
        refused.metadata.get_str("retry-after"),
        Some("1"),
        "a refusal carries a retry-after hint, same as a real 429"
    );
}

// ── 2. the rate is the knob: RateLimitConfig via the layered builder ───────

fn run_config_is_the_knob() {
    let config = RateLimitConfig::builder()
        .capacity(2)
        .refill_per_sec(5)
        .key(KeyConfig::PathAndMethod)
        .caps(CapsConfig::default())
        .build();

    let limiter = config
        .clone()
        .from_config(into_handle(Backend))
        .expect("valid config materializes a RateLimit");

    println!(
        "config: capacity={} refill_per_sec={} -> RateLimit materialized",
        config.capacity, config.refill_per_sec
    );

    // the burst is the same knob either way: hand-built TokenBucketConfig
    // (section 1) and this conflaguration-built RateLimitConfig lower to the
    // identical capacity/refill numbers, just from a different source.
    let admitted =
        block_on_ready(SendPipe::call(&limiter, request("/checkout"))).expect("call never errors");
    assert_eq!(admitted.status, 200, "a fresh bucket admits its first call");
}

// ── 3. refill: the REAL RateLimit, driven over an injected Clock ───────────

/// Deterministic, injectable [`Clock`]: `now_nanos` is whatever `advance` last
/// set it to. Backed by an `Arc<AtomicU64>` (not `Rc<Cell<_>>`) because
/// `RateLimit`'s `SendPipe` impl requires its clock to be `Send + Sync` — the
/// same reason `TimeClock`, its production default, is a plain `Copy` unit
/// struct reading a real atomic-backed monotonic source.
#[derive(Clone, Default)]
struct FakeClock {
    now_nanos: Arc<AtomicU64>,
}

impl FakeClock {
    fn advance(&self, duration: Duration) {
        let elapsed_ns = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
        self.now_nanos.fetch_add(elapsed_ns, Ordering::Relaxed);
    }
}

impl Clock for FakeClock {
    type Delay = core::future::Ready<()>;

    fn now_nanos(&self) -> u64 {
        self.now_nanos.load(Ordering::Relaxed)
    }

    fn delay(&self, _duration: Duration) -> Self::Delay {
        core::future::ready(())
    }
}

fn run_clock_refill() {
    let clock = FakeClock::default();
    let stack = RateLimit::with_clock(
        Backend,
        TokenBucketConfig {
            capacity: 2,
            refill_per_sec: 1,
        },
        KeyExtractor::ConstantKey("refill".into()),
        RateLimitCaps::default(),
        clock.clone(),
    );

    for attempt in 1..=2 {
        let response =
            block_on_ready(SendPipe::call(&stack, request("/orders"))).expect("call never errors");
        println!("attempt {attempt}: status {}", response.status);
        assert_eq!(
            response.status, 200,
            "attempt {attempt} is within the starting capacity of 2"
        );
    }

    let refused =
        block_on_ready(SendPipe::call(&stack, request("/orders"))).expect("call never errors");
    println!("attempt 3: status {} (bucket empty)", refused.status);
    assert_eq!(
        refused.status, 429,
        "bucket is empty and no time has passed on the clock"
    );

    println!("-- advancing the clock by 1s (no sleep) --");
    clock.advance(Duration::from_secs(1));
    let resumed =
        block_on_ready(SendPipe::call(&stack, request("/orders"))).expect("call never errors");
    println!(
        "attempt 4 (after a 1s clock-advance): status {}",
        resumed.status
    );
    assert_eq!(
        resumed.status, 200,
        "refill_per_sec=1 over 1s of clock-advance refills exactly one token"
    );

    let refused_again =
        block_on_ready(SendPipe::call(&stack, request("/orders"))).expect("call never errors");
    println!(
        "attempt 5 (same instant): status {} (that one token is already spent)",
        refused_again.status
    );
    assert_eq!(
        refused_again.status, 429,
        "the refilled token was consumed by attempt 4; no more time has passed"
    );
}
