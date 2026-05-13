# rate-limit — admit under a rate

## Builds on

[gate](../gate/README.md), [clock](../clock/README.md) — rate-limit = gate + a clock-refilled token bucket.

## What it demonstrates

`gate` gates admission on open/closed (`is_armed`). `rate-limit` is the same shape aimed at a different question: not "is anyone allowed through right now" but "how many are allowed through, over time". The gate condition becomes "does the token bucket still have a token", and the bucket's state moves on its own — refilled by a `Clock`, the same seam `Retry`/`Backoff` schedule against.

Two halves:

1. **admission** — `proxima_primitives::pipe::RateLimit` wraps an inner pipe with a per-key token bucket (real production type). A token available admits: the call passes through. An empty bucket refuses: a 429, the inner pipe never runs. Shown two ways: hand-built (`RateLimit::new` + `TokenBucketConfig`) and config-built (`RateLimitConfig::builder()...build().from_config(...)`, the `conflaguration` house pattern) — the rate is a knob, not a hardcoded number. The wrapped pipe, `Backend`, is `#[pipe(send, name = Backend)] async fn respond_ok(..)` — `RateLimit<Inner, ..>` requires `Inner: Clone` (it clones `self.inner` into the future on every call), and `#[proxima::pipe]` derives `Clone` on the generated struct unconditionally, so there is no hand-written `#[derive(Clone)]` here. See `00-foundations.md` section 7 for the general rule and a before/after against the pre-macro shape of this exact pipe.
2. **refill** — `RateLimit<Inner, Extractor, Clk>` is generic over `Clk: Clock`, the same injected-clock seam `Retry`/`Backoff` schedule against (`clock.now_nanos()`, never a bare real-time read). `RateLimit::new`/`with_caps` default `Clk` to the production `TimeClock`, so every existing caller is unaffected; `RateLimit::with_clock` takes any `Clock` impl. To prove "refilled by the clock" deterministically, with no sleeps, this example drives the REAL `RateLimit` (not a lookalike) over a `FakeClock` that only moves when `advance` is called — the same idiom the `clock` example teaches.

## Run

```
cargo run --example rate_limit
```

## What you'll see

```
admit under the rate, refuse once the bucket is empty
attempt 1: status 200
attempt 2: status 200
attempt 3: status 429 (bucket empty)

the rate is the knob: same numbers, via conflaguration
config: capacity=2 refill_per_sec=5 -> RateLimit materialized

refill: advance an injected Clock, admission resumes
attempt 1: status 200
attempt 2: status 200
attempt 3: status 429 (bucket empty)
-- advancing the clock by 1s (no sleep) --
attempt 4 (after a 1s clock-advance): status 200
attempt 5 (same instant): status 429 (that one token is already spent)
```

- **admit under rate**: with `capacity: 2`, the first two calls land inside the bucket's starting tokens and get a real `200` back from the inner pipe.
- **refuse over rate**: the third call finds the bucket empty (`refill_per_sec: 0` for this half — deterministic, no clock dependency) and gets `429` with a `retry-after` header, without the inner pipe ever running.
- **the rate is the knob**: `RateLimitConfig::builder().capacity(2).refill_per_sec(5)...build()` lowers to the same `capacity`/`refill_per_sec` numbers as the hand-built `TokenBucketConfig` — same limiter, config-driven.
- **refill on clock-advance**: section 3 builds the REAL `RateLimit` via `RateLimit::with_clock` at capacity 2, drains it, gets refused once empty, then advances a `FakeClock` by exactly 1 second (`refill_per_sec: 1`) — no real time passes, no thread sleeps — and the very next call is admitted again. One more call right after (same clock instant) is refused: that refilled token was just spent, and no further time has passed.

The example asserts every transition inline (`assert!`/`assert_eq!`), so a regression in the bucket math or the config lowering fails the run, not just the eyeball check.
