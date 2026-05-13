# chaos — fault injection as a pipe

Inject delay/drop/error; assert graceful degradation. `chaos.rs` is itself a pipe.

## Builds on

[retry](../retry/README.md) — the same pipe re-run absorbs an injected error.
[fallback](../fallback/README.md) — the alternate-pipe route absorbs total primary failure.

## What it demonstrates

Chaos testing in proxima is not a framework bolted on from outside — it is a `Pipe` composed IN FRONT of the system under test. `Chaos<Inner>` is a decorator: on every call it rolls a small SEEDED xorshift64* PRNG against a `ChaosPolicy` and injects exactly one of three fault kinds, or lets `inner` run clean.

| Fault kind | How injected | What absorbs it |
|---|---|---|
| `Error` | `Chaos` returns `Err(ChaosFault::Injected)` — `inner` is never called | `RetryController` re-runs the same pipe |
| `Dropped` | `Chaos` returns `Err(ChaosFault::Dropped)` — a blackholed response, `inner` never called | `RetryController` re-runs the same pipe |
| `Delay` | `Chaos` advances a fake `FaultClock` by a fixed amount, then calls `inner` normally | tolerated — not a failure, just added latency; nothing needs to absorb it |

No real sleeps and no real randomness anywhere: the fake clock is a `Cell<u64>` moved by `advance`, never `std::thread::sleep`, and the PRNG is seeded so the exact same fault sequence reproduces on every run — the same injected-clock idiom as `clock`, `backoff`, and `circuit_breaker`, applied to fault injection instead of timing.

Two scenarios stack a resilience combinator in front of the same `Chaos`-wrapped, otherwise-healthy `upstream_service`:

1. **`Chaos(35% error + 15% drop + 10% delay)` + `retry(4)`** — every attempt has a 50% chance of a direct fault. `RetryController::on_outcome` treats any `Err` as retryable (the default `RetryRules`), so `retry_call` just tries again. All 16 requests resolve `Ok`, some needing 2–3 attempts.
2. **`Chaos(30% error + 30% drop + 20% delay)` + `Fallback`** — an 80%-hostile primary. `Fallback` never retries; on any primary error it routes straight to `Cache`, a secondary that never fails. All 16 requests resolve `Ok` regardless of how hostile the policy is — this is a structural guarantee, not a tuned one: `Cache` always answers, so the assertion holds for any chaos intensity.

`Delay` faults show up in both runs without breaking either combinator: the call still succeeds, just after the fake clock moves — proof that not every injected fault needs absorbing, some just need tolerating.

## Run

```
cargo run --example chaos
```

## What you'll see

```
chaos: fault injection as a pipe, absorbed by retry + fallback

-- chaos(50% fault) + retry(4): every request still resolves --
  request 0: resolved Ok(Response { id: 0, source: Upstream }) after 1 attempt(s)
  ...
  request 5: resolved Ok(Response { id: 5, source: Upstream }) after 3 attempt(s)
  ...

  faults injected: 2 error, 2 drop, 2 delay, 14 clean (20 attempts over 16 requests)
  simulated chaos-clock advance: 150ms (no real sleep)
  16/16 requests recovered — graceful degradation via retry


-- chaos(80% fault) + fallback: every request still resolves --
  request 0: resolved Ok(Response { id: 0, source: Upstream }) via Upstream
  request 2: resolved Ok(Response { id: 2, source: Cache }) via Cache
  ...

  faults injected: 4 error, 4 drop, 1 delay, 7 clean over 16 requests
  simulated chaos-clock advance: 120ms (no real sleep)
  cache served 8 of 16 requests (primary's faults routed here)
  16/16 requests recovered — graceful degradation via fallback
```

- **Retry scenario**: 20 attempts over 16 requests — 4 requests each needed one extra retry after `Chaos` drew a fault, and every one of them still lands `Ok`. The fault tally (2 error + 2 drop + 2 delay + 14 clean = 20) accounts for every attempt, not just the first.
- **Fallback scenario**: exactly one roll per request (fallback never retries), and the cache tally (8 of 16) plus the fault tally (4 + 4 + 1 = 9 faults, one of which was a tolerated `Delay` that still succeeded on `Upstream`) accounts for every request.
- Both runs `assert_eq!` the success count against the request count, so a regression in either combinator's absorption fails the example, not just the eyeball check.
