# backoff — delay schedules against the clock

Retry with delay: constant, exponential, and jittered backoff, computed as a schedule and driven against an injected `Clock` — never a real sleep.

## Builds on

[retry](../retry/README.md), [clock](../clock/README.md) — backoff = retry + a clock-scheduled delay.

## What it demonstrates

`Backoff` (`proxima_primitives::pipe::resilience::backoff`) is pure math: given a 0-based attempt number it returns a `Duration`, nothing else, no I/O.

1. **CONSTANT** — `Backoff::Constant(duration)`: the same delay every attempt.
2. **EXPONENTIAL** — `Backoff::Exponential { initial, factor, max }`: `initial * factor^attempt`, saturating at `max`. Base sequence: `100ms, 200ms, 400ms, 800ms, 1.6s, 2s, 2s` (capped).
3. **JITTER** — a `Jitter` variant (`Full`, `Equal`, `Decorrelated`) randomises the base delay using caller-supplied entropy (`rand: u64`). No global RNG, no wall-clock read: the same `(attempt, rand, prev)` always reproduces the same delay.

`Retry` (`proxima_primitives::pipe::resilience::retry_exec`) is the executor that turns that math into an actual schedule: after a retryable outcome it calls `clock.delay(after)` and awaits it before the next attempt. This example's `ManualClock` resolves every `delay` immediately — no `std::thread::sleep`, no `tokio::time::sleep` — and advances its own `now_nanos` by the requested duration, recording each requested duration into a schedule. The retry loop still runs to completion in real time on the order of microseconds; the elapsed-time bookkeeping is entirely virtual, and it is deterministic because the clock, the backoff math, and the jitter entropy are all caller-supplied rather than read from the environment.

## Run

```
cargo run --example backoff
```

## What you'll see

```
constant: the same delay every attempt
attempts made: 4
delay schedule: [50ms, 50ms, 50ms]
clock advanced to: 150000000ns (no real sleep occurred)

exponential: delay doubles, saturates at max
attempts made: 8
delay schedule: [100ms, 200ms, 400ms, 800ms, 1.6s, 2s, 2s]
clock advanced to: 7100000000ns (no real sleep occurred)

jitter: randomised on top of the exponential base
-- bit-exact: Backoff::delay with caller-supplied rand, Jitter::Full --
  attempt 0: base=100ms rand=0 -> jittered=0ns
  attempt 1: base=200ms rand=150000 -> jittered=54ms
  attempt 2: base=400ms rand=999999999 -> jittered=234ms
  clock advanced to 288000000ns scheduling 3 delays, zero real sleeps
-- integrated: Retry drives Jitter::Equal end-to-end over the injected Clock --
  Jitter::Equal schedule: [50ms, 147ms, 304ms]
  every delay landed in [base/2, base] — jittered, still bounded, still no sleep
```

Jittered millisecond values above are illustrative; the example's assertions check ranges and determinism (a rerun of the same `(attempt, rand)` reproduces the same jittered delay via `Backoff::delay` directly), not fixed numbers where the algorithm intentionally varies.

- **CONSTANT**: 4 attempts produce exactly 3 gaps, every gap 50ms — `RetryController` calls `backoff.delay(attempt, ..)` once per retry, and `Backoff::Constant` ignores `attempt` entirely.
- **EXPONENTIAL**: the schedule is the doubling sequence up to `max`, read straight off `ManualClock`'s recorded durations, not recomputed by the example — proving `Retry` actually schedules what `Backoff::base_delay` predicts.
- **JITTER**: the first block calls `Backoff::delay` directly with hand-picked `rand` values and feeds each result to the clock by hand, proving the bit-exact math (`Full` jitter is always `<= base`, `Equal` is always in `[base/2, base]`) and that replaying the same `(attempt, rand)` is bit-for-bit reproducible. The second block runs the same `Jitter::Equal` shape through the real `Retry` pipe with a fixed `rand_seed`, proving the executor's internal entropy stream produces a schedule that still respects the shape's bounds end to end.
- Every section prints `clock.now_nanos()` after the run: it always equals the sum of the recorded schedule, and the whole example completes in real time far faster than the virtual delays it schedules — proof nothing actually slept.
