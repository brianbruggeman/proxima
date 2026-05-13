# retry — re-run a pipe on a `Retryable` decision

## Builds on

[filter](../filter/README.md) — retry is a `Retryable` decision over re-runs.

## What it demonstrates

`filter` teaches the gate shape: a `Decide<In>` predicate looks at the INPUT
and decides whether the inner pipe runs at all. Retry is the same shape
turned around — a decision looks at the OUTCOME of an attempt and decides
whether the inner pipe runs AGAIN. `RetryController::on_outcome` is that
decision as a pure function: no I/O, no clock, no sleeping. It takes the
last outcome (via the payload's `Retryable` impl) plus attempt/deadline
state and returns a `RetryAction` — `Retry { after }`, `Done`, or
`Exhausted`. Unlike `Filter<Inner, P>`, `RetryController` is not itself a
`Pipe` wrapper; its own doc comment says it plainly: "the caller drives the
attempt loop." This example's `retry_call` is that caller — call the pipe,
ask `on_outcome` what to do with the result, repeat.

Three jobs run through the same `retry_call` loop, each landing on a
different `RetryAction`:

- **retryable, succeeds** — the worker returns a `503`-shaped status (in the
  default `RetryRules`' retryable set) twice, then `200`. `on_outcome`
  returns `Retry` for each `503` (`retry_status` is in the set); once the
  worker returns `200` — outside the retryable set — the same check flips to
  `Done`.
- **retryable, exhausted** — the worker always returns `503`. `on_outcome`
  keeps returning `Retry` until the attempt count reaches `max_attempts`,
  then `Exhausted` — the last (still-failing) outcome is returned as-is, not
  silently swapped for something else.
- **not retryable** — the worker returns `422`, a status the default
  `RetryRules` doesn't recognize. `on_outcome` returns `Done` on the very
  first attempt: the worker is never called again. This is the direct
  parallel to `filter`'s dropped items — the decision gates the re-run, the
  loop doesn't get to guess.

`proxima_primitives::pipe::retry::Retry<Inner>` is the ready-made `Pipe`
wrapper for HTTP/event pipelines built on the same idea, generic over any
inner `SendPipe`. It is worth naming precisely because it does NOT delegate
to `RetryController`: it inlines its own copy of the decision directly
against `RetryRules::should_retry`, and it actually awaits a real backoff
between attempts. This example never sleeps, so it stays on the pure
controller instead of the timer-backed combinator.

## Run

```
cargo run --example retry
```

## What you'll see

```
retry: re-run a pipe on a RetryController decision

-- retryable, succeeds before the cap --
  worker attempt 1 for job 1: status 503
  on_outcome: retry after 50ms
  worker attempt 2 for job 1: status 503
  on_outcome: retry after 100ms
  worker attempt 3 for job 1: status 200
job 1: status 200 after 3 attempts

-- retryable, exhausts the cap --
  worker attempt 1 for job 2: status 503
  on_outcome: retry after 50ms
  worker attempt 2 for job 2: status 503
  on_outcome: retry after 100ms
  worker attempt 3 for job 2: status 503
job 2: status 503 after 3 attempts (cap reached)

-- not retryable, passes straight through --
  worker attempt 1 for job 3: status 422
job 3: status 422 after 1 attempt (not retryable)

all three jobs ran through the same retry_call loop; only RetryController::on_outcome's decision on each outcome changed the attempt count
```

Job 1's worker is called exactly 3 times — the attempt count the assertions
check — matching two `503`s plus the attempt that finally lands `200`. Job
2's worker is called exactly `max_attempts` (3) times and never more:
`on_outcome` reports `Exhausted` once the cap is hit, and the loop returns
the last (still-failing) outcome rather than looping forever. Job 3's worker
is called exactly once: `422` is outside the default retryable-status set,
so `on_outcome` returns `Done` on the first attempt — the clearest proof
that retrying is gated by the decision, not by whether the loop feels like
calling the worker again.
