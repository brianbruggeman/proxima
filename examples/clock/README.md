# clock

Time as an injectable seam. Schedule against a `Clock`, never `sleep`.

## Builds on

[transform](../transform/README.md) — the `Pipe` this example's `Timer` is written as, generic over the clock instead of a fixed one.

## What it demonstrates

`proxima_primitives::pipe::capabilities::Clock` is the trait every timer-driven combinator in proxima (`Retry` today, `Backoff`/`RateLimit`/`Deadline` later) is generic over instead of calling a thread- or timer-`sleep()` directly:

```rust
pub trait Clock {
    type Delay: Future<Output = ()>;
    fn now_nanos(&self) -> u64;
    fn delay(&self, dur: Duration) -> Self::Delay;
}
```

Two facts fall out of that shape:

- `now_nanos` and `delay` are the only way time enters the logic — nothing downstream can reach for the wall clock or a bare sleep behind the trait's back.
- `Delay` is an associated type, not `Pin<Box<dyn Future>>>` — the executor holds the delay inline in its state machine, no boxing, no alloc, no_std.

Production injects `TimeClock` (`proxima_primitives::pipe::clock::TimeClock`), which wraps `proxima-time`'s real monotonic driver. A test injects a `Clock` backed by nothing but a `Cell<u64>` — `now_nanos` reads it, and the only thing that ever writes it is an explicit `advance` call. Same trait, same `Timer` pipe on top; the only thing that changes is which `Clock` gets built into it. That is the entire mechanism the whole resilience unit stands on: nothing in `Retry`, `Backoff`, `RateLimit`, or `Deadline` needs a real clock or a test runtime slowed down by real waits — it needs a `Clock`.

The example builds a `Timer` (`Pipe<In = Duration, Out = &'static str>`) over a `FakeClock`, then polls its `call` future by hand (`Waker::noop`, no runtime) three times, calling `FakeClock::advance` between polls instead of waiting:

| poll | fake clock | result |
|---|---|---|
| #1 | `now_nanos = 0` | `Pending` — 30s hasn't happened |
| advance | `+15s` | |
| #2 | `now_nanos = 15_000_000_000` | `Pending` — halfway, still not due |
| advance | `+15s` | |
| #3 | `now_nanos = 30_000_000_000` | `Ready("fired")` — deadline crossed |

## Run

```
cargo run --example clock
```

## What you'll see

```
real clock (TimeClock, backs Retry/Backoff/Deadline in production): TimeClock

fake clock (FakeClock, starts at 0, only moves when told):
  now_nanos = 0

Timer scheduled for 30s against the fake clock. No thread sleeps, ever:
  poll #1 (t=0s):  Pending  — 30s hasn't happened, nothing is waiting on a clock tick
  advance(+15s) -> now_nanos = 15000000000
  poll #2 (t=15s): Pending  — halfway there, still not due
  advance(+15s) -> now_nanos = 30000000000
  poll #3 (t=30s): Ready("fired")  — deadline crossed, timer fires

zero real time passed. zero sleeps. zero threads parked — the fake clock made it deterministic.
```

The whole run finishes in whatever time it takes to print — `advance` moves the clock's `Cell<u64>` by 15 real seconds' worth of nanoseconds each call, but the process never waits for a single one of them. That is what "inject the clock" buys: the same `Timer` code that would really wait 30 seconds under `TimeClock` in production fires instantly and deterministically under `FakeClock` in a test, because the wait was never `sleep` — it was always "ask the clock".
