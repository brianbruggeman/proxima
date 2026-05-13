# deadline — a timeout as a fired completion

## Builds on

[signal](../signal/README.md) — a deadline = a signal fired by the clock.
[clock](../clock/README.md) — the injectable time seam a deadline is checked against; same `FakeClock`, driven by hand, no real sleeps.

## What it demonstrates

A timeout is not a special primitive — it is a completion (`signal`) whose fire condition is "the clock (`clock`) has crossed this instant" instead of "the terminal item arrived" or "the work finished."

`proxima_primitives::pipe::resilience::Deadline` is a plain timestamp comparison:

```rust
pub struct Deadline { /* deadline_nanos: u64 */ }

impl Deadline {
    pub fn new(now_nanos: u64, budget: Duration) -> Self;
    pub fn remaining(&self, now_nanos: u64) -> Duration;
    pub fn expired(&self, now_nanos: u64) -> bool;
}
```

No timer thread, no `sleep`, no future of its own — `expired(now_nanos)` is a `>=` on two `u64`s. The example wraps an inner future in a `DeadlineGuard` that polls the inner future first and, only if it isn't ready yet, asks `deadline.expired(clock.now_nanos())`. The instant that flips true, the guard fires a `Signal` (once, for good — the same sticky fire-once level `signal` taught) and resolves `Err`, dropping the inner future without polling it again.

"Slow" for the inner operation is modeled as a poll count, not real time, so the clock and the work advance on independent schedules — the example can drive either one ahead of the other by hand:

- **case 1 (in time):** budget 3s, inner needs 2 polls. Clock advances 1s between polls; by the time the inner op is ready (poll #3, t=2s) the deadline (3s) hasn't been crossed. `Ok`, and the deadline's `Signal` never fires.
- **case 2 (over deadline):** budget 2s, inner needs 5 polls. Clock advances past 2s (to t=3s) while the inner op still has 3 polls left. The deadline fires first: `Err(DeadlineExceeded)`, and the inner op is cancelled mid-flight, never finishing.

Zero real sleeps in either case — every "elapsed" second is `FakeClock::advance`, and every check of "has the deadline passed" is `Deadline::expired` against whatever `Clock::now_nanos()` currently reads.

## Run

```
cargo run --example deadline
```

## What you'll see

```
deadline: a timeout as a fired completion

--- case 1: inner finishes before the clock passes the deadline ---
budget: 3s, inner needs 2 polls to finish
  poll #1 (t=0s): Pending  — inner working, deadline not crossed
  advance(+1s) -> now_nanos = 1000000000
  poll #2 (t=1s): Pending  — inner working, deadline not crossed
  advance(+1s) -> now_nanos = 2000000000
  poll #3 (t=2s): Ready(Ok("operation complete"))  — inner finished first, deadline never fired (budget was 3s)
  fired.is_fired() = false  — confirmed

--- case 2: the clock passes the deadline before inner finishes ---
budget: 2s, inner needs 5 polls to finish — too slow
  poll #1 (t=0s): Pending  — inner working, deadline not crossed
  advance(+1s) -> now_nanos = 1000000000
  poll #2 (t=1s): Pending  — inner still working, still under budget
  advance(+2s) -> now_nanos = 3000000000 (past the 2s budget)
  poll #3 (t=3s): Ready(Err(deadline exceeded at 3000000000 ns))  — deadline crossed, inner cancelled with work still left
  fired.is_fired() = true  — confirmed, and it stays fired (sticky, like signal)

both cases: zero real sleeps — the fake clock made both outcomes deterministic.
```

Case 1's inner op finishes with time to spare (t=2s against a 3s budget) — the deadline's `Signal` stays unfired. Case 2's inner op still had 3 of its 5 required polls left when the clock crossed 2s — it was cancelled, not completed, and the `Signal` fired exactly once and stayed fired. That asymmetry is the whole mechanism: nothing about `Deadline` or `Clock` knows what the inner operation is doing; it only knows what time it is.
