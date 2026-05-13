# cancellation — a fired Signal, composed five ways

## Builds on

[signal](../signal/README.md) — the fire-once, sticky completion level every
mode here fires or observes.
[deadline](../deadline/README.md) — a timeout is one of the five modes: the
same fired `Signal`, decided by a clock instead of a caller.

## What it demonstrates

Cancellation in proxima is not a runtime feature — it is what falls out of
composing a fired [`Signal`](../proxima-core/src/signal.rs) with whatever
already holds the work. There is no `CancellationToken`, no `AbortHandle`,
no bespoke cancel type: every mode below is the same primitive, wired to a
different place.

| Mode | Mechanism | Cleanup guarantee |
|---|---|---|
| **Cooperative** | a checkpoint reads `Signal::is_fired()` between units of work | explicit — whatever the checkpoint releases before it returns |
| **Deadline** | `Deadline::expired(clock.now_nanos())` decides when to call `Signal::fire()` | same as cooperative; the clock replaces the caller as the thing deciding to fire |
| **Drop-to-cancel** | `Signal::guard()` fires its scope the instant the driver handle drops | none by itself — see cancel-with-cleanup for the actual finalizer |
| **Propagating** | `Signal::child()` merges every ancestor level into a descendant's own | inherited — each descendant still runs its own cleanup independently when it observes the fire |
| **Cancel-with-cleanup** | a `Drop` finalizer held by the cancelled value | exactly once, guaranteed by the language, not by discipline |

Two things repeat on purpose:

- **Signal decides *when*, never *what*.** Cooperative, deadline, and
  drop-to-cancel are three different answers to "when does the signal
  fire" (a caller, a clock, a dropped handle) — the fired level itself is
  identical in all three, and `is_fired()`/`fired()` don't know or care
  which one it was.
- **Cleanup is a drop, not a callback.** Cancel-with-cleanup's finalizer
  isn't invoked by `Signal` — it runs when the value carrying it is
  dropped, which is exactly the same event drop-to-cancel keys its
  cancellation off of. A cancelled future that owns a `Drop` guard gets
  its cleanup for free the moment it's discarded, whether that's because a
  checkpoint returned `Ready` and the caller let it go out of scope, or
  because the caller abandoned it outright (drop-to-cancel).

Everything is driven by hand — `poll_once` against `Waker::noop()`,
advancing a `Cell`-backed `FakeClock` for the deadline case — the same
no-sleep discipline `clock` and `deadline` used. "Slow" work is a poll
count, not real time, so every outcome (2 of 5 steps before cancel, a
clock crossing a 5s budget, a driver dropped one step short of commit) is
exact and reproducible on every run.

`Race<Sink, Policy>` (`proxima-primitives`'s concurrent fan-out) is the
pipe-algebra sibling of drop-to-cancel at a coarser grain: it documents
its own cancellation contract as "dropping a losing future IS the
cancellation," gated by a `DropSafe` marker bound. Same mechanism, this
example just shows it at the `Signal`/`Future` level instead of across a
fan-out of `SendPipe`s.

## Run

```
cargo run --example cancellation
```

## What you'll see

```
cancellation: a fired Signal, composed five ways

--- cooperative: Signal::is_fired() checked at each checkpoint ---
  checkpoint 1: pending, 1 step(s) done
  checkpoint 2: pending, 2 step(s) done
  signal.fire() — cancel after 2 of 5 steps
  checkpoint 3: cancelled at checkpoint
  completed = 2, resource_open = false

--- deadline: Deadline::expired(clock.now_nanos()) fires the signal ---
  advance(+3s) -> now_nanos = 3000000000 — under the 5s budget
  advance(+4s) -> now_nanos = 7000000000 — past the 5s budget
  signal.is_fired() = true — the clock cancelled, not a caller

--- drop-to-cancel: Signal::guard() fires its scope when dropped ---
  step 1: pending, driver still alive
  step 2: pending, driver still alive
  drop(driver) — the caller abandons the operation before it commits
  next poll: cancelled: driver dropped
  committed = false

--- propagating: Signal::child() merges ancestor levels into a descendant ---
  parent.fire()
  child_a=true child_b=true grandchild=true unrelated=false
  branch.fire() leaves root untouched — propagation flows root -> leaves, never back up

--- cancel-with-cleanup: a Drop finalizer, guaranteed exactly once ---
  step 1: pending, finalizer not run yet
  step 2: pending, finalizer not run yet
  cancelled, but the finalizer runs on drop, not on Ready
  cleanup_runs = 1 — exactly once, on the cancel path

all five modes: one primitive (Signal), five compositions, zero sleeps.
```

Cooperative stops at 2 of 5 steps — the checkpoint that would have run step
3 sees the fired signal instead and releases its resource. Deadline never
calls `signal.fire()` from anywhere but the clock check, and only after
the fake clock has been advanced past the 5s budget. Drop-to-cancel's
`committed` stays `false`: the work was one poll away from its commit when
the driver dropped, and the fired signal wins the very next poll.
Propagating's single `parent.fire()` reaches a child, a second child, and
a grandchild two levels down without ever touching them directly, while a
signal with no shared ancestor is provably untouched — and the reverse
direction (`branch.fire()`) proves that untouched-ness isn't an accident:
a child firing never reaches its parent. Cancel-with-cleanup's
`cleanup_runs` stays `0` right after the cancelled poll — proof that
observing `Ready("cancelled")` and the finalizer running are two different
events — and only becomes `1` once the value carrying the guard is
actually dropped.
