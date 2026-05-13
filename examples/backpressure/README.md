# backpressure

**Builds on:** gate, fan-in

## The one concept

Backpressure is not a feature you bolt on — it's what happens when a producer
and a consumer disagree about speed, and you have to pick who pays for the
mismatch. There is no `Backpressure` type in proxima. There is a queue with an
overflow policy (`BoundedQueue`/`FailMode`), a predicate that decides what
gets through (`Decide`/`Filter`), a dormancy switch that starves a
producer at the source (`Demand`/`AtomicGate` — the gate primitive), a
pull-mode drain that only takes what the consumer asks for
(`BatchSource` — the fan-in primitive), an accumulator
(`Batch`), and a last-write-wins cell (`Live`). Every strategy below is one
of those primitives pointed at a producer that outruns its consumer. Picking a
strategy is picking what you're willing to lose: time (block), the newest item
(drop-newest), the oldest item (drop-oldest), most of the stream (sample),
intermediate updates (coalesce), or nothing at all — at the cost of the
producer doing extra bookkeeping (batch, credit, demand).

## Strategy menu

| strategy     | primitive                              | when to use                                             | loss profile                          |
|--------------|-----------------------------------------|----------------------------------------------------------|----------------------------------------|
| block        | `BoundedQueue::enqueue_assisting`       | producer can afford to wait; losing an item is worse     | none — producer pays latency           |
| drop-newest  | `BoundedQueue::enqueue` + `FailMode::DropNewest` | queue order matters more than freshness          | the arriving item, queue untouched     |
| drop-oldest  | `BoundedQueue::enqueue` + `FailMode::DropOldest` | freshness matters more than order                | the oldest queued item                 |
| sample       | `Decide` (the seam `Filter` wraps)      | a statistical view of a high-volume stream is enough      | most of the stream, evenly             |
| credit       | `BatchSource::drain_batch`              | consumer must bound its own inflight work                | nothing — backlog just waits           |
| demand       | `Demand` + `AtomicGate`                 | no consumer is attached yet (cold start, disconnected)   | nothing queued — sends are a no-op     |
| batch        | `Batch`                                 | per-item flush cost dominates (syscalls, network RTTs)   | none — latency traded for fewer flushes|
| coalesce     | `Live`                                  | only the latest state matters (gauges, config, position) | every superseded value                 |

Two strategies named in the rung's one-liner aren't shown here as their own
primitive: **drop-newest/drop-oldest** are the *same* `BoundedQueue::enqueue`
call, distinguished only by `FailMode` — shown as two strategies above because
they have opposite loss profiles, not because they're different code.

## Run

```sh
cargo run --example backpressure
```

What you'll see (deterministic — no sleeps, no timing-dependent output):

```
block:       producer retried 1 time(s) for room, 0 dropped
drop-newest: kept [1, 2], dropped 3 (arrival order preserved)
drop-oldest: kept [2, 3], dropped 1 (freshest pair survives)
sample:      kept [0, 3, 6] of 9 produced (1-in-3)
credit:      pulled 2 of 5 queued (credit=2)
demand:      0 delivered while dormant, 3 delivered after controller.arm()
batch:       3 pushes coalesced into 1 flush of [1, 2, 3]
coalesce:    5 writes collapsed to the latest value (5)
all backpressure strategies verified
```

Each line is backed by an `assert!`/`assert_eq!` in `backpressure.rs` — the
printed line and the proof are the same run, not a paraphrase of it.

## Why block, credit, and demand are three different things

They're easy to conflate because all three avoid loss. They don't avoid the
same thing:

- **block** — the *producer* is throttled. It keeps trying to enqueue and pays
  the wait. The consumer's pace is invisible to it beyond "is there room."
- **credit** — the *consumer* is throttled. It declares up front how much
  work it can take (`out.len()` — the credit grant) and never pulls more,
  regardless of how much is queued.
- **demand** — the *pipe* is throttled at the source. Before any consumer has
  signaled it exists, sends aren't queued and aren't dropped — they're a
  no-op. This is the same `AtomicGate` the `gate` rung uses for dormancy;
  backpressure here is just dormancy applied to a producer instead of an idle
  resource.

## Note on scope

`Filter` (the config-expressible, HTTP-shaped admission pipe) is not used
directly for `sample` — it wraps `Decide` behind a `Rejectable`/`SendPipe`
bound that only pays for itself once there's a real inner pipe and a real
rejection response to build. The example composes `Decide` — the actual
predicate seam — directly, which is exactly what `proxima-primitives`' `When`
type documents itself for: "reuse it directly for sampling/canary/chaos
decisions." `Filter` is the seam wrapped for HTTP; `Decide` is the seam itself.
