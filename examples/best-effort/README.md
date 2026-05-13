# best-effort

**Builds on:** delivery

## The one concept

Best-effort is not a fourth delivery mode sitting next to at-most-once,
at-least-once, and exactly-once. It is the composite guarantee that falls
out when you refuse to let a local overload become a global stall: drop
locally, so a *presence* guarantee holds globally. `delivery`'s stronger
guarantees buy zero loss with a retry loop the producer can block on
(`enqueue_assisting`, `send_until_acked`); best-effort refuses that trade.
Every producer call into the lossy stage (`BoundedQueue::enqueue` under
`FailMode::DropOldest`, the same primitive `backpressure` teaches) returns
immediately — accepted or dropped — never retried, never blocked. The
pipeline as a whole is never *not* making progress.

This is `tracing`'s own model: a full telemetry buffer drops events rather
than block the request path. You lose completeness, you never lose
availability.

## Property menu

| property                              | how                                                                                 |
|----------------------------------------|--------------------------------------------------------------------------------------|
| availability — producer never stalls   | `queue.enqueue(item)` is O(1) and always returns; one call per item, zero retries — proven by `enqueue_calls == produced` |
| degradation — completeness sacrificed  | `FailMode::DropOldest` sheds the oldest queued item once the slow consumer falls behind — proven by `queue.dropped() > 0` |
| presence — never a total stall         | the consumer always catches up and the queue drains to empty — proven by `delivered.len() + queue.dropped() == produced` and `queue.len() == 0` |

## Run

```sh
cargo run --example best-effort
```

What you'll see (deterministic — a fixed producer/consumer schedule, no
sleeps, no real randomness):

```
best-effort: drop locally, presence holds globally

produced 20  delivered 7  dropped 13  (delivered + dropped = produced)
delivered, in order: [1, 6, 11, 16, 17, 18, 19]

contrast with delivery: at-least-once/exactly-once retry until every message is acked — zero loss, unbounded wait. best-effort refuses that trade: 20 enqueue calls for 20 items, 0 retries, 13 dropped locally — so the pipeline as a whole is never not making progress.
```

Each line is backed by an `assert!`/`assert_eq!` in `main.rs` — the printed
line and the proof are the same run, not a paraphrase of it. A fast producer
emits 20 items into a 4-slot `BoundedQueue`; a slow consumer only drains one
item for every five produced. The queue fills after 4 items and stays full
for the rest of the burst, so `DropOldest` evicts 13 of the 20 items before
the consumer ever gets to them — and every one of those 13 is still counted
(`dropped()`), not silently lost. The 7 that survive plus the 13 that were
dropped sum back to the 20 produced: nothing vanishes without being
accounted for, and something is always present at the far end.

## Why availability and completeness can't both be free

A producer that must never block and a sink that must never lose data are
in direct tension the instant the sink is slower than the source — one of
them has to give. `delivery` resolves the tension in favor of completeness:
`send_until_acked` retries until an ack lands, so the *producer* absorbs an
unbounded wait rather than let a message go missing. Best-effort resolves
the same tension in favor of availability: the lossy stage absorbs the
overload by shedding data it already holds, so the producer's call graph
never has a branch that waits. Neither resolution is "the right one" in the
abstract — `backpressure`'s strategy menu is exactly this choice, laid out
explicitly instead of being buried in a channel that decided for you. What
best-effort adds on top of `backpressure`'s lossy stage is the composite
claim: local drops compose into a *global* invariant — the pipeline
delivers `>= 1` output and it terminates — which is exactly what
`tracing`'s dropped-event counter promises a telemetry consumer: the
request path is never the thing that pays for a full buffer.
