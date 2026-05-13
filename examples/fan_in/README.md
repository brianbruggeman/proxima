# fan-in — merge many, pull only the ready

## Builds on

[transform](../transform/README.md) — each source yields Pipe outputs.

## What it demonstrates

Fan-in is a primitive: several sources merged into one stream. Round-robin
is the strategy that keeps the merge fair — the cursor steps past whoever
just emitted, so one perpetually-ready source can't starve the rest. The
merge is pull-based: each poll asks every live source in turn, and only ever
returns an item from a source that has one ready *right now*. A source with
nothing ready this pass is skipped, not treated as failed or drained. The
merge's own honest "not ready" answer only fires when *every* live source
gave that same answer in the same pass.

This is the plain merge — no gate, no priority, just many sources to one
stream. It is the substrate a gated pattern builds on: wrap each source so
"not armed yet" becomes just another reason it answers "not ready this
pass", with no change to the merge itself. Learn the merge here, on its
own, before a pattern composes it with readiness.

## Run

```
cargo run --example fan_in
```

## What you'll see

```
fan-in: merge 3 upstreams, pull only the ready
poll 1: nothing ready yet, all live upstreams pending
poll 2: drained ("orders", 1)
poll 3: drained ("payments", 10)
poll 4: drained ("shipping", 100)
poll 5: drained ("orders", 2)
poll 6: drained ("payments", 20)
poll 7: all upstreams drained
drained 5 items total: 2 orders, 2 payments, 1 shipping
```

Three upstreams (`orders`, `payments`, `shipping`), each a scripted
readiness schedule over a real `Counter` `Pipe`. All three start not-ready:
poll 1 scans every live source, finds nothing, and the merge itself reports
`Pending` — proof that "skip not-ready" composes all the way up, not just
per-source. From poll 2 on, sources interleave in round-robin order as they
become ready, and each `Ready(Some(item))` came from an actual `Counter::call`,
not a canned literal — merged items really are `Pipe` outputs. `shipping`
drains after two items while `orders` and `payments` still have one queued
each; `FanIn` marks it drained and the round-robin scan simply steps around
it, no error, no stall. The final `assert_eq!` on the exact drain order
pins the round-robin contract: this is not "eventually all items arrive in
some order", it's a deterministic interleave a regression would break.

## In algebra terms

The primitive, and nothing more:

- **fan-in** merges many sources into one, taking only what is ready. It is itself
  a pipe in the **source form** (`() -> Item`) — you pull it and get the next item,
  the same as pulling any source. Many-to-one, and the merge never inspects what
  produced an item, only whether one is ready.

Everything below is **this example's choices**, not what a fan-in is. Swap any of
them and it is still a fan-in:

- **fairness** — a *strategy*. This one is round-robin (step past whoever just
  emitted, so no source starves the rest); least-loaded or random would be the
  same shape with a different dial.
- **arity** — fixed at the type level here. A runtime-sized merge is the same
  primitive.
- **readiness and termination** — how a source says "nothing yet" versus "never
  again", and when the merge itself is finished, are contract details of this
  implementation.

## Read next

[Build a Kafka-style partitioner](../../docs/tutorials/build-a-kafka-style-partitioner.md)
puts this merge next to `FanOut` and a Kafka producer partitioner, and states
the pipe-vs-strategy rule `FanInStrategy` is built on.
