# fan-out

One request out to many; the dual of fan-in.

## Builds on

[transform](../transform/README.md) — each arm is a Pipe.

## What it demonstrates

Fan-out is a primitive of the pipe algebra: one input, broadcast to N sink
pipes. Fan-in (see `gate`) goes the other way — N sources pulled into one
stream, only what's ready. Fan-out is the write-side mirror of that: push
one input, every arm gets it.

A sink is not a special form — it is a pipe in its sink role, `In -> ()`,
the same shape `transform` taught. Broadcasting moves the input into the
last arm and clones it into the earlier ones, so N arms cost N-1 clones,
not N; for reference-counted payloads that's a refcount bump, not a copy.

Each arm here (`CapturingSink`) is written as `#[piped(send)] impl CapturingSink
{ async fn call(&self, ..) -> .. }` — the stateful form of `#[proxima::piped]`
(`00-foundations.md` section 7): the struct holds real fields (`label`,
`log: Arc<Mutex<Vec<String>>>`), so it can't be the fieldless struct the
macro's free-function form generates, and the macro only writes the trait
impl over the struct you already wrote. `FanOut<S, Policy>` itself, the
combinator each arm plugs into, is generic over `S` and stays hand-rolled —
the macro rejects a generic impl outright, so a combinator is never in
scope for it, only the leaf pipes wired into one.

What happens when one arm errors is a strategy, chosen independently of the
broadcast itself: all-or-nothing (stop on first error), best-effort (try
every arm, report what failed), or ignore-errors (try every arm, drop all
failures). This example wires the all-or-nothing strategy.

This is the shape behind mirroring and shadow traffic: send the same request
to a primary AND a shadow/canary/audit path, and let each arm process it on
its own terms without the caller knowing or caring how many arms there are.

## Run

```
cargo run --example fan_out
```

## What you'll see

```
fanning one request to 2 arms
primary arm received: ["primary: checkout order 42"]
mirror arm received:  ["mirror: checkout order 42"]
both arms received the one request, independently processed: fan-out proven
```

One `Message("checkout order 42")` is constructed exactly once and handed to
`fan.call(...)` exactly once. Both `primary_log` and `mirror_log` end up with
one entry each — proof the broadcast actually reached every arm, not just
the first or the last. Each arm records its own label, so the two entries
are provably independent captures (two different `Arc<Mutex<Vec<String>>>`
instances), not one shared log printed twice.

## In algebra terms

- fan-out (a primitive): one input broadcast to N sink pipes, each arm
  processing its own copy independently.
- fan-in (the read-side dual, see `gate`): N sources pulled into one stream.
- a sink is a pipe of the sink form (`In -> ()`) — not a bespoke shape.
- the fan error policy is a strategy, orthogonal to the broadcast itself:
  all-or-nothing / best-effort / ignore-errors.

## Read next

[Build a Kafka-style partitioner](../../docs/tutorials/build-a-kafka-style-partitioner.md)
puts this broadcast next to `FanIn` and a Kafka-style key partitioner
(`examples/fan_out_affinity`), and states the rule that classifies every
piece of that machinery as a pipe or a strategy.
