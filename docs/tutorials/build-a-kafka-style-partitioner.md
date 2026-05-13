# Build a Kafka-style partitioner

You already know this shape. A producer hashes a key and picks a partition;
every consumer in a group pulls from the partitions it was assigned, fairly,
skipping whichever aren't ready; a sticky partitioner batches records onto one
partition before moving to the next. This tutorial teaches proxima's pipe
algebra by mapping it onto exactly that vocabulary — not by asking you to
forget what you know and start over.

The one thing your Kafka experience does **not** hand you for free is the
question this tutorial answers: for any given piece of that machinery — the
partitioner, the assignor, the consumer-group merge, the replication
fan-out — **is it a `Pipe`, or is it a strategy?** proxima draws that line
sharply, in one sentence, in the source code itself, and every primitive in
this tutorial is classified by it. Miss the line and you will reach for the
wrong shape the first time you try to extend one of these — add a case to an
enum where you should have implemented a trait, or implement a trait where
you should have written a pipe.

Every code block below is copied from a real file in this repository and
cited by path and line number. Every `cargo run` transcript is real output
this tutorial's author captured by actually running the command.

**Prerequisites:** none. The one proxima concept you need — what a `Pipe`
is — gets a two-paragraph primer in §1. For the rest of the pipe algebra
(chaining, filtering, gates, signals, HTTP serving), see
[Foundations](./00-foundations.md); nothing here depends on it.
**You will:** classify every piece of Kafka's distribution model — the
consumer-group merge, the key partitioner, the sticky assignor, replication —
as either a proxima `Pipe` or a proxima strategy, and see why the line falls
exactly where it does.
**New concepts (in order):** `Pipe` (the one-paragraph version) ·
`FanIn`/`Exhausted` (pull-merge, N→1) · `FanInStrategy`/`Select` (which
ready source wins) · `FanOut`/`FanPolicy` (push-broadcast, 1→N, with a
delivery policy) · the keying-pipe/distribution-strategy split (worked
example) · `DrainFanIn`/`DrainSource` and `DrainFanOut`/`DrainSink` (the
zero-copy duals).
**Answer key:** [`examples/fan_in/main.rs`](../../examples/fan_in/main.rs),
[`examples/fan_out/main.rs`](../../examples/fan_out/main.rs),
[`examples/fan_out_affinity/main.rs`](../../examples/fan_out_affinity/main.rs)
— `cargo run --example fan_in`, `cargo run --example fan_out`,
`cargo run --example fan_out_affinity`.

## Contents

1. The rule that organizes this whole tutorial
2. Two axes: fan-in and fan-out, in your vocabulary
3. The consumer-group merge is a pipe: `FanIn`
4. Which partition wins is a strategy: `FanInStrategy` and `Select`
5. Replication is a pipe too: `FanOut`
6. The ack policy is a strategy: `FanPolicy`
7. The gap: Kafka's producer partitioner routes to ONE, not to ALL
8. Splitting the partitioner: a keying pipe, a payload-blind strategy
9. Running it: `HashAffinity` vs `RoundRobin` vs `Sticky`
10. Extend, not add — and what goes wrong when you don't split it
11. The zero-copy duals: `DrainFanIn`/`DrainSource`, `DrainFanOut`/`DrainSink`
12. Kafka / Kinesis / WAL, translated
13. What you built, and what isn't here

## 1. The rule that organizes this whole tutorial

A `Pipe` is proxima's one building block: a type with an `In`, an `Out`, and
an `Err`, and a `call` method that turns one into the other. It is the same
idea whether `In`/`Out` are both real types (a transform), `In` is `()` (a
source — it produces without consuming, like a partition reader), or `Out` is
`()` (a sink — it consumes without producing, like a producer write). Here is
the shape, trimmed for space, copied verbatim from
`proxima-primitives/src/pipe/primitives.rs:89–99`:

```rust
pub trait Pipe {
    type In;
    type Out;
    type Err: Debug + 'static;

    fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>>;
}
```

That's the whole primitive. proxima's distribution machinery — merging
partitions, broadcasting replicas, routing by key — is built entirely by
composing this one shape with itself. (`primitives.rs` also documents four
sibling traits — `SendPipe`, `UnpinPipe`, `UnpinSendPipe` — that add `Send`
and/or `Unpin` on top of the same contract; you'll meet two of them below,
where the distribution machinery actually needs them.)

Now the rule. Every piece of Kafka's distribution model is either something
that reads the record, or something that only ever sees a control value
derived from it — a key, an index, a count. proxima's source code states the
difference as a lint, verbatim, in the doc comment above the trait that
chooses which ready source a merge takes next
(`proxima-primitives/src/pipe/fan_in.rs:88–90`):

> The line, and it is readable straight off the signature: **if the item
> passes through it, it is a pipe; if it only answers a control question and
> never sees the item, it is a strategy — a plain function.**

Read that twice, because the rest of this tutorial is nothing but applying
it. A **pipe** takes the record (or something derived from the record) in
and hands something back — the record passed *through* it, and it is free to
read every byte. A **strategy** never receives the record at all; it answers
one narrow control question — *which* source, *which* partition, *do we
require every ack* — from data that is already reduced to the bare minimum
the question needs. The signature is the proof, not a comment claiming it: a
strategy that needs more than `(&self, key, n)` to answer "which partition"
is, definitionally, reading the record — and the moment it does, it has to
become a pipe, because that's what "reads the record" means in this algebra.

This line was not always drawn correctly in this codebase. The commit that
opened the fan-in strategy trait you'll meet in §4
(`67074baf`, "make the fan-in strategy an open trait") says so directly in
its own message: an earlier combinator called `Decide` took the item and
answered a bare `bool` — and because a `bool` throws away both the item and
the reason, two more types (`Rejectable`, `OnReject`) had to be grown just to
carry back what the `bool` destroyed. `Decide` is gone now; a filter is
`predicate.and_then(inner)` — `and_then` chains two pipes so the second only
runs if the first succeeded, and the predicate is an ordinary pipe,
`In -> Result<In, Err>`, not a bespoke `bool`-returning seam
(`proxima-primitives/src/pipe/filter.rs:24–33`). The
lesson generalizes: getting the pipe/strategy line wrong doesn't fail loudly,
it fails by *accretion* — companion types grow up around the wrong shape to
carry back what it threw away. Watch for that shape as you read the rest of
this tutorial; every strategy below is deliberately too narrow to need one.

## 2. Two axes: fan-in and fan-out, in your vocabulary

Kafka's distribution model has exactly two directions, and proxima names
them the same way:

- **fan-in**: N sources merged into 1 — a consumer pulling from every
  partition it was assigned, or every upstream a router reads from.
- **fan-out**: 1 input delivered to N destinations — a producer replicating
  a write to every in-sync replica, or mirroring traffic to a canary.

proxima has two *pairs* of primitives for this, not one — an owned/pull pair
and a zero-copy/push pair, built for different tiers:

| | owned, pull (`async`, waker-driven) | zero-copy, push (poll-mode, no waker) |
|---|---|---|
| fan-in (N→1) | `FanIn` — `fan_in.rs` | `DrainFanIn` — `drain_source.rs` |
| fan-out (1→N) | `FanOut` — `fanout.rs` | `DrainFanOut` — `drain_sink.rs` |

The left column is what most services want: async, works with any executor,
items are owned values. The right column is the kernel-bypass shape — DPDK
ring, NVMe queue, per-core telemetry ring — where an item is a borrowed
`&[u8]` view into a slot and copying it out would be the whole cost of the
operation. §3–§10 cover the left column, because that's where the
Kafka-shaped worked example lives; §11 introduces the right column and shows
where it currently stops short of the left column's design.

## 3. The consumer-group merge is a pipe: `FanIn`

`FanIn<S, Strategy, const N: usize>` merges `N` fixed sources into one
stream — the shape of a consumer pulling from every partition it owns,
taking only whichever has a record ready *right now* and skipping the rest.
Its own module doc says so directly: "the fan-in counterpart to the fan-out
family: N sources merged into one item stream"
(`proxima-primitives/src/pipe/fan_in.rs:3`).

`FanIn` **is** a pipe — in the source form (§1: `In = ()`, it produces
without consuming). Concretely it implements two of the sibling traits
mentioned in §1: `Pipe` and `UnpinPipe`
(`proxima-primitives/src/pipe/fan_in.rs:222–234` and `:236–248`):

```rust
impl<S, Strategy, const N: usize> Pipe for FanIn<S, Strategy, N>
where
    S: UnpinPipe<In = (), Err = Exhausted> + DropSafe,
    Strategy: FanInStrategy,
{
    type In = ();
    type Out = S::Out;
    type Err = Exhausted;
    // ...
}
```

`Exhausted` (`fan_in.rs:65–67`) is the merge's termination signal — a source
that will never produce again resolves it instead of a second
`Poll<Option<_>>` protocol next to `Pipe`. Every merged source has to
implement `UnpinPipe`, not the plain `Pipe` from §1 — `UnpinPipe` is the
sibling trait whose returned future is `Unpin`, so a caller can poll it in
place with no heap and no `unsafe`
(`proxima-primitives/src/pipe/primitives.rs:165`, doc comment above). A merge
over `N` sources has to hold every source's in-flight call future at once and
poll each one in turn; doing that with no heap is exactly what `UnpinPipe`
buys. Sources also need `DropSafe` (`proxima_core::markers::DropSafe`) — a
source found `Pending` this scan has its transient call future dropped and
gets re-asked fresh next scan, so a source must have no observable state that
depends on that future surviving.

Because `FanIn` implements `Pipe`, a `FanIn` can itself be one of the sources
inside a *bigger* `FanIn` — a merge of merges, e.g. per-broker partition
merges rolled up into a per-consumer merge — with no adapter type, proven by
a real test: `fan_in_nests_inside_a_bigger_fan_in`
(`fan_in.rs:469–485`).

This claim — "`FanIn` is a pipe" — is not asserted only in prose. It is
checked by the compiler on every build, in a test-only module whose whole
job is to fail to *compile* the moment it stops being true
(`proxima-primitives/src/pipe/mod.rs:267–314`, `mod algebra_claims`):

```rust
// fan-in IS a pipe, for any DropSafe UnpinPipe source.
fn _fan_in_is_a_pipe<S, Strategy, const N: usize>()
where
    S: super::primitives::UnpinPipe<In = (), Err = super::fan_in::Exhausted>
        + proxima_core::markers::DropSafe,
    Strategy: super::fan_in::FanInStrategy,
{
    assert_pipe::<super::fan_in::FanIn<S, Strategy, N>>();
}
```

`FanIn`, `FanInStrategy`, `Select`, and `Exhausted` are all public:
`proxima_primitives::pipe::{FanIn, FanInStrategy, Select, Exhausted}`
(re-exported at `proxima-primitives/src/pipe/mod.rs:186`).

## 4. Which partition wins is a strategy: `FanInStrategy` and `Select`

A merge can find more than one source ready in the same scan. *Which* one it
takes is answered by a separate seam, `FanInStrategy`
(`fan_in.rs:93–99`):

```rust
pub trait FanInStrategy {
    fn index(&self, step: usize, start: usize, n: usize) -> usize;
}
```

Read the signature against §1's rule: `step`, `start`, `n` — a position in
the scan and the source count. No record. No item. Not even the source
itself, just its index. That is the whole reason `FanInStrategy` is a trait
and not a pipe: it never has the record to read, so there is nothing for it
to be a pipe *of*. Picking an index runs once per source per scan on the hot
path; a pipe would build and poll a future to compute a `usize`
(`fan_in.rs:90–92`).

The built-in strategies cover the common consumer-group assignment shapes,
`Select` (`fan_in.rs:104–114`, trimmed):

```rust
pub enum Select {
    /// Resume the scan past whoever last emitted — fair, no source starves.
    RoundRobin,
    /// Always scan from the first source: earlier sources win every tie.
    Fifo,
    /// Always scan from the last source: later sources win every tie.
    Lifo,
}
```

`RoundRobin` is your consumer group's usual fair-share pull. `Fifo` is
priority order — put the sources in the order you want them preferred, and
notice there is deliberately no separate `Priority` variant: `Fifo` over an
ordered array already *is* priority order, and a `Priority` arm "would be a
second name for a choice you already made when you built the array"
(`fan_in.rs:76–78`). `FanIn::new` takes the sources and the strategy
together, always both (`fan_in.rs:143`):

```rust
pub fn new(sources: [S; N], strategy: Strategy) -> Self
```

Run the plain merge (`examples/fan_in/main.rs`, three upstreams — `orders`,
`payments`, `shipping` — each a scripted readiness schedule):

```
$ cargo run --example fan_in
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

`shipping` drains after two items while `orders` and `payments` still have
one queued each; `FanIn` marks it drained and the round-robin scan simply
steps around it — no error, no stall, matching Kafka's own "skip the
partition with nothing new, don't treat it as failed."

`FanInStrategy` is open **as of a very recent change** — worth knowing,
because it explains why the shape below (§10) works at all. Before
`67074baf`, `Select` was a closed three-variant enum welded onto `FanIn`;
weighted, least-loaded, or Kafka's own sticky assignor were simply not
expressible without editing this library. The commit turned `Select` into
one implementor of an open trait, `FanIn` itself gained a `Strategy` type
parameter (`FanIn<S, const N: usize>` became
`FanIn<S, Strategy, const N: usize>`), and the common case stayed exactly as
easy — `FanIn::new(sources, Select::RoundRobin)` still infers `Strategy`.
§10 shows the caller-defined strategy the commit's own test proves this
against.

## 5. Replication is a pipe too: `FanOut`

`FanOut<S, Policy>` is fan-in's dual: one input broadcast to every one of `N`
sink pipes — the shape of replicating a write to every in-sync replica, or
mirroring live traffic to a shadow/canary path. Its module doc: "broadcast
one input to N sink `SendPipe`s (a 1→N tee)"
(`proxima-primitives/src/pipe/fanout.rs:1`).

Concretely, `FanOut` implements `SendPipe`
(`proxima-primitives/src/pipe/fanout.rs:134`) — the sibling trait from §1
whose `call` is additionally `Send`, so it can be dispatched across cores.
It does **not** implement the plain `Pipe` from §1 today; there is no
`impl Pipe for FanOut` in the source, only `impl SendPipe for FanOut`. This
still satisfies §1's rule — the input passes through every sink, so
`FanOut` is unambiguously on the pipe side of the pipe/strategy line — it
just satisfies it through the cross-core sibling trait, not the base one. The
codebase's own compile-checked proof (§3's `algebra_claims` module,
`mod.rs:280–292`) states the claim exactly this way, in its own comment:
`// fan-out IS a pipe, for any sink and any fan policy.` — checked with
`assert_send_pipe`, not `assert_pipe`:

```rust
fn _fan_out_is_a_pipe<S, Policy>()
where
    S: SendPipe<Out = ()> + Clone,
    S::In: Clone + Send,
    S::Err: Send,
    Policy: FanPolicy,
{
    assert_send_pipe::<super::fanout::FanOut<S, Policy>>();
}
```

The input reaches every sink with the minimum number of clones — moved into
the last sink, cloned into the earlier ones, so N sinks cost N−1 clones, not
N (`fanout.rs:12–15`). Run it (`examples/fan_out/main.rs`, one
`"checkout order 42"` message, two arms — `primary` and `mirror`):

```
$ cargo run --example fan_out
fanning one request to 2 arms
primary arm received: ["primary: checkout order 42"]
mirror arm received:  ["mirror: checkout order 42"]
both arms received the one request, independently processed: fan-out proven
```

`FanOut` and `FanPolicy` are public:
`proxima_primitives::pipe::{FanOut, FanPolicy, AllOrNothing, BestEffort, IgnoreErrors}`
(`proxima-primitives/src/pipe/mod.rs:233`).

## 6. The ack policy is a strategy: `FanPolicy`

What happens when one sink fails is a separate, orthogonal axis — Kafka's
`acks=all` vs `acks=1` vs fire-and-forget. proxima names it `FanPolicy`, and
it is a strategy under §1's rule for an even stronger reason than
`FanInStrategy`: it never runs at all as a function over data. It is a
marker trait carrying two `const`s, so the failure reaction folds away at
compile time — no runtime branch on the hot path
(`proxima-primitives/src/pipe/fanout.rs:24–37`):

```rust
pub trait FanPolicy: Send + Sync + 'static {
    const SHORT_CIRCUIT: bool;
    const IGNORE_ERRORS: bool;
}
```

Three built-ins (`fanout.rs:39–63`), constructed by name at the call site
instead of a turbofish (`FanOut::all_or_nothing(sinks)`,
`FanOut::best_effort(sinks)`, `FanOut::ignore_errors(sinks)`):

| policy | maps to | `SHORT_CIRCUIT` | `IGNORE_ERRORS` |
|---|---|---|---|
| `AllOrNothing` | `acks=all`, stop on first NACK | `true` | `false` |
| `BestEffort` | try every replica, surface the first failure after | `false` | `false` |
| `IgnoreErrors` | fire-and-forget | `false` | `true` |

Like `FanInStrategy`, `FanPolicy` is an *open* trait — you can name a fourth
policy the library never defined by implementing it for your own
zero-sized marker type, the same "extend, not add" shape §10 shows for
`FanInStrategy`. It is a second, independent instance of the same repo-wide
pattern: the distribution primitive (`FanOut`) is closed; the axis that
decides *how it reacts to failure* is an open seam next to it, not a case
welded into the primitive.

## 7. The gap: Kafka's producer partitioner routes to ONE, not to ALL

Everything so far — §3's `FanIn`, §5's `FanOut` — covers two of Kafka's three
distribution shapes: the consumer-group merge (many → one, take what's
ready) and replication (one → all, broadcast). It does **not** cover the
third: a **producer partitioner**, which routes one record to exactly *one*
of N partitions, chosen by a key. `FanOut` cannot express this — it always
delivers to every sink; there is no "deliver to sink 2 only" mode, and no
`FanOutStrategy` trait anywhere in this crate (verified: no such name exists
in `proxima-primitives/src/pipe/fanout.rs` or anywhere else in the crate).
This is a real gap in the library today, not a simplification on this
tutorial's part.

`examples/fan_out_affinity/` closes the gap **consumer-side**, as a worked
pattern built entirely out of primitives you already have from §1 and §4 —
without adding a single new type to the library. Its own module doc frames
it exactly this way (`examples/fan_out_affinity/main.rs:3–19`):

> Fan-out AFFINITY: route one record to ONE of N partitions by a key, the way
> Kafka's producer partitioner does — the same key always lands on the same
> partition, so a whole customer's (or trace's) stream stays together.
>
> `proxima_primitives::pipe::FanOut` broadcasts one input to ALL arms — the
> "everyone" distribution. Affinity is the OTHER distribution: route to one
> arm by key.

Everything in this section and the next lives inside one file,
`examples/fan_out_affinity/main.rs`, which is an `[[example]]` **binary**
target, not a library module. Nothing defined there is `pub`; there is no
`use proxima_primitives::...::Distribute` you can reach for, because that
path does not exist. You cannot import this pattern — you copy it into your
own crate, which is exactly what an `examples/` directory is for.

## 8. Splitting the partitioner: a keying pipe, a payload-blind strategy

Kafka's own partitioner does two things in one function call: read the
record's key, then decide the partition. §1's rule says that cannot be one
seam in proxima, because "read the record" and "never see the record" cannot
both be true of the same thing. So the example splits it into two, each on
its own side of the line.

**Keying is a pipe.** It reads the whole record and produces a routing key —
the record passes *through* it, which is legal precisely because it is a
pipe (`examples/fan_out_affinity/main.rs:60–71`):

```rust
struct PartitionKey;

impl Pipe for PartitionKey {
    type In = Record;
    type Out = u64;
    type Err = Infallible;

    fn call(&self, record: Record) -> impl Future<Output = Result<u64, Infallible>> {
        let key = fnv1a(record.customer.as_bytes());
        async move { Ok(key) }
    }
}
```

(`fnv1a` is a small dependency-free hash defined in the same file,
`main.rs:179–186` — a real fleet would use the same hash on every producer,
the way Kafka's own default uses murmur2, so the mapping agrees everywhere.)

**Choosing the partition is a strategy.** It sees only the key `PartitionKey`
produced, never the record — the signature is the proof
(`examples/fan_out_affinity/main.rs:77–79`):

```rust
trait Distribute {
    fn partition(&self, key: u64, partitions: usize) -> usize;
}
```

`key: u64`, not `record: Record`. There is no way to widen this signature to
read the record without it becoming a pipe instead — which is precisely why
keying lives upstream, in its own pipe, and never inside `Distribute`.
Kafka's key partitioner, `hash(key) % partitions`, is `HashAffinity`
(`main.rs:81–89`):

```rust
struct HashAffinity;

impl Distribute for HashAffinity {
    fn partition(&self, key: u64, partitions: usize) -> usize {
        (key % partitions as u64) as usize
    }
}
```

The router composes the two, and — this is the part worth reading twice —
the router itself is the only thing that ever sees both the record and the
key at once; the strategy never does (`main.rs:144–156`):

```rust
fn route(records: &[Record], strategy: &dyn Distribute) -> [Vec<Record>; PARTITIONS] {
    let keyer = PartitionKey;
    let mut partitions: [Vec<Record>; PARTITIONS] = Default::default();
    for record in records {
        let key = block_on_ready(keyer.call(record.clone())).expect("keying is infallible");
        let index = strategy.partition(key, PARTITIONS);
        partitions[index].push(record.clone());
    }
    partitions
}
```

`keyer.call(record.clone())` — the pipe, reading the record. Then
`strategy.partition(key, PARTITIONS)` — the strategy, handed only the `u64`
that came out.

## 9. Running it: `HashAffinity` vs `RoundRobin` vs `Sticky`

Three strategies plug into the one `Distribute` seam, and the example runs
all three over the same eight-record stream (customers `ada`, `linus`,
`grace`, `dennis`, `ada`, `grace`, `linus`, `ada` — `main.rs:164–174`) so you
can see the difference directly. Real output, captured by running the
example:

```
$ cargo run --example fan_out_affinity

HashAffinity (key -> hash -> one partition)
  partition 0: ["linus#1", "grace#2", "grace#5", "linus#6"]
  partition 1: ["ada#0", "ada#4", "ada#7"]
  partition 2: ["dennis#3"]

RoundRobin (key ignored, scatter)
  partition 0: ["ada#0", "dennis#3", "linus#6"]
  partition 1: ["linus#1", "ada#4", "ada#7"]
  partition 2: ["grace#2", "grace#5"]

Sticky (caller-defined, library never heard of it)
  partition 0: ["ada#0", "linus#1", "grace#2", "dennis#3"]
  partition 1: ["ada#4", "grace#5", "linus#6", "ada#7"]
  partition 2: []

affinity proven: same key -> same partition; the strategy never saw a record.
```

Read the three blocks against each other:

- **`HashAffinity`** — every `ada` order (`#0`, `#4`, `#7`) lands on
  partition 1, every `linus` order on partition 0, every `dennis` order on
  partition 2. This is the Kafka guarantee: same key, same partition, always
  — because the strategy is a deterministic function of the key and nothing
  else. The example asserts this directly:
  `assert_affinity_holds` (`main.rs:202–215`) checks every record sits on
  `hash(customer) % PARTITIONS`.
- **`RoundRobin`** (`main.rs:93–111`, `Distribute::partition` ignores `_key`
  and walks a `Cell<usize>` cursor) — `ada`'s records scatter across
  partition 0 *and* partition 1. This is the contrast that proves the affinity
  above was the strategy's doing, not an accident of the input:
  `assert_customer_scatters` (`main.rs:219–231`) asserts at least one
  customer spans more than one partition.
- **`Sticky`** (`main.rs:113–142`, batch size 4) — the first four records
  land on partition 0, the next four on partition 1, in fat contiguous runs
  rather than per-key. This is Kafka's own sticky partitioner (2.4+):
  fewer, larger batches, at the cost of dropping the key-affinity guarantee
  entirely. Notice `Sticky::partition` also ignores `_key` — it is
  payload-blind *and* key-blind, pure `&self` state (a batch counter and a
  current-partition cursor), and it still satisfies `Distribute` because
  `Distribute` never demanded the key be used, only that it be the only
  input available.

Every future in this example resolves on its first poll — `PartitionKey` is
a plain hash, no real I/O — so `route` drives it with a one-shot
`block_on_ready` (`main.rs:236–244`) instead of a real executor. That's a
property of this example's payload, not of `Pipe` in general: a real keying
pipe reading from a network buffer would `.await` normally, under any
runtime.

## 10. Extend, not add — and what goes wrong when you don't split it

`Distribute` and `FanInStrategy` are both open traits, and both are open for
the identical reason: their signature is bounded so tightly to the control
question that nothing legitimate can ever need to widen it. `Distribute`
takes `(&self, key, n)`. `FanInStrategy` takes `(&self, step, start, n)`.
Neither has room to accept a record — and that is not a limitation someone
forgot to lift, it is the design: the moment a strategy needs the record, it
has stopped being a strategy and needs to become a pipe instead, upstream,
the way `PartitionKey` already is.

This is why both traits are simultaneously **open for extension** (implement
it, ship it in your own crate) and **closed for modification** (the library
ships only the stateless built-ins, and there is no reason to add more —
anything genuinely new is still expressible against the same three or four
arguments). Two real, working proofs, neither touching the library:

`StickyThen`, defined inside a `#[cfg(test)]` module in `fan_in.rs` itself —
"a strategy the library never heard of," pinning one source first, then
falling back to round-robin (`fan_in.rs:390–398`):

```rust
struct StickyThen(usize);
impl FanInStrategy for StickyThen {
    fn index(&self, step: usize, start: usize, n: usize) -> usize {
        if step == 0 { self.0 % n } else { (start + step) % n }
    }
}
```

And `Sticky` from §9, which is the same proof for `Distribute`, defined
entirely inside an example binary, not the library
(`examples/fan_out_affinity/main.rs:117–142`).

Recall §1's cautionary tale: `Decide` took the item and answered a `bool`,
and grew two companion types to carry back what the `bool` threw away. Look
at what `Distribute` and `FanInStrategy` do instead — they take *less* than
the item (a key, an index) and answer with *exactly* the value the caller
needs (a partition index, a source index), nothing more. A strategy that
needs to grow a companion type to carry back extra context is the same
symptom `Decide` had: proof the line was drawn in the wrong place, and the
fix is never a companion type, it's moving the read upstream into a pipe.

## 11. The zero-copy duals: `DrainFanIn`/`DrainSource`, `DrainFanOut`/`DrainSink`

Everything above is the **owned, pull** pair from §2's table: `async`,
waker-driven, items are owned values, works with any executor. proxima also
has a **zero-copy, push** pair for the kernel-bypass tier (DPDK ring, NVMe
completion queue, per-core telemetry ring) where an item is a borrowed
`&[u8]` view into a slot, and the entire cost of the operation would be
copying it out. This is genuinely different machinery, not a generic
parameter away from `FanIn`/`FanOut` — the module doc for the source side
explains why: an API shaped like `poll_next<'a>(&'a mut self) ->
Poll<Option<Item<'a>>>`, where each item borrows from `&mut self` for as
long as the caller keeps it (a "lending" API — the borrow is lent out, not
copied out), was tried and falsified: it does not compile inside an
array-scan loop on stable Rust
(`proxima-primitives/src/pipe/drain_source.rs:8–20`). The fix is a
**visitor-push** model instead of a pull: the source calls *you*, once per
ready item, and the borrow never outlives that one call.

`DrainSource` is the trait a zero-copy source implements
(`drain_source.rs:54–64`; `ControlFlow<()>` is `core::ops::ControlFlow` —
`Continue(())` means "keep draining," `Break(())` means "stop, I'm applying
backpressure," the same two states an `Option<()>` would need a comment to
explain):

```rust
pub trait DrainSource {
    type Item: ?Sized;

    fn drain_ready(
        &mut self,
        visitor: &mut dyn FnMut(&Self::Item) -> ControlFlow<()>,
    ) -> DrainState;
}
```

`DrainFanIn<S, const N: usize>` merges `N` of them, round-robin, with the
same cursor/live-set/remaining FSM shape as `FanIn`
(`drain_source.rs:70–75`). `DrainFanOut<K, const N: usize>` is the write-side
dual, pushing one borrowed item into every `DrainSink`
(`proxima-primitives/src/pipe/drain_sink.rs:240–275`):

```rust
pub fn push_all(&mut self, item: &K::Item) -> ControlFlow<()> {
    for sink in &mut self.sinks {
        sink.accept(item)?;
    }
    ControlFlow::Continue(())
}

pub fn push_best_effort(&mut self, item: &K::Item) {
    for sink in &mut self.sinks {
        if sink.has_capacity() {
            let _ = sink.accept(item);
        }
    }
}
```

`RingSource`/`RingSink` (`drain_source.rs:171`, `drain_sink.rs:47`) are the
concrete fixed-capacity stack-arena ring this tier is built for — no heap,
no copy beyond the one unavoidable slot write.

Here is the asymmetry worth knowing before you reach for this tier: **the
`Strategy`/`Policy` generalization from §4 and §6 has not propagated here.**
`DrainFanIn` takes no `Strategy` parameter at all — its round-robin order is
hardwired into `drain_each`, not a pluggable seam
(`drain_source.rs:70–75`, no type parameter beyond `S` and `N`). `DrainFanOut`
has no `Policy` type either — `push_all` and `push_best_effort` are two fixed
methods, not implementors of an open trait the way `FanPolicy` is. Both are
public (`proxima_primitives::pipe::{DrainFanIn, DrainSource, DrainFanOut,
DrainSink, RingSource, RingSink}`, `pipe/mod.rs:184–185`), both are
production-ready for what they do today, and both would need real design
work — not a small patch — to gain the same open seam `FanIn`/`FanOut`
already have. If your kernel-bypass consumer needs Kafka-style key affinity
on this tier, that design does not exist yet in this crate; build it the way
§7–§9 built one for the owned tier, or extend this library, but do not
assume the seam is already there.

## 12. Kafka / Kinesis / WAL, translated

The payoff — every piece of the vocabulary you started with, classified:

| your vocabulary | proxima classification | primitive | cite |
|---|---|---|---|
| consumer-group partition merge (pull from N assigned partitions) | pipe (`Pipe`/`UnpinPipe`) | `FanIn<S, Strategy, N>` | `fan_in.rs:131,222,236` |
| fair partition consumption (no partition starves) | strategy | `Select::RoundRobin` | `fan_in.rs:105–108,119` |
| static priority / ordered preference | strategy | `Select::Fifo` / `Select::Lifo` | `fan_in.rs:109–113,120–121` |
| custom assignor (e.g. cooperative-sticky) | strategy, open trait | your `impl FanInStrategy` (e.g. `StickyThen`) | `fan_in.rs:390–398` |
| replication to every in-sync replica / mirroring | pipe (`SendPipe`) | `FanOut<S, Policy>` | `fanout.rs:69,134` |
| `acks=all` / `acks=1` / fire-and-forget | strategy, open trait, compile-time | `FanPolicy` (`AllOrNothing`/`BestEffort`/`IgnoreErrors`) | `fanout.rs:24–63` |
| key-hash producer partitioner (`hash(key) % n`) | split: pipe (read key) + strategy (key → partition) | `PartitionKey` (pipe) + `HashAffinity` (strategy) | `examples/fan_out_affinity/main.rs:60–89` |
| keyless round-robin partitioner | strategy (example, consumer-side) | `RoundRobin` | `main.rs:93–111` |
| sticky partitioner (2.4+) | strategy, library never heard of it (example) | `Sticky` | `main.rs:117–142` |
| consistent hashing / rendezvous (HRW) — resize-stable partitioning | strategy — **same seam, not built** | none; would be `impl Distribute` over a ring, `&self`, key in | `examples/fan_out_affinity/README.md:67–70` (design note only) |
| WAL fan-out to replicas with an ack quorum | pipe (broadcast) + strategy (ack policy) | `FanOut` + `FanPolicy` | `fanout.rs` |
| kernel-bypass NIC/NVMe ring merge (zero-copy consumer) | pipe (visitor-push, **no strategy seam yet**) | `DrainFanIn`/`DrainSource` | `drain_source.rs:70,54` |
| kernel-bypass ring mirror (zero-copy replication) | pipe (visitor-push, **no policy seam yet**, two fixed methods) | `DrainFanOut`/`DrainSink`/`RingSink` | `drain_sink.rs:240,28,47` |

Every row that lands on "pipe" is something that reads the record. Every row
that lands on "strategy" answers a control question from data already
reduced to the minimum the question needs. That is the whole rule from §1,
applied to every shape Kafka, Kinesis, and a replicated WAL all have in
common — proxima did not invent a new distribution vocabulary, it drew one
line through the vocabulary you already had.

## 13. What you built, and what isn't here

You classified Kafka's full distribution model against proxima's own rule:
the consumer-group merge and replication are pipes (`FanIn`, `FanOut`); which
ready source wins and how failures are handled are strategies (`Select`,
`FanPolicy`); the producer partitioner splits into a keying pipe and a
payload-blind distribution strategy, because no single seam in this algebra
can be both at once. You proved two strategies "the library never heard of"
plug into that seam with zero library change, and you saw the same shape
break — into companion types papering over a lost item and reason — the one
time this codebase got the line wrong (`Decide`).

What this tutorial did **not** teach you, because it is not built:

- **`FanOutStrategy`.** There is no library seam for "route to one of N,"
  only "broadcast to all N." §7–§9's split (keying pipe + `Distribute`
  strategy) is a pattern demonstrated **consumer-side**, in one example
  binary — not a type you can `use`.
- **Consistent hashing / rendezvous (HRW).** Named in the affinity example's
  own notes as the resize-stable form of key affinity — key%n reshuffles
  every key on a partition-count change; these move only ~K/N keys — but
  not implemented anywhere in this repository as of this writing.
- **`DrainFanIn`/`DrainFanOut` strategy and policy seams.** §11's asymmetry:
  the zero-copy tier is still hardwired round-robin / two fixed push
  methods. The `Strategy`/`Policy` generalization that landed for the owned
  tier has not propagated here.

None of these are hedged as "coming soon" — they are gaps, stated plainly,
because a teaching document that presents an unlanded type as real teaches
you to write code that doesn't compile.

Where to go next: [Foundations](./00-foundations.md) covers the rest of the
pipe algebra this tutorial didn't need — chaining (`and_then`), filtering,
gates, signals, and turning a pipe into an HTTP server.
[Build a load balancer](./build-a-load-balancer.md) is the read-side mirror
of this tutorial's §3 — one request in, one backend picked out of many,
skipping the unhealthy — hand-rolled rather than built on `FanIn`, which is
itself worth noticing once you've read this far: not everything that
*could* be `FanIn` is worth being `FanIn`.
