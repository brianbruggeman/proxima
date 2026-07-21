# The pipe algebra

Everything in proxima is a **pipe**, and big things are small pipes composed.
That is the whole system. This page is the map: it names the four ideas the
algebra is built from and points you at the chapter — and the source file —
that teaches each one. Read it once, then read the chapters in order.

A pipe is a typed, async function with a name:

```text
call(In) -> Result<Out, Err>
```

You give it an input of type `In`; it hands back either an `Out` or an `Err`.
Nothing more. Because that shape is so small, you can build the rest of a
network stack by wrapping pipes in other pipes — and that is exactly what the
chapters do.

The algebra has **four layers**. Each layer is built out of the one before it,
so once you understand a pipe, the rest is composition rather than new
machinery.

> **This is algebra, not code.** These pages describe *shapes and how they
> compose*. The Rust that proves each shape lives in the chapters, as compiled
> examples — the map here, the territory there. When they disagree, the source
> wins.

## 1. Form — the pipe itself

A pipe is one trait, `Pipe`. You do not learn four traits; you pick the `In`
and `Out` types, and the *same* trait takes on four familiar shapes. `()` is
Rust's "nothing" value, and choosing it for `In` or `Out` is what makes a
shape degenerate:

| form      | shape       | what it is                            |
|-----------|-------------|---------------------------------------|
| transform | `In -> Out` | turns one thing into another          |
| source    | `() -> Out` | takes nothing, produces something     |
| sink      | `In -> ()`  | takes something, produces nothing     |
| observe   | `In -> In`  | hands its input back; acts on the side |

Only `transform` is load-bearing. `source`, `sink`, and `observe` are that
same trait with a `()` chosen for one side. There is no `Source` trait, no
`Sink` trait — just types.

→ **Chapter:** [transform](transform.md). **Source:**
`proxima_primitives::pipe` — the four forms are a compiled example
right in the trait's own doc-comment, so they cannot describe a pipe that no
longer exists.

## 2. Chain — how one pipe extends into the next

If a pipe is `In -> Out` and another is `Out -> Next`, you can join them into a
single `In -> Next` pipe. That join is a **chain**, and you write it left to
right:

```text
source.and_then(transform).and_then(sink)
```

`and_then` is not a separate type you have to import and learn. It is a
method every pipe gets for free, from one blanket sugar trait (`PipeExt`)
over the root `Pipe` — chaining is not part of the `Pipe` contract itself,
but nothing gates on it either, so it reads as if it were. The result of a
chain is just another pipe, so chains nest and keep composing without limit.

→ **Source:** `PipeExt::and_then` in `proxima_primitives::pipe`
(`proxima-primitives/src/pipe/ext.rs`). It returns a two-stage pipe; the type
system checks that the first pipe's `Out` matches the second's `In`, and
bridges the error channel for you.

## 3. Primitive — small reusable pipes built *on* the pipe

A **primitive** is a pipe that wraps another pipe (or several) to add one
reusable behavior. There are three, and each is exactly one idea:

- **filter** — a pass/drop gate in front of an inner pipe. A one-method
  predicate (`decide(&In) -> bool`) says yes or no; a `false` short-circuits
  before the inner pipe ever runs. → [filter](filter.md).
- **fan-out** — one input broadcast to *many* sink pipes. The write-side
  mirror of fan-in: push once, every arm receives it. → [fan-out](fan-out.md).
- **fan-in** — *many* sources merged into one, pulling only the sources that are
  ready right now. Which ready source gets picked is a **strategy**, not part of
  the merge — round-robin is one choice among several. → [fan-in](fan-in.md).

Notice what a primitive is *not*: it is not a new method bolted onto the pipe
trait. `filter`, `fan-out`, and `fan-in` are ordinary pipes that happen to hold
other pipes inside them. That is the whole trick the algebra keeps repeating.

## 4. Pattern — a named behavior composed *from* primitives

A **pattern** is one layer up again: a recognizable behavior assembled out of
primitives, with no new trait machinery at all.

- **gate** — readiness and backpressure. A pipe's `call` is always callable;
  readiness is never a method on the pipe. It is composed on top instead, three
  ways from pieces you already know: shed (a `filter` that rejects while
  closed), wait (a wrapper that goes dormant while closed), and balance (a
  `fan-in` that skips a not-ready source). One small gate seam, three
  consumers, no readiness method anywhere. → [gate](gate.md).

`gate` is only the first pattern. Retry, backoff, auth, IAM, a sentinel
boundary, a write-ahead log, a dead-letter queue, a cron, an event-driven
lambda, a proxy, a gateway, a load-balancer, a whole telemetry stack, and an
ETL pipeline are all patterns composed from the same forms and primitives. The
gallery lays them out as an explicit **scope & sequence** — each rung names what
it builds on, each shown as wiring. → **[the pattern gallery](patterns.md)**.

## Strategies — the dials, not the shapes

One more word completes the vocabulary. A **strategy** is a *policy* you set on a
primitive or pattern — the decision it consults, not the shape it is. Overflow
(block vs drop), fan error policy (all-or-nothing vs best-effort), a backoff
schedule, a circuit's open/half-open/closed — each is a dial you turn without
re-wiring anything. A strategy *with memory* (a token bucket, a circuit's state,
a round-robin cursor) is a small **finite state machine**: pure state and a
transition, no I/O. Backpressure is the best-known strategy — an overflow policy
carried on a bounded pipe, not a pattern composed on top. → the gallery's
[strategies](patterns.md#strategies--the-dials-patterns-turn-often-little-state-machines)
section.

## Two things that ride alongside the layers

These are not a fifth layer. They are orthogonal axes you reach for when the
job demands it.

- **The send tier** — the *same* pipe, upgraded along a ladder of additive
  bounds. A borrowing, no-copy pipe stuck on one thread is the permissive
  **root** (`Pipe`); adding `'static` lets it be owned and erased; adding
  `Send` lets it be dispatched to another core (`SendPipe`); adding `Unpin`
  lets its future be polled exactly where it sits, with no heap and no
  `unsafe` (`UnpinPipe`) — the rung a fan-in needs to hold several sources'
  futures at once. Both additive bounds together is the top rung
  (`UnpinSendPipe`). You climb only as far as your use demands — every rung
  past the root is a cost you opt into, not a tax the root form pays.
  → [send](send.md). **Source:** `Pipe`, `SendPipe`, `UnpinPipe`,
  `UnpinSendPipe` in `proxima_primitives::pipe`
  (`proxima-primitives/src/pipe/primitives.rs`).
- **The signal substrate** — fire-once async completion (end-of-stream, a
  drain going quiet). `Signal` itself lives *below* the algebra, in
  `proxima_core::signal`; the chapter shows the algebra driving it —
  observe → filter the terminal condition → fire → await — with no polling
  loop and no sleep. It is the completion sibling of `gate`: gate answers "is
  it ready *right now*" (repeatable), signal answers "has it finished *yet*"
  (once, for good). → [signal](signal.md).

## The whole algebra, on one hand

- **1 form**, worn four ways (transform · source · sink · observe)
- **1 chain** that joins forms into bigger pipes (`and_then`)
- **3 primitives** built on the pipe (filter · fan-out · fan-in)
- **patterns** composed from primitives — gate first, then the whole
  [gallery](patterns.md) (retry, auth, sentinel, wal, cron, proxy, ETL, …)
- **strategies** — the policy dials patterns turn (often little state machines)
- riding alongside: the **send tier** (cross a core) and the **signal
  substrate** (completion)

That is the entire vocabulary. Everything later in this book — retries,
rate limits, circuit breakers, logging, tracing, a whole HTTP gateway — is
these pieces composed. When a later chapter looks big, follow its **Builds
on** links back down to the pieces here; that decomposition *is* how it is
built, in the code and in the book.

Read on: [transform](transform.md) first.
