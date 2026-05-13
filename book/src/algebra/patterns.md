# The pattern gallery — a scope & sequence

A **pattern** is a behavior you *build* by wiring the algebra together, then
name. This page is a curriculum: the patterns are laid out in dependency order,
and every rung names exactly what it **builds on** — always things above it, so
you never meet a pattern before its prerequisites. Read top to bottom.

Each rung shows its wiring as pipes connected by data flow, in pure algebra
terms — the four **forms** (transform `In→Out`, source `()→Out`, sink `In→()`,
observe `In→In`), the three **primitives** (`filter`, `fan-out`, `fan-in`), the
**chain** that joins them, and the two patterns the [overview](index.md)
already taught: **gate** (readiness) and **signal** (fire-once completion). The
chapter behind each link is where you see it run.

Read a diagram left to right; `┬`/`├`/`└` is a branch, "loop back" is a feed
edge, and because a chain runs its second pipe only after the first, *flow order
is enforcement order.* Each rung's **Builds on** names are links — follow them
up the sequence or out to the chapter that proves them.

> **This page is algebra, not code.** Every diagram is a *composition* — shapes
> and how they connect — not Rust. The code that proves each one lives behind
> the chapter links, as compiled examples. When a picture and the source
> disagree, the source wins: these pages are the map, the code is the territory.

## Strategies — the dials patterns turn (often little state machines)

Before the sequence, one more word in the vocabulary. Alongside **forms**,
**primitives**, and **patterns** there are **strategies**: a strategy is a
*policy* carried on a primitive or pattern — the decision it consults, not the
shape it is. Same wiring, different behavior, by turning a dial.

A strategy *with memory* is a small **finite state machine**: pure state plus a
transition rule, advanced one step per event, yielding a decision — and no I/O,
so it tiers all the way down to bare metal. A token bucket depletes and refills;
a circuit moves closed → open → half-open → closed; a fair merge advances a
round-robin cursor. That is why the merge and the admission core in the source
are literally described as FSMs.

The dial-set (each is a strategy, and where you turn it):

```text
 overflow / backpressure  on a bounded pipe   block · drop-newest · drop-oldest · sample · coalesce
 fan error policy         on fan-out / gather all-or-nothing · best-effort · ignore-errors
 reject handling          on filter           drop (a stand-in Out) · raise an error
 retryable predicate      on retry            which errors loop, which are fatal
 backoff schedule         on backoff          constant · exponential · jitter        (an FSM over the delay)
 circuit state            on circuit-breaker  closed · open · half-open               (the FSM itself)
 eviction                 on cache            LRU · LFU · TTL
 balance policy           on load-balancer    round-robin · least-loaded · random
 delivery guarantee       composed per stage  at-most-once · at-least-once · exactly-once
```

A strategy never changes the wiring — it changes the decision the wiring reads.
That is why you can swap round-robin for least-loaded in a load-balancer without
touching its `fan-in`, or flip a log sink from block to drop without re-wiring
the fan-out. **backpressure**, met first below, is simply the overflow dial.

---

## Unit 1 — Time & reliability

### 1. clock
**Builds on:** the [*source* form](transform.md).

`clock` is a **source** of *now*, injected instead of read off the wall — so
tests drive time by hand and nothing ever sleeps.

```text
 () ─▶ clock (source) ─▶ the current instant
```

### 2. retry
**Builds on:** [filter](filter.md).

A normal **filter** reads the input and decides pass/drop. Point that same
`decide` at the *error* and loop:

```text
 In ─▶ inner pipe ─┬─ Ok ─────────────────────▶ Out
                   └─ Err ─▶ filter.decide(err)
                              ├─ retryable ─▶ loop back to inner pipe
                              └─ fatal ─────▶ Err
```

### 3. backoff
**Builds on:** [retry](#2-retry), [clock](#1-clock).

The retry loop, with the "loop back" edge made to wait on the clock first —
delay grows constant → exponential → jitter.

```text
 …─ retryable ─▶ wait on clock (growing delay) ─▶ loop back to inner pipe
```

### 4. fallback
**Builds on:** [retry](#2-retry).

The retry loop with one edge repointed: on failure, call a *different* pipe
instead of the same one.

```text
 In ─▶ primary ─┬─ Ok ─▶ Out
                └─ Err ─▶ alternate pipe ─▶ Out
```

### 5. circuit-breaker
**Builds on:** [gate](gate.md), [retry](#2-retry).

A **gate** in front of the inner pipe that trips *open* after the retry loop
reports N fatals in a row — then rejects fast instead of hammering a dead
downstream, until a cooldown re-arms it. Its closed/open/half-open dial is the
FSM strategy named above.

```text
 In ─▶ gate ─┬─ closed (healthy) ─▶ inner pipe ─┬─ Ok
             │                                  └─ Err ─▶ count; N in a row trips the gate open
             └─ open (tripped) ─▶ reject fast
```

### 6. rate-limit
**Builds on:** [gate](gate.md), [clock](#1-clock).

A **gate** whose readiness is a token bucket the **clock** refills.

```text
 In ─▶ gate ─┬─ token available? ─▶ inner pipe ─▶ Out
             └─ empty ─▶ reject
              ▲
      clock ──┘ refills the bucket the gate reads
```

### 7. deadline
**Builds on:** [signal](signal.md), [clock](#1-clock).

The **clock** fires a one-shot **signal** at time T; it races the inner pipe,
first to finish wins.

```text
 In ─▶ inner pipe ─▶ Out
           ╳ cancelled
           ▲
  clock ─▶ signal (fires once at T)
```

---

## Unit 2 — Identity & access

### 8. auth
**Builds on:** [filter](filter.md).

One **filter**: `decide` reads a credential and rejects before the inner pipe
ever runs.

```text
 In ─▶ auth (filter: valid credential?) ─┬─ pass ─▶ inner pipe ─▶ Out
                                         └─ 401 reject
```

### 9. iam
**Builds on:** [auth](#8-auth), [observe](transform.md).

Chain two filters and an observe. Because flow order is enforcement order, each stage
sees only what the last one passed:

```text
 In ─▶ authenticate ─┬─ pass ─▶ authorize ─┬─ pass ─▶ audit ─▶ handler ─▶ Out
      (filter: who?) │        (filter:      │        (observe:
                     └─ 401    may they?)    └─ 403   record it,
                        reject                 reject pass through)
```

---

## Unit 3 — Boundaries, capture & durability

### 10. sentinel
**Builds on:** [filter](filter.md), [signal](signal.md).

An in-band marker value that means *boundary* — end-of-stream, a poison pill, a
batch barrier. A **filter** recognizes the marker; recognizing it triggers a
terminal action (fire a **signal**, flush a buffer, close a sink) while ordinary
items flow past untouched. It is how a stream carries "stop here" in the data
itself, rather than out of band.

```text
 items ─▶ filter.decide(item)
            ├─ ordinary ─▶ pass through ─▶ downstream
            └─ sentinel ─▶ boundary action (fire signal · flush batch · close sink)
```

The [signal](signal.md) chapter is exactly this: the terminal item is the
sentinel, and firing the completion signal is the action. A poison pill that
shuts a worker down, and a barrier that flushes a micro-batch, are the same
shape with a different action.

### 11. record
**Builds on:** [observe](transform.md).

An **observe**: each item passes through unchanged while a copy is written
to a log on the side.

```text
 traffic ─▶ observe (write each item to a log) ─▶ traffic     (log fills on the side)
```

### 12. replay
**Builds on:** [record](#11-record), [source](transform.md).

The log record wrote, read back out as a **source**.

```text
 log ─▶ replay (source: emit each logged item) ─▶ downstream
```

### 13. cache
**Builds on:** [filter](filter.md), [transform](transform.md), [observe](transform.md), [deadline](#7-deadline).

A **filter** asks *hit?* Miss falls through to a **transform** that computes the
value and an **observe** that writes it back. A per-entry **deadline** turns
"stale" into just another miss.

```text
 In ─▶ lookup ─┬─ fresh hit ─────────▶ cached Out
      (filter)  ├─ expired ─┐
                └─ miss ─────┴─▶ fill (transform) ─▶ store (observe, stamp deadline) ─▶ Out
```

### 14. wal (write-ahead log)
**Builds on:** [observe](transform.md), [replay](#12-replay).

Append *before* apply — the write-ahead guarantee is just chain order — and
recover by replay.

```text
 write:    op ─▶ append (observe: durably log, completes BEFORE apply) ─▶ apply ─▶ Out
 recover:  log ─▶ replay (source) ─▶ apply     (the SAME apply pipe, fed from the log)
```

### 15. dead-letter queue
**Builds on:** [retry](#2-retry), [filter](filter.md), [sink](transform.md), [replay](#12-replay).

When processing fails for good (retries exhausted), don't drop the item and
don't block the stream — branch it to a park **sink**. Later, **replay** the
parked items back through the processor.

```text
 In ─▶ process ─┬─ Ok ────────────────────▶ Out
     (with      └─ Err (retries exhausted) ─▶ dead-letter sink   (park, don't lose)
      retry)
 recover:  dead-letter log ─▶ replay (source) ─▶ process   (re-drive the parked items later)
```

### 16. micro-batch
**Builds on:** [gate](gate.md), [deadline](#7-deadline), [sentinel](#10-sentinel), [sink](transform.md), [backpressure strategy](#strategies--the-dials-patterns-turn-often-little-state-machines).

A buffer that flushes on size **or** time **or** a barrier — readiness answers
OR'd together — draining to a **sink** under a chosen overflow policy.

```text
 items ─▶ buffer ─┬─ gate:     count == N?      ─┐
                  ├─ deadline: window elapsed?   ─┼─ any fires ─▶ flush the batch ─▶ sink
                  ├─ sentinel: barrier item?     ─┤
                  └─ none ─▶ keep buffering       ─┘
```

---

## Unit 4 — Observability, and where the algebra deliberately stops

Telemetry is the honest test of "everything is a pipe," because the answer is
**half yes** — and the half that says no is a design decision, not an omission.

### 17. export
**Builds on:** [fan-out](fan-out.md), [transform](transform.md).

The *shipping* half is the algebra. One record is **fanned out** to every
destination at once — console *and* file *and* OTLP, never one only. Each arm is
a **transform**, not a sink: handing telemetry to a collector is a call that
returns a response, so the arm's shape is `record -> response`.

```text
                        ┌─▶ console  (transform: record -> response)
 telemetry ─▶ fan-out ──┼─▶ file
                        └─▶ otlp
```

Each arm's overflow behavior is the [backpressure
strategy](#strategies--the-dials-patterns-turn-often-little-state-machines)
dial — lossless (block) vs lossy (drop) vs sample — chosen per destination
rather than hidden inside an async appender. → [export](../observe/export.md) ·
[logs](../observe/logs.md)

### 18. recording — the part that is *not* a pipe
**Builds on:** nothing. That is the point.

Incrementing a counter is a direct call on a handle. A span is not a pipe.
Nothing composes, nothing is chained, no `In -> Out` anywhere:

```text
 counter.add(1)          ← a method call, not a pipe
 span around a call      ← not a pipe either
```

This is deliberate. A counter bump sits on the hottest path in the program; a
pipe chain per increment would allocate and compose to record a single integer.
The algebra buys composition, and composition is not free — so the recording
edge opts out and the export edge opts in.

Read that as the rule the rest of this page implies but never says: **a pipe is
for things worth composing.** When the answer is "increment this number, now,"
reach for a function. An algebra that claims everything is worth composing is
selling something. → [metrics](../observe/metrics.md) · [traces](../observe/traces.md) · [instrument](../observe/instrument.md)

---

## Unit 5 — Triggers (what starts a pipe)

A pipe does nothing until a **source** calls it. Change the source and the name
changes; the handler need not.

### 19. cron
**Builds on:** [clock](#1-clock), [gate](gate.md), [source](transform.md).

```text
 clock ─▶ gate(schedule: fire now?) ─▶ tick (source) ─▶ handler ─▶ effect
```

### 20. event-driven lambda
**Builds on:** [fan-in](fan-in.md), [transform](transform.md).

```text
 event sources ─▶ fan-in ─▶ handler (transform, stateless per event) ─▶ effect (sink)
```

### 21. service
**Builds on:** the [*source* form](transform.md) (a listener).

```text
 request source (listener) ─▶ handler ─▶ response
```

---

## Unit 6 — Topology (full services, still just pipes)

### 22. proxy
**Builds on:** [transform](transform.md).

A **transform** whose map hands the request to an upstream pipe and returns its
reply.

```text
 request ─▶ forward (transform → upstream pipe) ─▶ response
```

### 23. gateway
**Builds on:** [proxy](#22-proxy), [auth](#8-auth), [filter](filter.md), [gate](gate.md).

Policy chained in front of the forward; chain order is enforcement order.

```text
 request ─▶ auth ─▶ route ─▶ rate-limit ─▶ proxy ─▶ response
           (filter)(filter) (gate)         (forward)
             │401     │404     │429
             └────────┴────────┴─▶ reject
```

### 24. load-balancer
**Builds on:** [proxy](#22-proxy), [fan-in](fan-in.md).

```text
 request ─▶ fan-in over backends ─▶ proxy ─▶ response
           (each backend gated by health;
            pull the ready one, skip the rest)
```

### 25. rest api / crud
**Builds on:** [transform](transform.md), [filter](filter.md).

```text
 request ─▶ route ─┬─ POST /items ─▶ create handler ─▶ response
          (filter)  ├─ GET  /items ─▶ read handler
                    ├─ PUT  /items ─▶ update handler
                    └─ DEL  /items ─▶ delete handler
                   (four transform handlers behind one routing filter)
```

### 26. integration
**Builds on:** [proxy](#22-proxy), [record](#11-record), [replay](#12-replay).

Front a third party, record it once, replay it in tests.

```text
 request ─▶ proxy wrapped in record ─▶ (live once) ─▶ replay ─▶ (offline in CI)
```

### More, built the same way

```text
 aggregator:    In ─▶ fan-out to N backends ─▶ gather replies ─▶ one Out   (backend-for-frontend)
 shadow:        In ─▶ fan-out ─┬─▶ primary (returned)  └─▶ canary (discarded)
 load-shedder:  In ─▶ gate(SHED when overloaded) ─┬─ admit ─▶ inner  └─ reject early
 WAF:           In ─▶ filter(auth) ─▶ filter(method) ─▶ filter(path) ─▶ filter(size) ─▶ inner
 webhook:       event ─▶ fan-out to N endpoints, each arm wrapped in retry+backoff
```

There is no fixed catalog. A pattern is *pick forms → chain them → wrap with
primitives → set the strategies → name it.* When you feel "there's more here,"
you are right: the pattern space is generated, not enumerated.

---

## Capstone — an ETL data pipeline
**Builds on:** everything above.

ETL is the four forms chained, then scaled by the primitives and patterns you
now know — every edge is a rung from earlier in this sequence:

```text
 extract ─▶ clean ─┬─ good ─▶ load
 (source)  (trans) └─ bad ──▶ dead-letter sink       [a filter splits the stream]

 scale each edge with a rung above:
   many sources ───▶ fan-in the extracts into one stream
   many sinks ─────▶ fan-out each record to warehouse + audit
   slow load ──────▶ micro-batch before it, backpressure dial on the buffer
   flaky source ───▶ retry + backoff on extract
   crash-safe ─────▶ wal: append before load, recover by replay
   bounded stage ──▶ deadline around any step
   end of a batch ─▶ sentinel barrier flushes the micro-batch
   done? ──────────▶ signal fires when the source drains
   test offline ───▶ record one real extract, replay it in CI
```

Every edge is a form or a primitive from earlier in this book. Nothing new was
invented to build a warehouse loader — that is what "everything is a pipe, and
big things are small pipes composed" means, shown rather than claimed.

Back to the [overview](index.md).
