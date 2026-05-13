# signal — fully-async fire-once completion

## Builds on

[filter](../filter/README.md) — a predicate recognises the terminal condition;
`signal` fires on it instead of just passing the item through.
[gate](../gate/README.md) — `signal` is the completion sibling of gate's
readiness: gate answers "is it ready right now" (repeatable), signal answers
"has it finished yet" (once, for good).

## What it demonstrates

Async completion as composition, not a bespoke wait primitive. This is the
**sentinel pattern**: a stream carries ordinary items and, eventually, one
terminal (sentinel) item; a filter recognises the sentinel, and only the
sentinel's pass-branch is allowed to fire completion. The shape is
**observe -> filter -> fire -> await**.

- **observe** — an observe, `In = Out` (the same shape `transform`
  taught you), watches every item stream by without changing it.
- **filter** — a filter wraps a terminal-condition predicate; every item
  that fails the predicate is dropped before the fire action is ever
  reached.
- **fire** — the filter's pass-branch is the fire action, reached only by
  the sentinel item. Its whole job is to fire the completion signal.
- **await** — a consumer task, parked on the awaitable since before the
  producer emitted a single item, resolves the instant the signal fires.
  No loop, no timeout, no re-check on a timer.

The anti-poll is the point. The completion signal is a sticky level (fires
once, stays fired), and the awaitable it hands back is a real `Future` —
pending until fired, registering a waker, ready once woken. The example
proves this isn't just claimed: the instrumented await point wraps the
consumer's await and counts every `poll()` call. The count comes out to
exactly **2** — one to park (register the waker, return pending), one to
resume (woken by the fire, return ready) — not the thousands of calls a
`loop { if condition { break } sleep(..) }` busy-poll would rack up. A late
observer (subscribing after the fire) resolves on its very first poll: the
level is sticky, so there is nothing to wait for, only a flag to read.

## Run

```
cargo run --example signal
```

## What you'll see

```
signal: fire-once completion, no polls

consumer: parked on signal.fired() (no poll loop, no timeout)
--- producer: observe -> filter(terminal) -> fire ---
  watch: item 1 passing by (seen 1 so far)
  item 1: observed, not terminal, dropped before the fire pipe
  watch: item 2 passing by (seen 2 so far)
  item 2: observed, not terminal, dropped before the fire pipe
  watch: item 3 passing by (seen 3 so far)
  item 3: observed, not terminal, dropped before the fire pipe
  watch: item 4 passing by (seen 4 so far)
  item 4: terminal -> Signal::fire()
consumer: woken by fire() -> proceeding

--- proof: the consumer never polled in a loop ---
consumer's await point was polled 2 time(s): once to park, once to wake
late observer (subscribes after fire): resolved after 1 poll
```

The consumer prints "parked" *before* the producer prints a single "watch"
line — proof it was genuinely waiting, not something the producer kicked off
afterward. Items 1–3 never reach the fire action (no fire line in the trace
for them); only item 4, the sentinel the terminal-condition predicate
approves, does. The consumer wakes on that single fire, and the poll count
nails down that waking was event-driven, not scheduled polling.

## In algebra terms

- **observe** — an observe watches every item go by; it changes nothing.
- **filter** — a filter recognises the terminal (sentinel) item; every
  other item is dropped before completion can be reached.
- **fire** — the filter's pass-branch, reached only by the sentinel, fires
  a fire-once completion signal. The signal is idempotent and sticky: it
  fires once and stays fired.
- **await** — a consumer awaits the signal directly — no poll loop, no
  sleep, no timer re-check. It parks once and wakes once, and a late
  awaiter (subscribing after the fire) resolves on its very first poll
  because the sticky signal has nothing left to wait for.
