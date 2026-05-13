# delivery

**Builds on:** backpressure, cancellation

## The one concept

At-most-once, at-least-once, and exactly-once are not three modes a channel
picks between — there is no `Delivery` type in proxima. Each is what falls
out of two independent per-stage choices: how many times does the sender
retry a send, and does the sink remember what it has already recorded. An
unreliable wire (`UnreliableSink`, scripted per message so the drops and
lost acks are deterministic) sits underneath all three; only the strategy
wrapped around it changes.

- **at-most-once** — one attempt, no retry. If the wire drops it, it's gone.
- **at-least-once** — retry until the sink's ack is observed
  (`send_until_acked`), giving up only when an attempt budget is spent — the
  give-up boundary is a `Signal` fired by that budget check, the same
  fire-then-checkpoint shape `deadline` uses. Never loses a message, but an
  ack can be lost after the item already landed, so the sender resends it —
  a duplicate.
- **exactly-once** — the *same* `send_until_acked` loop as at-least-once,
  with one more stage bolted on: the sink keeps a set of ids it has already
  recorded (`SinkLedger::Dedup`) and rejects a repeat instead of counting it
  again.

## Strategy menu

| guarantee      | per-stage choices                                  | loss / duplication profile                              |
|----------------|-----------------------------------------------------|-----------------------------------------------------------|
| at-most-once   | single attempt, no retry, no ack tracking            | may lose (wire drop); never duplicates (nothing retries)  |
| at-least-once  | retry until ack, bounded by an attempt budget        | never loses; may duplicate (ack lost after landing)       |
| exactly-once   | at-least-once + dedup key at the sink                | never loses; duplicates land raw but are rejected, not counted |

## Run

```sh
cargo run --example delivery
```

What you'll see (deterministic — no sleeps, no real randomness, a fixed
per-message wire script):

```
delivery: at-most-once / at-least-once / exactly-once, composed

at-most-once:  sent 5  delivered 3  lost 2  duplicates 0
at-least-once: sent 5  delivered 5 (raw 7)  lost 0  duplicates 2
exactly-once:  sent 5  delivered 5  lost 0  duplicates rejected 2

same wire, same messages: only the per-stage strategy choice changed the outcome.
```

Each line is backed by an `assert!`/`assert_eq!` in `main.rs` — the printed
line and the proof are the same run, not a paraphrase of it. Same five
messages, same wire script, every time: at-most-once structurally cannot
duplicate (it never retries, so `duplicates == 0` falls straight out of the
code shape, not luck); at-least-once's 7 raw landings against 5 unique ids
is exactly 2 messages that got an ack-lost attempt before their eventual
ack-ok; exactly-once's dedup stage rejects precisely those same 2 landings,
leaving `delivered == sent` — a set equality, not just a count match.

## Why loss and duplication are separate events

It's tempting to think one bit — "did the wire deliver it" — explains
everything. It doesn't. `WireEvent` has three states, not two:

- **`Dropped`** — the item never reached the sink. This is what
  `backpressure`'s `FailMode` already models: something admits or refuses an
  item, and refusal is silent loss.
- **`DeliveredAckLost`** — the item *did* reach the sink; only the
  acknowledgement back to the sender was lost. From the sink's side this is
  indistinguishable from a normal successful delivery — the wire looks fine
  going one direction and unreliable going the other.
- **`DeliveredAckOk`** — the item reached the sink and the sender knows it.

A queue's overflow policy has no way to express the second state — refusing
an item and losing its own confirmation are different failures at different
points in the round trip. That's why this rung scripts a fixed per-message
wire rather than reusing `BoundedQueue`: `Dropped` is a backpressure-shaped
event, but `DeliveredAckLost` isn't a queue decision at all, so forcing it
through `FailMode` would explain less than it hid. See the module doc for
the corresponding note on why `Signal` — not a new primitive — is what
gives at-least-once/exactly-once their give-up boundary.
