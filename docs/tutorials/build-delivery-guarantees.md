# Build delivery guarantees

**Prerequisites:** [Foundations](./00-foundations.md) — the `Pipe` and its `Err` type.
**You will:** compose at-most-once, at-least-once, and exactly-once delivery over an unreliable wire. None is a built-in mode; each is two choices — how many times the sender tries, and whether the sink remembers what it has seen.
**New concepts (in order):** the give-up `Signal` (from Foundations §10), used here as an attempt-budget boundary · at-most-once (one attempt) · at-least-once (retry-until-ack → duplicates) · exactly-once (retry + dedup ledger).
**Answer key:** [`examples/delivery/main.rs`](../../examples/delivery/main.rs) — `cargo run --example delivery`.

The example frames it: *"at-most-once / at-least-once / exactly-once. None is a built-in mode. Each is a small pipeline from the same two choices: how many times does the sender try, and does the sink remember what it has already seen."*

## The two axes

Everything below is one unreliable-wire `Pipe` plus two knobs:

- **Sender retries** — an `AttemptBudget` bounds how many times to resend; a `Signal` fires when the budget is spent (the give-up boundary) (`delivery/main.rs:148-181`).
- **Sink memory** — a `SinkLedger` either records every landing (`Raw`) or dedups by id (`Dedup`, rejecting repeats) (`delivery/main.rs:68-89`).

The wire returns one of three outcomes per attempt: `Dropped`, `DeliveredAckLost` (the sink got it but the ack was lost), `DeliveredAckOk` (`delivery/main.rs:42-53`).

## 1. At-most-once: send once, no retry

One attempt per message. A first-attempt drop is a permanent loss; nobody watches for the ack (`delivery/main.rs:201-227`):

```rust
let sink = UnreliableSink::new(id, script, SinkLedger::Raw(landed));
block_on_ready(sink.call(()));   // one attempt, whatever happens happens
```

`script` is a fixed list of what the wire does on each send attempt — `Dropped`, `DeliveredAckLost`, or `DeliveredAckOk`, in order — so the demo behaves the same way every run (`delivery/main.rs:183-199`). `UnreliableSink` is named a *sink* because it is the delivery destination, but the shape it is driven with here, `() -> SendOutcome`, is the *source* shape from Foundations §3 (`() → Out`): it takes nothing and reports what the wire did with the item already inside it, rather than accepting new input each call.

`block_on_ready` is not Foundations' `block_on` (§12, `futures::executor::block_on`, which drives a whole app to completion) — it is a tiny one-shot poll helper: it polls the future once against a no-op waker, which works here because every future in this example resolves on its very first poll (`delivery/main.rs:135-146`).

The example asserts at least one message is lost — proof the guarantee is real — and zero duplicates (one attempt can't duplicate).

## 2. At-least-once: retry until ack

Keep sending until the ack is observed, giving up only when the attempt budget is spent — the retry-until-ack loop, its give-up a fired `Signal` (`delivery/main.rs:160-181`):

```rust
fn send_until_acked(sink: &UnreliableSink, budget: &AttemptBudget) -> (SendOutcome, u32) {
    let give_up = Signal::new();
    let mut attempts = 0;
    loop {
        attempts += 1;
        let outcome = block_on_ready(sink.call(()));
        if outcome.is_success() { return (outcome, attempts); }
        if budget.exhausted(attempts) { give_up.fire(); }
        if give_up.is_fired() { return (outcome, attempts); }   // stop at the checkpoint
    }
}
```

Firing and checking are deliberately two separate steps: `give_up.fire()` only marks that the budget is spent — it does not itself exit the loop. `give_up.is_fired()` on the next line is the loop's single exit checkpoint, reached right after either a success or a fire, so every way out of the loop returns through the same place.

Never loses a message — but a `DeliveredAckLost` attempt plus its retry both land at the sink: a duplicate. The example asserts every message landed **and** at least one duplicate occurred (`delivery/main.rs:229-263`). A `Signal` is a fire-once, checkable/awaitable completion — the same shape [record](./build-a-record-replay-harness.md)'s `TerminalSignal` and [`examples/signal`](../../examples/signal) use.

## 3. Exactly-once: retry + dedup

The **same** retry-until-ack loop, plus one more stage: the sink remembers every id it has recorded and rejects a repeat instead of counting it twice (`delivery/main.rs:265-300`):

```rust
let ledger = SinkLedger::Dedup(seen, rejected);         // HashSet of ids seen
// SinkLedger::record: if !seen.insert(id) { rejected += 1 }   // a repeat is rejected, not stored
```

The delivered set equals the sent set (nothing lost, nothing double-counted), and the dedup actually rejected the duplicates the retry loop produced.

## What you built

Same wire, same messages — only the per-stage strategy choice changed the outcome:

- **at-most-once** — one attempt; loss possible, duplicates impossible.
- **at-least-once** — retry until ack (`Signal` give-up); no loss, duplicates possible.
- **exactly-once** — retry + a dedup ledger; no loss, no double-count.

None is a built-in "mode": each is composed from the two axes — retry count and sink memory. Delivery semantics are your choice, made in the open.
