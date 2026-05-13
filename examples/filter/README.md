# filter — pass or drop by a predicate

## Builds on

[transform](../transform/README.md) — filter wraps the `Pipe` you learned to write.

## What it demonstrates

`transform` teaches you to write one pipe: an input becomes an output. Filter
is a primitive that sits in front of that pipe and adds a gate. The gate is a
predicate — one question, asked of each incoming item: pass or drop? Only the
items the predicate approves reach the inner pipe as normal. Everything else
is dropped before the inner pipe is ever invoked — no special "filtered"
variant of the inner pipe's input, no trait method the inner pipe has to know
about.

What happens to a dropped item is a separate concern from the predicate: it's
a reject strategy. Drop turns the rejected item into a defined, non-error
outcome — built by the domain payload itself, so the filter never has to know
anything about its payload beyond how to reject it. Error would instead
surface the rejection as a failure of the call. This example wires the drop
strategy — that's the "drop" half of "pass or drop".

The primitive is generic over two things: the inner pipe and the predicate.
The same filter shape can gate an HTTP request, a rate limit (see the `gate`
example's SHED shape), or, here, a plain domain payload with no HTTP in sight.

## Run

```
cargo run --example filter
```

## What you'll see

```
filter: pass or drop by a predicate

order 1: dropped (below threshold)
  ledger processing order 2 ($45.00)
order 2: passed
  ledger processing order 3 ($99.00)
order 3: passed
order 4: dropped (below threshold)
  ledger processing order 5 ($50.00)
order 5: passed

processed: [2, 3, 5]
dropped:   [1, 4]
ledger called 3 times for 5 orders (2 dropped before reaching it)
```

Five orders go through the filter, gated at $20.00. Orders 1 and 4 are below
the threshold: no `ledger processing` line is printed for them, because the
filter never calls the inner pipe — the predicate runs first and short-circuits
straight to the payload's own drop outcome. Orders 2, 3, and 5 clear the
threshold and reach the ledger, which prints its own line before returning its
processed outcome.

The proof is the ledger's own call counter: it lands at exactly 3, matching the
count of processed orders, not 5. If the filter called the inner pipe
regardless of the predicate, that counter would read 5 and the inline assertion
would fail the run — a real regression fails the example, not just the eyeball
check.

## In algebra terms

- filter (a primitive): a predicate decides pass or drop before the inner pipe runs
- reject handling is a strategy: drop (the item becomes a defined non-error outcome) or error
- the primitive is generic over the inner pipe and the predicate, so the same shape composes over any domain payload
- the proof in this example is structural, not cosmetic: the inner pipe's call counter shows it was never invoked for dropped items — the gate short-circuits before the chain continues, it doesn't relabel results afterward
