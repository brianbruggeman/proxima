# fan_out_affinity

Route one record to **one** of N partitions by a key — Kafka's producer
partitioner, where the same key always lands on the same partition so a whole
customer's (or trace's) stream stays together.

```
cargo run --example fan_out_affinity
```

## Two seams, one line between them

`FanOut` broadcasts one input to *all* arms — the "everyone" distribution.
Affinity is the *other* distribution: route to **one** arm by key. Building it
takes two seams, and the algebra's own rule decides which is which:

- **Keying is a PIPE.** `PartitionKey: Pipe<In = Record, Out = u64>` reads the
  record to derive a routing key. The record passes *through* it — reading the
  payload is legal precisely because it is a pipe. Everything record-shaped a
  router could ever need is funnelled through this one pipe.

- **Choosing the partition is a STRATEGY.** `Distribute::partition(&self, key,
  n) -> usize` sees only the key, never the record, and answers one control
  question. The signature is the proof: `key`, not `Record`. It cannot be
  widened to read the payload without *becoming a pipe instead* — which is the
  whole reason keying lives in its own pipe upstream.

This is the same split `FanInStrategy` draws for fan-in (`fan_in.rs`): the merge
is a pipe, `Select` is a strategy that never sees an item.

## Extend, not add

Three strategies plug into the one seam:

| strategy | uses the key? | state | Kafka analog |
|---|---|---|---|
| `HashAffinity` | yes (`key % n`) | none | key partitioner |
| `RoundRobin` | no | `&self` cursor | keyless round-robin |
| `Sticky` | no | `&self` batch counter | sticky partitioner (2.4+) |

`Sticky` is defined **entirely in this example** — the library has never heard
of it, and needed no change to accept it. That is the point: the strategy trait
is *open for extension* (implement it for weighted, least-loaded, consistent-hash,
whatever) and *closed for modification* (the library ships only the stateless
built-ins). A strategy can never force the library to grow, because the moment
it needs more than `(key, n, &self)` it is reading the record — and that makes
it a pipe, upstream, not a strategy.

## What the run shows

- **HashAffinity** — every order for `ada` lands on one partition, every
  `grace` on one, etc. The run asserts each record sits on `hash(customer) % n`
  and no customer's stream is split.
- **RoundRobin** — the same customers scatter across partitions (the key is
  ignored). The run asserts at least one customer spans more than one partition,
  proving the affinity above was the strategy's doing.
- **Sticky** — records land in fat contiguous runs (fewer, larger batches),
  driven by a strategy the library never defined.

## Notes

- The example routes to a **single** partition (Kafka's model). The general
  fan-out strategy returns a *set* of targets; broadcast is the degenerate "all"
  and route-to-one is the singleton.
- A real fleet uses the *same* hash on every producer so the mapping agrees
  everywhere (Kafka uses murmur2; this uses FNV-1a for a dependency-free demo).
- For elastic partition counts, `key % n` reshuffles every key on a resize;
  **consistent hashing** or **rendezvous (HRW)** move only ~K/N keys and are the
  resize-stable affinity forms — each still a pure `Distribute` strategy
  (`&self` = the ring, key in, no record).

## Read next

[Build a Kafka-style partitioner](../../docs/tutorials/build-a-kafka-style-partitioner.md)
teaches this example end to end — the pipe-vs-strategy line, `FanIn`/`FanOut`,
and a full Kafka/Kinesis/WAL vocabulary mapping.
