# pmem — persistent memory

*(builds on: transform)*

Persistent memory is RAM you can address a byte at a time that survives losing
power. There is no block layer and no `write()` syscall between you and it: you
store to a pointer, and the store is durable once you have pushed it out of the
CPU's caches. That last clause is the whole difficulty. A normal store lands in
a cache line, and the CPU is free to write those lines back to the media in any
order it likes — so a power cut mid-update can leave the persistent image
holding some of your new bytes and some of your old ones, interleaved in a way
no ordinary program ever has to think about.

`proxima-storage` ships this as `proxima_storage::pmem`, and unlike the rest of
the frontier it is **not** behind a feature flag — it is unconditional, tier-3
`no_std` with no allocator (`proxima-storage/src/lib.rs:28`). It is pure Rust
over `core::arch` cache-maintenance intrinsics, with no PMDK and no C linked at
all.

The module is two ideas:

- **`persist`** — the ordering primitives (`flush`, `drain`, `persist`) that
  make stores to a borrowed region durable *in the order you meant*.
- **`cow`** — `CowRoot`, a copy-on-write atomic-root-swap state machine that
  turns "update this value durably" into a sequence safe to interrupt at any
  point.

## why copy-on-write, and not a log

`CowRoot` keeps **two slots** and one root word. An update never touches the
slot that is currently live. It writes the new value into the *dead* slot,
persists it, and only then flips a single 8-byte aligned root word — a write
hardware guarantees is all-or-nothing across a power failure (the SNIA/Intel
ADR guarantee). Then it persists the root.

Cut power at any point in that sequence and the root word is either the old
value or the new one, never a mixture. If it is old, the live slot is the old
data and the half-written dead slot is ignored. If it is new, the new data was
already persisted before the flip. Recovery therefore reads **only the root** —
there is no log to replay and no scan to perform. This is the shadow-paging
design LMDB uses for its meta page and ZFS uses for its uberblock; it was
chosen over undo- and redo-logging because its crash-reordering oracle is the
simplest to prove.

## the walkthrough

`proxima-storage/examples/cow_walkthrough.rs` drives `CowRoot` through every
legal transition and prints the persistent byte image after each one, so you
can watch the invariant hold:

```rust
{{#include ../../../proxima-storage/examples/cow_walkthrough.rs}}
```

```bash
cargo run -p proxima-storage --example cow_walkthrough
```

```text
after init             root=0  bytes=[00, 00, 00, 00, 00, 00, 00, 00, aa, aa, aa, aa, aa, aa, aa, aa, 00, 00, 00, 00, 00, 00, 00, 00]

driving the update FSM, NEW = [bb, bb, bb, bb, bb, bb, bb, bb]:
DeadSlotWritten        root=0  bytes=[00, 00, 00, 00, 00, 00, 00, 00, aa, aa, aa, aa, aa, aa, aa, aa, bb, bb, bb, bb, bb, bb, bb, bb]
DeadSlotPersisted      root=0  bytes=[00, 00, 00, 00, 00, 00, 00, 00, aa, aa, aa, aa, aa, aa, aa, aa, bb, bb, bb, bb, bb, bb, bb, bb]
RootFlipped            root=1  bytes=[01, 00, 00, 00, 00, 00, 00, 00, aa, aa, aa, aa, aa, aa, aa, aa, bb, bb, bb, bb, bb, bb, bb, bb]
Committed              root=1  bytes=[01, 00, 00, 00, 00, 00, 00, 00, aa, aa, aa, aa, aa, aa, aa, aa, bb, bb, bb, bb, bb, bb, bb, bb]

recover() returns the live slot: [bb, bb, bb, bb, bb, bb, bb, bb]
recovery is a single root read — no log, no replay.
```

Read the `root=` column down the transcript. It is `0` while the new value is
being written and persisted into the dead slot, and becomes `1` at exactly one
step — `RootFlipped`. Every line before that flip describes a region whose live
value is still the old `aa`s, no matter that the `bb`s are already sitting in
the bytes. That is the point: the `bb`s being *present* is not the same as the
`bb`s being *live*, and only the root word decides which.

## running it on real hardware

The example runs anywhere, including the machine you are reading this on. On a
host with no persistent memory `persist` is a documented no-op — the FSM logic
and the byte image are identical, only the durability barrier is vacuous. To
map a real region, reach for the `dax` feature: the std, Linux-only facade that
`mmap`s a DAX device or file over these same primitives.
