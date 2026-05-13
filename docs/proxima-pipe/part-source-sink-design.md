# Dissolving `Request`/`Response` into `Part` + degenerate Pipes

> **Crate consolidation / path note (2026-07).** `proxima-pipe` (named
> below) no longer exists as a standalone crate — it folded into
> `proxima-primitives`, and `Pipe`/`SendPipe` are now the associated-type
> forms at `proxima-primitives/src/pipe/primitives.rs` rather than the
> `Request`/`Response`-pinned trait this design doc argues against.

Status: DESIGN LOCKED 2026-07-01 (user decision, dialogue-derived). Foundational
— changes the Pipe boundary, not just a payload. Land behind a default-off
feature; do NOT break the production `Request<Bytes>` path until the Part path
is bench-proven end-to-end (principle 16).

## The problem with `Request`

`Request<P>` (proxima-pipe) fuses `method` + `path` + `headers` (+ owned
`RequestContext { path_params: BTreeMap<String,String> }`) into one owned,
complete value. That fusion is:

- **the allocation**: to hand borrowed wire-state to a `Pipe` across an
  await/dispatch boundary, the codec must OWN it — copy every header
  name/value into owned storage because the value outlives the recv buffer.
  Every residual `DC-*-EVENTS-OWN` alloc across h1/h2/h3 is this one act.
- **not RISC**: it bundles fields with different lifetimes/natures (method:
  transient; headers: mostly borrowable; body: a stream, not a value) into one
  type that forces ONE ownership decision — always the worst case (own
  everything) because one field defers.
- **a category error**: a request on the wire is a PROCESS (parts arriving over
  time, borrowed from a reused buffer); `Request` models it as a VALUE.

A cheaper `Request` (offset layout instead of owned Vecs — `Head { raw: Bytes,
layout }`) does NOT fix this: it still HOLDS method+path+all-headers at once.
Holding all parts simultaneously is the aggregate; changing owned→offset only
makes the same aggregate thriftier.

## The model

The distinction that matters is **holding all parts at once vs stepping one at
a time.** The flowing primitive is a borrowed part; you never hold the set:

```rust
pub enum Part<'a> {          // the "instruction set" of a message: enumerated
    Method(&'a [u8]),        // KINDS (a discriminated union, P11), not held
    Path(&'a [u8]),          // INSTANCES. An ISA has ADD/LOAD/STORE and that is
    Header(&'a [u8], &'a [u8]),
    Chunk(&'a [u8]),         // not "fusing"; neither is this.
    End,
}
```

Enumerating the kinds is fine; coexistence is the crime. A source holds exactly
one borrowed `Part` at a time.

### Source and Sink are degenerate Pipes

There is ONE primitive — `Pipe: In → Out` — and:

- **Source** = Pipe with degenerate input:  `() → Part`  (emits parts)
- **Sink**   = Pipe with degenerate output: `Part → ()`  (absorbs parts)

So a connection is pure Pipe composition:

```
codec_in : () → Part      (h1 walks the buffer; h2/h3 = decode_into/FieldSink)
handler  : Part → Part    (transform)
codec_out: Part → ()      (encode to wire)

connection = codec_in ▷ handler ▷ codec_out : () → ()
```

`In`/`Out` for a protocol Pipe are Part-streams. `SendPipe` is already generic
over `In`/`Out`; only the `Pipe` alias pins them to `Request<Bytes>`. We add the
Part-stream boundary as the zero-copy path; `Pipe`-over-`Request` stays as the
opt-in materialized convenience.

### Materialization is opt-in

- `Request`/`Response` = a handler DRAINS a source into an owned aggregate,
  deliberately, only when it wants random access / to mutate path_params / to
  store. `head.to_owned()`. Never on the dispatch path.
- An offset-indexed `Head` (random header lookup without owning bytes) is ALSO
  an opt-in materialization — a derived index over the source, not the primitive.

## Why this is zero-copy AND resolves the lifetime worry

`Part<'a>` borrows the buffer; the SOURCE (the value the Pipe holds) owns/refcounts
the buffer (`Bytes`), so it moves across awaits freely while the borrowed parts
are transient inside the handler. `next(&mut self) -> Option<Part<'_>>` is a
lending step — no GAT on the Pipe trait, the borrow is internal to the source.
1 buffer (h1: refcount-slice of the recv buffer once it is Bytes-backed; h2/h3:
the 1 decompressed HPACK/QPACK block), N borrowed parts, 0 per-header alloc.

`Part::next` IS `decode_into`/`FieldSink` promoted to be the `In` type — the h2/h3
decoders already yield headers one-at-a-time as a visitor; h1 walks lines. We had
the primitive; we stopped throwing it away by re-materializing into `Request`.

## Migration (do NOT big-bang; principle 16)

1. Land `Part`, `PartSource`/`PartSink` (the degenerate-Pipe traits), feature-gated
   default-off. Prove: an adapter shows h3's existing `decode_into` IS a
   `PartSource` (no codec rewrite), and a handler consuming it is 0-alloc
   (stats_alloc) vs the `Request` path. Bench + gate.
2. Provide `Request::from_source(&mut impl PartSource)` (the opt-in drain) so
   existing handlers migrate incrementally, not in a breaking sweep.
3. Per protocol (h3 first — decode_into exists), route the dispatch hot path
   through the source; keep `Request` handlers working via the drain shim.
4. Once all three are on the source path and benched, revisit whether
   `call(In)->Out` (value→value) should itself become Source▷Sink drive — that
   changes the Pipe trait, so it is its own research-rigor + bench decision, NOT
   folded into this.

## Open (research-rigor before step 4)

Is `call(value) -> value` the last aggregate hiding one level up? A pure
Source▷Sink drive may be the RISC-est shape, but it changes the Pipe trait and
every consumer — decide with a bench, not by assertion.

**RESOLVED 2026-07-01** — research-rigor tournament, unanimous 3-judge
panel: `call(In) -> Out` remains the driving boundary; sources/sinks stay
codec-internal leaves. Reopening is gated on three named preconditions
(a production `PartSink`, an async/Reset story for `PartSource`, and the
middleware falsifier bench actually run). Full resolution + rationale:
`docs/proxima-pipe/edges.md`; tournament recorded in `discipline.md` C7.
