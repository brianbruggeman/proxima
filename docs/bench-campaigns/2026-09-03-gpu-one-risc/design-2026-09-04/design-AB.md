# design-AB: the Metal decode path as bytes, pipes, sans-IO FSMs, RISC algebra

Synthesis of design-A and design-B against critique-A and critique-B. Provenance rules:
MEASURED (a counter someone ran), DERIVED (computed here from MEASURED inputs), ASSUMED
(spec sheet or unmeasured). Every `file:line` was opened in this session at
`/Users/brianbruggeman/repos/slot-0/proxima`. No build, no test, no measurement was run.

Sources are named per decision: `A §x` = design-A, `B §y` = design-B, `cA #n` / `cB #n` =
critique findings. Nothing appears here that is in neither A nor B unless a critique finding
forces it; those are marked FORCED-BY.

---

## 0. The shape, the contested decision, and the correction that reorders everything

### 0.1 Shape

- **Algebra (§A).** `BoundOpKind::Loop { stages }` replaces `Elementwise`, `Reduce` and
  `CachedAttention` (from **B §A.3**, chosen over A's `FusedRegion` per **cA #10**). Two map
  extensions are required and both are B's: `IndexPattern::compose` (**B §A.1**) and strided
  extent resolution in `unify_iteration_space` (**B §A.2**). The band is **declared** at the
  `Op` level as a `Domain` of half-spaces built from the **shipped `AxisIndex`** whose offset
  widens to `Offset::{Static, Symbol}` (A's `Domain`/`SymbolId` idea, but reusing `AxisIndex`
  per **cA #7**, and declared-not-derived per **cB #19**). `Op` and `ScalarOp` do not grow.
- **Orchestration (§B).** The FSM is authoritative and there is exactly **one** encoding of a
  step (**cB #20**). Two `Pipe` impls exist on the device edge (`Submit`, `Complete`) and one
  at the session face (`Session: Pipe<In = StepInputs, Out = StepOutcome>`), which is the shape
  `LoadedModel::call` (`proxima-model-interop/src/generate.rs:1533`) already has and the one
  Pipe on the path today. Every plan-time "pipe" in A §B.4 is deleted as a relocation
  (**cA #5**). `plan()` becomes pure; the eleven entry points at `omega/src/metal.rs:468, 519,
  535, 978, 1177, 1192, 1210, 1409, 1484, 1509, 1613` (verified by
  `grep -n '^pub fn \(plan\|execute\)'`, eleven, not nine or ten — **cA #14**, **cB #26**)
  collapse to `plan` + one `MetalDriver`.
- **Generic (§C).** Production decode from `ProgramSpec` data via `stack`/`extend` (**B §C.1**),
  typed roots and layer inputs (**A §C.2** merged with **B §C.2**), `cached_len` as a
  **symbol, not a leaf and not an integer buffer** (A's `SymbolId`, which is what disposes of
  **cB #10**).
- **Bytes (§D).** Redone below from the ROW 287 in-buffer census. The answer is negative for
  3.5 ms under the measured conditions, and the reachable floor is stated.

### 0.2 The contested decision, decided

**A said `FusedRegion { ops: Vec<BoundOp> }`; B said `Loop { stages: SmallVec<[Stage; N]> }`.
Taken: B.** `bind.rs:160-162` states the invariant verbatim, opened this session: `StepArg` is
"a plain index into a side table (`ComposedBody::steps`), never a `Box<dyn>` recursive tree."
`FusedRegion` makes `BoundOpKind` recursive through `BoundOp`, which is the shape that comment
rejects, and it carries `extents`/`domain` duplicating each member's own with the invariant in
prose (**cA #10**). `Loop`'s stages are the same flat side-table discipline one level up:
`StepArg::Stage(u16)` is backwards-only exactly as `StepArg::Step(u16)` is today.

**The cost of taking B here is real and is stated, because B never stated it** (**cB #17**):
`Loop` is RISC at the `Op` face (the vocabulary `op.rs:52-56` promises stays closed does not
grow) and it is a multi-stage traversal interpreter at the `BoundOp` face, which five emitters
must implement. §A.8 prices that migration and §E sequences it so that no emitter is ported
blind.

### 0.3 The correction that reorders everything: 5.96 ms of the token is not GPU execution

Both designs decomposed the token from the **serialized** per-op profile (ROW 281/282), which
`audit-2026-09-04.md:22` marks superseded. Using the ROW 287 in-buffer numbers the brief
requires (`dispatch-census.md:25-29`), all MEASURED:

| term | ops | ms | CoV |
|---|---:|---:|---|
| whole token, wall (ROW 286) | — | 33.00 | — |
| ALL GPU ops, one command buffer | 616 | 27.04 | 1.0% |
| MATVEC | 225 | 23.24 | 0.7% |
| NOT-MATVEC (attn+coop+elem) | 391 | 4.05 | 1.5% |
| **wall − ALL: host, submit, readback, sample** | — | **5.96** | DERIVED |

Three consequences neither design drew:

1. **The host residual is 18% of the token and is the ceiling on §B.** Even at zero bytes and
   zero dispatches, today's token cannot go below 5.96 ms. 3.5 ms is unreachable until this
   term is under ~0.5 ms. §B's zero-allocation budget, the one driver and the pure `plan()` are
   therefore not "margin" (as B §D.7 called them) — they are the first blocking term.
2. **Achieved weight bandwidth is 173.6 GB/s, not 161.** 4.034 GB of weights through 225
   matvecs in 23.24 ms (DERIVED from the census's MEASURED ms and §D.1's byte table). B's 161
   came from the superseded instrument (**cB #2**) and was internally inconsistent with B's own
   `dispatch_ns` (**cB #3**). The 2.3x gap to the 400 GB/s spec sheet remains unexplained and
   is worth more than every codec lever combined (§D.8).
3. **Non-matvec GPU time is 4.05 ms, not 6.34.** So the census ceiling of 616 → 264 dispatches
   (`dispatch-census.md:49`) is worth at most 4.05 ms, and only if per-dispatch cost is the
   whole of it. Elementwise is 290 ops in 1.76 ms = 6.1 µs each (MEASURED); attention is 32 ops
   at 3.0-3.6 ms with CoV 9.9% — the noisiest term in the census and the one A §E proposed to
   delete first (**cA #2**).

---

# D. Bytes — primary section, redone from the census

## D.1 The stream, from the census bytes

Inputs are the brief's census figures (FFN 3.171 GB, attn 0.755, output.weight 0.108, KV 0.134
at this context; `design-task.md:90-107`), which reconcile to 4.168 GB against the brief's
stated 4.169 GB. Parameter counts for openchat-3.5 (Mistral-7B geometry) reproduce them:

| family | params | codec | bits/w | bytes/token | provenance |
|---|---:|---|---:|---:|---|
| FFN gate+up+down ×32 | 5,637,144,576 | Q4_K | 4.5 | 3,170.9 MB | census; DERIVED reproduces |
| attn q+k+v+o ×32 | 1,342,177,280 | Q4_K | 4.5 | 755.0 MB | census; DERIVED reproduces |
| `output.weight` | 131,080,192 | Q6_K | 6.5625 | 107.5 MB | `byte-levers-probe.md:32` |
| KV read @ this ctx | — | f32 | 32 | 134.2 MB | census |
| `token_embd` | 131,080,192 | Q4_K | — | one row, 16 KB | not streamed |
| **total** | | | | **4,167.6 MB** | DERIVED |

A's 125 MB residual does not exist against these inputs and is not carried; the two designs
differed only because A priced KV at 34 positions and the census prices it at this context.
**cA #1c** (scaling an unexplained residual by a codec factor) is therefore moot — there is no
residual to scale. What remains open and is CARD D0's job: per-tensor codecs read from the gguf
tensor table (`proxima-gguf/src/types.rs:109-133`), because `omega/Cargo.toml` records Q5_K on
`blk.{0..3}.ffn_down.weight`, which moves 29.4 MB of the FFN row.

Time constants used below, all from §0.3: **173.6 GB/s MEASURED achieved**, 300 GB/s ASSUMED
sustained ceiling, 400 GB/s ASSUMED spec. Host residual 5.96 ms MEASURED today; 0.5 ms target
after §B. Non-matvec 4.05 ms MEASURED today; 0.7 ms DERIVED at 264 dispatches.

## D.2 L1 multi-token per pass — lossless, gated on a kernel that does not exist

Bytes: pass bytes ÷ k′ where k′ is accepted tokens per pass. Both designs agree the algebra
needs nothing (**A §D.2**, **B §D.1**): symbols are `[new_count, kv_bound_extent]`
(`generate.rs:2394`), the plan-cache key is `(new_count, bucket)` (`generate.rs:1248, 1323`),
and the `s` axis is symbolic through every builder.

**Precondition, unchanged from both designs:** the packed-row kernel batches 4 *output* rows
over one activation and has no s-axis fold (`msl.rs:3171-3200`; `PackedRowBlock { weight,
other, reduce_dim, codec }` at `msl.rs:3184-3189`, opened). At `s = k` it re-streams weights k
times and the lever returns exactly zero. **CARD D1a lands the `nr1` fold first and its gate is
weight bytes flat in k**, not a byte projection.

k′ carries a kill criterion here, which B did not give it (**cB #1**): the acceptance rate of
an n-gram drafter on this model is an unknown metric on held-out data — a discovery loop by the
same definition B applied to elision. **Pre-registered: k′ ≥ 1.8 at k = 4 on a 200-prompt
held-out set stratified by class (code / prose / chat), reported as a histogram per class, not
a mean. Kill at k′ < 1.5** (below which the extra per-pass compute is not repaid). Nothing in
the byte table is credited to L1 before CARD D2 reports it.

A's `k′ = 2.18` label ambiguity (**cA #1d**) is resolved by definition here: **k′ = accepted
tokens per pass including the free verified token**, so a 4-wide pass at α = 0.6 gives
`1 + α + α² + α³ = 2.18` ASSUMED, and CARD D2 measures it against that definition.

Verification is host-side O(k) integer work; there is no rollback machinery, because rejected
KV rows are overwritten by the next pass (both designs converge here; `byte-levers-probe.md:37-38`).

## D.3 L2 FFN row elision — the group size is forced by the codec, and ffn_down is why

**cB #4 is correct and it changes the design, not just the number.** `ffn_down` stores the
14336 hidden axis as the *contraction* axis (`reduce_dim` in `PackedRowBlock`,
`msl.rs:3184-3189`), so eliding hidden units is an intra-row skip, not a row-count reduction.
At unstructured selection every Q4_K superblock still contains a selected column and down's
bytes do not fall at all — recomputing B's Route 2 with down at full bytes puts the target out
of reach on both routes, which is why this is a design change and not a footnote.

**The fix, FORCED-BY cB #4 and cA #9:** the selection granularity is a **whole Q4_K superblock
on the contraction axis**. `Q4K_BLOCK_ELEMENTS = 256` (`omega/src/msl.rs:444`, cited by
**cA #9**), so the group size is 256 hidden units, not A's 128. Then:

- `gate`/`up`: the group is 256 whole output rows — a row-axis `Lookup`, bytes fall with `d`.
- `down`: the group is 256 contiguous contraction elements = exactly one Q4_K superblock per
  output row — a whole-block skip, bytes fall with `d`.
- `G = 14336/256 = 56` groups. Selection is over 56, never over 14336, which is the
  computed-address form the sparse memory admits (A §D.3's argument, at a granularity that is
  now true for all three matrices rather than for two of them).

**cA #9's other three defects are answered, not deflected:**

- *G is not constant.* Correct. `G = d_ff/256` grows with the model. The claim is downgraded to
  what is defensible: the channel is `O(G)` with `G` two orders below `d_ff`, the selector cost
  is counted in the arithmetic (below) rather than assumed away, and CARD D3b's gate is
  ns/element against the dense arm, so the fanout argument is settled by measurement and not by
  taxonomy.
- *The router weights do not exist.* They do not, and **CARD D3-calib produces them**: fit
  group scores by least squares against the observed per-group `Σ|gate·up|` L1 mass over N
  = 50k activation samples captured from the dense model on the calibration split, stored as a
  sidecar gguf tensor (`blk.{i}.ffn_router.weight`, 4096×56 at Q4_K = 0.13 MB/layer,
  4.1 MB/token total). This is a card with a cost, an artifact and a gate, not an input the
  design assumes.
- *`d = 0.35` carries no provenance.* It is **ASSUMED** and is CARD D3b's swept variable
  (`d ∈ {0.25, 0.35, 0.5, 0.7}`); the byte table below shows the result at each.

**The union effect is real and is priced (FORCED-BY cA #1a).** A k-token pass selects the union
of k rows' group sets. Under independence at d = 0.35 and k = 4 the union is
`1-(1-0.35)⁴ = 0.82`; consecutive tokens are correlated, so the true figure is between 0.35 and
0.82 and **nobody has measured it**. It is a gate, not an assumption: **CARD D3b measures union
density at k ∈ {1,2,4} and the byte table is recomputed from the measurement. Kill for the
L1×L2 composition if union density at k=4 exceeds 0.6.** The table below is given at
`d_union = 0.6`, the kill boundary — the pessimistic end of the admissible range, so the
conclusion does not depend on the unmeasured number being favourable.

**Selector cost, which B omitted entirely (FORCED-BY cB #15).** With `G = 56` the selector per
layer is: one router matvec (4096×56), one `Loop` whose stages are threshold + `Keep::Scan`
prefix + bounded compaction (one dispatch under §A's R2/R3, not five), and nothing else = **2
dispatches/layer = 64/token**. At the census's MEASURED 6.1 µs/elementwise dispatch that is
**0.39 ms/token**, and it is carried in §D.7's time column. B's five-to-six serially dependent
dispatches per layer (1.6-2.0 ms) is what the `Loop` stage fusion removes; if R2/R3 do not
fuse them, the selector costs 1.2 ms and CARD D3b's gate fails on time, which is the correct
coupling.

`Keep::Scan` has no cooperative Metal path today (`reduce_is_cooperative` requires
`Keep::Reduce`, `msl.rs:1041-1042`, and there are **three** gather-exclusion sites, not one:
`msl.rs:1047`, `msl.rs:1218`, and `classify_packed_row_block` at `msl.rs:1389` — verified by
`grep -n gather_count omega/src/msl.rs`; **cA #15**, **cB #12**). CARD D3a lifts all three and
its gate separates them: the 65 cooperative reduces/token must not regress, measured
independently of the matvec arm.

## D.4 L3 lower-bit codecs, L4 output.weight, L5 KV

Unchanged in substance from both designs; the corrections are to the gates.

- **L3 Q3_K on FFN only** (attention stays Q4_K): 3,170.9 → 2,422.6 MB (×0.764). Q2_K/Q3_K
  already parse (`proxima-gguf/src/types.rs:109-133, 245-281`) and are rejected at bind
  (`proxima-model-interop/src/bind.rs:70-73`). **FORCED-BY cB #25:** the checkpoint is Q4_K_S,
  so this is Q4_K → Q3_K double quantization, strictly worse than a native Q3_K from
  full-precision weights, and every published Q3_K quality expectation comes from the latter.
  The gate is therefore stated against the *measured* dense model and not against literature:
  EM-at-64 ≥ 0.97 and held-out perplexity delta ≤ 0.5%. If the double-quantization loss fails
  it, the fallback is a native Q3_K conversion from the f16 checkpoint, which is a different
  card and a different download.
- **L4a `output.weight` Q6_K → Q4_K**: 107.5 → 73.7 MB, codec selection only. Gate EM ≥ 0.98
  (tighter: this tensor sits on the sampled token).
- **L4b candidate-set gather**: rides D3a's row-gather emitter over the vocab axis
  (`check_gather_extent` passes at 32002, `shape.rs:297-310`). **FORCED-BY cB #24**: it gets a
  card (D5b), a construction (the same bounded compaction as D3b, over a candidate set built
  from the previous token's continuations plus a fixed high-frequency floor — never a search
  over 32002), and its own row in the table. At c = 4096: 73.7 → 9.4 MB. Gate: candidate-miss
  rate ≤ 0.1%, else revert to L4a alone.
- **L5 f16 KV**: 134.2 → 67.1 MB at this context, 537 → 269 MB at 2048, 2,147 → 1,074 MB at
  8192. `PackedCodec::Float16` is a codec slot with a stated reason (`metal.rs:441-444`);
  `BoundOp::dtype` is already per-node (`bind.rs:206-212`). Gate EM ≥ 0.99.
- **Q8_0 KV is dropped from this design.** It needs `ScalarOp::Round` against a set `op.rs:52-56`
  declares closed, and §D.7 does not need it (**cA #20**). Naming it and not taking it is the
  answer to the closed-set question.

## D.5 The selector is not a pipe; the drafter is

Kept verbatim from **A §D.3** because it is the sharpest result either design produced, and it
survives both critiques: the drafter runs once per pass on the host touching no weights, so it
is `Pipe<In = TokenWindow, Out = Draft>`; the elision selector runs once per layer and as a
host pipe would mean 32 device crossings per token on a path whose budget is one readback —
which is the shape the landed probe already has and which measured dispatch-bound (ROW 180/181,
0.20-0.29 ns/element vs 0.057 DRAM). Same question, opposite answers, discriminator stated:
**how many times per pass it crosses the device edge.**

## D.6 The arithmetic

MB/token at the census context. `d_union = 0.6` is the kill boundary from §D.3; `k′ = 2.18` is
ASSUMED and gated by CARD D2. Router 4.1 MB/token is carried in the L2 row.

| stage | FFN | attn | output | KV | pass | /token |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 3,170.9 | 755.0 | 107.5 | 134.2 | 4,167.6 | 4,167.6 |
| + L5 f16 KV | 3,170.9 | 755.0 | 107.5 | 67.1 | 4,100.5 | 4,100.5 |
| + L4a output Q4_K | 3,170.9 | 755.0 | 73.7 | 67.1 | 4,066.7 | 4,066.7 |
| + L3 Q3_K FFN | 2,422.6 | 755.0 | 73.7 | 67.1 | 3,318.4 | 3,318.4 |
| + L4b candidate gather | 2,422.6 | 755.0 | 9.4 | 67.1 | 3,254.1 | 3,254.1 |
| + L2 elision, d_union 0.6 (+router 4.1) | 1,457.7 | 755.0 | 9.4 | 67.1 | 2,289.2 | 2,289.2 |
| **+ L1, k′ = 2.18** | | | | | 2,289.2 | **1,050.1** |

Routes and times. Time = bytes/bandwidth + non-matvec dispatch + host residual.

| route | MB/token | @173.6 (M) | @300 (A) | @400 (A) | + 0.7 + 0.5 ms | verdict vs 3.5 |
|---|---:|---:|---:|---:|---|---|
| today | 4,167.6 | 24.01 | 13.89 | 10.42 | 24.01 + 4.05 + 5.96 = **33.0 MEASURED** | — |
| lossless only (L1) | 1,911.7 | 11.01 | 6.37 | 4.78 | 12.2 / 7.6 / 6.0 | misses |
| Route 1 (no elision: L1+L3+L4+L5) | 1,522.2 | 8.77 | 5.07 | 3.81 | 9.97 / 6.27 / 5.01 | misses |
| Route 2 (all levers, d_union 0.6) | 1,050.1 | 6.05 | 3.50 | 2.63 | 7.25 / 4.70 / 3.83 | misses at every measured bandwidth |
| Route 2 at d_union 0.35 (favourable) | 748.6 | 4.31 | 2.50 | 1.87 | 5.51 / 3.70 / 3.07 | meets only @400 ASSUMED |

**The bytes answer, stated plainly.**

**3.5 ms/token is not reachable under the measured conditions.** Under the only MEASURED
bandwidth on this box (173.6 GB/s achieved on the weight stream, DERIVED from
`dispatch-census.md:27`'s 23.24 ms and §D.1's 4.034 GB of weights), the full lever product
lands at **7.25 ms/token**, 2.1x above target. Reaching 3.5 ms requires **all four** of:
(i) achieved bandwidth ≥ ~300 GB/s, i.e. the unexplained 2.3x gap closed — §D.8, not a lever
in this design; (ii) union density at k = 4 measuring at the favourable end (0.35-0.45), not
the kill boundary; (iii) the host residual cut from 5.96 to ≤0.5 ms, which is §B in full; and
(iv) k′ ≥ 2.18 measured. Any one missing and the target is missed.

**The reachable floor, if none of the above lands beyond the measured facts:** Route 1
(lossless L1 + codec levers, no elision, no bandwidth fix) at 173.6 GB/s with §B's host work
= **~10.0 ms/token = 1.75x llama.cpp's 17.45**. Route 2 with the elision lever clearing its
quality gate = **~7.25 ms = 2.4x llama**. The owner's 5x is a *bandwidth* question before it is
a lever question.

**Sensitivity, recomputed from the table above** (each row re-derives; **cA #1b** was that A's
did not):

| omitted | pass | /token | @300 + 1.2 ms |
|---|---:|---:|---:|
| none | 2,289.2 | 1,050.1 | 4.70 |
| without L1 | 2,289.2 | 2,289.2 | 8.83 |
| without L2 | 3,254.1 | 1,492.7 | 6.18 |
| without L3 | 3,037.5 | 1,393.3 | 5.85 |
| without L4 | 2,387.3 | 1,095.1 | 4.85 |
| without L5 | 2,356.3 | 1,081.0 | 4.80 |
| `nr1` fold not landed | 4,167.6×k re-streamed | 4,167.6 | 15.09 |

L1 and L2 are the two levers whose omission moves the result by more than 40%; L3/L4/L5 are
margin. This reproduces B's ordering conclusion from corrected inputs.

## D.7 What is not a byte lever

§A's fusion and §B's dispatch collapse. Epilogue fusion removes `residual1`/`x_next`
materialization = 2 × 4096 × 4 B written and re-read per layer = 2.1 MB/token DERIVED, 0.05% of
the stream. They buy the 4.05 ms non-matvec term and (through §B) the 5.96 ms host term. Both
designs said this; it is repeated because §0.3 changes the size of what they buy.

## D.8 The term nobody can explain, and it is the largest one

173.6 GB/s MEASURED against 400 GB/s ASSUMED spec. **No mechanism is offered here.** Closing it
is worth 4.168 GB × (1/173.6 − 1/300) = **10.1 ms/token at zero quality risk**, more than every
lossy lever in §D combined. **CARD D0 is first in §E** and is instrumentation only: a pure-read
kernel with the identical access pattern (same quant-block stride, same threadgroup width, no
arithmetic) at three problem sizes, against a memcpy-shaped control that should saturate, CoV
over 5 runs. If the control does not saturate, the 400 GB/s figure is wrong for this access
shape and every projection above is re-priced against the measured ceiling instead.

---

# A. Attention in the RISC algebra

## A.1 What attention is, in `Op`/`ScalarOp`/`IndexMap`

Both designs agree and both are right: the seven-line chain (**B §A.0**, **A §A.1**) uses only
shipped `ScalarOp`s and shipped index patterns; `specs/causal_attention.toml` and
`specs/gqa_attention.toml` evaluate it. Attention needs no new algebra to be **expressed**. It
needs (a) a way to say those lines share one traversal, and (b) a way to say an iteration point
is out of range.

## A.2 Extension 1: `IndexPattern::compose` (from B §A.1)

```rust
// proxima-tensor/src/map.rs — pure, alloc tier, no config feature.
/// Substitutes `outer` into `inner`: given `inner` addressing an operand from an
/// iteration space I2 and `outer` addressing I2 from I1, returns the pattern
/// addressing that operand directly from I1. Coefficients multiply, offsets
/// accumulate, terms with equal `axis` merge. `None` when `inner` names an axis
/// `outer` does not project.
///
/// Composes: `AxisIndex` (map.rs:60-66) and `IndexPattern`. Exists because fusion
/// today is gated on `is_identity_projection` (bind.rs:1156-1164) — literally "the
/// map is the identity" — which is why RoPE's 2i/2i+1 and GQA's `h = group*u + g`
/// decline to fuse (dispatch-census.md:11).
#[must_use]
pub fn compose(outer: &IndexPattern, inner: &IndexPattern) -> Option<IndexPattern>;
```

Taken from **B §A.1** because A has no answer for RoPE/GQA fusion at all: A §A.6 asserts R2
"replaces identity with affine" without naming the operation that pushes a non-identity map
through a held node's own map. That operation is substitution and there is nothing weaker.
Two binary questions, both answered in B and verified here: `IndexPattern` has no composition
operator, and `compose_operand` (`bind.rs:1177+`) recurses only through maps already proven to
be the identity. New capability: fuse a strided or offset operand read — 128 RoPE dispatches
per token, measurable.

## A.3 Extension 2: strided extent resolution (from B §A.2)

`unify_iteration_space` (`shape.rs:224-244`) binds an axis extent only from a single
`coeff == 1, offset == 0` term; after A.2 an axis can appear only under a strided map and
inference reports `UnconstrainedDim` (`shape.rs:250-253`). A second pass, run only for slots no
unit projection pinned: single term `c·i + k` against operand extent E, `c > 0` ⇒
`N = (E-1-k)/c + 1`. Unit projection always wins, so existing programs bind byte-identically —
and CARD 0's gate is exactly that: `bind` over `specs/mistral_layer.toml` at real dims compares
`PartialEq` node-for-node to pre-card.

## A.4 Extension 3: the band is DECLARED, on the shipped affine type

This is where A and B conflict and both critiques land. **B derived the band by peephole-matching
`Select(Greater(A,B), NEG_INF, x)`** — which reproduces the failure mode audit item 1 names, one
level down (**cB #19**: a multiplicative 0/1 mask, an `Equal`-based mask, or a mask split across
two nodes derives nothing). **A declared it as `Domain`/`HalfSpace`** — but minted a second
affine-expression type field-for-field identical to the shipped `AxisIndex` (**cA #7**).

**Taken: declared (A's direction), on `AxisIndex` (cA #7's correction), with B's
`Offset::Symbol` supplying the live length.**

```rust
// proxima-tensor/src/map.rs

/// A linear offset that is either known at spec time or bound per call from the
/// same `symbols: &[u64]` slice `shape::infer` already takes. `Static` is what
/// every shipped `AxisIndex` carries today, so this widening is source- and
/// wire-compatible with `offset: i32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "config", serde(untagged))]
pub enum Offset {
    #[default]
    Zero,
    Static(i32),
    /// Index into the caller's `symbols` slice — the same index
    /// `Extent::Symbolic(u16)` carries (op.rs:45-48).
    Symbol(SymbolId),
}

/// Named so a bare `u16` cannot be passed where a symbol slot is expected (P11).
/// `Extent::Symbolic(u16)` changes to `Extent::Symbolic(SymbolId)` in the SAME
/// card, so the workspace never carries two spellings of one index (cA #21).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
pub struct SymbolId(pub u16);

/// `sum(terms) + offset >= 0` over the iteration space. Reuses `AxisIndex`
/// (map.rs:60-66) verbatim — a half-space IS an affine expression plus a sign
/// convention, and minting a second one is what cA #7 caught.
pub type HalfSpace = AxisIndex;

/// The iteration space's shape beyond its rectangular extents. Empty = the full
/// box = every op this crate has built to date.
///
/// SEMANTICS, and this is the decision: a point failing any constraint is NOT
/// evaluated — not read, not accumulated, not written. A `Domain` is a loop
/// bound an executor must honour, not a hint it may decline. That is what makes
/// KV bytes a function of the program instead of a function of whether an
/// analysis fired (cB #5).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
pub struct Domain {
    pub constraints: SmallVec<[HalfSpace; MAX_INLINE_CONSTRAINTS]>,
}
```

`AxisIndex::offset` widens from `i32` to `Offset`. That is one type in the crate, not two, and
it removes A's immediate drift (**cA #7**): an operand *address* can now carry a symbolic
offset too, which is what a KV write at `cached_len + s` needs and what A's split forbade.

**`Domain` attaches to the op whose iteration space it constrains** (`Op::Reduce`,
`Op::Elementwise`), with `#[serde(default)]` on the field — **FORCED-BY cA #11**, which caught
that adding a field to `Reduce` (`op.rs:153-164`, eight fields, no `#[serde(default)]` anywhere
in `op.rs`) breaks deserialization of all eleven files in `proxima-tensor/specs/`, including the
one CARD A1's own gate requires to parse. The card carries the struct-literal migration (25
sites in `bind.rs`, 7 in `shape.rs`, plus the model builders) explicitly.

**Why declared beats derived, in one line each:** declared cannot silently fail to fire on an
unrecognized mask spelling (**cB #19**); declared makes KV bytes a property of the program, so
§D's arithmetic does not depend on an analysis (**cB #5**); declared is what a TOML author
writes, which is §C's whole point. **What we keep from B's "hint" framing:** the masking
`Select` stays in the program until CARD A5 deletes it, and the property test compares the two
spellings, so a wrong domain is caught by parity rather than by inspection.

**Minimality, with the candidate A's proof missed (cA #8).** `Lookup::extent` already expresses
out-of-range *for a fetched index* (`bind.rs:127-129`), and it is the wrong instrument: it
bounds an address, not an iteration point, and it acts per operand — the band constrains the
key read, the value read and the accumulation, three places that must agree. That is the
compensator pattern, and it is why `IndexMap::Banded` was also rejected. Wider `Multiply` arity
(`op.rs:95-103`) does not touch the iteration space: still 2t trips. A tuple-valued reduce is
strictly larger and unnecessary (both designs agree).

## A.5 `BoundOpKind::Loop` (from B §A.3, with A's symbol instead of B's operand)

```rust
// proxima-tensor/src/bind.rs
pub use crate::sized::MAX_INLINE_STAGES;

/// One accumulation stage of a `Loop`. `fold: None` is a pure map over the axes
/// live at that point; `fold: Some(..)` folds `reduced_axes` away and its result
/// is addressable by later stages.
///
/// Composes `ComposedBody` (bind.rs:176-184) — the same flat side table, the same
/// backwards-only rule, one granularity up. NOT a recursive tree of `BoundOp`:
/// bind.rs:160-162 rejects that shape by name (cA #10).
#[derive(Debug, Clone, PartialEq)]
pub struct Stage {
    pub reduced_axes: SmallVec<[u16; MAX_INLINE_RANK]>,
    pub body: ComposedBody,
    pub fold: Option<Fold>,
    /// Resolved from the op's `Domain`, in this stage's iteration-axis space.
    /// Empty = full traversal. Binding, not advisory (§A.4).
    pub domain: BoundDomain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fold { pub op: ScalarOp, pub init: ReduceInit, pub keep: Keep }

/// `sum(coefficients[axis] * iter[axis]) + offset >= 0`, iteration-axis space.
/// The resolved counterpart of `map::HalfSpace`, exactly as `Layout` is the
/// resolved counterpart of `IndexPattern` (bind.rs:93-97).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundHalfSpace {
    pub coefficients: SmallVec<[i64; MAX_INLINE_RANK]>,
    /// `Symbol` renders as a uniform-block read, never a `constexpr` — one
    /// compiled kernel serves every cache length, which is what `entry_name`'s
    /// "dyn" marker buys today (msl.rs:2535-2541) without a ninth operand.
    pub offset: BoundOffset,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundOffset { Static(i64), Symbol(SymbolId) }

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BoundDomain { pub constraints: SmallVec<[BoundHalfSpace; MAX_INLINE_CONSTRAINTS]> }

/// Backwards-only, same rule as `StepArg::{Operand, Step}` (bind.rs:164-168).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepArg { Operand(u16), Step(u16), Stage(u16) }

pub enum BoundOpKind {
    Loop {
        stages: SmallVec<[Stage; MAX_INLINE_STAGES]>,
        operands: BoundOperands,
        output_axes: SmallVec<[u16; MAX_INLINE_RANK]>,
        out_layout: Layout,
        out_scatter: Option<Lookup>,
    },
    Iota,
    Constant { value: f32 },
}
```

`Elementwise` = one stage, `fold: None`. `Reduce` = one stage with a `Fold`. `CachedAttention`
is deleted. Net 5 kinds → 3.

**The live cache length is `BoundOffset::Symbol`, not B's `BandBound::Dynamic { slot }`.** This
is A's shape and it is what disposes of **cB #10**: B's dynamic bound reads a rank-0 *operand*,
which must be an integer buffer, and `map.rs:99-107` (opened) states that no backend plumbs
integer buffers — "Lifting the ceiling means adding real integer buffers, not raising a
constant." A symbol is a host-supplied `&[u64]` entry written into the dispatch's uniform
block. No integer buffer, no ninth operand, no `operands.len() == 8|9` discriminator
(`bind.rs:237`, `cpu.rs:4847`, `msl.rs:2030`, `msl.rs:2543`), no `i64::MIN`/`MAX` sentinels
(`bind.rs:2510-2511`). Gather indices remain f32 (14336 and 32002 are both below the 2²⁴ exact
ceiling `shape.rs:297-310` enforces), so **this design needs no integer-buffer work anywhere.**

**`MAX_INLINE_STAGES` overflow policy, FORCED-BY cB #18:** `SmallVec` spills to the heap
silently, which is an unbudgeted allocation on the bind path. Policy: the fusion rule **declines
to extend a `Loop` past `MAX_INLINE_STAGES`** and starts a new one — a stated cost bound, not a
spill. `sized::MAX_INLINE_STAGES` default 8 in `proxima-tensor-runtime.toml` beside
`MAX_INLINE_RANK` (`sized.rs:72`); attention is 5.

## A.6 The rules, and the lowering theorem

Rules run inside `BoundOpBuilder`, which is already `Pipe<In = (Op, Shapes), Out = ReadyBatch>`
(`bind.rs:992-1003`) with a `held` map (`bind.rs:724-731`). **No new pipe stage** — B abandoned
that for the right reason (second binary question: the call site `shapes.and_then(builder)` is
identical before and after).

- **R1 PROLOGUE** (exists, generalized by A.2): producer elementwise, map composes, not
  data-dependent. Profitability from `FusionCost`, replacing `quarantine_broadcast_operands`
  (`bind.rs:935-973`, the ratio hard-wired to "never") and the `StillLive` veto
  (`bind.rs:698-701, 775-778`).
- **R2 EPILOGUE** (new): a `Loop` whose last stage folds, consumed by exactly one elementwise
  whose iteration space equals `output_axes`, appends that elementwise as a `fold: None` stage.
  Removes `residual1`, `x_next`, `ffn_hidden` = 3/layer = 96/token.
- **R3 BROADCAST-EPILOGUE** (new): same, with the consumer over the `Loop`'s *full* space.
  rmsnorm = 2/layer = 64/token.
- **R4 is deleted.** B's band-derivation matcher does not exist in this design; the domain is
  declared (§A.4). What remains is *lowering*: a constraint monotone in a loop axis renders as
  the loop bound, otherwise as a guard. Rendering as a bound is what removes the 2t-trips-for-t
  defect (`bind.rs:2504-2506` sets `cached_key_rows = new_key_rows = key_shape[0]`;
  `msl.rs:2586` `continue`s half of them).
- **R5 ONLINE-SOFTMAX**, a lowering theorem per emitter, exactly as **B §A.4** states it: a
  `Maximum` fold over axis set T and `Add` folds over the same T whose bodies depend on the
  maximum only through `exp(x - Stage(i))` may run in one traversal with running rescale.
  Declining costs a second traversal inside the same kernel — same dispatch count, different
  arithmetic order. No model name, no operand count, no stride literal.

**The `Rescaled` regression trap, FORCED-BY cA #2 — this changes the card order.**
`render_cached_attention`'s inner loop (`msl.rs:2586`) *already* implements rescaled online
softmax and stages K/V through threadgroup memory with `simd_sum`/`simd_broadcast_first`. A
card that deletes it and lands `Nest` first regresses the 3.0-3.6 ms attention arm and fails its
own `M` gate. **Therefore R5 and the threadgroup/simdgroup staging land in the SAME card as the
deletion (CARD A4), never after it, and `render_loop`'s spec explicitly requires cooperative
operand staging for stages whose operand is invariant across the reduced axis** — the property
the hand kernel has. Principle 14: the incumbent wins on both correctness and time.

**Capabilities replace the bool** (`bind.rs:2635-2640`, `fuse_cached_attention`), keeping B's
per-rule set over A's `ScheduleSet` bitfield because a set of named rendering capabilities is
what a backend actually publishes:

```rust
/// Which structural rewrites a backend can render. Every field is independent, so
/// wgpu declines `online_softmax` and still receives prologue/epilogue.
///
/// NOT `Settings`/`Serialize` — FORCED-BY cB #9: proxima-tensor's `config` feature
/// is `["std", "dep:bon", "dep:conflaguration", ..]` (Cargo.toml:37) and lib.rs:103-104
/// states "The `alloc` tier never sees it", while bind IS the alloc tier. Defaults come
/// from `sized::` consts; the std runtime override lives in `omega`'s config, which is
/// the crate that has one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FusionRules {
    pub prologue: bool,
    pub epilogue: bool,
    pub broadcast_epilogue: bool,
    pub online_softmax: bool,
}

impl FusionRules {
    /// Seeded from the build-time floor, so the no_alloc tier and the std tier
    /// cannot disagree (conflaguration bridge).
    pub const DEFAULT: Self = Self {
        prologue: sized::FUSION_PROLOGUE,
        epilogue: sized::FUSION_EPILOGUE,
        broadcast_epilogue: sized::FUSION_BROADCAST_EPILOGUE,
        online_softmax: sized::FUSION_ONLINE_SOFTMAX,
    };
}
```

**FORCED-BY cB #21:** rules are **not** env-settable at runtime and are **part of the plan cache
key**, because R5 changes floating-point arithmetic order and the cost model changes which nodes
materialize. An env var must not change numerics under a shared cache key. `PlanKey` carries
`FusionRules` (§B.6).

`FusionCost` (**B §A.6**) keeps its three terms but its `bytes_per_second` default is **173.6
GB/s** (§0.3) rather than B's 161, and its `dispatch_ns` default is **6_100** (the census's
MEASURED elementwise 1.76 ms / 290 ops) rather than B's 10_300 from the superseded serialized
profile (**cB #2, #3**). Both are `sized::` consts, not literals in source (P12).

## A.7 Emitters, and the CPU evaluator, implement the same rule

`omega::msl::emit` (`msl.rs:871-895`) loses `render_cached_attention` and gains `render_loop`,
which walks `stages`, reads every address from each operand's `Layout::strides` (audit finding
2 in full), and renders `BoundDomain` as loop bounds or guards. `proxima_tensor::cpu` gains the
mirror arm, replacing the `CachedAttention` arms at `cpu.rs:4765, 4830, 4979, 17955, 18103,
18144` and the construction at `cpu.rs:19178`. Because both read `domain` and `Layout`, the
parity gate compares one rule against its own unfused form, not two hand-written
implementations.

**Observability, FORCED-BY cA #16 and audit item 12:** `classify_kind` (`metal.rs:1631-1692`)
today substring-greps generated MSL to recover the emitter's own routing decision, and under a
single `Loop` kind every fusion class would report one label. The emitter **returns** its
routing decision:

```rust
/// The emitter's own routing decision, returned rather than recovered by grepping
/// its output (metal.rs:1631-1692, whose comment records that the recovered label
/// was wrong for 216 of 225 ops until an arm was added).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelFamily {
    Elementwise, CooperativeReduce, PackedRowBlocked, TiledGemm,
    GatherSerial, FusedTraversal, Iota, Constant,
}
pub struct KernelSpec { pub id: KernelId, pub family: KernelFamily, pub entry: ArrayString<KERNEL_ENTRY_CAP> }
```

The profiler groups by `KernelFamily`, so the census's per-class decomposition survives the
migration and CARD A5's per-lever gate is measurable. `KERNEL_ENTRY_CAP` is a `sized::` const
with a stated overflow policy (bind errors; an entry name is generated, so overflow is a bug not
an input) — **FORCED-BY cB #29**.

## A.8 Migration off `CachedAttention`, with a rollback seam

`bind_with_fusion` is already `#[cfg(feature = "cached-attention-streaming")]`
(`bind.rs:2641-2648`, opened; `proxima-tensor/Cargo.toml:33` = `["std"]`, `omega/Cargo.toml:21`).
**That seam is kept, not removed** — **FORCED-BY cA #22 and cB #11.** The new path lands behind
`feature = "loop-fusion"`, both compile, and the highest-risk card is bisectable by feature
selection rather than by `git revert`. The old feature is deleted only in CARD A6, after A4 and
A5 have held their gates for a full measurement round.

Two consequences the critiques caught and this design carries:
- `cached-attention-streaming = ["std"]` means the fused path is std-gated today; `loop-fusion`
  must **not** inherit that — `Loop` is the alloc-tier bind path, and CARD 1's gate builds
  `--no-default-features --features alloc` and names the modules it compiled (P3's N==0 clause).
- `grep -c CachedAttention == 0` is not a sufficient gate while the variant sits behind a
  feature (**cB #11**); the gate is run under `--all-features`.

Deletion list, complete (A §A.8 plus the three sites **cA #16** found missing):
`cached_attention_candidates`, `cached_attention_single_range_candidates`, the eight literal
stride tuples (`bind.rs:2451-2474`), the rank gates (`bind.rs:2417-2433`),
`exact_merged_causal_mask_cached_len` (`bind.rs:2353`), `attention_dependencies` (`2520`),
`attention_consumers` (`2559`), `has_external_attention_consumer` (`2545`),
`removable_attention_dependencies` (`2585`), the `bool` (`2635`), the second `bind_plain`
(`2678`), the variant and its 10 fields (`bind.rs:240-251`), its arms in `operands` (`309`),
`element_body` (`326`), `split_axis` (`409`), `classify_kind`'s arm (`metal.rs:1631-1640`),
`pack_cached_attention_uniforms` (`metal.rs:2197`), `render_cached_attention` (`msl.rs:2513`),
the `operands().len() == 9` read (`msl.rs:2542`), the six `cpu.rs` arms and the construction at
`cpu.rs:19178`.

---

# B. The decode step as FSM × orchestration over pipes

## B.1 One encoding of the step, not two

**FORCED-BY cB #20 and cA #6:** both designs shipped an FSM *and* a pipe chain with no stated
wiring, and A's chain never reached `InFlight`/`Sampled` at all. Decision: **the FSM is the
step; a pipe is what a caller composes.** The typestate transitions are the implementation, one
enum exists for suspension across a GPU wait, and the pipe face is the *session*, whose `call`
produces one accepted run of tokens.

```rust
// proxima-tensor — alloc tier, no device, no `omega` dependency, no config feature.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)] pub struct TokenId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)] pub struct StepIndex(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)] pub struct SubmitTicket(pub u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)] pub struct SlotId(pub u32);

/// How many tokens are committed, and how many the last pass accepted. The value
/// D.2's verifier writes, so multi-token is a field and not a code path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Cursor { pub position: u32, pub accepted: u16 }

/// Typestate: exactly one path exists (bind -> encode -> submit -> complete ->
/// sample), so transitions consume the old value and no `match` decides what may
/// happen next.
pub struct ReadyStep<'p>    { plan: &'p Plan, cursor: Cursor }
pub struct EncodedStep<'p>  { plan: &'p Plan, cursor: Cursor, commands: CommandRange }
pub struct InFlightStep<'p> { plan: &'p Plan, cursor: Cursor, ticket: SubmitTicket }
pub struct SampledStep<'p>  { plan: &'p Plan, cursor: Cursor, counters: StepCounters }

/// The suspension form, and the ONLY runtime state discriminator in this design:
/// `poll_step` must resume a step it did not start.
#[must_use]
pub enum Step<'p> {
    Ready(ReadyStep<'p>), Encoded(EncodedStep<'p>),
    InFlight(InFlightStep<'p>), Sampled(SampledStep<'p>),
    Halted(Halt),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Halt { EndOfSequence(TokenId), BudgetSpent(StepIndex) }
```

**Lifetimes, FORCED-BY cA #4, cB #6.** `Pipe` is `type In; type Out; type Err; fn call(&self,
input: Self::In) -> impl Future<..>` (`proxima-primitives/src/pipe/primitives.rs:91-101`,
opened) — no lifetime parameters, no GATs. Therefore:

- **No borrowed view crosses a `Pipe` boundary.** `LogitsView` never appears in an `In` or an
  `Out`. Logits stay in the session's plan-owned readback buffer; sampling happens inside the
  `InFlight -> Sampled` transition, which is an inherent method and free to borrow. This is what
  keeps the zero-allocation budget honest: A's `Read::Out = LogitsView<'_>` forced either an
  inexpressible associated type or a 128 KB owned buffer per token.
- **`Step` is never a `Pipe` `In`/`Out`.** The pipe face carries POD only.
- **No `&mut` is produced from `&self`.** B's `Advance: Pipe<Out = ReadyStep<'p>>` with
  `scratch: &'p mut StepScratch` cannot exist; the scratch is plan-owned and reached through the
  session's `&mut self`, and the loop is `poll_step`, not an `AndThen` cycle — `AndThen` is
  strictly `First::Out = Second::In` (`primitives.rs:203-221`) and cannot express a cycle at all.
- **`encode`'s buffer is not lifetime-locked.** B's `encode<'p>(plan: &'p Plan, .., out: &'p mut
  [Command<'p>]) -> &'p [Command<'p>]` permits exactly one encode per plan (**cB #6**). The
  corrected signature separates the two lifetimes:

```rust
/// Pure, alloc tier. This is `execute_plan_with_placements`'s loop
/// (metal.rs:1074-1140) with every device call and every per-token `Vec` removed.
/// `'p` is the plan; `'c` is the caller's command scratch, reused every token.
#[must_use]
pub fn encode<'p, 'c>(
    plan: &'p Plan,
    bindings: &Bindings<'p>,
    out: &'c mut [Command],
) -> Result<&'c [Command], TensorError>;

/// Backend-neutral. `Barrier` carries the SPECIFIC slots, restoring what
/// `MTLBarrierScope::Buffers` (metal.rs:1105) and `HazardTracker::reset`'s
/// clear-both-sets (metal.rs:874-877) destroy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    Barrier { slots: BarrierRange },
    Dispatch { position: u32, kernel: KernelId, bindings: BindingRange,
               grid: Grid, uniforms: UniformRef },
}
```

`Command` is `Copy` and index-based (ranges into plan-owned tables), so the scratch is a plain
`[Command; N]` with no interior lifetimes — which is also what makes it tier-3 shaped.

## B.2 The pipe face: three pipes, and what each buys a caller

| pipe | In | Out | form | why it is a pipe |
|---|---|---|---|---|
| `Draft` | `TokenWindow` (POD, `Copy`) | `Draft` (POD, `Copy`) | transform | a caller can swap n-gram for a draft model with no other change (D.2) |
| `Submit` | `SubmitRequest` (plan id + command range) | `SubmitTicket` | transform | crosses the device edge; a caller can await it |
| `Session` | `StepInputs` (POD) | `StepOutcome` (POD) | transform | one call = one pass; this is `LoadedModel::call`'s existing shape (`generate.rs:1533`) |

```rust
/// POD in, POD out — no borrow crosses the boundary (cA #4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepInputs { pub draft: Draft, pub budget: Budget }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepOutcome {
    pub accepted: ArrayVec<TokenId, MAX_ACCEPTED>,
    pub cursor: Cursor,
    /// Owned by value, produced by the transition. Reading a step's counters
    /// twice does not compile, so `Counter::snapshot_and_reset`'s "exactly once
    /// per step" protocol (metal.rs:2719, honoured by a comment at
    /// generate.rs:2468-2482) stops being a discipline (cA #17).
    pub counters: StepCounters,
    pub halt: Option<Halt>,
}
```

Completion is **poll-based, not a blocking future** (**cA #12**): `waitUntilCompleted`
(`metal.rs:680`, `1146`) parks a thread inside async, which is principle 21 rung 3 used at rung
2's position and it forecloses overlapping the drafter with the GPU pass.

```rust
impl MetalDriver {
    /// P20 reactor surface. `addCompletedHandler` wakes the task; nothing parks a
    /// thread. This is what lets D.2's drafter run on the CPU while the pass is
    /// in flight. NOT a `Pipe` — `&mut In`, a `Context` and `Poll` match none of
    /// `Pipe`/`SendPipe`/`UnpinPipe` (primitives.rs:91-178), and cB #7 was right
    /// that asserting otherwise is false. It is the driver's poll surface; the
    /// pipe is `Submit`.
    pub fn poll_complete(&self, step: &mut InFlightStep<'_>, cx: &mut Context<'_>)
        -> Poll<Result<Completion, MetalError>>;
}
```

**Everything else is a function, not a pipe (cA #5).** `PlanSlots`, `ScheduleBarriers`,
`PackUniforms`, `Encode`, `Read`, `Sample`, `StopPolicy`, `Advance`, `BindInputs` are all
deleted from the design. Written both ways, `PlanSlots.call((resolved, retires, outputs)).await?`
versus `plan_slots(&resolved, &retires, &outputs)?` are the same line, and the pipe form is
strictly worse: `call` is async (a pure interval assignment returns a future) and takes `In` by
value (so `Vec<BoundOp>` moves into the first stage and the next cannot read it). They are
relocations.

**The device edge count.** `grep -n commandBuffer omega/src/metal.rs` returns **3** (588, 1041,
1343) — not ten (**cA #14**). Gate: `== 1` after CARD B2. Entry points are **eleven** (verified
above), of which nine are executors and two are planners; gate: `grep -c '^pub fn execute' == 0`
and one `plan`.

## B.3 `plan()` is pure, and `Schedule` is plan-time data

```rust
// proxima-tensor. `codecs` is `&[Option<BlockCodec>]` where `BlockCodec` is
// proxima-tensor's own enum — FORCED-BY cB #8: B's signature took `PackedCodec`,
// which is defined at omega/src/msl.rs:788, inverting the crate graph in the same
// breath as the comment denying it. `metal.rs:445-459` keeps its job of mapping
// `BlockCodec` -> `PackedCodec` and gains nothing to do.
#[must_use]
pub fn plan(
    program: &[Op],
    symbols: &[u64],
    codecs: &[Option<BlockCodec>],
    outputs: &[NodeId],
    rules: FusionRules,
) -> Result<Plan, TensorError>;

pub struct Plan { /* program, shapes, resolved, retires, outputs, block_nodes, schedule */ }

/// Everything the driver recomputes per token or builds with device IO, computed
/// once as pure data. Zero `Retained<_>`, zero `MTLBuffer`; compiles under
/// `--no-default-features --features alloc`.
pub struct Schedule {
    pub position_slot: Vec<SlotId>,
    pub slot_bytes: Vec<usize>,
    pub peak_bytes: usize,
    /// Which slots each barrier must cover. Hazards are a function of
    /// (position -> slot) reads/writes, which the plan fixes.
    pub barriers: Vec<BarrierSet>,
    pub uniforms: Vec<UniformBlob>,
    pub uniform_patches: Vec<UniformPatch>,
    pub kernels: Vec<KernelSpec>,
}

/// Pure, alloc tier, testable with `SlotId` and no device — the capability
/// today's generic `HazardTracker<Id>` gestures at (metal.rs:824-827) but cannot
/// deliver because its only caller lives inside `execute_plan_with_placements`.
#[must_use]
pub fn barrier_schedule(resolved: &[BoundOp], position_slot: &[SlotId]) -> Vec<BarrierSet>;
```

`Device::realize(&self, plan: &Plan, resident: &ResidentSet) -> Result<Residency, MetalError>`
is the device half, called once per plan. It absorbs `mark_resident`'s two-phase init
(`metal.rs:373`) and `register_checkpoint_mapping`'s side channel (`metal.rs:2957`) — both
named as unanswered by **cA #17**; `Plan::resident_nodes` (read at `metal.rs:563, 1026`) becomes
a field of `Residency`, and the `Plan` is a self-contained value with one construction path.

## B.4 The eight (six) thread-locals

`grep -c 'thread_local!' omega/src/metal.rs` = **6**, not eight (the brief and audit say eight;
the discrepancy is stated rather than silently corrected — **cA #14**). They become fields of
`MetalDriver`: `PIPELINE_CACHE`/`DEVICE_AND_QUEUE` (252, 266-273) → `device`/`queue`/`pipelines`;
`OUTPUT_BUFFER_POOL` (2477) and `BufferArena` → `slots`; checkpoint mapping (2933) → `residency`;
nocopy (3007) and resident copies (3150) → `residency`; `UNIFORM_BUFFERS` + `UNIFORM_CACHE_CLOCK`
(3246, 3266) → `uniforms`. The content-keyed uniform LRU (`evict_least_recently_used`,
`metal.rs:3315`) is **deleted**, not moved: a plan-owned uniform table indexed by position has
nothing to dedupe. Less work, same output.

The driver is `!Send`, per-core, shared-nothing. No mutex is introduced — a thread-local is a
missing owner, not a missing lock (P21 rung 1).

## B.5 Allocation budget: zero per token on a plan hit

Sites removed, each cited: `generate.rs:2301-2306` (`named_blocks` rebuilt every step),
`2323-2324` (`cached_len_scalar`, deleted outright — it is a symbol now), `2347-2366` (KV
scratch resize + 3 pushes/layer), `2370-2373` (two placement `Vec`s), `2400-2413` (roots),
`2503` (`next_ids = vec![token_id]`); `metal.rs:541`/`996` (`device_buffers: BTreeMap` per
call), `1073` (`pending_faults`), `1095-1100` (`hazard_inputs` per op = 616/token), `2183`
(`pack_uniforms -> Vec<u8>` per op = 616/token), `3326` (`Vec<u8>` uniform cache key per op).
**Also on the list, which B's table omitted (cB #22):** the logits readback staging buffer, the
sampling scratch (`proxima-tokenizer/src/sample.rs:277`), and `SmallVec` spill on
`stages`/`output_axes`/`BarrierSet` — the first two become plan-owned buffers, the third is
prevented by §A.5's decline-rather-than-spill policy.

```rust
/// Reads as documentation of the contract (P17): a decode step that hits its plan
/// performs no heap allocation, and the test asserts the STEP COUNT so a loop that
/// ran zero steps cannot pass. Zero work and successful work do not share a signal.
#[proxima::test]
async fn a_plan_hit_decode_step_allocates_nothing_and_runs_every_step() {
    let counting = CountingAllocator::install();
    let mut session = fixture_session();            // real openchat geometry, real gguf bytes
    session.decode_steps(WARMUP).await.expect("warmup builds the plan");
    let before = counting.total();
    let misses_before = session.plan_misses();
    let produced = session.decode_steps(STEPS).await.expect("steady state");

    assert_eq!(produced.len(), STEPS, "N==0 is RED: the loop must have run every step");
    assert_eq!(session.plan_misses(), misses_before, "steady state is a plan hit");
    assert_eq!(counting.total() - before, 0, "steady-state decode allocates nothing");
}
```

`CountingAllocator` does not exist in `proxima-tensor` or `omega`; the only instances are
private to `proxima-telemetry/tests/elevation_memory.rs:42` and
`proxima-telemetry/benches/bench_lossless_producer_assist.rs:38` (**cB #22**, **cA (A's own
grep)**). **It is promoted to `proxima-test` in CARD B0 and the two private copies are deleted**
— a prerequisite that surfaced is part of this work (P15), and a third copy is not.

## B.6 The plan cache

Bounded, `ArrayVec<(PlanKey, Plan), sized::PLAN_CACHE_ENTRIES>`, replacing the one-entry
clear-then-insert at `generate.rs:1325, 1373, 1417` and deleting `PlanCacheEntryVanished`, an
impossible-state error. **Capacity is derived, not asserted in prose (cA #32):**

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PlanKey {
    pub key_capacity: u32,
    pub new_tokens: u16,
    /// Part of the key because R5 reorders floating-point arithmetic (cB #21):
    /// two rule sets are two programs and must not share a cache entry.
    pub rules: FusionRules,
}
```

The live key space is `{prefill} ∪ {k = 1} ∪ {k = draft width}` = 3 in steady state with one
draft width, so `PLAN_CACHE_ENTRIES` defaults to 4 in `omega-runtime.toml` with the derivation
in the doc comment.

**B §B.7's "bind K/V at capacity" is taken, and it is safe here where it was not in B**
(**cB #5**): the domain is binding, not advisory (§A.4), so traversal is `[0, merged)` regardless
of the declared extent and an emitter cannot decline it. `symbols = [new_count, capacity]` is
then constant in steady state: one plan, one arena, one uniform set. `KV_BUCKET_TOKENS`
(`proxima-tensor/src/sized.rs:326`, asserted = 32 at `sized.rs:390`) moves to
`omega::sized::KV_EXTENT_BUCKET_TOKENS` **and its default becomes 1**, because a dynamic band
leaves bucketing nothing to buy. **FORCED-BY cA #26:** its consumers are in
`proxima-model-interop/src/generate.rs:695, 709-710, 2336`, so the card covers those call sites
and the gate is `grep -rc KV_BUCKET_TOKENS proxima-tensor/src proxima-model-interop/src == 0`;
interop reaches it through `omega`, an edge that already exists.

---

# C. Generic model programs

## C.1 Production decode from spec data

`ProgramSpec` already carries the house pattern (`spec.rs:69-72`) and `Vec<Op>: TryFrom` is
exercised on `specs/mistral_layer.toml` at real dimensions (`spec.rs:10313-10327`). The missing
piece is stacking. **Taken: B's `extend`/`stack`** over A's `lower_into`, because it keeps
lowering in `TryFrom` (a stacked spec is an ordinary spec, so nothing downstream learns a second
entry point).

```rust
impl ProgramSpec {
    /// Concatenates `other` after `self`, prefixing every id in `other` and
    /// rewiring `other`'s named inputs from `self`'s ids per `wiring`. Pure.
    pub fn extend(&self, prefix: &str, other: &ProgramSpec, wiring: &Wiring)
        -> Result<ProgramSpec, TensorError>;

    /// 32 layers = `fold` over 32 `extend`s with `{layer}` substituted in weight names.
    pub fn stack(&self, count: u32, wiring: &StackWiring) -> Result<ProgramSpec, TensorError>;
}

#[derive(Debug, Clone, PartialEq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "TENSOR_STACK")]
pub struct StackWiring {
    pub carry_in: String,
    pub carry_out: String,
    /// Contains `{layer}`: "blk.{layer}.ffn_down.weight".
    pub weight_names: Vec<String>,
    pub cache_roots: Vec<String>,
}
```

These derives are legal here and not in `bind`: `spec` is already behind the `config` feature
(`proxima-tensor/Cargo.toml:37` = `["std", "dep:bon", "dep:conflaguration", "dep:serde",
"smallvec/serde"]`, and `lib.rs:103-104` states the alloc tier never sees it). That is the
line **cB #9** drew and it is respected: **config-derived types live at the spec face, never on
the bind path.**

Gate (both designs agree, and it is the strongest gate in §E): the TOML-built program compares
`PartialEq` node-for-node against `mistral_single_range_cached_forward_program`'s output
**before** any run, and only then do the Rust builders (`spec.rs:898..7184`) delete.

## C.2 Typed roots, typed layer inputs

Taken from **A §C.2** (the `KeyRoots` enum, because it turns 15+ positional destructures into
exhaustive matches) merged with **B §C.2** (`MaskSource`, `HeadNorm` — rich enums where an
`Option` would have been a flag).

```rust
/// Replaces `pub type CachedLayerRoots = (NodeId, NodeId, NodeId)` (spec.rs:2333,
/// consumed at 15+ sites) and `Qwen35DenseAttentionRoots` (spec.rs:2344).
/// Exhaustive: a new rotary layout is a variant, and every consumption site is a
/// `match` the compiler checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum KeyRoots {
    /// Interleaved 2i/2i+1 rotary — Mistral/Llama. After A.2's `compose`, the two
    /// halves are stride-2 views of ONE buffer, so `kv_cache.{i}.k_even`/`k_odd`
    /// (generate.rs:677-680) stop existing as separate tensors.
    Paired { key: NodeId },
    /// Split-half (NEOX) with an unrotated remainder — Qwen3.5.
    Split { first: NodeId, second: NodeId, passthrough: NodeId },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheRoots { pub key: KeyRoots, pub value: NodeId }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub struct AttentionWeights { pub query: NodeId, pub key: NodeId, pub value: NodeId, pub output: NodeId }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub struct FeedForwardWeights { pub gate: NodeId, pub up: NodeId, pub down: NodeId }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub struct NormWeights { pub attention: NodeId, pub feed_forward: NodeId, pub epsilon: NodeId }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub struct RopeTables { pub cosine: NodeId, pub sine: NodeId }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum MaskSource { Causal, Supplied(NodeId) }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum HeadNorm { None, PerHead { query: NodeId, key: NodeId, inv_head_dim: NodeId } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerInputs {
    pub activations: NodeId,
    pub norms: NormWeights,
    pub attention: AttentionWeights,
    pub feed_forward: FeedForwardWeights,
    pub rope: RopeTables,
    pub cache: CacheRoots,
    pub mask: MaskSource,
    pub head_norm: HeadNorm,
}
```

Honest claim, kept from A: every field is a `NodeId`, so this makes a swap a *named* error at
the construction site, not a type error. 23 newtypes hosting nothing is what the no-newtype rule
forbids. **FORCED-BY cA #27:** `grep -c too_many_arguments proxima-tensor/src/spec.rs` is
**17**, spanning mistral and qwen35; CARD C2's gate is scoped to the mistral builders
(count falls to the qwen35 residual, asserted by name) and CARD C4 finishes qwen35 — the gate
matches what the card delivers.

## C.3 `cached_len`

Deleted as a leaf. It is `SymbolId` in the uniform block (§A.5). Gone with it: the named block
(`generate.rs:2323`), the f32 round trip, the device upload, the kernel's buffer read
(`msl.rs:2546`), the ninth operand, and `bind.rs:2446`'s rank-0 check. **No integer buffer is
introduced anywhere** (**cB #10**).

## C.4 What a new architecture costs

**Data only** when it composes shipped generators: Llama/Mistral, GQA/MQA (two-term
`IndexPattern`), sliding window (a second `HalfSpace`), ALiBi (`Iota`-derived bias), MoE
(`IndexMap::Computed`, live at `cpu.rs:7091`), NEOX vs interleaved rotary (`KeyRoots` + affine
maps), tree masks (`MaskSource::Supplied`).

**Rust** for: a new `ScalarOp` (the set at `op.rs:52-56` is closed and stays closed), a new
`BlockCodec`, a new emitter kernel family — **and for a selective scan.** **FORCED-BY cB #23,
which is correct and both designs got wrong in opposite directions:** B claimed "an SSM/Mamba
layer is data: its scan is `Keep::Scan`". `Op::Reduce` carries **one** `operand: NodeId`
(`op.rs:157`) and one `body: ScalarOp` (`op.rs:155`); `h_t = a_t·h_{t-1} + b_t·x_t` is a linear
recurrence over two per-step streams. It is not expressible, and `Qwen35SsmShape`
(`generate.rs:731`) exists in production because of it. The honest statement, which is A's
boundary stated at B's face: **a multi-operand fold is the one architecture class that needs
Rust today**, and closing it is a tuple-valued reduce — deliberately not taken here, named so
the next design starts from the fact rather than rediscovering it.

## C.5 Sizing

New `sized::` consts, all through the existing `build.rs` + TOML pattern, none a literal in
source (P12) — **FORCED-BY cA #18**, which caught five named with no home:

| const | crate / file | default | overflow policy |
|---|---|---|---|
| `MAX_INLINE_CONSTRAINTS` | proxima-tensor-runtime.toml `[map]` | 2 | spill (rare, cold) |
| `MAX_INLINE_STAGES` | proxima-tensor-runtime.toml `[bind]` | 8 | fusion declines (§A.5) |
| `FUSION_{PROLOGUE,EPILOGUE,BROADCAST_EPILOGUE,ONLINE_SOFTMAX}` | proxima-tensor-runtime.toml `[fusion]` | true | n/a |
| `FUSION_COST_{BYTES_PER_SECOND,DISPATCH_NS}` | proxima-tensor-runtime.toml `[fusion_cost]` | 173_600_000_000 / 6_100 | n/a |
| `MAX_INLINE_BARRIER_SLOTS` | omega-runtime.toml `[spans]` | 8 | spill |
| `PLAN_CACHE_ENTRIES` | omega-runtime.toml `[plan]` | 4 | LRU evict |
| `COMMAND_BUFFER_CAPACITY` | omega-runtime.toml `[spans]` | 1024 | encode errors |
| `KERNEL_ENTRY_CAP` | omega-runtime.toml `[spans]` | 96 | bind errors |
| `KV_EXTENT_BUCKET_TOKENS` | omega-runtime.toml `[kv]` | 1 | n/a |
| `MAX_DRAFT`, `MAX_ACCEPTED` | omega-runtime.toml `[decode]` | 8, 9 | draft truncates |

Runtime defaults are seeded from these consts (`fn default_x() -> T { sized::X }`) and a
`defaults_track_the_sized_floor` parity test pins it. **FORCED-BY cA #19:** the driver config
surfaces (`SubmitConfig`, `SampleConfig`, `DriverConfig`) carry `Builder + Deserialize +
Serialize + Settings` in `omega` (which is std) with a config↔builder parity fixture, not only
`ModelSpec`.

---

# E. Ordering, gates, tripwires

Standing battery, every card, N asserted nonzero on every count (a gate that cannot report its N
is not a gate):

- **P** parity ≤1e-4 vs `proxima_tensor::cpu::evaluate` on the real openchat-3.5 Q4_K_S program,
  every output node.
- **B** 20 runs byte-identical logits (the brief's ×20).
- **T** generated text identical to the pre-card run, same prompt, same seed. **Not applied to
  cards that reorder floating-point arithmetic** — **FORCED-BY cB #28**: R5 is a reassociation,
  greedy decode is a discrete argmax over logits differing in the last bits, so text identity is
  probabilistic. For those cards the gate is `assert_close(fused, unfused, 1e-6)` on the CPU
  oracle **plus** EM-at-64 ≥ 0.99 over 200 prompts.
- **M** ms/token and `gpu_exec_ms` not worse beyond CoV over 3 rounds; CoV reported.
- **A** allocation counter == 0 on plan-hit steps, step count asserted.
- **Q** (lossy cards, replacing T) EM-at-64 vs the full model, greedy, 200 held-out prompts,
  plus held-out perplexity delta. Thresholds per §D.
- **G** bytes/token measured from the driver's own counters, reported beside §D.6's prediction.
  Disagreement > 5% halts the sequence.
- **X** the cross-backend cell — **FORCED-BY cB #13, #16**: every card touching `BoundOpKind`
  runs `scripts/proxima-tensor-gate.sh` (alloc tier check, alloc tier clippy, `std without
  config`, config alone) **and** builds/tests `wgsl`, `cuda`, `wgpu_driver`, `proxima-onnx`, and
  `proxima-tensor/tests/rewrite_law_equivalence.rs`. Measured blast radius of the `Loop` port:
  74 `BoundOpKind::` sites in `cpu.rs`, 65 `bind.rs`, 39 `msl.rs`, 22 `wgsl.rs`, 19 `cuda.rs`,
  16 `metal.rs`, 12 `spec.rs`, 8 `wgpu_driver.rs`, plus examples and six `omega/tests/*`.

### Cards

| # | card | change | gate beyond the battery |
|---|---|---|---|
| 0 | **D0** | Measure sustained GB/s: pure-read kernel with the identical access pattern, 3 sizes, memcpy control that should saturate. Read per-tensor codecs from the gguf table. No source change. | sustained GB/s with CoV over 5 runs; the control saturates or the 400 figure is retracted; per-tensor codec table published |
| 1 | **B0** | `CountingAllocator` promoted to `proxima-test`; the two private copies deleted. | the two `proxima-telemetry` tests pass against the shared one; N asserted |
| 2 | **A0** | `IndexPattern::compose` + strided extent resolution. | `bind` over `mistral_layer.toml` at real dims `PartialEq` node-for-node to pre-card; a stride-2 fixture that returned `UnconstrainedDim` binds. **X** |
| 3 | **A1** | `Offset`/`SymbolId`; `AxisIndex::offset` widens; `Extent::Symbolic(SymbolId)`; `Domain` on `Op` with `#[serde(default)]`; the 32 struct-literal sites migrated; `shape::infer` and `cpu` honour it. | all eleven `specs/*.toml` deserialize (N = 11 asserted); property test: a `Domain`-restricted reduce equals the `Iota`/`Greater`/`Select` form over randomized extents; `causal_attention.toml` rewritten with a domain evaluates identically. **X** |
| 4 | **A2** | `BoundOpKind::Loop` + `StepArg::Stage` behind `feature = "loop-fusion"`; Elementwise/Reduce become 1-stage Loops; 5 emitters + CPU ported. Old path still selectable. | dispatch count still 616; **B**; **M**. **X** — this is the card the cross-backend cell exists for |
| 5 | **B1** | `plan()` pure (`BlockCodec`, not `PackedCodec`); `Schedule`; `barrier_schedule`; `Device::realize` absorbs `mark_resident` + `register_checkpoint_mapping`. | `plan()` compiles and runs on a non-macOS target **and names the modules the alloc build compiled**; barrier count per token identical to today's `BARRIERS_EMITTED` |
| 6 | **B2** | One `MetalDriver`; `Submit`/`poll_complete`; the nine executors deleted; `SubmitConfig` absorbs {named, placed, timed} and the `PROXIMA_METAL_OP_PROFILE_STEP` env branch (`generate.rs:2428-2455`). | `grep -c 'commandBuffer()' == 1` (from 3); `grep -c '^pub fn execute' == 0`; every `metal_parity` case passes through the new entry, N asserted |
| 7 | **B3** | Six thread-locals → fields; uniform LRU deleted; `PlanCache` bounded with `FusionRules` in the key. | `grep -c 'thread_local!' omega/src/metal.rs == 0`; two `MetalDriver`s on two threads produce identical logits |
| 8 | **B4** | `Step` FSM + `Session` pipe; the decode closure (`generate.rs:2278-2545`) replaced; `StepCounters` owned by value. | **A** with N = 100; a runnable walkthrough driving every legal transition (P11); host residual reported against the 5.96 ms MEASURED baseline |
| 9 | **A3** | `KernelSpec`/`KernelFamily`; `classify_kind` deleted; profiler groups by family. | per-family decomposition reproduces the census's 225/391 split within CoV before any fusion lands |
| 10 | **A4** | R5 online-softmax **and** cooperative operand staging in `render_loop`, in ONE card; `CachedAttention` and its 20 companions deleted under `--all-features`; `cached-attention-streaming` still selectable. | attention op count still 32; **attention-arm `gpu_exec_ms` ≤ the hand kernel's 3.0-3.6 ms** (cA #2 — this is the card that could regress); key-axis trip count == t read from the emitted MSL bound; `assert_close 1e-6` + EM ≥ 0.99 |
| 11 | **A5** | R2 epilogue + R3 broadcast-epilogue. | 616 → **520** MEASURED, per lever toggled independently |
| 12 | **A6** | `FusionCost` replaces `quarantine_broadcast_operands` + the `StillLive` veto; R1 over RoPE/GQA; the masking `Select` deleted where a `Domain` covers it; `cached-attention-streaming` removed. | 520 → **264** MEASURED per card, not projected; **X** |
| 13 | **B5** | K/V bound at capacity; `KV_EXTENT_BUCKET_TOKENS` → omega, default 1; interop call sites moved. | `plan_misses == 1` over 128 steps **and** `steps == 128`; `grep -rc KV_BUCKET_TOKENS proxima-tensor/src proxima-model-interop/src == 0` |
| 14 | **D5** | f16 KV. | **Q** EM ≥ 0.99; **G** at this ctx / 2048 / 8192 |
| 15 | **D4a** | `output.weight` Q6_K → Q4_K. | **Q** EM ≥ 0.98; **G** |
| 16 | **D1a** | `nr1` s-axis fold in `push_packed_row_blocked_body` (`msl.rs:3171-3200`). No program change. | weight bytes/pass MEASURED **flat** in k ∈ {1,2,4,8}, not linear. **G** |
| 17 | **D2** | Draft + verify as one program: `Draft` pipe (prompt-lookup), argmax/scan/accept chain, `Cursor::accepted`, k-row sampling. | **T** (lossless); **k′ per prompt class as a histogram; kill at k′ < 1.5**; **G** vs §D.6 |
| 18 | **D3-pre** | Lift the **three** gather routing sites (`msl.rs:1047`, `1218`, `1389`); row-axis `Lookup` on the packed-row body; block-aligned contraction-axis skip for `ffn_down`. | **ns/element ≤ the dense arm** at d = 0.35 (the gate ROW 180/181 failed at 0.2-0.29 vs 0.057); **the 65 cooperative reduces/token measured separately and not regressed** (cA #15) |
| 19 | **D3-calib** | Fit the group router (4096×56/layer) by least squares against per-group L1 mass over 50k calibration activations; store as a sidecar gguf tensor. | mass-recall of the fitted router vs the oracle top-`g` groups, p10 reported; artifact size 4.1 MB/token confirmed |
| 20 | **D3b** | `ScatterBounds::{Fault, Drop}`; threshold + `Keep::Scan` + bounded compaction as ONE fused `Loop`; group-gathered FFN at group = 256. | **Q** EM ≥ 0.95, ppl ≤ 1.0%; **random-group control must fail**; **union density at k ∈ {1,2,4} MEASURED, kill at k=4 union > 0.6**; selector dispatch count ≤ 2/layer and its ms reported; `plan_misses` still 1 |
| 21 | **D3-sweep** | `d ∈ {0.25, 0.35, 0.5, 0.7}`, group ∈ {256, 512}. | the §D.6 table recomputed from measurement; **G** |
| 22 | **D4b** | `output.weight` candidate-set gather at c = 4096. | candidate-miss rate ≤ 0.1%, else revert to D4a |
| 23 | **D3q** | Q3_K encoder/decoder in `proxima_gguf::quant`, MSL unpack, `BlockCodec::Q3K`, lift `UnrepresentableGgmlType` (`proxima-model-interop/src/bind.rs:70-73`). | **Q** EM ≥ 0.97, ppl ≤ 0.5%; the lowered `Vec<Op>` `PartialEq` to pre-card (program unchanged); **the double-quantization caveat measured, not assumed** (cB #25) |
| 24 | **C1** | `ProgramSpec::extend`/`stack`, `StackWiring`; production decode from `specs/mistral.toml`. | TOML-built program `PartialEq` node-for-node with the Rust builder's output **before** any run; then **P/B/T/M**; the `#[ignore]` at `spec.rs:10339-10341` removed |
| 25 | **C2** | `LayerInputs`, `KeyRoots`, `CacheRoots`, `MaskSource`, `HeadNorm`; `cached_len` leaf deleted. | `grep -c '"cached_len"' == 0`; `too_many_arguments` in `spec.rs` falls to the qwen35 residual, count asserted by name |
| 26 | **C3** | Driver config surfaces with `Settings + Builder` + parity fixtures. | config↔builder parity for every driver config type, N asserted |
| 27 | **C4** | qwen35 builders onto `LayerInputs`/`KeyRoots::Split`. | `grep -c too_many_arguments proxima-tensor/src/spec.rs == 0` |

**Ordering rationale, and the two ordering defects both critiques found are fixed.** A2's gate
required A3's deletions (**cA #3**) — here the discriminator/sentinel greps are gated on CARD
A4, which is the card that deletes them. B's cards 3/4 required an integer `cached_len` from
card 8 (**cB #14**) — here `cached_len` is a symbol from CARD A1, so nothing waits on it. There
is **one** ordering, not two (**cB #14**). D1a precedes D2 precedes D3-pre precedes D3b, so no
byte lever is credited before its kernel precondition is measured.

**Tripwires — stop and report:** any node's parity > 1e-4; any byte difference in the 20 runs;
text differs on a card that does not reorder arithmetic; ms/token or `gpu_exec_ms` worse beyond
CoV over 3 rounds; allocation counter nonzero on a plan hit; any grep-count gate nonzero; any
gate whose N is 0; measured bytes/token disagreeing with §D.6 by > 5%; a program that fuses on
one backend and not another (capabilities gate *whether*, never *which*); union density at k=4
> 0.6; k′ < 1.5; any **Q** threshold missed — that card reverts by feature selection and §D.6's
sensitivity table says whether the target is still reachable without it.

## E.2 What each constraint changed, and what it killed

- **The measured decomposition (§0.3).** Changed the order: the 5.96 ms host residual is the
  first blocking term, so §B's cards precede every byte lever except the two measurements.
  **Abandoned:** both designs' framing that §A/§B are "margin" bought against a 6.3 ms dispatch
  term — that term is 4.05 ms and the host term nobody counted is larger than it.
- **no_std + alloc tier (P3).** Forced the live cache length to be a symbol into a caller-supplied
  `&[u64]` rather than an operand read. **Abandoned:** B's `BandBound::Dynamic { slot }`, the
  smaller diff, because evaluating a loop bound then requires a buffer read that does not exist
  at tier 3 — and, per `map.rs:99-107`, requires an integer-buffer kind no backend plumbs. Also
  **abandoned:** `FusionRules` as a `Settings` type on the bind path (`Cargo.toml:37`,
  `lib.rs:103-104`), which does not compile at the tier it was placed in.
- **Reuse-first (P1).** Forced `HalfSpace = AxisIndex` rather than a second affine type.
  **Abandoned:** A's parallel `HalfSpace` struct — and with it the drift where a constraint could
  carry a symbolic offset and an address could not.
- **The recursion rule at `bind.rs:160-162`.** **Abandoned:** A's `FusedRegion { ops: Vec<BoundOp> }`,
  which was the more readable of the two and which makes `BoundOpKind` the recursive tree that
  comment rejects by name.
- **The pipe question, second half.** **Abandoned (nine):** `PlanSlots`, `ScheduleBarriers`,
  `PackUniforms`, `Encode`, `Read`, `Sample`, `BindInputs`, `Advance`, `StopPolicy`. Each call
  site is the identical line as a plain function, and `call`'s by-value `In` plus its `async`
  make the pipe form strictly worse for pure plan-time work. Retained from A: the asymmetry
  result — the **draft source is a pipe** (once per pass, host, zero weight bytes), the **elision
  selector is not** (32 device crossings per token; the landed probe already measured that shape
  dispatch-bound).
- **P20 box-free / reactor.** **Abandoned:** a `Submit` future that blocks on
  `waitUntilCompleted` (`metal.rs:680, 1146`). `poll_complete` is the surface, and it is what
  makes the drafter overlappable with the pass.
- **Principle 14, the incumbent wins.** Forced R5 and cooperative staging into the same card as
  the deletion. **Abandoned:** the two-card sequence in which a generic `Nest` emitter
  temporarily replaces a hand kernel that already implements rescaled online softmax with
  threadgroup staging (`msl.rs:2586`).
- **The measured loss (ROW 180/181).** Forced group = 256 (a whole Q4_K superblock,
  `msl.rs:444`) rather than 128, because `ffn_down`'s elided axis is the contraction axis
  (`msl.rs:3184-3189`) and an unstructured selection saves it nothing. **Abandoned:** A's
  128-unit group and, with it, the claim that all three FFN matrices elide under a row gather.
- **P15.** `CountingAllocator` does not exist in these crates; it is CARD B0, not a follow-up.
  The router weights do not exist; fitting them is CARD D3-calib, not an assumption.
- **What was NOT abandoned, said so the count is honest (cB #27):** the multi-token and elision
  levers both survive with preconditions and changed gates. A changed gate is not an abandonment.

## E.3 The central claim as a lint: what is NOT a pipe

1. **IR data** — `Op`, `BoundOp`, `Loop`, `Stage`, `Domain`, `HalfSpace`, `Offset`,
   `BoundDomain`, `Command`, `Plan`, `Schedule`, `KernelSpec`, `CacheRoots`, `LayerInputs`.
   Values that flow through pipes. Each earns its place by the second question: `Domain` lets a
   caller write a banded reduce whose iteration space is triangular; `Loop` lets a caller express
   five folds sharing one traversal; `KernelSpec` lets a profiler group by the emitter's own
   decision instead of by a substring of its output.
2. **Config** — `FusionRules`, `FusionCost`, `SubmitConfig`, `SampleConfig`, `StackWiring`,
   `ModelSpec`. P4's other first-class surface; the ones at the spec/driver face carry
   `Settings + Builder`, the ones on the bind path carry `sized::`-seeded consts (§A.6).
3. **`Step` and its four typestate structs** — a sans-IO FSM. `Pipe::call` is one
   `In -> Result<Out, Err>`; these transitions take different argument types and are driven an
   arbitrary number of times by a caller who owns the loop. Precedent verbatim:
   `proxima-primitives/src/pipe/sans_io.rs:41-52`.
4. **`MetalDriver`** — a resource owner for `!Send` device handles. Its behaviour (`Submit`) is a
   pipe; `poll_complete` is the P20 poll surface and is **not** claimed to be a pipe (**cB #7**).
5. **Newtype ids** — `TokenId`, `SlotId`, `SymbolId`, `KernelId`, `SubmitTicket`, `PlanKey`.
   P11's compile-time-correctness clause; `NodeId` (`op.rs:26-28`) is the shipped precedent.

Nothing here is a wrapper whose call site reads identically to what it wraps; the eleven that
would have been are listed as abandoned in §E.2 with the call site that killed each.

## E.4 Where information is destroyed today, and what restores it

| destroyed | site | restored by |
|---|---|---|
| capability SET → `bool` | `bind.rs:2639` | `FusionRules` (four independent fields) |
| band static-vs-dynamic → `operands.len() == 8\|9` | `bind.rs:237`, `cpu.rs:4847`, `msl.rs:2030`, `msl.rs:2543` | `BoundOffset::{Static, Symbol}` |
| band bounds → `i64::MIN`/`MAX` sentinels | `bind.rs:2510-2511`, `msl.rs:2530-2534` | `BoundDomain` |
| emitter routing → substring grep of MSL | `metal.rs:1631-1692` | `KernelSpec { family }` |
| hazard identity + scope → clear both sets, all buffers | `metal.rs:874-877`, `1105` | `BarrierSet { slots }` + `memoryBarrierWithResources` |
| which root is which → `(NodeId, NodeId, NodeId)` | `spec.rs:2333` | `KeyRoots` / `CacheRoots` |
| an integer → `f32` | `generate.rs:2323` | `SymbolId` in the uniform block (no buffer at all) |
| operand strides → eight literal tuples | `bind.rs:2451-2474` | `Layout::strides`, read by `render_loop` |
| step counters → a "read once" protocol in a comment | `metal.rs:2719`, `generate.rs:2468` | `StepOutcome` owns `StepCounters` by value |
| feature-off vs out-of-range → one `None` | `metal.rs:421-432` | the field exists unconditionally in `Schedule` |

## E.5 Teaching surface (P2), FORCED-BY cA #29

Every public type above names the primitive it composes in its doc comment and why the wrapper
exists: `Stage` names `ComposedBody` and the side-table rule; `HalfSpace` names `AxisIndex`;
`Domain` names `Op::Iota`'s mask composition and says when to write a domain versus a mask;
`Session` names `LoadedModel::call`; `Submit` names `Pipe` and `sans_io.rs:41-52` for why the
FSM is not one. CARD B4 and CARD A4 each ship a runnable walkthrough driving every legal
transition and every rule application, which is the P11 clause neither design supplied for both.

---

# F. Worked example, which is the test (P17)

Two tests. The first is the structural claim — **any program with the structure fuses** — and it
cannot pass on main, because `bind.rs:2465-2474` compares operand strides against eight literals.

```rust
/// Two programs computing the SAME banded softmax-weighted reduction, differing only
/// in memory layout: keys head-major, and keys position-major (a transposed cache).
/// Both must bind to ONE `Loop` with five stages and identical results, because the
/// rule reads strides from `Layout` and never compares them to a literal.
#[proxima::test]
#[case::head_major(KeyLayout::HeadMajor)]
#[case::position_major(KeyLayout::PositionMajor)]
async fn a_banded_softmax_reduction_fuses_at_any_operand_layout(#[case] layout: KeyLayout) {
    // arrange — real openchat-3.5 dimensions: 8 kv heads, group 4, head_dim 128,
    // 34 cached positions, 1 new token. The causal band is DECLARED, not matched.
    let program = banded_softmax_program(layout);
    let shapes = shape::infer(&program, &[SEQUENCE, MERGED_LENGTH]).expect("infers");

    // act
    let bound = bind::with_rules(&program, &shapes, &[OUTPUT], FusionRules::DEFAULT)
        .expect("binds");

    // assert — one bound op, five stages, no duplicated operands, no sentinel
    assert_eq!(bound.len(), 1, "the whole chain is one Loop at any layout");
    let BoundOpKind::Loop { stages, operands, .. } = &bound[0].kind
        else { panic!("attention must be a Loop, not a named kind") };
    assert_eq!(stages.len(), 5, "score, max, exp-sum, value-sum, divide");
    assert_eq!(operands.len(), 3, "Q, K, V — the cache length is a symbol, not an operand");
    assert_eq!(stages[1].domain.constraints.len(), 1, "one causal half-space");
    assert!(matches!(stages[1].domain.constraints[0].offset, BoundOffset::Symbol(_)),
            "the live length is a uniform read, not a compiled-in sentinel");

    // assert — the fused form equals its own unfused form, evaluated by the SAME
    // evaluator. Today `cpu.rs:4830` is a second hand-written implementation, so a
    // parity test compares two implementations rather than one rule against itself.
    let fused = cpu::evaluate_with_rules(&program, &shapes, &blocks, FusionRules::DEFAULT)
        .expect("fused runs");
    let plain = cpu::evaluate_with_rules(&program, &shapes, &blocks, FusionRules::NONE)
        .expect("unfused runs");
    assert_close(&fused, &plain, 1e-6);
}
```

The second is the allocation contract in §B.5, which asserts both zero allocations and N == 100
steps, because a zero-step run and a zero-allocation run are otherwise the same signal.

Neither test can establish that this is the right shape — green means does-what-it-does
(**cA #30**). The shape argument is §A.4's minimality (with the `Lookup::extent` candidate
examined, per **cA #8**) and §E.3's lint, and it is falsifiable by the list in §E.3, not by a
passing test.

---

# G. What this synthesis took from where

| decision | taken from | over | reason |
|---|---|---|---|
| `BoundOpKind::Loop { stages }` | B §A.3 | A's `FusedRegion` | `bind.rs:160-162` rejects the recursive tree by name (cA #10) |
| `IndexPattern::compose` + strided extents | B §A.1-A.2 | A had no answer | RoPE/GQA fusion has no weaker operation than substitution |
| band DECLARED as `Domain` | A §A.2 | B's R4 matcher | a peephole on one mask spelling reproduces audit item 1 (cB #19) |
| `HalfSpace = AxisIndex` | cA #7 | A's new struct | reuse-first; removes the address/constraint drift |
| live length as `SymbolId` | A §A.2/C.3 | B's `BandBound::Dynamic` operand | no integer buffers exist (`map.rs:99-107`, cB #10) |
| FSM authoritative, 3 pipes | cA #5, #6, cB #20 | both designs' parallel encodings | nine stages failed the second binary question |
| `poll_complete`, not a blocking future | B §B.4 | A §B.2 | `waitUntilCompleted` parks a thread in async (cA #12) |
| POD-only pipe boundaries | cA #4, cB #6 | both | `Pipe` has no lifetime params (`primitives.rs:91-101`) |
| `ProgramSpec::extend`/`stack` | B §C.1 | A's `lower_into` | keeps lowering in `TryFrom` |
| `KeyRoots` enum, `MaskSource`, `HeadNorm` | A §C.2 + B §C.2 | either alone | exhaustive match at 15+ sites; no `Option` as a flag |
| group = 256 (Q4_K superblock) | cB #4 + cA #9 | A's 128 | `ffn_down`'s elided axis is the contraction axis |
| time decomposition from ROW 287 | brief + audit #15 | both designs' ROW 281/282 | the serialized instrument measures a different program |
| feature seam kept for rollback | cA #22, cB #11 | A's rename-and-delete | `bind.rs:2641-2648` already provides it |

## Unfinished

<<UNFINISHED: not covered in the 30-minute window — (1) the `Wiring` type referenced by
`ProgramSpec::extend` is named but not given a field list; (2) `Bindings`, `BindingRange`,
`CommandRange`, `UniformRef`, `UniformPatch`, `BarrierRange`, `Grid`, `Budget`, `TokenWindow`,
`StepCounters`, `Completion`, `ResidentSet` are used in signatures and not defined (cA #24
raised this against A and it is only partly discharged here); (3) `ScatterBounds::{Fault, Drop}`
is carried from A §D.3 as CARD D3b's one IR extension but its interaction with the declared
`Domain` (a dropped scatter is an out-of-domain write) is not worked through; (4) the
sliding-window and ALiBi claims in §C.4 are asserted from the shipped generators and not written
out as programs; (5) no lowering-audit.md or tiers-census.md was present in the input
directory, so the tier claims rest on `Cargo.toml:37` + `lib.rs:103-104` + the four
`scripts/proxima-tensor-gate.sh` cells named in cB #9, and the per-crate tier matrix is not
enumerated>>
