# Algebra monomorphization — plan

Status: proposed. Owner asked for the whole algebra genericized so any combination of the numeric
axes is expressible, with missing combinations failing at COMPILE time rather than silently
degrading to a slow-but-correct body.

No commits without owner authorization.

## The defect this removes

The emitter layer currently selects a kernel body with chained runtime predicates over the bound
op's numeric configuration, and terminates in a correct-but-slow fallback:

```
plain_product && codec_supports_pair_dot && dtype == F32 && is_plain_product_reduce
  -> q4k_pair_dot ...
use_ggml_port = plain_product && codec in {Q4K,Q5K,Q6K} && feature && !split_k
use_q4_0_native = codec == Q4_0 && dtype == F32 && is_plain_product_reduce && feature && !split_k
use_single_fetch = codec == Q4K && !plain_product && feature && !split_k
else  -> the generic lane-spread preamble + per-codec arms
```

Consequences, measured:

- a missing (codec x epilogue) cell is a silent fallthrough to a per-element decode;
- `PACKED_ROW_BLOCK_SIMDGROUPS` and the whole `[packed_row_split_k]` section are inert on this
  path (`tiled_gemm_cooperative_scan.rs:1925` returns before `:1929`), so two build-time sweeps
  measured an unchanged binary;
- `metal-q4_0-native` measured as making no difference at all, because its predicate excludes every
  op our fused graph produces;
- end result: our Q4_0 matvec decodes per element while doing MORE per dispatch (fused epilogue),
  and is ~10% behind llama's kernel end-to-end with no test able to see it.

The fix is Rust monomorphism: one emitter per structural family, parameterized over the numeric
axes as types with associated consts, so the compiler folds the same constants the bespoke bodies
were hand-writing -- with the cell matrix as a compile-time product instead of a runtime selector.

## Axis inventory (the "any combo" space)

| axis | variants today | where it is decided |
|---|---|---|
| byte layout | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K..Q6_K, IQ4_NL, F16, BF16, F32 | `Codec` + per-codec arms |
| activation dtype | f32, f16, bf16 | `element_type` token |
| accumulator dtype | f32 | implicit |
| op shape | token-major, feature-major, multi-row fold group | `packed_row_dispatch` |
| rows per simdgroup | 2, 4, 6K:2, config | `PACKED_ROWS_PER_GROUP`, `q6k_rows` |
| lane spread | lanes_per_block 8 (K-quant), 2 (Q4_0 pair), 1 | per-body |
| reduction topology | one simdgroup, nsg 1..16, split-K 1..8 + combine | `[packed_row_block]`, `[packed_row_nsg]`, `[packed_row_split_k]` |
| epilogue | Identity, Plain, Broadcast, Scatter | `epilogue_body`/`_operands`/`_broadcast_axes`/`out_scatter` |
| twin output | single, twin | `ElementwiseTwin` |
| binding | resident mapping-offset, uploaded, interleaved, grouped planes | bind + `KvBufferSource` |
| grid form | flat 2D, vector-blocked, wide cooperative | `grid_threads`, `wide_grid` |

Every combination of the first eight axes is currently a cell that either has a hand-written body
or silently falls back.

## Target shape

```rust
trait ElementSpec   { const DTYPE: DType; const TOKEN: &'static str; }
trait PackedLayoutSpec {
    const BLOCK_BYTES: usize;
    const BLOCK_ELEMENTS: usize;
    const SCALE: ScaleSpec;      // F16At0 | F32At0 | DMinPair { .. } | ...
    const PAYLOAD: PayloadSpec;  // NibblesAt2 | NibblePlanes { .. } | KQuantQlQh { .. }
    fn decode_block(blk: &[u8], slot: usize) -> f32;   // the byte math, per layout, ONLY
}
trait EpilogueSpec { const SHAPE: EpilogueShape; /* body applied once per output element */ }
trait SplitSpec    { const SPLIT: usize; const COMBINE: CombineKind; }
trait BindingSpec  { const RESIDENT: bool; const LAYOUT: BufferLayout; }
trait GridSpec     { const LANES_PER_BLOCK: usize; const ROWS_PER_SIMDGROUP: usize; }

fn emit_packed_matvec<L, E, S, B, G>(source: &mut String, ..)
where L: PackedLayoutSpec, E: EpilogueSpec, S: SplitSpec, B: BindingSpec, G: GridSpec;
```

Rules:

- **no catch-all.** The dispatch is `match (layout, epilogue, split)` over the product of the
  impls. A new codec or epilogue shape fails to compile at every site that must handle it.
- **no provenance in the type.** the former `metal-q4k-ggml-port` (now rendered) / `metal-q4_0-native` become alternative
  impls of the same cell; the feature picks which one the arm instantiates.
- **no codec-meaningful parameters.** A parameter whose meaning depends on another parameter's
  value is a special case: it moves inside the arm (applied to `select` already).
- **admission is a capability, not a fallback.** `capability(layout, epilogue, split) ->
  Capability` is total; `Capability::Unsupported { missing }` is rejected at BIND with a typed
  error naming the missing cell. Runtime never chooses a slower path.
- **one `decode_block` per byte layout remains.** Q6_K's ql/qh/scale plane genuinely differs from
  Q4_0's single nibble plane. What is shared is the emitter: loop, lane spread, split/combine,
  barriers, epilogue write.

## Acceptance criteria (each names a count)

- **AC1 (matrix coverage).** A test enumerates every admitted cell
  `codecs x {Identity, Plain, Broadcast, Scatter} x split {1,2,4,8}` and asserts the observed
  count equals the expected count, prints the table, and executes each cell on device against the
  CPU evaluator (`evaluate`) with max relative error <= 1e-5. Expected count is fixed by the
  inventory at the time of the slice; the test asserts that exact integer.
- **AC2 (no catch-all).** A source-scanning test asserts **0** occurrences of `_ =>` /
  `unreachable`/silent fallback arms inside `omega/src/msl` emitter dispatch points, and **0**
  occurrences of the names of deleted bespoke bodies.
- **AC3 (bind rejection).** For each `Unsupported` cell in the inventory, a test asserts the bind
  returns a typed error whose message names the missing capability; count equals the number of
  Unsupported cells (which must be 0 for the shipped feature set, and non-zero only under a
  reduced feature set).
- **AC4 (output identity).** Every migration slice keeps `text_hash` identical on the fixed 4k
  prompt and the 32k prompt.
- **AC5 (perf non-regression).** Each slice reports the steady-median A/B on a quiet machine
  (`decode_steady_median_ms`), and the value must stay within the pre-slice band. Current band:
  4k 10.79-10.99 ms.
- **AC6 (bisect).** Every commit builds `--features metal` and passes AC1 + the lib suite.

## Slices

Each slice is standalone, green, and revertable. Validation command names its count.

| # | slice | validation |
|---|---|---|
| S0 | capability matrix generator: enumerate codec x epilogue x split, print the coverage table, assert unit test `cells == <inventory count>` | `cargo test -p proxima-omega --lib algebra_cells --features metal` |
| S1 | `ElementSpec`/`PackedLayoutSpec` traits + `Q4_0Layout::decode_block`, verified against `proxima_gguf::quant::q4_0::dequantize_block` on random blocks and a real checkpoint tensor | `cargo test -p proxima-omega --lib packed_layout_spec --features metal` (assert N block cases) |
| S2 | `emit_packed_matvec` wired for the `(Q4_0, Identity)` cell; existing bespoke Q4_0 arms still default | AC4 + AC5 on 4k |
| S3 | `Q4KLayout`, `Q5KLayout`, `Q6KLayout`; delete `push_q4k/q5k/q6k_ggml_port_body` and their selection arms | AC2 asserts those 3 names absent |
| S4 | flat codecs Q5_0/Q5_1/Q8_0/Q4_1 as impls; these cells appear with no new emitter code | AC1 count increases by 4 x epilogue x split |
| S5 | `EpilogueSpec::{Plain, Broadcast, Scatter}`; delete the `plain_product`/`is_plain_product_reduce` predicates and the fusion-driven body switching | AC1 count includes all 4 epilogue shapes |
| S6 | `SplitSpec` + `CombineKind`; delete the split-K branches from every body | AC1 covers split {1,2,4,8} |
| S7 | `BindingSpec` + `BufferLayout`; fold the interleaved-vs-grouped planes work (`msl/q4_0_planes.rs`) into it | AC1 covers both buffer layouts |
| S8 | same treatment for the remaining families: tiled gemm, cooperative reduce, elementwise/iota/constant, cached attention (3 forms) | AC1 extended per family; AC2 covers `omega/src/msl` |
| S9 | delete the selector entirely: dispatch becomes `match (layout, epilogue, split, binding)` | AC2 = 0 fallback arms; AC3 = 0 Unsupported under `metal` |

## Sequencing constraints

- S1 before S2 (an impl with no oracle is a guess).
- S3 after S2 proves the cell matrix executes (deleting three bodies before one generic body works
  is a straight regression with no replacement).
- S5 after S3 (epilogue generality is only meaningful once a single layout is generic).
- S7 independent of S5/S6 but shares the descriptor with the repack work; land whichever is green
  first and rebase the other.
- S8 does not block S9 for the packed families; S9's AC2 covers msl broadly, so S8 must finish
  before S9 seals.

## Rollback

Every slice lands behind an existing or new default-off feature. Its bespoke body is deleted only
in the slice where AC1 passes for the replacement, so a revert restores a working binary at every
commit.

## Review v2 — corrections that supersede the sections above (2026-11-08)

### Blocking — the plan as first written would fail

1. **Trait seam (`decode_block -> f32`) is wrong; use a sub-block fold.** The bespoke bodies
   amortise deliberately (`q4k_run8` 8 levels per decode, Q4_0 masked-word 4 nibbles with no
   runtime shift, algebraic zero-point fold). Per-element trait calls discard that and produce a
   generic body slower than the bodies they replace. Required:
   `const LEVELS: usize; fn fold_levels(blk: &[u8], base: usize, out: &mut [f32; LEVELS])`.
2. **The cell space is not a Cartesian product.** `Scatter` is a property of a data-dependent
   `out_map`; `split` is bounded by rows and grid form. Enumerating `codecs x epilogues x splits`
   creates impossible cells and forces special cases back into the test. Derive the legitimate
   space from `BoundOpKind` legality plus explicit constraints, enumerated IN CODE.
3. **`Unsupported` at bind has no seam.** Bind lives in `proxima-tensor/src/bind`, the impls in
   `omega`, and rejection surfaces at emit today. Needed: a `BackendCapabilities` table the bind
   queries, generated from the same enum as the arms so they cannot drift. Without it AC3 is
   unmeetable and the "fail at startup" promise is unimplementable.

### Material

- **Addressing is structural, not numeric.** A lane owns half an 18-byte block for Q4_0
  (`il = (lane%2)*8`) but a 32-element sub-block for K-quants (`ix = lane/8`):
  `PayloadSpec` needs `PairHalf | SubBlock`, not `LANES_PER_BLOCK: usize`.
- **Ragged handling is an axis, currently missing.** Feature extent not divisible by
  rows-per-simdgroup; reduction not divisible by block elements. Add to the axis table and to AC1
  as mandatory cells, or the matrix passes on multiples while real shapes are wrong.
- **AC2 must count arms, not grep text.** `grep -rn '_ =>' omega/src/msl` returns 36 hits in 11
  files, 15 in dispatch-shaped code. Replace with: dispatch arm count == AC1 cell count, and no arm
  routes two distinct cells to one body.
- **AC5 needs a per-cell twin.** End-to-end hides in-cell regressions; each migrated cell gets its
  own median + CoV, and >2% beyond CoV keeps the bespoke body rather than sealing the slice.
- **Instantiation cost is a first-class risk.** Up to ~1152 distinct MSL instantiations/pipelines;
  step 0 already shows 47 pipeline misses / 76 ms compile. Instantiate reachable cells only, and add
  a pipeline-count + compile-ms AC per slice.
- **One feature, not nine.** `metal-algebra-monomorph` for the whole programme, deleted at S9.
  Feature sprawl is how `metal-q4k-split-k` (inert) and `metal-q4k-single-fetch` (unreachable)
  shipped dead. (`metal-q4k-single-fetch` was removed entirely under PROXNUM-317.)
- **The instrument moves with the bodies.** `classify_kind` buckets by emitted source markers, so
  unifying bodies silently re-buckets every measurement (twice today). Update it in the same commit.
- **S5 re-decides the fusion default from data.** Fusion currently measures a 2.8% LOSS.

### Sequencing

- Split S8 per family; move **cached attention earlier** than the packed work — `CachedAttentionForm`
  is already a form enum and is the cheapest proof the target shape works.

## Review v3 — supersedes v2 where it conflicts (2026-11-08)

### Decisions changed, not wording

1. **Do NOT delete the ggml-port trio (supersedes v2's "delete the trio").** They are the only
   reference implementations for the K-quant cells, and deleting them removes the comparison arm
   every later epilogue/split slice needs. The correct endpoint is provenance leaving the *selector*,
   not the code leaving the tree: they become `Reference` impls of the SAME cell (behind one
   feature), never selectable in a default build. AC then asserts they are not instantiated
   (pipeline count / arm count), NOT that their names are absent. This is the reconciliation of
   "remove bespoke paths" with "keep a reference": they stop being *paths*; they stay *impls*.

2. **Tolerance is per split, not 1e-5 for everything (supersedes AC1's single threshold).** Split-K
   changes reduction order, so `split>1` cannot be bit-exact against a serial CPU fold by
   construction. AC1 becomes: `split == 1` cells compare BITS (0 differing bits); `split > 1` cells
   compare within the numeric policy's documented bound and must be gated on the same
   `NumericRewrite` the policy already has for split/merge. The test asserts a COUNT of cells in
   each tier, so a cell silently downgraded to the loose tier fails the count.

3. **The coverage denominator is `BoundOpKind`, not the packed family (supersedes "entire algebra"
   being implicit).** The plan must enumerate every `BoundOpKind` variant and state, per variant,
   which emitter family owns it. `Keep::Scan` is a distinct axis from `Keep::Reduce` (a scan is not
   tree-reducible and cannot share the fold body), and `RoundBatchedReduce`, `MoeTopK`,
   `TopFractionSelect`, `GatedDeltaNet` currently have their own emitters that S8's "same treatment"
   hides. AC1's count must be derived from that enumeration, so "entire" has a number.

4. **Descriptor crate boundary must be stated (B3 depends on it).** `PackedLayout`, `EpilogueShape`,
   `SplitSpec` *identity* types live in `proxima-tensor` so bind can answer capability; the const
   data and folds live in `omega` as impls. Without this placement the bind cannot see the cell and
   B3's capability table has nothing to key on.

5. **TTFT is a first-class AC, not just pipeline count.** Every instantiation is a pipeline compile
   on first use; step 0 already shows 76 ms of compile. Add: TTFT on the fixed 4k prompt must not
   regress beyond the session band, asserted per slice alongside the steady median.

6. **Feature granularity: max 4, one per family, never correctness-bearing.** One feature for six
   families makes every slice all-or-nothing to revert. Cap at one feature per emitter family
   (packed / attention / tiled+grouped / elementwise+leaf), each default-off, each affecting
   performance only — a feature whose absence changes correctness is a design error, not a flag.

### Still open — not resolvable by review, only by measurement

These are recorded rather than fixed; each is a slice-time decision with a named measurement:

- whether monomorphizing actually closes part of the 10% core gap, or only makes it
  correct-by-construction (measured body/geometry effects so far: 1.3% / 2.8% / 0.0%);
- whether the per-cell median for the generic body beats each bespoke body (M4's stop rule decides);
- whether instantiation count stays under a pipeline/compile budget that TTFT can tolerate;
- whether epilogue fusion should default on (currently a 2.8% loss, re-decided at S5);
- the size of the `BoundOpKind` denominator once enumerated in code (S0 produces the integer).

A plan cannot be fully settled against unknowns; these five are the deliberately-unsettled set, and
each has the measurement that settles it attached.

## Review v4 — convergence pass (2026-11-08)

Precondition **verified**: neither `BoundOpKind` (`bind/types_layout_boundop.rs:146`) nor `Codec`
(`proxima-primitives/src/codec.rs:18`) is `#[non_exhaustive]`, so exhaustive matching across the
matrix is legal and "missing cell = build failure" is implementable rather than aspirational.

Five additions, all cheap:

1. **Closure AC (adding a codec must break something).** `fn layout(Codec) -> PackedLayout` is an
   exhaustive match, so a new `Codec` already fails to compile. Additionally assert the layout
   table's entry count EQUALS the enumerated codec count, so a table edited out of step with the
   enum fails a count rather than silently defaulting.
2. **Cross-split consistency AC.** For each cell, `split in {1,2,4,8}` must agree with each OTHER
   within the loose bound. This is the check that catches a split arm summing the same full
   reduction `split` times — a bug this repo has actually shipped once (recorded in
   `push_q4k_single_fetch_body`'s doc: "~7x too large"; that body was removed under PROXNUM-317).
3. **CI tiering.** AC1 over the full matrix is a device test in the hundreds of cases; requiring it
   per commit makes AC6 impractical. Split it: `AC1-fast` = one cell per layout per commit,
   `AC1-full` = the whole matrix nightly. Both name their counts.
4. **Re-baseline per slice.** Changing the emitter changes kernel identity, hence plan/pipeline cache
   keys, hence comparability with every saved bench baseline. Each slice records the new baseline
   with its kernel identity, or a later A/B compares two different kernels and reads as a delta.
5. **Every family needs the same AC triple** (coverage count, arm-count equality, per-cell median) —
   "monomorphized" is defined per family by those three, not by the absence of a bespoke function.

### Convergence

Three passes have moved from *decisions* (v2: trait seam, matrix legality, bind seam; v3: reference
vs delete, tolerance tiers, coverage denominator, crate boundary, TTFT, feature granularity; v4:
closure, cross-split, CI tiering, re-baselining) to *test details*. A fourth pass would produce
wording, not decisions. What remains is the deliberately-unsettled set from v3 — five questions that
only S0/S2/S3 measurements can answer — plus these v4 test refinements. The plan is converged
against what reading the code can establish.

## What this does NOT fix

- the ~10% core gap vs llama is a step-structure question (618 dispatches, 842 uniform writes,
  ~250 cooperative reductions); monomorphization does not change dispatch count, it only makes the
  bodies correct-by-construction and deletes duplication. The measured body/geometry effects were
  1.3% (launch width), 2.8% (body choice), 0.0% (reference body).
- the KV band being ~3.8x llama's at 4k; separate defect, larger at 32k.
