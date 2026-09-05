# judge-r2-2 (round 2)

Mapping: Plan-1 = design-B2.md, Plan-2 = design-AB2.md, Plan-3 = design-AB.md.
Repo read-only at `/Users/brianbruggeman/repos/slot-0/proxima`, HEAD `ce05362`. No cargo run.

## Ranking

`[Plan-2, Plan-1, Plan-3]`

- **Plan-2 (design-AB2) first** — deciding axis: the primary section's measurement basis plus the
  declared-vs-derived band. It is the only candidate whose §D anchor reproduces the census of
  record *and* states the 2-3 ms per-class overlap that forbids subtracting an arm from the
  aggregate, the only one whose band cannot silently fail to fire on an unrecognised mask
  spelling, and the only one that cards the tier cells `tiers-census.md` measures red before
  landing the FSM.
- **Plan-1 (design-B2) second** — deciding axis: RISC claim. Its `ReduceChain` deletes the
  10-field macro-op, but the `Band` is *derived* by recognising a mask, which reproduces audit
  finding 1 one level down, and its "one REQUIRED extension" proof omits the
  `is_identity_projection` relaxation (`bind.rs:1156-1164`, opened) that its own RoPE
  dispatch-count claim needs. Everything else about it is the most complete work in the round.
- **Plan-3 (design-AB) third** — deciding axis: the ordering rests on a retracted number. Its
  §0.3 "correction that reorders everything" is the 5.96 ms host residual; the STEADY-STATE WALL
  block gives host = 1.30 ms. Compounded by three signatures that do not compile and a pure
  `plan()` resting on a type that does not exist.

## Grounded checks run this session (P6)

| claim | check | result |
|---|---|---|
| `BlockCodec` exists (Plan-3 `plan()` signature, §B.3) | `grep -rn BlockCodec --include='*.rs' .` | **0 matches**. `omega/src/metal.rs:445-459` is `packed_operands_of`, mapping `QuantizedBlock` -> `PackedCodec`, not `BlockCodec` -> `PackedCodec` as Plan-3 states |
| `QuantizedBlock` is std-gated | `proxima-tensor/src/cpu.rs:3091`; `lib.rs:193-194` `#[cfg(feature = "std")] pub mod cpu;` | confirmed — an alloc-tier `plan()` cannot take it |
| `kernel_cache_key -> String` runs on the plan-hit path | `omega/src/msl.rs:930`; sole non-test caller `omega/src/metal.rs:3875` inside `encode_op`; comment at `metal.rs:3871-3874` says on a pipeline-cache **hit** ("the steady-decode case") this is exactly what still runs | confirmed: 616 `String`s/token on the path the zero-alloc gate asserts. Plan-1 and Plan-2 card it; Plan-3 mentions it **0** times |
| `ArrayVec` is not `Copy` | `arrayvec 0.7.6` (`Cargo.toml:133`); `impl<T, const CAP: usize> Drop for ArrayVec` at `arrayvec-0.7.6/src/arrayvec.rs:49` | confirmed — Plan-3's `#[derive(Copy)] StepOutcome { accepted: ArrayVec<..> }` does not compile |
| `Pipe` shape | `proxima-primitives/src/pipe/primitives.rs`: `type In; type Out; type Err: Debug + 'static; fn call(&self, input: Self::In) -> impl Future<..>`, no lifetime params, no `Send` | all three quote it correctly |
| fusion blocked on identity maps | `bind.rs:1156-1164` `is_identity_projection` (offset == 0 && coeff == 1); `shape.rs:224-229` same gate in `unify_iteration_space` | **two** distinct gates. Plan-2/Plan-3 name both extensions; Plan-1 names only the extent resolver |
| `ScalarOp` has no rounding | `op.rs:60-77` (17 variants, no Round/Floor); closed-set doc `op.rs:52-56` | confirmed — Plan-1's L6-L8 rows use Q8_0 KV (`c_kv` 0.266) without addressing the write-path quantization; Plan-3 names this and drops the lever |
| thread_locals | `grep -c 'thread_local!' omega/src/metal.rs` = **6** | Plan-3 states the 6-vs-8 discrepancy; Plan-1 states it; Plan-2 says six |
| Plan-2's impossibility arithmetic | 3.5 - (1.5+1.30)/2.18 = 2.21 ms; x 173.6 GB/s = 384 MB/token = 837 MB/pass; untouched terms 755.0+67.1+73.7 = 895.9 | reproduces — but it is impossibility *under Plan-2's held-fixed lever set* (attn at Q4_K, no vocab shortlist), which Plan-1's L7/L8 does not hold fixed |

## Per-axis scores (0-5)

| axis | Plan-1 (B2) | Plan-2 (AB2) | Plan-3 (AB) |
|---|---:|---:|---:|
| risk surface | 4 | 5 | 3 |
| ordering | 4 | 5 | 1 |
| rollback | 3 | 5 | 4 |
| missing steps | 4 | 2 | 3 |
| hidden coupling | 4 | 5 | 2 |
| observability | 5 | 4 | 5 |
| scope discipline | 4 | 5 | 3 |
| pipe questions | 4 | 5 | 5 |
| no_std tier compliance | 4 | 5 | 2 |
| RISC claim | 3 | 5 | 5 |
| bytes arithmetic | 5 | 4 | 3 |
| quality gates / kill criteria | 5 | 5 | 5 |
| signatures type-check | 4 | 5 | 2 |
| abandoned designs | 5 | 4 | 5 |
| **total** | **58** | **60** | **48** |

### One line per cell

**Plan-1 (design-B2)**
- risk surface 4 — phase 0 (D-1 ceiling, D-2 union density/acceptance, E-0 exact-match harness with a degenerate control that must read 1.000) runs before any kernel; the lossy stack is re-gated as a stack after every card from D-6.
- ordering 4 — nr1 fold precedes every lever whose denominator is `A`; host correctly amortized per pass; but nine structural cards (B-1..B-9) precede the bandwidth work that owns the 11.4 ms GPU-side gap.
- rollback 3 — A.5 lands `ReduceChain` alongside `CachedAttention` and flips at step 5, but no feature seam is named for bisection and no card below that flip is separable per emitter.
- missing steps 4 — every task section (A/B/C/D/E) delivered; missing are `IndexPattern::compose` and any repair of omega's `--no-default-features` EXIT 101 / interop's absent alloc feature.
- hidden coupling 4 — thread-locals become `MetalSession` fields, `KernelRoute` is returned not grepped, barriers carry `resources` + `cause`; the KV Q8_0 lever is coupled to "same table" without the write-path rounding question.
- observability 5 — `Emitted { source, route, key }`, profiler groups by enum, and an explicit rich->poor destruction table (E.3) with the restoring construct per row.
- scope discipline 4 — 1110 lines, decisions stated up front, no restated task.
- pipe questions 4 — rules-as-pipes with both binary questions written out, and the elision selector abandoned *with the in-graph program written*; but `LoadSpec: Pipe` is a relocation by its own second question and eight driver pipes is a generous count.
- no_std tier 4 — the tier constraint visibly changed §C (`Cargo.toml:37`, `lib.rs:103-104` -> std-gated loader + `build.rs` bake, bind path provably serde-free); the red cells are not carded.
- RISC claim 3 — 10 fields to 3 and no new `Op`/`ScalarOp`/`IndexMap` variant, but the `Band` is derived from a mask spelling (audit finding 1's failure mode one level down) and the minimality proof covers only the extent holder.
- bytes arithmetic 5 — reconciles to 4,167.6 MB from element counts and codec ratios sourced to `msl.rs:439-455`, separates matvec 173.6 from attn 37-45 GB/s, gives `(1-d)^256` for the contraction axis, the union table, two fully worked ladder rows, and tags every factor MEASURED/DERIVED/ASSUMED.
- quality gates 5 — per-lever metric, threshold and kill (incl. a kernel-GB/s kill independent of bytes), and the stack re-measured after each lossy card because five lossy levers compound.
- signatures 4 — type-check against `primitives.rs`; `Pass<'p>` as `In`/`Out` is legal on `&'p ResolvePlan` but costs the one runtime `WrongState` arm; `cached_len` to `DType::Int32` sits unreconciled against its own "every backend carries every buffer as f32" citation.
- abandoned 5 — five, each named with the constraint that killed it and the call site written both ways (`Selector`, `Executor` trait, `MTLCommandBuffer` in `Encoded`, tuple monoid in `ScalarOp`, `Arc<Mutex<MetalSession>>`).

**Plan-2 (design-AB2)**
- risk surface 5 — D0's memcpy control retracts the 400 GB/s column if it does not saturate; A4 keeps the incumbent attention kernel in the same card as the generic emitter (P14); A2a..A2e is one emitter per card.
- ordering 5 — rebuilt explicitly from "the gap is 11.4 ms GPU-side" after voiding the residual, with a stated rule (no host card precedes a GPU card unless it is also a correctness fix, which H0 is).
- rollback 5 — per-emitter cards, feature gate, and `RuleSetId` in the plan key so two rule compositions cannot share a cache entry.
- missing steps 2 — **§C is absent** beyond decision 4: no `LayerBindings`, no typed `CachedLayerRoots`, no `cached_len` leaf, no `KV_BUCKET_TOKENS` relocation, no "what a new architecture needs" table; cards past the first five are an ordered sequence, not gated rows; 12+ types undefined (self-declared).
- hidden coupling 5 — `plan()` takes nothing codec-shaped (the `QuantizedBlock` std gate is honoured rather than mirrored), codec-dependent kernel choice moves to `omega::schedule`, and the cross-backend numeric contract (parity within a composition, never across two) is stated.
- observability 4 — `KernelFamily` returned by the emitter and extents folded into `KernelKey`; the profiler story is less worked than Plan-1's.
- scope discipline 5 — 429 lines, five contested decisions resolved in a table with one reason each.
- pipe questions 5 — names the discriminator both other candidates used implicitly ("a stage is a pipe when a caller can supply one the library does not ship") and applies it to admit fusion rules and reject `LoadSpec`, `PlanSlots`, `Encode`, `Sample` with the two lines written.
- no_std tier 5 — CARD T0/T1 repair the exact cells `tiers-census.md` measures red (`msl.rs:2546` `.to_string()`; interop's absent alloc feature) *before* B4 lands the FSM, and every tier gate names the modules the restricted build compiled (P3's N==0 clause).
- RISC claim 5 — `Loop` is subtractive (5 kinds -> 3), the `Domain` is declared and binding so it cannot fail to fire, both map extensions are named as required, and `Lookup::extent` is examined and rejected as bounding an address rather than an iteration point.
- bytes arithmetic 4 — anchor reproduces with the overlap caveat and the two bandwidth streams are never mixed; the impossibility result checks out; but `output.weight` top-k/tied and a lower-bit attention codec are named by the task and never priced, so the lever product is not enumerated over every lever.
- quality gates 5 — kill per lever including a control whose failure re-prices the `@300`/`@400` columns of its own table.
- signatures 5 — the `ArrayVec`-`Copy`, `PlanKey: Hash + Ord` over `FusionRules`, `serde(untagged)`-makes-`Symbol`-unreachable and feature-shaped-`Out` defects are each repaired and the correction stated.
- abandoned 4 — `BandBound::Runtime`, `LoadSpec`-as-pipe, the capability set, `Offset::Zero`; less systematically tabulated than the other two.

**Plan-3 (design-AB)**
- risk surface 3 — thorough tripwire list, but CARD A2 ports `BoundOpKind` across the 255 sites it itself enumerates in one card, and the zero-alloc gate fails on landing (see the `kernel_cache_key` row above).
- ordering 1 — the stated cause of the card order ("the host residual is 18% of the token and is the ceiling on §B") is the 5.96 ms figure the census supersedes; host is 1.30 ms.
- rollback 4 — deliberately keeps the `cached-attention-streaming` feature seam so the highest-risk card bisects by feature rather than by revert; nothing below the 255-site card.
- missing steps 3 — the most complete section and card coverage of the three (27 cards, full deletion list, sizing table, teaching surface), against three omissions: `kernel_cache_key`, the red tier cells, and 15 undefined types (12 disclosed).
- hidden coupling 2 — the pure `plan()` rests on `BlockCodec`, which does not exist, and misdescribes `metal.rs:445-459` as its mapping site; CARD B4/C1/C2 land in `proxima-model-interop`, which has no alloc feature, while the text calls the FSM alloc-tier.
- observability 5 — `KernelSpec { family }`, and CARD A3 requires the per-family decomposition to reproduce the census's 225/391 split *before* any fusion lands.
- scope discipline 3 — 1342 lines with substantial provenance restatement (§G duplicates decisions already argued in place).
- pipe questions 5 — nine relocations deleted with the call site that killed each, and the drafter-is-a-pipe / selector-is-not asymmetry decided by a named discriminator (device crossings per pass).
- no_std tier 2 — alloc-tier claims throughout with no card for omega's EXIT 101, and two different overflow policies (`spill` vs `decline`) for one hazard on one path, with the default-2 `MAX_INLINE_CONSTRAINTS` sitting exactly on the boundary its own §C.4 sliding-window example needs.
- RISC claim 5 — `Loop` 5->3, declared `Domain` on the shipped `AxisIndex`, both extensions, minimality argued against `Lookup::extent`, and the SSM/multi-operand-fold boundary named honestly rather than claimed as data.
- bytes arithmetic 3 — reconciles to 4,167.6 MB, but divides the whole stream by the *matvec* bandwidth, sums the arms to 28.06 against a measured 27.04 aggregate without the overlap, and carries the void 5.96 ms host term into every route row.
- quality gates 5 — per-lever EM/ppl thresholds, `k'` reported as a histogram per prompt class, a random-group control that must fail, and a union-density kill at 0.6.
- signatures 2 — three derives verified non-compiling, `untagged` makes `Offset::Symbol` unreachable from the TOML its §C thesis depends on, and three spellings of the barrier concept.
- abandoned 5 — eight, each tied to the constraint that killed it, including one abandoned on P14 grounds (the two-card sequence that would have regressed the hand attention kernel).
