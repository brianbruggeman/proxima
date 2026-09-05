# judge-r2-3 (round 2)

Mapping: Plan-1 = design-AB2.md, Plan-2 = design-AB.md, Plan-3 = design-B2.md.
Repo read read-only at `/Users/brianbruggeman/repos/slot-0/proxima`, HEAD ce05362. No cargo.

## Ranking

`[Plan-1, Plan-3, Plan-2]`

- **Plan-1 first** — deciding axis: signatures + bytes arithmetic. It is the only candidate
  whose primary-section model reconciles to the measured wall (23.24 + 4.05 + 1.30 = 28.59 vs
  28.82 MEASURED, 0.8% stated as unaccounted) and the only one with no signature that fails to
  compile.
- **Plan-3 second** — deciding axis: bytes arithmetic. Complete on every lettered section, but
  its ladder model `ms/token = M/B_eff + H` drops the MEASURED 4.05 ms non-matvec GPU term, so
  L0 predicts 26.6 against a 28.82 measured wall and L8's "3.55 ms" is understated by ~1.3 ms
  at A=3 — the row on which the primary question turns.
- **Plan-2 third** — deciding axis: ordering. Its §0.3 anchor (5.96 ms host residual, used 9×)
  is voided by the round-2 dispatch (`dispatch-census.md:25-30`, host = 1.30), and the card
  order is explicitly derived from it (`design-AB.md:1168-1170`).

## Verified against source (not taken from any candidate)

- `grep -rn BlockCodec` over the workspace = **0 matches**. Plan-2's pure `plan(.., codecs:
  &[Option<BlockCodec>], ..)` (design-AB.md:823-829) names a type that does not exist. Plan-1
  drops the argument on that ground; Plan-3 never took codecs.
- `QuantizedBlock<'a>` is `proxima-tensor/src/cpu.rs:3091` under `#[cfg(feature = "std")] pub mod
  cpu;` (`lib.rs:193-194`) — the std-gating both Plan-1 and Plan-3 rely on is real.
- `kernel_cache_key(..) -> Result<String, EmitError>` at `omega/src/msl.rs:930`, single
  production caller `omega/src/metal.rs:3875` (per op). Plan-1 and Plan-3 both delete it;
  Plan-2 never mentions it (`grep -c kernel_cache_key design-AB.md` = 0).
- `arrayvec 0.7.6` (workspace `Cargo.toml:133`). Plan-2's `StepOutcome` derives `Copy` while
  holding `ArrayVec<TokenId, MAX_ACCEPTED>` (design-AB.md:771-781) — not `Copy`. Plan-2's
  `PlanKey` derives `Hash + Ord` (920-927) over a `FusionRules` deriving neither (579-585).
- `Pipe` at `proxima-primitives/src/pipe/primitives.rs`: `type In/Out/Err: Debug + 'static;
  fn call(&self, input: Self::In) -> impl Future<..>`, no lifetimes, no `Send`. All three
  candidates' pipe impls conform.
- `unify_iteration_space` (`proxima-tensor/src/shape.rs:208`) resolves an extent only under
  `[term] && term.coeff == 1 && axis.offset == 0` (:224-227). All three name this correctly;
  Plan-3's line cite is the exact one.
- `IndexMap::Computed`/`scatter` (`map.rs:136`, `:175`), `Keep::Scan` (`op.rs:145`),
  `ScalarOp::Greater` (`op.rs:75`) exist; `map.rs:99-107` states every backend carries indices
  as f32 with a 2^24 exactness ceiling — Plan-3's in-graph selector is expressible as written.
- `tiers-census.md:3-9`: omega `--no-default-features` EXIT 101, omega `--features std` EXIT
  101, `proxima-model-interop` has **no alloc feature**. Plan-1 is the only candidate with
  repair cards (T0/T1) sequenced before it lands the FSM in interop.

## Per-axis scores (0-5)

| axis | Plan-1 | Plan-2 | Plan-3 | one-line reason |
|---|---:|---:|---:|---|
| risk surface | 4 | 3 | 4 | P1 splits the `Loop` port one-emitter-per-card (`crit #9`); P3 stages A-1..A-6 each parity-gated; P2 ports 5 emitters + CPU in one card (A2) against a measured blast radius of 255 sites it itself lists |
| ordering | 5 | 1 | 5 | P1 rebuilt from "gap 11.4 ms is GPU-side", host card only where it is also a correctness fix; P3 runs D-1/D-2/E-0 measurement-first with a kill that reorders; P2 orders §B ahead of every byte lever on a residual that does not exist |
| rollback | 4 | 5 | 3 | P2 keeps the shipped `cached-attention-streaming` seam beside `loop-fusion` so the riskiest card bisects by feature; P1 has per-emitter granularity + one feature; P3 names no seam, only step-wise parity |
| missing steps | 2 | 4 | 4 | P1 has **no §C** (typed `CachedLayerRoots`, `cached_len` as integer leaf, KV-bucket relocation unanswered) and cards past the first five are a sequence, not gated rows; P2/P3 cover A-E |
| hidden coupling | 5 | 4 | 4 | P1 puts `RuleSetId` in the plan key because `ShareAxis` reassociates FP, and adds `A2-lower` (shared `LoopWalk`) against `lowering-audit`'s 28 duplicated emitter fns; P3's rule composition changes FP but its `PlanKey(u64)` does not carry the rule set; P2 carries `FusionRules` in the key but the key does not compile |
| observability | 5 | 4 | 5 | P1/P3 both return the emitter's route with the source and put reduce extents in a POD `KernelKey`, closing the wrong-kernel-served hole (`lowering-audit` finding 1); P2 fixes `classify_kind` but leaves the String key |
| scope discipline | 5 | 4 | 4 | P1 declines the `LoadSpec` pipe and the capability set by writing the call site both ways; P2 keeps a `FusionRules` set it does not need; P3's `Pass` carries the full protocol as one enum, which is more surface than the loop needs |
| pipe questions | 5 | 3 | 5 | P1/P3 write the rule composition (`FusePrologue.and_then(..)`) and delete the bool with nothing replacing it; P2's 4-bool `FusionRules` fails the first question — the composition expresses it |
| no_std tier | 5 | 3 | 4 | P1 makes tier repair a card against the measured red cells and names the modules built (P3's N==0 clause); P3 lets the std-only `config` feature change §C's shape but ships no repair card; P2 states outright it had no tiers-census and reasons from Cargo.toml alone |
| RISC claim | 4 | 4 | 3 | P1/P2's `Loop` is subtractive (5 kinds -> 3) with a **declared** `Domain`; P3's `ReduceChain` is additive (3 kinds) and its `Band` is *derived* from an `Iota/Greater/Select` spelling — audit finding 1's silent-fallback failure mode one level down. Deduct on P1: its decision-1 reason ("`Band` cannot state causal and sliding-window jointly") is wrong — `lower`/`upper` as affine bounds on one axis states exactly that pair |
| bytes arithmetic | 5 | 2 | 3 | Recomputed: FFN 3170.9 + attn 755.0 + out 107.5 + KV 134.2 = 4167.6 (all three ✓); Q3_K FFN ×0.764 = 2422.6 (✓); P1's R1 3318.4/2.18 = 1522.2 -> 8.77 + 1.86 + 0.60 = 11.23 (✓, and it divides the non-matvec term by A); P3's L5 2591.0/2 = 1295.5 -> 7.46 + 0.65 = 8.11 (✓ internally) but every row omits non-matvec, so L0 = 26.6 against 28.82 MEASURED; P2's baseline row is 24.01 + 4.05 + **5.96** |
| quality gates / kill criteria | 5 | 5 | 5 | All three give EM-at-64 thresholds, union-density kills, a gathered-kernel GB/s kill, and a degenerate control that must fail; P1 and P3 both retract the 400 GB/s column if the memcpy arm does not saturate; P3 re-measures Q on the whole lossy stack after every card |
| signatures type-check | 5 | 1 | 5 | P2 has three that cannot compile (`BlockCodec` absent; `Copy` over `ArrayVec`; `Hash + Ord` over a non-`Hash` field). P1 fixes all three by name; P3's `impl<'p> Pipe for &'p ResolvePlan` with `In = Pass<'p>` is legal under a trait with no lifetime params |
| abandoned designs | 4 | 5 | 5 | P2 and P3 each name 5-6 abandonments with the constraint and the call site that killed them (`Arc<Mutex<MetalSession>>` -> owner; tuple monoid in `ScalarOp`; `Executor` trait; `FusedRegion`); P1 names fewer, mostly inherited |
| **total** | **63** | **48** | **59** | |

## Notes that did not fit an axis

- The 2t-for-t defect (`bind.rs:2504-2506` + `msl.rs:2586`) is answered structurally by all three.
  P1 is the only one that also *prices* it as a card with a measured trip count (A0-attn, ~1.65 ms).
- P3's §D.4 conclusion ("not on any number measured today") survives its own arithmetic omission,
  so the direction of the primary answer is the same in all three; only P1's magnitude is defensible
  against the measured wall.
- P1's §F declares the defect it inherits (`Loop` is RISC at the `Op` face and an interpreter at the
  `BoundOp` face) and adds the shared-lowering card for it. P2 states the same cost and declines the card.
