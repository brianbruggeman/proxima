# judge-r2-1 (round 2)

Mapping: Plan-1 = design-AB.md (round-1 incumbent), Plan-2 = design-B2.md, Plan-3 = design-AB2.md.
Read: design-task.md, all three candidates, critique-AB.md, dispatch-census.md, tiers-census.md
(headings), byte-levers-probe.md pointers via candidates. Repo reads at
/Users/brianbruggeman/repos/slot-0/proxima (read-only, no cargo): `grep -rn BlockCodec` over the
whole repo = **0** rows; `proxima-primitives/src/pipe/primitives.rs:91-102` (`type In; type Out;
type Err: Debug + 'static; fn call(&self, input: Self::In) -> impl Future<..>`, no lifetimes, no
Send); `proxima-tensor/src/lib.rs:193-194` = `#[cfg(feature = "std")] pub mod cpu;` with
`QuantizedBlock<'a>` at `cpu.rs:3091`; `shape.rs:208-244` `unify_iteration_space` resolves an
extent only under `[term] && term.coeff == 1 && axis.offset == 0`; `map.rs:95-110` (no integer
buffers, 2^24 gather ceiling); `bind.rs:2500-2512` (`cached_key_rows = new_key_rows =
key_shape[0]`, `cached_lower_inclusive: i64::MAX`).

## Ranking

`[Plan-3, Plan-2, Plan-1]`

## Scores (0-5)

| axis | Plan-1 (AB) | Plan-2 (B2) | Plan-3 (AB2) |
|---|---|---|---|
| risk surface | 3 | 3 | 4 |
| ordering | 1 | 3 | 5 |
| rollback | 4 | 3 | 5 |
| missing steps | 2 | 3 | 2 |
| hidden coupling | 3 | 3 | 4 |
| observability | 4 | 5 | 4 |
| scope discipline | 3 | 5 | 3 |
| pipe questions | 4 | 5 | 5 |
| no_std tier compliance | 2 | 4 | 5 |
| RISC claim | 4 | 3 | 5 |
| bytes arithmetic | 2 | 4 | 5 |
| quality gates / kill criteria | 4 | 5 | 4 |
| signatures type-check | 1 | 4 | 5 |
| abandoned designs justified | 5 | 5 | 3 |
| **total** | **42** | **55** | **59** |

## One line per cell

**Plan-1 (design-AB).**
- risk 3: `loop-fusion` feature seam beside `cached-attention-streaming` keeps the highest-risk card bisectable; CARD A2 ports 5 emitters + CPU in one card (74+65+39+22+19+16+12+8 `BoundOpKind::` sites, its own count).
- ordering 1: §0.3 ("the correction that reorders everything", AB:57-86) asserts a 5.96 ms host residual; `dispatch-census.md:25-30` says wall 28.82 = gpu_exec 27.52 + host 1.30 and "gap 11.4 ms is GPU-side" — AB:1168-1170 makes that void term the explicit reason 13 refactor cards precede the first byte lever, and CARD B4's gate reads "against the 5.96 ms MEASURED baseline", a gate that can neither pass nor fail.
- rollback 4: per-card feature selection instead of `git revert`, deletion deferred to A6 after two measurement rounds.
- missing steps 2: `kernel_cache_key -> String` (616/token on the plan-**hit** path) is absent from the zero-alloc site list AB:875-886, so gate **A** fails on landing; no card for omega's red `--no-default-features` cell or interop's missing alloc feature; `BlockCodec` must be minted with no card, no variants, no tier.
- hidden coupling 3: `FusionRules` enters `PlanKey`, so four flags become part of cache identity while `KernelId` is never defined and §A.5 relies on key-insensitivity to cache length.
- observability 4: `KernelFamily` returned by the emitter, `classify_kind` deleted, CARD A3 gates that the per-family split reproduces the census 225/391 before fusion lands.
- scope 3: every task section answered at length, but the scope order is set by the void term.
- pipe questions 4: nine plan-time pipes abandoned with the identical call site written out; `FusionRules` (4 bools) fails the second question — a caller cannot add a rule the core does not ship.
- no_std 2: alloc tier claimed throughout while `tiers-census.md` records omega EXIT 101 and interop with no alloc feature; `MAX_INLINE_CONSTRAINTS` gets "spill" and `MAX_INLINE_STAGES` "decline" for the same hazard on the same path, with causal+window = exactly 2 at a default of 2.
- RISC 4: `Loop` is subtractive (5 kinds -> 3), the band is declared not peephole-derived, `Op`/`ScalarOp` do not grow.
- bytes 2: FFN/attn/out/KV reconcile to 4,167.6 and the lever tables are per-lever, but the baseline row is 33.0 (the 7-step mean containing two plan-miss steps) and every route adds a host term derived from the void 5.96.
- gates 4: EM-at-64 thresholds per lever, union-density kill at 0.6, k' kill at 1.5, a random-group control that must fail.
- signatures 1: `#[derive(Copy)] StepOutcome { accepted: ArrayVec<..> }` (arrayvec has `Drop`), `PlanKey: Hash+PartialOrd+Ord` over a `FusionRules` deriving neither, `&[Option<BlockCodec>]` over a type that returns 0 repo-wide, `#[serde(untagged)] Offset` where `Static(i32)` and `Symbol(SymbolId)` are both bare integers so `Symbol` is unreachable from TOML.
- abandoned 5: eight named abandonments each tied to the constraint that killed it (FusedRegion vs `bind.rs:160-162`, `BandBound::Dynamic` vs `map.rs:99-107`, blocking `waitUntilCompleted`, the 128-unit group, the two-card A4 split).

**Plan-2 (design-B2).**
- risk 3: `ReduceChain` lands beside `CachedAttention` with a structural-equality gate before any numeric one, but the 5-emitter surface is one migration step (A-4) with no per-emitter granularity.
- ordering 3: Phase 0 (D-1 ceiling, D-2 union density + acceptance, E-0 control) is measurement-first and correct; Phase 1 then runs 15 structure cards before the first byte lever with no value ranking against the 11.4 ms GPU-side gap.
- rollback 3: §A.5's six steps each carry a parity gate against the step before; no feature seam is named for the emitter port.
- missing steps 3: tiers-census's red omega cells and interop's absent alloc feature are not carded although §B lands the FSM there; the lowering-audit's 28 duplicated emitter functions / `reduce_is_cooperative` x3 get no shared lowering answer.
- hidden coupling 3: `BandBound::Runtime { node: NodeId }` puts the live length back on an operand — the shape that produced `operands.len() == 8|9` — and `Band { axis, lower, upper }` is one interval on one axis, so §C.3's "sliding window = data only" holds only through a masking `Select`, which buys no KV bytes.
- observability 5: `Emitted { source, route, key }` replaces the MSL substring grep, `KernelKey` carries the reduce extents (closing the wrong-kernel-served hole at `msl.rs:930-981`), plus an explicit rich->poor destruction table.
- scope 5: A, B, C, D, E all closed in 1,110 lines; §C names exactly two Rust-required rows and justifies both.
- pipe questions 5: the rule-is-a-pipe composition is written (`FusePrologue.and_then(FuseEpilogue).and_then(ShareAxisChain)`), so the bool is deleted and nothing replaces it; the selector is written in-graph *and* as a pipe; the `Executor` trait is killed by writing both call sites.
- no_std 4: `Cargo.toml:37` + `lib.rs:103-104` drive the C.1 split (std-gated `LoadSpec`, `build.rs`-baked `&'static [Op]`, alloc-tier `plan(&[Op])`) — the constraint visibly changed the design; the red cells are still unaddressed.
- RISC 3: zero new `Op`/`ScalarOp`/`IndexMap` variants is proven and the online-softmax monoid is correctly refused (`op.rs:53-57`), but `ReduceChain` is additive (Elementwise + Reduce + ReduceChain) where `Loop` is subtractive, and `Band` cannot state causal and window jointly.
- bytes 4: per-lever pricing L0-L8 with two rows worked longhand, the `(1-d)^256` ffn_down contraction argument, the union table, and an explicit "not on any number measured today"; but one `B_eff` is applied to the whole 4,167.6 MB while the census gives 173.6 GB/s on the matvec stream and 37-45 on the KV stream, and L0 predicts 26.6 against a measured 28.82 with the 4.05 ms non-matvec term dropped and unreconciled.
- gates 5: per-lever kill criteria, Q re-measured on the whole lossy **stack** after every card, and E-0 is a degenerate control that must read exact-match 1.000 or the metric is void.
- signatures 4: type-check against the opened trait, including the `impl<'p> Pipe for &'p ResolvePlan` form; borrowed `LogitsView<'p>` does cross a pipe boundary (legal, but it is the constraint the other two designed away).
- abandoned 5: five abandonments each with the constraint and the written call site (`ScalarOp` tuple monoid, `Selector` type, `Executor` trait, `Arc<Mutex<MetalSession>>`, `Band { lower: i64, upper: i64 }`).

**Plan-3 (design-AB2).**
- risk 4: the highest-risk surface is split — A2a..A2e is one emitter per card — and CARD A4 keeps online-softmax + cooperative staging in the same card as the deletion (P14).
- ordering 5: the only candidate whose card order is rebuilt from the superseding decomposition ("gap 11.4 ms is GPU-side"), leading with D0/D1/D1b/A0-attn and admitting one host card only because H0 is also a correctness fix.
- rollback 5: per-emitter cards, feature gate, and D0 carries a retraction path — a non-saturating memcpy control re-prices every `@300`/`@400` column in §D.4.
- missing steps 2: §C is not written (asserted from shipped generators), 12 types appear in signatures with no field lists, `ScatterBounds::{Fault,Drop}` (the semantics of the elision lever) is unworked, and cards past the first five are prose — all disclosed in the closing block, but the task names §C a deliverable.
- hidden coupling 4: declares the defect its own shape creates (`Loop` is RISC at the `Op` face and a traversal interpreter at the `BoundOp` face) and adds the shared `lower::LoopWalk` card the lowering-audit's 28 duplicated functions demand.
- observability 4: `KernelKey` POD with extents + `Emitted { family }`; no per-family census-reproduction gate of the kind Plan-1 carries.
- scope 3: 31K against a 1,500-line budget, with §C and most of §E left open.
- pipe questions 5: states the discriminator the other two used implicitly — a stage is a pipe when a caller can supply one the library does not ship — then applies it to admit fusion rules and reject nine plan-time relocations, including rejecting B2's `LoadSpec` on the identical-call-site test.
- no_std 5: the only candidate that cards the tier repair itself (T0 for omega's `--no-default-features` EXIT 101 at `msl.rs:2546`, T1 for interop's absent alloc feature) *before* the FSM lands in interop, and requires every tier gate to name the modules the restricted build compiled (N==0 clause).
- RISC 5: `Loop` 5 kinds -> 3 with `Elementwise` = one `fold: None` stage and B2's `epilogue` subsumed by a trailing stage; `Domain: SmallVec<[HalfSpace; 4]>` states causal and sliding window jointly at the same size, one overflow policy for the whole bind path.
- bytes 5: separates the two bandwidth streams the census conflates (4,033.4 MB / 23.24 ms = 173.6 GB/s weights; 134.2 MB / 3.3 ms = 40.7 GB/s KV) and refuses to mix them; sums 616 ops to 28.59 against a measured 28.82 and names the 0.23 as unaccounted; states the number that hurts first — at 173.6 GB/s the fixed terms (attn 755.0 + KV f16 67.1 + out Q4_K 73.7 = 895.9 MB/pass) exceed the whole 836.5 MB/pass budget, so 3.5 ms misses with the FFN deleted. Residual: the route rows still divide KV bytes by the matvec bandwidth while counting attention time in `non_matvec` (~0.4 ms double count), and per-lever byte rows are bundled into R1/R2 rather than priced individually.
- gates 4: kill criteria cover elision, multi-token, the gathered kernel, the attention arm and the codec cards, and parity is asserted *within* a `RuleSetId` rather than across backends; per-lever quality thresholds are thinner than Plan-2's.
- signatures 5: repairs each round-1 non-compiling form against opened source — `AcceptedRun` is `Clone` not `Copy` (arrayvec `Drop`), `RuleSetId` makes `PlanKey: Hash + Ord` legal, `plan()` drops the codec argument because `grep -rn BlockCodec` = 0 and `QuantizedBlock<'a>` is std-gated and borrowed, `StepCounters` shape no longer changes with a feature flag.
- abandoned 3: the five-decision table carries the reasons, and `Lookup::extent` / `LoadSpec`-as-pipe / `BandBound::Runtime` are each rejected with a stated instrument, but fewer designs are named as abandoned-by-constraint than either other candidate.

## Deciding axis per slot

- **Plan-3 first** — bytes arithmetic: it is the only candidate that reconciles to `dispatch-census.md:25-30` without mixing the 173.6 GB/s matvec stream with the 40.7 GB/s KV stream, names its 0.23 ms unaccounted, and proves the target misses with the FFN deleted; paired with the only tier-repair cards against tiers-census's red cells.
- **Plan-2 second** — scope discipline: every task section closed with per-lever kill criteria and a degenerate control, lost on the RISC claim (`ReduceChain` additive, `Band` a single interval that cannot state causal + sliding window) and on applying one bandwidth across two measured streams.
- **Plan-1 third** — ordering: its self-described reordering correction (5.96 ms host residual) does not exist against the superseding census, and it is the stated reason 13 refactor cards precede the first byte lever; four signatures on the pipe boundary it argues from do not compile.
