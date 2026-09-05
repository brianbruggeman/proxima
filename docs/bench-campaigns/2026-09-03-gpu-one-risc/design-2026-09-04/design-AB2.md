# design-AB2 — round-2 synthesis (incumbent design-AB.md + blind design-B2.md, under critique-AB.md)

Base pinned: `/Users/brianbruggeman/repos/slot-0/proxima`, HEAD **35a139f** (`git rev-parse --short HEAD`,
run this session). Every `file:line` below was opened at that commit. No build, no cargo, no measurement
was run in this session; every number is tagged MEASURED / DERIVED / ASSUMED with its source.
Provenance of each step: `AB §x`, `B2 §y`, or `crit #n` (critique-AB finding n).

---

## 0. The shape, and the five contested decisions

**Shape.** (i) `BoundOpKind::Loop { stages }` with a **declared** `Domain` of half-spaces on the shipped
`AxisIndex` replaces `Elementwise`/`Reduce`/`CachedAttention` (5 kinds -> 3). (ii) A fusion rule **is a
pipe** `Pipe<In = BoundProgram, Out = BoundProgram>`; a backend's capability is the composition it
builds — there is no capability set type. (iii) `plan()` is pure, alloc-tier, and **takes no codecs**;
codec-dependent kernel selection is `omega::schedule(&Plan, &[Option<QuantizedBlock<'_>>])`, std side.
(iv) The step is one typestate FSM; POD crosses every pipe boundary. (v) Bytes: **3.5 ms/token is not
reachable**, and at the only measured bandwidth it is not reachable *even with the FFN entirely
deleted* — proof in §D.4.

**The five decisions the brief names, each decided with one reason.**

| # | conflict | taken | reason |
|---|---|---|---|
| 1 | `Loop { stages }` (AB) vs `ReduceChain { stages, operands, band, epilogue }` (B2) | **`Loop`** | `ReduceChain` is additive (Elementwise + Reduce + ReduceChain = 3 kinds, one of them new); `Loop` is subtractive (5 -> 3, `Elementwise` = 1 stage `fold: None`, `Reduce` = 1 stage with a `Fold`). And `Band { axis, lower, upper }` is one interval on one axis: it cannot state causal **and** sliding-window jointly, which is exactly 2 half-spaces. `Domain: SmallVec<[HalfSpace; N]>` states both at the same size. B2's `epilogue: Option<ComposedBody>` is subsumed by a trailing `fold: None` stage — one mechanism, not two. |
| 2 | `FusionRules` set (AB) vs rule-is-a-pipe (B2) | **B2's** | B2 §A.4 wrote the composition (`FusePrologue.and_then(FuseEpilogue).and_then(ShareAxisChain)`) and the call sites are **not** identical: a backend that composes rules can insert a rule the core does not ship; a backend handed a 4-bool set cannot. It passes both binary questions where AB's set passes only the first. It also deletes the type `crit #4` proved uncompilable (`PlanKey: Hash + Ord` over a `FusionRules` that derives neither). |
| 3 | barrier schedule: per-point `resources` (B2) vs `barriers: Vec<BarrierSet>` (AB) | **B2's `BarrierPoint`** | AB's `BarrierSet` names the slots and destroys both *where* and *why*; `BarrierPoint { before, resources, cause }` keeps all three. It also collapses `crit #4`'s three spellings (`BarrierSet` / `BarrierRange` / `MAX_INLINE_BARRIER_SLOTS`) into one name plus one sizing const. |
| 4 | spec loader tier (B2 §C) | **B2's tier split, without the pipe** | Take: std-gated loader producing `Vec<Op>`, `build.rs` baking `&'static [Op]` below std (conflaguration tier-2, the `proxima-telemetry` `sized` pattern). Reject: wrapping it in `LoadSpec: Pipe`. Written both ways, `LoadSpec.call(src).await?` and `lower_spec(src)?` are the same line — a relocation (P1 second question). |
| 5 | what pure `plan()` takes, given `QuantizedBlock` is std-gated (`crit #3`) | **nothing codec-shaped** | `grep -rn BlockCodec` = 0 repo-wide: AB rested a signature on a type that does not exist. The shipped spelling is `QuantizedBlock<'a>` at `proxima-tensor/src/cpu.rs:3091` inside `#[cfg(feature = "std")] pub mod cpu;` (`proxima-tensor/src/lib.rs:193-194`, opened). Minting an alloc-tier twin duplicates a variant list. So `plan()` drops the argument: kernel selection by codec is an **emitter** decision and lives where the emitter lives. |

**The discriminator that decides which plan-time things are pipes** (this is what AB and B2 disagreed
about without naming it): a stage is a pipe when *a caller can supply one the library does not ship*.
Fusion rules pass — a backend composes its own set, including its own rule. `PlanSlots`,
`ScheduleBarriers`, `PackUniforms`, `Encode`, `Read`, `Sample`, `BindInputs`, `Advance`, `StopPolicy`
fail: the library ships the only implementation of each and the call site is the identical line, so
they stay plain functions (AB's result, kept).

---

# D. Bytes and time — redone from the steady-state decomposition (PRIMARY)

## D.1 The anchor, which now reproduces

`dispatch-census.md:25-30` is the decomposition of record and it supersedes AB §0.3 entirely
(`crit #1`, `crit #2`): **wall 28.82 = gpu_exec 27.52 + host 1.30** on plan-HIT steps 3..7. There is no
5.96 ms host residual — that number was `33.00 − 27.04` over a 7-step mean containing two plan-miss
steps that pay bind+compile. AB's ordering rationale (`design-AB.md:1168-1170`) rested on it and is
**void**; the card order below is rebuilt from "gap 11.4 ms is GPU-side".

| term | ops | ms | provenance |
|---|---:|---:|---|
| MATVEC | 225 | 23.24 (CoV 0.7%) | MEASURED, census:32 |
| NOT-MATVEC aggregate | 391 | 4.05 (CoV 1.5%) | MEASURED, census:33 |
| host | — | 1.30 | MEASURED, census:27 |
| **sum** | 616 | **28.59** | DERIVED; wall 28.82 MEASURED — 0.23 ms (0.8%) unaccounted |

Per-class arms (attn 3.0-3.6 CoV 9.9%, coop 1.3-1.8 CoV 18%, elem 1.76 CoV 1.1%) sum to 6.1-7.2 ms
against a 4.05 ms aggregate, so **2-3 ms of non-matvec work overlaps in the production command buffer**
and a per-class arm may not be subtracted from the aggregate. Every route row below uses the aggregate.

**Bandwidth reconciliation (`crit #15`).** The census's 179 GB/s divides *all* 4.169 GB by the *matvec*
23.24 ms. KV is streamed by the 32 attention ops, not by the 225 matvecs. Weights-only:
4,033.4 MB / 23.24 ms = **173.6 GB/s MEASURED-derived** on the matvec stream, and
134.2 MB / 3.3 ms = **40.7 GB/s** on the attention/KV stream. Both figures are used, never mixed.

## D.2 The byte table, and the ffn_down granularity that forces the group size

Weights (openchat-3.5 Q4_K_S, Q4_K = 144 B / 256 elt, `omega/src/msl.rs:439`, `Q4K_BLOCK_ELEMENTS = 256`
at `msl.rs:444`, opened): gate+up 2,113.9 MB, down 1,056.96, attn 755.0, `output.weight` (Q6_K) 107.5
= 4,033.4 MB. KV f32 at this context 134.2 MB. Total 4,167.6 MB MEASURED-consistent with the census.

`ffn_down` contracts over the 14336 axis (`PackedRowBlock { weight, other, reduce_dim, codec }`,
`msl.rs:3184-3189`), and the Q4_K superblock is 256 elements **along that same axis**. Unstructured
elision at density d leaves `1-(1-d)^256` of down's blocks live: at d = 0.10 that is 1.000 —
**zero bytes saved**. So the selection granularity is a whole superblock: **G = 14336/256 = 56 groups**,
gate/up elide 256 whole rows per group, down drops one whole block per output row. Group density
`g_u >= d_u` always, and both are measured, never assumed.

Union: a k-token pass reads the union of k row sets. Independent bound `d_u = 1-(1-d)^k`; correlation
pulls it down by an unmeasured amount. Both densities are card D3's deliverable.

## D.3 The model

```
ms/token = [ (gate+up)·c_ffn·d_u + down·c_ffn·g_u + attn·c_attn + out·c_out·s + KV·c_kv + router ] / (A · B_w)
         + non_matvec / A + host / A
```
`A` = accepted tokens/pass (needs the nr1 s-axis fold first, `msl.rs:3171-3200` — without it weights
re-stream k times and A cancels to 1.0). `router` = 4.1 MB/token (56 groups x 4096 x Q4_K x 32 layers).
`non_matvec` = 4.05 MEASURED today; 1.5 ms DERIVED if the attention arm reaches its own byte floor
(134.2 MB / 173.6 GB/s = 0.77 ms) and the 616 -> 264 dispatch ceiling (census:62) lands.

## D.4 What is reachable, plainly

**At 173.6 GB/s, 3.5 ms/token is arithmetically impossible — even with the entire FFN deleted.**
Budget at A = 2.18 with the attention arm fixed: non_matvec/A + host/A = 1.5/2.18 + 1.30/2.18 = 1.29 ms,
leaving 2.21 ms x 173.6 GB/s = **384 MB/token = 837 MB/pass**. The terms that no elision lever touches
— attn 755.0 + KV(f16) 67.1 + out(Q4_K) 73.7 = **895.9 MB/pass** — already exceed it. FFN = 0 still
misses. This is the number that hurts and it is stated first.

| route | MB/pass | MB/token (A=2.18) | @173.6 M | @300 A | @400 A | vs 3.5 |
|---|---:|---:|---:|---:|---:|---|
| today (A=1) | 4,167.6 | 4,167.6 | 23.24+4.05+1.30 = **28.59** | — | — | — |
| R1 lossless+codecs (Q3_K FFN, out Q4_K, KV f16, no elision) | 3,318.4 | 1,522.2 | 8.77+1.86+0.60 = **11.23** | 5.07+1.86+0.60 = 7.53 | 3.81+1.86+0.60 = 6.27 | misses |
| R2 + elision at the kill boundary (d_u .60 / g_u .70) | 2,434.1 | 1,116.6 | 6.43+1.86+0.60 = **8.89** | 3.72+..= 6.18 | 2.79+..= 5.25 | misses |
| R2 favourable (d_u .35 / g_u .45) | 1,828.5 | 838.8 | 4.83+1.86+0.60 = **7.29** | 2.80+..= 5.26 | 2.10+..= 4.56 | misses |
| R2 favourable + attention arm at its byte floor (non_matvec 1.5) | 1,828.5 | 838.8 | 4.83+0.69+0.60 = **6.12** | 4.09 | **3.39** | meets only @400 ASSUMED |

Solving for each requirement independently, holding the others at their favourable value:
- **bandwidth B_w >= 379 GB/s** (95% of the M1 Max spec sheet) — 838.8 / (3.5 − 1.29).
- **A (accepted tokens/pass) >= 2.18**, and the nr1 fold landed first, or the denominator is 1.0.
- **d_u(k=4) <= 0.35 and g_u(k=4) <= 0.45** at the 256-group granularity.
- **attention arm at its byte floor** (3.0-3.6 -> ~0.77 ms), which is not a byte lever at all.
All four, simultaneously. Any one missing and 3.5 is missed.

**Reachable floor under measured conditions:** R1 at **11.23 ms/token** — 2.55x today, and 1.55x
*faster* than llama.cpp's 17.45 (today is 1.65x slower). With the elision stack clearing its quality
gate at favourable densities and the attention arm fixed, **6.12 ms** = 2.85x llama. The owner's 5x
over llama (3.5 ms) is a **bandwidth** question before it is a lever question: 173.6 of 400 GB/s is the
single largest unexplained term and closing it to 300 is worth 4.03 GB x (1/173.6 − 1/300) =
**9.8 ms/token at zero quality risk**, more than every lossy lever combined.

**Kill criteria** (each halts its lever and the sensitivity row says whether the target survives):
`d_u(4) > 0.6` or `g_u(4) > 0.75` kills the elision composition; `A < 1.5` at k=4 kills multi-token;
gathered packed-row GB/s < 0.85x the ungathered kernel kills the elision *kernel* regardless of bytes
(the landed CPU probe already measured the element-granular form at 0.20-0.29 ns/element vs 0.057 DRAM);
attention `gpu_exec_ms` worse than 3.6 + 9.9% CoV kills the `Loop` attention card; exact-match-at-64
< 0.98 (Q3_K/Q4_K codec cards) / < 0.95 (elision) kills that lever; D0's memcpy-shaped control failing
to saturate retracts the 400 GB/s figure and re-prices every `@300`/`@400` column above.

---

# A. Algebra — `Loop`, a declared `Domain`, rules as pipes

```rust
// proxima-tensor/src/map.rs — pure, alloc tier, NO serde on this path.
/// Offset that is known at spec time or bound per call from the same `symbols:
/// &[u64]` slice `shape::infer` already takes. Widens `AxisIndex::offset: i32`;
/// one affine type in the crate, not two (P1 reuse-first).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Offset { #[default] Static(i32), Symbol(SymbolId) }

/// `crit #4`: `Zero` is deleted (it was `Static(0)` under a derived `PartialEq`,
/// two spellings of one value, and the A1 migration gate compares `PartialEq`
/// node-for-node). Serde is EXTERNALLY tagged, never `untagged`: under
/// `untagged` both arms are a bare integer and `Symbol` is unreachable from
/// TOML — in a design whose §C thesis is that a TOML author writes the domain.
#[cfg(feature = "config")]
// #[serde(tag = "kind", content = "value")] on the spec-side mirror only.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SymbolId(pub u16);            // `Extent::Symbolic(u16)` moves to this in the same card

pub type HalfSpace = AxisIndex;          // a half-space IS an affine expr + a sign convention

/// Beyond-rectangular shape of an iteration space. Empty = the full box.
/// BINDING, not advisory: a point failing any constraint is not read, not
/// accumulated, not written. That is what makes KV bytes a property of the
/// program instead of a property of whether an analysis fired.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Domain { pub constraints: SmallVec<[HalfSpace; MAX_INLINE_CONSTRAINTS]> }
```

`MAX_INLINE_CONSTRAINTS` default is **4**, not AB's 2 (`crit #7`: causal + sliding window is exactly 2,
so 2 sits on the spill boundary at the default), and its overflow policy is **bind errors**, matching
`MAX_INLINE_STAGES`'s decline-rather-than-spill — one policy for one hazard on one path.

```rust
// proxima-tensor/src/bind.rs
#[derive(Debug, Clone, PartialEq)]
pub struct Stage {
    pub reduced_axes: SmallVec<[u16; MAX_INLINE_RANK]>,
    pub body: ComposedBody,
    pub fold: Option<Fold>,              // None = a map over the live axes (this is B2's `epilogue`)
    pub domain: BoundDomain,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub struct Fold { pub op: ScalarOp, pub init: ReduceInit, pub keep: Keep }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum BoundOffset { Static(i64), Symbol(SymbolId) }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundHalfSpace { pub coefficients: SmallVec<[i64; MAX_INLINE_RANK]>, pub offset: BoundOffset }
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BoundDomain { pub constraints: SmallVec<[BoundHalfSpace; MAX_INLINE_CONSTRAINTS]> }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum StepArg { Operand(u16), Step(u16), Stage(u16) }

pub enum BoundOpKind {
    Loop { stages: SmallVec<[Stage; MAX_INLINE_STAGES]>, operands: BoundOperands,
           output_axes: SmallVec<[u16; MAX_INLINE_RANK]>, out_layout: Layout,
           out_scatter: Option<Lookup> },
    Iota,
    Constant { value: f32 },
}
```

Flat side table, backwards-only, exactly as `bind.rs:160-162` requires ("a plain index into a side
table, never a `Box<dyn>` recursive tree"). The live cache length is `BoundOffset::Symbol`, never an
operand: `map.rs:99-107` states no backend plumbs integer buffers, so B2's `BandBound::Runtime { node }`
is rejected. Two map extensions, both B2's and both required: `IndexPattern::compose` (substitution —
nothing weaker pushes a non-identity map through a held node) and strided extent resolution in
`unify_iteration_space` (`shape.rs:224-244`), unit projection always winning so existing programs bind
byte-identically.

**Rules are pipes; capability is composition (decision 2).**

```rust
pub struct FusePrologue; pub struct FuseEpilogue; pub struct FuseBroadcastEpilogue; pub struct ShareAxis;

impl Pipe for FuseEpilogue {                     // primitives.rs:91-102: no lifetimes, call(&self, ..), !Send root
    type In = BoundProgram; type Out = BoundProgram; type Err = TensorError;
    fn call(&self, program: BoundProgram) -> impl Future<Output = Result<BoundProgram, TensorError>>
    { async move { fuse_epilogue(program) } }
}
// metal:  FusePrologue.and_then(FuseEpilogue).and_then(FuseBroadcastEpilogue).and_then(ShareAxis)
// wgpu:   FusePrologue.and_then(FuseEpilogue)                      // declines what it cannot render
```

No `FusionRules`, no bitfield, no bool. `fuse_cached_attention: bool` (`bind.rs:2635-2640`) is deleted
and **nothing replaces it**. The plan cache instead keys on a `RuleSetId(pub u32)` the driver supplies —
a newtype id (P11's permitted class), because `ShareAxis` reassociates floating-point and two rule
compositions must not share a cache entry. This is what makes `PlanKey` derive `Hash + Ord` legally
(`crit #4`: it could not over `FusionRules`).

**The cross-backend numeric contract (`crit #12`), which AB owed and did not state:** a rule composition
is part of the program's identity. Parity is asserted **within** a composition (fused vs unfused under
the *same* `RuleSetId`, `assert_close 1e-6` on the CPU oracle), never across two backends running two
compositions. The `X` gate compares each backend to the CPU evaluator *running that backend's own
composition*.

---

# B. Orchestration — one FSM, POD pipe boundaries, one device edge

`Pipe` at `proxima-primitives/src/pipe/primitives.rs:91-102` (opened): `type In; type Out; type Err:
Debug + 'static; fn call(&self, input: Self::In) -> impl Future<..>`. No lifetime parameters, no GATs,
no `Send`. Therefore no borrowed view is ever an `In` or an `Out`: `LogitsView` stays inside the
`InFlight -> Sampled` inherent transition, which is free to borrow.

```rust
pub struct ReadyStep<'p>{ plan:&'p Plan, cursor: Cursor }
pub struct EncodedStep<'p>{ plan:&'p Plan, cursor: Cursor, commands: CommandRange }
pub struct InFlightStep<'p>{ plan:&'p Plan, cursor: Cursor, ticket: SubmitTicket }
pub struct SampledStep<'p>{ plan:&'p Plan, run: AcceptedRun, counters: StepCounters }

#[must_use]
pub enum Step<'p> { Ready(ReadyStep<'p>), Encoded(EncodedStep<'p>), InFlight(InFlightStep<'p>), Sampled(SampledStep<'p>) }

/// B2's shape, taken over AB's `Cursor.accepted: u16` + `StepOutcome.accepted:
/// ArrayVec` + `halt: Option<Halt>` (`crit #15`: three encodings of one
/// quantity). The rejected suffix is kept because destroying it is what forces
/// a downstream reconstruction of what the verifier already knew.
#[derive(Debug, Clone, PartialEq)]
pub struct AcceptedRun {
    pub tokens: ArrayVec<TokenId, MAX_ACCEPTED>,
    pub rejected: ArrayVec<TokenId, MAX_DRAFT>,
    pub cursor: Cursor,
    pub halt: Option<Halt>,
}
```

`AcceptedRun` is `Clone`, **not `Copy`** — `crit #4` is correct that `arrayvec 0.7.6` has a `Drop` impl
and cannot be `Copy`. So the `Session` pipe's boundary contract is restated honestly: **`Out` is
move-only and allocation-free**, not "POD". `StepCounters` is `Copy` and exists unconditionally (its
*emission* is `#[cfg(feature = "instrument")]`, its shape is not — `crit #4`: a public `Out` may not
change shape with a feature flag).

Pipes on the path, and only these: `Draft: Pipe<In = TokenWindow, Out = DraftSpan>` (a caller can swap
n-gram for a draft model), `Submit: Pipe<In = SubmitRequest, Out = SubmitTicket>` (the one device edge),
`Session: Pipe<In = StepInputs, Out = AcceptedRun>` (one call = one pass). `crit #11` is accepted:
`LoadedModel::call` (`proxima-model-interop/src/generate.rs:1542-1556`, `In = (String, usize)`,
`Out = (Vec<u32>, String, bool)`) is a **different granularity and a lossy payload**; AB's claim that
`Session` "is the shape it already has" is withdrawn, and CARD H4 re-faces it onto `AcceptedRun` so the
bare `bool` and the per-call `String`/`Vec` stop existing.

```rust
/// Pure, alloc tier. Takes NO codecs (decision 5): `QuantizedBlock<'a>` is
/// `#[cfg(feature = "std")]` (`cpu.rs:3091`, `lib.rs:193-194`) and borrowed, and
/// `BlockCodec` does not exist (`grep -rn BlockCodec` = 0). Codec-dependent
/// kernel choice belongs to the emitter, so it moves to `omega::schedule`.
#[must_use] pub fn plan(program: &[Op], symbols: &[u64], outputs: &[NodeId], rules: RuleSetId)
    -> Result<Plan, TensorError>;

/// omega, std. Already holds `QuantizedBlock` (`metal.rs:450-456` maps it to
/// `PackedCodec`, `msl.rs:788`), so no crate-graph inversion and no new enum.
#[must_use] pub fn schedule(plan: &Plan, blocks: &[Option<QuantizedBlock<'_>>]) -> Result<Schedule, EmitError>;

#[derive(Debug, Clone, PartialEq)]
pub struct BarrierPoint {                 // decision 3, from B2 §B.3
    pub before: u32,
    pub resources: SmallVec<[SlotId; MAX_BARRIER_SLOTS]>,   // one name; `BarrierSet`/`BarrierRange` deleted
    pub cause: Hazard,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum Hazard { ReadAfterWrite, WriteAfterWrite, WriteAfterRead }
#[must_use] pub fn schedule_barriers(ops: &[BoundOp], residency: &ResidencyPlan) -> SmallVec<[BarrierPoint; 32]>;
```

**`KernelKey` is a POD struct, not a `String` (`crit #5`, and it is a correctness fix as well as an
allocation fix).** `omega/src/msl.rs:930` `kernel_cache_key(..) -> Result<String, EmitError>` is called
at `omega/src/metal.rs` inside `encode_op` once per op per step — 616 `String`s per token on the
plan-**hit** path, which is precisely the path the zero-allocation gate asserts. The same key **omits
the reduce extents** while `cooperative_reduce_width` bakes an extent-derived width into the source, so
`pipeline_for` can serve a wrong kernel from cache. One change closes both: a
`#[derive(Hash, PartialEq, Eq, Clone, Copy)] struct KernelKey { family: KernelFamily, codecs: CodecMask,
extents: ExtentDigest, .. }` carrying the extents. `KernelFamily` is returned by the emitter
(`Emitted { source, family, key }`) instead of recovered by substring-grepping MSL
(`metal.rs:1734` `classify_kind`).

---

# E. Cards — 30-minute slices, one gated deliverable each, ordered by measured value

Standing battery: **P** parity <=1e-4 vs `proxima_tensor::cpu::evaluate` on the real program;
**B** 20 runs byte-identical logits; **T** text identical (never on a card that reassociates FP —
there the gate is `assert_close 1e-6` + EM-at-64 >= 0.99); **M** ms/token and `gpu_exec_ms` not worse
beyond CoV (matvec 0.7%, attn **9.9%** — the attention CoV binds any attention card); **A** allocation
counter == 0 with the **step count asserted** (zero work and successful work emit the same signal);
**G** bytes/token from the driver's own counters against §D.3's prediction, >5% disagreement halts;
**X** the measurement is the deliverable, N and CoV reported.

Ordering rationale, stated because AB's is void (`crit #2`): the gap is **11.4 ms GPU-side**, of which
matvec bandwidth is the largest single term (23.24 ms at 43% of spec) and the attention arm is ~2.5 ms
above its own byte floor. Host is 1.30 ms total, so no host card precedes a GPU card except the one
that is also a *correctness* fix (H0, the wrong-kernel-served hole).

| # | card | 30-min deliverable | gate |
|---|---|---|---|
| **D0** | device streaming ceiling — **branch `perf/device-streaming-ceiling`, `omega/tests/device_streaming_ceiling.rs` exists**; this card *adds the two missing arms*: a **Q4_K-superblock-stride** read (144 B / 256 elt, `msl.rs:444`) beside the existing `float4`/`uint4` arms, and a **memcpy-shaped control that should saturate** | one number per (buffer source x access shape) cell | **X**: GB/s, CoV over 5 runs, N asserted. **Degenerate control: if the memcpy arm does not saturate, the 400 GB/s figure is retracted for this access shape and §D.4's `@300`/`@400` columns are re-priced.** The per-tensor-codec read that AB bundled here is split out as D0b (`crit #8`: D0 was two deliverables) |
| **D1** | matvec roofline ladder — **branch `perf/matvec-roofline-ladder`** | achieved GB/s vs D0's ceiling at 1/2/4/8 rows per activation | **X**: GB/s per arm + CoV; mechanism named for any arm below 0.85x the ceiling (a measurement without a mechanism is rung 1, not a result) |
| **D1b** | packed-row addressing arms — **branch `perf/packed-row-addressing`** | ns/element and GB/s per addressing arm | **X**: which arm, if any, moves 173.6 toward D0's ceiling; **M** unchanged on the production program |
| **A0-attn** | attention arm: measure the 32 ops against their own byte floor (134.2 MB / 3.3 ms = 40.7 GB/s vs matvec 173.6) before any `Loop` work | the mechanism for the 4.3x, from the emitted MSL loop bound at `msl.rs:2586` (`cached_key_rows = new_key_rows = key_shape[0]` at `bind.rs:2504-2506` means 2t trips for t of work) | **X**: measured trips vs t; if trips == 2t the fix is the `Domain` lowering and it is priced at ~1.65 ms |
| **H0** | `KernelKey` POD struct replaces `kernel_cache_key -> String` (`msl.rs:930`), **extents included** (`crit #5`, `crit #6`) | 616 `String`s/token deleted **and** the wrong-kernel-served hole closed | **A** on plan-hit steps with N = 100 asserted; **B**; a regression test that two ops differing only in reduce extent get different keys |

Then, in order: A0 (`IndexPattern::compose` + strided extents) -> A1a (`Offset`/`SymbolId`, map only)
-> A1b (`Domain` on `Op`, `#[serde(default)]`, the 32 struct-literal sites) -> A2a..A2e (`Loop` behind
`feature = "loop-fusion"`, **one emitter per card** — `crit #9`: AB's single A2 was 255 sites and had no
rollback granularity below the whole port) -> B1 (`plan()` pure + `omega::schedule`) -> B2 (one driver,
`grep -c 'commandBuffer()' == 1` from 3) -> B3 (six thread-locals, `grep -c 'thread_local!' == 0`) ->
B4 (`Step` FSM + `Session`) -> A4 (`ShareAxis` + cooperative staging **in one card**: `msl.rs:2586`
already implements rescaled online softmax with threadgroup staging, and landing a generic emitter
first regresses the 3.0-3.6 ms arm — P14, the incumbent wins) -> A5/A6 (epilogue, broadcast-epilogue,
prologue; 616 -> 520 -> 264 MEASURED per card, never projected) -> D2 (nr1 s-axis fold; gate: weight
bytes/pass **flat** in k) -> D3 (union density + acceptance harness: `d_u(k)`, `g_u(k)`, `A(k)` for
k in {2,4,8}; **kills the elision lever before any kernel is written**) -> the codec and elision cards.

**Tier repair is a card, not an assumption (`crit #7`).** `tiers-census.md` records omega
`--no-default-features` EXIT 101 (11 errors, `msl.rs:2546` `.to_string()` with no alloc import) and
`proxima-model-interop` with **no alloc feature at all**. CARD T0 repairs omega's two red cells and
CARD T1 adds interop's alloc feature, both **before** B4 lands the FSM in interop; every tier gate
names the modules the restricted build actually compiled (P3's N==0 clause).

---

# F. What is NOT a pipe, and why each is justified

1. **IR and plan data** — `Op`, `BoundOp`, `Loop`, `Stage`, `Domain`, `HalfSpace`, `Offset`, `Plan`,
   `Schedule`, `BarrierPoint`, `KernelKey`, `AcceptedRun`, `LayerInputs`. Values pipes carry. Each earns
   its place on the second question: `Domain` lets a caller write a banded reduce whose iteration space
   is triangular; `Loop` lets a caller express five folds sharing one traversal; `BarrierPoint` lets a
   caller see *which* resources conflict and *why*, which `MTLBarrierScope::Buffers` destroys.
2. **`Step` and its four typestate structs** — a sans-IO FSM. Transitions take different argument types
   and are driven an arbitrary number of times by a caller who owns the loop; `Pipe::call` is one
   `In -> Out`. Precedent: `proxima-primitives/src/pipe/sans_io.rs:41-52`.
3. **`MetalDriver`/`MetalSession`** — a resource owner for `!Send` device handles, `RefCell` not a
   mutex (a thread-local is a missing owner, not a missing lock, P21 rung 1). Its *behaviour* (`Submit`)
   is a pipe; `poll_complete(&self, step, cx) -> Poll<..>` is the P20 reactor surface and is **not**
   claimed to be a pipe.
4. **`plan()`, `schedule()`, `schedule_barriers()`, `lower_spec()`** — plain functions. Judgement call,
   named as one: written both ways the call sites are identical lines, and `call`'s `async` + by-value
   `In` make the pipe form strictly worse for pure plan-time work. The moment a caller supplies an
   alternative implementation they become pipes and nothing else changes — which is exactly the
   discriminator that *admitted* the fusion rules.
5. **Newtype ids** — `TokenId`, `SlotId`, `SymbolId`, `SubmitTicket`, `RuleSetId`, `PlanKey`,
   `StageId`. P11's compile-time clause; `NodeId` (`op.rs:26-28`) is the shipped precedent.

**Defect declared, not hidden (`crit #14`):** `Loop` is RISC at the `Op` face and a multi-stage
traversal form at the `BoundOp` face that five emitters must interpret, and `lowering-audit.md` already
measures 28 duplicated emitter functions with `reduce_is_cooperative` implemented three times with
diverged signatures. This synthesis therefore adds what AB declined: **CARD A2-lower, a shared
`lower::LoopWalk` in `proxima-tensor` that computes stage order, live axes, domain-as-bound-vs-guard,
and cooperative width once**, with each emitter consuming it. Without it the port multiplies the
duplication the audit measured, and R5-as-a-per-emitter-theorem is one rule implemented five times.

---

# G. Worked example, which is the test (P17)

```rust
/// Two programs computing the SAME banded softmax-weighted reduction, differing only in
/// memory layout: keys head-major, and keys position-major. Both must bind to ONE `Loop`
/// with five stages and identical results, because the rule reads strides from `Layout`
/// and never compares them to a literal — `bind.rs:2465-2474` compares against eight
/// literal stride tuples today, so this test cannot pass on main.
#[proxima::test]
#[case::head_major(KeyLayout::HeadMajor)]
#[case::position_major(KeyLayout::PositionMajor)]
async fn a_banded_softmax_reduction_fuses_at_any_operand_layout(#[case] layout: KeyLayout) {
    let program = banded_softmax_program(layout);            // real openchat dims: 8 kv heads, group 4, head_dim 128
    let shapes  = shape::infer(&program, &[SEQUENCE, MERGED_LENGTH]).expect("infers");
    let rules   = FusePrologue.and_then(FuseEpilogue).and_then(ShareAxis);

    let bound = rules.call(bind_plain(&program, &shapes, &[OUTPUT]).expect("binds")).await.expect("rules apply");

    assert_eq!(bound.len(), 1, "the whole chain is one Loop at any layout");
    let BoundOpKind::Loop { stages, operands, .. } = &bound[0].kind else { panic!("must be a Loop") };
    assert_eq!(stages.len(), 5, "score, max, exp-sum, value-sum, divide");
    assert_eq!(operands.len(), 3, "Q, K, V — the cache length is a symbol, not a ninth operand");
    assert!(matches!(stages[1].domain.constraints[0].offset, BoundOffset::Symbol(_)),
            "the live length is a uniform read, not a compiled-in sentinel");

    let fused = cpu::evaluate_with(&program, &shapes, &blocks, &rules).await.expect("fused runs");
    let plain = cpu::evaluate_with(&program, &shapes, &blocks, &NoRules).await.expect("unfused runs");
    assert_close(&fused, &plain, 1e-6);        // one rule against its own unfused form, not two implementations
}
```

Green here means does-what-it-does, never is-what-it-should-be. The shape argument is §A's minimality
(the `Lookup::extent` candidate was examined and rejected: it bounds an *address*, not an iteration
point, and acts per operand where the band must constrain the key read, the value read and the
accumulation together) and §F's lint — falsifiable by the list in §F, not by this test.

<<UNFINISHED: written at the 30-minute mark. Not closed in the window — (1) `Wiring`/`StackWiring`,
`Bindings`, `CommandRange`, `Grid`, `UniformPatch`, `Completion`, `ResidentSet`, `ExtentDigest`,
`CodecMask`, `DraftSpan`, `Budget`, `TokenWindow` are used in signatures and not given field lists
(`crit #10`; `KernelKey`, `BarrierPoint`, `AcceptedRun`, `RuleSetId` and the pure `plan()` signature ARE
now closed); (2) `ScatterBounds::{Fault, Drop}` and its interaction with the binding `Domain` (a dropped
scatter is an out-of-domain write) is the semantics of the elision lever and is unworked; (3) §C's
generic-architecture section (sliding window, ALiBi, MoE) is asserted from the shipped generators, not
written out as programs; (4) the `sized::` const table with per-const overflow policy is not
re-tabulated here, and `crit #6`'s reachability gate (`PACKED_ROW_BLOCK_SIMDGROUPS`'s only consumer is
unreachable — a swept constant that never reached a dispatch) needs one cell per new const; (5) cards
beyond the first five are listed as an ordered sequence, not as individually gated 30-minute rows>>
