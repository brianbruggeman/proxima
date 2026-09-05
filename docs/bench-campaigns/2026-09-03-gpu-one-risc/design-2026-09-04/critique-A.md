# critique-A — weaknesses in design-A.md

Read-only pass over `/Users/brianbruggeman/repos/slot-0/proxima` at `f3c4e98`. Every `file:line`
below was opened this session. Critique only.

---

## 1. (SEVERE, would sink it alone) The byte arithmetic that orders the whole plan is not reproducible from its own table, and one required-lever composition is arithmetically invalid

The design declares §D primary and derives the entire card order from §D.7. Two independent defects:

**(a) L1 x L2 are multiplied as if independent; they are not.** §D.7 applies L2 (FFN density
`d = 0.35`, 2,422.6 -> 847.9) to the *per-pass* column and then divides the whole pass by
`k' = 2.18`. A k-token pass has k query rows, and §D.3's selector runs per row: the FFN bytes a
pass moves are the bytes of the **union** of the k rows' selected group sets, not one row's. At
k=4 and independent selection the union density is `1-(1-0.35)^4 = 0.82`, not 0.35. The design
states no correlation assumption and no measurement of one. At union density 0.7 the pass total is
~2,645 MB -> 1,213 MB/token, above the 1,050 sustained budget and inside noise of the 1,400 peak
budget — i.e. the "meets with 21% margin" conclusion (line 1269) flips. The same defect applies to
L4.2's candidate-set gather (§D.5.2) composed with L1: k rows need k candidate sets.
Violates principle 18 (a load-bearing DERIVED number acted upon) and principle 19 (a conclusion
whose mechanism is not traced).

**(b) Three sensitivity rows do not reproduce from the stage table above them.** Recomputing from
§D.7's own rows:
- `without L5` — full-chain pass 1,797.6 with KV at 268.5; restoring KV to 537.0 gives 2,066.1
  per pass = **947.8** MB/token, not the stated **886.2** (line 1284). 6.5% error, above the
  design's own 5% halt threshold (line 1317).
- `L1 + L2 only` — 3,170.9x0.35 + 755.0 + 107.5 + 537.0 + 125 = 2,634.3 per pass = **1,208.4**,
  not **1,175.1** (line 1287).
- `without L3` — 1,109.8 + 755.0 + 9.4 + 268.5 + 125 = 2,267.7 = **1,040.2**, not **1,033.9**
  (line 1283).
The sensitivity table is the sole support for "L1 and L2 are both required" (line 1289), which is
the sole support for the card order. A table that cannot be re-derived cannot carry that.

**(c) The 125 MB residual is scaled by a lever whose applicability to it is unknown.** §D.1 states
125 MB is unexplained; §D.7's L3 row multiplies it by 0.764 (125 -> 95, line 1263) — a codec factor
applied to bytes whose codec is by definition not known. §D.1 itself partially attributes 29.4 MB
of the residual to Q5_K `ffn_down` (line 999), for which the correct factor is 3.4375/5.5 = 0.625.
Credit is taken for bytes that were never identified.

**(d) `k' = 2.18` is arithmetically `1 + a + a^2 + a^3` at a = 0.6, i.e. 3 drafts + 1 free token.**
The text says "`k = 4` at acceptance `alpha = 0.6`" (line 1251). If k=4 means 4 drafted tokens the
standard figure is `(1-a^5)/(1-a) = 2.31`; if it means a 4-wide pass, 2.18 is right. The label is
ambiguous and the pass-width interpretation is what the byte column needs, but nothing in the
design binds them. It is tagged ASSUMED and gated by CARD D2, which is correct practice; the
ambiguity is in what will be compared against.

## 2. (SEVERE) CARD A3 deletes a measured-fast hand kernel and replaces it with a generic emitter that cannot pass A3's own `M` gate; `RegionSchedule::Rescaled` (CARD A4) is not new — the shipped kernel already does it

`render_cached_attention`'s inner loop at `omega/src/msl.rs:2586` already implements online
rescaled softmax verbatim: `float next_max = max(maximum, score); float weight = exp(score -
next_max); float rescale = ...; sum = sum * rescale + weight; weighted[..] = weighted[..] * rescale
+ weight * shared_v[..]`. It also stages K/V through `threadgroup shared_k_even/shared_k_odd/
shared_v` and reduces with `simd_sum`/`simd_broadcast_first`. §A.5 presents `Rescaled` as the
single-pass form to be introduced at CARD A4 (line 1338) after CARD A3 (line 1337) has already
deleted `render_cached_attention`. Between those two cards attention runs `RegionSchedule::Nest`
— two passes over the key axis, no threadgroup staging, no simdgroup reduction — against a gate
(`M`, line 1311) that forbids `gpu_exec_ms` regressing beyond CoV. The census measures the fused
attention op at 3.515 ms of 33.0 (`dispatch-census.md:26`), so the regression is measurable and
A3 fails its own gate. Nothing in §A.7's `render_region` description mentions threadgroup memory,
cooperative loads, or simdgroup reductions, so even A4 is not stated to recover them.
Violates the task's "less work + same output = keep" and principle 14 (the incumbent wins).

## 3. (SEVERE) CARD A2's gate presupposes CARD A3's work; the `Domain` cards have no incremental path

CARD A2 (line 1330) gates on `grep -c 'operands().len() == 9' == 0` and "attention `i64::MIN`/
`i64::MAX` sentinels == 0". Those live inside `BoundOpKind::CachedAttention` (`bind.rs:240-251`)
and its readers (`cpu.rs:4847`, `omega/src/msl.rs:2030`, `msl.rs:2543`), which CARD A3 deletes
seven cards later. To pass A2, production's attention must already be spelled with a `Domain`;
but the production program is hand-written Rust (`spec.rs:898..7184`) and only moves to spec data
at CARD C1 (card 18). Rewriting it to the `Domain` spelling makes
`cached_attention_candidates` / `exact_merged_causal_mask_cached_len` stop matching (they match
the `Iota`/`Greater`/`Select` mask nodes), so attention silently falls back to the unfused chain —
the exact failure mode the design names as audit finding 1 — and CARD A2 fails gate `M` too.
The ordering axis the task asks about is answered "interleave", but the interleave is not
executable as written.

## 4. (SEVERE) The pipe chains as written do not typecheck, and the design says so only in a parenthetical

`Pipe` is `type In; type Out; type Err; fn call(&self, input: Self::In) -> impl Future<...>`
(`proxima-primitives/src/pipe/primitives.rs:91-102`). No lifetime parameters, no GATs.
- `Read::Out = LogitsView<'_>` and `Sample::In = LogitsView<'_>` (design lines 562, 571) are not
  expressible: an associated type cannot carry an elided per-call lifetime. The design's only
  acknowledgement is `// lifetimes elided in prose` (line 544). Forcing them `'static`/owned means
  a per-token owned logits buffer (32,002 f32 = 128 KB), which contradicts §B.6's zero-allocation
  budget and its counter test (line 766). This is the read-buffer-borrow trap.
- The plan chain `PlanSlots.and_then(ScheduleBarriers).and_then(PackUniforms)` (line 1386) cannot
  compose: `AndThen` requires `Second::In = First::Out` (`primitives.rs:206-208`), but
  `PlanSlots::Out = SlotPlan` while `ScheduleBarriers::In = (Vec<BoundOp>, SlotPlan)`, and
  `PackUniforms::In = Vec<BoundOp>` while `ScheduleBarriers::Out = BarrierSchedule`. As given, it
  is a compile error, and the composition is what §F.4 and the P1 abandonment of `PlanBuilder`
  (line 1385) rest on.

## 5. (SEVERE) `PlanSlots` / `ScheduleBarriers` / `PackUniforms` / `Encode` / `Sample` fail the second binary question — they are relocations

Write the call site both ways: `PlanSlots.call((resolved, retires, outputs)).await?` versus
`plan_slots(&resolved, &retires, &outputs)?`. The lines are the same work, and the pipe form is
strictly worse: `Pipe::call` is `async`, so a pure plan-time interval assignment now returns a
future and needs `block_on`; and `call(&self, input: Self::In)` takes `In` **by value**, so
`Vec<BoundOp>` is moved into the pipe and cannot be read by the next stage (finding 4). What can a
caller do that they could not before? Nothing. Per the rust.md rule ("what can a caller DO... if
the answer is nothing, it is not a type, it is a relocation"), these three unit structs are the
`Erased<P>` failure repeated. The same test kills `Encode`: `Ready::encode(self, ids)` (line 490)
already builds the command script; the `Encode` pipe (line 541) takes an already-`Encoded` state
and returns the script `Encoded::script()` (line 498) already returns.

## 6. (SEVERE) The FSM and the pipe chain are two parallel encodings of the same step, and the chain never drives the FSM

`Encode::In = (Encoded, StepInputs)` consumes the `Encoded` state and returns `CommandScript`;
`Encoded::submitted(ticket) -> InFlight` (line 500) and `InFlight::completed(logits) -> Sampled`
(line 509) are therefore never reachable from `Encode.and_then(Submit).and_then(Read)
.and_then(Sample)` (lines 581-584), whose `Out` is a bare `TokenId`. The `InFlight`/`Sampled`
transitions — including `Sampled::accepted()` (line 1077), which L1's whole KV-advance rule depends
on — are dead in the composed driver. The task asked for "FSM x sansio + FSM x orchestration over
pipes"; what is delivered is an FSM *and* a pipe chain with no stated wiring between them.

## 7. (MAJOR) `Domain`/`HalfSpace` duplicate a shipped primitive (`AxisIndex`), and the duplicate immediately diverges

`map.rs:60-66`: `pub struct AxisIndex { pub terms: SmallVec<[AxisTerm; MAX_INLINE_TERMS]>, pub
offset: i32 }`. The proposed `HalfSpace` (design lines 131-134) is field-for-field the same with
`offset: Offset` instead of `i32`. Reuse-first (P1) requires the design to write the expression
with the existing type and state why it fails; §A.3 enumerates four alternatives (no extension,
wider `Multiply`, `IndexMap::Banded`, tuple reduce) and never considers extending `AxisIndex`'s
offset to `Offset`. The consequence of the split is immediate drift: constraints can carry a
symbolic offset, operand addresses cannot, so a program whose *address* depends on the cache
length still cannot be written, and the crate now has two "affine expression" types.

## 8. (MAJOR) The design contradicts itself on whether the algebra can express out-of-range

§A.2 line 102: "the algebra cannot say that an iteration point is out of range" — the load-bearing
premise for `Domain`'s minimality. §D.3 line 1154 cites the opposite, correctly: `Lookup` carries
`extent` "so an executor can reject an out-of-range fetched index instead of reading past the
buffer" (`bind.rs:127-129`), and `ScatterBounds::Fault` is named as "today's behaviour". Out-of-
range exists in the bound IR; the design's §A.3 minimality proof never distinguishes the two
notions or rules out the shipped one. A minimality proof with an unexamined candidate is not a
proof (principle 6).

## 9. (MAJOR) The group-routed selector is a shape argument with no quality argument and no artifact

§D.3 (lines 1127-1148) claims the "computed address over constant G" form. Three defects:
- **`G` is not constant.** The design fixes the *group size* at 128 (`112 x 128`), so
  `G = d_ff/128` grows linearly with the hidden axis. The admitted form the memory grades as clean
  is "hash / index-computed routing, O(1) bucket"; a threshold comparison against every one of G
  learned scores is fanout Theta(G) = Theta(n/128) — the same content-addressed selection at a
  coarser granularity, which the design half-admits at line 1135 ("the two are indistinguishable at
  the API") and then does not resolve.
- **The router weights do not exist.** A `d_model x G` router (459k params, line 1138) is not in
  the openchat-3.5 checkpoint and cannot be read from it. No card trains, calibrates, or derives it;
  CARD D3b (line 1344) lands the selector and immediately gates on EM >= 0.95 with no artifact to
  evaluate. The second of the two levers the design calls *required* rests on a non-existent input.
- **`d = 0.35` carries no provenance tag** (line 1097), in a document that tags `k' = 2.18` ASSUMED
  and 400 GB/s ASSUMED. It is load-bearing in the required-lever table. Principle 18.
- **The byte model over-credits `ffn_down`.** The `d` factor is applied to all three FFN matrices
  (line 1096). For `gate`/`up` the selected axis is the output-row axis (whole rows, the claim
  "whole Q4_K blocks" holds). For `down` the selected axis is the **reduced** axis, so a 128-unit
  group is half of a `Q4K_BLOCK_ELEMENTS = 256` superblock (`omega/src/msl.rs:444`) inside every
  output row: the reads become strided partial superblocks, not skipped bytes. The design's own
  justification for group size ("one group is 128 x 4096 params, whole Q4_K blocks", line 1133) is
  false for one of the three matrices it prices.

## 10. (MAJOR) `FusedRegion` reintroduces the macro-op it deletes, one level up

`bind.rs:176-184`'s `ComposedBody { steps: Vec<BodyStep> }` is a flat side table with `StepArg`
indices precisely *because* a `Box<dyn>` recursive tree was rejected (`bind.rs:160-162`, verbatim:
"a plain index into a side table (`ComposedBody::steps`), never a `Box<dyn>` recursive tree").
`FusedRegion { ops: Vec<BoundOp>, ... }` (design line 225) makes `BoundOpKind` recursive through
`BoundOp` — the shape `ComposedBody` exists to avoid — and every exhaustive match over `BoundOp`
in `cpu.rs`, `msl.rs`, `metal.rs` must now recurse. It also carries `extents: Vec<u64>` and
`domain` duplicating each member's own `BoundOp::extents`, with the invariant "every member's
`BoundOp::extents` equals this" (line 232) stated in prose, unenforced — the same class of
prose-only invariant the audit flags at `metal.rs:3525` (audit line 37). And `BoundOpKind::Fused`
becomes by far the largest variant, inflating every `BoundOp`; no size measurement is offered
(P11's measured-large-variant clause).

## 11. (MAJOR) CARD A1 is a workspace-wide mechanical edit and breaks every shipped spec file, neither of which the card accounts for

`Reduce` (`op.rs:153-164`) has eight fields, no `#[non_exhaustive]`, `#[cfg_attr(feature =
"config", derive(Serialize, Deserialize))]`, and is constructed by struct literal at 25 sites in
`bind.rs`, 7 in `shape.rs`, plus the model builders in `spec.rs`. Adding `pub domain: Domain`
(design line 165) without `#[serde(default)]` — none is shown, and `op.rs` uses `serde(default)`
nowhere — makes all eleven files in `proxima-tensor/specs/` fail to deserialize, including
`causal_attention.toml`, which CARD A1's own gate requires to parse and evaluate (line 1329).
"Default = byte-identical to today" (design line 138) is true of the *semantics* and false of the
*compile*.

## 12. (MAJOR) `Submit` blocks inside an `async fn`, with no waker or poll surface

The single Metal edge is `Submit::call(&self, script) -> impl Future<Output = Result<Completion,
MetalError>>` (design line 556). Completion today is `command_buffer.waitUntilCompleted()`
(`omega/src/metal.rs:680`). A future whose body blocks a thread on the GPU is a blocking lock in
async by another name (principle 21 rung 3 misapplied at rung 2's position); the design never
mentions `addCompletedHandler`, a waker, or a poll-based `poll_complete`, which rust.md names as
the preferred surface for reactor-driven/cancellable code. For L1 (multi-token) and any pipelining
of pass n+1's encode against pass n's execution — the mechanism that keeps a k-token pass from
paying k times the host cost, which §0.2 line 49 promises — this is the load-bearing detail and it
is absent.

## 13. (MAJOR) The marker-propagation "compile-time proof" is a self-assertion

Design lines 614-617 claim `plan()`-as-a-chain "inherits `AndThen`'s marker propagation
(`primitives.rs:372-399`) ... which is the compile-time proof that planning does no I/O."
`proxima-core/src/markers.rs:61,70`: `pub trait IsPure {}`, `pub trait WithoutFilesystem {}` —
safe, empty, hand-implemented traits. `AndThen` propagates only what a human asserted on the
leaves; nothing checks that `PlanSlots` does no I/O. Rich -> poor: a structural claim reduced to a
marker the author writes by hand. Also `assert_all_markers` (`primitives.rs:626`) is inside
`#[cfg(test)] mod tests` and is private, so CARD B1's gate "assert_all_markers on the plan chain
compiles" (line 1333) cites a helper that does not exist outside that test module.

## 14. (MODERATE) Verifiable miscounts and mis-citations against the repo

- "exactly one `commandBuffer()` call site after CARD B2, versus **ten** today" (line 1426).
  `grep -n commandBuffer omega/src/metal.rs` returns **3** (lines 588, 1041, 1343). The gate at
  card B2 (`== 1`, line 1334) is fine; the "ten" is false and inflates the claimed reduction.
- "**nine** executor entry points (`metal.rs:519, 535, 978, 1177, 1192, 1210, 1409, 1484, 1509,
  1613`)" (line 20) lists **ten** line numbers, one of which (`1177 plan_named`) is not an
  executor. §B.3's table has ten rows and line 605 then says "produced ten functions". The design
  contradicts itself twice within one section.
- "`classify_packed_row_block` requires `gather_count == 0` (`msl.rs:1039-1052, 1389`)" (line 1107).
  `msl.rs:1039-1052` is `reduce_is_cooperative`; `classify_packed_row_block` is at 1389 and
  *calls* it. This matters (see finding 15).
- "six `thread_local!` blocks" (line 700) is correct (`grep -c` = 6); the task brief and audit say
  eight (`metal.rs:266, 273, 2551, ...`). The design silently corrects the brief without saying so,
  so the reader cannot tell which count the card B3 gate (`== 0`) is measured against.
- "`omega` gains a dependency on `proxima-primitives` (`cargo add --path ..`)" (line 532) —
  `proxima-tensor` is cited as already depending on it "e.g. `bind.rs:64`"; `bind.rs:64` is inside
  the `Lookup`/`BoundOperands` region, not a `use proxima_primitives` line.

## 15. (MODERATE) Hidden coupling: lifting `gather_count == 0` touches every reduce, not just the packed-row path

CARD D3a (line 1343) says "lift `gather_count == 0` (`msl.rs:1039-1052, 1389`)". The gate at
1039-1052 is inside `reduce_is_cooperative`, which `classify_packed_row_block` calls first
(`msl.rs:1393`) and which independently routes the **65 cooperative reduces per token**
(`dispatch-census.md:23`). Relaxing it changes the routing of every rmsnorm sum-of-squares as well
as the matvecs. The card's gate measures only "GB/s on the packed-row path at densities
{1.0, 0.5, 0.35}" and would not see a cooperative-reduce regression.

## 16. (MODERATE) Observability regresses: `classify_kind` collapses to one label and is not in the migration list

`omega/src/metal.rs:1631-1640` matches `bound.kind` and returns `&'static str` per kind, including
a `CachedAttention` arm. Under `BoundOpKind::Fused(FusedRegion)` every one of the five fusion
classes reports the same label, so the per-kind profile that produced the census's numbers
(`dispatch-census.md:25-27`: cached-attention 3.515 ms, packed-row 24.991 ms) can no longer
separate them — and CARD A5's gate ("bound ops/layer reported ... per lever, each toggled
independently", line 1339) needs exactly that separation. §A.8's deletion list (lines 428-438)
omits `classify_kind`'s arm, `cpu.rs:982`'s arm, and the `CachedAttention` construction at
`cpu.rs:19178`. The task brief's audit item 12 (profiler groups by `&str`; `classify_kind`
substring-greps generated MSL, audit line 14, status DESIGN B) is never addressed anywhere in §B.

## 17. (MODERATE) Named brief items with no answer

- **`Counter::snapshot_and_reset` "exactly once per step" protocol** (`metal.rs:2719`, brief item 5,
  audit line 32) — not mentioned.
- **`Plan` two-phase init via `mark_resident`** (`metal.rs:373`, `resident_nodes` at
  `metal.rs:336`, brief item 5/6) — `Residency` is introduced (line 720) but nothing states that
  `mark_resident`/`resident_nodes` are removed, and `Plan::resident_nodes` is read at
  `metal.rs:563` and `1026` on the encode path.
- **`register_checkpoint_mapping` side channel** is listed as becoming a `Device` field (line 715)
  with no statement of what happens to the existing call sites.
- **The encode loop and `BufferArena` construction** (brief item 5) remain loops; only the decode
  step becomes an enum. The brief asked for the hidden state machines as enums.

## 18. (MODERATE) P12 sizing: four new caps with no home

`MAX_INLINE_CONSTRAINTS` is correctly routed to `proxima-tensor-runtime.toml` (line 146, file
exists). `MAX_INLINE_REGION_OUTPUTS` (line 229), `MAX_INLINE_RESCALED_SUMS` (line 372),
`MAX_INLINE_BARRIER_SLOTS` (line 669), `MAX_DRAFT` (line 1060) and `PLAN_CACHE_ENTRIES` (line 736,
attributed to `omega::sized` with a default of "two entries" in prose) are named with no config
section, no default, and no env-override key. Principle 12 requires each to trace to a per-unit
sizing source.

## 19. (MODERATE) P4: config surface is asserted, not specified

§F.3 lists `FusionCapabilities`, `ScheduleSet`, `FusionPolicy`, `DriverConfig`, `SubmitPolicy`,
`SamplePolicy` as "principle 4 makes these first-class by construction" (line 1418). Only
`ModelSpec` is shown with `Builder + Deserialize + Serialize + Settings` (line 822), which is the
shape `TensorExecutionConfig` (`proxima-tensor/src/config.rs:43-46`) and `ProgramSpec`
(`spec.rs:69-72`) actually carry. `FusionCapabilities`/`ScheduleSet` derive only
`Debug, Clone, Copy, PartialEq, Eq, Default` (lines 333, 340) — no serde, no builder — so the
capability set is not expressible as data, and `ScheduleSet(u8)` is a private bitfield with no
`Deserialize`. The config<->builder parity fixture is required only for `ModelSpec` (card C2 gate,
line 1347); nothing checks the driver config surfaces.

## 20. (MODERATE) `ScalarOp::Round` violates a documented closed set, and the design knows it

`op.rs:50-56` states the invariant verbatim: "Closed on purpose, and it is the one closed set in
this crate that stays closed: these are scalar machine primitives, not an extension point."
§D.6/CARD D7 (lines 1240-1244, 1349) adds `ScalarOp::Round`. The design labels it "a deliberate
algebra decision", which is the paragraph-defending-a-type signal. It is also unnecessary to the
target by the design's own §D.7, which makes it pure scope.

## 21. (MODERATE) `SymbolId` is a half-migration; `Extent::Symbolic(u16)` keeps the bare id

`SymbolId(pub u16)` is justified as "the same index `Extent::Symbolic(u16)` already carries
(`op.rs:45-48`), given a name so a bare `u16` can never be passed where an axis position is
expected" (design lines 121-124). `Extent::Symbolic(u16)` (`op.rs:46-47`) is not changed, so the
bare `u16` remains at every existing site and the workspace now has two spellings of one index.
P11's newtype clause is satisfied in the new code and violated at the boundary with the old.

## 22. (MODERATE) Rollback: the highest-risk card has no A/B seam

§A.8 line 440: "`cached-attention-streaming` ... is renamed `region-fusion` and keeps its
default-on position; the two are never both compiled, so there is no migration window in which a
program can bind to either shape." That sentence *is* the rollback finding: CARD A3 simultaneously
deletes two matchers, the 10-field variant, the MSL renderer, six `cpu.rs` arms and the second
`bind_plain`, and introduces `plan_regions`/`bind_with_regions`/`render_region`, with no compiled
configuration in which the old path can be re-selected to bisect a parity or `gpu_exec_ms` failure.
`cached-attention-streaming` is a real feature today (`omega/Cargo.toml:21`,
`proxima-tensor/Cargo.toml:33`) and `bind_with_fusion` is already `#[cfg]`-gated
(`bind.rs:2641-2650`), so the seam that exists is deliberately removed.

## 23. (MINOR) `Err = MetalError` on stages declared pure and non-macOS-buildable

`PlanSlots`, `ScheduleBarriers`, `PackUniforms` (lines 625, 636, 642) and `Sample` (line 573) all
declare `type Err = MetalError` while `plan()` is claimed to "compile and run on a non-macOS
target" (line 677) and `Sample` is "PURE". The error type is the driver's; a pure interval
assignment that can only fail with a Metal error is a lying signature.

## 24. (MINOR) `CommandScript`, `StepInputs`, `Completion`, `LogitsView`, `Budget`, `NameTable`, `BlockTable`, `SlotTable`, `UniformTable`, `NoCopyCache`, `CheckpointMapping`, `FusionPolicy`, `DriverConfig`, `SubmitPolicy`, `SamplePolicy` are used but never defined

The task requires "concrete Rust signatures". `CommandScript<'plan>` in particular is load-bearing
in both crates at once: §B.1 puts the FSM in `proxima-tensor` with "zero Metal, zero `omega`
dependency" (line 452) and has `Encoded::script() -> &CommandScript<'plan>` (line 498), while §B.2
has `omega`'s `Encode::Out = CommandScript` (line 544). Whether a per-dispatch Metal argument list
is device-neutral enough to live in the IR crate is the crux of the sans-IO claim, and it is not
stated. Likewise `SubmitPolicy` carries `PerOpBuffer` (line 601) — the one variant given — with
no definition, and it is what three deleted entry points collapse into.

## 25. (MINOR) Sampling acquires a third home

`sample_next_token` lives in `proxima-tokenizer/src/sample.rs:277`. §B.2 makes `Sample` a pipe in
`omega/src/metal.rs` ("the only place in this design that touches a device", line 537) with
`SamplePolicy`, while `Sampled::token()`/`advance()` (lines 516-520) put sampling *state* in
`proxima-tensor`. Three crates for one decision, with no statement of which owns it.

## 26. (MINOR) `KV_BUCKET_TOKENS` relocation is under-scoped

`proxima-tensor/src/sized.rs:326` is the definition, but the consumers are in
`proxima-model-interop/src/generate.rs:695, 709-710, 2336` and it is asserted in
`sized.rs:390`. CARD C3's gate is `grep -c KV_BUCKET_TOKENS proxima-tensor/src == 0` (line 1348),
which does not cover the interop call sites that must switch to `omega::sized`, and interop
depending on `omega` for a policy constant is a new crate edge the design does not name.

## 27. (MINOR) CARD C2's gate exceeds what CARD C2 delivers

Gate: `grep -c 'too_many_arguments' proxima-tensor/src/spec.rs == 0` (line 1347). The actual count
is **17**, spanning the mistral *and* qwen35 builders and helpers. C2 delivers `LayerInputs`,
`KeyRoots`, `CacheRoots` — the mistral cached-layer path. The gate cannot pass without unscoped
work on the qwen35 builders.

## 28. (MINOR) The abandoned-designs list is thin where it matters

Six abandonments are recorded (lines 1361-1398), which formally satisfies the requirement. Three
are straw options that were never candidates: a second `fuse_regions: bool` beside the first
(line 1394), an `Executor` trait holding the same four entry points (line 1382), a process-global
`OnceLock<Device>` + mutex (line 1372). One is not a design at all — "crediting L1 and L2 in the
byte table before the two kernel folds" (line 1364) is an accounting error avoided. The
substantive one is `BoundOpKind::FlashAttention` (line 1377). No abandonment is recorded for §C
(the generic-model half of the task), none for the teaching surface (P2), none for the sizing axes
(P12), and none for §D's lever set — e.g. nothing states what was ruled out in choosing group
routing over the per-row predictor beyond a memory citation.

## 29. (MINOR) P2 teaching surface is not served

No new public type in the design carries the pointer principle 2 requires: which primitive it
composes, a link to that primitive's docs, and when to use the wrapper versus the primitive.
`Domain`, `FusedRegion`, `RegionSchedule`, `FusionCapabilities`, `Step` all have doc comments that
explain mechanism but name no primitive to learn from, and §E has no card for docs or for the
runnable walkthrough principle 11 requires per state machine (CARD B4 mentions "a walkthrough test
driving every legal transition", line 1336; nothing equivalent exists for `plan_regions`).

## 30. (MINOR) Tests are offered as evidence of shape

§G (line 1444) presents `a_banded_softmax_reduction_fuses_at_any_operand_layout` as "the structural
claim". It is a good test of the *rule as implemented*; it cannot establish that `Domain` +
`FusedRegion` is the right shape, and the design leans on it ("which is the falsifiable form of the
claim, not a preference", line 205, for the one-extension-five-fusions claim — which is an
assertion about four levers whose fusion is asserted, not measured, in the same table). Green means
does-what-it-does.

## 31. (MINOR) Provenance tag missing on the anchor number

4,169 MB/token is tagged "task brief, MEASURED upstream" (line 967). The brief states it without a
measurement artifact, and the design's own §D.1 derivation reaches 4,044 from parameter counts.
The 3.0% gap is honestly reported, but the anchor itself carries no artifact — and every ms figure
in §D.7, the 10.42 ms floor, and the 2.98x requirement descend from it. Principle 18 asks for the
artifact in the same breath.

## 32. (MINOR) `PlanCache` capacity is decided in prose

"a prefill/decode alternation thrashes; two entries covers it and the constant says why" (line 736).
The existing cache is keyed `(new_count, merged_len)` (`generate.rs:1245-1248`) and is *also*
duplicated as `placed_plans` (`generate.rs:1248`). Two entries covers exactly one alternation
between two shapes; a k-token speculative pass (§D.2) introduces a third `new_count` per step
class, and L1 makes `new_count` vary with the accepted count, so the key space the design itself
creates is larger than the capacity it sets.

---

## Axis scores (one line each)

- **risk surface** — largest single card (A3) deletes the only measured-fast attention path,
  six `cpu.rs` arms, two matchers and an emitter at once, with the co-existence feature explicitly
  removed (finding 2, 22).
- **ordering** — no, it does not force everything at once by intent, but as written A2's gate
  requires A3's deletions and A3's gate requires A4's schedule, so the increments do not stand
  alone (findings 2, 3).
- **rollback** — a card fails to its own revert; no compiled seam to bisect old vs new binding
  (finding 22); the byte cards do revert cleanly by sensitivity row, except the two rows that do
  not reproduce (finding 1b).
- **missing steps** — router artifact for L2 (9), `serde(default)`/struct-literal migration for A1
  (11), `classify_kind`/`cpu.rs:982`/`cpu.rs:19178` deletions (16), `mark_resident`, counters (17).
- **hidden coupling** — `gather_count == 0` is shared with the cooperative-reduce router (15);
  `CommandScript` straddles two crates (24); interop -> omega for `KV_BUCKET_TOKENS` (26).
- **observability** — regresses: one `Fused` label replaces five kinds, and the profiler item the
  brief names (audit 12) is untouched (16); no `proxima::telemetry` / `#[proxima::instrument]`
  usage named anywhere.
- **scope discipline** — 22 cards across `proxima-tensor` IR + bind + cpu, `omega` msl + metal,
  `proxima-model-interop`, `proxima-gguf` (new Q3_K codec), `proxima-tokenizer`, plus a new global
  allocator and a training-free router; §D.7 shows L3/L4/L5 are margin, and they are still cards.
- **pipe questions** — `PlanSlots`/`ScheduleBarriers`/`PackUniforms`/`Encode`/`Sample` fail the
  second binary question (5); `Domain`/`HalfSpace` fail the first against `AxisIndex` (7);
  `FusedRegion` is defensible as IR but re-creates the recursion `ComposedBody` rejected (10);
  `ScatterBounds`, `SubmitPolicy`, `FusionCapabilities` pass (they let a caller do something new);
  `PlanSlots`' companions (`SlotPlan`, `BarrierSchedule`, `UniformTable`) are compensators for the
  three relocated pipes.
- **no_std tier** — `Domain`/`HalfSpace` are alloc-tier-clean (SmallVec, no std), but the tier
  claim is never exercised: no card builds `proxima-tensor --no-default-features --features alloc`
  and asserts which modules compiled (P3's N==0 clause); `cached-attention-streaming = ["std"]`
  (`proxima-tensor/Cargo.toml:33`) means the renamed `region-fusion` inherits a std gate the design
  does not mention.
- **RISC claim** — `Domain` is a genuine algebra extension (addressing, attaches to the op, one
  concept); `FusedRegion` is a second IR: a nest-of-nests with its own `extents`, `domain` and
  `schedule` duplicating its members', unenforced (10). The macro-op does not return under a new
  name in the *matcher* sense — `plan_regions` is a structural rule with no model name — but it
  does return in the *representation* sense, and `RegionSchedule::Rescaled` re-encodes as an enum
  variant what the shipped kernel does inline (2).
- **bytes arithmetic** — recomputed: §D.1's parameter chain checks out exactly (FFN 5,637,144,576
  x 4.5/8 = 3,170.9 MB; attn 1,342,177,280 x 4.5/8 = 755.0; output 131,080,192 x 6.5625/8 = 107.5;
  KV@2048 = 536.9); the 125 MB residual is honestly flagged and then illegitimately scaled (1c);
  the L1 x L2 product is invalid (1a); three sensitivity rows do not reproduce, one by 6.5% (1b).
- **group-routed selector** — shape argument only: G is not constant, the router weights do not
  exist, `d = 0.35` is untagged, and `ffn_down`'s gather is on the contraction axis (9).
- **abandoned designs** — present and formally sufficient; three of six are straw, one is not a
  design, and §C/§D contribute none (28).
