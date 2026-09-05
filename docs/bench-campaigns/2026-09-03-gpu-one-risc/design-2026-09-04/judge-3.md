# judge-3

Read: design-task.md, dispatch-census.md, byte-levers-probe.md, judge-brief.md, all three
candidates in full. Repo reads (welded `cd /Users/brianbruggeman/repos/slot-0/proxima`, pwd
confirmed, no worktree path, no cargo): `proxima-primitives/src/pipe/primitives.rs` (whole file),
`proxima-tensor/src/bind.rs:155-170`, `proxima-tensor/src/map.rs:95-112`,
`proxima-tensor/src/shape.rs:280-295`, `omega/src/msl.rs:440-448`, `omega/src/msl.rs:3178-3195`,
plus counts: `grep -c 'thread_local!' omega/src/metal.rs` = 6;
`grep -n '^pub fn \(plan\|execute\)' omega/src/metal.rs | wc -l` = 11;
`grep -n commandBuffer omega/src/metal.rs | wc -l` = 3.

## Ranking

`[Plan-1, Plan-3, Plan-2]`

- **Plan-1 first — bytes arithmetic.** The only candidate that reconciles from the brief's four
  figures with no invented residual, prices the down-projection contraction axis at a real
  granularity (group = 256 = `Q4K_BLOCK_ELEMENTS`, verified `omega/src/msl.rs:444`), gates union
  density and reports the table at the kill boundary, and uses ROW 287 in-buffer numbers the brief
  names as the decomposition of record. It also reaches the negative conclusion (3.5 ms unreachable
  under the only measured bandwidth) instead of an arithmetic that meets the target.
- **Plan-3 second — quality gates and RISC claim.** Discovery-loop framing with a pre-registered
  hypothesis, mass-recall mechanism metric, ns/element (not bytes) as the elision gate, and a
  random-row degenerate control that must fail; `Loop` reduces 5 bound kinds to 3 with the
  lowering theorem stated as a theorem. Loses to Plan-1 on bytes (161 GB/s from the superseded
  ROW 281/282 instrument; `ffn_down` treated as scaling with `(1-e)`; union density absent) and on
  rollback (`CachedAttention` and eleven companions delete in one commit, no feature seam).
- **Plan-2 third — signatures do not type-check.** `Pipe` in
  `proxima-primitives/src/pipe/primitives.rs:91-101` has no lifetime parameters and no GATs, yet
  Plan-2 writes `type In = CommandScript<'_>`, `type Out = LogitsView<'_>`, `type In =
  (Encoded<'static>, StepInputs<'static>)` with "lifetimes elided in prose". Compounded by a
  125 MB residual invented in §D.1 and then scaled by a codec factor through the §D.7 table, and
  by `FusedRegion { ops: Vec<BoundOp> }` making `BoundOpKind` the recursive tree
  `proxima-tensor/src/bind.rs:160-162` rejects by name ("a plain index into a side table
  (`ComposedBody::steps`), never a `Box<dyn>` recursive tree" — opened, verbatim).

## Per-axis scores (0-5)

| axis | Plan-1 | Plan-2 | Plan-3 |
|---|---:|---:|---:|
| risk surface | 5 | 2 | 3 |
| ordering | 5 | 3 | 4 |
| rollback | 5 | 2 | 2 |
| missing steps | 4 | 3 | 3 |
| hidden coupling | 5 | 3 | 3 |
| observability | 5 | 2 | 5 |
| scope discipline | 4 | 3 | 4 |
| pipe questions | 5 | 2 | 4 |
| no_std tier compliance | 5 | 4 | 3 |
| RISC claim | 5 | 4 | 5 |
| bytes arithmetic | 5 | 2 | 3 |
| quality gates / kill criteria | 5 | 4 | 5 |
| signatures type-check | 5 | 1 | 2 |
| abandoned designs justified | 5 | 4 | 4 |
| **total (70)** | **68** | **39** | **50** |

## One-line justification per axis

### Plan-1 (design-AB)
- risk surface 5 — new path lands behind `feature = "loop-fusion"` with `cached-attention-streaming`
  still selectable, so the highest-risk card bisects by feature selection; R5 and cooperative
  threadgroup staging land in the SAME card as the deletion so the hand kernel at `msl.rs:2586` is
  never replaced by a slower generic one.
- ordering 5 — one ordering, with both predecessors' ordering defects named and fixed (A2's gate
  needing A3's deletions; B's cards 3/4 needing an integer `cached_len` from card 8); D1a → D2 →
  D3-pre → D3b so no byte lever is credited before its kernel precondition measures.
- rollback 5 — per-card revert by feature selection plus a sensitivity table that says whether the
  target survives each omission; `--all-features` used as the deletion gate because a grep count of
  0 is not sufficient while a variant sits behind a feature.
- missing steps 4 — an explicit `<<UNFINISHED>>` block naming five undefined types (`Wiring`,
  `Bindings`, `Grid`, `Budget`, `StepCounters`) and the unworked `ScatterBounds` × `Domain`
  interaction; honest, but they are real gaps in a design that is otherwise signature-complete.
- hidden coupling 5 — `FusionRules` in `PlanKey` because R5 reorders floating-point arithmetic;
  config-derive types kept off the bind path with the tier cited (`proxima-tensor/Cargo.toml:37`,
  `lib.rs:103-104`); `KV_BUCKET_TOKENS`'s interop call sites carried in the card that moves it.
- observability 5 — the emitter returns `KernelSpec { family: KernelFamily }` and CARD A3 lands it
  BEFORE any fusion, so the census's 225/391 decomposition survives the migration; `classify_kind`
  (`metal.rs:1631-1692`) deleted rather than left grepping MSL under a single `Loop` kind.
- scope discipline 4 — 27 cards is large, every one gated, but the span from `IndexPattern::compose`
  to a Q3_K requantizer is one plan.
- pipe questions 5 — nine candidate pipes deleted with the call site written both ways
  (`PlanSlots.call(..).await?` vs `plan_slots(..)?`) and the by-value `In` argument that makes the
  pipe form strictly worse for plan-time work; the drafter-is-a-pipe / selector-is-not asymmetry
  retained with the discriminator stated (device crossings per pass).
- no_std 5 — live cache length as `SymbolId` into the caller's `&[u64]`, no operand read, no
  integer buffer anywhere; CARD 1's gate builds `--no-default-features --features alloc` and names
  the modules compiled (the N==0 clause).
- RISC 5 — `Loop` replaces `Elementwise`/`Reduce`/`CachedAttention` (5 kinds → 3), the band is
  declared on the shipped `AxisIndex` rather than a second affine type, `Op`/`ScalarOp` do not
  grow, and the cost is stated (a multi-stage traversal interpreter five emitters must implement)
  rather than hidden.
- bytes arithmetic 5 — 4,167.6 MB reconciles from FFN 3,170.9 / attn 755.0 / output 107.5 / KV
  134.2 with parameter counts reproducing each; 173.6 GB/s DERIVED from ROW 287's 23.24 ms; group
  = 256 forced by `ffn_down`'s contraction axis (`msl.rs:3184-3189`); union density gated at
  `d_union = 0.6` with the table computed at that kill boundary.
- quality gates 5 — EM/ppl thresholds per lever, k′ histogram per prompt class with kill at 1.5,
  union-density kill at k=4 > 0.6, a random-group control that must fail, the double-quantization
  caveat measured rather than assumed, and text-identity withdrawn for cards that reassociate
  floating-point.
- signatures 5 — POD-only pipe boundaries because `Pipe` has no lifetime params;
  `encode<'p,'c>(plan, bindings, out: &'c mut [Command])` separates plan and scratch lifetimes;
  `poll_complete` explicitly NOT claimed as a `Pipe` (`&mut In` + `Context`/`Poll` match none of
  the four traits).
- abandoned 5 — `FusedRegion` (against `bind.rs:160-162`), `BandBound::Dynamic` (against
  `map.rs:99-107`), nine plan-time pipes, a blocking `Submit`, `Settings` on the bind path, the
  128-unit group, and the two-card Nest-then-Rescaled sequence — each with the constraint that
  killed it.

### Plan-2 (design-A)
- risk surface 2 — `cached-attention-streaming` is renamed `region-fusion` and "the two are never
  both compiled", so there is no migration window; CARD A3 lands `FusedRegion` + `plan_regions` +
  `bind_with_regions` + `render_region` + the CPU arm + both matcher deletions as one card.
- ordering 3 — D2 (draft+verify, a whole speculative decode path) is card 4, before the FSM and the
  driver collapse that its per-pass host cost depends on; CARD A2's gate greps for the `len() == 9`
  read and the sentinels that only CARD A3 deletes.
- rollback 2 — no feature seam and no per-card revert path; the sensitivity table covers byte
  levers only.
- missing steps 3 — `CommandScript`, `StepInputs`, `Wiring`-equivalent and `Budget` are used in
  signatures and never defined; `CountingAllocator` is correctly identified as absent and scoped
  into the work.
- hidden coupling 3 — `FusedRegion { ops: Vec<BoundOp> }` makes `BoundOpKind` recursive through
  `BoundOp`, and `extents`/`domain` duplicate each member's own with the invariant carried in prose.
- observability 2 — `classify_kind`'s MSL substring grep (audit item 12) is never addressed; under
  one `Fused` kind every fusion class reports one label and the census's per-class decomposition
  stops being measurable, which its own CARD A5 gate depends on.
- scope discipline 3 — 21 cards, bounded, but §B.4 adds three plan-time pipe types whose only
  justification is that `AndThen` marker propagation compiles.
- pipe questions 2 — `PlanSlots`/`ScheduleBarriers`/`PackUniforms` are pure plan-time functions
  wearing `Pipe`; the second binary question is asked of `Timed<P>`/`Executor`/`PlanBuilder` and
  correctly kills them, and is not asked of the three that were kept.
- no_std 4 — `Domain`'s dynamic offset as `SymbolId` into `&[u64]` rather than a buffer read is the
  right tier answer and is named as such, with the ninth-operand alternative abandoned for it.
- RISC claim 4 — "one extension, five fusions" is falsifiable and well argued, and
  `RegionSchedule::Rescaled` as a reassociation (citing `op.rs:136-147`'s own scan precedent) is
  sound; the region container is a coarser primitive than `Loop`'s stage list.
- bytes arithmetic 2 — §D.1 totals 4,044 MB against a stated 4,169 and carries the 125 MB gap into
  §D.7 as a `resid` column that is then scaled 125 → 95 by a codec factor; KV priced at 34
  positions instead of the census's 134.2 MB; §D.3 asserts all three FFN matrices scale with `d`,
  which the down-projection contraction axis (`msl.rs:3184-3189`) contradicts; union density of
  multi-token × elision is not raised; §D.7 concludes the target met with 21% margin.
- quality gates 4 — EM/perplexity thresholds and kill criteria per lever, candidate-miss rate for
  the vocab gather, and a secondary kill if GB/s does not fall in proportion to `d`; no degenerate
  control and no union-density gate.
- signatures 1 — four `Pipe` impls carry lifetimes in `In`/`Out` against a trait that has none
  (`primitives.rs:91-101`), annotated "lifetimes elided in prose"; `LogitsView<'_>` as `Pipe::Out`
  is inexpressible, which makes the whole §B.2 chain non-compiling as written.
- abandoned 4 — `BoundOpKind::FlashAttention`, `Timed<P>`, `Executor`, `PlanBuilder`, a global
  `OnceLock<Device>` + mutex, and a host-side `RowSelector`, each with the constraint that killed
  it and the call site that decided it.

### Plan-3 (design-B)
- risk surface 3 — R4 keeps the masking `Select` in the body, so a wrong band is slow rather than
  wrong (the strongest single risk decision in any of the three); offset by card 3 deleting
  `CachedAttention` and eleven companions in one commit with no feature seam.
- ordering 4 — risk-ascending byte order (f16 KV → output codec → `nr1` fold → drafter →
  bandwidth instrumentation → Q3_K → elision) is argued from the measured loss; cards 3 and 4 read
  a `cached_len` operand whose Int32 form arrives at card 8.
- rollback 2 — no seam, no per-card revert; the deletion commit is the rollback boundary.
- missing steps 3 — `Wiring` undefined; the selector's own dispatch cost (five to six serially
  dependent nodes per layer, 32 layers) is never priced against the FFN it gates, so §D.3's win is
  bytes-only on the time axis it elsewhere insists on.
- hidden coupling 3 — `FusionRules`/`FusionCost` carry `Builder + Deserialize + Serialize +
  Settings` on the bind path while `proxima-tensor`'s `config` feature pulls std; and
  `plan(codecs: &[Option<PackedCodec>])` puts an `omega/src/msl.rs`-defined type in a
  `proxima-tensor` signature.
- observability 5 — `KernelSpec { family: KernelFamily }` decided at plan time and carried in
  `Schedule`, `classify_kind`'s MSL grep replaced, the `placement_dump` env branch replaced by a
  structured event with a file-sink exporter.
- scope discipline 4 — 8 structural cards + 8 byte cards, each with one gate; no speculative types.
- pipe questions 4 — the `Fuse` stage is abandoned on the second binary question with the identical
  call site quoted; but the kept chain contains `Advance: TokenId -> ReadyStep<'p>` closing a cycle
  `AndThen` cannot express (`primitives.rs:203-221` is strictly `First::Out = Second::In`), and
  `ReadyStep` holds `scratch: &'p mut StepScratch` produced from a `&self` pipe.
- no_std 3 — `Schedule`/`Command`/`Band` as pure caller-owned data is the right shape and is what
  forces `plan()` out of the device; the `Settings` derives and the `PackedCodec` parameter cut
  against it.
- RISC claim 5 — `Loop { stages }` as one flat side table one granularity above `ComposedBody`,
  `IndexPattern::compose` justified as the operation with nothing weaker than substitution beneath
  it, R5 written as a lowering theorem an emitter may decline with identical results.
- bytes arithmetic 3 — 4.168 GB reconciles to 0.03% with per-row provenance tags, and the rank-128
  predictor's 42.5 MB/token is counted; but 161 GB/s comes from the superseded ROW 281/282
  instrument, `ffn_down` is priced as reading `(1-e)` of its columns with no block-alignment
  argument, and the `÷ a = 2.5` row is applied after 80% elision with no union-density treatment.
- quality gates 5 — pre-registered hypothesis, mass recall as the mechanism metric, ns/element
  against the dense arm as the elision gate (naming that a bytes-only gate would have passed the
  arm that already lost), a random-row control that must fail, KL in nats beside exact-match.
- signatures 2 — the seven-stage table's `In`/`Out` carry `'p` against a lifetime-free trait;
  `encode<'p>(plan: &'p Plan, .., out: &'p mut [Command<'p>]) -> &'p [Command<'p>]` unifies plan and
  scratch lifetimes so exactly one encode per plan is expressible; `SampledStep` holds a
  `LogitsView` crossing a pipe boundary.
- abandoned 4 — the `Fuse` pipe stage, `BoundOpKind::FlashAttention`, `Arc<Mutex<..>>` for the
  thread-locals, deleting the `Select` when a band is derived, and the bytes-only elision gate.

## Notes on contested claims I grounded rather than credited

- `grep -c 'thread_local!' omega/src/metal.rs` = **6**. The brief and audit say eight; Plan-1 and
  Plan-2 both state the discrepancy, Plan-3 repeats eight and lists seven sites.
- `grep -n '^pub fn \(plan\|execute\)' omega/src/metal.rs | wc -l` = **11**. Plan-1 says eleven,
  Plan-2 and Plan-3 say nine while listing ten.
- `grep -n commandBuffer omega/src/metal.rs | wc -l` = **3**. Plan-2's §F.5 says "versus ten today";
  Plan-1 states 3 and sets the gate at 1.
- `map.rs:99-107` (opened): every backend carries every buffer as f32 including `indices`;
  "Lifting the ceiling means adding real integer buffers, not raising a constant." This weakens
  Plan-3's `BandBound::Dynamic { slot }` reading a rank-0 integer operand, though `shape.rs:283-291`
  shows the logical-integer/physical-f32 convention already exists, so Plan-1's framing of it as
  fatal is stronger than the artifact supports. Plan-1's symbol still wins on the narrower ground
  that it removes the ninth operand and the `operands.len() == 8|9` discriminator outright.
- `omega/src/msl.rs:444`: `Q4K_BLOCK_ELEMENTS = 256`, doc states the whole K-quant super-block
  family is 256 wide — Plan-1's group size is sourced, Plan-2's 128 is not.
