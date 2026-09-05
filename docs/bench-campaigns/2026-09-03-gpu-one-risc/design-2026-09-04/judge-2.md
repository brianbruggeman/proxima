# judge-2 — design tournament round 1

Mapping used: Plan-1 = design-B.md, Plan-2 = design-AB.md, Plan-3 = design-A.md.
Read-only; no cargo. Repo claims below were re-opened at
`/Users/brianbruggeman/repos/slot-0/proxima` before being credited or penalised.

## Ranking

`[Plan-2, Plan-1, Plan-3]`

## Grounding (principle 6) — what I opened to settle contested claims

| claim | verified at | result |
|---|---|---|
| `Pipe` has no lifetime params / no GATs | `proxima-primitives/src/pipe/primitives.rs:91-102` | confirmed: `type In; type Out; type Err; fn call(&self, In) -> impl Future` |
| "never a `Box<dyn>` recursive tree" | `proxima-tensor/src/bind.rs:160-162` | confirmed verbatim, at scalar granularity |
| no integer buffers in any backend | `proxima-tensor/src/map.rs:99-107` | confirmed: every buffer is f32 incl. indices; "Lifting the ceiling means adding real integer buffers" |
| `config` feature is std-only, alloc tier never sees it | `proxima-tensor/Cargo.toml:37`, `lib.rs:103-104` | confirmed |
| `op.rs` `Reduce` has no `#[serde(default)]` anywhere | `proxima-tensor/src/op.rs` (serde only at 27/43/58 as cfg_attr derives) | confirmed; 11 files in `proxima-tensor/specs/` |
| `Q4K_BLOCK_ELEMENTS = 256`; `PackedRowBlock { .. reduce_dim .. }` | `omega/src/msl.rs:444`, `3184-3189` | confirmed |
| counts | `grep -c 'thread_local!' omega/src/metal.rs` = **6**; `grep -c '^pub fn \(plan\|execute\)'` = **11**; `grep -c 'commandBuffer()'` = **3**; `gather_count` exclusion sites at msl.rs **1047, 1218, 1389** = **3** | Plan-2's counts match at every point; Plan-1 says eight thread-locals / nine executors / one gather gate; Plan-3 says "ten `commandBuffer()` today" |
| `BoundOpKind::` blast radius | cpu.rs 74, bind.rs 65, msl.rs 39, wgsl.rs 22, cuda.rs 19, metal.rs 16 | matches Plan-2's table exactly |

## Scores (0-5)

| axis | Plan-1 (B) | Plan-2 (AB) | Plan-3 (A) |
|---|---|---|---|
| risk surface | 3 | 5 | 3 |
| ordering | 3 | 5 | 3 |
| rollback | 2 | 5 | 2 |
| missing steps | 3 | 4 | 3 |
| hidden coupling | 3 | 5 | 3 |
| observability | 4 | 5 | 2 |
| scope discipline | 4 | 4 | 4 |
| pipe questions (in code) | 4 | 5 | 2 |
| no_std tier compliance | 2 | 5 | 4 |
| RISC claim (extension vs macro-op renamed) | 4 | 4 | 3 |
| bytes arithmetic | 2 | 5 | 2 |
| quality gates / kill criteria | 4 | 5 | 4 |
| signatures type-check vs `primitives.rs` | 2 | 5 | 1 |
| abandoned designs justified by a constraint | 5 | 5 | 5 |
| **total (70)** | **45** | **62** | **41** |

## One line per axis, per candidate

### Plan-1 (design-B)

- risk surface 3 — card 3 deletes `CachedAttention` + eleven companions in one commit with no feature seam, while `bind.rs:2641-2648`'s existing `cached-attention-streaming` gate already offers one.
- ordering 3 — cards 3/4 rest on `BandBound::Dynamic` reading an integer operand, but `cached_len: Int32` is card 8.
- rollback 2 — no rollback mechanism named for the highest-blast-radius card (261 `BoundOpKind::` sites across six files).
- missing steps 3 — selector dispatch cost omitted entirely from D.3; `CountingAllocator`'s existence never checked; five emitters ported with no cross-backend gate cell.
- hidden coupling 3 — `FusionRules` is env-settable (`#[settings(prefix = "TENSOR_FUSION")]`) yet R5 reorders floating-point arithmetic and the rules are not in the plan-cache key.
- observability 4 — `KernelSpec { family: KernelFamily }` replaces `classify_kind`'s MSL substring grep, and E.4 tabulates every information-destroying site.
- scope discipline 4 — 18 cards, each with a named gate; nothing is filed as follow-up.
- pipe questions 4 — the `Fuse` stage is abandoned by writing the call site both ways (`shapes.and_then(builder)` identical), which is the rule applied correctly.
- no_std tier 2 — `FusionRules`/`FusionCost` carry `Settings + Deserialize + Serialize` in `bind.rs`, but `Cargo.toml:37` makes `config` std-only and `lib.rs:103-104` says the alloc tier never sees it; and C.3's `DType::Int32` leaf plus `BandBound::Dynamic`'s `DType::is_integer` check both require integer buffers `map.rs:99-107` says no backend plumbs.
- RISC 4 — `Loop { stages }` collapses 5 kinds to 3 and R4 keeps the masking `Select` so a rule can cost time but never accuracy; R4 itself is a peephole on one mask spelling.
- bytes arithmetic 2 — reproduces 4.168 GB from the given factors exactly, then prices time from ROW 281/282, which `dispatch-census.md:25-29` marks superseded and names ROW 287 as the decomposition to use; the resulting table is internally inconsistent (4.168 GB @161 GB/s + 0.7 ms dispatch = 26.6 ms, printed as "33.0 MEASURED"), the unaccounted 6.4 ms is the host residual; union density of multi-token × elision is not mentioned; D.3 prices `ffn_down` as "(1-e) of down's columns" without the contraction-axis/superblock argument.
- quality gates 4 — mass-recall distribution, ns/element against the dense arm, and a random-row degenerate control that must fail; D.1's acceptance rate gets no kill criterion.
- signatures 2 — the driver table uses `ReadyStep<'p>`/`EncodedStep<'p>` as `Pipe::In`/`Out` with no lifetime-carrying pipe structs given, `Advance` produces `&'p mut StepScratch` from `&self`, and `encode<'p>(.., out: &'p mut [Command<'p>]) -> &'p [Command<'p>]` ties the scratch to the plan (one encode per plan).
- abandoned 5 — `FlashAttention`, the `Fuse` stage, `Arc<Mutex>` for the thread-locals, deleting the `Select` under a derived band, and two designs killed by measured losses (`msl.rs:3171-3200` `nr0`, ROW 180/181).

### Plan-2 (design-AB)

- risk surface 5 — `loop-fusion` lands beside the existing `cached-attention-streaming` seam so the highest-risk card is bisectable by feature selection, and card A4 lands R5 with cooperative operand staging so the hand kernel is never temporarily replaced by a slower generic one.
- ordering 5 — both critiques' ordering defects are named and closed: sentinel/discriminator greps are gated on the card that deletes them, and `cached_len` is a symbol from A1 so nothing waits on integer buffers; there is one ordering, not two.
- rollback 5 — feature-selection revert per card plus a sensitivity table that says whether the target survives without the reverted lever.
- missing steps 4 — an explicit `<<UNFINISHED>>` block naming `Wiring`'s fields, twelve types used in signatures and not defined, and the `ScatterBounds`×`Domain` interaction; honest, but they are real gaps.
- hidden coupling 5 — `FusionRules` in `PlanKey` because R5 reorders arithmetic; `KV_BUCKET_TOKENS`'s interop call sites (`generate.rs:695, 709-710, 2336`) carried in the card; all three `gather_count` sites, which I confirmed are 1047/1218/1389.
- observability 5 — `KernelFamily` returned by the emitter, and card A3 lands it *before* any fusion so the census's 225/391 split is reproducible across the migration.
- scope discipline 4 — 27 cards is the largest surface of the three, mitigated by every card carrying a gate and an N assertion.
- pipe questions 5 — nine candidates abandoned by writing the call site both ways (`PlanSlots.call(..).await?` vs `plan_slots(..)?`), `poll_complete` explicitly *not* claimed as a pipe because `&mut In`/`Context`/`Poll` matches none of the four traits, and the draft-source-is/selector-is-not asymmetry with the discriminator stated (device crossings per pass).
- no_std tier 5 — catches that `Settings` on the bind path does not compile at the tier it was placed in, routes the live cache length through `symbols: &[u64]` rather than an integer buffer `map.rs:99-107` denies exists, and gates the alloc build on *naming the modules it compiled*.
- RISC 4 — takes `Loop` and then states the cost the other two hid: RISC at the `Op` face, a multi-stage traversal interpreter at the `BoundOp` face that five emitters must implement, with the migration priced against the measured 261-site blast radius.
- bytes arithmetic 5 — reproduces 4,167.6 MB from the brief's four factors; every time constant traced to ROW 287 (173.6 GB/s DERIVED from 23.24 ms, not 161 from the superseded instrument); the 5.96 ms host residual derived and carried in every route row; union density of multi-token × elision measured as a gate with the table given at the *kill boundary* rather than the favourable end; and the down-projection contraction-axis constraint is the one finding that changes the design rather than a number (group = 256 = one Q4_K superblock, `msl.rs:444`/`3184-3189`, both confirmed).
- quality gates 5 — every lossy lever has a metric, a threshold and a kill; `k'` gets the kill criterion Plan-1 omitted; a random-group control that must fail; the Q4_K→Q3_K double-quantization caveat stated as measured-not-assumed; text-identity explicitly withdrawn for cards that reassociate floating point.
- signatures 5 — the only candidate that reconciles with `primitives.rs:91-102`: POD-only across every pipe boundary, no borrowed view as an `In`/`Out`, `Step` never a pipe payload, and `encode<'p, 'c>` splitting plan lifetime from command scratch.
- abandoned 5 — `FusedRegion` (against `bind.rs:160-162`, which I confirmed), `BandBound::Dynamic` (against `map.rs:99-107`), `Settings` on the bind path, the blocking `Submit`, a parallel `HalfSpace` type, the 128-unit group, and the "§A/§B are margin" framing — each tied to a named constraint.

### Plan-3 (design-A)

- risk surface 3 — the `cached-attention-streaming` → `region-fusion` rename explicitly removes the migration window ("the two are never both compiled"), deleting the bisect seam on the largest card.
- ordering 3 — CARD A2's gate asserts `operands().len() == 9` and the `i64::MIN`/`MAX` sentinel counts are zero, but those deletions are CARD A3.
- rollback 2 — none named; the rename forecloses the one that exists.
- missing steps 3 — `Domain` is added as a field to `Op::Reduce`/`Elementwise` with no `#[serde(default)]`, and `op.rs` carries none while `proxima-tensor/specs/` holds 11 files (both confirmed); a dozen types appear in signatures undefined.
- hidden coupling 3 — `FusionPolicy`/`FusionCapabilities` change what materializes and what reassociates with no statement about the plan cache key.
- observability 2 — audit item 12 (`classify_kind` substring-grepping MSL, `metal.rs:1631-1692`) is not addressed at all, and under one `Fused` kind every fusion class collapses to one profiler label.
- scope discipline 4 — 22 cards with per-card gates.
- pipe questions 2 — `PlanSlots`/`ScheduleBarriers`/`PackUniforms` are pure plan-time functions given `async fn call` and by-value `In`, which is the relocation the second binary question exists to catch; and the `Encode→Submit→Read→Sample` chain never reaches `InFlight`/`Sampled`, so the FSM and the chain are two unwired encodings of one step.
- no_std tier 4 — `SymbolId` into the caller's `&[u64]` is the correct answer to the live cache length and Plan-3 reached it first; `cfg_attr(feature = "config")` gating on the new map types is right.
- RISC 3 — `FusedRegion { ops: Vec<BoundOp> }` makes `BoundOpKind` a nest of bound ops, the shape `bind.rs:160-162` rejects by name, and `ScheduleSet(u8)` re-encodes a capability set as a bitfield one step above the `bool` it replaces.
- bytes arithmetic 2 — the decomposition leaves a 125 MB "unexplained residual" against the brief's four factors, then carries it as a line item and *scales it by the codec factor* into D.7; the ms conversion excludes host and dispatch time entirely; the conclusion "met with 21% margin" rests on an ASSUMED 300 GB/s while the only measured achieved figure on this box is ~174 GB/s; union density is absent; `ffn_down` is asserted to scale with `d` with no block-granularity argument.
- quality gates 4 — per-lever EM/perplexity thresholds with kills, plus a secondary kill on GB/s failing to fall in proportion to `d`, which is the right shape for the ROW 180/181 risk.
- signatures 1 — `type In = CommandScript<'_>`, `type Out = LogitsView<'_>`, `Pipe<In = TokenWindow<'_>, Out = Draft>` and `(Encoded<'static>, StepInputs<'static>)` with "lifetimes elided in prose" do not type-check against `primitives.rs:91-102`, and four `impl Pipe` blocks carry associated types with no `fn call`.
- abandoned 5 — `FlashAttention`, `Timed<P>`, an `Executor` trait, `PlanBuilder`, a global `OnceLock<Device>` + mutex, fusion-leads ordering, and a host-side `RowSelector` killed by the landed probe's shape.

## Deciding axis per slot

- **Plan-2 first** — the only candidate whose bytes section reconciles with the census the brief mandates (ROW 287, 173.6 GB/s, the 5.96 ms host residual) *and* resolves both contested byte axes the task names: union density is a measured gate priced at its kill boundary, and the down-projection contraction axis forces group = 256 rather than being asserted away.
- **Plan-1 second** — sound algebra and the strongest abandonment section, lost on the tier and arithmetic axes: `Settings` on the alloc-tier bind path and an integer-dtype band operand that `map.rs:99-107` says no backend plumbs, plus a time model built from the instrument `dispatch-census.md:25` marks superseded and internally short by exactly the host residual it never names.
- **Plan-3 third** — its pipe surface does not type-check against `primitives.rs:91-102` (`'_` in associated-type position, `impl Pipe` blocks with no `call`) and its primary section reaches a positive verdict on the owner's target by scaling a self-declared unexplained residual and pricing it at an assumed bandwidth 1.7x the measured one.
