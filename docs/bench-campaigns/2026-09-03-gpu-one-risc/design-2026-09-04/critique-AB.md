# critique-AB (round 2, incumbent design-AB.md vs design-task.md + the superseding census)

Read-only, repo at `/Users/brianbruggeman/repos/slot-0/proxima`, HEAD `437e43b` (main). No cargo,
no build, no measurement run. Every `file:line` below was opened at HEAD unless the line is quoted
as the design's own (stale) citation. Findings are numbered most severe first; severity is a label,
not a filter.

---

## 1. The primary section's time model uses two constants the census contradicts; no route row reproduces, and the reachable-floor claim is understated by 2.3-3.6 ms

**Task §D ("THE PRIMARY SECTION"), P18/P19 (a number is not a result), P6.**

`design-AB.md:259-267` builds every route as `bytes/bandwidth + non-matvec + host`. Two of those
three terms do not survive the census.

**(a) Host = 5.96 ms is gone.** `dispatch-census.md:25-30` now states the steady-state
decomposition of record: **wall 28.82 = gpu_exec 27.52 + host 1.30** (emit 0.418, op_setup 0.158,
encode_dispatch 0.209, pipeline_lookup 0.013, readback 0.005, greedy 0.044,
build_position_inputs 0.002, unexplained 0.453), plan-HIT steps 3..7, and: "There is NO 5.96 ms
host residual." The design derived 5.96 as `33.00 − 27.04` (`design-AB.md:65-69`) from a wall that
the census line 26-27 says **includes two plan-miss steps that pay bind+compile**. The mechanism
of the 5.96 is therefore known and it is a population error: bind+compile amortised out of
plan-miss steps and then attributed to steady-state host work.

**(b) The design labels the same number DERIVED and MEASURED.** `design-AB.md:69` marks it
DERIVED; `design-AB.md:115` writes "Host residual 5.96 ms MEASURED today" and uses it as a time
constant. P18's provenance tag is not decoration; the design's own table disagrees with its own
prose eight lines later.

**(c) The 0.7 ms non-matvec constant contradicts §0.3's own warning.** `design-AB.md:116` sets
"Non-matvec 4.05 ms MEASURED today; 0.7 ms DERIVED at 264 dispatches", and every route row adds
`+0.7`. But `design-AB.md:84-86` states, correctly, that 616→264 "is worth at most 4.05 ms, and
only if per-dispatch cost is the whole of it." The 264-dispatch endpoint is 7 matvec + 1 attention
per layer (`dispatch-census.md:62`), i.e. the 32 attention dispatches **survive**, and the census
measures those 32 ops at **3.0-3.6 ms** (`dispatch-census.md:33-34`). Attention work is not
dispatch overhead: 0.134 GB of KV at 173.6 GB/s is 0.77 ms, so the attention arm is ~4x above its
own byte floor. A non-matvec floor of 0.7 ms erases measured GPU work.

**Recomputed route table** (design's own MB/token, `@173.6` MEASURED-derived, non-matvec floor
3.0 ms from the census attention arm, host 1.30 today / 0.5 after §B):

| route | MB/tok | design says | recomputed @173.6 (host 1.30 / 0.5) | recomputed @300 (host 0.5) |
|---|---:|---:|---:|---:|
| today | 4,167.6 | "24.01+4.05+5.96 = **33.0 MEASURED**" | 23.24+4.05+1.30 = **28.59** vs 28.82 measured | — |
| Route 1 | 1,522.2 | 9.97 | 8.77+3.0+1.30 = **13.07** / 12.27 | 5.07+3.0+0.5 = **8.57** |
| Route 2 (d_union .6) | 1,050.1 | 7.25 | 6.05+3.0+1.30 = **10.35** / 9.55 | 3.50+3.0+0.5 = **7.00** |
| Route 2 (d_union .35) | 748.6 | 5.51 / 3.70 / 3.07 | 4.31+3.0+1.30 = **8.61** / 7.81 | 2.50+3.0+0.5 = **6.00** |

Consequences: (i) the design's "today" row **does not reproduce** — `24.01 + 4.05 + 5.96 = 34.02`,
not the 33.0 it prints as MEASURED, a 1.0 ms arithmetic error in the anchor row (and 24.01
double-counts KV: 173.6 GB/s was derived over weights-only 4.034 GB at `design-AB.md:77-79`, then
applied to 4.1676 GB including KV); (ii) the "reachable floor" of `design-AB.md:280-284`
("~10.0 ms = 1.75x llama", "Route 2 ... ~7.25 ms = 2.4x") is understated by 2.3-3.1 ms; the honest
floors are ~12-13 ms and ~9.5-10.4 ms at the only measured bandwidth; (iii) 3.5 ms is missed by a
wider margin than stated at **every** cell including the favourable @400 one
(1.87+3.0+0.5 = 5.37).

## 2. §0.3, the section the design calls "the correction that reorders everything", is void — and it is the stated cause of the card order

`design-AB.md:57-86`, `1168-1171`, `1130`. Consequence 1 ("The host residual is 18% of the token
and is the ceiling on §B ... today's token cannot go below 5.96 ms ... §B's ... zero-allocation
budget, the one driver and the pure `plan()` are therefore not 'margin' — they are the first
blocking term") is false against `dispatch-census.md:25-30`: the host term is 1.30 ms, of which
the largest single item is 0.453 ms "unexplained", and §B in full is worth ≤0.8 ms. The census
also states the direction the design inverted: **"Gap 11.4 ms is GPU-side"** and steady-state
ratio 1.65x, not the 33.0/17.45 = 1.89x the design carries.

What breaks: `design-AB.md:1168-1170` is the explicit ordering rationale — "the 5.96 ms host
residual is the first blocking term, so §B's cards precede every byte lever except the two
measurements." Cards 5-8 (B1/B2/B3/B4) and cards 2-4 (A0/A1/A2) therefore sit ahead of **all**
byte levers on a justification that no longer exists, and CARD B4's gate (`design-AB.md:1130`)
reads "host residual reported against the 5.96 ms MEASURED baseline" — a gate against a number
that does not exist, which can neither pass nor fail. Note also that cards 14/15 (D5 f16 KV, D4a
output Q6_K→Q4_K) are pure codec selections with **no** dependency on cards 2-13; the ordering is
not forced by any precondition the design names, so 13 cards of refactor precede the first byte
lever against a task that names §D primary and an owner directive of "less work + same output".

## 3. `BlockCodec` — the type the pure-`plan()` signature rests on — does not exist, and the only shipped alternative is `#[cfg(feature = "std")]`

`design-AB.md:817-829` states: "`codecs` is `&[Option<BlockCodec>]` where `BlockCodec` is
**proxima-tensor's own enum** — FORCED-BY cB #8", and that `metal.rs:445-459` "keeps its job of
mapping `BlockCodec` -> `PackedCodec`". Verified: `grep -rn BlockCodec` over the whole repo
returns **0**. What `omega/src/metal.rs:450-456` actually maps is
`proxima_tensor::cpu::QuantizedBlock` → `PackedCodec` (`omega/src/msl.rs:788`), and
`QuantizedBlock<'a>` lives at `proxima-tensor/src/cpu.rs:3091` inside a module gated
`#[cfg(feature = "std")] pub mod cpu;` (`proxima-tensor/src/lib.rs:193-194`).

Downstream: the fix the design claims for cB #8 (crate-graph inversion) is asserted against a type
that must first be minted — with no card, no variant list, no tier statement — and CARD B1's gate
("`plan()` compiles and runs on a non-macOS target", `design-AB.md:1127`) is unprovable as
written, because the only shipped codec enum is std-gated and borrowed (`<'a>`), which also
collides with `Plan` being "a self-contained value" (`design-AB.md:859`). This is the same failure
`tiers-census.md` records for omega's std cell: "`proxima_tensor::{Evaluated, QuantizedBlock}`
unresolved: `std` does not enable proxima-tensor's std/cpu".

## 4. Signatures that do not type-check (judge-brief axis: "signatures type-check")

- **`StepOutcome` derives `Copy` around an `ArrayVec`** (`design-AB.md:771-781`).
  `arrayvec = "0.7.6"` (`Cargo.toml:133`); `ArrayVec<T, CAP>` has a `Drop` impl and is **not**
  `Copy`. `#[derive(Debug, Clone, Copy, PartialEq, Eq)] struct StepOutcome { accepted:
  ArrayVec<TokenId, MAX_ACCEPTED>, .. }` does not compile, and the design's whole pipe-boundary
  argument ("POD in, POD out", `design-AB.md:767`, §G row "POD-only pipe boundaries") is built on
  that derive.
- **`PlanKey` derives `Hash, PartialOrd, Ord` around `FusionRules`** (`design-AB.md:920-927`),
  which derives only `Debug, Clone, Copy, PartialEq, Eq` (`design-AB.md:579-585`). Does not
  compile.
- **`StepOutcome` also carries `StepCounters` by value** while today's counters are
  `#[cfg(feature = "instrument")]` (`omega/src/metal.rs:2719` region, `2824-2825`
  `snapshot_and_reset` calls) — the public `Out` type of the one `Session` pipe changes shape with
  a feature flag, which is unstated.
- **`#[serde(untagged)]` on `Offset`** (`design-AB.md:386-395`): `Static(i32)` and
  `Symbol(SymbolId)` are both a bare integer in TOML/JSON, so untagged resolves every integer to
  `Static` and `Symbol` is unreachable from data — in a design whose §C thesis is that a TOML
  author writes the domain. `Zero` and `Static(0)` are additionally two spellings of one value
  under a derived `PartialEq`, so CARD A0/A1's "`PartialEq` node-for-node to pre-card" gate
  (`design-AB.md:1124-1125`) depends on which spelling the 32-site migration happens to write.
- **Name collision in the pipe table** (`design-AB.md:762`): the pipe is named `Draft` and its
  `Out` is named `Draft`.
- `BarrierSet` (`design-AB.md:842`) vs `BarrierRange` (`design-AB.md:749`) vs
  `MAX_INLINE_BARRIER_SLOTS` (`design-AB.md:1077`) are three spellings of one concept across the
  `Schedule`/`Command`/sizing surfaces.

## 5. The zero-allocation budget omits the largest per-token allocation on the path — `kernel_cache_key`'s `String`, 616 per token

`design-AB.md:875-886` enumerates removed allocation sites. It does **not** include
`omega/src/msl.rs:930-933`, `pub(crate) fn kernel_cache_key(..) -> Result<String, EmitError>`,
called at `omega/src/metal.rs:3875` (`let cache_key = kernel_cache_key(bound, packed_operands)?;`)
inside `encode_op`, once per op per step. The comment immediately above it
(`omega/src/metal.rs:3871-3874`) states that on a pipeline-cache **hit** — "the steady-decode case"
— `emit` is never called and `kernel_cache_key`/`kernel_dispatch_shape` are exactly what still
runs. That is a `String` allocation (plus the `Vec`s inside `entry_name`/`operand_codecs`) 616
times per token on the very path CARD B4's gate `A` asserts is zero
(`design-AB.md:892-904`, `1106`). This is `lowering-audit.md` finding (7), verbatim, and the design
states at `design-AB.md:1338-1341` that it never saw that audit. Nothing in cards B1-B4 touches
`encode_op` or `pipeline_for`, so the gate as written fails on landing.

## 6. The lowering-audit findings are not accounted for, by the design's own admission, and two of them get worse under `Loop`

`design-AB.md:1338-1341` (`<<UNFINISHED>>` item 5): "no lowering-audit.md or tiers-census.md was
present in the input directory". Consequences, each grounded:

- **kernel cache key omits extents** (`lowering-audit.md` finding 1: `msl.rs:930-981` vs
  `cooperative_reduce_width` baking the extent-derived width into source at `msl.rs:4390-4406,
  4540`, served without re-emitting by `pipeline_for` at `metal.rs:2374-2400`). The design's
  `KernelId` (`design-AB.md:750`, `633`) is **undefined** — what it keys on is never stated — while
  §A.5 *relies* on key-insensitivity to the cache length ("one compiled kernel serves every cache
  length", `design-AB.md:482-484`). A `Loop` whose `BoundDomain` renders as a loop bound and whose
  stage set determines cooperative width makes the key strictly more extent-dependent. The latent
  wrong-kernel-served defect is inherited and enlarged, with no card.
- **`PACKED_ROW_BLOCK_SIMDGROUPS`'s only consumer is unreachable** (`omega/src/msl.rs:4322`;
  audit finding 6 — a swept constant that never reached a dispatch). The design adds **ten** new
  `sized::` consts (`design-AB.md:1071-1082`) with no gate that any of them reaches a dispatch;
  the standing battery (`design-AB.md:1094-1116`) has no reachability cell. This is the N==0 class
  the design cites elsewhere.
- **28 duplicated emitter functions, no shared lowering layer, `reduce_is_cooperative` ×3 with
  diverged signatures** (audit finding 8). CARD A2 (`design-AB.md:1126`) ports "5 emitters + CPU"
  to `Loop` in one card and the design explicitly declines a shared lowering layer; the
  duplication is multiplied across the largest new surface in the design.
- **27 of 43 lowering decisions are by NAME, 14 of them cargo features selecting kernel bodies.**
  The design removes two (the `fuse_cached_attention` bool, `classify_kind`) and **adds**
  `loop-fusion` plus four `FusionRules` flags that are part of the plan cache key
  (`design-AB.md:599-602`), i.e. net more by-name routing.

## 7. Tier claims are asserted where the tier cells are red today, and the crate the FSM actually lands in has no alloc tier

`tiers-census.md` (run at `f3c4e98`): omega `--no-default-features` **EXIT 101** (11 errors,
`msl.rs:2546` `.to_string()` without an alloc import), omega `--features std` **EXIT 101**, and
**`proxima-model-interop` has NO `alloc` feature**. The design:

- claims the alloc tier throughout §A/§B (`design-AB.md:682`, `1172-1177`, CARD 1's gate at
  `1653`-equivalent `design-AB.md:652-653`) and ships **no card** repairing omega's two red cells;
- lands CARD B4 (the `Step` FSM, `Session`, the decode closure replacement) and CARDs C1/C2 in
  `proxima-model-interop/src/{generate,…}` — the crate with no alloc tier at all — while stating
  the FSM is "proxima-tensor — alloc tier" (`design-AB.md:682`);
- incidentally targets `msl.rs:2546` for deletion in §C.3 (`design-AB.md:1043-1046`), which is one
  of the 11 no-default-features errors, without knowing or claiming it;
- `MAX_INLINE_CONSTRAINTS` default 2 carries overflow policy "spill (rare, cold)"
  (`design-AB.md:1073`) on the **same bind path** where `MAX_INLINE_STAGES` is given
  decline-rather-than-spill because a `SmallVec` spill is an unbudgeted allocation
  (`design-AB.md:524-528`). Two policies for one hazard. And §C.4 says sliding window is "a second
  `HalfSpace`" (`design-AB.md:1051`) — causal + window is exactly 2, so a common architecture sits
  on the spill boundary at the default.

## 8. CARD D0 does not match the device-ceiling harness that now exists, and it is two deliverables, not one

Branch `perf/device-streaming-ceiling` (`01b7a11`), file `omega/tests/device_streaming_ceiling.rs`
(read via `git show`). What the harness does: three **buffer-source** arms — no-copy resident over
the real openchat GGUF `mmap` (`create_no_copy_buffer`'s exact FFI + `StorageModeShared`), fresh
shared (`newBufferWithBytes_length_options`), private via blit; a streaming reduce with
`float4`/`uint4` strided loads and a live accumulator; grid width swept LOW/HIGH/WIDE; timed by
`commit()`→`waitUntilCompleted()` around exactly one dispatch, nothing subtracted; a **separate
empty-dispatch pipeline** for the fixed per-dispatch cost; `#[ignore]`d, host-local GGUF path.

CARD D0 (`design-AB.md:1122`, spec at `309-317`) asks for: "a pure-read kernel with the identical
access pattern (same **quant-block stride**, same threadgroup width, no arithmetic) at **three
problem sizes**, against a **memcpy-shaped control that should saturate**, CoV over 5 runs."
Mismatches, each of which changes what the card can conclude:

- the harness's access pattern is `float4`/`uint4` strided, **not** the Q4_K superblock stride the
  225 packed-row matvecs use (`omega/src/msl.rs:444`, `3184-3189`), so it cannot answer the
  design's own question ("is 400 GB/s wrong *for this access shape*");
- there is **no memcpy-shaped control**; the empty-dispatch pipeline measures per-dispatch cost and
  the blit fill is untimed, so the design's V5 degenerate control ("the control saturates or the
  400 figure is retracted") has no counterpart;
- the harness's primary variable is **buffer source** (no-copy mmap vs shared vs private) — a
  candidate mechanism for the 173.6-vs-400 gap that the design's §D.8 never names, and §D.8
  explicitly offers "no mechanism";
- D0 says "**No source change**" while the deliverable is a new kernel + test file;
- D0 bundles a second, unrelated deliverable ("Read per-tensor codecs from the gguf table") with a
  separate gate, so it is not one gated deliverable.

## 9. The first five cards are not 30-minute slices; two are multi-week

`design-AB.md:1122-1126`. D0 = two deliverables (above). B0 (`CountingAllocator` promoted to
`proxima-test`, two private copies deleted) is a slice. **A0** (`IndexPattern::compose` + a second
extent-resolution pass in `shape.rs:220-244`) is plausibly a day, not 30 minutes. **A1** is:
widen `AxisIndex::offset` `i32`→`Offset` (`proxima-tensor/src/map.rs:60-66`), change
`Extent::Symbolic(u16)`→`Symbolic(SymbolId)` (`op.rs:45-48`), add `Domain` to `Op::Reduce`
(8 fields, `op.rs:152-165`) and `Op::Elementwise`, migrate 32 struct-literal sites, teach
`shape::infer` and the CPU interpreter to honour a domain, and keep all **11** `specs/*.toml`
deserializing (`ls proxima-tensor/specs` = 11, confirmed) — one card. **A2** ports `BoundOpKind`
to `Loop` across the blast radius the design itself enumerates at `design-AB.md:1114-1116`:
74+65+39+22+19+16+12+8 = **255** sites plus examples and six `omega/tests/*`, five emitters and the
CPU evaluator, in one card whose only gates are "dispatch count still 616; B; M; X". A card that
cannot be bisected below 255 sites has no rollback granularity, which contradicts the design's own
rollback thesis (`design-AB.md:641-648`).

## 10. Fifteen types are used in signatures and never defined; the design discloses twelve

`design-AB.md:1330-1341`. The disclosed twelve are `Wiring`, `Bindings`, `BindingRange`,
`CommandRange`, `UniformRef`, `UniformPatch`, `BarrierRange`, `Grid`, `Budget`, `TokenWindow`,
`StepCounters`, `Completion`, `ResidentSet`. Undisclosed and also undefined: **`BlockCodec`**
(finding 3, presented as shipped), **`UniformBlob`** (`design-AB.md:843`), **`KernelId`**
(`design-AB.md:633, 750`), **`Residency`** (`design-AB.md:855`), **`Draft`**, **`BarrierSet`**,
**`ReduceInit`/`Keep`/`Fold`** are shipped so those are fine. What each undefined type needs before
the design is implementable, and what its absence blocks:

| type | missing | what it blocks |
|---|---|---|
| `Bindings<'p>` | field list; whether it holds device handles | whether `encode` is sans-IO/tier-3 at all (`design-AB.md:737-742`) |
| `BindingRange`, `CommandRange` | the plan-owned tables they index — `Plan`'s field list (`design-AB.md:831`) has no bindings or command table | `Command: Copy` "index-based, no interior lifetimes" is unverifiable |
| `Grid` | whether it is plan-constant | plan-time `Command` encoding is only valid once K/V is bound at capacity — **CARD B5, eight cards later** than CARD B1 that introduces `Schedule` |
| `UniformRef`, `UniformPatch`, `UniformBlob` | who owns the mutable uniform bytes and when they are patched | the zero-alloc-per-token claim and `&self`-vs-`&mut self` on the driver |
| `BarrierRange` vs `BarrierSet` | one name | `barrier_schedule`'s return type vs `Command::Barrier`'s payload |
| `KernelId` | what it keys on | pipeline-cache correctness (finding 6) |
| `StepCounters` | `Copy`, and its feature gating | `StepOutcome`'s derives (finding 4) |
| `Completion` | payload | the `InFlight → Sampled` transition's inputs |
| `ResidentSet`/`Residency` | field lists; how `register_checkpoint_mapping`'s offset keying survives | `Plan` as "a self-contained value" (`design-AB.md:855-859`) |
| `Budget`, `TokenWindow`, `Draft` | units / fields | `StepInputs`, `Halt::BudgetSpent`, the drafter pipe |
| `Wiring` | field list | `ProgramSpec::extend` — and `stack` is *defined as* a fold over `extend` (`design-AB.md:961-965`), so the defined type is driven by the undefined one |
| `BlockCodec` | existence | `plan()`'s purity + tier claim (finding 3) |

The other `<<UNFINISHED>>` items are live holes too: (3) `ScatterBounds::{Fault, Drop}` is CARD
D3b's one IR extension and its interaction with the binding `Domain` ("a dropped scatter is an
out-of-domain write") is unworked — that is the semantics of the elision lever, the second-largest
byte lever in §D.6; (4) sliding-window and ALiBi in §C.4 are asserted, not written as programs,
and §C.4 is the task's "name what a new architecture needs" deliverable.

## 11. `Session: Pipe` is claimed to be the shape that already exists; the shipped pipe is a different granularity and a different payload

`design-AB.md:764` and `0.1`: "`Session: Pipe<In = StepInputs, Out = StepOutcome>`, which is the
shape `LoadedModel::call` (`proxima-model-interop/src/generate.rs:1533`) already has". Opened:
the impl is at `proxima-model-interop/src/generate.rs:1542-1556` and reads
`type In = (String, usize); type Out = (Vec<u32>, String, bool);` — prompt → **the whole
generation**, not one pass. Granularity differs by N tokens, the `In` is an owned `String`, the
`Out` allocates a `Vec<u32>` and a `String` per call, and it ends in a bare `bool` — a rich→poor
collapse on the path the design is replacing that §E.4's information-destruction table
(`design-AB.md:1232-1243`) does not list. The design leaves this pipe untouched while claiming
zero steady-state allocation for the path it fronts.

## 12. `FusionRules` restores a destroyed capability set as four bools, and re-destroys the reason

`design-AB.md:579-597`, §E.4 row 1 ("capability SET → `bool`" restored by "`FusionRules` (four
independent fields)"). Four bools is the same rich→poor shape one level down: a backend cannot say
*why* it declines (`lowering-audit.md` finding 5: the fusion template misses silently, with no
rejection enum, unlike packed-row/tiled-gemm which have one), and R1's `FusionCost` profitability
decision records no reason either. Downstream: when a program stops fusing after a codec or shape
change, the only observable is a dispatch count, and the design's own tripwire
(`design-AB.md:1162`) — "a program that fuses on one backend and not another" — has no artifact to
name the cause.

Related and stronger: **R5 makes the same program numerically backend-dependent.**
`design-AB.md:551-555` says declining online-softmax "costs a second traversal ... different
arithmetic order", `599-602` says rules are in the plan cache key "because R5 reorders
floating-point arithmetic", and `583` says "wgpu declines `online_softmax`". So Metal and wgpu
evaluate one `Op` program to different bits, while the **X** gate (`design-AB.md:1111-1116`)
builds and tests wgsl/cuda/wgpu_driver against it and the standing **P** gate is parity ≤1e-4 to
the CPU evaluator. No cross-backend numeric contract is stated, and the tripwire "capabilities gate
*whether*, never *which*" is contradicted by R5 being a capability that changes which traversal is
generated.

## 13. Line citations do not resolve at HEAD; no base commit is pinned

`design-AB.md:4-6` claims "Every `file:line` was opened in this session". At HEAD `437e43b`
(`omega/src/metal.rs` last touched by `69eefb9`, HEAD~1), the counts hold but the lines do not,
from roughly line 1000 onward:

| design cites | actual at HEAD |
|---|---|
| `metal.rs:1631-1692` `classify_kind` | `metal.rs:1734` |
| `metal.rs:2183` `pack_uniforms` | `metal.rs:2286` |
| `metal.rs:2719` `snapshot_and_reset` | `metal.rs:2677` (doc), `2824-2825` (calls) |
| `metal.rs:1105` `MTLBarrierScope::Buffers` | `metal.rs:1207` |
| `commandBuffer` at 588, 1041, 1343 | 588, **1112**, **1446** |
| `thread_local!` at 252, 2477, 2933, 3007, 3150, 3246 | 252, **2580, 3036, 3110, 3253, 3349** |
| entry points at 468, 519, 535, 978, 1177, 1192, 1210, 1409, 1484, 1509, 1613 | 468, 519, 535, **1049, 1280, 1295, 1313, 1512, 1587, 1612, 1716** |
| `grep -c too_many_arguments spec.rs` = 17 (`design-AB.md:1036-1038`) | **18** |

The design pins no commit anywhere (contrast `dispatch-census.md:1` "main af918bb",
`tiers-census.md:1` "main f3c4e98"). CARD C2/C4's gates are literal grep counts
(`design-AB.md:1147, 1149`) and CARD B2/B3's are literal `grep -c` values
(`1128, 1129`); a gate stated as a count against an unpinned base is not re-provable from the
artifact alone (P16).

## 14. `Loop` is defended in a paragraph, and the defence names the finding

`design-AB.md:51-55`: "`Loop` is RISC at the `Op` face ... and it is **a multi-stage traversal
interpreter at the `BoundOp` face**, which five emitters must implement." Stated honestly, and it
is exactly the audit's charge against `CachedAttention` (task item 1: a macro-op carrying semantics
that is not `Op`/`ScalarOp`/`IndexMap` structure) relocated one level up: `Stage { reduced_axes,
body, fold, domain }` plus `StepArg::Stage(u16)` plus `output_axes` plus `out_scatter` is a new
execution vocabulary that five backends must each interpret, and the design supplies no shared
lowering layer for it (finding 6). The task asked for "a fusion RULE over that structure ... the
CPU evaluator and every emitter implement the same rule"; the design delivers the rules (R1-R5) but
also five independent interpreters of a new nested form, and R5 is explicitly a per-emitter theorem
("a lowering theorem per emitter", `design-AB.md:551`) — i.e. the same rule is implemented five
times, which is the shape the audit's finding 8 already measured as 28 duplicated functions.

## 15. Smaller findings, each grounded

- **`LayerInputs`/`KeyRoots` buy nothing at the type level, and the design says so.**
  `design-AB.md:1034-1035`: "every field is a `NodeId`, so this makes a swap a *named* error at the
  construction site, not a type error." P11's compile-time clause is "newtypes for every identifier
  so swapping arguments at a call site is a compile error." The honest admission is the finding:
  the struct is a relocation of a 23-argument list, and the shipped precedent `NodeId`
  (`proxima-tensor/src/op.rs:26-28`) is the newtype pattern it declines to apply.
- **`Cursor.accepted: u16` and `StepOutcome.accepted: ArrayVec<TokenId, _>`** are two sources of
  truth for one quantity (`design-AB.md:692`, `773`); likewise `Step::Halted(Halt)` and
  `StepOutcome.halt: Option<Halt>` (`709`, `780`) are two encodings of termination — in a design
  whose §B.1 thesis is "exactly **one** encoding of a step".
- **`KV_BUCKET_TOKENS` deletion gate is scoped too narrowly.** `design-AB.md:942` gates on
  `grep -rc KV_BUCKET_TOKENS proxima-tensor/src proxima-model-interop/src == 0`, but the constant
  is `#[cfg(all(feature = "std", feature = "kv-capacity-bucket"))]`
  (`proxima-tensor/src/sized.rs:325-326`), is asserted `== 32` by a test in the same file
  (`sized.rs:389-391`), and is generated from `proxima-tensor-runtime.toml`'s `[kv]` section — none
  of which the grep covers.
- **`FusionCost` defaults are DERIVED constants presented as tunables.**
  `design-AB.md:604-607`, `1076`: `bytes_per_second = 173_600_000_000` — a number derived from
  weights-only bytes over the ROW 287 matvec ms, which the census itself reports as **179 GB/s over
  4.169 GB** (`dispatch-census.md:33`). The design never reconciles its 173.6 with the census's 179,
  and bakes the unreconciled one into a build-time constant that decides what materializes.
- **The `dispatch_ns = 6_100` default** (`design-AB.md:606`, `1076`) is the *average* elementwise op
  time (1.76 ms / 290), i.e. per-op **work**, used as per-**dispatch** overhead in a cost model that
  decides fusion. That is the same conflation as finding 1(c), inside the optimiser.
- **The `X` gate lists `proxima-onnx`** (`design-AB.md:1113`) as a consumer to build per
  `BoundOpKind`-touching card, but no card owns the onnx port of `Loop`; the blast-radius table
  (`1114-1116`) does not count onnx sites.
- **`Domain` is `#[serde(default)]` on `Op::Reduce`** (`design-AB.md:428-433`): correct that
  `proxima-tensor/src/op.rs` carries **0** `serde(default)` today (verified) and 11 spec files must
  keep parsing — but `Reduce` also has `name: Option<String>` and derives `Eq`
  (`op.rs:151-165`), so `Domain` must be `Eq`; it is (`SmallVec<[AxisIndex; N]>`, and `AxisIndex`
  derives `Eq` at `map.rs:61`) **only if** `Offset` stays `Eq` — which the design's `Offset` does.
  No defect here; recorded because the design's minimality argument depends on it and did not state
  it.
- **Tests as evidence of shape.** `design-AB.md:1303-1306` explicitly declines to treat its two
  tests as shape evidence, and points at §A.4's minimality and §E.3's lint instead. That is the
  correct posture; the residual gap is that §E.3's lint (`1207-1228`) enumerates only the design's
  *own* types and does not run the lint over the shipped path it leaves in place — `LoadedModel`'s
  pipe (finding 11), `BoundOpBuilder` (`bind.rs:992-1003`), `ShapeTable::push`
  (`proxima-tensor/src/shape.rs:312-315` doc: "`In = Op`, `Out = (Op, Shapes)`") — so the claim
  "nothing here is a wrapper whose call site reads identically" is scoped to new types only.

---

## Axes not reached in the window

- `<<UNSCORED>>` observability beyond `KernelFamily`/`classify_kind` (the telemetry-export and
  per-family counter plumbing was not opened).
- `<<UNSCORED>>` the D3-calib router-fit card's statistics (least-squares fit against per-group L1
  mass; mass-recall gate) — not checked against any shipped sparse machinery.
- `<<UNSCORED>>` §D.2's `k' = 2.18` derivation and the prompt-lookup drafter's acceptance model.
- `<<UNSCORED>>` `proxima-onnx` blast radius.
