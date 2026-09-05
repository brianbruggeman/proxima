# Critique of design-B (read-only; repo at /Users/brianbruggeman/repos/slot-0/proxima, HEAD f3c4e98)

Every `file:line` below was opened in this session. Design cites main 3b9735e/af918bb; the
worktree is f3c4e98, so ranges are checked with a +/-3 line tolerance and drift is noted where
it changes meaning.

## Axis scores (one-line justification each)

| axis | score | justification |
|---|---|---|
| risk surface | HIGH | card 1 rewrites the `BoundOpKind` match in 20 files (74 sites cpu.rs, 65 bind.rs, 39 msl.rs, 22 wgsl.rs, 19 cuda.rs, 16 metal.rs, 12 spec.rs, 8 wgpu_driver.rs + onnx/model-interop/6 omega tests) behind a Metal-only gate |
| ordering | BROKEN | `BandBound::Dynamic` (card 3) requires the integer `cached_len` that lands in card 8; D1b needs card 8's `MaskSource` and card 5's `Cursor` but is 4th in the D order; two orderings (E.1 table, D.7 list) never merged |
| rollback | ABSENT | no card is feature-gated or dual-represented; card 3 deletes a variant + 11 companions "in one commit"; revert is the only mechanism, and `bind_with_fusion`'s existing `cached-attention-streaming` cfg gate (bind.rs:2642-2648) is never named |
| missing steps | HIGH | the rank-128 predictor's 75M parameters have no training/storage/distribution step; the selector's own dispatch cost is uncounted; the integer-buffer plumbing map.rs:100-107 demands is uncosted |
| hidden coupling | HIGH | `plan()` in proxima-tensor takes `PackedCodec`, which is defined in `omega/src/msl.rs:788` (omega depends on proxima-tensor, not the reverse); `scripts/proxima-tensor-gate.sh` cells and wgsl/cuda/onnx consumers unnamed |
| observability | MIXED | every card carries a counted gate with a non-zero N assertion (strong), but the two headline gates -- dispatch count and ms/token -- are Metal-only and prove nothing about the 4 other backends card 1 changes |
| scope discipline | MIXED | D.4b has a gate but no card, no route row, and no selector construction; §C.1 `ProgramSpec::stack`/`extend` is a new composition surface the task's "generic" requirement asks for, so it is in scope |
| pipe questions | PARTIAL | `Loop`/`Band`/`Schedule` answer both questions in writing, but `poll_complete` is asserted to be a `Pipe` and cannot be (primitives.rs:101), and `Advance` cannot produce `&'p mut` from `&self` |
| no_std tier | FAILS AS WRITTEN | `FusionRules`/`FusionCost` derive `Settings`/`Serialize` and live on the bind path, but proxima-tensor's `config` feature is `["std", "dep:bon", "dep:conflaguration", "dep:serde", ..]` (Cargo.toml:37) and lib.rs:103-104 says "the `alloc` tier never sees it" |
| RISC claim | SPLIT | `Loop` leaves `Op` untouched (the vocabulary op.rs:55-56 promises never grows, does not), so RISC holds at the TOML face; at the `BoundOp` face `Loop` is a nested program with its own scope rules, and the design never states the split |
| bytes arithmetic | ARITHMETIC OK, INPUTS WEAK | every table recomputes exactly (checked below); the two largest multipliers (`a = 2.5`, `e = 0.80`) and the 161 GB/s divisor are unsourced, unvalidated, and from a measurement the audit marks superseded |
| abandoned designs | 4 of 6 CONSTRAINT-DRIVEN | FlashAttention, the `Fuse` stage, `Arc<Mutex>`, today's device-owning `Plan` are constraint-driven; the two "abandoned by the byte probe" items were not abandoned at all (both levers survive, only their gates changed); the `Decide` stop type was already settled by owner rule, not by this design |

---

## Findings, most severe first

### 1. Route 1 reaches <=1.4 GB only through an unsourced acceptance rate, and the design calls Route 1 "no discovery-loop lever"

D.1 sets `a = 2.5` with the tag DERIVED (design line 79). It is not derived from anything --
no measurement, no citation, no literature pointer, no provenance. It is ASSUMED by the design's
own taxonomy. It is also the single largest factor in Route 1: recomputing D.7's Route 1
without it gives 3.318 GB/token, 2.4x above the <=1.4 GB target the task sets (task line 92).
Every other Route 1 lever combined moves 4.168 -> 3.318 GB (a 1.26x), and `/ 2.5` supplies the
remaining 2.5x.

Worse, the card that lands it (E.1 D1b) is an **n-gram prompt-lookup drafter** -- the weakest
drafter in the family -- and its gate is "100 tokens byte-identical to k=1 greedy at 3 seeds",
which validates that *verification* is correct and says nothing about `a`. There is no kill
criterion on acceptance anywhere. The design's summary "≤1.4 GB/token is reached by Route 1,
which contains no discovery-loop lever" (line 336) is false by the design's own definition of a
discovery loop (line 231: "the correct output is unknown"): the acceptance rate of an n-gram
drafter on this model and prompt distribution is exactly an unknown metric on held-out data with
no oracle. It gets a kill criterion for D.3 and none for D.1.

Downstream: the one route the design presents as low-risk and gate-able is load-bearing on the
one number it never measured, and the ordering (D.7 order item 4) puts it before the
instrumentation (item 5) that could size it.

### 2. The entire time decomposition is built on a measurement the audit input marks superseded

design-B's 24.991 ms packed-row, 3.515 ms attention, 2.978 ms elementwise, 0.851 ms cooperative
(lines 30, 292, 325) come from ROW 281/282 via dispatch-census.md:25-27, which states the
profile ran with **serialized command buffers**. audit-2026-09-04.md:22 (finding #15,
`metal.rs:1509`) says of exactly that instrument: "Op-timed profiler measures
one-command-buffer-per-op, not production; decompositions built from it measure a different
program -- superseded by the in-buffer ablation (ROW 287)."

design-B builds five load-bearing quantities on it: (a) the 161 GB/s achieved bandwidth, (b) the
whole of D.6 and its "2.5x gap I have no mechanism for", (c) the "6.3 ms that §A/§B buy",
(d) `FusionCost::dispatch_ns` default 10_300 (line 648), and (e) every `@161 GB/s` column in
D.7. ROW 287 is never mentioned in the design. Production runs one command buffer per token with
concurrent dispatch (task lines 15-19), so the per-op serialized times are measuring a different
schedule than the one being optimized.

Under the design's own §18 discipline, (c) is a causal claim ("§A/§B buy the 6.3 ms") whose only
artifact is the superseded instrument.

### 3. 161 GB/s is internally inconsistent with the design's own dispatch constant

The design divides 4.03 GB by 24.991 ms to get 161 GB/s (line 292), and separately sets
`dispatch_ns = 10_300` as a MEASURED default from the same census (line 648). 225 packed-row
dispatches x 10.3 us = 2.32 ms of the 24.991 ms is, by the design's own constant, dispatch and
not bytes: 4.03 GB / 22.67 ms = 178 GB/s. The "2.5x gap" of D.6 is 2.25x on the design's own
arithmetic.

Compounding it, the `@300 GB/s` column double-discounts: 161 GB/s empirically embeds per-dispatch
cost (225 dispatches remain in every route), while 300 GB/s is a clean-bandwidth figure, and the
design then adds a flat `+ ~0.7 ms` to both. And the `today` row reports 33.0 MEASURED under a
column header that says `+ ~0.7 ms dispatch` while 33.0 - 25.9 = 7.1 ms; a measured total and
derived projections share one column with different arithmetic.

Separately, 161 GB/s is derived from the **packed-row arm only** (weights, 4.03 GB) and then
applied to 4.168 GB, which includes 0.134 GB of KV whose time already sits inside the 3.515 ms
attention term the design counts a second time as removable dispatch overhead. The KV term is
double-counted.

### 4. The FFN down-projection cannot be elided by the row-gather card, and Route 2 misses the target without it

D.3's table (lines 160-164) takes `(1 - e)` on all three FFN matrices, stating "reads `(1-e)` of
gate/up rows and `(1-e)` of down's **columns**" (line 157). The emitter card (E.1 D3a) lifts a
**row-axis `Lookup`** on the packed-row kernel and argues "fetch `row = rows[j]` once per output
row, then read that row's quant blocks contiguously exactly as today" (line 227).

That argument holds for gate/up, whose elided axis is the *output* axis. It does not hold for
down: `ffn_down` is stored with the 14336 hidden axis contiguous (it is the contraction axis of
the packed-row reduce -- `PackedRowBlock { weight, other, reduce_dim, codec }`,
`msl.rs:3184-3189`), so eliding hidden units is an **intra-row gather at Q4_K superblock
granularity** (256 elements/block), not a row-count reduction. At e=0.80 with unstructured
selection, essentially every 256-element block still contains a selected column, so down's bytes
do not fall at all unless the selection is made block-structured -- which is neither proposed nor
gated.

Recomputing Route 2 with down at full bytes: FFN = 0.2 x 2.114 (gate/up) + 1.057 (down) = 1.480;
weights = 1.480 + 0.043 + 0.755 + 0.074 = 2.352; + f16 KV 0.067 = 2.419; / 2.5 = **0.968
GB/token** -> 6.0 ms @161, **3.9 ms @300**, +0.7 dispatch = 6.7 / 4.6 ms. The owner's 3.5 ms is
then unreachable on **both** routes at any bandwidth <= 400 GB/s. The one card that decides
whether §D reaches the target is the one whose access pattern the design does not analyse.

### 5. §B.7 turns `Band` from a stated hint into a bytes-correctness requirement, and §D.1's own mask defeats it

R4's design is explicitly "purely a profitability hint: the body still contains the masking
`Select`, so an executor that ignores this produces identical results, slower" (lines 462-465,
590-592). §B.7 then binds K/V at session **capacity** and relies on the band to supply the live
length (lines 962-967).

Under that combination, an emitter that declines the band, or a program whose mask R4 fails to
recognize, traverses `capacity` rather than `cached_len` -- at capacity 8192 and t=512 that is
16x the KV bytes, 2.147 GB instead of 0.134 GB per token, which alone exceeds the entire <=1.4 GB
budget. "Correct and slower" is true; "correct and inside the byte budget" is not, and §D's whole
arithmetic silently assumes the band always fires. `FusionRules::band` is a runtime-settable bool
(line 556) that can turn it off.

The conflict is internal, not hypothetical: D.1's tree/`Supplied` mask is stated to be data
(`MaskSource::Supplied`, line 1057) for which "R4 simply finds no derivable band" (line 110). A
supplied mask operand has no `Greater`/`Iota` structure at all, so no band derives on `t` either
-- D.1 and B.7 cannot both be on at once without the KV traversal reverting to capacity. Card 4's
gate (`plan_misses == 1 over 128 steps`) passes while this happens.

### 6. Three signatures do not type-check; one of them is the encode hot path

- `pub fn encode<'p>(plan: &'p Plan, bindings: &'p Bindings<'p>, out: &'p mut [Command<'p>]) ->
  Result<&'p [Command<'p>], TensorError>` (lines 908-909). `out` is mutably borrowed for `'p`,
  the plan's lifetime, and the return borrows it for `'p` too. The command scratch buffer is
  therefore locked for as long as the plan lives: **exactly one encode per plan, ever.** This is
  the lifetime-locking trap -- the driver cannot refill the buffer it must refill every token.
- `Advance: Pipe<In = TokenId, Out = ReadyStep<'p>>` (line 822) must yield
  `ReadyStep { scratch: &'p mut StepScratch }` (line 779), but `Pipe::call(&self, input)`
  (`proxima-primitives/src/pipe/primitives.rs:101`) takes `&self`. A `&self` method cannot hand
  out a `&'p mut`. No interior mutability is proposed.
- `pub struct SampledStep { cursor, logits: LogitsView, counters }` (line 783) carries no
  lifetime parameter while `LogitsView` must point at a readback staging buffer. Either it owns
  the logits (32002 f32 = 128 KB allocated per token, breaking B.6's zero-allocation budget) or
  the type is missing its lifetime. The design's own budget table (B.6) never lists the readback
  buffer.

### 7. `poll_complete` is asserted to be a `Pipe` and is not; "one Metal edge" is contradicted three lines later

`Pipe::call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>>`
(`primitives.rs:101`). The design writes `poll_complete(&self, step: &mut InFlightStep<'_>, cx:
&mut Context<'_>) -> Poll<Result<SampledStep, MetalError>>` (lines 876-877) and then states
"Both are `Pipe` impls" (line 881), while B.4's table lists `Complete` as
`Pipe<In = InFlightStep, Out = SampledStep>` (line 819). `&mut In`, a `Context`, and `Poll` match
none of `Pipe`/`SendPipe`/`UnpinPipe`/`UnpinSendPipe` (primitives.rs:91-178). §E.3's closing
claim "No behaviour in this design is a non-pipe" (line 1180) fails on its own driver.

The headline says "`omega` shrinks to one driver with one edge: `submit(&[Command])`" (line 17).
B.4's table marks **two** rows "the Metal edge" (lines 818-819), and `residency()` (line 871)
allocates arena slots, allocates uniform buffers and compiles kernels -- a third device-touching
entry the summary does not count.

### 8. `plan()` is placed in proxima-tensor but takes an `omega` type

`pub fn plan(program, symbols, codecs: &[Option<PackedCodec>], outputs, config) ->
Result<Plan, TensorError>` under the comment "proxima-tensor -- no_std + alloc, no device, no
`omega` dependency" (lines 706-711). `PackedCodec` is defined at `omega/src/msl.rs:788`, and
`omega` depends on `proxima-tensor` (omega/Cargo.toml:21,32). The signature inverts the crate
graph in the same breath as the comment denying it. `proxima-tensor` currently reaches quantized
blocks through its own `QuantizedBlock` (`cpu.rs:3091`), and `metal.rs:445-459` is the mapping
between them -- so the design has moved the mapping's *output* type upstream of its owner.

### 9. no_std tier: the config-derived types cannot exist at the alloc tier where the design puts them

`FusionRules` and `FusionCost` are declared in `proxima-tensor/src/bind.rs` with
`#[derive(.., Builder, Deserialize, Serialize, Settings)]` (lines 545, 640) and are parameters of
the bind path. In this crate, `config = ["std", "dep:bon", "dep:conflaguration", "dep:serde",
"smallvec/serde"]` (`proxima-tensor/Cargo.toml:37`), and `lib.rs:103-104` states plainly:
"`config`: the TOML/serde face, std-only, per the layering caveat in `rust.md`. The `alloc` tier
never sees it."

So the design's claim "Defaults are seeded from `sized::` consts so the no_std/no_alloc tier has
the same knobs (conflaguration bridge, §C.4)" (lines 565-566) is contradicted by the crate's own
feature layering: at the alloc tier there is no `Settings`, no `bon` builder, no serde. Either
the derives are cfg-gated (unstated, and then the two tiers have different types on the bind
signature) or bind moves to std-only (which defeats §B.2's pure-plan goal).

The gate that catches this is never named: `scripts/proxima-tensor-gate.sh:109-115` runs
`alloc tier check`, `alloc tier clippy`, **`std without config`**, and `config alone clippy`.
Card 5's gate mentions only `--no-default-features --features alloc` for `proxima-tensor` and
nothing for `omega`, `wgsl`, `cuda`, or the `std without config` cell that `FusionRules` breaks
first.

### 10. The "indices are already integers" claim is a type-level fact the backends do not implement

C.3 asserts `cached_len: Int32` is "Forced by two things, not taste: `BandBound::Dynamic`
requires `DType::is_integer` (`dtype.rs:57`), and `IndexMap::Computed`'s indices already require
it (`shape.rs:283-291`) ... one function, not a second exemption list" (lines 1063-1068).
`shape.rs:283-291` does check it (verified). But `map.rs:100-107` states the operational reality:

> "every backend this crate ships (`cpu.rs`'s interpreter, `omega`'s Metal driver) carries every
> buffer as f32 -- **including `indices`** -- rather than plumbing a second integer-buffer kind
> through the stack for this one case. ... **Lifting the ceiling means adding real integer
> buffers, not raising a constant.**"

So both `cached_len: Int32` and D.3's selector output require plumbing an integer buffer kind
through `cpu.rs`, `msl.rs`, `wgsl.rs`, `cuda.rs` and the binding tables -- a cost the design
prices as a dtype change on one line (`generate.rs:2323-2324`). `bind::index_node_ids`
(`bind.rs:2746`) exists precisely because index nodes are the exception to the float-only
executor; extending it to band operands is not "one mechanism" but a second exemption of the
same kind.

### 11. A claim about existing code the file does not support: the fuse bool does not suppress prologue fusion for wgpu/cuda

A.4 (line 549) and E.4 (line 1188) state that `fuse_cached_attention: bool` "made wgpu/cuda
decline prologue fusion they can render, merely because they cannot render attention."
`bind.rs:2630-2634` says the opposite: wgpu and cuda "call this directly with `false` -- the
fused rewrite never fires for them, and **the plain elementwise/reduce chain `bind_plain`
already produces is what they emit**", and `bind_with_fusion` at 2641 begins by calling
`bind_plain` for everyone. Prologue fusion lives in `bind_plain` and fires for every backend.
The bool collapses a capability set (a real finding, kept), but the specific harm attributed to
it is not in the file.

Also unnamed: the fused arm is `#[cfg(feature = "cached-attention-streaming")]`
(`bind.rs:2642-2648`, `omega/Cargo.toml:21`). Card 3's gate is `grep -c CachedAttention == 0`
against a variant that is already behind a cargo feature the design never mentions -- so the
gate can pass in a default build while the fused path still exists in a featured one.

### 12. `classify_packed_row_block`'s gather gate is not at the line range the D3a card names

D.3 and D.7 both cite "`classify_packed_row_block` requires `gather_count == 0`
(`msl.rs:1039-1052, 1389`)" (lines 221, 341), inherited verbatim from byte-levers-probe.md:19-21.
`msl.rs:1039-1052` is `reduce_is_cooperative`, whose gather gate is at 1047;
`classify_packed_row_block` begins at `msl.rs:1389`; and there is a **third** gather exclusion at
`msl.rs:1218` (`if gather_count(resolved) > 0`). The D3a card is written to lift one gate and
there are at least three routing sites that divert a gathered reduce. Card scope is
under-specified against the file.

### 13. Card 1 is a big bang across 20 files with a Metal-only gate

Card 1 lands `BoundOpKind::Loop`, converts `Elementwise` and `Reduce` to one-stage `Loop`s, and
ports "5 emitters + CPU" in one step (line 1102). Measured blast radius
(`grep -rn "BoundOpKind::"`): cpu.rs 74 sites, bind.rs 65, msl.rs 39, wgsl.rs 22, cuda.rs 19,
metal.rs 16, spec.rs 12, wgpu_driver.rs 8, plus `proxima-tensor/examples/scaling.rs` (5),
`proxima-tensor/tests/rewrite_law_equivalence.rs`, `proxima-model-interop/src/bind.rs`,
`proxima-onnx/examples/mnist_diag.rs`, `omega/src/error.rs`, and six `omega/tests/*`. The design
names none of the test, example, onnx or model-interop consumers, and offers no dual
representation (no `Loop` alongside the two kinds it replaces) so there is no incremental path
and no rollback short of reverting the commit.

Card 1's gate is "dispatch count still 616; 100x byte-identical logits; ms/token within CoV" --
all Metal. Nothing in the gate exercises wgsl, cuda, wgpu_driver, onnx, or the `rewrite_law_
equivalence` test, which are precisely the consumers the port breaks silently.

### 14. Card ordering inverts two real dependencies

- `BandBound::Dynamic { slot }` requires the slot's dtype to satisfy `DType::is_integer`
  (line 485), and C.3 says `cached_len` is Float32 today (`generate.rs:2323-2324`). `Dynamic`
  bands land in **card 3** and card 4 depends on them; `cached_len: Int32` lands in **card 8**.
  Cards 3 and 4 cannot pass before card 8. E.5's test asserts
  `BandBound::Dynamic` at card 3's gate (line 1227).
- D1b (drafter + verify) needs `MaskSource::Supplied` for the tree mask (card 8) and `Cursor`
  (card 5), but sits 4th in D.7's order, before both.
- Two orderings exist and are never merged: the card table 0..8 then D5..D3b (E.1), and the
  risk-ascending list 1..7 (D.7 line 346) that starts with D5. Which sequence a worker follows
  is undetermined.

### 15. The elision selector's own cost is nowhere in the arithmetic

D.3's selector program (lines 192-202) is, per layer: one rank-128 sketch (two matvecs), one
`Greater` elementwise over 14336, one `Keep::Scan` prefix sum over 14336, one `Select`, and one
scatter-`Reduce` over `Iota(14336)` -- five to six additional dispatches, each serially dependent
on the previous, before the gate/up matvec can start. x32 layers = 160-192 dispatches/token added
to the claimed 264. At the design's own `dispatch_ns = 10_300`, that is **1.6-2.0 ms/token**,
over half the 3.5 ms target, and it appears in no table in §D.7 and in no dispatch count in §A.5.

The `Keep::Scan` step also has no shown Metal path: `reduce_is_cooperative` requires
`keep: Keep::Reduce` (`msl.rs:1041-1042`), so a scan is off the cooperative route, and the
design never states which kernel family renders a 14336-wide prefix scan.

### 16. The rank-128 predictor's 75M parameters do not exist and the design never says where they come from

D.3 prices the predictor at 2.36M params/layer, 42.5 MB/token (lines 166-167) and ablates over
rank {64,128,256} (line 238). It never names how the scorer is fit: no training data, no
objective, no procedure, no compute cost, no storage format, no checkpoint-distribution step, no
statement of whether it is per-model or per-checkpoint. That is a missing step in the largest
byte lever (2.5 GB) and it is the step that decides whether Route 2 exists at all. The
discovery-loop framing (kill criteria, degenerate control, ablation) is applied to the
*evaluation* of a scorer the design has no plan to produce.

Relatedly, the pre-registered hypothesis "exact-match-at-64 >= 0.95 at e = 0.80" is stated for a
SwiGLU/SiLU model. The contextual-sparsity results this construction descends from are for
ReLU-family MLPs; the design asserts the hypothesis with no note that the activation is not
ReLU. Correctly gated (kill at 0.90), but the arithmetic in D.7 presents Route 2 as the path to
the owner's number with no probability attached to the hypothesis holding.

### 17. `Loop { stages }` is RISC at the `Op` face and CISC at the `BoundOp` face; the design never states the split

**For RISC.** `Op` is untouched -- the vocabulary that op.rs:55-56 promises "never grows" does
not grow, `ScalarOp`'s 17 variants (op.rs:60-78) are unchanged, and attention is still seven
ordinary ops (A.0, verified expressible: `Greater`, `Select`, `Maximum`, `Exponential`,
`Divide` all exist; `Op::Iota`'s doc at op.rs:211-222 states the causal-mask composition is why
the variant was added). Net -2 `BoundOpKind` variants and -11 companions. R1-R5 carry no model
name, no operand count, no stride literal.

**For CISC.** `BoundOpKind::Loop` carries `SmallVec<[Stage; MAX_INLINE_STAGES]>` where each
`Stage` is `{ reduced_axes, ComposedBody, Option<Fold>, Option<Band> }` and bodies reference
earlier stages via `StepArg::Stage(u16)` with a backwards-only scoping rule (lines 453-489).
That is a nested program with its own scope, its own accumulator liveness, and its own lowering
theorem -- strictly more expressive than the `CachedAttention` macro-op it replaces. Compare
what an emitter must implement today: three fixed shapes (`Elementwise`, `Reduce`,
`CachedAttention`, bind.rs:225-294), each with a closed field set. After: a general multi-stage
traversal interpreter with cross-stage register allocation and an optional band per stage, in
five emitters. The design's escape hatch ("any emitter may decline without changing results",
line 15) applies only to **R5**; no emitter may decline `Loop` itself, which is the whole
matcher surface. `BoundOp::element_body`/`split_axis`'s total-accessor special cases
(bind.rs:303, 326, 409) delete, but the accessor problem returns as "which stage's body?".

The honest statement -- RISC preserved where the TOML face lives, macro-op relocated one level
down where five emitters pay for it -- is not made anywhere in the design.

### 18. `MAX_INLINE_STAGES` is a fixed capacity with no overflow policy

`stages: SmallVec<[Stage; MAX_INLINE_STAGES]>` (line 493) and the const is added via build.rs
(line 1080). Nothing states what happens when a fusion chain exceeds it: does R2 decline, does
the builder spill to heap (SmallVec does, silently allocating on the *bind* path), or does bind
error? Fixed capacity is exactly the constraint that makes an overflow policy a real decision;
here it is unanswered, and the silent-spill default is an unbudgeted allocation in the crate the
design is making no_std-clean.

### 19. R4 is a peephole matcher -- the thing the design condemns in `CachedAttention`

R4 scans for `Select(Greater(A, B), NEG_INF, x)` (line 586). That is a syntactic pattern on one
spelling of a mask. The same mask written as a multiplicative 0/1 mask -- which is the form the
repo already uses on the two-range path, `is_future` as an `Op::Input` built by `causal_mask`
(design's own line 108, spec.rs:2360-2364) -- produces no band. So does `Equal`-based masking,
a comparison split across two elementwise nodes, or a `Greater` with reversed operands composed
through a `Negate`. Audit item 1's complaint is "a program that is the same attention laid out
differently silently falls back"; R4 reproduces that failure mode one level down, and §5 above
shows the fallback is no longer merely slower once B.7 lands.

### 20. Two parallel state machines encode the same step, and 13 types serve one token

`Step::poll_advance` (an exhaustive match over five variants, line 795) and the seven-stage
`AndThen` chain (lines 825-830) both encode bind -> encode -> submit -> complete -> sample.
Neither is declared authoritative, and no rule says which one a caller uses. The pipe chain is
also a straight line while decode is a cycle: `Advance: In = TokenId, Out = ReadyStep<'p>` (line
822) closes a loop `AndThen` cannot express (`AndThen` is strictly `First::Out = Second::In`,
primitives.rs:203-221), and the design never names what drives the iteration.

Compensator count for one step: `ReadyStep`, `BoundStep`, `EncodedStep`, `InFlightStep`,
`SampledStep`, `Step`, `Cursor`, `StepScratch`, `Bindings`, `Commands`, `SubmitTicket`,
`StepCounters`, `LogitsView` -- plus seven stage types. Asking the design's own diagnostic
(*what does this need in order to be used?*) of `Step`: it needs five typestate structs, a
scratch, a bindings table, a command buffer, a ticket, and two pipes -- and the two lifetimes in
finding 6 show at least two of them do not hold together.

### 21. Runtime-settable fusion changes numerics and the plan key

`FusionRules` and `FusionCost` are `Settings` with env prefixes `TENSOR_FUSION` /
`TENSOR_FUSION_COST` (lines 546, 641). R5 explicitly changes arithmetic order ("same dispatch
count, same buffers, **different arithmetic order**", line 606) and the cost model changes which
nodes materialize. So an environment variable changes the bound program and its floating-point
results. The standing gates ("100 tokens byte-identical", "parity <= 1e-4") then hold only for
one env configuration, and the plan cache key (§B.7, `symbols = [new_count, capacity]`) does not
include the rules -- two processes with different `TENSOR_FUSION_*` produce different programs
under the same key. No precedence is stated between a backend's `const RULES: FusionRules`
(line 566) and the env-sourced `Settings` values.

`FusionCost::bytes_per_second`'s default is 161_000_000_000 -- the number D.6 says it has "no
mechanism for" and may retract -- baked as the global cost constant that decides fusion for
cpu, wgpu, cuda and metal alike. One measured-on-Metal figure of unexplained provenance
arbitrates fusion on every target.

`plan()`'s signature (lines 707-711) takes `config: &PlanConfig` and no `FusionRules`; A.4 says
rules are a consumer capability passed at bind. How they reach `plan()` is unspecified.

### 22. The allocation-counter test does not compile and its infrastructure does not exist

B.6's test (lines 942-952) references `before_mark`, which is never bound (only `before` is).
`CountingAllocator` exists nowhere in `proxima-tensor` or `omega`; the only instances in the
workspace are private to `proxima-telemetry/tests/elevation_memory.rs:42` and
`proxima-telemetry/benches/bench_lossless_producer_assist.rs:38`. So the test that proves the
headline zero-allocation budget requires a third copy of a global allocator shim, uncosted and
uncited. The assertion of `produced.len() == STEPS` (the N != 0 discipline) is the strongest part
of the design's observability and it sits on a fixture that does not exist yet.

The budget itself is also incomplete: B.6's table (lines 921-934) lists 11 removed sites and
omits sampling (`sample_next_token`, `proxima-tokenizer/src/sample.rs:277`, cited elsewhere in
the design), detokenization, the logits readback staging buffer, and `SmallVec` spill on
`stages`/`output_axes`/`BarrierSet`.

### 23. §C.1's genericity claim is falsified by `Op::Reduce`'s own shape

C.1 states "An SSM/Mamba layer is data: its scan is `Keep::Scan` (`op.rs:139-147`)" (line 1022).
`Op::Reduce(Reduce)` carries **one** operand: `pub operand: NodeId` (`op.rs:157`) and one
`body: ScalarOp` (`op.rs:155`). A selective scan is `h_t = a_t * h_{t-1} + b_t * x_t` -- a linear
recurrence with two per-step data streams and a composed body. It is not a single-operand,
single-`ScalarOp` prefix fold, and expressing it requires either a fused body over two operand
streams inside the fold (which is what `Loop`'s stages provide at the *bound* level but not at
the `Op` level a TOML writes) or the (a, b) semiring composition -- i.e. exactly the
**tuple-valued reduce monoid** the design's own "Contested" section refuses (lines 20-26). The
repo has `Qwen35SsmShape` (`generate.rs:731`) in production, so this is not hypothetical: the
"new architecture = data only" claim fails on an architecture already shipped.

`ReduceInit::FirstElement`'s own doc (`op.rs:129-131`) records the same tension: "no synthetic
identity exists for a `(value, index)` accumulator."

### 24. §D.4b is scope without a card, without a construction, and without a route row

D.4b (lines 252-259) proposes candidate-set logits at c=512 saving ~105 MB, gated at top-1 recall
>= 0.999, selected "by a cheap sketch" -- the sketch is never constructed, its bytes are never
counted, and the design elsewhere states there is no top-k in the algebra (line 188). D.4b
appears in neither route table nor the D.7 order nor E.1's card list. It is a stated lever with a
kill criterion and no path to being built or measured.

### 25. Q3_K quality thresholds are inferred for a codec the design will reach by requantizing an already-lossy checkpoint

D.2 states the checkpoint is Q4_K_S, so "a requantizer in `proxima_gguf::quant`" is needed (line
141). Quantizing Q4_K -> Q3_K is double quantization and is strictly worse than a native Q3_K
from full-precision weights, which is where any published Q3_K quality expectation comes from.
The kill thresholds (exact-match-at-64 >= 0.90, mean KL <= 0.05 nats) are stated without that
caveat, and D.7's Route 1 depends on Q3_K clearing them. `metal.rs:445-459` also already maps
`Q4_0` and `BFloat16` to codec slots (lines 454-456), which the design's D.2 inventory omits.

### 26. Entry-point count is wrong in the design and in its source

B.4 says "Nine entry points today" and then lists ten (lines 837-842). `omega/src/metal.rs` has
eleven public entry points on this surface: `plan` (468), `execute` (519), `execute_plan` (535),
`execute_plan_with_placements` (978), `plan_named` (1177), `execute_plan_named` (1192),
`execute_plan_named_with_placements` (1210), `execute_plan_op_timed` (1409),
`execute_plan_named_op_timed` (1484), `execute_plan_with_placements_op_timed` (1509),
`execute_plan_named_with_placements_op_timed` (1613). Card 5's gate is "entry-point count 10 ->
1", which will not match what a `grep` finds. Inherited from the task, but the design states it
opened the file.

### 27. Two of the six "abandoned designs" were not abandoned

§E.2's list is the task's explicit E requirement. Four entries are constraint-driven and hold:
`BoundOpKind::FlashAttention` (reuse-first; the defending paragraph correctly identified as the
finding), the `Fuse` pipe stage (second binary question, and `BoundOpBuilder`'s existing `Pipe`
impl at bind.rs:992-1003 confirms the call site is identical), process-wide `Arc<Mutex<..>>`
(lock discipline), and today's device-owning `Plan` (no_std tier; `metal.rs:488-499` confirms
the device IO).

Two do not. "A measured loss beats a clean argument" (lines 1142-1149) presents (i) multi-token
amortization and (ii) bytes-gated elision as "abandoned twice" -- but D.1 and D.3 are both still
in the design, both in Route 2, and D.3 supplies Route 2's entire margin. What was abandoned is
an *assumption* about each; the levers survive with an added precondition and a changed gate.
Calling a gate change an abandonment inflates the count the task asked for. The `Decide`-shaped
stop type (line 1133) was settled by prior owner rule before this design started, so it is a
restatement, not a design this work gave up.

### 28. Card 3's "text identical over 100 tokens" gate cannot hold across R5

R5 explicitly reorders floating-point arithmetic ("different arithmetic order", line 606), and
card 3's gate is "text identical over 100 tokens" (line 1104) plus the standing "100 tokens
byte-identical where the card is lossless" (line 1096). Reassociated softmax accumulation is not
bit-identical to the two-pass form; greedy decoding is a discrete argmax over logits that differ
in the last bits, so text identity is a probabilistic property over 100 tokens, not a gate. The
design supplies the right gate for the rule elsewhere (`assert_close(.., 1e-6)`, line 1236) and
then asserts a bit-exact gate for the card that lands it.

### 29. Smaller items

- §D.0's "4.168 against the given 4.169 reconciles to 0.03%, which is the check that this is the
  right decomposition" (line 67): the 4.169 comes from the task statement (task line 91), so the
  agreement checks the design's arithmetic against the task's arithmetic, not against a
  measurement. I recomputed every row independently and they are all correct
  (FFN 5.637G x 0.5625 = 3.1709 GB; attn 1.342G x 0.5625 = 0.755; output 131.08M x 0.8203 =
  0.1075; KV 32x2x512x8x128x4 = 0.13422), as are both routes and every ms division. The
  arithmetic is sound; only the inputs are contested.
- `Stage.band: Option<Band>` and the retained masking `Select` are two representations of the
  same restriction with nothing checking that the band is not narrower than the mask. The
  two-sided parity gate (fused-CPU vs unfused-CPU) covers it only because `FusionRules::none()`
  derives no band -- which the design does not state as the reason the gate is sufficient.
- `KernelSpec { entry: ArrayString<..> }` (line 768) puts a fixed-capacity string in the plan;
  `arrayvec` is a proxima-tensor dependency (Cargo.toml:168) so this is fine, but no capacity
  const is named among C.4's new consts and no overflow policy is given for an entry name that
  exceeds it.
- C.4 states `KV_EXTENT_BUCKET_TOKENS` "moves to `omega/src/sized.rs`" and then that "the right
  fix is deletion, and the move is what the card lands" (lines 1074-1077). Landing a move whose
  stated correct outcome is deletion is a step the task's scope-discipline axis counts against:
  `KV_BUCKET_TOKENS` is asserted = 32 at `sized.rs:390` and the assertion moves with it.
- §E.3 item 5 says "`ByteStreamParser` is not used and not extended" -- correct scoping, but the
  task's sans-IO requirement (P11) asks for the FSM's runnable walkthrough driving every legal
  transition; E.5 supplies a fusion test and no walkthrough of `Step`'s five transitions.
- §A.5's "19 -> 8 bound ops/layer, 616 -> 264/token" reproduces dispatch-census.md:49's ceiling
  as a target (line 630, and card 7's gate says "520 -> 264 target, MEASURED per card"), which
  is stated correctly as a target. But the census's own line 49 computes 8/layer as
  7 matvec + 1 attention, and finding 15 above adds 5-6 selector dispatches/layer under Route 2 --
  the two sections' dispatch counts are not reconciled with each other.
