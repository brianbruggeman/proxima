# The plan-level fixpoint rewrite engine — design spec

This is the spec for an engine that does not exist yet. It exists to replace
one ad hoc, one-shot detector (`epilogue_fuse_plan`, `cpu.rs:1059`) with a
worklist that runs a fixed set of local rewrite laws to a fixpoint over the
same plan representation that detector already reads. Every law below is
either **LANDED** (grep-verified, `file:line` cited) as a special case
somewhere in the tree today, or **PROPOSED** (no landed instance found —
verified by grep, not assumed absent). Read the landed cases as worked
examples: the engine you are about to build is what happens when you stop
hand-writing a new admission function per shape and instead run one
general law to a fixpoint.

## 0. What plan the laws operate over

Two IRs exist in this crate, and the six laws below operate on the **second**
one. Get this wrong and every law statement is wrong.

- `Op` (`proxima-tensor/src/op.rs:175`) — the source program. Five variants:
  `Input`, `Elementwise`, `Reduce(Reduce)`, `Iota`, `Constant`. The doc
  comment directly above the enum (`op.rs:166`) says "the four generators" —
  that comment is stale (`Constant` was added later; its own doc at
  `op.rs:236-249` explains why, as a fix for two more-expensive workarounds
  that predate it). **Correction, source wins: `Op` has five variants, not
  four, and I found no "14-core" or "14-primitive" claim anywhere in this
  tree** — grepped `op.rs`, `lower.rs`, `discipline.md`, `rooflines.md` for
  `14-core`, `14 primitives`, `RISC`, zero hits tying a count of 14 to this
  IR. `ScalarOp` (`op.rs:60`, the elementwise/reduce body vocabulary) has 17
  variants, also not 14. If "14-core IR" is a real claim it lives somewhere
  I did not find in this checkout — treat it as unverified and use the
  actual shapes below.
- `BoundOp` / `BoundOpKind` (`proxima-tensor/src/bind.rs:201,221`) — what
  `bind::bind` resolves an `Op` program into. Two variants only:
  `Elementwise { body: ComposedBody, operands }` and
  `Reduce { element_body, reduce_op, init, keep, operands, output_axes,
  out_layout, .. }`. `bind.rs` already does its own fusion pass here
  (`compose_operand`/`compose_body`, law 3 below) before this list ever
  reaches `cpu.rs`.

`cpu.rs`'s existing epilogue-fusion machinery — `epilogue_fuse_plan`,
`detect_epilogue_kind`, `is_post_reduce_epilogue*` — all take
`resolved: &[BoundOp]`, the list `bind::bind` already produced. **That is
the plan level.** The six-law engine is a worklist over this same
`&[BoundOp]` list, run to a fixpoint, in the same position
`epilogue_fuse_plan` occupies today. It is not a change to `Op`, not a
change to `bind.rs`, and it does not reopen `bind.rs`'s own fusion pass —
see §6 (`bind.rs` untouched) for why that boundary is load-bearing, not
incidental.

## 1. Law 1 — epilogue absorption

**E ∘ Reduce → Reduceᴱ.** A single-consumer, row-local elementwise chain
sitting immediately after a reduce folds into the reduce's own write —
the reduce still executes and materializes its buffer exactly as before;
only the *consumer's* interpreted walk is replaced by one more match arm
in the same monomorphized kernel.

**LANDED**, three shapes:

- `EpilogueKind::{Clip, ClipNorm, Norm}`, `cpu.rs:911-921`. `Clip` =
  `max(reduce + bias, 0)`; `ClipNorm`/`Norm` add the affine-normalize tail.
- Admission: `is_post_reduce_epilogue`, `cpu.rs:770`. Detection:
  `detect_epilogue_kind`, `cpu.rs:927`. Plan build: `epilogue_fuse_plan`,
  `cpu.rs:1059`. Apply: `apply_epilogue_fused_monomorphic`, `cpu.rs:1324`.
  All private (`fn`, not `pub fn`) — internal to `proxima-tensor`, not part
  of its public surface.
- Gate: `EPILOGUE_FUSE_ENABLED`, `cpu.rs:1231`, **default-on**
  (`AtomicBool::new(true)`); toggled off only for bench/test
  (`cpu.rs:1233-1241`).
- discipline.md ROW 184 (`discipline.md:16597`) lands the three-kind
  monomorphized kernel, −28.47% e2e on the mnist mock, bit-identical.
  ROW 191 (`discipline.md:17107`) is covered under law 2 below — it widens
  this same admission mechanism to a fourth shape that does **not** fit
  law 1's own admission test, which is exactly why it needed a second law.

**Equivalence obligation.** Policy: bit-identical — the reduce still
executes and materializes its own buffer exactly as before; only the
consumer's interpreted walk becomes one more match arm in the same
monomorphized kernel, so no arithmetic reordering is introduced. Test:
`law1_clip_epilogue_fused_matches_unfused_bit_identical`
(`proxima-tensor/tests/rewrite_law_equivalence.rs`).

## 2. Law 2 — row-statistic absorption

**E(x, ρ(x)) → RowEpilogue(E, ρ)**, where ρ is a last-axis reduction of the
*same row* `x` belongs to. This is law 1 generalized: law 1's three kinds
all walk the reduce operand element-for-element (the reduce output IS the
per-position value the epilogue consumes). Row-statistic absorption instead
admits an epilogue whose reduce operand is read through a **keepdims
broadcast** — one value per row, re-read for every column — while a
*different*, non-reduce operand is walked element-for-element. `ρ = mean,
var` is LayerNorm. `ρ = max, sum` is softmax. Both are "compute a
row-summary statistic, then normalize every element of the row against it"
— one law, two ρ instantiations.

**LANDED, one instantiation (LayerNorm).** `EpilogueKind::LayerNorm`,
`cpu.rs:918-921`, doc at `cpu.rs:895-909`. Admission:
`is_post_reduce_epilogue_broadcast_reduce`, `cpu.rs:809` — proven
structurally disjoint from law 1's `is_post_reduce_epilogue` at the
call site (`cpu.rs:1092-1098`, both admissions tried, "a candidate can
satisfy at most one", `cpu.rs:1084-1091`'s own comment). Geometry gates:
`epilogue_reduce_operand_matches_leading_axes`,
`epilogue_broadcast_operand_matches_last_axis`,
`epilogue_is_scalar_broadcast` (all private, `cpu.rs`, added same commit as
`cpu.rs:1000` `epilogue_hoist_axis`, which this kind bypasses because
`gamma`/`beta` vary over the last axis while the reduce broadcasts over the
second-to-last — two disjoint broadcast axes the single-axis hoist function
cannot express).

Landing history matters here: ROW 185 (`discipline.md:16682`) first
measured that BGE's real 1244-node graph produced **zero** fusion hits
under law 1 alone — `is_post_reduce_epilogue` structurally rejects every
LayerNorm tail because the reduce is read through a broadcast, not walked
1:1. ROW 191 (`discipline.md:17107`) is the session that generalized the
admission to cover it — the "first full instance" of this law, per the
task brief, and confirmed here: `hits=75` on BGE-small (25 LayerNorm sites
× 3 sentences, full coverage, `discipline.md:17121`), 5.25 ns/element
fused vs 32.90 ns/element interpreted (`discipline.md:17123`), bit-identical
on every output element.

**Fan-out legality — why this is sound, stated once, generally.** The
reduce's output row is complete at the moment the epilogue runs: a
`Keep::Reduce` reduce (`bind.rs:226-234`) walks its full pre-reduction
extent and writes one row of `out_layout` before any consumer touches it
(there is no partial-row consumer in this executor — `run_reduce` finishes
a row, then moves to the next). So an epilogue that re-walks that
just-written row to compute ρ (mean/var, or max/sum) is reading data that
is already fully resolved, not racing the reduce that produced it. This is
the argument every ρ instantiation of this law needs, and it needs stating
only once — softmax's ρ (max, then sum-of-exp) satisfies it identically to
LayerNorm's (mean, then sum-of-squared-centered), because both are "a
second reduction over a row this same kernel just finished producing."

**PROPOSED, not landed: softmax as a second ρ instantiation.** No
`EpilogueKind::Softmax` or equivalent exists in `cpu.rs` — grepped
`Softmax|softmax` in `cpu.rs`, zero hits outside comments/tests describing
BGE's graph shape. §5 below derives what a softmax instance of this law
does to the attention block.

**Since this section was written, `layer_norm_cluster_plan` (`cpu.rs:1566`)
landed and widens the LayerNorm instantiation above from a single-hop tail
fusion to the FULL five-dispatch `R1→E1→E2→R2→tail` cluster
(`discipline.md` ROW 204) — a second, two-pass reduction reassociation on
top of the same admission this section already describes.**

**Equivalence obligation.** Policy: documented rtol, NOT bit-identical —
measured, not assumed: `docs/discipline.md` ROW 204 found
`bit_identical(fused-cluster vs unfused)=false` on the real BGE model,
"expected and explicitly permitted... the two-pass reduction reassociates
relative to the graph's own reduce order." The cluster kernel's own 4-lane
accumulator sum (`LayerNormRowFsm`, `cpu.rs:1468`) does not walk the same
pairing order as the unfused `R1`/`R2` reduce kernels, so the two arms agree
to within a few ULPs per accumulation step, not bit-for-bit. Test:
`law2_layer_norm_cluster_fused_matches_unfused_within_rtol`
(`proxima-tensor/tests/rewrite_law_equivalence.rs`, `atol=3e-4`,
`rtol=1e-3` on the combined `atol + rtol*|unfused|` bound). Residual named
by the test itself: at `hidden=1`, `reciprocal_n == 1.0` exactly trips
`bind.rs`'s own `eliminate_identity_multiply` (law 3's scale-fold special
case) on the `mean` node, removing the two-step composed body
`layer_norm_cluster_plan`'s own structural admission requires — the cluster
legitimately never fires at that one shape (`rewrite-algebra.md` §8's own
"unmatched structure falls through unchanged" contract), a genuine
confluence gap between law 3 and law 2 the test excludes rather than papers
over.

## 3. Law 3 — prologue absorption

**Reduce ∘ E → Reduceᴱ.** A single-consumer elementwise producer sitting
immediately before a reduce folds into the reduce's own *operand read*: the
reduce's inner loop evaluates the producer's body per element instead of
reading a separately materialized buffer.

**LANDED, and general — not "inside dot" specifically.** The brief's
framing names `dot`; source says otherwise and source wins: there is no
`dot`-specific code path in `bind.rs` — grepped `\bdot\b`/`MatMul` in
`bind.rs`, zero hits. The mechanism lives in the **general** reduce-lowering
arm of `push` (`bind.rs:585`, `pub fn push(&self, expr: &Op, shapes:
&Shapes)`), which handles every `Op::Reduce` — a matmul/dot contraction
is one *instance* of `Op::Reduce` (a reduce whose body is `Multiply`), not
a distinguished op the algebra special-cases. The fusion condition:
`fuses = retires.contains(&reduce.operand) && is_identity_projection(&reduce
.in_map) && held.borrow().contains_key(&reduce.operand)` (`bind.rs:689-691`
— single consumer, unshifted-projection read, still held). On a hit:
`compose_fused_operands` (`bind.rs:1066`, doc calls itself "the
reduce-fusion entry point") → `compose_operand` (`bind.rs:1406`, private) →
recursively `compose_body` (`bind.rs:1380`, private) for however many more
producers are held beneath it. Absorbed nodes are tracked in
`ComposeState::absorbed` (`bind.rs:1041-1045`) and dropped
(`drop_absorbed`, called at `bind.rs:1081`) — this is the mechanism that
makes law 3 a strict node-count reduction (§7).

**Scale-fold is this law with a constant operand** — `E` is
`Multiply(x, Constant(c))` instead of a general elementwise chain;
`eliminate_identity_multiply` (referenced `bind.rs:1113,1427`) is the
special case where `c == 1.0` and the multiply itself disappears rather
than folding. No separate mechanism; same `compose_operand` path.

**Equivalence obligation.** Policy: bit-identical — a strict node-count
reduction (§7's own termination argument): the producer's body runs per
element inside the reduce's own inner loop instead of being read from a
separately materialized buffer, same per-element arithmetic and
accumulation order, one fewer buffer round trip. Test:
`law3_prologue_absorption_fuses_and_matches_hand_reference`
(`proxima-tensor/tests/rewrite_law_equivalence.rs`), which also asserts
`resolved.len() == 1` as the engagement proof (a candidate this admission
rejected would bind as separate `BoundOp`s).

## 4. Law 4 — same-input widening

**k reduces sharing operand A, with independent constants B₁..Bₖ → one
reduce over A against B₁⊕..⊕Bₖ.** QKV projection is the canonical case:
`Q = x·Wq`, `K = x·Wk`, `V = x·Wv` share the input activation `x` and
differ only in the weight operand — one reduce over `x` against
`[Wq|Wk|Wv]` concatenated on the output axis produces the same three
results as one wider GEMM, 3 kernel launches → 1.

**PROPOSED. No landed instance.** Grepped `qkv`/`QKV` case-insensitively
across `proxima-tensor/src/` — every hit (`spec.rs:1945`,
`shape.rs:181,1272,1431,1455,1554`, `instrument.rs:1300`) is about a
**physically fused QKV weight on disk** (a checkpoint that already stores
`[Wq|Wk|Wv]` concatenated, and this crate's own chunk-axis machinery
splitting it back into three logical reads at *shape-inference* time) —
the opposite direction from this law, and not a plan-level rewrite at all.
No mechanism widens k independent reduces into one. This law has never
fired.

**Equivalence obligation.** PROPOSED — no landed instance, so no
fused/unfused pair exists to compare. Test:
`law4_same_input_widening_is_proposed_not_landed`
(`proxima-tensor/tests/rewrite_law_equivalence.rs`), `#[ignore]`, named and
documented as PROPOSED so the obligation is visible without fabricating
coverage for a mechanism that does not exist.

## 5. Law 5 — layout commutation

**Pure `IndexMap`s compose into producer writes / consumer reads;
Affine∘Affine is closed.** `IndexMap` (`proxima-tensor/src/map.rs:134`) has
exactly two variants: `Affine(IndexPattern)` and `Computed{..}`
(data-dependent, a gather/scatter). Two `Affine` maps in sequence — a
producer's output addressing composed with a consumer's read addressing —
describe one linear function of the iteration indices, which is itself
expressible as one `Affine` map. Where this closure holds, a pure-movement
node (transpose, reshape-that-doesn't-merge-real-axes, broadcast) never
needs to materialize: the consumer is built to read the *original* producer
directly, through the composed pattern.

**LANDED, at the ONNX-frontend lowering boundary** (`proxima-onnx/src/
lower.rs`), not yet generalized into a plan-level worklist law in
`proxima-tensor` itself — this is the gap the engine closes.
`Value` (`lower.rs:118`, private struct) carries two fields that are exactly
this composition, done by hand per producer:
- `view: Option<Vec<Option<u16>>>` (`lower.rs:121`) — set only by
  `lower_unsqueeze`. Doc (`lower.rs:104-116`): binding `Unsqueeze`'s output
  to a fresh `Op::Elementwise` would leave the inserted size-1 axis
  unpinned (`shape::infer` requires every iteration axis covered by an
  operand); instead `view` aliases straight through to the pre-`Unsqueeze`
  node, and the *consumer's* `IndexPattern` is built directly against that
  real node.
- `flatten_source: Option<(Vec<u64>, usize)>` (`lower.rs:137`) — set by
  `lower_flatten` when a reshape merges real, non-1 axes (the
  `[N,C,H,W] → [N,C·H·W]` LeNet-into-Gemm case). `lower_gemm` reads it and
  widens the matmul's *own* iteration space by one axis per real trailing
  axis, addressing the producer at its real rank, rather than materializing
  a physically-reshaped node.

These are fields on a private struct, not free functions named
`Value::view`/`flatten_source` (the brief's phrasing is imprecise; the
mechanism is real, the shape of the citation was wrong). Correcting: they
are per-value bookkeeping the ONNX lowering pass carries by hand, one
producer type at a time (`Unsqueeze`, `Flatten`). **This is exactly what a
general law-5 worklist rule would subsume**: instead of two hand-written
special cases at the frontend, one composition rule (`Affine ∘ Affine →
Affine`) applied wherever a plan-level movement node's IndexMap is a pure
projection of its producer's.

**Also landed, plan-level, one instance of the same idea**:
`resolve_reduce_axis_shape` (`cpu.rs:5871`, private) — unit-axis elision.
When every raw leading output axis has extent 1 (BGE's own `M=1`,
single-token batch), the address-computation axis list would otherwise
empty out entirely and both tile-plan builders' `len() == 1` gate would
reject it exactly as it rejects `len() == 2` (doc at `cpu.rs:5850-5870`,
citing `discipline.md` ROW 200's own M=1 regression). One raw axis is put
back, sound at any position because a size-1 axis's coordinate is always
`0` regardless of which index names it. This is layout commutation at the
addressing level, not the node level — it keeps a degenerate axis
addressable without materializing anything extra for it.

**Equivalence obligation.** Policy: bit-identical — a size-1 axis's
coordinate is always `0` regardless of which index names it, so eliding it
before the tile gates changes only which index list the address computation
walks, never the fold order or the operand values read. Test:
`law5_leading_unit_axis_elision_matches_unbatched_bit_identical`
(`proxima-tensor/tests/rewrite_law_equivalence.rs`), generalizing
`leading_unit_axis_tile_engagement.rs`'s fixed-shape seed property-style
over random `M`/`K`/`N` including remainder (non-tile-multiple) extents.

## 6. Law 6 — constant staging

**An all-`Constant` subgraph evaluates once, at plan time, and collapses to
one `Constant` node.** Two landed forms, at two different points in the
pipeline — worth keeping distinct, because they solve different problems:

**Lower-time folding (LANDED, strict node-count reduction, `proxima-onnx`).**
`FoldState` (`lower.rs:162-179`, private struct) carries
`constant_values: BTreeMap<String, f32>` (a uniform constant's scalar) and
`constant_arrays: BTreeMap<String, Vec<f32>>` (`lower.rs:173`, per-element
folds through `Constant`/`Shape`/`Gather`/`Unsqueeze`/`Concat` chains) —
these are **fields** the fold state threads through `lower_node` and its
callees, not free functions. Every constant-only subchain the ONNX export
carries (shape computation, index arithmetic) resolves during lowering and
never becomes an `Op` node at all. This is the node-count-reducing form:
N `Op` nodes collapse to 0 (the value is baked directly into a consumer's
static shape or a literal `Op::Constant`), not merely re-scheduled.

**Plan-time hoist, execute-once (LANDED, `proxima-tensor`, a *different*
optimization — amortization across repeated steps, not node-count
reduction).** discipline.md ROW 175 (`discipline.md:15683`):
`StaticArena` (`cpu.rs:480`, `pub struct`) gained `static_nodes:
BTreeSet<NodeId>` (`cpu.rs:521`, private field) — every live resolved node
whose `BoundOpKind` is `Constant`/`Iota`, computed by `static_resolved_nodes`
(`cpu.rs:563`, private) and run exactly once inside `build_static_arena`
(`cpu.rs:587`, `pub fn`), instead of every training step
(`run_resolved_nodes_in_arena`, `cpu.rs:712`, gained one skip clause). This
does not remove the node from the graph — a `Constant`/`Iota` node still
exists and still occupies a buffer slot — it removes it from the **per-step
replay set**. −9.3% on the sealed Adam step. The distinction matters for
§7's termination argument: this form is not itself a node-count-reducing
rewrite; it is what a plan-level law 6 rewrite (fold N constant nodes into
1) should feed once its result is bound into a repeated-execution context
like a training loop.

**PROPOSED, needs owner ratification: weight packing = law 6 ∘ law 5.**
Not landed anywhere in this tree — grepped `pack.*weight|panel|column.major`
in `proxima-tensor/src/`, no hits describing a packing pass. The argument
for it: ROW 203 (`discipline.md:17862`) measured why BGE's 96 in-graph GEMM
reduces run slower than the same shapes isolated-and-warm — cache regime
(H1) explains **~70.9% average** (66.3%–77.1% across the three real
sentence lengths, `discipline.md:17887-17895`) of the gap, and the
mechanism is **first-touch latency, not sustained bandwidth**: the cold
arm's own effective bandwidth (6.6–10.3 GB/s) sits at only 9.4%–14.7% of
this machine's measured 69.95–81.21 GB/s streaming ceiling
(`discipline.md:17885`, `rooflines.md:213`) — each cold call touches one
weight buffer exactly once before moving on, dominated by page-fault/
row-activation cost, not amortized streaming. A weight tensor is
all-`Constant` at plan time (it never changes across sentences at
batch=1); relaying it once, at plan-build time, into a layout the
executor's own tile kernels read with better locality (law 5's layout
composition, applied to a `Constant` producer instead of a movement node)
is literally law 6 applied to a `Constant` operand, with law 5 choosing the
target layout. **Prior art, named generically per the task's own
constraint**: at least one incumbent ML runtime (onnxruntime/MLAS) packs
constant 2-D weight operands once, at session-init time, into fixed-width
contiguous column panels sized to its own microkernel's register-blocking
factor — the general shape being described is "restride a constant once,
to the shape the hot kernel wants to stream," which is exactly what ROW
203's own first-touch-latency finding motivates here. This is **not**
independently bench-verified in this repo as a lever (ROW 203 explicitly
did not attempt it — "priced instead," `discipline.md:17937`) —
**PROPOSED — needs owner ratification**, both on the law-composition
argument and on whether the packing target layout should be chosen
per-kernel (law 5's own job) or fixed globally.

**Equivalence obligation (plan-time hoist, execute-once instance).**
Policy: bit-identical — `run_resolved_nodes_in_arena` skipping a
`static_nodes` member on every call after the first changes only how many
times a value is recomputed, never the bytes a downstream consumer reads.
Test: `law6_constant_staging_hoisted_matches_per_step_evaluation_bit_identical`
(`proxima-tensor/tests/rewrite_law_equivalence.rs`) proves the hoisted arena
path matches a fresh per-step `evaluate_named` call across multiple steps
with different non-constant inputs, for a multi-node all-`Constant`
subgraph. The "executed exactly once" engagement proof itself needs
`StaticArena`'s private fields, which this integration test's own external
surface cannot reach — that proof is `build_static_arena_runs_a_live_constant_once_and_never_again`
(`proxima-tensor/src/cpu.rs`, `#[cfg(test)] mod tests`, the ROW 174/175
corrupted-buffer test already in-tree). Weight packing (the PROPOSED
law 6∘law 5 instance above) has no equivalence test — nothing landed to
test.

## Membership, decided (PROPOSED — needs owner ratification)

Six laws proposed as the closure basis. Justification and exclusion, stated
plainly per member:

| law | in / out | one-line why |
|---|---|---|
| 1 epilogue absorption | IN | landed 3×, general admission-test shape, strictly reduces node count |
| 2 row-statistic absorption | IN | landed 1×, proven structurally disjoint from law 1 at the same call site, same strict-reduction argument |
| 3 prologue absorption | IN | landed and general (not `dot`-specific), the mechanism law 1/2 both compose with going the other direction |
| 4 same-input widening | IN, PROPOSED | no landed instance, but the admission test (shared operand, independent second operand) is a clean structural mirror of law 1's "shared reduce, independent epilogue" — same shape, opposite side of the reduce |
| 5 layout commutation | IN | landed twice at different layers (ONNX-frontend `Value` fields, plan-level `resolve_reduce_axis_shape`), never generalized into one rule — the gap this engine closes |
| 6 constant staging | IN | landed twice (lower-time fold, plan-time execute-once), and law 6∘law 5 (packing) is the highest-measured-mass unlanded lever (ROW 203, ~71%) |

**Excluded, and why**: any law that changes numeric semantics (algebraic
reassociation of a non-associative body, e.g. `Subtract`/`Divide`
reordering) is out — `ScalarOp::is_associative` (`op.rs:112-117`) already
draws this line for the reduce scheduler, and the six laws above are all
bit-identical rewrites (every landed instance cited above reports
bit-identical output against its unfused baseline). Any law requiring a
**data-dependent** decision (choosing which branch to fuse based on runtime
values) is out — `IndexMap::Computed` (`map.rs:136`) is the one place this
algebra admits data-dependence, and every law above operates only on the
`Affine` side, by construction of what "plan time" means.

## 7. Proof obligations

### Termination

Each law strictly reduces the count of live nodes in the resolved
`&[BoundOp]` list on every application:

- **Law 1/2**: merges a `Reduce` node and its consumer `Elementwise` node
  into one `Reduce` node carrying an `EpilogueKind`. The consumer node is
  removed. `−1` node per fire.
- **Law 3**: `compose_operand`'s `ComposeState::absorbed` (`bind.rs:1044`)
  names every producer node folded into the reduce's operand read;
  `drop_absorbed` removes them. `−1` per absorbed producer, and the
  recursion terminates because `held` (`bind.rs:1408`) only contains nodes
  already known single-consumer — each recursive step consumes one entry
  from a finite map.
- **Law 4**: `k` reduce nodes become 1. `−(k−1)` per fire, `k ≥ 2`.
- **Law 5**: a movement node (pure `Affine` `IndexMap`) is eliminated by
  redirecting its consumer to read the original producer through the
  composed map. `−1` per fire.
- **Law 6**: an all-`Constant` subgraph of `N` nodes becomes 1 `Constant`
  node. `−(N−1)` per fire, `N ≥ 2` (or `−N` if the whole subgraph folds
  into an existing literal with no residual node at all, as `lower.rs`'s
  `constant_values` fast path already does).

Node count is a non-negative integer. Every law strictly decreases it.
A worklist that only fires laws when their admission test passes therefore
terminates in at most (initial node count − final node count) total
applications — standard well-founded-measure termination for a rewriting
system. **The depth counter the brief specifies is a backstop, not the
termination argument**: its job is to catch an incorrectly-implemented law
that mis-reports progress (fires without actually shrinking the node list,
which would violate its own contract above) — a real bug, not a real
non-termination risk in a correctly implemented law. Set it to the current
node count as a hard ceiling and any law hitting it has a bug, not a
legitimately deep fixpoint.

### Confluence

**Locally commuting, by construction, for the pairs sharing a node from
opposite sides**: law 1/2 (epilogue, consumer side) and law 3 (prologue,
operand side) touch the same `Reduce` node from disjoint directions — one
rewrites what happens after the reduce writes its row, the other rewrites
what the reduce reads to produce it. Firing either first does not change
what admission test the other passes, because each only inspects the side
it doesn't touch. Law 1 and law 2 are mutually exclusive by construction —
`is_post_reduce_epilogue` and `is_post_reduce_epilogue_broadcast_reduce`
require opposite conditions on the same operand (non-broadcast walked vs.
broadcast-read), proven disjoint at the call site (`cpu.rs:1092-1098`) —
so there is no ordering question between them at all.

**Open question, stated honestly, not resolved here: law 4 vs. law 1 on
QKV bias.** If `Q = x·Wq + bq`, `K = x·Wk + bk`, `V = x·Wv + bv` each carry
their own per-output bias, and law 1 fires first (each of the three
reduces independently absorbs its own bias epilogue before law 4 ever
looks at them), does law 4's admission test — "k reduces sharing operand
A, independent second operand" — still recognize the three now-epilogue-
bearing `Reduce` nodes as the same shape? A concatenated bias vector
`[bq|bk|bv]` is representable as one wider epilogue on one wider reduce
(each output column reads its own slice), so the *semantics* survive
either order. But if law 4's structural match is written against a bare
`BoundOpKind::Reduce` shape (operand identity, no epilogue attached) and
law 1 has already rewritten each candidate into a
`BoundOpKind::Reduce`-with-`EpilogueKind`, the match may no longer fire —
not a wrong answer, a **permanently missed one**, since nothing re-splits
a fused epilogue back apart to try again. Two candidate resolutions, either
is defensible, neither is verified against an implementation that does not
exist yet: (a) give law 4 priority over law 1 in the worklist ordering, so
widening happens on bare reduces before any epilogue absorption narrows
what law 4 can recognize; or (b) make law 4's admission test blind to an
already-attached epilogue (match on the reduce's own operand/init/body
shape only, independent of what `EpilogueKind` it carries, and widen the
epilogues alongside the reduce). **Left open for the implementer — this is
exactly the kind of local race the fixpoint's worklist ordering has to
settle, and it cannot be settled by reading source that does not exist.**

## 8. Engine shape

Worklist to fixpoint, explicit state, depth counter, **no call recursion**
(owner directive, stated in the task, not independently re-derived here —
recorded as a binding constraint). Concretely: a queue of candidate node
indices seeded from the full resolved list; popping a candidate, trying
each law's admission test in a fixed priority order (§7's open question is
exactly where that order is decided), and on a hit, pushing every node
newly adjacent to the rewritten node back onto the queue (a node whose
consumer just changed shape may now admit a law it didn't before — this is
what "depth" means operationally, not a recursive call depth).

**Depth as an observable, not a hidden implementation detail.** Depth-1 is
every law fired directly against the raw, bound graph — the shapes
`epilogue_fuse_plan` already detects today in one pass. Depth-N is every
law enabled only because a prior substitution changed what a node's
neighbor looks like. §9 below is one worked depth-by-depth derivation.

**Fused kernels are sans-IO FSMs**, not nested function calls. A kernel
produced by chaining several laws' worth of fusion (e.g. §9's flash-
attention shape) is `state + step`, driven run-to-completion today (one
`step` call per output element/row, matching every landed kernel's own
shape — `apply_epilogue_fused_monomorphic`, `cpu.rs:1324`, is exactly this:
one `match` per fusion site, then a straight-line per-element loop, no
recursion). Band-streaming (a producer kernel yielding a completed row to
a consumer kernel before its own next row starts, rather than materializing
the whole intermediate) is named as a **later** capability, not built here
— it is what §9's final fusion step (`@V` consuming softmax'd rows) needs
beyond what `compose_operand`'s pure-elementwise inlining already does
today, because `compose_operand` only inlines scalar chains, not a second
reduction (softmax's own ρ) nested inside a first one's operand read.

**Plan-level only. `bind.rs` is untouched.** discipline.md ROW 166
(`discipline.md:14974`) is the landmine that makes this a hard boundary,
not a style preference: a *correct*, bit-identical change to `bind.rs`'s
own elementwise-fusion machinery (removing one dead node from the Adam
backward chain) regressed the sealed step **14.5×** (2.82ms → 41.4ms),
because removing that dead sibling made its live sibling newly eligible for
`compose_operand`'s recursive fusion into a downstream reduce, which then
fell off whatever fast dispatch path it held onto a generic per-element
loop. Rolled back in full, zero source diff landed. The lesson for this
engine: `bind.rs`'s own fusion is already load-bearing and already
performance-fragile in ways a session with a 60-minute budget cannot fully
characterize. The six laws operate one layer up, on the `&[BoundOp]` list
`bind.rs` already finished producing — they never re-open what `bind.rs`
decided to fuse. **Unmatched structure falls through unchanged** — a node
none of the six laws' admission tests accept is left exactly as `bind.rs`
resolved it; the engine has no "else materialize something different"
branch, matching every landed law's own behavior today (a non-admitted
candidate in `epilogue_fuse_plan` is simply absent from the returned plan
map, `cpu.rs:1064`, and executes as an ordinary unfused node).

## 9. Worked derivation: depth-by-depth, QK — softmax — @V

This is the "flash-attention form DERIVED" the task brief asks for, walked
exactly, against BGE's own real shapes (`discipline.md:17862`'s own MAC-mix
identity: 12 layers, 384 hidden, attention block per layer contracted as
QK^T + softmax + @V, 24 attention-matmul nodes for the pair across 12
layers). None of this is landed — law 2's softmax instantiation and law 3's
second application here are both **PROPOSED**, walked as a derivation the
engine should reproduce, not as a claim about code that exists.

**Depth 1 — raw bound graph.** `QK = Reduce(Multiply, operand=(Q, K))`,
scaled by `1/√d` as a separate `Elementwise` node, feeding `softmax`
(itself `max` → `subtract` → `exponential` → `sum` → `divide`, several
`Elementwise`/`Reduce` nodes), feeding `@V = Reduce(Multiply, operand=
(softmax_out, V))`. This is the unfused shape `bind.rs` hands `cpu.rs`
today for any attention block law 1/2/3 have not yet touched.

**Depth 2 — law 3 fires once.** The `1/√d` scale is a single-consumer
elementwise producer feeding `QK`'s own reduce. `compose_operand`
(`bind.rs:1406`) absorbs it into `QK`'s operand read (fold `1/√d` into `Q`
or `K` before the multiply-reduce, mathematically equivalent, one fewer
node): `QK_scaled = Reduce(Multiply, operand=(Q·(1/√d), K))`. This is
"scale-fold," §3's own claim, applied here concretely.

**Depth 3 — law 2 fires once, PROPOSED instantiation.** `softmax`'s ρ
(row max, then sum-of-exp) is a last-axis reduction of `QK_scaled`'s own
just-written row — the same shape law 2's landed LayerNorm instance
already proves sound (§2's fan-out legality argument: the row is complete
at epilogue time). A `RowEpilogue(QK_scaled, softmax)` admission — new
`EpilogueKind` variant, same shape as `LayerNorm`'s but with `ρ = max, sum`
instead of `mean, var` — folds the entire softmax computation into
`QK_scaled`'s own write. Result: one kernel that computes scaled `QK^T` and
emits already-normalized attention weights, per row, without ever
materializing an unnormalized `[S,S]` score matrix as a separate node.

**Depth 4 — law 3 fires again, and this is where the engine needs more
than what's landed today.** The `@V` reduce's operand is now the
row-epilogue kernel's output. If that output were an ordinary elementwise
chain, `compose_operand` would inline it exactly as depth 2 did. It is
not — it is the output of a **second reduction** (softmax's own ρ) nested
inside what `@V` needs to read one element at a time. This is precisely
flash-attention's own trick: never materialize the full normalized-weights
row; stream it, row by row, directly into the `@V` accumulation using a
running max/sum rather than a two-pass materialize-then-read. §8's
band-streaming note is what this depth actually requires: a producer
kernel (`QK_scaled` + softmax epilogue) yielding completed rows to a
consumer kernel (`@V`) as an FSM state transition, not a second
`compose_operand`-style scalar inline. **This is the concrete case that
motivates naming band-streaming as a later capability rather than
building it now** — depth 4 is real, derivable, and not implementable with
today's `compose_operand` alone.

Net: 2 reduce nodes (`QK`, `@V`) + 1 scale (`Elementwise`) + ~4 softmax
nodes (`max` reduce, `subtract`, `exp`, `sum` reduce, `divide`) = ~8 nodes
in the unfused graph, collapsing to **1 fused row-streaming attention
kernel** at full depth — the count §10 uses for the per-layer target.

## 10. Target: the optimal BGE graph

**The roofline IS this graph** — stated plainly, per the task brief. The
machine roofline for BGE-small's real per-sentence compute is **3.515
ms/sentence** (`rooflines.md:553`, mean across the three real sentence
lengths measured; range 3.074–3.955ms, `rooflines.md:540-544`), derived
from **170,465,280 MACs/sentence** (`rooflines.md:633`, ≈170.5 MMAC, the
average over the 7/8/9-token real sentences) against this machine's
measured **48.5 GMAC/s** NEON register-blocked FMA ceiling
(`rooflines.md:59`, `discipline.md:1903-1910`). This is a compute-bound
workload — arithmetic intensity 1.283 MACs/byte sits above the 0.642
MACs/byte ridge point (`rooflines.md:532-536`) — so bandwidth is never the
binding constraint at this shape; the six laws exist to close the gap
between BGE's current 26.68ms/sentence (ROW 191, `discipline.md:17123`)
and this 3.515ms ceiling — currently **7.59×** (`rooflines.md:563`).

**Kernel count derivation (PROPOSED — needs owner ratification; this is
the task brief's own "5 kernels/layer, ~63 dispatches" claim, derived here
from the six laws, not found pre-existing anywhere in this tree — grepped
`5 kernel|kernels.per.layer|63 dispatch` across `docs/`, zero hits).**
ROW 203's own MAC-mix identity (`discipline.md:17887`) confirms BGE's
current per-sentence GEMM population: 48 QKVO + 24 FFN + 24
attention-matmul = 96 GEMM-shaped nodes across 12 layers — 8 GEMM nodes
per layer (4 QKVO + 2 FFN + 2 attention). Applying the six laws:

| kernel | laws composing it | what collapses into it |
|---|---|---|
| 1. QKV fused projection | law 4 (widen, PROPOSED) | Q, K, V reduces (3 → 1); §7's open QKV-bias-ordering question decides whether law 1's bias epilogue attaches before or after this fires |
| 2. fused attention (QK^T·scale·softmax·@V) | law 3, law 2 (softmax instance, PROPOSED), law 3 again (band-streaming, not yet buildable, §9) | QK^T reduce, scale multiply, softmax (max/sub/exp/sum/div), @V reduce — §9's full depth-4 derivation |
| 3. O projection + residual + LayerNorm | law 1 (bias-as-epilogue, same shape as landed `Clip`/`Norm`), law 2 (LayerNorm, landed) | O reduce, residual add (foldable as an extra epilogue operand, same shape as an existing bias), LayerNorm tail |
| 4. FFN-up + GELU | law 1 (activation-as-epilogue; GELU lowers through `ScalarOp::Erf`, `lower.rs:343`, same single-consumer row-local shape law 1 already generalizes) | FFN-up reduce, GELU |
| 5. FFN-down + residual + LayerNorm | law 1, law 2 (landed) | FFN-down reduce, residual add, LayerNorm tail |

5 kernels/layer × 12 layers = 60 dispatches for the transformer stack.
BGE-small also runs an embedding gather/sum and a final CLS-pool outside
the 12-layer loop — a handful of additional kernels (not derived here in
detail; out of scope for the per-layer table) bring the total toward the
brief's own **~63 dispatches** figure, which this derivation is consistent
with (60 + ~3) but does not independently re-derive past the per-layer
count. **This entire table is PROPOSED, not measured**: laws 2's softmax
instance, law 3's second (streaming) application, and law 4 are none of
them landed. The 63 number is the brief's own figure, checked here for
consistency against ROW 203's real node-population data and the depth-4
derivation, not independently rediscovered from a different source.

## Admission rule

Per the task brief, stated as the binding rule for whoever extends this
engine: **a law instance is admitted by measured mass against the machine
roofline (48.5 GMAC/s / 69.95–81.21 GB/s, `rooflines.md:59,213`), never by
what an incumbent runtime happens to fire.** ROW 203's ~71% cache-regime
finding is the load-bearing evidence for prioritizing law 6∘law 5 (weight
packing) over, say, further micro-tuning inside a single already-fused
kernel — not because a comparable incumbent packs its weights (that is
background context, cited generically in §6, not a justification), but
because the measured gap between BGE's warm-isolated GEMM rate and its
in-graph rate is dominated by a mechanism (first-touch latency on a cold
weight buffer) that packing directly addresses, and the gap's *size*
(≈71% of the total warm-to-in-graph delta, `discipline.md:17895`) was
measured, not assumed.

## Residuals for the implementer, named honestly

1. **§7's QKV-bias-ordering question is unresolved.** Pick a resolution
   before implementing law 4's admission test, and record the choice —
   this spec deliberately does not decide it, since neither option is
   verified against code that does not exist yet.
2. **Law 2's softmax instantiation and law 3's second (streaming)
   application are both PROPOSED, not landed.** §9's derivation is a
   design walkthrough, not a description of a shipped kernel. Building it
   requires the band-streaming FSM capability §8 names as "later" — that
   capability does not exist today and is out of scope for a first landing
   of laws 1/2/3/5/6 alone.
3. **Weight packing (law 6∘law 5, §6) has no bench of its own.** ROW 203
   priced it as roughly one more 75-minute session
   (`discipline.md:17937`) and explicitly did not attempt it. Do not treat
   the ~71% figure as a packing-specific win estimate — it is the size of
   the gap packing is *aimed at*, not a measured packing result.
4. **The §10 kernel-count table is a derivation, not a discovery.** No
   file in this tree states "5 kernels/layer" or "63 dispatches" before
   this document. Ratify or correct it before treating it as a target the
   engine is scored against.
5. **`H2`, ROW 203's own un-isolated residual** (`discipline.md:17897-
   17902`, ~23–34% of the warm-to-in-graph gap, magnitude measured,
   mechanism not independently probed — candidates named: per-node arena
   bind/release across 96 interleaved GEMM nodes, and/or `Op`
   dispatch/span overhead) is not addressed by any of the six laws above.
   It may shrink as a side effect of fewer, larger fused kernels (60 vs
   96+ nodes means fewer arena take/release cycles per sentence), but that
   is a prediction, not a measurement — the next session that lands laws
   1–3/5/6 should re-run ROW 203's own H1/H3 methodology against the
   post-rewrite graph rather than assuming the residual closes.
