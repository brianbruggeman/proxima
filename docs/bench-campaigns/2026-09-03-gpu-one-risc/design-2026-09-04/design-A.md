# design-A: the Metal decode path as pipes, sans-IO FSMs, RISC algebra, generic

Read-only pass over `/Users/brianbruggeman/repos/slot-0/proxima` at the checked-out `HEAD`.
Every `file:line` below was opened in this session. No build, no test, no measurement was run;
every number quoted is carried from the task brief or `scratchpad/dispatch-census.md` and is
labelled with that provenance.

---

## 0. The shape, and the one contested decision

**Shape.** One new IR concept — an *iteration domain* (half-space constraints with
static-or-symbolic offsets) — and one new bound-IR constructor — `BoundOpKind::Fused(FusedRegion)`,
a sub-nest of already-bound ops sharing one iteration space. `BoundOpKind::CachedAttention`
(`proxima-tensor/src/bind.rs:240-251`, 10 fields) is deleted. Attention stops being a kind and
becomes an *instance* of the region rule, alongside all four generic fusion levers the census
names (`scratchpad/dispatch-census.md:32-49`). The driver becomes `Encode.and_then(Submit)
.and_then(Read).and_then(Sample)` over `proxima_primitives::pipe::AndThen`
(`proxima-primitives/src/pipe/primitives.rs:203-221`), and the nine executor entry points
(`omega/src/metal.rs:519, 535, 978, 1177, 1192, 1210, 1409, 1484, 1509, 1613`) collapse to that
one chain plus config fields. The decode step is an enum FSM whose transitions consume the old
variant and take the device's product as an *argument*, so it carries zero Metal dependency.

**The contested decision: tuple-valued reduce vs. a scheduling rewrite.**
Online (flash) softmax needs a `(max, sum, weighted[head_dim])` accumulator. The obvious algebra
move is to widen `Reduce` to a tuple monoid — `dtype`, `init`, `body` all become vectors, and
`ScalarOp` grows a rescale. I rejected it. `proxima-tensor/src/op.rs:136-147` already settled this
class of question for scans: *"Log-depth prefix sum is a scheduling decision about a `Keep::Scan`
reduce, not a different operation, which is why there is no separate `Scan` expression form."*
The same argument binds here: `Maximum` and `Add` are both associative
(`op.rs:112-118`, `is_associative`), so the single-pass rescaled recurrence is a legal
reassociation of the three-reduce softmax, not a different program. It therefore lives as
`RegionSchedule::Rescaled` — a property of the *bound* region, chosen by the backend that can
render it — and `Op`/`ScalarOp` do not move. Cost of being wrong: if a backend cannot render
`Rescaled`, it renders `Nest` and pays two passes over the key axis; correctness is identical
either way, which is why this is the cheap side of the fork.

**What is NOT a pipe in this design** is enumerated in §F, with the justification for each.

### 0.2 The byte target governs the order — D leads

Owner target (2026-09-04): 5x llama.cpp = **3.5 ms/token** on this box. The M1 Max spec sheet is
400 GB/s, so streaming 4.169 GB/token has a hard floor of **10.4 ms** (4.169/400) — 2.98x the
target. No kernel, fusion, dispatch or barrier change can cross that floor, because the floor is
not compute. **The only levers that reach 3.5 ms are bytes-per-token levers, and they must land
the product at <= 1.4 GB/token (<= 1.0 GB at a realistic 300 GB/s sustained ceiling).** §D is
therefore the primary section; §A/§B/§C are what make §D's levers expressible as programs rather
than as special cases, and two of them are prerequisites: §A's `Domain` is what lets a k-token
verification pass carry a correct causal band, and §B's pure `plan()` plus zero-alloc step is what
keeps a k-token pass from paying k times the host overhead.

The dispatch-count work in §A.6 (616 -> 264/token) is real and is not a byte lever. Recorded as
such, not as progress toward 3.5 ms.


---
## A. Attention in the RISC algebra

### A.1 What attention is, in `Op`/`ScalarOp`/`IndexMap` terms

A softmax-weighted banded reduction over a key axis is already writable with the shipped
generators. `proxima-tensor/specs/causal_attention.toml` and `specs/gqa_attention.toml` prove it,
and `specs/mistral_layer.toml` composes both with RoPE (that file's own header, lines 1-5, states
the claim: *"this file is the proof that composing them costs no new `Op` or `ScalarOp`"*).
The chain, over an iteration space `[s, u, g, t, p, d]` (query row, kv head, group, key row,
rope pair, head dim):

| # | `Op` | body / reduce | reduced axis | note |
|---|---|---|---|---|
| 1 | `Reduce` | `Add` over `Multiply(q_even, k_even)` | `p` | `element_body` is the fused product (`bind.rs:260-262`) |
| 2 | `Reduce` | `Add` over `Multiply(q_odd, k_odd)` | `p` | the RoPE-split second contraction |
| 3 | `Elementwise` | `Add`, then `Multiply` by an `Op::Constant` scale | — | `Constant` is a leaf (`op.rs:261-265`) |
| 4 | `Reduce` | `Maximum`, `init = NegativeInfinity` | `t` | `ReduceInit::NegativeInfinity`, `op.rs:124-132` |
| 5 | `Elementwise` | `Subtract` then `Exponential` | — | broadcast of #4 back over `t` |
| 6 | `Reduce` | `Add`, `init = Zero` | `t` | the softmax denominator |
| 7 | `Reduce` | `Add` over `Multiply(p_weight, v)` | `t` | the value contraction, output axis `d` |
| 8 | `Elementwise` | `Divide` | — | normalize by #6 |

Nothing above is new. GQA's `h = group*u + g` decomposition is an `IndexPattern` with two
`AxisTerm`s (`map.rs:60-66`), which is the same mechanism `map.rs:14` documents for convolution.
The causal mask today is `Select(Greater(Iota, Iota), -inf, score)` — `Op::Iota`'s doc
(`op.rs:209-222`) names exactly this composition.

### A.2 The one required extension: the iteration space is a polyhedron, not a box

The mask-tensor spelling is *expressible* but not *lowerable*. Two defects follow from it and only
from it:

- **It does not constrain iteration.** The mask is an `Elementwise` node over the full `s x t`
  space, so the reduce's iteration space stays rectangular. That is the mechanism behind
  `bind.rs:2504-2506` setting `cached_key_rows = new_key_rows = key_shape[0]` and the kernel loop
  at `msl.rs:2586` running `cached_key_rows + new_key_rows` iterations while `continue`-ing half
  of them: `2t` loop trips for `t` of work.
- **The true band is a runtime value, not a shape.** `bind.rs:2434-2448` states it in the source:
  `kv-capacity-bucket` widens the key extent past the merged length, so `key_shape[0] -
  query_shape[0]` overstates `cached_len`. That is *why* a ninth rank-0 operand exists
  (`bind.rs:2449, 2496`), why `operands().len() == 9` is read as a state discriminator at
  `msl.rs:2542` (and per the brief at `bind.rs:237`, `cpu.rs:4847`, `msl.rs:2030`), and why
  `cached_lower_inclusive`/`new_upper_inclusive` carry `i64::MAX`/`i64::MIN` sentinels
  (`bind.rs:2510-2511`, `msl.rs:2530-2534`).

Both are one missing fact: **the algebra cannot say that an iteration point is out of range.**
The extension, in `proxima-tensor/src/map.rs` (it is addressing, not computation, so it lives
beside `IndexPattern`):

```rust
/// A linear offset that is either known at spec time or bound per call from
/// the same `symbols: &[u64]` slice `shape::infer` already takes. `Symbol` is
/// what lets ONE compiled kernel serve every cache length: the emitter renders
/// it as a uniform read, never a `constexpr`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "config", serde(rename_all = "snake_case"))]
pub enum Offset {
    Static(i32),
    Symbol(SymbolId),
}

/// Index of a symbolic extent in the caller's `symbols` slice — the same
/// index `Extent::Symbolic(u16)` already carries (`op.rs:45-48`), given a name
/// so a bare `u16` can never be passed where an axis position is expected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
pub struct SymbolId(pub u16);

/// One constraint on an iteration space: `sum(coeff * iter[axis]) + offset >= 0`.
/// An iteration point failing any constraint contributes nothing — it is not
/// computed, not read, and for a `Reduce` not accumulated.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
pub struct HalfSpace {
    pub terms: SmallVec<[AxisTerm; MAX_INLINE_TERMS]>,
    pub offset: Offset,
}

/// The iteration space's shape beyond its rectangular extents. `Domain::FULL`
/// (no constraints) is every op shipped before this change, so the default is
/// byte-identical to today's behaviour.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
pub struct Domain {
    pub constraints: SmallVec<[HalfSpace; MAX_INLINE_CONSTRAINTS]>,
}
```

`MAX_INLINE_CONSTRAINTS` joins `proxima-tensor/src/sized.rs` next to `MAX_INLINE_TERMS`
(`map.rs:38`) and `MAX_INLINE_RANK` (`bind.rs:86`), with its default in
`proxima-tensor-runtime.toml` — principle 12, and the same `sized` mechanism `map.rs:38` already
uses. Default 2 (causal is one; sliding-window is two).

It attaches to the op whose iteration space it constrains:

```rust
pub struct Reduce {
    pub dtype: DType,
    pub body: ScalarOp,
    pub init: ReduceInit,
    pub operand: NodeId,
    pub in_map: IndexMap,
    pub out_map: IndexMap,
    pub keep: Keep,
    pub name: Option<String>,
    /// Constraints on THIS reduce's iteration space. Empty = the full box,
    /// which is every `Reduce` this crate has built to date.
    pub domain: Domain,
}
```

`Op::Elementwise` gains the same field. Spelling in TOML, mirroring `NodeSpec`'s existing
`in_map`/`out_map` string grammar (`spec.rs:141-151`):

```toml
[[node]]
op = "reduce"
id = "attended"
body = "add"
init = "zero"
keep = "reduce"
in_map = "stud->stugd"
out_map = "sugd->stugd"
# causal band. `?1` is symbol 1 (merged cache length), bound per call, not baked
domain = ["s - t + ?1 >= 0"]
```

### A.3 Proof that nothing smaller works

Four candidate smaller extensions, each ruled out by a defect it does not remove:

1. **No extension; keep the `Iota`/`Greater`/`Select` mask.** Does not remove either defect in
   §A.2. The band is data (a materialized `s x t` tensor), so a lowering pass can only recover a
   loop bound by *matching* the mask arithmetic — which is what `exact_merged_causal_mask_cached_len`
   and the eight literal stride tuples at `bind.rs:2451-2474` are. Rejected: it is the status quo.
2. **Wider `Multiply` arity** (today 2, `op.rs:95-103`). Lets one `Elementwise` carry
   `q * k * mask` in a single node. Does not touch the iteration space: still `2t` trips. Rejected
   as insufficient, not as wrong — arity is a separate, smaller question (§C.4).
3. **A band as an `IndexMap` variant** (`IndexMap::Banded { .. }`). Addresses *one operand*. The
   band constrains the key operand read, the value operand read, and the reduce's own
   accumulation — three places that must agree, with shape inference reconciling them. That is one
   fact written three times, which is the compensator pattern: N copies to make 1 concept usable.
   Rejected.
4. **A tuple-valued reduce.** Strictly larger than `Domain` (changes `Reduce`'s dtype/init/body
   cardinality, `shape::infer_reduce`, and every backend's reduce path) and unnecessary, per §0.
   Rejected.

`Domain` is also the *only* extension across all five fusions (§A.6) — which is the falsifiable
form of the claim, not a preference.

### A.4 The fusion rule, and the constructor it produces

`BoundOpKind::CachedAttention` is deleted. In its place, the bound IR gains the loop-nest
counterpart of `ComposedBody` — which already fuses *scalar bodies* inside one op
(`bind.rs:176-184`) but cannot fuse two loop nests:

```rust
/// A contiguous run of already-bound ops sharing ONE iteration space, executed
/// as one kernel. `ComposedBody` (bind.rs:176) fuses scalar bodies inside a
/// single op; this fuses whole nests. Attention, epilogue fusion, prologue
/// fusion, RoPE and the rmsnorm two-phase are all instances — no variant here
/// names any of them.
#[derive(Debug, Clone, PartialEq)]
pub struct FusedRegion {
    /// Members in program order. Every operand of `ops[i]` is either an
    /// external input (a NodeId not produced inside the region) or `ops[j]`
    /// for some `j < i`, whose value never reaches memory.
    pub ops: Vec<BoundOp>,
    /// Members whose node is live outside the region and must be written.
    /// More than one is normal: RoPE's even and odd halves are both cache
    /// roots (`spec.rs:2333`), so both are region outputs.
    pub outputs: SmallVec<[u16; MAX_INLINE_REGION_OUTPUTS]>,
    /// Region iteration extents. Every member's `BoundOp::extents` equals
    /// this; a member that does not vary along an axis says so through its
    /// operand `Layout` strides (stride 0, `bind.rs:88-92`) and its own
    /// `output_axes` — never through a field here.
    pub extents: Vec<u64>,
    pub domain: BoundDomain,
    pub schedule: RegionSchedule,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BoundOpKind {
    Fused(FusedRegion),
    Elementwise { body: ComposedBody, operands: BoundOperands },
    Reduce { /* unchanged */ },
    Iota,
    Constant { value: f32 },
}
```

`BoundDomain` is `Domain` with iteration-axis coefficients resolved, the exact relationship
`Layout` has to `IndexPattern` (`bind.rs:93-97`):

```rust
/// `sum(coefficients[axis] * iter[axis]) + offset >= 0`, in the BoundOp's own
/// iteration-axis space. The resolved counterpart of `map::HalfSpace`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundHalfSpace {
    pub coefficients: SmallVec<[i64; MAX_INLINE_RANK]>,
    pub offset: BoundOffset,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundOffset {
    Static(i64),
    /// Read from the driver's per-dispatch uniform block at this slot. Never
    /// a `constexpr` — that is what keeps one compiled kernel serving every
    /// cache length, the property `entry_name`'s "dyn" marker buys today
    /// (`msl.rs:2535-2541`) without a ninth operand to carry it.
    Symbol(SymbolId),
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BoundDomain { pub constraints: SmallVec<[BoundHalfSpace; MAX_INLINE_CONSTRAINTS]> }
```

**The rule** — stated as an invariant over structure, with no operand count, no stride literal, no
model name anywhere in it. It runs on `&[Op]` + `Shapes` + liveness *before* bind, which is what
kills the double-bind at `bind.rs:2641/2678`:

```rust
/// Partitions a program into fusible regions. A maximal run of consecutive
/// nodes `[first..=last]` is a region when, for every producer/consumer edge
/// crossed inside it:
///   R1 the producer's every consumer is inside the run (`live::annotate`);
///   R2 the consumer reads the producer through an AFFINE `IndexMap` (a
///      `Computed` gather ends a region — a data-dependent address cannot be
///      kept in a register);
///   R3 producer and consumer carry the same `Domain`;
///   R4 no member is a program output except through `FusedRegion::outputs`;
///   R5 the recompute factor — the product of region extents the producer does
///      not vary along — is within `policy.max_recompute_factor`, OR the
///      producer's materialized size is below `policy.min_materialized_bytes`.
///
/// R5 is the only cost term, and it is data (`FusionPolicy`), not a predicate
/// baked into the pass. It replaces `quarantine_broadcast_operands`
/// (`bind.rs:935-973`), which is R5 with the ratio hard-wired to "never".
#[must_use]
pub fn plan_regions(
    program: &[Op],
    shapes: &Shapes,
    outputs: &[NodeId],
    capabilities: FusionCapabilities,
    policy: FusionPolicy,
) -> Result<RegionPlan, TensorError>;

/// Region spans over the program, in node order, non-overlapping.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RegionPlan { pub spans: Vec<RegionSpan> }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionSpan { pub first: NodeId, pub last: NodeId, pub schedule: RegionSchedule }
```

and `bind` consumes it once:

```rust
#[must_use]
pub fn bind_with_regions(
    program: &[Op],
    shapes: &Shapes,
    outputs: &[NodeId],
    regions: &RegionPlan,
) -> Result<Vec<BoundOp>, TensorError>;
```

`bind_with_fusion(.., fuse_cached_attention: bool)` (`bind.rs:2635-2640`) is deleted. Its `bool`
destroyed the distinction between "cannot render regions at all" and "cannot render the rescaled
schedule"; the replacement carries both:

```rust
/// Which region schedules a backend can render. Absent schedules are never
/// proposed, so the plain chain survives for wgpu/cuda exactly as
/// `fuse_cached_attention = false` produced today.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FusionCapabilities {
    pub schedules: ScheduleSet,
    /// Largest member count this backend renders. `None` = unbounded.
    pub max_region_ops: Option<u16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ScheduleSet(u8);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionScheduleKind { Nest, Rescaled }

impl ScheduleSet {
    pub const EMPTY: Self = Self(0);
    #[must_use] pub const fn of(kinds: &[RegionScheduleKind]) -> Self;
    #[must_use] pub const fn contains(self, kind: RegionScheduleKind) -> bool;
}
```

Callers: `omega::metal` and `proxima_tensor::cpu` pass `ScheduleSet::of(&[Nest, Rescaled])`;
`omega::wgpu_driver` and `omega::cuda` pass `ScheduleSet::EMPTY` (their current `false`).

### A.5 `RegionSchedule` — where the single-pass softmax lives

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegionSchedule {
    /// Emit the nest as written: each member at the depth where every axis it
    /// varies along is bound. A member reading an accumulator that an inner
    /// loop produced forces a second pass over that axis.
    Nest,
    /// One pass, with running rescale. Legal when the region contains a
    /// `Maximum` reduce over axis `a` and one or more `Add` reduces over the
    /// same `a` whose element bodies depend on that maximum only through
    /// `Exponential(Subtract(x, m))`. Both bodies are associative
    /// (`ScalarOp::is_associative`, op.rs:112-118), which is the whole
    /// justification: this is a reassociation, not a different program.
    /// Names axis and members, never "softmax" or "attention".
    Rescaled { axis: u16, maximum: u16, sums: SmallVec<[u16; MAX_INLINE_RESCALED_SUMS]> },
}
```

The detection is a structural predicate over `BoundOpKind::Reduce`'s own fields
(`reduce_op`, `init`, `output_axes`, `element_body`'s `BodyStep`s, `bind.rs:169-184`). A program
that computes a stable logsumexp gets it; a program that computes attention laid out differently
gets it; a program that names itself `attention` and does something else does not.

### A.6 The census's four levers, against this rule

Per `scratchpad/dispatch-census.md:32-49`. For each: does the algebra express it today, does it
need an extension, and is it a rule or a matcher.

| lever | expressible today | extension needed | rule |
|---|---|---|---|
| **Epilogue fusion** (`residual1` #12, `x_next` #19, `ffn_hidden` #17; 3/layer, 96/token) | yes — an `Elementwise` consuming a `Reduce` is already two ordinary nodes | **none** | R1+R2+R3 with `Domain::FULL`. The consumer's iteration space equals the reduce's *output* space, so it is placed at the outer depth of the same nest, after the accumulator closes. `BoundOpBuilder` today fuses only *into* reduce operands (`bind.rs:697-732`, the prologue direction); the region is direction-agnostic because it never asks "which side of the reduce is this". |
| **Prologue-with-broadcast** (`normed` #2/#14 into q/k/v/gate/up; 2/layer, 64/token) | yes | **none** | R5. `quarantine_broadcast_operands` (`bind.rs:935-973`) is this rule with the ratio pinned at zero. Making the ratio `FusionPolicy` data is what unblocks it; the census's own framing (`dispatch-census.md:39-40`: a 3-op recompute on an already-loaded vector vs. a 16 KB materialization plus a dispatch) is exactly the two terms R5 compares. |
| **RoPE** (#6-9; 2/layer fused as a pair, 4/layer folded into the matvec epilogue) | yes | **none** | R2. Today the gate is `is_identity_projection` (`bind.rs:699`, census line 11: it fails on the `2i`/`2i+1` stride and on the GQA map `h = group*u + g`). R2 replaces "identity" with "affine", and R5 bounds the cost. The even/odd halves both staying visible is handled by `FusedRegion::outputs` being a *list*, which is why that field is plural. |
| **rmsnorm two-phase** (#1/#13; 2/layer, 64/token) | yes | **none** | The nest form: a `Maximum`-free `Add` reduce over `d`, then an `Elementwise` re-iterating `d` and reading the scalar. Crossing a depth boundary backwards is where the emitter places a threadgroup barrier — a structural property of the nest, computed, not configured. |
| **attention** (#10; 1/layer) | yes, as the 8-node chain in §A.1 | **`Domain`** (§A.2) | R1-R5 with a non-empty domain. |

So: **one extension, five fusions.** The census's ceiling (`dispatch-census.md:49`) of 19 -> 8
bound ops/layer, 616 -> 264 dispatches/token, is reached by *one* rule and *one* emitter rather
than four passes. I am not claiming that ceiling will be measured — it is the census's arithmetic,
and CARD gates in §E measure each card against ms/token directly.

### A.7 Emitters and the CPU evaluator implement the same rule

`omega::msl::emit` (`msl.rs:871-895`) gains one arm, and loses one:

```rust
// omega/src/msl.rs
- BoundOpKind::CachedAttention { .. } => render_cached_attention(resolved, &entry),
+ BoundOpKind::Fused(region)         => render_region(region, resolved, &entry, &quantized),
```

`render_region` emits a loop nest from `region.extents`, `region.domain`, and each member's own
`Layout::strides` — that last clause is the whole of audit finding 2. `render_cached_attention`
(`msl.rs:2578-2591`) is deleted along with the `qbase`/`kbase` arithmetic it hand-derives, the
`dynamic_cached_len` branch (`msl.rs:2542-2555`), and the `constexpr cached_key_rows`/`new_key_rows`
declarations (`msl.rs:2563-2566`). Domain constraints render as the loop bounds themselves where a
constraint is monotone in the loop axis (`s - t + ?1 >= 0` gives `t <= s + sym1`, a bound), and as
a guard otherwise. Rendering it as a bound rather than a `continue` is what removes the `2t`-for-`t`
trip count (`msl.rs:2586`).

`proxima_tensor::cpu` gains the mirror: a `Fused` arm walking the same nest with the same domain
test, replacing the `CachedAttention` arms at `cpu.rs:4765`, `cpu.rs:4830`, `cpu.rs:4979`,
`cpu.rs:17955`, `cpu.rs:18103`, `cpu.rs:18144`. Because both read `domain` and `Layout`, the CPU
oracle and the GPU kernel implement one rule, and the parity test in §E.CARD-A3 is a test of that
rule, not of two hand-matched implementations.

### A.8 Migration from `BoundOpKind::CachedAttention`

Deleted, in this order, each step behind the same gate (§E):

1. `bind.rs` — `cached_attention_candidates`, `cached_attention_single_range_candidates`
   (the eight-literal-stride block at `bind.rs:2451-2474` and the rank gates at `2417-2433` go
   with them), `attention_dependencies` (`2520`), `attention_consumers` (`2559`),
   `removable_attention_dependencies` (`2585`), `has_external_attention_consumer` (`2545`),
   `bind_with_fusion`'s `bool` (`2635`), the second `bind_plain` call (`2678`).
2. `bind.rs:240-251` — the `CachedAttention` variant, its 10 fields, the `i64::MIN`/`i64::MAX`
   sentinels, and its arms in `BoundOp::operands` (`bind.rs:309`), `element_body` (`326`),
   `split_axis` (`409`).
3. `msl.rs` — `render_cached_attention` and the `operands().len() == 9` read (`2542`).
4. `metal.rs` — `pack_cached_attention_uniforms` (`2197`) folds into `pack_uniforms` (`2183`): a region's uniforms are extents + strides + symbols, the shape it already produces.
5. `cpu.rs` — the six arms listed in §A.7.

The `cached-attention-streaming` feature (`omega/Cargo.toml:21`) is renamed `region-fusion` and
keeps its default-on position; the two are never both compiled, so there is no migration window
in which a program can bind to either shape.

---

## B. The decode step as FSM x orchestration over pipes

### B.1 The sans-IO step machine

Enum, one variant per state, each transition consuming the old value, each device-dependent
transition taking the device's *product* as an argument. Zero Metal, zero `omega` dependency:
this type lives in `proxima-tensor` (alloc tier) and is drivable from a fuzzer, a CPU loop, or a
DPDK-style poll loop.

```rust
/// One decode step. `'plan` borrows the compiled plan; nothing here owns a
/// device handle, a queue, or a buffer.
#[derive(Debug)]
#[must_use]
pub enum Step<'plan> {
    Ready(Ready<'plan>),
    Encoded(Encoded<'plan>),
    InFlight(InFlight<'plan>),
    Sampled(Sampled<'plan>),
    Halted(Halt),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TokenId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StepIndex(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubmitTicket(pub u64);

/// Awaiting this step's input tokens. Owns the position bookkeeping the
/// closure at `proxima-model-interop/src/generate.rs:2278-2545` carries in
/// captured locals today (`cached_len`, `next_ids`, `merged_len`).
#[derive(Debug)]
pub struct Ready<'plan> {
    plan: &'plan Plan,
    cached: CachedLength,
    index: StepIndex,
}

impl<'plan> Ready<'plan> {
    /// Bind this step's varying inputs and build the command script. Pure:
    /// no allocation on a plan hit (§B.5), no device call. `symbols` carries
    /// the merged length that `Domain`'s `BoundOffset::Symbol` reads, which is
    /// why `cached_len` is no longer a tensor (§C.3).
    pub fn encode(self, ids: &[TokenId]) -> Result<Encoded<'plan>, StepError>;
}

#[derive(Debug)]
pub struct Encoded<'plan> { /* plan, index, script offsets */ }

impl<'plan> Encoded<'plan> {
    /// Everything a driver needs to submit, borrowed from the plan. No copy.
    pub fn script(&self) -> &CommandScript<'plan>;
    /// The driver submitted; hand back the ticket it got.
    pub fn submitted(self, ticket: SubmitTicket) -> InFlight<'plan>;
}

#[derive(Debug)]
pub struct InFlight<'plan> { /* plan, index, ticket */ }

impl<'plan> InFlight<'plan> {
    pub fn ticket(&self) -> SubmitTicket;
    /// The driver's completion, with the logits it read back.
    pub fn completed(self, logits: LogitsView<'_>) -> Result<Sampled<'plan>, StepError>;
}

#[derive(Debug)]
pub struct Sampled<'plan> { /* plan, index, chosen */ }

impl<'plan> Sampled<'plan> {
    pub fn token(&self) -> TokenId;
    /// Advance the cache length and produce the next state. `Halted` when the
    /// token is an EOS or the budget is spent — the decision
    /// `decode_until_stop_or_budget` makes in a closure today.
    pub fn advance(self, budget: Budget) -> Step<'plan>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Halt { EndOfSequence(TokenId), BudgetSpent(StepIndex) }
```

An exhaustive `match` on `Step` is the driver loop. There is no `is_encoded` boolean, no
"am I in the right state" check, and no `Option` standing in for a state.

### B.2 The driver as a pipe chain

`omega` gains a dependency on `proxima-primitives` (`cargo add --path ../proxima-primitives`;
it is absent from `omega/Cargo.toml` today — `proxima-tensor` already depends on it, e.g.
`bind.rs:64`). Four stages, named by `In`/`Out`, no category invented:

```rust
// omega/src/metal.rs — the only place in this design that touches a device.

/// transform. PURE: builds the per-dispatch argument list from a plan and this
/// step's varying inputs. No device call, no allocation on a plan hit.
pub struct Encode;
impl Pipe for Encode {
    type In = (Encoded<'static>, StepInputs<'static>);   // lifetimes elided in prose
    type Out = CommandScript<'static>;
    type Err = MetalError;
    fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>>;
}

/// transform. THE Metal edge — the ONE place a command buffer is created,
/// encoded, committed and waited on. Replaces all nine entry points.
pub struct Submit<'device> { device: &'device Device, policy: SubmitPolicy }
impl Pipe for Submit<'_> {
    type In = CommandScript<'_>;
    type Out = Completion;
    type Err = MetalError;
    fn call(&self, script: Self::In) -> impl Future<Output = Result<Completion, MetalError>>;
}

/// transform. Reads back the nodes the plan marked readable.
pub struct Read<'device> { device: &'device Device }
impl Pipe for Read<'_> {
    type In = Completion;
    type Out = LogitsView<'_>;
    type Err = MetalError;
    fn call(&self, done: Completion) -> impl Future<Output = Result<LogitsView<'_>, MetalError>>;
}

/// transform. PURE.
pub struct Sample { policy: SamplePolicy }
impl Pipe for Sample {
    type In = LogitsView<'_>;
    type Out = TokenId;
    type Err = MetalError;
    fn call(&self, logits: Self::In) -> impl Future<Output = Result<TokenId, MetalError>>;
}
```

The step is the composition law, nothing else (`primitives.rs:203-221`):

```rust
let step = Encode
    .and_then(Submit { device: &device, policy: config.submit })
    .and_then(Read { device: &device })
    .and_then(Sample { policy: config.sample });
```

Per-token telemetry is an **observe** pipe — `Out = In` — inserted where wanted; it is not a
wrapper type and not a `Timed<P>` combinator. Per-op timing is not a pipe at all: it is
`SubmitPolicy::PerOpBuffer`, one field (§B.4).

### B.3 Nine entry points to one

| deleted | becomes |
|---|---|
| `execute` (`metal.rs:519`) | `plan(..)?` then the chain |
| `execute_plan` (`535`) | the chain, `StepInputs` with empty placements |
| `execute_plan_with_placements` (`978`) | `StepInputs::placements` |
| `plan_named` (`1177`) | `plan` + `NameTable` resolved once (a pure pre-pass over `Op::Input`'s `name`, `op.rs:185-189`) |
| `execute_plan_named` (`1192`) | as above |
| `execute_plan_named_with_placements` (`1210`) | as above + placements |
| `execute_plan_op_timed` (`1409`) | `SubmitPolicy::PerOpBuffer` |
| `execute_plan_named_op_timed` (`1484`) | as above |
| `execute_plan_with_placements_op_timed` (`1509`) | as above |
| `execute_plan_named_with_placements_op_timed` (`1613`) | as above |

Three axes ({named, placed, timed}) x one driver produced ten functions; two of them contained
the same copy-pasted block-upload loop (`metal.rs:1039-1060` vs `1530-1553`), which exists once
after this. The axes become: a plan-time name resolution, a `StepInputs` field, and a config
enum. That is principle 4's config-as-composition: a new axis is a field, not four more functions.

### B.4 Plan-time transforms, and a pure `plan()`

Everything below is a pure function of the plan. Each is a **transform pipe**, composed with
`AndThen`, so `plan()` is itself a pipe chain and inherits `AndThen`'s marker propagation
(`primitives.rs:372-399`: an `AndThen` of `WithoutFilesystem + WithoutNetwork + WithoutTime`
stages carries those markers, which is the compile-time proof that planning does no I/O).

```rust
/// transform: liveness -> slot assignment. Pure interval assignment; NO device.
/// Splits `build_buffer_arena` (`metal.rs:491`, called from inside `plan`).
pub struct PlanSlots;
impl Pipe for PlanSlots {
    type In  = (Vec<BoundOp>, Vec<Vec<NodeId>>, Vec<NodeId>);  // resolved, retires, outputs
    type Out = SlotPlan;
    type Err = MetalError;
}

/// transform: slot plan -> barrier schedule. This is `HazardTracker`
/// (`metal.rs:839-887`) run at PLAN time with `Id = SlotId` instead of per
/// token with `Id = *const ProtocolObject<dyn MTLBuffer>`. That type is
/// already generic over `Id` (`metal.rs:839`), so this is reuse, not new code.
pub struct ScheduleBarriers;
impl Pipe for ScheduleBarriers {
    type In  = (Vec<BoundOp>, SlotPlan);
    type Out = BarrierSchedule;
    type Err = MetalError;
}

/// transform: bound ops -> one uniform blob per position. `pack_uniforms`
/// (`metal.rs:2183`) is already pure; this just runs it once per plan.
pub struct PackUniforms;
impl Pipe for PackUniforms { type In = Vec<BoundOp>; type Out = UniformTable; type Err = MetalError; }
```

with

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SlotId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotPlan {
    /// Slot each position writes to, and its byte length.
    pub writes: Vec<(SlotId, usize)>,
    pub slot_bytes: Vec<usize>,
    pub peak_bytes: usize,
}

/// One entry per plan position. `None` = no barrier before this dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BarrierSchedule { pub before: Vec<Option<Barrier>> }

/// The exact slots that must be visible — not "all buffers". `HazardTracker`
/// knows the slot set at plan time, so `memoryBarrierWithResources` is
/// expressible where `MTLBarrierScope::Buffers` (`metal.rs:1105`) is used
/// today, and `HazardTracker::reset`'s clear-everything (`metal.rs:874-877`)
/// becomes a per-slot clear.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Barrier { pub resources: SmallVec<[SlotId; MAX_INLINE_BARRIER_SLOTS]> }
```

`plan()` becomes:

```rust
/// PURE. No `device_and_queue()`, no allocation of a device buffer, no
/// `MTLBuffer`. Compiles and runs on a non-macOS target, which is the test
/// that proves it (§E.CARD-B1).
#[must_use]
pub fn plan(
    program: &[Op],
    symbols: &[u64],
    codecs: &[BlockCodec],
    outputs: &[NodeId],
    capabilities: FusionCapabilities,
) -> Result<Plan, MetalError>;

/// The device half, split out. Called once per plan, at first execution.
impl Device {
    #[must_use]
    pub fn realize(&self, plan: &Plan) -> Result<Residency, MetalError>;
}
```

Note `codecs: &[BlockCodec]` replaces `blocks: &[QuantizedBlock<'_>]`: `plan` reads blocks *for
their codecs and shapes only* (its own doc, `metal.rs:462-464`), so taking the data was the
signature lying about what it uses.

### B.5 Owned state: the eight thread-locals

`omega/src/metal.rs` declares six `thread_local!` blocks (`252, 2477, 2933, 3007, 3150, 3246`) —
device+queue, output buffer pool, checkpoint mapping, no-copy cache, resident copies, uniform
buffers — plus the `register_checkpoint_mapping` side channel (`metal.rs:2957`). Every one becomes
a field with a single owner. Principle 21's resolution reads the same for a thread-local as for a
lock: **a thread_local is a missing owner**, and the fix is the owner, not a `Mutex`.

```rust
/// Owns the Metal device, its queue, and every cache keyed by device identity.
/// One per thread of execution by construction; two `Device`s on two cores
/// share nothing and need no lock (principle 21 rung 1).
pub struct Device {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipelines: BTreeMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    no_copy: NoCopyCache,
    checkpoint: Option<CheckpointMapping>,   // was the register_* side channel
}

/// Plan-owned, because their lifetime is the plan's: an arena slot and a
/// uniform blob are meaningless without the plan that indexed them.
pub struct Residency { arena: Vec<MetalBuffer>, uniforms: Vec<MetalBuffer>, outputs: Vec<MetalBuffer> }

/// Owns the plan cache. Replaces the one-entry `clear()`-then-`insert` cache
/// at `proxima-model-interop/src/generate.rs:1325, 1373, 1417`.
pub struct Driver {
    device: Device,
    plans: PlanCache,
    config: DriverConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PlanKey { pub key_capacity: u32, pub new_tokens: u32 }

/// Bounded, capacity from `omega::sized::PLAN_CACHE_ENTRIES`. The existing
/// cache holds ONE entry and clears on any miss, so a prefill/decode
/// alternation thrashes; two entries covers it and the constant says why.
pub struct PlanCache { entries: ArrayVec<(PlanKey, Plan), PLAN_CACHE_ENTRIES> }
```

`UNIFORM_BUFFERS`' content-keyed LRU (`metal.rs:3246-3325`, `evict_least_recently_used` at
`3315`) is **deleted**, not moved: it exists to dedupe uniform blobs across shapes, and a
plan-owned `UniformTable` indexed by position has nothing to dedupe. Less work, same output.

### B.6 Allocation budget per token

Target on a plan hit: **zero**. Sources removed, each cited:

| site | today | after |
|---|---|---|
| `generate.rs:2301-2306` | `named_blocks: Vec` rebuilt every step, capacity `owned + packed + packed_owned + 4 + block_count*3` | built once into a plan-owned `BlockTable`; per step only `ids`, `rope_cos`, `rope_sin` are overwritten in place |
| `generate.rs:2323` | `cached_len_scalar: [f32; 1]` + a named block | deleted — `cached_len` is `SymbolId` (§C.3) |
| `generate.rs:2361-2364, 2400-2406, 2504` | per-step vectors | plan-owned, reused |
| `metal.rs:541` | `device_buffers: BTreeMap` per call | plan-owned `SlotTable`, indexed by position |
| `metal.rs:1095-1100` | `hazard_inputs: Vec` per op | deleted — barriers are plan-time (§B.4) |
| `metal.rs:1073` | `pending_faults: Vec` per call | plan-owned, capacity from the plan's gather count |
| `metal.rs:2183` | `pack_uniforms -> Vec<u8>` per op per call | plan-time `UniformTable` |

The counter test — the clause is aspirational without it (principle 11):

```rust
/// A plan-hit decode step allocates nothing. Runs 100 steps against a real
/// openchat-3.5 Q4_K_S plan after the first (plan-miss) step, and asserts the
/// delta of a counting global allocator is exactly zero. Asserts the STEP
/// COUNT too: a zero-step run and a zero-alloc run report the same number
/// otherwise.
#[test]
fn a_plan_hit_decode_step_allocates_nothing_over_a_hundred_steps() {
    // arrange: one plan-miss step to populate the cache
    // act: 100 steps through Encode.and_then(Submit).and_then(Read).and_then(Sample)
    // assert: allocations == 0 AND steps_run == 100
}
```

The counting allocator does not exist today (grep for `GlobalAlloc` across `proxima-primitives/src`, `proxima-tensor/src`, `proxima-test` returned nothing); building it is part of this work (§E, P15).

---

## C. Generic model programs

### C.1 Production decode from spec data

`ProgramSpec` (`spec.rs:72-78`) is a flat `Vec<NodeSpec>` with `TryFrom<&ProgramSpec> for
Vec<Op>` (`spec.rs:361`), proven at the model's real dimensions by the test at `spec.rs:10313`.
Production does not use it: `proxima-tensor/src/spec.rs:898..7184` hand-writes
`append_mistral_*` / `qwen35_*`, and `proxima-model-interop/src/generate.rs:624` wraps the result
in `SingleRangeProgram`.

A 32-layer stack as TOML must not be 32 copies of a layer, and it must not grow a template
language. The minimal addition is **one lowering entry point**, not a new spec dialect: lower the
same spec repeatedly into a growing program, with a name prefix and externally supplied bindings
for the ids it does not define.

```rust
/// Bindings a spec does not define itself: the layer's input activations, and
/// any weight or cache leaf the caller has already appended. Keyed by the
/// spec's own `id`.
pub type SpecBindings<'ids> = &'ids [(&'ids str, NodeId)];

impl ProgramSpec {
    /// Append this spec's nodes to `program`, prefixing every `Op::Input`
    /// `name` with `prefix` and resolving any id present in `bound` to the
    /// caller's existing node instead of defining a new one. Returns the id ->
    /// NodeId map for this instantiation.
    ///
    /// This is the ONLY addition the data path needs to build a 32-layer
    /// model: a stack is this called 32 times with prefixes `blk.0.`..`blk.31.`,
    /// threading each call's output id in as the next call's `x` binding.
    #[must_use]
    pub fn lower_into<'ids>(
        &self,
        program: &mut Vec<Op>,
        prefix: &str,
        bound: SpecBindings<'ids>,
    ) -> Result<BTreeMap<String, NodeId>, TensorError>;
}
```

and the model as data, in the conflaguration house pattern (the same
`Builder + Deserialize + Serialize + Settings` shape `ProgramSpec` already carries at
`spec.rs:69-71` and `TensorExecutionConfig` at `config.rs:43-46`):

```rust
#[derive(Debug, Clone, PartialEq, Default, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "TENSOR_MODEL")]
#[builder(derive(Clone, Debug))]
pub struct ModelSpec {
    /// Prologue: token embedding gather. One `ProgramSpec`.
    pub embedding: StageSpec,
    /// Instantiated `block_count` times with prefix `blk.{n}.`.
    pub layer: StageSpec,
    /// Epilogue: final norm + lm_head.
    pub head: StageSpec,
    pub block_count: u32,
    /// Checkpoint metadata, by the gguf key it came from. Values substitute
    /// into `ExtentSpec::Named`, so a layer spec says `head_count` and not 32.
    pub dimension: BTreeMap<String, u32>,
}

/// A stage is either inline nodes or a path to a spec file — the same two
/// spellings `conflaguration`'s file/mapping sources already give a config.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum StageSpec { Inline(ProgramSpec), Path(String) }
```

`ExtentSpec` (`spec.rs:83-87`) gains one variant, matching its existing untagged-by-shape rule
(a bare integer is `Static`, `"?0"` is `Symbolic`):

```rust
pub enum ExtentSpec {
    Static(u32),
    Symbolic(String),        // "?0"
    /// "$head_count" — resolved from `ModelSpec::dimension` at lowering time.
    Named(String),
}
```

`SingleRangeProgram` (`generate.rs:624`) and `Qwen35SsmShape` (`generate.rs:731`) then hold
`ModelSpec`-derived data instead of hand-built programs. `find_input_node` (`generate.rs:634`)
and `locate_cache_input_nodes` are deleted: `lower_into` returns the id map, so the cache input
nodes are read out of it by name rather than rescanned O(program) per name.

### C.2 The layer builder's 23 NodeIds

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub struct AttentionWeights { pub query: NodeId, pub key: NodeId, pub value: NodeId, pub output: NodeId }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub struct FeedForwardWeights { pub gate: NodeId, pub up: NodeId, pub down: NodeId }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub struct NormWeights { pub attention: NodeId, pub feed_forward: NodeId, pub epsilon: NodeId }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub struct RopeTables { pub cosine: NodeId, pub sine: NodeId }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub struct CacheInputs { pub key: KeyRoots, pub value: NodeId }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerInputs {
    pub activations: NodeId,
    pub norms: NormWeights,
    pub attention: AttentionWeights,
    pub feed_forward: FeedForwardWeights,
    pub rope: RopeTables,
    pub cache: CacheInputs,
}

#[must_use]
pub fn append_cached_layer(
    program: &mut Vec<Op>,
    inputs: LayerInputs,
    shape: LayerShape,
) -> Result<LayerOutputs, TensorError>;
```

Honest claim: every field is a `NodeId`, so this does not make a swap a *type* error. It makes it
a *named* error at the construction site — `AttentionWeights { query, key, value, output }` cannot
be filled in the wrong order without writing the wrong field name. That is what removes the
`#[allow(clippy::too_many_arguments)]` and what a reader can check. Type-level distinctness would
cost 23 newtypes hosting nothing, which the no-newtype-to-host-an-impl rule forbids.

`CachedLayerRoots = (NodeId, NodeId, NodeId)` (`spec.rs:2333`, consumed at 15+ sites) and
`Qwen35DenseAttentionRoots = (NodeId, NodeId, NodeId, NodeId)` (`spec.rs:2344`) collapse to one
type with an exhaustive match, not two aliases:

```rust
/// The rotated key halves a layer contributes to the cache. Exhaustive: a new
/// rotary layout is a variant here, and every one of the 15+ consumption sites
/// is a `match` the compiler checks, not a positional destructure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum KeyRoots {
    /// Interleaved (`2i`/`2i+1`) rotary — Mistral/Llama.
    Paired { even: NodeId, odd: NodeId },
    /// Split-half (NEOX) rotary with an unrotated remainder — Qwen3.5.
    Split { first: NodeId, second: NodeId, passthrough: NodeId },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheRoots { pub key: KeyRoots, pub value: NodeId }
```

### C.3 `cached_len` and the KV bucket policy

`cached_len` travels today as an `Op::Input` bound to `QuantizedBlock::Float32(&[cached_len as
f32])` (`generate.rs:2323`) and is read back out of a device buffer by the kernel
(`msl.rs:2546`). Under `Domain`, it is `BoundOffset::Symbol(SymbolId)` read from the dispatch's
uniform block. Deleted: the leaf, the named block, the upload, the f32 round trip, the ninth
operand, and `bind.rs:2446`'s rank-0 check.

`KV_BUCKET_TOKENS` (`proxima-tensor/src/sized.rs:326`) is a Metal plan-cache-key policy living in
the IR crate. It moves to `omega/omega-runtime.toml` as `[kv] bucket_tokens` and
`omega::sized::KV_BUCKET_TOKENS`, alongside `[spans] uniform_cache_entries` and
`[packed_row_block] simdgroups`, which is where the file's own convention puts an execution-policy
knob. The IR then never learns the constant at all: the caller passes two symbols — the key
buffer's *capacity* (an extent) and the *merged length* (a domain offset) — and their relationship
is the driver's business. That is the clean form the extension buys, not a relocation.

### C.4 What a new architecture costs

**Data only** when the architecture is a composition of the shipped generators: Llama/Mistral,
GQA and MQA (an `IndexPattern` with two terms), sliding-window attention (a second `HalfSpace`),
ALiBi (an `Iota`-derived bias added before the softmax), MoE top-k (`IndexMap::Computed`, already
supported per `map.rs:83-92`), NEOX vs interleaved rotary (`KeyRoots` variants + affine maps),
tied embeddings (one leaf, two consumers).

**Rust** when the architecture needs a scalar primitive `ScalarOp` does not have — the set is
closed on purpose (`op.rs:52-56`) and stays closed, so `Floor`/`Round` (needed to *write* a
quantized KV cache, §D.1) is a deliberate algebra decision, not a variant to reach for. Rust is
also required when the architecture needs a non-scalar accumulator: Qwen3.5's gated DeltaNet
recurrence is a `Keep::Scan` whose state is a matrix, which the current `Reduce` cannot type. That
is the honest boundary, and naming it is why `Qwen35SsmShape` (`generate.rs:731`) exists today.

Three known algebra blockers, from the brief, and their status under this design:
`unify_iteration_space` resolving an extent only from a single-term coeff-1 offset-0 axis
(`shape.rs:224-227`) blocks the *interleaved* RoPE spelling — it does not block the even/odd
spelling this design fuses, so it is scoped to CARD C1 as a fallback resolution pass, not a
prerequisite for A. `Multiply` arity 2 (`op.rs:95-103`) is not on the critical path (§A.3 item 2).
`CachedLayerRoots` at 15+ sites is CARD C2.

---

## D. Bytes — the primary section

### D.0 The budget

| quantity | value | provenance |
|---|---|---|
| target | 3.5 ms/token (5x llama.cpp's 17.45) | owner, 2026-09-04 |
| M1 Max peak bandwidth | 400 GB/s | ASSUMED (spec sheet) |
| realistic sustained | ~300 GB/s | ASSUMED — CARD D0 measures it |
| budget at peak | **<= 1.400 GB/token** | DERIVED, 3.5 ms x 400 GB/s |
| budget at sustained | **<= 1.050 GB/token** | DERIVED, 3.5 ms x 300 GB/s |
| today | 4.169 GB/token | task brief, MEASURED upstream |
| floor today | 10.42 ms | DERIVED, 4.169/400 |
| required reduction | **2.98x at peak, 3.97x at sustained** | DERIVED |

CARD D0 is the first card in §E and it is a *measurement*, not a change: a streaming-only kernel
over the model's own buffers, reporting sustained GB/s with CoV. Every number below is priced
against that, not against 400.

### D.1 Where the 4.169 GB is

Parameter counts are exact for openchat-3.5 (Mistral-7B geometry: 32 layers, d_model 4096,
d_ff 14336, 32 heads, 8 kv heads, head_dim 128, vocab 32002). Byte figures are DERIVED from those
counts at the named codec's bits/weight (Q4_K 4.5, Q5_K 5.5, Q6_K 6.5625).

| tensor family | params | codec | bytes/token | share |
|---|---:|---|---:|---:|
| FFN (gate+up+down), 32 layers | 5,637,144,576 | Q4_K | 3,170.9 MB | 76.1% |
| attention (q+k+v+o), 32 layers | 1,342,177,280 | Q4_K | 755.0 MB | 18.1% |
| `output.weight` | 131,080,192 | Q6_K | 107.5 MB | 2.6% |
| `token_embd` | 131,080,192 | Q4_K | ~0.002 MB (one gathered row) | ~0% |
| KV cache @ 34 positions | — | f32 | 8.9 MB | 0.2% |
| KV cache @ 2048 positions | — | f32 | 537 MB | — |
| KV cache @ 8192 positions | — | f32 | 2,147 MB | — |
| norms, uniforms, activations | — | f32 | ~2 MB | ~0% |
| **DERIVED total @ 34 ctx** | | | **4,044 MB** | |
| **stated total** | | | **4,169 MB** | |
| **unexplained residual** | | | **125 MB (3.0%)** | |

`output.weight` at Q6_K is confirmed, not assumed: `scratchpad/byte-levers-probe.md:32` reports it
Q6_K at 107 MB, and 131,080,192 x 6.5625/8 = 107.5 MB reproduces that from the parameter count.

The residual is reported, not averaged away. Partial explanation: `omega/Cargo.toml` records Q5_K
on the `blk.{0..3}.ffn_down.weight` family, which is 4 x 58,720,256 x (5.5-4.5)/8 = 29.4 MB of it.
The remaining ~96 MB is unexplained; CARD D0 resolves it by reading the gguf tensor table's
per-tensor codecs (`proxima-gguf/src/types.rs:109-133`) rather than assuming a uniform Q4_K.

Two facts the table settles before any lever is chosen:

- **FFN is 76%.** A plan that does not move FFN bytes cannot reach 2.98x.
- **KV scales with context and at 8192 exceeds the weights.** Pricing KV at its 34-position value
  (0.2%) prices the benchmark, not the workload.

### D.2 L1 — multi-token per pass, as ONE program (LOSSLESS)

**Bytes: W/k'**, where `k'` is the expected number of accepted tokens per pass. This is the only
lever that divides all 3.93 GB of weights at once, and the only lossless one.

**What exists** (`byte-levers-probe.md:3-12`, verified file:line):

- The program and the plan cache are already `new_count`-generic: symbols are
  `[new_count, kv_bound_extent]` (`generate.rs:2394`), and the plan cache key is
  `(new_count, bucket)` (`generate.rs:1248, 1323`). The IR and the cache need no change for k > 1.
- The decode loop hardwires steady state to one token: `next_ids = vec![token_id]`
  (`generate.rs:2503, 2069`), and `sample_next_token` takes one logits row
  (`proxima-tokenizer/src/sample.rs:277`).
- No draft / speculative / verify machinery exists anywhere.

**The blocker that governs this lever, and it is a kernel, not the algebra.** The row-blocked
packed matvec batches 4 **output** rows per simdgroup over ONE activation vector (ggml's `nr0`)
and has no `s`-axis fold (`msl.rs:3171-3200`). Running it at `s = k` **re-streams the weights k
times**, so a k-token pass through today's decode kernel moves `k x W` bytes and the lever
returns exactly zero. The weight-once multi-token kernel exists only as `metal-tiled-gemm`
(`simdgroup_matrix`, Q4_K only, gated by `TILED_GEMM_MIN_TOKENS`; `omega/Cargo.toml:129`,
`msl.rs:1575-1608, 3961-4030`) and is unreachable from the decode path.

So L1 is **two pieces of work, in this order**:

1. **The `nr1` fold** — `push_packed_row_blocked_body` accumulates `s` activation columns per
   streamed weight row, exactly the axis `push_tiled_gemm_body` already folds. This is emitter
   work on an existing descriptor (`BoundOp::extents` already carries the `s` axis; the operand
   `Layout` already carries its stride, `bind.rs:88-92`). **No algebra.** Without it, nothing
   below is credited.
2. **Draft + verify as one program.** Every piece is shipped algebra:

| step | algebra | citation |
|---|---|---|
| k query rows in one pass | `symbols[0] = k`; sequence axis is `Extent::Symbolic` | `op.rs:41-48`; `generate.rs:2394` |
| causal band across k rows | one `HalfSpace`: `s - t + merged >= 0` | §A.2 — today needs a materialized `s x t` mask, and the fused matcher bakes the band from `query_shape[0]` (`bind.rs:2434-2448`) |
| per-row argmax | `Reduce Maximum`, then `Equal` against it, then `Reduce Add` of `Iota x Equal` | `op.rs:60-78` (`Equal`), `op.rs:209-231` (`Iota`); `ReduceInit::FirstElement` (`op.rs:124-132`) is the seed this form was added for |
| accept the longest matching prefix | `Keep::Scan` `Multiply` over the equality vector, then `Reduce Add` over the scan | `op.rs:139-147` |
| next pass's cache length | the accepted count, read back with the logits, bound as `SymbolId` | §A.2, §C.3 |

**The draft source IS a pipe** — and this is the asymmetry against §D.3's selector. It runs
**once per pass**, on the host, before any dispatch, and touches no weights:

```rust
/// transform. `In` is the generated-token buffer, `Out` is up to `k` proposed
/// continuations. An n-gram / prompt-lookup source reads only the token buffer
/// and costs ZERO weight bytes; a small draft model is the same shape with a
/// second plan behind it, which is why this is one pipe and not two designs.
pub trait DraftSource: Pipe<In = TokenWindow<'_>, Out = Draft> {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Draft { pub tokens: [TokenId; MAX_DRAFT], pub len: u8 }
```

A separate draft *model* would cost ~0.6 GB per drafted token at Q4 and consume the entire lever;
prompt-lookup drafting costs nothing, which is why it is the default `DraftSource` and the model
variant is the same pipe with a plan behind it.

**No rollback machinery.** A pass writes k rows into the KV cache; the next pass sets the cache
length to the accepted count, so rejected rows are overwritten in place — the probe reaches the
same conclusion (`byte-levers-probe.md:37-38`: "placed KV rollback = do not advance cached_len past
the accepted count"). Nothing is undone, so the FSM gains one field and no state:

```rust
impl<'plan> Sampled<'plan> {
    /// Tokens the verifier accepted this pass, `1..=k`. The next `Ready`'s
    /// cache length advances by exactly this much; rejected rows are
    /// overwritten by the next pass, never rolled back.
    #[must_use] pub fn accepted(&self) -> u32;
}
```

`sample_next_token`'s one-row signature (`sample.rs:277`) becomes k-row: `Sample`'s `In` is
`LogitsView` over k rows, which it already could be — the row count is data, not a signature.

**Interaction with §A/§B.** k >= 8 also makes the tiled `simdgroup_matrix` path eligible
(`min_tokens = 8`), so once the `nr1` fold lands there are two weight-once kernels and the choice
is a `sized` threshold, not a fork.

**Quality gate: text identical.** Exact verification means accepted tokens are exactly greedy
full-model output. **Kill criterion:** any text difference at temperature 0; or measured
`k' < 1.5` on the held-out prompt set, below which the extra per-pass compute is not repaid.
`k'` is MEASURED by CARD D2 (acceptance histogram per prompt class: code, prose, chat) before
any byte reduction is credited to it.

### D.3 L2 — dynamic row elision of the FFN (LOSSY)

**Bytes: FFN x d.** The hidden axis is the output axis of `ffn_gate`/`ffn_up` and the reduced axis
of `ffn_down`, so all three matrices scale with density `d`. At `d = 0.35`:
3,170.9 MB -> 1,109.8 MB — the single largest reduction available.

**What exists** (`byte-levers-probe.md:14-25`):

- `IndexMap::Computed { indices, index_map, base, gathered_dim }` (`map.rs:134-151`) is **live in
  production** for MoE expert routing on CPU — the `run_reduce_quantized` gather arm
  (`cpu.rs:7091`, sizing `cpu.rs:7205-7261`) already computes `y[j] = sum_k W[idx[j],k] * x[k]`.
  The form is not hypothetical; it is exercised today, only at whole-expert-slab granularity.
- **Metal rejects it on the fast path**: `classify_packed_row_block` requires `gather_count == 0`
  (`msl.rs:1039-1052, 1389`), so a gathered weight falls to the serial one-thread-per-output
  kernel (`push_gather_fetch`, `msl.rs:2286-2318`) with a per-operand fault slot
  (`metal.rs:3221-3233`). That is element-granular and is the opposite of the bandwidth regime.
- **No in-graph selector**: `ScalarOp` has no top-k, threshold-count, or argsort, and
  `Keep` is `Reduce | Scan` (`op.rs:60-78`, `op.rs:139-147`). The landed elision probe built the
  skip set **on the host**, CPU-only (`proxima-tensor/benches/bench_dynamic_elision.rs`,
  discipline ROW 180/181).

**Two constraints from prior measurement, which shape the lever rather than decorate it:**

1. **It must land on the kernel path.** ROW 180/181 (memory `project_sparse_dynamic_elision_probe`)
   measured the generic executor at 0.20-0.28 ns/element against raw DRAM streaming at
   0.0572 ns/element, cold and hot alike — *"the generic path is dispatch-bound EVERYWHERE, never
   bandwidth-bound"*, and elision through it prices as `elements_skipped x ~0.2 ns`, not as
   bytes/bandwidth. Only kernel paths reach the bandwidth regime. Combined with the Metal fact
   above, this is one statement: **row elision is worth zero bytes until the row-blocked packed
   kernel accepts a row-axis gather.** That is prerequisite work, and it is emitter work, not
   algebra: the gather already resolves to a `Lookup { indices, index_layout, element_stride,
   extent }` (`bind.rs:121-136`), and the row-blocked body needs to add `element_stride x
   fetched_index` to its weight base per output row instead of a fixed stride.
2. **The selector must COMPUTE the address, not SEARCH for it.** Memory
   `project_sparse_matmul_equivalence` grades a per-row top-k over the full hidden axis as
   *"content-addressed search, fanout Theta(n) — BOTH, strictly worse than dense"*, while
   *"hash / index-computed routing, O(1) bucket"* is clean and a top-k over a set that does not
   scale with n is admitted by the O(log n) allowance. So the design is **group-routed**:
   partition the 14336 hidden units into `G` fixed groups (default 112 x 128 — one group is
   128 x 4096 params, whole Q4_K blocks), and select from `G`, never from 14336. `G` is constant,
   the channel is O(log G), and the form is the admitted one. A Deja-Vu-style per-row predictor is
   the rejected one, and the two are indistinguishable at the API — which is why the constraint is
   written here and not left to the implementer.

**The selector, in the algebra.** A router matvec (`d_model x G` = 4096 x 112 = 459k params,
0.26 MB at Q4_K — 0.008% of the FFN it gates) produces G scores. Then, with a **fixed** output
capacity `g` so the gathered extent stays static (shape inference has no dynamic extents):

| step | algebra | status |
|---|---|---|
| `active = Greater(score, tau)` (0/1 over G) | `Elementwise`, `ScalarOp::Greater` | exists (`op.rs:75`) |
| `rank = prefix-sum(active)` | `Reduce` with `Keep::Scan`, `Add` | exists (`op.rs:139-147`) |
| write group id `j` to slot `rank[j]` when active | `Reduce` with a `Computed` `out_map` — a scatter | exists (`map.rs:109-130`) |
| slots past `g` | must be **dropped**, not faulted | **does not exist** |

That last row is L2's one algebra extension, and it is one enum, not an op:

```rust
/// What a scatter does with a destination index outside the destination's
/// extent. `Fault` is today's behaviour, and today's only behaviour: `Lookup`
/// carries `extent` so an executor can "reject an out-of-range fetched index
/// instead of reading past the buffer" (`bind.rs:127-129`), and Metal raises it
/// through the `Binding::Fault` slot (`msl.rs`'s `Binding` enum).
/// `Drop` is what a fixed-capacity compaction needs: a bounded selection is
/// defined to keep the first `g` and discard the rest, not to fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "config", serde(rename_all = "snake_case"))]
pub enum ScatterBounds { #[default] Fault, Drop }
```

carried on `IndexMap::Computed` as one field. Two binary questions: *a pipe?* No — it is IR data.
*What can a caller do that they could not?* Express a bounded compaction: today the same program
either faults or must size its destination to the worst case `G`, which is the full dense read
this lever exists to avoid. It earns the field.

**Is the selector a pipe? No** — and the reason is the asymmetry against §D.2's `DraftSource`.
The draft source runs once per pass, on the host, touching no weights. The elision selector runs
**once per layer**: as a host-side pipe it means 32 readbacks and 32 pipeline stalls per token on
a path whose entire budget is one readback (task brief, state of the path), and the landed probe
(host-side skip set, CPU-only) is exactly that shape and measured dispatch-bound. So the selector
is a node in the program; the draft source is a pipe. Same question, opposite answers, and the
discriminator is how many times per pass it crosses the device edge.

**Quality gate: text identical is NOT available.** Metric: **exact-match rate** of generated tokens
against the full model, greedy (temperature 0), 200 held-out prompts x 128 tokens, plus held-out
perplexity delta. **Kill criterion:** EM < 0.95 or perplexity delta > 1.0%. **Secondary kill:**
measured GB/s on the packed-row path does not fall in proportion to `d` — that would mean the
lever is dispatch-shaped (the ROW 180 finding reproduced on GPU) and the byte model is wrong.

### D.4 L3 — lower-bit codecs (LOSSY, zero program change)

**Bytes: weights x (bits/4.5).** Q4_K 4.5 -> Q3_K_S 3.4375 is x0.764; -> Q2_K 2.625 is x0.583.
(`byte-levers-probe.md:42` prices these x0.75 / x0.6; the difference is which K-quant sub-variant,
and CARD D4 pins it by measuring the produced file.)

**What exists** (`byte-levers-probe.md:27-32`): CPU `QuantizedBlock` (`cpu.rs:3091`) and Metal
`PackedCodec` (`msl.rs:788-805`) cover F32/Q4_K/Q5_K/Q6_K/Q8_0 plus F16/BF16, with Q4_0
unimplemented downstream. **Q2_K/Q3_K/Q4_0/Q5_0 already parse** (`proxima-gguf/src/types.rs:109-133,
245-281`) and are rejected at bind (`UnrepresentableGgmlType`,
`proxima-model-interop/src/bind.rs:70-73`; `capability.rs:141-156`). IQ*/TQ* were never attempted.

So the work is bounded and named: an encoder/decoder in `proxima_gguf::quant`, an MSL unpack body,
one `PackedCodec` variant, and lifting the bind rejection for the two codecs added. **The `Op`
graph, the `Domain`, the regions and the FSM do not move** — a codec is an operand property
carried outside the program (`packed_operands_of`, `metal.rs:445-460`; `operand_codecs`,
`msl.rs:860-869`; one character per operand in `kernel_cache_key`, `msl.rs:934-946`). That is the
whole reason this lever is cheap, and it is a property of the existing design, not something this
design adds.

Two binary questions for the variant: *a pipe?* No — an in-kernel decode, not an async step.
*What can a caller do?* Read a Q3_K tensor. Legitimate.

**Quality gate:** the EM/perplexity metric of §D.3. **Kill criterion:** EM < 0.97 or perplexity
delta > 0.5% — stricter than L2 because the change is uniform across every tensor with no
per-token adaptivity to recover from. L2 and L3 land in separate cards with EM measured after
each, so their quality signals are never confounded.

### D.5 L4 — `output.weight` (107.5 MB, 2.6%)

Three sub-levers, increasing risk:

1. **Q6_K -> Q4_K.** 107.5 -> 73.7 MB. L3's mechanism on one tensor; binds today
   (`byte-levers-probe.md:32`), so this is a codec-selection change with no new code at all.
2. **Candidate-set gather.** Logits for a candidate subset instead of all 32002 rows — the same
   `IndexMap::Computed` row gather as §D.3, over the vocabulary axis, with the same packed-row
   kernel prerequisite. At 4096 candidates (12.8%): 73.7 -> 9.4 MB. The candidate set comes from
   the computed-address discipline of §D.3 (previous-token continuations plus a fixed
   high-frequency floor), never a search over 32002.
3. **Safe variant (D5c).** Rank all 32002 rows at Q2_K (42.0 MB), then confirm the top candidate against its full-precision row (16 KB): bounded error, the fallback if (2) fails its gate.

**Tied embeddings are not available**: openchat-3.5 ships separate `token_embd` and
`output.weight` (both 131,080,192 params, §D.1), so nothing can be tied. Named so it is not counted.

**Quality gate for (2):** the full model's argmax must lie inside the candidate set. Metric:
**candidate-miss rate** on the held-out set. **Kill criterion:** miss rate > 0.1%, at which point
(3) replaces it.

### D.6 L5 — KV bytes (context-dependent)

At 34 positions this is 0.2%. At 2048 it is 537 MB (11% of the budget); at 8192 it is 2,147 MB,
larger than every weight in the model. Priced at the context the workload runs, both stated.

- **f16 cache.** 2x, 537 -> 269 MB @2048. **No algebra**: it is `DType::Float16` on the cache
  leaf; `BoundOp::dtype` already carries per-node dtype (`bind.rs:206-212`), `type_token` already
  emits `half` (`msl.rs:2529`), and `Convert<f32, f16>` is already a `Pipe` (`convert.rs:214`).
- **Q8_0 cache.** 4x, 537 -> 134 MB @2048. Reading a packed operand works today; **writing** one
  needs round-to-nearest, and `ScalarOp` has no `Round`/`Floor` (`op.rs:60-78`) — the only
  *scalar-vocabulary* extension in this design. §D.7 reaches the target **without** it, so it is
  scoped last (CARD D7), off the critical path: a new `ScalarOp` variant is a deliberate decision
  against a deliberately closed set (`op.rs:52-56`), per §C.4.

**Quality gate:** EM as above. **Kill criterion:** f16 EM < 0.99; Q8_0 EM < 0.97.

### D.7 The arithmetic — does the product reach <= 1.4 GB/token?

MB/token. The per-pass column is bytes moved by one forward pass; the per-token column divides by
`k'`. `k' = 2.18` is `k = 4` at acceptance `alpha = 0.6` (`1 + 0.6 + 0.36 + 0.216`), stated
ASSUMED and measured by CARD D2 before it is credited. **Every row below assumes the two kernel
prerequisites have landed** — the `nr1` s-axis fold (§D.2) and the row-gather on the packed-row
body (§D.3). Without them L1 and L2 are worth exactly zero, which is the first sensitivity row.

**At 2048 context (the serving case):**

| stage | FFN | attn | output | KV | resid | pass total | /token |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 3,170.9 | 755.0 | 107.5 | 537.0 | 125 | 4,695.4 | 4,695.4 |
| + L4.1 (`output` Q4_K) | 3,170.9 | 755.0 | 73.7 | 537.0 | 125 | 4,661.6 | 4,661.6 |
| + L5 (f16 KV) | 3,170.9 | 755.0 | 73.7 | 268.5 | 125 | 4,393.1 | 4,393.1 |
| + L3 (Q3_K_S, x0.764) | 2,422.6 | 576.8 | 73.7 | 268.5 | 95 | 3,436.6 | 3,436.6 |
| + L4.2 (candidate gather) | 2,422.6 | 576.8 | 9.4 | 268.5 | 95 | 3,372.3 | 3,372.3 |
| + L2 (FFN d = 0.35) | 847.9 | 576.8 | 9.4 | 268.5 | 95 | 1,797.6 | 1,797.6 |
| **+ L1 (k' = 2.18)** | | | | | | 1,797.6 | **824.6** |

**824.6 MB/token** = 2.06 ms at 400 GB/s, 2.75 ms at 300 GB/s. Both under 3.5 ms; the budget
(<= 1,400 at peak, <= 1,050 at sustained) is met with 21% margin against the sustained figure.

**At 34 context (the benchmark case):** the same chain with KV 8.9 -> 4.5 MB gives a pass total of
1,533.6 and **703.5 MB/token** = 1.76 ms / 2.35 ms.

**Sensitivity — which levers are necessary, not merely helpful:**

| omitted | /token @2048 | ms @400 | ms @300 | verdict |
|---|---:|---:|---:|---|
| **`nr1` fold not landed** | 4,695.4 (L1 returns 0; k passes re-stream W) | 11.74 | 15.65 | **misses** — prerequisite, not an optimization |
| **packed-row gather not landed** | L2 falls to the serial kernel, dispatch-bound | — | — | **misses** — prerequisite |
| none (full chain) | 824.6 | 2.06 | 2.75 | meets |
| **without L1** | 1,797.6 | 4.49 | 5.99 | **misses** — L1 necessary |
| **without L2** | 1,547.4 | 3.87 | 5.16 | **misses** — L2 necessary |
| without L3 | 1,033.9 | 2.58 | 3.45 | meets at peak, marginal at sustained |
| without L5 | 886.2 | 2.22 | 2.95 | meets |
| without L4 | 854.1 | 2.14 | 2.85 | meets |
| L1 only | 2,153.9 | 5.38 | 7.18 | misses |
| L1 + L2 only | 1,175.1 | 2.94 | 3.92 | meets at peak only |

**L1 and L2 are both required; L3/L4/L5 are the margin; the two kernel folds gate both.** That is
the inverse of the risk ordering — L1 is lossless and L2 is the riskiest — which is why §E lands
the two kernel prerequisites first, then L1 (lossless, larger multiplier, text-identical gate),
then the margin levers, and L2 last among the required pair because its quality gate is the one
that can fail.

**What the arithmetic does not include, and must not be read as including:** any effect of §A's
fusion or §B's dispatch collapse. Those change dispatch count and host overhead, not bytes. If the
33.0 ms measured today were entirely bandwidth-shaped it would be 10.4 ms, so 22.6 ms of it is
not — which is what §A/§B address, and which is why the byte arithmetic here is a **floor, not a
prediction of wall clock**. CARD D0's sustained-bandwidth measurement plus the per-op profile is
what separates the two. No ms/token claim is made here.

## E. Ordering, gates, tripwires

Every card's gate is the battery below unless the card names otherwise. A gate that cannot report
its N is not a gate; each asserts a count, and N == 0 is RED.

- **P** parity <= 1e-4 vs `proxima_tensor::cpu::evaluate` on the real openchat-3.5 Q4_K_S program,
  every output node, N = node count asserted nonzero.
- **B** 100 runs byte-identical logits (N = 100 asserted).
- **T** generated text identical to the pre-card run, same prompt, same seed.
- **M** ms/token and `gpu_exec_ms` not worse beyond CoV over 3 rounds; CoV reported.
- **A** allocation counter == 0 on plan-hit steps, step count asserted.
- **Q** (lossy cards only, replacing **T**) exact-match rate vs the full model, greedy, 200
  held-out prompts x 128 tokens, plus held-out perplexity delta. Per-card thresholds in §D.
- **G** bytes/token measured (sum of buffer bytes bound per pass, from the driver's own counters)
  and reported next to the §D.7 predicted figure. A lever whose measured bytes disagree with §D.7
  by more than 5% halts the sequence: the byte model is wrong and must be corrected before the
  next card.

### Order

The byte target governs. §A's `Domain` and the two kernel folds are prerequisites for the byte
levers; §B's cards are what keep a k-token pass from paying k times the host cost; §C's cards make
a second architecture data rather than Rust. The sequence interleaves them accordingly.

| # | card | change | gate beyond the battery |
|---|---|---|---|
| 0 | **D0** | Measure sustained GB/s (streaming kernel over the model's own buffers) and read per-tensor codecs from the gguf table (`proxima-gguf/src/types.rs:109-133`) to close §D.1's 125 MB residual. No source change to the decode path. | sustained GB/s with CoV over 5 runs; residual reported to <= 1% or named unexplained |
| 1 | **A1** | `Domain`/`HalfSpace`/`Offset`/`SymbolId` in `map.rs`+`op.rs`; `shape::infer`, `cpu`, and the spec grammar honour it. `Domain::FULL` default keeps every shipped program byte-identical. | property test: a `Domain`-restricted reduce equals the `Select(Greater(Iota,Iota), -inf, ..)` form over randomized extents; `causal_attention.toml` rewritten with a domain infers and evaluates identically |
| 2 | **A2** | `BoundOffset::Symbol` rendered as a uniform read; symbols plumbed through `plan`. | `grep -c 'operands().len() == 9'` == 0; attention `i64::MIN`/`i64::MAX` sentinels == 0; one compiled kernel serves 3 distinct cache lengths (pipeline-compile count == 1) |
| 3 | **D1a** | The `nr1` s-axis fold in `push_packed_row_blocked_body` (`msl.rs:3171-3200`). Prerequisite for L1; no program change. | weight bytes per pass at k in {1,2,4,8} measured: must be flat in k, not linear. **G** |
| 4 | **D2** | Draft + verify as one program: `DraftSource` pipe (prompt-lookup), the argmax/scan/accept chain, `Sampled::accepted()`, k-row `Sample`. Replaces `next_ids = vec![token_id]` (`generate.rs:2503`). | **T** (lossless — exact verification); acceptance histogram per prompt class; `k'` reported per class; **G** vs §D.7 |
| 5 | **B1** | `plan()` pure: `PlanSlots`/`ScheduleBarriers`/`PackUniforms` pipes; `Device::realize`. | `plan()` compiles and runs on a non-macOS target; barrier count per token identical to today's `BARRIERS_EMITTED`; `assert_all_markers` on the plan chain compiles (`primitives.rs:626-649` is the pattern) |
| 6 | **B2** | One `Driver`; `Encode`/`Submit`/`Read`/`Sample` pipes; the nine entry points deleted; `SubmitPolicy`. | `grep -c 'commandBuffer()'` == 1; `grep -c 'pub fn execute'` == 0; every `metal_parity` case passes through the new entry, N asserted |
| 7 | **B3** | Thread-locals to owned state; uniform LRU deleted; `PlanCache` bounded. | `grep -c 'thread_local!' omega/src/metal.rs` == 0; two `Driver`s on two threads produce identical logits; plan-cache hit rate over a prefill+decode alternation reported against the one-entry baseline |
| 8 | **B4** | `Step` FSM; the decode closure (`generate.rs:2278-2545`) replaced. | **A** with N = 100; a walkthrough test driving every legal transition (principle 11) |
| 9 | **A3** | `FusedRegion`, `plan_regions`, `bind_with_regions`, `render_region`, cpu `Fused` arm. Delete `CachedAttention`, both matchers, `render_cached_attention`, the second `bind_plain`. | **the structural test** (§G): the same attention at a different operand layout fuses — the case the eight literal stride tuples (`bind.rs:2451-2474`) reject today. Region count per layer == 1 for attention. `bind_plain` call sites == 1 |
| 10 | **A4** | `RegionSchedule::Rescaled`. | `gpu_exec_ms` <= A3's; key-axis trip count == t, read from the emitted MSL loop bound, not inferred from timing |
| 11 | **A5** | The census's four levers as regions (epilogue, prologue-with-broadcast, RoPE, rmsnorm two-phase) behind `FusionPolicy`. | bound ops/layer reported against the census's 19 baseline, per lever, each toggled independently |
| 12 | **D5a** | `output.weight` Q6_K -> Q4_K (codec selection only). | **Q**: EM >= 0.97, ppl delta <= 0.5%. **G** |
| 13 | **D6a** | f16 KV cache (`DType::Float16` on the cache leaf). | **Q**: EM >= 0.99. **G** at 34 / 2048 / 8192 context |
| 14 | **D4** | Q3_K encoder/decoder in `proxima_gguf::quant`, MSL unpack body, `PackedCodec::Q3K`, lift `UnrepresentableGgmlType` (`proxima-model-interop/src/bind.rs:70-73`). | **Q**: EM >= 0.97, ppl delta <= 0.5%. **G**. Program unchanged: assert the lowered `Vec<Op>` is `PartialEq` to the pre-card program |
| 15 | **D3a** | Row-axis gather on the packed-row body; lift `gather_count == 0` (`msl.rs:1039-1052, 1389`). Prerequisite for L2; no program change. | GB/s on the packed-row path at densities {1.0, 0.5, 0.35}: must fall in proportion to `d`. If it does not, the lever is dispatch-shaped (ROW 180 reproduced on GPU) and D3b does not start |
| 16 | **D3b** | `ScatterBounds::{Fault, Drop}`; the group router + threshold + scan + bounded-compaction selector; group-gathered FFN at fixed `g` of `G`. | **Q**: EM >= 0.95, ppl delta <= 1.0%. **G** vs §D.7. Selector cost reported as a fraction of the FFN it gates |
| 17 | **D5b** | `output.weight` candidate-set gather. | candidate-miss rate <= 0.1%; else fall back to D5c (Q2_K rank + full-precision confirm) |
| 18 | **C1** | `ProgramSpec::lower_into`, `ExtentSpec::Named`, `ModelSpec`; production decode built from `specs/mistral.toml`; `unify_iteration_space` fallback resolution (`shape.rs:224-227`). | the lowered program is `PartialEq` node-for-node with `mistral_single_range_cached_forward_program`'s output, asserted *before* any run. Then **P/B/T/M** |
| 19 | **C2** | `LayerInputs`, `KeyRoots`, `CacheRoots`. | `grep -c 'too_many_arguments' proxima-tensor/src/spec.rs` == 0; config<->builder parity fixture for `ModelSpec` (principle 4) |
| 20 | **C3** | `KV_BUCKET_TOKENS` to `omega`; the `cached_len` leaf deleted. | `grep -c KV_BUCKET_TOKENS proxima-tensor/src` == 0; `grep -c '"cached_len"'` == 0 |
| 21 | **D7** | `ScalarOp::Round`; Q8_0 KV write. Off the critical path — §D.7 meets the target without it. | **Q**: EM >= 0.97. **G** at 2048 / 8192 |

**Tripwires — stop and report, do not proceed:** any node's parity > 1e-4; any byte difference in
the 100 runs; text differs on a lossless card; ms/token or `gpu_exec_ms` worse beyond CoV over 3
rounds; allocation counter nonzero on a plan hit; any grep-count gate nonzero; any gate whose N is
0; measured bytes/token disagreeing with §D.7 by > 5%; a region that fuses on one backend and not
another for the same program (capabilities gate *whether*, never *which*); an EM or perplexity
threshold missed on a lossy card — that card reverts, and §D.7's sensitivity table says whether
the target is still reachable without it.

### What each constraint changed, and what it killed

- **The byte target itself.** It reordered everything: §A and §B are prerequisites and margin, not
  the plan. **Abandoned:** the ordering in which region fusion led and bytes followed — it lands
  616 -> 264 dispatches against a 10.4 ms floor and cannot reach 3.5 ms by any amount of it.
  Also abandoned: crediting L1 and L2 in the byte table before the two kernel folds, which is what
  `byte-levers-probe.md:8-11` and `:19-21` show would have been a paper win worth zero.
- **no_std + alloc tier (P3).** Forced `Domain`'s dynamic offset to be `SymbolId` into a
  caller-supplied `&[u64]` rather than a value read from a device buffer. **Abandoned:** keeping
  today's ninth rank-0 operand and giving it a name — the smallest possible diff, and it requires
  a buffer read (I/O) to evaluate a loop bound, which does not exist at tier 3. The same
  constraint removes the 8/9 discriminator, a defect it was not aimed at.
- **Lock-free (P21).** Forced the six `thread_local! RefCell` blocks into fields of `Device` and
  `Plan`. **Abandoned:** a process-global `OnceLock<Device>` with a lock around the pipeline
  cache — two drivers on two cores then serialize on every lookup, and a thread_local is a
  missing owner, not a missing lock.
- **Reuse-first (P1).** Forced the fused kind to generalize `ComposedBody` (`bind.rs:176`) into
  `FusedRegion` rather than be a better-shaped attention op. **Abandoned:**
  `BoundOpKind::FlashAttention { operands, layouts, domain }` — it fixed audit findings 2 and 3
  and was entirely defensible, and it fixes finding 1 for nothing: still a macro-op that only
  fuses attention, so a differently-laid-out program still falls back and none of the census's
  four levers get anything. Writing the paragraph defending it is what killed it.
- **The pipe question.** Forced the driver into `Encode.and_then(Submit).and_then(Read)
  .and_then(Sample)`. **Abandoned (three):** an `Executor` trait with
  `execute`/`execute_named`/`execute_placed`/`execute_timed` — four entry points wearing a trait;
  a `Timed<P>` wrapper — per-op timing is `SubmitPolicy::PerOpBuffer`, one field, and the wrapped
  call site is identical to the unwrapped one; a `PlanBuilder` — `PlanSlots.and_then(
  ScheduleBarriers).and_then(PackUniforms)` is the plan builder, written as the algebra.
  The same question, asked twice with opposite answers, produced §D's sharpest result: the
  **draft source is a pipe** (once per pass, host, zero weight bytes) and the **elision selector
  is not** (once per layer, 32 device crossings per token). **Abandoned:** a host-side
  `RowSelector` pipe, which is the shape the landed ROW 180/181 probe already has and which
  measured dispatch-bound.
- **P4 config.** Forced `fuse_cached_attention: bool` (`bind.rs:2639`) into
  `FusionCapabilities { schedules, max_region_ops }`, and `quarantine_broadcast_operands`'s
  hard-wired refusal (`bind.rs:935-973`) into `FusionPolicy` data. **Abandoned:** a second bool
  `fuse_regions` beside the first — the same information destruction, twice.
- **P15 do-the-correct-thing.** The counting global allocator does not exist (grep over
  `proxima-primitives/src`, `proxima-tensor/src`, `proxima-test` for `GlobalAlloc` returned
  nothing), so §B.6's gate is unprovable without it and it is in this work, not a follow-up.
## F. What is NOT a pipe, and why each is justified

Applying the central claim as a lint, per the four forms named by `In`/`Out`
(`primitives.rs:26-40`):

1. **`Step`, `Ready`, `Encoded`, `InFlight`, `Sampled`, `Halt`** — a sans-IO FSM. Not a pipe:
   `Pipe::call` is one `In -> Result<Out, Err>` step, and these transitions take different
   argument types (`&[TokenId]`, `SubmitTicket`, `LogitsView`) and are driven an arbitrary number
   of times by a caller who owns the loop. Precedent, verbatim, for exactly this argument:
   `proxima-primitives/src/pipe/sans_io.rs:41-52` ("there is no `In` shape that expresses 'attempt
   progress again against what's already buffered'"). Structurally justified.
2. **IR data — `Domain`, `HalfSpace`, `Offset`, `BoundDomain`, `BoundHalfSpace`, `BoundOffset`,
   `FusedRegion`, `RegionSchedule`, `RegionPlan`, `RegionSpan`.** Not computation; these are the
   *values* pipes carry. The pipe question does not apply to a payload. Each earns its place by
   the second binary question: with `Domain`, a caller can write a banded reduce whose iteration
   space is triangular — impossible before, at any spelling (§A.3). With `FusedRegion`, a caller
   can emit one kernel for a chain of nests sharing a domain — today only `CachedAttention` can,
   and only for attention.
3. **Config — `FusionCapabilities`, `ScheduleSet`, `FusionPolicy`, `DriverConfig`,
   `SubmitPolicy`, `SamplePolicy`, `ModelSpec`, `StageSpec`.** Principle 4 makes these
   first-class by construction. `ScheduleSet` in particular exists because a bool destroyed the
   distinction between two capabilities (§A.4).
4. **Plan-time results — `SlotPlan`, `BarrierSchedule`, `Barrier`, `UniformTable`,
   `Residency`, `PlanCache`.** Data. The *computations* that produce them are pipes
   (`PlanSlots`, `ScheduleBarriers`, `PackUniforms`), composed with `AndThen`.
5. **`Device`, `Driver`.** A pipe's `&self` state has to live somewhere; `Submit` and `Read`
   borrow it. Justified as the single device edge — and the count is the check: exactly one
   `commandBuffer()` call site after CARD B2, versus ten today.
6. **Newtype ids — `TokenId`, `SlotId`, `SymbolId`, `StepIndex`, `SubmitTicket`, `PlanKey`,
   `NodeId`.** Required by P11's compile-time-correctness clause; `NodeId` is the shipped
   precedent (`op.rs:26-28`).
7. **Typed records — `LayerInputs`, `AttentionWeights`, `FeedForwardWeights`, `NormWeights`,
   `RopeTables`, `CacheInputs`, `KeyRoots`, `CacheRoots`.** Data, replacing positional tuples and
   23-argument signatures. `KeyRoots` is the only one that adds capability beyond naming: it turns
   15+ positional destructures into exhaustive matches, so a new rotary layout is a compile error
   at every site instead of a silent mis-read.

Nothing in this design is a wrapper whose call site reads identically to the thing it wraps. The
three types that would have been (`Timed<P>`, `Executor`, `PlanBuilder`) are named in §E as
abandoned, with the call site that killed each.

---

## G. Worked example, which is the test (principle 17)

The example is the structural claim in §A: **any program with the structure fuses.** It is the
gate for CARD A3 and it is a test no current code can pass, because `bind.rs:2465-2474` compares
operand strides against eight literals.

```rust
/// Two programs computing the SAME banded softmax-weighted reduction, differing
/// only in memory layout: the first stores keys head-major, the second stores
/// them position-major (a transposed cache). Both must produce ONE fused region
/// with identical results, because the fusion rule reads strides from `Layout`
/// and never compares them to a literal.
#[proxima::test]
#[case::head_major(KeyLayout::HeadMajor)]
#[case::position_major(KeyLayout::PositionMajor)]
async fn a_banded_softmax_reduction_fuses_at_any_operand_layout(#[case] layout: KeyLayout) {
    // arrange: build the 8-node chain of §A.1 with a causal Domain, at the
    // given key layout. Real openchat-3.5 dimensions: 8 kv heads, group 4,
    // head_dim 128, 34 cached positions, 1 new token.
    let program = banded_softmax_program(layout);
    let shapes = shape::infer(&program, &[SEQUENCE, MERGED_LENGTH]).expect("infers");

    // act
    let plan = plan_regions(
        &program, &shapes, &[OUTPUT],
        FusionCapabilities { schedules: ScheduleSet::of(&[Nest, Rescaled]), max_region_ops: None },
        FusionPolicy::default(),
    ).expect("regions plan");
    let bound = bind_with_regions(&program, &shapes, &[OUTPUT], &plan).expect("binds");

    // assert: one region, containing all eight nodes, on the rescaled schedule
    assert_eq!(plan.spans.len(), 1, "the whole chain is one region at any layout");
    let [BoundOp { kind: BoundOpKind::Fused(region), .. }] = &bound[..] else {
        panic!("expected exactly one fused op, got {} ops", bound.len());
    };
    assert_eq!(region.ops.len(), 8, "every node of the chain is a member");
    assert!(matches!(region.schedule, RegionSchedule::Rescaled { .. }));

    // assert: the band is a loop bound, not a guard — t trips, not 2t
    assert_eq!(region.domain.constraints.len(), 1, "one causal half-space");

    // assert: parity against the CPU oracle at 1e-4, and against the SAME
    // program bound with ScheduleSet::EMPTY (the unfused chain) bit for bit
    // in structure and to 1e-4 in value.
    let fused = cpu::evaluate(&program, &symbols, &blocks, &[OUTPUT]).expect("fused runs");
    let plain = cpu::evaluate_unfused(&program, &symbols, &blocks, &[OUTPUT]).expect("plain runs");
    assert_close(&fused, &plain, 1e-4);
}
```

Two properties this asserts that no test on `main` can: the region count is layout-independent
(the eight literal stride tuples make that false today), and the fused result equals the unfused
result computed by the same evaluator (today the CPU `CachedAttention` arm at `cpu.rs:4830` is a
second hand-written implementation, so a parity test compares two implementations rather than one
rule against its own unfused form).

The second worked example is §B.6's allocation test, which is the gate for CARD B4 and asserts
both zero allocations and N == 100 steps, because a zero-step run and a zero-allocation run are
indistinguishable by the counter alone.
