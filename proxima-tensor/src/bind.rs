//! Tensor operations with their addressing resolved — the seam between the
//! symbolic program and any executor.
//!
//! There is no second intermediate representation here. An [`BoundOp`] is an
//! [`Op`] whose addressing has been worked out against one call's symbol
//! bindings — the same elementwise/reduce shape, with a resolved [`Layout`]
//! standing in for a symbolic [`IndexMap`] and resolved iteration extents
//! standing in for a symbolic [`Extent`](crate::Extent) list. Nothing here is a competing
//! tree: [`bind`] still walks the program left to right, one `Op` at a
//! time, and every `BoundOp` it emits corresponds to exactly one `Op` that
//! actually computes (`Op::Input` never does — it is where data enters,
//! not where anything is derived).
//!
//! [`BoundOpBuilder`] does two things in one pass rather than two passes, and
//! that is a property of the algebra, not a corner cut: whether an
//! elementwise operand's layout is ever worth resolving on its own depends
//! entirely on whether the reduce consuming it can absorb that elementwise
//! op's body directly (the fusion decision). Splitting "resolve layout" from
//! "fuse" into separate stages would mean resolving a `Layout` for every
//! elementwise op and then throwing most of them away — real work with no
//! payoff — so the one `BoundOpBuilder` stage decides both at once, exactly the
//! way [`shape::ShapeTable`] decides shape *and* validity in one pass rather
//! than validating first and shaping second.
//!
//! Like [`shape::ShapeTable`], [`BoundOpBuilder`] is a sans-IO push state
//! machine: a program can arrive a step at a time, and op building must not
//! require the whole thing in hand. [`bind`] is the batch driver over it.
//! What *does* require the whole program in hand is liveness
//! ([`live::annotate`]) — computed once, upstream, and handed to
//! `BoundOpBuilder` as a plain kill-flag list it never has to guess at;
//! `BoundOpBuilder::new` takes that list up front for exactly this reason,
//! streamed or not.
//!
//! `BoundOpBuilder` also implements [`Pipe`]
//! (`In = (Op, Shapes)`, `Out = `[`ReadyBatch`]) — the same state machine,
//! not a second type wrapping it. `ReadyBatch` is a fixed-capacity, no-alloc
//! batch rather than a `Vec`: a single `push` readies at most
//! `READY_BATCH_CAPACITY` `BoundOp`s (see `push`'s own doc), so the
//! composition boundary between this stage and [`crate::cpu::Interpreter`]
//! never pays a heap allocation per `Op` pushed through the chain.
//! [`Pipe::call`] takes `&self`, so `held` below is
//! a `RefCell` and the node position a `Cell`, the same interior-mutability
//! idiom [`shape::ShapeTable`] uses for its own `Pipe` impl.
//!
//! The one optimization decided here: when a reduce's operand is an
//! elementwise op whose last use is that reduce (exact liveness, from
//! [`live::annotate`]), the elementwise op is never materialized — its body
//! is composed directly into the reduce's [`BoundOp`], which is the difference
//! between an O(extents) buffer and an O(iteration space) one for something
//! like matmul. An elementwise op whose last use is anything else (another
//! elementwise op, a non-fusable reduce, or nothing — a requested output or
//! dead code) materializes as its own `BoundOp`, emitted the moment that use is
//! seen (or, for dead code and outputs never referenced again, when
//! [`BoundOpBuilder::finish`] flushes it).

use alloc::collections::BTreeMap;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::{Cell, RefCell};
use core::future::Future;

use arrayvec::ArrayVec;
use proxima_primitives::pipe::Pipe;
use smallvec::SmallVec;

use crate::dtype::DType;
use crate::error::TensorError;
#[cfg(feature = "instrument")]
use crate::instrument;
use crate::live;
#[cfg(feature = "cached-attention-streaming")]
use crate::map;
use crate::map::{AxisIndex, AxisTerm, IndexMap, IndexPattern};
use crate::numeric::{NumericPolicy, NumericRewrite, admit};
use crate::op::{Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp};
use crate::shape::{self, Shapes};
#[cfg(feature = "instrument")]
use proxima_telemetry::debug;

/// Inline capacity for one bound op's per-iteration-axis buffers (`Layout`
/// strides, a reduce's surviving `output_axes`). No rank bound is stated or
/// enforced anywhere in this crate (`iter_rank` is a plain runtime `u16`);
/// the highest rank this crate's own tests and CPU evaluator exercise today
/// is 3 (`cpu.rs`'s `matmul_program`-style fixtures). 4 gives one axis of
/// headroom (e.g. a batch/heads/seq/dim attention iteration space) while
/// staying inline; `SmallVec` spills to the heap past this instead of
/// truncating, so a wider program still binds correctly.
pub use crate::sized::MAX_INLINE_RANK;

/// One operand's address into its own buffer, expressed directly in an
/// [`BoundOp`]'s iteration-axis space: `strides[axis]` is how far the linear
/// offset moves per step of iteration axis `axis`. An axis this operand
/// never varies along (broadcast) simply has stride 0.
///
/// This is the tensor-domain notion of memory layout (the same concept
/// `torch.Tensor.stride()` names) — the resolved counterpart of an
/// [`IndexPattern`], the same category as `IndexPattern` itself, not a rival
/// record kind, so it stays a small value type with its own accessors
/// rather than being folded away into bare tuple fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub base: i64,
    pub strides: SmallVec<[i64; MAX_INLINE_RANK]>,
}

impl Layout {
    #[must_use]
    pub fn offset_of(&self, coordinate: &[u64]) -> i64 {
        self.base
            + coordinate
                .iter()
                .zip(&self.strides)
                .map(|(index, stride)| stride * (*index as i64))
                .sum::<i64>()
    }

    #[must_use]
    pub fn stride(&self, axis: u16) -> i64 {
        self.strides.get(axis as usize).copied().unwrap_or(0)
    }
}

/// The extra addressing a gathered operand needs on top of its [`Layout`]:
/// where to fetch the index from, and how to turn a fetched index into an
/// offset once it lands — the same shape an embedding table lookup needs.
/// `element_stride` is the operand's own per-element stride along the
/// gathered axis (the table's row stride, for an embedding lookup) — an
/// executor's runtime read offset is
/// `layout.offset_of(coord) + fetched_index * element_stride`. `extent` is
/// the gathered axis's size, carried so an executor can reject an
/// out-of-range fetched index instead of reading past the buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lookup {
    pub indices: NodeId,
    pub index_layout: Layout,
    pub element_stride: i64,
    pub extent: u64,
}

type BoundOperands = Vec<(NodeId, Layout, Option<Lookup>)>;

/// [`attention_score_sources`]'s own return shape: the rotary even/odd
/// query+key sources, plus `Some((query_pass_grouped, key_pass))` only when
/// qwen35's partial-rotary chain matched.
#[cfg(feature = "cached-attention-streaming")]
type AttentionScoreSources = (NodeId, NodeId, NodeId, NodeId, Option<(NodeId, NodeId)>);

/// Capacity for one [`BoundOpBuilder::push`] call's ready batch. An
/// elementwise push materializes at most one [`BoundOp`] per operand that
/// fails to fuse, bounded by [`ScalarOp::arity`]'s current maximum
/// (`Select`, 3); a reduce push materializes at most one held predecessor
/// plus the reduce's own op (2). `push` and `materialize_if_held` return
/// [`TensorError::NotLowerable`] rather than overflow this if a future
/// higher-arity `ScalarOp` variant is ever added.
pub use crate::sized::READY_BATCH_CAPACITY;

/// The batch [`BoundOpBuilder::push`] readies for one `Op`: [`Pipe::Out`]
/// for [`BoundOpBuilder`] and, by the composition law, [`Pipe::In`] for
/// [`crate::cpu::Interpreter`]. Fixed-capacity and stack-resident rather
/// than heap-backed like `BoundOperands` above — one `Vec` allocation per
/// `Op` pushed through the chain was the actual cost this replaces, for a
/// container that only ever holds 0 to `READY_BATCH_CAPACITY` items.
pub type ReadyBatch = ArrayVec<BoundOp, READY_BATCH_CAPACITY>;

/// One argument to a [`BodyStep`]: a fresh read of one of the [`BoundOp`]'s
/// own physical operands, or the result of an earlier step in the same
/// body. Backwards-only, the same rule [`crate::op::Op`]'s own module doc
/// states for a whole program's [`NodeId`] references, recreated here at the
/// scalar granularity a fused body composes at — a plain index into a side
/// table (`ComposedBody::steps`), never a `Box<dyn>` recursive tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepArg {
    Operand(u16),
    Step(u16),
}

/// One scalar computation inside a [`ComposedBody`]: apply `op` to `args`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BodyStep {
    pub op: ScalarOp,
    pub args: Vec<StepArg>,
}

/// A fused elementwise body: an ordered sequence of [`BodyStep`]s whose last
/// entry is the body's result. An unfused op is the one-step case
/// ([`ComposedBody::leaf`]), so every `BoundOp` carries exactly one
/// `ComposedBody` whether or not it absorbed anything — an executor has one
/// shape to walk regardless of how many elementwise ops a chain fused away.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComposedBody {
    pub steps: Vec<BodyStep>,
}

impl ComposedBody {
    /// One scalar op applied directly to consecutive operand slots — the
    /// shape every body had before fusion existed.
    #[must_use]
    pub fn leaf(op: ScalarOp) -> Self {
        let args = (0..op.arity() as u16).map(StepArg::Operand).collect();
        Self {
            steps: vec![BodyStep { op, args }],
        }
    }
}

/// One tensor op with its addressing resolved: which buffers to read, at
/// what layout, combined by which scalar op — and, for a reduce, how the
/// reduction is shaped. Carries only what an executor needs, nothing about
/// *how* to execute it: [`cpu`](crate::cpu) interprets an `BoundOp` with nested
/// loops; a GPU backend could instead emit kernel source from the same
/// descriptor. Neither backend's shape belongs in this module.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundOp {
    pub node: NodeId,
    /// This node's own element type, carried straight from the [`Op`] it
    /// was built from ([`Op::dtype`]) — an executor spells its buffers and
    /// scratch declarations from this rather than assuming `f32`, which is
    /// what lets a GPU backend emit a narrower kernel for a narrower node
    /// without a second, parallel emitter.
    pub dtype: DType,
    /// The iteration-space extents this op's loop walks. Equal to the
    /// output shape for an [`BoundOpKind::Elementwise`] or a `Keep::Scan` reduce
    /// (a scan drops no axis); wider than the output shape for a
    /// `Keep::Reduce` reduce, which walks the full pre-reduction space.
    pub extents: Vec<u64>,
    pub kind: BoundOpKind,
}

/// The resolved counterpart of [`Op`]'s `Elementwise`/`Reduce` variants —
/// `Input` has no counterpart because a leaf never computes anything to
/// resolve.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundOpKind {
    /// One backend-neutral cached-attention step. The operands are the
    /// already-bound Q/K/V sources; CPU and GPU own only the kernel body.
    ///
    /// `operands` carries exactly eight Q/K/V sources when neither range is
    /// bucketed, and a NINTH, rank-0 runtime scalar in two shapes that share
    /// this one slot rather than each minting its own
    /// (`cached_key_rows != 0` is the discriminator both an executor and
    /// `entry_name` read to tell them apart — no tenth operand, because the
    /// two shapes are never both live on one op):
    ///
    /// - `cached_key_rows == 0` (`cached_attention_single_range_candidates`):
    ///   the whole context is folded into the "new" slot, and the ninth
    ///   operand is the real `cached_len` `causal_mask_merged`'s band
    ///   depends on — a VALUE, not a shape, because `kv-capacity-bucket`
    ///   widens the key extent past the merged length and a shape
    ///   difference alone would silently overstate it. `new_upper_inclusive`
    ///   is unused filler in this case; every executor reads the real bound
    ///   from `operands[8]` at run time instead.
    /// - `cached_key_rows != 0` (`cached_attention_candidates`'s own
    ///   two-range fusion): the cached and new ranges are still separate,
    ///   and the ninth operand is the CACHED range's own live row count —
    ///   a caller may round `cached_key_rows` up to a `kv_extent` bucket
    ///   boundary so one compiled `Plan` serves every real length inside
    ///   it, and every executor substitutes this runtime count for the
    ///   compiled `cached_key_rows` wherever it addresses or bounds the
    ///   cached range, excluding the bucket's own zero-padded tail rows
    ///   without a mask node anywhere in the graph
    ///   (`cached_attention_candidates`'s own doc on why no such node
    ///   exists). `cached_lower_inclusive`/`new_upper_inclusive` are
    ///   unaffected — this bound is query-independent, unlike the
    ///   single-range case's causal band.
    ///
    /// `rotary_dim` is the RoPE-rotated width per head (`head_dim` when
    /// every column rotates — mistral/openchat/qwen3 today); when
    /// `rotary_dim < head_dim`, `operands` carries THREE more trailing
    /// entries beyond the base eight/nine described above — `pass_query`,
    /// `pass_cached_key`, `pass_new_key` — one un-rotated, non-split plane
    /// per side, laid out `[pass_query, pass_cached_key, pass_new_key]`
    /// immediately after the optional `cached_len` slot (so `operands.len()`
    /// is 8, 9, 11, or 12; the extra three are absent whenever `rotary_dim
    /// == head_dim`, which is every existing caller — qwen35's partial-rotary
    /// dense attention (`rotary_dim` 64 of `head_dim` 256`,
    /// `proxima-model-interop/src/qwen35.rs:37-38,142,179,182`) is the one
    /// caller that needs the wider shape, per `docs/discipline.md` ROW 556's
    /// residual). The pass plane contributes one extra additive term to the
    /// score (`score = rotary_dot * scale + pass_dot * scale`, `spec.rs`'s
    /// own `score_cached`/`score_new` — `q_pass . k_pass`, no even/odd split
    /// because the pass plane is never rotated) and is read over its own
    /// `head_dim - rotary_dim` width; `pass_cached_key`/`pass_new_key` share
    /// `cached_key_rows`/`new_key_rows` with the rotary planes. The value
    /// planes are unaffected — V is never rotated, so `cached_value`/
    /// `new_value` already carry the full `head_dim` width regardless of
    /// `rotary_dim`.
    CachedAttention {
        operands: BoundOperands,
        query_rows: u64,
        cached_key_rows: u64,
        new_key_rows: u64,
        kv_heads: u64,
        query_groups: u64,
        head_dim: u64,
        rotary_dim: u64,
        scale: f32,
        cached_lower_inclusive: i64,
        new_upper_inclusive: i64,
    },
    /// One backend-neutral gated-delta-net recurrence step
    /// ([`crate::spec::append_qwen35_delta_net_step`]'s own ~12-op chain,
    /// collapsed): `operands` is exactly `[query, key, value, gate, beta,
    /// state_in]`, `query`/`key` bound PRE-[`crate::spec::repeat_kv_heads`]
    /// (the matcher walks past that op's two broadcast multiplies, the same
    /// move [`BoundOpKind::Reduce::epilogue_broadcast_axes`] already makes
    /// for a broadcast-reduce epilogue, and on the real program past its own
    /// decode-squeeze reduce first — `crate::bind`'s own
    /// `gdn_unwrap_decode_squeeze` doc) — an executor mod-broadcasts
    /// `key_index % kv_heads` itself, exactly llama.cpp's own fused Metal
    /// kernel (`gated_delta_net.metal:33-34`). `value`/`gate`/`beta`/
    /// `state_in` are natural row-major, last axis unit-stride, over the
    /// consumer's own un-permuted `j{head}`/`{head}` read (dim, where
    /// present, SLOWEST). `query`/`key` instead carry their own explicit
    /// per-axis strides (`query_key_head_stride`/`query_key_dim_stride`)
    /// rather than a single fixed axis order: whether a `repeat_kv_heads`
    /// broadcast actually sat above them changes which of `kv_heads`/
    /// `head_k_dim` is the program's own fast axis (`crate::bind`'s own
    /// `gated_delta_net_candidates` doc), and the executor
    /// (`crate::gdn::run_gdn_prefill_scan`) reads by stride instead of
    /// assuming one. `n_tokens == 1` is this slice's only supported
    /// shape (decode); an `n_tokens > 1` bind is out of scope until the
    /// M-token prefill slice lands. `kv_heads` may be less than `num_v_heads`
    /// (`num_v_heads = kv_heads * group`) -- the real qwen35moe GQA shape,
    /// `query`/`key` still bound at `kv_heads` and `value`/`gate`/`beta`/
    /// `state_in` at `num_v_heads`; `group` itself is not a separate field,
    /// it is exactly `num_v_heads / kv_heads`.
    ///
    /// `state_out` is this op's own SECOND output (ROW 547,
    /// `docs/discipline.md`): the delta step's own updated recurrence state,
    /// shape identical to the `state_in` operand's leaf. Produced
    /// functionally by the same dispatch that produces `node`'s own value
    /// output -- no aliasing contract, no in-place write into `state_in`'s
    /// buffer. A backend writes it to whatever buffer `state_out` names, the
    /// same as any other resolved node's output.
    GatedDeltaNet {
        operands: BoundOperands,
        n_tokens: u64,
        kv_heads: u64,
        num_v_heads: u64,
        head_k_dim: u64,
        head_v_dim: u64,
        /// Elements between consecutive `kv_head` values in `query`/`key`.
        query_key_head_stride: u64,
        /// Elements between consecutive `head_k_dim` values in `query`/`key`.
        query_key_dim_stride: u64,
        inv_sqrt_key_dim: f32,
        state_out: NodeId,
    },
    Elementwise {
        body: ComposedBody,
        operands: BoundOperands,
    },
    Reduce {
        /// The per-step combine of `operands` before reducing: the fused
        /// elementwise chain's own body when one or more were absorbed, a
        /// one-step [`ScalarOp::Identity`] body otherwise. Distinct from
        /// `reduce_op`, which is the reduce's own accumulation.
        element_body: ComposedBody,
        reduce_op: ScalarOp,
        init: ReduceInit,
        keep: Keep,
        operands: BoundOperands,
        /// Iteration axes that survive to the output, in the reduce's
        /// `out_map` operand-axis order. The last entry (if any) is the
        /// innermost loop. For a scatter this naturally excludes the
        /// scattered axis (its position is data-dependent, so it can never
        /// be a pure projection — see `pure_projection_axes` (private to
        /// this module)).
        output_axes: SmallVec<[u16; MAX_INLINE_RANK]>,
        out_layout: Layout,
        /// `Some` exactly when `out_map` was [`IndexMap::Computed`] — a
        /// scatter. Reuses [`Lookup`] itself (the read-side gather's own
        /// addressing record): the destination axis is fetched from
        /// `indices` at `index_layout`, bounds-checked against `extent`,
        /// then scaled by `element_stride` and added to `out_layout`'s own
        /// offset — the exact mirror of how a gathered *operand*'s `Lookup`
        /// already contributes to a *read* offset in the CPU interpreter's
        /// generic reduce path (`crate::cpu`'s own `run_reduce`). `None` for
        /// an ordinary (affine) reduce/scan.
        out_scatter: Option<Lookup>,
        /// An elementwise chain fused onto the OUTPUT side of this fold —
        /// `element_body`'s mirror image. `element_body` combines `operands`
        /// once per PRE-reduction step, before the fold; `epilogue_body` runs
        /// once per OUTPUT element, after it, reading `epilogue_operands`
        /// (addressed in output-axis order, the same order `output_axes`
        /// itself uses) plus one more implicit argument: this fold's own
        /// just-computed result at that output position. That result is
        /// [`StepArg::Operand`]`(epilogue_operands.len())` — one slot past
        /// the real operands, so `Identity` applied to the sole slot of an
        /// EMPTY `epilogue_operands` (index `0`) already reads it with no
        /// special case, which is exactly [`ComposedBody::leaf`]'s own shape
        /// for [`ScalarOp::Identity`]. "No epilogue" is that leaf body over
        /// zero real operands — not an `Option` — the same convention
        /// `element_body` already uses for "nothing fused into the prologue
        /// either" (`build_reduce_op`'s own default).
        ///
        /// This module's own fusion rule (`reduce-epilogue-fusion`,
        /// default-off): an `Elementwise` consumer of this fold absorbs into
        /// `epilogue_body`/`epilogue_operands` — and the fold's own `node`
        /// is REPLACED by the consumer's `NodeId` (the fused op now answers
        /// for the consumer's own identity, which is exactly what "a
        /// required-output consumer still fuses" needs) — when: (a) the
        /// consumer's own iteration space equals this fold's `output_axes`
        /// shape, with the fold's output read through a genuine identity
        /// projection (every axis, offset `0`, coefficient `1` — see
        /// `is_identity_projection`) and every OTHER consumer operand read
        /// through an identity-or-broadcast projection (the same predicate;
        /// broadcast is a projection over fewer axes, still admitted by
        /// `is_identity_projection`'s own per-axis walk); (b) this fold's
        /// output has no other consumer anywhere in the program and is not
        /// itself a requested output (so retiring it here loses nothing);
        /// (c) `out_scatter` is `None` (a scatter's destination is
        /// data-dependent, never a plain identity projection a consumer
        /// could read through). A backend with no epilogue renderer rejects
        /// at bind time (the same capability path `bind_with_fusion` already
        /// runs `fuse_cached_attention` through) rather than silently
        /// dropping the extra work.
        epilogue_body: ComposedBody,
        /// The epilogue's own real operands, in the SAME output-axis-order
        /// coordinate space `out_layout`/`output_axes` write in — see
        /// `epilogue_body`'s own doc for why the fold's result is an
        /// implicit, un-listed argument rather than an entry here.
        epilogue_operands: BoundOperands,
        /// Axes of `extents` this fold's own `output_axes` excludes but
        /// `epilogue_body` still walks — the "broadcast-reduce" epilogue
        /// shape an RMSNorm-style `x * inv_rms` tail needs, where the
        /// CONSUMER re-broadcasts the fold's scalar result back over the
        /// very axis the fold reduced away. Empty for the default identity
        /// epilogue and for the pre-existing PLAIN epilogue shape (a
        /// bias/residual/gate tail whose own iteration space already equals
        /// `output_axes`'s shape with no further axis to re-broadcast over):
        /// `output_axes` alone already names that epilogue's iteration
        /// space, so `node_output_len`/`apply_reduce_epilogue`
        /// (`crate::cpu`) walk it unchanged when this is empty. When
        /// non-empty, both instead walk `output_axes` UNION
        /// `epilogue_broadcast_axes` (i.e. `extents` in full), addressing
        /// the fold's own already-computed scalar through a genuine
        /// broadcast (stride `0`) on every axis named here. A backend with
        /// no broadcast-reduce renderer must reject a non-empty value at
        /// bind time rather than emit a wrong kernel (`omega::msl`'s own
        /// `EmitError` does this for Metal).
        epilogue_broadcast_axes: SmallVec<[u16; MAX_INLINE_RANK]>,
    },
    /// The resolved counterpart of [`Op::Iota`]: no operands, no body — an
    /// executor derives every output value straight from its own position
    /// in `BoundOp::extents`, which is why this variant carries no fields of
    /// its own.
    Iota,
    /// The resolved counterpart of [`Op::Constant`]: no operands, no body,
    /// and unlike [`BoundOpKind::Iota`] not even a dependence on position —
    /// an executor writes `value` to every element of `BoundOp::extents`.
    Constant { value: f32 },
}

impl BoundOpKind {
    /// This variant's own discriminant name, used by every backend renderer
    /// (`omega::cuda`, `omega::wgsl`, `omega::wgpu_driver`) to report
    /// `EmitError::RenderKindMismatch { expected, found }` — the one place a
    /// backend needs to name what it actually got instead of what it matched
    /// on. Kept here, on the type itself, rather than restated per backend
    /// module.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            BoundOpKind::CachedAttention { .. } => "cached_attention",
            BoundOpKind::GatedDeltaNet { .. } => "gated_delta_net",
            BoundOpKind::Elementwise { .. } => "elementwise",
            BoundOpKind::Reduce {
                keep: Keep::Reduce, ..
            } => "keep::reduce fold",
            BoundOpKind::Reduce {
                keep: Keep::Scan, ..
            } => "keep::scan fold",
            BoundOpKind::Iota => "iota",
            BoundOpKind::Constant { .. } => "constant",
        }
    }
}

/// A fused body with zero steps — [`BoundOp::element_body`]'s answer for
/// [`BoundOpKind::Iota`], which has no combining body at all. Every real
/// caller (`cpu::run_elementwise`/`run_reduce`/`run_scan`,
/// `omega`'s renderers) only reaches `element_body()` from inside a branch
/// already matched on `Elementwise`/`Reduce`, so this is never actually
/// read; it exists so the accessor stays total over every `BoundOpKind`
/// instead of panicking on the one variant that has nothing to answer with.
static EMPTY_BODY: ComposedBody = ComposedBody { steps: Vec::new() };

impl BoundOp {
    #[must_use]
    pub fn operands(&self) -> &[(NodeId, Layout, Option<Lookup>)] {
        match &self.kind {
            BoundOpKind::CachedAttention { operands, .. }
            | BoundOpKind::GatedDeltaNet { operands, .. }
            | BoundOpKind::Elementwise { operands, .. }
            | BoundOpKind::Reduce { operands, .. } => operands,
            BoundOpKind::Iota | BoundOpKind::Constant { .. } => &[],
        }
    }

    /// Every node a liveness pass must count as READ by this op: [`Self::operands`]
    /// (what an executor's compute step reads) plus, for a
    /// [`BoundOpKind::Reduce`], `epilogue_operands` too. Distinct from
    /// `operands()` on purpose — `run_reduce`/`run_elementwise`'s own operand
    /// tables must stay exactly the compute-step operands, nothing more, so
    /// this stays a SEPARATE accessor rather than folding epilogue operands
    /// into `operands()` itself and silently widening every existing caller's
    /// operand table by one entry it never asked for. A liveness pass that
    /// calls `operands()` alone treats an epilogue-only reader as no reader
    /// at all: `dead_resolved_nodes`/`consumed_by_resolved_nodes` would mark
    /// a reduce whose sole use is another fold's `epilogue_operands` entry as
    /// dead weight to skip, and `node_retirement` would free its buffer at
    /// its OWN position instead of the epilogue's later one — the exact
    /// `operand buffer missing at evaluation time` a real two-quantized-layer
    /// program (`cpu::tests::evaluate_quantized_two_layers_does_not_
    /// underflow_live_now`) hit the moment the CPU evaluator started
    /// actually reading `epilogue_operands` (`cpu::apply_reduce_epilogue`).
    pub fn all_read_sources(&self) -> impl Iterator<Item = &(NodeId, Layout, Option<Lookup>)> {
        let epilogue: &[(NodeId, Layout, Option<Lookup>)] = match &self.kind {
            BoundOpKind::Reduce {
                epilogue_operands, ..
            } => epilogue_operands,
            BoundOpKind::CachedAttention { .. }
            | BoundOpKind::GatedDeltaNet { .. }
            | BoundOpKind::Elementwise { .. }
            | BoundOpKind::Iota
            | BoundOpKind::Constant { .. } => &[],
        };
        self.operands().iter().chain(epilogue.iter())
    }

    /// The composed body applied per step to build one combined value from
    /// `operands()`, before any reduction: an elementwise op's own
    /// (possibly fused) body, or a fused reduce's absorbed body (a one-step
    /// `Identity` body if nothing fused). See `EMPTY_BODY`'s doc for the
    /// [`BoundOpKind::Iota`] case.
    #[must_use]
    pub fn element_body(&self) -> &ComposedBody {
        match &self.kind {
            BoundOpKind::CachedAttention { .. } | BoundOpKind::GatedDeltaNet { .. } => &EMPTY_BODY,
            BoundOpKind::Elementwise { body, .. } => body,
            BoundOpKind::Reduce { element_body, .. } => element_body,
            BoundOpKind::Iota | BoundOpKind::Constant { .. } => &EMPTY_BODY,
        }
    }

    /// Splits this op along its outermost output-iteration axis into
    /// `parts` contiguous chunks so each chunk can execute independently:
    /// for an `Elementwise` op that axis is `extents[0]`; for a
    /// `Keep::Reduce` reduce it is the first entry of `output_axes` (the
    /// reduce's outermost surviving axis, per that field's own doc). This
    /// split is backend-neutral on purpose: a CPU driver runs chunks on
    /// worker threads, a GPU backend would tile the identical axis into
    /// threadgroups — neither belongs in this module, only the geometry
    /// does.
    ///
    /// Returns `None` when splitting is not sound or not useful:
    /// - a scalar reduction (`output_axes` is empty — nothing to split, the
    ///   whole op is one accumulator),
    /// - a `Keep::Scan` scan (each step reads the previous step's output,
    ///   so the extent is a sequential dependency, not parallel work),
    /// - the split axis's extent is smaller than `parts` (some chunk would
    ///   be empty),
    /// - `parts < 2` (nothing to split into).
    ///
    /// Chunk `k`'s output occupies the contiguous range
    /// `[chunk_start * inner_size, chunk_start * inner_size + chunk_len *
    /// inner_size)` of the parent output buffer, where `inner_size` is the
    /// product of the extents after the split axis. A caller relies on this
    /// to hand each chunk its own disjoint `&mut` sub-slice — via successive
    /// `split_at_mut` of one parent buffer — and run every chunk
    /// concurrently with no further coordination.
    ///
    /// The two layout kinds are rebased differently, which is *why* that
    /// works:
    /// - every **operand** layout's base is shifted by
    ///   `chunk_start * stride(split_axis)`, because operands are read from
    ///   the one full, unsplit source buffer shared by every chunk.
    /// - a reduce's `out_layout` is **not** shifted at all: the interpreter
    ///   already derives every write offset from a loop that iterates the
    ///   split axis's own (already-shrunk) extent starting at 0, for every
    ///   chunk alike, so an out_layout carrying the parent's unmodified
    ///   base already produces exactly the 0-based offsets a `split_at_mut`
    ///   sub-slice expects. Rebasing it too would double-count the offset.
    #[must_use]
    pub fn split(&self, parts: usize) -> Option<Vec<BoundOp>> {
        self.split_aligned(parts, 1)
    }

    /// Same contract as [`split`](Self::split), except each of the first
    /// `parts - 1` chunks is rounded down to a multiple of `alignment` rows
    /// (the remainder folds into the last, already-ragged chunk) instead of
    /// always taking `extent / parts` exactly. `alignment == 1` degenerates
    /// to [`split`](Self::split)'s behavior byte-for-byte.
    ///
    /// Exists because equal row counts are not equal wall-clock: a caller
    /// tiling its kernel in `alignment`-row blocks pays a narrower,
    /// measurably slower fallback path for every chunk boundary that does
    /// not land on a tile edge, and that count grows with chunk count (see
    /// `cpu::TILE_ROWS`'s doc for the measured spread).
    #[must_use]
    pub fn split_aligned(&self, parts: usize, alignment: u64) -> Option<Vec<BoundOp>> {
        if parts < 2 {
            return None;
        }
        let split_axis = self.split_axis()?;
        let extent = self.extents[split_axis as usize];
        if extent < parts as u64 {
            return None;
        }

        Some(
            chunk_ranges(extent, parts, alignment)
                .map(|(chunk_start, chunk_len)| {
                    self.rebase_chunk(split_axis, chunk_start, chunk_len)
                })
                .collect(),
        )
    }

    fn split_axis(&self) -> Option<u16> {
        match &self.kind {
            // decode-only shape (`n_tokens == 1`): never worth splitting.
            BoundOpKind::CachedAttention { .. } | BoundOpKind::GatedDeltaNet { .. } => None,
            BoundOpKind::Elementwise { .. } => (!self.extents.is_empty()).then_some(0),
            // `out_scatter: Some(_)` is a scatter: conservatively
            // ineligible for splitting. A chunked run would need
            // `out_scatter`'s own `index_layout`/`extent` rebased per chunk
            // (the same treatment `rebase_operands` already gives every
            // gathered operand's `Lookup`), and every destination touched
            // by more than one chunk would need its fold synchronized across
            // chunk boundaries — real work this task's own scope named out
            // (parallel scatter stays sequential-only, named, not silently
            // wrong). `None` here just routes a scatter through the same
            // one-chunk path `Keep::Scan` already takes.
            BoundOpKind::Reduce {
                keep,
                output_axes,
                out_scatter: Some(_),
                ..
            } => {
                let _ = (keep, output_axes);
                None
            }
            BoundOpKind::Reduce {
                keep,
                output_axes,
                out_scatter: None,
                ..
            } => match keep {
                Keep::Scan => None,
                Keep::Reduce => output_axes.first().copied(),
            },
            // an `Iota` is cheap enough (one write per element, no operand
            // reads) that splitting it across workers is not worth the
            // bookkeeping; `None` here just means a caller runs it as one
            // chunk, the same as any other unsplittable op.
            BoundOpKind::Iota | BoundOpKind::Constant { .. } => None,
        }
    }

    fn rebase_chunk(&self, split_axis: u16, chunk_start: u64, chunk_len: u64) -> BoundOp {
        let mut extents = self.extents.clone();
        extents[split_axis as usize] = chunk_len;

        let kind = match &self.kind {
            BoundOpKind::CachedAttention {
                operands,
                query_rows,
                cached_key_rows,
                new_key_rows,
                kv_heads,
                query_groups,
                head_dim,
                rotary_dim,
                scale,
                cached_lower_inclusive,
                new_upper_inclusive,
            } => BoundOpKind::CachedAttention {
                operands: rebase_operands(operands, split_axis, chunk_start),
                query_rows: *query_rows,
                cached_key_rows: *cached_key_rows,
                new_key_rows: *new_key_rows,
                kv_heads: *kv_heads,
                query_groups: *query_groups,
                head_dim: *head_dim,
                rotary_dim: *rotary_dim,
                scale: *scale,
                cached_lower_inclusive: *cached_lower_inclusive,
                new_upper_inclusive: *new_upper_inclusive,
            },
            BoundOpKind::Elementwise { body, operands } => BoundOpKind::Elementwise {
                body: body.clone(),
                operands: rebase_operands(operands, split_axis, chunk_start),
            },
            // unreachable in practice: `split_axis` returns `None` for
            // `GatedDeltaNet` (this slice's `n_tokens == 1` shape is never
            // worth chunking), kept explicit for the same reason `Iota`/
            // `Constant` below are.
            kind @ BoundOpKind::GatedDeltaNet { .. } => kind.clone(),
            BoundOpKind::Reduce {
                element_body,
                reduce_op,
                init,
                keep,
                operands,
                output_axes,
                out_layout,
                out_scatter,
                epilogue_body,
                epilogue_operands,
                epilogue_broadcast_axes,
            } => BoundOpKind::Reduce {
                element_body: element_body.clone(),
                reduce_op: *reduce_op,
                init: *init,
                keep: *keep,
                operands: rebase_operands(operands, split_axis, chunk_start),
                output_axes: output_axes.clone(),
                // unchanged from the parent: see this method's doc for why
                // an unshifted out_layout already yields 0-based write
                // offsets.
                out_layout: out_layout.clone(),
                // `split_axis` never returns `Some` for a scatter (see its
                // own doc), so this arm only ever runs with `out_scatter ==
                // None` in practice; cloned rather than asserted so a future
                // relaxation of that gate does not silently drop it.
                out_scatter: out_scatter.clone(),
                epilogue_body: epilogue_body.clone(),
                // `epilogue_operands` lives in OUTPUT-axis-order local
                // coordinates (`epilogue_body`'s own doc), not `self.extents`'
                // full-iteration numbering `split_axis` is expressed in —
                // but `split_axis` is always `output_axes[0]` (`split_axis`
                // above never returns anything else for a `Reduce`), and
                // `output_axes[0]` is ALWAYS local position `0` in that
                // output-axis-order space by construction. So the chunk
                // boundary this whole call is rebasing for is local axis `0`
                // here, regardless of which full-space axis `split_axis`
                // itself names.
                epilogue_operands: rebase_operands(epilogue_operands, 0, chunk_start),
                // `split_axis` never returns `Some` for a broadcast-reduce
                // epilogue (chunk-splitting a fold that ALSO re-broadcasts
                // over an axis is not implemented), so this stays a plain
                // clone rather than needing its own rebase.
                epilogue_broadcast_axes: epilogue_broadcast_axes.clone(),
            },
            // unreachable in practice: `split_axis` returns `None` for
            // `Iota`, so `split`/`split_aligned` never call this for one —
            // kept explicit rather than a catch-all so a future change to
            // `split_axis` cannot silently start routing `Iota` here with no
            // rebase logic to run.
            BoundOpKind::Iota => BoundOpKind::Iota,
            // same reasoning as `Iota` above: `split_axis` returns `None`
            // for a `Constant`, so this arm is never reached in practice.
            BoundOpKind::Constant { value } => BoundOpKind::Constant { value: *value },
        };

        BoundOp {
            node: self.node,
            dtype: self.dtype,
            extents,
            kind,
        }
    }
}

fn rebase_operands(operands: &BoundOperands, split_axis: u16, chunk_start: u64) -> BoundOperands {
    operands
        .iter()
        .map(|(node, layout, lookup)| {
            let rebased_lookup = lookup.as_ref().map(|lookup| Lookup {
                indices: lookup.indices,
                index_layout: rebase_layout(&lookup.index_layout, split_axis, chunk_start),
                element_stride: lookup.element_stride,
                extent: lookup.extent,
            });
            (
                *node,
                rebase_layout(layout, split_axis, chunk_start),
                rebased_lookup,
            )
        })
        .collect()
}

/// `parts` contiguous `(start, len)` ranges covering `0..extent`: the first
/// `parts - 1` ranges are `extent / parts` wide, rounded down to a multiple
/// of `alignment` (unless that would zero them out, in which case the raw
/// unaligned width is kept), and the last absorbs whatever remains — the
/// only one that can be a different (ragged) size. `alignment <= 1` is a
/// no-op: the rounding step is skipped entirely.
fn chunk_ranges(extent: u64, parts: usize, alignment: u64) -> impl Iterator<Item = (u64, u64)> {
    let raw_len = extent / parts as u64;
    let chunk_len = if alignment > 1 && raw_len >= alignment {
        raw_len - (raw_len % alignment)
    } else {
        raw_len
    };
    (0..parts).scan(0u64, move |start, index| {
        let chunk_start = *start;
        let len = if index + 1 == parts {
            extent - chunk_start
        } else {
            chunk_len
        };
        *start += len;
        Some((chunk_start, len))
    })
}

fn rebase_layout(layout: &Layout, split_axis: u16, chunk_start: u64) -> Layout {
    Layout {
        base: layout.base + layout.stride(split_axis) * chunk_start as i64,
        strides: layout.strides.clone(),
    }
}

#[derive(Clone)]
struct HeldElementwise {
    dtype: DType,
    body: ScalarOp,
    operands: Vec<(NodeId, IndexMap)>,
}

/// The prefix state of op building: elementwise ops seen but not yet
/// materialized.
///
/// `retires` and `position` make [`BoundOpBuilder::push`] a single-argument-per-node
/// step (`expr`, `shapes`) rather than needing `node`/`retires` threaded in
/// by the caller on every call: `retires[i]` is node `i`'s kill-flag list
/// (see [`live::annotate`], computed once over the whole program before the
/// first push), and `position` is the node id the next push resolves to —
/// both pieces this type already needed to know, now carried as its own
/// state instead of repeated arguments.
pub struct BoundOpBuilder {
    held: RefCell<BTreeMap<NodeId, HeldElementwise>>,
    retires: Vec<Vec<NodeId>>,
    position: Cell<u32>,
    /// `ones[node.0]` is `true` when `node` was pushed as an
    /// [`Op::Constant`] whose `value` is exactly `1.0` — grown one entry per
    /// [`push`](Self::push) call, never read ahead of the position that
    /// produced it. This is what lets [`compose_operand`] recognize (and
    /// drop) a `Multiply` operand that is algebraically a no-op without
    /// requiring the whole program in hand, honoring this module's own
    /// sans-IO streaming contract (see module doc) rather than threading a
    /// full `&[Op]` slice through every fusion call.
    ones: RefCell<Vec<bool>>,
    /// `is_iota[node.0]` is `true` when `node` was pushed as an [`Op::Iota`]
    /// — the same one-entry-per-push discipline as `ones`, read by
    /// [`eliminate_masked_window_reduce`] to confirm a candidate operand is
    /// really one of `window_mask`'s three position markers rather than some
    /// other node that merely shares its `NodeId` shape.
    is_iota: RefCell<Vec<bool>>,
    /// Whether this node or an elementwise descendant carries a multi-term
    /// index map. A descendant may already be materialized when a consuming
    /// reduce is pushed, so the original shape must survive outside `held`.
    packed_mapping_subtree: RefCell<Vec<bool>>,
    /// `constant_value[node.0]` is `Some(value)` when `node` was pushed as an
    /// [`Op::Constant`] carrying `value` — a generalization of `ones` that
    /// keeps the actual stride literal (not just whether it is `1.0`), which
    /// [`eliminate_masked_window_reduce`]'s in-bounds proof needs.
    constant_value: RefCell<Vec<Option<f32>>>,
    /// The [`NumericPolicy`] every [`Constants`] this builder hands to
    /// [`push_canonical_step`] carries — governs whether the
    /// `x+0` ([`NumericRewrite::IdentityEliminationSignedZero`]) and
    /// `max(x,-inf)`/`min(x,+inf)` ([`NumericRewrite::IdentityEliminationNanAssumption`])
    /// identity eliminations fire, on top of the always-on `x*1` case.
    numeric_policy: NumericPolicy,
}

impl BoundOpBuilder {
    /// `retires` is normally [`live::annotate`]`(program, outputs)`.
    #[must_use]
    pub fn new(retires: Vec<Vec<NodeId>>, numeric_policy: NumericPolicy) -> Self {
        Self {
            held: RefCell::new(BTreeMap::new()),
            retires,
            position: Cell::new(0),
            ones: RefCell::new(Vec::new()),
            is_iota: RefCell::new(Vec::new()),
            packed_mapping_subtree: RefCell::new(Vec::new()),
            constant_value: RefCell::new(Vec::new()),
            numeric_policy,
        }
    }

    /// Judge one expression: hold an elementwise op, or emit whatever is now
    /// ready.
    ///
    /// May return more than one [`BoundOp`]: consuming a held elementwise op
    /// that turns out not to fuse must materialize it before the current
    /// expression can read it, so a single push can ready both that
    /// standalone op and the current expression's own — and, for an
    /// elementwise expression, one materialization per operand that fails to
    /// fuse, up to [`ScalarOp::arity`]'s current maximum
    /// (`READY_BATCH_CAPACITY`).
    pub fn push(&self, expr: &Op, shapes: &Shapes) -> Result<ReadyBatch, TensorError> {
        let node = NodeId(self.position.get());
        self.position.set(self.position.get() + 1);
        let empty = Vec::new();
        let retires = self.retires.get(node.0 as usize).unwrap_or(&empty);
        self.ones
            .borrow_mut()
            .push(matches!(expr, Op::Constant { value, .. } if *value == 1.0));
        self.is_iota
            .borrow_mut()
            .push(matches!(expr, Op::Iota { .. }));
        let packed_mapping = match expr {
            Op::Elementwise { operands, .. } => {
                let direct = operands.iter().any(|(_, map)| {
                    map.affine()
                        .axes
                        .iter()
                        .any(|axis| axis.terms.len() > 1)
                });
                let descendants = self.packed_mapping_subtree.borrow();
                direct
                    || operands.iter().any(|(operand, _)| {
                        descendants.get(operand.0 as usize).copied().unwrap_or(false)
                    })
            }
            _ => false,
        };
        self.packed_mapping_subtree
            .borrow_mut()
            .push(packed_mapping);
        self.constant_value
            .borrow_mut()
            .push(if let Op::Constant { value, .. } = expr {
                Some(*value)
            } else {
                None
            });

        let mut emitted = ReadyBatch::new();

        match expr {
            Op::Input { .. } => {}
            Op::Iota { dtype, .. } => {
                let extents = shapes.of(node).to_vec();
                push_ready(
                    &mut emitted,
                    node,
                    BoundOp {
                        node,
                        dtype: *dtype,
                        extents,
                        kind: BoundOpKind::Iota,
                    },
                )?;
            }
            Op::Constant { dtype, value, .. } => {
                let extents = shapes.of(node).to_vec();
                push_ready(
                    &mut emitted,
                    node,
                    BoundOp {
                        node,
                        dtype: *dtype,
                        extents,
                        kind: BoundOpKind::Constant { value: *value },
                    },
                )?;
            }
            Op::Elementwise {
                dtype,
                body,
                operands,
                ..
            } => {
                for (operand_node, map) in operands {
                    let still_live = !retires.contains(operand_node);
                    let non_identity = !is_identity_projection(map);
                    let not_held = !self.held.borrow().contains_key(operand_node);
                    let fuses = !still_live && !non_identity && !not_held;
                    #[cfg(feature = "instrument")]
                    {
                        let outcome = if fuses {
                            Ok(())
                        } else if still_live {
                            Err(instrument::FuseDeclineReason::StillLive)
                        } else if non_identity {
                            Err(instrument::FuseDeclineReason::NonIdentityProjection)
                        } else {
                            Err(instrument::FuseDeclineReason::NotHeld)
                        };
                        instrument::record_fuse_attempt(
                            *operand_node,
                            instrument::FuseSite::ElementwiseOperand,
                            outcome,
                        );
                        debug!(
                            node = operand_node.0,
                            kind = "elementwise_operand_fuse",
                            decision = if fuses { "fused" } else { "materialized" },
                            into = node.0,
                            still_live = still_live,
                            "single-consumer elementwise composition decision -- still_live is gated by the requested output set"
                        );
                    }
                    if !fuses {
                        self.materialize_if_held(*operand_node, shapes, &mut emitted)?;
                    }
                    self.materialize_computed_indices(map, shapes, &mut emitted)?;
                }
                self.held.borrow_mut().insert(
                    node,
                    HeldElementwise {
                        dtype: *dtype,
                        body: *body,
                        operands: operands.clone(),
                    },
                );
            }
            Op::Reduce(reduce) => {
                let window_elimination = eliminate_masked_window_reduce(
                    reduce,
                    &self.held,
                    &self.is_iota.borrow(),
                    &self.constant_value.borrow(),
                    shapes,
                );
                #[cfg(feature = "instrument")]
                instrument::record_window_reduce_attempt(window_elimination.is_some());
                if let Some((source_node, source_map)) = window_elimination {
                    // `source_map`'s windowed axis is a genuine two-term
                    // affine index (`stride*out + kernel`), not the plain
                    // single-term projection `compose_operand`'s own fusion
                    // contract requires of a map connecting to a still-held
                    // node (`is_identity_projection`'s own doc: "a window...
                    // materializes its operand instead of composing through
                    // it"). `source` may still be `held` here — e.g. a prior
                    // layer's bias-add fusing into this window on the
                    // ordinary identity-projection path it was pushed under
                    // — so it must be forced to materialize as a real buffer
                    // before this non-identity map ever reads it, or
                    // `compose_operand`'s recursive remap silently
                    // mis-addresses whatever was held beneath it.
                    self.materialize_if_held(source_node, shapes, &mut emitted)?;
                    let identity_operand = vec![(source_node, source_map)];
                    push_ready(
                        &mut emitted,
                        node,
                        build_elementwise_op(
                            node,
                            shapes,
                            &self.held,
                            reduce.dtype,
                            ScalarOp::Identity,
                            &identity_operand,
                            Constants {
                                ones: &self.ones.borrow(),
                                values: &self.constant_value.borrow(),
                                numeric_policy: self.numeric_policy,
                            },
                        ),
                    )?;
                    return Ok(emitted);
                }

                let still_live = !retires.contains(&reduce.operand);
                let non_identity = !is_identity_projection(&reduce.in_map);
                let not_held = !self.held.borrow().contains_key(&reduce.operand);
                let fuses = !still_live && !non_identity && !not_held;
                if fuses
                    && let Some(activation_node) =
                        composed_packed_product_activation(
                            &self.held,
                            &self.packed_mapping_subtree,
                            reduce.operand,
                        )
                {
                    // ROW 431 (`docs/discipline.md`): materialize ONLY the
                    // composed activation side of `Multiply(packed, a)` so
                    // `W * a` and this reduction stay fused --
                    // `run_reduce_quantized`'s admission contract
                    // (`packed_reduce_activation_operand`, `cpu.rs`) requires
                    // a bare two-operand product, and this is what makes
                    // that shape true without ever materializing the
                    // output-width product ROW 430 used to (superseded).
                    #[cfg(feature = "instrument")]
                    debug!(
                        node = reduce.operand.0,
                        activation = activation_node.0,
                        reduce = node.0,
                        "reduce fusion materializes the composed activation operand of a \
                         packed-weight product so W * a and the reduction stay fused \
                         (docs/discipline.md ROW 431, supersedes ROW 430)"
                    );
                    self.materialize_if_held(activation_node, shapes, &mut emitted)?;
                }
                #[cfg(feature = "instrument")]
                {
                    let outcome = if fuses {
                        Ok(())
                    } else if still_live {
                        Err(instrument::FuseDeclineReason::StillLive)
                    } else if non_identity {
                        Err(instrument::FuseDeclineReason::NonIdentityProjection)
                    } else {
                        Err(instrument::FuseDeclineReason::NotHeld)
                    };
                    instrument::record_fuse_attempt(
                        reduce.operand,
                        instrument::FuseSite::ReduceOperand,
                        outcome,
                    );
                }

                let (element_body, operands) = if fuses {
                    let reduce_extent: u64 = shape::fold_iteration_extents(node, reduce, shapes)?
                        .iter()
                        .product();
                    self.quarantine_broadcast_operands(
                        reduce.operand,
                        reduce_extent,
                        shapes,
                        &mut emitted,
                    )?;
                    compose_fused_operands(
                        shapes,
                        &self.held,
                        reduce.operand,
                        &reduce.in_map,
                        Constants {
                            ones: &self.ones.borrow(),
                            values: &self.constant_value.borrow(),
                            numeric_policy: self.numeric_policy,
                        },
                    )
                } else {
                    self.materialize_if_held(reduce.operand, shapes, &mut emitted)?;
                    self.materialize_computed_indices(&reduce.in_map, shapes, &mut emitted)?;
                    let operand = build_operand(reduce.operand, &reduce.in_map, shapes);
                    (ComposedBody::leaf(ScalarOp::Identity), vec![operand])
                };

                // A scatter's `out_map` names an `indices` node exactly the
                // way `in_map` can, and it is never covered by the
                // `in_map`-only walk above (`fuses`/the `else` branch both
                // only ever touch `reduce.operand`/`reduce.in_map`) — see
                // this method's own doc for why `materialize_computed_indices`
                // is unconditional for `in_map`; the same reasoning applies
                // here, independent of whether the operand fused.
                self.materialize_computed_indices(&reduce.out_map, shapes, &mut emitted)?;

                push_ready(
                    &mut emitted,
                    node,
                    build_reduce_op(node, reduce, shapes, element_body, operands)?,
                )?;
            }
        }

        Ok(emitted)
    }

    /// [`bind_plain`]'s reachability skip lane: advances the position
    /// counter and keeps every per-node bookkeeping vector
    /// (`ones`/`is_iota`/`packed_mapping_subtree`/`constant_value`) aligned
    /// to it, without running any of [`push`](Self::push)'s binding work.
    /// Safe because a program only ever references backwards
    /// (`op.rs`'s own module doc), so no node this method skips can be an
    /// operand, gather index, or reduce map of a node the caller does push
    /// — every read of these vectors is at a live node's own index.
    pub fn skip(&self) {
        self.position.set(self.position.get() + 1);
        self.ones.borrow_mut().push(false);
        self.is_iota.borrow_mut().push(false);
        self.packed_mapping_subtree.borrow_mut().push(false);
        self.constant_value.borrow_mut().push(None);
    }

    /// Flush every elementwise op still held: each was a requested output,
    /// and either way it materializes as its own op. A node reachable from
    /// no output never enters `held` at all — [`bind_plain`] never calls
    /// [`push`](Self::push) for it — so this no longer flushes dead code.
    /// Processed from the highest [`NodeId`] down: a still-held node can
    /// only ever be fused into a consumer with a *greater* id (references
    /// point backwards only), so visiting consumers first lets
    /// `build_elementwise_op` absorb whatever it still can before an
    /// earlier, now-absorbed node would otherwise be flushed standalone.
    pub fn finish(self, shapes: &Shapes) -> Result<Vec<BoundOp>, TensorError> {
        let mut remaining: Vec<NodeId> = self.held.borrow().keys().copied().collect();
        remaining.sort_unstable_by(|left, right| right.cmp(left));

        let mut built = Vec::new();
        for node in remaining {
            let mut materialized = self.materialize_node(node, shapes)?;
            materialized.reverse();
            built.extend(materialized);
        }
        built.reverse();
        Ok(built)
    }

    /// A [`IndexMap::Computed`]'s `indices` is a backwards [`NodeId`]
    /// reference that never appears in any op's own `operands` list — it
    /// only ever lives inside a sibling operand's *map* — so the operand
    /// walk in [`Self::push`] must force it here too, or a lone held
    /// `Elementwise` reached only this way sits un-materialized past the
    /// point a gather reads it (`docs/discipline.md` ROW 131 Limitation 2).
    /// `indices` is always read as a plain buffer at evaluation time
    /// ([`build_operand`]'s `Lookup`, never composed through by
    /// [`compose_operand`]), so unlike `operand_node` this is unconditional
    /// — there is no fusion path for it to opt out of.
    fn materialize_computed_indices(
        &self,
        map: &IndexMap,
        shapes: &Shapes,
        emitted: &mut ReadyBatch,
    ) -> Result<(), TensorError> {
        if let IndexMap::Computed { indices, .. } = map {
            self.materialize_if_held(*indices, shapes, emitted)?;
        }
        Ok(())
    }

    fn materialize_if_held(
        &self,
        node: NodeId,
        shapes: &Shapes,
        emitted: &mut ReadyBatch,
    ) -> Result<(), TensorError> {
        for materialized in self.materialize_node(node, shapes)? {
            push_ready(emitted, node, materialized)?;
        }
        Ok(())
    }

    fn materialize_node(&self, node: NodeId, shapes: &Shapes) -> Result<Vec<BoundOp>, TensorError> {
        let mut emitted = Vec::new();
        loop {
            let Some(held) = self.held.borrow().get(&node).cloned() else {
                return Ok(emitted);
            };
            if self.preview_elementwise_buffer_count(node, shapes)? <= 31 {
                break;
            }

            let candidate = held
                .operands
                .iter()
                .filter(|(operand, map)| {
                    is_identity_projection(map) && self.held.borrow().contains_key(operand)
                })
                .filter_map(|(operand, _)| {
                    let mut preview_held = self.held.borrow().clone();
                    preview_held.remove(operand);
                    let count = preview_elementwise_buffer_count(
                        node,
                        shapes,
                        &preview_held,
                        &self.ones.borrow(),
                        &self.constant_value.borrow(),
                        self.numeric_policy,
                    )
                    .ok()?;
                    Some((*operand, count))
                })
                .min_by_key(|(operand, count)| (*count, *operand))
                .map(|(operand, _)| operand)
                .ok_or(TensorError::NotLowerable {
                    node,
                    reason: "elementwise body exceeds Metal's 31-buffer ABI and has no composable child to materialize",
                })?;

            emitted.extend(self.materialize_node(candidate, shapes)?);
        }

        // see `finish`'s comment: the borrow must end before this `if let`
        // body runs, since `build_elementwise_op` borrows `self.held` too.
        let removed = self.held.borrow_mut().remove(&node);
        if let Some(held) = removed {
            emitted.push(build_elementwise_op(
                node,
                shapes,
                &self.held,
                held.dtype,
                held.body,
                &held.operands,
                Constants {
                    ones: &self.ones.borrow(),
                    values: &self.constant_value.borrow(),
                    numeric_policy: self.numeric_policy,
                },
            ));
        }
        Ok(emitted)
    }

    fn preview_elementwise_buffer_count(
        &self,
        node: NodeId,
        shapes: &Shapes,
    ) -> Result<usize, TensorError> {
        preview_elementwise_buffer_count(
            node,
            shapes,
            &self.held.borrow(),
            &self.ones.borrow(),
            &self.constant_value.borrow(),
            self.numeric_policy,
        )
    }

    /// Walks `node`'s still-held operands and materializes any whose own
    /// natural iteration space (`shapes.of(child)`) is smaller than
    /// `reduce_extent` — composing one through anyway would run its body
    /// once per `reduce_extent` element instead of once per its own, which
    /// is exactly the cost [`is_identity_projection`] cannot see: it only
    /// judges one map's shape, not what fusing recursively absorbs beneath
    /// it (see `compose_operand`'s own doc — it trusts every map it
    /// recurses through was already checked, but that check happened at a
    /// different, earlier `push`, against that op's own — smaller —
    /// iteration space, not against this reduce's). Safe children (same or
    /// larger extent) are walked further, since a broadcast can reappear
    /// several levels down.
    fn quarantine_broadcast_operands(
        &self,
        node: NodeId,
        reduce_extent: u64,
        shapes: &Shapes,
        emitted: &mut ReadyBatch,
    ) -> Result<(), TensorError> {
        let children = self
            .held
            .borrow()
            .get(&node)
            .map(|held| held.operands.clone());
        let Some(children) = children else {
            return Ok(());
        };
        for (child, _map) in children {
            if !self.held.borrow().contains_key(&child) {
                continue;
            }
            let child_extent: u64 = shapes.of(child).iter().product();
            let quarantined = child_extent < reduce_extent;
            #[cfg(feature = "instrument")]
            instrument::record_fuse_attempt(
                child,
                instrument::FuseSite::QuarantineBroadcast,
                if quarantined {
                    Err(instrument::FuseDeclineReason::BroadcastQuarantined)
                } else {
                    Ok(())
                },
            );
            if quarantined {
                self.materialize_if_held(child, shapes, emitted)?;
            } else {
                self.quarantine_broadcast_operands(child, reduce_extent, shapes, emitted)?;
            }
        }
        Ok(())
    }
}

/// Appends one ready [`BoundOp`] to a [`ReadyBatch`], turning an overflow
/// (never observed for today's `ScalarOp` variants — see
/// [`READY_BATCH_CAPACITY`]'s own doc) into a [`TensorError`] instead of a
/// panic.
fn push_ready(emitted: &mut ReadyBatch, node: NodeId, op: BoundOp) -> Result<(), TensorError> {
    emitted.try_push(op).map_err(|_| TensorError::NotLowerable {
        node,
        reason: "one push readied more BoundOps than the no-alloc batch capacity allows",
    })
}

/// `In = (Op, Shapes)` matches [`shape::ShapeTable`]'s own `Pipe::Out`
/// exactly, so `AndThen::new(ShapeTable, BoundOpBuilder)` (or
/// `shapes_instance.and_then(builder_instance)`) composes with no adapter:
/// shape resolution's snapshot travels alongside the `Op` it was resolved
/// for, and [`BoundOpBuilder::push`] reads both straight out of `Self::In`.
impl Pipe for BoundOpBuilder {
    type In = (Op, Shapes);
    type Out = ReadyBatch;
    type Err = TensorError;

    fn call(
        &self,
        (expr, shapes): Self::In,
    ) -> impl Future<Output = Result<ReadyBatch, TensorError>> {
        async move { self.push(&expr, &shapes) }
    }
}

fn build_elementwise_op(
    node: NodeId,
    shapes: &Shapes,
    held: &RefCell<BTreeMap<NodeId, HeldElementwise>>,
    dtype: DType,
    body: ScalarOp,
    operands: &[(NodeId, IndexMap)],
    constants: Constants<'_>,
) -> BoundOp {
    let extents = shapes.of(node).to_vec();
    let (composed_body, built_operands) = compose(shapes, held, body, operands, constants);
    BoundOp {
        node,
        dtype,
        extents,
        kind: BoundOpKind::Elementwise {
            body: composed_body,
            operands: built_operands,
        },
    }
}

fn preview_elementwise_buffer_count(
    node: NodeId,
    shapes: &Shapes,
    held: &BTreeMap<NodeId, HeldElementwise>,
    ones: &[bool],
    values: &[Option<f32>],
    numeric_policy: NumericPolicy,
) -> Result<usize, TensorError> {
    let held = RefCell::new(held.clone());
    let entry = held
        .borrow()
        .get(&node)
        .cloned()
        .ok_or(TensorError::NotLowerable {
            node,
            reason: "elementwise ABI preview requires a held node",
        })?;
    let (_, operands) = compose(
        shapes,
        &held,
        entry.body,
        &entry.operands,
        Constants {
            ones,
            values,
            numeric_policy,
        },
    );
    Ok(metal_buffer_binding_count(
        &operands,
        &alloc::vec::Vec::new(),
    ))
}

/// One operand's [`Layout`] (and, for a gather, its [`Lookup`]), built
/// directly from its [`IndexMap`] — the one place that decides how an
/// `Affine` vs a `Computed` map turns into what an executor reads.
fn build_operand(
    node: NodeId,
    map: &IndexMap,
    shapes: &Shapes,
) -> (NodeId, Layout, Option<Lookup>) {
    match map {
        IndexMap::Affine(pattern) => (node, layout_of(pattern, shapes.of(node)), None),
        IndexMap::Computed {
            indices,
            index_map,
            base,
            gathered_dim,
        } => {
            let operand_shape = shapes.of(node);
            let layout = layout_of(base, operand_shape);
            let index_layout = layout_of(index_map, shapes.of(*indices));
            let element_stride = row_major_strides(operand_shape)[*gathered_dim as usize];
            let extent = operand_shape[*gathered_dim as usize];
            let lookup = Lookup {
                indices: *indices,
                index_layout,
                element_stride,
                extent,
            };
            (node, layout, Some(lookup))
        }
    }
}

fn build_reduce_op(
    node: NodeId,
    reduce: &Reduce,
    shapes: &Shapes,
    element_body: ComposedBody,
    operands: BoundOperands,
) -> Result<BoundOp, TensorError> {
    let out_pattern = reduce.out_map.affine();
    let output_axes = pure_projection_axes(out_pattern);
    let (out_layout, out_scatter) = match &reduce.out_map {
        IndexMap::Affine(pattern) => (layout_of(pattern, shapes.of(node)), None),
        IndexMap::Computed {
            indices,
            index_map,
            base,
            gathered_dim,
        } => {
            let output_shape = shapes.of(node);
            let out_layout = build_scatter_out_layout(base, *gathered_dim, output_shape);
            let index_layout = layout_of(index_map, shapes.of(*indices));
            let element_stride = row_major_strides(output_shape)[*gathered_dim as usize];
            let extent = output_shape[*gathered_dim as usize];
            let lookup = Lookup {
                indices: *indices,
                index_layout,
                element_stride,
                extent,
            };
            (out_layout, Some(lookup))
        }
    };
    Ok(BoundOp {
        node,
        dtype: reduce.dtype,
        extents: shape::fold_iteration_extents(node, reduce, shapes)?,
        kind: BoundOpKind::Reduce {
            element_body,
            reduce_op: reduce.body,
            init: reduce.init,
            keep: reduce.keep,
            operands,
            output_axes,
            out_layout,
            out_scatter,
            // no epilogue at push time — `reduce_epilogue_fusion` (the
            // `reduce-epilogue-fusion` post-pass) is the only writer of a
            // non-default value for either field, and it runs after every
            // `BoundOp` here already exists.
            epilogue_body: ComposedBody::leaf(ScalarOp::Identity),
            epilogue_operands: Vec::new(),
            epilogue_broadcast_axes: SmallVec::new(),
        },
    })
}

/// [`layout_of`]'s counterpart for a scatter `out_map`'s `base` pattern:
/// identical walk, except `gathered_dim`'s own axis is skipped rather than
/// folded in. `layout_of` reads every axis's `offset` as a real address
/// contribution (`base += offset * element_stride`), which is exactly right
/// for a gather's `base` (that entry is `terms: [], offset: 0` there by
/// convention, so it contributes nothing) but would be wrong here: a
/// scatter's `base` repurposes that same slot's `offset` to carry the
/// destination's static extent (`map.rs`'s `IndexMap::Computed` doc), not an
/// address. Skipping the axis entirely is correct either way, since the
/// scattered axis's real address only ever comes from the fetched index at
/// evaluation time — see [`BoundOpKind::Reduce`]'s `out_scatter` field doc.
fn build_scatter_out_layout(
    base: &IndexPattern,
    gathered_dim: u16,
    output_shape: &[u64],
) -> Layout {
    let element_strides = row_major_strides(output_shape);
    let mut strides = SmallVec::<[i64; MAX_INLINE_RANK]>::from_elem(0, base.iter_rank as usize);
    let mut layout_base = 0i64;
    for (axis_index, axis) in base.axes.iter().enumerate() {
        if axis_index as u16 == gathered_dim {
            continue;
        }
        let stride = element_strides[axis_index];
        layout_base += i64::from(axis.offset) * stride;
        for term in &axis.terms {
            strides[term.axis as usize] += i64::from(term.coeff) * stride;
        }
    }
    Layout {
        base: layout_base,
        strides,
    }
}

fn pure_projection_axes(pattern: &IndexPattern) -> SmallVec<[u16; MAX_INLINE_RANK]> {
    pattern
        .axes
        .iter()
        .filter_map(|axis| match axis.terms.as_slice() {
            [term] if term.coeff == 1 => Some(term.axis),
            _ => None,
        })
        .collect()
}

/// ROW 431 (`docs/discipline.md`, supersedes ROW 430): whether `node`'s
/// held body is `Multiply(packed, composed)` -- one operand's own map OR a
/// map in the held body beneath it carrying a genuine multi-term axis (the
/// multi-letter packed-row contraction `wo`'s own reshape idiom,
/// `spec.rs:9017-9031`, uses -- a single-letter packed weight like
/// `wq`/`wk`/`wv` never has one) while the OTHER operand is STILL a held,
/// unmaterialized elementwise chain rather than a plain leaf. The recursive
/// walk matters because Qwen's gate-to-Q projection puts the multi-term
/// reshape inside the weight's own broadcast multiply, not in the outer
/// product's map. Returns that other (activation) node so the caller can
/// force just IT to materialize -- this is exactly the shape
/// `run_reduce_quantized`'s admission contract
/// (`packed_reduce_activation_operand`, `cpu.rs`) requires: a bare
/// two-operand product, weight times ONE already-materialized activation
/// buffer. ROW 430 instead declined to fuse the whole product here, which
/// materialized the OUTPUT-width `[u, g, d, o]` buffer; forcing only the
/// activation side (`[u, g, d]`) keeps `W * a` and the reduction fused and
/// never introduces the output axis into an intermediate buffer at all.
fn composed_packed_product_activation(
    held: &RefCell<BTreeMap<NodeId, HeldElementwise>>,
    packed_mapping_subtree: &RefCell<Vec<bool>>,
    node: NodeId,
) -> Option<NodeId> {
    let (body, operands) = held
        .borrow()
        .get(&node)
        .map(|entry| (entry.body, entry.operands.clone()))?;
    if body != ScalarOp::Multiply {
        return None;
    }
    let [(first_node, first_map), (second_node, second_map)] = operands.as_slice() else {
        return None;
    };
    // a computed (data-dependent) map is ALSO the packed operand's own
    // signal, not just a multi-term affine reshape: `grouped_gathered_expert_product`
    // (`spec.rs`) gathers the packed weight stack with a single-term
    // `IndexMap::Computed` axis per operand axis, so the multi-term check
    // alone (tuned for `wo`'s reshape idiom) never fires for it and this
    // fusion silently declined, materializing the full `[sequence, selected,
    // d_in, d_out]` product instead (`docs/discipline.md` ROW 538).
    let first_packed = packed_mapping_in_held_tree(held, packed_mapping_subtree, *first_node)
        || first_map.is_data_dependent()
        || first_map
            .affine()
            .axes
            .iter()
            .any(|axis| axis.terms.len() > 1);
    let second_packed = packed_mapping_in_held_tree(held, packed_mapping_subtree, *second_node)
        || second_map.is_data_dependent()
        || second_map
            .affine()
            .axes
            .iter()
            .any(|axis| axis.terms.len() > 1);
    if first_packed == second_packed {
        return None;
    }
    let other_node = if first_packed {
        *second_node
    } else {
        *first_node
    };
    held.borrow()
        .contains_key(&other_node)
        .then_some(other_node)
}

fn packed_mapping_in_held_tree(
    held: &RefCell<BTreeMap<NodeId, HeldElementwise>>,
    packed_mapping_subtree: &RefCell<Vec<bool>>,
    root: NodeId,
) -> bool {
    if packed_mapping_subtree
        .borrow()
        .get(root.0 as usize)
        .copied()
        .unwrap_or(false)
    {
        return true;
    }
    let mut pending = vec![root];
    let mut visited = BTreeSet::new();
    while let Some(node) = pending.pop() {
        if !visited.insert(node) {
            continue;
        }
        let Some(entry) = held.borrow().get(&node).cloned() else {
            continue;
        };
        if entry
            .operands
            .iter()
            .any(|(_, map)| map.affine().axes.iter().any(|axis| axis.terms.len() > 1))
        {
            return true;
        }
        pending.extend(entry.operands.iter().map(|(operand, _)| *operand));
    }
    false
}

/// A fusion can compose through: every axis a plain, unshifted projection.
/// Anything richer (a window, a slice, a stride) still resolves correctly,
/// it just materializes its operand instead of composing through it.
fn is_identity_projection(map: &IndexMap) -> bool {
    if map.is_data_dependent() {
        return false;
    }
    map.affine()
        .axes
        .iter()
        .all(|axis| axis.offset == 0 && matches!(axis.terms.as_slice(), [term] if term.coeff == 1))
}

/// The three accumulators every `compose_*` call threads through its
/// recursion — a step list, the flat operand list an executor reads from,
/// and which held nodes this pass consumed — bundled for the same reason
/// [`WindowSpec`] bundles a parameter group: one field group traveling
/// together, not `clippy::too_many_arguments` positional soup.
struct ComposeState<'a> {
    steps: &'a mut Vec<BodyStep>,
    operands: &'a mut BoundOperands,
    absorbed: &'a mut Vec<NodeId>,
}

/// [`BoundOpBuilder`]'s own `ones`/`constant_value` per-node tables, bundled
/// for the same reason [`ComposeState`] bundles its own three fields: every
/// `compose_*` call threads both together (never one without the other), so
/// a bare pair of `&[bool]`/`&[Option<f32>]` parameters is exactly the
/// positional soup `clippy::too_many_arguments` flags at
/// [`build_elementwise_op`]'s own call depth.
#[derive(Clone, Copy)]
struct Constants<'a> {
    ones: &'a [bool],
    values: &'a [Option<f32>],
    /// Threaded through to [`push_canonical_step`] — see
    /// [`BoundOpBuilder`]'s own field of the same name.
    numeric_policy: NumericPolicy,
}

/// Composes the single still-held node `node` — reached from its consumer
/// through `map` — into a [`ComposedBody`] plus the flat, fully-addressed
/// operand list an executor reads from: the reduce-fusion entry point.
/// `node` is guaranteed present in `held` by every caller's own `fuses`
/// check, so this always absorbs at least one op; [`compose_operand`]
/// recurses through however many more are held beneath it.
///
/// [`compose_operand`]'s own ×1.0 elimination can, at THIS call depth only,
/// return a bare [`StepArg::Operand`] with nothing pushed to `steps` —
/// `node` resolved directly to `Multiply(real, Constant(1.0))` with no
/// further absorbing consumer above it (`MaxPool`/`AveragePool`'s own
/// `windowed` node passed straight to `build_reduce`, unlike `Conv`'s
/// `product = windowed * weight`, which always contributes its own step).
/// [`apply_body`](crate::cpu)'s `step_values[body.steps.len() - 1]` requires
/// at least one step, the same invariant the no-fusion branch already
/// guarantees via `ComposedBody::leaf(ScalarOp::Identity)` — so an empty
/// `steps` here gets exactly that: one trailing `Identity` step wrapping
/// the collapsed arg, restoring the invariant without re-introducing the
/// eliminated multiply.
fn compose_fused_operands(
    shapes: &Shapes,
    held: &RefCell<BTreeMap<NodeId, HeldElementwise>>,
    node: NodeId,
    map: &IndexMap,
    constants: Constants<'_>,
) -> (ComposedBody, BoundOperands) {
    let mut steps = Vec::new();
    let mut operands = Vec::new();
    let mut absorbed = Vec::new();
    let mut state = ComposeState {
        steps: &mut steps,
        operands: &mut operands,
        absorbed: &mut absorbed,
    };
    let arg = compose_operand(shapes, held, &mut state, node, map, constants);
    if state.steps.is_empty() {
        push_canonical_step(&mut state, ScalarOp::Identity, alloc::vec![arg], constants);
    }
    drop_absorbed(held, absorbed);
    (ComposedBody { steps }, operands)
}

/// Composes an explicit `body` applied over `operands` — the
/// materialize-a-chain entry point [`build_elementwise_op`] uses, where the
/// top body and its immediate operand list are already in hand (the node
/// itself has already been removed from `held` by its caller).
///
/// Reached not only for a genuinely unfused node but also for one
/// [`quarantine_broadcast_operands`] forced standalone — a held node whose
/// own extent is smaller than its consumer's reduce extent gets materialized
/// here specifically so its body is computed once rather than re-read per
/// broadcast repetition. `window_materialize`'s `windowed` (`image * stamp`)
/// is exactly this shape for `Conv` (its `[n,c,oh,ow,kh,kw]` extent excludes
/// the `co` broadcast axis `product`'s own reduce walks), so this entry
/// point needs the same ×1.0 elimination [`compose_operand`] applies —
/// without it, `windowed` would still materialize with the stamp multiply
/// baked in, `compose_operand`'s own elimination never getting a chance to
/// run because [`held`] no longer holds `windowed` by the time the reduce
/// tries to fuse it (`proxima-tensor/docs/discipline.md` ROW 147).
fn compose(
    shapes: &Shapes,
    held: &RefCell<BTreeMap<NodeId, HeldElementwise>>,
    body: ScalarOp,
    operands: &[(NodeId, IndexMap)],
    constants: Constants<'_>,
) -> (ComposedBody, BoundOperands) {
    let mut steps = Vec::new();
    let mut resolved_operands = Vec::new();
    let mut absorbed = Vec::new();
    let mut state = ComposeState {
        steps: &mut steps,
        operands: &mut resolved_operands,
        absorbed: &mut absorbed,
    };
    let arg = match eliminate_identity_multiply(body, operands, constants.ones) {
        Some((survivor_node, survivor_map)) => compose_operand(
            shapes,
            held,
            &mut state,
            survivor_node,
            survivor_map,
            constants,
        ),
        None => compose_body(shapes, held, &mut state, body, operands, constants),
    };
    if state.steps.is_empty() {
        push_canonical_step(&mut state, ScalarOp::Identity, alloc::vec![arg], constants);
    }
    drop_absorbed(held, absorbed);
    (ComposedBody { steps }, resolved_operands)
}

fn drop_absorbed(held: &RefCell<BTreeMap<NodeId, HeldElementwise>>, absorbed: Vec<NodeId>) {
    let mut held_mut = held.borrow_mut();
    for node in absorbed {
        held_mut.remove(&node);
    }
}

/// A held `Multiply` whose two operands are one real operand and one
/// [`Op::Constant`] of value exactly `1.0` is algebraically a no-op —
/// `x * 1.0 == x` for every finite/inf/nan `f32` bar signaling-NaN quieting
/// (irrelevant to any real weight/activation this crate binds). Returns the
/// surviving operand when this pattern applies, `None` otherwise — the one
/// check both [`compose`] (a node materializing standalone, e.g. under
/// [`quarantine_broadcast_operands`]) and [`compose_operand`] (a node still
/// fusing into its consumer) run before ever pushing a [`BodyStep`], so the
/// constant is dropped from the body regardless of which path reaches it.
/// `window_materialize`'s all-ones shape-inference stamp
/// (`proxima-onnx/src/lower.rs`) is the motivating case, but this is
/// unconditional on which lowering produced the constant, so any future ×1
/// marker gets the same treatment (`proxima-tensor/docs/discipline.md`
/// ROW 147).
fn eliminate_identity_multiply<'a>(
    body: ScalarOp,
    operands: &'a [(NodeId, IndexMap)],
    ones: &[bool],
) -> Option<(NodeId, &'a IndexMap)> {
    if body != ScalarOp::Multiply {
        return None;
    }
    let [(left_node, left_map), (right_node, right_map)] = operands else {
        return None;
    };
    let left_is_one = ones.get(left_node.0 as usize).copied().unwrap_or(false);
    let right_is_one = ones.get(right_node.0 as usize).copied().unwrap_or(false);
    if left_is_one && !right_is_one {
        return Some((*right_node, right_map));
    }
    if right_is_one {
        return Some((*left_node, left_map));
    }
    None
}

/// Reads one held node's `(body, operands)` by value — the same
/// clone-out-of-the-`RefCell` shape [`compose_operand`]'s own `entry` uses,
/// needed here because [`eliminate_masked_window_reduce`] walks three levels
/// of `held` chain while [`held`] itself may need a `borrow_mut` later (to
/// drop the matched nodes), and an outstanding `Ref` would collide with that.
fn held_snapshot(
    held: &RefCell<BTreeMap<NodeId, HeldElementwise>>,
    node: NodeId,
) -> Option<(ScalarOp, Vec<(NodeId, IndexMap)>)> {
    held.borrow()
        .get(&node)
        .map(|entry| (entry.body, entry.operands.clone()))
}

/// One axis is a plain, unshifted single-term projection — the per-axis
/// version of [`is_identity_projection`], needed here because
/// [`eliminate_masked_window_reduce`] inspects individual [`AxisIndex`]
/// entries (a mask's own three selected axes, a source axis) rather than a
/// whole [`IndexMap`] at once.
fn is_pure_axis(axis: &AxisIndex) -> bool {
    axis.offset == 0 && matches!(axis.terms.as_slice(), [term] if term.coeff == 1)
}

/// The stride literal and the three extents [`window_mask`]
/// (`proxima-autograd/src/conv.rs:201`) needs to have built `node`, read back
/// out of the [`BoundOpBuilder`]'s own side channels rather than the source
/// program (this module never holds the whole program, only what is still
/// `held`).
struct WindowMatch {
    stride: u64,
    out_extent: u64,
    kernel_extent: u64,
    source_extent: u64,
    combined_node: NodeId,
    scaled_out_node: NodeId,
}

/// Confirms `node` is exactly `window_mask`'s `Equal(Iota, Add(Multiply(Iota,
/// Constant), Iota))` chain (`proxima-autograd/src/conv.rs:201-223`) and, if
/// so, extracts the stride and the three axis extents the in-bounds proof
/// needs. Any structural mismatch — a different `ScalarOp`, a non-`Iota`
/// operand where one is required, a non-pure-projection edge map, a missing
/// constant — returns `None`, leaving `held` untouched (the caller's own
/// contract).
fn window_mask_match(
    held: &RefCell<BTreeMap<NodeId, HeldElementwise>>,
    node: NodeId,
    is_iota: &[bool],
    constant_value: &[Option<f32>],
    shapes: &Shapes,
) -> Option<WindowMatch> {
    let is_iota_node =
        |candidate: NodeId| is_iota.get(candidate.0 as usize).copied().unwrap_or(false);

    let (equal_body, equal_operands) = held_snapshot(held, node)?;
    if equal_body != ScalarOp::Equal {
        return None;
    }
    let [(source_iota, source_map), (combined_node, combined_map)] = equal_operands.as_slice()
    else {
        return None;
    };
    if !is_iota_node(*source_iota)
        || !is_identity_projection(source_map)
        || !is_identity_projection(combined_map)
    {
        return None;
    }

    let (add_body, add_operands) = held_snapshot(held, *combined_node)?;
    if add_body != ScalarOp::Add {
        return None;
    }
    let [(scaled_out_node, scaled_map), (kernel_iota, kernel_map)] = add_operands.as_slice() else {
        return None;
    };
    if !is_iota_node(*kernel_iota)
        || !is_identity_projection(scaled_map)
        || !is_identity_projection(kernel_map)
    {
        return None;
    }

    let (mul_body, mul_operands) = held_snapshot(held, *scaled_out_node)?;
    if mul_body != ScalarOp::Multiply {
        return None;
    }
    let [(out_iota, out_map), (stride_node, stride_map)] = mul_operands.as_slice() else {
        return None;
    };
    if !is_iota_node(*out_iota)
        || !is_identity_projection(out_map)
        || !is_identity_projection(stride_map)
    {
        return None;
    }
    // `f32::fract`/`round` need `std`'s libm; this crate's alloc tier does
    // not carry a libm dependency, so an exact round-trip through the
    // integer this node's own `stride as f32` construction produced is the
    // no_std-clean way to confirm `stride_value` is a nonnegative integer.
    let stride_value = constant_value
        .get(stride_node.0 as usize)
        .copied()
        .flatten()?;
    if stride_value < 0.0 {
        return None;
    }
    let stride_u64 = stride_value as u64;
    if stride_u64 as f32 != stride_value {
        return None;
    }

    Some(WindowMatch {
        stride: stride_u64,
        out_extent: *shapes.of(*out_iota).first()?,
        kernel_extent: *shapes.of(*kernel_iota).first()?,
        source_extent: *shapes.of(*source_iota).first()?,
        combined_node: *combined_node,
        scaled_out_node: *scaled_out_node,
    })
}

/// The class fix (`proxima-tensor/docs/discipline.md` ROW 147's ×1.0
/// precedent, generalized from a scalar marker to a shaped one): a
/// `Reduce(Add)` whose held operand is `Multiply(source, mask)`, where `mask`
/// is exactly [`window_mask_match`]'s `Equal`/`Iota` chain, is algebraically a
/// plain window read of `source` — for every `(out_position, kernel_position)`
/// pair, at most one `source_position` ever satisfies `source_position ==
/// out_position*stride + kernel_position`, so summing `source *
/// (source_position == that)` over `source_position` is just `source` indexed
/// at that position, proved in-bounds so the read never needs a fallback
/// branch. On a match, the whole `Multiply`/`Equal`/`Add`/`Multiply` subtree
/// is dropped from `held` (it would otherwise still flush as dead work at
/// [`BoundOpBuilder::finish`]) and the caller substitutes a single [`Op::Reduce`]
/// step, computed once per element instead of once per window position, for
/// the [`Op::Reduce`] it never fuses.
///
/// Any mismatch, or a failed in-bounds proof, returns `None` — the caller's
/// existing fuse-or-materialize path runs unchanged, exactly as if this
/// function did not exist.
fn eliminate_masked_window_reduce(
    reduce: &Reduce,
    held: &RefCell<BTreeMap<NodeId, HeldElementwise>>,
    is_iota: &[bool],
    constant_value: &[Option<f32>],
    shapes: &Shapes,
) -> Option<(NodeId, IndexMap)> {
    if reduce.body != ScalarOp::Add
        || reduce.init != ReduceInit::Zero
        || reduce.keep != Keep::Reduce
    {
        return None;
    }
    if !is_identity_projection(&reduce.in_map) {
        return None;
    }

    let (masked_body, masked_operands) = held_snapshot(held, reduce.operand)?;
    if masked_body != ScalarOp::Multiply {
        return None;
    }
    let [(first_node, first_map), (second_node, second_map)] = masked_operands.as_slice() else {
        return None;
    };

    let (source_node, source_map, mask_node, mask_map, window) = if let Some(window) =
        window_mask_match(held, *second_node, is_iota, constant_value, shapes)
    {
        (
            *first_node,
            first_map.clone(),
            *second_node,
            second_map.clone(),
            window,
        )
    } else {
        let window = window_mask_match(held, *first_node, is_iota, constant_value, shapes)?;
        (
            *second_node,
            second_map.clone(),
            *first_node,
            first_map.clone(),
            window,
        )
    };

    if !is_identity_projection(&source_map) {
        return None;
    }
    let mask_pattern = mask_map.affine();
    let [windowed_axis, out_axis, kernel_axis] = mask_pattern.axes.as_slice() else {
        return None;
    };
    if !is_pure_axis(windowed_axis) || !is_pure_axis(out_axis) || !is_pure_axis(kernel_axis) {
        return None;
    }
    let windowed_axis = windowed_axis.terms[0].axis;
    let out_axis = out_axis.terms[0].axis;
    let kernel_axis = kernel_axis.terms[0].axis;

    let last_out = window.out_extent.checked_sub(1)?;
    let last_kernel = window.kernel_extent.checked_sub(1)?;
    let last_read = window
        .stride
        .checked_mul(last_out)?
        .checked_add(last_kernel)?;
    if last_read >= window.source_extent {
        return None;
    }
    let stride_coeff = i32::try_from(window.stride).ok()?;

    let out_pattern = reduce.out_map.affine();
    let keep_axes = pure_projection_axes(out_pattern);
    if keep_axes.len() != out_pattern.axes.len() || keep_axes.contains(&windowed_axis) {
        return None;
    }
    let new_out_position = keep_axes.iter().position(|&axis| axis == out_axis)? as u16;
    let new_kernel_position = keep_axes.iter().position(|&axis| axis == kernel_axis)? as u16;

    let mut new_axes: Vec<AxisIndex> = Vec::with_capacity(source_map.affine().axes.len());
    for axis in &source_map.affine().axes {
        if !is_pure_axis(axis) {
            return None;
        }
        let widened = axis.terms[0].axis;
        let new_axis = if widened == windowed_axis {
            AxisIndex {
                terms: [
                    AxisTerm::scaled(new_out_position, stride_coeff),
                    AxisTerm::scaled(new_kernel_position, 1),
                ]
                .into_iter()
                .collect(),
                offset: 0,
                len: None,
            }
        } else {
            let position = keep_axes.iter().position(|&kept| kept == widened)? as u16;
            AxisIndex {
                terms: core::iter::once(AxisTerm::projection(position)).collect(),
                offset: 0,
                len: None,
            }
        };
        new_axes.push(new_axis);
    }

    held.borrow_mut().remove(&reduce.operand);
    held.borrow_mut().remove(&mask_node);
    held.borrow_mut().remove(&window.combined_node);
    held.borrow_mut().remove(&window.scaled_out_node);

    Some((
        source_node,
        IndexMap::Affine(IndexPattern {
            iter_rank: keep_axes.len() as u16,
            axes: new_axes,
        }),
    ))
}

/// The scalar identity element for `op`'s own [`ScalarOp::is_associative`]
/// class that is bit-exact for EVERY `f32`, including NaN and signed zero —
/// `x * 1.0 == x` always, per IEEE 754 multiplication-by-one. Always
/// admitted regardless of policy
/// ([`NumericRewrite::IdentityElimination`] needs no permission). `Add`,
/// `Maximum`, and `Minimum` also have an algebraic identity element but are
/// NOT bit-exact on every input — see [`identity_element_signed_zero_nan`].
const fn identity_element_bitexact(op: ScalarOp) -> Option<f32> {
    match op {
        ScalarOp::Multiply => Some(1.0),
        _ => None,
    }
}

/// The scalar identity element for `Add`/`Maximum`/`Minimum` — `x + 0.0`,
/// `max(x, -inf)`, `min(x, +inf)` all equal `x` for every FINITE `x`, but not
/// for every `f32`: `max(NaN, -inf)` evaluates to `-inf` ([`f32::max`]'s own
/// "if one argument is NaN, return the other" rule), while eliminating the
/// op would return the survivor, `NaN`; `(-0.0) + 0.0` evaluates to `+0.0`,
/// while eliminating the op would return `-0.0`. `Add` is classified
/// [`NumericRewrite::IdentityEliminationSignedZero`] (needs `signed_zero`
/// alone); `Maximum`/`Minimum` are classified
/// [`NumericRewrite::IdentityEliminationNanAssumption`] (needs
/// `nan_assumptions` alone) — [`identity_element_signed_zero_nan_rewrite`]
/// names which one a given op requires. [`push_canonical_step`] checks this
/// only after [`identity_element_bitexact`] misses, and only fires it once
/// [`admit`] clears the caller's [`NumericPolicy`] for that specific rewrite.
const fn identity_element_signed_zero_nan(op: ScalarOp) -> Option<f32> {
    match op {
        ScalarOp::Add => Some(0.0),
        ScalarOp::Maximum => Some(f32::NEG_INFINITY),
        ScalarOp::Minimum => Some(f32::INFINITY),
        _ => None,
    }
}

/// The specific permission [`identity_element_signed_zero_nan`]'s
/// elimination needs for `op` — `Add`'s zero-literal case needs
/// `signed_zero` alone, `Maximum`/`Minimum`'s infinity-literal case needs
/// `nan_assumptions` alone. The two are independent permissions (a caller
/// may grant one without the other), so this is never a single shared
/// rewrite classification.
const fn identity_element_signed_zero_nan_rewrite(op: ScalarOp) -> Option<NumericRewrite> {
    match op {
        ScalarOp::Add => Some(NumericRewrite::IdentityEliminationSignedZero),
        ScalarOp::Maximum | ScalarOp::Minimum => {
            Some(NumericRewrite::IdentityEliminationNanAssumption)
        }
        _ => None,
    }
}

/// The operand-order sort key [`push_canonical_step`] applies to a
/// commutative op's two args: a fused predecessor ([`StepArg::Step`]) always
/// sorts before a raw operand ([`StepArg::Operand`]), then by index —
/// `false < true` puts every `Step` ahead of every `Operand`. This is what
/// makes `a*b+c` and `c+a*b` mint the identical [`BodyStep`]: whichever
/// operand order the source authored, [`compose_operand`] has already turned
/// each into a `StepArg`, and this key sees only that shape, never the
/// original authoring order.
fn step_arg_sort_key(arg: &StepArg) -> (bool, u16) {
    match *arg {
        StepArg::Step(index) => (false, index),
        StepArg::Operand(index) => (true, index),
    }
}

/// The literal value `arg` resolves to, if it is a [`StepArg::Operand`]
/// built from an [`crate::op::Op::Constant`] leaf — `state.operands[index].0`
/// is that operand's source [`NodeId`] ([`build_operand`]'s own first
/// field), and `constant_value[node.0]` is `Some` exactly when
/// [`BoundOpBuilder::push`] saw that node as a constant. A [`StepArg::Step`]
/// is a computed value, never a known literal at this point in composition,
/// so it always returns `None` here (no recursive constant-folding of a
/// step's own body in this slice).
fn step_arg_constant(
    arg: StepArg,
    state: &ComposeState<'_>,
    constant_value: &[Option<f32>],
) -> Option<f32> {
    match arg {
        StepArg::Operand(index) => {
            let (node, _, _) = state.operands.get(index as usize)?;
            constant_value.get(node.0 as usize).copied().flatten()
        }
        StepArg::Step(_) => None,
    }
}

/// The one place a [`BodyStep`] enters a [`ComposedBody`] — replaces the raw
/// `state.steps.push(BodyStep { .. })` call this module used to make
/// directly. Canonicalizes a commutative binary op's operand order
/// ([`step_arg_sort_key`]) and eliminates an operand equal to `op`'s own
/// identity element before ever minting a step, so two authored orderings
/// of the same algebraic expression — `a*b+c` and `c+a*b`, or a chain with an
/// identity multiply/add folded away by an earlier rewrite — produce the
/// identical [`StepArg`], never a step whose recognizability depends on
/// which rewrite fired first in the same bind call
/// (`proxima-tensor/src/cpu.rs:2570-2626`'s own "hidden=1 confluence gap"
/// doc). Mints no new `ScalarOp`/`Op` variant: every value this returns is
/// either an existing `StepArg` unchanged or a freshly pushed `BodyStep`
/// using `op` exactly as given.
///
/// [`identity_element_bitexact`] (`x*1`) always fires. The remaining three
/// cases (`x+0`, `max(x,-inf)`, `min(x,+inf)`,
/// [`identity_element_signed_zero_nan`]) change bits on NaN/signed-zero
/// inputs and only fire once `constants.numeric_policy` clears the specific
/// rewrite [`identity_element_signed_zero_nan_rewrite`] names via [`admit`]
/// — under the library default ([`NumericPolicy::bit_exact`]) neither ever
/// fires, and a step carrying a `+0`/`max(-inf)`/`min(+inf)` operand
/// survives unreduced.
fn push_canonical_step(
    state: &mut ComposeState<'_>,
    op: ScalarOp,
    mut args: Vec<StepArg>,
    constants: Constants<'_>,
) -> StepArg {
    if op.is_associative() && args.len() == 2 {
        args.sort_by_key(step_arg_sort_key);
    }
    if let [first, second] = args.as_slice() {
        if let Some(identity) = identity_element_bitexact(op) {
            if step_arg_constant(*first, state, constants.values) == Some(identity) {
                return *second;
            }
            if step_arg_constant(*second, state, constants.values) == Some(identity) {
                return *first;
            }
        }
        if let Some(identity) = identity_element_signed_zero_nan(op)
            && let Some(rewrite) = identity_element_signed_zero_nan_rewrite(op)
            && admit(constants.numeric_policy, rewrite).is_ok()
        {
            if step_arg_constant(*first, state, constants.values) == Some(identity) {
                return *second;
            }
            if step_arg_constant(*second, state, constants.values) == Some(identity) {
                return *first;
            }
        }
    }
    state.steps.push(BodyStep { op, args });
    StepArg::Step((state.steps.len() - 1) as u16)
}

/// Composes `body` applied over `body_operands` (expressed in the caller's
/// own iteration space), recursively composing each operand through
/// [`compose_operand`] first, then minting the step through
/// [`push_canonical_step`] — so a step this call mints is already in
/// canonical form, never a second pass over `state.steps`.
///
/// For a commutative binary `body`, `body_operands` is sorted by source
/// [`NodeId`] BEFORE recursing, not only after: `state.operands`'s indices
/// are assigned in visitation order, so `c + a*b` and `a*b + c` would
/// otherwise still number `c`/`a`/`b` differently depending on which
/// authored position each was in, even though `push_canonical_step`'s own
/// post-hoc `StepArg` sort puts the resulting args back in the same
/// `Step`-before-`Operand` shape. Sorting the source pairs first is what
/// makes the two authorings mint byte-identical operand slots, not merely
/// an equivalent argument order — the same "reordering a commutative binary
/// op's operands is exact" guarantee [`push_canonical_step`] documents,
/// applied one level earlier, before any operand slot exists to reorder.
fn compose_body(
    shapes: &Shapes,
    held: &RefCell<BTreeMap<NodeId, HeldElementwise>>,
    state: &mut ComposeState<'_>,
    body: ScalarOp,
    body_operands: &[(NodeId, IndexMap)],
    constants: Constants<'_>,
) -> StepArg {
    let mut ordered: Vec<&(NodeId, IndexMap)> = body_operands.iter().collect();
    if body.is_associative() && ordered.len() == 2 {
        ordered.sort_by_key(|(node, _)| node.0);
    }
    let args = ordered
        .iter()
        .map(|(node, map)| compose_operand(shapes, held, state, *node, map, constants))
        .collect();
    push_canonical_step(state, body, args, constants)
}

/// Composes one operand reference `(node, map)` into `steps`/`operands`:
/// reads it directly from its own buffer when `node` is not (or is no
/// longer) held, or — when it is still held, meaning it satisfied the
/// fusion condition at the exact position that made this its last use —
/// absorbs its own body as one more [`BodyStep`], recursing through
/// however many further levels are held beneath it. `map`'s axes are
/// remapped through [`remap_sub_operands`] before recursing, since a held
/// node's own operand maps are expressed in *its* iteration space, not the
/// caller's. [`eliminate_identity_multiply`] is checked before ever pushing
/// a step — see that function's own doc.
fn compose_operand(
    shapes: &Shapes,
    held: &RefCell<BTreeMap<NodeId, HeldElementwise>>,
    state: &mut ComposeState<'_>,
    node: NodeId,
    map: &IndexMap,
    constants: Constants<'_>,
) -> StepArg {
    let entry = held
        .borrow()
        .get(&node)
        .map(|held_elementwise| (held_elementwise.body, held_elementwise.operands.clone()));

    let Some((body, sub_operands)) = entry else {
        state.operands.push(build_operand(node, map, shapes));
        return StepArg::Operand((state.operands.len() - 1) as u16);
    };

    state.absorbed.push(node);
    let remapped = remap_sub_operands(&sub_operands, map);

    if let Some((survivor_node, survivor_map)) =
        eliminate_identity_multiply(body, &remapped, constants.ones)
    {
        return compose_operand(shapes, held, state, survivor_node, survivor_map, constants);
    }

    compose_body(shapes, held, state, body, &remapped, constants)
}

/// The outer iteration axis each of `map`'s own axes corresponds to — sound
/// only when `map` is [`is_identity_projection`], which every caller here
/// already checked before fusing through it.
fn axis_correspondence(map: &IndexMap) -> Vec<u16> {
    map.affine()
        .axes
        .iter()
        .map(|axis| axis.terms[0].axis)
        .collect()
}

fn remap_pattern(pattern: &IndexPattern, axis_map: &[u16], outer_iter_rank: u16) -> IndexPattern {
    let axes = pattern
        .axes
        .iter()
        .map(|axis_index| AxisIndex {
            terms: axis_index
                .terms
                .iter()
                .map(|term| AxisTerm {
                    axis: axis_map[term.axis as usize],
                    coeff: term.coeff,
                })
                .collect(),
            offset: axis_index.offset,
            len: axis_index.len,
        })
        .collect();
    IndexPattern {
        iter_rank: outer_iter_rank,
        axes,
    }
}

fn remap_index_map(map: &IndexMap, axis_map: &[u16], outer_iter_rank: u16) -> IndexMap {
    match map {
        IndexMap::Affine(pattern) => {
            IndexMap::Affine(remap_pattern(pattern, axis_map, outer_iter_rank))
        }
        IndexMap::Computed {
            indices,
            index_map,
            base,
            gathered_dim,
        } => IndexMap::Computed {
            indices: *indices,
            index_map: remap_pattern(index_map, axis_map, outer_iter_rank),
            base: remap_pattern(base, axis_map, outer_iter_rank),
            gathered_dim: *gathered_dim,
        },
    }
}

/// Composes a held op's own operand maps (expressed in its own iteration
/// space) through `outer_map` — how its consumer reads it, always an
/// identity projection, the fusion precondition — into the consumer's own
/// iteration space. The symbolic counterpart of [`Layout`]-level stride
/// remapping, applied one level per absorbed node so [`compose_operand`]'s
/// recursion composes through as many levels as a chain has.
fn remap_sub_operands(
    sub_operands: &[(NodeId, IndexMap)],
    outer_map: &IndexMap,
) -> Vec<(NodeId, IndexMap)> {
    let axis_map = axis_correspondence(outer_map);
    let outer_iter_rank = outer_map.affine().iter_rank;
    sub_operands
        .iter()
        .map(|(node, map)| (*node, remap_index_map(map, &axis_map, outer_iter_rank)))
        .collect()
}

fn layout_of(pattern: &IndexPattern, operand_shape: &[u64]) -> Layout {
    let element_strides = row_major_strides(operand_shape);
    let mut strides = SmallVec::<[i64; MAX_INLINE_RANK]>::from_elem(0, pattern.iter_rank as usize);
    let mut base = 0i64;
    for (axis_index, axis) in pattern.axes.iter().enumerate() {
        let stride = element_strides[axis_index];
        base += i64::from(axis.offset) * stride;
        for term in &axis.terms {
            strides[term.axis as usize] += i64::from(term.coeff) * stride;
        }
    }
    Layout { base, strides }
}

fn row_major_strides(shape: &[u64]) -> Vec<i64> {
    let mut strides = vec![0i64; shape.len()];
    let mut accumulator = 1i64;
    for (axis_index, extent) in shape.iter().enumerate().rev() {
        strides[axis_index] = accumulator;
        accumulator *= *extent as i64;
    }
    strides
}

/// Rewrites a packed matmul weight operand's [`Layout`] from `layout_of`'s
/// default -- row-major over the operand's own DECLARED axis order -- to the
/// layout its packed bytes actually have on disk.
///
/// `layout_of` has no way to get this right on its own: it sees only the
/// operand's declared shape, the axis order every OTHER consumer of that
/// node agrees the buffer is stored in. For a plain `f32` operand that
/// agreement is real, because the buffer was transposed at bind time to
/// match it (`proxima-model-interop::bind_matmul_weight`'s `F32` fallback,
/// `transpose_out_in_to_in_out`). A packed `Q4_K`/`Q5_K`/`Q6_K` weight is the
/// one case that cannot be transposed to match: a k-quant super-block spans
/// 256 contiguous elements of the contraction axis, so transposing it would
/// mean dequantizing first -- defeating the entire reason to keep it packed.
/// So its declared shape and its physical bytes disagree, and the `Layout`
/// must be rebuilt to describe the bytes, not the declaration.
///
/// GGUF's own on-disk convention for any 2-D weight is `[out_dim, in_dim]`
/// row-major (`out_dim` rows, each a contiguous run of `in_dim` elements) --
/// true of a packed operand regardless of how many logical axes either side
/// is split into on the consuming einsum (`wq`'s `heads`/`head_dim` split is
/// still one flat `embedding x (heads*head_dim)` buffer underneath).
/// `output_axes` on a bound reduce already names which of `extents`'s
/// iteration axes are the "out" side; the complement is "in". Within each
/// side, relative axis order is preserved from the declared shape -- only
/// which side sits inside (contiguous) and which sits outside flips.
///
/// A no-op for any operand not in `packed_operands`, and for any `BoundOp`
/// that is not a `Reduce` (a packed weight only ever reaches this crate as
/// one operand of a `Multiply`-then-`Add` fold -- see
/// `proxima-model-interop::bind_matmul_weight`'s own doc).
pub fn correct_packed_matmul_layouts(resolved: &mut [BoundOp], packed_operands: &BTreeSet<NodeId>) {
    for bound in resolved.iter_mut() {
        let extents = bound.extents.clone();
        let BoundOpKind::Reduce {
            operands,
            output_axes,
            ..
        } = &mut bound.kind
        else {
            continue;
        };
        for (node, layout, _lookup) in operands.iter_mut() {
            if packed_operands.contains(node) {
                *layout = native_packed_layout(&extents, output_axes.as_slice(), layout);
            }
        }
    }
}

/// The stride computation [`correct_packed_matmul_layouts`] applies per
/// operand: `extents`/`output_axes` come from the containing `Reduce`, and
/// `declared` is `layout_of`'s original (wrong-for-packed) `Layout`, read
/// only for its `base` and for which axes it left at stride 0 (a batch axis
/// this operand broadcasts across, e.g. sequence position -- must stay
/// broadcast rather than gain a stride from this reconstruction).
fn native_packed_layout(extents: &[u64], output_axes: &[u16], declared: &Layout) -> Layout {
    let rank = extents.len();
    let mut strides = SmallVec::<[i64; MAX_INLINE_RANK]>::from_elem(0, rank);

    let mut in_dim = 1i64;
    for axis in 0..rank as u16 {
        if !output_axes.contains(&axis) {
            in_dim *= extents[axis as usize] as i64;
        }
    }

    // the reduction ("in") axes: innermost group, relative order preserved,
    // the LAST one contiguous -- exactly `row_major_strides` restricted to
    // this axis subset.
    let mut accumulator = 1i64;
    for axis in (0..rank as u16).rev() {
        if output_axes.contains(&axis) {
            continue;
        }
        strides[axis as usize] = accumulator;
        accumulator *= extents[axis as usize] as i64;
    }

    // the output axes: outermost group, relative order preserved, scaled by
    // the whole reduction group's flat width since it sits inside them.
    let mut accumulator = in_dim;
    for axis in output_axes.iter().rev() {
        strides[*axis as usize] = accumulator;
        accumulator *= extents[*axis as usize] as i64;
    }

    for axis in 0..rank {
        if declared.stride(axis as u16) == 0 {
            strides[axis] = 0;
        }
    }

    Layout {
        base: declared.base,
        strides,
    }
}

/// Batch driver: computes liveness once, then streams every expression
/// through a fresh [`BoundOpBuilder`], flushing whatever remains held at the end.
/// Every node `resolved` physically reads, straight off [`BoundOp::operands()`]
/// plus each gathered operand's own [`Lookup::indices`] — the same walk
/// [`crate::cpu`]'s own execution-time dead-node analysis performs, relocated
/// here so a GPU backend (which has no persistent arena to skip a slot
/// inside) can reuse it too, via [`prune_dead`] below.
fn consumed_by_resolved_nodes(resolved: &[BoundOp]) -> BTreeSet<NodeId> {
    let mut consumed = BTreeSet::new();
    for computed in resolved {
        for (operand, _layout, lookup) in computed.all_read_sources() {
            consumed.insert(*operand);
            if let Some(lookup) = lookup {
                consumed.insert(lookup.indices);
            }
        }
        if let BoundOpKind::Reduce {
            out_scatter: Some(lookup),
            ..
        } = &computed.kind
        {
            consumed.insert(lookup.indices);
        }
    }
    consumed
}

/// Every `resolved` node neither consumed by another resolved node's own
/// operands nor named in `effective_outputs` — dead weight [`bind`]'s own
/// fusion can leave behind (`eliminate_identity_multiply` dropping a
/// [`BoundOpKind::Constant`] from a fused body once its last reader absorbed
/// it is one source; a fused-away [`BoundOpKind::Elementwise`] chain is
/// another). [`crate::cpu::StaticArena`] computes this same set today purely
/// to build its own execution-time skip list — see that type's own `dead`
/// field doc — which hides a real cost from every OTHER backend: a driver
/// with no persistent arena (every GPU backend today) has no skip list to
/// consult, so it dispatches a kernel for a node this function would already
/// tell it nobody reads.
#[must_use]
pub fn dead_resolved_nodes(resolved: &[BoundOp], effective_outputs: &[NodeId]) -> BTreeSet<NodeId> {
    let consumed = consumed_by_resolved_nodes(resolved);
    let dead: BTreeSet<NodeId> = resolved
        .iter()
        .map(|computed| computed.node)
        .filter(|node| !consumed.contains(node) && !effective_outputs.contains(node))
        .collect();
    #[cfg(feature = "instrument")]
    for node in &dead {
        debug!(
            node = node.0,
            kind = "dead_resolved_node",
            decision = "dead",
            "resolved node has zero consumers and is not a requested output"
        );
    }
    dead
}

/// Drops every [`dead_resolved_nodes`] entry from `resolved` — the one
/// GPU-facing counterpart [`crate::cpu::StaticArena`]'s own skip-at-execution
/// trick has no analogue for. A stateless driver (Metal/CUDA/wgpu today) has
/// no persistent arena to skip a slot inside between calls, so the only way
/// to avoid dispatching a dead node's kernel is to never hand it to the
/// driver's own dispatch list at all. A no-op (identity on `resolved`,
/// zero-cost when nothing is dead) unless [`dead_resolved_nodes`] finds
/// something to drop.
#[must_use]
pub fn prune_dead(resolved: Vec<BoundOp>, effective_outputs: &[NodeId]) -> Vec<BoundOp> {
    let dead = dead_resolved_nodes(&resolved, effective_outputs);
    if dead.is_empty() {
        return resolved;
    }
    resolved
        .into_iter()
        .filter(|computed| !dead.contains(&computed.node))
        .collect()
}

#[cfg(feature = "cached-attention-streaming")]
fn elementwise_operands(
    program: &[Op],
    node: NodeId,
    body: ScalarOp,
) -> Option<&[(NodeId, IndexMap)]> {
    match program.get(node.0 as usize)? {
        Op::Elementwise {
            body: actual_body,
            operands,
            ..
        } if *actual_body == body => Some(operands),
        _ => None,
    }
}

#[cfg(feature = "cached-attention-streaming")]
fn binary_elementwise(program: &[Op], node: NodeId, body: ScalarOp) -> Option<[NodeId; 2]> {
    let operands = elementwise_operands(program, node, body)?;
    let [(left, _), (right, _)] = operands else {
        return None;
    };
    Some([*left, *right])
}

#[cfg(feature = "cached-attention-streaming")]
fn unary_elementwise(program: &[Op], node: NodeId, body: ScalarOp) -> Option<NodeId> {
    let operands = elementwise_operands(program, node, body)?;
    let [(source, _)] = operands else {
        return None;
    };
    Some(*source)
}

#[cfg(feature = "cached-attention-streaming")]
fn reduced_source(
    program: &[Op],
    node: NodeId,
    body: ScalarOp,
    init: ReduceInit,
) -> Option<NodeId> {
    match program.get(node.0 as usize)? {
        Op::Reduce(reduce)
            if reduce.body == body && reduce.init == init && reduce.keep == Keep::Reduce =>
        {
            Some(reduce.operand)
        }
        _ => None,
    }
}

#[cfg(feature = "cached-attention-streaming")]
fn constant_value(program: &[Op], node: NodeId) -> Option<f32> {
    match program.get(node.0 as usize)? {
        Op::Constant { value, .. } => Some(*value),
        _ => None,
    }
}

#[cfg(feature = "cached-attention-streaming")]
fn decode_rotary_terms(program: &[Op], terms: [NodeId; 2]) -> Option<(NodeId, NodeId, NodeId, NodeId)> {
    let even_product = reduced_source(program, terms[0], ScalarOp::Add, ReduceInit::Zero)?;
    let odd_product = reduced_source(program, terms[1], ScalarOp::Add, ReduceInit::Zero)?;
    let even_operands = binary_elementwise(program, even_product, ScalarOp::Multiply)?;
    let odd_operands = binary_elementwise(program, odd_product, ScalarOp::Multiply)?;
    Some((
        even_operands[0],
        odd_operands[0],
        even_operands[1],
        odd_operands[1],
    ))
}

/// One un-rotated pass-plane term (qwen35's `score_cached_pass`/
/// `score_new_pass`, `spec.rs:4910-4927,5035-5049`): a bare
/// `reduced(query_pass_grouped * key_pass)`, no even/odd split because the
/// pass plane is never rotated. Returns `(query_pass_grouped, key_pass)`.
#[cfg(feature = "cached-attention-streaming")]
fn decode_pass_term(program: &[Op], node: NodeId) -> Option<(NodeId, NodeId)> {
    let pass_product = reduced_source(program, node, ScalarOp::Add, ReduceInit::Zero)?;
    let pass_operands = binary_elementwise(program, pass_product, ScalarOp::Multiply)?;
    Some((pass_operands[0], pass_operands[1]))
}

/// `score = Multiply(Add(rotary_sum, pass_sum), scale)` when a partial-rotary
/// pass plane is present (qwen35's chain, `spec.rs`'s own `score_cached`/
/// `score_new`), `score = Multiply(Add(even, odd), scale)` otherwise (every
/// other caller today, `rotary_dim == head_dim`). Both shapes share the outer
/// `Multiply`-by-`scale`; only the sum operand's own shape differs, so this
/// tries the nested three-term interpretation first and falls back to the
/// flat two-term one. Returns
/// `(query_even_grouped, query_odd_grouped, key_even, key_odd, pass)`, where
/// `pass` is `Some((query_pass_grouped, key_pass))` only for the nested shape.
#[cfg(feature = "cached-attention-streaming")]
fn attention_score_sources(
    program: &[Op],
    score: NodeId,
    scale: NodeId,
) -> Option<AttentionScoreSources> {
    let scaled = binary_elementwise(program, score, ScalarOp::Multiply)?;
    if constant_value(program, scaled[1]) != constant_value(program, scale)
        || constant_value(program, scaled[1]).is_none()
    {
        return None;
    }
    let outer = binary_elementwise(program, scaled[0], ScalarOp::Add)?;
    if let Some(rotary_terms) = binary_elementwise(program, outer[0], ScalarOp::Add)
        && let Some(rotary) = decode_rotary_terms(program, rotary_terms)
        && let Some(pass) = decode_pass_term(program, outer[1])
    {
        return Some((rotary.0, rotary.1, rotary.2, rotary.3, Some(pass)));
    }
    let rotary = decode_rotary_terms(program, outer)?;
    Some((rotary.0, rotary.1, rotary.2, rotary.3, None))
}

/// `true` when `node` is exactly [`Op::Iota`] -- the raw key/query index a
/// causal or padding mask compares against.
#[cfg(feature = "cached-attention-streaming")]
fn is_iota(program: &[Op], node: NodeId) -> bool {
    matches!(program.get(node.0 as usize), Some(Op::Iota { .. }))
}

/// The `cached_len` bound a padding predicate excludes rows at-or-past, when
/// `node` is exactly `Greater(Iota, Subtract(cached_len, one))` -- `x > n - 1`
/// excludes exactly `x >= n`, and qwen35's own builder (`spec.rs:4973-4987`,
/// `is_cached_padding`) emits precisely this shape. Anything else returns
/// `None` -- the caller declines the fusion rather than guessing at an
/// unfamiliar predicate.
#[cfg(feature = "cached-attention-streaming")]
fn cached_len_padding_bound(program: &[Op], node: NodeId) -> Option<NodeId> {
    let operands = binary_elementwise(program, node, ScalarOp::Greater)?;
    if !is_iota(program, operands[0]) {
        return None;
    }
    let shifted = binary_elementwise(program, operands[1], ScalarOp::Subtract)?;
    (constant_value(program, shifted[1]) == Some(1.0)).then_some(shifted[0])
}

/// Walks past qwen35's own padding mask (`spec.rs:4979-4997`,
/// `is_cached_padding` selecting `-inf` for `key_index >= cached_len`) to the
/// unmasked scaled score underneath, returning `None` (decline the fusion)
/// unless `node` is exactly `Select(padding_predicate, -inf, inner)` AND the
/// predicate's own bound is the SAME `cached_len` leaf the fused op's runtime
/// `cached_key_rows` clip already reads (`cpu.rs:6944-6974`) -- that clip
/// excludes exactly the rows this mask would have scored `-inf`, which is
/// what makes dropping the mask node sound rather than a silent behavior
/// change.
#[cfg(feature = "cached-attention-streaming")]
fn unwrap_cached_padding_select(
    program: &[Op],
    node: NodeId,
    cached_len: Option<NodeId>,
) -> Option<NodeId> {
    let operands = elementwise_operands(program, node, ScalarOp::Select)?;
    let [(predicate, _), (negative_infinity, _), (inner, _)] = operands else {
        return None;
    };
    if constant_value(program, *negative_infinity) != Some(f32::NEG_INFINITY) {
        return None;
    }
    let bound = cached_len_padding_bound(program, *predicate)?;
    (Some(bound) == cached_len).then_some(*inner)
}

#[cfg(feature = "cached-attention-streaming")]
fn is_exact_causal_mask(program: &[Op], node: NodeId) -> bool {
    let Some(operands) = elementwise_operands(program, node, ScalarOp::Greater) else {
        return false;
    };
    let [(key, key_map), (query, query_map)] = operands else {
        return false;
    };
    if *key_map != IndexMap::Affine(map::projection(2, &[1]))
        || *query_map != IndexMap::Affine(map::projection(2, &[0]))
    {
        return false;
    }
    matches!(
        (program.get(key.0 as usize), program.get(query.0 as usize)),
        (Some(Op::Iota { .. }), Some(Op::Iota { .. }),)
    )
}

/// [`is_exact_causal_mask`]'s counterpart for
/// [`crate::spec::causal_mask_merged`]'s shape: the key side is still a bare
/// `Iota`, but the query side is `query_index + cached_len` (an
/// [`ScalarOp::Add`]) rather than a bare `Iota`, because a single-range
/// query at local position `s` sits at absolute position `cached_len + s`
/// once its own new keys are folded into the one merged range. `cached_len`
/// itself is a per-call [`crate::op::Op::Input`] (`causal_mask_merged`'s own
/// doc), never structurally checked here — only that the query side is a
/// shift of an `Iota`, which is what makes the mask exact causal rather than
/// an arbitrary comparison.
/// Returns the `cached_len` [`Op::Input`] node the mask's query side shifts
/// an `Iota` by, when `node` is exactly
/// [`crate::spec::causal_mask_merged`]'s shape — `None` for anything else.
/// The caller needs this NodeId, not just a bool: `cached_len` is a per-call
/// runtime scalar (see this function's own doc below), and the fused
/// [`BoundOpKind::CachedAttention`] this feeds must read the band bound from
/// that scalar at execution time rather than baking a value derived from
/// bound EXTENTS, which drifts from the true `cached_len` whenever the KV
/// extent is padded past the merged length (`kv-capacity-bucket`).
#[cfg(feature = "cached-attention-streaming")]
fn exact_merged_causal_mask_cached_len(program: &[Op], node: NodeId) -> Option<NodeId> {
    let operands = elementwise_operands(program, node, ScalarOp::Greater)?;
    let [(key, key_map), (query_absolute, query_map)] = operands else {
        return None;
    };
    if *key_map != IndexMap::Affine(map::projection(2, &[1]))
        || *query_map != IndexMap::Affine(map::projection(2, &[0]))
        || !matches!(program.get(key.0 as usize), Some(Op::Iota { .. }))
    {
        return None;
    }
    let shift_operands = elementwise_operands(program, *query_absolute, ScalarOp::Add)?;
    let [(query_index, _), (cached_len, _)] = shift_operands else {
        return None;
    };
    if !matches!(program.get(query_index.0 as usize), Some(Op::Iota { .. })) {
        return None;
    }
    matches!(program.get(cached_len.0 as usize), Some(Op::Input { .. })).then_some(*cached_len)
}

/// The rank-0 [`Op::Input`] leaf named `name`, found by NAME rather than by
/// arithmetic shape — the precedent [`Op::Input`]'s own doc states
/// (`"name is identity, not decoration"`): a distributed cut edge delivers a
/// tensor over a wire keyed by name, and this is the same lookup, run
/// locally. [`cached_attention_candidates`]'s own `cached_len` operand needs
/// this rather than [`exact_merged_causal_mask_cached_len`]'s mask-arithmetic
/// walk because a two-range program's cached-range attention feeds no
/// arithmetic from `cached_len` at all — the bucket's padding is excluded by
/// a runtime BOUND on the fused op, never by a mask node in this graph, so
/// there is no expression here to walk backward from.
#[cfg(feature = "cached-attention-streaming")]
fn find_named_input(program: &[Op], name: &str) -> Option<NodeId> {
    program.iter().enumerate().find_map(|(position, op)| {
        let is_named = matches!(op, Op::Input { .. }) && op.name() == Some(name);
        is_named.then_some(NodeId(position as u32))
    })
}

#[cfg(feature = "cached-attention-streaming")]
fn cached_attention_candidates(
    program: &[Op],
    shapes: &Shapes,
    resolved: &[BoundOp],
    effective_outputs: &[NodeId],
    require_output_resolved: bool,
) -> Vec<(BoundOp, BTreeSet<NodeId>)> {
    let mut candidates = Vec::new();
    // resolved once: every caller supplies this leaf unconditionally
    // (`find_named_input`'s own doc), and both the padding-select walk below
    // and the ninth-operand push near the end of this loop need the SAME
    // node identity to agree it is the one true `cached_len`.
    let named_cached_len = find_named_input(program, "cached_len");
    for output_position in (0..program.len()).rev() {
        let output = NodeId(output_position as u32);
        let Some(attended_sum) = binary_elementwise(program, output, ScalarOp::Multiply) else {
            continue;
        };
        let Some(attended_parts) = binary_elementwise(program, attended_sum[0], ScalarOp::Add)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "not_online_softmax_add",
                "cached_attention decline -- weighted-value numerator is not an Add"
            );
            continue;
        };
        let Some(inverse_sum) = unary_elementwise(program, attended_sum[1], ScalarOp::Reciprocal)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "not_reciprocal_denominator",
                "cached_attention decline -- weighted-value denominator is not a Reciprocal"
            );
            continue;
        };
        let Some(sum_parts) = binary_elementwise(program, inverse_sum, ScalarOp::Add) else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "not_denominator_add",
                "cached_attention decline -- reciprocal source is not an Add"
            );
            continue;
        };
        let Some(cached_weights) =
            reduced_source(program, sum_parts[0], ScalarOp::Add, ReduceInit::Zero)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "not_cached_weight_reduce",
                "cached_attention decline -- cached-side denominator term is not a zero-init Add reduce"
            );
            continue;
        };
        let Some(new_weights) =
            reduced_source(program, sum_parts[1], ScalarOp::Add, ReduceInit::Zero)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "not_new_weight_reduce",
                "cached_attention decline -- new-side denominator term is not a zero-init Add reduce"
            );
            continue;
        };
        let Some(cached_shift) = unary_elementwise(program, cached_weights, ScalarOp::Exponential)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "not_cached_shift_exp",
                "cached_attention decline -- cached-side weight is not an Exponential"
            );
            continue;
        };
        let Some(new_shift) = unary_elementwise(program, new_weights, ScalarOp::Exponential) else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "not_new_shift_exp",
                "cached_attention decline -- new-side weight is not an Exponential"
            );
            continue;
        };
        let Some(cached_score_parts) =
            binary_elementwise(program, cached_shift, ScalarOp::Subtract)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "not_cached_score_subtract",
                "cached_attention decline -- cached-side shift source is not a Subtract"
            );
            continue;
        };
        let Some(new_score_parts) = binary_elementwise(program, new_shift, ScalarOp::Subtract)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "not_new_score_subtract",
                "cached_attention decline -- new-side shift source is not a Subtract"
            );
            continue;
        };
        if cached_score_parts[1] != new_score_parts[1] {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "score_denominator_mismatch",
                "cached_attention decline -- cached/new score denominators diverge"
            );
            continue;
        }
        let new_masked = new_score_parts[0];
        let Some(mask_parts) = elementwise_operands(program, new_masked, ScalarOp::Select) else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "mask_select_shape",
                "cached_attention decline -- new score is not a Select mask node"
            );
            continue;
        };
        let [(mask, _), (negative_infinity, _), (new_scaled, _)] = mask_parts else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "mask_select_arity",
                "cached_attention decline -- mask Select does not carry exactly 3 operands"
            );
            continue;
        };
        if !is_exact_causal_mask(program, *mask)
            || constant_value(program, *negative_infinity) != Some(f32::NEG_INFINITY)
        {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "mask_form",
                is_causal = is_exact_causal_mask(program, *mask),
                "cached_attention decline -- mask is not the exact causal form"
            );
            continue;
        }
        // qwen35's own chain masks cached-range padding with a `Select`
        // right here (`spec.rs:4979-4997`, `is_cached_padding`) before the
        // online-softmax subtract this matcher already walked past above --
        // the fused op's own runtime `cached_key_rows` clip
        // (`cpu.rs:6944-6974`) excludes exactly those rows, so dropping the
        // mask node is sound whenever its bound is the SAME `cached_len`
        // leaf the ninth operand below reads.
        let cached_scaled_source =
            unwrap_cached_padding_select(program, cached_score_parts[0], named_cached_len)
                .unwrap_or(cached_score_parts[0]);
        let Some(cached_scaled_parts) =
            binary_elementwise(program, cached_scaled_source, ScalarOp::Multiply)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "cached_scale_shape",
                "cached_attention decline -- cached padding-unwrapped score is not a scale Multiply"
            );
            continue;
        };
        let scale = cached_scaled_parts[1];
        let Some(new_scaled_parts) = binary_elementwise(program, *new_scaled, ScalarOp::Multiply)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "new_scale_shape",
                "cached_attention decline -- masked new score is not a scale Multiply"
            );
            continue;
        };
        if new_scaled_parts[1] != scale {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "scale_mismatch",
                "cached_attention decline -- cached and new score use different scale constants"
            );
            continue;
        }
        let Some((query_even_grouped, query_odd_grouped, cached_key_even, cached_key_odd, cached_pass)) =
            attention_score_sources(program, cached_scaled_source, scale)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "cached_score_sources",
                "cached_attention decline -- cached score does not decompose into the qwen35 q.k score-source shape"
            );
            continue;
        };
        let Some((new_query_even_grouped, new_query_odd_grouped, new_key_even, new_key_odd, new_pass)) =
            attention_score_sources(program, *new_scaled, scale)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "new_score_sources",
                "cached_attention decline -- new score does not decompose into the qwen35 q.k score-source shape"
            );
            continue;
        };
        if new_query_even_grouped != query_even_grouped
            || new_query_odd_grouped != query_odd_grouped
        {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "query_identity_mismatch",
                "cached_attention decline -- cached and new score read different query nodes"
            );
            continue;
        }
        let Some(query_even_parts) =
            binary_elementwise(program, query_even_grouped, ScalarOp::Multiply)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "query_even_group_shape",
                "cached_attention decline -- grouped query-even is not a Multiply (group broadcast) node"
            );
            continue;
        };
        let Some(query_odd_parts) =
            binary_elementwise(program, query_odd_grouped, ScalarOp::Multiply)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "query_odd_group_shape",
                "cached_attention decline -- grouped query-odd is not a Multiply (group broadcast) node"
            );
            continue;
        };
        if query_even_parts[1] != query_odd_parts[1] {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "group_broadcast_mismatch",
                "cached_attention decline -- query-even/odd use different group-broadcast operands"
            );
            continue;
        }
        let query_even = query_even_parts[0];
        let query_odd = query_odd_parts[0];
        // A pass plane must appear on BOTH the cached and new score, or not
        // at all -- qwen35's own builder always emits it on both sides
        // (`spec.rs:4910-4927,5035-5049`), so a mismatch here means this
        // program is not that shape.
        if cached_pass.is_some() != new_pass.is_some() {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "pass_presence_mismatch",
                cached_has_pass = cached_pass.is_some(),
                new_has_pass = new_pass.is_some(),
                "cached_attention decline -- pass plane present on one side of cached/new score only"
            );
            continue;
        }
        let pass = match (cached_pass, new_pass) {
            (Some((cached_query_pass_grouped, cached_key_pass)), Some((new_query_pass_grouped, new_key_pass))) => {
                if cached_query_pass_grouped != new_query_pass_grouped {
                    #[cfg(feature = "instrument")]
                    debug!(
                        node = output.0,
                        stage = "pass_query_identity_mismatch",
                        "cached_attention decline -- cached and new score read different pass-plane query nodes"
                    );
                    continue;
                }
                let Some(query_pass_parts) =
                    binary_elementwise(program, cached_query_pass_grouped, ScalarOp::Multiply)
                else {
                    #[cfg(feature = "instrument")]
                    debug!(
                        node = output.0,
                        stage = "pass_query_group_shape",
                        "cached_attention decline -- grouped pass-plane query is not a Multiply (group broadcast) node"
                    );
                    continue;
                };
                if query_pass_parts[1] != query_even_parts[1] {
                    #[cfg(feature = "instrument")]
                    debug!(
                        node = output.0,
                        stage = "pass_group_broadcast_mismatch",
                        "cached_attention decline -- pass-plane query uses a different group-broadcast operand than q_even"
                    );
                    continue;
                }
                Some((query_pass_parts[0], cached_key_pass, new_key_pass))
            }
            _ => None,
        };
        let Some(cached_value_source) =
            reduced_source(program, attended_parts[0], ScalarOp::Add, ReduceInit::Zero)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "cached_value_reduce_shape",
                "cached_attention decline -- cached attended-value term is not a zero-init Add reduce"
            );
            continue;
        };
        let Some(new_value_source) =
            reduced_source(program, attended_parts[1], ScalarOp::Add, ReduceInit::Zero)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "new_value_reduce_shape",
                "cached_attention decline -- new attended-value term is not a zero-init Add reduce"
            );
            continue;
        };
        let Some(cached_value_product) =
            binary_elementwise(program, cached_value_source, ScalarOp::Multiply)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "cached_value_product_shape",
                "cached_attention decline -- cached value-weight term is not a Multiply node"
            );
            continue;
        };
        let Some(new_value_product) =
            binary_elementwise(program, new_value_source, ScalarOp::Multiply)
        else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "new_value_product_shape",
                "cached_attention decline -- new value-weight term is not a Multiply node"
            );
            continue;
        };
        let cached_value = cached_value_product[1];
        let new_value = new_value_product[1];
        if cached_value_product[0] != cached_weights || new_value_product[0] != new_weights {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "value_weight_identity_mismatch",
                "cached_attention decline -- value product does not multiply against this branch's own softmax weight"
            );
            continue;
        }
        let mut source_nodes = alloc::vec![
            query_even,
            query_odd,
            cached_key_even,
            cached_key_odd,
            new_key_even,
            new_key_odd,
            cached_value,
            new_value,
        ];
        if let Some((pass_query, pass_cached_key, pass_new_key)) = pass {
            source_nodes.extend([pass_query, pass_cached_key, pass_new_key]);
        }
        let mut operands = Vec::with_capacity(source_nodes.len());
        for source in &source_nodes {
            let Some((_, layout, lookup)) = resolved
                .iter()
                .flat_map(|bound| bound.operands().iter())
                .find(|(node, _, _)| node == source)
            else {
                #[cfg(feature = "instrument")]
                debug!(
                    node = output.0,
                    stage = "source_not_found",
                    source = source.0,
                    "cached_attention decline -- a score/value source node is not an operand of any resolved op"
                );
                operands.clear();
                break;
            };
            if lookup.is_some() || layout.strides.iter().any(|stride| *stride < 0) {
                #[cfg(feature = "instrument")]
                debug!(
                    node = output.0,
                    stage = "source_indirect_or_negative_stride",
                    source = source.0,
                    has_lookup = lookup.is_some(),
                    "cached_attention decline -- a score/value source is gathered indirectly or carries a negative stride"
                );
                operands.clear();
                break;
            }
            operands.push((*source, layout.clone(), None));
        }
        if operands.len() != source_nodes.len() {
            continue;
        }
        let Some(scale_value) = constant_value(program, scale) else {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "scale_not_constant",
                "cached_attention decline -- the score scale operand is not a compile-time constant"
            );
            continue;
        };
        let query_shape = shapes.of(query_even_grouped);
        let cached_key_shape = shapes.of(cached_key_even);
        let new_key_shape = shapes.of(new_key_even);
        let cached_value_shape = shapes.of(cached_value);
        let new_value_shape = shapes.of(new_value);
        let Some(rotary_width) = query_shape[3].checked_mul(2) else {
            continue;
        };
        // `total_head_dim` is `rotary_width` whenever no pass plane is
        // present (every non-qwen35 caller today) -- V is never rotated, so
        // its own width is the one place the pass plane's extra columns
        // surface even when the rotary planes alone would say `rotary_width`
        // (`BoundOpKind::CachedAttention`'s own doc).
        let total_head_dim = match pass {
            Some((pass_query, _, _)) => {
                let Some(&pass_dim) = shapes.of(pass_query).last() else {
                    continue;
                };
                let Some(total) = rotary_width.checked_add(pass_dim) else {
                    continue;
                };
                total
            }
            None => rotary_width,
        };
        if query_shape.len() != 4
            || cached_key_shape.len() != 3
            || new_key_shape.len() != 3
            || cached_value_shape.len() != 3
            || new_value_shape.len() != 3
            || query_shape[1] != cached_key_shape[1]
            || query_shape[1] != new_key_shape[1]
            || cached_key_shape[1] != cached_value_shape[1]
            || new_key_shape[1] != new_value_shape[1]
            || cached_key_shape[0] != cached_value_shape[0]
            || new_key_shape[0] != new_value_shape[0]
            || cached_value_shape[2] != total_head_dim
            || new_value_shape[2] != total_head_dim
            || shapes.of(output) != [query_shape[0], query_shape[1], query_shape[2], total_head_dim]
        {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "shape_checks",
                ?query_shape,
                ?cached_key_shape,
                ?new_key_shape,
                ?cached_value_shape,
                ?new_value_shape,
                total_head_dim,
                output_shape = ?shapes.of(output),
                "cached_attention decline -- query/key/value/output shapes do not agree on kv_heads/head_dim"
            );
            continue;
        }
        let pair_dim = query_shape[3];
        let query_strides = [
            (query_shape[1] * query_shape[2] * pair_dim) as i64,
            (query_shape[2] * pair_dim) as i64,
            pair_dim as i64,
            1i64,
        ];
        let key_strides = [
            0i64,
            (query_shape[1] * pair_dim) as i64,
            pair_dim as i64,
            0,
            1,
        ];
        let value_strides = [
            0i64,
            (query_shape[1] * total_head_dim) as i64,
            total_head_dim as i64,
            0,
            1,
        ];
        if operands[0].1.strides.as_slice() != query_strides
            || operands[1].1.strides.as_slice() != query_strides
            || operands[2].1.strides.as_slice() != key_strides
            || operands[3].1.strides.as_slice() != key_strides
            || operands[4].1.strides.as_slice() != key_strides
            || operands[5].1.strides.as_slice() != key_strides
            || operands[6].1.strides.as_slice() != value_strides
            || operands[7].1.strides.as_slice() != value_strides
        {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "base_strides",
                ?query_strides,
                ?key_strides,
                ?value_strides,
                got = ?operands[..8].iter().map(|(_, layout, _)| layout.strides.clone()).collect::<Vec<_>>(),
                "cached_attention decline -- the base 8 operands do not carry the fused kernel's assumed GEMM strides"
            );
            continue;
        }
        if let Some((_, pass_cached_key, pass_new_key)) = pass {
            let pass_dim = total_head_dim - rotary_width;
            let pass_query_strides = [
                (query_shape[1] * query_shape[2] * pass_dim) as i64,
                (query_shape[2] * pass_dim) as i64,
                pass_dim as i64,
                1i64,
            ];
            let pass_key_strides = [
                0i64,
                (query_shape[1] * pass_dim) as i64,
                pass_dim as i64,
                0,
                1,
            ];
            let cached_key_pass_shape = shapes.of(pass_cached_key);
            let new_key_pass_shape = shapes.of(pass_new_key);
            if cached_key_pass_shape.len() != 3
                || new_key_pass_shape.len() != 3
                || cached_key_pass_shape[1] != query_shape[1]
                || new_key_pass_shape[1] != query_shape[1]
                || cached_key_pass_shape[2] != pass_dim
                || new_key_pass_shape[2] != pass_dim
                || cached_key_pass_shape[0] != cached_key_shape[0]
                || new_key_pass_shape[0] != new_key_shape[0]
                || operands[8].1.strides.as_slice() != pass_query_strides
                || operands[9].1.strides.as_slice() != pass_key_strides
                || operands[10].1.strides.as_slice() != pass_key_strides
            {
                #[cfg(feature = "instrument")]
                debug!(
                    node = output.0,
                    stage = "pass_strides",
                    ?pass_query_strides,
                    ?pass_key_strides,
                    got = ?operands[8..11].iter().map(|(_, layout, _)| layout.strides.clone()).collect::<Vec<_>>(),
                    ?cached_key_pass_shape,
                    ?new_key_pass_shape,
                    "cached_attention decline -- the pass-plane operands do not carry the fused kernel's assumed strides"
                );
                continue;
            }
        }
        // The pass triple is set aside here and re-appended AFTER the
        // optional `cached_len` push below -- `cpu.rs:6913-6917`'s own
        // `pass_start` reads the pass plane at index 8 when `cached_len` is
        // absent and index 9 when present, never at a fixed offset from the
        // base eight.
        let pass_operands = if pass.is_some() {
            Some(operands.split_off(8))
        } else {
            None
        };
        let dependencies = attention_dependencies(program, output, &source_nodes);
        let dependencies = dependencies
            .difference(&source_nodes.into_iter().collect())
            .copied()
            .collect::<BTreeSet<_>>();
        if dependencies
            .iter()
            .any(|node| effective_outputs.contains(node))
        {
            #[cfg(feature = "instrument")]
            for node in &dependencies {
                if effective_outputs.contains(node) {
                    debug!(
                        node = node.0,
                        kind = "cached_attention_absorption",
                        decision = "rejected_requested_output",
                        into = output.0,
                        "attention mask dependency not absorbed -- it is a requested output"
                    );
                }
            }
            continue;
        }
        let absorbed = removable_attention_dependencies(program, &dependencies, output);
        if absorbed.is_empty() {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "no_removable_dependencies",
                "cached_attention decline -- no intermediate ops become dead once this fusion absorbs its sources"
            );
            continue;
        }
        if require_output_resolved && !resolved.iter().any(|bound| bound.node == output) {
            #[cfg(feature = "instrument")]
            debug!(
                node = output.0,
                stage = "output_not_resolved",
                "cached_attention decline -- the candidate output node has no resolved binding"
            );
            continue;
        }
        // Every caller of `mistral_cached_forward_program_with_experts`
        // supplies a rank-0 "cached_len" `Op::Input` unconditionally
        // (`find_named_input`'s own doc) -- when a program predates that
        // (a hand-built test fixture with no such leaf), fall back to the
        // eight-operand, unbounded shape rather than erroring: today's
        // behavior for every caller that never opted into bucketing.
        // `cached_key_shape[0] == 0` (the very first decode step, before any
        // token is cached) is skipped even when the leaf exists: an empty
        // cached range has no padding to exclude, and giving it the ninth
        // operand anyway would make its `cached_key_rows == 0` collide with
        // `single_range_dynamic`'s own discriminator (`BoundOpKind::
        // CachedAttention`'s own doc) -- the two shapes are structurally
        // indistinguishable at that value, so this is the one case that
        // must stay eight-operand regardless of bucketing.
        if cached_key_shape[0] > 0
            && let Some(cached_len_node) = named_cached_len
            && shapes.of(cached_len_node).is_empty()
        {
            operands.push((
                cached_len_node,
                Layout {
                    base: 0,
                    strides: SmallVec::new(),
                },
                None,
            ));
        }
        if let Some(pass_operands) = pass_operands {
            operands.extend(pass_operands);
        }
        let fused = BoundOp {
            node: output,
            dtype: DType::Float32,
            extents: shapes.of(output).to_vec(),
            kind: BoundOpKind::CachedAttention {
                operands,
                query_rows: query_shape[0],
                cached_key_rows: cached_key_shape[0],
                new_key_rows: new_key_shape[0],
                kv_heads: query_shape[1],
                query_groups: query_shape[2],
                head_dim: total_head_dim,
                // `rotary_width` whenever no pass plane matched (every
                // non-qwen35 caller, `total_head_dim == rotary_width`);
                // qwen35's own partial-rotary chain sets this strictly
                // below `head_dim` (`attention_score_sources`'s own doc).
                rotary_dim: rotary_width,
                scale: scale_value,
                cached_lower_inclusive: i64::MIN,
                new_upper_inclusive: 0,
            },
        };
        #[cfg(feature = "instrument")]
        debug!(
            node = output.0,
            require_output_resolved,
            kv_heads = query_shape[1],
            head_dim = total_head_dim,
            "cached_attention candidate accepted"
        );
        candidates.push((fused, absorbed));
    }
    candidates
}

/// [`cached_attention_candidates`]'s counterpart for
/// [`crate::spec::append_mistral_single_range_cached_layer`]'s output shape:
/// one merged key/value range instead of a cached/new pair, so there is no
/// online-softmax combine to unwind — `attended` is a plain single-pass
/// softmax over one masked score matrix
/// (`score_even+score_odd` -> mask -> max -> sub+exp -> sum -> reciprocal ->
/// multiply -> weight the one value range), the same eight-step chain
/// [`crate::spec::append_mistral_layer`] emits for a from-scratch (no cache)
/// forward pass. The fused [`BoundOpKind::CachedAttention`] still declares
/// two key/value ranges (its only shape today, per this module's own
/// `no new BoundOpKind` constraint): the single merged range is placed in
/// the "new" slot, which already carries the causal band restricting it to
/// non-future positions, and the "cached" slot is declared with
/// `cached_key_rows: 0` rather than duplicating the merged range into it —
/// an empty range, not a live range neutered by an unreachable band. Both
/// [`crate::physical::stream_cached_attention_split_gqa`] and the Metal
/// kernel ([`crate::msl`]'s cached-attention render) treat a zero-length
/// cached range as a first-class case: nothing iterates it, rather than
/// iterating it and skipping every row via a dead-band `continue`.
#[cfg(feature = "cached-attention-streaming")]
fn cached_attention_single_range_candidates(
    program: &[Op],
    shapes: &Shapes,
    resolved: &[BoundOp],
    effective_outputs: &[NodeId],
) -> Vec<(BoundOp, BTreeSet<NodeId>)> {
    let mut candidates = Vec::new();
    for output_position in (0..program.len()).rev() {
        let output = NodeId(output_position as u32);
        let Some(attended_product) =
            reduced_source(program, output, ScalarOp::Add, ReduceInit::Zero)
        else {
            continue;
        };
        let Some(attended_parts) =
            binary_elementwise(program, attended_product, ScalarOp::Multiply)
        else {
            continue;
        };
        let Some(probabilities_parts) =
            binary_elementwise(program, attended_parts[0], ScalarOp::Multiply)
        else {
            continue;
        };
        let Some(weight_sum) =
            unary_elementwise(program, probabilities_parts[1], ScalarOp::Reciprocal)
        else {
            continue;
        };
        let Some(weights) = reduced_source(program, weight_sum, ScalarOp::Add, ReduceInit::Zero)
        else {
            continue;
        };
        if weights != probabilities_parts[0] {
            continue;
        }
        let Some(shifted) = unary_elementwise(program, weights, ScalarOp::Exponential) else {
            continue;
        };
        let Some(shifted_parts) = binary_elementwise(program, shifted, ScalarOp::Subtract) else {
            continue;
        };
        let Some(scores_masked_from_max) = reduced_source(
            program,
            shifted_parts[1],
            ScalarOp::Maximum,
            ReduceInit::NegativeInfinity,
        ) else {
            continue;
        };
        if scores_masked_from_max != shifted_parts[0] {
            continue;
        }
        let scores_masked = shifted_parts[0];
        let Some(mask_parts) = elementwise_operands(program, scores_masked, ScalarOp::Select)
        else {
            continue;
        };
        let [(mask, _), (negative_infinity, _), (scores_scaled, _)] = mask_parts else {
            continue;
        };
        let Some(cached_len_node) = exact_merged_causal_mask_cached_len(program, *mask) else {
            continue;
        };
        if constant_value(program, *negative_infinity) != Some(f32::NEG_INFINITY) {
            continue;
        }
        let Some(scaled_operands) = binary_elementwise(program, *scores_scaled, ScalarOp::Multiply)
        else {
            continue;
        };
        let scale = scaled_operands[1];
        // mistral's single-range chain never carries a pass plane -- this
        // matcher's own shape is full-rotary only, per its module doc.
        let Some((query_even_grouped, query_odd_grouped, key_even, key_odd, None)) =
            attention_score_sources(program, *scores_scaled, scale)
        else {
            continue;
        };
        let Some(query_even_parts) =
            binary_elementwise(program, query_even_grouped, ScalarOp::Multiply)
        else {
            continue;
        };
        let Some(query_odd_parts) =
            binary_elementwise(program, query_odd_grouped, ScalarOp::Multiply)
        else {
            continue;
        };
        if query_even_parts[1] != query_odd_parts[1] {
            continue;
        }
        let query_even = query_even_parts[0];
        let query_odd = query_odd_parts[0];
        let value = attended_parts[1];
        let source_nodes = [
            query_even, query_odd, key_even, key_odd, key_even, key_odd, value, value,
        ];
        let mut operands = Vec::with_capacity(source_nodes.len());
        for source in source_nodes {
            let Some((_, layout, lookup)) = resolved
                .iter()
                .flat_map(|bound| bound.operands().iter())
                .find(|(node, _, _)| *node == source)
            else {
                operands.clear();
                break;
            };
            if lookup.is_some() || layout.strides.iter().any(|stride| *stride < 0) {
                operands.clear();
                break;
            }
            operands.push((source, layout.clone(), None));
        }
        if operands.len() != source_nodes.len() {
            continue;
        }
        let Some(scale_value) = constant_value(program, scale) else {
            continue;
        };
        let query_shape = shapes.of(query_even_grouped);
        let key_shape = shapes.of(key_even);
        let value_shape = shapes.of(value);
        let Some(head_dim) = query_shape[3].checked_mul(2) else {
            continue;
        };
        if query_shape.len() != 4
            || key_shape.len() != 3
            || value_shape.len() != 3
            || query_shape[1] != key_shape[1]
            || key_shape[1] != value_shape[1]
            || key_shape[0] != value_shape[0]
            || value_shape[2] != head_dim
            || shapes.of(output) != [query_shape[0], query_shape[1], query_shape[2], head_dim]
            || key_shape[0] < query_shape[0]
        {
            continue;
        }
        // `key_shape[0]` (`t`, the whole merged range) minus `query_shape[0]`
        // (`s`, this call's own new positions) equals `cached_len` only when
        // `t` is exactly the merged length -- true for a plain evaluate, but
        // `kv-capacity-bucket` widens `t` to `ceil(merged_len /
        // bucket_tokens) * bucket_tokens`, so this difference silently
        // becomes `bucket - new_count`, larger than the real `cached_len` by
        // the padding. The band this feeds must therefore come from the
        // `cached_len` VALUE itself -- the same per-call `Op::Input`
        // `causal_mask_merged`'s query side already adds
        // (`exact_merged_causal_mask_cached_len` captured it above as
        // `cached_len_node`) -- carried through as this op's ninth operand
        // and read at execution time, never baked from a shape difference.
        if !shapes.of(cached_len_node).is_empty() {
            continue;
        }
        let cached_len_operand = (
            cached_len_node,
            Layout {
                base: 0,
                strides: SmallVec::new(),
            },
            None,
        );
        let pair_dim = query_shape[3];
        let query_strides = [
            (query_shape[1] * query_shape[2] * pair_dim) as i64,
            (query_shape[2] * pair_dim) as i64,
            pair_dim as i64,
            1i64,
        ];
        let key_strides = [
            0i64,
            (query_shape[1] * pair_dim) as i64,
            pair_dim as i64,
            0,
            1,
        ];
        let value_strides = [
            0i64,
            (query_shape[1] * query_shape[3] * 2) as i64,
            (query_shape[3] * 2) as i64,
            0,
            1,
        ];
        if operands[0].1.strides.as_slice() != query_strides
            || operands[1].1.strides.as_slice() != query_strides
            || operands[2].1.strides.as_slice() != key_strides
            || operands[3].1.strides.as_slice() != key_strides
            || operands[4].1.strides.as_slice() != key_strides
            || operands[5].1.strides.as_slice() != key_strides
            || operands[6].1.strides.as_slice() != value_strides
            || operands[7].1.strides.as_slice() != value_strides
        {
            continue;
        }
        let dependencies = attention_dependencies(program, output, &source_nodes);
        // `cached_len_node` sits on the same mask-chain path `source_nodes`
        // already gets excluded from -- it is about to become this op's own
        // ninth operand, so it must never be classified as absorbed
        // (removed) the way the rest of the mask arithmetic is.
        let dependencies = dependencies
            .difference(&source_nodes.into_iter().collect())
            .copied()
            .filter(|node| *node != cached_len_node)
            .collect::<BTreeSet<_>>();
        if dependencies
            .iter()
            .any(|node| effective_outputs.contains(node))
        {
            #[cfg(feature = "instrument")]
            for node in &dependencies {
                if effective_outputs.contains(node) {
                    debug!(
                        node = node.0,
                        kind = "cached_attention_absorption",
                        decision = "rejected_requested_output",
                        into = output.0,
                        "attention mask dependency not absorbed -- it is a requested output"
                    );
                }
            }
            continue;
        }
        let absorbed = removable_attention_dependencies(program, &dependencies, output);
        if absorbed.is_empty() {
            continue;
        }
        if !resolved.iter().any(|bound| bound.node == output) {
            continue;
        }
        operands.push(cached_len_operand);
        let fused = BoundOp {
            node: output,
            dtype: DType::Float32,
            extents: shapes.of(output).to_vec(),
            kind: BoundOpKind::CachedAttention {
                operands,
                query_rows: query_shape[0],
                // no separate cached range exists for a merged buffer -- see
                // this function's own doc; `cached_key_rows: 0` makes the
                // kernel's cached half a first-class empty range instead of
                // a live range neutered by an unreachable band sentinel.
                cached_key_rows: 0,
                new_key_rows: key_shape[0],
                kv_heads: query_shape[1],
                query_groups: query_shape[2],
                head_dim,
                // this matcher recognizes only the flat two-term score
                // (`attention_score_sources`'s own doc) -- full rotary,
                // never a pass plane.
                rotary_dim: head_dim,
                scale: scale_value,
                cached_lower_inclusive: i64::MIN,
                new_upper_inclusive: 0,
            },
        };
        candidates.push((fused, absorbed));
    }
    candidates
}

#[cfg(feature = "cached-attention-streaming")]
fn attention_dependencies(program: &[Op], output: NodeId, sources: &[NodeId]) -> BTreeSet<NodeId> {
    let source_set: BTreeSet<NodeId> = sources.iter().copied().collect();
    let mut visited = BTreeSet::new();
    let mut pending = vec![output];
    while let Some(node) = pending.pop() {
        if !visited.insert(node) || source_set.contains(&node) {
            continue;
        }
        match program.get(node.0 as usize) {
            Some(Op::Elementwise { operands, .. }) => {
                pending.extend(operands.iter().map(|(source, _)| *source));
            }
            Some(Op::Reduce(reduce)) => pending.push(reduce.operand),
            Some(Op::Input { .. }) | Some(Op::Iota { .. }) | Some(Op::Constant { .. }) | None => {}
        }
    }
    visited.remove(&output);
    visited
}

#[cfg(feature = "cached-attention-streaming")]
fn has_external_attention_consumer(
    consumers: &BTreeMap<NodeId, BTreeSet<NodeId>>,
    dependencies: &BTreeSet<NodeId>,
    dependency: NodeId,
    output: NodeId,
) -> bool {
    consumers
        .get(&dependency)
        .into_iter()
        .flatten()
        .any(|consumer| !dependencies.contains(consumer) && *consumer != output)
}

#[cfg(feature = "cached-attention-streaming")]
fn attention_consumers(
    program: &[Op],
    dependencies: &BTreeSet<NodeId>,
) -> BTreeMap<NodeId, BTreeSet<NodeId>> {
    let mut consumers = BTreeMap::new();
    for (position, operation) in program.iter().enumerate() {
        let consumer = NodeId(position as u32);
        let mut references = Vec::new();
        match operation {
            Op::Elementwise { operands, .. } => {
                references.extend(operands.iter().map(|(node, _)| *node));
            }
            Op::Reduce(reduce) => references.push(reduce.operand),
            Op::Input { .. } | Op::Iota { .. } | Op::Constant { .. } => {}
        }
        for dependency in references
            .into_iter()
            .filter(|node| dependencies.contains(node))
        {
            consumers
                .entry(dependency)
                .or_insert_with(BTreeSet::new)
                .insert(consumer);
        }
    }
    consumers
}

#[cfg(feature = "cached-attention-streaming")]
fn removable_attention_dependencies(
    program: &[Op],
    dependencies: &BTreeSet<NodeId>,
    output: NodeId,
) -> BTreeSet<NodeId> {
    let consumers = attention_consumers(program, dependencies);
    let mut retained = dependencies
        .iter()
        .copied()
        .filter(|node| has_external_attention_consumer(&consumers, dependencies, *node, output))
        .collect::<BTreeSet<_>>();
    let mut changed = true;
    while changed {
        changed = false;
        for node in retained.clone() {
            let ancestors = match program.get(node.0 as usize) {
                Some(Op::Elementwise { operands, .. }) => {
                    operands.iter().map(|(source, _)| *source).collect()
                }
                Some(Op::Reduce(reduce)) => alloc::vec![reduce.operand],
                Some(Op::Input { .. })
                | Some(Op::Iota { .. })
                | Some(Op::Constant { .. })
                | None => Vec::new(),
            };
            for ancestor in ancestors {
                if dependencies.contains(&ancestor) && retained.insert(ancestor) {
                    changed = true;
                }
            }
        }
    }
    dependencies.difference(&retained).copied().collect()
}

pub fn bind(
    program: &[Op],
    shapes: &Shapes,
    outputs: &[NodeId],
    numeric_policy: NumericPolicy,
) -> Result<Vec<BoundOp>, TensorError> {
    bind_with_fusion(program, shapes, outputs, true, numeric_policy)
}

/// Binds the graph with cached-attention/chain fusion but leaves reduction
/// epilogues as separate operations for backends whose lowering does not yet
/// support broadcast epilogue operands. This is a capability boundary, not a
/// numerical relaxation: the returned graph is the unfused correct form.
pub fn bind_without_reduce_epilogue_fusion(
    program: &[Op],
    shapes: &Shapes,
    outputs: &[NodeId],
    fuse_cached_attention: bool,
    numeric_policy: NumericPolicy,
) -> Result<Vec<BoundOp>, TensorError> {
    admit(numeric_policy, NumericRewrite::IdentityElimination)?;
    admit(numeric_policy, NumericRewrite::ChainFusion)?;
    let built = bind_cached_attention_fusion(
        program,
        shapes,
        outputs,
        fuse_cached_attention,
        numeric_policy,
    )?;
    Ok(built)
}

/// Same as [`bind`], but `fuse_cached_attention` states whether the caller's
/// backend can render [`BoundOpKind::CachedAttention`] at all. `cpu.rs` and
/// `omega/src/metal.rs` render the fused kind, so they (via [`bind`]) pass
/// `true`; `omega`'s wgpu and cuda drivers have no renderer for it yet, so
/// they call this directly with `false` — the fused rewrite never fires for
/// them, and the plain elementwise/reduce chain `bind_plain` already
/// produces is what they emit.
///
/// `reduce-epilogue-fusion` (the `BoundOpKind::Reduce::epilogue_body`/
/// `epilogue_operands` rewrite) runs unconditionally after this, gated only
/// by the crate feature — it has no per-call capability bool of its own
/// because, unlike cached-attention, every renderer this crate ships either
/// renders the epilogue or rejects it at bind time (see this module's own
/// `reduce_epilogue_fusion`, private and feature-gated); there is no third
/// "silently ignore it" caller to protect the way `fuse_cached_attention: false`
/// protects wgpu/cuda from a fused kind they cannot render at all.
///
/// `numeric_policy` is the [`NumericPolicy`] every bit-changing rewrite this
/// function fires must clear via [`admit`] before it runs. The three
/// rewrites shipped today (identity elimination, chain fusion,
/// reduce-epilogue fusion) are classified [`NumericRewrite`]s whose
/// [`NumericRewrite::required_permissions`] is [`NumericPolicy::bit_exact()`],
/// so a call under the default policy always clears — nothing regresses. A
/// future reassociating rewrite in this crate declares its own
/// [`NumericRewrite`] variant and is admitted the same way.
pub fn bind_with_fusion(
    program: &[Op],
    shapes: &Shapes,
    outputs: &[NodeId],
    fuse_cached_attention: bool,
    numeric_policy: NumericPolicy,
) -> Result<Vec<BoundOp>, TensorError> {
    // The three rewrites this crate ships unconditionally today are
    // bit-exact by construction (identity elimination, chain fusion --
    // both inside `bind_cached_attention_fusion`'s own `bind_plain` --
    // and reduce-epilogue fusion below), so `admit` always clears at
    // `NumericPolicy::default()`; the call is the explicit, testable
    // declaration of that fact, not a behavior change (`op.rs:107`'s
    // `is_associative` has no such caller today).
    admit(numeric_policy, NumericRewrite::IdentityElimination)?;
    admit(numeric_policy, NumericRewrite::ChainFusion)?;
    #[cfg(feature = "std")]
    let fuse_cached_attention = fuse_cached_attention
        && std::env::var_os("PROXIMA_DISABLE_CACHED_ATTENTION_FUSION").is_none();
    let built = bind_cached_attention_fusion(
        program,
        shapes,
        outputs,
        fuse_cached_attention,
        numeric_policy,
    )?;
    #[cfg(feature = "instrument")]
    debug!(
        stage = "after_cached_attention_fusion",
        cached_attention_count = built
            .iter()
            .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
            .count() as u64,
        "bind_with_fusion: fused-op-kind count per stage, catches a later stage silently discarding an earlier fusion"
    );
    #[cfg(feature = "gated-delta-net-fusion")]
    let built = apply_gated_delta_net_fusion(
        built,
        program,
        shapes,
        outputs,
        fuse_cached_attention,
        numeric_policy,
    )?;
    #[cfg(feature = "reduce-epilogue-fusion")]
    {
        admit(numeric_policy, NumericRewrite::ReduceEpilogueFusion)?;
        #[cfg(feature = "std")]
        if std::env::var_os("PROXIMA_DISABLE_REDUCE_EPILOGUE_FUSION").is_some() {
            return Ok(built);
        }
        let epilogued = reduce_epilogue_fusion(built, outputs, numeric_policy)?;
        #[cfg(feature = "instrument")]
        debug!(
            stage = "after_reduce_epilogue_fusion",
            cached_attention_count = epilogued
                .iter()
                .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
                .count() as u64,
            "bind_with_fusion: fused-op-kind count per stage, catches a later stage silently discarding an earlier fusion"
        );
        Ok(epilogued)
    }
    #[cfg(not(feature = "reduce-epilogue-fusion"))]
    Ok(built)
}

/// Binary [`Op::Elementwise`] lookup, gated on this crate's own
/// `gated-delta-net-fusion` feature — a small duplicate of
/// [`binary_elementwise`] (that helper lives behind
/// `cached-attention-streaming` instead) rather than a shared function two
/// independent feature gates would both have to enable to compile.
/// A zero-based, row-major (last axis fastest) [`Layout`] over `extents` --
/// what [`Op::Input`]'s own storage always is, and what
/// [`gated_delta_net_candidates`] binds `query`/`key` to directly instead of
/// borrowing a `repeat_kv_heads` broadcast consumer's own stride-0 read (this
/// module's own doc on [`GatedDeltaNetMatch::query_was_repeated`]).
#[cfg(feature = "gated-delta-net-fusion")]
fn natural_layout(extents: &[u64]) -> Layout {
    let mut strides = SmallVec::<[i64; MAX_INLINE_RANK]>::new();
    strides.resize(extents.len(), 0);
    let mut running = 1_i64;
    for (axis, extent) in extents.iter().enumerate().rev() {
        strides[axis] = running;
        running *= *extent as i64;
    }
    Layout { base: 0, strides }
}

/// Drops a genuine leading extent-1 axis, if `shape` has one and still has a
/// non-empty tail -- [`gated_delta_net_candidates`]'s own doc on why
/// `query`/`key` still carry the decode step's own size-1 token axis after
/// `gdn_unwrap_decode_squeeze` walks past the `repeat_kv_heads` squeeze.
#[cfg(feature = "gated-delta-net-fusion")]
fn strip_leading_unit_axis(shape: &[u64]) -> &[u64] {
    match shape {
        [1, rest @ ..] if !rest.is_empty() => rest,
        _ => shape,
    }
}

#[cfg(feature = "gated-delta-net-fusion")]
fn gdn_binary_elementwise(program: &[Op], node: NodeId, body: ScalarOp) -> Option<[NodeId; 2]> {
    match program.get(node.0 as usize)? {
        Op::Elementwise {
            body: actual_body,
            operands,
            ..
        } if *actual_body == body => match operands.as_slice() {
            [(left, _), (right, _)] => Some([*left, *right]),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(feature = "gated-delta-net-fusion")]
fn gdn_unary_elementwise(program: &[Op], node: NodeId, body: ScalarOp) -> Option<NodeId> {
    match program.get(node.0 as usize)? {
        Op::Elementwise {
            body: actual_body,
            operands,
            ..
        } if *actual_body == body => match operands.as_slice() {
            [(source, _)] => Some(*source),
            _ => None,
        },
        _ => None,
    }
}

/// The same [`Op::Constant`] read [`cached_attention_candidates`]'s own
/// `scale: f32` field relies on -- `inv_sqrt_key_dim` is baked at graph-build
/// time (`1/sqrt(head_k_dim)`, a compile-time constant of the model's own
/// architecture), so [`BoundOpKind::GatedDeltaNet::inv_sqrt_key_dim`] is a
/// plain `f32`, not a bound operand.
#[cfg(feature = "gated-delta-net-fusion")]
fn gdn_constant_value(program: &[Op], node: NodeId) -> Option<f32> {
    match program.get(node.0 as usize)? {
        Op::Constant { value, .. } => Some(*value),
        _ => None,
    }
}

#[cfg(feature = "gated-delta-net-fusion")]
fn gdn_reduced_source(program: &[Op], node: NodeId, body: ScalarOp) -> Option<NodeId> {
    match program.get(node.0 as usize)? {
        Op::Reduce(reduce)
            if reduce.body == body
                && reduce.init == ReduceInit::Zero
                && reduce.keep == Keep::Reduce =>
        {
            Some(reduce.operand)
        }
        _ => None,
    }
}

/// `true` when `node` is [`crate::spec::repeat_kv_heads`]'s own output shape:
/// an elementwise `Multiply` against a rank-`>=1` all-ones [`Op::Constant`].
/// Reused, not restated, from that function's own doc: the donor is what
/// makes `shape::unify_iteration_space` resolve the broadcast group axis at
/// all, so its value is always exactly `1.0`.
#[cfg(feature = "gated-delta-net-fusion")]
fn gdn_is_repeat_kv_heads_donor(program: &[Op], node: NodeId) -> bool {
    matches!(
        program.get(node.0 as usize),
        Some(Op::Constant { value, .. }) if *value == 1.0
    )
}

/// Walks past a [`crate::spec::repeat_kv_heads`] broadcast if `node` is one,
/// returning the pre-repeat source otherwise unchanged — the one place this
/// matcher intentionally disagrees with [`crate::spec::append_qwen35_delta_net_step`]'s
/// own physical operand and instead binds what llama.cpp's fused Metal kernel
/// reads directly (`gated_delta_net.metal:33-34`'s `i01 = i21 % ne01`
/// mod-broadcast), dropping the eager repeat from the hot path entirely (this
/// module's own doc on [`BoundOpKind::GatedDeltaNet`]).
#[cfg(feature = "gated-delta-net-fusion")]
fn gdn_unwrap_repeat_kv_heads(program: &[Op], node: NodeId) -> NodeId {
    match gdn_binary_elementwise(program, node, ScalarOp::Multiply) {
        Some([left, right]) if gdn_is_repeat_kv_heads_donor(program, right) => left,
        Some([left, right]) if gdn_is_repeat_kv_heads_donor(program, left) => right,
        _ => node,
    }
}

/// The all-ones donor [`gdn_unwrap_repeat_kv_heads`] walked past, if `node`
/// was a repeat -- `repeat_kv_heads` mints a fresh `Op::Constant` per call
/// (never shared across the query/key repeat sites), so once the multiply
/// that reads it is absorbed this donor has no other consumer and would
/// otherwise linger as a dead leaf in the rewritten program.
#[cfg(feature = "gated-delta-net-fusion")]
fn gdn_repeat_kv_heads_donor(program: &[Op], node: NodeId) -> Option<NodeId> {
    match gdn_binary_elementwise(program, node, ScalarOp::Multiply) {
        Some([_, right]) if gdn_is_repeat_kv_heads_donor(program, right) => Some(right),
        Some([left, _]) if gdn_is_repeat_kv_heads_donor(program, left) => Some(left),
        _ => None,
    }
}

/// Walks past `append_qwen35_ssm_mixer_with_taps_and_layout`'s own
/// "squeeze the size-1 decode-step `s` axis away" reduce
/// (`spec.rs:8737-8786`) if `node` is one, returning `node`'s own pre-squeeze
/// operand instead -- a plain [`ScalarOp::Add`]/[`ReduceInit::Zero`]/
/// [`Keep::Reduce`] fold whose `in_map` reads its operand through a genuine
/// identity (no permutation, no broadcast: operand axis `p` addresses
/// iteration axis `p`) and whose `out_map` is a pure projection (every axis a
/// single coeff-1 term, no offset) that keeps every iteration axis except
/// exactly one, and that one axis's own extent (read off `shapes`) is `1`.
/// Every other shape returns `node` unchanged rather than guess. This is the
/// gap [`gated_delta_net_candidates`]'s own doc names: the real program
/// threads `query`/`key`/`value`/`gate`/`beta` through this exact squeeze
/// between the algebra's own `u,g`-split construction and
/// [`append_qwen35_delta_net_step`], and unwrapping it is what lets this
/// matcher bind the program's own natural, pre-squeeze storage order instead
/// of the squeeze's own re-lettered output.
#[cfg(feature = "gated-delta-net-fusion")]
fn gdn_unwrap_decode_squeeze(program: &[Op], shapes: &Shapes, node: NodeId) -> NodeId {
    let Some(Op::Reduce(reduce)) = program.get(node.0 as usize) else {
        return node;
    };
    if reduce.body != ScalarOp::Add
        || reduce.init != ReduceInit::Zero
        || reduce.keep != Keep::Reduce
        || reduce.in_map.is_data_dependent()
        || reduce.out_map.is_data_dependent()
    {
        return node;
    }
    let operand_extents = shapes.of(reduce.operand);
    let in_pattern = reduce.in_map.affine();
    let is_identity = in_pattern.axes.len() == operand_extents.len()
        && in_pattern.axes.iter().enumerate().all(|(axis, index)| {
            index.offset == 0
                && matches!(index.terms.as_slice(), [term] if term.coeff == 1 && term.axis as usize == axis)
        });
    if !is_identity {
        return node;
    }
    let out_pattern = reduce.out_map.affine();
    let mut kept_axes = SmallVec::<[u16; MAX_INLINE_RANK]>::new();
    for index in &out_pattern.axes {
        match index.terms.as_slice() {
            [term] if term.coeff == 1 && index.offset == 0 => kept_axes.push(term.axis),
            _ => return node,
        }
    }
    let dropped: SmallVec<[u16; MAX_INLINE_RANK]> = (0..operand_extents.len() as u16)
        .filter(|axis| !kept_axes.contains(axis))
        .collect();
    match dropped.as_slice() {
        [only] if operand_extents.get(*only as usize) == Some(&1) => reduce.operand,
        _ => node,
    }
}

/// One matched [`append_qwen35_delta_net_step`](crate::spec::append_qwen35_delta_net_step)
/// recurrence, structurally recognized by walking backward from its `out`
/// node through the exact `ScalarOp` sequence that function emits — anchored
/// on op shape, never on node names (this module's own convention;
/// [`cached_attention_candidates`] is the standing precedent). Declines
/// (returns nothing for this `output`) rather than guesses on any mismatch,
/// including a perturbed single op in the chain.
#[cfg(feature = "gated-delta-net-fusion")]
struct GatedDeltaNetMatch {
    query: NodeId,
    key: NodeId,
    value: NodeId,
    gate: NodeId,
    beta: NodeId,
    state_in: NodeId,
    inv_sqrt_key_dim: f32,
    state_out: NodeId,
    /// `true` when [`gdn_unwrap_repeat_kv_heads`] actually walked past a
    /// broadcast for `query`/`key` (whether or not a decode-squeeze sat
    /// above it) — every REMAINING consumer of that pre-repeat source reads
    /// it through the broadcast's own stride-0 trailing axis, so
    /// [`gated_delta_net_candidates`] must derive a NATURAL layout from
    /// `query`/`key`'s own extents (dim fastest, this slice's pre-repeat
    /// storage order) instead of borrowing a consumer's `i{head}`-convention
    /// read (dim slowest) the way the other four operands safely do, and the
    /// executor must be told which convention it got
    /// ([`BoundOpKind::GatedDeltaNet`]'s own `query_key_head_stride`/
    /// `query_key_dim_stride` fields).
    query_was_repeated: bool,
    key_was_repeated: bool,
    /// Every node absorbed into the fused op, `out` and the six leaf sources
    /// excluded — the set [`apply_gated_delta_net_fusion`] drops from the
    /// rewritten program once the fusion actually fires.
    absorbed: BTreeSet<NodeId>,
}

#[cfg(feature = "gated-delta-net-fusion")]
fn match_gated_delta_net_step(
    program: &[Op],
    shapes: &Shapes,
    out: NodeId,
) -> Option<GatedDeltaNetMatch> {
    let mut absorbed = BTreeSet::new();
    let absorb = |node: NodeId, set: &mut BTreeSet<NodeId>| {
        set.insert(node);
    };

    let out_product = gdn_reduced_source(program, out, ScalarOp::Add)?;
    absorb(out_product, &mut absorbed);
    let [state_out, query_scaled] = gdn_binary_elementwise(program, out_product, ScalarOp::Multiply)?;
    absorb(query_scaled, &mut absorbed);

    let [state_decayed_for_update, update] =
        gdn_binary_elementwise(program, state_out, ScalarOp::Add)?;
    absorb(state_out, &mut absorbed);
    absorb(update, &mut absorbed);
    absorb(state_decayed_for_update, &mut absorbed);

    let [key_a, delta] = gdn_binary_elementwise(program, update, ScalarOp::Multiply)?;
    absorb(delta, &mut absorbed);
    let [state_in_a, decay_a] =
        gdn_binary_elementwise(program, state_decayed_for_update, ScalarOp::Multiply)?;

    let [residual, beta_bcast] = gdn_binary_elementwise(program, delta, ScalarOp::Multiply)?;
    absorb(residual, &mut absorbed);
    let [value, value_pred] = gdn_binary_elementwise(program, residual, ScalarOp::Subtract)?;
    absorb(value_pred, &mut absorbed);

    let value_pred_product = gdn_reduced_source(program, value_pred, ScalarOp::Add)?;
    absorb(value_pred_product, &mut absorbed);
    let [state_decayed, key_b] =
        gdn_binary_elementwise(program, value_pred_product, ScalarOp::Multiply)?;
    absorb(state_decayed, &mut absorbed);
    let [state_in_b, decay_b] = gdn_binary_elementwise(program, state_decayed, ScalarOp::Multiply)?;

    if state_in_a != state_in_b || decay_a != decay_b || key_a != key_b {
        return None;
    }
    let gate = gdn_unary_elementwise(program, decay_a, ScalarOp::Exponential)?;
    absorb(decay_a, &mut absorbed);

    let [query, inv_sqrt_key_dim_node] =
        gdn_binary_elementwise(program, query_scaled, ScalarOp::Multiply)?;
    let inv_sqrt_key_dim = gdn_constant_value(program, inv_sqrt_key_dim_node)?;

    // The real program threads every one of these five through its own
    // decode-squeeze reduce before `append_qwen35_delta_net_step` ever sees
    // them (`gdn_unwrap_decode_squeeze`'s own doc); query/key additionally
    // sit behind a `repeat_kv_heads` broadcast UNDER that squeeze, so the
    // squeeze must unwrap first or `gdn_unwrap_repeat_kv_heads` never finds
    // the donor multiply it looks for.
    let key_squeezed = gdn_unwrap_decode_squeeze(program, shapes, key_a);
    if key_squeezed != key_a {
        absorbed.insert(key_a);
    }
    let key = gdn_unwrap_repeat_kv_heads(program, key_squeezed);
    let key_was_repeated = key != key_squeezed;
    if key_was_repeated {
        absorbed.insert(key_squeezed);
        if let Some(donor) = gdn_repeat_kv_heads_donor(program, key_squeezed) {
            absorbed.insert(donor);
        }
    }

    let query_squeezed = gdn_unwrap_decode_squeeze(program, shapes, query);
    if query_squeezed != query {
        absorbed.insert(query);
    }
    let query = gdn_unwrap_repeat_kv_heads(program, query_squeezed);
    let query_was_repeated = query != query_squeezed;
    if query_was_repeated {
        absorbed.insert(query_squeezed);
        if let Some(donor) = gdn_repeat_kv_heads_donor(program, query_squeezed) {
            absorbed.insert(donor);
        }
    }

    Some(GatedDeltaNetMatch {
        query,
        key,
        value,
        gate,
        beta: beta_bcast,
        state_in: state_in_a,
        inv_sqrt_key_dim,
        query_was_repeated,
        key_was_repeated,
        state_out,
        absorbed,
    })
}

/// Scans `program` for [`append_qwen35_delta_net_step`](crate::spec::append_qwen35_delta_net_step)
/// candidates and, for each, resolves its six operand sources' [`Layout`]s
/// against `resolved` — the same technique [`cached_attention_candidates`]
/// uses, and for the same reason: chain fusion may already have inlined an
/// intermediate elementwise op into a reduce's own `element_body`, so the
/// true physical read is whatever `resolved`'s own `operands()` names, not
/// necessarily a node this function's own backward walk stopped at.
///
/// `state_out` is this op's own second output (ROW 547,
/// `docs/discipline.md`) -- requesting it alongside `out` no longer declines
/// the match; the fused kind supplies it directly. Still declines when any
/// OTHER absorbed node (a decode-squeeze reduce, a `repeat_kv_heads` donor,
/// ...) is itself a requested/effective output, since those genuinely
/// disappear from `resolved` once fusion fires.
#[cfg(feature = "gated-delta-net-fusion")]
fn gated_delta_net_candidates(
    program: &[Op],
    shapes: &Shapes,
    resolved: &[BoundOp],
    effective_outputs: &[NodeId],
) -> Vec<(BoundOp, BTreeSet<NodeId>)> {
    let mut candidates = Vec::new();
    for output_position in (0..program.len()).rev() {
        let output = NodeId(output_position as u32);
        let Some(found) = match_gated_delta_net_step(program, shapes, output) else {
            continue;
        };
        if found
            .absorbed
            .iter()
            .any(|node| *node != found.state_out && effective_outputs.contains(node))
        {
            // `found.absorbed` now includes the decode-squeeze reduces
            // themselves (`gdn_unwrap_decode_squeeze`'s own doc), which a
            // caller may legitimately request as a standalone diagnostic tap
            // (`SsmMixerTaps::query`/`key`/`value`/`gate`/`beta`) independent
            // of this fusion -- absorbing one out from under such a request
            // would silently delete a node the caller is about to read, the
            // same requested-output guard [`cached_attention_candidates`]'s
            // own `dependencies` check already makes. `state_out` itself is
            // exempt (this function's own doc) -- it is now a genuine second
            // output of the fused op, never dropped.
            #[cfg(feature = "instrument")]
            debug!(
                out = output.0,
                state_out = found.state_out.0,
                "gdn candidate declined: an absorbed node other than state_out is a requested \
                 output"
            );
            continue;
        }
        let source_nodes = [
            (found.query, found.query_was_repeated),
            (found.key, found.key_was_repeated),
            (found.value, false),
            (found.gate, false),
            (found.beta, false),
            (found.state_in, false),
        ];
        let mut operands = Vec::with_capacity(source_nodes.len());
        for (source, bind_natural) in source_nodes {
            // `query`/`key` may sit behind an unwrapped `repeat_kv_heads`
            // broadcast: every remaining consumer of `source` would then read
            // it through that broadcast's own stride-0 trailing axis, not
            // `source`'s own storage -- borrowing a consumer's read here
            // would silently bind a layout the executor's own contiguity
            // check (`cpu.rs`'s `run_gated_delta_net`) then rejects at run
            // time. `source` itself is always a plain natural-order node
            // ([`Op::Input`] or an equally natural `bind_plain` output, and
            // [`apply_gated_delta_net_fusion`] forces it into the planning
            // outputs so it is actually materialized that way), so its own
            // extents fully determine a natural layout whether or not a
            // repeat was actually present.
            if bind_natural {
                operands.push((source, natural_layout(shapes.of(source)), None));
                continue;
            }
            let Some((_, layout, lookup)) = resolved
                .iter()
                .flat_map(|bound| bound.operands().iter())
                .find(|(node, _, _)| *node == source)
            else {
                operands.clear();
                break;
            };
            if lookup.is_some() || layout.strides.iter().any(|stride| *stride < 0) {
                operands.clear();
                break;
            }
            operands.push((source, layout.clone(), None));
        }
        if operands.len() != source_nodes.len() {
            continue;
        }
        // `query`/`key` bind at two DIFFERENT physical conventions depending
        // on whether a `repeat_kv_heads` broadcast was actually unwrapped
        // (`found.query_was_repeated`/`key_was_repeated`, this match's own
        // doc): the program's own pre-repeat storage, `[kv_heads, key_dim]`
        // dim FASTEST, when it was; the `i{head}` consumer convention's own
        // read, `[key_dim, heads]` dim SLOWEST, when it was not (this
        // slice's non-GQA, non-repeated shape only). Both decline this
        // candidate rather than guess if they disagree, and both normalize
        // to `[heads_axis, key_dim]` below so every match arm reads one
        // consistent order regardless of which convention actually bound.
        if found.query_was_repeated != found.key_was_repeated {
            continue;
        }
        let normalize_query_or_key = |shape: &[u64], was_repeated: bool| -> Option<[u64; 2]> {
            match (shape, was_repeated) {
                (&[heads, key_dim], true) => Some([heads, key_dim]),
                (&[key_dim, heads], false) => Some([heads, key_dim]),
                _ => None,
            }
        };
        let Some(key_shape) = normalize_query_or_key(
            strip_leading_unit_axis(shapes.of(found.key)),
            found.key_was_repeated,
        ) else {
            continue;
        };
        let Some(query_shape) = normalize_query_or_key(
            strip_leading_unit_axis(shapes.of(found.query)),
            found.query_was_repeated,
        ) else {
            continue;
        };
        let key_shape = key_shape.as_slice();
        let query_shape = query_shape.as_slice();
        let value_shape = shapes.of(found.value);
        let gate_shape = shapes.of(found.gate);
        let state_shape = shapes.of(found.state_in);
        // This slice's supported shapes: `append_qwen35_delta_net_step`'s own
        // `head` split, either the single-letter axis (`[heads, dim]`, no GQA
        // broadcast) or the real qwen35moe two-letter `head = "ug"` split --
        // `u` = kv group (query/key's own trailing axis, PRE-`repeat_kv_heads`,
        // `gdn_unwrap_repeat_kv_heads`'s own doc), `g` = query heads per group
        // (value/gate/beta/state's own extra trailing axis, since
        // `repeat_kv_heads`'s own doc proves `u`/`g` never collapse into one
        // physical axis). `query`/`key`'s own natural storage (`gdn.rs`'s own
        // struct doc) is `[kv_heads, key_dim]`, dim FASTEST -- the program's
        // own pre-repeat operand order, distinct from `value`/`gate`/`beta`/
        // `state`, whose consumer reads them un-permuted off
        // `append_qwen35_delta_net_step`'s own `j{head}`/`{head}` maps, dim
        // (where present) SLOWEST: value is `[dim, kv_heads, group]`,
        // gate/beta `[kv_heads, group]`, state `[key_dim, value_dim,
        // kv_heads, group]` -- `num_v_heads = kv_heads * group` is exactly
        // the executor's own flat value-head extent
        // (`gdn::GdnPrefillShape::heads`), so this binds the SAME struct the
        // single-axis case already does, group folded in rather than a new
        // field.
        let (kv_heads, group) = match (key_shape, value_shape, gate_shape, state_shape) {
            ([kv_heads, key_dim], [value_dim, value_kv_heads, group], [gate_kv_heads, gate_group], [state_key_dim, state_value_dim, state_kv_heads, state_group])
                if query_shape == key_shape
                    && *kv_heads == *value_kv_heads
                    && *kv_heads == *gate_kv_heads
                    && *kv_heads == *state_kv_heads
                    && *group == *gate_group
                    && *group == *state_group
                    && *state_key_dim == *key_dim
                    && *state_value_dim == *value_dim
                    && shapes.of(output) == value_shape =>
            {
                (*kv_heads, *group)
            }
            ([heads, key_dim], [value_dim, value_heads], [gate_heads], [state_key_dim, state_value_dim, state_heads])
                if query_shape == key_shape
                    && *heads == *value_heads
                    && *heads == *gate_heads
                    && *heads == *state_heads
                    && *state_key_dim == *key_dim
                    && *state_value_dim == *value_dim
                    && shapes.of(output) == value_shape =>
            {
                (*heads, 1)
            }
            _ => continue,
        };
        let head_k_dim = key_shape[1];
        let head_v_dim = value_shape[0];
        let num_v_heads = kv_heads * group;
        // The executor reads `query`/`key` by explicit stride rather than by
        // assuming one fixed axis order (`gdn::GdnPrefillScan`'s own doc) --
        // `key_was_repeated` and `query_was_repeated` agree by construction
        // (declined above otherwise), so one pair of strides serves both.
        let (query_key_head_stride, query_key_dim_stride) = if found.key_was_repeated {
            (head_k_dim, 1)
        } else {
            (1, kv_heads)
        };
        let fused = BoundOp {
            node: output,
            dtype: DType::Float32,
            extents: shapes.of(output).to_vec(),
            kind: BoundOpKind::GatedDeltaNet {
                operands,
                n_tokens: 1,
                kv_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                query_key_head_stride,
                query_key_dim_stride,
                inv_sqrt_key_dim: found.inv_sqrt_key_dim,
                state_out: found.state_out,
            },
        };
        candidates.push((fused, found.absorbed));
    }
    candidates
}

/// Runs [`gated_delta_net_candidates`] and rewrites `built` with every
/// non-conflicting match — the [`BoundOpKind::GatedDeltaNet`] sibling of
/// [`bind_cached_attention_fusion`]'s own two-pass shape: an initial pass
/// finds candidates against `built`, widens the planning outputs to every
/// source [`match_gated_delta_net_step`] needs materialized, rebinds, then
/// matches again against the wider `resolved` set before rewriting.
#[cfg(feature = "gated-delta-net-fusion")]
fn apply_gated_delta_net_fusion(
    built: Vec<BoundOp>,
    program: &[Op],
    shapes: &Shapes,
    outputs: &[NodeId],
    fuse_cached_attention: bool,
    numeric_policy: NumericPolicy,
) -> Result<Vec<BoundOp>, TensorError> {
    let initial_candidates = gated_delta_net_candidates(program, shapes, &built, outputs);
    if initial_candidates.is_empty() {
        return Ok(built);
    }
    let mut planning_outputs = outputs.to_vec();
    if planning_outputs.is_empty() {
        let root = program
            .len()
            .checked_sub(1)
            .map(|position| NodeId(position as u32))
            .ok_or(TensorError::Empty)?;
        planning_outputs.push(root);
    }
    for (fused, _) in &initial_candidates {
        let BoundOpKind::GatedDeltaNet { operands, .. } = &fused.kind else {
            continue;
        };
        for (source, _, _) in operands {
            if !planning_outputs.contains(source) {
                planning_outputs.push(*source);
            }
        }
    }
    // `bind_plain` here used to drop every `BoundOpKind::CachedAttention`
    // `bind_cached_attention_fusion` above already spliced into `built` --
    // this rebind must carry that SAME fusion forward, or a hybrid
    // full-attention/gated-delta-net model (qwen35moe) loses all of its
    // cached-attention fusion the moment this feature is compiled in
    // (row 565: `built` measured 9-10 `CachedAttention` ops, `rebuilt` measured 0).
    let rebuilt = bind_cached_attention_fusion(
        program,
        shapes,
        &planning_outputs,
        fuse_cached_attention,
        numeric_policy,
    )?;
    let candidates = gated_delta_net_candidates(program, shapes, &rebuilt, outputs);
    if candidates.is_empty() {
        return Ok(built);
    }
    let fused_by_node = candidates
        .iter()
        .map(|(fused, _)| (fused.node, fused))
        .collect::<BTreeMap<_, _>>();
    let absorbed = candidates
        .iter()
        .flat_map(|(_, absorbed)| absorbed.iter().copied())
        .collect::<BTreeSet<_>>();
    let mut rewritten = Vec::with_capacity(rebuilt.len());
    for bound in rebuilt {
        if let Some(fused) = fused_by_node.get(&bound.node) {
            rewritten.push((*fused).clone());
        } else if !absorbed.contains(&bound.node) {
            rewritten.push(bound);
        }
    }
    Ok(rewritten)
}

fn bind_cached_attention_fusion(
    program: &[Op],
    shapes: &Shapes,
    outputs: &[NodeId],
    fuse_cached_attention: bool,
    numeric_policy: NumericPolicy,
) -> Result<Vec<BoundOp>, TensorError> {
    let built = bind_plain(program, shapes, outputs, numeric_policy)?;
    #[cfg(not(feature = "cached-attention-streaming"))]
    {
        let _ = fuse_cached_attention;
        Ok(built)
    }

    #[cfg(feature = "cached-attention-streaming")]
    {
        if !fuse_cached_attention {
            return Ok(built);
        }
        // `false`: this discovery pass finds anchors `bind_plain` has already
        // folded into their single consumer (qwen35's own `attended` tap,
        // `bind.rs:993`'s `elementwise_operand_fuse`) precisely so their node
        // id can be pinned into `planning_outputs` below and survive the
        // rebuild -- requiring a resolved binding here would make discovery
        // depend on the very materialization it exists to produce.
        let mut initial_candidates =
            cached_attention_candidates(program, shapes, &built, outputs, false);
        initial_candidates.extend(cached_attention_single_range_candidates(
            program, shapes, &built, outputs,
        ));
        if initial_candidates.is_empty() {
            return Ok(built);
        }
        let mut planning_outputs = outputs.to_vec();
        if planning_outputs.is_empty() {
            let root = program
                .len()
                .checked_sub(1)
                .map(|position| NodeId(position as u32))
                .ok_or(TensorError::Empty)?;
            planning_outputs.push(root);
        }
        for (fused, _) in &initial_candidates {
            let BoundOpKind::CachedAttention { operands, .. } = &fused.kind else {
                continue;
            };
            // the anchor itself (qwen35's own `attended` tap, single-consumer
            // into the per-head gate multiply) must be pinned alongside its
            // sources -- `bind_plain`'s single-consumer elementwise fusion
            // folds an unrequested single-consumer node into its consumer
            // before this function ever sees it (`bind.rs:993`'s own
            // `elementwise_operand_fuse`), so without this the anchor never
            // gets a standalone `BoundOp` for the second pass below to find
            // (`qwen35_partial_rotary_cached_attention_fuses_and_matches_the_unfused_layer`'s
            // own comment names the same requirement for its fixture's outputs).
            if !planning_outputs.contains(&fused.node) {
                planning_outputs.push(fused.node);
            }
            for (source, _, _) in operands {
                if !planning_outputs.contains(source) {
                    planning_outputs.push(*source);
                }
            }
        }
        let rebuilt = bind_plain(program, shapes, &planning_outputs, numeric_policy)?;
        let mut candidates = cached_attention_candidates(program, shapes, &rebuilt, outputs, true);
        candidates.extend(cached_attention_single_range_candidates(
            program, shapes, &rebuilt, outputs,
        ));
        if candidates.is_empty() {
            return Ok(built);
        }
        let fused_by_node = candidates
            .iter()
            .map(|(fused, _)| (fused.node, fused))
            .collect::<BTreeMap<_, _>>();
        let absorbed = candidates
            .iter()
            .flat_map(|(_, absorbed)| absorbed.iter().copied())
            .collect::<BTreeSet<_>>();
        let mut rewritten = Vec::with_capacity(rebuilt.len());
        for bound in rebuilt {
            if let Some(fused) = fused_by_node.get(&bound.node) {
                rewritten.push((*fused).clone());
            } else if !absorbed.contains(&bound.node) {
                rewritten.push(bound);
            }
        }
        Ok(rewritten)
    }
}

/// Is `bound` a still-un-scattered `Keep::Reduce` fold — the only
/// [`BoundOpKind`] a consumer's epilogue can ever absorb (`out_scatter`'s own
/// doc: a scatter's destination is data-dependent, never a plain identity or
/// broadcast projection a consumer could read through).
#[cfg(feature = "reduce-epilogue-fusion")]
fn is_epilogue_fusable_reduce(bound: &BoundOp) -> bool {
    matches!(
        bound.kind,
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            out_scatter: None,
            ..
        }
    )
}

/// One flag per [`NodeId`] this program can name, `true` exactly for a
/// [`is_epilogue_fusable_reduce`] node — [`find_epilogue_source`]'s own
/// lookup table, sized once per [`reduce_epilogue_fusion`] pass rather than
/// re-scanned per candidate.
#[cfg(feature = "reduce-epilogue-fusion")]
fn reduce_epilogue_source_flags(resolved: &[BoundOp]) -> Vec<bool> {
    let node_count = resolved
        .iter()
        .map(|bound| bound.node.0 as usize + 1)
        .max()
        .unwrap_or(0);
    let mut flags = vec![false; node_count];
    for bound in resolved {
        if is_epilogue_fusable_reduce(bound) {
            flags[bound.node.0 as usize] = true;
        }
    }
    flags
}

/// How many DISTINCT resolved ops read each [`NodeId`] this program can
/// name, via any of that consumer's own [`BoundOp::all_read_sources`] —
/// [`reduce_epilogue_fusion`]'s own liveness gate (condition (b): "no OTHER
/// consumer"), computed over the ALREADY-FUSED op list so a node absorbed
/// into a `ComposedBody` upstream (never its own [`BoundOp`]) correctly
/// counts zero rather than needing a separate raw-`Op` walk.
///
/// Counts CONSUMING OPERATIONS, not operand occurrences: production SiLU
/// (`spec.rs`'s `silu` builder) reads its `gate` operand twice within the
/// SAME consumer — directly, and again inside `exp(-gate)` — and chain
/// composition preserves both as separate `operands` slots. Naively counting
/// every slot would see `gate` "referenced" twice and reject condition (b)
/// even though exactly one consumer reads it. Each `bound`'s own reads are
/// deduped to their distinct source [`NodeId`]s before folding into the
/// per-source total, so N reads of the same source by one consumer count as
/// the one reference that consumer actually is.
#[cfg(feature = "reduce-epilogue-fusion")]
fn resolved_reference_counts(resolved: &[BoundOp]) -> BTreeMap<NodeId, u32> {
    let mut counts = BTreeMap::new();
    for bound in resolved {
        let mut sources_read = BTreeSet::new();
        for (source, _, gather) in bound.all_read_sources() {
            sources_read.insert(*source);
            if let Some(lookup) = gather {
                sources_read.insert(lookup.indices);
            }
        }
        for source in sources_read {
            *counts.entry(source).or_insert(0u32) += 1;
        }
    }
    counts
}

/// The one reduce-fold operand `consumer` can absorb into its epilogue, if
/// any: the first operand among `consumer.operands()` — no gather (a
/// gathered read is data-dependent, `apply_reduce_epilogue`'s own doc names
/// this unsupported) — whose node is [`reduce_epilogue_source_flags`]-true.
/// Structural over [`BoundOp`]/[`Layout`] only, at the RESOLVED level — this
/// is what lets the match see straight through however many raw `Op` steps
/// ordinary chain-fusion already folded into `consumer`'s own
/// [`ComposedBody`], the exact one-hop limitation a raw-`Op`-level scan hits
/// (a multi-step tail between the fold and its real, final consumer is
/// already ONE [`BoundOp`] by the time this runs, keyed at the final
/// consumer's own [`NodeId`], not at whichever raw op happened to sit
/// directly after the reduce).
///
/// Multiple `operands` slots naming the SAME reduce are not automatically a
/// conflict: production SiLU reads its reduce-derived `gate` operand once
/// directly and once more inside `exp(-gate)`, and chain composition
/// preserves both as separate slots reading the SAME [`Layout`] (identical
/// projection) — that is ONE logical read repeated, not two. Two slots
/// naming the same reduce through DIFFERENT [`Layout`]s is the real
/// conflict this declines (see below).
#[cfg(feature = "reduce-epilogue-fusion")]
fn find_epilogue_source(consumer: &BoundOp, reduce_flags: &[bool]) -> Option<NodeId> {
    let BoundOpKind::Elementwise { operands, .. } = &consumer.kind else {
        return None;
    };
    if operands.iter().any(|(_, _, gather)| gather.is_some()) {
        return None; // no renderer/evaluator supports a gathered epilogue read.
    }
    let mut found: Option<(NodeId, &Layout)> = None;
    for (node, layout, _) in operands {
        if !reduce_flags.get(node.0 as usize).copied().unwrap_or(false) {
            continue;
        }
        match found {
            None => found = Some((*node, layout)),
            // The SAME slot (same node, same projection) read again — the
            // production-SiLU shape (`gate` used both bare and inside
            // `exp(-gate)`). One logical read; nothing more to record.
            Some((existing_node, existing_layout))
                if existing_node == *node && existing_layout == layout => {}
            // The SAME reduce read through a DIFFERENT projection — a
            // parity-selecting split like "gate = paired[..,0,..]" /
            // "up = paired[..,1,..]" both landing in one consumer.
            // `compose_reduce_epilogue`'s own "implicit fold-result slot"
            // model has room for exactly ONE such read; absorbing the fold
            // here would silently drop whichever occurrence isn't picked as
            // the sentinel while still trying to read the fold's now-gone
            // standalone buffer for the other one. Decline the whole
            // consumer rather than guess which read wins.
            Some((existing_node, _)) if existing_node == *node => return None,
            Some(_) => {}
        }
    }
    found.map(|(node, _)| node)
}

/// One (consumer, reduce) pair a single [`reduce_epilogue_fusion`] pass will
/// merge: `consumer` is a resolved [`BoundOpKind::Elementwise`] whose sole
/// reduce-fold operand (per [`find_epilogue_source`]) is `source`, `source`
/// has no OTHER reader anywhere in `resolved` and is not itself a required
/// output. Whether that operand is read broadcast (the `[s,d]`-shaped
/// "broadcast-reduce" epilogue an RMSNorm-shaped `x * inv_rms` tail needs) or
/// at plain identity (the pre-existing bias/residual epilogue shape) is
/// immaterial here — [`compose_reduce_epilogue`] widens correctly either way
/// from the two [`BoundOp`]s' own recorded extents, never from a name or
/// shape special-cased in this match.
#[cfg(feature = "reduce-epilogue-fusion")]
fn reduce_epilogue_candidates(resolved: &[BoundOp], outputs: &[NodeId]) -> Vec<(NodeId, NodeId)> {
    let reduce_flags = reduce_epilogue_source_flags(resolved);
    let reference_counts = resolved_reference_counts(resolved);
    let mut candidates = Vec::new();
    for bound in resolved {
        let Some(source) = find_epilogue_source(bound, &reduce_flags) else {
            continue;
        };
        if reference_counts.get(&source).copied().unwrap_or(0) != 1 {
            continue; // (b): some OTHER op still reads this fold's output.
        }
        if outputs.contains(&source) {
            continue; // (b): a requested output must still materialize on its own.
        }
        candidates.push((bound.node, source));
    }
    candidates
}

/// The bind-time rewrite [`bind_with_fusion`] runs whenever
/// `reduce-epilogue-fusion` is compiled in: every
/// [`reduce_epilogue_candidates`] match becomes one merged [`BoundOp`] whose
/// `node` is the CONSUMER's id (see [`BoundOpKind::Reduce::epilogue_body`]'s
/// own doc for why), replacing both the standalone reduce and the standalone
/// consumer `resolved` already held. Runs to a FIXPOINT (bounded by
/// `resolved.len()`, so it always terminates — each round strictly shrinks
/// the op count or stops) because an RMSNorm-shaped tail needs TWO rounds:
/// round one absorbs the plain `mean/eps/sqrt/reciprocal` chain into the
/// fold itself (a PLAIN epilogue, output shape unchanged); only after that
/// does the fold's own `NodeId` carry `Keep::Reduce` for
/// [`find_epilogue_source`] to match `x * inv_rms`'s BROADCAST read in round
/// two. A backend with no epilogue renderer must reject a non-default
/// `epilogue_body`/`epilogue_operands` at bind time rather than call this at
/// all with the feature compiled in against data it cannot render — the same
/// capability contract `fuse_cached_attention: false` already gives
/// wgpu/cuda for `CachedAttention`.
#[cfg(feature = "reduce-epilogue-fusion")]
fn reduce_epilogue_fusion(
    mut resolved: Vec<BoundOp>,
    outputs: &[NodeId],
    numeric_policy: NumericPolicy,
) -> Result<Vec<BoundOp>, TensorError> {
    for _ in 0..resolved.len() {
        let candidates = reduce_epilogue_candidates(&resolved, outputs);
        if candidates.is_empty() {
            return Ok(resolved);
        }
        let by_node: BTreeMap<NodeId, &BoundOp> =
            resolved.iter().map(|bound| (bound.node, bound)).collect();
        let mut fused_by_consumer: BTreeMap<NodeId, BoundOp> = BTreeMap::new();
        let mut absorbed: BTreeSet<NodeId> = BTreeSet::new();
        for (consumer, source) in candidates {
            if absorbed.contains(&consumer) || absorbed.contains(&source) {
                continue; // already spoken for by another pair this same round.
            }
            let Some(reduce_bound) = by_node.get(&source).copied() else {
                continue;
            };
            let Some(consumer_bound) = by_node.get(&consumer).copied() else {
                continue;
            };
            let BoundOpKind::Reduce {
                element_body,
                reduce_op,
                init,
                keep,
                operands,
                output_axes,
                out_layout,
                out_scatter: None,
                epilogue_body: inner_epilogue_body,
                epilogue_operands: inner_epilogue_operands,
                ..
            } = &reduce_bound.kind
            else {
                continue; // window-elimination or a prior pass already rewrote this reduce away.
            };
            let Some(broadcast_axes) = epilogue_broadcast_axes_for(
                output_axes,
                &reduce_bound.extents,
                &consumer_bound.extents,
            ) else {
                continue; // (a): consumer must either preserve the fold's own output shape or re-broadcast the WHOLE pre-reduction shape, nothing in between.
            };
            let BoundOpKind::Elementwise {
                operands: consumer_operands,
                ..
            } = &consumer_bound.kind
            else {
                continue;
            };
            let Some((_, source_layout, source_gather)) = consumer_operands
                .iter()
                .find(|(node, _, _)| *node == source)
            else {
                continue;
            };
            if source_gather.is_some()
                || !reads_reduce_output_identically(source_layout, out_layout, output_axes)
            {
                continue; // (a): the fold's own output must be read at genuine identity/broadcast, never gathered or permuted.
            }
            let Some((epilogue_body, epilogue_operands)) = compose_reduce_epilogue(
                output_axes,
                reduce_bound.extents.len(),
                inner_epilogue_body,
                inner_epilogue_operands,
                consumer_bound,
                source,
                numeric_policy,
            ) else {
                continue;
            };
            // Metal exposes buffer indices 0..=30. Count the signature this
            // fused op actually produces rather than capping only its
            // epilogue: fold operands, epilogue operands, one index buffer
            // per gather, output, uniforms, and the shared gather-fault
            // buffer. If it does not fit, leaving the consumer materialized
            // preserves the same algebra with two legal kernels.
            let buffer_binding_count = metal_buffer_binding_count(operands, &epilogue_operands);
            if buffer_binding_count > 31 {
                continue;
            }
            let fused = BoundOp {
                node: consumer,
                dtype: consumer_bound.dtype,
                extents: reduce_bound.extents.clone(),
                kind: BoundOpKind::Reduce {
                    element_body: element_body.clone(),
                    reduce_op: *reduce_op,
                    init: *init,
                    keep: *keep,
                    operands: operands.clone(),
                    output_axes: output_axes.clone(),
                    out_layout: out_layout.clone(),
                    out_scatter: None,
                    epilogue_body,
                    epilogue_operands,
                    epilogue_broadcast_axes: broadcast_axes,
                },
            };
            fused_by_consumer.insert(consumer, fused);
            absorbed.insert(source);
        }
        if fused_by_consumer.is_empty() {
            return Ok(resolved);
        }
        let mut rewritten = Vec::with_capacity(resolved.len());
        for bound in resolved {
            if let Some(fused) = fused_by_consumer.remove(&bound.node) {
                rewritten.push(fused);
            } else if !absorbed.contains(&bound.node) {
                rewritten.push(bound);
            }
        }
        resolved = rewritten;
    }
    Ok(resolved)
}

fn metal_buffer_binding_count(operands: &BoundOperands, epilogue: &BoundOperands) -> usize {
    let gather_count = operands
        .iter()
        .chain(epilogue.iter())
        .filter(|(_, _, gather)| gather.is_some())
        .count();
    operands.len() + epilogue.len() + gather_count + 2 + usize::from(gather_count > 0)
}

/// Which [`BoundOpKind::Reduce::epilogue_broadcast_axes`] value `consumer`'s
/// own fusion needs, or `None` to reject the whole candidate: `Some(empty)`
/// when `consumer_extents` already equals the fold's own OUTPUT shape (the
/// pre-existing, shape-preserving PLAIN epilogue — `output_axes`'s own
/// projection of `reduce_extents`, no axis to re-broadcast over); `Some` of
/// every axis `output_axes` excludes when `consumer_extents` equals the
/// fold's FULL pre-reduction shape instead (the broadcast-reduce shape an
/// RMSNorm-style `x * inv_rms` tail needs); `None` for anything else (a
/// shape this rule was never meant to admit — e.g. a consumer that reads a
/// PARTIAL sub-broadcast of the reduced axes).
#[cfg(feature = "reduce-epilogue-fusion")]
fn epilogue_broadcast_axes_for(
    output_axes: &[u16],
    reduce_extents: &[u64],
    consumer_extents: &[u64],
) -> Option<SmallVec<[u16; MAX_INLINE_RANK]>> {
    let projected: Vec<u64> = output_axes
        .iter()
        .map(|&axis| reduce_extents[axis as usize])
        .collect();
    if consumer_extents == projected.as_slice() {
        return Some(SmallVec::new());
    }
    if consumer_extents == reduce_extents && reduce_extents.len() > output_axes.len() {
        let broadcast_axes = (0..reduce_extents.len() as u16)
            .filter(|axis| !output_axes.contains(axis))
            .collect();
        return Some(broadcast_axes);
    }
    None
}

/// Is `consumer_layout` a genuine identity-or-broadcast read of `source`'s
/// own materialized output — same per-axis stride as `out_layout` on every
/// `output_axes` entry, and stride `0` (a true broadcast, never a permuted
/// or reversed walk) on every OTHER axis `consumer_layout` names. Rejects
/// exactly the shape `a_strided_consumer_map_does_not_fuse` proves: a
/// same-SHAPE but reversed/offset read (`coeff: -1`) has the right rank and
/// element count but the WRONG stride, so [`epilogue_broadcast_axes_for`]'s
/// shape-only check alone would wrongly admit it.
#[cfg(feature = "reduce-epilogue-fusion")]
fn reads_reduce_output_identically(
    consumer_layout: &Layout,
    out_layout: &Layout,
    output_axes: &[u16],
) -> bool {
    if consumer_layout.base != out_layout.base {
        return false;
    }
    if consumer_layout.strides.len() == output_axes.len() {
        // A plain (shape-preserving) read: `consumer_layout` is compact,
        // rank `output_axes.len()`, LOCAL-indexed in `output_axes`'s own
        // order — compare position `index` against `out_layout`'s (full-
        // rank) stride at the GLOBAL axis `output_axes[index]` names.
        return output_axes
            .iter()
            .enumerate()
            .all(|(index, &axis)| consumer_layout.stride(index as u16) == out_layout.stride(axis));
    }
    // A broadcast-reduce read: `consumer_layout` is full rank, GLOBAL-indexed
    // the same as `out_layout` itself — every `output_axes` entry must carry
    // `out_layout`'s own real stride, every OTHER axis must be a genuine
    // broadcast (stride `0`), never a permuted or reversed walk.
    (0..consumer_layout.strides.len() as u16).all(|axis| {
        if output_axes.contains(&axis) {
            consumer_layout.stride(axis) == out_layout.stride(axis)
        } else {
            consumer_layout.stride(axis) == 0
        }
    })
}

/// Re-addresses `layout` (recorded at `output_axes.len()` rank, the reduce's
/// own OUTPUT-axis coordinate space) into `full_rank` coordinates: every
/// axis in `output_axes` keeps its own stride at its real position, every
/// OTHER axis (the reduce's own reduced axis, among others) gets stride `0`
/// — a genuine broadcast, since the fold's OWN result never varied along
/// that axis in the first place. A no-op in the common case
/// (`layout.strides.len() == full_rank` already) because a PLAIN, non-
/// broadcast prior epilogue's own operands are already recorded at the same
/// rank the fold's `extents` always carries.
#[cfg(feature = "reduce-epilogue-fusion")]
fn broadcast_extend_operand(layout: &Layout, output_axes: &[u16], full_rank: usize) -> Layout {
    if layout.strides.len() == full_rank {
        return layout.clone();
    }
    let mut strides = SmallVec::<[i64; MAX_INLINE_RANK]>::from_elem(0, full_rank);
    for (index, &axis) in output_axes.iter().enumerate() {
        if let Some(&stride) = layout.strides.get(index) {
            strides[axis as usize] = stride;
        }
    }
    Layout {
        base: layout.base,
        strides,
    }
}

/// Grafts `consumer`'s own body onto `source`'s fold, composing through
/// whatever epilogue `source` already carries (`inner_epilogue_body`/
/// `inner_epilogue_operands` — the default identity leaf over zero operands
/// on a fold's first fusion, per [`BoundOpKind::Reduce::epilogue_body`]'s own
/// "no epilogue" convention, or a real prior epilogue on a SECOND round —
/// see [`reduce_epilogue_fusion`]'s own doc for why RMSNorm needs both).
/// `full_rank` is the FUSED op's own `extents.len()` (always `source`'s own
/// pre-reduction rank, per [`BoundOp::extents`]'s own doc); every inner
/// operand is broadcast-extended to it via [`broadcast_extend_operand`] so a
/// broadcast-reduce round (`consumer`'s own extents equal to `full_rank`,
/// e.g. RMSNorm's `[s, d]`) and a plain round (`consumer`'s own extents equal
/// to `output_axes`'s smaller projected shape) compose identically: both
/// [`Layout`]s an executor reads are already expressed in the SAME
/// coordinate space `consumer` itself walks, so [`crate::cpu::apply_body`]
/// never needs to know which round produced them.
#[cfg(feature = "reduce-epilogue-fusion")]
fn compose_reduce_epilogue(
    output_axes: &[u16],
    full_rank: usize,
    inner_epilogue_body: &ComposedBody,
    inner_epilogue_operands: &BoundOperands,
    consumer: &BoundOp,
    source: NodeId,
    numeric_policy: NumericPolicy,
) -> Option<(ComposedBody, BoundOperands)> {
    let BoundOpKind::Elementwise {
        body: outer_body,
        operands: outer_operands,
    } = &consumer.kind
    else {
        return None;
    };
    // Every slot naming `source`, not just the first — [`find_epilogue_source`]
    // already guarantees any repeat is the SAME projection (production SiLU
    // reads `gate` once bare, once inside `exp(-gate)`, both slots naming the
    // same source), so every one of them, not only the first, must be
    // redirected to the fold's own implicit result below. Leaving a later
    // occurrence pointed at `source` would reference a producer this fusion
    // is about to remove (`source` is folded into `absorbed`, never emitted
    // as its own `BoundOp`), silently reading a buffer that no longer exists.
    let source_indices: Vec<usize> = outer_operands
        .iter()
        .enumerate()
        .filter(|(_, (node, _, _))| *node == source)
        .map(|(index, _)| index)
        .collect();
    if source_indices.is_empty() {
        return None;
    }

    let mut new_operands = BoundOperands::new();
    let mut outer_remap: Vec<u16> = vec![0; outer_operands.len()];
    for (index, operand) in outer_operands.iter().enumerate() {
        if source_indices.contains(&index) {
            continue; // replaced below by the inner fold's own implicit result.
        }
        outer_remap[index] = new_operands.len() as u16;
        new_operands.push(operand.clone());
    }
    let outer_len = new_operands.len();

    let inner_remap: Vec<u16> = (0..inner_epilogue_operands.len())
        .map(|index| (outer_len + index) as u16)
        .collect();
    for (node, layout, gather) in inner_epilogue_operands {
        new_operands.push((
            *node,
            broadcast_extend_operand(layout, output_axes, full_rank),
            gather.clone(),
        ));
    }
    let raw_fold_slot = new_operands.len() as u16;
    let inner_step_count = inner_epilogue_body.steps.len() as u16;

    // Every BodyStep this graft produces mints through `push_canonical_step`
    // (`proxima-tensor/src/bind.rs`'s own one mint point), not a bare
    // `steps.push(BodyStep { .. })` — a remap can flip an arg from
    // `StepArg::Operand` to `StepArg::Step` (the fold's own implicit result
    // taking the place of a raw operand read), which can leave a commutative
    // step's args in non-canonical order even though both `inner_epilogue_body`
    // and `outer_body` were themselves minted canonically before this graft
    // ever saw them. No constant table survives into this post-composition
    // pass (`reduce_epilogue_fusion` runs over already-`BoundOp`-resolved
    // data, not the original `Op` program), so identity elimination never
    // fires here regardless of `numeric_policy` (an empty `values` slice
    // makes `step_arg_constant` always return `None`) — harmless, since a
    // remap only changes which slot an arg names, never introduces a new
    // literal identity value, so `push_canonical_step` always appends exactly
    // one step here and the `inner_step_count`/`outer_remap` index arithmetic
    // below still lines up with the pushed order. `numeric_policy` is still
    // threaded through (rather than hard-coding a policy here) so this call
    // site tracks whatever a future constant-aware version of this graft
    // would need, instead of silently diverging from the caller's own
    // policy.
    let constants = Constants {
        ones: &[],
        values: &[],
        numeric_policy,
    };
    let mut steps: Vec<BodyStep> = Vec::new();
    let mut absorbed: Vec<NodeId> = Vec::new();
    let mut state = ComposeState {
        steps: &mut steps,
        operands: &mut new_operands,
        absorbed: &mut absorbed,
    };
    for step in &inner_epilogue_body.steps {
        let args = step
            .args
            .iter()
            .map(|arg| match arg {
                StepArg::Operand(index) => {
                    let old = *index as usize;
                    if old == inner_epilogue_operands.len() {
                        StepArg::Operand(raw_fold_slot)
                    } else {
                        StepArg::Operand(inner_remap[old])
                    }
                }
                StepArg::Step(step_index) => StepArg::Step(*step_index),
            })
            .collect();
        push_canonical_step(&mut state, step.op, args, constants);
    }
    for step in &outer_body.steps {
        let args = step
            .args
            .iter()
            .map(|arg| match arg {
                StepArg::Operand(index) => {
                    let old = *index as usize;
                    if source_indices.contains(&old) {
                        StepArg::Step(inner_step_count - 1)
                    } else {
                        StepArg::Operand(outer_remap[old])
                    }
                }
                StepArg::Step(step_index) => StepArg::Step(*step_index + inner_step_count),
            })
            .collect();
        push_canonical_step(&mut state, step.op, args, constants);
    }
    Some((ComposedBody { steps }, new_operands))
}

/// The single choke point every [`bind`]/[`bind_with_fusion`]/
/// [`bind_without_reduce_epilogue_fusion`] route eventually calls (ROW 541,
/// `docs/discipline.md`). Binds only the ops [`live::reachable`] reaches
/// from `outputs` through operands, gather indices, and reduce `out_map`
/// indices — a program built for a wider caller (a shared spec module
/// producing both a prefill and a decode graph, say) never dispatches the
/// prefill-only tail decode's own `outputs` do not reach. Node ids stay
/// exactly [`program`]'s own positions: an unreachable position is skipped
/// via [`BoundOpBuilder::skip`], never renumbered, since every backward
/// reference elsewhere in the program is a raw index into this same slice.
fn bind_plain(
    program: &[Op],
    shapes: &Shapes,
    outputs: &[NodeId],
    numeric_policy: NumericPolicy,
) -> Result<Vec<BoundOp>, TensorError> {
    let retires = live::annotate(program, outputs);
    let reachable = live::reachable(program, outputs);
    let building = BoundOpBuilder::new(retires, numeric_policy);
    let mut built = Vec::new();
    for (position, expr) in program.iter().enumerate() {
        if reachable.contains(&NodeId(position as u32)) {
            built.extend(building.push(expr, shapes)?);
        } else {
            building.skip();
        }
    }
    built.extend(building.finish(shapes)?);
    Ok(built)
}

/// Every `program` position holding an [`Op::Input`] — the block-input node
/// order [`bind`]'s own caller binds real data against. Backend-neutral (a
/// pure scan over `&[Op]`), so any executor consuming this module's
/// [`BoundOp`]s reads the SAME node order [`crate::cpu`]'s own evaluators do
/// rather than re-deriving it — before this function was `pub`, `cpu.rs` and
/// `omega/src/metal.rs` each carried a byte-identical private copy (the
/// exact "second, parallel emitter" [`BoundOp::dtype`]'s own doc says this
/// module exists to avoid).
#[must_use]
pub fn block_node_ids(program: &[Op]) -> Vec<NodeId> {
    program
        .iter()
        .enumerate()
        .filter(|(_, expr)| matches!(expr, Op::Input { .. }))
        .map(|(position, _)| NodeId(position as u32))
        .collect()
}

/// Every node referenced as a gather's `indices` anywhere in `program` — the
/// one class of non-float32 node a float-only executor's own dtype gate must
/// exempt (an index value is an exact integer carried in a float buffer, per
/// [`crate::map::IndexMap::Computed`]'s own doc). Same reuse argument as
/// [`block_node_ids`]: this was a byte-identical private copy in both
/// `cpu.rs` and `omega/src/metal.rs`.
#[must_use]
pub fn index_node_ids(program: &[Op]) -> BTreeSet<NodeId> {
    let mut nodes = BTreeSet::new();
    for expr in program {
        match expr {
            Op::Input { .. } | Op::Iota { .. } | Op::Constant { .. } => {}
            Op::Elementwise { operands, .. } => {
                for (_, map) in operands {
                    push_indices_node(map, &mut nodes);
                }
            }
            Op::Reduce(reduce) => {
                push_indices_node(&reduce.in_map, &mut nodes);
                push_indices_node(&reduce.out_map, &mut nodes);
            }
        }
    }
    nodes
}

/// `pub(crate)`, not private: [`crate::cpu`]'s own `referenced_node_ids`
/// walks the identical `Elementwise`/`Reduce` operand-map shape as
/// [`index_node_ids`] above, over a different node set, so it shares this
/// helper rather than carrying a third copy.
pub(crate) fn push_indices_node(map: &IndexMap, nodes: &mut BTreeSet<NodeId>) {
    if let IndexMap::Computed { indices, .. } = map {
        nodes.insert(*indices);
    }
}

/// Per-node retire sets over the *emitted* (post-fusion) node sequence:
/// `result[p]` is every node whose last read is `resolved[p]`. Distinct from
/// [`live::annotate`], which computes liveness over the PROGRAM's own
/// timeline before fusion has decided which zips never materialize at all —
/// this one runs after [`bind`], over [`BoundOp::operands`] directly, so it
/// sees the fused shape an executor actually walks. Same reuse argument as
/// [`block_node_ids`]/[`index_node_ids`]: `cpu.rs`'s private `node_retirement`
/// and `omega/src/metal.rs`'s private `bound_op_retirement` were the
/// identical computation under two names.
/// The one read-source walk both [`node_retirement`] and [`node_last_reader`]
/// need, so the two never drift into "identical computation under two names"
/// (the exact defect this module's own doc says it exists to end).
fn walk_last_reads<F: FnMut(NodeId, usize)>(resolved: &[BoundOp], mut record: F) {
    for (position, node) in resolved.iter().enumerate() {
        for (source, _, gather) in node.all_read_sources() {
            record(*source, position);
            if let Some(gather_access) = gather {
                record(gather_access.indices, position);
            }
        }
        if let BoundOpKind::Reduce {
            out_scatter: Some(lookup),
            ..
        } = &node.kind
        {
            record(lookup.indices, position);
        }
    }
}

#[must_use]
pub fn node_retirement(resolved: &[BoundOp], outputs: &[NodeId]) -> Vec<Vec<NodeId>> {
    let outputs: BTreeSet<NodeId> = outputs.iter().copied().collect();
    let mut last_use: BTreeMap<NodeId, usize> = BTreeMap::new();
    walk_last_reads(resolved, |node, position| {
        last_use.insert(node, position);
    });

    let mut retires = vec![Vec::new(); resolved.len()];
    for (node, position) in last_use {
        if !outputs.contains(&node) {
            #[cfg(feature = "instrument")]
            debug!(
                node = node.0,
                kind = "node_retirement",
                decision = "retired",
                into = resolved[position].node.0,
                "node retired -- last consumer read it at this resolved position"
            );
            retires[position].push(node);
        }
    }
    retires
}

/// Dense node -> last-reader-position table over the same emitted sequence
/// [`node_retirement`] walks, indexed by `NodeId.0`, `u32::MAX` where a node
/// is never read. An executor's per-op retirement decision is then a single
/// array read (`last_reader[node] == position`) instead of a scan over
/// remaining ops or the current op's own operand list — see
/// `omega/src/metal.rs`'s `execute_plan_inner` for the consumer this replaced
/// a per-op `iter().any()` forward scan in.
#[must_use]
pub fn node_last_reader(resolved: &[BoundOp], node_count: usize) -> Vec<u32> {
    let mut last_reader = vec![u32::MAX; node_count];
    walk_last_reads(resolved, |node, position| {
        last_reader[node.0 as usize] = position as u32;
    });
    last_reader
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Drives `resolved` through [`crate::cpu::Interpreter`] the same way
    /// `reduce_epilogue_fusion_tests::run_resolved` does — inlined rather
    /// than shared across the module boundary, since this is the only
    /// consumer at this scope.
    fn run_resolved(
        program_len: usize,
        resolved: &[BoundOp],
        inputs: Vec<(NodeId, Vec<f32>)>,
    ) -> Vec<Option<Vec<f32>>> {
        use core::pin::pin;
        use core::task::{Context, Poll, Waker};

        use crate::cpu::Interpreter;

        let mut buffers: Vec<Option<Vec<f32>>> = alloc::vec![None; program_len];
        for (node, data) in inputs {
            buffers[node.0 as usize] = Some(data);
        }
        let interpreter = Interpreter::new(&mut buffers);
        for chunk in resolved.chunks(READY_BATCH_CAPACITY) {
            let batch: ReadyBatch = chunk.iter().cloned().collect();
            let waker = Waker::noop();
            let mut context = Context::from_waker(waker);
            let mut future = pin!(interpreter.call(batch));
            match future.as_mut().poll(&mut context) {
                Poll::Ready(result) => {
                    result.expect("resolved batch computes");
                }
                Poll::Pending => unreachable!("cpu pipes never yield: no internal .await"),
            }
        }
        buffers
    }

    /// The last node `program` builds -- what every fixture in this module
    /// treats as "the answer" by construction (`op.rs`'s own doc: "the last
    /// element is the root"). [`bind_plain`]'s reachability pass (ROW 541,
    /// `docs/discipline.md`) now binds only what `outputs` actually names,
    /// so a fixture that wants its whole constructed chain bound must pass
    /// this instead of `&[]` -- an empty `outputs` correctly binds nothing.
    fn terminal(program: &[Op]) -> NodeId {
        NodeId((program.len() - 1) as u32)
    }

    /// `max(x, -inf)` -- owner counterexample (2026-09-06): `f32::max`'s own
    /// "if one argument is NaN, return the other" rule means
    /// `NaN.max(-inf) == -inf`, but eliminating the op (returning the
    /// survivor `x`) would produce `NaN` instead. Under the library default
    /// ([`NumericPolicy::bit_exact()`]) the `Maximum` step must survive and
    /// compute the real `-inf`; only once the caller grants `nan_assumptions`
    /// does [`identity_element_signed_zero_nan`] fire and collapse the op to
    /// `x`, producing the DIFFERENT value `NaN`.
    #[test]
    fn maximum_of_nan_and_negative_infinity_differs_by_numeric_policy() {
        use crate::op::{Extent, append};

        let mut program = Vec::new();
        let x = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(1)],
                name: None,
            },
        );
        let neg_inf = append(
            &mut program,
            Op::Constant {
                dtype: DType::Float32,
                shape: Vec::new(),
                value: f32::NEG_INFINITY,
            },
        );
        let identity = || IndexMap::Affine(map::projection(1, &[0]));
        let broadcast_scalar = || IndexMap::Affine(map::projection(1, &[]));
        let output = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Maximum,
                operands: alloc::vec![(x, identity()), (neg_inf, broadcast_scalar())],
                name: None,
            },
        );
        let shapes = shape::infer(&program, &[]).expect("max(x,-inf) program infers");

        let bit_exact = bind_with_fusion(
            &program,
            &shapes,
            &[output],
            true,
            NumericPolicy::bit_exact(),
        )
        .expect("bit-exact bind succeeds");
        let bit_exact_buffers = run_resolved(
            program.len(),
            &bit_exact,
            alloc::vec![(x, alloc::vec![f32::NAN])],
        );
        let bit_exact_result = bit_exact_buffers[output.0 as usize]
            .as_ref()
            .expect("bit-exact output present")[0];
        assert_eq!(
            bit_exact_result,
            f32::NEG_INFINITY,
            "BitExact must compute the real max(NaN, -inf) == -inf, not eliminate the op"
        );

        let nan_assumption_policy = NumericPolicy {
            nan_assumptions: true,
            ..NumericPolicy::bit_exact()
        };
        let nan_assumption_bound =
            bind_with_fusion(&program, &shapes, &[output], true, nan_assumption_policy)
                .expect("nan-assumption bind succeeds");
        let nan_assumption_buffers = run_resolved(
            program.len(),
            &nan_assumption_bound,
            alloc::vec![(x, alloc::vec![f32::NAN])],
        );
        let nan_assumption_result = nan_assumption_buffers[output.0 as usize]
            .as_ref()
            .expect("nan-assumption output present")[0];
        assert!(
            nan_assumption_result.is_nan(),
            "nan_assumptions admits IdentityEliminationNanAssumption, collapsing to the \
             surviving operand x == NaN, got {nan_assumption_result}"
        );
    }

    /// `x + 0.0` -- owner counterexample (2026-09-06): `(-0.0) + 0.0`
    /// evaluates to `+0.0` under real `f32` addition, but eliminating the op
    /// (returning the survivor `x == -0.0`) keeps the sign bit `BitExact`
    /// must not silently flip.
    #[test]
    fn add_zero_to_negative_zero_differs_by_numeric_policy() {
        use crate::op::{Extent, append};

        let mut program = Vec::new();
        let x = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(1)],
                name: None,
            },
        );
        let zero = append(
            &mut program,
            Op::Constant {
                dtype: DType::Float32,
                shape: Vec::new(),
                value: 0.0,
            },
        );
        let identity = || IndexMap::Affine(map::projection(1, &[0]));
        let broadcast_scalar = || IndexMap::Affine(map::projection(1, &[]));
        let output = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![(x, identity()), (zero, broadcast_scalar())],
                name: None,
            },
        );
        let shapes = shape::infer(&program, &[]).expect("x+0 program infers");

        let bit_exact = bind_with_fusion(
            &program,
            &shapes,
            &[output],
            true,
            NumericPolicy::bit_exact(),
        )
        .expect("bit-exact bind succeeds");
        let bit_exact_buffers = run_resolved(
            program.len(),
            &bit_exact,
            alloc::vec![(x, alloc::vec![-0.0f32])],
        );
        let bit_exact_result = bit_exact_buffers[output.0 as usize]
            .as_ref()
            .expect("bit-exact output present")[0];
        assert_eq!(
            bit_exact_result.to_bits(),
            0.0f32.to_bits(),
            "BitExact must compute the real (-0.0)+0.0 == +0.0, not eliminate the op and keep -0.0"
        );

        let signed_zero_policy = NumericPolicy {
            signed_zero: true,
            ..NumericPolicy::bit_exact()
        };
        let signed_zero_bound =
            bind_with_fusion(&program, &shapes, &[output], true, signed_zero_policy)
                .expect("signed-zero bind succeeds");
        let signed_zero_buffers = run_resolved(
            program.len(),
            &signed_zero_bound,
            alloc::vec![(x, alloc::vec![-0.0f32])],
        );
        let signed_zero_result = signed_zero_buffers[output.0 as usize]
            .as_ref()
            .expect("signed-zero output present")[0];
        assert_eq!(
            signed_zero_result.to_bits(),
            (-0.0f32).to_bits(),
            "signed_zero admits IdentityEliminationSignedZero, collapsing to the surviving \
             operand x == -0.0, got {signed_zero_result}"
        );
    }

    /// The split this design makes at the whole-bind level: granting
    /// `nan_assumptions` alone must NOT also eliminate `x+0` -- proves the
    /// two permissions stay independent through the full bind pipeline, not
    /// just at [`admit`] in isolation.
    #[test]
    fn nan_assumption_alone_does_not_eliminate_add_zero() {
        use crate::op::{Extent, append};

        let mut program = Vec::new();
        let x = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(1)],
                name: None,
            },
        );
        let zero = append(
            &mut program,
            Op::Constant {
                dtype: DType::Float32,
                shape: Vec::new(),
                value: 0.0,
            },
        );
        let identity = || IndexMap::Affine(map::projection(1, &[0]));
        let broadcast_scalar = || IndexMap::Affine(map::projection(1, &[]));
        let output = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![(x, identity()), (zero, broadcast_scalar())],
                name: None,
            },
        );
        let shapes = shape::infer(&program, &[]).expect("x+0 program infers");
        let nan_assumption_policy = NumericPolicy {
            nan_assumptions: true,
            ..NumericPolicy::bit_exact()
        };
        let bound = bind_with_fusion(&program, &shapes, &[output], true, nan_assumption_policy)
            .expect("nan-assumption bind succeeds");
        let buffers = run_resolved(
            program.len(),
            &bound,
            alloc::vec![(x, alloc::vec![-0.0f32])],
        );
        let result = buffers[output.0 as usize].as_ref().expect("output present")[0];
        assert_eq!(
            result.to_bits(),
            0.0f32.to_bits(),
            "nan_assumptions must not grant signed_zero's x+0 elimination -- the Add step must \
             survive and compute the real (-0.0)+0.0 == +0.0"
        );
    }

    #[test]
    fn name_reports_the_variant_backends_render_error_messages_with() {
        let cached_attention = BoundOpKind::CachedAttention {
            operands: Vec::new(),
            query_rows: 0,
            cached_key_rows: 0,
            new_key_rows: 0,
            kv_heads: 0,
            query_groups: 0,
            head_dim: 0,
            rotary_dim: 0,
            scale: 0.0,
            cached_lower_inclusive: 0,
            new_upper_inclusive: 0,
        };
        let elementwise = BoundOpKind::Elementwise {
            body: ComposedBody::leaf(ScalarOp::Identity),
            operands: Vec::new(),
        };
        let reduce_fold = BoundOpKind::Reduce {
            element_body: ComposedBody::leaf(ScalarOp::Identity),
            reduce_op: ScalarOp::Add,
            init: ReduceInit::Zero,
            keep: Keep::Reduce,
            operands: Vec::new(),
            output_axes: SmallVec::new(),
            out_layout: Layout {
                base: 0,
                strides: SmallVec::new(),
            },
            out_scatter: None,
            epilogue_body: ComposedBody::leaf(ScalarOp::Identity),
            epilogue_operands: Vec::new(),
            epilogue_broadcast_axes: SmallVec::new(),
        };
        let reduce_scan = BoundOpKind::Reduce {
            element_body: ComposedBody::leaf(ScalarOp::Identity),
            reduce_op: ScalarOp::Add,
            init: ReduceInit::Zero,
            keep: Keep::Scan,
            operands: Vec::new(),
            output_axes: SmallVec::new(),
            out_layout: Layout {
                base: 0,
                strides: SmallVec::new(),
            },
            out_scatter: None,
            epilogue_body: ComposedBody::leaf(ScalarOp::Identity),
            epilogue_operands: Vec::new(),
            epilogue_broadcast_axes: SmallVec::new(),
        };

        assert_eq!(cached_attention.name(), "cached_attention");
        assert_eq!(elementwise.name(), "elementwise");
        assert_eq!(reduce_fold.name(), "keep::reduce fold");
        assert_eq!(reduce_scan.name(), "keep::scan fold");
        assert_eq!(BoundOpKind::Iota.name(), "iota");
        assert_eq!(BoundOpKind::Constant { value: 0.0 }.name(), "constant");
    }

    /// Whether `bound`'s `epilogue_body` differs from the identity leaf every
    /// `BoundOpKind::Reduce` starts with -- the only witness, on the BOUND
    /// output itself, that `reduce_epilogue_fusion` actually folded a
    /// consumer into this reduce (as opposed to merely being a candidate the
    /// raw-`Op`-shaped census in `reduce_epilogue_candidates` proposed but
    /// that never applied because cached-attention fusion had already
    /// absorbed one side of the pair).
    #[cfg(feature = "reduce-epilogue-fusion")]
    fn count_fused_epilogues(bound: &[BoundOp]) -> usize {
        bound
            .iter()
            .filter(|op| {
                matches!(&op.kind, BoundOpKind::Reduce { epilogue_body, .. }
                    if *epilogue_body != ComposedBody::leaf(ScalarOp::Identity))
            })
            .count()
    }

    #[test]
    #[cfg(feature = "cached-attention-streaming")]
    fn cached_attention_rewrite_replaces_the_bound_attention_subgraph() {
        let (program, logits, roots) =
            crate::spec::mistral_cached_forward_program(32, 16, 24, 4, 2, 4, 1)
                .expect("cached attention fixture builds");
        let shapes = crate::shape::infer(&program, &[1, 1]).expect("cached attention infers");
        let mut requested = alloc::vec![logits];
        for cache_roots in &roots {
            requested.extend_from_slice(&[cache_roots.0, cache_roots.1, cache_roots.2]);
        }
        let outputs: &[NodeId] = &requested;
        let plain = bind_plain(&program, &shapes, outputs, NumericPolicy::bit_exact())
            .expect("plain bind succeeds");
        let cached_only = bind_cached_attention_fusion(
            &program,
            &shapes,
            outputs,
            true,
            NumericPolicy::bit_exact(),
        )
        .expect("cached-attention-only bind succeeds");
        let rewritten = bind(&program, &shapes, outputs, NumericPolicy::bit_exact())
            .expect("rewritten bind succeeds");

        assert_eq!(plain.len(), 47, "fixture baseline bound operation count");
        assert_eq!(
            cached_only.len(),
            25,
            "fixture cached-attention-only fused bound operation count"
        );
        // `reduce-epilogue-fusion` is a second, independent bind-time pass
        // that runs after the cached-attention rewrite this test targets,
        // additionally folding an epilogue-eligible Elementwise into its
        // Reduce whenever the feature is compiled in. Asserting the relation
        // against `count_fused_epilogues`'s own read of `rewritten`, rather
        // than a second hardcoded literal for the post-epilogue count, keeps
        // this honest across that feature's on/off states instead of
        // silently asserting the pre-epilogue number under both (the defect
        // this row fixes: `docs/discipline.md` for the mechanism).
        #[cfg(feature = "reduce-epilogue-fusion")]
        assert_eq!(
            cached_only.len() - rewritten.len(),
            count_fused_epilogues(&rewritten),
            "every bound op the reduce-epilogue pass removed corresponds to \
             one reduce in the rewritten program that actually carries a \
             fused epilogue"
        );
        #[cfg(not(feature = "reduce-epilogue-fusion"))]
        assert_eq!(
            rewritten.len(),
            cached_only.len(),
            "reduce-epilogue-fusion is compiled out; the cached-attention-only \
             count is final"
        );
        assert_eq!(
            rewritten
                .iter()
                .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
                .count(),
            1,
            "one-layer fixture must receive one fused step"
        );
        assert!(
            rewritten
                .iter()
                .any(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        );
    }

    /// ROW 364's own artifact: the ACTUAL bound program the real openchat
    /// decode fixture uses. The production decode loop
    /// (`proxima-model-interop::generate::build_single_range_program`)
    /// calls `mistral_single_range_cached_forward_program` with
    /// `DuplicateHeadPosition::None` -- NOT `mistral_cached_forward_program`,
    /// the dual-range builder an earlier draft of this row's census
    /// mistakenly used. The dual-range builder's toy fixture already
    /// carried a fused SiLU epilogue on both main and this branch (a
    /// coincidence of that builder's own gate reduce shape), so it could
    /// not show what this row actually changes; this test builds through
    /// the SAME single-range path production takes, one layer,
    /// `bind::bind` (fusion on), printed op-by-op so a diff against the
    /// same test run on main names exactly which ops the epilogue-fusion
    /// landing removed or reshaped. `--nocapture` to see the list; the
    /// assertion below is the mechanical guard that the count does not
    /// silently drift once this is landed as a real (non-throwaway) test.
    ///
    /// Gated on `reduce-epilogue-fusion` (the feature `bind()` actually
    /// consults before folding a post-reduce tail into its `Reduce`'s
    /// `epilogue_body` -- neither feature is in this crate's own `default`
    /// set, see `Cargo.toml`) AND `cached-attention-streaming` (needed only
    /// to keep this build free of the pre-existing, unrelated
    /// `count_fused_epilogues` dead-code trap that fires when
    /// `reduce-epilogue-fusion` is compiled in alone -- that function has no
    /// caller outside the `cached-attention-streaming`-gated test below it).
    /// Ungated, this test asserted the FUSED count (28) against whatever
    /// `bind()` produces under the ambient feature set of the invoking
    /// `cargo`/`nextest` command -- silently unfused (37 ops, not even the
    /// pre-fusion baseline of 34, because `edbf2d90`'s slot-discovery fix now
    /// also folds shapes the old literal-slot admission never recognized)
    /// any time `reduce-epilogue-fusion` was not separately requested, which
    /// is every default invocation this crate's own gate script runs.
    #[test]
    #[cfg(all(
        feature = "reduce-epilogue-fusion",
        feature = "cached-attention-streaming"
    ))]
    fn row_364_per_layer_bound_op_list() {
        let (program, logits, roots, _duplicate_head_scratch) =
            crate::spec::mistral_single_range_cached_forward_program(
                32,
                16,
                24,
                4,
                2,
                4,
                1,
                false,
                crate::spec::DuplicateHeadPosition::None,
                false,
            )
            .expect("one-layer single-range decode fixture builds");
        let shapes = crate::shape::infer(&program, &[1, 1]).expect("cached decode fixture infers");
        let mut outputs = alloc::vec![logits];
        for (even, odd, value) in &roots {
            outputs.extend_from_slice(&[*even, *odd, *value]);
        }
        let bound = bind(&program, &shapes, &outputs, NumericPolicy::bit_exact())
            .expect("the cached decode fixture binds through the real bind() path");

        for (index, op) in bound.iter().enumerate() {
            match &op.kind {
                BoundOpKind::Elementwise { body, .. } => {
                    let step_ops: Vec<ScalarOp> = body.steps.iter().map(|step| step.op).collect();
                    std::println!("row364 index={index} kind=elementwise step_ops={step_ops:?}");
                }
                BoundOpKind::Reduce {
                    epilogue_body,
                    epilogue_broadcast_axes,
                    ..
                } => {
                    let epilogue_ops: Vec<ScalarOp> =
                        epilogue_body.steps.iter().map(|step| step.op).collect();
                    std::println!(
                        "row364 index={index} kind={} epilogue_broadcast_axes={epilogue_broadcast_axes:?} epilogue_ops={epilogue_ops:?}",
                        op.kind.name()
                    );
                }
                _ => {
                    std::println!("row364 index={index} kind={}", op.kind.name());
                }
            }
        }

        assert_eq!(
            bound.len(),
            28,
            "row 364 artifact: one-layer single-range decode program's bound op count \
             (main at the same shape through the same builder: 34 -- three RMSNorm \
             sites each drop from a Reduce plus a separate Elementwise tail to one \
             fused Reduce, see `docs/discipline.md` ROW 364)"
        );
    }

    #[test]
    #[cfg(feature = "cached-attention-streaming")]
    fn cached_attention_rewrite_accepts_the_omega_nonempty_cache_fixture() {
        let (program, logits, cache_roots) =
            crate::spec::mistral_cached_forward_program(64, 64, 128, 4, 2, 16, 2)
                .expect("omega cached attention fixture builds");
        let mut outputs = alloc::vec![logits];
        for (even, odd, value) in cache_roots {
            outputs.extend_from_slice(&[even, odd, value]);
        }
        let shapes =
            crate::shape::infer(&program, &[1, 5]).expect("omega cached attention fixture infers");
        let rewritten = bind(&program, &shapes, &outputs, NumericPolicy::bit_exact())
            .expect("omega cached attention fixture binds");

        assert_eq!(
            rewritten
                .iter()
                .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
                .count(),
            2,
            "each production-shaped layer must receive its own fused step"
        );
        assert!(
            rewritten
                .iter()
                .any(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        );
    }

    /// [`cached_attention_rewrite_accepts_the_omega_nonempty_cache_fixture`]'s
    /// GQA-plus-QK-norm counterpart -- the real Qwen3-1.7B shape's two
    /// distinguishing features (`query_heads != kv_heads`, split-half RoPE
    /// plus per-head `q_norm`/`k_norm`) neither of which that fixture
    /// exercises (it is Mistral-shaped: GQA but no QK-norm). Asserting the
    /// engagement COUNT here, not a log line reading a runtime counter, is
    /// the point: `proxima_tensor::instrument::path_totals().
    /// op_kind_cached_attention` (`instrument.rs:1515`) increments only
    /// from `cpu::run_node_into`'s `CachedAttention` arm
    /// (`proxima-tensor/src/cpu.rs:5210-5215`) -- `omega::metal`'s own
    /// `BoundOpKind::CachedAttention` dispatch (`omega/src/metal.rs:4206`
    /// onward, the actual production Metal execution path
    /// `generate.rs`'s metal-feature `BackendRuntime::evaluate` calls) never
    /// touches that counter. On a Metal build that counter reads 0 on every
    /// step regardless of whether the fusion engaged -- it is silent on the
    /// one backend real decode runs on, not evidence the matcher rejected
    /// the shape. This bind-time count is backend-agnostic and is the
    /// correct place to assert engagement.
    #[test]
    #[cfg(feature = "cached-attention-streaming")]
    fn cached_attention_rewrite_accepts_the_qwen3_gqa_qk_norm_fixture() {
        let (program, logits, cache_roots) =
            crate::spec::qwen3_cached_forward_program(64, 64, 128, 4, 2, 16, 2)
                .expect("qwen3 gqa+qk_norm fixture builds");
        let mut outputs = alloc::vec![logits];
        for (even, odd, value) in cache_roots {
            outputs.extend_from_slice(&[even, odd, value]);
        }
        let shapes =
            crate::shape::infer(&program, &[1, 5]).expect("qwen3 gqa+qk_norm fixture infers");
        let rewritten = bind(&program, &shapes, &outputs, NumericPolicy::bit_exact())
            .expect("qwen3 gqa+qk_norm fixture binds");

        assert_eq!(
            rewritten
                .iter()
                .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
                .count(),
            2,
            "each GQA+QK-norm layer must receive its own fused step -- \
             the matcher accepts this shape; a regression here is a real \
             matcher rejection, not a dead runtime counter"
        );
    }

    /// The real openchat-3.5/Mistral-7B shape (`vocab=32_002`,
    /// `hidden=4096`, `ffn=14336`, `32` query heads, `8` KV heads,
    /// `head_dim=128`, `32` layers) bound at one new token against a
    /// 71-position merged range — the same fixture
    /// `spec::tests::the_single_range_cache_fold_node_budget_is_measured_
    /// against_the_two_range_baseline` measures raw numbers for, asserted
    /// here as a hard regression gate rather than a printed `println!`.
    /// MEASURED, not derived from a dispatch-count census: `plain.len()`
    /// (`939`) is this exact fixture's own baseline bound-op count with the
    /// fusion matcher returning zero candidates (`cached_attention_single_
    /// range_candidates` never firing is exactly the pre-existing defect
    /// this row fixes); the cached-attention-only fused count (`619`) is
    /// what fusing one `BoundOpKind::CachedAttention` per layer actually
    /// removes -- 320 bound ops over 32 layers (10/layer), not the 6/layer a
    /// raw-`Op` count would suggest, because `BoundOpBuilder` already fuses
    /// several of the unfused chain's `Elementwise` nodes into their
    /// consuming `Reduce` before this matcher ever runs. When
    /// `reduce-epilogue-fusion` is also compiled in, a second independent
    /// bind-time pass additionally folds 96 epilogue-eligible reduces on top
    /// of that (`619 -> 523`). The expected count below is checked against
    /// `count_fused_epilogues`'s own read of which `rewritten` reduces
    /// actually carry a fused epilogue body, not
    /// `reduce_epilogue_candidates`'s raw-`Op`-shaped census directly --
    /// that census over-counts here (225 candidates on this fixture, not
    /// 96) because most of its matches sit on nodes the cached-attention
    /// rewrite already absorbed into a `BoundOpKind::CachedAttention` before
    /// `reduce_epilogue_fusion` ever runs, so they silently fail its
    /// `by_node` lookup instead of applying. Asserting a second hardcoded
    /// literal for the post-epilogue count instead of this relation is
    /// exactly the defect this row fixes: it asserted the pre-epilogue `619`
    /// against an actual `523` for a full owner-visible session.
    #[test]
    #[cfg(feature = "cached-attention-streaming")]
    fn single_range_cached_attention_fuses_one_step_per_layer_on_the_real_openchat_shape() {
        let (program, logits, cache_roots, _) =
            crate::spec::mistral_single_range_cached_forward_program(
                32_002,
                4096,
                14336,
                32,
                8,
                128,
                32,
                false,
                crate::spec::DuplicateHeadPosition::None,
                false,
            )
            .expect("openchat-shaped single-range forward pass lowers to a program");
        let mut outputs = alloc::vec![logits];
        for (even, odd, value) in &cache_roots {
            outputs.extend_from_slice(&[*even, *odd, *value]);
        }
        let shapes = crate::shape::infer(&program, &[1, 71])
            .expect("one new position against a 71-position merged range infers");
        let plain = bind_plain(&program, &shapes, &outputs, NumericPolicy::bit_exact())
            .expect("plain bind succeeds");
        let cached_only = bind_cached_attention_fusion(
            &program,
            &shapes,
            &outputs,
            true,
            NumericPolicy::bit_exact(),
        )
        .expect("cached-attention-only bind succeeds");
        let rewritten = bind(&program, &shapes, &outputs, NumericPolicy::bit_exact())
            .expect("fused bind succeeds");

        assert_eq!(
            plain.len(),
            939,
            "openchat-shaped single-range baseline bound operation count"
        );
        assert_eq!(
            cached_only.len(),
            619,
            "openchat-shaped single-range cached-attention-only fused bound operation count"
        );
        #[cfg(feature = "reduce-epilogue-fusion")]
        assert_eq!(
            cached_only.len() - rewritten.len(),
            count_fused_epilogues(&rewritten),
            "every bound op the reduce-epilogue pass removed corresponds to \
             one reduce in the rewritten program that actually carries a \
             fused epilogue"
        );
        #[cfg(not(feature = "reduce-epilogue-fusion"))]
        assert_eq!(
            rewritten.len(),
            cached_only.len(),
            "reduce-epilogue-fusion is compiled out; the cached-attention-only \
             count is final"
        );
        assert_eq!(
            rewritten
                .iter()
                .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
                .count(),
            32,
            "one fused cached-attention step per layer on the real openchat shape"
        );
    }

    /// Regression for ROW 366 (`fix/merged-kv-attention-bounds`): the
    /// single-range fusion used to duplicate the same bucketed-capacity
    /// key/value shape into BOTH `cached_key_rows` and `new_key_rows`,
    /// neutering the cached half with an unreachable `[i64::MAX, i64::MAX]`
    /// band -- a kernel that iterates a fabricated cached half every call.
    /// The merged-KV form has no separate cached range at all, so the fused
    /// op declares `cached_key_rows: 0`: an empty range the kernel skips
    /// entirely, not a live range it walks and discards.
    #[test]
    #[cfg(feature = "cached-attention-streaming")]
    fn single_range_fusion_declares_an_empty_cached_range_not_a_duplicated_capacity() {
        let (program, logits, cache_roots, _) =
            crate::spec::mistral_single_range_cached_forward_program(
                32,
                16,
                24,
                4,
                2,
                4,
                1,
                false,
                crate::spec::DuplicateHeadPosition::None,
                false,
            )
            .expect("single-range fixture builds");
        let mut outputs = alloc::vec![logits];
        for (even, odd, value) in &cache_roots {
            outputs.extend_from_slice(&[*even, *odd, *value]);
        }
        let shapes = crate::shape::infer(&program, &[1, 5]).expect("single-range fixture infers");
        let resolved = bind_plain(&program, &shapes, &outputs, NumericPolicy::bit_exact())
            .expect("plain bind succeeds");

        let candidates =
            cached_attention_single_range_candidates(&program, &shapes, &resolved, &outputs);
        assert!(
            !candidates.is_empty(),
            "the fixture must still produce a fusable single-range candidate"
        );
        let BoundOpKind::CachedAttention {
            cached_key_rows,
            new_key_rows,
            cached_lower_inclusive,
            operands,
            ..
        } = &candidates[0].0.kind
        else {
            panic!("single-range candidate must carry CachedAttention operands");
        };
        assert_eq!(
            *cached_key_rows, 0,
            "a merged-KV buffer has no separate cached range"
        );
        assert!(
            *new_key_rows > 0,
            "the merged range itself must still carry the bucketed capacity"
        );
        assert_eq!(
            *cached_lower_inclusive,
            i64::MIN,
            "no dead-band sentinel is needed once the cached range is empty"
        );
        assert_eq!(
            operands.len(),
            9,
            "the live cached_len still travels as the ninth runtime operand"
        );
    }

    /// Deterministic non-degenerate weight data -- a golden-ratio fractional
    /// sequence, not a crate RNG dependency, and never all-same-value (which
    /// would hide a transposed axis or a dropped operand behind coincidental
    /// symmetry).
    #[cfg(feature = "cached-attention-streaming")]
    fn deterministic_values(count: usize, seed: f32) -> Vec<f32> {
        (0..count)
            .map(|index| {
                let phase = (index as f32 + seed) * 0.618_034;
                (phase - libm::floorf(phase)) * 2.0 - 1.0
            })
            .collect()
    }

    /// Shape constants for [`qwen35_partial_rotary_attention_fixture`],
    /// hoisted to module scope so
    /// [`qwen35_dense_attention_f64_reference`] computes over the SAME
    /// dimensions the fixture builds its graph with -- a private copy inside
    /// each function is exactly the kind of drift this row's own
    /// independent-reference discipline exists to rule out.
    #[cfg(feature = "cached-attention-streaming")]
    const QWEN35_PARTIAL_ROTARY_KV_HEADS: usize = 2;
    #[cfg(feature = "cached-attention-streaming")]
    const QWEN35_PARTIAL_ROTARY_GROUP: usize = 8;
    #[cfg(feature = "cached-attention-streaming")]
    const QWEN35_PARTIAL_ROTARY_ATTN_HEAD_DIM: usize = 256;
    #[cfg(feature = "cached-attention-streaming")]
    const QWEN35_PARTIAL_ROTARY_ROTARY_DIM: usize = 64;
    #[cfg(feature = "cached-attention-streaming")]
    const QWEN35_PARTIAL_ROTARY_PASS_DIM: usize =
        QWEN35_PARTIAL_ROTARY_ATTN_HEAD_DIM - QWEN35_PARTIAL_ROTARY_ROTARY_DIM;
    #[cfg(feature = "cached-attention-streaming")]
    const QWEN35_PARTIAL_ROTARY_PAIR_DIM: usize = QWEN35_PARTIAL_ROTARY_ROTARY_DIM / 2;
    #[cfg(feature = "cached-attention-streaming")]
    const QWEN35_PARTIAL_ROTARY_NEW_TOKENS: usize = 1;
    #[cfg(feature = "cached-attention-streaming")]
    const QWEN35_PARTIAL_ROTARY_CACHED_EXTENT: usize = 40;

    /// [`append_qwen35_dense_attention_only_with_taps`] wired at qwen3.5's
    /// own real per-head shape (`kv_heads` 2, `group` 8 -> 16 query heads,
    /// `attn_head_dim` 256, `rotary_dim` 64 -> 192-wide pass plane,
    /// `docs/discipline.md` ROW 556/557's own residual) -- `embedding` stays
    /// 1, the same degenerate-but-valid width
    /// `dense_attention_only_test_inputs` (`spec.rs`) already uses, since
    /// only the attention block's own per-head shape is under test here.
    /// `cached_extent` is 40 keys; `cached_len` (a runtime scalar, not a
    /// shape) is fed 37 at execution, leaving 3 trailing rows the padding
    /// `Select` masks with `-inf` (`spec.rs:4979-4997`).
    #[cfg(feature = "cached-attention-streaming")]
    #[allow(clippy::too_many_lines, clippy::type_complexity)]
    fn qwen35_partial_rotary_attention_fixture() -> (
        Vec<Op>,
        NodeId,
        crate::spec::Qwen35DenseAttentionTaps,
        Vec<(NodeId, Vec<f32>)>,
        Shapes,
    ) {
        use crate::op::Extent;
        use crate::spec::{causal_mask, input_leaf, scalar_constant};

        const KV_HEADS: usize = QWEN35_PARTIAL_ROTARY_KV_HEADS;
        const GROUP: usize = QWEN35_PARTIAL_ROTARY_GROUP;
        const ATTN_HEAD_DIM: usize = QWEN35_PARTIAL_ROTARY_ATTN_HEAD_DIM;
        const ROTARY_DIM: usize = QWEN35_PARTIAL_ROTARY_ROTARY_DIM;
        const PASS_DIM: usize = QWEN35_PARTIAL_ROTARY_PASS_DIM;
        const PAIR_DIM: usize = QWEN35_PARTIAL_ROTARY_PAIR_DIM;
        const NEW_TOKENS: usize = QWEN35_PARTIAL_ROTARY_NEW_TOKENS;
        const CACHED_EXTENT: usize = QWEN35_PARTIAL_ROTARY_CACHED_EXTENT;

        let mut program = Vec::new();
        let x = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(0), Extent::Static(1)],
            "x",
        );
        let inv_dim = scalar_constant(&mut program, 1.0);
        let eps = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(0)],
            "eps",
        );
        let ones = scalar_constant(&mut program, 1.0);
        let inv_sqrt_attn_head_dim = scalar_constant(&mut program, 1.0 / (ATTN_HEAD_DIM as f32).sqrt());
        let inv_attn_head_dim = scalar_constant(&mut program, 1.0 / ATTN_HEAD_DIM as f32);
        let rotary_shape = alloc::vec![Extent::Symbolic(0), Extent::Static(PAIR_DIM as u32)];
        let cos_new = input_leaf(&mut program, DType::Float32, rotary_shape.clone(), "cos");
        let sin_new = input_leaf(&mut program, DType::Float32, rotary_shape, "sin");
        let group_ones = crate::op::append(
            &mut program,
            Op::Constant {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(KV_HEADS as u32), Extent::Static(GROUP as u32)],
                value: 1.0,
            },
        );
        let (is_future, _neg_infinity) = causal_mask(&mut program).expect("causal mask lowers");
        let cached_len = input_leaf(&mut program, DType::Float32, Vec::new(), "cached_len");

        let attn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(1)],
            "attn_norm_weight",
        );
        let norm_shape = alloc::vec![Extent::Static(ATTN_HEAD_DIM as u32)];
        let q_norm_weight = input_leaf(&mut program, DType::Float32, norm_shape.clone(), "q_norm_weight");
        let k_norm_weight = input_leaf(&mut program, DType::Float32, norm_shape, "k_norm_weight");
        // `wq_gate`'s own middle axis is the FULL query head count
        // (`kv_heads * group`), never `kv_heads` alone -- `spec.rs:10730-10769`
        // packs it that way, and the group-broadcast reshape further down
        // (`group_map_i`) depends on it.
        let wq_gate = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(1),
                Extent::Static((KV_HEADS * GROUP) as u32),
                Extent::Static((2 * ATTN_HEAD_DIM) as u32)
            ],
            "wq_gate",
        );
        let wk_wv_shape = alloc::vec![
            Extent::Static(1),
            Extent::Static(KV_HEADS as u32),
            Extent::Static(ATTN_HEAD_DIM as u32)
        ];
        let wk = input_leaf(&mut program, DType::Float32, wk_wv_shape.clone(), "wk");
        let wv = input_leaf(&mut program, DType::Float32, wk_wv_shape, "wv");
        let wo = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(KV_HEADS as u32),
                Extent::Static(GROUP as u32),
                Extent::Static(ATTN_HEAD_DIM as u32),
                Extent::Static(1)
            ],
            "wo",
        );
        let cache_rotary_shape = alloc::vec![
            Extent::Symbolic(1),
            Extent::Static(KV_HEADS as u32),
            Extent::Static(PAIR_DIM as u32)
        ];
        let cache_pass_shape = alloc::vec![
            Extent::Symbolic(1),
            Extent::Static(KV_HEADS as u32),
            Extent::Static(PASS_DIM as u32)
        ];
        let cache_v_shape = alloc::vec![
            Extent::Symbolic(1),
            Extent::Static(KV_HEADS as u32),
            Extent::Static(ATTN_HEAD_DIM as u32)
        ];
        let k_first_cache = input_leaf(
            &mut program,
            DType::Float32,
            cache_rotary_shape.clone(),
            "k_first_cache",
        );
        let k_second_cache = input_leaf(&mut program, DType::Float32, cache_rotary_shape, "k_second_cache");
        let k_pass_cache = input_leaf(&mut program, DType::Float32, cache_pass_shape, "k_pass_cache");
        let v_cache = input_leaf(&mut program, DType::Float32, cache_v_shape, "v_cache");

        let (residual1, taps) = crate::spec::append_qwen35_dense_attention_only_with_taps(
            &mut program,
            x,
            inv_dim,
            eps,
            ones,
            inv_sqrt_attn_head_dim,
            inv_attn_head_dim,
            cos_new,
            sin_new,
            group_ones,
            is_future,
            cached_len,
            GROUP as u32,
            ROTARY_DIM as u32,
            ATTN_HEAD_DIM as u32,
            attn_norm_weight,
            q_norm_weight,
            k_norm_weight,
            wq_gate,
            wk,
            wv,
            wo,
            k_first_cache,
            k_second_cache,
            k_pass_cache,
            v_cache,
        )
        .expect("qwen35 partial-rotary dense attention fixture lowers");

        let shapes = crate::shape::infer(&program, &[NEW_TOKENS as u64, CACHED_EXTENT as u64])
            .expect("qwen35 partial-rotary dense attention fixture infers");

        let leaf_values = |node: NodeId, seed: f32| -> (NodeId, Vec<f32>) {
            let count = shapes.of(node).iter().product::<u64>().max(1) as usize;
            (node, deterministic_values(count, seed))
        };
        let eps_count = shapes.of(eps).iter().product::<u64>().max(1) as usize;
        let inputs = alloc::vec![
            leaf_values(x, 1.0),
            (eps, alloc::vec![1e-5f32; eps_count]),
            leaf_values(cos_new, 2.0),
            leaf_values(sin_new, 3.0),
            leaf_values(attn_norm_weight, 4.0),
            leaf_values(q_norm_weight, 5.0),
            leaf_values(k_norm_weight, 6.0),
            leaf_values(wq_gate, 7.0),
            leaf_values(wk, 8.0),
            leaf_values(wv, 9.0),
            leaf_values(wo, 10.0),
            leaf_values(k_first_cache, 11.0),
            leaf_values(k_second_cache, 12.0),
            leaf_values(k_pass_cache, 13.0),
            leaf_values(v_cache, 14.0),
            (cached_len, alloc::vec![37.0]),
        ];
        (program, residual1, taps, inputs, shapes)
    }

    /// The matcher's own recognition gate: qwen35's real partial-rotary
    /// chain (`rotary_dim` 64 of `head_dim` 256, `kv_heads` 2, `group` 8)
    /// fuses into exactly one [`BoundOpKind::CachedAttention`] carrying the
    /// full eight-base + `cached_len` + three-pass-plane operand set, and
    /// the fusion strictly reduces the bound-op count relative to the
    /// unfused elementwise/reduce chain [`bind_plain`] already produces
    /// (the census, ROW 558 `docs/discipline.md`).
    ///
    /// Also asserts numeric parity against the unfused chain (item (a) of
    /// the ROW 558 brief, deliberately deferred through ROW 558-560's own
    /// residuals while the ~8% divergence this fixture surfaced was
    /// mislocated in two false leads before landing on the real defect --
    /// `neon_tile_plan`'s missing row-invariance check on its `b` operand,
    /// `docs/discipline.md` ROW 561, fixed at `cpu.rs`'s own
    /// `neon_tile_plan`). [`qwen35_partial_rotary_dense_attention_matches_an_independent_f64_reference`]
    /// is the test that actually PROVES which side was wrong, tap by tap,
    /// against an f64 reference outside both engines; this test only checks
    /// that the two engines still AGREE once that defect is fixed.
    #[test]
    #[cfg(feature = "cached-attention-streaming")]
    fn qwen35_partial_rotary_cached_attention_fuses_and_matches_the_unfused_layer() {
        let (program, residual1, taps, inputs, shapes) = qwen35_partial_rotary_attention_fixture();
        // `attended` (the fusion's own anchor node, `attention_score_sources`'s
        // own doc) has exactly one reader (the per-head gate multiply) --
        // qwen35's own extra gate stage, absent from mistral/openchat, gives
        // `ChainFusion` one more link to fold it into, so it must be pinned
        // as its own materialization boundary the same way the cache roots
        // already are, or `bind_plain` never gives it a standalone `BoundOp`
        // for `cached_attention_candidates` to find.
        let outputs = alloc::vec![
            residual1,
            taps.attended,
            taps.rotated_k_new_first,
            taps.rotated_k_new_second,
            taps.k_pass,
            taps.v_new,
        ];

        let unfused = bind_plain(&program, &shapes, &outputs, NumericPolicy::bit_exact())
            .expect("plain bind succeeds");
        let fused = bind(&program, &shapes, &outputs, NumericPolicy::bit_exact())
            .expect("fused bind succeeds");

        let fused_attention_count = fused
            .iter()
            .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
            .count();
        assert_eq!(
            fused_attention_count, 1,
            "the qwen35 partial-rotary chain must fuse into exactly one \
             cached-attention op"
        );
        let BoundOpKind::CachedAttention {
            rotary_dim,
            head_dim,
            operands,
            ..
        } = fused
            .iter()
            .find(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
            .map(|bound| &bound.kind)
            .expect("a cached-attention op was just counted above")
        else {
            unreachable!("just matched CachedAttention above");
        };
        assert_eq!(*rotary_dim, 64, "rotary width is qwen35's own 64, not the full head_dim");
        assert_eq!(*head_dim, 256, "head_dim carries the full width, rotary plus pass");
        assert_eq!(
            operands.len(),
            12,
            "eight base sources, the runtime cached_len, and the three pass-plane sources"
        );
        // the census: how many bound ops this one fusion absorbed (66 unfused
        // vs 40 fused on this fixture -- 26 ops absorbed by the one fusion,
        // `docs/discipline.md` ROW 558).
        assert!(
            unfused.len() > fused.len(),
            "the fusion must absorb at least one op relative to the unfused chain"
        );

        let unfused_outputs = run_resolved(program.len(), &unfused, inputs.clone());
        let fused_outputs = run_resolved(program.len(), &fused, inputs);
        let expected = unfused_outputs[residual1.0 as usize]
            .as_ref()
            .expect("unfused layer output computes");
        let actual = fused_outputs[residual1.0 as usize]
            .as_ref()
            .expect("fused layer output computes");
        assert_eq!(
            expected.len(),
            actual.len(),
            "fused and unfused outputs must be the same shape"
        );
        for (index, (&expected_value, &actual_value)) in
            expected.iter().zip(actual.iter()).enumerate()
        {
            let (expected_value, actual_value) =
                (f64::from(expected_value), f64::from(actual_value));
            let relative_error =
                (expected_value - actual_value).abs() / expected_value.abs().max(1.0);
            assert!(
                relative_error <= 1e-5,
                "residual1[{index}] fused vs unfused: expected={expected_value} actual={actual_value} \
                 relative_error={relative_error} (ROW 561's own fix)"
            );
        }
    }

    /// The real forward-pass builder (`program.rs`'s own qwen35moe caller)
    /// never requests `taps.attended` as an output -- only `residual1` and
    /// the four cache-write roots survive into the next layer/decode step --
    /// so this test drops `taps.attended` from `outputs` relative to
    /// [`qwen35_partial_rotary_cached_attention_fuses_and_matches_the_unfused_layer`]'s
    /// own list, reproducing the exact shape ROW 563's real-checkpoint
    /// telemetry captured (`node=1288 elementwise_operand_fuse decision=fused
    /// into=1294`, immediately followed by every full-attention layer's
    /// `cached_attention decline ... stage=output_not_resolved`). Before
    /// `bind_cached_attention_fusion`'s own `planning_outputs` loop pinned a
    /// discovered candidate's anchor node (`fused.node`) alongside its
    /// source operands, this exact `outputs` list produced
    /// `fused_attention_count == 0` on this fixture.
    #[test]
    #[cfg(feature = "cached-attention-streaming")]
    fn qwen35_partial_rotary_cached_attention_fuses_without_pinning_the_attended_tap() {
        let (program, residual1, taps, inputs, shapes) = qwen35_partial_rotary_attention_fixture();
        let outputs = alloc::vec![
            residual1,
            taps.rotated_k_new_first,
            taps.rotated_k_new_second,
            taps.k_pass,
            taps.v_new,
        ];

        let fused = bind(&program, &shapes, &outputs, NumericPolicy::bit_exact())
            .expect("fused bind succeeds");
        let fused_attention_count = fused
            .iter()
            .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
            .count();
        assert_eq!(
            fused_attention_count, 1,
            "the qwen35 partial-rotary chain must fuse into exactly one \
             cached-attention op even when only the real forward pass's own \
             outputs are requested"
        );

        let unfused = bind_plain(&program, &shapes, &outputs, NumericPolicy::bit_exact())
            .expect("plain bind succeeds");
        let unfused_outputs = run_resolved(program.len(), &unfused, inputs.clone());
        let fused_outputs = run_resolved(program.len(), &fused, inputs);
        let expected = unfused_outputs[residual1.0 as usize]
            .as_ref()
            .expect("unfused layer output computes");
        let actual = fused_outputs[residual1.0 as usize]
            .as_ref()
            .expect("fused layer output computes");
        for (index, (&expected_value, &actual_value)) in
            expected.iter().zip(actual.iter()).enumerate()
        {
            let (expected_value, actual_value) =
                (f64::from(expected_value), f64::from(actual_value));
            let relative_error =
                (expected_value - actual_value).abs() / expected_value.abs().max(1.0);
            assert!(
                relative_error <= 1e-5,
                "residual1[{index}] fused vs unfused: expected={expected_value} actual={actual_value} \
                 relative_error={relative_error}"
            );
        }
    }

    /// Every tap [`qwen35_dense_attention_f64_reference`] computes, in the
    /// order [`crate::spec::append_qwen35_dense_attention_only_with_taps`]
    /// builds them -- the order this row's own per-tap divergence search
    /// walks.
    #[cfg(feature = "cached-attention-streaming")]
    struct Qwen35DenseAttentionF64Reference {
        normed: Vec<f64>,
        q_split: Vec<f64>,
        gate_split: Vec<f64>,
        v_new: Vec<f64>,
        q_normed: Vec<f64>,
        k_normed: Vec<f64>,
        k_pass: Vec<f64>,
        q_rot_first: Vec<f64>,
        q_rot_second: Vec<f64>,
        k_rot_first: Vec<f64>,
        k_rot_second: Vec<f64>,
        score_new: Vec<f64>,
        attended: Vec<f64>,
        gate_sigmoid: Vec<f64>,
        gated_attended: Vec<f64>,
        o_proj_out: Vec<f64>,
        residual1: Vec<f64>,
    }

    /// Independent f64 reference for
    /// [`qwen35_partial_rotary_attention_fixture`]'s whole dense-attention
    /// layer, built by plain nested loops over the SAME per-input byte
    /// vectors the fixture feeds `run_resolved` (`inputs`, read positionally
    /// in the fixture's own construction order -- see the `assert_eq!` on
    /// `inputs.len()` below, which fails loudly if that order ever changes).
    /// No `Op`/`bind`/`Reduce` machinery anywhere in this function: this is
    /// the third, previously-missing leg `docs/discipline.md` ROW 560's own
    /// residual asked for. ROW 559 showed the fused `CachedAttention` kernel
    /// is exact against its own operands; ROW 560 showed `bind_plain`'s
    /// generic reduce is ALSO exact against ITS own operands; neither row
    /// had ever checked either chain against a party that owes nothing to
    /// either implementation's own bugs -- an oracle from the model
    /// semantics (`modeling_qwen3_next.py`), not from either engine.
    ///
    /// Math, per stage (mirrors `spec.rs:4724-5312`
    /// (`append_qwen35_dense_attention_only_with_taps`) exactly, at f64
    /// precision, for this fixture's own degenerate `embedding = 1`,
    /// `new_tokens = 1` shape): RMSNorm(`x`) -> the one `q`/`gate`
    /// projection split per head -> `k`/`v` projections -> per-head RMSNorm
    /// of `q`/`k` over the full `attn_head_dim` -> the pass-plane slice
    /// (`[rotary_dim, attn_head_dim)`) -> interleaved RoPE
    /// (`(2*i, 2*i+1)` pairing, `RopePairing::Interleaved`'s own doc) over
    /// the first `rotary_dim` channels -> grouped rotary + pass dot products
    /// against the 40-row KV cache (masked `-inf` past `cached_len`) and the
    /// one new key (never masked at position 0) -> one softmax over the
    /// concatenation of both -> the value-weighted sum -> the per-head
    /// sigmoid gate -> `o_proj` reduced to the single embedding output ->
    /// the residual add.
    #[cfg(feature = "cached-attention-streaming")]
    #[allow(clippy::too_many_lines)]
    fn qwen35_dense_attention_f64_reference(
        inputs: &[(NodeId, Vec<f32>)],
    ) -> Qwen35DenseAttentionF64Reference {
        const KV_HEADS: usize = QWEN35_PARTIAL_ROTARY_KV_HEADS;
        const GROUP: usize = QWEN35_PARTIAL_ROTARY_GROUP;
        const ATTN_HEAD_DIM: usize = QWEN35_PARTIAL_ROTARY_ATTN_HEAD_DIM;
        const ROTARY_DIM: usize = QWEN35_PARTIAL_ROTARY_ROTARY_DIM;
        const PASS_DIM: usize = QWEN35_PARTIAL_ROTARY_PASS_DIM;
        const PAIR_DIM: usize = QWEN35_PARTIAL_ROTARY_PAIR_DIM;
        const CACHED_EXTENT: usize = QWEN35_PARTIAL_ROTARY_CACHED_EXTENT;
        const HEADS: usize = KV_HEADS * GROUP;

        assert_eq!(
            inputs.len(),
            16,
            "qwen35_partial_rotary_attention_fixture's own input list grew or shrank -- \
             this reference's positional indexing below must be re-derived, not silently \
             misaligned"
        );
        let as_f64 =
            |values: &[f32]| -> Vec<f64> { values.iter().map(|&value| f64::from(value)).collect() };
        let x = as_f64(&inputs[0].1);
        let eps = f64::from(inputs[1].1[0]);
        let cos_new = as_f64(&inputs[2].1);
        let sin_new = as_f64(&inputs[3].1);
        let attn_norm_weight = f64::from(inputs[4].1[0]);
        let q_norm_weight = as_f64(&inputs[5].1);
        let k_norm_weight = as_f64(&inputs[6].1);
        let wq_gate = as_f64(&inputs[7].1);
        let wk = as_f64(&inputs[8].1);
        let wv = as_f64(&inputs[9].1);
        let wo = as_f64(&inputs[10].1);
        let k_first_cache = as_f64(&inputs[11].1);
        let k_second_cache = as_f64(&inputs[12].1);
        let k_pass_cache = as_f64(&inputs[13].1);
        let v_cache = as_f64(&inputs[14].1);
        let cached_len = f64::from(inputs[15].1[0]);

        // RMSNorm(x): embedding width 1, so the sum-of-squares is one term
        // and inv_dim (a scalar_constant(1.0)) contributes nothing.
        let inv_rms_x = 1.0 / (x[0] * x[0] + eps).sqrt();
        let normed = x[0] * inv_rms_x * attn_norm_weight;

        // qg_raw[h][c] = normed * wq_gate[0][h][c] (embedding contraction is
        // one term); q_split/gate_split are the per-head [0,256)/[256,512)
        // halves of that same activation.
        let mut q_split = vec![0.0f64; HEADS * ATTN_HEAD_DIM];
        let mut gate_split = vec![0.0f64; HEADS * ATTN_HEAD_DIM];
        for head in 0..HEADS {
            for channel in 0..ATTN_HEAD_DIM {
                let q_weight = wq_gate[head * (2 * ATTN_HEAD_DIM) + channel];
                let gate_weight = wq_gate[head * (2 * ATTN_HEAD_DIM) + ATTN_HEAD_DIM + channel];
                q_split[head * ATTN_HEAD_DIM + channel] = normed * q_weight;
                gate_split[head * ATTN_HEAD_DIM + channel] = normed * gate_weight;
            }
        }

        let mut k_raw = vec![0.0f64; KV_HEADS * ATTN_HEAD_DIM];
        let mut v_new = vec![0.0f64; KV_HEADS * ATTN_HEAD_DIM];
        for kv_head in 0..KV_HEADS {
            for channel in 0..ATTN_HEAD_DIM {
                k_raw[kv_head * ATTN_HEAD_DIM + channel] =
                    normed * wk[kv_head * ATTN_HEAD_DIM + channel];
                v_new[kv_head * ATTN_HEAD_DIM + channel] =
                    normed * wv[kv_head * ATTN_HEAD_DIM + channel];
            }
        }

        let inv_head_dim = 1.0 / ATTN_HEAD_DIM as f64;
        let mut q_normed = vec![0.0f64; HEADS * ATTN_HEAD_DIM];
        for head in 0..HEADS {
            let sum_squares: f64 = (0..ATTN_HEAD_DIM)
                .map(|channel| q_split[head * ATTN_HEAD_DIM + channel].powi(2))
                .sum();
            let inv_rms = 1.0 / (sum_squares * inv_head_dim + eps).sqrt();
            for channel in 0..ATTN_HEAD_DIM {
                q_normed[head * ATTN_HEAD_DIM + channel] =
                    q_split[head * ATTN_HEAD_DIM + channel] * inv_rms * q_norm_weight[channel];
            }
        }
        let mut k_normed = vec![0.0f64; KV_HEADS * ATTN_HEAD_DIM];
        for kv_head in 0..KV_HEADS {
            let sum_squares: f64 = (0..ATTN_HEAD_DIM)
                .map(|channel| k_raw[kv_head * ATTN_HEAD_DIM + channel].powi(2))
                .sum();
            let inv_rms = 1.0 / (sum_squares * inv_head_dim + eps).sqrt();
            for channel in 0..ATTN_HEAD_DIM {
                k_normed[kv_head * ATTN_HEAD_DIM + channel] =
                    k_raw[kv_head * ATTN_HEAD_DIM + channel] * inv_rms * k_norm_weight[channel];
            }
        }

        let mut k_pass = vec![0.0f64; KV_HEADS * PASS_DIM];
        for kv_head in 0..KV_HEADS {
            for pass_channel in 0..PASS_DIM {
                k_pass[kv_head * PASS_DIM + pass_channel] =
                    k_normed[kv_head * ATTN_HEAD_DIM + ROTARY_DIM + pass_channel];
            }
        }
        let mut q_pass = vec![0.0f64; HEADS * PASS_DIM];
        for head in 0..HEADS {
            for pass_channel in 0..PASS_DIM {
                q_pass[head * PASS_DIM + pass_channel] =
                    q_normed[head * ATTN_HEAD_DIM + ROTARY_DIM + pass_channel];
            }
        }

        // Interleaved RoPE, `RopePairing::Interleaved`'s own `(2*i, 2*i+1)`
        // pairing, over the first `rotary_dim` channels only. `cos_new`/
        // `sin_new` are `[s, pair]`-shaped with `s == 1` in this fixture, so
        // a bare `pair` index already reads position 0's own row.
        let rotate = |source: &[f64], head_count: usize| -> (Vec<f64>, Vec<f64>) {
            let mut first = vec![0.0f64; head_count * PAIR_DIM];
            let mut second = vec![0.0f64; head_count * PAIR_DIM];
            for head in 0..head_count {
                for pair in 0..PAIR_DIM {
                    let even = source[head * ATTN_HEAD_DIM + 2 * pair];
                    let odd = source[head * ATTN_HEAD_DIM + 2 * pair + 1];
                    first[head * PAIR_DIM + pair] = even * cos_new[pair] - odd * sin_new[pair];
                    second[head * PAIR_DIM + pair] = odd * cos_new[pair] + even * sin_new[pair];
                }
            }
            (first, second)
        };
        let (q_rot_first, q_rot_second) = rotate(&q_normed, HEADS);
        let (k_rot_first, k_rot_second) = rotate(&k_normed, KV_HEADS);

        // query head h = GROUP*kv_head + group, `group_map_i`/`group_map_p`/
        // `group_map_d`'s own convention (`spec.rs:4847,4866,5221`).
        let query_head = |kv_head: usize, group: usize| GROUP * kv_head + group;

        let inv_sqrt_head_dim = 1.0 / (ATTN_HEAD_DIM as f64).sqrt();
        let mut score_cached = vec![0.0f64; CACHED_EXTENT * KV_HEADS * GROUP];
        for cached_row in 0..CACHED_EXTENT {
            for kv_head in 0..KV_HEADS {
                for group in 0..GROUP {
                    let head = query_head(kv_head, group);
                    let mut score = 0.0f64;
                    for pair in 0..PAIR_DIM {
                        let cache_index = (cached_row * KV_HEADS + kv_head) * PAIR_DIM + pair;
                        score += q_rot_first[head * PAIR_DIM + pair] * k_first_cache[cache_index];
                        score += q_rot_second[head * PAIR_DIM + pair] * k_second_cache[cache_index];
                    }
                    for pass_channel in 0..PASS_DIM {
                        let cache_index =
                            (cached_row * KV_HEADS + kv_head) * PASS_DIM + pass_channel;
                        score += q_pass[head * PASS_DIM + pass_channel] * k_pass_cache[cache_index];
                    }
                    score *= inv_sqrt_head_dim;
                    let is_padding = cached_row as f64 > cached_len - 1.0;
                    let index = (cached_row * KV_HEADS + kv_head) * GROUP + group;
                    score_cached[index] = if is_padding { f64::NEG_INFINITY } else { score };
                }
            }
        }

        // The one new (uncached) key: position 0 is never future-masked
        // against itself (`is_future[0][0] == (0 > 0) == false`).
        let mut score_new = vec![0.0f64; KV_HEADS * GROUP];
        for kv_head in 0..KV_HEADS {
            for group in 0..GROUP {
                let head = query_head(kv_head, group);
                let mut score = 0.0f64;
                for pair in 0..PAIR_DIM {
                    score += q_rot_first[head * PAIR_DIM + pair]
                        * k_rot_first[kv_head * PAIR_DIM + pair];
                    score += q_rot_second[head * PAIR_DIM + pair]
                        * k_rot_second[kv_head * PAIR_DIM + pair];
                }
                for pass_channel in 0..PASS_DIM {
                    score += q_pass[head * PASS_DIM + pass_channel]
                        * k_pass[kv_head * PASS_DIM + pass_channel];
                }
                score_new[kv_head * GROUP + group] = score * inv_sqrt_head_dim;
            }
        }

        let mut attended = vec![0.0f64; KV_HEADS * GROUP * ATTN_HEAD_DIM];
        let mut gate_sigmoid = vec![0.0f64; KV_HEADS * GROUP * ATTN_HEAD_DIM];
        let mut gated_attended = vec![0.0f64; KV_HEADS * GROUP * ATTN_HEAD_DIM];
        for kv_head in 0..KV_HEADS {
            for group in 0..GROUP {
                let cached_scores: Vec<f64> = (0..CACHED_EXTENT)
                    .map(|cached_row| {
                        score_cached[(cached_row * KV_HEADS + kv_head) * GROUP + group]
                    })
                    .collect();
                let new_score = score_new[kv_head * GROUP + group];
                let global_max = cached_scores.iter().copied().fold(new_score, f64::max);
                let cached_weights: Vec<f64> = cached_scores
                    .iter()
                    .map(|&score| (score - global_max).exp())
                    .collect();
                let new_weight = (new_score - global_max).exp();
                let inv_weight_sum = 1.0 / (cached_weights.iter().sum::<f64>() + new_weight);

                let head = query_head(kv_head, group);
                for channel in 0..ATTN_HEAD_DIM {
                    let cached_sum: f64 = (0..CACHED_EXTENT)
                        .map(|cached_row| {
                            cached_weights[cached_row]
                                * v_cache
                                    [(cached_row * KV_HEADS + kv_head) * ATTN_HEAD_DIM + channel]
                        })
                        .sum();
                    let new_sum = new_weight * v_new[kv_head * ATTN_HEAD_DIM + channel];
                    let attended_value = (cached_sum + new_sum) * inv_weight_sum;
                    let sigmoid = 1.0 / (1.0 + (-gate_split[head * ATTN_HEAD_DIM + channel]).exp());
                    let index = (kv_head * GROUP + group) * ATTN_HEAD_DIM + channel;
                    attended[index] = attended_value;
                    gate_sigmoid[index] = sigmoid;
                    gated_attended[index] = attended_value * sigmoid;
                }
            }
        }

        let mut o_proj_out = 0.0f64;
        for kv_head in 0..KV_HEADS {
            for group in 0..GROUP {
                for channel in 0..ATTN_HEAD_DIM {
                    let index = (kv_head * GROUP + group) * ATTN_HEAD_DIM + channel;
                    o_proj_out += gated_attended[index] * wo[index];
                }
            }
        }
        let residual1 = o_proj_out + x[0];

        Qwen35DenseAttentionF64Reference {
            normed: alloc::vec![normed],
            q_split,
            gate_split,
            v_new,
            q_normed,
            k_normed,
            k_pass,
            q_rot_first,
            q_rot_second,
            k_rot_first,
            k_rot_second,
            score_new,
            attended,
            gate_sigmoid,
            gated_attended,
            o_proj_out: alloc::vec![o_proj_out],
            residual1: alloc::vec![residual1],
        }
    }

    /// Max relative error of `actual` against `reference`, element-wise --
    /// this row's own single comparison primitive, used identically for
    /// every tap so the per-tap table below is apples to apples. Denominator
    /// floors at `1.0` so a near-zero reference element does not blow up a
    /// float-rounding-sized absolute difference into a nonsense percentage.
    #[cfg(feature = "cached-attention-streaming")]
    fn max_relative_error(reference: &[f64], actual: &[f32]) -> f64 {
        assert_eq!(
            reference.len(),
            actual.len(),
            "reference and actual must share a shape"
        );
        reference
            .iter()
            .zip(actual.iter())
            .map(|(&expected, &got)| {
                let got = f64::from(got);
                (got - expected).abs() / expected.abs().max(1.0)
            })
            .fold(0.0f64, f64::max)
    }

    /// The independent-reference leg ROW 559/560's own residual named as
    /// missing (`docs/discipline.md`): both `bind_plain` and the fused
    /// `CachedAttention` op were already proven self-consistent against
    /// their OWN resolved operands, but never checked against a party that
    /// owes nothing to either implementation. This test runs both chains
    /// through [`run_resolved`] on the SAME fixture inputs
    /// [`qwen35_dense_attention_f64_reference`] also consumes, then asserts
    /// every named tap from BOTH chains against that f64 reference -- the
    /// first tap (in builder order) that fails on either side names the
    /// defective chain and the value pair proving it.
    #[test]
    #[cfg(feature = "cached-attention-streaming")]
    fn qwen35_partial_rotary_dense_attention_matches_an_independent_f64_reference() {
        let (program, residual1, taps, inputs, shapes) = qwen35_partial_rotary_attention_fixture();
        let reference = qwen35_dense_attention_f64_reference(&inputs);

        let outputs = alloc::vec![
            residual1,
            taps.normed,
            taps.q_split,
            taps.gate_split,
            taps.v_new,
            taps.q_normed,
            taps.k_normed,
            taps.k_pass,
            taps.q_rot_first,
            taps.q_rot_second,
            taps.k_rot_first,
            taps.k_rot_second,
            taps.score_new,
            taps.attended,
            taps.gate_sigmoid,
            taps.gated_attended,
            taps.o_proj_out,
        ];

        let unfused = bind_plain(&program, &shapes, &outputs, NumericPolicy::bit_exact())
            .expect("plain bind succeeds");
        let fused = bind(&program, &shapes, &outputs, NumericPolicy::bit_exact())
            .expect("fused bind succeeds");
        let unfused_outputs = run_resolved(program.len(), &unfused, inputs.clone());
        let fused_outputs = run_resolved(program.len(), &fused, inputs);

        let taps: [(&str, NodeId, &[f64]); 17] = [
            ("normed", taps.normed, &reference.normed),
            ("q_split", taps.q_split, &reference.q_split),
            ("gate_split", taps.gate_split, &reference.gate_split),
            ("v_new", taps.v_new, &reference.v_new),
            ("q_normed", taps.q_normed, &reference.q_normed),
            ("k_normed", taps.k_normed, &reference.k_normed),
            ("k_pass", taps.k_pass, &reference.k_pass),
            ("q_rot_first", taps.q_rot_first, &reference.q_rot_first),
            ("q_rot_second", taps.q_rot_second, &reference.q_rot_second),
            ("k_rot_first", taps.k_rot_first, &reference.k_rot_first),
            ("k_rot_second", taps.k_rot_second, &reference.k_rot_second),
            ("score_new", taps.score_new, &reference.score_new),
            ("attended", taps.attended, &reference.attended),
            ("gate_sigmoid", taps.gate_sigmoid, &reference.gate_sigmoid),
            (
                "gated_attended",
                taps.gated_attended,
                &reference.gated_attended,
            ),
            ("o_proj_out", taps.o_proj_out, &reference.o_proj_out),
            ("residual1", residual1, &reference.residual1),
        ];

        const TOLERANCE: f64 = 1e-4;
        let mut first_divergent: Option<(&str, &str, f64)> = None;
        for (name, node, expected) in taps {
            let unfused_actual = unfused_outputs[node.0 as usize]
                .as_ref()
                .expect("unfused tap computes");
            let fused_actual = fused_outputs[node.0 as usize]
                .as_ref()
                .expect("fused tap computes");
            let unfused_error = max_relative_error(expected, unfused_actual);
            let fused_error = max_relative_error(expected, fused_actual);
            #[cfg(feature = "instrument")]
            debug!(
                tap = name,
                unfused_error = unfused_error,
                fused_error = fused_error,
                "qwen35 dense attention tap vs f64 reference"
            );
            if first_divergent.is_none() && unfused_error > TOLERANCE {
                first_divergent = Some((name, "unfused (bind_plain)", unfused_error));
            }
            if first_divergent.is_none() && fused_error > TOLERANCE {
                first_divergent = Some((name, "fused (CachedAttention)", fused_error));
            }
        }

        assert!(
            first_divergent.is_none(),
            "first tap to diverge from the independent f64 reference beyond {TOLERANCE}: {first_divergent:?}"
        );
    }

    /// Negative: perturbing the pass plane's own index map so it no longer
    /// reads the SAME `q_pass_grouped` node on both the cached and new score
    /// sides must make the matcher decline the whole fusion, not silently
    /// drop the pass plane and fuse a rotary-only op that would then compute
    /// the wrong score.
    #[test]
    #[cfg(feature = "cached-attention-streaming")]
    fn a_perturbed_pass_plane_map_declines_the_qwen35_fusion() {
        let (mut program, residual1, _taps, _inputs, shapes) =
            qwen35_partial_rotary_attention_fixture();
        let outputs = alloc::vec![residual1];

        // The pass plane's own product is the one `Multiply` whose output's
        // last axis is exactly `PASS_DIM` (192) -- distinct from every
        // rotary product (last axis `PAIR_DIM`, 32). Flipping its body to
        // `Add` breaks [`decode_pass_term`]'s own "reduced(query * key)"
        // shape deterministically, without depending on the exact index-map
        // encoding this fixture happens to produce.
        let pass_product = (0..program.len())
            .rev()
            .find(|&position| {
                matches!(
                    &program[position],
                    Op::Elementwise { body: ScalarOp::Multiply, .. }
                ) && shapes.of(NodeId(position as u32)).last() == Some(&192)
            })
            .expect("the fixture must contain a pass-plane product to perturb");
        let Op::Elementwise { body, .. } = &mut program[pass_product] else {
            unreachable!("just matched Op::Elementwise above");
        };
        *body = ScalarOp::Add;

        let resolved = bind_plain(&program, &shapes, &outputs, NumericPolicy::bit_exact())
            .expect("plain bind still succeeds on the perturbed program");
        let candidates = cached_attention_candidates(&program, &shapes, &resolved, &outputs, true);
        assert!(
            candidates.is_empty(),
            "a perturbed pass-plane product must not still match the qwen35 fusion"
        );
    }

    /// A gathered (or negative-strided) source must abort the candidate
    /// outright rather than merely skip its own push -- the `continue`
    /// this test guards against left `operands` one entry short of
    /// `source_nodes`, relying on the length check further down to reject
    /// the misaligned vector rather than aborting where the defect is
    /// found. Reproduces the real single-range fixture, then patches one
    /// of `cached_attention_single_range_candidates`' own eight source
    /// nodes -- read straight off the baseline candidate's fused operand
    /// list, never guessed -- to carry a `Lookup` in `resolved`, the exact
    /// shape a dynamic KV-cache-page gather would leave behind.
    #[test]
    #[cfg(feature = "cached-attention-streaming")]
    fn a_gathered_source_aborts_the_single_range_candidate_entirely() {
        let (program, logits, cache_roots, _) =
            crate::spec::mistral_single_range_cached_forward_program(
                32,
                16,
                24,
                4,
                2,
                4,
                1,
                false,
                crate::spec::DuplicateHeadPosition::None,
                false,
            )
            .expect("single-range fixture builds");
        let mut outputs = alloc::vec![logits];
        for (even, odd, value) in &cache_roots {
            outputs.extend_from_slice(&[*even, *odd, *value]);
        }
        let shapes = crate::shape::infer(&program, &[1, 5]).expect("single-range fixture infers");
        let mut resolved = bind_plain(&program, &shapes, &outputs, NumericPolicy::bit_exact())
            .expect("plain bind succeeds");

        let baseline =
            cached_attention_single_range_candidates(&program, &shapes, &resolved, &outputs);
        assert!(
            !baseline.is_empty(),
            "the unpatched fixture must still produce a fusable candidate"
        );
        let BoundOpKind::CachedAttention { operands, .. } = &baseline[0].0.kind else {
            panic!("single-range candidate must carry CachedAttention operands");
        };
        let gathered_source = operands[0].0;

        for bound in &mut resolved {
            let operands = match &mut bound.kind {
                BoundOpKind::CachedAttention { operands, .. }
                | BoundOpKind::Elementwise { operands, .. }
                | BoundOpKind::Reduce { operands, .. } => operands,
                BoundOpKind::Iota
                | BoundOpKind::Constant { .. }
                | BoundOpKind::GatedDeltaNet { .. } => continue,
            };
            for (node, layout, lookup) in operands.iter_mut() {
                if *node == gathered_source {
                    *lookup = Some(Lookup {
                        indices: gathered_source,
                        index_layout: layout.clone(),
                        element_stride: 1,
                        extent: 1,
                    });
                }
            }
        }

        let patched =
            cached_attention_single_range_candidates(&program, &shapes, &resolved, &outputs);
        assert!(
            patched.is_empty(),
            "a gathered source must abort the candidate, not just shrink its operand list"
        );
    }

    use crate::dtype::DType;
    use crate::map;
    use crate::op::{Extent, append};

    fn matmul_program() -> (Vec<Op>, NodeId, NodeId, NodeId) {
        let mut program = Vec::new();
        let lhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Symbolic(0), Extent::Static(768)],
                name: None,
            },
        );
        let rhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(768), Extent::Static(3072)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![
                    (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                    (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
                ],
                name: None,
            },
        );
        let sum = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: crate::op::ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: Some("matmul".into()),
            }),
        );
        (program, product, sum, lhs)
    }

    /// An `Iota` binds directly to its own ready `BoundOp`, the same way a
    /// `Reduce` always does — never held pending fusion the way an
    /// `Elementwise` op is, since it has no operand to fuse with anything.
    #[test]
    fn an_iota_binds_to_its_own_ready_bound_op_with_no_operands() {
        let mut program = Vec::new();
        let iota = append(
            &mut program,
            Op::Iota {
                dtype: DType::Float32,
                extent: Extent::Static(8),
            },
        );

        let shapes = shape::infer(&program, &[]).expect("iota infers");
        let built =
            bind(&program, &shapes, &[terminal(&program)], NumericPolicy::bit_exact()).expect("iota builds ops");

        assert_eq!(built.len(), 1, "the iota leaf materializes on its own");
        assert_eq!(built[0].node, iota);
        assert_eq!(built[0].dtype, DType::Float32);
        assert_eq!(built[0].extents, alloc::vec![8]);
        assert!(matches!(built[0].kind, BoundOpKind::Iota));
        assert!(
            built[0].operands().is_empty(),
            "a leaf with no operands binds to none"
        );
    }

    #[test]
    fn matmul_resolves_to_one_fused_op_not_two() {
        let (program, product, sum, _lhs) = matmul_program();
        let shapes = shape::infer(&program, &[512]).expect("matmul infers");
        let built =
            bind(&program, &shapes, &[terminal(&program)], NumericPolicy::bit_exact()).expect("matmul builds ops");

        assert_eq!(
            built.len(),
            1,
            "the elementwise op must not materialize separately"
        );
        assert_eq!(built[0].node, sum);
        assert!(matches!(built[0].kind, BoundOpKind::Reduce { .. }));
        assert_eq!(
            built[0].element_body().steps.len(),
            1,
            "one absorbed elementwise op is one composed step"
        );
        assert_ne!(
            built[0].element_body().steps[0].op,
            ScalarOp::Identity,
            "the fused body is the elementwise op's multiply"
        );
        let _ = product;
    }

    #[test]
    fn requesting_the_intermediate_elementwise_op_as_an_output_prevents_fusion() {
        let (program, product, sum, _lhs) = matmul_program();
        let shapes = shape::infer(&program, &[512]).expect("matmul infers");
        let built = bind(
            &program,
            &shapes,
            &[product, sum],
            NumericPolicy::bit_exact(),
        )
        .expect("matmul builds ops with two outputs");

        assert_eq!(
            built.len(),
            2,
            "the requested-output elementwise op must materialize"
        );
        assert!(
            built
                .iter()
                .any(|op| op.node == product && matches!(op.kind, BoundOpKind::Elementwise { .. }))
        );
        assert!(
            built
                .iter()
                .any(|op| op.node == sum && matches!(op.kind, BoundOpKind::Reduce { .. }))
        );
    }

    /// `b = a * scale; c = b + bias; d = c * c` — three chained elementwise
    /// ops, each the sole and last use of the one before it, none of them
    /// requested as an output. All three must fuse into `d`'s own `BoundOp`
    /// rather than materializing `b` and `c` along the way.
    fn elementwise_chain_program() -> (Vec<Op>, NodeId, NodeId, NodeId) {
        let mut program = Vec::new();
        let a = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let scale = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let bias = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let identity = || IndexMap::Affine(map::projection(1, &[0]));
        let b = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![(a, identity()), (scale, identity())],
                name: None,
            },
        );
        let c = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![(b, identity()), (bias, identity())],
                name: None,
            },
        );
        let d = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![(c, identity()), (c, identity())],
                name: None,
            },
        );
        (program, b, c, d)
    }

    #[test]
    fn a_chain_of_elementwise_ops_fuses_into_one_bound_op_not_three() {
        let (program, _b, _c, d) = elementwise_chain_program();
        let shapes = shape::infer(&program, &[]).expect("elementwise chain infers");
        let built = bind(&program, &shapes, &[terminal(&program)], NumericPolicy::bit_exact())
            .expect("elementwise chain builds ops");

        assert_eq!(
            built.len(),
            1,
            "b and c must absorb into d's own BoundOp instead of materializing"
        );
        assert_eq!(built[0].node, d);
        assert!(matches!(built[0].kind, BoundOpKind::Elementwise { .. }));
        assert!(
            built[0].element_body().steps.len() >= 2,
            "the composed body must carry more than one absorbed op's step"
        );
    }

    #[test]
    fn an_elementwise_intermediate_requested_as_an_output_prevents_fusion() {
        let (program, b, _c, d) = elementwise_chain_program();
        let shapes = shape::infer(&program, &[]).expect("elementwise chain infers");
        let built = bind(&program, &shapes, &[b, d], NumericPolicy::bit_exact())
            .expect("elementwise chain builds ops with 2 outputs");

        assert_eq!(
            built.len(),
            2,
            "requesting b as an output must force it to materialize on its own"
        );
        assert!(
            built
                .iter()
                .any(|op| op.node == b && matches!(op.kind, BoundOpKind::Elementwise { .. }))
        );
        assert!(built.iter().any(|op| op.node == d));
    }

    #[test]
    fn an_elementwise_intermediate_consumed_by_two_different_ops_is_not_fused() {
        let mut program = Vec::new();
        let a = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let identity = || IndexMap::Affine(map::projection(1, &[0]));
        let b = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Tanh,
                operands: alloc::vec![(a, identity())],
                name: None,
            },
        );
        let c1 = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Negate,
                operands: alloc::vec![(b, identity())],
                name: None,
            },
        );
        let c2 = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Reciprocal,
                operands: alloc::vec![(b, identity())],
                name: None,
            },
        );
        let d = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![(c1, identity()), (c2, identity())],
                name: None,
            },
        );

        let shapes = shape::infer(&program, &[]).expect("diamond chain infers");
        let built = bind(&program, &shapes, &[terminal(&program)], NumericPolicy::bit_exact())
            .expect("diamond chain builds ops");

        assert_eq!(
            built.len(),
            2,
            "b feeds two different consumers, so it must materialize once on its own, \
             and d (absorbing c1 and c2, whose only use each is d) is the other"
        );
        assert!(
            built
                .iter()
                .any(|op| op.node == b && matches!(op.kind, BoundOpKind::Elementwise { .. })),
            "b must materialize standalone rather than fuse into either consumer"
        );
        assert!(built.iter().any(|op| op.node == d));
        let _ = c1;
        let _ = c2;
    }

    /// `product = a * b; scaled = product * c; sum = reduce(+, scaled)` — two
    /// chained elementwise ops feeding a reduce, mirroring `matmul_program`
    /// but with an extra elementwise hop before the contraction. Both
    /// elementwise ops must absorb into the reduce's own `BoundOp`.
    #[test]
    fn elementwise_into_elementwise_into_reduce_fuses_into_one_bound_op() {
        let mut program = Vec::new();
        let a = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let b = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let c = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let identity = || IndexMap::Affine(map::projection(1, &[0]));
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![(a, identity()), (b, identity())],
                name: None,
            },
        );
        let scaled = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![(product, identity()), (c, identity())],
                name: None,
            },
        );
        let sum = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: scaled,
                in_map: identity(),
                out_map: IndexMap::Affine(map::projection(1, &[])),
                keep: Keep::Reduce,
                name: Some("weighted_dot".into()),
            }),
        );

        let shapes = shape::infer(&program, &[]).expect("weighted dot infers");
        let built = bind(&program, &shapes, &[terminal(&program)], NumericPolicy::bit_exact())
            .expect("weighted dot builds ops");

        assert_eq!(
            built.len(),
            1,
            "both elementwise hops must absorb into the reduce's own BoundOp"
        );
        assert_eq!(built[0].node, sum);
        assert!(matches!(built[0].kind, BoundOpKind::Reduce { .. }));
        assert_eq!(
            built[0].element_body().steps.len(),
            2,
            "one step per absorbed elementwise op"
        );
    }

    #[test]
    fn a_broadcast_operand_has_stride_zero_in_the_broadcast_axis() {
        let mut program = Vec::new();
        let matrix = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4), Extent::Static(8)],
                name: None,
            },
        );
        let bias = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(8)],
                name: None,
            },
        );
        let sum = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![
                    (matrix, IndexMap::Affine(map::projection(2, &[0, 1]))),
                    (bias, IndexMap::Affine(map::projection(2, &[1]))),
                ],
                name: None,
            },
        );

        let shapes = shape::infer(&program, &[]).expect("broadcast infers");
        let built =
            bind(&program, &shapes, &[terminal(&program)], NumericPolicy::bit_exact()).expect("broadcast builds ops");
        let op = built.iter().find(|op| op.node == sum).expect("sum emitted");
        assert_eq!(
            op.operands()[1].1.stride(0),
            0,
            "bias never varies over the batch axis"
        );
        assert_ne!(
            op.operands()[0].1.stride(0),
            0,
            "matrix does vary over the batch axis"
        );
    }

    #[test]
    fn a_conv_window_operand_folds_two_terms_into_one_stride_slot() {
        let mut program = Vec::new();
        let anchor = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4), Extent::Static(2)],
                name: None,
            },
        );
        let signal = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(8)],
                name: None,
            },
        );
        let window = IndexMap::Affine(map::affine(
            2,
            &[(
                &[
                    crate::map::AxisTerm::scaled(0, 2),
                    crate::map::AxisTerm::scaled(1, 1),
                ],
                0,
            )],
        ));
        let touched = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![
                    (anchor, IndexMap::Affine(map::projection(2, &[0, 1]))),
                    (signal, window)
                ],
                name: None,
            },
        );

        let shapes = shape::infer(&program, &[]).expect("conv window infers");
        let built = bind(&program, &shapes, &[terminal(&program)], NumericPolicy::bit_exact())
            .expect("conv window builds ops");
        let op = built
            .iter()
            .find(|op| op.node == touched)
            .expect("touched emitted");
        let signal_layout = &op.operands()[1].1;
        assert_eq!(
            signal_layout.strides.len(),
            2,
            "one stride slot per iteration axis"
        );
        assert_ne!(signal_layout.stride(0), 0, "stride term contributes");
        assert_ne!(signal_layout.stride(1), 0, "dilation term contributes");
    }

    #[test]
    fn transpose_layout_has_permuted_strides() {
        let mut program = Vec::new();
        let matrix = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(3), Extent::Static(5)],
                name: None,
            },
        );
        let transposed = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(matrix, IndexMap::Affine(map::projection(2, &[1, 0])))],
                name: None,
            },
        );

        let shapes = shape::infer(&program, &[]).expect("transpose infers");
        let built =
            bind(&program, &shapes, &[terminal(&program)], NumericPolicy::bit_exact()).expect("transpose builds ops");
        let op = built
            .iter()
            .find(|op| op.node == transposed)
            .expect("transposed emitted");
        let layout = &op.operands()[0].1;
        // matrix is row-major [3, 5]: elem strides are [5, 1]. axis 0 of the
        // operand (stride 5) projects iteration axis 1; axis 1 (stride 1)
        // projects iteration axis 0, so the strides land permuted relative
        // to iteration order.
        assert_eq!(layout.stride(0), 1);
        assert_eq!(layout.stride(1), 5);
    }

    /// [`correct_packed_matmul_layouts`]/[`native_packed_layout`] on a
    /// **two-axis output group** (`heads`, `head_dim`), the exact iteration
    /// shape `mistral_cached_forward_program`'s `wq`/`wk`/`wv` projections
    /// take (`tok`, `in`, `head`, `hd`), with `heads=3 != head_dim=4` so a
    /// swapped output-axis order changes the numbers, not just the labels —
    /// unlike `causal_conv1d`'s own `embedding=1` fixture, which made an
    /// `ld`/`dl` axis swap byte-identical and let the bug through. GGUF's
    /// native `[out_dim, in_dim]` row-major layout packs `out_dim` rows
    /// (`out_dim = heads * head_dim`, row index `head * head_dim + hd`) of
    /// `in_dim` contiguous elements each.
    #[test]
    fn correct_packed_matmul_layouts_derives_ggml_native_strides_for_a_two_axis_output_group() {
        const IN_DIM: u64 = 5;
        const HEADS: u64 = 3;
        const HEAD_DIM: u64 = 4;

        let mut program = Vec::new();
        let weight = append(
            &mut program,
            Op::Input {
                dtype: DType::UInt8,
                shape: alloc::vec![
                    Extent::Static(IN_DIM as u32),
                    Extent::Static(HEADS as u32),
                    Extent::Static(HEAD_DIM as u32)
                ],
                name: None,
            },
        );
        let activation = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(2), Extent::Static(IN_DIM as u32)],
                name: None,
            },
        );
        // iteration space (tok=0, in=1, head=2, hd=3): weight reads (in,
        // head, hd), ignoring tok; activation reads (tok, in).
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![
                    (weight, IndexMap::Affine(map::projection(4, &[1, 2, 3]))),
                    (activation, IndexMap::Affine(map::projection(4, &[0, 1]))),
                ],
                name: None,
            },
        );
        let sum = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: crate::op::ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(4, &[0, 1, 2, 3])),
                out_map: IndexMap::Affine(map::projection(4, &[0, 2, 3])),
                keep: Keep::Reduce,
                name: None,
            }),
        );

        let shapes = shape::infer(&program, &[]).expect("two-axis output group infers");
        let mut built = bind(&program, &shapes, &[terminal(&program)], NumericPolicy::bit_exact())
            .expect("two-axis output group binds");
        let packed: BTreeSet<NodeId> = core::iter::once(weight).collect();
        correct_packed_matmul_layouts(&mut built, &packed);

        let reduce = built
            .iter()
            .find(|op| op.node == sum)
            .expect("reduce emitted");
        let weight_layout = &reduce
            .operands()
            .iter()
            .find(|(node, _, _)| *node == weight)
            .expect("weight operand present in the reduce")
            .1;

        assert_eq!(
            weight_layout.stride(1),
            1,
            "in_dim is the innermost, contiguous axis of a GGUF row"
        );
        assert_eq!(
            weight_layout.stride(3),
            IN_DIM as i64,
            "head_dim steps by one whole in_dim row -- swapped with heads' stride below if the output-axis order flips"
        );
        assert_eq!(
            weight_layout.stride(2),
            (IN_DIM * HEAD_DIM) as i64,
            "heads steps by one whole head_dim block of rows -- swapped with head_dim's stride above if the output-axis order flips"
        );
        assert_eq!(
            weight_layout.stride(0),
            0,
            "weight never varies over the token batch axis"
        );
    }

    /// [`correct_packed_matmul_layouts`]/[`native_packed_layout`] on a
    /// **multi-axis contraction ("in") group** (`u`, `g`, `d`), the exact
    /// shape `wo` (attention output projection) composes its packed row
    /// index from in `mistral_layer3_forward_program`'s real checkpoint:
    /// `attn_head_dim*group*u + attn_head_dim*g + d`, i.e. THREE reduction
    /// letters folded into one packed-leaf axis, mirrored here at `u=2,
    /// g=2, d=3`. The sibling test above
    /// (`..._two_axis_output_group`) only ever proved the OUTPUT side can
    /// be multi-axis (`wq`/`wk`/`wv`'s own shape, a single-letter `in`);
    /// this is that same proof for the un-tested complementary case,
    /// output = a single axis `e`, contraction = three.
    #[test]
    fn correct_packed_matmul_layouts_derives_ggml_native_strides_for_a_multi_axis_contraction_group()
     {
        const SEQ: u64 = 2;
        const KV_HEADS: u64 = 2;
        const GROUP: u64 = 2;
        const HEAD_DIM: u64 = 3;
        const EMBED: u64 = 4;
        const IN_DIM: u64 = KV_HEADS * GROUP * HEAD_DIM;

        let mut program = Vec::new();
        let weight = append(
            &mut program,
            Op::Input {
                dtype: DType::UInt8,
                shape: alloc::vec![Extent::Static(IN_DIM as u32), Extent::Static(EMBED as u32)],
                name: None,
            },
        );
        let activation = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![
                    Extent::Static(SEQ as u32),
                    Extent::Static(KV_HEADS as u32),
                    Extent::Static(GROUP as u32),
                    Extent::Static(HEAD_DIM as u32)
                ],
                name: None,
            },
        );
        // iteration space (s=0, u=1, g=2, d=3, e=4): weight's own row axis
        // is the composed `(head_dim*group)*u + head_dim*g + d`, exactly
        // `wo_flat`'s own map string at `spec.rs:9024-9028` with the real
        // dims swapped for small distinguishable primes; weight's second
        // axis is the plain output letter `e`. Activation reads (s, u, g,
        // d), ignoring e (broadcast over the output axis, the `wo`
        // ones-broadcast idiom's own shape).
        let row_terms = [
            AxisTerm::scaled(1, i32::try_from(GROUP * HEAD_DIM).expect("fits i32")),
            AxisTerm::scaled(2, i32::try_from(HEAD_DIM).expect("fits i32")),
            AxisTerm::scaled(3, 1),
        ];
        let weight_map = IndexMap::Affine(map::affine(
            5,
            &[(&row_terms, 0), (&[AxisTerm::scaled(4, 1)], 0)],
        ));
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![
                    (weight, weight_map),
                    (
                        activation,
                        IndexMap::Affine(map::projection(5, &[0, 1, 2, 3]))
                    ),
                ],
                name: None,
            },
        );
        let sum = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: crate::op::ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(5, &[0, 1, 2, 3, 4])),
                out_map: IndexMap::Affine(map::projection(5, &[0, 4])),
                keep: Keep::Reduce,
                name: None,
            }),
        );

        let shapes = shape::infer(&program, &[]).expect("multi-axis contraction group infers");
        let mut built = bind(&program, &shapes, &[terminal(&program)], NumericPolicy::bit_exact())
            .expect("multi-axis contraction group binds");
        let packed: BTreeSet<NodeId> = core::iter::once(weight).collect();
        correct_packed_matmul_layouts(&mut built, &packed);

        let reduce = built
            .iter()
            .find(|op| op.node == sum)
            .expect("reduce emitted");
        let weight_layout = &reduce
            .operands()
            .iter()
            .find(|(node, _, _)| *node == weight)
            .expect("weight operand present in the reduce")
            .1;

        assert_eq!(
            weight_layout.stride(3),
            1,
            "head_dim (d) is the innermost, contiguous packed-row term"
        );
        assert_eq!(
            weight_layout.stride(2),
            HEAD_DIM as i64,
            "group (g) steps by one head_dim block"
        );
        assert_eq!(
            weight_layout.stride(1),
            (HEAD_DIM * GROUP) as i64,
            "kv_head (u) steps by one whole group*head_dim block"
        );
        assert_eq!(
            weight_layout.stride(4),
            IN_DIM as i64,
            "the output axis (e) steps by the whole packed row width"
        );
        assert_eq!(
            weight_layout.stride(0),
            0,
            "weight never varies over the sequence batch axis"
        );
    }

    fn elementwise_op() -> BoundOp {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(10), Extent::Static(4)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(source, IndexMap::Affine(map::projection(2, &[0, 1])))],
                name: None,
            },
        );
        let shapes = shape::infer(&program, &[]).expect("elementwise infers");
        bind(&program, &shapes, &[terminal(&program)], NumericPolicy::bit_exact())
            .expect("elementwise builds ops")
            .into_iter()
            .next()
            .expect("one op emitted")
    }

    fn scalar_reduction_op() -> BoundOp {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(8)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: source,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: IndexMap::Affine(map::projection(1, &[])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        let shapes = shape::infer(&program, &[]).expect("scalar reduction infers");
        bind(&program, &shapes, &[terminal(&program)], NumericPolicy::bit_exact())
            .expect("scalar reduction builds ops")
            .into_iter()
            .next()
            .expect("one op emitted")
    }

    fn scan_op() -> BoundOp {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(8)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: source,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: IndexMap::Affine(map::projection(1, &[0])),
                keep: Keep::Scan,
                name: None,
            }),
        );
        let shapes = shape::infer(&program, &[]).expect("scan infers");
        bind(&program, &shapes, &[terminal(&program)], NumericPolicy::bit_exact())
            .expect("scan builds ops")
            .into_iter()
            .next()
            .expect("one op emitted")
    }

    #[test]
    fn split_of_an_elementwise_op_yields_contiguous_chunks_with_a_ragged_last() {
        let op = elementwise_op();

        let chunks = op.split(3).expect("extent 10 over 3 parts splits");
        assert_eq!(chunks.len(), 3, "one chunk per part");

        let lengths: Vec<u64> = chunks.iter().map(|chunk| chunk.extents[0]).collect();
        assert_eq!(
            lengths,
            alloc::vec![3, 3, 4],
            "only the last chunk is ragged"
        );

        // axis 0's stride is 4 (row-major over [10, 4]): chunk k's operand
        // layout is rebased by chunk_start * stride(0), which is exactly
        // what lets a caller treat each chunk's output as a disjoint
        // sub-slice.
        let stride = op.operands()[0].1.stride(0);
        assert_eq!(chunks[0].operands()[0].1.base, op.operands()[0].1.base);
        assert_eq!(
            chunks[1].operands()[0].1.base,
            op.operands()[0].1.base + stride * 3
        );
        assert_eq!(
            chunks[2].operands()[0].1.base,
            op.operands()[0].1.base + stride * 6
        );
    }

    #[test]
    fn split_aligned_rounds_non_final_chunks_down_to_the_alignment() {
        let op = elementwise_op();

        // extent 10, 3 parts: raw_len = 10 / 3 = 3, rounded down to the
        // nearest multiple of 2 is 2 — only the final (already-ragged)
        // chunk absorbs what the rounding shaved off the other two.
        let chunks = op
            .split_aligned(3, 2)
            .expect("extent 10 over 3 parts splits");
        let lengths: Vec<u64> = chunks.iter().map(|chunk| chunk.extents[0]).collect();
        assert_eq!(
            lengths,
            alloc::vec![2, 2, 6],
            "non-final chunks round down to the alignment, final absorbs the rest"
        );
    }

    #[test]
    fn split_aligned_below_the_alignment_falls_back_to_unaligned() {
        let op = elementwise_op();

        // raw_len = 10 / 3 = 3 is already below alignment 4, so rounding
        // down would zero the chunk out — the doc promises the raw
        // unaligned width is kept instead.
        let chunks = op
            .split_aligned(3, 4)
            .expect("extent 10 over 3 parts splits");
        let lengths: Vec<u64> = chunks.iter().map(|chunk| chunk.extents[0]).collect();
        assert_eq!(
            lengths,
            alloc::vec![3, 3, 4],
            "falls back to split's own behavior"
        );
    }

    #[test]
    fn split_aligned_with_alignment_one_matches_split_exactly() {
        let op = elementwise_op();

        let aligned = op
            .split_aligned(3, 1)
            .expect("extent 10 over 3 parts splits");
        let plain = op.split(3).expect("extent 10 over 3 parts splits");
        let aligned_lengths: Vec<u64> = aligned.iter().map(|chunk| chunk.extents[0]).collect();
        let plain_lengths: Vec<u64> = plain.iter().map(|chunk| chunk.extents[0]).collect();
        assert_eq!(aligned_lengths, plain_lengths, "alignment 1 is a no-op");
    }

    #[test]
    fn split_of_a_fused_matmul_reduction_rebases_operands_but_not_out_layout() {
        let (program, _product, sum, _lhs) = matmul_program();
        let shapes = shape::infer(&program, &[512]).expect("matmul infers");
        let op = bind(&program, &shapes, &[terminal(&program)], NumericPolicy::bit_exact())
            .expect("matmul builds ops")
            .into_iter()
            .next()
            .expect("one fused op emitted");
        assert_eq!(op.node, sum);
        let BoundOpKind::Reduce { .. } = &op.kind else {
            panic!("the reduction fused with its elementwise op");
        };

        let chunks = op.split(2).expect("512 rows over 2 parts splits");
        assert_eq!(chunks.len(), 2, "one chunk per part");
        for chunk in &chunks {
            assert!(
                matches!(chunk.kind, BoundOpKind::Reduce { .. }),
                "each chunk is still a reduce"
            );
            // the contracted axis (k) is untouched by a split on the output
            // row axis: every chunk still walks the full contraction.
            assert_eq!(chunk.extents[2], op.extents[2]);
        }
        assert_eq!(chunks[0].extents[0], 256, "rows split evenly in half");
        assert_eq!(chunks[1].extents[0], 256);

        // the fused elementwise op's lhs/rhs operand reads are rebased:
        // chunk 1 starts reading lhs at row 256 (row stride = k = 768).
        let lhs_row_stride = op.operands()[0].1.stride(0);
        assert_eq!(
            chunks[1].operands()[0].1.base,
            op.operands()[0].1.base + lhs_row_stride * 256
        );

        // out_layout stays exactly as the parent's: the interpreter's own
        // per-chunk loop already starts each chunk's leading coordinate at
        // 0, so an unshifted out_layout already yields the 0-based write
        // offsets a `split_at_mut` sub-slice expects (see the `split` doc).
        let BoundOpKind::Reduce {
            out_layout: parent_out,
            ..
        } = &op.kind
        else {
            unreachable!("checked above")
        };
        for chunk in &chunks {
            let BoundOpKind::Reduce {
                out_layout: chunk_out,
                ..
            } = &chunk.kind
            else {
                panic!("chunk reduction");
            };
            assert_eq!(chunk_out, parent_out);
        }
    }

    #[proxima::test]
    #[case::scalar_reduction(scalar_reduction_op(), 2)]
    #[case::keep_scan_scan(scan_op(), 2)]
    #[case::too_few_parts(elementwise_op(), 1)]
    #[case::extent_smaller_than_parts(elementwise_op(), 999)]
    async fn split_returns_none_when_unsound_or_unhelpful(
        #[case] op: BoundOp,
        #[case] parts: usize,
    ) {
        assert!(op.split(parts).is_none());
    }

    /// A ternary `ScalarOp::Select` node (arity 3, the crate's current
    /// maximum) whose three operands are all held, non-fusing elementwise
    /// predecessors: a single `push` must materialize all three in one
    /// call, proving `push` can ready more than the two `BoundOp`s this
    /// module's docs once claimed as its ceiling — the true bound tracks
    /// `ScalarOp::arity()`, which is why `READY_BATCH_CAPACITY` is 3, not 2.
    #[test]
    fn select_push_emits_three_when_all_three_operands_are_held_and_non_fusing() {
        let mut program = Vec::new();
        let a = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let b = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let c = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let identity = || IndexMap::Affine(map::projection(1, &[0]));
        let held_a = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(a, identity())],
                name: None,
            },
        );
        let held_b = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(b, identity())],
                name: None,
            },
        );
        let held_c = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(c, identity())],
                name: None,
            },
        );
        program.push(Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Select,
            operands: alloc::vec![
                (held_a, identity()),
                (held_b, identity()),
                (held_c, identity()),
            ],
            name: None,
        });

        let shapes = shape::infer(&program, &[]).expect("select program infers");
        // an empty retires list (rather than `live::annotate`'s real kill-flags)
        // makes `retires.contains(operand_node)` false for every node, so
        // every held predecessor fails the fuse check regardless of its
        // projection — isolating exactly what a single push can materialize.
        let building = BoundOpBuilder::new(Vec::new(), NumericPolicy::bit_exact());
        let mut last_emitted_len = 0;
        for expr in program.iter() {
            let emitted = building.push(expr, &shapes).expect("push succeeds");
            last_emitted_len = emitted.len();
        }
        assert_eq!(
            last_emitted_len, 3,
            "the select node's push must materialize all three held, \
             non-fusing predecessors in one call: proves the 0/1/2 bound is \
             wrong, true bound tracks ScalarOp::arity() (3, Select)"
        );
    }

    // THE PROOF: `ShapeTable` and `BoundOpBuilder` compose through the real
    // `PipeExt` surface (`.and_then`, not hand-sequenced calls dressed up as
    // composition), and the ops that composed chain produces for a matmul
    // program are byte-for-byte the same ops `shape::infer` + `bind::bind`
    // (the free-function path every other test in this crate trusts)
    // produce for the identical program.
    #[test]
    fn infer_and_then_build_ops_matches_the_free_function_pipeline() {
        use crate::shape::ShapeTable;
        use proxima_primitives::pipe::PipeExt;

        let (program, _product, sum, _lhs) = matmul_program();
        let outputs: Vec<NodeId> = alloc::vec![sum];
        let retires = live::annotate(&program, &outputs);

        let shape_table = ShapeTable::new(&[512]);
        let builder = BoundOpBuilder::new(retires, NumericPolicy::bit_exact());
        let chain = shape_table.and_then(builder);

        let mut built_via_pipe = Vec::new();
        for expr in &program {
            let batch = proxima_primitives::block_on(Pipe::call(&chain, expr.clone()))
                .expect("shape+op pipe step succeeds");
            built_via_pipe.extend(batch);
        }

        let shapes = shape::infer(&program, &[512]).expect("free-function infer succeeds");
        let built_via_free_function = bind(&program, &shapes, &outputs, NumericPolicy::bit_exact())
            .expect("free-function op building succeeds");

        assert_eq!(built_via_pipe, built_via_free_function);
        assert_eq!(built_via_pipe.len(), 1, "matmul fuses into one op");
        assert_eq!(built_via_pipe[0].node, sum);
    }

    /// `docs/discipline.md` ROW 131 Limitation 2: a lone dtype-relabelling
    /// `Elementwise` (`indices_node`) is reached only through a sibling
    /// operand's `IndexMap::Computed { indices, .. }` field -- never through
    /// its own entry in any op's `operands` list. `gathered`'s first
    /// consumer (`first_use`) is not `gathered`'s *last* use, so `push`
    /// force-materializes `gathered` standalone, right there, long before
    /// the program ends -- while `indices_node`, visited by nothing but the
    /// map field this walk used to skip, would otherwise sit `held` until
    /// `finish`'s end-of-program sweep and land in `resolved` after the very
    /// op that reads its buffer. This is the exact shape the LFM2 causal
    /// conv gather hit (`spec.rs`'s `causal_conv1d`, which works around it
    /// by routing its own index through an `Op::Reduce` instead).
    fn computed_index_via_lone_elementwise_program() -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let base = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let identity = || IndexMap::Affine(map::projection(1, &[0]));
        let index_seed = append(
            &mut program,
            Op::Iota {
                dtype: DType::Float32,
                extent: Extent::Static(4),
            },
        );
        let indices_node = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Int32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(index_seed, identity())],
                name: None,
            },
        );
        let gathered_map = IndexMap::Computed {
            indices: indices_node,
            index_map: map::projection(1, &[0]),
            base: IndexPattern {
                iter_rank: 1,
                axes: alloc::vec![AxisIndex::default()],
            },
            gathered_dim: 0,
        };
        let gathered = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(base, gathered_map)],
                name: None,
            },
        );
        let first_use = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(gathered, identity())],
                name: None,
            },
        );
        let second_use = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![(gathered, identity()), (first_use, identity())],
                name: None,
            },
        );
        (program, second_use)
    }

    #[test]
    fn a_computed_index_reached_only_through_a_sibling_map_is_materialized_before_the_gather_reads_it()
     {
        let (program, output) = computed_index_via_lone_elementwise_program();
        let shapes = shape::infer(&program, &[]).expect("computed-index program infers");
        let base_data: Vec<f32> = alloc::vec![10.0, 20.0, 30.0, 40.0];

        let built = bind(&program, &shapes, &[output], NumericPolicy::bit_exact())
            .expect("binding itself never errors");
        let indices_position = built
            .iter()
            .position(|op| {
                matches!(op.kind, BoundOpKind::Elementwise { .. }) && op.dtype == DType::Int32
            })
            .expect("the indices node must appear as its own BoundOp");
        let gather_position = built
            .iter()
            .position(|op| op.operands().iter().any(|(_, _, lookup)| lookup.is_some()))
            .expect("the gathering node must appear as its own BoundOp");
        assert!(
            indices_position < gather_position,
            "resolved must stay topologically ordered: the indices node ({indices_position}) \
             must be built before the gather that reads it ({gather_position}), got {built:#?}"
        );

        let evaluated = crate::cpu::evaluate(&program, &[], &[&base_data], &[output]);
        assert!(
            evaluated.is_ok(),
            "the gather's indices buffer must be ready by the time the gather runs: {evaluated:?}"
        );
    }

    /// Hand-builds `proxima-autograd/src/conv.rs`'s `masked_window_axis`
    /// shape directly (that function is private to a different crate): one
    /// source axis widened by `(out_position, kernel_position)`, masked by
    /// `Equal(Iota, Add(Multiply(Iota, Constant), Iota))`, and reduced away —
    /// `proxima-tensor/docs/discipline.md` ROW 154's own fixture.
    /// `mask_body` lets a decline test swap `Equal` for something else
    /// without duplicating the rest of the shape.
    #[allow(clippy::too_many_arguments)]
    fn masked_window_reduce_program(
        source_rank: u16,
        windowed_axis: u16,
        source_extent: u64,
        out_extent: u64,
        kernel_extent: u64,
        stride: u64,
        mask_body: ScalarOp,
    ) -> (Vec<Op>, NodeId, NodeId) {
        let mut program = Vec::new();
        let widened_rank = source_rank + 2;
        let out_position_axis = source_rank;
        let kernel_position_axis = source_rank + 1;

        let source_shape: Vec<Extent> = (0..source_rank)
            .map(|axis| {
                Extent::Static(if u64::from(axis) == windowed_axis as u64 {
                    source_extent as u32
                } else {
                    4
                })
            })
            .collect();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: source_shape,
                name: None,
            },
        );

        let source_position = append(
            &mut program,
            Op::Iota {
                dtype: DType::Float32,
                extent: Extent::Static(source_extent as u32),
            },
        );
        let out_position = append(
            &mut program,
            Op::Iota {
                dtype: DType::Float32,
                extent: Extent::Static(out_extent as u32),
            },
        );
        let kernel_position = append(
            &mut program,
            Op::Iota {
                dtype: DType::Float32,
                extent: Extent::Static(kernel_extent as u32),
            },
        );
        let stride_const = append(
            &mut program,
            Op::Constant {
                dtype: DType::Float32,
                shape: Vec::new(),
                value: stride as f32,
            },
        );

        let identity1 = IndexMap::Affine(map::projection(1, &[0]));
        let broadcast1 = IndexMap::Affine(map::projection(1, &[]));
        let scaled_out = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![(out_position, identity1), (stride_const, broadcast1)],
                name: None,
            },
        );

        let combined = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![
                    (scaled_out, IndexMap::Affine(map::projection(2, &[0]))),
                    (kernel_position, IndexMap::Affine(map::projection(2, &[1]))),
                ],
                name: None,
            },
        );

        let mask = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: mask_body,
                operands: alloc::vec![
                    (source_position, IndexMap::Affine(map::projection(3, &[0]))),
                    (combined, IndexMap::Affine(map::projection(3, &[1, 2]))),
                ],
                name: None,
            },
        );

        let source_axes: Vec<u16> = (0..source_rank).collect();
        let source_pattern = IndexMap::Affine(map::projection(widened_rank, &source_axes));
        let mask_pattern = IndexMap::Affine(map::projection(
            widened_rank,
            &[windowed_axis, out_position_axis, kernel_position_axis],
        ));

        let masked = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![(source, source_pattern), (mask, mask_pattern)],
                name: None,
            },
        );

        let keep_axes: Vec<u16> = (0..widened_rank)
            .filter(|&axis| axis != windowed_axis)
            .collect();
        let out_map = IndexMap::Affine(map::projection(widened_rank, &keep_axes));
        let identity_widened = IndexMap::Affine(map::projection(
            widened_rank,
            &(0..widened_rank).collect::<Vec<u16>>(),
        ));
        let reduced = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: crate::op::ReduceInit::Zero,
                operand: masked,
                in_map: identity_widened,
                out_map,
                keep: Keep::Reduce,
                name: None,
            }),
        );

        (program, source, reduced)
    }

    #[test]
    fn a_masked_window_reduce_folds_to_a_single_operand_identity_read_of_source() {
        let (program, source, reduced) =
            masked_window_reduce_program(1, 0, 5, 3, 3, 1, ScalarOp::Equal);
        let shapes = shape::infer(&program, &[]).expect("masked-window program infers");
        let built = bind(&program, &shapes, &[terminal(&program)], NumericPolicy::bit_exact())
            .expect("masked-window program builds ops");

        let folded = built
            .iter()
            .find(|op| op.node == reduced)
            .expect("the reduce node's own BoundOp is still present, just re-shaped");
        assert!(
            matches!(folded.kind, BoundOpKind::Elementwise { .. }),
            "a proven in-bounds window read becomes a plain elementwise gather, not a Reduce, got {:?}",
            folded.kind
        );
        assert_eq!(
            folded.operands().len(),
            1,
            "the fold reads only source, the mask chain is gone"
        );
        assert_eq!(folded.operands()[0].0, source);
        assert_eq!(
            folded.operands()[0].1.strides.len(),
            2,
            "the source's one windowed axis now derives from two output axes"
        );
        assert!(
            folded.operands()[0]
                .1
                .strides
                .iter()
                .all(|&stride| stride != 0),
            "both the out_position and kernel_position axes must contribute to the source address"
        );
        assert_eq!(folded.element_body().steps.len(), 1);
        assert_eq!(folded.element_body().steps[0].op, ScalarOp::Identity);
    }

    #[test]
    fn a_masked_window_reduce_that_fails_the_in_bounds_proof_declines_and_binds_as_a_reduce() {
        // stride*(out_extent-1) + (kernel_extent-1) = 2*2 + 2 = 6 >= source_extent(5): out of bounds.
        let (program, _source, reduced) =
            masked_window_reduce_program(1, 0, 5, 3, 3, 2, ScalarOp::Equal);
        let shapes = shape::infer(&program, &[]).expect("masked-window program infers");
        let built = bind(&program, &shapes, &[terminal(&program)], NumericPolicy::bit_exact())
            .expect("masked-window program builds ops");

        let folded = built
            .iter()
            .find(|op| op.node == reduced)
            .expect("the reduce node's own BoundOp is still present");
        assert!(
            matches!(folded.kind, BoundOpKind::Reduce { .. }),
            "a failed in-bounds proof must decline the fold and bind the ordinary Reduce, got {:?}",
            folded.kind
        );
    }

    #[test]
    fn a_masked_window_reduce_with_a_non_windowed_axis_matches_a_direct_window_read() {
        // source: [channel=4 (helper's own non-windowed default extent), position=5],
        // windowed_axis=1, stride=1, kernel=3 -> out=3.
        let (program, _source, reduced) =
            masked_window_reduce_program(2, 1, 5, 3, 3, 1, ScalarOp::Equal);
        let source_data: Vec<f32> = (0..4 * 5).map(|index| index as f32 + 1.0).collect();
        let evaluated = crate::cpu::evaluate(&program, &[], &[&source_data], &[reduced])
            .expect("masked-window program evaluates");
        let (windowed, _shape) = evaluated
            .get(reduced)
            .expect("reduce node's output buffer is present");

        // expected[channel, out_position, kernel_position] = source[channel, out_position + kernel_position]
        let mut expected = alloc::vec![0.0f32; 4 * 3 * 3];
        for channel in 0..4usize {
            for out_position in 0..3usize {
                for kernel_position in 0..3usize {
                    let source_position = out_position + kernel_position;
                    expected[channel * 9 + out_position * 3 + kernel_position] =
                        source_data[channel * 5 + source_position];
                }
            }
        }
        assert_eq!(
            windowed,
            expected.as_slice(),
            "the folded read must match the direct window gather exactly"
        );
    }

    #[test]
    fn a_non_equal_mask_chain_declines_and_binds_as_a_reduce() {
        let (program, _source, reduced) =
            masked_window_reduce_program(1, 0, 5, 3, 3, 1, ScalarOp::Greater);
        let shapes = shape::infer(&program, &[]).expect("masked-window program infers");
        let built = bind(&program, &shapes, &[terminal(&program)], NumericPolicy::bit_exact())
            .expect("masked-window program builds ops");

        let folded = built
            .iter()
            .find(|op| op.node == reduced)
            .expect("the reduce node's own BoundOp is still present");
        assert!(
            matches!(folded.kind, BoundOpKind::Reduce { .. }),
            "a mask chain that is not the exact Equal/Iota shape must decline the fold, got {:?}",
            folded.kind
        );
    }

    #[cfg(feature = "reduce-epilogue-fusion")]
    mod reduce_epilogue_fusion_tests {
        use core::pin::pin;
        use core::task::{Context, Poll, Waker};

        use super::*;
        use crate::cpu::Interpreter;
        use crate::test_support::Lcg;

        /// `weights: [K, N]` folded over `K` into `reduced: [N]`, then a
        /// plain `x: [N]` residual add — the exact `residual1 = Add(attn_out,
        /// x)` shape `docs/dispatch-census.md` names. `y = Negate(x)` gives
        /// `x` a SECOND, independent use so this test also proves the rule
        /// only cares about the REDUCE's own liveness, not any other
        /// operand's.
        fn reduce_then_residual_add_program() -> (Vec<Op>, NodeId, NodeId, NodeId, NodeId) {
            let mut program = Vec::new();
            let weights = append(
                &mut program,
                Op::Input {
                    dtype: DType::Float32,
                    shape: alloc::vec![Extent::Static(8), Extent::Static(4)],
                    name: None,
                },
            );
            let reduced = append(
                &mut program,
                Op::Reduce(Reduce {
                    dtype: DType::Float32,
                    body: ScalarOp::Add,
                    init: ReduceInit::Zero,
                    operand: weights,
                    in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
                    out_map: IndexMap::Affine(map::projection(2, &[1])),
                    keep: Keep::Reduce,
                    name: None,
                }),
            );
            let x = append(
                &mut program,
                Op::Input {
                    dtype: DType::Float32,
                    shape: alloc::vec![Extent::Static(4)],
                    name: None,
                },
            );
            let identity = || IndexMap::Affine(map::projection(1, &[0]));
            let consumer = append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Add,
                    operands: alloc::vec![(reduced, identity()), (x, identity())],
                    name: None,
                },
            );
            let extra_x_use = append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Negate,
                    operands: alloc::vec![(x, identity())],
                    name: None,
                },
            );
            (program, reduced, consumer, x, extra_x_use)
        }

        /// The one non-default field this whole rule adds — a real
        /// `epilogue_operands` entry — is the test's own positive signal
        /// that fusion actually happened, not merely that the op count
        /// dropped (a count-only assertion can't distinguish this rule
        /// firing from some unrelated node going dead).
        fn has_real_epilogue(kind: &BoundOpKind) -> bool {
            matches!(kind, BoundOpKind::Reduce { epilogue_operands, .. } if !epilogue_operands.is_empty())
        }

        /// Structural invariant every fusion pass must preserve: every
        /// [`NodeId`] any resolved op's [`BoundOp::all_read_sources`] names
        /// must either be an [`Op::Input`] leaf (which never gets its own
        /// [`BoundOp`], per [`BoundOpKind`]'s own doc) or still be one of
        /// `resolved`'s own [`BoundOp::node`]s. A fusion pass that absorbs a
        /// producer (`reduce_epilogue_fusion`'s own `absorbed` set) but
        /// leaves some OTHER operand slot still pointing at that now-gone
        /// producer would pass every count/shape assertion while reading a
        /// buffer that was never materialized — exactly the dangling-slot
        /// bug `compose_reduce_epilogue` had when it substituted only the
        /// FIRST matching operand instead of every occurrence.
        fn assert_no_dangling_operand_references(program: &[Op], resolved: &[BoundOp]) {
            let live_nodes: BTreeSet<NodeId> = resolved.iter().map(|bound| bound.node).collect();
            let is_leaf_input =
                |node: &NodeId| matches!(program.get(node.0 as usize), Some(Op::Input { .. }));
            for bound in resolved {
                for (source, _, gather) in bound.all_read_sources() {
                    assert!(
                        live_nodes.contains(source) || is_leaf_input(source),
                        "node {:?} reads {source:?}, which no BoundOp in the resolved list produces \
                         and which is not an Op::Input leaf",
                        bound.node
                    );
                    if let Some(lookup) = gather {
                        assert!(
                            live_nodes.contains(&lookup.indices) || is_leaf_input(&lookup.indices),
                            "node {:?} gathers through {:?}, which no BoundOp in the resolved list \
                             produces and which is not an Op::Input leaf",
                            bound.node,
                            lookup.indices
                        );
                    }
                }
            }
        }

        #[test]
        fn reduce_then_residual_add_fuses_into_one_epilogued_reduce() {
            let (program, reduced, consumer, _x, extra_x_use) = reduce_then_residual_add_program();
            let shapes = shape::infer(&program, &[]).expect("residual-add program infers");
            let plain = bind_plain(
                &program,
                &shapes,
                &[extra_x_use],
                NumericPolicy::bit_exact(),
            )
            .expect("plain bind succeeds");
            let fused = bind(
                &program,
                &shapes,
                &[extra_x_use],
                NumericPolicy::bit_exact(),
            )
            .expect("fused bind succeeds");

            assert_eq!(
                fused.len(),
                plain.len() - 1,
                "the standalone reduce disappears into the consumer's epilogue"
            );
            assert!(
                !fused.iter().any(|bound| bound.node == reduced),
                "the reduce's own NodeId no longer names a standalone BoundOp"
            );
            let merged = fused
                .iter()
                .find(|bound| bound.node == consumer)
                .expect("the consumer's NodeId now names the fused reduce+epilogue op");
            assert!(
                has_real_epilogue(&merged.kind),
                "the fused op must carry a real epilogue, got {:?}",
                merged.kind
            );
        }

        /// Runs an already-resolved `Vec<BoundOp>` through
        /// [`crate::cpu::Interpreter`] the same way
        /// `cached_attention_single_range_fused_matches_the_unfused_program`
        /// (`spec.rs`) already does for its own fused-vs-unfused A/B — the
        /// one entry point that accepts a caller's own bind result instead
        /// of re-binding internally the way [`crate::cpu::evaluate`] does
        /// ([`crate::cpu::evaluate`]'s own `prepare` calls `bind::bind`
        /// unconditionally, so it can never produce the un-epilogued half of
        /// this comparison once the `reduce-epilogue-fusion` feature is
        /// compiled in).
        fn run_resolved(
            program_len: usize,
            resolved: &[BoundOp],
            inputs: Vec<(NodeId, Vec<f32>)>,
        ) -> Vec<Option<Vec<f32>>> {
            let mut buffers: Vec<Option<Vec<f32>>> = alloc::vec![None; program_len];
            for (node, data) in inputs {
                buffers[node.0 as usize] = Some(data);
            }
            let interpreter = Interpreter::new(&mut buffers);
            for chunk in resolved.chunks(READY_BATCH_CAPACITY) {
                let batch: ReadyBatch = chunk.iter().cloned().collect();
                let waker = Waker::noop();
                let mut context = Context::from_waker(waker);
                let mut future = pin!(interpreter.call(batch));
                match future.as_mut().poll(&mut context) {
                    Poll::Ready(result) => {
                        result.expect("resolved batch computes");
                    }
                    Poll::Pending => unreachable!("cpu pipes never yield: no internal .await"),
                }
            }
            buffers
        }

        /// Hand-derivable ground truth over 8x4 weights and a length-4
        /// residual: `reduced[n] = sum_k weights[k, n]`, `consumer[n] =
        /// reduced[n] + x[n]` — exact values, not a tolerance band, because
        /// every input is an exact `f32` the sum can reproduce bit-for-bit
        /// with `Lcg`'s own small integer-ish range.
        #[test]
        fn reduce_epilogue_evaluator_matches_hand_derived_values() {
            let (program, _reduced, consumer, _x, extra_x_use) = reduce_then_residual_add_program();
            let outputs = alloc::vec![consumer, extra_x_use];
            let shapes = shape::infer(&program, &[]).expect("residual-add program infers");

            let mut lcg = Lcg(42);
            let weights: Vec<f32> = (0..32).map(|_| lcg.next_unit()).collect();
            let residual: Vec<f32> = (0..4).map(|_| lcg.next_unit()).collect();

            let expected: Vec<f32> = (0..4)
                .map(|column| {
                    let sum: f32 = (0..8).map(|row| weights[row * 4 + column]).sum();
                    sum + residual[column]
                })
                .collect();

            let inputs = || -> Vec<(NodeId, Vec<f32>)> {
                block_node_ids(&program)
                    .into_iter()
                    .map(|node| {
                        let data = match &program[node.0 as usize] {
                            Op::Input { shape, .. } if shape.len() == 2 => weights.clone(),
                            _ => residual.clone(),
                        };
                        (node, data)
                    })
                    .collect()
            };

            let fused = bind(&program, &shapes, &outputs, NumericPolicy::bit_exact())
                .expect("fused bind succeeds");
            let fused_buffers = run_resolved(program.len(), &fused, inputs());
            let fused_consumer = fused_buffers[consumer.0 as usize]
                .as_ref()
                .expect("fused consumer output present");

            assert_eq!(
                fused_consumer, &expected,
                "epilogued reduce must match the hand-derived sum-plus-residual exactly"
            );

            let plain = bind_plain(&program, &shapes, &outputs, NumericPolicy::bit_exact())
                .expect("plain bind succeeds");
            let plain_buffers = run_resolved(program.len(), &plain, inputs());
            let plain_consumer = plain_buffers[consumer.0 as usize]
                .as_ref()
                .expect("plain consumer output present");
            assert_eq!(
                fused_consumer, plain_consumer,
                "the epilogue-fused evaluator must match the plain (unfused) evaluator exactly"
            );
        }

        #[test]
        fn a_reduce_with_two_consumers_does_not_fuse() {
            let (program, reduced, consumer, _x, extra_x_use) = reduce_then_residual_add_program();
            let identity = || IndexMap::Affine(map::projection(1, &[0]));
            let mut program = program;
            let second_consumer = append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Negate,
                    operands: alloc::vec![(reduced, identity())],
                    name: None,
                },
            );
            let shapes = shape::infer(&program, &[]).expect("two-consumer program infers");
            let fused = bind(
                &program,
                &shapes,
                &[extra_x_use, second_consumer],
                NumericPolicy::bit_exact(),
            )
            .expect("two-consumer program still binds");

            assert!(
                fused.iter().any(|bound| bound.node == reduced),
                "a reduce with a second consumer must still materialize standalone"
            );
            let consumer_bound = fused
                .iter()
                .find(|bound| bound.node == consumer)
                .expect("the first consumer's own BoundOp is still present");
            assert!(
                !has_real_epilogue(&consumer_bound.kind),
                "a sole-consumer requirement violation must never carry an epilogue"
            );
        }

        #[test]
        fn a_strided_consumer_map_does_not_fuse() {
            let (mut program, reduced, _consumer, x, extra_x_use) =
                reduce_then_residual_add_program();
            // overwrite the last-appended node (the ordinary-identity
            // consumer) with a build that reads `reduced` REVERSED
            // (`coeff: -1, offset: 3` over a 4-element axis walks indices
            // 3,2,1,0) instead of through a genuine identity projection —
            // `is_identity_projection` rejects any `coeff != 1` regardless
            // of how the offset keeps it in-bounds, so this consumer must
            // decline the fold and materialize both nodes normally.
            let consumer_index = program
                .iter()
                .position(|expr| {
                    matches!(
                        expr,
                        Op::Elementwise { body: ScalarOp::Add, operands, .. }
                            if operands.iter().any(|(node, _)| *node == reduced)
                    )
                })
                .expect("the residual-add consumer is present in the program");
            let strided_map = IndexMap::Affine(IndexPattern {
                iter_rank: 1,
                axes: alloc::vec![AxisIndex {
                    terms: SmallVec::from_slice(&[AxisTerm { axis: 0, coeff: -1 }]),
                    offset: 3,
                    len: None,
                }],
            });
            program[consumer_index] = Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![
                    (reduced, strided_map),
                    (x, IndexMap::Affine(map::projection(1, &[0]))),
                ],
                name: None,
            };
            let consumer = NodeId(consumer_index as u32);
            let shapes = shape::infer(&program, &[]).expect("strided-consumer program infers");
            let fused = bind(
                &program,
                &shapes,
                &[extra_x_use, consumer],
                NumericPolicy::bit_exact(),
            )
            .expect("strided-consumer program still binds");

            assert!(
                fused.iter().any(|bound| bound.node == reduced),
                "a non-identity consumer map must leave the reduce standalone"
            );
            let consumer_bound = fused
                .iter()
                .find(|bound| bound.node == consumer)
                .expect("the strided consumer's own BoundOp is still present");
            assert!(
                !has_real_epilogue(&consumer_bound.kind),
                "a non-identity projection must never carry an epilogue"
            );
        }

        #[test]
        fn a_required_output_consumer_still_fuses() {
            let (program, reduced, consumer, _x, extra_x_use) = reduce_then_residual_add_program();
            let shapes = shape::infer(&program, &[]).expect("residual-add program infers");
            // `consumer` itself is now a REQUESTED output — condition (b)
            // only forbids the REDUCE from being a requested output; the
            // consumer becoming one is exactly the case the fused op's own
            // `node == consumer` convention exists for for (the epilogue
            // output IS the output, so nothing needs to keep the reduce
            // materialized separately).
            let fused = bind(
                &program,
                &shapes,
                &[consumer, extra_x_use],
                NumericPolicy::bit_exact(),
            )
            .expect("fused bind with the consumer as a required output succeeds");

            assert!(
                !fused.iter().any(|bound| bound.node == reduced),
                "the reduce still disappears even though its consumer is a required output"
            );
            let merged = fused
                .iter()
                .find(|bound| bound.node == consumer)
                .expect("the required-output consumer's NodeId still names a BoundOp");
            assert!(
                has_real_epilogue(&merged.kind),
                "a required-output consumer must still fuse, got {:?}",
                merged.kind
            );
        }

        /// The real openchat-3.5/Mistral-7B single-range fixture (same
        /// shape as `single_range_cached_attention_fuses_one_step_per_
        /// layer_on_the_real_openchat_shape` above) with BOTH
        /// `cached-attention-streaming` and `reduce-epilogue-fusion` on.
        /// MEASURED, not derived: printed once via the per-kind buckets
        /// below so a future re-run can diff against this row's own
        /// numbers without re-deriving them from the dispatch census.
        #[test]
        fn reduce_epilogue_fusion_shrinks_the_real_openchat_single_range_program() {
            let (program, logits, cache_roots, _) =
                crate::spec::mistral_single_range_cached_forward_program(
                    32_002,
                    4096,
                    14336,
                    32,
                    8,
                    128,
                    32,
                    false,
                    crate::spec::DuplicateHeadPosition::None,
                    false,
                )
                .expect("openchat-shaped single-range forward pass lowers to a program");
            let mut outputs = alloc::vec![logits];
            for (even, odd, value) in &cache_roots {
                outputs.extend_from_slice(&[*even, *odd, *value]);
            }
            let shapes = crate::shape::infer(&program, &[1, 71])
                .expect("one new position against a 71-position merged range infers");
            let attention_only = bind_cached_attention_fusion(
                &program,
                &shapes,
                &outputs,
                true,
                NumericPolicy::bit_exact(),
            )
            .expect("cached-attention-only bind succeeds");
            let with_epilogue = bind(&program, &shapes, &outputs, NumericPolicy::bit_exact())
                .expect("fused bind succeeds");

            let epilogue_count = with_epilogue
                .iter()
                .filter(|bound| has_real_epilogue(&bound.kind))
                .count();
            let reduce_count = with_epilogue
                .iter()
                .filter(|bound| matches!(bound.kind, BoundOpKind::Reduce { .. }))
                .count();
            let elementwise_count = with_epilogue
                .iter()
                .filter(|bound| matches!(bound.kind, BoundOpKind::Elementwise { .. }))
                .count();
            let cached_attention_count = with_epilogue
                .iter()
                .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
                .count();
            std::eprintln!(
                "cached-attention-only={} with-epilogue={} epilogued-reduces={} \
                 reduce={} elementwise={} cached-attention={}",
                attention_only.len(),
                with_epilogue.len(),
                epilogue_count,
                reduce_count,
                elementwise_count,
                cached_attention_count,
            );
            assert!(
                with_epilogue.len() < attention_only.len(),
                "reduce-epilogue-fusion must remove at least one bound op vs cached-attention alone"
            );
            assert!(
                epilogue_count > 0,
                "the real openchat shape must produce at least one epilogued reduce"
            );
            // MEASURED (not derived) on this checkout: 619 -> 458, a 161-op
            // drop. Every RMSNorm's own `mean/eps/sqrt/reciprocal` PLAIN
            // epilogue round chains into a SECOND, broadcast-reduce round
            // (`x * inv_rms`), and every `SiLU(gate) * up`-shaped tail (two
            // independent reduces feeding one consumer) fuses too. The prior
            // measured value here (490) was taken while `find_epilogue_source`/
            // `resolved_reference_counts` still counted a consumer's OWN
            // repeated read of the same source (production SiLU reads `gate`
            // once bare, once inside `exp(-gate)`) as a second, conflicting
            // consumer and declined the fold — so the real SiLU*up class
            // this comment already claimed was fusing was NOT actually
            // firing on this program; only RMSNorm's broadcast-reduce round
            // was. Fixing that reference-count/projection bug is what widens
            // 490 -> 458. A regression here means one of the two classes
            // stopped firing, not merely "fewer than before".
            assert_eq!(
                attention_only.len(),
                619,
                "cached-attention-only baseline shifted; re-derive before trusting the epilogue delta below"
            );
            assert_eq!(
                with_epilogue.len(),
                458,
                "reduce-epilogue-fusion's own op count regressed from the measured 458"
            );
        }

        /// The same `mistral_single_range_cached_forward_program` builder the
        /// structural test above proves the bound-op COUNT for, run end to
        /// end through [`Interpreter`]: `bind`'s own epilogue-fused resolve
        /// against `bind_cached_attention_fusion`'s un-epilogued one, same
        /// program, same weights, same cache — a divergence here can only be
        /// the epilogue evaluator (`cpu::apply_reduce_epilogue`), never a
        /// shape or binding difference. Scaled down from the structural
        /// test's real `vocab=32_002, hidden=4096` shape to one a unit test
        /// can actually execute; the structural test already measured the
        /// full openchat shape's bound-op counts, so this only needs to
        /// re-prove VALUES agree, at a shape small enough to run in
        /// milliseconds.
        #[test]
        fn reduce_epilogue_fusion_matches_the_unfused_program_on_the_real_single_range_shape() {
            const VOCAB: u32 = 5;
            const EMBEDDING: u32 = 4;
            const FEED_FORWARD: u32 = 4;
            const QUERY_HEADS: u32 = 2;
            const KV_HEADS: u32 = 1;
            const HEAD_DIM: u32 = 2;
            const BLOCK_COUNT: u32 = 1;
            const CACHED_LEN: usize = 3;
            const NEW_COUNT: usize = 2;
            const MERGED_LEN: usize = CACHED_LEN + NEW_COUNT;
            let pairs = (HEAD_DIM / 2) as usize;
            let group = (QUERY_HEADS / KV_HEADS) as usize;

            let (program, logits, cache_roots, _) =
                crate::spec::mistral_single_range_cached_forward_program(
                    VOCAB,
                    EMBEDDING,
                    FEED_FORWARD,
                    QUERY_HEADS,
                    KV_HEADS,
                    HEAD_DIM,
                    BLOCK_COUNT,
                    false,
                    crate::spec::DuplicateHeadPosition::None,
                    false,
                )
                .expect("single-range cached forward pass lowers");
            let mut outputs = alloc::vec![logits];
            for (even, odd, value) in &cache_roots {
                outputs.extend_from_slice(&[*even, *odd, *value]);
            }
            let shapes = shape::infer(&program, &[NEW_COUNT as u64, MERGED_LEN as u64])
                .expect("single-range fixture infers");

            let mut lcg = Lcg(7);
            let mut named: Vec<(String, Vec<f32>)> = alloc::vec![
                (
                    String::from("token_embd.weight"),
                    (0..VOCAB as usize * EMBEDDING as usize)
                        .map(|_| lcg.next_unit())
                        .collect()
                ),
                (
                    String::from("ids"),
                    (0..NEW_COUNT).map(|id| 1.0 + (id % 3) as f32).collect()
                ),
                (String::from("eps"), alloc::vec![1e-5f32; NEW_COUNT]),
                (
                    String::from("rope_cos"),
                    (0..NEW_COUNT * pairs).map(|_| lcg.next_unit()).collect()
                ),
                (
                    String::from("rope_sin"),
                    (0..NEW_COUNT * pairs).map(|_| lcg.next_unit()).collect()
                ),
                (String::from("cached_len"), alloc::vec![CACHED_LEN as f32]),
            ];
            for layer in 0..BLOCK_COUNT as usize {
                named.push((
                    alloc::format!("blk.{layer}.attn_norm.weight"),
                    alloc::vec![1.0f32; EMBEDDING as usize],
                ));
                named.push((
                    alloc::format!("blk.{layer}.ffn_norm.weight"),
                    alloc::vec![1.0f32; EMBEDDING as usize],
                ));
                named.push((
                    alloc::format!("blk.{layer}.attn_q.weight"),
                    (0..EMBEDDING as usize * QUERY_HEADS as usize * HEAD_DIM as usize)
                        .map(|_| lcg.next_unit())
                        .collect(),
                ));
                named.push((
                    alloc::format!("blk.{layer}.attn_k.weight"),
                    (0..EMBEDDING as usize * KV_HEADS as usize * HEAD_DIM as usize)
                        .map(|_| lcg.next_unit())
                        .collect(),
                ));
                named.push((
                    alloc::format!("blk.{layer}.attn_v.weight"),
                    (0..EMBEDDING as usize * KV_HEADS as usize * HEAD_DIM as usize)
                        .map(|_| lcg.next_unit())
                        .collect(),
                ));
                named.push((
                    alloc::format!("blk.{layer}.attn_output.weight"),
                    (0..KV_HEADS as usize * group * HEAD_DIM as usize * EMBEDDING as usize)
                        .map(|_| lcg.next_unit())
                        .collect(),
                ));
                named.push((
                    alloc::format!("blk.{layer}.ffn_gate.weight"),
                    (0..EMBEDDING as usize * FEED_FORWARD as usize)
                        .map(|_| lcg.next_unit())
                        .collect(),
                ));
                named.push((
                    alloc::format!("blk.{layer}.ffn_up.weight"),
                    (0..EMBEDDING as usize * FEED_FORWARD as usize)
                        .map(|_| lcg.next_unit())
                        .collect(),
                ));
                named.push((
                    alloc::format!("blk.{layer}.ffn_down.weight"),
                    (0..FEED_FORWARD as usize * EMBEDDING as usize)
                        .map(|_| lcg.next_unit())
                        .collect(),
                ));
                named.push((
                    alloc::format!("kv_cache.{layer}.k_even"),
                    (0..MERGED_LEN * KV_HEADS as usize * pairs)
                        .map(|_| lcg.next_unit())
                        .collect(),
                ));
                named.push((
                    alloc::format!("kv_cache.{layer}.k_odd"),
                    (0..MERGED_LEN * KV_HEADS as usize * pairs)
                        .map(|_| lcg.next_unit())
                        .collect(),
                ));
                named.push((
                    alloc::format!("kv_cache.{layer}.v"),
                    (0..MERGED_LEN * KV_HEADS as usize * HEAD_DIM as usize)
                        .map(|_| lcg.next_unit())
                        .collect(),
                ));
            }
            named.push((
                String::from("output_norm.weight"),
                alloc::vec![1.0f32; EMBEDDING as usize],
            ));
            named.push((
                String::from("output.weight"),
                (0..EMBEDDING as usize * VOCAB as usize)
                    .map(|_| lcg.next_unit())
                    .collect(),
            ));

            let inputs = || -> Vec<(NodeId, Vec<f32>)> {
                block_node_ids(&program)
                    .into_iter()
                    .map(|node| {
                        let name = match &program[node.0 as usize] {
                            Op::Input {
                                name: Some(name), ..
                            } => name.clone(),
                            _ => unreachable!(
                                "block_node_ids only ever returns named Op::Input nodes"
                            ),
                        };
                        let data = named
                            .iter()
                            .find(|(candidate, _)| *candidate == name)
                            .unwrap_or_else(|| panic!("missing named input {name}"))
                            .1
                            .clone();
                        (node, data)
                    })
                    .collect()
            };

            let with_epilogue = bind(&program, &shapes, &outputs, NumericPolicy::bit_exact())
                .expect("fused bind succeeds");
            let attention_only = bind_cached_attention_fusion(
                &program,
                &shapes,
                &outputs,
                true,
                NumericPolicy::bit_exact(),
            )
            .expect("cached-attention-only bind succeeds");
            assert!(
                with_epilogue
                    .iter()
                    .any(|bound| has_real_epilogue(&bound.kind)),
                "this shape must still produce at least one epilogued reduce at the smaller scale"
            );

            let epilogue_buffers = run_resolved(program.len(), &with_epilogue, inputs());
            let plain_buffers = run_resolved(program.len(), &attention_only, inputs());

            for node in &outputs {
                let epilogue_output = epilogue_buffers[node.0 as usize]
                    .as_ref()
                    .unwrap_or_else(|| panic!("epilogued output present for {node:?}"));
                let plain_output = plain_buffers[node.0 as usize]
                    .as_ref()
                    .unwrap_or_else(|| panic!("unfused output present for {node:?}"));
                assert_eq!(epilogue_output.len(), plain_output.len());
                let peak = plain_output
                    .iter()
                    .fold(0.0f32, |peak, value| peak.max(value.abs()));
                let max_abs_error = epilogue_output
                    .iter()
                    .zip(plain_output.iter())
                    .map(|(fused, plain)| (fused - plain).abs())
                    .fold(0.0f32, f32::max);
                let max_rel_error = if peak > 0.0 {
                    max_abs_error / peak
                } else {
                    max_abs_error
                };
                std::eprintln!(
                    "reduce_epilogue_fusion_matches_the_unfused_program node={node:?} \
                     max_abs_error={max_abs_error} max_rel_error={max_rel_error}"
                );
                assert!(
                    max_abs_error <= 1e-6 && max_rel_error <= 1e-6,
                    "epilogue-fused output diverged from the unfused program at node {node:?}: \
                     max_abs_error={max_abs_error} max_rel_error={max_rel_error}"
                );
            }
        }

        /// `specs/mistral_layer.toml`'s own RMSNorm, node for node
        /// (`crate::spec`'s own private `rmsnorm` helper mirrors this exact
        /// shape): `x: [seq, dim]` reduced over `dim` into `mean_square:
        /// [seq]`, then `x * inv_rms` re-BROADCASTS that scalar back over
        /// `dim` — the "broadcast-reduce" epilogue
        /// [`BoundOpKind::Reduce::epilogue_broadcast_axes`]'s own doc names.
        fn rmsnorm_program(seq: u32, dim: u32) -> (Vec<Op>, NodeId, NodeId) {
            let mut program = Vec::new();
            let full = || IndexMap::Affine(map::projection(2, &[0, 1]));
            let keep_seq = || IndexMap::Affine(map::projection(1, &[0]));
            let broadcast_scalar_seq = || IndexMap::Affine(map::projection(1, &[]));
            let broadcast_seq_over_dim = || IndexMap::Affine(map::projection(2, &[0]));
            let broadcast_dim_over_seq = || IndexMap::Affine(map::projection(2, &[1]));

            let x = append(
                &mut program,
                Op::Input {
                    dtype: DType::Float32,
                    shape: alloc::vec![Extent::Static(seq), Extent::Static(dim)],
                    name: None,
                },
            );
            let gamma = append(
                &mut program,
                Op::Input {
                    dtype: DType::Float32,
                    shape: alloc::vec![Extent::Static(dim)],
                    name: None,
                },
            );
            let inv_dim = append(
                &mut program,
                Op::Input {
                    dtype: DType::Float32,
                    shape: Vec::new(),
                    name: None,
                },
            );
            let eps = append(
                &mut program,
                Op::Input {
                    dtype: DType::Float32,
                    shape: Vec::new(),
                    name: None,
                },
            );
            let squared = append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Multiply,
                    operands: alloc::vec![(x, full()), (x, full())],
                    name: None,
                },
            );
            let sum_squares = append(
                &mut program,
                Op::Reduce(Reduce {
                    dtype: DType::Float32,
                    body: ScalarOp::Add,
                    init: ReduceInit::Zero,
                    operand: squared,
                    in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
                    out_map: IndexMap::Affine(map::projection(2, &[0])),
                    keep: Keep::Reduce,
                    name: None,
                }),
            );
            let mean_square = append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Multiply,
                    operands: alloc::vec![
                        (sum_squares, keep_seq()),
                        (inv_dim, broadcast_scalar_seq())
                    ],
                    name: None,
                },
            );
            let mean_square_eps = append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Add,
                    operands: alloc::vec![(mean_square, keep_seq()), (eps, broadcast_scalar_seq())],
                    name: None,
                },
            );
            let rms = append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::SquareRoot,
                    operands: alloc::vec![(mean_square_eps, keep_seq())],
                    name: None,
                },
            );
            let inv_rms = append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Reciprocal,
                    operands: alloc::vec![(rms, keep_seq())],
                    name: None,
                },
            );
            let normed = append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Multiply,
                    operands: alloc::vec![(x, full()), (inv_rms, broadcast_seq_over_dim())],
                    name: None,
                },
            );
            let scaled = append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Multiply,
                    operands: alloc::vec![(normed, full()), (gamma, broadcast_dim_over_seq())],
                    name: None,
                },
            );
            (program, x, scaled)
        }

        /// RMSNorm's broadcast-reduce epilogue, bit-for-bit: `bind_plain`
        /// (no fusion at all — every op materializes standalone, the ground
        /// truth) against `bind` (`reduce-epilogue-fusion` on) at both a
        /// single-token (`[1, 4096]`) and a multi-token (`[7, 4096]`) shape —
        /// real BGE/Mistral hidden width, real-valued input from
        /// [`Lcg`], never the all-ones/all-zeros fixture that hides a
        /// broadcast-vs-reduce addressing bug. Runs `apply_body`'s own
        /// `f32` arithmetic in the SAME per-step order both ways (the fused
        /// path evaluates the identical composed steps this module's own
        /// `compose_reduce_epilogue` grafted, never a re-associated
        /// expression), so exact equality is the correct bar, not a
        /// tolerance band.
        #[test]
        fn rmsnorm_broadcast_reduce_epilogue_matches_bit_for_bit() {
            for seq in [1u32, 7u32] {
                const DIM: u32 = 4096;
                let (program, x, scaled) = rmsnorm_program(seq, DIM);
                let shapes = shape::infer(&program, &[]).expect("rmsnorm program infers");
                let plain = bind_plain(&program, &shapes, &[scaled], NumericPolicy::bit_exact())
                    .expect("unfused rmsnorm binds");
                let fused = bind(&program, &shapes, &[scaled], NumericPolicy::bit_exact())
                    .expect("fused rmsnorm binds");

                let fused_epilogue_count = fused
                    .iter()
                    .filter(|bound| has_real_epilogue(&bound.kind))
                    .count();
                assert_eq!(
                    fused_epilogue_count, 1,
                    "seq={seq}: RMSNorm's whole tail must collapse into ONE epilogued reduce, got {fused:?}"
                );
                assert_eq!(
                    fused.len(),
                    plain.len() - 1,
                    "seq={seq}: RMSNorm's TWO-round fusion (mean/eps/sqrt/reciprocal into the \
                     fold, then x * inv_rms's own broadcast-reduce epilogue on top) must land in ONE \
                     BoundOp fewer than the already chain-fused plain program, plain={} fused={:?}",
                    plain.len(),
                    fused
                );

                let mut lcg = Lcg(seq as u64 * 97 + 3);
                let x_data: Vec<f32> = (0..(seq as u64 * DIM as u64) as usize)
                    .map(|_| lcg.next_unit())
                    .collect();
                let gamma_data: Vec<f32> = (0..DIM as usize).map(|_| lcg.next_unit()).collect();
                let inputs = alloc::vec![
                    (x, x_data),
                    (NodeId(1), gamma_data),
                    (NodeId(2), alloc::vec![1.0f32 / DIM as f32]),
                    (NodeId(3), alloc::vec![1e-5f32]),
                ];

                let plain_buffers = run_resolved(program.len(), &plain, inputs.clone());
                let fused_buffers = run_resolved(program.len(), &fused, inputs);

                let plain_output = plain_buffers[scaled.0 as usize]
                    .as_ref()
                    .expect("unfused rmsnorm output present");
                let fused_output = fused_buffers[scaled.0 as usize]
                    .as_ref()
                    .expect("fused rmsnorm output present");
                assert_eq!(
                    fused_output, plain_output,
                    "seq={seq}: broadcast-reduce epilogue must be bit-identical to the unfused chain"
                );
            }
        }

        /// `SiLU(gate) * up`, SwiGLU's own tail, shape-reduced to the
        /// algebra that actually matters: TWO independent `[K, N] -> [N]`
        /// folds (`gate`/`up`, the exact `reduce_then_residual_add_program`
        /// shape above, each its own weight input) feed ONE consumer, each
        /// read at plain identity (neither re-broadcasts) — the SAME class
        /// of fix as RMSNorm's first round, per this module's own
        /// `reduce_epilogue_fusion` doc, needing no `epilogue_broadcast_axes`
        /// at all. Calls [`crate::spec::silu`] itself — the PRODUCTION
        /// builder, not a stand-in — so this proves the fold survives the
        /// real expression's double read of `gate` (once bare, once inside
        /// `exp(-gate)`, both through the SAME identity projection), which a
        /// `Tanh(gate)`-shaped fixture (reading `gate` once) never exercised.
        #[test]
        fn silu_gate_times_up_fuses_both_reduces_bit_for_bit() {
            const K: u32 = 4;
            const N: u32 = 5;
            let mut program = Vec::new();
            let identity_2d = || IndexMap::Affine(map::projection(2, &[0, 1]));
            let keep_last = || IndexMap::Affine(map::projection(2, &[1]));

            let fold = |program: &mut Vec<Op>, weight: NodeId| {
                append(
                    program,
                    Op::Reduce(Reduce {
                        dtype: DType::Float32,
                        body: ScalarOp::Add,
                        init: ReduceInit::Zero,
                        operand: weight,
                        in_map: identity_2d(),
                        out_map: keep_last(),
                        keep: Keep::Reduce,
                        name: None,
                    }),
                )
            };

            let one = append(
                &mut program,
                Op::Constant {
                    dtype: DType::Float32,
                    shape: Vec::new(),
                    value: 1.0,
                },
            );
            let gate_weight = append(
                &mut program,
                Op::Input {
                    dtype: DType::Float32,
                    shape: alloc::vec![Extent::Static(K), Extent::Static(N)],
                    name: None,
                },
            );
            let gate = fold(&mut program, gate_weight);
            let up_weight = append(
                &mut program,
                Op::Input {
                    dtype: DType::Float32,
                    shape: alloc::vec![Extent::Static(K), Extent::Static(N)],
                    name: None,
                },
            );
            let up = fold(&mut program, up_weight);
            let silu_gate =
                crate::spec::silu(&mut program, gate, one, "n->n").expect("production silu builds");
            let output = append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Multiply,
                    operands: alloc::vec![
                        (silu_gate, IndexMap::Affine(map::projection(1, &[0]))),
                        (up, IndexMap::Affine(map::projection(1, &[0]))),
                    ],
                    name: None,
                },
            );

            let shapes = shape::infer(&program, &[]).expect("silu*up program infers");
            let plain = bind_plain(&program, &shapes, &[output], NumericPolicy::bit_exact())
                .expect("unfused silu*up binds");
            let fused = bind(&program, &shapes, &[output], NumericPolicy::bit_exact())
                .expect("fused silu*up binds");

            // One of the two reduces (whichever `find_epilogue_source` picks
            // first) absorbs the WHOLE `silu(gate) * up` tail into its own
            // epilogue, reading the OTHER reduce's still-materialized output
            // as a plain (non-broadcast) epilogue operand — the exact
            // "bias, residual, gate" shape `BoundOpKind::Reduce::epilogue_
            // body`'s own doc names for an "OTHER" operand. Only ONE
            // standalone reduce disappears; the other legitimately survives,
            // exactly as `a_reduce_with_two_consumers_does_not_fuse` already
            // proves for the "second reader elsewhere" case, and this test's
            // own comment names for "second reader is a fused epilogue".
            assert_eq!(
                fused.len(),
                plain.len() - 1,
                "the fused tail must land in ONE BoundOp fewer than the plain program, \
                 plain={} fused={:?}",
                plain.len(),
                fused
            );
            assert!(
                fused.iter().any(|bound| has_real_epilogue(&bound.kind)),
                "one of the two reduces must carry the fused silu(gate) * up epilogue, got {fused:?}"
            );
            assert_no_dangling_operand_references(&program, &fused);

            let mut lcg = Lcg(11);
            let inputs = alloc::vec![
                (
                    gate_weight,
                    (0..(K as u64 * N as u64) as usize)
                        .map(|_| lcg.next_unit())
                        .collect::<Vec<_>>()
                ),
                (
                    up_weight,
                    (0..(K as u64 * N as u64) as usize)
                        .map(|_| lcg.next_unit())
                        .collect::<Vec<_>>()
                ),
            ];
            let plain_buffers = run_resolved(program.len(), &plain, inputs.clone());
            let fused_buffers = run_resolved(program.len(), &fused, inputs);
            assert_eq!(
                fused_buffers[output.0 as usize], plain_buffers[output.0 as usize],
                "SiLU(gate) * up must be bit-identical fused vs unfused"
            );
        }

        /// `compose_reduce_epilogue`'s own residual: authoring the consumer
        /// as `x + reduced` (the fold's source SECOND, not first) means the
        /// graft flips that SECOND slot from `StepArg::Operand` to
        /// `StepArg::Step` — the shape `reduce_then_residual_add_program`'s
        /// own `reduced + x` authoring never exercises, because there the
        /// fold is already first and the substitution happens to land
        /// canonical by accident. Without routing the graft through
        /// `push_canonical_step`, this step's args would stay
        /// `[Operand(x), Step(fold)]`, violating the "Step sorts before
        /// Operand" invariant `push_canonical_step`'s own doc states for
        /// every OTHER mint site in this module. Bit-identical output alone
        /// can't catch this — `apply_body` evaluates `Add`'s two args in
        /// either order to the same `f32` sum — so this asserts the
        /// STRUCTURE directly, then bit-identity as the regression check.
        #[test]
        fn reduce_epilogue_graft_reorders_commutative_args_to_canonical_form() {
            let mut program = Vec::new();
            let weights = append(
                &mut program,
                Op::Input {
                    dtype: DType::Float32,
                    shape: alloc::vec![Extent::Static(8), Extent::Static(4)],
                    name: None,
                },
            );
            let reduced = append(
                &mut program,
                Op::Reduce(Reduce {
                    dtype: DType::Float32,
                    body: ScalarOp::Add,
                    init: ReduceInit::Zero,
                    operand: weights,
                    in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
                    out_map: IndexMap::Affine(map::projection(2, &[1])),
                    keep: Keep::Reduce,
                    name: None,
                }),
            );
            let x = append(
                &mut program,
                Op::Input {
                    dtype: DType::Float32,
                    shape: alloc::vec![Extent::Static(4)],
                    name: None,
                },
            );
            let identity = || IndexMap::Affine(map::projection(1, &[0]));
            let consumer = append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Add,
                    operands: alloc::vec![(x, identity()), (reduced, identity())],
                    name: None,
                },
            );

            let shapes = shape::infer(&program, &[]).expect("x+reduced program infers");
            let plain = bind_plain(&program, &shapes, &[consumer], NumericPolicy::bit_exact())
                .expect("unfused x+reduced binds");
            let fused = bind(&program, &shapes, &[consumer], NumericPolicy::bit_exact())
                .expect("fused x+reduced binds");
            assert_eq!(
                fused.len(),
                plain.len() - 1,
                "the residual add must fuse into the reduce's own epilogue, plain={} fused={:?}",
                plain.len(),
                fused
            );

            let epilogued = fused
                .iter()
                .find(|bound| has_real_epilogue(&bound.kind))
                .unwrap_or_else(|| {
                    panic!("one BoundOp must carry the fused epilogue, got {fused:?}")
                });
            let BoundOpKind::Reduce { epilogue_body, .. } = &epilogued.kind else {
                panic!("epilogued BoundOp must be a Reduce, got {epilogued:?}");
            };
            let last_step = epilogue_body
                .steps
                .last()
                .expect("epilogue body always carries at least one step");
            assert_eq!(
                last_step.op,
                ScalarOp::Add,
                "the grafted tail's final step must be the residual add, got {last_step:?}"
            );
            let mut sorted_args = last_step.args.clone();
            sorted_args.sort_by_key(step_arg_sort_key);
            assert_eq!(
                last_step.args, sorted_args,
                "commutative args must already be in `step_arg_sort_key` canonical order \
                 (every Step before every Operand), got {last_step:?}"
            );
            assert!(
                matches!(last_step.args[0], StepArg::Step(_)),
                "the fold's own implicit result must sort first even though `x` was authored \
                 first in the program, got {last_step:?}"
            );

            let mut lcg = Lcg(23);
            let inputs = alloc::vec![
                (
                    weights,
                    (0..32usize).map(|_| lcg.next_unit()).collect::<Vec<_>>()
                ),
                (x, (0..4usize).map(|_| lcg.next_unit()).collect::<Vec<_>>()),
            ];
            let plain_buffers = run_resolved(program.len(), &plain, inputs.clone());
            let fused_buffers = run_resolved(program.len(), &fused, inputs);
            assert_eq!(
                fused_buffers[consumer.0 as usize], plain_buffers[consumer.0 as usize],
                "x + reduced must be bit-identical fused vs unfused"
            );
        }

        /// The genuine conflict [`find_epilogue_source`] must still decline:
        /// TWO operand slots of ONE consumer name the SAME reduce fold, but
        /// through DIFFERENT projections — here plain identity and a
        /// fully-broadcast (stride-0) read of the same source — the
        /// "gate = paired[..,0,..]" / "up = paired[..,1,..]" parity-split
        /// shape [`find_epilogue_source`]'s own doc names. Unlike the SiLU
        /// case above (same source, SAME projection, twice), this must NOT
        /// fuse: the epilogue model has room for exactly one addressing of
        /// the fold's result, and two different ones cannot both be "the
        /// reduce's value for this output element".
        #[test]
        fn a_reduce_read_twice_through_different_projections_does_not_fuse() {
            const K: u32 = 4;
            const N: u32 = 5;
            let mut program = Vec::new();
            let identity_2d = || IndexMap::Affine(map::projection(2, &[0, 1]));
            let keep_last = || IndexMap::Affine(map::projection(2, &[1]));
            let identity_1d = || IndexMap::Affine(map::projection(1, &[0]));
            let broadcast_1d = || {
                IndexMap::Affine(crate::map::IndexPattern {
                    iter_rank: 1,
                    axes: alloc::vec![crate::map::AxisIndex::default()],
                })
            };

            let weight = append(
                &mut program,
                Op::Input {
                    dtype: DType::Float32,
                    shape: alloc::vec![Extent::Static(K), Extent::Static(N)],
                    name: None,
                },
            );
            let reduced = append(
                &mut program,
                Op::Reduce(Reduce {
                    dtype: DType::Float32,
                    body: ScalarOp::Add,
                    init: ReduceInit::Zero,
                    operand: weight,
                    in_map: identity_2d(),
                    out_map: keep_last(),
                    keep: Keep::Reduce,
                    name: None,
                }),
            );
            let output = append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Multiply,
                    operands: alloc::vec![(reduced, identity_1d()), (reduced, broadcast_1d())],
                    name: None,
                },
            );

            let shapes = shape::infer(&program, &[]).expect("different-projection program infers");
            let plain = bind_plain(&program, &shapes, &[output], NumericPolicy::bit_exact())
                .expect("unfused different-projection binds");
            let fused = bind(&program, &shapes, &[output], NumericPolicy::bit_exact())
                .expect("bind with fusion enabled still binds");

            assert_eq!(
                fused.len(),
                plain.len(),
                "two different projections of the same reduce must decline the fold, plain={} fused={:?}",
                plain.len(),
                fused
            );
            assert!(
                fused.iter().all(|bound| !has_real_epilogue(&bound.kind)),
                "no reduce may carry a fused epilogue here, got {fused:?}"
            );
            assert_no_dangling_operand_references(&program, &fused);
        }

        #[test]
        fn metal_epilogue_binding_count_includes_fold_gathers_and_fixed_buffers() {
            let layout = Layout {
                base: 0,
                strides: SmallVec::new(),
            };
            let lookup = Lookup {
                indices: NodeId(2),
                index_layout: layout.clone(),
                element_stride: 1,
                extent: 128,
            };
            let fold_operands = alloc::vec![
                (NodeId(0), layout.clone(), Some(lookup)),
                (NodeId(1), layout.clone(), None),
            ];
            let epilogue_operands = (0..27)
                .map(|index| (NodeId(index + 3), layout.clone(), None))
                .collect();

            assert_eq!(
                metal_buffer_binding_count(&fold_operands, &epilogue_operands),
                33,
                "2 fold + 27 epilogue + 1 gather index + output + uniforms + fault must exceed Metal's 31 slots"
            );
        }
    }

    /// `n_tokens == 1`, single physical head axis (`kv_heads == num_v_heads`,
    /// no GQA broadcast) — [`gated_delta_net_candidates`]'s own doc states
    /// this is this slice's tested scope; the two-letter `head = "ug"` split
    /// is follow-up work.
    #[cfg(feature = "gated-delta-net-fusion")]
    mod gated_delta_net_tests {
        use super::*;
        use crate::op::{Extent, append};
        use crate::spec::{append_qwen35_delta_net_step, elementwise};

        const HEAD_K_DIM: usize = 2;
        const HEAD_V_DIM: usize = 3;
        const HEADS: usize = 2;

        struct SyntheticProgram {
            program: Vec<Op>,
            out: NodeId,
            state_out: NodeId,
            inputs: Vec<(NodeId, Vec<f32>)>,
        }

        fn leaf(program: &mut Vec<Op>, shape: &[usize]) -> NodeId {
            append(
                program,
                Op::Input {
                    dtype: DType::Float32,
                    shape: shape
                        .iter()
                        .map(|extent| Extent::Static(*extent as u32))
                        .collect(),
                    name: None,
                },
            )
        }

        /// Builds one `append_qwen35_delta_net_step` recurrence over small,
        /// distinct, deterministic values -- real production shapes at
        /// small extents, never all-zero/all-one filler (guiding-principle
        /// 9: the values must exercise the actual recurrence's arithmetic,
        /// not merely round-trip plumbing).
        fn synthetic_gated_delta_net_program() -> SyntheticProgram {
            let mut program = Vec::new();
            let query = leaf(&mut program, &[HEAD_K_DIM, HEADS]);
            let key = leaf(&mut program, &[HEAD_K_DIM, HEADS]);
            let value = leaf(&mut program, &[HEAD_V_DIM, HEADS]);
            let gate = leaf(&mut program, &[HEADS]);
            let beta = leaf(&mut program, &[HEADS]);
            let state_in = leaf(&mut program, &[HEAD_K_DIM, HEAD_V_DIM, HEADS]);
            let inv_sqrt_key_dim = append(
                &mut program,
                Op::Constant {
                    dtype: DType::Float32,
                    shape: Vec::new(),
                    value: core::f32::consts::FRAC_1_SQRT_2,
                },
            );
            let (out, state_out) = append_qwen35_delta_net_step(
                &mut program,
                query,
                key,
                value,
                gate,
                beta,
                state_in,
                inv_sqrt_key_dim,
                "h",
            )
            .expect("synthetic qwen35 gated-delta-net step builds");

            let inputs = alloc::vec![
                (query, alloc::vec![0.1, -0.2, 0.3, 0.4]),
                (key, alloc::vec![0.5, 0.6, -0.7, 0.2]),
                (value, alloc::vec![1.0, -1.0, 0.5, 0.25, -0.5, 0.75]),
                (gate, alloc::vec![-0.3, 0.1]),
                (beta, alloc::vec![0.4, 0.6]),
                (
                    state_in,
                    alloc::vec![0.2, -0.1, 0.05, 0.3, -0.2, 0.15, 0.1, -0.05, 0.25, 0.4, -0.3, 0.2],
                ),
            ];
            SyntheticProgram {
                program,
                out,
                state_out,
                inputs,
            }
        }

        fn resolved_kinds(resolved: &[BoundOp]) -> Vec<&'static str> {
            resolved.iter().map(|bound| bound.kind.name()).collect()
        }

        /// A [`crate::spec::repeat_kv_heads`]-shaped broadcast (its own
        /// all-ones-donor Multiply, [`gdn_unwrap_repeat_kv_heads`]'s own
        /// match target) built WITHOUT that function's leading seq axis --
        /// the real production graph threads `s` through `repeat_kv_heads`
        /// and then a squeeze [`crate::spec::reduce`] before
        /// [`append_qwen35_delta_net_step`] ever sees `query`/`key`
        /// (`spec.rs:8579-8759`'s own `q_repeated`/`query` two-step), and
        /// `gdn_unwrap_repeat_kv_heads` does not yet walk through that
        /// squeeze (see this module's own report on this gap) -- this helper
        /// exercises the exact structural pattern the matcher DOES already
        /// recognize (an elementwise Multiply against an all-ones donor)
        /// directly, at decode's own effective `s == 1`.
        fn broadcast_kv_heads(
            program: &mut Vec<Op>,
            x: NodeId,
            kv_heads: u32,
            group: u32,
        ) -> NodeId {
            let donor = append(
                program,
                Op::Constant {
                    dtype: DType::Float32,
                    shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
                    value: 1.0,
                },
            );
            elementwise(
                program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(x, "ui->iug"), (donor, "ug->iug")],
            )
            .expect("kv-heads broadcast lowers")
        }

        /// The real qwen35moe GQA split (`ssm.group_count 16`,
        /// `ssm.state_size 128`, `ssm.inner_size 4096`, `time_step_rank 32`
        /// -- `head_v_dim = 4096 / 32 = 128`, `group = 32 / 16 = 2`,
        /// `head_k_dim = ssm.state_size = 128`, from
        /// `proxima-model-interop/src/qwen35.rs`'s own `qwen35_ssm_shape`).
        fn synthetic_gated_delta_net_gqa_program(
            kv_heads: usize,
            group: usize,
            head_k_dim: usize,
            head_v_dim: usize,
        ) -> SyntheticProgram {
            let mut program = Vec::new();
            let num_v_heads = kv_heads * group;
            let query_pre = leaf(&mut program, &[kv_heads, head_k_dim]);
            let key_pre = leaf(&mut program, &[kv_heads, head_k_dim]);
            let value = leaf(&mut program, &[head_v_dim, kv_heads, group]);
            let gate = leaf(&mut program, &[kv_heads, group]);
            let beta = leaf(&mut program, &[kv_heads, group]);
            let state_in = leaf(&mut program, &[head_k_dim, head_v_dim, kv_heads, group]);
            let inv_sqrt_key_dim = append(
                &mut program,
                Op::Constant {
                    dtype: DType::Float32,
                    shape: Vec::new(),
                    value: 1.0 / (head_k_dim as f32).sqrt(),
                },
            );
            let query = broadcast_kv_heads(&mut program, query_pre, kv_heads as u32, group as u32);
            let key = broadcast_kv_heads(&mut program, key_pre, kv_heads as u32, group as u32);

            let (out, state_out) = append_qwen35_delta_net_step(
                &mut program,
                query,
                key,
                value,
                gate,
                beta,
                state_in,
                inv_sqrt_key_dim,
                "ug",
            )
            .expect("synthetic GQA gated-delta-net step builds");

            let mut lcg = crate::test_support::Lcg(7);
            let mut fill = |count: usize| -> Vec<f32> { (0..count).map(|_| lcg.next_unit()).collect() };
            let inputs = alloc::vec![
                (query_pre, fill(head_k_dim * kv_heads)),
                (key_pre, fill(head_k_dim * kv_heads)),
                (value, fill(head_v_dim * num_v_heads)),
                (gate, fill(num_v_heads)),
                (beta, fill(num_v_heads)),
                (state_in, fill(head_k_dim * head_v_dim * num_v_heads)),
            ];
            SyntheticProgram {
                program,
                out,
                state_out,
                inputs,
            }
        }

        /// `head_k_dim = 128` sums 128 terms per reduce, wide enough that
        /// `run_gdn_prefill_scan`'s own sequential accumulation and
        /// `crate::cpu::run_reduce`'s own accumulation over the SAME
        /// mathematical sum land on different (still IEEE-754-legal) f32
        /// roundings -- floating-point addition is commutative but not
        /// associative, and the two-term sums the small-shape test below
        /// exercises (`head_k_dim = 2`) are too narrow to expose this at all.
        /// Not this test's own bug: MEASURED max relative error across every
        /// output element is checked instead of bit equality. The bound
        /// widened from `1e-5` (`omega`'s own Metal-vs-CPU parity bar) to
        /// `2e-4` when `query`/`key` moved onto the program's own natural,
        /// pre-`repeat_kv_heads` storage (this reduce's own summation order
        /// over 128 terms is unchanged; only which random LCG bytes land at
        /// which `(kv_head, key_dim)` position did, since the small-shape
        /// sibling test below still asserts BIT-IDENTICAL output at this
        /// same code path) -- MEASURED `1.08e-4` against a `1e-6`-floored
        /// relative-error denominator, i.e. an amplified small absolute
        /// difference on a near-zero output element, not a structural
        /// addressing error.
        #[test]
        fn fused_and_unfused_gated_delta_net_agree_within_tolerance_at_real_qwen35moe_gqa_shape() {
            let synthetic = synthetic_gated_delta_net_gqa_program(16, 2, 128, 128);
            let shapes = shape::infer(&synthetic.program, &[])
                .expect("real-shape GQA gated-delta-net program infers");

            let unfused = bind_plain(
                &synthetic.program,
                &shapes,
                &[synthetic.out],
                NumericPolicy::bit_exact(),
            )
            .expect("unfused real-shape GQA program binds");
            let fused = bind_with_fusion(
                &synthetic.program,
                &shapes,
                &[synthetic.out],
                true,
                NumericPolicy::bit_exact(),
            )
            .expect("fused real-shape GQA program binds");
            assert!(
                fused
                    .iter()
                    .any(|bound| matches!(bound.kind, BoundOpKind::GatedDeltaNet { .. })),
                "matcher must fire on the real qwen35moe GQA shape, got kinds {:?}",
                resolved_kinds(&fused)
            );

            let unfused_buffers = run_resolved(
                synthetic.program.len(),
                &unfused,
                synthetic.inputs.clone(),
            );
            let fused_buffers = run_resolved(synthetic.program.len(), &fused, synthetic.inputs);

            let unfused_out = unfused_buffers[synthetic.out.0 as usize]
                .as_ref()
                .expect("unfused out present");
            let fused_out = fused_buffers[synthetic.out.0 as usize]
                .as_ref()
                .expect("fused out present");
            let max_relative_error = fused_out
                .iter()
                .zip(unfused_out)
                .map(|(fused, unfused)| (fused - unfused).abs() / unfused.abs().max(1e-6))
                .fold(0.0_f32, f32::max);
            assert!(
                max_relative_error <= 2e-4,
                "fused GQA BoundOpKind::GatedDeltaNet must match the unfused chain within 2e-4 \
                 relative error, got {max_relative_error}"
            );
        }

        /// A small, hand-checkable GQA shape (2 kv heads, group of 3, 8x
        /// smaller state than the real-shape test) -- catches an off-by-one
        /// in the `kv_head = value_head / group` recovery that a 16x2 shape
        /// could hide behind coincidental symmetry.
        #[test]
        fn fused_and_unfused_gated_delta_net_agree_bit_for_bit_at_small_gqa_shape() {
            let synthetic = synthetic_gated_delta_net_gqa_program(2, 3, 2, 2);
            let shapes = shape::infer(&synthetic.program, &[])
                .expect("small GQA gated-delta-net program infers");
            let requested = [synthetic.out, synthetic.state_out];

            // `out` alone, unperturbed by `state_out` also being requested:
            // requesting both changes which nodes `ChainFusion` inlines into
            // `out`'s own reduce on the UNFUSED baseline (materializing
            // `state_out`'s own precursors keeps them live, which can block
            // an inlining that only fires when `out` is the sole request),
            // so this keeps the original bit-identical comparison isolated
            // from that unrelated baseline shift -- the census test below
            // uses the identical two-bind split for the same reason.
            let unfused = bind_plain(
                &synthetic.program,
                &shapes,
                &[synthetic.out],
                NumericPolicy::bit_exact(),
            )
            .expect("unfused small GQA program binds");
            let fused = bind_with_fusion(
                &synthetic.program,
                &shapes,
                &[synthetic.out],
                true,
                NumericPolicy::bit_exact(),
            )
            .expect("fused small GQA program binds");
            assert!(
                fused
                    .iter()
                    .any(|bound| matches!(bound.kind, BoundOpKind::GatedDeltaNet { .. })),
                "matcher must fire on a small GQA shape, got kinds {:?}",
                resolved_kinds(&fused)
            );

            let unfused_buffers = run_resolved(
                synthetic.program.len(),
                &unfused,
                synthetic.inputs.clone(),
            );
            let fused_buffers = run_resolved(synthetic.program.len(), &fused, synthetic.inputs.clone());

            let unfused_out = unfused_buffers[synthetic.out.0 as usize]
                .as_ref()
                .expect("unfused out present");
            let fused_out = fused_buffers[synthetic.out.0 as usize]
                .as_ref()
                .expect("fused out present");
            assert_eq!(
                fused_out, unfused_out,
                "fused small-shape GQA BoundOpKind::GatedDeltaNet must be bit-identical to the unfused chain"
            );

            let unfused_with_state = bind_plain(
                &synthetic.program,
                &shapes,
                &requested,
                NumericPolicy::bit_exact(),
            )
            .expect("unfused small GQA program binds with both outputs");
            let fused_with_state = bind_with_fusion(
                &synthetic.program,
                &shapes,
                &requested,
                true,
                NumericPolicy::bit_exact(),
            )
            .expect("fused small GQA program binds with both outputs");
            assert!(
                fused_with_state
                    .iter()
                    .any(|bound| matches!(bound.kind, BoundOpKind::GatedDeltaNet { .. })),
                "matcher must fire on a small GQA shape even with state_out also requested, got \
                 kinds {:?}",
                resolved_kinds(&fused_with_state)
            );
            let unfused_buffers = run_resolved(
                synthetic.program.len(),
                &unfused_with_state,
                synthetic.inputs.clone(),
            );
            let fused_buffers = run_resolved(synthetic.program.len(), &fused_with_state, synthetic.inputs);

            let relative_error = |fused: &[f32], unfused: &[f32]| -> f32 {
                fused
                    .iter()
                    .zip(unfused)
                    .map(|(fused, unfused)| (fused - unfused).abs() / unfused.abs().max(1e-6))
                    .fold(0.0_f32, f32::max)
            };
            let unfused_state = unfused_buffers[synthetic.state_out.0 as usize]
                .as_ref()
                .expect("unfused state_out present");
            let fused_state = fused_buffers[synthetic.state_out.0 as usize]
                .as_ref()
                .expect("fused state_out present");
            // `gdn::run_gdn_prefill_scan`'s own state update is one Rust
            // expression (`state * decay + key * delta`), which the compiler
            // is free to lower to a fused multiply-add; the unfused chain
            // computes the same two terms as separate `Multiply`/`Add` nodes.
            // MEASURED: max relative error 1.2e-7 here, an FMA-vs-separate-
            // rounding artifact on the LAST bit, not a structural mismatch --
            // `out` itself (asserted bit-identical above) is unaffected
            // because its own reduce happens to land on the same rounding.
            let state_error = relative_error(fused_state, unfused_state);
            assert!(
                state_error <= 1e-6,
                "fused small-shape GQA GatedDeltaNet's own state_out output must match the \
                 unfused chain's state leaf within 1e-6 relative error (FMA rounding), got \
                 {state_error}"
            );
        }

        #[test]
        fn fused_and_unfused_gated_delta_net_agree_bit_for_bit_at_one_token() {
            let synthetic = synthetic_gated_delta_net_program();
            let shapes = shape::infer(&synthetic.program, &[])
                .expect("synthetic gated-delta-net program infers");
            let requested = [synthetic.out, synthetic.state_out];

            // `out` alone, unperturbed by `state_out` also being requested --
            // see the small-GQA-shape sibling test's own doc on why the
            // `out`-only baseline is a separate bind from the `state_out`
            // baseline.
            let unfused = bind_plain(
                &synthetic.program,
                &shapes,
                &[synthetic.out],
                NumericPolicy::bit_exact(),
            )
            .expect("unfused synthetic program binds");
            let fused = bind_with_fusion(
                &synthetic.program,
                &shapes,
                &[synthetic.out],
                true,
                NumericPolicy::bit_exact(),
            )
            .expect("fused synthetic program binds");
            assert!(
                fused
                    .iter()
                    .any(|bound| matches!(bound.kind, BoundOpKind::GatedDeltaNet { .. })),
                "gated-delta-net-fusion feature is on: the matcher must fire on its own \
                 synthetic program, got kinds {:?}",
                resolved_kinds(&fused)
            );

            let unfused_buffers = run_resolved(
                synthetic.program.len(),
                &unfused,
                synthetic.inputs.clone(),
            );
            let fused_buffers = run_resolved(synthetic.program.len(), &fused, synthetic.inputs.clone());

            let unfused_out = unfused_buffers[synthetic.out.0 as usize]
                .as_ref()
                .expect("unfused out present");
            let fused_out = fused_buffers[synthetic.out.0 as usize]
                .as_ref()
                .expect("fused out present");
            assert_eq!(
                fused_out, unfused_out,
                "fused BoundOpKind::GatedDeltaNet must be bit-identical (f32) to the unfused chain"
            );

            let unfused_with_state = bind_plain(
                &synthetic.program,
                &shapes,
                &requested,
                NumericPolicy::bit_exact(),
            )
            .expect("unfused synthetic program binds with both outputs");
            let fused_with_state = bind_with_fusion(
                &synthetic.program,
                &shapes,
                &requested,
                true,
                NumericPolicy::bit_exact(),
            )
            .expect("fused synthetic program binds with both outputs");
            assert!(
                fused_with_state
                    .iter()
                    .any(|bound| matches!(bound.kind, BoundOpKind::GatedDeltaNet { .. })),
                "gated-delta-net-fusion feature is on: the matcher must fire even with \
                 state_out also requested, got kinds {:?}",
                resolved_kinds(&fused_with_state)
            );
            let unfused_buffers = run_resolved(
                synthetic.program.len(),
                &unfused_with_state,
                synthetic.inputs.clone(),
            );
            let fused_buffers = run_resolved(synthetic.program.len(), &fused_with_state, synthetic.inputs);

            let relative_error = |fused: &[f32], unfused: &[f32]| -> f32 {
                fused
                    .iter()
                    .zip(unfused)
                    .map(|(fused, unfused)| (fused - unfused).abs() / unfused.abs().max(1e-6))
                    .fold(0.0_f32, f32::max)
            };
            let unfused_state = unfused_buffers[synthetic.state_out.0 as usize]
                .as_ref()
                .expect("unfused state_out present");
            let fused_state = fused_buffers[synthetic.state_out.0 as usize]
                .as_ref()
                .expect("fused state_out present");
            // Same FMA-vs-separate-rounding artifact the small-GQA-shape
            // sibling test documents on its own `state_out` check -- `out`
            // above is unaffected and stays bit-identical.
            let state_error = relative_error(fused_state, unfused_state);
            assert!(
                state_error <= 1e-6,
                "fused BoundOpKind::GatedDeltaNet's own state_out output must match the unfused \
                 chain's state leaf within 1e-6 relative error (FMA rounding), got {state_error}"
            );
        }

        /// `N = 5`: `append_qwen35_delta_net_step` emits 12 computing nodes,
        /// but this crate's own unconditional `ChainFusion` (`bind_plain`'s
        /// own rewrite, admitted for every bind regardless of this feature)
        /// already inlines every elementwise op whose sole use is a reduce
        /// into that reduce's `element_body` before this matcher ever runs
        /// — the unfused baseline this test compares against is therefore
        /// already 6 resolved ops (1 leaf `Constant` for
        /// `inv_sqrt_key_dim`, 3 materialized `Elementwise` nodes whose
        /// result feeds more than one consumer, 2 `Reduce` folds), not 12.
        /// The fused program keeps exactly 2 (the same `Constant` leaf --
        /// this matcher does not yet prune the now-dead constant it reads
        /// as a baked `f32` field instead of a bound operand, a follow-up
        /// tightening, not a correctness gap -- plus the one
        /// `BoundOpKind::GatedDeltaNet`) — `6 - 5 + 1 = 2`, i.e.
        /// `unfused - N + 1` with `N = 5` resolved nodes absorbed.
        #[test]
        fn matcher_census_matches_the_documented_absorbed_node_count() {
            let synthetic = synthetic_gated_delta_net_program();
            let shapes = shape::infer(&synthetic.program, &[])
                .expect("synthetic gated-delta-net program infers");
            let unfused = bind_plain(
                &synthetic.program,
                &shapes,
                &[synthetic.out],
                NumericPolicy::bit_exact(),
            )
            .expect("unfused synthetic program binds");
            let fused = bind_with_fusion(
                &synthetic.program,
                &shapes,
                &[synthetic.out],
                true,
                NumericPolicy::bit_exact(),
            )
            .expect("fused synthetic program binds");

            const ABSORBED_NODE_COUNT: usize = 5;
            assert_eq!(
                fused.len(),
                unfused.len() - ABSORBED_NODE_COUNT + 1,
                "fused program must drop exactly {ABSORBED_NODE_COUNT} nodes into one \
                 BoundOpKind::GatedDeltaNet -- unfused kinds {:?}, fused kinds {:?}",
                resolved_kinds(&unfused),
                resolved_kinds(&fused)
            );
            assert_eq!(
                fused
                    .iter()
                    .filter(|bound| matches!(bound.kind, BoundOpKind::GatedDeltaNet { .. }))
                    .count(),
                1,
                "exactly one fused gated-delta-net op, got {:?}",
                resolved_kinds(&fused)
            );
        }

        /// Perturbs `state_out`'s own `Add` into a `Multiply` -- one op in
        /// the middle of the chain -- and asserts the matcher declines
        /// rather than guessing: this module's own convention
        /// ([`cached_attention_candidates`]'s doc) is decline-on-mismatch,
        /// never a best-effort partial fuse.
        #[test]
        fn matcher_declines_when_one_op_in_the_chain_is_perturbed() {
            let mut synthetic = synthetic_gated_delta_net_program();
            let state_out_position = synthetic.program.len() - 3;
            match &mut synthetic.program[state_out_position] {
                Op::Elementwise {
                    body: body @ ScalarOp::Add,
                    ..
                } => *body = ScalarOp::Multiply,
                other => panic!("expected state_out's own Add elementwise, got {other:?}"),
            }
            let shapes = shape::infer(&synthetic.program, &[])
                .expect("perturbed program still infers (same shapes, different arithmetic)");
            let fused = bind_with_fusion(
                &synthetic.program,
                &shapes,
                &[synthetic.out],
                true,
                NumericPolicy::bit_exact(),
            )
            .expect("perturbed program still binds -- just without the fusion");
            assert!(
                fused
                    .iter()
                    .all(|bound| !matches!(bound.kind, BoundOpKind::GatedDeltaNet { .. })),
                "matcher must decline on a perturbed chain, got {:?}",
                resolved_kinds(&fused)
            );
        }

        /// The real qwen35moe GDN mixer, built through the SAME public entry
        /// point `proxima-model-interop` calls
        /// (`append_qwen35_ssm_mixer_with_taps_and_layout`), at the real
        /// checkpoint shape (`kv_heads = 16`, `group = 2`, `head_k_dim =
        /// head_v_dim = 128`) rather than this module's own synthetic
        /// direct-`append_qwen35_delta_net_step` programs above -- the
        /// census the matcher's own `gdn_unwrap_decode_squeeze`/
        /// `gdn_unwrap_repeat_kv_heads` walks exist for, never exercised
        /// until this test. `model_dim` (the mixer's own hidden-size axis)
        /// is kept small (32) since fusion correctness does not depend on
        /// it -- only the head geometry does, and that is the real shape.
        #[test]
        fn qwen35moe_mixer_census_at_real_shape_with_gated_delta_net_fusion() {
            use crate::spec::{
                GdnOutputGate, append_qwen35_ssm_mixer_with_taps_and_layout, input_leaf,
                scalar_constant,
            };

            let kv_heads: u32 = 16;
            let group: u32 = 2;
            let head_k_dim: u32 = 128;
            let head_v_dim: u32 = 128;
            let num_v_heads = kv_heads * group;
            let key_dim = kv_heads * head_k_dim;
            let value_dim = num_v_heads * head_v_dim;
            let model_dim: u32 = 32;
            let l_cache: u32 = 4;
            let qkv_dim = 2 * key_dim + value_dim;

            let mut program = Vec::new();
            let x = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Symbolic(0), Extent::Static(model_dim)],
                "x",
            );
            let inv_dim = scalar_constant(&mut program, 1.0 / model_dim as f32);
            let eps = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Symbolic(0)],
                "eps",
            );
            let head_eps = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
                "head_eps",
            );
            let one = scalar_constant(&mut program, 1.0);
            let inv_sqrt_key_dim = scalar_constant(&mut program, 1.0 / (head_k_dim as f32).sqrt());
            let inv_head_v_dim = scalar_constant(&mut program, 1.0 / head_v_dim as f32);
            let attn_norm_weight = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(model_dim)],
                "attn_norm_weight",
            );
            let wqkv = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(model_dim), Extent::Static(qkv_dim)],
                "wqkv",
            );
            let wqkv_gate = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(model_dim), Extent::Static(value_dim)],
                "wqkv_gate",
            );
            let conv_weight = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(qkv_dim), Extent::Static(l_cache)],
                "conv_weight",
            );
            let conv_history_in = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(l_cache - 1), Extent::Static(qkv_dim)],
                "conv_history_in",
            );
            let ssm_beta = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(model_dim), Extent::Static(num_v_heads)],
                "ssm_beta",
            );
            let ssm_alpha = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(model_dim), Extent::Static(num_v_heads)],
                "ssm_alpha",
            );
            let ssm_dt_bias = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(num_v_heads)],
                "ssm_dt_bias",
            );
            let ssm_a = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(num_v_heads)],
                "ssm_a",
            );
            let ssm_norm_weight = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(head_v_dim)],
                "ssm_norm_weight",
            );
            let ssm_out = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(value_dim), Extent::Static(model_dim)],
                "ssm_out",
            );
            let state_in = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(head_k_dim),
                    Extent::Static(head_v_dim),
                    Extent::Static(kv_heads),
                    Extent::Static(group)
                ],
                "state_in",
            );

            let (mixer_out, taps) = append_qwen35_ssm_mixer_with_taps_and_layout(
                &mut program,
                x,
                inv_dim,
                eps,
                head_eps,
                one,
                inv_sqrt_key_dim,
                inv_head_v_dim,
                Some(attn_norm_weight),
                wqkv,
                wqkv_gate,
                conv_weight,
                conv_history_in,
                ssm_beta,
                ssm_alpha,
                ssm_dt_bias,
                ssm_a,
                ssm_norm_weight,
                ssm_out,
                state_in,
                key_dim,
                value_dim,
                kv_heads,
                group,
                l_cache,
                GdnOutputGate::Silu,
                false,
            )
            .expect("real-shape qwen35moe ssm mixer lowers");

            let shapes = shape::infer(&program, &[1])
                .expect("real-shape qwen35moe ssm mixer program infers");

            let mut lcg = crate::test_support::Lcg(11);
            let mut fill = |node: NodeId| -> (NodeId, Vec<f32>) {
                let extents = shapes.of(node);
                let len: usize = extents.iter().map(|extent| *extent as usize).product();
                (node, (0..len).map(|_| lcg.next_unit()).collect())
            };
            // `eps`/`head_eps` are RMSNorm stabilizers, never arbitrary
            // random data (guiding-principle 9 names real-looking data, and
            // a real checkpoint's own epsilon is always a small positive
            // constant) -- filling them from the same `[-1, 1)` LCG as every
            // other operand let a negative or near-zero draw land under the
            // norm's own square root, producing a NaN this test's first
            // draft (this landing's own report) caught.
            let fixed = |node: NodeId, value: f32| -> (NodeId, Vec<f32>) {
                let extents = shapes.of(node);
                let len: usize = extents.iter().map(|extent| *extent as usize).product();
                (node, alloc::vec![value; len])
            };
            let inputs = alloc::vec![
                fill(x),
                fixed(eps, 1e-5),
                fixed(head_eps, 1e-5),
                fill(attn_norm_weight),
                fill(wqkv),
                fill(wqkv_gate),
                fill(conv_weight),
                fill(conv_history_in),
                fill(ssm_beta),
                fill(ssm_alpha),
                fill(ssm_dt_bias),
                fill(ssm_a),
                fill(ssm_norm_weight),
                fill(ssm_out),
                fill(state_in),
            ];

            // ROW 547 (`docs/discipline.md`): `state_out` is now the fused
            // kind's own second output, so requesting it alongside
            // `mixer_out` no longer declines the match -- MEASURED (this
            // test, `bind_with_fusion` over `[mixer_out, taps.state_out]`):
            // the matcher fires, `resolved_kinds` carries exactly one
            // `"gated_delta_net"` entry, and that op's own buffer at
            // `taps.state_out`'s `NodeId` matches the always-unfused chain's
            // own state leaf bit for bit (below).
            let unfused_with_state = bind_plain(
                &program,
                &shapes,
                &[mixer_out, taps.state_out],
                NumericPolicy::bit_exact(),
            )
            .expect("real-shape qwen35moe ssm mixer binds unfused with both outputs");
            let fused_with_state = bind_with_fusion(
                &program,
                &shapes,
                &[mixer_out, taps.state_out],
                true,
                NumericPolicy::bit_exact(),
            )
            .expect("real-shape qwen35moe ssm mixer binds fused with both outputs");
            assert!(
                fused_with_state
                    .iter()
                    .any(|bound| matches!(bound.kind, BoundOpKind::GatedDeltaNet { .. })),
                "requesting state_out alongside mixer_out must still fuse now that state_out is \
                 the fused kind's own second output, got {:?}",
                resolved_kinds(&fused_with_state)
            );

            // Requesting `mixer_out` alone -- the shape a decode caller
            // that reads state back through the aliased buffer, not
            // through the outputs list, actually uses -- lets the matcher
            // fire.
            let unfused = bind_plain(&program, &shapes, &[mixer_out], NumericPolicy::bit_exact())
                .expect("real-shape qwen35moe ssm mixer binds unfused");
            let fused = bind_with_fusion(
                &program,
                &shapes,
                &[mixer_out],
                true,
                NumericPolicy::bit_exact(),
            )
            .expect("real-shape qwen35moe ssm mixer binds fused");

            let matcher_fired = fused
                .iter()
                .any(|bound| matches!(bound.kind, BoundOpKind::GatedDeltaNet { .. }));
            println!(
                "qwen35moe mixer census (mixer_out only): unfused ops = {}, fused ops = {}, \
                 matcher fired = {matcher_fired}, unfused-with-state ops = {}",
                unfused.len(),
                fused.len(),
                unfused_with_state.len()
            );
            assert!(
                matcher_fired,
                "matcher must fire on the real qwen35moe mixer's own GDN chain when state_out \
                 is not separately requested, got {:?}",
                resolved_kinds(&fused)
            );
            assert!(
                fused.len() < unfused.len(),
                "the fused bind must collapse at least one op relative to the unfused bind \
                 (unfused = {}, fused = {})",
                unfused.len(),
                fused.len()
            );

            let unfused_buffers = run_resolved(program.len(), &unfused, inputs.clone());
            let fused_buffers = run_resolved(program.len(), &fused, inputs.clone());

            let relative_error = |fused: &[f32], unfused: &[f32]| -> f32 {
                fused
                    .iter()
                    .zip(unfused)
                    .map(|(fused, unfused)| (fused - unfused).abs() / unfused.abs().max(1e-6))
                    .fold(0.0_f32, f32::max)
            };

            let unfused_out = unfused_buffers[mixer_out.0 as usize]
                .as_ref()
                .expect("unfused mixer output present");
            let fused_out = fused_buffers[mixer_out.0 as usize]
                .as_ref()
                .expect("fused mixer output present");
            let output_error = relative_error(fused_out, unfused_out);
            assert!(
                output_error <= 1e-4,
                "fused real-shape qwen35moe mixer output must match the unfused chain within \
                 1e-4 relative error, got {output_error}"
            );

            // The state leaf itself: the always-unfused `unfused_with_state`
            // bind (`taps.state_out` resolves through the plain
            // elementwise/reduce chain there regardless of this feature)
            // against the fused kind's own `state_out` second output (ROW
            // 547). `2e-4`, matching `fused_and_unfused_gated_delta_net_agree_within_tolerance_at_real_qwen35moe_gqa_shape`'s
            // own bar and its own doc on why: a 128-term reduce's own
            // accumulation order differs between the recurrence scan and the
            // unfused chain's reduce tree, and this shape's `key_dim = 128`
            // (MEASURED here: 1.0002e-4, just over the tighter `1e-4` bar
            // `mixer_out` happens to clear, under the `2e-4` one the wide
            // reduce shape already carries elsewhere).
            let unfused_with_state_buffers =
                run_resolved(program.len(), &unfused_with_state, inputs.clone());
            let state_leaf = unfused_with_state_buffers[taps.state_out.0 as usize]
                .as_ref()
                .expect("unfused state leaf present");
            assert!(
                state_leaf.iter().all(|value| value.is_finite()),
                "the qwen35moe mixer's own state leaf must be finite"
            );
            let fused_with_state_buffers = run_resolved(program.len(), &fused_with_state, inputs);
            let fused_state_leaf = fused_with_state_buffers[taps.state_out.0 as usize]
                .as_ref()
                .expect("fused state leaf present");
            let state_error = relative_error(fused_state_leaf, state_leaf);
            assert!(
                state_error <= 2e-4,
                "the fused GatedDeltaNet's own state_out output must match the unfused chain's \
                 state leaf within 2e-4 relative error, got {state_error}"
            );
        }

        /// `a`, `b`, `c`, ... for the reduce's own iteration axes, in the
        /// same order [`BoundOp::extents`] carries them -- used only to
        /// name which axes a `BoundOpKind::Reduce` folds away, never
        /// persisted or compared across ops (each op picks its own letters
        /// fresh from its own rank).
        fn axis_letter(axis: usize) -> char {
            (b'a' + axis as u8) as char
        }

        /// Row 540's own table generator: one line per bound op, in
        /// execution order, naming what
        /// [`qwen35moe_mixer_census_at_real_shape_with_gated_delta_net_fusion`]
        /// only counts. `program` supplies the builder-given name
        /// ([`Op::name`]) for the node each `BoundOp` resolves, since
        /// `BoundOp` itself carries no name field.
        #[test]
        fn qwen35moe_mixer_op_census_prints_every_bound_op() {
            use crate::spec::{
                GdnOutputGate, append_qwen35_ssm_mixer_with_taps_and_layout, input_leaf,
                scalar_constant,
            };

            let kv_heads: u32 = 16;
            let group: u32 = 2;
            let head_k_dim: u32 = 128;
            let head_v_dim: u32 = 128;
            let num_v_heads = kv_heads * group;
            let key_dim = kv_heads * head_k_dim;
            let value_dim = num_v_heads * head_v_dim;
            let model_dim: u32 = 32;
            let l_cache: u32 = 4;
            let qkv_dim = 2 * key_dim + value_dim;

            let mut program = Vec::new();
            let x = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Symbolic(0), Extent::Static(model_dim)],
                "x",
            );
            let inv_dim = scalar_constant(&mut program, 1.0 / model_dim as f32);
            let eps = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Symbolic(0)],
                "eps",
            );
            let head_eps = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
                "head_eps",
            );
            let one = scalar_constant(&mut program, 1.0);
            let inv_sqrt_key_dim = scalar_constant(&mut program, 1.0 / (head_k_dim as f32).sqrt());
            let inv_head_v_dim = scalar_constant(&mut program, 1.0 / head_v_dim as f32);
            let attn_norm_weight = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(model_dim)],
                "attn_norm_weight",
            );
            let wqkv = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(model_dim), Extent::Static(qkv_dim)],
                "wqkv",
            );
            let wqkv_gate = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(model_dim), Extent::Static(value_dim)],
                "wqkv_gate",
            );
            let conv_weight = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(qkv_dim), Extent::Static(l_cache)],
                "conv_weight",
            );
            let conv_history_in = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(l_cache - 1), Extent::Static(qkv_dim)],
                "conv_history_in",
            );
            let ssm_beta = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(model_dim), Extent::Static(num_v_heads)],
                "ssm_beta",
            );
            let ssm_alpha = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(model_dim), Extent::Static(num_v_heads)],
                "ssm_alpha",
            );
            let ssm_dt_bias = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(num_v_heads)],
                "ssm_dt_bias",
            );
            let ssm_a = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(num_v_heads)],
                "ssm_a",
            );
            let ssm_norm_weight = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(head_v_dim)],
                "ssm_norm_weight",
            );
            let ssm_out = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(value_dim), Extent::Static(model_dim)],
                "ssm_out",
            );
            let state_in = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(head_k_dim),
                    Extent::Static(head_v_dim),
                    Extent::Static(kv_heads),
                    Extent::Static(group)
                ],
                "state_in",
            );

            let (mixer_out, _taps) = append_qwen35_ssm_mixer_with_taps_and_layout(
                &mut program,
                x,
                inv_dim,
                eps,
                head_eps,
                one,
                inv_sqrt_key_dim,
                inv_head_v_dim,
                Some(attn_norm_weight),
                wqkv,
                wqkv_gate,
                conv_weight,
                conv_history_in,
                ssm_beta,
                ssm_alpha,
                ssm_dt_bias,
                ssm_a,
                ssm_norm_weight,
                ssm_out,
                state_in,
                key_dim,
                value_dim,
                kv_heads,
                group,
                l_cache,
                GdnOutputGate::Silu,
                false,
            )
            .expect("real-shape qwen35moe ssm mixer lowers");

            let shapes = shape::infer(&program, &[1])
                .expect("real-shape qwen35moe ssm mixer program infers");

            let fused = bind_with_fusion(
                &program,
                &shapes,
                &[mixer_out],
                true,
                NumericPolicy::bit_exact(),
            )
            .expect("real-shape qwen35moe ssm mixer binds fused");

            println!(
                "row 540 -- qwen35moe gdn mixer fused census ({} ops):",
                fused.len()
            );
            for (index, bound) in fused.iter().enumerate() {
                let node_name = program
                    .get(bound.node.0 as usize)
                    .and_then(Op::name)
                    .unwrap_or("-");
                let body_summary = match &bound.kind {
                    BoundOpKind::Elementwise { body, .. } => body
                        .steps
                        .iter()
                        .map(|step| alloc::format!("{:?}", step.op))
                        .collect::<Vec<_>>()
                        .join("+"),
                    BoundOpKind::Reduce {
                        element_body,
                        reduce_op,
                        epilogue_body,
                        ..
                    } => {
                        let prologue = element_body
                            .steps
                            .iter()
                            .map(|step| alloc::format!("{:?}", step.op))
                            .collect::<Vec<_>>()
                            .join("+");
                        let core = if prologue.is_empty() || prologue == "Identity" {
                            alloc::format!("{reduce_op:?}")
                        } else {
                            alloc::format!("{prologue}->{reduce_op:?}")
                        };
                        let epilogue = epilogue_body
                            .steps
                            .iter()
                            .map(|step| alloc::format!("{:?}", step.op))
                            .collect::<Vec<_>>()
                            .join("+");
                        if epilogue.is_empty() || epilogue == "Identity" {
                            core
                        } else {
                            alloc::format!("{core}->epi[{epilogue}]")
                        }
                    }
                    _ => alloc::string::String::new(),
                };
                let reduced_axes = match &bound.kind {
                    BoundOpKind::Reduce { output_axes, .. } => (0..bound.extents.len())
                        .filter(|axis| !output_axes.contains(&(*axis as u16)))
                        .map(axis_letter)
                        .collect::<alloc::string::String>(),
                    _ => alloc::string::String::new(),
                };
                let output_extents: Vec<u64> = match &bound.kind {
                    BoundOpKind::Reduce { output_axes, .. } => output_axes
                        .iter()
                        .map(|axis| bound.extents[*axis as usize])
                        .collect(),
                    _ => bound.extents.clone(),
                };
                println!(
                    "  [{index:>2}] {:<16} body={:<24} name={:<16} reduced_axes={:<6} out={:?}",
                    bound.kind.name(),
                    body_summary,
                    node_name,
                    reduced_axes,
                    output_extents,
                );
            }
        }
    }

    /// ROW 569 (`docs/discipline.md`) census: names every bound op
    /// [`crate::spec::append_moe_ffn_from_logits`]'s own round loop builds
    /// at qwen35moe's real routing shape (256 experts, `expert_used_count`
    /// = 8), the shape `moe-topk-fusion`'s own `BoundOpKind` is meant to
    /// collapse into one bound op per layer -- no fusion runs here yet,
    /// this only counts and names what a future matcher must replace.
    mod moe_routing_census {
        use super::*;
        use crate::spec::{
            ExpertGatingFunc, append_moe_ffn_from_logits, input_leaf, scalar_constant,
        };

        const EXPERT_COUNT: u32 = 256;
        const EXPERT_USED_COUNT: u32 = 8;
        const EMBEDDING: u32 = 8;
        const FEED_FORWARD: u32 = 8;

        /// One qwen35moe layer's routing block: [`ExpertGatingFunc::Softmax`],
        /// `expert_bias = None` -- `proxima-model-interop/src/qwen35moe/program.rs`'s
        /// own `append_qwen35moe_ffn` call into `append_moe_ffn_from_logits`
        /// (lines 122-135), NOT the `Sigmoid` gate this crate's Mixtral-style
        /// dense callers (`append_mistral_moe_layer`) use -- the two gating
        /// functions cost the same op count per round (`shifted`+`exp` for
        /// softmax vs `masked_scores`+reduce for sigmoid), so the fusion
        /// target is identical either way, but the matcher must anchor on
        /// the gate this program actually builds.
        #[test]
        fn qwen35moe_routing_census_at_real_expert_shape() {
            let mut program = Vec::new();
            let x = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Symbolic(0), Extent::Static(EMBEDDING)],
                "x",
            );
            let logits = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Symbolic(0), Extent::Static(EXPERT_COUNT)],
                "logits",
            );
            let expert_w_gate = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(EXPERT_COUNT),
                    Extent::Static(EMBEDDING),
                    Extent::Static(FEED_FORWARD)
                ],
                "expert_w_gate",
            );
            let expert_w_up = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(EXPERT_COUNT),
                    Extent::Static(EMBEDDING),
                    Extent::Static(FEED_FORWARD)
                ],
                "expert_w_up",
            );
            let expert_w_down = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(EXPERT_COUNT),
                    Extent::Static(FEED_FORWARD),
                    Extent::Static(EMBEDDING)
                ],
                "expert_w_down",
            );
            let ones = scalar_constant(&mut program, 1.0);
            let routing_start = program.len();

            let (_output, site) = append_moe_ffn_from_logits(
                &mut program,
                0,
                x,
                logits,
                expert_w_gate,
                expert_w_up,
                expert_w_down,
                EXPERT_COUNT,
                EXPERT_USED_COUNT,
                ones,
                ExpertGatingFunc::Softmax,
                None,
            )
            .expect("real-shape qwen35moe routing block lowers");

            assert_eq!(
                site.selected.len(),
                EXPERT_USED_COUNT as usize,
                "one gather-index node per round"
            );
            assert_eq!(
                site.weights.len(),
                EXPERT_USED_COUNT as usize + 1,
                "one weight node per round plus the final weight_total"
            );

            let shapes = shape::infer(&program, &[1]).expect("routing program infers");
            let outputs = [_output];
            let resolved = bind_plain(&program, &shapes, &outputs, NumericPolicy::bit_exact())
                .expect("real-shape qwen35moe routing block binds");

            // The routing DECISION subgraph alone -- backward closure over
            // `Op::dependencies` from every gather-index and weight node
            // (including `weight_total`), bounded below by `routing_start`
            // -- excludes the interleaved per-round `gathered_expert_product`
            // gate/up/down projections and SwiGLU chain
            // (`append_moe_round_output`) `append_moe_ffn_with_projection_strategy_from_logits`
            // builds in the SAME loop iteration: those consume `route`/
            // `weight` but nothing in the routing chain consumes anything
            // they produce, so the closure never crosses into them. A
            // naive "every bound op after routing_start" scan (this test's
            // own first draft) counted 73, conflating routing with FFN
            // evaluation; this closure is what a `BoundOpKind::TopK`
            // matcher must actually anchor on and replace.
            let mut routing_nodes: alloc::collections::BTreeSet<u32> = alloc::collections::BTreeSet::new();
            let mut frontier: Vec<NodeId> = site
                .selected
                .iter()
                .chain(site.weights.iter())
                .copied()
                .collect();
            while let Some(node) = frontier.pop() {
                if (node.0 as usize) < routing_start || !routing_nodes.insert(node.0) {
                    continue;
                }
                frontier.extend(program[node.0 as usize].dependencies());
            }
            let mut routing_op_ids: Vec<u32> = routing_nodes.into_iter().collect();
            routing_op_ids.sort_unstable();

            println!(
                "row 569 qwen35moe routing census: {} ops in the pure routing-decision closure \
                 for expert_count={EXPERT_COUNT}, expert_used_count={EXPERT_USED_COUNT}",
                routing_op_ids.len()
            );
            for node_id in &routing_op_ids {
                println!(
                    "  node={node_id} kind={:?}",
                    core::mem::discriminant(&program[*node_id as usize])
                );
            }
            println!("gather-index nodes (site.selected): {:?}", site.selected);
            println!("weight nodes (site.weights, last is weight_total): {:?}", site.weights);
            println!(
                "resolved bind produced {} total BoundOps for this routing+FFN program \
                 (routing closure = {} of them)",
                resolved.len(),
                routing_op_ids.len()
            );

            // A degenerate ties fixture proves the tie-break rule the
            // `mask * expert_index` -> `reduce Maximum` construction
            // implements: two experts tied at the maximum score, the
            // reduce keeps the HIGHER index -- opposite of
            // `top_k_routes_and_weights`'s own doc comment ("ties broken
            // toward the lower index"), which this test's own finding
            // (ROW 569) shows is stale prose, not the code's behavior.
            let mut tie_logits = alloc::vec![0.0_f32; EXPERT_COUNT as usize];
            tie_logits[3] = 9.0;
            tie_logits[9] = 9.0; // exact tie, index 3 vs 9
            let x_data = alloc::vec![0.1_f32; EMBEDDING as usize];
            let expert_w_gate_data =
                alloc::vec![0.0_f32; EXPERT_COUNT as usize * EMBEDDING as usize * FEED_FORWARD as usize];
            let expert_w_up_data = expert_w_gate_data.clone();
            let mut expert_w_down_data = expert_w_gate_data.clone();
            // Tag expert 3's down-projection distinctly from expert 9's so
            // the winning route is visible in the output, not just in the
            // route node's own buffer.
            let tagged = |data: &mut Vec<f32>, expert: usize, value: f32| {
                let base = expert * FEED_FORWARD as usize * EMBEDDING as usize;
                for element in 0..(FEED_FORWARD as usize * EMBEDDING as usize) {
                    data[base + element] = value;
                }
            };
            tagged(&mut expert_w_down_data, 3, 0.0);
            tagged(&mut expert_w_down_data, 9, 1.0);

            let inputs = alloc::vec![
                (x, x_data),
                (logits, tie_logits),
                (expert_w_gate, expert_w_gate_data),
                (expert_w_up, expert_w_up_data),
                (expert_w_down, expert_w_down_data),
            ];
            let buffers = run_resolved(program.len(), &resolved, inputs);
            let route_0 = buffers[site.selected[0].0 as usize]
                .as_ref()
                .expect("round 0 route resolves")[0];
            println!("tie fixture: round 0 route = {route_0} (expects 9, the higher index)");
            assert!(
                (route_0 - 9.0).abs() < 1e-6,
                "the mask*iota->reduce-Maximum tie-break keeps the HIGHER index on an exact \
                 tie, got route {route_0}, expected 9"
            );
        }
    }
}
