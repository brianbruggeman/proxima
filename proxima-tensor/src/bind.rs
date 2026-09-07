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
use crate::numeric::{NumericPolicy, NumericRewrite, admit};
#[cfg(feature = "instrument")]
use crate::instrument;
use crate::live;
#[cfg(feature = "cached-attention-streaming")]
use crate::map;
use crate::map::{AxisIndex, AxisTerm, IndexMap, IndexPattern};
use crate::op::{Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp};
use crate::shape::{self, Shapes};

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
    /// `operands` carries exactly eight Q/K/V sources for a two-range fusion
    /// (`cached_attention_candidates`) and a ninth, rank-0 `cached_len`
    /// scalar for a single-range fusion
    /// (`cached_attention_single_range_candidates`): that call's
    /// `causal_mask_merged` band depends on the true `cached_len` VALUE, not
    /// on any shape the bind-time extents alone determine (`kv-capacity-
    /// bucket` widens the key extent past the merged length, so a shape
    /// difference silently overstates it). `new_upper_inclusive` is the
    /// bound an executor uses only when `operands.len() == 8`; when it is 9,
    /// every executor reads the real bound from `operands[8]`'s buffer at
    /// run time instead, and `new_upper_inclusive` here is unused filler.
    CachedAttention {
        operands: BoundOperands,
        query_rows: u64,
        cached_key_rows: u64,
        new_key_rows: u64,
        kv_heads: u64,
        query_groups: u64,
        head_dim: u64,
        scale: f32,
        cached_lower_inclusive: i64,
        new_upper_inclusive: i64,
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
            | BoundOpKind::Elementwise { operands, .. }
            | BoundOpKind::Reduce { operands, .. } => {
                operands
            }
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
            BoundOpKind::CachedAttention { .. } => &EMPTY_BODY,
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
            BoundOpKind::CachedAttention { .. } => None,
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
                scale: *scale,
                cached_lower_inclusive: *cached_lower_inclusive,
                new_upper_inclusive: *new_upper_inclusive,
            },
            BoundOpKind::Elementwise { body, operands } => BoundOpKind::Elementwise {
                body: body.clone(),
                operands: rebase_operands(operands, split_axis, chunk_start),
            },
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
    /// `constant_value[node.0]` is `Some(value)` when `node` was pushed as an
    /// [`Op::Constant`] carrying `value` — a generalization of `ones` that
    /// keeps the actual stride literal (not just whether it is `1.0`), which
    /// [`eliminate_masked_window_reduce`]'s in-bounds proof needs.
    constant_value: RefCell<Vec<Option<f32>>>,
}

impl BoundOpBuilder {
    /// `retires` is normally [`live::annotate`]`(program, outputs)`.
    #[must_use]
    pub fn new(retires: Vec<Vec<NodeId>>) -> Self {
        Self {
            held: RefCell::new(BTreeMap::new()),
            retires,
            position: Cell::new(0),
            ones: RefCell::new(Vec::new()),
            is_iota: RefCell::new(Vec::new()),
            constant_value: RefCell::new(Vec::new()),
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
                            },
                        ),
                    )?;
                    return Ok(emitted);
                }

                let still_live = !retires.contains(&reduce.operand);
                let non_identity = !is_identity_projection(&reduce.in_map);
                let not_held = !self.held.borrow().contains_key(&reduce.operand);
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

    /// Flush every elementwise op still held: each was either a requested
    /// output or dead code, and either way it materializes as its own op.
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
            // NOT `if let Some(x) = self.held.borrow_mut()....` — that
            // temporary's `RefMut` lives to the end of the `if let` body
            // under Rust's temporary-lifetime-extension rule, and
            // `build_elementwise_op` below borrows `self.held` itself, so
            // the two would collide. Ending the borrow at this statement's
            // semicolon first avoids the re-entrant panic.
            let removed = self.held.borrow_mut().remove(&node);
            if let Some(held) = removed {
                built.push(build_elementwise_op(
                    node,
                    shapes,
                    &self.held,
                    held.dtype,
                    held.body,
                    &held.operands,
                    Constants {
                        ones: &self.ones.borrow(),
                        values: &self.constant_value.borrow(),
                    },
                ));
            }
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
        // see `finish`'s comment: the borrow must end before this `if let`
        // body runs, since `build_elementwise_op` borrows `self.held` too.
        let removed = self.held.borrow_mut().remove(&node);
        if let Some(held) = removed {
            let materialized = build_elementwise_op(
                node,
                shapes,
                &self.held,
                held.dtype,
                held.body,
                &held.operands,
                Constants {
                    ones: &self.ones.borrow(),
                    values: &self.constant_value.borrow(),
                },
            );
            push_ready(emitted, node, materialized)?;
        }
        Ok(())
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
    if steps.is_empty() {
        steps.push(BodyStep {
            op: ScalarOp::Identity,
            args: alloc::vec![arg],
        });
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
        Some((survivor_node, survivor_map)) => {
            compose_operand(shapes, held, &mut state, survivor_node, survivor_map, constants)
        }
        None => compose_body(shapes, held, &mut state, body, operands, constants),
    };
    if steps.is_empty() {
        steps.push(BodyStep {
            op: ScalarOp::Identity,
            args: alloc::vec![arg],
        });
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
/// class — `x op identity == x` for every finite/inf/nan `f32` — or `None`
/// for a `ScalarOp` with no such element. Generalizes
/// [`eliminate_identity_multiply`]'s single hard-coded `(Multiply, 1.0)` case
/// to the whole class `op.rs`'s own `is_associative` already names (`Add`,
/// `Multiply`, `Maximum`, `Minimum`; `op.rs:112-117`), so `x + 0`, `max(x,
/// -inf)`, and `min(x, +inf)` are recognized here the same way `x * 1` always
/// was.
const fn identity_element(op: ScalarOp) -> Option<f32> {
    match op {
        ScalarOp::Add => Some(0.0),
        ScalarOp::Multiply => Some(1.0),
        ScalarOp::Maximum => Some(f32::NEG_INFINITY),
        ScalarOp::Minimum => Some(f32::INFINITY),
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
fn step_arg_constant(arg: StepArg, state: &ComposeState<'_>, constant_value: &[Option<f32>]) -> Option<f32> {
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
/// [`identity_element`] before ever minting a step, so two authored orderings
/// of the same algebraic expression — `a*b+c` and `c+a*b`, or a chain with an
/// identity multiply/add folded away by an earlier rewrite — produce the
/// identical [`StepArg`], never a step whose recognizability depends on
/// which rewrite fired first in the same bind call
/// (`proxima-tensor/src/cpu.rs:2570-2626`'s own "hidden=1 confluence gap"
/// doc). Mints no new `ScalarOp`/`Op` variant: every value this returns is
/// either an existing `StepArg` unchanged or a freshly pushed `BodyStep`
/// using `op` exactly as given.
fn push_canonical_step(
    state: &mut ComposeState<'_>,
    op: ScalarOp,
    mut args: Vec<StepArg>,
    constant_value: &[Option<f32>],
) -> StepArg {
    if op.is_associative() && args.len() == 2 {
        args.sort_by_key(step_arg_sort_key);
    }
    if let (Some(identity), [first, second]) = (identity_element(op), args.as_slice()) {
        if step_arg_constant(*first, state, constant_value) == Some(identity) {
            return *second;
        }
        if step_arg_constant(*second, state, constant_value) == Some(identity) {
            return *first;
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
    push_canonical_step(state, body, args, constants.values)
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
    resolved
        .iter()
        .map(|computed| computed.node)
        .filter(|node| !consumed.contains(node) && !effective_outputs.contains(node))
        .collect()
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
fn binary_elementwise(
    program: &[Op],
    node: NodeId,
    body: ScalarOp,
) -> Option<[NodeId; 2]> {
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
            if reduce.body == body
                && reduce.init == init
                && reduce.keep == Keep::Reduce =>
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
fn attention_score_sources(
    program: &[Op],
    score: NodeId,
    scale: NodeId,
) -> Option<(NodeId, NodeId, NodeId, NodeId)> {
    let scaled = binary_elementwise(program, score, ScalarOp::Multiply)?;
    if constant_value(program, scaled[1]) != constant_value(program, scale)
        || constant_value(program, scaled[1]).is_none()
    {
        return None;
    }
    let score_sum = binary_elementwise(program, scaled[0], ScalarOp::Add)?;
    let even_product = reduced_source(program, score_sum[0], ScalarOp::Add, ReduceInit::Zero)?;
    let odd_product = reduced_source(program, score_sum[1], ScalarOp::Add, ReduceInit::Zero)?;
    let even_operands = binary_elementwise(program, even_product, ScalarOp::Multiply)?;
    let odd_operands = binary_elementwise(program, odd_product, ScalarOp::Multiply)?;
    Some((
        even_operands[0],
        odd_operands[0],
        even_operands[1],
        odd_operands[1],
    ))
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
        (
            Some(Op::Iota { .. }),
            Some(Op::Iota { .. }),
        )
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

#[cfg(feature = "cached-attention-streaming")]
fn cached_attention_candidates(
    program: &[Op],
    shapes: &Shapes,
    resolved: &[BoundOp],
    effective_outputs: &[NodeId],
) -> Vec<(BoundOp, BTreeSet<NodeId>)> {
    let mut candidates = Vec::new();
    for output_position in (0..program.len()).rev() {
        let output = NodeId(output_position as u32);
        let Some(attended_sum) = binary_elementwise(program, output, ScalarOp::Multiply) else {
            continue;
        };
        let Some(attended_parts) = binary_elementwise(program, attended_sum[0], ScalarOp::Add) else {
            continue;
        };
        let Some(inverse_sum) = unary_elementwise(program, attended_sum[1], ScalarOp::Reciprocal) else {
            continue;
        };
        let Some(sum_parts) = binary_elementwise(program, inverse_sum, ScalarOp::Add) else {
            continue;
        };
        let Some(cached_weights) = reduced_source(program, sum_parts[0], ScalarOp::Add, ReduceInit::Zero) else {
            continue;
        };
        let Some(new_weights) = reduced_source(program, sum_parts[1], ScalarOp::Add, ReduceInit::Zero) else {
            continue;
        };
        let Some(cached_shift) = unary_elementwise(program, cached_weights, ScalarOp::Exponential) else {
            continue;
        };
        let Some(new_shift) = unary_elementwise(program, new_weights, ScalarOp::Exponential) else {
            continue;
        };
        let Some(cached_score_parts) = binary_elementwise(program, cached_shift, ScalarOp::Subtract) else {
            continue;
        };
        let Some(new_score_parts) = binary_elementwise(program, new_shift, ScalarOp::Subtract) else {
            continue;
        };
        if cached_score_parts[1] != new_score_parts[1] {
            continue;
        }
        let new_masked = new_score_parts[0];
        let Some(mask_parts) = elementwise_operands(program, new_masked, ScalarOp::Select) else {
            continue;
        };
        let [(mask, _), (negative_infinity, _), (new_scaled, _)] = mask_parts else {
            continue;
        };
        if !is_exact_causal_mask(program, *mask)
            || constant_value(program, *negative_infinity) != Some(f32::NEG_INFINITY)
        {
            continue;
        }
        let Some(cached_scaled_parts) =
            binary_elementwise(program, cached_score_parts[0], ScalarOp::Multiply)
        else {
            continue;
        };
        let scale = cached_scaled_parts[1];
        let Some(new_scaled_parts) = binary_elementwise(program, *new_scaled, ScalarOp::Multiply) else {
            continue;
        };
        if new_scaled_parts[1] != scale {
            continue;
        }
        let Some((query_even_grouped, query_odd_grouped, cached_key_even, cached_key_odd)) =
            attention_score_sources(program, cached_score_parts[0], scale)
        else {
            continue;
        };
        let Some((new_query_even_grouped, new_query_odd_grouped, new_key_even, new_key_odd)) =
            attention_score_sources(program, *new_scaled, scale)
        else {
            continue;
        };
        if new_query_even_grouped != query_even_grouped || new_query_odd_grouped != query_odd_grouped {
            continue;
        }
        let Some(query_even_parts) = binary_elementwise(program, query_even_grouped, ScalarOp::Multiply)
        else {
            continue;
        };
        let Some(query_odd_parts) = binary_elementwise(program, query_odd_grouped, ScalarOp::Multiply)
        else {
            continue;
        };
        if query_even_parts[1] != query_odd_parts[1] {
            continue;
        }
        let query_even = query_even_parts[0];
        let query_odd = query_odd_parts[0];
        let Some(cached_value_source) =
            reduced_source(program, attended_parts[0], ScalarOp::Add, ReduceInit::Zero)
        else {
            continue;
        };
        let Some(new_value_source) =
            reduced_source(program, attended_parts[1], ScalarOp::Add, ReduceInit::Zero)
        else {
            continue;
        };
        let Some(cached_value_product) =
            binary_elementwise(program, cached_value_source, ScalarOp::Multiply)
        else {
            continue;
        };
        let Some(new_value_product) =
            binary_elementwise(program, new_value_source, ScalarOp::Multiply)
        else {
            continue;
        };
        let cached_value = cached_value_product[1];
        let new_value = new_value_product[1];
        if cached_value_product[0] != cached_weights || new_value_product[0] != new_weights {
            continue;
        }
        let source_nodes = [
            query_even,
            query_odd,
            cached_key_even,
            cached_key_odd,
            new_key_even,
            new_key_odd,
            cached_value,
            new_value,
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
        let cached_key_shape = shapes.of(cached_key_even);
        let new_key_shape = shapes.of(new_key_even);
        let cached_value_shape = shapes.of(cached_value);
        let new_value_shape = shapes.of(new_value);
        let Some(head_dim) = query_shape[3].checked_mul(2) else {
            continue;
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
            || cached_value_shape[2] != head_dim
            || new_value_shape[2] != head_dim
            || shapes.of(output)
                != [query_shape[0], query_shape[1], query_shape[2], head_dim]
        {
            continue;
        }
        let pair_dim = query_shape[3];
        let query_strides = [
            (query_shape[1] * query_shape[2] * pair_dim) as i64,
            (query_shape[2] * pair_dim) as i64,
            pair_dim as i64,
            1i64,
        ];
        let key_strides = [0i64, (query_shape[1] * pair_dim) as i64, pair_dim as i64, 0, 1];
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
        let dependencies = dependencies
            .difference(&source_nodes.into_iter().collect())
            .copied()
            .collect::<BTreeSet<_>>();
        if dependencies.iter().any(|node| effective_outputs.contains(node)) {
            continue;
        }
        let absorbed = removable_attention_dependencies(program, &dependencies, output);
        if absorbed.is_empty() {
            continue;
        }
        if !resolved.iter().any(|bound| bound.node == output) {
            continue;
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
                head_dim,
                scale: scale_value,
                cached_lower_inclusive: i64::MIN,
                new_upper_inclusive: 0,
            },
        };
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
        let Some(attended_product) = reduced_source(program, output, ScalarOp::Add, ReduceInit::Zero)
        else {
            continue;
        };
        let Some(attended_parts) = binary_elementwise(program, attended_product, ScalarOp::Multiply)
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
        let Some(mask_parts) = elementwise_operands(program, scores_masked, ScalarOp::Select) else {
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
        let Some((query_even_grouped, query_odd_grouped, key_even, key_odd)) =
            attention_score_sources(program, *scores_scaled, scale)
        else {
            continue;
        };
        let Some(query_even_parts) =
            binary_elementwise(program, query_even_grouped, ScalarOp::Multiply)
        else {
            continue;
        };
        let Some(query_odd_parts) = binary_elementwise(program, query_odd_grouped, ScalarOp::Multiply)
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
            query_even,
            query_odd,
            key_even,
            key_odd,
            key_even,
            key_odd,
            value,
            value,
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
        let cached_len_operand = (cached_len_node, Layout { base: 0, strides: SmallVec::new() }, None);
        let pair_dim = query_shape[3];
        let query_strides = [
            (query_shape[1] * query_shape[2] * pair_dim) as i64,
            (query_shape[2] * pair_dim) as i64,
            pair_dim as i64,
            1i64,
        ];
        let key_strides = [0i64, (query_shape[1] * pair_dim) as i64, pair_dim as i64, 0, 1];
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
        if dependencies.iter().any(|node| effective_outputs.contains(node)) {
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
fn attention_dependencies(
    program: &[Op],
    output: NodeId,
    sources: &[NodeId; 8],
) -> BTreeSet<NodeId> {
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
            consumers.entry(dependency).or_insert_with(BTreeSet::new).insert(consumer);
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
) -> Result<Vec<BoundOp>, TensorError> {
    bind_with_fusion(program, shapes, outputs, true, NumericPolicy::default())
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
/// [`NumericRewrite::minimum_level`] is [`NumericPolicy::BitExact`], so
/// [`bind`]'s own call with [`NumericPolicy::default`] always clears —
/// nothing regresses. A future reassociating rewrite in this crate declares
/// its own [`NumericRewrite`] variant and is admitted the same way.
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
    let built = bind_cached_attention_fusion(program, shapes, outputs, fuse_cached_attention)?;
    #[cfg(feature = "reduce-epilogue-fusion")]
    {
        admit(numeric_policy, NumericRewrite::ReduceEpilogueFusion)?;
        reduce_epilogue_fusion(built, outputs)
    }
    #[cfg(not(feature = "reduce-epilogue-fusion"))]
    Ok(built)
}

fn bind_cached_attention_fusion(
    program: &[Op],
    shapes: &Shapes,
    outputs: &[NodeId],
    fuse_cached_attention: bool,
) -> Result<Vec<BoundOp>, TensorError> {
    let built = bind_plain(program, shapes, outputs)?;
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
    let mut initial_candidates = cached_attention_candidates(program, shapes, &built, outputs);
    initial_candidates
        .extend(cached_attention_single_range_candidates(program, shapes, &built, outputs));
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
        for (source, _, _) in operands {
            if !planning_outputs.contains(source) {
                planning_outputs.push(*source);
            }
        }
    }
    let rebuilt = bind_plain(program, shapes, &planning_outputs)?;
    let mut candidates = cached_attention_candidates(program, shapes, &rebuilt, outputs);
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
            ) else {
                continue;
            };
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

    let mut steps: Vec<BodyStep> = inner_epilogue_body
        .steps
        .iter()
        .map(|step| BodyStep {
            op: step.op,
            args: step
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
                .collect(),
        })
        .collect();
    for step in &outer_body.steps {
        steps.push(BodyStep {
            op: step.op,
            args: step
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
                .collect(),
        });
    }
    Some((ComposedBody { steps }, new_operands))
}

fn bind_plain(
    program: &[Op],
    shapes: &Shapes,
    outputs: &[NodeId],
) -> Result<Vec<BoundOp>, TensorError> {
    let retires = live::annotate(program, outputs);
    let building = BoundOpBuilder::new(retires);
    let mut built = Vec::new();
    for expr in program {
        built.extend(building.push(expr, shapes)?);
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
#[must_use]
pub fn node_retirement(resolved: &[BoundOp], outputs: &[NodeId]) -> Vec<Vec<NodeId>> {
    let outputs: BTreeSet<NodeId> = outputs.iter().copied().collect();
    let mut last_use: BTreeMap<NodeId, usize> = BTreeMap::new();
    for (position, node) in resolved.iter().enumerate() {
        for (source, _, gather) in node.all_read_sources() {
            last_use.insert(*source, position);
            if let Some(gather_access) = gather {
                last_use.insert(gather_access.indices, position);
            }
        }
    }

    let mut retires = vec![Vec::new(); resolved.len()];
    for (node, position) in last_use {
        if !outputs.contains(&node) {
            retires[position].push(node);
        }
    }
    retires
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

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
        let (program, _, _) = crate::spec::mistral_cached_forward_program(32, 16, 24, 4, 2, 4, 1)
            .expect("cached attention fixture builds");
        let shapes = crate::shape::infer(&program, &[1, 1]).expect("cached attention infers");
        let outputs: &[NodeId] = &[];
        let plain = bind_plain(&program, &shapes, outputs).expect("plain bind succeeds");
        let cached_only = bind_cached_attention_fusion(&program, &shapes, outputs, true)
            .expect("cached-attention-only bind succeeds");
        let rewritten = bind(&program, &shapes, outputs).expect("rewritten bind succeeds");

        assert_eq!(plain.len(), 48, "fixture baseline bound operation count");
        assert_eq!(
            cached_only.len(),
            26,
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
        assert!(rewritten
            .iter()
            .any(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. })));
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
    #[test]
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
                crate::spec::DuplicateHeadPosition::None,
            )
            .expect("one-layer single-range decode fixture builds");
        let shapes = crate::shape::infer(&program, &[1, 1]).expect("cached decode fixture infers");
        let mut outputs = alloc::vec![logits];
        for (even, odd, value) in &roots {
            outputs.extend_from_slice(&[*even, *odd, *value]);
        }
        let bound = bind(&program, &shapes, &outputs)
            .expect("the cached decode fixture binds through the real bind() path");

        for (index, op) in bound.iter().enumerate() {
            match &op.kind {
                BoundOpKind::Elementwise { body, .. } => {
                    let step_ops: Vec<ScalarOp> = body.steps.iter().map(|step| step.op).collect();
                    std::println!(
                        "row364 index={index} kind=elementwise step_ops={step_ops:?}"
                    );
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
        let shapes = crate::shape::infer(&program, &[1, 5])
            .expect("omega cached attention fixture infers");
        let rewritten = bind(&program, &shapes, &outputs)
            .expect("omega cached attention fixture binds");

        assert_eq!(
            rewritten
                .iter()
                .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
                .count(),
            2,
            "each production-shaped layer must receive its own fused step"
        );
        assert!(rewritten
            .iter()
            .any(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. })));
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
        let (program, logits, cache_roots, _) = crate::spec::mistral_single_range_cached_forward_program(
            32_002, 4096, 14336, 32, 8, 128, 32, crate::spec::DuplicateHeadPosition::None,
        )
        .expect("openchat-shaped single-range forward pass lowers to a program");
        let mut outputs = alloc::vec![logits];
        for (even, odd, value) in &cache_roots {
            outputs.extend_from_slice(&[*even, *odd, *value]);
        }
        let shapes = crate::shape::infer(&program, &[1, 71])
            .expect("one new position against a 71-position merged range infers");
        let plain = bind_plain(&program, &shapes, &outputs).expect("plain bind succeeds");
        let cached_only = bind_cached_attention_fusion(&program, &shapes, &outputs, true)
            .expect("cached-attention-only bind succeeds");
        let rewritten = bind(&program, &shapes, &outputs).expect("fused bind succeeds");

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
                crate::spec::DuplicateHeadPosition::None,
            )
            .expect("single-range fixture builds");
        let mut outputs = alloc::vec![logits];
        for (even, odd, value) in &cache_roots {
            outputs.extend_from_slice(&[*even, *odd, *value]);
        }
        let shapes = crate::shape::infer(&program, &[1, 5]).expect("single-range fixture infers");
        let resolved = bind_plain(&program, &shapes, &outputs).expect("plain bind succeeds");

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
                crate::spec::DuplicateHeadPosition::None,
            )
            .expect("single-range fixture builds");
        let mut outputs = alloc::vec![logits];
        for (even, odd, value) in &cache_roots {
            outputs.extend_from_slice(&[*even, *odd, *value]);
        }
        let shapes =
            crate::shape::infer(&program, &[1, 5]).expect("single-range fixture infers");
        let mut resolved = bind_plain(&program, &shapes, &outputs).expect("plain bind succeeds");

        let baseline = cached_attention_single_range_candidates(&program, &shapes, &resolved, &outputs);
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
                BoundOpKind::Iota | BoundOpKind::Constant { .. } => continue,
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

        let patched = cached_attention_single_range_candidates(&program, &shapes, &resolved, &outputs);
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
        let built = bind(&program, &shapes, &[]).expect("iota builds ops");

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
        let built = bind(&program, &shapes, &[]).expect("matmul builds ops");

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
        let built =
            bind(&program, &shapes, &[product, sum]).expect("matmul builds ops with two outputs");

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
        let built = bind(&program, &shapes, &[]).expect("elementwise chain builds ops");

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
        let built =
            bind(&program, &shapes, &[b, d]).expect("elementwise chain builds ops with 2 outputs");

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
        let built = bind(&program, &shapes, &[]).expect("diamond chain builds ops");

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
        let built = bind(&program, &shapes, &[]).expect("weighted dot builds ops");

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
        let built = bind(&program, &shapes, &[]).expect("broadcast builds ops");
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
        let built = bind(&program, &shapes, &[]).expect("conv window builds ops");
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
        let built = bind(&program, &shapes, &[]).expect("transpose builds ops");
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
        let mut built = bind(&program, &shapes, &[]).expect("two-axis output group binds");
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
        bind(&program, &shapes, &[])
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
        bind(&program, &shapes, &[])
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
        bind(&program, &shapes, &[])
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
        let op = bind(&program, &shapes, &[])
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
        let building = BoundOpBuilder::new(Vec::new());
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
        let outputs: Vec<NodeId> = Vec::new();
        let retires = live::annotate(&program, &outputs);

        let shape_table = ShapeTable::new(&[512]);
        let builder = BoundOpBuilder::new(retires);
        let chain = shape_table.and_then(builder);

        let mut built_via_pipe = Vec::new();
        for expr in &program {
            let batch = proxima_primitives::block_on(Pipe::call(&chain, expr.clone()))
                .expect("shape+op pipe step succeeds");
            built_via_pipe.extend(batch);
        }

        let shapes = shape::infer(&program, &[512]).expect("free-function infer succeeds");
        let built_via_free_function =
            bind(&program, &shapes, &outputs).expect("free-function op building succeeds");

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

        let built = bind(&program, &shapes, &[output]).expect("binding itself never errors");
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
        let built = bind(&program, &shapes, &[]).expect("masked-window program builds ops");

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
        let built = bind(&program, &shapes, &[]).expect("masked-window program builds ops");

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
        let built = bind(&program, &shapes, &[]).expect("masked-window program builds ops");

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
            let plain = bind_plain(&program, &shapes, &[extra_x_use])
                .expect("plain bind succeeds");
            let fused = bind(&program, &shapes, &[extra_x_use]).expect("fused bind succeeds");

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

            let fused = bind(&program, &shapes, &outputs).expect("fused bind succeeds");
            let fused_buffers = run_resolved(program.len(), &fused, inputs());
            let fused_consumer = fused_buffers[consumer.0 as usize]
                .as_ref()
                .expect("fused consumer output present");

            assert_eq!(
                fused_consumer, &expected,
                "epilogued reduce must match the hand-derived sum-plus-residual exactly"
            );

            let plain = bind_plain(&program, &shapes, &outputs).expect("plain bind succeeds");
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
            let fused = bind(&program, &shapes, &[extra_x_use, second_consumer])
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
            let fused = bind(&program, &shapes, &[extra_x_use, consumer])
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
            let fused = bind(&program, &shapes, &[consumer, extra_x_use])
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
                    crate::spec::DuplicateHeadPosition::None,
                )
                .expect("openchat-shaped single-range forward pass lowers to a program");
            let mut outputs = alloc::vec![logits];
            for (even, odd, value) in &cache_roots {
                outputs.extend_from_slice(&[*even, *odd, *value]);
            }
            let shapes = crate::shape::infer(&program, &[1, 71])
                .expect("one new position against a 71-position merged range infers");
            let attention_only = bind_cached_attention_fusion(&program, &shapes, &outputs, true)
                .expect("cached-attention-only bind succeeds");
            let with_epilogue = bind(&program, &shapes, &outputs).expect("fused bind succeeds");

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
                    crate::spec::DuplicateHeadPosition::None,
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
                            _ => unreachable!("block_node_ids only ever returns named Op::Input nodes"),
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

            let with_epilogue = bind(&program, &shapes, &outputs).expect("fused bind succeeds");
            let attention_only = bind_cached_attention_fusion(&program, &shapes, &outputs, true)
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
                    operands: alloc::vec![(sum_squares, keep_seq()), (inv_dim, broadcast_scalar_seq())],
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
                let plain =
                    bind_plain(&program, &shapes, &[scaled]).expect("unfused rmsnorm binds");
                let fused = bind(&program, &shapes, &[scaled]).expect("fused rmsnorm binds");

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
                let x_data: Vec<f32> = (0..(seq as u64 * DIM as u64) as usize).map(|_| lcg.next_unit()).collect();
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
            let plain = bind_plain(&program, &shapes, &[output]).expect("unfused silu*up binds");
            let fused = bind(&program, &shapes, &[output]).expect("fused silu*up binds");

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
                fused_buffers[output.0 as usize],
                plain_buffers[output.0 as usize],
                "SiLU(gate) * up must be bit-identical fused vs unfused"
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
            let plain =
                bind_plain(&program, &shapes, &[output]).expect("unfused different-projection binds");
            let fused = bind(&program, &shapes, &[output]).expect("bind with fusion enabled still binds");

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
    }
}
