use super::*;

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

pub(super) type BoundOperands = Vec<(NodeId, Layout, Option<Lookup>)>;

/// [`attention_score_sources`]'s own return shape: the rotary even/odd
/// query+key sources, plus `Some((query_pass_grouped, key_pass))` only when
/// qwen35's partial-rotary chain matched.
#[cfg(feature = "cached-attention-streaming")]
pub(super) type AttentionScoreSources = (NodeId, NodeId, NodeId, NodeId, Option<(NodeId, NodeId)>);

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
    /// One qwen35moe layer's whole top-k routing decision (ROW 569,
    /// `docs/discipline.md`), collapsing
    /// [`crate::spec::append_moe_ffn`]'s own `expert_used_count`
    /// unrolled argmax-with-exclusion rounds into one bound op: `operands`
    /// carries exactly one entry, `scores` (the gate logits under
    /// [`crate::spec::ExpertGatingFunc::Softmax`], `scores` aliased to
    /// `logits`, no `expert_bias` -- this slice's only matched shape, the one
    /// `proxima-model-interop/src/qwen35moe/program.rs`'s own
    /// `append_qwen35moe_ffn` builds). `n_tokens == 1` is this slice's only
    /// supported shape (decode), the same restriction
    /// [`BoundOpKind::GatedDeltaNet`] carries for the same reason: an
    /// M-token prefill bind is out of scope until that slice lands.
    ///
    /// Each round picks the still-live expert with the MAXIMUM score
    /// (`mask = Equal(selection_scores, max_selection)`,
    /// `candidate = mask * expert_index`,
    /// `route = reduce(Maximum, Int32, candidate)`) -- an exact tie keeps the
    /// HIGHER index, because `reduce(Maximum, ...)` over `mask * expert_index`
    /// can only ever prefer the larger product (ROW 569's own census fixture
    /// proves this against a genuine tied-score pair). `weight_r =
    /// exp(max_selection_r - max_selection_0)` (the softmax-restricted-to-
    /// top-k shape [`crate::spec::append_moe_ffn`]'s own doc names),
    /// `weight_total = sum(weight_0..weight_{top_k-1})`; the caller's own
    /// `output = weighted_sum * (1 / weight_total)` renormalization
    /// [`crate::spec::append_moe_ffn`] builds AFTER this op is
    /// exactly the same consumer whether or not this kind fires, since
    /// `weight_total` is this op's own third kind of output, not
    /// recomputed.
    ///
    /// `routes`/`weights` are `top_k`-length, one entry per round, in round
    /// order — index `r` is round `r`'s own `route`/`weight` node, exactly
    /// the [`crate::spec::MoeSite::selected`]/[`crate::spec::MoeSite::weights`]
    /// entries the unfused chain already produces, so every downstream
    /// per-round `gathered_expert_product` consumer reads the identical
    /// `NodeId` whether or not this kind fired — this op only replaces how
    /// those values are PRODUCED, never what reads them. `routes[0]` is
    /// this op's own primary `node` (the same "first output is the bound
    /// op's own node, extra outputs are named fields" shape
    /// [`BoundOpKind::GatedDeltaNet::state_out`] established); `routes[1..]`,
    /// every `weights` entry, and `weight_total` are extra outputs, written
    /// by the same dispatch that writes `routes[0]`'s own buffer — ordinary
    /// `device_buffers`/interpreter-buffer-table entries, no placement
    /// threading, because none of them is cross-decode-step persistent
    /// recurrent state the way `GatedDeltaNet::state_out` is: every one of
    /// these 17 values is consumed entirely within the SAME forward
    /// evaluation that produced it.
    MoeTopK {
        operands: BoundOperands,
        expert_count: u64,
        top_k: u64,
        routes: Vec<NodeId>,
        weights: Vec<NodeId>,
        weight_total: NodeId,
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
    /// `k` round-sibling folds (the same [`ScalarOp`]/[`ReduceInit`]/
    /// `operands` shape, differing only in which route each round gathers)
    /// collapsed into one `BoundOp` by the `metal-moe-mul-mat-id` admission
    /// (`crate::bind::moe_round_group_candidates`) — a SEPARATE variant from
    /// [`BoundOpKind::Reduce`] rather than an `Option<u32>` field on it, so
    /// every backend match is forced to handle or explicitly decline this
    /// shape at compile time instead of silently falling through a `Reduce`
    /// catch-all and rendering round 0 only. `extents` carries one extra
    /// trailing axis of size `round_count` (the round axis), and a renderer
    /// that understands it folds `thread_position_in_grid.z` into that
    /// axis's own stride-addressed read/write the same way every other
    /// iteration axis already resolves. Every other field mirrors
    /// [`BoundOpKind::Reduce`]'s own doc exactly — see that variant for what
    /// each one means.
    RoundBatchedReduce {
        element_body: ComposedBody,
        reduce_op: ScalarOp,
        init: ReduceInit,
        keep: Keep,
        operands: BoundOperands,
        output_axes: SmallVec<[u16; MAX_INLINE_RANK]>,
        out_layout: Layout,
        out_scatter: Option<Lookup>,
        epilogue_body: ComposedBody,
        epilogue_operands: BoundOperands,
        epilogue_broadcast_axes: SmallVec<[u16; MAX_INLINE_RANK]>,
        /// How many round-sibling folds were collapsed into this one op —
        /// always `>= 2` (a single round never fires this fusion; see
        /// `crate::bind::moe_round_groups`'s own `>= 2` admission filter).
        round_count: u32,
        /// `round_count`-length, one entry per round, in round order — round
        /// `z`'s own gather-index `NodeId` (`crate::bind::moe_round_reduce_operand`'s
        /// own `route`, the same value `operands`'s own stack `Lookup::indices`
        /// carries for round 0 ONLY). This is the information [`BoundOpKind::Reduce`]
        /// -> [`Self::RoundBatchedReduce`] collapse used to DROP (rounds `1..k`'s
        /// own route nodes, along with their `BoundOp`s) — carrying the whole
        /// array here, [`MoeTopK`](Self::MoeTopK)'s own `routes` shape, is what
        /// lets a renderer read `round_routes[z]` for round `z` instead of
        /// reading round 0's own `operands`'s `Lookup` `k` times. A renderer
        /// walks `thread_position_in_grid.z` (or an equivalent per-round loop
        /// index) into this array to pick the gather source for that round's
        /// own dispatch, swapping it into `operands`'s stack `Lookup::indices`
        /// before addressing.
        round_routes: Vec<NodeId>,
        /// `round_count`-length, one entry per round, in round order — round
        /// `z`'s own output `NodeId`. `round_outputs[0]` is this op's own
        /// primary `node` (the same "first output is the bound op's own node,
        /// extra outputs are named fields" shape [`GatedDeltaNet::state_out`](Self::GatedDeltaNet)/
        /// [`MoeTopK::routes`](Self::MoeTopK) already establish); `round_outputs[1..]`
        /// are the k-1 round-sibling reduce nodes this collapse used to drop
        /// entirely, now ordinary extra outputs written by the same dispatch
        /// that writes `round_outputs[0]`'s buffer — every downstream
        /// per-round consumer (`append_moe_round_output`'s own activation/
        /// weight-scale chain) still reads the identical `NodeId` it always
        /// did, unperturbed by whether this kind fired.
        round_outputs: Vec<NodeId>,
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
            BoundOpKind::MoeTopK { .. } => "moe_topk",
            BoundOpKind::Elementwise { .. } => "elementwise",
            BoundOpKind::Reduce {
                keep: Keep::Reduce, ..
            } => "keep::reduce fold",
            BoundOpKind::Reduce {
                keep: Keep::Scan, ..
            } => "keep::scan fold",
            BoundOpKind::RoundBatchedReduce { .. } => "round_batched_reduce fold",
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
pub(super) static EMPTY_BODY: ComposedBody = ComposedBody { steps: Vec::new() };

impl BoundOp {
    #[must_use]
    pub fn operands(&self) -> &[(NodeId, Layout, Option<Lookup>)] {
        match &self.kind {
            BoundOpKind::CachedAttention { operands, .. }
            | BoundOpKind::GatedDeltaNet { operands, .. }
            | BoundOpKind::MoeTopK { operands, .. }
            | BoundOpKind::Elementwise { operands, .. }
            | BoundOpKind::Reduce { operands, .. }
            | BoundOpKind::RoundBatchedReduce { operands, .. } => operands,
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
            }
            | BoundOpKind::RoundBatchedReduce {
                epilogue_operands, ..
            } => epilogue_operands,
            BoundOpKind::CachedAttention { .. }
            | BoundOpKind::GatedDeltaNet { .. }
            | BoundOpKind::MoeTopK { .. }
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
            BoundOpKind::CachedAttention { .. }
            | BoundOpKind::GatedDeltaNet { .. }
            | BoundOpKind::MoeTopK { .. } => &EMPTY_BODY,
            BoundOpKind::Elementwise { body, .. } => body,
            BoundOpKind::Reduce { element_body, .. }
            | BoundOpKind::RoundBatchedReduce { element_body, .. } => element_body,
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
            BoundOpKind::CachedAttention { .. }
            | BoundOpKind::GatedDeltaNet { .. }
            | BoundOpKind::MoeTopK { .. } => None,
            // a round-merged fold is never split: `extents`' trailing round
            // axis has no `rebase_chunk` handling yet (this variant's own
            // doc), and every existing caller (`metal-moe-mul-mat-id`
            // decode-only) never chunks this op anyway.
            BoundOpKind::RoundBatchedReduce { .. } => None,
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
            // `GatedDeltaNet`/`MoeTopK` (both this slice's `n_tokens == 1`
            // shape, never worth chunking), kept explicit for the same
            // reason `Iota`/`Constant` below are.
            kind @ (BoundOpKind::GatedDeltaNet { .. } | BoundOpKind::MoeTopK { .. }) => {
                kind.clone()
            }
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
            // `RoundBatchedReduce` (see its own arm above), kept explicit
            // rather than a catch-all for the same reason `Iota`/`Constant`
            // below are.
            kind @ BoundOpKind::RoundBatchedReduce { .. } => kind.clone(),
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

pub(super) fn rebase_operands(
    operands: &BoundOperands,
    split_axis: u16,
    chunk_start: u64,
) -> BoundOperands {
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
pub(super) fn chunk_ranges(
    extent: u64,
    parts: usize,
    alignment: u64,
) -> impl Iterator<Item = (u64, u64)> {
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

pub(super) fn rebase_layout(layout: &Layout, split_axis: u16, chunk_start: u64) -> Layout {
    Layout {
        base: layout.base + layout.stride(split_axis) * chunk_start as i64,
        strides: layout.strides.clone(),
    }
}

#[derive(Clone)]
pub(super) struct HeldElementwise {
    pub(super) dtype: DType,
    pub(super) body: ScalarOp,
    pub(super) operands: Vec<(NodeId, IndexMap)>,
}
