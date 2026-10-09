use proxima_primitives::Codec;
use proxima_tensor::{DType, NodeId};

/// Everything [`crate::msl::emit`] can reject.
///
/// A [`proxima_tensor::BoundOp`] node cannot itself encode a *gather*
/// operand's addressing outside its own `Lookup` field, and
/// `proxima_tensor::shape::infer` already rejects a non-integer or
/// out-of-range gather index before a `BoundOp` node is ever built (see
/// `unify_iteration_space`'s checks over every elementwise/reduce operand).
/// So nothing here re-checks *that* — most variants below guard against a
/// malformed `BoundOp` node built directly through its public, all-`pub`-field
/// struct literal, never against something `bind::bind` itself would produce.
///
/// A forward *scatter* (`BoundOpKind::Reduce::out_scatter: Some(_)`) is the
/// one exception: `bind::bind` builds a real one whenever a program's
/// `Reduce::out_map` is data-dependent (`proxima-tensor`'s own forward-scatter
/// support), so [`Self::ScatterNotSupported`] is a genuine, reachable gate —
/// none of this crate's emitters render the sequential accumulate-in-order
/// fold `proxima_tensor::cpu::run_reduce_scatter` runs on the CPU, so a
/// scatter `BoundOp` is rejected here, named, rather than silently emitting a
/// kernel that ignores `out_scatter` and writes to the wrong address.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EmitError {
    /// A dispatch wider than the emitter's thread index can address, and wider
    /// than the widest form that emitter has for it. Metal launches only the
    /// low 32 bits of a 1D grid, so `msl::grid2d_for` widens the index rather
    /// than reach this; it fires for a grid the flat 2D form cannot cover
    /// either (more threadgroups than the x and y axes hold), for `wgsl`
    /// (whose `i32 gid` tops out at `i32::MAX` and has no 64-bit integer), and
    /// for `cuda` (a grid dimension above `i32::MAX` blocks), and for any
    /// thread count whose product overflows `u64` before a grid exists
    /// (`threads` saturates at `u64::MAX`).
    #[error(
        "node {node} needs {threads} threads, above the {limit} this emitter's grid can index -- \
         a wider launch is silently truncated by the driver and leaves the tail outputs unwritten"
    )]
    GridExceedsThreadIndex {
        node: NodeId,
        threads: u64,
        limit: u64,
    },

    /// The flat 2D grid form rewrites a kernel's linear `gid` and dispatches
    /// whole threadgroups; `reason` names the property of this kernel that the
    /// rewrite cannot preserve.
    #[error("node {node} cannot take the flat 2D grid form: {reason}")]
    WideGridUnsupported {
        node: NodeId,
        reason: &'static str,
    },

    #[error("node {node} elementwise body takes {expected} operands but the op carries {found}")]
    ArityMismatch {
        node: NodeId,
        expected: usize,
        found: usize,
    },

    /// `cpu::apply_scalar_op` always calls a reduce's reduction body with
    /// exactly two operands (`[accumulator, value]`); a reduction body that
    /// reads a third (`ScalarOp::Select`) reads past that slice there too —
    /// this rejects it at emit time instead of indexing out of bounds in the
    /// generated MSL.
    #[error(
        "node {node} reduction body is select, which reads a third operand a reduce step never supplies"
    )]
    ReductionBodyIsSelect { node: NodeId },

    #[error(
        "node {node} is a keep::scan scan over zero iteration axes, which has no reduced axis to scan along"
    )]
    EmptyScan { node: NodeId },

    /// See this type's own doc for why this is a real, reachable gate rather
    /// than defensive dead code: nothing in `msl`/`wgsl`/`cuda` renders the
    /// sequential accumulate-in-order fold a scatter's colliding writes need.
    #[error(
        "node {node} is a forward scatter (Reduce::out_map is data-dependent), which no GPU \
         emitter in this crate supports yet -- proxima_tensor::cpu::run_reduce_scatter is CPU-only"
    )]
    ScatterNotSupported { node: NodeId },

    /// `wgsl`/`cuda` still reject every [`proxima_tensor::BoundOpKind::GatedDeltaNet`]
    /// bind with this error -- `crate::msl::render_gated_delta_net` is the
    /// only renderer that emits a kernel for it so far, the same
    /// CPU-ahead-of-GPU gap [`Self::ScatterNotSupported`] names for a
    /// forward scatter.
    #[error(
        "node {node} is a gated delta net op, which no GPU emitter in this crate supports yet \
         -- proxima_tensor::cpu::run_gated_delta_net is CPU-only"
    )]
    GatedDeltaNetNotSupported { node: NodeId },

    /// Candidate B's integration (`R9/PROGRESS.md`'s own "Candidate B"
    /// sections): `crate::msl::render_cached_softmax_weights` renders this
    /// kind now -- this variant stays reachable for the precondition checks
    /// that renderer's own doc names (`new_key_rows != 1`, a gathered
    /// operand, or an operand `Layout` not already collapsed to
    /// `[key,row]`/`[row]`/`[row,dim]`), and for the other GPU backends
    /// (`wgsl`/`cuda`) this kind has no renderer on at all yet -- the same
    /// CPU-ahead-of-GPU gap [`Self::GatedDeltaNetNotSupported`] names for
    /// those two.
    #[error(
        "node {node} is a cached softmax weights op this backend cannot render -- either an \
         unsupported shape (new_key_rows != 1, a gathered operand, or an uncollapsed operand \
         layout) or a GPU emitter that does not support this kind yet"
    )]
    CachedSoftmaxWeightsNotSupported { node: NodeId },

    /// `crate::msl::render_gated_delta_net` keeps one `head_k_dim`-long
    /// state row resident in registers per thread
    /// (`omega-runtime.toml`'s `[gated_delta_net] head_k_dim_max`); a bind
    /// above that compiled cap is rejected here rather than silently
    /// clamped or overflowing the fixed-size register array.
    #[error(
        "node {node} gated delta net head_k_dim {head_k_dim} exceeds the compiled cap {cap} \
         -- raise omega-runtime.toml's [gated_delta_net] head_k_dim_max"
    )]
    GatedDeltaNetHeadKDimExceedsCap {
        node: NodeId,
        head_k_dim: u64,
        cap: u64,
    },

    /// `crate::msl::render_cached_attention` still assumes every rotary
    /// plane covers the whole `head_dim` (`BoundOpKind::CachedAttention`'s
    /// own doc: `rotary_dim == head_dim` is byte-identical to this backend's
    /// pre-partial-rotary shape) -- a partial-rotary bind (the recurrent-interval family's dense
    /// attention, `rotary_dim < head_dim`, three trailing pass-plane
    /// operands) is rejected here rather than silently dropping the pass
    /// plane's score contribution, the same CPU-ahead-of-GPU gap
    /// [`Self::GatedDeltaNetNotSupported`] names — `docs/discipline.md`
    /// ROW 556/557's own residual: teaching this kernel the pass plane is
    /// its own next slice.
    #[error(
        "node {node} is a partial-rotary cached attention bind (rotary_dim < head_dim), which \
         this metal backend does not render yet -- proxima_tensor::cpu::run_cached_attention is \
         the only executor with the pass-plane term so far"
    )]
    CachedAttentionPartialRotaryNotSupported { node: NodeId },

    /// A cached-attention bind whose operands carry a packed codec the
    /// renderer does not read: only the decode split form
    /// (`render_cached_attention_decode_split`) reads its cached K/V
    /// operands (2, 3 and 6) as `Codec::Float16`, and every other operand of
    /// every form is a plain buffer. Rejected here rather than bound as
    /// `float*` over half-width bytes, which would read the cache as garbage
    /// without any error.
    #[error("node {node} is a cached attention bind with a packed operand the kernel cannot read: {reason}")]
    CachedAttentionKvCodecNotSupported { node: NodeId, reason: &'static str },

    #[error("node {node} cannot select cached-attention MMA precision {precision}: {reason}")]
    CachedAttentionMmaPrecisionNotSupported {
        node: NodeId,
        precision: &'static str,
        reason: &'static str,
    },

    #[error("cached-attention variant axis {axis} does not support value {value} yet")]
    CachedAttentionVariantAxisNotSupported {
        axis: &'static str,
        value: &'static str,
    },

    #[error("node {node} cannot select a {rows}-row attention tile: {reason}")]
    CachedAttentionTileHeightNotSupported {
        node: NodeId,
        rows: u64,
        reason: &'static str,
    },

    #[error("node {node} cannot parallelize {query_rows} query rows across simdgroups: {reason}")]
    CachedAttentionQueryParallelismNotSupported {
        node: NodeId,
        query_rows: u64,
        reason: &'static str,
    },

    #[error("node {node} cannot prefetch cached K/V blocks: {reason}")]
    CachedAttentionPrefetchNotSupported {
        node: NodeId,
        reason: &'static str,
    },

    #[error("node {node} cannot apply its simdgroup lane topology to {query_rows} query rows: {reason}")]
    CachedAttentionSimdTopologyNotSupported {
        node: NodeId,
        query_rows: u64,
        reason: &'static str,
    },

    #[error("cached-attention variant selects {selected} K/V storage but the bound operands use {bound}")]
    CachedAttentionVariantStorageMismatch {
        selected: &'static str,
        bound: &'static str,
    },

    #[error("node {node} cannot select cached-attention K reuse: {reason}")]
    CachedAttentionKvReuseNotSupported {
        node: NodeId,
        reason: &'static str,
    },

    /// A packed scalar codec reached an op whose renderer has no reader for it.
    #[error("node {node} reads packed codec {codec:?}, which this metal renderer does not support")]
    PackedCodecNotSupported { node: NodeId, codec: Codec },

    /// `omega::execute`'s own upstream gate (`reject_unsupported_gpu_dtype`)
    /// never lets anything but `Float32`/`Float16` reach [`crate::msl::emit`]
    /// in practice, but [`crate::msl::emit`] is a public entry point a
    /// caller may reach directly with a hand-built `BoundOp`, so this stays
    /// a real rejection rather than a debug assertion.
    #[error("node {node} declares dtype {dtype:?}, which this metal backend does not emit")]
    UnsupportedDType { node: NodeId, dtype: DType },

    /// `wgsl::emit_wgsl`'s v1 scope has no gather kernel (no fault buffer, no
    /// indices binding) — see that module's own doc for why this is a
    /// deliberate v1 boundary rather than an oversight. Gated with the module
    /// itself: `wgsl` only exists behind `wgpu-backend`, so an intra-doc link
    /// to it is only resolvable in that same build.
    #[cfg(feature = "wgpu-backend")]
    #[error("node {node} gathers an operand, which the wgsl v1 emitter does not support yet")]
    GatherNotSupported { node: NodeId },

    /// `wgsl::emit_wgsl`'s v1 op set is elementwise, `Keep::Reduce`,
    /// `Keep::Scan`, `Iota`, and `Constant` -- this variant is reachable for
    /// whatever op kind is added next, not for either of those two.
    #[cfg(feature = "wgpu-backend")]
    #[error("node {node} is a {kind} op, which the wgsl v1 emitter does not support yet")]
    UnsupportedOpKind { node: NodeId, kind: &'static str },

    /// `cuda::emit_cuda`'s op set is elementwise, `Keep::Reduce`, and
    /// `Keep::Scan` only — `Iota`/`Constant` have no renderer yet, the same
    /// v1 boundary [`Self::UnsupportedOpKind`] draws for the wgsl emitter.
    #[cfg(feature = "cuda")]
    #[error("node {node} is a {kind} op, which the cuda emitter does not support yet")]
    CudaUnsupportedOpKind { node: NodeId, kind: &'static str },

    /// `wgsl::emit_wgsl` has no `Q3_K` unpack function -- `crate::msl`'s own
    /// `Codec::Q3K` is metal-only so far (`wgpu_driver::packed_operands_of`
    /// already routes a `Codec::Q3K` node to `None` for this same
    /// reason); this is the typed rejection a caller who somehow threads a
    /// `Some(Codec::Q3K)` through directly still hits, rather than a
    /// generated `wgsl` calling a function that does not exist.
    #[cfg(feature = "wgpu-backend")]
    #[error("node {node} reads a Q3_K operand, which the wgsl emitter does not support yet")]
    UnsupportedCodec { node: NodeId },

    /// The cuda counterpart of [`Self::UnsupportedCodec`] -- same gap,
    /// same reason.
    #[cfg(feature = "cuda")]
    #[error("node {node} reads a Q3_K operand, which the cuda emitter does not support yet")]
    CudaUnsupportedCodec { node: NodeId },

    /// `crate::msl::render_reduce`'s own gate: every reduce renderer except
    /// the tiled `simdgroup_matrix` GEMM path funnels its output write
    /// through `push_reduce_epilogue_write`, so a fused
    /// `proxima_tensor::BoundOpKind::Reduce::epilogue_body` renders there;
    /// the tiled path has no such hook yet, so a non-identity epilogue on a
    /// shape that would otherwise take it is rejected here, named, rather
    /// than silently dropping the fused work.
    #[error("node {node} cannot render its fused reduce epilogue: {reason}")]
    EpilogueNotSupported { node: NodeId, reason: &'static str },

    /// A keep-specific renderer (`render_reduce`/`render_scan`/
    /// `pack_reduce_uniforms`/`pack_scan_uniforms` and their wgpu/wgsl/cuda
    /// counterparts) re-destructures a `resolved.kind` its own caller already
    /// narrowed to one `Keep` variant -- this fires only if that upstream
    /// guarantee itself broke, so it names an internal contract break rather
    /// than a caller-supplied program's shape. No existing variant describes
    /// that class: `UnsupportedOpKind`/`CudaUnsupportedOpKind` reject a
    /// caller's `BoundOpKind`, not a renderer's own precondition.
    #[error("node {node} reached a {expected} renderer with kind {found}")]
    RenderKindMismatch {
        node: NodeId,
        expected: &'static str,
        found: &'static str,
    },

    /// A cooperative-reduce combine helper (`shuffle_combine_expr`/
    /// `cooperative_identity_token`/`subgroup_combine_fn` across the cuda and
    /// wgsl backends, and `simd_combine_fn`/`cooperative_identity_token` here
    /// in the metal `msl` backend) reached with a `reduce_op` its own
    /// caller's `is_cooperative_reduce_op`/`reduce_is_cooperative` gate
    /// should have excluded already -- the same internal-contract class as
    /// [`Self::RenderKindMismatch`], scoped to the op axis instead of the
    /// kind axis, so it earns its own variant rather than overloading that
    /// one with an unrelated `found` field.
    #[error(
        "node {node} reached a cooperative-reduce combine with op {op}, which is not associative-commutative"
    )]
    NonCooperativeReduceOp { node: NodeId, op: &'static str },

    /// `classify_packed_row_block`'s `NotKQuantCodec` gate already rejects
    /// `Codec::Q8_0`/`Codec::Q4_0`/`Codec::Float16`/
    /// `Codec::BFloat16` before `packed_row_block` can ever return
    /// `Some` for one of them, so the row-blocked body's per-codec match
    /// never legitimately reaches one of these four arms.
    #[error("node {node} packed operand codec {codec} never reaches the row-blocked path")]
    NonKQuantCodec { node: NodeId, codec: &'static str },

    /// `classify_tiled_gemm`'s own `token_axes.is_empty() ||
    /// feature_axes.is_empty()` gate already rejects an empty group before
    /// returning `Some`, so `crate::msl::push_tiled_gemm_body` never
    /// legitimately observes an empty `group`.
    #[error("node {node} tiled-GEMM {group} axis group is empty")]
    EmptyAxisGroup { node: NodeId, group: &'static str },

    /// `classify_tiled_gemm` builds `token_axes`/`feature_axes` as a subset
    /// of `output_axes` by construction, so every axis in either group is
    /// guaranteed to appear in `output_axes` when the render side looks it
    /// back up.
    #[error("node {node} tiled-GEMM axis {axis} is not one of the op's output axes")]
    AxisNotInOutputAxes { node: NodeId, axis: u16 },

    /// `classify_tiled_gemm`'s `#[cfg(not(feature = "metal-tiled-gemm"))]`
    /// arm always returns `Err(TiledGemmRejection::FeatureDisabled)`, so no
    /// caller in that build ever holds a `TiledGemmBlock` to render or
    /// dispatch threadgroups for.
    #[error("node {node} reached the tiled-GEMM path without the metal-tiled-gemm feature")]
    TiledGemmFeatureDisabled { node: NodeId },

    /// The block-staged cached-attention body (`crate::msl::
    /// render_cached_attention`'s `block_width > 1` arm) reinterprets each
    /// lane's real/imaginary K and Q loads as `device const float4*` --
    /// legal only when `head_dim / 2` (the per-key real-plane element count)
    /// is a multiple of 4, which keeps every `qbase`/`kbase` offset a
    /// multiple of 4 floats (16 bytes) regardless of `kv_heads`/`query_row`.
    /// A `head_dim` this does not hold for is rejected here rather than
    /// emitting a `float4` load Metal would refuse to validate.
    #[error(
        "node {node} head_dim {head_dim} is not a multiple of 8, so the block-staged attention kernel's float4 K/Q loads are not 16-byte aligned"
    )]
    AttentionBlockMisaligned { node: NodeId, head_dim: u64 },

    /// The cached-attention merge kernel holds one split per simdgroup lane,
    /// so `[attention_splits].max` above `SIMD_WIDTH` would silently drop the
    /// splits past lane 31 from the running max and the normalizer. Rejected
    /// here, at bind time, rather than merging a wrong softmax.
    #[error(
        "node {node} attention_splits.max {max} exceeds the merge kernel's {limit} lanes, so splits past lane {limit} would be dropped from the softmax merge"
    )]
    AttentionSplitsExceedSimdWidth { node: NodeId, max: u64, limit: u64 },

    /// `multi_row_index32_active`'s own doc: the index32 body's `base_blocks
    /// = operand_base / block_elements` split is only exact (bit-identical
    /// to the wide/unsplit decode) when `operand_base % block_elements ==
    /// 0`. Every GGUF-sourced packed tensor satisfies this by construction
    /// (rows start at a block boundary), so this is a genuine invariant
    /// violation, not a routing decision -- checked once more here, at
    /// render time, as defense in depth against admission and render ever
    /// disagreeing (`push_packed_row_multi_row_body`'s own doc names the
    /// same posture for its other multi-row experiments).
    #[error(
        "node {node} packed weight operand_base {base} is not a multiple of block_elements {block_elements} -- PROXIMA_MULTI_ROW_INDEX32's base_blocks split requires a block-aligned tensor base"
    )]
    Index32OperandBaseNotBlockAligned {
        node: NodeId,
        base: i64,
        block_elements: usize,
    },
}
