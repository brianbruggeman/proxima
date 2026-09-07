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
    /// `PackedCodec::Q3K` is metal-only so far (`wgpu_driver::packed_operands_of`
    /// already routes a `QuantizedBlock::Q3K` node to `None` for this same
    /// reason); this is the typed rejection a caller who somehow threads a
    /// `Some(PackedCodec::Q3K)` through directly still hits, rather than a
    /// generated `wgsl` calling a function that does not exist.
    #[cfg(feature = "wgpu-backend")]
    #[error("node {node} reads a Q3_K operand, which the wgsl emitter does not support yet")]
    UnsupportedPackedCodec { node: NodeId },

    /// The cuda counterpart of [`Self::UnsupportedPackedCodec`] -- same gap,
    /// same reason.
    #[cfg(feature = "cuda")]
    #[error("node {node} reads a Q3_K operand, which the cuda emitter does not support yet")]
    CudaUnsupportedPackedCodec { node: NodeId },

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
    /// `PackedCodec::Q8_0`/`PackedCodec::Q4_0`/`PackedCodec::Float16`/
    /// `PackedCodec::BFloat16` before `packed_row_block` can ever return
    /// `Some` for one of them, so the row-blocked body's per-codec match
    /// never legitimately reaches one of these four arms.
    #[error("node {node} packed operand codec {codec} never reaches the row-blocked path")]
    NonKQuantPackedCodec { node: NodeId, codec: &'static str },

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
}
