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

    /// `wgsl::emit_wgsl`'s v1 op set is elementwise, `Keep::Reduce`, and
    /// `Keep::Scan` only — `Iota`/`Constant` have no renderer yet.
    #[cfg(feature = "wgpu-backend")]
    #[error("node {node} is a {kind} op, which the wgsl v1 emitter does not support yet")]
    UnsupportedOpKind { node: NodeId, kind: &'static str },

    /// `cuda::emit_cuda`'s op set is elementwise, `Keep::Reduce`, and
    /// `Keep::Scan` only — `Iota`/`Constant` have no renderer yet, the same
    /// v1 boundary [`Self::UnsupportedOpKind`] draws for the wgsl emitter.
    #[cfg(feature = "cuda")]
    #[error("node {node} is a {kind} op, which the cuda emitter does not support yet")]
    CudaUnsupportedOpKind { node: NodeId, kind: &'static str },
}
