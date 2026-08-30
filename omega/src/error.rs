use proxima_tensor::{DType, NodeId};

/// Everything [`crate::msl::emit`] can reject.
///
/// A [`proxima_tensor::BoundOp`] node cannot itself encode a gather or
/// scatter — its operand and output addressing
/// ([`proxima_tensor::Layout`]) is pure affine base+stride arithmetic
/// with no indices field at all, and `proxima_tensor::shape::infer` already
/// rejects every data-dependent index map before a `BoundOp` node is ever
/// built (see `infer_reduce`'s `out_map.is_data_dependent()` check and
/// `unify_iteration_space`'s same check over every elementwise/reduce
/// operand). So nothing here re-checks that: every variant below guards
/// against a malformed `BoundOp` node built directly through its public,
/// all-`pub`-field struct literal — never against something `bind::bind`
/// itself would produce.
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
