use proxima_tensor::NodeId;

/// Everything [`crate::msl::emit`] can reject.
///
/// A [`proxima_tensor::Nest`] cannot itself encode a gather or scatter — its
/// operand and output addressing ([`proxima_tensor::StridedView`]) is pure
/// affine base+stride arithmetic with no indices field at all, and
/// `proxima_tensor::shape::infer` already rejects every data-dependent index
/// map before a `Nest` is ever built (see `infer_fold`'s
/// `out_map.is_data_dependent()` check and `unify_iteration_space`'s same
/// check over every zip/fold operand). So nothing here re-checks that: every
/// variant below guards against a malformed `Nest` built directly through its
/// public, all-`pub`-field struct literal — never against something
/// `nest::lower` itself would produce.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EmitError {
    #[error("node {node} zip body takes {expected} operands but the nest carries {found}")]
    ArityMismatch {
        node: NodeId,
        expected: usize,
        found: usize,
    },

    /// `cpu::apply_scalar_op` always calls a fold's reduction body with
    /// exactly two operands (`[accumulator, value]`); a reduction body that
    /// reads a third (`ScalarOp::Select`) reads past that slice there too —
    /// this rejects it at emit time instead of indexing out of bounds in the
    /// generated MSL.
    #[error(
        "node {node} reduction body is select, which reads a third operand a fold step never supplies"
    )]
    ReductionBodyIsSelect { node: NodeId },

    #[error(
        "node {node} is a keep::all scan over zero iteration dims, which has no folded dim to scan along"
    )]
    EmptyScan { node: NodeId },
}
