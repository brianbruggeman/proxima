use proxima_tensor::NodeId;
use proxima_tensor::TensorError;
use proxima_tensor::op::ScalarOp;

/// Every fault [`crate::adjoint::differentiate`] (or [`crate::sparse`]'s
/// helpers) can raise.
///
/// Each variant names the node and the shape of the program that defeated
/// it, mirroring [`proxima_tensor::TensorError`]'s own convention
/// (`proxima-tensor/src/error.rs:12-13`) of naming the offending node rather
/// than a bare "invalid program". Not `Copy`: it carries a `TensorError`,
/// which is `Clone` but not `Copy` (that type's own derive,
/// `proxima-tensor/src/error.rs:12`).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AutogradError {
    #[error("node {0} does not exist in the program handed to differentiate")]
    UnknownLoss(NodeId),

    #[error("loss node {node} has rank {rank}, not 0 -- differentiate needs a scalar loss")]
    LossNotScalar { node: NodeId, rank: usize },

    #[error(
        "node {node} is a Keep::Scan reduce outside the covered shape -- only ScalarOp::Add \
         over a rank-1 identity in_map/out_map (a plain cumulative sum, the reversed-suffix-sum \
         derivation in crate::adjoint::differentiate_scan) has an adjoint here; a non-Add scan \
         body (no known closed form for Maximum/Minimum), a non-identity map (strided/windowed \
         scan), or a rank > 1 iteration space is not implemented"
    )]
    ScanAdjointUnsupported { node: NodeId },

    #[error(
        "node {node} reduces with {body:?}, which has no adjoint rule here -- only \
         Add (broadcast), Multiply (divide-form: gradient * output / operand), and \
         Maximum/Minimum (masked routing to the argmax/argmin) are implemented; those \
         four are exactly this crate's is_associative reduce bodies, and a \
         non-associative body like Subtract or Divide has no well-defined reduce \
         adjoint here"
    )]
    UnsupportedReduceBody { node: NodeId, body: ScalarOp },

    #[error("node {node} elementwise-computes {body:?}, which has no local adjoint rule here")]
    UnsupportedElementwiseBody { node: NodeId, body: ScalarOp },

    #[error(
        "node {node} operand {operand} is read through an affine map with a \
         non-unit coefficient or more than one term per axis (a convolution-style \
         window) -- reusing that map as a Reduce out_map is rejected by \
         proxima-tensor/src/shape.rs:437-453 (\"reduce output maps must be pure \
         projections in v1\"), so this adjoint is not lowerable in this crate's \
         algebra today"
    )]
    NonProjectionOperandMap { node: NodeId, operand: NodeId },

    #[error(
        "node {node} operand {operand} is read through a gather (IndexMap::Computed) \
         whose index_map is not a pure projection -- this adjoint cannot line up \
         each gathered row with the index that selected it, so the compact \
         GatheredContribution this crate would otherwise hand back cannot be built"
    )]
    NonProjectionIndexMap { node: NodeId, operand: NodeId },

    #[error(
        "node {node}'s Reduce::in_map reads operand {operand} through a gather \
         (IndexMap::Computed) -- reducing directly over a gathered operand needs a \
         different derivation (reusing that in_map as this Reduce's own adjoint \
         out_map would itself be data-dependent, which \
         proxima-tensor/src/shape.rs:166-171 rejects at evaluation time with no \
         adjoint-specific diagnosis) and is not implemented; route the gather \
         through a separate Elementwise(Identity) node first"
    )]
    ReduceOverGatherUnsupported { node: NodeId, operand: NodeId },

    /// Forward scatter's adjoint IS a gather of the output gradient at the
    /// same destination indices the forward pass wrote to
    /// (`proxima-autograd/src/adjoint.rs`'s `differentiate_reduce` scatter
    /// arm derives and tests this for `body: Add`, the only body a colliding
    /// write reduces with unambiguously here). Any other body raised this:
    /// `Maximum`/`Minimum` would need to know, per destination, WHICH
    /// colliding source position actually won (this crate's own
    /// `Reduce(Maximum)` adjoint above needs the reduce's already-computed
    /// output for exactly that reason, and a scatter's output at a given
    /// destination is not a function this crate can invert to one source
    /// position without extra state the forward op does not carry);
    /// `Multiply`'s divide-form rule divides by the *other* colliding
    /// contributions' product, not a single input, so it does not reduce to
    /// the same "gather the numerator back" shape either. Named and rejected
    /// rather than silently misderived.
    #[error(
        "node {node}'s Reduce::out_map is data-dependent (a scatter) with body {body:?}; only \
         Add has a derived adjoint (a gather of the output gradient at the same destination \
         indices) -- see AutogradError::ScatterOutputUnsupported's own doc for why the other \
         reduce bodies do not reduce to that same shape"
    )]
    ScatterOutputUnsupported { node: NodeId, body: ScalarOp },

    #[error(
        "sparse row buffers disagree: {found} values for {row_len} elements per row \
         do not divide evenly by index count"
    )]
    SparseRowLengthMismatch { row_len: usize, found: usize },

    #[error("differentiate could not infer shapes for the program handed to it: {0}")]
    ShapeInference(#[from] TensorError),
}
