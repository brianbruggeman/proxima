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

    #[error(
        "node {node} elementwise-computes {body:?}, which has no local adjoint rule here"
    )]
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

    #[error(
        "node {node}'s Reduce::out_map is data-dependent (a scatter); \
         proxima-tensor/src/shape.rs:166-171 already rejects this program at \
         shape-inference time, so it never reaches an adjoint"
    )]
    ScatterOutputUnsupported { node: NodeId },

    #[error(
        "sparse row buffers disagree: {found} values for {row_len} elements per row \
         do not divide evenly by index count"
    )]
    SparseRowLengthMismatch { row_len: usize, found: usize },

    #[error("differentiate could not infer shapes for the program handed to it: {0}")]
    ShapeInference(#[from] TensorError),
}
