use proxima_tensor::NodeId;
use proxima_tensor::TensorError;
use proxima_tensor::op::ScalarOp;

/// Every fault [`crate::adjoint::differentiate`] can raise.
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
        "node {node} is a Keep::Scan reduce; its adjoint is a distinct derivation \
         (reversed prefix-sum for Add, no known closed form for Maximum/Minimum) \
         and is not implemented"
    )]
    ScanAdjointUnsupported { node: NodeId },

    #[error(
        "node {node} reduces with {body:?}, which has no adjoint rule here -- only \
         Add (broadcast) and Maximum/Minimum (masked routing to the argmax/argmin) \
         are implemented, matching this crate's is_associative bodies"
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
        "node {node} operand {operand} is read through a gather (IndexMap::Computed); \
         its adjoint is a scatter-add via mask composition \
         (proxima-tensor/src/cpu.rs:16062) but that composition is \
         O(destination x source) dense -- at embedding scale (vocab 128k x \
         4k updates) that is 524M mask elements to accumulate 4k values, so \
         it is rejected here rather than shipped unverified"
    )]
    GatherAdjointUnsupported { node: NodeId, operand: NodeId },

    #[error(
        "node {node}'s Reduce::out_map is data-dependent (a scatter); \
         proxima-tensor/src/shape.rs:166-171 already rejects this program at \
         shape-inference time, so it never reaches an adjoint"
    )]
    ScatterOutputUnsupported { node: NodeId },

    #[error("differentiate could not infer shapes for the program handed to it: {0}")]
    ShapeInference(#[from] TensorError),
}
