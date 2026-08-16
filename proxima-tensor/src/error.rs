use crate::dtype::DType;
use crate::expr::NodeId;

/// Every fault [`shape::infer`](crate::shape::infer) and the rest of the
/// crate can raise, from a malformed program to a spec that will not parse.
///
/// Every structural variant names the node it was found at. A program that
/// infers has in-range backwards references, consistent arity, an
/// accumulator wide enough for its fold, and resolvable shapes — which is
/// what lets [`nest::lower`](crate::nest::lower) and
/// [`cpu::evaluate`](crate::cpu::evaluate) walk it without re-checking.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TensorError {
    #[error("node {0} references node {1}, which is not defined yet")]
    NodeOutOfRange(NodeId, NodeId),

    #[error("node {0} references itself")]
    Cycle(NodeId),

    #[error("node {node} has {found} operands but its body takes {expected}")]
    ArityMismatch {
        node: NodeId,
        found: usize,
        expected: usize,
    },

    #[error("node {0} is a zip with no operands")]
    EmptyZip(NodeId),

    #[error("node {node} folds {element:?} into {accumulator:?}, which overflows")]
    NarrowAccumulator {
        node: NodeId,
        element: DType,
        accumulator: DType,
    },

    #[error("node {node} projects iteration dim {dim}, which its map does not declare")]
    IterDimOutOfRange { node: NodeId, dim: u16 },

    #[error("program has no expressions")]
    Empty,

    #[error("symbol `?{symbol}` is not bound in the provided symbol table")]
    UnboundSymbol { symbol: u16 },

    #[error("node {node} disagrees with itself on iteration dim {dim}: {left} vs {right}")]
    ExtentMismatch {
        node: NodeId,
        dim: u16,
        left: u64,
        right: u64,
    },

    #[error("node {node} iteration dim {dim} has no operand that constrains it")]
    UnconstrainedDim { node: NodeId, dim: u16 },

    #[error("node {node} dim {dim} reaches an index outside the operand's extent")]
    IndexOutOfBounds { node: NodeId, dim: u16 },

    #[error("node {node} cannot be lowered: {reason}")]
    NotLowerable { node: NodeId, reason: &'static str },

    #[error("output {0} does not name a node in the program")]
    UnknownOutput(NodeId),

    #[error("expected {expected} block operands but got {found}")]
    BlockCountMismatch { expected: usize, found: usize },

    #[error("node {node} block has {found} elements but its shape needs {expected}")]
    BlockSizeMismatch {
        node: NodeId,
        expected: usize,
        found: usize,
    },

    #[error("map `{0}` is not `operand->iteration` notation")]
    MalformedMap(alloc::string::String),

    #[error("map `{notation}` projects `{letter}`, which the iteration space lacks")]
    UnknownIndexLetter {
        notation: alloc::string::String,
        letter: char,
    },

    #[error("spec references node `{0}` before it is defined")]
    UnknownNode(alloc::string::String),

    #[error("node `{node}` has {inputs} inputs but {maps} maps")]
    SpecArityMismatch {
        node: alloc::string::String,
        inputs: usize,
        maps: usize,
    },

    #[error("extent `{0}` is not a number or a `?n` symbol")]
    MalformedExtent(alloc::string::String),

    #[error(
        "node {node} gathers indices from a node with dtype {dtype:?}, which is not an integer type"
    )]
    NonIntegerIndices { node: NodeId, dtype: DType },

    #[error("node {node} gathered_dim {dim} is out of range for the operand's rank")]
    GatheredDimOutOfRange { node: NodeId, dim: u16 },

    #[error("node {node} gather fetched index {index}, which is out of range for extent {extent}")]
    GatherIndexOutOfRange {
        node: NodeId,
        index: i64,
        extent: u64,
    },

    /// Gather indices ride in f32 buffers (see [`crate::map::IndexMap::Computed`]
    /// and `cpu.rs`'s module docs for why), and f32's 24-bit mantissa cannot
    /// represent every integer above `2^24` (16,777,216) exactly — a gathered
    /// dim wider than that could silently address the wrong row. Lifting this
    /// ceiling means adding a separate integer-buffer path; until then, this
    /// is a loud rejection instead of a silent wrong answer.
    #[error(
        "node {node} gathers a dim with extent {extent}, past 2^24 (16777216), the largest \
         integer an f32 index can represent exactly"
    )]
    GatherExtentExceedsExactFloat { node: NodeId, extent: u64 },
}
