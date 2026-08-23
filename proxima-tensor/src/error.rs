use crate::dtype::DType;
use crate::op::{NodeId, ScalarOp};

/// Every fault [`shape::infer`](crate::shape::infer) and the rest of the
/// crate can raise, from a malformed program to a spec that will not parse.
///
/// Every structural variant names the node it was found at. A program that
/// infers has in-range backwards references, consistent arity, an
/// accumulator wide enough for its reduce, and resolvable shapes — which is
/// what lets [`bind::bind`](crate::bind::bind) and
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

    #[error("node {0} is an elementwise op with no operands")]
    EmptyElementwise(NodeId),

    #[error("node {node} reduces {element:?} into {accumulator:?}, which overflows")]
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

    #[error("node {node} cannot be bound to an executable op: {reason}")]
    NotLowerable { node: NodeId, reason: &'static str },

    #[error("output {0} does not name a node in the program")]
    UnknownOutput(NodeId),

    #[error("expected {expected} input operands but got {found}")]
    InputCountMismatch { expected: usize, found: usize },

    /// [`cpu::evaluate_named`](crate::cpu::evaluate_named) binds `Op::Input`
    /// by name, so an input with no name (`Op::Input::name` is `None`) has
    /// nothing to bind by.
    #[error("node {0} is an input with no name; evaluate_named binds by name")]
    UnnamedInput(NodeId),

    /// A name [`cpu::evaluate_named`](crate::cpu::evaluate_named)'s program
    /// requires was not present in the caller-supplied `named` bindings.
    #[error("no binding supplied for input `{0}`")]
    UnboundInputName(alloc::string::String),

    #[error("node {node} input has {found} elements but its shape needs {expected}")]
    InputSizeMismatch {
        node: NodeId,
        expected: usize,
        found: usize,
    },

    // these five carry free-form parsed text and are only ever constructed
    // by `spec.rs`, which is itself `config`-only (std+alloc); scoping them
    // to `config` keeps `TensorError` alloc-free outside that tier instead
    // of dragging `alloc::string::String` into every build that can never
    // produce these variants.
    #[cfg(feature = "config")]
    #[error("map `{0}` is not `operand->iteration` notation")]
    MalformedMap(alloc::string::String),

    #[cfg(feature = "config")]
    #[error("map `{notation}` projects `{letter}`, which the iteration space lacks")]
    UnknownIndexLetter {
        notation: alloc::string::String,
        letter: char,
    },

    #[cfg(feature = "config")]
    #[error("spec references node `{0}` before it is defined")]
    UnknownNode(alloc::string::String),

    #[cfg(feature = "config")]
    #[error("node `{node}` has {inputs} inputs but {maps} maps")]
    SpecArityMismatch {
        node: alloc::string::String,
        inputs: usize,
        maps: usize,
    },

    #[cfg(feature = "config")]
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

    /// A chunk of a threaded nest never completed: the background pool
    /// caught and discarded a worker panic (see
    /// `prime::os::background::worker`'s `catch_unwind`) rather than
    /// resuming it, so this is the closest sound signal a caller gets — a
    /// scheduling failure, not a fault in the tensor program.
    #[error("nest chunk {chunk} of a threaded run did not complete: {reason}")]
    ThreadedChunkFailed {
        chunk: usize,
        reason: alloc::string::String,
    },

    /// The background thread pool backing a threaded nest could not be
    /// built (OS thread-spawn failure — resource exhaustion).
    #[error("nest thread pool unavailable: {0}")]
    ThreadedPoolUnavailable(alloc::string::String),

    /// The typed elementwise interpreter ([`crate::cpu::evaluate_typed`])
    /// rejects a body/dtype combination it cannot execute correctly: a
    /// transcendental (`exp`/`ln`/`sqrt`/`tanh`/`reciprocal`) on an integer
    /// dtype, or [`ScalarOp::Negate`] on an unsigned dtype (no representable
    /// negative). Raised at the node that names the bad combination, not a
    /// blanket rejection of the whole program.
    #[error("node {node} applies {op:?} to dtype {dtype:?}, which does not support it")]
    UnsupportedScalarOp {
        node: NodeId,
        op: ScalarOp,
        dtype: DType,
    },

    /// An integer [`ScalarOp::Divide`] hit a divisor of zero, or the one
    /// signed overflow case (`T::MIN / -1`) `checked_div` also refuses —
    /// unlike float division, neither has a representable result.
    #[error("node {node} integer division is undefined (division by zero, or T::MIN / -1)")]
    CheckedDivisionFailed { node: NodeId },

    /// [`cpu::matmul_q4k_f32`](crate::cpu::matmul_q4k_f32)'s row byte length
    /// is not a whole multiple of `Q4_K`'s packed super-block size, or the
    /// activation length does not match the declared reduction width.
    #[error("quantized matmul shape mismatch: {reason}")]
    QuantizedShapeMismatch { reason: &'static str },

    /// [`crate::spec::mistral_forward_program`]'s routed-FFN branch needs
    /// `1 <= expert_used_count <= expert_count`: zero experts selected per
    /// token is a config that can never route, and selecting more experts
    /// than exist has no meaning.
    #[error(
        "moe routing needs 1 <= expert_used_count <= expert_count, got expert_count={expert_count} \
         expert_used_count={expert_used_count}"
    )]
    InvalidExpertConfig {
        expert_count: u32,
        expert_used_count: u32,
    },

    /// [`crate::align::AlignedBuffer::new`]'s requested element count does
    /// not fit a `usize` byte length, or the caller-supplied `page_size` is
    /// not a valid `Layout` alignment (not a power of two, or the rounded
    /// byte length overflows `isize::MAX`) — a caller-controlled allocation
    /// request, never a plain out-of-memory condition.
    #[error("aligned buffer allocation rejected: {reason}")]
    AlignedAllocationRejected { reason: &'static str },
}
