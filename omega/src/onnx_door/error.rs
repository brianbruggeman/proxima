use proxima_onnx::{LowerError, OnnxError};

use crate::backend::BackendError;

/// The five stages of the door, ordered as they run. A [`DoorError`] names
/// the stage that raised it with this tag, and
/// [`DoorStage::kind`](super::DoorStage::kind) names the stage that produced a
/// state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StageKind {
    Parse,
    Validate,
    Lower,
    Plan,
    Execute,
}

impl StageKind {
    /// Lowercase stage name as it appears in error messages.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Parse => "parse",
            Self::Validate => "validate",
            Self::Lower => "lower",
            Self::Plan => "plan",
            Self::Execute => "execute",
        }
    }
}

/// Why the validate stage refused a graph before paying for lowering.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ValidationError {
    /// The model parsed but its `graph` field is absent.
    #[error("model carries no graph")]
    NoGraph,

    /// Every reason the graph cannot lower, gathered in one pass so a caller
    /// fixes them together instead of one lowering error at a time.
    #[error(
        "graph cannot lower: unlowerable ops (name, nodes) {unlowerable:?}, unpinned dimensions {unpinned:?}"
    )]
    Rejected {
        /// Op types with no lowering arm and their node counts, e.g.
        /// `[("DequantizeLinear", 12)]`.
        unlowerable: Vec<(String, usize)>,
        /// Symbolic input dimension names with no entry in the batch pins,
        /// e.g. `["batch_size", "sequence_length"]`.
        unpinned: Vec<String>,
    },
}

/// A door failure: one variant per stage, carrying that stage's own error, so
/// the caller learns where the call failed from one type.
///
/// This is the single error [`OnnxDoor`](super::OnnxDoor) returns; the four
/// error types it replaces ([`OnnxError`], [`LowerError`], [`BackendError`]
/// twice) stay reachable as the `#[source]` of each variant.
#[derive(Debug, thiserror::Error)]
pub enum DoorError {
    #[error("parse stage failed: {0}")]
    Parse(#[source] OnnxError),

    #[error("validate stage failed: {0}")]
    Validate(#[source] ValidationError),

    #[error("lower stage failed: {0}")]
    Lower(#[source] LowerError),

    #[error("plan stage failed: {0}")]
    Plan(#[source] BackendError),

    #[error("execute stage failed: {0}")]
    Execute(#[source] BackendError),
}

impl DoorError {
    /// The stage that raised this error.
    #[must_use]
    pub const fn stage(&self) -> StageKind {
        match self {
            Self::Parse(_) => StageKind::Parse,
            Self::Validate(_) => StageKind::Validate,
            Self::Lower(_) => StageKind::Lower,
            Self::Plan(_) => StageKind::Plan,
            Self::Execute(_) => StageKind::Execute,
        }
    }
}
