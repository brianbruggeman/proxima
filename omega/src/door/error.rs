use super::reader::Reader;
use super::writer::Writer;
use crate::backend::BackendError;

/// The eight stages of the door in pipeline order: reading is parse,
/// validate, lower; writing is emit, package, verify; running is plan,
/// execute. A [`DoorError`] names the stage that raised it with this tag, and
/// [`DoorStage::kind`](super::DoorStage::kind) names the stage that produced a
/// state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StageKind {
    Parse,
    Validate,
    Lower,
    Emit,
    Package,
    Verify,
    Plan,
    Execute,
}

impl StageKind {
    /// Every stage, in pipeline order.
    pub const ALL: [Self; 8] = [
        Self::Parse,
        Self::Validate,
        Self::Lower,
        Self::Emit,
        Self::Package,
        Self::Verify,
        Self::Plan,
        Self::Execute,
    ];

    /// Lowercase stage name as it appears in error messages.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Parse => "parse",
            Self::Validate => "validate",
            Self::Lower => "lower",
            Self::Emit => "emit",
            Self::Package => "package",
            Self::Verify => "verify",
            Self::Plan => "plan",
            Self::Execute => "execute",
        }
    }
}

/// A door failure: one variant per stage, carrying that stage's own error, so
/// the caller learns where the call failed from one type.
///
/// The reader's and writer's errors are the `#[source]` of their stages; the
/// backend's error is the source of plan and execute.
#[derive(Debug, thiserror::Error)]
pub enum DoorError<R: Reader, W: Writer> {
    #[error("parse stage failed: {0}")]
    Parse(#[source] R::ParseError),

    #[error("validate stage failed: {0}")]
    Validate(#[source] R::ValidateError),

    #[error("lower stage failed: {0}")]
    Lower(#[source] R::LowerError),

    #[error("emit stage failed: {0}")]
    Emit(#[source] W::Error),

    #[error("package stage failed: {0}")]
    Package(#[source] W::Error),

    #[error("verify stage failed: {0}")]
    Verify(#[source] W::Error),

    #[error("plan stage failed: {0}")]
    Plan(#[source] BackendError),

    #[error("execute stage failed: {0}")]
    Execute(#[source] BackendError),
}

impl<R: Reader, W: Writer> DoorError<R, W> {
    /// The stage that raised this error.
    #[must_use]
    pub const fn stage(&self) -> StageKind {
        match self {
            Self::Parse(_) => StageKind::Parse,
            Self::Validate(_) => StageKind::Validate,
            Self::Lower(_) => StageKind::Lower,
            Self::Emit(_) => StageKind::Emit,
            Self::Package(_) => StageKind::Package,
            Self::Verify(_) => StageKind::Verify,
            Self::Plan(_) => StageKind::Plan,
            Self::Execute(_) => StageKind::Execute,
        }
    }
}
