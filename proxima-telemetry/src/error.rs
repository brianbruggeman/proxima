#[cfg(feature = "std")]
use alloc::string::String;

// derived by path, not a `use`: this enum is itself named `Error`.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("ring full")]
    Full,
    #[error("invalid input")]
    InvalidInput,
    /// A background OS thread (e.g. the console drain pump) failed to spawn.
    /// Carries the OS error message so the log line explains itself.
    #[cfg(feature = "std")]
    #[error("thread spawn failed: {0}")]
    ThreadSpawn(String),
    /// Bridging `tracing::` events into a recorder failed because a global
    /// `tracing` subscriber was already installed elsewhere in the process.
    #[cfg(feature = "tracing-init")]
    #[error("global tracing subscriber already set: {0}")]
    GlobalSubscriberAlreadySet(String),
}

impl From<proxima_core::ring::CapacityError> for Error {
    fn from(_: proxima_core::ring::CapacityError) -> Self {
        Self::InvalidInput
    }
}
