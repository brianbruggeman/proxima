use core::fmt;

#[cfg(feature = "std")]
use alloc::string::String;

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Full,
    InvalidInput,
    /// A background OS thread (e.g. the console drain pump) failed to spawn.
    /// Carries the OS error message so the log line explains itself.
    #[cfg(feature = "std")]
    ThreadSpawn(String),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full => formatter.write_str("ring full"),
            Self::InvalidInput => formatter.write_str("invalid input"),
            #[cfg(feature = "std")]
            Self::ThreadSpawn(message) => write!(formatter, "thread spawn failed: {message}"),
        }
    }
}

impl core::error::Error for Error {}

impl From<proxima_core::ring::CapacityError> for Error {
    fn from(_: proxima_core::ring::CapacityError) -> Self {
        Self::InvalidInput
    }
}
