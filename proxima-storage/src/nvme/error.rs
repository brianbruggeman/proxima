use core::fmt;
use proxima_protocols::nvme::DecodeError;

/// Failure from driving a queue pair. Today only the codec/ring layer can fail;
/// the completion future polls cooperatively until the controller posts (a hung
/// command is a higher-layer NVMe Abort + watchdog concern, not a reap timeout).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvmeError {
    /// The codec rejected a slot or a ring cursor.
    Codec(DecodeError),
}

impl From<DecodeError> for NvmeError {
    fn from(error: DecodeError) -> Self {
        Self::Codec(error)
    }
}

impl fmt::Display for NvmeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec(error) => write!(formatter, "nvme codec: {error}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for NvmeError {}
