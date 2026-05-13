use core::fmt;

/// Decode failure for a wire view: the caller buffer was too short for the
/// fixed header, or a length field pointed past the available bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// Buffer shorter than the minimum fixed header for this layer.
    Truncated { need: usize, got: usize },
    /// A header length field (IHL, data offset) is below its legal minimum.
    BadHeaderLen { field: u8 },
    /// Version nibble did not match the expected protocol version.
    BadVersion { found: u8 },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { need, got } => {
                write!(formatter, "truncated: need {need} bytes, got {got}")
            }
            Self::BadHeaderLen { field } => write!(formatter, "bad header length field {field}"),
            Self::BadVersion { found } => write!(formatter, "bad version nibble {found}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DecodeError {}
