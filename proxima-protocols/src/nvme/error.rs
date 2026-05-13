use core::fmt;

/// Decode failure for an NVMe queue-entry view: the caller buffer was too short
/// for a fixed 64-byte SQE / 16-byte CQE slot, or a queue was constructed with a
/// depth outside the spec-legal range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// Buffer shorter than the fixed entry size for this slot.
    Truncated { need: usize, got: usize },
    /// Queue depth outside the NVMe-legal 2..=65536 entries (the doorbell is a
    /// 16-bit index, and a queue needs at least one full and one empty slot).
    BadQueueDepth { depth: u32 },
    /// A resumed ring cursor sits at or past the queue depth — an index the wrap
    /// arithmetic can never produce.
    BadCursor { cursor: u16, depth: u32 },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { need, got } => {
                write!(formatter, "truncated: need {need} bytes, got {got}")
            }
            Self::BadQueueDepth { depth } => {
                write!(formatter, "bad queue depth {depth}, must be 2..=65536")
            }
            Self::BadCursor { cursor, depth } => {
                write!(
                    formatter,
                    "bad ring cursor {cursor}, must be < depth {depth}"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DecodeError {}
