use core::fmt;

/// Decode failure for a split-virtqueue view: the caller buffer was too short
/// for the descriptor table / avail ring / used ring at the given queue size,
/// the queue size itself was illegal, or a descriptor chain walk found more
/// links than the queue could ever legally hold (a cyclic or adversarial
/// `next` chain — VIRTIO 1.2 spec §2.7.5 requires `queue_size` to bound chain
/// length).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// Buffer shorter than the fixed-size structure this view needs.
    Truncated { need: usize, got: usize },
    /// Queue size must be a nonzero power of two (spec §2.7.5 / §2.7.6).
    BadQueueSize { queue_size: u16 },
    /// Chain walk exceeded `queue_size` steps without terminating — the guest
    /// published a cyclic or malformed `next` chain.
    ChainTooLong { limit: u16 },
    /// The driver asked for more descriptors than are free (spec §2.7.13.1:
    /// a driver must not publish a chain it has no descriptors for).
    QueueFull { needed: usize, free: u16 },
    /// A chain with no buffers at all cannot be published.
    EmptyChain,
    /// The device completed a head index that is out of range, or that
    /// completes more descriptors than the driver has outstanding.
    BadUsedId { id: u32, queue_size: u16 },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { need, got } => {
                write!(formatter, "truncated: need {need} bytes, got {got}")
            }
            Self::BadQueueSize { queue_size } => {
                write!(
                    formatter,
                    "bad queue size {queue_size}, must be a nonzero power of two"
                )
            }
            Self::ChainTooLong { limit } => {
                write!(formatter, "descriptor chain exceeds queue size {limit}")
            }
            Self::QueueFull { needed, free } => {
                write!(formatter, "need {needed} descriptors, only {free} free")
            }
            Self::EmptyChain => write!(formatter, "descriptor chain has no buffers"),
            Self::BadUsedId { id, queue_size } => {
                write!(
                    formatter,
                    "device completed head {id}, not outstanding in a queue of {queue_size}"
                )
            }
        }
    }
}

impl core::error::Error for DecodeError {}
