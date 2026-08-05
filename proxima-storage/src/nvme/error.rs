use proxima_protocols::nvme::DecodeError;

/// Failure from driving a queue pair. Today only the codec/ring layer can fail;
/// the completion future polls cooperatively until the controller posts (a hung
/// command is a higher-layer NVMe Abort + watchdog concern, not a reap timeout).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum NvmeError {
    /// The codec rejected a slot or a ring cursor.
    #[error("nvme codec: {0}")]
    Codec(#[from] DecodeError),
}
