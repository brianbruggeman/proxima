use crate::pmem::PmemError;

/// Errors from the pmem DAX facade.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DaxError {
    /// The leaf layout/FSM rejected the operation (bad slot length, region too
    /// small, value-length mismatch).
    #[error("pmem layout: {0}")]
    Pmem(#[from] PmemError),

    /// A filesystem operation (open, set_len, metadata) failed.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// `mmap` of the region failed.
    #[cfg(target_os = "linux")]
    #[error("mmap failed: {0}")]
    Mmap(rustix::io::Errno),

    /// `msync` of a file-backed mapping failed during `persist`.
    #[cfg(target_os = "linux")]
    #[error("msync failed: {0}")]
    Msync(rustix::io::Errno),
}
