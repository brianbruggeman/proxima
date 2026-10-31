use std::io;

#[derive(Debug, thiserror::Error)]
pub enum KvmError {
    #[error("{operation}: {source}")]
    Syscall {
        operation: &'static str,
        #[source]
        source: io::Error,
    },
    #[error("{0}")]
    Guest(String),
}

impl KvmError {
    #[must_use]
    pub fn syscall(operation: &'static str, source: io::Error) -> Self {
        Self::Syscall { operation, source }
    }

    #[must_use]
    pub fn last_os(operation: &'static str) -> Self {
        Self::syscall(operation, io::Error::last_os_error())
    }

    #[must_use]
    pub fn guest(message: impl Into<String>) -> Self {
        Self::Guest(message.into())
    }

    #[must_use]
    pub fn errno(&self) -> i32 {
        match self {
            Self::Syscall { source, .. } => source.raw_os_error().unwrap_or(libc::EIO),
            Self::Guest(_) => libc::EIO,
        }
    }

    #[must_use]
    pub fn from_ioctl(operation: &'static str, error: kvm_ioctls::Error) -> Self {
        Self::syscall(operation, io::Error::from_raw_os_error(error.errno()))
    }
}

pub type KvmResult<T> = Result<T, KvmError>;
