use std::io;
use std::path::PathBuf;

/// Why the on-disk PTX store could not serve a request.
///
/// Callers treat every variant as a miss; the store never panics and never
/// fails a run on its own.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("kernel cache name `{name}` is not a non-empty hex string")]
    InvalidName { name: String },

    #[error("{operation} {path} failed: {source}")]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

impl StoreError {
    pub(super) fn io(operation: &'static str, path: PathBuf, source: io::Error) -> Self {
        Self::Io {
            operation,
            path,
            source,
        }
    }
}
