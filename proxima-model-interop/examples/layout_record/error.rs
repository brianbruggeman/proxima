use std::io;
use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum RecordError {
    #[error("usage: layout_record --granite <gguf> --e2b <gguf> --out <dir>")]
    Usage,
    #[error("io on {path}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("gguf header of {path} did not parse within 512 MiB")]
    Header { path: PathBuf },
    #[error("no tensor of type {kind} matching {fragment:?} in {path}")]
    NoTensor {
        path: PathBuf,
        kind: String,
        fragment: String,
    },
    #[error("tensor {tensor} holds fewer than {needed} bytes of the requested rows")]
    Short { tensor: String, needed: usize },
    #[error("incumbent {stage} failed for {codec}: {reason}")]
    Quant {
        stage: &'static str,
        codec: String,
        reason: String,
    },
    #[error("recorded {found} codec variants, expected {expected}")]
    CodecCount { found: usize, expected: usize },
}

impl RecordError {
    pub fn io(path: &std::path::Path) -> impl FnOnce(io::Error) -> Self + '_ {
        move |source| Self::Io {
            path: path.to_path_buf(),
            source,
        }
    }

    pub fn quant(
        stage: &'static str,
        codec: impl std::fmt::Debug,
        reason: impl std::fmt::Debug,
    ) -> Self {
        Self::Quant {
            stage,
            codec: format!("{codec:?}"),
            reason: format!("{reason:?}"),
        }
    }
}
