use std::io;
use std::path::PathBuf;

use proxima_core::ProximaError;

#[derive(Debug, thiserror::Error)]
pub enum OciError {
    #[error("image reference `{reference}` is invalid: {reason}")]
    Reference { reference: String, reason: String },

    #[error("digest `{digest}` is invalid: {reason}")]
    Digest { digest: String, reason: String },

    #[error("{what} failed its digest check: expected {expected}, actual {actual}")]
    DigestMismatch { what: String, expected: String, actual: String },

    #[error("fetch of `{url}` failed: {reason}")]
    Transport { url: String, reason: String },

    #[error("registry answered {status} for `{url}`: {body}")]
    Status { url: String, status: u16, body: String },

    #[error("registry auth for `{url}` failed: {reason}")]
    Auth { url: String, reason: String },

    #[error("{what} is not valid json: {source}")]
    Json {
        what: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("no manifest for {os}/{architecture} in the index; it lists [{available}]")]
    NoPlatform { os: String, architecture: String, available: String },

    #[error("layer {digest} has media type `{media_type}`, which the flattener does not unpack")]
    LayerMediaType { digest: String, media_type: String },

    #[error("layer {digest} is not a readable tar archive: {source}")]
    Layer {
        digest: String,
        #[source]
        source: io::Error,
    },

    #[error("image entry `{path}` cannot be placed in the rootfs: {reason}")]
    Entry { path: String, reason: String },

    #[error("cache at `{path}`: {source}")]
    Cache {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("image config has no entrypoint and no cmd; there is nothing to run")]
    NothingToRun,

    #[error("prime runtime: {0}")]
    Runtime(String),
}

impl From<OciError> for ProximaError {
    fn from(error: OciError) -> Self {
        Self::Upstream(error.to_string())
    }
}
