//! Static file upstream rooted at a directory. `..` and out-of-root
//! symlinks reject; missing files return 404; Content-Type is inferred
//! from the file extension.

use std::future::Future;
use std::path::{Component, Path, PathBuf};
use std::pin::Pin;

use bon::Builder;
use bytes::Bytes;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use proxima_primitives::pipe::SendPipe;

use crate::error::ProximaError;
use crate::pipe::{PipeHandle, into_handle};
use crate::pipe_factory::PipeFactory;
use crate::request::{Request, Response};

pub struct FsUpstream {
    label: String,
    root: PathBuf,
    index: String,
}

impl FsUpstream {
    pub fn new(label: impl Into<String>, root: PathBuf, index: String) -> Self {
        Self {
            label: label.into(),
            root,
            index,
        }
    }

    /// This upstream's label, set at construction (TARGET 3 — served-Pipe
    /// naming now lives at the mount-site label, not the handle).
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }
}

impl SendPipe for FsUpstream {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        let root = self.root.clone();
        let index = self.index.clone();
        async move {
            let path = std::str::from_utf8(&request.path).unwrap_or("/");
            let resolved = match resolve_path(&root, path, &index) {
                Ok(p) => p,
                Err(err) => return Ok(err),
            };
            // synchronous, not tokio::fs::read: every served pipe is wrapped
            // in proxima_listen::Offload (see proxima-http's `listener` module), which
            // drives this future via futures::executor::block_on on a
            // background-pool thread with no tokio reactor. tokio::fs::read
            // internally spawn_blocking()s onto a tokio runtime that isn't
            // there and panics; resolve_path's canonicalize() calls above are
            // already synchronous for the same reason.
            match std::fs::read(&resolved) {
                Ok(bytes) => {
                    let content_type = content_type_for(&resolved);
                    Ok(Response::new(200)
                        .with_header("content-type", content_type)
                        .with_body(Bytes::from(bytes)))
                }
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(not_found()),
                Err(err) => Ok(Response::new(500)
                    .with_body(Bytes::from(format!("fs read: {err}").into_bytes()))),
            }
        }
    }
}


// reject .. components and absolute paths from the request before
// joining onto root. canonicalize the result and reject anything
// that escaped root (symlink-out attack).
fn resolve_path(root: &Path, request_path: &str, index: &str) -> Result<PathBuf, Response<Bytes>> {
    let trimmed = request_path.trim_start_matches('/');
    let candidate = Path::new(trimmed);
    for component in candidate.components() {
        match component {
            Component::Normal(_) => {}
            _ => return Err(forbidden()),
        }
    }
    let mut joined = root.to_path_buf();
    joined.push(candidate);
    if joined.is_dir() {
        joined.push(index);
    }
    let canonical = joined.canonicalize().map_err(|err| {
        if err.kind() == std::io::ErrorKind::NotFound {
            not_found()
        } else {
            Response::new(500)
                .with_body(Bytes::from(format!("fs canonicalize: {err}").into_bytes()))
        }
    })?;
    let canonical_root = root.canonicalize().map_err(|err| {
        Response::new(500).with_body(Bytes::from(
            format!("fs canonicalize root: {err}").into_bytes(),
        ))
    })?;
    if !canonical.starts_with(&canonical_root) {
        return Err(forbidden());
    }
    Ok(canonical)
}

fn content_type_for(path: &Path) -> &'static str {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("html" | "htm") => "text/html; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("js" | "mjs") => "application/javascript; charset=utf-8",
        Some("json") => "application/json; charset=utf-8",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("svg") => "image/svg+xml",
        Some("webp") => "image/webp",
        Some("ico") => "image/x-icon",
        Some("txt" | "md") => "text/plain; charset=utf-8",
        Some("wasm") => "application/wasm",
        Some("woff") => "font/woff",
        Some("woff2") => "font/woff2",
        Some("xml") => "application/xml; charset=utf-8",
        _ => "application/octet-stream",
    }
}

fn not_found() -> Response<Bytes> {
    Response::new(404).with_body(Bytes::from_static(b"not found"))
}

fn forbidden() -> Response<Bytes> {
    Response::new(403).with_body(Bytes::from_static(b"forbidden"))
}

/// Typed config surface for [`FsUpstream`] — the static-file upstream.
#[derive(Debug, Clone, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_FS")]
#[builder(derive(Clone, Debug), on(String, into))]
pub struct FsConfig {
    /// Handler label. `label` is a serde alias for `name`.
    #[setting(default = "fs")]
    #[serde(default = "default_label", alias = "label")]
    #[builder(default = default_label())]
    pub name: String,

    /// Filesystem root the upstream serves from. Required.
    #[setting(default)]
    #[serde(default)]
    #[builder(default)]
    pub root: String,

    /// Index file served when the resolved path is a directory.
    #[setting(default = "index.html")]
    #[serde(default = "default_index")]
    #[builder(default = default_index())]
    pub index: String,
}

fn default_label() -> String {
    "fs".to_string()
}

fn default_index() -> String {
    "index.html".to_string()
}

impl Validate for FsConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.root.is_empty() {
            errors.push(ValidationMessage::new(
                "root",
                "fs upstream requires `root`",
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl FsConfig {
    /// Materialise this config into a runtime [`FsUpstream`].
    pub fn from_config(self) -> Result<FsUpstream, ProximaError> {
        self.validate()
            .map_err(|err| ProximaError::Config(format!("{err}")))?;
        Ok(FsUpstream::new(
            self.name,
            PathBuf::from(self.root),
            self.index,
        ))
    }
}

pub struct FsPipeFactory;

impl PipeFactory for FsPipeFactory {
    fn name(&self) -> &str {
        "fs"
    }

    fn build(
        &self,
        spec: &Value,
        _inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        Box::pin(async move {
            let config: FsConfig = serde_json::from_value(spec)
                .map_err(|err| ProximaError::Config(format!("fs config: {err}")))?;
            Ok(into_handle(config.from_config()?))
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    async fn build_request(path: &str) -> Request<Bytes> {
        Request::builder()
            .method("GET")
            .path(path)
            .build()
            .expect("builder")
    }

    // principle-4 parity: the fluent builder and the config value must lower to
    // identical FsUpstream state.
    #[test]
    fn parity_fluent_builder_and_config_value_match() {
        let from_value: FsConfig = serde_json::from_value(json!({
            "name": "assets",
            "root": "/srv/www",
            "index": "home.html",
        }))
        .expect("from_value");
        let from_value = from_value.from_config().expect("from_config value");

        let from_builder = FsConfig::builder()
            .name("assets")
            .root("/srv/www")
            .index("home.html")
            .build()
            .from_config()
            .expect("from_config builder");

        assert_eq!(from_value.label, from_builder.label);
        assert_eq!(from_value.root, from_builder.root);
        assert_eq!(from_value.index, from_builder.index);
    }

    #[proxima::test]
    async fn serves_existing_file() {
        let dir = tempdir().expect("tempdir");
        let file = dir.path().join("hello.txt");
        std::fs::write(&file, b"hello fs\n").expect("write");

        let factory = FsPipeFactory;
        let handle = factory
            .build(&json!({"root": dir.path().to_str().unwrap()}), None)
            .await
            .expect("build");

        let response = SendPipe::call(&handle, build_request("/hello.txt").await)
            .await
            .expect("call");
        assert_eq!(response.status, 200);
        assert_eq!(
            response.metadata.get_str("content-type"),
            Some("text/plain; charset=utf-8")
        );
        let body = response.collect_body().await.expect("collect");
        assert_eq!(&body[..], b"hello fs\n");
    }

    #[proxima::test]
    async fn missing_file_404s() {
        let dir = tempdir().expect("tempdir");
        let factory = FsPipeFactory;
        let handle = factory
            .build(&json!({"root": dir.path().to_str().unwrap()}), None)
            .await
            .expect("build");
        let response = SendPipe::call(&handle, build_request("/nope").await)
            .await
            .expect("call");
        assert_eq!(response.status, 404);
    }

    #[proxima::test]
    async fn directory_traversal_is_rejected() {
        let dir = tempdir().expect("tempdir");
        let factory = FsPipeFactory;
        let handle = factory
            .build(&json!({"root": dir.path().to_str().unwrap()}), None)
            .await
            .expect("build");
        let response = SendPipe::call(&handle, build_request("/../etc/passwd").await)
            .await
            .expect("call");
        assert_eq!(response.status, 403);
    }

    #[proxima::test]
    async fn directory_resolves_to_index() {
        let dir = tempdir().expect("tempdir");
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).expect("mkdir");
        std::fs::write(sub.join("index.html"), b"<html></html>").expect("write");

        let factory = FsPipeFactory;
        let handle = factory
            .build(&json!({"root": dir.path().to_str().unwrap()}), None)
            .await
            .expect("build");
        let response = SendPipe::call(&handle, build_request("/sub").await)
            .await
            .expect("call");
        assert_eq!(response.status, 200);
        assert_eq!(
            response.metadata.get_str("content-type"),
            Some("text/html; charset=utf-8")
        );
    }
}
