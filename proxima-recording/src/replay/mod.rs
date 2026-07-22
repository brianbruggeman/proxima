//! Cassette replay-by-match-key (folded from `proxima-replay`):
//! [`ReplayUpstream`] serves a recorded HTTP interaction back by matching a
//! live request's method + path + query (+ optional body digest) against the
//! recording's index — no upstream call needed.
//!
//! The keying core ([`keying`], [`meta`]) is sans-IO and compiles at the
//! `alloc` tier; the cassette-loading adapter below (file I/O, `Runtime`,
//! the `Pipe`/`PipeFactory` surface) additionally needs `std`.

pub mod keying;
pub mod meta;
#[cfg(feature = "std")]
pub mod ws;

pub use keying::{MatchSpec, content_digest, match_key_from_request_with};
pub use meta::{CASSETTE_META_KIND, CassetteMeta};

#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(feature = "std")]
use std::future::Future;
#[cfg(feature = "std")]
use std::path::PathBuf;
#[cfg(feature = "std")]
use std::pin::Pin;
#[cfg(feature = "std")]
use std::sync::Arc;

#[cfg(feature = "std")]
use bytes::Bytes;
#[cfg(feature = "std")]
use futures::StreamExt;
#[cfg(feature = "std")]
use serde_json::Value;

#[cfg(feature = "std")]
use crate::event::{HttpEvent, InteractionId, ProtocolEvent, RecordingEvent};
#[cfg(feature = "std")]
use crate::factory::RecordingSourceRegistry;
#[cfg(feature = "std")]
use crate::jsonl::JsonlSource;
#[cfg(feature = "std")]
use crate::source::{DynRecordingSource, RecordingSource};
#[cfg(feature = "std")]
use proxima_core::ProximaError;
#[cfg(feature = "std")]
use proxima_primitives::pipe::body::ResponseStream;
#[cfg(feature = "std")]
use proxima_primitives::pipe::SendPipe;
#[cfg(feature = "std")]
use proxima_primitives::pipe::handler::{PipeHandle, into_handle};
#[cfg(feature = "std")]
use proxima_primitives::pipe::pipe_factory::PipeFactory;
#[cfg(feature = "std")]
use proxima_primitives::pipe::request::{Request, Response};
#[cfg(feature = "std")]
use proxima_runtime::Runtime;

#[cfg(feature = "std")]
use keying::match_key_from_recording;

#[cfg(feature = "std")]
#[derive(Debug, Clone)]
struct RecordedResponse {
    status: u16,
    headers: Vec<(String, String)>,
    chunks: Vec<Bytes>,
}

#[cfg(feature = "std")]
pub struct ReplayUpstream {
    label: String,
    by_match_key: HashMap<String, RecordedResponse>,
    source_path: PathBuf,
    match_spec: MatchSpec,
    meta: Option<CassetteMeta>,
}

#[cfg(feature = "replay-config")]
pub mod config;
#[cfg(feature = "replay-config")]
pub use config::ReplayConfig;

#[cfg(feature = "std")]
impl ReplayUpstream {
    pub async fn from_jsonl(
        path: impl Into<PathBuf>,
        label: impl Into<String>,
        runtime: Arc<dyn Runtime>,
    ) -> Result<Self, ProximaError> {
        Self::from_jsonl_with(path, label, runtime, MatchSpec::default()).await
    }

    pub async fn from_jsonl_with(
        path: impl Into<PathBuf>,
        label: impl Into<String>,
        runtime: Arc<dyn Runtime>,
        match_spec: MatchSpec,
    ) -> Result<Self, ProximaError> {
        let path = path.into();
        let source = JsonlSource::new(&path, runtime);
        let indexed = index_recording(&source, match_spec).await?;
        Ok(Self {
            label: label.into(),
            by_match_key: indexed.by_match_key,
            source_path: path,
            match_spec,
            meta: indexed.meta,
        })
    }

    /// Build a replay upstream from any `DynRecordingSource`. Used by the
    /// factory to plug in arbitrary registered formats (jsonl, bin, oram-
    /// wrapped, etc) without coupling the replay machinery to a single
    /// on-disk format.
    pub async fn from_source(
        source: DynRecordingSource,
        label: impl Into<String>,
    ) -> Result<Self, ProximaError> {
        Self::from_source_with(source, label, MatchSpec::default()).await
    }

    pub async fn from_source_with(
        source: DynRecordingSource,
        label: impl Into<String>,
        match_spec: MatchSpec,
    ) -> Result<Self, ProximaError> {
        let indexed = index_recording(source.as_ref(), match_spec).await?;
        Ok(Self {
            label: label.into(),
            by_match_key: indexed.by_match_key,
            source_path: PathBuf::new(),
            match_spec,
            meta: indexed.meta,
        })
    }

    /// This replay upstream's label, set at construction. Carries no
    /// runtime behaviour beyond identification (TARGET 3 — served-Pipe
    /// naming now lives at the mount-site label, not the handle).
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    #[must_use]
    pub fn known_keys(&self) -> Vec<String> {
        self.by_match_key.keys().cloned().collect()
    }

    #[must_use]
    pub fn source_path(&self) -> &std::path::Path {
        &self.source_path
    }

    /// Provenance stamp of the loaded recording, when present. Absent for
    /// recordings made before the stamp existed.
    #[must_use]
    pub fn meta(&self) -> Option<&CassetteMeta> {
        self.meta.as_ref()
    }
}

#[cfg(feature = "std")]
impl SendPipe for ReplayUpstream {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let key = match_key_from_request_with(&request, self.match_spec);
        let recorded = self.by_match_key.get(&key).cloned();
        async move {
            match recorded {
                Some(response) => Ok(rebuild_response(response)),
                None => Err(ProximaError::ReplayMiss { fingerprint: key }),
            }
        }
    }
}

#[cfg(feature = "std")]
fn rebuild_response(recorded: RecordedResponse) -> Response<Bytes> {
    let stream = futures::stream::iter(recorded.chunks.into_iter().map(Ok));
    let mut response = Response::new(recorded.status).with_stream(ResponseStream::new(stream));
    for (name, value) in recorded.headers {
        response = response.with_header(name, value);
    }
    response
}

#[cfg(feature = "std")]
struct IndexedRecording {
    by_match_key: HashMap<String, RecordedResponse>,
    meta: Option<CassetteMeta>,
}

#[cfg(feature = "std")]
async fn index_recording(
    source: &dyn RecordingSource,
    match_spec: MatchSpec,
) -> Result<IndexedRecording, ProximaError> {
    let mut in_flight: HashMap<InteractionId, RecordingDraft> = HashMap::new();
    let mut output: HashMap<String, RecordedResponse> = HashMap::new();
    let mut meta: Option<CassetteMeta> = None;
    let mut events = source.events();
    while let Some(event) = events.next().await {
        let RecordingEvent { id, event, .. } = event?;
        match event {
            ProtocolEvent::Http(HttpEvent::Started { request, .. }) => {
                let key = match_key_from_recording(&request);
                in_flight.insert(id, RecordingDraft::with_base_key(key));
            }
            ProtocolEvent::Http(HttpEvent::RequestChunk { data, .. }) => {
                if let Some(draft) = in_flight.get_mut(&id) {
                    draft.body_digest = keying::fnv1a64_fold(draft.body_digest, &data);
                }
            }
            ProtocolEvent::Http(HttpEvent::ResponseStarted { status, headers }) => {
                if let Some(draft) = in_flight.get_mut(&id) {
                    draft.status = Some(status);
                    draft.headers = headers;
                }
            }
            ProtocolEvent::Http(HttpEvent::ResponseChunk { data, .. }) => {
                if let Some(draft) = in_flight.get_mut(&id) {
                    draft.chunks.push(data);
                }
            }
            ProtocolEvent::Http(HttpEvent::Ended { .. }) => {
                if let Some(draft) = in_flight.remove(&id)
                    && let Some(status) = draft.status
                {
                    let key =
                        keying::finish_match_key(draft.base_key, draft.body_digest, match_spec);
                    output.insert(
                        key,
                        RecordedResponse {
                            status,
                            headers: draft.headers,
                            chunks: draft.chunks,
                        },
                    );
                }
            }
            ProtocolEvent::Custom { kind, payload } if kind == CASSETTE_META_KIND => {
                meta = Some(CassetteMeta::from_payload(&payload)?);
            }
            // request-ended carries no information for keying; non-HTTP
            // events (Pipeline, Process, other Custom) are unrelated to
            // HTTP replay and ignored.
            ProtocolEvent::Http(HttpEvent::RequestEnded)
            | ProtocolEvent::Pipeline(_)
            | ProtocolEvent::Process(_)
            | ProtocolEvent::Custom { .. } => {}
        }
    }
    Ok(IndexedRecording {
        by_match_key: output,
        meta,
    })
}

#[cfg(feature = "std")]
struct RecordingDraft {
    base_key: String,
    body_digest: u64,
    status: Option<u16>,
    headers: Vec<(String, String)>,
    chunks: Vec<Bytes>,
}

#[cfg(feature = "std")]
impl RecordingDraft {
    fn with_base_key(base_key: String) -> Self {
        Self {
            base_key,
            body_digest: keying::FNV_OFFSET_BASIS,
            status: None,
            headers: Vec::new(),
            chunks: Vec::new(),
        }
    }
}

#[cfg(feature = "std")]
pub struct ReplayPipeFactory {
    sources: Arc<RecordingSourceRegistry>,
}

#[cfg(feature = "std")]
impl ReplayPipeFactory {
    #[must_use]
    pub fn new(sources: Arc<RecordingSourceRegistry>) -> Self {
        Self { sources }
    }
}

#[cfg(feature = "std")]
impl PipeFactory for ReplayPipeFactory {
    fn name(&self) -> &str {
        "replay"
    }

    fn build(
        &self,
        spec: &Value,
        _inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        let sources = self.sources.clone();
        Box::pin(async move {
            let label = spec
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("replay")
                .to_string();
            // resolve the source through the registry so plugin formats
            // (oram-wrapped, bin, custom binary, etc) work without code changes here.
            let source = sources.resolve(&spec).await?;
            let upstream = ReplayUpstream::from_source(source, label).await?;
            Ok(into_handle(upstream))
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::event::{RecordMeta, RecordingEvent, RequestHeader};
    use crate::{Format, JsonFormat};
    use bytes::Bytes;
    use prime::os::runtime::PrimeRuntime;
    use tempfile::tempdir;
    use time::OffsetDateTime;

    fn prime() -> Arc<dyn Runtime> {
        Arc::new(PrimeRuntime::new(1).expect("prime"))
    }

    fn fixture_id(seed: u8) -> InteractionId {
        InteractionId::from_bytes([seed; 16])
    }

    async fn write_fixture(path: &std::path::Path) {
        let id = fixture_id(7);
        let events = vec![
            RecordingEvent {
                id,
                ts_ms: 0,
                parent: None,
                event: ProtocolEvent::Http(HttpEvent::Started {
                    ts: OffsetDateTime::UNIX_EPOCH,
                    pipe: "echo".into(),
                    request: RequestHeader {
                        method: "POST".into(),
                        path: "/v1/chat".into(),
                        headers: BTreeMap::new(),
                        query: [("model".to_string(), "model-a".to_string())]
                            .into_iter()
                            .collect(),
                    },
                    meta: None,
                }),
            },
            RecordingEvent {
                id,
                ts_ms: 10,
                parent: None,
                event: ProtocolEvent::Http(HttpEvent::ResponseStarted {
                    status: 200,
                    headers: vec![("content-type".into(), "application/json".into())],
                }),
            },
            RecordingEvent {
                id,
                ts_ms: 11,
                parent: None,
                event: ProtocolEvent::Http(HttpEvent::ResponseChunk {
                    data: Bytes::from_static(b"{\"ok\":1}"),
                    metadata: Default::default(),
                }),
            },
            RecordingEvent {
                id,
                ts_ms: 50,
                parent: None,
                event: ProtocolEvent::Http(HttpEvent::Ended {
                    latency_ms: 40,
                    meta: RecordMeta::default(),
                }),
            },
        ];
        // write the fixture through the real JSONL codec (one enveloped JSON
        // object per line) — the exact bytes JsonlSource reads back.
        let bytes = JsonFormat::new()
            .encode_block(events)
            .expect("encode jsonl");
        tokio::fs::write(path, bytes).await.expect("write fixture");
    }

    #[proxima::test(runtime = "tokio")]
    async fn replay_match_returns_recorded_response() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("echo.jsonl");
        write_fixture(&path).await;
        let upstream = ReplayUpstream::from_jsonl(&path, "replay", prime())
            .await
            .expect("load");
        let request = Request::builder()
            .method("POST")
            .path("/v1/chat")
            .query_param("model", "model-a")
            .build()
            .expect("builder");
        let response = upstream.call(request).await.expect("call");
        assert_eq!(response.status, 200);
        let body = response.collect_body().await.expect("collect");
        assert_eq!(&body[..], b"{\"ok\":1}");
    }

    #[proxima::test(runtime = "tokio")]
    async fn replay_miss_returns_typed_error() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("echo.jsonl");
        write_fixture(&path).await;
        let upstream = ReplayUpstream::from_jsonl(&path, "replay", prime())
            .await
            .expect("load");
        let request = Request::builder()
            .method("POST")
            .path("/different/path")
            .build()
            .expect("builder");
        let outcome = upstream.call(request).await;
        assert!(matches!(outcome, Err(ProximaError::ReplayMiss { .. })));
    }
}
