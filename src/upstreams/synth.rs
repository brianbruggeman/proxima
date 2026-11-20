use std::collections::BTreeMap;
use std::convert::Infallible;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use bon::Builder;
use bytes::Bytes;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::body::ResponseStream;
use proxima_primitives::pipe::SendPipe;

use crate::error::ProximaError;
use crate::pipe::{PipeHandle, into_handle};
use crate::pipe_factory::PipeFactory;
use crate::request::{Request, Response};
#[cfg(any(feature = "http1", feature = "http-prime-deps"))]
use crate::templates::{TemplateContext, expand, expand_value};

enum SynthBody {
    Buffered(Bytes),
    Streamed(ResponseStream),
}

impl SynthBody {
    fn apply(self, response: Response<Bytes>) -> Response<Bytes> {
        match self {
            SynthBody::Buffered(bytes) => response.with_body(bytes),
            SynthBody::Streamed(stream) => response.with_stream(stream),
        }
    }
}

#[derive(Debug, Clone)]
struct ChunkSpec {
    ts_ms: u64,
    bytes: Bytes,
}

#[derive(Debug, Clone)]
enum BodySource {
    Static(Bytes),
    Template(Value),
    Chunks(Vec<ChunkSpec>),
}

pub struct SynthUpstream {
    label: String,
    status: u16,
    headers: BTreeMap<String, String>,
    body: BodySource,
    delay_chunks: bool,
}

impl SynthUpstream {
    pub fn new(label: impl Into<String>, status: u16, body: impl Into<Bytes>) -> Self {
        Self {
            label: label.into(),
            status,
            headers: BTreeMap::new(),
            body: BodySource::Static(body.into()),
            delay_chunks: false,
        }
    }

    /// This upstream's label, set at construction (TARGET 3 — served-Pipe
    /// naming now lives at the mount-site label, not the handle).
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Returns the exact body length if it's known up front
    /// (`Static`, or non-delayed `Chunks`). Streaming sources
    /// (`Template`, delayed `Chunks`) return `None` because the
    /// length depends on per-request expansion / timing.
    fn static_body_len(&self) -> Option<usize> {
        match &self.body {
            BodySource::Static(bytes) => Some(bytes.len()),
            BodySource::Chunks(chunks) if !self.delay_chunks => {
                Some(chunks.iter().map(|chunk| chunk.bytes.len()).sum())
            }
            _ => None,
        }
    }

    fn non_template_body(&self) -> Option<SynthBody> {
        match &self.body {
            BodySource::Static(bytes) => Some(SynthBody::Buffered(bytes.clone())),
            BodySource::Template(_) => None,
            BodySource::Chunks(chunks) => {
                if !self.delay_chunks {
                    let plain: Vec<Bytes> =
                        chunks.iter().map(|chunk| chunk.bytes.clone()).collect();
                    let stream =
                        ResponseStream::new(futures::stream::iter(plain.into_iter().map(Ok)));
                    return Some(SynthBody::Streamed(stream));
                }
                let chunk_specs = chunks.clone();
                let stream = async_stream::stream! {
                    let mut prior_ts_ms = 0_u64;
                    for chunk in chunk_specs.into_iter() {
                        let delta = chunk.ts_ms.saturating_sub(prior_ts_ms);
                        if delta > 0 {
                            proxima_core::time::sleep(Duration::from_millis(delta)).await;
                        }
                        prior_ts_ms = chunk.ts_ms;
                        yield Ok(chunk.bytes);
                    }
                };
                Some(SynthBody::Streamed(ResponseStream::new(stream)))
            }
        }
    }
}

impl SendPipe for SynthUpstream {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        let status = self.status;
        let headers = self.headers.clone();
        let pre_body = self.non_template_body();
        let template = match &self.body {
            BodySource::Template(value) => Some(value.clone()),
            _ => None,
        };
        let static_body_len = self.static_body_len();
        async move {
            let body = match (pre_body, template) {
                (Some(body), _) => body,
                (None, Some(template_value)) => {
                    #[cfg(any(feature = "http1", feature = "http-prime-deps"))]
                    {
                        expand_template_body(&template_value, request).await?
                    }
                    #[cfg(not(any(feature = "http1", feature = "http-prime-deps")))]
                    {
                        let _ = (template_value, request);
                        return Err(ProximaError::Config(
                            "synth template body requires the `http1` or \
                             `http-prime-deps` feature (templates engine)"
                                .into(),
                        ));
                    }
                }
                (None, None) => SynthBody::Buffered(Bytes::new()),
            };
            let mut response = body.apply(Response::new(status));
            // Static + known-length bodies get a Content-Length header
            // emitted automatically. Without it, the listener would
            // default to chunked transfer-encoding for responses with
            // no length declared, which is correct but unnecessarily
            // burdens simple clients (and breaks the load-test
            // client that assumes a fixed-length body). For
            // streaming bodies (BodySource::Chunks / Template) we
            // skip — the length isn't known up front.
            let user_supplied_content_length = headers
                .keys()
                .any(|name| name.eq_ignore_ascii_case("content-length"));
            if !user_supplied_content_length && let Some(length) = static_body_len {
                response = response.with_header(
                    proxima_primitives::pipe::HeaderName::ContentLength,
                    length.to_string(),
                );
            }
            for (name, value) in headers {
                response = response.with_header(name, value);
            }
            Ok(response)
        }
    }
}

#[cfg(any(feature = "http1", feature = "http-prime-deps"))]
async fn expand_template_body(
    template: &Value,
    request: Request<Bytes>,
) -> Result<SynthBody, ProximaError> {
    let references_body = match template {
        Value::String(text) => text.contains("{{body"),
        structured => structured.to_string().contains("{{body"),
    };
    // only materialize the request body when the template references it —
    // otherwise leave it for downstream middleware / upstreams.
    let (request, body_value) = if references_body {
        let (request, buffered) = request.body_bytes().await?;
        let value = if buffered.is_empty() {
            None
        } else {
            serde_json::from_slice::<Value>(&buffered).ok()
        };
        (request, value)
    } else {
        (request, None)
    };
    let trace_id_str = request
        .context
        .trace_id
        .as_deref()
        .and_then(|bytes| std::str::from_utf8(bytes).ok());
    let pipe_str = request
        .context
        .pipe_label
        .as_deref()
        .and_then(|bytes| std::str::from_utf8(bytes).ok());
    let context = TemplateContext {
        request_id: None,
        trace_id: trace_id_str,
        pipe: pipe_str,
        body: body_value.as_ref(),
    };
    let bytes = match template {
        Value::String(text) => expand(text, &context).into_bytes(),
        structured => value_bytes(&expand_value(structured, &context))?,
    };
    Ok(SynthBody::Buffered(Bytes::from(bytes)))
}

fn value_bytes(value: &Value) -> Result<Vec<u8>, ProximaError> {
    serde_json::to_vec(value)
        .map_err(|err| ProximaError::Config(format!("synth structured body: {err}")))
}

fn parse_env_text(raw: &str) -> Result<Option<Value>, Infallible> {
    Ok(Some(Value::String(raw.to_string())))
}

/// Base64-encoded timed chunk in a synth streaming body — the config twin of
/// `ChunkSpec` (private). `ts_ms` is the absolute offset from the start of the response;
/// deltas drive the inter-chunk sleep when `delay_chunks` is set.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkConfig {
    #[serde(default)]
    pub ts_ms: u64,
    /// base64 of the raw chunk bytes (e.g. `YWI=` for `ab`).
    pub b64: String,
}

/// Typed config surface for [`SynthUpstream`] — the canned-response upstream.
///
/// The body is resolved in priority order matching the historical hand-parser:
/// `body_chunks` (streaming) > `body_template` (per-request expansion) >
/// `body` (static string) > empty. `name`/`label` are aliases for the pipe
/// label.
#[derive(Debug, Clone, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_SYNTH")]
#[builder(derive(Clone, Debug), on(String, into))]
pub struct SynthConfig {
    /// Handler label. `label` is a serde alias for `name`.
    #[setting(default = "synth")]
    #[serde(default = "default_label", alias = "label")]
    #[builder(default = default_label())]
    pub name: String,

    /// HTTP status code the synth response carries.
    #[setting(default = 200)]
    #[serde(default = "default_status")]
    #[builder(default = default_status())]
    pub status: u16,

    /// Response headers, applied verbatim. Values must be strings.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub headers: BTreeMap<String, String>,

    /// Static body — lowest body priority. A string is sent as text; a map or
    /// list is sent as compact JSON. The environment always sets text.
    #[setting(default, resolve_with = "parse_env_text")]
    #[serde(default)]
    #[builder(into)]
    pub body: Option<Value>,

    /// Per-request template body (e.g. `trace={{trace_id}}`) — middle priority.
    /// A map or list has each string leaf expanded, then is sent as compact JSON.
    #[setting(default, resolve_with = "parse_env_text")]
    #[serde(default)]
    #[builder(into)]
    pub body_template: Option<Value>,

    /// Timed base64 chunks — highest priority; produces a streaming body.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub body_chunks: Vec<ChunkConfig>,

    /// Sleep between chunks by `ts_ms` delta when streaming `body_chunks`.
    #[setting(default)]
    #[serde(default)]
    #[builder(default)]
    pub delay_chunks: bool,
}

fn default_label() -> String {
    "synth".to_string()
}

fn default_status() -> u16 {
    200
}

impl Validate for SynthConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self
            .body_chunks
            .iter()
            .any(|chunk| BASE64.decode(&chunk.b64).is_err())
        {
            errors.push(ValidationMessage::new(
                "body_chunks",
                "each entry's `b64` must be valid base64",
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl SynthConfig {
    /// Resolve the priority-ordered body source. Errors on invalid base64 in a
    /// chunk's `b64` field (same contract as the historical hand-parser).
    fn resolve_body(&self) -> Result<BodySource, ProximaError> {
        if !self.body_chunks.is_empty() {
            let mut parsed: Vec<ChunkSpec> = Vec::with_capacity(self.body_chunks.len());
            for chunk in &self.body_chunks {
                let decoded = BASE64
                    .decode(&chunk.b64)
                    .map_err(|err| ProximaError::Config(format!("synth body_chunk b64: {err}")))?;
                parsed.push(ChunkSpec {
                    ts_ms: chunk.ts_ms,
                    bytes: Bytes::from(decoded),
                });
            }
            return Ok(BodySource::Chunks(parsed));
        }
        if let Some(template) = &self.body_template {
            return Ok(BodySource::Template(template.clone()));
        }
        if let Some(body) = &self.body {
            let bytes = match body {
                Value::String(text) => text.clone().into_bytes(),
                structured => value_bytes(structured)?,
            };
            return Ok(BodySource::Static(Bytes::from(bytes)));
        }
        Ok(BodySource::Static(Bytes::new()))
    }

    /// Materialise this config into a runtime [`SynthUpstream`].
    pub fn from_config(self) -> Result<SynthUpstream, ProximaError> {
        self.validate()
            .map_err(|err| ProximaError::Config(format!("{err}")))?;
        let body = self.resolve_body()?;
        Ok(SynthUpstream {
            label: self.name,
            status: self.status,
            headers: self.headers,
            body,
            delay_chunks: self.delay_chunks,
        })
    }
}

pub struct SynthPipeFactory;

impl PipeFactory for SynthPipeFactory {
    fn name(&self) -> &str {
        "synth"
    }

    fn build(
        &self,
        spec: &Value,
        _inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        Box::pin(async move {
            let config: SynthConfig = serde_json::from_value(spec)
                .map_err(|err| ProximaError::Config(format!("synth config: {err}")))?;
            Ok(into_handle(config.from_config()?))
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    #[proxima::test]
    async fn synth_returns_static_body() {
        let factory = SynthPipeFactory;
        let handle = factory
            .build(&json!({"status": 201, "body": "hello"}), None)
            .await
            .expect("build");
        let request = Request::builder()
            .method("GET")
            .path("/")
            .build()
            .expect("builder");
        let response = SendPipe::call(&handle, request).await.expect("call");
        assert_eq!(response.status, 201);
        let body = response.collect_body().await.expect("collect");
        assert_eq!(&body[..], b"hello");
    }

    #[proxima::test]
    async fn synth_template_expands_trace_id() {
        let factory = SynthPipeFactory;
        let handle = factory
            .build(
                &json!({"status": 200, "body_template": "trace={{trace_id}}"}),
                None,
            )
            .await
            .expect("build");
        let mut request = Request::builder()
            .method("GET")
            .path("/")
            .build()
            .expect("builder");
        request.context.trace_id = Some(std::sync::Arc::from(b"01ARZ".as_slice()));
        let response = SendPipe::call(&handle, request).await.expect("call");
        let body = response.collect_body().await.expect("collect");
        assert_eq!(&body[..], b"trace=01ARZ");
    }

    #[proxima::test]
    async fn synth_chunks_yield_in_order_without_delay() {
        let factory = SynthPipeFactory;
        // base64 of "ab" and "cd"
        let spec = json!({
            "status": 200,
            "body_chunks": [
                {"ts_ms": 0, "b64": "YWI="},
                {"ts_ms": 0, "b64": "Y2Q="},
            ],
        });
        let handle = factory.build(&spec, None).await.expect("build");
        let request = Request::builder()
            .method("GET")
            .path("/")
            .build()
            .expect("builder");
        let response = SendPipe::call(&handle, request).await.expect("call");
        let mut chunk_stream = response.into_chunk_stream();
        let mut joined: Vec<u8> = Vec::new();
        while let Some(chunk) = futures::StreamExt::next(&mut chunk_stream).await {
            joined.extend_from_slice(&chunk.expect("chunk"));
        }
        assert_eq!(&joined[..], b"abcd");
    }

    #[proxima::test]
    async fn synth_template_expands_body_field_ref() {
        let factory = SynthPipeFactory;
        let handle = factory
            .build(
                &json!({
                    "status": 200,
                    "body_template": "hello, {{body.name}}",
                }),
                None,
            )
            .await
            .expect("build");
        let request = Request::builder()
            .method("POST")
            .path("/")
            .body(Bytes::from_static(br#"{"name":"brian"}"#))
            .build()
            .expect("builder");
        let response = SendPipe::call(&handle, request).await.expect("call");
        let body = response.collect_body().await.expect("collect");
        assert_eq!(&body[..], b"hello, brian");
    }

    #[proxima::test]
    async fn synth_template_missing_body_field_renders_empty() {
        let factory = SynthPipeFactory;
        let handle = factory
            .build(
                &json!({
                    "status": 200,
                    "body_template": "hello, {{body.name}}",
                }),
                None,
            )
            .await
            .expect("build");
        let request = Request::builder()
            .method("POST")
            .path("/")
            .body(Bytes::from_static(b"{}"))
            .build()
            .expect("builder");
        let response = SendPipe::call(&handle, request).await.expect("call");
        let body = response.collect_body().await.expect("collect");
        assert_eq!(&body[..], b"hello, ");
    }

    // principle-4 parity: the fluent builder and the config value must lower to
    // identical SynthUpstream state.
    #[test]
    fn parity_fluent_builder_and_config_value_match() {
        let from_value: SynthConfig = serde_json::from_value(json!({
            "name": "canned",
            "status": 503,
            "headers": {"retry-after": "5"},
            "body": "unavailable",
            "delay_chunks": false,
        }))
        .expect("from_value");
        let from_value = from_value.from_config().expect("from_config value");

        let from_builder = SynthConfig::builder()
            .name("canned")
            .status(503)
            .headers(BTreeMap::from([(
                "retry-after".to_string(),
                "5".to_string(),
            )]))
            .body("unavailable")
            .build()
            .from_config()
            .expect("from_config builder");

        assert_eq!(from_value.label, from_builder.label);
        assert_eq!(from_value.status, from_builder.status);
        assert_eq!(from_value.headers, from_builder.headers);
        assert_eq!(from_value.delay_chunks, from_builder.delay_chunks);
        assert!(matches!(
            (&from_value.body, &from_builder.body),
            (BodySource::Static(left), BodySource::Static(right)) if left == right
        ));
    }

    #[proxima::test]
    async fn synth_headers_are_propagated() {
        let factory = SynthPipeFactory;
        let handle = factory
            .build(
                &json!({
                    "status": 200,
                    "body": "ok",
                    "headers": {"x-custom": "value"},
                }),
                None,
            )
            .await
            .expect("build");
        let request = Request::builder()
            .method("GET")
            .path("/")
            .build()
            .expect("builder");
        let response = SendPipe::call(&handle, request).await.expect("call");
        assert_eq!(response.metadata.get_str("x-custom"), Some("value"));
    }

    fn check_body_json(body: &[u8], expected: &Value) -> Result<(), String> {
        let parsed: Value = serde_json::from_slice(body).map_err(|err| format!("not json: {err}"))?;
        if &parsed == expected {
            Ok(())
        } else {
            Err(format!("expected {expected}, got {parsed}"))
        }
    }

    fn y1() -> Value {
        json!({"title": "split stage body", "body": "adds StageBody"})
    }

    fn y2() -> Value {
        json!({"content": [{"type": "text", "text": "review of {{body.prompt}}"}]})
    }

    fn y3() -> Bytes {
        Bytes::from_static(br#"{"prompt":"PROXIMA-24: adds StageBody"}"#)
    }

    fn y4() -> Value {
        json!({"content": [{"type": "text", "text": "review of PROXIMA-24: adds StageBody"}]})
    }

    async fn respond(spec: Value, request_body: Bytes) -> Vec<u8> {
        let handle = SynthPipeFactory.build(&spec, None).await.expect("build");
        let request = Request::builder()
            .method("POST")
            .path("/")
            .body(request_body)
            .build()
            .expect("builder");
        let response = SendPipe::call(&handle, request).await.expect("call");
        response.collect_body().await.expect("collect").to_vec()
    }

    #[proxima::test]
    async fn synth_payload_structured_body_is_compact_json() {
        let answer = respond(json!({"body": y1()}), Bytes::new()).await;
        check_body_json(&answer, &y1()).expect("body json");
    }

    #[proxima::test]
    async fn synth_payload_structured_template_expands_nested_string() {
        let answer = respond(json!({"body_template": y2()}), y3()).await;
        check_body_json(&answer, &y4()).expect("template json");
    }

    #[proxima::test]
    async fn synth_payload_structured_template_keeps_whole_value_and_blanks_absent() {
        let template = json!({"echo": "{{body.n}}", "miss": "{{body.absent}}"});
        let answer = respond(
            json!({"body_template": template}),
            Bytes::from_static(br#"{"n":5}"#),
        )
        .await;
        check_body_json(&answer, &json!({"echo": 5, "miss": ""})).expect("template json");
    }

    #[proxima::test]
    async fn synth_payload_env_body_stays_text() {
        let config = temp_env::with_vars(
            [
                ("PROXIMA_SYNTH_BODY", Some("plain text")),
                ("PROXIMA_SYNTH_BODY_TEMPLATE", None::<&str>),
            ],
            || SynthConfig::from_env().expect("from_env"),
        );
        assert_eq!(config.body, Some(Value::String("plain text".to_string())));
        let handle = into_handle(config.from_config().expect("from_config"));
        let request = Request::builder()
            .method("GET")
            .path("/")
            .build()
            .expect("builder");
        let response = SendPipe::call(&handle, request).await.expect("call");
        let answer = response.collect_body().await.expect("collect");
        assert_eq!(&answer[..], b"plain text");
    }

    #[proxima::test]
    async fn ctl_payload_string_body_is_text() {
        let text = y1().to_string();
        let answer = respond(json!({"body": text}), Bytes::new()).await;
        check_body_json(&answer, &y1()).expect("body json");
    }

    #[proxima::test]
    async fn ctl_payload_string_template_expands_text() {
        let text = y2().to_string();
        let answer = respond(json!({"body_template": text}), y3()).await;
        check_body_json(&answer, &y4()).expect("template json");
    }

    #[test]
    fn ctl_payload_check_body_json_rejects_mismatch() {
        let bytes = serde_json::to_vec(&y1()).expect("bytes");
        assert!(check_body_json(&bytes, &y4()).is_err());
    }
}
