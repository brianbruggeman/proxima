use std::future::Future;
use std::pin::Pin;
use std::time::Instant;

use bytes::Bytes;
use hyper::body::Incoming;
use serde_json::Value;
use tracing::warn;

use crate::http1::hyper_body::StreamingHyperBody;
use crate::http1::http_config::{HttpConfig, HttpUpstreamConfig};
use crate::http1::shared_http::SharedHttpClient;
use crate::templates::{TemplateContext, expand};
use proxima_core::ProximaError;
use proxima_primitives::pipe::body::ResponseStream;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::handler::{PipeHandle, into_handle};
use proxima_primitives::pipe::pipe_factory::PipeFactory;
use proxima_primitives::pipe::request::{Request, Response};

pub struct HttpUpstream {
    base_url: String,
    label: String,
    client: SharedHttpClient,
    config: HttpUpstreamConfig,
}

impl HttpUpstream {
    /// Builds an upstream with its own pool. Prefer
    /// `with_shared_client` so co-located upstreams share one pool.
    pub fn new(base_url: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            label: label.into(),
            client: SharedHttpClient::new(),
            config: HttpUpstreamConfig::default(),
        }
    }

    pub fn with_shared_client(
        base_url: impl Into<String>,
        label: impl Into<String>,
        client: SharedHttpClient,
    ) -> Self {
        Self {
            base_url: base_url.into(),
            label: label.into(),
            client,
            config: HttpUpstreamConfig::default(),
        }
    }

    #[must_use]
    pub fn with_config(mut self, config: HttpUpstreamConfig) -> Self {
        self.config = config;
        self
    }
}

fn build_uri(
    base_url: &str,
    path: &[u8],
    query: &proxima_primitives::pipe::header_list::HeaderList,
) -> Result<hyper::Uri, ProximaError> {
    let trimmed_base = base_url.trim_end_matches('/');
    let needs_slash = !path.starts_with(b"/");
    let mut estimated = trimmed_base.len() + path.len() + usize::from(needs_slash);
    if !query.is_empty() {
        // worst case: every char escapes to "%XX" so 3x byte budget. cheap upper bound.
        let query_bound: usize = query
            .iter()
            .map(|(name, value)| 3 * (name.len() + value.len()) + 2)
            .sum();
        estimated = estimated.saturating_add(1).saturating_add(query_bound);
    }
    let mut uri = String::with_capacity(estimated);
    uri.push_str(trimmed_base);
    if needs_slash {
        uri.push('/');
    }
    // path is bytes-internal; the URI string accepts a UTF-8 view. typical
    // paths are ASCII so lossy is benign.
    uri.push_str(&String::from_utf8_lossy(path));
    let mut first = true;
    for (name, value) in query {
        uri.push(if first { '?' } else { '&' });
        first = false;
        urlencode_into(name.as_ref(), &mut uri);
        uri.push('=');
        urlencode_into(value.as_ref(), &mut uri);
    }
    uri.parse::<hyper::Uri>()
        .map_err(|error| ProximaError::Config(format!("invalid uri '{uri}': {error}")))
}

fn urlencode_into(input: &[u8], out: &mut String) {
    for &byte in input {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            _ => {
                const HEX: &[u8; 16] = b"0123456789ABCDEF";
                out.push('%');
                out.push(HEX[(byte >> 4) as usize] as char);
                out.push(HEX[(byte & 0x0F) as usize] as char);
            }
        }
    }
}

impl SendPipe for HttpUpstream {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let started = Instant::now();
        let telemetry = request.context.telemetry.clone();
        let context_for_labels = request.context.clone();
        let label = self.label.clone();
        let config = self.config.clone();
        let client = self.client.clone();
        let base_url = self.base_url.clone();
        async move {
            let trace_id = request.context.trace_id.clone();
            let baggage = request.context.baggage.clone();
            let pipe_label = request.context.pipe_label.clone();
            let cancel = request.context.cancel.clone();
            let mut method = request.method.clone();
            let path = request.path.clone();
            let query = request.query.clone();
            let mut headers = request.metadata.clone();
            let body = request.into_chunk_stream();
            if let Some(override_method) = config.method_override {
                method = proxima_primitives::pipe::Method::from_bytes(override_method.as_bytes());
            }
            if let Some(allowed) = &config.forward_request_headers {
                headers.retain(|name, _| {
                    allowed.iter().any(|allowed_name| {
                        allowed_name.as_ref().eq_ignore_ascii_case(name.as_ref())
                    })
                });
            }
            let trace_id_str = trace_id
                .as_deref()
                .and_then(|bytes| std::str::from_utf8(bytes).ok());
            let pipe_label_str = pipe_label
                .as_deref()
                .and_then(|bytes| std::str::from_utf8(bytes).ok());
            let template_context = TemplateContext {
                request_id: None,
                trace_id: trace_id_str,
                pipe: pipe_label_str,
                body: None,
            };
            for (name, value) in &config.injected_request_headers {
                let expanded = expand(value, &template_context);
                headers.insert(name.clone(), expanded);
            }
            // propagate proxima's restamped trace context + baggage to the
            // origin so a single trace spans the hop. an operator-injected
            // header above wins (insert_if_absent).
            if let Some(trace_id_bytes) = trace_id.as_deref() {
                headers.insert_if_absent(proxima_telemetry::propagation::TRACEPARENT, trace_id_bytes);
            }
            if let Some(baggage_bytes) = baggage.as_deref() {
                headers.insert_if_absent(proxima_telemetry::propagation::BAGGAGE, baggage_bytes);
            }
            let uri = build_uri(&base_url, path.as_ref(), &query)?;
            let mut http_request = hyper::Request::builder()
                .method(method.as_bytes())
                .uri(uri.clone());
            for (name, value) in &headers {
                http_request = http_request.header(name.as_ref(), value.as_ref());
            }
            let request_built = http_request
                .body(StreamingHyperBody::new(body))
                .map_err(|error| ProximaError::Upstream(format!("build request: {error}")))?;
            // Race the upstream call against per-request cancellation.
            // If cancel fires (client disconnect, deadline, operator stop),
            // the hyper future is dropped, the upstream socket closes,
            // and we return early with a typed cancellation error.
            let upstream_future = client.request(request_built);
            let outcome: Result<hyper::Response<Incoming>, ProximaError> = match config.timeout {
                Some(timeout) => tokio::select! {
                    biased;
                    _ = cancel.fired() => Err(ProximaError::Upstream("cancelled".into())),
                    raced = proxima_core::time::timeout(timeout, upstream_future) => match raced {
                        Ok(Ok(response)) => Ok(response),
                        Ok(Err(error)) => Err(error),
                        Err(_) => Err(ProximaError::Timeout(timeout)),
                    },
                },
                None => tokio::select! {
                    biased;
                    _ = cancel.fired() => Err(ProximaError::Upstream("cancelled".into())),
                    raced = upstream_future => raced,
                },
            };
            let response = match outcome {
                Ok(response) => response,
                Err(error) => {
                    let labels = context_for_labels
                        .metric_labels(&[("upstream", label.as_str()), ("status_class", "error")]);
                    telemetry.counter_inc("proxima.upstream.errors_total", &labels, 1);
                    return Err(error);
                }
            };
            let status = response.status().as_u16();
            let translated = translate_response(response)?;
            let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let labels = context_for_labels.metric_labels(&[
                ("upstream", label.as_str()),
                ("status_class", status_class(status)),
            ]);
            telemetry.counter_inc("proxima.upstream.calls_total", &labels, 1);
            telemetry.histogram_record("proxima.upstream.latency_ms", &labels, elapsed_ms);
            Ok(translated)
        }
    }
}

fn status_class(status: u16) -> &'static str {
    match status {
        100..=199 => "1xx",
        200..=299 => "2xx",
        300..=399 => "3xx",
        400..=499 => "4xx",
        500..=599 => "5xx",
        _ => "other",
    }
}

fn translate_response(
    response: hyper::Response<Incoming>,
) -> Result<Response<Bytes>, ProximaError> {
    let status = response.status().as_u16();
    let mut translated = Response::new(status);
    for (name, value) in response.headers() {
        match value.to_str() {
            Ok(text) => {
                translated = translated.with_header(name.as_str().to_string(), text.to_string());
            }
            Err(_) => warn!(header = %name, "skipping non-utf8 response header"),
        }
    }
    let stream = body_stream_from_incoming(response.into_body());
    Ok(translated.with_stream(ResponseStream::new(stream)))
}

fn body_stream_from_incoming(
    incoming: Incoming,
) -> impl futures::stream::Stream<Item = Result<Bytes, ProximaError>> + Send {
    use futures::stream;
    use http_body_util::BodyStream;
    let frames = BodyStream::new(incoming);
    stream::unfold(frames, |mut frames| async move {
        loop {
            use futures::StreamExt;
            match frames.next().await? {
                Ok(frame) => {
                    if let Ok(data) = frame.into_data() {
                        return Some((Ok(data), frames));
                    }
                }
                Err(error) => {
                    return Some((
                        Err(ProximaError::Upstream(format!("read body: {error}"))),
                        frames,
                    ));
                }
            }
        }
    })
}

pub struct HttpPipeFactory {
    client: SharedHttpClient,
}

impl HttpPipeFactory {
    #[must_use]
    pub fn new() -> Self {
        Self {
            client: SharedHttpClient::new(),
        }
    }

    #[must_use]
    pub fn with_shared_client(client: SharedHttpClient) -> Self {
        Self { client }
    }
}

impl Default for HttpPipeFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl PipeFactory for HttpPipeFactory {
    fn name(&self) -> &str {
        "http"
    }

    fn build(
        &self,
        spec: &Value,
        _inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        let client = self.client.clone();
        Box::pin(async move {
            let config: HttpConfig = serde_json::from_value(spec)
                .map_err(|err| ProximaError::Config(format!("http config: {err}")))?;
            let handle: PipeHandle = into_handle(config.into_upstream(client)?);
            Ok(handle)
        })
    }
}

// `HeaderForward`/`HttpHeadersConfig`/`HttpUpstreamConfig`/`HttpConfig` (plus
// `into_runtime_config`) live in `http_config.rs` — pure data, no hyper, so
// the prime-native client (`client.rs`/`prime_upstream.rs`) can use them
// without pulling this hyper-backed module in.
impl HttpConfig {
    /// Materialise the `http` upstream over a shared client pool.
    pub fn into_upstream(self, client: SharedHttpClient) -> Result<HttpUpstream, ProximaError> {
        let runtime = self.into_runtime_config()?;
        Ok(HttpUpstream::with_shared_client(self.url, self.name, client).with_config(runtime))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::http1::http_config::{HeaderForward, HttpHeadersConfig};

    #[test]
    fn build_uri_concatenates_base_and_path() {
        let uri = build_uri(
            "http://example.test",
            b"/users/42",
            &proxima_primitives::pipe::header_list::HeaderList::new(),
        )
        .expect("uri");
        assert_eq!(uri.to_string(), "http://example.test/users/42");
    }

    #[test]
    fn build_uri_handles_missing_leading_slash() {
        let uri = build_uri(
            "http://example.test",
            b"users",
            &proxima_primitives::pipe::header_list::HeaderList::new(),
        )
        .expect("uri");
        assert_eq!(uri.to_string(), "http://example.test/users");
    }

    #[test]
    fn build_uri_appends_query_pairs() {
        let mut query = proxima_primitives::pipe::header_list::HeaderList::new();
        query.insert("q", "alpha beta");
        query.insert("limit", "10");
        let uri = build_uri("http://example.test", b"/search", &query).expect("uri");
        let rendered = uri.to_string();
        assert!(rendered.contains("limit=10"));
        assert!(rendered.contains("q=alpha%20beta"));
    }

    #[test]
    fn factory_requires_url_field() {
        let factory = HttpPipeFactory::new();
        let outcome = futures::executor::block_on(factory.build(&serde_json::json!({}), None));
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[test]
    fn factory_with_shared_client_reuses_pool_across_upstreams() {
        let shared = SharedHttpClient::new();
        let factory_one = HttpPipeFactory::with_shared_client(shared.clone());
        let factory_two = HttpPipeFactory::with_shared_client(shared.clone());
        // factory_one + factory_two + the test holder = 3 strong references to the same Arc<Client>.
        assert_eq!(
            shared.strong_count(),
            3,
            "all factories must share one Arc<Client>"
        );
        drop(factory_one);
        drop(factory_two);
        assert_eq!(shared.strong_count(), 1);
    }

    // principle-4 parity: the fluent builder and the config value must lower to
    // identical HttpUpstreamConfig (method, timeout, forward list, injected).
    #[test]
    fn parity_fluent_builder_and_config_value_match() {
        let from_value: HttpConfig = serde_json::from_value(serde_json::json!({
            "url": "http://origin.test",
            "name": "edge",
            "method": "POST",
            "timeout": "5s",
            "headers": {
                "forward": ["x-trace-id", "Authorization"],
                "request": {"x-injected": "1"},
            },
        }))
        .expect("from_value");
        let from_value = from_value.into_runtime_config().expect("runtime value");

        let from_builder = HttpConfig::builder()
            .url("http://origin.test")
            .name("edge")
            .method("POST".to_string())
            .timeout("5s".to_string())
            .headers(
                HttpHeadersConfig::builder()
                    .forward(HeaderForward::List(alloc_vec()))
                    .request(injected_map())
                    .build(),
            )
            .build()
            .into_runtime_config()
            .expect("runtime builder");

        assert_eq!(from_value.method_override, from_builder.method_override);
        assert_eq!(from_value.timeout, from_builder.timeout);
        assert_eq!(
            from_value.forward_request_headers,
            from_builder.forward_request_headers
        );
        assert_eq!(
            from_value.injected_request_headers,
            from_builder.injected_request_headers
        );
    }

    fn alloc_vec() -> Vec<String> {
        vec!["x-trace-id".to_string(), "Authorization".to_string()]
    }

    fn injected_map() -> BTreeMap<String, String> {
        let mut map = BTreeMap::new();
        map.insert("x-injected".to_string(), "1".to_string());
        map
    }
}
