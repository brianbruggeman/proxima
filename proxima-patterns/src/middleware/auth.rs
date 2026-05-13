use std::collections::BTreeSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use bon::Builder;
use bytes::Bytes;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use proxima_core::ProximaError;
use proxima_primitives::pipe::{Pipe, SendPipe};
use proxima_primitives::pipe::handler::{Handler, PipeHandle, ThreadLocalPipeHandle, into_handle};
use proxima_primitives::pipe::pipe_factory::PipeFactory;
use proxima_primitives::pipe::request::{Request, Response};
use proxima_primitives::pipe::telemetry_surface::Labels;

const METRIC_REJECTED: &str = "proxima.auth.rejected_total";
const METRIC_ADMITTED: &str = "proxima.auth.admitted_total";

/// Token-based bearer auth pipe. Reads a configured request header
/// and admits requests whose value matches one of the configured
/// allow-list tokens. Rejected requests short-circuit with the
/// configured `on_unauthorized_status` (default 401) and a
/// `WWW-Authenticate` header naming the realm.
///
/// Generic over the inner handle: `Auth<PipeHandle>` impls `Handler`
/// (Send path, default); `Auth<ThreadLocalPipeHandle>` impls
/// `ThreadLocalHandler` (per-thread / DPDK path).
pub struct Auth<Inner = PipeHandle> {
    pub inner: Inner,
    pub header: String,
    pub allow: BTreeSet<String>,
    pub realm: Arc<[u8]>,
    pub on_unauthorized_status: u16,
    pub strip_prefix: Option<String>,
}

impl Auth<PipeHandle> {
    pub fn from_spec(inner: PipeHandle, value: &Value) -> Result<Self, ProximaError> {
        let config: AuthConfig = serde_json::from_value(value.clone())
            .map_err(|err| ProximaError::Config(format!("auth config: {err}")))?;
        config.into_auth(inner)
    }
}

fn default_header() -> String {
    "authorization".to_string()
}

fn default_realm() -> String {
    "proxima".to_string()
}

fn default_on_unauthorized_status() -> u16 {
    401
}

/// The `strip_prefix` default is `Some("Bearer ")` — an absent field means
/// "strip the bearer prefix", matching the historical hand-parser. Set it
/// explicitly to `""` (present-but-empty) to admit raw header values.
fn default_strip_prefix() -> Option<String> {
    Some("Bearer ".to_string())
}

/// Typed config surface for the `auth` middleware — token-based bearer auth
/// over a request header with an allow-list.
#[derive(Debug, Clone, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_AUTH")]
#[builder(derive(Clone, Debug), on(String, into))]
pub struct AuthConfig {
    /// Allow-listed tokens; at least one is required.
    #[setting(skip)]
    pub allow: BTreeSet<String>,

    /// Request header carrying the credential. Defaults to `authorization`.
    #[setting(default = "authorization")]
    #[serde(default = "default_header")]
    #[builder(default = default_header())]
    pub header: String,

    /// Realm advertised in the `WWW-Authenticate` header on rejection.
    #[setting(default = "proxima")]
    #[serde(default = "default_realm")]
    #[builder(default = default_realm())]
    pub realm: String,

    /// Status returned on a rejected request. Defaults to 401.
    #[setting(default = 401)]
    #[serde(default = "default_on_unauthorized_status")]
    #[builder(default = default_on_unauthorized_status())]
    pub on_unauthorized_status: u16,

    /// Prefix stripped from the header value before matching. The config wire
    /// form defaults to `Some("Bearer ")` (an absent field strips the bearer
    /// prefix); the builder defaults to `None` (bon treats `Option` as
    /// optional) — call `.strip_prefix(Some("Bearer ".into()))` to match.
    /// `None`/`Some("")` admit the raw value.
    #[setting(skip)]
    #[serde(default = "default_strip_prefix")]
    pub strip_prefix: Option<String>,
}

impl Validate for AuthConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.allow.is_empty() {
            errors.push(ValidationMessage::new(
                "allow",
                "must contain at least one token",
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl AuthConfig {
    /// Materialise the auth middleware around `inner`.
    pub fn into_auth(self, inner: PipeHandle) -> Result<Auth<PipeHandle>, ProximaError> {
        self.validate()
            .map_err(|err| ProximaError::Config(format!("{err}")))?;
        Ok(Auth {
            inner,
            header: self.header,
            allow: self.allow,
            realm: Arc::from(self.realm.as_bytes()),
            on_unauthorized_status: self.on_unauthorized_status,
            strip_prefix: self.strip_prefix,
        })
    }
}

impl<Inner> Auth<Inner> {
    fn extract_token(&self, request: &Request<Bytes>) -> Option<String> {
        let raw_bytes = request
            .metadata
            .iter()
            .find(|(name, _)| name.as_ref().eq_ignore_ascii_case(self.header.as_bytes()))
            .map(|(_, value)| value)?;
        let raw = std::str::from_utf8(raw_bytes).ok()?;
        match self.strip_prefix.as_ref() {
            Some(prefix) => raw.strip_prefix(prefix.as_str()).map(str::to_string),
            None => Some(raw.to_string()),
        }
    }

    fn rejected_response(&self) -> Response<Bytes> {
        Response::new(self.on_unauthorized_status)
            .with_header(
                "www-authenticate",
                format!("Bearer realm=\"{}\"", String::from_utf8_lossy(&self.realm)),
            )
            .with_body("unauthorized")
    }
}

impl<Inner> SendPipe for Auth<Inner>
where
    Inner: Handler + Clone,
{
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let telemetry = request.context.telemetry.clone();
        let context_labels = request.context.metric_labels(&[]);
        let token = self.extract_token(&request);
        let admitted = token.as_deref().is_some_and(|raw| self.allow.contains(raw));
        let rejected_response = (!admitted).then(|| self.rejected_response());
        let inner = self.inner.clone();
        let realm = Arc::clone(&self.realm);

        async move {
            let realm_str = std::str::from_utf8(&realm).unwrap_or("");
            let labels = with_extra(&context_labels, "realm", realm_str);
            if admitted {
                telemetry.counter_inc(METRIC_ADMITTED, &labels, 1);
                SendPipe::call(&inner, request).await
            } else {
                telemetry.counter_inc(METRIC_REJECTED, &labels, 1);
                Ok(rejected_response
                    .unwrap_or_else(|| Response::new(500).with_body("auth state inconsistency")))
            }
        }
    }
}


impl Pipe for Auth<ThreadLocalPipeHandle> {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        let telemetry = request.context.telemetry.clone();
        let context_labels = request.context.metric_labels(&[]);
        let token = self.extract_token(&request);
        let admitted = token.as_deref().is_some_and(|raw| self.allow.contains(raw));
        let rejected_response = (!admitted).then(|| self.rejected_response());
        let inner = self.inner.clone();
        let realm = Arc::clone(&self.realm);

        async move {
            let realm_str = std::str::from_utf8(&realm).unwrap_or("");
            let labels = with_extra(&context_labels, "realm", realm_str);
            if admitted {
                telemetry.counter_inc(METRIC_ADMITTED, &labels, 1);
                Pipe::call(&inner, request).await
            } else {
                telemetry.counter_inc(METRIC_REJECTED, &labels, 1);
                Ok(rejected_response
                    .unwrap_or_else(|| Response::new(500).with_body("auth state inconsistency")))
            }
        }
    }
}


pub struct AuthFactory;

impl PipeFactory for AuthFactory {
    fn name(&self) -> &str {
        "auth"
    }

    fn build(
        &self,
        spec: &Value,
        inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        Box::pin(async move {
            let inner =
                inner.ok_or_else(|| ProximaError::Config("auth requires an inner pipe".into()))?;
            let auth = Auth::from_spec(inner, &spec)?;
            Ok(into_handle(auth))
        })
    }
}

fn with_extra(base: &Labels, key: &str, value: &str) -> Labels {
    let mut pairs: Vec<(String, String)> = base.entries().to_vec();
    pairs.push((key.to_string(), value.to_string()));
    let pair_refs: Vec<(&str, &str)> = pairs
        .iter()
        .map(|(name, value)| (name.as_str(), value.as_str()))
        .collect();
    Labels::from_pairs(&pair_refs)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
mod tests {
    use super::*;
    use proxima_primitives::pipe::telemetry_surface::{Telemetry, TelemetryHandle};
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::Mutex;

    #[derive(Default)]
    struct Metrics {
        counters: Mutex<HashMap<(String, Vec<(String, String)>), u64>>,
    }

    impl Telemetry for Metrics {
        fn counter_inc(&self, metric: &str, labels: &Labels, by: u64) {
            let key = (metric.to_string(), labels.entries().to_vec());
            *self.counters.lock().unwrap().entry(key).or_insert(0) += by;
        }
        fn gauge_set(&self, _: &str, _: &Labels, _: i64) {}
        fn histogram_record(&self, _: &str, _: &Labels, _: f64) {}
    }

    impl Metrics {
        fn counter(&self, metric: &str, labels: &Labels) -> Option<u64> {
            let key = (metric.to_string(), labels.entries().to_vec());
            self.counters.lock().unwrap().get(&key).copied()
        }
    }

    struct AlwaysOk;

    impl SendPipe for AlwaysOk {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            async { Ok(Response::ok("ok")) }
        }
    }


    fn build_request(token: Option<&str>) -> (Request<Bytes>, Arc<Metrics>) {
        let metrics: Arc<Metrics> = Arc::new(Metrics::default());
        let telemetry: TelemetryHandle = metrics.clone();
        let mut builder = Request::builder()
            .method("GET")
            .path("/")
            .telemetry(telemetry);
        if let Some(value) = token {
            builder = builder.header("authorization", value);
        }
        let request = builder.build().expect("builder");
        (request, metrics)
    }

    fn auth(allow: &[&str]) -> Auth {
        Auth {
            inner: into_handle(AlwaysOk),
            header: "authorization".into(),
            allow: allow.iter().map(|raw| (*raw).to_string()).collect(),
            realm: Arc::from(b"proxima".as_slice()),
            on_unauthorized_status: 401,
            strip_prefix: Some("Bearer ".into()),
        }
    }

    #[proxima::test]
    async fn admits_request_with_matching_token() {
        let stack = auth(&["abc123"]);
        let (request, _metrics) = build_request(Some("Bearer abc123"));
        let response = SendPipe::call(&stack, request).await.expect("call");
        assert_eq!(response.status, 200);
    }

    #[proxima::test]
    async fn rejects_request_with_missing_header() {
        let stack = auth(&["abc123"]);
        let (request, _metrics) = build_request(None);
        let response = SendPipe::call(&stack, request).await.expect("call");
        assert_eq!(response.status, 401);
        assert!(response.metadata.contains_key("www-authenticate"));
    }

    #[proxima::test]
    async fn rejects_request_with_unknown_token() {
        let stack = auth(&["abc123"]);
        let (request, _metrics) = build_request(Some("Bearer not-allowed"));
        let response = SendPipe::call(&stack, request).await.expect("call");
        assert_eq!(response.status, 401);
    }

    #[proxima::test]
    async fn admits_request_with_raw_header_when_strip_prefix_disabled() {
        let mut stack = auth(&["abc123"]);
        stack.header = "x-api-token".into();
        stack.strip_prefix = None;
        let metrics: Arc<Metrics> = Arc::new(Metrics::default());
        let telemetry: TelemetryHandle = metrics.clone();
        let request = Request::builder()
            .method("GET")
            .path("/")
            .telemetry(telemetry)
            .header("x-api-token", "abc123")
            .build()
            .expect("builder");
        let response = SendPipe::call(&stack, request).await.expect("call");
        assert_eq!(response.status, 200);
    }

    #[proxima::test]
    async fn rejected_increments_telemetry_counter() {
        let stack = auth(&["abc123"]);
        let (request, metrics) = build_request(Some("Bearer wrong"));
        let _ = SendPipe::call(&stack, request).await.expect("call");
        let labels = Labels::from_pairs(&[("realm", "proxima")]);
        let rejected = metrics.counter(METRIC_REJECTED, &labels);
        assert_eq!(rejected, Some(1));
    }

    // principle-4 parity: the fluent builder and the config value must lower to
    // identical Auth state (header, allow set, realm, status, strip prefix).
    #[test]
    fn parity_fluent_builder_and_config_value_match() {
        let from_value: AuthConfig = serde_json::from_value(serde_json::json!({
            "header": "x-token",
            "allow": ["one", "two"],
            "realm": "my-app",
            "on_unauthorized_status": 403,
            "strip_prefix": "Token ",
        }))
        .expect("from_value");
        let from_value = from_value
            .into_auth(into_handle(AlwaysOk))
            .expect("into_auth value");

        let from_builder = AuthConfig::builder()
            .header("x-token")
            .allow(["one".to_string(), "two".to_string()].into_iter().collect())
            .realm("my-app")
            .on_unauthorized_status(403)
            .strip_prefix("Token ")
            .build()
            .into_auth(into_handle(AlwaysOk))
            .expect("into_auth builder");

        assert_eq!(from_value.header, from_builder.header);
        assert_eq!(from_value.allow, from_builder.allow);
        assert_eq!(from_value.realm.as_ref(), from_builder.realm.as_ref());
        assert_eq!(
            from_value.on_unauthorized_status,
            from_builder.on_unauthorized_status
        );
        assert_eq!(from_value.strip_prefix, from_builder.strip_prefix);
    }

    #[proxima::test]
    async fn from_spec_requires_non_empty_allow() {
        let outcome = Auth::from_spec(into_handle(AlwaysOk), &serde_json::json!({"allow": []}));
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[proxima::test]
    async fn from_spec_parses_full_config() {
        let spec = serde_json::json!({
            "header": "x-token",
            "allow": ["one", "two"],
            "realm": "my-app",
            "on_unauthorized_status": 403,
            "strip_prefix": "",
        });
        let auth = Auth::from_spec(into_handle(AlwaysOk), &spec).expect("parse");
        assert_eq!(auth.header, "x-token");
        assert_eq!(auth.allow.len(), 2);
        assert_eq!(auth.realm.as_ref(), b"my-app");
        assert_eq!(auth.on_unauthorized_status, 403);
        assert_eq!(auth.strip_prefix.as_deref(), Some(""));
    }
}
