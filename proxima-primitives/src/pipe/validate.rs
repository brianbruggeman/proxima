use bytes::Bytes;
use core::future::Future;

use crate::pipe::SendPipe;
pub use crate::pipe::capabilities::{CheckOutcome, Checkable};
use crate::pipe::primitives::Pipe;

use crate::pipe::handler::{PipeHandle, ThreadLocalPipeHandle};
use crate::pipe::request::{Request, Response};
use proxima_core::ProximaError;

// proxima-config's schema module (Schema/SchemaRegistry/ValidationError) is
// std-only, and the HTTP `Checkable` path uses serde_json; the whole
// schema-backed validate surface is std-tier. The generic `Validate<Inner, Op>`
// over `Checkable<Op>` is alloc-tier.
#[cfg(feature = "std")]
use crate::pipe::handler::into_handle;
#[cfg(feature = "std")]
use crate::pipe::pipe_factory::PipeFactory;
#[cfg(feature = "std")]
use alloc::sync::Arc;
#[cfg(feature = "std")]
use proxima_config::schema::{Schema, SchemaRegistry, ValidationError};
#[cfg(feature = "std")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "std")]
use serde_json::{Value, json};
#[cfg(feature = "std")]
use std::pin::Pin;

#[cfg(feature = "std")]
const METRIC_REJECTED: &str = "proxima.validate.rejected_total";
#[cfg(feature = "std")]
const METRIC_ADMITTED: &str = "proxima.validate.admitted_total";

// ── main struct ───────────────────────────────────────────────────────────────

/// Predicate/schema-gated admission middleware. Generic over the inner pipe AND
/// the check op. The std default op is the HTTP `ValidateOp` (proxima-config's
/// schema module); under no_std+alloc that type is absent, so the default
/// falls back to `()` and callers name their own `Checkable` op explicitly.
#[cfg(feature = "std")]
pub struct Validate<Inner = PipeHandle, Op = ValidateOp> {
    pub inner: Inner,
    op: Op,
}

#[cfg(not(feature = "std"))]
pub struct Validate<Inner = PipeHandle, Op = ()> {
    pub inner: Inner,
    op: Op,
}

impl<Inner, Op> Validate<Inner, Op> {
    #[must_use]
    pub fn new(inner: Inner, op: Op) -> Self {
        Self { inner, op }
    }
}

impl<Inner, Op> SendPipe for Validate<Inner, Op>
where
    Inner: SendPipe + Clone,
    Inner::In: Checkable<Op, Out = Inner::Out> + Send,
    Inner::Out: Send,
    Inner::Err: From<ProximaError> + Send,
    Op: Clone + Send + Sync + 'static,
{
    type In = Inner::In;
    type Out = Inner::Out;
    type Err = Inner::Err;

    fn call(
        &self,
        input: Inner::In,
    ) -> impl Future<Output = Result<Inner::Out, Inner::Err>> + Send {
        let op = self.op.clone();
        let inner = self.inner.clone();
        async move {
            match input.check(&op).await.map_err(Inner::Err::from)? {
                CheckOutcome::Pass(admitted) => SendPipe::call(&inner, admitted).await,
                CheckOutcome::Reject(rejection) => Ok(rejection),
            }
        }
    }
}

impl<Op> Pipe for Validate<ThreadLocalPipeHandle, Op>
where
    Request<Bytes>: Checkable<Op, Out = Response<Bytes>>,
    Op: Clone + Send + Sync + 'static,
{
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        input: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        let op = self.op.clone();
        let inner = self.inner.clone();
        async move {
            match input.check(&op).await? {
                CheckOutcome::Pass(admitted) => Pipe::call(&inner, admitted).await,
                CheckOutcome::Reject(rejection) => Ok(rejection),
            }
        }
    }
}

// ── HTTP op type + Checkable impl ─────────────────────────────────────────────

#[cfg(feature = "std")]
#[derive(Clone)]
pub struct ValidateOp {
    pub schema: Schema,
    pub schemas: Arc<SchemaRegistry>,
}

#[cfg(feature = "std")]
impl ValidateOp {
    #[must_use]
    pub fn new(schema: Schema, schemas: Arc<SchemaRegistry>) -> Self {
        Self { schema, schemas }
    }
}

#[cfg(feature = "std")]
impl Checkable<ValidateOp> for Request<Bytes> {
    type Out = Response<Bytes>;

    fn check(
        self,
        op: &ValidateOp,
    ) -> impl Future<Output = Result<CheckOutcome<Self, Response<Bytes>>, ProximaError>> + Send
    {
        let schema = op.schema.clone();
        let schemas = op.schemas.clone();
        async move {
            let telemetry = self.context.telemetry.clone();
            let labels = self.context.metric_labels(&[]);
            let (request, buffered) = self.body_bytes().await?;
            let Request::<Bytes> {
                method,
                path,
                query,
                metadata,
                payload: _,
                stream: _,
                context,
            } = request;
            if buffered.is_empty()
                && matches!(
                    method,
                    crate::pipe::method::Method::Get
                        | crate::pipe::method::Method::Head
                        | crate::pipe::method::Method::Delete
                )
            {
                telemetry.counter_inc(METRIC_ADMITTED, &labels, 1);
                let forwarded = Request {
                    method,
                    path,
                    query,
                    metadata,
                    payload: buffered,
                    stream: None,
                    context,
                };
                return Ok(CheckOutcome::Pass(forwarded));
            }
            let value: Value = if buffered.is_empty() {
                Value::Null
            } else {
                serde_json::from_slice(&buffered).map_err(|err| {
                    ProximaError::Body(format!("validate: request payload is not JSON: {err}"))
                })?
            };
            if let Err(err) = schema.validate(&value, &schemas) {
                telemetry.counter_inc(METRIC_REJECTED, &labels, 1);
                return Ok(CheckOutcome::Reject(rejected_response(&err)));
            }
            telemetry.counter_inc(METRIC_ADMITTED, &labels, 1);
            let forwarded = Request {
                method,
                path,
                query,
                metadata,
                payload: buffered,
                stream: None,
                context,
            };
            Ok(CheckOutcome::Pass(forwarded))
        }
    }
}

#[cfg(feature = "std")]
fn rejected_response(err: &ValidationError) -> Response<Bytes> {
    let body = json!({
        "error": "validation_failed",
        "path": err.path_string(),
        "message": err.message,
    });
    Response::new(400)
        .with_header("content-type", "application/json")
        .with_body(body.to_string())
}

// ── factory + constructors ────────────────────────────────────────────────────

#[cfg(feature = "std")]
impl Validate<PipeHandle, ValidateOp> {
    pub fn from_schema(inner: PipeHandle, schema: Schema, schemas: Arc<SchemaRegistry>) -> Self {
        Self::new(inner, ValidateOp::new(schema, schemas))
    }

    pub fn from_spec(
        inner: PipeHandle,
        value: &Value,
        schemas: &Arc<SchemaRegistry>,
    ) -> Result<Self, ProximaError> {
        let config: ValidateConfig = serde_json::from_value(value.clone())
            .map_err(|err| ProximaError::Config(format!("validate config: {err}")))?;
        config.from_config(inner, schemas)
    }
}

/// A schema reference: either a name resolved against the [`SchemaRegistry`], or
/// an inline [`Schema`] definition. Untagged so a JSON string deserialises to
/// [`SchemaRef::Named`] and a JSON object to [`SchemaRef::Inline`] — matching the
/// historical hand-parser's string-then-object precedence.
#[cfg(feature = "std")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SchemaRef {
    Named(String),
    Inline(Schema),
}

/// Typed config surface for the `validate` middleware. The [`SchemaRegistry`] is
/// runtime state supplied at materialisation (not serialisable data), mirroring
/// how `callback` resolves its function from a registry.
#[cfg(feature = "std")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidateConfig {
    pub schema: SchemaRef,
}

#[cfg(feature = "std")]
impl ValidateConfig {
    /// Fluent constructor referencing a named schema.
    #[must_use]
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            schema: SchemaRef::Named(name.into()),
        }
    }

    /// Fluent constructor carrying an inline schema.
    #[must_use]
    pub fn inline(schema: Schema) -> Self {
        Self {
            schema: SchemaRef::Inline(schema),
        }
    }

    /// Materialise the validate middleware around `inner`, resolving a named
    /// schema against `schemas`.
    pub fn from_config(
        self,
        inner: PipeHandle,
        schemas: &Arc<SchemaRegistry>,
    ) -> Result<Validate<PipeHandle, ValidateOp>, ProximaError> {
        let schema = match self.schema {
            SchemaRef::Named(name) => schemas.get(&name).ok_or_else(|| {
                ProximaError::Config(format!(
                    "validate middleware references unknown schema `{name}`"
                ))
            })?,
            SchemaRef::Inline(schema) => schema,
        };
        Ok(Validate::new(
            inner,
            ValidateOp::new(schema, schemas.clone()),
        ))
    }
}

#[cfg(feature = "std")]
pub struct ValidateFactory {
    schemas: Arc<SchemaRegistry>,
}

#[cfg(feature = "std")]
impl ValidateFactory {
    #[must_use]
    pub fn new(schemas: Arc<SchemaRegistry>) -> Self {
        Self { schemas }
    }
}

#[cfg(feature = "std")]
impl PipeFactory for ValidateFactory {
    fn name(&self) -> &str {
        "validate"
    }

    fn build(
        &self,
        spec: &Value,
        inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        Box::pin(async move {
            let inner = inner
                .ok_or_else(|| ProximaError::Config("validate requires an inner pipe".into()))?;
            let validate = Validate::from_spec(inner, &spec, &self.schemas)?;
            Ok(into_handle(validate))
        })
    }
}

// `#[proxima::test]` pulls in the `proxima` dev-dependency, which the
// loom build keeps out of the graph (see
// `[target.'cfg(not(loom))'.dev-dependencies]` in Cargo.toml); these
// tests are unrelated to the Notify/watch loom protocol.
#[cfg(all(test, not(loom)))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::future::Future;

    use super::*;
    use crate::pipe::handler::into_handle;
    use crate::pipe::request::Request;
    use crate::pipe::telemetry_surface::{NoopTelemetry, TelemetryHandle};
    use proxima_config::schema::{FieldFlags, StructField};

    fn echo_pipe() -> PipeHandle {
        struct EchoPipe;
        impl SendPipe for EchoPipe {
            type In = Request<Bytes>;
            type Out = Response<Bytes>;
            type Err = ProximaError;

            fn call(
                &self,
                request: Request<Bytes>,
            ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
                async move {
                    let (_, body) = request.body_bytes().await?;
                    Ok(Response::new(200).with_body(body))
                }
            }
        }
        into_handle(EchoPipe)
    }

    fn user_schema() -> Schema {
        Schema::Struct {
            name: "User".into(),
            fields: vec![StructField {
                name: "name".into(),
                schema: Schema::String {
                    pattern: None,
                    format: None,
                    min_len: Some(1),
                    max_len: None,
                },
                flags: FieldFlags::default(),
            }],
        }
    }

    fn noop_telemetry() -> TelemetryHandle {
        NoopTelemetry::handle()
    }

    // principle-4 parity: the fluent builder and the config value must lower to
    // identical ValidateOp state (the resolved schema).
    #[test]
    fn parity_fluent_builder_and_config_value_match() {
        let schemas = Arc::new(SchemaRegistry::new());
        let schema_json = serde_json::to_value(user_schema()).expect("schema json");

        let from_value: ValidateConfig =
            serde_json::from_value(json!({ "schema": schema_json })).expect("from_value");
        let from_value = from_value
            .from_config(echo_pipe(), &schemas)
            .expect("from_config value");

        let from_builder = ValidateConfig::inline(user_schema())
            .from_config(echo_pipe(), &schemas)
            .expect("from_config builder");

        assert_eq!(
            serde_json::to_value(&from_value.op.schema).expect("ser value"),
            serde_json::to_value(&from_builder.op.schema).expect("ser builder"),
        );
    }

    #[proxima::test]
    async fn admits_valid_body() {
        let schemas = Arc::new(SchemaRegistry::new());
        let wrapped = Validate::from_schema(echo_pipe(), user_schema(), schemas);
        let telemetry = noop_telemetry();
        let request = Request::builder()
            .method("POST")
            .path("/")
            .body(bytes::Bytes::from_static(br#"{"name":"brian"}"#))
            .header("content-type", "application/json")
            .telemetry(telemetry)
            .build()
            .expect("builder");
        let response = SendPipe::call(&wrapped, request).await.expect("call");
        assert_eq!(response.status, 200);
    }

    #[proxima::test]
    async fn rejects_missing_required_field() {
        let schemas = Arc::new(SchemaRegistry::new());
        let wrapped = Validate::from_schema(echo_pipe(), user_schema(), schemas);
        let telemetry = noop_telemetry();
        let request = Request::builder()
            .method("POST")
            .path("/")
            .body(bytes::Bytes::from_static(b"{}"))
            .header("content-type", "application/json")
            .telemetry(telemetry)
            .build()
            .expect("builder");
        let response = SendPipe::call(&wrapped, request).await.expect("call");
        assert_eq!(response.status, 400);
        let body = response.collect_body().await.expect("body");
        let parsed: Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(parsed["error"], "validation_failed");
        assert!(
            parsed["message"]
                .as_str()
                .unwrap_or("")
                .contains("missing required field `name`")
        );
    }

    #[proxima::test]
    async fn resolves_named_schema_from_registry() {
        let schemas = Arc::new(SchemaRegistry::new());
        schemas.register("User", user_schema()).expect("register");
        let wrapped = Validate::from_spec(echo_pipe(), &json!({"schema": "User"}), &schemas)
            .expect("from spec");
        let telemetry = noop_telemetry();
        let request = Request::builder()
            .method("POST")
            .path("/")
            .body(bytes::Bytes::from_static(br#"{"name":"brian"}"#))
            .header("content-type", "application/json")
            .telemetry(telemetry)
            .build()
            .expect("builder");
        let response = SendPipe::call(&wrapped, request).await.expect("call");
        assert_eq!(response.status, 200);
    }

    #[proxima::test]
    async fn empty_body_get_short_circuits_validation() {
        let schemas = Arc::new(SchemaRegistry::new());
        let wrapped = Validate::from_schema(echo_pipe(), user_schema(), schemas);
        let telemetry = noop_telemetry();
        let request = Request::builder()
            .method("GET")
            .path("/todos")
            .telemetry(telemetry)
            .build()
            .expect("builder");
        let response = SendPipe::call(&wrapped, request).await.expect("call");
        assert_eq!(
            response.status, 200,
            "GET with empty body must skip validate"
        );
    }

    #[proxima::test]
    async fn empty_body_delete_short_circuits_validation() {
        let schemas = Arc::new(SchemaRegistry::new());
        let wrapped = Validate::from_schema(echo_pipe(), user_schema(), schemas);
        let telemetry = noop_telemetry();
        let request = Request::builder()
            .method("DELETE")
            .path("/todos/abc")
            .telemetry(telemetry)
            .build()
            .expect("builder");
        let response = SendPipe::call(&wrapped, request).await.expect("call");
        assert_eq!(
            response.status, 200,
            "DELETE with empty body must skip validate"
        );
    }

    #[proxima::test]
    async fn from_spec_errors_on_unknown_named_schema() {
        let schemas = Arc::new(SchemaRegistry::new());
        let outcome =
            Validate::from_spec(echo_pipe(), &json!({"schema": "MissingShape"}), &schemas);
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[derive(Clone, PartialEq, Debug)]
    struct EventMsg {
        kind: &'static str,
        value: u64,
    }

    #[derive(Clone)]
    struct EventOp {
        allowed_kind: &'static str,
    }

    impl Checkable<EventOp> for EventMsg {
        type Out = EventMsg;

        fn check(
            self,
            op: &EventOp,
        ) -> impl Future<Output = Result<CheckOutcome<Self, Self::Out>, ProximaError>> + Send
        {
            let allowed = op.allowed_kind;
            async move {
                if self.kind == allowed {
                    Ok(CheckOutcome::Pass(self))
                } else {
                    Ok(CheckOutcome::Reject(EventMsg {
                        kind: "rejected",
                        value: 0,
                    }))
                }
            }
        }
    }

    #[derive(Clone)]
    struct EventSink;

    impl SendPipe for EventSink {
        type In = EventMsg;
        type Out = EventMsg;
        type Err = ProximaError;

        fn call(
            &self,
            input: EventMsg,
        ) -> impl Future<Output = Result<EventMsg, ProximaError>> + Send {
            async move {
                Ok(EventMsg {
                    kind: "processed",
                    value: input.value + 1,
                })
            }
        }
    }

    #[proxima::test]
    async fn validate_is_generic_over_a_non_http_payload() {
        let stack = Validate::new(
            EventSink,
            EventOp {
                allowed_kind: "ping",
            },
        );

        let admitted = SendPipe::call(
            &stack,
            EventMsg {
                kind: "ping",
                value: 10,
            },
        )
        .await
        .expect("admitted call");
        assert_eq!(
            admitted,
            EventMsg {
                kind: "processed",
                value: 11
            }
        );

        let rejected = SendPipe::call(
            &stack,
            EventMsg {
                kind: "unknown",
                value: 99,
            },
        )
        .await
        .expect("rejected call");
        assert_eq!(
            rejected,
            EventMsg {
                kind: "rejected",
                value: 0
            }
        );
    }
}
