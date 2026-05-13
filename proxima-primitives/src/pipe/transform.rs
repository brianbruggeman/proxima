use alloc::vec::Vec;
use bytes::Bytes;
use core::future::Future;

use crate::pipe::SendPipe;
pub use crate::pipe::capabilities::ApplyOps;
use crate::pipe::primitives::Pipe;

use crate::pipe::handler::{PipeHandle, ThreadLocalPipeHandle};
use crate::pipe::request::{Request, Response};
use proxima_core::ProximaError;

#[cfg(feature = "std")]
use crate::pipe::mutate::MutateOp;
#[cfg(feature = "std")]
use crate::pipe::handler::into_handle;
#[cfg(feature = "std")]
use crate::pipe::pipe_factory::PipeFactory;
#[cfg(feature = "std")]
use alloc::string::String;
#[cfg(feature = "std")]
use regex::Regex;
#[cfg(feature = "std")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "std")]
use serde_json::Value;
#[cfg(feature = "std")]
use std::pin::Pin;

// ── main struct ───────────────────────────────────────────────────────────────

/// Request/response transformation middleware. Generic over the inner pipe
/// AND the op types; the std default type parameters (the HTTP `RequestOp` /
/// `ResponseOp`, which carry `regex::Regex`) keep all existing call sites
/// unchanged. Under no_std+alloc the HTTP op types are absent, so the defaults
/// fall back to `()`; callers name their own op types explicitly.
#[cfg(feature = "std")]
pub struct Transform<Inner = PipeHandle, InOp = RequestOp, OutOp = ResponseOp> {
    pub inner: Inner,
    pub(crate) in_ops: Vec<InOp>,
    pub(crate) out_ops: Vec<OutOp>,
}

#[cfg(not(feature = "std"))]
pub struct Transform<Inner = PipeHandle, InOp = (), OutOp = ()> {
    pub inner: Inner,
    pub(crate) in_ops: Vec<InOp>,
    pub(crate) out_ops: Vec<OutOp>,
}

impl<Inner, InOp, OutOp> Transform<Inner, InOp, OutOp> {
    #[must_use]
    pub fn new(inner: Inner) -> Self {
        Self {
            inner,
            in_ops: Vec::new(),
            out_ops: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_request_op(mut self, op: InOp) -> Self {
        self.in_ops.push(op);
        self
    }

    #[must_use]
    pub fn with_response_op(mut self, op: OutOp) -> Self {
        self.out_ops.push(op);
        self
    }
}

impl<Inner, InOp, OutOp> SendPipe for Transform<Inner, InOp, OutOp>
where
    Inner: SendPipe + Clone,
    Inner::In: ApplyOps<InOp> + Send,
    Inner::Out: ApplyOps<OutOp>,
    Inner::Err: Send,
    InOp: Clone + Send + Sync + 'static,
    OutOp: Clone + Send + Sync + 'static,
{
    type In = Inner::In;
    type Out = Inner::Out;
    type Err = Inner::Err;

    fn call(
        &self,
        input: Inner::In,
    ) -> impl Future<Output = Result<Inner::Out, Inner::Err>> + Send {
        let mapped = input.apply(&self.in_ops);
        let out_ops = self.out_ops.clone();
        let inner = self.inner.clone();
        async move {
            let output = SendPipe::call(&inner, mapped).await?;
            Ok(output.apply(&out_ops))
        }
    }
}

impl<InOp, OutOp> Pipe for Transform<ThreadLocalPipeHandle, InOp, OutOp>
where
    Request<Bytes>: ApplyOps<InOp>,
    Response<Bytes>: ApplyOps<OutOp>,
    InOp: Clone + Send + Sync + 'static,
    OutOp: Clone + Send + Sync + 'static,
{
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        input: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        let mapped = input.apply(&self.in_ops);
        let out_ops = self.out_ops.clone();
        let inner = self.inner.clone();
        async move {
            let output = Pipe::call(&inner, mapped).await?;
            Ok(output.apply(&out_ops))
        }
    }
}

// ── factory ──────────────────────────────────────────────────────────────────

/// Serialisable request op — the config mirror of [`RequestOp`]. Tagged by
/// `op`; `RewritePath` carries the pattern as a string (compiled to a
/// [`Regex`] at [`TransformConfig::into_transform`] time), and `Mutate` flattens
/// the existing serde [`MutateOp`]. The `body`/`callback` ops are accepted by
/// the wire form but rejected at lowering, preserving the historical errors.
#[cfg(feature = "std")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum RequestOpConfig {
    SetHeader {
        name: String,
        value: String,
    },
    RemoveHeader {
        name: String,
    },
    RewritePath {
        pattern: String,
        template: String,
    },
    Mutate {
        #[serde(flatten)]
        mutate: MutateOp,
    },
    Body,
    SetBody,
    RewriteBody,
    Callback,
}

#[cfg(feature = "std")]
impl RequestOpConfig {
    fn into_op(self) -> Result<RequestOp, ProximaError> {
        match self {
            RequestOpConfig::SetHeader { name, value } => Ok(RequestOp::SetHeader { name, value }),
            RequestOpConfig::RemoveHeader { name } => Ok(RequestOp::RemoveHeader { name }),
            RequestOpConfig::RewritePath { pattern, template } => {
                let pattern = Regex::new(&pattern)
                    .map_err(|err| ProximaError::Config(format!("rewrite_path pattern: {err}")))?;
                Ok(RequestOp::RewritePath { pattern, template })
            }
            RequestOpConfig::Mutate { mutate } => Ok(RequestOp::Mutate(mutate)),
            RequestOpConfig::Body | RequestOpConfig::SetBody | RequestOpConfig::RewriteBody => {
                Err(ProximaError::Config(
                    "transform body ops not implemented in v1; use a downstream pipe for body transforms"
                        .into(),
                ))
            }
            RequestOpConfig::Callback => Err(ProximaError::Config(
                "transform callbacks land in chunk 2 alongside the callback upstream registry".into(),
            )),
        }
    }
}

/// Serialisable response op — the config mirror of [`ResponseOp`].
#[cfg(feature = "std")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ResponseOpConfig {
    SetHeader {
        name: String,
        value: String,
    },
    RemoveHeader {
        name: String,
    },
    Mutate {
        #[serde(flatten)]
        mutate: MutateOp,
    },
}

#[cfg(feature = "std")]
impl ResponseOpConfig {
    fn into_op(self) -> ResponseOp {
        match self {
            ResponseOpConfig::SetHeader { name, value } => ResponseOp::SetHeader { name, value },
            ResponseOpConfig::RemoveHeader { name } => ResponseOp::RemoveHeader { name },
            ResponseOpConfig::Mutate { mutate } => ResponseOp::Mutate(mutate),
        }
    }
}

/// Typed config surface for the `transform` middleware — ordered request- and
/// response-side op pipelines.
#[cfg(feature = "std")]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TransformConfig {
    #[serde(default)]
    pub request_pipeline: alloc::vec::Vec<RequestOpConfig>,
    #[serde(default)]
    pub response_pipeline: alloc::vec::Vec<ResponseOpConfig>,
}

#[cfg(feature = "std")]
impl TransformConfig {
    /// Fluent constructor: start from an empty pipeline.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Fluent: append a request-side op.
    #[must_use]
    pub fn with_request_op(mut self, op: RequestOpConfig) -> Self {
        self.request_pipeline.push(op);
        self
    }

    /// Fluent: append a response-side op.
    #[must_use]
    pub fn with_response_op(mut self, op: ResponseOpConfig) -> Self {
        self.response_pipeline.push(op);
        self
    }

    /// Materialise the transform middleware around `inner`, compiling any
    /// `rewrite_path` patterns and rejecting unimplemented body/callback ops.
    pub fn into_transform(self, inner: PipeHandle) -> Result<Transform<PipeHandle>, ProximaError> {
        let mut transform = Transform::new(inner);
        for op in self.request_pipeline {
            transform.in_ops.push(op.into_op()?);
        }
        for op in self.response_pipeline {
            transform.out_ops.push(op.into_op());
        }
        Ok(transform)
    }
}

#[cfg(feature = "std")]
impl Transform<PipeHandle> {
    pub fn from_spec(inner: PipeHandle, value: &Value) -> Result<Self, ProximaError> {
        let config: TransformConfig = serde_json::from_value(value.clone())
            .map_err(|err| ProximaError::Config(format!("transform config: {err}")))?;
        config.into_transform(inner)
    }
}

#[cfg(feature = "std")]
pub struct TransformFactory;

#[cfg(feature = "std")]
impl PipeFactory for TransformFactory {
    fn name(&self) -> &str {
        "transform"
    }

    fn build(
        &self,
        spec: &Value,
        inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        Box::pin(async move {
            let inner = inner
                .ok_or_else(|| ProximaError::Config("transform requires an inner pipe".into()))?;
            let transform = Transform::from_spec(inner, &spec)?;
            Ok(into_handle(transform))
        })
    }
}

// ── HTTP op types ─────────────────────────────────────────────────────────────
//
// `RequestOp` carries a `regex::Regex` (RewritePath), and the parse/apply path
// uses serde_json + the std-only `MutateOp`/regex; the whole HTTP op surface is
// std-tier. The generic `Transform<Inner, InOp, OutOp>` above is alloc-tier.

#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub enum RequestOp {
    SetHeader { name: String, value: String },
    RemoveHeader { name: String },
    RewritePath { pattern: Regex, template: String },
    Mutate(MutateOp),
}

#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub enum ResponseOp {
    SetHeader { name: String, value: String },
    RemoveHeader { name: String },
    Mutate(MutateOp),
}

#[cfg(feature = "std")]
impl ApplyOps<RequestOp> for Request<Bytes> {
    fn apply(mut self, ops: &[RequestOp]) -> Self {
        for op in ops {
            match op {
                RequestOp::SetHeader { name, value } => {
                    self.metadata.insert(name.clone(), value.clone());
                }
                RequestOp::RemoveHeader { name } => {
                    self.metadata.remove(name);
                }
                RequestOp::RewritePath { pattern, template } => {
                    let path_view = std::str::from_utf8(&self.path).unwrap_or("");
                    let rewritten = pattern.replace(path_view, template.as_str()).into_owned();
                    self.path = bytes::Bytes::from(rewritten);
                }
                RequestOp::Mutate(mutate) => {
                    self = mutate.apply_to(self);
                }
            }
        }
        self
    }
}

#[cfg(feature = "std")]
impl ApplyOps<ResponseOp> for Response<Bytes> {
    fn apply(mut self, ops: &[ResponseOp]) -> Self {
        for op in ops {
            match op {
                ResponseOp::SetHeader { name, value } => {
                    self.metadata.insert(name.clone(), value.clone());
                }
                ResponseOp::RemoveHeader { name } => {
                    self.metadata.remove(name);
                }
                ResponseOp::Mutate(mutate) => {
                    self = mutate.apply_to(self);
                }
            }
        }
        self
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
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::pipe::handler::into_handle;

    struct Capture {
        captured: Arc<Mutex<Option<Request<Bytes>>>>,
    }

    impl Capture {
        fn new() -> (Self, Arc<Mutex<Option<Request<Bytes>>>>) {
            let captured: Arc<Mutex<Option<Request<Bytes>>>> = Arc::new(Mutex::new(None));
            (
                Self {
                    captured: captured.clone(),
                },
                captured,
            )
        }
    }

    impl SendPipe for Capture {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            *self.captured.lock().expect("capture lock") = Some(Request {
                method: request.method.clone(),
                path: request.path.clone(),
                query: request.query.clone(),
                metadata: request.metadata.clone(),
                payload: bytes::Bytes::new(),
                stream: None,
                context: request.context.clone(),
            });
            async move {
                let mut response = Response::ok("ok");
                response.metadata.insert("server", "test/1.0");
                Ok(response)
            }
        }
    }


    fn build_request() -> Request<Bytes> {
        Request::builder()
            .method("GET")
            .path("/v1/items/42")
            .header("user-agent", "rstest")
            .build()
            .expect("builder")
    }

    fn dummy() -> PipeHandle {
        let (capture, _) = Capture::new();
        into_handle(capture)
    }

    #[proxima::test]
    async fn set_header_adds_header_to_request() {
        let (capture, captured) = Capture::new();
        let stack = Transform::new(into_handle(capture)).with_request_op(RequestOp::SetHeader {
            name: "x-trace".into(),
            value: "abc".into(),
        });
        let _ = SendPipe::call(&stack, build_request()).await.expect("call");
        let guard = captured.lock().expect("captured");
        let request = guard.as_ref().expect("captured request");
        assert_eq!(request.metadata.get_str("x-trace"), Some("abc"));
    }

    #[proxima::test]
    async fn remove_header_drops_header_from_request() {
        let (capture, captured) = Capture::new();
        let stack = Transform::new(into_handle(capture)).with_request_op(RequestOp::RemoveHeader {
            name: "user-agent".into(),
        });
        let _ = SendPipe::call(&stack, build_request()).await.expect("call");
        let guard = captured.lock().expect("captured");
        let request = guard.as_ref().expect("captured request");
        assert!(!request.metadata.contains_key("user-agent"));
    }

    #[proxima::test]
    async fn rewrite_path_swaps_v1_to_v2() {
        let (capture, captured) = Capture::new();
        let pattern = Regex::new("^/v1/(.*)").expect("regex");
        let stack = Transform::new(into_handle(capture)).with_request_op(RequestOp::RewritePath {
            pattern,
            template: "/v2/$1".into(),
        });
        let _ = SendPipe::call(&stack, build_request()).await.expect("call");
        let guard = captured.lock().expect("captured");
        let request = guard.as_ref().expect("captured request");
        assert_eq!(request.path, "/v2/items/42");
    }

    #[proxima::test]
    async fn response_remove_header_strips_header_from_response() {
        let (capture, _) = Capture::new();
        let stack =
            Transform::new(into_handle(capture)).with_response_op(ResponseOp::RemoveHeader {
                name: "server".into(),
            });
        let response = SendPipe::call(&stack, build_request()).await.expect("call");
        assert!(!response.metadata.contains_key("server"));
    }

    #[test]
    fn body_op_is_a_config_error_in_v1() {
        let value = serde_json::json!({
            "request_pipeline": [{"op": "body", "value": "x"}],
        });
        let outcome = Transform::from_spec(dummy(), &value);
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[test]
    fn callback_op_is_a_config_error_until_chunk_2() {
        let value = serde_json::json!({
            "request_pipeline": [{"op": "callback", "name": "redact"}],
        });
        let outcome = Transform::from_spec(dummy(), &value);
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[test]
    fn from_spec_parses_set_remove_rewrite_pipeline() {
        let value = serde_json::json!({
            "request_pipeline": [
                {"op": "set_header", "name": "x-corr", "value": "abc"},
                {"op": "remove_header", "name": "cookie"},
                {"op": "rewrite_path", "pattern": "^/old/(.*)", "template": "/new/$1"},
            ],
            "response_pipeline": [
                {"op": "set_header", "name": "x-served-by", "value": "proxima"},
            ],
        });
        let transform = Transform::from_spec(dummy(), &value).expect("parse");
        assert_eq!(transform.in_ops.len(), 3);
        assert_eq!(transform.out_ops.len(), 1);
    }

    fn echo_body_pipe() -> PipeHandle {
        struct EchoBody;
        impl SendPipe for EchoBody {
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
        into_handle(EchoBody)
    }

    fn body_request(body: &'static [u8]) -> Request<Bytes> {
        Request::builder()
            .method("POST")
            .path("/")
            .body(bytes::Bytes::from_static(body))
            .build()
            .expect("builder")
    }

    #[proxima::test]
    async fn response_mutate_corrupts_the_body_deterministically() {
        use crate::pipe::mutate::{MutateOp, Mutation};
        const PAYLOAD: &[u8] = b"the quick brown fox jumps over the lazy dog";
        let stack = Transform::new(echo_body_pipe()).with_response_op(ResponseOp::Mutate(
            MutateOp::new(Mutation::BitFlip { bits: 3 }, 0x5EED),
        ));

        let response = SendPipe::call(&stack, body_request(PAYLOAD))
            .await
            .expect("call");
        let corrupted = response.collect_body().await.expect("body");

        let expected = Mutation::BitFlip { bits: 3 }.mutate(0x5EED, 0, PAYLOAD);
        assert_eq!(
            &corrupted[..],
            &expected[..],
            "transform corrupts with the seeded mutation"
        );
        assert_ne!(&corrupted[..], PAYLOAD, "body was actually corrupted");
    }

    // principle-4 parity: the fluent config builder and the config value must
    // lower to identical Transform op state (counts + the parsed ops).
    #[test]
    fn parity_fluent_builder_and_config_value_match() {
        use crate::pipe::mutate::{MutateOp, Mutation};
        let from_value: TransformConfig = serde_json::from_value(serde_json::json!({
            "request_pipeline": [
                {"op": "set_header", "name": "x-corr", "value": "abc"},
                {"op": "rewrite_path", "pattern": "^/old/(.*)", "template": "/new/$1"},
            ],
            "response_pipeline": [
                {"op": "mutate", "kind": "truncate", "seed": 11},
            ],
        }))
        .expect("from_value");
        let from_value = from_value
            .into_transform(dummy())
            .expect("into_transform value");

        let from_builder = TransformConfig::new()
            .with_request_op(RequestOpConfig::SetHeader {
                name: "x-corr".into(),
                value: "abc".into(),
            })
            .with_request_op(RequestOpConfig::RewritePath {
                pattern: "^/old/(.*)".into(),
                template: "/new/$1".into(),
            })
            .with_response_op(ResponseOpConfig::Mutate {
                mutate: MutateOp::new(Mutation::Truncate, 11),
            })
            .into_transform(dummy())
            .expect("into_transform builder");

        assert_eq!(from_value.in_ops.len(), from_builder.in_ops.len());
        assert_eq!(from_value.out_ops.len(), from_builder.out_ops.len());
        match (&from_value.in_ops[1], &from_builder.in_ops[1]) {
            (
                RequestOp::RewritePath {
                    pattern: left_pattern,
                    template: left_template,
                },
                RequestOp::RewritePath {
                    pattern: right_pattern,
                    template: right_template,
                },
            ) => {
                assert_eq!(left_pattern.as_str(), right_pattern.as_str());
                assert_eq!(left_template, right_template);
            }
            other => panic!("expected matching rewrite_path ops, got {other:?}"),
        }
        match (&from_value.out_ops[0], &from_builder.out_ops[0]) {
            (ResponseOp::Mutate(left), ResponseOp::Mutate(right)) => assert_eq!(left, right),
            other => panic!("expected matching mutate ops, got {other:?}"),
        }
    }

    #[test]
    fn from_spec_parses_mutate_op_into_the_pipeline() {
        use crate::pipe::mutate::{MutateOp, Mutation};
        let value = serde_json::json!({
            "response_pipeline": [
                {"op": "mutate", "kind": "truncate", "seed": 11},
            ],
        });
        let transform = Transform::from_spec(dummy(), &value).expect("parse");
        assert_eq!(transform.out_ops.len(), 1);
        match &transform.out_ops[0] {
            ResponseOp::Mutate(op) => {
                assert_eq!(
                    op,
                    &MutateOp::new(Mutation::Truncate, 11),
                    "parses the seeded mutate op"
                );
            }
            other => panic!("expected a mutate op, got {other:?}"),
        }
    }

    #[derive(Clone, PartialEq, Debug)]
    struct CounterMsg(u64);

    #[derive(Clone)]
    enum CounterMsgOp {
        Double,
        AddConst(u64),
    }

    impl ApplyOps<CounterMsgOp> for CounterMsg {
        fn apply(mut self, ops: &[CounterMsgOp]) -> Self {
            for op in ops {
                match op {
                    CounterMsgOp::Double => self.0 *= 2,
                    CounterMsgOp::AddConst(constant) => self.0 += constant,
                }
            }
            self
        }
    }

    #[derive(Clone)]
    struct CounterSink;

    impl SendPipe for CounterSink {
        type In = CounterMsg;
        type Out = CounterMsg;
        type Err = ProximaError;

        fn call(
            &self,
            input: CounterMsg,
        ) -> impl Future<Output = Result<CounterMsg, ProximaError>> + Send {
            async move { Ok(CounterMsg(input.0 + 1)) }
        }
    }

    #[proxima::test]
    async fn transform_is_generic_over_a_non_http_payload() {
        let stack = Transform::new(CounterSink)
            .with_request_op(CounterMsgOp::Double)
            .with_response_op(CounterMsgOp::AddConst(10));
        let out = SendPipe::call(&stack, CounterMsg(3)).await.expect("call");
        assert_eq!(
            out,
            CounterMsg(17),
            "Double(3)=6 -> sink=7 -> AddConst(10)=17"
        );
    }
}
