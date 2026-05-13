use alloc::sync::Arc;
use bytes::Bytes;
use core::future::Future;
use portable_atomic::{AtomicU64, Ordering};

use crate::pipe::SendPipe;
use crate::pipe::handler::{PipeHandle, into_handle};
use crate::pipe::primitives::Pipe;
use crate::pipe::when::When;
use serde::{Deserialize, Serialize};

use crate::pipe::request::Request;
use proxima_core::ProximaError;

#[cfg(feature = "std")]
use crate::pipe::pipe_factory::PipeFactory;
#[cfg(feature = "std")]
use serde_json::Value;
#[cfg(feature = "std")]
use std::pin::Pin;

// ── predicate seam ────────────────────────────────────────────────────────────
//
// A filter used to be a bespoke combinator (`Filter<Inner, Predicate>`)
// re-implementing pass/reject short-circuiting next to `AndThen`, fed by a
// `Decide<In>::decide(&self, &In) -> bool` seam that threw the item and the
// rejection reason away — so `Rejectable` and `OnReject` grew up beside it to
// carry them back. The collapse: a decision IS a pipe, `In -> Result<In,
// Err>` (`Ok` = admit, the item survives; `Err` = reject). A filter is then
// just `predicate.and_then(inner)` — `AndThen`'s own `?` already
// short-circuits the inner pipe on a first-stage `Err` (see
// `primitives.rs`'s `and_then_short_circuits_before_the_second_stage_on_first_stage_error`),
// so nothing new is needed here.

/// The config-expressible predicate set. Implements [`SendPipe`]/[`Pipe`]
/// directly (`In = Out = Request<Bytes>`): `Ok(request)` on admit,
/// `Err(ProximaError::Forbidden(..))` on reject — the same 403 payload
/// [`FilterConfig`]'s [`RejectMode::Drop`] produces, since drop was always
/// this crate's default reject mode ([`FilterConfig::default`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Predicate {
    Always,
    Never,
    When {
        #[serde(flatten)]
        gate: When,
        #[serde(skip)]
        calls: Arc<AtomicU64>,
    },
    Unless {
        #[serde(flatten)]
        gate: When,
        #[serde(skip)]
        calls: Arc<AtomicU64>,
    },
}

impl PartialEq for Predicate {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Predicate::Always, Predicate::Always) => true,
            (Predicate::Never, Predicate::Never) => true,
            (Predicate::When { gate: left, .. }, Predicate::When { gate: right, .. }) => {
                left == right
            }
            (Predicate::Unless { gate: left, .. }, Predicate::Unless { gate: right, .. }) => {
                left == right
            }
            _ => false,
        }
    }
}

impl Predicate {
    #[must_use]
    pub fn when(gate: When) -> Self {
        Predicate::When {
            gate,
            calls: Arc::new(AtomicU64::new(0)),
        }
    }

    #[must_use]
    pub fn unless(gate: When) -> Self {
        Predicate::Unless {
            gate,
            calls: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Whether this call admits — the gate's own answer, ignoring the item
    /// (every variant here decides from its own state, never the payload).
    fn admits(&self) -> bool {
        match self {
            Predicate::Always => true,
            Predicate::Never => false,
            Predicate::When { gate, calls } => {
                let index = calls.fetch_add(1, Ordering::Relaxed);
                gate.fires(index)
            }
            Predicate::Unless { gate, calls } => {
                let index = calls.fetch_add(1, Ordering::Relaxed);
                !gate.fires(index)
            }
        }
    }
}

impl SendPipe for Predicate {
    type In = Request<Bytes>;
    type Out = Request<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        input: Request<Bytes>,
    ) -> impl Future<Output = Result<Request<Bytes>, ProximaError>> + Send {
        let admits = self.admits();
        async move {
            if admits {
                Ok(input)
            } else {
                Err(ProximaError::Forbidden("forbidden".into()))
            }
        }
    }
}

impl Pipe for Predicate {
    type In = Request<Bytes>;
    type Out = Request<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        input: Request<Bytes>,
    ) -> impl Future<Output = Result<Request<Bytes>, ProximaError>> {
        SendPipe::call(self, input)
    }
}

/// What error a rejected call produces — a rename of the former `OnReject`,
/// same two variants, same JSON strings. It selects WHICH `ProximaError`
/// [`FilterConfig::call`] builds on reject; it is plain data read once per
/// call, not a combinator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RejectMode {
    Error,
    Drop,
}

fn reject_error(mode: RejectMode) -> ProximaError {
    match mode {
        RejectMode::Drop => ProximaError::Forbidden("forbidden".into()),
        RejectMode::Error => ProximaError::Config("filter: predicate rejected request".into()),
    }
}

// ── serde config + factory ───────────────────────────────────────────────────

/// The predicate-gated pass/reject decision, as config and as the pipe
/// itself: `FilterConfig` is both the 1:1 serialisable mirror AND the
/// `SendPipe`/`Pipe` implementor (`In = Out = Request<Bytes>`) — its two
/// fields are exactly what a decision needs (the gate, and which error a
/// reject produces), so no separate combinator carries them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilterConfig {
    pub predicate: Predicate,
    pub on_reject: RejectMode,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            predicate: Predicate::Always,
            on_reject: RejectMode::Drop,
        }
    }
}

impl SendPipe for FilterConfig {
    type In = Request<Bytes>;
    type Out = Request<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        input: Request<Bytes>,
    ) -> impl Future<Output = Result<Request<Bytes>, ProximaError>> + Send {
        let admits = self.predicate.admits();
        let on_reject = self.on_reject;
        async move {
            if admits {
                Ok(input)
            } else {
                Err(reject_error(on_reject))
            }
        }
    }
}

impl Pipe for FilterConfig {
    type In = Request<Bytes>;
    type Out = Request<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        input: Request<Bytes>,
    ) -> impl Future<Output = Result<Request<Bytes>, ProximaError>> {
        SendPipe::call(self, input)
    }
}

impl FilterConfig {
    /// Compose the decision in front of `inner` and erase — `predicate.
    /// and_then(inner)` in one call, matching every other `into_*` factory
    /// entry point in this crate (`Delay::into_delay`, `Transform::
    /// into_transform`, ...).
    #[must_use]
    pub fn into_filter(self, inner: PipeHandle) -> PipeHandle {
        into_handle(SendPipe::and_then(self, inner))
    }

    #[cfg(feature = "std")]
    pub fn from_spec(inner: PipeHandle, value: &Value) -> Result<PipeHandle, ProximaError> {
        let config: FilterConfig = serde_json::from_value(value.clone())
            .map_err(|err| ProximaError::Config(format!("filter config: {err}")))?;
        Ok(config.into_filter(inner))
    }
}

#[cfg(feature = "std")]
pub struct FilterFactory;

#[cfg(feature = "std")]
impl PipeFactory for FilterFactory {
    fn name(&self) -> &str {
        "filter"
    }

    fn build(
        &self,
        spec: &Value,
        inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        Box::pin(async move {
            let inner = inner
                .ok_or_else(|| ProximaError::Config("filter requires an inner pipe".into()))?;
            FilterConfig::from_spec(inner, &spec)
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
    use std::sync::atomic::{AtomicUsize, Ordering as StdOrdering};

    use super::*;
    use crate::pipe::handler::into_handle;
    use crate::pipe::request::{Request, Response};

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

    // proves the inner pipe is never reached on a reject — the same claim
    // `reject_with_drop_produces_a_forbidden_error` made by inspecting the
    // old sentinel `Out`, now made directly since a reject no longer reaches
    // the inner pipe at all (it short-circuits in the `Err` channel).
    fn counting_echo_pipe() -> (PipeHandle, Arc<AtomicUsize>) {
        struct CountingEcho(Arc<AtomicUsize>);
        impl SendPipe for CountingEcho {
            type In = Request<Bytes>;
            type Out = Response<Bytes>;
            type Err = ProximaError;

            fn call(
                &self,
                request: Request<Bytes>,
            ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
                self.0.fetch_add(1, StdOrdering::Relaxed);
                async move {
                    let (_, body) = request.body_bytes().await?;
                    Ok(Response::new(200).with_body(body))
                }
            }
        }
        let calls = Arc::new(AtomicUsize::new(0));
        (into_handle(CountingEcho(calls.clone())), calls)
    }

    fn build_request() -> Request<Bytes> {
        Request::builder()
            .method("POST")
            .path("/")
            .body(bytes::Bytes::from_static(b"hello"))
            .build()
            .expect("builder")
    }

    #[proxima::test]
    async fn passes_through_when_predicate_is_true() {
        let stack = FilterConfig {
            predicate: Predicate::Always,
            on_reject: RejectMode::Drop,
        }
        .into_filter(echo_pipe());
        let response = SendPipe::call(&stack, build_request()).await.expect("call");
        assert_eq!(response.status, 200);
        let body = response.collect_body().await.expect("body");
        assert_eq!(
            &body[..],
            b"hello",
            "passing request reaches the inner echo pipe"
        );
    }

    #[proxima::test]
    async fn reject_with_error_returns_err() {
        let stack = FilterConfig {
            predicate: Predicate::Never,
            on_reject: RejectMode::Error,
        }
        .into_filter(echo_pipe());
        let outcome = SendPipe::call(&stack, build_request()).await;
        assert!(
            matches!(outcome, Err(ProximaError::Config(_))),
            "RejectMode::Error surfaces the config error"
        );
    }

    #[proxima::test]
    async fn reject_with_drop_produces_a_forbidden_error() {
        let (inner, calls) = counting_echo_pipe();
        let stack = FilterConfig {
            predicate: Predicate::Never,
            on_reject: RejectMode::Drop,
        }
        .into_filter(inner);
        let outcome = SendPipe::call(&stack, build_request()).await;
        match outcome {
            Err(ProximaError::Forbidden(payload)) => {
                assert_eq!(
                    payload, "forbidden",
                    "RejectMode::Drop's payload is the edge's 403 body verbatim"
                );
            }
            other => panic!("expected Forbidden, got {other:?}"),
        }
        assert_eq!(
            calls.load(StdOrdering::Relaxed),
            0,
            "a rejected call never reaches the inner pipe"
        );
    }

    #[test]
    fn config_clone_and_serde_round_trip_are_lossless() {
        let config = FilterConfig {
            predicate: Predicate::Never,
            on_reject: RejectMode::Error,
        };

        assert_eq!(
            config.clone(),
            config,
            "the config's own Clone/PartialEq round-trips"
        );
        let json = serde_json::to_value(&config).expect("serialize");
        let parsed: FilterConfig = serde_json::from_value(json).expect("deserialize");
        assert_eq!(parsed, config, "serde round-trip is lossless");
    }

    #[proxima::test]
    async fn from_spec_parses_serde_config() {
        let value = serde_json::json!({
            "predicate": {"kind": "never"},
            "on_reject": "drop",
        });
        let stack = FilterConfig::from_spec(echo_pipe(), &value).expect("from spec");
        let outcome = SendPipe::call(&stack, build_request()).await;
        assert!(
            matches!(outcome, Err(ProximaError::Forbidden(_))),
            "never + drop rejects with Forbidden, proving the parsed config actually runs"
        );
    }

    #[proxima::test]
    async fn when_gated_filter_drops_and_passes_by_seed() {
        let gate = When::prob(0.5).seed(0x5EED);
        let stack = FilterConfig {
            predicate: Predicate::when(gate),
            on_reject: RejectMode::Drop,
        }
        .into_filter(echo_pipe());

        let mut observed = Vec::new();
        for _ in 0..32_u64 {
            let outcome = SendPipe::call(&stack, build_request()).await;
            observed.push(outcome.is_ok());
        }

        let expected: Vec<bool> = (0..32_u64).map(|index| gate.fires(index)).collect();
        assert_eq!(
            observed, expected,
            "filter pass/drop tracks the gate's deterministic sequence"
        );
    }

    #[proxima::test]
    async fn when_prob_one_always_passes_and_prob_zero_always_drops() {
        let always = FilterConfig {
            predicate: Predicate::when(When::prob(1.0).seed(3)),
            on_reject: RejectMode::Drop,
        }
        .into_filter(echo_pipe());
        let never = FilterConfig {
            predicate: Predicate::when(When::prob(0.0).seed(3)),
            on_reject: RejectMode::Drop,
        }
        .into_filter(echo_pipe());
        for _ in 0..16 {
            let passed = SendPipe::call(&always, build_request()).await;
            assert!(passed.is_ok(), "prob 1.0 always passes");
            let dropped = SendPipe::call(&never, build_request()).await;
            assert!(dropped.is_err(), "prob 0.0 always drops");
        }
    }

    #[test]
    fn when_config_round_trips_and_flattens_to_a_single_tagged_object() {
        let config = FilterConfig {
            predicate: Predicate::when(When::prob(0.3).seed(0xC0FFEE)),
            on_reject: RejectMode::Drop,
        };

        let json = serde_json::to_value(&config).expect("serialize");
        let parsed: FilterConfig = serde_json::from_value(json.clone()).expect("deserialize");

        assert_eq!(parsed, config, "serde round-trip is lossless");
        assert_eq!(
            json,
            serde_json::json!({
                "predicate": {"kind": "when", "prob": 0.3, "seed": 0xC0FFEE_u64},
                "on_reject": "drop",
            }),
            "When flattens to a single tagged object"
        );
    }

    #[proxima::test]
    async fn unless_gated_filter_rejects_on_a_fire() {
        let gate = When::prob(0.5).seed(0x5EED);
        let stack = FilterConfig {
            predicate: Predicate::unless(gate),
            on_reject: RejectMode::Drop,
        }
        .into_filter(echo_pipe());

        let mut observed = Vec::new();
        for _ in 0..32_u64 {
            let outcome = SendPipe::call(&stack, build_request()).await;
            observed.push(outcome.is_err());
        }

        let expected: Vec<bool> = (0..32_u64).map(|index| gate.fires(index)).collect();
        assert_eq!(
            observed, expected,
            "Unless drops exactly when the gate fires"
        );
    }

    #[test]
    fn unless_config_round_trips_with_its_own_tag() {
        let config = FilterConfig {
            predicate: Predicate::unless(When::prob(0.3).seed(7)),
            on_reject: RejectMode::Drop,
        };
        let json = serde_json::to_value(&config).expect("serialize");
        let parsed: FilterConfig = serde_json::from_value(json.clone()).expect("deserialize");
        assert_eq!(parsed, config, "Unless round-trips through config");
        assert_eq!(
            json,
            serde_json::json!({
                "predicate": {"kind": "unless", "prob": 0.3, "seed": 7},
                "on_reject": "drop",
            }),
            "Unless carries the `unless` tag, distinct from When"
        );
    }

    #[proxima::test]
    async fn from_spec_parses_when_predicate() {
        let value = serde_json::json!({
            "predicate": {"kind": "when", "prob": 0.25, "seed": 11},
            "on_reject": "error",
        });
        let stack = FilterConfig::from_spec(echo_pipe(), &value).expect("from spec");

        let gate = When::prob(0.25).seed(11);
        let mut observed = Vec::new();
        for _ in 0..16_u64 {
            let outcome = SendPipe::call(&stack, build_request()).await;
            observed.push(outcome.is_ok());
            if let Err(error) = outcome {
                assert!(
                    matches!(error, ProximaError::Config(_)),
                    "on_reject: error surfaces ProximaError::Config, got {error:?}"
                );
            }
        }

        let expected: Vec<bool> = (0..16_u64).map(|index| gate.fires(index)).collect();
        assert_eq!(
            observed, expected,
            "from_spec's when predicate matches the gate's deterministic sequence"
        );
    }

    #[derive(Clone, PartialEq, Debug)]
    struct SensorReading {
        celsius: i32,
    }

    #[derive(Clone)]
    struct Threshold {
        max_celsius: i32,
    }

    // no `Rejectable`/`Decide` seam: Threshold IS the decision pipe, reusing
    // SensorReading as both the admit `Out` and the reject `Err` — the same
    // shape the `filter`/`gate`/`signal` examples teach for a non-HTTP payload.
    impl SendPipe for Threshold {
        type In = SensorReading;
        type Out = SensorReading;
        type Err = SensorReading;

        fn call(
            &self,
            reading: SensorReading,
        ) -> impl Future<Output = Result<SensorReading, SensorReading>> + Send {
            let admits = reading.celsius <= self.max_celsius;
            async move {
                if admits {
                    Ok(reading)
                } else {
                    Err(SensorReading { celsius: i32::MIN })
                }
            }
        }
    }

    #[derive(Clone)]
    struct ReadingSink;

    impl SendPipe for ReadingSink {
        type In = SensorReading;
        type Out = SensorReading;
        type Err = SensorReading;

        fn call(
            &self,
            input: SensorReading,
        ) -> impl Future<Output = Result<SensorReading, SensorReading>> + Send {
            async move {
                Ok(SensorReading {
                    celsius: input.celsius + 1,
                })
            }
        }
    }

    #[proxima::test]
    async fn filter_is_generic_over_a_non_http_payload() {
        let stack = Threshold { max_celsius: 100 }.and_then(ReadingSink);

        let admitted = SendPipe::call(&stack, SensorReading { celsius: 20 }).await;
        assert_eq!(admitted, Ok(SensorReading { celsius: 21 }));

        let dropped = SendPipe::call(&stack, SensorReading { celsius: 250 }).await;
        assert_eq!(dropped, Err(SensorReading { celsius: i32::MIN }));
    }
}
