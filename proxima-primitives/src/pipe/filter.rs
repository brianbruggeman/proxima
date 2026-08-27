use alloc::sync::Arc;
use bytes::Bytes;
use core::fmt::Debug;
use core::future::Future;
use core::marker::PhantomData;
use portable_atomic::{AtomicU64, Ordering};

use crate::pipe::SendPipe;
use crate::pipe::ext::PipeExt;
use crate::pipe::handler::{PipeHandle, into_handle};
use crate::pipe::primitives::{Pipe, UnpinPipe, UnpinSendPipe};
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
//
// `Predicate`/`FilterConfig` themselves used to hardcode `In = Out =
// Request<Bytes>`, `Err = ProximaError` in every trait impl — first-pass
// residue from this crate being HTTP-only at the time, not a design
// constraint. The pattern (a decision is `In -> Result<In, Err>`) is
// payload-agnostic, so the types are now generic over `In`/`Err` too, each
// defaulted to the crate's own HTTP types so every existing caller keeps
// compiling unparameterized. `Err: From<RejectMode>` is the seam that
// lets a reject build an `Err` value of a type this module has never
// heard of -- the same error law `AndThen` already uses for its own
// `Second::Err: From<First::Err>` bound (`primitives.rs:207`), not a
// parallel one.

impl From<RejectMode> for ProximaError {
    fn from(mode: RejectMode) -> Self {
        match mode {
            RejectMode::Drop => ProximaError::Forbidden("forbidden".into()),
            RejectMode::Error => {
                ProximaError::Config("filter: predicate rejected request".into())
            }
        }
    }
}

/// The config-expressible predicate set, generic over the payload `In` it
/// passes through unchanged on admit and the `Err` it rejects with.
/// Implements [`SendPipe`]/[`Pipe`] directly (`In -> Result<In, Err>`):
/// `Ok(input)` on admit, `Err::forbidden()` on reject — the same payload
/// [`FilterConfig`]'s [`RejectMode::Drop`] produces, since drop was always
/// this crate's default reject mode ([`FilterConfig::default`]). Defaults to
/// the crate's own HTTP types (`Request<Bytes>`, [`ProximaError`]) so a
/// caller instantiates `Predicate::<MyIn, MyErr>` to gate their own payload
/// without minting a type and without touching this one.
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", bound = "")]
pub enum Predicate<In = Request<Bytes>, Err = ProximaError> {
    Always {
        #[serde(skip)]
        _marker: PhantomData<fn(In) -> Err>,
    },
    Never {
        #[serde(skip)]
        _marker: PhantomData<fn(In) -> Err>,
    },
    When {
        #[serde(flatten)]
        gate: When,
        #[serde(skip)]
        calls: Arc<AtomicU64>,
        #[serde(skip)]
        _marker: PhantomData<fn(In) -> Err>,
    },
    Unless {
        #[serde(flatten)]
        gate: When,
        #[serde(skip)]
        calls: Arc<AtomicU64>,
        #[serde(skip)]
        _marker: PhantomData<fn(In) -> Err>,
    },
}

// hand-rolled instead of `#[derive(Debug, Clone, PartialEq)]`: rustc's
// built-in derives add an `In: Trait, Err: Trait` bound for every generic
// parameter that appears anywhere in the definition, including inside
// `PhantomData` — which would force every instantiation's payload and error
// type to implement Debug/Clone/PartialEq even though neither is ever
// stored. Same fix `Race<Sink, Policy>` (`race.rs`) already uses for its own
// `PhantomData<fn() -> Policy>` marker.
impl<In, Err> Debug for Predicate<In, Err> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Predicate::Always { .. } => formatter.debug_struct("Always").finish(),
            Predicate::Never { .. } => formatter.debug_struct("Never").finish(),
            Predicate::When { gate, calls, .. } => formatter
                .debug_struct("When")
                .field("gate", gate)
                .field("calls", &calls.load(Ordering::Relaxed))
                .finish(),
            Predicate::Unless { gate, calls, .. } => formatter
                .debug_struct("Unless")
                .field("gate", gate)
                .field("calls", &calls.load(Ordering::Relaxed))
                .finish(),
        }
    }
}

impl<In, Err> Clone for Predicate<In, Err> {
    fn clone(&self) -> Self {
        match self {
            Predicate::Always { .. } => Predicate::Always {
                _marker: PhantomData,
            },
            Predicate::Never { .. } => Predicate::Never {
                _marker: PhantomData,
            },
            Predicate::When { gate, calls, .. } => Predicate::When {
                gate: *gate,
                calls: Arc::clone(calls),
                _marker: PhantomData,
            },
            Predicate::Unless { gate, calls, .. } => Predicate::Unless {
                gate: *gate,
                calls: Arc::clone(calls),
                _marker: PhantomData,
            },
        }
    }
}

impl<In, Err> PartialEq for Predicate<In, Err> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Predicate::Always { .. }, Predicate::Always { .. }) => true,
            (Predicate::Never { .. }, Predicate::Never { .. }) => true,
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

impl<In, Err> Predicate<In, Err> {
    #[must_use]
    pub fn always() -> Self {
        Predicate::Always {
            _marker: PhantomData,
        }
    }

    #[must_use]
    pub fn never() -> Self {
        Predicate::Never {
            _marker: PhantomData,
        }
    }

    #[must_use]
    pub fn when(gate: When) -> Self {
        Predicate::When {
            gate,
            calls: Arc::new(AtomicU64::new(0)),
            _marker: PhantomData,
        }
    }

    #[must_use]
    pub fn unless(gate: When) -> Self {
        Predicate::Unless {
            gate,
            calls: Arc::new(AtomicU64::new(0)),
            _marker: PhantomData,
        }
    }

    /// Whether this call admits — the gate's own answer, ignoring the item
    /// (every variant here decides from its own state, never the payload).
    fn admits(&self) -> bool {
        match self {
            Predicate::Always { .. } => true,
            Predicate::Never { .. } => false,
            Predicate::When { gate, calls, .. } => {
                let index = calls.fetch_add(1, Ordering::Relaxed);
                gate.fires(index)
            }
            Predicate::Unless { gate, calls, .. } => {
                let index = calls.fetch_add(1, Ordering::Relaxed);
                !gate.fires(index)
            }
        }
    }
}

impl<In, Err> SendPipe for Predicate<In, Err>
where
    In: Send + 'static,
    Err: From<RejectMode> + Debug + Send + 'static,
{
    type In = In;
    type Out = In;
    type Err = Err;

    fn call(&self, input: In) -> impl Future<Output = Result<In, Err>> + Send {
        let admits = self.admits();
        async move {
            if admits {
                Ok(input)
            } else {
                Err(Err::from(RejectMode::Drop))
            }
        }
    }
}

// base tier stands alone rather than delegating to `SendPipe::call`: `Pipe`
// carries no `Send` bound, so a `Predicate<In, Err>` instantiated at a
// non-`Send` `In`/`Err` must still get this impl.
impl<In, Err> Pipe for Predicate<In, Err>
where
    In: 'static,
    Err: From<RejectMode> + Debug + 'static,
{
    type In = In;
    type Out = In;
    type Err = Err;

    fn call(&self, input: In) -> impl Future<Output = Result<In, Err>> {
        let admits = self.admits();
        async move {
            if admits {
                Ok(input)
            } else {
                Err(Err::from(RejectMode::Drop))
            }
        }
    }
}

// `admits()` resolves synchronously (a plain in-memory decision, no `.await`
// anywhere in the real body above) — `core::future::ready` is the exact
// future that describes that, and it's `Unpin` unconditionally, so no
// hand-written poll struct is needed for a leaf pipe that never suspends.
impl<In, Err> UnpinPipe for Predicate<In, Err>
where
    In: 'static,
    Err: From<RejectMode> + Debug + 'static,
{
    type In = In;
    type Out = In;
    type Err = Err;

    fn call(&self, input: In) -> impl Future<Output = Result<In, Err>> + Unpin {
        let admits = self.admits();
        core::future::ready(if admits {
            Ok(input)
        } else {
            Err(Err::from(RejectMode::Drop))
        })
    }
}

impl<In, Err> UnpinSendPipe for Predicate<In, Err>
where
    In: Send + 'static,
    Err: From<RejectMode> + Debug + Send + 'static,
{
    type In = In;
    type Out = In;
    type Err = Err;

    fn call(&self, input: In) -> impl Future<Output = Result<In, Err>> + Send + Unpin {
        let admits = self.admits();
        core::future::ready(if admits {
            Ok(input)
        } else {
            Err(Err::from(RejectMode::Drop))
        })
    }
}

/// What error a rejected call produces — a rename of the former `OnReject`,
/// same two variants, same JSON strings. [`FilterConfig::call`] converts it
/// straight into `Err` via `Err::from(self.on_reject)`; it is plain data
/// read once per call, not a combinator, and carries no `In`/`Err` of its
/// own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RejectMode {
    Error,
    Drop,
}

// ── serde config + factory ───────────────────────────────────────────────────

/// The predicate-gated pass/reject decision, as config and as the pipe
/// itself: `FilterConfig` is both the 1:1 serialisable mirror AND the
/// `SendPipe`/`Pipe` implementor (`In -> Result<In, Err>`, generic over both
/// like [`Predicate`]) — its two fields are exactly what a decision needs
/// (the gate, and which error a reject produces), so no separate combinator
/// carries them, and no separate type is needed to use it at a payload of
/// the caller's own: `FilterConfig { predicate, on_reject }.and_then(inner)`
/// composes at any `In`/`Err` the same way `Predicate` does. Only
/// [`FilterConfig::into_filter`]/[`FilterConfig::from_spec`] stay pinned to
/// this crate's HTTP [`PipeHandle`], because [`PipeFactory`]'s own signature
/// is pinned there — not a constraint this module introduces.
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FilterConfig<In = Request<Bytes>, Err = ProximaError> {
    pub predicate: Predicate<In, Err>,
    pub on_reject: RejectMode,
}

// hand-rolled for the same reason as `Predicate`'s own Debug/Clone/PartialEq
// (see the comment above those impls): a derive would add a spurious
// `In: Trait, Err: Trait` bound neither field actually needs.
impl<In, Err> Debug for FilterConfig<In, Err> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("FilterConfig")
            .field("predicate", &self.predicate)
            .field("on_reject", &self.on_reject)
            .finish()
    }
}

impl<In, Err> Clone for FilterConfig<In, Err> {
    fn clone(&self) -> Self {
        Self {
            predicate: self.predicate.clone(),
            on_reject: self.on_reject,
        }
    }
}

impl<In, Err> PartialEq for FilterConfig<In, Err> {
    fn eq(&self, other: &Self) -> bool {
        self.predicate == other.predicate && self.on_reject == other.on_reject
    }
}

impl<In, Err> Default for FilterConfig<In, Err> {
    fn default() -> Self {
        Self {
            predicate: Predicate::always(),
            on_reject: RejectMode::Drop,
        }
    }
}

impl<In, Err> SendPipe for FilterConfig<In, Err>
where
    In: Send + 'static,
    Err: From<RejectMode> + Debug + Send + 'static,
{
    type In = In;
    type Out = In;
    type Err = Err;

    fn call(&self, input: In) -> impl Future<Output = Result<In, Err>> + Send {
        let admits = self.predicate.admits();
        let on_reject = self.on_reject;
        async move {
            if admits {
                Ok(input)
            } else {
                Err(Err::from(on_reject))
            }
        }
    }
}

impl<In, Err> Pipe for FilterConfig<In, Err>
where
    In: 'static,
    Err: From<RejectMode> + Debug + 'static,
{
    type In = In;
    type Out = In;
    type Err = Err;

    fn call(&self, input: In) -> impl Future<Output = Result<In, Err>> {
        let admits = self.predicate.admits();
        let on_reject = self.on_reject;
        async move {
            if admits {
                Ok(input)
            } else {
                Err(Err::from(on_reject))
            }
        }
    }
}

// same rationale as `Predicate`'s `UnpinPipe`/`UnpinSendPipe` impls: the
// decision is synchronous, so `core::future::ready` is the exact future,
// unconditionally `Unpin`.
impl<In, Err> UnpinPipe for FilterConfig<In, Err>
where
    In: 'static,
    Err: From<RejectMode> + Debug + 'static,
{
    type In = In;
    type Out = In;
    type Err = Err;

    fn call(&self, input: In) -> impl Future<Output = Result<In, Err>> + Unpin {
        let admits = self.predicate.admits();
        let on_reject = self.on_reject;
        core::future::ready(if admits {
            Ok(input)
        } else {
            Err(Err::from(on_reject))
        })
    }
}

impl<In, Err> UnpinSendPipe for FilterConfig<In, Err>
where
    In: Send + 'static,
    Err: From<RejectMode> + Debug + Send + 'static,
{
    type In = In;
    type Out = In;
    type Err = Err;

    fn call(&self, input: In) -> impl Future<Output = Result<In, Err>> + Send + Unpin {
        let admits = self.predicate.admits();
        let on_reject = self.on_reject;
        core::future::ready(if admits {
            Ok(input)
        } else {
            Err(Err::from(on_reject))
        })
    }
}

// `into_filter`/`from_spec` stay pinned to `Request<Bytes>`/`ProximaError`:
// they compose with `crate::pipe::handler::PipeHandle`, which IS that fixed
// alias (`alloc_tier::PipeHandle<Request<Bytes>, Response<Bytes>>`), and
// `from_spec` backs `FilterFactory`, whose `PipeFactory` trait is dyn-object
// safe and fixed to the same alias for the same reason every other factory
// in this module is (`DelayFactory`, `RetryFactory`, ...). That pin is
// pre-existing workspace architecture (`pipe_factory.rs`, `alloc_tier.rs`),
// not something `Predicate`/`FilterConfig` themselves impose any more.
impl FilterConfig<Request<Bytes>, ProximaError> {
    /// Compose the decision in front of `inner` and erase — `predicate.
    /// and_then(inner)` in one call, matching every other `into_*` factory
    /// entry point in this crate (`Delay::into_delay`, `Transform::
    /// into_transform`, ...).
    #[must_use]
    pub fn into_filter(self, inner: PipeHandle) -> PipeHandle {
        into_handle(PipeExt::and_then(self, inner))
    }

    #[cfg(feature = "std")]
    pub fn from_spec(inner: PipeHandle, value: &Value) -> Result<PipeHandle, ProximaError> {
        let config: FilterConfig<Request<Bytes>, ProximaError> =
            serde_json::from_value(value.clone())
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
            predicate: Predicate::always(),
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
            predicate: Predicate::never(),
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
            predicate: Predicate::never(),
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
            predicate: Predicate::never(),
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

    // the reject sentinel a non-HTTP `Err` needs: `Predicate`/`FilterConfig`
    // pass `In` straight through on admit, so reusing `SensorReading` as
    // both `Out` and `Err` (as `Threshold` below already does by hand) is
    // the natural instantiation, not a special case this impl carves out.
    impl From<RejectMode> for SensorReading {
        fn from(_mode: RejectMode) -> Self {
            SensorReading { celsius: i32::MIN }
        }
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

    // base-tier mirror, delegating straight through — every pipe implements
    // the root `Pipe` too, which is what lets `PipeExt::and_then` reach it.
    impl Pipe for Threshold {
        type In = SensorReading;
        type Out = SensorReading;
        type Err = SensorReading;

        fn call(
            &self,
            reading: SensorReading,
        ) -> impl Future<Output = Result<SensorReading, SensorReading>> {
            SendPipe::call(self, reading)
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

    impl Pipe for ReadingSink {
        type In = SensorReading;
        type Out = SensorReading;
        type Err = SensorReading;

        fn call(
            &self,
            input: SensorReading,
        ) -> impl Future<Output = Result<SensorReading, SensorReading>> {
            SendPipe::call(self, input)
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

    // the same claim as `filter_is_generic_over_a_non_http_payload`, but for
    // `Predicate`/`FilterConfig` themselves rather than a hand-rolled
    // decision pipe: `Predicate::<SensorReading, SensorReading>` and
    // `FilterConfig::<SensorReading, SensorReading>` are instantiated here
    // purely by type inference from `.and_then(ReadingSink)` and the
    // `SendPipe::call` argument below — no turbofish, no new type minted,
    // and `filter.rs` was never touched to add this payload.
    #[proxima::test]
    async fn predicate_and_filter_config_are_generic_over_a_non_http_payload() {
        let always_admits =
            Predicate::<SensorReading, SensorReading>::when(When::prob(1.0).seed(1))
                .and_then(ReadingSink);
        let admitted = SendPipe::call(&always_admits, SensorReading { celsius: 20 }).await;
        assert_eq!(admitted, Ok(SensorReading { celsius: 21 }));

        let stack = FilterConfig::<SensorReading, SensorReading> {
            predicate: Predicate::never(),
            on_reject: RejectMode::Drop,
        }
        .and_then(ReadingSink);
        let dropped = SendPipe::call(&stack, SensorReading { celsius: 20 }).await;
        assert_eq!(dropped, Err(SensorReading { celsius: i32::MIN }));

        let error_mode = FilterConfig::<SensorReading, SensorReading> {
            predicate: Predicate::never(),
            on_reject: RejectMode::Error,
        }
        .and_then(ReadingSink);
        let errored = SendPipe::call(&error_mode, SensorReading { celsius: 20 }).await;
        assert_eq!(
            errored,
            Err(SensorReading { celsius: i32::MIN }),
            "RejectMode::Error still routes through From<RejectMode> for a non-HTTP Err"
        );
    }

    // ── UnpinPipe / UnpinSendPipe tier (Stage 2) ────────────────────────────

    fn poll_ready<F: Future + core::marker::Unpin>(mut future: F) -> F::Output {
        let waker = core::task::Waker::noop();
        let mut cx = core::task::Context::from_waker(waker);
        match Pin::new(&mut future).poll(&mut cx) {
            core::task::Poll::Ready(output) => output,
            core::task::Poll::Pending => {
                panic!("Predicate/FilterConfig never suspend; expected Ready")
            }
        }
    }

    #[test]
    fn predicate_unpin_pipe_matches_send_pipe_on_admit_and_reject() {
        let admits: Predicate = Predicate::always();
        let rejects: Predicate = Predicate::never();

        assert!(
            poll_ready(UnpinPipe::call(&admits, build_request())).is_ok(),
            "Always admits"
        );
        assert!(
            matches!(
                poll_ready(UnpinPipe::call(&rejects, build_request())),
                Err(ProximaError::Forbidden(_))
            ),
            "Never rejects with Forbidden"
        );
    }

    #[test]
    fn predicate_unpin_send_pipe_is_send_and_unpin() {
        fn needs_send_unpin<F: Future + Send + Unpin>(_: &F) {}
        let predicate: Predicate = Predicate::always();
        let call = UnpinSendPipe::call(&predicate, build_request());
        needs_send_unpin(&call);
    }

    #[test]
    fn filter_config_unpin_pipe_matches_send_pipe_on_admit_and_reject() {
        let admits: FilterConfig = FilterConfig {
            predicate: Predicate::always(),
            on_reject: RejectMode::Drop,
        };
        let rejects: FilterConfig = FilterConfig {
            predicate: Predicate::never(),
            on_reject: RejectMode::Error,
        };

        assert!(poll_ready(UnpinPipe::call(&admits, build_request())).is_ok());
        assert!(matches!(
            poll_ready(UnpinPipe::call(&rejects, build_request())),
            Err(ProximaError::Config(_))
        ));
    }

    #[test]
    fn filter_config_unpin_send_pipe_is_send_and_unpin() {
        fn needs_send_unpin<F: Future + Send + Unpin>(_: &F) {}
        let config: FilterConfig = FilterConfig::default();
        let call = UnpinSendPipe::call(&config, build_request());
        needs_send_unpin(&call);
    }
}
