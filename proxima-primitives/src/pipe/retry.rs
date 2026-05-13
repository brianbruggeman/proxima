use alloc::collections::BTreeSet;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use bytes::Bytes;
use core::future::Future;
use core::time::Duration;
use portable_atomic::{AtomicU64, Ordering};

use crate::pipe::SendPipe;
use crate::pipe::capabilities::{Clock, Idempotent, Replayable, Retryable};
use crate::pipe::resilience::{Backoff, Jitter, RetryAction, RetryController};
use crate::pipe::retry_rules::RetryRules;

use crate::pipe::clock::TimeClock;
use crate::pipe::labeled::Labeled;
use crate::pipe::handler::PipeHandle;
use crate::pipe::request::{Request, Response};
use crate::pipe::telemetry_surface::{Labels, TelemetryHandle};
use proxima_core::ProximaError;

// the HTTP-handle pipe impls (primitives::Pipe over the dispatch handles) are
// std-tier; their trait paths are only needed there.
#[cfg(feature = "std")]
use crate::pipe::handler::ThreadLocalPipeHandle;
#[cfg(feature = "std")]
use bon::Builder;
#[cfg(feature = "std")]
use conflaguration::{Settings, Validate, ValidationMessage};
#[cfg(feature = "std")]
use crate::pipe::primitives::Pipe;
#[cfg(feature = "std")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "std")]
use serde_json::Value;
#[cfg(feature = "std")]
use std::pin::Pin;
// `Replay` is std-only in proxima-transport (its stream types want std Mutex/Vec
// freely); the HTTP `Replayable` instantiation below is therefore std-tier. The
// generic `Retry<Inner>` core needs only the `Replayable` trait (forms, no_std).
#[cfg(feature = "std")]
use crate::pipe::handler::into_handle;
#[cfg(feature = "std")]
use crate::pipe::pipe_factory::PipeFactory;
#[cfg(feature = "std")]
use crate::transport::Replay;

/// Default replay-tee byte cap for the generic core's `Retry::new`. Mirrors
/// `crate::transport::DEFAULT_REPLAY_CAP_BYTES` (which is std-only), so the
/// alloc tier carries its own copy.
const DEFAULT_REPLAY_CAP_BYTES: usize = 4 * 1024 * 1024;

const DEFAULT_MAX_ATTEMPTS: u32 = 3;
const DEFAULT_BASE_DELAY_MS: u64 = 50;
const DEFAULT_MAX_DELAY_MS: u64 = 2_000;
const DEFAULT_BUDGET_CAPACITY: u64 = 100;
const DEFAULT_BUDGET_REFILL_PER_SUCCESS: u64 = 10;
const RETRY_COST: u64 = 100;

const METRIC_ATTEMPTS: &str = "proxima.retry.attempts_total";
const METRIC_SUCCEEDED: &str = "proxima.retry.succeeded_total";
const METRIC_BUDGET_EXHAUSTED: &str = "proxima.retry.budget_exhausted_total";
const METRIC_REPLAY_CAP_EXCEEDED: &str = "proxima.retry.replay_cap_exceeded_total";

// ── retry budget ──────────────────────────────────────────────────────────────

pub struct RetryBudget {
    capacity: u64,
    tokens: AtomicU64,
    refill_per_success: u64,
}

impl RetryBudget {
    #[must_use]
    pub fn new(capacity: u64, refill_per_success: u64) -> Self {
        Self {
            capacity,
            tokens: AtomicU64::new(capacity),
            refill_per_success,
        }
    }

    #[must_use]
    pub fn tokens(&self) -> u64 {
        self.tokens.load(Ordering::Relaxed)
    }

    pub fn try_consume(&self, cost: u64) -> bool {
        loop {
            let current = self.tokens.load(Ordering::Acquire);
            if current < cost {
                return false;
            }
            if self
                .tokens
                .compare_exchange_weak(
                    current,
                    current - cost,
                    Ordering::Release,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                return true;
            }
        }
    }

    pub fn refill(&self) {
        let cap = self.capacity;
        let amount = self.refill_per_success;
        let _ = self
            .tokens
            .fetch_update(Ordering::Release, Ordering::Acquire, |current| {
                Some((current + amount).min(cap))
            });
    }
}

impl Default for RetryBudget {
    fn default() -> Self {
        Self::new(DEFAULT_BUDGET_CAPACITY, DEFAULT_BUDGET_REFILL_PER_SUCCESS)
    }
}

// ── main struct ───────────────────────────────────────────────────────────────

/// Retry middleware. Generic over the inner pipe AND the clock — composes the
/// pure [`RetryController`] decision core (the same sans-IO engine
/// `resilience::Retry<Inner, Clk>` drives) instead of re-deciding against
/// `RetryRules` directly, and schedules its between-attempt delay against the
/// injected [`Clock`] instead of a bare `sleep`. `Clk` defaults to
/// [`TimeClock`], the production monotonic clock, so every existing caller
/// (`Retry::new`, `RetryConfig`, `RetryFactory`) is unaffected.
pub struct Retry<Inner = PipeHandle, Clk = TimeClock> {
    pub inner: Inner,
    pub rules: RetryRules,
    pub max_attempts: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
    pub budget: Option<Arc<RetryBudget>>,
    pub replay_cap_bytes: usize,
    clock: Clk,
}

// Hardcoded to `TimeClock` (not `Clk: Default`) so `Retry::new(...)` resolves
// without turbofish, the same way `RateLimit::new`/`with_caps` pin `Clk` — a
// bare `Clk: Default` bound would leave the type parameter ambiguous at every
// existing call site with no annotation to pin it.
impl<Inner> Retry<Inner, TimeClock> {
    #[must_use]
    pub fn new(inner: Inner) -> Self {
        Self::with_clock(inner, TimeClock)
    }
}

impl<Inner, Clk> Retry<Inner, Clk> {
    /// Materialise with an explicit clock — the seam a deterministic test or
    /// example injects a fake clock through; production code goes via `new`,
    /// which defaults `Clk` to [`TimeClock`].
    #[must_use]
    pub fn with_clock(inner: Inner, clock: Clk) -> Self {
        Self {
            inner,
            rules: RetryRules::default(),
            max_attempts: DEFAULT_MAX_ATTEMPTS,
            base_delay: Duration::from_millis(DEFAULT_BASE_DELAY_MS),
            max_delay: Duration::from_millis(DEFAULT_MAX_DELAY_MS),
            budget: None,
            replay_cap_bytes: DEFAULT_REPLAY_CAP_BYTES,
            clock,
        }
    }

    #[must_use]
    pub fn with_max_attempts(mut self, attempts: u32) -> Self {
        self.max_attempts = attempts.max(1);
        self
    }

    #[must_use]
    pub fn with_base_delay(mut self, delay: Duration) -> Self {
        self.base_delay = delay;
        self
    }

    #[must_use]
    pub fn with_max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    #[must_use]
    pub fn with_budget(mut self, budget: Arc<RetryBudget>) -> Self {
        self.budget = Some(budget);
        self
    }

    #[must_use]
    pub fn with_predicate(mut self, predicate: RetryPredicate) -> Self {
        match predicate {
            RetryPredicate::OnStatus(set) => self.rules.retry_on_status.extend(set),
            RetryPredicate::OnAnyError => self.rules.retry_on_error = true,
            RetryPredicate::OnIdempotentOnly => self.rules.idempotent_only = true,
        }
        self
    }

    #[must_use]
    pub fn with_replay_cap_bytes(mut self, cap: usize) -> Self {
        self.replay_cap_bytes = cap;
        self
    }

    /// Build the pure decision core for one call, from this middleware's
    /// current knobs. No deadline: `Retry` has never enforced a wall-clock
    /// budget, only an attempt cap — this mirrors that unchanged.
    fn controller(&self) -> RetryController {
        RetryController {
            rules: self.rules.clone(),
            backoff: Backoff::Exponential {
                initial: self.base_delay,
                factor: 2,
                max: self.max_delay,
            },
            jitter: Jitter::Full,
            max_attempts: self.max_attempts,
            deadline: None,
        }
    }
}

async fn run_retry<In, Out, Err, Call, Fut, Clk>(
    args: RunArgs<In, Clk>,
    call_attempt: Call,
) -> Result<Out, Err>
where
    In: Replayable + Idempotent + Labeled,
    Out: Retryable,
    Err: From<ProximaError>,
    Call: Fn(In) -> Fut,
    Fut: Future<Output = Result<Out, Err>>,
    Clk: Clock,
{
    let RunArgs {
        controller,
        clock,
        input,
        budget,
        replay_cap_bytes,
    } = args;
    let telemetry = input.telemetry();
    let base_labels = input.labels();
    let allow = input.is_idempotent() || !controller.rules.idempotent_only;
    let (first_input, source) = input.fork(replay_cap_bytes);

    telemetry.counter_inc(
        METRIC_ATTEMPTS,
        &with_extra(&base_labels, "outcome", "started"),
        1,
    );
    let mut last_outcome = call_attempt(first_input).await;

    if !allow {
        return finalize(last_outcome, &telemetry, &base_labels);
    }

    let mut attempt = 0u32;
    let mut prev_delay = Duration::ZERO;
    loop {
        let now_nanos = clock.now_nanos();
        let rand = attempt_rand(attempt);
        let after = match controller.on_outcome(attempt, &last_outcome, now_nanos, rand, prev_delay)
        {
            RetryAction::Done | RetryAction::Exhausted => break,
            RetryAction::Retry { after } => after,
        };
        if let Some(budget) = budget.as_ref()
            && !budget.try_consume(RETRY_COST)
        {
            telemetry.counter_inc(
                METRIC_BUDGET_EXHAUSTED,
                &with_extra(&base_labels, "reason", "budget"),
                1,
            );
            break;
        }
        if !after.is_zero() {
            clock.delay(after).await;
        }
        prev_delay = after;
        attempt += 1;

        let input = match In::replay(&source) {
            Ok(input) => input,
            Err(error) => {
                telemetry.counter_inc(METRIC_REPLAY_CAP_EXCEEDED, &base_labels, 1);
                return Err(error.into());
            }
        };
        telemetry.counter_inc(
            METRIC_ATTEMPTS,
            &with_extra(&base_labels, "outcome", "started"),
            1,
        );
        last_outcome = call_attempt(input).await;
    }
    finalize(last_outcome, &telemetry, &base_labels)
}

struct RunArgs<In, Clk> {
    controller: RetryController,
    clock: Clk,
    input: In,
    budget: Option<Arc<RetryBudget>>,
    replay_cap_bytes: usize,
}

fn finalize<Out: Retryable, Err>(
    outcome: Result<Out, Err>,
    telemetry: &TelemetryHandle,
    base_labels: &Labels,
) -> Result<Out, Err> {
    if matches!(&outcome, Ok(out) if out.is_success()) {
        telemetry.counter_inc(METRIC_SUCCEEDED, base_labels, 1);
    }
    outcome
}

impl<Inner, Clk> SendPipe for Retry<Inner, Clk>
where
    Inner: SendPipe + Clone + Send + Sync + 'static,
    Inner::In: Replayable + Idempotent + Labeled + Send + 'static,
    Inner::Out: Retryable + Send + 'static,
    Inner::Err: From<ProximaError> + core::fmt::Debug + Send + 'static,
    Clk: Clock + Clone + Send + Sync + 'static,
    Clk::Delay: Send,
{
    type In = Inner::In;
    type Out = Inner::Out;
    type Err = Inner::Err;

    fn call(
        &self,
        input: Inner::In,
    ) -> impl Future<Output = Result<Inner::Out, Inner::Err>> + Send {
        let inner = self.inner.clone();
        run_retry(
            RunArgs {
                controller: self.controller(),
                clock: self.clock.clone(),
                input,
                budget: self.budget.clone(),
                replay_cap_bytes: self.replay_cap_bytes,
            },
            move |attempt_input| {
                let inner = inner.clone();
                async move { SendPipe::call(&inner, attempt_input).await }
            },
        )
    }
}

#[cfg(feature = "std")]
impl<Clk> Pipe for Retry<ThreadLocalPipeHandle, Clk>
where
    ThreadLocalPipeHandle:
        Pipe<In = Request<Bytes>, Out = Response<Bytes>, Err = ProximaError>,
    Clk: Clock + Clone + Send + Sync + 'static,
{
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        input: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        let inner = self.inner.clone();
        run_retry(
            RunArgs {
                controller: self.controller(),
                clock: self.clock.clone(),
                input,
                budget: self.budget.clone(),
                replay_cap_bytes: self.replay_cap_bytes,
            },
            move |attempt_input| {
                let inner = inner.clone();
                async move { Pipe::call(&inner, attempt_input).await }
            },
        )
    }
}

// std: the process-global rng decorrelates jitter entropy across retry
// instances. `Backoff::delay` (invoked by `RetryController::on_outcome`) turns
// this raw entropy into the actual jittered duration — this only sources it.
#[cfg(feature = "std")]
fn attempt_rand(_attempt: u32) -> u64 {
    fastrand::u64(..)
}

// no_std: no process-global rng (fastrand's global needs std for the seed), so
// seed a per-call Rng from the attempt. Deterministic within an instance; the
// backoff envelope is preserved, only cross-instance decorrelation is lost.
#[cfg(not(feature = "std"))]
fn attempt_rand(attempt: u32) -> u64 {
    fastrand::Rng::with_seed(u64::from(attempt)).u64(..)
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

// ── HTTP instantiation ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum RetryPredicate {
    OnStatus(BTreeSet<u16>),
    OnAnyError,
    OnIdempotentOnly,
}

#[cfg(feature = "std")]
pub struct HttpReplay {
    method: crate::pipe::method::Method,
    path: bytes::Bytes,
    query: crate::pipe::header_list::HeaderList,
    metadata: crate::pipe::header_list::HeaderList,
    context: crate::pipe::request::RequestContext,
    tee: Replay<bytes::Bytes>,
}

#[cfg(feature = "std")]
impl Replayable for Request<Bytes> {
    type Source = HttpReplay;

    fn fork(self, replay_cap_bytes: usize) -> (Request<Bytes>, HttpReplay) {
        let Request::<Bytes> {
            method,
            path,
            query,
            metadata,
            payload,
            stream,
            context,
        } = self;
        let (tee, primary_body) =
            Replay::wrap_bytes(source_stream(payload, stream), replay_cap_bytes);
        let first = build_request(&method, &path, &query, &metadata, &context, primary_body);
        (
            first,
            HttpReplay {
                method,
                path,
                query,
                metadata,
                context,
                tee,
            },
        )
    }

    fn replay(source: &HttpReplay) -> Result<Request<Bytes>, ProximaError> {
        let body = source.tee.replay()?;
        Ok(build_request(
            &source.method,
            &source.path,
            &source.query,
            &source.metadata,
            &source.context,
            body,
        ))
    }
}

impl Idempotent for Request<Bytes> {
    fn is_idempotent(&self) -> bool {
        is_idempotent_method(&self.method)
    }
}

impl Retryable for Response<Bytes> {
    fn retry_status(&self) -> Option<u16> {
        Some(self.status)
    }

    fn is_success(&self) -> bool {
        (200..400).contains(&self.status)
    }
}

const RESPONSE_REDELIVERY_IDEMPOTENT_DEFAULT: bool = false;

#[cfg(feature = "std")]
pub struct HttpResponseReplay {
    status: u16,
    metadata: crate::pipe::header_list::HeaderList,
    tee: Replay<bytes::Bytes>,
}

#[cfg(feature = "std")]
impl Replayable for Response<Bytes> {
    type Source = HttpResponseReplay;

    fn fork(self, replay_cap_bytes: usize) -> (Response<Bytes>, HttpResponseReplay) {
        let status = self.status;
        let metadata = self.metadata.clone();
        let (tee, primary_body) = Replay::wrap_bytes(self.into_chunk_stream(), replay_cap_bytes);
        let first = build_response(status, &metadata, primary_body);
        (
            first,
            HttpResponseReplay {
                status,
                metadata,
                tee,
            },
        )
    }

    fn replay(source: &HttpResponseReplay) -> Result<Response<Bytes>, ProximaError> {
        let body = source.tee.replay()?;
        Ok(build_response(source.status, &source.metadata, body))
    }
}

impl Idempotent for Response<Bytes> {
    fn is_idempotent(&self) -> bool {
        RESPONSE_REDELIVERY_IDEMPOTENT_DEFAULT
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeliveryOutcome {
    Ack,
    Nak,
    WriteError,
}

impl Retryable for DeliveryOutcome {
    fn retry_status(&self) -> Option<u16> {
        None
    }

    fn is_success(&self) -> bool {
        matches!(self, DeliveryOutcome::Ack)
    }
}

#[cfg(feature = "std")]
fn build_response(
    status: u16,
    metadata: &crate::pipe::header_list::HeaderList,
    body: crate::pipe::body::ChunkStream,
) -> Response<Bytes> {
    Response {
        status,
        metadata: metadata.clone(),
        payload: bytes::Bytes::new(),
        stream: Some(crate::pipe::body::ResponseStream::from_chunk_stream(body)),
        #[cfg(feature = "std")]
        upgrade: None,
    }
}

#[cfg(feature = "std")]
/// Serialisable retry-budget config. Mirrors [`RetryBudget`]'s constructor;
/// presence of a `budget` block (even empty) enables the budget at defaults,
/// matching the historical hand-parser.
#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy, Builder, Serialize, Deserialize)]
#[builder(derive(Clone, Debug))]
pub struct BudgetConfig {
    #[serde(default = "default_budget_capacity")]
    #[builder(default = default_budget_capacity())]
    pub capacity: u64,
    #[serde(default = "default_budget_refill")]
    #[builder(default = default_budget_refill())]
    pub refill_per_success: u64,
}

#[cfg(feature = "std")]
fn default_budget_capacity() -> u64 {
    DEFAULT_BUDGET_CAPACITY
}

#[cfg(feature = "std")]
fn default_budget_refill() -> u64 {
    DEFAULT_BUDGET_REFILL_PER_SUCCESS
}

/// Typed config surface for the `retry` middleware.
#[cfg(feature = "std")]
#[derive(Debug, Clone, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_RETRY")]
#[builder(derive(Clone, Debug))]
pub struct RetryConfig {
    /// Total attempts (>= 1). Defaults to 3.
    #[setting(default = 3)]
    #[serde(default = "default_max_attempts")]
    #[builder(default = default_max_attempts())]
    pub max_attempts: u32,

    /// Base backoff delay in ms. Defaults to 50.
    #[setting(default = 50)]
    #[serde(default = "default_base_delay_ms")]
    #[builder(default = default_base_delay_ms())]
    pub base_delay_ms: u64,

    /// Backoff ceiling in ms. Defaults to 2000.
    #[setting(default = 2000)]
    #[serde(default = "default_max_delay_ms")]
    #[builder(default = default_max_delay_ms())]
    pub max_delay_ms: u64,

    /// Replay buffer cap in bytes for request-body replay across attempts.
    #[setting(default = 4194304)]
    #[serde(default = "default_replay_cap_bytes")]
    #[builder(default = default_replay_cap_bytes())]
    pub replay_cap_bytes: usize,

    /// Status codes that trigger a retry.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub retry_on_status: Vec<u16>,

    /// Retry on any inner error.
    #[setting(default)]
    #[serde(default)]
    #[builder(default)]
    pub retry_on_error: bool,

    /// Only retry idempotent requests.
    #[setting(default)]
    #[serde(default)]
    #[builder(default)]
    pub idempotent_only: bool,

    /// Optional retry budget. Presence enables it (defaults when fields omitted).
    #[setting(skip)]
    #[serde(default)]
    pub budget: Option<BudgetConfig>,
}

#[cfg(feature = "std")]
fn default_max_attempts() -> u32 {
    DEFAULT_MAX_ATTEMPTS
}

#[cfg(feature = "std")]
fn default_base_delay_ms() -> u64 {
    DEFAULT_BASE_DELAY_MS
}

#[cfg(feature = "std")]
fn default_max_delay_ms() -> u64 {
    DEFAULT_MAX_DELAY_MS
}

#[cfg(feature = "std")]
fn default_replay_cap_bytes() -> usize {
    DEFAULT_REPLAY_CAP_BYTES
}

#[cfg(feature = "std")]
impl Validate for RetryConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.max_attempts == 0 {
            errors.push(ValidationMessage::new("max_attempts", "must be >= 1"));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

#[cfg(feature = "std")]
impl RetryConfig {
    /// Materialise the retry middleware around `inner`, on the production
    /// [`TimeClock`].
    pub fn from_config(
        self,
        inner: PipeHandle,
    ) -> Result<Retry<PipeHandle, TimeClock>, ProximaError> {
        self.validate()
            .map_err(|err| ProximaError::Config(format!("{err}")))?;
        let mut retry = Retry::new(inner);
        retry.max_attempts = self.max_attempts.max(1);
        retry.base_delay = Duration::from_millis(self.base_delay_ms);
        retry.max_delay = Duration::from_millis(self.max_delay_ms);
        retry.replay_cap_bytes = self.replay_cap_bytes;
        for status in self.retry_on_status {
            retry.rules.retry_on_status.insert(status);
        }
        retry.rules.retry_on_error = self.retry_on_error;
        retry.rules.idempotent_only = self.idempotent_only;
        if let Some(budget) = self.budget {
            retry.budget = Some(Arc::new(RetryBudget::new(
                budget.capacity,
                budget.refill_per_success,
            )));
        }
        Ok(retry)
    }
}

#[cfg(feature = "std")]
impl Retry<PipeHandle, TimeClock> {
    pub fn from_spec(inner: PipeHandle, value: &Value) -> Result<Self, ProximaError> {
        let config: RetryConfig = serde_json::from_value(value.clone())
            .map_err(|err| ProximaError::Config(format!("retry config: {err}")))?;
        config.from_config(inner)
    }
}

#[cfg(feature = "std")]
pub struct RetryFactory;

#[cfg(feature = "std")]
impl PipeFactory for RetryFactory {
    fn name(&self) -> &str {
        "retry"
    }

    fn build(
        &self,
        spec: &Value,
        inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        Box::pin(async move {
            let inner =
                inner.ok_or_else(|| ProximaError::Config("retry requires an inner pipe".into()))?;
            let retry = Retry::from_spec(inner, &spec)?;
            Ok(into_handle(retry))
        })
    }
}

#[cfg(feature = "std")]
fn source_stream(
    payload: bytes::Bytes,
    stream: Option<crate::pipe::body::RequestStream>,
) -> crate::pipe::body::ChunkStream {
    match stream {
        Some(stream) => stream.into_chunk_stream(),
        None => Box::pin(futures::stream::once(async move { Ok(payload) })),
    }
}

#[cfg(feature = "std")]
fn build_request(
    method: &crate::pipe::method::Method,
    path: &bytes::Bytes,
    query: &crate::pipe::header_list::HeaderList,
    metadata: &crate::pipe::header_list::HeaderList,
    context: &crate::pipe::request::RequestContext,
    body: crate::pipe::body::ChunkStream,
) -> Request<Bytes> {
    Request {
        method: method.clone(),
        path: bytes::Bytes::clone(path),
        query: query.clone(),
        metadata: metadata.clone(),
        payload: bytes::Bytes::new(),
        stream: Some(crate::pipe::body::RequestStream::from_chunk_stream(body)),
        context: context.clone(),
    }
}

fn is_idempotent_method(method: &crate::pipe::method::Method) -> bool {
    use crate::pipe::method::Method;
    matches!(
        method,
        Method::Get | Method::Head | Method::Put | Method::Delete | Method::Options
    )
}

// `#[proxima::test]` pulls in the `proxima` dev-dependency, which the
// loom build keeps out of the graph (see
// `[target.'cfg(not(loom))'.dev-dependencies]` in Cargo.toml); these
// tests are unrelated to the Notify/watch loom protocol.
#[cfg(all(test, not(loom)))]
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
    use std::collections::HashMap;
    use std::future::Future;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;
    use crate::pipe::handler::into_handle;
    use crate::pipe::telemetry_surface::{Telemetry, TelemetryHandle};
    use bytes::Bytes;

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

    struct FailUntil {
        threshold: u32,
        observed: AtomicU32,
        failure_status: u16,
    }

    impl SendPipe for FailUntil {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            let attempt = self.observed.fetch_add(1, Ordering::SeqCst) + 1;
            let threshold = self.threshold;
            let status = self.failure_status;
            async move {
                if attempt < threshold {
                    Ok(Response::new(status))
                } else {
                    Ok(Response::ok("ok"))
                }
            }
        }
    }


    fn stub_inner() -> PipeHandle {
        into_handle(FailUntil {
            threshold: 1,
            observed: AtomicU32::new(0),
            failure_status: 503,
        })
    }

    // principle-4 parity: the fluent builder and the config value must lower to
    // identical Retry state (attempts, delays, rules, budget, replay cap).
    #[test]
    fn parity_fluent_builder_and_config_value_match() {
        let from_value: RetryConfig = serde_json::from_value(serde_json::json!({
            "max_attempts": 5,
            "base_delay_ms": 25,
            "max_delay_ms": 1000,
            "replay_cap_bytes": 8192,
            "retry_on_status": [502, 503],
            "retry_on_error": true,
            "idempotent_only": true,
            "budget": {"capacity": 50, "refill_per_success": 5},
        }))
        .expect("from_value");
        let from_value = from_value
            .from_config(stub_inner())
            .expect("from_config value");

        let from_builder = RetryConfig::builder()
            .max_attempts(5)
            .base_delay_ms(25)
            .max_delay_ms(1000)
            .replay_cap_bytes(8192)
            .retry_on_status(alloc::vec![502, 503])
            .retry_on_error(true)
            .idempotent_only(true)
            .budget(
                BudgetConfig::builder()
                    .capacity(50)
                    .refill_per_success(5)
                    .build(),
            )
            .build()
            .from_config(stub_inner())
            .expect("from_config builder");

        assert_eq!(from_value.max_attempts, from_builder.max_attempts);
        assert_eq!(from_value.base_delay, from_builder.base_delay);
        assert_eq!(from_value.max_delay, from_builder.max_delay);
        assert_eq!(from_value.replay_cap_bytes, from_builder.replay_cap_bytes);
        assert_eq!(
            from_value.rules.retry_on_status,
            from_builder.rules.retry_on_status
        );
        assert_eq!(
            from_value.rules.retry_on_error,
            from_builder.rules.retry_on_error
        );
        assert_eq!(
            from_value.rules.idempotent_only,
            from_builder.rules.idempotent_only
        );
        assert_eq!(from_value.budget.is_some(), from_builder.budget.is_some());
    }

    fn build_request(method: &str) -> Request<Bytes> {
        Request::builder()
            .method(method)
            .path("/v1/x")
            .body("payload")
            .build()
            .expect("builder")
    }

    fn build_request_with_metrics(method: &str) -> (Request<Bytes>, Arc<Metrics>) {
        let metrics: Arc<Metrics> = Arc::new(Metrics::default());
        let telemetry: TelemetryHandle = metrics.clone();
        let request = Request::builder()
            .method(method)
            .path("/v1/x")
            .body("payload")
            .telemetry(telemetry)
            .build()
            .expect("builder");
        (request, metrics)
    }

    #[proxima::test]
    async fn retry_succeeds_after_retried_503() {
        let svc = FailUntil {
            threshold: 3,
            observed: AtomicU32::new(0),
            failure_status: 503,
        };
        let stack = Retry::new(into_handle(svc))
            .with_max_attempts(5)
            .with_base_delay(Duration::ZERO)
            .with_max_delay(Duration::ZERO);
        let (request, metrics) = build_request_with_metrics("GET");
        let response = SendPipe::call(&stack, request).await.expect("call");
        assert_eq!(response.status, 200);
        let success = metrics
            .counter(METRIC_SUCCEEDED, &Labels::empty())
            .expect("succeeded counter");
        assert_eq!(success, 1);
    }

    #[proxima::test]
    async fn retry_replays_request_body_on_each_attempt() {
        struct EchoBody {
            attempts: Arc<AtomicU32>,
        }
        impl SendPipe for EchoBody {
            type In = Request<Bytes>;
            type Out = Response<Bytes>;
            type Err = ProximaError;

            fn call(
                &self,
                request: Request<Bytes>,
            ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
                let attempts = self.attempts.clone();
                async move {
                    let attempt = attempts.fetch_add(1, Ordering::SeqCst);
                    let (_, body) = request.body_bytes().await?;
                    if attempt < 1 {
                        Ok(Response::new(503).with_body(body))
                    } else {
                        Ok(Response::ok(body))
                    }
                }
            }
        }

        let attempts = Arc::new(AtomicU32::new(0));
        let svc = EchoBody {
            attempts: attempts.clone(),
        };
        let stack = Retry::new(into_handle(svc))
            .with_max_attempts(3)
            .with_base_delay(Duration::ZERO)
            .with_max_delay(Duration::ZERO);
        let response = SendPipe::call(&stack, build_request("PUT"))
            .await
            .expect("call");
        assert_eq!(response.status, 200);
        let body = response.collect_body().await.expect("collect");
        assert_eq!(&body[..], b"payload");
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[proxima::test]
    async fn retry_skipped_when_method_not_idempotent_and_idempotent_only() {
        let svc = FailUntil {
            threshold: 3,
            observed: AtomicU32::new(0),
            failure_status: 503,
        };
        let stack = Retry::new(into_handle(svc))
            .with_max_attempts(5)
            .with_base_delay(Duration::ZERO)
            .with_max_delay(Duration::ZERO)
            .with_predicate(RetryPredicate::OnIdempotentOnly);
        let response = SendPipe::call(&stack, build_request("POST"))
            .await
            .expect("call");
        assert_eq!(response.status, 503, "non-idempotent must not retry");
    }

    #[proxima::test]
    async fn retry_budget_exhausted_returns_last_response() {
        let svc = FailUntil {
            threshold: 100,
            observed: AtomicU32::new(0),
            failure_status: 503,
        };
        let budget = Arc::new(RetryBudget::new(150, 0));
        let stack = Retry::new(into_handle(svc))
            .with_max_attempts(10)
            .with_base_delay(Duration::ZERO)
            .with_max_delay(Duration::ZERO)
            .with_budget(budget.clone());
        let (request, metrics) = build_request_with_metrics("GET");
        let response = SendPipe::call(&stack, request).await.expect("call");
        assert_eq!(response.status, 503);
        let exhausted = metrics
            .counter(
                METRIC_BUDGET_EXHAUSTED,
                &Labels::from_pairs(&[("reason", "budget")]),
            )
            .expect("budget exhausted counter");
        assert!(exhausted >= 1);
    }

    #[proxima::test]
    async fn replay_cap_exceeded_surfaces_typed_error() {
        struct ConsumingFailUntil {
            attempts: AtomicU32,
        }
        impl SendPipe for ConsumingFailUntil {
            type In = Request<Bytes>;
            type Out = Response<Bytes>;
            type Err = ProximaError;

            fn call(
                &self,
                request: Request<Bytes>,
            ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
                let attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
                async move {
                    let (_, _) = request.body_bytes().await?;
                    if attempt < 5 {
                        Ok(Response::new(503))
                    } else {
                        Ok(Response::ok("ok"))
                    }
                }
            }
        }

        let svc = ConsumingFailUntil {
            attempts: AtomicU32::new(0),
        };
        let stack = Retry::new(into_handle(svc))
            .with_max_attempts(3)
            .with_base_delay(Duration::ZERO)
            .with_max_delay(Duration::ZERO)
            .with_replay_cap_bytes(2);
        let outcome = SendPipe::call(&stack, build_request("GET")).await;
        assert!(matches!(outcome, Err(ProximaError::Body(_))));
    }

    #[derive(Clone, PartialEq, Debug)]
    struct CountEvent(u64);

    impl Replayable for CountEvent {
        type Source = CountEvent;
        fn fork(self, _replay_cap_bytes: usize) -> (CountEvent, CountEvent) {
            (self.clone(), self)
        }
        fn replay(source: &CountEvent) -> Result<CountEvent, ProximaError> {
            Ok(source.clone())
        }
    }
    impl Idempotent for CountEvent {
        fn is_idempotent(&self) -> bool {
            true
        }
    }
    impl Labeled for CountEvent {
        fn telemetry(&self) -> TelemetryHandle {
            Arc::new(Metrics::default())
        }
        fn labels(&self) -> Labels {
            Labels::empty()
        }
    }
    impl Retryable for CountEvent {
        fn retry_status(&self) -> Option<u16> {
            None
        }
        fn is_success(&self) -> bool {
            true
        }
    }

    #[derive(Clone)]
    struct FlakyEventSink {
        fail_until: u32,
        observed: Arc<AtomicU32>,
    }

    impl SendPipe for FlakyEventSink {
        type In = CountEvent;
        type Out = CountEvent;
        type Err = ProximaError;

        fn call(
            &self,
            input: CountEvent,
        ) -> impl Future<Output = Result<CountEvent, ProximaError>> + Send {
            let attempt = self.observed.fetch_add(1, Ordering::SeqCst) + 1;
            let fail_until = self.fail_until;
            async move {
                if attempt < fail_until {
                    Err(ProximaError::Config("sink not ready".into()))
                } else {
                    Ok(CountEvent(input.0 * 2))
                }
            }
        }
    }

    #[proxima::test]
    async fn retry_is_generic_over_a_non_http_event_type() {
        let sink = FlakyEventSink {
            fail_until: 3,
            observed: Arc::new(AtomicU32::new(0)),
        };
        let stack = Retry::new(sink)
            .with_max_attempts(5)
            .with_base_delay(Duration::ZERO)
            .with_max_delay(Duration::ZERO);
        let out = SendPipe::call(&stack, CountEvent(21))
            .await
            .expect("event call");
        assert_eq!(
            out,
            CountEvent(42),
            "retried an event pipe to success with zero HTTP involvement, no policy type"
        );
    }

    #[derive(Clone)]
    struct FlakyClientDelivery {
        nak_until: u32,
        observed: Arc<AtomicU32>,
        last_body: Arc<Mutex<Option<Vec<u8>>>>,
    }

    impl SendPipe for FlakyClientDelivery {
        type In = Response<Bytes>;
        type Out = DeliveryOutcome;
        type Err = ProximaError;

        fn call(
            &self,
            response: Response<Bytes>,
        ) -> impl Future<Output = Result<DeliveryOutcome, ProximaError>> + Send {
            let attempt = self.observed.fetch_add(1, Ordering::SeqCst) + 1;
            let nak_until = self.nak_until;
            let last_body = self.last_body.clone();
            async move {
                let body = response.collect_body().await?;
                if attempt < nak_until {
                    Err(ProximaError::Config("client nak".into()))
                } else {
                    *last_body.lock().unwrap() = Some(body.to_vec());
                    Ok(DeliveryOutcome::Ack)
                }
            }
        }
    }

    #[proxima::test]
    async fn retry_redelivers_a_response_to_a_flaky_client() {
        let observed = Arc::new(AtomicU32::new(0));
        let last_body = Arc::new(Mutex::new(None));
        let sink = FlakyClientDelivery {
            nak_until: 3,
            observed: observed.clone(),
            last_body: last_body.clone(),
        };
        let stack = Retry::new(sink)
            .with_max_attempts(5)
            .with_base_delay(Duration::ZERO)
            .with_max_delay(Duration::ZERO);
        let response = Response::ok("deliver-me");
        let outcome = SendPipe::call(&stack, response)
            .await
            .expect("delivery call");
        assert_eq!(
            outcome,
            DeliveryOutcome::Ack,
            "the SAME Retry<Inner> drove the delivery to a successful ack"
        );
        assert_eq!(observed.load(Ordering::SeqCst), 3, "two naks then an ack");
        assert_eq!(
            last_body.lock().unwrap().as_deref(),
            Some(&b"deliver-me"[..]),
            "body replayed intact on each re-delivery"
        );
    }

    #[proxima::test]
    async fn response_redelivery_is_not_idempotent_by_default_and_idempotent_only_guards_it() {
        let observed = Arc::new(AtomicU32::new(0));
        let sink = FlakyClientDelivery {
            nak_until: 3,
            observed: observed.clone(),
            last_body: Arc::new(Mutex::new(None)),
        };
        let stack = Retry::new(sink)
            .with_max_attempts(5)
            .with_base_delay(Duration::ZERO)
            .with_max_delay(Duration::ZERO)
            .with_predicate(RetryPredicate::OnIdempotentOnly);
        let outcome = SendPipe::call(&stack, Response::ok("deliver-me")).await;
        assert!(
            matches!(outcome, Err(ProximaError::Config(_))),
            "idempotent_only must suppress unsafe re-delivery"
        );
        assert_eq!(
            observed.load(Ordering::SeqCst),
            1,
            "exactly one delivery attempt — no retry"
        );
    }

    #[test]
    fn delivery_outcome_classifies_ack_as_success_only() {
        assert!(DeliveryOutcome::Ack.is_success());
        assert!(!DeliveryOutcome::Nak.is_success());
        assert!(!DeliveryOutcome::WriteError.is_success());
        assert_eq!(
            DeliveryOutcome::Ack.retry_status(),
            None,
            "delivery is statusless — error/nak governs, not http status"
        );
    }

    #[proxima::test]
    async fn response_replay_forks_the_body_for_redelivery() {
        let response = Response::ok("body-bytes");
        let (first, source) = Replayable::fork(response, DEFAULT_REPLAY_CAP_BYTES);
        let first_body = first.collect_body().await.expect("first body");
        assert_eq!(&first_body[..], b"body-bytes");
        let replayed = Response::replay(&source).expect("replay");
        let replayed_body = replayed.collect_body().await.expect("replayed body");
        assert_eq!(
            &replayed_body[..],
            b"body-bytes",
            "the tee replays the response body verbatim"
        );
    }

    // ── Clock-injectable determinism ────────────────────────────────────────
    //
    // `Retry::with_clock` takes any `Clock` impl, so the between-attempt delay
    // can be proven deterministically — `delay` resolves instantly and records
    // what it was asked to wait, never a real sleep.

    #[derive(Clone, Default)]
    struct FakeClock {
        now_nanos: Arc<AtomicU64>,
        delays: Arc<Mutex<Vec<Duration>>>,
    }

    impl FakeClock {
        fn delays(&self) -> Vec<Duration> {
            self.delays.lock().unwrap().clone()
        }
    }

    impl Clock for FakeClock {
        type Delay = std::future::Ready<()>;

        fn now_nanos(&self) -> u64 {
            self.now_nanos.load(Ordering::Relaxed)
        }

        fn delay(&self, duration: Duration) -> Self::Delay {
            self.delays.lock().unwrap().push(duration);
            std::future::ready(())
        }
    }

    #[proxima::test]
    async fn fake_clock_drives_backoff_between_retries_no_real_sleep() {
        let clock = FakeClock::default();
        let svc = FailUntil {
            threshold: 3,
            observed: AtomicU32::new(0),
            failure_status: 503,
        };
        // a base/max delay of seconds would make a real-sleep implementation
        // take real seconds to run; this test still completes instantly
        // because `FakeClock::delay` never actually waits.
        let stack = Retry::with_clock(into_handle(svc), clock.clone())
            .with_max_attempts(5)
            .with_base_delay(Duration::from_secs(5))
            .with_max_delay(Duration::from_secs(30));
        let (request, metrics) = build_request_with_metrics("GET");
        let response = SendPipe::call(&stack, request).await.expect("call");

        assert_eq!(response.status, 200, "retried past two 503s to success");
        let success = metrics
            .counter(METRIC_SUCCEEDED, &Labels::empty())
            .expect("succeeded counter");
        assert_eq!(success, 1);

        let delays = clock.delays();
        assert_eq!(
            delays.len(),
            2,
            "two retryable 503s before the third attempt succeeds"
        );
        for delay in delays {
            assert!(
                delay <= Duration::from_secs(30),
                "delay {delay:?} exceeds the configured max_delay"
            );
        }
    }

    #[proxima::test]
    async fn fake_clock_non_retryable_error_stops_after_one_attempt() {
        let clock = FakeClock::default();
        let svc = FailUntil {
            threshold: 100,
            observed: AtomicU32::new(0),
            failure_status: 422,
        };
        let stack = Retry::with_clock(into_handle(svc), clock.clone())
            .with_max_attempts(5)
            .with_base_delay(Duration::from_secs(5))
            .with_max_delay(Duration::from_secs(30));
        let response = SendPipe::call(&stack, build_request("GET"))
            .await
            .expect("call");

        assert_eq!(
            response.status, 422,
            "422 is outside the default retry_on_status set"
        );
        assert!(
            clock.delays().is_empty(),
            "non-retryable outcome must not schedule any delay"
        );
    }
}
