use bytes::Bytes;
use std::borrow::Cow;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use crate::pipe::SendPipe;
use crate::pipe::capabilities::Clock;
pub use crate::pipe::capabilities::{ExceededAction, KeyOf};
use crate::pipe::primitives::Pipe;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::pipe::bucket_table::BucketTable;
use crate::pipe::clock::TimeClock;
use crate::pipe::labeled::Labeled;
use crate::pipe::handler::{PipeHandle, ThreadLocalPipeHandle, into_handle};
use crate::pipe::pipe_factory::PipeFactory;
use crate::pipe::request::{Request, Response};
use crate::pipe::telemetry_surface::Labels;
use proxima_core::ProximaError;

const MICROS_PER_TOKEN: u64 = 1_000_000;
const NANOS_PER_SEC: u128 = 1_000_000_000;

const METRIC_REJECTED: &str = "proxima.rate_limit.rejected_total";
const METRIC_ADMITTED: &str = "proxima.rate_limit.admitted_total";

const MISSING_HEADER_KEY: &str = "__missing__";

// ── HTTP key extractor ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum KeyExtractor {
    ConstantKey(String),
    Header(String),
    PathAndMethod,
}

impl KeyOf<KeyExtractor> for Request<Bytes> {
    type Rejection = Response<Bytes>;

    fn rate_key<'a>(&'a self, extractor: &'a KeyExtractor) -> Cow<'a, [u8]> {
        match extractor {
            KeyExtractor::ConstantKey(value) => Cow::Borrowed(value.as_bytes()),
            KeyExtractor::Header(name) => self
                .metadata
                .iter()
                .find(|(header_name, _)| header_name.as_ref().eq_ignore_ascii_case(name.as_bytes()))
                .map_or(
                    Cow::Borrowed(MISSING_HEADER_KEY.as_bytes()),
                    |(_, value)| Cow::Borrowed(value.as_ref()),
                ),
            KeyExtractor::PathAndMethod => {
                let mut buffer =
                    Vec::with_capacity(self.method.as_bytes().len() + 1 + self.path.len());
                buffer.extend_from_slice(self.method.as_bytes());
                buffer.push(b' ');
                buffer.extend_from_slice(&self.path);
                Cow::Owned(buffer)
            }
        }
    }

    fn build_rejection(action: &ExceededAction) -> Response<Bytes> {
        rejected_response(action)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TokenBucketConfig {
    pub capacity: u64,
    pub refill_per_sec: u64,
}

/// Caps on the per-key bucket map.
#[derive(Debug, Clone, Copy)]
pub struct RateLimitCaps {
    pub max_keys: usize,
    pub idle_ttl: Duration,
    pub sweep_period: Duration,
}

impl Default for RateLimitCaps {
    fn default() -> Self {
        Self {
            max_keys: 100_000,
            idle_ttl: Duration::from_secs(600),
            sweep_period: Duration::from_secs(30),
        }
    }
}

/// Token-bucket rate limiter. Generic over the inner handle, the extractor
/// type, AND the clock — mirrors `resilience::Retry<Inner, Clk>`'s injected-clock
/// seam so refill timing is deterministically testable. `Clk` defaults to
/// [`TimeClock`], the production monotonic clock, so every existing caller
/// (`RateLimit::new`, `RateLimitConfig`, `RateLimitFactory`) is unaffected.
///
/// Sans-IO: the limiter owns no runtime and spawns no task. Map growth is bounded
/// by `caps.max_keys` (LRU eviction on insert, inline on the call path). Proactive
/// reclamation of idle buckets is the host's job — call [`sweep_idle`] on whatever
/// cadence fits (read `caps.sweep_period` for the recommended interval) from a
/// `proxima_core::time::interval` loop on your own runtime. That keeps the primitive
/// runtime-agnostic (prime, tokio, embassy, bare-metal — it neither knows nor
/// cares which drives the host's tick).
///
/// [`sweep_idle`]: RateLimit::sweep_idle
pub struct RateLimit<Inner = PipeHandle, Extractor = KeyExtractor, Clk = TimeClock> {
    inner: Inner,
    config: TokenBucketConfig,
    extractor: Extractor,
    on_exceeded: ExceededAction,
    buckets: BucketTable<AtomicBucket>,
    caps: RateLimitCaps,
    clock: Clk,
}

// Hardcoded to `TimeClock` (not `Clk: Default`) so `RateLimit::new(...)` and
// `RateLimit::with_caps(...)` resolve without turbofish, the same way
// `HashMap::new()` pins its defaulted hasher param — a bare `Clk: Default`
// bound here would leave the type parameter ambiguous at every existing call
// site with no annotation to pin it.
impl<Inner, Extractor> RateLimit<Inner, Extractor, TimeClock> {
    #[must_use]
    pub fn new(inner: Inner, config: TokenBucketConfig, extractor: Extractor) -> Self {
        Self::with_caps(inner, config, extractor, RateLimitCaps::default())
    }

    #[must_use]
    pub fn with_caps(
        inner: Inner,
        config: TokenBucketConfig,
        extractor: Extractor,
        caps: RateLimitCaps,
    ) -> Self {
        Self::with_clock(inner, config, extractor, caps, TimeClock)
    }
}

impl<Inner, Extractor, Clk> RateLimit<Inner, Extractor, Clk> {
    /// Materialise with an explicit clock — the seam a deterministic test or
    /// example injects a fake clock through; production code goes via `new`
    /// / `with_caps`, which default `Clk` to [`TimeClock`].
    #[must_use]
    pub fn with_clock(
        inner: Inner,
        config: TokenBucketConfig,
        extractor: Extractor,
        caps: RateLimitCaps,
        clock: Clk,
    ) -> Self {
        let buckets: BucketTable<AtomicBucket> = BucketTable::with_max_keys(caps.max_keys);
        Self {
            inner,
            config,
            extractor,
            on_exceeded: ExceededAction::Reject429 {
                retry_after_ms: 1000,
            },
            buckets,
            caps,
            clock,
        }
    }

    // only called from `mod tests` below, which is itself `not(loom)`-gated.
    #[cfg(all(test, not(loom)))]
    pub(crate) fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    #[must_use]
    pub fn with_action(mut self, action: ExceededAction) -> Self {
        self.on_exceeded = action;
        self
    }
}

impl<Inner, Extractor, Clk: Clock> RateLimit<Inner, Extractor, Clk> {
    /// Evict buckets idle longer than `caps.idle_ttl`. Sans-IO: the host calls
    /// this on its own cadence (the limiter spawns nothing). Bounded growth does
    /// not depend on it — `caps.max_keys` already caps the map via LRU eviction
    /// on insert; this just reclaims idle memory sooner.
    pub fn sweep_idle(&self) {
        sweep_idle_at(
            &self.buckets,
            self.clock.now_nanos() / 1_000,
            self.caps.idle_ttl,
        );
    }

    pub(crate) fn bucket_for(&self, key: &[u8]) -> Arc<AtomicBucket> {
        // proactively cap before a possible insert; the table also self-defends
        // on a full wrap, but evicting here keeps `count <= max_keys`. when the
        // key already exists get_or_insert returns it unchanged, so an eviction
        // here only ever sheds a colder distinct key.
        if self.buckets.len() >= self.caps.max_keys {
            evict_lru(&self.buckets);
        }
        let config = self.config;
        let now_nanos = self.clock.now_nanos();
        let bucket = self
            .buckets
            .get_or_insert(key, || AtomicBucket::new(config, now_nanos));
        bucket.touch(now_nanos);
        bucket
    }
}

/// Serialisable key-extraction strategy — the config mirror of [`KeyExtractor`].
/// The `path_and_method` default matches the historical hand-parser.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "key", rename_all = "snake_case")]
pub enum KeyConfig {
    /// One shared bucket keyed by a constant string.
    Constant {
        #[serde(default = "default_constant_value", rename = "constant_value")]
        value: String,
    },
    /// Per-value bucket keyed by a request header.
    Header { header_name: String },
    /// Per-(method, path) bucket — the default.
    #[default]
    PathAndMethod,
}

fn default_constant_value() -> String {
    "__global__".to_string()
}

impl From<KeyConfig> for KeyExtractor {
    fn from(config: KeyConfig) -> Self {
        match config {
            KeyConfig::Constant { value } => KeyExtractor::ConstantKey(value),
            KeyConfig::Header { header_name } => KeyExtractor::Header(header_name),
            KeyConfig::PathAndMethod => KeyExtractor::PathAndMethod,
        }
    }
}

/// Serialisable caps on the per-key bucket map (durations in ms). Mirrors
/// [`RateLimitCaps`] with the same defaults.
#[derive(Debug, Clone, Copy, Builder, Serialize, Deserialize)]
#[builder(derive(Clone, Debug))]
pub struct CapsConfig {
    #[serde(default = "default_max_keys")]
    #[builder(default = default_max_keys())]
    pub max_keys: usize,
    #[serde(default = "default_idle_ttl_ms")]
    #[builder(default = default_idle_ttl_ms())]
    pub idle_ttl_ms: u64,
    #[serde(default = "default_sweep_ms")]
    #[builder(default = default_sweep_ms())]
    pub sweep_ms: u64,
}

fn default_max_keys() -> usize {
    100_000
}

fn default_idle_ttl_ms() -> u64 {
    600_000
}

fn default_sweep_ms() -> u64 {
    30_000
}

impl Default for CapsConfig {
    fn default() -> Self {
        Self {
            max_keys: default_max_keys(),
            idle_ttl_ms: default_idle_ttl_ms(),
            sweep_ms: default_sweep_ms(),
        }
    }
}

impl From<CapsConfig> for RateLimitCaps {
    fn from(config: CapsConfig) -> Self {
        Self {
            max_keys: config.max_keys,
            idle_ttl: Duration::from_millis(config.idle_ttl_ms),
            sweep_period: Duration::from_millis(config.sweep_ms),
        }
    }
}

/// Typed config surface for the `rate_limit` middleware (token bucket).
#[derive(Debug, Clone, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_RATE_LIMIT")]
#[builder(derive(Clone, Debug))]
pub struct RateLimitConfig {
    /// Token bucket capacity (burst). Required.
    #[setting(default)]
    #[serde(default)]
    #[builder(default)]
    pub capacity: u64,

    /// Steady-state refill rate in tokens per second. Required.
    #[setting(default)]
    #[serde(default)]
    #[builder(default)]
    pub refill_per_sec: u64,

    /// Key-extraction strategy. Flattened so `key = "header"` + `header_name`
    /// sit at the top level, matching the historical spec.
    #[setting(skip)]
    #[serde(default, flatten)]
    #[builder(default)]
    pub key: KeyConfig,

    /// Per-key bucket-map caps.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub caps: CapsConfig,

    /// `Retry-After` (ms) advertised on a 429. `None` keeps the limiter default.
    #[setting(default)]
    #[serde(default)]
    pub retry_after_ms: Option<u64>,
}

impl Validate for RateLimitConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.capacity == 0 {
            errors.push(ValidationMessage::new(
                "capacity",
                "rate_limit.capacity required",
            ));
        }
        if self.refill_per_sec == 0 {
            errors.push(ValidationMessage::new(
                "refill_per_sec",
                "rate_limit.refill_per_sec required",
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl RateLimitConfig {
    /// Materialise the limiter around `inner`, on the production [`TimeClock`].
    pub fn from_config(
        self,
        inner: PipeHandle,
    ) -> Result<RateLimit<PipeHandle, KeyExtractor, TimeClock>, ProximaError> {
        self.validate()
            .map_err(|err| ProximaError::Config(format!("{err}")))?;
        let mut limiter = RateLimit::with_caps(
            inner,
            TokenBucketConfig {
                capacity: self.capacity,
                refill_per_sec: self.refill_per_sec,
            },
            self.key.into(),
            self.caps.into(),
        );
        if let Some(retry_after) = self.retry_after_ms {
            limiter.on_exceeded = ExceededAction::Reject429 {
                retry_after_ms: retry_after,
            };
        }
        Ok(limiter)
    }
}

impl RateLimit<PipeHandle, KeyExtractor, TimeClock> {
    pub fn from_spec(inner: PipeHandle, value: &Value) -> Result<Self, ProximaError> {
        let config: RateLimitConfig = serde_json::from_value(value.clone())
            .map_err(|err| ProximaError::Config(format!("rate_limit config: {err}")))?;
        config.from_config(inner)
    }
}

fn evict_lru(buckets: &BucketTable<AtomicBucket>) {
    buckets.evict_one_lru(|bucket| bucket.last_access_micros.load(Ordering::Relaxed));
}

fn sweep_idle_at(buckets: &BucketTable<AtomicBucket>, now_micros: u64, idle_ttl: Duration) {
    let ttl_micros = idle_ttl.as_micros() as u64;
    buckets.sweep_idle(now_micros, ttl_micros, |bucket| {
        bucket.last_access_micros.load(Ordering::Relaxed)
    });
}

async fn run_rate_limit<In, Out, Err, Key, Call, Fut>(
    args: RunArgs<In, Key>,
    call_inner: Call,
) -> Result<Out, Err>
where
    In: KeyOf<Key, Rejection = Out> + Labeled,
    Key: Send + Sync,
    Call: Fn(In) -> Fut,
    Fut: Future<Output = Result<Out, Err>>,
{
    let RunArgs {
        extractor,
        input,
        bucket,
        action,
        now_nanos,
    } = args;
    let telemetry = input.telemetry();
    let base_labels = input.labels();
    let key = input.rate_key(&extractor);
    let admitted = bucket.try_take(MICROS_PER_TOKEN, now_nanos);
    let label_key_str = std::str::from_utf8(key.as_ref()).unwrap_or("");
    let labels = with_extra(&base_labels, "key", label_key_str);
    if admitted {
        telemetry.counter_inc(METRIC_ADMITTED, &labels, 1);
        call_inner(input).await
    } else {
        telemetry.counter_inc(METRIC_REJECTED, &labels, 1);
        Ok(In::build_rejection(&action))
    }
}

struct RunArgs<In, Key> {
    extractor: Key,
    input: In,
    bucket: Arc<AtomicBucket>,
    action: ExceededAction,
    now_nanos: u64,
}

impl<Inner, Extractor, Clk> SendPipe for RateLimit<Inner, Extractor, Clk>
where
    Inner: SendPipe + Clone + Send + Sync + 'static,
    Inner::In: KeyOf<Extractor, Rejection = Inner::Out> + Labeled + Send + 'static,
    Inner::Out: Send + 'static,
    Inner::Err: Send + 'static,
    Extractor: Clone + Send + Sync + 'static,
    Clk: Clock + Send + Sync + 'static,
{
    type In = Inner::In;
    type Out = Inner::Out;
    type Err = Inner::Err;

    fn call(
        &self,
        input: Inner::In,
    ) -> impl Future<Output = Result<Inner::Out, Inner::Err>> + Send {
        let bucket = self.bucket_for(input.rate_key(&self.extractor).as_ref());
        let now_nanos = self.clock.now_nanos();
        let inner = self.inner.clone();
        run_rate_limit(
            RunArgs {
                extractor: self.extractor.clone(),
                input,
                bucket,
                action: self.on_exceeded,
                now_nanos,
            },
            move |admitted| {
                let inner = inner.clone();
                async move { SendPipe::call(&inner, admitted).await }
            },
        )
    }
}

impl<Extractor, Clk> Pipe for RateLimit<ThreadLocalPipeHandle, Extractor, Clk>
where
    Request<Bytes>: KeyOf<Extractor, Rejection = Response<Bytes>>,
    Extractor: Clone + Send + Sync + 'static,
    Clk: Clock + Send + Sync + 'static,
{
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        input: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        let bucket = self.bucket_for(input.rate_key(&self.extractor).as_ref());
        let now_nanos = self.clock.now_nanos();
        let inner = self.inner.clone();
        run_rate_limit(
            RunArgs {
                extractor: self.extractor.clone(),
                input,
                bucket,
                action: self.on_exceeded,
                now_nanos,
            },
            move |admitted| {
                let inner = inner.clone();
                async move { Pipe::call(&inner, admitted).await }
            },
        )
    }
}

fn rejected_response(action: &ExceededAction) -> Response<Bytes> {
    match action {
        ExceededAction::Reject429 { retry_after_ms } => {
            let retry_after_seconds = retry_after_ms.div_ceil(1000).max(1);
            Response::new(429)
                .with_header("retry-after", retry_after_seconds.to_string())
                .with_body("rate limit exceeded")
        }
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

pub struct RateLimitFactory;

impl PipeFactory for RateLimitFactory {
    fn name(&self) -> &str {
        "rate_limit"
    }

    fn build(
        &self,
        spec: &Value,
        inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        Box::pin(async move {
            let inner = inner
                .ok_or_else(|| ProximaError::Config("rate_limit requires an inner pipe".into()))?;
            let limiter = RateLimit::from_spec(inner, &spec)?;
            Ok(into_handle(limiter))
        })
    }
}

pub(crate) struct AtomicBucket {
    tokens_micro: AtomicU64,
    last_refill_ns: AtomicU64,
    pub(crate) last_access_micros: AtomicU64,
    capacity_micro: u64,
    refill_per_sec_micro: u64,
}

impl AtomicBucket {
    /// `now_nanos` comes from the injected [`Clock`] — the bucket itself holds
    /// no clock and reads no real time, so its refill math is deterministic
    /// under a fake clock.
    fn new(config: TokenBucketConfig, now_nanos: u64) -> Self {
        let capacity_micro = config.capacity.saturating_mul(MICROS_PER_TOKEN);
        let refill_per_sec_micro = config.refill_per_sec.saturating_mul(MICROS_PER_TOKEN);
        Self {
            tokens_micro: AtomicU64::new(capacity_micro),
            last_refill_ns: AtomicU64::new(now_nanos),
            last_access_micros: AtomicU64::new(now_nanos / 1_000),
            capacity_micro,
            refill_per_sec_micro,
        }
    }

    fn touch(&self, now_nanos: u64) {
        self.last_access_micros
            .store(now_nanos / 1_000, Ordering::Relaxed);
    }

    // `now_nanos` is read once per call by the caller (the injected Clock),
    // not re-read on each CAS retry below — a CAS spin is contention on this
    // same bucket, not a new logical "check", so one timestamp covers it.
    pub(crate) fn try_take(&self, cost_micro: u64, now_nanos: u64) -> bool {
        loop {
            let last_ns = self.last_refill_ns.load(Ordering::Acquire);
            let current = self.tokens_micro.load(Ordering::Acquire);
            let elapsed_ns = now_nanos.saturating_sub(last_ns);
            let added_micro = (u128::from(elapsed_ns) * u128::from(self.refill_per_sec_micro)
                / NANOS_PER_SEC) as u64;
            let refilled = current.saturating_add(added_micro).min(self.capacity_micro);
            if refilled < cost_micro {
                if added_micro > 0
                    && self
                        .tokens_micro
                        .compare_exchange(current, refilled, Ordering::Release, Ordering::Acquire)
                        .is_ok()
                {
                    self.last_refill_ns.store(now_nanos, Ordering::Release);
                }
                return false;
            }
            let after = refilled - cost_micro;
            if self
                .tokens_micro
                .compare_exchange(current, after, Ordering::Release, Ordering::Acquire)
                .is_ok()
            {
                self.last_refill_ns.store(now_nanos, Ordering::Release);
                return true;
            }
        }
    }
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

    use super::*;
    use crate::pipe::handler::into_handle;
    use crate::pipe::telemetry_surface::{Telemetry, TelemetryHandle};

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


    fn build_request_with_metrics(method: &str, path: &str) -> (Request<Bytes>, Arc<Metrics>) {
        let metrics: Arc<Metrics> = Arc::new(Metrics::default());
        let telemetry: TelemetryHandle = metrics.clone();
        let request = Request::builder()
            .method(method)
            .path(path)
            .telemetry(telemetry)
            .build()
            .expect("builder");
        (request, metrics)
    }

    // principle-4 parity: the fluent builder and the config value must lower to
    // identical RateLimit state (token config, extractor, caps, exceeded action).
    #[test]
    fn parity_fluent_builder_and_config_value_match() {
        let from_value: RateLimitConfig = serde_json::from_value(serde_json::json!({
            "capacity": 10,
            "refill_per_sec": 5,
            "key": "header",
            "header_name": "x-api-key",
            "caps": {"max_keys": 1000, "idle_ttl_ms": 60000, "sweep_ms": 5000},
            "retry_after_ms": 2000,
        }))
        .expect("from_value");
        let from_value = from_value
            .from_config(into_handle(AlwaysOk))
            .expect("from_config value");

        let from_builder = RateLimitConfig::builder()
            .capacity(10)
            .refill_per_sec(5)
            .key(KeyConfig::Header {
                header_name: "x-api-key".to_string(),
            })
            .caps(
                CapsConfig::builder()
                    .max_keys(1000)
                    .idle_ttl_ms(60000)
                    .sweep_ms(5000)
                    .build(),
            )
            .retry_after_ms(2000)
            .build()
            .from_config(into_handle(AlwaysOk))
            .expect("from_config builder");

        assert_eq!(from_value.config.capacity, from_builder.config.capacity);
        assert_eq!(
            from_value.config.refill_per_sec,
            from_builder.config.refill_per_sec
        );
        assert_eq!(from_value.caps.max_keys, from_builder.caps.max_keys);
        assert_eq!(from_value.caps.idle_ttl, from_builder.caps.idle_ttl);
        assert_eq!(from_value.caps.sweep_period, from_builder.caps.sweep_period);
        assert_eq!(
            format!("{:?}", from_value.extractor),
            format!("{:?}", from_builder.extractor)
        );
        assert_eq!(
            format!("{:?}", from_value.on_exceeded),
            format!("{:?}", from_builder.on_exceeded)
        );
    }

    #[proxima::test]
    async fn token_bucket_admits_until_empty_then_rejects() {
        let stack = RateLimit::new(
            into_handle(AlwaysOk),
            TokenBucketConfig {
                capacity: 2,
                refill_per_sec: 0,
            },
            KeyExtractor::ConstantKey("global".into()),
        );

        for _ in 0..2 {
            let (request, _metrics) = build_request_with_metrics("GET", "/");
            let response = SendPipe::call(&stack, request).await.expect("call");
            assert_eq!(response.status, 200);
        }
        let (request, _metrics) = build_request_with_metrics("GET", "/");
        let response = SendPipe::call(&stack, request).await.expect("call");
        assert_eq!(response.status, 429);
        assert_eq!(
            response.metadata.get_str("retry-after"),
            Some("1"),
            "retry-after must be present and round up to seconds",
        );
    }

    #[proxima::test]
    async fn distinct_keys_have_independent_buckets() {
        let stack = RateLimit::new(
            into_handle(AlwaysOk),
            TokenBucketConfig {
                capacity: 1,
                refill_per_sec: 0,
            },
            KeyExtractor::PathAndMethod,
        );

        let (request_a, _metrics_a) = build_request_with_metrics("GET", "/a");
        let (request_b, _metrics_b) = build_request_with_metrics("GET", "/b");
        assert_eq!(
            SendPipe::call(&stack, request_a).await.expect("a").status,
            200
        );
        assert_eq!(
            SendPipe::call(&stack, request_b).await.expect("b").status,
            200
        );

        let (request_a_again, _) = build_request_with_metrics("GET", "/a");
        assert_eq!(
            SendPipe::call(&stack, request_a_again)
                .await
                .expect("a again")
                .status,
            429
        );
    }

    #[proxima::test]
    async fn reject_increments_telemetry_counter() {
        let stack = RateLimit::new(
            into_handle(AlwaysOk),
            TokenBucketConfig {
                capacity: 1,
                refill_per_sec: 0,
            },
            KeyExtractor::ConstantKey("global".into()),
        );

        let metrics: Arc<Metrics> = Arc::new(Metrics::default());
        let telemetry: TelemetryHandle = metrics.clone();
        let one = Request::builder()
            .method("GET")
            .path("/")
            .telemetry(telemetry.clone())
            .build()
            .expect("builder one");
        let two = Request::builder()
            .method("GET")
            .path("/")
            .telemetry(telemetry)
            .build()
            .expect("builder two");

        let _ = SendPipe::call(&stack, one).await.expect("first admitted");
        let _ = SendPipe::call(&stack, two).await.expect("second rejected");

        let labels = Labels::from_pairs(&[("key", "global")]);
        let rejected = metrics.counter(METRIC_REJECTED, &labels);
        assert_eq!(rejected, Some(1));
    }

    #[proxima::test]
    async fn max_keys_cap_evicts_lru_on_insert() {
        let limiter = RateLimit::with_caps(
            into_handle(AlwaysOk),
            TokenBucketConfig {
                capacity: 1,
                refill_per_sec: 0,
            },
            KeyExtractor::Header("x-tenant".into()),
            RateLimitCaps {
                max_keys: 2,
                idle_ttl: Duration::from_secs(600),
                sweep_period: Duration::from_secs(60),
            },
        );

        let mut original_a: Option<Arc<AtomicBucket>> = None;
        for tenant in ["a", "b", "c"] {
            let metrics: Arc<Metrics> = Arc::new(Metrics::default());
            let telemetry: TelemetryHandle = metrics.clone();
            let request = Request::builder()
                .method("GET")
                .path("/")
                .header("x-tenant", tenant)
                .telemetry(telemetry)
                .build()
                .expect("builder");
            let bucket = limiter.bucket_for(tenant.as_bytes());
            if tenant == "a" {
                original_a = Some(bucket);
            }
            std::hint::black_box(request);
        }
        assert_eq!(
            limiter.bucket_count(),
            2,
            "max_keys cap must evict on insert"
        );
        // a was the lru victim before c's insert; reinserting it makes a fresh
        // bucket (distinct Arc), proving the original was evicted.
        let reinserted_a = limiter.bucket_for(b"a");
        assert!(
            !Arc::ptr_eq(&reinserted_a, original_a.as_ref().expect("a inserted")),
            "a was evicted; reinsert yields a fresh bucket"
        );
    }

    #[test]
    fn sweep_idle_at_evicts_only_buckets_past_ttl() {
        // injected clock -> deterministic, no sleep. fresh bucket touched "now",
        // stale bucket last seen long ago; sweep at now with a 1s ttl drops only stale.
        let buckets: BucketTable<AtomicBucket> = BucketTable::with_max_keys(8);
        let config = TokenBucketConfig {
            capacity: 1,
            refill_per_sec: 1,
        };
        let now_micros = 10_000_000;
        let idle_ttl = Duration::from_secs(1);

        let fresh = buckets.get_or_insert(b"fresh", || AtomicBucket::new(config, 0));
        fresh
            .last_access_micros
            .store(now_micros, Ordering::Relaxed);

        let stale = buckets.get_or_insert(b"stale", || AtomicBucket::new(config, 0));
        stale
            .last_access_micros
            .store(now_micros - 5_000_000, Ordering::Relaxed);

        sweep_idle_at(&buckets, now_micros, idle_ttl);

        assert_eq!(
            buckets.len(),
            1,
            "only the stale bucket past ttl is evicted"
        );
        let fresh_again = buckets.get_or_insert(b"fresh", || AtomicBucket::new(config, 0));
        assert!(
            Arc::ptr_eq(&fresh_again, &fresh),
            "fresh bucket survives unchanged"
        );
        let stale_again = buckets.get_or_insert(b"stale", || AtomicBucket::new(config, 0));
        assert!(
            !Arc::ptr_eq(&stale_again, &stale),
            "stale was evicted, fresh on reinsert"
        );
    }

    #[test]
    fn key_of_constant_returns_borrowed_slice() {
        let extractor = KeyExtractor::ConstantKey("tenant-42".into());
        let request = Request::builder()
            .method("GET")
            .path("/")
            .build()
            .expect("builder");
        match request.rate_key(&extractor) {
            Cow::Borrowed(value) => assert_eq!(value, b"tenant-42"),
            Cow::Owned(_) => panic!("ConstantKey must not allocate"),
        }
    }

    #[test]
    fn key_of_header_borrows_from_request_value() {
        let extractor = KeyExtractor::Header("x-api-key".into());
        let request = Request::builder()
            .method("GET")
            .path("/")
            .header("X-API-Key", "abc123")
            .build()
            .expect("builder");
        match request.rate_key(&extractor) {
            Cow::Borrowed(value) => assert_eq!(value, b"abc123"),
            Cow::Owned(_) => panic!("Header extractor must borrow from the request"),
        }
    }

    #[test]
    fn key_of_path_and_method_owns_the_synthesized_key() {
        let extractor = KeyExtractor::PathAndMethod;
        let request = Request::builder()
            .method("GET")
            .path("/v1/items/42")
            .build()
            .expect("builder");
        match request.rate_key(&extractor) {
            Cow::Borrowed(_) => panic!("PathAndMethod must synthesize a new String"),
            Cow::Owned(value) => assert_eq!(value, b"GET /v1/items/42"),
        }
    }

    #[test]
    fn parse_caps_uses_defaults_when_block_absent() {
        let caps: RateLimitCaps = CapsConfig::default().into();
        assert_eq!(caps.max_keys, 100_000);
        assert_eq!(caps.idle_ttl, Duration::from_secs(600));
    }

    #[test]
    fn parse_caps_overrides_individual_fields() {
        let value = serde_json::json!({"max_keys": 16, "idle_ttl_ms": 5000, "sweep_ms": 1000});
        let config: CapsConfig = serde_json::from_value(value).expect("parse");
        let caps: RateLimitCaps = config.into();
        assert_eq!(caps.max_keys, 16);
        assert_eq!(caps.idle_ttl, Duration::from_millis(5000));
        assert_eq!(caps.sweep_period, Duration::from_millis(1000));
    }

    #[derive(Clone, PartialEq, Debug)]
    struct TenantEvent {
        tenant: String,
        value: u64,
    }

    #[derive(Clone)]
    struct TenantExtractor;

    impl KeyOf<TenantExtractor> for TenantEvent {
        type Rejection = TenantEvent;

        fn rate_key<'a>(&'a self, _extractor: &'a TenantExtractor) -> Cow<'a, [u8]> {
            Cow::Borrowed(self.tenant.as_bytes())
        }

        fn build_rejection(_action: &ExceededAction) -> TenantEvent {
            TenantEvent {
                tenant: "__rate_limited__".into(),
                value: 0,
            }
        }
    }

    impl Labeled for TenantEvent {
        fn telemetry(&self) -> TelemetryHandle {
            Arc::new(Metrics::default())
        }

        fn labels(&self) -> Labels {
            Labels::empty()
        }
    }

    #[derive(Clone)]
    struct EventPassthrough;

    impl SendPipe for EventPassthrough {
        type In = TenantEvent;
        type Out = TenantEvent;
        type Err = ProximaError;

        fn call(
            &self,
            input: TenantEvent,
        ) -> impl Future<Output = Result<TenantEvent, ProximaError>> + Send {
            async move {
                Ok(TenantEvent {
                    value: input.value + 1,
                    ..input
                })
            }
        }
    }

    #[proxima::test]
    async fn rate_limit_is_generic_over_a_non_request_key_type() {
        let stack = RateLimit::new(
            EventPassthrough,
            TokenBucketConfig {
                capacity: 1,
                refill_per_sec: 0,
            },
            TenantExtractor,
        );

        let admitted = SendPipe::call(
            &stack,
            TenantEvent {
                tenant: "acme".into(),
                value: 10,
            },
        )
        .await
        .expect("first event admitted");
        assert_eq!(admitted.value, 11, "passthrough increments value");

        let limited = SendPipe::call(
            &stack,
            TenantEvent {
                tenant: "acme".into(),
                value: 10,
            },
        )
        .await
        .expect("second event returns Ok with sentinel");
        assert_eq!(
            limited.tenant, "__rate_limited__",
            "exhausted bucket returns build_rejection sentinel"
        );
        assert_eq!(limited.value, 0);

        let other = SendPipe::call(
            &stack,
            TenantEvent {
                tenant: "beta".into(),
                value: 5,
            },
        )
        .await
        .expect("different tenant has its own bucket");
        assert_eq!(other.value, 6, "beta bucket is independent");
    }

    // ── Clock-injectable determinism ────────────────────────────────────────
    //
    // `RateLimit::with_clock` takes any `Clock` impl, so refill can be proven
    // deterministically — `advance` moves virtual time, never a real sleep.

    #[derive(Clone, Default)]
    struct FakeClock {
        now_nanos: Arc<AtomicU64>,
    }

    impl FakeClock {
        fn advance(&self, duration: Duration) {
            let elapsed_ns = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
            self.now_nanos.fetch_add(elapsed_ns, Ordering::Relaxed);
        }
    }

    impl Clock for FakeClock {
        type Delay = std::future::Ready<()>;

        fn now_nanos(&self) -> u64 {
            self.now_nanos.load(Ordering::Relaxed)
        }

        fn delay(&self, _duration: Duration) -> Self::Delay {
            std::future::ready(())
        }
    }

    #[proxima::test]
    async fn fake_clock_drives_fill_exhaust_advance_refill_exhaust() {
        let clock = FakeClock::default();
        let stack = RateLimit::with_clock(
            into_handle(AlwaysOk),
            TokenBucketConfig {
                capacity: 2,
                refill_per_sec: 1,
            },
            KeyExtractor::ConstantKey("global".into()),
            RateLimitCaps::default(),
            clock.clone(),
        );

        // fill -> exhaust: capacity is 2, so two calls admit and a third refuses.
        for attempt in 0..2 {
            let (request, _metrics) = build_request_with_metrics("GET", "/");
            let response = SendPipe::call(&stack, request).await.expect("call");
            assert_eq!(response.status, 200, "attempt {attempt} within capacity");
        }
        let (request, _metrics) = build_request_with_metrics("GET", "/");
        let exhausted = SendPipe::call(&stack, request).await.expect("call");
        assert_eq!(exhausted.status, 429, "bucket exhausted, no time elapsed");

        // no clock advance yet: still exhausted at the same instant.
        let (request, _metrics) = build_request_with_metrics("GET", "/");
        let still_exhausted = SendPipe::call(&stack, request).await.expect("call");
        assert_eq!(
            still_exhausted.status, 429,
            "no wall-clock time passed, no fake-clock advance either"
        );

        // advance -> refill resumes: refill_per_sec=1 over 1s refills one token.
        clock.advance(Duration::from_secs(1));
        let (request, _metrics) = build_request_with_metrics("GET", "/");
        let resumed = SendPipe::call(&stack, request).await.expect("call");
        assert_eq!(resumed.status, 200, "1s advance refills exactly one token");

        // exhaust again at the same instant: that one refilled token is spent.
        let (request, _metrics) = build_request_with_metrics("GET", "/");
        let exhausted_again = SendPipe::call(&stack, request).await.expect("call");
        assert_eq!(
            exhausted_again.status, 429,
            "the refilled token was just spent; no further clock advance"
        );
    }

    #[proxima::test]
    async fn fake_clock_never_refills_when_no_time_advances() {
        let clock = FakeClock::default();
        let stack = RateLimit::with_clock(
            into_handle(AlwaysOk),
            TokenBucketConfig {
                capacity: 1,
                refill_per_sec: 1_000_000,
            },
            KeyExtractor::ConstantKey("global".into()),
            RateLimitCaps::default(),
            clock,
        );

        let (first, _metrics) = build_request_with_metrics("GET", "/");
        let admitted = SendPipe::call(&stack, first).await.expect("call");
        assert_eq!(admitted.status, 200);

        // even a generous refill rate cannot admit a second call: the clock
        // never advanced, so try_take's elapsed_ns is exactly zero.
        for _ in 0..5 {
            let (request, _metrics) = build_request_with_metrics("GET", "/");
            let response = SendPipe::call(&stack, request).await.expect("call");
            assert_eq!(
                response.status, 429,
                "no clock advance means no refill, regardless of call count"
            );
        }
    }
}
