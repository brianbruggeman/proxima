use bytes::Bytes;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::time::Duration;

use bon::Builder;
use conflaguration::Settings;
use futures::FutureExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::pipe::ProximaError;
use crate::pipe::primitives::Pipe;
use crate::pipe::SendPipe;
use crate::pipe::handler::{Handler, PipeHandle, ThreadLocalPipeHandle, into_handle};
use crate::pipe::pipe_factory::PipeFactory;
use crate::pipe::request::{Request, Response};
use crate::pipe::telemetry_surface::Labels;

const METRIC_TIMEOUT: &str = "proxima.isolate.timeout_total";
const METRIC_PANIC: &str = "proxima.isolate.panic_total";

/// blast-radius barrier around an inner Pipe. enforces:
/// - **time budget** — `proxima_core::time::timeout` drops the inner future and
///   returns 503 when exceeded; the rest of the chain stays healthy.
/// - **panic barrier** — `catch_unwind` converts a panic in the inner
///   future into a 500 response instead of unwinding through the chain.
///
/// either guard is independently optional; with both off this is a
/// pass-through.
///
/// generic over the inner handle so the same wrapper composes a `Send`
/// inner (`PipeHandle`, the default — yields `impl Handler`) or a
/// per-thread `!Send`-tolerant inner (`ThreadLocalPipeHandle` — yields
/// `impl ThreadLocalHandler`). The Send variant participates in
/// cross-thread bootstrap; the per-thread variant lets DPDK / per-core
/// runtimes hold `Rc`/`RefCell`/`!Send` impl state.
pub struct Isolate<Inner = PipeHandle> {
    pub inner: Inner,
    pub timeout: Option<Duration>,
    pub panic_barrier: bool,
}

impl<Inner> Isolate<Inner> {
    #[must_use]
    pub fn new(inner: Inner) -> Self {
        Self {
            inner,
            timeout: None,
            panic_barrier: false,
        }
    }

    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    #[must_use]
    pub fn with_panic_barrier(mut self, enabled: bool) -> Self {
        self.panic_barrier = enabled;
        self
    }
}

impl Isolate<PipeHandle> {
    pub fn from_spec(inner: PipeHandle, value: &Value) -> Result<Self, ProximaError> {
        let config: IsolateConfig = serde_json::from_value(value.clone())
            .map_err(|err| ProximaError::Config(format!("isolate config: {err}")))?;
        Ok(config.into_isolate(inner))
    }
}

/// Typed config surface for the `isolate` middleware — a blast-radius barrier
/// with an optional time budget and panic barrier around an inner pipe.
#[derive(Debug, Clone, Default, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_ISOLATE")]
#[builder(derive(Clone, Debug))]
pub struct IsolateConfig {
    /// Time budget in ms; the inner future is dropped and a 503 returned when
    /// exceeded. `None` disables the timeout guard.
    #[setting(default)]
    #[serde(default)]
    pub timeout_ms: Option<u64>,

    /// Convert an inner panic into a 500 instead of unwinding the chain.
    #[setting(default)]
    #[serde(default)]
    #[builder(default)]
    pub panic_barrier: bool,
}

impl IsolateConfig {
    /// Materialise the isolate middleware around `inner`.
    #[must_use]
    pub fn into_isolate(self, inner: PipeHandle) -> Isolate<PipeHandle> {
        Isolate {
            inner,
            timeout: self.timeout_ms.map(Duration::from_millis),
            panic_barrier: self.panic_barrier,
        }
    }
}

impl<Inner> SendPipe for Isolate<Inner>
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
        let inner = self.inner.clone();
        let timeout = self.timeout;
        let panic_barrier = self.panic_barrier;
        let telemetry = request.context.telemetry.clone();
        let labels = request.context.metric_labels(&[]);
        async move {
            let guarded = run_with_panic_barrier(inner, request, panic_barrier);
            let outcome = match timeout {
                Some(budget) => match proxima_core::time::timeout(budget, guarded).await {
                    Ok(outcome) => outcome.unwrap_or_else(|reason| panic_response(&reason)),
                    Err(_elapsed) => {
                        telemetry.counter_inc(METRIC_TIMEOUT, &labels, 1);
                        Ok(timeout_response(budget))
                    }
                },
                None => guarded
                    .await
                    .unwrap_or_else(|reason| panic_response(&reason)),
            };
            outcome.map(|response| record_panic_metric(response, &telemetry, &labels))
        }
    }
}

impl Pipe for Isolate<ThreadLocalPipeHandle> {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        let inner = self.inner.clone();
        let timeout = self.timeout;
        let panic_barrier = self.panic_barrier;
        let telemetry = request.context.telemetry.clone();
        let labels = request.context.metric_labels(&[]);
        async move {
            let guarded = run_with_panic_barrier_tls(inner, request, panic_barrier);
            let outcome = match timeout {
                Some(budget) => match proxima_core::time::timeout(budget, guarded).await {
                    Ok(outcome) => outcome.unwrap_or_else(|reason| panic_response(&reason)),
                    Err(_elapsed) => {
                        telemetry.counter_inc(METRIC_TIMEOUT, &labels, 1);
                        Ok(timeout_response(budget))
                    }
                },
                None => guarded
                    .await
                    .unwrap_or_else(|reason| panic_response(&reason)),
            };
            outcome.map(|response| record_panic_metric(response, &telemetry, &labels))
        }
    }
}

/// run inner.call with optional catch_unwind. returns Ok(outcome) when no
/// panic was caught, Err(reason) when the inner future panicked. the outer
/// caller then decides whether the panic gets a timeout-style mapping or
/// the direct panic_response.
async fn run_with_panic_barrier<Inner: Handler + Clone>(
    inner: Inner,
    request: Request<Bytes>,
    enabled: bool,
) -> Result<Result<Response<Bytes>, ProximaError>, String> {
    if !enabled {
        return Ok(SendPipe::call(&inner, request).await);
    }
    let future = AssertUnwindSafe(SendPipe::call(&inner, request));
    match future.catch_unwind().await {
        Ok(outcome) => Ok(outcome),
        Err(panic) => Err(describe_panic(panic.as_ref())),
    }
}

async fn run_with_panic_barrier_tls(
    inner: ThreadLocalPipeHandle,
    request: Request<Bytes>,
    enabled: bool,
) -> Result<Result<Response<Bytes>, ProximaError>, String> {
    if !enabled {
        return Ok(Pipe::call(&inner, request).await);
    }
    let future = AssertUnwindSafe(Pipe::call(&inner, request));
    match future.catch_unwind().await {
        Ok(outcome) => Ok(outcome),
        Err(panic) => Err(describe_panic(panic.as_ref())),
    }
}

fn describe_panic(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

fn panic_response(reason: &str) -> Result<Response<Bytes>, ProximaError> {
    Ok(Response::new(500)
        .with_header("x-proxima-isolated", "panic")
        .with_body(bytes::Bytes::from(format!(
            "isolated pipe panicked: {reason}"
        ))))
}

fn timeout_response(budget: Duration) -> Response<Bytes> {
    Response::new(503)
        .with_header("x-proxima-isolated", "timeout")
        .with_body(bytes::Bytes::from(format!(
            "isolated pipe exceeded {}ms budget",
            budget.as_millis()
        )))
}

/// look at the outgoing response for our own panic header so we can bump
/// the metric exactly once, regardless of whether the panic was caught
/// directly or by way of the timeout wrapper.
fn record_panic_metric(
    response: Response<Bytes>,
    telemetry: &crate::pipe::telemetry_surface::TelemetryHandle,
    labels: &Labels,
) -> Response<Bytes> {
    if response.metadata.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case(b"x-proxima-isolated") && value.as_ref() == b"panic"
    }) {
        telemetry.counter_inc(METRIC_PANIC, labels, 1);
    }
    response
}

pub struct IsolateFactory;

impl PipeFactory for IsolateFactory {
    fn name(&self) -> &str {
        "isolate"
    }

    fn build(
        &self,
        spec: &Value,
        inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        Box::pin(async move {
            let inner = inner
                .ok_or_else(|| ProximaError::Config("isolate requires an inner pipe".into()))?;
            let isolate = Isolate::from_spec(inner, &spec)?;
            Ok(into_handle(isolate))
        })
    }
}

// `#[proxima::test]` and inline `tokio::task::{LocalSet, spawn_local}` pull
// in the `proxima` / `tokio` dev-dependencies, which the loom build keeps
// out of the graph (see `[target.'cfg(not(loom))'.dev-dependencies]` in
// Cargo.toml); these tests are unrelated to the Notify/watch loom protocol.
#[cfg(all(test, not(loom)))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::pipe::telemetry_surface::{Telemetry, TelemetryHandle};
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[derive(Default)]
    struct CounterRecorder {
        counters: Mutex<HashMap<String, u64>>,
    }

    impl Telemetry for CounterRecorder {
        fn counter_inc(&self, metric: &str, _labels: &Labels, by: u64) {
            *self
                .counters
                .lock()
                .unwrap()
                .entry(metric.to_string())
                .or_insert(0) += by;
        }
        fn gauge_set(&self, _: &str, _: &Labels, _: i64) {}
        fn histogram_record(&self, _: &str, _: &Labels, _: f64) {}
    }

    impl CounterRecorder {
        fn counter(&self, metric: &str) -> Option<u64> {
            self.counters.lock().unwrap().get(metric).copied()
        }
    }

    struct SleepingPipe {
        millis: u64,
    }

    impl SendPipe for SleepingPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            let millis = self.millis;
            async move {
                proxima_core::time::sleep(Duration::from_millis(millis)).await;
                Ok(Response::ok("done"))
            }
        }
    }

    struct PanickingPipe;

    impl SendPipe for PanickingPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            async { panic!("intentional test panic") }
        }
    }

    fn request_with_metrics() -> (Request<Bytes>, Arc<CounterRecorder>) {
        let metrics: Arc<CounterRecorder> = Arc::new(CounterRecorder::default());
        let telemetry: TelemetryHandle = metrics.clone();
        let request = Request::builder()
            .method("GET")
            .path("/x")
            .body("")
            .telemetry(telemetry)
            .build()
            .expect("builder");
        (request, metrics)
    }

    // principle-4 parity: the fluent builder and the config value must lower to
    // identical Isolate state (timeout + panic barrier).
    #[test]
    fn parity_fluent_builder_and_config_value_match() {
        let from_value: IsolateConfig =
            serde_json::from_value(serde_json::json!({"timeout_ms": 250, "panic_barrier": true}))
                .expect("from_value");
        let from_value = from_value.into_isolate(into_handle(SleepingPipe { millis: 0 }));

        let from_builder = IsolateConfig::builder()
            .timeout_ms(250)
            .panic_barrier(true)
            .build()
            .into_isolate(into_handle(SleepingPipe { millis: 0 }));

        assert_eq!(from_value.timeout, from_builder.timeout);
        assert_eq!(from_value.timeout, Some(Duration::from_millis(250)));
        assert_eq!(from_value.panic_barrier, from_builder.panic_barrier);
    }

    #[proxima::test]
    async fn passthrough_when_no_guards_enabled() {
        let inner = into_handle(SleepingPipe { millis: 0 });
        let isolate = Isolate::new(inner);
        let (request, _metrics) = request_with_metrics();
        let response = SendPipe::call(&isolate, request).await.expect("call");
        assert_eq!(response.status, 200);
    }

    #[proxima::test]
    async fn timeout_returns_503_and_increments_metric() {
        // real-time test — 10ms budget vs 1s sleep is plenty of headroom
        // without bringing in tokio's `test-util` feature for pause/advance.
        let inner = into_handle(SleepingPipe { millis: 1_000 });
        let isolate = Isolate::new(inner).with_timeout(Duration::from_millis(10));
        let (request, metrics) = request_with_metrics();
        let response = SendPipe::call(&isolate, request).await.expect("call");
        assert_eq!(response.status, 503);
        let isolated = response
            .metadata
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(b"x-proxima-isolated"))
            .expect("isolated header");
        assert_eq!(isolated.1.as_ref(), b"timeout");
        let count = metrics.counter(METRIC_TIMEOUT).expect("timeout counter");
        assert_eq!(count, 1);
    }

    #[proxima::test]
    async fn panic_barrier_converts_panic_to_500_and_increments_metric() {
        let inner = into_handle(PanickingPipe);
        let isolate = Isolate::new(inner).with_panic_barrier(true);
        let (request, metrics) = request_with_metrics();
        let response = SendPipe::call(&isolate, request).await.expect("call");
        assert_eq!(response.status, 500);
        let isolated = response
            .metadata
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(b"x-proxima-isolated"))
            .expect("isolated header");
        assert_eq!(isolated.1.as_ref(), b"panic");
        let count = metrics.counter(METRIC_PANIC).expect("panic counter");
        assert_eq!(count, 1);
    }

    #[proxima::test(runtime = "tokio")]
    async fn panic_without_barrier_propagates() {
        let inner = into_handle(PanickingPipe);
        let isolate = Isolate::new(inner); // barrier disabled
        let (request, _metrics) = request_with_metrics();
        // panic propagates through .await — catch via spawn_local's JoinError.
        // Handler::call returns ?Send (per-core only); must spawn on LocalSet.
        let outcome = tokio::task::LocalSet::new()
            .run_until(async move {
                tokio::task::spawn_local(async move { SendPipe::call(&isolate, request).await })
                    .await
            })
            .await;
        assert!(outcome.is_err(), "panic must propagate without barrier");
    }

    #[proxima::test]
    async fn timeout_does_not_fire_for_fast_inner() {
        let inner = into_handle(SleepingPipe { millis: 0 });
        let isolate = Isolate::new(inner).with_timeout(Duration::from_millis(100));
        let (request, _metrics) = request_with_metrics();
        let response = SendPipe::call(&isolate, request).await.expect("call");
        assert_eq!(response.status, 200);
    }

    #[proxima::test]
    async fn neighbor_request_unaffected_after_inner_panic() {
        // proves the "rest of chain unaffected" acceptance: build a chain
        // where one call panics (caught by barrier), then a second call
        // through the same Isolate succeeds.
        let toggled = Arc::new(AtomicBool::new(false));
        struct OnceFail {
            toggled: Arc<AtomicBool>,
        }

        impl SendPipe for OnceFail {
            type In = Request<Bytes>;
            type Out = Response<Bytes>;
            type Err = ProximaError;

            fn call(
                &self,
                _request: Request<Bytes>,
            ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
                let toggled = self.toggled.clone();
                async move {
                    if toggled.swap(true, Ordering::SeqCst) {
                        Ok(Response::ok("recovered"))
                    } else {
                        panic!("first-call panic")
                    }
                }
            }
        }

        let inner = into_handle(OnceFail {
            toggled: toggled.clone(),
        });
        let isolate = Isolate::new(inner).with_panic_barrier(true);
        let (request_one, _m1) = request_with_metrics();
        let first = SendPipe::call(&isolate, request_one).await.expect("first");
        assert_eq!(first.status, 500);
        let (request_two, _m2) = request_with_metrics();
        let second = SendPipe::call(&isolate, request_two).await.expect("second");
        assert_eq!(second.status, 200);
    }
}
