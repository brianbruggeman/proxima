//! Periodic producer [`SourcePipe`](crate::pipe::source::SourcePipe) — wraps an
//! interval driver and dispatches a per-tick [`Request`] through an inner
//! [`Handler`](crate::pipe::handler::Handler).
//!
//! `IntervalPipe` is the [`SourcePipe`](crate::pipe::source::SourcePipe) that
//! converts "fire every N" into the source-shape
//! [`ProducerLifecycle`](crate::pipe::ProducerLifecycle) drives via
//! `spawn_from_source`. Each tick calls the configured request factory, then
//! dispatches the resulting `Request` through the inner handler.
//!
//! `IntervalPipe` is a source, not a handler: mounting one on a listener is a
//! compile error (its `SendPipe` shape is `Signal -> ()`, not
//! `Request<Bytes> -> Response<Bytes>`) — strictly better than the old
//! runtime 405 a caller only discovered by dispatching a request into it.
//!
//! # Composed primitives
//!
//! - [`proxima_core::time::interval`] — the tick source, runtime-agnostic.
//!   Tests observe real short-period ticks event-driven, bounded by
//!   [`proxima_core::time::timeout`].
//! - [`proxima_core::signal::Signal`] — the cooperative cancellation the
//!   lifecycle passes as `SendPipe::call`'s input; the loop returns once it
//!   observes the signal fire.
//!
//! # Why a wrapper exists vs. composing the primitives directly
//!
//! Every caller could spawn a `tokio::time::interval` future themselves
//! and dispatch through their inner handler. The wrapper centralises:
//! - cancellation-aware bias (cancel wins over a tick if both are ready,
//!   so shutdown doesn't fire one last spurious request)
//! - first-tick suppression so the first dispatch happens one period
//!   after startup, not immediately at construction
//! - the request-factory boundary that keeps `IntervalPipe` content-agnostic
//!   (`carry: Some(Carry::new(MyEvent::heartbeat()))`,
//!   `body: Bytes::from_static(b"ping")`, or a `carry` payload — decide what
//!   the tick payload is)

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use core::time::Duration;

use bytes::Bytes;
use futures::FutureExt;
use proxima_core::factory::Named;
use proxima_core::signal::Signal;

use crate::pipe::handler::PipeHandle;
use crate::pipe::header_list::HeaderList;
use crate::pipe::request::{Request, RequestContext};
use crate::pipe::source::{SourceFactory, SourceHandle, into_source_handle};
use proxima_core::ProximaError;
use crate::pipe::SendPipe;

/// Type-erased request-factory: called per tick to produce the [`Request`]
/// dispatched into the inner Pipe.
///
/// `Send + Sync + 'static` because the factory is moved into the spawned
/// background task and may be called from any worker thread.
pub type RequestFactory = Arc<dyn Fn() -> Request<Bytes> + Send + Sync + 'static>;

/// Default method-byte for an interval tick: `b"TICK"`. Configurable via
/// [`IntervalPipeBuilder::method`].
pub const DEFAULT_TICK_METHOD: &[u8] = b"TICK";

/// Default path for an interval tick: `"/tick"`. Configurable via
/// [`IntervalPipeBuilder::path`].
pub const DEFAULT_TICK_PATH: &[u8] = b"/tick";

/// Periodic producer Pipe. Fires every `period`, dispatches a Request
/// built by `request_factory` to `inner`.
///
/// The pipe is content-agnostic: the request factory decides whether
/// the body is empty `Bytes`, `Bytes::from_static(b"...")`,
/// a `Carry` typed payload, or stream — anything the spine can hold. The
/// method-byte and path are configurable per the discriminant
/// convention; default to `TICK` / `/tick` so a bare `IntervalPipe`
/// works as a heartbeat source without any extra config.
pub struct IntervalPipe {
    period: Duration,
    inner: PipeHandle,
    request_factory: RequestFactory,
    name: String,
}

impl IntervalPipe {
    /// Construct directly. Most callers should use [`Self::builder`].
    #[must_use]
    pub fn new(
        period: Duration,
        inner: PipeHandle,
        request_factory: RequestFactory,
        name: impl Into<String>,
    ) -> Self {
        Self {
            period,
            inner,
            request_factory,
            name: name.into(),
        }
    }

    /// Fluent builder entry point (workspace principle 4).
    #[must_use]
    pub fn builder() -> IntervalPipeBuilder {
        IntervalPipeBuilder::default()
    }

    /// The source's label, set via [`IntervalPipeBuilder::name`] (default
    /// `"interval"`). Used as the `ProducerLifecycle::spawn_from_source`
    /// task name by convention; carries no other runtime behaviour.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Build the default `RequestFactory` — emits a [`Request`] with
    /// method [`DEFAULT_TICK_METHOD`], path [`DEFAULT_TICK_PATH`], no
    /// headers, and an empty `Bytes` body. Useful when the inner Pipe only
    /// cares about the timing edge, not the payload.
    #[must_use]
    pub fn empty_request_factory() -> RequestFactory {
        Arc::new(|| Request {
            method: crate::pipe::method::Method::from_bytes(DEFAULT_TICK_METHOD),
            path: Bytes::from_static(DEFAULT_TICK_PATH),
            query: HeaderList::new(),
            metadata: HeaderList::new(),
            payload: Bytes::new(),
            stream: None,
            context: RequestContext::default(),
        })
    }
}

impl SendPipe for IntervalPipe {
    type In = Signal;
    type Out = ();
    type Err = ProximaError;

    fn call(&self, cancel: Signal) -> impl core::future::Future<Output = Result<(), ProximaError>> + Send {
        let inner = self.inner.clone();
        let period = self.period;
        let factory = self.request_factory.clone();
        async move {
            run_interval_loop(period, inner, factory, cancel).await;
            Ok(())
        }
    }
}

/// Config-driven [`SourceFactory`] for [`IntervalPipe`]. `spec` carries the
/// tick period in milliseconds; the inner handler and name are supplied by
/// the caller building the factory instance (mirrors the request-factory
/// boundary: the factory decides shape, the handler decides content).
pub struct IntervalSourceFactory {
    name: String,
    inner: PipeHandle,
}

impl IntervalSourceFactory {
    /// Register a thin factory that builds an [`IntervalPipe`] under `name`,
    /// dispatching every tick to `inner`.
    #[must_use]
    pub fn new(name: impl Into<String>, inner: PipeHandle) -> Self {
        Self {
            name: name.into(),
            inner,
        }
    }
}

impl Named for IntervalSourceFactory {
    fn name(&self) -> &str {
        &self.name
    }
}

impl SourceFactory for IntervalSourceFactory {
    fn build(&self, spec: &serde_json::Value) -> Result<SourceHandle, ProximaError> {
        let period_ms = spec
            .get("period_ms")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| ProximaError::Config("interval source requires period_ms".into()))?;
        let pipe = IntervalPipe::builder()
            .period(Duration::from_millis(period_ms))
            .inner(self.inner.clone())
            .name(self.name.clone())
            .build()
            .map_err(|err| ProximaError::Config(format!("interval source: {err}")))?;
        Ok(into_source_handle(pipe))
    }
}

async fn run_interval_loop(
    period: Duration,
    inner: PipeHandle,
    request_factory: RequestFactory,
    cancel: Signal,
) {
    let mut interval = proxima_core::time::interval(period);
    interval.set_missed_tick_behavior(proxima_core::time::MissedTickBehavior::Skip);
    // First tick fires immediately; skip it so the first dispatch is one
    // period after startup. Matches the principle-of-least-surprise for
    // heartbeats: "every 5s starting in 5s" rather than "at t=0 and every
    // 5s thereafter".
    interval.tick().await;
    loop {
        // futures::select_biased! per workspace tokio-elimination
        // discipline. biased: cancel wins over tick if both are ready,
        // so shutdown doesn't fire one last spurious dispatch.
        let cancel_fut = Box::pin(cancel.fired()).fuse();
        let tick_fut = Box::pin(interval.tick()).fuse();
        futures::pin_mut!(cancel_fut);
        futures::pin_mut!(tick_fut);
        let should_fire = futures::select_biased! {
            _ = cancel_fut => {
                tracing::debug!(target = "interval_pipe", "interval cancelled");
                return;
            }
            _ = tick_fut => true,
        };
        if should_fire {
            let request = request_factory();
            if let Err(err) = inner.call(request).await {
                tracing::error!(?err, target = "interval_pipe", "inner call failed");
            }
        }
    }
}

/// Builder for [`IntervalPipe`] (workspace principle 4).
#[derive(Default)]
pub struct IntervalPipeBuilder {
    period: Option<Duration>,
    inner: Option<PipeHandle>,
    request_factory: Option<RequestFactory>,
    method: Option<Bytes>,
    path: Option<Bytes>,
    name: Option<String>,
}

impl IntervalPipeBuilder {
    /// Set the tick period.
    #[must_use]
    pub fn period(mut self, period: Duration) -> Self {
        self.period = Some(period);
        self
    }

    /// Set the inner Pipe that receives the dispatched [`Request`].
    #[must_use]
    pub fn inner(mut self, inner: PipeHandle) -> Self {
        self.inner = Some(inner);
        self
    }

    /// Set a fully custom request factory. Each tick calls this closure
    /// to construct the Request. Overrides any [`method`]/[`path`]
    /// settings (those only affect the default factory).
    ///
    /// [`method`]: Self::method
    /// [`path`]: Self::path
    #[must_use]
    pub fn request_factory<F>(mut self, factory: F) -> Self
    where
        F: Fn() -> Request<Bytes> + Send + Sync + 'static,
    {
        self.request_factory = Some(Arc::new(factory));
        self
    }

    /// Override the method-byte used by the default factory. Ignored if
    /// a custom [`request_factory`] is set.
    ///
    /// [`request_factory`]: Self::request_factory
    #[must_use]
    pub fn method(mut self, method: impl Into<Bytes>) -> Self {
        self.method = Some(method.into());
        self
    }

    /// Override the path used by the default factory. Ignored if a
    /// custom [`request_factory`] is set.
    ///
    /// [`request_factory`]: Self::request_factory
    #[must_use]
    pub fn path(mut self, path: impl Into<Bytes>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Set the source's label (default `"interval"`). Used as the
    /// `spawn_from_source` task name and in tracing.
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Build the immutable [`IntervalPipe`].
    ///
    /// # Errors
    ///
    /// Returns `Err` if `period` or `inner` is unset.
    pub fn build(self) -> Result<IntervalPipe, IntervalBuildError> {
        let period = self.period.ok_or(IntervalBuildError::MissingPeriod)?;
        let inner = self.inner.ok_or(IntervalBuildError::MissingInner)?;
        let factory = match self.request_factory {
            Some(factory) => factory,
            None => {
                let method = crate::pipe::method::Method::from_wire(
                    self.method
                        .unwrap_or_else(|| Bytes::from_static(DEFAULT_TICK_METHOD)),
                );
                let path = self
                    .path
                    .unwrap_or_else(|| Bytes::from_static(DEFAULT_TICK_PATH));
                Arc::new(move || Request {
                    method: method.clone(),
                    path: path.clone(),
                    query: HeaderList::new(),
                    metadata: HeaderList::new(),
                    payload: Bytes::new(),
                    stream: None,
                    context: RequestContext::default(),
                })
            }
        };
        let name = self.name.unwrap_or_else(|| "interval".into());
        Ok(IntervalPipe::new(period, inner, factory, name))
    }
}

/// Errors from [`IntervalPipeBuilder::build`].
#[derive(Debug)]
#[non_exhaustive]
pub enum IntervalBuildError {
    /// `.period(...)` was not called.
    MissingPeriod,
    /// `.inner(...)` was not called.
    MissingInner,
}

impl core::fmt::Display for IntervalBuildError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingPeriod => formatter.write_str("IntervalPipe builder missing period"),
            Self::MissingInner => formatter.write_str("IntervalPipe builder missing inner"),
        }
    }
}

impl core::error::Error for IntervalBuildError {}

// `#[proxima::test]` pulls in the `proxima` dev-dependency, which the
// loom build keeps out of the graph (see
// `[target.'cfg(not(loom))'.dev-dependencies]` in Cargo.toml); these
// tests are unrelated to the Notify/watch loom protocol.
#[cfg(all(test, not(loom)))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::sync::Arc;
    use std::sync::Mutex;

    use crate::pipe::handler::PipeHandle;
    use crate::pipe::request::Response;
    use crate::pipe::SendPipe;

    use super::*;

    struct CountingPipe {
        count: Mutex<usize>,
        last_method: Mutex<crate::pipe::method::Method>,
    }

    impl CountingPipe {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                count: Mutex::new(0),
                last_method: Mutex::new(crate::pipe::method::Method::default()),
            })
        }

        fn count(&self) -> usize {
            *self.count.lock().unwrap()
        }

        fn last_method(&self) -> crate::pipe::method::Method {
            self.last_method.lock().unwrap().clone()
        }
    }

    impl SendPipe for CountingPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            request: Request<Bytes>,
        ) -> impl core::future::Future<Output = Result<Response<Bytes>, ProximaError>> + Send
        {
            let method = request.method.clone();
            *self.count.lock().unwrap() += 1;
            *self.last_method.lock().unwrap() = method;
            async { Ok(Response::ok("")) }
        }
    }

    fn into_handle(pipe: Arc<CountingPipe>) -> PipeHandle {
        pipe
    }

    #[proxima::test]
    async fn builder_returns_error_on_missing_period() {
        let inner = into_handle(CountingPipe::new());
        let result = IntervalPipe::builder().inner(inner).build();
        assert!(matches!(result, Err(IntervalBuildError::MissingPeriod)));
    }

    #[proxima::test]
    async fn builder_returns_error_on_missing_inner() {
        let result = IntervalPipe::builder()
            .period(Duration::from_secs(1))
            .build();
        assert!(matches!(result, Err(IntervalBuildError::MissingInner)));
    }

    /// Drive `IntervalPipe` as a `SourcePipe` inline until `capture` has seen
    /// `ticks` dispatches, then fire the cancel signal so the loop returns.
    /// The loop future never resolves on its own before that, so the watcher
    /// arm always wins; the outer timeout bounds a broken loop.
    async fn drive_until_ticks(pipe: &IntervalPipe, capture: &Arc<CountingPipe>, ticks: usize) {
        let cancel = Signal::new();
        let loop_future = SendPipe::call(pipe, cancel.clone()).fuse();
        let capture_for_watch = capture.clone();
        let watcher = async move {
            while capture_for_watch.count() < ticks {
                proxima_core::time::sleep(Duration::from_millis(1)).await;
            }
        }
        .fuse();
        futures::pin_mut!(loop_future, watcher);
        let raced = async {
            futures::select_biased! {
                () = watcher => (),
                _ = loop_future => panic!("interval loop exited without being fired"),
            }
        };
        proxima_core::time::timeout(Duration::from_secs(10), raced)
            .await
            .expect("interval_pipe should reach the tick count before the timeout");
        cancel.fire();
    }

    #[proxima::test]
    async fn interval_pipe_fires_three_ticks_with_default_method() {
        let capture = CountingPipe::new();
        let inner = into_handle(capture.clone());
        let pipe = IntervalPipe::builder()
            .period(Duration::from_millis(5))
            .inner(inner)
            .name("heartbeat")
            .build()
            .expect("builder");
        drive_until_ticks(&pipe, &capture, 3).await;
        assert!(capture.count() >= 3);
        assert_eq!(
            capture.last_method(),
            crate::pipe::method::Method::from_bytes(DEFAULT_TICK_METHOD)
        );
    }

    #[proxima::test]
    async fn interval_pipe_respects_custom_request_factory() {
        let capture = CountingPipe::new();
        let inner = into_handle(capture.clone());
        let pipe = IntervalPipe::builder()
            .period(Duration::from_millis(5))
            .inner(inner)
            .request_factory(|| Request {
                method: crate::pipe::method::Method::from_bytes(b"CUSTOM"),
                path: Bytes::from_static(b"/custom"),
                query: HeaderList::new(),
                metadata: HeaderList::new(),
                payload: Bytes::new(),
                stream: None,
                context: RequestContext::default(),
            })
            .build()
            .expect("builder");
        drive_until_ticks(&pipe, &capture, 1).await;
        assert!(capture.count() >= 1);
        assert_eq!(
            capture.last_method(),
            crate::pipe::method::Method::from_bytes(b"CUSTOM")
        );
    }

    #[proxima::test]
    async fn interval_source_factory_builds_from_period_ms_spec() {
        let capture = CountingPipe::new();
        let factory = IntervalSourceFactory::new("heartbeat", into_handle(capture.clone()));
        let source = factory
            .build(&serde_json::json!({ "period_ms": 5 }))
            .expect("factory build");

        let cancel = Signal::new();
        let call_future = SendPipe::call(&source, cancel.clone()).fuse();
        let capture_for_watch = capture.clone();
        let watcher = async move {
            while capture_for_watch.count() < 1 {
                proxima_core::time::sleep(Duration::from_millis(1)).await;
            }
        }
        .fuse();
        futures::pin_mut!(call_future, watcher);
        futures::select_biased! {
            () = watcher => (),
            _ = call_future => panic!("interval source exited without ticking"),
        }
        cancel.fire();
    }

    #[test]
    fn interval_source_factory_rejects_a_spec_missing_period_ms() {
        let factory = IntervalSourceFactory::new(
            "heartbeat",
            into_handle(CountingPipe::new()),
        );
        let outcome = factory.build(&serde_json::json!({}));
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }
}
