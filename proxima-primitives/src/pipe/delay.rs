//! Per-call artificial-latency [`SendPipe`]/[`Pipe`] combinator — waits a
//! sampled [`Dist`]ribution before dispatching each call to the inner pipe.
//!
//! # Composed primitives
//!
//! - [`Clock`] — the injectable sleep seam
//!   `Delay` schedules its wait against, the same seam
//!   [`Retry`](crate::pipe::retry::Retry) and
//!   [`RateLimit`](crate::pipe::rate_limit::RateLimit) are generic over.
//!   `Delay` only calls `Clock::delay`; it never reads `Clock::now_nanos`
//!   (it has no deadline arithmetic to do).
//! - [`TimeClock`] — the production `Clock`, and the default `Clk` type
//!   parameter, so every existing caller (`Delay::new`, `DelayConfig`,
//!   `DelayFactory`) is unaffected.
//!
//! # Why a wrapper exists vs. composing the primitive directly
//!
//! `Clock::delay` alone is just a future to await; `Delay` adds the policy
//! around it: per-call-index sampling (`Dist::Const` / `Dist::Range`, seeded
//! and deterministic), an optional [`When`] gate that skips the wait on a
//! miss, and the serde/`PipeFactory` surface (`DelayConfig`, `DelayFactory`)
//! that lets a pipeline graph configure it declaratively.

use alloc::sync::Arc;
use bytes::Bytes;
use core::future::Future;
use core::time::Duration;
use portable_atomic::{AtomicU64, Ordering};

use crate::pipe::SendPipe;
use crate::pipe::capabilities::Clock;
use crate::pipe::clock::TimeClock;
use crate::pipe::primitives::Pipe;
use crate::pipe::when::When;
use serde::{Deserialize, Serialize};

use crate::pipe::handler::{PipeHandle, ThreadLocalPipeHandle};
use crate::pipe::request::{Request, Response};
use proxima_core::ProximaError;

#[cfg(feature = "std")]
use crate::pipe::handler::into_handle;
#[cfg(feature = "std")]
use crate::pipe::pipe_factory::PipeFactory;
#[cfg(feature = "std")]
use serde_json::Value;
#[cfg(feature = "std")]
use std::pin::Pin;

/// How a [`Delay`] picks the duration to wait before each call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Dist {
    Const { ms: u64 },
    Range { min_ms: u64, max_ms: u64, seed: u64 },
}

impl Dist {
    #[must_use]
    pub fn sample(&self, call_index: u64) -> Duration {
        match self {
            Dist::Const { ms } => Duration::from_millis(*ms),
            Dist::Range {
                min_ms,
                max_ms,
                seed,
            } => {
                let (low, high) = if min_ms <= max_ms {
                    (*min_ms, *max_ms)
                } else {
                    (*max_ms, *min_ms)
                };
                let mut rng = fastrand::Rng::with_seed(seed.wrapping_add(call_index));
                Duration::from_millis(rng.u64(low..=high))
            }
        }
    }
}

impl Default for Dist {
    fn default() -> Self {
        Dist::Const { ms: 0 }
    }
}

// ── main struct ──────────────────────────────────────────────────────────────

pub struct Delay<Inner = PipeHandle, Clk = TimeClock> {
    pub inner: Inner,
    dist: Dist,
    clock: Clk,
    when: Option<When>,
    calls: Arc<AtomicU64>,
}

impl<Inner> Delay<Inner, TimeClock> {
    #[must_use]
    pub fn new(inner: Inner, dist: Dist) -> Self {
        Self::with_clock(inner, dist, TimeClock)
    }
}

impl<Inner, Clk> Delay<Inner, Clk> {
    /// Materialise with an explicit [`Clock`] — the seam a deterministic test
    /// or example injects a fake clock through; production code goes via
    /// [`Delay::new`], which defaults `Clk` to [`TimeClock`].
    #[must_use]
    pub fn with_clock(inner: Inner, dist: Dist, clock: Clk) -> Self {
        Self {
            inner,
            dist,
            clock,
            when: None,
            calls: Arc::new(AtomicU64::new(0)),
        }
    }

    #[must_use]
    pub fn with_when(mut self, when: When) -> Self {
        self.when = Some(when);
        self
    }

    #[must_use]
    pub fn dist(&self) -> Dist {
        self.dist
    }

    #[must_use]
    pub fn when(&self) -> Option<When> {
        self.when
    }

    fn wait_for(&self, call_index: u64) -> Duration {
        match self.when {
            Some(gate) if !gate.fires(call_index) => Duration::ZERO,
            _ => self.dist.sample(call_index),
        }
    }
}

impl<Inner, Clk> SendPipe for Delay<Inner, Clk>
where
    Inner: SendPipe + Clone + Send + Sync + 'static,
    Inner::In: Send,
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
        let call_index = self.calls.fetch_add(1, Ordering::Relaxed);
        let wait = self.wait_for(call_index);
        let sleep = self.clock.delay(wait);
        let inner = self.inner.clone();
        async move {
            sleep.await;
            SendPipe::call(&inner, input).await
        }
    }
}

impl<Clk> Pipe for Delay<ThreadLocalPipeHandle, Clk>
where
    ThreadLocalPipeHandle:
        Pipe<In = Request<Bytes>, Out = Response<Bytes>, Err = ProximaError>,
    Clk: Clock + Clone + 'static,
{
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        input: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        let call_index = self.calls.fetch_add(1, Ordering::Relaxed);
        let wait = self.wait_for(call_index);
        let sleep = self.clock.delay(wait);
        let inner = self.inner.clone();
        async move {
            sleep.await;
            Pipe::call(&inner, input).await
        }
    }
}

// ── serde config + factory ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DelayConfig {
    pub dist: Dist,
}

impl DelayConfig {
    #[must_use]
    pub fn into_delay(self, inner: PipeHandle) -> Delay<PipeHandle, TimeClock> {
        Delay::new(inner, self.dist)
    }
}

impl Delay<PipeHandle, TimeClock> {
    #[must_use]
    pub fn to_config(&self) -> DelayConfig {
        DelayConfig { dist: self.dist }
    }

    #[cfg(feature = "std")]
    pub fn from_spec(inner: PipeHandle, value: &Value) -> Result<Self, ProximaError> {
        let config: DelayConfig = serde_json::from_value(value.clone())
            .map_err(|err| ProximaError::Config(format!("delay config: {err}")))?;
        Ok(config.into_delay(inner))
    }
}

#[cfg(feature = "std")]
pub struct DelayFactory;

#[cfg(feature = "std")]
impl PipeFactory for DelayFactory {
    fn name(&self) -> &str {
        "delay"
    }

    fn build(
        &self,
        spec: &Value,
        inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        Box::pin(async move {
            let inner =
                inner.ok_or_else(|| ProximaError::Config("delay requires an inner pipe".into()))?;
            let delay = Delay::from_spec(inner, &spec)?;
            Ok(into_handle(delay))
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::future::Future;
    use std::sync::atomic::AtomicBool;
    use std::task::{Context, Poll};

    use futures::task::noop_waker;
    use proxima_core::time::drivers::mock::MockDriver;
    use proxima_core::time::{Driver, Instant};

    use super::*;
    use crate::pipe::handler::into_handle;

    #[derive(Clone)]
    struct MockClock {
        driver: Arc<MockDriver>,
    }

    impl MockClock {
        fn new() -> Self {
            Self {
                driver: Arc::new(MockDriver::new()),
            }
        }

        fn advance(&self, delta: Duration) {
            self.driver.advance(delta);
        }
    }

    struct MockSleep {
        driver: Arc<MockDriver>,
        deadline: Instant,
    }

    impl Future for MockSleep {
        type Output = ();
        fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
            if self.driver.now() >= self.deadline {
                Poll::Ready(())
            } else {
                self.driver
                    .schedule_wake(self.deadline, context.waker().clone());
                Poll::Pending
            }
        }
    }

    impl Clock for MockClock {
        type Delay = MockSleep;

        fn now_nanos(&self) -> u64 {
            u64::try_from(self.driver.now().into_monotonic().as_nanos()).unwrap_or(u64::MAX)
        }

        fn delay(&self, duration: Duration) -> MockSleep {
            let deadline = self.driver.now() + duration;
            MockSleep {
                driver: self.driver.clone(),
                deadline,
            }
        }
    }

    #[derive(Clone)]
    struct RanFlag {
        ran: Arc<AtomicBool>,
    }

    impl SendPipe for RanFlag {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            let ran = self.ran.clone();
            async move {
                ran.store(true, Ordering::SeqCst);
                Ok(Response::ok("ok"))
            }
        }
    }


    fn build_request() -> Request<Bytes> {
        Request::builder()
            .method("GET")
            .path("/v1/x")
            .body("payload")
            .build()
            .expect("builder")
    }

    #[test]
    fn delay_waits_for_the_configured_duration_before_running_inner() {
        let clock = MockClock::new();
        let ran = Arc::new(AtomicBool::new(false));
        let inner = RanFlag { ran: ran.clone() };
        let delay = Delay::with_clock(inner, Dist::Const { ms: 100 }, clock.clone());

        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);
        let mut call = Box::pin(SendPipe::call(&delay, build_request()));

        assert!(
            matches!(call.as_mut().poll(&mut context), Poll::Pending),
            "delay parks before its deadline"
        );
        assert!(
            !ran.load(Ordering::SeqCst),
            "inner pipe must not run while the delay is pending"
        );

        clock.advance(Duration::from_millis(99));
        assert!(
            matches!(call.as_mut().poll(&mut context), Poll::Pending),
            "still parked one ms short of the deadline"
        );
        assert!(!ran.load(Ordering::SeqCst), "inner still has not run");

        clock.advance(Duration::from_millis(1));
        match call.as_mut().poll(&mut context) {
            Poll::Ready(result) => {
                let response = result.expect("call ok");
                assert_eq!(response.status, 200);
            }
            Poll::Pending => panic!("delay must complete once the mock clock crosses the deadline"),
        }
        assert!(
            ran.load(Ordering::SeqCst),
            "inner pipe ran after the delay elapsed"
        );
    }

    #[test]
    fn zero_delay_runs_inner_on_first_poll() {
        let clock = MockClock::new();
        let ran = Arc::new(AtomicBool::new(false));
        let inner = RanFlag { ran: ran.clone() };
        let delay = Delay::with_clock(inner, Dist::Const { ms: 0 }, clock);

        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);
        let mut call = Box::pin(SendPipe::call(&delay, build_request()));

        match call.as_mut().poll(&mut context) {
            Poll::Ready(result) => assert_eq!(result.expect("call ok").status, 200),
            Poll::Pending => panic!("zero delay must not park"),
        }
        assert!(ran.load(Ordering::SeqCst), "inner ran immediately");
    }

    #[test]
    fn range_sample_is_deterministic_and_within_bounds() {
        let dist = Dist::Range {
            min_ms: 10,
            max_ms: 50,
            seed: 0xC0FFEE,
        };
        let first: Vec<Duration> = (0..256).map(|index| dist.sample(index)).collect();
        let second: Vec<Duration> = (0..256).map(|index| dist.sample(index)).collect();
        assert_eq!(
            first, second,
            "same (seed, bounds) yields an identical sample sequence"
        );
        for sample in first {
            assert!(
                sample >= Duration::from_millis(10) && sample <= Duration::from_millis(50),
                "sample {sample:?} stays within [10ms, 50ms]"
            );
        }
    }

    #[test]
    fn const_sample_ignores_the_call_index() {
        let dist = Dist::Const { ms: 42 };
        for index in 0..64 {
            assert_eq!(
                dist.sample(index),
                Duration::from_millis(42),
                "const delay is index-independent"
            );
        }
    }

    #[test]
    fn config_builder_round_trip_parity() {
        fn echo_pipe() -> PipeHandle {
            struct EchoPipe;
            impl SendPipe for EchoPipe {
                type In = Request<Bytes>;
                type Out = Response<Bytes>;
                type Err = ProximaError;
                fn call(
                    &self,
                    _request: Request<Bytes>,
                ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send
                {
                    async move { Ok(Response::ok("ok")) }
                }
            }
            into_handle(EchoPipe)
        }

        let config = DelayConfig {
            dist: Dist::Range {
                min_ms: 5,
                max_ms: 25,
                seed: 7,
            },
        };

        let delay = config.into_delay(echo_pipe());
        let back = delay.to_config();
        let json = serde_json::to_value(config).expect("serialize");
        let parsed: DelayConfig = serde_json::from_value(json.clone()).expect("deserialize");

        assert_eq!(
            back, config,
            "builder projects back to the originating config"
        );
        assert_eq!(parsed, config, "serde round-trip is lossless");
        assert_eq!(
            json,
            serde_json::json!({ "dist": { "kind": "range", "min_ms": 5, "max_ms": 25, "seed": 7 } }),
            "Dist serializes as a tagged object"
        );
    }

    #[test]
    fn from_spec_parses_const_dist() {
        fn echo_pipe() -> PipeHandle {
            struct EchoPipe;
            impl SendPipe for EchoPipe {
                type In = Request<Bytes>;
                type Out = Response<Bytes>;
                type Err = ProximaError;
                fn call(
                    &self,
                    _request: Request<Bytes>,
                ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send
                {
                    async move { Ok(Response::ok("ok")) }
                }
            }
            into_handle(EchoPipe)
        }

        let value = serde_json::json!({ "dist": { "kind": "const", "ms": 250 } });
        let delay = Delay::from_spec(echo_pipe(), &value).expect("from spec");
        assert_eq!(delay.dist(), Dist::Const { ms: 250 });
    }

    #[derive(Clone, PartialEq, Debug)]
    struct CountEvent(u64);

    #[derive(Clone)]
    struct EventSink;

    impl SendPipe for EventSink {
        type In = CountEvent;
        type Out = CountEvent;
        type Err = ProximaError;

        fn call(
            &self,
            input: CountEvent,
        ) -> impl Future<Output = Result<CountEvent, ProximaError>> + Send {
            async move { Ok(CountEvent(input.0 + 1)) }
        }
    }

    #[test]
    fn delay_is_generic_over_a_non_http_payload() {
        let clock = MockClock::new();
        let delay = Delay::with_clock(EventSink, Dist::Const { ms: 30 }, clock.clone());

        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);
        let mut call = Box::pin(SendPipe::call(&delay, CountEvent(41)));

        assert!(
            matches!(call.as_mut().poll(&mut context), Poll::Pending),
            "event call parks before the deadline"
        );
        clock.advance(Duration::from_millis(30));
        match call.as_mut().poll(&mut context) {
            Poll::Ready(result) => assert_eq!(result.expect("event call"), CountEvent(42)),
            Poll::Pending => panic!("event call must complete once the deadline passes"),
        }
    }

    #[test]
    fn gated_delay_skips_sleep_on_a_miss_and_parks_on_a_fire() {
        let gate = When::prob(1.0).seed(1);
        let clock = MockClock::new();
        let ran = Arc::new(AtomicBool::new(false));
        let delay = Delay::with_clock(
            RanFlag { ran: ran.clone() },
            Dist::Const { ms: 50 },
            clock.clone(),
        )
        .with_when(gate);

        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);
        let mut call = Box::pin(SendPipe::call(&delay, build_request()));

        assert!(
            matches!(call.as_mut().poll(&mut context), Poll::Pending),
            "fired gate parks the delay"
        );
        assert!(
            !ran.load(Ordering::SeqCst),
            "inner has not run while parked"
        );
        clock.advance(Duration::from_millis(50));
        assert!(
            matches!(call.as_mut().poll(&mut context), Poll::Ready(_)),
            "delay completes after the deadline"
        );
        assert!(
            ran.load(Ordering::SeqCst),
            "inner ran after the gated delay elapsed"
        );
    }

    #[test]
    fn gated_delay_with_a_never_firing_gate_runs_inner_immediately() {
        let gate = When::prob(0.0).seed(1);
        let clock = MockClock::new();
        let ran = Arc::new(AtomicBool::new(false));
        let delay = Delay::with_clock(
            RanFlag { ran: ran.clone() },
            Dist::Const { ms: 9_999 },
            clock,
        )
        .with_when(gate);

        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);
        let mut call = Box::pin(SendPipe::call(&delay, build_request()));

        match call.as_mut().poll(&mut context) {
            Poll::Ready(result) => assert_eq!(result.expect("call ok").status, 200),
            Poll::Pending => panic!("a never-firing gate must skip the sleep"),
        }
        assert!(ran.load(Ordering::SeqCst), "inner ran without any wait");
    }
}
