//! `TimedReplay` — replay a recording in one of two timing modes.
//!
//! The durable read path ([`crate::pipe::log_pipe::ReplayLog`]) and every
//! [`RecordingSource`] yield events strictly IN RECORD ORDER, as fast as the
//! reader produces them. That is the [`ReplayMode::CausalOrder`] behaviour:
//! order is preserved, inter-event wall time is collapsed to zero. It is what
//! verify / diff / fast-forward want.
//!
//! Some consumers instead want the recording to play back at its ORIGINAL
//! cadence — a load replayer that reproduces the upstream's real arrival
//! pattern, or a UI that animates a captured stream as it actually happened.
//! [`ReplayMode::TimingIntact`] honours the recorded inter-event deltas: each
//! event carries a `ts_ms`, and between consecutive events the replayer sleeps
//! `ts_ms[i] - ts_ms[i-1]` (saturating, so a non-monotonic clock never
//! rewinds) through the injectable [`Clock`] seam before yielding the next.
//!
//! # Composed primitives
//!
//! - [`Clock`] — the same injectable sleep seam
//!   [`Delay`](proxima_primitives::pipe::Delay), `Retry`, and `RateLimit` are
//!   generic over. `TimedReplay` only calls `Clock::delay`; it never reads
//!   `Clock::now_nanos`.
//! - [`TimeClock`](proxima_primitives::pipe::clock::TimeClock) — the
//!   production `Clock`, and the default `Clk` type parameter, so every
//!   existing caller (`TimedReplay::new`, `ReplayConfig`) is unaffected.
//!
//! Production delegates to `proxima_core::time::sleep` (registers a waker via
//! the active driver's `schedule_wake`, fired by the driver — never a busy
//! poll); tests inject a `Clock` backed by a deterministic `MockDriver` and
//! advance it by hand, so a timing-intact replay test waits zero real time.

use core::time::Duration;

use async_stream::try_stream;
use serde::{Deserialize, Serialize};

use crate::source::{DynRecordingSource, RecordingEventStream};
use proxima_core::ProximaError;
use proxima_primitives::pipe::capabilities::Clock;
use proxima_primitives::pipe::clock::TimeClock;

// ── replay mode — config-expressible, the bidirectional twin of the builder ──

/// How a [`TimedReplay`] paces the events it yields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReplayMode {
    /// Emit events strictly in record order, as fast as the source yields
    /// them — inter-event wall time collapses to zero. The default: it is the
    /// existing replay behaviour and what verify / diff / fast-forward want.
    #[default]
    CausalOrder,
    /// Emit events honouring their recorded inter-event deltas: before each
    /// event after the first, sleep `ts_ms[i] - ts_ms[i-1]` (saturating) so
    /// the replay reproduces the original cadence.
    TimingIntact,
}

impl ReplayMode {
    /// Whether this mode sleeps between events.
    #[must_use]
    pub fn honors_timing(self) -> bool {
        matches!(self, ReplayMode::TimingIntact)
    }
}

// ── main struct ──────────────────────────────────────────────────────────────

/// Replay a [`RecordingSource`] under a chosen [`ReplayMode`]. Generic over
/// the injected [`Clock`]; the default type parameter keeps production at
/// [`TimeClock`]. [`TimedReplay::events`] yields the same events the
/// underlying source yields, in the same order — under `TimingIntact` it sleeps
/// the recorded inter-event delta before each event after the first.
pub struct TimedReplay<Clk = TimeClock> {
    source: DynRecordingSource,
    mode: ReplayMode,
    clock: Clk,
}

impl TimedReplay<TimeClock> {
    /// Build a replay over the production [`TimeClock`]. The fluent twin of
    /// the [`ReplayConfig`] surface.
    #[must_use]
    pub fn new(source: DynRecordingSource, mode: ReplayMode) -> Self {
        Self::with_clock(source, mode, TimeClock)
    }
}

impl<Clk> TimedReplay<Clk> {
    /// Build a replay over an explicit [`Clock`] — the seam tests use to
    /// inject a deterministic mock clock.
    #[must_use]
    pub fn with_clock(source: DynRecordingSource, mode: ReplayMode, clock: Clk) -> Self {
        Self {
            source,
            mode,
            clock,
        }
    }

    /// Switch the replay mode fluently.
    #[must_use]
    pub fn mode(mut self, mode: ReplayMode) -> Self {
        self.mode = mode;
        self
    }

    /// The configured replay mode.
    #[must_use]
    pub fn replay_mode(&self) -> ReplayMode {
        self.mode
    }

    /// Project the built replay back to its config (the inverse of
    /// [`ReplayConfig::into_replay`]). Powers the round-trip parity guarantee.
    #[must_use]
    pub fn to_config(&self) -> ReplayConfig {
        ReplayConfig { mode: self.mode }
    }
}

impl<Clk> TimedReplay<Clk>
where
    Clk: Clock + Clone + Send + Sync + 'static,
    Clk::Delay: Send,
{
    /// The paced event stream. In [`ReplayMode::CausalOrder`] it is the
    /// underlying source's stream unchanged. In [`ReplayMode::TimingIntact`]
    /// each event after the first is preceded by a sleep of the recorded
    /// inter-event delta `ts_ms[i] - ts_ms[i-1]` (saturating).
    #[must_use]
    pub fn events<'replay>(&'replay self) -> RecordingEventStream<'replay> {
        let inner = self.source.events();
        if !self.mode.honors_timing() {
            return inner;
        }
        let clock = self.clock.clone();
        let stream = try_stream! {
            let mut inner = inner;
            let mut previous_ts: Option<u64> = None;
            // futures::StreamExt::next would be cleaner, but the trait is not in
            // scope of the macro body; poll the stream via the explicit helper.
            while let Some(item) = next_event(&mut inner).await {
                let event = item?;
                if let Some(prior) = previous_ts {
                    let delta = event.ts_ms().saturating_sub(prior);
                    if delta > 0 {
                        clock.delay(Duration::from_millis(delta)).await;
                    }
                }
                previous_ts = Some(event.ts_ms());
                yield event;
            }
        };
        Box::pin(stream)
    }
}

// pull one item from the boxed stream without pulling StreamExt into the
// try_stream! macro's expansion scope.
async fn next_event(
    stream: &mut RecordingEventStream<'_>,
) -> Option<Result<crate::event::RecordingEvent, ProximaError>> {
    use futures::stream::StreamExt;
    stream.next().await
}

// ── serde config + bidirectional twin ────────────────────────────────────────
//
// `ReplayConfig` is the serde/conflaguration surface; it is the bidirectional
// twin of the fluent builder (`TimedReplay::new` / `.mode`). `to_config` /
// `into_replay` round-trip a built replay through the config and back.

/// Serde/conflaguration config for a [`TimedReplay`]. Bidirectional with the
/// fluent builder via [`ReplayConfig::into_replay`] / [`TimedReplay::to_config`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayConfig {
    #[serde(default)]
    pub mode: ReplayMode,
}

impl ReplayConfig {
    /// Build a replay over the production [`TimeClock`] from this config.
    #[must_use]
    pub fn into_replay(self, source: DynRecordingSource) -> TimedReplay<TimeClock> {
        TimedReplay::new(source, self.mode)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::event::{FrameMetadata, HttpEvent, InteractionId, ProtocolEvent, RecordingEvent};
    use crate::{BinFormat, BinSource};
    use bytes::Bytes;
    use core::future::Future;
    use futures::stream::StreamExt;
    use futures::task::noop_waker;
    use prime::os::runtime::PrimeRuntime;
    use proxima_primitives::pipe::SendPipe;
    use proxima_runtime::Runtime;
    use proxima_core::time::drivers::mock::MockDriver;
    use proxima_core::time::{Driver, Instant};
    use std::pin::Pin;
    use std::task::{Context, Poll};

    use crate::source::{RecordingEventStream, RecordingSource};

    use crate::pipe::log_pipe::AppendLog;

    // an in-memory source whose stream is immediately ready (no offloaded
    // I/O). The events it replays are REAL recorded data — read back off a real
    // bin recording first — so the per-poll timing test isolates the clock
    // seam as the ONLY source of Pending, never the file read.
    struct InMemorySource {
        events: Vec<RecordingEvent>,
    }

    impl RecordingSource for InMemorySource {
        fn events<'lifetime>(&'lifetime self) -> RecordingEventStream<'lifetime> {
            let events = self.events.clone();
            Box::pin(futures::stream::iter(events.into_iter().map(Ok)))
        }
    }

    // read every event back off a real bin recording, driven by the runtime.
    fn read_back(path: &std::path::Path, runtime: &Arc<dyn Runtime>) -> Vec<RecordingEvent> {
        let source = BinSource::new(path, Arc::clone(runtime));
        futures::executor::block_on(async {
            source.events().map(|item| item.unwrap()).collect().await
        })
    }

    // ── the mock clock seam — proxima-time's deterministic MockDriver ─────────
    //
    // Wraps a directly-constructed `MockDriver` (proxima-time's own
    // deterministic driver). `delay` registers the waker via the driver's real
    // `schedule_wake`; the test fires it by calling `advance()`. This is the
    // SAME non-polling registration path the global driver uses — never a real
    // wait. We cannot bind the global `BOUND_DRIVER` to the mock from this
    // crate's test build, so we inject the mock through the `Clock` seam.

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

    fn prime() -> Arc<dyn Runtime> {
        Arc::new(PrimeRuntime::new(1).expect("prime"))
    }

    fn chunk(id: InteractionId, ts_ms: u64, data: &'static [u8]) -> RecordingEvent {
        RecordingEvent {
            id,
            ts_ms,
            parent: None,
            event: ProtocolEvent::Http(HttpEvent::ResponseChunk {
                data: Bytes::from_static(data),
                metadata: FrameMetadata::new(),
            }),
        }
    }

    // record three events with KNOWN ts_ms deltas (10ms, then 25ms) to a real
    // bin recording, returning the recording path + the runtime that reads it.
    fn record_three(path: &std::path::Path, runtime: &Arc<dyn Runtime>) -> Vec<RecordingEvent> {
        let id = InteractionId::new();
        let events = vec![
            chunk(id, 100, b"first"),
            chunk(id, 110, b"second"),
            chunk(id, 135, b"third"),
        ];
        futures::executor::block_on(async {
            let writer = AppendLog::open(
                path,
                Box::new(BinFormat::new().unwrap()),
                Arc::clone(runtime),
            )
            .unwrap();
            writer.call(events.clone()).await.unwrap();
            writer.flush().await.unwrap();
        });
        events
    }

    // ── causal-order: yields every recorded event in order, no waiting ────────

    #[test]
    fn causal_order_yields_real_recorded_events_in_order_without_waiting() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rec.bin");
        let runtime = prime();
        let written = record_three(&path, &runtime);

        let source: DynRecordingSource = Arc::new(BinSource::new(&path, Arc::clone(&runtime)));
        let replay = TimedReplay::with_clock(source, ReplayMode::CausalOrder, MockClock::new());

        // no clock advance — causal order must release every event regardless.
        let replayed: Vec<RecordingEvent> = futures::executor::block_on(async {
            replay.events().map(|item| item.unwrap()).collect().await
        });
        assert_eq!(
            replayed, written,
            "causal order replays the full recording in order"
        );
    }

    // ── timing-intact: each event releases only after its recorded delta ──────
    //
    // The recording has ts_ms 100, 110, 135 -> deltas 10ms then 25ms. Under
    // TimingIntact the stream parks before yielding event N until the mock
    // clock advances by that event's recorded delta. Driven entirely by hand:
    // assert pending-before-advance, ready-after. Zero real waiting.

    #[test]
    fn timing_intact_releases_each_event_at_its_recorded_offset() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rec.bin");
        let runtime = prime();
        let written = record_three(&path, &runtime);
        // real recorded events, read back off disk, then replayed from memory so
        // the ONLY thing that can park the stream is the inter-event clock.
        let recorded = read_back(&path, &runtime);
        assert_eq!(
            recorded, written,
            "read-back yields the real recorded events"
        );

        let source: DynRecordingSource = Arc::new(InMemorySource { events: recorded });
        let clock = MockClock::new();
        let replay = TimedReplay::with_clock(source, ReplayMode::TimingIntact, clock.clone());

        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);
        let mut stream = replay.events();

        // first event has no predecessor -> released on first poll, no sleep.
        let first = match stream.as_mut().poll_next(&mut context) {
            Poll::Ready(Some(item)) => item.unwrap(),
            other => panic!("first event must release immediately: {other:?}"),
        };
        assert_eq!(
            first, written[0],
            "first recorded event released without a delay"
        );

        // second event: 10ms recorded delta -> parked until the clock crosses it.
        assert!(
            matches!(stream.as_mut().poll_next(&mut context), Poll::Pending),
            "second event parks before its 10ms delta elapses"
        );
        clock.advance(Duration::from_millis(9));
        assert!(
            matches!(stream.as_mut().poll_next(&mut context), Poll::Pending),
            "still parked one ms short of the recorded delta"
        );
        clock.advance(Duration::from_millis(1));
        let second = match stream.as_mut().poll_next(&mut context) {
            Poll::Ready(Some(item)) => item.unwrap(),
            other => panic!("second event must release once its delta elapses: {other:?}"),
        };
        assert_eq!(
            second, written[1],
            "second event released at its recorded +10ms offset"
        );

        // third event: 25ms recorded delta from the second.
        assert!(
            matches!(stream.as_mut().poll_next(&mut context), Poll::Pending),
            "third event parks before its 25ms delta elapses"
        );
        clock.advance(Duration::from_millis(25));
        let third = match stream.as_mut().poll_next(&mut context) {
            Poll::Ready(Some(item)) => item.unwrap(),
            other => panic!("third event must release once its delta elapses: {other:?}"),
        };
        assert_eq!(
            third, written[2],
            "third event released at its recorded +25ms offset"
        );

        // stream is exhausted.
        assert!(
            matches!(stream.as_mut().poll_next(&mut context), Poll::Ready(None)),
            "stream completes after the last recorded event"
        );
    }

    // ── config <-> builder round-trip parity (principle 4) ────────────────────

    #[test]
    fn config_builder_round_trip_parity() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rec.bin");
        let runtime = prime();
        record_three(&path, &runtime);

        let config = ReplayConfig {
            mode: ReplayMode::TimingIntact,
        };
        let source: DynRecordingSource = Arc::new(BinSource::new(&path, runtime));

        // config -> builder -> config, and config -> json -> config.
        let replay = config.into_replay(source);
        let back = replay.to_config();
        let json = serde_json::to_value(config).expect("serialize");
        let parsed: ReplayConfig = serde_json::from_value(json.clone()).expect("deserialize");

        assert_eq!(
            back, config,
            "builder projects back to the originating config"
        );
        assert_eq!(parsed, config, "serde round-trip is lossless");
        assert_eq!(
            json,
            serde_json::json!({ "mode": { "kind": "timing_intact" } }),
            "ReplayMode serializes as a tagged object"
        );
    }

    #[test]
    fn default_mode_is_causal_order() {
        assert_eq!(ReplayMode::default(), ReplayMode::CausalOrder);
        assert!(!ReplayMode::CausalOrder.honors_timing());
        assert!(ReplayMode::TimingIntact.honors_timing());
        assert_eq!(ReplayConfig::default().mode, ReplayMode::CausalOrder);
    }

    // ── fluent mode switch reaches the same config as the constructor ─────────

    #[test]
    fn fluent_mode_switch_matches_constructor() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rec.bin");
        let runtime = prime();
        record_three(&path, &runtime);

        let source: DynRecordingSource = Arc::new(BinSource::new(&path, runtime));
        let replay =
            TimedReplay::new(source, ReplayMode::CausalOrder).mode(ReplayMode::TimingIntact);
        assert_eq!(replay.replay_mode(), ReplayMode::TimingIntact);
    }
}
