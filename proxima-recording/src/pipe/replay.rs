//! `TimedReplay` — replay a recording in one of two timing modes.
//!
//! The durable read path ([`crate::pipe::log_pipe::ReplayLog`]) and every
//! [`RecordingSource`](crate::source::RecordingSource) yield events strictly IN RECORD ORDER, as fast as the
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
//! - [`TimeClock`] — the
//!   production `Clock`, and the default `Clk` type parameter, so every
//!   existing caller of `TimedReplay::new` is unaffected.
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

// ── replay mode — the whole config surface, serde-expressible on its own ─────

/// How a [`TimedReplay`] paces the events it yields. `Serialize`/`Deserialize`
/// so an embedding config can carry the mode as a plain field; there is no
/// separate config struct because a one-field wrapper would not let a caller
/// do anything this enum plus [`TimedReplay::new`] does not already do.
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

/// Replay a [`RecordingSource`](crate::source::RecordingSource) under a chosen [`ReplayMode`]. Generic over
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
    /// Build a replay over the production [`TimeClock`].
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

    /// The configured replay mode — the projection back to config, and the
    /// other half of the round-trip parity guarantee.
    #[must_use]
    pub fn replay_mode(&self) -> ReplayMode {
        self.mode
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex, mpsc};
    use std::time::Instant;

    use crate::event::{FrameMetadata, HttpEvent, InteractionId, ProtocolEvent, RecordingEvent};
    use crate::{BinFormat, BinSource};
    use bytes::Bytes;
    use futures::stream::StreamExt;
    use futures::task::noop_waker;
    use prime::os::core_shard;
    use prime::os::runtime::PrimeRuntime;
    use proxima_clock::coarse::TickCell;
    use proxima_clock::ticks::Ticks;
    use proxima_primitives::pipe::SendPipe;
    use proxima_primitives::pipe::clock::testing::MockClock;
    use proxima_runtime::{CoreId, Runtime};
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

    // the mock clock seam: `proxima_primitives::pipe::clock::testing::MockClock`
    // (see that module for why it, not `RecordingClock` — this test needs
    // `delay` to genuinely pend until `advance()` crosses the deadline, the
    // SAME non-polling `schedule_wake` registration path the global driver
    // uses, never a real wait). We cannot bind the global `BOUND_DRIVER` to
    // the mock from this crate's test build, so we inject it through the
    // `Clock` seam instead.

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

    // same shape as `chunk`, but for a payload size only known at runtime
    // (varying per event in the multi-connection fixture below).
    fn chunk_owned(id: InteractionId, ts_ms: u64, data: Vec<u8>) -> RecordingEvent {
        RecordingEvent {
            id,
            ts_ms,
            parent: None,
            event: ProtocolEvent::Http(HttpEvent::ResponseChunk {
                data: Bytes::from(data),
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

    // several interleaved connections, non-uniform inter-arrival gaps (short
    // keepalive-style ticks mixed with multi-second bursts) and varying
    // payload sizes, spanning at least 32 recorded seconds — the fixture the
    // memory-speed replay test proves spacing preservation against. Built
    // through the same real AppendLog/BinFormat path as `record_three`, not
    // hand-written JSON.
    fn record_multi_connection_trace(
        path: &std::path::Path,
        runtime: &Arc<dyn Runtime>,
    ) -> Vec<RecordingEvent> {
        const CONNECTION_COUNT: usize = 4;
        const MINIMUM_SPAN_MILLIS: u64 = 32_000;
        const DELTA_PATTERN_MILLIS: [u64; 12] =
            [15, 750, 40, 3200, 90, 2500, 20, 4800, 300, 60, 1800, 10];
        const PAYLOAD_SIZE_PATTERN: [usize; 10] =
            [64, 512, 128, 4096, 256, 1024, 32, 2048, 96, 768];

        let connections: Vec<InteractionId> = (0..CONNECTION_COUNT)
            .map(|_| InteractionId::new())
            .collect();

        let mut events = Vec::new();
        let mut ts_ms = 0_u64;
        let mut step = 0_usize;
        loop {
            ts_ms += DELTA_PATTERN_MILLIS[step % DELTA_PATTERN_MILLIS.len()];
            let connection = connections[(step * 3 + 1) % CONNECTION_COUNT];
            let payload_size = PAYLOAD_SIZE_PATTERN[step % PAYLOAD_SIZE_PATTERN.len()];
            events.push(chunk_owned(connection, ts_ms, vec![0x5A; payload_size]));
            step += 1;
            if ts_ms >= MINIMUM_SPAN_MILLIS {
                break;
            }
        }

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

        let mode = ReplayMode::TimingIntact;
        let source: DynRecordingSource = Arc::new(BinSource::new(&path, runtime));

        // config -> builder -> config, and config -> json -> config.
        let replay = TimedReplay::new(source, mode);
        let json = serde_json::to_value(mode).expect("serialize");
        let parsed: ReplayMode = serde_json::from_value(json.clone()).expect("deserialize");

        assert_eq!(
            replay.replay_mode(),
            mode,
            "builder projects back to the originating config"
        );
        assert_eq!(parsed, mode, "serde round-trip is lossless");
        assert_eq!(
            json,
            serde_json::json!({ "kind": "timing_intact" }),
            "ReplayMode serializes as a tagged object"
        );
    }

    #[test]
    fn default_mode_is_causal_order() {
        assert_eq!(ReplayMode::default(), ReplayMode::CausalOrder);
        assert!(!ReplayMode::CausalOrder.honors_timing());
        assert!(ReplayMode::TimingIntact.honors_timing());
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

    // ── timing-intact at memory speed: virtual-clocked prime shard ───────────
    //
    // `TimedReplay<TimeClock>` is the production type — no bespoke virtual
    // clock capability. `TimeClock::delay` resolves through
    // `proxima_core::time::sleep`, which (this crate's dev-dependency on
    // `proxima-core` builds against the `time-driver-prime-wheel` external
    // driver) routes through the calling worker's own `TimerWheel`. Running
    // the replay as a task dispatched onto a `launch_with_virtual_clock`
    // shard means every `Clock::delay` call it makes resolves through that
    // SAME wheel, auto-advanced by prime's existing idle-park hook — no new
    // waker, no parallel timer mechanism.

    #[test]
    fn timing_intact_replays_a_thirty_second_trace_at_memory_speed_on_a_virtual_prime_shard() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi_connection.bin");
        let runtime = prime();
        let written = record_multi_connection_trace(&path, &runtime);
        let recorded_span_ms = written.last().unwrap().ts_ms - written.first().unwrap().ts_ms;
        assert!(
            recorded_span_ms >= 30_000,
            "fixture must span at least 30 recorded seconds, spans {recorded_span_ms}ms"
        );

        let recorded = read_back(&path, &runtime);
        assert_eq!(
            recorded, written,
            "read-back yields the real recorded events"
        );

        let source: DynRecordingSource = Arc::new(InMemorySource { events: recorded });
        let replay = TimedReplay::new(source, ReplayMode::TimingIntact);

        let cell = Arc::new(TickCell::new(Ticks::ZERO));
        let shard = core_shard::launch_with_virtual_clock(CoreId(9001), None, 2, 16, cell)
            .expect("launch virtual-clocked prime shard");

        let captured: Arc<Mutex<Vec<(u64, RecordingEvent)>>> = Arc::new(Mutex::new(Vec::new()));
        let captured_for_task = Arc::clone(&captured);
        let (done_sender, done_receiver) = mpsc::channel::<()>();

        shard
            .dispatch_factory(Box::new(move || {
                Box::pin(async move {
                    let mut events = replay.events();
                    while let Some(item) = events.next().await {
                        let event = item.expect("replayed event");
                        let tick = core_shard::current_tick();
                        captured_for_task
                            .lock()
                            .expect("captured mutex poisoned")
                            .push((tick, event));
                    }
                    let _ = done_sender.send(());
                }) as Pin<Box<dyn Future<Output = ()> + 'static>>
            }))
            .expect("dispatch replay factory");

        let started = Instant::now();
        done_receiver.recv_timeout(Duration::from_secs(2)).expect(
            "replay of a 32-simulated-second trace must complete via virtual auto-advance \
                 alone; a real wait would take over 30s and blow well past this 2s bound",
        );
        let elapsed = started.elapsed();

        shard
            .shutdown_and_join()
            .expect("shutdown virtual-clocked shard");

        let captured = captured.lock().expect("captured mutex poisoned").clone();
        let replayed_events: Vec<RecordingEvent> = captured
            .iter()
            .map(|(_tick, event)| event.clone())
            .collect();
        assert_eq!(
            replayed_events, written,
            "relative ordering of replayed events must match the recording exactly"
        );

        for index in 1..captured.len() {
            let (previous_tick, _) = captured[index - 1];
            let (current_tick, _) = captured[index];
            let recorded_delta = written[index]
                .ts_ms
                .saturating_sub(written[index - 1].ts_ms);
            let simulated_delta = current_tick.saturating_sub(previous_tick);
            assert_eq!(
                simulated_delta, recorded_delta,
                "event {index} must be replayed {recorded_delta} simulated ms after its \
                 predecessor, not collapsed to zero"
            );
        }

        assert!(
            elapsed < Duration::from_millis(500),
            "a {recorded_span_ms}ms recording must replay in well under 500ms of wall time, \
             took {elapsed:?}"
        );
    }
}
