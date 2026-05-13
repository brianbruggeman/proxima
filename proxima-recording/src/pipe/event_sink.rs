//! `RecordingSink` — the per-event durable sink trait, relocated onto the
//! spigot substrate, plus `EventTap` for live observation.
//!
//! RISC note (principle 1): this is NOT a new peer trait — it is the canonical
//! recording-sink abstraction *relocated* from `proxima-recording-core` (whose
//! `BinSink`/`JsonlSink`/`BroadcastSink`/registry are deleted) into the
//! pipe-flavored substrate, reshaped to the spigot model. The batch-coalescing
//! that `append_batch` used to do now lives in [`crate::pipe::AccumulatingSink`]; the
//! durable terminal is [`crate::pipe::LazyFanOut`]. Consumers that hold a
//! heterogeneous per-event sink (the pipeline executor, the control planes)
//! keep calling `.append(event)` / `.flush()` — only the import + construction
//! move to the spigot types.
//!
//! `EventTap` replaces `BroadcastSink`: it forwards every event to an inner
//! durable sink AND publishes it to a [`proxima_primitives::sync::broadcast`] channel for
//! live tailing. `proxima_primitives::sync::broadcast` is the workspace broadcast primitive
//! (RISC: reused instead of `tokio::sync::broadcast`); `subscribe()` hands a
//! late tailer its own receiver, lossy-on-lag exactly like the old sink.

use core::future::Future;
use core::pin::Pin;
use std::sync::Arc;

use crate::event::RecordingEvent;
use proxima_primitives::pipe::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::sync::broadcast;

use crate::pipe::accumulate::AccumulatingSink;

/// Boxed `Send` future for the object-safe sink methods. Mirrors the shape the
/// pipeline executor / control planes need to hold a `dyn` sink and `.await`
/// it across the per-event append loop.
pub type AppendFuture<'lifetime> =
    Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'lifetime>>;

/// Per-event durable recording sink. The heterogeneous-sink seam for consumers
/// (executor, control planes) that hold one sink and append events one at a
/// time; concrete impls ([`AccumulatingSink`], [`EventTap`], the control-plane
/// routing sinks) all ride the spigot, so a disarmed sink pumps nothing.
pub trait RecordingSink: Send + Sync {
    fn append<'lifetime>(&'lifetime self, event: RecordingEvent) -> AppendFuture<'lifetime>;

    fn flush<'lifetime>(&'lifetime self) -> AppendFuture<'lifetime>;
}

/// Shared, object-safe per-event sink handle.
pub type DynRecordingSink = Arc<dyn RecordingSink>;

impl RecordingSink for AccumulatingSink {
    fn append<'lifetime>(&'lifetime self, event: RecordingEvent) -> AppendFuture<'lifetime> {
        Box::pin(SendPipe::call(self, event))
    }

    fn flush<'lifetime>(&'lifetime self) -> AppendFuture<'lifetime> {
        Box::pin(AccumulatingSink::flush(self))
    }
}

/// Live-observation tee: forward to a durable inner sink + publish to a
/// broadcast for tailers. The durable record is authoritative; publishing is
/// best-effort (no subscribers, or a lagging one, never stalls the append).
pub struct EventTap {
    inner: DynRecordingSink,
    sender: broadcast::Sender<RecordingEvent>,
}

impl EventTap {
    /// Wrap a durable inner sink, buffering up to `channel_capacity` events
    /// per subscriber for live tailing.
    #[must_use]
    pub fn new(inner: DynRecordingSink, channel_capacity: usize) -> Self {
        let (sender, _initial) = broadcast::channel(channel_capacity);
        Self { inner, sender }
    }

    /// A fresh live-tail receiver. Late subscribers see only events appended
    /// after they subscribe; a slow one observes `RecvError::Lagged`.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<RecordingEvent> {
        self.sender.subscribe()
    }
}

impl RecordingSink for EventTap {
    fn append<'lifetime>(&'lifetime self, event: RecordingEvent) -> AppendFuture<'lifetime> {
        Box::pin(async move {
            // best-effort publish: Err == no active tailer, the steady state.
            let _ = self.sender.send(event.clone());
            self.inner.append(event).await
        })
    }

    fn flush<'lifetime>(&'lifetime self) -> AppendFuture<'lifetime> {
        self.inner.flush()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::sync::Arc as StdArc;

    use crate::event::{FrameMetadata, HttpEvent, InteractionId, ProtocolEvent};
    use crate::{BinFormat, RecordingEvent};
    use bytes::Bytes;
    use prime::os::runtime::PrimeRuntime;
    use proxima_runtime::Runtime;

    use crate::pipe::dest::{FormatKind, SinkSpec};
    use crate::pipe::lazy::{LazyFanOut, deferred_runtime};
    use crate::pipe::log_pipe::ReplayLog;
    use crate::pipe::log_pipe::test_support::drain;

    fn event(tag: u8) -> RecordingEvent {
        RecordingEvent {
            id: InteractionId::new(),
            ts_ms: u64::from(tag),
            parent: None,
            event: ProtocolEvent::Http(HttpEvent::ResponseChunk {
                data: Bytes::copy_from_slice(&[tag; 4]),
                metadata: FrameMetadata::new(),
            }),
        }
    }

    // a tap forwards every append to the durable AND a live subscriber sees
    // exactly the events appended after it subscribed, in order.
    #[test]
    fn tap_forwards_to_durable_and_tails_live() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rec.bin");
        let runtime: StdArc<dyn Runtime> = StdArc::new(PrimeRuntime::new(1).expect("prime"));
        let spigot = deferred_runtime();
        spigot.set(StdArc::clone(&runtime)).ok();
        let durable = StdArc::new(LazyFanOut::new(
            vec![SinkSpec::new(path.to_str().unwrap(), FormatKind::Bin)],
            spigot,
        ));
        let accumulate: DynRecordingSink = StdArc::new(AccumulatingSink::new(durable, 1));
        let tap = EventTap::new(accumulate, 16);

        let events = vec![event(1), event(2), event(3)];
        let (replayed, tailed) = futures::executor::block_on(async {
            let mut rx = tap.subscribe();
            for ev in &events {
                tap.append(ev.clone()).await.unwrap();
            }
            tap.flush().await.unwrap();
            let mut tailed = Vec::new();
            while let Ok(ev) = rx.recv().await {
                tailed.push(ev);
                if tailed.len() == events.len() {
                    break;
                }
            }
            let replayed = drain(
                &ReplayLog::open(&path, Box::new(BinFormat::new().unwrap()), runtime).unwrap(),
            )
            .await;
            (replayed, tailed)
        });
        assert_eq!(replayed, events, "durable got every event");
        assert_eq!(tailed, events, "live tail saw every event in order");
    }
}
