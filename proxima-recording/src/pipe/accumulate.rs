//! `AccumulatingSink` — per-event front for the batch-shaped durable terminal.
//!
//! The durable terminal ([`LazyFanOut`]) writes ONE codec block per `call` (a
//! `Vec<RecordingEvent>`), so the block compressor (`BinFormat`'s zstd) earns
//! its ratio per batch. But some producers append ONE event at a time (the
//! `record` drainer, the process bridge's per-line stdout/stderr). A naive
//! per-event bridge would call `call(vec![event])` and write a zstd block per
//! event — defeating the compression. This sink coalesces per-event appends
//! into batches and flushes a block when the buffer reaches `batch_size`;
//! `flush`/`sync` drain the remainder at an interaction / time boundary.
//!
//! Like the terminal it fronts, it respects the spigot: while the durable is
//! disarmed (no runtime, or no destination) `call` buffers nothing — an event
//! with nowhere to go is dropped, not held.

use core::future::Future;
use std::sync::Arc;

use crate::event::RecordingEvent;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::{Batch, ProximaError};

use crate::pipe::lazy::LazyFanOut;

/// Default coalescing window in events: a typical chunked-streaming turn is
/// ~40 events, so one batch ≈ a couple of turns — enough for the block
/// compressor without holding events long past their interaction.
pub const DEFAULT_BATCH_EVENTS: usize = 64;

/// Per-event durable sink: buffers appends, flushes batches to [`LazyFanOut`].
///
/// The coalescing buffer is a generic [`proxima_primitives::pipe::Batch`] — this sink owns
/// only the spigot gate and the durable terminal.
pub struct AccumulatingSink {
    durable: Arc<LazyFanOut>,
    batch: Batch<RecordingEvent>,
}

impl AccumulatingSink {
    /// Front a durable terminal, coalescing into `batch_size`-event blocks.
    #[must_use]
    pub fn new(durable: Arc<LazyFanOut>, batch_size: usize) -> Self {
        Self {
            durable,
            batch: Batch::new(batch_size),
        }
    }

    /// Front a durable terminal with the default batch window.
    #[must_use]
    pub fn with_defaults(durable: Arc<LazyFanOut>) -> Self {
        Self::new(durable, DEFAULT_BATCH_EVENTS)
    }

    /// Whether the underlying terminal has somewhere to pump.
    #[must_use]
    pub fn is_armed(&self) -> bool {
        self.durable.is_armed()
    }

    /// Persist any buffered events, then flush the durable terminal.
    pub async fn flush(&self) -> Result<(), ProximaError> {
        let batch = self.batch.drain();
        if !batch.is_empty() {
            self.durable.call(batch).await?;
        }
        self.durable.flush().await
    }

    /// Persist any buffered events, then fsync the durable terminal.
    pub async fn sync(&self) -> Result<(), ProximaError> {
        let batch = self.batch.drain();
        if !batch.is_empty() {
            self.durable.call(batch).await?;
        }
        self.durable.sync().await
    }
}

impl SendPipe for AccumulatingSink {
    type In = RecordingEvent;
    type Out = ();
    type Err = ProximaError;

    fn call(&self, event: RecordingEvent) -> impl Future<Output = Result<(), ProximaError>> + Send {
        // synchronous: gate on the spigot, then buffer (just a lock, no await).
        // a disarmed durable has nowhere to pump, so the event is dropped, not
        // held; only a now-full batch needs the sink await.
        let full = if self.durable.is_armed() {
            self.batch.push(event)
        } else {
            None
        };
        let durable = Arc::clone(&self.durable);
        async move {
            match full {
                Some(batch) => durable.call(batch).await,
                None => Ok(()),
            }
        }
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
    use crate::pipe::lazy::deferred_runtime;
    use crate::pipe::log_pipe::ReplayLog;
    use crate::pipe::log_pipe::test_support::drain;

    fn event(id: InteractionId) -> RecordingEvent {
        RecordingEvent {
            id,
            ts_ms: 7,
            parent: None,
            event: ProtocolEvent::Http(HttpEvent::ResponseChunk {
                data: Bytes::from_static(b"data: hello\n\n"),
                metadata: FrameMetadata::new(),
            }),
        }
    }

    // per-event appends coalesce into batches; every event is durable after
    // flush, and replay reassembles them in order.
    #[test]
    fn per_event_appends_coalesce_and_replay_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rec.bin");
        let runtime: StdArc<dyn Runtime> = StdArc::new(PrimeRuntime::new(1).expect("prime"));
        let id = InteractionId::new();
        let events: Vec<RecordingEvent> = (0..10).map(|_| event(id)).collect();

        let spigot = deferred_runtime();
        let durable = StdArc::new(LazyFanOut::new(
            vec![SinkSpec::new(path.to_str().unwrap(), FormatKind::Bin)],
            StdArc::clone(&spigot),
        ));

        let replayed = futures::executor::block_on(async {
            spigot.set(StdArc::clone(&runtime)).ok();
            // batch_size 4: 10 events -> two full blocks flushed mid-stream,
            // two left for the explicit flush.
            let sink = AccumulatingSink::new(StdArc::clone(&durable), 4);
            for ev in &events {
                sink.call(ev.clone()).await.unwrap();
            }
            sink.flush().await.unwrap();
            drain(&ReplayLog::open(&path, Box::new(BinFormat::new().unwrap()), runtime).unwrap())
                .await
        });
        assert_eq!(replayed, events, "all per-event appends durable + ordered");
    }

    // disarmed: per-event calls drop, no file is created, no buffering grows.
    #[test]
    fn disarmed_drops_without_buffering() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rec.bin");
        let durable = StdArc::new(LazyFanOut::new(
            vec![SinkSpec::new(path.to_str().unwrap(), FormatKind::Bin)],
            deferred_runtime(),
        ));
        let sink = AccumulatingSink::new(durable, 4);
        assert!(!sink.is_armed());
        futures::executor::block_on(async {
            for _ in 0..100 {
                sink.call(event(InteractionId::new())).await.unwrap();
            }
            sink.flush().await.unwrap();
        });
        assert!(!path.exists(), "disarmed sink writes nothing");
    }
}
