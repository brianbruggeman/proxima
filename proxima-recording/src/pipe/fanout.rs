//! `FanOut` — broadcast one event stream to N durable terminal Pipes, each
//! with its own format + destination.
//!
//! This is the "with pipes we fan out to multiple durables, each its own
//! formatting" shape: a composition over `Vec<AppendLog>`. Each sink is an
//! `AppendLog` carrying its own `Box<dyn Format>`, so the format polymorphism
//! lives inside the sink — the fan-out needs no dyn-pipe erasure. `call(events)`
//! appends the batch to every sink in order; a failure in any sink fails the
//! call (durable fan-out is all-or-nothing, not best-effort). Durability
//! control (`flush`/`sync`) drives every sink, mirroring `AppendLog`.

use core::future::Future;

use crate::event::RecordingEvent;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::{AllOrNothing, FanOut as PipeFanOut, ProximaError};

use crate::pipe::log_pipe::AppendLog;

/// Broadcast composition over N durable sink Pipes.
///
/// A thin durability layer over the generic [`proxima_primitives::pipe::FanOut`] (1→N tee,
/// move-into-last / clone-into-earlier) specialised to `AppendLog` sinks under
/// [`AllOrNothing`] — a sink error fails the call, since a durable fan-out is
/// all-or-nothing, not best-effort. This newtype adds only the `flush`/`sync`
/// durability control the generic combinator (which knows only `SendPipe`)
/// cannot express. `Clone` is a refcount bump on the shared sinks.
#[derive(Clone)]
pub struct FanOut {
    inner: PipeFanOut<AppendLog, AllOrNothing>,
}

impl FanOut {
    #[must_use]
    pub fn new(sinks: Vec<AppendLog>) -> Self {
        Self {
            inner: PipeFanOut::new(sinks),
        }
    }

    #[must_use]
    pub fn sink_count(&self) -> usize {
        self.inner.sink_count()
    }

    /// Flush every sink. Returns once all sinks have flushed.
    pub async fn flush(&self) -> Result<(), ProximaError> {
        for sink in self.inner.sinks() {
            sink.flush().await?;
        }
        Ok(())
    }

    /// Fsync every sink — the batch is on stable storage on every durable.
    pub async fn sync(&self) -> Result<(), ProximaError> {
        for sink in self.inner.sinks() {
            sink.sync().await?;
        }
        Ok(())
    }
}

impl SendPipe for FanOut {
    type In = Vec<RecordingEvent>;
    type Out = ();
    type Err = ProximaError;

    fn call(
        &self,
        events: Vec<RecordingEvent>,
    ) -> impl Future<Output = Result<(), ProximaError>> + Send {
        self.inner.call(events)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use crate::{BinFormat, JsonFormat};
    use proxima_runtime::Runtime;

    use crate::pipe::log_pipe::ReplayLog;
    use crate::pipe::log_pipe::test_support::{drain, sample_events};

    // one event stream fanned out to TWO durables — bin at one path, json at
    // another — and BOTH replay the full stream. "multiple durables, each its
    // own formatting" proven, runtime-agnostic.
    #[test]
    fn fans_out_to_bin_and_json_each_replays_full_stream() {
        let dir = tempfile::tempdir().unwrap();
        let bin_path = dir.path().join("audit.bin");
        let json_path = dir.path().join("audit.jsonl");
        let runtime: Arc<dyn Runtime> =
            Arc::new(prime::os::runtime::PrimeRuntime::new(1).expect("prime"));
        let written = sample_events();

        let replayed_bin;
        let replayed_json;
        {
            let bin_sink = AppendLog::open(
                &bin_path,
                Box::new(BinFormat::new().unwrap()),
                Arc::clone(&runtime),
            )
            .unwrap();
            let json_sink = AppendLog::open(
                &json_path,
                Box::new(JsonFormat::new()),
                Arc::clone(&runtime),
            )
            .unwrap();
            let recorder = FanOut::new(vec![bin_sink, json_sink]);
            assert_eq!(recorder.sink_count(), 2);

            replayed_bin = futures::executor::block_on(async {
                recorder.call(written.clone()).await.unwrap();
                recorder.flush().await.unwrap();
                drain(
                    &ReplayLog::open(
                        &bin_path,
                        Box::new(BinFormat::new().unwrap()),
                        Arc::clone(&runtime),
                    )
                    .unwrap(),
                )
                .await
            });
            replayed_json = futures::executor::block_on(drain(
                &ReplayLog::open(
                    &json_path,
                    Box::new(JsonFormat::new()),
                    Arc::clone(&runtime),
                )
                .unwrap(),
            ));
        }
        assert_eq!(replayed_bin, written, "bin durable replays the full stream");
        assert_eq!(
            replayed_json, written,
            "json durable replays the full stream"
        );
    }
}
