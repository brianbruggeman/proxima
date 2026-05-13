//! `LazyFanOut` — the durable recording terminal with a spigot.
//!
//! A recording sink should not pump unless it has somewhere to go. The pipe
//! graph is built at config-load time, but the serve-runtime (the off-core
//! blocking-I/O backend every durable write rides — `rt_fs::offload`) is not
//! bound until serve. So the durable terminal is built *disarmed*: it holds
//! its destinations + a shared [`DeferredRuntime`] spigot that is empty until
//! the App turns it on once, at serve (`spigot.set(runtime)`).
//!
//! While disarmed (spigot empty, or zero sinks because recording is disabled),
//! `call` opens no file and pumps nothing, and [`LazyFanOut::is_armed`] lets a
//! producer skip building events at all — no serialization for a stream with
//! no sink. Once armed, the first `call` opens every destination (memoized)
//! and fans the batch out exactly like an eager [`FanOut`].

use core::future::Future;
use std::sync::{Arc, Mutex, OnceLock};

use crate::event::RecordingEvent;
use proxima_primitives::pipe::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_runtime::Runtime;

use crate::pipe::dest::SinkSpec;
use crate::pipe::fanout::FanOut;
use crate::pipe::log_pipe::AppendLog;

/// The spigot: a runtime shared across every durable terminal in one App,
/// empty until serve turns it on. `OnceLock` so the arm is set-once and every
/// holder observes it without a lock on the read path.
pub type DeferredRuntime = Arc<OnceLock<Arc<dyn Runtime>>>;

/// A fresh, un-turned spigot. The App holds one, threads clones into every
/// recording factory, and `set`s it at serve.
#[must_use]
pub fn deferred_runtime() -> DeferredRuntime {
    Arc::new(OnceLock::new())
}

/// Durable fan-out that opens lazily once its spigot is armed.
pub struct LazyFanOut {
    sinks: Vec<SinkSpec>,
    spigot: DeferredRuntime,
    opened: Mutex<Option<FanOut>>,
}

impl LazyFanOut {
    /// Build disarmed: destinations recorded, spigot shared, nothing opened.
    #[must_use]
    pub fn new(sinks: Vec<SinkSpec>, spigot: DeferredRuntime) -> Self {
        Self {
            sinks,
            spigot,
            opened: Mutex::new(None),
        }
    }

    /// Whether the terminal has somewhere to pump: spigot turned on AND at
    /// least one destination. Producers gate event construction on this.
    #[must_use]
    pub fn is_armed(&self) -> bool {
        self.spigot.get().is_some() && !self.sinks.is_empty()
    }

    /// Open (once) and return a cheap handle to fan out on, or `None` while
    /// disarmed. Never holds the memo lock across the caller's write `.await`.
    fn armed_fanout(&self) -> Result<Option<FanOut>, ProximaError> {
        let Some(runtime) = self.spigot.get() else {
            return Ok(None);
        };
        if self.sinks.is_empty() {
            return Ok(None);
        }
        let mut guard = self
            .opened
            .lock()
            .map_err(|err| ProximaError::Record(format!("lazy fanout poisoned: {err}")))?;
        if guard.is_none() {
            let mut logs = Vec::with_capacity(self.sinks.len());
            for sink in &self.sinks {
                logs.push(AppendLog::open(
                    &sink.path,
                    sink.codec()?,
                    Arc::clone(runtime),
                )?);
            }
            *guard = Some(FanOut::new(logs));
        }
        Ok(guard.as_ref().map(FanOut::clone))
    }

    /// Flush every destination — no-op while disarmed/unopened.
    pub async fn flush(&self) -> Result<(), ProximaError> {
        if let Some(fanout) = self.armed_fanout()? {
            fanout.flush().await
        } else {
            Ok(())
        }
    }

    /// Fsync every destination — no-op while disarmed/unopened.
    pub async fn sync(&self) -> Result<(), ProximaError> {
        if let Some(fanout) = self.armed_fanout()? {
            fanout.sync().await
        } else {
            Ok(())
        }
    }
}

impl SendPipe for LazyFanOut {
    type In = Vec<RecordingEvent>;
    type Out = ();
    type Err = ProximaError;

    fn call(
        &self,
        events: Vec<RecordingEvent>,
    ) -> impl Future<Output = Result<(), ProximaError>> + Send {
        let opened = self.armed_fanout();
        async move {
            match opened? {
                Some(fanout) => fanout.call(events).await,
                None => Ok(()),
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    use crate::event::{FrameMetadata, HttpEvent, InteractionId, ProtocolEvent};
    use crate::{BinFormat, RecordingEvent};
    use bytes::Bytes;
    use prime::os::runtime::PrimeRuntime;

    use crate::pipe::dest::{FormatKind, SinkSpec};
    use crate::pipe::log_pipe::ReplayLog;
    use crate::pipe::log_pipe::test_support::drain;

    fn event() -> RecordingEvent {
        RecordingEvent {
            id: InteractionId::new(),
            ts_ms: 7,
            parent: None,
            event: ProtocolEvent::Http(HttpEvent::ResponseChunk {
                data: Bytes::from_static(b"data: hello\n\n"),
                metadata: FrameMetadata::new(),
            }),
        }
    }

    // spigot off: call pumps nothing, opens no file, reports disarmed.
    #[test]
    fn disarmed_pumps_nothing_and_creates_no_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rec.bin");
        let lazy = LazyFanOut::new(
            vec![SinkSpec::new(path.to_str().unwrap(), FormatKind::Bin)],
            deferred_runtime(),
        );
        assert!(!lazy.is_armed(), "no spigot -> disarmed");
        futures::executor::block_on(async {
            lazy.call(vec![event()]).await.unwrap();
            lazy.flush().await.unwrap();
        });
        assert!(!path.exists(), "disarmed terminal opens no file");
    }

    // zero sinks stays disarmed even with the spigot on (recording disabled).
    #[test]
    fn armed_spigot_with_no_sinks_is_still_disarmed() {
        let spigot = deferred_runtime();
        spigot
            .set(Arc::new(PrimeRuntime::new(1).expect("prime")) as Arc<dyn Runtime>)
            .ok();
        let lazy = LazyFanOut::new(Vec::new(), spigot);
        assert!(!lazy.is_armed(), "no destinations -> nowhere to pump");
        futures::executor::block_on(lazy.call(vec![event()])).unwrap();
    }

    // turn the spigot on, then the same terminal opens + persists on first call.
    #[test]
    fn arming_the_spigot_opens_and_persists() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rec.bin");
        let spigot = deferred_runtime();
        let lazy = LazyFanOut::new(
            vec![SinkSpec::new(path.to_str().unwrap(), FormatKind::Bin)],
            Arc::clone(&spigot),
        );
        let events = vec![event(), event()];

        let runtime: Arc<dyn Runtime> = Arc::new(PrimeRuntime::new(1).expect("prime"));
        let replayed = futures::executor::block_on(async {
            assert!(!lazy.is_armed());
            spigot.set(Arc::clone(&runtime)).ok();
            assert!(lazy.is_armed(), "spigot on + a sink -> armed");
            lazy.call(events.clone()).await.unwrap();
            lazy.flush().await.unwrap();
            drain(&ReplayLog::open(&path, lazy_codec(), Arc::clone(&runtime)).unwrap()).await
        });
        assert_eq!(replayed, events, "armed terminal persists the batch");
    }

    fn lazy_codec() -> Box<dyn crate::Format> {
        Box::new(BinFormat::new().unwrap())
    }
}
