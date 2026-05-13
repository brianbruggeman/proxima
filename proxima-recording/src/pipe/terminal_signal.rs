//! `TerminalSignal` — a [`RecordingSink`] decorator that fires a
//! [`Signal`](proxima_core::signal::Signal) once a caller-chosen terminal
//! event has been durably flushed, so a caller `.await`s completion instead
//! of polling the cassette from outside.
//!
//! Composes exactly like [`EventTap`](crate::pipe::event_sink::EventTap): wraps
//! any [`DynRecordingSink`], forwards every `append`/`flush` unchanged, and
//! layers one observation on top. The terminal predicate stays
//! event-shape-agnostic (`Fn(&RecordingEvent) -> bool`), so this isn't
//! HTTP-specific — a websocket close frame or a process-exit event drives
//! the same fire.
//!
//! Firing happens in `flush`, not `append`: an appended event may only be
//! buffered ([`crate::pipe::AccumulatingSink`] coalesces into batches), so seeing
//! the terminal event is remembered but the signal only fires once a
//! subsequent flush has actually completed — durable, not merely queued.
//!
//! One-shot by design, matching [`Signal`](proxima_core::signal::Signal)
//! itself: built for a single wait window (one recorded interaction), not a
//! long-lived sink that repeatedly quiesces and un-quiesces across many
//! interactions over its lifetime.

use core::sync::atomic::{AtomicBool, Ordering};

use crate::event::RecordingEvent;
use proxima_core::signal::{Fired, Signal};

use crate::pipe::event_sink::{AppendFuture, DynRecordingSink, RecordingSink};

pub struct TerminalSignal<Predicate> {
    inner: DynRecordingSink,
    is_terminal: Predicate,
    seen_terminal: AtomicBool,
    signal: Signal,
}

impl<Predicate> TerminalSignal<Predicate>
where
    Predicate: Fn(&RecordingEvent) -> bool + Send + Sync + 'static,
{
    /// Wrap a durable inner sink; `is_terminal` names the event that marks
    /// the interaction complete (e.g. an HTTP `Ended`).
    #[must_use]
    pub fn new(inner: DynRecordingSink, is_terminal: Predicate) -> Self {
        Self {
            inner,
            is_terminal,
            seen_terminal: AtomicBool::new(false),
            signal: Signal::new(),
        }
    }

    /// Resolves once the terminal event has been appended AND a subsequent
    /// flush has completed. Sticky: a late caller (awaiting after the fire)
    /// resolves on its first poll, no re-wait.
    #[must_use]
    pub fn drained(&self) -> Fired {
        self.signal.fired()
    }
}

impl<Predicate> RecordingSink for TerminalSignal<Predicate>
where
    Predicate: Fn(&RecordingEvent) -> bool + Send + Sync + 'static,
{
    fn append<'lifetime>(&'lifetime self, event: RecordingEvent) -> AppendFuture<'lifetime> {
        let terminal = (self.is_terminal)(&event);
        Box::pin(async move {
            let result = self.inner.append(event).await;
            if result.is_ok() && terminal {
                self.seen_terminal.store(true, Ordering::Release);
            }
            result
        })
    }

    fn flush<'lifetime>(&'lifetime self) -> AppendFuture<'lifetime> {
        Box::pin(async move {
            let result = self.inner.flush().await;
            if result.is_ok() && self.seen_terminal.load(Ordering::Acquire) {
                self.signal.fire();
            }
            result
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use core::future::Future;
    use core::pin::Pin;
    use core::task::{Context, Poll, Waker};
    use std::sync::Arc;
    use std::sync::Mutex as StdMutex;

    use crate::event::{HttpEvent, InteractionId, ProtocolEvent, RecordMeta};

    #[derive(Default)]
    struct MemorySink {
        events: StdMutex<Vec<RecordingEvent>>,
    }

    impl RecordingSink for MemorySink {
        fn append<'lifetime>(&'lifetime self, event: RecordingEvent) -> AppendFuture<'lifetime> {
            Box::pin(async move {
                self.events.lock().expect("memory sink lock").push(event);
                Ok(())
            })
        }

        fn flush<'lifetime>(&'lifetime self) -> AppendFuture<'lifetime> {
            Box::pin(async { Ok(()) })
        }
    }

    fn request_ended(seq: u64) -> RecordingEvent {
        RecordingEvent {
            id: InteractionId::from_bytes([(seq % 256) as u8; 16]),
            ts_ms: seq,
            parent: None,
            event: ProtocolEvent::Http(HttpEvent::RequestEnded),
        }
    }

    fn response_ended(seq: u64) -> RecordingEvent {
        RecordingEvent {
            id: InteractionId::from_bytes([(seq % 256) as u8; 16]),
            ts_ms: seq,
            parent: None,
            event: ProtocolEvent::Http(HttpEvent::Ended {
                latency_ms: seq,
                meta: RecordMeta::default(),
            }),
        }
    }

    fn is_response_ended(event: &RecordingEvent) -> bool {
        matches!(event.event, ProtocolEvent::Http(HttpEvent::Ended { .. }))
    }

    fn noop_context() -> Context<'static> {
        Context::from_waker(Waker::noop())
    }

    #[test]
    fn non_terminal_events_alone_never_fire_the_signal() {
        let inner: DynRecordingSink = Arc::new(MemorySink::default());
        let sink = TerminalSignal::new(inner, is_response_ended);
        let mut context = noop_context();
        let mut drained = sink.drained();

        futures::executor::block_on(sink.append(request_ended(0))).expect("append");
        futures::executor::block_on(sink.flush()).expect("flush");

        assert_eq!(
            Pin::new(&mut drained).poll(&mut context),
            Poll::Pending,
            "flushing only non-terminal events must never fire"
        );
    }

    #[test]
    fn drained_resolves_only_after_the_terminal_event_is_flushed() {
        let inner: DynRecordingSink = Arc::new(MemorySink::default());
        let sink = TerminalSignal::new(inner, is_response_ended);
        let mut context = noop_context();
        let mut drained = sink.drained();

        assert_eq!(Pin::new(&mut drained).poll(&mut context), Poll::Pending);

        futures::executor::block_on(sink.append(response_ended(1))).expect("append terminal");
        assert_eq!(
            Pin::new(&mut drained).poll(&mut context),
            Poll::Pending,
            "appended but not yet flushed -- not durable yet"
        );

        futures::executor::block_on(sink.flush()).expect("flush");
        assert_eq!(
            Pin::new(&mut drained).poll(&mut context),
            Poll::Ready(()),
            "terminal event appended AND flushed -- durable, signal fires"
        );
    }

    #[test]
    fn late_subscriber_after_fire_resolves_on_first_poll() {
        let inner: DynRecordingSink = Arc::new(MemorySink::default());
        let sink = TerminalSignal::new(inner, is_response_ended);

        futures::executor::block_on(async {
            sink.append(response_ended(0)).await.expect("append");
            sink.flush().await.expect("flush");
        });

        let mut context = noop_context();
        let mut late = sink.drained();
        assert_eq!(
            Pin::new(&mut late).poll(&mut context),
            Poll::Ready(()),
            "sticky level: a subscriber arriving after the fire resolves immediately"
        );
    }
}
