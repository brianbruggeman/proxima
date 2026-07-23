//! Shared outer-wait driver for STATEFUL connection protocols that race
//! socket-byte-arrival against an async push channel (redis pub/sub,
//! pgwire LISTEN/NOTIFY) and a shutdown signal — the sibling of
//! [`crate::serve_pipe::handle_connection`] (the ONE CONNECT/upgrade
//! driver pgwire and redis already share) for the OUTER WAIT step of their
//! own `main_loop`'s `select_biased!`.
//!
//! # Why the inner decode+dispatch loop stays in the caller
//!
//! Every source [`FanIn`] merges must be
//! [`proxima_core::markers::DropSafe`] — `fan_in.rs`'s own `UnpinPipe +
//! DropSafe` bound on every merged source, enforced at `FanIn::call`'s
//! trait bounds (see that module's doc: a source's in-flight `call` future
//! is polled once per scan and DROPPED if not ready, so only a source
//! whose partial/cancelled call leaves no observable torn state may be
//! raced). [`proxima_core::markers::DropSafe`]'s own doc draws the exact
//! line: "purely computational pipes (codecs, in-memory transforms) ...
//! NOT justified: a streaming send whose partial body leaves the peer
//! mid-message."
//!
//! An arbitrary business `handler.call()` dispatch is precisely the
//! disqualified case: dropping it mid-flight could leave a half-applied
//! command, an unacknowledged PUBLISH, or a query the engine started but
//! never finished. So decode+dispatch — the inner loop that calls the
//! business handler once per parsed frame — stays sequential and OUTSIDE
//! the race, in the caller. [`wait_for_wire_event`] only multiplexes the
//! WAIT step: "is there anything to do yet" (more bytes, a queued push, or
//! shutdown) — never the dispatch itself.

use bytes::Bytes;
use futures::io::{AsyncWrite, AsyncWriteExt};

use proxima_core::ProximaError;
use proxima_core::markers::DropSafe;
use proxima_primitives::pipe::{Exhausted, FanIn, FanInStrategy, Pipe, UnpinPipe};

/// What a caller-supplied wait source (a per-connection `UnpinPipe<In =
/// (), Out = WireEvent, Err = Exhausted> + DropSafe` enum — e.g.
/// `Read`/`Push`/`Shutdown` variants) produces once it has something.
/// Fixed here, not per-caller, so [`wait_for_wire_event`] is the ONE
/// driver every such enum composes against, regardless of protocol.
#[derive(Debug)]
#[non_exhaustive]
pub enum WireEvent {
    /// New bytes arrived on the read half — hand them to the caller's own
    /// parser (`Connection::feed_bytes`, `buf.extend_from_slice`, ...).
    Read(Bytes),
    /// Out-of-band bytes, already wire-encoded, ready to write immediately
    /// (a pub/sub push, a LISTEN/NOTIFY notification).
    Push(Bytes),
    /// The peer closed the connection, or a graceful-shutdown signal
    /// fired — stop waiting.
    Stop,
    /// A source hit a real I/O error (not a benign EOF) — surfaced through
    /// this variant rather than `UnpinPipe::Err` (fixed to [`Exhausted`] by
    /// `FanIn`'s own contract, which carries no detail) so the failure
    /// still propagates with full context instead of being swallowed as a
    /// plain `Stop`.
    Failed(std::io::Error),
}

/// Races `sources` (typically Read + Push + Shutdown) via [`FanIn`],
/// writing every [`WireEvent::Push`] straight onto `write_half` and
/// re-polling, until [`WireEvent::Read`] arrives (returned to the caller,
/// `Some(bytes)`) or the wait ends ([`WireEvent::Stop`], or every source
/// [`Exhausted`]; both surface as `None`).
///
/// The caller's own decode+dispatch loop runs BEFORE this is called (drain
/// whatever is already buffered) and again after it returns `Some` — this
/// function is exactly the wait step in between, nothing more.
///
/// # Errors
/// A write failure while flushing a `Push` event onto `write_half`, or a
/// source reporting [`WireEvent::Failed`].
pub async fn wait_for_wire_event<S, Strategy, const SOURCES: usize, W>(
    sources: &FanIn<S, Strategy, SOURCES>,
    write_half: &mut W,
) -> Result<Option<Bytes>, ProximaError>
where
    S: UnpinPipe<In = (), Out = WireEvent, Err = Exhausted> + DropSafe,
    Strategy: FanInStrategy,
    W: AsyncWrite + Unpin,
{
    loop {
        match Pipe::call(sources, ()).await {
            Ok(WireEvent::Read(bytes)) => return Ok(Some(bytes)),
            Ok(WireEvent::Push(bytes)) => {
                write_half
                    .write_all(&bytes)
                    .await
                    .map_err(ProximaError::Io)?;
                write_half.flush().await.map_err(ProximaError::Io)?;
            }
            Ok(WireEvent::Stop) | Err(Exhausted) => return Ok(None),
            Ok(WireEvent::Failed(error)) => return Err(ProximaError::Io(error)),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use core::cell::RefCell;
    use core::future::Future;
    use core::pin::Pin;
    use core::task::{Context, Poll};

    use proxima_primitives::pipe::Select;

    /// A scripted source: yields the next entry in `steps` on each call,
    /// `Exhausted` once the script runs out. `Unpin` (a hand-written poll
    /// struct, not an async block — see `primitives.rs`'s own note on why
    /// `UnpinPipe` needs this shape).
    struct Scripted {
        steps: RefCell<std::vec::Vec<Result<WireEvent, Exhausted>>>,
    }

    impl DropSafe for Scripted {}

    /// Holds its outcome behind `Option` (not a bare value) so `poll` can
    /// MOVE it out on the one poll that matters — `WireEvent::Failed`
    /// carries an `io::Error`, which is not `Clone`.
    struct ScriptedCall(Option<Result<WireEvent, Exhausted>>);

    impl Future for ScriptedCall {
        type Output = Result<WireEvent, Exhausted>;
        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Ready(
                self.get_mut()
                    .0
                    .take()
                    .expect("ScriptedCall polled after it already resolved"),
            )
        }
    }

    impl UnpinPipe for Scripted {
        type In = ();
        type Out = WireEvent;
        type Err = Exhausted;

        fn call(&self, (): ()) -> impl Future<Output = Result<WireEvent, Exhausted>> + Unpin {
            let mut steps = self.steps.borrow_mut();
            if steps.is_empty() {
                return ScriptedCall(Some(Err(Exhausted)));
            }
            ScriptedCall(Some(steps.remove(0)))
        }
    }

    /// An in-memory `AsyncWrite` recording every byte written — proves
    /// `wait_for_wire_event` writes `Push` events itself, not just returns
    /// them.
    #[derive(Default)]
    struct RecordingWriter {
        written: std::vec::Vec<u8>,
    }

    impl AsyncWrite for RecordingWriter {
        fn poll_write(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            self.written.extend_from_slice(buf);
            Poll::Ready(Ok(buf.len()))
        }
        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }
        fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    /// Dependency-free executor — `Scripted` is `RefCell`-backed (never
    /// `Sync`), so it cannot cross `#[proxima::test]`'s `Send + 'static`
    /// future bound; mirrors `fan_in.rs`'s own `block_on` test helper.
    fn block_on<Fut: Future>(future: Fut) -> Fut::Output {
        let mut pinned = core::pin::pin!(future);
        let mut context = Context::from_waker(std::task::Waker::noop());
        loop {
            if let Poll::Ready(output) = pinned.as_mut().poll(&mut context) {
                return output;
            }
        }
    }

    #[test]
    fn read_event_returns_to_the_caller_with_no_write() {
        let source = Scripted {
            steps: RefCell::new(std::vec![Ok(WireEvent::Read(Bytes::from_static(b"hi")))]),
        };
        let fan = FanIn::new([source], Select::Fifo);
        let mut writer = RecordingWriter::default();

        let outcome = block_on(wait_for_wire_event(&fan, &mut writer)).expect("wait");
        assert_eq!(outcome, Some(Bytes::from_static(b"hi")));
        assert!(writer.written.is_empty(), "a Read event must not write anything");
    }

    #[test]
    fn push_events_are_written_then_the_wait_continues_to_a_read() {
        let source = Scripted {
            steps: RefCell::new(std::vec![
                Ok(WireEvent::Push(Bytes::from_static(b"push-1"))),
                Ok(WireEvent::Push(Bytes::from_static(b"push-2"))),
                Ok(WireEvent::Read(Bytes::from_static(b"bytes"))),
            ]),
        };
        let fan = FanIn::new([source], Select::Fifo);
        let mut writer = RecordingWriter::default();

        let outcome = block_on(wait_for_wire_event(&fan, &mut writer)).expect("wait");
        assert_eq!(outcome, Some(Bytes::from_static(b"bytes")));
        assert_eq!(writer.written, b"push-1push-2");
    }

    #[test]
    fn failed_event_propagates_as_a_real_error_not_a_silent_stop() {
        let source = Scripted {
            steps: RefCell::new(std::vec![Ok(WireEvent::Failed(std::io::Error::other(
                "boom",
            )))]),
        };
        let fan = FanIn::new([source], Select::Fifo);
        let mut writer = RecordingWriter::default();

        let outcome = block_on(wait_for_wire_event(&fan, &mut writer));
        assert!(
            matches!(outcome, Err(ProximaError::Io(_))),
            "expected a real error, got {outcome:?}"
        );
    }

    #[test]
    fn stop_event_ends_the_wait_with_none() {
        let source = Scripted {
            steps: RefCell::new(std::vec![Ok(WireEvent::Stop)]),
        };
        let fan = FanIn::new([source], Select::Fifo);
        let mut writer = RecordingWriter::default();

        let outcome = block_on(wait_for_wire_event(&fan, &mut writer)).expect("wait");
        assert_eq!(outcome, None);
    }

    #[test]
    fn every_source_exhausted_ends_the_wait_with_none() {
        let source = Scripted {
            steps: RefCell::new(std::vec::Vec::new()),
        };
        let fan = FanIn::new([source], Select::Fifo);
        let mut writer = RecordingWriter::default();

        let outcome = block_on(wait_for_wire_event(&fan, &mut writer)).expect("wait");
        assert_eq!(outcome, None);
    }
}
