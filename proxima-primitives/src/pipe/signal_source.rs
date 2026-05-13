//! [`SignalSource`] — an [`UnpinPipe`] face over [`proxima_core::signal::Signal`]
//! — a cancellation scope as a one-shot source.
//!
//! The signal is a sticky level; its source face resolves `Ok(())` exactly
//! once when the level fires (immediately for a late subscriber) and then
//! resolves [`Exhausted`] forever after. That makes scope-merge plain source
//! algebra: `FanIn` over signal sources races cancellation against any other
//! source without bespoke select plumbing.

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, Ordering};
use core::task::{Context, Poll};

use proxima_core::markers::DropSafe;
use proxima_core::signal::{Fired, Signal};

use crate::pipe::fan_in::Exhausted;
use crate::pipe::primitives::UnpinPipe;

/// One observer's source face over a [`Signal`]. Construct per
/// observer (it tracks whether this observer already yielded).
pub struct SignalSource {
    signal: Signal,
    yielded: AtomicBool,
}

impl SignalSource {
    #[must_use]
    pub fn new(signal: &Signal) -> Self {
        Self {
            signal: signal.clone(),
            yielded: AtomicBool::new(false),
        }
    }
}

impl DropSafe for SignalSource {}

/// The future behind [`SignalSource::call`]. Once already-yielded, resolves
/// [`Exhausted`] with no further work; otherwise it owns a fresh [`Fired`] —
/// cheap (an `Arc` clone plus a `None` waker slot) and, per `Fired`'s own
/// `Drop` impl, safe to abandon mid-poll (the exact `DropSafe` shape
/// [`crate::pipe::fan_in::FanIn`] requires of every merged source).
enum SignalCall<'source> {
    Yielded,
    Pending {
        source: &'source SignalSource,
        fired: Fired,
    },
}

impl Future for SignalCall<'_> {
    type Output = Result<(), Exhausted>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.get_mut() {
            SignalCall::Yielded => Poll::Ready(Err(Exhausted)),
            SignalCall::Pending { source, fired } => match Pin::new(fired).poll(cx) {
                Poll::Ready(()) => {
                    source.yielded.store(true, Ordering::Relaxed);
                    Poll::Ready(Ok(()))
                }
                Poll::Pending => Poll::Pending,
            },
        }
    }
}

impl UnpinPipe for SignalSource {
    type In = ();
    type Out = ();
    type Err = Exhausted;

    fn call(&self, (): ()) -> impl Future<Output = Result<(), Exhausted>> + Unpin {
        if self.yielded.load(Ordering::Relaxed) {
            SignalCall::Yielded
        } else {
            SignalCall::Pending {
                source: self,
                fired: self.signal.fired(),
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn noop_context() -> Context<'static> {
        Context::from_waker(std::task::Waker::noop())
    }

    fn call_once(source: &SignalSource, cx: &mut Context<'_>) -> Poll<Result<(), Exhausted>> {
        let mut call = source.call(());
        Pin::new(&mut call).poll(cx)
    }

    #[test]
    fn yields_unit_once_when_fired_then_exhausts() {
        let signal = Signal::new();
        let source = SignalSource::new(&signal);
        let mut cx = noop_context();
        assert_eq!(call_once(&source, &mut cx), Poll::Pending);
        signal.fire();
        assert_eq!(call_once(&source, &mut cx), Poll::Ready(Ok(())));
        assert_eq!(call_once(&source, &mut cx), Poll::Ready(Err(Exhausted)));
    }

    #[test]
    fn late_subscriber_yields_immediately() {
        let signal = Signal::new();
        signal.fire();
        let source = SignalSource::new(&signal);
        let mut cx = noop_context();
        assert_eq!(call_once(&source, &mut cx), Poll::Ready(Ok(())));
    }
}
