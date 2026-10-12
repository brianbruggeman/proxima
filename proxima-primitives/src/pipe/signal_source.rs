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
    fired: Fired,
    yielded: AtomicBool,
}

impl SignalSource {
    #[must_use]
    pub fn new(signal: &Signal) -> Self {
        Self {
            fired: signal.fired(),
            yielded: AtomicBool::new(false),
        }
    }
}

impl DropSafe for SignalSource {}

/// The future behind [`SignalSource::call`]. Once already-yielded, resolves
/// [`Exhausted`] with no further work; otherwise it polls the source's own
/// [`Fired`], so the waker registration lives in the source and abandoning
/// this future (as [`crate::pipe::fan_in::FanIn`] does each scan) loses nothing.
enum SignalCall<'source> {
    Yielded,
    Pending { source: &'source SignalSource },
}

impl Future for SignalCall<'_> {
    type Output = Result<(), Exhausted>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.get_mut() {
            SignalCall::Yielded => Poll::Ready(Err(Exhausted)),
            SignalCall::Pending { source } => match source.fired.poll_shared(cx) {
                Poll::Ready(()) => {
                    source.yielded.store(true, Ordering::Relaxed);
                    Poll::Ready(Ok(()))
                }
                Poll::Pending => Poll::Pending,
            },
        }
    }
}

impl SignalSource {
    fn call_future(&self) -> SignalCall<'_> {
        if self.yielded.load(Ordering::Relaxed) {
            SignalCall::Yielded
        } else {
            SignalCall::Pending { source: self }
        }
    }
}

impl UnpinPipe for SignalSource {
    type In = ();
    type Out = ();
    type Err = Exhausted;

    fn call(&self, (): ()) -> impl Future<Output = Result<(), Exhausted>> + Unpin {
        self.call_future()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::task::{Wake, Waker};

    use super::*;
    use crate::pipe::fan_in::{FanIn, Select};
    use crate::pipe::primitives::Pipe;

    fn noop_context() -> Context<'static> {
        Context::from_waker(std::task::Waker::noop())
    }

    fn call_once(source: &SignalSource, cx: &mut Context<'_>) -> Poll<Result<(), Exhausted>> {
        let mut call = source.call(());
        Pin::new(&mut call).poll(cx)
    }


    struct CountingWaker(AtomicUsize);

    impl Wake for CountingWaker {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn counting_waker() -> (Arc<CountingWaker>, Waker) {
        let counter = Arc::new(CountingWaker(AtomicUsize::new(0)));
        let waker = Waker::from(counter.clone());
        (counter, waker)
    }

    enum Arm {
        Cancel(SignalSource),
        Idle,
    }

    impl DropSafe for Arm {}

    enum ArmCall<'arm> {
        Cancel(SignalCall<'arm>),
        Idle,
    }

    impl Future for ArmCall<'_> {
        type Output = Result<(), Exhausted>;

        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            match self.get_mut() {
                ArmCall::Cancel(call) => Pin::new(call).poll(cx),
                ArmCall::Idle => Poll::Pending,
            }
        }
    }

    impl UnpinPipe for Arm {
        type In = ();
        type Out = ();
        type Err = Exhausted;

        fn call(&self, (): ()) -> impl Future<Output = Result<(), Exhausted>> + Unpin {
            match self {
                Arm::Cancel(source) => ArmCall::Cancel(source.call_future()),
                Arm::Idle => ArmCall::Idle,
            }
        }
    }

    fn merged(signal: &Signal) -> FanIn<Arm, Select, 2> {
        FanIn::new(
            [Arm::Cancel(SignalSource::new(signal)), Arm::Idle],
            Select::RoundRobin,
        )
    }

    fn poll_merge(
        merge: &FanIn<Arm, Select, 2>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<(), Exhausted>> {
        let mut call = Pipe::call(merge, ());
        Pin::new(&mut call).poll(cx)
    }

    #[test]
    fn fan_in_parked_on_a_signal_source_is_woken_by_fire() {
        let signal = Signal::new();
        let merge = merged(&signal);
        let (counter, waker) = counting_waker();
        let mut cx = Context::from_waker(&waker);
        assert_eq!(poll_merge(&merge, &mut cx), Poll::Pending);

        signal.fire();

        assert!(counter.0.load(Ordering::Relaxed) >= 1, "fire must wake the parked fan-in");
        assert_eq!(poll_merge(&merge, &mut cx), Poll::Ready(Ok(())));
    }

    #[test]
    fn dropping_the_source_unregisters_its_waker() {
        let signal = Signal::new();
        let (counter, waker) = counting_waker();
        let mut cx = Context::from_waker(&waker);
        let source = SignalSource::new(&signal);
        assert_eq!(call_once(&source, &mut cx), Poll::Pending);

        drop(source);
        signal.fire();

        assert_eq!(counter.0.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn fan_in_over_an_unfired_signal_source_is_never_woken() {
        let signal = Signal::new();
        let merge = merged(&signal);
        let (counter, waker) = counting_waker();
        let mut cx = Context::from_waker(&waker);
        for _ in 0..4 {
            assert_eq!(poll_merge(&merge, &mut cx), Poll::Pending);
        }
        assert_eq!(counter.0.load(Ordering::Relaxed), 0);
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
