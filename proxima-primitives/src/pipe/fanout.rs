//! `FanOut` — broadcast one input to N sink [`SendPipe`]s (a 1→N tee).
//!
//! Generalises the broadcast shape downstream crates kept re-implementing per
//! payload (recording fans `RecordingEvent`s to durable logs; telemetry fans
//! records to exporters). It composes [`SendPipe`] — the sinks are ordinary
//! pipes, so "a sink" needs no bespoke trait. Generic over the sink type `S` and
//! a [`FanPolicy`], both monomorphised: a given instantiation is the same
//! machine code a hand-rolled concrete fan-out would be, with no `dyn` erasure on
//! the hot path. That static dispatch is exactly what lets it out-run a
//! `Vec<Box<dyn _>>` broadcast (tracing's layer model).
//!
//! The input reaches every sink with the minimum number of clones: it is *moved*
//! into the last sink and *cloned* into the earlier ones. Clone is expected to be
//! a refcount bump (Arc-backed payloads such as `bytes::Bytes`), not a deep copy
//! — fan out `Arc<T>` when `T` is not cheaply clonable.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::future::Future;
use core::marker::PhantomData;

use crate::pipe::SendPipe;

/// How a [`FanOut`] reacts to a sink error.
///
/// A marker trait carrying a single `const` so the loop's reaction folds at
/// monomorphisation — no runtime branch on the hot path.
pub trait FanPolicy: Send + Sync + 'static {
    /// Stop at the first sink error (`true`), or attempt every sink and surface
    /// the first error after all have run (`false`).
    const SHORT_CIRCUIT: bool;
    /// Drop per-sink errors silently — every sink is attempted and the call
    /// always returns `Ok` (`true`), or surface the first error (`false`).
    /// Ignored when `SHORT_CIRCUIT` is `true`. A `const`, so the `first_err`
    /// slot folds away for the policies that don't keep it.
    const IGNORE_ERRORS: bool;
}

/// Durable fan-out: a sink error fails the call and skips the remaining sinks.
/// The recording fan-out's semantics.
pub struct AllOrNothing;
impl FanPolicy for AllOrNothing {
    const SHORT_CIRCUIT: bool = true;
    const IGNORE_ERRORS: bool = false;
}

/// Best-effort fan-out: every sink is attempted; the first error (if any) is
/// surfaced after all run, so one broken sink cannot suppress the others. The
/// telemetry exporter fan-out's semantics.
pub struct BestEffort;
impl FanPolicy for BestEffort {
    const SHORT_CIRCUIT: bool = false;
    const IGNORE_ERRORS: bool = false;
}

/// Fire-and-forget fan-out: every sink is attempted and per-sink errors are
/// dropped — the call always returns `Ok`. For telemetry exporters that must
/// never fail the caller because one downstream is broken.
pub struct IgnoreErrors;
impl FanPolicy for IgnoreErrors {
    const SHORT_CIRCUIT: bool = false;
    const IGNORE_ERRORS: bool = true;
}

/// Broadcast composition over N sink [`SendPipe`]s.
///
/// `Clone` is a refcount bump on the shared `Arc<Vec<S>>`, so a `FanOut` handle
/// is cheap to hand to each `call` site.
pub struct FanOut<S, Policy = AllOrNothing> {
    sinks: Arc<Vec<S>>,
    policy: PhantomData<fn() -> Policy>,
}

impl<S, Policy> Clone for FanOut<S, Policy> {
    fn clone(&self) -> Self {
        Self {
            sinks: Arc::clone(&self.sinks),
            policy: PhantomData,
        }
    }
}

impl<S, Policy> FanOut<S, Policy> {
    /// Fan out to `sinks`, in order, under `Policy`.
    #[must_use]
    pub fn new(sinks: Vec<S>) -> Self {
        Self {
            sinks: Arc::new(sinks),
            policy: PhantomData,
        }
    }

    /// Number of sinks the input is broadcast to.
    #[must_use]
    pub fn sink_count(&self) -> usize {
        self.sinks.len()
    }

    /// The sinks, for the durability/introspection a wrapper layers on top
    /// (e.g. driving each sink's `flush`/`sync`).
    #[must_use]
    pub fn sinks(&self) -> &[S] {
        &self.sinks
    }
}

impl<S> FanOut<S, AllOrNothing> {
    /// Construct the durable, all-or-nothing fan-out (the default policy), read
    /// at the call site instead of via a turbofish.
    #[must_use]
    pub fn all_or_nothing(sinks: Vec<S>) -> Self {
        Self::new(sinks)
    }
}

impl<S> FanOut<S, BestEffort> {
    /// Construct the best-effort fan-out — attempt every sink regardless of
    /// per-sink failure.
    #[must_use]
    pub fn best_effort(sinks: Vec<S>) -> Self {
        Self::new(sinks)
    }
}

impl<S> FanOut<S, IgnoreErrors> {
    /// Construct the fire-and-forget fan-out — attempt every sink and drop all
    /// per-sink errors (the call always returns `Ok`).
    #[must_use]
    pub fn ignore_errors(sinks: Vec<S>) -> Self {
        Self::new(sinks)
    }
}

impl<S, Policy> SendPipe for FanOut<S, Policy>
where
    S: SendPipe,
    S::In: Clone + Send,
    Policy: FanPolicy,
{
    type In = S::In;
    type Out = ();
    type Err = S::Err;

    fn call(&self, item: S::In) -> impl Future<Output = Result<(), S::Err>> + Send {
        let sinks = Arc::clone(&self.sinks);
        async move {
            // move into the last sink, clone into the earlier ones — clone is a
            // refcount bump for Arc-backed payloads, so N sinks cost N-1 bumps.
            let Some((last, rest)) = sinks.split_last() else {
                return Ok(());
            };
            let mut first_err: Option<S::Err> = None;
            for sink in rest {
                if let Err(err) = sink.call(item.clone()).await {
                    if Policy::SHORT_CIRCUIT {
                        return Err(err);
                    }
                    if !Policy::IGNORE_ERRORS {
                        first_err.get_or_insert(err);
                    }
                }
            }
            if let Err(err) = last.call(item).await {
                if Policy::SHORT_CIRCUIT {
                    return Err(err);
                }
                if !Policy::IGNORE_ERRORS {
                    first_err.get_or_insert(err);
                }
            }
            match first_err {
                Some(err) => Err(err),
                None => Ok(()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use alloc::vec;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use futures::executor::block_on;

    #[derive(Debug, PartialEq)]
    struct SinkErr(u32);

    // a sink that records every call (and optionally fails), so a test can
    // assert which sinks the fan-out actually reached.
    struct CountingSink {
        calls: Arc<AtomicUsize>,
        fail: bool,
    }

    impl SendPipe for CountingSink {
        type In = Payload;
        type Out = ();
        type Err = SinkErr;

        fn call(&self, input: Payload) -> impl Future<Output = Result<(), SinkErr>> + Send {
            let calls = Arc::clone(&self.calls);
            let fail = self.fail;
            async move {
                calls.fetch_add(1, Ordering::Relaxed);
                if fail { Err(SinkErr(input.0)) } else { Ok(()) }
            }
        }
    }

    // a payload whose clone bumps a shared counter — proves the move-last
    // optimisation (N sinks => N-1 clones, not N).
    #[derive(Debug)]
    struct Payload(u32, Arc<AtomicUsize>);

    impl Clone for Payload {
        fn clone(&self) -> Self {
            self.1.fetch_add(1, Ordering::Relaxed);
            Self(self.0, Arc::clone(&self.1))
        }
    }

    fn ok_sink(calls: &Arc<AtomicUsize>) -> CountingSink {
        CountingSink {
            calls: Arc::clone(calls),
            fail: false,
        }
    }

    #[test]
    fn fans_input_to_every_sink() {
        let calls = Arc::new(AtomicUsize::new(0));
        let clones = Arc::new(AtomicUsize::new(0));
        let fan =
            FanOut::<_, AllOrNothing>::new(vec![ok_sink(&calls), ok_sink(&calls), ok_sink(&calls)]);

        block_on(fan.call(Payload(7, Arc::clone(&clones)))).unwrap();

        assert_eq!(
            calls.load(Ordering::Relaxed),
            3,
            "every sink received the input"
        );
        assert_eq!(
            clones.load(Ordering::Relaxed),
            2,
            "moved into last, cloned into the other two"
        );
    }

    #[test]
    fn single_sink_moves_without_cloning() {
        let calls = Arc::new(AtomicUsize::new(0));
        let clones = Arc::new(AtomicUsize::new(0));
        let fan = FanOut::all_or_nothing(vec![ok_sink(&calls)]);

        block_on(fan.call(Payload(1, Arc::clone(&clones)))).unwrap();

        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(clones.load(Ordering::Relaxed), 0, "one sink is a pure move");
    }

    #[test]
    fn all_or_nothing_stops_at_first_error() {
        let calls = Arc::new(AtomicUsize::new(0));
        let clones = Arc::new(AtomicUsize::new(0));
        let sinks = vec![
            CountingSink {
                calls: Arc::clone(&calls),
                fail: true,
            },
            ok_sink(&calls),
            ok_sink(&calls),
        ];
        let fan = FanOut::<_, AllOrNothing>::new(sinks);

        let err = block_on(fan.call(Payload(42, Arc::clone(&clones)))).unwrap_err();

        assert_eq!(err, SinkErr(42));
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "later sinks are skipped after the failure"
        );
    }

    #[test]
    fn ignore_errors_attempts_all_and_always_returns_ok() {
        let calls = Arc::new(AtomicUsize::new(0));
        let clones = Arc::new(AtomicUsize::new(0));
        let sinks = vec![
            CountingSink {
                calls: Arc::clone(&calls),
                fail: true,
            },
            ok_sink(&calls),
            CountingSink {
                calls: Arc::clone(&calls),
                fail: true,
            },
        ];
        let fan = FanOut::ignore_errors(sinks);

        block_on(fan.call(Payload(5, Arc::clone(&clones)))).expect("ignore_errors never fails");

        assert_eq!(calls.load(Ordering::Relaxed), 3, "every sink was attempted");
    }

    #[test]
    fn best_effort_attempts_all_then_surfaces_first_error() {
        let calls = Arc::new(AtomicUsize::new(0));
        let clones = Arc::new(AtomicUsize::new(0));
        let sinks = vec![
            CountingSink {
                calls: Arc::clone(&calls),
                fail: true,
            },
            ok_sink(&calls),
            CountingSink {
                calls: Arc::clone(&calls),
                fail: true,
            },
        ];
        let fan = FanOut::best_effort(sinks);

        let err = block_on(fan.call(Payload(9, Arc::clone(&clones)))).unwrap_err();

        assert_eq!(err, SinkErr(9), "the first error is surfaced");
        assert_eq!(calls.load(Ordering::Relaxed), 3, "every sink was attempted");
    }
}
