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
use core::pin::Pin;
use core::task::{Context, Poll};

use crate::pipe::{Pipe, SendPipe, UnpinPipe, UnpinSendPipe};

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

// The base-tier mirror of the `SendPipe` impl above — same broadcast, no
// `Send` bound on the sinks or the returned future, so `PipeExt` (which only
// ever assumes `Pipe`) reaches `FanOut` too. Kept byte-for-byte parallel to
// the `SendPipe` arm on purpose: one broadcast law, two tiers.
impl<S, Policy> Pipe for FanOut<S, Policy>
where
    S: Pipe,
    S::In: Clone,
    Policy: FanPolicy,
{
    type In = S::In;
    type Out = ();
    type Err = S::Err;

    fn call(&self, item: S::In) -> impl Future<Output = Result<(), S::Err>> {
        let sinks = Arc::clone(&self.sinks);
        async move {
            let Some((last, rest)) = sinks.split_last() else {
                return Ok(());
            };
            let mut first_err: Option<S::Err> = None;
            for sink in rest {
                if let Err(err) = Pipe::call(sink, item.clone()).await {
                    if Policy::SHORT_CIRCUIT {
                        return Err(err);
                    }
                    if !Policy::IGNORE_ERRORS {
                        first_err.get_or_insert(err);
                    }
                }
            }
            if let Err(err) = Pipe::call(last, item).await {
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

/// Hand-written poll state machine backing `FanOut`'s `UnpinPipe`/
/// `UnpinSendPipe` impls — no `Box::pin`, no `async move`. One sink is
/// in-flight (`current`) at a time, same sequential broadcast the async-block
/// `Pipe`/`SendPipe` forms above run, just polled in place.
///
/// `S::call`'s return type is RPITIT and, like any `&self` method, MAY borrow
/// `self` (here, the specific sink) — so `current`'s type (`SinkFut`) can
/// neither be named nor be produced from a `Fn(&S, In) -> SinkFut` seed
/// closure taking the sink BY REFERENCE per call: that shape is inherently
/// higher-ranked (`for<'a> Fn(&'a S, ..)`), and a borrowing `SinkFut` cannot
/// be one fixed type across arbitrarily many distinct per-call lifetimes.
/// `seed` instead CAPTURES `sinks: &'fan [S]` once (a single, concrete
/// lifetime, exactly like [`start_and_then`](super::primitives)'s `next`
/// capturing `&Second`) and takes only an index (`usize`, no lifetime) per
/// call — `sinks[index].call(..)` then borrows the ONE captured slice
/// reference every time, so `SinkFut` really is one fixed type, and `seed`'s
/// own `Fn::Output` pins it. Every later transition calls `seed` again —
/// through the ABSTRACT `Seed` type, never `S::call` directly — so it
/// type-checks generically.
///
/// `In`/`Out`/`Err` are free type parameters rather than `S::In`/`S::Out`/
/// `S::Err`, so the SAME state machine serves both `UnpinPipe` (`S: Pipe`)
/// and `UnpinSendPipe` (`S: SendPipe`) — two unrelated standalone traits —
/// without duplicating this machinery per tier the way `FanIn`'s `Send`
/// mirror had to (its merge loop calls `S::call` directly, with no `seed`
/// indirection to hide behind).
enum FanOutUnpinCall<'fan, S, In, Out, Err, Policy, Seed, SinkFut> {
    /// No sinks: the async form's `sinks.split_last() == None` arm.
    Empty,
    Running {
        sinks: &'fan [S],
        index: usize,
        /// The broadcast item, consumed (moved) into the LAST sink; `None`
        /// once that has happened. Every earlier sink gets `item.clone()` —
        /// mirrors the async form's move-last/clone-rest optimisation.
        item: Option<In>,
        current: SinkFut,
        first_err: Option<Err>,
        seed: Seed,
        marker: PhantomData<fn() -> (Out, Policy)>,
    },
}

/// Construct [`FanOutUnpinCall`]'s starting state. A free fn, not a bare
/// struct literal: `SinkFut` appears in no argument directly — only in
/// `seed`'s own `Fn` bound — and a struct literal gives the compiler nothing
/// to solve a missing field's type from (see the type's own doc, and
/// [`start_and_then`](super::primitives) for the general shape of this
/// trick).
fn start_fan_out<'fan, S, In, Out, Err, Policy, Seed, SinkFut>(
    sinks: &'fan [S],
    item: In,
    seed: Seed,
) -> FanOutUnpinCall<'fan, S, In, Out, Err, Policy, Seed, SinkFut>
where
    In: Clone,
    Seed: Fn(usize, In) -> SinkFut,
{
    if sinks.is_empty() {
        return FanOutUnpinCall::Empty;
    }
    let is_last = sinks.len() == 1;
    let (call_item, remaining_item) = if is_last {
        (item, None)
    } else {
        (item.clone(), Some(item))
    };
    let current = seed(0, call_item);
    FanOutUnpinCall::Running {
        sinks,
        index: 0,
        item: remaining_item,
        current,
        first_err: None,
        seed,
        marker: PhantomData,
    }
}

impl<S, In, Out, Err, Policy, Seed, SinkFut> Future
    for FanOutUnpinCall<'_, S, In, Out, Err, Policy, Seed, SinkFut>
where
    In: Clone + Unpin,
    Err: Unpin,
    Seed: Fn(usize, In) -> SinkFut + Unpin,
    SinkFut: Future<Output = Result<Out, Err>> + Unpin,
    Policy: FanPolicy,
{
    type Output = Result<(), Err>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // `Self: Unpin` follows structurally from the bounds above, so
        // `get_mut` needs no `unsafe` pin projection.
        let this = self.get_mut();
        loop {
            match this {
                FanOutUnpinCall::Empty => return Poll::Ready(Ok(())),
                FanOutUnpinCall::Running {
                    sinks,
                    index,
                    item,
                    current,
                    first_err,
                    seed,
                    ..
                } => match Pin::new(&mut *current).poll(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(outcome) => {
                        if let Err(err) = outcome {
                            if Policy::SHORT_CIRCUIT {
                                return Poll::Ready(Err(err));
                            }
                            if !Policy::IGNORE_ERRORS {
                                first_err.get_or_insert(err);
                            }
                        }
                        let next_index = *index + 1;
                        if next_index >= sinks.len() {
                            return Poll::Ready(match first_err.take() {
                                Some(err) => Err(err),
                                None => Ok(()),
                            });
                        }
                        let is_last = next_index == sinks.len() - 1;
                        // `item` is `Some` until moved into the last sink, by
                        // construction — these `None` arms are unreachable in
                        // practice; parking (rather than panicking) is the
                        // house style for a violated-by-construction
                        // invariant.
                        let call_item = if is_last {
                            match item.take() {
                                Some(value) => value,
                                None => return Poll::Pending,
                            }
                        } else {
                            match item.as_ref() {
                                Some(value) => value.clone(),
                                None => return Poll::Pending,
                            }
                        };
                        *current = seed(next_index, call_item);
                        *index = next_index;
                        // loop: the freshly-seeded sink may already be ready.
                    }
                },
            }
        }
    }
}

impl<S, Policy> UnpinPipe for FanOut<S, Policy>
where
    S: UnpinPipe,
    S::In: Clone + Unpin,
    S::Err: Unpin,
    Policy: FanPolicy,
{
    type In = S::In;
    type Out = ();
    type Err = S::Err;

    fn call(&self, item: S::In) -> impl Future<Output = Result<(), S::Err>> + Unpin {
        let sinks: &[S] = &self.sinks;
        start_fan_out::<S, S::In, S::Out, S::Err, Policy, _, _>(
            sinks,
            item,
            move |index, call_item| UnpinPipe::call(&sinks[index], call_item),
        )
    }
}

impl<S, Policy> UnpinSendPipe for FanOut<S, Policy>
where
    S: UnpinSendPipe,
    S::In: Clone + Send + Unpin,
    S::Err: Send + Unpin,
    Policy: FanPolicy,
{
    type In = S::In;
    type Out = ();
    type Err = S::Err;

    fn call(&self, item: S::In) -> impl Future<Output = Result<(), S::Err>> + Send + Unpin {
        let sinks: &[S] = &self.sinks;
        start_fan_out::<S, S::In, S::Out, S::Err, Policy, _, _>(
            sinks,
            item,
            move |index, call_item| UnpinSendPipe::call(&sinks[index], call_item),
        )
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

    // ── UnpinPipe / UnpinSendPipe tier ──────────────────────────────────────

    // mirrors `CountingSink`, but immediately ready via `core::future::ready`
    // (`UnpinPipe`) — proves `FanOut`'s hand-rolled poll state machine, not
    // just the async-block `SendPipe` form.
    struct UnpinCountingSink {
        calls: Arc<AtomicUsize>,
        fail: bool,
    }

    impl UnpinPipe for UnpinCountingSink {
        type In = Payload;
        type Out = ();
        type Err = SinkErr;

        fn call(&self, input: Payload) -> impl Future<Output = Result<(), SinkErr>> + Unpin {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let fail = self.fail;
            core::future::ready(if fail { Err(SinkErr(input.0)) } else { Ok(()) })
        }
    }

    impl UnpinSendPipe for UnpinCountingSink {
        type In = Payload;
        type Out = ();
        type Err = SinkErr;

        fn call(&self, input: Payload) -> impl Future<Output = Result<(), SinkErr>> + Send + Unpin {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let fail = self.fail;
            core::future::ready(if fail { Err(SinkErr(input.0)) } else { Ok(()) })
        }
    }

    fn unpin_ok_sink(calls: &Arc<AtomicUsize>) -> UnpinCountingSink {
        UnpinCountingSink {
            calls: Arc::clone(calls),
            fail: false,
        }
    }

    fn poll_once<F: Future + Unpin>(future: &mut F) -> Poll<F::Output> {
        let waker = core::task::Waker::noop();
        let mut cx = Context::from_waker(waker);
        Pin::new(future).poll(&mut cx)
    }

    #[test]
    fn unpin_fans_input_to_every_sink_with_move_last_clone_rest() {
        let calls = Arc::new(AtomicUsize::new(0));
        let clones = Arc::new(AtomicUsize::new(0));
        let fan = FanOut::<_, AllOrNothing>::new(vec![
            unpin_ok_sink(&calls),
            unpin_ok_sink(&calls),
            unpin_ok_sink(&calls),
        ]);

        let mut call = UnpinPipe::call(&fan, Payload(7, Arc::clone(&clones)));
        match poll_once(&mut call) {
            Poll::Ready(Ok(())) => {}
            other => panic!("expected ready, got {other:?}"),
        }

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
    fn unpin_empty_sinks_resolves_ok_immediately() {
        let fan: FanOut<UnpinCountingSink, AllOrNothing> = FanOut::new(vec![]);
        let clones = Arc::new(AtomicUsize::new(0));
        let mut call = UnpinPipe::call(&fan, Payload(1, clones));
        assert_eq!(poll_once(&mut call), Poll::Ready(Ok(())));
    }

    #[test]
    fn unpin_all_or_nothing_stops_at_first_error() {
        let calls = Arc::new(AtomicUsize::new(0));
        let sinks = vec![
            UnpinCountingSink {
                calls: Arc::clone(&calls),
                fail: true,
            },
            unpin_ok_sink(&calls),
            unpin_ok_sink(&calls),
        ];
        let fan = FanOut::<_, AllOrNothing>::new(sinks);

        let mut call = UnpinPipe::call(&fan, Payload(42, Arc::new(AtomicUsize::new(0))));
        match poll_once(&mut call) {
            Poll::Ready(Err(SinkErr(42))) => {}
            other => panic!("expected the first sink's error, got {other:?}"),
        }
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "later sinks are skipped after the failure"
        );
    }

    #[test]
    fn unpin_send_chain_is_send_and_unpin() {
        fn needs_send_unpin<F: Future + Send + Unpin>(_: &F) {}
        let fan = FanOut::<_, AllOrNothing>::new(vec![UnpinCountingSink {
            calls: Arc::new(AtomicUsize::new(0)),
            fail: false,
        }]);
        let call = UnpinSendPipe::call(&fan, Payload(1, Arc::new(AtomicUsize::new(0))));
        needs_send_unpin(&call);
    }

    // resolves `Pending` exactly once per sink (registering the waker), then
    // `Ready` — proves the state machine resumes correctly across SEPARATE
    // `poll()` calls (not just when every sink resolves on the first poll),
    // without re-invoking `call` on a sink already in flight.
    struct SlowSinkCall {
        calls: Arc<AtomicUsize>,
        polled_once: core::cell::Cell<bool>,
    }

    impl Future for SlowSinkCall {
        type Output = Result<(), SinkErr>;
        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            if self.polled_once.replace(true) {
                self.calls.fetch_add(1, Ordering::Relaxed);
                Poll::Ready(Ok(()))
            } else {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }

    struct SlowUnpinSink {
        calls: Arc<AtomicUsize>,
    }

    impl UnpinPipe for SlowUnpinSink {
        type In = Payload;
        type Out = ();
        type Err = SinkErr;

        fn call(&self, _input: Payload) -> impl Future<Output = Result<(), SinkErr>> + Unpin {
            SlowSinkCall {
                calls: Arc::clone(&self.calls),
                polled_once: core::cell::Cell::new(false),
            }
        }
    }

    #[test]
    fn unpin_fan_out_resumes_across_polls_for_every_sink() {
        let calls = Arc::new(AtomicUsize::new(0));
        let fan = FanOut::<_, AllOrNothing>::new(vec![
            SlowUnpinSink {
                calls: Arc::clone(&calls),
            },
            SlowUnpinSink {
                calls: Arc::clone(&calls),
            },
        ]);
        let mut call = UnpinPipe::call(&fan, Payload(1, Arc::new(AtomicUsize::new(0))));

        let mut polls = 0;
        loop {
            polls += 1;
            assert!(polls <= 10, "state machine looped without making progress");
            match poll_once(&mut call) {
                Poll::Ready(Ok(())) => break,
                Poll::Ready(Err(err)) => panic!("unexpected error {err:?}"),
                Poll::Pending => {}
            }
        }
        assert_eq!(
            calls.load(Ordering::Relaxed),
            2,
            "both sinks eventually ran"
        );
    }
}
