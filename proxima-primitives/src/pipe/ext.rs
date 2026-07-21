//! Fluent combinator sugar over the pipe algebra — [`PipeExt`], a single
//! blanket, own-tier extension trait over the root [`Pipe`] (sugar only,
//! never a capability — nothing gates on it) so `a.and_then(b)` reads
//! left-to-right instead of `AndThen::new(a, b)`.
//!
//! One trait, not four: every pipe in this crate implements the base [`Pipe`]
//! (the higher tiers — [`SendPipe`], [`UnpinPipe`], [`UnpinSendPipe`] — are
//! ADDITIVE constraints on top, never a replacement for it), so a single
//! `PipeExt: Pipe + Sized` blanket already reaches every pipe, and there is no
//! second trait providing the same method names to be ambiguous against. The
//! combinator VALUE `and_then`/`filter`/`fanout`/`fanin` build ([`AndThen`],
//! [`FanOut`], [`FanIn`]) carries whatever higher tiers its own stages
//! qualify for regardless of which trait constructed it — `SendPipe`/
//! `UnpinPipe`/`UnpinSendPipe`-ness is a property of the concrete types
//! involved, not of the call that built the value.
//!
//! `and_then` and `filter` are native: both build [`AndThen`] over two pipes.
//! `filter` gates `self` behind a predicate — `predicate` runs first, and
//! only an admitted item reaches `self` — matching the order
//! `FilterConfig::into_filter` already composes in (`predicate.and_then(inner)`,
//! see `pipe::filter`), just read fluently from the inner pipe's side instead
//! of the predicate's.
//!
//! `fanout` is native too: [`FanOut`] now carries a base `Pipe` impl (`pipe::
//! fanout`) alongside its original `SendPipe` one, so building it needs
//! nothing beyond `Pipe` + `Clone`.
//!
//! `fanin` carries one extra bound: [`FanIn`] only exists over [`UnpinPipe`]
//! sources shaped `In = (), Err = Exhausted` (`pipe::fan_in`) — that shape,
//! not the tier itself, is what's extra; a `Pipe` that also happens to be
//! such an `UnpinPipe` source can still call it.

use crate::pipe::fan_in::{Exhausted, FanIn, FanInStrategy};
use crate::pipe::primitives::{AndThen, Pipe, UnpinPipe};
use proxima_core::markers::DropSafe;

#[cfg(feature = "alloc")]
use crate::pipe::fanout::{AllOrNothing, FanOut};
#[cfg(feature = "alloc")]
use alloc::vec;

/// Fluent sugar over the root [`Pipe`]. See the module doc for what each
/// method builds.
pub trait PipeExt: Pipe + Sized {
    /// `self.and_then(next)` is `AndThen::new(self, next)` — reads
    /// left-to-right at the call site.
    fn and_then<Next>(self, next: Next) -> AndThen<Self, Next>
    where
        Next: Pipe<In = Self::Out>,
        Next::Err: From<Self::Err>,
    {
        AndThen::new(self, next)
    }

    /// Gate `self` behind `predicate`.
    fn filter<Pred>(self, predicate: Pred) -> AndThen<Pred, Self>
    where
        Pred: Pipe<Out = Self::In>,
        Self::Err: From<Pred::Err>,
    {
        AndThen::new(predicate, self)
    }

    /// Broadcast to `self` and `other`, all-or-nothing.
    #[cfg(feature = "alloc")]
    fn fanout(self, other: Self) -> FanOut<Self, AllOrNothing>
    where
        Self: Clone,
        Self::In: Clone,
    {
        FanOut::new(vec![self, other])
    }

    /// Merge `self` and `other` into a two-source [`FanIn`]. `FanIn` sources
    /// are `UnpinPipe`-shaped (`In = ()`, `Err = Exhausted`), hence the extra
    /// bound.
    fn fanin<Strategy>(self, other: Self, strategy: Strategy) -> FanIn<Self, Strategy, 2>
    where
        Self: UnpinPipe<In = (), Err = Exhausted> + DropSafe,
        Strategy: FanInStrategy,
    {
        FanIn::new([self, other], strategy)
    }
}

impl<P: Pipe> PipeExt for P {}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{FanInStrategy, Pipe, PipeExt};
    use core::future::Future;

    // dependency-free executor, matching `primitives.rs`'s own test helper —
    // no `proxima::test` dependency needed to prove the sugar layer works.
    fn block_on<Fut: Future>(future: Fut) -> Fut::Output {
        let mut pinned = core::pin::pin!(future);
        let mut context = core::task::Context::from_waker(core::task::Waker::noop());
        loop {
            if let core::task::Poll::Ready(output) = pinned.as_mut().poll(&mut context) {
                return output;
            }
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    struct Overflow;

    struct Increment;
    impl Pipe for Increment {
        type In = u64;
        type Out = u64;
        type Err = Overflow;
        fn call(&self, input: u64) -> impl Future<Output = Result<u64, Overflow>> {
            async move { input.checked_add(1).ok_or(Overflow) }
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    enum ChainError {
        Overflowed,
        Rejected,
    }
    impl From<Overflow> for ChainError {
        fn from(_: Overflow) -> Self {
            ChainError::Overflowed
        }
    }

    struct Double;
    impl Pipe for Double {
        type In = u64;
        type Out = u64;
        type Err = ChainError;
        fn call(&self, input: u64) -> impl Future<Output = Result<u64, ChainError>> {
            async move { Ok(input * 2) }
        }
    }

    // the predicate `.filter` gates the chain behind — rejects odd input
    // before `Increment`/`Double` ever run.
    struct RejectOdd;
    impl Pipe for RejectOdd {
        type In = u64;
        type Out = u64;
        type Err = ChainError;
        fn call(&self, input: u64) -> impl Future<Output = Result<u64, ChainError>> {
            async move {
                if input.is_multiple_of(2) {
                    Ok(input)
                } else {
                    Err(ChainError::Rejected)
                }
            }
        }
    }

    #[test]
    fn and_then_and_filter_compose_through_the_ext_sugar() {
        // `use PipeExt` (via `use super::*` above) is what makes `.and_then`
        // and `.filter` resolve here — no prelude in this crate.
        let pipeline = Increment.and_then(Double).filter(RejectOdd);

        let admitted = block_on(Pipe::call(&pipeline, 4));
        assert_eq!(admitted, Ok(10), "4 -> RejectOdd passes -> +1 -> *2 -> 10");

        let rejected = block_on(Pipe::call(&pipeline, 3));
        assert_eq!(
            rejected,
            Err(ChainError::Rejected),
            "3 is odd: RejectOdd stops the chain before Increment runs"
        );
    }

    #[cfg(feature = "alloc")]
    mod fanout_ext {
        use super::{Pipe, PipeExt};
        use alloc::sync::Arc;
        use core::convert::Infallible;
        use core::future::Future;
        use core::sync::atomic::{AtomicUsize, Ordering};

        #[derive(Clone)]
        struct RecordingSink {
            calls: Arc<AtomicUsize>,
        }

        impl Pipe for RecordingSink {
            type In = u32;
            type Out = ();
            type Err = Infallible;
            fn call(&self, _input: u32) -> impl Future<Output = Result<(), Infallible>> {
                self.calls.fetch_add(1, Ordering::Relaxed);
                async move { Ok(()) }
            }
        }

        #[test]
        fn fanout_ext_broadcasts_to_both_arms() {
            let calls = Arc::new(AtomicUsize::new(0));
            let fan = RecordingSink {
                calls: Arc::clone(&calls),
            }
            .fanout(RecordingSink {
                calls: Arc::clone(&calls),
            });

            let outcome = super::block_on(Pipe::call(&fan, 7));

            assert_eq!(outcome, Ok(()));
            assert_eq!(calls.load(Ordering::Relaxed), 2, "both arms ran");
        }
    }

    struct OnceSource {
        emitted: core::cell::Cell<bool>,
        value: u32,
    }
    impl proxima_core::markers::DropSafe for OnceSource {}
    impl super::UnpinPipe for OnceSource {
        type In = ();
        type Out = u32;
        type Err = super::Exhausted;
        fn call(&self, (): ()) -> impl Future<Output = Result<u32, super::Exhausted>> + Unpin {
            if self.emitted.replace(true) {
                core::future::ready(Err(super::Exhausted))
            } else {
                core::future::ready(Ok(self.value))
            }
        }
    }
    // base-tier mirror, delegating straight through — needed for `PipeExt`
    // (the `.fanin` sugar) to reach `OnceSource` at all.
    impl Pipe for OnceSource {
        type In = ();
        type Out = u32;
        type Err = super::Exhausted;
        fn call(&self, (): ()) -> impl Future<Output = Result<u32, super::Exhausted>> {
            super::UnpinPipe::call(self, ())
        }
    }

    #[test]
    fn fanin_ext_merges_both_sources_via_the_sugar() {
        let fan = (OnceSource {
            emitted: core::cell::Cell::new(false),
            value: 1,
        })
        .fanin(
            OnceSource {
                emitted: core::cell::Cell::new(false),
                value: 2,
            },
            super::super::fan_in::Select::RoundRobin,
        );

        let waker = core::task::Waker::noop();
        let mut context = core::task::Context::from_waker(waker);
        let mut merged = alloc_free_collect(&fan, &mut context);
        merged.sort_unstable();
        assert_eq!(merged, [1, 2], "both sources drain through the fanin sugar");
    }

    // no-alloc drain: a fixed-size buffer, since `FanIn` itself is no-alloc.
    fn alloc_free_collect<S, Strategy, const N: usize>(
        fan: &super::FanIn<S, Strategy, N>,
        cx: &mut core::task::Context<'_>,
    ) -> [u32; N]
    where
        S: super::UnpinPipe<In = (), Out = u32, Err = super::Exhausted>
            + proxima_core::markers::DropSafe,
        Strategy: FanInStrategy,
    {
        let mut out = [0u32; N];
        let mut count = 0;
        for _ in 0..(N * 4) {
            if count == N {
                break;
            }
            let mut call = Pipe::call(fan, ());
            match core::pin::Pin::new(&mut call).poll(cx) {
                core::task::Poll::Ready(Ok(value)) => {
                    out[count] = value;
                    count += 1;
                }
                core::task::Poll::Ready(Err(super::Exhausted)) => break,
                core::task::Poll::Pending => {}
            }
        }
        out
    }
}
