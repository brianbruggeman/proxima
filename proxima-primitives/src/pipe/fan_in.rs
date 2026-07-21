//! `FanIn<S, const N>` — sans-IO N→1 merge, itself a [`Pipe`]/[`UnpinPipe`].
//!
//! The fan-in counterpart to the fan-out family: N sources merged into one
//! item stream. Modeled as an explicit FSM because, unlike fan-out's
//! stateless `all`, a merge carries persistent cross-poll state — which
//! sources are still live, and a fairness cursor so one hot source cannot
//! starve the others.
//!
//! `FanIn` used to speak a bespoke protocol (`PollSource::poll_next(&mut
//! self, cx) -> Poll<Option<Item>>`) parallel to the pipe algebra. It now
//! IS a pipe: `Pipe::call(&self, ()) -> Result<S::Out, Exhausted>`. The
//! merged sources are `UnpinPipe<In = (), Err = Exhausted>` — a source
//! calls itself with nothing and produces an item, or resolves
//! [`Exhausted`] to say it will never produce again. Termination lives in
//! the `Err` channel instead of a second `Option`-shaped sentinel.
//!
//! TIER: this is the T0 floor — **no_std + no-alloc**. The sources live in a
//! `[S; N]` array (arity fixed at the type level), liveness is `[AtomicBool;
//! N]` (an atomic in place of `[bool; N]` because [`Pipe::call`] takes `&self`
//! — the merge's cross-poll state can no longer live behind `&mut self`), and
//! polling is `core::task`. No heap, no spawn, no channel — the kernel-bypass
//! merge shape (\*DK: merge N fixed NIC/NVMe queues with zero allocation), and
//! it tiers all the way down to bare metal.
//!
//! A runtime-arity no-alloc variant would back the sources with a
//! `heapless::Vec<S, CAP>` whose `CAP` is a build.rs/conflaguration sizing const
//! (the existing `RETRY_STATUS_CAP` pattern) — not built here; the const-`N`
//! form needs no build-time axis (the caller names the arity).
//!
//! `Item` (`S::Out`) is owned. The GAT lending form that makes the merge
//! zero-copy — the merged item borrowing into the producing source's ring
//! slot — is [`crate::pipe::drain_source::DrainFanIn`], a separate no_std
//! leaf built on the push-visitor model instead of this pull/`Pipe` one.
//!
//! ## Scan, don't race
//!
//! Each call to the merge's `call(())` future scans the live sources ONCE, in
//! [`Select`] order, and returns the first ready item. It does not drive `N`
//! sources concurrently and take a winner — that would be a `Race`/`Select`
//! combinator, a different (and heavier) primitive. A source whose `call(())`
//! is not yet ready is polled once, found `Pending`, and its in-flight future
//! is then DROPPED — the merge asks the source again (a fresh `call(())`) on
//! the next poll. This is why every merged source must be
//! [`proxima_core::markers::DropSafe`]: the source, not the transient call
//! future, is what registers the waker for "I have something now" — the call
//! future is disposable scaffolding around that registration, not the state
//! itself (see `proxima_core::signal::Fired` for the canonical shape: it
//! registers a waker slot with the level it observes and cleans the slot up
//! on `Drop`, so constructing a fresh one per scan is exactly as cheap and
//! correct as reusing one).

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use core::task::{Context, Poll};

use proxima_core::markers::DropSafe;

use crate::pipe::primitives::{Pipe, UnpinPipe, UnpinSendPipe};

/// A source's `call` will never produce again — the merge's termination
/// signal. Replaces the old `PollSource::poll_next` returning `Ready(None)`:
/// termination lives in the `Err` channel, so a merge is `Result<Out,
/// Exhausted>`, not a second `Poll<Option<..>>` protocol next to `Pipe`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("source exhausted: will never produce again")]
pub struct Exhausted;

/// Which ready source the merge takes next.
///
/// The primitive is "many sources → one, taking only what is ready". Choosing
/// *among* the ready is a strategy — a dial, not part of the merge — so it is
/// named at construction rather than welded into the merge. Same wiring, same
/// contract, different answer.
///
/// Priority is `Fifo` over an ordered array: put the sources in the order you
/// want them preferred. That is why there is no `Priority` arm — it would be a
/// second name for a choice you already made when you built the array.
///
/// # Why this is a trait and not a pipe
///
/// A strategy never sees an item. It answers a control question — which source
/// to try next — from the scan's own position; no payload passes through it, so
/// there is nothing for it to be a pipe *of*. Contrast a seam that DOES take
/// the item: that one must be a pipe, or it ends up answering with a `bool` and
/// growing companions to carry back the item and the reason it threw away.
///
/// The line, and it is readable straight off the signature: **if the item
/// passes through it, it is a pipe; if it only answers a control question and
/// never sees the item, it is a strategy — a plain function.** Picking an index
/// runs once per source per scan on the hot path; a pipe would build and poll a
/// future to compute a `usize`.
pub trait FanInStrategy {
    /// The source index to try at `step` of a scan over `n` sources that began
    /// at `start` — the cursor, one past whoever last emitted. Must return a
    /// value in `0..n`, and over `step` in `0..n` should visit each source once
    /// or a source can never be drained.
    fn index(&self, step: usize, start: usize, n: usize) -> usize;
}

/// The built-in strategies. The trait above is the open seam — implement it for
/// least-loaded, random, weighted, whatever the merge needs; these are the ones
/// that need no state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Select {
    /// Resume the scan past whoever last emitted, so a perpetually-ready source
    /// cannot starve the rest. Fair; no source is preferred.
    RoundRobin,
    /// Always scan from the first source: earlier sources win every tie. This
    /// is also priority order — order the array by priority.
    Fifo,
    /// Always scan from the last source: later sources win every tie.
    Lifo,
}

impl FanInStrategy for Select {
    fn index(&self, step: usize, start: usize, n: usize) -> usize {
        match self {
            Select::RoundRobin => (start + step) % n,
            Select::Fifo => step,
            Select::Lifo => n - 1 - step,
        }
    }
}

/// Fixed-arity N→1 merge over `[S; N]`, taking only sources that are ready.
/// Resolves [`Exhausted`] once every source has. No_std + no-alloc. Which
/// ready source wins is [`Select`], named by the caller. Itself a
/// [`Pipe`]/[`UnpinPipe`] (source form: `In = ()`), so a `FanIn` nests inside
/// a bigger `FanIn` with no adapter.
pub struct FanIn<S, Strategy, const N: usize> {
    sources: [S; N],
    live: [AtomicBool; N],
    remaining: AtomicUsize,
    cursor: AtomicUsize,
    strategy: Strategy,
}

impl<S, Strategy, const N: usize> FanIn<S, Strategy, N> {
    /// Merge `sources`, choosing among the ready ones by `strategy`. All start
    /// live; the merge ends when all have drained.
    #[must_use]
    pub fn new(sources: [S; N], strategy: Strategy) -> Self {
        Self {
            sources,
            live: core::array::from_fn(|_| AtomicBool::new(true)),
            remaining: AtomicUsize::new(N),
            cursor: AtomicUsize::new(0),
            strategy,
        }
    }

    /// The strategy this merge was built with.
    #[must_use]
    pub fn strategy(&self) -> &Strategy {
        &self.strategy
    }

    /// Sources not yet drained.
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.remaining.load(Ordering::Relaxed)
    }
}

/// The future behind `FanIn::call` — one scan pass over the live sources,
/// starting from the merge's cursor in [`Select`] order. `Unpin` because it
/// only ever borrows `fan` and holds no self-referential state — the whole
/// point of the `UnpinPipe` tier (see `primitives.rs`'s module doc): a caller
/// can `Pin::new(&mut call).poll(cx)` with no `unsafe`, no `Box`.
struct FanInCall<'fan, S, Strategy, const N: usize> {
    fan: &'fan FanIn<S, Strategy, N>,
}

impl<S, Strategy, const N: usize> Future for FanInCall<'_, S, Strategy, N>
where
    S: UnpinPipe<In = (), Err = Exhausted>,
    Strategy: FanInStrategy,
{
    type Output = Result<S::Out, Exhausted>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let fan = self.fan;
        if fan.remaining.load(Ordering::Relaxed) == 0 {
            return Poll::Ready(Err(Exhausted));
        }
        let cursor = fan.cursor.load(Ordering::Relaxed);
        // the strategy decides only WHERE the scan starts and which way it
        // walks; the merge itself is the same either way.
        for step in 0..N {
            let index = fan.strategy.index(step, cursor, N);
            if !fan.live[index].load(Ordering::Relaxed) {
                continue;
            }
            // a fresh call future per scan, polled once, then dropped — the
            // source (not this transient future) is what remembers readiness;
            // see the module doc's DropSafe note.
            let mut call = fan.sources[index].call(());
            match Pin::new(&mut call).poll(cx) {
                Poll::Ready(Ok(item)) => {
                    fan.cursor.store((index + 1) % N, Ordering::Relaxed);
                    return Poll::Ready(Ok(item));
                }
                Poll::Ready(Err(Exhausted)) => {
                    fan.live[index].store(false, Ordering::Relaxed);
                    let remaining = fan.remaining.fetch_sub(1, Ordering::Relaxed) - 1;
                    if remaining == 0 {
                        return Poll::Ready(Err(Exhausted));
                    }
                }
                Poll::Pending => {}
            }
        }
        // remaining > 0 and nothing emitted this pass: a fully-drained pass
        // would have hit `remaining == 0` above and returned, so at least one
        // live source returned Pending (and registered on itself, per the
        // module doc — not on the `call` future we just dropped).
        Poll::Pending
    }
}

/// The `UnpinSendPipe`-tier mirror of [`FanInCall`] — same one-scan-pass
/// algorithm, calling `UnpinSendPipe::call` instead of `UnpinPipe::call`. A
/// separate type, not a second `impl Future` on `FanInCall`: `UnpinPipe` and
/// `UnpinSendPipe` are standalone traits (a source can implement one, both,
/// or neither), so a source satisfying both would make two `Future` impls on
/// the same concrete `FanInCall` overlap (E0119) — coherence needs its own
/// struct per tier, same as `AndThen`'s and `FanOut`'s separate `Pipe`/
/// `SendPipe` impl bodies.
struct FanInSendCall<'fan, S, Strategy, const N: usize> {
    fan: &'fan FanIn<S, Strategy, N>,
}

impl<S, Strategy, const N: usize> Future for FanInSendCall<'_, S, Strategy, N>
where
    S: UnpinSendPipe<In = (), Err = Exhausted>,
    Strategy: FanInStrategy,
{
    type Output = Result<S::Out, Exhausted>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let fan = self.fan;
        if fan.remaining.load(Ordering::Relaxed) == 0 {
            return Poll::Ready(Err(Exhausted));
        }
        let cursor = fan.cursor.load(Ordering::Relaxed);
        for step in 0..N {
            let index = fan.strategy.index(step, cursor, N);
            if !fan.live[index].load(Ordering::Relaxed) {
                continue;
            }
            let mut call = UnpinSendPipe::call(&fan.sources[index], ());
            match Pin::new(&mut call).poll(cx) {
                Poll::Ready(Ok(item)) => {
                    fan.cursor.store((index + 1) % N, Ordering::Relaxed);
                    return Poll::Ready(Ok(item));
                }
                Poll::Ready(Err(Exhausted)) => {
                    fan.live[index].store(false, Ordering::Relaxed);
                    let remaining = fan.remaining.fetch_sub(1, Ordering::Relaxed) - 1;
                    if remaining == 0 {
                        return Poll::Ready(Err(Exhausted));
                    }
                }
                Poll::Pending => {}
            }
        }
        Poll::Pending
    }
}

impl<S, Strategy, const N: usize> Pipe for FanIn<S, Strategy, N>
where
    S: UnpinPipe<In = (), Err = Exhausted> + DropSafe,
    Strategy: FanInStrategy,
{
    type In = ();
    type Out = S::Out;
    type Err = Exhausted;

    fn call(&self, (): ()) -> impl Future<Output = Result<Self::Out, Exhausted>> {
        FanInCall { fan: self }
    }
}

impl<S, Strategy, const N: usize> UnpinPipe for FanIn<S, Strategy, N>
where
    S: UnpinPipe<In = (), Err = Exhausted> + DropSafe,
    Strategy: FanInStrategy,
{
    type In = ();
    type Out = S::Out;
    type Err = Exhausted;

    fn call(&self, (): ()) -> impl Future<Output = Result<Self::Out, Exhausted>> + Unpin {
        FanInCall { fan: self }
    }
}

impl<S, Strategy, const N: usize> UnpinSendPipe for FanIn<S, Strategy, N>
where
    S: UnpinSendPipe<In = (), Err = Exhausted> + DropSafe,
    Strategy: FanInStrategy + Send + Sync + 'static,
{
    type In = ();
    type Out = S::Out;
    type Err = Exhausted;

    fn call(&self, (): ()) -> impl Future<Output = Result<Self::Out, Exhausted>> + Send + Unpin {
        FanInSendCall { fan: self }
    }
}

// dropping an in-flight `FanInCall` mid-scan leaves no observable partial
// state (it has only read atomics and dropped whichever source `call` future
// it was mid-poll of, which is safe precisely because that source is itself
// `DropSafe`) — so a `FanIn` of `DropSafe` sources is itself `DropSafe`,
// which is what lets one nest inside an outer `FanIn` (the outer's `S` bound
// demands it).
impl<S: DropSafe, Strategy, const N: usize> DropSafe for FanIn<S, Strategy, N> {}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use core::task::Waker;

    #[derive(Clone, Copy)]
    enum Step {
        Yield(u32),
        Pend,
        Done,
    }

    // a source driven by a fixed script of call outcomes — each call consumes
    // one step, so `Pend` then a later `Yield` exercises the not-drained path.
    // `pos` is atomic because `UnpinPipe::call` takes `&self`.
    struct Script<const M: usize> {
        steps: [Step; M],
        pos: AtomicUsize,
    }

    impl<const M: usize> Script<M> {
        fn new(steps: [Step; M]) -> Self {
            Self {
                steps,
                pos: AtomicUsize::new(0),
            }
        }
    }

    impl<const M: usize> DropSafe for Script<M> {}

    // resolves immediately to a fixed `Poll` value (never truly pends across
    // polls) — the hand-written poll struct an `UnpinPipe::call` needs in
    // place of an `!Unpin` async block.
    struct ScriptCall(Poll<Result<u32, Exhausted>>);

    impl Future for ScriptCall {
        type Output = Result<u32, Exhausted>;

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            self.0
        }
    }

    impl<const M: usize> UnpinPipe for Script<M> {
        type In = ();
        type Out = u32;
        type Err = Exhausted;

        fn call(&self, (): ()) -> impl Future<Output = Result<u32, Exhausted>> + Unpin {
            let pos = self.pos.load(Ordering::Relaxed);
            if pos >= M {
                return ScriptCall(Poll::Ready(Err(Exhausted)));
            }
            let step = self.steps[pos];
            self.pos.store(pos + 1, Ordering::Relaxed);
            match step {
                Step::Yield(value) => ScriptCall(Poll::Ready(Ok(value))),
                Step::Pend => ScriptCall(Poll::Pending),
                Step::Done => ScriptCall(Poll::Ready(Err(Exhausted))),
            }
        }
    }

    // `ScriptCall` only ever holds a `Poll<Result<u32, Exhausted>>` value —
    // trivially `Send` — so `Script` reaches the `UnpinSendPipe` tier too,
    // proving `FanIn::UnpinSendPipe` (this file's Stage 2 addition).
    impl<const M: usize> UnpinSendPipe for Script<M> {
        type In = ();
        type Out = u32;
        type Err = Exhausted;

        fn call(&self, (): ()) -> impl Future<Output = Result<u32, Exhausted>> + Send + Unpin {
            let pos = self.pos.load(Ordering::Relaxed);
            if pos >= M {
                return ScriptCall(Poll::Ready(Err(Exhausted)));
            }
            let step = self.steps[pos];
            self.pos.store(pos + 1, Ordering::Relaxed);
            match step {
                Step::Yield(value) => ScriptCall(Poll::Ready(Ok(value))),
                Step::Pend => ScriptCall(Poll::Pending),
                Step::Done => ScriptCall(Poll::Ready(Err(Exhausted))),
            }
        }
    }

    // drive a fan-in to completion into a fixed buffer (no-alloc); returns count.
    fn drain<S, Strategy, const N: usize>(fan: FanIn<S, Strategy, N>, out: &mut [u32]) -> usize
    where
        S: UnpinPipe<In = (), Out = u32, Err = Exhausted> + DropSafe,
        Strategy: FanInStrategy,
    {
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        let mut count = 0;
        for _ in 0..10_000 {
            let mut call = Pipe::call(&fan, ());
            match Pin::new(&mut call).poll(&mut cx) {
                Poll::Ready(Ok(value)) => {
                    out[count] = value;
                    count += 1;
                }
                Poll::Ready(Err(Exhausted)) => break,
                Poll::Pending => {}
            }
        }
        count
    }

    #[test]
    fn merges_all_sources_in_round_robin_order() {
        let fan = FanIn::new(
            [
                Script::new([Step::Yield(0), Step::Yield(1), Step::Done]),
                Script::new([Step::Yield(10), Step::Yield(11), Step::Done]),
                Script::new([Step::Yield(20), Step::Yield(21), Step::Done]),
            ],
            Select::RoundRobin,
        );
        let mut buf = [0u32; 16];
        let count = drain(fan, &mut buf);
        assert_eq!(
            &buf[..count],
            &[0, 10, 20, 1, 11, 21],
            "round-robin fairness"
        );
    }

    #[test]
    fn drained_source_is_skipped() {
        let fan = FanIn::new(
            [
                Script::new([Step::Done, Step::Done, Step::Done]),
                Script::new([Step::Yield(1), Step::Yield(2), Step::Done]),
                Script::new([Step::Yield(3), Step::Done, Step::Done]),
            ],
            Select::RoundRobin,
        );
        let mut buf = [0u32; 16];
        let count = drain(fan, &mut buf);
        let got = &mut buf[..count];
        got.sort_unstable();
        assert_eq!(
            got,
            &[1, 2, 3],
            "items from live sources, drained one skipped"
        );
    }

    // the strategy is load-bearing, not decoration: same sources, same merge,
    // three dials, three different orders. Fifo prefers the earliest source
    // (== priority order), Lifo the latest, RoundRobin nobody.
    /// The trait is the open seam: a strategy the library never heard of.
    /// Pins one source first, then falls back to round-robin — the "sticky
    /// primary" shape, defined entirely by the caller.
    struct StickyThen(usize);
    impl FanInStrategy for StickyThen {
        fn index(&self, step: usize, start: usize, n: usize) -> usize {
            if step == 0 {
                self.0 % n
            } else {
                (start + step) % n
            }
        }
    }

    #[test]
    fn a_caller_defined_strategy_drives_the_merge() {
        let fan = FanIn::new(
            [
                Script::new([Step::Yield(0), Step::Done]),
                Script::new([Step::Yield(10), Step::Done]),
                Script::new([Step::Yield(20), Step::Done]),
            ],
            StickyThen(2),
        );
        let mut buf = [0u32; 8];
        let count = drain(fan, &mut buf);
        assert_eq!(count, 3, "every source still drains");
        assert_eq!(
            buf[0], 20,
            "the caller's own strategy picked source #2 first"
        );
    }

    #[test]
    fn select_decides_which_ready_source_wins() {
        fn drain_with(select: Select) -> [u32; 3] {
            let fan = FanIn::new(
                [
                    Script::new([Step::Yield(0), Step::Done]),
                    Script::new([Step::Yield(10), Step::Done]),
                    Script::new([Step::Yield(20), Step::Done]),
                ],
                select,
            );
            let mut buf = [0u32; 8];
            let count = drain(fan, &mut buf);
            assert_eq!(count, 3, "every source yields exactly one item");
            [buf[0], buf[1], buf[2]]
        }

        assert_eq!(
            drain_with(Select::Fifo),
            [0, 10, 20],
            "earliest source first"
        );
        assert_eq!(drain_with(Select::Lifo), [20, 10, 0], "latest source first");
        assert_eq!(
            drain_with(Select::RoundRobin),
            [0, 10, 20],
            "fair: the cursor steps past whoever just emitted"
        );
    }

    #[test]
    fn all_done_terminates_immediately() {
        let fan = FanIn::new(
            [Script::new([Step::Done]), Script::new([Step::Done])],
            Select::RoundRobin,
        );
        let mut buf = [0u32; 4];
        assert_eq!(drain(fan, &mut buf), 0);
    }

    #[test]
    fn pending_source_is_not_drained() {
        let fan = FanIn::new(
            [Script::new([Step::Pend, Step::Yield(7), Step::Done])],
            Select::RoundRobin,
        );
        let mut buf = [0u32; 4];
        let count = drain(fan, &mut buf);
        assert_eq!(
            &buf[..count],
            &[7],
            "Pending re-polled, not treated as drained"
        );
    }

    #[test]
    fn live_count_tracks_draining() {
        let fan = FanIn::new(
            [Script::new([Step::Yield(1)]), Script::new([Step::Yield(2)])],
            Select::RoundRobin,
        );
        assert_eq!(fan.live_count(), 2);
    }

    // compile-time proof: FanIn nests inside a bigger FanIn with no adapter —
    // it needs to be UnpinPipe<In = (), Err = Exhausted> AND DropSafe itself.
    #[test]
    fn fan_in_nests_inside_a_bigger_fan_in() {
        let inner_a = FanIn::new(
            [Script::new([Step::Yield(1), Step::Done])],
            Select::RoundRobin,
        );
        let inner_b = FanIn::new(
            [Script::new([Step::Yield(2), Step::Done])],
            Select::RoundRobin,
        );
        let outer = FanIn::new([inner_a, inner_b], Select::RoundRobin);
        let mut buf = [0u32; 4];
        let count = drain(outer, &mut buf);
        let got = &mut buf[..count];
        got.sort_unstable();
        assert_eq!(
            got,
            &[1, 2],
            "both nested fan-ins drain through the outer merge"
        );
    }

    // ── UnpinSendPipe tier (Stage 2) ────────────────────────────────────────

    // `UnpinSendPipe::call`'s merge loop is `FanInSendCall`, a separate type
    // from `FanInCall` (coherence: `UnpinPipe`/`UnpinSendPipe` are standalone
    // traits, see its doc) — drive it through the `Send` entry point
    // specifically, not `Pipe`/`UnpinPipe`, to prove that path for real.
    fn drain_send<S, Strategy, const N: usize>(fan: FanIn<S, Strategy, N>, out: &mut [u32]) -> usize
    where
        S: UnpinSendPipe<In = (), Out = u32, Err = Exhausted> + DropSafe,
        Strategy: FanInStrategy + Send + Sync + 'static,
    {
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        let mut count = 0;
        for _ in 0..10_000 {
            let mut call = UnpinSendPipe::call(&fan, ());
            match Pin::new(&mut call).poll(&mut cx) {
                Poll::Ready(Ok(value)) => {
                    out[count] = value;
                    count += 1;
                }
                Poll::Ready(Err(Exhausted)) => break,
                Poll::Pending => {}
            }
        }
        count
    }

    #[test]
    fn unpin_send_pipe_merges_all_sources_in_round_robin_order() {
        let fan = FanIn::new(
            [
                Script::new([Step::Yield(0), Step::Yield(1), Step::Done]),
                Script::new([Step::Yield(10), Step::Yield(11), Step::Done]),
                Script::new([Step::Yield(20), Step::Yield(21), Step::Done]),
            ],
            Select::RoundRobin,
        );
        let mut buf = [0u32; 16];
        let count = drain_send(fan, &mut buf);
        assert_eq!(
            &buf[..count],
            &[0, 10, 20, 1, 11, 21],
            "same round-robin fairness as the UnpinPipe tier"
        );
    }

    #[test]
    fn unpin_send_pipe_future_is_send_and_unpin() {
        fn needs_send_unpin<F: Future + Send + Unpin>(_: &F) {}
        let fan = FanIn::new([Script::new([Step::Yield(1), Step::Done])], Select::Fifo);
        let call = UnpinSendPipe::call(&fan, ());
        needs_send_unpin(&call);
    }
}
