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
//! [`FanInVec`] is the runtime-arity no-alloc variant: sources live in a
//! `heapless::Vec<S, FAN_IN_SOURCE_CAP>` whose cap is a build.rs/conflaguration
//! sizing const (the `RETRY_STATUS_CAP` pattern in `retry_rules.rs`, mirrored
//! here). The const-`N` form above needs no build-time axis (the caller names
//! the arity at the type level); `FanInVec` is for the case where the arity is
//! only known at runtime (a CLI/config-driven source count) and still must not
//! allocate.
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

use crate::pipe::fan_in_sized::FAN_IN_SOURCE_CAP;
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

/// How many retirements the merge needs before it resolves [`Exhausted`].
/// Orthogonal to [`FanInStrategy`]: that trait answers "which source next";
/// this one answers "how many must complete" — a control question about the
/// scan's STOPPING point, not its ORDER, so it is its own dial rather than a
/// second method grafted onto `FanInStrategy` (a caller wanting
/// "round-robin order, quorum of 2" would otherwise need one combined type
/// per `(order, count)` pairing instead of composing two independent ones).
///
/// Called after every retirement with `retired` (sources that have resolved
/// [`Exhausted`]) and the merge's total `n`. Once it answers `true`, the
/// merge resolves `Err(Exhausted)` even if sources remain live — they are
/// simply never polled again.
pub trait FanInCompletion {
    fn satisfied(&self, retired: usize, n: usize) -> bool;
}

/// End the merge once `self.0` sources have retired, leaving any others
/// live but unpolled. The capability [`FanInStrategy::index`] cannot express:
/// picking an index says nothing about how many retirements are enough.
///
/// "Every source must retire" is not a second type: it is this one with
/// `self.0` bound to the merge's own arity. `retired >= n` and `retired ==
/// n` agree for every `retired` in `0..=n`, because a scan never retires
/// more than `n` sources — so [`FanIn::new`] builds exactly `Quorum(N)`
/// (`N` the const-generic arity) as its default completion, rather than a
/// parallel "all-must-arrive" type expressing the same policy twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Quorum(pub usize);

impl FanInCompletion for Quorum {
    fn satisfied(&self, retired: usize, _n: usize) -> bool {
        retired >= self.0
    }
}

/// Fixed-arity N→1 merge over `[S; N]`, taking only sources that are ready.
/// Resolves [`Exhausted`] once [`FanInCompletion`] is satisfied (every source
/// retired, by default). No_std + no-alloc. Which ready source wins is
/// [`Select`], named by the caller. Itself a [`Pipe`]/[`UnpinPipe`] (source
/// form: `In = ()`), so a `FanIn` nests inside a bigger `FanIn` with no
/// adapter.
pub struct FanIn<S, Strategy, const N: usize, Completion = Quorum> {
    sources: [S; N],
    live: [AtomicBool; N],
    remaining: AtomicUsize,
    cursor: AtomicUsize,
    strategy: Strategy,
    completion: Completion,
}

impl<S, Strategy, const N: usize> FanIn<S, Strategy, N, Quorum> {
    /// Merge `sources`, choosing among the ready ones by `strategy`. All start
    /// live; the merge ends when every source has drained — `Quorum(N)`, the
    /// arity `N` already carries at the type level, so "all must arrive"
    /// needs no threshold the caller has to repeat. Use
    /// [`FanIn::with_completion`] to name a lower threshold, such as
    /// `Quorum(2)` for a 3-source merge.
    #[must_use]
    pub fn new(sources: [S; N], strategy: Strategy) -> Self {
        Self::with_completion(sources, strategy, Quorum(N))
    }
}

impl<S, Strategy, const N: usize, Completion> FanIn<S, Strategy, N, Completion> {
    /// Merge `sources`, choosing among the ready ones by `strategy`, ending
    /// once `completion` is satisfied.
    #[must_use]
    pub fn with_completion(sources: [S; N], strategy: Strategy, completion: Completion) -> Self {
        Self {
            sources,
            live: core::array::from_fn(|_| AtomicBool::new(true)),
            remaining: AtomicUsize::new(N),
            cursor: AtomicUsize::new(0),
            strategy,
            completion,
        }
    }

    /// The strategy this merge was built with.
    #[must_use]
    pub fn strategy(&self) -> &Strategy {
        &self.strategy
    }

    /// The completion policy this merge was built with.
    #[must_use]
    pub fn completion(&self) -> &Completion {
        &self.completion
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
struct FanInCall<'fan, S, Strategy, const N: usize, Completion = Quorum> {
    fan: &'fan FanIn<S, Strategy, N, Completion>,
}

impl<S, Strategy, const N: usize, Completion> Future for FanInCall<'_, S, Strategy, N, Completion>
where
    S: UnpinPipe<In = (), Err = Exhausted>,
    Strategy: FanInStrategy,
    Completion: FanInCompletion,
{
    type Output = Result<S::Out, Exhausted>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let fan = self.fan;
        let retired = N - fan.remaining.load(Ordering::Relaxed);
        if fan.completion.satisfied(retired, N) {
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
                    let retired = N - remaining;
                    if fan.completion.satisfied(retired, N) {
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
struct FanInSendCall<'fan, S, Strategy, const N: usize, Completion = Quorum> {
    fan: &'fan FanIn<S, Strategy, N, Completion>,
}

impl<S, Strategy, const N: usize, Completion> Future
    for FanInSendCall<'_, S, Strategy, N, Completion>
where
    S: UnpinSendPipe<In = (), Err = Exhausted>,
    Strategy: FanInStrategy,
    Completion: FanInCompletion,
{
    type Output = Result<S::Out, Exhausted>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let fan = self.fan;
        let retired = N - fan.remaining.load(Ordering::Relaxed);
        if fan.completion.satisfied(retired, N) {
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
                    let retired = N - remaining;
                    if fan.completion.satisfied(retired, N) {
                        return Poll::Ready(Err(Exhausted));
                    }
                }
                Poll::Pending => {}
            }
        }
        Poll::Pending
    }
}

impl<S, Strategy, const N: usize, Completion> Pipe for FanIn<S, Strategy, N, Completion>
where
    S: UnpinPipe<In = (), Err = Exhausted> + DropSafe,
    Strategy: FanInStrategy,
    Completion: FanInCompletion,
{
    type In = ();
    type Out = S::Out;
    type Err = Exhausted;

    fn call(&self, (): ()) -> impl Future<Output = Result<Self::Out, Exhausted>> {
        FanInCall { fan: self }
    }
}

impl<S, Strategy, const N: usize, Completion> UnpinPipe for FanIn<S, Strategy, N, Completion>
where
    S: UnpinPipe<In = (), Err = Exhausted> + DropSafe,
    Strategy: FanInStrategy,
    Completion: FanInCompletion,
{
    type In = ();
    type Out = S::Out;
    type Err = Exhausted;

    fn call(&self, (): ()) -> impl Future<Output = Result<Self::Out, Exhausted>> + Unpin {
        FanInCall { fan: self }
    }
}

impl<S, Strategy, const N: usize, Completion> UnpinSendPipe for FanIn<S, Strategy, N, Completion>
where
    S: UnpinSendPipe<In = (), Err = Exhausted> + DropSafe,
    Strategy: FanInStrategy + Send + Sync + 'static,
    Completion: FanInCompletion + Send + Sync + 'static,
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
impl<S: DropSafe, Strategy, const N: usize, Completion> DropSafe for FanIn<S, Strategy, N, Completion> {}

/// `FanInVec::new` rejected a source count over `FAN_IN_SOURCE_CAP`. Reported,
/// never silently truncated — a dropped source would be a silently lost stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("source count {attempted} exceeds fan-in capacity {capacity}")]
pub struct CapacityExceeded {
    /// How many sources the caller tried to merge.
    pub attempted: usize,
    /// `FAN_IN_SOURCE_CAP`, the ceiling that was exceeded.
    pub capacity: usize,
}

/// Runtime-arity N→1 merge — the variant this module's doc names as missing.
/// [`FanIn`] fixes arity at the type level (`[S; N]`); this backs the sources
/// with `heapless::Vec<S, FAN_IN_SOURCE_CAP>` so the caller names the count at
/// construction (a CLI/config-driven source set) instead of at compile time.
/// Still no_std + no-alloc: no heap, no spawn, no channel. Same contract as
/// `FanIn` — `Pipe`/`UnpinPipe`/`UnpinSendPipe` with `In = ()`, `Out = S::Out`,
/// `Err = Exhausted` — and the same scan-don't-race fairness (see the module
/// doc's "Scan, don't race" section; the algorithm is identical, only the
/// backing store and the runtime length differ).
pub struct FanInVec<S, Strategy> {
    sources: heapless::Vec<S, FAN_IN_SOURCE_CAP>,
    live: heapless::Vec<AtomicBool, FAN_IN_SOURCE_CAP>,
    remaining: AtomicUsize,
    cursor: AtomicUsize,
    strategy: Strategy,
}

impl<S, Strategy> FanInVec<S, Strategy> {
    /// Merge `sources`, choosing among the ready ones by `strategy`. All start
    /// live; the merge ends when all have drained. `sources` must know its own
    /// length ([`ExactSizeIterator`]) so an over-capacity count is caught
    /// before anything is pushed — an array, `Vec`, or `heapless::Vec` all
    /// qualify. More than `FAN_IN_SOURCE_CAP` sources is [`CapacityExceeded`],
    /// not truncation: zero sources is valid and resolves [`Exhausted`] on the
    /// first call.
    pub fn new<I>(sources: I, strategy: Strategy) -> Result<Self, CapacityExceeded>
    where
        I: IntoIterator<Item = S>,
        I::IntoIter: ExactSizeIterator,
    {
        let iter = sources.into_iter();
        let attempted = iter.len();
        if attempted > FAN_IN_SOURCE_CAP {
            return Err(CapacityExceeded {
                attempted,
                capacity: FAN_IN_SOURCE_CAP,
            });
        }
        let mut backing: heapless::Vec<S, FAN_IN_SOURCE_CAP> = heapless::Vec::new();
        let mut live: heapless::Vec<AtomicBool, FAN_IN_SOURCE_CAP> = heapless::Vec::new();
        for source in iter {
            backing.push(source).map_err(|_| CapacityExceeded {
                attempted,
                capacity: FAN_IN_SOURCE_CAP,
            })?;
            live.push(AtomicBool::new(true)).map_err(|_| CapacityExceeded {
                attempted,
                capacity: FAN_IN_SOURCE_CAP,
            })?;
        }
        let count = backing.len();
        Ok(Self {
            sources: backing,
            live,
            remaining: AtomicUsize::new(count),
            cursor: AtomicUsize::new(0),
            strategy,
        })
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

/// The future behind `FanInVec::call` — same one-scan-pass algorithm as
/// [`FanInCall`], over `fan.sources.len()` instead of a const `N`.
struct FanInVecCall<'fan, S, Strategy> {
    fan: &'fan FanInVec<S, Strategy>,
}

impl<S, Strategy> Future for FanInVecCall<'_, S, Strategy>
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
        let n = fan.sources.len();
        let cursor = fan.cursor.load(Ordering::Relaxed);
        for step in 0..n {
            let index = fan.strategy.index(step, cursor, n);
            if !fan.live[index].load(Ordering::Relaxed) {
                continue;
            }
            let mut call = fan.sources[index].call(());
            match Pin::new(&mut call).poll(cx) {
                Poll::Ready(Ok(item)) => {
                    fan.cursor.store((index + 1) % n, Ordering::Relaxed);
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

/// The `UnpinSendPipe`-tier mirror of [`FanInVecCall`], same relationship as
/// [`FanInSendCall`] to [`FanInCall`].
struct FanInVecSendCall<'fan, S, Strategy> {
    fan: &'fan FanInVec<S, Strategy>,
}

impl<S, Strategy> Future for FanInVecSendCall<'_, S, Strategy>
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
        let n = fan.sources.len();
        let cursor = fan.cursor.load(Ordering::Relaxed);
        for step in 0..n {
            let index = fan.strategy.index(step, cursor, n);
            if !fan.live[index].load(Ordering::Relaxed) {
                continue;
            }
            let mut call = UnpinSendPipe::call(&fan.sources[index], ());
            match Pin::new(&mut call).poll(cx) {
                Poll::Ready(Ok(item)) => {
                    fan.cursor.store((index + 1) % n, Ordering::Relaxed);
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

impl<S, Strategy> Pipe for FanInVec<S, Strategy>
where
    S: UnpinPipe<In = (), Err = Exhausted> + DropSafe,
    Strategy: FanInStrategy,
{
    type In = ();
    type Out = S::Out;
    type Err = Exhausted;

    fn call(&self, (): ()) -> impl Future<Output = Result<Self::Out, Exhausted>> {
        FanInVecCall { fan: self }
    }
}

impl<S, Strategy> UnpinPipe for FanInVec<S, Strategy>
where
    S: UnpinPipe<In = (), Err = Exhausted> + DropSafe,
    Strategy: FanInStrategy,
{
    type In = ();
    type Out = S::Out;
    type Err = Exhausted;

    fn call(&self, (): ()) -> impl Future<Output = Result<Self::Out, Exhausted>> + Unpin {
        FanInVecCall { fan: self }
    }
}

impl<S, Strategy> UnpinSendPipe for FanInVec<S, Strategy>
where
    S: UnpinSendPipe<In = (), Err = Exhausted> + DropSafe,
    Strategy: FanInStrategy + Send + Sync + 'static,
{
    type In = ();
    type Out = S::Out;
    type Err = Exhausted;

    fn call(&self, (): ()) -> impl Future<Output = Result<Self::Out, Exhausted>> + Send + Unpin {
        FanInVecSendCall { fan: self }
    }
}

// same reasoning as `FanIn`'s `DropSafe` impl: dropping an in-flight
// `FanInVecCall` mid-scan leaves no observable partial state, so a `FanInVec`
// of `DropSafe` sources is itself `DropSafe`.
impl<S: DropSafe, Strategy> DropSafe for FanInVec<S, Strategy> {}

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
    fn drain<S, Strategy, const N: usize, Completion>(
        fan: FanIn<S, Strategy, N, Completion>,
        out: &mut [u32],
    ) -> usize
    where
        S: UnpinPipe<In = (), Out = u32, Err = Exhausted> + DropSafe,
        Strategy: FanInStrategy,
        Completion: FanInCompletion,
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

    #[test]
    fn quorum_completion_ends_the_merge_before_every_source_retires() {
        let fan = FanIn::with_completion(
            [
                Script::new([Step::Yield(1), Step::Done]),
                Script::new([Step::Yield(2), Step::Done]),
                // "never retires" for the two calls this test actually makes
                // it before quorum ends the merge — an array element must share
                // the other sources' concrete type (`[S; N]` is homogeneous, no
                // trait object), so a perpetual-Pend Script stands in for a
                // dedicated never-retiring source rather than minting one.
                Script::new([Step::Pend, Step::Pend]),
            ],
            Select::RoundRobin,
            Quorum(2),
        );
        let mut buf = [0u32; 8];
        let count = drain(fan, &mut buf);
        assert_eq!(
            count, 2,
            "both live sources yield exactly once before quorum ends the merge"
        );
    }

    // proves the binding claim behind deleting `AllMustArrive`: the
    // "all-must-arrive" default `FanIn::new` builds is not a distinct policy,
    // it is `Quorum` bound to the merge's own arity — an explicit
    // `Quorum(N)` for an N-source merge must behave identically to `new`'s
    // default, for every select order.
    #[test]
    fn explicit_quorum_at_arity_matches_the_default_for_every_select_order() {
        fn drain_default(select: Select) -> [u32; 3] {
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

        fn drain_explicit_quorum_at_arity(select: Select) -> [u32; 3] {
            let fan = FanIn::with_completion(
                [
                    Script::new([Step::Yield(0), Step::Done]),
                    Script::new([Step::Yield(10), Step::Done]),
                    Script::new([Step::Yield(20), Step::Done]),
                ],
                select,
                Quorum(3),
            );
            let mut buf = [0u32; 8];
            let count = drain(fan, &mut buf);
            assert_eq!(count, 3, "every source yields exactly one item");
            [buf[0], buf[1], buf[2]]
        }

        for select in [Select::Fifo, Select::Lifo, Select::RoundRobin] {
            assert_eq!(
                drain_default(select),
                drain_explicit_quorum_at_arity(select),
                "FanIn::new's default completion is Quorum bound to the arity"
            );
        }
    }

    // ── UnpinSendPipe tier (Stage 2) ────────────────────────────────────────

    // `UnpinSendPipe::call`'s merge loop is `FanInSendCall`, a separate type
    // from `FanInCall` (coherence: `UnpinPipe`/`UnpinSendPipe` are standalone
    // traits, see its doc) — drive it through the `Send` entry point
    // specifically, not `Pipe`/`UnpinPipe`, to prove that path for real.
    fn drain_send<S, Strategy, const N: usize, Completion>(
        fan: FanIn<S, Strategy, N, Completion>,
        out: &mut [u32],
    ) -> usize
    where
        S: UnpinSendPipe<In = (), Out = u32, Err = Exhausted> + DropSafe,
        Strategy: FanInStrategy + Send + Sync + 'static,
        Completion: FanInCompletion + Send + Sync + 'static,
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

    // ── FanInVec: runtime-arity variant ─────────────────────────────────────

    // drive a runtime-arity fan-in to completion into a fixed buffer; returns count.
    fn drain_vec<S, Strategy>(fan: &FanInVec<S, Strategy>, out: &mut [u32]) -> usize
    where
        S: UnpinPipe<In = (), Out = u32, Err = Exhausted> + DropSafe,
        Strategy: FanInStrategy,
    {
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        let mut count = 0;
        for _ in 0..10_000 {
            let mut call = Pipe::call(fan, ());
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

    // like `drain_vec`, but stops after `target` items instead of draining —
    // for checking fairness ordering while a hot source is still live.
    fn take_vec<S, Strategy>(fan: &FanInVec<S, Strategy>, out: &mut [u32], target: usize) -> usize
    where
        S: UnpinPipe<In = (), Out = u32, Err = Exhausted> + DropSafe,
        Strategy: FanInStrategy,
    {
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        let mut count = 0;
        while count < target {
            let mut call = Pipe::call(fan, ());
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
    fn vec_merges_all_sources_in_round_robin_order() {
        let fan = FanInVec::new(
            [
                Script::new([Step::Yield(0), Step::Yield(1), Step::Done]),
                Script::new([Step::Yield(10), Step::Yield(11), Step::Done]),
                Script::new([Step::Yield(20), Step::Yield(21), Step::Done]),
            ],
            Select::RoundRobin,
        )
        .expect("3 sources fit the default cap");
        let mut buf = [0u32; 16];
        let count = drain_vec(&fan, &mut buf);
        assert_eq!(
            &buf[..count],
            &[0, 10, 20, 1, 11, 21],
            "round-robin fairness, same as the fixed-arity form"
        );
    }

    #[test]
    fn vec_a_hot_source_does_not_starve_the_others() {
        // source #0 has a long backlog (always ready for many calls); sources
        // #1 and #2 have exactly one item, then pad with `Done` to match the
        // array's fixed width. Round-robin fairness means the hot source does
        // not get a second turn before the other two are served.
        let fan = FanInVec::new(
            [
                Script::new([Step::Yield(0), Step::Yield(1), Step::Yield(2), Step::Yield(3)]),
                Script::new([Step::Yield(100), Step::Done, Step::Done, Step::Done]),
                Script::new([Step::Yield(200), Step::Done, Step::Done, Step::Done]),
            ],
            Select::RoundRobin,
        )
        .expect("3 sources fit the default cap");
        let mut buf = [0u32; 8];
        let count = take_vec(&fan, &mut buf, 3);
        assert_eq!(
            &buf[..count],
            &[0, 100, 200],
            "every source is visited once before the hot source's second item"
        );
    }

    #[test]
    fn vec_drained_source_is_dropped_and_merge_continues() {
        let fan = FanInVec::new(
            [
                Script::new([Step::Done, Step::Done, Step::Done]),
                Script::new([Step::Yield(1), Step::Yield(2), Step::Done]),
                Script::new([Step::Yield(3), Step::Done, Step::Done]),
            ],
            Select::RoundRobin,
        )
        .expect("3 sources fit the default cap");
        let mut buf = [0u32; 16];
        let count = drain_vec(&fan, &mut buf);
        let got = &mut buf[..count];
        got.sort_unstable();
        assert_eq!(
            got,
            &[1, 2, 3],
            "items from live sources, drained one skipped, merge keeps going"
        );
    }

    #[test]
    fn vec_all_sources_exhausted_resolves_exhausted() {
        let fan = FanInVec::new(
            [Script::new([Step::Done]), Script::new([Step::Done])],
            Select::RoundRobin,
        )
        .expect("2 sources fit the default cap");
        let mut buf = [0u32; 4];
        assert_eq!(
            drain_vec(&fan, &mut buf),
            0,
            "every source starts exhausted, so the merge resolves Exhausted immediately"
        );
        assert_eq!(fan.live_count(), 0);
    }

    #[test]
    fn vec_exceeding_cap_is_reported_not_truncated() {
        // exclusive range, not `0..=CAP`: `RangeInclusive<usize>` does not
        // implement `ExactSizeIterator` (it cannot always represent its own
        // length in a `usize`); `Range<usize>` does.
        let too_many = (0..(FAN_IN_SOURCE_CAP + 1))
            .map(|index| Script::new([Step::Yield(index as u32), Step::Done]));
        let result = FanInVec::new(too_many, Select::RoundRobin);
        let err = match result {
            Ok(_) => panic!("CAP + 1 sources must be rejected, not truncated"),
            Err(err) => err,
        };
        assert_eq!(err.attempted, FAN_IN_SOURCE_CAP + 1);
        assert_eq!(err.capacity, FAN_IN_SOURCE_CAP);
    }

    #[test]
    fn vec_arity_of_zero_resolves_exhausted_immediately() {
        let fan = FanInVec::<Script<1>, Select>::new(core::iter::empty(), Select::RoundRobin)
            .expect("zero sources is within capacity");
        assert_eq!(fan.live_count(), 0);
        let mut buf = [0u32; 1];
        assert_eq!(
            drain_vec(&fan, &mut buf),
            0,
            "0 sources => immediately Exhausted"
        );
    }

    #[test]
    fn vec_arity_of_one_drains_correctly() {
        let fan = FanInVec::new([Script::new([Step::Yield(42), Step::Done])], Select::RoundRobin)
            .expect("1 source fits the default cap");
        let mut buf = [0u32; 4];
        let count = drain_vec(&fan, &mut buf);
        assert_eq!(&buf[..count], &[42]);
    }

    #[test]
    fn vec_live_count_tracks_draining() {
        let fan = FanInVec::new(
            [Script::new([Step::Yield(1)]), Script::new([Step::Yield(2)])],
            Select::RoundRobin,
        )
        .expect("2 sources fit the default cap");
        assert_eq!(fan.live_count(), 2);
    }

    // `UnpinSendPipe::call`'s merge loop is `FanInVecSendCall` — drive it
    // through the `Send` entry point specifically, mirroring the fixed-arity
    // form's `drain_send`/`unpin_send_pipe_*` tests.
    fn drain_vec_send<S, Strategy>(fan: &FanInVec<S, Strategy>, out: &mut [u32]) -> usize
    where
        S: UnpinSendPipe<In = (), Out = u32, Err = Exhausted> + DropSafe,
        Strategy: FanInStrategy + Send + Sync + 'static,
    {
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        let mut count = 0;
        for _ in 0..10_000 {
            let mut call = UnpinSendPipe::call(fan, ());
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
    fn vec_unpin_send_pipe_merges_all_sources_in_round_robin_order() {
        let fan = FanInVec::new(
            [
                Script::new([Step::Yield(0), Step::Yield(1), Step::Done]),
                Script::new([Step::Yield(10), Step::Yield(11), Step::Done]),
                Script::new([Step::Yield(20), Step::Yield(21), Step::Done]),
            ],
            Select::RoundRobin,
        )
        .expect("3 sources fit the default cap");
        let mut buf = [0u32; 16];
        let count = drain_vec_send(&fan, &mut buf);
        assert_eq!(
            &buf[..count],
            &[0, 10, 20, 1, 11, 21],
            "same round-robin fairness as the UnpinPipe tier"
        );
    }

    #[test]
    fn vec_unpin_send_pipe_future_is_send_and_unpin() {
        fn needs_send_unpin<F: Future + Send + Unpin>(_: &F) {}
        let fan = FanInVec::new([Script::new([Step::Yield(1), Step::Done])], Select::Fifo)
            .expect("1 source fits the default cap");
        let call = UnpinSendPipe::call(&fan, ());
        needs_send_unpin(&call);
    }
}
