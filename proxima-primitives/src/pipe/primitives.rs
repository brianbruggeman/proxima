//! The pipe primitives: the no_std + no-alloc root the whole algebra is built from.
//!
//! [`Pipe`] is the root form — typed In/Out/Err, RPITIT, NO `Send` bound. This
//! INVERTS the legacy arrangement: local is root, `Send` is the additive
//! constraint ([`SendPipe`]), because Send-everywhere is a work-stealing
//! assumption and prime is per-core shared-nothing. [`AndThen`] is the
//! composition law (`Second::Err: From<First::Err>` at the call site); the
//! 13 `proxima-core` markers AND-propagate through it.
//!
//! Nothing here allocates or touches `std`: it compiles on a bare
//! `#![no_std]` target (proven by the crate's core cliff build).

use core::fmt::Debug;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

/// Core-tier root form: typed In/Out/Err, no `Send` bound on `Self` or the
/// returned future. The minimal contract a pipe must satisfy; usable on a
/// per-core, shared-nothing `!Send` worker (`Rc`, `RefCell`, per-core arenas).
///
/// `Err` is an associated type so error interpretation is the only place the
/// signature carries protocol meaning; nominal byte pipes set `Err` to their
/// protocol error.
///
/// # The four forms
///
/// There are not four traits to learn. Pick `In` and `Out` — where `()` is
/// Rust's "nothing" value — and this one trait becomes four familiar shapes.
/// Only `transform` is load-bearing; the other three are it, degenerated:
///
/// | form      | shape       | what it is                                |
/// |-----------|-------------|-------------------------------------------|
/// | transform | `In -> Out` | turns one thing into another              |
/// | source    | `() -> Out` | takes nothing, produces something         |
/// | sink      | `In -> ()`  | takes something, produces nothing         |
/// | observe   | `In -> In`  | hands back its input; acts on the side    |
///
/// All four, as real code — this block is compiled by `cargo test`, so it
/// cannot describe a trait that no longer exists:
///
/// ```
/// use core::convert::Infallible;
/// use core::future::Future;
/// use proxima_primitives::pipe::Pipe;
///
/// // transform: In -> Out
/// struct Double;
/// impl Pipe for Double {
///     type In = u64;
///     type Out = u64;
///     type Err = Infallible;
///     fn call(&self, input: u64) -> impl Future<Output = Result<u64, Infallible>> {
///         async move { Ok(input * 2) }
///     }
/// }
///
/// // source: () -> Out. Nothing goes in.
/// struct Always;
/// impl Pipe for Always {
///     type In = ();
///     type Out = u64;
///     type Err = Infallible;
///     fn call(&self, _input: ()) -> impl Future<Output = Result<u64, Infallible>> {
///         async move { Ok(7) }
///     }
/// }
///
/// // sink: In -> (). Nothing comes out.
/// struct Discard;
/// impl Pipe for Discard {
///     type In = u64;
///     type Out = ();
///     type Err = Infallible;
///     fn call(&self, _input: u64) -> impl Future<Output = Result<(), Infallible>> {
///         async move { Ok(()) }
///     }
/// }
///
/// // observe: In -> In. Out == In is what makes it an observe.
/// struct Echo;
/// impl Pipe for Echo {
///     type In = u64;
///     type Out = u64;
///     type Err = Infallible;
///     fn call(&self, input: u64) -> impl Future<Output = Result<u64, Infallible>> {
///         async move { Ok(input) }
///     }
/// }
/// ```
pub trait Pipe {
    /// Input the pipe consumes.
    type In;
    /// Output the pipe produces.
    type Out;
    /// Error the pipe can fail with. `Debug + 'static`; `Send` is added only on
    /// the cross-core [`SendPipe`] form.
    type Err: Debug + 'static;

    /// Apply the pipe. The returned future is NOT required to be `Send`.
    fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>>;
}

/// Additive cross-core form: the pipe and its returned future are `Send`, so it
/// can be dispatched across cores. Standalone (not `SendPipe: Pipe`) because an
/// RPITIT future's `Send`-ness cannot be strengthened by a subtrait on stable.
///
/// There is no blanket bridge from [`Pipe`], and there cannot be one: writing
/// `impl<P: Pipe + Send> SendPipe for P` requires bounding `P::call`'s returned
/// future — a bound on an RPITIT return type, i.e. return-type notation, which
/// is unstable (rust#109417). So each additive constraint costs a full standalone
/// copy of the contract. When RTN stabilises, every tier below collapses back
/// into `Pipe` plus a bound at the use site, and these traits are deletable.
pub trait SendPipe: Send + Sync + 'static {
    /// Input the pipe consumes.
    type In;
    /// Output the pipe produces.
    type Out;
    /// Error the pipe can fail with — `Send` here so it can cross cores.
    type Err: Debug + Send + 'static;

    /// Apply the pipe. The returned future IS `Send`.
    fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send;
}

/// Additive in-place form: the returned future is `Unpin`, so a caller can poll
/// it where it sits — `Pin::new(&mut fut)` is safe. No `unsafe`, no `Box`, no
/// allocation, no pin-projection.
///
/// `Unpin` is the rung after `Send` on the same ladder: `Pipe` (borrow, `!Send`)
/// → `+ 'static` (own it, erase) → `+ Send` (cross a core) → `+ Unpin` (poll in
/// place). Climb only as far as the use demands. What it buys: a caller holding
/// several of these can ask each "anything ready?" and poll them where they
/// sit — on a bare target, with no heap.
///
/// The cost is real and belongs to the impl: an `async move { .. }` block is
/// `!Unpin`, so an implementor returns a hand-written poll struct instead — a
/// `Future` with a `poll`, which is what a ring-backed source already is.
///
/// Standalone for the same reason [`SendPipe`] is: no blanket bridge is
/// expressible on stable (see `SendPipe`'s note on rust#109417).
pub trait UnpinPipe {
    /// Input the pipe consumes.
    type In;
    /// Output the pipe produces.
    type Out;
    /// Error the pipe can fail with.
    type Err: Debug + 'static;

    /// Apply the pipe. The returned future IS `Unpin` — pollable in place.
    fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>> + Unpin;
}

/// Both additive constraints at once: the pipe and its future are `Send`, and
/// the future is `Unpin` — the top rung, crossing a core AND pollable in place.
///
/// Two constraints means four standalone traits, because none of them compose on
/// stable: `Pipe`, `+ Send`, `+ Unpin`, `+ Send + Unpin`. That tax is the direct
/// cost of rust#109417; with return-type notation every rung collapses back into
/// `Pipe` plus a bound at the use site, and all three of these are deletable.
///
/// Do not reach for this rung by default. Wanting it usually means a caller is
/// paying to poll `Send` futures in place; wanting it *speculatively* means
/// nothing needs it yet.
pub trait UnpinSendPipe: Send + Sync + 'static {
    /// Input the pipe consumes.
    type In;
    /// Output the pipe produces.
    type Out;
    /// Error the pipe can fail with — `Send` here so it can cross cores.
    type Err: Debug + Send + 'static;

    /// Apply the pipe. The returned future is `Send` AND `Unpin`.
    fn call(
        &self,
        input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send + Unpin;
}

/// Type-chained sequence: feed `input` through `first`, then feed the
/// intermediate through `second`. The type system enforces
/// `first::Out = second::In`; the error channel is bridged by
/// `Second::Err: From<First::Err>` (thiserror `#[from]` chains supply the
/// impls). This is the composition law of the form family — the only operator
/// the backport lands at G2; Race/Tee/Quorum follow when a use case asks.
///
/// `AndThen` is itself a [`Pipe`] (and a [`SendPipe`] when both stages are), so
/// chains nest: `AndThen::new(a, AndThen::new(b, c))`.
#[derive(Debug, Clone, Copy)]
pub struct AndThen<First, Second> {
    first: First,
    second: Second,
}

impl<First, Second> AndThen<First, Second> {
    /// Construct a two-stage `AndThen`. Nest for longer chains.
    #[must_use]
    pub const fn new(first: First, second: Second) -> Self {
        Self { first, second }
    }
}

impl<First, Second> Pipe for AndThen<First, Second>
where
    First: Pipe,
    Second: Pipe<In = First::Out>,
    Second::Err: From<First::Err>,
{
    type In = First::In;
    type Out = Second::Out;
    type Err = Second::Err;

    fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>> {
        async move {
            // `?` converts First::Err into Second::Err via the From bound at
            // this composition site — the error law lives here, not on the trait.
            let intermediate = self.first.call(input).await?;
            self.second.call(intermediate).await
        }
    }
}

impl<First, Second> SendPipe for AndThen<First, Second>
where
    First: SendPipe,
    Second: SendPipe<In = First::Out>,
    Second::Err: From<First::Err>,
    First::In: Send,
{
    type In = First::In;
    type Out = Second::Out;
    type Err = Second::Err;

    fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let intermediate = self.first.call(input).await?;
            self.second.call(intermediate).await
        }
    }
}

/// Hand-written poll state machine backing `AndThen`'s [`UnpinPipe`]/
/// [`UnpinSendPipe`] impls — no `Box::pin`, no `async move` (a compiler-
/// generated async block is not provably `Unpin`, which is the entire reason
/// this tier needs a hand-written `Future` in the first place). Two states,
/// `First` then `Second`, matching the two pipe stages.
///
/// The second stage's future type is RPITIT — unnameable from outside its own
/// impl — so it cannot be a field of `First` the way `future: FirstFut` is.
/// `next` carries it instead, as a closure produced but not yet CALLED
/// (`Option` so `poll` can move it out once, on the `First -> Second`
/// transition). This is exactly [`start_and_then`]'s reason for existing: see
/// its doc for how `SecondFut` gets resolved with no named type anywhere.
enum AndThenUnpinCall<FirstFut, Next, SecondFut> {
    First {
        future: FirstFut,
        next: Option<Next>,
    },
    Second {
        future: SecondFut,
    },
}

/// Construct [`AndThenUnpinCall`]'s `First` state. A free fn, not a bare
/// struct literal, because `SecondFut` appears in NEITHER argument here —
/// only in the `next: FnOnce(Intermediate) -> SecondFut` bound below — and a
/// struct literal gives the compiler nothing to solve a missing field's type
/// from. A function-level generic parameter does: `next`'s CONCRETE closure
/// type uniquely determines `SecondFut` via its own `FnOnce` impl (the same
/// mechanism that lets `Option::map`/`Iterator::map` infer their output type
/// from a closure with no type annotation anywhere).
fn start_and_then<FirstFut, Intermediate, FirstErr, Next, SecondFut>(
    future: FirstFut,
    next: Next,
) -> AndThenUnpinCall<FirstFut, Next, SecondFut>
where
    FirstFut: Future<Output = Result<Intermediate, FirstErr>>,
    Next: FnOnce(Intermediate) -> SecondFut,
{
    AndThenUnpinCall::First {
        future,
        next: Some(next),
    }
}

impl<FirstFut, Intermediate, FirstErr, Next, SecondFut, SecondOut, SecondErr> Future
    for AndThenUnpinCall<FirstFut, Next, SecondFut>
where
    FirstFut: Future<Output = Result<Intermediate, FirstErr>> + Unpin,
    Next: FnOnce(Intermediate) -> SecondFut + Unpin,
    SecondFut: Future<Output = Result<SecondOut, SecondErr>> + Unpin,
    SecondErr: From<FirstErr>,
{
    type Output = Result<SecondOut, SecondErr>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // `Self: Unpin` follows structurally from the bounds above (every
        // field is `Unpin`), so `get_mut` needs no `unsafe` pin projection.
        let this = self.get_mut();
        loop {
            match this {
                AndThenUnpinCall::First { future, next } => {
                    match Pin::new(future).poll(cx) {
                        Poll::Ready(Ok(intermediate)) => {
                            // `next` is `None` only if this future is polled
                            // again after already resolving — not a memory
                            // safety issue, so parking here (rather than
                            // panicking) is the house style for that case.
                            let Some(next) = next.take() else {
                                return Poll::Pending;
                            };
                            *this = AndThenUnpinCall::Second {
                                future: next(intermediate),
                            };
                            // loop: the Second stage may already be ready.
                        }
                        Poll::Ready(Err(err)) => return Poll::Ready(Err(SecondErr::from(err))),
                        Poll::Pending => return Poll::Pending,
                    }
                }
                AndThenUnpinCall::Second { future } => return Pin::new(future).poll(cx),
            }
        }
    }
}

impl<First, Second> UnpinPipe for AndThen<First, Second>
where
    First: UnpinPipe,
    Second: UnpinPipe<In = First::Out>,
    Second::Err: From<First::Err>,
{
    type In = First::In;
    type Out = Second::Out;
    type Err = Second::Err;

    fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>> + Unpin {
        let second = &self.second;
        start_and_then(self.first.call(input), move |intermediate| {
            second.call(intermediate)
        })
    }
}

impl<First, Second> UnpinSendPipe for AndThen<First, Second>
where
    First: UnpinSendPipe,
    Second: UnpinSendPipe<In = First::Out>,
    Second::Err: From<First::Err>,
{
    type In = First::In;
    type Out = Second::Out;
    type Err = Second::Err;

    fn call(
        &self,
        input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send + Unpin {
        let second = &self.second;
        start_and_then(self.first.call(input), move |intermediate| {
            second.call(intermediate)
        })
    }
}

// Marker propagation — an `AndThen` carries a marker only when BOTH stages do.
// AND-composition via blanket impls (mirrors the legacy proxima-process
// operators.rs and proxima-core's marker doc: OR-propagation would need
// overlapping blankets / specialization). The markers live in
// `proxima-core::markers`; `AndThen` is local, so these foreign-trait /
// local-type impls are coherence-free.
mod marker_propagation {
    use super::AndThen;
    use proxima_core::markers::{
        AllocFree, Commutative, Deterministic, DropSafe, IdempotentSideEffectFree, IsPure, NoStd,
        Reproducible, WithoutFilesystem, WithoutNetwork, WithoutRandom, WithoutSpawn, WithoutTime,
    };

    impl<First: DropSafe, Second: DropSafe> DropSafe for AndThen<First, Second> {}
    impl<First: NoStd, Second: NoStd> NoStd for AndThen<First, Second> {}
    impl<First: AllocFree, Second: AllocFree> AllocFree for AndThen<First, Second> {}
    impl<First: IsPure, Second: IsPure> IsPure for AndThen<First, Second> {}
    impl<First: Deterministic, Second: Deterministic> Deterministic for AndThen<First, Second> {}
    impl<First: Reproducible, Second: Reproducible> Reproducible for AndThen<First, Second> {}
    impl<First: IdempotentSideEffectFree, Second: IdempotentSideEffectFree> IdempotentSideEffectFree
        for AndThen<First, Second>
    {
    }
    impl<First: Commutative, Second: Commutative> Commutative for AndThen<First, Second> {}

    impl<First: WithoutFilesystem, Second: WithoutFilesystem> WithoutFilesystem
        for AndThen<First, Second>
    {
    }
    impl<First: WithoutNetwork, Second: WithoutNetwork> WithoutNetwork for AndThen<First, Second> {}
    impl<First: WithoutSpawn, Second: WithoutSpawn> WithoutSpawn for AndThen<First, Second> {}
    impl<First: WithoutTime, Second: WithoutTime> WithoutTime for AndThen<First, Second> {}
    impl<First: WithoutRandom, Second: WithoutRandom> WithoutRandom for AndThen<First, Second> {}
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{AndThen, Pipe, SendPipe};
    use crate::pipe::ext::PipeExt;
    use core::convert::Infallible;
    use core::future::Future;
    use core::marker::PhantomData;
    use proxima_core::markers::{
        AllocFree, Commutative, Deterministic, DropSafe, IdempotentSideEffectFree, IsPure, NoStd,
        Reproducible, WithoutFilesystem, WithoutNetwork, WithoutRandom, WithoutSpawn, WithoutTime,
    };

    /// Dependency-free executor for the always-ready probe futures — keeps the
    /// leaf crate no_std-pure even in tests (the `proxima` umbrella test macro
    /// would be a dependency cycle). `Waker::noop` is stable since 1.85.
    fn block_on<Fut: Future>(future: Fut) -> Fut::Output {
        let mut pinned = core::pin::pin!(future);
        let mut context = core::task::Context::from_waker(core::task::Waker::noop());
        loop {
            if let core::task::Poll::Ready(output) = pinned.as_mut().poll(&mut context) {
                return output;
            }
        }
    }

    /// Pure passthrough probe: carries every marker, impls both forms so it
    /// exercises `AndThen` on the no-Send root AND the cross-core form.
    /// `PhantomData<fn() -> Value>` keeps it `Send + Sync` for any `Value`.
    struct Identity<Value>(PhantomData<fn() -> Value>);

    impl<Value> Identity<Value> {
        const fn new() -> Self {
            Self(PhantomData)
        }
    }

    impl<Value> Pipe for Identity<Value> {
        type In = Value;
        type Out = Value;
        type Err = Infallible;

        fn call(&self, input: Value) -> impl Future<Output = Result<Value, Infallible>> {
            async move { Ok(input) }
        }
    }

    impl<Value: Send + 'static> SendPipe for Identity<Value> {
        type In = Value;
        type Out = Value;
        type Err = Infallible;

        fn call(&self, input: Value) -> impl Future<Output = Result<Value, Infallible>> + Send {
            async move { Ok(input) }
        }
    }

    impl<Value> DropSafe for Identity<Value> {}
    impl<Value> NoStd for Identity<Value> {}
    impl<Value> AllocFree for Identity<Value> {}
    impl<Value> IsPure for Identity<Value> {}
    impl<Value> Deterministic for Identity<Value> {}
    impl<Value> Reproducible for Identity<Value> {}
    impl<Value> IdempotentSideEffectFree for Identity<Value> {}
    impl<Value> Commutative for Identity<Value> {}
    impl<Value> WithoutFilesystem for Identity<Value> {}
    impl<Value> WithoutNetwork for Identity<Value> {}
    impl<Value> WithoutSpawn for Identity<Value> {}
    impl<Value> WithoutTime for Identity<Value> {}
    impl<Value> WithoutRandom for Identity<Value> {}

    /// Fallible probe with its own error type — proves a non-trivial `Err`.
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

    /// Second-stage probe whose error type is DISTINCT from `Increment`'s but
    /// absorbs it via `From` — this is what makes `AndThen<Increment, Halve>`
    /// type-check, proving the composition law `Second::Err: From<First::Err>`.
    #[derive(Debug, PartialEq, Eq)]
    enum HalveError {
        Odd,
        Upstream(Overflow),
    }

    impl From<Overflow> for HalveError {
        fn from(value: Overflow) -> Self {
            HalveError::Upstream(value)
        }
    }

    struct Halve;

    impl Pipe for Halve {
        type In = u64;
        type Out = u64;
        type Err = HalveError;

        fn call(&self, input: u64) -> impl Future<Output = Result<u64, HalveError>> {
            async move {
                if input.is_multiple_of(2) {
                    Ok(input / 2)
                } else {
                    Err(HalveError::Odd)
                }
            }
        }
    }

    /// Third-stage probe whose error absorbs BOTH upstream error types via
    /// `From` — proves the two nestings of a 3-stage chain
    /// (`a.and_then(b).and_then(c)` vs `AndThen::new(a, AndThen::new(b, c))`)
    /// both type-check and agree on output.
    #[derive(Debug, PartialEq, Eq)]
    enum DoubleError {
        TooBig,
        Halve(HalveError),
    }

    impl From<HalveError> for DoubleError {
        fn from(value: HalveError) -> Self {
            DoubleError::Halve(value)
        }
    }

    impl From<Overflow> for DoubleError {
        fn from(value: Overflow) -> Self {
            DoubleError::Halve(HalveError::from(value))
        }
    }

    struct Double;

    impl Pipe for Double {
        type In = u64;
        type Out = u64;
        type Err = DoubleError;

        fn call(&self, input: u64) -> impl Future<Output = Result<u64, DoubleError>> {
            async move { input.checked_mul(2).ok_or(DoubleError::TooBig) }
        }
    }

    /// Always-erroring first stage — used to prove a downstream stage never
    /// runs when an upstream stage short-circuits the chain.
    struct AlwaysFail;

    impl Pipe for AlwaysFail {
        type In = u64;
        type Out = u64;
        type Err = Overflow;

        fn call(&self, _input: u64) -> impl Future<Output = Result<u64, Overflow>> {
            async move { Err(Overflow) }
        }
    }

    /// Records whether it was ever invoked, via a caller-owned flag borrowed
    /// for the probe's lifetime — proves `AndThen::call`'s `?` short-circuit
    /// truly never reaches the second stage, not just that the output matches.
    struct SpyRef<'flag> {
        ran: &'flag core::cell::Cell<bool>,
    }

    impl Pipe for SpyRef<'_> {
        type In = u64;
        type Out = u64;
        type Err = HalveError;

        fn call(&self, input: u64) -> impl Future<Output = Result<u64, HalveError>> {
            self.ran.set(true);
            async move { Ok(input) }
        }
    }

    #[test]
    fn series_pipes_identity_through_identity() {
        // Identity implements BOTH Pipe and SendPipe, so `.and_then()` would be
        // ambiguous (E0034) between PipeExt and SendPipeExt without a
        // fully-qualified call, which reads worse than the explicit
        // constructor. Left as AndThen::new.
        let chain = AndThen::new(Identity::<u64>::new(), Identity::<u64>::new());
        let out = block_on(Pipe::call(&chain, 41)).expect("infallible identity chain");
        assert_eq!(out, 41);
    }

    #[test]
    fn series_composes_distinct_error_types_via_from() {
        // 5 -> increment -> 6 -> halve -> 3. Both stages succeed.
        let chain = Increment.and_then(Halve);
        let out = block_on(Pipe::call(&chain, 5)).expect("even intermediate halves");
        assert_eq!(out, 3);
    }

    #[test]
    fn series_propagates_first_stage_error_converted_by_from() {
        // u64::MAX -> increment overflows -> Overflow -> From -> HalveError::Upstream.
        let chain = Increment.and_then(Halve);
        let err = block_on(Pipe::call(&chain, u64::MAX)).expect_err("overflow propagates");
        assert_eq!(err, HalveError::Upstream(Overflow));
    }

    #[test]
    fn series_surfaces_second_stage_error_unchanged() {
        // 6 -> increment -> 7 (odd) -> halve rejects.
        let chain = Increment.and_then(Halve);
        let err = block_on(Pipe::call(&chain, 6)).expect_err("odd intermediate rejected");
        assert_eq!(err, HalveError::Odd);
    }

    fn assert_send<Type: Send>(_: &Type) {}

    #[test]
    fn series_is_a_send_pipe_with_a_send_future() {
        // same ambiguity as series_pipes_identity_through_identity — Identity
        // implements both Pipe and SendPipe, so `.and_then()` is ambiguous here.
        let chain = AndThen::new(Identity::<u64>::new(), Identity::<u64>::new());
        let future = SendPipe::call(&chain, 7);
        assert_send(&future);
        let out = block_on(future).expect("send chain");
        assert_eq!(out, 7);
    }

    // compile-time proof: an AndThen of two fully-marked pipes carries every
    // marker. If propagation regressed, this fails to compile — the marker
    // gates are structural, not asserted at runtime.
    fn assert_all_markers<Type>()
    where
        Type: NoStd
            + AllocFree
            + IsPure
            + Deterministic
            + Reproducible
            + IdempotentSideEffectFree
            + Commutative
            + WithoutFilesystem
            + WithoutNetwork
            + WithoutSpawn
            + WithoutTime
            + WithoutRandom
            + DropSafe,
    {
    }

    #[test]
    fn series_propagates_all_markers_from_both_stages() {
        assert_all_markers::<AndThen<Identity<u64>, Identity<u64>>>();
        // nested chains still carry the markers (AND of AND).
        assert_all_markers::<AndThen<Identity<u64>, AndThen<Identity<u64>, Identity<u64>>>>();
    }

    #[test]
    fn and_then_matches_series_new_for_a_two_stage_chain() {
        let fluent = block_on(Pipe::call(&Increment.and_then(Halve), 5));
        let manual = block_on(Pipe::call(&AndThen::new(Increment, Halve), 5));
        assert_eq!(fluent, manual);
        assert_eq!(fluent.expect("even intermediate halves"), 3);
    }

    #[test]
    fn and_then_matches_series_new_on_the_error_path() {
        let fluent = block_on(Pipe::call(&Increment.and_then(Halve), u64::MAX));
        let manual = block_on(Pipe::call(&AndThen::new(Increment, Halve), u64::MAX));
        assert_eq!(fluent, manual);
        assert_eq!(
            fluent.expect_err("overflow propagates"),
            HalveError::Upstream(Overflow)
        );
    }

    #[test]
    fn and_then_chain_matches_nested_series_new_for_a_three_stage_chain() {
        // a.and_then(b).and_then(c) associates as AndThen::new(AndThen::new(a, b), c);
        // AndThen::new(a, AndThen::new(b, c)) associates the other way. Both are
        // legal compositions of the same three stages and must agree on output.
        let left_leaning = Increment.and_then(Halve).and_then(Double);
        let right_leaning = AndThen::new(Increment, AndThen::new(Halve, Double));

        let left_output = block_on(Pipe::call(&left_leaning, 5)).expect("5 -> 6 -> 3 -> 6");
        let right_output = block_on(Pipe::call(&right_leaning, 5)).expect("5 -> 6 -> 3 -> 6");
        assert_eq!(left_output, right_output);
        assert_eq!(left_output, 6);
    }

    #[test]
    fn and_then_short_circuits_before_the_second_stage_on_first_stage_error() {
        let ran = core::cell::Cell::new(false);
        let chain = AlwaysFail.and_then(SpyRef { ran: &ran });

        let err = block_on(Pipe::call(&chain, 1)).expect_err("first stage always fails");

        assert_eq!(err, HalveError::Upstream(Overflow));
        assert!(
            !ran.get(),
            "second stage must not run after first stage errors"
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod unpin_tier_tests {
    use super::{AndThen, UnpinPipe, UnpinSendPipe};
    use core::convert::Infallible;
    use core::future::Future;
    use core::pin::Pin;
    use core::task::{Context, Poll, Waker};

    // What an `Unpin` source actually is: a hand-written poll struct, not an
    // async block. A ring pop is exactly this shape already.
    struct RingPop(u8);
    impl Future for RingPop {
        type Output = Result<u8, Infallible>;
        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Ready(Ok(self.0))
        }
    }

    struct Ring(u8);
    impl UnpinPipe for Ring {
        type In = ();
        type Out = u8;
        type Err = Infallible;
        fn call(&self, (): ()) -> impl Future<Output = Result<u8, Infallible>> + Unpin {
            RingPop(self.0)
        }
    }
    impl UnpinSendPipe for Ring {
        type In = ();
        type Out = u8;
        type Err = Infallible;
        fn call(&self, (): ()) -> impl Future<Output = Result<u8, Infallible>> + Send + Unpin {
            RingPop(self.0)
        }
    }

    // THE POINT: poll N sources in place. No unsafe, no Box, no alloc, and no
    // future is abandoned — each is polled where it sits.
    fn merge_in_place<S: UnpinPipe<In = ()>, const N: usize>(
        sources: &[S; N],
        cx: &mut Context<'_>,
    ) -> Poll<Result<S::Out, S::Err>> {
        for source in sources {
            let mut call = source.call(());
            if let Poll::Ready(out) = Pin::new(&mut call).poll(cx) {
                return Poll::Ready(out);
            }
        }
        Poll::Pending
    }

    #[test]
    fn unpin_future_polls_in_place_without_unsafe_or_alloc() {
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        let sources = [Ring(7), Ring(9)];
        match merge_in_place(&sources, &mut cx) {
            Poll::Ready(Ok(value)) => assert_eq!(value, 7, "first ready source wins"),
            other => panic!("expected the first source to be ready, got {other:?}"),
        }
    }

    #[test]
    fn send_unpin_tier_is_the_cross_core_mergeable_form() {
        fn assert_send_unpin<S: UnpinSendPipe<In = ()>>(source: &S) {
            fn needs_send_unpin<F: Future + Send + Unpin>(_: &F) {}
            let call = UnpinSendPipe::call(source, ());
            needs_send_unpin(&call);
        }
        assert_send_unpin(&Ring(1));
    }

    // second-stage probe: doubles, always immediately ready — an Unpin pipe
    // that never suspends, the ring-pop shape `core::future::ready` gives for
    // free.
    struct Doubler;
    impl UnpinPipe for Doubler {
        type In = u8;
        type Out = u16;
        type Err = Infallible;
        fn call(&self, input: u8) -> impl Future<Output = Result<u16, Infallible>> + Unpin {
            core::future::ready(Ok(u16::from(input) * 2))
        }
    }
    impl UnpinSendPipe for Doubler {
        type In = u8;
        type Out = u16;
        type Err = Infallible;
        fn call(&self, input: u8) -> impl Future<Output = Result<u16, Infallible>> + Send + Unpin {
            core::future::ready(Ok(u16::from(input) * 2))
        }
    }

    #[test]
    fn and_then_unpin_chain_composes_two_pipes_without_a_box() {
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        let chain = AndThen::new(Ring(21), Doubler);
        let mut call = UnpinPipe::call(&chain, ());
        match Pin::new(&mut call).poll(&mut cx) {
            Poll::Ready(Ok(value)) => assert_eq!(value, 42, "21 -> ring pop -> double -> 42"),
            other => panic!("expected ready, got {other:?}"),
        }
    }

    #[test]
    fn and_then_unpin_send_chain_is_send_and_unpin() {
        fn needs_send_unpin<F: Future + Send + Unpin>(_: &F) {}
        let chain = AndThen::new(Ring(5), Doubler);
        let call = UnpinSendPipe::call(&chain, ());
        needs_send_unpin(&call);
    }

    // first-stage probe: reports `Pending` exactly once (registering the
    // waker), `Ready` on every poll after — proves the AndThen state machine
    // genuinely resumes across separate `poll()` calls (not just when both
    // stages resolve on the first poll) without re-running the first stage.
    struct PendOnceThenReady {
        value: u8,
        polled_once: core::cell::Cell<bool>,
    }
    impl Future for PendOnceThenReady {
        type Output = Result<u8, Infallible>;
        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            if self.polled_once.replace(true) {
                Poll::Ready(Ok(self.value))
            } else {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }

    struct SlowRing(u8);
    impl UnpinPipe for SlowRing {
        type In = ();
        type Out = u8;
        type Err = Infallible;
        fn call(&self, (): ()) -> impl Future<Output = Result<u8, Infallible>> + Unpin {
            PendOnceThenReady {
                value: self.0,
                polled_once: core::cell::Cell::new(false),
            }
        }
    }

    #[test]
    fn and_then_unpin_chain_resumes_across_polls_without_rerunning_first_stage() {
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        let chain = AndThen::new(SlowRing(21), Doubler);
        let mut call = UnpinPipe::call(&chain, ());
        assert_eq!(
            Pin::new(&mut call).poll(&mut cx),
            Poll::Pending,
            "first stage not ready yet"
        );
        match Pin::new(&mut call).poll(&mut cx) {
            Poll::Ready(Ok(value)) => {
                assert_eq!(value, 42, "second poll resumes at First, then runs Second");
            }
            other => panic!("expected ready on the second poll, got {other:?}"),
        }
    }

    // first-stage probe that always fails — proves the Unpin chain
    // short-circuits before the second stage exactly like the async-block
    // `Pipe`/`SendPipe` forms do.
    #[derive(Debug, PartialEq, Eq)]
    struct RingError;

    struct FailingRing;
    impl UnpinPipe for FailingRing {
        type In = ();
        type Out = u8;
        type Err = RingError;
        fn call(&self, (): ()) -> impl Future<Output = Result<u8, RingError>> + Unpin {
            core::future::ready(Err(RingError))
        }
    }

    struct CountingDoubler {
        calls: core::cell::Cell<u32>,
    }
    impl UnpinPipe for CountingDoubler {
        type In = u8;
        type Out = u16;
        type Err = RingError;
        fn call(&self, input: u8) -> impl Future<Output = Result<u16, RingError>> + Unpin {
            self.calls.set(self.calls.get() + 1);
            core::future::ready(Ok(u16::from(input) * 2))
        }
    }

    #[test]
    fn and_then_unpin_chain_short_circuits_before_the_second_stage() {
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        let second = CountingDoubler {
            calls: core::cell::Cell::new(0),
        };
        let chain = AndThen::new(FailingRing, second);
        let mut call = UnpinPipe::call(&chain, ());
        match Pin::new(&mut call).poll(&mut cx) {
            Poll::Ready(Err(RingError)) => {}
            other => panic!("expected the first stage's error, got {other:?}"),
        }
        assert_eq!(
            chain.second.calls.get(),
            0,
            "second stage must not run after first stage errors"
        );
    }
}
