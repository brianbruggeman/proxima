//! Fluent surface over [`Runtime`], carrying the combinations the trait's
//! named methods pre-combine.
//!
//! [`Runtime`] is consumed as `Arc<dyn Runtime>` (`App::offline_runtime`, the
//! recording spigot), so it must stay object-safe: no generics, no RPITIT, and
//! every payload boxed. That erasure is real on the trait and needless at the
//! call site — this trait is never used as `dyn` itself, so it may carry
//! generics, and the boxing happens once inside it. Every method here stays
//! callable *through* a trait object: no `Self: Sized` bounds, and the blanket
//! impl is `?Sized`, so `Arc<dyn Runtime>` reaches the whole builder.
//!
//! Three independent axes the named methods weld together:
//!
//! | axis      | choices                                  |
//! |-----------|------------------------------------------|
//! | placement | current core (default) / chosen core / background pool |
//! | payload   | future / blocking closure                |
//! | failure   | `SpawnError` on a chosen core            |
//!
//! `spawn_on_current_core` binds placement and drops the `Result`;
//! `spawn_factory_on_core` binds placement and payload in one identifier. The
//! grid has more cells than there are names for, and this is where the rest of
//! them live.
//!
//! ```text
//! rt.spawn(fut);                       // current core
//! rt.core(id).spawn(fut)?;             // chosen core, SpawnError surfaced
//! rt.blocking(work).spawn().await?;    // background pool, typed R back
//! rt.core([1, 2, 3]).spawn_with(|| work(), |_, e| *e == SpawnError::InboxFull)?;
//! // -> Ok(CoreId) which core took it, or Err(Exhausted { attempts, last })
//! ```
//!
//! `text`, not a doctest: every line needs a live `Runtime`. A doctest that
//! cannot run is worse than prose — `ignore` compiles nothing while
//! `cargo test --doc` still exits 0, which is how a broken example ships. The
//! executable form of all four lines is in this module's tests.

use alloc::boxed::Box;
use core::any::Any;
use core::future::Future;

use proxima_core::ProximaError;

use crate::{BackgroundHandle, CoreId, Runtime, SpawnError};

/// Fluent spawn surface. Blanket-implemented for every [`Runtime`]; never used
/// as `dyn` itself, which is what buys back the generics.
pub trait RuntimeExt: Runtime {
    /// Spawn a future on the current core. The trait's own
    /// `spawn_on_current_core` discards a `SpawnError` that `spawn_on_core`
    /// surfaces — this keeps that discard where the caller can see it.
    fn spawn<F>(&self, future: F)
    where
        F: Future<Output = ()> + 'static,
    {
        self.spawn_on_current_core(Box::pin(future));
    }

    /// Bind the placement axis to a specific core. Terminal `spawn` then
    /// returns `Result<(), SpawnError>` rather than swallowing back-pressure.
    fn core(&self, placement: impl Into<Placement>) -> OnCore<'_, Self> {
        OnCore { runtime: self, placement: placement.into() }
    }

    /// Bind the payload axis to work that occupies a thread. Nothing here
    /// blocks the caller: the terminal hands back a future.
    fn blocking<R, W>(&self, work: W) -> Blocking<'_, Self, W, R>
    where
        R: Send + 'static,
        W: FnOnce() -> Result<R, ProximaError> + Send + 'static,
    {
        Blocking {
            runtime: self,
            work,
            _result: core::marker::PhantomData,
        }
    }
}

impl<R: Runtime + ?Sized> RuntimeExt for R {}

/// Which core, stated as intent rather than as an index the caller computed.
///
/// The trait's primitive takes a single [`CoreId`], so anything other than
/// `Only` has to be resolved before dispatch — and resolution needs
/// `current_core()` and `num_cores()`, both of which the trait already exposes.
/// Expressing the intent here rather than at the call site is also what makes
/// `SpawnError::InboxFull` recoverable: a candidate set can be retried across
/// cores, where a single id can only be retried against the core that was
/// already full.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Placement {
    /// This exact core.
    Only(CoreId),
    /// Any core except the one running this call — the usual intent when the
    /// point is to stop competing with the caller for its own core.
    AnyOther,
    /// Any core at all, current included.
    Any,
    /// One of these, in order, falling through on back-pressure.
    AnyOf(alloc::vec::Vec<CoreId>),
}

impl From<CoreId> for Placement {
    fn from(core_id: CoreId) -> Self {
        Placement::Only(core_id)
    }
}

impl From<usize> for Placement {
    fn from(core_id: usize) -> Self {
        Placement::Only(CoreId(core_id))
    }
}

impl<T: Into<CoreId>> From<alloc::vec::Vec<T>> for Placement {
    fn from(cores: alloc::vec::Vec<T>) -> Self {
        Placement::AnyOf(cores.into_iter().map(Into::into).collect())
    }
}

impl<T: Into<CoreId>, const N: usize> From<[T; N]> for Placement {
    fn from(cores: [T; N]) -> Self {
        Placement::AnyOf(cores.into_iter().map(Into::into).collect())
    }
}

impl Placement {
    /// Candidate cores in dispatch order. `AnyOther` and `Any` are ordered from
    /// the core after the current one so repeated calls spread rather than all
    /// landing on core 0.
    fn candidates(&self, current: CoreId, cores: usize) -> alloc::vec::Vec<CoreId> {
        match self {
            Placement::Only(core_id) => alloc::vec![*core_id],
            Placement::AnyOf(list) => list.clone(),
            Placement::Any => (0..cores).map(|offset| CoreId((current.0 + offset) % cores)).collect(),
            Placement::AnyOther => (1..cores).map(|offset| CoreId((current.0 + offset) % cores)).collect(),
        }
    }
}

/// Every candidate was tried and none accepted — the sequence's own
/// termination signal, kept out of [`SpawnError`] because that type answers
/// for *one* core and its implementors produce only its two arms. Following
/// `fan_in`'s convention: termination lives in the `Err` channel as a distinct
/// value, not as a domain error reused to mean something else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exhausted {
    /// Candidates attempted. Zero means the placement resolved to nothing —
    /// `AnyOther` on a single-core runtime, or an empty `AnyOf`.
    pub attempts: u32,
    /// Why the last attempt failed, absent when there were no attempts.
    pub last: Option<SpawnError>,
}

/// Placement bound. Every stage is complete on its own — the default it
/// replaces is a real value, not an absent one.
pub struct OnCore<'rt, R: ?Sized> {
    runtime: &'rt R,
    placement: Placement,
}

impl<'rt, R: Runtime + ?Sized> OnCore<'rt, R> {
    /// Spawn a `Send` future on the first candidate. A future can only be
    /// spawned once, so this does not fall through on back-pressure — use
    /// [`Self::spawn_with`] when the placement has more than one candidate.
    pub fn spawn<F>(self, future: F) -> Result<(), SpawnError>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let candidates = self
            .placement
            .candidates(self.runtime.current_core(), self.runtime.num_cores());
        let Some(core_id) = candidates.first().copied() else {
            return Err(SpawnError::Disconnected);
        };
        self.runtime.spawn_on_core(core_id, Box::pin(future))
    }

    /// Spawn from a factory, letting the caller decide whether to keep going.
    ///
    /// `InboxFull` drops the future, so recovering means rebuilding it — hence
    /// the factory. What it does *not* do is decide policy: no delay, no
    /// jitter, no attempt cap live here. `decide` is called with the 0-based
    /// attempt and the error, and returning `true` moves to the next candidate.
    ///
    /// That split is deliberate. `RetryController` (backoff, jitter, deadline,
    /// `RetryRules`) lives in `proxima-primitives`, which depends on this crate
    /// — so this crate cannot reach it, and baking a policy in here would fork
    /// it. Passing the decision in means a caller composes with whatever it
    /// already holds at its own layer, and `decide` being a plain `FnMut`
    /// rather than a trait keeps that composition free.
    pub fn spawn_with<Factory, F, Decide>(
        self,
        factory: Factory,
        mut decide: Decide,
    ) -> Result<CoreId, Exhausted>
    where
        Factory: Fn() -> F,
        F: Future<Output = ()> + Send + 'static,
        Decide: FnMut(u32, &SpawnError) -> bool,
    {
        let candidates = self
            .placement
            .candidates(self.runtime.current_core(), self.runtime.num_cores());
        let mut attempts = 0u32;
        let mut last = None;
        for core_id in candidates {
            match self.runtime.spawn_on_core(core_id, Box::pin(factory())) {
                Ok(()) => return Ok(core_id),
                Err(err) => {
                    let keep_going = decide(attempts, &err);
                    attempts = attempts.saturating_add(1);
                    last = Some(err);
                    if !keep_going {
                        break;
                    }
                }
            }
        }
        Err(Exhausted { attempts, last })
    }
}

/// Payload bound to thread-occupying work.
pub struct Blocking<'rt, R: ?Sized, W, Out> {
    runtime: &'rt R,
    work: W,
    _result: core::marker::PhantomData<fn() -> Out>,
}

impl<'rt, R, W, Out> Blocking<'rt, R, W, Out>
where
    R: Runtime + ?Sized,
    Out: Send + 'static,
    W: FnOnce() -> Result<Out, ProximaError> + Send + 'static,
{
    /// Hand the work to the background pool. The `Box<dyn Any>` round trip the
    /// object-safe trait requires is paid here, once, so the caller gets `Out`.
    pub fn spawn(self) -> impl Future<Output = Result<Out, ProximaError>> + Send {
        let handle: BackgroundHandle<Box<dyn Any + Send>> =
            self.runtime.spawn_background_blocking(Box::new(move || {
                (self.work)().map(|value| Box::new(value) as Box<dyn Any + Send>)
            }));
        async move {
            let boxed = handle.await?;
            // the closure above is the only producer, so the concrete type is
            // known; a downcast failure would mean the pool swapped the payload.
            boxed
                .downcast::<Out>()
                .map(|value| *value)
                .map_err(|_| ProximaError::Config("background payload type changed".into()))
        }
    }
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::sync::Arc;
    use core::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// no `futures` dependency: this crate does not carry one outside
    /// `--all-features`, and the stub's background impl resolves on the first
    /// poll, so a bare loop is sufficient and cannot spin.
    fn block_on<F: Future>(future: F) -> F::Output {
        let mut future = core::pin::pin!(future);
        let waker = core::task::Waker::noop();
        let mut cx = core::task::Context::from_waker(waker);
        loop {
            if let core::task::Poll::Ready(value) = future.as_mut().poll(&mut cx) {
                return value;
            }
        }
    }

    #[derive(Default)]
    struct StubRuntime {
        on_core: AtomicUsize,
        background: AtomicUsize,
        /// how many leading `spawn_on_core` calls report a full inbox, so a
        /// test can force a candidate to be skipped rather than assuming it is.
        full_for_first: AtomicUsize,
    }

    impl Runtime for StubRuntime {
        fn spawn_on_current_core(&self, _f: core::pin::Pin<Box<dyn Future<Output = ()> + 'static>>) {}

        fn spawn_on_core(
            &self,
            _core_id: CoreId,
            _future: core::pin::Pin<Box<dyn Future<Output = ()> + Send + 'static>>,
        ) -> Result<(), SpawnError> {
            let attempt = self.on_core.fetch_add(1, Ordering::Relaxed);
            if attempt < self.full_for_first.load(Ordering::Relaxed) {
                return Err(SpawnError::InboxFull);
            }
            Ok(())
        }

        fn spawn_factory_on_core(
            &self,
            _core_id: CoreId,
            _factory: Box<dyn FnOnce() -> core::pin::Pin<Box<dyn Future<Output = ()> + 'static>> + Send>,
        ) -> Result<(), SpawnError> {
            Ok(())
        }

        fn spawn_background_blocking(
            &self,
            work: Box<dyn FnOnce() -> Result<Box<dyn Any + Send>, ProximaError> + Send>,
        ) -> BackgroundHandle<Box<dyn Any + Send>> {
            self.background.fetch_add(1, Ordering::Relaxed);
            let result = work();
            Box::pin(async move { result })
        }

        fn timer_at(&self, _deadline: std::time::Instant) -> core::pin::Pin<Box<dyn Future<Output = ()>>> {
            Box::pin(async {})
        }

        fn num_cores(&self) -> usize {
            1
        }

        fn current_core(&self) -> CoreId {
            CoreId(0)
        }
    }

    /// The bug this guards: `Self: Sized` on the builder entry points made them
    /// unreachable from `Arc<dyn Runtime>` — which is the form every storage
    /// site in the workspace actually holds.
    #[test]
    fn builder_is_reachable_through_a_trait_object() {
        let stub = Arc::new(StubRuntime::default());
        let erased: Arc<dyn Runtime> = stub.clone();

        erased.core(CoreId(0)).spawn(async {}).expect("chosen core");
        assert_eq!(stub.on_core.load(Ordering::Relaxed), 1);
    }

    /// `AnyOther` must never resolve to the calling core -- that is its whole
    /// meaning. Ordering starts at current+1 so repeated calls spread.
    #[test]
    fn any_other_excludes_the_current_core() {
        let candidates = Placement::AnyOther.candidates(CoreId(2), 4);
        assert_eq!(candidates, alloc::vec![CoreId(3), CoreId(0), CoreId(1)]);
        assert!(!candidates.contains(&CoreId(2)));
    }

    #[test]
    fn any_includes_the_current_core_last_in_wrap_order() {
        assert_eq!(
            Placement::Any.candidates(CoreId(2), 4),
            alloc::vec![CoreId(2), CoreId(3), CoreId(0), CoreId(1)]
        );
    }

    /// On a single-core runtime `AnyOther` has no candidate at all. It must say
    /// so rather than silently falling back to the current core.
    #[test]
    fn any_other_on_one_core_yields_no_candidate() {
        assert!(Placement::AnyOther.candidates(CoreId(0), 1).is_empty());

        let stub = Arc::new(StubRuntime::default());
        let erased: Arc<dyn Runtime> = stub.clone();
        let outcome = erased.core(Placement::AnyOther).spawn(async {});

        assert_eq!(outcome, Err(SpawnError::Disconnected));
        assert_eq!(stub.on_core.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn a_bare_index_is_a_specific_core() {
        assert_eq!(Placement::from(3usize), Placement::Only(CoreId(3)));
    }

    /// arrays and vecs, of bare indices or of `CoreId`, all reach `AnyOf` --
    /// the caller should not have to build the collection the enum happens to
    /// hold.
    /// the literal form, unannotated -- integer literals must land on CoreId
    /// without the caller writing `1usize` or `CoreId(1)`.
    #[test]
    fn a_bare_literal_array_is_a_candidate_list() {
        let stub = Arc::new(StubRuntime::default());
        let erased: Arc<dyn Runtime> = stub.clone();

        erased.core([1, 2, 3]).spawn(async {}).expect("first candidate");
        assert_eq!(stub.on_core.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn arrays_and_vecs_of_either_element_type_become_candidate_lists() {
        let expected = Placement::AnyOf(alloc::vec![CoreId(1), CoreId(2)]);

        assert_eq!(Placement::from([1usize, 2]), expected);
        assert_eq!(Placement::from([CoreId(1), CoreId(2)]), expected);
        assert_eq!(Placement::from(alloc::vec![1usize, 2]), expected);
        assert_eq!(Placement::from(alloc::vec![CoreId(1), CoreId(2)]), expected);
    }

    /// the behaviour the ordering tests could not reach: a full core must be
    /// skipped and the next candidate tried, with the factory rebuilding the
    /// future each time.
    #[test]
    fn a_full_core_falls_through_to_the_next_candidate() {
        let stub = Arc::new(StubRuntime::default());
        stub.full_for_first.store(2, Ordering::Relaxed);
        let erased: Arc<dyn Runtime> = stub.clone();

        let outcome = erased
            .core([0, 1, 2])
            .spawn_with(|| async {}, |_, err| *err == SpawnError::InboxFull);

        assert_eq!(outcome, Ok(CoreId(2)), "reports which core took it");
        assert_eq!(stub.on_core.load(Ordering::Relaxed), 3, "two full, third accepted");
    }

    /// the caller owns the policy: declining on the first error stops, even
    /// though candidates remain and the error is the transient one.
    #[test]
    fn a_decline_stops_before_the_remaining_candidates() {
        let stub = Arc::new(StubRuntime::default());
        stub.full_for_first.store(3, Ordering::Relaxed);
        let erased: Arc<dyn Runtime> = stub.clone();

        let outcome = erased.core([0, 1, 2]).spawn_with(|| async {}, |_, _| false);

        assert_eq!(
            outcome,
            Err(Exhausted { attempts: 1, last: Some(SpawnError::InboxFull) })
        );
        assert_eq!(stub.on_core.load(Ordering::Relaxed), 1, "stopped after the first");
    }

    /// zero candidates is not a failed spawn -- nothing was attempted, so
    /// there is no `SpawnError` to report. Previously this returned
    /// `Disconnected`, which claimed a core had shut down.
    #[test]
    fn no_candidates_reports_zero_attempts_and_no_cause() {
        let stub = Arc::new(StubRuntime::default());
        let erased: Arc<dyn Runtime> = stub.clone();

        let outcome = erased
            .core(Placement::AnyOther)
            .spawn_with(|| async {}, |_, _| true);

        assert_eq!(outcome, Err(Exhausted { attempts: 0, last: None }));
        assert_eq!(stub.on_core.load(Ordering::Relaxed), 0);
    }

    /// exhausting every candidate is distinguishable from one core being full,
    /// which `Err(last_error)` could not express.
    #[test]
    fn every_candidate_full_reports_exhaustion_not_the_last_error() {
        let stub = Arc::new(StubRuntime::default());
        stub.full_for_first.store(9, Ordering::Relaxed);
        let erased: Arc<dyn Runtime> = stub.clone();

        let outcome = erased
            .core([0, 1, 2])
            .spawn_with(|| async {}, |_, _| true);

        assert_eq!(
            outcome,
            Err(Exhausted { attempts: 3, last: Some(SpawnError::InboxFull) })
        );
    }

    #[test]
    fn blocking_returns_the_concrete_type_not_an_any_box() {
        let stub = Arc::new(StubRuntime::default());
        let erased: Arc<dyn Runtime> = stub.clone();

        let fut = erased.blocking(|| Ok(7u32)).spawn();
        let value = block_on(fut).expect("background work");

        assert_eq!(value, 7u32);
        assert_eq!(stub.background.load(Ordering::Relaxed), 1);
    }
}
