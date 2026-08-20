//! fixed-cohort spin-barrier primitive for CPU-bound compute rounds.
//!
//! stage 1 only: a standalone, default-off addition to `prime`. nothing in
//! `proxima-tensor` depends on this yet — the cutover to a q4k matmul cohort
//! is a later stage and out of scope here.
//!
//! # why this exists (measured)
//!
//! same M1 Max, same file/prompt, CPU only: ggml holds 80.9% parallel
//! efficiency at 8 threads on `matmul_q4k_q8k_f32`; proxima holds 53.4%.
//! over 200 real calls at w=8: correct floor (kernel CPU / workers) is
//! 91,161 ns/call, measured wall is 153,763 ns/call (1.69x off), with
//! `ProximaBackgroundPool::spawn` + `recv_wait` costing 11,254 + 11,818 ns
//! of that gap per call.
//!
//! `ProximaBackgroundPool::spawn` (`background.rs:154`) pushes onto a shared
//! `crossbeam_deque::Injector`; its workers park on
//! `crossbeam_utils::sync::Parker`, which is `Mutex` + `Condvar` underneath
//! (verified: `crossbeam-utils/src/sync/parker.rs:314-317`,
//! `struct Inner { lock: Mutex<()>, cvar: Condvar }`) — an OS wake round
//! trip on every one of 1350 matmul calls per forward pass. ggml instead
//! wakes worker threads ONCE per graph (`ggml_graph_compute_kickoff`) and
//! spins on an atomic barrier between graph nodes (`ggml_barrier`,
//! `n_barrier`/`n_barrier_passed`, `ggml_thread_cpu_relax()`).
//! `ThreadCohort` amortizes the wake to once per session: a fixed set of
//! dedicated member threads spin-wait on a round counter and self-select
//! chunks via a shared cursor — the same shape.
//!
//! # why not teach `ProximaBackgroundPool` this instead
//!
//! `background.rs:299` (`inner.injector.steal()`) has one shared queue and
//! no `spawn_on(worker, job)`, so a barrier round has no way to know its own
//! cohort's cardinality. `background.rs:300-313` holds a worker to
//! completion of exactly one job, so a long-lived barrier residency would
//! starve the shared `Injector` the rest of the process needs. the cohort
//! therefore owns dedicated threads for the lifetime of the session, never
//! the shared pool.
//!
//! # reuse, not reinvention
//!
//! `crossbeam_utils::sync::{Parker, Unparker}` for the at-rest fallback
//! (same as `background.rs:37`), `crossbeam_utils::CachePadded` to kill
//! false sharing between control-block fields (as in `core/inbox.rs:165`),
//! the `parked_count` short-circuit notify trick (`background.rs:205`),
//! and join-on-drop (`background.rs:266-279`). no `Injector`, no `Job`
//! enum, no channel — a barrier round is not a heterogeneous job queue.
//! the park itself is `park_timeout`, not an unbounded `park` — see
//! [`PARK_TIMEOUT`] for why: under heavy CPU oversubscription a targeted
//! `unpark()` was observed arriving while its member was mid-transition
//! into `park()`, stranding it; a bounded timeout makes the wait
//! self-healing instead of chasing that race inside `Parker` itself.

#![cfg(feature = "runtime-prime-cohort")]

use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::panic;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use crossbeam_utils::CachePadded;
use crossbeam_utils::sync::{Parker, Unparker};

use proxima_core::ProximaError;

/// index of one unit of work within a round. a newtype so `run_chunk`
/// cannot be called with a raw `usize` meant for something else — the
/// signature itself teaches the contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChunkIndex(pub usize);

/// one round of cohort work. object-safe by construction (no generics, no
/// `Self`-returning methods) so a single fixed set of member threads can
/// run an unbounded, unrecompiled sequence of concrete `CohortRound`
/// implementations over the cohort's lifetime — the reference stored in
/// the control block is `&dyn CohortRound`, never an owned `Box`.
pub trait CohortRound: Sync {
    /// number of disjoint chunks this round claims to have. members
    /// self-select chunk indices `0..chunks()` via a shared atomic cursor.
    fn chunks(&self) -> usize;

    /// run exactly one chunk. called at most once per chunk index per
    /// round, from whichever member thread claims it — never assume which
    /// thread, or that chunks run in index order.
    fn run_chunk(&self, chunk: ChunkIndex);
}

/// outcome of one [`CohortSession::run`] call.
#[must_use]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoundReport {
    /// chunks whose `run_chunk` returned without panicking.
    pub completed: usize,
    /// chunks whose `run_chunk` panicked. the panic is caught per-chunk —
    /// one abandoned chunk never strands the round or the other members.
    pub abandoned: usize,
    /// the first abandoned chunk's index, if any. `None` iff `abandoned == 0`.
    pub first_abandoned: Option<ChunkIndex>,
    /// member thread count the round ran against.
    pub members: usize,
}

/// immutable configuration for a [`ThreadCohort`]. mirrors [`CohortBuilder`]
/// field-for-field (§4 config ↔ builder parity) so a cohort can be described
/// as data as easily as it is constructed fluently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CohortConfig {
    /// number of dedicated member threads. NOT the leader — the leader
    /// (whoever holds the [`CohortSession`]) never runs a chunk itself.
    pub members: NonZeroUsize,
    /// bounded spin budget, in `core::hint::spin_loop()` polls, a member
    /// spends waiting for the round counter to advance before parking.
    pub spin_polls: u32,
}

/// fluent builder for [`CohortConfig`]. `.build()` yields the immutable
/// config; [`ThreadCohort::from_config`] accepts that config directly, so
/// callers can round-trip through data with no divergent path.
#[derive(Debug, Clone, Copy)]
pub struct CohortBuilder {
    members: NonZeroUsize,
    spin_polls: u32,
}

impl Default for CohortBuilder {
    fn default() -> Self {
        Self {
            members: NonZeroUsize::MIN,
            spin_polls: 2_000,
        }
    }
}

impl CohortBuilder {
    #[must_use]
    pub fn members(mut self, members: NonZeroUsize) -> Self {
        self.members = members;
        self
    }

    #[must_use]
    pub fn spin_polls(mut self, spin_polls: u32) -> Self {
        self.spin_polls = spin_polls;
        self
    }

    #[must_use]
    pub fn build(self) -> CohortConfig {
        CohortConfig {
            members: self.members,
            spin_polls: self.spin_polls,
        }
    }
}

/// cache-line-padded shared state between the leader and every member
/// thread. `round` is the single publication edge: the leader bumps it
/// with `Release` after populating `chunks`/`round_ptr`; members observe
/// the bump with `Acquire` before touching either. `done` is the join
/// edge: members bump it with `Release` on every path (including a
/// caught panic) so the leader's `Acquire` spin is guaranteed to observe
/// every member's writes before returning.
struct Control {
    round: CachePadded<AtomicU64>,
    chunks: CachePadded<AtomicUsize>,
    /// ggml's `current_chunk` — members self-select disjoint work via
    /// `fetch_add(Relaxed)`; chunks are disjoint by construction so no
    /// ordering is required on the claim itself.
    cursor: CachePadded<AtomicUsize>,
    done: CachePadded<AtomicU64>,
    completed: CachePadded<AtomicUsize>,
    lost: CachePadded<AtomicUsize>,
    first_abandoned: CachePadded<AtomicUsize>,
    session_open: CachePadded<AtomicBool>,
    parked_count: CachePadded<AtomicUsize>,
    shutdown: CachePadded<AtomicBool>,
    /// last round each member finished, for external liveness inspection.
    /// not surfaced on `RoundReport` today — kept as the control block's
    /// per-member diagnostic slot the design calls for.
    progress: Box<[CachePadded<AtomicU64>]>,
    unparkers: Box<[Unparker]>,
    /// type-erased pointer to the round object currently in flight. valid
    /// exactly while `round` names an active round the leader has not yet
    /// observed `done == members` for. see the safety comment at the
    /// write site in [`CohortSession::run`].
    round_ptr: UnsafeCell<Option<NonNull<dyn CohortRound>>>,
}

// SAFETY: every field but `round_ptr` is already `Sync` (atomics,
// `CachePadded` of atomics, boxed slices of `Sync` types). `round_ptr`'s
// `UnsafeCell` is written exactly once per round by the leader (before the
// `round` Release bump) and read only by members after they observe that
// bump via Acquire — the same single-writer-then-many-readers contract
// `core/inbox.rs`'s `Lane` uses for its `cached_head`/`cached_tail` cells,
// just gated on `round` instead of `head`/`tail`.
unsafe impl Sync for Control {}
// SAFETY: `Control` is only ever held behind `Arc` and moved into member
// threads at spawn time, before any round is active (`round_ptr` is `None`
// until the first `CohortSession::run`). the erased pointer type itself has
// no thread affinity — it is `Send` in every respect but the auto-trait
// deriver's blindness to `UnsafeCell<Option<NonNull<_>>>`.
unsafe impl Send for Control {}

/// fixed-cohort spin-barrier: a pool of dedicated member threads that run
/// disjoint chunks of one [`CohortRound`] per round, woken once per round
/// via an atomic counter rather than once per unit of work. see the module
/// docs for why this exists instead of `ProximaBackgroundPool`
/// ([`super::background::ProximaBackgroundPool`]).
pub struct ThreadCohort {
    config: CohortConfig,
    control: Arc<Control>,
    handles: Vec<Option<thread::JoinHandle<()>>>,
}

impl ThreadCohort {
    #[must_use]
    pub fn builder() -> CohortBuilder {
        CohortBuilder::default()
    }

    /// spawn `config.members` dedicated threads. each parks immediately —
    /// no round is active until [`CohortSession::run`] bumps the round
    /// counter.
    pub fn from_config(config: CohortConfig) -> Result<Self, ProximaError> {
        let member_count = config.members.get();
        let mut parkers = Vec::with_capacity(member_count);
        let mut unparkers = Vec::with_capacity(member_count);
        for _ in 0..member_count {
            let parker = Parker::new();
            unparkers.push(parker.unparker().clone());
            parkers.push(parker);
        }
        let progress: Vec<CachePadded<AtomicU64>> = (0..member_count)
            .map(|_| CachePadded::new(AtomicU64::new(0)))
            .collect();

        let control = Arc::new(Control {
            round: CachePadded::new(AtomicU64::new(0)),
            chunks: CachePadded::new(AtomicUsize::new(0)),
            cursor: CachePadded::new(AtomicUsize::new(0)),
            done: CachePadded::new(AtomicU64::new(0)),
            completed: CachePadded::new(AtomicUsize::new(0)),
            lost: CachePadded::new(AtomicUsize::new(0)),
            first_abandoned: CachePadded::new(AtomicUsize::new(usize::MAX)),
            session_open: CachePadded::new(AtomicBool::new(false)),
            parked_count: CachePadded::new(AtomicUsize::new(0)),
            shutdown: CachePadded::new(AtomicBool::new(false)),
            progress: progress.into_boxed_slice(),
            unparkers: unparkers.into_boxed_slice(),
            round_ptr: UnsafeCell::new(None),
        });

        let mut handles = Vec::with_capacity(member_count);
        for (index, parker) in parkers.into_iter().enumerate() {
            let control_for_member = Arc::clone(&control);
            let spin_polls = config.spin_polls;
            let handle = thread::Builder::new()
                .name(format!("proxima-cohort-{index}"))
                .spawn(move || member_loop(&control_for_member, index, &parker, spin_polls))
                .map_err(|err| ProximaError::Config(format!("spawn cohort member: {err}")))?;
            handles.push(Some(handle));
        }

        Ok(Self {
            config,
            control,
            handles,
        })
    }

    #[must_use]
    pub fn config(&self) -> CohortConfig {
        self.config
    }

    #[must_use]
    pub fn members(&self) -> usize {
        self.config.members.get()
    }

    /// open the single session this cohort permits at a time. errs if a
    /// session is already open — `CohortSession` is `!Send + !Sync` so
    /// within one thread this only fires on a bug (nested `enter` before
    /// the prior session drops).
    pub fn enter(&self) -> Result<CohortSession<'_>, ProximaError> {
        self.control
            .session_open
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| ProximaError::Config("cohort session already open".into()))?;
        Ok(CohortSession {
            cohort: self,
            _not_send_sync: PhantomData,
        })
    }
}

impl Drop for ThreadCohort {
    fn drop(&mut self) {
        self.control.shutdown.store(true, Ordering::Release);
        for unparker in &self.control.unparkers {
            unparker.unpark();
        }
        for slot in &mut self.handles {
            if let Some(handle) = slot.take() {
                let _ = handle.join();
            }
        }
    }
}

/// an open cohort session. `!Send + !Sync` (via the raw-pointer marker) so
/// only the thread that opened it can drive rounds — `run` requires no
/// internal locking because there is, by construction, exactly one caller.
pub struct CohortSession<'cohort> {
    cohort: &'cohort ThreadCohort,
    _not_send_sync: PhantomData<*mut ()>,
}

impl CohortSession<'_> {
    #[must_use]
    pub fn members(&self) -> usize {
        self.cohort.members()
    }

    /// run one round to completion: publish `round` to every member,
    /// spin until all have reported `done`, and report what happened.
    /// blocks the calling thread — this is the spin-barrier itself.
    pub fn run<Round: CohortRound>(&self, round: &Round) -> RoundReport {
        let control = &self.cohort.control;
        let members = self.cohort.members();
        let chunk_total = round.chunks();

        control.chunks.store(chunk_total, Ordering::Relaxed);
        control.cursor.store(0, Ordering::Relaxed);
        control.done.store(0, Ordering::Relaxed);
        control.completed.store(0, Ordering::Relaxed);
        control.lost.store(0, Ordering::Relaxed);
        control.first_abandoned.store(usize::MAX, Ordering::Relaxed);

        let round_dyn: &dyn CohortRound = round;
        let erased: NonNull<dyn CohortRound> = NonNull::from(round_dyn);
        // SAFETY: erase the borrow's lifetime to 'static so the fat pointer
        // fits `Control::round_ptr`. sound because this function does not
        // return until the spin loop below observes `done == members` —
        // i.e. until every member thread has returned from its last
        // `round_ptr` dereference — so `round` cannot go out of scope while
        // any member could still read through the erased pointer. same
        // argument `std::thread::scope` relies on for its scoped borrows.
        let erased_static: NonNull<dyn CohortRound + 'static> =
            unsafe { std::mem::transmute(erased) };
        // SAFETY: single-writer (this session, per its `!Send + !Sync`
        // contract) write before the `round` Release bump below, which is
        // the publication edge members Acquire-read before ever touching
        // `round_ptr`.
        unsafe {
            *control.round_ptr.get() = Some(erased_static);
        }

        control.round.fetch_add(1, Ordering::Release);
        if control.parked_count.load(Ordering::Acquire) > 0 {
            for unparker in &control.unparkers {
                unparker.unpark();
            }
        }

        while control.done.load(Ordering::Acquire) < members as u64 {
            core::hint::spin_loop();
        }

        // SAFETY: every member has reported done (just observed above via
        // Acquire), so no member can still be dereferencing `round_ptr`.
        unsafe {
            *control.round_ptr.get() = None;
        }

        let completed = control.completed.load(Ordering::Relaxed);
        let abandoned = control.lost.load(Ordering::Relaxed);
        let first_abandoned_raw = control.first_abandoned.load(Ordering::Relaxed);
        let first_abandoned = if first_abandoned_raw == usize::MAX {
            None
        } else {
            Some(ChunkIndex(first_abandoned_raw))
        };

        RoundReport {
            completed,
            abandoned,
            first_abandoned,
            members,
        }
    }
}

impl Drop for CohortSession<'_> {
    fn drop(&mut self) {
        self.cohort
            .control
            .session_open
            .store(false, Ordering::Release);
    }
}

/// per-member thread body. loops forever: wait for the round counter to
/// advance (spin then park), run every chunk it can claim, report done,
/// repeat. returns only on `shutdown`.
fn member_loop(control: &Control, member_index: usize, parker: &Parker, spin_polls: u32) {
    let mut local_round = 0_u64;
    loop {
        let Some(new_round) = wait_for_round(control, local_round, parker, spin_polls) else {
            return;
        };
        local_round = new_round;
        run_round(control);
        control.progress[member_index].store(local_round, Ordering::Relaxed);
    }
}

/// worst-case latency for a member to notice a round it was not woken for.
/// bounds an otherwise-unbounded park: measured under heavy CPU
/// oversubscription (many parallel test processes each spinning multiple
/// dedicated threads), an `unpark()` racing a thread mid-transition into
/// `park()` can be delayed indefinitely by OS scheduling even though
/// crossbeam's `Parker` itself has no missed-wakeup window — the observed
/// failure was a real thread parked with its target `unpark()` already
/// consumed elsewhere in the storm of concurrent parks/unparks across
/// every cohort test's own dedicated threads. `park_timeout` makes the
/// wait self-healing: a member that missed its wake re-polls `round`
/// itself within one timeout instead of blocking forever.
const PARK_TIMEOUT: Duration = Duration::from_micros(50);

/// spin `spin_polls` times, then park (bounded by [`PARK_TIMEOUT`]), until
/// `round` advances past `local_round` or `shutdown` fires. returns the
/// new round value, or `None` on shutdown.
fn wait_for_round(
    control: &Control,
    local_round: u64,
    parker: &Parker,
    spin_polls: u32,
) -> Option<u64> {
    loop {
        let current = control.round.load(Ordering::Acquire);
        if current != local_round {
            return Some(current);
        }
        if control.shutdown.load(Ordering::Acquire) {
            return None;
        }
        for _ in 0..spin_polls {
            core::hint::spin_loop();
            let current = control.round.load(Ordering::Acquire);
            if current != local_round {
                return Some(current);
            }
        }
        // park: increment parked_count BEFORE the re-check so a concurrent
        // leader observes `parked_count > 0` and fires its unpark — the
        // same race `background.rs:317-333` handles for the shared pool.
        control.parked_count.fetch_add(1, Ordering::AcqRel);
        let current = control.round.load(Ordering::Acquire);
        let shutting_down = control.shutdown.load(Ordering::Acquire);
        if current != local_round || shutting_down {
            control.parked_count.fetch_sub(1, Ordering::AcqRel);
            if current != local_round {
                return Some(current);
            }
            continue;
        }
        parker.park_timeout(PARK_TIMEOUT);
        control.parked_count.fetch_sub(1, Ordering::AcqRel);
    }
}

/// claim and run chunks until the shared cursor exhausts `control.chunks`,
/// then report done. a panicking chunk is caught, counted as abandoned, and
/// does not stop the member from claiming the next chunk.
fn run_round(control: &Control) {
    let chunk_total = control.chunks.load(Ordering::Relaxed);
    loop {
        let index = control.cursor.fetch_add(1, Ordering::Relaxed);
        if index >= chunk_total {
            break;
        }
        // SAFETY: the publication edge (`control.round`'s Release/Acquire
        // pair) was already crossed in `wait_for_round` before this
        // function runs, so `round_ptr` is populated for the duration of
        // this round.
        let round_ref = unsafe { *control.round_ptr.get() };
        let Some(round_ref) = round_ref else {
            // should not happen given the publication edge above; treat
            // defensively as an abandoned chunk rather than panicking.
            control.lost.fetch_add(1, Ordering::Relaxed);
            let _ = control.first_abandoned.compare_exchange(
                usize::MAX,
                index,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
            continue;
        };
        // SAFETY: `round_ref` points at the `CohortRound` the leader
        // published for this round; it stays valid until every member
        // reports `done`, which this member has not yet done.
        let outcome = panic::catch_unwind(panic::AssertUnwindSafe(|| unsafe {
            round_ref.as_ref().run_chunk(ChunkIndex(index));
        }));
        match outcome {
            Ok(()) => {
                control.completed.fetch_add(1, Ordering::Relaxed);
            }
            Err(_) => {
                control.lost.fetch_add(1, Ordering::Relaxed);
                let _ = control.first_abandoned.compare_exchange(
                    usize::MAX,
                    index,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
            }
        }
    }
    // Release: this is the join edge the leader's Acquire spin observes,
    // guaranteeing every write above (completed/lost/first_abandoned, and
    // every side effect `run_chunk` had on caller-owned state) is visible
    // once the leader sees `done == members`. fired on both the normal
    // path above and the panic-caught path — a panicking member never
    // strands the leader.
    control.done.fetch_add(1, Ordering::Release);
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::AtomicUsize as CountAtomicUsize;

    use super::*;

    struct CountingAllocator;

    static ALLOC_COUNT: CountAtomicUsize = CountAtomicUsize::new(0);

    // SAFETY: delegates every call straight to `System`; the only added
    // behavior is a `Relaxed` counter bump, which is sound at any point in
    // the allocator contract.
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    #[global_allocator]
    static ALLOCATOR: CountingAllocator = CountingAllocator;

    struct CountingRound {
        chunk_count: usize,
        seen: Vec<AtomicUsize>,
    }

    impl CountingRound {
        fn new(chunk_count: usize) -> Self {
            Self {
                chunk_count,
                seen: (0..chunk_count).map(|_| AtomicUsize::new(0)).collect(),
            }
        }
    }

    impl CohortRound for CountingRound {
        fn chunks(&self) -> usize {
            self.chunk_count
        }

        fn run_chunk(&self, chunk: ChunkIndex) {
            self.seen[chunk.0].fetch_add(1, Ordering::Relaxed);
        }
    }

    struct PanicOnRound {
        chunk_count: usize,
        panic_at: usize,
    }

    impl CohortRound for PanicOnRound {
        fn chunks(&self) -> usize {
            self.chunk_count
        }

        fn run_chunk(&self, chunk: ChunkIndex) {
            if chunk.0 == self.panic_at {
                panic!("intentional cohort chunk panic");
            }
        }
    }

    fn cohort_with_members(members: usize) -> ThreadCohort {
        let config = ThreadCohort::builder()
            .members(NonZeroUsize::new(members).expect("nonzero member count"))
            .spin_polls(64)
            .build();
        ThreadCohort::from_config(config).expect("build cohort")
    }

    #[test]
    fn every_chunk_runs_exactly_once_over_many_rounds() {
        let cohort = cohort_with_members(4);
        let session = cohort.enter().expect("open session");
        for _ in 0..50 {
            let round = CountingRound::new(97);
            let report = session.run(&round);
            assert_eq!(report.completed, 97, "every chunk should complete");
            assert_eq!(report.abandoned, 0);
            for (index, seen) in round.seen.iter().enumerate() {
                assert_eq!(
                    seen.load(Ordering::Relaxed),
                    1,
                    "chunk {index} ran {} times, expected exactly 1",
                    seen.load(Ordering::Relaxed)
                );
            }
        }
    }

    #[test]
    fn abandoned_chunk_is_reported_and_leader_returns() {
        let cohort = cohort_with_members(4);
        let session = cohort.enter().expect("open session");
        let round = PanicOnRound {
            chunk_count: 20,
            panic_at: 7,
        };
        let report = session.run(&round);
        assert_eq!(report.abandoned, 1, "exactly one chunk should panic");
        assert_eq!(report.first_abandoned, Some(ChunkIndex(7)));
        assert_eq!(report.completed, 19);

        // leader must still be usable for a subsequent, clean round.
        let clean_round = CountingRound::new(10);
        let clean_report = session.run(&clean_round);
        assert_eq!(clean_report.completed, 10);
        assert_eq!(clean_report.abandoned, 0);
    }

    #[test]
    fn zero_allocations_per_round() {
        let cohort = cohort_with_members(4);
        let session = cohort.enter().expect("open session");
        // warm up over many small rounds: freshly spawned member threads
        // are not guaranteed to be scheduled by the OS in time to claim a
        // chunk in the very first round, and a member's first-ever
        // `catch_unwind` call lazily initializes a std thread-local the
        // first time it runs on that thread. many rounds give every member
        // repeated chances to be scheduled and pay that cost before we
        // start counting.
        for _ in 0..100 {
            let warmup = CountingRound::new(16);
            let _ = session.run(&warmup);
        }

        let round = CountingRound::new(32);
        let before = ALLOC_COUNT.load(Ordering::Relaxed);
        for _ in 0..200 {
            let _ = session.run(&round);
        }
        let after = ALLOC_COUNT.load(Ordering::Relaxed);
        assert_eq!(
            after, before,
            "expected zero allocations across 200 rounds, saw {}",
            after - before
        );
    }

    #[test]
    fn config_and_builder_round_trip() {
        let config = ThreadCohort::builder()
            .members(NonZeroUsize::new(6).expect("nonzero"))
            .spin_polls(500)
            .build();
        let cohort = ThreadCohort::from_config(config).expect("build cohort");
        assert_eq!(cohort.config(), config);
    }

    #[test]
    fn zero_chunk_round_completes_with_no_work() {
        let cohort = cohort_with_members(3);
        let session = cohort.enter().expect("open session");
        let round = CountingRound::new(0);
        let report = session.run(&round);
        assert_eq!(report.completed, 0);
        assert_eq!(report.abandoned, 0);
        assert_eq!(report.members, 3);
    }

    #[test]
    fn fewer_chunks_than_members_leaves_some_idle() {
        let cohort = cohort_with_members(8);
        let session = cohort.enter().expect("open session");
        let round = CountingRound::new(3);
        let report = session.run(&round);
        assert_eq!(report.completed, 3);
        assert_eq!(report.abandoned, 0);
        assert_eq!(report.members, 8);
        for seen in &round.seen {
            assert_eq!(seen.load(Ordering::Relaxed), 1);
        }
    }

    #[test]
    fn nested_enter_errs_while_session_open() {
        let cohort = cohort_with_members(2);
        let _session = cohort.enter().expect("first session opens");
        let second = cohort.enter();
        assert!(second.is_err(), "a second concurrent session must err");
    }

    #[test]
    fn session_reusable_after_drop() {
        let cohort = cohort_with_members(2);
        {
            let session = cohort.enter().expect("first session");
            let round = CountingRound::new(5);
            let _ = session.run(&round);
        }
        let session = cohort.enter().expect("session reopens after drop");
        let round = CountingRound::new(5);
        let report = session.run(&round);
        assert_eq!(report.completed, 5);
    }
}
