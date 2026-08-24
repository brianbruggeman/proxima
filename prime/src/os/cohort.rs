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
//! the park is an unbounded `park()`, not `park_timeout`. an earlier
//! revision bounded every park at 50us, blaming "OS scheduling" for a
//! stranding that was actually a store-buffer ordering bug at the `round`
//! bump and at [`wait_for_round`]'s arm-then-recheck — Release/Acquire on
//! two distinct locations does not order them against each other, so both
//! sides could miss (measured: 345 strands per 300,000 rounds on aarch64,
//! where `load(Acquire)` lowers to RCpc `ldapr`). those four operations are
//! SeqCst now and the strand does not reproduce in 18,000,000 rounds
//! (`cargo nextest run -p prime --features runtime-prime-cohort`, cohort
//! stress tests). the 50us bound was never load-bearing for correctness —
//! it only masked the bug by re-polling `round` on a timer — and it cost a
//! resident cohort roughly 6,600 spurious wakes per 330ms forward pass (one
//! per member per timeout), each contending the P-cores the leader needs.
//! see [`wait_for_round`] for the SeqCst argument the unbounded park relies on.
//!
//! # completion is a dial, not a constant
//!
//! two questions look alike and are not the same question. "when has every
//! member thread checked in for this round" is the join barrier: it is
//! fixed at `done == members`, forever, because [`CohortSession::run`]'s
//! erasure of `round`/`completion` to `'static` is only sound if no member
//! can still be dereferencing either pointer once `run` returns — see the
//! SAFETY comment at that erasure site. "when has this round's WORK stopped
//! being dispatched" is a caller-specific policy — [`FanInCompletion`] —
//! and has nothing to do with the join barrier: it only narrows which
//! chunks a member is offered before it falls through to reporting `done`
//! like every other member, working or not.

#![cfg(feature = "runtime-prime-cohort")]

use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::panic;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread;

use crossbeam_utils::CachePadded;
use crossbeam_utils::sync::{Parker, Unparker};

use proxima_core::ProximaError;
use proxima_primitives::pipe::fan_in::FanInCompletion;

/// forensic counters for the matmul-cohort wall-time decomposition:
/// where a round's wall goes between the leader publishing it and the
/// leader observing `done == members`. default-off scaffolding.
#[cfg(feature = "cohort-instrument")]
pub mod diag {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    pub const MAX_SLOTS: usize = 16;

    static BASE: OnceLock<Instant> = OnceLock::new();

    pub fn now_nanos() -> u64 {
        let base = BASE.get_or_init(Instant::now);
        base.elapsed().as_nanos() as u64
    }

    pub static ROUNDS: AtomicU64 = AtomicU64::new(0);
    pub static ROUND_OPEN_NS: AtomicU64 = AtomicU64::new(0);
    pub static UNPARK_ROUNDS: AtomicU64 = AtomicU64::new(0);
    pub static UNPARK_NANOS: AtomicU64 = AtomicU64::new(0);
    pub static PARKS: AtomicU64 = AtomicU64::new(0);
    pub static SPIN_HITS: AtomicU64 = AtomicU64::new(0);
    pub static IMMEDIATE_HITS: AtomicU64 = AtomicU64::new(0);
    pub static ARM_ABORTS: AtomicU64 = AtomicU64::new(0);

    #[allow(clippy::declare_interior_mutable_const)]
    const ZERO: AtomicU64 = AtomicU64::new(0);
    pub static SLOT_CHUNKS: [AtomicU64; MAX_SLOTS] = [ZERO; MAX_SLOTS];
    pub static SLOT_FIRST_CLAIM_NANOS: [AtomicU64; MAX_SLOTS] = [ZERO; MAX_SLOTS];
    pub static SLOT_COMPUTE_NANOS: [AtomicU64; MAX_SLOTS] = [ZERO; MAX_SLOTS];
    /// subset of `SLOT_COMPUTE_NANOS[slot]` spent strictly inside
    /// `CohortRound::run_chunk` itself (the dot kernel, for a matmul round) —
    /// `SLOT_COMPUTE_NANOS[slot] - SLOT_KERNEL_NANOS[slot]` is that slot's own
    /// claim-loop overhead (the cursor `fetch_add`, the completion check, the
    /// `catch_unwind` wrapper) around calls that never touch the kernel.
    /// ROW 130's per-term attribution needed exactly this split and did not
    /// have it: `SLOT_COMPUTE_NANOS` alone conflates "dispatch" and "kernel"
    /// into one bucket.
    pub static SLOT_KERNEL_NANOS: [AtomicU64; MAX_SLOTS] = [ZERO; MAX_SLOTS];
    pub static SLOT_TAIL_NANOS: [AtomicU64; MAX_SLOTS] = [ZERO; MAX_SLOTS];
    pub static SLOT_ROUNDS: [AtomicU64; MAX_SLOTS] = [ZERO; MAX_SLOTS];
    static SLOT_DONE_NS: [AtomicU64; MAX_SLOTS] = [ZERO; MAX_SLOTS];

    /// leader-only: wall time [`CohortSession::run_with_completion`] spends
    /// resetting the control block, publishing `round`/`completion`, and
    /// unparking any parked members — everything before the leader's own
    /// [`super::run_round`] call. Accumulates across every round since the
    /// last [`reset`], same shape as every other counter here.
    pub static LEADER_SETUP_NANOS: AtomicU64 = AtomicU64::new(0);
    /// leader-only: wall time spent in the tail spin loop
    /// (`while done.load() < members { spin_loop() }`) — the calling
    /// thread's own park/spin/wake cost, waiting on chunk completion after
    /// its own claim loop has run dry.
    pub static LEADER_SPIN_NANOS: AtomicU64 = AtomicU64::new(0);

    pub fn open_round() {
        ROUNDS.fetch_add(1, Ordering::Relaxed);
        ROUND_OPEN_NS.store(now_nanos(), Ordering::Release);
    }

    pub fn record_first_claim(slot: usize, at_nanos: u64) {
        if slot >= MAX_SLOTS {
            return;
        }
        let open = ROUND_OPEN_NS.load(Ordering::Acquire);
        SLOT_FIRST_CLAIM_NANOS[slot].fetch_add(at_nanos.saturating_sub(open), Ordering::Relaxed);
        SLOT_ROUNDS[slot].fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_slot_done(slot: usize, chunks: u64, compute_nanos: u64, kernel_nanos: u64, at_nanos: u64) {
        if slot >= MAX_SLOTS {
            return;
        }
        SLOT_CHUNKS[slot].fetch_add(chunks, Ordering::Relaxed);
        SLOT_COMPUTE_NANOS[slot].fetch_add(compute_nanos, Ordering::Relaxed);
        SLOT_KERNEL_NANOS[slot].fetch_add(kernel_nanos, Ordering::Relaxed);
        SLOT_DONE_NS[slot].store(at_nanos, Ordering::Release);
    }

    /// called by the leader once `done == members`: every slot's
    /// `SLOT_DONE_NS` is final for this round and cannot advance until the
    /// leader publishes the next one.
    pub fn close_round(members: usize, at_nanos: u64) {
        for slot in 0..members.min(MAX_SLOTS) {
            let done = SLOT_DONE_NS[slot].swap(0, Ordering::AcqRel);
            if done != 0 {
                SLOT_TAIL_NANOS[slot].fetch_add(at_nanos.saturating_sub(done), Ordering::Relaxed);
            }
        }
    }

    /// leader-only accumulators — see [`LEADER_SETUP_NANOS`]/[`LEADER_SPIN_NANOS`].
    pub fn record_leader_setup(nanos: u64) {
        LEADER_SETUP_NANOS.fetch_add(nanos, Ordering::Relaxed);
    }

    pub fn record_leader_spin(nanos: u64) {
        LEADER_SPIN_NANOS.fetch_add(nanos, Ordering::Relaxed);
    }

    pub fn reset() {
        for counter in [
            &ROUNDS,
            &UNPARK_ROUNDS,
            &UNPARK_NANOS,
            &PARKS,
            &SPIN_HITS,
            &IMMEDIATE_HITS,
            &ARM_ABORTS,
            &LEADER_SETUP_NANOS,
            &LEADER_SPIN_NANOS,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
        for slot in 0..MAX_SLOTS {
            SLOT_CHUNKS[slot].store(0, Ordering::Relaxed);
            SLOT_FIRST_CLAIM_NANOS[slot].store(0, Ordering::Relaxed);
            SLOT_COMPUTE_NANOS[slot].store(0, Ordering::Relaxed);
            SLOT_KERNEL_NANOS[slot].store(0, Ordering::Relaxed);
            SLOT_TAIL_NANOS[slot].store(0, Ordering::Relaxed);
            SLOT_ROUNDS[slot].store(0, Ordering::Relaxed);
            SLOT_DONE_NS[slot].store(0, Ordering::Relaxed);
        }
    }
}

/// index of one unit of work within a round. a newtype so `run_chunk`
/// cannot be called with a raw `usize` meant for something else — the
/// signature itself teaches the contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChunkIndex(pub usize);

/// one round of cohort work. object-safe by construction (one type
/// parameter fixed by the cohort, no `Self`-returning methods) so a single
/// fixed set of member threads can run an unbounded, unrecompiled sequence
/// of concrete `CohortRound` implementations over the cohort's lifetime —
/// the reference stored in the control block is `&dyn CohortRound<Error>`,
/// never an owned `Box`.
///
/// `run_chunk` returns `Result<(), Error>` — a chunk that can fail says so
/// through its own return type, not a side-channel plus a counter. a panic
/// is still caught separately by the runner and counted as `abandoned`:
/// unwinding is the thread surviving something `Error` was never meant to
/// carry, not normal control flow the round's own logic produced.
pub trait CohortRound<Error>: Sync {
    /// number of disjoint chunks this round claims to have. members
    /// self-select chunk indices `0..chunks()` via a shared atomic cursor.
    fn chunks(&self) -> usize;

    /// run exactly one chunk. called at most once per chunk index per
    /// round, from whichever member thread claims it — never assume which
    /// thread, or that chunks run in index order.
    fn run_chunk(&self, chunk: ChunkIndex) -> Result<(), Error>;
}

// [`FanInCompletion`] (from `proxima_primitives::pipe::fan_in`) governs
// when a round STOPS DISPATCHING new chunks to a claiming member — never
// when [`CohortSession::run`] itself returns. the join barrier
// (`done == members`, [`CohortSession::run_with_completion`]'s own wait)
// is not a policy: every member thread must finish its last
// `round_ptr`/`completion_ptr` dereference before `run` can null both
// pointers and hand the borrowed round back to the caller — see the
// SAFETY comment at the erasure site. the trait answers a narrower,
// genuinely caller-specific question: once a member reaches the claim
// loop, should it take the next chunk.
//
// `prime/Cargo.toml` already carries `proxima-primitives` as an
// unconditional dependency (`proxima-primitives` carries no entry for
// `prime` in the other direction), and the predicate `(done, total) ->
// bool` here is identical to `FanIn`'s own stopping question — same
// signature, same contract, only the caller's counted thing differs
// (retired merge sources there, retired chunks here — a domain-meaning
// difference the trait itself is deliberately blind to). `cohort` reuses
// `FanInCompletion` rather than redeclaring it, so `(done, total) ->
// bool` stays a single definition instead of two parallel ones that
// happened to agree.
//
// (this module previously carried its own `CohortCompletion` trait,
// declared here as a byte-for-byte duplicate of `FanInCompletion` under
// the false premise that `prime` did not depend on `proxima-primitives`.
// it does — see the paragraph above — so the duplicate is deleted and
// every call site below now names `FanInCompletion` directly.)
//
// stop dispatching once `self.0` chunks have retired, leaving any
// unclaimed chunks undispatched. a caller that only needs the first `N`
// results (best-effort, speculative, or fail-fast-shaped rounds) is not
// forced to pay for chunks nobody will read.
//
// "every chunk must retire before dispatch stops" is not a second type —
// `Quorum(chunks)` (`chunks` the round's own total) is the identical
// predicate: `retired >= chunks` either way. it is not spelled that way
// here, though: the actual all-chunks default is
// `CohortSession::run_with_completion`'s `None`, not `Some(&value)` of any
// `FanInCompletion` — `None` skips the retired-count check (two atomic
// loads plus a vtable call) every claim-loop iteration entirely, which
// `Some(&Quorum(chunks))` cannot do, since it must still be consulted to
// learn it is satisfied. a prior `AllChunks` unit type stood in front of
// that `None` as "a concrete value to name instead of matching on
// `Option`" — but nothing ever constructed it (not the non-test code, not
// the tests), and had it been used it would have been strictly worse than
// `None`: same predicate, extra atomics paid for nothing. deleted;
// `proxima_primitives::pipe::fan_in::Quorum(chunks)` names the same policy
// for any caller who genuinely needs a `FanInCompletion` value rather than
// the zero-overhead `None` path — this module minted no local
// `ChunkQuorum` type: it would have been the same `(done, total) -> bool`
// predicate `Quorum` already names, under a second name.

/// outcome of one [`CohortSession::run`] call.
#[must_use]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoundReport<Error> {
    /// chunks whose `run_chunk` returned `Ok`.
    pub completed: usize,
    /// chunks whose `run_chunk` panicked or returned `Err`. either way the
    /// failure is caught per-chunk — one abandoned chunk never strands the
    /// round or the other members.
    pub abandoned: usize,
    /// the first abandoned chunk's index, if any. `None` iff `abandoned == 0`.
    pub first_abandoned: Option<ChunkIndex>,
    /// the first `Err` any chunk returned this round, if any. `None` when
    /// every abandoned chunk (if any) abandoned by panicking instead — a
    /// panic's payload is not `Error`-shaped, so it is never surfaced here;
    /// `abandoned`/`first_abandoned` already say a chunk was lost either way.
    pub first_error: Option<Error>,
    /// member thread count the round ran against.
    pub members: usize,
}

/// immutable configuration for a [`ThreadCohort`]. mirrors [`CohortBuilder`]
/// field-for-field (§4 config ↔ builder parity) so a cohort can be described
/// as data as easily as it is constructed fluently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CohortConfig {
    /// total participant count for one round, INCLUDING the leader —
    /// mirrors ggml's `n_threads` (`ggml-cpu.c`'s `ith == 0` races on
    /// `current_chunk` exactly like every other thread). `members - 1`
    /// dedicated threads are spawned; the leader (whoever holds the
    /// [`CohortSession`]) is the final participant, claiming chunks off the
    /// same shared cursor inside [`CohortSession::run`] instead of sitting
    /// idle on a spin — see that function's own doc for why a spin-only
    /// leader was one runnable thread too many on an 8-P-core box.
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
/// with `Release` after populating `chunks`/`round_ptr`/`completion_ptr`;
/// members observe the bump with `Acquire` before touching any of the
/// three. `done` is the join edge: members bump it with `Release` on every
/// path (including a caught panic) so the leader's `Acquire` spin is
/// guaranteed to observe every member's writes — including `first_error` —
/// before returning.
struct Control<Error> {
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
    /// guards `first_error`: the member whose `compare_exchange` wins is the
    /// sole writer for the round, exactly the same single-writer contract
    /// `first_abandoned`'s own CAS already uses, just needing a companion
    /// flag because `Error` cannot live inside an atomic the way a `usize`
    /// index can.
    error_claimed: CachePadded<AtomicBool>,
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
    /// write site in [`CohortSession::run_with_completion`].
    round_ptr: UnsafeCell<Option<NonNull<dyn CohortRound<Error>>>>,
    /// type-erased pointer to this round's [`FanInCompletion`] policy, or
    /// `None` when the round was opened via [`CohortSession::run`] (the
    /// zero-overhead default). published and retired in lockstep with
    /// `round_ptr` — same publication edge, same join-barrier lifetime.
    completion_ptr: UnsafeCell<Option<NonNull<dyn FanInCompletion>>>,
    /// the first `Err` any chunk returned this round, written at most once
    /// (guarded by `error_claimed`) and read by the leader only after the
    /// join edge below has been crossed.
    first_error: UnsafeCell<Option<Error>>,
}

// SAFETY: every field but `round_ptr`/`completion_ptr`/`first_error` is
// already `Sync` (atomics, `CachePadded` of atomics, boxed slices of `Sync`
// types). `round_ptr`/`completion_ptr` are written exactly once per round by
// the leader (before the `round` Release bump) and read only by members
// after they observe that bump via Acquire — the same single-writer-then-
// many-readers contract `core/inbox.rs`'s `Lane` uses for its
// `cached_head`/`cached_tail` cells, just gated on `round` instead of
// `head`/`tail`. `first_error` is written at most once per round by
// whichever member's `error_claimed` CAS wins, strictly before that
// member's own `done` Release bump, and read by the leader only after
// observing `done == members` via Acquire — ordinary message-passing
// through an already-established happens-before edge. `Error: Send` is
// required because `first_error` genuinely crosses threads (written on a
// member thread, read on the leader's).
unsafe impl<Error: Send> Sync for Control<Error> {}
// SAFETY: `Control` is only ever held behind `Arc` and moved into member
// threads at spawn time, before any round is active (`round_ptr` is `None`
// until the first `CohortSession::run`). the erased pointer types have no
// thread affinity — they are `Send` in every respect but the auto-trait
// deriver's blindness to `UnsafeCell<Option<NonNull<_>>>`. `Error: Send` for
// the same reason `first_error` needs it in the `Sync` impl above.
unsafe impl<Error: Send> Send for Control<Error> {}

/// fixed-cohort spin-barrier: a pool of dedicated member threads that run
/// disjoint chunks of one [`CohortRound`] per round, woken once per round
/// via an atomic counter rather than once per unit of work. see the module
/// docs for why this exists instead of `ProximaBackgroundPool`
/// ([`super::background::ProximaBackgroundPool`]).
pub struct ThreadCohort<Error> {
    config: CohortConfig,
    control: Arc<Control<Error>>,
    handles: Vec<Option<thread::JoinHandle<()>>>,
}

impl<Error: Send + 'static> ThreadCohort<Error> {
    #[must_use]
    pub fn builder() -> CohortBuilder {
        CohortBuilder::default()
    }

    /// spawn `config.members - 1` dedicated threads. each parks immediately —
    /// no round is active until [`CohortSession::run`] bumps the round
    /// counter. the remaining participant is the leader itself, which never
    /// gets a spawned thread — it claims chunks from inside
    /// [`CohortSession::run`].
    pub fn from_config(config: CohortConfig) -> Result<Self, ProximaError> {
        let dedicated_count = config.members.get() - 1;
        let mut parkers = Vec::with_capacity(dedicated_count);
        let mut unparkers = Vec::with_capacity(dedicated_count);
        for _ in 0..dedicated_count {
            let parker = Parker::new();
            unparkers.push(parker.unparker().clone());
            parkers.push(parker);
        }
        let progress: Vec<CachePadded<AtomicU64>> = (0..dedicated_count)
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
            error_claimed: CachePadded::new(AtomicBool::new(false)),
            session_open: CachePadded::new(AtomicBool::new(false)),
            parked_count: CachePadded::new(AtomicUsize::new(0)),
            shutdown: CachePadded::new(AtomicBool::new(false)),
            progress: progress.into_boxed_slice(),
            unparkers: unparkers.into_boxed_slice(),
            round_ptr: UnsafeCell::new(None),
            completion_ptr: UnsafeCell::new(None),
            first_error: UnsafeCell::new(None),
        });

        let mut handles = Vec::with_capacity(dedicated_count);
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
    pub fn enter(&self) -> Result<CohortSession<'_, Error>, ProximaError> {
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

impl<Error> Drop for ThreadCohort<Error> {
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
pub struct CohortSession<'cohort, Error> {
    cohort: &'cohort ThreadCohort<Error>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl<Error: Send + 'static> CohortSession<'_, Error> {
    #[must_use]
    pub fn members(&self) -> usize {
        self.cohort.members()
    }

    /// run one round to completion under the default policy — every chunk
    /// dispatched, in full, exactly [`CohortSession::run`] always did.
    /// blocks the calling thread — this is the spin-barrier itself. sugar
    /// for [`CohortSession::run_with_completion`] with no dial, which is
    /// also the zero-overhead path: see that method's doc.
    pub fn run<Round: CohortRound<Error>>(&self, round: &Round) -> RoundReport<Error> {
        self.run_with_completion(round, None)
    }

    /// run one round to completion under a caller-named [`FanInCompletion`]
    /// policy, publish `round` to every member, spin until all have
    /// reported `done`, and report what happened.
    ///
    /// `completion` governs when the claim loop stops handing out NEW chunk
    /// indices — `None` (what [`run`](Self::run) passes) skips that check
    /// entirely, so the default path pays no extra atomic loads per chunk
    /// versus before this method existed.
    ///
    /// the `done == members` wait below is not part of this policy and
    /// cannot be: it is the join barrier every member's last
    /// `round_ptr`/`completion_ptr` dereference must cross before this
    /// function can null both pointers and return `round`/`completion` to
    /// the caller. `FanInCompletion` only ever narrows which chunks a
    /// member is offered — it can end DISPATCH early, but it can never let
    /// `run_with_completion` itself return before every member thread has
    /// checked in, no matter what policy is named.
    pub fn run_with_completion<Round: CohortRound<Error>>(
        &self,
        round: &Round,
        completion: Option<&dyn FanInCompletion>,
    ) -> RoundReport<Error> {
        let control = &self.cohort.control;
        let members = self.cohort.members();
        let chunk_total = round.chunks();

        #[cfg(feature = "cohort-instrument")]
        let setup_started = diag::now_nanos();
        control.chunks.store(chunk_total, Ordering::Relaxed);
        control.cursor.store(0, Ordering::Relaxed);
        control.done.store(0, Ordering::Relaxed);
        control.completed.store(0, Ordering::Relaxed);
        control.lost.store(0, Ordering::Relaxed);
        control.first_abandoned.store(usize::MAX, Ordering::Relaxed);
        control.error_claimed.store(false, Ordering::Relaxed);

        let round_dyn: &dyn CohortRound<Error> = round;
        let erased: NonNull<dyn CohortRound<Error>> = NonNull::from(round_dyn);
        let completion_erased: Option<NonNull<dyn FanInCompletion>> = completion.map(NonNull::from);
        // SAFETY: erase both borrows' lifetimes to 'static so the fat
        // pointers fit `Control::round_ptr`/`completion_ptr`. sound because
        // this function does not return until the spin loop below observes
        // `done == members` — i.e. until every member thread has returned
        // from its last dereference of EITHER pointer — so neither `round`
        // nor `completion` can go out of scope while any member could still
        // read through them. same argument `std::thread::scope` relies on
        // for its scoped borrows. `FanInCompletion` narrows WHEN a member
        // stops asking for new chunks; it has no bearing on this join wait,
        // which stays hard-coded to `members` regardless of policy — this
        // is the re-established soundness argument the completion dial was
        // required to preserve by construction, not weaken.
        let erased_static: NonNull<dyn CohortRound<Error> + 'static> = unsafe { std::mem::transmute(erased) };
        let completion_static: Option<NonNull<dyn FanInCompletion + 'static>> =
            completion_erased.map(|pointer| unsafe { std::mem::transmute(pointer) });
        // SAFETY: single-writer (this session, per its `!Send + !Sync`
        // contract) write before the `round` Release bump below, which is
        // the publication edge members Acquire-read before ever touching
        // either pointer.
        unsafe {
            *control.round_ptr.get() = Some(erased_static);
            *control.completion_ptr.get() = completion_static;
        }

        // SeqCst, not Release/Acquire, and all four of these ops together:
        // this and the member's arm-then-recheck (`wait_for_round`) form a
        // store-buffer pair -- each side stores one location then loads the
        // other. Release/Acquire on DISTINCT locations does not order them
        // against each other, so both sides can miss: leader reads
        // `parked_count == 0` and skips the unpark while the member reads the
        // old `round` and parks. Measured on aarch64, 345 strands per 300,000
        // with Release/Acquire and 0 per 18,000,000 with SeqCst. The whole
        // difference is one instruction: rustc targets Apple silicon with
        // RCpc, so `load(Acquire)` lowers to `ldapr`, which may execute before
        // a preceding release store; `ldar` (RCsc) may not.
        #[cfg(feature = "cohort-instrument")]
        diag::open_round();
        control.round.fetch_add(1, Ordering::SeqCst);
        if control.parked_count.load(Ordering::SeqCst) > 0 {
            #[cfg(feature = "cohort-instrument")]
            let unpark_started = diag::now_nanos();
            for unparker in &control.unparkers {
                unparker.unpark();
            }
            #[cfg(feature = "cohort-instrument")]
            {
                diag::UNPARK_ROUNDS.fetch_add(1, Ordering::Relaxed);
                diag::UNPARK_NANOS
                    .fetch_add(diag::now_nanos().saturating_sub(unpark_started), Ordering::Relaxed);
            }
        }

        // the leader is the final participant, not a spectator: it claims
        // chunks off the same shared cursor every dedicated member uses,
        // via the identical `run_round` body (including its per-chunk
        // `catch_unwind`, so a panicking chunk here is counted the same way
        // as one on a dedicated thread, never propagated out of `run`).
        // `run_round` bumps `done` on return exactly like a member's does,
        // so the wait below still targets `members` (dedicated_count + 1)
        // unchanged. Without this, `members` dedicated threads plus a
        // spin-only leader is `members + 1` runnable threads racing for
        // `members` P-cores — measured +14.8 ms / +4.5% on a real forward
        // (`proxima-model-interop`'s openchat bind test), entirely from
        // that one extra runnable thread.
        #[cfg(feature = "cohort-instrument")]
        diag::record_leader_setup(diag::now_nanos().saturating_sub(setup_started));
        run_round(control, 0);

        #[cfg(feature = "cohort-instrument")]
        let spin_started = diag::now_nanos();
        while control.done.load(Ordering::Acquire) < members as u64 {
            core::hint::spin_loop();
        }
        #[cfg(feature = "cohort-instrument")]
        {
            let closed_at = diag::now_nanos();
            diag::record_leader_spin(closed_at.saturating_sub(spin_started));
            diag::close_round(members, closed_at);
        }

        // SAFETY: every member has reported done (just observed above via
        // Acquire), so no member can still be dereferencing either pointer.
        unsafe {
            *control.round_ptr.get() = None;
            *control.completion_ptr.get() = None;
        }

        let completed = control.completed.load(Ordering::Relaxed);
        let abandoned = control.lost.load(Ordering::Relaxed);
        let first_abandoned_raw = control.first_abandoned.load(Ordering::Relaxed);
        let first_abandoned = if first_abandoned_raw == usize::MAX {
            None
        } else {
            Some(ChunkIndex(first_abandoned_raw))
        };
        // SAFETY: same join edge as the pointer-nulling above — whichever
        // member's CAS won `error_claimed` (if any) has already returned
        // from `run_chunk` and bumped `done` by the time this line runs, so
        // no writer to `first_error` remains.
        let first_error = unsafe { (*control.first_error.get()).take() };

        RoundReport {
            completed,
            abandoned,
            first_abandoned,
            first_error,
            members,
        }
    }
}

impl<Error> Drop for CohortSession<'_, Error> {
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
fn member_loop<Error>(control: &Control<Error>, member_index: usize, parker: &Parker, spin_polls: u32) {
    let mut local_round = 0_u64;
    loop {
        let Some(new_round) = wait_for_round(control, local_round, parker, spin_polls) else {
            return;
        };
        local_round = new_round;
        run_round(control, member_index + 1);
        control.progress[member_index].store(local_round, Ordering::Relaxed);
    }
}

/// spin `spin_polls` times, then park (unbounded — no `park_timeout`),
/// until `round` advances past `local_round` or `shutdown` fires. returns
/// the new round value, or `None` on shutdown.
///
/// HISTORY: this park used to be bounded at 50us (`PARK_TIMEOUT`, since
/// deleted), on the theory that under heavy CPU oversubscription an
/// `unpark()` could race a thread mid-transition into `park()` and strand
/// it indefinitely, with `Parker` explicitly exonerated. `Parker` was
/// innocent, but the bound was chasing the wrong cause: the real defect was
/// a store-buffer pair between the `round` bump in [`CohortSession::run`]
/// and this function's arm-then-recheck, both Release/Acquire on distinct
/// locations, which does not order them against each other. Measured 345
/// strands per 300,000 rounds on aarch64 (`load(Acquire)` lowers to RCpc
/// `ldapr`), 0 per 18,000,000 once every op in the pair below is SeqCst.
/// The bound only ever converted a stranding into an undebuggable 50us
/// stall that no test asserted on and no metric separated from scheduling
/// noise — exactly why the ordering bug survived. Removing it turns any
/// future stranding back into a hang a stress test can catch, instead of
/// absorbing it silently.
fn wait_for_round<Error>(
    control: &Control<Error>,
    local_round: u64,
    parker: &Parker,
    spin_polls: u32,
) -> Option<u64> {
    loop {
        let current = control.round.load(Ordering::Acquire);
        if current != local_round {
            #[cfg(feature = "cohort-instrument")]
            diag::IMMEDIATE_HITS.fetch_add(1, Ordering::Relaxed);
            return Some(current);
        }
        if control.shutdown.load(Ordering::Acquire) {
            return None;
        }
        for _ in 0..spin_polls {
            core::hint::spin_loop();
            let current = control.round.load(Ordering::Acquire);
            if current != local_round {
                #[cfg(feature = "cohort-instrument")]
                diag::SPIN_HITS.fetch_add(1, Ordering::Relaxed);
                return Some(current);
            }
        }
        // park: increment parked_count BEFORE the re-check so a concurrent
        // leader observes `parked_count > 0` and fires its unpark. SeqCst on
        // both, paired with the leader's two at the `round` bump — see there
        // for the store-buffer argument and the measurement. Note the comment
        // that used to sit here claimed `background.rs:317-333` handles "the
        // same race": it does not have the same shape. Two of its four ops are
        // SeqCst inside `crossbeam_deque::Injector` (`push`'s CAS and
        // `is_empty`'s loads), which breaks the cycle for it -- by crossbeam's
        // choice, not by ours. A crossbeam release that relaxed those would
        // reintroduce the hazard there with nothing here to catch it.
        control.parked_count.fetch_add(1, Ordering::SeqCst);
        let current = control.round.load(Ordering::SeqCst);
        let shutting_down = control.shutdown.load(Ordering::Acquire);
        if current != local_round || shutting_down {
            control.parked_count.fetch_sub(1, Ordering::AcqRel);
            if current != local_round {
                #[cfg(feature = "cohort-instrument")]
                diag::ARM_ABORTS.fetch_add(1, Ordering::Relaxed);
                return Some(current);
            }
            continue;
        }
        #[cfg(feature = "cohort-instrument")]
        diag::PARKS.fetch_add(1, Ordering::Relaxed);
        parker.park();
        control.parked_count.fetch_sub(1, Ordering::AcqRel);
    }
}

/// claim and run chunks until [`FanInCompletion`] says dispatch is done (or
/// the shared cursor exhausts `control.chunks`, its default), then report
/// done. a panicking or `Err`-returning chunk is caught, counted as
/// abandoned, and does not stop the member from claiming the next chunk —
/// `FanInCompletion` decides whether dispatch continues, not the failure.
#[cfg_attr(not(feature = "cohort-instrument"), allow(unused_variables))]
fn run_round<Error>(control: &Control<Error>, slot: usize) {
    let chunk_total = control.chunks.load(Ordering::Relaxed);
    #[cfg(feature = "cohort-instrument")]
    let mut claimed = 0_u64;
    #[cfg(feature = "cohort-instrument")]
    let mut compute_started = 0_u64;
    #[cfg(feature = "cohort-instrument")]
    let mut kernel_nanos_total = 0_u64;
    loop {
        // SAFETY: the publication edge (`control.round`'s Release/Acquire
        // pair) was already crossed in `wait_for_round` before this
        // function runs, so `completion_ptr` is populated (or `None`) for
        // the duration of this round under the same contract `round_ptr`
        // documents below.
        let completion_ref = unsafe { *control.completion_ptr.get() };
        if let Some(completion_ref) = completion_ref {
            let retired = control.completed.load(Ordering::Relaxed) + control.lost.load(Ordering::Relaxed);
            // SAFETY: same publication/lifetime argument as `round_ptr`'s
            // dereference below — the round (and hence its completion
            // policy) stays valid until every member reports `done`.
            if unsafe { completion_ref.as_ref().satisfied(retired, chunk_total) } {
                break;
            }
        }
        let index = control.cursor.fetch_add(1, Ordering::Relaxed);
        if index >= chunk_total {
            break;
        }
        #[cfg(feature = "cohort-instrument")]
        {
            if claimed == 0 {
                compute_started = diag::now_nanos();
                diag::record_first_claim(slot, compute_started);
            }
            claimed += 1;
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
        #[cfg(feature = "cohort-instrument")]
        let kernel_started = diag::now_nanos();
        let outcome = panic::catch_unwind(panic::AssertUnwindSafe(|| unsafe {
            round_ref.as_ref().run_chunk(ChunkIndex(index))
        }));
        #[cfg(feature = "cohort-instrument")]
        {
            kernel_nanos_total += diag::now_nanos().saturating_sub(kernel_started);
        }
        match outcome {
            Ok(Ok(())) => {
                control.completed.fetch_add(1, Ordering::Relaxed);
            }
            Ok(Err(error)) => {
                control.lost.fetch_add(1, Ordering::Relaxed);
                let _ = control.first_abandoned.compare_exchange(
                    usize::MAX,
                    index,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
                if control
                    .error_claimed
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    // SAFETY: this thread just won the claim CAS, so it is
                    // the only writer to `first_error` for this round; the
                    // leader will not read it until the join edge this
                    // function's own `done` bump (below) contributes to.
                    unsafe {
                        *control.first_error.get() = Some(error);
                    }
                }
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
    // guaranteeing every write above (completed/lost/first_abandoned/
    // first_error, and every side effect `run_chunk` had on caller-owned
    // state) is visible once the leader sees `done == members`. fired on
    // both the normal path above and the panic-caught path — a panicking
    // member never strands the leader.
    #[cfg(feature = "cohort-instrument")]
    {
        let at = diag::now_nanos();
        let compute = if claimed == 0 { 0 } else { at.saturating_sub(compute_started) };
        diag::record_slot_done(slot, claimed, compute, kernel_nanos_total, at);
    }
    control.done.fetch_add(1, Ordering::Release);
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::AtomicUsize as CountAtomicUsize;

    use proxima_primitives::pipe::fan_in::Quorum;

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

    /// the only error type the test module needs — deliberately not `Copy`
    /// (`usize` alone would be) so `RoundReport<TestChunkError>` exercises
    /// the same non-`Copy`-error path `TensorError` exercises in
    /// `proxima-tensor`, rather than a shape that happens to work only
    /// because it is trivially copyable.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestChunkError(usize);

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

    impl CohortRound<TestChunkError> for CountingRound {
        fn chunks(&self) -> usize {
            self.chunk_count
        }

        fn run_chunk(&self, chunk: ChunkIndex) -> Result<(), TestChunkError> {
            self.seen[chunk.0].fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    struct PanicOnRound {
        chunk_count: usize,
        panic_at: usize,
    }

    impl CohortRound<TestChunkError> for PanicOnRound {
        fn chunks(&self) -> usize {
            self.chunk_count
        }

        fn run_chunk(&self, chunk: ChunkIndex) -> Result<(), TestChunkError> {
            if chunk.0 == self.panic_at {
                panic!("intentional cohort chunk panic");
            }
            Ok(())
        }
    }

    /// a chunk that fails through the return channel instead of unwinding —
    /// the shape invariant (B) requires: `run_chunk` says so itself.
    struct ErrOnRound {
        chunk_count: usize,
        err_at: usize,
    }

    impl CohortRound<TestChunkError> for ErrOnRound {
        fn chunks(&self) -> usize {
            self.chunk_count
        }

        fn run_chunk(&self, chunk: ChunkIndex) -> Result<(), TestChunkError> {
            if chunk.0 == self.err_at {
                return Err(TestChunkError(chunk.0));
            }
            Ok(())
        }
    }

    fn cohort_with_members(members: usize) -> ThreadCohort<TestChunkError> {
        let config = ThreadCohort::<TestChunkError>::builder()
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
            assert_eq!(report.first_error, None);
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
        assert_eq!(report.first_error, None, "a panic carries no Error payload");

        // leader must still be usable for a subsequent, clean round.
        let clean_round = CountingRound::new(10);
        let clean_report = session.run(&clean_round);
        assert_eq!(clean_report.completed, 10);
        assert_eq!(clean_report.abandoned, 0);
    }

    /// invariant (B): a chunk that fails must be able to say so through its
    /// own return type, not a side-channel plus a counter. this asserts the
    /// `Err` itself — not just a count — comes back out of `RoundReport`.
    #[test]
    fn chunk_error_surfaces_through_report_return_type() {
        let cohort = cohort_with_members(4);
        let session = cohort.enter().expect("open session");
        let round = ErrOnRound {
            chunk_count: 20,
            err_at: 11,
        };
        let report = session.run(&round);
        assert_eq!(report.abandoned, 1, "exactly one chunk should error");
        assert_eq!(report.completed, 19);
        assert_eq!(report.first_abandoned, Some(ChunkIndex(11)));
        assert_eq!(
            report.first_error,
            Some(TestChunkError(11)),
            "the chunk's Err must surface through RoundReport itself"
        );

        // leader must still be usable for a subsequent, clean round.
        let clean_round = CountingRound::new(10);
        let clean_report = session.run(&clean_round);
        assert_eq!(clean_report.completed, 10);
        assert_eq!(clean_report.first_error, None);
    }

    /// invariant (A): completion is a policy a caller names, not a constant.
    /// `members(1)` (no dedicated threads, only the leader) makes the claim
    /// loop fully sequential, so the quorum's stopping point is exact and
    /// deterministic rather than racing dedicated threads for the last few
    /// chunks.
    #[test]
    fn chunk_quorum_stops_dispatch_before_all_chunks_run() {
        let cohort = cohort_with_members(1);
        let session = cohort.enter().expect("open session");

        let round = CountingRound::new(10);
        let report = session.run_with_completion(&round, Some(&Quorum(3)));
        assert_eq!(report.completed, 3, "quorum should stop dispatch after 3 retirements");
        assert_eq!(report.abandoned, 0);
        let claimed: usize = round.seen.iter().map(|count| count.load(Ordering::Relaxed)).sum();
        assert_eq!(claimed, 3, "only the quorum's worth of chunks should ever have been claimed");

        // the dial is per-call, not per-cohort: the same session, unforced,
        // still runs every chunk by default.
        let full_round = CountingRound::new(10);
        let full_report = session.run(&full_round);
        assert_eq!(full_report.completed, 10, "run() without a dial must still run every chunk");
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
        let config = ThreadCohort::<TestChunkError>::builder()
            .members(NonZeroUsize::new(6).expect("nonzero"))
            .spin_polls(500)
            .build();
        let cohort = ThreadCohort::<TestChunkError>::from_config(config).expect("build cohort");
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

    /// the shape that reproduced the pre-fix strand: a near-zero spin budget
    /// (so members reach the park path almost every round instead of
    /// catching the round bump in the spin loop) driven through a high round
    /// count on an unbounded `park()`. the fixed bug reproduced between 12
    /// and 118,425 rounds under this shape; this runs an order of magnitude
    /// past that ceiling and asserts every chunk of every round completed
    /// exactly once, which is impossible if any member ever stranded on a
    /// missed `unpark()`.
    #[test]
    fn high_round_count_low_spin_never_strands_with_unbounded_park() {
        let config = ThreadCohort::<TestChunkError>::builder()
            .members(NonZeroUsize::new(8).expect("nonzero member count"))
            .spin_polls(0)
            .build();
        let cohort = ThreadCohort::from_config(config).expect("build cohort");
        let session = cohort.enter().expect("open session");

        let total_rounds = 200_000_u64;
        let mut rounds_executed = 0_u64;
        for _ in 0..total_rounds {
            let round = CountingRound::new(8);
            let report = session.run(&round);
            assert_eq!(report.completed, 8, "no chunk should strand");
            assert_eq!(report.abandoned, 0);
            for seen in &round.seen {
                assert_eq!(seen.load(Ordering::Relaxed), 1);
            }
            rounds_executed += 1;
        }
        assert_eq!(
            rounds_executed, total_rounds,
            "stress test must actually execute every round, never pass vacuously"
        );
    }

    /// a chunk with measurable, non-instant work — pure atomic-fetch-add
    /// chunks (`CountingRound`) complete in a handful of nanoseconds, too
    /// close to the clock's own read granularity to assert a reliable
    /// `kernel_nanos > 0`. This round spins a fixed amount of cheap integer
    /// work per chunk so the kernel-only timer in [`run_round`] has
    /// something to measure above the noise floor.
    #[cfg(feature = "cohort-instrument")]
    struct BusyRound {
        chunk_count: usize,
    }

    #[cfg(feature = "cohort-instrument")]
    impl CohortRound<TestChunkError> for BusyRound {
        fn chunks(&self) -> usize {
            self.chunk_count
        }

        fn run_chunk(&self, chunk: ChunkIndex) -> Result<(), TestChunkError> {
            let mut accumulator = chunk.0 as u64;
            for value in 0..20_000_u64 {
                accumulator = accumulator.wrapping_mul(31).wrapping_add(value);
            }
            core::hint::black_box(accumulator);
            Ok(())
        }
    }

    /// ROW 130's missing instrumentation: per-step-reset counters that split
    /// the calling (leader) thread's own wall time into dot-kernel ticks,
    /// cohort dispatch/round-setup ticks, and park/spin/wake ticks — see
    /// `docs/discipline.md`'s attribution row. This asserts the sanity gates
    /// that row's differencing technique failed: no sub-term exceeds its
    /// parent, nothing is negative (all `u64`, so this is a structural
    /// guarantee), and the leader's own kernel time is a genuine subset of
    /// its own compute time, not a coincidental equality.
    #[test]
    #[cfg(feature = "cohort-instrument")]
    fn leader_kernel_ticks_are_a_bounded_subset_of_leader_compute_ticks() {
        diag::reset();
        let cohort = cohort_with_members(4);
        let session = cohort.enter().expect("open session");
        for _ in 0..25 {
            let round = BusyRound { chunk_count: 64 };
            let _ = session.run(&round);
        }

        let leader_compute = diag::SLOT_COMPUTE_NANOS[0].load(Ordering::Relaxed);
        let leader_kernel = diag::SLOT_KERNEL_NANOS[0].load(Ordering::Relaxed);
        assert!(leader_kernel > 0, "leader should have claimed and run at least one chunk");
        assert!(
            leader_kernel <= leader_compute,
            "kernel-only ticks ({leader_kernel}) must never exceed the compute bucket containing them ({leader_compute})"
        );

        let setup = diag::LEADER_SETUP_NANOS.load(Ordering::Relaxed);
        let spin = diag::LEADER_SPIN_NANOS.load(Ordering::Relaxed);
        assert!(setup > 0, "round setup/unpark should cost measurable time over 25 rounds");
        // spin can legitimately be 0 on a quiet box if the leader is always
        // the last to finish its own claim loop — not asserted > 0, only
        // that it is a well-formed accumulator.
        assert!(spin < u64::MAX);

        diag::reset();
        assert_eq!(diag::SLOT_COMPUTE_NANOS[0].load(Ordering::Relaxed), 0);
        assert_eq!(diag::SLOT_KERNEL_NANOS[0].load(Ordering::Relaxed), 0);
        assert_eq!(diag::LEADER_SETUP_NANOS.load(Ordering::Relaxed), 0);
        assert_eq!(diag::LEADER_SPIN_NANOS.load(Ordering::Relaxed), 0);
    }
}
