//! pinned per-core worker. composes C1 (Inbox) + C2 (TimerWheel) +
//! C3 (Reactor) + C4 (LocalExecutor) into the runtime worker that owns
//! one OS thread and one logical core.
//!
//! the worker loop drains cross-core spawn requests from the inbox,
//! polls the local executor's ready tasks, advances the timer wheel,
//! and parks on the reactor when idle (block until I/O ready or next
//! timer deadline). `Shutdown` request breaks the loop cleanly.

#![cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-reactor",
    any(
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-inbox-dynamic"
    ),
))]

use std::cell::{Cell, RefCell, UnsafeCell};
use std::future::Future;
use std::pin::Pin;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{self, Ordering};
use std::task::{Context, Poll};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use proxima_core::ProximaError;
use proxima_runtime::{CoreId, SpawnError, SpawnRequest};

#[cfg(not(feature = "runtime-prime-inbox-dynamic"))]
use super::super::core::inbox as inbox_impl;
#[cfg(feature = "runtime-prime-inbox-dynamic")]
use super::super::core::inbox_dynamic as inbox_impl;
use super::super::core::inline_task::InlineTask;
use super::super::core::local_executor::LocalExecutor;
use super::super::core::sized;
use super::super::core::timer::{Clock, Tick, TimerKey, TimerWheel};
use super::reactor::{Reactor, Wakeup};
use inbox_impl::Producer;

/// initial / busy-period spin count on an empty inbox before the worker
/// commits to a `reactor.turn` syscall. tuned for the bench's burst pattern
/// (producer feeds 1 task at a time; worker drains 1 while producer is
/// mid-burst). a kevent syscall is ~1μs; 256 spin_loops is ~100-300ns on
/// M1 — spinning that long hides the syscall cost when work is arriving.
///
/// P12: traces to `prime-runtime.toml`'s `[reactor]` section
/// (`spin_before_park_busy`), env-overridable per target
/// (`PRIME_REACTOR_SPIN_BEFORE_PARK_BUSY`) — see `build.rs`. this const
/// re-exports the generated `sized` value under its original name so the
/// call sites below are unchanged.
const SPIN_BEFORE_PARK_BUSY: u32 = sized::REACTOR_SPIN_BEFORE_PARK_BUSY;
/// post-idle spin count. spin only catches CROSS-CORE inbox traffic; if
/// recent wakes haven't been driven by the inbox, spin is wasted CPU. set
/// to 0 — when persistent idle is detected (no inbox arrivals over
/// `IDLE_PARK_THRESHOLD` consecutive parks), drop the spin entirely. lifts
/// back to BUSY the instant an inbox push lands.
///
/// the h2_load_5way bench (all wakes come from the reactor — same-core
/// I/O readiness, NOT cross-core inbox) drove this tuning: 477μs → 37μs
/// once we stopped spinning on the wrong channel. See
/// `docs/runtime-prime/discipline.md` v9-f and
/// `docs/pipe-to-metal/discipline.md` `reactor-spin-park` for the re-proof.
/// P12: `prime-runtime.toml`'s `[reactor].spin_before_park_idle`
/// (`PRIME_REACTOR_SPIN_BEFORE_PARK_IDLE`).
const SPIN_BEFORE_PARK_IDLE: u32 = sized::REACTOR_SPIN_BEFORE_PARK_IDLE;
/// consecutive empty parks WITHOUT INBOX ARRIVALS before we decide
/// cross-core traffic is dead and drop to IDLE mode.
///
/// P12: `prime-runtime.toml`'s `[reactor].idle_park_threshold`
/// (`PRIME_REACTOR_IDLE_PARK_THRESHOLD`).
const IDLE_PARK_THRESHOLD: u32 = sized::REACTOR_IDLE_PARK_THRESHOLD;

thread_local! {
    /// set once at worker startup; tasks query this via `current_core()`.
    static CURRENT_CORE: Cell<Option<CoreId>> = const { Cell::new(None) };
    /// pointer to the worker thread's `LocalExecutor`. used by
    /// `spawn_on_current_core` (called from within a running task). null
    /// outside a worker.
    static CURRENT_EXECUTOR: Cell<*const LocalExecutor> = const { Cell::new(ptr::null()) };
    /// pointer to the worker thread's `RefCell<TimerWheel<StdClock>>`. used
    /// by `timer_at` futures to register themselves on poll.
    static CURRENT_TIMER: Cell<*const RefCell<TimerWheel<StdClock>>> = const {
        Cell::new(ptr::null())
    };
    /// raw pointer to the worker thread's `Reactor`. used by proxima's
    /// TcpListener / TcpStream poll methods to register their wakers. Null
    /// outside a proxima worker. Uses raw `*mut Reactor` (not RefCell) to
    /// avoid the runtime borrow-tracking branch on every WouldBlock; the
    /// reactor is single-thread-owned by construction so no dynamic check
    /// is needed.
    pub(super) static CURRENT_REACTOR: Cell<*mut Reactor> = const {
        Cell::new(ptr::null_mut())
    };
    /// Set by OS I/O futures when a poll registers a reactor waker and
    /// returns `Pending`. The worker consumes this after `executor.tick()`
    /// so I/O-bound tasks can park on the reactor immediately instead of
    /// paying one extra outer-loop lap through the generic busy fast path.
    static REACTOR_PENDING_THIS_TICK: Cell<bool> = const { Cell::new(false) };
}

/// returns the CoreId of the calling thread, or None if not on a worker.
#[must_use]
pub fn current_core() -> Option<CoreId> {
    CURRENT_CORE.with(Cell::get)
}

/// spawn a `?Send` future on the current worker's local executor. panics if
/// called from outside a proxima worker thread.
pub fn spawn_on_current_core(future: Pin<Box<dyn Future<Output = ()> + 'static>>) {
    CURRENT_EXECUTOR.with(|cell| {
        let raw = cell.get();
        assert!(
            !raw.is_null(),
            "spawn_on_current_core: not on a proxima worker thread"
        );
        // SAFETY: the worker only clears CURRENT_EXECUTOR after the loop
        // exits. while a task is running on this thread, the executor is
        // alive on its stack frame.
        unsafe { (*raw).spawn_local_pin(future) };
    });
}

/// future that resolves once the current core's timer reaches `deadline`.
/// must be polled on a proxima worker thread; panics otherwise.
pub fn timer_at(deadline: Tick) -> TimerAtFuture {
    TimerAtFuture {
        deadline,
        key: None,
    }
}

/// current tick (ms since shard launch) from the worker's timer wheel.
/// must be called on a proxima worker thread; panics otherwise. pairs
/// with [`timer_at`] so a caller can schedule a relative delay
/// (`timer_at(current_tick() + delay_ms)`).
#[must_use]
pub fn current_tick() -> Tick {
    CURRENT_TIMER.with(|cell| {
        let raw = cell.get();
        assert!(
            !raw.is_null(),
            "current_tick: not on a proxima worker thread"
        );
        // SAFETY: same as timer_at — the worker keeps the timer alive for
        // its whole lifetime while a task runs on this thread.
        let timer_ref = unsafe { &*raw };
        timer_ref.borrow().now()
    })
}

/// non-panicking [`current_tick`]: `None` when not on a proxima worker
/// thread. lets a mixed-runtime caller — e.g. a tokio-hosted client whose
/// binary links the prime-wheel timer — fall back to a wall clock instead
/// of aborting.
#[must_use]
pub fn current_tick_checked() -> Option<Tick> {
    CURRENT_TIMER.with(|cell| {
        let raw = cell.get();
        if raw.is_null() {
            None
        } else {
            // SAFETY: same as current_tick — the worker keeps the timer alive
            // for its whole lifetime while a task runs on this thread.
            let timer_ref = unsafe { &*raw };
            Some(timer_ref.borrow().now())
        }
    })
}

/// `true` when the calling thread is a proxima worker (its per-core timer
/// wheel is reachable). Lets the link-time timer driver route off-worker
/// callers to a wall-clock fallback instead of panicking.
#[must_use]
pub fn on_worker() -> bool {
    CURRENT_TIMER.with(|cell| !cell.get().is_null())
}

/// Borrow the calling worker's [`Reactor`] to register an externally-owned
/// fd (e.g. an AF_XDP socket) for readiness, returning `None` off a proxima
/// worker thread so the caller can fall back to busy-poll. This is the same
/// per-core reactor the built-in `TcpStream`/`UdpSocket` register with; it
/// exists so an out-of-crate fd source reuses the epoll registration instead
/// of inventing a new source kind. Same worker-affinity contract as those
/// types: poll only on the worker that produced the registration.
pub fn with_current_reactor<Return>(apply: impl FnOnce(&mut Reactor) -> Return) -> Option<Return> {
    let raw = CURRENT_REACTOR.with(Cell::get);
    if raw.is_null() {
        return None;
    }
    // SAFETY: CURRENT_REACTOR is the worker's own Reactor pointer, valid for
    // the worker thread's lifetime and cleared on exit. The worker holds no
    // reactor borrow while polling tasks, so this &mut does not alias; the
    // per-core (no work-stealing) topology keeps polling on the owning thread.
    Some(apply(unsafe { &mut *raw }))
}

#[inline]
pub(crate) fn note_reactor_pending() {
    REACTOR_PENDING_THIS_TICK.with(|cell| cell.set(true));
}

#[inline]
fn take_reactor_pending_hint() -> bool {
    REACTOR_PENDING_THIS_TICK.with(|cell| {
        let pending = cell.get();
        cell.set(false);
        pending
    })
}

/// Register `waker` to fire at `deadline` on the calling worker's timer wheel.
/// The low-level primitive behind the `proxima_core::time` prime-wheel
/// `Driver`: `Driver::schedule_wake` routes here so
/// `proxima_core::time::Sleep` (Send, tiered) runs on prime's per-core wheel
/// instead of a cross-thread std timer.
pub fn schedule_wake(deadline: Tick, waker: core::task::Waker) {
    CURRENT_TIMER.with(|cell| {
        let raw = cell.get();
        assert!(
            !raw.is_null(),
            "schedule_wake: not on a proxima worker thread"
        );
        // SAFETY: same as current_tick — the worker keeps the timer alive while
        // a task runs on this thread.
        let timer_ref = unsafe { &*raw };
        timer_ref.borrow_mut().register(deadline, waker);
    });
}

pub struct TimerAtFuture {
    deadline: Tick,
    key: Option<TimerKey>,
}

impl Future for TimerAtFuture {
    type Output = ();
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        CURRENT_TIMER.with(|cell| {
            let raw = cell.get();
            assert!(!raw.is_null(), "timer_at: not on a proxima worker thread");
            // SAFETY: same as CURRENT_EXECUTOR — worker keeps the timer
            // alive for its whole lifetime.
            let timer_ref = unsafe { &*raw };
            let mut timer = timer_ref.borrow_mut();
            let now = timer.now();
            if now >= this.deadline {
                if let Some(key) = this.key.take() {
                    timer.cancel(key);
                }
                return Poll::Ready(());
            }
            // re-register on each poll — current_tick has moved.
            if let Some(key) = this.key.take() {
                timer.cancel(key);
            }
            let new_key = timer.register(this.deadline, context.waker().clone());
            this.key = Some(new_key);
            Poll::Pending
        })
    }
}

impl Drop for TimerAtFuture {
    fn drop(&mut self) {
        if let Some(key) = self.key.take() {
            CURRENT_TIMER.with(|cell| {
                let raw = cell.get();
                if !raw.is_null() {
                    // SAFETY: same justification as poll.
                    let timer_ref = unsafe { &*raw };
                    timer_ref.borrow_mut().cancel(key);
                }
            });
        }
    }
}

/// drop-guard that clears the thread-local pointers on worker exit.
struct CurrentGuards;
impl Drop for CurrentGuards {
    fn drop(&mut self) {
        CURRENT_CORE.with(|cell| cell.set(None));
        CURRENT_EXECUTOR.with(|cell| cell.set(ptr::null()));
        CURRENT_TIMER.with(|cell| cell.set(ptr::null()));
        CURRENT_REACTOR.with(|cell| cell.set(ptr::null_mut()));
    }
}

/// std-time monotonic clock for the timer wheel. tick = milliseconds since
/// shard launch. lives next to `CoreShard` because it depends on `std`.
struct StdClock {
    epoch: Instant,
}

impl StdClock {
    fn new() -> Self {
        Self {
            epoch: Instant::now(),
        }
    }
}

impl Clock for StdClock {
    fn now(&self) -> Tick {
        self.epoch.elapsed().as_millis() as Tick
    }
}

/// handle to a running per-core worker. holds the sender side of the
/// cross-core inbox, the wakeup handle (to interrupt the worker's
/// `reactor.turn` when an inbox push arrives during idle), and the join
/// handle for the worker thread.
pub struct CoreShardHandle {
    pub(crate) producer: Producer<SpawnRequest<InlineTask>>,
    wakeup: Wakeup,
    /// the core this handle drives. used by teardown to detect a self-join
    /// (Drop running ON its own worker thread) and detach instead of
    /// deadlocking.
    core_id: CoreId,
    /// `Option` so `Drop` can `take` and `.join()` the handle.
    join: Option<JoinHandle<()>>,
}

impl CoreShardHandle {
    /// dispatch a `Send` future to this core for execution. Returns
    /// `SpawnError::InboxFull` if the lane is at capacity (caller should
    /// retry — the future is consumed on Err), or `SpawnError::Disconnected`
    /// if the worker has shut down. fires the wakeup so a parked worker
    /// wakes immediately (vs waiting out the reactor idle).
    pub fn dispatch_send(
        &self,
        future: std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + 'static>>,
    ) -> Result<(), SpawnError> {
        // try_send_mpsc, not try_send: this handle lives behind
        // `Arc<dyn Runtime>` and producers from many threads concurrently
        // call dispatch_send. try_send is SPSC-only — using it across
        // threads would race on a single lane's head/tail (UB).
        // try_send_mpsc assigns each calling thread its own SPSC lane
        // (lazy, via the inbox's monotonic `used_lanes` counter and a
        // thread-local cache). single-thread callers pay ~5-10ns vs
        // try_send's direct store; multi-thread callers gain correctness.
        let request = SpawnRequest::Send(future);
        if let Err(send_err) = self.producer.try_send_mpsc(request) {
            return Err(map_inbox_error(&send_err));
        }
        self.wakeup.fire();
        Ok(())
    }

    /// typed-task fast path. wraps `F` in an `InlineTask` (inline byte
    /// buffer for small `F`, single `Box<F>` for oversized), then ships
    /// the inlined task across the inbox — skipping the per-spawn
    /// `Pin<Box<dyn Future>>` fat-pointer allocation that
    /// [`dispatch_send`] requires. on arrival the worker pushes the
    /// `InlineTask` straight into its slab; polls dispatch through the
    /// per-`F` vtable, no `dyn Future` indirection.
    ///
    /// same back-pressure semantics as [`dispatch_send`]: inbox-full
    /// returns `Err(SpawnError::InboxFull)` with the future consumed.
    pub fn dispatch_send_inline<F>(&self, future: F) -> Result<(), SpawnError>
    where
        F: core::future::Future<Output = ()> + Send + 'static,
    {
        let task = InlineTask::new(future);
        let request = SpawnRequest::SendInline(task);
        if let Err(send_err) = self.producer.try_send_mpsc(request) {
            return Err(map_inbox_error(&send_err));
        }
        self.wakeup.fire();
        Ok(())
    }

    /// dispatch a factory closure to this core. the closure runs on the
    /// target worker and produces the `?Send` future to spawn locally.
    pub fn dispatch_factory(
        &self,
        factory: Box<
            dyn FnOnce() -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
                + Send
                + 'static,
        >,
    ) -> Result<(), SpawnError> {
        let request = SpawnRequest::Factory(factory);
        if let Err(send_err) = self.producer.try_send_mpsc(request) {
            return Err(map_inbox_error(&send_err));
        }
        self.wakeup.fire();
        Ok(())
    }

    /// quiesce this core: stop accepting new inbox pushes from any
    /// thread (subsequent `dispatch_*` returns `SpawnError::Disconnected`
    /// via the inbox-level `SendError::Closed`). Already-in-flight work
    /// continues; the worker drains until empty. Idempotent.
    ///
    /// Pair with [`Self::shutdown_and_join`] for a clean two-phase
    /// shutdown: quiesce → drain in-flight → shutdown. The wakeup fire
    /// is included so a parked worker observes the close immediately.
    pub fn quiesce(&self) {
        self.producer.close();
        self.wakeup.fire();
    }

    /// signal shutdown and join the worker thread. returns `Ok` after the
    /// worker has finished processing in-flight work.
    ///
    /// uses `try_send` (the SPSC fast path on the initial Producer's
    /// lane 0) rather than `try_send_mpsc`: shutdown is single-threaded
    /// by definition (only one shutdown_and_join / Drop fires per
    /// handle), so the SPSC contract holds. Avoids the lane-exhaustion
    /// panic when small-num_lanes test/runtime configurations call
    /// shutdown from a thread that never dispatched anything.
    pub fn shutdown_and_join(mut self) -> Result<(), ProximaError> {
        let _ = self.producer.try_send(SpawnRequest::Shutdown);
        self.wakeup.fire();
        if let Some(handle) = self.join.take() {
            // self-join guard: if a task on THIS worker triggered the
            // shutdown, joining our own thread returns EDEADLK and panics.
            // detach instead — the worker exits on the Shutdown we just sent.
            if current_core() == Some(self.core_id) {
                std::mem::forget(handle);
            } else {
                handle
                    .join()
                    .map_err(|_| ProximaError::Body("core shard worker panicked".into()))?;
            }
        }
        Ok(())
    }
}

impl Drop for CoreShardHandle {
    fn drop(&mut self) {
        // single-threaded by definition: use try_send (lane 0). see
        // `shutdown_and_join` comment for rationale.
        let _ = self.producer.try_send(SpawnRequest::Shutdown);
        self.wakeup.fire();
        if let Some(handle) = self.join.take() {
            // self-join guard: when the last `Arc<PrimeRuntime>` ref is held
            // by a task on this worker (serve_http captures `self.clone()` in
            // every connection handler), Drop runs ON this worker thread.
            // `pthread_join(self)` returns EDEADLK and panics — the
            // "Resource deadlock avoided" crash that took :9091 down. detach
            // instead; the worker exits on the Shutdown we just sent.
            if current_core() == Some(self.core_id) {
                std::mem::forget(handle);
            } else {
                let _ = handle.join();
            }
        }
    }
}

/// launch a worker thread pinned (best-effort) to `affinity`. returns a
/// handle holding the cross-core producer and the join handle.
pub fn launch(
    core_id: CoreId,
    affinity: Option<core_affinity::CoreId>,
) -> Result<CoreShardHandle, ProximaError> {
    launch_with_lanes(
        core_id,
        affinity,
        sized::INBOX_CAPACITY,
        sized::INBOX_CAPACITY,
    )
}

/// optional per-worker setup hook. Runs once on the worker thread before
/// the executor loop starts; the returned value lives on the worker's
/// stack for the worker's lifetime. lets P2 compat mode park a
/// `tokio::runtime::EnterGuard` on the worker (handle re-entry for
/// `tokio::*` API calls) without leaking tokio types into this file.
///
/// the closure itself must be `Send` because it ships from the main
/// thread to the worker thread at launch. the returned token is NOT
/// required to be `Send` — it never leaves the worker thread —
/// which lets it hold `!Send` types like `tokio::runtime::EnterGuard`.
pub type WorkerSetup = Box<dyn FnOnce() -> Box<dyn std::any::Any> + Send>;

/// launch with explicit inbox sizing (num lanes + per-lane capacity).
pub fn launch_with_lanes(
    core_id: CoreId,
    affinity: Option<core_affinity::CoreId>,
    num_lanes: usize,
    lane_capacity: usize,
) -> Result<CoreShardHandle, ProximaError> {
    launch_with_lanes_and_setup(core_id, affinity, num_lanes, lane_capacity, None)
}

/// launch_with_lanes + an optional `WorkerSetup`. The setup closure runs
/// once on the worker thread after thread-local context publication and
/// before the executor loop; its returned token is held on the worker
/// stack for the worker's lifetime.
pub fn launch_with_lanes_and_setup(
    core_id: CoreId,
    affinity: Option<core_affinity::CoreId>,
    num_lanes: usize,
    lane_capacity: usize,
    setup: Option<WorkerSetup>,
) -> Result<CoreShardHandle, ProximaError> {
    #[cfg(not(feature = "runtime-prime-inbox-dynamic"))]
    let (producer, consumer) =
        inbox_impl::channel::<SpawnRequest<InlineTask>>(num_lanes, lane_capacity);
    // dynamic inbox: num_lanes is the eager FLOOR; lanes grow lazily to the
    // ceiling on demand (no task loss), rings allocate per active producer.
    #[cfg(feature = "runtime-prime-inbox-dynamic")]
    let (producer, consumer) = {
        let config = inbox_impl::InboxDynamicConfig {
            floor: num_lanes.max(1),
            ceiling: 1024,
            release: inbox_impl::ReleasePolicy::Always,
            lane_capacity,
        };
        inbox_impl::channel::<SpawnRequest<InlineTask>>(&config)
    };
    // build the reactor outside the worker so the wakeup handle is
    // available to the parent thread before launch returns.
    let reactor = Reactor::new()
        .map_err(|err| ProximaError::Config(format!("init proxima reactor: {err}")))?;
    let wakeup = reactor.wakeup();
    let join = thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .name(format!("proxima-core-{}", core_id.0))
        .spawn(move || worker_main(core_id, affinity, consumer, reactor, setup))
        .map_err(|err| ProximaError::Config(format!("spawn proxima core worker: {err}")))?;
    Ok(CoreShardHandle {
        producer,
        wakeup,
        core_id,
        join: Some(join),
    })
}

/// Convert the inbox's typed `SendError` into the Runtime-trait-level
/// `SpawnError`. The future or factory wrapped in the `SendError` is
/// dropped here — the trait contract documents that the value is consumed
/// on `Err`. Callers that want to retry use [`spawn_on_core_blocking_with`].
fn map_inbox_error<T>(send_err: &inbox_impl::SendError<T>) -> SpawnError {
    match send_err {
        inbox_impl::SendError::Full(_) => SpawnError::InboxFull,
        inbox_impl::SendError::Disconnected(_) => SpawnError::Disconnected,
        // NoLanes is a permanent inbox-sizing error for this caller
        // thread — there is no "wait and retry" recovery. Map to
        // Disconnected so the trait surface signals "this lane is
        // unreachable from this thread, give up" rather than spinning.
        #[cfg(not(feature = "runtime-prime-inbox-dynamic"))]
        inbox_impl::SendError::NoLanes(_) => SpawnError::Disconnected,
        // dynamic inbox: Busy = at lane ceiling, transient. Map to InboxFull
        // (retryable backpressure) — the task is NEVER dropped.
        #[cfg(feature = "runtime-prime-inbox-dynamic")]
        inbox_impl::SendError::Busy(_) => SpawnError::InboxFull,
        // Closed = inbox quiesced via `CoreShardHandle::quiesce()`.
        // Semantically the same as Disconnected at the trait surface
        // (don't retry). Distinct from Full (transient) and
        // Disconnected (target gone); collapsed here because callers
        // (retry helpers) want the same outcome.
        inbox_impl::SendError::Closed(_) => SpawnError::Disconnected,
    }
}

fn worker_main(
    core_id: CoreId,
    affinity: Option<core_affinity::CoreId>,
    consumer: inbox_impl::Consumer<SpawnRequest<InlineTask>>,
    reactor: Reactor,
    setup: Option<WorkerSetup>,
) {
    let _guards = CurrentGuards;

    // setup token lives on the worker's stack for the worker's lifetime.
    // P2 compat mode uses this to hold a `tokio::runtime::EnterGuard`
    // that re-enters the per-core sister tokio runtime for `tokio::*`
    // API calls inside prime tasks. on None, no-op.
    let _setup_token: Option<Box<dyn std::any::Any>> = setup.map(|s| s());

    if let Some(target) = affinity {
        // core_affinity::set_for_current panics on Linux (via libc::CPU_SET)
        // when the CoreId exceeds the platform's CPU count. guard with a
        // validity check so callers can pass bogus ids for testing without
        // aborting (best-effort semantics).
        let valid = core_affinity::get_core_ids()
            .map(|ids| ids.iter().any(|cid| cid.id == target.id))
            .unwrap_or(false);
        if valid {
            let _ = core_affinity::set_for_current(target);
        }
    }
    CURRENT_CORE.with(|cell| cell.set(Some(core_id)));

    // hand a clone of the reactor's wakeup to the LocalExecutor so cross-
    // thread TaskWakers (e.g. results delivered by ProximaBackgroundPool
    // worker threads, oneshot channels signaled from outside the worker)
    // can break us out of `reactor.turn` after pushing into `remote_ready`.
    // without this, a parked worker with no I/O to wait on stays parked
    // forever — the h2_spawn_blocking/prime bench hang.
    let executor_wakeup = reactor.wakeup();
    let executor = LocalExecutor::with_remote_wake(Some(Arc::new(move || executor_wakeup.fire())));
    let clock = StdClock::new();
    let timer: RefCell<TimerWheel<StdClock>> =
        RefCell::new(TimerWheel::new(StdClock { epoch: clock.epoch }));
    // UnsafeCell rather than RefCell: the worker thread is the unique
    // accessor for the lifetime of this scope (CoreShard is single-threaded
    // by construction). Skipping runtime borrow checks is worth ~10-30 ns
    // per WouldBlock on the I/O hot path. callers (TcpListener/TcpStream
    // poll methods, and the worker loop below) coordinate by being purely
    // sequential — no future can re-enter the reactor while another borrow
    // is live because no method on Reactor calls `poll`.
    let reactor: UnsafeCell<Reactor> = UnsafeCell::new(reactor);

    // declare this thread as the executor's worker thread. wakers fired on
    // this thread now route to the local (no-atomics) queue; cross-thread
    // wakers fall back to the SegQueue.
    executor.arm();

    // publish pointers for tasks running on this thread.
    CURRENT_EXECUTOR.with(|cell| cell.set(&executor as *const _));
    CURRENT_TIMER.with(|cell| cell.set(&timer as *const _));
    CURRENT_REACTOR.with(|cell| cell.set(reactor.get()));

    // tracks consecutive empty parks where NO inbox activity was observed.
    // the spin gate's only purpose is catching cross-core inbox pushes
    // before the reactor.turn syscall — so the relevant signal for "spin
    // is helpful" is "did we recently see inbox traffic?" not "did anything
    // happen?". for I/O-bound workloads (h2 traffic, every wake comes via
    // reactor.turn, no inbox traffic) the spin is wasted time.
    let mut empty_parks_since_inbox: u32 = 0;

    loop {
        let mut shutdown = false;
        let mut inbox_arrivals: u32 = 0;
        // 1. drain inbox.
        loop {
            match consumer.try_recv() {
                Ok(SpawnRequest::Send(future)) => {
                    executor.spawn_local_pin(future);
                    inbox_arrivals = inbox_arrivals.saturating_add(1);
                }
                Ok(SpawnRequest::SendInline(task)) => {
                    // eager-poll: tasks that complete on first poll
                    // (counter += 1, fire-and-forget logging, etc.)
                    // never touch the slab. see LocalExecutor doc for
                    // correctness notes.
                    executor.spawn_local_inline_eager(task);
                    inbox_arrivals = inbox_arrivals.saturating_add(1);
                }
                Ok(SpawnRequest::Factory(factory)) => {
                    let future = factory();
                    executor.spawn_local_pin(future);
                    inbox_arrivals = inbox_arrivals.saturating_add(1);
                }
                Ok(SpawnRequest::Shutdown) => {
                    shutdown = true;
                    break;
                }
                Err(inbox_impl::TryRecvError::Empty) => break,
                Err(inbox_impl::TryRecvError::Disconnected) => {
                    shutdown = true;
                    break;
                }
            }
        }
        if shutdown {
            break;
        }

        // 2. poll ready tasks.
        let polled = executor.tick();

        if inbox_arrivals > 0 {
            // producer-driven workload (cross-core spawn). the spin gate
            // will be useful — reset to BUSY mode.
            empty_parks_since_inbox = 0;
        }

        // 3. fast path: executor had work — loop back without touching
        // the clock or timer wheel. timers are millisecond-precision;
        // discovering them on the NEXT idle iteration is fine. saves
        // one Instant::now() + RefCell borrow + advance scan per
        // busy iteration — measurable on I/O-bound workloads (h2 hot
        // path runs ~4-8 worker iterations per request).
        let reactor_pending_after_tick = take_reactor_pending_hint();
        if polled > 0 && !reactor_pending_after_tick {
            continue;
        }
        #[cfg(feature = "runtime-prime-reactor-trace")]
        if reactor_pending_after_tick {
            crate::trace::record_after_tick();
        }

        // 4. executor was idle — drain io_uring completions first (linux +
        // io-uring feature). wakers fired here push tasks to the executor's
        // ready queue; the fast-path check below picks them up on the next
        // tick without going to the park section.
        #[cfg(all(target_os = "linux", feature = "io-uring"))]
        {
            use super::io_uring::reactor::drain_cqes_if_initialized;
            match drain_cqes_if_initialized() {
                Ok(uring_fired) if uring_fired > 0 => continue,
                Ok(_) => {}
                Err(err) => tracing::warn!(error = %err, "io_uring drain failed"),
            }
        }

        // fetch clock and run timer.
        let now = clock.now();
        let fired = timer.borrow_mut().advance(now);
        #[cfg(feature = "runtime-prime-reactor-trace")]
        if reactor_pending_after_tick {
            crate::trace::record_timer_done();
        }

        // 5. if idle, spin-poll briefly before parking. parking is expensive
        // (a kevent/epoll_wait syscall pair, ~1μs+ round-trip); for burst
        // workloads where the producer feeds tasks one-at-a-time, parking
        // between each push burns more time than the work itself. spin a
        // few hundred no-op iterations checking the inbox before committing
        // to a real park.
        if fired == 0 {
            let mut got_work = false;
            #[cfg(feature = "runtime-prime-reactor-harvest-io")]
            let has_live_reactor_sources = {
                // SAFETY: same worker-unique reactor access invariant as
                // the park section below. This read happens outside task
                // polling, before any mutable reactor borrow is created.
                let reactor_ref: &Reactor = unsafe { &*reactor.get() };
                reactor_ref.live_sources() > 0
            };
            #[cfg(not(feature = "runtime-prime-reactor-harvest-io"))]
            let has_live_reactor_sources = false;

            let spin_iters = if has_live_reactor_sources {
                SPIN_BEFORE_PARK_IDLE
            } else if empty_parks_since_inbox < IDLE_PARK_THRESHOLD {
                SPIN_BEFORE_PARK_BUSY
            } else {
                SPIN_BEFORE_PARK_IDLE
            };
            for _ in 0..spin_iters {
                core::hint::spin_loop();
                match consumer.try_recv() {
                    Ok(req) => {
                        match req {
                            SpawnRequest::Send(future) => {
                                executor.spawn_local_pin(future);
                            }
                            SpawnRequest::SendInline(task) => {
                                executor.spawn_local_inline_eager(task);
                            }
                            SpawnRequest::Factory(factory) => {
                                let future = factory();
                                executor.spawn_local_pin(future);
                            }
                            SpawnRequest::Shutdown => {
                                shutdown = true;
                            }
                        }
                        got_work = true;
                        break;
                    }
                    Err(inbox_impl::TryRecvError::Empty) => {}
                    Err(inbox_impl::TryRecvError::Disconnected) => {
                        shutdown = true;
                        got_work = true;
                        break;
                    }
                }
            }
            #[cfg(feature = "runtime-prime-reactor-trace")]
            if reactor_pending_after_tick {
                crate::trace::record_spin_done();
            }
            if shutdown {
                break;
            }
            if got_work {
                // spin caught an inbox push — producer is active. reset.
                empty_parks_since_inbox = 0;
                continue;
            }

            // still nothing — park on the reactor. wakeup.fire from a
            // producer will interrupt us.
            empty_parks_since_inbox = empty_parks_since_inbox.saturating_add(1);
            let next_deadline = timer.borrow().next_deadline();
            let timeout = match next_deadline {
                Some(deadline) => {
                    let delta = deadline.saturating_sub(now);
                    Some(Duration::from_millis(delta))
                }
                None => None,
            };
            {
                // SAFETY: the worker thread is the unique accessor for
                // `reactor` (UnsafeCell). No outstanding borrow can exist
                // here because: (a) every TcpStream / TcpListener method
                // that touches the reactor does so via `with_reactor_mut`
                // which scopes the borrow to a single non-async function
                // call; (b) the executor's `tick()` returned before we got
                // here. The only re-entrancy risk would be a waker callback
                // calling into the reactor — wakers only push into ready
                // queues, never the reactor.
                let reactor_mut: &mut Reactor = unsafe { &mut *reactor.get() };
                reactor_mut.arm_wakeup();
                // Dekker-pattern fence (matches the one in
                // `Wakeup::fire`). arm_wakeup wrote `needs_wake` with
                // Release; the recheck below reads the inbox `tail`
                // with Acquire (via consumer.try_recv). Release/Acquire
                // on different atomics doesn't establish cross-variable
                // happens-before — without this SeqCst fence pair,
                // worker and producer can both observe stale loads:
                // worker sees empty inbox AND producer sees needs_wake
                // = false. Worker parks; task wedged. With the fence
                // pair (worker + Wakeup::fire) participating in the
                // SeqCst total order, whichever sequenced-first, the
                // other side observes its preceding store: producer
                // fires (worker arm visible) OR worker drains
                // (producer push visible).
                atomic::fence(Ordering::SeqCst);
                #[cfg(feature = "runtime-prime-reactor-trace")]
                if reactor_pending_after_tick {
                    crate::trace::record_arm_wakeup();
                }
                // race close: between the tick() at step 2 returning
                // polled=0 and `arm_wakeup` here, a producer may have
                // (a) pushed into the cross-core inbox AND called
                // `wakeup.fire()`, or (b) the cross-thread TaskWaker
                // pushed into `remote_ready` AND called the same
                // `wakeup.fire()`. In either case the fire was a
                // no-op because `needs_wake` was false. Without a
                // recheck, the work is lost and we park forever on
                // `turn`. After `arm_wakeup` every subsequent fire
                // WILL queue a kevent; the only remaining gap is the
                // work already landed before arm_wakeup ran. Drain
                // both surfaces — INBOX and executor ready queues
                // — before committing to a park.
                //
                // INBOX recheck is load-bearing: missing it deadlocked
                // W4 under contention (a single task never picked up
                // because its `try_send_mpsc` + `fire()` raced with
                // this worker's spin-loop end → arm_wakeup). Repro
                // pre-fix was `examples/w4_mutex_repro` — 100 iters,
                // hangs by ~iter 50 with `started per core = [1, 0,
                // 1, 1]` (one task wedged in the inbox).
                let mut inbox_drained: usize = 0;
                let mut inbox_shutdown = false;
                loop {
                    match consumer.try_recv() {
                        Ok(SpawnRequest::Send(future)) => {
                            executor.spawn_local_pin(future);
                            inbox_drained = inbox_drained.saturating_add(1);
                        }
                        Ok(SpawnRequest::SendInline(task)) => {
                            executor.spawn_local_inline_eager(task);
                            inbox_drained = inbox_drained.saturating_add(1);
                        }
                        Ok(SpawnRequest::Factory(factory)) => {
                            let future = factory();
                            executor.spawn_local_pin(future);
                            inbox_drained = inbox_drained.saturating_add(1);
                        }
                        Ok(SpawnRequest::Shutdown) => {
                            inbox_shutdown = true;
                            break;
                        }
                        Err(inbox_impl::TryRecvError::Empty) => break,
                        Err(inbox_impl::TryRecvError::Disconnected) => {
                            inbox_shutdown = true;
                            break;
                        }
                    }
                }
                if inbox_shutdown {
                    reactor_mut.disarm_wakeup();
                    break;
                }
                let recheck_polled = executor.tick();
                let recheck_fired = timer.borrow_mut().advance(clock.now());
                #[cfg(feature = "runtime-prime-reactor-trace")]
                if reactor_pending_after_tick {
                    crate::trace::record_recheck_done();
                }
                if inbox_drained + recheck_polled + recheck_fired > 0 {
                    #[cfg(feature = "runtime-prime-reactor-trace")]
                    if reactor_pending_after_tick {
                        crate::trace::record_recheck_continue(
                            inbox_drained,
                            recheck_polled,
                            recheck_fired,
                        );
                    }
                    reactor_mut.disarm_wakeup();
                    if inbox_drained > 0 {
                        empty_parks_since_inbox = 0;
                    }
                    continue;
                }
                #[cfg(feature = "runtime-prime-reactor-trace")]
                crate::trace::record_turn_enter();
                let turn_result = reactor_mut.turn(timeout);
                #[cfg(feature = "runtime-prime-reactor-trace")]
                crate::trace::record_turn_exit(turn_result.as_ref().map_or(0, |fired| *fired));
                if let Err(err) = turn_result
                    && err.kind() != std::io::ErrorKind::Interrupted
                {
                    tracing::warn!(error = %err, "proxima reactor turn failed");
                }
                reactor_mut.disarm_wakeup();
                // drain io_uring CQEs immediately after the epoll park so that
                // completions triggered by the ring-fd EPOLLIN wake are
                // processed before the next executor tick.
                #[cfg(all(target_os = "linux", feature = "io-uring"))]
                if let Err(err) = super::io_uring::reactor::drain_cqes_if_initialized() {
                    tracing::warn!(error = %err, "io_uring post-park drain failed");
                }
            }
        }
    }

    // tear the executor down while the reactor is still alive: each task's
    // source deregisters on drop, which clears the waker the reactor holds for
    // it. locals drop in reverse declaration order, so without this the reactor
    // (UnsafeCell, declared after executor) frees first and the tasks dropping
    // last deregister into freed reactor memory — a use-after-free.
    drop(executor);
}

/// Inverted-compat launch (design D2): like [`launch_with_lanes`], but the
/// worker thread runs [`worker_main_inverted`] — it OWNS a sister tokio
/// current-thread runtime on its own thread and ticks the prime executor
/// inside `sister.block_on(...)`, so raw `tokio::spawn` from a prime task
/// takes tokio's LOCAL fast path (no per-spawn driver.unpark/kevent).
///
/// gated on `prime-tokio-compat-inverted`. minimal park; full Dekker-park
/// fidelity tracked in discipline-inverted-compat.md.
#[cfg(feature = "prime-tokio-compat-inverted")]
pub fn launch_inverted(
    core_id: CoreId,
    affinity: Option<core_affinity::CoreId>,
) -> Result<CoreShardHandle, ProximaError> {
    launch_inverted_with_lanes(
        core_id,
        affinity,
        sized::INBOX_CAPACITY,
        sized::INBOX_CAPACITY,
    )
}

/// [`launch_inverted`] with explicit inbox sizing.
#[cfg(feature = "prime-tokio-compat-inverted")]
pub fn launch_inverted_with_lanes(
    core_id: CoreId,
    affinity: Option<core_affinity::CoreId>,
    num_lanes: usize,
    lane_capacity: usize,
) -> Result<CoreShardHandle, ProximaError> {
    #[cfg(not(feature = "runtime-prime-inbox-dynamic"))]
    let (producer, consumer) =
        inbox_impl::channel::<SpawnRequest<InlineTask>>(num_lanes, lane_capacity);
    #[cfg(feature = "runtime-prime-inbox-dynamic")]
    let (producer, consumer) = {
        let config = inbox_impl::InboxDynamicConfig {
            floor: num_lanes.max(1),
            ceiling: 1024,
            release: inbox_impl::ReleasePolicy::Always,
            lane_capacity,
        };
        inbox_impl::channel::<SpawnRequest<InlineTask>>(&config)
    };
    let reactor = Reactor::new()
        .map_err(|err| ProximaError::Config(format!("init proxima reactor: {err}")))?;
    let wakeup = reactor.wakeup();
    let join = thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .name(format!("proxima-core-inverted-{}", core_id.0))
        .spawn(move || worker_main_inverted(core_id, affinity, consumer, reactor))
        .map_err(|err| {
            ProximaError::Config(format!("spawn proxima inverted core worker: {err}"))
        })?;
    Ok(CoreShardHandle {
        producer,
        wakeup,
        core_id,
        join: Some(join),
    })
}

/// Inverted-compat worker (design D2). The worker thread OWNS a sister tokio
/// current-thread runtime (NOT a separate driver thread) and holds its
/// `EnterGuard` on the stack for its whole life, so `tokio::spawn` from a
/// prime task running here resolves to the sister. Each loop iteration ticks
/// the prime executor INSIDE `sister.block_on(...)` — that is the property
/// proven to make the prime-task `tokio::spawn` LOCAL (~95ns vs ~379ns
/// remote). A bounded `yield_now` per iteration lets sister-spawned tasks make
/// progress.
///
/// This is the minimal D2 proof: the park below is a short bounded
/// `reactor.turn` with the wakeup armed/disarmed each cycle and an inbox
/// recheck — it does NOT replicate `worker_main`'s full Dekker-fence wakeup
/// machinery. minimal park; full Dekker-park fidelity tracked in
/// discipline-inverted-compat.md.
#[cfg(feature = "prime-tokio-compat-inverted")]
fn worker_main_inverted(
    core_id: CoreId,
    affinity: Option<core_affinity::CoreId>,
    consumer: inbox_impl::Consumer<SpawnRequest<InlineTask>>,
    reactor: Reactor,
) {
    let _guards = CurrentGuards;

    if let Some(target) = affinity {
        let valid = core_affinity::get_core_ids()
            .map(|ids| ids.iter().any(|cid| cid.id == target.id))
            .unwrap_or(false);
        if valid {
            let _ = core_affinity::set_for_current(target);
        }
    }
    CURRENT_CORE.with(|cell| cell.set(Some(core_id)));

    // the sister runtime is OWNED by this worker thread and never moves off
    // it. building it here (not on a separate driver thread) is the whole
    // point of D2: `block_on` runs ON this thread, so `tokio::spawn` issued
    // by code inside it is a LOCAL schedule.
    let sister = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .thread_name(format!("prime-inverted-sister-{}", core_id.0))
        .build()
    {
        Ok(value) => value,
        Err(err) => {
            tracing::error!(error = %err, core = core_id.0, "inverted sister runtime build failed");
            return;
        }
    };
    // EnterGuard is !Send — held on the worker stack, never moved across
    // threads. with it live, `tokio::runtime::Handle::current()` (and thus
    // `tokio::spawn`) from any prime task polled on this thread resolves to
    // the sister.
    let _enter = sister.handle().enter();

    // C2b (linux): register prime's epoll fd into the sister's reactor so the
    // idle park waits on ONE thing — the sister mio — yet still wakes on prime
    // readiness (prime I/O OR an inbox wakeup, both surfacing as prime's epoll
    // fd becoming readable). built inside the sister context (EnterGuard live)
    // so it registers with the sister's IO driver. the fd stays owned by
    // `reactor`; `InvertedWakeFd` does not close it. on non-linux the worker
    // falls back to the bounded sister/prime park below (C2a).
    #[cfg(target_os = "linux")]
    let wake_async = {
        let prime_poll_fd = reactor.raw_poll_fd();
        match tokio::io::unix::AsyncFd::with_interest(
            InvertedWakeFd(prime_poll_fd),
            tokio::io::Interest::READABLE,
        ) {
            Ok(value) => value,
            Err(err) => {
                tracing::error!(error = %err, core = core_id.0, "inverted wake AsyncFd registration failed");
                return;
            }
        }
    };

    let executor_wakeup = reactor.wakeup();
    let executor = LocalExecutor::with_remote_wake(Some(Arc::new(move || executor_wakeup.fire())));
    let clock = StdClock::new();
    let timer: RefCell<TimerWheel<StdClock>> =
        RefCell::new(TimerWheel::new(StdClock { epoch: clock.epoch }));
    let reactor: UnsafeCell<Reactor> = UnsafeCell::new(reactor);

    executor.arm();

    CURRENT_EXECUTOR.with(|cell| cell.set(&executor as *const _));
    CURRENT_TIMER.with(|cell| cell.set(&timer as *const _));
    CURRENT_REACTOR.with(|cell| cell.set(reactor.get()));

    // sister spawned-task count from the previous idle check. used to tell a
    // draining tokio::spawn burst (count changing → keep driving) apart from
    // tasks parked on I/O/timers (count stable → fall to the bounded park).
    let mut prev_sister_alive: usize = 0;

    loop {
        let mut shutdown = false;
        // 1. drain inbox onto the local executor.
        loop {
            match consumer.try_recv() {
                Ok(SpawnRequest::Send(future)) => {
                    executor.spawn_local_pin(future);
                }
                Ok(SpawnRequest::SendInline(task)) => {
                    executor.spawn_local_inline_eager(task);
                }
                Ok(SpawnRequest::Factory(factory)) => {
                    let future = factory();
                    executor.spawn_local_pin(future);
                }
                Ok(SpawnRequest::Shutdown) => {
                    shutdown = true;
                    break;
                }
                Err(inbox_impl::TryRecvError::Empty) => break,
                Err(inbox_impl::TryRecvError::Disconnected) => {
                    shutdown = true;
                    break;
                }
            }
        }
        if shutdown {
            break;
        }

        // 2. tick the prime executor INSIDE sister.block_on. block_on does NOT
        // require a 'static future, so the async block can borrow `&executor`
        // off this stack frame. polling prime tasks here means any
        // `tokio::spawn` they issue lands on the sister LOCALLY (the D2
        // property). the bounded `yield_now` loop then lets those sister-
        // spawned tasks make progress without an unbounded drain.
        let polled = sister.block_on(async {
            let count = executor.tick();
            for _ in 0..SISTER_DRIVE_YIELDS {
                tokio::task::yield_now().await;
            }
            count
        });

        if polled > 0 {
            prev_sister_alive = sister.metrics().num_alive_tasks();
            continue;
        }

        // keep driving the sister while its spawned-task count is still
        // changing — that is a tokio::spawn burst actively draining, and a
        // park here would stall the ready tasks for the park timeout. when
        // the count is stable and non-zero the sister is waiting on I/O or
        // timers, so fall through to the bounded park (it is redriven within
        // INVERTED_PARK_TIMEOUT). count==0 means idle: park for real.
        let sister_alive = sister.metrics().num_alive_tasks();
        let draining = sister_alive != 0 && sister_alive != prev_sister_alive;
        prev_sister_alive = sister_alive;
        if draining {
            continue;
        }

        // count is stable and non-zero: the sister has tasks parked on I/O or
        // timers, not draining. on non-linux (C2a) drive the sister's OWN
        // reactor for a bounded slice (block_on parks on mio once its run queue
        // empties), so socket/timer readiness is serviced — a task that becomes
        // ready mid-slice runs at once; prime wakeups stall up to the bound
        // (cross-reactor). on linux the unified park below (C2b) handles this
        // case without the bounded poll.
        #[cfg(not(target_os = "linux"))]
        if sister_alive > 0 {
            sister.block_on(async {
                tokio::time::sleep(INVERTED_PARK_TIMEOUT).await;
            });
            continue;
        }

        // 3. advance the timer wheel.
        let now = clock.now();
        let fired = timer.borrow_mut().advance(now);
        if fired > 0 {
            continue;
        }

        // 4. minimal park: arm the wakeup, recheck the inbox (a producer may
        // have pushed + fired between the tick above and here), then a short
        // bounded reactor.turn. a producer's wakeup.fire interrupts it; the
        // bounded timeout caps how long a missed-wake costs. minimal park;
        // full Dekker-park fidelity tracked in discipline-inverted-compat.md.
        // SAFETY: the worker thread is the unique accessor for `reactor`
        // (UnsafeCell); no reactor borrow is outstanding here (tick returned,
        // no waker re-enters the reactor).
        let reactor_mut: &mut Reactor = unsafe { &mut *reactor.get() };
        reactor_mut.arm_wakeup();
        atomic::fence(Ordering::SeqCst);
        let mut rearmed = false;
        loop {
            match consumer.try_recv() {
                Ok(SpawnRequest::Send(future)) => {
                    executor.spawn_local_pin(future);
                    rearmed = true;
                }
                Ok(SpawnRequest::SendInline(task)) => {
                    executor.spawn_local_inline_eager(task);
                    rearmed = true;
                }
                Ok(SpawnRequest::Factory(factory)) => {
                    let future = factory();
                    executor.spawn_local_pin(future);
                    rearmed = true;
                }
                Ok(SpawnRequest::Shutdown) => {
                    shutdown = true;
                    break;
                }
                Err(inbox_impl::TryRecvError::Empty) => break,
                Err(inbox_impl::TryRecvError::Disconnected) => {
                    shutdown = true;
                    break;
                }
            }
        }
        if shutdown {
            reactor_mut.disarm_wakeup();
            break;
        }
        if rearmed {
            reactor_mut.disarm_wakeup();
            continue;
        }
        // C2a (non-linux): bounded park on prime's reactor; a producer's
        // wakeup.fire interrupts it, the timeout caps a missed-wake stall.
        #[cfg(not(target_os = "linux"))]
        if let Err(err) = reactor_mut.turn(Some(INVERTED_PARK_TIMEOUT))
            && err.kind() != std::io::ErrorKind::Interrupted
        {
            tracing::warn!(error = %err, "proxima inverted reactor turn failed");
        }
        // C2b (linux): unified park. wait on the sister's reactor (which
        // services tokio I/O + timers while parked) until prime's epoll fd is
        // readable — prime I/O OR an inbox wakeup eventfd write — then drain
        // prime's ready sources non-blocking. the fd bridge means prime wakeups
        // are instant (no bounded poll for them); the INVERTED_PARK_TIMEOUT
        // pacemaker only bounds sister-timer latency (see below).
        #[cfg(target_os = "linux")]
        {
            // bounded by INVERTED_PARK_TIMEOUT as a pacemaker: the readable()
            // wakes instantly on prime epoll-fd readiness (prime work), and the
            // timeout guarantees the sister time driver still turns at least
            // every INVERTED_PARK_TIMEOUT so a sister timer registered by
            // prime-task code fires (host-b caught this — an unbounded park
            // hung tokio::time::sleep awaited in a prime task).
            sister.block_on(async {
                if let Ok(Ok(mut guard)) =
                    tokio::time::timeout(INVERTED_PARK_TIMEOUT, wake_async.readable()).await
                {
                    guard.clear_ready();
                }
            });
            if let Err(err) = reactor_mut.turn(Some(std::time::Duration::ZERO))
                && err.kind() != std::io::ErrorKind::Interrupted
            {
                tracing::warn!(error = %err, "proxima inverted reactor drain failed");
            }
        }
        reactor_mut.disarm_wakeup();
    }

    drop(executor);
}

/// per-iteration bounded yields that let sister-spawned tokio tasks make
/// progress inside the worker's `block_on`. small fixed count — enough for a
/// burst of `tokio::spawn`ed leaf tasks to complete without an unbounded
/// drain that could starve the prime executor.
#[cfg(feature = "prime-tokio-compat-inverted")]
const SISTER_DRIVE_YIELDS: u32 = sized::COMPAT_SISTER_DRIVE_YIELDS;

/// park-timeout for the inverted worker. on non-linux (C2a) it bounds the
/// bounded poll. on linux (C2b) it is the pacemaker on the unified park: the
/// prime epoll-fd bridge wakes instantly on prime work, but a sister timer
/// registered by prime-task code does not set the sister mio park deadline, so
/// without a periodic turn the timer never fires. capping the park at this
/// timeout guarantees the sister time driver advances at least this often.
#[cfg(feature = "prime-tokio-compat-inverted")]
const INVERTED_PARK_TIMEOUT: Duration = Duration::from_millis(sized::COMPAT_PARK_TIMEOUT_MS);

/// non-owning wrapper so `tokio::io::unix::AsyncFd` can register prime's epoll
/// fd into the sister reactor (C2b unified park) without taking ownership — the
/// fd stays owned by prime's `Reactor`, so this must NOT close it on drop.
#[cfg(all(feature = "prime-tokio-compat-inverted", target_os = "linux"))]
struct InvertedWakeFd(std::os::fd::RawFd);

#[cfg(all(feature = "prime-tokio-compat-inverted", target_os = "linux"))]
impl std::os::fd::AsRawFd for InvertedWakeFd {
    fn as_raw_fd(&self) -> std::os::fd::RawFd {
        self.0
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn launch_then_send_runs_the_future_on_the_worker() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_future = counter.clone();
        let handle = launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        handle
            .dispatch_send(Box::pin(async move {
                counter_for_future.fetch_add(1, Ordering::AcqRel);
            }))
            .expect("send");
        // poll until the worker observes our send. up to 1s.
        let deadline = Instant::now() + Duration::from_secs(1);
        while counter.load(Ordering::Acquire) == 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(counter.load(Ordering::Acquire), 1);
        handle.shutdown_and_join().expect("shutdown");
    }

    #[test]
    fn factory_spawn_runs_local_future() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_factory = counter.clone();
        let handle = launch_with_lanes(CoreId(1), None, 2, 16).expect("launch");
        handle
            .dispatch_factory(Box::new(move || {
                let counter = counter_for_factory.clone();
                Box::pin(async move {
                    counter.fetch_add(1, Ordering::AcqRel);
                })
            }))
            .expect("send");
        let deadline = Instant::now() + Duration::from_secs(1);
        while counter.load(Ordering::Acquire) == 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(counter.load(Ordering::Acquire), 1);
        handle.shutdown_and_join().expect("shutdown");
    }

    #[test]
    fn shutdown_joins_cleanly() {
        let handle = launch_with_lanes(CoreId(2), None, 2, 16).expect("launch");
        handle.shutdown_and_join().expect("shutdown");
    }

    #[test]
    fn current_core_inside_future_returns_dispatched_id() {
        let observed: Arc<std::sync::Mutex<Option<CoreId>>> = Arc::new(std::sync::Mutex::new(None));
        let observed_for_future = observed.clone();
        let handle = launch_with_lanes(CoreId(5), None, 2, 16).expect("launch");
        handle
            .dispatch_send(Box::pin(async move {
                let id = current_core();
                *observed_for_future.lock().unwrap() = id;
            }))
            .expect("send");
        let deadline = Instant::now() + Duration::from_secs(1);
        while observed.lock().unwrap().is_none() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(*observed.lock().unwrap(), Some(CoreId(5)));
        handle.shutdown_and_join().expect("shutdown");
    }

    #[test]
    fn affinity_failure_does_not_block_launch() {
        // pass an out-of-range affinity; launch should still succeed (best-effort).
        let bogus = core_affinity::CoreId { id: usize::MAX };
        let handle = launch_with_lanes(CoreId(0), Some(bogus), 2, 16).expect("launch");
        handle.shutdown_and_join().expect("shutdown");
    }

    /// Bug A reproduction (generic cross-thread wake): a task awaits a
    /// future whose waker is fired from a thread other than the worker
    /// thread. Pre-fix, the cross-thread `TaskWaker::do_wake` pushes
    /// into `remote_ready` but does NOT signal the worker's reactor, so
    /// a worker parked on `reactor.turn(None)` (no I/O, no timer
    /// deadline) stays parked indefinitely.
    ///
    /// Repro shape: spawn one task that waits on an external flag.
    /// The task registers its waker on first poll, then returns Pending.
    /// The worker drains, polls once, and ends up parked on the reactor.
    /// A separate thread sets the flag and fires the stored waker.
    /// Without the fix the worker remains parked forever; we time out
    /// at 1s. With the fix the cross-thread wake fires a reactor
    /// `Wakeup`, the worker leaves `turn`, drains `remote_ready`, and
    /// completes the task in milliseconds.
    #[test]
    fn cross_thread_wake_unparks_worker_blocked_on_reactor() {
        use std::sync::Mutex;
        use std::sync::atomic::AtomicBool;
        use std::task::Waker;

        struct FlagFuture {
            flag: Arc<AtomicBool>,
            waker_slot: Arc<Mutex<Option<Waker>>>,
        }
        impl Future for FlagFuture {
            type Output = ();
            fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
                if self.flag.load(Ordering::Acquire) {
                    return Poll::Ready(());
                }
                *self.waker_slot.lock().expect("waker slot mutex") = Some(context.waker().clone());
                Poll::Pending
            }
        }

        let handle = launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        let done = Arc::new(AtomicBool::new(false));
        let flag = Arc::new(AtomicBool::new(false));
        let waker_slot: Arc<Mutex<Option<Waker>>> = Arc::new(Mutex::new(None));

        let done_for_task = done.clone();
        let flag_for_task = flag.clone();
        let waker_for_task = waker_slot.clone();
        handle
            .dispatch_send(Box::pin(async move {
                FlagFuture {
                    flag: flag_for_task,
                    waker_slot: waker_for_task,
                }
                .await;
                done_for_task.store(true, Ordering::Release);
            }))
            .expect("dispatch task");

        // wait for the worker to register the task's waker and drop into
        // a steady parked state. 200ms is comfortably past
        // IDLE_PARK_THRESHOLD * spin cost.
        std::thread::sleep(Duration::from_millis(200));
        assert!(
            !done.load(Ordering::Acquire),
            "task should still be pending — flag not yet set",
        );

        // separate thread: set the flag and fire the stored waker.
        // This routes through TaskWaker::do_wake's CROSS-THREAD branch
        // because we're not on the worker thread.
        let flag_for_waker = flag.clone();
        let waker_for_waker = waker_slot.clone();
        let waker_thread = std::thread::spawn(move || {
            flag_for_waker.store(true, Ordering::Release);
            let waker = waker_for_waker
                .lock()
                .expect("waker mutex")
                .take()
                .expect("waker captured by first poll");
            waker.wake();
        });

        let deadline = Instant::now() + Duration::from_secs(1);
        while !done.load(Ordering::Acquire) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        waker_thread.join().expect("waker thread join");
        assert!(
            done.load(Ordering::Acquire),
            "cross-thread wake must unpark a worker blocked on reactor.turn — \
             missing reactor signal in TaskWaker cross-thread branch",
        );
        handle.shutdown_and_join().expect("shutdown");
    }

    /// Bug A residual probe — narrower than the generic 2-wake test:
    /// a prime task that calls `runtime.spawn_background_blocking(...)`
    /// twice in succession on the same prime worker. Mirrors the
    /// bench's BgPool BlockingHashPipe pattern. If this hangs on the
    /// second call, the bug is in BgPool wake-chain across sequential
    /// awaits.
    #[cfg(feature = "runtime-prime-bgpool")]
    #[test]
    fn prime_task_does_two_sequential_spawn_background_blocking_calls() {
        use crate::os::runtime::PrimeRuntime;
        use proxima_runtime::{BackgroundPool, Runtime};
        use std::any::Any;
        use std::sync::Arc;
        use std::sync::atomic::AtomicUsize as StdAtomic;
        use std::sync::atomic::Ordering as StdOrdering;

        let pool: Arc<dyn BackgroundPool> =
            Arc::new(crate::os::background::ProximaBackgroundPool::new().expect("bg pool"));
        let runtime: Arc<dyn Runtime> = Arc::new(
            PrimeRuntime::new(1)
                .expect("prime runtime")
                .with_background_pool(pool),
        );
        let done = Arc::new(StdAtomic::new(0_usize));
        let done_for_task = done.clone();
        let runtime_for_task = runtime.clone();
        runtime
            .spawn_factory_on_core(
                CoreId(0),
                Box::new(move || {
                    let runtime = runtime_for_task;
                    Box::pin(async move {
                        eprintln!("[two-bg-test] task started");
                        for index in 0..2 {
                            eprintln!("[two-bg-test] iter {index} pre-spawn");
                            let work: Box<
                                dyn FnOnce() -> Result<
                                        Box<dyn Any + Send>,
                                        proxima_core::ProximaError,
                                    > + Send,
                            > = Box::new(move || {
                                eprintln!("[two-bg-test] iter {index} work executing on bg thread");
                                Ok(Box::new(index as u32) as Box<dyn Any + Send>)
                            });
                            eprintln!("[two-bg-test] iter {index} awaiting bg handle");
                            let value_any = runtime
                                .spawn_background_blocking(work)
                                .await
                                .expect("bg result");
                            eprintln!("[two-bg-test] iter {index} got result");
                            let value = value_any.downcast::<u32>().expect("downcast");
                            done_for_task.fetch_add(*value as usize + 1, StdOrdering::AcqRel);
                        }
                        eprintln!("[two-bg-test] task done");
                    }) as Pin<Box<dyn Future<Output = ()> + 'static>>
                }),
            )
            .expect("spawn factory");

        let deadline = Instant::now() + Duration::from_secs(5);
        while done.load(StdOrdering::Acquire) < 3 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        let final_value = done.load(StdOrdering::Acquire);
        assert_eq!(
            final_value, 3,
            "expected both bg calls to complete (final 3 = 1+2), got {final_value} — \
             second spawn_background_blocking hung",
        );
    }

    /// Bug A residual probe: two SEQUENTIAL cross-thread wakes from the
    /// same external thread. Tests whether `Wakeup`'s `needs_wake`
    /// arm/disarm cycle or `TaskWaker::do_wake`'s reactor signal has a
    /// re-arm hole across cycles. Bench-pattern repro confirms the
    /// SECOND BgPool await on the same prime worker hangs; if this
    /// unit test also hangs on its second wake, the bug is in the prime
    /// wake mechanism itself and a fix lives in this file.
    #[test]
    fn two_sequential_cross_thread_wakes_both_unpark_the_worker() {
        use futures::channel::oneshot;

        let handle = launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        let done = Arc::new(AtomicUsize::new(0));
        let done_for_task = done.clone();

        let (sender1, receiver1) = oneshot::channel::<u32>();
        let (sender2, receiver2) = oneshot::channel::<u32>();
        handle
            .dispatch_send(Box::pin(async move {
                let value1 = receiver1.await.expect("oneshot1");
                done_for_task.fetch_add(value1 as usize, Ordering::AcqRel);
                let value2 = receiver2.await.expect("oneshot2");
                done_for_task.fetch_add(value2 as usize, Ordering::AcqRel);
            }))
            .expect("dispatch");

        // Let the worker poll once, hit Pending on receiver1, park.
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(done.load(Ordering::Acquire), 0);

        // First wake from a separate thread.
        let sender1_thread = std::thread::spawn(move || {
            sender1.send(10).expect("send1");
        });
        let deadline_first = Instant::now() + Duration::from_secs(1);
        while done.load(Ordering::Acquire) < 10 && Instant::now() < deadline_first {
            std::thread::sleep(Duration::from_millis(5));
        }
        sender1_thread.join().expect("sender1 thread");
        assert_eq!(
            done.load(Ordering::Acquire),
            10,
            "first cross-thread wake must unpark worker",
        );

        // Let the worker park on receiver2.
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(done.load(Ordering::Acquire), 10);

        // SECOND wake. Critical case: same task, same exec_id, same
        // remote_ready, same Wakeup. If the wake mechanism has an
        // arm/disarm re-arm hole, this hangs.
        let sender2_thread = std::thread::spawn(move || {
            sender2.send(20).expect("send2");
        });
        let deadline_second = Instant::now() + Duration::from_secs(1);
        while done.load(Ordering::Acquire) < 30 && Instant::now() < deadline_second {
            std::thread::sleep(Duration::from_millis(5));
        }
        sender2_thread.join().expect("sender2 thread");
        assert_eq!(
            done.load(Ordering::Acquire),
            30,
            "SECOND cross-thread wake must also unpark worker — \
             the bench's bg-pool sequential-request hang exactly matches \
             this if the second wake fails",
        );
        handle.shutdown_and_join().expect("shutdown");
    }

    /// Bug A reproduction (BgPool / oneshot path): mirrors the path used
    /// by `ProximaBackgroundPool::spawn` — a `futures::channel::oneshot`
    /// awaited inside a task spawned on the prime worker. The bg-pool
    /// thread sends on the oneshot, which fires the receiver-side waker
    /// from a thread other than the worker. Same root cause as the
    /// generic test above; this test pins the exact pattern that the
    /// `bench_h2_spawn_blocking/prime` arm exercises.
    #[test]
    fn cross_thread_oneshot_wake_unparks_worker_blocked_on_reactor() {
        use futures::channel::oneshot;

        let handle = launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        let done = Arc::new(AtomicUsize::new(0));
        let done_for_task = done.clone();

        let (sender, receiver) = oneshot::channel::<u32>();
        handle
            .dispatch_send(Box::pin(async move {
                let value = receiver.await.expect("oneshot recv");
                done_for_task.fetch_add(value as usize, Ordering::AcqRel);
            }))
            .expect("dispatch task");

        // give the worker time to poll once, hit Pending, and park on
        // reactor.turn.
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(done.load(Ordering::Acquire), 0);

        // separate thread does the cross-thread send — exactly mirrors
        // what ProximaBackgroundPool's worker does on job completion.
        let sender_thread = std::thread::spawn(move || {
            sender.send(42).expect("oneshot send");
        });

        let deadline = Instant::now() + Duration::from_secs(1);
        while done.load(Ordering::Acquire) == 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        sender_thread.join().expect("sender thread join");
        assert_eq!(
            done.load(Ordering::Acquire),
            42,
            "oneshot cross-thread wake must unpark worker blocked on reactor.turn",
        );
        handle.shutdown_and_join().expect("shutdown");
    }

    /// When the per-core inbox lane fills, `dispatch_send` returns
    /// `Err`. The bench at `benches/bench_spawn_burst.rs` exposed the
    /// bug: with the default lane capacity of 1024 and a 10k-task burst,
    /// the producer overflows. With a small lane (cap=8) we reproduce
    /// the failure in milliseconds: spawn far more than the lane can
    /// hold, deliberately NOT giving the worker time to drain, then
    /// observe that at least one send returns Err.
    ///
    /// The contract this test pins: **`dispatch_send` MUST surface
    /// overflow to the caller**. Higher layers (PrimeRuntime's
    /// `Runtime` trait impl) currently swallow this error — that's a
    /// separate bug, documented in the sibling test below.
    #[test]
    fn inbox_overflow_surfaces_at_dispatch_send() {
        // Launch a shard with a tiny lane (8 slots). To FORCE overflow
        // we hold the worker from draining by sleeping the spawned
        // tasks longer than our send loop takes. Each task parks for
        // 500ms on a barrier we never release; producer streams sends
        // faster than the worker can dequeue (it dequeues exactly 0
        // because the worker is busy polling the stuck task body).
        let handle = launch_with_lanes(CoreId(0), None, 2, 8).expect("launch");
        let stuck_at = Arc::new(std::sync::Barrier::new(2));
        let stuck_at_for_first = stuck_at.clone();
        // First task: blocks on the barrier on the worker thread.
        // Subsequent dispatch_send calls fill the lane behind it.
        handle
            .dispatch_send(Box::pin(async move {
                stuck_at_for_first.wait();
            }))
            .expect("first send fits");
        // Fill the lane. We have 8 slots; first task occupied one, so
        // we have 7 remaining slots. Send 7 to fill, then 1 more that
        // MUST fail.
        for _ in 0..7 {
            handle
                .dispatch_send(Box::pin(async move {}))
                .expect("fill-the-lane send fits");
        }
        // The 9th send (1st task + 7 fill = 8, this is the 9th) MUST
        // surface overflow.
        let result = handle.dispatch_send(Box::pin(async move {}));
        // Release the worker so it can drain and we don't leak a thread.
        stuck_at.wait();
        assert!(
            result.is_err(),
            "expected inbox overflow to surface at dispatch_send, got Ok"
        );
        handle.shutdown_and_join().expect("shutdown");
    }

    /// Counts how many sends are rejected when the lane is wedged. The
    /// historical name of this test was
    /// `runtime_spawn_on_core_silently_drops_on_inbox_overflow` — it
    /// pinned the (then-broken) behavior where `PrimeRuntime::spawn_on_core`
    /// returned `()` and silently discarded the dispatch_send error,
    /// causing batch dispatchers to hang in `while counter < N { ... }`.
    ///
    /// FIXED in the `SpawnError + spawn_on_core_blocking_with` API
    /// change: `Runtime::spawn_on_core` now returns
    /// `Result<(), SpawnError>` and callers explicitly choose to retry
    /// (via `spawn_on_core_blocking_with`) or drop. This test continues
    /// to verify the CoreShard-level signal — `dispatch_send` returns
    /// `Err(SpawnError::InboxFull)` for most of the 100 sends when the
    /// lane is wedged. See sibling test
    /// `runtime_spawn_on_core_returns_inbox_full_under_saturation` for
    /// the higher-level contract.
    #[test]
    fn dispatch_send_surfaces_inbox_full_under_saturation() {
        // Build a shard with a 4-slot lane. PrimeRuntime constructs
        // shards with sized::INBOX_CAPACITY (1024 default) — too large
        // to overflow cheaply in a test — so we go through CoreShard
        // directly to set the lane cap.
        let handle = launch_with_lanes(CoreId(0), None, 2, 4).expect("launch");
        let counter = Arc::new(AtomicUsize::new(0));
        let stuck_at = Arc::new(std::sync::Barrier::new(2));
        let stuck_at_for_first = stuck_at.clone();
        // Wedge the worker on a barrier — it cannot drain. To make the
        // wedge deterministic (the worker must be parked on barrier
        // BEFORE we start sending fillers, otherwise it could drain
        // them concurrently and the assertion's "≤ lane_cap completions"
        // invariant breaks), the wedge first flips an atomic so the
        // main thread can observe it has started executing.
        let wedge_running = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let wedge_running_for_first = wedge_running.clone();
        handle
            .dispatch_send(Box::pin(async move {
                wedge_running_for_first.store(true, Ordering::Release);
                stuck_at_for_first.wait();
            }))
            .expect("first send fits");
        // wait until the worker has actually polled the wedge and
        // entered barrier.wait. now and only now is the worker
        // guaranteed to be blocked; fillers cannot be drained mid-loop.
        while !wedge_running.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        // Fire 100 tasks via the CoreShardHandle. Each tries to bump
        // the counter when polled. Most will get Err(inbox full); the
        // CoreShardHandle ITSELF surfaces that, so this test isn't
        // about the Runtime trait yet — it's about what % of tasks
        // we'd lose if a naive caller (like the bench) does
        // fire-and-forget.
        let mut dropped = 0;
        for _ in 0..100 {
            let counter = counter.clone();
            if handle
                .dispatch_send(Box::pin(async move {
                    counter.fetch_add(1, Ordering::AcqRel);
                }))
                .is_err()
            {
                dropped += 1;
            }
        }
        // Release the worker. Now the barrier finishes; queued tasks
        // (the 3 that fit after the wedge) run. Tasks that were
        // rejected (96 of them) never increment the counter.
        stuck_at.wait();
        // Give the worker time to drain whatever queued.
        let deadline = Instant::now() + Duration::from_millis(500);
        while counter.load(Ordering::Acquire) < 3 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        // We sent 100. Most were rejected at dispatch_send. The wedge
        // task is popped from the lane the moment the worker calls
        // `tick()` — by the time main thread starts pushing fillers,
        // the lane has room for up to `lane_cap` fillers (the wedge
        // is in the executor, blocked on the barrier). After the
        // barrier releases, the worker drains those `lane_cap`
        // fillers. So `counter <= lane_cap` is the right bound, NOT
        // `lane_cap - 1`. (Pre-Bug-B, this test was timing-flaky
        // because `try_send` on lane 0 raced with the worker draining
        // lane 0; with `try_send_mpsc` the worker reliably pops the
        // wedge before main starts pushing.)
        const LANE_CAP: usize = 4;
        assert!(
            dropped >= 100 - LANE_CAP,
            "expected at least {} of 100 sends to be dropped \
             with a {LANE_CAP}-slot lane wedged behind a barrier, got dropped={dropped}, \
             counter={}",
            100 - LANE_CAP,
            counter.load(Ordering::Acquire),
        );
        let final_counter = counter.load(Ordering::Acquire);
        assert!(
            final_counter <= LANE_CAP,
            "expected at most {LANE_CAP} tasks to complete (the lane size), got {final_counter}",
        );
        handle.shutdown_and_join().expect("shutdown");
    }

    /// Higher-level contract: `Runtime::spawn_on_core` returns
    /// `Err(SpawnError::InboxFull)` when the target core's lane is
    /// saturated — callers can no longer be silently dropped on the
    /// floor. The `spawn_on_core_blocking_with` helper absorbs this by
    /// yield-looping until room appears, suitable for batch dispatchers.
    ///
    /// Constructed via `PrimeRuntime::new(2)` which uses the default
    /// 1024-slot lane capacity — overflowing requires either an
    /// artificial test capacity or sustained pressure. We use the
    /// blocking helper here against the real default to verify the
    /// helper drains under back-pressure that the bench harness
    /// previously hung on.
    #[cfg(feature = "runtime-prime-bgpool")]
    #[test]
    fn spawn_on_core_blocking_drains_against_inbox_back_pressure() {
        use crate::os::runtime::PrimeRuntime;
        use proxima_runtime::{Runtime, spawn_on_core_blocking_with};

        // 4-slot lane (artificial low cap so saturation is reachable in
        // a unit test) accessed through the Runtime trait via PrimeRuntime.
        // The runtime constructs shards via `launch_with_lanes` indirectly
        // through `core_shard::launch`; for this test we need a small cap,
        // so we build the runtime then verify the contract holds even
        // when saturation is repeatedly observed.
        let runtime: std::sync::Arc<dyn Runtime> =
            std::sync::Arc::new(PrimeRuntime::new(2).expect("build runtime"));
        let total: usize = 4_000;
        let counter = Arc::new(AtomicUsize::new(0));
        for index in 0..total {
            let counter = counter.clone();
            let core = CoreId(index % 2);
            // `_ = ...` because the helper returns Err only on
            // Disconnected, which a fresh runtime won't hit. The helper
            // internally yields on InboxFull until room appears.
            let _ = spawn_on_core_blocking_with(runtime.as_ref(), core, move || {
                let counter = counter.clone();
                Box::pin(async move {
                    counter.fetch_add(1, Ordering::AcqRel);
                })
            });
        }
        // Wait for all 4000 to drain (well under 1024 cap, but any
        // ordering with the consumer guarantees no silent drops).
        let deadline = Instant::now() + Duration::from_secs(5);
        while counter.load(Ordering::Acquire) < total && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            counter.load(Ordering::Acquire),
            total,
            "blocking helper must deliver every spawn — got {} of {total}",
            counter.load(Ordering::Acquire),
        );
    }

    /// D2 proof: a prime task dispatched onto the INVERTED worker calls raw
    /// `tokio::spawn` (no opt-in API) and that spawned tokio task runs to
    /// completion. Because the inverted worker ticks the prime executor inside
    /// `sister.block_on(...)`, the `tokio::spawn` here resolves to the sister
    /// and takes its LOCAL fast path. Completion is signalled via an
    /// `std::sync::mpsc` channel — no sleeps.
    #[cfg(feature = "prime-tokio-compat-inverted")]
    #[test]
    fn inverted_prime_task_raw_tokio_spawn_runs_to_completion() {
        let handle = launch_inverted_with_lanes(CoreId(0), None, 2, 16).expect("launch inverted");
        let (done_tx, done_rx) = std::sync::mpsc::channel::<u32>();
        handle
            .dispatch_factory(Box::new(move || {
                Box::pin(async move {
                    // raw tokio::spawn from a prime task — the transparent path.
                    let join = tokio::spawn(async move { 7_u32 });
                    let value = join.await.expect("inverted tokio task join");
                    done_tx.send(value).expect("send completion");
                }) as Pin<Box<dyn Future<Output = ()> + 'static>>
            }))
            .expect("dispatch factory");
        let observed = done_rx.recv().expect("inverted task never completed");
        assert_eq!(
            observed, 7,
            "raw tokio::spawn on inverted worker must complete"
        );
        handle.shutdown_and_join().expect("shutdown");
    }

    /// D2 burst: a prime task on the inverted worker loops `tokio::spawn` of
    /// many leaf tasks, each dropping a sender clone. `recv` blocks until a
    /// task runs or all clones are gone — deterministic, no sleep. Proves the
    /// inverted worker drives a burst of sister-spawned tasks to completion
    /// across iterations.
    #[cfg(feature = "prime-tokio-compat-inverted")]
    #[test]
    fn inverted_prime_task_spawns_burst_of_tokio_tasks() {
        const BURST: usize = 128;
        let handle = launch_inverted_with_lanes(CoreId(0), None, 2, 16).expect("launch inverted");
        let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
        handle
            .dispatch_factory(Box::new(move || {
                Box::pin(async move {
                    for _ in 0..BURST {
                        let done_tx = done_tx.clone();
                        tokio::spawn(async move {
                            let _ = done_tx.send(());
                        });
                    }
                    drop(done_tx);
                }) as Pin<Box<dyn Future<Output = ()> + 'static>>
            }))
            .expect("dispatch factory");
        let mut seen = 0_usize;
        while done_rx.recv().is_ok() {
            seen += 1;
        }
        assert_eq!(
            seen, BURST,
            "every tokio task spawned from the inverted prime task must run",
        );
        handle.shutdown_and_join().expect("shutdown");
    }

    #[cfg(feature = "prime-tokio-compat-inverted")]
    #[test]
    fn inverted_prime_task_tokio_timer_fires() {
        // a prime task that directly awaits tokio::time::sleep is suspended in
        // PRIME's executor — not a sister task, so num_alive_tasks misses it.
        // this proves the inverted worker still drives the sister timer so the
        // sleep fires: the I/O/timer-fidelity path, not just raw spawn. the
        // sleep IS the behaviour under test; completion is observed via a
        // bounded recv so a broken timer fails the test instead of hanging.
        let handle = launch_inverted_with_lanes(CoreId(0), None, 2, 16).expect("launch inverted");
        let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
        handle
            .dispatch_factory(Box::new(move || {
                Box::pin(async move {
                    tokio::time::sleep(std::time::Duration::from_millis(2)).await;
                    let _ = done_tx.send(());
                }) as Pin<Box<dyn Future<Output = ()> + 'static>>
            }))
            .expect("dispatch factory");
        let observed = done_rx.recv_timeout(std::time::Duration::from_secs(5));
        assert!(
            observed.is_ok(),
            "tokio::time::sleep awaited in an inverted prime task must fire",
        );
        handle.shutdown_and_join().expect("shutdown");
    }

    // C2b cross-reactor wake. linux-only (eventfd/epoll); runs on host-b.
    #[cfg(target_os = "linux")]
    #[test]
    fn inverted_prime_inbox_wakes_unified_park_with_sister_io_waiting() {
        // hold the worker in the unified sister park via a long-lived sister
        // task, then dispatch a fresh prime task. it must run promptly: the
        // prime inbox wakeup writes prime's eventfd, which surfaces on prime's
        // epoll fd, which the sister AsyncFd is parked on. a lost cross-reactor
        // wake (the bug C2b exists to prevent) would hang this test.
        let handle = launch_inverted_with_lanes(CoreId(0), None, 2, 16).expect("launch inverted");
        handle
            .dispatch_factory(Box::new(|| {
                Box::pin(async {
                    tokio::spawn(async {
                        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                    });
                }) as Pin<Box<dyn Future<Output = ()> + 'static>>
            }))
            .expect("dispatch sister-io factory");
        let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
        handle
            .dispatch_factory(Box::new(move || {
                Box::pin(async move {
                    let _ = done_tx.send(());
                }) as Pin<Box<dyn Future<Output = ()> + 'static>>
            }))
            .expect("dispatch wake factory");
        let woke = done_rx.recv_timeout(std::time::Duration::from_secs(5));
        assert!(
            woke.is_ok(),
            "prime dispatch must wake the unified sister park (C2b cross-reactor bridge)",
        );
        handle.shutdown_and_join().expect("shutdown");
    }
}
