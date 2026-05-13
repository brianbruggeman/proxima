//! per-thread `?Send` future executor. owns a task slab + split ready
//! queue: a `local_ready: UnsafeCell<Vec<u32>>` used by spawns and same-
//! thread wakes (no atomics, no contention) plus a `remote_ready:
//! Arc<SegQueue<u32>>` for wakes coming from other threads.
//!
//! the local/remote split is the same trick tokio's CurrentThread scheduler
//! uses: on the bench's hot path (1 producer, all wakes from worker thread)
//! the local Vec carries every push, atomic-free; only cross-thread wakers
//! pay the SegQueue cost.
//!
//! safety: `local_ready` is exposed via raw pointer in `TaskWaker`. wakers
//! deref it ONLY after confirming they're on the executor's worker thread
//! (via `CURRENT_EXEC_ID == self.exec_id`). on that thread, no concurrent
//! access is possible.
//!
//! design notes:
//! - wakers are cached per slab slot, built once when the slot is first
//!   allocated and reused across spawns to the same slot.
//! - the waker captures only the slot index (no generation). stale wakes
//!   from a dead future cause one wasted poll on whatever lives in the
//!   slot now; the future itself returns `Pending`/`Ready` correctly.
//!
//! no_std + alloc only — `core::*`, `alloc::*`, and `crossbeam_queue`.

extern crate alloc;

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
#[cfg(feature = "std")]
use core::cell::Cell;
use core::cell::{RefCell, UnsafeCell};
use core::future::Future;
use core::marker::PhantomData;
use core::pin::Pin;
use core::ptr;
#[cfg(feature = "std")]
use core::sync::atomic::AtomicU64;
use core::sync::atomic::{AtomicU32, Ordering};
use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

use crossbeam_queue::SegQueue;

use super::inline_task::InlineTask;
use super::sized;

type TaskFuture = Pin<Box<dyn Future<Output = ()> + 'static>>;

/// the actual stored future. inline variant skips the `Pin<Box<dyn Future>>`
/// fat-pointer dispatch entirely — `InlineTask` carries its own per-`F`
/// vtable, so polls are an indirect call through two function pointers
/// (slot waker + inline-task poll_fn) instead of through a `dyn Future`
/// vtable.
enum TaskBody {
    /// legacy / `!Send` / oversized future path. `spawn_local_pin` lands here.
    Boxed(TaskFuture),
    /// fast path for `Send + 'static` futures up to `INLINE_TASK_BYTES`.
    /// `spawn_local_inline` lands here. avoids the per-spawn `Box::pin`
    /// allocation and the fat-pointer dispatch.
    Inline(InlineTask),
}

impl TaskBody {
    #[inline(always)]
    fn poll(&mut self, context: &mut Context<'_>) -> Poll<()> {
        match self {
            Self::Boxed(future) => future.as_mut().poll(context),
            // SAFETY: executor's worker-thread invariant — only the owning
            // thread polls; no concurrent poll on the same `InlineTask`.
            Self::Inline(task) => unsafe { task.poll(context) },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaskHandle {
    index: u32,
    generation: u32,
}

struct Slot {
    /// boxed once when the slot is first allocated (see `spawn_with_body`),
    /// then reused (its `Option` toggled in place) across every future spawn
    /// that lands on this slot. gives the `Task` — and therefore any inline
    /// `InlineTask` bytes nested inside it — a heap address that is stable
    /// across `Vec<Slot>` resizes AND across nested `spawn_local` calls made
    /// from within a poll (which can themselves grow `slab.slots`). polling
    /// dereferences this box in place; the task is never copied onto the
    /// executor's stack, which is what let `!Unpin` futures get relocated
    /// after `Pin::new_unchecked` had already been established on them.
    task: Box<Option<Task>>,
    /// long-lived waker for this slot, built once when the slot was first
    /// allocated; reused across spawns. Boxed so the Waker has a stable
    /// heap address across Vec<Slot> resizes — we pass `&*waker` to
    /// `Context::from_waker` without cloning (saves an Arc bump per poll).
    waker: Box<Waker>,
}

struct Task {
    body: TaskBody,
    /// kept for future cancellation paths via `TaskHandle.generation`;
    /// the executor's hot path doesn't consult it.
    #[allow(dead_code)]
    generation: u32,
}

/// Hand-rolled waker state. Heap-allocated once per slab slot, reused for
/// the slot's lifetime. The associated `Waker` carries `Arc<TaskWaker>`
/// as its data pointer, with a static `RawWakerVTable` of plain function
/// pointers — no `dyn Wake` virtual dispatch, no per-fire vtable lookup
/// beyond the four function pointers in the vtable struct itself.
///
/// We could shave further by inlining `*Arc<TaskWaker>` into `RawWaker`'s
/// data pointer directly and managing the refcount by hand, but the
/// `Arc` form is already lock-free on the wake path and keeps the lifetime
/// math obviously safe.
/// callback fired by a cross-thread `TaskWaker::do_wake` after pushing
/// into `remote_ready`. typically signals the worker's reactor (kqueue
/// `EVFILT_USER` or `eventfd`) so a worker parked on `reactor.turn`
/// leaves the syscall and drains the cross-thread ready queue.
///
/// without this, a cross-thread wake reaches `remote_ready` but the
/// parked worker has no `kevent` / `epoll_wait` event to observe,
/// leaving the task wedged until some I/O-driven wake happens to
/// elapse.
pub type RemoteWake = Arc<dyn Fn() + Send + Sync + 'static>;

struct TaskWaker {
    /// used by `do_wake` to decide whether to push to `local_ready`
    /// or `remote_ready`. only meaningful under std (TLS-backed).
    #[cfg(feature = "std")]
    exec_id: u64,
    index: u32,
    /// raw pointer to the executor's `UnsafeCell<Vec<u32>>`. only deref
    /// when `CURRENT_EXEC_ID == exec_id` (i.e. we're on the worker thread).
    /// only populated under std.
    #[cfg(feature = "std")]
    local_ready: *const UnsafeCell<Vec<u32>>,
    /// MPSC fallback for wakes coming from other threads.
    remote_ready: Arc<SegQueue<u32>>,
    /// fired AFTER `remote_ready.push` on the cross-thread wake path.
    /// `None` when the executor is used outside a worker that owns a
    /// reactor (unit tests, in-process drivers); the cross-thread wake
    /// still publishes via the SegQueue but no parked syscall needs
    /// breaking.
    remote_wake: Option<RemoteWake>,
}

// SAFETY: `local_ready` raw pointer is only dereferenced when the waker's
// `wake` method confirms we're on the executor's owning thread via the
// `CURRENT_EXEC_ID` thread-local. on that thread, no other thread can
// access the underlying Vec (the executor is !Send), so single-threaded
// access invariants hold.
unsafe impl Send for TaskWaker {}
unsafe impl Sync for TaskWaker {}

impl TaskWaker {
    #[inline]
    fn do_wake(&self) {
        #[cfg(feature = "std")]
        if CURRENT_EXEC_ID.with(Cell::get) == self.exec_id {
            // SAFETY: same-thread; pointer valid for the executor's lifetime
            // (the waker's Arc keeps the TaskWaker alive, which holds the
            // pointer; the executor itself outlives every waker because
            // executors are dropped only after all spawned tasks finish).
            unsafe { (*(*self.local_ready).get()).push(self.index) };
            #[cfg(feature = "runtime-prime-reactor-trace")]
            crate::trace::record_ready_push();
            return;
        }
        self.remote_ready.push(self.index);
        #[cfg(feature = "runtime-prime-reactor-trace")]
        crate::trace::record_ready_push();
        if let Some(callback) = &self.remote_wake {
            (callback)();
        }
    }
}

// Hand-rolled RawWaker vtable. The data pointer is `Arc<TaskWaker>` as
// raw bytes (`Arc::into_raw` / `Arc::from_raw`). This avoids the `dyn Wake`
// vtable that `impl Wake for TaskWaker + Arc::into::<Waker>()` synthesises.
unsafe fn taskwaker_clone(data: *const ()) -> RawWaker {
    // SAFETY: `data` was produced by `Arc::into_raw` on `Arc<TaskWaker>`.
    // `Arc::increment_strong_count` bumps the refcount without consuming
    // the original.
    unsafe { Arc::increment_strong_count(data.cast::<TaskWaker>()) };
    RawWaker::new(data, &TASKWAKER_VTABLE)
}

unsafe fn taskwaker_wake(data: *const ()) {
    // SAFETY: takes ownership of the Arc that `data` represents (consumes
    // one refcount); reconstitute, call do_wake, then drop.
    let arc = unsafe { Arc::from_raw(data.cast::<TaskWaker>()) };
    arc.do_wake();
    // drop happens here, decrementing the refcount.
}

unsafe fn taskwaker_wake_by_ref(data: *const ()) {
    // SAFETY: we DON'T take ownership of the Arc — borrow the TaskWaker.
    let waker = unsafe { &*data.cast::<TaskWaker>() };
    waker.do_wake();
}

unsafe fn taskwaker_drop(data: *const ()) {
    // SAFETY: takes ownership of one Arc refcount; Arc::from_raw + drop.
    drop(unsafe { Arc::from_raw(data.cast::<TaskWaker>()) });
}

static TASKWAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(
    taskwaker_clone,
    taskwaker_wake,
    taskwaker_wake_by_ref,
    taskwaker_drop,
);

#[inline]
fn build_waker(inner: Arc<TaskWaker>) -> Waker {
    let raw = Arc::into_raw(inner).cast::<()>();
    let raw_waker = RawWaker::new(raw, &TASKWAKER_VTABLE);
    // SAFETY: the vtable functions all uphold the RawWaker contract
    // (clone bumps refcount, wake/wake_by_ref call into do_wake, drop
    // decrements). Each Waker built from this constructor owns one Arc
    // refcount, balanced by `taskwaker_drop` when the Waker is dropped.
    unsafe { Waker::from_raw(raw_waker) }
}

struct Slab {
    slots: Vec<Slot>,
    free_list: Vec<u32>,
}

/// per-thread executor. `!Send` by intent — one core owns it.
pub struct LocalExecutor {
    /// executor identity, compared against `CURRENT_EXEC_ID` TLS on
    /// each wake to route local vs remote. only meaningful under std.
    #[cfg(feature = "std")]
    id: u64,
    /// UnsafeCell rather than RefCell — the executor is `!Send` and the
    /// only callers that touch the slab are spawn_local_pin, tick, and the
    /// internal `poll_targets`/`complete` helpers, all of which run on the
    /// owning worker thread. Wakers don't touch the slab; they only push
    /// indices into `local_ready` / `remote_ready`. Skipping the RefCell
    /// dynamic borrow check saves ~10ns per slab access — observable
    /// across the poll path's slot lookups (one per poll, plus one on
    /// completion).
    slab: UnsafeCell<Slab>,
    /// fast local queue. only the worker thread (the one that observes
    /// `CURRENT_EXEC_ID == self.id`) pushes here. only used under std.
    #[cfg(feature = "std")]
    local_ready: UnsafeCell<Vec<u32>>,
    /// MPSC catchall for cross-thread wakes. always Arc-shared.
    remote_ready: Arc<SegQueue<u32>>,
    /// optional callback handed to every `TaskWaker` so cross-thread
    /// wakes can break the worker out of `reactor.turn`. supplied by
    /// the core-shard worker at startup; unit-test executors pass `None`.
    remote_wake: Option<RemoteWake>,
    next_generation: AtomicU32,
    /// slot index of the task currently being polled, or None when the
    /// executor is between polls. lives on the executor (per-instance
    /// state), NOT in a thread-local — the popcorn report flags TLS as
    /// a no_std blocker.
    /// gated on c15-prime-hooks; zero cost when the feature is off.
    #[cfg(feature = "runtime-prime-full")]
    current_slot: Cell<Option<u32>>,
    _not_send: PhantomData<*const ()>,
}

#[cfg(feature = "std")]
static NEXT_EXEC_ID: AtomicU64 = AtomicU64::new(1);

// std-only TLS; deferred-debt for no_std cliff. C1 (thread-identity-trait)
// in woolly-watching-cupcake routes this through ThreadIdentity with a
// std-backed default and a no_std single-thread stub.
// DC5 transitional gate: the TLS block + arm/disarm/push_ready/do_wake
// std path are unavailable under alloc-only. Under no_std, all wakes route
// via remote_ready (always correct, marginally slower). C3 (reactor-direct-wake)
// provides the no_std-clean replacement.
#[cfg(feature = "std")]
thread_local! {
    /// id of the LocalExecutor currently being polled on this thread, or 0
    /// when this thread is not actively driving any executor. wakers use
    /// this to decide whether to push to `local_ready` (fast, single-thread)
    /// or `remote_ready` (cross-thread SegQueue).
    static CURRENT_EXEC_ID: Cell<u64> = const { Cell::new(0) };
}

impl LocalExecutor {
    #[must_use]
    pub fn new() -> Self {
        Self::with_remote_wake(None)
    }

    /// build an executor with a `RemoteWake` callback. the callback fires
    /// after every cross-thread `TaskWaker::do_wake` push, giving the
    /// owning worker a chance to leave a parked `reactor.turn` and drain
    /// `remote_ready`.
    ///
    /// pass `None` for in-process drivers / unit-test executors that
    /// don't park on a reactor.
    #[must_use]
    pub fn with_remote_wake(remote_wake: Option<RemoteWake>) -> Self {
        // pre-reserve the slab to TASK_SLAB_INITIAL_CAP so the first burst
        // of spawns doesn't trigger Vec's doubling-reallocation cycle.
        Self {
            #[cfg(feature = "std")]
            id: NEXT_EXEC_ID.fetch_add(1, Ordering::Relaxed),
            slab: UnsafeCell::new(Slab {
                slots: Vec::with_capacity(sized::TASK_SLAB_INITIAL_CAP),
                free_list: Vec::with_capacity(sized::TASK_SLAB_INITIAL_CAP),
            }),
            #[cfg(feature = "std")]
            local_ready: UnsafeCell::new(Vec::with_capacity(sized::TASK_SLAB_INITIAL_CAP)),
            remote_ready: Arc::new(SegQueue::new()),
            remote_wake,
            next_generation: AtomicU32::new(1),
            #[cfg(feature = "runtime-prime-full")]
            current_slot: Cell::new(None),
            _not_send: PhantomData,
        }
    }

    /// declare this thread as the executor's owning worker. tasks polled by
    /// `tick`/`block_on` will see `CURRENT_EXEC_ID == self.id` and route
    /// wakers to the local queue. the caller MUST call `disarm` before the
    /// executor goes away or the thread switches contexts.
    ///
    /// Under alloc-only (no `std`), this is a no-op: all wakes route via
    /// `remote_ready` until C3 (reactor-direct-wake) lands.
    pub fn arm(&self) {
        #[cfg(feature = "std")]
        CURRENT_EXEC_ID.with(|cell| cell.set(self.id));
    }

    /// undo `arm`. wakers fired after this call will route to `remote_ready`.
    ///
    /// Under alloc-only (no `std`), this is a no-op.
    pub fn disarm(&self) {
        #[cfg(feature = "std")]
        CURRENT_EXEC_ID.with(|cell| cell.set(0));
    }

    /// return the slab slot index of the task currently being polled,
    /// or `None` when called outside a poll (between ticks, during spawn,
    /// etc.). only meaningful when called on the owning worker thread.
    /// gated on `runtime-prime-full`; zero cost when the feature is off.
    #[cfg(feature = "runtime-prime-full")]
    #[must_use]
    pub fn current_slot(&self) -> Option<u32> {
        self.current_slot.get()
    }

    /// spawn a `?Send` future. task runs on the owning thread.
    pub fn spawn_local<F>(&self, future: F) -> TaskHandle
    where
        F: Future<Output = ()> + 'static,
    {
        self.spawn_local_pin(Box::pin(future))
    }

    /// spawn an already-pinned `?Send` future.
    pub fn spawn_local_pin(&self, future: TaskFuture) -> TaskHandle {
        self.spawn_with_body(TaskBody::Boxed(future))
    }

    /// fast path for `Send + 'static` futures: skip the per-spawn
    /// `Box::pin` allocation by carrying the future inline in an
    /// `InlineTask` (when it fits) or a `Box<F>` (when it doesn't).
    /// Either way the slab dispatches via a per-`F` vtable rather than
    /// a `dyn Future` fat-pointer.
    pub fn spawn_local_inline(&self, task: InlineTask) -> TaskHandle {
        self.spawn_with_body(TaskBody::Inline(task))
    }

    /// eager-poll variant of [`spawn_local_inline`]. Polls the task
    /// ONCE on the spawn stack with `Waker::noop()`. If the future
    /// resolves on the first poll (the common case for `counter +=
    /// 1` style sync work that dominates spawn-burst workloads), the
    /// slab is never touched and the return is `None`. Otherwise the
    /// task is installed in the slab and pushed to `local_ready` so
    /// the next `tick()` re-polls it with the real per-slot Waker
    /// — the future then re-registers against the actual waker.
    ///
    /// # Correctness notes
    ///
    /// **The first poll uses a noop Waker.** A future that captures
    /// `ctx.waker().clone()` on first poll and STORES it (without
    /// re-registering on subsequent polls) will silently never wake.
    /// This is OK for well-behaved futures (tokio-style, async/await
    /// state machines, proxima's reactor-driven I/O) — all of them
    /// re-register on every `Pending` poll. A future that registers
    /// once and trusts the waker pointer to be stable across polls
    /// is buggy under any executor that uses per-task wakers, not
    /// just this one; we won't accommodate it.
    ///
    /// **No starvation risk for Pending tasks.** On `Pending`, the
    /// task is installed in the slab via `spawn_with_body`, which
    /// calls `push_ready` — `tick()` will re-poll it with the real
    /// per-slot Waker on the immediately-following iteration. The
    /// future's first-poll noop-waker registration is irrelevant
    /// because the re-poll happens unconditionally, not in response
    /// to the noop-waker firing.
    ///
    /// **No double-poll risk for Ready tasks.** If the first poll
    /// returns `Ready(())`, the `InlineTask` is dropped on the spawn
    /// stack (its `Drop` impl runs the vtable's `drop_fn`). The
    /// future never enters the slab; no second poll happens.
    ///
    /// **Known residual hazard for `!Unpin` futures on `Pending`.** The
    /// first poll above happens on THIS function's stack, before the
    /// task is installed in the slab; if it returns `Pending`, `task` is
    /// then moved by value into `spawn_with_body`. For a future that is
    /// `!Unpin` and has already established internal self-references
    /// during that one poll, this move is unsound by the same argument
    /// documented at `Slot::task` and `poll_inline` — it just happens
    /// once here (the stack-to-slab transition) rather than on every
    /// tick. Fixing it would mean installing into the slab before ANY
    /// poll, which gives up the zero-slab-touch fast path this function
    /// exists for (see `eager_poll_thousand_sync_tasks_zero_slab_growth`).
    /// Left as-is pending a decision on that trade-off; every poll after
    /// the first is in-place and address-stable.
    pub fn spawn_local_inline_eager(&self, task: InlineTask) -> Option<TaskHandle> {
        let noop = Waker::noop();
        let mut context = Context::from_waker(noop);
        // SAFETY: executor's worker-thread invariant — we own the
        // task on this stack; nobody else can poll it concurrently.
        match unsafe { task.poll(&mut context) } {
            Poll::Ready(()) => None,
            Poll::Pending => Some(self.spawn_with_body(TaskBody::Inline(task))),
        }
    }

    fn spawn_with_body(&self, body: TaskBody) -> TaskHandle {
        let generation = self.next_generation.fetch_add(1, Ordering::Relaxed);
        // SAFETY: see `LocalExecutor::slab` field doc. We own the worker
        // thread for the duration of any spawn/tick path; no concurrent
        // borrow can exist (wakers don't touch the slab).
        let slab = unsafe { &mut *self.slab.get() };
        let task = Task { body, generation };
        let index = if let Some(free) = slab.free_list.pop() {
            // writes into the slot's existing box in place — no new
            // allocation, and the task's storage address (already visited
            // by a prior occupant) never had to move to make room for it.
            *slab.slots[free as usize].task = Some(task);
            free
        } else {
            let raw = slab.slots.len();
            assert!(raw < u32::MAX as usize, "task slab capacity > u32::MAX");
            let index = raw as u32;
            // first-time slot allocation — build the waker once.
            let inner = Arc::new(TaskWaker {
                #[cfg(feature = "std")]
                exec_id: self.id,
                index,
                #[cfg(feature = "std")]
                local_ready: &self.local_ready as *const _,
                remote_ready: self.remote_ready.clone(),
                remote_wake: self.remote_wake.clone(),
            });
            let waker = build_waker(inner);
            // one-time box for this slot's lifetime — every future spawn
            // that lands here (via the free_list branch above) reuses it.
            slab.slots.push(Slot {
                task: Box::new(Some(task)),
                waker: Box::new(waker),
            });
            index
        };
        let handle = TaskHandle { index, generation };
        self.push_ready(index);
        handle
    }

    fn push_ready(&self, index: u32) {
        #[cfg(feature = "std")]
        if CURRENT_EXEC_ID.with(Cell::get) == self.id {
            // SAFETY: same-thread; no concurrent access possible.
            unsafe { (*self.local_ready.get()).push(index) };
            return;
        }
        self.remote_ready.push(index);
    }

    /// poll all currently-ready tasks once. returns the number of tasks
    /// polled.
    pub fn tick(&self) -> usize {
        #[cfg(feature = "std")]
        {
            // drain any cross-thread wakes into the local queue first.
            while let Some(index) = self.remote_ready.pop() {
                // SAFETY: same-thread; tick() runs only on the worker thread.
                unsafe { (*self.local_ready.get()).push(index) };
            }
            let mut polled = 0;
            // process local_ready by repeatedly draining its current snapshot.
            // wakes from within poll push to the same Vec; we re-check after
            // each batch.
            loop {
                // SAFETY: same-thread.
                let batch_len = unsafe { (*self.local_ready.get()).len() };
                if batch_len == 0 {
                    break;
                }
                for _ in 0..batch_len {
                    // SAFETY: same-thread. pop_front equivalent via swap_remove(0)
                    // would change order; we use pop() which is LIFO. for our
                    // workload order doesn't matter — the task body decides.
                    let index = unsafe { (*self.local_ready.get()).pop() };
                    let Some(index) = index else { break };
                    polled += 1;
                    let Some((task_ptr, waker_ptr)) = self.poll_targets(index) else {
                        continue;
                    };
                    // SAFETY: `waker_ptr` points into the Box<Waker> owned by
                    // slot[index]. The Box has a stable heap address that does
                    // not move when the surrounding Vec<Slot> resizes (only
                    // the Box pointer in the slot moves, not the Waker
                    // pointee). The Box outlives this poll because we never
                    // drop the slot or replace its waker during a poll cycle.
                    let waker_ref: &Waker = unsafe { &*waker_ptr };
                    let mut context = Context::from_waker(waker_ref);
                    #[cfg(feature = "runtime-prime-full")]
                    self.current_slot.set(Some(index));
                    #[cfg(feature = "runtime-prime-reactor-trace")]
                    crate::trace::record_task_poll_start();
                    // SAFETY: `task_ptr` points into slot[index]'s
                    // `Box<Option<Task>>`, a heap allocation independent of
                    // `slab.slots`'s buffer. exclusive access is guaranteed
                    // by the single-poll-per-slot invariant, and stays valid
                    // even if this poll's body makes a nested `spawn_local`
                    // call that grows `slab.slots`.
                    let poll_result = unsafe { (*task_ptr).body.poll(&mut context) };
                    #[cfg(feature = "runtime-prime-full")]
                    self.current_slot.set(None);
                    if let Poll::Ready(()) = poll_result {
                        self.complete(index);
                    }
                }
                // any new wakes added to local during this batch get processed
                // in the next loop iteration; remote may also have arrived.
                while let Some(index) = self.remote_ready.pop() {
                    // SAFETY: same-thread.
                    unsafe { (*self.local_ready.get()).push(index) };
                }
            }
            polled
        }

        // under alloc-only (no TLS), all wakes arrive via remote_ready.
        // drain it directly until empty. C3 (reactor-direct-wake) provides
        // the std-parity performance replacement.
        #[cfg(not(feature = "std"))]
        {
            let mut polled = 0;
            while let Some(index) = self.remote_ready.pop() {
                polled += 1;
                let Some((task_ptr, waker_ptr)) = self.poll_targets(index) else {
                    continue;
                };
                let waker_ref: &Waker = unsafe { &*waker_ptr };
                let mut context = Context::from_waker(waker_ref);
                #[cfg(feature = "runtime-prime-reactor-trace")]
                crate::trace::record_task_poll_start();
                // SAFETY: see the `std` tick() branch above — same
                // in-place-poll invariant applies here.
                let poll_result = unsafe { (*task_ptr).body.poll(&mut context) };
                if let Poll::Ready(()) = poll_result {
                    self.complete(index);
                }
            }
            polled
        }
    }

    /// drive this executor until the supplied root future resolves — the
    /// single-thread, in-place spelling of the workspace `block_on` verb (cf.
    /// the no-runtime `proxima_primitives::block_on` poll loop and the
    /// runtime-backed `proxima_runtime::block_on(&dyn Runtime, ..)`). Here the
    /// executor IS the driver: it ticks its own ready queue on the calling
    /// thread until `root` produces a value.
    pub fn block_on<F>(&self, root: F) -> F::Output
    where
        F: Future + 'static,
        F::Output: 'static,
    {
        self.arm();
        let output: Arc<RefCell<Option<F::Output>>> = Arc::new(RefCell::new(None));
        let output_for_task = output.clone();
        self.spawn_local(async move {
            let value = root.await;
            *output_for_task.borrow_mut() = Some(value);
        });
        loop {
            self.tick();
            if output.borrow().is_some() {
                break;
            }
            assert!(
                !self.ready_is_empty(),
                "block_on: no progress (no reactor wakeups; integrate C2/C3 in C5)",
            );
        }
        self.disarm();
        match output.borrow_mut().take() {
            Some(value) => value,
            None => unreachable!("root task completed but output cell empty"),
        }
    }

    /// Borrow the task and long-lived waker resident in `slot[index]`,
    /// without moving either out of the slab. `task_ptr` points into the
    /// slot's own `Box<Option<Task>>` — a heap allocation independent of
    /// `slab.slots`'s backing buffer, so it stays valid even if the poll
    /// this pointer is used for makes a nested `spawn_local` call that
    /// grows `slab.slots` (that only relocates `Slot` structs, i.e. two
    /// box pointers per slot, never the boxed `Task`/`Waker` bytes
    /// themselves). Returns `None` if the slot is empty (a stale wake on
    /// a completed or never-spawned index) so the caller can skip.
    ///
    /// Pre-v11 this took the `Task` out by value onto the caller's stack
    /// for the poll and moved it back on `Pending` — a per-poll `Box::pin`
    /// alloc had already been eliminated by then (see the `InlineTask`
    /// doc), but the stack round trip itself relocated any inline future
    /// after `Pin::new_unchecked` had already been established on it,
    /// which is unsound for `!Unpin` futures. Boxing `task` once per slot
    /// (the same treatment `waker` already got) and polling through the
    /// box removes that relocation entirely.
    #[inline(always)]
    fn poll_targets(&self, index: u32) -> Option<(*mut Task, *const Waker)> {
        // SAFETY: see `LocalExecutor::slab` field doc.
        let slab = unsafe { &mut *self.slab.get() };
        let slot = slab.slots.get_mut(index as usize)?;
        let waker_ptr: *const Waker = &*slot.waker;
        let task_ref: &mut Task = (*slot.task).as_mut()?;
        let task_ptr: *mut Task = task_ref;
        Some((task_ptr, waker_ptr))
    }

    #[inline(always)]
    fn complete(&self, index: u32) {
        // SAFETY: see `LocalExecutor::slab` field doc.
        let slab = unsafe { &mut *self.slab.get() };
        let Some(slot) = slab.slots.get_mut(index as usize) else {
            return;
        };
        if slot.task.is_some() {
            // drops the finished `Task` in place; the box itself is kept
            // (and its `Option` reused) for the slot's next occupant.
            *slot.task = None;
            slab.free_list.push(index);
        }
    }

    fn ready_is_empty(&self) -> bool {
        #[cfg(feature = "std")]
        {
            // SAFETY: same-thread.
            let local_empty = unsafe { (*self.local_ready.get()).is_empty() };
            local_empty && self.remote_ready.is_empty()
        }
        #[cfg(not(feature = "std"))]
        self.remote_ready.is_empty()
    }
}

impl Default for LocalExecutor {
    fn default() -> Self {
        Self::new()
    }
}

// suppress unused-import warning when no inherent usage relies on `ptr`.
#[allow(dead_code)]
fn _ptr_unused_warning_guard() -> *const () {
    ptr::null()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use alloc::rc::Rc as StdRc;
    use core::cell::Cell as StdCell;
    use core::marker::PhantomPinned;

    #[test]
    fn block_on_runs_a_ready_future_to_completion() {
        let executor = LocalExecutor::new();
        let value = executor.block_on(async { 42_u64 });
        assert_eq!(value, 42);
    }

    #[test]
    fn spawn_local_runs_nested_tasks() {
        let executor = LocalExecutor::new();
        let counter = StdRc::new(StdCell::new(0_u32));
        let counter_for_nested = counter.clone();
        executor.arm();
        executor.spawn_local(async move {
            counter_for_nested.set(counter_for_nested.get() + 1);
        });
        executor.disarm();
        let counter_for_root = counter.clone();
        executor.block_on(async move {
            counter_for_root.set(counter_for_root.get() + 10);
        });
        assert_eq!(counter.get(), 11);
    }

    #[test]
    fn pending_future_resumes_after_explicit_wake() {
        struct YieldOnce {
            yielded: bool,
        }
        impl Future for YieldOnce {
            type Output = ();
            fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
                let this = self.get_mut();
                if this.yielded {
                    Poll::Ready(())
                } else {
                    this.yielded = true;
                    context.waker().wake_by_ref();
                    Poll::Pending
                }
            }
        }
        let executor = LocalExecutor::new();
        let observed = StdRc::new(StdCell::new(false));
        let observed_for_task = observed.clone();
        executor.block_on(async move {
            YieldOnce { yielded: false }.await;
            observed_for_task.set(true);
        });
        assert!(observed.get());
    }

    #[test]
    fn tick_returns_count_of_polled_tasks() {
        let executor = LocalExecutor::new();
        executor.arm();
        for _ in 0..5 {
            executor.spawn_local(async {});
        }
        let polled = executor.tick();
        executor.disarm();
        assert!(polled >= 5, "expected at least 5 polled, got {polled}");
    }

    #[test]
    fn many_independent_tasks_all_complete() {
        let executor = LocalExecutor::new();
        let counter = StdRc::new(StdCell::new(0_u32));
        executor.arm();
        for _ in 0..1000 {
            let counter = counter.clone();
            executor.spawn_local(async move {
                counter.set(counter.get() + 1);
            });
        }
        executor.disarm();
        executor.block_on(async {});
        assert_eq!(counter.get(), 1000);
    }

    #[test]
    fn future_that_completes_immediately_does_not_loop() {
        let executor = LocalExecutor::new();
        let value = executor.block_on(async { 7_u8 });
        assert_eq!(value, 7);
    }

    // ----- eager-poll slab-elision correctness tests -----
    //
    // these hit the dragon: noop-waker first-poll, slab installation
    // on Pending, re-poll with real waker. each test pins one
    // failure mode the elision must NOT introduce.

    use super::InlineTask;
    use core::sync::atomic::AtomicU32;
    use core::sync::atomic::AtomicUsize as StdAtomicUsizeLE;

    #[test]
    fn eager_poll_returns_none_for_sync_ready_future() {
        // contract: sync-ready future never touches the slab. on
        // success the slab.slots Vec stays at len 0 (no slot
        // allocated, no waker built).
        let executor = LocalExecutor::new();
        executor.arm();
        let observed = Arc::new(AtomicU32::new(0));
        let observed_for_task = observed.clone();
        let task = InlineTask::new(async move {
            observed_for_task.fetch_add(1, Ordering::AcqRel);
        });
        let handle = executor.spawn_local_inline_eager(task);
        assert!(handle.is_none(), "sync-ready future must return None");
        assert_eq!(
            observed.load(Ordering::Acquire),
            1,
            "future body ran exactly once"
        );
        // SAFETY: same thread, no concurrent access.
        let slab_len = unsafe { (*executor.slab.get()).slots.len() };
        assert_eq!(
            slab_len, 0,
            "sync-ready future must not allocate a slab slot — leaked {slab_len}",
        );
        executor.disarm();
    }

    #[test]
    fn eager_poll_pending_future_runs_to_completion_via_real_waker() {
        // contract: a future that returns Pending on first poll must
        // still drive to completion when the real per-slot waker is
        // installed and re-polls happen via tick(). this would fail
        // if the noop waker were treated as canonical.
        struct YieldOnce {
            yielded: bool,
        }
        impl Future for YieldOnce {
            type Output = ();
            fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
                let this = self.get_mut();
                if this.yielded {
                    Poll::Ready(())
                } else {
                    this.yielded = true;
                    // re-register with whatever waker context gives us.
                    // on the real second poll this picks up the slab
                    // slot's waker; on the noop first poll it picks up
                    // the noop. either way, wake_by_ref + return Pending.
                    context.waker().wake_by_ref();
                    Poll::Pending
                }
            }
        }
        let executor = LocalExecutor::new();
        executor.arm();
        let task = InlineTask::new(YieldOnce { yielded: false });
        let handle = executor.spawn_local_inline_eager(task);
        assert!(handle.is_some(), "Pending future must enter the slab");
        let polled = executor.tick();
        executor.disarm();
        assert!(polled >= 1, "tick must observe the slab-installed task");
    }

    #[test]
    fn eager_poll_does_not_leak_first_pending_future_drop() {
        // contract: if a future returns Pending and is installed in
        // the slab, its destructor runs exactly once — when the slab
        // slot is freed on Ready (later) OR when the executor drops.
        // never twice (e.g., from the eager-poll stack frame too).
        struct DropOnce {
            counter: Arc<AtomicU32>,
            polled: bool,
        }
        impl Future for DropOnce {
            type Output = ();
            fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
                let this = self.get_mut();
                if this.polled {
                    Poll::Ready(())
                } else {
                    this.polled = true;
                    context.waker().wake_by_ref();
                    Poll::Pending
                }
            }
        }
        impl Drop for DropOnce {
            fn drop(&mut self) {
                self.counter.fetch_add(1, Ordering::AcqRel);
            }
        }
        let drops = Arc::new(AtomicU32::new(0));
        let executor = LocalExecutor::new();
        executor.arm();
        let task = InlineTask::new(DropOnce {
            counter: drops.clone(),
            polled: false,
        });
        let _ = executor.spawn_local_inline_eager(task);
        executor.tick();
        executor.disarm();
        assert_eq!(
            drops.load(Ordering::Acquire),
            1,
            "future destructor must run exactly once",
        );
    }

    #[test]
    fn eager_poll_yield_n_times_eventually_completes() {
        // starvation guard: a future that yields N times must
        // eventually run to completion. ten yields, all driven by
        // wake_by_ref(real-waker) → tick re-poll cycle.
        const YIELDS: u32 = 10;
        struct YieldN {
            count: u32,
            limit: u32,
        }
        impl Future for YieldN {
            type Output = ();
            fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
                let this = self.get_mut();
                if this.count >= this.limit {
                    return Poll::Ready(());
                }
                this.count += 1;
                context.waker().wake_by_ref();
                Poll::Pending
            }
        }
        let executor = LocalExecutor::new();
        executor.arm();
        let task = InlineTask::new(YieldN {
            count: 0,
            limit: YIELDS,
        });
        let handle = executor.spawn_local_inline_eager(task);
        assert!(handle.is_some(), "first Pending poll must install slab");
        for _ in 0..(YIELDS + 1) {
            let polled = executor.tick();
            if polled == 0 {
                break;
            }
        }
        executor.disarm();
        // task is complete iff its slot's task field is None and the
        // free_list contains its index. simpler: re-tick must observe
        // no ready tasks.
        let polled = executor.tick();
        assert_eq!(polled, 0, "task must have completed and been freed");
    }

    #[test]
    fn eager_poll_future_that_stores_noop_waker_still_resumes_via_real_waker() {
        // dragon: a "lazy" future stores the FIRST poll's waker and
        // only re-registers on transition. the noop waker would
        // silently never fire. our re-poll-on-install policy must
        // override this — the real per-slot waker is observed via
        // the second tick-driven poll. the future then re-stores the
        // real one and subsequent wakes work.
        //
        // we model this by having the future deliberately NOT
        // wake_by_ref on first poll (it "thinks" the stored waker
        // is enough). on second poll (with real waker), it triggers
        // wake_by_ref and yields Pending; on third poll, Ready.
        struct StoreFirstPollWaker {
            captured: Option<Waker>,
            polls: u32,
        }
        impl Future for StoreFirstPollWaker {
            type Output = ();
            fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
                let this = self.get_mut();
                this.polls += 1;
                if this.polls == 1 {
                    // first poll: capture the noop waker and yield.
                    // we do NOT call wake_by_ref — the noop wouldn't
                    // do anything anyway, but in any case we rely on
                    // executor policy: install-then-repoll.
                    this.captured = Some(context.waker().clone());
                    return Poll::Pending;
                }
                if this.polls == 2 {
                    // second poll (driven by executor's mandatory
                    // re-poll after slab install). capture the REAL
                    // waker and yield via wake_by_ref so we drive a
                    // third poll. captured waker on second-poll wake
                    // makes us self-yielding.
                    this.captured = Some(context.waker().clone());
                    if let Some(waker) = &this.captured {
                        waker.wake_by_ref();
                    }
                    return Poll::Pending;
                }
                Poll::Ready(())
            }
        }
        let executor = LocalExecutor::new();
        executor.arm();
        let task = InlineTask::new(StoreFirstPollWaker {
            captured: None,
            polls: 0,
        });
        let handle = executor.spawn_local_inline_eager(task);
        assert!(handle.is_some(), "first-poll Pending must install slab");
        // drive ticks until quiescent
        for _ in 0..5 {
            if executor.tick() == 0 {
                break;
            }
        }
        executor.disarm();
        let polled_again = executor.tick();
        assert_eq!(
            polled_again, 0,
            "future must have reached Ready and been freed",
        );
    }

    #[test]
    fn eager_poll_thousand_sync_tasks_zero_slab_growth() {
        // perf-correctness: 1000 sync-ready tasks must leave the
        // slab empty. if even one task accidentally installs a slot,
        // slab.slots.len() > 0 catches it.
        let executor = LocalExecutor::new();
        executor.arm();
        let counter = Arc::new(StdAtomicUsizeLE::new(0));
        for _ in 0..1_000 {
            let counter = counter.clone();
            let task = InlineTask::new(async move {
                counter.fetch_add(1, Ordering::AcqRel);
            });
            let handle = executor.spawn_local_inline_eager(task);
            assert!(handle.is_none(), "sync-ready task wrongly slotted");
        }
        executor.disarm();
        assert_eq!(counter.load(Ordering::Acquire), 1_000);
        // SAFETY: same-thread; the eager path never touched the slab.
        let slab_len = unsafe { (*executor.slab.get()).slots.len() };
        assert_eq!(slab_len, 0, "1000 sync tasks grew slab by {slab_len}");
    }

    #[test]
    fn eager_poll_mixed_sync_and_async_does_not_corrupt_ordering() {
        // alternating sync and Pending-then-Ready tasks. the slab
        // installs only for the Pending cases; ticks drive each to
        // Ready. final counter must reflect every task body running.
        let executor = LocalExecutor::new();
        executor.arm();
        let counter = Arc::new(AtomicU32::new(0));
        struct YieldOnce {
            yielded: bool,
            counter: Arc<AtomicU32>,
        }
        impl Future for YieldOnce {
            type Output = ();
            fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
                let this = self.get_mut();
                if this.yielded {
                    this.counter.fetch_add(1, Ordering::AcqRel);
                    Poll::Ready(())
                } else {
                    this.yielded = true;
                    context.waker().wake_by_ref();
                    Poll::Pending
                }
            }
        }
        for index in 0..200_u32 {
            if index % 2 == 0 {
                let counter = counter.clone();
                let task = InlineTask::new(async move {
                    counter.fetch_add(1, Ordering::AcqRel);
                });
                let _ = executor.spawn_local_inline_eager(task);
            } else {
                let task = InlineTask::new(YieldOnce {
                    yielded: false,
                    counter: counter.clone(),
                });
                let _ = executor.spawn_local_inline_eager(task);
            }
        }
        // drive all pending tasks
        for _ in 0..10 {
            if executor.tick() == 0 {
                break;
            }
        }
        executor.disarm();
        assert_eq!(
            counter.load(Ordering::Acquire),
            200,
            "every task body must have run exactly once",
        );
    }

    #[test]
    fn inline_task_storage_address_is_stable_across_pending_polls() {
        // a `!Unpin` future that records the address of its own field on
        // first poll and asserts every later poll observes the SAME
        // address. `checkout`/`checkin` moving the whole `Task` (and
        // therefore the `InlineTask`'s inline byte storage) out to
        // `tick()`'s stack and back violates the address stability that
        // `Pin` promises once a future has been polled — this test
        // fails today because that move happens on every `Pending` poll.
        struct AddressWitness {
            payload: u64,
            recorded_address: Option<*const u64>,
            polls: u32,
            _pin: PhantomPinned,
        }
        // SAFETY: test-only future; single-threaded executor drives it,
        // so the raw pointer field is never touched concurrently.
        unsafe impl Send for AddressWitness {}
        impl Future for AddressWitness {
            type Output = ();
            fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
                // SAFETY: we only read/write plain fields, never move `self`.
                let this = unsafe { self.get_unchecked_mut() };
                this.polls += 1;
                let current_address = &raw const this.payload;
                match this.recorded_address {
                    None => this.recorded_address = Some(current_address),
                    Some(recorded) => assert_eq!(
                        recorded, current_address,
                        "InlineTask storage moved between polls at poll {}: pin violated",
                        this.polls,
                    ),
                }
                if this.polls >= 3 {
                    Poll::Ready(())
                } else {
                    context.waker().wake_by_ref();
                    Poll::Pending
                }
            }
        }

        let executor = LocalExecutor::new();
        executor.arm();
        let task = InlineTask::new(AddressWitness {
            payload: 0,
            recorded_address: None,
            polls: 0,
            _pin: PhantomPinned,
        });
        executor.spawn_local_inline(task);
        for _ in 0..5 {
            if executor.tick() == 0 {
                break;
            }
        }
        executor.disarm();
    }
}
