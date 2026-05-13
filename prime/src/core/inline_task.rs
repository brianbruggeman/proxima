//! type-erased task wrapper that inlines small futures into a fixed
//! byte buffer to avoid the per-spawn `Pin<Box<dyn Future>>` heap
//! allocation that the `Runtime` trait surface mandates.
//!
//! shape: one stack-sized `InlineTask` struct that holds either the
//! future inline (when `size_of::<F>() <= INLINE_TASK_BYTES` and
//! `align_of::<F>() <= INLINE_TASK_ALIGN`) or a `Box<F>` pointer
//! tagged via the vtable. dispatch is via a per-`F` monomorphized
//! `poll_fn` — no `dyn Future` fat pointer indirection.
//!
//! the inline case is the common one: the spawn-burst bench's async
//! block captures one `Arc<AtomicUsize>` (~16 bytes); h2 handler
//! futures with a handful of captured channels run 24-48 bytes.
//! futures bigger than `INLINE_TASK_BYTES` fall back to `Box<F>` —
//! still cheaper than `Pin<Box<dyn Future>>` because the vtable is
//! external rather than carried in a fat pointer.
//!
//! safety contract: `InlineTask` is `Send` iff every `F` it can hold
//! is `Send`. constructors take `F: Send + 'static`, so the type-
//! erased payload is always `Send`. the storage is `MaybeUninit` and
//! is initialized exactly once at construction, dropped exactly once
//! by `Drop` (which calls the vtable's `drop_fn`).

use core::cell::UnsafeCell;
use core::future::Future;
use core::marker::PhantomData;
use core::mem::{ManuallyDrop, MaybeUninit};
use core::pin::Pin;
use core::ptr;
use core::task::{Context, Poll};

extern crate alloc;
use alloc::boxed::Box;

/// inline storage size — chosen so most async-block-based handler
/// futures fit without spilling to Box. 56 bytes leaves room for a
/// handful of captured `Arc<...>` + small state-machine discriminants.
pub const INLINE_TASK_BYTES: usize = 56;
/// alignment of the inline storage. matches `u64` so 8-byte-aligned
/// pointer-sized captures land naturally.
pub const INLINE_TASK_ALIGN: usize = 8;

#[repr(C, align(8))]
struct InlineStorage(MaybeUninit<[u8; INLINE_TASK_BYTES]>);

/// type-erased async task. `Send` because constructors require
/// `F: Send + 'static`.
pub struct InlineTask {
    vtable: &'static InlineTaskVtable,
    /// `UnsafeCell` because the poll path takes `&mut` into the storage
    /// via `vtable.poll_fn` while the surrounding `InlineTask` is
    /// behind a shared reference (`Slot` holds it inline). only the
    /// worker thread polls it, so no actual aliasing.
    storage: UnsafeCell<InlineStorage>,
}

// SAFETY: every constructor requires `F: Send + 'static`, so the
// type-erased payload behind `storage` is always `Send`. the `vtable`
// pointer is `&'static` to a function-pointer struct — itself `Send`.
unsafe impl Send for InlineTask {}

struct InlineTaskVtable {
    /// poll the future at `storage`. caller holds the worker-thread
    /// invariant so `storage` is exclusively borrowed.
    poll_fn: unsafe fn(*mut u8, &mut Context<'_>) -> Poll<()>,
    /// drop the future at `storage`. called once on `InlineTask::drop`
    /// regardless of whether the future ever ran.
    drop_fn: unsafe fn(*mut u8),
}

impl InlineTask {
    /// build an `InlineTask` from a typed future. inlines `F` if it
    /// fits the storage budget, otherwise falls back to a single
    /// heap allocation. no `Pin<Box<dyn Future>>` in either path.
    #[inline]
    #[must_use]
    pub fn new<F>(future: F) -> Self
    where
        F: Future<Output = ()> + Send + 'static,
    {
        if core::mem::size_of::<F>() <= INLINE_TASK_BYTES
            && core::mem::align_of::<F>() <= INLINE_TASK_ALIGN
        {
            Self::new_inline(future)
        } else {
            Self::new_boxed(future)
        }
    }

    #[inline]
    fn new_inline<F>(future: F) -> Self
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let mut storage = InlineStorage(MaybeUninit::uninit());
        // SAFETY: storage is uninit; we cast its byte buffer pointer
        // to `*mut F` and write the future into it. size_of::<F>() <=
        // INLINE_TASK_BYTES is enforced by the caller's branch.
        unsafe {
            let dst = storage.0.as_mut_ptr().cast::<F>();
            ptr::write(dst, future);
        }
        Self {
            vtable: vtable_for_inline::<F>(),
            storage: UnsafeCell::new(storage),
        }
    }

    #[inline]
    fn new_boxed<F>(future: F) -> Self
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let boxed: Box<F> = Box::new(future);
        let raw: *mut F = Box::into_raw(boxed);
        let mut storage = InlineStorage(MaybeUninit::uninit());
        // SAFETY: storage is uninit; we cast the byte buffer pointer
        // to `*mut *mut F` and write the raw pointer. *mut F fits in
        // 8 bytes, INLINE_TASK_BYTES >= 8.
        unsafe {
            let dst = storage.0.as_mut_ptr().cast::<*mut F>();
            ptr::write(dst, raw);
        }
        Self {
            vtable: vtable_for_boxed::<F>(),
            storage: UnsafeCell::new(storage),
        }
    }

    /// poll the inner future. invariant: caller has exclusive access
    /// (single-thread executor; same as `LocalExecutor` slot polls).
    ///
    /// # Safety
    ///
    /// must be called on the executor's worker thread that owns this
    /// `InlineTask`. only one `poll` may be in flight at a time on
    /// any given `InlineTask` instance.
    #[inline]
    pub unsafe fn poll(&self, context: &mut Context<'_>) -> Poll<()> {
        let storage_ptr = self.storage.get().cast::<u8>();
        // SAFETY: storage was initialized at construction with the
        // payload that vtable.poll_fn expects. invariant requires
        // exclusive access for the duration of the call.
        unsafe { (self.vtable.poll_fn)(storage_ptr, context) }
    }
}

impl Drop for InlineTask {
    #[inline]
    fn drop(&mut self) {
        let storage_ptr = self.storage.get().cast::<u8>();
        // SAFETY: storage was initialized exactly once at construction;
        // Drop runs exactly once, after which the storage is logically
        // uninit. vtable.drop_fn matches the construction path.
        unsafe { (self.vtable.drop_fn)(storage_ptr) }
    }
}

fn vtable_for_inline<F>() -> &'static InlineTaskVtable
where
    F: Future<Output = ()> + Send + 'static,
{
    struct Vtables<F>(PhantomData<F>);
    impl<F> Vtables<F>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        const VTABLE: InlineTaskVtable = InlineTaskVtable {
            poll_fn: poll_inline::<F>,
            drop_fn: drop_inline::<F>,
        };
    }
    &Vtables::<F>::VTABLE
}

fn vtable_for_boxed<F>() -> &'static InlineTaskVtable
where
    F: Future<Output = ()> + Send + 'static,
{
    struct Vtables<F>(PhantomData<F>);
    impl<F> Vtables<F>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        const VTABLE: InlineTaskVtable = InlineTaskVtable {
            poll_fn: poll_boxed::<F>,
            drop_fn: drop_boxed::<F>,
        };
    }
    &Vtables::<F>::VTABLE
}

unsafe fn poll_inline<F>(storage: *mut u8, context: &mut Context<'_>) -> Poll<()>
where
    F: Future<Output = ()>,
{
    // SAFETY: storage was constructed via `new_inline::<F>` so it holds
    // a properly aligned `F`. exclusive borrow by caller invariant.
    let future_ref: &mut F = unsafe { &mut *storage.cast::<F>() };
    // SAFETY: sound only because callers of `InlineTask::poll` uphold a
    // stronger promise than "moved by-value before the first poll": once
    // an `InlineTask` has been polled at all, its storage must never move
    // again for the rest of its life. `LocalExecutor` upholds this by
    // boxing a slot's `Task` once (see `local_executor::Slot::task`) and
    // always polling through that box in place — never copying the task
    // onto the caller's stack. callers that poll an `InlineTask` outside
    // that executor must uphold the same promise themselves.
    let pinned = unsafe { Pin::new_unchecked(future_ref) };
    pinned.poll(context)
}

unsafe fn poll_boxed<F>(storage: *mut u8, context: &mut Context<'_>) -> Poll<()>
where
    F: Future<Output = ()>,
{
    // SAFETY: storage holds a `*mut F` (heap-allocated by `new_boxed`).
    // we deref to a `&mut F`, then pin in place — the heap allocation
    // gives a stable address.
    let raw: *mut F = unsafe { *storage.cast::<*mut F>() };
    let pinned = unsafe { Pin::new_unchecked(&mut *raw) };
    pinned.poll(context)
}

unsafe fn drop_inline<F>(storage: *mut u8) {
    // SAFETY: storage was initialized with a `F` and `drop_inline` runs
    // exactly once. drop in place.
    unsafe { ptr::drop_in_place(storage.cast::<F>()) };
}

unsafe fn drop_boxed<F>(storage: *mut u8) {
    // SAFETY: storage holds a `*mut F`. reconstitute the Box and drop
    // — runs F's destructor + deallocates.
    let raw: *mut F = unsafe { *storage.cast::<*mut F>() };
    drop(unsafe { Box::from_raw(raw) });
}

/// the actual payload sent across the cross-core inbox for typed
/// spawns. `ManuallyDrop` lets the receiver take ownership without
/// the inbox's `MaybeUninit` slot running drop twice.
pub struct TypedSpawnRequest {
    pub task: ManuallyDrop<InlineTask>,
}

impl TypedSpawnRequest {
    #[must_use]
    pub fn new(task: InlineTask) -> Self {
        Self {
            task: ManuallyDrop::new(task),
        }
    }

    /// take ownership of the inner `InlineTask`. the caller becomes
    /// responsible for dropping it.
    ///
    /// # Safety
    ///
    /// must be called exactly once per `TypedSpawnRequest` instance.
    /// after the call, this struct's `task` field is logically
    /// uninitialized and must not be accessed.
    #[inline]
    pub unsafe fn take(mut self) -> InlineTask {
        // SAFETY: caller guarantees this is the only `take` call.
        unsafe { ManuallyDrop::take(&mut self.task) }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use alloc::sync::Arc;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use core::task::{Context, Waker};

    fn noop_context() -> Context<'static> {
        Context::from_waker(Waker::noop())
    }

    #[test]
    fn inline_task_runs_inline_future_to_completion() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_future = counter.clone();
        let task = InlineTask::new(async move {
            counter_for_future.fetch_add(1, Ordering::AcqRel);
        });
        let mut context = noop_context();
        // SAFETY: single-thread test; we own the task.
        let outcome = unsafe { task.poll(&mut context) };
        assert!(matches!(outcome, Poll::Ready(())));
        assert_eq!(counter.load(Ordering::Acquire), 1);
    }

    #[test]
    fn inline_task_falls_back_to_box_when_future_too_large() {
        // build an async block with a large local — forces size_of
        // beyond the inline budget.
        let big: [u64; 16] = [0; 16]; // 128 bytes
        let task = InlineTask::new(async move {
            let _hold = big;
        });
        let mut context = noop_context();
        // SAFETY: single-thread test.
        let outcome = unsafe { task.poll(&mut context) };
        assert!(matches!(outcome, Poll::Ready(())));
    }

    #[test]
    fn drop_runs_destructor_when_task_never_polled() {
        let counter = Arc::new(AtomicUsize::new(0));
        struct TrackedFuture {
            counter: Arc<AtomicUsize>,
        }
        impl Future for TrackedFuture {
            type Output = ();
            fn poll(self: core::pin::Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<()> {
                Poll::Pending
            }
        }
        impl Drop for TrackedFuture {
            fn drop(&mut self) {
                self.counter.fetch_add(1, Ordering::AcqRel);
            }
        }
        // construct the task with a concrete future struct so the captured
        // field is owned at construction time. dropping without polling
        // must still run F::drop (i.e. inline-task drop_fn dispatches
        // correctly).
        let task = InlineTask::new(TrackedFuture {
            counter: counter.clone(),
        });
        drop(task);
        assert_eq!(counter.load(Ordering::Acquire), 1);
    }
}
