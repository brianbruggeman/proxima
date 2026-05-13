//! Async, thread-non-blocking mutex for the executor-driven tiers.
//!
//! Folded from the former `proxima-lock` crate (Workstream F, RISC-dedup;
//! `proxima-lock` had zero workspace consumers). Exposed at the crate root
//! as [`AsyncMutex`] — distinct from [`crate::sync::Mutex`] (the `futures::lock::
//! Mutex`-backed async mutex already at the crate root, std-only today) and
//! from [`crate::sync::blocking::Mutex`] (the OS-thread-parking mutex): this type
//! reaches bare-metal `no_std + alloc` where neither of those can.
//!
//! Unlike [`crate::sync::blocking`]'s futex tier, `lock()` here YIELDS the task
//! (returns `Pending` and registers a `Waker`) instead of parking a thread —
//! pure atomics + wakers, no futex, no OS wait primitive. It therefore
//! reaches every tier prime's executor can drive, including bare-metal
//! `no_std + alloc`. This is the "async gate" serializer preferred in async
//! context.
//!
//! ## Protocol
//!
//! A single `AtomicBool` is the locked flag. Waiters live in a lock-free MPSC
//! [`SegQueue<Waker>`](crossbeam_queue::SegQueue).
//!
//! `lock()` poll:
//! 1. fast path — CAS `false -> true` (Acquire). Success returns the guard.
//! 2. contended — push `waker.clone()`, then RE-CHECK the CAS
//!    (register-then-recheck, the same window-closing idiom as
//!    `proxima_core::park`). If the recheck wins, return the guard (a stale
//!    waker may remain queued; harmless — see the wake site). Else `Pending`.
//!
//! Guard drop / unlock:
//! 1. store `false` (Release).
//! 2. drain the queue and wake EVERY waiter.
//!
//! ## Two lost-wakeup windows, both closed
//!
//! **Window 1 — stale waker (fixed by wake-ALL).** A waiter that acquired via
//! the step-2 recheck leaves a stale waker queued. Waking a single waiter could
//! pop that stale entry (a no-op) and strand a real waiter forever. Draining and
//! waking everyone makes stale wakes harmless no-ops and leaves no real waiter
//! un-woken; every woken task re-polls and re-CASes, exactly one wins, the rest
//! re-register. Fine for a low-contention control-plane lock.
//!
//! **Window 2 — Dekker store-load race (fixed by `SeqCst` fences).** The waiter
//! does `push(queue); load(flag)`; the releaser does `store(flag); read(queue)`.
//! These are stores and loads on two DIFFERENT locations, so plain
//! Release/Acquire (and real x86 TSO) permit BOTH the releaser's queue-read to
//! miss the push AND the waiter's flag-load to miss the release — a double-miss
//! that strands the waiter. A `SeqCst` fence between the write and the read on
//! each side forbids it: the two fences are totally ordered, so whichever fence
//! is first, the other side observes its write (releaser drains the push, or
//! waiter's recheck sees the release). At least one always fires.

use core::cell::UnsafeCell;
use core::future::Future;
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, Ordering, fence};
use core::task::{Context, Poll, Waker};

use crossbeam_queue::SegQueue;

/// An async mutex that suspends the task instead of blocking the thread.
///
/// `Send + Sync where T: Send` — access is fully serialized, so `T` need not be
/// `Sync` for the mutex to be shared across tasks/threads.
pub struct AsyncMutex<T> {
    locked: AtomicBool,
    waiters: SegQueue<Waker>,
    value: UnsafeCell<T>,
}

// serialized access hands out `&mut T` one holder at a time, so `T: Send` alone
// makes the mutex both movable and shareable across threads.
unsafe impl<T: Send> Send for AsyncMutex<T> {}
unsafe impl<T: Send> Sync for AsyncMutex<T> {}

impl<T> AsyncMutex<T> {
    /// A fresh, unlocked mutex. `const` so it can back a `static`.
    #[must_use]
    pub const fn new(value: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
            waiters: SegQueue::new(),
            value: UnsafeCell::new(value),
        }
    }

    /// Acquire the lock, yielding to the executor while it is held elsewhere.
    pub async fn lock(&self) -> AsyncMutexGuard<'_, T> {
        Lock { mutex: self }.await
    }

    /// Acquire without waiting: `Some(guard)` if free, `None` if held.
    #[must_use]
    pub fn try_lock(&self) -> Option<AsyncMutexGuard<'_, T>> {
        if self.acquire() {
            Some(AsyncMutexGuard {
                mutex: self,
                _access: PhantomData,
            })
        } else {
            None
        }
    }

    fn acquire(&self) -> bool {
        self.locked
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }
}

/// The `lock()` future. Registers its waker on the first contended poll and
/// re-checks before yielding, so a release racing the registration cannot be
/// lost.
struct Lock<'mutex, T> {
    mutex: &'mutex AsyncMutex<T>,
}

impl<'mutex, T> Future for Lock<'mutex, T> {
    type Output = AsyncMutexGuard<'mutex, T>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let mutex = self.mutex;
        if mutex.acquire() {
            return Poll::Ready(AsyncMutexGuard {
                mutex,
                _access: PhantomData,
            });
        }
        // register, then recheck: closes the release-between-check-and-park window.
        mutex.waiters.push(context.waker().clone());
        // pairs with the releaser's fence; forbids the two-location store-load
        // double-miss (window 2 in the module docs).
        fence(Ordering::SeqCst);
        if mutex.acquire() {
            return Poll::Ready(AsyncMutexGuard {
                mutex,
                _access: PhantomData,
            });
        }
        Poll::Pending
    }
}

/// RAII guard; releases the lock and wakes waiters on drop. Deref/DerefMut to
/// the protected value.
pub struct AsyncMutexGuard<'mutex, T> {
    mutex: &'mutex AsyncMutex<T>,
    // carries `&mut T`'s auto-trait bounds onto the guard: without it the guard
    // would be `Sync` from `T: Send` alone, yet `Deref` hands out `&T` (needs
    // `T: Sync` to share). `&mut T` makes guard `Sync` iff `T: Sync` — sound.
    _access: PhantomData<&'mutex mut T>,
}

impl<T> Deref for AsyncMutexGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // the locked flag guarantees this guard is the sole live accessor.
        unsafe { &*self.mutex.value.get() }
    }
}

impl<T> DerefMut for AsyncMutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.mutex.value.get() }
    }
}

impl<T> Drop for AsyncMutexGuard<'_, T> {
    fn drop(&mut self) {
        self.mutex.locked.store(false, Ordering::Release);
        // pairs with the waiter's fence; guarantees this drain observes any push
        // whose recheck did not observe the release above (window 2).
        fence(Ordering::SeqCst);
        // why: wake ALL, not one — a recheck-acquirer leaves a stale waker in the
        // queue; a wake-one could pop that no-op and strand a real waiter. Stale
        // wakes are harmless; every woken task re-polls and re-CASes.
        while let Some(waker) = self.mutex.waiters.pop() {
            waker.wake();
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    extern crate std;

    use super::*;
    use core::task::{RawWaker, RawWakerVTable};
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::vec::Vec;

    use futures::executor::{ThreadPool, block_on};
    use futures::task::SpawnExt;

    // N futures on a shared thread pool each add M under the lock; a lost wakeup
    // or a torn critical section shows up as a final sum below N*M.
    #[test]
    fn contended_counter_sums_exactly() {
        const TASKS: usize = 16;
        const ADDS: usize = 2_000;

        let pool = ThreadPool::builder()
            .pool_size(4)
            .create()
            .expect("thread pool");
        let mutex = Arc::new(AsyncMutex::new(0usize));

        let handles: Vec<_> = (0..TASKS)
            .map(|_| {
                let mutex = Arc::clone(&mutex);
                pool.spawn_with_handle(async move {
                    for _ in 0..ADDS {
                        *mutex.lock().await += 1;
                    }
                })
                .expect("spawn task")
            })
            .collect();

        for handle in handles {
            block_on(handle);
        }

        assert_eq!(block_on(async { *mutex.lock().await }), TASKS * ADDS);
    }

    #[test]
    fn try_lock_excludes_while_held() {
        let mutex = AsyncMutex::new(7u32);
        let guard = mutex.try_lock().expect("free lock acquires");
        assert!(mutex.try_lock().is_none());
        drop(guard);
        assert!(mutex.try_lock().is_some());
    }

    // a Waker that just counts wakes, so we can drive the register-then-recheck
    // race deterministically on one thread with no executor and no sleeps.
    fn counting_waker(count: &Arc<AtomicUsize>) -> Waker {
        fn clone(pointer: *const ()) -> RawWaker {
            let count = unsafe { Arc::from_raw(pointer.cast::<AtomicUsize>()) };
            let cloned = Arc::clone(&count);
            let _ = Arc::into_raw(count);
            RawWaker::new(Arc::into_raw(cloned).cast(), &VTABLE)
        }
        fn wake(pointer: *const ()) {
            let count = unsafe { Arc::from_raw(pointer.cast::<AtomicUsize>()) };
            count.fetch_add(1, Ordering::SeqCst);
        }
        fn wake_by_ref(pointer: *const ()) {
            let count = unsafe { Arc::from_raw(pointer.cast::<AtomicUsize>()) };
            count.fetch_add(1, Ordering::SeqCst);
            let _ = Arc::into_raw(count);
        }
        fn drop_waker(pointer: *const ()) {
            drop(unsafe { Arc::from_raw(pointer.cast::<AtomicUsize>()) });
        }
        static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake_by_ref, drop_waker);

        let raw = RawWaker::new(Arc::into_raw(Arc::clone(count)).cast(), &VTABLE);
        unsafe { Waker::from_raw(raw) }
    }

    // forces the interleaving the protocol must survive: a waiter registers and
    // yields (Pending) while the lock is held, the holder releases, and the
    // recheck path must wake the waiter and let its next poll acquire — no hang.
    #[test]
    fn register_then_recheck_race_wakes_and_acquires() {
        let mutex = AsyncMutex::new(0u32);
        let holder = mutex.try_lock().expect("first acquire");

        let count = Arc::new(AtomicUsize::new(0));
        let waker = counting_waker(&count);
        let mut context = Context::from_waker(&waker);

        let mut waiter = Box::pin(Lock { mutex: &mutex });
        assert!(
            waiter.as_mut().poll(&mut context).is_pending(),
            "waiter yields while lock is held"
        );
        assert_eq!(count.load(Ordering::SeqCst), 0, "no premature wake");

        drop(holder);
        assert_eq!(count.load(Ordering::SeqCst), 1, "release wakes the waiter");

        match waiter.as_mut().poll(&mut context) {
            Poll::Ready(mut guard) => {
                *guard += 1;
                assert_eq!(*guard, 1);
            }
            Poll::Pending => panic!("waiter must acquire after release"),
        }
    }
}
