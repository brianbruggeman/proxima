//! R7 of the runtime-shaped initiative: `RuntimeFactory` impl for
//! `PrimeRuntime` plus the `*Like` adapters for prime's per-session
//! primitives.
//!
//! `Mutex`/`Notify`/`Sleep` are backed by `proxima-primitives`'
//! native, tokio-free async-gate primitives (`AsyncMutex`, `Notify`,
//! `proxima_core::time::Sleep`) — available unconditionally, no
//! `prime-tokio-compat` needed. `JoinSet` has no native cross-core
//! spawn-with-a-handle equivalent yet (`proxima_runtime::Runtime::
//! spawn_on_core` is fire-and-forget: no handle, no join, no abort);
//! the tokio-backed `PrimeJoinSet`, and the `RuntimeFactory` impl
//! that needs it (the trait requires all four associated types), stay
//! behind the `prime-tokio-compat` feature until a native `JoinSet`
//! lands (`docs/pipe-to-metal/edges.md`, "prime-tokio-feature-split
//! (task #8 remainder)" — the native form is a separate research-rigor
//! task, not built here). Nothing outside this module's own tests
//! consumes `RuntimeFactory for PrimeRuntime` today, so this gap has
//! zero blast radius on the default (tokio-free) build.

#![cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]

use std::ops::{Deref, DerefMut};
#[cfg(any(test, feature = "prime-tokio-compat"))]
use std::time::Duration;

use proxima_primitives::sync::{AsyncMutex, AsyncMutexGuard, Notify};
use proxima_runtime::{MutexLike, NotifyLike};

#[cfg(feature = "prime-tokio-compat")]
use std::future::Future;

#[cfg(feature = "prime-tokio-compat")]
use proxima_runtime::{JoinError, JoinSetLike, RuntimeFactory};

use crate::os::runtime::PrimeRuntime;

/// Prime-backed [`MutexLike`] adapter. Wraps `proxima_primitives::sync::
/// AsyncMutex<T>` — a waker-based async gate mutex (principle 21):
/// suspends the task instead of parking a thread, no tokio, reaches
/// every tier prime's executor can drive.
pub struct PrimeMutex<T>(pub AsyncMutex<T>);

/// Prime-backed lock guard exposing `DerefMut<Target = T>` per the
/// `MutexLike::Guard` contract.
pub struct PrimeMutexGuard<'guard, T>(pub AsyncMutexGuard<'guard, T>);

impl<T> Deref for PrimeMutexGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> DerefMut for PrimeMutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T: Send + 'static> MutexLike<T> for PrimeMutex<T> {
    type Guard<'guard> = PrimeMutexGuard<'guard, T>;

    async fn lock(&self) -> Self::Guard<'_> {
        PrimeMutexGuard(self.0.lock().await)
    }
}

/// Prime-backed [`NotifyLike`] adapter. Wraps `proxima_primitives::sync::
/// Notify` (`event_listener`-backed; `notify_one`/`notified` semantics
/// match `tokio::sync::Notify` byte-for-byte), tokio-free.
pub struct PrimeNotify(pub Notify);

impl NotifyLike for PrimeNotify {
    fn notify_one(&self) {
        self.0.notify_one();
    }

    async fn notified(&self) {
        self.0.notified().await;
    }
}

/// Sleep future backed by `proxima_core::time::Sleep` — a concrete
/// `Unpin` struct, so unlike the `!Unpin` `tokio::time::Sleep` this
/// replaces, no `Box::pin` is needed (a box-free win, principle 20).
pub type PrimeSleep = proxima_core::time::Sleep;

/// Prime-backed [`JoinSetLike`] adapter. Wraps `tokio::task::JoinSet<T>`.
/// Gated behind `prime-tokio-compat`: prime has no native cross-core
/// spawn-with-a-handle primitive today, so this is the only
/// `JoinSetLike` implementation available until one lands.
#[cfg(feature = "prime-tokio-compat")]
pub struct PrimeJoinSet<T>(pub tokio::task::JoinSet<T>);

#[cfg(feature = "prime-tokio-compat")]
impl<T: Send + 'static> JoinSetLike<T> for PrimeJoinSet<T> {
    fn spawn<F>(&mut self, future: F)
    where
        F: Future<Output = T> + Send + 'static,
    {
        self.0.spawn(future);
    }

    async fn join_next(&mut self) -> Option<Result<T, JoinError>> {
        self.0.join_next().await.map(|result| {
            result.map_err(|err| {
                if err.is_cancelled() {
                    JoinError::Cancelled
                } else {
                    JoinError::Panicked
                }
            })
        })
    }

    fn abort_all(&mut self) {
        self.0.abort_all();
    }
}

/// `RuntimeFactory` requires a `JoinSet` associated type alongside
/// `Mutex`/`Notify`/`Sleep`; without a native one, the whole impl
/// stays behind `prime-tokio-compat` alongside `PrimeJoinSet` above.
/// `Mutex`/`Notify`/`Sleep` here are the same tokio-free types exposed
/// unconditionally above — `prime-tokio-compat` only adds `JoinSet`,
/// it does not change how the other three work.
#[cfg(feature = "prime-tokio-compat")]
impl RuntimeFactory for PrimeRuntime {
    type Mutex<T: Send + 'static> = PrimeMutex<T>;
    type Notify = PrimeNotify;
    type JoinSet<T: Send + 'static> = PrimeJoinSet<T>;
    type Sleep = PrimeSleep;

    fn new_mutex<T: Send + 'static>(value: T) -> Self::Mutex<T> {
        PrimeMutex(AsyncMutex::new(value))
    }

    fn new_notify() -> Self::Notify {
        PrimeNotify(Notify::new())
    }

    fn new_join_set<T: Send + 'static>() -> Self::JoinSet<T> {
        PrimeJoinSet(tokio::task::JoinSet::new())
    }

    fn sleep(duration: Duration) -> Self::Sleep {
        proxima_core::time::sleep(duration)
    }
}

// ---------------------------------------------------------------------------
// R8: prime-pinned (non-Send) primitives via LocalRuntimeFactory.
// ---------------------------------------------------------------------------

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::rc::Rc;
use std::task::Waker;

use proxima_runtime::{LocalMutexLike, LocalNotifyLike, LocalRuntimeFactory};

/// Inner state shared between every clone of [`PrimeLocalMutex`] / guard.
struct LocalMutexInner<T> {
    /// Locked flag — `Cell<bool>` because we only access this from one
    /// core. No atomic operations.
    locked: Cell<bool>,
    /// Wait queue of wakers parked while the lock is held. `RefCell`
    /// because we only borrow it during the lock/unlock window.
    waiters: RefCell<VecDeque<Waker>>,
    /// The protected value.
    value: RefCell<T>,
}

/// R8 prime-pinned non-Send async mutex. Uses `Rc<RefCell<T>>` shape
/// without any atomic operations on the lock state — valid because
/// every clone stays on the same core (`!Send` guard contract).
///
/// Whether this is measurably faster than [`PrimeMutex`] (tokio-backed)
/// on the uncontended fast path is a question for the bench harness.
/// The API exists so callers can express the single-core constraint
/// at the type level AND so the option can be measured.
pub struct PrimeLocalMutex<T>(Rc<LocalMutexInner<T>>);

/// Guard returned by [`PrimeLocalMutex::lock`]. Holds the lock until
/// dropped; `!Send` so it cannot escape the core that acquired it.
pub struct PrimeLocalMutexGuard<'guard, T> {
    inner: &'guard LocalMutexInner<T>,
}

impl<T> core::ops::Deref for PrimeLocalMutexGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: while a guard exists, `locked` is true so no other
        // task on this core can borrow `value`. the RefCell borrow is
        // single-borrow because we only ever lend out one guard at a
        // time (the lock flag enforces it).
        unsafe { &*self.inner.value.as_ptr() }
    }
}

impl<T> core::ops::DerefMut for PrimeLocalMutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: same as deref — exclusive access guaranteed by
        // `locked == true`.
        unsafe { &mut *self.inner.value.as_ptr() }
    }
}

impl<T> Drop for PrimeLocalMutexGuard<'_, T> {
    fn drop(&mut self) {
        self.inner.locked.set(false);
        if let Some(waker) = self.inner.waiters.borrow_mut().pop_front() {
            waker.wake();
        }
    }
}

impl<T: 'static> LocalMutexLike<T> for PrimeLocalMutex<T> {
    type Guard<'guard>
        = PrimeLocalMutexGuard<'guard, T>
    where
        Self: 'guard;

    fn lock(&self) -> impl core::future::Future<Output = Self::Guard<'_>> {
        LocalMutexLockFuture { mutex: self }
    }
}

/// Future returned by [`PrimeLocalMutex::lock`]. Polls the lock flag
/// and parks via the per-mutex waker queue when the lock is taken.
struct LocalMutexLockFuture<'mutex, T> {
    mutex: &'mutex PrimeLocalMutex<T>,
}

impl<'mutex, T: 'static> core::future::Future for LocalMutexLockFuture<'mutex, T> {
    type Output = PrimeLocalMutexGuard<'mutex, T>;
    fn poll(
        self: core::pin::Pin<&mut Self>,
        context: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Self::Output> {
        let inner = &*self.mutex.0;
        if !inner.locked.get() {
            inner.locked.set(true);
            core::task::Poll::Ready(PrimeLocalMutexGuard { inner })
        } else {
            inner
                .waiters
                .borrow_mut()
                .push_back(context.waker().clone());
            core::task::Poll::Pending
        }
    }
}

/// Per-waiter entry. The `woken` flag is the signal `notify_one`
/// flips so the waiter's next poll can distinguish "woken for real"
/// from a spurious wake.
struct WaiterEntry {
    waker: Waker,
    woken: Rc<Cell<bool>>,
}

/// R8 prime-pinned non-Send notify. `Cell<usize>` permit counter +
/// `RefCell<VecDeque<WaiterEntry>>` waiter queue. No atomic operations.
pub struct PrimeLocalNotify {
    permits: Cell<usize>,
    waiters: RefCell<VecDeque<WaiterEntry>>,
}

impl LocalNotifyLike for PrimeLocalNotify {
    fn notify_one(&self) {
        if let Some(entry) = self.waiters.borrow_mut().pop_front() {
            entry.woken.set(true);
            entry.waker.wake();
        } else {
            self.permits.set(self.permits.get().saturating_add(1));
        }
    }

    fn notified(&self) -> impl core::future::Future<Output = ()> + '_ {
        LocalNotifiedFuture {
            notify: self,
            woken: None,
        }
    }
}

struct LocalNotifiedFuture<'notify> {
    notify: &'notify PrimeLocalNotify,
    /// `None` = not yet parked. `Some(flag)` = parked; the flag is
    /// flipped by `notify_one` when the waiter is popped + woken.
    woken: Option<Rc<Cell<bool>>>,
}

impl<'notify> core::future::Future for LocalNotifiedFuture<'notify> {
    type Output = ();
    fn poll(
        mut self: core::pin::Pin<&mut Self>,
        context: &mut core::task::Context<'_>,
    ) -> core::task::Poll<()> {
        // already parked — check the shared flag for a real wake.
        if let Some(flag) = self.woken.as_ref() {
            if flag.get() {
                return core::task::Poll::Ready(());
            }
            // spurious wake — stay parked; the existing entry in the
            // queue still holds the waker.
            return core::task::Poll::Pending;
        }
        // not yet parked — consume a permit if available.
        let pending = self.notify.permits.get();
        if pending > 0 {
            self.notify.permits.set(pending - 1);
            return core::task::Poll::Ready(());
        }
        // park: register a shared woken flag + the waker.
        let woken = Rc::new(Cell::new(false));
        self.notify.waiters.borrow_mut().push_back(WaiterEntry {
            waker: context.waker().clone(),
            woken: Rc::clone(&woken),
        });
        self.woken = Some(woken);
        core::task::Poll::Pending
    }
}

impl LocalRuntimeFactory for PrimeRuntime {
    type LocalMutex<T: 'static> = PrimeLocalMutex<T>;
    type LocalNotify = PrimeLocalNotify;

    fn new_local_mutex<T: 'static>(value: T) -> Self::LocalMutex<T> {
        PrimeLocalMutex(Rc::new(LocalMutexInner {
            locked: Cell::new(false),
            waiters: RefCell::new(VecDeque::new()),
            value: RefCell::new(value),
        }))
    }

    fn new_local_notify() -> Self::LocalNotify {
        PrimeLocalNotify {
            permits: Cell::new(0),
            waiters: RefCell::new(VecDeque::new()),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use core::task::{Context, Poll, Waker};

    use super::*;

    // unconditional — proves the native Mutex/Notify/Sleep swap directly,
    // without going through `RuntimeFactory` (which stays behind
    // `prime-tokio-compat` until a native JoinSet exists, see this file's
    // module docs).

    #[proxima::test(runtime = "tokio")]
    async fn prime_mutex_round_trips_a_value() {
        let mutex = PrimeMutex(AsyncMutex::new(42i32));
        {
            let mut guard = mutex.lock().await;
            *guard += 1;
        }
        let guard = mutex.lock().await;
        assert_eq!(*guard, 43);
    }

    // manually driven (no executor, no sleep, no spawned task) so this
    // stays tokio-free and deterministic — the same style already used by
    // proxima-primitives' own `AsyncMutex` tests
    // (`register_then_recheck_race_wakes_and_acquires`).
    #[test]
    fn prime_notify_wakes_a_parked_waiter() {
        let notify = PrimeNotify(Notify::new());
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        let mut waiting = Box::pin(notify.notified());
        assert!(
            waiting.as_mut().poll(&mut context).is_pending(),
            "no permit pending yet — must park"
        );
        notify.notify_one();
        assert!(matches!(
            waiting.as_mut().poll(&mut context),
            Poll::Ready(())
        ));
    }

    #[proxima::test(runtime = "tokio")]
    async fn prime_sleep_resolves_after_duration() {
        let start = std::time::Instant::now();
        proxima_core::time::sleep(Duration::from_millis(10)).await;
        assert!(start.elapsed() >= Duration::from_millis(8));
    }

    // `prime-tokio-compat`-only — proves the `RuntimeFactory` impl (which
    // needs the tokio-backed `PrimeJoinSet` to satisfy all four associated
    // types) still delegates `Mutex`/`Notify`/`Sleep` to the same native
    // types the unconditional tests above exercise directly, and that
    // `PrimeJoinSet` itself still round-trips through tokio.
    #[cfg(feature = "prime-tokio-compat")]
    #[proxima::test(runtime = "tokio")]
    async fn prime_runtime_factory_delegates_to_native_trio_plus_tokio_joinset() {
        let mutex = PrimeRuntime::new_mutex(42i32);
        {
            let mut guard = mutex.lock().await;
            *guard += 1;
        }
        assert_eq!(*mutex.lock().await, 43);

        let notify = std::sync::Arc::new(PrimeRuntime::new_notify());
        let waker = std::sync::Arc::clone(&notify);
        waker.notify_one();
        notify.notified().await;

        let mut set: PrimeJoinSet<u32> = PrimeRuntime::new_join_set();
        set.spawn(async { 10u32 });
        set.spawn(async { 20u32 });
        let mut total = 0u32;
        while let Some(result) = set.join_next().await {
            total += result.unwrap();
        }
        assert_eq!(total, 30);

        let start = std::time::Instant::now();
        PrimeRuntime::sleep(Duration::from_millis(10)).await;
        assert!(start.elapsed() >= Duration::from_millis(8));
    }

    // ---- R8 — LocalRuntimeFactory parity tests ----
    //
    // `LocalRuntimeFactory` itself (PrimeLocalMutex/PrimeLocalNotify above)
    // is tokio-free; these tests use `#[tokio::test]` purely as a
    // convenient single-threaded async test harness (`LocalSet` +
    // `spawn_local` for the two-waiter race), so they stay gated behind
    // `prime-tokio-compat` for the `tokio` dependency the harness itself
    // needs, not because the primitive under test needs tokio.

    #[cfg(feature = "prime-tokio-compat")]
    #[tokio::test(flavor = "current_thread")]
    async fn local_mutex_round_trips_a_value() {
        let mutex = PrimeRuntime::new_local_mutex(0i32);
        {
            let mut guard = mutex.lock().await;
            *guard = 42;
        }
        let guard = mutex.lock().await;
        assert_eq!(*guard, 42);
    }

    #[cfg(feature = "prime-tokio-compat")]
    #[tokio::test(flavor = "current_thread")]
    async fn local_mutex_serializes_two_concurrent_lockers() {
        // two tasks racing on the same lock should observe a
        // serialized "1 then 2" increment from a 0 starting state —
        // proving lock acquisition is mutually exclusive.
        let mutex = std::rc::Rc::new(PrimeRuntime::new_local_mutex(0u32));
        let one = std::rc::Rc::clone(&mutex);
        let two = std::rc::Rc::clone(&mutex);
        let inspect = std::rc::Rc::clone(&mutex);
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async move {
                let h1 = tokio::task::spawn_local(async move {
                    let mut guard = one.lock().await;
                    *guard += 1;
                });
                let h2 = tokio::task::spawn_local(async move {
                    let mut guard = two.lock().await;
                    *guard += 1;
                });
                h1.await.unwrap();
                h2.await.unwrap();
                assert_eq!(*inspect.lock().await, 2);
            })
            .await;
    }

    #[cfg(feature = "prime-tokio-compat")]
    #[tokio::test(flavor = "current_thread")]
    async fn local_notify_wake_after_park_resolves() {
        let notify = std::rc::Rc::new(PrimeRuntime::new_local_notify());
        let waker = std::rc::Rc::clone(&notify);
        let local = tokio::task::LocalSet::new();
        let result = local
            .run_until(async move {
                let waiter = tokio::task::spawn_local(async move {
                    waker.notified().await;
                    true
                });
                tokio::task::yield_now().await;
                notify.notify_one();
                waiter.await.unwrap()
            })
            .await;
        assert!(result);
    }

    #[cfg(feature = "prime-tokio-compat")]
    #[tokio::test(flavor = "current_thread")]
    async fn local_notify_pending_permit_resolves_immediately() {
        let notify = PrimeRuntime::new_local_notify();
        notify.notify_one();
        // permit pending — the first notified() should resolve
        // without parking.
        notify.notified().await;
    }
}
