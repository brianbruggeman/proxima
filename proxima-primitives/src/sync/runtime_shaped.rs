//! R3 of the runtime-shaped initiative: `Mutex<T, R: RuntimeFactory>`
//! generic over the runtime that supplies the underlying lock.
//!
//! Gated behind the `runtime-shaped-mutex` feature so existing
//! `use proxima_primitives::sync::Mutex;` call sites continue to compile against
//! the workspace-default (async-lock-backed) `Mutex<T>` in
//! `proxima_primitives::sync::mutex`. The generic variant is opt-in for
//! consumers that want to thread the runtime selection through their
//! types (`Mutex<T>` defaults to `TokioPerCoreRuntime`).
//!
//! Future rows (R4-R7) add the same shape for RwLock / Notify /
//! mpsc / JoinSet / Sleep; this commit establishes the pattern.

use proxima_runtime::tokio::TokioPerCoreRuntime;
use proxima_runtime::{MutexLike, NotifyLike, RuntimeFactory};

/// Runtime-parameterized async mutex.
///
/// `T` is the protected value; `R` is the runtime that supplies the
/// underlying lock primitive. Default `R = TokioPerCoreRuntime` so
/// `Mutex::new(value)` works on the tokio path without any
/// annotation. Prime users opt in with
/// `Mutex::<T, PrimeRuntime>::new(value)`.
pub struct Mutex<T: Send + 'static, R: RuntimeFactory = TokioPerCoreRuntime> {
    inner: R::Mutex<T>,
}

impl<T: Send + 'static, R: RuntimeFactory> Mutex<T, R> {
    /// Construct a new mutex protecting `value`. Delegates to
    /// `R::new_mutex` so the underlying primitive is whatever the
    /// runtime ships (tokio: `tokio::sync::Mutex`; prime: TBD).
    #[must_use]
    pub fn new(value: T) -> Self {
        Self {
            inner: <R as RuntimeFactory>::new_mutex(value),
        }
    }

    /// Acquire the lock. The returned guard exposes
    /// `DerefMut<Target = T>` until dropped.
    pub async fn lock(&self) -> <R::Mutex<T> as MutexLike<T>>::Guard<'_> {
        self.inner.lock().await
    }
}

/// Runtime-parameterized notification primitive (R4).
///
/// `notify_one()` wakes one parked waiter; `notified().await` parks
/// until a signal arrives. Default `R = TokioPerCoreRuntime` so
/// `Notify::new()` works without annotation; prime users opt in
/// with `Notify::<PrimeRuntime>::new()`.
pub struct Notify<R: RuntimeFactory = TokioPerCoreRuntime> {
    inner: R::Notify,
}

impl<R: RuntimeFactory> Notify<R> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: <R as RuntimeFactory>::new_notify(),
        }
    }

    /// Wake one waiter, or store a pending notification if no waiter
    /// is currently parked.
    pub fn notify_one(&self) {
        self.inner.notify_one();
    }

    /// Park until the next signal arrives.
    pub async fn notified(&self) {
        self.inner.notified().await;
    }
}

impl<R: RuntimeFactory> Default for Notify<R> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // implicit-default test: `Mutex::new(...)` without an `R`
    // annotation must compile against the TokioPerCoreRuntime
    // default. exercise the lock cycle to make sure the guard
    // surface threads through.
    #[proxima::test(runtime = "tokio")]
    async fn default_runtime_resolves_to_tokio_and_locks() {
        let mutex: Mutex<i32> = Mutex::new(0);
        {
            let mut guard = mutex.lock().await;
            *guard = 42;
        }
        let guard = mutex.lock().await;
        assert_eq!(*guard, 42);
    }

    #[proxima::test(runtime = "tokio")]
    async fn explicit_tokio_annotation_compiles_and_locks() {
        let mutex: Mutex<String, TokioPerCoreRuntime> = Mutex::new(String::new());
        let mut guard = mutex.lock().await;
        guard.push_str("hi");
        assert_eq!(*guard, "hi");
    }

    #[proxima::test(runtime = "tokio")]
    async fn default_runtime_notify_wakes_one_waiter() {
        let notify: std::sync::Arc<Notify> = std::sync::Arc::new(Notify::new());
        let waker = std::sync::Arc::clone(&notify);
        let waiter = tokio::spawn(async move {
            waker.notified().await;
            true
        });
        // park the waiter
        tokio::task::yield_now().await;
        notify.notify_one();
        assert!(waiter.await.unwrap());
    }

    #[proxima::test(runtime = "tokio")]
    async fn explicit_tokio_annotation_notify_compiles() {
        let _notify: Notify<TokioPerCoreRuntime> = Notify::new();
    }
}
