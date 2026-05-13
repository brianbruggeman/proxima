//! R2 of the runtime-shaped initiative: `RuntimeFactory` impl for
//! `TokioPerCoreRuntime` plus the `*Like` adapters for tokio's
//! per-session primitives.
//!
//! Each adapter is a NEWTYPE around the tokio primitive so future
//! changes (e.g. instrumented variants) don't break the `R::Mutex<T>`
//! associated-type fingerprint that downstream generic code locks
//! onto.

use std::future::Future;
use std::ops::{Deref, DerefMut};
use std::pin::Pin;
use std::time::Duration;

use crate::{JoinError, JoinSetLike, MutexLike, NotifyLike, RuntimeFactory};

use super::TokioPerCoreRuntime;

/// Tokio-backed [`MutexLike`] adapter. Wraps `tokio::sync::Mutex<T>`.
pub struct TokioMutex<T>(pub tokio::sync::Mutex<T>);

/// Tokio-backed lock guard exposing `DerefMut<Target = T>` per the
/// `MutexLike::Guard` contract.
pub struct TokioMutexGuard<'guard, T>(pub tokio::sync::MutexGuard<'guard, T>);

impl<T> Deref for TokioMutexGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> DerefMut for TokioMutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T: Send + 'static> MutexLike<T> for TokioMutex<T> {
    type Guard<'guard> = TokioMutexGuard<'guard, T>;

    async fn lock(&self) -> Self::Guard<'_> {
        TokioMutexGuard(self.0.lock().await)
    }
}

/// Tokio-backed [`NotifyLike`] adapter. Wraps `tokio::sync::Notify`.
pub struct TokioNotify(pub tokio::sync::Notify);

impl NotifyLike for TokioNotify {
    fn notify_one(&self) {
        self.0.notify_one();
    }

    async fn notified(&self) {
        self.0.notified().await;
    }
}

/// Tokio-backed [`JoinSetLike`] adapter. Wraps `tokio::task::JoinSet<T>`.
pub struct TokioJoinSet<T>(pub tokio::task::JoinSet<T>);

impl<T: Send + 'static> JoinSetLike<T> for TokioJoinSet<T> {
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

/// Boxed sleep future — `tokio::time::Sleep` is `!Unpin`, so the
/// associated type for `RuntimeFactory::Sleep` becomes a pinned
/// boxed future to keep the trait surface uniform across runtimes.
pub type TokioSleep = Pin<Box<dyn Future<Output = ()> + Send>>;

impl RuntimeFactory for TokioPerCoreRuntime {
    type Mutex<T: Send + 'static> = TokioMutex<T>;
    type Notify = TokioNotify;
    type JoinSet<T: Send + 'static> = TokioJoinSet<T>;
    type Sleep = TokioSleep;

    fn new_mutex<T: Send + 'static>(value: T) -> Self::Mutex<T> {
        TokioMutex(tokio::sync::Mutex::new(value))
    }

    fn new_notify() -> Self::Notify {
        TokioNotify(tokio::sync::Notify::new())
    }

    fn new_join_set<T: Send + 'static>() -> Self::JoinSet<T> {
        TokioJoinSet(tokio::task::JoinSet::new())
    }

    fn sleep(duration: Duration) -> Self::Sleep {
        Box::pin(tokio::time::sleep(duration))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[proxima::test(runtime = "tokio")]
    async fn tokio_mutex_round_trips_a_value() {
        let mutex = TokioPerCoreRuntime::new_mutex(42i32);
        {
            let mut guard = mutex.lock().await;
            *guard += 1;
        }
        let guard = mutex.lock().await;
        assert_eq!(*guard, 43);
    }

    #[proxima::test(runtime = "tokio")]
    async fn tokio_notify_wakes_a_waiter() {
        let notify = std::sync::Arc::new(TokioPerCoreRuntime::new_notify());
        let waker = std::sync::Arc::clone(&notify);
        let waiter = tokio::spawn(async move {
            waker.notified().await;
            true
        });
        // give the waiter a moment to park
        tokio::task::yield_now().await;
        notify.notify_one();
        assert!(waiter.await.unwrap());
    }

    #[proxima::test(runtime = "tokio")]
    async fn tokio_join_set_drains_to_completion() {
        let mut set: TokioJoinSet<u32> = TokioPerCoreRuntime::new_join_set();
        set.spawn(async { 1u32 });
        set.spawn(async { 2u32 });
        set.spawn(async { 3u32 });
        let mut total = 0u32;
        while let Some(result) = set.join_next().await {
            total += result.unwrap();
        }
        assert_eq!(total, 6);
    }

    #[proxima::test(runtime = "tokio")]
    async fn tokio_join_set_abort_surfaces_cancelled() {
        let mut set: TokioJoinSet<u32> = TokioPerCoreRuntime::new_join_set();
        set.spawn(async {
            tokio::time::sleep(Duration::from_secs(60)).await;
            42u32
        });
        set.abort_all();
        match set.join_next().await {
            Some(Err(JoinError::Cancelled)) => {}
            other => panic!("expected Cancelled, got {other:?}"),
        }
    }

    #[proxima::test(runtime = "tokio")]
    async fn tokio_sleep_resolves_after_duration() {
        let start = std::time::Instant::now();
        TokioPerCoreRuntime::sleep(Duration::from_millis(10)).await;
        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_millis(8));
    }
}
