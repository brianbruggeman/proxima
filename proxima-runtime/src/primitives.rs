//! `RuntimeFactory` sibling trait + the `*Like` primitive shapes.
//!
//! Design constraint: [`Runtime`] stays dyn-compatible because the
//! runtime threads `&dyn Runtime` through per-core dispatch, and GAT
//! associated-type factories on it would break that workspace-wide.
//! Resolution: a SIBLING trait `RuntimeFactory: Runtime` carries the
//! typed primitive factories. `TokioPerCoreRuntime` and `PrimeRuntime`
//! impl BOTH; consumers that need typed Mutex / Notify / JoinSet /
//! Sleep bound on `R: RuntimeFactory` (which implies `Runtime`).

use core::future::Future;
use core::ops::DerefMut;
#[cfg(feature = "std")]
use std::time::Duration;

#[cfg(all(feature = "alloc", feature = "std"))]
use crate::Runtime;

/// Async mutex shape. The guard type is borrowed so callers can hold
/// the lock across an `await` without paying a clone. Implementations
/// supply the actual lock primitive — `tokio::sync::Mutex<T>` for the
/// tokio impl, a per-core single-writer lock for prime.
pub trait MutexLike<T: Send + 'static>: Send + Sync {
    type Guard<'guard>: DerefMut<Target = T> + Send
    where
        Self: 'guard;

    /// Acquire the lock. The returned future resolves when the guard
    /// is available; the guard holds the lock until dropped.
    fn lock(&self) -> impl Future<Output = Self::Guard<'_>> + Send;
}

/// Notification primitive. `notify_one` wakes one waiter; subsequent
/// `notified().await` resumes when the next signal arrives.
pub trait NotifyLike: Send + Sync {
    /// Wake one waiter, or store a pending notification if no waiter
    /// is currently parked.
    fn notify_one(&self);

    /// Park until the next `notify_one` signal arrives. Consumes one
    /// pending notification atomically with parking.
    fn notified(&self) -> impl Future<Output = ()> + Send + '_;
}

/// A handle that yields a task's result on `await` and supports
/// abort. Matches `tokio::task::JoinSet`'s shape.
pub trait JoinSetLike<T>: Send {
    /// Spawn a future onto the set. The future is `Send + 'static`
    /// because it crosses task boundaries.
    fn spawn<F>(&mut self, future: F)
    where
        F: Future<Output = T> + Send + 'static;

    /// Resolve the next completed task, or `None` if the set is
    /// empty. `JoinError` wraps the panic / cancellation outcome.
    fn join_next(&mut self) -> impl Future<Output = Option<Result<T, JoinError>>> + Send + '_;

    /// Abort every spawned task. Subsequent `join_next` calls drain
    /// the cancellation outcomes.
    fn abort_all(&mut self);
}

/// Opaque join-failure reason. Wraps the runtime's native error type
/// (`tokio::task::JoinError` for the tokio impl).
#[derive(Debug)]
pub enum JoinError {
    /// Task panicked.
    Panicked,
    /// Task was cancelled (typically via `abort_all`).
    Cancelled,
}

/// Sibling extension of [`Runtime`]: typed factory methods for the
/// per-session sync / task / time primitives. Implemented by every
/// concrete runtime that ships in the workspace (`TokioPerCoreRuntime`,
/// `PrimeRuntime`). Consumers bound on `R: RuntimeFactory` when they
/// need typed primitives; consumers that only need to spawn stay on
/// `R: Runtime` and the trait stays dyn-compatible.
#[cfg(all(feature = "alloc", feature = "std"))]
pub trait RuntimeFactory: Runtime {
    type Mutex<T: Send + 'static>: MutexLike<T>;
    type Notify: NotifyLike;
    type JoinSet<T: Send + 'static>: JoinSetLike<T>;
    /// `Send` because the awaiter may move across thread boundaries on
    /// tokio's multi-thread runtime.
    type Sleep: Future<Output = ()> + Send;

    fn new_mutex<T: Send + 'static>(value: T) -> Self::Mutex<T>;
    fn new_notify() -> Self::Notify;
    fn new_join_set<T: Send + 'static>() -> Self::JoinSet<T>;
    fn sleep(duration: Duration) -> Self::Sleep;
}

/// Non-Send mutex shape for runtimes that pin tasks to a
/// single core. The guard is `!Send` so callers can't accidentally
/// move it across cores; in exchange the underlying primitive can
/// skip the atomic/notification dance a Send Mutex needs.
///
/// Implementations are typically `Rc<RefCell<T>>`-shaped plus a
/// per-core `VecDeque<Waker>` waiter queue. Not exposed on
/// `RuntimeFactory` because tokio's multi-thread runtime can't
/// honor the !Send guard contract — `LocalRuntimeFactory` is a
/// separate opt-in for prime-style per-core-pinned runtimes.
pub trait LocalMutexLike<T: 'static> {
    type Guard<'guard>: DerefMut<Target = T>
    where
        Self: 'guard;

    /// Acquire the lock. The returned future is `!Send` so it
    /// can only be polled by the core that spawned it.
    fn lock(&self) -> impl Future<Output = Self::Guard<'_>>;
}

/// Non-Send notify shape. Same single-core opt-in contract as
/// [`LocalMutexLike`].
pub trait LocalNotifyLike {
    fn notify_one(&self);
    fn notified(&self) -> impl Future<Output = ()> + '_;
}

/// Sibling extension of [`Runtime`] for runtimes that pin
/// tasks to a single core (prime's per-core executor, embedded
/// runtimes with no multi-thread executor at all). Provides
/// non-Send primitive factories that skip the atomic synchronization
/// a Send-shaped primitive needs.
///
/// `TokioPerCoreRuntime` does NOT impl this trait because its
/// multi-thread runtime can move tasks across cores; the !Send
/// guards would be a soundness bug. Prime opts in because its
/// `spawn_on_current_core` keeps the future pinned to one thread.
///
/// Perf claim: the Local primitives use `Rc<RefCell<T>>` shape
/// without any atomic operations on the lock state. **Whether this
/// is measurably faster than `RuntimeFactory::Mutex<T>` on the
/// uncontended fast path is a question the bench harness has to
/// answer**; the API exists so the option can be measured (and so
/// callers can express the single-core constraint at the type
/// level).
#[cfg(all(feature = "alloc", feature = "std"))]
pub trait LocalRuntimeFactory: Runtime {
    type LocalMutex<T: 'static>: LocalMutexLike<T>;
    type LocalNotify: LocalNotifyLike;

    fn new_local_mutex<T: 'static>(value: T) -> Self::LocalMutex<T>;
    fn new_local_notify() -> Self::LocalNotify;
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn join_error_is_debug_constructible() {
        // smoke-test the error enum so a future impl doesn't add a
        // variant that breaks Debug.
        let _ = format!("{:?}", JoinError::Panicked);
        let _ = format!("{:?}", JoinError::Cancelled);
    }
}
