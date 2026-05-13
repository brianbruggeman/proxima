//! `proxima::sync::Semaphore` — counted async semaphore, shape-
//! compatible with `tokio::sync::Semaphore`. Backed by
//! `async_lock::Semaphore` for permit counting, plus a local
//! `event_listener::Event` + `AtomicBool` for close-driven cancellation
//! (which async-lock doesn't expose).
//!
//! Acquire-with-close-race semantics: `acquire().await` races the
//! underlying permit acquire against a close-notification listener.
//! If the semaphore is closed before the permit lands, the future
//! resolves to `Err(AcquireError)`. Matches tokio's contract.
//!
//! `available_permits()` is tracked locally via an `AtomicUsize`
//! since async-lock doesn't expose its internal counter. The counter
//! decrements on `acquire`/`try_acquire` success and increments on
//! permit drop (via the custom `SemaphorePermit` wrapper).
//!
//! # Non-coverage
//!
//! - `acquire_many(n)` / `try_acquire_many(n)` — async-lock has no
//!   atomic N-permit primitive; rolling one requires careful
//!   retry-with-event-listener atomic logic to preserve the
//!   "all-or-nothing" guarantee. Out of scope for A2.b; a future
//!   consumer can promote.
//! - `acquire_owned()` — owned-permit variant for `Arc<Semaphore>`.
//!   The Arc-backed path exists on async-lock as `acquire_arc()`;
//!   shimming an `OwnedSemaphorePermit` over it is straightforward
//!   but no caller today.

use core::fmt;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use event_listener::Event;
use futures::FutureExt;

/// Counted async semaphore.
#[derive(Debug)]
pub struct Semaphore {
    inner: async_lock::Semaphore,
    closed: AtomicBool,
    close_event: Event,
    available: AtomicUsize,
}

impl Semaphore {
    /// Build a semaphore with `permits` initial permits.
    ///
    /// Under `--cfg loom`, `async_lock::Semaphore::new` and
    /// `event_listener::Event::new` are deliberately non-`const` (loom's
    /// model needs a runtime execution context). Only the `const`
    /// qualifier is affected; the constructed `Semaphore` is identical
    /// either way. `Semaphore` isn't part of the loom-tested
    /// Notify/watch protocol, but the crate is built as one unit under
    /// `--cfg loom`, so this split just keeps it compiling.
    #[cfg(not(loom))]
    #[must_use]
    pub const fn new(permits: usize) -> Self {
        Self {
            inner: async_lock::Semaphore::new(permits),
            closed: AtomicBool::new(false),
            close_event: Event::new(),
            available: AtomicUsize::new(permits),
        }
    }

    #[cfg(loom)]
    #[must_use]
    pub fn new(permits: usize) -> Self {
        Self {
            inner: async_lock::Semaphore::new(permits),
            closed: AtomicBool::new(false),
            close_event: Event::new(),
            available: AtomicUsize::new(permits),
        }
    }

    /// Acquire one permit, awaiting if necessary. Returns
    /// `Err(AcquireError)` if the semaphore is (or becomes) closed
    /// while waiting.
    pub async fn acquire(&self) -> Result<SemaphorePermit<'_>, AcquireError> {
        if self.is_closed() {
            return Err(AcquireError(()));
        }
        let listener = self.close_event.listen();
        let acquire_future = self.inner.acquire().fuse();
        let mut close_future = listener.fuse();
        futures::pin_mut!(acquire_future);
        futures::select_biased! {
            _ = close_future => Err(AcquireError(())),
            guard = acquire_future => {
                self.available.fetch_sub(1, Ordering::Release);
                Ok(SemaphorePermit { guard: Some(guard), owner: self })
            }
        }
    }

    /// Try to acquire one permit without waiting.
    pub fn try_acquire(&self) -> Result<SemaphorePermit<'_>, TryAcquireError> {
        if self.is_closed() {
            return Err(TryAcquireError::Closed);
        }
        match self.inner.try_acquire() {
            Some(guard) => {
                self.available.fetch_sub(1, Ordering::Release);
                Ok(SemaphorePermit {
                    guard: Some(guard),
                    owner: self,
                })
            }
            None => Err(TryAcquireError::NoPermits),
        }
    }

    /// Inject `n` additional permits into the pool.
    pub fn add_permits(&self, n: usize) {
        self.inner.add_permits(n);
        self.available.fetch_add(n, Ordering::Release);
    }

    /// Current available permit count. Snapshot only — may change
    /// concurrently before the caller observes the return value.
    #[must_use]
    pub fn available_permits(&self) -> usize {
        self.available.load(Ordering::Acquire)
    }

    /// Close the semaphore. Pending and subsequent `acquire`s return
    /// `Err(AcquireError)`. Idempotent.
    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.close_event.notify(usize::MAX);
    }

    /// `true` iff `close()` has been called.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }
}

/// Acquired permit. Releases back into the pool on drop.
#[derive(Debug)]
pub struct SemaphorePermit<'a> {
    guard: Option<async_lock::SemaphoreGuard<'a>>,
    owner: &'a Semaphore,
}

impl SemaphorePermit<'_> {
    /// Drop without releasing the permit. Equivalent to tokio's
    /// `SemaphorePermit::forget`. The available-permits counter is
    /// not incremented; the permit is "leaked," matching tokio.
    pub fn forget(mut self) {
        if let Some(guard) = self.guard.take() {
            guard.forget();
        }
    }
}

impl Drop for SemaphorePermit<'_> {
    fn drop(&mut self) {
        if self.guard.is_some() {
            self.owner.available.fetch_add(1, Ordering::Release);
        }
    }
}

/// Error returned by `acquire().await` when the semaphore is closed.
#[derive(Debug)]
pub struct AcquireError(());

impl fmt::Display for AcquireError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("semaphore closed")
    }
}

impl std::error::Error for AcquireError {}

/// Error returned by `try_acquire()`.
#[derive(Debug)]
pub enum TryAcquireError {
    /// Semaphore is closed.
    Closed,
    /// No permits currently available.
    NoPermits,
}

impl fmt::Display for TryAcquireError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Closed => formatter.write_str("semaphore closed"),
            Self::NoPermits => formatter.write_str("no permits available"),
        }
    }
}

impl std::error::Error for TryAcquireError {}

// Note: SemaphoreGuard / SemaphoreGuardArc re-exports stay available
// for callers that need the underlying async-lock types (e.g. when
// integrating with a third-party crate that takes async-lock guards).
pub use async_lock::{SemaphoreGuard, SemaphoreGuardArc};

// `#[proxima::test]` and inline `tokio::spawn` pull in the `proxima` /
// `tokio` dev-dependencies, which the loom build keeps out of the graph
// (see `[target.'cfg(not(loom))'.dev-dependencies]`); these tests are
// unrelated to the Notify/watch loom protocol anyway.
#[cfg(all(test, not(loom)))]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default,
        clippy::type_complexity,
        clippy::useless_vec,
        clippy::needless_range_loop,
        clippy::default_constructed_unit_structs
    )]
    use super::*;

    #[proxima::test]
    async fn acquire_returns_ok_when_permit_available() {
        let semaphore = Semaphore::new(2);
        assert_eq!(semaphore.available_permits(), 2);
        let permit = semaphore.acquire().await.expect("permit");
        assert_eq!(semaphore.available_permits(), 1);
        drop(permit);
        assert_eq!(semaphore.available_permits(), 2);
    }

    #[proxima::test]
    async fn try_acquire_returns_no_permits_when_exhausted() {
        let semaphore = Semaphore::new(1);
        let _held = semaphore.acquire().await.expect("first permit");
        let result = semaphore.try_acquire();
        assert!(matches!(result, Err(TryAcquireError::NoPermits)));
    }

    #[proxima::test]
    async fn close_then_acquire_returns_acquire_error() {
        let semaphore = Semaphore::new(1);
        semaphore.close();
        assert!(semaphore.is_closed());
        let result = semaphore.acquire().await;
        assert!(matches!(result, Err(AcquireError(_))));
    }

    #[proxima::test]
    async fn close_wakes_pending_acquires() {
        let semaphore = std::sync::Arc::new(Semaphore::new(1));
        let held = semaphore.acquire().await.expect("first permit");
        let pending_sem = semaphore.clone();
        // collapse to bool inside the task: the permit's lifetime is
        // bound to `pending_sem` which lives only inside the closure,
        // so don't return the Result itself
        let pending = tokio::spawn(async move { pending_sem.acquire().await.is_err() });
        // give the pending task a chance to register the listener
        tokio::task::yield_now().await;
        semaphore.close();
        let pending_was_err = pending.await.expect("join");
        assert!(pending_was_err);
        drop(held);
    }

    #[proxima::test]
    async fn add_permits_increases_available_count() {
        let semaphore = Semaphore::new(1);
        semaphore.add_permits(3);
        assert_eq!(semaphore.available_permits(), 4);
    }
}
