//! `proxima::sync::Mutex` — async mutex, shape-compatible with
//! `tokio::sync::Mutex`. Backed by `futures::lock::Mutex` (no runtime
//! coupling). Thin newtype: forwards `lock` / `get_mut` / `into_inner`
//! to the futures variant; reshapes `try_lock` so the return is
//! `Result<MutexGuard<'_, T>, TryLockError>` matching tokio's shape
//! rather than the futures crate's `Option<MutexGuard<'_, T>>`.
//!
//! Cancel-safety nuance: tokio's `lock()` future is documented
//! cancel-safe — dropping the future before it resolves leaves the
//! mutex untouched. `futures::lock::Mutex::lock()` has the same
//! property (the future never holds the lock until it resolves), so
//! the semantics carry over.
//!
//! # Non-coverage
//!
//! The following tokio APIs are NOT shimmed:
//!
//! - `lock_owned() -> OwnedMutexGuard<T>` — tokio's owned-guard
//!   variant requires `Arc<Mutex<T>>` self and returns a guard whose
//!   lifetime is decoupled from any reference. `futures::lock::Mutex`
//!   has no analog. Migration: clone the `Arc<Mutex<T>>`, hold it
//!   alongside the guard, drop guard first.
//! - `try_lock_owned() -> Result<OwnedMutexGuard<T>, TryLockError>` —
//!   same as above; no owned variant.
//! - `Mutex::blocking_lock()` — sync-context lock acquisition. Don't
//!   call sync APIs from async code; if you need this, the design is
//!   probably wrong. Use `std::sync::Mutex` for non-async paths.

use core::fmt;

pub use futures::lock::{MutexGuard, MutexLockFuture};

/// Async mutex matching `tokio::sync::Mutex`'s API shape, backed by
/// `futures::lock::Mutex`.
#[derive(Debug, Default)]
pub struct Mutex<T: ?Sized>(futures::lock::Mutex<T>);

impl<T> Mutex<T> {
    /// Construct a new mutex around `value`.
    #[must_use]
    pub fn new(value: T) -> Self {
        Self(futures::lock::Mutex::new(value))
    }

    /// Unwrap the mutex, returning the inner value.
    pub fn into_inner(self) -> T {
        self.0.into_inner()
    }
}

impl<T: ?Sized> Mutex<T> {
    /// Acquire the mutex asynchronously.
    pub fn lock(&self) -> MutexLockFuture<'_, T> {
        self.0.lock()
    }

    /// Try to acquire without waiting. Matches tokio's `Result` shape
    /// so callers can `?` the error or `match` on success without
    /// adapting the futures-crate `Option` form.
    pub fn try_lock(&self) -> Result<MutexGuard<'_, T>, TryLockError> {
        self.0.try_lock().ok_or(TryLockError(()))
    }

    /// Mutable access without locking — `&mut self` proves exclusivity
    /// statically.
    pub fn get_mut(&mut self) -> &mut T {
        self.0.get_mut()
    }
}

/// Error returned by `Mutex::try_lock` when the lock would block.
/// Matches `tokio::sync::TryLockError`'s shape (unit-payload).
#[derive(Debug)]
pub struct TryLockError(());

/// Crate-internal constructor — other primitives (`RwLock`) that
/// also reshape `Option` → `Result<_, TryLockError>` need to build
/// the error without exposing the private inner.
pub(crate) fn try_lock_error() -> TryLockError {
    TryLockError(())
}

impl fmt::Display for TryLockError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("operation would block")
    }
}

impl std::error::Error for TryLockError {}
