//! `proxima::sync::RwLock` — async reader-writer lock, shape-
//! compatible with `tokio::sync::RwLock`. Backed by
//! `async_lock::RwLock` with a thin newtype that reshapes
//! `try_read` / `try_write` from `Option` (async-lock) to
//! `Result<_, TryLockError>` (tokio). The async-lock variant is
//! fair (FIFO across readers and writers) and provides the same
//! `read(&self).await` / `write(&self).await` shape as tokio.
//!
//! # Non-coverage
//!
//! - `read_owned() -> OwnedRwLockReadGuard<T>` /
//!   `write_owned() -> OwnedRwLockWriteGuard<T>` — Arc-coupled owned
//!   guards. tokio offers these for guards that outlive a borrow of
//!   the lock; `async_lock::RwLock` has no analog. Mirror of the
//!   `lock_owned` gap on Mutex.
//! - `try_read_owned() -> Result<OwnedRwLockReadGuard<T>, TryLockError>`
//!   and `try_write_owned()` — same.
//! - `read_arc()` / `write_arc()` — async-lock's Arc-coupled variants;
//!   not re-exported because no caller currently needs them.
//! - `RwLock::blocking_read()` / `RwLock::blocking_write()` — sync-
//!   context acquisition. Don't use sync acquisition from async code.

pub use async_lock::{RwLockReadGuard, RwLockWriteGuard};

use crate::sync::TryLockError;

/// Async reader-writer lock matching `tokio::sync::RwLock`'s shape.
#[derive(Debug, Default)]
pub struct RwLock<T: ?Sized>(async_lock::RwLock<T>);

impl<T> RwLock<T> {
    /// New lock containing `value`.
    #[must_use]
    pub fn new(value: T) -> Self {
        Self(async_lock::RwLock::new(value))
    }

    /// Consume the lock, returning the inner value.
    pub fn into_inner(self) -> T {
        self.0.into_inner()
    }
}

impl<T: ?Sized> RwLock<T> {
    /// Acquire shared read access asynchronously.
    pub async fn read(&self) -> RwLockReadGuard<'_, T> {
        self.0.read().await
    }

    /// Acquire exclusive write access asynchronously.
    pub async fn write(&self) -> RwLockWriteGuard<'_, T> {
        self.0.write().await
    }

    /// Try to acquire a read guard without waiting.
    pub fn try_read(&self) -> Result<RwLockReadGuard<'_, T>, TryLockError> {
        self.0.try_read().ok_or_else(crate::sync::mutex::try_lock_error)
    }

    /// Try to acquire a write guard without waiting.
    pub fn try_write(&self) -> Result<RwLockWriteGuard<'_, T>, TryLockError> {
        self.0.try_write().ok_or_else(crate::sync::mutex::try_lock_error)
    }

    /// Mutable access without locking — `&mut self` proves exclusivity
    /// statically.
    pub fn get_mut(&mut self) -> &mut T {
        self.0.get_mut()
    }
}
