//! `proxima::sync` — runtime-agnostic concurrency primitives shaped like
//! `tokio::sync`. Users get the API surface they expect (`use
//! proxima_primitives::sync::Mutex;`) without pulling tokio. Backing crates are
//! `futures` (channels, lock::Mutex), `async-lock` (RwLock, Semaphore),
//! `event-listener` (Notify), and `async-broadcast` (watch, broadcast).
//!
//! The trait set re-exported here is *concurrency-defining*; runtime
//! coupling (spawn, timers, per-core context) stays on the
//! `Runtime` trait. See the proxima-tokio-elimination-library plan (P1.3).
//!
//! Tier: `notify`, `oneshot`, `shutdown`, and `AsyncMutex` (the
//! `async-mutex` feature) compile under `no_std + alloc` (each backing
//! crate has a no_std + alloc path; see DC-SYNC). The `watch` primitive
//! uses `std::sync::RwLock<T>` for its value cache and is gated behind
//! the `std` feature, as are the other remaining primitives (mpsc,
//! mutex, once_cell, rwlock, semaphore, broadcast) — full no_std + alloc
//! landing for those is deferred per DC-SYNC.
//!
//! [`blocking`] and [`task`] fold in the former `proxima-lock` and
//! `proxima-task` crates (Workstream F, RISC-dedup) — one runtime-agnostic
//! concurrency/task crate instead of three.

// Backing crates (async-*, event-listener, futures) expose no_std + alloc
// surfaces but the cliff requires per-crate feature-flag analysis to make
// the proxima-sync API surface available under no_std. `notify` + `oneshot`
// have made that crossing (see their `alloc` gates below); the remaining
// primitives still gate behind `std` pending the same per-crate analysis.
// Full no_std + alloc landing for those is deferred per DC-SYNC.
#[cfg(feature = "async-mutex")]
mod async_mutex;
pub mod blocking;
#[cfg(feature = "std")]
pub mod broadcast;
// alloc-tier (not std): only notify's AtomicBool/Ordering re-exports are
// reachable here without std; watch.rs is the sole consumer of this module's
// std-only RwLock re-export, and stays std-gated internally (see below).
#[cfg(feature = "alloc")]
mod loom_atomic;
#[cfg(feature = "std")]
pub mod mpsc;
#[cfg(feature = "std")]
mod mutex;
// re-gated from std to alloc: both event-listener and the AtomicBool it
// backs support no_std + alloc, so prime can depend on Notify without
// pulling proxima-primitives/std (and its tokio dependency) at all.
#[cfg(feature = "alloc")]
mod notify;
#[cfg(feature = "std")]
mod once_cell;
// re-gated from std to alloc: futures::channel::oneshot has no runtime
// coupling and compiles no_std + alloc.
#[cfg(feature = "alloc")]
pub mod oneshot;
// not part of the Notify/watch loom protocol; its regular
// proxima-runtime `tokio` feature dependency pulls tokio, which the loom
// build keeps tokio-free.
#[cfg(all(feature = "runtime-shaped-mutex", not(loom)))]
pub mod runtime_shaped;
#[cfg(feature = "std")]
mod rwlock;
#[cfg(feature = "std")]
mod semaphore;
// cross-core graceful shutdown (folded in from the former proxima-shutdown
// satellite crate) — its ResourceRegistry primitive is no_std + alloc; the
// ambient std convenience and ShutdownBarrier gate on `std` internally.
#[cfg(feature = "alloc")]
pub mod shutdown;
pub mod task;
#[cfg(feature = "std")]
pub mod watch;

/// Multi-party rendezvous, shape-compatible with `tokio::sync::Barrier`.
/// Direct re-export of `async_lock::Barrier`. `wait()` resolves once
/// `N` tasks have called it. Useful for staged pipeline startup
/// (every worker reaches a barrier together) and integration tests.
#[cfg(feature = "std")]
pub use async_lock::Barrier;

#[cfg(feature = "async-mutex")]
pub use async_mutex::{AsyncMutex, AsyncMutexGuard};
#[cfg(feature = "std")]
pub use mutex::{Mutex, MutexGuard, MutexLockFuture, TryLockError};
#[cfg(feature = "alloc")]
pub use notify::{Notified, Notify};
#[cfg(feature = "std")]
pub use once_cell::OnceCell;
#[cfg(feature = "std")]
pub use rwlock::{RwLock, RwLockReadGuard, RwLockWriteGuard};
#[cfg(feature = "std")]
pub use semaphore::{
    AcquireError, Semaphore, SemaphoreGuard, SemaphoreGuardArc, SemaphorePermit, TryAcquireError,
};
