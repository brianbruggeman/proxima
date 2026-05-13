//! `proxima_primitives::sync::blocking` — canonical tier-resolved BLOCKING mutex.
//!
//! Folded from the former `proxima-lock` crate (Workstream F, RISC-dedup:
//! `proxima-lock` had zero workspace consumers — an adoption gap, not dead
//! code). One `Mutex<T>` API, backend chosen by tier:
//!
//! - **std** (default): re-exports `parking_lot::{Mutex, MutexGuard}`
//!   unchanged — a genuine passthrough, no behavior change for std
//!   consumers.
//! - **`no_std` + `blocking`**: `lock_api::Mutex` over a futex raw mutex
//!   ([`futex::RawFutexMutex`]) backed by `atomic-wait` (Linux `futex`,
//!   macOS `__ulock`, Windows `WaitOnAddress`).
//!
//! There is deliberately no spin fallback: a blocking lock needs an OS wait
//! primitive, so `no_std` with neither `std` nor `blocking` exposes no
//! `Mutex` at all rather than silently degrading to a spinlock.
//!
//! This module's `Mutex` PARKS a real OS thread — the lock-of-last-resort
//! outside async contexts per the workspace hot-path discipline. It is
//! deliberately namespaced away from [`crate::sync::Mutex`] (the crate-root
//! ASYNC mutex, `futures::lock::Mutex`-backed, which yields the task
//! instead of parking a thread) so the two never collide under one name.

#[cfg(feature = "std")]
pub use parking_lot::{Mutex, MutexGuard};

#[cfg(all(not(feature = "std"), feature = "blocking"))]
mod futex;
#[cfg(all(not(feature = "std"), feature = "blocking"))]
pub use futex::{Mutex, MutexGuard, RawFutexMutex};
