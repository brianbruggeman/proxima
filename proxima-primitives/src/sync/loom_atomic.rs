//! Atomics/lock shim consumed by [`crate::sync::notify`] and [`crate::sync::watch`].
//!
//! Under `--cfg loom` every symbol here resolves to loom's instrumented
//! equivalent, so the model checker sees every atomic operation in the
//! real permit/version protocol — including the ones inside
//! `event_listener::Event`, which ships its own `#[cfg(loom)]` branches
//! and becomes loom-instrumented for free once its `loom` feature is
//! enabled (see `proxima-primitives/Cargo.toml`'s `[target.'cfg(loom)'.dependencies]`).
//!
//! The `cfg(not(loom))` path re-exports the exact `core`/`alloc`/`std`
//! items `notify.rs`/`watch.rs` used before this shim existed, so normal
//! builds are unchanged.

#[cfg(loom)]
pub(crate) use loom::sync::Arc;
#[cfg(loom)]
pub(crate) use loom::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
#[cfg(loom)]
pub(crate) use loom::sync::{RwLock, RwLockReadGuard};

// `Arc`/`AtomicU64`/`AtomicUsize`/`RwLock`/`RwLockReadGuard` back `watch.rs`
// only, which stays std-gated in `sync/mod.rs`; this module now also
// compiles under plain `alloc` (no std) for `notify.rs`'s `AtomicBool` +
// `Ordering` alone, so the watch-only symbols stay behind `std` to avoid
// unused-import errors at the alloc tier.
#[cfg(all(not(loom), feature = "std"))]
pub(crate) use alloc::sync::Arc;
#[cfg(not(loom))]
pub(crate) use core::sync::atomic::{AtomicBool, Ordering};
#[cfg(all(not(loom), feature = "std"))]
pub(crate) use core::sync::atomic::{AtomicU64, AtomicUsize};
#[cfg(all(not(loom), feature = "std"))]
pub(crate) use std::sync::{RwLock, RwLockReadGuard};
