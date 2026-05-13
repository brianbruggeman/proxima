//! Sans-IO persistent-memory persistence and crash-consistency leaf.
//!
//! Tier-3: compiles under `#![no_std]` with no allocator. Two parts:
//!
//! - [`persist`] — the ordering primitives ([`flush`], [`drain`], [`persist`])
//!   that make stores to a borrowed region durable in the right order, over
//!   `core::arch` cache-maintenance intrinsics. PMDK `libpmem` is the parity
//!   reference; this is pure Rust with zero C linked, the storage analog of the
//!   pure-Rust DPDK / SPDK stacks next door.
//! - [`cow`] — [`CowRoot`], a copy-on-write atomic-root-swap crash-consistent
//!   update state machine. An update writes the new value into the currently
//!   dead of two slots, persists it, then flips a single 8-byte aligned root
//!   word (power-fail atomic per the SNIA/Intel ADR guarantee), then persists
//!   the root. [`CowRoot::recover`] reads only the root and returns the slot it
//!   selects — no log, no replay. This is the shadow-paging design LMDB
//!   (meta-page txnid) and ZFS (uberblock) use; it was chosen over undo- and
//!   redo-logging because its crash-reordering oracle is the simplest to prove
//!   (see `docs/pmem/discipline.md`).
//!
//! The codec parses and persists; the I/O facade that maps a real pmem region
//! (DAX `mmap`) lives elsewhere — this layer never touches a syscall.
pub mod cow;
pub mod error;
pub mod persist;

pub use cow::{CowRoot, UpdateState};
pub use error::PmemError;
pub use persist::{CACHE_LINE, drain, flush, persist};
