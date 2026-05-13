//! Shared dispatch contract — protocol envelope + marker traits.
//!
//! This crate is the load-bearing "single source of truth" that
//! both proxima-process's libc-interpose shim AND proxima-vm
//! consume. Per `proxima.decision.libc_shim_vm_parity`, both
//! mechanisms speak identical `ChildRequest` / `ChildResponse`
//! over identical wire bytes; this crate defines the types so
//! they can.
//!
//! # Tier
//!
//! `no_std + alloc` by default. No std dependencies; pulls in
//! only `serde` + `alloc`. Crates that need just the protocol
//! shape (proxima-vm, downstream sandbox plugins) can depend on
//! this crate without inheriting proxima-process's larger
//! dep tree (nix, bon, conflaguration, etc.).
//!
//! # What's here
//!
//! - [`protocol`] — `ChildRequest` / `ChildResponse` /
//!   `ReadResponse` / `WriteResponse` typed envelope, serde-
//!   derived for postcard framing.
//! - [`markers`] — empty marker traits for compile-time effect
//!   tracking (`NoStd`, `AllocFree`, `IsPure`, `Without*`,
//!   `Deterministic`, etc.). Pure data types — re-exported by
//!   proxima-process for backward compat.
//!
//! Both modules are also re-exported at the crate root for
//! convenience.

pub mod markers;
pub mod protocol;

pub use markers::{
    AllocFree, Commutative, Deterministic, IdempotentSideEffectFree, IsPure, NoStd, Reproducible,
    WithoutFilesystem, WithoutNetwork, WithoutRandom, WithoutSpawn, WithoutTime,
};
pub use protocol::{ChildRequest, ChildResponse, ReadResponse, WriteResponse};
