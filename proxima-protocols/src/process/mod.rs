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
//!
//! The compile-time effect markers (`NoStd`, `AllocFree`, `IsPure`,
//! `Without*`, `Deterministic`, ...) are `proxima_core::markers`' and
//! are named here so a consumer of the dispatch contract reaches them
//! without a second dependency. There is no `markers` submodule: a
//! module whose whole body was `pub use proxima_core::markers::*` let
//! an open set through invisibly and let the two paths drift (the glob
//! carried `DropSafe`, this list did not).

pub mod protocol;

pub use protocol::{ChildRequest, ChildResponse, ReadResponse, WriteResponse};
pub use proxima_core::markers::{
    AllocFree, Commutative, Deterministic, DropSafe, IdempotentSideEffectFree, IsPure, NoStd,
    Reproducible, WithoutFilesystem, WithoutNetwork, WithoutRandom, WithoutSpawn, WithoutTime,
};
