//! Storage substrates: the NVMe queue-pair engine, the sans-IO persistent-memory
//! (pmem) leaf, and the std DAX/file-backed mmap facade over that leaf. Folded
//! from the formerly separate `proxima-nvme`, `proxima-pmem`, and
//! `proxima-pmem-dax` crates into one crate with feature-gated modules — each
//! module keeps its own tier discipline (see the module docs).
//!
//! - [`nvme`] (feature `nvme`): tier-3 no_std + no-alloc queue-pair engine
//!   driving the sans-IO [`proxima_protocols::nvme`] codec.
//! - [`pmem`]: tier-3 no_std + no-alloc persistence and crash-consistency
//!   primitives, unconditionally available.
//! - [`dax`] (feature `dax`): std DAX/file-backed mmap facade over [`pmem`],
//!   Linux-only for the pmem-native fast tier (the portable [`dax::FileCell`]
//!   floor works on any OS).
#![cfg_attr(not(feature = "std"), no_std)]

// only proxima-pmem's cow FSM tests reach for alloc collections; everything
// else in the crate is either no-alloc or genuinely std.
#[cfg(test)]
extern crate alloc;

#[cfg(feature = "dax")]
pub mod dax;
#[cfg(feature = "nvme")]
pub mod nvme;
pub mod pmem;
