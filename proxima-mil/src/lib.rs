//! Sans-IO writer for Core ML ML Programs: the MIL protobuf encoder, weight
//! blob writer and package assembler that omega's door plugs in as its Emit
//! and Package stages.
//!
//! The crate is `no_std`; `alloc` and `std` are additive features. Every
//! tunable axis is a key of `proxima-mil.toml`, resolved by `build.rs` into
//! [`sized`]; teaching pointer: `proxima-tensor/build.rs` and
//! `proxima_build::sizing::SizingSource` are the pattern this mirrors.
#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(test)]
extern crate std;

#[cfg(feature = "alloc")]
pub mod census;
pub mod sized;

/// Names of the modules this build compiles, so a tier build that gates every
/// module off reports zero names instead of passing silently.
#[cfg(feature = "alloc")]
pub const MODULES: [&str; 2] = ["census", "sized"];

/// Names of the modules this build compiles, so a tier build that gates every
/// module off reports zero names instead of passing silently.
#[cfg(not(feature = "alloc"))]
pub const MODULES: [&str; 1] = ["sized"];
