//! MSL kernel emission from the proxima-tensor [`Nest`](proxima_tensor::Nest)
//! descriptor — the GPU half of the Nest seam.
//!
//! `proxima-tensor`'s [`cpu`](proxima_tensor::cpu) module interprets a `Nest`
//! with nested loops on the CPU; this crate emits Metal Shading Language
//! source from the *same* descriptor instead. Neither backend's shape leaks
//! into `proxima-tensor` — `nest::lower` says only what to compute (which
//! buffers, at what strides, combined by which scalar op, optionally folded);
//! it never says how. [`msl::emit`] is this crate's answer to "how," the way
//! `cpu::evaluate` is proxima-tensor's own.
//!
//! This crate stops at source text plus the buffer-index -> data mapping a
//! driver needs to dispatch it ([`msl::Kernel`], [`msl::Binding`]). Packing
//! the runtime uniforms buffer, compiling the source into an `MTLLibrary`,
//! allocating device buffers, and actually dispatching a compute pass are all
//! the device driver's job, composed on top.
//!
//! # Tiers
//!
//! `alloc` (no_std + alloc): the whole crate — emission is pure string
//! generation over an already-lowered `Nest`, so it never needs floating-point
//! transcendentals or an allocator beyond `alloc`. `std` (default) only adds
//! `std::error::Error` on [`EmitError`].

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod error;
pub mod msl;

pub use error::EmitError;
pub use msl::{Binding, GridSpec, Kernel, emit};
