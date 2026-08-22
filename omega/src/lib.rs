//! MSL kernel emission from the proxima-tensor
//! [`BoundOp`](proxima_tensor::BoundOp) descriptor — the GPU half of the
//! bound-addressing seam.
//!
//! `proxima-tensor`'s [`cpu`](proxima_tensor::cpu) module interprets a
//! `BoundOp` with nested loops on the CPU; this crate emits Metal Shading
//! Language source from the *same* descriptor instead. Neither backend's
//! shape leaks into `proxima-tensor` — `bind::bind` says only what to
//! compute (which buffers, at what layout, combined by which scalar op,
//! optionally reduced); it never says how. [`msl::emit`] is this crate's
//! answer to "how," the way `cpu::evaluate` is proxima-tensor's own.
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
//! generation over an already-bound `BoundOp`, so it never needs
//! floating-point transcendentals or an allocator beyond `alloc`. `std`
//! (default) only adds `std::error::Error` on [`EmitError`].

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
pub mod backend;
pub mod error;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;
pub mod msl;
pub mod sized;

pub use error::EmitError;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use metal::{
    MetalError, Plan, execute, execute_plan, execute_plan_named, page_size, plan, plan_named,
};
pub use msl::{
    Binding, GridSpec, Kernel, PackedCodec, PackedOperands, Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMENTS,
    Q4K_UNPACK_MSL, Q5K_BLOCK_BYTES, Q5K_UNPACK_MSL, Q6K_BLOCK_BYTES, Q6K_UNPACK_MSL, emit,
};
