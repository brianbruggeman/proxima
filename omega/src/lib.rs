//! MSL kernel emission from the proxima-tensor
//! [`BoundOp`](proxima_tensor::BoundOp) descriptor — the GPU half of the
//! bound-addressing seam.
//!
//! `proxima-tensor`'s `cpu` module interprets a
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

#[cfg(not(any(feature = "std", feature = "alloc")))]
compile_error!(
    "omega currently requires the `alloc` feature (or `std`, which implies \
     it) -- the whole crate is alloc-tier msl emission over proxima-tensor's \
     op/map/shape surface"
);

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
pub mod backend;
pub mod error;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;
pub mod msl;
pub mod sized;
#[cfg(feature = "wgpu-backend")]
pub mod wgpu_driver;
#[cfg(feature = "wgpu-backend")]
pub mod wgsl;

pub use error::EmitError;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use metal::{
    MetalError, Plan, execute, execute_plan, execute_plan_named, page_size, plan, plan_named,
};
pub use msl::{
    BF16_UNPACK_MSL, BFLOAT16_BLOCK_BYTES, BFLOAT16_BLOCK_ELEMENTS, Binding,
    FLOAT16_BLOCK_BYTES, FLOAT16_BLOCK_ELEMENTS, GridSpec, Kernel, PackedCodec, PackedOperands,
    Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMENTS, Q4_0_UNPACK_MSL, Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMENTS,
    Q4K_UNPACK_MSL, Q5K_BLOCK_BYTES, Q5K_UNPACK_MSL, Q6K_BLOCK_BYTES, Q6K_UNPACK_MSL,
    Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS, Q8_0_UNPACK_MSL, emit,
};
#[cfg(feature = "wgpu-backend")]
pub use wgpu_driver::{WgpuError, WgpuPlan, execute_plan as execute_plan_wgpu, execute_plan_named as execute_plan_named_wgpu, plan as plan_wgpu, plan_named as plan_named_wgpu};
#[cfg(feature = "wgpu-backend")]
pub use wgsl::{WORKGROUP_SIZE, WgslKernel, emit_wgsl};
