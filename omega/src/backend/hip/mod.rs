//! The ROCm runtime over the HIP half of the [`crate::cuda`] emitter.
//!
//! `crate::cuda` is deliberately usable without ROCm: it emits HIP C++ text
//! and links nothing. This module is the separate runtime half. It resolves
//! `libamdhip64` and `libhiprtc` at run time through [`crate::ffi::hip`] — no
//! link-time dependency, no headers, no build-script probing — compiles emitted
//! source to a code object with hipRTC, loads the code object, and launches
//! kernels through the HIP module API.
//!
//! # Why `dlopen` and not a `-sys` crate
//!
//! A ROCm build must stay buildable on a host with no ROCm installed — the
//! same property `cuda` (emitter) keeps against `cuda-driver`, and what lets
//! this crate's macOS CI gate compile the lane. A `hip-sys` dependency
//! requires headers and a link path at BUILD time; resolving symbols at RUN
//! time turns "no ROCm here" into a typed
//! [`error::HipDriverError::RuntimeNotInstalled`] instead of a build failure.
//!
//! # What is proven where
//!
//! The device path is only reachable when `libamdhip64` resolves and a GPU is
//! present. This host has neither, so what is exercised here is: the
//! symbol-resolution contract (an absent runtime is a typed error naming the
//! sonames tried), the element-window arithmetic and the kernel-parameter
//! order.
//! Numerical parity on a real device is a separate gate that needs the
//! hardware; [`CudaDriver::smoke_add_one`](crate::cuda_driver::CudaDriver::smoke_add_one) on a
//! `CudaDriver<HipRuntime>` (from [`shared::shared_driver`]) is that gate's entry point.
//!
//! # Scope
//!
//! [`runtime::HipRuntime`] is the second [`GpuRuntime`](crate::gpu_runtime::GpuRuntime) of the one
//! CUDA executor: a `DevicePlan<HipRuntime>` lowers and launches exactly as a `CudaPlan` does, with
//! the kernels rendered as HIP C++. This module is the device boundary beneath it: allocation,
//! copies and launch over the safe surface of [`crate::ffi::hip`].

pub mod binding;
pub mod buffer;
pub mod code_object;
pub mod device;
pub mod error;
pub mod launch;
pub mod module;
pub mod runtime;
pub mod shared;
pub mod window;

#[cfg(test)]
mod rocm_tests;
#[cfg(test)]
mod runtime_tests;
