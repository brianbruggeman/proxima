//! The CUDA half of omega's backends: the card-C spelling of the backend-neutral render.
//!
//! Teaching pointer: [`dialect::Cuda`] implements [`crate::render::Dialect`]; callers use
//! `render::{load, store, unpack}(&Cuda { target }, ..)` directly. HIP is the same dialect over
//! [`crate::cuda::target::Target::HIP`].

pub mod dialect;

#[cfg(test)]
mod dialect_tests;
