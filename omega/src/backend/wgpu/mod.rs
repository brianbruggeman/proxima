//! The wgpu half of omega's backends: the WGSL spelling of the backend-neutral render.
//!
//! Teaching pointer: [`dialect::Wgsl`] implements [`crate::render::Dialect`]; callers use
//! `render::{load, store, unpack}(&Wgsl, ..)` directly.

pub mod dialect;

#[cfg(test)]
mod dialect_tests;
