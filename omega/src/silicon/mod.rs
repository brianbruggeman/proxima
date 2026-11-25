//! One independent area per hardware: each module owns its device, runtime and launch.

#[cfg(feature = "cuda-driver")]
pub mod hip;
