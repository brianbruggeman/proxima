//! The one place in omega that declares C-ABI types, `extern "C"` signatures and `libloading`
//! symbol tables. Every other module calls the safe or narrow functions exposed here.

#[cfg(feature = "cuda-driver")]
pub mod hip;
