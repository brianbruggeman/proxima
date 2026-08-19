//! Block-quantization codecs: unpacking a GGML block format into `f32`
//! weights and packing `f32` weights back down. Alloc-tier, no IO, no
//! allocation on the hot path — every function here takes a borrowed
//! input slice and writes into a caller-provided output slice.
//!
//! This is the piece [`crate`]'s module doc used to call "a separate,
//! already-sized job": the parser reports a tensor's [`crate::GgmlType`]
//! and byte range faithfully, and this module turns those raw bytes into
//! numbers (and back), one block format at a time. [`q4_k`] landed first;
//! [`q8_0`], [`q6_k`], and [`q5_k`] followed.

pub mod q4_k;
pub mod q5_k;
pub mod q6_k;
pub mod q8_0;
pub mod policy;
