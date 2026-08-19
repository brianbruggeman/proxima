//! Block-quantization codecs: unpacking a GGML block format into `f32`
//! weights and packing `f32` weights back down. Alloc-tier, no IO, no
//! allocation on the hot path — every function here takes a borrowed
//! input slice and writes into a caller-provided output slice.
//!
//! This is the piece [`crate`]'s module doc used to call "a separate,
//! already-sized job": the parser reports a tensor's [`crate::GgmlType`]
//! and byte range faithfully, and this module turns those raw bytes into
//! numbers (and back), one block format at a time. [`q4_k`] is the first.

pub mod q4_k;
