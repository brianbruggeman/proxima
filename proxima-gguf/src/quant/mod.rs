//! Block-quantization codecs: unpacking a GGML block format into `f32`
//! weights and packing `f32` weights back down. Alloc-tier, no IO, no
//! allocation on the hot path — every function here takes a borrowed
//! input slice and writes into a caller-provided output slice.
//!
//! This is the piece [`crate`]'s module doc used to call "a separate,
//! already-sized job": the parser reports a tensor's [`crate::GgmlType`]
//! and byte range faithfully, and this module turns those raw bytes into
//! numbers (and back), one block format at a time. [`q4_k`] landed first;
//! [`q8_0`], [`q6_k`], [`q5_k`], and [`q4_0`] followed. [`mod@f16`]/[`bf16`]
//! closed a different gap: both types were tag-only in
//! [`crate::types::GgmlType`] (readable byte ranges, no conversion) until
//! these two landed — not a block codec at all (`block_elements: 1`), just
//! a widening convert composed from the existing `half` crate dependency.

pub mod bf16;
pub mod f16;
pub mod q4_0;
pub mod q4_k;
pub mod q5_k;
pub mod q6_k;
pub mod q8_0;
pub mod policy;

use thiserror::Error;

/// Everything that can go wrong sizing a block-quant codec call, shared by
/// every codec in this module (`q4_0`/`q4_k`/`q5_k`/`q6_k`/`q8_0`) instead
/// of each declaring its own structurally identical type. Never a panic: a
/// malformed or mis-sized buffer is always an `Err`. `codec` carries which
/// codec raised it, so the rendered message still names it.
#[derive(Debug, Error, PartialEq, Eq, Clone, Copy)]
pub enum QuantError {
    #[error("input length {found} bytes is not a multiple of the {codec} block size {block_bytes}")]
    InputNotBlockMultiple {
        codec: &'static str,
        found: usize,
        block_bytes: usize,
    },
    #[error("input length {found} elements is not a multiple of the {codec} {unit} size {block_elements}")]
    InputNotElementMultiple {
        codec: &'static str,
        /// `"super-block"` for the K-quants (`q4_k`/`q5_k`/`q6_k`), plain
        /// `"block"` for `q4_0`/`q8_0`, which have no sub-block structure.
        unit: &'static str,
        found: usize,
        block_elements: usize,
    },
    #[error("output slice has {found} elements, expected {expected}")]
    OutputSizeMismatch { found: usize, expected: usize },
}
