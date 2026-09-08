//! Block-quantization codecs: unpacking a GGML block format into `f32`
//! weights and packing `f32` weights back down. Alloc-tier, no IO, no
//! allocation on the hot path — every function here takes a borrowed
//! input slice and writes into a caller-provided output slice.
//!
//! This is the piece [`crate`]'s module doc used to call "a separate,
//! already-sized job": the parser reports a tensor's [`crate::GgmlType`]
//! and byte range faithfully, and this module turns those raw bytes into
//! numbers (and back), one block format at a time. [`q4_k`] landed first;
//! [`q8_0`], [`q6_k`], [`q5_k`], [`q4_0`], [`q3_k`], and [`q2_k`] followed. [`mod@f16`]/[`bf16`]
//! closed a different gap: both types were tag-only in
//! [`crate::types::GgmlType`] (readable byte ranges, no conversion) until
//! these two landed — not a block codec at all (`block_elements: 1`), just
//! a widening convert composed from the existing `half` crate dependency.

pub mod bf16;
pub mod f16;
pub mod iq2_xs;
pub mod iq3_xxs;
pub mod iq4_nl;
pub mod policy;
pub mod q2_k;
pub mod q3_k;
pub mod q4_0;
pub mod q4_k;
pub mod q5_1;
pub mod q5_k;
pub mod q6_k;
pub mod q8_0;
pub mod tables;

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
    #[error(
        "input length {found} elements is not a multiple of the {codec} {unit} size {block_elements}"
    )]
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

/// Real-tensor fixture shared by every codec's `#[cfg(test)]` module (P9:
/// real-world data, one definition per guiding-principle 1 rather than a
/// copy per codec file). `std`-only: it reads a real GGUF file off disk,
/// which the `alloc`-only tier never does.
#[cfg(all(test, feature = "std"))]
pub(crate) mod real_weights {
    #![allow(clippy::expect_used)]
    extern crate std;

    use alloc::vec::Vec;
    use std::path::Path;

    /// `Qwen/Qwen3-1.7B`, `Q4_K_M`, host-local (same fixture
    /// `proxima-model-interop::test_support::qwen3_gguf_path` resolves by
    /// default) -- overridable so a host without this exact blob can point
    /// at any local `Q4_K_M`-class checkpoint with a `token_embd.weight`
    /// (`Q4_K`) and `output.weight` (`Q6_K`) tensor.
    fn qwen3_path() -> std::string::String {
        std::env::var("PROXIMA_QWEN3_GGUF").unwrap_or_else(|_| {
            "/Users/brianbruggeman/.ollama/models/blobs/\
             sha256-3d0b790534fe4b79525fc3692950408dca41171676ed7e21db57af5c65ef6ab6"
                .into()
        })
    }

    /// Dequantizes `min_elements`-worth (rounded up to a whole number of
    /// `Q4_K` super-blocks) of the checkpoint's own `token_embd.weight`
    /// tensor to `f32`, using [`super::super::q4_k::dequantize`] -- this
    /// crate's own already-tested decoder, not a third-party oracle. The
    /// output is a REAL embedding-table weight distribution: exactly the
    /// "encode a real weight row" input principle 9 asks for, reused by
    /// every codec's round-trip test rather than each codec re-deriving a
    /// synthetic stand-in.
    ///
    /// # Panics
    /// If the fixture file is missing (set `PROXIMA_QWEN3_GGUF` or stage
    /// one at the default path), has no `token_embd.weight` tensor, or
    /// that tensor is not `Q4_K` -- loud failure, never a silent skip.
    pub(crate) fn qwen3_token_embd_f32(min_elements: usize) -> Vec<f32> {
        let path = qwen3_path();
        assert!(
            Path::new(&path).exists(),
            "no host-local qwen3 gguf fixture at {path}: set PROXIMA_QWEN3_GGUF or stage one"
        );
        let (parsed, bytes) =
            crate::edge::read_file(Path::new(&path)).expect("qwen3 fixture must parse as gguf");
        let tensor = parsed
            .tensors
            .iter()
            .find(|candidate| candidate.name == "token_embd.weight")
            .expect("qwen3 fixture must have a token_embd.weight tensor");
        assert_eq!(
            tensor.ggml_type,
            crate::types::GgmlType::Q4_K,
            "qwen3 fixture's token_embd.weight must be Q4_K (Q4_K_M layout)"
        );

        let block_count = min_elements.div_ceil(super::q4_k::QK_K).max(1);
        let needed_bytes = super::q4_k::bytes_for_blocks(block_count);
        let file_len = bytes.len() as u64;
        let full_range = parsed
            .tensor_data_range(tensor, file_len)
            .expect("token_embd.weight range must fit the file");
        let start = full_range.start as usize;
        let end = (start + needed_bytes).min(full_range.end as usize);
        let block_bytes = &bytes[start..end];

        let element_count = super::q4_k::elements_for_blocks(block_bytes.len() / super::q4_k::BLOCK_BYTES);
        let mut output = alloc::vec![0.0f32; element_count];
        super::q4_k::dequantize(block_bytes, &mut output).expect("well-formed q4_k byte run");
        output
    }
}
