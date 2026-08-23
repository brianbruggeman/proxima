//! `F16`: IEEE-754 binary16, one element per 2-byte "block"
//! ([`crate::types::GgmlType::F16`]'s own [`crate::types::GgmlType::block_layout`]
//! already reports `block_elements: 1, block_bytes: 2` — not a codec this
//! module invents, just the shape that table already names).
//!
//! Unlike `q4_k`/`q5_k`/`q6_k`/`q8_0`, there is no bit-packing, no
//! sub-block scale, nothing to unpack: an `F16` element converts to `f32`
//! entirely on its own. The conversion itself composes the existing
//! [`half::f16`] dependency's `to_f32`/`from_f32` (already
//! round-to-nearest-even on ties, per `proxima-tensor/src/convert.rs`'s own
//! doc on reusing this same crate) rather than a hand-rolled bit-shift —
//! the pipe question, answered by writing the expression: `half` already
//! is the primitive, so this module is a thin byte-oriented wrapper
//! matching the other codecs' `dequantize`/`quantize` shape, not a second
//! implementation of IEEE-754 half-precision conversion.

use crate::quant::QuantError;

/// This codec's name as it appears in a rendered [`QuantError`] message.
const CODEC: &str = "f16";

/// Elements per "block" — always 1, [`crate::types::GgmlType::F16`]'s own
/// [`crate::types::GgmlType::block_layout`] agrees.
pub const QK_F16: usize = 1;

/// Bytes per "block" — 2, an IEEE-754 binary16 value.
pub const BLOCK_BYTES: usize = 2;

/// Number of whole `F16` elements a byte run decodes to, or `None` if
/// `byte_len` is not an exact multiple of [`BLOCK_BYTES`].
#[must_use]
pub const fn blocks_for_bytes(byte_len: usize) -> Option<usize> {
    if byte_len.is_multiple_of(BLOCK_BYTES) {
        Some(byte_len / BLOCK_BYTES)
    } else {
        None
    }
}

/// Exact packed byte length for `element_count` `F16` elements.
#[must_use]
pub const fn bytes_for_blocks(element_count: usize) -> usize {
    element_count * BLOCK_BYTES
}

/// Dequantizes one `F16` element. `block` must be exactly [`BLOCK_BYTES`]
/// bytes and `output` exactly one element — callers go through
/// [`dequantize`], which validates both.
pub fn dequantize_block(block: &[u8], output: &mut [f32]) {
    let mut bytes = [0u8; BLOCK_BYTES];
    bytes.copy_from_slice(block);
    output[0] = half::f16::from_le_bytes(bytes).to_f32();
}

/// Dequantizes a run of `F16` elements. `data` is borrowed, `output` is
/// caller-provided — no allocation on this path.
///
/// # Errors
/// [`QuantError::InputNotBlockMultiple`] if `data.len()` is not a
/// multiple of [`BLOCK_BYTES`]; [`QuantError::OutputSizeMismatch`] if
/// `output.len()` does not exactly match the decoded element count.
pub fn dequantize(data: &[u8], output: &mut [f32]) -> Result<(), QuantError> {
    let block_count = blocks_for_bytes(data.len()).ok_or(QuantError::InputNotBlockMultiple {
        codec: CODEC,
        found: data.len(),
        block_bytes: BLOCK_BYTES,
    })?;
    if output.len() != block_count {
        return Err(QuantError::OutputSizeMismatch {
            found: output.len(),
            expected: block_count,
        });
    }
    for (block, out) in data.as_chunks::<BLOCK_BYTES>().0.iter().zip(output.iter_mut()) {
        dequantize_block(block, core::slice::from_mut(out));
    }
    Ok(())
}

/// Quantizes a run of `f32` weights into `F16` bytes. `input` is
/// borrowed, `output` is caller-provided — no allocation on this path.
///
/// # Errors
/// [`QuantError::OutputSizeMismatch`] if `output.len()` does not exactly
/// match `input.len() * BLOCK_BYTES`.
pub fn quantize(input: &[f32], output: &mut [u8]) -> Result<(), QuantError> {
    let expected = bytes_for_blocks(input.len());
    if output.len() != expected {
        return Err(QuantError::OutputSizeMismatch {
            found: output.len(),
            expected,
        });
    }
    for (chunk, &value) in output.as_chunks_mut::<BLOCK_BYTES>().0.iter_mut().zip(input.iter()) {
        chunk.copy_from_slice(&half::f16::from_f32(value).to_le_bytes());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]

    use alloc::vec;

    use super::{BLOCK_BYTES, CODEC, QuantError, dequantize, quantize};

    /// Hand-packed fixture: 1.0, 2.0, -1.0, 0.0, and 0.5 are all exact in
    /// binary16 (each a power of two or zero), so `assert_eq!` needs no
    /// epsilon — computed by hand from the IEEE-754 binary16 bit layout
    /// (sign:1, exponent:5 bias 15, mantissa:10), not by calling
    /// [`super::quantize`] to build the fixture.
    #[test]
    fn dequantize_matches_hand_packed_ieee754_binary16_fixture() {
        // 1.0 = 0x3C00, 2.0 = 0x4000, -1.0 = 0xBC00, 0.0 = 0x0000, 0.5 = 0x3800
        let bytes: [u8; 10] = [0x00, 0x3C, 0x00, 0x40, 0x00, 0xBC, 0x00, 0x00, 0x00, 0x38];
        let mut output = [0.0f32; 5];
        dequantize(&bytes, &mut output).expect("5 whole f16 elements");
        assert_eq!(output, [1.0, 2.0, -1.0, 0.0, 0.5]);
    }

    #[test]
    fn quantize_dequantize_round_trips_exactly_representable_values() {
        let input = vec![1.0f32, -2.5, 0.0, 100.0, -0.125];
        let mut packed = vec![0u8; input.len() * BLOCK_BYTES];
        quantize(&input, &mut packed).expect("well-formed input");
        let mut output = vec![0.0f32; input.len()];
        dequantize(&packed, &mut output).expect("well-formed packed bytes");
        assert_eq!(output, input);
    }

    /// `f32`'s 24-bit mantissa carries more precision than binary16's
    /// 11-bit (10 explicit + implicit leading 1) mantissa — round-tripping
    /// a value that needs more than 11 significant bits must lose exactly
    /// the bits binary16 cannot hold, a deterministic loss computed by
    /// hand, not an epsilon picked to make the assertion pass.
    #[test]
    fn quantize_dequantize_loses_precision_beyond_binary16_mantissa_deterministically() {
        // binary16's mantissa at this exponent (value in [1,2)) has step
        // 2^-10. `1.0 + 2^-10 + 2^-12` sits 1/4 of a step past the
        // representable grid point `1.0 + 2^-10`, and 3/4 of a step short
        // of the next one (`1.0 + 2*2^-10`) — an unambiguous nearest
        // rounding, no round-to-even tie to reason about.
        let input = 1.0f32 + 2.0f32.powi(-10) + 2.0f32.powi(-12);
        let mut packed = [0u8; BLOCK_BYTES];
        quantize(core::slice::from_ref(&input), &mut packed).expect("one element");
        let mut output = [0.0f32];
        dequantize(&packed, &mut output).expect("one element");
        assert_eq!(output[0], 1.0f32 + 2.0f32.powi(-10));
        assert_ne!(output[0], input, "binary16 must not carry f32's extra mantissa bit");
    }

    #[test]
    fn dequantize_rejects_non_block_multiple_length() {
        let data = vec![0u8; BLOCK_BYTES + 1];
        let mut output = vec![0.0f32; 1];
        let error = dequantize(&data, &mut output).unwrap_err();
        assert_eq!(
            error,
            QuantError::InputNotBlockMultiple {
                codec: CODEC,
                found: BLOCK_BYTES + 1,
                block_bytes: BLOCK_BYTES,
            }
        );
    }

    #[test]
    fn dequantize_rejects_output_size_mismatch() {
        let data = vec![0u8; BLOCK_BYTES * 2];
        let mut output = vec![0.0f32; 1];
        let error = dequantize(&data, &mut output).unwrap_err();
        assert_eq!(
            error,
            QuantError::OutputSizeMismatch {
                found: 1,
                expected: 2,
            }
        );
    }

    #[test]
    fn quantize_rejects_output_size_mismatch() {
        let input = vec![0.0f32; 2];
        let mut output = vec![0u8; BLOCK_BYTES];
        let error = quantize(&input, &mut output).unwrap_err();
        assert_eq!(
            error,
            QuantError::OutputSizeMismatch {
                found: BLOCK_BYTES,
                expected: BLOCK_BYTES * 2,
            }
        );
    }
}
