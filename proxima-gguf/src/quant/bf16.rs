//! `Bf16` (`bfloat16`): one element per 2-byte "block"
//! ([`crate::types::GgmlType::Bf16`]'s own
//! [`crate::types::GgmlType::block_layout`] already reports
//! `block_elements: 1, block_bytes: 2`). Same non-block, no-scale shape as
//! [`crate::quant::f16`] — see that module's doc for why this is a
//! composed conversion (the existing [`half::bf16`] dependency), not a
//! hand-rolled codec.
//!
//! `bf16` shares `f32`'s 8-bit exponent and truncates the 23-bit mantissa
//! to 7 bits, so widening `bf16 -> f32` is exact (a zero-extend of the
//! bit pattern, `half::bf16::to_f32` performs it losslessly) — it is the
//! narrowing direction, [`quantize`], that discards the low 16 mantissa
//! bits and is where precision is actually lost.

use crate::quant::QuantError;

/// This codec's name as it appears in a rendered [`QuantError`] message.
const CODEC: &str = "bf16";

/// Elements per "block" — always 1.
pub const QK_BF16: usize = 1;

/// Bytes per "block" — 2, a bfloat16 value.
pub const BLOCK_BYTES: usize = 2;

/// Number of whole `Bf16` elements a byte run decodes to, or `None` if
/// `byte_len` is not an exact multiple of [`BLOCK_BYTES`].
#[must_use]
pub const fn blocks_for_bytes(byte_len: usize) -> Option<usize> {
    if byte_len.is_multiple_of(BLOCK_BYTES) {
        Some(byte_len / BLOCK_BYTES)
    } else {
        None
    }
}

/// Exact packed byte length for `element_count` `Bf16` elements.
#[must_use]
pub const fn bytes_for_blocks(element_count: usize) -> usize {
    element_count * BLOCK_BYTES
}

/// Dequantizes one `Bf16` element. `block` must be exactly
/// [`BLOCK_BYTES`] bytes and `output` exactly one element — callers go
/// through [`dequantize`], which validates both.
pub fn dequantize_block(block: &[u8], output: &mut [f32]) {
    let mut bytes = [0u8; BLOCK_BYTES];
    bytes.copy_from_slice(block);
    output[0] = half::bf16::from_le_bytes(bytes).to_f32();
}

/// Dequantizes a run of `Bf16` elements. `data` is borrowed, `output` is
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
    for (block, out) in data
        .as_chunks::<BLOCK_BYTES>()
        .0
        .iter()
        .zip(output.iter_mut())
    {
        dequantize_block(block, core::slice::from_mut(out));
    }
    Ok(())
}

/// Quantizes a run of `f32` weights into `Bf16` bytes. `input` is
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
    for (chunk, &value) in output
        .as_chunks_mut::<BLOCK_BYTES>()
        .0
        .iter_mut()
        .zip(input.iter())
    {
        chunk.copy_from_slice(&half::bf16::from_f32(value).to_le_bytes());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]

    use alloc::vec;

    use super::{BLOCK_BYTES, CODEC, QuantError, dequantize, quantize};

    /// Hand-packed fixture: `bf16` is `f32`'s top 16 bits, so 1.0, 2.0,
    /// -1.0, 0.0, and 0.5 (each a power of two or zero, all exact in
    /// `bf16`) are computed by hand from `f32`'s own bit layout truncated
    /// to 16 bits, not by calling [`super::quantize`].
    #[test]
    fn dequantize_matches_hand_packed_bfloat16_fixture() {
        // f32 1.0 = 0x3F800000 -> top 16 bits 0x3F80; 2.0 = 0x40000000 ->
        // 0x4000; -1.0 = 0xBF800000 -> 0xBF80; 0.0 -> 0x0000; 0.5 =
        // 0x3F000000 -> 0x3F00. Little-endian byte order per element.
        let bytes: [u8; 10] = [0x80, 0x3F, 0x00, 0x40, 0x80, 0xBF, 0x00, 0x00, 0x00, 0x3F];
        let mut output = [0.0f32; 5];
        dequantize(&bytes, &mut output).expect("5 whole bf16 elements");
        assert_eq!(output, [1.0, 2.0, -1.0, 0.0, 0.5]);
    }

    #[test]
    fn quantize_dequantize_round_trips_exactly_representable_values() {
        let input = vec![1.0f32, -2.0, 0.0, 128.0, -0.5];
        let mut packed = vec![0u8; input.len() * BLOCK_BYTES];
        quantize(&input, &mut packed).expect("well-formed input");
        let mut output = vec![0.0f32; input.len()];
        dequantize(&packed, &mut output).expect("well-formed packed bytes");
        assert_eq!(output, input);
    }

    /// `bf16` truncates `f32`'s 23-bit mantissa to 7 bits: any value whose
    /// significant bits extend past the top 7 mantissa bits must lose
    /// exactly the discarded low bits, a deterministic loss computed by
    /// hand from the bit pattern, never an epsilon chosen to pass.
    #[test]
    fn quantize_dequantize_loses_precision_beyond_bfloat16_mantissa_deterministically() {
        // bf16's mantissa at this exponent (value in [1,2)) has step
        // 2^-7. `1.0 + 2^-7 + 2^-9` sits 1/4 of a step past the
        // representable grid point `1.0 + 2^-7`, and 3/4 of a step short
        // of the next one (`1.0 + 2*2^-7`) — an unambiguous nearest
        // rounding, no round-to-even tie to reason about.
        let input = 1.0f32 + 2.0f32.powi(-7) + 2.0f32.powi(-9);
        let mut packed = [0u8; BLOCK_BYTES];
        quantize(core::slice::from_ref(&input), &mut packed).expect("one element");
        let mut output = [0.0f32];
        dequantize(&packed, &mut output).expect("one element");
        assert_eq!(output[0], 1.0f32 + 2.0f32.powi(-7));
        assert_ne!(
            output[0], input,
            "bfloat16 must not carry f32's extra mantissa bits"
        );
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
