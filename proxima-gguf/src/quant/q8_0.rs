//! `Q8_0`: 8-bit weights in 32-element blocks, one `f16` scale per block,
//! no sub-block scales, no bit-packing. `x = q*d` per element
//! (`ggml-quants.c:343-357`, cited on [`dequantize_block`]).
//!
//! Layout, from llama.cpp (`/Users/brianbruggeman/repos/others/llama.cpp`):
//! `ggml/src/ggml-common.h:209-214` — `block_q8_0` is 34 bytes per 32
//! elements (`ggml_half d`, `int8_t qs[32]` (`QK8_0`)). `QK8_0` is 32
//! (`ggml-common.h:209`). Deliberately not `block_q8_1`
//! (`ggml-common.h:216-227`), which adds a second `ggml_half s = d *
//! sum(qs[i])` field used only by some integer-dot-product matmul
//! kernels — GGUF tensor storage (the format this crate reads) uses
//! `Q8_0`, never `Q8_1`; `Q8_1` is a runtime activation-quantization
//! format, not a value `ggml_type` a `.gguf` tensor directory names. That
//! 34-byte figure is cross-checked here at compile time against
//! [`crate::types::GgmlType::Q8_0`]'s already-landed
//! [`crate::types::GgmlType::block_layout`] rather than re-typed by hand.

use crate::quant::QuantError;
use crate::types::GgmlType;

/// This codec's name as it appears in a rendered [`QuantError`] message.
const CODEC: &str = "q8_0";

/// Elements per block (`ggml-common.h:209`, `#define QK8_0 32`).
pub const QK8_0: usize = 32;

/// Bytes per block. Derived from the already-landed
/// [`crate::types::GgmlType::block_layout`] (`types.rs`), not re-typed —
/// that table was itself checked against `ggml-common.h`'s
/// `static_assert(sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0, ...)`
/// at `ggml-common.h:214`.
pub const BLOCK_BYTES: usize = {
    let layout = GgmlType::Q8_0.block_layout();
    assert!(layout.block_elements as usize == QK8_0, "GgmlType::Q8_0 block_elements drifted from QK8_0");
    layout.block_bytes as usize
};

const D_OFFSET: usize = 0;
const QS_OFFSET: usize = 2;

/// Number of whole `Q8_0` blocks a byte run decodes to, or `None` if
/// `byte_len` is not an exact multiple of [`BLOCK_BYTES`].
#[must_use]
pub const fn blocks_for_bytes(byte_len: usize) -> Option<usize> {
    if byte_len.is_multiple_of(BLOCK_BYTES) {
        Some(byte_len / BLOCK_BYTES)
    } else {
        None
    }
}

/// Exact packed byte length for `block_count` `Q8_0` blocks.
#[must_use]
pub const fn bytes_for_blocks(block_count: usize) -> usize {
    block_count * BLOCK_BYTES
}

/// Exact `f32` element count for `block_count` `Q8_0` blocks.
#[must_use]
pub const fn elements_for_blocks(block_count: usize) -> usize {
    block_count * QK8_0
}

fn f16_at(block: &[u8], offset: usize) -> half::f16 {
    let mut bytes = [0u8; 2];
    bytes.copy_from_slice(&block[offset..offset + 2]);
    half::f16::from_le_bytes(bytes)
}

/// Dequantizes one 32-element `Q8_0` block. `block` must be exactly
/// [`BLOCK_BYTES`] bytes and `output` exactly [`QK8_0`] elements —
/// callers go through [`dequantize`], which validates both.
///
/// Ports `dequantize_row_q8_0` (`ggml-quants.c:343-357`) exactly: each
/// signed byte scales by the block's single `f16` delta, no sub-block
/// structure at all.
pub fn dequantize_block(block: &[u8], output: &mut [f32]) {
    let d = f16_at(block, D_OFFSET).to_f32();
    let qs = &block[QS_OFFSET..QS_OFFSET + QK8_0];
    for (out, &byte) in output.iter_mut().zip(qs.iter()) {
        *out = f32::from(byte as i8) * d;
    }
}

/// Dequantizes a run of `Q8_0` blocks. `data` is borrowed, `output` is
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
    let expected = elements_for_blocks(block_count);
    if output.len() != expected {
        return Err(QuantError::OutputSizeMismatch {
            found: output.len(),
            expected,
        });
    }
    for (block, out_chunk) in data.chunks_exact(BLOCK_BYTES).zip(output.chunks_exact_mut(QK8_0)) {
        dequantize_block(block, out_chunk);
    }
    Ok(())
}

/// Quantizes one 32-element chunk into a `Q8_0` block.
///
/// Ports `quantize_row_q8_0_ref` (`ggml-quants.c:187-208`) exactly: the
/// block's `d` is `amax / 127` (`amax` the block's absolute max, `127`
/// the signed-int8 positive range), each level is `round(x * (1/d))`
/// clamped by construction to `[-127, 127]` (amax itself maps to exactly
/// ±127, nothing rounds past it), and `d` is zero (levels all zero) only
/// when every input in the block is zero.
fn quantize_block(x: &[f32], output: &mut [u8]) {
    let mut amax = 0.0f32;
    for &value in x {
        amax = amax.max(value.abs());
    }
    let d = amax / 127.0;
    let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };

    let block_scale = half::f16::from_f32(d);
    output[D_OFFSET..D_OFFSET + 2].copy_from_slice(&block_scale.to_le_bytes());

    let qs = &mut output[QS_OFFSET..QS_OFFSET + QK8_0];
    for (out, &value) in qs.iter_mut().zip(x.iter()) {
        *out = libm::roundf(value * inv_d) as i8 as u8;
    }
}

/// Quantizes a run of `f32` weights into `Q8_0` blocks. `input` is
/// borrowed, `output` is caller-provided — no allocation on this path.
///
/// # Errors
/// [`QuantError::InputNotElementMultiple`] if `input.len()` is not a
/// multiple of [`QK8_0`]; [`QuantError::OutputSizeMismatch`] if
/// `output.len()` does not exactly match the packed byte count.
pub fn quantize(input: &[f32], output: &mut [u8]) -> Result<(), QuantError> {
    if !input.len().is_multiple_of(QK8_0) {
        return Err(QuantError::InputNotElementMultiple {
            codec: CODEC,
            unit: "block",
            found: input.len(),
            block_elements: QK8_0,
        });
    }
    let block_count = input.len() / QK8_0;
    let expected = bytes_for_blocks(block_count);
    if output.len() != expected {
        return Err(QuantError::OutputSizeMismatch {
            found: output.len(),
            expected,
        });
    }
    for (chunk, out_block) in input.chunks_exact(QK8_0).zip(output.chunks_exact_mut(BLOCK_BYTES)) {
        quantize_block(chunk, out_block);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use alloc::vec;
    use alloc::vec::Vec;

    use proxima_telemetry::debug;

    use super::{BLOCK_BYTES, CODEC, QK8_0, QuantError, dequantize, quantize};

    /// One block, hand-packed and hand-decoded, checked against the
    /// `x = q*d` formula computed by hand — not by calling
    /// [`super::quantize`] to build the fixture. `d=0.25` is exact in
    /// `f16`, so every expected value below is an exact multiple of
    /// `0.25` in `f32`; `assert_eq!` needs no epsilon.
    #[test]
    fn dequantize_block_matches_hand_packed_fixture() {
        // qs chosen to cover the full signed int8 range, including -128
        // (which does not round-trip through quantize, but is a legal
        // wire byte a decoder must still read correctly) and both
        // positive and negative values.
        let qs: [i8; QK8_0] = [
            0, 1, -1, 127, -127, -128, 64, -64, 5, -5, 100, -100, 3, -3, 50, -50, 2, -2, 90, -90, 10, -10, 60, -60,
            7, -7, 40, -40, 20, -20, 80, -80,
        ];

        let mut block = [0u8; BLOCK_BYTES];
        block[0..2].copy_from_slice(&half::f16::from_f32(0.25).to_le_bytes()); // d
        for (byte, &value) in block[2..2 + QK8_0].iter_mut().zip(qs.iter()) {
            *byte = value as u8;
        }

        let expected: Vec<f32> = qs.iter().map(|&value| f32::from(value) * 0.25).collect();

        let mut output = [0.0f32; QK8_0];
        dequantize(&block, &mut output).expect("well-formed single block");
        assert_eq!(output.as_slice(), expected.as_slice());
    }

    /// All-zero input hits the `amax == 0.0` fast path: `d = 0`, every
    /// level `0`. The round trip must be bit-exact, not merely close.
    #[test]
    fn quantize_dequantize_zero_vector_is_bit_exact() {
        let input = vec![0.0f32; QK8_0];
        let mut packed = vec![0u8; BLOCK_BYTES];
        quantize(&input, &mut packed).expect("one block");
        let mut output = vec![0.0f32; QK8_0];
        dequantize(&packed, &mut output).expect("one block");
        assert_eq!(output, input);
    }

    /// Round-trips a smooth, multi-block, non-degenerate signal and
    /// reports (does not hide) the measured max and RMS error. 8 bits
    /// over a range of roughly `[-3.5, 3.5]` gives a per-level step
    /// around `7.0 / 254 ~= 0.0276`; `0.03` absolute max-error and
    /// `0.02` RMS are loose sanity bounds around that, not tuned to the
    /// measured numbers -- and both are far smaller than q4_K's
    /// equivalent bounds (`0.6`/`0.2`), since q8_0 carries roughly twice
    /// the bits per weight.
    #[test]
    fn quantize_dequantize_smooth_signal_round_trip_error() {
        let elements = QK8_0 * 4;
        let input: Vec<f32> = (0..elements)
            .map(|index| {
                let value = index as f32;
                3.0 * (value * 0.05).sin() + 0.5 * (value * 0.37).cos()
            })
            .collect();
        let mut packed = vec![0u8; BLOCK_BYTES * 4];
        quantize(&input, &mut packed).expect("four blocks");
        let mut output = vec![0.0f32; elements];
        dequantize(&packed, &mut output).expect("four blocks");

        let mut max_error = 0.0f32;
        let mut sum_sq_error = 0.0f64;
        for (got, want) in output.iter().zip(input.iter()) {
            assert!(got.is_finite(), "dequantized value must be finite, got {got}");
            let diff = (got - want).abs();
            max_error = max_error.max(diff);
            sum_sq_error += f64::from(diff) * f64::from(diff);
        }
        let rms_error = (sum_sq_error / elements as f64).sqrt();
        debug!(max_error, rms_error, "quant.q8_0 smooth-signal round trip");
        assert!(max_error < 0.03, "max_error={max_error} exceeds loose sanity bound");
        assert!(rms_error < 0.02, "rms_error={rms_error} exceeds loose sanity bound");
    }

    #[test]
    fn dequantize_rejects_non_block_multiple_length() {
        let data = vec![0u8; BLOCK_BYTES - 1];
        let mut output = vec![0.0f32; QK8_0];
        let error = dequantize(&data, &mut output).unwrap_err();
        assert_eq!(
            error,
            QuantError::InputNotBlockMultiple {
                codec: CODEC,
                found: BLOCK_BYTES - 1,
                block_bytes: BLOCK_BYTES,
            }
        );
    }

    #[test]
    fn dequantize_rejects_output_size_mismatch() {
        let data = vec![0u8; BLOCK_BYTES];
        let mut output = vec![0.0f32; QK8_0 - 1];
        let error = dequantize(&data, &mut output).unwrap_err();
        assert_eq!(
            error,
            QuantError::OutputSizeMismatch {
                found: QK8_0 - 1,
                expected: QK8_0,
            }
        );
    }

    #[test]
    fn quantize_rejects_non_element_multiple_length() {
        let input = vec![0.0f32; QK8_0 - 1];
        let mut output = vec![0u8; BLOCK_BYTES];
        let error = quantize(&input, &mut output).unwrap_err();
        assert_eq!(
            error,
            QuantError::InputNotElementMultiple {
                codec: CODEC,
                unit: "block",
                found: QK8_0 - 1,
                block_elements: QK8_0,
            }
        );
    }

    #[test]
    fn quantize_rejects_output_size_mismatch() {
        let input = vec![0.0f32; QK8_0];
        let mut output = vec![0u8; BLOCK_BYTES - 1];
        let error = quantize(&input, &mut output).unwrap_err();
        assert_eq!(
            error,
            QuantError::OutputSizeMismatch {
                found: BLOCK_BYTES - 1,
                expected: BLOCK_BYTES,
            }
        );
    }

    /// A truncated block run that is neither a whole block nor empty —
    /// typed error, never a panic or an out-of-bounds read.
    #[test]
    fn dequantize_rejects_truncated_partial_block() {
        let partial_bytes = BLOCK_BYTES + BLOCK_BYTES / 2;
        let data = vec![0u8; partial_bytes];
        let mut output = vec![0.0f32; QK8_0];
        let error = dequantize(&data, &mut output).unwrap_err();
        assert_eq!(
            error,
            QuantError::InputNotBlockMultiple {
                codec: CODEC,
                found: partial_bytes,
                block_bytes: BLOCK_BYTES,
            }
        );
    }
}
