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
    assert!(
        layout.block_elements as usize == QK8_0,
        "GgmlType::Q8_0 block_elements drifted from QK8_0"
    );
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
    for (block, out_chunk) in data
        .as_chunks::<BLOCK_BYTES>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<QK8_0>().0)
    {
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
    for (chunk, out_block) in input
        .as_chunks::<QK8_0>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<BLOCK_BYTES>().0)
    {
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
            0, 1, -1, 127, -127, -128, 64, -64, 5, -5, 100, -100, 3, -3, 50, -50, 2, -2, 90, -90,
            10, -10, 60, -60, 7, -7, 40, -40, 20, -20, 80, -80,
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
            assert!(
                got.is_finite(),
                "dequantized value must be finite, got {got}"
            );
            let diff = (got - want).abs();
            max_error = max_error.max(diff);
            sum_sq_error += f64::from(diff) * f64::from(diff);
        }
        let rms_error = (sum_sq_error / elements as f64).sqrt();
        debug!(max_error, rms_error, "quant.q8_0 smooth-signal round trip");
        assert!(
            max_error < 0.03,
            "max_error={max_error} exceeds loose sanity bound"
        );
        assert!(
            rms_error < 0.02,
            "rms_error={rms_error} exceeds loose sanity bound"
        );
    }

    /// A single dominant outlier against an otherwise-zero block: `amax`
    /// is set entirely by index 0, so `d = x0/127` and every other level
    /// should quantize to `0`. Pathological in the sense that a naive
    /// scale-by-RMS codec would waste most of its dynamic range on this
    /// shape; `Q8_0`'s per-block `amax` scale is exactly the right
    /// response.
    #[test]
    fn quantize_dequantize_single_outlier_block() {
        let mut input = vec![0.0f32; QK8_0];
        input[0] = 100.0;
        let mut packed = vec![0u8; BLOCK_BYTES];
        quantize(&input, &mut packed).expect("one block");
        let mut output = vec![0.0f32; QK8_0];
        dequantize(&packed, &mut output).expect("one block");
        let max_error = output
            .iter()
            .zip(input.iter())
            .map(|(got, want)| (got - want).abs())
            .fold(0.0f32, f32::max);
        // bound per Q1: max|dequant(encode(x)) - x| <= scale/2, scale = amax/127
        let scale = 100.0f32 / 127.0;
        assert!(
            max_error <= scale / 2.0 + 1e-4,
            "max_error={max_error} exceeds scale/2={}",
            scale / 2.0
        );
        assert_eq!(
            output[1..],
            input[1..],
            "non-outlier positions must quantize to exactly zero"
        );
    }

    /// Alternating-sign block: every other element flips sign at the same
    /// magnitude, exercising the signed range symmetrically rather than
    /// one-sided.
    #[test]
    fn quantize_dequantize_alternating_sign_block() {
        let input: Vec<f32> = (0..QK8_0)
            .map(|index| if index % 2 == 0 { 2.5 } else { -2.5 })
            .collect();
        let mut packed = vec![0u8; BLOCK_BYTES];
        quantize(&input, &mut packed).expect("one block");
        let mut output = vec![0.0f32; QK8_0];
        dequantize(&packed, &mut output).expect("one block");
        // amax == 2.5 exactly, so every level lands at +-127, but `d =
        // amax/127` itself is NOT exactly representable in f16 -- the
        // bound is scale/2, same as every other Q8_0 round trip, not
        // bit-exact.
        let scale = 2.5f32 / 127.0;
        let max_error = output
            .iter()
            .zip(input.iter())
            .map(|(got, want)| (got - want).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error <= scale / 2.0 + 1e-4, "max_error={max_error}");
    }

    /// Encoding the same input twice must yield byte-identical output --
    /// no hidden nondeterminism (uninitialized scratch, iteration-order
    /// dependent float sums) in the encoder.
    #[test]
    fn quantize_is_deterministic_across_repeated_calls() {
        let elements = QK8_0 * 3;
        let input: Vec<f32> = (0..elements)
            .map(|index| (index as f32 * 0.31).sin() * 7.0)
            .collect();
        let mut first = vec![0u8; BLOCK_BYTES * 3];
        let mut second = vec![0u8; BLOCK_BYTES * 3];
        quantize(&input, &mut first).expect("three blocks");
        quantize(&input, &mut second).expect("three blocks");
        assert_eq!(first, second);
    }

    /// `encode(dequant(encode(x))) == encode(x)`: once a value has landed
    /// on the codec's own grid, re-encoding the dequantized result must
    /// reproduce the exact same bytes -- the grid is idempotent under one
    /// more round trip, even though `x` itself was not on the grid.
    #[test]
    fn quantize_is_idempotent_at_the_codec_grid() {
        let elements = QK8_0 * 3;
        let input: Vec<f32> = (0..elements)
            .map(|index| (index as f32 * 0.17).cos() * 4.0)
            .collect();
        let mut once = vec![0u8; BLOCK_BYTES * 3];
        quantize(&input, &mut once).expect("three blocks");
        let mut on_grid = vec![0.0f32; elements];
        dequantize(&once, &mut on_grid).expect("three blocks");
        let mut twice = vec![0u8; BLOCK_BYTES * 3];
        quantize(&on_grid, &mut twice).expect("three blocks");
        assert_eq!(once, twice);
    }

    /// Real weight data (`token_embd.weight`, dequantized from the
    /// checkpoint's own `Q4_K` bytes, principle 9), re-encoded through
    /// `Q8_0`: max-abs error must sit within `scale/2` per block, and the
    /// measured RMS is logged, not hidden.
    #[cfg(feature = "std")]
    #[test]
    fn quantize_dequantize_real_qwen3_weights_round_trip_error() {
        let input = crate::quant::real_weights::qwen3_token_embd_f32(QK8_0 * 64);
        let blocks = input.len() / QK8_0;
        let mut packed = vec![0u8; BLOCK_BYTES * blocks];
        quantize(&input, &mut packed).expect("real-weight blocks");
        let mut output = vec![0.0f32; input.len()];
        dequantize(&packed, &mut output).expect("real-weight blocks");

        let mut max_error = 0.0f32;
        let mut sum_sq_error = 0.0f64;
        for (chunk_index, chunk) in input.chunks(QK8_0).enumerate() {
            let amax = chunk.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
            let scale = amax / 127.0;
            let out_chunk = &output[chunk_index * QK8_0..(chunk_index + 1) * QK8_0];
            for (got, want) in out_chunk.iter().zip(chunk.iter()) {
                let diff = (got - want).abs();
                assert!(
                    diff <= scale / 2.0 + 1e-4,
                    "block {chunk_index}: diff={diff} exceeds scale/2={}",
                    scale / 2.0
                );
                max_error = max_error.max(diff);
                sum_sq_error += f64::from(diff) * f64::from(diff);
            }
        }
        let rms_error = (sum_sq_error / input.len() as f64).sqrt();
        debug!(
            max_error,
            rms_error, "quant.q8_0 real-qwen3-weights round trip"
        );
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
