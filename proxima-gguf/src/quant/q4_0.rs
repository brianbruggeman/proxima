//! `Q4_0`: 4-bit signed-ish weights in 32-element blocks, one `f16` scale
//! per block, no sub-block scales, no minimum term. `value = scale *
//! (nibble - 8)` per element (`dequantize_row_q4_0`,
//! `ggml-quants.c:249-267`, cited on [`dequantize_block`]).
//!
//! Layout, from llama.cpp (`/Users/brianbruggeman/repos/others/llama.cpp`):
//! `ggml/src/ggml-common.h:167-172`:
//! ```c
//! #define QK4_0 32
//! typedef struct {
//!     ggml_half d;           // delta
//!     uint8_t qs[QK4_0 / 2]; // nibbles / quants
//! } block_q4_0;
//! static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2, "wrong q4_0 block size/padding");
//! ```
//! 18 bytes per 32 elements (`ggml_half d`, 16 packed nibble bytes). Cross-
//! checked here at compile time against
//! [`crate::types::GgmlType::Q4_0`]'s already-landed
//! [`crate::types::GgmlType::block_layout`] rather than re-typed by hand --
//! this is the simplest legacy quant format this crate carries: no
//! sub-block scale/min hierarchy the way `Q4_K`/`Q5_K`/`Q6_K` have, and no
//! second `min` field the way `Q4_1` (`ggml-common.h:174-182`) has.

use crate::quant::QuantError;
use crate::types::GgmlType;

/// This codec's name as it appears in a rendered [`QuantError`] message.
const CODEC: &str = "q4_0";

/// Elements per block (`ggml-common.h:167`, `#define QK4_0 32`).
pub const QK4_0: usize = 32;

/// Bytes per block. Derived from the already-landed
/// [`crate::types::GgmlType::block_layout`] (`types.rs`), not re-typed --
/// that table was itself checked against `ggml-common.h`'s
/// `static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2, ...)`
/// at `ggml-common.h:172`.
pub const BLOCK_BYTES: usize = {
    let layout = GgmlType::Q4_0.block_layout();
    assert!(layout.block_elements as usize == QK4_0, "GgmlType::Q4_0 block_elements drifted from QK4_0");
    layout.block_bytes as usize
};

const D_OFFSET: usize = 0;
const QS_OFFSET: usize = 2;
/// Half the block's elements -- each packed byte carries two nibbles, one
/// per half of the block (`x[j]` in the low nibble, `x[QK4_0/2 + j]` in the
/// high nibble; `dequantize_row_q4_0`, `ggml-quants.c:259-264`).
const HALF_BLOCK: usize = QK4_0 / 2;

/// Number of whole `Q4_0` blocks a byte run decodes to, or `None` if
/// `byte_len` is not an exact multiple of [`BLOCK_BYTES`].
#[must_use]
pub const fn blocks_for_bytes(byte_len: usize) -> Option<usize> {
    if byte_len.is_multiple_of(BLOCK_BYTES) {
        Some(byte_len / BLOCK_BYTES)
    } else {
        None
    }
}

/// Exact packed byte length for `block_count` `Q4_0` blocks.
#[must_use]
pub const fn bytes_for_blocks(block_count: usize) -> usize {
    block_count * BLOCK_BYTES
}

/// Exact `f32` element count for `block_count` `Q4_0` blocks.
#[must_use]
pub const fn elements_for_blocks(block_count: usize) -> usize {
    block_count * QK4_0
}

fn f16_at(block: &[u8], offset: usize) -> half::f16 {
    let mut bytes = [0u8; 2];
    bytes.copy_from_slice(&block[offset..offset + 2]);
    half::f16::from_le_bytes(bytes)
}

/// Dequantizes one 32-element `Q4_0` block. `block` must be exactly
/// [`BLOCK_BYTES`] bytes and `output` exactly [`QK4_0`] elements -- callers
/// go through [`dequantize`], which validates both.
///
/// Ports `dequantize_row_q4_0` (`ggml-quants.c:249-267`) exactly: each
/// packed byte carries two 4-bit levels, low nibble at `output[j]` and high
/// nibble at `output[HALF_BLOCK + j]`, each recentered by subtracting 8
/// (the format has no separate `min` field -- the bias is the fixed
/// midpoint of the 4-bit range) then scaled by the block's single `f16`
/// delta.
pub fn dequantize_block(block: &[u8], output: &mut [f32]) {
    let d = f16_at(block, D_OFFSET).to_f32();
    let qs = &block[QS_OFFSET..QS_OFFSET + HALF_BLOCK];
    for (index, &byte) in qs.iter().enumerate() {
        let low = i32::from(byte & 0x0F) - 8;
        let high = i32::from(byte >> 4) - 8;
        output[index] = low as f32 * d;
        output[HALF_BLOCK + index] = high as f32 * d;
    }
}

/// Dequantizes a run of `Q4_0` blocks. `data` is borrowed, `output` is
/// caller-provided -- no allocation on this path.
///
/// # Errors
/// [`QuantError::InputNotBlockMultiple`] if `data.len()` is not a multiple
/// of [`BLOCK_BYTES`]; [`QuantError::OutputSizeMismatch`] if `output.len()`
/// does not exactly match the decoded element count.
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
    for (block, out_chunk) in data.as_chunks::<BLOCK_BYTES>().0.iter().zip(output.as_chunks_mut::<QK4_0>().0) {
        dequantize_block(block, out_chunk);
    }
    Ok(())
}

/// Quantizes one 32-element chunk into a `Q4_0` block.
///
/// Ports `quantize_row_q4_0_ref` (`ggml-quants.c:25-59`) exactly: `max` is
/// the SIGNED value achieving the block's largest magnitude (not `amax`
/// itself), `d = max / -8` so that element scales to exactly the nibble
/// `0` (`0 - 8 == -8`, and `-8 * (max/-8) == max`), each other level is
/// `(int8_t)(x*id + 8.5)` clamped above at `15` (no explicit lower clamp --
/// llama.cpp's own reference does not add one, trusting `max`'s own
/// derivation to keep every `x*id + 8.5` inside `[0, 16.5]`), and the two
/// nibbles for position `j` and `HALF_BLOCK + j` pack into one byte with
/// the first in the low bits.
fn quantize_block(x: &[f32], output: &mut [u8]) {
    let mut amax = 0.0f32;
    let mut max = 0.0f32;
    for &value in x {
        if amax < value.abs() {
            amax = value.abs();
            max = value;
        }
    }
    let d = max / -8.0;
    let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };

    let block_scale = half::f16::from_f32(d);
    output[D_OFFSET..D_OFFSET + 2].copy_from_slice(&block_scale.to_le_bytes());

    let qs = &mut output[QS_OFFSET..QS_OFFSET + HALF_BLOCK];
    for (index, out) in qs.iter_mut().enumerate() {
        let x0 = x[index] * inv_d;
        let x1 = x[HALF_BLOCK + index] * inv_d;
        let level0 = 15i8.min((x0 + 8.5) as i8);
        let level1 = 15i8.min((x1 + 8.5) as i8);
        *out = (level0 as u8) | ((level1 as u8) << 4);
    }
}

/// Quantizes a run of `f32` weights into `Q4_0` blocks. `input` is
/// borrowed, `output` is caller-provided -- no allocation on this path.
///
/// # Errors
/// [`QuantError::InputNotElementMultiple`] if `input.len()` is not a
/// multiple of [`QK4_0`]; [`QuantError::OutputSizeMismatch`] if
/// `output.len()` does not exactly match the packed byte count.
pub fn quantize(input: &[f32], output: &mut [u8]) -> Result<(), QuantError> {
    if !input.len().is_multiple_of(QK4_0) {
        return Err(QuantError::InputNotElementMultiple {
            codec: CODEC,
            unit: "block",
            found: input.len(),
            block_elements: QK4_0,
        });
    }
    let block_count = input.len() / QK4_0;
    let expected = bytes_for_blocks(block_count);
    if output.len() != expected {
        return Err(QuantError::OutputSizeMismatch {
            found: output.len(),
            expected,
        });
    }
    for (chunk, out_block) in input.as_chunks::<QK4_0>().0.iter().zip(output.as_chunks_mut::<BLOCK_BYTES>().0) {
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

    use super::{BLOCK_BYTES, CODEC, HALF_BLOCK, QK4_0, QS_OFFSET, QuantError, dequantize, quantize};

    /// One block, hand-packed and hand-decoded, checked against the
    /// `x = (nibble - 8) * d` formula computed by hand -- not by calling
    /// [`super::quantize`] to build the fixture. `d = 0.5` is exact in
    /// `f16`, so every expected value below is an exact multiple of `0.5`
    /// in `f32`; `assert_eq!` needs no epsilon.
    #[test]
    fn dequantize_block_matches_hand_packed_fixture() {
        // nibbles chosen to cover the full 4-bit range (0..=15) across both
        // halves of the block, low nibble first / high nibble second per
        // packed byte.
        let low: [u8; HALF_BLOCK] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let high: [u8; HALF_BLOCK] = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];

        let mut block = [0u8; BLOCK_BYTES];
        block[0..2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes()); // d
        for index in 0..HALF_BLOCK {
            block[QS_OFFSET + index] = low[index] | (high[index] << 4);
        }

        let expected: Vec<f32> = low
            .iter()
            .chain(high.iter())
            .map(|&nibble| (i32::from(nibble) - 8) as f32 * 0.5)
            .collect();

        let mut output = [0.0f32; QK4_0];
        dequantize(&block, &mut output).expect("well-formed single block");
        assert_eq!(output.as_slice(), expected.as_slice());
    }

    /// All-zero input hits the `amax == 0.0` fast path: `d = 0`, every
    /// level rounds to nibble `8` (`8 - 8 == 0`). The round trip must be
    /// bit-exact, not merely close.
    #[test]
    fn quantize_dequantize_zero_vector_is_bit_exact() {
        let input = vec![0.0f32; QK4_0];
        let mut packed = vec![0u8; BLOCK_BYTES];
        quantize(&input, &mut packed).expect("one block");
        let mut output = vec![0.0f32; QK4_0];
        dequantize(&packed, &mut output).expect("one block");
        assert_eq!(output, input);
    }

    /// Round-trips a smooth, multi-block, non-degenerate signal. Rather
    /// than a tuned epsilon, the bound asserted here is derived from the
    /// format's own math: `quantize_block`'s nibble search is a
    /// round-to-nearest quantizer in the `x * inv_d` domain (`(x*inv_d +
    /// 8.5) as i8` truncates a positive value, which is `round(x*inv_d +
    /// 8)`), so its worst-case error in that domain is half a level
    /// (`0.5`), and translating back through `x = level * d` bounds the
    /// worst-case error in `x`-space at `0.5 * d` for whichever block's own
    /// `d` is largest. `d` itself is computed per block exactly as
    /// [`super::quantize_block`] computes it (`max / -8`) so the bound is
    /// the actual analytic ceiling, not a guess.
    #[test]
    fn quantize_dequantize_smooth_signal_round_trip_error() {
        let blocks = 4;
        let elements = QK4_0 * blocks;
        let input: Vec<f32> = (0..elements)
            .map(|index| {
                let value = index as f32;
                3.0 * (value * 0.05).sin() + 0.5 * (value * 0.37).cos()
            })
            .collect();

        let mut analytic_max_error = 0.0f32;
        for chunk in input.chunks(QK4_0) {
            let mut amax = 0.0f32;
            let mut max = 0.0f32;
            for &value in chunk {
                if amax < value.abs() {
                    amax = value.abs();
                    max = value;
                }
            }
            let d = (max / -8.0).abs();
            analytic_max_error = analytic_max_error.max(0.5 * d);
        }
        // float slop for the f16 round trip the scale itself takes.
        let analytic_max_error = analytic_max_error * 1.02 + 1e-4;

        let mut packed = vec![0u8; BLOCK_BYTES * blocks];
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
        debug!(max_error, rms_error, analytic_max_error, "quant.q4_0 smooth-signal round trip");
        assert!(
            max_error <= analytic_max_error,
            "max_error={max_error} exceeds the format's own analytic bound {analytic_max_error}"
        );
    }

    #[test]
    fn dequantize_rejects_non_block_multiple_length() {
        let data = vec![0u8; BLOCK_BYTES - 1];
        let mut output = vec![0.0f32; QK4_0];
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
        let mut output = vec![0.0f32; QK4_0 - 1];
        let error = dequantize(&data, &mut output).unwrap_err();
        assert_eq!(
            error,
            QuantError::OutputSizeMismatch {
                found: QK4_0 - 1,
                expected: QK4_0,
            }
        );
    }

    #[test]
    fn quantize_rejects_non_element_multiple_length() {
        let input = vec![0.0f32; QK4_0 - 1];
        let mut output = vec![0u8; BLOCK_BYTES];
        let error = quantize(&input, &mut output).unwrap_err();
        assert_eq!(
            error,
            QuantError::InputNotElementMultiple {
                codec: CODEC,
                unit: "block",
                found: QK4_0 - 1,
                block_elements: QK4_0,
            }
        );
    }

    #[test]
    fn quantize_rejects_output_size_mismatch() {
        let input = vec![0.0f32; QK4_0];
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

    /// A truncated block run that is neither a whole block nor empty --
    /// typed error, never a panic or an out-of-bounds read.
    #[test]
    fn dequantize_rejects_truncated_partial_block() {
        let partial_bytes = BLOCK_BYTES + BLOCK_BYTES / 2;
        let data = vec![0u8; partial_bytes];
        let mut output = vec![0.0f32; QK4_0];
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
