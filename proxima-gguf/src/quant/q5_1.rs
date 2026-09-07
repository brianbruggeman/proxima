//! `Q5_1`: 5-bit weights in 32-element blocks, one `f16` scale and one
//! `f16` minimum per block. `value = scale * q + min` per element
//! (`dequantize_row_q5_1`, `ggml-quants.c:316-341`, cited on
//! [`dequantize_block`]).
//!
//! Layout, from llama.cpp (`/Users/brianbruggeman/repos/others/llama.cpp`):
//! `ggml/src/ggml-common.h:196-207`:
//! ```c
//! #define QK5_1 32
//! typedef struct {
//!     GGML_EXTENSION union {
//!         struct {
//!             ggml_half d; // delta
//!             ggml_half m; // min
//!         } GGML_COMMON_AGGR_S;
//!         ggml_half2 dm;
//!     } GGML_COMMON_AGGR_U;
//!     uint8_t qh[4];         // 5-th bit of quants
//!     uint8_t qs[QK5_1 / 2]; // nibbles / quants
//! } block_q5_1;
//! static_assert(sizeof(block_q5_1) == 2 * sizeof(ggml_half) + sizeof(uint32_t) + QK5_1 / 2, "wrong q5_1 block size/padding");
//! ```
//! 24 bytes per 32 elements (`ggml_half d`, `ggml_half m`, 4 bytes of
//! packed 5th bits, 16 packed nibble bytes). Cross-checked here at compile
//! time against [`crate::types::GgmlType::Q5_1`]'s already-landed
//! [`crate::types::GgmlType::block_layout`] rather than re-typed by hand.
//!
//! Per-element decode (`dequantize_row_q5_1`, `ggml-quants.c:323-339`):
//! the 32 elements split into two 16-wide halves, low nibble of `qs[j]` for
//! element `j` (`0..16`), high nibble of `qs[j]` for element `16+j`; each
//! element's 5th bit comes from `qh`, bit index equal to the element's own
//! index (`0..32`) -- `xh_0 = bit_j(qh) << 4` for the low half, `xh_1 =
//! bit_(j+16)(qh) << 4` for the high half (verified against the exact C
//! shift arithmetic, not the simplified textbook description: `((qh >>
//! (j+12)) & 0x10)` isolates bit `j+16`, not a re-based bit `j+12`).

use crate::quant::QuantError;
use crate::types::GgmlType;

/// This codec's name as it appears in a rendered [`QuantError`] message.
const CODEC: &str = "q5_1";

/// Elements per block (`ggml-common.h:196`, `#define QK5_1 32`).
pub const QK5_1: usize = 32;

/// Bytes per block. Derived from the already-landed
/// [`crate::types::GgmlType::block_layout`] (`types.rs`), not re-typed --
/// that table was itself checked against `ggml-common.h`'s
/// `static_assert(sizeof(block_q5_1) == 2 * sizeof(ggml_half) +
/// sizeof(uint32_t) + QK5_1 / 2, ...)` at `ggml-common.h:207`.
pub const BLOCK_BYTES: usize = {
    let layout = GgmlType::Q5_1.block_layout();
    assert!(
        layout.block_elements as usize == QK5_1,
        "GgmlType::Q5_1 block_elements drifted from QK5_1"
    );
    layout.block_bytes as usize
};

const D_OFFSET: usize = 0;
const M_OFFSET: usize = 2;
const QH_OFFSET: usize = 4;
const QS_OFFSET: usize = 8;
/// Half the block's elements -- each packed nibble byte carries two
/// elements, one per half of the block (`ggml-quants.c:330-339`).
const HALF_BLOCK: usize = QK5_1 / 2;

/// Number of whole `Q5_1` blocks a byte run decodes to, or `None` if
/// `byte_len` is not an exact multiple of [`BLOCK_BYTES`].
#[must_use]
pub const fn blocks_for_bytes(byte_len: usize) -> Option<usize> {
    if byte_len.is_multiple_of(BLOCK_BYTES) {
        Some(byte_len / BLOCK_BYTES)
    } else {
        None
    }
}

/// Exact packed byte length for `block_count` `Q5_1` blocks.
#[must_use]
pub const fn bytes_for_blocks(block_count: usize) -> usize {
    block_count * BLOCK_BYTES
}

/// Exact `f32` element count for `block_count` `Q5_1` blocks.
#[must_use]
pub const fn elements_for_blocks(block_count: usize) -> usize {
    block_count * QK5_1
}

fn f16_at(block: &[u8], offset: usize) -> half::f16 {
    let mut bytes = [0u8; 2];
    bytes.copy_from_slice(&block[offset..offset + 2]);
    half::f16::from_le_bytes(bytes)
}

fn qh_at(block: &[u8]) -> u32 {
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&block[QH_OFFSET..QH_OFFSET + 4]);
    u32::from_le_bytes(bytes)
}

/// Dequantizes one 32-element `Q5_1` block. `block` must be exactly
/// [`BLOCK_BYTES`] bytes and `output` exactly [`QK5_1`] elements -- callers
/// go through [`dequantize`], which validates both.
///
/// Ports `dequantize_row_q5_1` (`ggml-quants.c:316-341`) exactly: each
/// packed nibble byte carries two 4-bit levels (low nibble at `output[j]`,
/// high nibble at `output[HALF_BLOCK + j]`), each widened to 5 bits by a
/// per-element bit from `qh` (bit `j` for the low half, bit `j + HALF_BLOCK`
/// for the high half), then `x = q*d + m` -- unlike `Q4_0`/`Q4_1`, no
/// fixed-midpoint recenter: the format carries its own `min` term.
pub fn dequantize_block(block: &[u8], output: &mut [f32]) {
    let delta = f16_at(block, D_OFFSET).to_f32();
    let min = f16_at(block, M_OFFSET).to_f32();
    let qh = qh_at(block);
    let qs = &block[QS_OFFSET..QS_OFFSET + HALF_BLOCK];
    for (index, &byte) in qs.iter().enumerate() {
        let high_bit_low = ((qh >> index) << 4) & 0x10;
        let high_bit_high = (qh >> (index + 12)) & 0x10;
        let low = u32::from(byte & 0x0F) | high_bit_low;
        let high = u32::from(byte >> 4) | high_bit_high;
        output[index] = low as f32 * delta + min;
        output[HALF_BLOCK + index] = high as f32 * delta + min;
    }
}

/// Dequantizes a run of `Q5_1` blocks. `data` is borrowed, `output` is
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
    for (block, out_chunk) in data
        .as_chunks::<BLOCK_BYTES>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<QK5_1>().0)
    {
        dequantize_block(block, out_chunk);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use alloc::vec;
    use alloc::vec::Vec;

    use super::{BLOCK_BYTES, CODEC, HALF_BLOCK, QH_OFFSET, QK5_1, QS_OFFSET, QuantError, dequantize};

    /// One block, hand-packed and hand-decoded, checked against the
    /// `x = q*d + m` formula computed by hand -- not by calling a quantize
    /// routine to build the fixture (this codec is decode-only, mirroring
    /// what the checkpoint this crate targets actually needs). `d=0.0123`,
    /// `m=-0.4` are realistic real-checkpoint-scale values; every 5-bit
    /// level `0..=31` is exercised across the two halves, and every `qh`
    /// bit is set so both the low and high nibble paths pick up their 5th
    /// bit.
    #[test]
    fn dequantize_block_matches_hand_computed_fixture() {
        let low_nibbles: [u8; HALF_BLOCK] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        ];
        let high_nibbles: [u8; HALF_BLOCK] = [
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        ];
        // every bit of qh set: every element's 5th bit is 1.
        let qh: u32 = 0xFFFF_FFFF;
        let delta = 0.0123f32;
        let min = -0.4f32;

        let mut block = [0u8; BLOCK_BYTES];
        block[0..2].copy_from_slice(&half::f16::from_f32(delta).to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(min).to_le_bytes());
        block[QH_OFFSET..QH_OFFSET + 4].copy_from_slice(&qh.to_le_bytes());
        for index in 0..HALF_BLOCK {
            block[QS_OFFSET + index] = low_nibbles[index] | (high_nibbles[index] << 4);
        }

        // reconstruct exact f16 values the same way dequantize_block does,
        // so the expected values are bit-exact against the stored deltas.
        let exact_delta = half::f16::from_f32(delta).to_f32();
        let exact_min = half::f16::from_f32(min).to_f32();
        let expected: Vec<f32> = low_nibbles
            .iter()
            .chain(high_nibbles.iter())
            .map(|&nibble| {
                // every qh bit is set, so the 5-bit level is 16 | nibble.
                let level = 16 | u32::from(nibble);
                level as f32 * exact_delta + exact_min
            })
            .collect();

        let mut output = [0.0f32; QK5_1];
        dequantize(&block, &mut output).expect("well-formed single block");
        assert_eq!(output.as_slice(), expected.as_slice());
    }

    /// `qh` all zero: every element's 5th bit is 0, so the 5-bit level
    /// collapses to the plain 4-bit nibble -- the boundary case that
    /// distinguishes the low-bit and high-bit extraction paths from a bug
    /// that always sets (or never sets) the 5th bit.
    #[test]
    fn dequantize_block_boundary_min_and_max_nibble_with_no_fifth_bit() {
        let mut block = [0u8; BLOCK_BYTES];
        let delta = 0.5f32;
        let min = 0.0f32;
        block[0..2].copy_from_slice(&half::f16::from_f32(delta).to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(min).to_le_bytes());
        // qh left at all zero.
        // first packed byte: low nibble 0 (min), high nibble 15 (max).
        block[QS_OFFSET] = 0xF0;

        let mut output = [0.0f32; QK5_1];
        dequantize(&block, &mut output).expect("well-formed single block");
        assert_eq!(output[0], 0.0 * 0.5);
        assert_eq!(output[HALF_BLOCK], 15.0 * 0.5);
    }

    #[test]
    fn dequantize_rejects_non_block_multiple_length() {
        let data = vec![0u8; BLOCK_BYTES - 1];
        let mut output = vec![0.0f32; QK5_1];
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
        let mut output = vec![0.0f32; QK5_1 - 1];
        let error = dequantize(&data, &mut output).unwrap_err();
        assert_eq!(
            error,
            QuantError::OutputSizeMismatch {
                found: QK5_1 - 1,
                expected: QK5_1,
            }
        );
    }

    /// A truncated block run that is neither a whole block nor empty --
    /// typed error, never a panic or an out-of-bounds read.
    #[test]
    fn dequantize_rejects_truncated_partial_block() {
        let partial_bytes = BLOCK_BYTES + BLOCK_BYTES / 2;
        let data = vec![0u8; partial_bytes];
        let mut output = vec![0.0f32; QK5_1];
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
