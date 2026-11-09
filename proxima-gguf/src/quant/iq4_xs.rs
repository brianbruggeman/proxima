//! `IQ4_XS`: 4-bit non-linear (table-indexed) weights in 256-element
//! super-blocks, one `f16` super-block scale, eight 6-bit sub-block scales
//! split across a high-bit plane and four low-nibble bytes, and the same
//! 16-entry codebook `IQ4_NL` uses.
//! `value = super_scale * (sub_scale - 32) * kvalues_iq4nl[nibble]`
//! (`dequantize_row_iq4_xs`, `ggml/src/ggml-quants.c:2743-2765` at llama.cpp
//! `f1ea20621` -- the tree on this host; line numbers here are that tree's,
//! the function names are the stable citation), cited on [`dequantize_block`].
//!
//! Layout, from llama.cpp (`/Users/brianbruggeman/repos/others/llama.cpp`,
//! `f1ea20621`): `ggml/src/ggml-common.h:454-460` (typedef 454-459,
//! `static_assert` 460):
//! ```c
//! typedef struct {
//!     ggml_half d;
//!     uint16_t scales_h;
//!     uint8_t  scales_l[QK_K/64];
//!     uint8_t  qs[QK_K/2];
//! } block_iq4_xs;
//! static_assert(sizeof(block_iq4_xs) == sizeof(ggml_half) + sizeof(uint16_t) + QK_K/64 + QK_K/2, "wrong iq4_xs block size/padding");
//! ```
//! `QK_K` is 256 (`ggml-common.h:89`), so this is 136 bytes per 256 elements.
//! Cross-checked here at compile time against
//! [`crate::types::GgmlType::Iq4Xs`]'s already-landed
//! [`crate::types::GgmlType::wire_shape`] rather than re-typed by hand.
//!
//! Per-element decode: the super-block holds `QK_K/32` (eight) 32-element
//! sub-blocks. Sub-block `ib`'s 6-bit scale is
//! `((scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((scales_h >> 2*ib) & 3) << 4)`
//! (`ggml-quants.c:2754`); the scale is biased by 32 rather than signed
//! (`dl = d * (ls - 32)`, `ggml-quants.c:2755`). The sub-block's 16 packed
//! bytes hold two nibbles per byte, the low nibble addressing the first half
//! of the sub-block and the high nibble the second half
//! (`ggml-quants.c:2756-2759`) -- the same half-block nibble split `Q4_0` and
//! [`crate::quant::iq4_nl`] use, with each nibble looked up in
//! [`proxima_numeric::tables::KVALUES_IQ4_NL`] instead of recentred.

use crate::quant::QuantError;
use crate::types::GgmlType;
use proxima_numeric::tables::KVALUES_IQ4_NL;

/// This codec's name as it appears in a rendered [`QuantError`] message.
const CODEC: &str = "iq4_xs";

/// Elements per super-block (`ggml-common.h:89`, `#define QK_K 256`).
pub const QK_K: usize = 256;

/// Bytes per super-block. Derived from the already-landed
/// [`crate::types::GgmlType::wire_shape`] (`types.rs`), not re-typed --
/// that table was itself checked against `ggml-common.h`'s
/// `static_assert(sizeof(block_iq4_xs) == sizeof(ggml_half) +
/// sizeof(uint16_t) + QK_K/64 + QK_K/2, ...)` at `ggml-common.h:460`.
pub const BLOCK_BYTES: usize = {
    let layout = GgmlType::Iq4Xs.wire_shape();
    assert!(
        layout.block_elements as usize == QK_K,
        "GgmlType::Iq4Xs block_elements drifted from QK_K"
    );
    layout.block_bytes as usize
};

const D_OFFSET: usize = 0;
/// The 2-byte high-bit plane of every sub-block scale: `ggml-quants.c:2753`
/// reads `((x[i].scales_h >> 2*ib) & 3) << 4` for sub-block `ib`.
const SCALES_H_OFFSET: usize = 2;
/// Four bytes holding eight 4-bit scale nibbles, two per byte
/// (`x[i].scales_l[ib/2] >> 4*(ib%2)`, `ggml-quants.c:2753`).
const SCALES_L_OFFSET: usize = 4;
/// Nibble region: `QK_K/2` bytes for 256 elements (`ggml-quants.c:2749`,
/// `qs = x[i].qs`).
const QS_OFFSET: usize = SCALES_L_OFFSET + QK_K / 64;

/// `QK_K/32` 32-element sub-blocks, each with its own 6-bit scale and 16
/// packed nibble bytes.
const SUBBLOCK_COUNT: usize = QK_K / 32;

/// Elements a sub-block's packed region covers -- two nibbles per byte
/// (`ggml-quants.c:2752-2759`).
const SUBBLOCK_ELEMENTS: usize = 32;

/// The scale bias every sub-block subtracts (`dl = d * (ls - 32)`,
/// `ggml-quants.c:2754`).
const SCALE_BIAS: i32 = 32;

fn f16_at(block: &[u8], offset: usize) -> half::f16 {
    let mut bytes = [0u8; 2];
    bytes.copy_from_slice(&block[offset..offset + 2]);
    half::f16::from_le_bytes(bytes)
}

fn u16_at(block: &[u8], offset: usize) -> u16 {
    let mut bytes = [0u8; 2];
    bytes.copy_from_slice(&block[offset..offset + 2]);
    u16::from_le_bytes(bytes)
}

/// Number of whole `IQ4_XS` super-blocks a byte run decodes to, or `None`
/// if `byte_len` is not an exact multiple of [`BLOCK_BYTES`].
#[must_use]
pub const fn blocks_for_bytes(byte_len: usize) -> Option<usize> {
    if byte_len.is_multiple_of(BLOCK_BYTES) {
        Some(byte_len / BLOCK_BYTES)
    } else {
        None
    }
}

/// Exact packed byte length for `block_count` `IQ4_XS` super-blocks.
#[must_use]
pub const fn bytes_for_blocks(block_count: usize) -> usize {
    block_count * BLOCK_BYTES
}

/// Exact `f32` element count for `block_count` `IQ4_XS` super-blocks.
#[must_use]
pub const fn elements_for_blocks(block_count: usize) -> usize {
    block_count * QK_K
}

/// Dequantizes one 256-element `IQ4_XS` super-block. `block` must be
/// exactly [`BLOCK_BYTES`] bytes and `output` exactly [`QK_K`] elements --
/// callers go through [`dequantize`], which validates both.
///
/// Ports `dequantize_row_iq4_xs` (`ggml-quants.c:2743-2765`) exactly -- see
/// this module's own doc for the per-field decode this walks.
pub fn dequantize_block(block: &[u8], output: &mut [f32]) {
    let delta = f16_at(block, D_OFFSET).to_f32();
    let scales_h = u16_at(block, SCALES_H_OFFSET);

    for sub_block in 0..SUBBLOCK_COUNT {
        let low_nibble = (block[SCALES_L_OFFSET + sub_block / 2] >> (4 * (sub_block % 2))) & 0xf;
        let high_bits = ((scales_h >> (2 * sub_block)) & 3) as u8;
        let scale_index = f32::from(low_nibble | (high_bits << 4));
        let sub_scale = delta * (scale_index - SCALE_BIAS as f32);

        let packed = &block[QS_OFFSET + sub_block * (SUBBLOCK_ELEMENTS / 2)
            ..QS_OFFSET + (sub_block + 1) * (SUBBLOCK_ELEMENTS / 2)];
        let out_base = sub_block * SUBBLOCK_ELEMENTS;
        let half_block = SUBBLOCK_ELEMENTS / 2;
        for (index, byte) in packed.iter().enumerate() {
            let low = usize::from(byte & 0xf);
            let high = usize::from(byte >> 4);
            output[out_base + index] = sub_scale * f32::from(KVALUES_IQ4_NL[low]);
            output[out_base + half_block + index] = sub_scale * f32::from(KVALUES_IQ4_NL[high]);
        }
    }
}

/// Dequantizes a run of `IQ4_XS` super-blocks. `data` is borrowed, `output`
/// is caller-provided -- no allocation on this path.
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
        .zip(output.as_chunks_mut::<{ QK_K }>().0)
    {
        dequantize_block(block, out_chunk);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    /// A hand-packed block whose every field is a distinct value, so a
    /// field swap or an off-by-one in the scale plane shows up as a wrong
    /// element rather than a plausible one.
    fn hand_packed() -> [u8; BLOCK_BYTES] {
        let mut block = [0u8; BLOCK_BYTES];
        block[D_OFFSET..D_OFFSET + 2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
        // sub-block 0: scale nibble 0 with high bits 3 -> ls = 48 -> dl = 8.0
        // sub-block 1: scale nibble 15 with high bits 0 -> ls = 15 -> dl = -8.5
        block[SCALES_L_OFFSET] = 0xf0;
        block[SCALES_H_OFFSET..SCALES_H_OFFSET + 2]
            .copy_from_slice(&0b0000_0000_0000_0011u16.to_le_bytes());
        for (index, byte) in block[QS_OFFSET..QS_OFFSET + QK_K / 2]
            .iter_mut()
            .enumerate()
        {
            *byte = (index % 16) as u8 | (((index + 8) % 16) as u8) << 4;
        }
        block
    }

    #[test]
    fn decodes_the_first_subblock_from_the_biased_scale_and_the_nibble_codebook() {
        let block = hand_packed();
        let mut output = [0.0f32; QK_K];
        dequantize_block(&block, &mut output);

        // sub-block 0: ls = 0 | (3 << 4) = 48 -> dl = 0.5 * (48 - 32) = 8.0
        // sub-block 1: ls = 15 | 0 = 15 -> dl = 0.5 * (15 - 32) = -8.5
        assert_eq!(output[0], 8.0 * -127.0);
        assert_eq!(output[16], 8.0 * 1.0);
        assert_eq!(output[32], -8.5 * -127.0);
        assert_eq!(output[48], -8.5 * 1.0);
    }

    #[test]
    fn a_zero_scale_nibble_is_the_bias_not_a_zero_scale() {
        let mut block = hand_packed();
        block[SCALES_H_OFFSET..SCALES_H_OFFSET + 2].copy_from_slice(&0u16.to_le_bytes());
        block[SCALES_L_OFFSET] = 0x00;
        let mut output = [0.0f32; QK_K];
        dequantize_block(&block, &mut output);
        // ls = 0 keeps the -32 bias, so the value is non-zero -- the
        // difference between a biased 6-bit scale and a plain one.
        assert_eq!(output[0], -16.0 * -127.0);
    }

    #[test]
    fn rejects_a_short_output_and_a_non_multiple_input() {
        let block = hand_packed();
        let error = dequantize(&block, &mut [0.0f32; 8]).expect_err("short output is refused");
        assert_eq!(
            error,
            QuantError::OutputSizeMismatch {
                found: 8,
                expected: QK_K
            }
        );
        let error = dequantize(&block[..BLOCK_BYTES - 1], &mut [0.0f32; QK_K])
            .expect_err("a partial block is refused");
        assert_eq!(
            error,
            QuantError::InputNotBlockMultiple {
                codec: CODEC,
                found: BLOCK_BYTES - 1,
                block_bytes: BLOCK_BYTES
            }
        );
    }
}
