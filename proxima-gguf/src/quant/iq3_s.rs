//! `IQ3_S`: ~3.4375-bit non-linear (grid-indexed) weights in 256-element
//! super-blocks, one `f16` super-block scale, eight 4-bit sub-block scales
//! packed two per byte, a 9-bit grid index split across a byte plane and a
//! high-bit plane, and explicit sign bits over 8-element groups.
//! `value = super_scale * (1 + 2*sub_scale) * grid_byte * sign`
//! (`dequantize_row_iq3_s`, `ggml/src/ggml-quants.c:2607-2646` at llama.cpp
//! `f1ea20621` -- the tree on this host; line numbers here are that tree's,
//! the function names are the stable citation).
//!
//! cited on [`dequantize_block`].
//!
//! Layout, from llama.cpp (`/Users/brianbruggeman/repos/others/llama.cpp`,
//! `f1ea20621`): `ggml/src/ggml-common.h:415-422` (typedef 415-421,
//! `static_assert` 422):
//! ```c
//! #define IQ3S_N_SCALE QK_K/64
//! typedef struct {
//!     ggml_half d;
//!     uint8_t qs[QK_K/4];
//!     uint8_t qh[QK_K/32];
//!     uint8_t signs[QK_K/8];
//!     uint8_t scales[IQ3S_N_SCALE];
//! } block_iq3_s;
//! static_assert(sizeof(block_iq3_s) == sizeof(ggml_half) + 13*(QK_K/32) + IQ3S_N_SCALE, "wrong iq3_s block size/padding");
//! ```
//! `QK_K` is 256 (`ggml-common.h:89`), so this is 110 bytes per 256 elements.
//! Cross-checked here at compile time against
//! [`crate::types::GgmlType::Iq3S`]'s already-landed
//! [`crate::types::GgmlType::wire_shape`] rather than re-typed by hand.
//!
//! Per-element decode, sub-block pair by sub-block pair
//! (`ggml-quants.c:2612-2645`): the super-block holds `QK_K/32` (eight)
//! 32-element sub-blocks processed two at a time, the first with
//! `db1 = d * (1 + 2*(scales[ib/2] & 0xf))` and the second with
//! `db2 = d * (1 + 2*(scales[ib/2] >> 4))` -- a POSITIVE bias, unlike the
//! `IQ`-family's signed 6-bit scales. Each 32-element sub-block is four
//! 8-element groups; a group reads two grid entries from
//! [`proxima_numeric::tables::IQ3S_GRID`], the first filling lanes 0..4 and
//! the second lanes 4..8 (`ggml-quants.c:2622-2623`, `:2633-2634`). The 9-bit
//! index is
//! `qs[2*l] | ((qh[0] << (8-2*l)) & 256)` (and `(7-2*l)` for the second
//! entry), so bit `l` of the `qh` byte is the ninth bit of that group's
//! index. Each of the 8 lanes is negated per its own bit in
//! [`proxima_numeric::tables::KMASK_IQ2XS`], applied to that group's own
//! `signs[l]` byte.

use crate::quant::QuantError;
use crate::types::GgmlType;
use proxima_numeric::tables::{IQ3S_GRID, KMASK_IQ2XS};

/// This codec's name as it appears in a rendered [`QuantError`] message.
const CODEC: &str = "iq3_s";

/// Elements per super-block (`ggml-common.h:89`, `#define QK_K 256`).
pub const QK_K: usize = 256;

/// Bytes per super-block. Derived from the already-landed
/// [`crate::types::GgmlType::wire_shape`] (`types.rs`), not re-typed --
/// that table was itself checked against `ggml-common.h`'s
/// `static_assert(sizeof(block_iq3_s) == sizeof(ggml_half) +
/// 13*(QK_K/32) + IQ3S_N_SCALE, ...)` at `ggml-common.h:422`.
pub const BLOCK_BYTES: usize = {
    let layout = GgmlType::Iq3S.wire_shape();
    assert!(
        layout.block_elements as usize == QK_K,
        "GgmlType::Iq3S block_elements drifted from QK_K"
    );
    layout.block_bytes as usize
};

const D_OFFSET: usize = 0;
/// Low eight bits of every grid index, `QK_K/4` bytes (`ggml-quants.c:2614`,
/// `qs = x[i].qs`).
const QS_OFFSET: usize = 2;
/// The ninth index bit per 32-element sub-block, `QK_K/32` bytes
/// (`ggml-quants.c:2615`, `qh = x[i].qh`).
const QH_OFFSET: usize = QS_OFFSET + QK_K / 4;
/// One sign byte per 8-element group, `QK_K/8` bytes (`ggml-quants.c:2616`,
/// `signs = x[i].signs`).
const SIGNS_OFFSET: usize = QH_OFFSET + QK_K / 32;
/// Eight 4-bit sub-block scales, two per byte (`ggml-quants.c:2619-2620`).
const SCALES_OFFSET: usize = SIGNS_OFFSET + QK_K / 8;

/// `QK_K/32` 32-element sub-blocks, each with its own 4-bit scale.
const SUBBLOCK_COUNT: usize = QK_K / 32;

/// The ninth grid-index bit (`| ((qh[..] << ..) & 256)`,
/// `ggml-quants.c:2626-2627`).
const NINTH_INDEX_BIT: usize = 256;

fn f16_at(block: &[u8], offset: usize) -> half::f16 {
    let mut bytes = [0u8; 2];
    bytes.copy_from_slice(&block[offset..offset + 2]);
    half::f16::from_le_bytes(bytes)
}

/// Number of whole `IQ3_S` super-blocks a byte run decodes to, or `None` if
/// `byte_len` is not an exact multiple of [`BLOCK_BYTES`].
#[must_use]
pub const fn blocks_for_bytes(byte_len: usize) -> Option<usize> {
    if byte_len.is_multiple_of(BLOCK_BYTES) {
        Some(byte_len / BLOCK_BYTES)
    } else {
        None
    }
}

/// Exact packed byte length for `block_count` `IQ3_S` super-blocks.
#[must_use]
pub const fn bytes_for_blocks(block_count: usize) -> usize {
    block_count * BLOCK_BYTES
}

/// Exact `f32` element count for `block_count` `IQ3_S` super-blocks.
#[must_use]
pub const fn elements_for_blocks(block_count: usize) -> usize {
    block_count * QK_K
}

/// One 8-element group: two grid entries and one sign byte.
///
/// The two entries carry separate ninth bits (`grid1` from bit `8-2*l` of the
/// widened `qh` byte, `grid2` from bit `7-2*l`,
/// `ggml-quants.c:2626-2627`), so a group's two grid points are addressed
/// independently.
fn decode_group(
    block: &[u8],
    qs_offset: usize,
    grid_high_bits: (usize, usize),
    sign_byte: u8,
    scale: f32,
    output: &mut [f32],
) {
    let low = usize::from(block[qs_offset]) | grid_high_bits.0;
    let high = usize::from(block[qs_offset + 1]) | grid_high_bits.1;
    let grid1 = IQ3S_GRID[low].to_le_bytes();
    let grid2 = IQ3S_GRID[high].to_le_bytes();
    for lane in 0..4 {
        let sign1 = if sign_byte & KMASK_IQ2XS[lane] != 0 {
            -1.0
        } else {
            1.0
        };
        let sign2 = if sign_byte & KMASK_IQ2XS[lane + 4] != 0 {
            -1.0
        } else {
            1.0
        };
        output[lane] = scale * f32::from(grid1[lane]) * sign1;
        output[4 + lane] = scale * f32::from(grid2[lane]) * sign2;
    }
}

/// Dequantizes one 256-element `IQ3_S` super-block. `block` must be exactly
/// [`BLOCK_BYTES`] bytes and `output` exactly [`QK_K`] elements -- callers
/// go through [`dequantize`], which validates both.
///
/// Ports `dequantize_row_iq3_s` (`ggml-quants.c:2607-2646`) exactly -- see
/// this module's own doc for the per-field decode this walks.
pub fn dequantize_block(block: &[u8], output: &mut [f32]) {
    let delta = f16_at(block, D_OFFSET).to_f32();

    for pair in 0..SUBBLOCK_COUNT / 2 {
        let packed_scales = block[SCALES_OFFSET + pair];
        let first_scale = delta * (1.0 + 2.0 * f32::from(packed_scales & 0xf));
        let second_scale = delta * (1.0 + 2.0 * f32::from(packed_scales >> 4));

        for half in 0..2 {
            let scale = if half == 0 { first_scale } else { second_scale };
            let qh = block[QH_OFFSET + pair * 2 + half];
            let qs_base = QS_OFFSET + pair * 16 + half * 8;
            let signs_base = SIGNS_OFFSET + pair * 8 + half * 4;
            let out_base = pair * 64 + half * 32;
            for group in 0..4 {
                let widened = usize::from(qh);
                let grid_high_bits = (
                    (widened << (8 - 2 * group)) & NINTH_INDEX_BIT,
                    (widened << (7 - 2 * group)) & NINTH_INDEX_BIT,
                );
                decode_group(
                    block,
                    qs_base + group * 2,
                    grid_high_bits,
                    block[signs_base + group],
                    scale,
                    &mut output[out_base + group * 8..out_base + (group + 1) * 8],
                );
            }
        }
    }
}

/// Dequantizes a run of `IQ3_S` super-blocks. `data` is borrowed, `output`
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

    /// A block whose scale planes, grid indices and signs are all distinct,
    /// so a swapped plane or an off-by-one group appointment changes the
    /// output.
    fn hand_packed() -> [u8; BLOCK_BYTES] {
        let mut block = [0u8; BLOCK_BYTES];
        block[D_OFFSET..D_OFFSET + 2].copy_from_slice(&half::f16::from_f32(0.25).to_le_bytes());
        // every sub-block scale nibble distinct: scales[0] = 0x10 -> db1 = 0.25, db2 = 0.75
        block[SCALES_OFFSET] = 0x10;
        block[SCALES_OFFSET + 1] = 0x32;
        // index 0 in every qs byte, no ninth bits, all low bytes
        for byte in &mut block[QS_OFFSET..QH_OFFSET] {
            *byte = 0;
        }
        for byte in &mut block[QH_OFFSET..SIGNS_OFFSET] {
            *byte = 0;
        }
        for (index, byte) in block[SIGNS_OFFSET..SCALES_OFFSET].iter_mut().enumerate() {
            *byte = 0xaa ^ (index as u8);
        }
        block
    }

    #[test]
    fn decodes_grid_index_zero_with_the_positive_biased_scale() {
        let block = hand_packed();
        let mut output = [0.0f32; QK_K];
        dequantize_block(&block, &mut output);

        // grid[0] = 0x01010101 -> magnitude 1 in every lane.
        // sub-block 0 uses db1 = 0.25*(1+2*0) = 0.25, group 0 sign byte 0xaa.
        let expected_sign = if KMASK_IQ2XS[0] & 0xaa != 0 {
            -1.0
        } else {
            1.0
        };
        assert_eq!(output[0], 0.25 * 1.0 * expected_sign);
        // sub-block 1 (the second half of the pair) uses db2 = 0.25*(1+2*1) = 0.75.
        let expected_sign = if KMASK_IQ2XS[0] & (0xaa ^ 4) != 0 {
            -1.0
        } else {
            1.0
        };
        assert_eq!(output[32], 0.75 * 1.0 * expected_sign);
    }

    #[test]
    fn the_ninth_index_bit_widens_the_grid_address() {
        let mut block = hand_packed();
        // set qh bit 0 for the first sub-block: group 0's index becomes 256
        block[QH_OFFSET] = 0b0000_0001;
        let mut output = [0.0f32; QK_K];
        dequantize_block(&block, &mut output);
        let grid = IQ3S_GRID[256].to_le_bytes();
        assert_eq!(output[0].abs(), 0.25 * f32::from(grid[0]));
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
