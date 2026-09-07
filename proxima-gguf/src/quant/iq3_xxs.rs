//! `IQ3_XXS`: ~3.0625-bit non-linear (grid-indexed) weights in 256-element
//! super-blocks, one `f16` scale per super-block, 8 sub-block 4-bit scales
//! packed into a 32-bit aux word alongside a 7-bit sign field per group, and
//! a 256-entry codebook shared with no other format's own grid.
//! `value = scale_for_subblock * grid_byte * sign`
//! (`dequantize_row_iq3_xxs`, `ggml-quants.c:2278-2306`, cited on
//! [`dequantize_block`]).
//!
//! Layout, from llama.cpp (`/Users/brianbruggeman/repos/others/llama.cpp`):
//! `ggml/src/ggml-common.h:362-369`:
//! ```c
//! typedef struct {
//!     ggml_half d;
//!     uint8_t qs[3*QK_K/8];
//! } block_iq3_xxs;
//! static_assert(sizeof(block_iq3_xxs) == sizeof(ggml_half) + 3*(QK_K/8), "wrong iq3_xxs block size/padding");
//! ```
//! `QK_K` is 256 (`ggml-common.h:89`), so this is 98 bytes per 256 elements
//! (`ggml_half d`, then a 96-byte `qs` region split by the decoder itself
//! into a 64-byte grid-index region followed by a 32-byte scale-and-sign
//! region -- the struct declares one flat `qs[96]` array, not two separate
//! fields). Cross-checked here at compile time against
//! [`crate::types::GgmlType::Iq3Xxs`]'s already-landed
//! [`crate::types::GgmlType::block_layout`] rather than re-typed by hand.
//!
//! Per-element decode (`dequantize_row_iq3_xxs`, `ggml-quants.c:2287-2305`):
//! `qs[0..64]` holds 64 single-byte indices into
//! [`crate::quant::tables::iq3xxs_grid::IQ3XXS_GRID`] (a FULL byte index,
//! unlike `IQ2_XS`'s 9-bit field -- the grid only has 256 entries here);
//! `qs[64..96]` holds 8 packed 32-bit little-endian words, one per
//! sub-block, whose top 4 bits are the sub-block's scale nibble (`db = d *
//! (0.5 + nibble) * 0.5`) and whose low 28 bits carry four 7-bit sign
//! indices (one per 8-element group) into the SAME
//! [`crate::quant::tables::iq2xs_grid::KSIGNS_IQ2XS`] table `IQ2_XS` uses
//! (`ggml-quants.c:2294` calls `ksigns_iq2xs` directly -- not a distinct
//! `IQ3_XXS`-owned sign table). Each group reads TWO grid entries (4 bytes
//! each), the first filling lanes `0..4`, the second lanes `4..8`, each
//! byte negated per its own bit in the shared sign lookup.

use crate::quant::QuantError;
use crate::quant::tables::iq2xs_grid::{KMASK_IQ2XS, KSIGNS_IQ2XS};
use crate::quant::tables::iq3xxs_grid::IQ3XXS_GRID;
use crate::types::GgmlType;

/// This codec's name as it appears in a rendered [`QuantError`] message.
const CODEC: &str = "iq3_xxs";

/// Elements per super-block (`ggml-common.h:89`, `#define QK_K 256`).
pub const QK_K: usize = 256;

/// Bytes per super-block. Derived from the already-landed
/// [`crate::types::GgmlType::block_layout`] (`types.rs`), not re-typed --
/// that table was itself checked against `ggml-common.h`'s
/// `static_assert(sizeof(block_iq3_xxs) == sizeof(ggml_half) +
/// 3*(QK_K/8), ...)` at `ggml-common.h:369`.
pub const BLOCK_BYTES: usize = {
    let layout = GgmlType::Iq3Xxs.block_layout();
    assert!(
        layout.block_elements as usize == QK_K,
        "GgmlType::Iq3Xxs block_elements drifted from QK_K"
    );
    layout.block_bytes as usize
};

const D_OFFSET: usize = 0;
const QS_OFFSET: usize = 2;
/// Grid-index region: `QK_K/4` single-byte indices (`ggml-quants.c:2287`,
/// `qs = x[i].qs`, consumed 8 bytes at a time per sub-block).
const GRID_INDEX_LEN: usize = QK_K / 4;
/// Scale-and-sign region immediately after the grid-index region
/// (`ggml-quants.c:2288`, `scales_and_signs = qs + QK_K/4`).
const SCALES_AND_SIGNS_OFFSET: usize = QS_OFFSET + GRID_INDEX_LEN;
/// `QK_K/32` sub-blocks, one 32-bit aux word (scale nibble + 4 sign fields)
/// each.
const SUBBLOCK_COUNT: usize = QK_K / 32;

fn f16_at(block: &[u8], offset: usize) -> half::f16 {
    let mut bytes = [0u8; 2];
    bytes.copy_from_slice(&block[offset..offset + 2]);
    half::f16::from_le_bytes(bytes)
}

fn u32_at(block: &[u8], offset: usize) -> u32 {
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&block[offset..offset + 4]);
    u32::from_le_bytes(bytes)
}

/// Number of whole `IQ3_XXS` super-blocks a byte run decodes to, or `None`
/// if `byte_len` is not an exact multiple of [`BLOCK_BYTES`].
#[must_use]
pub const fn blocks_for_bytes(byte_len: usize) -> Option<usize> {
    if byte_len.is_multiple_of(BLOCK_BYTES) {
        Some(byte_len / BLOCK_BYTES)
    } else {
        None
    }
}

/// Exact packed byte length for `block_count` `IQ3_XXS` super-blocks.
#[must_use]
pub const fn bytes_for_blocks(block_count: usize) -> usize {
    block_count * BLOCK_BYTES
}

/// Exact `f32` element count for `block_count` `IQ3_XXS` super-blocks.
#[must_use]
pub const fn elements_for_blocks(block_count: usize) -> usize {
    block_count * QK_K
}

/// Dequantizes one 256-element `IQ3_XXS` super-block. `block` must be
/// exactly [`BLOCK_BYTES`] bytes and `output` exactly [`QK_K`] elements --
/// callers go through [`dequantize`], which validates both.
///
/// Ports `dequantize_row_iq3_xxs` (`ggml-quants.c:2278-2306`) exactly -- see
/// this module's own doc for the per-field decode this walks.
pub fn dequantize_block(block: &[u8], output: &mut [f32]) {
    let delta = f16_at(block, D_OFFSET).to_f32();

    for sub_block in 0..SUBBLOCK_COUNT {
        let aux32 = u32_at(block, SCALES_AND_SIGNS_OFFSET + sub_block * 4);
        let db = delta * (0.5 + f32::from((aux32 >> 28) as u8)) * 0.5;
        let grid_base = QS_OFFSET + sub_block * 8;

        for group in 0..4 {
            let sign_index = ((aux32 >> (7 * group)) & 127) as usize;
            let signs = KSIGNS_IQ2XS[sign_index];
            let index1 = usize::from(block[grid_base + group * 2]);
            let index2 = usize::from(block[grid_base + group * 2 + 1]);
            let grid1 = IQ3XXS_GRID[index1].to_le_bytes();
            let grid2 = IQ3XXS_GRID[index2].to_le_bytes();

            let out_base = sub_block * 32 + group * 8;
            for lane in 0..4 {
                let sign1 = if signs & KMASK_IQ2XS[lane] != 0 {
                    -1.0
                } else {
                    1.0
                };
                let sign2 = if signs & KMASK_IQ2XS[lane + 4] != 0 {
                    -1.0
                } else {
                    1.0
                };
                output[out_base + lane] = db * f32::from(grid1[lane]) * sign1;
                output[out_base + 4 + lane] = db * f32::from(grid2[lane]) * sign2;
            }
        }
    }
}

/// Dequantizes a run of `IQ3_XXS` super-blocks. `data` is borrowed,
/// `output` is caller-provided -- no allocation on this path.
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
        .zip(output.as_chunks_mut::<QK_K>().0)
    {
        dequantize_block(block, out_chunk);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use alloc::vec;

    use super::{
        BLOCK_BYTES, CODEC, D_OFFSET, IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS, QK_K, QS_OFFSET,
        QuantError, SCALES_AND_SIGNS_OFFSET, dequantize,
    };

    /// One super-block's first sub-block hand-packed and hand-decoded
    /// against `db * grid_byte * sign` computed by hand -- not by calling a
    /// quantize routine (this codec is decode-only). Exercises two distinct
    /// grid entries by index (`IQ3XXS_GRID[0]` = `0x04040404`, every lane
    /// magnitude `4`; `IQ3XXS_GRID[1]` = `0x04040414`, one lane magnitude
    /// `20`) plus a non-zero sign field (bit `0` set, flipping the group's
    /// first lane) reusing `IQ2_XS`'s own `KSIGNS_IQ2XS`/`KMASK_IQ2XS`
    /// tables, exactly as `dequantize_row_iq3_xxs` does. `db` is derived
    /// once per whole sub-block (shared by all 4 of its groups, not
    /// re-derived per group) -- sub-block 0's other 3 groups still see its
    /// scale nibble `3`, just with an all-zero sign field and grid index.
    /// The remaining 7 sub-blocks are left at an all-zero aux word /
    /// grid-index pattern, which still decodes to a real value via
    /// `IQ3XXS_GRID[0]`'s uniform
    /// magnitude-4 lanes and `db = d * 0.25`.
    #[test]
    fn dequantize_block_matches_hand_computed_fixture() {
        let delta = 0.02f32;
        let exact_delta = half::f16::from_f32(delta).to_f32();

        let mut block = [0u8; BLOCK_BYTES];
        block[D_OFFSET..D_OFFSET + 2].copy_from_slice(&half::f16::from_f32(delta).to_le_bytes());

        // sub-block 0, group 0: grid indices 0 and 1, sign field bit 0 set.
        block[QS_OFFSET] = 0; // index1
        block[QS_OFFSET + 1] = 1; // index2
        // aux32: top 4 bits scale nibble 3, low 7 bits (group 0's sign
        // field) = 1 (bit 0 set -> lane 0 of grid1 flips).
        let aux32: u32 = (3u32 << 28) | 1u32;
        block[SCALES_AND_SIGNS_OFFSET..SCALES_AND_SIGNS_OFFSET + 4]
            .copy_from_slice(&aux32.to_le_bytes());

        let db0 = exact_delta * (0.5 + 3.0) * 0.5;
        let grid0 = IQ3XXS_GRID[0].to_le_bytes();
        let grid1 = IQ3XXS_GRID[1].to_le_bytes();
        let signs0 = KSIGNS_IQ2XS[1];
        let mut expected = [0.0f32; QK_K];
        for lane in 0..4 {
            let sign1 = if signs0 & KMASK_IQ2XS[lane] != 0 {
                -1.0
            } else {
                1.0
            };
            let sign2 = if signs0 & KMASK_IQ2XS[lane + 4] != 0 {
                -1.0
            } else {
                1.0
            };
            expected[lane] = db0 * f32::from(grid0[lane]) * sign1;
            expected[4 + lane] = db0 * f32::from(grid1[lane]) * sign2;
        }
        // `db` is derived from the WHOLE sub-block's aux word (top 4 bits),
        // shared by all 4 of that sub-block's groups -- not re-derived per
        // group. Sub-block 0's remaining 3 groups (1..4) still read
        // `aux32`'s scale nibble 3 (`db0`), just with sign field 0 (the
        // corresponding 7-bit slice of `aux32` above is unset) and grid
        // index 0 (unset `qs` bytes). Sub-blocks 1..8 have an all-zero aux
        // word entirely -- scale nibble 0 (`db = d * 0.25`).
        let zero_db = exact_delta * 0.25;
        let no_flip = KSIGNS_IQ2XS[0];
        for sub_block in 0..8 {
            let db = if sub_block == 0 { db0 } else { zero_db };
            for group in 0..4 {
                if sub_block == 0 && group == 0 {
                    continue;
                }
                let out_base = sub_block * 32 + group * 8;
                for lane in 0..4 {
                    let sign1 = if no_flip & KMASK_IQ2XS[lane] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    let sign2 = if no_flip & KMASK_IQ2XS[lane + 4] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    expected[out_base + lane] = db * f32::from(grid0[lane]) * sign1;
                    expected[out_base + 4 + lane] = db * f32::from(grid0[lane]) * sign2;
                }
            }
        }

        let mut output = [0.0f32; QK_K];
        dequantize(&block, &mut output).expect("well-formed single super-block");
        assert_eq!(output.as_slice(), expected.as_slice());
    }

    /// Boundary sub-block scale nibbles: `0` (minimum, `db = d*0.25`) and
    /// `15` (maximum, `db = d*7.75`) -- the two extremes of the 4-bit
    /// nibble range packed into the aux word's top 4 bits.
    #[test]
    fn dequantize_block_boundary_min_and_max_scale_nibble() {
        let delta = 0.5f32;
        let mut block = [0u8; BLOCK_BYTES];
        block[D_OFFSET..D_OFFSET + 2].copy_from_slice(&half::f16::from_f32(delta).to_le_bytes());
        // sub-block 0: scale nibble 0 (min). Every grid index left at 0.
        // sub-block 1: scale nibble 15 (max, aux32 top 4 bits all set).
        let max_aux32: u32 = 0xF000_0000;
        block[SCALES_AND_SIGNS_OFFSET + 4..SCALES_AND_SIGNS_OFFSET + 8]
            .copy_from_slice(&max_aux32.to_le_bytes());

        let mut output = [0.0f32; QK_K];
        dequantize(&block, &mut output).expect("well-formed single super-block");
        let min_db = delta * 0.25;
        let max_db = delta * 7.75;
        assert_eq!(output[0], min_db * 4.0);
        assert_eq!(output[32], max_db * 4.0);
    }

    #[test]
    fn dequantize_rejects_non_block_multiple_length() {
        let data = vec![0u8; BLOCK_BYTES - 1];
        let mut output = vec![0.0f32; QK_K];
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
        let mut output = vec![0.0f32; QK_K - 1];
        let error = dequantize(&data, &mut output).unwrap_err();
        assert_eq!(
            error,
            QuantError::OutputSizeMismatch {
                found: QK_K - 1,
                expected: QK_K,
            }
        );
    }

    /// A truncated block run that is neither a whole block nor empty --
    /// typed error, never a panic or an out-of-bounds read.
    #[test]
    fn dequantize_rejects_truncated_partial_block() {
        let partial_bytes = BLOCK_BYTES + BLOCK_BYTES / 2;
        let data = vec![0u8; partial_bytes];
        let mut output = vec![0.0f32; QK_K];
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
