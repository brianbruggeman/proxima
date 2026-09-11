//! `IQ2_XS`: 2.3125-bit non-linear (grid-indexed) weights in 256-element
//! super-blocks, one `f16` scale per super-block, 8 sub-block scale nibbles,
//! and a 512-entry codebook + 128-entry sign table per output element.
//! `value = scale_for_subblock * grid_byte * sign`
//! (`dequantize_row_iq2_xs`, `ggml-quants.c:2219-2242`, cited on
//! [`dequantize_block`]).
//!
//! Layout, from llama.cpp (`/Users/brianbruggeman/repos/others/llama.cpp`):
//! `ggml/src/ggml-common.h:344-351`:
//! ```c
//! typedef struct {
//!     ggml_half d;
//!     uint16_t qs[QK_K/8];
//!     uint8_t  scales[QK_K/32];
//! } block_iq2_xs;
//! static_assert(sizeof(block_iq2_xs) == sizeof(ggml_half) + QK_K/8*sizeof(uint16_t) + QK_K/32, "wrong iq2_xs block size/padding");
//! ```
//! `QK_K` is 256 (`ggml-common.h:89`), so this is 74 bytes per 256 elements
//! (`ggml_half d`, 32 packed `u16` grid+sign entries, 8 sub-block scale
//! nibble-pairs). Cross-checked here at compile time against
//! [`crate::types::GgmlType::Iq2Xs`]'s already-landed
//! [`crate::types::GgmlType::block_layout`] rather than re-typed by hand.
//!
//! Per-element decode (`dequantize_row_iq2_xs`, `ggml-quants.c:2229-2240`):
//! the 256 elements split into 8 sub-blocks of 32 (`ib32`); each sub-block's
//! `scales` byte holds two nibbles -- low nibble scales its first 16
//! elements, high nibble its last 16 -- via `db = d * (0.5 + nibble) *
//! 0.25`. Each sub-block's 4 `qs` entries (one per 8-element group) pack a
//! 9-bit index into [`crate::quant::tables::iq2xs_grid::IQ2XS_GRID`] (low 9
//! bits) and a 7-bit index into
//! [`crate::quant::tables::iq2xs_grid::KSIGNS_IQ2XS`] (remaining bits): the
//! grid entry's 8 packed bytes (read little-endian) are the group's 8
//! magnitude values, each negated per its own bit in the sign lookup byte
//! (tested against [`crate::quant::tables::iq2xs_grid::KMASK_IQ2XS`]).

use crate::quant::QuantError;
use crate::quant::tables::iq2xs_grid::{IQ2XS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use crate::types::GgmlType;

/// This codec's name as it appears in a rendered [`QuantError`] message.
const CODEC: &str = "iq2_xs";

/// Elements per super-block (`ggml-common.h:89`, `#define QK_K 256`).
pub const QK_K: usize = 256;

/// Bytes per super-block. Derived from the already-landed
/// [`crate::types::GgmlType::block_layout`] (`types.rs`), not re-typed --
/// that table was itself checked against `ggml-common.h`'s
/// `static_assert(sizeof(block_iq2_xs) == sizeof(ggml_half) +
/// QK_K/8*sizeof(uint16_t) + QK_K/32, ...)` at `ggml-common.h:351`.
pub const BLOCK_BYTES: usize = {
    let layout = GgmlType::Iq2Xs.block_layout();
    assert!(
        layout.block_elements as usize == QK_K,
        "GgmlType::Iq2Xs block_elements drifted from QK_K"
    );
    layout.block_bytes as usize
};

const D_OFFSET: usize = 0;
const QS_OFFSET: usize = 2;
/// `QK_K/8` `u16` grid+sign entries (`ggml-common.h:346`).
const QS_LEN: usize = QK_K / 8;
const SCALES_OFFSET: usize = QS_OFFSET + QS_LEN * 2;
/// `QK_K/32` sub-blocks, one scale byte (two nibbles) each.
const SUBBLOCK_COUNT: usize = QK_K / 32;

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

/// Number of whole `IQ2_XS` super-blocks a byte run decodes to, or `None`
/// if `byte_len` is not an exact multiple of [`BLOCK_BYTES`].
#[must_use]
pub const fn blocks_for_bytes(byte_len: usize) -> Option<usize> {
    if byte_len.is_multiple_of(BLOCK_BYTES) {
        Some(byte_len / BLOCK_BYTES)
    } else {
        None
    }
}

/// Exact packed byte length for `block_count` `IQ2_XS` super-blocks.
#[must_use]
pub const fn bytes_for_blocks(block_count: usize) -> usize {
    block_count * BLOCK_BYTES
}

/// Exact `f32` element count for `block_count` `IQ2_XS` super-blocks.
#[must_use]
pub const fn elements_for_blocks(block_count: usize) -> usize {
    block_count * QK_K
}

/// Dequantizes one 256-element `IQ2_XS` super-block. `block` must be
/// exactly [`BLOCK_BYTES`] bytes and `output` exactly [`QK_K`] elements --
/// callers go through [`dequantize`], which validates both.
///
/// Ports `dequantize_row_iq2_xs` (`ggml-quants.c:2219-2242`) exactly -- see
/// this module's own doc for the per-field decode this walks.
pub fn dequantize_block(block: &[u8], output: &mut [f32]) {
    let delta = f16_at(block, D_OFFSET).to_f32();
    let scales = &block[SCALES_OFFSET..SCALES_OFFSET + SUBBLOCK_COUNT];

    for (sub_block, &scale_byte) in scales.iter().enumerate() {
        let db_low = delta * (0.5 + f32::from(scale_byte & 0x0F)) * 0.25;
        let db_high = delta * (0.5 + f32::from(scale_byte >> 4)) * 0.25;

        for group in 0..4 {
            let entry_index = sub_block * 4 + group;
            let qs_value = u16_at(block, QS_OFFSET + entry_index * 2);
            let grid_index = usize::from(qs_value & 511);
            let sign_index = usize::from(qs_value >> 9);
            let grid_bytes = IQ2XS_GRID[grid_index].to_le_bytes();
            let signs = KSIGNS_IQ2XS[sign_index];
            let db = if group < 2 { db_low } else { db_high };

            let out_base = sub_block * 32 + group * 8;
            for lane in 0..8 {
                let magnitude = f32::from(grid_bytes[lane]);
                let sign = if signs & KMASK_IQ2XS[lane] != 0 {
                    -1.0
                } else {
                    1.0
                };
                output[out_base + lane] = db * magnitude * sign;
            }
        }
    }
}

/// Dequantizes a run of `IQ2_XS` super-blocks. `data` is borrowed, `output`
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
        BLOCK_BYTES, CODEC, D_OFFSET, IQ2XS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS, QK_K, QS_OFFSET,
        QuantError, SCALES_OFFSET, dequantize,
    };

    /// One super-block's first sub-block hand-packed and hand-decoded
    /// against `db * grid_byte * sign` computed by hand -- not by calling a
    /// quantize routine (this codec is decode-only). Exercises three
    /// distinct grid entries by index (`IQ2XS_GRID[0]` = `0x0808...08`, all
    /// magnitude-8 lanes; `IQ2XS_GRID[1]` =
    /// `0x080808080808082b`, one magnitude-43 lane; `IQ2XS_GRID[3]` =
    /// `0x0808080808082b08`, a different magnitude-43 lane position) plus a
    /// non-zero sign index (`KSIGNS_IQ2XS[1] = 129 = 0b1000_0001`, flipping
    /// lanes 0 and 7) so the sign path is covered inside the same fixture.
    /// The remaining 7 sub-blocks are left at an all-zero scale/qs pattern,
    /// which still decodes to a real (non-degenerate) value via
    /// `IQ2XS_GRID[0]`'s all-`8` magnitude and `db = d * 0.125`.
    #[test]
    fn dequantize_block_matches_hand_computed_fixture() {
        let delta = 0.02f32;
        let exact_delta = half::f16::from_f32(delta).to_f32();

        let mut block = [0u8; BLOCK_BYTES];
        block[0..2].copy_from_slice(&half::f16::from_f32(delta).to_le_bytes());

        // sub-block 0: scale nibbles 1 (low) / 2 (high).
        block[SCALES_OFFSET] = 0x21;
        // group 0 (low half): grid index 0, sign index 0 (no flip).
        block[QS_OFFSET..QS_OFFSET + 2].copy_from_slice(&0u16.to_le_bytes());
        // group 1 (low half): grid index 1, sign index 0.
        block[QS_OFFSET + 2..QS_OFFSET + 4].copy_from_slice(&1u16.to_le_bytes());
        // group 2 (high half): grid index 2, sign index 0.
        block[QS_OFFSET + 4..QS_OFFSET + 6].copy_from_slice(&2u16.to_le_bytes());
        // group 3 (high half): grid index 3, sign index 1 (flips lanes 0/7).
        let qs3 = (1u16 << 9) | 3u16;
        block[QS_OFFSET + 6..QS_OFFSET + 8].copy_from_slice(&qs3.to_le_bytes());

        let mut expected = [0.0f32; QK_K];
        let db_low = exact_delta * (0.5 + 1.0) * 0.25;
        let db_high = exact_delta * (0.5 + 2.0) * 0.25;
        let fill_group = |expected: &mut [f32; QK_K],
                          base: usize,
                          grid_index: usize,
                          db: f32,
                          sign_index: usize| {
            let bytes = IQ2XS_GRID[grid_index].to_le_bytes();
            let signs = KSIGNS_IQ2XS[sign_index];
            for lane in 0..8 {
                let sign = if signs & KMASK_IQ2XS[lane] != 0 {
                    -1.0
                } else {
                    1.0
                };
                expected[base + lane] = db * f32::from(bytes[lane]) * sign;
            }
        };
        fill_group(&mut expected, 0, 0, db_low, 0);
        fill_group(&mut expected, 8, 1, db_low, 0);
        fill_group(&mut expected, 16, 2, db_high, 0);
        fill_group(&mut expected, 24, 3, db_high, 1);
        // sub-blocks 1..8: all-zero scale/qs -- grid index 0 (magnitude 8
        // every lane), sign index 0 (no flip), db = d * 0.125.
        let zero_db = exact_delta * 0.125;
        for sub_block in 1..8 {
            for group in 0..4 {
                fill_group(&mut expected, sub_block * 32 + group * 8, 0, zero_db, 0);
            }
        }

        let mut output = [0.0f32; QK_K];
        dequantize(&block, &mut output).expect("well-formed single super-block");
        assert_eq!(output.as_slice(), expected.as_slice());
    }

    /// Boundary sub-block scale nibbles: `0` (minimum, `db = d*0.125`) and
    /// `15` (maximum, `db = d*3.875`) -- the two extremes of the 4-bit
    /// nibble range this format's scale packs into.
    #[test]
    fn dequantize_block_boundary_min_and_max_scale_nibble() {
        let delta = 0.5f32;
        let mut block = [0u8; BLOCK_BYTES];
        block[D_OFFSET..D_OFFSET + 2].copy_from_slice(&half::f16::from_f32(delta).to_le_bytes());
        block[SCALES_OFFSET] = 0xF0; // low nibble 0 (min), high nibble 15 (max).
        // every group: grid index 0 (all magnitude-8 lanes), sign index 0.

        let mut output = [0.0f32; QK_K];
        dequantize(&block, &mut output).expect("well-formed single super-block");
        let min_db = delta * 0.125;
        let max_db = delta * 3.875;
        assert_eq!(output[0], min_db * 8.0);
        assert_eq!(output[16], max_db * 8.0);
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
