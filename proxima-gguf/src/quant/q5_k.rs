//! `Q5_K`: 5-bit weights in 256-element super-blocks, each split into 8
//! sub-blocks of 32 with its own 6-bit (scale, min) pair -- the same
//! super-block/sub-block shape and the same bit-interleaved scale packing
//! as [`super::q4_k`], plus a `qh` high-bit plane supplying each weight's
//! 5th bit. `x = d*sc*q - dmin*m` per sub-block, `q` a 5-bit value
//! assembled from a `qs` nibble and one `qh` bit (`ggml-quants.c:1476-1501`,
//! cited on [`dequantize_block`]).
//!
//! Layout, from llama.cpp (`/Users/brianbruggeman/repos/others/llama.cpp`):
//! `ggml/src/ggml-common.h:302-313` -- `block_q5_K` is 176 bytes per 256
//! elements (`ggml_half d`, `ggml_half dmin`, `uint8_t scales[12]`
//! (`K_SCALE_SIZE`), `uint8_t qh[32]` (`QK_K/8`), `uint8_t qs[128]`
//! (`QK_K/2`)) -- `qh` sits between `scales` and `qs`, unlike a naive
//! guess that would tack it on at the end. `QK_K` is 256 and
//! `K_SCALE_SIZE` is 12 (`ggml-common.h:89-90`). That 176-byte figure is
//! cross-checked here at compile time against
//! [`crate::types::GgmlType::Q5_K`]'s already-landed
//! [`crate::types::GgmlType::block_layout`] rather than re-typed by hand.
//!
//! Shared with [`super::q4_k`]: the super-block/sub-block shape (8
//! sub-blocks of 32), the 6-bit bit-interleaved `(scale, min)` packing
//! (`get_scale_min_k4`/`pack_scales` below are byte-for-byte the same
//! algorithm as `q4_k`'s, duplicated rather than imported -- each codec
//! module owns its primitives independently, matching this crate's
//! one-format-per-file layout, the same choice [`super::q6_k`] already
//! made for `nearest_int`). Not shared: the per-element quant width (5
//! bits here via `qs` nibble + `qh` bit, vs `q4_k`'s plain 4-bit nibble)
//! and therefore every level clamp (`0..=31`, not `0..=15`).

use thiserror::Error;

use crate::types::GgmlType;

/// Elements per super-block (`ggml-common.h:89`, `#define QK_K 256`).
pub const QK_K: usize = 256;

/// Elements per sub-block: `QK_K` is split into 8 sub-blocks of 32, same
/// as [`super::q4_k`] (`ggml-quants.c:1402`, `for (int j = 0; j < QK_K/32;
/// ++j)`).
pub const SUB_BLOCK_ELEMENTS: usize = 32;

/// Sub-blocks per super-block.
pub const SUB_BLOCKS: usize = QK_K / SUB_BLOCK_ELEMENTS;

/// Bytes of bit-packed (scale, min) pairs per super-block
/// (`ggml-common.h:90`, `#define K_SCALE_SIZE 12`).
pub const K_SCALE_SIZE: usize = 12;

/// Bytes per super-block. Derived from the already-landed
/// [`crate::types::GgmlType::block_layout`] (`types.rs`), not re-typed --
/// that table was itself checked against `ggml-common.h`'s
/// `static_assert(sizeof(block_q5_K) == 2*sizeof(ggml_half) +
/// K_SCALE_SIZE + QK_K/2 + QK_K/8, ...)` at `ggml-common.h:314`.
pub const BLOCK_BYTES: usize = {
    let layout = GgmlType::Q5_K.block_layout();
    assert!(layout.block_elements as usize == QK_K, "GgmlType::Q5_K block_elements drifted from QK_K");
    layout.block_bytes as usize
};

const D_OFFSET: usize = 0;
const DMIN_OFFSET: usize = 2;
const SCALES_OFFSET: usize = 4;
const QH_OFFSET: usize = SCALES_OFFSET + K_SCALE_SIZE;
const QH_BYTES: usize = QK_K / 8;
const QS_OFFSET: usize = QH_OFFSET + QH_BYTES;

/// Everything that can go wrong sizing a `Q5_K` codec call. Never a
/// panic: a malformed or mis-sized buffer is always an `Err`.
#[derive(Debug, Error, PartialEq, Eq, Clone, Copy)]
pub enum QuantError {
    #[error("input length {found} bytes is not a multiple of the q5_k block size {block_bytes}")]
    InputNotBlockMultiple { found: usize, block_bytes: usize },
    #[error("input length {found} elements is not a multiple of the q5_k super-block size {block_elements}")]
    InputNotElementMultiple {
        found: usize,
        block_elements: usize,
    },
    #[error("output slice has {found} elements, expected {expected}")]
    OutputSizeMismatch { found: usize, expected: usize },
}

/// Number of whole `Q5_K` super-blocks a byte run decodes to, or `None`
/// if `byte_len` is not an exact multiple of [`BLOCK_BYTES`].
#[must_use]
pub const fn blocks_for_bytes(byte_len: usize) -> Option<usize> {
    if byte_len.is_multiple_of(BLOCK_BYTES) {
        Some(byte_len / BLOCK_BYTES)
    } else {
        None
    }
}

/// Exact packed byte length for `block_count` `Q5_K` super-blocks.
#[must_use]
pub const fn bytes_for_blocks(block_count: usize) -> usize {
    block_count * BLOCK_BYTES
}

/// Exact `f32` element count for `block_count` `Q5_K` super-blocks.
#[must_use]
pub const fn elements_for_blocks(block_count: usize) -> usize {
    block_count * QK_K
}

/// Ties-to-even rounding, porting the IEEE-754 magic-number trick in
/// `ggml-quants.c:366-371` (`nearest_int`) bit-for-bit -- see
/// [`super::q4_k`]'s copy of this same function for the full derivation;
/// duplicated here rather than shared because each codec module owns its
/// primitives independently, matching this crate's one-format-per-file
/// layout.
fn nearest_int(value: f32) -> i32 {
    let shifted = value + 12_582_912.0;
    let bits = shifted.to_bits();
    (bits & 0x007f_ffff) as i32 - 0x0040_0000
}

/// Unpacks one sub-block's 6-bit scale and min out of the 12-byte
/// `scales` field. Ports `get_scale_min_k4` (`ggml-quants.c:625-632`)
/// exactly -- byte-for-byte the same function [`super::q4_k`] ports,
/// duplicated rather than imported (see module doc).
fn get_scale_min_k4(sub_block: usize, scales: &[u8; K_SCALE_SIZE]) -> (u8, u8) {
    if sub_block < 4 {
        (scales[sub_block] & 63, scales[sub_block + 4] & 63)
    } else {
        let scale = (scales[sub_block + 4] & 0x0F) | ((scales[sub_block - 4] >> 6) << 4);
        let min = (scales[sub_block + 4] >> 4) | ((scales[sub_block] >> 6) << 4);
        (scale, min)
    }
}

/// Dequantizes one 256-element `Q5_K` super-block. `block` must be
/// exactly [`BLOCK_BYTES`] bytes and `output` exactly [`QK_K`] elements --
/// callers go through [`dequantize`], which validates both.
///
/// Ports `dequantize_row_q5_K` (`ggml-quants.c:1476-1501`) exactly,
/// including its nibble order (the same trap [`super::q4_k`] documents: a
/// `qs` byte's low nibble and high nibble land 32 output elements apart)
/// plus the high-bit plane's own indexing trap: `qh` is 32 bytes for the
/// *whole* super-block and its byte index never advances past the current
/// 64-element chunk's local position `l` (`0..32`, `ggml-quants.c:1495-1496`,
/// `qh[l] & u1`) -- unlike `qs`, which does advance by 32 bytes every
/// 64-element chunk (`ql += 32` at `ggml-quants.c:1497`). What changes
/// across the four 64-element chunks is only which two bits of that same
/// `qh[l]` byte apply: `u1`/`u2` start at `1`/`2` and are shifted left by
/// 2 after every chunk (`ggml-quants.c:1498`), so `qh[l]`'s 8 bits supply
/// the high bit for 8 different output elements across the super-block
/// (`l`, `l+32`, `l+64`, ..., `l+224`), two bits (one low-half, one
/// high-half) per chunk.
fn dequantize_block(block: &[u8], output: &mut [f32]) {
    let d = f16_at(block, D_OFFSET).to_f32();
    let dmin = f16_at(block, DMIN_OFFSET).to_f32();
    let mut scales = [0u8; K_SCALE_SIZE];
    scales.copy_from_slice(&block[SCALES_OFFSET..SCALES_OFFSET + K_SCALE_SIZE]);
    let qh = &block[QH_OFFSET..QH_OFFSET + QH_BYTES];
    let qs = &block[QS_OFFSET..QS_OFFSET + QK_K / 2];

    let mut sub_block = 0usize;
    let mut qs_offset = 0usize;
    let mut out_offset = 0usize;
    let mut mask_lo = 1u8;
    let mut mask_hi = 2u8;
    for _ in 0..QK_K / 64 {
        let (scale_lo, min_lo) = get_scale_min_k4(sub_block, &scales);
        let (scale_hi, min_hi) = get_scale_min_k4(sub_block + 1, &scales);
        let scale_lo = d * f32::from(scale_lo);
        let min_lo = dmin * f32::from(min_lo);
        let scale_hi = d * f32::from(scale_hi);
        let min_hi = dmin * f32::from(min_hi);

        for offset in 0..SUB_BLOCK_ELEMENTS {
            let high_lo = if qh[offset] & mask_lo != 0 { 16.0 } else { 0.0 };
            let high_hi = if qh[offset] & mask_hi != 0 { 16.0 } else { 0.0 };
            output[out_offset + offset] = scale_lo * (f32::from(qs[qs_offset + offset] & 0x0F) + high_lo) - min_lo;
            output[out_offset + SUB_BLOCK_ELEMENTS + offset] =
                scale_hi * (f32::from(qs[qs_offset + offset] >> 4) + high_hi) - min_hi;
        }

        qs_offset += SUB_BLOCK_ELEMENTS;
        out_offset += 64;
        sub_block += 2;
        mask_lo <<= 2;
        mask_hi <<= 2;
    }
}

fn f16_at(block: &[u8], offset: usize) -> half::f16 {
    let mut bytes = [0u8; 2];
    bytes.copy_from_slice(&block[offset..offset + 2]);
    half::f16::from_le_bytes(bytes)
}

/// Dequantizes a run of `Q5_K` super-blocks. `data` is borrowed, `output`
/// is caller-provided -- no allocation on this path.
///
/// # Errors
/// [`QuantError::InputNotBlockMultiple`] if `data.len()` is not a
/// multiple of [`BLOCK_BYTES`]; [`QuantError::OutputSizeMismatch`] if
/// `output.len()` does not exactly match the decoded element count.
pub fn dequantize(data: &[u8], output: &mut [f32]) -> Result<(), QuantError> {
    let block_count = blocks_for_bytes(data.len()).ok_or(QuantError::InputNotBlockMultiple {
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
    for (block, out_chunk) in data.chunks_exact(BLOCK_BYTES).zip(output.chunks_exact_mut(QK_K)) {
        dequantize_block(block, out_chunk);
    }
    Ok(())
}

/// One sub-block's weighted-least-squares (scale, min) search: candidate
/// levels `L[i] in [0, nmax]`, the affine map `x[i] ~= scale*L[i] + min`.
///
/// Ports `make_qkx2_quants` (`ggml-quants.c:544-623`) with `n` fixed at
/// [`SUB_BLOCK_ELEMENTS`] (32) and `use_mad` fixed at `false`, same as
/// [`super::q4_k`]'s copy -- the only difference is `nmax` fixed at `31`
/// here (`ggml-quants.c:1408`, `make_qkx2_quants(32, 31, ...)`) instead of
/// `q4_k`'s `15`, since `Q5_K` levels are 5-bit.
fn make_qkx2_quants_32(x: &[f32; SUB_BLOCK_ELEMENTS], weights: &[f32; SUB_BLOCK_ELEMENTS]) -> ([u8; SUB_BLOCK_ELEMENTS], f32, f32) {
    const NMAX: i32 = 31;
    const NMAX_F: f32 = NMAX as f32;

    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = weights[0] * x[0];
    for index in 1..SUB_BLOCK_ELEMENTS {
        min = min.min(x[index]);
        max = max.max(x[index]);
        sum_w += weights[index];
        sum_x += weights[index] * x[index];
    }
    min = min.min(0.0);
    if max == min {
        return ([0u8; SUB_BLOCK_ELEMENTS], -min, 0.0);
    }

    let mut levels = [0u8; SUB_BLOCK_ELEMENTS];
    let iscale = NMAX_F / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0.0f32;
    for index in 0..SUB_BLOCK_ELEMENTS {
        let level = nearest_int(iscale * (x[index] - min)).clamp(0, NMAX) as u8;
        levels[index] = level;
        let diff = scale * f32::from(level) + min - x[index];
        best_error += weights[index] * diff * diff;
    }

    let mut candidate_levels = [0u8; SUB_BLOCK_ELEMENTS];
    for step in 0..=20i32 {
        let candidate_iscale = (-1.0 + 0.1 * step as f32 + NMAX_F) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0.0f32, 0.0f32, 0.0f32);
        for index in 0..SUB_BLOCK_ELEMENTS {
            let level = nearest_int(candidate_iscale * (x[index] - min)).clamp(0, NMAX) as u8;
            candidate_levels[index] = level;
            let (weight, level_f) = (weights[index], f32::from(level));
            sum_l += weight * level_f;
            sum_l2 += weight * level_f * level_f;
            sum_xl += weight * level_f * x[index];
        }
        let denominator = sum_w * sum_l2 - sum_l * sum_l;
        if denominator <= 0.0 {
            continue;
        }
        let mut candidate_scale = (sum_w * sum_xl - sum_x * sum_l) / denominator;
        let mut candidate_min = (sum_l2 * sum_x - sum_l * sum_xl) / denominator;
        if candidate_min > 0.0 {
            candidate_min = 0.0;
            candidate_scale = sum_xl / sum_l2;
        }
        let mut error = 0.0f32;
        for index in 0..SUB_BLOCK_ELEMENTS {
            let diff = candidate_scale * f32::from(candidate_levels[index]) + candidate_min - x[index];
            error += weights[index] * diff * diff;
        }
        if error < best_error {
            levels = candidate_levels;
            best_error = error;
            scale = candidate_scale;
            min = candidate_min;
        }
    }
    (levels, -min, scale)
}

/// Packs 8 sub-block `(scale, min)` pairs into the 12-byte bit-interleaved
/// [`K_SCALE_SIZE`] field. Ports the packing half of
/// `quantize_row_q5_K_ref` (`ggml-quants.c:1421-1434`) -- byte-for-byte
/// the same layout [`super::q4_k::pack_scales`] ports, duplicated rather
/// than imported (see module doc); the exact inverse [`get_scale_min_k4`]
/// unpacks.
fn pack_scales(scale_codes: &[u8; SUB_BLOCKS], min_codes: &[u8; SUB_BLOCKS]) -> [u8; K_SCALE_SIZE] {
    let mut packed = [0u8; K_SCALE_SIZE];
    for sub_block in 0..SUB_BLOCKS {
        let scale_code = scale_codes[sub_block];
        let min_code = min_codes[sub_block];
        if sub_block < 4 {
            packed[sub_block] = scale_code;
            packed[sub_block + 4] = min_code;
        } else {
            packed[sub_block + 4] = (scale_code & 0x0F) | ((min_code & 0x0F) << 4);
            packed[sub_block - 4] |= (scale_code >> 4) << 6;
            packed[sub_block] |= (min_code >> 4) << 6;
        }
    }
    packed
}

/// Quantizes one 256-element chunk into a `Q5_K` super-block.
///
/// Ports `quantize_row_q5_K_ref` (`ggml-quants.c:1389-1474`) -- the
/// reference strategy, not the newer `quantize_row_q5_K_impl`'s
/// `quant_weights`-aware search (`ggml-quants.c:1503-...`), same
/// trade-off [`super::q4_k::quantize_block`] documents for its own
/// `_impl` gap. Per sub-block: importance weights `av_x + |x|`, a
/// [`make_qkx2_quants_32`] search (`nmax=31`) for that sub-block's own
/// (scale, min), then a linear pass fitting all 8 sub-block scales/mins
/// into shared 6-bit codes, then levels recomputed once against the
/// fp16-rounded packed (scale, min) and clamped to `[0, 31]`
/// (`ggml-quants.c:1446`) -- five bits, not four. The final pack step is
/// where `Q5_K` diverges from `Q4_K`: each level over 15 has 16
/// subtracted and its high bit set in `qh` (`ggml-quants.c:1456-1470`)
/// before the low 4 bits go into `qs` alongside `Q4_K`'s usual
/// two-sub-blocks-per-byte nibble packing.
fn quantize_block(x: &[f32], output: &mut [u8]) {
    let mut levels = [0u8; QK_K];
    let mut mins = [0.0f32; SUB_BLOCKS];
    let mut scales = [0.0f32; SUB_BLOCKS];

    for sub_block in 0..SUB_BLOCKS {
        let mut chunk = [0.0f32; SUB_BLOCK_ELEMENTS];
        chunk.copy_from_slice(&x[sub_block * SUB_BLOCK_ELEMENTS..(sub_block + 1) * SUB_BLOCK_ELEMENTS]);
        let sum_sq: f32 = chunk.iter().map(|value| value * value).sum();
        let av_x = libm::sqrtf(sum_sq / SUB_BLOCK_ELEMENTS as f32);
        let mut weights = [0.0f32; SUB_BLOCK_ELEMENTS];
        for (weight, value) in weights.iter_mut().zip(chunk.iter()) {
            *weight = av_x + value.abs();
        }
        let (sub_levels, sub_min, sub_scale) = make_qkx2_quants_32(&chunk, &weights);
        levels[sub_block * SUB_BLOCK_ELEMENTS..(sub_block + 1) * SUB_BLOCK_ELEMENTS].copy_from_slice(&sub_levels);
        mins[sub_block] = sub_min;
        scales[sub_block] = sub_scale;
    }

    let max_scale = scales.iter().copied().fold(0.0f32, f32::max);
    let max_min = mins.iter().copied().fold(0.0f32, f32::max);
    let scale_step = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
    let min_step = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

    let mut scale_codes = [0u8; SUB_BLOCKS];
    let mut min_codes = [0u8; SUB_BLOCKS];
    for sub_block in 0..SUB_BLOCKS {
        scale_codes[sub_block] = (nearest_int(scale_step * scales[sub_block]) as u8).min(63);
        min_codes[sub_block] = (nearest_int(min_step * mins[sub_block]) as u8).min(63);
    }
    let packed_scales = pack_scales(&scale_codes, &min_codes);

    let block_scale = half::f16::from_f32(max_scale / 63.0);
    let block_min = half::f16::from_f32(max_min / 63.0);

    for sub_block in 0..SUB_BLOCKS {
        let (scale_code, min_code) = get_scale_min_k4(sub_block, &packed_scales);
        let sub_scale = block_scale.to_f32() * f32::from(scale_code);
        if sub_scale == 0.0 {
            continue;
        }
        let sub_min = block_min.to_f32() * f32::from(min_code);
        for offset in 0..SUB_BLOCK_ELEMENTS {
            let index = sub_block * SUB_BLOCK_ELEMENTS + offset;
            levels[index] = nearest_int((x[index] + sub_min) / sub_scale).clamp(0, 31) as u8;
        }
    }

    output[D_OFFSET..D_OFFSET + 2].copy_from_slice(&block_scale.to_le_bytes());
    output[DMIN_OFFSET..DMIN_OFFSET + 2].copy_from_slice(&block_min.to_le_bytes());
    output[SCALES_OFFSET..SCALES_OFFSET + K_SCALE_SIZE].copy_from_slice(&packed_scales);

    let (qh, qs) = output[QH_OFFSET..].split_at_mut(QH_BYTES);
    qh.fill(0);

    let mut qs_offset = 0usize;
    let mut mask_lo = 1u8;
    let mut mask_hi = 2u8;
    for base in (0..QK_K).step_by(64) {
        for offset in 0..SUB_BLOCK_ELEMENTS {
            let mut level_lo = levels[base + offset];
            if level_lo > 15 {
                level_lo -= 16;
                qh[offset] |= mask_lo;
            }
            let mut level_hi = levels[base + SUB_BLOCK_ELEMENTS + offset];
            if level_hi > 15 {
                level_hi -= 16;
                qh[offset] |= mask_hi;
            }
            qs[qs_offset + offset] = level_lo | (level_hi << 4);
        }
        qs_offset += SUB_BLOCK_ELEMENTS;
        mask_lo <<= 2;
        mask_hi <<= 2;
    }
}

/// Quantizes a run of `f32` weights into `Q5_K` super-blocks. `input` is
/// borrowed, `output` is caller-provided -- no allocation on this path
/// beyond the fixed-size stack scratch each block's optimization search
/// needs.
///
/// # Errors
/// [`QuantError::InputNotElementMultiple`] if `input.len()` is not a
/// multiple of [`QK_K`]; [`QuantError::OutputSizeMismatch`] if
/// `output.len()` does not exactly match the packed byte count.
pub fn quantize(input: &[f32], output: &mut [u8]) -> Result<(), QuantError> {
    if !input.len().is_multiple_of(QK_K) {
        return Err(QuantError::InputNotElementMultiple {
            found: input.len(),
            block_elements: QK_K,
        });
    }
    let block_count = input.len() / QK_K;
    let expected = bytes_for_blocks(block_count);
    if output.len() != expected {
        return Err(QuantError::OutputSizeMismatch {
            found: output.len(),
            expected,
        });
    }
    for (chunk, out_block) in input.chunks_exact(QK_K).zip(output.chunks_exact_mut(BLOCK_BYTES)) {
        quantize_block(chunk, out_block);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use alloc::vec;
    use alloc::vec::Vec;

    use super::{
        BLOCK_BYTES, K_SCALE_SIZE, QH_BYTES, QK_K, QuantError, SUB_BLOCK_ELEMENTS, SUB_BLOCKS, dequantize, quantize,
    };

    /// One super-block, hand-packed and hand-decoded, checked against the
    /// `x = d*sc*q - dmin*m` formula computed by hand -- not by calling
    /// [`super::quantize`] to build the fixture. `d=1.0`, `dmin=0.5`
    /// (both exact in `f16`) so every expected value below is an exact
    /// integer or half-integer in `f32`; `assert_eq!` needs no epsilon.
    ///
    /// The 8 sub-block `(scale, min)` pairs are the same values
    /// [`super::q4_k`]'s hand-packed fixture uses (same packing, ported
    /// unchanged), covering both the direct (`sub_block < 4`) and
    /// interleaved (`sub_block >= 4`) branches of [`super::get_scale_min_k4`].
    /// The trap this fixture is built to catch: `qh` is indexed by local
    /// position within a 64-element chunk (`0..32`), not by the running
    /// `qs` cursor -- so two probes that land in *different* 64-element
    /// chunks but at the *same* local offset must read *different* bits
    /// (`mask_lo`/`mask_hi` having shifted) of the *same* `qh` byte.
    #[test]
    fn dequantize_block_matches_hand_packed_fixture() {
        // sc = [3, 45, 12, 63, 33, 7, 58, 21] (used only in the comments
        // below and the derivation this fixture was built from)
        const MIN: [u32; SUB_BLOCKS] = [61, 2, 44, 9, 50, 19, 6, 63];
        let packed_scales: [u8; K_SCALE_SIZE] = [131, 45, 204, 127, 253, 66, 44, 201, 33, 55, 106, 245];

        // qs is all zero except one probe byte per 32-byte region, each
        // touching one low-nibble and one high-nibble output element --
        // same probe layout as q4_k's fixture, at local offset l=0.
        let mut qs = [0u8; QK_K / 2];
        qs[0] = 0xD7; // sub_block 0 low nibble = 7 (elem 0), sub_block 1 high nibble = 13 (elem 32)
        qs[32] = 0x2C; // sub_block 2 low nibble = 12 (elem 64), sub_block 3 high nibble = 2 (elem 96)
        qs[64] = 0xF1; // sub_block 4 low nibble = 1 (elem 128), sub_block 5 high nibble = 15 (elem 160)
        qs[96] = 0x59; // sub_block 6 low nibble = 9 (elem 192), sub_block 7 high nibble = 5 (elem 224)

        // qh[0] carries the high bit for local offset l=0 across all four
        // 64-element chunks: bits 0/1 for chunk 0 (elems 0, 32), bits 2/3
        // for chunk 1 (elems 64, 96), bits 4/5 for chunk 2 (elems 128,
        // 160), bits 6/7 for chunk 3 (elems 192, 224). Set every other
        // bit (0b01010101 = 0x55) so half the probes get +16 and half
        // don't, exercising both branches of the high-bit trap in one
        // fixture: elem 0 (+16), elem 32 (no), elem 64 (+16), elem 96
        // (no), elem 128 (+16), elem 160 (no), elem 192 (+16), elem 224
        // (no).
        let mut qh = [0u8; QH_BYTES];
        qh[0] = 0x55;

        let mut block = [0u8; BLOCK_BYTES];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes()); // d
        block[2..4].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes()); // dmin
        block[4..4 + K_SCALE_SIZE].copy_from_slice(&packed_scales);
        block[4 + K_SCALE_SIZE..4 + K_SCALE_SIZE + QH_BYTES].copy_from_slice(&qh);
        block[4 + K_SCALE_SIZE + QH_BYTES..].copy_from_slice(&qs);

        // expected[i] = d*sc[s]*q - dmin*m[s] for element i's sub-block s,
        // computed by hand (d=1.0, dmin=0.5): zero everywhere except the
        // eight probe positions. `qh[0]` supplies the high bit only for
        // local offset l=0 within each 64-element chunk -- every other
        // element in a sub-block has qh=0, so the baseline (q=0) is
        // `-dmin*m` for every non-probe element, same as q4_k's fixture;
        // the trap only shows up at the eight probed indices.
        let mut expected = [0.0f32; QK_K];
        for (sub_block, out) in expected.chunks_exact_mut(SUB_BLOCK_ELEMENTS).enumerate() {
            out.fill(-0.5 * MIN[sub_block] as f32);
        }
        expected[0] = 3.0 * (7.0 + 16.0) - 0.5 * 61.0; // sub_block 0 (even): sc=3, m=61, q=7+16=23 -> 69 - 30.5 = 38.5
        expected[32] = 45.0 * 13.0 - 0.5 * 2.0; // sub_block 1 (odd): sc=45, m=2, q=13 -> 585 - 1 = 584.0
        expected[64] = 12.0 * (12.0 + 16.0) - 0.5 * 44.0; // sub_block 2 (even): sc=12, m=44, q=28 -> 336 - 22 = 314.0
        expected[96] = 63.0 * 2.0 - 0.5 * 9.0; // sub_block 3 (odd): sc=63, m=9, q=2 -> 126 - 4.5 = 121.5
        expected[128] = 33.0 * (1.0 + 16.0) - 0.5 * 50.0; // sub_block 4 (even): sc=33, m=50, q=17 -> 561 - 25 = 536.0
        expected[160] = 7.0 * 15.0 - 0.5 * 19.0; // sub_block 5 (odd): sc=7, m=19, q=15 -> 105 - 9.5 = 95.5
        expected[192] = 58.0 * (9.0 + 16.0) - 0.5 * 6.0; // sub_block 6 (even): sc=58, m=6, q=25 -> 1450 - 3 = 1447.0
        expected[224] = 21.0 * 5.0 - 0.5 * 63.0; // sub_block 7 (odd): sc=21, m=63, q=5 -> 105 - 31.5 = 73.5

        let mut output = [0.0f32; QK_K];
        dequantize(&block, &mut output).expect("well-formed single block");
        assert_eq!(output, expected);
    }

    /// All-zero input hits `make_qkx2_quants`'s `max == min` fast path
    /// (`ggml-quants.c:563-566`) exactly: `scale = 0`, `min = 0`. The
    /// round trip must be bit-exact, not merely close.
    #[test]
    fn quantize_dequantize_zero_vector_is_bit_exact() {
        let input = vec![0.0f32; QK_K];
        let mut packed = vec![0u8; BLOCK_BYTES];
        quantize(&input, &mut packed).expect("one block");
        let mut output = vec![0.0f32; QK_K];
        dequantize(&packed, &mut output).expect("one block");
        assert_eq!(output, input);
    }

    /// A degenerate control that is NOT the trivial all-zero case: every
    /// element equal to a nonzero constant -- same control [`super::q4_k`]
    /// and [`super::q6_k`] both run, exercising the real
    /// weighted-least-squares path and the `d`/`dmin` `f16` rounding.
    #[test]
    fn quantize_dequantize_constant_nonzero_vector_is_near_exact() {
        let input = vec![5.0f32; QK_K];
        let mut packed = vec![0u8; BLOCK_BYTES];
        quantize(&input, &mut packed).expect("one block");
        let mut output = vec![0.0f32; QK_K];
        dequantize(&packed, &mut output).expect("one block");
        let max_error = output
            .iter()
            .zip(input.iter())
            .map(|(got, want)| (got - want).abs())
            .fold(0.0f32, f32::max);
        eprintln!("q5_k_constant_nonzero_vector max_error={max_error}");
        assert!(
            max_error < 0.01,
            "constant-vector round trip should be near-exact, measured max_error={max_error}"
        );
    }

    /// Round-trips a smooth, multi-block, non-degenerate signal and
    /// reports (does not hide) the measured max and RMS error. `Q5_K`
    /// sits between `Q4_K`'s 4.5 bits/weight and `Q6_K`'s 6.5625: bounds
    /// (`0.3` max, `0.1` RMS) are chosen strictly between `q4_k`'s
    /// (`0.6`/`0.2`) and `q6_k`'s (`0.15`/`0.05`), not tuned to the
    /// measured numbers.
    #[test]
    fn quantize_dequantize_smooth_signal_round_trip_error() {
        let elements = QK_K * 4;
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
        eprintln!("q5_k_smooth_signal_round_trip max_error={max_error} rms_error={rms_error}");
        assert!(max_error < 0.3, "max_error={max_error} exceeds loose sanity bound");
        assert!(rms_error < 0.1, "rms_error={rms_error} exceeds loose sanity bound");
    }

    #[test]
    fn dequantize_rejects_non_block_multiple_length() {
        let data = vec![0u8; BLOCK_BYTES - 1];
        let mut output = vec![0.0f32; QK_K];
        let error = dequantize(&data, &mut output).unwrap_err();
        assert_eq!(
            error,
            QuantError::InputNotBlockMultiple {
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

    #[test]
    fn quantize_rejects_non_element_multiple_length() {
        let input = vec![0.0f32; QK_K - 1];
        let mut output = vec![0u8; BLOCK_BYTES];
        let error = quantize(&input, &mut output).unwrap_err();
        assert_eq!(
            error,
            QuantError::InputNotElementMultiple {
                found: QK_K - 1,
                block_elements: QK_K,
            }
        );
    }

    #[test]
    fn quantize_rejects_output_size_mismatch() {
        let input = vec![0.0f32; QK_K];
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
        let mut output = vec![0.0f32; QK_K];
        let error = dequantize(&data, &mut output).unwrap_err();
        assert_eq!(
            error,
            QuantError::InputNotBlockMultiple {
                found: partial_bytes,
                block_bytes: BLOCK_BYTES,
            }
        );
    }
}
