//! `Q4_K`: 4-bit weights in 256-element super-blocks, each split into 8
//! sub-blocks of 32 with its own 6-bit (scale, min) pair. `x = d*sc*q -
//! dmin*m` per sub-block (`ggml-quants.c:1281-1292`, cited on
//! [`dequantize_block`]).
//!
//! Layout, from llama.cpp (`/Users/brianbruggeman/repos/others/llama.cpp`):
//! `ggml/src/ggml-common.h:281-296` — `block_q4_K` is 144 bytes per 256
//! elements (`ggml_half d`, `ggml_half dmin`, `uint8_t scales[12]`
//! (`K_SCALE_SIZE`), `uint8_t qs[128]` (`QK_K/2`)). `QK_K` is 256 and
//! `K_SCALE_SIZE` is 12 (`ggml-common.h:89-90`). That 144-byte figure is
//! cross-checked here at compile time against
//! [`crate::types::GgmlType::Q4_K`]'s already-landed
//! [`crate::types::GgmlType::block_layout`] rather than re-typed by hand.

use thiserror::Error;

use crate::types::GgmlType;

/// Elements per super-block (`ggml-common.h:89`, `#define QK_K 256`).
pub const QK_K: usize = 256;

/// Elements per sub-block: `QK_K` is split into 8 sub-blocks of 32
/// (`ggml-common.h:284`, "8 blocks of 32 elements each").
pub const SUB_BLOCK_ELEMENTS: usize = 32;

/// Sub-blocks per super-block.
pub const SUB_BLOCKS: usize = QK_K / SUB_BLOCK_ELEMENTS;

/// Bytes of bit-packed (scale, min) pairs per super-block
/// (`ggml-common.h:90`, `#define K_SCALE_SIZE 12`).
pub const K_SCALE_SIZE: usize = 12;

/// Bytes per super-block. Derived from the already-landed
/// [`crate::types::GgmlType::block_layout`] (`types.rs`), not re-typed —
/// that table was itself checked against `ggml-common.h`'s
/// `static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_half) +
/// K_SCALE_SIZE + QK_K/2, ...)` at `ggml-common.h:296`.
pub const BLOCK_BYTES: usize = {
    let layout = GgmlType::Q4_K.block_layout();
    assert!(layout.block_elements as usize == QK_K, "GgmlType::Q4_K block_elements drifted from QK_K");
    layout.block_bytes as usize
};

const D_OFFSET: usize = 0;
const DMIN_OFFSET: usize = 2;
const SCALES_OFFSET: usize = 4;
const QS_OFFSET: usize = SCALES_OFFSET + K_SCALE_SIZE;

/// Everything that can go wrong sizing a `Q4_K` codec call. Never a
/// panic: a malformed or mis-sized buffer is always an `Err`.
#[derive(Debug, Error, PartialEq, Eq, Clone, Copy)]
pub enum QuantError {
    #[error("input length {found} bytes is not a multiple of the q4_k block size {block_bytes}")]
    InputNotBlockMultiple { found: usize, block_bytes: usize },
    #[error("input length {found} elements is not a multiple of the q4_k super-block size {block_elements}")]
    InputNotElementMultiple {
        found: usize,
        block_elements: usize,
    },
    #[error("output slice has {found} elements, expected {expected}")]
    OutputSizeMismatch { found: usize, expected: usize },
}

/// Number of whole `Q4_K` super-blocks a byte run decodes to, or `None`
/// if `byte_len` is not an exact multiple of [`BLOCK_BYTES`].
#[must_use]
pub const fn blocks_for_bytes(byte_len: usize) -> Option<usize> {
    if byte_len.is_multiple_of(BLOCK_BYTES) {
        Some(byte_len / BLOCK_BYTES)
    } else {
        None
    }
}

/// Exact packed byte length for `block_count` `Q4_K` super-blocks.
#[must_use]
pub const fn bytes_for_blocks(block_count: usize) -> usize {
    block_count * BLOCK_BYTES
}

/// Exact `f32` element count for `block_count` `Q4_K` super-blocks.
#[must_use]
pub const fn elements_for_blocks(block_count: usize) -> usize {
    block_count * QK_K
}

/// Ties-to-even rounding, porting the IEEE-754 magic-number trick in
/// `ggml-quants.c:366-371` (`nearest_int`) bit-for-bit rather than reaching
/// for a float-rounding method: adding `12582912.0` (`1.5 * 2^23`) forces
/// the FP adder to round `value` to the mantissa's integer precision under
/// the default round-to-nearest-even mode, after which the low 23 bits of
/// the sum's bit pattern are the rounded integer plus a `2^22` bias to
/// keep it positive through the add. This also sidesteps `f32::round` /
/// `round_ties_even` not existing on `core` (no_std, no libm) — the bit
/// trick is core-only arithmetic, needed here regardless of tier.
fn nearest_int(value: f32) -> i32 {
    let shifted = value + 12_582_912.0;
    let bits = shifted.to_bits();
    (bits & 0x007f_ffff) as i32 - 0x0040_0000
}

/// Unpacks one sub-block's 6-bit scale and min out of the 12-byte
/// `scales` field. Ports `get_scale_min_k4` (`ggml-quants.c:625-632`)
/// exactly — the packing is bit-interleaved across non-adjacent bytes,
/// not a fixed stride, and branches on `sub_block < 4`.
fn get_scale_min_k4(sub_block: usize, scales: &[u8; K_SCALE_SIZE]) -> (u8, u8) {
    if sub_block < 4 {
        (scales[sub_block] & 63, scales[sub_block + 4] & 63)
    } else {
        let scale = (scales[sub_block + 4] & 0x0F) | ((scales[sub_block - 4] >> 6) << 4);
        let min = (scales[sub_block + 4] >> 4) | ((scales[sub_block] >> 6) << 4);
        (scale, min)
    }
}

/// Dequantizes one 256-element `Q4_K` super-block. `block` must be
/// exactly [`BLOCK_BYTES`] bytes and `output` exactly [`QK_K`] elements —
/// callers go through [`dequantize`], which validates both.
///
/// Ports `dequantize_row_q4_K` (`ggml-quants.c:1274-1297`) exactly,
/// including its nibble order: a `qs` byte's low nibble and high nibble
/// land 32 output elements apart, not adjacent (`ggml-quants.c:1291-1292`
/// -- the two inner `l in 0..32` loops write elements `[0,32)` from the
/// low nibbles of `qs[0..32)` and `[32,64)` from the *same 32 bytes'*
/// high nibbles, before the `qs` cursor advances).
fn dequantize_block(block: &[u8], output: &mut [f32]) {
    let d = f16_at(block, D_OFFSET).to_f32();
    let dmin = f16_at(block, DMIN_OFFSET).to_f32();
    let mut scales = [0u8; K_SCALE_SIZE];
    scales.copy_from_slice(&block[SCALES_OFFSET..SCALES_OFFSET + K_SCALE_SIZE]);
    let qs = &block[QS_OFFSET..QS_OFFSET + QK_K / 2];

    let mut sub_block = 0usize;
    let mut qs_offset = 0usize;
    let mut out_offset = 0usize;
    for _ in 0..QK_K / 64 {
        let (scale_lo, min_lo) = get_scale_min_k4(sub_block, &scales);
        let (scale_hi, min_hi) = get_scale_min_k4(sub_block + 1, &scales);
        let scale_lo = d * f32::from(scale_lo);
        let min_lo = dmin * f32::from(min_lo);
        let scale_hi = d * f32::from(scale_hi);
        let min_hi = dmin * f32::from(min_hi);

        for offset in 0..SUB_BLOCK_ELEMENTS {
            output[out_offset + offset] = scale_lo * f32::from(qs[qs_offset + offset] & 0x0F) - min_lo;
            output[out_offset + SUB_BLOCK_ELEMENTS + offset] =
                scale_hi * f32::from(qs[qs_offset + offset] >> 4) - min_hi;
        }

        qs_offset += SUB_BLOCK_ELEMENTS;
        out_offset += 64;
        sub_block += 2;
    }
}

fn f16_at(block: &[u8], offset: usize) -> half::f16 {
    let mut bytes = [0u8; 2];
    bytes.copy_from_slice(&block[offset..offset + 2]);
    half::f16::from_le_bytes(bytes)
}

/// Dequantizes a run of `Q4_K` super-blocks. `data` is borrowed, `output`
/// is caller-provided — no allocation on this path.
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
/// [`SUB_BLOCK_ELEMENTS`] (32, the only size `Q4_K` calls it with) and
/// `use_mad` fixed at `false` (squared error, matching
/// `quantize_row_q4_K_ref`'s call at `ggml-quants.c:1221`). Grid-searches
/// 21 candidate scales (`rmin=-1.0`, `rdelta=0.1`, `nstep=20`), and for
/// each solves the weighted normal equations for the best affine fit
/// given that candidate's rounded levels, keeping whichever candidate
/// minimizes weighted squared error.
fn make_qkx2_quants_32(x: &[f32; SUB_BLOCK_ELEMENTS], weights: &[f32; SUB_BLOCK_ELEMENTS]) -> ([u8; SUB_BLOCK_ELEMENTS], f32, f32) {
    const NMAX: i32 = 15;
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
/// `quantize_row_q4_K_ref` (`ggml-quants.c:1232-1244`) — the exact
/// inverse layout [`get_scale_min_k4`] unpacks. `scale`/`min` codes are
/// truncated to `u8` before the `min(63)` clamp, mirroring C's implicit
/// narrowing-then-clamp order (`uint8_t ls = nearest_int(...); ls =
/// MIN(63, ls);`), including its wraparound on a hypothetical negative
/// code.
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

/// Quantizes one 256-element chunk into a `Q4_K` super-block.
///
/// Ports `quantize_row_q4_K_ref` (`ggml-quants.c:1201-1264`) — the
/// reference strategy, not the newer `quantize_row_q4_K_impl`'s
/// `make_qkx3_quants` + `make_qp_quants` p-norm super-block-scale search
/// (`ggml-quants.c:1298-1367`). Per sub-block: importance weights
/// `av_x + |x|` (`av_x` the sub-block's RMS), a [`make_qkx2_quants_32`]
/// search for that sub-block's own (scale, min), then a single linear
/// pass (`code = round(63 * value / max_over_sub_blocks(value))`, no
/// search) to fit all 8 sub-block scales and all 8 mins into their shared
/// 6-bit codes. Levels are then recomputed once against the fp16-rounded
/// packed (scale, min), matching C's re-derivation pass exactly (`if
/// (!d) continue;` included — a zeroed sub-block scale leaves that
/// sub-block's levels as [`make_qkx2_quants_32`] left them).
///
/// Fidelity: the per-sub-block search is byte-for-byte the same
/// algorithm as production. The super-block scale/min encoding is the
/// simpler reference linear fit, not production's iterative p-norm
/// search — a real accuracy gap for `_impl` users, but `_ref` is itself
/// a real, still-shipped llama.cpp code path, not an invented
/// simplification.
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
            levels[index] = nearest_int((x[index] + sub_min) / sub_scale).clamp(0, 15) as u8;
        }
    }

    output[D_OFFSET..D_OFFSET + 2].copy_from_slice(&block_scale.to_le_bytes());
    output[DMIN_OFFSET..DMIN_OFFSET + 2].copy_from_slice(&block_min.to_le_bytes());
    output[SCALES_OFFSET..SCALES_OFFSET + K_SCALE_SIZE].copy_from_slice(&packed_scales);

    let qs = &mut output[QS_OFFSET..QS_OFFSET + QK_K / 2];
    let mut qs_offset = 0usize;
    for base in (0..QK_K).step_by(64) {
        for offset in 0..SUB_BLOCK_ELEMENTS {
            qs[qs_offset + offset] = levels[base + offset] | (levels[base + SUB_BLOCK_ELEMENTS + offset] << 4);
        }
        qs_offset += SUB_BLOCK_ELEMENTS;
    }
}

/// Quantizes a run of `f32` weights into `Q4_K` super-blocks. `input` is
/// borrowed, `output` is caller-provided — no allocation on this path
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
        BLOCK_BYTES, K_SCALE_SIZE, QK_K, QuantError, SUB_BLOCK_ELEMENTS, SUB_BLOCKS, dequantize, quantize,
    };

    /// One super-block, hand-packed and hand-decoded, checked against the
    /// `x = d*sc*q - dmin*m` formula computed by hand — not by calling
    /// [`super::quantize`] to build the fixture. `d=1.0`, `dmin=0.5`
    /// (both exact in `f16`) so every expected value below is an exact
    /// integer or half-integer in `f32`; `assert_eq!` needs no epsilon.
    ///
    /// The 8 sub-block `(scale, min)` pairs, `sc`/`m` below, were packed
    /// into the 12 [`K_SCALE_SIZE`] bytes by hand following
    /// `ggml-quants.c:1232-1244`'s layout (mirrored in
    /// [`super::pack_scales`], not used to build this fixture) and
    /// verified against [`super::get_scale_min_k4`]
    /// (`ggml-quants.c:625-632`) bit by bit in the scratch work this test
    /// was derived from — every `sc`/`m` pair here exercises both the
    /// direct (`sub_block < 4`) and interleaved (`sub_block >= 4`)
    /// branches, including sub-blocks whose 6-bit code needs its top 2
    /// bits (`>= 16`).
    #[test]
    fn dequantize_block_matches_hand_packed_fixture() {
        // sc = [3, 45, 12, 63, 33, 7, 58, 21] (used only in the comments
        // below and the derivation this fixture was built from)
        const MIN: [u32; SUB_BLOCKS] = [61, 2, 44, 9, 50, 19, 6, 63];
        // hand-derived packing (see doc comment): scales[j]=sc[j]|((sc[j+4]>>4)<<6),
        // scales[j+4]=m[j]|((m[j+4]>>4)<<6) for j<4; scales[j+4]=(sc[j]&0xF)|((m[j]&0xF)<<4) for j>=4.
        let packed_scales: [u8; K_SCALE_SIZE] = [131, 45, 204, 127, 253, 66, 44, 201, 33, 55, 106, 245];

        // qs is all zero except one probe byte per 32-byte region, each
        // touching one low-nibble and one high-nibble output element.
        let mut qs = [0u8; QK_K / 2];
        qs[0] = 0xD7; // sub_block 0 low nibble = 7 (elem 0), sub_block 1 high nibble = 13 (elem 32)
        qs[32] = 0x2C; // sub_block 2 low nibble = 12 (elem 64), sub_block 3 high nibble = 2 (elem 96)
        qs[64] = 0xF1; // sub_block 4 low nibble = 1 (elem 128), sub_block 5 high nibble = 15 (elem 160)
        qs[96] = 0x59; // sub_block 6 low nibble = 9 (elem 192), sub_block 7 high nibble = 5 (elem 224)

        let mut block = [0u8; BLOCK_BYTES];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes()); // d
        block[2..4].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes()); // dmin
        block[4..4 + K_SCALE_SIZE].copy_from_slice(&packed_scales);
        block[4 + K_SCALE_SIZE..].copy_from_slice(&qs);

        // expected[i] = d*sc[s]*q - dmin*m[s] for element i's sub-block s,
        // computed by hand (d=1.0, dmin=0.5): zero everywhere except the
        // four probe positions.
        let mut expected = [0.0f32; QK_K];
        for (sub_block, out) in expected.chunks_exact_mut(SUB_BLOCK_ELEMENTS).enumerate() {
            out.fill(-0.5 * MIN[sub_block] as f32);
        }
        expected[0] = 3.0 * 7.0 - 0.5 * 61.0; // sub_block 0: sc=3, m=61, q=7 -> 21 - 30.5 = -9.5
        expected[32] = 45.0 * 13.0 - 0.5 * 2.0; // sub_block 1: sc=45, m=2, q=13 -> 585 - 1 = 584.0
        expected[64] = 12.0 * 12.0 - 0.5 * 44.0; // sub_block 2: sc=12, m=44, q=12 -> 144 - 22 = 122.0
        expected[96] = 63.0 * 2.0 - 0.5 * 9.0; // sub_block 3: sc=63, m=9, q=2 -> 126 - 4.5 = 121.5
        expected[128] = 33.0 * 1.0 - 0.5 * 50.0; // sub_block 4: sc=33, m=50, q=1 -> 33 - 25 = 8.0
        expected[160] = 7.0 * 15.0 - 0.5 * 19.0; // sub_block 5: sc=7, m=19, q=15 -> 105 - 9.5 = 95.5
        expected[192] = 58.0 * 9.0 - 0.5 * 6.0; // sub_block 6: sc=58, m=6, q=9 -> 522 - 3 = 519.0
        expected[224] = 21.0 * 5.0 - 0.5 * 63.0; // sub_block 7: sc=21, m=63, q=5 -> 105 - 31.5 = 73.5

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
    /// element equal to a nonzero constant. `make_qkx2_quants` does not
    /// take its `max == min` shortcut here (it clamps `min` to `<= 0`
    /// first, so `min` becomes `0.0` while `max` stays `5.0`), so this
    /// exercises the real weighted-least-squares path and the `d`/`dmin`
    /// `f16` rounding — round trip should be near-exact, not bit-exact.
    /// If it were not, the implementation has a bug a lossy-error
    /// tolerance elsewhere would hide.
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
        eprintln!("constant_nonzero_vector max_error={max_error}");
        assert!(
            max_error < 0.01,
            "constant-vector round trip should be near-exact, measured max_error={max_error}"
        );
    }

    /// Round-trips a smooth, multi-block, non-degenerate signal and
    /// reports (does not hide) the measured max and RMS error. `4.5`
    /// bits/weight over a value range of roughly `[-3.5, 3.5]` gives a
    /// per-level step around `7.0 / 15.0 ~= 0.47` at the sub-block-scale
    /// level; `0.6` absolute max-error and `0.2` RMS are loose sanity
    /// bounds around that, not tuned to the measured numbers.
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
        eprintln!("smooth_signal_round_trip max_error={max_error} rms_error={rms_error}");
        assert!(max_error < 0.6, "max_error={max_error} exceeds loose sanity bound");
        assert!(rms_error < 0.2, "rms_error={rms_error} exceeds loose sanity bound");
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
}
