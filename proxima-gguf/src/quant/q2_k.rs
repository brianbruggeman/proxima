//! `Q2_K`: 2-bit weights in 256-element super-blocks, split into 16
//! sub-blocks of 16, each with its own 4-bit scale AND 4-bit min packed
//! into a single byte: `x = d*sc*q - dmin*m` per sub-block
//! (`ggml-quants.c`, `dequantize_row_q2_K`) — the same affine shape
//! [`super::q4_k`]/[`super::q5_k`] use, but with a plain 4-bit code per
//! field instead of their bit-interleaved 6-bit scheme, since 15 (the max
//! 4-bit code) already covers every value a `Q2_K` sub-block scale/min
//! needs.
//!
//! Layout, from llama.cpp (`/Users/brianbruggeman/repos/others/llama.cpp`):
//! `ggml/src/ggml-common.h` — `block_q2_K` is 84 bytes per 256 elements
//! (`uint8_t scales[16]` (`QK_K/16`, one byte per sub-block: low nibble
//! scale code, high nibble min code), `uint8_t qs[64]` (`QK_K/4`, 2-bit
//! quants), `ggml_half d`, `ggml_half dmin`) — `d`/`dmin` trail the block,
//! same trailing position [`super::q4_k`]/[`super::q5_k`] use. That
//! 84-byte figure is cross-checked here at compile time against
//! [`crate::types::GgmlType::Q2_K`]'s already-landed
//! [`crate::types::GgmlType::block_layout`] rather than re-typed by hand.
//!
//! Shared with [`super::q4_k`]/[`super::q5_k`]: the affine `x =
//! d*sc*q - dmin*m` per-sub-block shape with both a scale and a min. Not
//! shared: `Q2_K`'s scale/min pair packs into one plain byte per
//! sub-block (`scales[16]`, no bit-interleaving across bytes), where
//! `Q4_K`/`Q5_K` bit-interleave 8 six-bit `(scale, min)` pairs into 12
//! bytes — `Q2_K` can afford the simpler packing because its 16
//! sub-blocks each need only a 4-bit code, not 6.

use crate::quant::QuantError;
use crate::types::GgmlType;

/// This codec's name as it appears in a rendered [`QuantError`] message.
const CODEC: &str = "q2_k";

/// Elements per super-block (`ggml-common.h:89`, `#define QK_K 256`).
pub const QK_K: usize = 256;

/// Elements per sub-block: `QK_K` is split into 16 sub-blocks of 16
/// (`ggml-quants.c`, `dequantize_row_q2_K`'s `is` index runs `0..16`),
/// same sub-block width [`super::q3_k`]/[`super::q6_k`] use.
pub const SUB_BLOCK_ELEMENTS: usize = 16;

/// Sub-blocks per super-block.
pub const SUB_BLOCKS: usize = QK_K / SUB_BLOCK_ELEMENTS;

/// Bytes of packed 4-bit `(scale, min)` pairs per super-block — one byte
/// per sub-block, unlike [`super::q4_k`]/[`super::q5_k`]'s bit-interleaved
/// `K_SCALE_SIZE` (12) bytes for the same pair count.
pub const SCALES_BYTES: usize = SUB_BLOCKS;

const QS_BYTES: usize = QK_K / 4;

/// Bytes per super-block. Derived from the already-landed
/// [`crate::types::GgmlType::block_layout`] (`types.rs`), not re-typed —
/// that table was itself checked against `ggml-common.h`'s
/// `static_assert(sizeof(block_q2_K) == 2*sizeof(ggml_half) + QK_K/16 +
/// QK_K/4, ...)`.
pub const BLOCK_BYTES: usize = {
    let layout = GgmlType::Q2_K.block_layout();
    assert!(
        layout.block_elements as usize == QK_K,
        "GgmlType::Q2_K block_elements drifted from QK_K"
    );
    layout.block_bytes as usize
};

const SCALES_OFFSET: usize = 0;
const QS_OFFSET: usize = SCALES_OFFSET + SCALES_BYTES;
const D_OFFSET: usize = QS_OFFSET + QS_BYTES;
const DMIN_OFFSET: usize = D_OFFSET + 2;

/// Number of whole `Q2_K` super-blocks a byte run decodes to, or `None`
/// if `byte_len` is not an exact multiple of [`BLOCK_BYTES`].
#[must_use]
pub const fn blocks_for_bytes(byte_len: usize) -> Option<usize> {
    if byte_len.is_multiple_of(BLOCK_BYTES) {
        Some(byte_len / BLOCK_BYTES)
    } else {
        None
    }
}

/// Exact packed byte length for `block_count` `Q2_K` super-blocks.
#[must_use]
pub const fn bytes_for_blocks(block_count: usize) -> usize {
    block_count * BLOCK_BYTES
}

/// Exact `f32` element count for `block_count` `Q2_K` super-blocks.
#[must_use]
pub const fn elements_for_blocks(block_count: usize) -> usize {
    block_count * QK_K
}

/// Ties-to-even rounding, porting the IEEE-754 magic-number trick in
/// `ggml-quants.c:366-371` (`nearest_int`) bit-for-bit — see
/// [`super::q4_k::nearest_int`]'s doc for the full derivation; duplicated
/// here rather than shared because each codec module owns its primitives
/// independently, matching this crate's one-format-per-file layout (the
/// same choice [`super::q3_k::nearest_int`] documents).
fn nearest_int(value: f32) -> i32 {
    let shifted = value + 12_582_912.0;
    let bits = shifted.to_bits();
    (bits & 0x007f_ffff) as i32 - 0x0040_0000
}

/// Unpacks one sub-block's 4-bit scale and 4-bit min out of its single
/// `scales` byte: low nibble is the scale code, high nibble is the min
/// code (`ggml-quants.c`, `dequantize_row_q2_K`: `dl = d*(sc&0xF); ml =
/// min*(sc>>4);`).
fn unpack_scale_min(sub_block: usize, scales: &[u8; SCALES_BYTES]) -> (u8, u8) {
    let byte = scales[sub_block];
    (byte & 0x0F, byte >> 4)
}

/// Packs one sub-block's 4-bit scale and 4-bit min codes into their
/// shared byte — the exact inverse [`unpack_scale_min`] reads back.
fn pack_scale_min(sub_block: usize, scale_code: u8, min_code: u8, scales: &mut [u8; SCALES_BYTES]) {
    scales[sub_block] = (scale_code & 0x0F) | ((min_code & 0x0F) << 4);
}

/// Dequantizes one 256-element `Q2_K` super-block. `block` must be
/// exactly [`BLOCK_BYTES`] bytes and `output` exactly [`QK_K`] elements —
/// callers go through [`dequantize`], which validates both.
///
/// Ports `dequantize_row_q2_K` (`ggml-quants.c`) exactly: two 128-element
/// chunks, each reading a fixed 32-byte `qs` window; within a chunk, 4
/// passes at `shift in [0, 2, 4, 6]`, each pass consuming two sub-blocks'
/// worth of `scales` (one for `qs[0..16)`, one for `qs[16..32)` of the
/// same window) — the same "shared window, advancing shift" shape
/// [`super::q3_k::dequantize_block`]'s `qs_window` documents, simpler here
/// because there is no separate `hmask` plane to index.
pub fn dequantize_block(block: &[u8], output: &mut [f32]) {
    let d = f16_at(block, D_OFFSET).to_f32();
    let dmin = f16_at(block, DMIN_OFFSET).to_f32();
    let mut scales = [0u8; SCALES_BYTES];
    scales.copy_from_slice(&block[SCALES_OFFSET..SCALES_OFFSET + SCALES_BYTES]);
    let qs = &block[QS_OFFSET..QS_OFFSET + QS_BYTES];

    // Per sub-block decode plan (dl, ml, shift, its own 16-byte window into
    // `qs`), built once in output order so the write pass below is a plain
    // zipped iteration with no fallible indexing.
    let mut sub_block = 0usize;
    let mut plan: [(f32, f32, u32, &[u8]); SUB_BLOCKS] = [(0.0, 0.0, 0, &[]); SUB_BLOCKS];
    for chunk in 0..(QK_K / 128) {
        let qs_window = &qs[chunk * 32..chunk * 32 + 32];
        let (low_half, high_half) = qs_window.split_at(SUB_BLOCK_ELEMENTS);
        let mut shift = 0u32;
        for _ in 0..4 {
            let (scale_lo, min_lo) = unpack_scale_min(sub_block, &scales);
            plan[sub_block] = (d * f32::from(scale_lo), dmin * f32::from(min_lo), shift, low_half);
            sub_block += 1;

            let (scale_hi, min_hi) = unpack_scale_min(sub_block, &scales);
            plan[sub_block] = (d * f32::from(scale_hi), dmin * f32::from(min_hi), shift, high_half);
            sub_block += 1;

            shift += 2;
        }
    }

    for (out_chunk, &(dl, ml, shift, window)) in output
        .as_chunks_mut::<SUB_BLOCK_ELEMENTS>()
        .0
        .iter_mut()
        .zip(plan.iter())
    {
        for (out_value, &packed) in out_chunk.iter_mut().zip(window) {
            let level = (packed >> shift) & 0x03;
            *out_value = dl * f32::from(level) - ml;
        }
    }
}

fn f16_at(block: &[u8], offset: usize) -> half::f16 {
    let mut bytes = [0u8; 2];
    bytes.copy_from_slice(&block[offset..offset + 2]);
    half::f16::from_le_bytes(bytes)
}

/// Dequantizes a run of `Q2_K` super-blocks. `data` is borrowed, `output`
/// is caller-provided — no allocation on this path.
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
        .zip(output.as_chunks_mut::<QK_K>().0)
    {
        dequantize_block(block, out_chunk);
    }
    Ok(())
}

/// One sub-block's weighted-least-squares (scale, min) search: candidate
/// levels `L[i] in [0, nmax]`, the affine map `x[i] ~= scale*L[i] - min`
/// (note the sign: this function returns `min` such that the caller
/// reconstructs `scale*L - min`, matching [`super::q4_k`]'s own
/// `make_qkx2_quants_32` convention).
///
/// Ports `make_qkx2_quants` (`ggml-quants.c:544-623`) with `n` fixed at
/// [`SUB_BLOCK_ELEMENTS`] (16, the only size `Q2_K` calls it with) and
/// `nmax` fixed at 3 (2-bit levels) — the same grid-search shape
/// [`super::q4_k::make_qkx2_quants_32`] documents in full, duplicated here
/// (not shared) at a different `n`/`nmax` instantiation, matching this
/// crate's one-primitive-per-module convention.
fn make_qkx2_quants_16(
    x: &[f32; SUB_BLOCK_ELEMENTS],
    weights: &[f32; SUB_BLOCK_ELEMENTS],
) -> ([u8; SUB_BLOCK_ELEMENTS], f32, f32) {
    const NMAX: i32 = 3;
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
            let diff =
                candidate_scale * f32::from(candidate_levels[index]) + candidate_min - x[index];
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

/// Quantizes one 256-element chunk into a `Q2_K` super-block.
///
/// Ports `quantize_row_q2_K_ref` (`ggml-quants.c`): per sub-block,
/// importance weights `|x|` (unlike [`super::q4_k`]'s `av_x + |x|`), a
/// [`make_qkx2_quants_16`] search for that sub-block's own (scale, min),
/// then a single linear pass (`code = round(15 * value /
/// max_over_sub_blocks(value))`, no search) fitting all 16 sub-block
/// scales and all 16 mins into their own plain 4-bit codes (`q4scale =
/// 15`, unlike `Q4_K`'s bit-interleaved 6-bit codes). Levels are then
/// recomputed once against the fp16-rounded packed (scale, min), matching
/// C's re-derivation pass exactly (`if (!d) continue;` included).
fn quantize_block(x: &[f32], output: &mut [u8]) {
    let mut levels = [0u8; QK_K];
    let mut mins = [0.0f32; SUB_BLOCKS];
    let mut scales = [0.0f32; SUB_BLOCKS];

    for sub_block in 0..SUB_BLOCKS {
        let mut chunk = [0.0f32; SUB_BLOCK_ELEMENTS];
        chunk.copy_from_slice(
            &x[sub_block * SUB_BLOCK_ELEMENTS..(sub_block + 1) * SUB_BLOCK_ELEMENTS],
        );
        let mut weights = [0.0f32; SUB_BLOCK_ELEMENTS];
        for (weight, value) in weights.iter_mut().zip(chunk.iter()) {
            *weight = value.abs();
        }
        let (sub_levels, sub_min, sub_scale) = make_qkx2_quants_16(&chunk, &weights);
        levels[sub_block * SUB_BLOCK_ELEMENTS..(sub_block + 1) * SUB_BLOCK_ELEMENTS]
            .copy_from_slice(&sub_levels);
        mins[sub_block] = sub_min;
        scales[sub_block] = sub_scale;
    }

    let max_scale = scales.iter().copied().fold(0.0f32, f32::max);
    let max_min = mins.iter().copied().fold(0.0f32, f32::max);
    const Q4SCALE: f32 = 15.0;
    let scale_step = if max_scale > 0.0 { Q4SCALE / max_scale } else { 0.0 };
    let min_step = if max_min > 0.0 { Q4SCALE / max_min } else { 0.0 };

    let mut packed_scales = [0u8; SCALES_BYTES];
    for sub_block in 0..SUB_BLOCKS {
        let scale_code = (nearest_int(scale_step * scales[sub_block]) as u8).min(15);
        let min_code = (nearest_int(min_step * mins[sub_block]) as u8).min(15);
        pack_scale_min(sub_block, scale_code, min_code, &mut packed_scales);
    }

    let block_scale = half::f16::from_f32(max_scale / Q4SCALE);
    let block_min = half::f16::from_f32(max_min / Q4SCALE);

    for sub_block in 0..SUB_BLOCKS {
        let (scale_code, min_code) = unpack_scale_min(sub_block, &packed_scales);
        let sub_scale = block_scale.to_f32() * f32::from(scale_code);
        if sub_scale == 0.0 {
            continue;
        }
        let sub_min = block_min.to_f32() * f32::from(min_code);
        for offset in 0..SUB_BLOCK_ELEMENTS {
            let index = sub_block * SUB_BLOCK_ELEMENTS + offset;
            levels[index] = nearest_int((x[index] + sub_min) / sub_scale).clamp(0, 3) as u8;
        }
    }

    output[D_OFFSET..D_OFFSET + 2].copy_from_slice(&block_scale.to_le_bytes());
    output[DMIN_OFFSET..DMIN_OFFSET + 2].copy_from_slice(&block_min.to_le_bytes());
    output[SCALES_OFFSET..SCALES_OFFSET + SCALES_BYTES].copy_from_slice(&packed_scales);

    let qs = &mut output[QS_OFFSET..QS_OFFSET + QS_BYTES];
    for base in (0..QK_K).step_by(128) {
        for local in 0..32 {
            let low = levels[base + local];
            let mid_low = levels[base + local + 32];
            let mid_high = levels[base + local + 64];
            let high = levels[base + local + 96];
            qs[base / 4 + local] = low | (mid_low << 2) | (mid_high << 4) | (high << 6);
        }
    }
}

/// Quantizes a run of `f32` weights into `Q2_K` super-blocks. `input` is
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
            codec: CODEC,
            unit: "super-block",
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
    for (chunk, out_block) in input
        .as_chunks::<QK_K>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<BLOCK_BYTES>().0)
    {
        quantize_block(chunk, out_block);
    }
    Ok(())
}
