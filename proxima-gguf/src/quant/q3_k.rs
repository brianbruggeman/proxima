//! `Q3_K`: 3-bit weights in 256-element super-blocks, split into 16
//! sub-blocks of 16 with its own signed 6-bit scale — no per-sub-block
//! min, unlike [`super::q4_k`]/[`super::q5_k`]: `x = d*sc*q`
//! (`ggml-common.h:270`, "weight is represented as `x = a * q`").
//! `q` is a 3-bit value assembled from a 2-bit `qs` lane and one
//! `hmask` high bit, biased so the raw 2-bit field reads `q+4` when the
//! high bit is clear and `q` directly when it is set
//! (`ggml-quants.c:1050-1098`, cited on [`dequantize_block`]).
//!
//! Layout, from llama.cpp (`/Users/brianbruggeman/repos/others/llama.cpp`):
//! `ggml/src/ggml-common.h:273-278` — `block_q3_K` is 110 bytes per 256
//! elements (`uint8_t hmask[32]` (`QK_K/8`, one high bit per element),
//! `uint8_t qs[64]` (`QK_K/4`, low 2 bits), `uint8_t scales[12]`
//! (`K_SCALE_SIZE`, six-bit signed scales), `ggml_half d`) — `d` trails
//! the block, same trailing position [`super::q6_k`] uses, unlike
//! [`super::q4_k`]/[`super::q5_k`]/[`super::q8_0`] where it leads. That
//! 110-byte figure is cross-checked here at compile time against
//! [`crate::types::GgmlType::Q3_K`]'s already-landed
//! [`crate::types::GgmlType::block_layout`] rather than re-typed by hand.
//!
//! Shared with [`super::q6_k`]: the super-block/sub-block shape (16
//! sub-blocks of 16), no per-sub-block min (`x = d*sc*q`, not
//! `d*sc*q - dmin*m`). Not shared: `Q3_K`'s scale is packed into
//! [`K_SCALE_SIZE`] (12) bytes via the same bit-interleaved 6-bit scheme
//! [`super::q4_k`]/[`super::q5_k`] use for their `(scale, min)` pairs
//! (here carrying only a scale, no min), whereas `Q6_K` packs 16 plain
//! signed 8-bit scale bytes; and `Q3_K`'s 3-bit level is a 2-bit `qs`
//! lane plus one `hmask` high bit, not `Q6_K`'s 6-bit two-plane split.

use crate::quant::QuantError;
use crate::types::GgmlType;

/// This codec's name as it appears in a rendered [`QuantError`] message.
const CODEC: &str = "q3_k";

/// Elements per super-block (`ggml-common.h:89`, `#define QK_K 256`).
pub const QK_K: usize = 256;

/// Elements per sub-block: `QK_K` is split into 16 sub-blocks of 16
/// (`ggml-quants.c:1116`, `for (j = 0; j < QK_K/16; ++j)`), same
/// sub-block width [`super::q6_k`] uses.
pub const SUB_BLOCK_ELEMENTS: usize = 16;

/// Sub-blocks per super-block.
pub const SUB_BLOCKS: usize = QK_K / SUB_BLOCK_ELEMENTS;

/// Bytes of bit-packed signed 6-bit scales per super-block
/// (`ggml-common.h:90`, `#define K_SCALE_SIZE 12`) — same constant name
/// and byte count [`super::q4_k`]/[`super::q5_k`] pack a `(scale, min)`
/// pair into; here it carries 16 scale-only codes instead of 8 pairs.
pub const K_SCALE_SIZE: usize = 12;

const HMASK_BYTES: usize = QK_K / 8;
const QS_BYTES: usize = QK_K / 4;

/// Bytes per super-block. Derived from the already-landed
/// [`crate::types::GgmlType::block_layout`] (`types.rs`), not re-typed —
/// that table was itself checked against `ggml-common.h`'s
/// `static_assert(sizeof(block_q3_K) == sizeof(ggml_half) + QK_K/4 +
/// QK_K/8 + 12, ...)` at `ggml-common.h:279`.
pub const BLOCK_BYTES: usize = {
    let layout = GgmlType::Q3_K.block_layout();
    assert!(
        layout.block_elements as usize == QK_K,
        "GgmlType::Q3_K block_elements drifted from QK_K"
    );
    layout.block_bytes as usize
};

const HMASK_OFFSET: usize = 0;
const QS_OFFSET: usize = HMASK_OFFSET + HMASK_BYTES;
const SCALES_OFFSET: usize = QS_OFFSET + QS_BYTES;
const D_OFFSET: usize = SCALES_OFFSET + K_SCALE_SIZE;

/// Below this absolute max sub-block scale, [`make_q3_quants_16`] treats
/// the sub-block as all-zero (`ggml-quants.c:16`, `#define
/// GROUP_MAX_EPS 1e-15f`).
const GROUP_MAX_EPS: f32 = 1e-15;

/// Number of whole `Q3_K` super-blocks a byte run decodes to, or `None`
/// if `byte_len` is not an exact multiple of [`BLOCK_BYTES`].
#[must_use]
pub const fn blocks_for_bytes(byte_len: usize) -> Option<usize> {
    if byte_len.is_multiple_of(BLOCK_BYTES) {
        Some(byte_len / BLOCK_BYTES)
    } else {
        None
    }
}

/// Exact packed byte length for `block_count` `Q3_K` super-blocks.
#[must_use]
pub const fn bytes_for_blocks(block_count: usize) -> usize {
    block_count * BLOCK_BYTES
}

/// Exact `f32` element count for `block_count` `Q3_K` super-blocks.
#[must_use]
pub const fn elements_for_blocks(block_count: usize) -> usize {
    block_count * QK_K
}

/// Ties-to-even rounding, porting the IEEE-754 magic-number trick in
/// `ggml-quants.c:366-371` (`nearest_int`) bit-for-bit — see
/// [`super::q4_k`]'s copy of this same function for the full derivation;
/// duplicated here rather than shared because each codec module owns its
/// primitives independently, matching this crate's one-format-per-file
/// layout.
fn nearest_int(value: f32) -> i32 {
    let shifted = value + 12_582_912.0;
    let bits = shifted.to_bits();
    (bits & 0x007f_ffff) as i32 - 0x0040_0000
}

/// Unpacks one sub-block's signed 6-bit scale out of the 12-byte
/// `scales` field, biased-unbiased form (`0..63` code minus 32). Ports
/// the inline unpack `quantize_row_q3_K_ref` re-runs on its own just-packed
/// bytes to recompute each sub-block's rounded scale
/// (`ggml-quants.c:1010-1012`) — algebraically the same byte layout
/// [`dequantize_block`]'s `aux`-word shuffle reads, just index-at-a-time
/// instead of four-lanes-at-once.
fn unpack_scale(sub_block: usize, scales: &[u8; K_SCALE_SIZE]) -> i8 {
    let low = if sub_block < 8 {
        scales[sub_block] & 0x0F
    } else {
        scales[sub_block - 8] >> 4
    };
    let high = (scales[8 + sub_block % 4] >> (2 * (sub_block / 4))) & 0x03;
    let combined = low | (high << 4);
    combined as i8 - 32
}

/// Packs one sub-block's already-biased 6-bit scale code (`0..63`) into
/// the 12-byte [`K_SCALE_SIZE`] field. Ports the packing half of
/// `quantize_row_q3_K_ref` (`ggml-quants.c:996-1006`) — the exact inverse
/// [`unpack_scale`] reads back. Callers zero `scales` once up front and
/// call this once per sub-block (`0..`[`SUB_BLOCKS`]), same as the C
/// loop's `|=` accumulation.
fn pack_scale(sub_block: usize, code: u8, scales: &mut [u8; K_SCALE_SIZE]) {
    if sub_block < 8 {
        scales[sub_block] |= code & 0x0F;
    } else {
        scales[sub_block - 8] |= (code & 0x0F) << 4;
    }
    scales[8 + sub_block % 4] |= (code >> 4) << (2 * (sub_block / 4));
}

/// Dequantizes one 256-element `Q3_K` super-block. `block` must be
/// exactly [`BLOCK_BYTES`] bytes and `output` exactly [`QK_K`] elements —
/// callers go through [`dequantize`], which validates both.
///
/// Ports `dequantize_row_q3_K` (`ggml-quants.c:1050-1098`) exactly,
/// including its `aux`-word scale unshuffle (four `uint32_t` lanes
/// reinterpreted as sixteen signed bytes) and its two nested-loop
/// indexing traps: `qs` is read through a 32-byte window that advances by
/// 32 bytes every 128 output elements (`q += 32` once per outer `n`
/// iteration) while `hmask` is read through the *same* fixed 32-byte
/// range for both halves of the super-block, distinguished only by which
/// bit of the shared `m` mask applies (`m` shifts once per inner `j`
/// iteration and is never reset, so its 8 values span both outer `n`
/// iterations) — the same "shared bit, different byte range" shape
/// [`super::q5_k`]'s `qh` indexing trap documents, inverted: there `qh`'s
/// byte index is fixed and the *mask* rotates per chunk; here `hmask`'s
/// byte index differs per half (`0..16` vs `16..32`) but is read with the
/// *same* mask bit for both halves of a given `(chunk, j)` pair.
pub fn dequantize_block(block: &[u8], output: &mut [f32]) {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;

    let d_all = f16_at(block, D_OFFSET).to_f32();
    let hmask = &block[HMASK_OFFSET..HMASK_OFFSET + HMASK_BYTES];
    let qs = &block[QS_OFFSET..QS_OFFSET + QS_BYTES];

    let mut raw_scales = [0u8; K_SCALE_SIZE];
    raw_scales.copy_from_slice(&block[SCALES_OFFSET..SCALES_OFFSET + K_SCALE_SIZE]);
    let mut aux = [0u32; 4];
    for (word, chunk) in aux.iter_mut().zip(raw_scales.as_chunks::<4>().0) {
        *word = u32::from_le_bytes(*chunk);
    }
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    let mut scale_bytes = [0u8; SUB_BLOCKS];
    for (word, chunk) in aux.iter().zip(scale_bytes.as_chunks_mut::<4>().0) {
        *chunk = word.to_le_bytes();
    }

    let mut mask = 1u8;
    let mut sub_block = 0usize;
    let mut out_offset = 0usize;
    for chunk in 0..(QK_K / 128) {
        let qs_window = &qs[chunk * 32..chunk * 32 + 32];
        let mut shift = 0u32;
        for _ in 0..4 {
            let scale_lo = d_all * f32::from(scale_bytes[sub_block] as i8 - 32);
            sub_block += 1;
            for local in 0..SUB_BLOCK_ELEMENTS {
                let level = (qs_window[local] >> shift) & 0x03;
                let correction = if hmask[local] & mask != 0 { 0.0 } else { 4.0 };
                output[out_offset] = scale_lo * (f32::from(level) - correction);
                out_offset += 1;
            }

            let scale_hi = d_all * f32::from(scale_bytes[sub_block] as i8 - 32);
            sub_block += 1;
            for local in 0..SUB_BLOCK_ELEMENTS {
                let index = local + SUB_BLOCK_ELEMENTS;
                let level = (qs_window[index] >> shift) & 0x03;
                let correction = if hmask[index] & mask != 0 { 0.0 } else { 4.0 };
                output[out_offset] = scale_hi * (f32::from(level) - correction);
                out_offset += 1;
            }

            shift += 2;
            mask <<= 1;
        }
    }
}

fn f16_at(block: &[u8], offset: usize) -> half::f16 {
    let mut bytes = [0u8; 2];
    bytes.copy_from_slice(&block[offset..offset + 2]);
    half::f16::from_le_bytes(bytes)
}

/// Dequantizes a run of `Q3_K` super-blocks. `data` is borrowed, `output`
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

/// One sub-block's iterative-RMSE level search, `nmax` fixed at 4
/// (3-bit levels, biased into `[0, 7]`).
///
/// Ports `make_q3_quants` (`ggml-quants.c:442-495`) with `n` fixed at
/// [`SUB_BLOCK_ELEMENTS`] (16, the only size `Q3_K` calls it with) and
/// `do_rmse` fixed at `true` (`quantize_row_q3_K_ref`'s call at
/// `ggml-quants.c:986`) — a different search shape than
/// [`super::q6_k::make_qx_quants_16`]'s grid sweep over candidate
/// inverse scales: this one starts from one `iscale` guess, then
/// iteratively reassigns individual levels (up to 5 passes) whenever
/// doing so improves the weighted least-squares fit, greedily.
fn make_q3_quants_16(x: &[f32; SUB_BLOCK_ELEMENTS]) -> ([i8; SUB_BLOCK_ELEMENTS], f32) {
    const NMAX: i32 = 4;
    const NMAX_F: f32 = NMAX as f32;

    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &value in x {
        let absolute = value.abs();
        if absolute > amax {
            amax = absolute;
            max = value;
        }
    }
    if amax < GROUP_MAX_EPS {
        return ([0i8; SUB_BLOCK_ELEMENTS], 0.0);
    }

    let iscale = -NMAX_F / max;
    let mut levels = [0i32; SUB_BLOCK_ELEMENTS];
    let mut sum_lx = 0.0f32;
    let mut sum_l2 = 0.0f32;
    for (index, &value) in x.iter().enumerate() {
        let level = nearest_int(iscale * value).clamp(-NMAX, NMAX - 1);
        levels[index] = level;
        let weight = value * value;
        sum_lx += weight * value * level as f32;
        sum_l2 += weight * (level * level) as f32;
    }

    for _ in 0..5 {
        let mut changed = 0u32;
        for index in 0..SUB_BLOCK_ELEMENTS {
            let value = x[index];
            let weight = value * value;
            let excluded_lx = sum_lx - weight * value * levels[index] as f32;
            if excluded_lx <= 0.0 {
                continue;
            }
            let excluded_l2 = sum_l2 - weight * (levels[index] * levels[index]) as f32;
            let new_level = nearest_int(value * excluded_l2 / excluded_lx).clamp(-NMAX, NMAX - 1);
            if new_level == levels[index] {
                continue;
            }
            let new_lx = excluded_lx + weight * value * new_level as f32;
            let new_l2 = excluded_l2 + weight * (new_level * new_level) as f32;
            if new_l2 > 0.0 && new_lx * new_lx * sum_l2 > sum_lx * sum_lx * new_l2 {
                levels[index] = new_level;
                sum_lx = new_lx;
                sum_l2 = new_l2;
                changed += 1;
            }
        }
        if changed == 0 {
            break;
        }
    }

    let mut biased = [0i8; SUB_BLOCK_ELEMENTS];
    for (index, level) in levels.iter().enumerate() {
        biased[index] = (level + NMAX) as i8;
    }
    let scale = if sum_l2 > 0.0 { sum_lx / sum_l2 } else { 0.0 };
    (biased, scale)
}

/// Quantizes one 256-element chunk into a `Q3_K` super-block.
///
/// Ports `quantize_row_q3_K_ref` (`ggml-quants.c:974-1048`) — the
/// reference strategy, not the newer `quantize_row_q3_K_impl`'s
/// `quant_weights`-aware search, same trade-off
/// [`super::q4_k::quantize_block`] documents for its own `_impl` gap.
/// Per sub-block: a [`make_q3_quants_16`] search (`nmax=4`) gives a
/// first-pass level set and float scale; the sub-block with the largest
/// `|scale|` sets the super-block's `d`; every sub-block's scale is then
/// re-quantized to a shared signed 6-bit code against that `d`, and (per
/// `ggml-quants.c:1010-1024`) levels are recomputed once more from the
/// *rounded* `(d, sc)` pair — except where the rounded product is exactly
/// zero, which keeps the first pass's levels unchanged, matching C's
/// `if (!d) continue;`. The final pack step splits each 3-bit level
/// (`0..7`) into a 2-bit `qs` lane plus one `hmask` high bit.
fn quantize_block(x: &[f32], output: &mut [u8]) {
    let mut levels = [0i8; QK_K];
    let mut scales = [0.0f32; SUB_BLOCKS];

    for sub_block in 0..SUB_BLOCKS {
        let mut chunk = [0.0f32; SUB_BLOCK_ELEMENTS];
        chunk.copy_from_slice(
            &x[sub_block * SUB_BLOCK_ELEMENTS..(sub_block + 1) * SUB_BLOCK_ELEMENTS],
        );
        let (sub_levels, sub_scale) = make_q3_quants_16(&chunk);
        levels[sub_block * SUB_BLOCK_ELEMENTS..(sub_block + 1) * SUB_BLOCK_ELEMENTS]
            .copy_from_slice(&sub_levels);
        scales[sub_block] = sub_scale;
    }

    let mut max_scale = 0.0f32;
    let mut amax = 0.0f32;
    for &scale in &scales {
        let absolute = scale.abs();
        if absolute > amax {
            amax = absolute;
            max_scale = scale;
        }
    }

    let mut packed_scales = [0u8; K_SCALE_SIZE];
    let block_d = if max_scale != 0.0 {
        let iscale = -32.0 / max_scale;
        for (sub_block, &scale) in scales.iter().enumerate() {
            let code = (nearest_int(iscale * scale).clamp(-32, 31) + 32) as u8;
            pack_scale(sub_block, code, &mut packed_scales);
        }
        half::f16::from_f32(1.0 / iscale)
    } else {
        half::f16::from_f32(0.0)
    };

    for sub_block in 0..SUB_BLOCKS {
        let sc = unpack_scale(sub_block, &packed_scales);
        let dd = block_d.to_f32() * f32::from(sc);
        if dd == 0.0 {
            continue;
        }
        for offset in 0..SUB_BLOCK_ELEMENTS {
            let index = sub_block * SUB_BLOCK_ELEMENTS + offset;
            let level = nearest_int(x[index] / dd).clamp(-4, 3);
            levels[index] = (level + 4) as i8;
        }
    }

    let mut hmask = [0u8; HMASK_BYTES];
    let mut byte_index = 0usize;
    let mut bit = 1u8;
    for level in &mut levels {
        if *level > 3 {
            hmask[byte_index] |= bit;
            *level -= 4;
        }
        byte_index += 1;
        if byte_index == HMASK_BYTES {
            byte_index = 0;
            bit <<= 1;
        }
    }

    let mut qs = [0u8; QS_BYTES];
    for base in (0..QK_K).step_by(128) {
        for local in 0..32 {
            let low = levels[base + local] as u8;
            let mid_low = levels[base + local + 32] as u8;
            let mid_high = levels[base + local + 64] as u8;
            let high = levels[base + local + 96] as u8;
            qs[base / 4 + local] = low | (mid_low << 2) | (mid_high << 4) | (high << 6);
        }
    }

    output[D_OFFSET..D_OFFSET + 2].copy_from_slice(&block_d.to_le_bytes());
    output[HMASK_OFFSET..HMASK_OFFSET + HMASK_BYTES].copy_from_slice(&hmask);
    output[QS_OFFSET..QS_OFFSET + QS_BYTES].copy_from_slice(&qs);
    output[SCALES_OFFSET..SCALES_OFFSET + K_SCALE_SIZE].copy_from_slice(&packed_scales);
}

/// Quantizes a run of `f32` weights into `Q3_K` super-blocks. `input` is
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use alloc::vec;
    use alloc::vec::Vec;

    use proxima_telemetry::debug;

    use super::{
        BLOCK_BYTES, CODEC, HMASK_BYTES, K_SCALE_SIZE, QK_K, QuantError, SUB_BLOCKS, dequantize,
        pack_scale, quantize, unpack_scale,
    };

    /// One super-block, hand-packed and hand-decoded, checked against the
    /// `x = d*sc*q` formula computed by hand — not by calling
    /// [`super::quantize`] to build the fixture. `d=1.0` (exact in
    /// `f16`), so every expected value below is an exact integer in
    /// `f32`; `assert_eq!` needs no epsilon.
    ///
    /// The trap this fixture is built to catch: `hmask`'s byte index runs
    /// `0..16` for the first half of a `(chunk, j)` pair and `16..32` for
    /// the second half, but BOTH halves share the *same* mask bit — only
    /// `qs`'s shift advances within a `chunk`, not `hmask`'s bit. Probing
    /// one element per half per `j`, across all 4 `j` values of `chunk`
    /// 0, exercises 8 of the 16 sub-blocks with 8 distinct mask bits.
    #[test]
    fn dequantize_block_matches_hand_packed_fixture() {
        // sc = [5, -12, 30, -3, 17, -25, 8, -31, ...] for sub-blocks 0..8;
        // sub-blocks 8..16 left at code 32 (sc=0) so their contribution
        // is always zero regardless of qs/hmask content there.
        const SCALE_CODE: [u8; 8] = [37, 20, 62, 29, 49, 7, 40, 1]; // sc = code - 32
        let mut packed_scales = [0u8; K_SCALE_SIZE];
        for (sub_block, &code) in SCALE_CODE.iter().enumerate() {
            pack_scale(sub_block, code, &mut packed_scales);
        }
        for sub_block in 8..SUB_BLOCKS {
            pack_scale(sub_block, 32, &mut packed_scales);
        }
        for (sub_block, &code) in SCALE_CODE.iter().enumerate() {
            assert_eq!(
                unpack_scale(sub_block, &packed_scales),
                code as i8 - 32,
                "packed scale must round-trip through unpack_scale"
            );
        }

        // qs: one probe byte per 32-byte window position covering the 4
        // `j` values of chunk 0 -- byte at local offset `l` supplies both
        // halves' 2-bit fields at shift `2*j`.
        let mut qs = [0u8; QK_K / 4];
        qs[0] = 0b01_10_11_00; // j=0: low half lane=0b00=0, high half lane=0b11=3 (shift 0, bits 0-1)... see below
        qs[1] = 0b11_01_10_11;
        qs[2] = 0b00_11_01_10;
        qs[3] = 0b10_00_11_01;

        // hmask: bits 0-3 used by chunk 0's four `j` values, one bit per
        // `j`, shared between the two halves of that `j`.
        let mut hmask = [0u8; HMASK_BYTES];
        hmask[0] = 0b0000_1010; // j=0 bit(0)=0 (low half not set), j=2 bit(2) set
        hmask[16] = 0b0000_0110; // j=0 high half (byte 16) bit(0) not set; j=1 bit(1) set

        let mut block = [0u8; BLOCK_BYTES];
        block[super::HMASK_OFFSET..super::HMASK_OFFSET + HMASK_BYTES].copy_from_slice(&hmask);
        block[super::QS_OFFSET..super::QS_OFFSET + QK_K / 4].copy_from_slice(&qs);
        block[super::SCALES_OFFSET..super::SCALES_OFFSET + K_SCALE_SIZE]
            .copy_from_slice(&packed_scales);
        block[super::D_OFFSET..super::D_OFFSET + 2]
            .copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());

        let mut output = [0.0f32; QK_K];
        dequantize(&block, &mut output).expect("well-formed single block");

        // Recompute the expected value at element 0 (sub-block 0, local 0,
        // j=0, shift=0, mask=1) directly from the fixture bytes, the same
        // arithmetic `dequantize_block` performs, rather than re-deriving
        // by hand a second time -- this test's job is to catch an indexing
        // regression in the loop structure, not to re-prove the formula
        // (the zero-vector and constant-vector tests below do that).
        let scale0 = f32::from(SCALE_CODE[0] as i8 - 32);
        let level0 = qs[0] & 0x03;
        let correction0 = if hmask[0] & 1 != 0 { 0.0 } else { 4.0 };
        assert_eq!(output[0], scale0 * (f32::from(level0) - correction0));

        // Every element in sub-blocks 8..16 has scale code 32 (sc=0), so
        // the whole second half of the super-block must decode to exactly
        // zero regardless of qs/hmask bits there.
        for &value in &output[128..QK_K] {
            assert_eq!(value, 0.0, "sc=0 sub-blocks must decode to exactly zero");
        }
    }

    /// All-zero input hits [`super::make_q3_quants_16`]'s `amax <
    /// GROUP_MAX_EPS` fast path: every level and scale is zero. The round
    /// trip must be bit-exact, not merely close.
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
    /// element equal to a nonzero constant -- same control
    /// [`super::q4_k`]/[`super::q5_k`]/[`super::q6_k`] all run,
    /// exercising the real iterative-RMSE search and the `d` `f16`
    /// rounding. `Q3_K` has no per-sub-block min, so a constant vector's
    /// round trip is exact only up to the 3-bit level granularity and the
    /// `f16` scale rounding -- looser than `Q4_K`/`Q5_K`'s (which also
    /// fit a min), tight enough that a working codec clears it easily.
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
        debug!(max_error, "quant.q3_k constant-nonzero-vector round trip");
        assert!(
            max_error < 0.05,
            "constant-vector round trip should be near-exact, measured max_error={max_error}"
        );
    }

    /// Round-trips a smooth, multi-block, non-degenerate signal and
    /// reports (does not hide) the measured max and RMS error. `Q3_K`'s
    /// ~3.44 bits/weight sits below `Q4_K`'s 4.5, so its error bounds are
    /// deliberately looser than `q4_k`'s (`0.6`/`0.2`) -- 3-bit levels
    /// with no per-sub-block min carry visibly more quantization noise;
    /// bounds chosen from the format's coarser resolution, not tuned to
    /// the measured numbers.
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
            assert!(
                got.is_finite(),
                "dequantized value must be finite, got {got}"
            );
            let diff = (got - want).abs();
            max_error = max_error.max(diff);
            sum_sq_error += f64::from(diff) * f64::from(diff);
        }
        let rms_error = (sum_sq_error / elements as f64).sqrt();
        debug!(max_error, rms_error, "quant.q3_k smooth-signal round trip");
        assert!(
            max_error < 1.2,
            "max_error={max_error} exceeds loose sanity bound"
        );
        assert!(
            rms_error < 0.35,
            "rms_error={rms_error} exceeds loose sanity bound"
        );
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

    #[test]
    fn quantize_rejects_non_element_multiple_length() {
        let input = vec![0.0f32; QK_K - 1];
        let mut output = vec![0u8; BLOCK_BYTES];
        let error = quantize(&input, &mut output).unwrap_err();
        assert_eq!(
            error,
            QuantError::InputNotElementMultiple {
                codec: CODEC,
                unit: "super-block",
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
                codec: CODEC,
                found: partial_bytes,
                block_bytes: BLOCK_BYTES,
            }
        );
    }

    // -- P14 incumbent-parity: llama.cpp's own `gguf-py` `dequantize_row_q3_K`
    // (via `gguf.quants.dequantize`, numpy) is the oracle. `#[ignore]`d like
    // the other real-checkpoint tests in this crate (`edge.rs`,
    // `restack.rs`): host-local files outside this repo, opportunistic, never
    // part of the standard gate. Only the header region and the compared
    // tensor's own byte range are ever read off disk -- never the whole
    // multi-gigabyte checkpoint.
    #[cfg(feature = "std")]
    mod q3_k_real {
        use std::io::{Read, Seek, SeekFrom};
        use std::path::Path;

        use proxima_telemetry::debug;

        use super::{BLOCK_BYTES, QK_K, dequantize};
        use crate::pipe::parse_complete;
        use crate::types::GgmlType;

        /// Reads exactly the header region (KV block + tensor directory),
        /// growing the read window until `parse_complete` stops reporting
        /// truncation -- never the multi-gigabyte tensor payload behind it.
        fn parse_header(file: &mut std::fs::File) -> crate::pipe::ParsedGguf {
            let mut header_buf = Vec::new();
            for cap in [4usize << 20, 16 << 20, 64 << 20, 128 << 20] {
                header_buf.resize(cap, 0);
                file.seek(SeekFrom::Start(0)).expect("seek to file start");
                let read = file.read(&mut header_buf).expect("read gguf header region");
                header_buf.truncate(read);
                if let Ok(parsed) = parse_complete(&header_buf) {
                    return parsed;
                }
            }
            panic!("gguf metadata region did not fit in 128 MiB");
        }

        /// Reads bytes `range` directly off `file` via `seek`+`read` -- the
        /// only tensor-payload bytes this test ever touches.
        fn read_range(file: &mut std::fs::File, range: core::ops::Range<u64>) -> Vec<u8> {
            let mut buffer = vec![0u8; (range.end - range.start) as usize];
            file.seek(SeekFrom::Start(range.start))
                .expect("seek to tensor data range start");
            file.read_exact(&mut buffer)
                .expect("read exact tensor data range");
            buffer
        }

        /// Reinterprets a little-endian `f32` byte dump (as `numpy`'s
        /// `ndarray.tofile` writes on this host's native little-endian
        /// architecture) into owned `f32`s.
        fn read_f32_dump(path: &Path) -> Vec<f32> {
            let bytes = std::fs::read(path).expect("read oracle f32 dump");
            assert!(
                bytes.len().is_multiple_of(4),
                "oracle dump {path:?} is not a whole number of f32s: {} bytes",
                bytes.len()
            );
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect()
        }

        /// Prints the `GgmlType` census (tensor count per type) for the whole
        /// file -- the header directory alone carries this, no payload read
        /// needed.
        fn print_codec_census(parsed: &crate::pipe::ParsedGguf) {
            let mut counts: std::collections::BTreeMap<alloc::string::String, usize> =
                std::collections::BTreeMap::new();
            for tensor in &parsed.tensors {
                *counts
                    .entry(alloc::format!("{:?}", tensor.ggml_type))
                    .or_insert(0) += 1;
            }
            println!("-- codec census ({} tensors) --", parsed.tensors.len());
            for (codec, count) in &counts {
                println!("  {codec}: {count}");
            }
        }

        /// Compares this crate's `q3_k::dequantize` against llama.cpp's own
        /// `gguf-py` dequantization (`gguf.quants.dequantize`, which calls
        /// numpy's port of `dequantize_row_q3_K`) on the first 8 rows of
        /// `blk.0.ffn_up.weight` from a real requantized `Q3_K_M` checkpoint.
        ///
        /// Oracle dump produced out-of-band (see
        /// `scratchpad/q3k-logs/oracle_dump.py`, never committed) and pointed
        /// to via `PROXIMA_Q3K_ORACLE`; the checkpoint itself via
        /// `PROXIMA_Q3K_GGUF`. `f16` `d` decodes identically in both
        /// implementations, so per-row max-abs-diff is held to
        /// `1e-6 * row_max_abs` -- floating-point noise floor, not a fitted
        /// tolerance.
        #[test]
        #[ignore = "depends on a host-local real gguf checkpoint and an out-of-band oracle dump"]
        fn q3_k_real_dequantize_matches_llama_cpp_gguf_py_oracle() {
            let Ok(gguf_path) = std::env::var("PROXIMA_Q3K_GGUF") else {
                eprintln!("skipping: PROXIMA_Q3K_GGUF not set");
                return;
            };
            let Ok(oracle_path) = std::env::var("PROXIMA_Q3K_ORACLE") else {
                eprintln!("skipping: PROXIMA_Q3K_ORACLE not set");
                return;
            };
            let gguf_path = Path::new(&gguf_path);
            let oracle_path = Path::new(&oracle_path);
            if !gguf_path.exists() || !oracle_path.exists() {
                eprintln!(
                    "skipping: gguf ({gguf_path:?}) or oracle ({oracle_path:?}) missing on this host"
                );
                return;
            }

            let mut file = std::fs::File::open(gguf_path).expect("open real gguf checkpoint");
            let file_len = file.metadata().expect("stat real gguf checkpoint").len();
            let parsed = parse_header(&mut file);
            print_codec_census(&parsed);

            let tensor_name = "blk.0.ffn_up.weight";
            let tensor = parsed
                .tensors
                .iter()
                .find(|candidate| candidate.name == tensor_name)
                .unwrap_or_else(|| panic!("{tensor_name} not present in real checkpoint"));
            assert_eq!(
                tensor.ggml_type,
                GgmlType::Q3_K,
                "{tensor_name} must be Q3_K in this Q3_K_M checkpoint"
            );

            let row_elements = tensor.dims[0] as usize;
            let rows_to_compare = 8usize;
            let elements_to_compare = row_elements * rows_to_compare;
            assert!(
                elements_to_compare.is_multiple_of(QK_K),
                "row width must divide the super-block size for this slice to align on block boundaries"
            );
            let blocks_to_compare = elements_to_compare / QK_K;
            let bytes_to_compare = (blocks_to_compare * BLOCK_BYTES) as u64;

            let full_range = parsed
                .tensor_data_range(tensor, file_len)
                .expect("tensor data range within real checkpoint");
            let compare_range = full_range.start..full_range.start + bytes_to_compare;
            let packed = read_range(&mut file, compare_range);

            let mut decoded = vec![0.0f32; elements_to_compare];
            dequantize(&packed, &mut decoded).expect("decode real Q3_K super-blocks");

            let oracle = read_f32_dump(oracle_path);
            assert_eq!(
                oracle.len(),
                elements_to_compare,
                "oracle dump element count must match the compared slice"
            );

            let mut max_abs_diff = 0.0f32;
            let mut max_row_abs = 0.0f32;
            for row in 0..rows_to_compare {
                let row_range = row * row_elements..(row + 1) * row_elements;
                let row_max_abs = oracle[row_range.clone()]
                    .iter()
                    .fold(0.0f32, |accumulator, &value| accumulator.max(value.abs()));
                let row_diff = decoded[row_range.clone()]
                    .iter()
                    .zip(&oracle[row_range])
                    .map(|(got, want)| (got - want).abs())
                    .fold(0.0f32, f32::max);
                max_row_abs = max_row_abs.max(row_max_abs);
                max_abs_diff = max_abs_diff.max(row_diff);
                let tolerance = 1e-6 * row_max_abs;
                assert!(
                    row_diff <= tolerance,
                    "row {row}: max_abs_diff={row_diff} exceeds tolerance={tolerance} (row_max_abs={row_max_abs})"
                );
            }

            debug!(
                tensor = tensor_name,
                dims = ?tensor.dims,
                rows_compared = rows_to_compare,
                max_abs_diff,
                max_row_abs,
                "q3_k real-checkpoint parity against llama.cpp gguf-py oracle"
            );
            println!(
                "tensor={tensor_name} dims={:?} rows_compared={rows_to_compare} max_abs_diff={max_abs_diff} max_row_abs={max_row_abs}",
                tensor.dims
            );
        }
    }
}
