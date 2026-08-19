//! `Q6_K`: 6-bit weights in 256-element super-blocks, split into 16
//! sub-blocks of 16 with its own 8-bit signed scale. `x = d*sc*q` per
//! sub-block, `q` a 6-bit value biased by 32 into `[-32, 31]`
//! (`ggml-quants.c:1684-1713`, cited on [`dequantize_block`]).
//!
//! Layout, from llama.cpp (`/Users/brianbruggeman/repos/others/llama.cpp`):
//! `ggml/src/ggml-common.h:320-326` — `block_q6_K` is 210 bytes per 256
//! elements (`uint8_t ql[128]` (`QK_K/2`, low 4 bits), `uint8_t qh[64]`
//! (`QK_K/4`, high 2 bits), `int8_t scales[16]` (`QK_K/16`), `ggml_half
//! d`) -- `d` trails the block, unlike [`super::q4_k`]/[`super::q8_0`]
//! where it leads. That 210-byte figure is cross-checked here at compile
//! time against [`crate::types::GgmlType::Q6_K`]'s already-landed
//! [`crate::types::GgmlType::block_layout`] rather than re-typed by hand.

use crate::quant::QuantError;
use crate::types::GgmlType;

/// This codec's name as it appears in a rendered [`QuantError`] message.
const CODEC: &str = "q6_k";

/// Elements per super-block (`ggml-common.h:89`, `#define QK_K 256`).
pub const QK_K: usize = 256;

/// Elements per sub-block: `QK_K` is split into 16 sub-blocks of 16
/// (`ggml-common.h:318`, "16 blocks of 16 elements each").
pub const SUB_BLOCK_ELEMENTS: usize = 16;

/// Sub-blocks per super-block.
pub const SUB_BLOCKS: usize = QK_K / SUB_BLOCK_ELEMENTS;

const QL_BYTES: usize = QK_K / 2;
const QH_BYTES: usize = QK_K / 4;
const SCALES_BYTES: usize = SUB_BLOCKS;

/// Bytes per super-block. Derived from the already-landed
/// [`crate::types::GgmlType::block_layout`] (`types.rs`), not re-typed —
/// that table was itself checked against `ggml-common.h`'s
/// `static_assert(sizeof(block_q6_K) == sizeof(ggml_half) + QK_K/16 +
/// 3*QK_K/4, ...)` at `ggml-common.h:326`.
pub const BLOCK_BYTES: usize = {
    let layout = GgmlType::Q6_K.block_layout();
    assert!(layout.block_elements as usize == QK_K, "GgmlType::Q6_K block_elements drifted from QK_K");
    layout.block_bytes as usize
};

const QL_OFFSET: usize = 0;
const QH_OFFSET: usize = QL_OFFSET + QL_BYTES;
const SCALES_OFFSET: usize = QH_OFFSET + QH_BYTES;
const D_OFFSET: usize = SCALES_OFFSET + SCALES_BYTES;

/// Below this absolute max sub-block scale, `make_qx_quants` treats the
/// super-block as all-zero (`ggml-quants.c:16`, `#define GROUP_MAX_EPS
/// 1e-15f`).
const GROUP_MAX_EPS: f32 = 1e-15;

/// Number of whole `Q6_K` super-blocks a byte run decodes to, or `None`
/// if `byte_len` is not an exact multiple of [`BLOCK_BYTES`].
#[must_use]
pub const fn blocks_for_bytes(byte_len: usize) -> Option<usize> {
    if byte_len.is_multiple_of(BLOCK_BYTES) {
        Some(byte_len / BLOCK_BYTES)
    } else {
        None
    }
}

/// Exact packed byte length for `block_count` `Q6_K` super-blocks.
#[must_use]
pub const fn bytes_for_blocks(block_count: usize) -> usize {
    block_count * BLOCK_BYTES
}

/// Exact `f32` element count for `block_count` `Q6_K` super-blocks.
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

/// Unpacks one 128-element half's four packed values at output position
/// `l` (`0..32`) back into their 6-bit levels (`0..63`, no sign bias
/// applied — dequantize subtracts 32, quantize already added it).
/// Mirrors the bit layout `dequantize_row_q6_K`/`quantize_row_q6_K_ref`
/// share (`ggml-quants.c:1696-1705`, `ggml-quants.c:1668-1676`): `ql`'s
/// low/high nibbles hold each value's low 4 bits, `qh`'s 2-bit lanes hold
/// the high 2 bits.
fn unpack_levels(ql_half: &[u8], qh_half: &[u8], l: usize) -> [u8; 4] {
    let high = qh_half[l];
    [
        (ql_half[l] & 0x0F) | ((high & 0x03) << 4),
        (ql_half[l + SUB_BLOCK_ELEMENTS * 2] & 0x0F) | (((high >> 2) & 0x03) << 4),
        (ql_half[l] >> 4) | (((high >> 4) & 0x03) << 4),
        (ql_half[l + SUB_BLOCK_ELEMENTS * 2] >> 4) | (((high >> 6) & 0x03) << 4),
    ]
}

/// Dequantizes one 256-element `Q6_K` super-block. `block` must be
/// exactly [`BLOCK_BYTES`] bytes and `output` exactly [`QK_K`] elements —
/// callers go through [`dequantize`], which validates both.
///
/// Ports `dequantize_row_q6_K` (`ggml-quants.c:1684-1713`) exactly: two
/// 128-element halves, each split into four 32-wide lanes sharing one
/// `qh` byte per lane position, each lane using its own signed 8-bit
/// sub-block scale.
pub fn dequantize_block(block: &[u8], output: &mut [f32]) {
    let d = f16_at(block, D_OFFSET).to_f32();
    let ql = &block[QL_OFFSET..QL_OFFSET + QL_BYTES];
    let qh = &block[QH_OFFSET..QH_OFFSET + QH_BYTES];
    let scales = &block[SCALES_OFFSET..SCALES_OFFSET + SCALES_BYTES];

    for half in 0..2 {
        let ql_half = &ql[half * 64..half * 64 + 64];
        let qh_half = &qh[half * 32..half * 32 + 32];
        let scale_half = &scales[half * 8..half * 8 + 8];
        let out_half = &mut output[half * 128..half * 128 + 128];
        for l in 0..32 {
            let sub_block = l / SUB_BLOCK_ELEMENTS;
            let levels = unpack_levels(ql_half, qh_half, l);
            for (lane, &level) in levels.iter().enumerate() {
                let scale = f32::from(scale_half[sub_block + lane * 2] as i8);
                let quant = f32::from(level) - 32.0;
                out_half[l + lane * 32] = d * scale * quant;
            }
        }
    }
}

fn f16_at(block: &[u8], offset: usize) -> half::f16 {
    let mut bytes = [0u8; 2];
    bytes.copy_from_slice(&block[offset..offset + 2]);
    half::f16::from_le_bytes(bytes)
}

/// Dequantizes a run of `Q6_K` super-blocks. `data` is borrowed, `output`
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
    for (block, out_chunk) in data.chunks_exact(BLOCK_BYTES).zip(output.chunks_exact_mut(QK_K)) {
        dequantize_block(block, out_chunk);
    }
    Ok(())
}

/// One sub-block's scale search: candidate inverse scales around
/// `-nmax/max`, each scored by weighted `sum(w*x*l)^2 / sum(w*l^2)` with
/// `w = x[i]^2`, keeping the best.
///
/// Ports `make_qx_quants` (`ggml-quants.c:373-440`) with `n` fixed at
/// [`SUB_BLOCK_ELEMENTS`] (16, the only size `Q6_K` calls it with),
/// `nmax` fixed at 32, `rmse_type` fixed at `1` (`w = x[i]*x[i]`,
/// `quantize_row_q6_K_ref`'s call at `ggml-quants.c:1628`), and `qw`
/// fixed at "absent" (`Q6_K`'s call passes `NULL`). Returns the levels
/// (`0..63`, sign-biased by `+32`) alongside the scale — both discarded
/// by the caller when the corresponding super-block scale code rounds to
/// zero, matching C's `L` staying at this first pass's values in that
/// case (`ggml-quants.c:1654-1656`, the `if (!d) continue;` skip).
fn make_qx_quants_16(x: &[f32; SUB_BLOCK_ELEMENTS]) -> ([u8; SUB_BLOCK_ELEMENTS], f32) {
    const NMAX: i32 = 32;
    const NMAX_F: f32 = NMAX as f32;

    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &value in x {
        let ax = value.abs();
        if ax > amax {
            amax = ax;
            max = value;
        }
    }
    if amax < GROUP_MAX_EPS {
        return ([0u8; SUB_BLOCK_ELEMENTS], 0.0);
    }

    let iscale = -NMAX_F / max;
    let mut levels = [0u8; SUB_BLOCK_ELEMENTS];
    let mut sum_lx = 0.0f32;
    let mut sum_l2 = 0.0f32;
    for (index, &value) in x.iter().enumerate() {
        let level = nearest_int(iscale * value).clamp(-NMAX, NMAX - 1);
        levels[index] = (level + NMAX) as u8;
        let weight = value * value;
        sum_lx += weight * value * level as f32;
        sum_l2 += weight * (level * level) as f32;
    }
    let mut scale = if sum_l2 > 0.0 { sum_lx / sum_l2 } else { 0.0 };
    let mut best = scale * sum_lx;

    for step in -9..=9i32 {
        if step == 0 {
            continue;
        }
        let candidate_iscale = -(NMAX_F + 0.1 * step as f32) / max;
        let mut candidate_sum_lx = 0.0f32;
        let mut candidate_sum_l2 = 0.0f32;
        for &value in x {
            let level = nearest_int(candidate_iscale * value).clamp(-NMAX, NMAX - 1);
            let weight = value * value;
            candidate_sum_lx += weight * value * level as f32;
            candidate_sum_l2 += weight * (level * level) as f32;
        }
        if candidate_sum_l2 > 0.0 && candidate_sum_lx * candidate_sum_lx > best * candidate_sum_l2 {
            for (index, &value) in x.iter().enumerate() {
                let level = nearest_int(candidate_iscale * value).clamp(-NMAX, NMAX - 1);
                levels[index] = (level + NMAX) as u8;
            }
            scale = candidate_sum_lx / candidate_sum_l2;
            best = scale * candidate_sum_lx;
        }
    }
    (levels, scale)
}

/// Quantizes one 256-element chunk into a `Q6_K` super-block.
///
/// Ports `quantize_row_q6_K_ref` (`ggml-quants.c:1614-1682`): per
/// sub-block, a [`make_qx_quants_16`] search for that sub-block's own
/// scale; the super-block's single `f16` scale is fit from whichever
/// sub-block had the largest-magnitude scale (sign included, not just
/// magnitude); each sub-block's 8-bit signed scale code is then
/// `round(iscale * sub_scale)` clamped only on the high end (`min(127,
/// ...)` -- the reference never clamps the low end, matching production
/// bit-for-bit); levels are finally recomputed once against each
/// sub-block's own re-derived `d = block_d * sub_code`, with sub-blocks
/// whose code rounds to zero left at whatever [`make_qx_quants_16`]'s
/// first pass computed (`ggml-quants.c:1654-1656`'s `if (!d) continue;`).
fn quantize_block(x: &[f32], output: &mut [u8]) {
    let mut levels = [0u8; QK_K];
    let mut scales = [0.0f32; SUB_BLOCKS];
    let mut max_scale = 0.0f32;
    let mut max_abs_scale = 0.0f32;

    for sub_block in 0..SUB_BLOCKS {
        let mut chunk = [0.0f32; SUB_BLOCK_ELEMENTS];
        chunk.copy_from_slice(&x[sub_block * SUB_BLOCK_ELEMENTS..(sub_block + 1) * SUB_BLOCK_ELEMENTS]);
        let (sub_levels, scale) = make_qx_quants_16(&chunk);
        levels[sub_block * SUB_BLOCK_ELEMENTS..(sub_block + 1) * SUB_BLOCK_ELEMENTS].copy_from_slice(&sub_levels);
        scales[sub_block] = scale;
        let abs_scale = scale.abs();
        if abs_scale > max_abs_scale {
            max_abs_scale = abs_scale;
            max_scale = scale;
        }
    }

    if max_abs_scale < GROUP_MAX_EPS {
        output.fill(0);
        return;
    }

    let iscale = -128.0 / max_scale;
    let block_scale = half::f16::from_f32(1.0 / iscale);

    let mut scale_codes = [0i8; SUB_BLOCKS];
    for (code, &scale) in scale_codes.iter_mut().zip(scales.iter()) {
        *code = nearest_int(iscale * scale).min(127) as i8;
    }

    for (sub_block, &code) in scale_codes.iter().enumerate() {
        let sub_d = block_scale.to_f32() * f32::from(code);
        if sub_d == 0.0 {
            continue;
        }
        for offset in 0..SUB_BLOCK_ELEMENTS {
            let index = sub_block * SUB_BLOCK_ELEMENTS + offset;
            let level = nearest_int(x[index] / sub_d).clamp(-32, 31);
            levels[index] = (level + 32) as u8;
        }
    }

    output[D_OFFSET..D_OFFSET + 2].copy_from_slice(&block_scale.to_le_bytes());
    for (byte, &code) in output[SCALES_OFFSET..SCALES_OFFSET + SCALES_BYTES].iter_mut().zip(scale_codes.iter()) {
        *byte = code as u8;
    }

    let (ql, rest) = output[..QH_OFFSET + QH_BYTES].split_at_mut(QH_OFFSET);
    for half in 0..2 {
        let base = half * 128;
        let ql_half = &mut ql[half * 64..half * 64 + 64];
        let qh_half = &mut rest[half * 32..half * 32 + 32];
        for l in 0..32 {
            let l1 = levels[base + l];
            let l2 = levels[base + l + 32];
            let l3 = levels[base + l + 64];
            let l4 = levels[base + l + 96];
            ql_half[l] = (l1 & 0x0F) | (l3 << 4);
            ql_half[l + 32] = (l2 & 0x0F) | (l4 << 4);
            qh_half[l] = (l1 >> 4) | ((l2 >> 4) << 2) | ((l3 >> 4) << 4) | ((l4 >> 4) << 6);
        }
    }
}

/// Quantizes a run of `f32` weights into `Q6_K` super-blocks. `input` is
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

    use proxima_telemetry::debug;

    use super::{BLOCK_BYTES, CODEC, QH_BYTES, QK_K, QL_BYTES, QuantError, SCALES_BYTES, dequantize, quantize};

    /// One super-block, hand-packed and hand-decoded, checked against the
    /// `x = d*sc*q` formula computed by hand -- not by calling
    /// [`super::quantize`] to build the fixture. `d=1.0` (exact in
    /// `f16`), every scale an integer, every probe level chosen so
    /// `q = level - 32` is an exact integer: every expected value below
    /// is an exact `f32`, so `assert_eq!` needs no epsilon.
    ///
    /// One probe element per (half, lane) pair -- 8 probes total, one in
    /// each of the two 128-element halves' four 32-wide lanes -- each
    /// hand-packed into `ql`'s nibble and `qh`'s 2-bit lane following
    /// `ggml-quants.c:1696-1705`'s layout (mirrored in
    /// [`super::unpack_levels`], not used to build this fixture).
    #[test]
    // some probe values are zero, making a couple of the four-term qh/ql
    // formulas below identity operations -- kept symbolic (matching the
    // dequant formula term-for-term) rather than hand-simplified, so the
    // fixture stays visibly a direct application of the packing formula.
    #[allow(clippy::identity_op)]
    fn dequantize_block_matches_hand_packed_fixture() {
        let mut ql = [0u8; QL_BYTES];
        let mut qh = [0u8; QH_BYTES];
        let mut scales = [0i8; SCALES_BYTES];

        // half 0, l=3: lane 0 (elem 3) level=40 (q=8), lane 1 (elem 35)
        // level=20 (q=-12), lane 2 (elem 67) level=50 (q=18), lane 3
        // (elem 99) level=10 (q=-22). sub_block = l/16 = 0, so scales
        // used are scale_half[0+0*2]=scales[0], [0+1*2]=scales[2],
        // [0+2*2]=scales[4], [0+3*2]=scales[6].
        ql[3] = (40 & 0x0F) | (50 << 4); // low=l1&0xF=8, high=l3&0xF=2 -> 0x28
        ql[3 + 32] = (20 & 0x0F) | (10 << 4); // low=l2&0xF=4, high=l4&0xF=10 -> 0xA4
        qh[3] = (40u8 >> 4) | ((20u8 >> 4) << 2) | ((50u8 >> 4) << 4) | ((10u8 >> 4) << 6); // 2|(1<<2)|(3<<4)|(0<<6) = 0x36
        scales[0] = 3;
        scales[2] = -4;
        scales[4] = 5;
        scales[6] = -2;

        // half 1, l=20: sub_block = 20/16 = 1 (local to this half). The
        // half-1 `scale_half` slice is `scales[8..16]` (the encoder's `sc`
        // pointer advances by 8 after each 128-element half,
        // `ggml-quants.c:1710`), so the global scale indices are
        // 8+1=9, 8+3=11, 8+5=13, 8+7=15. Output positions are 128+20,
        // 128+52, 128+84, 128+116.
        let half1_ql = 64 + 20;
        ql[half1_ql] = (15 & 0x0F) | (0 << 4); // l1=15, l3=0
        ql[half1_ql + 32] = (63 & 0x0F) | (32 << 4); // l2=63, l4=32
        qh[32 + 20] = (15u8 >> 4) | ((63u8 >> 4) << 2) | ((0u8 >> 4) << 4) | ((32u8 >> 4) << 6); // 0|(3<<2)|0|(2<<6)=0x8C
        scales[9] = 7;
        scales[11] = -1;
        scales[13] = 6;
        scales[15] = -8;

        let mut block = [0u8; BLOCK_BYTES];
        block[0..QL_BYTES].copy_from_slice(&ql);
        block[QL_BYTES..QL_BYTES + QH_BYTES].copy_from_slice(&qh);
        for (byte, &code) in block[QL_BYTES + QH_BYTES..QL_BYTES + QH_BYTES + SCALES_BYTES]
            .iter_mut()
            .zip(scales.iter())
        {
            *byte = code as u8;
        }
        block[QL_BYTES + QH_BYTES + SCALES_BYTES..].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());

        // A sub-block's scale multiplies every one of the 16 `l` positions
        // it covers, not just the hand-picked probe -- the 15 untouched
        // `ql`/`qh` bytes in each probed sub-block still decode to
        // `level=0`, i.e. `q=-32`, so their expected value is
        // `scale * -32.0`, not `0.0`. Fill each probed lane's 16-wide
        // `is`-region with that baseline first (mirroring
        // `super::q4_k`'s hand-computed test, which fills a whole
        // sub-block's baseline the same way), then overwrite the single
        // probe position with its real, level-derived value.
        let mut expected = [0.0f32; QK_K];
        expected[0..16].fill(3.0 * -32.0); // half 0, is=0, lane 0 (sc=3)
        expected[32..48].fill(-4.0 * -32.0); // half 0, is=0, lane 1 (sc=-4)
        expected[64..80].fill(5.0 * -32.0); // half 0, is=0, lane 2 (sc=5)
        expected[96..112].fill(-2.0 * -32.0); // half 0, is=0, lane 3 (sc=-2)
        expected[144..160].fill(7.0 * -32.0); // half 1, is=1, lane 0 (sc=7)
        expected[176..192].fill(32.0); // half 1, is=1, lane 1 (sc=-1, baseline = -1 * -32)
        expected[208..224].fill(6.0 * -32.0); // half 1, is=1, lane 2 (sc=6)
        expected[240..256].fill(-8.0 * -32.0); // half 1, is=1, lane 3 (sc=-8)

        expected[3] = 3.0 * (40.0 - 32.0); // sc=3, q=8 -> 24.0
        expected[3 + 32] = -4.0 * (20.0 - 32.0); // sc=-4, q=-12 -> 48.0
        expected[3 + 64] = 5.0 * (50.0 - 32.0); // sc=5, q=18 -> 90.0
        expected[3 + 96] = -2.0 * (10.0 - 32.0); // sc=-2, q=-22 -> 44.0
        expected[128 + 20] = 7.0 * (15.0 - 32.0); // sc=7, q=-17 -> -119.0
        expected[128 + 52] = -(63.0 - 32.0); // sc=-1, q=31 -> -31.0
        expected[128 + 84] = 6.0 * (0.0 - 32.0); // sc=6, q=-32 -> -192.0 (matches the sub-block baseline)
        expected[128 + 116] = -8.0 * (32.0 - 32.0); // sc=-8, q=0 -> -0.0

        let mut output = [0.0f32; QK_K];
        dequantize(&block, &mut output).expect("well-formed single block");
        assert_eq!(output.as_slice(), expected.as_slice());
    }

    /// All-zero input hits `make_qx_quants`'s `amax < GROUP_MAX_EPS` fast
    /// path for every sub-block, so `max_abs_scale` stays `0.0` and the
    /// whole super-block is memset to zero bytes. Round trip must be
    /// bit-exact.
    #[test]
    fn quantize_dequantize_zero_vector_is_bit_exact() {
        let input = vec![0.0f32; QK_K];
        let mut packed = vec![0u8; BLOCK_BYTES];
        quantize(&input, &mut packed).expect("one block");
        let mut output = vec![0.0f32; QK_K];
        dequantize(&packed, &mut output).expect("one block");
        assert_eq!(output, input);
    }

    /// Round-trips a smooth, multi-block, non-degenerate signal and
    /// reports (does not hide) the measured max and RMS error. 6.5625
    /// bits/weight sits between q4_K's 4.5 and q8_0's 8, so the loose
    /// sanity bounds (`0.15` max, `0.05` RMS) are chosen between those
    /// two codecs' own bounds, not tuned to the measured numbers.
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
        debug!(max_error, rms_error, "quant.q6_k smooth-signal round trip");
        assert!(max_error < 0.15, "max_error={max_error} exceeds loose sanity bound");
        assert!(rms_error < 0.05, "rms_error={rms_error} exceeds loose sanity bound");
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

    /// A truncated block run that is neither a whole block nor empty —
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
