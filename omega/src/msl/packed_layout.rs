//! the byte-layout table and the per-layout fold the generic packed matvec will
//! monomorphize over (S0/S1 of `specs/algebra-monomorphization.md`).
//!
//! two corrections from the plan review are load-bearing here:
//!
//! 1. the fold seam is per SUB-BLOCK, not per element. every bespoke body in
//!    `packed_row_blocked_ggml.rs` amortises deliberately (`q4k_run8` pulls eight
//!    levels per decode, Q4_0's masked-word dot extracts four nibbles with no
//!    runtime shift), so a `fn decode(blk, slot) -> f32` called per element would
//!    throw that away and produce a generic body slower than the ones it
//!    replaces. `fold_levels` decodes a whole block-slot at once and hands the
//!    emitter `LEVELS` floats.
//! 2. the table is TOTAL over [`Codec`] and exhaustive, with `Unsupported` as an
//!    explicit per-codec answer rather than a wildcard. adding a codec therefore
//!    fails to compile here, and the pinned count in the tests fails too.

#![expect(
    dead_code,
    reason = "S0/S1 scaffolding: the table and the Q4_0 fold are verified against \
              proxima-gguf's own dequantizer here, and are wired into the emitter in S2 \
              (specs/algebra-monomorphization.md)"
)]

use proxima_numeric::codec::Codec;

/// the byte layout a packed-row matvec body is specialized for. one variant per
/// layout the emitter can address -- distinct codecs share a variant only when
/// their bytes decode identically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Layout {
    /// 18 B: binary16 scale at +0, 32 nibbles at +2, level - 8.
    Q4_0,
    /// 20 B: binary16 scale, binary16 min, 32 nibbles.
    Q4_1,
    /// 22 B: binary16 scale, 4+1-bit high bits, 32 nibbles.
    Q5_0,
    /// 24 B: scale, min, qh word, 32 nibbles.
    Q5_1,
    /// 34 B: binary16 scale, 32 signed bytes.
    Q8_0,
    /// 210 B super-block: d, dmin, 16 x 4-bit scales, 256 nibbles.
    Q2K,
    /// 110 B super-block: 16 x 6-bit scales mixed with 256 x 2-bit levels.
    Q3K,
    /// 144 B super-block: d, dmin, 12 x 6-bit scales, 256 nibbles.
    Q4K,
    /// 176 B super-block: as Q4K plus a 32-byte qh high-bit plane.
    Q5K,
    /// 210 B super-block: ql, qh and 16 x 8-bit scales.
    Q6K,
    /// 18 B iq4_nl: binary16 scale plus 16 bytes of two 4-bit levels per byte,
    /// non-linear lookup table.
    Iq4Nl,
    /// unpacked 2-byte elements.
    Float16,
    BFloat16,
    /// no layout this emitter can address: rejected at admission, never emitted.
    Unsupported,
}

/// the one place a [`Codec`] becomes a [`Layout`]. exhaustive by construction --
/// there is no wildcard arm, so a new codec is a compile error right here.
pub(crate) const fn layout(codec: Codec) -> Layout {
    match codec {
        Codec::Q4_0 => Layout::Q4_0,
        Codec::Q4_1 => Layout::Q4_1,
        Codec::Q5_0 => Layout::Q5_0,
        Codec::Q5_1 => Layout::Q5_1,
        Codec::Q8_0 | Codec::Q8_1 => Layout::Q8_0,
        Codec::Q2K => Layout::Q2K,
        Codec::Q3K => Layout::Q3K,
        Codec::Q4K => Layout::Q4K,
        Codec::Q5K => Layout::Q5K,
        Codec::Q6K => Layout::Q6K,
        Codec::Iq4Nl => Layout::Iq4Nl,
        Codec::Float16 => Layout::Float16,
        Codec::BFloat16 => Layout::BFloat16,
        // the remaining wire codecs have no packed-row body and no generic
        // decode in this crate; each is named rather than swept into a wildcard
        // so this list is the inventory a future layout addition edits.
        Codec::Q8K
        | Codec::Iq1S
        | Codec::Iq1M
        | Codec::Iq2Xxs
        | Codec::Iq2Xs
        | Codec::Iq2S
        | Codec::Iq3Xxs
        | Codec::Iq3S
        | Codec::Iq4Xs
        | Codec::Tq10
        | Codec::Tq20
        | Codec::Mxfp4
        | Codec::Nvfp4
        | Codec::Q1_0
        | Codec::Q2_0
        | Codec::Pq2_0
        | Codec::Ptq1_0
        | Codec::BFloat8
        | Codec::Bf8E5M2
        | Codec::Bf4E2M1
        | Codec::Fp6E3M2 => Layout::Unsupported,
    }
}

/// one layout's block geometry plus the fold the shared emitter calls. the
/// associated consts are the whole point: they let the monomorphized body fold
/// offsets and bounds exactly as the hand-written bodies did.
pub(crate) trait LayoutFold {
    /// bytes one block occupies -- equal to the codec's own block stride.
    const BLOCK_BYTES: usize;
    /// elements one block covers.
    const BLOCK_ELEMENTS: usize;
    /// floats one [`Self::fold_levels`] call writes.
    const LEVELS: usize;

    /// decode one block's `LEVELS` values into `out[..LEVELS]`, starting at the
    /// block's own byte 0. `out` must hold at least `LEVELS` floats.
    fn fold_levels(block: &[u8], out: &mut [f32]);
}

/// Q4_0: 32 elements, a binary16 scale then 16 nibble bytes, value = scale *
/// (level - 8).
pub(crate) struct Q4_0Fold;

impl LayoutFold for Q4_0Fold {
    const BLOCK_BYTES: usize = 18;
    const BLOCK_ELEMENTS: usize = 32;
    const LEVELS: usize = 32;

    fn fold_levels(block: &[u8], out: &mut [f32]) {
        let scale = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for pair in 0..(Self::BLOCK_ELEMENTS / 2) {
            let byte = block[2 + pair];
            out[pair] = scale * (f32::from(byte & 0x0f) - 8.0);
            out[pair + Self::BLOCK_ELEMENTS / 2] = scale * (f32::from(byte >> 4) - 8.0);
        }
    }
}

/// IEEE binary16 -> f32, without pulling a crate into this seam. subnormals,
/// infinities and NaN are all preserved bit-for-bit by the standard widening.
fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = u32::from(bits & 0x03ff);
    let widened = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            // subnormal: renormalize into f32's exponent range
            let mut mantissa = mantissa;
            let mut shift = 0_u32;
            while mantissa & 0x0400 == 0 {
                mantissa <<= 1;
                shift += 1;
            }
            let mantissa = (mantissa & 0x03ff) << 13;
            sign | ((127 - 15 - shift + 1) << 23) | mantissa
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((u32::from(exponent) + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(widened)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use proxima_gguf::quant::q4_0;

    /// deterministic bytes with no `rand` dependency.
    fn bytes(seed: u32, len: usize) -> Vec<u8> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 16) as u8
            })
            .collect()
    }

    #[test]
    fn the_layout_table_is_total_and_pins_the_unsupported_count() {
        // every codec is NAMED by `layout` (the match has no wildcard, so a new
        // codec fails to compile). this test pins how many are Unsupported today,
        // so adding one silently is a count failure as well.
        let all = [
            Codec::Q4K,
            Codec::Q5K,
            Codec::Q6K,
            Codec::Q8_0,
            Codec::Q3K,
            Codec::Q4_0,
            Codec::Float16,
            Codec::BFloat16,
            Codec::Q2K,
            Codec::Q5_1,
            Codec::Q5_0,
            Codec::Q4_1,
            Codec::Q8_1,
            Codec::Q8K,
            Codec::Iq1S,
            Codec::Iq1M,
            Codec::Iq2Xxs,
            Codec::Iq2Xs,
            Codec::Iq2S,
            Codec::Iq3Xxs,
            Codec::Iq3S,
            Codec::Iq4Nl,
            Codec::Iq4Xs,
            Codec::Tq10,
            Codec::Tq20,
            Codec::Mxfp4,
            Codec::Nvfp4,
            Codec::Q1_0,
            Codec::Q2_0,
            Codec::Pq2_0,
            Codec::Ptq1_0,
            Codec::BFloat8,
            Codec::Bf8E5M2,
            Codec::Bf4E2M1,
            Codec::Fp6E3M2,
        ];
        assert_eq!(all.len(), 35, "every Codec variant must be listed once");
        let unsupported = all
            .iter()
            .filter(|&&codec| layout(codec) == Layout::Unsupported)
            .count();
        assert_eq!(unsupported, 21, "pinned: codecs with no layout body");
        let distinct = all.iter().map(|&c| layout(c)).collect::<BTreeSet<_>>();
        assert_eq!(distinct.len(), 14, "pinned: distinct layouts reachable today");
    }

    #[test]
    fn q4_0_fold_matches_the_gguf_dequantizer_block_for_block() {
        // the whole reason the fold seam exists: it must reproduce proxma-gguf's
        // own block decode exactly, since that decode is the CPU oracle every
        // later cell is checked against.
        for seed in 0..64_u32 {
            let block = bytes(seed.wrapping_mul(2_654_435_761), q4_0::BLOCK_BYTES);
            let mut folded = [0.0_f32; <Q4_0Fold as LayoutFold>::LEVELS];
            Q4_0Fold::fold_levels(&block, &mut folded);
            let mut expected = [0.0_f32; q4_0::QK4_0];
            q4_0::dequantize_block(&block, &mut expected);
            assert_eq!(
                folded.map(f32::to_bits),
                expected.map(f32::to_bits),
                "seed {seed}: fold must be bit-exact against the gguf dequantizer",
            );
        }
    }

    #[test]
    fn half_widening_preserves_subnormals_infinities_and_nan() {
        // the Q4_0 scale is the one place a NaN/Inf/subnormal reaches the body
        // through a hand-rolled widening, so pin the edges rather than trusting
        // the arithmetic.
        let cases = [
            (0x0000_u16, 0.0_f32),
            (0x8000, -0.0),
            (0x3c00, 1.0),
            (0xbc00, -1.0),
            (0x0001, 5.960_464_5e-8),
            (0x7c00, f32::INFINITY),
            (0xfc00, f32::NEG_INFINITY),
        ];
        for (bits, expected) in cases {
            assert_eq!(half_to_f32(bits).to_bits(), expected.to_bits(), "bits {bits:#06x}");
        }
        assert!(half_to_f32(0x7e00).is_nan(), "a NaN payload must stay NaN");
    }
}
