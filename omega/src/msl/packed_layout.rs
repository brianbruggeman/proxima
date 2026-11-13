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
//! 2. the table is TOTAL over [`Codec`]: a format is a description, so
//!    `layout` reads its geometry and answers `Unsupported` for any geometry no
//!    body addresses. adding a format changes the pinned counts in the tests.

#![expect(
    dead_code,
    reason = "S0/S1 scaffolding: the table and the Q4_0 fold are verified against \
              proxima-gguf's own dequantizer here, and are wired into the emitter in S2 \
              (specs/algebra-monomorphization.md)"
)]

use proxima_numeric::codec::Codec;
use proxima_numeric::{FloatLayout, SignEncoding};

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
    /// 84 B super-block: d, dmin, 16 x 4-bit scales, 256 x 2-bit levels.
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

/// the one place a numeric format becomes a [`Layout`]. a format is a description
/// (`Scalar` / `Block` / `Levels`, not an enum of names), so the layout is read from
/// the description's own geometry: the arm kind, elements per block and bytes per
/// block. a geometry no body can address is `Unsupported` -- the typed refusal
/// admission already keys on -- so every format the type can express is answered.
pub(crate) const fn layout(format: &Codec) -> Layout {
    let elements = format.elements_per_block();
    let bytes = format.bytes_per_block();
    match format {
        Codec::Scalar(scalar) => scalar_layout(scalar),
        Codec::Block(_) => block_layout(elements, bytes),
        // the one level-set geometry with a body: 32 elements in 18 bytes under a
        // non-linear table (a linear ladder of that geometry is a `Block`).
        Codec::Levels(_) if elements == 32 && bytes == 18 => Layout::Iq4Nl,
        Codec::Levels(_) => Layout::Unsupported,
    }
}

const fn scalar_layout(scalar: &FloatLayout) -> Layout {
    let ieee_shaped = matches!(scalar.sign, SignEncoding::Magnitude) && scalar.bias_override.is_none();
    match (ieee_shaped, scalar.exponent_bits, scalar.mantissa_bits) {
        (true, 5, 10) => Layout::Float16,
        (true, 8, 7) => Layout::BFloat16,
        _ => Layout::Unsupported,
    }
}

/// block geometry (elements, bytes) -> layout. `Q8_1` carries the `Q8_0` codes with
/// a stored sum behind the scale, so its 36-byte block shares the `Q8_0` body.
const fn block_layout(elements: u16, bytes: u16) -> Layout {
    match (elements, bytes) {
        (32, 18) => Layout::Q4_0,
        (32, 20) => Layout::Q4_1,
        (32, 22) => Layout::Q5_0,
        (32, 24) => Layout::Q5_1,
        (32, 34 | 36) => Layout::Q8_0,
        (256, 84) => Layout::Q2K,
        (256, 110) => Layout::Q3K,
        (256, 144) => Layout::Q4K,
        (256, 176) => Layout::Q5K,
        (256, 210) => Layout::Q6K,
        _ => Layout::Unsupported,
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
    fn the_layout_table_is_total_and_names_every_unsupported_format() {
        // every format the type enumerates is answered by `layout`. the unsupported set is
        // pinned by name so a regression names the format that lost (or gained) a body.
        // beyond the 21 hand-listed in the old table: Q8_1Grouped (the one entry added to
        // `Codec::ALL`, format.rs:457) and the two `Codec::UNTAGGED` formats Float32 and Bool.
        let all: Vec<Codec> = Codec::ALL
            .iter()
            .copied()
            .chain(Codec::UNTAGGED.iter().map(|(format, _)| *format))
            .collect();
        let unsupported: BTreeSet<&str> = all
            .iter()
            .filter(|format| layout(format) == Layout::Unsupported)
            .map(|format| format.debug_name().expect("every enumerated format has a debug name"))
            .collect();
        let expected: BTreeSet<&str> = [
            "Q8K", "Iq1S", "Iq1M", "Iq2Xxs", "Iq2Xs", "Iq2S", "Iq3Xxs", "Iq3S", "Iq4Xs", "Tq10",
            "Tq20", "Mxfp4", "Nvfp4", "Q1_0", "Q2_0", "Pq2_0", "Ptq1_0", "BFloat8", "Bf8E5M2",
            "Bf4E2M1", "Fp6E3M2", "Q8_1Grouped", "Float32", "Bool",
        ]
        .into_iter()
        .collect();
        assert_eq!(unsupported, expected, "formats with no layout body");
        let distinct = all.iter().map(layout).collect::<BTreeSet<_>>();
        assert_eq!((all.len(), distinct.len()), (38, 14), "formats, distinct layouts reachable today");
    }

    #[test]
    fn each_addressable_layout_is_the_one_its_own_geometry_describes() {
        let expected = [
            (Codec::Q4_0, Layout::Q4_0),
            (Codec::Q4_1, Layout::Q4_1),
            (Codec::Q5_0, Layout::Q5_0),
            (Codec::Q5_1, Layout::Q5_1),
            (Codec::Q8_0, Layout::Q8_0),
            (Codec::Q8_1, Layout::Q8_0),
            (Codec::Q2K, Layout::Q2K),
            (Codec::Q3K, Layout::Q3K),
            (Codec::Q4K, Layout::Q4K),
            (Codec::Q5K, Layout::Q5K),
            (Codec::Q6K, Layout::Q6K),
            (Codec::Iq4Nl, Layout::Iq4Nl),
            (Codec::Float16, Layout::Float16),
            (Codec::BFloat16, Layout::BFloat16),
            (Codec::Q8K, Layout::Unsupported),
            (Codec::Iq3S, Layout::Unsupported),
            (Codec::Bf8E5M2, Layout::Unsupported),
        ];
        for (format, layout_expected) in expected {
            assert_eq!(layout(&format), layout_expected, "{format:?}");
        }
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
