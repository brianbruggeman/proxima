//! The one format identity: every numeric storage format this workspace knows is a
//! [`NumericFormat`], and nothing outside this crate declares one.
//!
//! The three arms are disjoint by what a stored element *means*:
//!
//! - [`NumericFormat::Scalar`] — one number per value, decoded from its own bit fields by
//!   [`crate::float_layout_codec`] (f32, bf16, e4m3fn, int8, ...). This is `llvm::fltSemantics`.
//! - [`NumericFormat::Block`] — a run of codes under a shared scale, where a code's own
//!   [`FloatLayout`] *is* its value once scaled (Q4_0's nibble is `code - 8`; Fp6E3M2's six-bit
//!   field is an E3M2 float). The exact bit packing is the [`BlockLayout`] this arm carries, so
//!   one walker reads every format in the arm.
//! - [`NumericFormat::Levels`] — a run of codes whose value comes from an explicit
//!   [`LevelTable`] rather than from the code's own layout: ternary (`-1, 0, +1`), IQ4_NL's
//!   trained 16-level ladder, and the trained magnitude codebooks of the IQ family.
//!
//! The Block arm is [`BlockLayout`] rather than a separate `BlockFormat` struct: the sketch of a
//! block as `element + elements + bytes + scale + levels + high_bits` cannot express Q3_K's
//! two-plane mixed-radix codes or Q8_1's four interleaved scale runs, and a second struct
//! carrying the same facts would be a second source of truth for the same bytes. `BlockLayout`
//! *is* the block arm; its accessors ([`BlockLayout::levels`], [`BlockLayout::scale`],
//! [`BlockLayout::high_bits`]) are the semantic projections the sketch named.
//!
//! Teaching pointer: [`NumericFormat::tag`] is the stable wire identity, [`NumericFormat::name`]
//! the config spelling, and [`crate::decode`] the one place a format becomes numbers.

use crate::block_layout::{BlockLayout, Field};
use crate::float_layout::FloatLayout;

/// The level set a code selects, when it is not the code's own [`FloatLayout`] value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LevelTable {
    /// `value = zero_point + step * code`; the ternary ladder is `{-1, 0, +1}`.
    Linear {
        /// How many codes the ladder has; a code at or above it is not a level.
        count: u16,
        zero_point: i16,
        step: i16,
    },
    /// `value = table[code]`; IQ4_NL's 16-entry non-linear ladder.
    Explicit(&'static [i8]),
    /// IQ2_XS: a 9-bit grid index and a 7-bit sign index packed in one `u16` per eight elements.
    Iq2XsGrid,
    /// IQ3_XXS: two 8-bit grid indices per eight elements, one 32-bit aux word per 32.
    Iq3XxsGrid,
    /// A trained codebook whose table and packing land with its format frontend. [`NumericFormat`]
    /// describes the format; this crate refuses to decode it rather than guess the codebook.
    External,
}

impl LevelTable {
    /// The level a code stands for, for the arithmetic arms; `None` for a codebook.
    #[must_use]
    pub const fn level(&self, code: u32) -> Option<f32> {
        match self {
            Self::Linear {
                count,
                zero_point,
                step,
            } => {
                if code < *count as u32 {
                    Some((*zero_point as i32 + *step as i32 * code as i32) as f32)
                } else {
                    None
                }
            }
            Self::Explicit(values) => {
                if (code as usize) < values.len() {
                    Some(values[code as usize] as f32)
                } else {
                    None
                }
            }
            Self::Iq2XsGrid | Self::Iq3XxsGrid | Self::External => None,
        }
    }
}

/// Where a level-set format's codes live.
///
/// The split is a finding about the representation, not a convenience: `Packed` formats (ternary,
/// IQ4_NL) have an affine code address, so their exact packing is a [`BlockLayout`] and one walker
/// reads them. The trained codebooks (IQ2_XS, IQ3_XXS) address codes through a group record that
/// is not an affine map of the element index, so they carry only their block shape.
#[expect(
    clippy::large_enum_variant,
    reason = "the packed arm is a const block layout description, never heap-allocated and \
              always reached by reference"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LevelGeometry {
    /// An affine code map: the block layout is the exact packing.
    Packed(BlockLayout),
    /// A trained codebook: block shape only, because the code address is not affine.
    Codebook { elements: u16, bytes: u16, scale: Field },
}

/// A level-set format: block geometry plus the code -> level map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LevelFormat {
    /// Where the codes live.
    pub geometry: LevelGeometry,
    /// What a code means.
    pub levels: LevelTable,
}

impl LevelFormat {
    /// A level format projected from a [`BlockLayout`]: the geometry is the layout's, the level
    /// map decides what a code is worth. One source for both, so a packing fix cannot leave the
    /// level-set spelling behind.
    #[must_use]
    pub const fn from_layout(layout: &BlockLayout, levels: LevelTable) -> Self {
        Self {
            geometry: LevelGeometry::Packed(*layout),
            levels,
        }
    }

    /// Elements one block covers.
    #[must_use]
    pub const fn elements(&self) -> u16 {
        match &self.geometry {
            LevelGeometry::Packed(layout) => layout.elements,
            LevelGeometry::Codebook { elements, .. } => *elements,
        }
    }

    /// Bytes one block occupies, scale included.
    #[must_use]
    pub const fn bytes(&self) -> u16 {
        match &self.geometry {
            LevelGeometry::Packed(layout) => layout.block_bytes,
            LevelGeometry::Codebook { bytes, .. } => *bytes,
        }
    }

    /// The block scale and where it lives.
    #[must_use]
    pub const fn scale(&self) -> Field {
        match &self.geometry {
            LevelGeometry::Packed(layout) => layout.scale,
            LevelGeometry::Codebook { scale, .. } => *scale,
        }
    }

    /// The exact packing, for the affine formats.
    #[must_use]
    pub const fn packing(&self) -> Option<BlockLayout> {
        match &self.geometry {
            LevelGeometry::Packed(layout) => Some(*layout),
            LevelGeometry::Codebook { .. } => None,
        }
    }

    /// The linear ladder a block layout's own affine code map describes.
    #[must_use]
    pub const fn ladder(layout: &BlockLayout) -> LevelTable {
        let plane_bits = layout.element.low.bits as u32
            + match layout.element.high {
                Some(plane) => plane.bits as u32,
                None => 0,
            };
        let count = if layout.element.packed_radix > 1 {
            layout.element.packed_radix as u16
        } else if plane_bits >= 16 {
            // a code plane this wide has more levels than a u16 can name; `u16::MAX` is the
            // largest set the ladder can describe and no real format reaches it
            u16::MAX
        } else {
            (1u32 << plane_bits) as u16
        };
        LevelTable::Linear {
            count,
            zero_point: -layout.element.bias,
            step: layout.multiplier,
        }
    }

    /// A ternary or explicit-ladder format, projected from its packing.
    #[must_use]
    pub const fn ladder_of(layout: &BlockLayout) -> Self {
        Self::from_layout(layout, Self::ladder(layout))
    }

    /// A codebook format whose table lands with its gguf decoder: geometry only, and a decode
    /// that reports [`crate::NumericError::NoDecoder`] rather than a guessed number.
    #[must_use]
    pub const fn external(elements: u16, bytes: u16) -> Self {
        Self {
            geometry: LevelGeometry::Codebook {
                elements,
                bytes,
                scale: F16_SCALE,
            },
            levels: LevelTable::External,
        }
    }
}

const F16_SCALE: Field = Field {
    layout: FloatLayout::F16,
    byte_offset: 0,
};

/// One numeric storage format.
///
/// `Self::Block` carries a whole [`BlockLayout`] (520 bytes measured) against a `Scalar`'s 8
/// bytes, so this enum is 520 bytes wide. It is never heap-allocated and every value in
/// [`Self::ALL`] is a compile-time constant; boxing the block arm would trade a `memcpy` the
/// caller already holds for an allocation on a per-tensor hot path, and every decode entry point
/// takes it by reference for exactly that reason.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum NumericFormat {
    /// One number per value: exponent bits, mantissa bits, specials, bias, sign.
    Scalar(FloatLayout),
    /// A run of codes under a shared scale, decoded through its exact packing.
    Block(BlockLayout),
    /// A run of codes whose value is an explicit level.
    Levels(LevelFormat),
}

/// The `Debug` spelling of every wire format, in [`NumericFormat::ALL`] order. These are the
/// variant identifiers the enum carried before it became one type with three arms, and they are
/// load-bearing: they appear in telemetry tags and recorded training artifacts
/// (`proxima-autograd/tests/packed_brain_float_moe_training.rs` compares `"Bf8E5M2"`), so a
/// rename would silently re-label evidence that is already on disk.
impl core::fmt::Debug for NumericFormat {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if let Some(position) = NumericFormat::ALL.iter().position(|named| named == self) {
            return formatter.write_str(NumericFormat::DEBUG_NAMES[position]);
        }
        match self {
            NumericFormat::Scalar(layout) => formatter.debug_tuple("Scalar").field(layout).finish(),
            NumericFormat::Block(layout) => formatter.debug_tuple("Block").field(layout).finish(),
            NumericFormat::Levels(levels) => formatter.debug_tuple("Levels").field(levels).finish(),
        }
    }
}

impl NumericFormat {
    /// Elements one block covers; 1 for a scalar.
    #[must_use]
    pub const fn elements_per_block(&self) -> u16 {
        match self {
            Self::Scalar(_) => 1,
            Self::Block(layout) => layout.elements,
            Self::Levels(levels) => levels.elements(),
        }
    }

    /// Bytes one block occupies; whole bytes of the scalar for a scalar format.
    #[must_use]
    pub const fn bytes_per_block(&self) -> u16 {
        match self {
            Self::Scalar(layout) => layout.width().div_ceil(8) as u16,
            Self::Block(layout) => layout.block_bytes,
            Self::Levels(levels) => levels.bytes(),
        }
    }

    /// The block's own scale layout, `None` for a scalar format (which *is* its own value).
    #[must_use]
    pub const fn scale(&self) -> Option<Field> {
        match self {
            Self::Scalar(_) => None,
            Self::Block(layout) => Some(layout.scale),
            Self::Levels(levels) => Some(levels.scale()),
        }
    }

    /// The exact packing of a block format; `None` for a scalar and for a trained codebook whose
    /// code address is not an affine map.
    #[must_use]
    pub const fn block_layout(&self) -> Option<BlockLayout> {
        match self {
            Self::Scalar(_) => None,
            Self::Block(layout) => Some(*layout),
            Self::Levels(levels) => levels.packing(),
        }
    }

    /// The scalar layout one element decodes through, when the element *is* a scalar.
    #[must_use]
    pub const fn element_layout(&self) -> Option<FloatLayout> {
        match self {
            Self::Scalar(layout) => Some(*layout),
            Self::Block(layout) => Some(layout.element.code),
            Self::Levels(levels) => match &levels.geometry {
                LevelGeometry::Packed(layout) => Some(layout.element.code),
                LevelGeometry::Codebook { .. } => None,
            },
        }
    }

    /// Whether the buffer holds an activation a `BlockQuantize` node already quantized.
    #[must_use]
    pub fn is_quantized_activation(&self) -> bool {
        *self == Self::Q8_1 || *self == Self::Q8_1Grouped
    }

    /// Elements that share the smallest whole-byte storage unit: 2 for a four-bit scalar (two
    /// E2M1 nibbles per byte), 1 for a scalar at or above a byte, and the block's own element
    /// count for a block or level-set format.
    #[must_use]
    pub const fn packed_elements(&self) -> u16 {
        match self {
            // a zero-width layout (sign Absent, no mantissa, no exponent) stores no element and
            // no byte; `decode_blocks` rejects it, and 0 here keeps `8 / 0` out of the accessor
            Self::Scalar(layout) if layout.width() == 0 => 0,
            Self::Scalar(layout) if layout.width() < 8 => 8 / layout.width() as u16,
            Self::Scalar(_) => 1,
            Self::Block(layout) => layout.elements,
            Self::Levels(levels) => levels.elements(),
        }
    }

    /// Bytes of the smallest whole-byte storage unit; the sibling of [`Self::packed_elements`].
    #[must_use]
    pub const fn packed_bytes(&self) -> u16 {
        match self {
            Self::Scalar(layout) if layout.width() == 0 => 0,
            Self::Scalar(layout) if layout.width() < 8 => 1,
            Self::Scalar(layout) => layout.width().div_ceil(8) as u16,
            Self::Block(layout) => layout.block_bytes,
            Self::Levels(levels) => levels.bytes(),
        }
    }

    /// Weights per fp6 E3M2 block, one f16 absmax scale each.
    pub const FP6_E3M2_BLOCK_ELEMENTS: usize = 32;

    /// Bytes per fp6 E3M2 block: the 6-bit E3M2 fields packed, then the f16 scale.
    pub const FP6_E3M2_BLOCK_BYTES: usize = BlockLayout::FP6_E3M2.block_bytes as usize;

    /// Every format, in wire-tag order: index i carries tag i.
    pub const ALL: [Self; 36] = [
        Self::Q2K,
        Self::Q3K,
        Self::Q4K,
        Self::Q5K,
        Self::Q6K,
        Self::Q8_0,
        Self::Q4_0,
        Self::Float16,
        Self::BFloat16,
        Self::Q5_1,
        Self::Q5_0,
        Self::Q4_1,
        Self::Q8_1,
        Self::Q8K,
        Self::Iq1S,
        Self::Iq1M,
        Self::Iq2Xxs,
        Self::Iq2Xs,
        Self::Iq2S,
        Self::Iq3Xxs,
        Self::Iq3S,
        Self::Iq4Nl,
        Self::Iq4Xs,
        Self::Tq10,
        Self::Tq20,
        Self::Mxfp4,
        Self::Nvfp4,
        Self::Q1_0,
        Self::Q2_0,
        Self::BFloat8,
        Self::Bf8E5M2,
        Self::Bf4E2M1,
        Self::Fp6E3M2,
        Self::Pq2_0,
        Self::Ptq1_0,
        Self::Q8_1Grouped,
    ];

    /// The `Debug` spelling of every format, same order as [`Self::ALL`].
    pub const DEBUG_NAMES: [&'static str; 36] = [
        "Q2K",
        "Q3K",
        "Q4K",
        "Q5K",
        "Q6K",
        "Q8_0",
        "Q4_0",
        "Float16",
        "BFloat16",
        "Q5_1",
        "Q5_0",
        "Q4_1",
        "Q8_1",
        "Q8K",
        "Iq1S",
        "Iq1M",
        "Iq2Xxs",
        "Iq2Xs",
        "Iq2S",
        "Iq3Xxs",
        "Iq3S",
        "Iq4Nl",
        "Iq4Xs",
        "Tq10",
        "Tq20",
        "Mxfp4",
        "Nvfp4",
        "Q1_0",
        "Q2_0",
        "BFloat8",
        "Bf8E5M2",
        "Bf4E2M1",
        "Fp6E3M2",
        "Pq2_0",
        "Ptq1_0",
        "Q8_1Grouped",
    ];

    /// Every canonical name, same order as [`Self::ALL`]. One table serves
    /// [`Self::name`] and [`Self::from_name`]; with [`Self::DEBUG_NAMES`] it is one of three
    /// constant arrays a new format adds one row to, and nothing else in the workspace.
    pub const NAMES: [&'static str; 36] = [
        "q2_k",
        "q3_k",
        "q4_k",
        "q5_k",
        "q6_k",
        "q8_0",
        "q4_0",
        "float16",
        "bfloat16",
        "q5_1",
        "q5_0",
        "q4_1",
        "q8_1",
        "q8_k",
        "iq1_s",
        "iq1_m",
        "iq2_xxs",
        "iq2_xs",
        "iq2_s",
        "iq3_xxs",
        "iq3_s",
        "iq4_nl",
        "iq4_xs",
        "tq1_0",
        "tq2_0",
        "mxfp4",
        "nvfp4",
        "q1_0",
        "q2_0",
        "float8_e5m2",
        "bfloat8",
        "e2m1",
        "fp6_e3m2",
        "pq2_0",
        "ptq1_0",
        "q8_1_grouped",
    ];

    /// The stable wire byte a sidecar tags this format with, unchanged from the codec tag it
    /// replaces, so a sidecar written before this crate existed still decodes. `None` for a
    /// scalar layout that is not one of the wire codecs.
    #[must_use]
    pub fn tag(&self) -> Option<u8> {
        Self::ALL
            .iter()
            .position(|named| named == self)
            .map(|position| position as u8)
    }

    /// The inverse of [`Self::tag`]. `None` for a byte no format claims.
    #[must_use]
    pub fn from_tag(tag: u8) -> Option<Self> {
        Self::ALL.get(usize::from(tag)).copied()
    }

    /// The format's canonical lowercase name. A scalar layout outside the wire codec set falls
    /// back to its [`FloatLayout`] spelling.
    #[must_use]
    pub fn name(&self) -> Option<&'static str> {
        Self::ALL
            .iter()
            .position(|named| named == self)
            .map(|position| Self::NAMES[position])
            .or_else(|| match self {
                Self::Scalar(layout) => layout.name(),
                Self::Block(_) | Self::Levels(_) => None,
            })
    }

    /// The format a canonical name denotes; the inverse of [`Self::name`] over [`Self::ALL`].
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::NAMES
            .iter()
            .position(|named| *named == name)
            .map(|position| Self::ALL[position])
    }

    /// q2_k: 256 four-bit codes, 16 sub-blocks each with a 4-bit scale and min.
    pub const Q2K: Self = Self::Block(BlockLayout::Q2_K);
    pub const Q3K: Self = Self::Block(BlockLayout::Q3_K);
    pub const Q4K: Self = Self::Block(BlockLayout::Q4_K);
    pub const Q5K: Self = Self::Block(BlockLayout::Q5_K);
    pub const Q6K: Self = Self::Block(BlockLayout::Q6_K);
    pub const Q8_0: Self = Self::Block(BlockLayout::Q8_0);
    pub const Q4_0: Self = Self::Block(BlockLayout::Q4_0);
    pub const Q5_1: Self = Self::Block(BlockLayout::Q5_1);
    pub const Q5_0: Self = Self::Block(BlockLayout::Q5_0);
    pub const Q4_1: Self = Self::Block(BlockLayout::Q4_1);
    pub const Q8_1: Self = Self::Block(BlockLayout::Q8_1);
    pub const Q8K: Self = Self::Block(BlockLayout::Q8_K);
    pub const Q8_1Grouped: Self = Self::Block(BlockLayout::Q8_1_GROUPED);

    /// f16: one IEEE binary16 element per value.
    pub const Float16: Self = Self::Scalar(FloatLayout::F16);
    /// bf16: one bfloat16 element per value.
    pub const BFloat16: Self = Self::Scalar(FloatLayout::BF16);
    /// ggml's `BFloat8`: E5M2 with IEEE infinities.
    pub const BFloat8: Self = Self::Scalar(FloatLayout::E5M2);
    /// bf8 E5M2 under the GGUF saturation policy.
    pub const Bf8E5M2: Self = Self::Scalar(FloatLayout::E5M2_SATURATING);
    /// bf4: one E2M1 element per value.
    pub const Bf4E2M1: Self = Self::Scalar(FloatLayout::E2M1);
    /// fp6 E3M2: 32 six-bit fields under one f16 scale, 26 bytes per block.
    pub const Fp6E3M2: Self = Self::Block(BlockLayout::FP6_E3M2);
    /// MXFP4: 32 E2M1 fields under one E8M0 block scale, 17 bytes per block.
    pub const Mxfp4: Self = Self::Block(BlockLayout::MXFP4);
    /// NVFP4: 64 E2M1 fields under four E4M3 sub-block scales, 36 bytes per block.
    pub const Nvfp4: Self = Self::Block(BlockLayout::NVFP4);

    /// q1_0: 128 one-bit levels over {-1, +1} under one f16 scale.
    pub const Q1_0: Self = Self::Levels(LevelFormat::ladder_of(&BlockLayout::Q1_0));
    /// q2_0: 64 two-bit levels over {-1, 0, +1, +2}.
    pub const Q2_0: Self = Self::Levels(LevelFormat::ladder_of(&BlockLayout::Q2_0));
    pub const Pq2_0: Self = Self::Levels(LevelFormat::ladder_of(&BlockLayout::PQ2_0));
    pub const Ptq1_0: Self = Self::Levels(LevelFormat::ladder_of(&BlockLayout::PTQ1_0));
    /// tq1_0: 256 ternary trits packed base-3, 54 bytes per super-block.
    pub const Tq10: Self = Self::Levels(LevelFormat::ladder_of(&BlockLayout::TQ1_0));
    /// tq2_0: 256 two-bit levels, 66 bytes per super-block.
    pub const Tq20: Self = Self::Levels(LevelFormat::ladder_of(&BlockLayout::TQ2_0));

    /// iq4_nl: Q4_0's nibble packing with a trained 16-entry level ladder.
    pub const Iq4Nl: Self = Self::Levels(LevelFormat::from_layout(
        &BlockLayout::IQ4_NL,
        LevelTable::Explicit(&crate::tables::KVALUES_IQ4_NL),
    ));

    /// iq2_xs: 256 elements per super-block, a 9-bit grid index plus a sign index per group.
    pub const Iq2Xs: Self = Self::Levels(LevelFormat {
        geometry: LevelGeometry::Codebook {
            elements: 256,
            bytes: 74,
            scale: F16_SCALE,
        },
        levels: LevelTable::Iq2XsGrid,
    });

    /// iq3_xxs: 256 elements per super-block, two 8-bit grid indices per eight elements.
    pub const Iq3Xxs: Self = Self::Levels(LevelFormat {
        geometry: LevelGeometry::Codebook {
            elements: 256,
            bytes: 98,
            scale: F16_SCALE,
        },
        levels: LevelTable::Iq3XxsGrid,
    });

    /// iq1_s, iq1_m, iq2_xxs, iq2_s, iq3_s, iq4_xs: codebooks that land with their gguf decoder.
    pub const Iq1S: Self = Self::Levels(LevelFormat::external(256, 50));
    pub const Iq1M: Self = Self::Levels(LevelFormat::external(256, 56));
    pub const Iq2Xxs: Self = Self::Levels(LevelFormat::external(256, 66));
    pub const Iq2S: Self = Self::Levels(LevelFormat::external(256, 82));
    pub const Iq3S: Self = Self::Levels(LevelFormat::external(256, 110));
    pub const Iq4Xs: Self = Self::Levels(LevelFormat::external(256, 136));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::float_layout::{SignEncoding as Sign, Specials as Specs};

    /// Every float layout constant the crate declares. Pinned by count so a new constant has to
    /// be added here and to [`crate::float_layout::NAMED_LAYOUTS`] in the same edit.
    const FLOAT_LAYOUT_CONSTANTS: [FloatLayout; 30] = [
        FloatLayout::E5M2,
        FloatLayout::E4M3FN,
        FloatLayout::E5M2_SATURATING,
        FloatLayout::F16,
        FloatLayout::E2M1,
        FloatLayout::E2M2,
        FloatLayout::E3M2,
        FloatLayout::E2M3,
        FloatLayout::BF32,
        FloatLayout::BF16,
        FloatLayout::BF8,
        FloatLayout::BF4,
        FloatLayout::F64,
        FloatLayout::F8_E8M0,
        FloatLayout::INT8,
        FloatLayout::UINT8,
        FloatLayout::INT16,
        FloatLayout::UINT16,
        FloatLayout::INT32,
        FloatLayout::UINT32,
        FloatLayout::INT64,
        FloatLayout::UINT64,
        FloatLayout::INT128,
        FloatLayout::UINT128,
        FloatLayout::UINT1,
        FloatLayout::UINT2,
        FloatLayout::UINT3,
        FloatLayout::UINT4,
        FloatLayout::UINT5,
        FloatLayout::UINT6,
    ];

    #[test]
    fn the_debug_table_stays_aligned_with_the_format_table() {
        assert_eq!(NumericFormat::DEBUG_NAMES.len(), NumericFormat::ALL.len());
        let mut seen = std::vec::Vec::new();
        for (index, format) in NumericFormat::ALL.iter().enumerate() {
            let spelling = std::format!("{format:?}");
            assert_eq!(spelling, NumericFormat::DEBUG_NAMES[index], "row {index}");
            assert!(
                !seen.contains(&spelling),
                "two formats share the debug spelling {spelling}"
            );
            seen.push(spelling);
        }
        assert!(
            std::format!("{:?}", NumericFormat::Scalar(FloatLayout::E2M3)).starts_with("Scalar("),
            "a scalar outside the wire set prints structurally, not as a codec name"
        );
    }

    #[test]
    fn a_zero_width_scalar_keeps_packed_elements_total() {
        let empty = FloatLayout {
            exponent_bits: 0,
            mantissa_bits: 0,
            specials: Specs::None,
            bias_override: None,
            sign: Sign::Absent,
        };
        assert_eq!(NumericFormat::Scalar(empty).packed_elements(), 0);
        assert_eq!(NumericFormat::Scalar(empty).packed_bytes(), 0);
        assert_eq!(NumericFormat::Scalar(empty).bytes_per_block(), 0);
    }

    #[test]
    fn every_float_layout_constant_is_reachable_as_a_scalar_format() {
        assert_eq!(FLOAT_LAYOUT_CONSTANTS.len(), 30, "pinned: declared float layouts");
        for layout in FLOAT_LAYOUT_CONSTANTS {
            let format = NumericFormat::Scalar(layout);
            assert_eq!(format.elements_per_block(), 1);
            assert_eq!(
                usize::from(format.bytes_per_block()),
                usize::from(layout.width().div_ceil(8))
            );
        }
    }

    #[test]
    fn the_scalar_arm_is_the_layout_identity_not_a_copy() {
        assert_eq!(
            NumericFormat::Scalar(FloatLayout::BF32),
            NumericFormat::Scalar(FloatLayout::BF32)
        );
        assert_eq!(
            NumericFormat::Scalar(FloatLayout::BF32).element_layout(),
            Some(FloatLayout::BF32)
        );
    }

    #[test]
    fn the_format_table_is_total_pinned_and_ordered_by_wire_tag() {
        assert_eq!(NumericFormat::ALL.len(), 36, "pinned: every codec is one format");
        assert_eq!(NumericFormat::NAMES.len(), NumericFormat::ALL.len());

        for (index, format) in NumericFormat::ALL.iter().enumerate() {
            assert_eq!(format.tag(), Some(index as u8), "{format:?} carries tag {index}");
            assert_eq!(NumericFormat::from_tag(index as u8), Some(*format));
        }
        assert_eq!(NumericFormat::from_tag(36), None);
        assert_eq!(NumericFormat::from_tag(u8::MAX), None);
    }

    #[test]
    fn every_format_name_round_trips_and_no_two_share_one() {
        let mut seen = std::vec::Vec::new();
        for (index, format) in NumericFormat::ALL.iter().enumerate() {
            let name = NumericFormat::NAMES[index];
            assert_eq!(format.name(), Some(name), "{format:?}");
            assert_eq!(NumericFormat::from_name(name), Some(*format));
            assert!(!seen.contains(&name), "duplicate format name {name}");
            seen.push(name);
        }
        assert_eq!(NumericFormat::from_name("q4_0_almost"), None);
    }

    #[test]
    fn the_ternary_formats_are_level_sets_over_their_own_packing() {
        let cases = [
            (NumericFormat::Q1_0, BlockLayout::Q1_0),
            (NumericFormat::Q2_0, BlockLayout::Q2_0),
            (NumericFormat::Pq2_0, BlockLayout::PQ2_0),
            (NumericFormat::Ptq1_0, BlockLayout::PTQ1_0),
            (NumericFormat::Tq10, BlockLayout::TQ1_0),
            (NumericFormat::Tq20, BlockLayout::TQ2_0),
        ];
        assert_eq!(cases.len(), 6, "pinned: ternary-family formats");

        for (format, packing) in cases {
            assert_eq!(
                format.block_layout(),
                Some(packing),
                "{format:?} must report the packing it is projected from"
            );
            let NumericFormat::Levels(levels) = format else {
                panic!("{format:?} must be a level-set format");
            };
            assert_eq!(
                levels.levels,
                LevelFormat::ladder(&packing),
                "{format:?}'s ladder must be its own packing's"
            );
        }
    }

    #[test]
    fn ternary_levels_are_the_signed_ladder_and_iq4_nl_is_the_trained_table() {
        let NumericFormat::Levels(q1) = NumericFormat::Q1_0 else {
            panic!("q1_0 is a level format");
        };
        assert_eq!(q1.levels.level(0), Some(-1.0));
        assert_eq!(q1.levels.level(1), Some(1.0));
        assert_eq!(q1.levels.level(2), None, "q1_0 has two levels");

        let NumericFormat::Levels(q2) = NumericFormat::Q2_0 else {
            panic!("q2_0 is a level format");
        };
        assert_eq!(
            (0..4).map(|code| q2.levels.level(code)).collect::<std::vec::Vec<_>>(),
            std::vec![Some(-1.0), Some(0.0), Some(1.0), Some(2.0)]
        );

        let NumericFormat::Levels(iq4) = NumericFormat::Iq4Nl else {
            panic!("iq4_nl is a level format");
        };
        assert_eq!(iq4.levels.level(0), Some(-127.0));
        assert_eq!(iq4.levels.level(15), Some(113.0));
        assert_ne!(iq4.levels.level(8), Some(0.0), "the ladder is non-linear");
    }

    #[test]
    fn quantized_activation_formats_are_exactly_the_q8_1_pair() {
        let activation = NumericFormat::ALL
            .iter()
            .filter(|format| format.is_quantized_activation())
            .count();
        assert_eq!(activation, 2);
    }

    #[test]
    fn the_block_arm_reports_its_own_scale_and_high_plane() {
        assert_eq!(
            NumericFormat::Mxfp4.scale(),
            Some(Field {
                layout: FloatLayout::F8_E8M0,
                byte_offset: 0
            })
        );
        assert_eq!(BlockLayout::Q5_0.high_bits(), Some(1));
        assert_eq!(BlockLayout::Q4_0.high_bits(), None);
        assert_eq!(NumericFormat::Scalar(FloatLayout::F16).scale(), None);
    }
}

#[cfg(all(test, feature = "std"))]
mod workspace_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::fs;
    use std::path::{Path, PathBuf};

    /// The declarations that must exist only in this crate. `Codec` is not listed: it is a
    /// `pub use` of [`crate::NumericFormat`] inside this crate's own `codec` module, never a
    /// second definition anywhere.
    const FORBIDDEN: [&str; 8] = [
        "pub enum Codec",
        "pub enum DType",
        "pub enum NumericFormat",
        "pub struct FloatLayout",
        "pub struct NumericPolicy",
        "pub struct Numerics",
        "pub struct BlockLayout",
        "pub struct LevelFormat",
    ];

    fn workspace_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the crate sits inside the workspace")
            .to_path_buf()
    }

    /// `line` declares `pattern`, with the pattern ending on a word boundary so
    /// `pub enum CodecPolicyWeights` is not a hit for `pub enum Codec`.
    fn declares(line: &str, pattern: &str) -> bool {
        let mut rest = line;
        while let Some(found) = rest.find(pattern) {
            let after = rest[found + pattern.len()..].chars().next();
            if !after.is_some_and(|character| character.is_alphanumeric() || character == '_') {
                return true;
            }
            rest = &rest[found + pattern.len()..];
        }
        false
    }

    fn rust_files(root: &Path, found: &mut Vec<PathBuf>) {
        let Ok(entries) = fs::read_dir(root) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if path.is_dir() {
                if matches!(name.as_ref(), "target" | ".git" | "node_modules" | "proxima-numeric")
                {
                    continue;
                }
                rust_files(&path, found);
            } else if name.ends_with(".rs") {
                found.push(path);
            }
        }
    }

    #[test]
    fn no_numeric_type_is_defined_outside_this_crate() {
        let mut files = Vec::new();
        rust_files(&workspace_root(), &mut files);
        assert!(
            files.len() > 200,
            "the scan must actually see the workspace, saw {} files",
            files.len()
        );

        let mut offenders = Vec::new();
        for path in &files {
            let Ok(text) = fs::read_to_string(path) else {
                continue;
            };
            for line in text.lines() {
                if FORBIDDEN.iter().any(|pattern| declares(line, pattern)) {
                    offenders.push(std::format!("{}: {}", path.display(), line.trim()));
                }
            }
        }
        assert_eq!(
            offenders.len(),
            0,
            "a numeric type is defined outside proxima-numeric: {offenders:#?}"
        );
    }

    #[test]
    fn this_crate_depends_on_no_workspace_crate() {
        let manifest = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"))
            .expect("the crate has a manifest");
        let dependencies = manifest
            .split("[dependencies]")
            .nth(1)
            .expect("the manifest declares dependencies")
            .split("\n[")
            .next()
            .expect("the dependency section ends");
        let workspace_dependencies = dependencies
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .filter(|line| line.starts_with("proxima-") || line.contains("path = \"../"))
            .count();
        assert_eq!(
            workspace_dependencies, 0,
            "proxima-numeric must stay a dependency sink; it found {workspace_dependencies}"
        );
        assert!(
            dependencies.contains("thiserror"),
            "the manifest parsing must actually read the dependency section"
        );
    }
}
