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

use crate::block_layout::{BlockLayout, Field, Plane};
use crate::dtype::DType;
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
    /// `value = sign * magnitude * 2^-unit_shift`: the code indexes `grid`, one entry per `lanes`
    /// elements, each entry packing `lanes` unsigned byte magnitudes (lane 0 in the low byte);
    /// element `i` takes lane `i % lanes`. With `sign_table` `None`, `sign` is a
    /// one-bit-per-element plane and a set bit negates (IQ3_S's `signs[QK_K/8]`). With
    /// `Some(table)` it is a plane of indices into `table` shared by eight elements, and element
    /// `i` is negated when bit `i % 8` of `table[index]` is set (IQ3_XXS's seven-bit index into
    /// `ksigns_iq2xs`; eight elements span two grid entries, so the sign cannot be per entry).
    /// `unit_shift` is the constant the dequantizer folds into the sub-block scale: 0 for IQ3_S,
    /// 2 for IQ3_XXS (`0.5 * 0.5`).
    SignedGrid {
        grid: &'static [u32],
        lanes: u8,
        sign: Plane,
        sign_table: Option<&'static [u8]>,
        unit_shift: u8,
    },
    /// `value = sign * magnitude * 2^-unit_shift`: the code indexes `grid`, one `u64` entry per
    /// eight elements, each packing eight unsigned magnitude bytes (lane 0 in the low byte);
    /// element `i` takes lane `i % 8`. The sign is `sign`: with `sign_table` `None` it is a
    /// one-bit-per-element plane (a set bit negates; IQ2_S); with `Some(table)` it is a plane of
    /// indices into `table`, and element `i` is negated when bit `i % 8` of `table[index]` is set
    /// (IQ2_XXS's seven-bit index into `ksigns_iq2xs` and its `kmask_iq2xs`). `unit_shift` is the
    /// constant `0.25 * 0.5` the IQ2 dequantizers fold into the sub-block scale: 3.
    SignedGrid64 {
        grid: &'static [u64],
        sign: Plane,
        sign_table: Option<&'static [u8]>,
        unit_shift: u8,
    },
    /// `value = grid_lane + delta`, `delta = +-2^-delta_shift`: the code indexes `grid`, one `u64`
    /// entry per eight elements, each packing eight *signed* byte levels (lane 0 in the low byte);
    /// element `i` takes lane `i % 8`. `delta` is a one-bit-per-element plane addressed like the
    /// code; a set bit makes the delta negative. IQ1_S and IQ1_M's `iq1s_grid` with their
    /// `IQ1S_DELTA = 0.125 = 2^-3`; the delta is added *before* the sub-block scale multiplies.
    DeltaGrid {
        grid: &'static [u64],
        delta: Plane,
        delta_shift: u8,
    },
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
            Self::SignedGrid { .. }
            | Self::SignedGrid64 { .. }
            | Self::DeltaGrid { .. } => None,
        }
    }
}

/// Where a level-set format's codes live.
///
/// Every named format is `Packed`: each carries its exact packing as a [`BlockLayout`]. `Codebook`
/// is the shape-only arm for a format whose code address is not an affine map of the element
/// index; no named format uses it today.
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
}

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
        if let Some(name) = self.debug_name() {
            return formatter.write_str(name);
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

    /// Bytes `elements` values occupy packed in this format, `None` unless `elements` is a whole
    /// number of blocks: 64 values of Q4_0 are 2 blocks, 36 bytes; 17 are not whole.
    #[must_use]
    pub const fn packed_len(&self, elements: usize) -> Option<usize> {
        let block_elements = match self.elements_per_block() {
            0 => 1,
            count => count as usize,
        };
        if !elements.is_multiple_of(block_elements) {
            return None;
        }
        Some(elements / block_elements * self.bytes_per_block() as usize)
    }

    /// Whether two formats are scalar layouts that convert into each other changing no bit
    /// ([`FloatLayout::is_identity_of`]). identity is a property of scalar layouts; any block or
    /// level format answers `false`, even against itself.
    #[must_use]
    pub const fn is_identity_of(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Scalar(layout), Self::Scalar(wider)) => layout.is_identity_of(wider),
            _ => false,
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

    /// The packing of a format whose values the layout alone determines: every `Block`, and a
    /// `Levels` format only when its level map is the layout's own linear ladder
    /// ([`LevelFormat::ladder`]). `None` for a scalar, for a trained codebook, and for a format
    /// whose level table is not that ladder (IQ4_NL, IQ4_XS, IQ3_S): a reader of the layout alone
    /// would decode those with the wrong levels. Such a caller reads [`LevelFormat::packing`] and
    /// [`LevelFormat::levels`] explicitly.
    #[must_use]
    pub const fn block_layout(&self) -> Option<BlockLayout> {
        match self {
            Self::Scalar(_) => None,
            Self::Block(layout) => Some(*layout),
            Self::Levels(levels) => match levels.packing() {
                Some(layout) => match (levels.levels, LevelFormat::ladder(&layout)) {
                    (
                        LevelTable::Linear { count, zero_point, step },
                        LevelTable::Linear {
                            count: own_count,
                            zero_point: own_zero_point,
                            step: own_step,
                        },
                    ) if count == own_count && zero_point == own_zero_point && step == own_step => {
                        Some(layout)
                    }
                    _ => None,
                },
                None => None,
            },
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

    /// Every wire-tagged format, in wire-tag order: index i carries tag i. A named format with no
    /// wire tag is in [`Self::UNTAGGED`] instead, so adding one never shifts a tag.
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

    /// Named formats that carry no wire tag, with their `Debug` spelling. Lives beside
    /// [`Self::ALL`] and [`Self::DEBUG_NAMES`] so [`Self::debug_name`] resolves every named
    /// constant, tagged or not, in one place.
    pub const UNTAGGED: [(Self, &'static str); 2] =
        [(Self::Float32, "Float32"), (Self::Bool, "Bool")];

    /// The `Debug` spelling of a named format, wire-tagged or untagged; `None` for a layout this
    /// crate has no constant for.
    #[must_use]
    pub fn debug_name(&self) -> Option<&'static str> {
        Self::ALL
            .iter()
            .position(|named| named == self)
            .map(|position| Self::DEBUG_NAMES[position])
            .or_else(|| {
                Self::UNTAGGED
                    .iter()
                    .find_map(|(named, name)| (named == self).then_some(*name))
            })
    }

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
                Self::Levels(_) if *self == Self::Bool => Some(Self::BOOL_NAME),
                Self::Block(_) | Self::Levels(_) => None,
            })
    }

    /// The canonical spelling of [`Self::Bool`], the one `DType::Bool` serializes as.
    pub const BOOL_NAME: &'static str = "bool";

    /// Spellings [`Self::from_name`] also accepts, beside [`Self::NAMES`]: the ggml / llama.cpp
    /// `--cache-type-k` short names for the three scalar types they spell differently. The
    /// block types (`q8_0`, `q4_0`, `iq4_nl`, ...) already share their ggml spelling in
    /// [`Self::NAMES`]. [`Self::name`] never returns these.
    pub const ALIASES: [(&'static str, Self); 3] = [
        ("f32", Self::Float32),
        ("f16", Self::Float16),
        ("bf16", Self::BFloat16),
    ];

    /// The format a name denotes: a canonical name from [`Self::NAMES`], `bool`, a name in
    /// [`Self::ALIASES`], or a scalar layout's own name (`float32`). The inverse of
    /// [`Self::name`] over every format that has one.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        if name == Self::BOOL_NAME {
            return Some(Self::Bool);
        }
        Self::NAMES
            .iter()
            .position(|named| *named == name)
            .map(|position| Self::ALL[position])
            .or_else(|| {
                Self::ALIASES
                    .iter()
                    .find_map(|(alias, format)| (*alias == name).then_some(*format))
            })
            .or_else(|| FloatLayout::from_name(name).map(Self::Scalar))
    }

    /// Bytes one element occupies stored alone, rounded up to a whole byte; a block format reports
    /// its block's bytes spread over its elements, rounded up. Equals `DType::size_bytes` for every
    /// `DType` twin.
    #[must_use]
    pub const fn size_bytes(&self) -> usize {
        match self {
            Self::Scalar(layout) => (layout.width() as usize).div_ceil(8),
            Self::Block(_) | Self::Levels(_) => {
                (self.bytes_per_block() as usize).div_ceil(self.elements_per_block() as usize)
            }
        }
    }

    /// Storage bits one element occupies: the layout's own width for a scalar, the block's bits
    /// spread over its elements (rounded down) otherwise. Equals `DType::bits` for every `DType`
    /// twin.
    #[must_use]
    pub const fn bits(&self) -> usize {
        match self {
            Self::Scalar(layout) => layout.width() as usize,
            Self::Block(_) | Self::Levels(_) => {
                self.bytes_per_block() as usize * 8 / self.elements_per_block() as usize
            }
        }
    }

    /// Whether the element is a scalar with an exponent field; a block or level format is a
    /// quantized code, not a float element.
    #[must_use]
    pub const fn is_float(&self) -> bool {
        match self {
            Self::Scalar(layout) => layout.is_float(),
            Self::Block(_) | Self::Levels(_) => false,
        }
    }

    /// Whether a fold can accumulate in this encoding without widening.
    #[must_use]
    pub const fn accumulates_in_place(&self) -> bool {
        match self {
            Self::Scalar(layout) => layout.accumulates_in_place(),
            Self::Block(_) | Self::Levels(_) => false,
        }
    }

    /// Whether the element is a whole-number scalar suitable for gather indices.
    #[must_use]
    pub const fn is_integer(&self) -> bool {
        match self {
            Self::Scalar(layout) => layout.is_integer(),
            Self::Block(_) | Self::Levels(_) => false,
        }
    }

    /// Whether the element is a signed whole-number scalar.
    #[must_use]
    pub const fn is_signed_integer(&self) -> bool {
        match self {
            Self::Scalar(layout) => layout.is_signed_integer(),
            Self::Block(_) | Self::Levels(_) => false,
        }
    }

    /// The `DType` that describes the same elements: a scalar layout, or [`Self::Bool`]. `None`
    /// for a block or level format, which `DType` cannot express.
    #[must_use]
    pub fn to_dtype(&self) -> Option<DType> {
        match self {
            Self::Scalar(layout) => Some(DType::Layout(*layout)),
            Self::Levels(_) if self.is_bool() => Some(DType::Bool),
            Self::Block(_) | Self::Levels(_) => None,
        }
    }

    /// Whether this is [`Self::Bool`]: the same equality [`Self::name`] spells `"bool"` by, so a
    /// ladder that only resembles it (the code at another bit) is neither.
    #[must_use]
    pub fn is_bool(&self) -> bool {
        *self == Self::Bool
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

    /// f32: one IEEE binary32 element per value, the layout `DType::FLOAT32` uses. It has no wire
    /// tag, so it is in [`Self::UNTAGGED`], not [`Self::ALL`].
    pub const Float32: Self = Self::Scalar(FloatLayout::BF32);
    /// bool: a two-level ladder `{0, 1}`, one byte per element, no scale bytes. Not in
    /// [`Self::ALL`]: it has no wire tag, and `DType::Bool` is its only prior spelling.
    pub const Bool: Self = Self::Levels(LevelFormat::ladder_of(&BlockLayout::BOOL));

    pub const Int8: Self = Self::Scalar(FloatLayout::INT8);
    pub const UInt8: Self = Self::Scalar(FloatLayout::UINT8);
    pub const Int16: Self = Self::Scalar(FloatLayout::INT16);
    pub const UInt16: Self = Self::Scalar(FloatLayout::UINT16);
    pub const Int32: Self = Self::Scalar(FloatLayout::INT32);
    pub const UInt32: Self = Self::Scalar(FloatLayout::UINT32);
    pub const Int64: Self = Self::Scalar(FloatLayout::INT64);
    pub const UInt64: Self = Self::Scalar(FloatLayout::UINT64);
    pub const Int128: Self = Self::Scalar(FloatLayout::INT128);
    pub const UInt128: Self = Self::Scalar(FloatLayout::UINT128);
    pub const Float64: Self = Self::Scalar(FloatLayout::F64);
    pub const Float8E4M3Fn: Self = Self::Scalar(FloatLayout::E4M3FN);
    /// fp6 E3M2 as one scalar element per value, the layout `DType::FP6_E3_M2` uses; distinct from
    /// [`Self::Fp6E3M2`], the 32-element block under an f16 scale.
    pub const Fp6E3M2Scalar: Self = Self::Scalar(FloatLayout::E3M2);
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

    /// iq2_xs: 256 elements in 74 bytes, a 9-bit grid index and a 7-bit sign index per eight
    /// elements under 4-bit sub-block scales ([`BlockLayout::IQ2_XS`]).
    pub const Iq2Xs: Self = Self::Levels(LevelFormat::from_layout(
        &BlockLayout::IQ2_XS,
        LevelTable::SignedGrid64 {
            grid: &crate::tables::IQ2XS_GRID,
            sign: BlockLayout::IQ2_XS_SIGN_INDEX,
            sign_table: Some(&crate::tables::KSIGNS_IQ2XS),
            unit_shift: 3,
        },
    ));

    /// iq3_xxs: 256 elements in 98 bytes, an 8-bit grid index per four elements and a 7-bit sign
    /// index per eight under 4-bit sub-block scales ([`BlockLayout::IQ3_XXS`]).
    pub const Iq3Xxs: Self = Self::Levels(LevelFormat::from_layout(
        &BlockLayout::IQ3_XXS,
        LevelTable::SignedGrid {
            grid: &crate::tables::IQ3XXS_GRID,
            lanes: 4,
            sign: BlockLayout::IQ3_XXS_SIGN_INDEX,
            sign_table: Some(&crate::tables::KSIGNS_IQ2XS),
            unit_shift: 2,
        },
    ));

    /// iq3_s: 256 elements in 110 bytes, a 9-bit grid index per four elements plus a sign bit per
    /// element under 4-bit sub-block scales ([`BlockLayout::IQ3_S`]).
    pub const Iq3S: Self = Self::Levels(LevelFormat::from_layout(
        &BlockLayout::IQ3_S,
        LevelTable::SignedGrid {
            grid: &crate::tables::IQ3S_GRID,
            lanes: 4,
            sign: BlockLayout::IQ3_S_SIGNS,
            sign_table: None,
            unit_shift: 0,
        },
    ));

    /// iq4_xs: 256 elements in 136 bytes, IQ4_NL's ladder under 6-bit sub-block scales
    /// ([`BlockLayout::IQ4_XS`]).
    pub const Iq4Xs: Self = Self::Levels(LevelFormat::from_layout(
        &BlockLayout::IQ4_XS,
        LevelTable::Explicit(&crate::tables::KVALUES_IQ4_NL),
    ));

    /// iq1_s: 256 elements in 50 bytes, an 11-bit ternary-grid index per eight elements under
    /// 3-bit sub-block scales and a per-sub-block delta ([`BlockLayout::IQ1_S`]).
    pub const Iq1S: Self = Self::Levels(LevelFormat::from_layout(
        &BlockLayout::IQ1_S,
        LevelTable::DeltaGrid {
            grid: &crate::tables::IQ1S_GRID,
            delta: BlockLayout::IQ1_S_DELTA,
            delta_shift: 3,
        },
    ));

    /// iq1_m: 256 elements in 56 bytes, IQ1_S's grid with a delta per eight elements and a
    /// super-scale scattered over the sub-block scale words ([`BlockLayout::IQ1_M`]).
    pub const Iq1M: Self = Self::Levels(LevelFormat::from_layout(
        &BlockLayout::IQ1_M,
        LevelTable::DeltaGrid {
            grid: &crate::tables::IQ1S_GRID,
            delta: BlockLayout::IQ1_M_DELTA,
            delta_shift: 3,
        },
    ));

    /// iq2_xxs: 256 elements in 66 bytes, an 8-bit grid index and a 7-bit sign index per eight
    /// elements under 4-bit sub-block scales ([`BlockLayout::IQ2_XXS`]).
    pub const Iq2Xxs: Self = Self::Levels(LevelFormat::from_layout(
        &BlockLayout::IQ2_XXS,
        LevelTable::SignedGrid64 {
            grid: &crate::tables::IQ2XXS_GRID,
            sign: BlockLayout::IQ2_XXS_SIGN_INDEX,
            sign_table: Some(&crate::tables::KSIGNS_IQ2XS),
            unit_shift: 3,
        },
    ));

    /// iq2_s: 256 elements in 82 bytes, a 10-bit grid index per eight elements plus a sign bit per
    /// element under 4-bit sub-block scales ([`BlockLayout::IQ2_S`]).
    pub const Iq2S: Self = Self::Levels(LevelFormat::from_layout(
        &BlockLayout::IQ2_S,
        LevelTable::SignedGrid64 {
            grid: &crate::tables::IQ2S_GRID,
            sign: BlockLayout::IQ2_S_SIGNS,
            sign_table: None,
            unit_shift: 3,
        },
    ));
}

#[cfg(feature = "config")]
impl serde::Serialize for NumericFormat {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.name() {
            Some(name) => serializer.serialize_str(name),
            None => Err(serde::ser::Error::custom("numeric format has no name")),
        }
    }
}

#[cfg(feature = "config")]
impl<'de> serde::Deserialize<'de> for NumericFormat {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct NumericFormatVisitor;

        impl serde::de::Visitor<'_> for NumericFormatVisitor {
            type Value = NumericFormat;

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter.write_str("a numeric format name such as \"q8_0\", \"f16\" or \"bfloat16\"")
            }

            fn visit_str<E: serde::de::Error>(self, name: &str) -> Result<NumericFormat, E> {
                NumericFormat::from_name(name).ok_or_else(|| {
                    E::custom(format_args!("unknown numeric format name `{name}`"))
                })
            }
        }

        deserializer.deserialize_str(NumericFormatVisitor)
    }
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
    fn packed_len_is_whole_blocks_of_bytes_or_none() {
        assert_eq!(NumericFormat::Float16.packed_len(512), Some(1024));
        assert_eq!(NumericFormat::Q4_0.packed_len(64), Some(36));
        assert_eq!(NumericFormat::Q4_0.packed_len(17), None);
    }

    #[test]
    fn an_untagged_format_names_itself_in_debug_and_holds_no_wire_tag() {
        for (format, name) in NumericFormat::UNTAGGED {
            assert_eq!(std::format!("{format:?}"), name);
            assert!(
                !NumericFormat::ALL.contains(&format),
                "{name} would shift the wire tags"
            );
        }
        assert_eq!(std::format!("{:?}", NumericFormat::Float32), "Float32");
        assert_eq!(NumericFormat::Float32, DType::FLOAT32.format().expect("f32 is numeric"));
    }

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
            nan: Default::default(),
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
    fn llama_cache_type_spellings_resolve_to_the_format_ggml_type_maps_them_to() {
        let cases = [
            ("f32", NumericFormat::Float32),
            ("f16", NumericFormat::Float16),
            ("bf16", NumericFormat::BFloat16),
            ("q8_0", NumericFormat::Q8_0),
            ("q4_0", NumericFormat::Q4_0),
            ("q4_1", NumericFormat::Q4_1),
            ("q5_0", NumericFormat::Q5_0),
            ("q5_1", NumericFormat::Q5_1),
            ("iq4_nl", NumericFormat::Iq4Nl),
        ];
        for (spelling, expected) in cases {
            assert_eq!(NumericFormat::from_name(spelling), Some(expected), "{spelling}");
        }
    }

    #[test]
    fn aliases_never_come_back_from_name_and_every_pre_existing_name_still_resolves() {
        for (alias, format) in NumericFormat::ALIASES {
            assert_ne!(format.name(), Some(alias), "{alias} must stay an alias");
            assert!(!NumericFormat::NAMES.contains(&alias), "{alias} duplicates a canonical name");
        }
        for (index, name) in NumericFormat::NAMES.iter().enumerate() {
            assert_eq!(NumericFormat::from_name(name), Some(NumericFormat::ALL[index]), "{name}");
        }
        assert_eq!(NumericFormat::from_name("bool"), Some(NumericFormat::Bool));
        assert_eq!(NumericFormat::from_name("float32"), Some(NumericFormat::Float32));
        assert_eq!(NumericFormat::from_name("q9_9"), None);
    }

    #[cfg(feature = "config")]
    #[test]
    fn every_named_format_round_trips_through_serde_to_itself() {
        let named = NumericFormat::ALL
            .iter()
            .chain([NumericFormat::Float32, NumericFormat::Bool].iter());
        for format in named {
            let encoded = serde_json::to_string(format).expect("a named format serializes");
            let decoded: NumericFormat =
                serde_json::from_str(&encoded).expect("its own spelling deserializes");
            assert_eq!(decoded, *format, "{encoded}");
        }
    }

    #[cfg(feature = "config")]
    #[test]
    fn deserializing_a_cache_type_spelling_yields_its_format() {
        let decoded: NumericFormat = serde_json::from_str("\"bf16\"").expect("bf16 is an alias");
        assert_eq!(decoded, NumericFormat::BFloat16);
    }

    #[cfg(feature = "config")]
    #[test]
    fn deserializing_an_unknown_name_fails_naming_the_input() {
        let error = serde_json::from_str::<NumericFormat>("\"q9_9\"").expect_err("q9_9 is not a format");
        assert!(error.to_string().contains("q9_9"), "{error}");
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
    fn every_dtype_constant_has_a_numeric_format_twin_with_identical_methods() {
        let twins = [
            (DType::Bool, NumericFormat::Bool),
            (DType::INT8, NumericFormat::Int8),
            (DType::UINT8, NumericFormat::UInt8),
            (DType::INT16, NumericFormat::Int16),
            (DType::UINT16, NumericFormat::UInt16),
            (DType::INT32, NumericFormat::Int32),
            (DType::UINT32, NumericFormat::UInt32),
            (DType::INT64, NumericFormat::Int64),
            (DType::UINT64, NumericFormat::UInt64),
            (DType::INT128, NumericFormat::Int128),
            (DType::UINT128, NumericFormat::UInt128),
            (DType::BFLOAT16, NumericFormat::BFloat16),
            (DType::BFLOAT8, NumericFormat::Bf8E5M2),
            (DType::FLOAT8_E4M3FN, NumericFormat::Float8E4M3Fn),
            (DType::FLOAT8_E5M2, NumericFormat::BFloat8),
            (DType::FP6_E3_M2, NumericFormat::Fp6E3M2Scalar),
            (DType::FLOAT16, NumericFormat::Float16),
            (DType::FLOAT32, NumericFormat::Float32),
            (DType::FLOAT64, NumericFormat::Float64),
        ];
        assert_eq!(twins.len(), 19, "pinned: every DType constant plus Bool");

        for (dtype, format) in twins {
            assert_eq!(dtype.numeric_format(), format, "{dtype:?}");
            assert_eq!(format.to_dtype(), Some(dtype), "{dtype:?}");
            assert_eq!(format.size_bytes(), dtype.size_bytes(), "{dtype:?} size_bytes");
            assert_eq!(format.bits(), dtype.bits(), "{dtype:?} bits");
            assert_eq!(format.is_float(), dtype.is_float(), "{dtype:?} is_float");
            assert_eq!(
                format.accumulates_in_place(),
                dtype.accumulates_in_place(),
                "{dtype:?} accumulates_in_place"
            );
            assert_eq!(format.is_integer(), dtype.is_integer(), "{dtype:?} is_integer");
            assert_eq!(
                format.is_signed_integer(),
                dtype.is_signed_integer(),
                "{dtype:?} is_signed_integer"
            );
        }
        assert_eq!(NumericFormat::Q4_0.to_dtype(), None);
    }

    #[test]
    fn a_bool_ladder_with_its_code_at_another_bit_is_not_bool() {
        let NumericFormat::Levels(mut levels) = NumericFormat::Bool else {
            panic!("bool is a level format");
        };
        let LevelGeometry::Packed(mut layout) = levels.geometry else {
            panic!("bool is a packed ladder");
        };
        layout.element.low.base_bits = 7;
        levels.geometry = LevelGeometry::Packed(layout);
        let shifted = NumericFormat::Levels(levels);

        assert!(NumericFormat::Bool.is_bool() && !shifted.is_bool());
        assert_eq!(shifted.to_dtype(), None);
        assert_eq!(shifted.name(), None);
        assert_eq!(NumericFormat::Bool.to_dtype(), Some(DType::Bool));
    }

    #[test]
    fn bool_is_a_two_level_unscaled_ladder_that_decodes_through_the_shared_walker() {
        let NumericFormat::Levels(levels) = NumericFormat::Bool else {
            panic!("bool is a level format");
        };
        assert_eq!(
            levels.levels,
            LevelTable::Linear {
                count: 2,
                zero_point: 0,
                step: 1
            }
        );
        assert_eq!(NumericFormat::Bool.bytes_per_block(), 1);
        assert_eq!(NumericFormat::Bool.elements_per_block(), 1);
        assert_eq!(NumericFormat::Bool.scale(), Some(Field::UNIT));
        assert_eq!(NumericFormat::Bool.packed_len(5), Some(5));
        assert_eq!(NumericFormat::from_name("bool"), Some(NumericFormat::Bool));
        assert_eq!(NumericFormat::Bool.name(), Some("bool"));
        assert_eq!(std::format!("{:?}", NumericFormat::Bool), "Bool");

        let mut decoded = [9.0_f32; 4];
        for (index, byte) in [0_u8, 1, 1, 0].iter().enumerate() {
            crate::decode_blocks(
                &NumericFormat::Bool,
                core::slice::from_ref(byte),
                &mut decoded[index..=index],
            )
            .expect("a bool byte decodes as one block");
        }
        assert_eq!(decoded, [0.0, 1.0, 1.0, 0.0]);
        let mut padded = [9.0_f32; 1];
        crate::decode_blocks(&NumericFormat::Bool, &[0b11_u8], &mut padded)
            .expect("a bool byte decodes as one block");
        assert_eq!(padded, [1.0], "the code is bit 0; the seven padding bits are not read");
    }

    #[test]
    fn scalar_twins_keep_their_structural_debug_spelling() {
        for format in [
            NumericFormat::Int8,
            NumericFormat::UInt128,
            NumericFormat::Float64,
            NumericFormat::Float8E4M3Fn,
            NumericFormat::Fp6E3M2Scalar,
        ] {
            assert!(
                std::format!("{format:?}").starts_with("Scalar("),
                "{format:?} must print as it did before it had a constant"
            );
            assert_eq!(format.debug_name(), None);
        }
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
                byte_offset: 0,
                pieces: None
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
