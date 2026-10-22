//! A block layout describes how a run of stored numbers is packed: how many elements share one
//! scale (and optional min), where each bit plane of an element's code lives, how sub-blocks of
//! the block carry their own scale and min, and how a code becomes a value,
//! `(multiplier * code - bias) * scale * sub_scale + min_multiplier * (min * sub_min)`.
//!
//! Teaching pointer: [`FloatLayout`] is the data of one stored number; this is the data of a
//! block of them. One CPU walker (`proxima_gguf::quant::block_layout`) and one MSL emitter
//! (`omega::msl::layout_render::block_unpack_msl`) read it, so a new block format is a constant
//! here plus its parity fixture, not a decoder. Where the bits of a code live is an affine map
//! from the element index: the index is split into mixed-radix digits and each digit moves the
//! bit address by a stride, which expresses contiguous runs, nibble-split pairs, and the
//! interleaved lanes of the K formats with one rule.

use crate::float_layout::FloatLayout;

/// One digit of an element index: `radix` values, each moving the bit address by `stride_bits`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Digit {
    pub radix: u16,
    pub stride_bits: u16,
}

/// The most digits a plane's address map uses (Q3_K needs four).
pub const MAX_DIGITS: usize = 4;

/// Maximum disjoint element ranges needed by current packed block layouts.
pub const MAX_CODE_REGIONS: usize = 3;

/// One bit plane of a code: `bits` wide, at the bit address
/// `base_bits + sum(digit_value * stride_bits)`, with the index split least significant digit
/// first. Unused digits have radix 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Plane {
    pub bits: u8,
    pub base_bits: u32,
    pub digits: [Digit; MAX_DIGITS],
}

impl Plane {
    /// The bit address of the plane's value for element `index`.
    #[must_use]
    pub const fn address(&self, index: usize) -> usize {
        let mut remaining = index;
        let mut address = self.base_bits as usize;
        let mut position = 0;
        while position < MAX_DIGITS {
            let digit = self.digits[position];
            let radix = digit.radix as usize;
            address += (remaining % radix) * digit.stride_bits as usize;
            remaining /= radix;
            position += 1;
        }
        address
    }
}

/// A code assembled from one or two planes (the second holds the bits above the first).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Coded {
    /// Layout of the assembled code: unsigned for offset-binary codes, `TwosComplement` for i8.
    pub code: FloatLayout,
    pub low: Plane,
    pub high: Option<Plane>,
    /// Subtracted from the multiplied code.
    pub bias: i16,
    /// Zero uses the low/high bit planes directly; a value above one extracts a mixed-radix
    /// digit from each region's packed word before applying the shared code layout.
    pub packed_radix: u8,
    /// Source maps for code ranges whose packed word stores multiple radix digits.
    pub regions: [CodeRegion; MAX_CODE_REGIONS],
    /// Number of active source maps in `regions`.
    pub region_count: u8,
}

/// A logical element range that reads one packed word and selects one mixed-radix digit from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct CodeRegion {
    pub start: u16,
    pub elements: u16,
    pub source: Plane,
    pub digit_axis: u8,
}

/// A scale or min stored once per block: its layout and byte offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Field {
    pub layout: FloatLayout,
    pub byte_offset: u16,
}

/// Sub-blocks of the block, each with its own scale code and optionally a min code, indexed by
/// `element / (elements / count)` and addressed by the planes of the codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SubBlocks {
    pub count: u16,
    pub scale: Coded,
    pub min: Option<Coded>,
}

/// The packing of one block of `elements` codes under a shared scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct BlockLayout {
    /// Elements per block; 32 for Q4_0, 256 for the K formats.
    pub elements: u16,
    /// Bytes per block; 18 for Q4_0, 144 for Q4_K.
    pub block_bytes: u16,
    /// The element code; its `bias` is the offset-binary bias (8 for Q4_0, 32 for Q6_K).
    pub element: Coded,
    /// Multiplies the code before the bias: 2 for Q1_0, whose value is `(2 * code - 1) * scale`.
    pub multiplier: i16,
    pub scale: Field,
    pub min: Option<Field>,
    /// Applied to the min term: 1 where the min is added (Q4_1), -1 where it is subtracted (Q4_K).
    pub min_multiplier: i8,
    pub sub: Option<SubBlocks>,
}

const F16_SCALE: Field = field_at(FloatLayout::F16, 0);

const fn field_at(layout: FloatLayout, byte_offset: u16) -> Field {
    Field {
        layout,
        byte_offset,
    }
}

const NO_DIGIT: Digit = Digit {
    radix: 1,
    stride_bits: 0,
};

const EMPTY_PLANE: Plane = plane(1, 0, [NO_DIGIT; MAX_DIGITS]);
const EMPTY_CODE_REGION: CodeRegion = CodeRegion {
    start: 0,
    elements: 0,
    source: EMPTY_PLANE,
    digit_axis: 0,
};

const fn digit(radix: u16, stride_bits: u16) -> Digit {
    Digit { radix, stride_bits }
}

const fn plane(bits: u8, base_bits: u32, digits: [Digit; MAX_DIGITS]) -> Plane {
    Plane {
        bits,
        base_bits,
        digits,
    }
}

const fn coded(code: FloatLayout, low: Plane, high: Option<Plane>, bias: i16) -> Coded {
    Coded {
        code,
        low,
        high,
        bias,
        packed_radix: 0,
        regions: [EMPTY_CODE_REGION; MAX_CODE_REGIONS],
        region_count: 0,
    }
}

const fn radix_coded(
    code: FloatLayout,
    radix: u8,
    regions: [CodeRegion; MAX_CODE_REGIONS],
    region_count: u8,
    bias: i16,
) -> Coded {
    Coded {
        code,
        low: EMPTY_PLANE,
        high: None,
        bias,
        packed_radix: radix,
        regions,
        region_count,
    }
}

const fn code_region(start: u16, elements: u16, source: Plane, digit_axis: u8) -> CodeRegion {
    CodeRegion {
        start,
        elements,
        source,
        digit_axis,
    }
}

const fn nibbles(base_bits: u32, span: u16) -> Plane {
    plane(
        4,
        base_bits,
        [digit(span, 8), digit(2, 4), NO_DIGIT, NO_DIGIT],
    )
}

const fn run(bits: u8, base_bits: u32, elements: u16) -> Plane {
    plane(
        bits,
        base_bits,
        [digit(elements, bits as u16), NO_DIGIT, NO_DIGIT, NO_DIGIT],
    )
}

const fn linear(
    elements: u16,
    block_bytes: u16,
    element: Coded,
    multiplier: i16,
    min: Option<Field>,
) -> BlockLayout {
    BlockLayout {
        elements,
        block_bytes,
        element,
        multiplier,
        scale: F16_SCALE,
        min,
        min_multiplier: 1,
        sub: None,
    }
}

const QK_QS_PAIR_LANES: [Digit; MAX_DIGITS] = [digit(32, 8), digit(2, 4), digit(4, 256), NO_DIGIT];

const K4_SCALE_BASE: u32 = 32;

const fn k4_sub_blocks() -> SubBlocks {
    let scale = coded(
        FloatLayout::UINT6,
        plane(
            4,
            K4_SCALE_BASE,
            [digit(4, 8), digit(2, 64), NO_DIGIT, NO_DIGIT],
        ),
        Some(plane(
            2,
            K4_SCALE_BASE + 4,
            [digit(4, 8), digit(2, 2), NO_DIGIT, NO_DIGIT],
        )),
        0,
    );
    let min = coded(
        FloatLayout::UINT6,
        plane(
            4,
            K4_SCALE_BASE + 32,
            [digit(4, 8), digit(2, 36), NO_DIGIT, NO_DIGIT],
        ),
        Some(plane(
            2,
            K4_SCALE_BASE + 36,
            [digit(4, 8), digit(2, 2), NO_DIGIT, NO_DIGIT],
        )),
        0,
    );
    SubBlocks {
        count: 8,
        scale,
        min: Some(min),
    }
}

impl BlockLayout {
    pub const Q4_0: Self = linear(
        32,
        18,
        coded(FloatLayout::UINT4, nibbles(16, 16), None, 8),
        1,
        None,
    );
    pub const Q4_1: Self = linear(
        32,
        20,
        coded(FloatLayout::UINT4, nibbles(32, 16), None, 0),
        1,
        Some(field_at(FloatLayout::F16, 2)),
    );
    pub const Q5_0: Self = linear(
        32,
        22,
        coded(
            FloatLayout::UINT5,
            nibbles(48, 16),
            Some(run(1, 16, 32)),
            16,
        ),
        1,
        None,
    );
    pub const Q5_1: Self = linear(
        32,
        24,
        coded(FloatLayout::UINT5, nibbles(64, 16), Some(run(1, 32, 32)), 0),
        1,
        Some(field_at(FloatLayout::F16, 2)),
    );
    pub const Q8_0: Self = linear(
        32,
        34,
        coded(FloatLayout::INT8, run(8, 16, 32), None, 0),
        1,
        None,
    );
    pub const Q8_1: Self = linear(
        32,
        36,
        coded(FloatLayout::INT8, run(8, 32, 32), None, 0),
        1,
        None,
    );
    pub const Q1_0: Self = linear(
        128,
        18,
        coded(FloatLayout::UINT1, run(1, 16, 128), None, 1),
        2,
        None,
    );
    pub const Q2_0: Self = linear(
        64,
        18,
        coded(FloatLayout::UINT2, run(2, 16, 64), None, 1),
        1,
        None,
    );
    pub const TQ2_0: Self = Self {
        elements: 256,
        block_bytes: 66,
        element: coded(
            FloatLayout::UINT2,
            plane(2, 0, [digit(32, 8), digit(4, 2), digit(2, 256), NO_DIGIT]),
            None,
            1,
        ),
        multiplier: 1,
        scale: field_at(FloatLayout::F16, 64),
        min: None,
        min_multiplier: 1,
        sub: None,
    };
    pub const TQ1_0: Self = Self {
        elements: 256,
        block_bytes: 54,
        element: radix_coded(
            FloatLayout::UINT2,
            3,
            [
                code_region(
                    0,
                    160,
                    plane(8, 0, [digit(32, 8), digit(5, 0), NO_DIGIT, NO_DIGIT]),
                    1,
                ),
                code_region(
                    160,
                    80,
                    plane(8, 256, [digit(16, 8), digit(5, 0), NO_DIGIT, NO_DIGIT]),
                    1,
                ),
                code_region(
                    240,
                    16,
                    plane(8, 384, [digit(4, 8), digit(4, 0), NO_DIGIT, NO_DIGIT]),
                    1,
                ),
            ],
            3,
            1,
        ),
        multiplier: 1,
        scale: field_at(FloatLayout::F16, 52),
        min: None,
        min_multiplier: 1,
        sub: None,
    };

    pub const Q2_K: Self = Self {
        elements: 256,
        block_bytes: 84,
        element: coded(
            FloatLayout::UINT2,
            plane(
                2,
                128,
                [digit(16, 8), digit(2, 128), digit(4, 2), digit(2, 256)],
            ),
            None,
            0,
        ),
        multiplier: 1,
        scale: field_at(FloatLayout::F16, 80),
        min: Some(field_at(FloatLayout::F16, 82)),
        min_multiplier: -1,
        sub: Some(SubBlocks {
            count: 16,
            scale: coded(
                FloatLayout::UINT4,
                plane(4, 0, [digit(16, 8), NO_DIGIT, NO_DIGIT, NO_DIGIT]),
                None,
                0,
            ),
            min: Some(coded(
                FloatLayout::UINT4,
                plane(4, 4, [digit(16, 8), NO_DIGIT, NO_DIGIT, NO_DIGIT]),
                None,
                0,
            )),
        }),
    };
    pub const Q3_K: Self = Self {
        elements: 256,
        block_bytes: 110,
        element: coded(
            FloatLayout::UINT3,
            plane(
                2,
                256,
                [digit(16, 8), digit(2, 128), digit(4, 2), digit(2, 256)],
            ),
            Some(plane(
                1,
                0,
                [digit(16, 8), digit(2, 128), digit(4, 1), digit(2, 4)],
            )),
            4,
        ),
        multiplier: 1,
        scale: field_at(FloatLayout::F16, 108),
        min: None,
        min_multiplier: 1,
        sub: Some(SubBlocks {
            count: 16,
            scale: coded(
                FloatLayout::UINT6,
                plane(4, 768, [digit(4, 8), digit(2, 32), digit(2, 4), NO_DIGIT]),
                Some(plane(
                    2,
                    832,
                    [digit(4, 8), digit(4, 2), NO_DIGIT, NO_DIGIT],
                )),
                32,
            ),
            min: None,
        }),
    };
    pub const Q4_K: Self = Self {
        elements: 256,
        block_bytes: 144,
        element: coded(FloatLayout::UINT4, plane(4, 128, QK_QS_PAIR_LANES), None, 0),
        multiplier: 1,
        scale: F16_SCALE,
        min: Some(field_at(FloatLayout::F16, 2)),
        min_multiplier: -1,
        sub: Some(k4_sub_blocks()),
    };
    pub const Q5_K: Self = Self {
        elements: 256,
        block_bytes: 176,
        element: coded(
            FloatLayout::UINT5,
            plane(4, 384, QK_QS_PAIR_LANES),
            Some(plane(
                1,
                128,
                [digit(32, 8), digit(2, 1), digit(4, 2), NO_DIGIT],
            )),
            0,
        ),
        multiplier: 1,
        scale: F16_SCALE,
        min: Some(field_at(FloatLayout::F16, 2)),
        min_multiplier: -1,
        sub: Some(k4_sub_blocks()),
    };
    pub const Q6_K: Self = Self {
        elements: 256,
        block_bytes: 210,
        element: coded(
            FloatLayout::UINT6,
            plane(
                4,
                0,
                [digit(32, 8), digit(2, 256), digit(2, 4), digit(2, 512)],
            ),
            Some(plane(
                2,
                1024,
                [digit(32, 8), digit(4, 2), digit(2, 256), NO_DIGIT],
            )),
            32,
        ),
        multiplier: 1,
        scale: field_at(FloatLayout::F16, 208),
        min: None,
        min_multiplier: 1,
        sub: Some(SubBlocks {
            count: 16,
            scale: coded(
                FloatLayout::INT8,
                plane(8, 1536, [digit(16, 8), NO_DIGIT, NO_DIGIT, NO_DIGIT]),
                None,
                0,
            ),
            min: None,
        }),
    };
    pub const Q8_K: Self = Self {
        elements: 256,
        block_bytes: 292,
        element: coded(FloatLayout::INT8, run(8, 32, 256), None, 0),
        multiplier: 1,
        scale: field_at(FloatLayout::BF32, 0),
        min: None,
        min_multiplier: 1,
        sub: None,
    };
}

/// Walks a plane's addresses for consecutive element indexes without dividing: the digits are an
/// odometer, so each step adds one stride and carries.
#[derive(Debug, Clone, Copy)]
pub struct PlaneCursor {
    plane: Plane,
    counters: [u16; MAX_DIGITS],
    address: usize,
}

impl Plane {
    /// A cursor positioned at element 0.
    #[must_use]
    pub const fn cursor(&self) -> PlaneCursor {
        PlaneCursor {
            plane: *self,
            counters: [0; MAX_DIGITS],
            address: self.base_bits as usize,
        }
    }
}

impl PlaneCursor {
    /// The bit address of the current element.
    #[must_use]
    pub const fn address(&self) -> usize {
        self.address
    }

    /// Moves to the next element index.
    pub const fn advance(&mut self) {
        let mut position = 0;
        while position < MAX_DIGITS {
            let digit = self.plane.digits[position];
            let next = self.counters[position] + 1;
            if next < digit.radix {
                self.counters[position] = next;
                self.address += digit.stride_bits as usize;
                return;
            }
            self.counters[position] = 0;
            self.address -= (digit.radix as usize - 1) * digit.stride_bits as usize;
            position += 1;
        }
    }
}
