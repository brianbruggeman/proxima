//! A block layout describes how a run of stored numbers is packed: how many elements share one
//! scale (and optional min), where each bit plane of an element's code lives, and how a code
//! becomes a value, `(multiplier * code - bias) * scale + min`.
//!
//! Teaching pointer: [`FloatLayout`] is the data of one stored number; this is the data of a
//! block of them. One CPU walker (`proxima_gguf::quant::block_layout`) and one MSL emitter
//! (`omega::msl::layout_render::block_unpack_msl`) read it, so a new linear block format is a
//! constant here plus its parity fixture, not a decoder.

use crate::float_layout::FloatLayout;

/// How the elements of one plane are addressed in the block's bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum PlaneOrder {
    /// Element `i` occupies bits `i * bits ..` counted from the least significant bit of the
    /// plane's first byte.
    Contiguous,
    /// Four-bit codes, two per byte: elements `0 .. n/2` in the low nibbles of the plane's `n/2`
    /// bytes and elements `n/2 .. n` in the high nibbles (Q4_0, Q4_1, Q5_0, Q5_1).
    NibbleSplit,
}

/// One bit plane of an element code: `bits` wide, starting at `byte_offset` in the block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Plane {
    pub bits: u8,
    pub byte_offset: u16,
    pub order: PlaneOrder,
}

/// A scale or min stored in the block: its layout and byte offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Field {
    pub layout: FloatLayout,
    pub byte_offset: u16,
}

/// The packing of one block of `elements` codes under a shared scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct BlockLayout {
    /// Elements per block; 32 for Q4_0, 128 for Q1_0.
    pub elements: u16,
    /// Bytes per block; 18 for Q4_0.
    pub block_bytes: u16,
    /// Layout of an assembled code: unsigned for offset-binary codes, `TwosComplement` for Q8.
    pub code: FloatLayout,
    /// The low bits of each code.
    pub low: Plane,
    /// The bits above `low.bits`, when the code is split over two planes (Q5_0 and Q5_1).
    pub high: Option<Plane>,
    pub scale: Field,
    pub min: Option<Field>,
    /// Multiplies the code before the bias: 2 for Q1_0, whose value is `(2 * code - 1) * scale`.
    pub multiplier: i16,
    /// Subtracted from the multiplied code: 8 for Q4_0, 16 for Q5_0, 0 for Q8_1 and Q4_1.
    pub bias: i16,
}

const F16_SCALE: Field = Field {
    layout: FloatLayout::F16,
    byte_offset: 0,
};

const fn field_at(byte_offset: u16) -> Field {
    Field {
        layout: FloatLayout::F16,
        byte_offset,
    }
}

const fn plane(bits: u8, byte_offset: u16, order: PlaneOrder) -> Plane {
    Plane {
        bits,
        byte_offset,
        order,
    }
}

impl BlockLayout {
    pub const Q4_0: Self = Self {
        elements: 32,
        block_bytes: 18,
        code: FloatLayout::UINT4,
        low: plane(4, 2, PlaneOrder::NibbleSplit),
        high: None,
        scale: F16_SCALE,
        min: None,
        multiplier: 1,
        bias: 8,
    };
    pub const Q4_1: Self = Self {
        elements: 32,
        block_bytes: 20,
        code: FloatLayout::UINT4,
        low: plane(4, 4, PlaneOrder::NibbleSplit),
        high: None,
        scale: F16_SCALE,
        min: Some(field_at(2)),
        multiplier: 1,
        bias: 0,
    };
    pub const Q5_0: Self = Self {
        elements: 32,
        block_bytes: 22,
        code: FloatLayout::UINT5,
        low: plane(4, 6, PlaneOrder::NibbleSplit),
        high: Some(plane(1, 2, PlaneOrder::Contiguous)),
        scale: F16_SCALE,
        min: None,
        multiplier: 1,
        bias: 16,
    };
    pub const Q5_1: Self = Self {
        elements: 32,
        block_bytes: 24,
        code: FloatLayout::UINT5,
        low: plane(4, 8, PlaneOrder::NibbleSplit),
        high: Some(plane(1, 4, PlaneOrder::Contiguous)),
        scale: F16_SCALE,
        min: Some(field_at(2)),
        multiplier: 1,
        bias: 0,
    };
    pub const Q8_0: Self = Self {
        elements: 32,
        block_bytes: 34,
        code: FloatLayout::INT8,
        low: plane(8, 2, PlaneOrder::Contiguous),
        high: None,
        scale: F16_SCALE,
        min: None,
        multiplier: 1,
        bias: 0,
    };
    pub const Q8_1: Self = Self {
        elements: 32,
        block_bytes: 36,
        code: FloatLayout::INT8,
        low: plane(8, 4, PlaneOrder::Contiguous),
        high: None,
        scale: F16_SCALE,
        min: None,
        multiplier: 1,
        bias: 0,
    };
    pub const Q1_0: Self = Self {
        elements: 128,
        block_bytes: 18,
        code: FloatLayout::UINT1,
        low: plane(1, 2, PlaneOrder::Contiguous),
        high: None,
        scale: F16_SCALE,
        min: None,
        multiplier: 2,
        bias: 1,
    };
    pub const Q2_0: Self = Self {
        elements: 64,
        block_bytes: 18,
        code: FloatLayout::UINT2,
        low: plane(2, 2, PlaneOrder::Contiguous),
        high: None,
        scale: F16_SCALE,
        min: None,
        multiplier: 1,
        bias: 1,
    };
}
