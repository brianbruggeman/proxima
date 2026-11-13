//! Whether a [`NumericFormat`] is served by a row-blocked matvec that reads one whole block (or one
//! scalar) per lane, and what that reader needs to know about it.
//!
//! A [`RowBlockShape::Block`] is a layout whose block is a single unit-scale run of codes: no
//! sub-block scales, no per-run scale copies, whole-byte-addressable code planes, a scale (and an
//! optional min) that sits whole in the block, and 256 a whole number of blocks so a 256-element
//! addressing span is `span_step_bytes` bytes. A [`RowBlockShape::Scalar`] is a float format whose
//! element is a flat whole-byte (or sub-byte divisor) code. `omega`'s Metal row-blocked matvec
//! renders the shape it is given; [`row_block_shape`] is the one place that decides it.
//!
//! Teaching pointer: the sibling is [`crate::int8_dot::int8_dot_shape`], the same posture for the
//! CUDA int8-dot kernel family: a pure, no_std, no-alloc decision from layout fields. No format is
//! named here, so a new block format that fits lowers by being a constant in
//! [`crate::block_layout`], not by editing a whitelist.

use crate::block_layout::{BlockLayout, Field, Plane};
use crate::format::NumericFormat;

const BYTE_BITS: usize = 8;
const SPAN_ELEMENTS: usize = 256;
const MAX_CODE_BITS: u8 = 8;
const MAX_SCALAR_BITS: u8 = 32;
const MAX_SUB_BYTE_SCALAR_BITS: u8 = 16;

/// What a row-blocked reader needs to render one admitted format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowBlockShape {
    /// One unit-scale block per lane step.
    Block {
        /// Elements per block; divides 256.
        elements: u16,
        /// Bytes that 256 elements span: `block_bytes * (256 / elements)`.
        span_step_bytes: u32,
        /// The code has a second (high) plane, as in Q5_0.
        high_plane: bool,
        /// The block carries a whole-block min term, as in Q4_1.
        min: bool,
    },
    /// A scalar float element (F16, BF16): read through the format's own load, no scale.
    Scalar,
}

/// The row-block shape of `format`, or `None` when its layout needs a reader this family does not
/// render: sub-block or run scales, a code plane that straddles a byte or leaves the block, a
/// scale or min that is not whole in the block, a mixed-radix code, a block size that does not
/// divide 256, a level table that is not the layout's own ladder, or a scalar whose width is not
/// a flat load.
#[must_use]
pub fn row_block_shape(format: &NumericFormat) -> Option<RowBlockShape> {
    match format {
        NumericFormat::Scalar(layout) => {
            let width = layout.width();
            let flat = width != 0
                && width <= MAX_SCALAR_BITS
                && (width.is_multiple_of(8) || width <= MAX_SUB_BYTE_SCALAR_BITS);
            flat.then_some(RowBlockShape::Scalar)
        }
        NumericFormat::Block(_) | NumericFormat::Levels(_) => format.block_layout().as_ref().and_then(block_shape),
    }
}

fn block_shape(layout: &BlockLayout) -> Option<RowBlockShape> {
    let elements = usize::from(layout.elements);
    let unit_scale = layout.sub.is_none() && layout.scale_runs.is_none();
    let divides_span = elements > 0 && SPAN_ELEMENTS.is_multiple_of(elements);
    let fields_whole = field_is_whole(&layout.scale, layout.block_bytes) && layout.min.as_ref().is_none_or(|min| field_is_whole(min, layout.block_bytes));
    if !(unit_scale && divides_span && fields_whole && codes_addressable(layout)) {
        return None;
    }
    Some(RowBlockShape::Block {
        elements: layout.elements,
        span_step_bytes: u32::from(layout.block_bytes) * u32::try_from(SPAN_ELEMENTS / elements).ok()?,
        high_plane: layout.element.high.is_some(),
        min: layout.min.is_some(),
    })
}

fn field_is_whole(field: &Field, block_bytes: u16) -> bool {
    let bytes = usize::from(field.layout.width()).div_ceil(BYTE_BITS);
    field.pieces.is_none() && usize::from(field.byte_offset) + bytes <= usize::from(block_bytes)
}

fn codes_addressable(layout: &BlockLayout) -> bool {
    let element = &layout.element;
    let low_bits = element.low.bits;
    let high_bits = element.high.as_ref().map_or(0, |plane| plane.bits);
    let widths_fit = element.packed_radix == 0
        && (1..=MAX_CODE_BITS).contains(&low_bits)
        && element.high.as_ref().is_none_or(|plane| plane.bits >= 1)
        && u16::from(low_bits) + u16::from(high_bits) <= u16::from(MAX_CODE_BITS);
    let block_bytes = usize::from(layout.block_bytes);
    let elements = usize::from(layout.elements);
    widths_fit && plane_in_bytes(&element.low, elements, block_bytes) && element.high.as_ref().is_none_or(|plane| plane_in_bytes(plane, elements, block_bytes))
}

fn plane_in_bytes(plane: &Plane, elements: usize, block_bytes: usize) -> bool {
    (0..elements).all(|index| {
        let address = plane.address(index);
        address % BYTE_BITS + usize::from(plane.bits) <= BYTE_BITS && address / BYTE_BITS < block_bytes
    })
}

#[cfg(test)]
mod tests {
    use super::{RowBlockShape, row_block_shape};
    use crate::decode::{decode_block, decode_blocks};
    use crate::format::NumericFormat;

    const fn block(elements: u16, span_step_bytes: u32, high_plane: bool, min: bool) -> Option<RowBlockShape> {
        Some(RowBlockShape::Block { elements, span_step_bytes, high_plane, min })
    }

    const SCALAR: Option<RowBlockShape> = Some(RowBlockShape::Scalar);

    /// Every format `NumericFormat::ALL` names, in `ALL` order, with the answer and the layout
    /// field that decides it.
    fn refusal_and_admission_table() -> [(&'static str, Option<RowBlockShape>, &'static str); 36] {
        [
            ("Q2K", None, "sub is some: 16 sub-blocks"),
            ("Q3K", None, "sub is some: 16 sub-blocks"),
            ("Q4K", None, "sub is some: 8 sub-blocks with min"),
            ("Q5K", None, "sub is some: 8 sub-blocks with min"),
            ("Q6K", None, "sub is some: 16 sub-blocks"),
            ("Q8_0", block(32, 272, false, false), "32 elements, 34 bytes: 8 blocks span 272"),
            ("Q4_0", block(32, 144, false, false), "32 elements, 18 bytes: nibble plane in bytes 2..17"),
            ("Float16", SCALAR, "scalar width 16 is a flat load"),
            ("BFloat16", SCALAR, "scalar width 16 is a flat load"),
            ("Q5_1", block(32, 192, true, true), "high bit run plus whole-block min at byte 2"),
            ("Q5_0", block(32, 176, true, false), "high bit run, no min"),
            ("Q4_1", block(32, 160, false, true), "whole-block min at byte 2"),
            ("Q8_1", block(32, 288, false, false), "int8 run at byte 4; the stored sum is not read"),
            ("Q8K", block(256, 292, false, false), "256 elements, bf32 scale, int8 run at byte 4"),
            ("Iq1S", None, "level table is a grid, not the layout's ladder"),
            ("Iq1M", None, "level table is a grid, not the layout's ladder"),
            ("Iq2Xxs", None, "level table is a signed grid, not the layout's ladder"),
            ("Iq2Xs", None, "level table is a signed grid, not the layout's ladder"),
            ("Iq2S", None, "level table is a signed grid, not the layout's ladder"),
            ("Iq3Xxs", None, "level table is a signed grid, not the layout's ladder"),
            ("Iq3S", None, "level table is a signed grid, not the layout's ladder"),
            ("Iq4Nl", None, "explicit 16-entry level table is not the layout's ladder"),
            ("Iq4Xs", None, "explicit level table and sub is some"),
            ("Tq10", None, "packed_radix is 3: mixed-radix code"),
            ("Tq20", block(256, 66, false, false), "256 elements, 2-bit plane in bytes 0..63, scale at byte 64"),
            ("Mxfp4", block(32, 136, false, false), "one-byte e8m0 scale, nibble plane from byte 1"),
            ("Nvfp4", None, "scale_runs is some: four e4m3 scales"),
            ("Q1_0", block(128, 36, false, false), "128 elements: two blocks span 256"),
            ("Q2_0", block(64, 72, false, false), "64 elements: four blocks span 256"),
            ("BFloat8", SCALAR, "scalar width 8 is a flat load"),
            ("Bf8E5M2", SCALAR, "scalar width 8 is a flat load"),
            ("Bf4E2M1", SCALAR, "scalar width 4 divides a byte"),
            ("Fp6E3M2", None, "6-bit plane straddles a byte at element 1"),
            ("Pq2_0", block(128, 68, false, false), "128 elements, 2-bit plane in bytes 2..33"),
            ("Ptq1_0", None, "packed_radix is 3: mixed-radix code"),
            ("Q8_1Grouped", None, "scale_runs is some: four run scales"),
        ]
    }

    #[test]
    fn every_named_format_has_one_decided_row_in_all_order() {
        let table = refusal_and_admission_table();
        assert_eq!(table.len(), NumericFormat::ALL.len(), "one row per format in NumericFormat::ALL");
        assert_eq!(table.len(), 36);
        for (index, (name, expected, reason)) in table.iter().enumerate() {
            assert_eq!(*name, NumericFormat::DEBUG_NAMES[index], "row {index} names ALL[{index}]");
            assert_eq!(row_block_shape(&NumericFormat::ALL[index]), *expected, "{name}: {reason}");
        }
    }

    #[test]
    fn the_table_admits_seventeen_formats_and_refuses_nineteen() {
        let admitted = NumericFormat::ALL.iter().filter(|format| row_block_shape(format).is_some()).count();
        assert_eq!((admitted, NumericFormat::ALL.len() - admitted), (17, 19));
    }

    /// `row_block_extent_multiple` (omega `emit_and_classify.rs`) admits these today. Q8_0 and Q4_0
    /// get `Some`. The K formats have sub-blocks, so design rule 1 refuses them here and they stay
    /// on their rendered sub-block bodies; Fp6E3M2's 6-bit plane straddles bytes (rule 3), so it
    /// stays on the legacy super-block arm; Iq4Xs is a level format with sub-blocks and an explicit
    /// table, rendered by its own level body. PROXNUM-427 section 3.1, `legacy_super_block_arm`.
    #[test]
    fn todays_admitted_codecs_are_some_or_named_legacy_arms() {
        let admitted_and_served_here = [NumericFormat::Q8_0, NumericFormat::Q4_0];
        let admitted_on_legacy_arms = [
            (NumericFormat::Q3K, "sub"),
            (NumericFormat::Q4K, "sub"),
            (NumericFormat::Q5K, "sub"),
            (NumericFormat::Q6K, "sub"),
            (NumericFormat::Fp6E3M2, "straddle"),
            (NumericFormat::Iq4Xs, "level"),
        ];
        for format in admitted_and_served_here {
            assert!(row_block_shape(&format).is_some(), "{format:?} is admitted today and meets every rule");
        }
        for (format, reason) in admitted_on_legacy_arms {
            assert_eq!(row_block_shape(&format), None, "{format:?} stays on its own arm: {reason}");
        }
        for format in [NumericFormat::Q3K, NumericFormat::Q4K, NumericFormat::Q5K, NumericFormat::Q6K] {
            let layout = format.block_layout().expect("a K format is a block layout");
            assert!(layout.sub.is_some(), "{format:?} is refused by rule 1 because it has sub-blocks");
        }
    }

    /// PROXNUM-427 section 4: one Q5_0 block, scale 0.0625 (f16 0x2C00), `qh` 0x80098019, and
    /// `qs[j] = (j << 4) | (15 - j)`; each value is `(code - 16) * 0.0625`.
    #[test]
    fn q5_0_worked_example_block_decodes_to_the_designs_32_values() {
        let block_bytes: [u8; 22] = [
            0x00, 0x2C, 0x19, 0x80, 0x09, 0x80, 0x0F, 0x1E, 0x2D, 0x3C, 0x4B, 0x5A, 0x69, 0x78, 0x87, 0x96, 0xA5, 0xB4, 0xC3, 0xD2, 0xE1, 0xF0,
        ];
        let expected: [f32; 32] = [
            0.9375, -0.125, -0.1875, 0.75, 0.6875, -0.375, -0.4375, -0.5, -0.5625, -0.625, -0.6875, -0.75, -0.8125, -0.875, -0.9375, 0.0, 0.0, -0.9375,
            -0.875, 0.1875, -0.75, -0.6875, -0.625, -0.5625, -0.5, -0.4375, -0.375, -0.3125, -0.25, -0.1875, -0.125, 0.9375,
        ];
        let mut decoded = [0.0f32; 32];
        decode_block(&NumericFormat::Q5_0, &block_bytes, &mut decoded).expect("22 bytes is one Q5_0 block");
        assert_eq!(decoded, expected);
    }

    /// PROXNUM-427 section 9.4: eight little-endian halves covering 1, -1, 0.5, the smallest
    /// subnormal, the largest normal, -4, 0.333251953125 and the smallest normal.
    #[test]
    fn f16_worked_example_bytes_decode_to_the_designs_8_values() {
        let bytes: [u8; 16] = [0x00, 0x3C, 0x00, 0xBC, 0x00, 0x38, 0x01, 0x00, 0xFF, 0x7B, 0x00, 0xC4, 0x55, 0x35, 0x00, 0x04];
        let expected: [f32; 8] = [1.0, -1.0, 0.5, f32::from_bits(0x3380_0000), 65504.0, -4.0, f32::from_bits(0x3EAA_A000), f32::from_bits(0x3880_0000)];
        let mut decoded = [0.0f32; 8];
        let count = decode_blocks(&NumericFormat::Float16, &bytes, &mut decoded).expect("16 bytes are eight f16");
        assert_eq!(count, 8);
        assert_eq!(decoded, expected);
    }

    /// the f32 bit constants in the f16 worked example are the widening of halves 0x3555 and 0x0400.
    #[test]
    fn f16_worked_example_f32_constants_are_the_widened_halves() {
        assert_eq!(half::f16::from_bits(0x3555).to_f32().to_bits(), 0x3EAA_A000);
        assert_eq!(half::f16::from_bits(0x0400).to_f32().to_bits(), 0x3880_0000);
    }
}
