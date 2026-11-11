//! A simple block format resolved once into the two slice conversions that carry it.
//!
//! Teaching pointer: [`crate::scalar_codec::ScalarCodec`] does this for one stored number per
//! element; this does it for a block of them. `encode_run` and `decode_run` are the two loops,
//! written once over a [`BlockLayout`]. [`BlockCodec::resolve`] instantiates them once per row of
//! the block table ([`crate::block_layout::named_block_table`]) with the layout as a literal
//! constant, so the element count, byte count, every plane address, the scale and min offsets, the
//! multiplier and the bias are folded from the layout's fields. A layout that is not in the table
//! (the K-quant, grouped and codebook families, or a layout with no name) has no instantiation and
//! keeps [`crate::encode::encode_blocks`] and [`crate::decode::decode_blocks`].
//!
//! How a block's scale is fitted is read from the layout too ([`fit_of`]): an offset-binary code
//! centred on its bias is fitted to the signed value of largest magnitude (`Q4_0`, `Q5_0`), a
//! signed or off-centre integer code to the peak magnitude (`Q8_0`, `Q2_0`), a one-bit code with a
//! multiplier of two to the mean magnitude (`Q1_0`). These are the rules of the incumbents in
//! `proxima-gguf`, reproduced operation for operation so the bytes are identical; a layout whose
//! fit is not one of them (a stored min, free bytes beside the scale, a float code) has a decode
//! instantiation and no encode instantiation.

use core::mem::MaybeUninit;

use crate::block_layout::{BlockLayout, Field, MAX_DIGITS, Plane, named_block_table};
use crate::error::NumericError;
use crate::float_layout::SignEncoding;
use crate::float_layout_codec::{decode, encode};
use crate::format::{LevelTable, NumericFormat};

/// An encode instantiation: whole blocks of values into whole blocks of bytes.
pub type EncodeFn = fn(&[f32], &mut [u8]);

/// A decode instantiation: whole blocks of bytes into one slot per element.
pub type DecodeFn = fn(&[u8], &mut [MaybeUninit<f32>]);

/// How a layout's block scale is fitted to its values, decided from the layout alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fit {
    /// `scale = signed_max / -bias`, `code = min(top, trunc(value / scale + bias + 0.5))`.
    SignedMax,
    /// `scale = peak / reach`, `code = clamp(round(value / scale)) + bias`.
    PeakRound,
    /// `scale = mean(|value|)`, `code = value >= 0`.
    MeanSign,
    /// No instantiation: the runtime encoder keeps this layout.
    Runtime,
}

const fn code_bits(layout: &BlockLayout) -> u32 {
    layout.element.low.bits as u32
        + match layout.element.high {
            Some(plane) => plane.bits as u32,
            None => 0,
        }
}

const fn is_unsigned_integer(layout: &BlockLayout) -> bool {
    layout.element.code.exponent_bits == 0
        && matches!(layout.element.code.sign, SignEncoding::Absent)
}

const fn is_twos_complement_integer(layout: &BlockLayout) -> bool {
    layout.element.code.exponent_bits == 0
        && matches!(layout.element.code.sign, SignEncoding::TwosComplement)
}

/// A one-bit code worth `+1` or `-1`: the value is the scale or its negation, and a negation (unlike
/// a multiply by `-1.0`) is defined for a NaN scale, which is the incumbent's behaviour.
const fn is_sign_code(layout: &BlockLayout) -> bool {
    layout.multiplier == 2
        && layout.element.bias == 1
        && code_bits(layout) == 1
        && layout.min.is_none()
        && is_unsigned_integer(layout)
}

/// Bytes the scale, the min and the codes account for; a layout with bytes beyond that (`Q8_1`'s
/// stored sum) carries data the fits here do not write.
const fn accounted_bytes(layout: &BlockLayout) -> usize {
    let scale = layout.scale.layout.width().div_ceil(8) as usize;
    let codes = (layout.elements as usize * code_bits(layout) as usize).div_ceil(8);
    scale + codes
}

/// The range of `value` codes (before the bias) an integer layout stores.
const fn integer_bounds(layout: &BlockLayout) -> (i32, i32) {
    let bits = code_bits(layout);
    if is_twos_complement_integer(layout) {
        (-(1i32 << (bits - 1)), (1i32 << (bits - 1)) - 1)
    } else {
        let bias = layout.element.bias as i32;
        (-bias, (1i32 << bits) - 1 - bias)
    }
}

const fn reach_of(layout: &BlockLayout) -> i32 {
    let (lowest, highest) = integer_bounds(layout);
    if highest < -lowest { highest } else { -lowest }
}

const fn fit_of(layout: &BlockLayout) -> Fit {
    let integer = is_unsigned_integer(layout) || is_twos_complement_integer(layout);
    if !layout.is_simple()
        || layout.min.is_some()
        || !integer
        || layout.scale.layout.exponent_bits == 0
        || layout.scale.byte_offset as usize + layout.scale.layout.width().div_ceil(8) as usize
            > layout.block_bytes as usize
        || accounted_bytes(layout) != layout.block_bytes as usize
        || code_bits(layout) > 8
    {
        return Fit::Runtime;
    }
    let bits = code_bits(layout);
    let bias = layout.element.bias as i32;
    if is_sign_code(layout) {
        Fit::MeanSign
    } else if layout.multiplier != 1 {
        Fit::Runtime
    } else if is_unsigned_integer(layout) && bias == 1i32 << (bits - 1) {
        Fit::SignedMax
    } else if reach_of(layout) > 0 && (is_twos_complement_integer(layout) || bias > 0) {
        Fit::PeakRound
    } else {
        Fit::Runtime
    }
}

/// The bits `[address, address + width)` of `block`, low bit first.
#[inline(always)]
fn read_bits<const BYTES: usize>(block: &[u8; BYTES], address: usize, width: u32) -> u32 {
    let first = address / 8;
    let shift = (address % 8) as u32;
    let span = (shift + width).div_ceil(8) as usize;
    let mut word = 0u64;
    for step in 0..span {
        word |= u64::from(block[first + step]) << (8 * step);
    }
    ((word >> shift) as u32) & (u32::MAX >> (32 - width))
}

/// Ors `bits` into `block` at `[address, address + width)`; the block starts zeroed.
#[inline(always)]
fn put_bits<const BYTES: usize>(block: &mut [u8; BYTES], address: usize, width: u32, bits: u32) {
    let first = address / 8;
    let shift = (address % 8) as u32;
    let span = (shift + width).div_ceil(8) as usize;
    let word = u64::from(bits & (u32::MAX >> (32 - width))) << shift;
    for step in 0..span {
        block[first + step] |= (word >> (8 * step)) as u8;
    }
}

/// [`read_bits`] for a plane of at most eight bits, which spans at most two bytes, so the word is
/// 32 bits wide and a vector lane does not need 64.
#[inline(always)]
fn read_code_bits<const BYTES: usize>(block: &[u8; BYTES], address: usize, width: u32) -> u32 {
    let first = address / 8;
    let shift = (address % 8) as u32;
    let span = (shift + width).div_ceil(8) as usize;
    let mut word = 0u32;
    for step in 0..span {
        word |= u32::from(block[first + step]) << (8 * step);
    }
    (word >> shift) & (u32::MAX >> (32 - width))
}

/// [`put_bits`] for a plane of at most eight bits.
#[inline(always)]
fn put_code_bits<const BYTES: usize>(block: &mut [u8; BYTES], address: usize, width: u32, bits: u32) {
    let first = address / 8;
    let shift = (address % 8) as u32;
    let span = (shift + width).div_ceil(8) as usize;
    let word = (bits & (u32::MAX >> (32 - width))) << shift;
    for step in 0..span {
        block[first + step] |= (word >> (8 * step)) as u8;
    }
}

#[inline(always)]
fn read_field<const BYTES: usize>(block: &[u8; BYTES], field: Field) -> f32 {
    let width = field.layout.width() as u32;
    decode(read_bits(block, field.byte_offset as usize * 8, width), field.layout)
}

/// The elements a plane's digits enumerate: the product of their radices.
const fn plane_span(plane: &Plane) -> usize {
    let mut span = 1;
    let mut position = 0;
    while position < MAX_DIGITS {
        span *= plane.digits[position].radix as usize;
        position += 1;
    }
    span
}

/// Whether every plane of the code enumerates exactly the block's elements, which is what lets
/// [`walk_plane`] visit each element once, in index order.
const fn planes_cover(layout: &BlockLayout) -> bool {
    let elements = layout.elements as usize;
    plane_span(&layout.element.low) == elements
        && match layout.element.high {
            Some(plane) => plane_span(&plane) == elements,
            None => true,
        }
}

/// Visits `(index, bit_address)` for every element in index order, as nested loops over the plane's
/// digits: the highest digit is the outermost loop and the lowest the innermost. Each outer step
/// adds a constant to the address and the innermost loop advances it by one constant stride, so for
/// a layout that is a literal the loop is a strided run over the block's bytes, which the optimiser
/// vectorises; no division or modulus by a radix is left. [`Plane::address`] is the same map taken
/// one index at a time.
#[inline(always)]
fn walk_plane(plane: &Plane, mut visit: impl FnMut(usize, usize)) {
    let [lowest, second, third, highest] = plane.digits;
    let mut index = 0;
    for outer in 0..highest.radix as usize {
        let outer_address = plane.base_bits as usize + outer * highest.stride_bits as usize;
        for middle in 0..third.radix as usize {
            let middle_address = outer_address + middle * third.stride_bits as usize;
            for inner in 0..second.radix as usize {
                let row_address = middle_address + inner * second.stride_bits as usize;
                for step in 0..lowest.radix as usize {
                    visit(index, row_address + step * lowest.stride_bits as usize);
                    index += 1;
                }
            }
        }
    }
}

/// The raw code of every element: the low plane, then the high plane shifted above it.
#[inline(always)]
fn gather_codes<const ELEMENTS: usize, const BYTES: usize>(
    layout: &BlockLayout,
    block: &[u8; BYTES],
) -> [u8; ELEMENTS] {
    let element = &layout.element;
    let mut codes = [0u8; ELEMENTS];
    walk_plane(&element.low, |index, address| {
        codes[index] = read_code_bits(block, address, element.low.bits as u32) as u8;
    });
    if let Some(plane) = &element.high {
        walk_plane(plane, |index, address| {
            codes[index] |= (read_code_bits(block, address, plane.bits as u32) << element.low.bits) as u8;
        });
    }
    codes
}

const fn is_integer_code(layout: &BlockLayout) -> bool {
    (is_unsigned_integer(layout) || is_twos_complement_integer(layout)) && code_bits(layout) <= 16
}

/// `multiplier * code - bias` of an integer code, in the integer domain: every operand is a small
/// integer, so the one conversion to `f32` is exact and equal to converting the code first.
#[inline(always)]
fn integer_factor(layout: &BlockLayout, raw: u32) -> i32 {
    let bits = code_bits(layout);
    let code = if is_twos_complement_integer(layout) {
        ((raw << (32 - bits)) as i32) >> (32 - bits)
    } else {
        raw as i32
    };
    code * i32::from(layout.multiplier) - i32::from(layout.element.bias)
}

#[inline(always)]
fn decode_chunk<const ELEMENTS: usize, const BYTES: usize>(
    layout: &BlockLayout,
    block: &[u8; BYTES],
    out: &mut [MaybeUninit<f32>; ELEMENTS],
) {
    let scale = read_field(block, layout.scale);
    let multiplier = f32::from(layout.multiplier);
    let bias = f32::from(layout.element.bias);
    let min_term = match layout.min {
        Some(field) => f32::from(layout.min_multiplier) * read_field(block, field),
        None => 0.0,
    };
    let value_of = |raw: u32| {
        let factor = if is_integer_code(layout) {
            integer_factor(layout, raw) as f32
        } else {
            decode(raw, layout.element.code) * multiplier - bias
        };
        let value = if is_sign_code(layout) {
            if factor > 0.0 { scale } else { -scale }
        } else {
            factor * scale
        };
        if layout.min.is_some() { value + min_term } else { value }
    };
    // ieee 754 does not say which nan payload survives an add of two nans, so the min's is chosen
    // here and the result does not depend on which operand an isa's add returns
    if layout.min.is_some() && min_term.is_nan() {
        out.iter_mut().for_each(|slot| {
            slot.write(min_term);
        });
        return;
    }
    let codes = gather_codes::<ELEMENTS, BYTES>(layout, block);
    for (slot, &byte) in out.iter_mut().zip(&codes) {
        slot.write(value_of(u32::from(byte)));
    }
}

#[inline(always)]
fn decode_run<const ELEMENTS: usize, const BYTES: usize>(
    layout: BlockLayout,
    data: &[u8],
    out: &mut [MaybeUninit<f32>],
) {
    let blocks = data.as_chunks::<BYTES>().0;
    let slots = out.as_chunks_mut::<ELEMENTS>().0;
    for (block, slot) in blocks.iter().zip(slots) {
        decode_chunk(&layout, block, slot);
    }
}

#[inline(always)]
fn raw_of_value(layout: &BlockLayout, value: i32) -> u32 {
    if is_twos_complement_integer(layout) {
        (value as u32) & ((1u32 << code_bits(layout)) - 1)
    } else {
        (value + layout.element.bias as i32) as u32
    }
}

/// Packs the block-local `codes` into `block` with the same nested digit walk the decode reads by.
#[inline(always)]
fn scatter_codes<const ELEMENTS: usize, const BYTES: usize>(
    layout: &BlockLayout,
    codes: &[u8; ELEMENTS],
    block: &mut [u8; BYTES],
) {
    let element = &layout.element;
    walk_plane(&element.low, |index, address| {
        put_code_bits(block, address, element.low.bits as u32, u32::from(codes[index]));
    });
    if let Some(plane) = &element.high {
        walk_plane(plane, |index, address| {
            put_code_bits(block, address, plane.bits as u32, u32::from(codes[index] >> element.low.bits));
        });
    }
}

/// The signed value of largest magnitude, the first one on a tie, as `quantize_row_q4_0_ref` picks.
#[inline(always)]
fn signed_max<const ELEMENTS: usize>(input: &[f32; ELEMENTS]) -> f32 {
    let mut amax = 0.0f32;
    let mut max = 0.0f32;
    for &value in input {
        if amax < value.abs() {
            amax = value.abs();
            max = value;
        }
    }
    max
}

#[inline(always)]
fn peak<const ELEMENTS: usize>(input: &[f32; ELEMENTS]) -> f32 {
    input.iter().fold(0.0f32, |peak, value| peak.max(value.abs()))
}

#[inline(always)]
fn inverse_of(scale: f32) -> f32 {
    if scale == 0.0 { 0.0 } else { 1.0 / scale }
}

/// Fits the block's scale and writes the raw codes; returns the scale the block stores.
#[inline(always)]
fn fit_codes<const ELEMENTS: usize>(
    layout: &BlockLayout,
    fit: Fit,
    input: &[f32; ELEMENTS],
    codes: &mut [u8; ELEMENTS],
) -> f32 {
    match fit {
        Fit::SignedMax => {
            let bias = layout.element.bias as i32;
            let scale = signed_max(input) / -(bias as f32);
            let inverse = inverse_of(scale);
            let top = (1i32 << code_bits(layout)) - 1;
            let offset = bias as f32 + 0.5;
            for (code, &value) in codes.iter_mut().zip(input) {
                *code = top.min((value * inverse + offset) as i32) as u8;
            }
            scale
        }
        Fit::PeakRound | Fit::Runtime => {
            let scale = peak(input) / reach_of(layout) as f32;
            let inverse = inverse_of(scale);
            let (lowest, highest) = integer_bounds(layout);
            for (code, &value) in codes.iter_mut().zip(input) {
                let rounded = (libm::roundf(value * inverse) as i32).clamp(lowest, highest);
                *code = raw_of_value(layout, rounded) as u8;
            }
            scale
        }
        Fit::MeanSign => {
            let mut sum = 0.0f32;
            for &value in input {
                sum += value.abs();
            }
            for (code, &value) in codes.iter_mut().zip(input) {
                *code = u8::from(value >= 0.0);
            }
            sum / ELEMENTS as f32
        }
    }
}

#[inline(always)]
fn encode_chunk<const ELEMENTS: usize, const BYTES: usize>(
    layout: &BlockLayout,
    fit: Fit,
    input: &[f32; ELEMENTS],
    out: &mut [u8; BYTES],
) {
    let mut codes = [0u8; ELEMENTS];
    let scale = fit_codes(layout, fit, input, &mut codes);
    let mut block = [0u8; BYTES];
    let width = layout.scale.layout.width() as u32;
    put_bits(&mut block, layout.scale.byte_offset as usize * 8, width, encode(scale, layout.scale.layout));
    scatter_codes(layout, &codes, &mut block);
    *out = block;
}

#[inline(always)]
fn encode_run<const ELEMENTS: usize, const BYTES: usize>(
    layout: BlockLayout,
    fit: Fit,
    values: &[f32],
    out: &mut [u8],
) {
    let inputs = values.as_chunks::<ELEMENTS>().0;
    let blocks = out.as_chunks_mut::<BYTES>().0;
    for (input, block) in inputs.iter().zip(blocks) {
        encode_chunk(&layout, fit, input, block);
    }
}

/// The encode and decode of one named simple block layout, as plain function pointers to
/// instantiations with the layout as a literal. No allocation, no trait object.
///
/// Example: resolving [`BlockLayout::Q4_0`] and storing 64 values takes 36 bytes of output.
#[derive(Clone, Copy)]
pub struct BlockCodec {
    encode: Option<EncodeFn>,
    decode: DecodeFn,
    elements: usize,
    block_bytes: usize,
}

macro_rules! resolve_named_block {
    ($layout:expr, [ $(($name:ident, $text:literal, $encodes:literal)),* $(,)? ]) => {{
        $(
            if *$layout == BlockLayout::$name {
                const _: () = assert!(BlockLayout::$name.is_simple());
                const _: () = assert!(planes_cover(&BlockLayout::$name));
                const _: () = assert!(code_bits(&BlockLayout::$name) <= 8);
                const ELEMENTS: usize = BlockLayout::$name.elements as usize;
                const BYTES: usize = BlockLayout::$name.block_bytes as usize;
                return Some(BlockCodec {
                    encode: match (fit_of(&BlockLayout::$name), $encodes) {
                        (Fit::Runtime, _) | (_, false) => None,
                        _ => Some(|values, out| {
                            encode_run::<ELEMENTS, BYTES>(
                                BlockLayout::$name,
                                fit_of(&BlockLayout::$name),
                                values,
                                out,
                            );
                        }),
                    },
                    decode: |data, out| decode_run::<ELEMENTS, BYTES>(BlockLayout::$name, data, out),
                    elements: ELEMENTS,
                    block_bytes: BYTES,
                });
            }
        )*
        None
    }};
}

impl BlockCodec {
    /// The instantiation for a layout in the block table, `None` for any other.
    #[must_use]
    pub fn resolve(layout: &BlockLayout) -> Option<Self> {
        named_block_table!(resolve_named_block!(layout,))
    }

    /// The instantiation for a [`NumericFormat`] whose packing is a table layout: a block format,
    /// or a linear-ladder level format over one. `None` for scalars, codebooks and explicit tables.
    #[must_use]
    pub fn for_format(format: &NumericFormat) -> Option<Self> {
        match format {
            NumericFormat::Block(layout) => Self::resolve(layout),
            NumericFormat::Levels(levels) => match (levels.packing(), levels.levels) {
                (Some(layout), LevelTable::Linear { .. }) if levels.levels == layout.levels() => {
                    Self::resolve(&layout)
                }
                _ => None,
            },
            NumericFormat::Scalar(_) => None,
        }
    }

    /// Elements per block.
    #[must_use]
    pub const fn elements(&self) -> usize {
        self.elements
    }

    /// Stored bytes per block.
    #[must_use]
    pub const fn block_bytes(&self) -> usize {
        self.block_bytes
    }

    /// Whether this layout has an encode instantiation.
    #[must_use]
    pub const fn can_encode(&self) -> bool {
        self.encode.is_some()
    }

    /// Encodes whole blocks of `values` into `out`.
    ///
    /// # Errors
    /// [`NumericError::UnsupportedEncode`] for a layout with no encode instantiation,
    /// [`NumericError::InputNotElementMultiple`] when `values` is not whole blocks,
    /// [`NumericError::OutputSizeMismatch`] when `out` is not exactly the packed byte count.
    #[must_use = "an error leaves the output unwritten"]
    pub fn encode(&self, values: &[f32], out: &mut [u8]) -> Result<(), NumericError> {
        let Some(run) = self.encode else {
            return Err(NumericError::UnsupportedEncode {
                format: "block",
                reason: "no compile-time encode for this layout's fit",
            });
        };
        if !values.len().is_multiple_of(self.elements) {
            return Err(NumericError::InputNotElementMultiple {
                format: "block",
                found: values.len(),
                elements: self.elements,
            });
        }
        let expected = values.len() / self.elements * self.block_bytes;
        if out.len() != expected {
            return Err(NumericError::OutputSizeMismatch {
                found: out.len(),
                expected,
            });
        }
        run(values, out);
        Ok(())
    }

    /// Decodes whole blocks of `data` into the caller-owned, possibly uninitialised `out`,
    /// initialising every slot.
    ///
    /// # Errors
    /// [`NumericError::InputNotBlockMultiple`] when `data` is not whole blocks,
    /// [`NumericError::OutputSizeMismatch`] when `out` is not one slot per element; nothing is
    /// written in either case.
    #[must_use = "an error leaves the output uninitialised"]
    pub fn decode(&self, data: &[u8], out: &mut [MaybeUninit<f32>]) -> Result<(), NumericError> {
        if !data.len().is_multiple_of(self.block_bytes) {
            return Err(NumericError::InputNotBlockMultiple {
                format: "block",
                found: data.len(),
                block_bytes: self.block_bytes,
            });
        }
        let expected = data.len() / self.block_bytes * self.elements;
        if out.len() != expected {
            return Err(NumericError::OutputSizeMismatch {
                found: out.len(),
                expected,
            });
        }
        (self.decode)(data, out);
        Ok(())
    }

    /// The raw encode instantiation, if any; the caller guarantees whole blocks and exact sizes.
    #[must_use]
    pub const fn encode_fn(&self) -> Option<EncodeFn> {
        self.encode
    }

    /// The raw decode instantiation; the caller guarantees whole blocks and exact sizes.
    #[must_use]
    pub const fn decode_fn(&self) -> DecodeFn {
        self.decode
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::float_layout::FloatLayout;

    const RESOLVING_NAMED_FORMATS: usize = 13;

    fn offset_binary_u8_layout() -> BlockLayout {
        let mut layout = BlockLayout::Q8_0;
        layout.element.code = FloatLayout::UINT8;
        layout.element.bias = 128;
        layout
    }

    #[test]
    fn signed_max_fit_keeps_eight_bit_offset_binary_codes_unsaturated() {
        let layout = offset_binary_u8_layout();
        let mut input = [0.0f32; 32];
        input[0] = 1.0;
        input[1] = -2.0;
        input[2] = 1.984_375;
        let mut codes = [0u8; 32];

        let scale = fit_codes(&layout, Fit::SignedMax, &input, &mut codes);

        assert_eq!(scale, 0.015_625);
        assert_eq!(&codes[..4], &[192, 0, 255, 128]);
        assert!(codes[3..].iter().all(|&code| code == 128));
    }

    fn layout_is_table_shaped(layout: &BlockLayout) -> bool {
        layout.is_simple() && planes_cover(layout) && code_bits(layout) <= 8
    }

    fn format_should_resolve(format: &NumericFormat) -> bool {
        match format {
            NumericFormat::Block(layout) => layout_is_table_shaped(layout),
            NumericFormat::Levels(levels) => match (levels.packing(), levels.levels) {
                (Some(layout), LevelTable::Linear { .. }) => {
                    levels.levels == layout.levels() && layout_is_table_shaped(&layout)
                }
                _ => false,
            },
            NumericFormat::Scalar(_) => false,
        }
    }

    #[test]
    fn every_named_format_resolves_exactly_when_the_rule_says_it_does() {
        let mut resolving = 0usize;
        for (format, name) in NumericFormat::ALL.iter().zip(NumericFormat::NAMES) {
            let resolves = BlockCodec::for_format(format).is_some();

            assert_eq!(resolves, format_should_resolve(format), "{name}");
            resolving += usize::from(resolves);
        }

        assert_eq!(resolving, RESOLVING_NAMED_FORMATS);
    }

    #[test]
    fn each_clause_of_is_simple_has_a_layout_that_does_not_resolve() {
        assert!(!BlockLayout::Q4_K.is_simple() && BlockLayout::Q4_K.sub.is_some());
        assert!(!BlockLayout::NVFP4.is_simple() && BlockLayout::NVFP4.scale_runs.is_some());
        assert!(!BlockLayout::TQ1_0.is_simple() && BlockLayout::TQ1_0.element.packed_radix > 1);
        for layout in [BlockLayout::Q4_K, BlockLayout::NVFP4, BlockLayout::TQ1_0] {
            assert!(BlockCodec::resolve(&layout).is_none());
        }
    }
}
