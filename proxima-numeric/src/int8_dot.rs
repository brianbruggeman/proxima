//! Whether a [`BlockLayout`] decomposes into int8 codes per 32-element chunk, and how.
//!
//! An int8 chunk decomposition is a layout whose every 32-element chunk is a scale (optionally a
//! min) times int8 values, `multiplier * code - bias`, each recoverable from the block's bytes with
//! integer-lane arithmetic. That is the arithmetic an int8-dot reader (a `dp4a` or int8 `mma`
//! kernel) needs, and [`int8_dot_shape`] is the one place that decides it: bind asks it whether a
//! matmul's activation can be quantized to q8_1 against this weight, and
//! `omega::cuda::block_dot` renders the named shape as CUDA source.
//!
//! Teaching pointer: [`BlockLayout`] is the data; this reads its fields and nothing else, so the
//! answer is structural. No allocation, no format named: a new block format that fits one of the
//! four shapes lowers by being a constant in [`crate::block_layout`], not by editing a reader.

use crate::block_layout::{BlockLayout, Coded, Field, Plane};
use crate::float_layout::{FloatLayout, SignEncoding};

/// Elements one int8 dot step covers: eight `dp4a` words.
pub const INT8_DOT_ELEMENTS: usize = 32;

const NIBBLE_BITS: usize = 4;
const CHUNK_NIBBLE_BITS: usize = INT8_DOT_ELEMENTS * NIBBLE_BITS;
const HALF_CHUNK: usize = INT8_DOT_ELEMENTS / 2;
const ADDRESS_ALIGNMENT_BITS: usize = 16;
const BYTE_BITS: usize = 8;
const MAX_PARTS: usize = 2;
const WORD_ELEMENTS: usize = 4;
const TABLE_ENTRY_LIMIT: usize = 1 << 16;
const RADIX_LIMIT: u32 = 128;

/// How a block's 32-element chunks decompose into int8 codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Int8DotShape {
    /// Mixed-radix digits of stored bytes (the ternary formats): each code is one digit.
    Radix {
        /// Values per digit.
        radix: u8,
    },
    /// A single plane of `bits` (1, 2, 4 or 8) wide codes, chunk after chunk, as one contiguous
    /// bit run starting `base_byte` bytes into the block.
    BitRuns {
        bits: u8,
        base_byte: usize,
    },
    /// A 4-bit plane whose chunk holds its first half in the low nibbles of 16 bytes and its
    /// second half in the high nibbles (Q4_0), with an optional one-bit plane as the fifth bit
    /// (Q5_0).
    NibbleSplit {
        low_base_byte: usize,
        /// Byte offset of the one-bit-per-element high plane, when the code has a fifth bit.
        high_base_byte: Option<usize>,
    },
    /// A 4-bit plane plus an optional one or two bit plane, strided one byte per element, under
    /// per-sub-block scales (the K formats).
    SubBlocks {
        /// Runs of `32 / parts` elements per chunk that each carry their own scale.
        parts: u8,
        /// Each chunk also carries a signed min term.
        min: bool,
    },
}

/// The int8 chunk decomposition of `layout`, or `None` when its codes do not decompose that way:
/// a float code, a whole-block min, a plane layout none of the four shapes names, an affine
/// value that leaves int8, or a scale field or index table the reader cannot address.
///
/// `omega::cuda::block_dot::block_dot_i8` returns `Some` for exactly the layouts this does.
#[must_use]
pub fn int8_dot_shape(layout: &BlockLayout) -> Option<Int8DotShape> {
    if layout.element.packed_radix > 1 {
        return radix_shape(layout);
    }
    bit_run_shape(layout)
        .or_else(|| nibble_split_shape(layout))
        .or_else(|| sub_block_shape(layout))
}

fn bit_run_shape(layout: &BlockLayout) -> Option<Int8DotShape> {
    let element = &layout.element;
    let bits = usize::from(element.low.bits);
    let elements = usize::from(layout.elements);
    let linear = layout.sub.is_none()
        && layout.min.is_none()
        && element.high.is_none()
        && element.packed_radix == 0
        && elements.is_multiple_of(INT8_DOT_ELEMENTS)
        && matches!(bits, 1 | 2 | 4 | 8);
    if !linear || !values_fit_int8(element, layout.multiplier, bits) {
        return None;
    }
    let base = contiguous_base(&element.low, elements, bits)?;
    let two_bit_table = bits == 2;
    let expressible = scale_is_readable(&layout.scale) && (two_bit_table || affine_fits(layout.multiplier, element.bias, bits));
    expressible.then_some(Int8DotShape::BitRuns { bits: element.low.bits, base_byte: base / BYTE_BITS })
}

fn nibble_split_shape(layout: &BlockLayout) -> Option<Int8DotShape> {
    let element = &layout.element;
    let elements = usize::from(layout.elements);
    let linear = layout.sub.is_none()
        && layout.min.is_none()
        && layout.scale_runs.is_none()
        && element.packed_radix == 0
        && elements.is_multiple_of(INT8_DOT_ELEMENTS);
    let low_base = nibble_split_base(&element.low, elements).filter(|_| linear)?;
    let high_base = match &element.high {
        Some(plane) => Some(bit_run_base(plane, elements)?),
        None => None,
    };
    let code_bits = NIBBLE_BITS + usize::from(high_base.is_some());
    let expressible = values_fit_int8(element, layout.multiplier, code_bits)
        && scale_is_readable(&layout.scale)
        && affine_fits(layout.multiplier, element.bias, code_bits);
    expressible.then_some(Int8DotShape::NibbleSplit {
        low_base_byte: low_base / BYTE_BITS,
        high_base_byte: high_base.map(|base| base / BYTE_BITS),
    })
}

fn sub_block_shape(layout: &BlockLayout) -> Option<Int8DotShape> {
    let sub = layout.sub.as_ref()?;
    let elements = usize::from(layout.elements);
    let count = usize::from(sub.count);
    let parts = parts_per_chunk(elements, count)?;
    let min = layout.min.as_ref().zip(sub.min.as_ref());
    let eligible = layout.scale_runs.is_none()
        && layout.element.packed_radix == 0
        && elements.is_multiple_of(INT8_DOT_ELEMENTS)
        && layout.block_bytes.is_multiple_of(2)
        && layout.min.is_some() == sub.min.is_some()
        && (min.is_none() || parts == 1);
    if !eligible {
        return None;
    }
    let chunks = elements / INT8_DOT_ELEMENTS;
    let low = &layout.element.low;
    let high = layout.element.high.as_ref();
    let code_bits = NIBBLE_BITS + high.map_or(0, |plane| usize::from(plane.bits));
    let planes_addressable = usize::from(low.bits) == NIBBLE_BITS
        && chunk_table_fits(low, chunks)
        && high.is_none_or(|plane| chunk_table_fits(plane, chunks));
    let scales_addressable = scale_is_readable(&layout.scale)
        && sub_code_addressable(&sub.scale, count)
        && min.is_none_or(|(field, coded)| scale_is_readable(field) && sub_code_addressable(coded, count));
    let expressible = planes_addressable
        && scales_addressable
        && values_fit_int8(&layout.element, layout.multiplier, code_bits)
        && affine_fits(layout.multiplier, layout.element.bias, code_bits);
    expressible.then_some(Int8DotShape::SubBlocks { parts: u8::try_from(parts).ok()?, min: min.is_some() })
}

fn radix_shape(layout: &BlockLayout) -> Option<Int8DotShape> {
    let element = &layout.element;
    let radix = u32::from(element.packed_radix);
    let elements = usize::from(layout.elements);
    let usable = layout.sub.is_none()
        && layout.min.is_none()
        && elements.is_multiple_of(INT8_DOT_ELEMENTS)
        && radix > 1
        && radix < RADIX_LIMIT
        && radix * 255 < 1 << 16
        && codes_fit_int8(element, layout.multiplier, radix);
    let sourced = usable && (0..elements).all(|index| radix_digit_source(element, radix, index).is_some());
    let expressible = sourced && scale_is_readable(&layout.scale) && (0..INT8_DOT_ELEMENTS / WORD_ELEMENTS).all(|word| radix_word_expressible(layout, radix, word));
    expressible.then_some(Int8DotShape::Radix { radix: element.packed_radix })
}

/// For radix-packed element `index`: the byte holding its digit and `radix^lane mod 256`, the
/// multiplier that brings that digit to the top of the byte. `None` when the element lies in no
/// region or its digit is not one whole byte at a byte address.
#[must_use]
pub fn radix_digit_source(element: &Coded, radix: u32, index: usize) -> Option<(usize, u32)> {
    let regions = element.regions.get(..usize::from(element.region_count))?;
    let region = regions.iter().find(|region| {
        let start = usize::from(region.start);
        (start..start + usize::from(region.elements)).contains(&index)
    })?;
    let local = index - usize::from(region.start);
    let address = region.source.address(local);
    let axis = usize::from(region.digit_axis);
    let divisor: usize = region.source.digits.iter().take(axis).map(|digit| usize::from(digit.radix)).product();
    let lane_radix = usize::from(region.source.digits.get(axis)?.radix);
    let lane = (local / divisor) % lane_radix;
    let factor = (0..lane).fold(1u32, |product, _| (product * radix) % 256);
    (region.source.bits == 8 && address.is_multiple_of(8)).then_some((address / 8, factor))
}

/// The four codes from element `first` are one digit of four consecutive bytes: that run's first
/// byte and factor.
fn word_run(element: &Coded, radix: u32, first: usize) -> Option<(usize, u32)> {
    let (first_byte, factor) = radix_digit_source(element, radix, first)?;
    (1..WORD_ELEMENTS)
        .all(|offset| radix_digit_source(element, radix, first + offset) == Some((first_byte + offset, factor)))
        .then_some((first_byte, factor))
}

/// Codes 0 and 2 share one byte and codes 1 and 3 share another, both inside one aligned word.
fn word_pair(element: &Coded, radix: u32, first: usize) -> bool {
    let source = |offset: usize| radix_digit_source(element, radix, first + offset);
    matches!(
        (source(0), source(1), source(2), source(3)),
        (Some((even, one)), Some((odd, two)), Some((even_again, three)), Some((odd_again, four)))
            if even == even_again && odd == odd_again && even / 4 == odd / 4 && one < 256 && two < 256 && three < 256 && four < 256
    )
}

/// A word decodes either with one instruction stream selected by chunk position (aligned runs and
/// byte pairs), per chunk, or by a selector table of `byte | factor << 8` fields; only that table
/// cannot address a run at byte 256 or beyond.
fn radix_word_expressible(layout: &BlockLayout, radix: u32, word: usize) -> bool {
    let element = &layout.element;
    let chunks = usize::from(layout.elements) / INT8_DOT_ELEMENTS;
    let first_of = |chunk: usize| chunk * INT8_DOT_ELEMENTS + word * WORD_ELEMENTS;
    let aligned = usize::from(layout.block_bytes).is_multiple_of(4);
    let every_chunk_decodes = (0..chunks).all(|chunk| match word_run(element, radix, first_of(chunk)) {
        Some((byte, _)) => byte.is_multiple_of(4),
        None => word_pair(element, radix, first_of(chunk)),
    });
    let any_pair = (0..chunks).any(|chunk| word_run(element, radix, first_of(chunk)).is_none());
    if aligned && any_pair && every_chunk_decodes {
        return true;
    }
    let mut first_run: Option<(usize, u32)> = None;
    let mut uniform = true;
    let mut addressable = true;
    for run in (0..chunks).filter_map(|chunk| word_run(element, radix, first_of(chunk))) {
        match first_run {
            None => first_run = Some(run),
            Some(first) => uniform &= first == run,
        }
        addressable &= run.0 < 256;
    }
    uniform || addressable
}

fn codes_fit_int8(element: &Coded, multiplier: i16, radix: u32) -> bool {
    multiplier > 0
        && (0..radix).all(|code| {
            let value = i32::from(multiplier) * i32::try_from(code).unwrap_or(i32::MAX) - i32::from(element.bias);
            (-128..=127).contains(&value)
        })
}

fn signed_code(code: FloatLayout, raw: i32, bits: usize) -> Option<i32> {
    match code.sign {
        SignEncoding::Absent => Some(raw),
        SignEncoding::TwosComplement => Some(if raw >= 1 << (bits - 1) { raw - (1 << bits) } else { raw }),
        SignEncoding::Magnitude => None,
    }
}

fn values_fit_int8(element: &Coded, multiplier: i16, bits: usize) -> bool {
    (0..1i32 << bits).all(|raw| {
        signed_code(element.code, raw, bits)
            .map(|code| i32::from(multiplier) * code - i32::from(element.bias))
            .is_some_and(|value| (-128..=127).contains(&value))
    })
}

/// `x * multiplier - bias` on each byte of a packed word stays inside the byte: the multiplier is
/// non-negative and the largest code times it fits eight bits, unless the affine is the identity.
fn affine_fits(multiplier: i16, bias: i16, bits: usize) -> bool {
    if multiplier == 1 && bias == 0 {
        return true;
    }
    u32::try_from(multiplier).ok().is_some_and(|scale| ((1u32 << bits) - 1) * scale <= 255)
}

/// The block scale is a 16 or 32 bit field at an offset aligned to its width.
fn scale_is_readable(field: &Field) -> bool {
    let offset = field.byte_offset;
    match field.layout.width() {
        16 => offset.is_multiple_of(2),
        32 => offset.is_multiple_of(4),
        _ => false,
    }
}

/// The bit address of chunk 0, when every chunk is a contiguous run on a 16-bit boundary.
fn contiguous_base(plane: &Plane, elements: usize, bits: usize) -> Option<usize> {
    let base = plane.address(0);
    let contiguous = (0..elements).all(|index| plane.address(index) == base + index * bits);
    (contiguous && base.is_multiple_of(ADDRESS_ALIGNMENT_BITS)).then_some(base)
}

fn nibble_split_offset(position: usize) -> usize {
    (position % HALF_CHUNK) * 8 + (position / HALF_CHUNK) * NIBBLE_BITS
}

fn nibble_split_base(plane: &Plane, elements: usize) -> Option<usize> {
    let base = plane.address(0);
    let split = usize::from(plane.bits) == NIBBLE_BITS
        && (0..elements).all(|index| {
            plane.address(index) == base + index / INT8_DOT_ELEMENTS * CHUNK_NIBBLE_BITS + nibble_split_offset(index % INT8_DOT_ELEMENTS)
        });
    (split && base.is_multiple_of(ADDRESS_ALIGNMENT_BITS)).then_some(base)
}

fn bit_run_base(plane: &Plane, elements: usize) -> Option<usize> {
    let base = plane.address(0);
    let run = plane.bits == 1 && (0..elements).all(|index| plane.address(index) == base + index);
    (run && base.is_multiple_of(ADDRESS_ALIGNMENT_BITS)).then_some(base)
}

/// Runs of `32 / size` per chunk when a sub-block is `size` elements and divides a chunk.
fn parts_per_chunk(elements: usize, sub_count: usize) -> Option<usize> {
    let size = elements.checked_div(sub_count).filter(|size| *size > 0 && elements.is_multiple_of(sub_count))?;
    let parts = INT8_DOT_ELEMENTS / size;
    (INT8_DOT_ELEMENTS.is_multiple_of(size) && (1..=MAX_PARTS).contains(&parts)).then_some(parts)
}

/// Where chunk `chunk` of a strided plane lives: the even byte offset of its first element and the
/// bit shift inside that byte, consecutive elements one byte apart.
#[must_use]
pub fn chunk_byte_run(plane: &Plane, chunk: usize) -> Option<(u32, u32)> {
    let first = plane.address(chunk * INT8_DOT_ELEMENTS);
    let strided = (0..INT8_DOT_ELEMENTS).all(|lane| plane.address(chunk * INT8_DOT_ELEMENTS + lane) == first + lane * BYTE_BITS);
    let shift = first % BYTE_BITS;
    let fits = shift + usize::from(plane.bits) <= BYTE_BITS && (first / BYTE_BITS).is_multiple_of(2);
    if !(strided && fits) {
        return None;
    }
    Some((u32::try_from(first / BYTE_BITS).ok()?, u32::try_from(shift).ok()?))
}

fn fits_table_entry(value: u32) -> bool {
    usize::try_from(value).is_ok_and(|entry| entry < TABLE_ENTRY_LIMIT)
}

fn chunk_table_fits(plane: &Plane, chunks: usize) -> bool {
    (0..chunks).all(|chunk| {
        chunk_byte_run(plane, chunk).is_some_and(|(byte, shift)| fits_table_entry(byte) && fits_table_entry(shift))
    })
}

/// A per-sub-block scale or min code the reader indexes by sub-block: each plane's bit address
/// stays in one byte and inside a 16-bit table entry, and the code is not sign-magnitude.
fn sub_code_addressable(coded: &Coded, count: usize) -> bool {
    let plane_addressable = |plane: &Plane| {
        (0..count).all(|sub| {
            let address = plane.address(sub);
            address % BYTE_BITS + usize::from(plane.bits) <= BYTE_BITS && address < TABLE_ENTRY_LIMIT
        })
    };
    plane_addressable(&coded.low)
        && coded.high.as_ref().is_none_or(plane_addressable)
        && !matches!(coded.code.sign, SignEncoding::Magnitude)
}

#[cfg(test)]
mod tests {
    use super::{Int8DotShape, int8_dot_shape};
    use crate::block_layout::BlockLayout;
    use crate::format::NumericFormat;

    /// Q4_0: `element.low` is the 4-bit `nibbles(16, 16)` plane (base 16 bits, digits `(16, 8)`
    /// then `(2, 4)`, the nibble-split address map), `element.high` none, `min` and `sub` none, 32
    /// elements, f16 scale at byte 0, bias 8 so codes land in -8..=7.
    #[test]
    fn q4_0_is_a_nibble_split_plane_after_the_two_byte_scale() {
        assert_eq!(
            int8_dot_shape(&BlockLayout::Q4_0),
            Some(Int8DotShape::NibbleSplit { low_base_byte: 2, high_base_byte: None })
        );
    }

    /// Q8_0: `element.low.bits == 8`, no high plane, `sub` and `min` none, 32 elements,
    /// contiguous bytes from bit 16, f16 scale at offset 0.
    #[test]
    fn q8_0_is_an_eight_bit_run_after_the_two_byte_scale() {
        assert_eq!(int8_dot_shape(&BlockLayout::Q8_0), Some(Int8DotShape::BitRuns { bits: 8, base_byte: 2 }));
    }

    /// Q6_K: `sub` is some (16 sub-blocks over 256 elements, so 2 parts per 32-element chunk),
    /// `min` is none, `element.low.bits == 4` with a 2-bit high plane, `scale_runs` none.
    #[test]
    fn q6_k_is_sub_block_scales_in_two_parts_without_a_min() {
        assert_eq!(int8_dot_shape(&BlockLayout::Q6_K), Some(Int8DotShape::SubBlocks { parts: 2, min: false }));
    }

    /// Q4_K: `sub` has 8 sub-blocks over 256 elements (one part per chunk) and carries a min
    /// code; `layout.min` is some.
    #[test]
    fn q4_k_is_sub_block_scales_in_one_part_with_a_min() {
        assert_eq!(int8_dot_shape(&BlockLayout::Q4_K), Some(Int8DotShape::SubBlocks { parts: 1, min: true }));
    }

    /// Q4_1: `layout.min` is some with no `sub`; the whole-block min is not a per-chunk term.
    #[test]
    fn q4_1_with_a_whole_block_min_has_no_decomposition() {
        assert_eq!(int8_dot_shape(&BlockLayout::Q4_1), None);
    }

    /// Q5_0: `element.low` is `nibbles(48, 16)` (base 48 bits = byte 6), `element.high` is
    /// `run(1, 16, 32)`, one bit per element from bit 16 = byte 2, bias 16.
    #[test]
    fn q5_0_is_a_nibble_split_plane_with_a_bit_run_high_plane() {
        assert_eq!(
            int8_dot_shape(&BlockLayout::Q5_0),
            Some(Int8DotShape::NibbleSplit { low_base_byte: 6, high_base_byte: Some(2) })
        );
    }

    /// TQ1_0: `element.packed_radix == 3` with three byte-wide regions and an f16 scale at byte
    /// 52, so the codes are radix-three digits of stored bytes.
    #[test]
    fn tq1_0_is_radix_three() {
        assert_eq!(int8_dot_shape(&BlockLayout::TQ1_0), Some(Int8DotShape::Radix { radix: 3 }));
    }

    /// F16 is `NumericFormat::Scalar`: `block_layout()` is none, so there is no layout to decompose.
    #[test]
    fn f16_has_no_block_layout_and_so_no_decomposition() {
        let layout = NumericFormat::Float16.block_layout();
        assert_eq!(layout.as_ref().and_then(int8_dot_shape), None);
    }

    /// MXFP4: scale is `F8_E8M0` (`width() == 8`, neither 16 nor 32) and `low` is `nibbles(8, 16)`
    /// whose base is not 16-bit aligned. FP6_E3M2: `low.bits == 6`, none of 1, 2, 4, 8, and no
    /// `sub`.
    #[test]
    fn float_coded_block_layouts_have_no_decomposition() {
        for layout in [BlockLayout::MXFP4, BlockLayout::FP6_E3M2] {
            assert_eq!(int8_dot_shape(&layout), None, "{layout:?} stores a float code");
        }
    }
}
