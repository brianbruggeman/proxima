//! A simple block format resolved once into the two slice conversions that carry it.
//!
//! Teaching pointer: [`crate::float_layout_codec`] does this for one stored number per element;
//! this does it for a block of them. `encode_run` and `decode_run` are the two loops, written once
//! over a [`BlockLayout`]. [`FormatCodec::block`] instantiates them once per row of
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
use crate::float_layout::SignEncoding;
use crate::float_layout_codec::{decode, encode};
use crate::format::{LevelTable, NumericFormat};
use crate::format_codec::FormatCodec;
use crate::fused;

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

/// Stores the `width` low bits of `bits` at byte `offset`. When `OWNED` (see [`scale_owns_its_bytes`])
/// it is one little-endian integer store, so the scale is a store of its own width and not a run of
/// byte stores a vectoriser would join to the code bytes beside it; otherwise the bytes are shared
/// with code bits already in `block`, so the field is ored in.
#[inline(always)]
fn put_field<const BYTES: usize, const OWNED: bool>(block: &mut [u8; BYTES], offset: usize, width: u32, bits: u32) {
    let bytes = width.div_ceil(8) as usize;
    let masked = bits & (u32::MAX >> (32 - width));
    if OWNED {
        block[offset..offset + bytes].copy_from_slice(&masked.to_le_bytes()[..bytes]);
        return;
    }
    for (step, byte) in masked.to_le_bytes()[..bytes].iter().enumerate() {
        block[offset + step] |= byte;
    }
}

/// Whether the scale is a whole number of bytes that no code bit lies in: the property that lets
/// [`put_field`] overwrite those bytes instead of oring into them.
const fn scale_owns_its_bytes(layout: &BlockLayout) -> bool {
    let width = layout.scale.layout.width() as usize;
    if !width.is_multiple_of(8) {
        return false;
    }
    let first = layout.scale.byte_offset as usize * 8;
    let end = first + width;
    let high_touches = match &layout.element.high {
        Some(plane) => plane_touches(plane, layout.elements as usize, first, end),
        None => false,
    };
    !plane_touches(&layout.element.low, layout.elements as usize, first, end) && !high_touches
}

/// Whether any code of the plane lies in the bit range `[first, end)`.
const fn plane_touches(plane: &Plane, elements: usize, first: usize, end: usize) -> bool {
    let mut index = 0;
    while index < elements {
        let address = plane.address(index);
        if address < end && address + plane.bits as usize > first {
            return true;
        }
        index += 1;
    }
    false
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

/// Ors `bits` into `block` at `[address, address + width)` for a plane of at most eight bits; the
/// block starts zeroed.
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

/// Codes a register word carries for a plane whose innermost digit is a contiguous run: as many as
/// fit in 64 bits and in the run, trimmed until the word is a whole number of bytes; zero when no
/// count does.
const fn word_codes(plane: &Plane) -> usize {
    let bits = plane.bits as usize;
    if bits == 0 {
        return 0;
    }
    let radix = plane.digits[0].radix as usize;
    let mut count = if 64 / bits < radix { 64 / bits } else { radix };
    while count > 0 && !(count * bits).is_multiple_of(8) {
        count -= 1;
    }
    count
}

/// Whether a plane's innermost digit is a contiguous run (`stride_bits == bits`) that starts on a
/// byte, whose every outer step moves by whole bytes and whose length is a whole number of register
/// words: the property that lets the run be packed and unpacked a word at a time instead of a code
/// at a time.
const fn is_run_plane(plane: &Plane) -> bool {
    let lowest = plane.digits[0];
    if plane.bits == 0 || lowest.stride_bits != plane.bits as u16 || !plane.base_bits.is_multiple_of(8) {
        return false;
    }
    let mut position = 1;
    while position < MAX_DIGITS {
        let digit = plane.digits[position];
        if digit.radix > 1 && !digit.stride_bits.is_multiple_of(8) {
            return false;
        }
        position += 1;
    }
    let count = word_codes(plane);
    count > 0 && (lowest.radix as usize).is_multiple_of(count)
}

/// Whether a plane is moved a register word at a time: a run plane whose codes are narrower than a
/// byte. A byte-wide code is already one byte per element, which the staged path reads and writes as
/// a plain vector of bytes, so a word shuffle there only adds work.
const fn is_word_packed(plane: &Plane) -> bool {
    plane.bits < 8 && is_run_plane(plane)
}

/// [`is_word_packed`] for the plane that holds the bits above the low plane, if there is one.
const fn is_high_word_packed(layout: &BlockLayout) -> bool {
    match &layout.element.high {
        Some(plane) => is_word_packed(plane),
        None => false,
    }
}

/// Whether the high plane is exactly one register word of codes, which the fit can fill as it
/// produces each code: a vector unit then shifts and sums every code into the word, where reading
/// them back out of the staged array one lane at a time costs an instruction a lane.
const fn is_high_single_word(layout: &BlockLayout) -> bool {
    match &layout.element.high {
        Some(plane) => {
            is_word_packed(plane)
                && word_codes(plane) * plane.bits as usize == 32
                && plane_span(plane) * plane.bits as usize == 32
                && plane_span(plane) == layout.elements as usize
        }
        None => false,
    }
}

/// Whether decode reads every code straight out of register words: one word-packed plane and no
/// other, so there is no code to stage.
const fn unpacks_by_word(layout: &BlockLayout) -> bool {
    layout.element.high.is_none() && is_word_packed(&layout.element.low)
}

/// Whether every code is already a byte in index order: no high plane, an eight-bit low plane, and
/// element `index` at byte `base + index`. Decode then reads each code from the block where it lies;
/// staging them in an array first is a copy of the whole run, which the optimiser cannot forward
/// because the block and the output are not proven disjoint.
const fn reads_codes_in_place(layout: &BlockLayout) -> bool {
    let plane = &layout.element.low;
    if layout.element.high.is_some() || plane.bits != 8 || !(plane.base_bits as usize).is_multiple_of(8) {
        return false;
    }
    let mut index = 0;
    while index < layout.elements as usize {
        if plane.address(index) != plane.base_bits as usize + 8 * index {
            return false;
        }
        index += 1;
    }
    true
}

/// Whether one byte of the low plane carries two codes, element `index` in its low half and element
/// `index + elements / 2` in its high half, and the single bit-plane word above them holds element
/// `index`'s bit at bit `index`. Decode then converts each pair from one 32-bit lane per code, which
/// the optimiser widens with one table lookup per four codes; staging the codes as bytes first hands
/// it a byte-to-float conversion that it widens in three steps per four codes.
const fn pairs_halves_with_high_word(layout: &BlockLayout) -> bool {
    let low = &layout.element.low;
    let high = match &layout.element.high {
        Some(plane) => plane,
        None => return false,
    };
    let elements = layout.elements as usize;
    if !is_high_single_word(layout)
        || high.bits != 1
        || low.bits * 2 != 8
        || !low.base_bits.is_multiple_of(8)
        || layout.elements < 2 * low.bits as u16
    {
        return false;
    }
    let mut index = 0;
    while index < elements {
        if high.address(index) != high.base_bits as usize + index {
            return false;
        }
        index += 1;
    }
    let half = elements / 2;
    index = 0;
    while index < half {
        let byte_address = low.base_bits as usize + 8 * index;
        if low.address(index) != byte_address || low.address(index + half) != byte_address + low.bits as usize {
            return false;
        }
        index += 1;
    }
    true
}

/// [`walk_plane`] one run at a time: visits `(first_index, byte_address)` for each run of the
/// innermost digit's radix elements, in index order.
#[inline(always)]
fn walk_runs(plane: &Plane, mut visit: impl FnMut(usize, usize)) {
    let [lowest, second, third, highest] = plane.digits;
    let mut index = 0;
    for outer in 0..highest.radix as usize {
        let outer_address = plane.base_bits as usize + outer * highest.stride_bits as usize;
        for middle in 0..third.radix as usize {
            let middle_address = outer_address + middle * third.stride_bits as usize;
            for inner in 0..second.radix as usize {
                let row_address = middle_address + inner * second.stride_bits as usize;
                visit(index, row_address / 8);
                index += lowest.radix as usize;
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
fn decode_chunk<
    const ELEMENTS: usize,
    const BYTES: usize,
    const BY_WORD: bool,
    const IN_PLACE: bool,
    const PAIRED: bool,
>(
    layout: &BlockLayout,
    block: &[u8; BYTES],
    out: &mut [MaybeUninit<f32>; ELEMENTS],
) {
    let multiplier = f32::from(layout.multiplier);
    let bias = f32::from(layout.element.bias);
    let has_min = layout.min.is_some();
    let min_term = match layout.min {
        Some(field) => f32::from(layout.min_multiplier) * read_field(block, field),
        None => 0.0,
    };
    let scale = read_field(block, layout.scale);
    let value_of = |raw: u32| {
        let factor = if is_integer_code(layout) {
            integer_factor(layout, raw) as f32
        } else {
            decode(raw, layout.element.code) * multiplier - bias
        };
        if is_sign_code(layout) {
            let value = if factor > 0.0 { scale } else { -scale };
            return if has_min { value + min_term } else { value };
        }
        if has_min {
            fused::mul_add(factor, scale, min_term)
        } else {
            factor * scale
        }
    };
    if BY_WORD {
        unpack_runs(&layout.element.low, block, out, value_of);
        return;
    }
    if IN_PLACE {
        let first = layout.element.low.base_bits as usize / 8;
        for (slot, &byte) in out.iter_mut().zip(&block[first..first + ELEMENTS]) {
            slot.write(value_of(u32::from(byte)));
        }
        return;
    }
    if PAIRED {
        decode_halves(layout, block, out, value_of);
        return;
    }
    let codes = gather_codes::<ELEMENTS, BYTES>(layout, block);
    for (slot, &byte) in out.iter_mut().zip(&codes) {
        slot.write(value_of(u32::from(byte)));
    }
}

/// Decodes a layout that [`pairs_halves_with_high_word`]: each low-plane byte gives the code of
/// element `index` and of element `index + elements / 2`, each joined to its bit of the high word.
#[inline(always)]
fn decode_halves<const ELEMENTS: usize, const BYTES: usize>(
    layout: &BlockLayout,
    block: &[u8; BYTES],
    out: &mut [MaybeUninit<f32>; ELEMENTS],
    value_of: impl Fn(u32) -> f32,
) {
    let nibble = layout.element.low.bits as u32;
    let mask = u32::MAX >> (32 - nibble);
    let high_word = match &layout.element.high {
        Some(plane) => read_bits(block, plane.base_bits as usize, 32),
        None => 0,
    };
    let first = layout.element.low.base_bits as usize / 8;
    let half = ELEMENTS / 2;
    // one call of `value_of` per code, not two per pair: a closure with two call sites is left out
    // of line, a call per element
    let bit = 1 << nibble;
    for (index, &byte) in block[first..first + half].iter().enumerate() {
        let byte = u32::from(byte);
        let below = (byte & mask) | (((high_word >> index) << nibble) & bit);
        let above = (byte >> nibble) | ((high_word >> (index + half - nibble as usize)) & bit);
        for (side, code) in [below, above].into_iter().enumerate() {
            out[side * half + index].write(value_of(code));
        }
    }
}

/// Reads a run-plane code a register word at a time and converts each straight into its slot: the
/// shift that isolates a code is the constant `lane * bits`, so no code is staged in a byte array.
#[inline(always)]
fn unpack_runs<const ELEMENTS: usize, const BYTES: usize>(
    plane: &Plane,
    block: &[u8; BYTES],
    out: &mut [MaybeUninit<f32>; ELEMENTS],
    value_of: impl Fn(u32) -> f32,
) {
    let bits = plane.bits as u32;
    let count = word_codes(plane);
    let word_bytes = count * bits as usize / 8;
    let mask = u32::MAX >> (32 - bits);
    let words = plane.digits[0].radix as usize / count;
    walk_runs(plane, |index, first| {
        for word in 0..words {
            let mut bytes = [0u8; 8];
            bytes[..word_bytes].copy_from_slice(&block[first + word * word_bytes..][..word_bytes]);
            let packed = u64::from_le_bytes(bytes);
            for lane in 0..count {
                let offset = lane as u32 * bits;
                let raw = if offset % 32 + bits <= 32 {
                    // a 32-bit half of the word, so a vector lane stays 32 bits wide
                    ((packed >> (offset & !31)) as u32 >> (offset % 32)) & mask
                } else {
                    (packed >> offset) as u32 & mask
                };
                out[index + word * count + lane].write(value_of(raw));
            }
        }
    });
}

#[inline(always)]
fn decode_run<
    const ELEMENTS: usize,
    const BYTES: usize,
    const BY_WORD: bool,
    const IN_PLACE: bool,
    const PAIRED: bool,
>(
    layout: BlockLayout,
    data: &[u8],
    out: &mut [MaybeUninit<f32>],
) {
    let blocks = data.as_chunks::<BYTES>().0;
    let slots = out.as_chunks_mut::<ELEMENTS>().0;
    for (block, slot) in blocks.iter().zip(slots) {
        decode_chunk::<ELEMENTS, BYTES, BY_WORD, IN_PLACE, PAIRED>(&layout, block, slot);
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
fn scatter_codes<
    const ELEMENTS: usize,
    const BYTES: usize,
    const LOW_BY_WORD: bool,
    const HIGH_BY_WORD: bool,
    const HIGH_FUSED: bool,
>(
    layout: &BlockLayout,
    codes: &[u8; ELEMENTS],
    high_word: u32,
    block: &mut [u8; BYTES],
) {
    let element = &layout.element;
    scatter_plane::<BYTES, LOW_BY_WORD>(&element.low, 0, block, |index| codes[index]);
    if let Some(plane) = &element.high {
        if HIGH_FUSED {
            let start = plane.base_bits as usize / 8;
            block[start..start + 4].copy_from_slice(&high_word.to_le_bytes());
        } else {
            scatter_plane::<BYTES, HIGH_BY_WORD>(plane, element.low.bits as u32, block, |index| codes[index]);
        }
    }
}

/// Packs one plane of the codes `code_of` yields (the plane's bits sit `lift` bits up in the code).
/// A word-packed plane writes each register word's bytes once; any other plane ors a code at a time.
#[inline(always)]
fn scatter_plane<const BYTES: usize, const BY_WORD: bool>(
    plane: &Plane,
    lift: u32,
    block: &mut [u8; BYTES],
    code_of: impl Fn(usize) -> u8,
) {
    let bits = plane.bits as u32;
    let mask = u32::MAX >> (32 - bits);
    if !BY_WORD {
        walk_plane(plane, |index, address| {
            put_code_bits(block, address, bits, u32::from(code_of(index)) >> lift);
        });
        return;
    }
    let count = word_codes(plane);
    let word_bytes = count * bits as usize / 8;
    let words = plane.digits[0].radix as usize / count;
    walk_runs(plane, |index, first| {
        for word in 0..words {
            let mut low = 0u32;
            let mut high = 0u32;
            let mut straddle = 0u64;
            for lane in 0..count {
                let offset = lane as u32 * bits;
                let code = (u32::from(code_of(index + word * count + lane)) >> lift) & mask;
                // codes occupy disjoint bits, so a sum is the or, and a sum is what a vector unit reduces
                if offset + bits <= 32 {
                    low = low.wrapping_add(code << offset);
                } else if offset >= 32 {
                    high = high.wrapping_add(code << (offset - 32));
                } else {
                    straddle = straddle.wrapping_add(u64::from(code) << offset);
                }
            }
            let packed = u64::from(low) | (u64::from(high) << 32) | straddle;
            let start = first + word * word_bytes;
            block[start..start + word_bytes].copy_from_slice(&packed.to_le_bytes()[..word_bytes]);
        }
    });
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

/// The magnitude bits of an `f32`: the sign cleared.
const MAGNITUDE_MASK: u32 = 0x7fff_ffff;

/// The magnitude bits of positive infinity: any magnitude above it is a nan.
const INFINITY_BITS: u32 = 0x7f80_0000;

/// `fold` of the magnitudes with the C `MAX` macro, in element order, whatever the elements are.
#[inline(always)]
fn peak_ordered<const ELEMENTS: usize>(input: &[f32; ELEMENTS]) -> f32 {
    // ggml-impl.h:40 MAX(a, b) is (a) > (b) ? (a) : (b), so a nan in the last element stays nan
    input.iter().fold(0.0f32, |peak, value| {
        let magnitude = value.abs();
        if peak > magnitude { peak } else { magnitude }
    })
}

/// [`peak_ordered`], with a reduction a vector unit runs when the block holds no nan. The bits of a
/// non-negative float order as the floats do and every nan's magnitude bits exceed infinity's, so
/// the unsigned max of the magnitude bits is the plain max of the magnitudes when it is at most
/// infinity's bits, and is the sign that a nan is somewhere in the block when it is not.
#[inline(always)]
fn peak<const ELEMENTS: usize>(input: &[f32; ELEMENTS]) -> f32 {
    let top = input.iter().fold(0u32, |top, value| top.max(value.to_bits() & MAGNITUDE_MASK));
    if top <= INFINITY_BITS { f32::from_bits(top) } else { peak_ordered(input) }
}

#[inline(always)]
fn inverse_of(scale: f32) -> f32 {
    if scale == 0.0 { 0.0 } else { 1.0 / scale }
}

/// The block's scale and the inverse the codes are fitted with (unused by a sign code).
#[inline(always)]
fn fit_scale<const ELEMENTS: usize>(layout: &BlockLayout, fit: Fit, input: &[f32; ELEMENTS]) -> (f32, f32) {
    match fit {
        Fit::SignedMax => {
            let scale = signed_max(input) / -(layout.element.bias as f32);
            (scale, inverse_of(scale))
        }
        Fit::PeakRound | Fit::Runtime => {
            let scale = peak(input) / reach_of(layout) as f32;
            (scale, inverse_of(scale))
        }
        Fit::MeanSign => {
            let mut sum = 0.0f32;
            for &value in input {
                sum += value.abs();
            }
            (sum / ELEMENTS as f32, 0.0)
        }
    }
}

/// The raw code of one value under the fit's `inverse`.
#[inline(always)]
fn code_of(layout: &BlockLayout, fit: Fit, inverse: f32, value: f32) -> u8 {
    match fit {
        Fit::SignedMax => {
            let top = (1i32 << code_bits(layout)) - 1;
            top.min((value * inverse + (layout.element.bias as f32 + 0.5)) as i32) as u8
        }
        Fit::PeakRound | Fit::Runtime => {
            let (lowest, highest) = integer_bounds(layout);
            let rounded = (libm::roundf(value * inverse) as i32).clamp(lowest, highest);
            raw_of_value(layout, rounded) as u8
        }
        Fit::MeanSign => u8::from(value >= 0.0),
    }
}

/// Fits the block's scale and writes the raw codes; returns the scale the block stores and, when
/// `HIGH_FUSED`, the word of the high plane's bits (code `lane`'s bits at `lane * bits`), summed in
/// the loop that produces the codes so a vector unit shifts each lane into place.
#[inline(always)]
fn fit_codes<const ELEMENTS: usize, const HIGH_FUSED: bool>(
    layout: &BlockLayout,
    fit: Fit,
    input: &[f32; ELEMENTS],
    codes: &mut [u8; ELEMENTS],
) -> (f32, u32) {
    let (scale, inverse) = fit_scale(layout, fit, input);
    let lift = u32::from(layout.element.low.bits);
    let bits = layout.element.high.map_or(0, |plane| u32::from(plane.bits));
    let mask = u32::MAX.checked_shr(32 - bits).unwrap_or(0);
    let mut word = 0u32;
    for (lane, (code, &value)) in codes.iter_mut().zip(input).enumerate() {
        *code = code_of(layout, fit, inverse, value);
        if HIGH_FUSED {
            word = word.wrapping_add(((u32::from(*code) >> lift) & mask).wrapping_shl(lane as u32 * bits));
        }
    }
    (scale, word)
}

#[inline(always)]
fn encode_chunk<
    const ELEMENTS: usize,
    const BYTES: usize,
    const LOW_BY_WORD: bool,
    const HIGH_BY_WORD: bool,
    const HIGH_FUSED: bool,
    const SCALE_OWNED: bool,
>(
    layout: &BlockLayout,
    fit: Fit,
    input: &[f32; ELEMENTS],
    out: &mut [u8; BYTES],
) {
    let mut block = [0u8; BYTES];
    let mut codes = [0u8; ELEMENTS];
    let (scale, high_word) = fit_codes::<ELEMENTS, HIGH_FUSED>(layout, fit, input, &mut codes);
    scatter_codes::<ELEMENTS, BYTES, LOW_BY_WORD, HIGH_BY_WORD, HIGH_FUSED>(layout, &codes, high_word, &mut block);
    let width = layout.scale.layout.width() as u32;
    put_field::<BYTES, SCALE_OWNED>(&mut block, layout.scale.byte_offset as usize, width, encode(scale, layout.scale.layout));
    *out = block;
}

#[inline(always)]
fn encode_run<
    const ELEMENTS: usize,
    const BYTES: usize,
    const LOW_BY_WORD: bool,
    const HIGH_BY_WORD: bool,
    const HIGH_FUSED: bool,
    const SCALE_OWNED: bool,
>(
    layout: BlockLayout,
    fit: Fit,
    values: &[f32],
    out: &mut [u8],
) {
    let inputs = values.as_chunks::<ELEMENTS>().0;
    let blocks = out.as_chunks_mut::<BYTES>().0;
    for (input, block) in inputs.iter().zip(blocks) {
        encode_chunk::<ELEMENTS, BYTES, LOW_BY_WORD, HIGH_BY_WORD, HIGH_FUSED, SCALE_OWNED>(&layout, fit, input, block);
    }
}

macro_rules! resolve_named_block {
    ($layout:expr, [ $(($name:ident, $text:literal)),* $(,)? ]) => {{
        $(
            if *$layout == BlockLayout::$name {
                const _: () = assert!(BlockLayout::$name.is_simple());
                const _: () = assert!(planes_cover(&BlockLayout::$name));
                const _: () = assert!(code_bits(&BlockLayout::$name) <= 8);
                const _: () = assert!(BlockLayout::$name.min.is_none() || is_integer_code(&BlockLayout::$name));
                const ELEMENTS: usize = BlockLayout::$name.elements as usize;
                const BYTES: usize = BlockLayout::$name.block_bytes as usize;
                const FIT: Fit = fit_of(&BlockLayout::$name);
                const LOW_BY_WORD: bool = is_word_packed(&BlockLayout::$name.element.low);
                const HIGH_BY_WORD: bool = is_high_word_packed(&BlockLayout::$name);
                const HIGH_FUSED: bool = is_high_single_word(&BlockLayout::$name);
                const SCALE_OWNED: bool = scale_owns_its_bytes(&BlockLayout::$name);
                const UNPACK_BY_WORD: bool = unpacks_by_word(&BlockLayout::$name);
                const IN_PLACE: bool = reads_codes_in_place(&BlockLayout::$name);
                const PAIRED: bool = pairs_halves_with_high_word(&BlockLayout::$name);
                let encode: Option<EncodeFn> = match FIT {
                    Fit::Runtime => None,
                    _ => Some(|values, out| {
                        encode_run::<ELEMENTS, BYTES, LOW_BY_WORD, HIGH_BY_WORD, HIGH_FUSED, SCALE_OWNED>(
                            BlockLayout::$name,
                            FIT,
                            values,
                            out,
                        );
                    }),
                };
                let decode: DecodeFn = |data, out| {
                    decode_run::<ELEMENTS, BYTES, UNPACK_BY_WORD, IN_PLACE, PAIRED>(BlockLayout::$name, data, out);
                };
                return Some(FormatCodec::table(encode, decode, ELEMENTS, BYTES));
            }
        )*
        None
    }};
}

/// The table instantiation for a layout in the block table, `None` for any other; the name in its
/// errors is set by [`FormatCodec::block`].
pub(crate) fn resolve_block(layout: &BlockLayout) -> Option<FormatCodec> {
    named_block_table!(resolve_named_block!(layout,))
}

/// The table instantiation for a [`NumericFormat`] whose packing is a table layout: a block format,
/// or a linear-ladder level format over one. `None` for scalars, codebooks and explicit tables.
pub(crate) fn block_for_format(format: &NumericFormat) -> Option<FormatCodec> {
    match format {
        NumericFormat::Block(layout) => resolve_block(layout),
        NumericFormat::Levels(levels) => match (levels.packing(), levels.levels) {
            (Some(layout), LevelTable::Linear { .. }) if levels.levels == layout.levels() => {
                resolve_block(&layout)
            }
            _ => None,
        },
        NumericFormat::Scalar(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_layout::{Coded, Digit};
    use crate::float_layout::{FloatLayout, NanPolicy, Specials};

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

        let (scale, _) = fit_codes::<32, false>(&layout, Fit::SignedMax, &input, &mut codes);

        assert_eq!(scale, 0.015_625);
        assert_eq!(&codes[..4], &[192, 0, 255, 128]);
        assert!(codes[3..].iter().all(|&code| code == 128));
    }

    fn block_of(first: f32, fill: f32) -> [f32; 32] {
        let mut block = [fill; 32];
        block[0] = first;
        block
    }

    fn kv_like_block(seed: u32) -> [f32; 32] {
        let mut state = seed;
        let mut block = [0.0f32; 32];
        for (index, slot) in block.iter_mut().enumerate() {
            let mut gaussian = -6.0f32;
            for _ in 0..12 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                gaussian += (state >> 8) as f32 / (1u32 << 24) as f32;
            }
            *slot = if matches!(index % 64, 5 | 37) { gaussian * 24.0 } else { gaussian };
        }
        block
    }

    fn random_bit_block(seed: u32) -> [f32; 32] {
        let mut state = seed;
        let mut block = [0.0f32; 32];
        for slot in &mut block {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *slot = f32::from_bits(state ^ (state << 13));
        }
        block
    }

    fn assert_peak_is_the_ordered_fold(block: &[f32; 32], what: &str) {
        assert_eq!(peak(block).to_bits(), peak_ordered(block).to_bits(), "{what}");
    }

    #[test]
    fn signed_max_is_the_first_largest_magnitude_ignoring_nans_and_keeping_zero_for_zeros() {
        let mut tie = block_of(-3.0, 1.0);
        tie[9] = 3.0;
        assert_eq!(signed_max(&tie).to_bits(), (-3.0f32).to_bits(), "the first of a tie wins");
        assert_eq!(signed_max(&[-0.0; 32]).to_bits(), 0.0f32.to_bits(), "all zeros keep the initial zero");
        let mut with_nan = block_of(f32::NAN, 2.0);
        with_nan[31] = -f32::NAN;
        assert_eq!(signed_max(&with_nan).to_bits(), 2.0f32.to_bits(), "a nan never wins");
        assert!(signed_max(&[f32::NAN; 32]) == 0.0, "only nans keep the initial zero");
        let mut infinite = kv_like_block(3);
        infinite[17] = f32::NEG_INFINITY;
        assert_eq!(signed_max(&infinite), f32::NEG_INFINITY);
    }

    #[test]
    fn peak_without_a_nan_is_the_largest_magnitude_whatever_its_sign_or_place() {
        let mut block = kv_like_block(7);
        block[11] = -97.25;
        block[30] = f32::NEG_INFINITY;
        assert_eq!(peak(&block), f32::INFINITY);
        block[30] = -0.0;
        assert_eq!(peak(&block).to_bits(), 97.25f32.to_bits());
        assert_eq!(peak(&[-0.0; 32]).to_bits(), 0.0f32.to_bits());
        assert_peak_is_the_ordered_fold(&block, "no nan");
    }

    #[test]
    fn peak_is_nan_when_the_last_element_is_nan() {
        let mut block = block_of(5.0, 1.0);
        block[31] = f32::NAN;

        assert!(peak(&block).is_nan());
        assert_peak_is_the_ordered_fold(&block, "nan last");
    }

    #[test]
    fn peak_restarts_after_the_last_nan_so_a_leading_nan_is_forgotten() {
        let block = block_of(f32::NAN, 1.0);
        assert_eq!(peak(&block), 1.0, "nan first, then ones");
        assert_peak_is_the_ordered_fold(&block, "nan first");

        let mut middle = block_of(100.0, 2.0);
        middle[15] = -f32::NAN;
        middle[16] = 3.0;
        assert_eq!(peak(&middle), 3.0, "the 100 before the nan and the 2s before it are forgotten");
        assert_peak_is_the_ordered_fold(&middle, "nan in the middle");
    }

    #[test]
    fn peak_with_two_nans_keeps_what_follows_the_second() {
        let mut block = block_of(9.0, 1.0);
        block[3] = f32::NAN;
        block[20] = f32::from_bits(0x7fa0_0001);
        block[25] = 4.0;
        block[26] = -6.5;

        assert_eq!(peak(&block), 6.5);
        assert_peak_is_the_ordered_fold(&block, "two nans");

        block[31] = f32::from_bits(0xffc0_1234);
        assert!(peak(&block).is_nan());
        assert_eq!(peak(&block).to_bits(), peak_ordered(&block).to_bits(), "the nan payload is the last one's");
    }

    #[test]
    fn peak_matches_the_ordered_fold_on_kv_like_and_random_bit_blocks() {
        let mut nan_blocks = 0usize;
        for seed in 1..=2000u32 {
            let kv = kv_like_block(seed);
            let random = random_bit_block(seed);
            assert_peak_is_the_ordered_fold(&kv, "kv like");
            assert_peak_is_the_ordered_fold(&random, "random bit patterns");
            nan_blocks += usize::from(random.iter().any(|value| value.is_nan()));
        }

        assert!(nan_blocks > 0, "the random bit patterns must reach the nan path");
    }

    #[test]
    fn run_planes_are_the_contiguous_byte_aligned_ones_whatever_the_format_is_called() {
        let runs = [
            BlockLayout::Q8_0.element.low,
            BlockLayout::Q2_0.element.low,
            BlockLayout::PQ2_0.element.low,
            BlockLayout::Q1_0.element.low,
            BlockLayout::Q5_0.element.high.expect("q5_0 has a high plane"),
        ];
        let not_runs = [BlockLayout::Q4_0.element.low, BlockLayout::Q5_0.element.low];

        assert!(runs.iter().all(is_run_plane));
        assert!(not_runs.iter().all(|plane| !is_run_plane(plane)));
        assert!(!is_word_packed(&BlockLayout::Q8_0.element.low), "a byte-wide code is not unpacked");
        assert!(unpacks_by_word(&BlockLayout::PQ2_0) && !unpacks_by_word(&BlockLayout::Q5_0));
        assert!(is_high_word_packed(&BlockLayout::Q5_0) && !is_high_word_packed(&BlockLayout::Q4_0));
        assert_eq!(word_codes(&BlockLayout::PQ2_0.element.low), 32);
        assert_eq!(word_codes(&BlockLayout::Q8_0.element.low), 8);
        assert_eq!(word_codes(&BlockLayout::Q5_0.element.high.expect("high plane")), 32);
    }

    #[test]
    fn a_run_that_is_not_a_whole_number_of_bytes_is_not_a_run_plane() {
        let mut plane = BlockLayout::Q2_0.element.low;
        plane.digits[0].radix = 3;

        assert!(!is_run_plane(&plane));
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
            let resolves = block_for_format(format).is_some();

            assert_eq!(resolves, format_should_resolve(format), "{name}");
            resolving += usize::from(resolves);
        }

        assert_eq!(resolving, RESOLVING_NAMED_FORMATS);
    }

    const ONE_RUN_DIGIT: Digit = Digit { radix: 1, stride_bits: 0 };

    /// q5_0-like, 64 elements: a 4-bit nibble-split low plane and a 1-bit high plane that is two
    /// 32-bit runs, so the high plane is two register words and not the one a fused fit can fill.
    const TWO_WORD_HIGH_PLANE: BlockLayout = BlockLayout {
        elements: 64,
        block_bytes: 42,
        element: Coded {
            low: Plane {
                bits: 4,
                base_bits: 80,
                digits: [Digit { radix: 32, stride_bits: 8 }, Digit { radix: 2, stride_bits: 4 }, ONE_RUN_DIGIT, ONE_RUN_DIGIT],
            },
            high: Some(Plane {
                bits: 1,
                base_bits: 16,
                digits: [Digit { radix: 32, stride_bits: 1 }, Digit { radix: 2, stride_bits: 32 }, ONE_RUN_DIGIT, ONE_RUN_DIGIT],
            }),
            ..BlockLayout::Q5_0.element
        },
        ..BlockLayout::Q5_0
    };

    /// q4_0 with a 12-bit scale, so the first code shares the scale's last byte.
    const SCALE_SHARES_A_BYTE: BlockLayout = BlockLayout {
        element: Coded {
            low: Plane {
                bits: 4,
                base_bits: 12,
                digits: [Digit { radix: 32, stride_bits: 4 }, ONE_RUN_DIGIT, ONE_RUN_DIGIT, ONE_RUN_DIGIT],
            },
            ..BlockLayout::Q4_0.element
        },
        scale: Field {
            layout: FloatLayout {
                exponent_bits: 5,
                mantissa_bits: 6,
                specials: Specials::InfinityOnOverflow,
                bias_override: None,
                sign: SignEncoding::Magnitude,
                nan: NanPolicy::IEEE,
            },
            byte_offset: 0,
        },
        ..BlockLayout::Q4_0
    };

    fn kv_like_block_of_64(seed: u32) -> [f32; 64] {
        let mut block = [0.0f32; 64];
        block[..32].copy_from_slice(&kv_like_block(seed));
        block[32..].copy_from_slice(&kv_like_block(seed + 1));
        block
    }

    #[test]
    fn a_high_plane_of_two_words_is_not_fused_and_encodes_as_the_generic_path_does() {
        const FIT: Fit = fit_of(&TWO_WORD_HIGH_PLANE);
        const LOW_BY_WORD: bool = is_word_packed(&TWO_WORD_HIGH_PLANE.element.low);
        const HIGH_BY_WORD: bool = is_high_word_packed(&TWO_WORD_HIGH_PLANE);
        const HIGH_FUSED: bool = is_high_single_word(&TWO_WORD_HIGH_PLANE);
        let input = kv_like_block_of_64(11);
        let mut chosen = [0u8; 42];
        let mut generic = [0u8; 42];
        let mut forced = [0u8; 42];

        encode_chunk::<64, 42, LOW_BY_WORD, HIGH_BY_WORD, HIGH_FUSED, true>(&TWO_WORD_HIGH_PLANE, FIT, &input, &mut chosen);
        encode_chunk::<64, 42, false, false, false, true>(&TWO_WORD_HIGH_PLANE, FIT, &input, &mut generic);
        encode_chunk::<64, 42, LOW_BY_WORD, HIGH_BY_WORD, true, true>(&TWO_WORD_HIGH_PLANE, FIT, &input, &mut forced);

        assert_ne!(FIT, Fit::Runtime, "the layout must reach the encode instantiation");
        const { assert!(!HIGH_FUSED && HIGH_BY_WORD, "two words of high bits are word-packed but not fused") };
        assert!(is_high_single_word(&BlockLayout::Q5_0), "q5_0's 32 high bits are the one word");
        assert_eq!(chosen, generic, "the instantiation the layout selects matches the per-element path");
        assert_ne!(forced, generic, "forcing the fuse on this layout is the bug the clause prevents");
    }

    #[test]
    fn a_scale_that_is_not_whole_bytes_keeps_the_code_bits_in_its_last_byte() {
        const SCALE_OWNED: bool = scale_owns_its_bytes(&SCALE_SHARES_A_BYTE);
        let layout = SCALE_SHARES_A_BYTE;
        let input = kv_like_block(5);
        let mut codes = [0u8; 32];
        let mut block = [0u8; 18];

        let (scale, _) = fit_codes::<32, false>(&layout, fit_of(&layout), &input, &mut codes);
        encode_chunk::<32, 18, false, false, false, SCALE_OWNED>(&layout, fit_of(&layout), &input, &mut block);

        const { assert!(!SCALE_OWNED, "the first code lies in the scale's last byte") };
        assert!(scale_owns_its_bytes(&BlockLayout::Q4_0) && scale_owns_its_bytes(&BlockLayout::Q5_0));
        assert_ne!(fit_of(&layout), Fit::Runtime, "the layout must reach the encode instantiation");
        assert_ne!(codes[0], 0, "the first code must be nonzero to be lost");
        assert_eq!(read_bits(&block, 0, 12), encode(scale, layout.scale.layout) & 0xfff, "the 12 scale bits");
        assert_eq!(gather_codes::<32, 18>(&layout, &block), codes, "every code, the first one beside the scale");
    }

    #[test]
    fn only_a_nibble_plane_with_one_bit_word_above_it_pairs_halves() {
        assert!(pairs_halves_with_high_word(&BlockLayout::Q5_0) && pairs_halves_with_high_word(&BlockLayout::Q5_1));
        for layout in [BlockLayout::Q4_0, BlockLayout::Q4_1, BlockLayout::Q8_0, BlockLayout::Q2_0, BlockLayout::TQ2_0] {
            assert!(!pairs_halves_with_high_word(&layout), "{layout:?} has no one-bit word above a nibble plane");
        }
    }

    #[test]
    fn each_clause_of_is_simple_has_a_layout_that_does_not_resolve() {
        assert!(!BlockLayout::Q4_K.is_simple() && BlockLayout::Q4_K.sub.is_some());
        assert!(!BlockLayout::NVFP4.is_simple() && BlockLayout::NVFP4.scale_runs.is_some());
        assert!(!BlockLayout::TQ1_0.is_simple() && BlockLayout::TQ1_0.element.packed_radix > 1);
        for layout in [BlockLayout::Q4_K, BlockLayout::NVFP4, BlockLayout::TQ1_0] {
            assert!(FormatCodec::block(&layout).is_none());
        }
    }
}
