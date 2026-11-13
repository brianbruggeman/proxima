//! The inverse of [`crate::decode`]: `f32` values into any format this crate can describe.
//!
//! Reading is exact and total; writing is quantization, so every arm either produces bytes or
//! names what it cannot express. The matrix this module fills:
//!
//! - `Scalar(layout)` — one value per element, through [`crate::float_layout_codec::encode`],
//!   whose rounding and special-value policy are the format's own.
//! - `Block(layout)` with integer codes — a run of codes under a shared absolute-maximum scale,
//!   fitted per run and written back through the layout's own planes. A layout that
//!   [`FormatCodec::for_format`] resolves with [`FormatCodec::can_encode`] is encoded by that codec
//!   ([`crate::block_codec`], whose bytes match llama.cpp's `quantize_row_*_ref` for Q4_0, Q5_0,
//!   Q8_0, Q8_1 and TQ2_0); only the layouts the table cannot encode run the walker in this module.
//!   `proxima-gguf`'s `block_layout_encode` is a thin adapter over [`encode_blocks`].
//! - `Block(layout)` with float codes — the OCP MX and NV `E2M1` fields and the FP6 `E3M2` field:
//!   the code is a small float, so the nearest code is the nearest representable value, chosen
//!   through [`crate::float_layout_codec::encode`] rather than by rounding a ratio.
//! - mixed-radix packings (`TQ1_0`, `PTQ1_0`) — a code is a base-3 lane of a shared storage word,
//!   so the writer assembles the whole word and stores it once; `pack_digits` is the exact inverse
//!   of the decoder's lane extraction, checked over every digit string the packings can produce.
//! - `Levels` with a linear ladder (ternary) — the same machinery as an integer code: a trit *is*
//!   a narrow integer code under the layout's bias and step.
//! - `Levels(Explicit)` — IQ4_NL's trained 16-entry ladder: the scale is fitted so the block peak
//!   lands on the largest table magnitude, and every code is the nearest table entry.
//!
//! The census test pins which of the 36 formats land in which arm. Two families refuse, each
//! because one shared absolute-maximum scale is the wrong model rather than because the code is
//! missing: the K-quants (a sub-block scale and a stored min per sub-block, chosen by a
//! quantization-aware search) and the trained codebooks (`IQ1_*`..`IQ4_XS`, whose tables the
//! walker does not carry).
//!
//! The walker divides each value by the scale already rounded to its stored width; llama.cpp
//! multiplies by the inverse of the unrounded scale (and Q4_0 and Q5_0 fit the signed maximum, not
//! the peak). The two differ by a code on some elements, which is why a layout with a codec never
//! takes the walker: the walker is the encoder of last resort, not a second encoder.

use crate::block_layout::{BlockLayout, Plane, PlaneCursor};
use crate::convert::{Overflow, Rounding};
use crate::error::NumericError;
use crate::float_layout::FloatLayout;
use crate::float_layout_codec;
use crate::format::{LevelTable, NumericFormat};
use crate::format_codec::FormatCodec;

const MAX_CODE_BITS: u8 = 12;
const SUM_BYTES: usize = 2;

/// Which writer a layout's element codes need.
#[derive(Debug, Clone, Copy)]
enum Codes {
    /// Narrow integers under the layout's multiplier and bias: every one-, two-, four- and
    /// eight-bit code this workspace stores.
    Integer { lowest: i32, highest: i32 },
    /// A float layout's own values: the OCP MX and NV `E2M1` and FP6 `E3M2` fields, whose code is
    /// a small float rather than an integer, so the nearest code is the nearest *representable
    /// value* and not a rounded ratio.
    Float,
}

/// A layout's code range and value mapping, read once per run of blocks.
#[derive(Debug, Clone, Copy)]
struct Linear {
    codes: Codes,
    code_layout: FloatLayout,
    reach: f32,
    multiplier: f32,
    bias: f32,
    sum_gap: Option<usize>,
}

fn code_bits(layout: &BlockLayout) -> u8 {
    layout.element.low.bits + layout.element.high.map_or(0, |plane| plane.bits)
}

/// How many codes a layout holds: a mixed-radix packing names its radix, every other code its
/// plane width.
fn code_space(layout: &BlockLayout) -> u32 {
    if layout.element.packed_radix > 1 {
        u32::from(layout.element.packed_radix)
    } else {
        1u32 << code_bits(layout)
    }
}

/// The lowest and highest value a code stands for, over the layout's whole code space. Codes that
/// decode to a non-finite value are not levels and are skipped.
fn code_value_range(layout: &BlockLayout) -> (f32, f32) {
    (0..code_space(layout)).fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(lowest, highest), raw| {
            let value = float_layout_codec::decode(raw, layout.element.code);
            if value.is_finite() {
                (lowest.min(value), highest.max(value))
            } else {
                (lowest, highest)
            }
        },
    )
}

fn check_linear(layout: &BlockLayout) -> Result<bool, &'static str> {
    if layout.sub.is_some() {
        return Err("sub-blocks carry their own scales");
    }
    if layout.min.is_some() {
        return Err("a stored min is not an absolute-maximum scale");
    }
    if layout.element.packed_radix > 1 && layout.runs() > 1 {
        return Err("mixed-radix codes spread over several scale runs");
    }
    if code_bits(layout) > MAX_CODE_BITS {
        return Err("element codes wider than the encoder walks");
    }
    if layout.element.code.exponent_bits == 0 {
        return Ok(false);
    }
    float_layout_codec::layout_supported(&layout.element.code)
        .map(|()| true)
        .map_err(|_| "element codes outside the scalar codec")
}

fn touches(plane: &Plane, elements: usize, bytes: core::ops::Range<usize>) -> bool {
    (0..elements).any(|index| bytes.contains(&(plane.address(index) / 8)))
}

/// llama's `Q8_1` carries `s = d * sum(q)` in the two bytes after the scale; a layout that leaves
/// those bytes free -- no code there, and no other run's scale there -- is the one place a
/// consumer reads it.
fn sum_slot(layout: &BlockLayout) -> Option<usize> {
    let gap = usize::from(layout.scale.layout.width()).div_ceil(8);
    let elements = usize::from(layout.elements);
    let planes = [Some(layout.element.low), layout.element.high];
    let free = (0..layout.runs()).all(|run| {
        let start = layout.run_scale_byte(run) + gap;
        let bytes = start..start + SUM_BYTES;
        bytes.end <= usize::from(layout.block_bytes)
            && planes
                .iter()
                .flatten()
                .all(|plane| !touches(plane, elements, bytes.clone()))
            && (0..layout.runs()).all(|other| {
                let at = layout.run_scale_byte(other);
                bytes.end <= at || bytes.start >= at + gap
            })
    });
    free.then_some(gap)
}

fn linear(layout: &BlockLayout) -> Result<Linear, &'static str> {
    let float_codes = check_linear(layout)?;
    let (lowest_code, highest_code) = code_value_range(layout);
    let multiplier = f32::from(layout.multiplier);
    let bias = f32::from(layout.element.bias);
    let value_low = multiplier * lowest_code - bias;
    let value_high = multiplier * highest_code - bias;
    let reach = value_high.min(-value_low);
    if reach <= 0.0 {
        return Err("no code magnitude on both sides of zero");
    }
    let codes = if float_codes {
        Codes::Float
    } else {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a narrow integer code decodes to its own integer, so the walk is exact"
        )]
        Codes::Integer {
            lowest: lowest_code as i32,
            highest: highest_code as i32,
        }
    };
    Ok(Linear {
        codes,
        code_layout: layout.element.code,
        reach,
        multiplier,
        bias,
        sum_gap: sum_slot(layout),
    })
}

impl Linear {
    fn code_for(&self, value: f32, scale: f32, (lowest, highest): (i32, i32)) -> i32 {
        let scaled = if scale == 0.0 { 0.0 } else { value / scale };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the quotient is clamped into the code range immediately after"
        )]
        let code = libm::roundf((scaled + self.bias) / self.multiplier) as i32;
        code.clamp(lowest, highest)
    }

    fn value_of(&self, code: i32) -> f32 {
        self.multiplier * code as f32 - self.bias
    }
}

fn put_bits(plane: &Plane, cursor: &mut PlaneCursor, bits: u32, block: &mut [u8]) {
    put_bits_at(plane.bits, cursor.address(), bits, block);
    cursor.advance();
}

fn put_bits_at(width: u8, at: usize, bits: u32, block: &mut [u8]) {
    let mask = (1u32 << width) - 1;
    let mut value = bits & mask;
    let mut address = at;
    let mut written = 0u32;
    while written < u32::from(width) {
        let within = (address % 8) as u32;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a byte-wide write of the low eight bits"
        )]
        {
            block[address / 8] |= (value << within) as u8;
        }
        value >>= 8 - within;
        written += 8 - within;
        address += usize::from(8 - within as u8);
    }
}

fn cursors_of(layout: &BlockLayout) -> [PlaneCursor; 2] {
    let mut cursors = [layout.element.low.cursor(), layout.element.low.cursor()];
    if let Some(high) = &layout.element.high {
        cursors[1] = high.cursor();
    }
    cursors
}

fn put_code(layout: &BlockLayout, cursors: &mut [PlaneCursor; 2], raw: u32, block: &mut [u8]) {
    let element = &layout.element;
    put_bits(&element.low, &mut cursors[0], raw, block);
    if let Some(high) = &element.high {
        put_bits(high, &mut cursors[1], raw >> element.low.bits, block);
    }
}

fn put_scale(layout: &BlockLayout, offset: usize, code: u32, block: &mut [u8]) {
    let bytes = usize::from(layout.scale.layout.width().div_ceil(8));
    for byte in 0..bytes {
        block[offset + byte] = (code >> (8 * byte)) as u8;
    }
}

/// The scale a block's peak is stored at, read back so the codes round against the value the
/// decoder will use.
///
/// A scale with a mantissa rounds to the nearest stored value; that is the incumbent's rule and it
/// keeps the top element within one code step. A scale with **no** mantissa (`E8M0`, one bit of
/// resolution) cannot: rounding `peak / reach` down to the nearest power of two is a 41% error and
/// the peak then clamps to the largest code, so the quotient is rounded *away from zero* instead
/// and the block peak always lands inside the code range.
fn stored_scale(peak: f32, reach: f32, layout: FloatLayout) -> (u32, f32) {
    let rounding = if layout.mantissa_bits == 0 {
        Rounding::AwayFromZero
    } else {
        Rounding::NearestTiesEven
    };
    let mut nan_count = 0;
    let code = float_layout_codec::encode_with(
        peak / reach,
        layout,
        rounding,
        Overflow::Format,
        &mut nan_count,
    );
    (code, float_layout_codec::decode(code, layout))
}

fn quantize_run(
    layout: &BlockLayout,
    shape: &Linear,
    bounds: (i32, i32),
    run: u16,
    input: &[f32],
    cursors: &mut [PlaneCursor; 2],
    block: &mut [u8],
) {
    let peak = input
        .iter()
        .fold(0.0f32, |peak, value| peak.max(value.abs()));
    let (scale_code, scale) = stored_scale(peak, shape.reach, layout.scale.layout);
    let scale_at = layout.run_scale_byte(run);
    put_scale(layout, scale_at, scale_code, block);
    let mut value_sum = 0.0f32;
    for value in input {
        let code = shape.code_for(*value, scale, bounds);
        put_code(
            layout,
            cursors,
            float_layout_codec::encode(code as f32, layout.element.code),
            block,
        );
        value_sum += shape.value_of(code);
    }
    if let Some(gap) = shape.sum_gap {
        let at = scale_at + gap;
        let code = float_layout_codec::encode(scale * value_sum, FloatLayout::F16);
        block[at..at + SUM_BYTES].copy_from_slice(&(code as u16).to_le_bytes());
    }
}

fn quantize_block(
    layout: &BlockLayout,
    shape: &Linear,
    input: &[f32],
    block: &mut [u8],
) -> Result<(), NumericError> {
    if layout.element.packed_radix > 1 {
        return quantize_radix_block(layout, shape, input, block);
    }
    if matches!(shape.codes, Codes::Float) {
        let code_layout = shape.code_layout;
        return quantize_coded_block(layout, shape.reach, input, block, |target| {
            float_layout_codec::encode(target, code_layout)
        });
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(NumericError::NonFinite {
            format: format_of(layout),
        });
    }
    let Codes::Integer { lowest, highest } = shape.codes else {
        return Err(NumericError::UnsupportedEncode {
            format: format_of(layout),
            reason: "a non-integer code reached the integer writer",
        });
    };
    block.fill(0);
    let mut cursors = cursors_of(layout);
    for (run, group) in input
        .chunks(usize::from(layout.run_elements()))
        .enumerate()
    {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a block holds at most 32 runs and each is at most 256 elements"
        )]
        quantize_run(
            layout,
            shape,
            (lowest, highest),
            run as u16,
            group,
            &mut cursors,
            block,
        );
    }
    Ok(())
}

/// The most lanes a shared storage word carries (`TQ1_0` packs five trits into one byte).
const MAX_LANES: usize = 8;

/// The digits a storage word holds, read exactly as [`crate::decode`] reads them.
fn unpack_digits(word: u32, lanes: usize, radix: u32, bits: u8, out: &mut [u32]) {
    let span = (1u32 << bits) - 1;
    for (lane, slot) in out.iter_mut().take(lanes).enumerate() {
        let factor = radix.pow(lane as u32);
        let residue = word.wrapping_mul(factor) & span;
        *slot = (residue * radix) >> bits;
    }
}

/// The storage word whose lanes unpack to `digits`, or `None` when no byte of the word space
/// carries exactly this digit string.
///
/// The ideal word is the base-`radix` integer scaled into `bits`; the floor of that can lose the
/// last lane, so the candidates around it are tried and the first exact one wins. The exhaustive
/// test covers every digit string the workspace's packings can produce.
fn pack_digits(digits: &[u32], radix: u32, bits: u8) -> Option<u32> {
    let span = 1u32 << bits;
    let lanes = digits.len();
    let total = radix.checked_pow(lanes as u32)?;
    let mut integer = 0u32;
    for digit in digits {
        integer = integer * radix + *digit;
    }
    let base = integer.saturating_mul(span) / total;
    let mut candidate = base.saturating_sub(1);
    let mut unpacked = [0u32; MAX_LANES];
    while candidate <= base + 2 && candidate < span {
        unpack_digits(candidate, lanes, radix, bits, &mut unpacked);
        if unpacked[..lanes] == *digits {
            return Some(candidate);
        }
        candidate += 1;
    }
    None
}

/// A block whose codes share storage words: `TQ1_0` and `PTQ1_0` pack five base-3 digits into one
/// byte, so a code is a lane of a word rather than a field of its own. One scale per block, the
/// same absolute-maximum rule as every other block format.
fn quantize_radix_block(
    layout: &BlockLayout,
    shape: &Linear,
    input: &[f32],
    block: &mut [u8],
) -> Result<(), NumericError> {
    if input.iter().any(|value| !value.is_finite()) {
        return Err(NumericError::NonFinite {
            format: format_of(layout),
        });
    }
    block.fill(0);
    let peak = input
        .iter()
        .fold(0.0f32, |peak, value| peak.max(value.abs()));
    let (scale_code, scale) = stored_scale(peak, shape.reach, layout.scale.layout);
    put_scale(layout, layout.run_scale_byte(0), scale_code, block);
    let Codes::Integer { lowest, highest } = shape.codes else {
        return Err(NumericError::UnsupportedEncode {
            format: format_of(layout),
            reason: "a non-integer code reached the mixed-radix writer",
        });
    };
    let bounds = (lowest, highest);
    let coded = &layout.element;
    let radix = u32::from(coded.packed_radix);
    for region in coded
        .regions
        .iter()
        .take(usize::from(coded.region_count))
    {
        let axis = usize::from(region.digit_axis);
        let lanes = usize::from(region.source.digits[axis].radix);
        let before = region
            .source
            .digits
            .iter()
            .take(axis)
            .fold(1usize, |product, digit| product * usize::from(digit.radix));
        let start = usize::from(region.start);
        let elements = usize::from(region.elements);
        for word in 0..before.min(elements) {
            let mut digits = [0u32; MAX_LANES];
            let mut present = 0usize;
            for (lane, slot) in digits.iter_mut().take(lanes).enumerate() {
                let local = word + lane * before;
                if local >= elements {
                    break;
                }
                let code = shape.code_for(input[start + local], scale, bounds);
                *slot = float_layout_codec::encode(code as f32, coded.code);
                present += 1;
            }
            let Some(packed) = pack_digits(&digits[..present], radix, region.source.bits) else {
                return Err(NumericError::UnsupportedEncode {
                    format: format_of(layout),
                    reason: "a packed-radix word with no exact storage",
                });
            };
            put_bits_at(region.source.bits, region.source.address(word), packed, block);
        }
    }
    Ok(())
}

/// A block whose code is picked from a discrete set rather than computed as a ratio: a trained
/// ladder (`IQ4_NL`), or a float element layout (`E2M1`, `E3M2`). The scale places the block peak
/// on `reach`, then `choose` maps the scaled value to its code.
fn quantize_coded_block(
    layout: &BlockLayout,
    reach: f32,
    input: &[f32],
    block: &mut [u8],
    choose: impl Fn(f32) -> u32,
) -> Result<(), NumericError> {
    if input.iter().any(|value| !value.is_finite()) {
        return Err(NumericError::NonFinite {
            format: format_of(layout),
        });
    }
    block.fill(0);
    let mut cursors = cursors_of(layout);
    for (run, group) in input
        .chunks(usize::from(layout.run_elements()))
        .enumerate()
    {
        let peak = group
            .iter()
            .fold(0.0f32, |peak, value| peak.max(value.abs()));
        let (scale_code, scale) = stored_scale(peak, reach, layout.scale.layout);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a block holds at most 32 runs"
        )]
        let scale_at = layout.run_scale_byte(run as u16);
        put_scale(layout, scale_at, scale_code, block);
        for value in group {
            let target = if scale == 0.0 { 0.0 } else { value / scale };
            put_code(layout, &mut cursors, choose(target), block);
        }
    }
    Ok(())
}

fn format_of(layout: &BlockLayout) -> &'static str {
    NumericFormat::ALL
        .iter()
        .find(|format| format.block_layout().as_ref() == Some(layout))
        .and_then(NumericFormat::name)
        .unwrap_or("block")
}

/// The largest value magnitude `layout` reaches on both sides of zero, in units of its scale: the
/// divisor of the block maximum, and the test of whether this module encodes `layout` at all.
///
/// # Errors
/// [`NumericError::UnsupportedEncode`] naming what the layout has that a shared absolute-maximum
/// scale cannot express.
pub fn reach(layout: &BlockLayout) -> Result<f32, NumericError> {
    linear(layout)
        .map(|shape| shape.reach)
        .map_err(|reason| NumericError::UnsupportedEncode {
            format: format_of(layout),
            reason,
        })
}

fn packed_geometry(
    format: &NumericFormat,
    values: &[f32],
    out: &[u8],
) -> Result<(usize, usize, usize), NumericError> {
    let elements = usize::from(format.elements_per_block());
    let block_bytes = usize::from(format.bytes_per_block());
    if elements == 0 || block_bytes == 0 {
        return Err(NumericError::InvalidGeometry {
            format: format.name().unwrap_or("unnamed"),
            reason: "a format with no elements or bytes",
        });
    }
    if !values.len().is_multiple_of(elements) {
        return Err(NumericError::InputNotElementMultiple {
            format: format.name().unwrap_or("unnamed"),
            found: values.len(),
            elements,
        });
    }
    let block_count = values.len() / elements;
    let expected = block_count * block_bytes;
    if out.len() != expected {
        return Err(NumericError::OutputSizeMismatch {
            found: out.len(),
            expected,
        });
    }
    crate::decode::geometry_fits(format)?;
    Ok((elements, block_bytes, block_count))
}

// a scalar's geometry is its layout, so a named layout needs no geometry check beyond encode_slice's own
#[inline]
fn encode_scalar_blocks(
    format: &NumericFormat,
    layout: &FloatLayout,
    values: &[f32],
    out: &mut [u8],
) -> Result<usize, NumericError> {
    float_layout_codec::encode_slice(layout, values, out)
        .map_err(|fault| fault.into_error(layout, format.name().unwrap_or("unnamed")))?;
    Ok(values.len())
}

// the codec writes into the front of a window and takes any value; this entry keeps its contract of an
// exact-size output and a typed refusal of a nan or infinity
fn encode_with_codec(
    format: &NumericFormat,
    codec: &FormatCodec,
    values: &[f32],
    out: &mut [u8],
) -> Result<usize, NumericError> {
    let (_elements, _block_bytes, block_count) = packed_geometry(format, values, out)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(NumericError::NonFinite {
            format: format.block_layout().as_ref().map_or("block", format_of),
        });
    }
    codec.encode(values, out)?;
    Ok(block_count)
}

/// Encode `values` into `format`, writing `bytes_per_block` bytes per `elements_per_block` values,
/// and returns the block count.
///
/// One value per block for a scalar format; a whole block of `f32` for every block and level-set
/// format whose geometry a shared absolute-maximum scale can express.
///
/// # Errors
/// [`NumericError::InputNotElementMultiple`] when `values` is not a whole number of blocks,
/// [`NumericError::OutputSizeMismatch`] when `out` is not exactly the packed byte count,
/// [`NumericError::UnsupportedEncode`] for a format this module cannot quantize, and
/// [`NumericError::NonFinite`] for a NaN or infinite block element.
pub fn encode_blocks(
    format: &NumericFormat,
    values: &[f32],
    out: &mut [u8],
) -> Result<usize, NumericError> {
    if !matches!(format, NumericFormat::Scalar(_))
        && let Some(codec) = FormatCodec::for_format(format).filter(FormatCodec::can_encode)
    {
        return encode_with_codec(format, &codec, values, out);
    }
    match format {
        NumericFormat::Scalar(layout) => encode_scalar_blocks(format, layout, values, out),
        NumericFormat::Block(layout) => {
            let (elements, block_bytes, block_count) = packed_geometry(format, values, out)?;
            let shape = linear(layout).map_err(|reason| NumericError::UnsupportedEncode {
                format: format.name().unwrap_or("block"),
                reason,
            })?;
            for (group, block) in values.chunks(elements).zip(out.chunks_mut(block_bytes)) {
                quantize_block(layout, &shape, group, block)?;
            }
            Ok(block_count)
        }
        NumericFormat::Levels(levels) => {
            let (elements, block_bytes, block_count) = packed_geometry(format, values, out)?;
            let name = format.name().unwrap_or("levels");
            let Some(layout) = levels.packing() else {
                return Err(NumericError::UnsupportedEncode {
                    format: name,
                    reason: "a trained codebook needs a search over its table",
                });
            };
            match levels.levels {
                LevelTable::Linear { .. } => {
                    let shape = linear(&layout).map_err(|reason| NumericError::UnsupportedEncode {
                        format: name,
                        reason,
                    })?;
                    for (group, block) in values.chunks(elements).zip(out.chunks_mut(block_bytes))
                    {
                        quantize_block(&layout, &shape, group, block)?;
                    }
                    Ok(block_count)
                }
                LevelTable::Explicit(_) if layout.sub.is_some() => {
                    Err(NumericError::UnsupportedEncode {
                        format: name,
                        reason: "this ladder sits under sub-block scales the encoder does not fit",
                    })
                }
                LevelTable::Explicit(table) => {
                    let reach = ladder_reach(table);
                    for (group, block) in values.chunks(elements).zip(out.chunks_mut(block_bytes))
                    {
                        quantize_coded_block(&layout, reach, group, block, |target| {
                            nearest_level(table, target)
                        })?;
                    }
                    Ok(block_count)
                }
                LevelTable::SignedGrid { .. }
                | LevelTable::SignedGrid64 { .. }
                | LevelTable::DeltaGrid { .. } => {
                    Err(NumericError::UnsupportedEncode {
                        format: name,
                        reason: "this codebook has no inverse in this crate",
                    })
                }
            }
        }
    }
}

/// The magnitude a trained ladder reaches on both sides of zero, in table units.
fn ladder_reach(values: &[i8]) -> f32 {
    let low = values.iter().copied().min().unwrap_or(0) as f32;
    let high = values.iter().copied().max().unwrap_or(0) as f32;
    high.min(-low).max(1.0)
}

fn nearest_level(values: &[i8], target: f32) -> u32 {
    let mut best = 0usize;
    let mut best_distance = f32::INFINITY;
    for (index, level) in values.iter().enumerate() {
        let distance = (f32::from(*level) - target).abs();
        if distance < best_distance {
            best_distance = distance;
            best = index;
        }
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a trained ladder has at most 256 entries"
    )]
    {
        best as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_layout::BlockLayout;
    use crate::decode::decode_blocks;

    /// half of a binary16 ulp at 1.0: every stored scale is a binary16, so the scale itself is
    /// only known to this much and every value inherits that relative error
    const F16_EPSILON: f32 = 1.0 / 1024.0;

    fn values(count: usize, seed: u64) -> std::vec::Vec<f32> {
        let mut state = seed | 1;
        (0..count)
            .map(|index| {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut mixed = state;
                mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                mixed ^= mixed >> 31;
                let unit = (mixed >> 11) as f32 / (1u64 << 53) as f32;
                let signed = unit * 2.0 - 1.0;
                4.0 * signed * (1.0 + (index % 7) as f32)
            })
            .collect()
    }

    fn packed(format: &NumericFormat, input: &[f32]) -> std::vec::Vec<u8> {
        let elements = usize::from(format.elements_per_block());
        let bytes = usize::from(format.bytes_per_block());
        let mut out = std::vec![0u8; input.len() / elements * bytes];
        encode_blocks(format, input, &mut out).expect("a supported format encodes");
        out
    }

    fn decoded(format: &NumericFormat, packed: &[u8]) -> std::vec::Vec<f32> {
        let elements = usize::from(format.elements_per_block());
        let bytes = usize::from(format.bytes_per_block());
        let mut out = std::vec![0.0f32; packed.len() / bytes * elements];
        decode_blocks(format, packed, &mut out).expect("the bytes this encoder wrote decode");
        out
    }

    /// The scale the decoder reads back, for the formats whose scale is a binary16.
    fn grid_scale(peak: f32, reach: f32) -> f32 {
        stored_scale(peak, reach, FloatLayout::F16).1
    }

    fn peak_of(block: &[f32]) -> f32 {
        block.iter().fold(0.0f32, |peak, value| peak.max(value.abs()))
    }

    #[test]
    fn a_scalar_format_round_trips_every_code_its_layout_holds() {
        let format = NumericFormat::BFloat16;
        let input: std::vec::Vec<f32> = (0..512)
            .map(|step| crate::float_layout_codec::decode(step % 0x8000, FloatLayout::BF16))
            .collect();
        let bytes = packed(&format, &input);
        assert_eq!(bytes.len(), 512 * 2, "one binary16 per value");
        for (original, restore) in input.iter().zip(decoded(&format, &bytes)) {
            if original.is_nan() {
                assert!(restore.is_nan(), "a stored nan decodes to a nan");
            } else {
                assert_eq!(original.to_bits(), restore.to_bits());
            }
        }
    }

    #[test]
    fn q4_0_round_trips_within_half_a_code_of_its_block_peak() {
        let format = NumericFormat::Q4_0;
        let input = values(32 * 5, 0x5eed);
        let stated = reach(&BlockLayout::Q4_0).expect("q4_0 is a shared-scale layout");
        let restored = decoded(&format, &packed(&format, &input));
        for (originals, decodes) in input.chunks(32).zip(restored.chunks(32)) {
            let peak = peak_of(originals);
            let bound = (peak / stated) / 2.0 + 2.0 * peak * F16_EPSILON;
            for (original, restore) in originals.iter().zip(decodes) {
                assert!(
                    (original - restore).abs() <= bound,
                    "{original} decoded to {restore}, bound {bound}"
                );
            }
        }
    }

    #[test]
    fn encoding_a_decoded_q4_0_run_reproduces_its_own_bytes() {
        let format = NumericFormat::Q4_0;
        let first = packed(&format, &values(32 * 4, 0xfeed));
        let second = packed(&format, &decoded(&format, &first));
        assert_eq!(first, second, "the quantizer is idempotent on its own grid");
    }

    #[test]
    fn ternary_round_trips_onto_its_three_levels() {
        let format = NumericFormat::Tq20;
        let input = values(256 * 2, 0x7a1);
        let restored = decoded(&format, &packed(&format, &input));
        for (originals, decodes) in input.chunks(256).zip(restored.chunks(256)) {
            let peak = peak_of(originals);
            let bound = peak / 2.0 + 2.0 * peak * F16_EPSILON;
            let scale = grid_scale(peak, 1.0);
            for (original, restore) in originals.iter().zip(decodes) {
                assert!(
                    (original - restore).abs() <= bound,
                    "{original} decoded to {restore}, bound {bound}"
                );
                let units = restore / scale;
                assert!(
                    (units - units.round()).abs() < 1.0e-5,
                    "a trit lands on a whole level, not {units}"
                );
            }
        }
    }

    #[test]
    fn iq4_nl_round_trips_through_its_trained_ladder() {
        let format = NumericFormat::Iq4Nl;
        let input = values(32 * 4, 0x1a4);
        let table = crate::tables::KVALUES_IQ4_NL;
        let restored = decoded(&format, &packed(&format, &input));
        for (originals, decodes) in input.chunks(32).zip(restored.chunks(32)) {
            let peak = peak_of(originals);
            let scale = grid_scale(peak, ladder_reach(&table));
            let bound = 12.5 * scale + 2.0 * peak * F16_EPSILON;
            for (original, restore) in originals.iter().zip(decodes) {
                assert!(
                    (original - restore).abs() <= bound,
                    "{original} decoded to {restore}, bound {bound}"
                );
                let units = restore / scale;
                assert!(
                    table
                        .iter()
                        .any(|level| (f32::from(*level) - units).abs() < 1.0e-5),
                    "a code lands on a trained level, not {units}"
                );
            }
        }
    }

    /// The OCP MX v1.0 `E2M1` magnitude grid (§5.3.2 Table 4), ascending: the code for each is the
    /// index, and the sign bit is code 8.
    const E2M1_GRID: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

    #[test]
    fn mxfp4_writes_the_ocp_mx_block_byte_for_byte() {
        // one shared E8M0 scale byte (127 = 2^0), then element i in the low nibble of byte 1+(i%16)
        // and element i+16 in its high nibble; the grid is its own code, so the block is the codes
        // in order
        let mut input = std::vec![0.0f32; 32];
        for (index, value) in E2M1_GRID.iter().enumerate() {
            input[index] = *value;
            input[index + 8] = -*value;
        }
        let mut block = std::vec![0u8; 17];
        encode_blocks(&NumericFormat::Mxfp4, &input, &mut block).expect("mxfp4 encodes");
        let mut expected = std::vec![127u8];
        expected.extend(0u8..16);
        assert_eq!(block, expected, "OCP MX E2M1 grid at scale 2^0");
    }

    #[test]
    fn nvfp4_writes_one_e4m3_scale_per_sixteen_and_six_bit_fields_do_not_straddle() {
        let mut input = std::vec![0.0f32; 64];
        for (index, value) in E2M1_GRID.iter().enumerate() {
            input[index] = *value;
            input[index + 8] = -*value;
        }
        let mut block = std::vec![0u8; 36];
        encode_blocks(&NumericFormat::Nvfp4, &input, &mut block).expect("nvfp4 encodes");
        // E4M3FN(1.0) is code 56, and the three runs above zero keep a zero scale
        let mut expected = std::vec![56u8, 0, 0, 0];
        expected.extend(0u8..16);
        expected.extend(std::iter::repeat_n(0u8, 16));
        assert_eq!(block, expected, "one E4M3 scale per 16 elements");
    }

    #[test]
    fn fp6_e3m2_writes_six_bit_fields_across_byte_boundaries() {
        // 28 is E3M2's largest magnitude (2^4 * 1.75, §5.3.2 Table 4), so a block of ones sits on
        // the top code 0b011111; six-bit fields do not divide a byte, so laid LSB-first they repeat
        // with period 3 bytes: byte0 111_01111, byte1 11_01111_1, byte2 1_01111_11
        let input = std::vec![1.0f32; 32];
        let mut block = std::vec![0u8; 26];
        encode_blocks(&NumericFormat::Fp6E3M2, &input, &mut block).expect("fp6 e3m2 encodes");
        let scale = uint16_bytes(&block[0..2]);
        let straddled: std::vec::Vec<u8> = (0..24).map(|step| [0xdf, 0xf7, 0x7d][step % 3]).collect();
        assert_eq!(&block[2..], &straddled[..], "six-bit top codes across byte borders");
        let mut wide = std::vec![0.0f32; 32];
        decode_blocks(&NumericFormat::Fp6E3M2, &block, &mut wide).expect("it decodes");
        for value in &wide {
            assert_eq!(value.to_bits(), (28.0 * scale).to_bits(), "the top code is 28");
        }
        assert!(
            (scale * 28.0 - 1.0).abs() < 1.0e-3,
            "the f16 scale carries the peak over the largest code, got {scale}"
        );
    }

    fn uint16_bytes(bytes: &[u8]) -> f32 {
        float_layout_codec::decode(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])), FloatLayout::F16)
    }

    #[test]
    fn every_encodable_format_round_trips_its_own_values_and_the_census_is_pinned() {
        let mut encodable = std::vec::Vec::new();
        let mut refused = std::vec::Vec::new();
        for format in NumericFormat::ALL {
            let elements = usize::from(format.elements_per_block());
            let bytes = usize::from(format.bytes_per_block());
            let input: std::vec::Vec<f32> = (0..elements)
                .map(|index| (index % 7) as f32 / 4.0 - 0.75)
                .collect();
            let mut packed = std::vec![0u8; bytes];
            match encode_blocks(&format, &input, &mut packed) {
                Ok(blocks) => {
                    assert_eq!(blocks, 1, "{format:?} is one block");
                    let mut wide = std::vec![0.0f32; elements];
                    decode_blocks(&format, &packed, &mut wide).expect("what this encoder wrote decodes");
                    let mut again = std::vec![0u8; bytes];
                    encode_blocks(&format, &wide, &mut again).expect("a decoded block re-encodes");
                    let mut twice = std::vec![0.0f32; elements];
                    decode_blocks(&format, &again, &mut twice).expect("the second pass decodes");
                    for (once, twice) in wide.iter().zip(&twice) {
                        assert_eq!(
                            once.to_bits(),
                            twice.to_bits(),
                            "{format:?} drifted on a second pass"
                        );
                    }
                    encodable.push(format.name().unwrap_or("?"));
                }
                Err(NumericError::UnsupportedEncode { reason, .. }) => {
                    assert!(!reason.is_empty(), "{format:?} refuses without a reason");
                    refused.push(format.name().unwrap_or("?"));
                }
                Err(other) => panic!("{format:?} refused with {other:?}, not UnsupportedEncode"),
            }
        }
        assert_eq!(
            refused,
            [
                "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "iq1_s", "iq1_m",
                "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "iq4_xs",
            ],
            "pinned: the formats a shared scale cannot express"
        );
        assert_eq!(
            encodable.len() + refused.len(),
            NumericFormat::ALL.len(),
            "pinned: every format is classified"
        );
        assert_eq!(encodable.len(), 23, "pinned: formats with an encoder");
    }

    fn oracle_block() -> std::vec::Vec<f32> {
        (0..32).map(|index| (((index * 37) % 64) as f32 - 20.0) / 16.0).collect()
    }

    fn hex_of(bytes: &[u8]) -> std::string::String {
        bytes.iter().map(|byte| std::format!("{byte:02x}")).collect()
    }

    #[test]
    fn q4_1_matches_llama_cpp_b12110_quantize_row_q4_1_ref_on_the_oracle_block() {
        let bytes = packed(&NumericFormat::Q4_1, &oracle_block());

        assert_eq!(hex_of(&bytes), "333400bd40d962fb952eb751da730c953ec851ea");
    }

    #[test]
    fn q5_1_matches_llama_cpp_b12110_quantize_row_q5_1_ref_on_the_oracle_block() {
        let bytes = packed(&NumericFormat::Q5_1, &oracle_block());

        assert_eq!(hex_of(&bytes), "113000bd2ab55aa980a2d5f72a4c7f91c4e6193b6e80b3d5");
    }

    #[test]
    fn q4_1_rounds_an_exact_half_up_where_ties_to_even_would_give_zero() {
        let mut block = std::vec![0.0f32; 32];
        block[1] = 30.0;
        block[2] = 1.0;

        let bytes = packed(&NumericFormat::Q4_1, &block);

        // d = 30/15 = 2 (0x4000), min 0; (1.0-0)*0.5+0.5 = 1.0 truncates to code 1; 30 maps to 15
        let mut expected = std::vec![0u8; 20];
        expected[..4].copy_from_slice(&[0x00, 0x40, 0x00, 0x00]);
        expected[4 + 1] = 0x0f;
        expected[4 + 2] = 0x01;
        assert_eq!(bytes, expected);
    }

    #[test]
    fn stored_min_formats_round_trip_the_oracle_block_within_half_a_step() {
        for format in [NumericFormat::Q4_1, NumericFormat::Q5_1] {
            let input = oracle_block();
            let bytes = packed(&format, &input);
            let step = half::f16::from_le_bytes([bytes[0], bytes[1]]).to_f32();

            let output = decoded(&format, &bytes);

            for (index, (before, after)) in input.iter().zip(&output).enumerate() {
                assert!((before - after).abs() <= step / 2.0 + 1e-6, "{format:?} element {index}: {before} -> {after}");
            }
        }
    }

    #[test]
    fn every_legal_base_three_word_has_an_exact_storage() {
        // five trits in a byte is 3^5 = 243 digit strings; four trits (TQ1_0's last region) is 81
        let mut checked = 0;
        for (lanes, total) in [(5usize, 243u32), (4, 81)] {
            for integer in 0..total {
                let mut digits = [0u32; MAX_LANES];
                let mut rest = integer;
                for lane in (0..lanes).rev() {
                    digits[lane] = rest % 3;
                    rest /= 3;
                }
                let packed = pack_digits(&digits[..lanes], 3, 8)
                    .unwrap_or_else(|| panic!("{lanes} trits {integer} have no exact byte"));
                let mut back = [0u32; MAX_LANES];
                unpack_digits(packed, lanes, 3, 8, &mut back);
                assert_eq!(back[..lanes], digits[..lanes], "trits {integer}");
                checked += 1;
            }
        }
        assert_eq!(checked, 324, "pinned: every base-3 word this workspace packs");
    }

    #[test]
    fn ternary_radix_round_trips_onto_three_trits() {
        for format in [NumericFormat::Tq10, NumericFormat::Ptq1_0] {
            let input = values(usize::from(format.elements_per_block()) * 2, 0x723);
            let bytes = packed(&format, &input);
            let restored = decoded(&format, &bytes);
            let elements = usize::from(format.elements_per_block());
            for (originals, decodes) in input.chunks(elements).zip(restored.chunks(elements)) {
                let peak = peak_of(originals);
                let scale = grid_scale(peak, 1.0);
                let bound = peak / 2.0 + 2.0 * peak * F16_EPSILON;
                for (original, restore) in originals.iter().zip(decodes) {
                    assert!(
                        (original - restore).abs() <= bound,
                        "{format:?}: {original} decoded to {restore}, bound {bound}"
                    );
                    let trit = restore / scale;
                    assert!(
                        (trit - trit.round()).abs() < 1.0e-5 && trit.abs() <= 1.0,
                        "{format:?}: {trit} is not a trit"
                    );
                }
            }
        }
    }

    #[test]
    fn layouts_a_shared_scale_cannot_express_are_a_typed_error() {
        for (layout, reason) in [
            (
                BlockLayout::Q4_1,
                "a stored min is not an absolute-maximum scale",
            ),
            (BlockLayout::Q4_K, "sub-blocks carry their own scales"),
        ] {
            let error = reach(&layout).expect_err("not a shared-scale linear layout");
            assert_eq!(
                error,
                NumericError::UnsupportedEncode {
                    format: format_of(&layout),
                    reason,
                }
            );
        }
    }

    #[test]
    fn a_trained_grid_has_no_encoder_in_this_crate() {
        for format in [NumericFormat::Iq2Xs, NumericFormat::Iq3Xxs] {
            let elements = usize::from(format.elements_per_block());
            let mut out = std::vec![0u8; usize::from(format.bytes_per_block())];
            let error = encode_blocks(&format, &values(elements, 3), &mut out)
                .expect_err("a grid code needs a search over its codebook");
            assert!(
                matches!(error, NumericError::UnsupportedEncode { .. }),
                "{error:?}"
            );
        }
    }

    #[test]
    fn a_non_finite_block_element_is_a_typed_error() {
        let format = NumericFormat::Q8_0;
        let mut input = values(32, 3);
        input[5] = f32::NAN;
        let error = encode_blocks(&format, &input, &mut std::vec![0u8; 34])
            .expect_err("nan has no code");
        assert_eq!(
            error,
            NumericError::NonFinite {
                format: format_of(&BlockLayout::Q8_0),
            }
        );
    }

    #[test]
    fn a_layout_that_cannot_address_its_own_block_is_refused_not_indexed() {
        let mut lying = BlockLayout::Q4_0;
        lying.block_bytes = 1;
        let format = NumericFormat::Block(lying);
        let error = encode_blocks(&format, &values(32, 11), &mut std::vec![0u8; 1])
            .expect_err("32 nibbles do not fit in one byte");
        assert!(
            matches!(error, NumericError::InvalidGeometry { .. }),
            "{error:?}"
        );
    }

    #[test]
    fn a_partial_block_and_a_wrong_sized_output_are_typed() {
        let format = NumericFormat::Q8_0;
        let short = encode_blocks(&format, &values(31, 9), &mut std::vec![0u8; 34])
            .expect_err("31 values is not a whole block");
        assert_eq!(
            short,
            NumericError::InputNotElementMultiple {
                format: "q8_0",
                found: 31,
                elements: 32,
            }
        );
        let mis_sized = encode_blocks(&format, &values(32, 9), &mut std::vec![0u8; 33])
            .expect_err("q8_0 is 34 bytes per 32 values");
        assert_eq!(
            mis_sized,
            NumericError::OutputSizeMismatch {
                found: 33,
                expected: 34,
            }
        );
    }

    /// A 128-value activation row: gaussian-ish with two outlier channels per 64, the shape of a
    /// real key row, from splitmix64 seeded 0x6b76_5f72_6f77_0001 and a sum of twelve uniforms.
    fn activation_row() -> std::vec::Vec<f32> {
        let mut state = 0x6b76_5f72_6f77_0001u64;
        let mut unit = || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut mixed = state;
            mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            ((mixed ^ (mixed >> 31)) >> 40) as f32 / (1u64 << 24) as f32
        };
        (0..128)
            .map(|index| {
                let gaussian = (0..12).map(|_| unit()).sum::<f32>() - 6.0;
                if matches!(index % 64, 5 | 37) { gaussian * 24.0 } else { gaussian }
            })
            .collect()
    }

    // quantize_row_{q4_0,q5_0,q8_0,q8_1}_ref of activation_row(), as llama.cpp itself writes them:
    // libggml-base.dylib (built 2026-09-06 14:34 local, llama.cpp checkout
    // f1ea206218210afb913ae2f5d2c51faed35915da) called through dlopen by the xcheck-enc harness
    const LLAMA_Q4_0: [u8; 72] = [
        0xc1, 0x3c, 0x98, 0xa7, 0x88, 0x78, 0x77, 0x80, 0x88, 0x89, 0xb8, 0x87, 0xa6, 0x89, 0x98, 0x78, 0x98, 0x78,
        0x6a, 0x43, 0x88, 0x88, 0x98, 0x98, 0x88, 0x80, 0x87, 0x88, 0x98, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88,
        0x91, 0x3e, 0x89, 0x89, 0x87, 0x88, 0x98, 0x90, 0x88, 0x68, 0x98, 0x87, 0x77, 0x88, 0x98, 0x79, 0x87, 0x99,
        0x4e, 0x41, 0x78, 0x87, 0x87, 0x98, 0x88, 0x80, 0x78, 0x98, 0x88, 0x88, 0x88, 0x98, 0x88, 0x88, 0x88, 0x78,
    ];
    const LLAMA_Q5_0: [u8; 88] = [
        0xc1, 0x38, 0x8c, 0xd9, 0xe7, 0x5f, 0x2f, 0x3e, 0x01, 0xe1, 0xfe, 0x00, 0x0f, 0x02, 0x50, 0x1e, 0x3c, 0x12,
        0x10, 0xef, 0x21, 0xf1, 0x6a, 0x3f, 0x9f, 0xfe, 0xdd, 0x3d, 0x01, 0xf0, 0x10, 0x11, 0x00, 0xf0, 0x0e, 0x00,
        0x1f, 0xf0, 0x00, 0x00, 0x00, 0x00, 0xf1, 0xf1, 0x91, 0x3a, 0x9b, 0xb9, 0x7f, 0xd9, 0x11, 0x03, 0x0e, 0x00,
        0x30, 0x10, 0x1f, 0xd0, 0x10, 0xff, 0xfe, 0x01, 0x10, 0xe2, 0x0e, 0x22, 0x4e, 0x3d, 0xd9, 0xf7, 0xbe, 0x3a,
        0xf0, 0x1f, 0x0f, 0x11, 0x01, 0x00, 0xf0, 0x10, 0xf0, 0x00, 0xf0, 0x1f, 0x10, 0x00, 0xf0, 0xe0,
    ];
    const LLAMA_Q8_0: [u8; 136] = [
        0xcb, 0x2c, 0xf8, 0xee, 0x07, 0x08, 0xef, 0x81, 0xfb, 0x0f, 0x03, 0xec, 0xe1, 0x10, 0xff, 0xfb, 0x08, 0x06,
        0x0f, 0x1b, 0x02, 0xf3, 0xf6, 0x03, 0xfe, 0xff, 0x29, 0x06, 0x18, 0x07, 0x08, 0xf3, 0x0f, 0xf6, 0x79, 0x33,
        0x06, 0x01, 0xfd, 0x04, 0xff, 0x81, 0xf3, 0x03, 0xf9, 0xfc, 0x03, 0x03, 0x00, 0x01, 0x04, 0x06, 0x02, 0xfa,
        0x08, 0x08, 0x02, 0xfa, 0x00, 0x00, 0x0a, 0xfc, 0xfc, 0x00, 0x01, 0x00, 0xfb, 0xf9, 0x9e, 0x2e, 0x0b, 0x15,
        0xf3, 0xff, 0xfe, 0x81, 0xfb, 0xff, 0x04, 0xf5, 0xf0, 0x06, 0xfe, 0x0d, 0xef, 0x0c, 0x06, 0xfe, 0x00, 0x01,
        0x17, 0x09, 0x06, 0xe5, 0x09, 0xfc, 0xf7, 0x03, 0x09, 0xf3, 0x01, 0x12, 0x58, 0x31, 0x01, 0xf8, 0xf5, 0x06,
        0x07, 0x81, 0xff, 0x03, 0x01, 0xfe, 0x03, 0xf8, 0x00, 0x00, 0xfd, 0xff, 0xf4, 0x06, 0x00, 0x0c, 0x04, 0xff,
        0xf5, 0x08, 0xf9, 0xfc, 0xfb, 0x0b, 0x05, 0xfe, 0xfb, 0xf4,
    ];
    const LLAMA_Q8_1: [u8; 144] = [
        0xcb, 0x2c, 0x3e, 0xc5, 0xf8, 0xee, 0x07, 0x08, 0xef, 0x81, 0xfb, 0x0f, 0x03, 0xec, 0xe1, 0x10, 0xff, 0xfb,
        0x08, 0x06, 0x0f, 0x1b, 0x02, 0xf3, 0xf6, 0x03, 0xfe, 0xff, 0x29, 0x06, 0x18, 0x07, 0x08, 0xf3, 0x0f, 0xf6,
        0x79, 0x33, 0x4c, 0xcf, 0x06, 0x01, 0xfd, 0x04, 0xff, 0x81, 0xf3, 0x03, 0xf9, 0xfc, 0x03, 0x03, 0x00, 0x01,
        0x04, 0x06, 0x02, 0xfa, 0x08, 0x08, 0x02, 0xfa, 0x00, 0x00, 0x0a, 0xfc, 0xfc, 0x00, 0x01, 0x00, 0xfb, 0xf9,
        0x9e, 0x2e, 0x11, 0xc9, 0x0b, 0x15, 0xf3, 0xff, 0xfe, 0x81, 0xfb, 0xff, 0x04, 0xf5, 0xf0, 0x06, 0xfe, 0x0d,
        0xef, 0x0c, 0x06, 0xfe, 0x00, 0x01, 0x17, 0x09, 0x06, 0xe5, 0x09, 0xfc, 0xf7, 0x03, 0x09, 0xf3, 0x01, 0x12,
        0x58, 0x31, 0x64, 0xce, 0x01, 0xf8, 0xf5, 0x06, 0x07, 0x81, 0xff, 0x03, 0x01, 0xfe, 0x03, 0xf8, 0x00, 0x00,
        0xfd, 0xff, 0xf4, 0x06, 0x00, 0x0c, 0x04, 0xff, 0xf5, 0x08, 0xf9, 0xfc, 0xfb, 0x0b, 0x05, 0xfe, 0xfb, 0xf4,
    ];

    #[test]
    fn activation_row_bytes_equal_llama_cpp_quantize_row_ref_for_q4_0_q5_0_q8_0_and_q8_1() {
        let row = activation_row();
        let cases: [(NumericFormat, &[u8]); 4] = [
            (NumericFormat::Q4_0, &LLAMA_Q4_0),
            (NumericFormat::Q5_0, &LLAMA_Q5_0),
            (NumericFormat::Q8_0, &LLAMA_Q8_0),
            (NumericFormat::Q8_1, &LLAMA_Q8_1),
        ];
        for (format, expected) in cases {
            assert_eq!(packed(&format, &row), expected, "{:?}", format.name());
        }
    }

    #[test]
    fn encode_blocks_writes_the_codec_bytes_for_every_layout_the_codec_encodes() {
        let input = values(256 * 4, 21);
        let mut routed = 0usize;
        for format in NumericFormat::ALL.iter().filter(|format| !matches!(format, NumericFormat::Scalar(_))) {
            let Some(codec) = FormatCodec::for_format(format).filter(FormatCodec::can_encode) else {
                continue;
            };
            let blocks = input.len() / codec.elements();
            let mut expected = std::vec![0u8; blocks * codec.block_bytes()];
            codec.encode(&input, &mut expected).expect("the codec sizes match");

            assert_eq!(packed(format, &input), expected, "{:?}", format.name());
            routed += 1;
        }
        assert!(routed >= 8, "only {routed} formats resolved an encoder");
    }
}
