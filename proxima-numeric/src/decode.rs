//! The one place a [`NumericFormat`] becomes numbers: scalar decode/encode through
//! [`crate::float_layout_codec`], and one block walker for every `Block` and `Levels` format.
//!
//! Zero allocation on every path: the caller owns the output slice, exactly as
//! `omega::msl::LayoutFold::fold_levels` does. Every error is typed, never a panic and never a
//! silently wrong number.
//!
//! The surface is free functions over a `&NumericFormat`, not an `Encode`/`Decode` trait pair: a
//! trait would need one impl per format instantiation to express what a single `match` on the
//! format's three arms already expresses, and the repo's rule is that a type which adds no
//! capability is a relocation.
//!
//! Teaching pointer: [`decode_blocks`] is the whole conversion surface for a tensor payload;
//! [`NumericFormat::tag`] is its wire identity; [`crate::float_layout_codec`] is the scalar arm's
//! arithmetic. The per-format routines in `proxima-gguf::quant::*` are the incumbents this walker
//! is checked against bit for bit.

use crate::block_layout::{BlockLayout, Coded, Field, Plane};
use crate::error::NumericError;
use crate::float_layout::FloatLayout;
use crate::float_layout_codec;
use crate::format::{LevelFormat, LevelGeometry, LevelTable, NumericFormat};
use crate::tables::{IQ2XS_GRID, IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};

/// Decode one scalar of `format` from the first `bytes_per_block` bytes of `bytes`.
///
/// # Errors
/// [`NumericError::WidthUnsupported`] for a layout wider than 32 bits, and
/// [`NumericError::NoDecoder`] for a block or codebook format.
pub fn decode_scalar(format: &NumericFormat, bytes: &[u8]) -> Result<f32, NumericError> {
    let NumericFormat::Scalar(layout) = format else {
        return Err(NumericError::NoDecoder {
            format: format_name(format),
        });
    };
    float_layout_codec::layout_supported(layout)?;
    let width = layout.width();
    let byte_count = usize::from(width.div_ceil(8));
    let Some(head) = bytes.get(..byte_count) else {
        return Err(NumericError::BlockSizeMismatch {
            format: format_name(format),
            found: bytes.len(),
            expected: byte_count,
        });
    };
    Ok(float_layout_codec::decode(le_code(head), *layout))
}

/// Encode one scalar of `format` into the first `bytes_per_block` bytes of `bytes`.
///
/// # Errors
/// [`NumericError::WidthUnsupported`] for a layout wider than 32 bits,
/// [`NumericError::NoDecoder`] for a block format, and [`NumericError::OutputSizeMismatch`] when
/// `bytes` cannot hold the scalar.
pub fn encode_scalar(
    value: f32,
    format: &NumericFormat,
    bytes: &mut [u8],
) -> Result<usize, NumericError> {
    let NumericFormat::Scalar(layout) = format else {
        return Err(NumericError::NoDecoder {
            format: format_name(format),
        });
    };
    float_layout_codec::layout_supported(layout)?;
    let width = layout.width();
    let byte_count = usize::from(width.div_ceil(8));
    let Some(head) = bytes.get_mut(..byte_count) else {
        return Err(NumericError::OutputSizeMismatch {
            found: bytes.len(),
            expected: byte_count,
        });
    };
    let code = float_layout_codec::encode(value, *layout);
    for (byte, slot) in head.iter_mut().enumerate() {
        *slot = (code >> (8 * byte)) as u8;
    }
    Ok(byte_count)
}

/// Decode one block of `format`. `block` must be `bytes_per_block` bytes and `out` must hold
/// `elements_per_block` values.
///
/// [`NumericFormat::name`]-free by design: this runs once per block, so the name is resolved only
/// inside the error arms, never on the success path.
///
/// The walk indexes bytes directly, so `format` must have sound geometry. [`decode_blocks`]
/// checks that once per payload and is the entry point a tensor path uses; a
/// caller driving this per block owns that precondition.
///
/// # Errors
/// [`NumericError::BlockSizeMismatch`], [`NumericError::OutputSizeMismatch`],
/// [`NumericError::NoDecoder`] for a codebook format, [`NumericError::LevelOutOfTable`] for a code
/// outside the ladder, and [`NumericError::WidthUnsupported`] / [`NumericError::LayoutUnsupported`]
/// for a scalar outside the codec's domain.
pub fn decode_block(
    format: &NumericFormat,
    block: &[u8],
    out: &mut [f32],
) -> Result<(), NumericError> {
    let expected_bytes = usize::from(format.bytes_per_block());
    if block.len() != expected_bytes {
        return Err(NumericError::BlockSizeMismatch {
            format: format_name(format),
            found: block.len(),
            expected: expected_bytes,
        });
    }
    let expected_values = usize::from(format.elements_per_block());
    if out.len() != expected_values {
        return Err(NumericError::OutputSizeMismatch {
            found: out.len(),
            expected: expected_values,
        });
    }
    match format {
        NumericFormat::Scalar(layout) => {
            float_layout_codec::layout_supported(layout)?;
            out[0] = float_layout_codec::decode(le_code(block), *layout);
            Ok(())
        }
        NumericFormat::Block(layout) => {
            decode_layout_block(layout, block, out);
            Ok(())
        }
        NumericFormat::Levels(levels) => decode_levels_block(levels, format_name(format), block, out),
    }
}

/// Decode a whole payload of `format` into caller-owned storage. Returns the block count.
///
/// # Errors
/// [`NumericError::InvalidGeometry`] when the format cannot address its own block (a caller-built
/// [`crate::BlockLayout`] that lies), [`NumericError::InputNotBlockMultiple`] when `data` is not a
/// whole number of blocks, [`NumericError::OutputSizeMismatch`] when `out` is not exactly the
/// decoded element count, and whatever [`decode_block`] reports for this format.
pub fn decode_blocks(
    format: &NumericFormat,
    data: &[u8],
    out: &mut [f32],
) -> Result<usize, NumericError> {
    if let NumericFormat::Scalar(layout) = format {
        float_layout_codec::decode_slice(layout, data, out)
            .map_err(|fault| fault.into_error(layout, format_name(format)))?;
        return Ok(out.len());
    }
    geometry_fits(format)?;
    let block_bytes = usize::from(format.bytes_per_block());
    if block_bytes == 0 || !data.len().is_multiple_of(block_bytes) {
        return Err(NumericError::InputNotBlockMultiple {
            format: format_name(format),
            found: data.len(),
            block_bytes,
        });
    }
    let block_count = data.len() / block_bytes;
    let elements = usize::from(format.elements_per_block());
    let expected = block_count * elements;
    if out.len() != expected {
        return Err(NumericError::OutputSizeMismatch {
            found: out.len(),
            expected,
        });
    }
    for (block, chunk) in data
        .chunks_exact(block_bytes)
        .zip(out.chunks_exact_mut(elements))
    {
        decode_block(format, block, chunk)?;
    }
    Ok(block_count)
}

/// The name an error carries. Cold by construction: every caller is an error arm.
fn format_name(format: &NumericFormat) -> &'static str {
    format.name().unwrap_or("unnamed")
}

fn le_code(bytes: &[u8]) -> u32 {
    bytes
        .iter()
        .enumerate()
        .fold(0u32, |code, (byte, value)| {
            code | (u32::from(*value) << (8 * byte))
        })
}

const MAX_SUB_BLOCKS: usize = 32;

/// The bits at `bit_address`, `bits` wide, little-endian within the byte run it spans. Unlike a
/// single-byte read this spans byte boundaries, which is what a six-bit field needs.
fn bits_at(block: &[u8], bit_address: usize, bits: u8) -> u32 {
    let mut value = 0u32;
    let mut read = 0u32;
    while read < u32::from(bits) {
        let address = bit_address + read as usize;
        let byte = u32::from(block[address / 8]) >> (address % 8);
        value |= byte << read;
        read += 8 - (address % 8) as u32;
    }
    mask(bits) & value
}

const fn mask(bits: u8) -> u32 {
    if bits >= 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    }
}

/// `code` assembled from the coded's planes at element `index`.
fn code_at(coded: &Coded, index: usize, block: &[u8]) -> u32 {
    if coded.packed_radix > 1 {
        return packed_radix_code(coded, index, block);
    }
    let low = bits_at(block, coded.low.address(index), coded.low.bits);
    match &coded.high {
        Some(high) => low | (bits_at(block, high.address(index), high.bits) << coded.low.bits),
        None => low,
    }
}

fn packed_radix_code(coded: &Coded, index: usize, block: &[u8]) -> u32 {
    let mut region_index = 0;
    while region_index < usize::from(coded.region_count) {
        let region = &coded.regions[region_index];
        let start = usize::from(region.start);
        let end = start + usize::from(region.elements);
        if (start..end).contains(&index) {
            let local_index = index - start;
            let packed = bits_at(
                block,
                region.source.address(local_index),
                region.source.bits,
            );
            let mut lane = local_index;
            for digit in region
                .source
                .digits
                .iter()
                .take(usize::from(region.digit_axis))
            {
                lane /= usize::from(digit.radix);
            }
            lane %= usize::from(region.source.digits[usize::from(region.digit_axis)].radix);
            let mut factor = 1u32;
            for _ in 0..lane {
                factor *= u32::from(coded.packed_radix);
            }
            let storage_bits = u32::from(region.source.bits);
            let residue = packed.wrapping_mul(factor) & mask(region.source.bits);
            return (residue * u32::from(coded.packed_radix)) >> storage_bits;
        }
        region_index += 1;
    }
    0
}

fn field_value(field: &Field, block: &[u8]) -> f32 {
    let offset = usize::from(field.byte_offset);
    let bytes = usize::from(field.layout.width()).div_ceil(8);
    float_layout_codec::decode(le_code(&block[offset..offset + bytes]), field.layout)
}

fn sub_value(coded: &Coded, sub_block: usize, block: &[u8]) -> f32 {
    float_layout_codec::decode(code_at(coded, sub_block, block), coded.code)
        - f32::from(coded.bias)
}

struct Terms {
    scales: [f32; MAX_SUB_BLOCKS],
    mins: [f32; MAX_SUB_BLOCKS],
}

fn terms(layout: &BlockLayout, block: &[u8]) -> Terms {
    let scale = field_value(&layout.scale, block);
    let min = layout.min.map(|field| field_value(&field, block));
    let mut terms = Terms {
        scales: [scale; MAX_SUB_BLOCKS],
        mins: [min.unwrap_or(0.0); MAX_SUB_BLOCKS],
    };
    if layout.scale_runs.is_some() {
        for run in 0..usize::from(layout.runs()).min(MAX_SUB_BLOCKS) {
            let field = Field {
                byte_offset: layout.run_scale_byte(run as u16) as u16,
                ..layout.scale
            };
            terms.scales[run] = field_value(&field, block);
        }
    }
    if let Some(sub) = &layout.sub {
        for sub_block in 0..usize::from(sub.count).min(MAX_SUB_BLOCKS) {
            terms.scales[sub_block] = scale * sub_value(&sub.scale, sub_block, block);
            if let (Some(offset), Some(coded)) = (min, &sub.min) {
                terms.mins[sub_block] = offset * sub_value(coded, sub_block, block);
            }
        }
    }
    terms
}

/// One `Block` format's block: codes through their own layout, scaled and biased.
fn decode_layout_block(layout: &BlockLayout, block: &[u8], out: &mut [f32]) {
    let terms = terms(layout, block);
    let span = layout
        .sub
        .map_or(usize::from(layout.run_elements()), |sub| {
            usize::from(layout.elements / sub.count)
        });
    let element = &layout.element;
    let mut low = element.low.cursor();
    let mut high = element.high.map(|plane| (plane, plane.cursor()));
    let (multiplier, bias) = (f32::from(layout.multiplier), f32::from(element.bias));
    let min_multiplier = f32::from(layout.min_multiplier);
    for (index, slot) in out.iter_mut().enumerate() {
        let code = if element.packed_radix > 1 {
            float_layout_codec::decode(packed_radix_code(element, index, block), element.code)
        } else {
            let mut raw = bits_at(block, low.address(), element.low.bits);
            low.advance();
            if let Some((plane, cursor)) = high.as_mut() {
                raw |= bits_at(block, cursor.address(), plane.bits) << element.low.bits;
                cursor.advance();
            }
            float_layout_codec::decode(raw, element.code)
        };
        let sub_block = index / span;
        let value = (code * multiplier - bias) * terms.scales[sub_block];
        *slot = if layout.min.is_some() {
            value + min_multiplier * terms.mins[sub_block]
        } else {
            value
        };
    }
}

/// One `Levels` format's block: codes through an explicit level table, scaled by the block scale.
fn decode_levels_block(
    levels: &LevelFormat,
    name: &'static str,
    block: &[u8],
    out: &mut [f32],
) -> Result<(), NumericError> {
    match levels.levels {
        LevelTable::Linear { .. } | LevelTable::Explicit(_) => {
            let Some(packing) = levels.packing() else {
                return Err(NumericError::NoDecoder { format: name });
            };
            let codes = &packing.element;
            let scale = field_value(&packing.scale, block);
            let mut low = codes.low.cursor();
            let mut high = codes.high.map(|plane| (plane, plane.cursor()));
            for (index, slot) in out.iter_mut().enumerate() {
                let code = if codes.packed_radix > 1 {
                    packed_radix_code(codes, index, block)
                } else {
                    let mut raw = bits_at(block, low.address(), codes.low.bits);
                    low.advance();
                    if let Some((plane, cursor)) = high.as_mut() {
                        raw |= bits_at(block, cursor.address(), plane.bits) << codes.low.bits;
                        cursor.advance();
                    }
                    raw
                };
                let level = levels
                    .levels
                    .level(code)
                    .ok_or(NumericError::LevelOutOfTable { format: name, code })?;
                *slot = level * scale;
            }
            Ok(())
        }
        LevelTable::Iq2XsGrid => {
            decode_iq2_xs(block, out);
            Ok(())
        }
        LevelTable::Iq3XxsGrid => {
            decode_iq3_xxs(block, out);
            Ok(())
        }
        LevelTable::External => Err(NumericError::NoDecoder { format: name }),
    }
}

fn u16_at(block: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([block[offset], block[offset + 1]])
}

fn f16_at(block: &[u8], offset: usize) -> f32 {
    float_layout_codec::decode(u32::from(u16_at(block, offset)), FloatLayout::F16)
}

fn u32_at(block: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        block[offset],
        block[offset + 1],
        block[offset + 2],
        block[offset + 3],
    ])
}

const IQ2_XS_QS_OFFSET: usize = 2;
const IQ2_XS_SCALES_OFFSET: usize = IQ2_XS_QS_OFFSET + 256 / 8 * 2;
const IQ3_XXS_QS_OFFSET: usize = 2;
const IQ3_XXS_SCALES_OFFSET: usize = IQ3_XXS_QS_OFFSET + 256 / 4;

fn decode_iq2_xs(block: &[u8], out: &mut [f32]) {
    let delta = f16_at(block, 0);
    for sub_block in 0..256 / 32 {
        let scale_byte = block[IQ2_XS_SCALES_OFFSET + sub_block];
        let db_low = delta * (0.5 + f32::from(scale_byte & 0x0F)) * 0.25;
        let db_high = delta * (0.5 + f32::from(scale_byte >> 4)) * 0.25;
        for group in 0..4 {
            let qs_value = u16_at(block, IQ2_XS_QS_OFFSET + (sub_block * 4 + group) * 2);
            let grid_bytes = IQ2XS_GRID[usize::from(qs_value & 511)].to_le_bytes();
            let signs = KSIGNS_IQ2XS[usize::from(qs_value >> 9)];
            let db = if group < 2 { db_low } else { db_high };
            let base = sub_block * 32 + group * 8;
            for lane in 0..8 {
                let sign = if signs & KMASK_IQ2XS[lane] != 0 {
                    -1.0
                } else {
                    1.0
                };
                out[base + lane] = db * f32::from(grid_bytes[lane]) * sign;
            }
        }
    }
}

fn decode_iq3_xxs(block: &[u8], out: &mut [f32]) {
    let delta = f16_at(block, 0);
    for sub_block in 0..256 / 32 {
        let aux = u32_at(block, IQ3_XXS_SCALES_OFFSET + sub_block * 4);
        let db = delta * (0.5 + f32::from((aux >> 28) as u8)) * 0.5;
        let grid_base = IQ3_XXS_QS_OFFSET + sub_block * 8;
        for group in 0..4 {
            let signs = KSIGNS_IQ2XS[((aux >> (7 * group)) & 127) as usize];
            let grid1 = IQ3XXS_GRID[usize::from(block[grid_base + group * 2])].to_le_bytes();
            let grid2 = IQ3XXS_GRID[usize::from(block[grid_base + group * 2 + 1])].to_le_bytes();
            let base = sub_block * 32 + group * 8;
            for lane in 0..4 {
                let sign1 = if signs & KMASK_IQ2XS[lane] != 0 {
                    -1.0
                } else {
                    1.0
                };
                let sign2 = if signs & KMASK_IQ2XS[lane + 4] != 0 {
                    -1.0
                } else {
                    1.0
                };
                out[base + lane] = db * f32::from(grid1[lane]) * sign1;
                out[base + 4 + lane] = db * f32::from(grid2[lane]) * sign2;
            }
        }
    }
}

/// The furthest bit a plane reaches across `elements` indexes, for the structural check that a
/// layout's planes stay inside its block.
#[must_use]
pub fn plane_span(plane: &Plane, elements: usize) -> usize {
    (0..elements)
        .map(|index| plane.address(index) + usize::from(plane.bits))
        .max()
        .unwrap_or(0)
}

/// Whether a format's own description can address its own block: every plane and field inside
/// `bytes_per_block`, every sub-block count a divisor of the element count, every packed-radix
/// region contiguous and covering the block, and every level table the size of its code space.
///
/// The walker indexes bytes and arrays directly, so this is what turns a caller-built
/// [`crate::BlockLayout`] that lies about its geometry into a typed error instead of a panic. It
/// runs once per payload, never per block.
pub(crate) fn geometry_fits(format: &NumericFormat) -> Result<(), NumericError> {
    let unsound = |reason: &'static str| NumericError::InvalidGeometry {
        format: format_name(format),
        reason,
    };
    match format {
        NumericFormat::Scalar(layout) => {
            if layout.width() == 0 {
                return Err(unsound("a scalar with no bits"));
            }
            float_layout_codec::layout_supported(layout)
        }
        NumericFormat::Block(layout) => {
            block_geometry_fits(layout).map_err(unsound)?;
            code_layouts_supported(layout)
        }
        NumericFormat::Levels(levels) => match &levels.geometry {
            LevelGeometry::Packed(layout) => {
                block_geometry_fits(layout).map_err(unsound)?;
                code_layouts_supported(layout)?;
                let coded = &layout.element;
                // a packed-radix code names `radix` values regardless of its storage plane width
                // (`LevelFormat::ladder` derives the count the same way), every other code names
                // the full width of its planes
                let plane_bits = coded.low.bits + coded.high.map_or(0, |plane| plane.bits);
                let code_space = if coded.packed_radix > 1 {
                    usize::from(coded.packed_radix)
                } else if plane_bits >= 16 {
                    usize::MAX
                } else {
                    1usize << plane_bits
                };
                match levels.levels {
                    LevelTable::Linear { count, .. } if usize::from(count) != code_space => {
                        Err(unsound("a linear ladder whose count is not its code space"))
                    }
                    LevelTable::Explicit(values) if values.len() != code_space => {
                        Err(unsound("an explicit ladder whose length is not its code space"))
                    }
                    _ => Ok(()),
                }
            }
            LevelGeometry::Codebook {
                elements,
                bytes,
                scale,
            } => {
                let bits = usize::from(*bytes) * 8;
                if *elements == 0 || *bytes == 0 {
                    return Err(unsound("a codebook block with no elements or bytes"));
                }
                if usize::from(scale.byte_offset) * 8 + usize::from(scale.layout.width()) > bits {
                    return Err(unsound("a codebook scale outside its block"));
                }
                float_layout_codec::layout_supported(&scale.layout)
            }
        },
    }
}

/// Every scalar layout a block's codes and fields decode through must itself be executable;
/// otherwise the walker would hand [`float_layout_codec::decode`] a layout outside its domain.
fn code_layouts_supported(layout: &BlockLayout) -> Result<(), NumericError> {
    float_layout_codec::layout_supported(&layout.element.code)?;
    float_layout_codec::layout_supported(&layout.scale.layout)?;
    if let Some(min) = &layout.min {
        float_layout_codec::layout_supported(&min.layout)?;
    }
    if let Some(sub) = &layout.sub {
        float_layout_codec::layout_supported(&sub.scale.code)?;
        if let Some(min) = &sub.min {
            float_layout_codec::layout_supported(&min.code)?;
        }
    }
    Ok(())
}

fn block_geometry_fits(layout: &BlockLayout) -> Result<(), &'static str> {
    if layout.elements == 0 || layout.block_bytes == 0 {
        return Err("a block with no elements or bytes");
    }
    let bits = usize::from(layout.block_bytes) * 8;
    let elements = usize::from(layout.elements);
    let element = &layout.element;
    if plane_span(&element.low, elements) > bits {
        return Err("an element plane outside the block");
    }
    if let Some(high) = &element.high
        && plane_span(high, elements) > bits
    {
        return Err("an element high plane outside the block");
    }
    for field in [Some(layout.scale), layout.min].into_iter().flatten() {
        if usize::from(field.byte_offset) * 8 + usize::from(field.layout.width()) > bits {
            return Err("a scale or min field outside the block");
        }
    }
    if let Some(runs) = layout.scale_runs {
        let count = usize::from(runs.count);
        if count == 0 || !elements.is_multiple_of(count) {
            return Err("a scale-run count that does not divide the element count");
        }
        let scale_bytes = usize::from(layout.scale.layout.width().div_ceil(8));
        let last =
            usize::from(layout.scale.byte_offset) + (count - 1) * usize::from(runs.stride_bytes);
        if last + scale_bytes > usize::from(layout.block_bytes) {
            return Err("a scale run outside the block");
        }
    }
    if let Some(sub) = &layout.sub {
        let count = usize::from(sub.count);
        if count == 0 || count > MAX_SUB_BLOCKS || !elements.is_multiple_of(count) {
            return Err("a sub-block count this block cannot carry");
        }
        if plane_span(&sub.scale.low, count) > bits {
            return Err("a sub-block scale plane outside the block");
        }
        if let Some(high) = &sub.scale.high
            && plane_span(high, count) > bits
        {
            return Err("a sub-block scale high plane outside the block");
        }
        if let Some(min) = &sub.min
            && plane_span(&min.low, count) > bits
        {
            return Err("a sub-block min plane outside the block");
        }
    }
    if element.packed_radix > 1 {
        let mut covered = 0usize;
        for region in element.regions.iter().take(usize::from(element.region_count)) {
            if usize::from(region.start) != covered || region.elements == 0 {
                return Err("packed-radix regions that do not tile the block");
            }
            if plane_span(&region.source, usize::from(region.elements)) > bits {
                return Err("a packed-radix region outside the block");
            }
            covered += usize::from(region.elements);
        }
        if covered != elements {
            return Err("packed-radix regions that do not cover every element");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::float_layout::{FloatLayout, Specials};

    /// The scalar layouts the shared 32-bit codec executes, exhaustive on every one at or below
    /// 16 bits and sampled above it.
    const EXECUTABLE_LAYOUTS: [FloatLayout; 24] = [
        FloatLayout::E5M2,
        FloatLayout::E5M2_SATURATING,
        FloatLayout::E4M3FN,
        FloatLayout::F16,
        FloatLayout::E2M1,
        FloatLayout::E2M2,
        FloatLayout::E3M2,
        FloatLayout::E2M3,
        FloatLayout::BF16,
        FloatLayout::BF32,
        FloatLayout::F8_E8M0,
        FloatLayout::INT8,
        FloatLayout::UINT8,
        FloatLayout::INT16,
        FloatLayout::UINT16,
        FloatLayout::INT32,
        FloatLayout::UINT32,
        FloatLayout::UINT1,
        FloatLayout::UINT2,
        FloatLayout::UINT3,
        FloatLayout::UINT4,
        FloatLayout::UINT5,
        FloatLayout::UINT6,
        FloatLayout::SIGN_MAGNITUDE_INT8,
    ];

    /// A deterministic splitmix64 so a "sampled" arm is reproducible without a `rand` dependency.
    struct SplitMix(u64);

    impl SplitMix {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut mixed = self.0;
            mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            mixed ^ (mixed >> 31)
        }
    }

    /// A code that decodes to a non-finite value: a NaN payload, or an infinity code under a
    /// policy that reserves one. Re-narrowing a non-finite value is a policy decision, not an
    /// identity, so the round-trip invariant is stated over finite codes.
    fn is_non_finite(code: u32, layout: FloatLayout) -> bool {
        !crate::float_layout_codec::decode(code, layout).is_finite()
    }

    #[test]
    fn every_executable_scalar_layout_round_trips_its_own_codes_bit_exactly() {
        assert_eq!(EXECUTABLE_LAYOUTS.len(), 24, "pinned: executed scalar layouts");
        let mut tiny_formats = 0;
        for layout in EXECUTABLE_LAYOUTS {
            if layout.width() > 32 {
                continue;
            }
            let modulus = 1u64 << layout.width();
            let codes = if layout.width() <= 16 {
                tiny_formats += 1;
                modulus
            } else {
                10_000
            };
            let mut generator = SplitMix(u64::from(layout.width()) * 0x5deece66d);
            for step in 0..codes {
                let code = if layout.width() <= 16 {
                    step as u32
                } else {
                    (generator.next() % modulus) as u32
                };
                if is_non_finite(code, layout) {
                    continue;
                }
                let value = crate::float_layout_codec::decode(code, layout);
                let renarrowed = crate::convert::narrow_to_layout(
                    value,
                    layout,
                    crate::convert::Rounding::NearestTiesEven,
                    crate::convert::Overflow::Format,
                )
                .expect("an executable layout narrows");
                assert_eq!(
                    renarrowed.to_bits(),
                    value.to_bits(),
                    "{layout:?} code {code:#x} ({value}) must re-narrow to itself"
                );
            }
        }
        assert_eq!(tiny_formats, 21, "pinned: exhaustively checked layouts");
    }

    #[test]
    fn a_layout_wider_than_the_shared_codec_reports_its_width() {
        for layout in [
            FloatLayout::F64,
            FloatLayout::INT64,
            FloatLayout::UINT64,
            FloatLayout::INT128,
            FloatLayout::UINT128,
        ] {
            let error = crate::convert::narrow_to_layout(
                1.0,
                layout,
                crate::convert::Rounding::NearestTiesEven,
                crate::convert::Overflow::Format,
            )
            .expect_err("a 64-bit or wider layout has no 32-bit codec");
            assert_eq!(
                error,
                NumericError::WidthUnsupported {
                    width: layout.width(),
                    max: 32
                }
            );
        }
    }

    #[test]
    fn explicit_rounding_and_overflow_decide_the_narrowed_value() {
        use crate::convert::{Overflow, Rounding, narrow_to_layout};

        // 1 + 2^-8 is exactly halfway between bf16's 1.0 and its next level 1.0078125.
        let value = 1.0 + f64::from_bits(0) as f32 * 0.0 + 2.0f32.powi(-8);
        assert_eq!(
            narrow_to_layout(value, FloatLayout::BF16, Rounding::NearestTiesEven, Overflow::Format),
            Ok(1.0)
        );
        assert_eq!(
            narrow_to_layout(value, FloatLayout::BF16, Rounding::AwayFromZero, Overflow::Format),
            Ok(1.007_812_5)
        );
        assert_eq!(
            narrow_to_layout(value, FloatLayout::BF16, Rounding::TowardZero, Overflow::Format),
            Ok(1.0)
        );

        // 1000.0 overflows e5m2's 57344? No: 1000 is in range; use 1e6.
        assert_eq!(
            narrow_to_layout(1.0e6, FloatLayout::E5M2, Rounding::NearestTiesEven, Overflow::Format),
            Ok(f32::INFINITY)
        );
        assert_eq!(
            narrow_to_layout(1.0e6, FloatLayout::E5M2, Rounding::NearestTiesEven, Overflow::Saturate),
            Ok(FloatLayout::E5M2.max_normal())
        );
    }

    #[test]
    fn an_integer_layout_saturates_rather_than_wrapping() {
        use crate::convert::{Overflow, Rounding, narrow_to_format};

        let int8 = NumericFormat::Scalar(FloatLayout::INT8);
        assert_eq!(
            narrow_to_format(1000.0, &int8, Rounding::NearestTiesEven, Overflow::Format),
            Ok(127.0)
        );
        assert_eq!(
            narrow_to_format(-1000.0, &int8, Rounding::NearestTiesEven, Overflow::Format),
            Ok(-128.0)
        );
    }

    #[test]
    fn every_block_and_level_format_stays_inside_its_own_block() {
        let mut checked = 0;
        for format in NumericFormat::ALL {
            let bits = usize::from(format.bytes_per_block()) * 8;
            match format {
                NumericFormat::Scalar(_) => continue,
                NumericFormat::Block(layout) => {
                    assert_block_inside(&layout, bits, format);
                    checked += 1;
                }
                NumericFormat::Levels(levels) => {
                    assert!(
                        usize::from(levels.scale().byte_offset) * 8
                            + usize::from(levels.scale().layout.width())
                            <= bits,
                        "{format:?} scale escapes its block"
                    );
                    if let Some(packing) = levels.packing() {
                        let codes = &packing.element;
                        assert!(
                            plane_span(&codes.low, usize::from(levels.elements())) <= bits,
                            "{format:?} low plane escapes its block"
                        );
                        if let Some(high) = &codes.high {
                            assert!(
                                plane_span(high, usize::from(levels.elements())) <= bits,
                                "{format:?} high plane escapes its block"
                            );
                        }
                    }
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 31, "pinned: non-scalar formats with a checked block");
    }

    fn assert_block_inside(layout: &BlockLayout, bits: usize, format: NumericFormat) {
        assert!(
            plane_span(&layout.element.low, usize::from(layout.elements)) <= bits,
            "{format:?} low plane escapes its block"
        );
        if let Some(high) = &layout.element.high {
            assert!(
                plane_span(high, usize::from(layout.elements)) <= bits,
                "{format:?} high plane escapes its block"
            );
        }
        for field in [Some(layout.scale), layout.min].into_iter().flatten() {
            assert!(
                usize::from(field.byte_offset) * 8 + usize::from(field.layout.width()) <= bits,
                "{format:?} field escapes its block"
            );
        }
        if let Some(sub) = &layout.sub {
            for region in sub.scale.regions.iter().take(usize::from(sub.scale.region_count)) {
                assert!(
                    plane_span(&region.source, usize::from(sub.count)) <= bits,
                    "{format:?} sub-block region escapes its block"
                );
            }
        }
    }

    #[test]
    fn a_scale_run_that_walks_off_the_block_is_a_typed_error() {
        // four runs of 32 values over a 144-byte block; a 64-byte stride puts the fourth scale at
        // byte 192, so the walker would read two bytes past the block instead of calling the
        // layout unsound
        let mut lying = BlockLayout::Q8_1_GROUPED;
        lying.scale_runs = Some(crate::block_layout::ScaleRuns {
            count: 4,
            stride_bytes: 64,
        });
        let error = decode_blocks(
            &NumericFormat::Block(lying),
            &[0_u8; 144],
            &mut [0.0; 128],
        )
        .expect_err("a scale run outside the block is unsound");
        assert!(
            matches!(error, NumericError::InvalidGeometry { .. }),
            "{error:?}"
        );
    }

    #[test]
    fn a_short_or_overlong_payload_is_a_typed_error_not_a_guess() {
        let format = NumericFormat::Q4_0;
        assert_eq!(
            decode_blocks(&format, &[0_u8; 17], &mut [0.0; 32]),
            Err(NumericError::InputNotBlockMultiple {
                format: "q4_0",
                found: 17,
                block_bytes: 18
            })
        );
        assert_eq!(
            decode_blocks(&format, &[0_u8; 18], &mut [0.0; 31]),
            Err(NumericError::OutputSizeMismatch {
                found: 31,
                expected: 32
            })
        );
    }

    #[test]
    fn a_codebook_without_its_table_refuses_rather_than_guessing() {
        let mut out = [0.0_f32; 256];
        assert_eq!(
            decode_blocks(&NumericFormat::Iq1S, &[0_u8; 50], &mut out),
            Err(NumericError::NoDecoder { format: "iq1_s" })
        );
    }

    #[test]
    fn q4_0_decodes_the_documented_ladder_under_its_scale() {
        let mut block = [0_u8; 18];
        let scale = crate::float_layout_codec::encode(0.5, FloatLayout::F16);
        block[..2].copy_from_slice(&(scale as u16).to_le_bytes());
        block[2] = 0xf0;
        let mut out = [0.0_f32; 32];
        decode_block(&NumericFormat::Q4_0, &block, &mut out).expect("q4_0 decodes");
        assert_eq!(out[0], -4.0, "low nibble 0 is level -8 at 0.5 scale");
        assert_eq!(out[16], 3.5, "high nibble 15 is level 7 at 0.5 scale");
    }

    /// A worked example for the one block geometry this crate introduced rather than moved:
    /// fp6 E3M2 is 32 six-bit fields after a leading f16 scale (`omega::msl::
    /// scaled_e3m2_unpack_msl` reads `block[0] | block[1] << 8` for the scale and addresses the
    /// fields from `block + 2`), 26 bytes per block. Four fields pack little-endian into three
    /// bytes: `0x00, 0x0C, 0x1F, 0x2C` are `0xB1F300` and decode to 0, 1, 28, -1.
    #[test]
    fn an_fp6_e3m2_block_decodes_the_worked_example() {
        let mut block = [0u8; 26];
        block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        block[2..5].copy_from_slice(&[0x00, 0xf3, 0xb1]);
        let mut out = [0.0_f32; 32];
        decode_block(&NumericFormat::Fp6E3M2, &block, &mut out).expect("one fp6 block decodes");
        assert_eq!(&out[..5], &[0.0, 1.0, 28.0, -1.0, 0.0]);
        assert_eq!(out[5..], [0.0; 27]);
    }

    /// MXFP4: one E8M0 exponent byte then 16 nibble bytes, 32 E2M1 values per block. The scale
    /// byte 127 is `2^0`; E2M1's code 0 is 0.0 and code 2 is 1.0 (code 1 is 0.5), so a nibble
    /// byte `0x20` puts 0.0 in the low lane and 1.0 in the high lane.
    #[test]
    fn an_mxfp4_block_decodes_the_worked_example() {
        let mut block = [0u8; 17];
        block[0] = 127;
        block[1] = 0x20;
        let mut out = [0.0_f32; 32];
        decode_block(&NumericFormat::Mxfp4, &block, &mut out).expect("one mxfp4 block decodes");
        assert_eq!(out[0], 0.0, "low nibble of the first byte is level 0");
        assert_eq!(out[16], 1.0, "high nibble of the first byte is code 2, i.e. 1.0");
        assert_eq!(out[1], 0.0);
    }

    /// NVFP4: four E4M3 sub-block scales (one per 16 elements) then 32 nibble bytes, 64 E2M1
    /// values per block. E4M3 `0x38` is 1.0, so a `0x20` nibble byte places code 0 (0.0) at
    /// element 0 and code 2 (1.0) at element 32 (the nibble split the 32-byte run uses).
    #[test]
    fn an_nvfp4_block_decodes_the_worked_example() {
        let mut block = [0u8; 36];
        block[..4].fill(0x38);
        block[4] = 0x20;
        let mut out = [0.0_f32; 64];
        decode_block(&NumericFormat::Nvfp4, &block, &mut out).expect("one nvfp4 block decodes");
        assert_eq!(out[0], 0.0);
        assert_eq!(out[32], 1.0, "high nibble of byte 0 is element 32");
        assert_eq!(out[16], 0.0, "sub-block 1 starts at element 16");
    }

    #[test]
    fn a_layout_outside_the_codec_domain_is_a_typed_error_everywhere() {
        use crate::convert::{Overflow, Rounding, narrow_to_layout};
        use crate::float_layout::SignEncoding;

        // 32 bits wide, so the width gate passes, but 30 mantissa bits are more than the
        // `23 - mantissa_bits` widening shift can express
        let wide = FloatLayout {
            exponent_bits: 1,
            mantissa_bits: 30,
            specials: Specials::InfinityOnOverflow,
            bias_override: None,
            sign: SignEncoding::Magnitude,
            nan: Default::default(),
        };
        let format = NumericFormat::Scalar(wide);

        assert!(matches!(
            decode_scalar(&format, &[0; 4]),
            Err(NumericError::LayoutUnsupported { .. })
        ));
        assert!(matches!(
            decode_blocks(&format, &[0; 4], &mut [0.0; 1]),
            Err(NumericError::LayoutUnsupported { .. })
        ));
        assert!(matches!(
            narrow_to_layout(1.0, wide, Rounding::NearestTiesEven, Overflow::Format),
            Err(NumericError::LayoutUnsupported { .. })
        ));
    }

    #[test]
    fn a_block_layout_that_lies_about_its_geometry_is_rejected_not_indexed() {
        // Q4_0's planes reach bit 144; claiming a 4-byte block would read past the slice
        let mut lying = BlockLayout::Q4_0;
        lying.block_bytes = 4;
        let mut out = [0.0_f32; 32];
        assert_eq!(
            decode_blocks(&NumericFormat::Block(lying), &[0_u8; 4], &mut out),
            Err(NumericError::InvalidGeometry {
                format: "unnamed",
                reason: "an element plane outside the block",
            })
        );
    }

    #[test]
    fn a_radix_layout_whose_regions_do_not_cover_is_rejected() {
        // TQ1_0 tiles 160 + 80 + 16 elements; dropping the last region leaves 240 of 256
        let mut lying = BlockLayout::TQ1_0;
        lying.element.region_count = 2;
        let mut out = [0.0_f32; 256];
        assert_eq!(
            decode_blocks(&NumericFormat::Block(lying), &[0_u8; 54], &mut out),
            Err(NumericError::InvalidGeometry {
                format: "unnamed",
                reason: "packed-radix regions that do not cover every element",
            })
        );
    }

    #[test]
    fn a_scale_run_count_of_zero_is_rejected_rather_than_dividing() {
        let mut lying = BlockLayout::Q8_1_GROUPED;
        lying.scale_runs = Some(crate::block_layout::ScaleRuns {
            count: 0,
            stride_bytes: 4,
        });
        let mut out = [0.0_f32; 128];
        assert_eq!(
            decode_blocks(&NumericFormat::Block(lying), &[0_u8; 144], &mut out),
            Err(NumericError::InvalidGeometry {
                format: "unnamed",
                reason: "a scale-run count that does not divide the element count",
            })
        );
        assert_eq!(lying.runs(), 1, "the accessor stays total for a caller-built layout");
    }

    #[test]
    fn an_e5m2_nan_code_decodes_to_nan_not_to_a_number() {
        assert!(crate::float_layout_codec::decode(0x7d, FloatLayout::E5M2).is_nan());
        assert!(!crate::float_layout_codec::decode(0x7b, FloatLayout::E5M2).is_nan());
    }

    #[test]
    fn the_saturating_bf8_policy_never_stores_an_infinity() {
        assert_eq!(FloatLayout::E5M2_SATURATING.specials, Specials::SaturateOnOverflow);
        assert_eq!(
            narrow_to_scalar(f32::INFINITY, FloatLayout::E5M2_SATURATING),
            FloatLayout::E5M2_SATURATING.max_normal()
        );
        assert_eq!(
            narrow_to_scalar(f32::INFINITY, FloatLayout::E5M2).to_bits(),
            f32::INFINITY.to_bits()
        );
    }

    fn narrow_to_scalar(value: f32, layout: FloatLayout) -> f32 {
        crate::convert::narrow_to_layout(
            value,
            layout,
            crate::convert::Rounding::NearestTiesEven,
            crate::convert::Overflow::Format,
        )
        .expect("layout narrows")
    }
}
