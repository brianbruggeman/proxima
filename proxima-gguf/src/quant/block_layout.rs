//! The one CPU block walker: decodes any [`BlockLayout`] by reading its planes, assembling each
//! element code, and applying
//! `(multiplier * code - bias) * scale * sub_scale + min_multiplier * (min * sub_min)`.
//!
//! Teaching pointer: `proxima_primitives::block_layout::BlockLayout` is the data, the integer
//! path of [`super::float_layout`] turns a code into its number, and the scales and mins decode
//! through the same codec. The per-codec decoders (`q4_0.rs`, `q5_1.rs`, `q4_k.rs`, ...) are the
//! incumbents this walker is checked against bit for bit.
//!
//! Cost: per block the scale, min and every sub-block scale and min are decoded once; per element
//! the planes advance as odometers (one stride added, no division), so the walk is linear with a
//! small constant.

use proxima_primitives::block_layout::{BlockLayout, Coded, Field, Plane};

use crate::quant::QuantError;
use crate::quant::float_layout::decode;

const CODEC: &str = "block_layout";
const MAX_SUB_BLOCKS: usize = 32;

fn bits_at(plane: &Plane, address: usize, block: &[u8]) -> u32 {
    let mask = (1u32 << plane.bits) - 1;
    u32::from(block[address / 8] >> (address % 8)) & mask
}

fn code_value(coded: &Coded, index: usize, block: &[u8]) -> f32 {
    if coded.packed_radix > 1 {
        let mut region_index = 0;
        while region_index < usize::from(coded.region_count) {
            let region = &coded.regions[region_index];
            let start = usize::from(region.start);
            let end = start + usize::from(region.elements);
            if (start..end).contains(&index) {
                let local_index = index - start;
                let packed = bits_at(&region.source, region.source.address(local_index), block);
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
                let storage_mask = (1u32 << storage_bits) - 1;
                let residue = packed.wrapping_mul(factor) & storage_mask;
                let code = (residue * u32::from(coded.packed_radix)) >> storage_bits;
                return decode(code, coded.code);
            }
            region_index += 1;
        }
        return 0.0;
    }
    let low = bits_at(&coded.low, coded.low.address(index), block);
    let raw = coded.high.map_or(low, |high| {
        low | (bits_at(&high, high.address(index), block) << coded.low.bits)
    });
    decode(raw, coded.code)
}

fn field_value(field: &Field, block: &[u8]) -> f32 {
    let offset = usize::from(field.byte_offset);
    let bytes = usize::from(field.layout.width()) / 8;
    let code = (0..bytes).fold(0u32, |code, byte| {
        code | (u32::from(block[offset + byte]) << (8 * byte))
    });
    decode(code, field.layout)
}

fn sub_value(coded: &Coded, sub_block: usize, block: &[u8]) -> f32 {
    code_value(coded, sub_block, block) - f32::from(coded.bias)
}

struct SubTerms {
    scales: [f32; MAX_SUB_BLOCKS],
    mins: [f32; MAX_SUB_BLOCKS],
}

fn sub_terms(layout: &BlockLayout, scale: f32, min: Option<f32>, block: &[u8]) -> SubTerms {
    let mut terms = SubTerms {
        scales: [scale; MAX_SUB_BLOCKS],
        mins: [min.unwrap_or(0.0); MAX_SUB_BLOCKS],
    };
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

/// Decodes one block. `block` must be `layout.block_bytes` bytes and `output` `layout.elements`
/// values; [`dequantize`] checks both.
pub fn dequantize_block(layout: &BlockLayout, block: &[u8], output: &mut [f32]) {
    let scale = field_value(&layout.scale, block);
    let min = layout.min.map(|field| field_value(&field, block));
    let terms = sub_terms(layout, scale, min, block);
    let span = layout.sub.map_or(usize::from(layout.elements), |sub| {
        usize::from(layout.elements / sub.count)
    });
    let element = &layout.element;
    let mut low = element.low.cursor();
    let mut high = element.high.map(|plane| (plane, plane.cursor()));
    let (multiplier, bias) = (f32::from(layout.multiplier), f32::from(element.bias));
    let min_multiplier = f32::from(layout.min_multiplier);
    for (index, out) in output.iter_mut().enumerate() {
        let code = if element.packed_radix > 1 {
            code_value(element, index, block)
        } else {
            let mut raw = bits_at(&element.low, low.address(), block);
            low.advance();
            if let Some((plane, cursor)) = high.as_mut() {
                raw |= bits_at(plane, cursor.address(), block) << element.low.bits;
                cursor.advance();
            }
            decode(raw, element.code)
        };
        let sub_block = index / span;
        let value = (code * multiplier - bias) * terms.scales[sub_block];
        *out = if min.is_some() {
            value + min_multiplier * terms.mins[sub_block]
        } else {
            value
        };
    }
}

/// Decodes a run of blocks into caller-owned storage.
///
/// # Errors
/// [`QuantError::InputNotBlockMultiple`] if `data` is not a whole number of blocks;
/// [`QuantError::OutputSizeMismatch`] if `output` is not exactly the decoded element count.
pub fn dequantize(layout: &BlockLayout, data: &[u8], output: &mut [f32]) -> Result<(), QuantError> {
    let block_bytes = usize::from(layout.block_bytes);
    if block_bytes == 0 || !data.len().is_multiple_of(block_bytes) {
        return Err(QuantError::InputNotBlockMultiple {
            codec: CODEC,
            found: data.len(),
            block_bytes,
        });
    }
    let elements = usize::from(layout.elements);
    let expected = data.len() / block_bytes * elements;
    if output.len() != expected {
        return Err(QuantError::OutputSizeMismatch {
            found: output.len(),
            expected,
        });
    }
    for (block, out) in data
        .chunks_exact(block_bytes)
        .zip(output.chunks_exact_mut(elements))
    {
        dequantize_block(layout, block, out);
    }
    Ok(())
}
