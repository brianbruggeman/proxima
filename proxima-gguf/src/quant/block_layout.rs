//! The one CPU block walker: decodes any [`BlockLayout`] by reading its planes, assembling each
//! element code, and applying `(multiplier * code - bias) * scale + min`.
//!
//! Teaching pointer: `proxima_primitives::block_layout::BlockLayout` is the data, the integer
//! path of [`super::float_layout`] turns a code into its number, and the f16 scale and min decode
//! through the same codec. The per-codec decoders (`q4_0.rs`, `q5_1.rs`, ...) are the incumbents
//! this walker is checked against bit for bit.

use proxima_primitives::block_layout::{BlockLayout, Plane, PlaneOrder};

use crate::quant::QuantError;
use crate::quant::float_layout::decode;

const CODEC: &str = "block_layout";

fn plane_bits(plane: &Plane, index: usize, elements: usize, block: &[u8]) -> u32 {
    let offset = usize::from(plane.byte_offset);
    let mask = (1u32 << plane.bits) - 1;
    match plane.order {
        PlaneOrder::Contiguous => {
            let bit = index * usize::from(plane.bits);
            u32::from(block[offset + bit / 8] >> (bit % 8)) & mask
        }
        PlaneOrder::NibbleSplit => {
            let half = elements / 2;
            let shift = if index >= half { 4 } else { 0 };
            u32::from(block[offset + index % half] >> shift) & mask
        }
    }
}

fn code_at(layout: &BlockLayout, index: usize, block: &[u8]) -> u32 {
    let elements = usize::from(layout.elements);
    let low = plane_bits(&layout.low, index, elements, block);
    layout.high.map_or(low, |high| {
        low | (plane_bits(&high, index, elements, block) << layout.low.bits)
    })
}

fn field_value(field: &proxima_primitives::block_layout::Field, block: &[u8]) -> f32 {
    let offset = usize::from(field.byte_offset);
    let code = u32::from(block[offset]) | (u32::from(block[offset + 1]) << 8);
    decode(code, field.layout)
}

/// Decodes one block. `block` must be `layout.block_bytes` bytes and `output` `layout.elements`
/// values; [`dequantize`] checks both.
pub fn dequantize_block(layout: &BlockLayout, block: &[u8], output: &mut [f32]) {
    let scale = field_value(&layout.scale, block);
    let min = layout.min.map(|field| field_value(&field, block));
    for (index, out) in output.iter_mut().enumerate() {
        let level = decode(code_at(layout, index, block), layout.code);
        let value = (level * f32::from(layout.multiplier) - f32::from(layout.bias)) * scale;
        *out = min.map_or(value, |offset| value + offset);
    }
}

/// Decodes a run of blocks into caller-owned storage.
///
/// # Errors
/// [`QuantError::InputNotBlockMultiple`] if `data` is not a whole number of blocks;
/// [`QuantError::OutputSizeMismatch`] if `output` is not exactly the decoded element count.
pub fn dequantize(
    layout: &BlockLayout,
    data: &[u8],
    output: &mut [f32],
) -> Result<(), QuantError> {
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
    for (block, out) in data.chunks_exact(block_bytes).zip(output.chunks_exact_mut(elements)) {
        dequantize_block(layout, block, out);
    }
    Ok(())
}
