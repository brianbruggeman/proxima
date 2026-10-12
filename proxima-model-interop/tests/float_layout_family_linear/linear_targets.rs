use proxima_gguf::GgmlType;
use proxima_primitives::block_layout::BlockLayout;

use proxima_primitives::block_layout::Coded;
use proxima_primitives::float_layout::SignEncoding;

use crate::reencode::{Target, put_field, write_coded};

/// Fits a scale (and min) to one block of values and writes the codes of a linear layout: a
/// min-max fit when the layout carries a min, an absolute-maximum fit over the symmetric part
/// of the level range otherwise (or over `limit` levels when given). Only layouts with
/// `multiplier` 1 and no sub-blocks are encodable.
pub fn encode_block(layout: &BlockLayout, values: &[f32], limit: Option<f32>, block: &mut [u8]) {
    assert_eq!(
        layout.multiplier, 1,
        "the fit encoder serves multiplier-1 layouts"
    );
    assert!(
        layout.sub.is_none(),
        "the fit encoder serves linear layouts"
    );
    let bias = i32::from(layout.element.bias);
    let (lowest, highest) = code_range(&layout.element);
    let (low_value, high_value) = (lowest - bias, highest - bias);
    let (floor, ceiling) = values.iter().fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(floor, ceiling), &value| (floor.min(value), ceiling.max(value)),
    );
    let (scale, minimum) = if layout.min.is_some() {
        ((ceiling - floor) / (highest - lowest) as f32, floor)
    } else {
        let magnitude = floor.abs().max(ceiling.abs());
        let limit = limit.unwrap_or_else(|| low_value.abs().min(high_value).max(1) as f32);
        (magnitude / limit, 0.0)
    };
    let scale = put_field(
        layout.scale.layout,
        if scale > 0.0 { scale } else { 1.0 },
        layout.scale.byte_offset,
        block,
    );
    let minimum = match &layout.min {
        Some(field) => put_field(field.layout, minimum, field.byte_offset, block),
        None => minimum,
    };
    let bits = layout.element.low.bits + layout.element.high.map_or(0, |plane| plane.bits);
    for (index, &value) in values.iter().enumerate() {
        let level = ((value - minimum) / scale).round() as i32;
        let code = (level + bias).clamp(lowest, highest);
        let raw = (code as u32) & ((1u32 << bits) - 1);
        write_coded(&layout.element, index, raw, block);
    }
}

fn encode_linear(layout: &BlockLayout, limit: f32, values: &[f32]) -> Vec<u8> {
    let (elements, block_bytes) = (
        usize::from(layout.elements),
        usize::from(layout.block_bytes),
    );
    let mut bytes = vec![0u8; values.len() / elements * block_bytes];
    for (chunk, block) in values
        .chunks_exact(elements)
        .zip(bytes.chunks_exact_mut(block_bytes))
    {
        encode_block(layout, chunk, Some(limit), block);
    }
    bytes
}

pub fn encode_q8_1(_source: &[u8], values: &[f32]) -> Vec<u8> {
    encode_linear(&BlockLayout::Q8_1, 127.0, values)
}

pub fn encode_q5_0(_source: &[u8], values: &[f32]) -> Vec<u8> {
    encode_linear(&BlockLayout::Q5_0, 8.0, values)
}

/// Shipped codec to the linear-family codec that replaces it: Q8_0 goes to Q8_1 and Q4_0 to
/// Q5_0, each target holds every shipped level exactly, so the re-encode keeps the shipped scale
/// and loses nothing.
pub fn linear_target(shipped: GgmlType) -> Option<Target> {
    match shipped {
        GgmlType::Q8_0 => Some(Target {
            kind: GgmlType::Q8_1,
            encode: encode_q8_1,
        }),
        GgmlType::Q4_0 => Some(Target {
            kind: GgmlType::Q5_0,
            encode: encode_q5_0,
        }),
        _ => None,
    }
}
fn code_range(coded: &Coded) -> (i32, i32) {
    let bits = coded.low.bits + coded.high.map_or(0, |plane| plane.bits);
    if coded.code.sign == SignEncoding::Absent {
        (0, (1i32 << bits) - 1)
    } else {
        (-(1i32 << (bits - 1)), (1i32 << (bits - 1)) - 1)
    }
}
