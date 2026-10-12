use proxima_gguf::GgmlType;
use proxima_gguf::quant::float_layout::decode;
use proxima_primitives::block_layout::BlockLayout;
use proxima_primitives::float_layout::FloatLayout;

use proxima_primitives::block_layout::Plane;

use crate::reencode::{Target, put_field, write_coded};

/// Rewrites Q8_0 as Q8_K: one f32 scale over 256 elements, `absmax / 127`.
pub fn encode_q8_k(_source: &[u8], values: &[f32]) -> Vec<u8> {
    let layout = BlockLayout::Q8_K;
    let (elements, block_bytes) = (
        usize::from(layout.elements),
        usize::from(layout.block_bytes),
    );
    let mut bytes = vec![0u8; values.len() / elements * block_bytes];
    for (chunk, block) in values
        .chunks_exact(elements)
        .zip(bytes.chunks_exact_mut(block_bytes))
    {
        let magnitude = chunk.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
        let scale = if magnitude > 0.0 {
            magnitude / 127.0
        } else {
            1.0
        };
        put_field(layout.scale.layout, scale, layout.scale.byte_offset, block);
        for (index, &value) in chunk.iter().enumerate() {
            let level = (value / scale).round().clamp(-128.0, 127.0) as i32;
            write_coded(&layout.element, index, (level as u32) & 0xff, block);
        }
    }
    bytes
}

/// Rewrites Q4_0 as Q6_K keeping every shipped level exact: a Q4_0 level `q - 8` becomes the
/// Q6_K code `4 * (q - 8) + 32`, and the Q4_0 block scale becomes `d * sub_scale / 4` with one
/// shared f16 `d` per 256 elements and an i8 sub-scale per 16.
pub fn encode_q6_k_from_q4_0(source: &[u8], values: &[f32]) -> Vec<u8> {
    let layout = BlockLayout::Q6_K;
    let sub = layout.sub.expect("Q6_K has sub-blocks");
    let (source_bytes, source_elements) = (18usize, 32usize);
    let scale_of = |index: usize| {
        let at = index * source_bytes;
        decode(
            u32::from(source[at]) | (u32::from(source[at + 1]) << 8),
            FloatLayout::F16,
        )
    };
    let block_count = values.len() / usize::from(layout.elements);
    let mut bytes = vec![0u8; block_count * usize::from(layout.block_bytes)];
    for (block_index, block) in bytes
        .chunks_exact_mut(usize::from(layout.block_bytes))
        .enumerate()
    {
        let first = block_index * 8;
        let largest = (first..first + 8).fold(0.0f32, |acc, index| acc.max(scale_of(index).abs()));
        let super_scale = if largest > 0.0 {
            largest / (4.0 * 127.0)
        } else {
            1.0
        };
        let super_scale = put_field(
            layout.scale.layout,
            super_scale,
            layout.scale.byte_offset,
            block,
        );
        for sub_block in 0..usize::from(sub.count) {
            let shipped = scale_of(first + sub_block / 2);
            let sub_scale = (shipped / (4.0 * super_scale)).round().clamp(-128.0, 127.0) as i32;
            write_coded(&sub.scale, sub_block, (sub_scale as u32) & 0xff, block);
        }
        let shipped_scales: Vec<f32> = (0..8).map(|offset| scale_of(first + offset)).collect();
        let mut low = layout.element.low.cursor();
        let mut high = layout.element.high.map(|plane| plane.cursor());
        for index in 0..usize::from(layout.elements) {
            let shipped = shipped_scales[index / source_elements];
            let level = if shipped == 0.0 {
                0
            } else {
                (values[block_index * 256 + index] / shipped).round() as i32
            };
            let code = (4 * level + 32).clamp(0, 63) as u32;
            write_at(&layout.element.low, low.address(), code, block);
            low.advance();
            if let (Some(plane), Some(cursor)) = (&layout.element.high, high.as_mut()) {
                write_at(
                    plane,
                    cursor.address(),
                    code >> layout.element.low.bits,
                    block,
                );
                cursor.advance();
            }
        }
    }
    bytes
}

/// Shipped codec to the K-family codec: Q8_0 goes to Q8_K and Q4_0 to Q6_K.
pub fn k_target(shipped: GgmlType) -> Option<Target> {
    match shipped {
        GgmlType::Q8_0 => match std::env::var("FLD_K_Q8_0_TARGET").as_deref() {
            Ok("q6_k") => Some(Target {
                kind: GgmlType::Q6_K,
                encode: encode_q6_k_fit,
            }),
            _ => Some(Target {
                kind: GgmlType::Q8_K,
                encode: encode_q8_k,
            }),
        },
        GgmlType::Q4_0 => Some(Target {
            kind: GgmlType::Q6_K,
            encode: encode_q6_k_from_q4_0,
        }),
        _ => None,
    }
}

/// Refits any shipped tensor into Q6_K with the crate's own K-quant encoder.
pub fn encode_q6_k_fit(_source: &[u8], values: &[f32]) -> Vec<u8> {
    let layout = BlockLayout::Q6_K;
    let mut bytes =
        vec![0u8; values.len() / usize::from(layout.elements) * usize::from(layout.block_bytes)];
    proxima_gguf::quant::q6_k::quantize(values, &mut bytes).expect("finite weights quantize");
    bytes
}

fn write_at(plane: &Plane, address: usize, value: u32, block: &mut [u8]) {
    block[address / 8] |= ((value & ((1 << plane.bits) - 1)) << (address % 8)) as u8;
}
