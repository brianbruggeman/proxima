use proxima_gguf::quant::dispatch;
use proxima_gguf::quant::float_layout::{decode, encode};
use proxima_gguf::{GgmlType, GgufModel, ParsedGguf, TensorPayload, write_complete};
use proxima_primitives::block_layout::{BlockLayout, Plane, PlaneOrder};
use proxima_primitives::float_layout::FloatLayout;

/// Shipped codec to the family codec that replaces it, with the level limit the shipped grid
/// used: Q8_0 goes to Q8_1 and Q4_0 to Q5_0, each target holds every shipped level exactly, so
/// the re-encode keeps the shipped scale and loses nothing.
pub fn target_of(shipped: GgmlType) -> Option<(GgmlType, BlockLayout, f32)> {
    match shipped {
        GgmlType::Q8_0 => Some((GgmlType::Q8_1, BlockLayout::Q8_1, 127.0)),
        GgmlType::Q4_0 => match std::env::var("FLD_LINEAR_Q4_0_TARGET").as_deref() {
            Ok("none") => None,
            Ok("q4_1") => Some((GgmlType::Q4_1, BlockLayout::Q4_1, 8.0)),
            Ok("q5_1") => Some((GgmlType::Q5_1, BlockLayout::Q5_1, 8.0)),
            _ => Some((GgmlType::Q5_0, BlockLayout::Q5_0, 8.0)),
        },
        _ => None,
    }
}

fn write_plane(plane: &Plane, index: usize, elements: usize, value: u32, block: &mut [u8]) {
    let offset = usize::from(plane.byte_offset);
    match plane.order {
        PlaneOrder::Contiguous => {
            let bit = index * usize::from(plane.bits);
            block[offset + bit / 8] |= ((value & ((1 << plane.bits) - 1)) << (bit % 8)) as u8;
        }
        PlaneOrder::NibbleSplit => {
            let half = elements / 2;
            let shift = if index >= half { 4 } else { 0 };
            block[offset + index % half] |= ((value & 0xf) << shift) as u8;
        }
    }
}

fn put_field(layout: FloatLayout, value: f32, offset: u16, block: &mut [u8]) -> f32 {
    let code = encode(value, layout);
    let at = usize::from(offset);
    block[at] = code as u8;
    block[at + 1] = (code >> 8) as u8;
    decode(code, layout)
}

/// Fits a scale (and min) to one block of values and writes the codes: a min-max fit when the
/// layout carries a min, an absolute-maximum fit over the symmetric part of the level range
/// otherwise. Only layouts with `multiplier` 1 are encodable.
pub fn encode_block(layout: &BlockLayout, values: &[f32], limit: Option<f32>, block: &mut [u8]) {
    assert_eq!(
        layout.multiplier, 1,
        "the fit encoder serves multiplier-1 layouts"
    );
    let code_bits = layout.low.bits + layout.high.map_or(0, |plane| plane.bits);
    let signed = layout.code.sign != proxima_primitives::float_layout::SignEncoding::Absent;
    let (lowest, highest) = if signed {
        (-(1i32 << (code_bits - 1)), (1i32 << (code_bits - 1)) - 1)
    } else {
        (0, (1i32 << code_bits) - 1)
    };
    let (low_value, high_value) = (
        lowest - i32::from(layout.bias),
        highest - i32::from(layout.bias),
    );
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
    for (index, &value) in values.iter().enumerate() {
        let level = ((value - minimum) / scale).round() as i32;
        let code = (level + i32::from(layout.bias)).clamp(lowest, highest);
        let raw = (code as u32) & ((1u32 << code_bits) - 1);
        let elements = usize::from(layout.elements);
        write_plane(&layout.low, index, elements, raw, block);
        if let Some(high) = &layout.high {
            write_plane(high, index, elements, raw >> layout.low.bits, block);
        }
    }
}

pub struct Reencoded {
    pub bytes: Vec<u8>,
    pub counts: Vec<(GgmlType, usize)>,
}

/// Rewrites every two-dimensional block weight of a shipped family codec into its family target
/// by dequantizing through the dispatch seam and fitting the target layout; every other tensor
/// is carried over byte for byte.
pub fn reencode(parsed: &ParsedGguf, file: &[u8]) -> Reencoded {
    let mut owned: Vec<Option<(GgmlType, Vec<u8>)>> = Vec::new();
    for tensor in &parsed.tensors {
        let range = parsed
            .tensor_data_range(tensor, file.len() as u64)
            .expect("tensor range fits the file");
        let data = &file[range.start as usize..range.end as usize];
        let target = target_of(tensor.ggml_type)
            .filter(|_| tensor.dims.len() == 2 && tensor.name.starts_with("blk."));
        owned.push(target.map(|(kind, layout, limit)| {
            let mut values = vec![0.0f32; tensor.element_count() as usize];
            dispatch::dequantize(tensor.ggml_type, data, &mut values)
                .expect("shipped tensor decodes");
            let (elements, block_bytes) = (
                usize::from(layout.elements),
                usize::from(layout.block_bytes),
            );
            let mut bytes = vec![0u8; values.len() / elements * block_bytes];
            for (chunk, block) in values
                .chunks_exact(elements)
                .zip(bytes.chunks_exact_mut(block_bytes))
            {
                encode_block(&layout, chunk, Some(limit), block);
            }
            (kind, bytes)
        }));
    }
    let tensors = parsed
        .tensors
        .iter()
        .zip(&owned)
        .map(|(tensor, replacement)| {
            let range = parsed
                .tensor_data_range(tensor, file.len() as u64)
                .expect("tensor range fits the file");
            let (ggml_type, data) = match replacement {
                Some((kind, bytes)) => (*kind, bytes.as_slice()),
                None => (
                    tensor.ggml_type,
                    &file[range.start as usize..range.end as usize],
                ),
            };
            TensorPayload {
                name: tensor.name.clone(),
                dims: tensor.dims.clone(),
                ggml_type,
                data,
            }
        })
        .collect();
    let model = GgufModel {
        version: parsed.version,
        metadata: parsed.metadata.clone(),
        tensors,
    };
    let bytes = write_complete(&model).expect("the re-encoded model serializes");
    let counts = [
        GgmlType::Q8_1,
        GgmlType::Q5_1,
        GgmlType::Q4_1,
        GgmlType::Q5_0,
    ]
    .into_iter()
    .map(|kind| {
        let count = owned
            .iter()
            .flatten()
            .filter(|(found, _)| *found == kind)
            .count();
        (kind, count)
    })
    .collect();
    Reencoded { bytes, counts }
}
