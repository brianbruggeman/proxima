use proxima_gguf::quant::dispatch;
use proxima_gguf::quant::float_layout::{decode, encode};
use proxima_gguf::{GgmlType, GgufModel, ParsedGguf, TensorPayload, write_complete};
use proxima_primitives::block_layout::{Coded, Plane};
use proxima_primitives::float_layout::FloatLayout;

/// Where a shipped tensor goes: its family codec and the function that produces the family
/// bytes from the shipped bytes and their decoded values.
pub struct Target {
    pub kind: GgmlType,
    pub encode: fn(&[u8], &[f32]) -> Vec<u8>,
}

pub fn write_bits(plane: &Plane, index: usize, value: u32, block: &mut [u8]) {
    let address = plane.address(index);
    block[address / 8] |= ((value & ((1 << plane.bits) - 1)) << (address % 8)) as u8;
}

pub fn write_coded(coded: &Coded, index: usize, raw: u32, block: &mut [u8]) {
    write_bits(&coded.low, index, raw, block);
    if let Some(high) = &coded.high {
        write_bits(high, index, raw >> coded.low.bits, block);
    }
}

pub fn put_field(layout: FloatLayout, value: f32, offset: u16, block: &mut [u8]) -> f32 {
    let code = encode(value, layout);
    let at = usize::from(offset);
    for byte in 0..usize::from(layout.width()) / 8 {
        block[at + byte] = (code >> (8 * byte)) as u8;
    }
    decode(code, layout)
}

pub struct Reencoded {
    pub bytes: Vec<u8>,
    pub counts: Vec<(GgmlType, usize)>,
}

/// Rewrites every two-dimensional `blk.` weight whose shipped codec `target_of` names by
/// dequantizing through the dispatch seam and encoding into the target; every other tensor is
/// carried over byte for byte.
pub fn reencode(
    parsed: &ParsedGguf,
    file: &[u8],
    target_of: fn(GgmlType) -> Option<Target>,
) -> Reencoded {
    let mut owned: Vec<Option<(GgmlType, Vec<u8>)>> = Vec::new();
    for tensor in &parsed.tensors {
        let range = parsed
            .tensor_data_range(tensor, file.len() as u64)
            .expect("tensor range fits the file");
        let data = &file[range.start as usize..range.end as usize];
        let target = target_of(tensor.ggml_type)
            .filter(|_| tensor.dims.len() == 2 && tensor.name.starts_with("blk."));
        owned.push(target.map(|target| {
            let mut values = vec![0.0f32; tensor.element_count() as usize];
            dispatch::dequantize(tensor.ggml_type, data, &mut values)
                .expect("shipped tensor decodes");
            (target.kind, (target.encode)(data, &values))
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
    let mut counts: Vec<(GgmlType, usize)> = Vec::new();
    for (kind, _) in owned.iter().flatten() {
        match counts.iter_mut().find(|(found, _)| found == kind) {
            Some((_, count)) => *count += 1,
            None => counts.push((*kind, 1)),
        }
    }
    Reencoded { bytes, counts }
}
