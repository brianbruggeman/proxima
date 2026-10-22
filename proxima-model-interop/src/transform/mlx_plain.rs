//! Plain (float) MLX export: every GGUF weight tensor as bf16, f16 or f32.
//!
//! The output is a list of [`PlainTensor`] (name, shape, dtype, bytes), the
//! `(name, dtype, shape, bytes)` quadruple both formats agree on (see
//! [`super::gguf_to_safetensors`], which refuses block-quantized sources
//! where this path decodes them). Q8_0 and Q4_0 rows decode to `f32`
//! through [`proxima_gguf::quant::dispatch::dequantize`] and are then cast
//! to the export dtype with round-to-nearest-even by the `half` crate.
//!
//! Decoding is one GGUF row at a time into a single scratch row, so the
//! only allocations are the scratch row and each tensor's output buffer.
//! `MlxExport::assemble` (slice 1) consumes the returned tensors.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use half::{bf16, f16};
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::quant::dispatch::dequantize;
use proxima_gguf::{GgmlType, GgufError, TensorInfo};
use proxima_tensor::DType;

use crate::error::InteropError;

/// The dtype mlx-community's plain conversions store.
pub const DEFAULT_PLAIN_DTYPE: DType = DType::BFLOAT16;

/// One exported float tensor. `shape` is row-major (GGUF `dims` reversed),
/// so the last entry is the contiguous axis the GGUF stores first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlainTensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: DType,
    pub bytes: Vec<u8>,
}

/// Export every tensor of `parsed` as `dtype` (`BFloat16`, `Float16` or
/// `Float32`; pass [`DEFAULT_PLAIN_DTYPE`] for the default).
///
/// # Errors
///
/// [`InteropError::UnrepresentableDType`] for any other `dtype`;
/// [`InteropError::Quant`] when a source type has no decoder or its bytes
/// are not whole blocks; [`InteropError::Gguf`] for a tensor range outside
/// `file_bytes`.
pub fn export_plain(
    parsed: &ParsedGguf,
    file_bytes: &[u8],
    dtype: DType,
) -> Result<Vec<PlainTensor>, InteropError> {
    parsed
        .tensors
        .iter()
        .map(|tensor| export_plain_leaf(parsed, file_bytes, tensor, dtype))
        .collect()
}

pub(super) fn export_plain_leaf(
    parsed: &ParsedGguf,
    file_bytes: &[u8],
    tensor: &TensorInfo,
    dtype: DType,
) -> Result<PlainTensor, InteropError> {
    let range = parsed.tensor_data_range(tensor, file_bytes.len() as u64)?;
    export_plain_tensor(
        tensor,
        &file_bytes[range.start as usize..range.end as usize],
        dtype,
    )
}

/// One tensor of [`export_plain`]; `data` is exactly the tensor's bytes.
///
/// # Errors
///
/// As [`export_plain`].
pub fn export_plain_tensor(
    tensor: &TensorInfo,
    data: &[u8],
    dtype: DType,
) -> Result<PlainTensor, InteropError> {
    let row_elements = tensor.dims.first().copied().unwrap_or(0) as usize;
    let rows = tensor
        .element_count()
        .checked_div(row_elements as u64)
        .ok_or(GgufError::Overflow {
            context: "tensor row length",
        })? as usize;
    let row_bytes = data.len().checked_div(rows).unwrap_or(0);
    let mut scratch = vec![0.0_f32; row_elements];
    let mut bytes = Vec::with_capacity(rows * row_elements * dtype.size_bytes());
    for row in data.chunks_exact(row_bytes.max(1)) {
        decode_row(tensor.ggml_type, row, &mut scratch)?;
        cast_row(dtype, &scratch, &mut bytes).ok_or_else(|| {
            InteropError::UnrepresentableDType {
                tensor: tensor.name.clone(),
                dtype,
            }
        })?;
    }
    Ok(PlainTensor {
        name: tensor.name.clone(),
        shape: tensor.dims.iter().rev().copied().collect(),
        dtype,
        bytes,
    })
}

fn decode_row(kind: GgmlType, row: &[u8], scratch: &mut [f32]) -> Result<(), InteropError> {
    if kind == GgmlType::F32 {
        row.as_chunks::<4>()
            .0
            .iter()
            .zip(scratch.iter_mut())
            .for_each(|(word, slot)| *slot = f32::from_le_bytes(*word));
        return Ok(());
    }
    Ok(dequantize(kind, row, scratch)?)
}

fn cast_row(dtype: DType, values: &[f32], out: &mut Vec<u8>) -> Option<()> {
    if dtype == DType::BFLOAT16 {
        values
            .iter()
            .for_each(|&value| out.extend_from_slice(&bf16::from_f32(value).to_le_bytes()));
    } else if dtype == DType::FLOAT16 {
        values
            .iter()
            .for_each(|&value| out.extend_from_slice(&f16::from_f32(value).to_le_bytes()));
    } else if dtype == DType::FLOAT32 {
        values
            .iter()
            .for_each(|&value| out.extend_from_slice(&value.to_le_bytes()));
    } else {
        return None;
    }
    Some(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::fs::File;
    use std::path::Path;

    use arrayvec::ArrayVec;
    use proxima_gguf::{GgufModel, TensorPayload, parse_complete, write_complete};

    use super::*;

    const GRANITE_GGUF: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80";

    const SAMPLE_F32_BITS: [u32; 6] = [
        0x3F80_0000,
        0xC000_0000,
        0x3F00_0000,
        0x477F_E000,
        0x3F80_8000,
        0x3F81_8000,
    ];
    const SAMPLE_BF16_BITS: [u16; 6] = [0x3F80, 0xC000, 0x3F00, 0x4780, 0x3F80, 0x3F82];
    const SAMPLE_F16_BITS: [u16; 6] = [0x3C00, 0xC000, 0x3800, 0x7BFF, 0x3C04, 0x3C0C];

    fn gguf_with(name: &str, dims: &[u64], kind: GgmlType, data: &[u8]) -> Vec<u8> {
        let dims: ArrayVec<u64, 4> = dims.iter().copied().collect();
        write_complete(&GgufModel {
            version: 3,
            metadata: Vec::new(),
            tensors: vec![TensorPayload {
                name: name.to_string(),
                dims,
                ggml_type: kind,
                data,
            }],
        })
        .expect("writes the single-tensor gguf")
    }

    fn export_single(bytes: &[u8], dtype: DType) -> PlainTensor {
        let parsed = parse_complete(bytes).expect("parses the gguf just written");
        export_plain(&parsed, bytes, dtype)
            .expect("exports")
            .remove(0)
    }

    fn sample_f32_tensor(dtype: DType) -> PlainTensor {
        let data: Vec<u8> = SAMPLE_F32_BITS
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect();
        export_single(
            &gguf_with("sample.weight", &[3, 2], GgmlType::F32, &data),
            dtype,
        )
    }

    fn le_bytes(words: &[u16]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    fn with_one_byte_altered(bytes: &[u8]) -> Vec<u8> {
        let mut altered = bytes.to_vec();
        altered[0] ^= 0x01;
        altered
    }

    fn first_granite_f32_tensor(run: impl FnOnce(&TensorInfo, &[u8])) {
        assert!(
            Path::new(GRANITE_GGUF).exists(),
            "host-local gguf blob missing at {GRANITE_GGUF}"
        );
        let file = File::open(GRANITE_GGUF).unwrap();
        let mapping = unsafe { memmap2::Mmap::map(&file) }.unwrap();
        let parsed = parse_complete(&mapping).unwrap();
        let tensor = parsed
            .tensors
            .iter()
            .find(|tensor| tensor.ggml_type == GgmlType::F32 && tensor.dims.len() == 1)
            .expect("granite carries an f32 norm vector");
        let range = parsed
            .tensor_data_range(tensor, mapping.len() as u64)
            .unwrap();
        run(tensor, &mapping[range.start as usize..range.end as usize]);
    }

    fn real_values(data: &[u8]) -> Vec<f32> {
        data.as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect()
    }

    fn q8_0_gguf() -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&0x3800_u16.to_le_bytes());
        data.extend((0..32_i8).map(|index| (index - 16) as u8));
        data.extend_from_slice(&0x3400_u16.to_le_bytes());
        data.extend((0..32).map(|index| if index == 0 { 0x80 } else { 0x7F }));
        gguf_with("blk.0.ffn_down.weight", &[32, 2], GgmlType::Q8_0, &data)
    }

    fn q8_0_expected() -> Vec<f32> {
        let mut values: Vec<f32> = (0..32).map(|index| (index as f32 - 16.0) * 0.5).collect();
        values.push(-32.0);
        values.extend([31.75; 31]);
        values
    }

    fn q4_0_gguf() -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&0x4000_u16.to_le_bytes());
        data.extend((0..16_u8).map(|index| index | ((15 - index) << 4)));
        data.extend_from_slice(&0x3800_u16.to_le_bytes());
        data.extend([0x80_u8; 16]);
        gguf_with("blk.0.attn_q.weight", &[32, 2], GgmlType::Q4_0, &data)
    }

    fn q4_0_expected() -> Vec<f32> {
        let mut values: Vec<f32> = (0..16).map(|nibble| 2.0 * (nibble as f32 - 8.0)).collect();
        values.extend((0..16).map(|index| 2.0 * ((15 - index) as f32 - 8.0)));
        values.extend([-4.0; 16]);
        values.extend([0.0; 16]);
        values
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    #[test]
    fn mlx_plain_bf16_default_bytes_equal_hand_derived_and_half_bits_on_granite() {
        let exported = sample_f32_tensor(DEFAULT_PLAIN_DTYPE);

        assert_eq!(exported.dtype, DType::BFLOAT16);
        assert_eq!(exported.shape, vec![2, 3]);
        assert_eq!(exported.bytes, le_bytes(&SAMPLE_BF16_BITS));
        first_granite_f32_tensor(|tensor, data| {
            let real = export_plain_tensor(tensor, data, DEFAULT_PLAIN_DTYPE).unwrap();
            let expected: Vec<u8> = real_values(data)
                .iter()
                .flat_map(|value| bf16::from_f32(*value).to_le_bytes())
                .collect();
            assert_eq!(real.bytes, expected);
        });
    }

    #[test]
    fn mlx_plain_bf16_control_altered_expected_byte_differs() {
        let exported = sample_f32_tensor(DEFAULT_PLAIN_DTYPE);

        assert_ne!(
            exported.bytes,
            with_one_byte_altered(&le_bytes(&SAMPLE_BF16_BITS))
        );
    }

    #[test]
    fn mlx_plain_f16_option_bytes_equal_hand_derived_and_half_bits_on_granite() {
        let exported = sample_f32_tensor(DType::FLOAT16);

        assert_eq!(exported.dtype, DType::FLOAT16);
        assert_eq!(exported.bytes, le_bytes(&SAMPLE_F16_BITS));
        first_granite_f32_tensor(|tensor, data| {
            let real = export_plain_tensor(tensor, data, DType::FLOAT16).unwrap();
            let expected: Vec<u8> = real_values(data)
                .iter()
                .flat_map(|value| f16::from_f32(*value).to_le_bytes())
                .collect();
            assert_eq!(real.bytes, expected);
        });
    }

    #[test]
    fn mlx_plain_f16_control_altered_expected_byte_differs() {
        let exported = sample_f32_tensor(DType::FLOAT16);

        assert_ne!(
            exported.bytes,
            with_one_byte_altered(&le_bytes(&SAMPLE_F16_BITS))
        );
    }

    #[test]
    fn mlx_plain_f32_option_bytes_equal_source_bits_and_granite_bytes() {
        let exported = sample_f32_tensor(DType::FLOAT32);
        let expected: Vec<u8> = SAMPLE_F32_BITS
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect();

        assert_eq!(exported.dtype, DType::FLOAT32);
        assert_eq!(exported.bytes, expected);
        first_granite_f32_tensor(|tensor, data| {
            let real = export_plain_tensor(tensor, data, DType::FLOAT32).unwrap();
            assert_eq!(real.bytes, data);
        });
    }

    #[test]
    fn mlx_plain_f32_control_altered_expected_byte_differs() {
        let exported = sample_f32_tensor(DType::FLOAT32);
        let expected: Vec<u8> = SAMPLE_F32_BITS
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect();

        assert_ne!(exported.bytes, with_one_byte_altered(&expected));
    }

    #[test]
    fn mlx_plain_q8_0_block_equals_hand_derived_ggml_layout_values() {
        let exported = export_single(&q8_0_gguf(), DType::FLOAT32);

        assert_eq!(exported.shape, vec![2, 32]);
        assert_eq!(exported.bytes, f32_bytes(&q8_0_expected()));
    }

    #[test]
    fn mlx_plain_q8_0_control_altered_expected_byte_differs() {
        let exported = export_single(&q8_0_gguf(), DType::FLOAT32);

        assert_ne!(
            exported.bytes,
            with_one_byte_altered(&f32_bytes(&q8_0_expected()))
        );
    }

    #[test]
    fn mlx_plain_q4_0_block_equals_hand_derived_ggml_layout_values() {
        let exported = export_single(&q4_0_gguf(), DType::FLOAT32);

        assert_eq!(exported.shape, vec![2, 32]);
        assert_eq!(exported.bytes, f32_bytes(&q4_0_expected()));
    }

    #[test]
    fn mlx_plain_q4_0_control_altered_expected_byte_differs() {
        let exported = export_single(&q4_0_gguf(), DType::FLOAT32);

        assert_ne!(
            exported.bytes,
            with_one_byte_altered(&f32_bytes(&q4_0_expected()))
        );
    }
}
