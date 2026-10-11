#![allow(clippy::unwrap_used, clippy::expect_used)]

//! The GGUF edge of the e3m2 codec: the blocks travel under a proxima-own tensor type id and
//! never reach a ggml or safetensors consumer. Composes `proxima_gguf::write_complete` /
//! `parse_complete` (the transform's writer path, see `safetensors_to_gguf`) and
//! `proxima_gguf::quant::scaled_float` (the block encoder).

use std::collections::BTreeMap;
use std::path::Path;

use proxima_gguf::quant::q8_0;
use proxima_gguf::quant::scaled_float::{BLOCK_BYTES, BLOCK_ELEMENTS, quantize_block};
use proxima_gguf::tensor::MAX_DIMS;
use proxima_gguf::{GgmlType, GgufModel, TensorPayload, parse_complete, write_complete};
use proxima_model_interop::InteropError;
use proxima_model_interop::dtype_to_ggml;
use proxima_model_interop::safetensors_to_gguf;
use proxima_primitives::float_layout::FloatLayout;
use proxima_safetensors::{SafetensorsModel, TensorPayload as SafetensorsPayload};
use proxima_tensor::DType;

const GRANITE_DEFAULT: &str = "/Users/brianbruggeman/.ollama/models/blobs/\
     sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80";
const TENSOR_NAME: &str = "blk.0.ffn_gate.weight";

fn granite_q8_0_weights() -> Vec<f32> {
    let path =
        std::env::var("PROXIMA_ARCH_GRANITE_MOE_GGUF").unwrap_or_else(|_| GRANITE_DEFAULT.into());
    assert!(
        Path::new(&path).exists(),
        "no granite gguf at {path}: set PROXIMA_ARCH_GRANITE_MOE_GGUF or stage the blob"
    );
    let (parsed, bytes) =
        proxima_gguf::edge::read_file(Path::new(&path)).expect("granite blob must parse as gguf");
    let tensor = parsed
        .tensors
        .iter()
        .find(|tensor| tensor.ggml_type == GgmlType::Q8_0 && tensor.dims.as_slice() == [1024, 512])
        .expect("granite holds a 1024x512 q8_0 tensor");
    let range = parsed
        .tensor_data_range(tensor, bytes.len() as u64)
        .expect("tensor range must fit the file");
    let mut weights = vec![0.0f32; tensor.element_count() as usize];
    q8_0::dequantize(
        &bytes[range.start as usize..range.end as usize],
        &mut weights,
    )
    .expect("well-formed q8_0 tensor");
    weights
}

fn e3m2_blocks(weights: &[f32]) -> Vec<u8> {
    weights
        .as_chunks::<BLOCK_ELEMENTS>()
        .0
        .iter()
        .flat_map(|source| {
            let mut block = [0u8; BLOCK_BYTES];
            quantize_block(source, FloatLayout::E3M2, &mut block).expect("finite granite block");
            block
        })
        .collect()
}

#[test]
fn fp6_e3m2_gguf_round_trip_bytes_identical() {
    let weights = granite_q8_0_weights();
    let blocks = e3m2_blocks(&weights);
    let mut dims = arrayvec::ArrayVec::<u64, MAX_DIMS>::new();
    dims.push(1024);
    dims.push(512);
    let model = GgufModel {
        version: 3,
        metadata: Vec::new(),
        tensors: vec![TensorPayload {
            name: TENSOR_NAME.into(),
            dims,
            ggml_type: GgmlType::Fp6E3M2,
            data: &blocks,
        }],
    };

    let file = write_complete(&model).expect("writes the e3m2 tensor");
    let parsed = parse_complete(&file).expect("parses the e3m2 tensor back");

    let tensor = &parsed.tensors[0];
    let range = parsed
        .tensor_data_range(tensor, file.len() as u64)
        .expect("range fits the file");
    assert_eq!(tensor.ggml_type, GgmlType::Fp6E3M2);
    assert_eq!(blocks.len(), weights.len() / BLOCK_ELEMENTS * BLOCK_BYTES);
    assert_eq!(
        &file[range.start as usize..range.end as usize],
        blocks.as_slice()
    );
}

#[test]
fn fp6_e3m2_gguf_export_rejected_dtype_to_ggml_is_none() {
    assert_eq!(dtype_to_ggml(DType::Fp6E3M2), None);
}

#[test]
fn fp6_e3m2_gguf_export_rejected_transform_returns_unrepresentable_dtype() {
    let block = [0u8; BLOCK_BYTES];
    let model = SafetensorsModel {
        tensors: vec![SafetensorsPayload {
            name: TENSOR_NAME.into(),
            dtype: DType::Fp6E3M2,
            shape: vec![BLOCK_ELEMENTS as u64],
            data: &block,
        }],
        metadata: BTreeMap::new(),
    };

    let outcome = safetensors_to_gguf(&model);

    assert!(matches!(
        outcome,
        Err(InteropError::UnrepresentableDType { dtype: DType::Fp6E3M2, ref tensor }) if tensor == TENSOR_NAME
    ));
}
