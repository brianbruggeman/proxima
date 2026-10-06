use alloc::string::ToString;
use alloc::vec;

use proxima_gguf::{GgmlType, parse_complete, write_complete};
use proxima_safetensors::parse_complete as parse_safetensors;
use proxima_tensor::DType;

use super::*;

fn pattern_bytes(len: usize) -> Vec<u8> {
    (0..len).map(|index| (index % 251) as u8).collect()
}

fn dims(values: &[u64]) -> ArrayVec<u64, MAX_DIMS> {
    values.iter().copied().collect()
}

/// Builds a small GGUF model with mixed metadata types (string, u32,
/// bool, and an array) and two mappable-dtype tensors, then drives it
/// GGUF -> safetensors -> GGUF and checks: every tensor's name, dtype,
/// shape, and bytes are exact; every metadata value survives as text,
/// exactly matching the documented lossy-metadata contract (a `U32`
/// value comes back as the `String` its `Debug` output would produce,
/// not as a `U32`).
#[test]
fn round_trip_preserves_every_tensor_byte_and_documents_metadata_loss() {
    let embedding_data = pattern_bytes(4 * 4 * 4); // 4x4 f32
    let norm_data = pattern_bytes(3 * 2); // 3 f16

    let original_metadata = vec![
        (
            "general.architecture".to_string(),
            MetadataValue::String("llama".to_string()),
        ),
        ("llama.context_length".to_string(), MetadataValue::U32(4096)),
        ("general.quantized".to_string(), MetadataValue::Bool(false)),
    ];

    let gguf_model = GgufModel {
        version: 3,
        metadata: original_metadata.clone(),
        tensors: vec![
            GgufTensorPayload {
                name: "token_embd.weight".to_string(),
                dims: dims(&[4, 4]),
                ggml_type: GgmlType::F32,
                data: embedding_data.as_slice(),
            },
            GgufTensorPayload {
                name: "output_norm.weight".to_string(),
                dims: dims(&[3]),
                ggml_type: GgmlType::F16,
                data: norm_data.as_slice(),
            },
        ],
    };
    let gguf_bytes = write_complete(&gguf_model).expect("writes source gguf");
    let parsed_gguf = parse_complete(&gguf_bytes).expect("parses source gguf");

    let safetensors_model =
        gguf_to_safetensors(&parsed_gguf, &gguf_bytes).expect("gguf -> safetensors");
    assert_eq!(safetensors_model.tensors.len(), 2);
    let embedding = safetensors_model
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .expect("embedding present");
    assert_eq!(embedding.dtype, DType::Float32);
    assert_eq!(embedding.shape, alloc::vec![4, 4]);
    assert_eq!(embedding.data, embedding_data.as_slice());

    // metadata degraded to text: exact string equality with the source
    // value's own Debug output (or verbatim for the String variant).
    for (key, original_value) in &original_metadata {
        let expected_text = match original_value {
            MetadataValue::String(text) => text.clone(),
            other => format!("{other:?}"),
        };
        assert_eq!(
            safetensors_model.metadata.get(key),
            Some(&expected_text),
            "metadata key {key}"
        );
    }

    let safetensors_bytes =
        proxima_safetensors::write_complete(&safetensors_model).expect("writes safetensors");
    let reparsed_manifest =
        parse_safetensors(&safetensors_bytes).expect("parses safetensors back");
    assert_eq!(reparsed_manifest.tensors.len(), 2);

    let roundtrip_gguf = safetensors_to_gguf(&safetensors_model).expect("safetensors -> gguf");
    let roundtrip_bytes = write_complete(&roundtrip_gguf).expect("writes round-tripped gguf");
    let roundtrip_parsed = parse_complete(&roundtrip_bytes).expect("parses round-tripped gguf");

    assert_eq!(roundtrip_parsed.tensors.len(), 2);
    for original in &parsed_gguf.tensors {
        let roundtrip_tensor = roundtrip_parsed
            .tensors
            .iter()
            .find(|candidate| candidate.name == original.name)
            .expect("tensor present after round trip");
        assert_eq!(
            roundtrip_tensor.dims, original.dims,
            "{} dims",
            original.name
        );
        assert_eq!(
            roundtrip_tensor.ggml_type, original.ggml_type,
            "{} ggml_type",
            original.name
        );

        let original_range = parsed_gguf
            .tensor_data_range(original, gguf_bytes.len() as u64)
            .expect("original range");
        let roundtrip_range = roundtrip_parsed
            .tensor_data_range(roundtrip_tensor, roundtrip_bytes.len() as u64)
            .expect("round-trip range");
        assert_eq!(
            &roundtrip_bytes[roundtrip_range.start as usize..roundtrip_range.end as usize],
            &gguf_bytes[original_range.start as usize..original_range.end as usize],
            "{} payload bytes",
            original.name
        );
    }

    // every original metadata value now round-trips as a String, per
    // the documented lossy-type (not lossy-value) contract.
    for (key, original_value) in &original_metadata {
        let expected_text = match original_value {
            MetadataValue::String(text) => text.clone(),
            other => format!("{other:?}"),
        };
        assert_eq!(
            roundtrip_parsed.metadata_value(key),
            Some(&MetadataValue::String(expected_text)),
            "metadata key {key} after full round trip"
        );
    }
}

#[test]
fn quantized_tensor_errors_instead_of_silently_dropping_or_corrupting() {
    let data = pattern_bytes(18); // one Q4_0 block
    let model = GgufModel {
        version: 3,
        metadata: Vec::new(),
        tensors: vec![GgufTensorPayload {
            name: "blk.0.attn_q.weight".to_string(),
            dims: dims(&[32]),
            ggml_type: GgmlType::Q4_0,
            data: data.as_slice(),
        }],
    };
    let bytes = write_complete(&model).expect("writes quantized gguf");
    let parsed = parse_complete(&bytes).expect("parses quantized gguf");

    let outcome = gguf_to_safetensors(&parsed, &bytes);
    assert!(matches!(
        outcome,
        Err(InteropError::UnrepresentableGgmlType { .. })
    ));
}

#[test]
fn bool_dtype_tensor_errors_instead_of_silently_dropping_or_corrupting() {
    let data = [1u8];
    let model = SafetensorsModel {
        tensors: vec![SafetensorsTensorPayload {
            name: "mask".to_string(),
            dtype: DType::Bool,
            shape: alloc::vec![1],
            data: &data,
        }],
        metadata: BTreeMap::new(),
    };
    let outcome = safetensors_to_gguf(&model);
    assert!(matches!(
        outcome,
        Err(InteropError::UnrepresentableDType { .. })
    ));
}
