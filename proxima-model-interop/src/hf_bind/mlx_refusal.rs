//! Refusal of mlx-lm quantized safetensors directories.
//!
//! mlx-lm stores each quantized matrix as three tensors: `<name>.weight`
//! (dtype U32, bit-packed), `<name>.scales` and `<name>.biases`, and records
//! `quantization: {group_size, bits, mode}` in `config.json`. Native affine
//! import is a separate spec, so a [`Manifest`] with that shape is refused
//! before any tensor binds; the binders in [`crate::hf_bind`] call
//! [`refuse_mlx_quantized_header`] first, and a caller that also holds the
//! `config.json` bytes calls [`refuse_mlx_quantized`].

use alloc::format;
use alloc::string::String;

use proxima_safetensors::{Manifest, TensorEntry};
use proxima_tensor::DType;

use crate::error::InteropError;

const WEIGHT_SUFFIX: &str = ".weight";

fn packed_weight(entry: &TensorEntry) -> Option<&str> {
    match entry.dtype {
        DType::UINT32 => entry.name.strip_suffix(WEIGHT_SUFFIX),
        _ => None,
    }
}

fn first_mlx_triple(manifest: &Manifest) -> Option<&str> {
    manifest.tensors.iter().find_map(|entry| {
        let base = packed_weight(entry)?;
        let has_scales = manifest.tensor(&format!("{base}.scales")).is_some();
        let has_biases = manifest.tensor(&format!("{base}.biases")).is_some();
        (has_scales && has_biases).then_some(entry.name.as_str())
    })
}

fn config_declares_quantization(config_json: &[u8]) -> bool {
    serde_json::from_slice::<serde_json::Value>(config_json)
        .ok()
        .and_then(|config| config.get("quantization").map(serde_json::Value::is_object))
        .unwrap_or(false)
}

fn first_packed_or_any(manifest: &Manifest) -> String {
    manifest
        .tensors
        .iter()
        .find(|entry| packed_weight(entry).is_some())
        .or_else(|| manifest.tensors.first())
        .map_or_else(|| String::from("quantization"), |entry| entry.name.clone())
}

/// Refuses a [`Manifest`] whose header has an mlx-lm `.weight` (U32) /
/// `.scales` / `.biases` triple.
///
/// # Errors
///
/// [`InteropError::MlxQuantizedNotLoadable`] naming the first packed weight.
pub fn refuse_mlx_quantized_header(manifest: &Manifest) -> Result<(), InteropError> {
    match first_mlx_triple(manifest) {
        Some(tensor) => Err(InteropError::MlxQuantizedNotLoadable {
            tensor: String::from(tensor),
        }),
        None => Ok(()),
    }
}

/// [`refuse_mlx_quantized_header`], plus refusal when `config_json` (the
/// directory's `config.json` bytes) carries a `quantization` object.
///
/// # Errors
///
/// [`InteropError::MlxQuantizedNotLoadable`] naming the first packed U32
/// weight, or the first tensor when the header holds none.
pub fn refuse_mlx_quantized(
    manifest: &Manifest,
    config_json: Option<&[u8]>,
) -> Result<(), InteropError> {
    refuse_mlx_quantized_header(manifest)?;
    match config_json {
        Some(bytes) if config_declares_quantization(bytes) => {
            Err(InteropError::MlxQuantizedNotLoadable {
                tensor: first_packed_or_any(manifest),
            })
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::fs;

    use proxima_primitives::pipe::sans_io::Outcome;
    use proxima_safetensors::{SafetensorsModel, SafetensorsParser, TensorPayload, write_complete};
    use std::collections::BTreeMap;

    use super::*;

    const OUT_FEATURES: u64 = 128;
    const IN_FEATURES: u64 = 256;
    const GROUP_SIZE: u64 = 64;
    const BITS: u64 = 4;
    const CONFIG_JSON: &str = r#"{"model_type":"llama","hidden_size":256,"quantization":{"group_size":64,"bits":4,"mode":"affine"}}"#;

    fn byte_len(dtype_bytes: u64, shape: &[u64]) -> usize {
        usize::try_from(dtype_bytes * shape.iter().product::<u64>()).expect("fixture fits usize")
    }

    fn write_mlx_directory(directory: &std::path::Path) {
        let packed_shape = [OUT_FEATURES, IN_FEATURES * BITS / 32];
        let group_shape = [OUT_FEATURES, IN_FEATURES / GROUP_SIZE];
        let packed = std::vec![0u8; byte_len(4, &packed_shape)];
        let groups = std::vec![0u8; byte_len(2, &group_shape)];
        let name = "model.layers.0.self_attn.q_proj";
        let tensors = std::vec![
            TensorPayload {
                name: format!("{name}.weight"),
                dtype: DType::UINT32,
                shape: packed_shape.to_vec(),
                data: &packed
            },
            TensorPayload {
                name: format!("{name}.scales"),
                dtype: DType::BFLOAT16,
                shape: group_shape.to_vec(),
                data: &groups
            },
            TensorPayload {
                name: format!("{name}.biases"),
                dtype: DType::BFLOAT16,
                shape: group_shape.to_vec(),
                data: &groups
            },
        ];
        let bytes = write_complete(&SafetensorsModel {
            tensors,
            metadata: BTreeMap::new(),
        })
        .expect("fixture serializes");
        fs::write(directory.join("model.safetensors"), bytes).expect("write safetensors");
        fs::write(directory.join("config.json"), CONFIG_JSON).expect("write config");
    }

    fn read_manifest(directory: &std::path::Path) -> Manifest {
        let bytes = fs::read(directory.join("model.safetensors")).expect("read safetensors");
        let mut parser = SafetensorsParser::new();
        parser.feed(&bytes);
        match parser.poll().expect("header parses") {
            Outcome::Event(manifest) => manifest.clone(),
            Outcome::NeedMore => panic!("complete file must yield a manifest"),
        }
    }

    #[test]
    fn mlx_quantized_load_refused_naming_first_packed_tensor() {
        let directory = tempfile::tempdir().expect("tempdir");
        write_mlx_directory(directory.path());
        let manifest = read_manifest(directory.path());
        let config = fs::read(directory.path().join("config.json")).expect("read config");

        let header_outcome = refuse_mlx_quantized_header(&manifest);
        let config_outcome = refuse_mlx_quantized(&Manifest::default(), Some(&config));

        let expected = "model.layers.0.self_attn.q_proj.weight";
        assert!(
            matches!(&header_outcome, Err(InteropError::MlxQuantizedNotLoadable { tensor }) if tensor == expected),
            "header triple must be refused naming {expected}, got {header_outcome:?}"
        );
        assert!(
            matches!(
                &config_outcome,
                Err(InteropError::MlxQuantizedNotLoadable { .. })
            ),
            "config quantization object must be refused, got {config_outcome:?}"
        );
    }
}
