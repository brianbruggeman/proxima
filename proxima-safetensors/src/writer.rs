//! The write half of this crate, sans-IO the same way [`crate::parser`] is:
//! [`write_complete`] never opens a file — it takes an in-memory
//! [`SafetensorsModel`] and returns owned bytes; the caller owns getting
//! those bytes wherever they're going.
//!
//! Wire layout mirrors [`crate::parser`] exactly: `[u64 LE header
//! len][header JSON][raw tensor bytes]`. Unlike GGUF, safetensors has no
//! alignment padding between tensors — `data_offsets` are simply the
//! running byte count as each tensor's bytes are appended in order.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use proxima_tensor::DType;

use crate::dtype::dtype_to_wire;
use crate::error::SafetensorsError;

/// One tensor to be written: name, dtype, shape, and its raw byte payload,
/// borrowed from whatever buffer the caller already holds it in.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorPayload<'a> {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub data: &'a [u8],
}

/// An in-memory safetensors model ready to serialize: every tensor with its
/// actual bytes, plus the optional `__metadata__` free-form string map.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SafetensorsModel<'a> {
    pub tensors: Vec<TensorPayload<'a>>,
    pub metadata: BTreeMap<String, String>,
}

/// Serializes one complete safetensors byte stream in a single call.
/// Stateless — mirrors [`crate::pipe::parse_complete`]'s free-function
/// shape.
///
/// # Errors
///
/// [`SafetensorsError::ReservedTensorName`] for a tensor named
/// `__metadata__`; [`SafetensorsError::ReservedMetadataKey`] if
/// `model.metadata` itself sets [`crate::sized::FORMAT_VERSION_KEY`] --
/// that entry is always written by this function, from
/// [`crate::sized::FORMAT_VERSION_MAJOR`]/[`crate::sized::FORMAT_VERSION_MINOR`],
/// never from the caller; [`SafetensorsError::DuplicateTensorName`] for
/// repeated names; [`SafetensorsError::UnsupportedDtype`] if `dtype` has no
/// safetensors wire string (currently `Int128`/`UInt128`);
/// [`SafetensorsError::TensorDataLengthMismatch`] if a tensor's `data`
/// doesn't match the byte length its `shape` and `dtype` imply.
pub fn write_complete(model: &SafetensorsModel<'_>) -> Result<Vec<u8>, SafetensorsError> {
    let mut seen_names: Vec<&str> = Vec::with_capacity(model.tensors.len());
    let mut data = Vec::new();
    let mut header = serde_json::Map::new();

    let mut meta_object = serde_json::Map::new();
    meta_object.insert(
        crate::sized::FORMAT_VERSION_KEY.to_string(),
        serde_json::Value::String(crate::version::current_version_string()),
    );
    for (key, value) in &model.metadata {
        if key == crate::sized::FORMAT_VERSION_KEY {
            return Err(SafetensorsError::ReservedMetadataKey { key: key.clone() });
        }
        meta_object.insert(key.clone(), serde_json::Value::String(value.clone()));
    }
    header.insert("__metadata__".to_string(), serde_json::Value::Object(meta_object));

    for tensor in &model.tensors {
        if tensor.name == "__metadata__" {
            return Err(SafetensorsError::ReservedTensorName {
                name: tensor.name.clone(),
            });
        }
        if seen_names.contains(&tensor.name.as_str()) {
            return Err(SafetensorsError::DuplicateTensorName {
                name: tensor.name.clone(),
            });
        }
        seen_names.push(tensor.name.as_str());

        let dtype_wire = dtype_to_wire(tensor.dtype).ok_or_else(|| SafetensorsError::UnsupportedDtype {
            tensor: tensor.name.clone(),
            dtype: alloc::format!("{:?}", tensor.dtype),
        })?;

        let element_count: u64 = tensor.shape.iter().product();
        let expected = element_count * tensor.dtype.size_bytes() as u64;
        if expected != tensor.data.len() as u64 {
            return Err(SafetensorsError::TensorDataLengthMismatch {
                tensor: tensor.name.clone(),
                expected,
                found: tensor.data.len() as u64,
            });
        }

        let start = data.len() as u64;
        data.extend_from_slice(tensor.data);
        let end = data.len() as u64;

        let mut entry = serde_json::Map::new();
        entry.insert("dtype".to_string(), serde_json::Value::String(dtype_wire.to_string()));
        entry.insert(
            "shape".to_string(),
            serde_json::Value::Array(tensor.shape.iter().map(|dim| (*dim).into()).collect()),
        );
        entry.insert(
            "data_offsets".to_string(),
            serde_json::Value::Array(alloc::vec![start.into(), end.into()]),
        );
        header.insert(tensor.name.clone(), serde_json::Value::Object(entry));
    }

    let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).map_err(|error| {
        SafetensorsError::MalformedJson {
            reason: error.to_string(),
        }
    })?;

    let mut wire = Vec::with_capacity(8 + header_bytes.len() + data.len());
    wire.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    wire.extend_from_slice(&header_bytes);
    wire.extend_from_slice(&data);
    Ok(wire)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::dtype::map_dtype;
    use crate::parser::SafetensorsParser;
    use crate::tests::build_buffer;

    fn pattern_bytes(len: usize) -> Vec<u8> {
        (0..len).map(|index| (index % 251) as u8).collect()
    }

    fn parse_whole(buf: &[u8]) -> Result<crate::Manifest, SafetensorsError> {
        SafetensorsParser::new().push(buf)?.finish()
    }

    /// Drives `crate::tests::build_buffer` — the reader's own fixture
    /// builder — to get a ground-truth manifest and byte pattern
    /// independent of this writer, then builds a [`SafetensorsModel`] out
    /// of the SAME `(name, dtype, shape, byte_len)` table and asserts the
    /// writer's output parses back to that same reference manifest, with
    /// every tensor's bytes exact.
    #[test]
    fn write_complete_round_trips_every_field_and_every_tensor_byte() {
        let entries: &[(&str, &str, &[u64], usize)] = &[
            ("scalar", "F32", &[], 4),
            ("vector", "I8", &[8], 8),
            ("matrix", "BF16", &[2, 3], 12),
            ("cube", "BOOL", &[2, 2, 2], 8),
        ];
        let metadata_entries: &[(&str, &str)] = &[("format", "pt"), ("author", "test")];
        let version_stamp = crate::version::current_version_string();
        let reference_metadata_entries: Vec<(&str, &str)> = metadata_entries
            .iter()
            .copied()
            .chain(core::iter::once((crate::sized::FORMAT_VERSION_KEY, version_stamp.as_str())))
            .collect();
        let (reference_bytes, reference_offsets) = build_buffer(entries, &reference_metadata_entries);
        let reference_manifest = parse_whole(&reference_bytes).expect("reference buffer parses");

        let owned_payloads: Vec<Vec<u8>> = entries
            .iter()
            .map(|(_, _, _, byte_len)| pattern_bytes(*byte_len))
            .collect();
        let tensors: Vec<TensorPayload<'_>> = entries
            .iter()
            .zip(&owned_payloads)
            .map(|((name, dtype, shape, _), data)| TensorPayload {
                name: (*name).to_string(),
                dtype: map_dtype(name, dtype).expect("known dtype"),
                shape: shape.to_vec(),
                data: data.as_slice(),
            })
            .collect();
        let mut metadata = BTreeMap::new();
        for (key, value) in metadata_entries {
            metadata.insert((*key).to_string(), (*value).to_string());
        }

        let model = SafetensorsModel { tensors, metadata };
        let written = write_complete(&model).expect("writes synthetic model");
        let manifest = parse_whole(&written).expect("written buffer parses back");

        assert_eq!(manifest, reference_manifest);

        let header_len = u64::from_le_bytes(written[..8].try_into().expect("8 bytes")) as usize;
        let data_region = &written[8 + header_len..];
        for (name, (expected_start, expected_end)) in &reference_offsets {
            let entry = manifest.tensor(name).expect("tensor present");
            assert_eq!(entry.data_offsets, (*expected_start, *expected_end), "tensor {name} offsets");
            let actual = &data_region[*expected_start as usize..*expected_end as usize];
            let payload = &owned_payloads[entries.iter().position(|(candidate, ..)| candidate == name).expect("entry present")];
            assert_eq!(actual, payload.as_slice(), "tensor {name} payload bytes");
        }
    }

    #[test]
    fn write_complete_rejects_tensor_data_length_mismatch() {
        let data = [0u8; 4];
        let model = SafetensorsModel {
            tensors: vec![TensorPayload {
                name: "bad".to_string(),
                dtype: DType::Float32,
                shape: vec![4], // implies 16 bytes, data is only 4
                data: &data,
            }],
            metadata: BTreeMap::new(),
        };
        let outcome = write_complete(&model);
        assert!(matches!(
            outcome,
            Err(SafetensorsError::TensorDataLengthMismatch { .. })
        ));
    }

    #[test]
    fn write_complete_rejects_duplicate_tensor_names() {
        let data = [0u8; 4];
        let model = SafetensorsModel {
            tensors: vec![
                TensorPayload {
                    name: "dup".to_string(),
                    dtype: DType::Float32,
                    shape: vec![1],
                    data: &data,
                },
                TensorPayload {
                    name: "dup".to_string(),
                    dtype: DType::Float32,
                    shape: vec![1],
                    data: &data,
                },
            ],
            metadata: BTreeMap::new(),
        };
        let outcome = write_complete(&model);
        assert!(matches!(
            outcome,
            Err(SafetensorsError::DuplicateTensorName { .. })
        ));
    }

    #[test]
    fn write_complete_rejects_reserved_metadata_tensor_name() {
        let data = [0u8; 4];
        let model = SafetensorsModel {
            tensors: vec![TensorPayload {
                name: "__metadata__".to_string(),
                dtype: DType::Float32,
                shape: vec![1],
                data: &data,
            }],
            metadata: BTreeMap::new(),
        };
        let outcome = write_complete(&model);
        assert!(matches!(
            outcome,
            Err(SafetensorsError::ReservedTensorName { .. })
        ));
    }

    #[test]
    fn write_complete_stamps_a_format_version_a_caller_wrote_none() {
        let data = [0u8; 4];
        let model = SafetensorsModel {
            tensors: vec![TensorPayload {
                name: "t".to_string(),
                dtype: DType::Float32,
                shape: vec![1],
                data: &data,
            }],
            metadata: BTreeMap::new(),
        };
        let written = write_complete(&model).expect("writes a model with no caller metadata");
        let manifest = parse_whole(&written).expect("written buffer parses back");

        assert_eq!(
            manifest.metadata.get(crate::sized::FORMAT_VERSION_KEY),
            Some(&crate::version::current_version_string()),
            "write_complete must always stamp the current format version"
        );
        assert_eq!(
            manifest.format_version().expect("stamped version is valid"),
            (crate::sized::FORMAT_VERSION_MAJOR, crate::sized::FORMAT_VERSION_MINOR)
        );
    }

    #[test]
    fn write_complete_rejects_a_caller_supplied_format_version_key() {
        let data = [0u8; 4];
        let mut metadata = BTreeMap::new();
        metadata.insert(crate::sized::FORMAT_VERSION_KEY.to_string(), "9.9".to_string());
        let model = SafetensorsModel {
            tensors: vec![TensorPayload {
                name: "t".to_string(),
                dtype: DType::Float32,
                shape: vec![1],
                data: &data,
            }],
            metadata,
        };
        let outcome = write_complete(&model);
        assert!(matches!(
            outcome,
            Err(SafetensorsError::ReservedMetadataKey { ref key }) if key == crate::sized::FORMAT_VERSION_KEY
        ));
    }

    /// The exact fixture shape every safetensors file this workspace wrote
    /// before this stamp existed has: no `__metadata__` entry under
    /// [`crate::sized::FORMAT_VERSION_KEY`] at all -- built directly via
    /// `crate::tests::build_buffer`, bypassing `write_complete`'s always-on
    /// stamp, since that is the only way to reproduce a genuinely
    /// pre-change file byte-for-byte.
    #[test]
    fn a_pre_change_file_with_no_stamp_at_all_still_loads_as_v1() {
        let entries: &[(&str, &str, &[u64], usize)] = &[("w", "F32", &[2], 8)];
        let (pre_change_bytes, _offsets) = build_buffer(entries, &[("format", "pt")]);

        let manifest = parse_whole(&pre_change_bytes).expect("pre-change file still parses");
        assert!(
            !manifest.metadata.contains_key(crate::sized::FORMAT_VERSION_KEY),
            "fixture must genuinely carry no version stamp"
        );
        assert_eq!(manifest.format_version().expect("absent stamp is accepted"), (1, 0));
    }

    #[test]
    fn an_unknown_major_version_is_a_typed_error_naming_file_and_supported_versions() {
        let unknown_major = crate::sized::FORMAT_VERSION_MAJOR + 1;
        let entries: &[(&str, &str, &[u64], usize)] = &[("w", "F32", &[2], 8)];
        let stamp = alloc::format!("{unknown_major}.0");
        let (bytes, _offsets) = build_buffer(entries, &[(crate::sized::FORMAT_VERSION_KEY, stamp.as_str())]);

        let manifest = parse_whole(&bytes).expect("header itself is well-formed");
        let outcome = manifest.format_version();
        assert!(matches!(
            outcome,
            Err(SafetensorsError::UnsupportedFormatVersion { supported_major, .. })
                if supported_major == crate::sized::FORMAT_VERSION_MAJOR
        ));
        let message = outcome.unwrap_err().to_string();
        assert!(message.contains(&stamp), "error must name the file's own version: {message}");
        assert!(
            message.contains(&crate::sized::FORMAT_VERSION_MAJOR.to_string()),
            "error must name the supported range: {message}"
        );
    }
}
