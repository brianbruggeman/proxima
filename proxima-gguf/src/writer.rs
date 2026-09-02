//! The write half of this crate: the same sans-IO contract as
//! [`crate::parser`], mirrored. [`write_complete`] never opens a file or
//! writes to disk — it takes an in-memory [`GgufModel`] and returns owned
//! bytes; the caller (an `std::fs::File`, a socket, anything) owns getting
//! those bytes wherever they're going.
//!
//! Byte layout mirrors [`crate::parser::GgufParser`] exactly: magic,
//! version, tensor/kv counts, the KV block, the tensor directory, then the
//! alignment-padded data section. Between tensors in the data section, each
//! tensor's byte run is padded up to the resolved alignment before the next
//! one starts — the same accounting [`crate::parser::GgufParser`] performs
//! via `tensor_size_total` when it validates each tensor's declared offset.

use alloc::string::String;
use alloc::vec::Vec;

use arrayvec::ArrayVec;

use crate::error::GgufError;
use crate::parser::{MAGIC, pad_to_alignment};
use crate::sized::{DEFAULT_ALIGNMENT, MAX_SUPPORTED_VERSION};
use crate::tensor::{MAX_DIMS, MAX_NAME_LEN, TensorInfo};
use crate::types::GgmlType;
use crate::value::{MetadataArray, MetadataValue};

/// One tensor to be written: directory fields plus its raw byte payload,
/// borrowed from whatever buffer the caller already holds it in (never
/// copied by this crate).
#[derive(Debug, Clone, PartialEq)]
pub struct TensorPayload<'a> {
    pub name: String,
    pub dims: ArrayVec<u64, MAX_DIMS>,
    pub ggml_type: GgmlType,
    pub data: &'a [u8],
}

/// An in-memory GGUF model ready to serialize: KV metadata in the order it
/// should appear, and every tensor with its actual bytes. Alignment is
/// resolved the same way the reader resolves it — from a `general.alignment`
/// entry in `metadata` if present, [`DEFAULT_ALIGNMENT`] otherwise — so a
/// model built from a [`crate::pipe::ParsedGguf`]'s own metadata always
/// writes back out at the alignment it was read at.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufModel<'a> {
    pub version: u32,
    pub metadata: Vec<(String, MetadataValue)>,
    pub tensors: Vec<TensorPayload<'a>>,
}

/// Serializes one complete GGUF byte stream in a single call. Stateless —
/// mirrors [`crate::pipe::parse_complete`]'s free-function shape.
///
/// # Errors
///
/// [`GgufError::UnsupportedVersion`] for a version this crate's own reader
/// couldn't parse back; [`GgufError::DuplicateKey`] /
/// [`GgufError::DuplicateTensorName`] for repeated keys or names;
/// [`GgufError::NameTooLong`] for a tensor name over
/// [`crate::tensor::MAX_NAME_LEN`]; [`GgufError::InvalidAlignment`] /
/// [`GgufError::InvalidAlignmentType`] for a malformed `general.alignment`;
/// [`GgufError::RowSizeNotBlockMultiple`] if a tensor's first dimension
/// isn't a multiple of its `ggml_type`'s block size;
/// [`GgufError::TensorDataLengthMismatch`] if a tensor's `data` doesn't
/// match the byte length its dims and `ggml_type` imply.
pub fn write_complete(model: &GgufModel<'_>) -> Result<Vec<u8>, GgufError> {
    if model.version == 0 || model.version == 1 || model.version > MAX_SUPPORTED_VERSION {
        return Err(GgufError::UnsupportedVersion {
            version: model.version,
        });
    }

    let alignment = resolve_alignment(&model.metadata)?;
    check_unique_keys(&model.metadata)?;
    check_tensors(&model.tensors)?;

    let mut buf = Vec::new();
    buf.extend_from_slice(&MAGIC);
    buf.extend_from_slice(&model.version.to_le_bytes());
    buf.extend_from_slice(&(model.tensors.len() as i64).to_le_bytes());
    buf.extend_from_slice(&(model.metadata.len() as i64).to_le_bytes());

    for (key, value) in &model.metadata {
        write_kv(&mut buf, key, value);
    }

    let mut running_offset = 0u64;
    for tensor in &model.tensors {
        write_tensor_entry(&mut buf, tensor, running_offset);
        let nbytes = tensor.data.len() as u64;
        running_offset += pad_to_alignment(nbytes, alignment);
    }

    let data_offset = pad_to_alignment(buf.len() as u64, alignment);
    buf.resize(data_offset as usize, 0);

    for tensor in &model.tensors {
        buf.extend_from_slice(tensor.data);
        let padded = pad_to_alignment(tensor.data.len() as u64, alignment);
        let pad_len = (padded - tensor.data.len() as u64) as usize;
        buf.resize(buf.len() + pad_len, 0);
    }

    Ok(buf)
}

fn resolve_alignment(metadata: &[(String, MetadataValue)]) -> Result<u32, GgufError> {
    let Some((_, value)) = metadata.iter().find(|(key, _)| key == "general.alignment") else {
        return Ok(DEFAULT_ALIGNMENT);
    };
    let alignment = value.as_u32().ok_or(GgufError::InvalidAlignmentType)?;
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(GgufError::InvalidAlignment { value: alignment });
    }
    Ok(alignment)
}

fn check_unique_keys(metadata: &[(String, MetadataValue)]) -> Result<(), GgufError> {
    let mut seen: Vec<&str> = Vec::with_capacity(metadata.len());
    for (key, _) in metadata {
        if seen.contains(&key.as_str()) {
            return Err(GgufError::DuplicateKey { key: key.clone() });
        }
        seen.push(key.as_str());
    }
    Ok(())
}

fn check_tensors(tensors: &[TensorPayload<'_>]) -> Result<(), GgufError> {
    let mut seen: Vec<&str> = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        if tensor.name.len() > MAX_NAME_LEN {
            return Err(GgufError::NameTooLong {
                len: tensor.name.len(),
                max: MAX_NAME_LEN,
            });
        }
        if seen.contains(&tensor.name.as_str()) {
            return Err(GgufError::DuplicateTensorName {
                name: tensor.name.clone(),
            });
        }
        seen.push(tensor.name.as_str());

        let block_size = tensor.ggml_type.block_layout().block_elements;
        let row_len = tensor.dims.first().copied().unwrap_or(1);
        if block_size == 0 || row_len % block_size != 0 {
            return Err(GgufError::RowSizeNotBlockMultiple {
                tensor: tensor.name.clone(),
                ne0: row_len,
                block_size,
            });
        }

        let info = TensorInfo {
            name: tensor.name.clone(),
            dims: tensor.dims.clone(),
            ggml_type: tensor.ggml_type,
            offset: 0,
        };
        let expected = info.nbytes().ok_or(GgufError::Overflow {
            context: "tensor byte size",
        })?;
        if expected != tensor.data.len() as u64 {
            return Err(GgufError::TensorDataLengthMismatch {
                tensor: tensor.name.clone(),
                expected,
                found: tensor.data.len(),
            });
        }
    }
    Ok(())
}

fn write_string(buf: &mut Vec<u8>, value: &str) {
    buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
    buf.extend_from_slice(value.as_bytes());
}

fn write_kv(buf: &mut Vec<u8>, key: &str, value: &MetadataValue) {
    write_string(buf, key);
    buf.extend_from_slice(&(value.metadata_type() as u32).to_le_bytes());
    write_value(buf, value);
}

fn write_value(buf: &mut Vec<u8>, value: &MetadataValue) {
    match value {
        MetadataValue::U8(v) => buf.push(*v),
        MetadataValue::I8(v) => buf.push(*v as u8),
        MetadataValue::U16(v) => buf.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::I16(v) => buf.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::U32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::I32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::F32(v) => buf.extend_from_slice(&v.to_bits().to_le_bytes()),
        MetadataValue::Bool(v) => buf.push(u8::from(*v)),
        MetadataValue::String(v) => write_string(buf, v),
        MetadataValue::U64(v) => buf.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::I64(v) => buf.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::F64(v) => buf.extend_from_slice(&v.to_bits().to_le_bytes()),
        MetadataValue::Array(array) => write_array(buf, array),
    }
}

fn write_array(buf: &mut Vec<u8>, array: &MetadataArray) {
    buf.extend_from_slice(&(array.element_metadata_type() as u32).to_le_bytes());
    buf.extend_from_slice(&(array.len() as u64).to_le_bytes());
    match array {
        MetadataArray::U8(values) => buf.extend_from_slice(values),
        MetadataArray::I8(values) => values.iter().for_each(|v| buf.push(*v as u8)),
        MetadataArray::U16(values) => values
            .iter()
            .for_each(|v| buf.extend_from_slice(&v.to_le_bytes())),
        MetadataArray::I16(values) => values
            .iter()
            .for_each(|v| buf.extend_from_slice(&v.to_le_bytes())),
        MetadataArray::U32(values) => values
            .iter()
            .for_each(|v| buf.extend_from_slice(&v.to_le_bytes())),
        MetadataArray::I32(values) => values
            .iter()
            .for_each(|v| buf.extend_from_slice(&v.to_le_bytes())),
        MetadataArray::F32(values) => values
            .iter()
            .for_each(|v| buf.extend_from_slice(&v.to_bits().to_le_bytes())),
        MetadataArray::Bool(values) => values.iter().for_each(|v| buf.push(u8::from(*v))),
        MetadataArray::String(values) => values.iter().for_each(|v| write_string(buf, v)),
        MetadataArray::U64(values) => values
            .iter()
            .for_each(|v| buf.extend_from_slice(&v.to_le_bytes())),
        MetadataArray::I64(values) => values
            .iter()
            .for_each(|v| buf.extend_from_slice(&v.to_le_bytes())),
        MetadataArray::F64(values) => values
            .iter()
            .for_each(|v| buf.extend_from_slice(&v.to_bits().to_le_bytes())),
    }
}

fn write_tensor_entry(buf: &mut Vec<u8>, tensor: &TensorPayload<'_>, offset: u64) {
    write_string(buf, &tensor.name);
    buf.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
    for dim in &tensor.dims {
        buf.extend_from_slice(&dim.to_le_bytes());
    }
    buf.extend_from_slice(&tensor.ggml_type.to_wire().to_le_bytes());
    buf.extend_from_slice(&offset.to_le_bytes());
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::string::ToString;
    use alloc::vec;

    use super::*;
    use crate::parse_complete;

    fn dims(values: &[u64]) -> ArrayVec<u64, MAX_DIMS> {
        values.iter().copied().collect()
    }

    fn pattern_bytes(len: usize) -> Vec<u8> {
        (0..len).map(|index| (index % 251) as u8).collect()
    }

    /// Reuses `crate::tests::synthetic_gguf()` — the reader's own fixture —
    /// for every metadata entry and tensor directory field (names, dims,
    /// types, alignment), and only synthesizes what that fixture never
    /// carried in the first place: real tensor payload bytes. Tensor 1
    /// (`72` bytes at alignment `16`) is not a multiple of the alignment,
    /// so this exercises the inter-tensor padding path in one pass.
    #[test]
    fn write_complete_round_trips_every_field_and_every_tensor_byte() {
        let reference = parse_complete(&crate::tests::synthetic_gguf()).expect("reference parses");

        let payloads: Vec<Vec<u8>> = reference
            .tensors
            .iter()
            .map(|tensor| pattern_bytes(tensor.nbytes().expect("computable nbytes") as usize))
            .collect();
        assert_eq!(
            payloads[1].len(),
            72,
            "tensor 1 must be a non-multiple-of-16 byte length"
        );

        let model = GgufModel {
            version: reference.version,
            metadata: reference.metadata.clone(),
            tensors: reference
                .tensors
                .iter()
                .zip(&payloads)
                .map(|(tensor, data)| TensorPayload {
                    name: tensor.name.clone(),
                    dims: tensor.dims.clone(),
                    ggml_type: tensor.ggml_type,
                    data: data.as_slice(),
                })
                .collect(),
        };

        let written = write_complete(&model).expect("writes synthetic model");
        let parsed_back = parse_complete(&written).expect("written bytes parse");

        assert_eq!(parsed_back.version, reference.version);
        assert_eq!(parsed_back.tensor_count, reference.tensor_count);
        assert_eq!(parsed_back.kv_count, reference.kv_count);
        assert_eq!(parsed_back.metadata, reference.metadata);
        assert_eq!(parsed_back.alignment, reference.alignment);
        assert_eq!(parsed_back.data_offset, reference.data_offset);
        assert_eq!(parsed_back.tensors.len(), reference.tensors.len());

        for (index, (parsed_tensor, reference_tensor)) in parsed_back
            .tensors
            .iter()
            .zip(&reference.tensors)
            .enumerate()
        {
            assert_eq!(
                parsed_tensor.name, reference_tensor.name,
                "tensor {index} name"
            );
            assert_eq!(
                parsed_tensor.dims, reference_tensor.dims,
                "tensor {index} dims"
            );
            assert_eq!(
                parsed_tensor.ggml_type, reference_tensor.ggml_type,
                "tensor {index} ggml_type"
            );
            assert_eq!(
                parsed_tensor.offset, reference_tensor.offset,
                "tensor {index} byte offset"
            );

            let range = parsed_back
                .tensor_data_range(parsed_tensor, written.len() as u64)
                .expect("range within written buffer");
            let actual = &written[range.start as usize..range.end as usize];
            assert_eq!(
                actual,
                payloads[index].as_slice(),
                "tensor {index} payload bytes"
            );
        }
    }

    /// No `general.alignment` key at all — the writer must fall back to
    /// `DEFAULT_ALIGNMENT` (32) exactly the way the reader does, and a
    /// tensor whose byte length isn't a multiple of 32 must still round
    /// trip, proving the default-alignment padding path specifically.
    #[test]
    fn write_complete_pads_correctly_at_default_alignment() {
        let data = pattern_bytes(12); // 3 * F32 = 12 bytes, not a multiple of 32
        let model = GgufModel {
            version: 3,
            metadata: vec![(
                "general.architecture".to_string(),
                MetadataValue::String("test".to_string()),
            )],
            tensors: vec![TensorPayload {
                name: "weight".to_string(),
                dims: dims(&[3]),
                ggml_type: GgmlType::F32,
                data: data.as_slice(),
            }],
        };

        let written = write_complete(&model).expect("writes default-alignment model");
        let parsed = parse_complete(&written).expect("parses back");

        assert_eq!(parsed.alignment, DEFAULT_ALIGNMENT);
        assert_eq!(parsed.data_offset % u64::from(DEFAULT_ALIGNMENT), 0);
        let tensor = &parsed.tensors[0];
        let range = parsed
            .tensor_data_range(tensor, written.len() as u64)
            .expect("range within written buffer");
        assert_eq!(
            &written[range.start as usize..range.end as usize],
            data.as_slice()
        );
    }

    #[test]
    fn write_complete_rejects_tensor_data_length_mismatch() {
        let model = GgufModel {
            version: 3,
            metadata: Vec::new(),
            tensors: vec![TensorPayload {
                name: "bad".to_string(),
                dims: dims(&[4]),
                ggml_type: GgmlType::F32,
                data: &[0u8; 8], // should be 16 bytes (4 * f32)
            }],
        };
        let outcome = write_complete(&model);
        assert!(matches!(
            outcome,
            Err(GgufError::TensorDataLengthMismatch { .. })
        ));
    }

    #[test]
    fn write_complete_rejects_duplicate_metadata_keys() {
        let model = GgufModel {
            version: 3,
            metadata: vec![
                ("dup".to_string(), MetadataValue::U32(1)),
                ("dup".to_string(), MetadataValue::U32(2)),
            ],
            tensors: Vec::new(),
        };
        let outcome = write_complete(&model);
        assert!(matches!(outcome, Err(GgufError::DuplicateKey { .. })));
    }
}
