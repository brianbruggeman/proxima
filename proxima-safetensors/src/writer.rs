//! The write half of this crate, sans-IO the same way [`crate::parser`] is:
//! [`write_complete`] never opens a file — it takes an in-memory
//! [`SafetensorsModel`] and returns owned bytes; the caller owns getting
//! those bytes wherever they're going.
//!
//! Wire layout mirrors [`crate::parser`] exactly: `[u64 LE header
//! len][header JSON][raw tensor bytes]`. Unlike GGUF, safetensors has no
//! alignment padding between tensors — `data_offsets` are simply the
//! running byte count as each tensor's bytes are appended in order.
//! [`write_complete_aligned`] gets alignment inside that rule: the format
//! allows no holes, so a gap is a filler tensor (see [`FILLER_NAME_PREFIX`]).

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use proxima_numeric::DType;

use crate::dtype::dtype_to_wire;
use crate::error::SafetensorsError;

/// The header is padded with spaces so the tensor data starts on an 8-byte
/// boundary, as the reference `safetensors` writer does: a mapped file then
/// holds every 2-, 4- and 8-byte element at an address its own width divides.
const DATA_ALIGNMENT_BYTES: usize = 8;

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
    let headers: Vec<TensorHeader> = model
        .tensors
        .iter()
        .map(|tensor| TensorHeader {
            name: tensor.name.clone(),
            dtype: tensor.dtype,
            shape: tensor.shape.clone(),
        })
        .collect();
    let mut wire = encode_header(&headers, &model.metadata)?;
    wire.reserve(model.tensors.iter().map(|tensor| tensor.data.len()).sum());
    for (tensor, header) in model.tensors.iter().zip(&headers) {
        let expected = header.byte_len();
        if expected != tensor.data.len() as u64 {
            return Err(SafetensorsError::TensorDataLengthMismatch {
                tensor: tensor.name.clone(),
                expected,
                found: tensor.data.len() as u64,
            });
        }
        wire.extend_from_slice(tensor.data);
    }
    Ok(wire)
}

/// One tensor's name, dtype and shape: everything a safetensors header
/// records about it before any of its bytes exist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorHeader {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
}

impl TensorHeader {
    /// The byte length of the payload `dtype` and `shape` imply.
    #[must_use]
    pub fn byte_len(&self) -> u64 {
        self.shape.iter().product::<u64>() * self.dtype.size_bytes() as u64
    }
}

/// The bytes that precede the payload of a safetensors stream holding
/// `tensors` in order: the little-endian header length and the header JSON.
/// [`write_complete`] is this prefix followed by each tensor's bytes, so a
/// caller that streams the payload writes this first and then the same bytes.
///
/// # Errors
///
/// As [`write_complete`], except [`SafetensorsError::TensorDataLengthMismatch`],
/// which needs the payload.
pub fn encode_header(
    tensors: &[TensorHeader],
    metadata: &BTreeMap<String, String>,
) -> Result<Vec<u8>, SafetensorsError> {
    let slots = plan_slots(tensors, 1);
    encode_header_slots(tensors, metadata, &slots, DATA_ALIGNMENT_BYTES as u64)
}

/// The prefix of every filler tensor the aligned layout inserts. A filler is
/// named `{FILLER_NAME_PREFIX}{ordinal}` (ordinal counts fillers from zero in
/// file order), is a `U8` vector, and holds zeros. A reader skips a tensor for
/// which [`is_filler_name`] is true; the writer refuses a caller tensor that
/// uses the prefix.
pub const FILLER_NAME_PREFIX: &str = "__pad.";

/// Whether `name` is a filler tensor the aligned layout inserted.
#[must_use]
pub fn is_filler_name(name: &str) -> bool {
    name.starts_with(FILLER_NAME_PREFIX)
}

/// One position in the data region: a caller tensor (by index) or a run of
/// filler bytes. The layout is a function of tensor lengths and `alignment`
/// only, because `data_start` is itself a multiple of `alignment`: an absolute
/// offset is a multiple of `alignment` exactly when the data offset is.
#[derive(Debug, Clone, Copy)]
enum Slot {
    Tensor(usize),
    Filler(u64),
}

/// Orders `tensors` into slots, inserting a filler before every tensor of at
/// least `alignment` bytes that would not otherwise start on `alignment`.
fn plan_slots(tensors: &[TensorHeader], alignment: u64) -> Vec<Slot> {
    let mut slots = Vec::with_capacity(tensors.len());
    let mut end = 0u64;
    for (index, tensor) in tensors.iter().enumerate() {
        let len = tensor.byte_len();
        if alignment > 1 && len >= alignment {
            let gap = (alignment - end % alignment) % alignment;
            if gap > 0 {
                slots.push(Slot::Filler(gap));
                end += gap;
            }
        }
        slots.push(Slot::Tensor(index));
        end += len;
    }
    slots
}

/// Serializes `model` like [`write_complete`], laid out so every tensor of at
/// least `alignment` bytes starts at an absolute file offset (`8 +` header
/// length `+` its data offset) that is a multiple of `alignment`, and the data
/// region itself starts on `alignment`. Gaps between tensors are filler
/// tensors (see [`FILLER_NAME_PREFIX`]); the header is padded with spaces.
/// A mapping whose base is aligned to `alignment` then holds every large
/// tensor at an aligned address.
///
/// # Errors
///
/// As [`write_complete`], plus [`SafetensorsError::ZeroAlignment`] and
/// [`SafetensorsError::ReservedFillerName`].
pub fn write_complete_aligned(
    model: &SafetensorsModel<'_>,
    alignment: usize,
) -> Result<Vec<u8>, SafetensorsError> {
    if alignment == 0 {
        return Err(SafetensorsError::ZeroAlignment);
    }
    let headers: Vec<TensorHeader> = model
        .tensors
        .iter()
        .map(|tensor| TensorHeader {
            name: tensor.name.clone(),
            dtype: tensor.dtype,
            shape: tensor.shape.clone(),
        })
        .collect();
    let slots = plan_slots(&headers, alignment as u64);
    let data_alignment = (alignment as u64).max(DATA_ALIGNMENT_BYTES as u64);
    let mut wire = encode_header_slots(&headers, &model.metadata, &slots, data_alignment)?;
    wire.reserve(model.tensors.iter().map(|tensor| tensor.data.len()).sum());
    for slot in &slots {
        match *slot {
            Slot::Filler(gap) => wire.resize(wire.len() + gap as usize, 0),
            Slot::Tensor(index) => {
                let (tensor, header) = (&model.tensors[index], &headers[index]);
                let expected = header.byte_len();
                if expected != tensor.data.len() as u64 {
                    return Err(SafetensorsError::TensorDataLengthMismatch {
                        tensor: tensor.name.clone(),
                        expected,
                        found: tensor.data.len() as u64,
                    });
                }
                wire.extend_from_slice(tensor.data);
            }
        }
    }
    Ok(wire)
}

fn encode_header_slots(
    tensors: &[TensorHeader],
    metadata: &BTreeMap<String, String>,
    slots: &[Slot],
    data_alignment: u64,
) -> Result<Vec<u8>, SafetensorsError> {
    let mut seen_names: Vec<&str> = Vec::with_capacity(tensors.len());
    let mut header = serde_json::Map::new();

    let mut meta_object = serde_json::Map::new();
    meta_object.insert(
        crate::sized::FORMAT_VERSION_KEY.to_string(),
        serde_json::Value::String(crate::version::current_version_string()),
    );
    for (key, value) in metadata {
        if key == crate::sized::FORMAT_VERSION_KEY {
            return Err(SafetensorsError::ReservedMetadataKey { key: key.clone() });
        }
        meta_object.insert(key.clone(), serde_json::Value::String(value.clone()));
    }
    header.insert(
        "__metadata__".to_string(),
        serde_json::Value::Object(meta_object),
    );

    let mut end = 0u64;
    let mut fillers = 0usize;
    for slot in slots {
        let (name, dtype_wire, shape, len) = match *slot {
            Slot::Filler(gap) => {
                let name = alloc::format!("{FILLER_NAME_PREFIX}{fillers}");
                fillers += 1;
                (name, "U8", alloc::vec![gap], gap)
            }
            Slot::Tensor(index) => {
                let tensor = &tensors[index];
                if tensor.name == "__metadata__" {
                    return Err(SafetensorsError::ReservedTensorName {
                        name: tensor.name.clone(),
                    });
                }
                if is_filler_name(&tensor.name) {
                    return Err(SafetensorsError::ReservedFillerName {
                        name: tensor.name.clone(),
                    });
                }
                if seen_names.contains(&tensor.name.as_str()) {
                    return Err(SafetensorsError::DuplicateTensorName {
                        name: tensor.name.clone(),
                    });
                }
                seen_names.push(tensor.name.as_str());
                let dtype_wire = dtype_to_wire(tensor.dtype).ok_or_else(|| {
                    SafetensorsError::UnsupportedDtype {
                        tensor: tensor.name.clone(),
                        dtype: alloc::format!("{:?}", tensor.dtype),
                    }
                })?;
                (
                    tensor.name.clone(),
                    dtype_wire,
                    tensor.shape.clone(),
                    tensor.byte_len(),
                )
            }
        };
        let start = end;
        end += len;

        let mut entry = serde_json::Map::new();
        entry.insert(
            "dtype".to_string(),
            serde_json::Value::String(dtype_wire.to_string()),
        );
        entry.insert(
            "shape".to_string(),
            serde_json::Value::Array(shape.iter().map(|dim| (*dim).into()).collect()),
        );
        entry.insert(
            "data_offsets".to_string(),
            serde_json::Value::Array(alloc::vec![start.into(), end.into()]),
        );
        header.insert(name, serde_json::Value::Object(entry));
    }

    let mut header_bytes =
        serde_json::to_vec(&serde_json::Value::Object(header)).map_err(|error| {
            SafetensorsError::MalformedJson {
                reason: error.to_string(),
            }
        })?;
    let padded = (8 + header_bytes.len()).next_multiple_of(data_alignment as usize) - 8;
    header_bytes.resize(padded, b' ');

    let mut prefix = Vec::with_capacity(8 + header_bytes.len());
    prefix.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    prefix.extend_from_slice(&header_bytes);
    Ok(prefix)
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
        SafetensorsParser::new().push(buf)?.into_manifest()
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
            .chain(core::iter::once((
                crate::sized::FORMAT_VERSION_KEY,
                version_stamp.as_str(),
            )))
            .collect();
        let (reference_bytes, reference_offsets) =
            build_buffer(entries, &reference_metadata_entries);
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
            assert_eq!(
                entry.data_offsets,
                (*expected_start, *expected_end),
                "tensor {name} offsets"
            );
            let actual = &data_region[*expected_start as usize..*expected_end as usize];
            let payload = &owned_payloads[entries
                .iter()
                .position(|(candidate, ..)| candidate == name)
                .expect("entry present")];
            assert_eq!(actual, payload.as_slice(), "tensor {name} payload bytes");
        }
    }

    #[test]
    fn write_complete_starts_tensor_data_on_an_eight_byte_boundary_for_every_header_length() {
        let data = [0u8; 4];

        for name_length in 1..=16 {
            let model = SafetensorsModel {
                tensors: vec![TensorPayload {
                    name: "t".repeat(name_length),
                    dtype: DType::BFLOAT16,
                    shape: vec![2],
                    data: &data,
                }],
                metadata: BTreeMap::new(),
            };

            let written = write_complete(&model).expect("writes");

            let header_len = u64::from_le_bytes(written[..8].try_into().expect("8 bytes")) as usize;
            assert_eq!(
                (8 + header_len) % DATA_ALIGNMENT_BYTES,
                0,
                "tensor name length {name_length}"
            );
            let manifest = parse_whole(&written).expect("a padded header still parses");
            assert!(manifest.tensor(&"t".repeat(name_length)).is_some());
            assert_eq!(&written[8 + header_len..], data.as_slice());
        }
    }

    fn byte_tensor<'a>(name: &str, data: &'a [u8]) -> TensorPayload<'a> {
        TensorPayload {
            name: name.to_string(),
            dtype: map_dtype(name, "U8").expect("U8 maps"),
            shape: vec![data.len() as u64],
            data,
        }
    }

    #[test]
    fn aligned_write_starts_every_large_tensor_on_the_alignment_and_reads_back_identical() {
        const ALIGNMENT: usize = 64;
        let payloads: Vec<(&str, Vec<u8>)> = vec![
            ("small_head", pattern_bytes(3)),
            ("large_a", pattern_bytes(100)),
            ("small_mid", pattern_bytes(5)),
            ("exact_alignment", pattern_bytes(ALIGNMENT)),
            ("large_b", pattern_bytes(200)),
            ("small_tail", pattern_bytes(7)),
        ];
        let model = SafetensorsModel {
            tensors: payloads
                .iter()
                .map(|(name, data)| byte_tensor(name, data))
                .collect(),
            metadata: BTreeMap::new(),
        };

        let written = write_complete_aligned(&model, ALIGNMENT).expect("writes aligned");

        let header_len = u64::from_le_bytes(written[..8].try_into().expect("8 bytes")) as usize;
        let data_start = 8 + header_len;
        assert_eq!(
            data_start % ALIGNMENT,
            0,
            "data region starts on the alignment"
        );
        let manifest = parse_whole(&written).expect("aligned file parses");
        let mut seen = 0;
        for entry in &manifest.tensors {
            let (start, end) = (entry.data_offsets.0 as usize, entry.data_offsets.1 as usize);
            let bytes = &written[data_start + start..data_start + end];
            if is_filler_name(&entry.name) {
                assert!(
                    bytes.iter().all(|byte| *byte == 0),
                    "filler {} is zeros",
                    entry.name
                );
                continue;
            }
            let expected = &payloads
                .iter()
                .find(|(name, _)| *name == entry.name)
                .expect("a written tensor names a payload")
                .1;
            assert_eq!(bytes, expected.as_slice(), "tensor {} bytes", entry.name);
            if expected.len() >= ALIGNMENT {
                assert_eq!(
                    (data_start + start) % ALIGNMENT,
                    0,
                    "tensor {} absolute offset",
                    entry.name
                );
            }
            seen += 1;
        }
        assert_eq!(
            seen,
            payloads.len(),
            "every original tensor is present once"
        );
        assert!(
            manifest
                .tensors
                .iter()
                .any(|entry| is_filler_name(&entry.name)),
            "this layout needs at least one filler"
        );
    }

    #[test]
    fn aligned_write_of_one_byte_alignment_matches_write_complete() {
        let payloads = [pattern_bytes(3), pattern_bytes(100)];
        let model = SafetensorsModel {
            tensors: vec![
                byte_tensor("first", &payloads[0]),
                byte_tensor("second", &payloads[1]),
            ],
            metadata: BTreeMap::new(),
        };

        assert_eq!(
            write_complete_aligned(&model, 1).expect("aligned"),
            write_complete(&model).expect("plain")
        );
    }

    #[test]
    fn aligned_write_refuses_zero_alignment_and_a_caller_filler_name() {
        let data = pattern_bytes(4);
        let plain = SafetensorsModel {
            tensors: vec![byte_tensor("w", &data)],
            metadata: BTreeMap::new(),
        };
        let reserved = SafetensorsModel {
            tensors: vec![byte_tensor(&alloc::format!("{FILLER_NAME_PREFIX}0"), &data)],
            metadata: BTreeMap::new(),
        };

        assert_eq!(
            write_complete_aligned(&plain, 0),
            Err(SafetensorsError::ZeroAlignment)
        );
        assert!(matches!(
            write_complete_aligned(&reserved, 16),
            Err(SafetensorsError::ReservedFillerName { .. })
        ));
    }

    #[test]
    fn write_complete_rejects_tensor_data_length_mismatch() {
        let data = [0u8; 4];
        let model = SafetensorsModel {
            tensors: vec![TensorPayload {
                name: "bad".to_string(),
                dtype: DType::FLOAT32,
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
                    dtype: DType::FLOAT32,
                    shape: vec![1],
                    data: &data,
                },
                TensorPayload {
                    name: "dup".to_string(),
                    dtype: DType::FLOAT32,
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
                dtype: DType::FLOAT32,
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
                dtype: DType::FLOAT32,
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
            (
                crate::sized::FORMAT_VERSION_MAJOR,
                crate::sized::FORMAT_VERSION_MINOR
            )
        );
    }

    #[test]
    fn write_complete_rejects_a_caller_supplied_format_version_key() {
        let data = [0u8; 4];
        let mut metadata = BTreeMap::new();
        metadata.insert(
            crate::sized::FORMAT_VERSION_KEY.to_string(),
            "9.9".to_string(),
        );
        let model = SafetensorsModel {
            tensors: vec![TensorPayload {
                name: "t".to_string(),
                dtype: DType::FLOAT32,
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
            !manifest
                .metadata
                .contains_key(crate::sized::FORMAT_VERSION_KEY),
            "fixture must genuinely carry no version stamp"
        );
        assert_eq!(
            manifest.format_version().expect("absent stamp is accepted"),
            (1, 0)
        );
    }

    #[test]
    fn an_unknown_major_version_is_a_typed_error_naming_file_and_supported_versions() {
        let unknown_major = crate::sized::FORMAT_VERSION_MAJOR + 1;
        let entries: &[(&str, &str, &[u64], usize)] = &[("w", "F32", &[2], 8)];
        let stamp = alloc::format!("{unknown_major}.0");
        let (bytes, _offsets) = build_buffer(
            entries,
            &[(crate::sized::FORMAT_VERSION_KEY, stamp.as_str())],
        );

        let manifest = parse_whole(&bytes).expect("header itself is well-formed");
        let outcome = manifest.format_version();
        assert!(matches!(
            outcome,
            Err(SafetensorsError::UnsupportedFormatVersion { supported_major, .. })
                if supported_major == crate::sized::FORMAT_VERSION_MAJOR
        ));
        let message = outcome.unwrap_err().to_string();
        assert!(
            message.contains(&stamp),
            "error must name the file's own version: {message}"
        );
        assert!(
            message.contains(&crate::sized::FORMAT_VERSION_MAJOR.to_string()),
            "error must name the supported range: {message}"
        );
    }
}
