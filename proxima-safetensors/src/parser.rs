//! Sans-IO FSM over the whole safetensors byte stream — patterned after
//! `proxima_codec::DelimiterFraming` (`proxima-codec/src/lib.rs`): the
//! cursor (here, "which phase, and how many data bytes have we counted")
//! lives IN the discriminated-enum variant, never beside a buffer that
//! could drift out of sync with it. [`SafetensorsParser::push`] is a
//! self-consuming `Self -> Self` transition, not a
//! [`Pipe`](proxima_primitives::pipe::Pipe) — see `crate::header_codec` for
//! why that stateless one-shot step is the `Pipe`, and this stateful
//! accumulation loop is not.
//!
//! Only the still-unparsed header bytes are ever buffered. Once the
//! header is parsed, tensor-data bytes are COUNTED, never buffered or
//! copied — this crate hands back byte ranges, never the tensor bytes
//! themselves.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use proxima_tensor::DType;

use crate::dtype::map_dtype;
use crate::error::SafetensorsError;
use crate::header_codec::HeaderCodec;
use crate::sized::{HEADER_LEN_BYTES, MAX_HEADER_BYTES};

/// One tensor's parsed directory entry. `data_offsets` are relative to
/// the start of the byte buffer — i.e. the first byte AFTER the header —
/// per spec; never an absolute file position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorEntry {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub data_offsets: (u64, u64),
}

impl TensorEntry {
    /// `END - BEGIN`, the tensor's raw byte size.
    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.data_offsets.1 - self.data_offsets.0
    }
}

/// Parsed directory: every tensor entry, plus the `__metadata__`
/// free-form string map if the file carried one. `__metadata__` is never
/// reported as a tensor entry.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Manifest {
    pub tensors: Vec<TensorEntry>,
    pub metadata: BTreeMap<String, String>,
}

impl Manifest {
    /// Look up one tensor's directory entry by name.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&TensorEntry> {
        self.tensors.iter().find(|entry| entry.name == name)
    }

    /// The number of tensor-data bytes the header declares — the
    /// farthest `END` offset across every tensor, or 0 if there are none.
    #[must_use]
    pub fn declared_data_len(&self) -> u64 {
        self.tensors
            .iter()
            .map(|entry| entry.data_offsets.1)
            .max()
            .unwrap_or(0)
    }

    /// Validates and returns the `(major, minor)` format-version stamp
    /// [`crate::writer::write_complete`] writes into `self.metadata` — see
    /// `crate::version` for the accept/reject table. A file with no stamp
    /// (every file this workspace wrote before the stamp existed) is
    /// accepted as `(1, 0)`.
    ///
    /// # Errors
    ///
    /// [`SafetensorsError::InvalidFormatVersion`] if the stamp doesn't
    /// parse as `major.minor`; [`SafetensorsError::UnsupportedFormatVersion`]
    /// if its major exceeds what this reader supports.
    pub fn format_version(&self) -> Result<(u16, u16), SafetensorsError> {
        crate::version::parse(&self.metadata)
    }
}

/// Sans-IO parser FSM. Feed chunks via [`Self::push`], split anywhere;
/// call [`Self::finish`] once no more bytes are coming.
#[derive(Debug, Clone)]
pub enum SafetensorsParser {
    /// Accumulating the 8-byte length prefix and then the declared JSON
    /// header. `buf` never grows past `8 + max_header_bytes`.
    Header { buf: Vec<u8>, max_header_bytes: u64 },
    /// Header parsed; counting tensor-data bytes as they arrive.
    TensorData { manifest: Manifest, seen: u64 },
}

impl SafetensorsParser {
    /// Applies [`crate::sized::MAX_HEADER_BYTES`], the build-time floor.
    /// [`Self::with_config`] is the `std`-tier entry point for a
    /// per-process override.
    #[must_use]
    pub const fn new() -> Self {
        Self::Header {
            buf: Vec::new(),
            max_header_bytes: MAX_HEADER_BYTES,
        }
    }

    /// Feed the next chunk, however it was split from the whole stream.
    pub fn push(self, chunk: &[u8]) -> Result<Self, SafetensorsError> {
        match self {
            Self::Header { mut buf, max_header_bytes } => {
                buf.extend_from_slice(chunk);
                match HeaderCodec.parse_frame_with_limit(&buf, max_header_bytes) {
                    Ok((header_json, consumed)) => {
                        let manifest = parse_manifest(header_json)?;
                        let tail_start = consumed;
                        let mut state = Self::TensorData { manifest, seen: 0 };
                        if tail_start < buf.len() {
                            state = state.push(&buf[tail_start..])?;
                        }
                        Ok(state)
                    }
                    Err(SafetensorsError::TruncatedInput { .. }) => Ok(Self::Header { buf, max_header_bytes }),
                    Err(error) => Err(error),
                }
            }
            Self::TensorData { manifest, seen } => {
                let seen = seen + chunk.len() as u64;
                Ok(Self::TensorData { manifest, seen })
            }
        }
    }

    /// Signal end of input. Validates every tensor's `data_offsets`
    /// against the bytes actually counted and returns the finished
    /// [`Manifest`], or the typed error explaining what was wrong.
    pub fn finish(self) -> Result<Manifest, SafetensorsError> {
        match self {
            Self::Header { buf, .. } => {
                let needed = declared_total_len(&buf).unwrap_or(HEADER_LEN_BYTES as u64);
                Err(SafetensorsError::TruncatedInput {
                    needed,
                    available: buf.len() as u64,
                })
            }
            Self::TensorData { manifest, seen } => {
                validate_offsets_in_bounds(&manifest, seen)?;
                Ok(manifest)
            }
        }
    }
}

#[cfg(feature = "std")]
impl SafetensorsParser {
    /// Same starting state as [`Self::new`], with
    /// `max_header_bytes` resolved from `config` instead of
    /// [`crate::sized::MAX_HEADER_BYTES`] directly -- the `std`-tier
    /// per-process override path.
    #[must_use]
    pub fn with_config(config: &crate::config::SafetensorsParserConfig) -> Self {
        Self::Header {
            buf: Vec::new(),
            max_header_bytes: config.max_header_bytes,
        }
    }
}

impl Default for SafetensorsParser {
    fn default() -> Self {
        Self::new()
    }
}

/// If `buf` already holds the 8-byte length prefix, the total byte count
/// (`8 + declared header length`) the parser is still waiting on.
fn declared_total_len(buf: &[u8]) -> Option<u64> {
    if buf.len() < HEADER_LEN_BYTES {
        return None;
    }
    let mut len_bytes = [0_u8; HEADER_LEN_BYTES];
    len_bytes.copy_from_slice(&buf[..HEADER_LEN_BYTES]);
    Some(HEADER_LEN_BYTES as u64 + u64::from_le_bytes(len_bytes))
}

fn validate_offsets_in_bounds(manifest: &Manifest, buffer_len: u64) -> Result<(), SafetensorsError> {
    for entry in &manifest.tensors {
        if entry.data_offsets.1 > buffer_len {
            return Err(SafetensorsError::OffsetOutOfBounds {
                tensor: entry.name.clone(),
                start: entry.data_offsets.0,
                end: entry.data_offsets.1,
                buffer_len,
            });
        }
    }
    Ok(())
}

fn parse_manifest(header_json: &[u8]) -> Result<Manifest, SafetensorsError> {
    let value: serde_json::Value = serde_json::from_slice(header_json).map_err(|error| {
        SafetensorsError::MalformedJson {
            reason: error.to_string(),
        }
    })?;
    let object = value.as_object().ok_or(SafetensorsError::HeaderNotAnObject)?;

    let mut tensors = Vec::new();
    let mut metadata = BTreeMap::new();

    for (name, entry) in object {
        if name.as_str() == "__metadata__" {
            metadata = parse_metadata(entry)?;
            continue;
        }
        tensors.push(parse_tensor_entry(name, entry)?);
    }

    check_no_overlaps(&tensors)?;
    Ok(Manifest { tensors, metadata })
}

fn parse_metadata(value: &serde_json::Value) -> Result<BTreeMap<String, String>, SafetensorsError> {
    let object = value.as_object().ok_or(SafetensorsError::InvalidField {
        tensor: "__metadata__".to_string(),
        field: "__metadata__",
    })?;
    let mut metadata = BTreeMap::new();
    for (key, val) in object {
        let string_val = val.as_str().ok_or_else(|| SafetensorsError::InvalidField {
            tensor: "__metadata__".to_string(),
            field: "__metadata__",
        })?;
        metadata.insert(key.clone(), string_val.to_string());
    }
    Ok(metadata)
}

fn parse_tensor_entry(name: &str, value: &serde_json::Value) -> Result<TensorEntry, SafetensorsError> {
    let dtype_str = value
        .get("dtype")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| SafetensorsError::MissingField {
            tensor: name.to_string(),
            field: "dtype",
        })?;
    let dtype = map_dtype(name, dtype_str)?;

    let shape_values = value
        .get("shape")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| SafetensorsError::MissingField {
            tensor: name.to_string(),
            field: "shape",
        })?;
    let mut shape = Vec::with_capacity(shape_values.len());
    for dim in shape_values {
        let dim = dim.as_u64().ok_or_else(|| SafetensorsError::InvalidField {
            tensor: name.to_string(),
            field: "shape",
        })?;
        shape.push(dim);
    }

    let offsets = value
        .get("data_offsets")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| SafetensorsError::MissingField {
            tensor: name.to_string(),
            field: "data_offsets",
        })?;
    if offsets.len() != 2 {
        return Err(SafetensorsError::InvalidField {
            tensor: name.to_string(),
            field: "data_offsets",
        });
    }
    let start = offsets[0]
        .as_u64()
        .ok_or_else(|| SafetensorsError::InvalidField {
            tensor: name.to_string(),
            field: "data_offsets",
        })?;
    let end = offsets[1]
        .as_u64()
        .ok_or_else(|| SafetensorsError::InvalidField {
            tensor: name.to_string(),
            field: "data_offsets",
        })?;
    if start > end {
        return Err(SafetensorsError::InvalidOffsets {
            tensor: name.to_string(),
            start,
            end,
        });
    }

    let element_count: u64 = shape.iter().product();
    let expected = element_count * dtype.size_bytes() as u64;
    let found = end - start;
    if expected != found {
        return Err(SafetensorsError::TensorDataLengthMismatch {
            tensor: name.to_string(),
            expected,
            found,
        });
    }

    Ok(TensorEntry {
        name: name.to_string(),
        dtype,
        shape,
        data_offsets: (start, end),
    })
}

fn check_no_overlaps(tensors: &[TensorEntry]) -> Result<(), SafetensorsError> {
    let mut sorted: Vec<&TensorEntry> = tensors.iter().collect();
    sorted.sort_by_key(|entry| entry.data_offsets.0);
    for pair in sorted.windows(2) {
        let (first, second) = (pair[0], pair[1]);
        if first.data_offsets.1 > second.data_offsets.0 {
            return Err(SafetensorsError::OverlappingTensors {
                first: first.name.clone(),
                second: second.name.clone(),
            });
        }
    }
    Ok(())
}
