//! Sans-IO FSM over the whole safetensors byte stream. The cursor (here,
//! "which phase, and how many data bytes have we counted") lives IN the
//! discriminated-enum variant, never beside a buffer that could drift out
//! of sync with it — the same invariant `proxima_codec::DelimiterFraming`
//! (`proxima-codec/src/lib.rs`) buys with its own self-consuming
//! `push`/`next_frame`.
//!
//! [`SafetensorsParser`] implements
//! [`proxima_primitives::pipe::sans_io::ByteStreamParser`] — the same
//! `feed`/`poll`, `&mut self` contract `proxima-gguf::GgufParser` and
//! `proxima-onnx::OnnxParser` satisfy. `poll` computes the new variant
//! (`Header` -> `TensorData`) as an owned value first, from data already
//! read out of the old variant, then assigns `*self` to it in one step —
//! no `unsafe`, no placeholder, no extra allocation, and the enum-folded
//! cursor invariant `DelimiterFraming` motivates is unchanged; only the
//! public boundary moved from self-consuming to `&mut self`.
//! [`SafetensorsParser::push`] stays as a convenience built on
//! `feed`/`poll` for callers that prefer threading an owned `Self` through
//! a fold — it is sugar now, not the primitive.
//!
//! Neither shape is a [`Pipe`](proxima_primitives::pipe::Pipe) — see
//! `crate::header_codec` for why that stateless one-shot step (parsing one
//! already-complete header frame) is the `Pipe`, and this stateful
//! multi-chunk accumulation loop is not.
//!
//! Only the still-unparsed header bytes are ever buffered. Once the
//! header is parsed, tensor-data bytes are COUNTED, never buffered or
//! copied — this crate hands back byte ranges, never the tensor bytes
//! themselves.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use proxima_primitives::pipe::sans_io::{ByteStreamParser, Outcome};
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

    /// Append bytes fed by the caller. `Header` phase buffers them (the
    /// still-unparsed length prefix + JSON); `TensorData` phase only
    /// counts them — never buffered or copied.
    pub fn feed(&mut self, chunk: &[u8]) {
        match self {
            Self::Header { buf, .. } => buf.extend_from_slice(chunk),
            Self::TensorData { seen, .. } => *seen += chunk.len() as u64,
        }
    }

    /// Attempt one unit of progress against the currently buffered bytes.
    /// Emits exactly one [`Manifest`] event, the moment the header frame
    /// completes; `TensorData` phase has nothing further to report
    /// incrementally (byte counting alone needs no event) and always
    /// answers [`Outcome::NeedMore`] until [`Self::finish`] validates the
    /// total.
    pub fn poll(&mut self) -> Result<Outcome<&Manifest>, SafetensorsError> {
        let Self::Header { buf, max_header_bytes } = self else {
            return Ok(Outcome::NeedMore);
        };
        match HeaderCodec.parse_frame_with_limit(buf, *max_header_bytes) {
            Ok((header_json, consumed)) => {
                let manifest = parse_manifest(header_json)?;
                let seen = (buf.len() - consumed) as u64;
                *self = Self::TensorData { manifest, seen };
                let Self::TensorData { manifest, .. } = self else {
                    unreachable!("just assigned TensorData above")
                };
                Ok(Outcome::Event(manifest))
            }
            Err(SafetensorsError::TruncatedInput { .. }) => Ok(Outcome::NeedMore),
            Err(error) => Err(error),
        }
    }

    /// The caller has no more bytes to feed. Read-only: validates every
    /// tensor's `data_offsets` against the bytes actually counted, or
    /// reports the header as truncated if it never completed. Use
    /// [`Self::into_manifest`] to consume the parser and get the finished
    /// [`Manifest`] back once this returns `Ok(())`.
    pub fn finish(&self) -> Result<(), SafetensorsError> {
        match self {
            Self::Header { buf, .. } => {
                let needed = declared_total_len(buf).unwrap_or(HEADER_LEN_BYTES as u64);
                Err(SafetensorsError::TruncatedInput {
                    needed,
                    available: buf.len() as u64,
                })
            }
            Self::TensorData { manifest, seen } => validate_offsets_in_bounds(manifest, *seen),
        }
    }

    /// Consume the parser and hand back the finished [`Manifest`], or the
    /// typed error [`Self::finish`] would have reported. The owning
    /// counterpart to [`Self::finish`]'s read-only check — most callers
    /// that have already driven the parser to completion want the
    /// manifest, not just a validity bit.
    pub fn into_manifest(self) -> Result<Manifest, SafetensorsError> {
        self.finish()?;
        match self {
            Self::TensorData { manifest, .. } => Ok(manifest),
            Self::Header { .. } => unreachable!("finish() already rejected the Header phase"),
        }
    }

    /// Feed the next chunk, however it was split from the whole stream,
    /// and drain it immediately. Convenience sugar over [`Self::feed`] +
    /// [`Self::poll`] for callers that prefer threading an owned `Self`
    /// through a fold instead of holding a `&mut` binding — the shape
    /// `proxima_codec::DelimiterFraming::push` uses.
    pub fn push(mut self, chunk: &[u8]) -> Result<Self, SafetensorsError> {
        self.feed(chunk);
        loop {
            match self.poll()? {
                Outcome::NeedMore => break,
                Outcome::Event(_manifest) => {}
            }
        }
        Ok(self)
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

impl ByteStreamParser for SafetensorsParser {
    type Event<'a>
        = &'a Manifest
    where
        Self: 'a;
    type Error = SafetensorsError;

    fn feed(&mut self, bytes: &[u8]) {
        Self::feed(self, bytes);
    }

    fn poll(&mut self) -> Result<Outcome<&Manifest>, SafetensorsError> {
        Self::poll(self)
    }

    fn finish(&self) -> Result<(), SafetensorsError> {
        Self::finish(self)
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
