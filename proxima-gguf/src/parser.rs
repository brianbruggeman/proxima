//! The sans-IO GGUF parser: a byte-fed state machine, patterned after
//! `proxima-protocols`' `http1_codec::h1_connection::Connection`
//! (`feed_bytes` / `poll` over an internal growing buffer, `Poll::NeedInput`
//! signalling "give me more bytes" — see
//! `proxima-protocols/src/http1_codec/h1_connection.rs:206-260`). Like that
//! connection type, `GgufParser` does no IO of its own: it never opens a
//! file, never seeks, never mmaps. The caller (an mmap'd slice, a
//! `std::fs::read`, bytes off a socket, anything) owns getting bytes in
//! front of it via [`GgufParser::push`].
//!
//! Layout: `gguf.h:1-30` (header + KV + tensor-directory shape),
//! `gguf.cpp:319-636` (`gguf_init_from_file_impl`, the exact validation
//! order this mirrors).
//!
//! `GgufEvent` is fully owned (no borrow into the parser), so the public
//! boundary is the self-consuming `push(self, chunk) -> (Self,
//! Vec<GgufEvent>)` shape: internally it still steps phase-by-phase over an
//! accumulation buffer, draining every event a chunk unlocks before handing
//! the parser back.

use alloc::string::String;
use alloc::vec::Vec;

use arrayvec::ArrayVec;

use crate::error::GgufError;
use crate::reader::{Reader, StringError};
use crate::sized::{MAX_DIMS, MAX_NAME_LEN};
use crate::tensor::TensorInfo;
use crate::types::{GgmlType, MetadataType, ScalarType};
use crate::value::{MetadataArray, MetadataValue};

/// GGUF magic bytes (`gguf.h:41`). Not config -- this is the format's
/// identity byte pattern, not a policy knob.
pub const MAGIC: [u8; 4] = *b"GGUF";

/// One unit of progress the parser can report.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufEvent {
    /// Magic, version, and both directory counts have been read.
    Header {
        version: u32,
        tensor_count: u64,
        kv_count: u64,
    },
    /// One metadata KV pair.
    Metadata { key: String, value: MetadataValue },
    /// One tensor directory entry, offset already validated contiguous.
    Tensor(TensorInfo),
    /// All metadata and the tensor directory are parsed. `data_offset` is
    /// the alignment-padded byte offset (from the start of the stream)
    /// where the data section begins; add each `TensorInfo::offset` to it
    /// to get that tensor's absolute file offset.
    Complete { data_offset: u64, alignment: u32 },
}

#[derive(Debug, Clone, PartialEq)]
enum Phase {
    Magic,
    Version,
    TensorCount,
    KvCount { tensor_count: u64 },
    Kv { tensor_count: u64, remaining: u64 },
    Tensor { remaining: u64, index: u64 },
    Done,
}

/// The state machine itself. Owns one growing accumulation buffer;
/// completed items are drained off the front so the buffer only ever holds
/// the not-yet-parsed tail of whatever's been fed.
pub struct GgufParser {
    accumulator: Vec<u8>,
    phase: Phase,
    stream_pos: u64,
    pending_version: u32,
    seen_keys: Vec<String>,
    seen_tensor_names: Vec<String>,
    resolved_alignment: Option<u32>,
    tensor_size_total: u64,
    max_supported_version: u32,
    default_alignment: u32,
}

impl Default for GgufParser {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufParser {
    #[must_use]
    pub fn new() -> Self {
        Self {
            accumulator: Vec::new(),
            phase: Phase::Magic,
            stream_pos: 0,
            pending_version: 0,
            seen_keys: Vec::new(),
            seen_tensor_names: Vec::new(),
            resolved_alignment: None,
            tensor_size_total: 0,
            max_supported_version: crate::sized::MAX_SUPPORTED_VERSION,
            default_alignment: crate::sized::DEFAULT_ALIGNMENT,
        }
    }

    /// Construct a parser using [`crate::config::GgufParserConfig`]'s
    /// resolved `max_supported_version`/`default_alignment` instead of the
    /// build-time `sized` floor. `std`-only: the no_std+alloc floor has no
    /// runtime config source, so [`Self::new`] is the only constructor
    /// there and always uses `crate::sized` directly.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn with_config(config: &crate::config::GgufParserConfig) -> Self {
        Self {
            max_supported_version: config.max_supported_version,
            default_alignment: config.default_alignment,
            ..Self::new()
        }
    }

    /// Append bytes fed by the caller. Never blocks, never inspects the
    /// bytes — parsing happens in [`Self::poll`].
    fn feed(&mut self, bytes: &[u8]) {
        self.accumulator.extend_from_slice(bytes);
    }

    /// True once [`GgufEvent::Complete`] has been emitted.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.phase == Phase::Done
    }

    /// The caller has no more bytes to feed. `Ok(())` only if the parser
    /// had already reached `Phase::Done` — otherwise the stream ended
    /// mid-item.
    pub fn finish(&self) -> Result<(), GgufError> {
        if self.phase == Phase::Done {
            Ok(())
        } else {
            Err(GgufError::TruncatedInput)
        }
    }

    /// Feed the next chunk, however it was split from the whole stream,
    /// and drain every event it unlocks before handing the parser back.
    pub fn push(mut self, chunk: &[u8]) -> Result<(Self, Vec<GgufEvent>), GgufError> {
        self.feed(chunk);
        let mut events = Vec::new();
        while let Some(event) = self.poll()? {
            events.push(event);
        }
        Ok((self, events))
    }

    /// Attempt one unit of progress against the currently buffered bytes.
    fn poll(&mut self) -> Result<Option<GgufEvent>, GgufError> {
        match self.phase.clone() {
            Phase::Magic => self.poll_magic(),
            Phase::Version => self.poll_version(),
            Phase::TensorCount => self.poll_tensor_count(),
            Phase::KvCount { tensor_count } => self.poll_kv_count(tensor_count),
            Phase::Kv {
                tensor_count,
                remaining,
            } => self.poll_kv(tensor_count, remaining),
            Phase::Tensor { remaining, index } => self.poll_tensor(remaining, index),
            Phase::Done => Ok(None),
        }
    }

    fn commit(&mut self, consumed: usize) {
        self.accumulator.drain(0..consumed);
        self.stream_pos += consumed as u64;
    }

    fn poll_magic(&mut self) -> Result<Option<GgufEvent>, GgufError> {
        let mut reader = Reader::new(self.accumulator.as_slice());
        let Some(found) = reader.u32() else {
            return Ok(None);
        };
        let found_bytes = found.to_le_bytes();
        if found_bytes != MAGIC {
            return Err(GgufError::BadMagic { found: found_bytes });
        }
        let consumed = reader.consumed();
        self.commit(consumed);
        self.phase = Phase::Version;
        self.poll()
    }

    fn poll_version(&mut self) -> Result<Option<GgufEvent>, GgufError> {
        let mut reader = Reader::new(self.accumulator.as_slice());
        let Some(version) = reader.u32() else {
            return Ok(None);
        };
        if version == 0 || version == 1 || version > self.max_supported_version {
            return Err(GgufError::UnsupportedVersion { version });
        }
        let consumed = reader.consumed();
        self.commit(consumed);
        self.pending_version = version;
        self.phase = Phase::TensorCount;
        self.poll()
    }

    fn poll_tensor_count(&mut self) -> Result<Option<GgufEvent>, GgufError> {
        let mut reader = Reader::new(self.accumulator.as_slice());
        let Some(raw) = reader.i64() else {
            return Ok(None);
        };
        let tensor_count = u64::try_from(raw).map_err(|_| GgufError::Overflow {
            context: "tensor count",
        })?;
        let consumed = reader.consumed();
        self.commit(consumed);
        self.phase = Phase::KvCount { tensor_count };
        self.poll()
    }

    fn poll_kv_count(&mut self, tensor_count: u64) -> Result<Option<GgufEvent>, GgufError> {
        let mut reader = Reader::new(self.accumulator.as_slice());
        let Some(raw) = reader.i64() else {
            return Ok(None);
        };
        let kv_count = u64::try_from(raw).map_err(|_| GgufError::Overflow {
            context: "kv count",
        })?;
        let consumed = reader.consumed();
        self.commit(consumed);
        self.phase = Phase::Kv {
            tensor_count,
            remaining: kv_count,
        };
        Ok(Some(GgufEvent::Header {
            version: self.pending_version,
            tensor_count,
            kv_count,
        }))
    }

    fn poll_kv(&mut self, tensor_count: u64, remaining: u64) -> Result<Option<GgufEvent>, GgufError> {
        if remaining == 0 {
            let alignment = self.resolved_alignment.unwrap_or(self.default_alignment);
            if alignment == 0 || !alignment.is_power_of_two() {
                return Err(GgufError::InvalidAlignment { value: alignment });
            }
            self.resolved_alignment = Some(alignment);
            self.phase = Phase::Tensor {
                remaining: tensor_count,
                index: 0,
            };
            return self.poll();
        }

        let mut reader = Reader::new(self.accumulator.as_slice());
        let Some(key_result) = reader.string() else {
            return Ok(None);
        };
        let key = key_result.map_err(string_error_into_gguf)?;

        let Some(raw_type) = reader.u32() else {
            return Ok(None);
        };
        let metadata_type = MetadataType::from_wire(raw_type).ok_or_else(|| {
            GgufError::InvalidMetadataType {
                key: key.clone(),
                raw: raw_type,
            }
        })?;

        let value = if metadata_type == MetadataType::Array {
            let Some(raw_element_type) = reader.u32() else {
                return Ok(None);
            };
            let element_type = MetadataType::from_wire(raw_element_type).ok_or_else(|| {
                GgufError::InvalidMetadataType {
                    key: key.clone(),
                    raw: raw_element_type,
                }
            })?;
            let scalar_type = ScalarType::from_metadata_type(element_type)
                .ok_or_else(|| GgufError::NestedArrayNotSupported { key: key.clone() })?;
            let Some(len) = reader.u64() else {
                return Ok(None);
            };
            let len_usize = usize::try_from(len).map_err(|_| GgufError::Overflow {
                context: "metadata array length",
            })?;
            let Some(array) = read_array(&mut reader, scalar_type, len_usize)? else {
                return Ok(None);
            };
            MetadataValue::Array(array)
        } else {
            let scalar_type = ScalarType::from_metadata_type(metadata_type)
                .ok_or_else(|| GgufError::NestedArrayNotSupported { key: key.clone() })?;
            let Some(value) = read_scalar(&mut reader, scalar_type) else {
                return Ok(None);
            };
            value.map_err(string_error_into_gguf)?
        };

        if self.seen_keys.contains(&key) {
            return Err(GgufError::DuplicateKey { key });
        }
        let consumed = reader.consumed();
        self.commit(consumed);

        if key == "general.alignment" {
            let alignment = value
                .as_u32()
                .ok_or(GgufError::InvalidAlignmentType)?;
            self.resolved_alignment = Some(alignment);
        }
        self.seen_keys.push(key.clone());

        self.phase = Phase::Kv {
            tensor_count,
            remaining: remaining - 1,
        };
        Ok(Some(GgufEvent::Metadata { key, value }))
    }

    fn poll_tensor(&mut self, remaining: u64, index: u64) -> Result<Option<GgufEvent>, GgufError> {
        if remaining == 0 {
            let alignment = self.resolved_alignment.unwrap_or(self.default_alignment);
            let data_offset = pad_to_alignment(self.stream_pos, alignment);
            self.phase = Phase::Done;
            return Ok(Some(GgufEvent::Complete {
                data_offset,
                alignment,
            }));
        }

        let mut reader = Reader::new(self.accumulator.as_slice());
        let Some(name_result) = reader.string() else {
            return Ok(None);
        };
        let name = name_result.map_err(string_error_into_gguf)?;
        if name.len() > MAX_NAME_LEN {
            return Err(GgufError::NameTooLong {
                len: name.len(),
                max: MAX_NAME_LEN,
            });
        }

        let Some(n_dims) = reader.u32() else {
            return Ok(None);
        };
        if n_dims as usize > MAX_DIMS {
            return Err(GgufError::TooManyDimensions {
                tensor: name,
                found: n_dims,
            });
        }

        let mut dims: ArrayVec<u64, MAX_DIMS> = ArrayVec::new();
        for _ in 0..n_dims {
            let Some(dim) = reader.u64() else {
                return Ok(None);
            };
            // never fails: n_dims <= MAX_DIMS was checked above.
            let _ = dims.try_push(dim);
        }

        let Some(raw_type) = reader.i32() else {
            return Ok(None);
        };
        let ggml_type = GgmlType::from_wire(raw_type).ok_or_else(|| GgufError::InvalidGgmlType {
            tensor: name.clone(),
            raw: raw_type,
        })?;

        let Some(offset) = reader.u64() else {
            return Ok(None);
        };

        let block_size = ggml_type.block_layout().block_elements;
        let row_len = dims.first().copied().unwrap_or(1);
        if block_size == 0 || row_len % block_size != 0 {
            return Err(GgufError::RowSizeNotBlockMultiple {
                tensor: name,
                ne0: row_len,
                block_size,
            });
        }

        if self.seen_tensor_names.contains(&name) {
            return Err(GgufError::DuplicateTensorName { name });
        }
        if offset != self.tensor_size_total {
            return Err(GgufError::TensorOffsetMismatch {
                tensor: name,
                expected: self.tensor_size_total,
                found: offset,
            });
        }

        let consumed = reader.consumed();
        self.commit(consumed);

        let tensor = TensorInfo {
            name: name.clone(),
            dims,
            ggml_type,
            offset,
        };
        let nbytes = tensor.nbytes().ok_or(GgufError::Overflow {
            context: "tensor byte size",
        })?;
        let alignment = self.resolved_alignment.unwrap_or(self.default_alignment);
        self.tensor_size_total = self
            .tensor_size_total
            .checked_add(pad_to_alignment(nbytes, alignment))
            .ok_or(GgufError::Overflow {
                context: "tensor directory total size",
            })?;
        self.seen_tensor_names.push(name);

        self.phase = Phase::Tensor {
            remaining: remaining - 1,
            index: index + 1,
        };
        Ok(Some(GgufEvent::Tensor(tensor)))
    }
}

pub(crate) fn pad_to_alignment(value: u64, alignment: u32) -> u64 {
    let alignment = u64::from(alignment);
    if alignment == 0 {
        return value;
    }
    let remainder = value % alignment;
    if remainder == 0 {
        value
    } else {
        value + (alignment - remainder)
    }
}

fn string_error_into_gguf(error: StringError) -> GgufError {
    match error {
        StringError::TooLarge(len) => GgufError::StringTooLarge { len },
        StringError::InvalidUtf8 => GgufError::InvalidUtf8,
    }
}

fn read_scalar(
    reader: &mut Reader<'_>,
    scalar_type: ScalarType,
) -> Option<Result<MetadataValue, StringError>> {
    match scalar_type {
        ScalarType::U8 => reader.u8().map(|v| Ok(MetadataValue::U8(v))),
        ScalarType::I8 => reader.i8().map(|v| Ok(MetadataValue::I8(v))),
        ScalarType::U16 => reader.u16().map(|v| Ok(MetadataValue::U16(v))),
        ScalarType::I16 => reader.i16().map(|v| Ok(MetadataValue::I16(v))),
        ScalarType::U32 => reader.u32().map(|v| Ok(MetadataValue::U32(v))),
        ScalarType::I32 => reader.i32().map(|v| Ok(MetadataValue::I32(v))),
        ScalarType::F32 => reader.f32().map(|v| Ok(MetadataValue::F32(v))),
        ScalarType::Bool => reader.bool().map(|v| Ok(MetadataValue::Bool(v))),
        ScalarType::String => reader
            .string()
            .map(|result| result.map(MetadataValue::String)),
        ScalarType::U64 => reader.u64().map(|v| Ok(MetadataValue::U64(v))),
        ScalarType::I64 => reader.i64().map(|v| Ok(MetadataValue::I64(v))),
        ScalarType::F64 => reader.f64().map(|v| Ok(MetadataValue::F64(v))),
    }
}

/// Reads `len` elements of `scalar_type`, or `Ok(None)` if the buffer runs
/// out partway (the whole array is retried from scratch on the next `poll`,
/// same rollback discipline as every other field).
fn read_array(
    reader: &mut Reader<'_>,
    scalar_type: ScalarType,
    len: usize,
) -> Result<Option<MetadataArray>, GgufError> {
    macro_rules! collect_numeric {
        ($read:ident, $variant:ident) => {{
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                match reader.$read() {
                    Some(v) => values.push(v),
                    None => return Ok(None),
                }
            }
            Ok(Some(MetadataArray::$variant(values)))
        }};
    }

    match scalar_type {
        ScalarType::U8 => collect_numeric!(u8, U8),
        ScalarType::I8 => collect_numeric!(i8, I8),
        ScalarType::U16 => collect_numeric!(u16, U16),
        ScalarType::I16 => collect_numeric!(i16, I16),
        ScalarType::U32 => collect_numeric!(u32, U32),
        ScalarType::I32 => collect_numeric!(i32, I32),
        ScalarType::F32 => collect_numeric!(f32, F32),
        ScalarType::Bool => collect_numeric!(bool, Bool),
        ScalarType::U64 => collect_numeric!(u64, U64),
        ScalarType::I64 => collect_numeric!(i64, I64),
        ScalarType::F64 => collect_numeric!(f64, F64),
        ScalarType::String => {
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                match reader.string() {
                    Some(Ok(text)) => values.push(text),
                    Some(Err(error)) => return Err(string_error_into_gguf(error)),
                    None => return Ok(None),
                }
            }
            Ok(Some(MetadataArray::String(values)))
        }
    }
}
