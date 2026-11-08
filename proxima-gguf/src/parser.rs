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

use alloc::collections::VecDeque;
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
    Flush,
    Done,
}

/// The state machine itself. Owns one growing accumulation buffer;
/// completed items are drained off the front so the buffer only ever holds
/// the not-yet-parsed tail of whatever's been fed.
pub struct GgufParser {
    accumulator: Vec<u8>,
    /// How far into `accumulator` the parser has consumed. Bytes before it
    /// are dead but are not moved -- see [`GgufParser::commit`].
    cursor: usize,
    phase: Phase,
    stream_pos: u64,
    pending_version: u32,
    seen_keys: Vec<String>,
    seen_tensor_names: Vec<String>,
    resolved_alignment: Option<u32>,
    tensor_size_total: u64,
    group128_size_total: u64,
    group64_consistent: bool,
    group128_consistent: bool,
    pending_tensors: VecDeque<TensorInfo>,
    file_len: Option<u64>,
    max_supported_version: u32,
    default_alignment: u32,
}

impl Default for GgufParser {
    fn default() -> Self {
        Self::new()
    }
}

/// The same `&mut self` `feed`/`poll` contract `Self::feed`/`Self::poll`
/// already establish (this module's own doc: `GgufParser` imitated
/// `h1_connection::Connection`'s shape before this trait existed to name
/// it), now expressed against the shared
/// [`proxima_primitives::pipe::sans_io::ByteStreamParser`] contract so a
/// generic driver (`drive_to_completion`) can run `GgufParser`,
/// `proxima_onnx::OnnxParser`, and `proxima_safetensors::SafetensorsParser`
/// alike.
impl proxima_primitives::pipe::sans_io::ByteStreamParser for GgufParser {
    type Event<'a>
        = GgufEvent
    where
        Self: 'a;
    type Error = GgufError;

    fn feed(&mut self, bytes: &[u8]) {
        Self::feed(self, bytes);
    }

    fn poll(&mut self) -> Result<proxima_primitives::pipe::sans_io::Outcome<GgufEvent>, GgufError> {
        Ok(match Self::poll(self)? {
            Some(event) => proxima_primitives::pipe::sans_io::Outcome::Event(event),
            None => proxima_primitives::pipe::sans_io::Outcome::NeedMore,
        })
    }

    fn finish(&self) -> Result<(), GgufError> {
        Self::finish(self)
    }
}

impl GgufParser {
    #[must_use]
    pub fn new() -> Self {
        Self {
            accumulator: Vec::new(),
            cursor: 0,
            phase: Phase::Magic,
            stream_pos: 0,
            pending_version: 0,
            seen_keys: Vec::new(),
            seen_tensor_names: Vec::new(),
            resolved_alignment: None,
            tensor_size_total: 0,
            group128_size_total: 0,
            group64_consistent: true,
            group128_consistent: true,
            pending_tensors: VecDeque::new(),
            file_len: None,
            max_supported_version: crate::sized::MAX_SUPPORTED_VERSION,
            default_alignment: crate::sized::DEFAULT_ALIGNMENT,
        }
    }

    /// Total stream length when the caller has it all in hand. Only used to
    /// size a trailing type-42 tensor that no later tensor offset can
    /// discriminate.
    #[must_use]
    pub fn with_file_len(mut self, file_len: u64) -> Self {
        self.file_len = Some(file_len);
        self
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
    ///
    /// Reclaims already-consumed bytes first, so a caller streaming many
    /// small chunks does not grow the buffer without bound. That memmove is
    /// O(unparsed tail) once per chunk, not once per field the way
    /// [`Self::commit`]'s `drain` was.
    fn feed(&mut self, bytes: &[u8]) {
        if self.cursor > 0 {
            self.accumulator.drain(0..self.cursor);
            self.cursor = 0;
        }
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
            Phase::Flush => Ok(Some(self.flush_next())),
            Phase::Done => Ok(None),
        }
    }

    /// Advances past `consumed` bytes without moving the ones behind it.
    ///
    /// This was `accumulator.drain(0..consumed)`, which memmoves the entire
    /// remaining tail forward on every field parsed. A caller that hands the
    /// whole file as one chunk -- which `parse_complete` does, and which is
    /// the only way to parse from an mmap -- therefore paid O(file_len) per
    /// field: measured 318 commits moving 1.23 TB to read a directory of a
    /// few hundred KB, 27.2s of a 29.6s load.
    fn commit(&mut self, consumed: usize) {
        self.cursor += consumed;
        self.stream_pos += consumed as u64;
    }

    /// The not-yet-consumed tail. Every `poll_*` reads from here rather than
    /// from the front of `accumulator`, which is what lets [`Self::commit`]
    /// be a cursor bump instead of a memmove.
    fn available(&self) -> &[u8] {
        &self.accumulator[self.cursor..]
    }

    fn poll_magic(&mut self) -> Result<Option<GgufEvent>, GgufError> {
        let mut reader = Reader::new(self.available());
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
        let mut reader = Reader::new(self.available());
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
        let mut reader = Reader::new(self.available());
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
        let mut reader = Reader::new(self.available());
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

    fn poll_kv(
        &mut self,
        tensor_count: u64,
        remaining: u64,
    ) -> Result<Option<GgufEvent>, GgufError> {
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

        let mut reader = Reader::new(self.available());
        let Some(key_result) = reader.string() else {
            return Ok(None);
        };
        let key = key_result.map_err(string_error_into_gguf)?;

        let Some(raw_type) = reader.u32() else {
            return Ok(None);
        };
        let metadata_type =
            MetadataType::from_wire(raw_type).ok_or_else(|| GgufError::InvalidMetadataType {
                key: key.clone(),
                raw: raw_type,
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
            let alignment = value.as_u32().ok_or(GgufError::InvalidAlignmentType)?;
            self.resolved_alignment = Some(alignment);
        }
        self.seen_keys.push(key.clone());

        self.phase = Phase::Kv {
            tensor_count,
            remaining: remaining - 1,
        };
        Ok(Some(GgufEvent::Metadata { key, value }))
    }

    fn poll_tensor(
        &mut self,
        mut remaining: u64,
        mut index: u64,
    ) -> Result<Option<GgufEvent>, GgufError> {
        while remaining > 0 {
            if !self.read_tensor(remaining, index)? {
                return Ok(None);
            }
            remaining -= 1;
            index += 1;
        }
        self.resolve_q2_0_layout()?;
        self.phase = Phase::Flush;
        Ok(Some(self.flush_next()))
    }

    fn flush_next(&mut self) -> GgufEvent {
        if let Some(tensor) = self.pending_tensors.pop_front() {
            return GgufEvent::Tensor(tensor);
        }
        let alignment = self.resolved_alignment.unwrap_or(self.default_alignment);
        let data_offset = pad_to_alignment(self.stream_pos, alignment);
        self.phase = Phase::Done;
        GgufEvent::Complete {
            data_offset,
            alignment,
        }
    }

    fn read_tensor(&mut self, remaining: u64, index: u64) -> Result<bool, GgufError> {
        let mut reader = Reader::new(self.available());
        let Some(name_result) = reader.string() else {
            return Ok(false);
        };
        let name = name_result.map_err(string_error_into_gguf)?;
        if name.len() > MAX_NAME_LEN {
            return Err(GgufError::NameTooLong {
                len: name.len(),
                max: MAX_NAME_LEN,
            });
        }

        let Some(n_dims) = reader.u32() else {
            return Ok(false);
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
                return Ok(false);
            };
            // never fails: n_dims <= MAX_DIMS was checked above.
            let _ = dims.try_push(dim);
        }

        let Some(raw_type) = reader.i32() else {
            return Ok(false);
        };
        let ggml_type =
            GgmlType::from_wire(raw_type).ok_or_else(|| GgufError::InvalidGgmlType {
                tensor: name.clone(),
                raw: raw_type,
            })?;

        let Some(offset) = reader.u64() else {
            return Ok(false);
        };

        let block_size = ggml_type.wire_shape().block_elements;
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
        let consumed = reader.consumed();
        self.group64_consistent &= offset == self.tensor_size_total;
        self.group128_consistent &= offset == self.group128_size_total;
        if !self.group64_consistent && !self.group128_consistent {
            return Err(self.offset_error(name, offset));
        }

        self.commit(consumed);

        let tensor = TensorInfo {
            name: name.clone(),
            dims,
            ggml_type,
            offset,
        };
        let alignment = self.resolved_alignment.unwrap_or(self.default_alignment);
        self.tensor_size_total =
            Self::advance_total(self.tensor_size_total, tensor.nbytes(), alignment)?;
        self.group128_size_total = Self::advance_total(
            self.group128_size_total,
            tensor.nbytes_as(group128_type(ggml_type)),
            alignment,
        )?;
        self.seen_tensor_names.push(name);
        self.pending_tensors.push_back(tensor);

        self.phase = Phase::Tensor {
            remaining: remaining - 1,
            index: index + 1,
        };
        Ok(true)
    }

    fn advance_total(total: u64, nbytes: Option<u64>, alignment: u32) -> Result<u64, GgufError> {
        let nbytes = nbytes.ok_or(GgufError::Overflow {
            context: "tensor byte size",
        })?;
        total
            .checked_add(pad_to_alignment(nbytes, alignment))
            .ok_or(GgufError::Overflow {
                context: "tensor directory total size",
            })
    }

    fn offset_error(&self, tensor: String, found: u64) -> GgufError {
        if self.tensor_size_total == self.group128_size_total {
            return GgufError::TensorOffsetMismatch {
                tensor,
                expected: self.tensor_size_total,
                found,
            };
        }
        GgufError::Q2_0LayoutUnresolved {
            tensor,
            group64_expected: self.tensor_size_total,
            group128_expected: self.group128_size_total,
            found,
        }
    }

    fn resolve_q2_0_layout(&mut self) -> Result<(), GgufError> {
        let group128 = match (self.group64_consistent, self.group128_consistent) {
            (false, true) => true,
            (true, true) => self.trailing_tensor_is_group128()?,
            _ => false,
        };
        if group128 {
            self.reinterpret_as_pq2_0()?;
        }
        Ok(())
    }

    fn trailing_tensor_is_group128(&self) -> Result<bool, GgufError> {
        let (Some(file_len), Some(last)) = (self.file_len, self.pending_tensors.back()) else {
            return Ok(false);
        };
        if self.tensor_size_total == self.group128_size_total {
            return Ok(false);
        }
        let alignment = self.resolved_alignment.unwrap_or(self.default_alignment);
        let data_offset = pad_to_alignment(self.stream_pos, alignment);
        let span = file_len.saturating_sub(data_offset.saturating_add(last.offset));
        let group64 = last.nbytes().unwrap_or(0);
        let group128 = last.nbytes_as(group128_type(last.ggml_type)).unwrap_or(0);
        let matches =
            |expected: u64| span == expected || span == pad_to_alignment(expected, alignment);
        match (matches(group64), matches(group128)) {
            (false, true) => Ok(true),
            (true, false) => Ok(false),
            _ => Err(GgufError::Q2_0LayoutUnresolved {
                tensor: last.name.clone(),
                group64_expected: group64,
                group128_expected: group128,
                found: span,
            }),
        }
    }

    fn reinterpret_as_pq2_0(&mut self) -> Result<(), GgufError> {
        for tensor in self
            .pending_tensors
            .iter_mut()
            .filter(|tensor| tensor.ggml_type == GgmlType::Q2_0)
        {
            let block_size = GgmlType::Pq2_0.wire_shape().block_elements;
            let row_len = tensor.dims.first().copied().unwrap_or(1);
            if row_len % block_size != 0 {
                return Err(GgufError::RowSizeNotBlockMultiple {
                    tensor: tensor.name.clone(),
                    ne0: row_len,
                    block_size,
                });
            }
            tensor.ggml_type = GgmlType::Pq2_0;
        }
        #[cfg(feature = "std")]
        {
            let reinterpreted = self
                .pending_tensors
                .iter()
                .filter(|tensor| tensor.ggml_type == GgmlType::Pq2_0)
                .count();
            proxima_telemetry::warn!(
                reinterpreted,
                "type 42 tensors span group-128 blocks, reading them as pq2_0"
            );
        }
        Ok(())
    }
}

fn group128_type(ggml_type: GgmlType) -> GgmlType {
    match ggml_type {
        GgmlType::Q2_0 => GgmlType::Pq2_0,
        other => other,
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
