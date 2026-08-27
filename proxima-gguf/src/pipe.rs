//! [`parse_complete`]: "I already have the whole metadata region as one
//! contiguous slice, parse all of it." That call is stateless from the
//! caller's point of view — no cursor to thread back in, no partial-input
//! case to report.
//!
//! [`crate::parser::GgufParser`] itself is a different shape: its `push` is
//! a self-consuming `Self -> Self` state transition threaded across a
//! caller-controlled number of chunks, with real internal state (the
//! accumulation buffer, the phase, the duplicate-key set). The FSM stays a
//! plain self-consuming type; this module is the single-call convenience
//! built on top of it.

use alloc::string::String;
use alloc::vec::Vec;

use crate::error::GgufError;
use crate::parser::{GgufEvent, GgufParser};
use crate::tensor::TensorInfo;
use crate::value::MetadataValue;

/// The fully-materialized result of parsing one complete GGUF metadata
/// region: header fields, every KV pair in file order, every tensor
/// directory entry, and where the data section starts.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedGguf {
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub metadata: Vec<(String, MetadataValue)>,
    pub tensors: Vec<TensorInfo>,
    pub data_offset: u64,
    pub alignment: u32,
}

impl ParsedGguf {
    /// Look up a metadata value by key (linear scan — KV counts are in the
    /// tens to low hundreds, not worth an index).
    #[must_use]
    pub fn metadata_value(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata
            .iter()
            .find(|(candidate, _)| candidate == key)
            .map(|(_, value)| value)
    }

    /// Absolute byte range of `tensor`'s data within the file that begins
    /// at `self.data_offset` — the range a caller mmaps or slices out.
    ///
    /// # Errors
    ///
    /// [`GgufError::Overflow`] if the tensor's byte size can't be computed
    /// (pathological dimensions), [`GgufError::TensorDataOutOfRange`] if
    /// that range would run past `file_len`.
    pub fn tensor_data_range(
        &self,
        tensor: &TensorInfo,
        file_len: u64,
    ) -> Result<core::ops::Range<u64>, GgufError> {
        let nbytes = tensor.nbytes().ok_or(GgufError::Overflow {
            context: "tensor byte size",
        })?;
        let start = self
            .data_offset
            .checked_add(tensor.offset)
            .ok_or(GgufError::Overflow {
                context: "tensor absolute offset",
            })?;
        let end = start.checked_add(nbytes).ok_or(GgufError::Overflow {
            context: "tensor absolute end",
        })?;
        if end > file_len {
            return Err(GgufError::TensorDataOutOfRange {
                tensor: tensor.name.clone(),
                start,
                end,
                file_len,
            });
        }
        Ok(start..end)
    }
}

/// A GGUF header plus tensor directory is a few hundred KB even for a
/// multi-gigabyte checkpoint, so [`parse_complete`] walks `input` a slice at
/// a time and stops at [`GgufParser::is_complete`] rather than handing over
/// the whole file. Feeding the whole thing copied every tensor byte into the
/// parser's accumulator to read a directory that sits entirely at the front:
/// measured 4.14 GB copied, which also forced the caller's whole mmap
/// resident and set the process's peak RSS.
const DIRECTORY_CHUNK_BYTES: usize = 1 << 20;

/// Parses one complete, already-assembled byte slice in a single call.
/// Stateless — a fresh [`GgufParser`] is built and driven internally on
/// every call.
///
/// Only reads as far into `input` as the tensor directory extends; the
/// tensor payload behind it is never touched. `input` still has to *be* the
/// whole file, because [`ParsedGguf::tensor_data_range`] validates offsets
/// against `input.len()`.
///
/// # Errors
///
/// Any [`GgufError`] the underlying [`GgufParser`] surfaces, plus
/// [`GgufError::TruncatedInput`] if `input` ends before the tensor
/// directory is fully parsed.
pub fn parse_complete(input: &[u8]) -> Result<ParsedGguf, GgufError> {
    let mut parser = GgufParser::new();
    let mut events = Vec::new();
    let mut offset = 0usize;

    while offset < input.len() && !parser.is_complete() {
        let end = input.len().min(offset + DIRECTORY_CHUNK_BYTES);
        let (advanced, unlocked) = parser.push(&input[offset..end])?;
        parser = advanced;
        events.extend(unlocked);
        offset = end;
    }

    let mut header = None;
    let mut metadata = Vec::new();
    let mut tensors = Vec::new();
    let mut completion = None;

    for event in events {
        match event {
            GgufEvent::Header {
                version,
                tensor_count,
                kv_count,
            } => header = Some((version, tensor_count, kv_count)),
            GgufEvent::Metadata { key, value } => {
                metadata.push((key, value));
            }
            GgufEvent::Tensor(tensor) => tensors.push(tensor),
            GgufEvent::Complete {
                data_offset,
                alignment,
            } => {
                completion = Some((data_offset, alignment));
            }
        }
    }

    parser.finish()?;
    let (version, tensor_count, kv_count) = header.ok_or(GgufError::TruncatedInput)?;
    let (data_offset, alignment) = completion.ok_or(GgufError::TruncatedInput)?;

    Ok(ParsedGguf {
        version,
        tensor_count,
        kv_count,
        metadata,
        tensors,
        data_offset,
        alignment,
    })
}
