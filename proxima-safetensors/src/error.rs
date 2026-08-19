//! Typed errors for the safetensors reader. Every malformed-input path in
//! `SCOPE` maps to one variant here — never a panic, never an
//! out-of-bounds read.

use alloc::string::String;

/// Everything that can go wrong parsing a safetensors byte stream.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SafetensorsError {
    /// The input ended before the 8-byte header-length prefix, the
    /// declared JSON header, or the declared tensor-data region was
    /// fully delivered.
    #[error("truncated input: needed at least {needed} bytes, got {available}")]
    TruncatedInput { needed: u64, available: u64 },

    /// The declared header length exceeds the safety cap (matches the
    /// reference `safetensors` crate's `MAX_HEADER_SIZE`, published to
    /// stop a corrupt or malicious length prefix from driving an
    /// unbounded allocation).
    #[error("header length {declared} exceeds max {max}")]
    HeaderTooLarge { declared: u64, max: u64 },

    /// The header bytes are not valid UTF-8 JSON, or do not decode into
    /// the `{name: {dtype, shape, data_offsets}}` shape the spec
    /// requires.
    #[error("malformed header json: {reason}")]
    MalformedJson { reason: String },

    /// The header's top-level JSON value is not an object.
    #[error("header json is not an object")]
    HeaderNotAnObject,

    /// A tensor entry is missing a required field.
    #[error("tensor {tensor:?} missing field {field}")]
    MissingField { tensor: String, field: &'static str },

    /// A tensor entry's field has the wrong JSON shape (e.g. `shape` is
    /// not an array of non-negative integers).
    #[error("tensor {tensor:?} field {field} has the wrong shape")]
    InvalidField { tensor: String, field: &'static str },

    /// The safetensors `dtype` string has no `proxima_tensor::DType`
    /// counterpart at this commit.
    #[error("tensor {tensor:?} has unsupported dtype {dtype:?}")]
    UnsupportedDtype { tensor: String, dtype: String },

    /// `data_offsets` are inverted (`start > end`).
    #[error("tensor {tensor:?} has invalid offsets [{start}, {end})")]
    InvalidOffsets { tensor: String, start: u64, end: u64 },

    /// `data_offsets` reference bytes past the end of the byte buffer
    /// that followed the header.
    #[error(
        "tensor {tensor:?} offsets [{start}, {end}) exceed the {buffer_len}-byte data buffer"
    )]
    OffsetOutOfBounds {
        tensor: String,
        start: u64,
        end: u64,
        buffer_len: u64,
    },

    /// Two tensors' `data_offsets` ranges overlap — the spec guarantees
    /// this never happens in a well-formed file.
    #[error("tensors {first:?} and {second:?} have overlapping data_offsets")]
    OverlappingTensors { first: String, second: String },
}
