//! Typed parse failures. Every malformed-input path returns one of these —
//! never a panic, never an out-of-bounds read.

use alloc::string::String;

use thiserror::Error;

/// Everything that can go wrong parsing a GGUF byte stream.
#[derive(Debug, Error, PartialEq, Eq, Clone)]
pub enum GgufError {
    #[error("bad magic: expected \"GGUF\", found {found:?}")]
    BadMagic { found: [u8; 4] },

    #[error("unsupported gguf version: {version}")]
    UnsupportedVersion { version: u32 },

    #[error("key '{key}' has invalid metadata type {raw}")]
    InvalidMetadataType { key: String, raw: u32 },

    #[error("key '{key}' is an array of arrays, which gguf does not support")]
    NestedArrayNotSupported { key: String },

    #[error("tensor '{tensor}' has invalid ggml type {raw}")]
    InvalidGgmlType { tensor: String, raw: i32 },

    #[error("string field is not valid utf-8")]
    InvalidUtf8,

    #[error("string length {len} does not fit in this platform's usize")]
    StringTooLarge { len: u64 },

    #[error("duplicate metadata key '{key}'")]
    DuplicateKey { key: String },

    #[error("duplicate tensor name '{name}'")]
    DuplicateTensorName { name: String },

    #[error("general.alignment must be a power of two, found {value}")]
    InvalidAlignment { value: u32 },

    #[error("general.alignment must be stored as u32")]
    InvalidAlignmentType,

    #[error("tensor '{tensor}' has {found} dimensions, at most 4 are supported")]
    TooManyDimensions { tensor: String, found: u32 },

    #[error(
        "tensor '{tensor}' row size (ne[0]={ne0}) is not a multiple of block size {block_size}"
    )]
    RowSizeNotBlockMultiple {
        tensor: String,
        ne0: u64,
        block_size: u64,
    },

    #[error(
        "tensor '{tensor}' has offset {found}, expected {expected} (data section must be contiguous)"
    )]
    TensorOffsetMismatch {
        tensor: String,
        expected: u64,
        found: u64,
    },

    #[error("tensor '{tensor}' data range [{start}, {end}) exceeds file length {file_len}")]
    TensorDataOutOfRange {
        tensor: String,
        start: u64,
        end: u64,
        file_len: u64,
    },

    #[error("tensor name is {len} bytes, at most {max} are supported")]
    NameTooLong { len: usize, max: usize },

    #[error("input ended before the gguf stream was fully parsed")]
    TruncatedInput,

    #[error("arithmetic overflow computing {context}")]
    Overflow { context: &'static str },

    #[error(
        "tensor '{tensor}' data is {found} bytes, expected {expected} from its dims and ggml type"
    )]
    TensorDataLengthMismatch {
        tensor: String,
        expected: u64,
        found: usize,
    },
}
