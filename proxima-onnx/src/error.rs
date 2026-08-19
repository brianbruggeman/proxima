//! Typed parse failures. Every malformed-input path returns one of these --
//! never a panic, never an out-of-bounds read.

use proxima_protocols::protobuf_wire::ParseError as WireError;
use thiserror::Error;

/// Everything that can go wrong parsing an ONNX (protobuf) byte stream.
#[derive(Debug, Error)]
pub enum OnnxError {
    /// A malformed wire primitive (varint overflow, deprecated/unknown wire
    /// type, a length-delimited field whose declared length runs past the
    /// buffer it lives in) surfaced by `proxima_protocols::protobuf_wire`
    /// while decoding an already-complete message slice. Never raised for
    /// "not enough bytes buffered yet" -- the byte-fed [`crate::parser::OnnxParser`]
    /// handles that as [`crate::parser::PollOutcome::NeedMore`] and only
    /// calls into a message decoder once a field's bytes are fully present.
    #[error("field {field}: {source}")]
    Wire {
        field: u32,
        #[source]
        source: WireError,
    },

    #[error(
        "field {field} declares a length-delimited payload of {declared} bytes, exceeding the sanity cap of {cap} bytes"
    )]
    DeclaredLengthTooLarge {
        field: u32,
        declared: u64,
        cap: u64,
    },

    #[error("field {field} expected wire type {expected}, found {found}")]
    WireTypeMismatch {
        field: u32,
        expected: &'static str,
        found: &'static str,
    },

    #[error("field {field} string payload is not valid utf-8")]
    InvalidUtf8 { field: u32 },

    #[error("caller reported end-of-input, but the parser was mid-field")]
    TruncatedAtFinish,
}
