//! [`parse_complete`]: "I already have the whole `ModelProto` byte slice,
//! parse all of it." That call is stateless from the caller's point of
//! view -- no cursor to thread back in, no partial-input case to report.
//!
//! [`crate::parser::OnnxParser`] itself is a different shape, for the same
//! reason `GgufParser` is: its `feed`/`poll` pair needs `&mut self` across a
//! caller-controlled number of calls with real internal state (the
//! accumulation buffer, the cursor). The FSM stays a plain `&mut self`
//! type; this module is the single-call convenience built directly on the
//! whole-slice decoder, bypassing the FSM entirely (it has no incremental
//! need to serve here).

use crate::decode::decode_model_proto;
use crate::error::OnnxError;
use crate::messages::ModelProto;

/// Parses one complete, already-assembled `ModelProto` byte slice.
///
/// # Errors
///
/// Any [`OnnxError`] [`decode_model_proto`] surfaces for malformed wire
/// content.
pub fn parse_complete(input: &[u8]) -> Result<ModelProto<'_>, OnnxError> {
    decode_model_proto(input)
}
