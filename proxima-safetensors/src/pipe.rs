//! [`parse_complete`]: "I already have the whole safetensors byte buffer
//! as one contiguous slice, parse all of it." That call is stateless from
//! the caller's point of view -- no cursor to thread back in, no
//! partial-input case to report.
//!
//! [`crate::parser::SafetensorsParser`] itself is a different shape: its
//! `push` is a self-consuming `Self -> Self` state transition threaded
//! across a caller-controlled number of chunks -- see `crate::header_codec`
//! and `crate::parser` module docs for why. The FSM stays a plain
//! self-consuming type; this module is the single-call convenience built on
//! top of it for callers that already hold the whole buffer.

use crate::error::SafetensorsError;
use crate::parser::{Manifest, SafetensorsParser};

/// Parses one complete, already-assembled byte slice in a single call.
/// Stateless -- a fresh [`SafetensorsParser`] is built and driven internally
/// on every call.
///
/// # Errors
///
/// Any [`SafetensorsError`] `SafetensorsParser::push`/`finish` surfaces.
pub fn parse_complete(input: &[u8]) -> Result<Manifest, SafetensorsError> {
    SafetensorsParser::new().push(input)?.finish()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec::Vec;

    use super::*;

    fn build_minimal_buffer() -> Vec<u8> {
        let json = br#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut wire = Vec::new();
        wire.extend_from_slice(&(json.len() as u64).to_le_bytes());
        wire.extend_from_slice(json);
        wire.extend_from_slice(&[0_u8; 4]);
        wire
    }

    #[test]
    fn parse_complete_reads_a_single_tensor_manifest() {
        let wire = build_minimal_buffer();

        let manifest = parse_complete(&wire).expect("parses via free function");

        assert_eq!(manifest.tensors.len(), 1);
        assert_eq!(manifest.tensors[0].dtype, proxima_tensor::DType::Float32);
    }
}
