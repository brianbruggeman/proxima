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
use crate::parser::{Manifest, SafetensorsParser, declared_total_len};

/// Parses one complete, already-assembled byte slice in a single call.
/// Stateless -- a fresh [`SafetensorsParser`] is built and driven internally
/// on every call.
///
/// # Errors
///
/// Any [`SafetensorsError`] `SafetensorsParser::push`/`into_manifest` surfaces.
pub fn parse_complete(input: &[u8]) -> Result<Manifest, SafetensorsError> {
    let (header, tensor_data) = split_header(input);
    SafetensorsParser::new()
        .push(header)?
        .push(tensor_data)?
        .into_manifest()
}

/// Splits `input` at the end of its declared header. The parser buffers every
/// byte it is fed until the header completes, so feeding a mapped multi-GB
/// checkpoint whole would copy it onto the heap; the tensor data is fed
/// after the header so it is only counted. An input whose header is not fully
/// present is fed whole and reports the truncation itself.
fn split_header(input: &[u8]) -> (&[u8], &[u8]) {
    declared_total_len(input)
        .and_then(|end| usize::try_from(end).ok())
        .filter(|end| *end <= input.len())
        .map_or((input, &[]), |end| input.split_at(end))
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
    fn split_header_leaves_the_tensor_data_out_of_the_header_slice() {
        let wire = build_minimal_buffer();

        let (header, tensor_data) = split_header(&wire);

        assert_eq!(header.len(), wire.len() - 4);
        assert_eq!(tensor_data, &[0_u8; 4]);
    }

    #[test]
    fn split_header_keeps_a_truncated_header_whole() {
        let wire = build_minimal_buffer();
        let truncated = &wire[..10];

        let (header, tensor_data) = split_header(truncated);

        assert_eq!(header, truncated);
        assert!(tensor_data.is_empty());
        assert!(parse_complete(truncated).is_err());
    }

    #[test]
    fn parse_complete_rejects_a_tensor_whose_data_is_shorter_than_declared() {
        let wire = build_minimal_buffer();
        let short = &wire[..wire.len() - 1];

        let outcome = parse_complete(short);

        assert!(matches!(
            outcome,
            Err(SafetensorsError::OffsetOutOfBounds { .. })
        ));
    }

    #[test]
    fn parse_complete_reads_a_single_tensor_manifest() {
        let wire = build_minimal_buffer();

        let manifest = parse_complete(&wire).expect("parses via free function");

        assert_eq!(manifest.tensors.len(), 1);
        assert_eq!(manifest.tensors[0].dtype, proxima_numeric::DType::FLOAT32);
    }
}
