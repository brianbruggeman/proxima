//! Sans-IO safetensors reader.
//!
//! Format checked against the published spec at
//! `https://github.com/huggingface/safetensors/blob/main/README.md`
//! ("Format" section) and the reference implementation's `Dtype` enum and
//! offset-validation logic in
//! `https://raw.githubusercontent.com/huggingface/safetensors/main/safetensors/src/tensor.rs`
//! (both fetched 2026-08-18): an 8-byte little-endian `u64` header length,
//! that many bytes of UTF-8 JSON mapping tensor name to `{dtype, shape,
//! data_offsets: [BEGIN, END]}`, then the raw tensor byte buffer.
//! `data_offsets` are relative to the start of the byte buffer (the first
//! byte after the header), `END` is one-past. `__metadata__` is a
//! reserved key holding a free-form string-to-string map and is never
//! reported as a tensor.
//!
//! Sans-IO: this crate never opens a file, reads, writes, or mmaps. Callers
//! feed bytes (in any chunking) through [`SafetensorsParser`] and read back
//! [`Manifest`]/[`TensorEntry`] records — pure names, dtypes, shapes, and
//! byte ranges. No tensor byte is ever copied, dequantized, or converted
//! here. [`writer::write_complete`] is the dual: it takes an in-memory
//! [`writer::SafetensorsModel`] and returns owned bytes the caller writes
//! wherever it wants.
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(feature = "std")]
pub mod config;
mod dtype;
mod error;
mod header_codec;
mod parser;
mod pipe;
pub mod sized;
mod writer;

pub use dtype::{dtype_to_wire, map_dtype};
pub use error::SafetensorsError;
pub use header_codec::HeaderCodec;
pub use parser::{Manifest, SafetensorsParser, TensorEntry};
pub use pipe::{ParseComplete, parse_complete};
pub use sized::{HEADER_LEN_BYTES, MAX_HEADER_BYTES};
pub use writer::{SafetensorsModel, TensorPayload, WriteComplete, write_complete};

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;
    use proxima_tensor::DType;
    use rstest::rstest;

    /// Builds a real safetensors wire buffer in memory: `[u64 LE header
    /// len][header JSON][raw tensor bytes, contiguous, in declaration
    /// order]`. `entries` are `(name, dtype_wire, shape, byte_len)`;
    /// `metadata` becomes the `__metadata__` entry if non-empty.
    pub(crate) fn build_buffer(
        entries: &[(&str, &str, &[u64], usize)],
        metadata: &[(&str, &str)],
    ) -> (Vec<u8>, alloc::collections::BTreeMap<String, (u64, u64)>) {
        let mut header = alloc::string::String::from("{");
        let mut data = Vec::new();
        let mut offsets = alloc::collections::BTreeMap::new();

        if !metadata.is_empty() {
            header.push_str("\"__metadata__\":{");
            for (index, (key, val)) in metadata.iter().enumerate() {
                if index > 0 {
                    header.push(',');
                }
                header.push_str(&alloc::format!("{key:?}:{val:?}"));
            }
            header.push_str("},");
        }

        for (index, (name, dtype, shape, byte_len)) in entries.iter().enumerate() {
            if index > 0 {
                header.push(',');
            }
            let start = data.len() as u64;
            for byte_index in 0..*byte_len {
                data.push((byte_index % 251) as u8);
            }
            let end = data.len() as u64;
            offsets.insert(String::from(*name), (start, end));
            let shape_json = shape
                .iter()
                .map(alloc::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join(",");
            header.push_str(&alloc::format!(
                "{name:?}:{{\"dtype\":\"{dtype}\",\"shape\":[{shape_json}],\"data_offsets\":[{start},{end}]}}"
            ));
        }
        header.push('}');

        let header_bytes = header.into_bytes();
        let mut wire = Vec::new();
        wire.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        wire.extend_from_slice(&header_bytes);
        wire.extend_from_slice(&data);
        (wire, offsets)
    }

    fn parse_whole(buf: &[u8]) -> Result<Manifest, SafetensorsError> {
        SafetensorsParser::new().push(buf)?.into_manifest()
    }

    fn parse_in_chunks(buf: &[u8], split_points: &[usize]) -> Result<Manifest, SafetensorsError> {
        let mut parser = SafetensorsParser::new();
        let mut start = 0;
        for &point in split_points {
            parser = parser.push(&buf[start..point])?;
            start = point;
        }
        parser = parser.push(&buf[start..])?;
        parser.into_manifest()
    }

    #[test]
    fn round_trips_multiple_dtypes_and_ranks_and_excludes_metadata() {
        let entries: &[(&str, &str, &[u64], usize)] = &[
            ("scalar", "F32", &[], 4),
            ("vector", "I8", &[8], 8),
            ("matrix", "BF16", &[2, 3], 12),
            ("cube", "BOOL", &[2, 2, 2], 8),
        ];
        let metadata: &[(&str, &str)] = &[("format", "pt"), ("author", "test")];
        let (buf, expected_offsets) = build_buffer(entries, metadata);

        let manifest = parse_whole(&buf).expect("well-formed buffer parses");

        assert_eq!(manifest.tensors.len(), 4);
        assert!(manifest.tensor("__metadata__").is_none());
        assert_eq!(manifest.metadata.get("format").map(String::as_str), Some("pt"));
        assert_eq!(manifest.metadata.get("author").map(String::as_str), Some("test"));

        let scalar = manifest.tensor("scalar").expect("scalar present");
        assert_eq!(scalar.dtype, DType::Float32);
        assert_eq!(scalar.shape, Vec::<u64>::new());

        let vector = manifest.tensor("vector").expect("vector present");
        assert_eq!(vector.dtype, DType::Int8);
        assert_eq!(vector.shape, vec![8]);

        let matrix = manifest.tensor("matrix").expect("matrix present");
        assert_eq!(matrix.dtype, DType::BFloat16);
        assert_eq!(matrix.shape, vec![2, 3]);

        let cube = manifest.tensor("cube").expect("cube present");
        assert_eq!(cube.dtype, DType::Bool);
        assert_eq!(cube.shape, vec![2, 2, 2]);

        // exact byte offsets, not just "some offsets" — an off-by-the-
        // header-length bug would still pass a looser check.
        for (name, (expected_start, expected_end)) in &expected_offsets {
            let entry = manifest.tensor(name).expect("entry present");
            assert_eq!(
                entry.data_offsets,
                (*expected_start, *expected_end),
                "tensor {name} offsets"
            );
        }
    }

    #[test]
    fn chunk_boundary_splits_yield_byte_identical_results() {
        let entries: &[(&str, &str, &[u64], usize)] = &[
            ("a", "F32", &[4], 16),
            ("b", "U8", &[100], 100),
            ("c", "F16", &[3, 3], 18),
        ];
        let (buf, _offsets) = build_buffer(entries, &[("k", "v")]);
        let whole = parse_whole(&buf).expect("whole buffer parses");

        // mid-header-length: split inside the 8-byte length prefix.
        let mid_len_prefix = parse_in_chunks(&buf, &[3]).expect("splits mid length prefix");
        assert_eq!(mid_len_prefix, whole);

        // mid-JSON: split partway through the header JSON body.
        let header_len =
            u64::from_le_bytes(buf[..8].try_into().expect("8 bytes")) as usize;
        let mid_json_point = 8 + header_len / 2;
        let mid_json = parse_in_chunks(&buf, &[mid_json_point]).expect("splits mid json");
        assert_eq!(mid_json, whole);

        // mid-tensor-record: split partway through the raw tensor bytes.
        let mid_tensor_point = 8 + header_len + 10;
        let mid_tensor = parse_in_chunks(&buf, &[mid_tensor_point]).expect("splits mid tensor");
        assert_eq!(mid_tensor, whole);

        // every awkward point in the same pass, one byte at a time
        // across the whole buffer, is the strongest form of this test.
        let byte_at_a_time = parse_in_chunks(&buf, &(1..buf.len()).collect::<Vec<_>>())
            .expect("splits at every byte boundary");
        assert_eq!(byte_at_a_time, whole);
    }

    fn parse_via_generic_driver(buf: &[u8], split_points: &[usize]) -> Result<Manifest, SafetensorsError> {
        use proxima_primitives::pipe::sans_io::drive_to_completion;

        let mut chunks = Vec::new();
        let mut start = 0usize;
        for &point in split_points {
            chunks.push(&buf[start..point]);
            start = point;
        }
        chunks.push(&buf[start..]);

        let mut parser = SafetensorsParser::new();
        let mut manifest = None;
        drive_to_completion(&mut parser, chunks, |event| {
            manifest = Some(event.clone());
        })?;
        Ok(manifest.expect("finish() succeeded so poll already emitted the manifest event"))
    }

    /// Proves `proxima_primitives::pipe::sans_io::ByteStreamParser` is
    /// load-bearing for `SafetensorsParser` too: the exact same generic
    /// `drive_to_completion` function `proxima-gguf` and `proxima-onnx`
    /// use in their own `sans_io` proof tests drives this crate's
    /// `&mut self` `feed`/`poll`, at the same awkward chunk boundaries
    /// `chunk_boundary_splits_yield_byte_identical_results` exercises by
    /// hand above.
    #[test]
    fn sans_io_generic_driver_matches_hand_rolled_loop() {
        let entries: &[(&str, &str, &[u64], usize)] = &[
            ("a", "F32", &[4], 16),
            ("b", "U8", &[100], 100),
            ("c", "F16", &[3, 3], 18),
        ];
        let (buf, _offsets) = build_buffer(entries, &[("k", "v")]);
        let whole = parse_whole(&buf).expect("whole buffer parses");

        let header_len = u64::from_le_bytes(buf[..8].try_into().expect("8 bytes")) as usize;
        let split_schedules: [&[usize]; 3] = [&[3], &[8 + header_len / 2], &[8 + header_len + 10]];
        for splits in split_schedules {
            let via_driver = parse_via_generic_driver(&buf, splits).expect("splits parse via generic driver");
            assert_eq!(via_driver, whole, "splits={splits:?} diverged via generic driver");
        }

        let byte_at_a_time: Vec<usize> = (1..buf.len()).collect();
        let via_driver =
            parse_via_generic_driver(&buf, &byte_at_a_time).expect("byte-at-a-time parses via generic driver");
        assert_eq!(via_driver, whole);
    }

    #[test]
    fn truncated_length_prefix_is_a_typed_error() {
        let outcome = SafetensorsParser::new().push(&[1, 2, 3]).unwrap().finish();
        assert!(matches!(
            outcome,
            Err(SafetensorsError::TruncatedInput { .. })
        ));
    }

    #[test]
    fn truncated_header_json_is_a_typed_error() {
        let (buf, _) = build_buffer(&[("a", "F32", &[1], 4)], &[]);
        let header_len = u64::from_le_bytes(buf[..8].try_into().unwrap()) as usize;
        let cut = &buf[..8 + header_len / 2];
        let outcome = SafetensorsParser::new().push(cut).unwrap().finish();
        assert!(matches!(
            outcome,
            Err(SafetensorsError::TruncatedInput { .. })
        ));
    }

    #[test]
    fn header_length_exceeding_max_is_a_typed_error() {
        let mut wire = vec![0_u8; 8];
        wire[..8].copy_from_slice(&(MAX_HEADER_BYTES + 1).to_le_bytes());
        let outcome = SafetensorsParser::new().push(&wire).and_then(|parser| parser.finish());
        assert!(matches!(
            outcome,
            Err(SafetensorsError::HeaderTooLarge { .. })
        ));
    }

    #[test]
    fn malformed_json_is_a_typed_error() {
        let bad_json = b"not json at all";
        let mut wire = Vec::new();
        wire.extend_from_slice(&(bad_json.len() as u64).to_le_bytes());
        wire.extend_from_slice(bad_json);
        let outcome = SafetensorsParser::new().push(&wire).and_then(|parser| parser.finish());
        assert!(matches!(
            outcome,
            Err(SafetensorsError::MalformedJson { .. })
        ));
    }

    #[test]
    fn data_offsets_outside_the_byte_buffer_is_a_typed_error() {
        // header declares 100 bytes of tensor data but the file only
        // supplies 10 — the classic "trust the header, read past the
        // real buffer" bug this reader must never make.
        let json = br#"{"t":{"dtype":"F32","shape":[25],"data_offsets":[0,100]}}"#;
        let mut wire = Vec::new();
        wire.extend_from_slice(&(json.len() as u64).to_le_bytes());
        wire.extend_from_slice(json);
        wire.extend_from_slice(&[0_u8; 10]);
        let outcome = SafetensorsParser::new().push(&wire).and_then(|parser| parser.finish());
        assert!(matches!(
            outcome,
            Err(SafetensorsError::OffsetOutOfBounds { .. })
        ));
    }

    #[test]
    fn overlapping_tensors_is_a_typed_error() {
        let json =
            br#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}"#;
        let mut wire = Vec::new();
        wire.extend_from_slice(&(json.len() as u64).to_le_bytes());
        wire.extend_from_slice(json);
        wire.extend_from_slice(&[0_u8; 12]);
        let outcome = SafetensorsParser::new().push(&wire).and_then(|parser| parser.finish());
        assert!(matches!(
            outcome,
            Err(SafetensorsError::OverlappingTensors { .. })
        ));
    }

    #[test]
    fn unsupported_dtype_is_a_typed_error_not_a_guess() {
        // C64 (complex) has no DType counterpart — unlike F64, which the
        // dtype widening in a0f5f97 gave a real Float64 mapping to.
        let json = br#"{"t":{"dtype":"C64","shape":[1],"data_offsets":[0,8]}}"#;
        let mut wire = Vec::new();
        wire.extend_from_slice(&(json.len() as u64).to_le_bytes());
        wire.extend_from_slice(json);
        wire.extend_from_slice(&[0_u8; 8]);
        let outcome = SafetensorsParser::new().push(&wire).and_then(|parser| parser.finish());
        assert!(matches!(
            outcome,
            Err(SafetensorsError::UnsupportedDtype { .. })
        ));
    }

    #[rstest]
    #[case::inverted_offsets(br#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[8,0]}}"#)]
    #[case::missing_dtype(br#"{"t":{"shape":[1],"data_offsets":[0,4]}}"#)]
    #[case::missing_shape(br#"{"t":{"dtype":"F32","data_offsets":[0,4]}}"#)]
    #[case::missing_offsets(br#"{"t":{"dtype":"F32","shape":[1]}}"#)]
    #[case::header_not_object(b"[1,2,3]")]
    fn malformed_header_shapes_never_panic(#[case] json: &[u8]) {
        let mut wire = Vec::new();
        wire.extend_from_slice(&(json.len() as u64).to_le_bytes());
        wire.extend_from_slice(json);
        let outcome = SafetensorsParser::new().push(&wire).and_then(|parser| parser.finish());
        assert!(outcome.is_err(), "expected a typed error, got {outcome:?}");
    }

    #[test]
    fn empty_tensor_directory_is_a_valid_empty_manifest() {
        let (buf, _) = build_buffer(&[], &[]);
        let manifest = parse_whole(&buf).expect("empty directory is valid");
        assert!(manifest.tensors.is_empty());
        assert_eq!(manifest.declared_data_len(), 0);
    }
}
