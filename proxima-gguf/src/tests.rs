//! Crate-level integration tests: a hand-built synthetic GGUF (multiple
//! metadata value types including an array and a string, several tensors
//! with different `ggml_type`s and dimensions, non-trivial alignment
//! padding), driven through both the single-shot [`crate::parse_complete`]
//! path and the raw [`crate::parser::GgufParser`] FSM split at arbitrary
//! chunk boundaries. Real data, not `AAAA`-style stub bytes: every value
//! encodes exactly the way `gguf.cpp`'s writer would.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use alloc::string::ToString;
use alloc::vec::Vec;

use crate::error::GgufError;
use crate::parser::{GgufEvent, GgufParser};
use crate::pipe::ParsedGguf;
use crate::tensor::TensorInfo;
use crate::types::GgmlType;
use crate::value::{MetadataArray, MetadataValue};

const RAW_TYPE_U32: u32 = 4;
const RAW_TYPE_BOOL: u32 = 7;
const RAW_TYPE_STRING: u32 = 8;
const RAW_TYPE_ARRAY: u32 = 9;
const RAW_GGML_F32: i32 = 0;
const RAW_GGML_F16: i32 = 1;
const RAW_GGML_Q4_0: i32 = 2;

fn push_string(buf: &mut Vec<u8>, value: &str) {
    buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
    buf.extend_from_slice(value.as_bytes());
}

fn push_kv_string(buf: &mut Vec<u8>, key: &str, value: &str) {
    push_string(buf, key);
    buf.extend_from_slice(&RAW_TYPE_STRING.to_le_bytes());
    push_string(buf, value);
}

fn push_kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
    push_string(buf, key);
    buf.extend_from_slice(&RAW_TYPE_U32.to_le_bytes());
    buf.extend_from_slice(&value.to_le_bytes());
}

fn push_kv_bool(buf: &mut Vec<u8>, key: &str, value: bool) {
    push_string(buf, key);
    buf.extend_from_slice(&RAW_TYPE_BOOL.to_le_bytes());
    buf.push(u8::from(value));
}

fn push_kv_array_string(buf: &mut Vec<u8>, key: &str, values: &[&str]) {
    push_string(buf, key);
    buf.extend_from_slice(&RAW_TYPE_ARRAY.to_le_bytes());
    buf.extend_from_slice(&RAW_TYPE_STRING.to_le_bytes());
    buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        push_string(buf, value);
    }
}

fn push_tensor(buf: &mut Vec<u8>, name: &str, dims: &[u64], ggml_type_raw: i32, offset: u64) {
    push_string(buf, name);
    buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for dim in dims {
        buf.extend_from_slice(&dim.to_le_bytes());
    }
    buf.extend_from_slice(&ggml_type_raw.to_le_bytes());
    buf.extend_from_slice(&offset.to_le_bytes());
}

fn align_up(value: u64, alignment: u64) -> u64 {
    let remainder = value % alignment;
    if remainder == 0 {
        value
    } else {
        value + (alignment - remainder)
    }
}

/// A synthetic GGUF fixture plus the ground-truth values it was built from,
/// so tests assert against numbers computed independently of the parser.
struct Fixture {
    bytes: Vec<u8>,
    alignment: u32,
    data_offset: u64,
    tensor_offsets: [u64; 3],
    /// Byte length of the header + KV block + tensor directory, before
    /// alignment padding — the last byte the FSM actually needs to reach
    /// `Phase::Done`. Everything past this in `bytes` is padding the parser
    /// never inspects.
    meta_len: usize,
}

/// Alignment 16 (non-default — the format default is 32, `gguf.h:46`) with
/// three tensors of three different `ggml_type`s and non-power-of-32
/// element counts, so every offset in the directory needs real padding
/// math to land on, not a coincidence of round numbers.
fn build_fixture() -> Fixture {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&3i64.to_le_bytes()); // tensor_count
    buf.extend_from_slice(&5i64.to_le_bytes()); // kv_count

    push_kv_string(&mut buf, "general.architecture", "llama");
    push_kv_u32(&mut buf, "general.alignment", 16);
    push_kv_u32(&mut buf, "llama.context_length", 4096);
    push_kv_array_string(&mut buf, "tokenizer.ggml.tokens", &["a", "b", "c"]);
    push_kv_bool(&mut buf, "general.quantized", true);

    // tensor 0: 8*4=32 elements, F32 (block=1, 4 bytes/elem) -> 128 bytes,
    // already a multiple of 16.
    let nbytes0 = 32 * 4u64;
    let offset0 = 0u64;
    push_tensor(&mut buf, "token_embd.weight", &[8, 4], RAW_GGML_F32, offset0);

    // tensor 1: 64*2=128 elements, Q4_0 (block=32 elems / 18 bytes) ->
    // 4 blocks * 18 = 72 bytes, padded to 80.
    let nbytes1_raw = (64 * 2u64 / 32) * 18;
    let offset1 = align_up(offset0 + nbytes0, 16);
    push_tensor(
        &mut buf,
        "blk.0.attn_q.weight",
        &[64, 2],
        RAW_GGML_Q4_0,
        offset1,
    );

    // tensor 2: 8 elements, F16 (block=1, 2 bytes/elem) -> 16 bytes,
    // already a multiple of 16.
    let offset2 = align_up(offset1 + align_up(nbytes1_raw, 16), 16);
    push_tensor(&mut buf, "output_norm.weight", &[8], RAW_GGML_F16, offset2);

    let meta_len = buf.len();
    let data_offset = align_up(buf.len() as u64, 16);
    while (buf.len() as u64) < data_offset {
        buf.push(0);
    }

    Fixture {
        bytes: buf,
        alignment: 16,
        data_offset,
        tensor_offsets: [offset0, offset1, offset2],
        meta_len,
    }
}

pub(crate) fn synthetic_gguf() -> Vec<u8> {
    build_fixture().bytes
}

fn assert_fixture_parsed(parsed: &ParsedGguf, fixture: &Fixture) {
    assert_eq!(parsed.version, 3);
    assert_eq!(parsed.tensor_count, 3);
    assert_eq!(parsed.kv_count, 5);
    assert_eq!(parsed.alignment, fixture.alignment);
    assert_eq!(parsed.data_offset, fixture.data_offset);

    assert_eq!(
        parsed.metadata_value("general.architecture"),
        Some(&MetadataValue::String("llama".to_string()))
    );
    assert_eq!(
        parsed.metadata_value("general.alignment"),
        Some(&MetadataValue::U32(16))
    );
    assert_eq!(
        parsed.metadata_value("llama.context_length"),
        Some(&MetadataValue::U32(4096))
    );
    assert_eq!(
        parsed.metadata_value("general.quantized"),
        Some(&MetadataValue::Bool(true))
    );
    match parsed.metadata_value("tokenizer.ggml.tokens") {
        Some(MetadataValue::Array(MetadataArray::String(tokens))) => {
            assert_eq!(tokens, &["a".to_string(), "b".to_string(), "c".to_string()]);
        }
        other => panic!("expected string array, got {other:?}"),
    }

    assert_eq!(parsed.tensors.len(), 3);
    let expected: [(&str, GgmlType, &[u64]); 3] = [
        ("token_embd.weight", GgmlType::F32, &[8, 4]),
        ("blk.0.attn_q.weight", GgmlType::Q4_0, &[64, 2]),
        ("output_norm.weight", GgmlType::F16, &[8]),
    ];
    for (index, tensor) in parsed.tensors.iter().enumerate() {
        let (name, ggml_type, dims) = expected[index];
        assert_eq!(tensor.name, name, "tensor {index} name");
        assert_eq!(tensor.ggml_type, ggml_type, "tensor {index} type");
        assert_eq!(tensor.dims.as_slice(), dims, "tensor {index} dims");
        assert_eq!(
            tensor.offset, fixture.tensor_offsets[index],
            "tensor {index} byte offset must be exact"
        );
    }
}

#[test]
fn parse_complete_round_trips_every_field() {
    let fixture = build_fixture();
    let parsed = crate::parse_complete(&fixture.bytes).expect("parse synthetic gguf");
    assert_fixture_parsed(&parsed, &fixture);
}

#[test]
fn tensor_byte_offsets_exact_with_nontrivial_alignment() {
    let fixture = build_fixture();
    // tensor 0 starts at 0 (128 bytes, already 16-aligned); tensor 1 at 128
    // (72 bytes padded to 80); tensor 2 at 208 (16 bytes) — all hand-derived
    // in `build_fixture`'s comments, independent of the parser.
    assert_eq!(fixture.tensor_offsets, [0, 128, 208]);
    // data_offset is the metadata region's own length, alignment-padded —
    // asserted against `meta_len` (the exact byte count the fixture wrote)
    // rather than a hand-counted literal, so this test can't silently drift
    // from the bytes it's actually checking.
    assert_eq!(fixture.data_offset, align_up(fixture.meta_len as u64, 16));
}

#[test]
fn tensor_info_nbytes_matches_hand_computed_footprint() {
    let fixture = build_fixture();
    let parsed = crate::parse_complete(&fixture.bytes).expect("parse synthetic gguf");
    let nbytes: Vec<u64> = parsed
        .tensors
        .iter()
        .map(|tensor| tensor.nbytes().expect("computable nbytes"))
        .collect();
    assert_eq!(nbytes, alloc::vec![128, 72, 16]);
}

fn parse_via_chunks(bytes: &[u8], chunk_size: usize) -> Result<ParsedGguf, GgufError> {
    let mut parser = GgufParser::new();
    let mut header = None;
    let mut metadata = Vec::new();
    let mut tensors = Vec::new();
    let mut completion = None;

    for chunk in bytes.chunks(chunk_size.max(1)) {
        let events;
        (parser, events) = parser.push(chunk)?;
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
                } => completion = Some((data_offset, alignment)),
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

#[test]
fn fsm_produces_identical_results_across_arbitrary_chunk_boundaries() {
    let fixture = build_fixture();
    let whole = parse_via_chunks(&fixture.bytes, fixture.bytes.len()).expect("whole-buffer parse");
    assert_fixture_parsed(&whole, &fixture);

    // 1 and 3 land mid-length-prefix and mid-string-bytes somewhere in
    // every field; 7 and 13 land mid-tensor-entry; all four are checked
    // against the same fixture assertions used for the single-shot path.
    for chunk_size in [1usize, 3, 7, 13] {
        let split = parse_via_chunks(&fixture.bytes, chunk_size)
            .unwrap_or_else(|error| panic!("chunk_size={chunk_size} failed: {error:?}"));
        assert_eq!(split, whole, "chunk_size={chunk_size} diverged from whole-buffer parse");
    }
}

#[test]
fn rejects_bad_magic_without_panicking() {
    let mut fixture = build_fixture();
    fixture.bytes[0] = b'X';
    let outcome = crate::parse_complete(&fixture.bytes);
    assert!(matches!(outcome, Err(GgufError::BadMagic { .. })));
}

#[test]
fn rejects_unsupported_version_without_panicking() {
    let mut fixture = build_fixture();
    fixture.bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
    let outcome = crate::parse_complete(&fixture.bytes);
    assert!(matches!(
        outcome,
        Err(GgufError::UnsupportedVersion { version: 99 })
    ));
}

#[test]
fn rejects_truncated_file_without_panicking() {
    let fixture = build_fixture();
    // `meta_len` is the last meaningful byte (end of the tensor directory);
    // bytes past it are pure alignment padding the FSM never even looks at,
    // so a cut has to land at or before `meta_len` to actually truncate.
    for cut in [4usize, 8, 12, 20, fixture.meta_len - 1] {
        let outcome = crate::parse_complete(&fixture.bytes[..cut]);
        assert!(
            matches!(outcome, Err(GgufError::TruncatedInput)),
            "cut={cut} should be TruncatedInput, got {outcome:?}"
        );
    }
}

#[test]
fn every_truncation_length_either_parses_or_returns_a_typed_error_never_panics() {
    let fixture = build_fixture();
    for cut in 0..=fixture.bytes.len() {
        let _ = crate::parse_complete(&fixture.bytes[..cut]);
    }
}

#[test]
fn tensor_data_range_rejects_range_past_file_end() {
    let fixture = build_fixture();
    let parsed = crate::parse_complete(&fixture.bytes).expect("parse synthetic gguf");
    let tensor: &TensorInfo = &parsed.tensors[0];
    let short_file_len = fixture.data_offset; // no payload bytes actually present
    let outcome = parsed.tensor_data_range(tensor, short_file_len);
    assert!(matches!(
        outcome,
        Err(GgufError::TensorDataOutOfRange { .. })
    ));
}

#[test]
fn tensor_data_range_accepts_range_within_a_sufficiently_long_file() {
    let fixture = build_fixture();
    let parsed = crate::parse_complete(&fixture.bytes).expect("parse synthetic gguf");
    let tensor: &TensorInfo = &parsed.tensors[0];
    let generous_file_len = fixture.data_offset + 1024;
    let range = parsed
        .tensor_data_range(tensor, generous_file_len)
        .expect("range within file bounds");
    assert_eq!(range.start, fixture.data_offset + fixture.tensor_offsets[0]);
    assert_eq!(range.end - range.start, 128);
}

#[test]
fn duplicate_metadata_key_is_rejected() {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0i64.to_le_bytes());
    buf.extend_from_slice(&2i64.to_le_bytes());
    push_kv_u32(&mut buf, "dup", 1);
    push_kv_u32(&mut buf, "dup", 2);
    let outcome = crate::parse_complete(&buf);
    assert!(matches!(outcome, Err(GgufError::DuplicateKey { .. })));
}

#[test]
fn tensor_offset_gap_is_rejected() {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&1i64.to_le_bytes());
    buf.extend_from_slice(&0i64.to_le_bytes());
    // offset should be 0 for the first tensor; write 4 to force a gap.
    push_tensor(&mut buf, "bad", &[1], RAW_GGML_F32, 4);
    let outcome = crate::parse_complete(&buf);
    assert!(matches!(
        outcome,
        Err(GgufError::TensorOffsetMismatch { .. })
    ));
}

#[test]
fn invalid_ggml_type_is_rejected() {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&1i64.to_le_bytes());
    buf.extend_from_slice(&0i64.to_le_bytes());
    push_tensor(&mut buf, "bad", &[1], 4, 0); // 4 is a retired gap value
    let outcome = crate::parse_complete(&buf);
    assert!(matches!(outcome, Err(GgufError::InvalidGgmlType { .. })));
}

#[test]
fn non_power_of_two_alignment_is_rejected() {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0i64.to_le_bytes());
    buf.extend_from_slice(&1i64.to_le_bytes());
    push_kv_u32(&mut buf, "general.alignment", 24);
    let outcome = crate::parse_complete(&buf);
    assert!(matches!(outcome, Err(GgufError::InvalidAlignment { .. })));
}

// -- Real Q4_K tensor from a host-local GGUF, not a synthetic fixture.
// Opportunistic: this model cache is specific to this machine (found via
// `find` over `~/.lmstudio`), so the test skips cleanly (not a failure)
// wherever that path is absent. Only a small prefix (metadata + tensor
// directory) and one sample slice of one tensor's packed bytes are read
// via direct `seek`+`read` — never the whole multi-GB file.
#[cfg(feature = "std")]
mod real_file {
    use std::io::{Read, Seek, SeekFrom};

    use proxima_telemetry::debug;

    use crate::pipe::parse_complete;
    use crate::quant::{q4_k, q5_k, q6_k, q8_0};
    use crate::types::GgmlType;

    const FIXTURE_PATH: &str =
        "/Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf";

    /// A Mixtral-8x7B checkpoint whose `attn_k`/`attn_v` tensors are
    /// stored as `Q8_0` (verified by inspection: `Q4_K_S` quantizes most
    /// tensors but leaves attention key/value projections at 8-bit),
    /// unlike [`FIXTURE_PATH`]'s dense 7B model, which has none.
    const Q8_0_FIXTURE_PATH: &str = "/Users/brianbruggeman/.lmstudio/models/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF/Nous-Hermes-2-Mixtral-8x7B-DPO.Q4_K_S.gguf";

    /// Sample cap in bytes, rounded down to a whole number of `Q4_K`
    /// blocks: 2 MiB gives ~14.5k blocks, ~3.7M weights -- enough for a
    /// real histogram without reading gigabytes.
    const SAMPLE_CAP_BYTES: usize = 2 * 1024 * 1024;

    #[test]
    fn dequantizes_a_real_q4_k_tensor_slice() {
        let path = std::path::Path::new(FIXTURE_PATH);
        if !path.exists() {
            eprintln!("skipping: no host-local gguf fixture at {FIXTURE_PATH}");
            return;
        }

        let mut file = std::fs::File::open(path).expect("open host-local gguf fixture");
        let file_len = file.metadata().expect("stat gguf fixture").len();

        // Grow the metadata-region read until parse_complete stops
        // reporting truncation -- the tensor directory for a ~1000-tensor
        // model fits well under a few MiB, nowhere near the payload.
        let mut header_buf = alloc::vec::Vec::new();
        let parsed = 'grow: {
            for cap in [4usize << 20, 16 << 20, 64 << 20] {
                header_buf.resize(cap, 0);
                file.seek(SeekFrom::Start(0)).expect("seek to file start");
                let read = file.read(&mut header_buf).expect("read gguf header region");
                header_buf.truncate(read);
                if let Ok(parsed) = parse_complete(&header_buf) {
                    break 'grow parsed;
                }
            }
            panic!("gguf metadata region did not fit in 64 MiB");
        };

        let tensor = parsed
            .tensors
            .iter()
            .find(|tensor| tensor.ggml_type == GgmlType::Q4_K)
            .expect("fixture model has at least one Q4_K tensor");
        let range = parsed
            .tensor_data_range(tensor, file_len)
            .expect("tensor data range within file bounds");

        let available = (range.end - range.start) as usize;
        let sample_bytes = available.min(SAMPLE_CAP_BYTES) / q4_k::BLOCK_BYTES * q4_k::BLOCK_BYTES;
        let block_count = sample_bytes / q4_k::BLOCK_BYTES;
        assert!(block_count > 0, "tensor '{}' is smaller than one q4_k block", tensor.name);

        let mut packed = alloc::vec![0u8; sample_bytes];
        file.seek(SeekFrom::Start(range.start)).expect("seek to tensor data");
        file.read_exact(&mut packed).expect("read sampled tensor bytes");

        let element_count = q4_k::elements_for_blocks(block_count);
        let mut weights = alloc::vec![0.0f32; element_count];
        q4_k::dequantize(&packed, &mut weights).expect("dequantize sampled q4_k blocks");

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut buckets = [0u32; 10];
        for &value in &weights {
            assert!(value.is_finite(), "dequantized weight must be finite, got {value}");
            min = min.min(value);
            max = max.max(value);
            sum += f64::from(value);
            let bucket_index = (((value + 1.0) / 2.0 * 10.0) as i32).clamp(0, 9) as usize;
            buckets[bucket_index] += 1;
        }
        let mean = sum / element_count as f64;
        let variance = weights.iter().map(|value| (f64::from(*value) - mean).powi(2)).sum::<f64>() / element_count as f64;

        debug!(
            tensor = %tensor.name,
            blocks = block_count as u64,
            elements = element_count as u64,
            min,
            max,
            mean,
            stddev = variance.sqrt(),
            ?buckets,
            "quant.q4_k real tensor slice sample stats over [-1.0, 1.0] in 10 buckets"
        );
    }

    /// Sample cap for the `Q8_0` tensor slice, rounded down to a whole
    /// number of `Q8_0` blocks: 2 MiB gives ~64.7k blocks, ~2.07M
    /// weights.
    const Q8_0_SAMPLE_CAP_BYTES: usize = 2 * 1024 * 1024;

    #[test]
    fn dequantizes_a_real_q8_0_tensor_slice() {
        let path = std::path::Path::new(Q8_0_FIXTURE_PATH);
        if !path.exists() {
            eprintln!("skipping: no host-local gguf fixture at {Q8_0_FIXTURE_PATH}");
            return;
        }

        let mut file = std::fs::File::open(path).expect("open host-local gguf fixture");
        let file_len = file.metadata().expect("stat gguf fixture").len();

        // Same growing-window strategy as the q4_K test above: the
        // metadata region for a MoE model with many experts is larger
        // than a dense model's, so start from a bigger floor.
        let mut header_buf = alloc::vec::Vec::new();
        let parsed = 'grow: {
            for cap in [16usize << 20, 64 << 20, 128 << 20] {
                header_buf.resize(cap, 0);
                file.seek(SeekFrom::Start(0)).expect("seek to file start");
                let read = file.read(&mut header_buf).expect("read gguf header region");
                header_buf.truncate(read);
                if let Ok(parsed) = parse_complete(&header_buf) {
                    break 'grow parsed;
                }
            }
            panic!("gguf metadata region did not fit in 128 MiB");
        };

        // `attn_k`/`attn_v` are the tensors this model stores as `Q8_0`
        // (the gap this codec closes -- `Q4_K_S` quantizes most weights
        // to 4 bits but leaves attention key/value projections at 8);
        // picking one by name (rather than the first `Q8_0` tensor found)
        // documents which weights this sample actually is.
        let tensor = parsed
            .tensors
            .iter()
            .find(|tensor| tensor.ggml_type == GgmlType::Q8_0 && tensor.name.contains("attn_k"))
            .or_else(|| parsed.tensors.iter().find(|tensor| tensor.ggml_type == GgmlType::Q8_0))
            .expect("fixture model has at least one Q8_0 tensor");
        let range = parsed
            .tensor_data_range(tensor, file_len)
            .expect("tensor data range within file bounds");

        let available = (range.end - range.start) as usize;
        let sample_bytes = available.min(Q8_0_SAMPLE_CAP_BYTES) / q8_0::BLOCK_BYTES * q8_0::BLOCK_BYTES;
        let block_count = sample_bytes / q8_0::BLOCK_BYTES;
        assert!(block_count > 0, "tensor '{}' is smaller than one q8_0 block", tensor.name);

        let mut packed = alloc::vec![0u8; sample_bytes];
        file.seek(SeekFrom::Start(range.start)).expect("seek to tensor data");
        file.read_exact(&mut packed).expect("read sampled tensor bytes");

        let element_count = q8_0::elements_for_blocks(block_count);
        let mut weights = alloc::vec![0.0f32; element_count];
        q8_0::dequantize(&packed, &mut weights).expect("dequantize sampled q8_0 blocks");

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut buckets = [0u32; 10];
        for &value in &weights {
            assert!(value.is_finite(), "dequantized weight must be finite, got {value}");
            min = min.min(value);
            max = max.max(value);
            sum += f64::from(value);
            let bucket_index = (((value + 1.0) / 2.0 * 10.0) as i32).clamp(0, 9) as usize;
            buckets[bucket_index] += 1;
        }
        let mean = sum / element_count as f64;
        let variance = weights.iter().map(|value| (f64::from(*value) - mean).powi(2)).sum::<f64>() / element_count as f64;

        debug!(
            tensor = %tensor.name,
            blocks = block_count as u64,
            elements = element_count as u64,
            min,
            max,
            mean,
            stddev = variance.sqrt(),
            ?buckets,
            "quant.q8_0 real tensor slice sample stats over [-1.0, 1.0] in 10 buckets"
        );
    }

    /// Sample cap for the `Q6_K` tensor slice, rounded down to a whole
    /// number of `Q6_K` blocks: 2 MiB gives ~9979 blocks, ~2.55M weights.
    const Q6_K_SAMPLE_CAP_BYTES: usize = 2 * 1024 * 1024;

    #[test]
    fn dequantizes_a_real_q6_k_tensor_slice() {
        let path = std::path::Path::new(Q8_0_FIXTURE_PATH);
        if !path.exists() {
            eprintln!("skipping: no host-local gguf fixture at {Q8_0_FIXTURE_PATH}");
            return;
        }

        let mut file = std::fs::File::open(path).expect("open host-local gguf fixture");
        let file_len = file.metadata().expect("stat gguf fixture").len();

        let mut header_buf = alloc::vec::Vec::new();
        let parsed = 'grow: {
            for cap in [16usize << 20, 64 << 20, 128 << 20] {
                header_buf.resize(cap, 0);
                file.seek(SeekFrom::Start(0)).expect("seek to file start");
                let read = file.read(&mut header_buf).expect("read gguf header region");
                header_buf.truncate(read);
                if let Ok(parsed) = parse_complete(&header_buf) {
                    break 'grow parsed;
                }
            }
            panic!("gguf metadata region did not fit in 128 MiB");
        };

        // This `Q4_K_S` checkpoint stores exactly one tensor as `Q6_K`
        // (the gap this codec closes second, after `Q8_0`) -- pick
        // whichever one it is by type, there being only the one.
        let tensor = parsed
            .tensors
            .iter()
            .find(|tensor| tensor.ggml_type == GgmlType::Q6_K)
            .expect("fixture model has at least one Q6_K tensor");
        let range = parsed
            .tensor_data_range(tensor, file_len)
            .expect("tensor data range within file bounds");

        let available = (range.end - range.start) as usize;
        let sample_bytes = available.min(Q6_K_SAMPLE_CAP_BYTES) / q6_k::BLOCK_BYTES * q6_k::BLOCK_BYTES;
        let block_count = sample_bytes / q6_k::BLOCK_BYTES;
        assert!(block_count > 0, "tensor '{}' is smaller than one q6_k block", tensor.name);

        let mut packed = alloc::vec![0u8; sample_bytes];
        file.seek(SeekFrom::Start(range.start)).expect("seek to tensor data");
        file.read_exact(&mut packed).expect("read sampled tensor bytes");

        let element_count = q6_k::elements_for_blocks(block_count);
        let mut weights = alloc::vec![0.0f32; element_count];
        q6_k::dequantize(&packed, &mut weights).expect("dequantize sampled q6_k blocks");

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut buckets = [0u32; 10];
        for &value in &weights {
            assert!(value.is_finite(), "dequantized weight must be finite, got {value}");
            min = min.min(value);
            max = max.max(value);
            sum += f64::from(value);
            let bucket_index = (((value + 1.0) / 2.0 * 10.0) as i32).clamp(0, 9) as usize;
            buckets[bucket_index] += 1;
        }
        let mean = sum / element_count as f64;
        let variance = weights.iter().map(|value| (f64::from(*value) - mean).powi(2)).sum::<f64>() / element_count as f64;

        debug!(
            tensor = %tensor.name,
            blocks = block_count as u64,
            elements = element_count as u64,
            min,
            max,
            mean,
            stddev = variance.sqrt(),
            ?buckets,
            "quant.q6_k real tensor slice sample stats over [-1.0, 1.0] in 10 buckets"
        );
    }

    /// Sample cap for the `Q5_K` tensor slice, rounded down to a whole
    /// number of `Q5_K` blocks: 2 MiB gives ~11.9k blocks, ~3.05M weights.
    const Q5_K_SAMPLE_CAP_BYTES: usize = 2 * 1024 * 1024;

    #[test]
    fn dequantizes_a_real_q5_k_tensor_slice() {
        let path = std::path::Path::new(Q8_0_FIXTURE_PATH);
        if !path.exists() {
            eprintln!("skipping: no host-local gguf fixture at {Q8_0_FIXTURE_PATH}");
            return;
        }

        let mut file = std::fs::File::open(path).expect("open host-local gguf fixture");
        let file_len = file.metadata().expect("stat gguf fixture").len();

        let mut header_buf = alloc::vec::Vec::new();
        let parsed = 'grow: {
            for cap in [16usize << 20, 64 << 20, 128 << 20] {
                header_buf.resize(cap, 0);
                file.seek(SeekFrom::Start(0)).expect("seek to file start");
                let read = file.read(&mut header_buf).expect("read gguf header region");
                header_buf.truncate(read);
                if let Ok(parsed) = parse_complete(&header_buf) {
                    break 'grow parsed;
                }
            }
            panic!("gguf metadata region did not fit in 128 MiB");
        };

        // This `Q4_K_S` checkpoint stores 64 tensors as `Q5_K` -- the gap
        // this codec closes last, after `Q8_0` and `Q6_K` -- pick the
        // first one found.
        let tensor = parsed
            .tensors
            .iter()
            .find(|tensor| tensor.ggml_type == GgmlType::Q5_K)
            .expect("fixture model has at least one Q5_K tensor");
        let range = parsed
            .tensor_data_range(tensor, file_len)
            .expect("tensor data range within file bounds");

        let available = (range.end - range.start) as usize;
        let sample_bytes = available.min(Q5_K_SAMPLE_CAP_BYTES) / q5_k::BLOCK_BYTES * q5_k::BLOCK_BYTES;
        let block_count = sample_bytes / q5_k::BLOCK_BYTES;
        assert!(block_count > 0, "tensor '{}' is smaller than one q5_k block", tensor.name);

        let mut packed = alloc::vec![0u8; sample_bytes];
        file.seek(SeekFrom::Start(range.start)).expect("seek to tensor data");
        file.read_exact(&mut packed).expect("read sampled tensor bytes");

        let element_count = q5_k::elements_for_blocks(block_count);
        let mut weights = alloc::vec![0.0f32; element_count];
        q5_k::dequantize(&packed, &mut weights).expect("dequantize sampled q5_k blocks");

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut buckets = [0u32; 10];
        for &value in &weights {
            assert!(value.is_finite(), "dequantized weight must be finite, got {value}");
            min = min.min(value);
            max = max.max(value);
            sum += f64::from(value);
            let bucket_index = (((value + 1.0) / 2.0 * 10.0) as i32).clamp(0, 9) as usize;
            buckets[bucket_index] += 1;
        }
        let mean = sum / element_count as f64;
        let variance = weights.iter().map(|value| (f64::from(*value) - mean).powi(2)).sum::<f64>() / element_count as f64;

        debug!(
            tensor = %tensor.name,
            blocks = block_count as u64,
            elements = element_count as u64,
            min,
            max,
            mean,
            stddev = variance.sqrt(),
            ?buckets,
            "quant.q5_k real tensor slice sample stats over [-1.0, 1.0] in 10 buckets"
        );
    }

    /// The payoff this whole codec line of work has been building toward:
    /// every one of the Mixtral checkpoint's tensors either has a landed
    /// codec ([`q4_k`], [`q5_k`], [`q6_k`], [`q8_0`]) or is a native
    /// (non-block) type this crate already reads directly (`F32`/`F16`).
    /// Walks the full tensor directory -- not a sample -- since only the
    /// metadata region (a few MiB) is read here, never tensor payload
    /// bytes. Reports the type histogram and the total so a future format
    /// gap shows up as a named, counted `GgmlType`, not a silent skip.
    #[test]
    fn every_mixtral_tensor_has_a_codec_or_is_native() {
        let path = std::path::Path::new(Q8_0_FIXTURE_PATH);
        if !path.exists() {
            eprintln!("skipping: no host-local gguf fixture at {Q8_0_FIXTURE_PATH}");
            return;
        }

        let mut file = std::fs::File::open(path).expect("open host-local gguf fixture");

        let mut header_buf = alloc::vec::Vec::new();
        let parsed = 'grow: {
            for cap in [16usize << 20, 64 << 20, 128 << 20] {
                header_buf.resize(cap, 0);
                file.seek(SeekFrom::Start(0)).expect("seek to file start");
                let read = file.read(&mut header_buf).expect("read gguf header region");
                header_buf.truncate(read);
                if let Ok(parsed) = parse_complete(&header_buf) {
                    break 'grow parsed;
                }
            }
            panic!("gguf metadata region did not fit in 128 MiB");
        };

        let mut histogram: alloc::vec::Vec<(GgmlType, u32)> = alloc::vec::Vec::new();
        let mut uncovered: alloc::vec::Vec<(alloc::string::String, GgmlType)> = alloc::vec::Vec::new();

        for tensor in &parsed.tensors {
            let ggml_type = tensor.ggml_type;
            match histogram.iter_mut().find(|(kind, _)| *kind == ggml_type) {
                Some((_, count)) => *count += 1,
                None => histogram.push((ggml_type, 1)),
            }

            let covered = matches!(
                ggml_type,
                GgmlType::F32 | GgmlType::F16 | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K | GgmlType::Q8_0
            );
            if !covered {
                uncovered.push((tensor.name.clone(), ggml_type));
            }
        }

        let total: u32 = histogram.iter().map(|(_, count)| *count).sum();
        debug!(total, ?histogram, "gguf mixtral tensor codec coverage");
        assert_eq!(total as usize, parsed.tensors.len(), "histogram must account for every tensor exactly once");
        assert!(
            uncovered.is_empty(),
            "tensors with no codec and not natively f32/f16: {uncovered:?}"
        );
    }

    // -- PART 2 policy evidence: error-vs-bytes curve per tensor role,
    // measured from real weights. Each sample dequantizes a real tensor's
    // packed bytes (never a synthetic one) to `f32` -- that dequantized
    // slice is the reference -- then re-quantizes that reference at every
    // block level this crate supports and dequantizes each back, reporting
    // max and RMS error plus packed byte size per level. Caveat, stated
    // plainly: for `Q4_K`-sourced samples the reference is itself already
    // `Q4_K`-quantized data (llama.cpp never ships an unquantized copy
    // alongside it), so this measures *re*-quantization error on top of
    // the original quantization loss, not error against the true original
    // f16/f32 weights. It still ranks the four levels correctly relative
    // to each other, which is what a policy needs. For the `output.weight`
    // sample the source is `Q6_K` (the only type llama.cpp ever stores
    // that role as), so the same caveat applies at one level higher.

    struct CurveLevel {
        level: GgmlType,
        bytes: usize,
        max_error: f32,
        rms_error: f64,
    }

    fn measure(level: GgmlType, bytes: usize, reference: &[f32], roundtrip: &[f32]) -> CurveLevel {
        let mut max_error = 0.0f32;
        let mut sum_sq_error = 0.0f64;
        for (want, got) in reference.iter().zip(roundtrip.iter()) {
            assert!(got.is_finite(), "requantized value must be finite, got {got}");
            let diff = (got - want).abs();
            max_error = max_error.max(diff);
            sum_sq_error += f64::from(diff) * f64::from(diff);
        }
        let rms_error = (sum_sq_error / reference.len() as f64).sqrt();
        CurveLevel {
            level,
            bytes,
            max_error,
            rms_error,
        }
    }

    /// Re-quantizes `reference` (truncated to a multiple of 256 elements,
    /// the largest super-block size in play) at `Q4_K`/`Q5_K`/`Q6_K`/`Q8_0`
    /// and measures round-trip error plus packed size at each level.
    fn requantize_curve(reference: &[f32]) -> alloc::vec::Vec<CurveLevel> {
        let usable = reference.len() / 256 * 256;
        let reference = &reference[..usable];
        let mut results = alloc::vec::Vec::new();

        let q4_blocks = usable / q4_k::QK_K;
        let mut q4_packed = alloc::vec![0u8; q4_k::bytes_for_blocks(q4_blocks)];
        q4_k::quantize(reference, &mut q4_packed).expect("quantize q4_k");
        let mut q4_roundtrip = alloc::vec![0.0f32; usable];
        q4_k::dequantize(&q4_packed, &mut q4_roundtrip).expect("dequantize q4_k");
        results.push(measure(GgmlType::Q4_K, q4_packed.len(), reference, &q4_roundtrip));

        let q5_blocks = usable / q5_k::QK_K;
        let mut q5_packed = alloc::vec![0u8; q5_k::bytes_for_blocks(q5_blocks)];
        q5_k::quantize(reference, &mut q5_packed).expect("quantize q5_k");
        let mut q5_roundtrip = alloc::vec![0.0f32; usable];
        q5_k::dequantize(&q5_packed, &mut q5_roundtrip).expect("dequantize q5_k");
        results.push(measure(GgmlType::Q5_K, q5_packed.len(), reference, &q5_roundtrip));

        let q6_blocks = usable / q6_k::QK_K;
        let mut q6_packed = alloc::vec![0u8; q6_k::bytes_for_blocks(q6_blocks)];
        q6_k::quantize(reference, &mut q6_packed).expect("quantize q6_k");
        let mut q6_roundtrip = alloc::vec![0.0f32; usable];
        q6_k::dequantize(&q6_packed, &mut q6_roundtrip).expect("dequantize q6_k");
        results.push(measure(GgmlType::Q6_K, q6_packed.len(), reference, &q6_roundtrip));

        let q8_blocks = usable / q8_0::QK8_0;
        let mut q8_packed = alloc::vec![0u8; q8_0::bytes_for_blocks(q8_blocks)];
        q8_0::quantize(reference, &mut q8_packed).expect("quantize q8_0");
        let mut q8_roundtrip = alloc::vec![0.0f32; usable];
        q8_0::dequantize(&q8_packed, &mut q8_roundtrip).expect("dequantize q8_0");
        results.push(measure(GgmlType::Q8_0, q8_packed.len(), reference, &q8_roundtrip));

        results
    }

    /// Reads and dequantizes a bounded sample (never the whole tensor) of
    /// one real tensor, whatever block type it's actually stored as
    /// (`Q4_K` or `Q6_K` -- the two source types this curve draws real
    /// samples from before re-quantizing them further).
    fn dequantize_real_sample(
        file: &mut std::fs::File,
        parsed: &crate::pipe::ParsedGguf,
        file_len: u64,
        tensor: &crate::tensor::TensorInfo,
        cap_bytes: usize,
    ) -> alloc::vec::Vec<f32> {
        let range = parsed.tensor_data_range(tensor, file_len).expect("tensor data range within file bounds");
        let available = (range.end - range.start) as usize;
        match tensor.ggml_type {
            GgmlType::Q4_K => {
                let sample_bytes = available.min(cap_bytes) / q4_k::BLOCK_BYTES * q4_k::BLOCK_BYTES;
                let mut packed = alloc::vec![0u8; sample_bytes];
                file.seek(SeekFrom::Start(range.start)).expect("seek to tensor data");
                file.read_exact(&mut packed).expect("read sampled tensor bytes");
                let elements = q4_k::elements_for_blocks(sample_bytes / q4_k::BLOCK_BYTES);
                let mut out = alloc::vec![0.0f32; elements];
                q4_k::dequantize(&packed, &mut out).expect("dequantize sampled q4_k source");
                out
            }
            GgmlType::Q6_K => {
                let sample_bytes = available.min(cap_bytes) / q6_k::BLOCK_BYTES * q6_k::BLOCK_BYTES;
                let mut packed = alloc::vec![0u8; sample_bytes];
                file.seek(SeekFrom::Start(range.start)).expect("seek to tensor data");
                file.read_exact(&mut packed).expect("read sampled tensor bytes");
                let elements = q6_k::elements_for_blocks(sample_bytes / q6_k::BLOCK_BYTES);
                let mut out = alloc::vec![0.0f32; elements];
                q6_k::dequantize(&packed, &mut out).expect("dequantize sampled q6_k source");
                out
            }
            other => panic!("dequantize_real_sample: unsupported source type {other:?} for '{}'", tensor.name),
        }
    }

    fn find_tensor<'parsed>(parsed: &'parsed crate::pipe::ParsedGguf, name: &str) -> &'parsed crate::tensor::TensorInfo {
        parsed
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap_or_else(|| panic!("no tensor named '{name}'"))
    }

    /// PART 2's deliverable: for a sample of real tensors spanning
    /// `token_embd`, `attn_q`/`attn_k`/`attn_v`/`attn_output`,
    /// `ffn_gate`/`ffn_up`/`ffn_down`, and `output.weight`, measures the
    /// error-vs-bytes curve across all four supported block levels and
    /// reports it -- including any role where the extra bytes from a
    /// higher level buy almost nothing, which is exactly the signal a
    /// policy should be built on. Samples are drawn from layer 8 (never
    /// layer 0) so `ffn_down`/`attn_v`'s early-layer bump
    /// (`quant::policy`'s real-file coverage tests) doesn't contaminate
    /// the "plain `Q4_K`" sample for those two roles.
    #[test]
    fn role_precision_error_vs_bytes_curve_from_real_tensors() {
        let mixtral_path = std::path::Path::new(Q8_0_FIXTURE_PATH);
        let openchat_path = std::path::Path::new(FIXTURE_PATH);
        if !mixtral_path.exists() || !openchat_path.exists() {
            eprintln!("skipping: host-local gguf fixtures not both present");
            return;
        }

        let mut mixtral_file = std::fs::File::open(mixtral_path).expect("open mixtral fixture");
        let mixtral_len = mixtral_file.metadata().expect("stat mixtral fixture").len();
        let mut mixtral_header = alloc::vec::Vec::new();
        let mixtral_parsed = 'grow: {
            for cap in [16usize << 20, 64 << 20, 128 << 20] {
                mixtral_header.resize(cap, 0);
                mixtral_file.seek(SeekFrom::Start(0)).expect("seek");
                let read = mixtral_file.read(&mut mixtral_header).expect("read");
                mixtral_header.truncate(read);
                if let Ok(parsed) = parse_complete(&mixtral_header) {
                    break 'grow parsed;
                }
            }
            panic!("mixtral metadata region did not fit");
        };

        let mut openchat_file = std::fs::File::open(openchat_path).expect("open openchat fixture");
        let openchat_len = openchat_file.metadata().expect("stat openchat fixture").len();
        let mut openchat_header = alloc::vec::Vec::new();
        let openchat_parsed = 'grow: {
            for cap in [4usize << 20, 16 << 20, 64 << 20] {
                openchat_header.resize(cap, 0);
                openchat_file.seek(SeekFrom::Start(0)).expect("seek");
                let read = openchat_file.read(&mut openchat_header).expect("read");
                openchat_header.truncate(read);
                if let Ok(parsed) = parse_complete(&openchat_header) {
                    break 'grow parsed;
                }
            }
            panic!("openchat metadata region did not fit");
        };

        const CAP: usize = 2 * 1024 * 1024;
        let mut report: alloc::vec::Vec<(&str, alloc::vec::Vec<CurveLevel>)> = alloc::vec::Vec::new();

        let openchat_roles: [(&str, &str); 5] = [
            ("token_embd", "token_embd.weight"),
            ("attn_q", "blk.8.attn_q.weight"),
            ("attn_k", "blk.8.attn_k.weight"),
            ("attn_v", "blk.8.attn_v.weight"),
            ("attn_output", "blk.8.attn_output.weight"),
        ];
        for (role, tensor_name) in openchat_roles {
            let tensor = find_tensor(&openchat_parsed, tensor_name);
            let reference = dequantize_real_sample(&mut openchat_file, &openchat_parsed, openchat_len, tensor, CAP);
            report.push((role, requantize_curve(&reference)));
        }

        let mixtral_roles: [(&str, &str); 4] = [
            ("ffn_gate", "blk.8.ffn_gate.0.weight"),
            ("ffn_up", "blk.8.ffn_up.0.weight"),
            ("ffn_down", "blk.8.ffn_down.0.weight"),
            ("output_weight", "output.weight"),
        ];
        for (role, tensor_name) in mixtral_roles {
            let tensor = find_tensor(&mixtral_parsed, tensor_name);
            let reference = dequantize_real_sample(&mut mixtral_file, &mixtral_parsed, mixtral_len, tensor, CAP);
            report.push((role, requantize_curve(&reference)));
        }

        assert_eq!(report.len(), 9, "expected exactly the 9 sampled roles");

        for (role, curve) in &report {
            assert_eq!(curve.len(), 4, "role {role}: expected all four levels measured");
            for level in curve {
                debug!(
                    role = %role,
                    level = ?level.level,
                    bytes = level.bytes as u64,
                    max_error = level.max_error,
                    rms_error = level.rms_error,
                    "gguf real-tensor precision-vs-bytes curve point"
                );
            }
            let q4 = &curve[0];
            let q8 = &curve[3];
            let rms_gain = q4.rms_error - q8.rms_error;
            let byte_cost = q8.bytes as i64 - q4.bytes as i64;
            debug!(
                role = %role,
                rms_gain,
                byte_cost,
                "gguf real-tensor precision curve q4_k to q8_0 summary"
            );
        }
    }
}

// -- Metadata + tensor-directory survey across three host-local GGUF files
// spanning dense (Mistral-7B, DeepSeek-Coder-33B) and MoE (Mixtral-8x7B)
// architectures. Only the header region (KV block + tensor directory) is
// ever read -- never the multi-GB tensor payload. `#[ignore]`d: this is a
// point-in-time report meant to be driven by hand
// (`cargo test -p proxima-gguf --features std reports_real_models -- --ignored --nocapture`),
// not part of the standard gate, and (like `real_file` above) opportunistic
// on a host-local model cache.
#[cfg(feature = "std")]
mod real_models {
    use alloc::collections::BTreeMap;
    use alloc::format;
    use alloc::string::String;
    use alloc::vec::Vec;
    use std::io::{Read, Seek, SeekFrom};
    use std::path::Path;

    use crate::pipe::{ParsedGguf, parse_complete};
    use crate::tensor::TensorInfo;
    use crate::types::GgmlType;
    use crate::value::{MetadataArray, MetadataValue};

    const MODEL_PATHS: [&str; 4] = [
        "/Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf",
        "/Users/brianbruggeman/.lmstudio/models/TheBloke/deepseek-coder-33B-instruct-GGUF/deepseek-coder-33b-instruct.Q4_K_S.gguf",
        "/Users/brianbruggeman/.lmstudio/models/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF/Nous-Hermes-2-Mixtral-8x7B-DPO.Q4_K_S.gguf",
        // LFM2.5-8B-A1B: hybrid conv/attention MoE (`Lfm2MoeForCausalLM`) --
        // 18 of 24 layers are short-convolution, only 6 are attention, and
        // the FFN is fine-grained MoE (top-4 of 32) on all but the first 2
        // dense layers. The visible contrast case for `GROUP_PATTERNS`
        // (built from dense/Mixtral names) and for `print_expert_layout`
        // (Mixtral's stacked `ffn_gate_exps.weight` vs whatever this ships).
        "/Users/brianbruggeman/.lmstudio/models/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf",
    ];

    /// Growth ladder for the header-region read. A ~1000-tensor model's KV
    /// block + tensor directory fits well under this even with a 32k-token
    /// vocabulary array inline -- nowhere near the multi-GB tensor payload.
    const HEADER_GROW_CAPS: [usize; 4] = [4 << 20, 16 << 20, 64 << 20, 128 << 20];

    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

    fn parse_header(path: &Path) -> (u64, ParsedGguf) {
        let mut file = std::fs::File::open(path).expect("open host-local gguf fixture");
        let file_len = file.metadata().expect("stat gguf fixture").len();
        let mut header_buf = Vec::new();
        for cap in HEADER_GROW_CAPS {
            header_buf.resize(cap, 0);
            file.seek(SeekFrom::Start(0)).expect("seek to file start");
            let read = file.read(&mut header_buf).expect("read gguf header region");
            header_buf.truncate(read);
            if let Ok(parsed) = parse_complete(&header_buf) {
                return (file_len, parsed);
            }
        }
        panic!(
            "gguf metadata region for {} did not fit in {} MiB",
            path.display(),
            HEADER_GROW_CAPS[HEADER_GROW_CAPS.len() - 1] >> 20
        );
    }

    fn describe_array(array: &MetadataArray) -> String {
        let len = array.len();
        match array {
            MetadataArray::String(values) => format!("string[{len}] sample={:?}", &values[..values.len().min(3)]),
            MetadataArray::F32(values) => format!("f32[{len}] sample={:?}", &values[..values.len().min(3)]),
            MetadataArray::I32(values) => format!("i32[{len}] sample={:?}", &values[..values.len().min(3)]),
            MetadataArray::U32(values) => format!("u32[{len}] sample={:?}", &values[..values.len().min(3)]),
            MetadataArray::I64(values) => format!("i64[{len}] sample={:?}", &values[..values.len().min(3)]),
            MetadataArray::U64(values) => format!("u64[{len}] sample={:?}", &values[..values.len().min(3)]),
            MetadataArray::F64(values) => format!("f64[{len}] sample={:?}", &values[..values.len().min(3)]),
            MetadataArray::Bool(values) => format!("bool[{len}] sample={:?}", &values[..values.len().min(3)]),
            MetadataArray::U8(values) => format!("u8[{len}] sample={:?}", &values[..values.len().min(3)]),
            MetadataArray::I8(values) => format!("i8[{len}] sample={:?}", &values[..values.len().min(3)]),
            MetadataArray::U16(values) => format!("u16[{len}] sample={:?}", &values[..values.len().min(3)]),
            MetadataArray::I16(values) => format!("i16[{len}] sample={:?}", &values[..values.len().min(3)]),
        }
    }

    fn describe_value(value: &MetadataValue) -> String {
        match value {
            MetadataValue::Array(array) => describe_array(array),
            other => format!("{other:?}"),
        }
    }

    fn print_all_metadata(parsed: &ParsedGguf) {
        println!("-- metadata ({} keys) --", parsed.metadata.len());
        for (key, value) in &parsed.metadata {
            println!("  {key} = {}", describe_value(value));
        }
    }

    fn find_by_suffix<'a>(parsed: &'a ParsedGguf, suffix: &str) -> Option<(&'a str, &'a MetadataValue)> {
        parsed
            .metadata
            .iter()
            .find(|(key, _)| key.ends_with(suffix))
            .map(|(key, value)| (key.as_str(), value))
    }

    /// Every field the caller asked about, found by arch-agnostic key
    /// *suffix* rather than an assumed `llama.*` prefix -- a non-llama
    /// architecture string would otherwise silently miss every lookup.
    fn print_highlights(parsed: &ParsedGguf) {
        println!("-- highlights (found by key suffix, architecture-agnostic) --");
        let suffixes = [
            "architecture",
            "block_count",
            "context_length",
            "embedding_length",
            "attention.head_count",
            "attention.head_count_kv",
            "feed_forward_length",
            "rope.dimension_count",
            "rope.freq_base",
            "expert_count",
            "expert_used_count",
        ];
        for suffix in suffixes {
            match find_by_suffix(parsed, suffix) {
                Some((key, value)) => println!("  {key} = {}", describe_value(value)),
                None => println!("  <no key ends with '{suffix}'>"),
            }
        }
        if let Some((key, MetadataValue::Array(MetadataArray::String(tokens)))) = find_by_suffix(parsed, "tokenizer.ggml.tokens") {
            println!("  {key}.len() [vocab_size] = {}", tokens.len());
        }
    }

    fn print_tensor_summary(parsed: &ParsedGguf) {
        println!("-- tensors ({} total) --", parsed.tensors.len());
        let mut histogram: BTreeMap<i32, (GgmlType, u64, u128)> = BTreeMap::new();
        let mut total_bytes: u128 = 0;
        for tensor in &parsed.tensors {
            let nbytes = u128::from(tensor.nbytes().unwrap_or(0));
            total_bytes += nbytes;
            let entry = histogram.entry(tensor.ggml_type.to_wire()).or_insert((tensor.ggml_type, 0, 0));
            entry.1 += 1;
            entry.2 += nbytes;
        }
        for (ggml_type, count, bytes) in histogram.into_values() {
            println!("  {ggml_type:?}: count={count} bytes={bytes} ({:.3} GiB)", bytes as f64 / GIB);
        }
        println!("  total packed tensor bytes (on disk) = {total_bytes} ({:.3} GiB)", total_bytes as f64 / GIB);
    }

    const GROUP_PATTERNS: [(&str, &str); 10] = [
        ("embedding", "token_embd"),
        ("attn_q", "attn_q"),
        ("attn_k", "attn_k"),
        ("attn_v", "attn_v"),
        ("attn_output", "attn_output"),
        ("ffn_gate", "ffn_gate"),
        ("ffn_up", "ffn_up"),
        ("ffn_down", "ffn_down"),
        ("norm", "norm"),
        ("output", "output.weight"),
    ];

    fn print_functional_groups(parsed: &ParsedGguf) {
        println!("-- functional groups (sample names) --");
        for (label, pattern) in GROUP_PATTERNS {
            let matches: Vec<&TensorInfo> = parsed.tensors.iter().filter(|tensor| tensor.name.contains(pattern)).collect();
            println!("  {label} (pattern='{pattern}', {} matches):", matches.len());
            for tensor in matches.iter().take(3) {
                println!("    {} dims={:?} type={:?}", tensor.name, tensor.dims.as_slice(), tensor.ggml_type);
            }
        }
    }

    /// The load-bearing MoE question: are `blk.0`'s expert FFN tensors
    /// separate per-expert entries (`blk.0.ffn_gate.0.weight` ..
    /// `blk.0.ffn_gate.7.weight`) or one stacked 3D tensor
    /// (`blk.0.ffn_gate_exps.weight`)? Printed for every model, not just
    /// Mixtral, so the dense models are the visible contrast case.
    fn print_expert_layout(parsed: &ParsedGguf) {
        println!("-- blk.0 ffn_gate* tensors (expert layout) --");
        let mut matches: Vec<&TensorInfo> = parsed
            .tensors
            .iter()
            .filter(|tensor| tensor.name.starts_with("blk.0.") && tensor.name.contains("ffn_gate"))
            .collect();
        matches.sort_by(|left, right| left.name.cmp(&right.name));
        if matches.is_empty() {
            println!("  <no blk.0 ffn_gate tensor found>");
        }
        for tensor in matches {
            println!(
                "  {} dims={:?} type={:?} nbytes={:?}",
                tensor.name,
                tensor.dims.as_slice(),
                tensor.ggml_type,
                tensor.nbytes()
            );
        }
    }

    /// `cpu.rs`'s `reject_non_float32` gates evaluation to f32 buffers, so a
    /// quantized model must be fully dequantized before it can run through
    /// `evaluate`/`evaluate_parallel`. Computed purely from the tensor
    /// directory's dims -- no tensor payload bytes are read.
    fn print_dequant_memory_cost(parsed: &ParsedGguf) {
        let mut packed_bytes: u128 = 0;
        let mut f32_bytes: u128 = 0;
        for tensor in &parsed.tensors {
            packed_bytes += u128::from(tensor.nbytes().unwrap_or(0));
            f32_bytes += u128::from(tensor.element_count()) * 4;
        }
        println!("-- dequant-to-f32 memory cost (reject_non_float32 requires this before evaluate) --");
        println!("  packed on disk   = {packed_bytes} bytes ({:.3} GiB)", packed_bytes as f64 / GIB);
        println!("  dequantized f32  = {f32_bytes} bytes ({:.3} GiB)", f32_bytes as f64 / GIB);
        println!(
            "  expansion ratio  = {:.2}x",
            f32_bytes as f64 / (packed_bytes.max(1) as f64)
        );
    }

    fn print_full_tensor_list(parsed: &ParsedGguf) {
        println!("-- full tensor directory ({} entries) --", parsed.tensors.len());
        for tensor in &parsed.tensors {
            println!("  {} dims={:?} type={:?}", tensor.name, tensor.dims.as_slice(), tensor.ggml_type);
        }
    }

    #[test]
    #[ignore = "point-in-time report against host-local multi-GB gguf files; run with --ignored --nocapture"]
    fn reports_real_models() {
        for path_str in MODEL_PATHS {
            let path = Path::new(path_str);
            if !path.exists() {
                println!("\n==== SKIP {path_str} (not present on this host) ====");
                continue;
            }
            println!("\n======== {path_str} ========");
            let (file_len, parsed) = parse_header(path);
            println!("file_len = {file_len} bytes ({:.3} GiB)", file_len as f64 / GIB);
            println!(
                "version={} tensor_count={} kv_count={} alignment={} data_offset={}",
                parsed.version, parsed.tensor_count, parsed.kv_count, parsed.alignment, parsed.data_offset
            );
            print_all_metadata(&parsed);
            print_highlights(&parsed);
            print_tensor_summary(&parsed);
            print_functional_groups(&parsed);
            print_expert_layout(&parsed);
            print_dequant_memory_cost(&parsed);
            print_full_tensor_list(&parsed);
        }
    }
}

