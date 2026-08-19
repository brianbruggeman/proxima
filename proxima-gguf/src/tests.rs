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
use core::future::Future;
use core::pin::pin;
use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

use proxima_primitives::pipe::primitives::Pipe;

use crate::error::GgufError;
use crate::parser::{GgufEvent, GgufParser, PollOutcome};
use crate::pipe::{ParseComplete, ParsedGguf};
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
        parser.feed(chunk);
        loop {
            match parser.poll()? {
                PollOutcome::NeedMore => break,
                PollOutcome::Event(GgufEvent::Header {
                    version,
                    tensor_count,
                    kv_count,
                }) => header = Some((version, tensor_count, kv_count)),
                PollOutcome::Event(GgufEvent::Metadata { key, value }) => {
                    metadata.push((key, value));
                }
                PollOutcome::Event(GgufEvent::Tensor(tensor)) => tensors.push(tensor),
                PollOutcome::Event(GgufEvent::Complete {
                    data_offset,
                    alignment,
                }) => completion = Some((data_offset, alignment)),
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

// -- Pipe conformance: ParseComplete::call's future must be ready on the
// first poll (it does no awaiting internally), so a no-op waker suffices.

fn noop_raw_waker() -> RawWaker {
    fn clone(_: *const ()) -> RawWaker {
        noop_raw_waker()
    }
    fn no_op(_: *const ()) {}
    let vtable = &RawWakerVTable::new(clone, no_op, no_op, no_op);
    RawWaker::new(core::ptr::null(), vtable)
}

fn poll_ready<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    // SAFETY: the vtable's functions are all no-ops over a null data
    // pointer; nothing is ever dereferenced.
    let waker = unsafe { Waker::from_raw(noop_raw_waker()) };
    let mut cx = Context::from_waker(&waker);
    match future.as_mut().poll(&mut cx) {
        Poll::Ready(output) => output,
        Poll::Pending => panic!("ParseComplete::call must be ready on first poll"),
    }
}

#[test]
fn parse_complete_pipe_matches_the_free_function() {
    let fixture = build_fixture();
    let via_pipe = poll_ready(ParseComplete::new().call(fixture.bytes.as_slice()))
        .expect("pipe parse of synthetic gguf");
    assert_fixture_parsed(&via_pipe, &fixture);
}

