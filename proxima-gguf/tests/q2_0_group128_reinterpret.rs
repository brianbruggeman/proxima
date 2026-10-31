//! Files labelled ggml type 42 (`Q2_0`, 64 elements / 18 bytes upstream) whose tensor data is
//! really PrismML's group-128 `PQ2_0` (128 elements / 34 bytes). The rule under test: a type-42
//! file is read as `PQ2_0` exactly when every offset in its directory is consistent with
//! group-128 sizes and not with group-64 sizes.
//!
//! Directory fragments below are the first tensors of the real checkpoints:
//! - `Ternary-Bonsai-1.7B-Q2_0.gguf` (sha256 d97d94eb...228a), data section at 5945280,
//!   `blk.0.attn_k.weight` at +82516128: group-128 sizes. Differs from
//!   `Ternary-Bonsai-1.7B-PQ2_0.gguf` only in the type ids and `general.file_type` (198 bytes).
//! - `Ternary-Bonsai-8B-Q2_0.gguf` (sha256 3c8d7047...f60b), same relationship to
//!   `Ternary-Bonsai-8B-PQ2_0.gguf` (sha256 1376f942...b922).
//! - `Ternary-Bonsai-1.7B-Q2_0_g64.gguf` (sha256 6d0ecb3d...5c91), data at 5945280,
//!   `blk.0.attn_k.weight` at +87369536: genuine group-64 sizes.

#![allow(clippy::expect_used)]

use std::io::Read;
use std::path::Path;

use proxima_gguf::types::GgmlType;
use proxima_gguf::{GgufError, ParsedGguf, parse_complete};

const RAW_Q2_0: i32 = 42;
const RAW_F32: i32 = 0;
const ALIGNMENT: u64 = 32;
const MODELS: &str = "/mnt/data5/models";

fn push_string(buf: &mut Vec<u8>, value: &str) {
    buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
    buf.extend_from_slice(value.as_bytes());
}

fn push_tensor(buf: &mut Vec<u8>, name: &str, dims: &[u64], raw_type: i32, offset: u64) {
    push_string(buf, name);
    buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for dim in dims {
        buf.extend_from_slice(&dim.to_le_bytes());
    }
    buf.extend_from_slice(&raw_type.to_le_bytes());
    buf.extend_from_slice(&offset.to_le_bytes());
}

fn directory(tensors: &[(&str, &[u64], i32, u64)]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&(tensors.len() as i64).to_le_bytes());
    buf.extend_from_slice(&0i64.to_le_bytes());
    for (name, dims, raw_type, offset) in tensors {
        push_tensor(&mut buf, name, dims, *raw_type, *offset);
    }
    buf
}

fn with_data(mut header: Vec<u8>, data_len: u64) -> Vec<u8> {
    let data_offset = header.len().next_multiple_of(ALIGNMENT as usize);
    header.resize(data_offset + data_len as usize, 0);
    header
}

fn types_of(parsed: &ParsedGguf) -> Vec<GgmlType> {
    parsed
        .tensors
        .iter()
        .map(|tensor| tensor.ggml_type)
        .collect()
}

const OUTPUT_NORM: (&str, &[u64], i32, u64) = ("output_norm.weight", &[2048], RAW_F32, 0);
const TOKEN_EMBD: &[u64] = &[2048, 151_669];
const ATTN_K: &[u64] = &[2048, 1024];
const ATTN_K_NORM: &[u64] = &[128];

#[test]
fn legacy_group128_offsets_read_type_42_as_pq2_0() {
    let bytes = directory(&[
        OUTPUT_NORM,
        ("token_embd.weight", TOKEN_EMBD, RAW_Q2_0, 8192),
        ("blk.0.attn_k.weight", ATTN_K, RAW_Q2_0, 82_516_128),
        ("blk.0.attn_k_norm.weight", ATTN_K_NORM, RAW_F32, 83_073_184),
    ]);

    let parsed = parse_complete(&bytes).expect("group-128 offsets are consistent");

    assert_eq!(
        types_of(&parsed),
        [
            GgmlType::F32,
            GgmlType::Pq2_0,
            GgmlType::Pq2_0,
            GgmlType::F32
        ]
    );
}

#[test]
fn genuine_group64_offsets_keep_type_42_as_q2_0() {
    let bytes = directory(&[
        OUTPUT_NORM,
        ("token_embd.weight", TOKEN_EMBD, RAW_Q2_0, 8192),
        ("blk.0.attn_k.weight", ATTN_K, RAW_Q2_0, 87_369_536),
        ("blk.0.attn_k_norm.weight", ATTN_K_NORM, RAW_F32, 87_959_360),
    ]);

    let parsed = parse_complete(&bytes).expect("group-64 offsets are consistent");

    assert_eq!(
        types_of(&parsed),
        [GgmlType::F32, GgmlType::Q2_0, GgmlType::Q2_0, GgmlType::F32]
    );
}

#[test]
fn mixed_group64_and_group128_spans_name_the_tensor_and_both_expected_offsets() {
    let bytes = directory(&[
        OUTPUT_NORM,
        ("token_embd.weight", TOKEN_EMBD, RAW_Q2_0, 8192),
        ("blk.0.attn_k.weight", ATTN_K, RAW_Q2_0, 87_369_536),
        ("blk.0.attn_k_norm.weight", ATTN_K_NORM, RAW_F32, 83_073_184),
    ]);

    let outcome = parse_complete(&bytes);

    assert!(
        matches!(
            &outcome,
            Err(GgufError::Q2_0LayoutUnresolved {
                tensor,
                group64_expected: 87_959_360,
                group128_expected: 83_073_184,
                found: 83_073_184,
            }) if tensor == "blk.0.attn_k_norm.weight"
        ),
        "got {outcome:?}"
    );
}

#[test]
fn offsets_matching_neither_layout_without_type_42_stay_a_plain_offset_mismatch() {
    let bytes = directory(&[
        OUTPUT_NORM,
        ("blk.0.attn_k_norm.weight", ATTN_K_NORM, RAW_F32, 4),
    ]);

    let outcome = parse_complete(&bytes);

    assert!(
        matches!(outcome, Err(GgufError::TensorOffsetMismatch { .. })),
        "got {outcome:?}"
    );
}

const TRAILING_ATTN_K: (&str, &[u64], i32, u64) = ("blk.0.attn_k.weight", ATTN_K, RAW_Q2_0, 8192);
const GROUP64_ATTN_K_BYTES: u64 = 2048 * 1024 / 64 * 18;
const GROUP128_ATTN_K_BYTES: u64 = 2048 * 1024 / 128 * 34;

#[test]
fn trailing_type_42_tensor_is_sized_by_the_end_of_the_data_section() {
    let header = directory(&[OUTPUT_NORM, TRAILING_ATTN_K]);

    let legacy = parse_complete(&with_data(header.clone(), 8192 + GROUP128_ATTN_K_BYTES))
        .expect("span equals the group-128 size");
    let upstream = parse_complete(&with_data(header, 8192 + GROUP64_ATTN_K_BYTES))
        .expect("span equals the group-64 size");

    assert_eq!(legacy.tensors[1].ggml_type, GgmlType::Pq2_0);
    assert_eq!(upstream.tensors[1].ggml_type, GgmlType::Q2_0);
}

#[test]
fn trailing_type_42_tensor_with_a_foreign_span_is_rejected() {
    let header = directory(&[OUTPUT_NORM, TRAILING_ATTN_K]);

    let outcome = parse_complete(&with_data(header, 8192 + GROUP64_ATTN_K_BYTES + 4096));

    assert!(
        matches!(
            &outcome,
            Err(GgufError::Q2_0LayoutUnresolved {
                tensor,
                group64_expected: GROUP64_ATTN_K_BYTES,
                group128_expected: GROUP128_ATTN_K_BYTES,
                ..
            }) if tensor == "blk.0.attn_k.weight"
        ),
        "got {outcome:?}"
    );
}

fn read_directory(name: &str) -> Option<ParsedGguf> {
    let path = Path::new(MODELS).join(name);
    let mut file = std::fs::File::open(&path).ok().or_else(|| {
        eprintln!("skipping: {} is absent", path.display());
        None
    })?;
    let mut buf = vec![0u8; 64 << 20];
    let read = file.read(&mut buf).expect("read gguf directory prefix");
    buf.truncate(read);
    Some(parse_complete(&buf).unwrap_or_else(|error| panic!("{name}: {error}")))
}

#[test]
fn real_mislabelled_checkpoints_read_identically_to_their_pq2_0_twins() {
    for (mislabelled, twin) in [
        (
            "Ternary-Bonsai-1.7B-Q2_0.gguf",
            "Ternary-Bonsai-1.7B-PQ2_0.gguf",
        ),
        (
            "Ternary-Bonsai-8B-Q2_0.gguf",
            "Ternary-Bonsai-8B-PQ2_0.gguf",
        ),
    ] {
        let (Some(parsed), Some(expected)) = (read_directory(mislabelled), read_directory(twin))
        else {
            continue;
        };

        assert_eq!(parsed.tensors, expected.tensors, "{mislabelled}");
        assert_eq!(parsed.data_offset, expected.data_offset, "{mislabelled}");
    }
}

#[test]
fn real_group64_checkpoint_keeps_every_type_42_tensor_as_q2_0() {
    let Some(parsed) = read_directory("Ternary-Bonsai-1.7B-Q2_0_g64.gguf") else {
        return;
    };

    let quantized = parsed
        .tensors
        .iter()
        .filter(|tensor| tensor.dims.len() == 2)
        .map(|tensor| tensor.ggml_type)
        .collect::<Vec<_>>();

    assert!(quantized.contains(&GgmlType::Q2_0));
    assert!(!quantized.contains(&GgmlType::Pq2_0));
}
