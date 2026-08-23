//! Round-trips a REAL tensor's REAL bytes -- not a synthetic fixture --
//! through [`write_complete`] and back through [`parse_complete`], against
//! `~/.lmstudio/models/HuggingFaceTB/SmolLM2-135M-Instruct/model.safetensors`
//! (269,060,552 bytes on this host, `header_len` 30528, 272 `BF16` tensors,
//! `__metadata__: {"format":"pt"}`; checked directly on this file with
//! `xxd`/manual header decode, 2026-08-23).
//!
//! `writer::tests::write_complete_round_trips_every_field_and_every_tensor_
//! byte` already proves the writer agrees with our own reader on a
//! synthetic fixture -- that proves the two agree with EACH OTHER, not that
//! either is correct against the real wire format (guiding principle 9:
//! synthetic round-trips test plumbing, not the contract). This is the
//! oracle that matters: real bytes a real training/inference stack wrote,
//! parsed by our reader, re-serialized by our writer, re-parsed by our
//! reader again, and asserted byte-identical against the ORIGINAL real
//! bytes -- never against our own re-encoding of them.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{Read, Seek, SeekFrom};

use proxima_safetensors::{Manifest, SafetensorsModel, SafetensorsParser, TensorPayload, write_complete};
use proxima_tensor::DType;

const REAL_SMOLLM2_SAFETENSORS_PATH: &str =
    "/Users/brianbruggeman/.lmstudio/models/HuggingFaceTB/SmolLM2-135M-Instruct/model.safetensors";

/// Small (1152-byte), non-degenerate, real on-disk tensor: BF16, shape
/// [576] -- big enough that a byte-order or offset bug would show up, small
/// enough to read whole without touching the 269 MB tensor-data region.
const TARGET_TENSOR: &str = "model.layers.0.input_layernorm.weight";

/// Parses the real file's header through the actual [`SafetensorsParser`]
/// FSM (not a hand-rolled JSON read) -- the same "trust the real reader,
/// not a shortcut" posture `bf16_real_checkpoint_parity.rs`'s own
/// `real_manifest` helper uses, satisfying [`SafetensorsParser::finish`]'s
/// bounds check by pushing zero-filled dummy chunks up to the declared
/// total (only the COUNT is checked at `finish`, never the content, so
/// this never allocates or reads the real multi-GiB data section).
fn real_manifest(path: &std::path::Path) -> Option<(Manifest, u64, std::fs::File)> {
    let mut file = std::fs::File::open(path).ok()?;

    let mut len_prefix = [0u8; 8];
    file.read_exact(&mut len_prefix).ok()?;
    let header_len = u64::from_le_bytes(len_prefix);

    let mut header_bytes = vec![0u8; header_len as usize];
    file.read_exact(&mut header_bytes).ok()?;

    let parser = SafetensorsParser::new().push(&len_prefix).ok()?.push(&header_bytes).ok()?;
    let SafetensorsParser::TensorData { manifest, seen } = parser else {
        eprintln!("safetensors header did not fully parse from the first {header_len} bytes");
        return None;
    };

    let declared = manifest.declared_data_len();
    let dummy_chunk = vec![0u8; 1 << 20];
    let mut remaining = declared.saturating_sub(seen);
    let mut parser = SafetensorsParser::TensorData { manifest, seen };
    while remaining > 0 {
        let take = remaining.min(dummy_chunk.len() as u64) as usize;
        parser = parser.push(&dummy_chunk[..take]).ok()?;
        remaining -= take as u64;
    }
    let manifest = parser.finish().ok()?;

    let data_start = 8 + header_len;
    Some((manifest, data_start, file))
}

#[test]
fn write_complete_round_trips_a_real_tensors_real_bytes_byte_identical() {
    let path = std::path::Path::new(REAL_SMOLLM2_SAFETENSORS_PATH);
    let Some((manifest, data_start, mut file)) = real_manifest(path) else {
        eprintln!("real safetensors file not found at {REAL_SMOLLM2_SAFETENSORS_PATH}; test skipped");
        return;
    };
    let Some(entry) = manifest.tensor(TARGET_TENSOR) else {
        eprintln!("{TARGET_TENSOR} not present in this checkpoint; test skipped");
        return;
    };
    assert_eq!(entry.dtype, DType::BFloat16, "{TARGET_TENSOR} must be the real bf16 tensor this test targets");
    assert_eq!(entry.shape, vec![576u64], "{TARGET_TENSOR}'s real shape must match this test's fixture");
    assert_eq!(manifest.metadata.get("format").map(String::as_str), Some("pt"), "real __metadata__ must round-trip through our own reader");

    let byte_len = entry.byte_len() as usize;
    assert_eq!(byte_len, 576 * 2, "bf16 is 2 bytes/element");

    let mut real_bytes = vec![0u8; byte_len];
    file.seek(SeekFrom::Start(data_start + entry.data_offsets.0)).expect("seek to real tensor data");
    file.read_exact(&mut real_bytes).expect("read exact real tensor byte range");
    // non-degenerate: a real trained bf16 layernorm weight is never all-zero.
    assert!(real_bytes.iter().any(|&byte| byte != 0), "real tensor bytes must not be degenerate all-zero");

    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert("format".to_string(), "pt".to_string());
    let model = SafetensorsModel {
        tensors: vec![TensorPayload {
            name: TARGET_TENSOR.to_string(),
            dtype: entry.dtype,
            shape: entry.shape.clone(),
            data: &real_bytes,
        }],
        metadata,
    };
    let written = write_complete(&model).expect("writer accepts a real tensor's real bytes");

    let reparsed = proxima_safetensors::parse_complete(&written).expect("writer's own output re-parses through our reader");
    let reparsed_entry = reparsed.tensor(TARGET_TENSOR).expect("re-parsed manifest carries the tensor back");
    assert_eq!(reparsed_entry.dtype, entry.dtype, "dtype must round-trip");
    assert_eq!(reparsed_entry.shape, entry.shape, "shape must round-trip");
    assert_eq!(reparsed_entry.byte_len(), entry.byte_len(), "byte_len must round-trip");
    assert_eq!(reparsed.metadata.get("format").map(String::as_str), Some("pt"), "metadata must round-trip through the writer");

    let written_header_len = u64::from_le_bytes(written[..8].try_into().expect("8-byte length prefix")) as usize;
    let written_data_start = 8 + written_header_len;
    let (start, end) = reparsed_entry.data_offsets;
    let round_tripped_bytes = &written[written_data_start + start as usize..written_data_start + end as usize];

    assert_eq!(
        round_tripped_bytes, real_bytes.as_slice(),
        "round-tripped tensor bytes must be byte-identical to the ORIGINAL real bytes on disk, not merely to our own re-encoding of them"
    );
}
