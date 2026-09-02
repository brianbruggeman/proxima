//! Parity for the composed `bf16` matmul kernel (`proxima_tensor::cpu::
//! matmul_bf16_f32`) on the REAL bytes `model.layers.0.mlp.gate.biases`
//! carries in the Qwen3-30B-A3B MLX-4bit checkpoint on this host -- not
//! synthetic, per guiding-principles §9. This checkpoint's header (checked
//! directly, `xxd`/`dd` on the file) reports 300 `BF16` tensors and 120
//! `U32` tensors (MLX's packed 4-bit format): every quantized weight's
//! per-group `scales`/`biases` are stored as real on-disk `bf16`, which is
//! exactly the byte shape [`proxima_tensor::cpu::matmul_bf16_f32`] reaches.
//!
//! `model.layers.0.mlp.gate.biases` is `[128, 32]` (rows=128, k=32) -- a
//! small, real, non-degenerate tensor, chosen only for its shape and
//! real-file provenance; what it semantically represents (an MLX
//! dequantization bias, not a dense weight) is irrelevant to this test,
//! which checks the `bf16 -> f32` matmul kernel against an INDEPENDENT
//! decode (`half::bf16::from_le_bytes` called directly here, never through
//! [`proxima_tensor::cpu::matmul_bf16_f32`] itself) of the same real bytes.
//!
//! Skips (does not fail) when the real file is not present on this host --
//! matching every other `*_real_checkpoint_parity.rs` test in this
//! workspace's posture.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{Read, Seek, SeekFrom};

use proxima_safetensors::{Manifest, SafetensorsParser};
use proxima_tensor::DType;

const REAL_QWEN3_SAFETENSORS_PATH: &str = "/Users/brianbruggeman/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-MLX-4bit/model-00001-of-00004.safetensors";

const TARGET_TENSOR: &str = "model.layers.0.mlp.gate.biases";

/// Parses the real file's header via the actual [`SafetensorsParser`] FSM,
/// then satisfies [`SafetensorsParser::finish`]'s bounds check (it counts
/// every declared tensor-data byte, per this crate's own "counted, never
/// buffered" contract) by pushing zero-filled dummy chunks up to the
/// declared total -- never allocating or reading the real multi-GiB data
/// section, since only the COUNT is checked, not the content.
fn real_manifest(path: &std::path::Path) -> Option<(Manifest, u64, std::fs::File)> {
    let mut file = std::fs::File::open(path).ok()?;

    let mut len_prefix = [0u8; 8];
    file.read_exact(&mut len_prefix).ok()?;
    let header_len = u64::from_le_bytes(len_prefix);

    let mut header_bytes = vec![0u8; header_len as usize];
    file.read_exact(&mut header_bytes).ok()?;

    let parser = SafetensorsParser::new()
        .push(&len_prefix)
        .ok()?
        .push(&header_bytes)
        .ok()?;
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

    // `data_offsets` are relative to the first byte after the 8-byte
    // length prefix AND the header JSON itself -- this is that base.
    let data_start = 8 + header_len;
    Some((manifest, data_start, file))
}

#[test]
fn matmul_bf16_f32_on_real_mlp_gate_biases_bytes_matches_an_independent_bf16_decode() {
    let path = std::path::Path::new(REAL_QWEN3_SAFETENSORS_PATH);
    let Some((manifest, data_start, mut file)) = real_manifest(path) else {
        eprintln!("real safetensors file not found at {REAL_QWEN3_SAFETENSORS_PATH}; test skipped");
        return;
    };
    let Some(entry) = manifest.tensor(TARGET_TENSOR) else {
        eprintln!("{TARGET_TENSOR} not present in this checkpoint; test skipped");
        return;
    };
    assert_eq!(
        entry.dtype,
        DType::BFloat16,
        "{TARGET_TENSOR} must be the real bf16 tensor this test targets"
    );
    assert_eq!(
        entry.shape,
        vec![128, 32],
        "{TARGET_TENSOR}'s real shape must match this test's fixture"
    );

    let rows = entry.shape[0] as usize;
    let k = entry.shape[1] as usize;
    let byte_len = entry.byte_len() as usize;
    assert_eq!(byte_len, rows * k * 2, "bf16 is 2 bytes/element");

    let mut weight_bytes = vec![0u8; byte_len];
    file.seek(SeekFrom::Start(data_start + entry.data_offsets.0))
        .expect("seek to tensor data");
    file.read_exact(&mut weight_bytes)
        .expect("read exact real tensor byte range");

    // Non-degenerate, deterministic activation -- not all-ones or all-zero.
    let activation: Vec<f32> = (0..k)
        .map(|index| ((index % 7) as f32 - 3.0) * 0.5)
        .collect();

    let actual = proxima_tensor::cpu::matmul_bf16_f32(&weight_bytes, rows, &activation)
        .expect("well-formed real bf16 matmul");

    // Independent reference: decode the SAME real bytes by hand
    // (`half::bf16::from_le_bytes`), never by calling back into
    // `matmul_bf16_f32`/`dot_bf16_f32` -- guiding-principle 14, the
    // reference must be correct by construction, not a mirror of the
    // thing under test.
    let expected: Vec<f32> = weight_bytes
        .chunks_exact(k * 2)
        .map(|row_bytes| {
            row_bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| half::bf16::from_le_bytes(*pair).to_f32())
                .zip(&activation)
                .map(|(weight, &value)| weight * value)
                .sum::<f32>()
        })
        .collect();

    assert_eq!(actual.len(), rows, "degenerate gate: no outputs compared");
    assert_eq!(actual.len(), expected.len());

    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(
            got.is_finite(),
            "matmul_bf16_f32 produced a non-finite value on real checkpoint bytes: {got}"
        );
        max_diff = max_diff.max((got - want).abs());
    }
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let relative = if max_magnitude > 0.0 {
        max_diff / max_magnitude
    } else {
        max_diff
    };
    eprintln!(
        "real {TARGET_TENSOR} ({rows}x{k} bf16) matmul_bf16_f32 vs independent decode: \
         max_diff={max_diff} max_magnitude={max_magnitude} relative={relative}"
    );
    assert!(
        relative < 1e-5,
        "matmul_bf16_f32 disagrees with an independent decode of the SAME real checkpoint bytes: \
         relative={relative} max_diff={max_diff}"
    );
}
