//! Device parity for `BFloat16`, on the REAL bytes
//! `model.layers.0.mlp.gate.biases` carries in the Qwen3-30B-A3B MLX-4bit
//! checkpoint on this host -- not synthetic, per guiding-principles §9. This
//! checkpoint's header (checked directly via `SafetensorsParser`) reports
//! 300 `BF16` tensors and 120 `U32` tensors (MLX's packed 4-bit format):
//! every quantized weight's per-group `scales`/`biases` are stored as real
//! on-disk `bf16`, the same fixture-loading shape
//! `proxima-safetensors/tests/bf16_real_checkpoint_parity.rs` already reads
//! for the CPU path (`matmul_bf16_f32`); this file reuses that loader,
//! pointed at Metal instead.
//!
//! Unlike `Float16`, MSL has no native `bfloat` storage type on this
//! driver's baseline toolchain, so a `BFloat16` weight genuinely needs an
//! unpack kernel -- `omega::msl::BF16_UNPACK_MSL`'s widen-by-shift
//! (`bfloat16` is the top 16 bits of an `f32`; reconstructing it is
//! `bits << 16` reinterpreted, no rounding table). This test proves that
//! kernel is bit-correct against real checkpoint bytes.
//!
//! Skips (does not fail) when the real file is not present on this host --
//! matching every other `*_real_checkpoint_parity.rs` test in this
//! workspace's posture.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{Read, Seek, SeekFrom};

use proxima_safetensors::{Manifest, SafetensorsParser};
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp,
    append, evaluate, map,
};

const REAL_QWEN3_SAFETENSORS_PATH: &str = "/Users/brianbruggeman/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-MLX-4bit/model-00001-of-00004.safetensors";

const TARGET_TENSOR: &str = "model.layers.0.mlp.gate.biases";

/// Same loader `proxima-safetensors/tests/bf16_real_checkpoint_parity.rs`
/// already uses -- restated here since this is a standalone integration
/// test binary in a different crate and cannot import a sibling crate's
/// test-only helper.
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

    let data_start = 8 + header_len;
    Some((manifest, data_start, file))
}

/// `[rows, k] x [k, 1] -> [rows, 1]`, `weight_dtype` distinguishing "packed
/// bytes" (`BFloat16`) from the dequantized `f32` oracle.
fn matmul_program(rows: u32, k: u32, weight_dtype: DType) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: weight_dtype,
            shape: vec![Extent::Static(rows), Extent::Static(k)],
            name: None,
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(k), Extent::Static(1)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (activation, IndexMap::Affine(map::projection(3, &[2, 1]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("bf16_real_matmul".into()),
        }),
    );
    (program, sum)
}

#[test]
fn metal_matmul_on_real_mlp_gate_biases_bf16_bytes_matches_the_dequantized_f32_cpu_path() {
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

    // Independent reference decode of the SAME real bytes (`half::bf16::
    // from_le_bytes` called directly here, never through the kernel under
    // test) -- guiding-principle 14, the reference must be correct by
    // construction, not a mirror of the thing it checks.
    let dequantized: Vec<f32> = weight_bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| half::bf16::from_le_bytes(*pair).to_f32())
        .collect();

    let (packed_program, packed_sum) = matmul_program(rows as u32, k as u32, DType::BFloat16);
    let metal = omega::execute(
        &packed_program,
        &[],
        &[
            QuantizedBlock::BFloat16(&weight_bytes),
            QuantizedBlock::Float32(&activation),
        ],
        &[packed_sum],
    )
    .expect("metal executes a bf16 matmul on real mlp.gate.biases bytes");

    let (f32_program, f32_sum) = matmul_program(rows as u32, k as u32, DType::Float32);
    let cpu = evaluate(&f32_program, &[], &[&dequantized, &activation], &[f32_sum])
        .expect("dequantized f32 cpu matmul evaluates");

    let actual = metal.root();
    let expected = cpu.root();
    assert_eq!(actual.len(), rows, "degenerate gate: no outputs compared");
    assert_eq!(actual.len(), expected.len());

    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(got.is_finite(), "metal produced a non-finite value: {got}");
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
        "real {TARGET_TENSOR} (BFloat16, {rows}x{k}) metal vs dequantized-f32 cpu: \
         max_diff={max_diff} max_magnitude={max_magnitude} relative={relative}"
    );
    assert!(
        relative < 1e-5,
        "bf16 widen-by-shift disagrees with the dequantized reference on REAL checkpoint bytes: \
         relative={relative} max_diff={max_diff}"
    );
}
