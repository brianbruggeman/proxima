//! PROVEN CAUSE (read, not guessed, same shape as `gemma4_q5_1_metal_cpu_parity.rs`):
//! gemma4's real checkpoint quantizes every MoE expert's down-projection
//! (`blk.{layer}.ffn_down_exps.weight`, layers 1..29 -- layer 0 alone is
//! `Q5_1`) as `Q5_0` -- confirmed by directly parsing the real 13GB
//! checkpoint's GGUF header (`proxima_gguf::parse_complete` against
//! `~/.ollama/models/blobs/sha256-ea549b76...`): `blk.1.ffn_down_exps.weight
//! dims=[704, 2816, 128] ggml_type=Q5_0`, every other layer 1..29 the same
//! codec. `Q5_0` is `Q5_1` with the per-block `min` term dropped (symmetric,
//! `value = d * (level - 16)`) -- llama.cpp's simplest 5-bit legacy format,
//! same reasoning `Q5_1`'s own doc gives for why a K-quant cannot represent
//! this row length: `704 = 22 * QK5_0` (32), not a multiple of `QK_K` (256).
//!
//! Before this file's companion source changes (`proxima-tensor`'s
//! `QuantizedBlock::Q5_0` CPU kernel, `omega::msl::PackedCodec::Q5_0`'s
//! unpack kernel, `proxima-model-interop::bind::Codec::Q5_0`), a
//! `Q5_0` expert dequantized to `f32` on load -- 30 layers x 128 experts x
//! [704, 2816] x 4 bytes/f32 is the multi-GB device allocation this file's
//! own memory-collapse claim rests on. This file proves the KERNEL half
//! (correctness) empirically, mirroring
//! `gemma4_q5_1_metal_cpu_parity.rs::metal_matches_cpu_on_real_q5_1_down_projection_bytes`
//! exactly: [`quantized_matmul_program`] is the same minimal
//! `[rows, k] x [k, 1] -> [rows, 1]` shape, fed REAL `Q5_0` bytes sliced
//! directly out of the real gemma4 checkpoint's `blk.1.ffn_down_exps.weight`
//! tensor (expert 0's own contiguous row range -- no repacking, no
//! hand-rolled encoder; `proxima_gguf::quant::q5_0` ships no `quantize`
//! function at all, only `dequantize`, so real on-disk bytes are the only
//! way to exercise this codec's format faithfully). CPU
//! (`evaluate_quantized_exact`, which has a `Q5_0` dot kernel --
//! `dot_q5_0_f32`/`matmul_q5_0_f32`, `proxima-tensor/src/cpu/gemm_dot_quant.rs`)
//! is cross-checked against an independent dequantize-then-f32-evaluate
//! reference before Metal ever runs, so a Metal divergence cannot be blamed
//! on a bad CPU oracle.
#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::env;
use std::fs::File;

use proxima_gguf::parse_complete;
use proxima_gguf::quant::q5_0::{BLOCK_BYTES, QK5_0, dequantize};
use proxima_gguf::types::GgmlType;
use proxima_tensor::map::projection;
use proxima_tensor::op::{Extent, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp, append};
use proxima_tensor::{DType, IndexMap, QuantizedBlock};

/// [`proxima_tensor::cpu::tests::quantized_matmul_program`]'s own shape --
/// see `gemma4_q5_1_metal_cpu_parity.rs`'s copy of this function for the
/// full rationale (identical here; duplicated because `tests/` binaries
/// cannot share a helper module across files without a `tests/common`
/// crate this workspace does not have for this test pair yet).
fn quantized_matmul_program(rows: u32, k: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc_vec(&[rows, k]),
            name: None,
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc_vec(&[k, 1]),
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(3, &[0, 2]))),
                (activation, IndexMap::Affine(projection(3, &[2, 1]))),
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
            in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("quantized_matmul".into()),
        }),
    );
    (program, sum)
}

fn alloc_vec(dims: &[u32]) -> Vec<Extent> {
    dims.iter().map(|&dim| Extent::Static(dim)).collect()
}

/// The real gemma4 checkpoint's `blk.1.ffn_down_exps.weight` tensor: `Q5_0`,
/// `dims=[704, 2816, 128]` (`ne0=704=expert_feed_forward`, the
/// row/contraction axis; `ne1=2816=embedding`, the output-row axis;
/// `ne2=128=experts`) -- `704 = 22 * QK5_0`, so `rows` real output rows for
/// expert 0, sliced from the FRONT of the tensor's own byte range, is a
/// byte-exact contiguous prefix needing no repacking at all.
fn real_q5_0_expert0_down_projection_rows(rows: usize) -> (Vec<u8>, usize) {
    let path = env::var("GEMMA4_GGUF_PATH").unwrap_or_else(|_| {
        "/Users/brianbruggeman/.ollama/models/blobs/sha256-ea549b7688d4c95019754880c21e3f29c58c985a7a1c3b37b9eebd0a95224129"
            .to_string()
    });
    let file = File::open(&path).expect("open real gemma4 gguf checkpoint");
    let mapping = unsafe { memmap2::Mmap::map(&file) }.expect("mmap real gemma4 gguf checkpoint");
    let file_bytes: &[u8] = &mapping;
    let parsed = parse_complete(file_bytes).expect("parse real gemma4 gguf header");
    let tensor = parsed
        .tensors
        .iter()
        .find(|tensor| tensor.name == "blk.1.ffn_down_exps.weight")
        .expect("blk.1.ffn_down_exps.weight present in the real checkpoint");
    assert_eq!(
        tensor.ggml_type,
        GgmlType::Q5_0,
        "blk.1.ffn_down_exps.weight must be Q5_0 on this checkpoint -- if this fails, the \
         checkpoint's own quantization changed and this test's whole premise needs re-deriving"
    );
    let expert_feed_forward = tensor.dims[0] as usize;
    assert_eq!(
        expert_feed_forward % QK5_0,
        0,
        "ffn_down_exps.weight's row length must be Q5_0-block-aligned"
    );
    let range = parsed
        .tensor_data_range(tensor, file_bytes.len() as u64)
        .expect("tensor_data_range for blk.1.ffn_down_exps.weight");
    let row_bytes = expert_feed_forward / QK5_0 * BLOCK_BYTES;
    let slice_bytes = row_bytes * rows;
    let start = range.start as usize;
    (
        file_bytes[start..start + slice_bytes].to_vec(),
        expert_feed_forward,
    )
}

/// Max-abs diff, relative to the reference row's own norm.
fn relative_error(found: &[f32], wanted: &[f32]) -> f32 {
    let norm: f32 = wanted.iter().map(|value| value * value).sum::<f32>().sqrt().max(1e-6);
    found
        .iter()
        .zip(wanted.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max)
        / norm
}

/// The one cell this file exists to cover: a `Q5_0`-quantized weight (real
/// bytes off the real gemma4 checkpoint's own `blk.1.ffn_down_exps.weight`
/// expert-0 rows) run through the minimal, axis-unambiguous
/// `[rows, k] x [k, 1] -> [rows, 1]` quantized-matmul shape -- CPU vs an
/// independent dequantize-then-f32 reference (self-check on the oracle),
/// then CPU vs Metal.
#[test]
fn metal_matches_cpu_on_real_q5_0_ffn_down_exps_bytes() {
    let rows = 4_usize;
    let (weight_bytes, k) = real_q5_0_expert0_down_projection_rows(rows);

    let activation: Vec<f32> = (0..k)
        .map(|index| ((index as f32 * 0.618_034) % 1.0) * 2.0 - 1.0)
        .collect();

    let mut dequantized_weight = vec![0.0f32; rows * k];
    let block_row_bytes = k / QK5_0 * BLOCK_BYTES;
    for (row_bytes, row_f32) in weight_bytes
        .chunks_exact(block_row_bytes)
        .zip(dequantized_weight.chunks_exact_mut(k))
    {
        dequantize(row_bytes, row_f32).expect("real q5_0 row decodes under its own real block count");
    }
    let independent_reference: Vec<f32> = (0..rows)
        .map(|row| {
            dequantized_weight[row * k..(row + 1) * k]
                .iter()
                .zip(activation.iter())
                .map(|(weight, value)| weight * value)
                .sum::<f32>()
        })
        .collect();

    let (program, sum) = quantized_matmul_program(rows as u32, k as u32);
    let blocks = [
        QuantizedBlock::Packed { codec: Codec::Q5_0, bytes: &weight_bytes },
        QuantizedBlock::Float32(&activation),
    ];

    let cpu = proxima_tensor::cpu::evaluate_quantized_exact(&program, &[], &blocks, &[sum])
        .expect("cpu evaluates the real q5_0 quantized matmul");
    let cpu_values = cpu.get(sum).expect("cpu produced the matmul output").0;
    let cpu_vs_independent = relative_error(cpu_values, &independent_reference);
    std::println!("gemma4_q5_0_parity cpu_vs_independent_dequant relative_error={cpu_vs_independent:e}");
    assert!(
        cpu_vs_independent <= 1e-4,
        "cpu's own Q5_0 quantized-matmul kernel disagrees with an independent dequantize-then-f32 \
         reference on real checkpoint bytes -- the CPU oracle itself is broken, not just Metal: \
         relative_error={cpu_vs_independent:e}"
    );

    let metal = omega::execute(
        &program,
        &[],
        &blocks,
        &[sum],
        proxima_tensor::NumericPolicy::default(),
    )
    .expect("metal evaluates the real q5_0 quantized matmul");
    let metal_values = metal.get(sum).expect("metal produced the matmul output").0;
    let metal_vs_cpu = relative_error(metal_values, cpu_values);
    std::println!(
        "gemma4_q5_0_parity metal_vs_cpu relative_error={metal_vs_cpu:e} cpu={cpu_values:?} \
         metal={metal_values:?}"
    );
    assert!(
        metal_vs_cpu <= 1e-3,
        "metal disagrees with cpu on a Q5_0-quantized weight (real bytes off gemma4's own \
         blk.1.ffn_down_exps.weight, expert 0): relative_error={metal_vs_cpu:e}"
    );
}
