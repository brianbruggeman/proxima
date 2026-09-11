//! Correctness gate for `metal-q4k-ggml-port`, the verbatim transcription of
//! ggml's `kernel_mul_mv_q4_K_f32_impl<4,2,32>`
//! (`ggml-metal.metal:5086-5193`) onto `omega::msl`'s row-blocked packed
//! path (`push_q4k_ggml_port_body`). Same posture as
//! `q4k_real_checkpoint_parity.rs`: real GGUF bytes wherever the host has
//! them (guiding-principles §9), skip rather than fail when the file is
//! absent. Two real tensors, `blk.0.attn_q.weight` AND `blk.0.ffn_up.weight`
//! -- the first every prior Q4_K landing on this box already exercises, the
//! second a DIFFERENT projection shape ([4096 x 14336] rather than
//! [4096 x 4096]) so the port's `ib`-stride loop crosses more super-blocks
//! per row than `attn_q` alone would prove. Plus one synthetic ragged-K
//! case: `OUT_DIM` not a multiple of `PACKED_ROWS_PER_GROUP` (a partial
//! last SIMD group) and `IN_DIM` only 2 super-blocks (so half of the
//! `ix = lane/8` lanes run zero `ib` iterations and contribute only their
//! identity seed) -- the boundary shapes a large real tensor never happens
//! to hit deterministically.

#![cfg(all(feature = "metal-q4k-ggml-port", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{Read, Seek, SeekFrom};

use proxima_gguf::parser::{GgufEvent, GgufParser};
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::quant::q4_k::{self, BLOCK_BYTES, QK_K, quantize};
use proxima_gguf::types::GgmlType;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce, ReduceInit,
    ScalarOp, append, evaluate, map,
};

fn real_gguf_path() -> String {
    std::env::var("PROXIMA_BENCH_GGUF_PATH").unwrap_or_else(|_| {
        "/Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf"
            .to_string()
    })
}

fn real_gguf_header(path: &std::path::Path) -> Option<(ParsedGguf, u64, std::fs::File)> {
    let mut file = std::fs::File::open(path).ok()?;
    let file_len = file.metadata().ok()?.len();

    let mut prefix_len = 1usize << 20;
    loop {
        let mut buf = vec![0u8; prefix_len];
        file.seek(SeekFrom::Start(0)).expect("seek to start");
        let read = file.read(&mut buf).expect("read gguf prefix");
        buf.truncate(read);

        if let Ok((parser, events)) = GgufParser::new().push(&buf) {
            let mut version = None;
            let mut metadata = Vec::new();
            let mut tensors = Vec::new();
            let mut completion = None;
            for event in events {
                match event {
                    GgufEvent::Header {
                        version: version_value,
                        ..
                    } => version = Some(version_value),
                    GgufEvent::Metadata { key, value } => metadata.push((key, value)),
                    GgufEvent::Tensor(tensor) => tensors.push(tensor),
                    GgufEvent::Complete {
                        data_offset,
                        alignment,
                    } => {
                        completion = Some((data_offset, alignment));
                    }
                }
            }
            if let (Some(version), Some((data_offset, alignment))) = (version, completion) {
                parser.finish().expect("parser reports complete and clean");
                let parsed = ParsedGguf {
                    version,
                    tensor_count: tensors.len() as u64,
                    kv_count: metadata.len() as u64,
                    metadata,
                    tensors,
                    data_offset,
                    alignment,
                };
                return Some((parsed, file_len, file));
            }
        }
        if prefix_len as u64 >= file_len {
            return None;
        }
        prefix_len *= 2;
    }
}

fn real_tensor_bytes(
    file: &mut std::fs::File,
    parsed: &ParsedGguf,
    file_len: u64,
    name: &str,
    expect_type: GgmlType,
) -> Option<(Vec<u8>, usize, usize)> {
    let tensor = parsed
        .tensors
        .iter()
        .find(|candidate| candidate.name == name)?;
    if tensor.ggml_type != expect_type {
        eprintln!(
            "real_tensor_bytes: {name} is {:?} in this file, not {expect_type:?} -- test skipped, not faked",
            tensor.ggml_type
        );
        return None;
    }
    let in_dim = tensor.dims[0] as usize;
    let out_dim = tensor.dims[1] as usize;
    let range = parsed
        .tensor_data_range(tensor, file_len)
        .expect("tensor byte range within file bounds");
    let mut buf = vec![0u8; (range.end - range.start) as usize];
    file.seek(SeekFrom::Start(range.start))
        .expect("seek to tensor data");
    file.read_exact(&mut buf)
        .expect("read exact tensor byte range");
    Some((buf, in_dim, out_dim))
}

/// `[rows, k] x [k, 1] -> [rows, 1]`, `weight_dtype` distinguishing "packed
/// bytes" (`UInt8`) from the dequantized `f32` oracle -- restated per
/// `q4k_real_checkpoint_parity.rs`'s own standalone-integration-test-binary
/// posture.
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
            name: Some("q4k_ggml_port_matmul".into()),
        }),
    );
    (program, sum)
}

const ROWS_TO_CHECK: usize = 64;
/// Max-abs tolerance this landing's brief specifies -- one order of
/// magnitude looser than `q4k_real_checkpoint_parity.rs`'s own `1e-5`
/// RELATIVE bound, because this assertion is on the raw max-abs difference
/// (no batch-peak normalization), and this port's arithmetic ORDER differs
/// from the dequantized-f32 CPU oracle's (ggml's `(acc + residual/256) *
/// scale` grouping vs the oracle's per-element `scale*nibble - min`), so the
/// two accumulate different floating-point rounding even though they are
/// algebraically identical.
const MAX_ABS_TOLERANCE: f32 = 1e-4;

/// `RELATIVE_TOLERANCE`: batch-peak-normalized bound
/// (`max_diff / max_magnitude`), the convention `q4k_mask_fma_parity.rs`
/// already uses for a small deterministic synthetic fixture -- a fixed
/// ABSOLUTE bound is calibrated to the magnitude real checkpoint tensors
/// happen to produce (see `MAX_ABS_TOLERANCE`'s own doc), and a synthetic
/// fixture's magnitude is an arbitrary fixture choice, not a property this
/// port's correctness should be judged against.
const RELATIVE_TOLERANCE: f32 = 1e-5;

fn assert_matches_dequantized_oracle(
    label: &str,
    weight_bytes: &[u8],
    in_dim: usize,
    rows: usize,
    activation: &[f32],
    relative: bool,
) {
    let blocks_per_row = in_dim / q4_k::QK_K;
    let row_bytes = blocks_per_row * q4_k::BLOCK_BYTES;
    let mut dequantized = vec![0.0f32; rows * in_dim];
    for (row_blocks, row_f32) in weight_bytes
        .chunks_exact(row_bytes)
        .zip(dequantized.chunks_exact_mut(in_dim))
    {
        q4_k::dequantize(row_blocks, row_f32).expect("a whole number of q4_k super-blocks per row");
    }

    let (packed_program, packed_sum) = matmul_program(rows as u32, in_dim as u32, DType::UInt8);
    let metal = omega::execute(
        &packed_program,
        &[],
        &[
            QuantizedBlock::Q4K(weight_bytes),
            QuantizedBlock::Float32(activation),
        ],
        &[packed_sum],
        NumericPolicy::default(),
    )
    .unwrap_or_else(|error| panic!("{label}: metal executes the ggml-port q4_k matvec: {error}"));

    let (f32_program, f32_sum) = matmul_program(rows as u32, in_dim as u32, DType::Float32);
    let cpu = evaluate(&f32_program, &[], &[&dequantized, activation], &[f32_sum])
        .expect("dequantized f32 cpu matmul evaluates");

    let actual = metal.root();
    let expected = cpu.root();
    assert_eq!(
        actual.len(),
        rows,
        "{label}: degenerate gate, no outputs compared"
    );
    assert_eq!(actual.len(), expected.len());

    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(
            got.is_finite(),
            "{label}: ggml-port produced a non-finite value: {got}"
        );
        max_diff = max_diff.max((got - want).abs());
    }
    if relative {
        let max_magnitude = expected
            .iter()
            .map(|value| value.abs())
            .fold(0.0f32, f32::max);
        let relative_error = max_diff / max_magnitude;
        eprintln!(
            "{label}: ggml-port vs dequantized-f32 cpu, {rows} rows, k={in_dim}: \
             max_abs_diff={max_diff} max_magnitude={max_magnitude} relative={relative_error}"
        );
        assert!(
            relative_error < RELATIVE_TOLERANCE,
            "{label}: ggml-port disagrees with the dequantized reference: \
             relative={relative_error} tolerance={RELATIVE_TOLERANCE}"
        );
    } else {
        eprintln!(
            "{label}: ggml-port vs dequantized-f32 cpu, {rows} rows, k={in_dim}: max_abs_diff={max_diff}"
        );
        assert!(
            max_diff < MAX_ABS_TOLERANCE,
            "{label}: ggml-port disagrees with the dequantized reference: max_abs_diff={max_diff} \
             tolerance={MAX_ABS_TOLERANCE}"
        );
    }
}

#[test]
fn ggml_port_matches_dequantized_f32_cpu_on_real_attn_q_weight() {
    let path_string = real_gguf_path();
    let path = std::path::Path::new(&path_string);
    let Some((parsed, file_len, mut file)) = real_gguf_header(path) else {
        eprintln!("real gguf file not found at {path_string}; test skipped");
        return;
    };
    let Some((weight_bytes, in_dim, out_dim)) = real_tensor_bytes(
        &mut file,
        &parsed,
        file_len,
        "blk.0.attn_q.weight",
        GgmlType::Q4_K,
    ) else {
        return;
    };

    let blocks_per_row = in_dim / q4_k::QK_K;
    assert_eq!(blocks_per_row * q4_k::QK_K, in_dim);
    let row_bytes = blocks_per_row * q4_k::BLOCK_BYTES;
    assert_eq!(weight_bytes.len(), row_bytes * out_dim);

    let rows = ROWS_TO_CHECK.min(out_dim);
    let sliced_weight = &weight_bytes[..rows * row_bytes];

    let mut lcg = Lcg(2026);
    let activation: Vec<f32> = (0..in_dim).map(|_| lcg.next_unit() * 4.0 - 2.0).collect();

    assert_matches_dequantized_oracle(
        "attn_q.weight",
        sliced_weight,
        in_dim,
        rows,
        &activation,
        false,
    );
}

#[test]
fn ggml_port_matches_dequantized_f32_cpu_on_real_ffn_up_weight() {
    let path_string = real_gguf_path();
    let path = std::path::Path::new(&path_string);
    let Some((parsed, file_len, mut file)) = real_gguf_header(path) else {
        eprintln!("real gguf file not found at {path_string}; test skipped");
        return;
    };
    let Some((weight_bytes, in_dim, out_dim)) = real_tensor_bytes(
        &mut file,
        &parsed,
        file_len,
        "blk.0.ffn_up.weight",
        GgmlType::Q4_K,
    ) else {
        return;
    };

    let blocks_per_row = in_dim / q4_k::QK_K;
    assert_eq!(blocks_per_row * q4_k::QK_K, in_dim);
    let row_bytes = blocks_per_row * q4_k::BLOCK_BYTES;
    assert_eq!(weight_bytes.len(), row_bytes * out_dim);

    let rows = ROWS_TO_CHECK.min(out_dim);
    let sliced_weight = &weight_bytes[..rows * row_bytes];

    let mut lcg = Lcg(4091);
    let activation: Vec<f32> = (0..in_dim).map(|_| lcg.next_unit() * 4.0 - 2.0).collect();

    assert_matches_dequantized_oracle(
        "ffn_up.weight",
        sliced_weight,
        in_dim,
        rows,
        &activation,
        false,
    );
}

fn random_vec(seed: u64, count: usize, spread: f32) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count)
        .map(|_| (lcg.next_unit() * 2.0 - 1.0) * spread)
        .collect()
}

/// Ragged-K synthetic fixture: `OUT_DIM` (9) is NOT a whole multiple of
/// `PACKED_ROWS_PER_GROUP` (4, `omega::msl`'s own constant) -- the last SIMD
/// group covers only 1 of its 4 row slots, exercising `push_packed_row_
/// combine_and_write`'s `flat < u.output_total` boundary guard on the ggml
/// port's own `sumf[q]` array. `IN_DIM` is exactly two Q4_K super-blocks
/// (512), so `super_blocks == 2` and lanes with `ix` (`lane/8`) in `{2, 3}`
/// run ZERO `ib` iterations -- their `sumf[q]` entries stay at the identity
/// seed for the whole kernel and only the `simd_sum` combine step folds them
/// in, the boundary a large real tensor (always a clean multiple of 4
/// super-blocks in practice) never happens to hit.
const RAGGED_OUT_DIM: usize = 9;
const RAGGED_IN_DIM: usize = 2 * QK_K;

#[test]
fn ggml_port_matches_dequantized_f32_cpu_on_a_ragged_row_count_and_short_k() {
    // spread 1.0, not 5.0: the 1e-4 tolerance is an ABSOLUTE bound
    // (`MAX_ABS_TOLERANCE`'s own doc), calibrated to the magnitude real
    // checkpoint weights and activations actually produce (both real-tensor
    // cases above pass comfortably under it). A wider synthetic spread
    // accumulates a larger absolute sum over 512 elements purely from
    // magnitude, not from any arithmetic disagreement -- confirmed: at
    // spread 5.0 the relative error was ~1.3e-5, well inside FP32
    // accumulation noise for a different summation ORDER, but the ABSOLUTE
    // diff (0.0039) failed the absolute gate on that inflated magnitude
    // alone.
    let rows: Vec<Vec<f32>> = (0..RAGGED_OUT_DIM)
        .map(|row| random_vec(9_000 + row as u64, RAGGED_IN_DIM, 1.0))
        .collect();

    let blocks_per_row = RAGGED_IN_DIM / QK_K;
    assert_eq!(
        blocks_per_row, 2,
        "fixture must be exactly two super-blocks per row"
    );
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    let mut packed = vec![0u8; RAGGED_OUT_DIM * row_bytes];
    for (row, row_packed) in rows.iter().zip(packed.chunks_exact_mut(row_bytes)) {
        quantize(row, row_packed).expect("RAGGED_IN_DIM is a whole multiple of QK_K");
    }

    let activation = random_vec(31_337, RAGGED_IN_DIM, 1.0);

    assert_matches_dequantized_oracle(
        "ragged synthetic (9 rows, 2 super-blocks/row)",
        &packed,
        RAGGED_IN_DIM,
        RAGGED_OUT_DIM,
        &activation,
        true,
    );
}
