//! Device parity for `Q5_0` at a reduce of 896, on the REAL bytes the ollama qwen2.5-0.5b blob
//! (`~/.ollama/models/blobs/sha256-c5396e06af294bd101b30dce59131a76d2b773e76950acc870eda801d3ab0515`)
//! carries for `blk.0.attn_output.weight` (896 x 896) and `blk.0.ffn_gate.weight` (896 x 4864),
//! per guiding-principles section 9.
//!
//! 896 is 28 `Q5_0` blocks and not a whole number of 256-element spans, so before
//! `PackedBodySelect::BlockRendered` these ops ran on the cooperative kernel. This test proves the
//! block-rendered plane body is numerically the dequantized f32 matvec, at the same relative
//! tolerance `q4_0_real_checkpoint_parity.rs` uses, and carries a control: the same device output
//! compared against an oracle whose neighbouring rows are swapped must exceed that tolerance.
//!
//! Skips (does not fail) when the real file is not present on this host, matching
//! `q4_0_real_checkpoint_parity.rs`; once the file is present a tensor that is not `Q5_0` fails.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{Read, Seek, SeekFrom};

use proxima_gguf::parser::{GgufEvent, GgufParser};
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::quant::q5_0;
use proxima_gguf::types::GgmlType;
use proxima_numeric::codec::Codec;
use proxima_numeric::{DType, NumericPolicy};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp, append, evaluate, map};

const REAL_QWEN25_GGUF_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-c5396e06af294bd101b30dce59131a76d2b773e76950acc870eda801d3ab0515";
const TOLERANCE: f32 = 1e-5;

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
                    GgufEvent::Header { version: version_value, .. } => version = Some(version_value),
                    GgufEvent::Metadata { key, value } => metadata.push((key, value)),
                    GgufEvent::Tensor(tensor) => tensors.push(tensor),
                    GgufEvent::Complete { data_offset, alignment } => completion = Some((data_offset, alignment)),
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

fn real_tensor_bytes(file: &mut std::fs::File, parsed: &ParsedGguf, file_len: u64, name: &str) -> (Vec<u8>, usize, usize) {
    let tensor = parsed
        .tensors
        .iter()
        .find(|candidate| candidate.name == name)
        .unwrap_or_else(|| panic!("{name} is in the qwen2.5 blob"));
    assert_eq!(tensor.ggml_type, GgmlType::Q5_0, "{name} is Q5_0 in the qwen2.5 blob");
    let in_dim = tensor.dims[0] as usize;
    let out_dim = tensor.dims[1] as usize;
    let range = parsed.tensor_data_range(tensor, file_len).expect("tensor byte range within file bounds");
    let mut buf = vec![0u8; (range.end - range.start) as usize];
    file.seek(SeekFrom::Start(range.start)).expect("seek to tensor data");
    file.read_exact(&mut buf).expect("read exact tensor byte range");
    (buf, in_dim, out_dim)
}

/// `[rows, k] x [k, 1] -> [rows, 1]`, `weight_dtype` distinguishing packed bytes (`UInt8`) from the
/// dequantized `f32` oracle.
fn matmul_program(rows: u32, k: u32, weight_dtype: DType) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(&mut program, Op::Input { dtype: weight_dtype, shape: vec![Extent::Static(rows), Extent::Static(k)], name: None });
    let activation = append(&mut program, Op::Input { dtype: DType::FLOAT32, shape: vec![Extent::Static(k), Extent::Static(1)], name: None });
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
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
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("q5_0_real_matmul".into()),
        }),
    );
    (program, sum)
}

fn relative_difference(actual: &[f32], expected: &[f32]) -> (f32, f32) {
    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(got.is_finite(), "metal produced a non-finite value: {got}");
        max_diff = max_diff.max((got - want).abs());
    }
    let max_magnitude = expected.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
    (max_diff / max_magnitude, max_diff)
}

fn parity_case(tensor_name: &str, rows_to_check: usize) {
    let path = std::path::Path::new(REAL_QWEN25_GGUF_PATH);
    let Some((parsed, file_len, mut file)) = real_gguf_header(path) else {
        eprintln!("real gguf file not found at {REAL_QWEN25_GGUF_PATH}; test skipped");
        return;
    };
    let (weight_bytes, in_dim, out_dim) = real_tensor_bytes(&mut file, &parsed, file_len, tensor_name);
    assert_eq!(in_dim, 896, "{tensor_name} reduces over the qwen2.5 hidden size, 28 blocks and not a multiple of 256");
    let row_bytes = (in_dim / q5_0::QK5_0) * q5_0::BLOCK_BYTES;
    assert_eq!(weight_bytes.len(), row_bytes * out_dim, "{tensor_name} byte length matches its declared shape");

    let rows = rows_to_check.min(out_dim);
    let sliced_weight = &weight_bytes[..rows * row_bytes];
    let mut lcg = Lcg(2026);
    let activation: Vec<f32> = (0..in_dim).map(|_| lcg.next_unit() * 4.0 - 2.0).collect();

    let mut dequantized = vec![0.0f32; rows * in_dim];
    for (row_blocks, row_f32) in sliced_weight.chunks_exact(row_bytes).zip(dequantized.chunks_exact_mut(in_dim)) {
        q5_0::dequantize(row_blocks, row_f32).expect("a whole number of q5_0 blocks per row");
    }

    let (packed_program, packed_sum) = matmul_program(rows as u32, in_dim as u32, DType::UINT8);
    let metal = omega::execute(
        &packed_program,
        &[],
        &[QuantizedBlock::Packed { codec: Codec::Q5_0, bytes: sliced_weight }, QuantizedBlock::Float32(&activation)],
        &[packed_sum],
        NumericPolicy::default(),
    )
    .expect("metal executes a packed q5_0 matmul on real qwen2.5 bytes");
    let (f32_program, f32_sum) = matmul_program(rows as u32, in_dim as u32, DType::FLOAT32);
    let cpu = evaluate(&f32_program, &[], &[&dequantized, &activation], &[f32_sum]).expect("dequantized f32 cpu matmul evaluates");

    let (actual, expected) = (metal.root(), cpu.root());
    assert_eq!(actual.len(), rows, "degenerate gate: no outputs compared");
    assert_eq!(actual.len(), expected.len());
    let (relative, max_diff) = relative_difference(actual, expected);
    eprintln!("real {tensor_name} (Q5_0, {rows} of {out_dim} rows, k={in_dim}) metal vs dequantized-f32 cpu: max_diff={max_diff} relative={relative}");
    assert!(relative < TOLERANCE, "q5_0 block-rendered body disagrees with the dequantized reference on {tensor_name}: relative={relative} max_diff={max_diff}");

    let mut swapped = expected.to_vec();
    swapped.swap(0, 1);
    let (control_relative, _) = relative_difference(actual, &swapped);
    eprintln!("control: rows 0 and 1 swapped in the oracle, relative={control_relative}");
    assert!(
        control_relative >= TOLERANCE,
        "control must fail: the metal output must not match an oracle with neighbouring rows swapped, relative={control_relative}"
    );
}

#[test]
fn metal_q5_0_attn_output_at_reduce_896_matches_the_dequantized_f32_cpu_path() {
    parity_case("blk.0.attn_output.weight", 64);
}

#[test]
fn metal_q5_0_attn_output_with_a_partial_row_group_matches_the_dequantized_f32_cpu_path() {
    parity_case("blk.0.attn_output.weight", 61);
}

#[test]
fn metal_q5_0_ffn_gate_at_reduce_896_matches_the_dequantized_f32_cpu_path() {
    parity_case("blk.0.ffn_gate.weight", 64);
}
