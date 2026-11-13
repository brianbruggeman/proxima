//! Device parity for the scalar run body (`Fold::Plain`, `PACKED_ROW_SCALAR_RUN` weights per lane)
//! on the REAL F16 rows `Llama-3.2-1B-Instruct-f16.gguf` carries for `blk.0.attn_output.weight`
//! (2048 x 2048) and `blk.0.ffn_gate.weight` (2048 x 8192), per guiding-principles section 9.
//!
//! The oracle is an f64 dot product over the f16-decoded weights and the same f32 activations the
//! device reads, so the only difference the tolerance has to cover is the device's f32
//! accumulation. Each case carries a control: the same device output compared against an oracle
//! whose neighbouring rows are swapped must exceed the tolerance. A negated product must decline
//! the row-blocked body with `BlockBodyNotRendered` (needs `instrument`, which exports the
//! diagnostic seam).
//!
//! Skips (does not fail) when the checkpoint is absent on this host, matching
//! `q5_0_real_checkpoint_parity.rs`; once the file is present a tensor that is not F16 fails.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{Read, Seek, SeekFrom};

use proxima_gguf::parser::{GgufEvent, GgufParser};
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::quant::f16;
use proxima_gguf::types::GgmlType;
use proxima_numeric::codec::Codec;
use proxima_numeric::{DType, NumericPolicy};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp, append, map};

const REAL_LLAMA32_F16_GGUF_PATH: &str = "/Users/brianbruggeman/models/Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct-f16.gguf";
const TOLERANCE: f64 = 1e-6;
const HIDDEN: usize = 2048;

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

fn real_tensor_rows(file: &mut std::fs::File, parsed: &ParsedGguf, file_len: u64, name: &str, rows: usize) -> (Vec<u8>, usize) {
    let tensor = parsed.tensors.iter().find(|candidate| candidate.name == name).unwrap_or_else(|| panic!("{name} is in the llama-3.2 f16 checkpoint"));
    assert_eq!(tensor.ggml_type, GgmlType::F16, "{name} is F16 in the llama-3.2 f16 checkpoint");
    let in_dim = tensor.dims[0] as usize;
    let out_dim = tensor.dims[1] as usize;
    let range = parsed.tensor_data_range(tensor, file_len).expect("tensor byte range within file bounds");
    assert_eq!((range.end - range.start) as usize, in_dim * out_dim * 2, "{name} byte length matches its declared f16 shape");
    let taken = rows.min(out_dim);
    let mut buf = vec![0u8; taken * in_dim * 2];
    file.seek(SeekFrom::Start(range.start)).expect("seek to tensor data");
    file.read_exact(&mut buf).expect("read the leading rows of the tensor");
    (buf, in_dim)
}

fn matmul_program(rows: u32, k: u32, weight_dtype: DType, negated: bool) -> (Vec<Op>, NodeId) {
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
    let reduced = if negated {
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
                body: ScalarOp::Negate,
                operands: vec![(product, IndexMap::Affine(map::projection(3, &[0, 1, 2])))],
                name: None,
            },
        )
    } else {
        product
    };
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: reduced,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("f16_real_matmul".into()),
        }),
    );
    (program, sum)
}

fn relative_difference(actual: &[f32], expected: &[f64]) -> (f64, f64) {
    let mut max_diff = 0.0f64;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(got.is_finite(), "metal produced a non-finite value: {got}");
        max_diff = max_diff.max((f64::from(got) - want).abs());
    }
    let max_magnitude = expected.iter().map(|value| value.abs()).fold(0.0f64, f64::max);
    (max_diff / max_magnitude, max_diff)
}

fn parity_case(tensor_name: &str, rows_to_check: usize) {
    let path = std::path::Path::new(REAL_LLAMA32_F16_GGUF_PATH);
    let Some((parsed, file_len, mut file)) = real_gguf_header(path) else {
        eprintln!("real gguf file not found at {REAL_LLAMA32_F16_GGUF_PATH}; test skipped");
        return;
    };
    let (weight_bytes, in_dim) = real_tensor_rows(&mut file, &parsed, file_len, tensor_name, rows_to_check);
    assert_eq!(in_dim, HIDDEN, "{tensor_name} reduces over the llama-3.2 hidden size");
    let rows = weight_bytes.len() / (in_dim * 2);
    assert_eq!(rows, rows_to_check, "degenerate gate: the requested rows were read");

    let mut lcg = Lcg(2026);
    let activation: Vec<f32> = (0..in_dim).map(|_| lcg.next_unit() * 4.0 - 2.0).collect();
    let mut decoded = vec![0.0f32; rows * in_dim];
    f16::dequantize(&weight_bytes, &mut decoded).expect("a whole run of f16 elements");
    let oracle: Vec<f64> = decoded
        .chunks_exact(in_dim)
        .map(|row| row.iter().zip(&activation).map(|(&weight, &act)| f64::from(weight) * f64::from(act)).sum())
        .collect();

    let (program, sum) = matmul_program(rows as u32, in_dim as u32, DType::FLOAT16, false);
    let metal = omega::execute(
        &program,
        &[],
        &[QuantizedBlock::Packed { codec: Codec::Float16, bytes: &weight_bytes }, QuantizedBlock::Float32(&activation)],
        &[sum],
        NumericPolicy::default(),
    )
    .expect("metal executes an f16 matvec on real llama-3.2 bytes");
    let actual = metal.root();
    assert_eq!(actual.len(), rows, "degenerate gate: no outputs compared");
    let (relative, max_diff) = relative_difference(actual, &oracle);
    eprintln!("real {tensor_name} (F16, {rows} rows, k={in_dim}) metal vs f64 host oracle: max_diff={max_diff} relative={relative}");
    assert!(relative < TOLERANCE, "f16 scalar run body disagrees with the f64 oracle on {tensor_name}: relative={relative} max_diff={max_diff}");

    let mut swapped = oracle.clone();
    swapped.swap(0, 1);
    let (control_relative, _) = relative_difference(actual, &swapped);
    eprintln!("control: rows 0 and 1 swapped in the oracle, relative={control_relative}");
    assert!(control_relative >= TOLERANCE, "control must fail: the output must not match an oracle with neighbouring rows swapped, relative={control_relative}");
}

#[test]
fn metal_f16_attn_output_at_reduce_2048_matches_the_f64_host_oracle() {
    parity_case("blk.0.attn_output.weight", 64);
}

#[test]
fn metal_f16_attn_output_with_a_partial_row_group_matches_the_f64_host_oracle() {
    parity_case("blk.0.attn_output.weight", 61);
}

#[test]
fn metal_f16_ffn_gate_at_reduce_2048_matches_the_f64_host_oracle() {
    parity_case("blk.0.ffn_gate.weight", 64);
}

#[test]
fn metal_f16_ffn_gate_with_a_partial_row_group_matches_the_f64_host_oracle() {
    parity_case("blk.0.ffn_gate.weight", 61);
}

#[cfg(feature = "instrument")]
#[test]
fn a_negated_f16_product_declines_the_row_blocked_body() {
    use omega::msl::PackedRowBlockRejection;

    let (program, sum) = matmul_program(24, HIDDEN as u32, DType::FLOAT16, true);
    let shapes = proxima_tensor::infer(&program, &[]).expect("the synthetic program infers");
    let mut bound = proxima_tensor::bind(&program, &shapes, &[sum], NumericPolicy::default()).expect("binds");
    proxima_tensor::correct_packed_matmul_layouts(&mut bound, &[NodeId(0)].into_iter().collect());
    let resolved = bound.iter().find(|op| op.node == sum).expect("the reduce node is bound");

    let rejection = omega::msl::diagnose_packed_row_block(resolved, &[Some(Codec::Float16), None]);
    assert_eq!(rejection, Err(PackedRowBlockRejection::BlockBodyNotRendered));
}

#[cfg(feature = "instrument")]
#[test]
fn a_plain_f16_product_at_reduce_2048_takes_the_row_blocked_body() {
    let (program, sum) = matmul_program(24, HIDDEN as u32, DType::FLOAT16, false);
    let shapes = proxima_tensor::infer(&program, &[]).expect("the synthetic program infers");
    let mut bound = proxima_tensor::bind(&program, &shapes, &[sum], NumericPolicy::default()).expect("binds");
    proxima_tensor::correct_packed_matmul_layouts(&mut bound, &[NodeId(0)].into_iter().collect());
    let resolved = bound.iter().find(|op| op.node == sum).expect("the reduce node is bound");

    assert_eq!(omega::msl::diagnose_packed_row_block(resolved, &[Some(Codec::Float16), None]), Ok(()));
}
