//! Device parity for `Float16`, on the REAL bytes `blk.0.ffn_gate_inp.weight`
//! carries in `Nous-Hermes-2-Mixtral-8x7B-DPO.Q4_K_S.gguf` -- not synthetic,
//! per guiding-principles §9. This checkpoint's own quantizer
//! (`proxima_gguf::quant::policy::PrecisionPolicy::llama_cpp_q4_k_s_moe_8_expert`)
//! leaves every one of the 32 MoE router weights (`blk.{0..31}.ffn_gate_inp.weight`)
//! at `F16` rather than quantizing them -- a real deployment case this
//! session exists to unblock: without a Metal `Float16` path, this Mixtral
//! checkpoint cannot run on the GPU at all.
//!
//! `Float16` is not a quantization: MSL's `half` is IEEE-754 binary16
//! natively, so the weight's on-disk bytes bind directly as a
//! `device const half*` buffer with no unpack kernel (see
//! `omega::msl::FLOAT16_BLOCK_BYTES`'s own doc) -- this test proves that
//! direct-bind path is bit-correct against real checkpoint bytes, the same
//! contract `q8_0_real_checkpoint_parity.rs` proves for its own codec.
//!
//! Skips (does not fail) when the real file is not present on this host --
//! matching every other `*_real_checkpoint_parity.rs` test in this
//! workspace's posture.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{Read, Seek, SeekFrom};

use proxima_gguf::parser::{GgufEvent, GgufParser};
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::quant::f16;
use proxima_gguf::types::GgmlType;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp, append, evaluate, map,
};
use proxima_tensor::test_support::Lcg;

const REAL_MIXTRAL_GGUF_PATH: &str =
    "/Users/brianbruggeman/.lmstudio/models/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF/Nous-Hermes-2-Mixtral-8x7B-DPO.Q4_K_S.gguf";

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
                    GgufEvent::Complete { data_offset, alignment } => {
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
    let tensor = parsed.tensors.iter().find(|candidate| candidate.name == name)?;
    if tensor.ggml_type != expect_type {
        eprintln!(
            "real_tensor_bytes: {name} is {:?} in this file, not {expect_type:?} -- test skipped, not faked",
            tensor.ggml_type
        );
        return None;
    }
    let in_dim = tensor.dims[0] as usize;
    let out_dim = tensor.dims[1] as usize;
    let range = parsed.tensor_data_range(tensor, file_len).expect("tensor byte range within file bounds");
    let mut buf = vec![0u8; (range.end - range.start) as usize];
    file.seek(SeekFrom::Start(range.start)).expect("seek to tensor data");
    file.read_exact(&mut buf).expect("read exact tensor byte range");
    Some((buf, in_dim, out_dim))
}

/// `[rows, k] x [k, 1] -> [rows, 1]`, `weight_dtype` distinguishing "packed
/// bytes" (`Float16`) from the dequantized `f32` oracle -- same shape
/// `q8_0_real_checkpoint_parity.rs`'s own `matmul_program` builds, restated
/// here since this is a standalone integration test binary.
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
            name: Some("f16_real_matmul".into()),
        }),
    );
    (program, sum)
}

#[test]
fn metal_matmul_on_real_ffn_gate_inp_f16_bytes_matches_the_dequantized_f32_cpu_path() {
    let path = std::path::Path::new(REAL_MIXTRAL_GGUF_PATH);
    let Some((parsed, file_len, mut file)) = real_gguf_header(path) else {
        eprintln!("real gguf file not found at {REAL_MIXTRAL_GGUF_PATH}; test skipped");
        return;
    };
    let Some((weight_bytes, in_dim, out_dim)) =
        real_tensor_bytes(&mut file, &parsed, file_len, "blk.0.ffn_gate_inp.weight", GgmlType::F16)
    else {
        return;
    };

    assert_eq!(weight_bytes.len(), in_dim * out_dim * 2, "blk.0.ffn_gate_inp.weight byte length matches its declared f16 shape");

    let mut lcg = Lcg(2026);
    let activation: Vec<f32> = (0..in_dim).map(|_| lcg.next_unit() * 4.0 - 2.0).collect();

    let mut dequantized = vec![0.0f32; out_dim * in_dim];
    f16::dequantize(&weight_bytes, &mut dequantized).expect("blk.0.ffn_gate_inp.weight decodes as a whole run of f16 elements");

    let (packed_program, packed_sum) = matmul_program(out_dim as u32, in_dim as u32, DType::Float16);
    let metal = omega::execute(
        &packed_program,
        &[],
        &[QuantizedBlock::Float16(&weight_bytes), QuantizedBlock::Float32(&activation)],
        &[packed_sum],
    )
    .expect("metal executes an f16 matmul on real blk.0.ffn_gate_inp.weight bytes");

    let (f32_program, f32_sum) = matmul_program(out_dim as u32, in_dim as u32, DType::Float32);
    let cpu =
        evaluate(&f32_program, &[], &[&dequantized, &activation], &[f32_sum]).expect("dequantized f32 cpu matmul evaluates");

    let actual = metal.root();
    let expected = cpu.root();
    assert_eq!(actual.len(), out_dim, "degenerate gate: no outputs compared");
    assert_eq!(actual.len(), expected.len());

    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(got.is_finite(), "metal produced a non-finite value: {got}");
        max_diff = max_diff.max((got - want).abs());
    }
    let max_magnitude = expected.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
    let relative = if max_magnitude > 0.0 { max_diff / max_magnitude } else { max_diff };
    eprintln!(
        "real blk.0.ffn_gate_inp.weight (F16, {out_dim}x{in_dim}) metal vs dequantized-f32 cpu: \
         max_diff={max_diff} max_magnitude={max_magnitude} relative={relative}"
    );
    assert!(
        relative < 1e-5,
        "f16 direct-bind disagrees with the dequantized reference on REAL checkpoint bytes: \
         relative={relative} max_diff={max_diff}"
    );
}
