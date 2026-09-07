//! Device parity for `Q3_K`, on the REAL bytes `blk.0.ffn_up.weight` carries
//! in a real `Q3_K_M` requantized checkpoint -- not synthetic, per
//! guiding-principles §9. Same posture as `q4k_real_checkpoint_parity.rs`/
//! `q5k_real_checkpoint_parity.rs`/`q6k_real_checkpoint_parity.rs`: metal's
//! packed `Q3_K` matmul (`msl.rs`'s `Q3K_UNPACK_MSL`/`Q3K_PAIR_DOT_MSL`,
//! selected by `push_packed_row_blocked_body`'s `PackedCodec::Q3K` arm)
//! against `proxima_gguf::quant::q3_k::dequantize` + the CPU interpreter --
//! never against a Python oracle (that comparison lives in
//! `proxima-gguf/src/quant/q3_k.rs`'s own `#[ignore]`d
//! `q3_k_real_dequantize_matches_llama_cpp_gguf_py_oracle`).
//!
//! Skips (does not fail) when the real file is not present on this host --
//! matching the sibling real-checkpoint tests' own posture. The checkpoint
//! path is overridable via `PROXIMA_Q3K_GGUF`.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{Read, Seek, SeekFrom};

use proxima_gguf::parser::{GgufEvent, GgufParser};
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::quant::q3_k;
use proxima_gguf::types::GgmlType;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce, ReduceInit,
    ScalarOp, append, evaluate, map,
};

/// Real `Q3_K_M` GGUF checkpoint path, overridable via `PROXIMA_Q3K_GGUF` --
/// the hardcoded default only ever resolves on the machine it was captured
/// on, same shape `q4k_real_checkpoint_parity.rs`'s own
/// `PROXIMA_BENCH_GGUF_PATH` takes.
fn real_gguf_path() -> String {
    std::env::var("PROXIMA_Q3K_GGUF").unwrap_or_else(|_| {
        "/private/tmp/claude-501/-Users-brianbruggeman-repos-slot-0/6e203711-bd50-48cc-9ade-409668bdafdd/scratchpad/models/openchat-3.5-1210.Q3_K_M.gguf"
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

/// `[rows, k] x [k, 1] -> [rows, 1]`, same shape the sibling real-checkpoint
/// parity tests build.
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
            name: Some("q3k_real_matmul".into()),
        }),
    );
    (program, sum)
}

/// Running the full row count on a device parity test is unnecessary to
/// prove the unpack is bit-correct, so this takes a prefix of rows (each
/// row's packed bytes are contiguous, so a byte-level prefix slice is
/// exactly a row-count prefix) -- same posture the sibling real-checkpoint
/// tests take.
const ROWS_TO_CHECK: usize = 64;

/// `push_packed_row_blocked_body`'s `Q3_K` `plain_product` arm
/// (`q3k_pair_dot`) is the DEFAULT path for a plain weight*activation
/// matmul, unlike `Q5_K`'s pair-dot which needs `metal-q5k-pair-dot` -- see
/// `Q3K_PAIR_DOT_MSL`'s own doc for why no new Cargo feature was needed.
/// 1e-4 tolerance mirrors `Q5_K`'s own paired-body test, one order of
/// magnitude looser than a scalar per-element loop's 1e-5: the paired body
/// accumulates across four interleaved sub-block partials rather than one
/// running sum, expected floating-point reordering, not a correctness gap.
#[test]
fn metal_matmul_on_real_ffn_up_q3k_bytes_matches_the_dequantized_f32_cpu_path() {
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
        GgmlType::Q3_K,
    ) else {
        return;
    };

    let blocks_per_row = in_dim / q3_k::QK_K;
    assert_eq!(
        blocks_per_row * q3_k::QK_K,
        in_dim,
        "blk.0.ffn_up.weight's in_dim is a whole number of Q3_K super-blocks"
    );
    let row_bytes = blocks_per_row * q3_k::BLOCK_BYTES;
    assert_eq!(
        weight_bytes.len(),
        row_bytes * out_dim,
        "blk.0.ffn_up.weight byte length matches its declared shape"
    );

    let rows = ROWS_TO_CHECK.min(out_dim);
    let sliced_weight = &weight_bytes[..rows * row_bytes];

    let mut lcg = Lcg(2026);
    let activation: Vec<f32> = (0..in_dim).map(|_| lcg.next_unit() * 4.0 - 2.0).collect();

    let mut dequantized = vec![0.0f32; rows * in_dim];
    for (row_blocks, row_f32) in sliced_weight
        .chunks_exact(row_bytes)
        .zip(dequantized.chunks_exact_mut(in_dim))
    {
        q3_k::dequantize(row_blocks, row_f32).expect("a whole number of q3_k super-blocks per row");
    }

    let (packed_program, packed_sum) = matmul_program(rows as u32, in_dim as u32, DType::UInt8);
    let metal = omega::execute(
        &packed_program,
        &[],
        &[
            QuantizedBlock::Q3K(sliced_weight),
            QuantizedBlock::Float32(&activation),
        ],
        &[packed_sum],
        NumericPolicy::default(),
    )
    .expect("metal executes a packed q3_k matmul on real blk.0.ffn_up.weight bytes");

    let (f32_program, f32_sum) = matmul_program(rows as u32, in_dim as u32, DType::Float32);
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
    let relative = max_diff / max_magnitude;
    eprintln!(
        "real blk.0.ffn_up.weight (Q3_K, {rows} of {out_dim} rows, k={in_dim}) metal vs dequantized-f32 cpu: \
         max_diff={max_diff} max_magnitude={max_magnitude} relative={relative}"
    );
    assert!(
        relative < 1e-4,
        "packed unpack disagrees with the dequantized reference on REAL checkpoint bytes: \
         relative={relative} max_diff={max_diff}"
    );
}
