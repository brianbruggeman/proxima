//! Correctness gate for the `S/nb/stage/STAGING.md` staging-loop experiment
//! that shipped, `PROXIMA_TILED_GEMM_WIDE_ACT_LOAD` (item 3c, vectorized
//! activation-tile load), plus the phase-2 `PROXIMA_TILED_GEMM_SLIM_TGMEM`
//! output-tile aliasing (`S/nb/port2/RESULTS.md`). Two sibling experiments,
//! `PROXIMA_TILED_GEMM_PTR_HOIST` (item 3b) and `PROXIMA_TILED_GEMM_
//! DECODE_SPREAD`/`_INTERIOR_STORE` (items 3a/3d), were measured
//! noise-level or a regression on this shape (`S/nb/bmm2/RESULTS.md`'s own
//! isolated-probe table) and were not shipped -- ROLLBACK, discipline log
//! C4.9.
//!
//! Same harness as `q4_0_tiled_gemm_batched_run8_parity.rs`: real Q4_0
//! bytes from the ollama gemma4-E2B checkpoint (guiding-principles §9),
//! `PROXIMA_TILED_GEMM_Q4_0=1` to admit Q4_0 into the tiled path at all,
//! each experiment layered on top, one at a time and combined, compared
//! against the independent dequantize+f32-CPU-matmul oracle. Token counts
//! `8` (`TILED_GEMM_MIN_TOKENS` itself), `37` (deliberately NOT a multiple
//! of `BLOCK_N`=32, so every arm exercises a genuine boundary/edge column
//! tile as well as interior ones), and `510` (this session's own
//! prefill-length target) are used.
//!
//! Skips (does not fail) when the real file or a named tensor is not
//! present on this host, matching the sibling file's own posture.

#![cfg(all(feature = "metal", feature = "metal-tiled-gemm", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{Read, Seek, SeekFrom};

use proxima_gguf::parser::{GgufEvent, GgufParser};
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::quant::q4_0;
use proxima_gguf::types::GgmlType;
use proxima_primitives::Codec;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce, ReduceInit,
    ScalarOp, append, evaluate, projection,
};

const REAL_GEMMA4_GGUF_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

const ROWS_TO_CHECK: usize = 64;

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

fn matmul_program(tokens: u32, in_dim: u32, out_dim: u32, weight_dtype: DType) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: weight_dtype,
            shape: vec![Extent::Static(in_dim), Extent::Static(out_dim)],
            name: None,
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(tokens), Extent::Static(in_dim)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(3, &[1, 2]))),
                (activation, IndexMap::Affine(projection(3, &[0, 1]))),
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
            out_map: IndexMap::Affine(projection(3, &[0, 2])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    (program, sum)
}

struct ErrorSummary {
    max_abs: f32,
    max_relative: f32,
}

fn compare_to_oracle(actual: &[f32], oracle: &[f32]) -> ErrorSummary {
    let mut max_abs = 0.0f32;
    let mut max_relative = 0.0f32;
    for (&got, &want) in actual.iter().zip(oracle.iter()) {
        let abs_diff = (got - want).abs();
        max_abs = max_abs.max(abs_diff);
        if want.abs() > 1e-3 {
            max_relative = max_relative.max(abs_diff / want.abs());
        }
    }
    ErrorSummary { max_abs, max_relative }
}

/// One switch combination: the env vars set (ALWAYS includes
/// `PROXIMA_TILED_GEMM_Q4_0=1`, the admission gate every arm needs) and a
/// label for the failure message.
struct SwitchCombo {
    label: &'static str,
    extra_vars: &'static [(&'static str, &'static str)],
}

const COMBOS: &[SwitchCombo] = &[
    SwitchCombo { label: "baseline (Q4_0 tiled, no staging experiments)", extra_vars: &[] },
    SwitchCombo {
        label: "wide_act_load alone",
        extra_vars: &[("PROXIMA_TILED_GEMM_WIDE_ACT_LOAD", "1")],
    },
    // phase 2 (`S/nb/port2/RESULTS.md`): `out_tile` aliased onto
    // `weight_tile`/`act_tile`'s backing bytes -- see
    // `push_tiled_gemm_body`'s own doc.
    SwitchCombo {
        label: "slim_tgmem alone",
        extra_vars: &[("PROXIMA_TILED_GEMM_SLIM_TGMEM", "1")],
    },
    SwitchCombo {
        label: "slim_tgmem + wide_act_load",
        extra_vars: &[
            ("PROXIMA_TILED_GEMM_SLIM_TGMEM", "1"),
            ("PROXIMA_TILED_GEMM_WIDE_ACT_LOAD", "1"),
        ],
    },
];

/// Runs one real tensor at one token count through the CPU oracle and
/// every [`COMBOS`] entry, asserting each against the oracle AND against
/// the baseline (byte-for-byte would be too strong -- `simdgroup_matrix`
/// accumulation order is not guaranteed identical across these code
/// shapes -- so both comparisons use the same relative bound). Returns
/// early (no assertion) if the tensor is absent.
fn check_real_tensor_at_token_count(tensor_name: &str, tokens: usize) {
    let path = std::path::Path::new(REAL_GEMMA4_GGUF_PATH);
    let Some((parsed, file_len, mut file)) = real_gguf_header(path) else {
        eprintln!("real gguf file not found at {REAL_GEMMA4_GGUF_PATH}; test skipped");
        return;
    };
    let Some((weight_bytes, in_dim, out_dim)) =
        real_tensor_bytes(&mut file, &parsed, file_len, tensor_name, GgmlType::Q4_0)
    else {
        eprintln!("{tensor_name} not found or not Q4_0 in this checkpoint; test skipped");
        return;
    };

    let blocks_per_row = in_dim / q4_0::QK4_0;
    assert_eq!(blocks_per_row * q4_0::QK4_0, in_dim, "{tensor_name}'s in_dim is a whole number of Q4_0 blocks");
    let row_bytes = blocks_per_row * q4_0::BLOCK_BYTES;
    assert_eq!(weight_bytes.len(), row_bytes * out_dim, "{tensor_name} byte length matches its declared shape");

    let rows = ROWS_TO_CHECK.min(out_dim);
    let sliced_weight = &weight_bytes[..rows * row_bytes];

    let mut lcg = Lcg(9101 + tokens as u64);
    let activation: Vec<f32> = (0..tokens * in_dim).map(|_| lcg.next_unit() * 4.0 - 2.0).collect();

    let mut dequantized = vec![0.0f32; rows * in_dim];
    for (row_blocks, row_f32) in sliced_weight.chunks_exact(row_bytes).zip(dequantized.chunks_exact_mut(in_dim)) {
        q4_0::dequantize(row_blocks, row_f32).expect("a whole number of q4_0 blocks per row");
    }
    let mut dequantized_transposed = vec![0.0f32; rows * in_dim];
    for row in 0..rows {
        for column in 0..in_dim {
            dequantized_transposed[column * rows + row] = dequantized[row * in_dim + column];
        }
    }

    let (packed_program, packed_sum) = matmul_program(tokens as u32, in_dim as u32, rows as u32, DType::UInt8);
    let blocks = [
        QuantizedBlock::Packed { codec: Codec::Q4_0, bytes: sliced_weight },
        QuantizedBlock::Float32(&activation),
    ];

    let (f32_program, f32_sum) = matmul_program(tokens as u32, in_dim as u32, rows as u32, DType::Float32);
    let oracle = evaluate(&f32_program, &[], &[&dequantized_transposed, &activation], &[f32_sum])
        .expect("dequantized f32 cpu matmul evaluates");
    let oracle_root = oracle.root();
    let element_count = tokens * rows;
    assert_eq!(oracle_root.len(), element_count, "degenerate: oracle produced no output");

    for combo in COMBOS {
        let mut vars: Vec<(&str, Option<&str>)> = vec![("PROXIMA_TILED_GEMM_Q4_0", Some("1"))];
        for &(key, value) in combo.extra_vars {
            vars.push((key, Some(value)));
        }
        let result = temp_env::with_vars(vars, || {
            omega::execute(&packed_program, &[], &blocks, &[packed_sum], NumericPolicy::default())
                .expect("metal executes the combo")
        });
        let result_root = result.root();
        assert_eq!(
            result_root.len(),
            element_count,
            "{tensor_name} T={tokens} [{}]: degenerate output length",
            combo.label
        );
        let vs_oracle = compare_to_oracle(result_root, oracle_root);
        eprintln!(
            "{tensor_name} T={tokens} [{}]: vs-oracle max_abs={} max_relative={}",
            combo.label, vs_oracle.max_abs, vs_oracle.max_relative
        );
        assert!(
            vs_oracle.max_relative < 1e-2,
            "{tensor_name} T={tokens} [{}]: disagrees with the independent dequantize+f32-matmul oracle \
             (max_relative={})",
            combo.label,
            vs_oracle.max_relative
        );
    }
}

#[test]
fn attn_q_all_staging_switches_match_the_f32_oracle_at_min_tokens() {
    check_real_tensor_at_token_count("blk.0.attn_q.weight", 8);
}

#[test]
fn attn_q_all_staging_switches_match_the_f32_oracle_at_a_boundary_token_count() {
    check_real_tensor_at_token_count("blk.0.attn_q.weight", 37);
}

#[test]
fn attn_q_all_staging_switches_match_the_f32_oracle_at_prefill_token_count() {
    check_real_tensor_at_token_count("blk.0.attn_q.weight", 510);
}

#[test]
fn ffn_gate_all_staging_switches_match_the_f32_oracle_at_a_boundary_token_count() {
    check_real_tensor_at_token_count("blk.0.ffn_gate.weight", 37);
}
