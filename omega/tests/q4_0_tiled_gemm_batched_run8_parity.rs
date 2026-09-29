//! Correctness gate for `PROXIMA_TILED_GEMM_Q4_0`'s batched `q4_0_run8`
//! weight-tile decode (`omega::msl::Q4_0_RUN8_MSL`), on REAL bytes from the
//! ollama gemma4-E2B blob (per guiding-principles §9, same file
//! `q4_0_real_checkpoint_parity.rs` already reads). Two real weight
//! tensors: `blk.0.attn_q.weight` (the attention Q projection) and
//! `blk.0.ffn_gate.weight` (the FFN gate, `in_dim=1536` -> ffn width), at
//! `T = 8, 64, 510` tokens -- `8` is `TILED_GEMM_MIN_TOKENS` itself (the
//! boundary the tiled path's admission gate opens at), `510` is this
//! session's own prefill-length target.
//!
//! Compares THREE things per shape: the new batched-`q4_0_run8` tiled path
//! (`PROXIMA_TILED_GEMM_Q4_0` at its default, unset -- admits since this
//! switch flipped default ON) against (a) today's row-blocked path (the
//! switch explicitly `"0"` -- `Codec::Q4_0` only takes the tiled path when
//! this override is off) and (b) an f32 CPU reference built by dequantizing
//! the real packed rows and
//! running `proxima_tensor::cpu::evaluate`'s plain f32 matmul -- the same
//! independent-oracle discipline `attn_multi_axis_tiled_gemm_parity.rs` and
//! `q4k_matmul_layout.rs` both hold their own tiled/row-blocked arms to.
//!
//! Skips (does not fail) when the real file or a named tensor is not present
//! on this host, matching `q4_0_real_checkpoint_parity.rs`'s own posture.

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

// Real weight rows compared per tensor -- bounded the same way
// `q4_0_real_checkpoint_parity.rs::ROWS_TO_CHECK` bounds its own device
// parity gate, so a T=510 run stays a reasonable single-test runtime.
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

/// `[tokens, in_dim] x [in_dim, out_dim] -> [tokens, out_dim]`, weight
/// declared reduction-axis-first (`q4k_matmul_layout.rs`'s own convention,
/// the real spec's declared order) with an explicit `tokens` axis so
/// `classify_tiled_gemm`'s `TILED_GEMM_MIN_TOKENS` gate can actually open.
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
    // iteration space (tok, in, out): weight reads (in, out), activation
    // reads (tok, in).
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
    ErrorSummary {
        max_abs,
        max_relative,
    }
}

fn differing_word_count(a: &[f32], b: &[f32]) -> (usize, f32) {
    let mut count = 0usize;
    let mut max_abs = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if x.to_bits() != y.to_bits() {
            count += 1;
        }
        max_abs = max_abs.max((x - y).abs());
    }
    (count, max_abs)
}

/// Runs one real tensor at one token count through: the new batched
/// `q4_0_run8` tiled path, today's row-blocked path, and the f32 CPU
/// oracle. Returns early (no assertion) if the tensor is absent, matching
/// this file's own real-checkpoint skip posture.
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
    assert_eq!(
        blocks_per_row * q4_0::QK4_0,
        in_dim,
        "{tensor_name}'s in_dim is a whole number of Q4_0 blocks"
    );
    let row_bytes = blocks_per_row * q4_0::BLOCK_BYTES;
    assert_eq!(
        weight_bytes.len(),
        row_bytes * out_dim,
        "{tensor_name} byte length matches its declared shape"
    );

    let rows = ROWS_TO_CHECK.min(out_dim);
    let sliced_weight = &weight_bytes[..rows * row_bytes];

    let mut lcg = Lcg(2026 + tokens as u64);
    let activation: Vec<f32> = (0..tokens * in_dim)
        .map(|_| lcg.next_unit() * 4.0 - 2.0)
        .collect();

    let mut dequantized = vec![0.0f32; rows * in_dim];
    for (row_blocks, row_f32) in sliced_weight
        .chunks_exact(row_bytes)
        .zip(dequantized.chunks_exact_mut(in_dim))
    {
        q4_0::dequantize(row_blocks, row_f32).expect("a whole number of q4_0 blocks per row");
    }
    // dequantized is [rows, in_dim] row-major; matmul_program declares the
    // weight [in_dim, out_dim] (reduction-axis-first), so transpose here.
    let mut dequantized_transposed = vec![0.0f32; rows * in_dim];
    for row in 0..rows {
        for column in 0..in_dim {
            dequantized_transposed[column * rows + row] = dequantized[row * in_dim + column];
        }
    }

    let (packed_program, packed_sum) =
        matmul_program(tokens as u32, in_dim as u32, rows as u32, DType::UInt8);
    let blocks = [
        QuantizedBlock::Packed { codec: Codec::Q4_0, bytes: sliced_weight },
        QuantizedBlock::Float32(&activation),
    ];

    let current = temp_env::with_var("PROXIMA_TILED_GEMM_Q4_0", Some("0"), || {
        omega::execute(
            &packed_program,
            &[],
            &blocks,
            &[packed_sum],
            NumericPolicy::default(),
        )
        .expect("metal executes today's row-blocked q4_0 matmul")
    });

    let new_tiled = omega::execute(
        &packed_program,
        &[],
        &blocks,
        &[packed_sum],
        NumericPolicy::default(),
    )
    .expect("metal executes the new batched-q4_0_run8 tiled matmul (default on, unset)");

    let (f32_program, f32_sum) =
        matmul_program(tokens as u32, in_dim as u32, rows as u32, DType::Float32);
    let oracle = evaluate(
        &f32_program,
        &[],
        &[&dequantized_transposed, &activation],
        &[f32_sum],
    )
    .expect("dequantized f32 cpu matmul evaluates");

    let current_root = current.root();
    let new_root = new_tiled.root();
    let oracle_root = oracle.root();
    let element_count = tokens * rows;
    assert_eq!(current_root.len(), element_count, "degenerate: current produced no output");
    assert_eq!(new_root.len(), element_count, "degenerate: new tiled path produced no output");
    assert_eq!(oracle_root.len(), element_count, "degenerate: oracle produced no output");

    let (diff_words, max_abs_vs_current) = differing_word_count(new_root, current_root);
    let new_vs_oracle = compare_to_oracle(new_root, oracle_root);
    let current_vs_oracle = compare_to_oracle(current_root, oracle_root);

    eprintln!(
        "{tensor_name} T={tokens} rows={rows} in_dim={in_dim}: \
         new-vs-current differing_words={diff_words}/{element_count} max_abs_diff={max_abs_vs_current} \
         new-vs-oracle max_abs={} max_relative={} \
         current-vs-oracle max_abs={} max_relative={}",
        new_vs_oracle.max_abs,
        new_vs_oracle.max_relative,
        current_vs_oracle.max_abs,
        current_vs_oracle.max_relative,
    );

    // both paths dequantize the SAME real Q4_0 bytes and differ from the f32
    // oracle only in fp32 summation order (tiled `simdgroup_matrix` reduction
    // vs row-blocked `q4k_run8`-style pairwise accumulation) -- not a second
    // lossy quantization step (contrast `q4k_matmul_layout.rs`'s CPU int8-dot
    // arm, which needs a much wider bound for that reason). 1% relative on
    // the far side of a 1e-3 magnitude floor comfortably covers fp32
    // reduction-order drift over `in_dim=1536` terms.
    assert!(
        new_vs_oracle.max_relative < 1e-2,
        "{tensor_name} T={tokens}: new batched-q4_0_run8 tiled path disagrees with the independent \
         dequantize+f32-matmul oracle (max_relative={})",
        new_vs_oracle.max_relative
    );
    assert!(
        current_vs_oracle.max_relative < 1e-2,
        "{tensor_name} T={tokens}: today's row-blocked path disagrees with the independent \
         dequantize+f32-matmul oracle (max_relative={}) -- the oracle itself is suspect if this fires",
        current_vs_oracle.max_relative
    );
}

/// Same real weight/activation as [`check_real_tensor_at_token_count`], but
/// with a fused, GENERIC (non-broadcast) epilogue appended on top of the
/// matmul's output -- an elementwise multiply against a second per-element
/// buffer, the same "extra operand read + combine" shape a real fused GeGLU
/// `gate * up` epilogue takes, without hard-coding GeGLU's own nonlinearity
/// (`push_reduce_epilogue_write`'s own doc: the emitter is generic over the
/// composed body, never a per-shape special case). Exercises
/// `push_tiled_gemm_body`'s new epilogue write tail end to end on real Metal
/// hardware, not just the no-device admission gate
/// [`msl::tests::q4_0_broadcast_epilogue_declines_tiled_gemm_admission_without_erroring`]
/// drives in `omega`'s own unit tests.
fn check_real_tensor_epilogue_at_token_count(tensor_name: &str, tokens: usize) {
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
    let row_bytes = blocks_per_row * q4_0::BLOCK_BYTES;
    let rows = ROWS_TO_CHECK.min(out_dim);
    let sliced_weight = &weight_bytes[..rows * row_bytes];

    let mut lcg = Lcg(4052 + tokens as u64);
    let activation: Vec<f32> = (0..tokens * in_dim)
        .map(|_| lcg.next_unit() * 4.0 - 2.0)
        .collect();
    let epilogue_scale: Vec<f32> = (0..tokens * rows)
        .map(|_| lcg.next_unit() * 2.0 - 1.0)
        .collect();

    let mut dequantized = vec![0.0f32; rows * in_dim];
    for (row_blocks, row_f32) in sliced_weight
        .chunks_exact(row_bytes)
        .zip(dequantized.chunks_exact_mut(in_dim))
    {
        q4_0::dequantize(row_blocks, row_f32).expect("a whole number of q4_0 blocks per row");
    }
    let mut dequantized_transposed = vec![0.0f32; rows * in_dim];
    for row in 0..rows {
        for column in 0..in_dim {
            dequantized_transposed[column * rows + row] = dequantized[row * in_dim + column];
        }
    }

    let append_epilogue = |program: &mut Vec<Op>, matmul_out: NodeId, dtype: DType| -> NodeId {
        let scale = append(
            program,
            Op::Input {
                dtype,
                shape: vec![Extent::Static(tokens as u32), Extent::Static(rows as u32)],
                name: None,
            },
        );
        append(
            program,
            Op::Elementwise {
                dtype,
                body: ScalarOp::Multiply,
                operands: vec![
                    (matmul_out, IndexMap::Affine(projection(2, &[0, 1]))),
                    (scale, IndexMap::Affine(projection(2, &[0, 1]))),
                ],
                name: None,
            },
        )
    };

    let (mut packed_program, packed_sum) =
        matmul_program(tokens as u32, in_dim as u32, rows as u32, DType::UInt8);
    let packed_epilogue = append_epilogue(&mut packed_program, packed_sum, DType::Float32);
    let blocks = [
        QuantizedBlock::Packed { codec: Codec::Q4_0, bytes: sliced_weight },
        QuantizedBlock::Float32(&activation),
        QuantizedBlock::Float32(&epilogue_scale),
    ];

    let current = temp_env::with_var("PROXIMA_TILED_GEMM_Q4_0", Some("0"), || {
        omega::execute(
            &packed_program,
            &[],
            &blocks,
            &[packed_epilogue],
            NumericPolicy::default(),
        )
        .expect("metal executes today's row-blocked q4_0 matmul with a fused epilogue")
    });

    let new_tiled = omega::execute(
        &packed_program,
        &[],
        &blocks,
        &[packed_epilogue],
        NumericPolicy::default(),
    )
    .expect(
        "metal executes the new batched-q4_0_run8 tiled matmul with a fused epilogue (default \
         on, unset)",
    );

    let (mut f32_program, f32_sum) =
        matmul_program(tokens as u32, in_dim as u32, rows as u32, DType::Float32);
    let f32_epilogue = append_epilogue(&mut f32_program, f32_sum, DType::Float32);
    let oracle = evaluate(
        &f32_program,
        &[],
        &[&dequantized_transposed, &activation, &epilogue_scale],
        &[f32_epilogue],
    )
    .expect("dequantized f32 cpu matmul with epilogue evaluates");

    let current_root = current.root();
    let new_root = new_tiled.root();
    let oracle_root = oracle.root();
    let element_count = tokens * rows;
    assert_eq!(current_root.len(), element_count, "degenerate: current produced no output");
    assert_eq!(new_root.len(), element_count, "degenerate: new tiled path produced no output");
    assert_eq!(oracle_root.len(), element_count, "degenerate: oracle produced no output");

    let (diff_words, max_abs_vs_current) = differing_word_count(new_root, current_root);
    let new_vs_oracle = compare_to_oracle(new_root, oracle_root);
    let current_vs_oracle = compare_to_oracle(current_root, oracle_root);

    eprintln!(
        "{tensor_name} EPILOGUE T={tokens} rows={rows} in_dim={in_dim}: \
         new-vs-current differing_words={diff_words}/{element_count} max_abs_diff={max_abs_vs_current} \
         new-vs-oracle max_abs={} max_relative={} \
         current-vs-oracle max_abs={} max_relative={}",
        new_vs_oracle.max_abs,
        new_vs_oracle.max_relative,
        current_vs_oracle.max_abs,
        current_vs_oracle.max_relative,
    );

    assert!(
        new_vs_oracle.max_relative < 1e-2,
        "{tensor_name} T={tokens}: new batched-q4_0_run8 tiled path with a fused epilogue \
         disagrees with the independent oracle (max_relative={})",
        new_vs_oracle.max_relative
    );
    assert!(
        current_vs_oracle.max_relative < 1e-2,
        "{tensor_name} T={tokens}: today's row-blocked path with a fused epilogue disagrees \
         with the independent oracle (max_relative={}) -- the oracle itself is suspect if this fires",
        current_vs_oracle.max_relative
    );
}

#[test]
fn attn_q_projection_batched_run8_matches_row_blocked_and_f32_oracle_with_a_fused_epilogue() {
    for tokens in [8usize, 64, 510] {
        check_real_tensor_epilogue_at_token_count("blk.0.attn_q.weight", tokens);
    }
}

#[test]
fn ffn_gate_projection_batched_run8_matches_row_blocked_and_f32_oracle_with_a_fused_epilogue() {
    for tokens in [8usize, 64, 510] {
        check_real_tensor_epilogue_at_token_count("blk.0.ffn_gate.weight", tokens);
    }
}

#[test]
fn attn_q_projection_batched_run8_matches_row_blocked_and_f32_oracle() {
    for tokens in [8usize, 64, 510] {
        check_real_tensor_at_token_count("blk.0.attn_q.weight", tokens);
    }
}

#[test]
fn ffn_gate_projection_batched_run8_matches_row_blocked_and_f32_oracle() {
    for tokens in [8usize, 64, 510] {
        check_real_tensor_at_token_count("blk.0.ffn_gate.weight", tokens);
    }
}
