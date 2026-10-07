//! Byte-identity gate for `PROXIMA_TILED_GEMM_MM_LAYOUT` (see
//! [`omega::msl`]'s `push_mm_layout_k_loop`): the ggml `kernel_mul_mm` tile
//! layout (weight tile in 8x8 `[k][feature]` blocks, activation tile in 8x8
//! `[token][k]` blocks, `token x feature` accumulators, Q4_0 decoded with
//! ggml's fused multiply-add form) changes which threadgroup address each
//! element lives at and which operand order the matrix multiply takes -- never
//! a decoded value and never the order of a K-sum. So this lever's bar is
//! byte-identical output against the row-major tile layout
//! (`PROXIMA_TILED_GEMM_MM_LAYOUT=0`) at every shape: both codecs
//! `classify_tiled_gemm` admits, an interior tile grid, a partial feature tile
//! (rows not a multiple of `TILED_GEMM_BLOCK_M`, 64), a partial token tile
//! (tokens not a multiple of `TILED_GEMM_BLOCK_N`, 32), the generic
//! (non-unit-stride) activation read (`PROXIMA_TILED_GEMM_WIDE_ACT_LOAD=0`),
//! and composed with the direct device store (`PROXIMA_TILED_GEMM_DIRECT_STORE=1`).
//!
//! The same bar gates `PROXIMA_TILED_GEMM_DYNAMIC_TGMEM`: the kernel takes its threadgroup
//! backing store as the `[[threadgroup(0)]]` argument and the launch binds its length, which
//! moves no value, so the default must match `PROXIMA_TILED_GEMM_DYNAMIC_TGMEM=0` word for word;
//! a launch that bound too little memory would not.
//!
//! Real weight bytes: `Q4_0` from the ollama gemma4-E2B blob, `Q4_K` from the
//! `openchat-3.5-1210.Q4_K_S.gguf` checkpoint (overridable via
//! `PROXIMA_BENCH_GGUF_PATH`), per guiding-principles section 9. Skips (does
//! not fail) when a real file or tensor is not present on this host, matching
//! every other real-checkpoint test's posture; each test prints the word count
//! it compared, so a skip is visible as the absence of that line.

#![cfg(all(feature = "metal", feature = "metal-tiled-gemm", target_os = "macos"))]

use std::io::{Read, Seek, SeekFrom};

use proxima_gguf::parser::{GgufEvent, GgufParser};
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::quant::{q4_0, q4_k};
use proxima_gguf::types::GgmlType;
use proxima_primitives::Codec;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce, ReduceInit,
    ScalarOp, append, projection,
};

const MM_LAYOUT: &str = "PROXIMA_TILED_GEMM_MM_LAYOUT";
const DYNAMIC_TGMEM: &str = "PROXIMA_TILED_GEMM_DYNAMIC_TGMEM";
const RESTAGE: [(&str, Option<&str>); 1] = [("PROXIMA_TILED_GEMM_DIRECT_STORE", Some("0"))];

const REAL_GEMMA4_GGUF_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

fn real_q4k_gguf_path() -> String {
    std::env::var("PROXIMA_BENCH_GGUF_PATH").unwrap_or_else(|_| {
        "/Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf"
            .to_string()
    })
}

type TestResult<T> = Result<T, String>;

fn real_gguf_header(
    path: &std::path::Path,
) -> TestResult<Option<(ParsedGguf, u64, std::fs::File)>> {
    let Ok(mut file) = std::fs::File::open(path) else {
        return Ok(None);
    };
    let file_len = file
        .metadata()
        .map_err(|error| format!("stat {}: {error}", path.display()))?
        .len();

    let mut prefix_len = 1usize << 20;
    loop {
        let mut buf = vec![0u8; prefix_len];
        file.seek(SeekFrom::Start(0))
            .map_err(|error| format!("seek to start of {}: {error}", path.display()))?;
        let read = file
            .read(&mut buf)
            .map_err(|error| format!("read gguf prefix of {}: {error}", path.display()))?;
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
                parser
                    .finish()
                    .map_err(|error| format!("gguf parser did not finish clean: {error:?}"))?;
                let parsed = ParsedGguf {
                    version,
                    tensor_count: tensors.len() as u64,
                    kv_count: metadata.len() as u64,
                    metadata,
                    tensors,
                    data_offset,
                    alignment,
                };
                return Ok(Some((parsed, file_len, file)));
            }
        }
        if prefix_len as u64 >= file_len {
            return Ok(None);
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
) -> TestResult<Option<(Vec<u8>, usize, usize)>> {
    let Some(tensor) = parsed
        .tensors
        .iter()
        .find(|candidate| candidate.name == name)
    else {
        return Ok(None);
    };
    if tensor.ggml_type != expect_type {
        eprintln!(
            "real_tensor_bytes: {name} is {:?} in this file, not {expect_type:?} -- test skipped, not faked",
            tensor.ggml_type
        );
        return Ok(None);
    }
    let in_dim = tensor.dims[0] as usize;
    let out_dim = tensor.dims[1] as usize;
    let range = parsed
        .tensor_data_range(tensor, file_len)
        .map_err(|error| format!("tensor {name} byte range outside the file: {error:?}"))?;
    let mut buf = vec![0u8; (range.end - range.start) as usize];
    file.seek(SeekFrom::Start(range.start))
        .map_err(|error| format!("seek to tensor {name}: {error}"))?;
    file.read_exact(&mut buf)
        .map_err(|error| format!("read tensor {name} bytes: {error}"))?;
    Ok(Some((buf, in_dim, out_dim)))
}

/// Same iteration-space convention as `tiled_gemm_direct_store_parity.rs`'s
/// own `matmul_program`: `[tokens, in_dim] x [in_dim, rows] -> [tokens,
/// rows]`, `weight_dtype` distinguishing "packed bytes" (`UInt8`) from a
/// plain `f32` activation operand.
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

/// One lever compared at one shape: `lever` unset (the default) against `lever` set to `"0"`, with
/// `extra_env` held identical on both sides.
struct Case<'a> {
    tokens: usize,
    rows_wanted: usize,
    lever: &'a str,
    extra_env: &'a [(&'a str, Option<&'a str>)],
    label: &'a str,
}

fn run_lever_byte_identity(
    codec: Codec,
    weight_bytes: &[u8],
    in_dim: usize,
    rows: usize,
    case: &Case,
) -> TestResult<()> {
    let Case { tokens, lever, extra_env, label, .. } = *case;
    let mut lcg = Lcg(31001 + tokens as u64 + rows as u64);
    let activation: Vec<f32> = (0..tokens * in_dim)
        .map(|_| lcg.next_unit() * 4.0 - 2.0)
        .collect();

    let (packed_program, packed_sum) =
        matmul_program(tokens as u32, in_dim as u32, rows as u32, DType::UInt8);
    let blocks = [
        QuantizedBlock::Packed { codec, bytes: weight_bytes },
        QuantizedBlock::Float32(&activation),
    ];

    let run = |lever_value: Option<&str>| {
        let mut vars = vec![(lever, lever_value)];
        vars.extend_from_slice(extra_env);
        temp_env::with_vars(vars, || {
            omega::execute(
                &packed_program,
                &[],
                &blocks,
                &[packed_sum],
                NumericPolicy::default(),
            )
            .map_err(|error| format!("{label}: metal did not execute the tiled matmul: {error:?}"))
        })
    };
    let baseline = run(Some("0"))?;
    let candidate = run(None)?;

    let baseline_root = baseline.root();
    let candidate_root = candidate.root();
    let element_count = tokens * rows;
    assert_eq!(baseline_root.len(), element_count, "degenerate: {lever}=0 produced no output");
    assert_eq!(candidate_root.len(), element_count, "degenerate: the {lever} default produced no output");

    let differing = baseline_root
        .iter()
        .zip(candidate_root.iter())
        .filter(|(left, right)| left.to_bits() != right.to_bits())
        .count();
    eprintln!(
        "{label} codec={codec:?} tokens={tokens} rows={rows} in_dim={in_dim} lever={lever} extra_env={extra_env:?}: \
         default-vs-{lever}=0 differing_words={differing}/{element_count}"
    );
    assert_eq!(
        differing, 0,
        "{label} codec={codec:?} tokens={tokens} rows={rows}: {lever} on must produce \
         BIT-IDENTICAL output to {lever}=0 -- {differing}/{element_count} words differed"
    );
    Ok(())
}

fn check_q4_0(tensor_name: &str, case: &Case) -> TestResult<()> {
    let path = std::path::Path::new(REAL_GEMMA4_GGUF_PATH);
    let Some((parsed, file_len, mut file)) = real_gguf_header(path)? else {
        eprintln!("real gguf file not found at {REAL_GEMMA4_GGUF_PATH}; test skipped");
        return Ok(());
    };
    let Some((weight_bytes, in_dim, out_dim)) =
        real_tensor_bytes(&mut file, &parsed, file_len, tensor_name, GgmlType::Q4_0)?
    else {
        eprintln!("{tensor_name} not found or not Q4_0 in this checkpoint; test skipped");
        return Ok(());
    };
    let row_bytes = in_dim / q4_0::QK4_0 * q4_0::BLOCK_BYTES;
    let rows = case.rows_wanted.min(out_dim);
    run_lever_byte_identity(Codec::Q4_0, &weight_bytes[..rows * row_bytes], in_dim, rows, case)
}

fn check_q4k(tensor_name: &str, case: &Case) -> TestResult<()> {
    let path_string = real_q4k_gguf_path();
    let path = std::path::Path::new(&path_string);
    let Some((parsed, file_len, mut file)) = real_gguf_header(path)? else {
        eprintln!("real gguf file not found at {path_string}; test skipped");
        return Ok(());
    };
    let Some((weight_bytes, in_dim, out_dim)) =
        real_tensor_bytes(&mut file, &parsed, file_len, tensor_name, GgmlType::Q4_K)?
    else {
        eprintln!("{tensor_name} not found or not Q4_K in this checkpoint; test skipped");
        return Ok(());
    };
    let row_bytes = in_dim / q4_k::QK_K * q4_k::BLOCK_BYTES;
    let rows = case.rows_wanted.min(out_dim);
    run_lever_byte_identity(Codec::Q4K, &weight_bytes[..rows * row_bytes], in_dim, rows, case)
}

const WIDE_ACT_LOAD_OFF: [(&str, Option<&str>); 2] = [
    ("PROXIMA_TILED_GEMM_DIRECT_STORE", Some("0")),
    ("PROXIMA_TILED_GEMM_WIDE_ACT_LOAD", Some("0")),
];
const DIRECT_STORE_ON: [(&str, Option<&str>); 1] = [("PROXIMA_TILED_GEMM_DIRECT_STORE", Some("1"))];
const ROW_MAJOR_LAYOUT: [(&str, Option<&str>); 1] = [("PROXIMA_TILED_GEMM_MM_LAYOUT", Some("0"))];

#[test]
fn q4_0_mm_layout_matches_row_major_with_a_full_tile_grid() -> TestResult<()> {
    let case = Case { tokens: 512, rows_wanted: 128, lever: MM_LAYOUT, extra_env: &RESTAGE, label: "full_tile_grid" };
    check_q4_0("blk.0.attn_q.weight", &case)
}

#[test]
fn q4_0_mm_layout_matches_row_major_with_a_partial_token_tile() -> TestResult<()> {
    let case = Case { tokens: 510, rows_wanted: 128, lever: MM_LAYOUT, extra_env: &RESTAGE, label: "partial_token_tile" };
    check_q4_0("blk.0.attn_q.weight", &case)
}

#[test]
fn q4_0_mm_layout_matches_row_major_with_a_partial_feature_tile() -> TestResult<()> {
    let case = Case { tokens: 510, rows_wanted: 100, lever: MM_LAYOUT, extra_env: &RESTAGE, label: "partial_feature_tile" };
    check_q4_0("blk.0.attn_q.weight", &case)
}

#[test]
fn q4_0_mm_layout_matches_row_major_at_the_gemma4_prefill_width() -> TestResult<()> {
    let case = Case { tokens: 971, rows_wanted: 256, lever: MM_LAYOUT, extra_env: &RESTAGE, label: "prefill_width" };
    check_q4_0("blk.0.ffn_gate.weight", &case)
}

#[test]
fn q4_0_mm_layout_matches_row_major_at_the_gemma4_prefill_width_with_the_direct_store_default() -> TestResult<()> {
    let case = Case {
        tokens: 971,
        rows_wanted: 256,
        lever: MM_LAYOUT,
        extra_env: &[],
        label: "prefill_width_direct_store_default",
    };
    check_q4_0("blk.0.ffn_gate.weight", &case)
}

#[test]
fn q4_0_mm_layout_matches_row_major_combined_with_direct_store() -> TestResult<()> {
    let case = Case {
        tokens: 510,
        rows_wanted: 128,
        lever: MM_LAYOUT,
        extra_env: &DIRECT_STORE_ON,
        label: "combined_with_direct_store",
    };
    check_q4_0("blk.0.attn_q.weight", &case)
}

#[test]
fn q4_0_mm_layout_matches_row_major_on_the_generic_activation_read() -> TestResult<()> {
    let case = Case {
        tokens: 510,
        rows_wanted: 128,
        lever: MM_LAYOUT,
        extra_env: &WIDE_ACT_LOAD_OFF,
        label: "generic_activation_read",
    };
    check_q4_0("blk.0.attn_q.weight", &case)
}

#[test]
fn q4k_mm_layout_matches_row_major_with_a_full_tile_grid() -> TestResult<()> {
    let case = Case { tokens: 512, rows_wanted: 128, lever: MM_LAYOUT, extra_env: &RESTAGE, label: "full_tile_grid" };
    check_q4k("blk.0.attn_q.weight", &case)
}

#[test]
fn q4k_mm_layout_matches_row_major_with_partial_tiles() -> TestResult<()> {
    let case = Case { tokens: 510, rows_wanted: 100, lever: MM_LAYOUT, extra_env: &RESTAGE, label: "partial_tiles" };
    check_q4k("blk.0.attn_q.weight", &case)
}

#[test]
fn q4_0_dynamic_threadgroup_memory_matches_the_declared_array_at_the_gemma4_prefill_width() -> TestResult<()> {
    let case = Case { tokens: 971, rows_wanted: 256, lever: DYNAMIC_TGMEM, extra_env: &[], label: "dynamic_prefill_width" };
    check_q4_0("blk.0.ffn_gate.weight", &case)
}

#[test]
fn q4_0_dynamic_threadgroup_memory_matches_the_declared_array_with_partial_tiles() -> TestResult<()> {
    let case = Case { tokens: 510, rows_wanted: 100, lever: DYNAMIC_TGMEM, extra_env: &[], label: "dynamic_partial_tiles" };
    check_q4_0("blk.0.attn_q.weight", &case)
}

#[test]
fn q4_0_dynamic_threadgroup_memory_matches_the_declared_array_on_the_row_major_layout() -> TestResult<()> {
    let case = Case {
        tokens: 510,
        rows_wanted: 128,
        lever: DYNAMIC_TGMEM,
        extra_env: &ROW_MAJOR_LAYOUT,
        label: "dynamic_row_major",
    };
    check_q4_0("blk.0.attn_q.weight", &case)
}

#[test]
fn q4k_dynamic_threadgroup_memory_matches_the_declared_array_with_partial_tiles() -> TestResult<()> {
    let case = Case { tokens: 510, rows_wanted: 100, lever: DYNAMIC_TGMEM, extra_env: &[], label: "dynamic_partial_tiles" };
    check_q4k("blk.0.attn_q.weight", &case)
}
