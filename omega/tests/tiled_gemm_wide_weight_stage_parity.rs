//! Byte-identity gate for `PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE`
//! (see `docs/model-interop/discipline.md` ROW C4.12): the wide schedule changes WHO decodes which weight
//! element, how its bytes are loaded (`q4_0_run8_wide`'s `ushort` reads
//! instead of `q4_0_run8`'s `uchar` reads for `Q4_0`; `Q4_K` keeps
//! `q4k_header_for`/`q4k_run8` unchanged) and how the result is stored into
//! `weight_tile` (`half4` vector stores instead of scalar `half` stores) --
//! never the decoded VALUE (same `(level - 8) * d` / `scale * level -
//! minimum` expression per element, per
//! [`crate::identity::MetalOnlyExtras::tiled_gemm_wide_weight_stage`]'s own
//! doc). So, matching `tiled_gemm_direct_store_parity.rs`'s own posture,
//! this lever's bar is byte-identical ON vs OFF at every shape -- both
//! codecs `classify_tiled_gemm` admits (`Q4_0` behind `PROXIMA_TILED_GEMM_
//! Q4_0`, default on; `Q4_K` unconditionally), including a partial edge
//! tile (a `rows` count not a multiple of `TILED_GEMM_BLOCK_M`, 64) and
//! combined with `PROXIMA_TILED_GEMM_DIRECT_STORE=1` (a pure store-path
//! lever downstream of this one's own weight-tile contents).
//!
//! Real weight bytes: `Q4_0` from the same ollama gemma4-E2B blob
//! `tiled_gemm_direct_store_parity.rs` reads; `Q4_K` from the same
//! `openchat-3.5-1210.Q4_K_S.gguf` checkpoint `q4k_real_checkpoint_parity.rs`
//! reads (overridable via `PROXIMA_BENCH_GGUF_PATH`), per guiding-principles
//! §9. Skips (does not fail) when a real file or tensor is not present on
//! this host, matching every other real-checkpoint test's posture.

#![cfg(all(feature = "metal", feature = "metal-tiled-gemm", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

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

const REAL_GEMMA4_GGUF_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

fn real_q4k_gguf_path() -> String {
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

fn run_wide_weight_stage_byte_identity(
    codec: Codec,
    weight_bytes: &[u8],
    in_dim: usize,
    rows: usize,
    tokens: usize,
    also_direct_store: bool,
    label: &str,
) {
    let mut lcg = Lcg(9001 + tokens as u64 + rows as u64);
    let activation: Vec<f32> = (0..tokens * in_dim)
        .map(|_| lcg.next_unit() * 4.0 - 2.0)
        .collect();

    let (packed_program, packed_sum) =
        matmul_program(tokens as u32, in_dim as u32, rows as u32, DType::UInt8);
    let blocks = [
        QuantizedBlock::Packed { codec, bytes: weight_bytes },
        QuantizedBlock::Float32(&activation),
    ];

    let direct_store_value = if also_direct_store { Some("1") } else { None };

    let off = temp_env::with_vars(
        [
            ("PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE", Some("0")),
            ("PROXIMA_TILED_GEMM_DIRECT_STORE", direct_store_value),
        ],
        || {
            omega::execute(
                &packed_program,
                &[],
                &blocks,
                &[packed_sum],
                NumericPolicy::default(),
            )
            .expect("metal executes the non-wide tiled matmul")
        },
    );

    let on = temp_env::with_vars(
        [
            ("PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE", Some("1")),
            ("PROXIMA_TILED_GEMM_DIRECT_STORE", direct_store_value),
        ],
        || {
            omega::execute(
                &packed_program,
                &[],
                &blocks,
                &[packed_sum],
                NumericPolicy::default(),
            )
            .expect("metal executes the wide-weight-stage tiled matmul")
        },
    );

    let off_root = off.root();
    let on_root = on.root();
    let element_count = tokens * rows;
    assert_eq!(off_root.len(), element_count, "degenerate: OFF produced no output");
    assert_eq!(on_root.len(), element_count, "degenerate: ON produced no output");

    let mut differing = 0usize;
    let mut max_abs_diff = 0.0f32;
    for (&off_value, &on_value) in off_root.iter().zip(on_root.iter()) {
        if off_value.to_bits() != on_value.to_bits() {
            differing += 1;
        }
        max_abs_diff = max_abs_diff.max((off_value - on_value).abs());
    }
    eprintln!(
        "{label} codec={codec:?} tokens={tokens} rows={rows} in_dim={in_dim} \
         also_direct_store={also_direct_store}: wide-weight-stage on-vs-off \
         differing_words={differing}/{element_count} max_abs_diff={max_abs_diff}"
    );

    assert_eq!(
        off_root, on_root,
        "{label} codec={codec:?} tokens={tokens} rows={rows}: PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE=1 \
         must produce BIT-IDENTICAL output to the one-thread-per-row path (same decoded weight \
         values, only who computes them / how bytes are loaded and stored differs) -- \
         {differing}/{element_count} words differed, max_abs_diff={max_abs_diff}"
    );
}

fn check_q4_0(tensor_name: &str, tokens: usize, rows_wanted: usize, also_direct_store: bool, label: &str) {
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
    let rows = rows_wanted.min(out_dim);
    let sliced_weight = &weight_bytes[..rows * row_bytes];

    run_wide_weight_stage_byte_identity(
        Codec::Q4_0,
        sliced_weight,
        in_dim,
        rows,
        tokens,
        also_direct_store,
        label,
    );
}

fn check_q4k(tensor_name: &str, tokens: usize, rows_wanted: usize, also_direct_store: bool, label: &str) {
    let path_string = real_q4k_gguf_path();
    let path = std::path::Path::new(&path_string);
    let Some((parsed, file_len, mut file)) = real_gguf_header(path) else {
        eprintln!("real gguf file not found at {path_string}; test skipped");
        return;
    };
    let Some((weight_bytes, in_dim, out_dim)) =
        real_tensor_bytes(&mut file, &parsed, file_len, tensor_name, GgmlType::Q4_K)
    else {
        eprintln!("{tensor_name} not found or not Q4_K in this checkpoint; test skipped");
        return;
    };

    let blocks_per_row = in_dim / q4_k::QK_K;
    let row_bytes = blocks_per_row * q4_k::BLOCK_BYTES;
    let rows = rows_wanted.min(out_dim);
    let sliced_weight = &weight_bytes[..rows * row_bytes];

    run_wide_weight_stage_byte_identity(
        Codec::Q4K,
        sliced_weight,
        in_dim,
        rows,
        tokens,
        also_direct_store,
        label,
    );
}

/// `tokens=510` (this session's own prefill-length target) keeps every
/// K-tile interior (`in_dim=1536` is `TILED_GEMM_BLOCK_K`(32)-aligned);
/// `rows=128` is a whole multiple of `TILED_GEMM_BLOCK_M` (64), so the whole
/// dispatch grid is interior tiles.
#[test]
fn q4_0_wide_weight_stage_matches_default_path_at_a_fully_interior_shape() {
    check_q4_0("blk.0.attn_q.weight", 510, 128, false, "fully_interior");
}

/// `rows=100` is NOT a multiple of `TILED_GEMM_BLOCK_M` (64) --
/// `100 % 64 == 36` -- so the last row-tile is a genuine partial feature-row
/// tile, exercising the setup phase's `wws_feat < feature_extent` (out-of-
/// bound row, zero-fill) branch even with the switch on.
#[test]
fn q4_0_wide_weight_stage_matches_default_path_with_a_partial_edge_tile() {
    check_q4_0("blk.0.attn_q.weight", 510, 100, false, "partial_edge_tile");
}

/// Composed with `PROXIMA_TILED_GEMM_DIRECT_STORE=1` -- a pure store-path
/// lever downstream of this one's own `weight_tile` contents, so the two
/// switches must compose to the same bit-identical result.
#[test]
fn q4_0_wide_weight_stage_matches_default_path_combined_with_direct_store() {
    check_q4_0("blk.0.attn_q.weight", 510, 128, true, "combined_with_direct_store");
}

/// `Q4_K` admits `classify_tiled_gemm` unconditionally (no env override
/// needed) -- `blk.0.attn_q.weight` is 4096 x 4096 in this checkpoint,
/// `4096` a whole multiple of both `TILED_GEMM_BLOCK_K` (32) and
/// `TILED_GEMM_BLOCK_M` (64).
#[test]
fn q4k_wide_weight_stage_matches_default_path_at_a_fully_interior_shape() {
    check_q4k("blk.0.attn_q.weight", 32, 128, false, "fully_interior");
}

/// `rows=100` is NOT a multiple of `TILED_GEMM_BLOCK_M` (64), the same
/// partial-edge-tile shape [`q4_0_wide_weight_stage_matches_default_path_
/// with_a_partial_edge_tile`] exercises for `Q4_0`.
#[test]
fn q4k_wide_weight_stage_matches_default_path_with_a_partial_edge_tile() {
    check_q4k("blk.0.attn_q.weight", 32, 100, false, "partial_edge_tile");
}
