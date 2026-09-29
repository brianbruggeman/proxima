//! Byte-identity gate for `PROXIMA_TILED_GEMM_DIRECT_STORE` (see
//! `docs/model-interop/discipline.md` ROW C4.11): the fast arm writes the SAME `acc[]` accumulator
//! values `push_tiled_gemm_restage_writeback` already computes -- only the
//! store path differs (a device-direct `simdgroup_store` for an interior
//! tile instead of a threadgroup restage + scalar bounds-checked copy). So,
//! unlike the byte-exact-vs-CPU-oracle discipline the tiled-GEMM Q4_0/dense
//! admissions hold themselves to, this lever's own bar is byte-identical
//! ON vs OFF, at every shape -- including one with a partial edge tile, the
//! one case the fast path must decline for correctness (`direct_store_
//! interior`'s own runtime gate).
//!
//! Real weight bytes: the same ollama gemma4-E2B blob
//! `q4_0_tiled_gemm_batched_run8_parity.rs` reads (per guiding-principles
//! §9), `blk.0.attn_q.weight`. Two token counts: `510` (this session's own
//! prefill-length target, and `1536`'s own K value keeps every K-tile
//! interior) and `510` against a `rows` count NOT a multiple of
//! `TILED_GEMM_BLOCK_M` (64) -- `100` (`100 % 64 == 36`, a genuine partial
//! feature-row tile) -- so at least one tile in the dispatch grid takes the
//! `direct_store_interior == false` fallback even with the switch on.
//! Skips (does not fail) when the real file or tensor is not present on
//! this host, matching every other real-checkpoint test's posture.

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
    ScalarOp, append, projection,
};

const REAL_GEMMA4_GGUF_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

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

/// Same iteration-space convention as `q4_0_tiled_gemm_batched_run8_parity.rs`'s
/// own `matmul_program`.
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

/// Runs one real weight tensor, at one (tokens, rows) shape, through the
/// tiled-GEMM Q4_0 path TWICE -- `PROXIMA_TILED_GEMM_DIRECT_STORE` unset
/// (off, today's restage-only path) and explicit `"1"` (on) -- and asserts
/// the two runs are bit-for-bit identical. Returns early (no assertion) if
/// the tensor is absent, matching this crate's other real-checkpoint tests.
fn check_direct_store_byte_identity(tensor_name: &str, tokens: usize, rows_wanted: usize, label: &str) {
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

    let mut lcg = Lcg(9001 + tokens as u64 + rows as u64);
    let activation: Vec<f32> = (0..tokens * in_dim)
        .map(|_| lcg.next_unit() * 4.0 - 2.0)
        .collect();

    let (packed_program, packed_sum) =
        matmul_program(tokens as u32, in_dim as u32, rows as u32, DType::UInt8);
    let blocks = [
        QuantizedBlock::Packed { codec: Codec::Q4_0, bytes: sliced_weight },
        QuantizedBlock::Float32(&activation),
    ];

    let direct_store_off = temp_env::with_var("PROXIMA_TILED_GEMM_DIRECT_STORE", None::<&str>, || {
        omega::execute(
            &packed_program,
            &[],
            &blocks,
            &[packed_sum],
            NumericPolicy::default(),
        )
        .expect("metal executes the restage-only tiled q4_0 matmul")
    });

    let direct_store_on = temp_env::with_var("PROXIMA_TILED_GEMM_DIRECT_STORE", Some("1"), || {
        omega::execute(
            &packed_program,
            &[],
            &blocks,
            &[packed_sum],
            NumericPolicy::default(),
        )
        .expect("metal executes the direct-store tiled q4_0 matmul")
    });

    let off_root = direct_store_off.root();
    let on_root = direct_store_on.root();
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
        "{tensor_name} {label} tokens={tokens} rows={rows} in_dim={in_dim}: \
         direct_store on-vs-off differing_words={differing}/{element_count} max_abs_diff={max_abs_diff}"
    );

    assert_eq!(
        off_root, on_root,
        "{tensor_name} {label} tokens={tokens} rows={rows}: PROXIMA_TILED_GEMM_DIRECT_STORE=1 must \
         produce BIT-IDENTICAL output to the restage-only path (same acc[] values, only the store \
         path differs) -- {differing}/{element_count} words differed, max_abs_diff={max_abs_diff}"
    );
}

/// `tokens=510` keeps every K-tile (K=1536, `TILED_GEMM_BLOCK_K`-aligned)
/// interior; `rows=128` is a whole multiple of `TILED_GEMM_BLOCK_M` (64), so
/// the WHOLE dispatch grid is interior tiles -- this is the shape the fast
/// arm is meant to win on.
#[test]
fn direct_store_matches_restage_path_at_a_fully_interior_shape() {
    check_direct_store_byte_identity("blk.0.attn_q.weight", 510, 128, "fully_interior");
}

/// `rows=100` is NOT a multiple of `TILED_GEMM_BLOCK_M` (64) --
/// `100 % 64 == 36` -- so the last row-tile of the dispatch grid is a
/// genuine boundary tile even with the switch on, exercising `direct_store_
/// interior`'s own runtime fallback to the restage path for that tile
/// while interior tiles still take the direct arm.
#[test]
fn direct_store_matches_restage_path_with_a_partial_edge_tile() {
    check_direct_store_byte_identity("blk.0.attn_q.weight", 510, 100, "partial_edge_tile");
}

/// Small token count (`8`, `TILED_GEMM_MIN_TOKENS` itself) keeps the run
/// fast while still exercising a single dispatched threadgroup end to end.
#[test]
fn direct_store_matches_restage_path_at_the_minimum_admitted_token_count() {
    check_direct_store_byte_identity("blk.0.attn_q.weight", 8, 64, "min_tokens");
}
