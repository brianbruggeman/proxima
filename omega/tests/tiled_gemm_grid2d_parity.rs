//! Byte-identity gate for `PROXIMA_TILED_GEMM_GRID2D` (see
//! `docs/model-interop/discipline.md` ROW C4.19):
//! the switch changes HOW a thread learns its own tile coordinates
//! (`threadgroup_position_in_grid`/`thread_index_in_threadgroup`/
//! `simdgroup_index_in_threadgroup` read directly, vs a flattened `uint gid
//! [[thread_position_in_grid]]` divided/modded against `num_col_tiles`) and
//! HOW the driver dispatches (`dispatchThreadgroups` against an explicit
//! `(col_tiles, row_tiles, depth)` grid, vs `dispatchThreads` against a
//! flattened `row_tiles * col_tiles * 128` thread count) -- never which tile
//! a thread computes, so, matching `tiled_gemm_wide_weight_stage_parity.rs`'s
//! own posture, this lever's bar is byte-identical ON vs OFF at every shape:
//! both tiled-GEMM codecs (`Q4_0`, `Q4_K`), the dense-batched score path
//! (feature-fastest AND batch-innermost), the dense-batched P.V path, a
//! partial edge tile for each, and composed with `PROXIMA_TILED_GEMM_
//! WIDE_WEIGHT_STAGE=1`/`PROXIMA_TILED_GEMM_DIRECT_STORE=1`.
//!
//! Real weight bytes reused from `tiled_gemm_wide_weight_stage_parity.rs`
//! (`Q4_0` from the ollama gemma4-E2B blob, `Q4_K` from the `openchat`
//! checkpoint, per guiding-principles §9); dense score/P.V shapes reused
//! from `dense_batched_direct_store_parity.rs`'s own MEASURED real-prefill
//! constants. Skips (does not fail) when a real file or tensor is not
//! present on this host, matching every other real-checkpoint test's
//! posture.

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

/// Same iteration-space convention as `tiled_gemm_wide_weight_stage_parity.
/// rs`'s own `matmul_program`.
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

fn assert_bit_identical(off_root: &[f32], on_root: &[f32], element_count: usize, label: &str) {
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
        "{label}: grid2d on-vs-off differing_words={differing}/{element_count} max_abs_diff={max_abs_diff}"
    );
    assert_eq!(
        differing, 0,
        "{label}: PROXIMA_TILED_GEMM_GRID2D=1 must produce BIT-IDENTICAL output to the \
         flattened-gid path -- {differing}/{element_count} words differed, max_abs_diff={max_abs_diff}"
    );
}

#[allow(clippy::too_many_arguments)]
fn run_grid2d_byte_identity(
    codec: Codec,
    weight_bytes: &[u8],
    in_dim: usize,
    rows: usize,
    tokens: usize,
    wide_weight_stage: bool,
    direct_store: bool,
    label: &str,
) {
    let mut lcg = Lcg(21001 + tokens as u64 + rows as u64);
    let activation: Vec<f32> = (0..tokens * in_dim)
        .map(|_| lcg.next_unit() * 4.0 - 2.0)
        .collect();

    let (packed_program, packed_sum) =
        matmul_program(tokens as u32, in_dim as u32, rows as u32, DType::UInt8);
    let blocks = [
        QuantizedBlock::Packed { codec, bytes: weight_bytes },
        QuantizedBlock::Float32(&activation),
    ];

    let wws_value = if wide_weight_stage { Some("1") } else { None };
    let dstore_value = if direct_store { Some("1") } else { None };

    let off = temp_env::with_vars(
        [
            ("PROXIMA_TILED_GEMM_GRID2D", None::<&str>),
            ("PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE", wws_value),
            ("PROXIMA_TILED_GEMM_DIRECT_STORE", dstore_value),
        ],
        || {
            omega::execute(
                &packed_program,
                &[],
                &blocks,
                &[packed_sum],
                NumericPolicy::default(),
            )
            .expect("metal executes the flattened-gid tiled matmul")
        },
    );
    let on = temp_env::with_vars(
        [
            ("PROXIMA_TILED_GEMM_GRID2D", Some("1")),
            ("PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE", wws_value),
            ("PROXIMA_TILED_GEMM_DIRECT_STORE", dstore_value),
        ],
        || {
            omega::execute(
                &packed_program,
                &[],
                &blocks,
                &[packed_sum],
                NumericPolicy::default(),
            )
            .expect("metal executes the grid2d tiled matmul")
        },
    );

    assert_bit_identical(off.root(), on.root(), tokens * rows, label);
}

fn check_q4_0(
    tensor_name: &str,
    tokens: usize,
    rows_wanted: usize,
    wide_weight_stage: bool,
    direct_store: bool,
    label: &str,
) {
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

    run_grid2d_byte_identity(
        Codec::Q4_0,
        sliced_weight,
        in_dim,
        rows,
        tokens,
        wide_weight_stage,
        direct_store,
        label,
    );
}

fn check_q4k(
    tensor_name: &str,
    tokens: usize,
    rows_wanted: usize,
    wide_weight_stage: bool,
    direct_store: bool,
    label: &str,
) {
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

    run_grid2d_byte_identity(
        Codec::Q4K,
        sliced_weight,
        in_dim,
        rows,
        tokens,
        wide_weight_stage,
        direct_store,
        label,
    );
}

/// `tokens=510` (this session's own prefill-length target), `rows=128` a
/// whole multiple of `TILED_GEMM_BLOCK_M` (64) -- interior tiles only.
#[test]
fn q4_0_grid2d_matches_flattened_gid_at_a_fully_interior_shape() {
    check_q4_0("blk.0.attn_q.weight", 510, 128, false, false, "q4_0_fully_interior");
}

/// `rows=100` is NOT a multiple of `TILED_GEMM_BLOCK_M` (64) -- a genuine
/// partial feature-row tile.
#[test]
fn q4_0_grid2d_matches_flattened_gid_with_a_partial_edge_tile() {
    check_q4_0("blk.0.attn_q.weight", 510, 100, false, false, "q4_0_partial_edge_tile");
}

/// Composed with `PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE=1` -- the largest
/// single lever this crate has landed (`C4.12`), and the one whose own
/// preamble this switch's signature change sits directly upstream of.
#[test]
fn q4_0_grid2d_matches_flattened_gid_combined_with_wide_weight_stage() {
    check_q4_0("blk.0.attn_q.weight", 510, 128, true, false, "q4_0_combined_with_wws");
}

/// Composed with `PROXIMA_TILED_GEMM_DIRECT_STORE=1`.
#[test]
fn q4_0_grid2d_matches_flattened_gid_combined_with_direct_store() {
    check_q4_0("blk.0.attn_q.weight", 510, 128, false, true, "q4_0_combined_with_dstore");
}

/// All three switches composed at once.
#[test]
fn q4_0_grid2d_matches_flattened_gid_combined_with_wws_and_direct_store() {
    check_q4_0("blk.0.attn_q.weight", 510, 128, true, true, "q4_0_combined_with_wws_and_dstore");
}

/// `Q4_K` admits `classify_tiled_gemm` unconditionally.
#[test]
fn q4k_grid2d_matches_flattened_gid_at_a_fully_interior_shape() {
    check_q4k("blk.0.attn_q.weight", 32, 128, false, false, "q4k_fully_interior");
}

/// `rows=100` is NOT a multiple of `TILED_GEMM_BLOCK_M` (64).
#[test]
fn q4k_grid2d_matches_flattened_gid_with_a_partial_edge_tile() {
    check_q4k("blk.0.attn_q.weight", 32, 100, false, false, "q4k_partial_edge_tile");
}

/// Composed with `PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE=1` and
/// `PROXIMA_TILED_GEMM_DIRECT_STORE=1` at once.
#[test]
fn q4k_grid2d_matches_flattened_gid_combined_with_wws_and_direct_store() {
    check_q4k("blk.0.attn_q.weight", 32, 128, true, true, "q4k_combined_with_wws_and_dstore");
}

/// Iteration dims matching `dense_batched_direct_store_parity.rs`'s own
/// `dense_batched_feature_fastest_program`: 0=token, 1=reduce, 2=batch,
/// 3=feature (feature fastest).
fn dense_batched_feature_fastest_program(
    token: u32,
    reduce_len: u32,
    batch: u32,
    feature: u32,
) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![
                Extent::Static(feature),
                Extent::Static(batch),
                Extent::Static(reduce_len),
            ],
            name: None,
        },
    );
    let other = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![
                Extent::Static(token),
                Extent::Static(batch),
                Extent::Static(reduce_len),
            ],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(4, &[3, 2, 1]))),
                (other, IndexMap::Affine(projection(4, &[0, 2, 1]))),
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
            in_map: IndexMap::Affine(projection(4, &[0, 1, 2, 3])),
            out_map: IndexMap::Affine(projection(4, &[0, 2, 3])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    (program, sum)
}

/// Iteration dims matching `dense_batched_direct_store_parity.rs`'s own
/// `dense_batched_batch_innermost_program`: 0=token, 1=reduce, 2=feature,
/// 3=batch (batch fastest -- gemma4's real `score_even`/`score_odd` layout).
fn dense_batched_batch_innermost_program(
    token: u32,
    reduce_len: u32,
    feature: u32,
    batch: u32,
) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![
                Extent::Static(feature),
                Extent::Static(batch),
                Extent::Static(reduce_len),
            ],
            name: None,
        },
    );
    let other = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![
                Extent::Static(token),
                Extent::Static(batch),
                Extent::Static(reduce_len),
            ],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(4, &[2, 3, 1]))),
                (other, IndexMap::Affine(projection(4, &[0, 3, 1]))),
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
            in_map: IndexMap::Affine(projection(4, &[0, 1, 2, 3])),
            out_map: IndexMap::Affine(projection(4, &[0, 2, 3])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    (program, sum)
}

// mirrors the shape of every sibling parity test's own multi-shape fixture
// runner; splitting the shape/switch params into a struct would obscure the
// call sites more than it would clarify them.
#[allow(clippy::too_many_arguments)]
fn run_dense_grid2d_byte_identity(
    program: &[Op],
    sum: NodeId,
    weight: &[f32],
    other: &[f32],
    element_count: usize,
    wide_weight_stage: bool,
    direct_store: bool,
    label: &str,
) {
    let blocks = [QuantizedBlock::Float32(weight), QuantizedBlock::Float32(other)];
    let wws_value = if wide_weight_stage { Some("1") } else { None };
    let dstore_value = if direct_store { Some("1") } else { None };

    let off = temp_env::with_vars(
        [
            ("PROXIMA_TILED_GEMM_GRID2D", None::<&str>),
            ("PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE", wws_value),
            ("PROXIMA_TILED_GEMM_DIRECT_STORE", dstore_value),
            ("PROXIMA_TILED_GEMM_DENSE", None::<&str>),
        ],
        || {
            omega::execute(program, &[], &blocks, &[sum], NumericPolicy::default())
                .expect("metal executes the flattened-gid dense-batched matmul")
        },
    );
    let on = temp_env::with_vars(
        [
            ("PROXIMA_TILED_GEMM_GRID2D", Some("1")),
            ("PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE", wws_value),
            ("PROXIMA_TILED_GEMM_DIRECT_STORE", dstore_value),
            ("PROXIMA_TILED_GEMM_DENSE", None::<&str>),
        ],
        || {
            omega::execute(program, &[], &blocks, &[sum], NumericPolicy::default())
                .expect("metal executes the grid2d dense-batched matmul")
        },
    );

    assert_bit_identical(off.root(), on.root(), element_count, label);
}

/// Real gemma4-E2B weather-prompt (562 real tokens) score shape, MEASURED
/// via `dstore/prompt5_strides_step0.log` (`node=139`), batch-innermost
/// (gemma4's real `score_even`/`score_odd` layout) -- `feature=576` a whole
/// `TILED_GEMM_BLOCK_M`(64) multiple (interior row-tiles), `token=563` NOT a
/// `TILED_GEMM_BLOCK_N`(32) multiple (`563 % 32 == 19`, a genuine partial
/// edge column-tile).
#[test]
fn dense_score_grid2d_matches_flattened_gid_at_the_real_weather_prefill_shape() {
    let (program, sum) = dense_batched_batch_innermost_program(563, 128, 576, 8);
    let weight = random_vec(11001, 576 * 8 * 128);
    let other = random_vec(11502, 563 * 8 * 128);
    run_dense_grid2d_byte_identity(
        &program,
        sum,
        &weight,
        &other,
        563 * 576 * 8,
        false,
        false,
        "dense_score_weather_shape",
    );
}

/// Same shape composed with `WIDE_WEIGHT_STAGE`/`DIRECT_STORE` -- the dense
/// body has no codec decode arm (`wide_weight_stage_active`'s own doc: this
/// never applies to `push_dense_batched_gemm_body`), so this is a compose-
/// with-`DIRECT_STORE`-only combination, matching this path's own real
/// eligibility.
#[test]
fn dense_score_grid2d_matches_flattened_gid_combined_with_direct_store() {
    let (program, sum) = dense_batched_batch_innermost_program(563, 128, 576, 8);
    let weight = random_vec(11003, 576 * 8 * 128);
    let other = random_vec(11504, 563 * 8 * 128);
    run_dense_grid2d_byte_identity(
        &program,
        sum,
        &weight,
        &other,
        563 * 576 * 8,
        false,
        true,
        "dense_score_weather_shape_with_dstore",
    );
}

/// Real gemma4-E2B hippo-prompt (26 real tokens) score shape, MEASURED via
/// `dstore/prompt4_strides_step0.log` (`node=139`) -- `feature=32 <
/// TILED_GEMM_BLOCK_M(64)` and `token=27 < TILED_GEMM_BLOCK_N(32)`, every
/// tile a partial edge tile.
#[test]
fn dense_score_grid2d_matches_flattened_gid_with_a_partial_edge_tile() {
    let (program, sum) = dense_batched_batch_innermost_program(27, 128, 32, 8);
    let weight = random_vec(11005, 32 * 8 * 128);
    let other = random_vec(11506, 27 * 8 * 128);
    run_dense_grid2d_byte_identity(
        &program,
        sum,
        &weight,
        &other,
        27 * 32 * 8,
        false,
        false,
        "dense_score_hippo_partial_edge_tile",
    );
}

/// Real gemma4-E2B weather-prompt `attended` (P.V) shape, MEASURED via
/// `dstore/prompt5_strides_step0.log` (`node=162`): `feature_extent=256
/// token_extent=563 batch_extent=8 reduction_k=32`, feature-fastest layout
/// (`out_stride_feature=1`, `DIRECT_STORE`-eligible in production).
#[test]
fn dense_pv_grid2d_matches_flattened_gid_at_the_real_weather_attended_shape() {
    let (program, sum) = dense_batched_feature_fastest_program(563, 32, 8, 256);
    let weight = random_vec(11007, 256 * 8 * 32);
    let other = random_vec(11508, 563 * 8 * 32);
    run_dense_grid2d_byte_identity(
        &program,
        sum,
        &weight,
        &other,
        563 * 256 * 8,
        false,
        false,
        "dense_pv_weather_attended_shape",
    );
}

/// Same P.V shape composed with `DIRECT_STORE=1` -- this feature-fastest
/// layout is exactly the one production shape where the fast arm engages.
#[test]
fn dense_pv_grid2d_matches_flattened_gid_combined_with_direct_store() {
    let (program, sum) = dense_batched_feature_fastest_program(563, 32, 8, 256);
    let weight = random_vec(11009, 256 * 8 * 32);
    let other = random_vec(11510, 563 * 8 * 32);
    run_dense_grid2d_byte_identity(
        &program,
        sum,
        &weight,
        &other,
        563 * 256 * 8,
        false,
        true,
        "dense_pv_weather_attended_shape_with_dstore",
    );
}

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit() * 2.0 - 1.0).collect()
}
