//! Device parity for `Q4_0`, on the REAL bytes `blk.0.attn_k.weight` carries
//! in the ollama gemma4-E2B blob
//! (`~/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd`)
//! -- not synthetic, per guiding-principles §9. All attn/ffn weight tensors
//! in that checkpoint are `Q4_0` (541 tensors total), so `blk.0.attn_k.weight`
//! is the same tensor name `q8_0_real_checkpoint_parity.rs` reads, just from
//! a different host-local checkpoint whose quantizer happened to pick `Q4_0`
//! there.
//!
//! `Q4_0` is a flat 32-element block (one `f16` scale, 16 packed nibble
//! bytes, no sub-block scale/min hierarchy) -- structurally the same KIND as
//! `Q8_0`: no super-block, unlike `Q4_K`/`Q5_K`/`Q6_K`'s shared 256-element
//! super-block (see `q4_0::dequantize_block`'s own doc). `blk.0.attn_k.weight`'s
//! `k=1536` is a whole multiple of 256, so this op takes the row-blocked fast
//! path (`classify_packed_row_block` whitelists `Q4_0` via the super-element
//! accessor landed in `4c170d01b`; eight contiguous `Q4_0` blocks span
//! exactly one K-quant super-block's 256 elements) rather than the fully
//! generic per-element path -- this test proves that fast path is
//! numerically correct against real checkpoint bytes, the same contract
//! `q8_0_real_checkpoint_parity.rs` proves for its own codec.
//!
//! Skips (does not fail) when the real file is not present on this host --
//! matching `q8_0_real_checkpoint_parity.rs`'s own posture.

#![cfg(all(feature = "metal", target_os = "macos"))]
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
    ScalarOp, append, evaluate, map,
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

/// `[rows, k] x [k, 1] -> [rows, 1]`, `weight_dtype` distinguishing "packed
/// bytes" (`UInt8`) from the dequantized `f32` oracle -- same shape
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
            name: Some("q4_0_real_matmul".into()),
        }),
    );
    (program, sum)
}

/// `blk.0.attn_k.weight` is 1536 x 256 (`embedding x kv_dim`) in this
/// checkpoint; running the full 256 rows on a device parity test is
/// unnecessary to prove the unpack is bit-correct, so this takes a prefix of
/// rows (each row's packed bytes are contiguous, so a byte-level prefix
/// slice is exactly a row-count prefix).
const ROWS_TO_CHECK: usize = 64;

#[test]
fn metal_matmul_on_real_attn_k_q4_0_bytes_matches_the_dequantized_f32_cpu_path() {
    let path = std::path::Path::new(REAL_GEMMA4_GGUF_PATH);
    let Some((parsed, file_len, mut file)) = real_gguf_header(path) else {
        eprintln!("real gguf file not found at {REAL_GEMMA4_GGUF_PATH}; test skipped");
        return;
    };
    let Some((weight_bytes, in_dim, out_dim)) = real_tensor_bytes(
        &mut file,
        &parsed,
        file_len,
        "blk.0.attn_k.weight",
        GgmlType::Q4_0,
    ) else {
        return;
    };

    let blocks_per_row = in_dim / q4_0::QK4_0;
    assert_eq!(
        blocks_per_row * q4_0::QK4_0,
        in_dim,
        "blk.0.attn_k.weight's in_dim is a whole number of Q4_0 blocks"
    );
    let row_bytes = blocks_per_row * q4_0::BLOCK_BYTES;
    assert_eq!(
        weight_bytes.len(),
        row_bytes * out_dim,
        "blk.0.attn_k.weight byte length matches its declared shape"
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
        q4_0::dequantize(row_blocks, row_f32).expect("a whole number of q4_0 blocks per row");
    }

    let (packed_program, packed_sum) = matmul_program(rows as u32, in_dim as u32, DType::UInt8);
    let metal = omega::execute(
        &packed_program,
        &[],
        &[
            QuantizedBlock::Packed { codec: Codec::Q4_0, bytes: sliced_weight },
            QuantizedBlock::Float32(&activation),
        ],
        &[packed_sum],
        NumericPolicy::default(),
    )
    .expect("metal executes a packed q4_0 matmul on real blk.0.attn_k.weight bytes");

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
        "real blk.0.attn_k.weight (Q4_0, {rows} of {out_dim} rows, k={in_dim}) metal vs dequantized-f32 cpu: \
         max_diff={max_diff} max_magnitude={max_magnitude} relative={relative}"
    );
    assert!(
        relative < 1e-5,
        "packed unpack disagrees with the dequantized reference on REAL checkpoint bytes: \
         relative={relative} max_diff={max_diff}"
    );
}
