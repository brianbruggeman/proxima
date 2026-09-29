//! Regression gate for the thread-index overflow: a Metal dispatch whose grid
//! exceeds `u32::MAX` threads was launched as `threads mod 2^32`, so every
//! output past that prefix read back as zero with no driver error.
//!
//! The shape is gemma4-E2B's `per_layer_model_proj`: an F16 weight
//! `[8960, 1536]` (the checkpoint's own `per_layer_model_proj.weight`,
//! `dims [1536, 8960], ggml_type F16`) against `[rows, 1536]` activations,
//! reduced over the 1536 axis at the 256 lanes `wide_cooperative_reduce_width`
//! asks for: `rows * 8960 * 256` threads, which crosses `2^32` at 1873 rows
//! (1872 rows is 4_294_082_560 threads and still fits).
//!
//! Three independent checks per row count: no output reads back as zero unless
//! its own dot product is (a truncated grid leaves millions of them, an exact
//! cancellation leaves a handful); the rows
//! the 1872-row linear-form dispatch also computes are byte-identical to it
//! (the wide form must not change a single bit of what the linear form
//! produced); and the tail rows past the threshold agree with
//! `proxima_tensor::evaluate` on the same rows.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{Read, Seek, SeekFrom};

use proxima_gguf::parser::{GgufEvent, GgufParser};
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::quant::f16;
use proxima_gguf::types::GgmlType;
use proxima_primitives::Codec;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce, ReduceInit,
    ScalarOp, append, projection,
};

const REDUCTION_LEN: u32 = 1536;
const FEATURES: u32 = 8960;
const LAST_ROW_COUNT_AT_FULL_LANE_WIDTH: u32 = 1872;
const TAIL_ORACLE_ROWS: u32 = 2;
const MAX_COINCIDENTAL_ZERO_OUTPUTS: usize = 16;

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit() * 2.0 - 1.0).collect()
}

fn projection_program(rows: u32, weight_dtype: DType) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weights = append(
        &mut program,
        Op::Input {
            dtype: weight_dtype,
            shape: vec![Extent::Static(FEATURES), Extent::Static(REDUCTION_LEN)],
            name: None,
        },
    );
    let activations = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(rows), Extent::Static(REDUCTION_LEN)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weights, IndexMap::Affine(projection(3, &[1, 2]))),
                (activations, IndexMap::Affine(projection(3, &[0, 2]))),
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
            out_map: IndexMap::Affine(projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("per_layer_model_proj".into()),
        }),
    );
    (program, sum)
}

fn run_on_metal(rows: u32, activations: &[f32], weight_bytes: &[u8]) -> Vec<f32> {
    let (program, sum) = projection_program(rows, DType::Float16);
    let blocks = [
        QuantizedBlock::Packed {
            codec: Codec::Float16,
            bytes: weight_bytes,
        },
        QuantizedBlock::Float32(activations),
    ];
    omega::execute(&program, &[], &blocks, &[sum], NumericPolicy::llama_relaxed())
        .expect("metal executes the per-layer projection")
        .root()
        .to_vec()
}

fn run_on_cpu(rows: u32, activations: &[f32], dequantized_weights: &[f32]) -> Vec<f32> {
    let (program, sum) = projection_program(rows, DType::Float32);
    proxima_tensor::evaluate(&program, &[], &[dequantized_weights, activations], &[sum])
        .expect("cpu oracle evaluates the per-layer projection")
        .root()
        .to_vec()
}

fn bits_differing(left: &[f32], right: &[f32]) -> usize {
    left.iter()
        .zip(right)
        .filter(|(left_value, right_value)| left_value.to_bits() != right_value.to_bits())
        .count()
}

fn synthetic_f16_weight_bytes() -> Vec<u8> {
    let weight_values = random_vec(0x51A7_0002, REDUCTION_LEN as usize * FEATURES as usize);
    let mut weight_bytes = vec![0u8; weight_values.len() * 2];
    f16::quantize(&weight_values, &mut weight_bytes).expect("weights encode as f16");
    weight_bytes
}

fn assert_projection_is_fully_populated_and_correct(rows: u32, weight_bytes: &[u8]) {
    let features = FEATURES as usize;
    let row_len = REDUCTION_LEN as usize;
    let mut dequantized = vec![0.0f32; row_len * features];
    f16::dequantize(weight_bytes, &mut dequantized).expect("weights decode from f16");
    let activations = random_vec(0x51A7_0001, rows as usize * row_len);

    let metal = run_on_metal(rows, &activations, weight_bytes);
    assert_eq!(metal.len(), rows as usize * features, "rows={rows}: output element count");

    let zero_positions: Vec<usize> = metal
        .iter()
        .enumerate()
        .filter_map(|(position, value)| (value.to_bits() == 0).then_some(position))
        .collect();
    assert!(
        zero_positions.len() <= MAX_COINCIDENTAL_ZERO_OUTPUTS,
        "rows={rows}: {} of {} outputs read back as exactly zero -- the dispatch grid of {} threads \
         was truncated to its low 32 bits",
        zero_positions.len(),
        metal.len(),
        u64::from(rows) * u64::from(FEATURES) * 256
    );
    for position in zero_positions {
        let (row, feature) = (position / features, position % features);
        let dot: f64 = (0..row_len)
            .map(|reduction| {
                f64::from(activations[row * row_len + reduction])
                    * f64::from(dequantized[feature * row_len + reduction])
            })
            .sum();
        assert!(
            dot.abs() <= 1e-3,
            "rows={rows}: output (row {row}, feature {feature}) read back as exactly zero but its dot \
             product is {dot}"
        );
    }

    let reference_rows = LAST_ROW_COUNT_AT_FULL_LANE_WIDTH.min(rows);
    let reference = run_on_metal(
        reference_rows,
        &activations[..reference_rows as usize * row_len],
        weight_bytes,
    );
    let shared = reference_rows as usize * features;
    assert_eq!(
        bits_differing(&metal[..shared], &reference[..shared]),
        0,
        "rows={rows}: the first {reference_rows} rows must be bit-identical to the {reference_rows}-row dispatch"
    );

    let tail_rows = TAIL_ORACLE_ROWS.min(rows);
    let tail_first = (rows - tail_rows) as usize;
    let oracle = run_on_cpu(tail_rows, &activations[tail_first * row_len..], &dequantized);
    let tail = &metal[tail_first * features..];
    assert_eq!(oracle.len(), tail.len(), "rows={rows}: tail oracle element count");
    let worst = tail
        .iter()
        .zip(&oracle)
        .map(|(gpu_value, cpu_value)| (gpu_value - cpu_value).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst <= 1e-3,
        "rows={rows}: the last {tail_rows} rows differ from the cpu oracle by up to {worst} over {} elements",
        tail.len()
    );
}

#[test]
fn per_layer_projection_at_1873_rows_is_fully_populated_and_matches_the_cpu_oracle() {
    assert_projection_is_fully_populated_and_correct(1873, &synthetic_f16_weight_bytes());
}

#[test]
fn per_layer_projection_at_1880_rows_is_fully_populated_and_matches_the_cpu_oracle() {
    assert_projection_is_fully_populated_and_correct(1880, &synthetic_f16_weight_bytes());
}

#[test]
fn per_layer_projection_at_the_last_linear_row_count_is_fully_populated() {
    assert_projection_is_fully_populated_and_correct(
        LAST_ROW_COUNT_AT_FULL_LANE_WIDTH,
        &synthetic_f16_weight_bytes(),
    );
}

const REAL_GEMMA4_E2B_GGUF_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

fn real_per_layer_model_proj_weight_bytes() -> Vec<u8> {
    let mut file = std::fs::File::open(REAL_GEMMA4_E2B_GGUF_PATH)
        .expect("open the real gemma4-E2B checkpoint (host-local blob)");
    let file_len = file.metadata().expect("checkpoint metadata").len();
    let mut prefix_len = 1usize << 22;
    let parsed = loop {
        let mut buffer = vec![0u8; prefix_len];
        file.seek(SeekFrom::Start(0)).expect("seek to start");
        let read = file.read(&mut buffer).expect("read gguf prefix");
        buffer.truncate(read);
        if let Ok((parser, events)) = GgufParser::new().push(&buffer) {
            let mut version = None;
            let mut metadata = Vec::new();
            let mut tensors = Vec::new();
            let mut completion = None;
            for event in events {
                match event {
                    GgufEvent::Header { version: found, .. } => version = Some(found),
                    GgufEvent::Metadata { key, value } => metadata.push((key, value)),
                    GgufEvent::Tensor(tensor) => tensors.push(tensor),
                    GgufEvent::Complete { data_offset, alignment } => {
                        completion = Some((data_offset, alignment));
                    }
                }
            }
            if let (Some(version), Some((data_offset, alignment))) = (version, completion) {
                parser.finish().expect("parser reports complete and clean");
                break ParsedGguf {
                    version,
                    tensor_count: tensors.len() as u64,
                    kv_count: metadata.len() as u64,
                    metadata,
                    tensors,
                    data_offset,
                    alignment,
                };
            }
        }
        assert!(
            (prefix_len as u64) < file_len,
            "the gguf header did not parse within the whole file"
        );
        prefix_len *= 2;
    };
    let tensor = parsed
        .tensors
        .iter()
        .find(|candidate| candidate.name == "per_layer_model_proj.weight")
        .expect("gemma4-E2B carries per_layer_model_proj.weight");
    assert_eq!(tensor.ggml_type, GgmlType::F16, "the real projection weight is F16");
    assert_eq!(
        tensor.dims.as_slice(),
        [u64::from(REDUCTION_LEN), u64::from(FEATURES)]
    );
    let range = parsed
        .tensor_data_range(tensor, file_len)
        .expect("tensor byte range within file bounds");
    let mut bytes = vec![0u8; (range.end - range.start) as usize];
    file.seek(SeekFrom::Start(range.start)).expect("seek to tensor data");
    file.read_exact(&mut bytes).expect("read the whole tensor");
    bytes
}

#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn real_per_layer_model_proj_weight_is_fully_populated_at_the_overflow_row_counts() {
    let weight_bytes = real_per_layer_model_proj_weight_bytes();

    assert_projection_is_fully_populated_and_correct(1873, &weight_bytes);
    assert_projection_is_fully_populated_and_correct(1880, &weight_bytes);
}
