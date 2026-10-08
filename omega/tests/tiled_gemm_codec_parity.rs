//! Correctness gate for the codec-generic tiled GEMM: every codec with a
//! `tiled_decode` description (`Q8_0`, `Float16`, `Q3_K`, `Q5_0`, `Q5_1`,
//! `Q5_K`, `Q6_K`, plus the `Q4_0` and `Q4_K` the description absorbed) is
//! multiplied against a dense activation through the dense tiled kernel and,
//! as a gathered expert slab, through the expert-grouped kernel, at prefill
//! width. The oracle is the CPU dequant path: `proxima_gguf`'s own `dequantize`
//! of the same packed bytes (the decoders llama.cpp's `dequantize_*` are held to),
//! then a plain f32 product accumulated in f64 in source order, so no
//! activation quantization (the K-quant CPU matmul's int8-dot arm) and no
//! reduction-order slack enter it. The Metal kernels stage the dequantized
//! weight tile as `half`, so the two agree to half-precision weight rounding;
//! the tolerance below is that rounding.
//!
//! The weights are produced by each codec's canonical encoder (`proxima_gguf`'s
//! `quantize`, and a port of ggml's `quantize_row_q5_0_ref`/`_q5_1_ref` for the
//! two legacy codecs `proxima_gguf` only decodes), so every block carries the
//! scale, min, high-bit and sub-block structure a real checkpoint's does, not a
//! stub. Shapes leave partial row and token tiles on both edges, and the
//! flat-block codecs also run at a reduce extent (160) the 256-element
//! row-blocked kernel refuses, which is the admission the description's block
//! (32) buys. Each case asserts the tiled kernel was the one selected and that
//! it produced output, and a control compares the Metal result against an
//! oracle built from other weights, which must fail.

#![cfg(all(feature = "metal", feature = "metal-tiled-gemm", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_gguf::quant::{f16, q3_k, q4_0, q4_k, q5_0, q5_1, q5_k, q6_k, q8_0};
use proxima_gguf::types::GgmlType;
use proxima_primitives::Codec;
use proxima_tensor::map::{self, AxisIndex, AxisTerm};
use rayon::prelude::*;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce, ReduceInit,
    ScalarOp, append,
};

const RELATIVE_TOLERANCE: f32 = 2.0e-3;

struct Outputs {
    oracle: Vec<f32>,
    metal: Vec<f32>,
    kernel_keys: Vec<String>,
}

struct MetalRun {
    output: Vec<f32>,
    kernel_keys: Vec<String>,
}

fn ggml_type(codec: Codec) -> GgmlType {
    match codec {
        Codec::Q4_0 => GgmlType::Q4_0,
        Codec::Q4K => GgmlType::Q4_K,
        Codec::Q5_0 => GgmlType::Q5_0,
        Codec::Q5_1 => GgmlType::Q5_1,
        Codec::Q8_0 => GgmlType::Q8_0,
        Codec::Q3K => GgmlType::Q3_K,
        Codec::Q5K => GgmlType::Q5_K,
        Codec::Q6K => GgmlType::Q6_K,
        Codec::Float16 => GgmlType::F16,
        other => panic!("{other:?} has no tiled decode description, so no case here"),
    }
}

fn row_bytes(codec: Codec, row_length: usize) -> usize {
    let layout = ggml_type(codec).block_layout();
    let elements = layout.block_elements as usize;
    assert_eq!(row_length % elements, 0, "{codec:?}: a row is a whole number of blocks");
    row_length / elements * layout.block_bytes as usize
}

fn unit_values(seed: u64, count: usize, low: f32, high: f32) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count)
        .map(|_| low + (lcg.next_unit() + 1.0) * 0.5 * (high - low))
        .collect()
}

fn fp16_bytes(value: f32) -> [u8; 2] {
    half::f16::from_f32(value).to_le_bytes()
}

/// ggml's `quantize_row_q5_0_ref`: the scale is the signed extreme over -16,
/// each level is the value over the scale plus 16.5 truncated and capped at 31,
/// and the 5th bit of element `j` lands in bit `j` (first half) or `j + 16`
/// (second half) of the 32-bit `qh`.
fn quantize_q5_0(row: &[f32], output: &mut [u8]) {
    let (blocks, _) = row.as_chunks::<32>();
    let (targets, _) = output.as_chunks_mut::<22>();
    for (block, target) in blocks.iter().zip(targets.iter_mut()) {
        let (_, extreme) = block.iter().fold((0.0f32, 0.0f32), |(amax, extreme), &value| {
            if amax < value.abs() { (value.abs(), value) } else { (amax, extreme) }
        });
        let scale = extreme / -16.0;
        let inverse = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        target[0..2].copy_from_slice(&fp16_bytes(scale));
        let mut high_bits = 0u32;
        for index in 0..16 {
            let low = (((block[index] * inverse + 16.5) as i32).clamp(0, 31)) as u32;
            let high = (((block[16 + index] * inverse + 16.5) as i32).clamp(0, 31)) as u32;
            target[6 + index] = ((low & 0x0F) | ((high & 0x0F) << 4)) as u8;
            high_bits |= ((low & 0x10) >> 4) << index;
            high_bits |= ((high & 0x10) >> 4) << (index + 16);
        }
        target[2..6].copy_from_slice(&high_bits.to_le_bytes());
    }
}

/// ggml's `quantize_row_q5_1_ref`: scale `(max - min) / 31`, min kept, each
/// level the value over the scale from the min, rounded.
fn quantize_q5_1(row: &[f32], output: &mut [u8]) {
    let (blocks, _) = row.as_chunks::<32>();
    let (targets, _) = output.as_chunks_mut::<24>();
    for (block, target) in blocks.iter().zip(targets.iter_mut()) {
        let minimum = block.iter().copied().fold(f32::INFINITY, f32::min);
        let maximum = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = (maximum - minimum) / 31.0;
        let inverse = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        target[0..2].copy_from_slice(&fp16_bytes(scale));
        target[2..4].copy_from_slice(&fp16_bytes(minimum));
        let mut high_bits = 0u32;
        for index in 0..16 {
            let low = (((block[index] - minimum) * inverse + 0.5) as i32).clamp(0, 31) as u32;
            let high = (((block[16 + index] - minimum) * inverse + 0.5) as i32).clamp(0, 31) as u32;
            target[8 + index] = ((low & 0x0F) | ((high & 0x0F) << 4)) as u8;
            high_bits |= ((low & 0x10) >> 4) << index;
            high_bits |= ((high & 0x10) >> 4) << (index + 16);
        }
        target[4..8].copy_from_slice(&high_bits.to_le_bytes());
    }
}

fn quantize_row(codec: Codec, row: &[f32], output: &mut [u8]) {
    match codec {
        Codec::Q4_0 => q4_0::quantize(row, output).expect("q4_0 row"),
        Codec::Q4K => q4_k::quantize(row, output).expect("q4_k row"),
        Codec::Q8_0 => q8_0::quantize(row, output).expect("q8_0 row"),
        Codec::Q3K => q3_k::quantize(row, output).expect("q3_k row"),
        Codec::Q5K => q5_k::quantize(row, output).expect("q5_k row"),
        Codec::Q6K => q6_k::quantize(row, output).expect("q6_k row"),
        Codec::Float16 => f16::quantize(row, output).expect("f16 row"),
        Codec::Q5_0 => quantize_q5_0(row, output),
        Codec::Q5_1 => quantize_q5_1(row, output),
        other => panic!("{other:?} has no encoder here"),
    }
}

fn quantize_rows(codec: Codec, values: &[f32], row_length: usize) -> Vec<u8> {
    let encoded_row = row_bytes(codec, row_length);
    let mut bytes = vec![0u8; values.len() / row_length * encoded_row];
    for (row, target) in values
        .chunks_exact(row_length)
        .zip(bytes.chunks_exact_mut(encoded_row))
    {
        quantize_row(codec, row, target);
    }
    bytes
}

fn dequantize_all(codec: Codec, bytes: &[u8], count: usize) -> Vec<f32> {
    let mut values = vec![0.0f32; count];
    let decoded = match codec {
        Codec::Q4_0 => q4_0::dequantize(bytes, &mut values),
        Codec::Q4K => q4_k::dequantize(bytes, &mut values),
        Codec::Q8_0 => q8_0::dequantize(bytes, &mut values),
        Codec::Q3K => q3_k::dequantize(bytes, &mut values),
        Codec::Q5K => q5_k::dequantize(bytes, &mut values),
        Codec::Q6K => q6_k::dequantize(bytes, &mut values),
        Codec::Float16 => f16::dequantize(bytes, &mut values),
        Codec::Q5_0 => q5_0::dequantize(bytes, &mut values),
        Codec::Q5_1 => q5_1::dequantize(bytes, &mut values),
        other => panic!("{other:?} has no decoder here"),
    };
    decoded.expect("the packed bytes are whole blocks of their codec");
    values
}

fn input(program: &mut Vec<Op>, dtype: DType, shape: &[usize], name: &str) -> NodeId {
    append(
        program,
        Op::Input {
            dtype,
            shape: shape.iter().map(|&extent| Extent::Static(extent as u32)).collect(),
            name: Some(name.into()),
        },
    )
}

fn reduce_over_last_axis(program: &mut Vec<Op>, product: NodeId, name: &str) -> NodeId {
    append(
        program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some(name.into()),
        }),
    )
}

struct DenseShape {
    tokens: usize,
    rows: usize,
    k: usize,
}

fn dense_program(shape: &DenseShape) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = input(&mut program, DType::UInt8, &[shape.rows, shape.k], "weight");
    let activation = input(&mut program, DType::Float32, &[shape.tokens, shape.k], "activation");
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(map::projection(3, &[1, 2]))),
                (activation, IndexMap::Affine(map::projection(3, &[0, 2]))),
            ],
            name: None,
        },
    );
    let sum = reduce_over_last_axis(&mut program, product, "dense_product");
    (program, sum)
}

struct GroupedShape {
    tokens: usize,
    rows: usize,
    k: usize,
    experts: usize,
}

fn expert_gather_map(route: NodeId) -> IndexMap {
    IndexMap::Computed {
        indices: route,
        index_map: map::projection(3, &[0]),
        base: map::IndexPattern {
            iter_rank: 3,
            axes: vec![
                AxisIndex::default(),
                AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(1)).collect(),
                    offset: 0,
                    len: None,
                },
                AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(2)).collect(),
                    offset: 0,
                    len: None,
                },
            ],
        },
        gathered_dim: 0,
    }
}

fn grouped_program(shape: &GroupedShape) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = input(
        &mut program,
        DType::UInt8,
        &[shape.experts, shape.rows, shape.k],
        "weight",
    );
    let route = input(&mut program, DType::Int32, &[shape.tokens], "route");
    let activation = input(&mut program, DType::Float32, &[shape.tokens, shape.k], "activation");
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, expert_gather_map(route)),
                (activation, IndexMap::Affine(map::projection(3, &[0, 2]))),
            ],
            name: None,
        },
    );
    let sum = reduce_over_last_axis(&mut program, product, "expert_product");
    (program, sum)
}

fn heavy_expert_route(tokens: usize, experts: usize) -> Vec<f32> {
    let mut lcg = Lcg(77);
    let populated = [0usize, 1, 5, experts - 2];
    (0..tokens)
        .map(|_| {
            let draw = (lcg.next_unit() + 1.0) * 0.5;
            if draw < 0.6 {
                3.0
            } else {
                populated[((draw * 1000.0) as usize) % populated.len()] as f32
            }
        })
        .collect()
}

fn run_metal(program: &[Op], output: NodeId, named: &[(&str, QuantizedBlock<'_>)]) -> MetalRun {
    let outputs = [output];
    let plan = omega::plan_named(program, &[], named, &outputs, NumericPolicy::default())
        .expect("metal plans the packed product");
    let kernel_keys = plan.kernel_keys().expect("plan exposes its kernel keys");
    let metal = omega::execute_plan_named(&plan, named)
        .expect("metal runs the packed product on a real device");
    MetalRun {
        output: metal.root().to_vec(),
        kernel_keys,
    }
}

fn weight_values(seed: u64, count: usize) -> Vec<f32> {
    unit_values(seed, count, -1.5, 1.5)
}

fn dot_in_f64(row: &[f32], token: &[f32]) -> f32 {
    let sum: f64 = row
        .iter()
        .zip(token)
        .map(|(weight, value)| f64::from(*weight) * f64::from(*value))
        .sum();
    sum as f32
}

fn dense_oracle(weights: &[f32], activation: &[f32], shape: &DenseShape) -> Vec<f32> {
    let per_token: Vec<Vec<f32>> = activation
        .par_chunks_exact(shape.k)
        .map(|token| weights.chunks_exact(shape.k).map(|row| dot_in_f64(row, token)).collect())
        .collect();
    per_token.concat()
}

fn grouped_oracle(weights: &[f32], route: &[f32], activation: &[f32], shape: &GroupedShape) -> Vec<f32> {
    let expert_len = shape.rows * shape.k;
    let tokens: Vec<(&[f32], f32)> = activation.chunks_exact(shape.k).zip(route.iter().copied()).collect();
    let per_token: Vec<Vec<f32>> = tokens
        .par_iter()
        .map(|(token, expert)| {
            let slab = &weights[*expert as usize * expert_len..][..expert_len];
            slab.chunks_exact(shape.k).map(|row| dot_in_f64(row, token)).collect()
        })
        .collect();
    per_token.concat()
}

fn run_dense(codec: Codec, shape: &DenseShape, weight_seed: u64, oracle_seed: u64) -> Outputs {
    let (program, output) = dense_program(shape);
    let count = shape.rows * shape.k;
    let bytes = quantize_rows(codec, &weight_values(weight_seed, count), shape.k);
    let activation = unit_values(2000, shape.tokens * shape.k, -1.0, 1.0);
    let named = [
        ("weight", QuantizedBlock::Packed { codec, bytes: &bytes }),
        ("activation", QuantizedBlock::Float32(&activation)),
    ];
    let metal = run_metal(&program, output, &named);
    let oracle_bytes = quantize_rows(codec, &weight_values(oracle_seed, count), shape.k);
    let oracle_weights = dequantize_all(codec, &oracle_bytes, count);
    Outputs {
        oracle: dense_oracle(&oracle_weights, &activation, shape),
        metal: metal.output,
        kernel_keys: metal.kernel_keys,
    }
}

fn run_grouped(codec: Codec, shape: &GroupedShape, route: &[f32], weight_seed: u64, oracle_seed: u64) -> Outputs {
    let (program, output) = grouped_program(shape);
    let count = shape.experts * shape.rows * shape.k;
    let bytes = quantize_rows(codec, &weight_values(weight_seed, count), shape.k);
    let activation = unit_values(2000, shape.tokens * shape.k, -1.0, 1.0);
    let named = [
        ("weight", QuantizedBlock::Packed { codec, bytes: &bytes }),
        ("route", QuantizedBlock::Float32(route)),
        ("activation", QuantizedBlock::Float32(&activation)),
    ];
    let metal = run_metal(&program, output, &named);
    let oracle_bytes = quantize_rows(codec, &weight_values(oracle_seed, count), shape.k);
    let oracle_weights = dequantize_all(codec, &oracle_bytes, count);
    Outputs {
        oracle: grouped_oracle(&oracle_weights, route, &activation, shape),
        metal: metal.output,
        kernel_keys: metal.kernel_keys,
    }
}

fn dense_tiled_kernel_ran(keys: &[String]) -> bool {
    keys.iter().any(|key| key.contains("_mml") && key.contains("_wws"))
}

fn grouped_kernel_ran(keys: &[String]) -> bool {
    keys.iter().any(|key| key.contains("_g10") && key.contains("E_w128"))
}

fn worst_relative_row_error(oracle: &[f32], metal: &[f32], row_width: usize) -> f32 {
    oracle
        .chunks_exact(row_width)
        .zip(metal.chunks_exact(row_width))
        .map(|(want_row, got_row)| {
            let norm = want_row.iter().map(|value| value * value).sum::<f32>().sqrt();
            let worst = want_row
                .iter()
                .zip(got_row)
                .map(|(want, got)| (want - got).abs())
                .fold(0.0f32, f32::max);
            worst / norm.max(1.0e-6)
        })
        .fold(0.0f32, f32::max)
}

fn assert_matches_oracle(label: &str, outputs: &Outputs, element_count: usize, row_width: usize) {
    assert_eq!(outputs.oracle.len(), element_count, "{label}: degenerate, oracle produced no output");
    assert_eq!(outputs.metal.len(), element_count, "{label}: degenerate, metal produced no output");
    let worst = worst_relative_row_error(&outputs.oracle, &outputs.metal, row_width);
    eprintln!("{label}: elements={element_count} worst_relative_row_error={worst:e}");
    assert!(
        worst <= RELATIVE_TOLERANCE,
        "{label}: tiled result drifted from the f32 oracle by {worst:e} of a row's norm"
    );
}

fn assert_dense_codec_matches(codec: Codec, shape: DenseShape) {
    let outputs = run_dense(codec, &shape, 1000, 1000);
    assert!(
        dense_tiled_kernel_ran(&outputs.kernel_keys),
        "{codec:?}: the dense tiled kernel was never selected: {:?}",
        outputs.kernel_keys
    );
    assert_matches_oracle(
        &format!("dense {codec:?} tokens={} rows={} k={}", shape.tokens, shape.rows, shape.k),
        &outputs,
        shape.tokens * shape.rows,
        shape.rows,
    );
}

fn assert_grouped_codec_matches(codec: Codec, shape: GroupedShape) {
    let route = heavy_expert_route(shape.tokens, shape.experts);
    let outputs = run_grouped(codec, &shape, &route, 1000, 1000);
    assert!(
        grouped_kernel_ran(&outputs.kernel_keys),
        "{codec:?}: the expert-grouped kernel was never selected: {:?}",
        outputs.kernel_keys
    );
    assert_matches_oracle(
        &format!(
            "grouped {codec:?} tokens={} rows={} k={} experts={}",
            shape.tokens, shape.rows, shape.k, shape.experts
        ),
        &outputs,
        shape.tokens * shape.rows,
        shape.rows,
    );
}

#[proxima::test(runtime = "tokio")]
#[case::q8_0(Codec::Q8_0)]
#[case::float16(Codec::Float16)]
#[case::q3k(Codec::Q3K)]
#[case::q5_0(Codec::Q5_0)]
#[case::q5_1(Codec::Q5_1)]
#[case::q5k(Codec::Q5K)]
#[case::q6k(Codec::Q6K)]
#[case::q4_0(Codec::Q4_0)]
#[case::q4k(Codec::Q4K)]
async fn dense_tiled_gemm_matches_the_cpu_dequant_oracle_with_partial_row_and_token_tiles(#[case] codec: Codec) {
    assert_dense_codec_matches(codec, DenseShape { tokens: 200, rows: 100, k: 512 });
}

#[proxima::test(runtime = "tokio")]
#[case::q8_0(Codec::Q8_0)]
#[case::float16(Codec::Float16)]
#[case::q5_0(Codec::Q5_0)]
#[case::q5_1(Codec::Q5_1)]
#[case::q4_0(Codec::Q4_0)]
async fn flat_codecs_run_the_tiled_gemm_at_an_extent_the_row_blocked_kernel_refuses(#[case] codec: Codec) {
    assert_dense_codec_matches(codec, DenseShape { tokens: 193, rows: 70, k: 160 });
}

#[test]
fn dense_1024_by_512_projection_shape_matches_for_q8_0() {
    assert_dense_codec_matches(Codec::Q8_0, DenseShape { tokens: 510, rows: 1024, k: 1024 });
}

#[test]
fn dense_per_layer_projection_shape_matches_for_float16() {
    assert_dense_codec_matches(Codec::Float16, DenseShape { tokens: 200, rows: 640, k: 1536 });
}

#[test]
fn control_dense_oracle_built_from_other_weights_disagrees_by_a_gross_factor() {
    for codec in [Codec::Q8_0, Codec::Float16, Codec::Q3K, Codec::Q5_0, Codec::Q5_1, Codec::Q5K, Codec::Q6K] {
        let shape = DenseShape { tokens: 200, rows: 100, k: 512 };
        let outputs = run_dense(codec, &shape, 1000, 4000);
        let gross = worst_relative_row_error(&outputs.oracle, &outputs.metal, shape.rows);
        assert!(
            gross > RELATIVE_TOLERANCE,
            "{codec:?}: a metal result compared against an oracle of other weights must not pass: {gross:e}"
        );
    }
}

#[proxima::test(runtime = "tokio")]
#[case::q8_0(Codec::Q8_0)]
#[case::float16(Codec::Float16)]
#[case::q3k(Codec::Q3K)]
#[case::q5_0(Codec::Q5_0)]
#[case::q5_1(Codec::Q5_1)]
#[case::q5k(Codec::Q5K)]
#[case::q6k(Codec::Q6K)]
#[case::q4_0(Codec::Q4_0)]
#[case::q4k(Codec::Q4K)]
async fn expert_grouped_gemm_matches_the_cpu_dequant_oracle_for_a_heavy_expert_and_empty_experts(
    #[case] codec: Codec,
) {
    assert_grouped_codec_matches(codec, GroupedShape { tokens: 300, rows: 192, k: 512, experts: 8 });
}

#[proxima::test(runtime = "tokio")]
#[case::q8_0(Codec::Q8_0)]
#[case::q5_0(Codec::Q5_0)]
#[case::q5_1(Codec::Q5_1)]
#[case::float16(Codec::Float16)]
async fn expert_grouped_gemm_runs_flat_codecs_at_an_extent_the_row_blocked_kernel_refuses(#[case] codec: Codec) {
    assert_grouped_codec_matches(codec, GroupedShape { tokens: 161, rows: 100, k: 160, experts: 6 });
}

#[test]
fn control_grouped_oracle_built_from_other_experts_disagrees_by_a_gross_factor() {
    for codec in [Codec::Q8_0, Codec::Float16, Codec::Q3K, Codec::Q5_0, Codec::Q5_1, Codec::Q5K, Codec::Q6K] {
        let shape = GroupedShape { tokens: 300, rows: 192, k: 512, experts: 8 };
        let route = heavy_expert_route(shape.tokens, shape.experts);
        let outputs = run_grouped(codec, &shape, &route, 1000, 4000);
        let gross = worst_relative_row_error(&outputs.oracle, &outputs.metal, shape.rows);
        assert!(
            gross > RELATIVE_TOLERANCE,
            "{codec:?}: a metal result compared against an oracle of other experts must not pass: {gross:e}"
        );
    }
}

#[test]
fn the_q5_0_encoder_round_trips_through_the_cpu_dequantizer_within_one_step() {
    let values = weight_values(31, 32 * 12);
    let bytes = quantize_rows(Codec::Q5_0, &values, 32 * 12);
    let decoded = dequantize_all(Codec::Q5_0, &bytes, values.len());
    let (blocks, _) = values.as_chunks::<32>();
    let (decoded_blocks, _) = decoded.as_chunks::<32>();
    for (block_index, (block, decoded_block)) in blocks.iter().zip(decoded_blocks).enumerate() {
        let amax = block.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
        let step = amax / 16.0;
        for (want, got) in block.iter().zip(decoded_block) {
            assert!(
                (want - got).abs() <= step * 1.05 + 1.0e-3,
                "block {block_index}: {want} decoded as {got}, one step is {step}"
            );
        }
    }
}

#[test]
fn the_q5_1_encoder_round_trips_through_the_cpu_dequantizer_within_one_step() {
    let values = weight_values(32, 32 * 12);
    let bytes = quantize_rows(Codec::Q5_1, &values, 32 * 12);
    let decoded = dequantize_all(Codec::Q5_1, &bytes, values.len());
    let (blocks, _) = values.as_chunks::<32>();
    let (decoded_blocks, _) = decoded.as_chunks::<32>();
    for (block_index, (block, decoded_block)) in blocks.iter().zip(decoded_blocks).enumerate() {
        let minimum = block.iter().copied().fold(f32::INFINITY, f32::min);
        let maximum = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let step = (maximum - minimum) / 31.0;
        for (want, got) in block.iter().zip(decoded_block) {
            assert!(
                (want - got).abs() <= step * 0.55 + 1.0e-3,
                "block {block_index}: {want} decoded as {got}, one step is {step}"
            );
        }
    }
}
