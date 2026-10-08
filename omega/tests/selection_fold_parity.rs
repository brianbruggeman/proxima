//! Correctness gate for a gathered quantized fold whose route moves along a
//! reduced axis: a `Q8_0` expert slab gathered by a `[sequence, selected]`
//! route, with the selected axis AND the input axis both reduced, so one
//! dispatch applies every selected expert of a down projection and sums them
//! into `[sequence, rows]` without a `[sequence, selected, rows]` intermediate.
//!
//! Two oracles that share nothing with the executors under test: a plain loop
//! over the dequantized slab (the definition of the fold), and
//! `proxima_tensor::cpu`'s evaluator over the dequantized slab as f32. The
//! Metal result must match both. Routing is what a top-k router produces: every
//! token names `selected` distinct experts and some experts receive no token.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use omega::MetalError;
use proxima_gguf::quant::q8_0::{BLOCK_BYTES, QK8_0, dequantize, quantize};
use proxima_primitives::Codec;
use proxima_tensor::map::{self, AxisIndex, AxisTerm};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce, ReduceInit,
    ScalarOp, TensorError, append,
};

const RELATIVE_TOLERANCE: f32 = 1.0e-4;

#[derive(Clone, Copy)]
struct Shape {
    sequence: usize,
    selected: usize,
    rows: usize,
    k: usize,
    experts: usize,
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

fn expert_gather_map(route: NodeId) -> IndexMap {
    IndexMap::Computed {
        indices: route,
        index_map: map::projection(4, &[0, 1]),
        base: map::IndexPattern {
            iter_rank: 4,
            axes: vec![
                AxisIndex::default(),
                AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(3)).collect(),
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

fn program(shape: &Shape, weight_dtype: DType) -> (Vec<Op>, NodeId) {
    let Shape { sequence, selected, rows, k, experts } = *shape;
    let mut program = Vec::new();
    let weight = input(&mut program, weight_dtype, &[experts, rows, k], "weight");
    let route = input(&mut program, DType::Float32, &[sequence, selected], "route");
    let activation = input(&mut program, DType::Float32, &[sequence, selected, k], "activation");
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, expert_gather_map(route)),
                (activation, IndexMap::Affine(map::projection(4, &[0, 1, 2]))),
            ],
            name: None,
        },
    );
    let folded = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(4, &[0, 1, 2, 3])),
            out_map: IndexMap::Affine(map::projection(4, &[0, 3])),
            keep: Keep::Reduce,
            name: Some("selection_folded_down".into()),
        }),
    );
    (program, folded)
}

fn unit_values(seed: u64, count: usize, low: f32, high: f32) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count)
        .map(|_| low + (lcg.next_unit() + 1.0) * 0.5 * (high - low))
        .collect()
}

fn quantized_expert_stack(shape: &Shape) -> Vec<u8> {
    let row_bytes = shape.k / QK8_0 * BLOCK_BYTES;
    let mut stack = vec![0u8; shape.experts * shape.rows * row_bytes];
    for expert in 0..shape.experts {
        let magnitude = 0.5 + expert as f32 * 0.1;
        let weights = unit_values(1000 + expert as u64, shape.rows * shape.k, -magnitude, magnitude);
        let expert_bytes = shape.rows * row_bytes;
        let target = &mut stack[expert * expert_bytes..(expert + 1) * expert_bytes];
        for (row, row_target) in weights.chunks_exact(shape.k).zip(target.chunks_exact_mut(row_bytes)) {
            quantize(row, row_target).expect("a row is a whole number of q8_0 blocks");
        }
    }
    stack
}

fn top_k_route(sequence: usize, selected: usize, experts: usize, seed: u64) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    let mut route = Vec::with_capacity(sequence * selected);
    for _ in 0..sequence {
        let mut pool: Vec<usize> = (0..experts).collect();
        for slot in 0..selected {
            let draw = ((lcg.next_unit() + 1.0) * 0.5 * (experts - slot) as f32) as usize;
            let chosen = pool.remove(draw.min(pool.len() - 1));
            route.push(chosen as f32);
        }
    }
    route
}

fn definition_of_the_fold(shape: &Shape, slab: &[f32], route: &[f32], activation: &[f32]) -> Vec<f32> {
    let Shape { sequence, selected, rows, k, .. } = *shape;
    let mut folded = vec![0.0f32; sequence * rows];
    for token in 0..sequence {
        for slot in 0..selected {
            let expert = route[token * selected + slot] as usize;
            let hidden = &activation[(token * selected + slot) * k..(token * selected + slot + 1) * k];
            for row in 0..rows {
                let weights = &slab[(expert * rows + row) * k..(expert * rows + row + 1) * k];
                let dot: f32 = weights.iter().zip(hidden).map(|(weight, value)| weight * value).sum();
                folded[token * rows + row] += dot;
            }
        }
    }
    folded
}

fn named<'data>(
    weight: QuantizedBlock<'data>,
    route: &'data [f32],
    activation: &'data [f32],
) -> Vec<(&'static str, QuantizedBlock<'data>)> {
    vec![
        ("weight", weight),
        ("route", QuantizedBlock::Float32(route)),
        ("activation", QuantizedBlock::Float32(activation)),
    ]
}

struct Outputs {
    definition: Vec<f32>,
    cpu: Vec<f32>,
    metal: Vec<f32>,
}

fn run(shape: &Shape, route: &[f32]) -> Outputs {
    let stack = quantized_expert_stack(shape);
    let mut slab = vec![0.0f32; shape.experts * shape.rows * shape.k];
    dequantize(&stack, &mut slab).expect("the stack is whole q8_0 blocks");
    let activation = unit_values(2000, shape.sequence * shape.selected * shape.k, -1.0, 1.0);
    let definition = definition_of_the_fold(shape, &slab, route, &activation);

    let (reference_program, reference_output) = program(shape, DType::Float32);
    let reference_named = named(QuantizedBlock::Float32(&slab), route, &activation);
    let mut free_buffers = Vec::new();
    let mut validated = None;
    let cpu = proxima_tensor::cpu::evaluate_quantized_named_with_scratch(
        &reference_program,
        &[],
        &reference_named,
        &[reference_output],
        &mut free_buffers,
        &mut validated,
    )
    .expect("cpu evaluates the selection fold over the dequantized slab");

    let (program, output) = program(shape, DType::UInt8);
    let named = named(QuantizedBlock::Packed { codec: Codec::Q8_0, bytes: &stack }, route, &activation);
    let outputs = [output];
    let plan = omega::plan_named(&program, &[], &named, &outputs, NumericPolicy::default())
        .expect("metal plans the selection fold over packed q8_0 experts");
    let metal = omega::execute_plan_named(&plan, &named)
        .expect("metal runs the selection fold over packed q8_0 experts on a real device");
    Outputs { definition, cpu: cpu.root().to_vec(), metal: metal.root().to_vec() }
}

fn worst_relative_row_error(oracle: &[f32], candidate: &[f32], row_width: usize) -> f32 {
    oracle
        .chunks_exact(row_width)
        .zip(candidate.chunks_exact(row_width))
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

fn assert_fold_matches_both_oracles(shape: &Shape, route: &[f32]) {
    let outputs = run(shape, route);
    let element_count = shape.sequence * shape.rows;
    assert_eq!(outputs.definition.len(), element_count, "degenerate: definition produced no output");
    assert_eq!(outputs.cpu.len(), element_count, "degenerate: cpu produced no output");
    assert_eq!(outputs.metal.len(), element_count, "degenerate: metal produced no output");
    let cpu_error = worst_relative_row_error(&outputs.definition, &outputs.cpu, shape.rows);
    let metal_error = worst_relative_row_error(&outputs.definition, &outputs.metal, shape.rows);
    eprintln!(
        "sequence={} selected={} rows={} k={} experts={} cpu_error={cpu_error:e} metal_error={metal_error:e}",
        shape.sequence, shape.selected, shape.rows, shape.k, shape.experts
    );
    assert!(cpu_error <= RELATIVE_TOLERANCE, "cpu drifted from the definition by {cpu_error:e}");
    assert!(metal_error <= RELATIVE_TOLERANCE, "metal drifted from the definition by {metal_error:e}");
}

#[test]
fn a_decode_step_folds_its_selected_experts_into_one_row() {
    let shape = Shape { sequence: 1, selected: 8, rows: 192, k: 512, experts: 32 };
    let route = top_k_route(shape.sequence, shape.selected, shape.experts, 11);
    assert_fold_matches_both_oracles(&shape, &route);
}

#[test]
fn a_verify_width_batch_folds_each_token_over_its_own_experts() {
    let shape = Shape { sequence: 8, selected: 8, rows: 192, k: 512, experts: 32 };
    let route = top_k_route(shape.sequence, shape.selected, shape.experts, 12);
    assert_fold_matches_both_oracles(&shape, &route);
}

#[test]
fn a_prefill_chunk_folds_every_token_over_its_own_experts() {
    let shape = Shape { sequence: 120, selected: 8, rows: 128, k: 256, experts: 32 };
    let route = top_k_route(shape.sequence, shape.selected, shape.experts, 13);
    assert_fold_matches_both_oracles(&shape, &route);
}

#[test]
fn ragged_rows_and_tokens_fold_correctly() {
    let shape = Shape { sequence: 41, selected: 4, rows: 100, k: 256, experts: 6 };
    let route = top_k_route(shape.sequence, shape.selected, shape.experts, 21);
    assert_fold_matches_both_oracles(&shape, &route);
}

#[test]
fn control_an_oracle_routed_to_other_experts_disagrees_by_a_gross_factor() {
    let shape = Shape { sequence: 8, selected: 8, rows: 192, k: 512, experts: 32 };
    let route = top_k_route(shape.sequence, shape.selected, shape.experts, 12);
    let other_route = top_k_route(shape.sequence, shape.selected, shape.experts, 99);
    let outputs = run(&shape, &route);
    let stack = quantized_expert_stack(&shape);
    let mut slab = vec![0.0f32; shape.experts * shape.rows * shape.k];
    dequantize(&stack, &mut slab).expect("the stack is whole q8_0 blocks");
    let activation = unit_values(2000, shape.sequence * shape.selected * shape.k, -1.0, 1.0);
    let wrong = definition_of_the_fold(&shape, &slab, &other_route, &activation);

    let gap = worst_relative_row_error(&wrong, &outputs.metal, shape.rows);

    assert!(gap > 0.1, "the control must disagree by a gross factor, got {gap:e}");
}

#[test]
fn an_expert_index_past_the_slab_raises_the_same_error_as_cpu() {
    let shape = Shape { sequence: 8, selected: 8, rows: 128, k: 256, experts: 12 };
    let mut route = top_k_route(shape.sequence, shape.selected, shape.experts, 31);
    route[10] = shape.experts as f32;
    let stack = quantized_expert_stack(&shape);
    let mut slab = vec![0.0f32; shape.experts * shape.rows * shape.k];
    dequantize(&stack, &mut slab).expect("the stack is whole q8_0 blocks");
    let activation = unit_values(2000, shape.sequence * shape.selected * shape.k, -1.0, 1.0);

    let (reference_program, reference_output) = program(&shape, DType::Float32);
    let reference_named = named(QuantizedBlock::Float32(&slab), &route, &activation);
    let mut free_buffers = Vec::new();
    let mut validated = None;
    let cpu_error = proxima_tensor::cpu::evaluate_quantized_named_with_scratch(
        &reference_program,
        &[],
        &reference_named,
        &[reference_output],
        &mut free_buffers,
        &mut validated,
    )
    .expect_err("cpu rejects an expert index past the slab");
    let TensorError::GatherIndexOutOfRange { extent: cpu_extent, .. } = cpu_error else {
        panic!("cpu raised {cpu_error} instead of GatherIndexOutOfRange");
    };

    let (program, output) = program(&shape, DType::UInt8);
    let named = named(QuantizedBlock::Packed { codec: Codec::Q8_0, bytes: &stack }, &route, &activation);
    let outputs = [output];
    let plan = omega::plan_named(&program, &[], &named, &outputs, NumericPolicy::default())
        .expect("metal plans the selection fold");
    let metal_error = omega::execute_plan_named(&plan, &named)
        .expect_err("metal rejects an expert index past the slab");
    let MetalError::Tensor(TensorError::GatherIndexOutOfRange { index, extent, .. }) = metal_error else {
        panic!("metal raised {metal_error:?} instead of GatherIndexOutOfRange");
    };

    assert_eq!(extent, cpu_extent);
    assert_eq!(index, shape.experts as i64);
}
