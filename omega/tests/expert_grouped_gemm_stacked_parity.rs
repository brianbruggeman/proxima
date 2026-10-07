//! Correctness gate for the stacked form of the expert-grouped tiled GEMM
//! (`metal-grouped-gemm`): a `Q8_0` expert slab gathered by a
//! `[sequence, selected]` route, reduced over the input axis, so one dispatch
//! covers every selected expert of a projection. The token group is two axes,
//! the kernel decomposes each flat token into `(sequence, selected)`, and the
//! activation is either one row shared by every selected expert (gate and up)
//! or its own row per selected expert (down). The oracle is
//! `proxima_tensor::cpu`'s evaluator over the same Q8_0 bytes dequantized to
//! f32 (the CPU quantized-matmul path expresses one token axis only); the Metal
//! kernel stages the weight tile as `half`, so the tolerance is that rounding.
//!
//! Routing is what a top-k router produces: every token names `selected`
//! distinct experts, some experts receive no token, and one expert receives far
//! more than a tile of them.

#![cfg(all(feature = "metal", feature = "metal-grouped-gemm", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_gguf::quant::q8_0::{BLOCK_BYTES, QK8_0, dequantize, quantize};
use proxima_primitives::Codec;
use proxima_tensor::map::{self, AxisIndex, AxisTerm};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce, ReduceInit,
    ScalarOp, TensorError, append,
};
use omega::MetalError;

const MIN_TOKENS_FOR_TILES: usize = 160;
const RELATIVE_TOLERANCE: f32 = 2.0e-3;

struct Shape {
    sequence: usize,
    selected: usize,
    rows: usize,
    k: usize,
    experts: usize,
}

struct Case {
    shape: Shape,
    route: Vec<f32>,
    activation_per_selected: bool,
    fuse_gate: bool,
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
                    terms: core::iter::once(AxisTerm::projection(2)).collect(),
                    offset: 0,
                    len: None,
                },
                AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(3)).collect(),
                    offset: 0,
                    len: None,
                },
            ],
        },
        gathered_dim: 0,
    }
}

fn program(case: &Case, weight_dtype: DType) -> (Vec<Op>, NodeId) {
    let Shape { sequence, selected, rows, k, experts } = case.shape;
    let mut program = Vec::new();
    let weight = input(&mut program, weight_dtype, &[experts, rows, k], "weight");
    let route = input(&mut program, DType::Float32, &[sequence, selected], "route");
    let (activation, activation_map) = if case.activation_per_selected {
        (
            input(&mut program, DType::Float32, &[sequence, selected, k], "activation"),
            map::projection(4, &[0, 1, 3]),
        )
    } else {
        (
            input(&mut program, DType::Float32, &[sequence, k], "activation"),
            map::projection(4, &[0, 3]),
        )
    };
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, expert_gather_map(route)),
                (activation, IndexMap::Affine(activation_map)),
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
            in_map: IndexMap::Affine(map::projection(4, &[0, 1, 2, 3])),
            out_map: IndexMap::Affine(map::projection(4, &[0, 1, 2])),
            keep: Keep::Reduce,
            name: Some("stacked_expert_product".into()),
        }),
    );
    if !case.fuse_gate {
        return (program, sum);
    }
    let gate = input(&mut program, DType::Float32, &[sequence, selected, rows], "gate");
    let gated = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (sum, IndexMap::Affine(map::projection(3, &[0, 1, 2]))),
                (gate, IndexMap::Affine(map::projection(3, &[0, 1, 2]))),
            ],
            name: None,
        },
    );
    (program, gated)
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
        let magnitude = 0.5 + expert as f32;
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

fn heavy_expert_top_k_route(sequence: usize, selected: usize, experts: usize) -> Vec<f32> {
    let mut route = top_k_route(sequence, selected, experts, 77);
    for token in 0..sequence {
        let row = &mut route[token * selected..(token + 1) * selected];
        if token % 5 != 0 && !row.contains(&3.0) {
            row[token % selected] = 3.0;
        }
    }
    route
}

struct Outputs {
    oracle: Vec<f32>,
    metal: Vec<f32>,
    kernel_keys: Vec<String>,
}

fn dequantized(shape: &Shape, stack: &[u8]) -> Vec<f32> {
    let mut values = vec![0.0f32; shape.experts * shape.rows * shape.k];
    dequantize(stack, &mut values).expect("the stack is whole q8_0 blocks");
    values
}

fn named_blocks<'data>(
    case: &'data Case,
    weight: QuantizedBlock<'data>,
    activation: &'data [f32],
    gate: &'data [f32],
) -> Vec<(&'static str, QuantizedBlock<'data>)> {
    let mut named = vec![
        ("weight", weight),
        ("route", QuantizedBlock::Float32(&case.route)),
        ("activation", QuantizedBlock::Float32(activation)),
    ];
    if case.fuse_gate {
        named.push(("gate", QuantizedBlock::Float32(gate)));
    }
    named
}

fn activation_len(case: &Case) -> usize {
    let Shape { sequence, selected, k, .. } = case.shape;
    if case.activation_per_selected { sequence * selected * k } else { sequence * k }
}

fn run(case: &Case) -> Outputs {
    let Shape { sequence, selected, rows, .. } = case.shape;
    let stack = quantized_expert_stack(&case.shape);
    let reference_stack = dequantized(&case.shape, &stack);
    let activation = unit_values(2000, activation_len(case), -1.0, 1.0);
    let gate = unit_values(3000, sequence * selected * rows, 0.5, 1.5);
    let (reference_program, reference_output) = program(case, DType::Float32);
    let reference_named =
        named_blocks(case, QuantizedBlock::Float32(&reference_stack), &activation, &gate);
    let mut free_buffers = Vec::new();
    let mut validated = None;
    let oracle = proxima_tensor::cpu::evaluate_quantized_named_with_scratch(
        &reference_program,
        &[],
        &reference_named,
        &[reference_output],
        &mut free_buffers,
        &mut validated,
    )
    .expect("cpu evaluates the stacked expert product over the dequantized stack");
    let (program, output) = program(case, DType::UInt8);
    let named = named_blocks(
        case,
        QuantizedBlock::Packed { codec: Codec::Q8_0, bytes: &stack },
        &activation,
        &gate,
    );
    let outputs = [output];
    let plan = omega::plan_named(&program, &[], &named, &outputs, NumericPolicy::default())
        .expect("metal plans the stacked q8_0 expert product");
    let kernel_keys = plan.kernel_keys().expect("plan exposes its kernel keys");
    let metal = omega::execute_plan_named(&plan, &named)
        .expect("metal runs the stacked q8_0 expert product on a real device");
    Outputs { oracle: oracle.root().to_vec(), metal: metal.root().to_vec(), kernel_keys }
}

fn stacked_grouped_kernel_ran(keys: &[String]) -> bool {
    let bodies = ["F_w128", "F1_w128", "FN_w128"];
    keys.iter()
        .any(|key| key.contains("_g10") && bodies.iter().any(|body| key.contains(body)))
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

fn assert_stacked_matches_oracle(case: &Case) {
    let outputs = run(case);
    let Shape { sequence, selected, rows, .. } = case.shape;
    let element_count = sequence * selected * rows;
    assert_eq!(outputs.oracle.len(), element_count, "degenerate: oracle produced no output");
    assert_eq!(outputs.metal.len(), element_count, "degenerate: metal produced no output");
    assert!(
        stacked_grouped_kernel_ran(&outputs.kernel_keys),
        "the stacked expert-grouped kernel was never selected: {:?}",
        outputs.kernel_keys
    );
    let worst = worst_relative_row_error(&outputs.oracle, &outputs.metal, rows);
    eprintln!(
        "sequence={sequence} selected={selected} rows={rows} k={} experts={} per_selected_activation={} fuse_gate={} worst_relative_row_error={worst:e}",
        case.shape.k, case.shape.experts, case.activation_per_selected, case.fuse_gate
    );
    assert!(
        worst <= RELATIVE_TOLERANCE,
        "stacked result drifted from the f32 oracle by {worst:e} of a row's norm"
    );
}

#[test]
fn a_shared_activation_gate_over_top_k_routing_matches_the_f32_oracle() {
    let shape = Shape { sequence: 120, selected: 8, rows: 192, k: 512, experts: 32 };
    let route = top_k_route(shape.sequence, shape.selected, shape.experts, 5);
    assert_stacked_matches_oracle(&Case { shape, route, activation_per_selected: false, fuse_gate: false });
}

#[test]
fn a_per_selected_activation_down_over_top_k_routing_matches_the_f32_oracle() {
    let shape = Shape { sequence: 120, selected: 8, rows: 128, k: 256, experts: 32 };
    let route = top_k_route(shape.sequence, shape.selected, shape.experts, 9);
    assert_stacked_matches_oracle(&Case { shape, route, activation_per_selected: true, fuse_gate: false });
}

#[test]
fn one_expert_named_by_most_tokens_spans_many_token_tiles() {
    let shape = Shape { sequence: 150, selected: 4, rows: 128, k: 256, experts: 12 };
    let route = heavy_expert_top_k_route(shape.sequence, shape.selected, shape.experts);
    assert_stacked_matches_oracle(&Case { shape, route, activation_per_selected: false, fuse_gate: false });
}

#[test]
fn ragged_rows_and_tokens_match_the_f32_oracle() {
    let shape = Shape { sequence: 41, selected: 4, rows: 100, k: 256, experts: 6 };
    let route = top_k_route(shape.sequence, shape.selected, shape.experts, 21);
    assert_stacked_matches_oracle(&Case { shape, route, activation_per_selected: true, fuse_gate: false });
}

#[test]
fn a_fused_epilogue_reads_its_operands_at_the_sequence_and_selected_coordinates() {
    let shape = Shape { sequence: 60, selected: 4, rows: 192, k: 512, experts: 8 };
    let route = top_k_route(shape.sequence, shape.selected, shape.experts, 33);
    assert_stacked_matches_oracle(&Case { shape, route, activation_per_selected: false, fuse_gate: true });
}

#[test]
fn below_the_tiled_token_minimum_the_stacked_op_keeps_the_dense_gather_kernel() {
    let shape = Shape { sequence: 1, selected: 8, rows: 128, k: 256, experts: 32 };
    let route = top_k_route(shape.sequence, shape.selected, shape.experts, 41);
    assert!(shape.sequence * shape.selected < MIN_TOKENS_FOR_TILES);
    let outputs = run(&Case { shape, route, activation_per_selected: false, fuse_gate: false });
    assert!(
        !stacked_grouped_kernel_ran(&outputs.kernel_keys),
        "a decode step must not take the prefill tile kernel: {:?}",
        outputs.kernel_keys
    );
    let worst = worst_relative_row_error(&outputs.oracle, &outputs.metal, 128);
    assert!(worst <= RELATIVE_TOLERANCE, "decode-width stacked op drifted by {worst:e}");
}

#[test]
fn an_expert_index_past_the_slab_raises_the_same_error_as_cpu() {
    let shape = Shape { sequence: 60, selected: 4, rows: 64, k: 256, experts: 8 };
    let mut route = top_k_route(shape.sequence, shape.selected, shape.experts, 3);
    route[137] = shape.experts as f32 + 5.0;
    let case = Case { shape, route, activation_per_selected: false, fuse_gate: false };
    let stack = quantized_expert_stack(&case.shape);
    let reference_stack = dequantized(&case.shape, &stack);
    let activation = unit_values(2000, activation_len(&case), -1.0, 1.0);
    let (reference_program, reference_output) = program(&case, DType::Float32);
    let reference_named =
        named_blocks(&case, QuantizedBlock::Float32(&reference_stack), &activation, &[]);
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
    .expect_err("cpu rejects the out-of-range expert");
    let (program, output) = program(&case, DType::UInt8);
    let named = named_blocks(
        &case,
        QuantizedBlock::Packed { codec: Codec::Q8_0, bytes: &stack },
        &activation,
        &[],
    );
    let outputs = [output];
    let plan = omega::plan_named(&program, &[], &named, &outputs, NumericPolicy::default())
        .expect("metal plans the stacked q8_0 expert product");
    let metal_error = omega::execute_plan_named(&plan, &named)
        .expect_err("metal rejects the out-of-range expert instead of clamping it away");
    let TensorError::GatherIndexOutOfRange { node: cpu_node, extent: cpu_extent, .. } = cpu_error
    else {
        panic!("cpu raised {cpu_error} instead of GatherIndexOutOfRange");
    };
    let MetalError::Tensor(TensorError::GatherIndexOutOfRange { node, index, extent }) = metal_error
    else {
        panic!("metal raised {metal_error:?} instead of GatherIndexOutOfRange");
    };
    assert_eq!((node, extent), (cpu_node, cpu_extent));
    assert_eq!(index, case.shape.experts as i64 + 5, "the fault carries the raw fetched index");
}

#[test]
fn control_an_oracle_routed_to_other_experts_disagrees_by_a_gross_factor() {
    let shape = Shape { sequence: 60, selected: 4, rows: 192, k: 512, experts: 8 };
    let route = top_k_route(shape.sequence, shape.selected, shape.experts, 3);
    let matching = run(&Case {
        shape: Shape { ..shape },
        route: route.clone(),
        activation_per_selected: false,
        fuse_gate: false,
    });
    let rotated: Vec<f32> = route.iter().map(|expert| (expert + 1.0) % shape.experts as f32).collect();
    let misrouted = run(&Case { shape, route: rotated, activation_per_selected: false, fuse_gate: false });
    let gross = worst_relative_row_error(&misrouted.oracle, &matching.metal, 192);
    assert!(
        gross > RELATIVE_TOLERANCE,
        "a metal result compared against an oracle routed elsewhere must not pass: {gross:e}"
    );
}

const DECODE_AND_VERIFY_WIDTHS: [usize; 4] = [1, 2, 3, 4];

fn assert_matches_oracle_below_the_tile_minimum(case: &Case) -> Vec<String> {
    let outputs = run(case);
    let Shape { sequence, selected, rows, .. } = case.shape;
    assert_eq!(outputs.oracle.len(), sequence * selected * rows, "degenerate: oracle produced no output");
    assert_eq!(outputs.metal.len(), sequence * selected * rows, "degenerate: metal produced no output");
    assert!(
        !stacked_grouped_kernel_ran(&outputs.kernel_keys),
        "{sequence} tokens: a decode or verify width must not take the prefill tile kernel: {:?}",
        outputs.kernel_keys
    );
    let worst = worst_relative_row_error(&outputs.oracle, &outputs.metal, rows);
    assert!(
        worst <= RELATIVE_TOLERANCE,
        "{sequence} tokens, per_selected_activation={}: stacked result drifted from the f32 oracle by {worst:e}",
        case.activation_per_selected
    );
    outputs.kernel_keys
}

#[test]
fn a_stacked_projection_at_decode_and_verify_widths_matches_the_f32_oracle() {
    for sequence in DECODE_AND_VERIFY_WIDTHS {
        for activation_per_selected in [false, true] {
            let shape = Shape { sequence, selected: 8, rows: 192, k: 512, experts: 32 };
            let route = top_k_route(shape.sequence, shape.selected, shape.experts, 50 + sequence as u64);
            assert_matches_oracle_below_the_tile_minimum(&Case {
                shape,
                route,
                activation_per_selected,
                fuse_gate: false,
            });
        }
    }
}

#[test]
fn a_stacked_down_at_a_decode_step_takes_the_packed_matvec_not_the_dense_gather() {
    let shape = Shape { sequence: 1, selected: 8, rows: 192, k: 512, experts: 32 };
    let route = top_k_route(shape.sequence, shape.selected, shape.experts, 61);
    let per_selected = assert_matches_oracle_below_the_tile_minimum(&Case {
        shape: Shape { ..shape },
        route: route.clone(),
        activation_per_selected: true,
        fuse_gate: false,
    });
    let shared = assert_matches_oracle_below_the_tile_minimum(&Case {
        shape,
        route,
        activation_per_selected: false,
        fuse_gate: false,
    });
    let grouped_matvec_keys = |keys: &[String]| keys.iter().filter(|key| key.contains("_dg")).count();
    assert!(grouped_matvec_keys(&shared) > 0, "the shared-activation form runs the packed matvec: {shared:?}");
    assert_eq!(
        grouped_matvec_keys(&per_selected),
        grouped_matvec_keys(&shared),
        "the per-selected-activation form must take the same kernel family as the shared one: {per_selected:?} vs {shared:?}"
    );
}
