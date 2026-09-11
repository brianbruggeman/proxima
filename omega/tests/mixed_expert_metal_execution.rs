//! Device execution coverage for the HOBBIT expert-source bridge.
//!
//! This uses the same gathered expert product shape as the MoE lowering:
//! `sum_k weight[route[s], out, k] * activation[s, k]`.  Unlike the regular
//! gathered parity fixture, its source table contains an independently packed
//! Q2_K entry and Q4_K entry.  The execution entry point must therefore use
//! the source descriptor ABI rather than the plan's placeholder Q4_K block.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::expect_used)]

use std::collections::BTreeMap;

use proxima_gguf::quant::{q2_k, q4_k};
use proxima_tensor::cpu::{
    ExpertEntry, ExpertSource, QuantizedBlock,
    evaluate_quantized_named_exact_with_scratch_and_experts,
};
use proxima_tensor::map::{self, AxisIndex, AxisTerm};
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, Reduce, ReduceInit, ScalarOp, append,
};

const EXPERTS: usize = 3;
// The real qwen35moe gate/up expert projections are 512 x 2048. Keeping the
// production row and reduction extents here exercises the same Metal lowering
// decisions while two aliased expert entries keep the fixture small.
const ROWS: usize = 512;
const WIDTH: usize = 8 * q2_k::QK_K;
const SEQUENCE: usize = 2;

fn gathered_expert_program() -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::UInt8,
            shape: vec![
                Extent::Static(EXPERTS as u32),
                Extent::Static(ROWS as u32),
                Extent::Static(WIDTH as u32),
            ],
            name: Some("weight".into()),
        },
    );
    let route = append(
        &mut program,
        Op::Input {
            dtype: DType::Int32,
            shape: vec![Extent::Static(SEQUENCE as u32)],
            name: Some("route".into()),
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![
                Extent::Static(SEQUENCE as u32),
                Extent::Static(WIDTH as u32),
            ],
            name: Some("activation".into()),
        },
    );
    let gather = IndexMap::Computed {
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
    };
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, gather),
                (activation, IndexMap::Affine(map::projection(3, &[0, 2]))),
            ],
            name: Some("mixed_hobbit_product".into()),
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
            name: Some("mixed_hobbit_reduce".into()),
        }),
    );
    (program, sum)
}

fn append_gathered_reduce(
    program: &mut Vec<Op>,
    weight: NodeId,
    route: NodeId,
    activation: NodeId,
) -> NodeId {
    let gather = IndexMap::Computed {
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
    };
    let product = append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, gather),
                (activation, IndexMap::Affine(map::projection(3, &[0, 2]))),
            ],
            name: None,
        },
    );
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
            name: None,
        }),
    )
}

fn two_route_program() -> (Vec<Op>, NodeId, NodeId) {
    let mut program = Vec::new();
    let first_weight = append(
        &mut program,
        Op::Input {
            dtype: DType::UInt8,
            shape: vec![
                Extent::Static(EXPERTS as u32),
                Extent::Static(ROWS as u32),
                Extent::Static(WIDTH as u32),
            ],
            name: Some("first_weight".into()),
        },
    );
    let second_weight = append(
        &mut program,
        Op::Input {
            dtype: DType::UInt8,
            shape: vec![
                Extent::Static(EXPERTS as u32),
                Extent::Static(ROWS as u32),
                Extent::Static(WIDTH as u32),
            ],
            name: Some("second_weight".into()),
        },
    );
    let first_route = append(
        &mut program,
        Op::Input {
            dtype: DType::Int32,
            shape: vec![Extent::Static(1)],
            name: Some("first_route".into()),
        },
    );
    let second_route = append(
        &mut program,
        Op::Input {
            dtype: DType::Int32,
            shape: vec![Extent::Static(1)],
            name: Some("second_route".into()),
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1), Extent::Static(WIDTH as u32)],
            name: Some("activation".into()),
        },
    );
    let first = append_gathered_reduce(&mut program, first_weight, first_route, activation);
    let second = append_gathered_reduce(&mut program, second_weight, second_route, activation);
    (program, first, second)
}

fn expert_weights(seed: usize) -> Vec<f32> {
    (0..ROWS * WIDTH)
        .map(|index| ((index * (seed * 11 + 3) % 127) as f32 - 63.0) * 0.015)
        .collect()
}

fn pack_q2k(values: &[f32]) -> Vec<u8> {
    let mut packed = vec![0_u8; ROWS * (WIDTH / q2_k::QK_K) * q2_k::BLOCK_BYTES];
    q2_k::quantize(values, &mut packed).expect("the Q2_K fixture has whole super-block rows");
    packed
}

fn pack_q4k(values: &[f32]) -> Vec<u8> {
    let mut packed = vec![0_u8; ROWS * (WIDTH / q4_k::QK_K) * q4_k::BLOCK_BYTES];
    q4_k::quantize(values, &mut packed).expect("the Q4_K fixture has whole super-block rows");
    packed
}

fn activation() -> Vec<f32> {
    (0..SEQUENCE * WIDTH)
        .map(|index| ((index * 17 % 97) as f32 - 48.0) * 0.02)
        .collect()
}

#[test]
fn mixed_q2k_q4k_expert_source_executes_on_metal() {
    let (program, output) = gathered_expert_program();
    let low = pack_q2k(&expert_weights(1));
    let high = pack_q4k(&expert_weights(2));
    let placeholder = [high.as_slice(), high.as_slice(), high.as_slice()].concat();
    let routes = [2.0_f32, 0.0_f32];
    let activations = activation();
    let named = [
        ("weight", QuantizedBlock::Q4K(&placeholder)),
        ("route", QuantizedBlock::Float32(&routes)),
        ("activation", QuantizedBlock::Float32(&activations)),
    ];
    let plan = omega::plan_named(&program, &[], &named, &[output], NumericPolicy::default())
        .expect("the Q4_K placeholder supplies the plan's static shape");
    let entries = [
        ExpertEntry {
            block: QuantizedBlock::Q2K(&low),
            out_dim: ROWS as u32,
            in_dim: WIDTH as u32,
            epoch: 3,
        },
        ExpertEntry {
            block: QuantizedBlock::Q4K(&high),
            out_dim: ROWS as u32,
            in_dim: WIDTH as u32,
            epoch: 4,
        },
        ExpertEntry {
            block: QuantizedBlock::Q4K(&high),
            out_dim: ROWS as u32,
            in_dim: WIDTH as u32,
            epoch: 5,
        },
    ];
    let selected = [0_u32, 2_u32];
    let source = ExpertSource::with_selected_expert_ids(&entries, &selected);
    let sources = BTreeMap::from([(NodeId(0), source)]);

    let mut cpu_scratch = Vec::new();
    let mut validated_weight_nodes = None;
    let expected = evaluate_quantized_named_exact_with_scratch_and_experts(
        &program,
        &[],
        &named,
        &[output],
        &mut cpu_scratch,
        &mut validated_weight_nodes,
        Some(&sources),
    )
    .expect("the CPU evaluator supplies the mixed-codec reference");
    let evaluated = omega::metal::execute_plan_named_with_expert_sources(&plan, &named, &sources)
        .expect("the mixed Q2_K/Q4_K source table runs through the Metal descriptor ABI");
    assert_eq!(evaluated.root().len(), SEQUENCE * ROWS);
    assert!(
        evaluated.root().iter().all(|value| value.is_finite()),
        "mixed expert output must not contain NaN or infinity: {:?}",
        evaluated.root()
    );
    for (index, (actual, expected)) in evaluated
        .root()
        .iter()
        .zip(expected.root().iter())
        .enumerate()
    {
        assert!(
            (actual - expected).abs() <= 1.0e-4,
            "mixed expert output {index} differs: metal={actual} cpu={expected}"
        );
    }
}

#[cfg(feature = "metal-output-placement")]
#[test]
fn mixed_expert_source_and_recurrent_state_placement_share_one_execution() {
    const STATE_ELEMENTS: usize = 8;

    let (mut program, expert_output) = gathered_expert_program();
    let state_input = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(STATE_ELEMENTS as u32)],
            name: Some("state".into()),
        },
    );
    let state_output = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: vec![(state_input, IndexMap::Affine(map::projection(1, &[0])))],
            name: Some("state_out".into()),
        },
    );

    let low = pack_q2k(&expert_weights(1));
    let high = pack_q4k(&expert_weights(2));
    let placeholder = [high.as_slice(), high.as_slice(), high.as_slice()].concat();
    let routes = [2.0_f32, 0.0_f32];
    let activations = activation();
    let state_placeholder = [0.0_f32; STATE_ELEMENTS];
    let named = [
        ("weight", QuantizedBlock::Q4K(&placeholder)),
        ("route", QuantizedBlock::Float32(&routes)),
        ("activation", QuantizedBlock::Float32(&activations)),
        ("state", QuantizedBlock::Float32(&state_placeholder)),
    ];
    let plan = omega::plan_named(
        &program,
        &[],
        &named,
        &[expert_output, state_output],
        NumericPolicy::default(),
    )
    .expect("the combined expert and recurrent-state program plans");

    let entries = [
        ExpertEntry {
            block: QuantizedBlock::Q2K(&low),
            out_dim: ROWS as u32,
            in_dim: WIDTH as u32,
            epoch: 3,
        },
        ExpertEntry {
            block: QuantizedBlock::Q4K(&high),
            out_dim: ROWS as u32,
            in_dim: WIDTH as u32,
            epoch: 4,
        },
        ExpertEntry {
            block: QuantizedBlock::Q4K(&high),
            out_dim: ROWS as u32,
            in_dim: WIDTH as u32,
            epoch: 5,
        },
    ];
    let selected = [0_u32, 2_u32];
    let source = ExpertSource::with_selected_expert_ids(&entries, &selected);
    let sources = BTreeMap::from([(NodeId(0), source)]);

    let input_buffer = omega::allocate_placed_buffer(STATE_ELEMENTS * size_of::<f32>())
        .expect("allocates the recurrent input buffer");
    let output_buffer = omega::allocate_placed_buffer(STATE_ELEMENTS * size_of::<f32>())
        .expect("allocates the recurrent output buffer");
    let state: Vec<f32> = (0..STATE_ELEMENTS)
        .map(|index| index as f32 + 0.25)
        .collect();
    let (seed_program, seed_output) = {
        let mut seed_program = Vec::new();
        let seed_input = append(
            &mut seed_program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(STATE_ELEMENTS as u32)],
                name: None,
            },
        );
        let seed_output = append(
            &mut seed_program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: vec![(seed_input, IndexMap::Affine(map::projection(1, &[0])))],
                name: None,
            },
        );
        (seed_program, seed_output)
    };
    let seed_plan = omega::plan(
        &seed_program,
        &[],
        &[QuantizedBlock::Float32(&state)],
        &[seed_output],
        NumericPolicy::default(),
    )
    .expect("the recurrent-state seed plans");
    omega::execute_plan_with_placements(
        &seed_plan,
        &[QuantizedBlock::Float32(&state)],
        &[],
        &[(seed_output, &input_buffer, 0)],
        &mut Vec::new(),
    )
    .expect("seeds the recurrent input buffer");

    let evaluated = omega::execute_plan_named_with_placements_and_expert_sources(
        &plan,
        &named,
        &[(state_input, &input_buffer, 0)],
        &[(state_output, &output_buffer, 0)],
        &sources,
    )
    .expect("one Metal execution accepts both HOBBIT sources and placed state");

    assert!(
        evaluated.get(expert_output).is_some(),
        "the mixed expert reduction must execute while state is placed"
    );
    assert!(
        evaluated.get(state_output).is_none(),
        "a placed state output must not also be copied back through Evaluated"
    );
    assert_eq!(
        omega::read_placed_buffer_f32(&output_buffer, 0, STATE_ELEMENTS),
        state,
        "the recurrent state must survive the mixed-codec expert execution"
    );
}

#[test]
fn mixed_kernel_cache_keeps_each_route_index_binding() {
    let (program, first, second) = two_route_program();
    let low = pack_q2k(&expert_weights(1));
    let high = pack_q4k(&expert_weights(2));
    let placeholder = [high.as_slice(), high.as_slice(), high.as_slice()].concat();
    let first_route = [0.0_f32];
    let second_route = [1.0_f32];
    let activations = &activation()[..WIDTH];
    let named = [
        ("first_weight", QuantizedBlock::Q4K(&placeholder)),
        ("second_weight", QuantizedBlock::Q4K(&placeholder)),
        ("first_route", QuantizedBlock::Float32(&first_route)),
        ("second_route", QuantizedBlock::Float32(&second_route)),
        ("activation", QuantizedBlock::Float32(activations)),
    ];
    let plan = omega::plan_named(
        &program,
        &[],
        &named,
        &[first, second],
        NumericPolicy::default(),
    )
    .expect("both structurally identical gathered reductions plan together");
    let entries = [
        ExpertEntry {
            block: QuantizedBlock::Q2K(&low),
            out_dim: ROWS as u32,
            in_dim: WIDTH as u32,
            epoch: 3,
        },
        ExpertEntry {
            block: QuantizedBlock::Q4K(&high),
            out_dim: ROWS as u32,
            in_dim: WIDTH as u32,
            epoch: 4,
        },
        ExpertEntry {
            block: QuantizedBlock::Q4K(&high),
            out_dim: ROWS as u32,
            in_dim: WIDTH as u32,
            epoch: 5,
        },
    ];
    let selected = [0_u32, 1_u32];
    let source = ExpertSource::with_selected_expert_ids(&entries, &selected);
    let sources = BTreeMap::from([(NodeId(0), source), (NodeId(1), source)]);
    let mut cpu_scratch = Vec::new();
    let mut validated_weight_nodes = None;
    let expected = evaluate_quantized_named_exact_with_scratch_and_experts(
        &program,
        &[],
        &named,
        &[first, second],
        &mut cpu_scratch,
        &mut validated_weight_nodes,
        Some(&sources),
    )
    .expect("the two route indices have an exact CPU reference");
    let evaluated = omega::metal::execute_plan_named_with_expert_sources(&plan, &named, &sources)
        .expect("both route-index bindings execute through Metal");

    for node in [first, second] {
        let (actual, _) = evaluated
            .get(node)
            .expect("Metal retained the requested route output");
        let (expected, _) = expected
            .get(node)
            .expect("CPU retained the requested route output");
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1.0e-4,
                "route output {node:?} element {index} differs: metal={actual} cpu={expected}"
            );
        }
    }
}

#[test]
fn mixed_expert_source_rejects_unsupported_codec_before_device_execution() {
    let (program, output) = gathered_expert_program();
    let placeholder_row = pack_q4k(&expert_weights(9));
    let placeholder = [
        placeholder_row.as_slice(),
        placeholder_row.as_slice(),
        placeholder_row.as_slice(),
    ]
    .concat();
    let routes = [0.0_f32, 0.0_f32];
    let activations = activation();
    let named = [
        ("weight", QuantizedBlock::Q4K(&placeholder)),
        ("route", QuantizedBlock::Float32(&routes)),
        ("activation", QuantizedBlock::Float32(&activations)),
    ];
    let plan = omega::plan_named(&program, &[], &named, &[output], NumericPolicy::default())
        .expect("the placeholder has a valid gathered Q4_K shape");
    let unsupported = [0_u8; 1];
    let entries = [ExpertEntry {
        block: QuantizedBlock::Q3K(&unsupported),
        out_dim: ROWS as u32,
        in_dim: WIDTH as u32,
        epoch: 0,
    }];
    let sources = BTreeMap::from([(NodeId(0), ExpertSource::new(&entries))]);

    let error = omega::metal::execute_plan_named_with_expert_sources(&plan, &named, &sources)
        .expect_err("a Q3_K expert source must have a typed mixed-codec rejection");
    assert!(matches!(
        error,
        omega::MetalError::ExpertSourceUnsupported {
            node: NodeId(0),
            reason: "mixed expert lowering only has Q2_K, Q4_K, and Q6_K decoders",
        }
    ));
}
