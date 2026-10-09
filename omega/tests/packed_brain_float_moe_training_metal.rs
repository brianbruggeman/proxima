#![cfg(all(feature = "cpu", feature = "metal", target_os = "macos"))]

extern crate moe_test_json as pilot_json;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::fs;

use moe_test_json as serde_json;
use omega::backend::{Engine, GpuDriver, Plan, execute_plan_named, plan_named};
use omega::{Codec, PackedOperands, emit};
use proxima_autograd::adjoint::{Differentiated, GatheredContribution, differentiate};
use proxima_autograd::optimizer::{AdamConfig, AdamOperands, adam_step, step_input};
use proxima_autograd::sparse::dedupe_and_sum_rows;
use proxima_tensor::cpu::{
    ExpertEntry, ExpertSource, QuantizedBlock, evaluate_quantized_named_with_scratch_and_experts,
};
use proxima_tensor::map::{AxisIndex, AxisTerm, IndexPattern};
use proxima_tensor::op::{self, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp};
use proxima_tensor::spec::gathered_expert_product;
use proxima_tensor::{DType, Extent, IndexMap, NumericPolicy, map};

#[path = "../../proxima-autograd/examples/low_precision_moe_training_pilot.rs"]
pub mod pilot;

const MASTER: [f32; 12] = [1.1, 0.0, 0.0, 1.1, 2.2, 0.0, -1.1, 0.0, 3.3, 4.0, 5.0, 6.0];
const ROUTES: [i32; 3] = [1, 1, 0];
const ACTIVATIONS: [f32; 6] = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
const TARGETS: [f32; 6] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
const EXPECTED_PREDICTIONS: [f32; 6] = [2.0, -1.0, 2.0, -1.0, 0.0, 1.0];
const EXPECTED_COMPACT_GRADIENTS: [f32; 12] =
    [2.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
const EXPECTED_COALESCED_GRADIENTS: [f32; 12] =
    [0.0, 0.0, 0.0, 1.0, 3.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0];
const ABSOLUTE_TOLERANCE: f32 = 1e-6;

struct Fixture {
    program: Vec<Op>,
    expert_stack: NodeId,
    product: NodeId,
    predictions: NodeId,
    loss: NodeId,
}

struct RunValues {
    predictions: Vec<f32>,
    loss: f32,
    compact_gradients: Vec<f32>,
}

struct AdamValues {
    masters: Vec<f32>,
    first_moments: Vec<f32>,
    second_moments: Vec<f32>,
}

struct Observation {
    codec: Codec,
    packed_bytes: Vec<u8>,
    metal: RunValues,
    cpu: RunValues,
    coalesced_gradients: Vec<f32>,
    cpu_coalesced_gradients: Vec<f32>,
    metal_update: AdamValues,
    cpu_update: AdamValues,
    reference: ReferenceValues,
}

struct ReferenceValues {
    predictions: Vec<f32>,
    loss: f32,
    compact_gradients: Vec<f32>,
    coalesced_gradients: Vec<f32>,
    update: AdamValues,
}

struct LoopStep {
    codec: Codec,
    step: u32,
    master_input: Vec<f32>,
    moment_one_input: Vec<f32>,
    moment_two_input: Vec<f32>,
    packed_bytes: Vec<u8>,
    metal: RunValues,
    cpu: RunValues,
    metal_coalesced: Vec<f32>,
    cpu_coalesced: Vec<f32>,
    metal_update: AdamValues,
    cpu_update: AdamValues,
    scalar: ReferenceValues,
}

struct PilotBatchFixture {
    program: Vec<Op>,
    expert_stack: NodeId,
    packed_gather: NodeId,
    logits: NodeId,
    token_losses: NodeId,
    mean_loss: NodeId,
}

struct PilotBatchValues {
    logits: Vec<f32>,
    token_losses: Vec<f32>,
    mean_loss: f32,
    compact_gradients: Vec<f32>,
}

const LOOP_ADAM: AdamConfig = AdamConfig {
    learning_rate: 0.5,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
};

fn build_fixture() -> Fixture {
    let mut program = Vec::new();
    let expert_stack = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(3), Extent::Static(2), Extent::Static(2)],
            name: Some("expert_stack".into()),
        },
    );
    let route = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Int32,
            shape: vec![Extent::Static(3)],
            name: Some("route".into()),
        },
    );
    let activation = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(3), Extent::Static(2)],
            name: Some("activation".into()),
        },
    );
    let target = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(3), Extent::Static(2)],
            name: Some("target".into()),
        },
    );
    let product = gathered_expert_product(&mut program, expert_stack, route, activation);
    let predictions = op::append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 2])),
            keep: Keep::Reduce,
            name: Some("routed_prediction".into()),
        }),
    );
    let matrix_map = IndexMap::Affine(map::projection(2, &[0, 1]));
    let difference = op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Subtract,
            operands: vec![
                (predictions, matrix_map.clone()),
                (target, matrix_map.clone()),
            ],
            name: None,
        },
    );
    let squared = op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (difference, matrix_map.clone()),
                (difference, matrix_map.clone()),
            ],
            name: None,
        },
    );
    let half = op::append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: Vec::new(),
            value: 0.5,
        },
    );
    let scaled = op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (squared, matrix_map),
                (half, IndexMap::Affine(map::projection(2, &[]))),
            ],
            name: None,
        },
    );
    let loss = op::append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: scaled,
            in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            out_map: IndexMap::Affine(map::projection(2, &[])),
            keep: Keep::Reduce,
            name: Some("batch_loss".into()),
        }),
    );
    Fixture {
        program,
        expert_stack,
        product,
        predictions,
        loss,
    }
}

fn differentiate_fixture(fixture: &Fixture) -> (Differentiated, GatheredContribution) {
    let differentiated = differentiate(&fixture.program, fixture.loss)
        .expect("the fixed routed loss differentiates");
    let gathered = differentiated
        .gathered_gradients_of(fixture.expert_stack)
        .next()
        .expect("the expert stack produces one compact gathered contribution");
    (differentiated, gathered)
}

fn encode_master(codec: Codec) -> Vec<u8> {
    encode_masters(codec, &MASTER)
}

fn encode_masters(codec: Codec, masters: &[f32]) -> Vec<u8> {
    match codec {
        Codec::Bf8E5M2 => masters
            .iter()
            .map(|value| proxima_gguf::quant::bf8_e5m2::encode(*value))
            .collect(),
        Codec::Bf4E2M1 => masters
            .chunks_exact(2)
            .map(|pair| {
                proxima_gguf::quant::bf4_e2m1::pack_pair(pair[0], pair[1])
                    .expect("the finite master pair encodes as BF4")
            })
            .collect(),
        _ => panic!("fixture only uses BF8 E5M2 and BF4 E2M1"),
    }
}

fn input_blocks(bytes: &[u8]) -> [(&'static str, QuantizedBlock<'_>); 4] {
    [
        (
            "expert_stack",
            QuantizedBlock::Packed {
                codec: Codec::Bf8E5M2,
                bytes,
            },
        ),
        ("route", QuantizedBlock::Int32(&ROUTES)),
        ("activation", QuantizedBlock::Float32(&ACTIVATIONS)),
        ("target", QuantizedBlock::Float32(&TARGETS)),
    ]
}

fn codec_blocks(codec: Codec, bytes: &[u8]) -> [(&'static str, QuantizedBlock<'_>); 4] {
    let mut blocks = input_blocks(bytes);
    blocks[0].1 = QuantizedBlock::Packed { codec, bytes };
    blocks
}

fn expert_entries(codec: Codec, bytes: &[u8]) -> [ExpertEntry<'_>; 3] {
    let bytes_per_expert = match codec {
        Codec::Bf8E5M2 => 4,
        Codec::Bf4E2M1 => 2,
        _ => panic!("fixture only uses BF8 E5M2 and BF4 E2M1"),
    };
    std::array::from_fn(|expert_index| {
        let start = expert_index * bytes_per_expert;
        ExpertEntry {
            block: QuantizedBlock::Packed {
                codec,
                bytes: &bytes[start..start + bytes_per_expert],
            },
            out_dim: 2,
            in_dim: 2,
            epoch: 1,
        }
    })
}

fn assert_msl_reads_packed_bytes(fixture: &Fixture, differentiated: &Differentiated, codec: Codec) {
    let shapes = proxima_tensor::infer(&differentiated.program, &[])
        .expect("differentiated fixture shapes infer");
    let mut bound = proxima_tensor::bind(
        &differentiated.program,
        &shapes,
        &[fixture.product],
        NumericPolicy::default(),
    )
    .expect("the packed product binds for MSL source inspection");
    proxima_tensor::correct_packed_matmul_layouts(
        &mut bound,
        &BTreeSet::from([fixture.expert_stack]),
    );
    let resolved = bound
        .iter()
        .find(|bound_op| bound_op.node == fixture.product)
        .expect("the gathered product has a bound MSL operation");
    let packed_operands: PackedOperands = [(fixture.expert_stack, codec)].into_iter().collect();
    let kernel = emit(resolved, &packed_operands, NumericPolicy::default())
        .expect("the packed gathered product emits MSL");
    let decoder = match codec {
        Codec::Bf8E5M2 => "bf8_e5m2_element(in0 +",
        Codec::Bf4E2M1 => "bf4_e2m1_element(in0 +",
        _ => panic!("fixture only uses BF8 E5M2 and BF4 E2M1"),
    };
    assert!(
        kernel.source.contains("device const uchar* in0"),
        "Metal must bind the original packed byte span as uchar input:\n{}",
        kernel.source
    );
    assert!(
        kernel.source.contains(decoder),
        "Metal must decode {codec:?} directly from the packed input:\n{}",
        kernel.source
    );
    assert!(
        kernel
            .bindings
            .contains(&omega::Binding::Input(fixture.expert_stack)),
        "the gathered kernel must bind the original packed expert stack"
    );
    assert!(
        !kernel.source.contains("expanded_expert") && !kernel.source.contains("expert_stack_f32"),
        "the emitted Metal kernel must not name an expanded FP32 expert table"
    );
}

fn read_run_values(
    values: &proxima_tensor::cpu::Evaluated,
    fixture: &Fixture,
    gathered: GatheredContribution,
) -> RunValues {
    RunValues {
        predictions: values
            .get(fixture.predictions)
            .expect("prediction output exists")
            .0
            .to_vec(),
        loss: values.get(fixture.loss).expect("loss output exists").0[0],
        compact_gradients: values
            .get(gathered.values)
            .expect("compact gradient output exists")
            .0
            .to_vec(),
    }
}

fn run_cpu(
    fixture: &Fixture,
    differentiated: &Differentiated,
    gathered: GatheredContribution,
    codec: Codec,
    bytes: &[u8],
) -> RunValues {
    let named = codec_blocks(codec, bytes);
    let entries = expert_entries(codec, bytes);
    let mut expert_sources = BTreeMap::new();
    expert_sources.insert(fixture.expert_stack, ExpertSource::new(&entries));
    let outputs = [fixture.loss, fixture.predictions, gathered.values];
    let evaluated = evaluate_quantized_named_with_scratch_and_experts(
        &differentiated.program,
        &[],
        &named,
        &outputs,
        &mut Vec::new(),
        &mut None,
        Some(&expert_sources),
    )
    .expect("CPU evaluates the differentiated packed program through ExpertSource");
    read_run_values(&evaluated, fixture, gathered)
}

fn run_metal(
    fixture: &Fixture,
    differentiated: &Differentiated,
    gathered: GatheredContribution,
    codec: Codec,
    bytes: &[u8],
) -> RunValues {
    assert_msl_reads_packed_bytes(fixture, differentiated, codec);
    let named = codec_blocks(codec, bytes);
    let outputs = [fixture.loss, fixture.predictions, gathered.values];
    let mut plan = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Metal),
        &differentiated.program,
        &[],
        &named,
        &outputs,
        NumericPolicy::default(),
    )
    .expect("Metal plans the differentiated packed graph");
    let evaluated = execute_plan_named(&mut plan, &named)
        .expect("Metal executes the differentiated packed graph on a device");
    read_run_values(&evaluated, fixture, gathered)
}

fn assert_close(actual: &[f32], expected: &[f32], payload: &str) {
    assert_eq!(actual.len(), expected.len(), "{payload} length");
    for (index, (actual_value, expected_value)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual_value.is_finite() && (actual_value - expected_value).abs() <= ABSOLUTE_TOLERANCE,
            "{payload}[{index}] differs: actual={actual_value:?}, expected={expected_value:?}, tolerance={ABSOLUTE_TOLERANCE}"
        );
    }
}

fn run_adam_row(parameters: &[f32], gradients: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    run_adam_state_row(
        parameters,
        gradients,
        &[0.0; 4],
        &[0.0; 4],
        1,
        AdamConfig::default(),
    )
}

fn run_adam_state_row(
    parameters: &[f32],
    gradients: &[f32],
    first_moments: &[f32],
    second_moments: &[f32],
    global_step: u32,
    config: AdamConfig,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut program = Vec::new();
    let parameter = append_vector_input(&mut program, "parameter");
    let gradient = append_vector_input(&mut program, "gradient");
    let moment_one = append_vector_input(&mut program, "moment_one");
    let moment_two = append_vector_input(&mut program, "moment_two");
    let step = step_input(&mut program, "step");
    let (updated_parameter, updated_moment_one, updated_moment_two) = adam_step(
        &mut program,
        &config,
        1,
        AdamOperands {
            param: parameter,
            grad: gradient,
            m: moment_one,
            v: moment_two,
        },
        step,
    );
    let evaluated = proxima_tensor::cpu::evaluate_named(
        &program,
        &[],
        &[
            ("parameter", parameters),
            ("gradient", gradients),
            ("moment_one", first_moments),
            ("moment_two", second_moments),
            ("step", &[global_step as f32]),
        ],
        &[updated_parameter, updated_moment_one, updated_moment_two],
    )
    .expect("selected expert Adam row evaluates");
    (
        evaluated
            .get(updated_parameter)
            .expect("updated parameters")
            .0
            .to_vec(),
        evaluated
            .get(updated_moment_one)
            .expect("updated first moments")
            .0
            .to_vec(),
        evaluated
            .get(updated_moment_two)
            .expect("updated second moments")
            .0
            .to_vec(),
    )
}

fn append_vector_input(program: &mut Vec<Op>, name: &str) -> NodeId {
    op::append(
        program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: Some(name.into()),
        },
    )
}

fn apply_sparse_update(gradients: &[f32]) -> AdamValues {
    let mut masters = MASTER;
    let mut first_moments = [0.0; 12];
    let mut second_moments = [0.0; 12];
    let (selected_experts, selected_gradients) =
        dedupe_and_sum_rows(&ROUTES.map(|route| route as f32), gradients, 4)
            .expect("three compact rows coalesce by routed expert");
    for (row_index, expert_id) in selected_experts.iter().copied().enumerate() {
        let expert_start = expert_id as usize * 4;
        let gradient_start = row_index * 4;
        let mut canonical_gradient = [0.0; 4];
        for output_index in 0..2 {
            for input_index in 0..2 {
                canonical_gradient[output_index * 2 + input_index] =
                    selected_gradients[gradient_start + input_index * 2 + output_index];
            }
        }
        let (updated_master, updated_first, updated_second) = run_adam_row(
            &masters[expert_start..expert_start + 4],
            &canonical_gradient,
        );
        masters[expert_start..expert_start + 4].copy_from_slice(&updated_master);
        first_moments[expert_start..expert_start + 4].copy_from_slice(&updated_first);
        second_moments[expert_start..expert_start + 4].copy_from_slice(&updated_second);
    }
    AdamValues {
        masters: masters.to_vec(),
        first_moments: first_moments.to_vec(),
        second_moments: second_moments.to_vec(),
    }
}

fn independent_reference() -> ReferenceValues {
    let decoded = [
        1.0f32, 0.0, 0.0, 1.0, 2.0, 0.0, -1.0, 0.0, 3.0, 4.0, 5.0, 6.0,
    ];
    let mut predictions = vec![0.0; 6];
    let mut compact_gradients = vec![0.0; 12];
    let mut coalesced_gradients = vec![0.0; 12];
    let mut loss = 0.0f32;
    for token_index in 0..3 {
        let expert_index = ROUTES[token_index] as usize;
        let activation_start = token_index * 2;
        let mut prediction = [0.0; 2];
        for output_index in 0..2 {
            for input_index in 0..2 {
                prediction[output_index] += decoded
                    [expert_index * 4 + output_index * 2 + input_index]
                    * ACTIVATIONS[activation_start + input_index];
            }
        }
        predictions[activation_start..activation_start + 2].copy_from_slice(&prediction);
        let mut output_gradient = [0.0; 2];
        for output_index in 0..2 {
            output_gradient[output_index] =
                prediction[output_index] - TARGETS[activation_start + output_index];
            loss += 0.5 * output_gradient[output_index] * output_gradient[output_index];
        }
        let compact_start = token_index * 4;
        for input_index in 0..2 {
            for output_index in 0..2 {
                let value =
                    ACTIVATIONS[activation_start + input_index] * output_gradient[output_index];
                compact_gradients[compact_start + input_index * 2 + output_index] = value;
                coalesced_gradients[expert_index * 4 + output_index * 2 + input_index] += value;
            }
        }
    }
    let update = independent_adam(&coalesced_gradients);
    ReferenceValues {
        predictions,
        loss,
        compact_gradients,
        coalesced_gradients,
        update,
    }
}

fn independent_adam(gradients: &[f32]) -> AdamValues {
    let mut masters = MASTER;
    let mut first_moments = [0.0; 12];
    let mut second_moments = [0.0; 12];
    let learning_rate = 0.001f32;
    let beta_one = 0.9f32;
    let beta_two = 0.999f32;
    let epsilon = 1e-8f32;
    for expert_index in 0..2 {
        for element_index in 0..4 {
            let index = expert_index * 4 + element_index;
            let gradient = gradients[index];
            first_moments[index] = (1.0 - beta_one) * gradient;
            second_moments[index] = (1.0 - beta_two) * gradient * gradient;
            let corrected_first = first_moments[index] / (1.0 - beta_one);
            let corrected_second = second_moments[index] / (1.0 - beta_two);
            masters[index] -= learning_rate * corrected_first / (corrected_second.sqrt() + epsilon);
        }
    }
    AdamValues {
        masters: masters.to_vec(),
        first_moments: first_moments.to_vec(),
        second_moments: second_moments.to_vec(),
    }
}

fn coalesce_compact(compact: &[f32]) -> Vec<f32> {
    let mut coalesced = vec![0.0; 12];
    for token_index in 0..3 {
        let expert_index = ROUTES[token_index] as usize;
        for input_index in 0..2 {
            for output_index in 0..2 {
                coalesced[expert_index * 4 + output_index * 2 + input_index] +=
                    compact[token_index * 4 + input_index * 2 + output_index];
            }
        }
    }
    coalesced
}

fn scalar_loop_step(
    codec: Codec,
    bytes: &[u8],
    masters: &[f32],
    first_moments: &[f32],
    second_moments: &[f32],
    step: u32,
) -> ReferenceValues {
    let decoded: Vec<f32> = match codec {
        Codec::Bf8E5M2 => bytes
            .iter()
            .map(|value| proxima_gguf::quant::bf8_e5m2::decode(*value))
            .collect(),
        Codec::Bf4E2M1 => bytes
            .iter()
            .flat_map(|value| proxima_gguf::quant::bf4_e2m1::unpack_pair(*value))
            .collect(),
        _ => panic!("fixture only uses BF8 E5M2 and BF4 E2M1"),
    };
    let mut predictions = vec![0.0; 6];
    let mut compact_gradients = vec![0.0; 12];
    let mut loss = 0.0f32;
    for token_index in 0..3 {
        let expert_index = ROUTES[token_index] as usize;
        let token_start = token_index * 2;
        for output_index in 0..2 {
            for input_index in 0..2 {
                predictions[token_start + output_index] += decoded
                    [expert_index * 4 + output_index * 2 + input_index]
                    * ACTIVATIONS[token_start + input_index];
            }
            let residual =
                predictions[token_start + output_index] - TARGETS[token_start + output_index];
            loss += 0.5 * residual * residual;
            for input_index in 0..2 {
                compact_gradients[token_index * 4 + input_index * 2 + output_index] =
                    ACTIVATIONS[token_start + input_index] * residual;
            }
        }
    }
    let coalesced_gradients = coalesce_compact(&compact_gradients);
    let mut updated_masters = masters.to_vec();
    let mut updated_first = first_moments.to_vec();
    let mut updated_second = second_moments.to_vec();
    let correction_one = 1.0 - LOOP_ADAM.beta1.powi(step as i32);
    let correction_two = 1.0 - LOOP_ADAM.beta2.powi(step as i32);
    for expert_index in [0usize, 1] {
        for element_index in 0..4 {
            let index = expert_index * 4 + element_index;
            let gradient = coalesced_gradients[index];
            updated_first[index] =
                LOOP_ADAM.beta1 * first_moments[index] + (1.0 - LOOP_ADAM.beta1) * gradient;
            updated_second[index] = LOOP_ADAM.beta2 * second_moments[index]
                + (1.0 - LOOP_ADAM.beta2) * gradient * gradient;
            let corrected_first = updated_first[index] / correction_one;
            let corrected_second = updated_second[index] / correction_two;
            updated_masters[index] -= LOOP_ADAM.learning_rate * corrected_first
                / (corrected_second.sqrt() + LOOP_ADAM.epsilon);
        }
    }
    ReferenceValues {
        predictions,
        loss,
        compact_gradients,
        coalesced_gradients,
        update: AdamValues {
            masters: updated_masters,
            first_moments: updated_first,
            second_moments: updated_second,
        },
    }
}

fn apply_loop_update(
    masters: &[f32],
    first_moments: &[f32],
    second_moments: &[f32],
    gradients: &[f32],
    step: u32,
) -> AdamValues {
    let (selected_experts, selected_gradients) =
        dedupe_and_sum_rows(&ROUTES.map(|route| route as f32), gradients, 4)
            .expect("loop compact rows coalesce by routed expert");
    let mut updated = AdamValues {
        masters: masters.to_vec(),
        first_moments: first_moments.to_vec(),
        second_moments: second_moments.to_vec(),
    };
    for (row_index, expert_id) in selected_experts.iter().copied().enumerate() {
        let expert_start = expert_id as usize * 4;
        let gradient_start = row_index * 4;
        let mut canonical_gradient = [0.0; 4];
        for output_index in 0..2 {
            for input_index in 0..2 {
                canonical_gradient[output_index * 2 + input_index] =
                    selected_gradients[gradient_start + input_index * 2 + output_index];
            }
        }
        let (new_master, new_first, new_second) = run_adam_state_row(
            &masters[expert_start..expert_start + 4],
            &canonical_gradient,
            &first_moments[expert_start..expert_start + 4],
            &second_moments[expert_start..expert_start + 4],
            step,
            LOOP_ADAM,
        );
        updated.masters[expert_start..expert_start + 4].copy_from_slice(&new_master);
        updated.first_moments[expert_start..expert_start + 4].copy_from_slice(&new_first);
        updated.second_moments[expert_start..expert_start + 4].copy_from_slice(&new_second);
    }
    updated
}

fn observe(codec: Codec) -> Observation {
    let fixture = build_fixture();
    let (differentiated, gathered) = differentiate_fixture(&fixture);
    let packed_bytes = encode_master(codec);
    let cpu = run_cpu(&fixture, &differentiated, gathered, codec, &packed_bytes);
    let metal = run_metal(&fixture, &differentiated, gathered, codec, &packed_bytes);
    let (selected_experts, selected_rows) = dedupe_and_sum_rows(
        &ROUTES.map(|route| route as f32),
        &metal.compact_gradients,
        4,
    )
    .expect("Metal compact rows coalesce by routed expert");
    let mut coalesced_gradients = vec![0.0; 12];
    for (row_index, expert_index) in selected_experts.iter().copied().enumerate() {
        let row = &selected_rows[row_index * 4..row_index * 4 + 4];
        for output_index in 0..2 {
            for input_index in 0..2 {
                coalesced_gradients[expert_index as usize * 4 + output_index * 2 + input_index] =
                    row[input_index * 2 + output_index];
            }
        }
    }
    let cpu_update = apply_sparse_update(&cpu.compact_gradients);
    let metal_update = apply_sparse_update(&metal.compact_gradients);
    let cpu_coalesced_gradients = coalesce_compact(&cpu.compact_gradients);
    Observation {
        codec,
        packed_bytes,
        metal,
        cpu,
        coalesced_gradients,
        cpu_coalesced_gradients,
        metal_update,
        cpu_update,
        reference: independent_reference(),
    }
}

fn observe_loop(codec: Codec) -> Vec<LoopStep> {
    let fixture = build_fixture();
    let (differentiated, gathered) = differentiate_fixture(&fixture);
    let mut masters = MASTER.to_vec();
    let mut first_moments = vec![0.0; 12];
    let mut second_moments = vec![0.0; 12];
    let mut records = Vec::new();
    for step in 1..=2 {
        let packed_bytes = encode_masters(codec, &masters);
        let master_input = masters.clone();
        let moment_one_input = first_moments.clone();
        let moment_two_input = second_moments.clone();
        let cpu = run_cpu(&fixture, &differentiated, gathered, codec, &packed_bytes);
        eprintln!("loop Metal dispatch codec={codec:?} step={step}");
        let metal = run_metal(&fixture, &differentiated, gathered, codec, &packed_bytes);
        let cpu_coalesced = coalesce_compact(&cpu.compact_gradients);
        let metal_coalesced = coalesce_compact(&metal.compact_gradients);
        let cpu_update = apply_loop_update(
            &masters,
            &first_moments,
            &second_moments,
            &cpu.compact_gradients,
            step,
        );
        let metal_update = apply_loop_update(
            &masters,
            &first_moments,
            &second_moments,
            &metal.compact_gradients,
            step,
        );
        let scalar = scalar_loop_step(
            codec,
            &packed_bytes,
            &masters,
            &first_moments,
            &second_moments,
            step,
        );
        assert_close(
            &metal.predictions,
            &cpu.predictions,
            &format!("loop CPU/Metal predictions codec={codec:?} step={step}"),
        );
        assert_close(&[metal.loss], &[cpu.loss], "loop CPU/Metal loss");
        assert_close(
            &metal.compact_gradients,
            &cpu.compact_gradients,
            "loop CPU/Metal compact gradients",
        );
        assert_close(
            &metal.predictions,
            &scalar.predictions,
            "loop scalar predictions",
        );
        assert_close(&[metal.loss], &[scalar.loss], "loop scalar loss");
        assert_close(
            &metal.compact_gradients,
            &scalar.compact_gradients,
            "loop scalar compact gradients",
        );
        assert_close(
            &metal_coalesced,
            &cpu_coalesced,
            "loop CPU/Metal coalesced gradients",
        );
        assert_close(
            &metal_coalesced,
            &scalar.coalesced_gradients,
            "loop scalar coalesced gradients",
        );
        assert_close(
            &metal_update.masters,
            &cpu_update.masters,
            "loop CPU/Metal masters",
        );
        assert_close(
            &metal_update.first_moments,
            &cpu_update.first_moments,
            "loop CPU/Metal first moments",
        );
        assert_close(
            &metal_update.second_moments,
            &cpu_update.second_moments,
            "loop CPU/Metal second moments",
        );
        assert_close(
            &metal_update.masters,
            &scalar.update.masters,
            "loop scalar masters",
        );
        assert_close(
            &metal_update.first_moments,
            &scalar.update.first_moments,
            "loop scalar first moments",
        );
        assert_close(
            &metal_update.second_moments,
            &scalar.update.second_moments,
            "loop scalar second moments",
        );
        if step == 1 {
            assert_close(
                &metal_update.masters[..8],
                &[1.1, 0.0, 0.0, 0.6, 1.7, 0.0, -0.6, 0.0],
                "step-one updated masters",
            );
        } else {
            let expected_bytes = match codec {
                Codec::Bf8E5M2 => vec![60, 0, 0, 57, 63, 0, 185, 0, 67, 68, 69, 70],
                Codec::Bf4E2M1 => vec![2, 16, 3, 9, 101, 118],
                _ => panic!("fixture only uses BF8 E5M2 and BF4 E2M1"),
            };
            assert_eq!(packed_bytes, expected_bytes);
            assert_ne!(packed_bytes, encode_master(codec));
            match codec {
                Codec::Bf4E2M1 => {
                    assert_close(
                        &metal.predictions,
                        &[1.5, -0.5, 1.5, -0.5, 0.0, 0.5],
                        "step-two BF4 predictions",
                    );
                    assert_close(&[metal.loss], &[1.625], "step-two BF4 loss");
                    assert_close(
                        &metal.compact_gradients,
                        &[1.5, -0.5, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
                        "step-two BF4 compact gradients",
                    );
                }
                Codec::Bf8E5M2 => {
                    assert_close(
                        &metal.predictions,
                        &[1.75, -0.625, 1.75, -0.625, 0.0, 0.625],
                        "step-two BF8 predictions",
                    );
                    assert_close(&[metal.loss], &[2.3984375], "step-two BF8 loss");
                    assert_close(
                        &metal.compact_gradients,
                        &[
                            1.75, -0.625, 0.0, 0.0, 0.75, -0.625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.625,
                        ],
                        "step-two BF8 compact gradients",
                    );
                }
                _ => unreachable!(),
            }
            assert_ne!(
                metal_update.first_moments,
                vec![0.0; 12],
                "step two carries first moments"
            );
            assert_ne!(
                metal_update.second_moments,
                vec![0.0; 12],
                "step two carries second moments"
            );
        }
        masters = metal_update.masters.clone();
        first_moments = metal_update.first_moments.clone();
        second_moments = metal_update.second_moments.clone();
        records.push(LoopStep {
            codec,
            step,
            master_input,
            moment_one_input,
            moment_two_input,
            packed_bytes,
            metal,
            cpu,
            metal_coalesced,
            cpu_coalesced,
            metal_update,
            cpu_update,
            scalar,
        });
    }
    records
}

fn build_pilot_batch_fixture() -> PilotBatchFixture {
    let mut program = Vec::new();
    let expert_stack = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(2), Extent::Static(4), Extent::Static(4)],
            name: Some("experts".into()),
        },
    );
    let routes = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(8)],
            name: Some("expert_ids".into()),
        },
    );
    let activations = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(8), Extent::Static(4)],
            name: Some("inputs".into()),
        },
    );
    let targets = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(8), Extent::Static(4)],
            name: Some("targets".into()),
        },
    );
    let gathered_map = IndexMap::Computed {
        indices: routes,
        index_map: map::projection(3, &[0]),
        base: IndexPattern {
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
    let packed_gather = op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: vec![(expert_stack, gathered_map)],
            name: Some("selected_pilot_expert".into()),
        },
    );
    let product = op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (activations, IndexMap::Affine(map::projection(3, &[0, 1]))),
                (
                    packed_gather,
                    IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                ),
            ],
            name: None,
        },
    );
    let logits = op::append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 2])),
            keep: Keep::Reduce,
            name: Some("pilot_logits".into()),
        }),
    );
    let row_map = IndexMap::Affine(map::projection(2, &[0, 1]));
    let vector_map = IndexMap::Affine(map::projection(2, &[0]));
    let vector_one_map = IndexMap::Affine(map::projection(1, &[0]));
    let maximum = op::append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Maximum,
            init: ReduceInit::NegativeInfinity,
            operand: logits,
            in_map: row_map.clone(),
            out_map: vector_map.clone(),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    let centered = op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Subtract,
            operands: vec![(logits, row_map.clone()), (maximum, vector_map.clone())],
            name: None,
        },
    );
    let exponentials = op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Exponential,
            operands: vec![(centered, row_map.clone())],
            name: None,
        },
    );
    let exp_sum = op::append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: exponentials,
            in_map: row_map.clone(),
            out_map: vector_map.clone(),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    let log_sum = op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Logarithm,
            operands: vec![(exp_sum, vector_one_map.clone())],
            name: None,
        },
    );
    let log_normalizer = op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: vec![
                (maximum, vector_one_map.clone()),
                (log_sum, vector_one_map.clone()),
            ],
            name: None,
        },
    );
    let target_product = op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![(logits, row_map.clone()), (targets, row_map.clone())],
            name: None,
        },
    );
    let target_logits = op::append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: target_product,
            in_map: row_map.clone(),
            out_map: vector_map.clone(),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    let token_losses = op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Subtract,
            operands: vec![
                (log_normalizer, vector_one_map.clone()),
                (target_logits, vector_one_map.clone()),
            ],
            name: Some("pilot_token_losses".into()),
        },
    );
    let mean_scale = op::append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: Vec::new(),
            value: 0.125,
        },
    );
    let mean_scaled = op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (token_losses, IndexMap::Affine(map::projection(1, &[0]))),
                (mean_scale, IndexMap::Affine(map::projection(1, &[]))),
            ],
            name: None,
        },
    );
    let mean_loss = op::append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: mean_scaled,
            in_map: IndexMap::Affine(map::projection(1, &[0])),
            out_map: IndexMap::Affine(map::projection(1, &[])),
            keep: Keep::Reduce,
            name: Some("pilot_mean_loss".into()),
        }),
    );
    PilotBatchFixture {
        program,
        expert_stack,
        packed_gather,
        logits,
        token_losses,
        mean_loss,
    }
}

fn pilot_batch_inputs() -> ([f32; 8], [f32; 32], [f32; 32]) {
    let routes = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let mut activations = [0.0; 32];
    let mut targets = [0.0; 32];
    for token_index in 0..8 {
        activations[token_index * 4 + token_index % 4] = 1.0;
        targets[token_index * 4 + (token_index + 1) % 4] = 1.0;
    }
    (routes, activations, targets)
}

fn run_pilot_batch(
    fixture: &PilotBatchFixture,
    differentiated: &Differentiated,
    gathered: GatheredContribution,
    codec: Codec,
    packed_bytes: &[u8],
    backend: Engine,
) -> PilotBatchValues {
    let (routes, activations, targets) = pilot_batch_inputs();
    let named = [
        (
            "experts",
            QuantizedBlock::Packed {
                codec,
                bytes: packed_bytes,
            },
        ),
        ("expert_ids", QuantizedBlock::Float32(&routes)),
        ("inputs", QuantizedBlock::Float32(&activations)),
        ("targets", QuantizedBlock::Float32(&targets)),
    ];
    let outputs = [
        fixture.logits,
        fixture.token_losses,
        fixture.mean_loss,
        gathered.values,
    ];
    let evaluated = match backend {
        Engine::Cpu => {
            let expert_len = packed_bytes.len() / 2;
            let entries: [ExpertEntry<'_>; 2] = std::array::from_fn(|expert_index| {
                let start = expert_index * expert_len;
                ExpertEntry {
                    block: QuantizedBlock::Packed {
                        codec,
                        bytes: &packed_bytes[start..start + expert_len],
                    },
                    out_dim: 4,
                    in_dim: 4,
                    epoch: 1,
                }
            });
            let mut expert_sources = BTreeMap::new();
            expert_sources.insert(fixture.expert_stack, ExpertSource::new(&entries));
            evaluate_quantized_named_with_scratch_and_experts(
                &differentiated.program,
                &[],
                &named,
                &outputs,
                &mut Vec::new(),
                &mut None,
                Some(&expert_sources),
            )
            .expect("CPU evaluates the pilot packed batch")
        }
        Engine::Gpu => {
            let packed_operands: PackedOperands =
                [(fixture.expert_stack, codec)].into_iter().collect();
            let shapes = proxima_tensor::infer(&differentiated.program, &[])
                .expect("pilot graph shapes infer");
            let mut bound = proxima_tensor::bind(
                &differentiated.program,
                &shapes,
                &[fixture.packed_gather],
                NumericPolicy::default(),
            )
            .expect("pilot packed product binds");
            proxima_tensor::correct_packed_matmul_layouts(
                &mut bound,
                &BTreeSet::from([fixture.expert_stack]),
            );
            let resolved = bound
                .iter()
                .find(|item| item.node == fixture.packed_gather)
                .expect("packed pilot gather binds");
            let kernel = emit(resolved, &packed_operands, NumericPolicy::default())
                .expect("pilot packed kernel emits");
            let decoder = match codec {
                Codec::Bf8E5M2 => "bf8_e5m2_element(in0 +",
                Codec::Bf4E2M1 => "bf4_e2m1_element(in0 +",
                _ => panic!("pilot uses BF8/BF4"),
            };
            assert!(
                kernel.source.contains(decoder),
                "Metal source reads codec bytes directly"
            );
            let mut plan = plan_named(
                Engine::Gpu,
                Some(GpuDriver::Metal),
                &differentiated.program,
                &[],
                &named,
                &outputs,
                NumericPolicy::default(),
            )
            .expect("Metal plans the pilot packed batch");
            execute_plan_named(&mut plan, &named).expect("Metal executes the pilot packed batch")
        }
    };
    PilotBatchValues {
        logits: evaluated
            .get(fixture.logits)
            .expect("pilot logits exist")
            .0
            .to_vec(),
        token_losses: evaluated
            .get(fixture.token_losses)
            .expect("pilot token losses exist")
            .0
            .to_vec(),
        mean_loss: evaluated
            .get(fixture.mean_loss)
            .expect("pilot mean loss exists")
            .0[0],
        compact_gradients: evaluated
            .get(gathered.values)
            .expect("pilot gathered gradients exist")
            .0
            .to_vec(),
    }
}

fn pilot_batch_arrays(
    token_ids: &[u32; 8],
    target_ids: &[u32; 8],
    route_ids: &[f32; 8],
) -> ([f32; 8], [f32; 32], [f32; 32]) {
    let mut activations = [0.0; 32];
    let mut targets = [0.0; 32];
    for token_index in 0..8 {
        activations[token_index * 4 + token_ids[token_index] as usize % 4] = 1.0;
        targets[token_index * 4 + target_ids[token_index] as usize] = 1.0;
    }
    (*route_ids, activations, targets)
}

fn pilot_named_blocks<'a>(
    codec: Codec,
    packed_bytes: &'a [u8],
    routes: &'a [f32; 8],
    activations: &'a [f32; 32],
    targets: &'a [f32; 32],
) -> [(&'static str, QuantizedBlock<'a>); 4] {
    [
        (
            "experts",
            QuantizedBlock::Packed {
                codec,
                bytes: packed_bytes,
            },
        ),
        ("expert_ids", QuantizedBlock::Float32(routes)),
        ("inputs", QuantizedBlock::Float32(activations)),
        ("targets", QuantizedBlock::Float32(targets)),
    ]
}

fn pilot_outputs(fixture: &PilotBatchFixture, gathered: GatheredContribution) -> [NodeId; 4] {
    [
        fixture.logits,
        fixture.token_losses,
        fixture.mean_loss,
        gathered.values,
    ]
}

fn plan_pilot_metal(
    fixture: &PilotBatchFixture,
    differentiated: &Differentiated,
    gathered: GatheredContribution,
    codec: Codec,
    packed_bytes: &[u8],
    token_ids: &[u32; 8],
    target_ids: &[u32; 8],
    routes: &[f32; 8],
) -> Plan {
    let (routes, activations, targets) = pilot_batch_arrays(token_ids, target_ids, routes);
    let named = pilot_named_blocks(codec, packed_bytes, &routes, &activations, &targets);
    let packed_operands: PackedOperands = [(fixture.expert_stack, codec)].into_iter().collect();
    let shapes = proxima_tensor::infer(&differentiated.program, &[])
        .expect("64-step pilot graph shapes infer");
    let mut bound = proxima_tensor::bind(
        &differentiated.program,
        &shapes,
        &[fixture.packed_gather],
        NumericPolicy::default(),
    )
    .expect("64-step pilot packed product binds");
    proxima_tensor::correct_packed_matmul_layouts(
        &mut bound,
        &BTreeSet::from([fixture.expert_stack]),
    );
    let resolved = bound
        .iter()
        .find(|item| item.node == fixture.packed_gather)
        .expect("64-step packed gather binds");
    let kernel = emit(resolved, &packed_operands, NumericPolicy::default())
        .expect("64-step packed kernel emits");
    let decoder = match codec {
        Codec::Bf8E5M2 => "bf8_e5m2_element(in0 +",
        Codec::Bf4E2M1 => "bf4_e2m1_element(in0 +",
        _ => panic!("64-step pilot uses BF8/BF4"),
    };
    assert!(
        kernel.source.contains(decoder),
        "Metal decodes packed bytes directly"
    );
    plan_named(
        Engine::Gpu,
        Some(GpuDriver::Metal),
        &differentiated.program,
        &[],
        &named,
        &pilot_outputs(fixture, gathered),
        NumericPolicy::default(),
    )
    .expect("Metal builds one reusable 64-step plan")
}

fn execute_pilot_plan(
    fixture: &PilotBatchFixture,
    gathered: GatheredContribution,
    codec: Codec,
    packed_bytes: &[u8],
    token_ids: &[u32; 8],
    target_ids: &[u32; 8],
    route_ids: &[f32; 8],
    plan: &mut Plan,
) -> PilotBatchValues {
    let (routes, activations, targets) = pilot_batch_arrays(token_ids, target_ids, route_ids);
    let named = pilot_named_blocks(codec, packed_bytes, &routes, &activations, &targets);
    let evaluated =
        execute_plan_named(plan, &named).expect("reused Metal plan executes current payload");
    PilotBatchValues {
        logits: evaluated
            .get(fixture.logits)
            .expect("pilot logits exist")
            .0
            .to_vec(),
        token_losses: evaluated
            .get(fixture.token_losses)
            .expect("pilot token losses exist")
            .0
            .to_vec(),
        mean_loss: evaluated
            .get(fixture.mean_loss)
            .expect("pilot mean loss exists")
            .0[0],
        compact_gradients: evaluated
            .get(gathered.values)
            .expect("pilot gradients exist")
            .0
            .to_vec(),
    }
}

fn evaluate_pilot_cpu(
    fixture: &PilotBatchFixture,
    differentiated: &Differentiated,
    gathered: GatheredContribution,
    codec: Codec,
    packed_bytes: &[u8],
    token_ids: &[u32; 8],
    target_ids: &[u32; 8],
    route_ids: &[f32; 8],
) -> PilotBatchValues {
    let (routes, activations, targets) = pilot_batch_arrays(token_ids, target_ids, route_ids);
    let named = pilot_named_blocks(codec, packed_bytes, &routes, &activations, &targets);
    let expert_len = packed_bytes.len() / 2;
    let entries: [ExpertEntry<'_>; 2] = std::array::from_fn(|expert_index| {
        let start = expert_index * expert_len;
        ExpertEntry {
            block: QuantizedBlock::Packed {
                codec,
                bytes: &packed_bytes[start..start + expert_len],
            },
            out_dim: 4,
            in_dim: 4,
            epoch: 1,
        }
    });
    let mut expert_sources = BTreeMap::new();
    expert_sources.insert(fixture.expert_stack, ExpertSource::new(&entries));
    let outputs = pilot_outputs(fixture, gathered);
    let evaluated = evaluate_quantized_named_with_scratch_and_experts(
        &differentiated.program,
        &[],
        &named,
        &outputs,
        &mut Vec::new(),
        &mut None,
        Some(&expert_sources),
    )
    .expect("CPU evaluates current packed pilot payload");
    PilotBatchValues {
        logits: evaluated
            .get(fixture.logits)
            .expect("pilot logits exist")
            .0
            .to_vec(),
        token_losses: evaluated
            .get(fixture.token_losses)
            .expect("pilot token losses exist")
            .0
            .to_vec(),
        mean_loss: evaluated
            .get(fixture.mean_loss)
            .expect("pilot mean loss exists")
            .0[0],
        compact_gradients: evaluated
            .get(gathered.values)
            .expect("pilot gradients exist")
            .0
            .to_vec(),
    }
}

fn pilot_adam_state(
    masters: &[f32],
    gradients: &[f32],
    first_moments: &[f32],
    second_moments: &[f32],
    step: u32,
) -> AdamValues {
    let (program, updated_master, updated_first, updated_second) = pilot::optimizer_program();
    let step_value = [step as f32];
    let evaluated = proxima_tensor::cpu::evaluate_named(
        &program,
        &[],
        &[
            ("parameters", masters),
            ("gradients", gradients),
            ("first_moment", first_moments),
            ("second_moment", second_moments),
            ("step", &step_value),
        ],
        &[updated_master, updated_first, updated_second],
    )
    .expect("pilot Adam state update evaluates on host");
    AdamValues {
        masters: evaluated
            .get(updated_master)
            .expect("updated masters exist")
            .0
            .to_vec(),
        first_moments: evaluated
            .get(updated_first)
            .expect("updated first moments exist")
            .0
            .to_vec(),
        second_moments: evaluated
            .get(updated_second)
            .expect("updated second moments exist")
            .0
            .to_vec(),
    }
}

fn scalar_pilot_batch_routed(
    codec: Codec,
    packed_bytes: &[u8],
    token_ids: &[u32; 8],
    target_ids: &[u32; 8],
    routes: &[f32; 8],
    masters: &[f32],
    first_moments: &[f32],
    second_moments: &[f32],
    step: u32,
) -> (PilotBatchValues, Vec<f32>, AdamValues) {
    let decoded = match codec {
        Codec::Bf8E5M2 => packed_bytes
            .iter()
            .map(|byte| proxima_gguf::quant::bf8_e5m2::decode(*byte))
            .collect::<Vec<_>>(),
        Codec::Bf4E2M1 => packed_bytes
            .iter()
            .flat_map(|byte| proxima_gguf::quant::bf4_e2m1::unpack_pair(*byte))
            .collect::<Vec<_>>(),
        _ => panic!("scalar pilot supports BF8/BF4"),
    };
    let mut logits = Vec::with_capacity(32);
    let mut token_losses = Vec::with_capacity(8);
    let mut compact_gradients = vec![0.0; 128];
    for token_index in 0..8 {
        let token_id = token_ids[token_index] as usize;
        let target_id = target_ids[token_index] as usize;
        let expert_id = routes[token_index] as usize;
        let row_start = expert_id * 16 + token_id * 4;
        let token_logits = &decoded[row_start..row_start + 4];
        let maximum = token_logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let exponentials = token_logits
            .iter()
            .map(|value| (value - maximum).exp())
            .collect::<Vec<_>>();
        let exponent_sum = exponentials.iter().sum::<f32>();
        let probabilities = exponentials
            .iter()
            .map(|value| value / exponent_sum)
            .collect::<Vec<_>>();
        logits.extend_from_slice(token_logits);
        token_losses.push(maximum + exponent_sum.ln() - token_logits[target_id]);
        let input_index = token_id % 4;
        for output_index in 0..4 {
            let gradient =
                probabilities[output_index] - if output_index == target_id { 1.0 } else { 0.0 };
            compact_gradients[token_index * 16 + input_index * 4 + output_index] = gradient / 8.0;
        }
    }
    let (_, coalesced) = dedupe_and_sum_rows(routes, &compact_gradients, 16)
        .expect("scalar route gradients coalesce");
    let summed_token_gradients = coalesced
        .iter()
        .map(|gradient| gradient * 8.0)
        .collect::<Vec<_>>();
    let updated = pilot::scalar_adam_step(
        masters,
        first_moments,
        second_moments,
        &summed_token_gradients,
        step,
    );
    let mean_loss = token_losses.iter().sum::<f32>() / 8.0;
    (
        PilotBatchValues {
            logits,
            token_losses,
            mean_loss,
            compact_gradients,
        },
        coalesced,
        AdamValues {
            masters: updated.0,
            first_moments: updated.1,
            second_moments: updated.2,
        },
    )
}

fn pilot_adam(masters: &[f32], gradients: &[f32]) -> AdamValues {
    let (program, updated_master, updated_first, updated_second) = pilot::optimizer_program();
    let evaluated = proxima_tensor::cpu::evaluate_named(
        &program,
        &[],
        &[
            ("parameters", masters),
            ("gradients", gradients),
            ("first_moment", &[0.0; 32]),
            ("second_moment", &[0.0; 32]),
            ("step", &[1.0]),
        ],
        &[updated_master, updated_first, updated_second],
    )
    .expect("pilot Adam update evaluates on host");
    AdamValues {
        masters: evaluated
            .get(updated_master)
            .expect("updated pilot masters exist")
            .0
            .to_vec(),
        first_moments: evaluated
            .get(updated_first)
            .expect("pilot first moments exist")
            .0
            .to_vec(),
        second_moments: evaluated
            .get(updated_second)
            .expect("pilot second moments exist")
            .0
            .to_vec(),
    }
}

fn pilot_seed17(codec: Codec) -> (Vec<f32>, Vec<u8>) {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../proxima-tensor/specs/packed_pilot_moe_training/results/packed-tiny-pilot.json"
    ))
    .expect("the retained tiny-pilot JSON parses");
    let format_name = match codec {
        Codec::Bf8E5M2 => "bf8_e5m2",
        Codec::Bf4E2M1 => "bf4_e2m1",
        _ => panic!("pilot uses BF8 and BF4"),
    };
    let arm = fixture["arms"]
        .as_array()
        .expect("pilot arms exist")
        .iter()
        .find(|arm| arm["seed"] == 17 && arm["format"] == format_name)
        .expect("seed-17 codec arm exists");
    let step = &arm["steps"][0];
    let masters = serde_json::from_value(arm["initial_parameters"].clone())
        .expect("pilot initial masters decode");
    let bytes = serde_json::from_value(step["train_packed_bytes"].clone())
        .expect("pilot packed training bytes decode");
    (masters, bytes)
}

fn scalar_pilot_batch(
    codec: Codec,
    masters: &[f32],
    bytes: &[u8],
) -> (PilotBatchValues, Vec<f32>, AdamValues) {
    let format = match codec {
        Codec::Bf8E5M2 => proxima_autograd::low_precision::WeightFormat::Bf8E5M2,
        Codec::Bf4E2M1 => proxima_autograd::low_precision::WeightFormat::Bf4E2M1,
        _ => panic!("pilot uses BF8 and BF4"),
    };
    let token_ids = [0_u32, 1, 2, 3, 0, 1, 2, 3];
    let target_ids = [1_u32, 2, 3, 0, 1, 2, 3, 0];
    let mut logits = Vec::with_capacity(32);
    let mut token_losses = Vec::with_capacity(8);
    let mut compact_gradients = Vec::with_capacity(128);
    for (token_id, target_id) in token_ids.into_iter().zip(target_ids) {
        let (token_logits, token_loss, token_gradient) =
            pilot::scalar_token(masters, bytes, format, token_id, target_id);
        logits.extend(token_logits);
        token_losses.push(token_loss);
        compact_gradients.extend(token_gradient.into_iter().map(|value| value / 8.0));
    }
    let routes = [0.0_f32, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let (_, coalesced_gradients) = dedupe_and_sum_rows(&routes, &compact_gradients, 16)
        .expect("scalar token gradients coalesce by expert");
    let mut updated_masters = masters.to_vec();
    let mut first_moment = vec![0.0; 32];
    let mut second_moment = vec![0.0; 32];
    for index in 0..32 {
        let gradient = coalesced_gradients[index];
        first_moment[index] = 0.1 * gradient;
        second_moment[index] = 0.001 * gradient * gradient;
        updated_masters[index] -= 0.001 * gradient / (gradient.abs() + 1e-8);
    }
    let mean_loss = token_losses.iter().sum::<f32>() / 8.0;
    (
        PilotBatchValues {
            logits,
            token_losses,
            mean_loss,
            compact_gradients,
        },
        coalesced_gradients,
        AdamValues {
            masters: updated_masters,
            first_moments: first_moment,
            second_moments: second_moment,
        },
    )
}

fn run_pilot_observation(codec: Codec) -> serde_json::Value {
    let (masters, packed_bytes) = pilot_seed17(codec);
    let fixture = build_pilot_batch_fixture();
    let differentiated = differentiate(&fixture.program, fixture.mean_loss)
        .expect("the batched pilot mean cross-entropy differentiates");
    let gathered = differentiated
        .gathered_gradients_of(fixture.expert_stack)
        .next()
        .expect("pilot expert matrix has gathered gradients");
    let cpu = run_pilot_batch(
        &fixture,
        &differentiated,
        gathered,
        codec,
        &packed_bytes,
        Engine::Cpu,
    );
    let (scalar, scalar_coalesced, scalar_update) =
        scalar_pilot_batch(codec, &masters, &packed_bytes);
    assert_close(&cpu.logits, &scalar.logits, "pilot CPU/scalar logits");
    assert_close(
        &cpu.token_losses,
        &scalar.token_losses,
        "pilot CPU/scalar token losses",
    );
    assert_close(
        &cpu.compact_gradients,
        &scalar.compact_gradients,
        "pilot CPU/scalar compact gradients",
    );
    let metal = run_pilot_batch(
        &fixture,
        &differentiated,
        gathered,
        codec,
        &packed_bytes,
        Engine::Gpu,
    );
    let routes = [0.0_f32, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let (coalesced_routes, coalesced_gradients) =
        dedupe_and_sum_rows(&routes, &metal.compact_gradients, 16)
            .expect("Metal compact pilot gradients coalesce");
    let (_, cpu_coalesced) = dedupe_and_sum_rows(&routes, &cpu.compact_gradients, 16)
        .expect("CPU compact pilot gradients coalesce");
    let metal_update = pilot_adam(&masters, &coalesced_gradients);
    let cpu_update = pilot_adam(&masters, &cpu_coalesced);
    assert_eq!(coalesced_routes, [0, 1]);
    assert_close(&metal.logits, &cpu.logits, "pilot Metal/CPU logits");
    assert_close(&metal.logits, &scalar.logits, "pilot Metal/scalar logits");
    assert_close(
        &metal.token_losses,
        &cpu.token_losses,
        "pilot Metal/CPU token losses",
    );
    assert_close(
        &metal.token_losses,
        &scalar.token_losses,
        "pilot Metal/scalar token losses",
    );
    assert_close(
        &[metal.mean_loss],
        &[cpu.mean_loss],
        "pilot Metal/CPU mean loss",
    );
    assert_close(
        &[metal.mean_loss],
        &[scalar.mean_loss],
        "pilot Metal/scalar mean loss",
    );
    assert_close(
        &metal.compact_gradients,
        &cpu.compact_gradients,
        "pilot Metal/CPU compact gradients",
    );
    assert_close(
        &metal.compact_gradients,
        &scalar.compact_gradients,
        "pilot Metal/scalar compact gradients",
    );
    assert_close(
        &coalesced_gradients,
        &cpu_coalesced,
        "pilot Metal/CPU coalesced gradients",
    );
    assert_close(
        &coalesced_gradients,
        &scalar_coalesced,
        "pilot Metal/scalar coalesced gradients",
    );
    assert_close(
        &metal_update.masters,
        &cpu_update.masters,
        "pilot Metal/CPU updated masters",
    );
    assert_close(
        &metal_update.masters,
        &scalar_update.masters,
        "pilot Metal/scalar updated masters",
    );
    assert_close(
        &metal_update.first_moments,
        &scalar_update.first_moments,
        "pilot Metal/scalar first moments",
    );
    assert_close(
        &metal_update.second_moments,
        &scalar_update.second_moments,
        "pilot Metal/scalar second moments",
    );
    let reshape = |values: &[f32], row: usize| -> Vec<Vec<f32>> {
        values.chunks_exact(row).map(<[f32]>::to_vec).collect()
    };
    let encode = |backend: &str,
                  output: &PilotBatchValues,
                  update: &AdamValues,
                  coalesced: &[f32]| {
        serde_json::json!({
            "backend": backend, "logits": reshape(&output.logits, 4), "token_losses": output.token_losses,
            "mean_loss": output.mean_loss, "compact_gradients": reshape(&output.compact_gradients, 16),
            "coalesced_gradients": reshape(coalesced, 16), "updated_masters": update.masters,
            "first_moment": update.first_moments, "second_moment": update.second_moments,
        })
    };
    let (_routes, activations, targets) = pilot_batch_inputs();
    let routes_i32 = [0, 1, 0, 1, 0, 1, 0, 1];
    let token_ids = [0, 1, 2, 3, 0, 1, 2, 3];
    let target_ids = [1, 2, 3, 0, 1, 2, 3, 0];
    serde_json::json!({
        "codec": format!("{codec:?}"), "seed": 17, "step": 1,
        "routes": routes_i32, "token_ids": token_ids, "target_ids": target_ids,
        "targets": reshape(&targets, 4), "initial_masters": masters,
        "packed_bytes": packed_bytes, "packed_byte_len": packed_bytes.len(),
        "backend": "metal", "logits": reshape(&metal.logits, 4),
        "token_losses": metal.token_losses, "mean_loss": metal.mean_loss,
        "compact_gradients": reshape(&metal.compact_gradients, 16),
        "coalesced_gradients": reshape(&coalesced_gradients, 16),
        "updated_masters": metal_update.masters, "first_moment": metal_update.first_moments,
        "second_moment": metal_update.second_moments,
        "cpu_reference": encode("cpu", &cpu, &cpu_update, &cpu_coalesced),
        "scalar_reference": encode("scalar", &scalar, &scalar_update, &scalar_coalesced),
        "activations": reshape(&activations, 4),
    })
}

#[test]
fn packed_pilot_cross_entropy_64step_records_payload() {
    const TRAIN_IDS: [u32; 8] = [0, 1, 2, 3, 0, 1, 2, 3];
    const TRAIN_TARGETS: [u32; 8] = [1, 2, 3, 0, 1, 2, 3, 0];
    const TRAIN_ROUTES: [f32; 8] = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    const HELD_OUT_IDS: [u32; 8] = [0, 0, 1, 1, 2, 2, 3, 3];
    const HELD_OUT_TARGETS: [u32; 8] = [0, 1, 1, 2, 2, 3, 3, 0];
    const HELD_OUT_ROUTES: [f32; 8] = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
    const TRAIN_ROUTES_JSON: [u8; 8] = [0, 1, 0, 1, 0, 1, 0, 1];
    const HELD_OUT_ROUTES_JSON: [u8; 8] = [0, 0, 1, 1, 0, 0, 1, 1];

    let mut records = Vec::with_capacity(128);
    for codec in [Codec::Bf8E5M2, Codec::Bf4E2M1] {
        let (mut masters, _) = pilot_seed17(codec);
        let mut first_moments = vec![0.0; 32];
        let mut second_moments = vec![0.0; 32];
        let fixture = build_pilot_batch_fixture();
        let differentiated = differentiate(&fixture.program, fixture.mean_loss)
            .expect("64-step pilot graph differentiates");
        let gathered = differentiated
            .gathered_gradients_of(fixture.expert_stack)
            .next()
            .expect("64-step expert gradients gather");
        let initial_bytes = encode_masters(codec, &masters);
        let mut plan_build_count = 0;
        let mut plan = plan_pilot_metal(
            &fixture,
            &differentiated,
            gathered,
            codec,
            &initial_bytes,
            &TRAIN_IDS,
            &TRAIN_TARGETS,
            &TRAIN_ROUTES,
        );
        plan_build_count += 1;
        let mut plan_execution_count = 0;
        for step in 1..=64_u32 {
            let master_input = masters.clone();
            let first_moment_input = first_moments.clone();
            let second_moment_input = second_moments.clone();
            let train_packed_bytes = encode_masters(codec, &master_input);
            let metal_train = execute_pilot_plan(
                &fixture,
                gathered,
                codec,
                &train_packed_bytes,
                &TRAIN_IDS,
                &TRAIN_TARGETS,
                &TRAIN_ROUTES,
                &mut plan,
            );
            plan_execution_count += 1;
            let cpu_train = evaluate_pilot_cpu(
                &fixture,
                &differentiated,
                gathered,
                codec,
                &train_packed_bytes,
                &TRAIN_IDS,
                &TRAIN_TARGETS,
                &TRAIN_ROUTES,
            );
            let (scalar_train, scalar_coalesced, scalar_update) = scalar_pilot_batch_routed(
                codec,
                &train_packed_bytes,
                &TRAIN_IDS,
                &TRAIN_TARGETS,
                &TRAIN_ROUTES,
                &master_input,
                &first_moment_input,
                &second_moment_input,
                step,
            );
            let (coalesced_routes, metal_coalesced) =
                dedupe_and_sum_rows(&TRAIN_ROUTES, &metal_train.compact_gradients, 16)
                    .expect("Metal train gradients coalesce by route");
            let (_, cpu_coalesced) =
                dedupe_and_sum_rows(&TRAIN_ROUTES, &cpu_train.compact_gradients, 16)
                    .expect("CPU train gradients coalesce by route");
            assert_eq!(coalesced_routes, [0, 1]);
            let metal_update = pilot_adam_state(
                &master_input,
                &metal_coalesced,
                &first_moment_input,
                &second_moment_input,
                step,
            );
            let cpu_update = pilot_adam_state(
                &master_input,
                &cpu_coalesced,
                &first_moment_input,
                &second_moment_input,
                step,
            );
            for (metal_values, cpu_values, scalar_values, label) in [
                (
                    &metal_train.logits,
                    &cpu_train.logits,
                    &scalar_train.logits,
                    "train logits",
                ),
                (
                    &metal_train.token_losses,
                    &cpu_train.token_losses,
                    &scalar_train.token_losses,
                    "train token losses",
                ),
                (
                    &metal_train.compact_gradients,
                    &cpu_train.compact_gradients,
                    &scalar_train.compact_gradients,
                    "compact gradients",
                ),
                (
                    &metal_coalesced,
                    &cpu_coalesced,
                    &scalar_coalesced,
                    "coalesced gradients",
                ),
                (
                    &metal_update.masters,
                    &cpu_update.masters,
                    &scalar_update.masters,
                    "updated masters",
                ),
                (
                    &metal_update.first_moments,
                    &cpu_update.first_moments,
                    &scalar_update.first_moments,
                    "updated first moments",
                ),
                (
                    &metal_update.second_moments,
                    &cpu_update.second_moments,
                    &scalar_update.second_moments,
                    "updated second moments",
                ),
            ] {
                assert_close(
                    metal_values,
                    cpu_values,
                    &format!("64-step CPU/Metal {label} at {step}"),
                );
                assert_close(
                    metal_values,
                    scalar_values,
                    &format!("64-step scalar/Metal {label} at {step}"),
                );
            }
            assert_close(
                &[metal_train.mean_loss],
                &[cpu_train.mean_loss],
                "64-step CPU/Metal train loss",
            );
            assert_close(
                &[metal_train.mean_loss],
                &[scalar_train.mean_loss],
                "64-step scalar/Metal train loss",
            );

            let held_out_packed_bytes = encode_masters(codec, &metal_update.masters);
            let metal_held_out = execute_pilot_plan(
                &fixture,
                gathered,
                codec,
                &held_out_packed_bytes,
                &HELD_OUT_IDS,
                &HELD_OUT_TARGETS,
                &HELD_OUT_ROUTES,
                &mut plan,
            );
            plan_execution_count += 1;
            let cpu_held_out = evaluate_pilot_cpu(
                &fixture,
                &differentiated,
                gathered,
                codec,
                &held_out_packed_bytes,
                &HELD_OUT_IDS,
                &HELD_OUT_TARGETS,
                &HELD_OUT_ROUTES,
            );
            let (scalar_held_out, _, _) = scalar_pilot_batch_routed(
                codec,
                &held_out_packed_bytes,
                &HELD_OUT_IDS,
                &HELD_OUT_TARGETS,
                &HELD_OUT_ROUTES,
                &metal_update.masters,
                &metal_update.first_moments,
                &metal_update.second_moments,
                step,
            );
            assert_close(
                &metal_held_out.logits,
                &cpu_held_out.logits,
                "held-out CPU/Metal logits",
            );
            assert_close(
                &metal_held_out.logits,
                &scalar_held_out.logits,
                "held-out scalar/Metal logits",
            );
            assert_close(
                &metal_held_out.token_losses,
                &cpu_held_out.token_losses,
                "held-out CPU/Metal token losses",
            );
            assert_close(
                &metal_held_out.token_losses,
                &scalar_held_out.token_losses,
                "held-out scalar/Metal token losses",
            );
            assert_close(
                &[metal_held_out.mean_loss],
                &[cpu_held_out.mean_loss],
                "held-out CPU/Metal mean loss",
            );
            assert_close(
                &[metal_held_out.mean_loss],
                &[scalar_held_out.mean_loss],
                "held-out scalar/Metal mean loss",
            );

            let encode_reference =
                |backend: &str,
                 train: &PilotBatchValues,
                 coalesced: &[f32],
                 update: &AdamValues,
                 held_out: &PilotBatchValues| {
                    serde_json::json!({
                        "backend": backend,
                        "train_logits": train.logits.chunks_exact(4).map(<[f32]>::to_vec).collect::<Vec<_>>(),
                        "train_token_losses": train.token_losses,
                        "train_mean_loss": train.mean_loss,
                        "compact_gradients": train.compact_gradients.chunks_exact(16).map(<[f32]>::to_vec).collect::<Vec<_>>(),
                        "coalesced_gradients": coalesced.chunks_exact(16).map(<[f32]>::to_vec).collect::<Vec<_>>(),
                        "updated_masters": update.masters,
                        "updated_first_moment": update.first_moments,
                        "updated_second_moment": update.second_moments,
                        "held_out_logits": held_out.logits.chunks_exact(4).map(<[f32]>::to_vec).collect::<Vec<_>>(),
                        "held_out_token_losses": held_out.token_losses,
                        "held_out_mean_loss": held_out.mean_loss,
                    })
                };
            records.push(serde_json::json!({
                "codec": codec_name(codec), "seed": 17, "step": step, "backend": "metal",
                "plan_build_count": plan_build_count, "plan_execution_count": plan_execution_count,
                "master_input": master_input, "first_moment_input": first_moment_input,
                "second_moment_input": second_moment_input, "train_packed_bytes": train_packed_bytes,
                "routes": TRAIN_ROUTES_JSON, "token_ids": TRAIN_IDS, "target_ids": TRAIN_TARGETS,
                "train_logits": metal_train.logits.chunks_exact(4).map(<[f32]>::to_vec).collect::<Vec<_>>(),
                "train_token_losses": metal_train.token_losses, "train_mean_loss": metal_train.mean_loss,
                "compact_gradients": metal_train.compact_gradients.chunks_exact(16).map(<[f32]>::to_vec).collect::<Vec<_>>(),
                "coalesced_gradients": metal_coalesced.chunks_exact(16).map(<[f32]>::to_vec).collect::<Vec<_>>(),
                "updated_masters": metal_update.masters, "updated_first_moment": metal_update.first_moments,
                "updated_second_moment": metal_update.second_moments, "held_out_packed_bytes": held_out_packed_bytes,
                "held_out_routes": HELD_OUT_ROUTES_JSON, "held_out_token_ids": HELD_OUT_IDS,
                "held_out_target_ids": HELD_OUT_TARGETS,
                "held_out_logits": metal_held_out.logits.chunks_exact(4).map(<[f32]>::to_vec).collect::<Vec<_>>(),
                "held_out_token_losses": metal_held_out.token_losses,
                "held_out_mean_loss": metal_held_out.mean_loss,
                "cpu_reference": encode_reference("cpu", &cpu_train, &cpu_coalesced, &cpu_update, &cpu_held_out),
                "scalar_reference": encode_reference("scalar", &scalar_train, &scalar_coalesced, &scalar_update, &scalar_held_out),
            }));
            masters = metal_update.masters;
            first_moments = metal_update.first_moments;
            second_moments = metal_update.second_moments;
        }
        assert_eq!(plan_build_count, 1);
        assert_eq!(plan_execution_count, 128);
    }
    assert_eq!(records.len(), 128);
    println!(
        "METAL_PACKED_PILOT_64STEP_JSON={}",
        serde_json::to_string(&records).expect("64-step payload serializes")
    );
}

#[test]
fn packed_pilot_cross_entropy_matches_cpu_and_scalar() {
    for codec in [Codec::Bf8E5M2, Codec::Bf4E2M1] {
        let _record = run_pilot_observation(codec);
    }
}

#[test]
fn packed_pilot_cross_entropy_records_payload() {
    let records = [
        run_pilot_observation(Codec::Bf8E5M2),
        run_pilot_observation(Codec::Bf4E2M1),
    ];
    println!(
        "METAL_PACKED_PILOT_JSON={}",
        serde_json::to_string(&records).expect("pilot payload serializes")
    );
}

fn assert_observation(observation: &Observation) {
    assert_close(
        &observation.metal.predictions,
        &observation.cpu.predictions,
        "CPU/Metal predictions",
    );
    assert_close(
        &[observation.metal.loss],
        &[observation.cpu.loss],
        "CPU/Metal loss",
    );
    assert_close(
        &observation.metal.compact_gradients,
        &observation.cpu.compact_gradients,
        "CPU/Metal compact gradients",
    );
    assert_close(
        &observation.cpu.predictions,
        &observation.reference.predictions,
        "CPU reference predictions",
    );
    assert_close(
        &[observation.cpu.loss],
        &[observation.reference.loss],
        "CPU reference loss",
    );
    assert_close(
        &observation.cpu.compact_gradients,
        &observation.reference.compact_gradients,
        "CPU reference compact gradients",
    );
    assert_close(
        &observation.coalesced_gradients,
        &observation.reference.coalesced_gradients,
        "coalesced gradients against independent reference",
    );
    assert_close(
        &observation.metal.predictions,
        &EXPECTED_PREDICTIONS,
        "predictions",
    );
    assert_close(&[observation.metal.loss], &[4.0], "loss");
    assert_close(
        &observation.metal.compact_gradients,
        &EXPECTED_COMPACT_GRADIENTS,
        "compact gradients",
    );
    assert_close(
        &observation.coalesced_gradients,
        &EXPECTED_COALESCED_GRADIENTS,
        "coalesced gradients",
    );
    assert_close(
        &observation.cpu_update.masters,
        &observation.reference.update.masters,
        "CPU masters against independent reference",
    );
    assert_close(
        &observation.metal_update.masters,
        &observation.reference.update.masters,
        "Metal masters against independent reference",
    );
    assert_close(
        &observation.metal_update.first_moments,
        &observation.reference.update.first_moments,
        "Metal first moments against independent reference",
    );
    assert_close(
        &observation.metal_update.second_moments,
        &observation.reference.update.second_moments,
        "Metal second moments against independent reference",
    );
    assert_eq!(
        &observation.metal_update.masters[8..12],
        &[3.3, 4.0, 5.0, 6.0]
    );
    assert_eq!(&observation.metal_update.first_moments[8..12], &[0.0; 4]);
    assert_eq!(&observation.metal_update.second_moments[8..12], &[0.0; 4]);
}

#[test]
fn metal_differentiated_packed_moe_matches_cpu() {
    for codec in [Codec::Bf8E5M2, Codec::Bf4E2M1] {
        let observation = observe(codec);
        assert_close(
            &observation.metal.predictions,
            &observation.cpu.predictions,
            "predictions",
        );
        assert_close(&[observation.metal.loss], &[observation.cpu.loss], "loss");
        assert_close(
            &observation.metal.compact_gradients,
            &observation.cpu.compact_gradients,
            "compact gradients",
        );
        assert_close(
            &observation.metal.predictions,
            &EXPECTED_PREDICTIONS,
            "pinned predictions",
        );
        assert_close(&[observation.metal.loss], &[4.0], "pinned loss");
        assert_close(
            &observation.metal.compact_gradients,
            &EXPECTED_COMPACT_GRADIENTS,
            "pinned compact gradients",
        );
    }
}

#[test]
fn metal_compact_gradient_drives_sparse_master_update() {
    for codec in [Codec::Bf8E5M2, Codec::Bf4E2M1] {
        let observation = observe(codec);
        assert_observation(&observation);
    }
}

#[test]
fn metal_training_payload_record_is_complete() {
    let observations = [observe(Codec::Bf8E5M2), observe(Codec::Bf4E2M1)];
    for observation in &observations {
        assert_observation(observation);
    }
    let json = format!(
        "[\n{},\n{}\n]\n",
        record_json(&observations[0]),
        record_json(&observations[1])
    );
    assert_eq!(json.matches("\"codec\":").count(), 2);
    assert_eq!(json.matches("\"backend\":\"metal\"").count(), 2);
    assert_eq!(json.matches("\"backend\":\"cpu\"").count(), 2);
    assert_eq!(json.matches("\"backend\":\"scalar\"").count(), 2);
    for required_key in [
        "codec",
        "seed",
        "step",
        "backend",
        "master_input",
        "routes",
        "activations",
        "targets",
        "packed_bytes",
        "predictions",
        "loss",
        "compact_gradient_rows",
        "coalesced_gradient_rows",
        "updated_master_rows",
        "first_moment_rows",
        "second_moment_rows",
        "cpu_reference",
        "scalar_reference",
    ] {
        assert!(
            json.contains(&format!("\"{required_key}\":")),
            "missing JSON key {required_key}"
        );
    }
    assert!(json.contains("\"codec\":\"Bf8E5M2\""));
    assert!(json.contains("\"codec\":\"Bf4E2M1\""));
    println!("METAL_TRAINING_JSON={json}");

    let result_directory = tempfile::tempdir().expect("isolated result directory is created");
    let path = result_directory
        .path()
        .join("results/metal-packed-training.json");
    fs::create_dir_all(path.parent().expect("result parent exists"))
        .expect("result directory is created");
    fs::write(&path, &json).expect("Metal training evidence is written");
    let persisted = fs::read_to_string(path).expect("Metal training evidence is readable");
    assert_eq!(persisted, json);
}

#[test]
fn metal_training_loop_reencodes_masters_and_carries_adam_state() {
    for codec in [Codec::Bf8E5M2, Codec::Bf4E2M1] {
        let records = observe_loop(codec);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].step, 1);
        assert_eq!(records[1].step, 2);
        assert_close(
            &records[1].master_input,
            &records[0].metal_update.masters,
            "master carry-forward",
        );
        assert_close(
            &records[1].moment_one_input,
            &records[0].metal_update.first_moments,
            "first moment carry-forward",
        );
        assert_close(
            &records[1].moment_two_input,
            &records[0].metal_update.second_moments,
            "second moment carry-forward",
        );
        assert_eq!(
            records[1].packed_bytes,
            encode_masters(codec, &records[1].master_input)
        );
    }
}

#[test]
fn metal_training_loop_records_each_step() {
    let mut records = Vec::new();
    for codec in [Codec::Bf8E5M2, Codec::Bf4E2M1] {
        records.extend(observe_loop(codec));
    }
    assert_eq!(records.len(), 4);
    assert_eq!(
        records
            .iter()
            .map(|record| (codec_name(record.codec), record.step))
            .collect::<BTreeSet<_>>()
            .len(),
        4
    );
    let json = format!(
        "[{}]",
        records
            .iter()
            .map(loop_record_json)
            .collect::<Vec<_>>()
            .join(",")
    );
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("loop evidence JSON parses");
    let parsed_records = parsed.as_array().expect("loop evidence is an array");
    assert_eq!(parsed_records.len(), 4);
    for record in parsed_records {
        for key in [
            "codec",
            "step",
            "backend",
            "master_input",
            "moment_one_input",
            "moment_two_input",
            "routes",
            "activations",
            "targets",
            "packed_bytes",
            "predictions",
            "loss",
            "compact_gradient_rows",
            "coalesced_gradient_rows",
            "updated_master_rows",
            "first_moment_rows",
            "second_moment_rows",
            "cpu_reference",
            "scalar_reference",
        ] {
            assert!(record.get(key).is_some(), "missing record field {key}");
        }
        assert_eq!(record["backend"], "metal");
        for nested_name in ["cpu_reference", "scalar_reference"] {
            for key in [
                "backend",
                "predictions",
                "loss",
                "compact_gradient_rows",
                "coalesced_gradient_rows",
                "updated_master_rows",
                "first_moment_rows",
                "second_moment_rows",
            ] {
                assert!(
                    record[nested_name].get(key).is_some(),
                    "missing {nested_name}.{key}"
                );
            }
        }
        assert_eq!(record["cpu_reference"]["backend"], "cpu");
        assert_eq!(record["scalar_reference"]["backend"], "scalar");
    }
    println!("METAL_TRAINING_LOOP_JSON={json}");
    let result_directory = tempfile::tempdir().expect("isolated result directory is created");
    let path = result_directory
        .path()
        .join("results/metal-packed-training-loop.json");
    fs::create_dir_all(path.parent().expect("result parent exists"))
        .expect("result directory is created");
    fs::write(&path, &json).expect("two-step evidence is written");
    assert_eq!(
        fs::read_to_string(path).expect("two-step evidence is readable"),
        json
    );
}

fn loop_record_json(record: &LoopStep) -> String {
    let cpu_update = &record.cpu_update;
    let metal_update = &record.metal_update;
    let scalar = &record.scalar;
    format!(
        "{{\"codec\":\"{}\",\"step\":{},\"backend\":\"metal\",\"master_input\":{},\"moment_one_input\":{},\"moment_two_input\":{},\"routes\":{},\"activations\":{},\"targets\":{},\"packed_bytes\":{},\"predictions\":{},\"loss\":{},\"compact_gradient_rows\":{},\"coalesced_gradient_rows\":{},\"updated_master_rows\":{},\"first_moment_rows\":{},\"second_moment_rows\":{},\"cpu_reference\":{{\"backend\":\"cpu\",\"predictions\":{},\"loss\":{},\"compact_gradient_rows\":{},\"coalesced_gradient_rows\":{},\"updated_master_rows\":{},\"first_moment_rows\":{},\"second_moment_rows\":{}}},\"scalar_reference\":{{\"backend\":\"scalar\",\"predictions\":{},\"loss\":{},\"compact_gradient_rows\":{},\"coalesced_gradient_rows\":{},\"updated_master_rows\":{},\"first_moment_rows\":{},\"second_moment_rows\":{}}}}}",
        codec_name(record.codec),
        record.step,
        json_f32(&record.master_input),
        json_f32(&record.moment_one_input),
        json_f32(&record.moment_two_input),
        json_i32(&ROUTES),
        json_f32(&ACTIVATIONS),
        json_f32(&TARGETS),
        json_u8(&record.packed_bytes),
        json_f32(&record.metal.predictions),
        record.metal.loss,
        json_rows(&record.metal.compact_gradients, 4),
        json_rows(&record.metal_coalesced, 4),
        json_rows(&metal_update.masters, 4),
        json_rows(&metal_update.first_moments, 4),
        json_rows(&metal_update.second_moments, 4),
        json_f32(&record.cpu.predictions),
        record.cpu.loss,
        json_rows(&record.cpu.compact_gradients, 4),
        json_rows(&record.cpu_coalesced, 4),
        json_rows(&cpu_update.masters, 4),
        json_rows(&cpu_update.first_moments, 4),
        json_rows(&cpu_update.second_moments, 4),
        json_f32(&scalar.predictions),
        scalar.loss,
        json_rows(&scalar.compact_gradients, 4),
        json_rows(&scalar.coalesced_gradients, 4),
        json_rows(&scalar.update.masters, 4),
        json_rows(&scalar.update.first_moments, 4),
        json_rows(&scalar.update.second_moments, 4)
    )
}

fn record_json(observation: &Observation) -> String {
    let mut record = String::new();
    let codec_name = codec_name(observation.codec);
    write!(
        record,
        "{{\"codec\":\"{codec_name}\",\"seed\":17,\"step\":1,\"backend\":\"metal\",\"master_input\":{},\"routes\":{},\"activations\":{},\"targets\":{},\"packed_bytes\":{},\"predictions\":{},\"loss\":{},\"compact_gradient_rows\":{},\"coalesced_gradient_rows\":{},\"updated_master_rows\":{},\"first_moment_rows\":{},\"second_moment_rows\":{},\"cpu_reference\":{{\"backend\":\"cpu\",\"predictions\":{},\"loss\":{},\"compact_gradient_rows\":{},\"coalesced_gradient_rows\":{},\"updated_master_rows\":{},\"first_moment_rows\":{},\"second_moment_rows\":{}}},\"scalar_reference\":{{\"backend\":\"scalar\",\"predictions\":{},\"loss\":{},\"compact_gradient_rows\":{},\"coalesced_gradient_rows\":{},\"updated_master_rows\":{},\"first_moment_rows\":{},\"second_moment_rows\":{}}}",
        json_f32(&MASTER),
        json_i32(&ROUTES),
        json_f32(&ACTIVATIONS),
        json_f32(&TARGETS),
        json_u8(&observation.packed_bytes),
        json_f32(&observation.metal.predictions),
        observation.metal.loss,
        json_rows(&observation.metal.compact_gradients, 4),
        json_rows(&observation.coalesced_gradients, 4),
        json_rows(&observation.metal_update.masters, 4),
        json_rows(&observation.metal_update.first_moments, 4),
        json_rows(&observation.metal_update.second_moments, 4),
        json_f32(&observation.cpu.predictions),
        observation.cpu.loss,
        json_rows(&observation.cpu.compact_gradients, 4),
        json_rows(&observation.cpu_coalesced_gradients, 4),
        json_rows(&observation.cpu_update.masters, 4),
        json_rows(&observation.cpu_update.first_moments, 4),
        json_rows(&observation.cpu_update.second_moments, 4),
        json_f32(&observation.reference.predictions),
        observation.reference.loss,
        json_rows(&observation.reference.compact_gradients, 4),
        json_rows(&observation.reference.coalesced_gradients, 4),
        json_rows(&observation.reference.update.masters, 4),
        json_rows(&observation.reference.update.first_moments, 4),
        json_rows(&observation.reference.update.second_moments, 4),
    )
    .expect("writing into a String is infallible");
    record.push('}');
    record
}

fn codec_name(codec: Codec) -> &'static str {
    match codec {
        Codec::Bf8E5M2 => "Bf8E5M2",
        Codec::Bf4E2M1 => "Bf4E2M1",
        _ => panic!("fixture only uses BF8 E5M2 and BF4 E2M1"),
    }
}

fn json_rows(values: &[f32], row_len: usize) -> String {
    let rows = values
        .chunks_exact(row_len)
        .map(json_f32)
        .collect::<Vec<_>>();
    format!("[{}]", rows.join(","))
}

fn json_f32(values: &[f32]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| format!("{value:?}"))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn json_i32(values: &[i32]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(i32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn json_u8(values: &[u8]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(u8::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}
