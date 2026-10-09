use std::collections::BTreeMap;
use std::error::Error;
use std::io::{self, Write};
use std::process::Command;
use std::time::Instant;

use pilot_json as serde_json;
use proxima_autograd::adjoint::differentiate;
use proxima_autograd::low_precision::WeightFormat;
use proxima_autograd::optimizer::{AdamConfig, AdamOperands, adam_step, step_input};
use proxima_autograd::sparse::dedupe_and_sum_rows;
use proxima_primitives::Codec;
use proxima_tensor::cpu::{
    ExpertEntry, ExpertSource, QuantizedBlock, evaluate_named,
    evaluate_quantized_named_with_scratch_and_experts,
};
use proxima_tensor::dtype::DType;
use proxima_tensor::map::{self, AxisIndex, AxisTerm, IndexMap, IndexPattern};
use proxima_tensor::op::{self, Extent, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp};
use proxima_tensor::test_support::Lcg;

pub const TRAIN_INPUTS: [u32; 8] = [0, 1, 2, 3, 0, 1, 2, 3];
pub const TRAIN_TARGETS: [u32; 8] = [1, 2, 3, 0, 1, 2, 3, 0];
pub const HELD_OUT_INPUTS: [u32; 8] = [0, 0, 1, 1, 2, 2, 3, 3];
pub const HELD_OUT_TARGETS: [u32; 8] = [0, 1, 1, 2, 2, 3, 3, 0];

pub struct Model {
    pub differentiated_program: Vec<Op>,
    pub expert_weights: NodeId,
    pub logits: NodeId,
    pub loss: NodeId,
    pub gathered_gradient: NodeId,
}

fn projection(rank: u16, axes: &[u16]) -> IndexMap {
    IndexMap::Affine(map::projection(rank, axes))
}

fn append_elementwise(
    program: &mut Vec<Op>,
    body: ScalarOp,
    operands: Vec<(NodeId, IndexMap)>,
) -> NodeId {
    op::append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body,
            operands,
            name: None,
        },
    )
}

fn append_reduce(
    program: &mut Vec<Op>,
    body: ScalarOp,
    init: ReduceInit,
    operand: NodeId,
    in_map: IndexMap,
    out_map: IndexMap,
) -> NodeId {
    op::append(
        program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body,
            init,
            operand,
            in_map,
            out_map,
            keep: Keep::Reduce,
            name: None,
        }),
    )
}

pub fn build_model() -> Result<Model, Box<dyn Error>> {
    let mut program = Vec::new();
    let expert_weights = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(2), Extent::Static(4), Extent::Static(4)],
            name: Some("experts".into()),
        },
    );
    let expert_ids = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1)],
            name: Some("expert_ids".into()),
        },
    );
    let inputs = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1), Extent::Static(4)],
            name: Some("inputs".into()),
        },
    );
    let targets = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1), Extent::Static(4)],
            name: Some("targets".into()),
        },
    );

    let expert_gather = IndexMap::Computed {
        indices: expert_ids,
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
    let selected_expert = append_elementwise(
        &mut program,
        ScalarOp::Identity,
        vec![(expert_weights, expert_gather)],
    );
    let products = append_elementwise(
        &mut program,
        ScalarOp::Multiply,
        vec![
            (inputs, projection(3, &[0, 1])),
            (selected_expert, projection(3, &[0, 1, 2])),
        ],
    );
    let logits = append_reduce(
        &mut program,
        ScalarOp::Add,
        ReduceInit::Zero,
        products,
        projection(3, &[0, 1, 2]),
        projection(3, &[0, 2]),
    );
    let maximum = append_reduce(
        &mut program,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        logits,
        projection(2, &[0, 1]),
        projection(2, &[0]),
    );
    let centered = append_elementwise(
        &mut program,
        ScalarOp::Subtract,
        vec![
            (logits, projection(2, &[0, 1])),
            (maximum, projection(2, &[0])),
        ],
    );
    let exponentials = append_elementwise(
        &mut program,
        ScalarOp::Exponential,
        vec![(centered, projection(2, &[0, 1]))],
    );
    let exponential_sum = append_reduce(
        &mut program,
        ScalarOp::Add,
        ReduceInit::Zero,
        exponentials,
        projection(2, &[0, 1]),
        projection(2, &[0]),
    );
    let log_sum = append_elementwise(
        &mut program,
        ScalarOp::Logarithm,
        vec![(exponential_sum, projection(1, &[0]))],
    );
    let log_normalizer = append_elementwise(
        &mut program,
        ScalarOp::Add,
        vec![
            (maximum, projection(1, &[0])),
            (log_sum, projection(1, &[0])),
        ],
    );
    let target_products = append_elementwise(
        &mut program,
        ScalarOp::Multiply,
        vec![
            (logits, projection(2, &[0, 1])),
            (targets, projection(2, &[0, 1])),
        ],
    );
    let target_logits = append_reduce(
        &mut program,
        ScalarOp::Add,
        ReduceInit::Zero,
        target_products,
        projection(2, &[0, 1]),
        projection(2, &[0]),
    );
    let token_losses = append_elementwise(
        &mut program,
        ScalarOp::Subtract,
        vec![
            (log_normalizer, projection(1, &[0])),
            (target_logits, projection(1, &[0])),
        ],
    );
    let loss = append_reduce(
        &mut program,
        ScalarOp::Add,
        ReduceInit::Zero,
        token_losses,
        projection(1, &[0]),
        projection(1, &[]),
    );
    let differentiated = differentiate(&program, loss)?;
    let gathered_gradient = differentiated
        .gathered_gradients_of(expert_weights)
        .next()
        .ok_or_else(|| io::Error::other("expert matrix gather has no gathered gradient"))?;

    Ok(Model {
        differentiated_program: differentiated.program,
        expert_weights,
        logits,
        loss,
        gathered_gradient: gathered_gradient.values,
    })
}

fn one_hot(token_id: u32) -> [f32; 4] {
    let mut values = [0.0; 4];
    values[token_id as usize] = 1.0;
    values
}

pub fn evaluate_token(
    model: &Model,
    weight_values: &[f32],
    token_id: u32,
    target_id: u32,
) -> Result<(Vec<f32>, f32, Vec<f32>), Box<dyn Error>> {
    let input_values = one_hot(token_id);
    let target_values = one_hot(target_id);
    let route = [(token_id % 2) as f32];
    let evaluated = evaluate_named(
        &model.differentiated_program,
        &[],
        &[
            ("experts", weight_values),
            ("expert_ids", &route),
            ("inputs", &input_values),
            ("targets", &target_values),
        ],
        &[model.logits, model.loss, model.gathered_gradient],
    )?;
    let logits = evaluated
        .get(model.logits)
        .map(|(values, _)| values.to_vec())
        .ok_or_else(|| io::Error::other("token logits output missing"))?;
    let loss = evaluated
        .get(model.loss)
        .and_then(|(values, _)| values.first().copied())
        .ok_or_else(|| io::Error::other("token loss output missing"))?;
    let gradient = evaluated
        .get(model.gathered_gradient)
        .map(|(values, _)| values.to_vec())
        .ok_or_else(|| io::Error::other("gathered expert gradient missing"))?;
    Ok((logits, loss, gradient))
}

pub fn encode_packed_weights(
    master_weights: &[f32],
    format: WeightFormat,
) -> Result<Vec<u8>, Box<dyn Error>> {
    match format {
        WeightFormat::Bf8E5M2 => Ok(master_weights
            .iter()
            .map(|value| proxima_gguf::quant::bf8_e5m2::encode(*value))
            .collect()),
        WeightFormat::Bf4E2M1 => master_weights
            .chunks_exact(2)
            .map(|pair| {
                proxima_gguf::quant::bf4_e2m1::pack_pair(pair[0], pair[1]).map_err(Into::into)
            })
            .collect(),
        WeightFormat::Fp32 => Err(io::Error::other("FP32 weights do not use packed bytes").into()),
    }
}

fn decode_weights(master_weights: &[f32], packed_bytes: &[u8], format: WeightFormat) -> Vec<f32> {
    match format {
        WeightFormat::Fp32 => master_weights.to_vec(),
        WeightFormat::Bf8E5M2 => packed_bytes
            .iter()
            .map(|value| proxima_gguf::quant::bf8_e5m2::decode(*value))
            .collect(),
        WeightFormat::Bf4E2M1 => packed_bytes
            .iter()
            .flat_map(|value| proxima_gguf::quant::bf4_e2m1::unpack_pair(*value))
            .collect(),
    }
}

pub fn scalar_token(
    master_weights: &[f32],
    packed_bytes: &[u8],
    format: WeightFormat,
    token_id: u32,
    target_id: u32,
) -> (Vec<f32>, f32, Vec<f32>) {
    let decoded = decode_weights(master_weights, packed_bytes, format);
    let expert_start = (token_id as usize % 2) * 16;
    let mut logits = vec![0.0; 4];
    for output_index in 0..4 {
        logits[output_index] = decoded[expert_start + token_id as usize * 4 + output_index];
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials: Vec<f32> = logits.iter().map(|value| (value - maximum).exp()).collect();
    let exponent_sum: f32 = exponentials.iter().sum();
    let probabilities: Vec<f32> = exponentials
        .iter()
        .map(|value| value / exponent_sum)
        .collect();
    let loss = maximum + exponent_sum.ln() - logits[target_id as usize];
    let mut compact_gradient = vec![0.0; 16];
    for output_index in 0..4 {
        let output_gradient = probabilities[output_index]
            - if output_index == target_id as usize {
                1.0
            } else {
                0.0
            };
        compact_gradient[token_id as usize * 4 + output_index] = output_gradient;
    }
    (logits, loss, compact_gradient)
}

pub fn scalar_adam_step(
    master_input: &[f32],
    first_moment_input: &[f32],
    second_moment_input: &[f32],
    coalesced_gradients: &[f32],
    step: u32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut updated_masters = master_input.to_vec();
    let mut updated_first = first_moment_input.to_vec();
    let mut updated_second = second_moment_input.to_vec();
    let beta_one = 0.9f32;
    let beta_two = 0.999f32;
    let first_correction = 1.0 - beta_one.powi(step as i32);
    let second_correction = 1.0 - beta_two.powi(step as i32);
    for index in 0..master_input.len() {
        let gradient = coalesced_gradients[index] / 8.0;
        updated_first[index] = beta_one * first_moment_input[index] + (1.0 - beta_one) * gradient;
        updated_second[index] =
            beta_two * second_moment_input[index] + (1.0 - beta_two) * gradient * gradient;
        let corrected_first = updated_first[index] / first_correction;
        let corrected_second = updated_second[index] / second_correction;
        updated_masters[index] -= 0.001 * corrected_first / (corrected_second.sqrt() + 1e-8);
    }
    (updated_masters, updated_first, updated_second)
}

pub fn expert_byte_spans(format: WeightFormat, format_name: &str) -> Vec<serde_json::Value> {
    if format == WeightFormat::Fp32 {
        return Vec::new();
    }
    let bytes_per_expert = match format {
        WeightFormat::Bf8E5M2 => 16,
        WeightFormat::Bf4E2M1 => 8,
        WeightFormat::Fp32 => 0,
    };
    (0..2)
        .map(|expert_id| {
            serde_json::json!({
                "expert_id": expert_id,
                "offset": expert_id * bytes_per_expert,
                "length": bytes_per_expert,
                "codec": format_name,
                "out_dim": 4,
                "in_dim": 4
            })
        })
        .collect()
}

pub fn evaluate_packed_token(
    model: &Model,
    packed_bytes: &[u8],
    format: WeightFormat,
    token_id: u32,
    target_id: u32,
    epoch: u64,
) -> Result<(Vec<f32>, f32, Vec<f32>), Box<dyn Error>> {
    let codec = match format {
        WeightFormat::Bf8E5M2 => Codec::Bf8E5M2,
        WeightFormat::Bf4E2M1 => Codec::Bf4E2M1,
        WeightFormat::Fp32 => {
            return Err(io::Error::other("packed evaluator requires BF8 or BF4").into());
        }
    };
    let bytes_per_expert = match codec {
        Codec::Bf8E5M2 => 16,
        Codec::Bf4E2M1 => 8,
        _ => unreachable!("codec selection is BF8 or BF4"),
    };
    let entries: [ExpertEntry<'_>; 2] = std::array::from_fn(|expert_id| {
        let byte_start = expert_id * bytes_per_expert;
        ExpertEntry {
            block: QuantizedBlock::Packed {
                codec,
                bytes: &packed_bytes[byte_start..byte_start + bytes_per_expert],
            },
            out_dim: 4,
            in_dim: 4,
            epoch,
        }
    });
    let mut expert_sources = BTreeMap::new();
    expert_sources.insert(model.expert_weights, ExpertSource::new(&entries));
    let input_values = one_hot(token_id);
    let target_values = one_hot(target_id);
    let route = [(token_id % 2) as f32];
    let named = [
        (
            "experts",
            QuantizedBlock::Packed {
                codec,
                bytes: packed_bytes,
            },
        ),
        ("expert_ids", QuantizedBlock::Float32(&route)),
        ("inputs", QuantizedBlock::Float32(&input_values)),
        ("targets", QuantizedBlock::Float32(&target_values)),
    ];
    let evaluated = evaluate_quantized_named_with_scratch_and_experts(
        &model.differentiated_program,
        &[],
        &named,
        &[model.logits, model.loss, model.gathered_gradient],
        &mut Vec::new(),
        &mut None,
        Some(&expert_sources),
    )?;
    let logits = evaluated
        .get(model.logits)
        .map(|(values, _)| values.to_vec())
        .ok_or_else(|| io::Error::other("packed token logits output missing"))?;
    let loss = evaluated
        .get(model.loss)
        .and_then(|(values, _)| values.first().copied())
        .ok_or_else(|| io::Error::other("packed token loss output missing"))?;
    let gradient = evaluated
        .get(model.gathered_gradient)
        .map(|(values, _)| values.to_vec())
        .ok_or_else(|| io::Error::other("packed gathered expert gradient missing"))?;
    Ok((logits, loss, gradient))
}

pub fn optimizer_program() -> (Vec<Op>, NodeId, NodeId, NodeId) {
    let mut program = Vec::new();
    let parameters = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(32)],
            name: Some("parameters".into()),
        },
    );
    let gradients = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(32)],
            name: Some("gradients".into()),
        },
    );
    let first_moment = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(32)],
            name: Some("first_moment".into()),
        },
    );
    let second_moment = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(32)],
            name: Some("second_moment".into()),
        },
    );
    let step = step_input(&mut program, "step");
    let (updated_parameters, updated_first, updated_second) = adam_step(
        &mut program,
        &AdamConfig::default(),
        1,
        AdamOperands {
            param: parameters,
            grad: gradients,
            m: first_moment,
            v: second_moment,
        },
        step,
    );
    (program, updated_parameters, updated_first, updated_second)
}

pub fn run_optimizer(
    optimizer: &[Op],
    output_parameters: NodeId,
    output_first: NodeId,
    output_second: NodeId,
    parameters: &[f32],
    gradients: &[f32],
    first_moment: &[f32],
    second_moment: &[f32],
    step: f32,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), Box<dyn Error>> {
    let step_values = [step];
    let evaluated = evaluate_named(
        optimizer,
        &[],
        &[
            ("parameters", parameters),
            ("gradients", gradients),
            ("first_moment", first_moment),
            ("second_moment", second_moment),
            ("step", &step_values),
        ],
        &[output_parameters, output_first, output_second],
    )?;
    let copy_output = |node| -> Result<Vec<f32>, io::Error> {
        evaluated
            .get(node)
            .map(|(values, _)| values.to_vec())
            .ok_or_else(|| io::Error::other("optimizer output missing"))
    };
    Ok((
        copy_output(output_parameters)?,
        copy_output(output_first)?,
        copy_output(output_second)?,
    ))
}

pub fn peak_rss_bytes() -> Result<u64, io::Error> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // getrusage initializes the complete output struct on success.
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if result != 0 {
        return Err(io::Error::last_os_error());
    }
    let usage = unsafe { usage.assume_init() };
    #[cfg(target_os = "macos")]
    let bytes = usage.ru_maxrss as u64;
    #[cfg(not(target_os = "macos"))]
    let bytes = (usage.ru_maxrss as u64).saturating_mul(1024);
    Ok(bytes)
}

pub fn run_arm(
    model: &Model,
    optimizer: (&[Op], NodeId, NodeId, NodeId),
    format: WeightFormat,
    format_name: &str,
    seed: u64,
    output: &mut impl Write,
) -> Result<(), Box<dyn Error>> {
    let mut lcg = Lcg(seed);
    let mut parameters: Vec<f32> = (0..32).map(|_| lcg.next_unit() * 0.5).collect();
    let initial_parameters = parameters.clone();
    let mut first_moment = vec![0.0_f32; 32];
    let mut second_moment = vec![0.0_f32; 32];
    let mut step_records = Vec::with_capacity(64);

    for step in 1..=64 {
        let started = Instant::now();
        let master_input = parameters.clone();
        let first_moment_input = first_moment.clone();
        let second_moment_input = second_moment.clone();
        let training_packed_bytes = match format {
            WeightFormat::Fp32 => Vec::new(),
            _ => encode_packed_weights(&master_input, format)?,
        };
        let mut losses = 0.0_f32;
        let mut train_logits = Vec::with_capacity(8);
        let mut train_token_losses = Vec::with_capacity(8);
        let mut route_ids = Vec::with_capacity(8);
        let mut gradient_ids = Vec::with_capacity(8);
        let mut gradient_values = Vec::with_capacity(8 * 16);
        for (token_id, target_id) in TRAIN_INPUTS.into_iter().zip(TRAIN_TARGETS) {
            let (logits, loss, gradient) = match format {
                WeightFormat::Fp32 => evaluate_token(model, &master_input, token_id, target_id)?,
                _ => evaluate_packed_token(
                    model,
                    &training_packed_bytes,
                    format,
                    token_id,
                    target_id,
                    step as u64,
                )?,
            };
            losses += loss;
            train_logits.push(logits);
            train_token_losses.push(loss);
            route_ids.push(token_id % 2);
            gradient_ids.push((token_id % 2) as f32);
            gradient_values.extend_from_slice(&gradient);
        }
        let training_loss = losses / 8.0;
        let (unique_ids, summed_gradients) =
            dedupe_and_sum_rows(&gradient_ids, &gradient_values, 16)?;
        if unique_ids != [0, 1] || summed_gradients.len() != 32 {
            return Err(io::Error::other("fixed router did not activate both experts").into());
        }
        let coalesced_gradients = summed_gradients.clone();
        let averaged_gradients: Vec<f32> = summed_gradients
            .into_iter()
            .map(|gradient| gradient / 8.0)
            .collect();
        let (optimizer_program, new_parameters, new_first, new_second) = optimizer;
        (parameters, first_moment, second_moment) = run_optimizer(
            optimizer_program,
            new_parameters,
            new_first,
            new_second,
            &parameters,
            &averaged_gradients,
            &first_moment,
            &second_moment,
            step as f32,
        )?;

        let held_out_packed_bytes = match format {
            WeightFormat::Fp32 => Vec::new(),
            _ => encode_packed_weights(&parameters, format)?,
        };
        let mut held_out_loss = 0.0_f32;
        let mut held_out_logits = Vec::with_capacity(8);
        let mut held_out_token_losses = Vec::with_capacity(8);
        for (token_id, target_id) in HELD_OUT_INPUTS.into_iter().zip(HELD_OUT_TARGETS) {
            let (logits, loss, _) = match format {
                WeightFormat::Fp32 => evaluate_token(model, &parameters, token_id, target_id)?,
                _ => evaluate_packed_token(
                    model,
                    &held_out_packed_bytes,
                    format,
                    token_id,
                    target_id,
                    step as u64,
                )?,
            };
            held_out_loss += loss;
            held_out_logits.push(logits);
            held_out_token_losses.push(loss);
        }
        held_out_loss /= 8.0;
        let held_out_route_ids: Vec<u32> = HELD_OUT_INPUTS
            .iter()
            .map(|token_id| token_id % 2)
            .collect();
        let elapsed_ns = started.elapsed().as_nanos().max(1);
        let tokens_per_second = 8_000_000_000.0 / elapsed_ns as f64;
        let non_finite_count = parameters
            .iter()
            .chain(&averaged_gradients)
            .chain(std::iter::once(&training_loss))
            .chain(std::iter::once(&held_out_loss))
            .filter(|value| !value.is_finite())
            .count();

        let mut scalar_train_logits = Vec::with_capacity(8);
        let mut scalar_train_losses = Vec::with_capacity(8);
        let mut scalar_compact_gradients = Vec::with_capacity(8);
        let mut scalar_coalesced_gradients = vec![0.0_f32; 32];
        for (token_id, target_id) in TRAIN_INPUTS.into_iter().zip(TRAIN_TARGETS) {
            let (logits, loss, compact_gradient) = scalar_token(
                &master_input,
                &training_packed_bytes,
                format,
                token_id,
                target_id,
            );
            let expert_id = token_id as usize % 2;
            for (element_index, value) in compact_gradient.iter().copied().enumerate() {
                scalar_coalesced_gradients[expert_id * 16 + element_index] += value;
            }
            scalar_train_logits.push(logits);
            scalar_train_losses.push(loss);
            scalar_compact_gradients.push(compact_gradient);
        }
        let (scalar_updated_masters, scalar_updated_first, scalar_updated_second) =
            scalar_adam_step(
                &master_input,
                &first_moment_input,
                &second_moment_input,
                &scalar_coalesced_gradients,
                step,
            );
        let scalar_held_out_bytes = match format {
            WeightFormat::Fp32 => Vec::new(),
            _ => encode_packed_weights(&scalar_updated_masters, format)?,
        };
        let mut scalar_held_out_logits = Vec::with_capacity(8);
        let mut scalar_held_out_losses = Vec::with_capacity(8);
        for (token_id, target_id) in HELD_OUT_INPUTS.into_iter().zip(HELD_OUT_TARGETS) {
            let (logits, loss, _) = scalar_token(
                &scalar_updated_masters,
                &scalar_held_out_bytes,
                format,
                token_id,
                target_id,
            );
            scalar_held_out_logits.push(logits);
            scalar_held_out_losses.push(loss);
        }
        let scalar_train_loss = scalar_train_losses.iter().sum::<f32>() / 8.0;
        let scalar_held_out_loss = scalar_held_out_losses.iter().sum::<f32>() / 8.0;
        let train_spans = expert_byte_spans(format, format_name);
        let held_out_spans = expert_byte_spans(format, format_name);
        step_records.push(serde_json::json!({
            "step": step,
            "backend": if format == WeightFormat::Fp32 { "cpu-fp32" } else { "cpu-packed" },
            "master_input": master_input,
            "first_moment_input": first_moment_input,
            "second_moment_input": second_moment_input,
            "train_packed_bytes": training_packed_bytes,
            "train_expert_byte_spans": train_spans,
            "held_out_packed_bytes": held_out_packed_bytes,
            "held_out_expert_byte_spans": held_out_spans,
            "input_ids": TRAIN_INPUTS,
            "expert_ids": route_ids,
            "target_ids": TRAIN_TARGETS,
            "logits": train_logits,
            "token_losses": train_token_losses,
            "train_loss": training_loss,
            "held_out_input_ids": HELD_OUT_INPUTS,
            "held_out_expert_ids": held_out_route_ids,
            "held_out_target_ids": HELD_OUT_TARGETS,
            "held_out_logits": held_out_logits,
            "held_out_token_losses": held_out_token_losses,
            "held_out_loss": held_out_loss,
            "compact_gradient_rows": gradient_values.chunks_exact(16).collect::<Vec<_>>(),
            "coalesced_gradient_rows": coalesced_gradients.chunks_exact(16).collect::<Vec<_>>(),
            "updated_master_rows": parameters.chunks_exact(16).collect::<Vec<_>>(),
            "updated_first_moment_rows": first_moment.chunks_exact(16).collect::<Vec<_>>(),
            "updated_second_moment_rows": second_moment.chunks_exact(16).collect::<Vec<_>>(),
            "non_finite_count": non_finite_count,
            "wall_time_ns": elapsed_ns,
            "tokens_per_second": tokens_per_second,
            "scalar_reference": {
                "logits": scalar_train_logits,
                "token_losses": scalar_train_losses,
                "train_loss": scalar_train_loss,
                "held_out_logits": scalar_held_out_logits,
                "held_out_token_losses": scalar_held_out_losses,
                "held_out_loss": scalar_held_out_loss,
                "compact_gradient_rows": scalar_compact_gradients,
                "coalesced_gradient_rows": scalar_coalesced_gradients.chunks_exact(16).collect::<Vec<_>>(),
                "updated_master_rows": scalar_updated_masters.chunks_exact(16).collect::<Vec<_>>(),
                "updated_first_moment_rows": scalar_updated_first.chunks_exact(16).collect::<Vec<_>>(),
                "updated_second_moment_rows": scalar_updated_second.chunks_exact(16).collect::<Vec<_>>()
            }
        }));
    }
    let report = serde_json::json!({
        "format": format_name,
        "seed": seed,
        "parameters": 32,
        "hyperparameter_search_trials": 0,
        "checkpoint_step": 64,
        "initial_parameters": initial_parameters,
        "steps": step_records,
        "final_parameters": parameters,
        "peak_rss_bytes": peak_rss_bytes()?
    });
    serde_json::to_writer(output, &report)?;
    Ok(())
}

pub fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments.len() == 3 && arguments[0] == "--arm-worker" {
        let format_name = arguments[1].as_str();
        let format = match format_name {
            "bf8_e5m2" => WeightFormat::Bf8E5M2,
            "bf4_e2m1" => WeightFormat::Bf4E2M1,
            "fp32" => WeightFormat::Fp32,
            _ => return Err(io::Error::other("unknown arm-worker format").into()),
        };
        let seed = arguments[2].parse::<u64>()?;
        let model = build_model()?;
        let optimizer = optimizer_program();
        let mut output = io::BufWriter::new(io::stdout().lock());
        run_arm(
            &model,
            (&optimizer.0, optimizer.1, optimizer.2, optimizer.3),
            format,
            format_name,
            seed,
            &mut output,
        )?;
        output.flush()?;
        return Ok(());
    }
    if arguments
        != [
            "--fixture",
            "tiny",
            "--steps",
            "64",
            "--seeds",
            "3",
            "--report",
            "json",
        ]
    {
        return Err(
            io::Error::other("expected --fixture tiny --steps 64 --seeds 3 --report json").into(),
        );
    }
    let mut output = io::BufWriter::new(io::stdout().lock());
    write!(
        output,
        "{{\"fixture\":\"tiny\",\"scale_cost\":\"unmeasured\",\"training_steps\":64,\"optimizer\":{{\"name\":\"adam\",\"learning_rate\":0.001,\"beta1\":0.9,\"beta2\":0.999,\"epsilon\":0.00000001}},\"arms\":["
    )?;
    let arms = ["bf8_e5m2", "bf4_e2m1", "fp32"];
    let seeds = [17_u64, 29, 43];
    let mut first_arm = true;
    for format_name in arms {
        for seed in seeds {
            if !first_arm {
                write!(output, ",")?;
            }
            let seed_argument = seed.to_string();
            let child = Command::new(std::env::current_exe()?)
                .args(["--arm-worker", format_name, &seed_argument])
                .output()?;
            if !child.status.success() {
                return Err(
                    io::Error::other(String::from_utf8_lossy(&child.stderr).into_owned()).into(),
                );
            }
            output.write_all(&child.stdout)?;
            first_arm = false;
        }
    }
    writeln!(output, "]}}")?;
    output.flush()?;
    Ok(())
}
