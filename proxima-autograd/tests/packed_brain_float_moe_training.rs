use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use pilot_json as serde_json;
use proxima_autograd::adjoint::differentiate;
use proxima_autograd::optimizer::{AdamConfig, AdamOperands, adam_step, step_input};
use proxima_autograd::sparse::dedupe_and_sum_rows;
use proxima_primitives::Codec;
use proxima_tensor::DType;
use proxima_tensor::cpu::{
    ExpertEntry, ExpertSource, QuantizedBlock, evaluate_named,
    evaluate_quantized_named_with_scratch_and_experts,
};
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::op::{self, Extent, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp};
use proxima_tensor::spec::gathered_expert_product;

const MASTER: [f32; 12] = [1.1, 0.0, 0.0, 1.1, 2.2, 0.0, -1.1, 0.0, 3.3, 4.0, 5.0, 6.0];
const ROUTES: [f32; 3] = [1.0, 1.0, 0.0];
const ACTIVATIONS: [f32; 6] = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
const TARGETS: [f32; 6] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
static TELEMETRY_LOCK: Mutex<()> = Mutex::new(());

struct TrainingGraph {
    program: Vec<Op>,
    expert_stack: NodeId,
    predictions: NodeId,
    loss: NodeId,
}

struct DispatchRecord {
    position: u64,
    expert_index: u64,
    codec: String,
    packed_bytes: String,
}

struct StepResult {
    codec: Codec,
    packed_bytes: Vec<u8>,
    predictions: Vec<f32>,
    loss: f32,
    gradients: Vec<f32>,
    masters: Vec<f32>,
    moment_one: Vec<f32>,
    moment_two: Vec<f32>,
    dispatch_records: Vec<DispatchRecord>,
}

fn training_graph() -> TrainingGraph {
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
    TrainingGraph {
        program,
        expert_stack,
        predictions,
        loss,
    }
}

fn encode_master(codec: Codec) -> Vec<u8> {
    match codec {
        Codec::Bf8E5M2 => MASTER
            .iter()
            .map(|value| proxima_gguf::quant::bf8_e5m2::encode(*value))
            .collect(),
        Codec::Bf4E2M1 => MASTER
            .chunks_exact(2)
            .map(|pair| {
                proxima_gguf::quant::bf4_e2m1::pack_pair(pair[0], pair[1])
                    .expect("finite master pair encodes")
            })
            .collect(),
        _ => unreachable!("fixture only exercises the two admitted brain-float codecs"),
    }
}

fn master_entry_bytes(codec: Codec, bytes: &[u8]) -> (usize, [ExpertEntry<'_>; 3]) {
    let bytes_per_expert = match codec {
        Codec::Bf8E5M2 => 4,
        Codec::Bf4E2M1 => 2,
        _ => unreachable!("fixture only exercises the two admitted brain-float codecs"),
    };
    let entries = std::array::from_fn(|expert_index| {
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
    });
    (bytes_per_expert, entries)
}

fn run_adam_row(
    parameters: &[f32],
    gradients: &[f32],
    moment_one_values: &[f32],
    moment_two_values: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut program = Vec::new();
    let parameter = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: Some("parameter".into()),
        },
    );
    let gradient = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: Some("gradient".into()),
        },
    );
    let moment_one = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: Some("moment_one".into()),
        },
    );
    let moment_two = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: Some("moment_two".into()),
        },
    );
    let step = step_input(&mut program, "step");
    let (updated_parameter, updated_moment_one, updated_moment_two) = adam_step(
        &mut program,
        &AdamConfig::default(),
        1,
        AdamOperands {
            param: parameter,
            grad: gradient,
            m: moment_one,
            v: moment_two,
        },
        step,
    );
    let evaluated = evaluate_named(
        &program,
        &[],
        &[
            ("parameter", parameters),
            ("gradient", gradients),
            ("moment_one", moment_one_values),
            ("moment_two", moment_two_values),
            ("step", &[1.0]),
        ],
        &[updated_parameter, updated_moment_one, updated_moment_two],
    )
    .expect("selected expert Adam row evaluates");
    (
        evaluated
            .get(updated_parameter)
            .expect("updated parameter row")
            .0
            .to_vec(),
        evaluated
            .get(updated_moment_one)
            .expect("updated first moment row")
            .0
            .to_vec(),
        evaluated
            .get(updated_moment_two)
            .expect("updated second moment row")
            .0
            .to_vec(),
    )
}

fn run_training_step(
    codec: Codec,
    pipe: &proxima_telemetry::pipes::InMemoryPipe,
    recorder: &proxima_telemetry::recorder::Recorder,
) -> StepResult {
    let log_start = pipe.logs().len();
    let packed_bytes = encode_master(codec);
    let (_, entries) = master_entry_bytes(codec, &packed_bytes);
    let graph = training_graph();
    let differentiated =
        differentiate(&graph.program, graph.loss).expect("routed quadratic loss differentiates");
    let gathered = differentiated
        .gathered_gradients_of(graph.expert_stack)
        .next()
        .expect("expert table gradient remains compact by route");
    let named = [
        (
            "expert_stack",
            QuantizedBlock::Packed {
                codec,
                bytes: &packed_bytes,
            },
        ),
        ("route", QuantizedBlock::Float32(&ROUTES)),
        ("activation", QuantizedBlock::Float32(&ACTIVATIONS)),
        ("target", QuantizedBlock::Float32(&TARGETS)),
    ];
    let mut expert_sources = BTreeMap::new();
    expert_sources.insert(graph.expert_stack, ExpertSource::new(&entries));
    let evaluated = evaluate_quantized_named_with_scratch_and_experts(
        &differentiated.program,
        &[],
        &named,
        &[graph.loss, graph.predictions, gathered.values],
        &mut Vec::new(),
        &mut None,
        Some(&expert_sources),
    )
    .expect("packed computed-gather training forward and adjoint evaluate");
    recorder.drain();
    let loss = evaluated.get(graph.loss).expect("loss payload present").0[0];
    let predictions = evaluated
        .get(graph.predictions)
        .expect("prediction payload present")
        .0
        .to_vec();
    let compact_gradient = evaluated
        .get(gathered.values)
        .expect("compact gathered-gradient payload present")
        .0;
    let (selected_experts, gathered_rows) = dedupe_and_sum_rows(&ROUTES, compact_gradient, 4)
        .expect("three gathered rows of four values coalesce");

    let mut gradients = vec![0.0; MASTER.len()];
    for (selected_row, expert_index) in selected_experts.iter().copied().enumerate() {
        let gathered_row = &gathered_rows[selected_row * 4..(selected_row + 1) * 4];
        let expert_gradient =
            &mut gradients[expert_index as usize * 4..(expert_index as usize + 1) * 4];
        for output_index in 0..2 {
            for input_index in 0..2 {
                expert_gradient[output_index * 2 + input_index] =
                    gathered_row[input_index * 2 + output_index];
            }
        }
    }

    let mut masters = MASTER;
    let mut moment_one = [0.0; 12];
    let mut moment_two = [0.0; 12];
    for expert_index in selected_experts {
        let start = expert_index as usize * 4;
        let (updated_master, updated_first, updated_second) = run_adam_row(
            &masters[start..start + 4],
            &gradients[start..start + 4],
            &moment_one[start..start + 4],
            &moment_two[start..start + 4],
        );
        masters[start..start + 4].copy_from_slice(&updated_master);
        moment_one[start..start + 4].copy_from_slice(&updated_first);
        moment_two[start..start + 4].copy_from_slice(&updated_second);
    }
    let logs = pipe.logs();
    let dispatch_records = parse_dispatch_records(&logs[log_start..]);
    StepResult {
        codec,
        packed_bytes,
        predictions,
        loss,
        gradients,
        masters: masters.to_vec(),
        moment_one: moment_one.to_vec(),
        moment_two: moment_two.to_vec(),
        dispatch_records,
    }
}

fn telemetry_capture() -> (
    proxima_telemetry::pipes::InMemoryPipe,
    Arc<proxima_telemetry::recorder::Recorder>,
    MutexGuard<'static, ()>,
) {
    let state_guard = TELEMETRY_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let pipe = proxima_telemetry::pipes::InMemoryPipe::new();
    proxima_telemetry::emit::global::install(proxima_telemetry::emit::EnvFilter::parse("debug"));
    let recorder = Arc::new(
        proxima_telemetry::recorder::Recorder::builder()
            .pipe(pipe.clone())
            .core_count(1)
            .start()
            .expect("telemetry recorder starts"),
    );
    proxima_telemetry::export::set_default_recorder(Arc::clone(&recorder));
    (pipe, recorder, state_guard)
}

fn independent_reference(codec: Codec) -> (Vec<f32>, f32, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let decoded = match codec {
        Codec::Bf8E5M2 | Codec::Bf4E2M1 => {
            [1.0, 0.0, 0.0, 1.0, 2.0, 0.0, -1.0, 0.0, 3.0, 4.0, 5.0, 6.0]
        }
        _ => unreachable!("fixture only exercises the two admitted brain-float codecs"),
    };
    let mut predictions = vec![0.0; 6];
    let mut gradients = vec![0.0; MASTER.len()];
    let mut loss = 0.0f32;
    for token_index in 0..3 {
        let expert_index = ROUTES[token_index] as usize;
        let activation_row = &ACTIVATIONS[token_index * 2..token_index * 2 + 2];
        for output_index in 0..2 {
            let mut prediction = 0.0f32;
            for input_index in 0..2 {
                prediction += decoded[expert_index * 4 + output_index * 2 + input_index]
                    * activation_row[input_index];
            }
            predictions[token_index * 2 + output_index] = prediction;
            let difference = prediction - TARGETS[token_index * 2 + output_index];
            loss += 0.5 * difference * difference;
            for input_index in 0..2 {
                gradients[expert_index * 4 + output_index * 2 + input_index] +=
                    difference * activation_row[input_index];
            }
        }
    }
    let mut masters = MASTER.to_vec();
    let mut moment_one = vec![0.0; MASTER.len()];
    let mut moment_two = vec![0.0; MASTER.len()];
    let learning_rate = 0.001f32;
    let beta_one = 0.9f32;
    let beta_two = 0.999f32;
    let epsilon = 1e-8f32;
    for index in 0..MASTER.len() {
        let gradient = gradients[index];
        moment_one[index] = (1.0 - beta_one) * gradient;
        moment_two[index] = (1.0 - beta_two) * gradient * gradient;
        let corrected_one = moment_one[index] / (1.0 - beta_one);
        let corrected_two = moment_two[index] / (1.0 - beta_two);
        masters[index] -= learning_rate * corrected_one / (corrected_two.sqrt() + epsilon);
    }
    (
        predictions,
        loss,
        gradients,
        masters,
        moment_one,
        moment_two,
    )
}

fn assert_f32_values(actual: &[f32], expected: &[f32], payload_name: &str) {
    assert_eq!(actual.len(), expected.len(), "{payload_name} length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual == expected || (actual.is_nan() && expected.is_nan()),
            "{payload_name}[{index}] differs: actual={actual:?} expected={expected:?}"
        );
    }
}

#[test]
fn packed_gathered_step_updates_only_selected_master_rows() {
    let (pipe, recorder, _state_guard) = telemetry_capture();
    let bf8 = run_training_step(Codec::Bf8E5M2, &pipe, &recorder);
    let bf4 = run_training_step(Codec::Bf4E2M1, &pipe, &recorder);

    for result in [&bf8, &bf4] {
        assert_eq!(result.predictions, [2.0, -1.0, 2.0, -1.0, 0.0, 1.0]);
        assert_eq!(result.loss, 4.0);
        assert_eq!(
            result.gradients,
            [0.0, 0.0, 0.0, 1.0, 3.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(&result.masters[8..12], &[3.3, 4.0, 5.0, 6.0]);
        assert_eq!(&result.moment_one[8..12], &[0.0; 4]);
        assert_eq!(&result.moment_two[8..12], &[0.0; 4]);
    }
    let expected_bf8 = [
        (0, 1, "Bf8E5M2", "[64, 0, 188, 0]"),
        (1, 1, "Bf8E5M2", "[64, 0, 188, 0]"),
        (2, 0, "Bf8E5M2", "[60, 0, 0, 60]"),
    ];
    let expected_bf4 = [
        (0, 1, "Bf4E2M1", "[4, 10]"),
        (1, 1, "Bf4E2M1", "[4, 10]"),
        (2, 0, "Bf4E2M1", "[2, 32]"),
    ];
    assert_dispatch_sequence(&bf8.dispatch_records, &expected_bf8);
    assert_dispatch_sequence(&bf4.dispatch_records, &expected_bf4);
}

fn parse_dispatch_records(records: &[proxima_telemetry::log::LogRecord]) -> Vec<DispatchRecord> {
    records
        .iter()
        .filter(|record| {
            format!("{:?}", record.body).contains("selected packed brain-float expert span")
        })
        .map(|record| {
            let mut position = None;
            let mut expert_index = None;
            let mut codec = None;
            let mut packed_bytes = None;
            for tag in &record.attrs {
                if let proxima_telemetry::tag::Tag::Scalar { key, value } = tag {
                    match (*key, value) {
                        ("position", proxima_telemetry::tag::ScalarValue::U64(value)) => {
                            position = Some(*value);
                        }
                        ("expert_index", proxima_telemetry::tag::ScalarValue::U64(value)) => {
                            expert_index = Some(*value);
                        }
                        ("codec", proxima_telemetry::tag::ScalarValue::Bytes(value)) => {
                            codec = Some(String::from_utf8_lossy(value).into_owned());
                        }
                        (
                            "selected_packed_bytes",
                            proxima_telemetry::tag::ScalarValue::Bytes(value),
                        ) => packed_bytes = Some(String::from_utf8_lossy(value).into_owned()),
                        _ => {}
                    }
                }
            }
            DispatchRecord {
                position: position.expect("dispatch record has position"),
                expert_index: expert_index.expect("dispatch record has selected expert"),
                codec: codec.expect("dispatch record names codec"),
                packed_bytes: packed_bytes.expect("dispatch record retains selected bytes"),
            }
        })
        .collect()
}

fn assert_dispatch_sequence(actual: &[DispatchRecord], expected: &[(u64, u64, &str, &str); 3]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.position, expected.0);
        assert_eq!(actual.expert_index, expected.1);
        assert_eq!(actual.codec, expected.2);
        assert_eq!(actual.packed_bytes, expected.3);
    }
}

#[test]
fn packed_bf4_bf8_updates_match_fp32_ste_reference() {
    let (pipe, recorder, _state_guard) = telemetry_capture();
    for codec in [Codec::Bf8E5M2, Codec::Bf4E2M1] {
        let actual = run_training_step(codec, &pipe, &recorder);
        let (predictions, loss, gradients, masters, moment_one, moment_two) =
            independent_reference(codec);
        assert_f32_values(&actual.predictions, &predictions, "predictions");
        assert_f32_values(&[actual.loss], &[loss], "loss");
        assert_f32_values(&actual.gradients, &gradients, "coalesced gradients");
        assert_f32_values(&actual.masters, &masters, "FP32 masters");
        assert_f32_values(&actual.moment_one, &moment_one, "first moments");
        assert_f32_values(&actual.moment_two, &moment_two, "second moments");
    }
}

#[test]
fn packed_training_payload_record_is_complete() {
    let (pipe, recorder, _state_guard) = telemetry_capture();
    let records = [
        training_record(
            run_training_step(Codec::Bf8E5M2, &pipe, &recorder),
            17,
            "cpu",
        ),
        training_record(
            run_training_step(Codec::Bf4E2M1, &pipe, &recorder),
            17,
            "cpu",
        ),
    ];
    assert_eq!(records.len(), 2);
    for record in &records {
        for key in [
            "codec",
            "backend",
            "seed",
            "step",
            "master_input",
            "routes",
            "activations",
            "targets",
            "packed_bytes",
            "selected_dispatch_spans",
            "predictions",
            "loss",
            "coalesced_gradients",
            "final_masters",
            "moment_one",
            "moment_two",
            "reference",
        ] {
            assert!(
                record.get(key).is_some_and(|value| !value.is_null()),
                "missing {key}"
            );
        }
        assert_eq!(record["seed"], 17);
        assert_eq!(record["backend"], "cpu");
        assert_eq!(record["step"], 1);
    }
    let json = serde_json::to_vec_pretty(&records).expect("training evidence serializes");
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../proxima-tensor/specs/packed_brain_float_moe_training/results/packed-training.json",
    );
    fs::create_dir_all(path.parent().expect("training artifact parent directory"))
        .expect("training artifact directory exists");
    fs::write(&path, &json).expect("training evidence artifact is written");
    let decoded: serde_json::Value =
        serde_json::from_slice(&fs::read(path).expect("training evidence can be read back"))
            .expect("training evidence remains valid JSON");
    assert_eq!(decoded.as_array().map(Vec::len), Some(2));
}

fn training_record(result: StepResult, seed: u64, backend: &str) -> serde_json::Value {
    let reference = independent_reference(result.codec);
    let selected_dispatch_spans = result
        .dispatch_records
        .iter()
        .map(|record| {
            serde_json::json!({
                "position": record.position,
                "expert_index": record.expert_index,
                "codec": record.codec,
                "packed_bytes": record.packed_bytes,
            })
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "codec": format!("{:?}", result.codec),
        "backend": backend,
        "seed": seed,
        "step": 1,
        "master_input": MASTER,
        "routes": ROUTES,
        "activations": ACTIVATIONS,
        "targets": TARGETS,
        "packed_bytes": result.packed_bytes,
        "selected_dispatch_spans": selected_dispatch_spans,
        "predictions": result.predictions,
        "loss": result.loss,
        "coalesced_gradients": result.gradients,
        "final_masters": result.masters,
        "moment_one": result.moment_one,
        "moment_two": result.moment_two,
        "reference": {
            "predictions": reference.0,
            "loss": reference.1,
            "coalesced_gradients": reference.2,
            "final_masters": reference.3,
            "moment_one": reference.4,
            "moment_two": reference.5,
        },
    })
}
