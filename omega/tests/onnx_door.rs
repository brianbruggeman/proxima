//! The ONNX door's contract on synthetic graphs: a two-layer MLP with
//! hand-computed softmax outputs, one failing graph per stage, the stage
//! order, the pipe shape and its composition, and the config surface.
//!
//! Models are built as `proxima_onnx` message structs and encoded with
//! `write_model_proto`, so the door parses real wire bytes.

#![cfg(feature = "onnx-door")]
// every expect below runs against a model this file builds itself; a failure
// there IS the test failing, not a case to recover.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;

use omega::backend::{BackendError, Engine, GpuDriver};
use omega::onnx_door::{
    DoorBatch, DoorConfig, DoorError, DoorOutput, DoorStage, OnnxDoor, StageKind, ValidationError,
};
use proxima_onnx::messages::{
    AttributeProto, Dimension, DimensionValue, GraphProto, ModelProto, NodeProto, TensorProto,
    TensorShapeProto, TypeProto, TypeProtoTensor, TypeValue, ValueInfoProto,
};
use proxima_onnx::{LowerError, write_model_proto};
use proxima_primitives::pipe::{Pipe, PipeExt};
use proxima_tensor::NumericPolicy;

const MLP_INPUT: [f32; 6] = [1.0, 0.5, -1.0, 0.0, 2.0, 1.0];
const MLP_SOFTMAX: [f32; 4] = [0.599_888_4, 0.400_111_6, 0.434_749_25, 0.565_250_74];
const TOLERANCE: f32 = 1e-3;

fn float_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn fixed(extent: i64) -> Dimension<'static> {
    Dimension {
        value: Some(DimensionValue::Value(extent)),
        denotation: "",
    }
}

fn symbolic(name: &'static str) -> Dimension<'static> {
    Dimension {
        value: Some(DimensionValue::Param(name)),
        denotation: "",
    }
}

fn float_input(name: &'static str, dims: Vec<Dimension<'static>>) -> ValueInfoProto<'static> {
    let tensor = TypeProtoTensor {
        elem_type: 1,
        shape: Some(TensorShapeProto { dim: dims }),
    };
    ValueInfoProto {
        name,
        r#type: Some(TypeProto {
            value: Some(TypeValue::Tensor(tensor)),
            denotation: "",
        }),
        doc_string: "",
    }
}

fn untyped_output(name: &'static str) -> ValueInfoProto<'static> {
    ValueInfoProto {
        name,
        ..ValueInfoProto::default()
    }
}

fn weights<'a>(name: &'a str, dims: &[i64], raw: &'a [u8]) -> TensorProto<'a> {
    TensorProto {
        dims: dims.to_vec(),
        data_type: 1,
        name,
        raw_data: Some(raw),
        ..TensorProto::default()
    }
}

fn node<'a>(op_type: &'a str, inputs: &[&'a str], output: &'a str) -> NodeProto<'a> {
    NodeProto {
        input: inputs.to_vec(),
        output: vec![output],
        name: output,
        op_type,
        ..NodeProto::default()
    }
}

fn encode(graph: GraphProto<'_>) -> Vec<u8> {
    write_model_proto(&ModelProto {
        ir_version: 8,
        graph: Some(graph),
        ..ModelProto::default()
    })
}

fn two_layer_mlp_bytes() -> Vec<u8> {
    let w1 = float_bytes(&[
        0.1, 0.2, -0.1, 0.05, 0.3, -0.2, 0.4, 0.1, -0.5, 0.1, 0.2, -0.3,
    ]);
    let b1 = float_bytes(&[0.1, -0.1, 0.05, 0.0]);
    let w2 = float_bytes(&[0.2, -0.3, 0.1, 0.4, -0.2, 0.05, 0.3, 0.1]);
    let b2 = float_bytes(&[0.0, 0.1]);
    let softmax = NodeProto {
        attribute: vec![AttributeProto {
            name: "axis",
            type_raw: 2,
            i: 1,
            ..AttributeProto::default()
        }],
        ..node("Softmax", &["logits"], "y")
    };
    encode(GraphProto {
        node: vec![
            node("Gemm", &["x", "W1", "b1"], "h"),
            node("Relu", &["h"], "hr"),
            node("Gemm", &["hr", "W2", "b2"], "logits"),
            softmax,
        ],
        name: "mlp",
        initializer: vec![
            weights("W1", &[3, 4], &w1),
            weights("b1", &[4], &b1),
            weights("W2", &[4, 2], &w2),
            weights("b2", &[2], &b2),
        ],
        input: vec![float_input("x", vec![fixed(2), fixed(3)])],
        output: vec![untyped_output("y")],
        ..GraphProto::default()
    })
}

fn relu_over_symbolic_batch_bytes() -> Vec<u8> {
    encode(GraphProto {
        node: vec![node("Relu", &["x"], "y")],
        name: "relu",
        input: vec![float_input("x", vec![symbolic("batch_size"), fixed(3)])],
        output: vec![untyped_output("y")],
        ..GraphProto::default()
    })
}

fn mlp_batch() -> DoorBatch {
    DoorBatch {
        pins: BTreeMap::new(),
        inputs: vec![("x".to_owned(), MLP_INPUT.to_vec())],
    }
}

fn cpu_door(bytes: Vec<u8>) -> OnnxDoor {
    OnnxDoor::from_bytes(bytes, &DoorConfig::default()).expect("the default config resolves")
}

fn within(actual: &[f32], expected: &[f32], tolerance: f32) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(left, right)| (left - right).abs() < tolerance)
}

fn failure(result: Result<DoorOutput, DoorError>) -> DoorError {
    result.expect_err("the call must fail")
}

#[proxima::test]
async fn onnx_door_contract_mlp_matches_hand_computed_softmax_values() {
    let door = cpu_door(two_layer_mlp_bytes());

    let output = door.call(mlp_batch()).await.expect("the mlp runs on cpu");

    let (values, shape) = output.get("y").expect("y is a declared graph output");
    assert_eq!(shape, &[2, 2]);
    assert!(
        within(values, &MLP_SOFTMAX, TOLERANCE),
        "{values:?} vs {MLP_SOFTMAX:?}"
    );
}

#[proxima::test]
async fn onnx_door_contract_mlp_control_altered_expected_value_fails() {
    let door = cpu_door(two_layer_mlp_bytes());
    let mut altered = MLP_SOFTMAX;
    altered[0] += 0.1;

    let output = door.call(mlp_batch()).await.expect("the mlp runs on cpu");

    let (values, _) = output.get("y").expect("y is a declared graph output");
    assert!(!within(values, &altered, TOLERANCE));
}

#[proxima::test]
async fn onnx_door_contract_parse_failure_reports_stage_parse() {
    let bytes = two_layer_mlp_bytes();
    let truncated = bytes[..bytes.len() - 3].to_vec();
    let door = cpu_door(truncated);

    let error = failure(door.call(mlp_batch()).await);

    assert_eq!(error.stage(), StageKind::Parse);
    assert!(matches!(error, DoorError::Parse(_)), "{error:?}");
}

#[proxima::test]
async fn onnx_door_contract_validate_failure_lists_every_unlowerable_op() {
    let bytes = encode(GraphProto {
        node: vec![
            node("Foo", &["x"], "a"),
            node("Bar", &["a"], "b"),
            node("Foo", &["b"], "c"),
            node("Bar", &["c"], "d"),
            node("Foo", &["d"], "e"),
            node("Bar", &["e"], "y"),
        ],
        name: "unlowerable",
        input: vec![float_input("x", vec![fixed(2), fixed(3)])],
        output: vec![untyped_output("y")],
        ..GraphProto::default()
    });
    let door = cpu_door(bytes);

    let error = failure(door.call(mlp_batch()).await);

    assert_eq!(error.stage(), StageKind::Validate);
    let DoorError::Validate(ValidationError::Rejected {
        unlowerable,
        unpinned,
    }) = error
    else {
        panic!("expected a rejected graph");
    };
    assert_eq!(
        unlowerable,
        vec![("Bar".to_owned(), 3), ("Foo".to_owned(), 3)]
    );
    assert!(unpinned.is_empty());
}

#[proxima::test]
async fn onnx_door_contract_unpinned_symbolic_dimension_reports_stage_validate_with_the_dimension_name()
 {
    let door = cpu_door(relu_over_symbolic_batch_bytes());
    let values = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
    let mut batch = DoorBatch {
        pins: BTreeMap::new(),
        inputs: vec![("x".to_owned(), values)],
    };

    let error = failure(door.call(batch.clone()).await);
    batch.pins.insert("batch_size".to_owned(), 2);
    let pinned = door.call(batch).await.expect("the pinned batch runs");

    assert_eq!(error.stage(), StageKind::Validate);
    assert!(
        matches!(
            &error,
            DoorError::Validate(ValidationError::Rejected { unpinned, .. })
                if unpinned == &["batch_size".to_owned()]
        ),
        "{error:?}"
    );
    let (relu, shape) = pinned.get("y").expect("y is a declared graph output");
    assert_eq!(shape, &[2, 3]);
    assert_eq!(relu, &[1.0, 0.0, 3.0, 0.0, 5.0, 0.0]);
}

#[proxima::test]
async fn onnx_door_contract_lower_failure_reports_stage_lower() {
    let bytes = encode(GraphProto {
        node: vec![node("Add", &["x", "ghost"], "y")],
        name: "dangling",
        input: vec![float_input("x", vec![fixed(2), fixed(3)])],
        output: vec![untyped_output("y")],
        ..GraphProto::default()
    });
    let door = cpu_door(bytes);

    let error = failure(door.call(mlp_batch()).await);

    assert_eq!(error.stage(), StageKind::Lower);
    assert!(
        matches!(
            &error,
            DoorError::Lower(LowerError::UnknownValue { value, .. }) if value == "ghost"
        ),
        "{error:?}"
    );
}

#[cfg(not(feature = "wgpu-backend"))]
#[proxima::test]
async fn onnx_door_contract_plan_failure_reports_stage_plan() {
    let config = DoorConfig::builder()
        .engine("gpu")
        .gpu_driver("wgpu")
        .build();
    let door = OnnxDoor::from_bytes(two_layer_mlp_bytes(), &config).expect("the names resolve");

    let error = failure(door.call(mlp_batch()).await);

    assert_eq!(error.stage(), StageKind::Plan);
    assert!(
        matches!(
            &error,
            DoorError::Plan(BackendError::NotCompiled {
                backend: "wgpu",
                ..
            })
        ),
        "{error:?}"
    );
}

#[proxima::test]
async fn onnx_door_contract_execute_failure_reports_stage_execute() {
    let door = cpu_door(two_layer_mlp_bytes());
    let short_input = DoorBatch {
        pins: BTreeMap::new(),
        inputs: vec![("x".to_owned(), MLP_INPUT[..5].to_vec())],
    };

    let error = failure(door.call(short_input).await);

    assert_eq!(error.stage(), StageKind::Execute);
    assert!(matches!(error, DoorError::Execute(_)), "{error:?}");
}

#[test]
fn onnx_door_contract_stages_are_ordered_parse_validate_lower_plan_execute() {
    let bytes = two_layer_mlp_bytes();
    let door = cpu_door(bytes.clone());
    let batch = mlp_batch();
    let mut stage = DoorStage::start(&bytes).expect("the mlp parses");
    let mut kinds = vec![stage.kind()];

    for _ in 0..5 {
        stage = stage.advance(&door, &batch).expect("every stage succeeds");
        kinds.push(stage.kind());
    }

    assert_eq!(
        kinds,
        [
            StageKind::Parse,
            StageKind::Validate,
            StageKind::Lower,
            StageKind::Plan,
            StageKind::Execute,
            StageKind::Plan
        ]
    );
    assert!(kinds[..5].windows(2).all(|pair| pair[0] < pair[1]));
}

fn assert_door_pipe<P>(_pipe: &P)
where
    P: Pipe<In = DoorBatch, Out = DoorOutput, Err = DoorError>,
{
}

#[test]
fn onnx_door_contract_door_is_a_pipe_with_in_door_batch_out_door_output_err_door_error() {
    let door = cpu_door(two_layer_mlp_bytes());

    assert_door_pipe(&door);
}

struct PackInput;

impl Pipe for PackInput {
    type In = Vec<f32>;
    type Out = DoorBatch;
    type Err = DoorError;

    fn call(&self, input: Vec<f32>) -> impl Future<Output = Result<DoorBatch, DoorError>> {
        async move {
            Ok(DoorBatch {
                pins: BTreeMap::new(),
                inputs: vec![("x".to_owned(), input)],
            })
        }
    }
}

#[proxima::test]
async fn onnx_door_contract_door_composes_after_a_pipe_with_and_then() {
    let chain = PackInput.and_then(cpu_door(two_layer_mlp_bytes()));

    let output = chain
        .call(MLP_INPUT.to_vec())
        .await
        .expect("the chain runs the mlp");

    let (values, _) = output.get("y").expect("y is a declared graph output");
    assert!(within(values, &MLP_SOFTMAX, TOLERANCE));
}

#[test]
fn onnx_door_contract_config_builder_equals_conflaguration_loader() {
    let names = [
        "OMEGA_ONNX_DOOR_ENGINE",
        "OMEGA_ONNX_DOOR_GPU_DRIVER",
        "OMEGA_ONNX_DOOR_NUMERIC_POLICY",
    ];
    let unset: Vec<(&str, Option<&str>)> = names.iter().map(|name| (*name, None)).collect();
    let set = vec![
        (names[0], Some("gpu")),
        (names[1], Some("metal")),
        (names[2], Some("llama_relaxed")),
    ];
    let built = DoorConfig::builder()
        .engine("gpu")
        .gpu_driver("metal")
        .numeric_policy("llama_relaxed")
        .build();

    let defaults = temp_env::with_vars(unset, DoorConfig::from_env).expect("empty env loads");
    let loaded = temp_env::with_vars(set, DoorConfig::from_env).expect("set env loads");
    let tweaked = DoorConfig::builder_from(&loaded)
        .numeric_policy("fast")
        .build();

    assert_eq!(defaults, DoorConfig::builder().build());
    assert_eq!(defaults, DoorConfig::default());
    assert_eq!(loaded, built);
    assert_eq!(loaded.resolve_engine().expect("gpu"), Engine::Gpu);
    assert_eq!(
        loaded.resolve_gpu_driver().expect("metal"),
        Some(GpuDriver::Metal)
    );
    assert_eq!(
        loaded.resolve_numeric_policy().expect("relaxed"),
        NumericPolicy::llama_relaxed()
    );
    assert_eq!(tweaked.engine, loaded.engine);
    assert_eq!(tweaked.numeric_policy, "fast");
}
