//! One door type serves ONNX in with CPU out and ONNX in with MIL out: the same
//! generic function runs both, the MIL instance enters the writing stages in
//! pipeline order, and the walkthrough prints the transitions of each.

#![cfg(all(feature = "onnx-door", feature = "mil-writer"))]
// every expect below runs against a model this file builds itself; a failure
// there IS the test failing, not a case to recover.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod door_support;
#[path = "door_support/walkthrough.rs"]
mod walkthrough;
#[path = "door_support/walkthrough_mil.rs"]
mod walkthrough_mil;

use std::fmt::Debug;

use door_support::{mlp_batch, two_layer_mlp_bytes};
use omega::door::{Door, DoorStage, MilSink, NoSink, StageKind, Writer};
use omega::onnx_door::{DoorConfig, OnnxReader};
use proxima_mil::writer::PACKAGE_PATHS;
use proxima_onnx::lower::lower_graph;
use proxima_onnx::pipe::parse_complete;
use proxima_primitives::pipe::Pipe;
use proxima_tensor::Lowered;

const MLP_SOFTMAX: [f32; 4] = [0.599_888_4, 0.400_111_6, 0.434_749_25, 0.565_250_74];
const TOLERANCE: f32 = 1e-3;

fn door_of<W: Writer>(writer: W) -> Door<OnnxReader, W> {
    let settings = DoorConfig::default()
        .settings()
        .expect("the default config resolves");
    Door::new(OnnxReader::new(), writer, two_layer_mlp_bytes(), settings)
}

fn lowered_mlp() -> Lowered {
    let bytes = two_layer_mlp_bytes();
    let model = parse_complete(&bytes).expect("the mlp parses");
    let graph = model.graph.expect("the mlp carries a graph");
    lower_graph(&graph).expect("the mlp lowers")
}

async fn run_and_write<W>(door: &Door<OnnxReader, W>) -> (Vec<f32>, W::Package)
where
    W: Writer + Debug + 'static,
{
    let output = door
        .call(mlp_batch())
        .await
        .expect("the mlp runs on the cpu");
    let (values, _) = output.get("y").expect("y is a declared output");
    let package = door.write(lowered_mlp()).expect("the mlp writes");
    (values.to_vec(), package)
}

fn count(lines: &[String], needle: &str) -> usize {
    lines.iter().filter(|line| line.contains(needle)).count()
}

#[proxima::test]
async fn door_shared_same_door_type_runs_onnx_in_cpu_out_and_onnx_in_mil_out() {
    let (cpu_values, nothing) = run_and_write(&door_of(NoSink)).await;
    let (mil_values, package) = run_and_write(&door_of(MilSink)).await;

    assert_eq!(nothing, ());
    assert_eq!(cpu_values, mil_values);
    assert!(
        cpu_values
            .iter()
            .zip(MLP_SOFTMAX)
            .all(|(actual, expected)| (actual - expected).abs() < TOLERANCE)
    );
    let paths: Vec<&str> = package.files().iter().map(|(path, _)| *path).collect();
    assert_eq!(paths, PACKAGE_PATHS);
    assert!(package.files().iter().all(|(_, bytes)| !bytes.is_empty()));
}

#[test]
fn door_shared_mil_instance_enters_parse_validate_lower_emit_package_verify_in_order() {
    let door = door_of(MilSink);
    let bytes = two_layer_mlp_bytes();
    let batch = mlp_batch();
    let mut kinds = Vec::new();

    let mut stage = DoorStage::begin(door.reader(), &bytes).expect("the mlp parses");
    kinds.push(stage.kind());
    for _ in 0..2 {
        stage = stage.advance(&door, &batch).expect("reading succeeds");
        kinds.push(stage.kind());
    }
    let DoorStage::Lowered { lowered } = stage else {
        panic!("two advances from parsed leave the door lowered");
    };
    let mut stage = DoorStage::emit(door.writer(), lowered).expect("the mlp emits");
    kinds.push(stage.kind());
    for _ in 0..2 {
        stage = stage.advance(&door, &batch).expect("writing succeeds");
        kinds.push(stage.kind());
    }

    assert_eq!(
        kinds,
        [
            StageKind::Parse,
            StageKind::Validate,
            StageKind::Lower,
            StageKind::Emit,
            StageKind::Package,
            StageKind::Verify
        ]
    );
}

#[test]
fn door_shared_walkthrough_lines_are_10_for_the_onnx_instance_and_16_for_the_mil_instance() {
    let onnx = walkthrough::onnx_lines().expect("the onnx walkthrough runs");
    let mil = walkthrough_mil::mil_lines().expect("the mil walkthrough runs");

    assert_eq!(onnx.len(), 10);
    assert_eq!(
        (
            count(&onnx, "outcome=advanced"),
            count(&onnx, "outcome=failed")
        ),
        (5, 5)
    );
    assert_eq!(mil.len(), 16);
    assert_eq!(
        (
            count(&mil, "outcome=advanced"),
            count(&mil, "outcome=failed")
        ),
        (8, 8)
    );
    let failed_stages: Vec<&str> = StageKind::ALL.iter().map(|kind| kind.name()).collect();
    assert!(
        failed_stages
            .iter()
            .all(|stage| count(&mil, &format!("stage={stage} outcome=failed")) == 1)
    );
    assert_eq!(count(&onnx, "stage=emit"), 0);
    assert_eq!(count(&onnx, "stage=package"), 0);
    assert_eq!(count(&onnx, "stage=verify"), 0);
}
