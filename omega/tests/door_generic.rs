//! The one door is generic over the format: its stage enum is defined once,
//! its source names no format outside a reader, `OnnxDoor` is an instance of
//! it, and an instance without a writer never enters the writing stages.

#![cfg(feature = "onnx-door")]
// every expect below runs against a model this file builds itself; a failure
// there IS the test failing, not a case to recover.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod door_support;

use std::fs;
use std::path::Path;

use door_support::{mlp_batch, two_layer_mlp_bytes};
use omega::door::{Door, NoSink, StageKind};
use omega::onnx_door::{DoorConfig, DoorStage, OnnxDoor, OnnxReader};
use proxima_primitives::pipe::Pipe;

const SOURCE_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
const DOOR_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/door");

const MLP_SOFTMAX: [f32; 4] = [0.599_888_4, 0.400_111_6, 0.434_749_25, 0.565_250_74];
const TOLERANCE: f32 = 1e-3;

fn within(actual: &[f32], expected: &[f32], tolerance: f32) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(left, right)| (left - right).abs() < tolerance)
}

fn rust_sources(directory: &Path) -> Vec<String> {
    fs::read_dir(directory)
        .expect("the source directory exists")
        .flat_map(|entry| {
            let path = entry.expect("a directory entry reads").path();
            match path.is_dir() {
                true => rust_sources(&path),
                false => read_rust_file(&path),
            }
        })
        .collect()
}

fn read_rust_file(path: &Path) -> Vec<String> {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("rs") => vec![fs::read_to_string(path).expect("a source file reads")],
        _ => Vec::new(),
    }
}

fn count_stage_enums(source: &str) -> usize {
    source.matches("enum DoorStage").count()
}

fn count_format_name(source: &str) -> usize {
    source.to_lowercase().matches("onnx").count()
}

#[test]
fn door_generic_stage_enum_is_defined_once() {
    let sources = rust_sources(Path::new(SOURCE_ROOT));

    let definitions: usize = sources.iter().map(|source| count_stage_enums(source)).sum();

    assert_eq!(definitions, 1);
}

#[test]
fn door_generic_stage_counter_control_counts_two_in_a_string_holding_two() {
    let doubled = "pub enum DoorStage<'a> { A }\npub enum DoorStage<'b> { B }";

    assert_eq!(count_stage_enums(doubled), 2);
}

#[test]
fn door_generic_sources_name_no_format_outside_readers() {
    let sources = rust_sources(Path::new(DOOR_ROOT));

    let mentions: usize = sources.iter().map(|source| count_format_name(source)).sum();

    assert_eq!(mentions, 0);
}

#[test]
fn door_generic_name_counter_control_counts_one_in_a_string_containing_the_name() {
    assert_eq!(count_format_name("pub struct OnnxReader;"), 1);
}

#[proxima::test]
async fn door_generic_onnx_instance_is_door_of_onnx_reader_and_no_sink() {
    let bytes = two_layer_mlp_bytes();
    let door: Door<OnnxReader, NoSink> =
        OnnxDoor::from_bytes(bytes, &DoorConfig::default()).expect("the default config resolves");

    let output = door.call(mlp_batch()).await.expect("the mlp runs on cpu");

    let (values, _) = output.get("y").expect("y is a declared graph output");
    assert!(within(values, &MLP_SOFTMAX, TOLERANCE), "{values:?}");
}

#[test]
fn door_generic_stage_kinds_list_eight_in_pipeline_order() {
    let names: Vec<&str> = StageKind::ALL.iter().map(|kind| kind.name()).collect();

    assert_eq!(
        names,
        [
            "parse", "validate", "lower", "emit", "package", "verify", "plan", "execute"
        ]
    );
    assert!(StageKind::ALL.windows(2).all(|pair| pair[0] < pair[1]));
}

#[test]
fn door_generic_onnx_instance_never_enters_emit_package_verify() {
    let bytes = two_layer_mlp_bytes();
    let door = OnnxDoor::from_bytes(bytes.clone(), &DoorConfig::default())
        .expect("the default config resolves");
    let batch = mlp_batch();
    let mut stage = DoorStage::start(&bytes).expect("the mlp parses");
    let mut log = vec![stage.kind()];

    for _ in 0..5 {
        stage = stage.advance(&door, &batch).expect("every stage succeeds");
        log.push(stage.kind());
    }

    let mut entered = log.clone();
    entered.sort();
    entered.dedup();
    assert_eq!(
        entered,
        [
            StageKind::Parse,
            StageKind::Validate,
            StageKind::Lower,
            StageKind::Plan,
            StageKind::Execute
        ]
    );
}
